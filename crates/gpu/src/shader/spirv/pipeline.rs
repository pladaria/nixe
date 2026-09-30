//! Compile the complete patch chain and its live descriptor ABI on cache misses.
//! No resource resolution, Vulkan types or draw-time interface scans belong here.
use super::*;
use crate::{PipelineStages, TessellationControl};

/// Structural specialization only: default levels, resources and draw arguments
/// are deliberately absent. Host float guarantees are device-wide cache inputs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpirvTessellationOptions {
    pub input_control_points: u8,
    pub mode: TessellationMode,
    pub float32: SpirvFloat32Capabilities,
    pub float64: SpirvFloat64Capabilities,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpirvPipelineBinding {
    pub resource: ShaderResourceAccess,
    pub stages: PipelineStages,
}

/// Linked stage artifacts and set-zero descriptor ABI. Backends retain this on
/// the cached native pipeline, rather than reflecting SPIR-V or scanning guest
/// resource declarations on each draw. Host limits/layout/lifetimes are still
/// the consuming backend's responsibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpirvTessellationShaders {
    modules: [SpirvShaderModule; 4],
    bindings: Box<[SpirvPipelineBinding]>,
    input_control_points: u8,
    output_control_points: u32,
    required_default_levels: Option<u8>,
}

impl SpirvTessellationShaders {
    /// Fixed order: vertex, control, evaluation, fragment.
    pub fn modules(&self) -> &[SpirvShaderModule; 4] {
        &self.modules
    }

    pub fn bindings(&self) -> &[SpirvPipelineBinding] {
        &self.bindings
    }

    pub const fn output_control_points(&self) -> u32 {
        self.output_control_points
    }

    /// Patch assembly size for the matching native graphics pipeline.
    pub const fn input_control_points(&self) -> u8 {
        self.input_control_points
    }

    /// Default-control pipelines reserve [0, 24) for control-stage push constants.
    pub const fn push_constant_bytes(&self) -> u32 {
        if self.required_default_levels.is_some() {
            SpirvDefaultControlShader::PARAMETER_BYTES
        } else {
            0
        }
    }

    /// Constant-time validation/extraction at the consuming draw. Level values
    /// never invalidate this compiled shader chain.
    pub fn parameters(&self, control: TessellationControl) -> Result<Option<[u32; 6]>> {
        match self.required_default_levels {
            Some(required) => default_control::parameters(required, control).map(Some),
            None if control == TessellationControl::Shader => Ok(None),
            None => Err(SpirvShaderError::Options(
                "guest control shader pipeline cannot consume default levels",
            )),
        }
    }
}

/// Compile a native tessellation chain, including its semantic adapter when the
/// guest has no control shader. The input patch size is not the control shader's
/// output size: TES arrays use the latter. Original position values pass between
/// stages unchanged; viewport/raster conventions remain a backend obligation.
pub fn lower_tessellation_shaders_to_spirv(
    vertex: &VerifiedShaderIr,
    control: Option<&VerifiedShaderIr>,
    evaluation: &VerifiedShaderIr,
    fragment: &VerifiedShaderIr,
    options: SpirvTessellationOptions,
) -> Result<SpirvTessellationShaders> {
    for (ir, stage) in [
        (Some(vertex), ShaderStage::Vertex),
        (control, ShaderStage::TessellationControl),
        (Some(evaluation), ShaderStage::TessellationEvaluation),
        (Some(fragment), ShaderStage::Fragment),
    ] {
        if ir.is_some_and(|ir| ir.ir().stage() != stage) {
            return Err(SpirvShaderError::Options(
                "incorrect tessellation stage chain",
            ));
        }
    }
    if options.input_control_points == 0 {
        return Err(SpirvShaderError::Options(
            "input patch size must be nonzero",
        ));
    }
    validate_shader_stage_link(evaluation.ir(), fragment.ir())
        .map_err(SpirvShaderError::StageInterface)?;
    // TES is the final geometry-producing stage in this chain (no transform
    // feedback). Only fragment-consumed user components are observable; retain
    // builtins such as Position/PointSize for fixed-function rasterization.
    // Drop dead stores before backward liveness, so their arithmetic/resources
    // disappear too. This is cache-miss compilation, never draw-time filtering.
    // https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-iointerfaces
    let linked_evaluation = evaluation.prune_raster_outputs(
        fragment
            .ir()
            .inputs()
            .iter()
            .map(|e| (e.location(), e.component())),
    );
    let evaluation = &linked_evaluation;
    let base = SpirvShaderOptions {
        input_control_points: 0,
        tessellation_mode: None,
        float32: options.float32,
        float64: options.float64,
    };
    let (control, required_default_levels, output_control_points) = if let Some(control) = control {
        for (producer, consumer) in [(vertex, control), (control, evaluation)] {
            validate_shader_stage_link(producer.ir(), consumer.ir())
                .map_err(SpirvShaderError::StageInterface)?;
        }
        let points = control.ir().tessellation_control_points().unwrap();
        (
            lower_shader_ir_to_spirv(
                control,
                SpirvShaderOptions {
                    input_control_points: u32::from(options.input_control_points),
                    ..base
                },
            )?,
            None,
            points,
        )
    } else {
        let adapter = lower_default_tessellation_control_to_spirv(
            vertex,
            evaluation,
            SpirvDefaultControlOptions {
                input_control_points: options.input_control_points,
                domain: options.mode.domain,
                push_constant_offset: 0,
            },
        )?;
        let (module, required) = adapter.into_parts();
        (
            module,
            Some(required),
            u32::from(options.input_control_points),
        )
    };
    let modules = [
        lower_shader_ir_to_spirv(vertex, base)?,
        control,
        lower_shader_ir_to_spirv(
            evaluation,
            SpirvShaderOptions {
                input_control_points: output_control_points,
                tessellation_mode: Some(options.mode),
                ..base
            },
        )?,
        lower_shader_ir_to_spirv(fragment, base)?,
    ];
    let mut bindings: [Option<SpirvPipelineBinding>; 256] = [None; 256];
    for (module, stages) in modules.iter().zip([
        PipelineStages::VERTEX_SHADER,
        PipelineStages::TESSELLATION_CONTROL_SHADER,
        PipelineStages::TESSELLATION_EVALUATION_SHADER,
        PipelineStages::FRAGMENT_SHADER,
    ]) {
        for &resource in module.bindings() {
            let entry = &mut bindings[usize::from(resource.binding())];
            if let Some(previous) = entry {
                if previous.resource != resource {
                    return Err(SpirvShaderError::BindingConflict {
                        binding: resource.binding(),
                    });
                }
                previous.stages = previous.stages.union(stages);
            } else {
                *entry = Some(SpirvPipelineBinding { resource, stages });
            }
        }
    }
    Ok(SpirvTessellationShaders {
        modules,
        bindings: bindings.into_iter().flatten().collect(),
        input_control_points: options.input_control_points,
        output_control_points,
        required_default_levels,
    })
}
