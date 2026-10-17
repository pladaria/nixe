//! Device-independent native shader emission. No guest ISA or Vulkan ownership.
//!
//! Registers become scalar SSA values, never a GPU register-file interpreter.
//! Predicated side effects use structured selections and definitions merge with
//! OpPhi. Unsupported guest branches stop until CFG structurization is available.
//! https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

use std::collections::{BTreeMap, HashMap};

use rspirv::{binary::Assemble, dr::Builder, spirv as spv};

use super::*;
use crate::{
    TessellationDomain, TessellationMode, TessellationOutput, TessellationSpacing,
    TessellationWinding,
};

mod default_control;
pub(crate) use default_control::parameters as default_control_parameters;
mod pipeline;
pub use pipeline::{
    SpirvPipelineBinding, SpirvTessellationOptions, SpirvTessellationShaders,
    lower_raster_shaders_to_spirv, lower_tessellation_shaders_to_spirv,
};
pub(crate) mod math;
pub(crate) use math::Math;
mod comparison;
mod float;
mod half;
mod half_fma;
mod integer_conversion;
mod minmax;
mod multiply_rz;
mod preserve;
mod rounding;
mod sfu;
pub use default_control::{
    SpirvDefaultControlOptions, SpirvDefaultControlShader,
    lower_default_tessellation_control_to_spirv,
};
mod interface;
pub(crate) use interface::InterfaceLayout;
mod operations;
mod resources;
#[cfg(test)]
mod tests;

/// Enabled host numerical guarantees, not guesses based on GPU vendor. Native
/// device creation must query/enable these before supplying them to the emitter.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SpirvFloat32Capabilities {
    pub denorm_preserve: bool,
    pub denorm_flush_to_zero: bool,
    pub rounding_mode_rtz: bool,
    /// Full width independence is needed when selecting a different mode from
    /// the preserving FP16 or RTE FP64 repair environment.
    pub independent_denormals: bool,
    pub independent_rounding: bool,
    pub rounding_mode_rte: bool,
    pub signed_zero_inf_nan_preserve: bool,
    /// VK_KHR_shader_fma with shaderFmaFloat32 enabled. GLSL.std.450 Fma alone
    /// does not guarantee a fused, correctly rounded result.
    pub fused_multiply_add: bool,
}

/// Optional wide arithmetic for exact repair of float32 underflow. No float64
/// subnormal support is required: all repair intermediates are normal or zero.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SpirvFloat64Capabilities {
    pub enabled: bool,
    pub rounding_mode_rte: bool,
    pub signed_zero_inf_nan_preserve: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpirvShaderOptions {
    /// Convert only the final pre-raster position, after guest output reads.
    pub depth_clip_negative_one_to_one: bool,
    pub vertex_position_transform: Option<crate::VertexPositionTransform>,
    /// Incoming patch cardinality: draw patch size for TCS, TCS output size for TES.
    /// Zero for stages without arrayed patch inputs.
    pub input_control_points: u32,
    /// Required only for evaluation shaders. Part of the native shader cache key.
    pub tessellation_mode: Option<TessellationMode>,
    /// Enabled Float16 with independently legal DenormPreserve16,
    /// RoundingModeRTE16 and SignedZeroInfNanPreserve16 execution modes.
    pub native_half_conversions: bool,
    /// Enabled shaderFmaFloat16, in addition to the conversion guarantees.
    pub native_half_fma: bool,
    pub float32: SpirvFloat32Capabilities,
    pub float64: SpirvFloat64Capabilities,
}

/// SPIR-V 1.3 module targeting Vulkan 1.1 plus declared extensions. Native
/// integration must still link interfaces and apply its final position convention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpirvShaderModule {
    words: Box<[u32]>,
    bindings: Box<[ShaderResourceAccess]>,
}

impl SpirvShaderModule {
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Set-zero bindings actually emitted after dead-code elimination, sorted
    /// by binding number. Do not build native layouts from unused IR declarations.
    pub fn bindings(&self) -> &[ShaderResourceAccess] {
        &self.bindings
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpirvShaderError {
    Stage(ShaderStage),
    Options(&'static str),
    Interface(ShaderIoLocation),
    StageInterface(ShaderStageInterfaceError),
    BindingConflict {
        binding: u8,
    },
    DefaultTessellationLevels {
        required: u8,
        defined: u8,
    },
    Instruction {
        source: ShaderSourceLocation,
        reason: &'static str,
    },
    Construction(String),
}

impl Display for SpirvShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SpirvShaderError {}
impl From<rspirv::dr::Error> for SpirvShaderError {
    fn from(error: rspirv::dr::Error) -> Self {
        Self::Construction(error.to_string())
    }
}
type Result<T> = std::result::Result<T, SpirvShaderError>;

/// Emits the supported native graphics subset. No device-side validation,
/// subprocess, or per-draw work is performed here; call on a shader-cache miss.
pub fn lower_shader_ir_to_spirv(
    shader: &VerifiedShaderIr,
    options: SpirvShaderOptions,
) -> Result<SpirvShaderModule> {
    let ir = shader.ir();
    if !matches!(
        ir.stage,
        ShaderStage::Vertex
            | ShaderStage::TessellationControl
            | ShaderStage::TessellationEvaluation
            | ShaderStage::Fragment
    ) {
        return Err(SpirvShaderError::Stage(ir.stage));
    }
    let patch_stage = matches!(
        ir.stage,
        ShaderStage::TessellationControl | ShaderStage::TessellationEvaluation
    );
    if patch_stage == (options.input_control_points == 0) {
        return Err(SpirvShaderError::Options(
            "patch stages require a nonzero input cardinality; other stages require zero",
        ));
    }
    if (ir.stage == ShaderStage::TessellationEvaluation) != options.tessellation_mode.is_some() {
        return Err(SpirvShaderError::Options(
            "only evaluation shaders require a tessellation mode",
        ));
    }
    // Resources are consumed by operations below, so unused declarations do not
    // create descriptors or force unnecessary backend features.
    let live = liveness::live_instructions(ir)
        .map_err(|(source, reason)| SpirvShaderError::Instruction { source, reason })?;
    let mut emitter = Emitter::new(options);
    emitter.emit_interfaces(ir)?;
    let bindings = emitter.emit_resources(ir, &live)?;
    let void = emitter.math.b.type_void();
    let function_type = emitter.math.b.type_function(void, []);
    let entry =
        emitter
            .math
            .b
            .begin_function(void, None, spv::FunctionControl::NONE, function_type)?;
    emitter.math.entry = entry;
    emitter.configure_stage(ir)?;
    emitter.math.block = emitter.math.b.begin_block(None)?;
    for (instruction, live) in ir.instructions().iter().zip(live) {
        if !live {
            continue;
        }
        emitter.math.source = instruction.source;
        // Liveness has already rejected conditional exits and excluded Never.
        if matches!(instruction.operation, ShaderOperation::Exit) {
            break;
        }
        if instruction.predicate == ShaderPredicate::Always {
            emitter.operation(&instruction.operation)?;
        } else {
            emitter.predicated(instruction)?;
        }
    }
    emitter.io.final_position(&mut emitter.math)?;
    emitter.math.b.ret()?;
    emitter.math.b.end_function()?;
    Ok(SpirvShaderModule {
        words: emitter.math.b.module().assemble().into_boxed_slice(),
        bindings,
    })
}

struct Emitter {
    math: Math,
    registers: Vec<Option<u32>>,
    predicates: [Option<u32>; 8],
    io: InterfaceLayout,
    readonly_buffers: [Option<u32>; 256],
    buffer_word_pointer: Option<u32>,
}

impl Emitter {
    fn new(options: SpirvShaderOptions) -> Self {
        Self {
            math: Math::new(options),
            registers: Vec::new(),
            predicates: [None; 8],
            io: InterfaceLayout::default(),
            readonly_buffers: [None; 256],
            buffer_word_pointer: None,
        }
    }

    fn emit_interfaces(&mut self, ir: &ShaderIr) -> Result<()> {
        self.io.emit_interfaces(
            &mut self.math,
            ir.stage,
            ir.inputs(),
            ir.outputs(),
            ir.tessellation_control_points,
        )
    }
    fn load_interface(
        &mut self,
        input: bool,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
    ) -> Result<u32> {
        self.io
            .load_interface(&mut self.math, input, location, component, vertex)
    }
    fn store_interface(
        &mut self,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
        value: u32,
    ) -> Result<()> {
        self.io
            .store_interface(&mut self.math, location, component, vertex, value)
    }
    fn configure_stage(&mut self, ir: &ShaderIr) -> Result<()> {
        use spv::ExecutionMode as M;
        let model = match ir.stage {
            ShaderStage::Vertex => spv::ExecutionModel::Vertex,
            ShaderStage::Fragment => {
                self.math
                    .b
                    .execution_mode(self.math.entry, M::OriginUpperLeft, []);
                if ir
                    .outputs
                    .iter()
                    .any(|e| e.location == ShaderIoLocation::FragmentDepth)
                {
                    self.math
                        .b
                        .execution_mode(self.math.entry, M::DepthReplacing, []);
                }
                spv::ExecutionModel::Fragment
            }
            ShaderStage::TessellationControl => {
                self.math.b.capability(spv::Capability::Tessellation);
                self.math.b.execution_mode(
                    self.math.entry,
                    M::OutputVertices,
                    [ir.tessellation_control_points.unwrap()],
                );
                spv::ExecutionModel::TessellationControl
            }
            ShaderStage::TessellationEvaluation => {
                self.math.b.capability(spv::Capability::Tessellation);
                let mode = self.math.options.tessellation_mode.unwrap();
                let domain = match mode.domain {
                    TessellationDomain::Triangles => M::Triangles,
                    TessellationDomain::Quads => M::Quads,
                    TessellationDomain::Isolines => M::Isolines,
                };
                let spacing = match mode.spacing {
                    TessellationSpacing::Equal => M::SpacingEqual,
                    TessellationSpacing::FractionalEven => M::SpacingFractionalEven,
                    TessellationSpacing::FractionalOdd => M::SpacingFractionalOdd,
                };
                self.math.b.execution_mode(self.math.entry, domain, []);
                self.math.b.execution_mode(self.math.entry, spacing, []);
                match (mode.domain, mode.output) {
                    (_, TessellationOutput::Points) => {
                        self.math
                            .b
                            .execution_mode(self.math.entry, M::PointMode, [])
                    }
                    (TessellationDomain::Isolines, TessellationOutput::Lines) => {}
                    (
                        TessellationDomain::Triangles | TessellationDomain::Quads,
                        TessellationOutput::Triangles(winding),
                    ) => {
                        self.math.b.execution_mode(
                            self.math.entry,
                            match winding {
                                TessellationWinding::Clockwise => M::VertexOrderCw,
                                TessellationWinding::CounterClockwise => M::VertexOrderCcw,
                            },
                            [],
                        );
                    }
                    _ => {
                        return Err(SpirvShaderError::Options(
                            "domain and output topology are incompatible",
                        ));
                    }
                }
                spv::ExecutionModel::TessellationEvaluation
            }
            _ => unreachable!("stage validated before emission"),
        };
        self.math
            .b
            .entry_point(model, self.math.entry, "main", &self.io.variables);
        Ok(())
    }

    fn read(&self, register: ShaderRegister) -> Result<u32> {
        self.registers
            .get(usize::from(register.index()))
            .copied()
            .flatten()
            .ok_or_else(|| self.math.unsupported("missing SSA register definition"))
    }
    fn write(&mut self, register: ShaderRegister, id: u32) {
        let index = usize::from(register.index());
        if self.registers.len() <= index {
            self.registers.resize(index + 1, None);
        }
        self.registers[index] = Some(id);
    }
    fn predicate(&mut self, predicate: ShaderPredicate) -> Result<u32> {
        Ok(match predicate {
            ShaderPredicate::Always => self.math.true_value,
            ShaderPredicate::Never => self.math.false_value,
            ShaderPredicate::Register { register, inverted } => {
                let value = self.predicates[usize::from(register)]
                    .ok_or_else(|| self.math.unsupported("missing SSA predicate definition"))?;
                if inverted {
                    self.math.b.logical_not(self.math.boolean, None, value)?
                } else {
                    value
                }
            }
        })
    }

    fn predicated(&mut self, instruction: &ShaderInstruction) -> Result<()> {
        let condition = self.predicate(instruction.predicate)?;
        let header = self.math.block;
        let body = self.math.b.id();
        let merge = self.math.b.id();
        let mut definitions = Vec::new();
        instruction.operation.visit_destination_registers(|r| {
            if !definitions.iter().any(|(existing, _)| *existing == r) {
                definitions.push((
                    r,
                    self.registers
                        .get(usize::from(r.index()))
                        .copied()
                        .flatten(),
                ));
            }
        });
        let old_predicates = self.predicates;
        self.math
            .b
            .selection_merge(merge, spv::SelectionControl::NONE)?;
        self.math.b.branch_conditional(condition, body, merge, [])?;
        self.math.block = self.math.b.begin_block(Some(body))?;
        self.operation(&instruction.operation)?;
        let body_end = self.math.block;
        self.math.b.branch(merge)?;
        self.math.block = self.math.b.begin_block(Some(merge))?;
        for (register, previous) in definitions {
            let new = self.read(register)?;
            let old = previous.unwrap_or(self.math.undefined_uint);
            let merged = self
                .math
                .b
                .phi(self.math.uint, None, [(new, body_end), (old, header)])?;
            self.write(register, merged);
        }
        for (index, old) in old_predicates.into_iter().enumerate() {
            if self.predicates[index] != old {
                let previous = old.unwrap_or(self.math.undefined_bool);
                self.predicates[index] = Some(self.math.b.phi(
                    self.math.boolean,
                    None,
                    [
                        (self.predicates[index].unwrap(), body_end),
                        (previous, header),
                    ],
                )?);
            }
        }
        Ok(())
    }
}
