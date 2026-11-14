//! Pipeline-owned SPIR-V and reflection. A guest pipeline miss translates once;
//! no program artifact, stage reuse, or persistent SPIR-V cache is consulted.
use super::*;
use nixe_gpu::{ShaderStage, shader_module as spirv};
use std::sync::Arc;
#[derive(Clone, Debug)]
pub struct CompiledStage {
    pub stage: ShaderStage,
    pub module: spirv::Module,
    pub workgroup_size: Option<[u32; 3]>,
    pub shared_memory_bytes: u32,
    pub output_control_points: Option<u32>,
    pub guest: Option<Arc<GuestStageMetadata>>,
}
impl CompiledStage {
    pub fn new(
        stage: ShaderStage,
        module: spirv::Module,
        workgroup_size: Option<[u32; 3]>,
        shared_memory_bytes: u32,
        output_control_points: Option<u32>,
    ) -> Self {
        Self {
            stage,
            module,
            workgroup_size,
            shared_memory_bytes,
            output_control_points,
            guest: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct GuestStageMetadata {
    pub info: nixe_shader_recompiler::artifact::ShaderInfo,
    pub bindings: Box<[nixe_shader_recompiler::artifact::ResourceBinding]>,
    pub interface: nixe_shader_recompiler::artifact::StageInterface,
    pub execution: nixe_shader_recompiler::artifact::ExecutionMetadata,
}
impl CompiledStage {
    pub fn from_guest(
        artifact: nixe_shader_recompiler::StageArtifact,
    ) -> Result<Self, CompileError> {
        use nixe_shader_recompiler::{Stage, artifact::StageArtifact};
        let bindings =
            nixe_gpu::shader_module::artifact_bindings(&artifact).map_err(CompileError::from)?;
        let StageArtifact {
            stage,
            words,
            info,
            bindings: origins,
            push_constant_bytes,
            interface,
            execution,
        } = artifact;
        Ok(Self {
            stage: nixe_gpu::shader_compilation::neutral_stage(stage),
            module: nixe_gpu::shader_module::Module {
                words,
                bindings,
                push_constant_bytes,
            },
            workgroup_size: (stage == Stage::Compute).then_some(execution.workgroup_size),
            shared_memory_bytes: execution.shared_memory_size,
            output_control_points: (stage == Stage::TessellationControl)
                .then_some(execution.output_vertices),
            guest: Some(Arc::new(GuestStageMetadata {
                info,
                bindings: origins,
                interface,
                execution,
            })),
        })
    }
}
#[derive(Debug)]
pub struct ShaderArtifact {
    pub stages: Box<[Arc<CompiledStage>]>,
}
impl ShaderArtifact {
    pub fn new<S: Into<Arc<CompiledStage>>>(stages: Box<[S]>) -> Self {
        Self {
            stages: stages.into_iter().map(Into::into).collect(),
        }
    }
}
/// Main ShaderPools during gameplay; a fresh compiler belongs to each disk
/// loading job. Every pipeline emits its own stages, as CreateGraphicsPipeline.
pub(crate) fn compile_guest(
    compiler: &mut nixe_shader_recompiler::Compiler,
    programs: &[nixe_shader_recompiler::ProgramId],
    configuration: &nixe_gpu::shader_compilation::Configuration,
    runtime: Option<&nixe_shader_recompiler::passes::GraphicsRuntimeState>,
) -> Result<Arc<ShaderArtifact>, CompileError> {
    let prepared = if let Some(runtime) = runtime {
        compiler.prepare_graphics(programs, runtime, &configuration.profile)
    } else if programs.len() == 1 {
        compiler
            .prepare_stage(programs[0], nixe_shader_recompiler::RuntimeInfo::default())
            .map(|stage| vec![stage])
    } else {
        return Err(CompileError::Failed(
            "compute pipeline requires exactly one program".into(),
        ));
    }
    .map_err(CompileError::from)?;
    let mut bindings = nixe_shader_recompiler::artifact::Bindings::default();
    let mut stages = Vec::with_capacity(prepared.len());
    for stage in prepared {
        let artifact = compiler
            .emit_prepared(
                stage,
                &configuration.profile,
                &configuration.settings,
                &mut bindings,
            )
            .map_err(CompileError::from)?;
        stages.push(Arc::new(CompiledStage::from_guest(artifact)?));
    }
    Ok(Arc::new(ShaderArtifact {
        stages: stages.into_boxed_slice(),
    }))
}
