//! Owned guest environments and replay into the renderer's ShaderPools.
use nixe_shader_recompiler::{
    Compiler, CompilerSettings, Environment, HostTranslateInfo, Profile, ProgramId, Result, Stage,
    diagnostics::{CompileError, FailureCategory},
    environment::SnapshotEnvironment,
};
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Configuration {
    pub host: HostTranslateInfo,
    pub profile: Profile,
    pub settings: CompilerSettings,
}
#[derive(Clone, Debug)]
pub struct ShaderInputs {
    pub stages: Box<[SnapshotEnvironment]>,
    pub serializable: bool,
}
impl ShaderInputs {
    /// Disk replay translates once, just as a live recording environment does.
    pub fn translate(
        self,
        compiler: &mut Compiler,
        configuration: &Configuration,
    ) -> Result<Vec<ProgramId>> {
        compiler.reset()?;
        let count = self.stages.len();
        let mut ids = Vec::with_capacity(count);
        let mut vertex_a = None;
        for mut environment in self.stages {
            let mut id = compiler.translate(
                &mut environment,
                &configuration.host,
                &configuration.settings,
            )?;
            match environment.metadata().stage {
                Stage::VertexA => {
                    vertex_a = Some(id);
                    continue;
                }
                Stage::VertexB => {
                    if let Some(a) = vertex_a.take() {
                        id = compiler.merge_dual_vertex(
                            a,
                            id,
                            &environment,
                            &configuration.settings,
                        )?;
                    }
                }
                Stage::Compute if count != 1 => {
                    return Err(invalid("Compute capture cannot be linked with graphics"));
                }
                _ => (),
            }
            ids.push(id);
        }
        if vertex_a.is_some() {
            return Err(invalid("Vertex A requires a matching Vertex B"));
        }
        Ok(ids)
    }
}
fn invalid(message: &str) -> CompileError {
    CompileError::new(FailureCategory::InvalidArgument, message)
}

pub fn neutral_stage(stage: Stage) -> crate::ShaderStage {
    match stage {
        Stage::VertexA | Stage::VertexB => crate::ShaderStage::Vertex,
        Stage::TessellationControl => crate::ShaderStage::TessellationControl,
        Stage::TessellationEval => crate::ShaderStage::TessellationEvaluation,
        Stage::Geometry => crate::ShaderStage::Geometry,
        Stage::Fragment => crate::ShaderStage::Fragment,
        Stage::Compute => crate::ShaderStage::Compute,
    }
}
