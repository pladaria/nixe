//! Immutable capture products and cold compilation, independent of a renderer.
pub mod depth;
pub mod resources;
use crate::{
    shader_encoding::{self, Encode},
    shader_identity::ContentId,
};
use nixe_shader_recompiler::{
    Compiler, CompilerSettings, Environment, HostTranslateInfo, PreparedStage, Profile, Result,
    RuntimeInfo, Stage,
    artifact::{Bindings, ShaderInfo, StageArtifact},
    diagnostics::{CompileError, FailureCategory},
    environment::SnapshotEnvironment,
    passes::GraphicsRuntimeState,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Configuration {
    pub host: HostTranslateInfo,
    pub profile: Profile,
    pub settings: CompilerSettings,
}
impl Configuration {
    pub fn identity(&self) -> ContentId {
        let mut bytes = shader_encoding::bytes(&self.host);
        self.profile.encode(&mut bytes);
        self.settings.encode(&mut bytes);
        ContentId::of(b"nixe-recompiler-configuration", &[&bytes])
    }
}

#[derive(Clone, Debug)]
pub struct StageSource {
    pub environment: SnapshotEnvironment,
    pub info: ShaderInfo,
    pub binding_group: Option<u8>,
}

#[derive(Clone, Debug)]
pub struct ShaderInputs {
    pub identity: ContentId,
    pub stages: Box<[StageSource]>,
}
impl ShaderInputs {
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .stages
                .iter()
                .map(|stage| {
                    std::mem::size_of::<StageSource>()
                        + environment_heap_bytes(&stage.environment)
                        + info_heap_bytes(&stage.info)
                })
                .sum::<usize>()
    }
    pub fn new(stages: Box<[StageSource]>, configuration: &Configuration) -> Self {
        let mut bytes = Vec::new();
        (stages.len() as u32).encode(&mut bytes);
        for stage in &stages {
            shader_encoding::environment_specialization(&stage.environment, &mut bytes);
            stage.binding_group.encode(&mut bytes);
        }
        let identity = ContentId::of(
            b"nixe-recompiler-source-v2",
            &[&configuration.identity().0, &bytes],
        );
        Self { identity, stages }
    }

    /// A linked program is one job. No dependent work is queued from a worker.
    pub fn compile(
        &self,
        compiler: &mut Compiler,
        configuration: &Configuration,
        graphics: Option<&GraphicsRuntimeState>,
    ) -> Result<Vec<StageArtifact>> {
        let stages = self.prepare(compiler, configuration, graphics)?;
        let mut bindings = Bindings::default();
        stages
            .into_iter()
            .map(|stage| {
                compiler.emit_prepared(
                    stage,
                    &configuration.profile,
                    &configuration.settings,
                    &mut bindings,
                )
            })
            .collect()
    }

    pub fn prepare(
        &self,
        compiler: &mut Compiler,
        configuration: &Configuration,
        graphics: Option<&GraphicsRuntimeState>,
    ) -> Result<Vec<PreparedStage>> {
        compiler.reset()?;
        let mut ids = Vec::with_capacity(self.stages.len());
        let mut vertex_a = None;
        for source in &self.stages {
            let mut environment = source.environment.clone();
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
                Stage::Compute if self.stages.len() == 1 && graphics.is_none() => {
                    return Ok(vec![compiler.prepare_stage(id, RuntimeInfo::default())?]);
                }
                Stage::Compute => {
                    return Err(invalid("Compute capture cannot be linked with graphics"));
                }
                _ => (),
            }
            ids.push(id);
        }
        if vertex_a.is_some() {
            return Err(invalid("Vertex A requires a matching Vertex B"));
        }
        let graphics =
            graphics.ok_or_else(|| invalid("Graphics compilation requires runtime state"))?;
        compiler.prepare_graphics(&ids, graphics)
    }

    /// Includes every stage-local observation, resolved linkage and incoming
    /// allocation counter. A generated stage conservatively depends on the link.
    pub fn stage_key(
        &self,
        stage: &PreparedStage,
        configuration: &Configuration,
        bindings: &Bindings,
    ) -> ContentId {
        let mut bytes = shader_encoding::bytes(&stage.stage);
        let mut found = false;
        for source in &self.stages {
            let source_stage = source.environment.metadata().stage;
            if source_stage == stage.stage
                || (source_stage == Stage::VertexA && stage.stage == Stage::VertexB)
            {
                shader_encoding::environment_specialization(&source.environment, &mut bytes);
                source.binding_group.encode(&mut bytes);
                found = true;
            }
        }
        if !found {
            bytes.extend_from_slice(&self.identity.0);
        }
        stage.runtime.encode(&mut bytes);
        bindings.encode(&mut bytes);
        ContentId::of(
            b"nixe-recompiler-stage-v2",
            &[&configuration.identity().0, &bytes],
        )
    }
}

/// Conservative B-tree node overhead, including sparse edge nodes. Canonical
/// guest allocations are shared external resources and are charged elsewhere.
pub fn environment_heap_bytes(environment: &SnapshotEnvironment) -> usize {
    use std::{collections::BTreeMap, mem::size_of};
    fn map_bytes<K, V>(map: &BTreeMap<K, V>) -> usize {
        map.len() * (size_of::<(K, V)>() + 64)
    }
    let o = environment.observations();
    map_bytes(&o.instructions)
        + map_bytes(&o.cbuf_words)
        + map_bytes(&o.texture_handles)
        + map_bytes(&o.texture_types)
        + map_bytes(&o.texture_pixel_formats)
        + map_bytes(&o.texture_integer_formats)
        + map_bytes(&o.replacements)
}

pub fn info_heap_bytes(info: &ShaderInfo) -> usize {
    use nixe_shader_recompiler::artifact::*;
    use std::mem::size_of;
    info.constant_buffer_descriptors.capacity() * size_of::<ConstantBufferDescriptor>()
        + info.storage_buffers_descriptors.capacity() * size_of::<StorageBufferDescriptor>()
        + info.texture_buffer_descriptors.capacity() * size_of::<TextureBufferDescriptor>()
        + info.image_buffer_descriptors.capacity() * size_of::<ImageBufferDescriptor>()
        + info.texture_descriptors.capacity() * size_of::<TextureDescriptor>()
        + info.image_descriptors.capacity() * size_of::<ImageDescriptor>()
        + info.legacy_stores_mapping.len() * 72
}

fn invalid(message: &str) -> CompileError {
    CompileError::new(FailureCategory::InvalidArgument, message)
}

pub fn artifact_key(
    source: ContentId,
    configuration: &Configuration,
    graphics: Option<&GraphicsRuntimeState>,
) -> ContentId {
    let mut bytes = shader_encoding::bytes(&configuration.host);
    configuration.profile.encode(&mut bytes);
    configuration.settings.encode(&mut bytes);
    graphics.is_some().encode(&mut bytes);
    if let Some(graphics) = graphics {
        graphics.encode(&mut bytes);
    }
    ContentId::of(b"nixe-recompiler-artifact", &[&source.0, &bytes])
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
