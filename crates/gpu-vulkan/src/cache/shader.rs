use super::persistent::word;
use super::*;
use nixe_gpu::{
    ShaderStage,
    shader_identity::{ContentId, ShaderNamespace},
    shader_module as spirv,
};
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ArtifactKey(ContentId);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StageKey(ContentId);
use std::{sync::Arc, time::Instant};
mod stages;

#[derive(Clone, Debug)]
pub struct CompiledStage {
    pub stage: ShaderStage,
    pub module: spirv::Module,
    pub identity: ContentId,
    pub workgroup_size: Option<[u32; 3]>,
    pub shared_memory_bytes: u32,
    pub output_control_points: Option<u32>,
    binding_end: Option<nixe_shader_recompiler::artifact::Bindings>,
    pub guest: Option<Box<GuestStageMetadata>>,
}
impl CompiledStage {
    pub fn new(
        stage: ShaderStage,
        module: spirv::Module,
        workgroup_size: Option<[u32; 3]>,
        shared_memory_bytes: u32,
        output_control_points: Option<u32>,
    ) -> Self {
        let mut bytes: Vec<_> = module.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        word(&mut bytes, module.push_constant_bytes);
        for b in &module.bindings {
            for value in [
                b.resource.index() as u32,
                b.binding,
                b.count,
                b.sampler.map_or(u32::MAX, |s| s.index() as u32),
                b.kind as u32,
                u32::from(b.readable),
                u32::from(b.writable),
            ] {
                word(&mut bytes, value);
            }
        }
        let identity = ContentId::of(b"nixe-native-stage-v2", &[&bytes]);
        Self {
            stage,
            module,
            identity,
            workgroup_size,
            shared_memory_bytes,
            output_control_points,
            binding_end: None,
            guest: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct GuestStageMetadata {
    pub info: nixe_shader_recompiler::artifact::ShaderInfo,
    pub bindings: Box<[nixe_shader_recompiler::artifact::ResourceBinding]>,
    pub host_uniforms: Box<[nixe_shader_recompiler::artifact::HostUniformLayout]>,
    pub interface: nixe_shader_recompiler::artifact::StageInterface,
    pub execution: nixe_shader_recompiler::artifact::ExecutionMetadata,
}
impl CompiledStage {
    pub fn from_guest(
        artifact: nixe_shader_recompiler::StageArtifact,
    ) -> Result<Self, CompileError> {
        use nixe_shader_recompiler::{
            Stage,
            artifact::{HostUniformLayout as H, StageArtifact},
        };
        let identity = ContentId::of(
            b"nixe-guest-stage-v1",
            &[&nixe_gpu::shader_encoding::encode_artifacts(
                std::slice::from_ref(&artifact),
            )],
        );
        let bindings =
            nixe_gpu::shader_module::artifact_bindings(&artifact).map_err(CompileError::from)?;
        let mut push_constant_bytes = 0;
        let end = |offset: u32, size: u32| {
            offset
                .checked_add(size)
                .ok_or_else(|| CompileError::Failed("host uniform byte range overflow".into()))
        };
        for uniform in &artifact.host_uniforms {
            let bytes = match *uniform {
                H::RenderAreaPushConstants { offset, size } => end(offset, size)?,
                H::TessellationLevelsPushConstants { offset, .. } => end(offset, 24)?,
                H::RescalingPushConstants {
                    texture_mask_offset,
                    texture_mask_words,
                    image_mask_offset,
                    image_mask_words,
                    down_factor_offset,
                } => {
                    let mask = |offset, words: u32| {
                        end(
                            offset,
                            words.checked_mul(4).ok_or_else(|| {
                                CompileError::Failed("host uniform mask size overflow".into())
                            })?,
                        )
                    };
                    mask(texture_mask_offset, texture_mask_words)?
                        .max(mask(image_mask_offset, image_mask_words)?)
                        .max(
                            down_factor_offset
                                .map(|offset| end(offset, 4))
                                .transpose()?
                                .unwrap_or(0),
                        )
                }
                H::RescalingUniformConstant { .. } => {
                    return Err(CompileError::Failed(
                        "Vulkan requires push constants for shader host uniforms".into(),
                    ));
                }
            };
            push_constant_bytes = push_constant_bytes.max(bytes);
        }
        let StageArtifact {
            stage,
            words,
            info,
            bindings: origins,
            host_uniforms,
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
            identity,
            workgroup_size: (stage == Stage::Compute).then_some(execution.workgroup_size),
            shared_memory_bytes: execution.shared_memory_size,
            output_control_points: (stage == Stage::TessellationControl)
                .then_some(execution.output_vertices),
            binding_end: None,
            guest: Some(Box::new(GuestStageMetadata {
                info,
                bindings: origins,
                host_uniforms,
                interface,
                execution,
            })),
        })
    }
    fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>()
            + self.module.words.len() * 4
            + std::mem::size_of_val(self.module.bindings.as_ref());
        if let Some(metadata) = &self.guest {
            let info = &metadata.info;
            bytes += std::mem::size_of::<GuestStageMetadata>()
                + std::mem::size_of_val(metadata.bindings.as_ref())
                + std::mem::size_of_val(metadata.host_uniforms.as_ref())
                + nixe_gpu::shader_compilation::info_heap_bytes(info);
        }
        bytes
    }
    fn to_guest(&self) -> nixe_shader_recompiler::StageArtifact {
        use nixe_shader_recompiler::Stage;
        let metadata = self.guest.as_ref().expect("guest stage metadata");
        nixe_shader_recompiler::StageArtifact {
            stage: match self.stage {
                ShaderStage::Vertex => Stage::VertexB,
                ShaderStage::TessellationControl => Stage::TessellationControl,
                ShaderStage::TessellationEvaluation => Stage::TessellationEval,
                ShaderStage::Geometry => Stage::Geometry,
                ShaderStage::Fragment => Stage::Fragment,
                ShaderStage::Compute => Stage::Compute,
            },
            words: self.module.words.clone(),
            info: metadata.info.clone(),
            bindings: metadata.bindings.clone(),
            host_uniforms: metadata.host_uniforms.clone(),
            interface: metadata.interface.clone(),
            execution: metadata.execution.clone(),
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CompilationTimings {
    pub translation_ns: u64,
    pub spirv_ns: u64,
    pub disk_hit: bool,
}
#[derive(Debug)]
pub struct ShaderArtifact {
    pub identity: ContentId,
    pub stages: Box<[Arc<CompiledStage>]>,
    pub timings: CompilationTimings,
}

impl ShaderArtifact {
    pub fn new<S: Into<Arc<CompiledStage>>>(stages: Box<[S]>) -> Self {
        let stages: Box<_> = stages.into_iter().map(Into::into).collect();
        Self {
            identity: native_identity(&stages),
            stages,
            timings: Default::default(),
        }
    }
}
pub struct ShaderCompiler {
    namespace: ShaderNamespace,
    scratch: Arc<[std::sync::Mutex<Option<nixe_shader_recompiler::Compiler>>]>,
    guest_store: Option<Arc<persistent::Store>>,
    pool: Arc<CompilePool>,
    cache: CompilationCache<ArtifactKey, ShaderArtifact>,
    stages: Arc<stages::Stages>,
}
impl ShaderCompiler {
    pub(crate) fn bind_wakeup(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.pool.bind_wakeup(wake);
    }
    pub fn new(
        namespace: ShaderNamespace,
        pool: Arc<CompilePool>,
        config: &CacheConfiguration,
    ) -> Self {
        Self {
            namespace,
            scratch: (0..pool.workers())
                .map(|_| std::sync::Mutex::new(None))
                .collect(),
            guest_store: persistent::store(
                config.directory.as_deref(),
                "guest-recompiler-v1",
                config.persistent_bytes,
            )
            .map(Arc::new),
            pool,
            cache: CompilationCache::new(config.shader_entries, config.shader_bytes),
            stages: Arc::new(stages::Stages::new(config.shader_entries)),
        }
    }
    pub fn request_guest(
        &self,
        inputs: Arc<nixe_gpu::shader_compilation::ShaderInputs>,
        configuration: Arc<nixe_gpu::shader_compilation::Configuration>,
        runtime: Option<nixe_shader_recompiler::passes::GraphicsRuntimeState>,
    ) -> Result<Pending<ShaderArtifact>, CompileError> {
        self.request_guest_with_depth(inputs, configuration, runtime, None)
    }
    pub fn request_guest_with_depth(
        &self,
        inputs: Arc<nixe_gpu::shader_compilation::ShaderInputs>,
        configuration: Arc<nixe_gpu::shader_compilation::Configuration>,
        runtime: Option<nixe_shader_recompiler::passes::GraphicsRuntimeState>,
        depth: Option<nixe_gpu::FragmentDepthTransform>,
    ) -> Result<Pending<ShaderArtifact>, CompileError> {
        let depth_key = |depth: Option<nixe_gpu::FragmentDepthTransform>| {
            let mut bytes = vec![u8::from(depth.is_some())];
            if let Some(depth) = depth {
                bytes.extend(
                    depth
                        .coefficients()
                        .into_iter()
                        .flat_map(|v| v.to_bits().to_le_bytes()),
                );
            }
            bytes
        };
        let depth_bytes = depth_key(depth);
        let content = nixe_gpu::shader_compilation::artifact_key(
            inputs.identity,
            &configuration,
            runtime.as_ref(),
        );
        let key = ArtifactKey(ContentId::of(
            b"nixe-native-guest-artifact-v1",
            &[
                &self.namespace.guest,
                &self.namespace.isa.to_le_bytes(),
                &content.0,
                &depth_bytes,
            ],
        ));
        let store = self.guest_store.clone();
        let scratch = Arc::clone(&self.scratch);
        let stage_cache = Arc::clone(&self.stages);
        self.cache.request_sized(
            &self.pool,
            key,
            inputs.retained_bytes(),
            |artifact| {
                artifact
                    .stages
                    .iter()
                    .map(|stage| stage.retained_bytes())
                    .sum()
            },
            move |worker| {
                let cached = store
                    .as_ref()
                    .and_then(|store| store.read(key.0))
                    .and_then(|bytes| nixe_gpu::shader_encoding::decode_artifacts(&bytes));
                let disk_hit = cached.is_some();
                let mut timings = CompilationTimings {
                    disk_hit,
                    ..Default::default()
                };
                let stages = if let Some(artifacts) = cached {
                    artifacts
                        .into_iter()
                        .map(|artifact| CompiledStage::from_guest(artifact).map(Arc::new))
                        .collect::<Result<Box<_>, _>>()?
                } else {
                    let mut slot = scratch[worker].lock().map_err(|_| {
                        CompileError::Failed("shader compiler scratch poisoned".into())
                    })?;
                    if slot.is_none() {
                        *slot = Some(
                            nixe_shader_recompiler::Compiler::new().map_err(CompileError::from)?,
                        );
                    }
                    let compiler = slot.as_mut().unwrap();
                    let start = Instant::now();
                    let prepared = inputs
                        .prepare(compiler, &configuration, runtime.as_ref())
                        .map_err(CompileError::from)?;
                    timings.translation_ns = start.elapsed().as_nanos() as u64;
                    let mut bindings = nixe_shader_recompiler::artifact::Bindings::default();
                    let mut stages = Vec::with_capacity(prepared.len());
                    for stage in prepared {
                        let stage_depth = (stage.stage == nixe_shader_recompiler::Stage::Fragment)
                            .then_some(depth)
                            .flatten();
                        let source = inputs.stage_key(&stage, &configuration, &bindings);
                        let key = StageKey(ContentId::of(
                            b"nixe-native-guest-stage-v2",
                            &[&source.0, &depth_key(stage_depth)],
                        ));
                        let compiled = stage_cache.get(key, || {
                            let start = Instant::now();
                            if let Some(depth) = stage_depth {
                                nixe_gpu::shader_compilation::depth::specialize(
                                    compiler, &stage, depth,
                                )
                                .map_err(CompileError::from)?;
                            }
                            let artifact = compiler
                                .emit_prepared(
                                    stage,
                                    &configuration.profile,
                                    &configuration.settings,
                                    &mut bindings,
                                )
                                .map_err(CompileError::from)?;
                            timings.spirv_ns += start.elapsed().as_nanos() as u64;
                            let mut compiled = CompiledStage::from_guest(artifact)?;
                            compiled.binding_end = Some(bindings.clone());
                            Ok(Arc::new(compiled))
                        })?;
                        bindings = compiled
                            .binding_end
                            .clone()
                            .expect("emitted allocation counters");
                        stages.push(compiled);
                    }
                    if let Some(store) = &store
                        && let Err(error) = store.write(
                            key.0,
                            &nixe_gpu::shader_encoding::encode_artifacts(
                                &stages
                                    .iter()
                                    .map(|stage| stage.to_guest())
                                    .collect::<Vec<_>>(),
                            ),
                        )
                    {
                        log::warn!("guest shader cache write: {error}");
                    }
                    stages.into_boxed_slice()
                };
                Ok(ShaderArtifact {
                    identity: native_identity(&stages),
                    stages,
                    timings,
                })
            },
        )
    }
    /// Compose already-ready frontend values here. Do not submit and wait for
    /// another job from a worker callback on this same bounded pool.
    pub fn usage(&self) -> (usize, usize) {
        self.cache.usage()
    }
    pub(crate) fn cache_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.cache.counters()
    }
    pub(crate) fn stage_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.stages.counters.clone()
    }
}
fn native_identity(stages: &[Arc<CompiledStage>]) -> ContentId {
    let identities: Vec<_> = stages.iter().map(|s| s.identity.0.as_slice()).collect();
    ContentId::of(b"nixe-native-program-v1", &identities)
}
