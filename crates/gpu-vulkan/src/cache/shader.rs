use super::persistent::{Reader, word};
use super::*;
use nixe_gpu::{
    ShaderResourceKind, ShaderStage,
    ssa::{
        Function, ResourceId,
        artifact::{ArtifactKey, ContentId, ShaderNamespace, specialization},
        spirv,
    },
};
use std::{sync::Arc, time::Instant};

#[derive(Clone, Debug)]
pub struct CompiledStage {
    pub stage: ShaderStage,
    pub module: spirv::Module,
    pub identity: ContentId,
    pub workgroup_size: Option<[u32; 3]>,
    pub shared_memory_bytes: u32,
    pub output_control_points: Option<u32>,
}
impl CompiledStage {
    pub fn new(
        stage: ShaderStage,
        module: spirv::Module,
        workgroup_size: Option<[u32; 3]>,
        shared_memory_bytes: u32,
        output_control_points: Option<u32>,
    ) -> Self {
        let bytes: Vec<_> = module.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let identity = ContentId::of(b"nixe-native-stage-v1", &[&bytes]);
        Self {
            stage,
            module,
            identity,
            workgroup_size,
            shared_memory_bytes,
            output_control_points,
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
    pub stages: Box<[CompiledStage]>,
    pub timings: CompilationTimings,
}

impl ShaderArtifact {
    pub fn new(stages: Box<[CompiledStage]>) -> Self {
        Self {
            identity: native_identity(&stages),
            stages,
            timings: Default::default(),
        }
    }
}
pub struct ShaderCompiler {
    namespace: ShaderNamespace,
    pool: Arc<CompilePool>,
    cache: CompilationCache<ArtifactKey, ShaderArtifact>,
    store: Option<Arc<persistent::Store>>,
}
impl ShaderCompiler {
    pub fn new(
        namespace: ShaderNamespace,
        pool: Arc<CompilePool>,
        config: &CacheConfiguration,
    ) -> Self {
        Self {
            namespace,
            pool,
            cache: CompilationCache::new(config.shader_entries, config.shader_bytes),
            store: persistent::store(
                config.directory.as_deref(),
                "spirv-v1",
                config.persistent_bytes,
            )
            .map(Arc::new),
        }
    }
    /// Compose already-ready frontend values here. Do not submit and wait for
    /// another job from a worker callback on this same bounded pool.
    pub fn request<T: Send + Sync + 'static>(
        &self,
        source: ContentId,
        options: Box<[spirv::Options]>,
        compile: impl FnOnce() -> Result<T, CompileError> + Send + 'static,
        functions: for<'a> fn(&'a T) -> Vec<&'a Function>,
    ) -> Result<Pending<ShaderArtifact>, CompileError> {
        let key = ArtifactKey::new(self.namespace, source, specialization(&options));
        let store = self.store.clone();
        self.cache.request_sized(
            &self.pool,
            key,
            options.len() * 4096,
            |artifact| {
                artifact
                    .stages
                    .iter()
                    .map(|s| {
                        s.module.words.len() * 4
                            + s.module.bindings.len() * std::mem::size_of::<spirv::Binding>()
                    })
                    .sum()
            },
            move |_| {
                if let Some(data) = store.as_ref().and_then(|s| s.read(key.0))
                    && let Some(stages) = decode(&data)
                {
                    return Ok(ShaderArtifact {
                        identity: native_identity(&stages),
                        stages,
                        timings: CompilationTimings {
                            disk_hit: true,
                            ..Default::default()
                        },
                    });
                }
                let start = Instant::now();
                let code = compile()?;
                let functions = functions(&code);
                let translation_ns = start.elapsed().as_nanos() as u64;
                if functions.is_empty() || functions.len() != options.len() {
                    return Err(CompileError::Failed(
                        "shader stages and emission options differ".into(),
                    ));
                }
                let stages: Vec<_> = functions
                    .iter()
                    .zip(options.iter())
                    .map(|(&f, &o)| (f, o))
                    .collect();
                let start = Instant::now();
                let modules = spirv::emit_pipeline(&stages).map_err(CompileError::from)?;
                let spirv_ns = start.elapsed().as_nanos() as u64;
                let stages: Box<_> = functions
                    .iter()
                    .zip(modules)
                    .zip(options.iter())
                    .map(|((f, module), option)| {
                        CompiledStage::new(
                            f.stage,
                            module,
                            f.workgroup_size,
                            option.shared_memory_bytes,
                            f.tessellation_control_points,
                        )
                    })
                    .collect();
                let data = encode(&stages);
                if let Some(store) = store
                    && let Err(e) = store.write(key.0, &data)
                {
                    log::warn!("native shader cache write: {e}");
                }
                Ok(ShaderArtifact {
                    identity: native_identity(&stages),
                    stages,
                    timings: CompilationTimings {
                        translation_ns,
                        spirv_ns,
                        disk_hit: false,
                    },
                })
            },
        )
    }
    pub fn usage(&self) -> (usize, usize) {
        self.cache.usage()
    }
}
fn native_identity(stages: &[CompiledStage]) -> ContentId {
    let identities: Vec<_> = stages.iter().map(|s| s.identity.0.as_slice()).collect();
    ContentId::of(b"nixe-native-program-v1", &identities)
}
fn stage(value: ShaderStage) -> u32 {
    match value {
        ShaderStage::Vertex => 0,
        ShaderStage::TessellationControl => 1,
        ShaderStage::TessellationEvaluation => 2,
        ShaderStage::Geometry => 3,
        ShaderStage::Fragment => 4,
        ShaderStage::Compute => 5,
    }
}
fn kind(value: ShaderResourceKind) -> u32 {
    match value {
        ShaderResourceKind::ConstantBuffer => 0,
        ShaderResourceKind::StorageBuffer => 1,
        ShaderResourceKind::SampledImage => 2,
        ShaderResourceKind::SampledImage2DArray => 3,
        ShaderResourceKind::SampledDepthImage => 4,
        ShaderResourceKind::SampledDepthImage2DArray => 5,
        ShaderResourceKind::StorageImage => 6,
        ShaderResourceKind::Sampler => 7,
        ShaderResourceKind::ComparisonSampler => 8,
        ShaderResourceKind::SampledImageCube => 9,
    }
}
fn encode(stages: &[CompiledStage]) -> Vec<u8> {
    let mut b = Vec::new();
    word(&mut b, stages.len() as u32);
    for s in stages {
        word(&mut b, stage(s.stage));
        word(&mut b, s.module.push_constant_bytes);
        for size in s.workgroup_size.unwrap_or([0; 3]) {
            word(&mut b, size);
        }
        word(&mut b, s.shared_memory_bytes);
        word(&mut b, s.output_control_points.unwrap_or(0));
        word(&mut b, s.module.bindings.len() as u32);
        for binding in &s.module.bindings {
            word(&mut b, binding.resource.index() as u32);
            word(&mut b, binding.binding);
            word(&mut b, kind(binding.kind));
            word(
                &mut b,
                u32::from(binding.readable) | u32::from(binding.writable) << 1,
            );
        }
        word(&mut b, s.module.words.len() as u32);
        for &v in &s.module.words {
            word(&mut b, v);
        }
    }
    b
}
fn decode(bytes: &[u8]) -> Option<Box<[CompiledStage]>> {
    let mut r = Reader(bytes);
    let count = r.u32()?;
    if count == 0 || count > 6 {
        return None;
    }
    let mut stages = Vec::new();
    for _ in 0..count {
        let stage = *[
            ShaderStage::Vertex,
            ShaderStage::TessellationControl,
            ShaderStage::TessellationEvaluation,
            ShaderStage::Geometry,
            ShaderStage::Fragment,
            ShaderStage::Compute,
        ]
        .get(r.u32()? as usize)?;
        let push_constant_bytes = r.u32()?;
        let dimensions = [r.u32()?, r.u32()?, r.u32()?];
        let workgroup_size = if dimensions == [0; 3] {
            None
        } else {
            if dimensions.contains(&0) {
                return None;
            }
            Some(dimensions)
        };
        let shared_memory_bytes = r.u32()?;
        let output_control_points = r.u32()?;
        let output_control_points = (output_control_points != 0).then_some(output_control_points);
        if (stage == ShaderStage::Compute) != workgroup_size.is_some() {
            return None;
        }
        let count = r.u32()?;
        if count as usize > r.0.len() / 16 {
            return None;
        }
        let mut bindings = Vec::new();
        for _ in 0..count {
            let resource = ResourceId::new(r.u32()?);
            let binding = r.u32()?;
            let kind = *[
                ShaderResourceKind::ConstantBuffer,
                ShaderResourceKind::StorageBuffer,
                ShaderResourceKind::SampledImage,
                ShaderResourceKind::SampledImage2DArray,
                ShaderResourceKind::SampledDepthImage,
                ShaderResourceKind::SampledDepthImage2DArray,
                ShaderResourceKind::StorageImage,
                ShaderResourceKind::Sampler,
                ShaderResourceKind::ComparisonSampler,
                ShaderResourceKind::SampledImageCube,
            ]
            .get(r.u32()? as usize)?;
            let access = r.u32()?;
            if access > 3 {
                return None;
            }
            bindings.push(spirv::Binding {
                resource,
                binding,
                kind,
                readable: access & 1 != 0,
                writable: access & 2 != 0,
            });
        }
        let count = r.u32()? as usize;
        if count < 5 || count > r.0.len() / 4 {
            return None;
        }
        let words: Box<_> = r
            .take(count * 4)?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if words[0] != 0x07230203 {
            return None;
        }
        stages.push(CompiledStage::new(
            stage,
            spirv::Module {
                words,
                bindings: bindings.into_boxed_slice(),
                push_constant_bytes,
            },
            workgroup_size,
            shared_memory_bytes,
            output_control_points,
        ));
    }
    r.0.is_empty().then(|| stages.into_boxed_slice())
}

impl crate::VulkanDevice {
    /// Startup defaults from enabled features. The concrete frontend fills only
    /// stage/guest specialization; each resulting variant is keyed on a miss.
    pub fn shader_options(&self) -> spirv::Options {
        let p = self.profile();
        let f = &p.float_controls;
        spirv::Options {
            shader: nixe_gpu::SpirvShaderOptions {
                depth_clip_negative_one_to_one: false,
                vertex_position_transform: None,
                input_control_points: 0,
                tessellation_mode: None,
                float32: nixe_gpu::SpirvFloat32Capabilities {
                    denorm_preserve: f.float32_denorm_preserve,
                    rounding_mode_rte: f.float32_round_to_nearest_even,
                    signed_zero_inf_nan_preserve: f.float32_signed_zero_inf_nan_preserve,
                    fused_multiply_add: f.fma_float32,
                },
                float64: nixe_gpu::SpirvFloat64Capabilities {
                    enabled: p.core.shader_float64 != 0,
                    rounding_mode_rte: f.float64_round_to_nearest_even,
                    signed_zero_inf_nan_preserve: f.float64_signed_zero_inf_nan_preserve,
                },
            },
            integer64: p.core.shader_int64 != 0,
            subgroup_quad: p
                .subgroup
                .stages
                .contains(ash::vk::ShaderStageFlags::FRAGMENT)
                && p.subgroup
                    .operations
                    .contains(ash::vk::SubgroupFeatureFlags::QUAD),
            shared_memory_bytes: 0,
            local_memory_bytes: 0,
        }
    }
}
