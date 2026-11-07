//! The graphics owner selects and compiles pipelines before locking resource
//! caches, as PipelineCache::CurrentGraphicsPipeline/CurrentComputePipeline.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp
use super::pipeline_resources::PipelineResources;
use super::*;
use maxwell::fixed_state::FixedPipelineState;

/// Current fixed state and pipeline transitions belong to the selected channel,
/// outside the coherence backend borrow used by compiler memory reads.
#[derive(Default)]
pub(super) struct PipelineContext {
    pub(super) current_graphics: Option<usize>,
    pub(super) pipeline_state: Option<FixedPipelineState>,
    pub(super) pixel_depth: Option<bool>,
}
enum Graphics {
    Captured(Arc<GuestShaderCode>),
    Shader(Pending<ShaderArtifact>),
    Compiled {
        shader: Arc<ShaderArtifact>,
        graphics: GraphicsState,
        vertex_attributes: u32,
    },
    Ready(usize),
}
pub(super) type ConfiguredGraphics = (
    smallvec::SmallVec<[crate::residency::execution::ShaderImage; 64]>,
    crate::residency::execution::DescriptorQueue,
    Draw,
);
pub(super) type ConfigureGraphics = fn(
    &GraphicsPipeline,
    &mut NativeExecution<'_>,
    &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
    &nixe_gpu_maxwell::MaxwellThreeDDrawState,
    &nixe_gpu_maxwell::MaxwellGpuMemory,
    [Option<u8>; 5],
    Draw,
    u128,
) -> Result<ConfiguredGraphics, VulkanError>;
pub(super) struct GraphicsPipeline {
    pub(super) key: FixedPipelineState,
    transitions: Vec<usize>,
    pub(super) pipeline: Arc<NativePipeline>,
    pub(super) resources: PipelineResources,
    pub(super) vertex_attributes: u32,
    pub(super) configure: ConfigureGraphics,
}
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct ComputeKey {
    shader: u64,
    shared_memory: u32,
    workgroup: [u32; 3],
}
impl ComputeKey {
    fn environment_key(self) -> Vec<u8> {
        let mut bytes = vec![1];
        bytes.extend_from_slice(&self.shader.to_le_bytes());
        bytes.extend_from_slice(&self.shared_memory.to_le_bytes());
        for size in self.workgroup {
            bytes.extend_from_slice(&size.to_le_bytes());
        }
        bytes
    }
    fn from_environment_key(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 25 || bytes[0] != 1 {
            return None;
        }
        Some(Self {
            shader: u64::from_le_bytes(bytes[1..9].try_into().ok()?),
            shared_memory: u32::from_le_bytes(bytes[9..13].try_into().ok()?),
            workgroup: std::array::from_fn(|i| {
                u32::from_le_bytes(bytes[13 + i * 4..17 + i * 4].try_into().unwrap())
            }),
        })
    }
}
enum ComputePipelineState {
    Captured(Arc<GuestShaderCode>),
    Shader(Pending<ShaderArtifact>),
    Compiled(Arc<ShaderArtifact>),
    Ready(usize),
}
pub(super) struct ComputePipeline {
    key: ComputeKey,
    pub(super) pipeline: Arc<NativePipeline>,
    pub(super) resources: PipelineResources,
}
enum DiskPipeline {
    Graphics(FixedPipelineState),
    Compute(ComputeKey),
}
pub(super) struct PipelineCache {
    pub(super) shaders: Arc<ShaderCompiler>,
    pub(super) compiler: Arc<PipelineCompiler>,
    graphics: HashMap<FixedPipelineState, Graphics>,
    pub(super) graphics_pipelines: Vec<GraphicsPipeline>,
    compute: HashMap<ComputeKey, ComputePipelineState>,
    pub(super) compute_pipelines: Vec<ComputePipeline>,
    pub(super) current_compute: Option<usize>,
    capture_compiler: Option<nixe_shader_recompiler::Compiler>,
    archive: Option<nixe_gpu_maxwell::recompiler::archive::EnvironmentArchive>,
}
impl PipelineCache {
    pub(super) fn new(shaders: Arc<ShaderCompiler>, compiler: Arc<PipelineCompiler>) -> Self {
        Self {
            shaders,
            compiler,
            graphics: HashMap::default(),
            graphics_pipelines: Vec::new(),
            compute: HashMap::default(),
            compute_pipelines: Vec::new(),
            current_compute: None,
            capture_compiler: None,
            archive: None,
        }
    }
    pub(super) fn set_capture_cache(
        &mut self,
        directory: &std::path::Path,
        namespace: ContentId,
        profile: &DeviceProfile,
        budget: u64,
    ) {
        let archive = match nixe_gpu_maxwell::recompiler::archive::EnvironmentArchive::new(
            directory,
            namespace,
            profile.shader_compiler.clone(),
            budget,
        ) {
            Ok(archive) => archive,
            Err(error) => {
                log::warn!("shader environment serialization: {error}");
                return;
            }
        };
        let files = archive.load();
        self.archive = Some(archive);
        let mut remaining = Vec::new();
        for environment in files {
            if let Some(key) = FixedPipelineState::from_environment_key(&environment.key) {
                if !key.matches_profile(profile) {
                    continue;
                }
                self.graphics
                    .insert(key.clone(), Graphics::Captured(environment.inputs));
                remaining.push(DiskPipeline::Graphics(key));
            } else if let Some(key) = ComputeKey::from_environment_key(&environment.key) {
                self.compute
                    .insert(key, ComputePipelineState::Captured(environment.inputs));
                remaining.push(DiskPipeline::Compute(key));
            }
        }
        let total = remaining.len();
        let mut loaded = 0;
        let mut building = Vec::new();
        let mut statistics = maxwell::NativeFrontendStatistics::default();
        let mut drained = false;
        while !remaining.is_empty() {
            let before = remaining.len();
            remaining.retain(|entry| {
                let result = match entry {
                    DiskPipeline::Graphics(key) => {
                        self.advance_graphics(key, profile, &mut statistics)
                    }
                    DiskPipeline::Compute(key) => self.advance_compute(*key, profile),
                };
                match result {
                    Ok(Some(index)) => {
                        let (entry, pipeline) = match entry {
                            DiskPipeline::Graphics(key) => (
                                DiskPipeline::Graphics(key.clone()),
                                Arc::clone(&self.graphics_pipelines[index].pipeline),
                            ),
                            DiskPipeline::Compute(key) => (
                                DiskPipeline::Compute(*key),
                                Arc::clone(&self.compute_pipelines[index].pipeline),
                            ),
                        };
                        building.push((entry, pipeline));
                        false
                    }
                    Ok(None) => true,
                    Err(error) => {
                        log::warn!("cached shader pipeline: {error}");
                        false
                    }
                }
            });
            if remaining.is_empty() {
                break;
            }
            if remaining.len() != before {
                drained = false;
            }
            let pending = remaining.iter().find_map(|entry| match entry {
                DiskPipeline::Graphics(key) => match self.graphics.get(key) {
                    Some(Graphics::Shader(pending)) => Some(pending.clone()),
                    _ => None,
                },
                DiskPipeline::Compute(key) => match self.compute.get(key) {
                    Some(ComputePipelineState::Shader(pending)) => Some(pending.clone()),
                    _ => None,
                },
            });
            if let Some(pending) = pending {
                let _ = pending.wait();
                drained = false;
            } else if !drained {
                loaded += self.finish_disk_pipelines(&mut building);
                drained = true;
            } else {
                // Optional disk entries which cannot fit the configured cold
                // compilation budget remain live misses, as failed cache loads.
                log::warn!(
                    "{} cached pipelines exceed the compilation admission budget",
                    remaining.len()
                );
                break;
            }
        }
        for entry in &remaining {
            match entry {
                DiskPipeline::Graphics(key) => {
                    self.graphics.remove(key);
                }
                DiskPipeline::Compute(key) => {
                    self.compute.remove(key);
                }
            }
        }
        loaded += self.finish_disk_pipelines(&mut building);
        if total != 0 {
            log::info!(
                "loaded {} native pipeline environments before guest execution",
                loaded
            );
        }
    }
    fn finish_disk_pipelines(
        &mut self,
        building: &mut Vec<(DiskPipeline, Arc<NativePipeline>)>,
    ) -> usize {
        let mut loaded = 0;
        for (entry, pipeline) in building.drain(..) {
            match pipeline.handle() {
                Ok(_) => loaded += 1,
                Err(error) => {
                    log::warn!("cached native pipeline construction: {error}");
                    match entry {
                        DiskPipeline::Graphics(key) => {
                            self.graphics.remove(&key);
                        }
                        DiskPipeline::Compute(key) => {
                            self.compute.remove(&key);
                        }
                    }
                }
            }
        }
        loaded
    }
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn capture_inputs(
        &mut self,
        key: Vec<u8>,
        profile: &DeviceProfile,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        statistics: &mut maxwell::NativeFrontendStatistics,
        requests: impl FnOnce() -> Result<
            Vec<(
                Option<u8>,
                nixe_gpu_maxwell::recompiler::CaptureRequest,
                usize,
            )>,
            VulkanError,
        >,
    ) -> Result<Arc<GuestShaderCode>, VulkanError> {
        if self.capture_compiler.is_none() {
            self.capture_compiler = Some(
                nixe_shader_recompiler::Compiler::new()
                    .map_err(|e| VulkanError::Compilation(e.to_string().into()))?,
            );
        }
        let inputs = nixe_gpu_maxwell::recompiler::capture_program(
            self.capture_compiler.as_mut().unwrap(),
            space,
            requests()?,
            &profile.shader_compiler,
            access,
        )
        .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        statistics.shader_captures += 1;
        if let Some(archive) = &self.archive {
            archive.serialize(key, Arc::clone(&inputs));
        }
        Ok(inputs)
    }
    fn request_shader(
        &self,
        profile: &DeviceProfile,
        code: &Arc<GuestShaderCode>,
        fixed: Option<&FixedPipelineState>,
    ) -> Result<Pending<ShaderArtifact>, CompileError> {
        let runtime = fixed.map(FixedPipelineState::runtime).transpose()?;
        self.shaders.request_guest_with_depth(
            Arc::clone(code),
            Arc::new(profile.shader_compiler.clone()),
            runtime,
            fixed.and_then(FixedPipelineState::fragment_depth_transform),
        )
    }
    fn select_graphics(&mut self, context: &mut PipelineContext, index: usize) {
        if let Some(previous) = context.current_graphics {
            self.graphics_pipelines[previous].transitions.push(index);
        }
        context.current_graphics = Some(index);
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn current_graphics(
        &mut self,
        profile: &DeviceProfile,
        context: &mut PipelineContext,
        statistics: &mut maxwell::NativeFrontendStatistics,
        shaders: &maxwell::GraphicsShaders,
        registers: &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
        draw: &nixe_gpu_maxwell::MaxwellThreeDDrawState,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<bool, VulkanError> {
        context.refresh_fixed_pipeline_state(profile, shaders, registers, draw)?;
        statistics.fixed_state_refreshes += 1;
        let id = context
            .pipeline_state
            .as_ref()
            .expect("current pipeline state");
        if let Some(index) = context.current_graphics {
            let pipeline = &self.graphics_pipelines[index];
            let next = if pipeline.key == *id {
                Some(index)
            } else {
                pipeline
                    .transitions
                    .iter()
                    .copied()
                    .find(|&next| self.graphics_pipelines[next].key == *id)
            };
            if let Some(next) = next {
                context.current_graphics = Some(next);
                return Ok(true);
            }
        }
        if !self.graphics.contains_key(id) {
            let inputs = self.capture_inputs(
                id.environment_key(),
                profile,
                space,
                access,
                statistics,
                || {
                    shaders
                        .infos
                        .iter()
                        .enumerate()
                        .filter_map(|(index, info)| {
                            info.as_ref().map(|info| {
                                nixe_gpu_maxwell::recompiler::graphics_request(registers, index)
                                    .map(|request| {
                                        (
                                            registers
                                                .shader_bindings()
                                                .pipeline(index)
                                                .effective_group(),
                                            request,
                                            info.size_bytes,
                                        )
                                    })
                                    .map_err(|e| VulkanError::Compilation(e.to_string().into()))
                            })
                        })
                        .collect()
                },
            )?;
            self.graphics.insert(id.clone(), Graphics::Captured(inputs));
        }
        let Some(index) = self.advance_graphics(id, profile, statistics)? else {
            return Ok(false);
        };
        self.select_graphics(context, index);
        Ok(true)
    }
    fn advance_graphics(
        &mut self,
        id: &FixedPipelineState,
        profile: &DeviceProfile,
        statistics: &mut maxwell::NativeFrontendStatistics,
    ) -> Result<Option<usize>, VulkanError> {
        let mut entry = self
            .graphics
            .remove(id)
            .expect("admitted graphics pipeline");
        loop {
            entry = match entry {
                Graphics::Ready(index) => {
                    self.graphics.insert(id.clone(), Graphics::Ready(index));
                    return Ok(Some(index));
                }
                Graphics::Captured(inputs) => {
                    match self.request_shader(profile, &inputs, Some(id)) {
                        Ok(pending) => Graphics::Shader(pending),
                        Err(CompileError::Backpressure | CompileError::Budget) => {
                            self.graphics.insert(id.clone(), Graphics::Captured(inputs));
                            return Ok(None);
                        }
                        Err(error) => {
                            return Err(VulkanError::Compilation(error.to_string().into()));
                        }
                    }
                }
                Graphics::Shader(pending) => {
                    let Some(shader) = pending.poll() else {
                        self.graphics.insert(id.clone(), Graphics::Shader(pending));
                        return Ok(None);
                    };
                    let shader =
                        shader.map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
                    let shader = Self::graphics_shader(id, profile, shader)?;
                    let vertex_attributes = shader
                        .stages
                        .iter()
                        .find(|stage| stage.stage == nixe_gpu::ShaderStage::Vertex)
                        .and_then(|stage| stage.guest.as_ref())
                        .map_or(0, |stage| {
                            (0..32).fold(0, |mask, location| {
                                mask | (u32::from(stage.info.loads.generic(location)) << location)
                            })
                        });
                    Graphics::Compiled {
                        shader,
                        graphics: id.graphics(profile, vertex_attributes)?,
                        vertex_attributes,
                    }
                }
                Graphics::Compiled {
                    shader,
                    graphics,
                    vertex_attributes,
                } => {
                    let pipeline = match self.compiler.request(PipelineRecipe::Graphics {
                        shader: Arc::clone(&shader),
                        state: graphics.clone(),
                    }) {
                        Ok(pipeline) => pipeline,
                        Err(CompileError::Backpressure | CompileError::Budget) => {
                            self.graphics.insert(
                                id.clone(),
                                Graphics::Compiled {
                                    shader,
                                    graphics,
                                    vertex_attributes,
                                },
                            );
                            return Ok(None);
                        }
                        Err(error) => {
                            return Err(VulkanError::Compilation(error.to_string().into()));
                        }
                    };
                    let index = self.graphics_pipelines.len();
                    let resources = PipelineResources::new(&shader, &pipeline.layout.key)?;
                    let configure = GraphicsPipeline::configure_func(&shader, &resources);
                    self.graphics_pipelines.push(GraphicsPipeline {
                        configure,
                        key: id.clone(),
                        transitions: Vec::new(),
                        pipeline,
                        vertex_attributes,
                        resources,
                    });
                    self.graphics.insert(id.clone(), Graphics::Ready(index));
                    statistics.prepared_graphics += 1;
                    return Ok(Some(index));
                }
            };
        }
    }
    fn graphics_shader(
        id: &FixedPipelineState,
        profile: &DeviceProfile,
        mut shader: Arc<ShaderArtifact>,
    ) -> Result<Arc<ShaderArtifact>, VulkanError> {
        if let Some(tessellation) = id.tessellation()
            && matches!(
                tessellation.control,
                nixe_gpu::TessellationControl::DefaultLevels { .. }
            )
        {
            let interface = |kind| {
                shader
                    .stages
                    .iter()
                    .find(|stage| stage.stage == kind)
                    .and_then(|stage| stage.guest.as_ref())
                    .map(|stage| &stage.interface)
                    .ok_or(VulkanError::InvalidResource(
                        "default tessellation control requires vertex and evaluation interfaces",
                    ))
            };
            let artifact = nixe_gpu::shader_module::synthetic::default_control(
                interface(nixe_gpu::ShaderStage::Vertex)?,
                interface(nixe_gpu::ShaderStage::TessellationEvaluation)?,
                u32::from(tessellation.input_control_points),
                tessellation.mode.domain,
                profile.shader_compiler.profile.supported_spirv,
            )
            .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
            let control = CompiledStage::from_guest(artifact)
                .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
            let mut stages = shader.stages.to_vec();
            stages.insert(1, Arc::new(control));
            shader = Arc::new(ShaderArtifact::new(stages.into_boxed_slice()));
        }
        Ok(shader)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn current_compute(
        &mut self,
        profile: &DeviceProfile,
        launch: &nixe_gpu_maxwell::MaxwellResolvedComputeLaunch,
        shader: &nixe_gpu_maxwell::recompiler::ShaderInfo,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        statistics: &mut maxwell::NativeFrontendStatistics,
    ) -> Result<Option<usize>, VulkanError> {
        let key = ComputeKey {
            shader: shader.unique_hash,
            shared_memory: launch.shared_memory_size(),
            workgroup: launch.workgroup_size(),
        };
        if let Some(slot) = self.current_compute
            && self.compute_pipelines[slot].key == key
        {
            return Ok(Some(slot));
        }
        if !self.compute.contains_key(&key) {
            let inputs = self.capture_inputs(
                key.environment_key(),
                profile,
                space,
                access,
                statistics,
                || Ok(vec![(None, launch.capture_request(), shader.size_bytes)]),
            )?;
            self.compute
                .insert(key, ComputePipelineState::Captured(inputs));
        }
        let slot = self.advance_compute(key, profile)?;
        if let Some(slot) = slot {
            self.current_compute = Some(slot);
        }
        Ok(slot)
    }
    fn advance_compute(
        &mut self,
        key: ComputeKey,
        profile: &DeviceProfile,
    ) -> Result<Option<usize>, VulkanError> {
        let mut entry = self
            .compute
            .remove(&key)
            .expect("admitted compute pipeline");
        loop {
            entry = match entry {
                ComputePipelineState::Ready(slot) => {
                    self.compute.insert(key, ComputePipelineState::Ready(slot));
                    return Ok(Some(slot));
                }
                ComputePipelineState::Captured(inputs) => match self
                    .request_shader(profile, &inputs, None)
                {
                    Ok(pending) => ComputePipelineState::Shader(pending),
                    Err(CompileError::Backpressure | CompileError::Budget) => {
                        self.compute
                            .insert(key, ComputePipelineState::Captured(inputs));
                        return Ok(None);
                    }
                    Err(error) => return Err(VulkanError::Compilation(error.to_string().into())),
                },
                ComputePipelineState::Shader(pending) => {
                    let Some(shader) = pending.poll() else {
                        self.compute
                            .insert(key, ComputePipelineState::Shader(pending));
                        return Ok(None);
                    };
                    ComputePipelineState::Compiled(
                        shader.map_err(|e| VulkanError::Compilation(e.to_string().into()))?,
                    )
                }
                ComputePipelineState::Compiled(shader) => {
                    let pipeline = match self
                        .compiler
                        .request(PipelineRecipe::Compute(Arc::clone(&shader)))
                    {
                        Ok(pipeline) => pipeline,
                        Err(CompileError::Backpressure | CompileError::Budget) => {
                            self.compute
                                .insert(key, ComputePipelineState::Compiled(shader));
                            return Ok(None);
                        }
                        Err(error) => {
                            return Err(VulkanError::Compilation(error.to_string().into()));
                        }
                    };
                    let resources = PipelineResources::new(&shader, &pipeline.layout.key)?;
                    let slot = self.compute_pipelines.len();
                    self.compute_pipelines.push(ComputePipeline {
                        key,
                        pipeline,
                        resources,
                    });
                    self.compute.insert(key, ComputePipelineState::Ready(slot));
                    return Ok(Some(slot));
                }
            };
        }
    }
}
