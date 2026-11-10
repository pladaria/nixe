//! Pipeline-owned translation and native construction, with Eden's last,
//! transition, and global table lookup order. There are no intermediate caches.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp
use super::pipeline_resources::PipelineResources;
use super::*;
use maxwell::fixed_state::FixedPipelineState;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
pub(super) type ConfigureGraphics = fn(
    &GraphicsPipeline,
    &mut NativeExecution<'_>,
    &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
    &nixe_gpu_maxwell::MaxwellThreeDDrawState,
    &nixe_gpu_maxwell::MaxwellGpuMemory,
    [Option<u8>; 5],
    Draw,
    u128,
) -> Result<(), VulkanError>;
pub(super) struct GraphicsPipeline {
    pub(super) key: FixedPipelineState,
    transition_keys: Vec<FixedPipelineState>,
    transitions: Vec<usize>,
    pub(super) pipeline: NativePipeline,
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

pub(super) struct ComputePipeline {
    key: ComputeKey,
    pub(super) pipeline: NativePipeline,
    pub(super) resources: PipelineResources,
}
// Disk jobs move complete metadata to its owner; boxing would add an allocation
// to every loaded graphics pipeline just to shrink this temporary message.
#[allow(clippy::large_enum_variant)]
enum DiskPipeline {
    Graphics(GraphicsPipeline),
    Compute(ComputePipeline),
}
#[derive(Default)]
struct FlushStatus {
    in_flight: AtomicBool,
    bytes: AtomicUsize,
}
pub(super) struct PipelineCache {
    pub(super) compiler: Arc<PipelineCompiler>,
    graphics: HashMap<([u32; 2], FixedPipelineState), usize>,
    pub(super) graphics_pipelines: Vec<GraphicsPipeline>,
    compute: HashMap<([u32; 2], ComputeKey), usize>,
    pub(super) compute_pipelines: Vec<ComputePipeline>,
    pub(super) current_graphics: Option<usize>,
    pub(super) pipeline_state: Option<FixedPipelineState>,
    configuration: Arc<nixe_gpu::shader_compilation::Configuration>,
    main_compiler: Option<nixe_shader_recompiler::Compiler>,
    archive: Option<crate::cache::archive::EnvironmentArchive>,
    flush: Arc<FlushStatus>,
    pipelines_since_flush: usize,
    last_flush: Option<Instant>,
}
impl PipelineCache {
    pub(super) fn new(
        compiler: Arc<PipelineCompiler>,
        configuration: &nixe_gpu::shader_compilation::Configuration,
    ) -> Self {
        Self {
            compiler,
            graphics: HashMap::default(),
            graphics_pipelines: Vec::new(),
            compute: HashMap::default(),
            compute_pipelines: Vec::new(),
            current_graphics: None,
            pipeline_state: None,
            configuration: Arc::new(configuration.clone()),
            main_compiler: None,
            archive: None,
            flush: Arc::default(),
            pipelines_since_flush: 0,
            last_flush: None,
        }
    }
    fn scale_key(&self) -> [u32; 2] {
        let resolution = self.configuration.settings.resolution_info;
        [resolution.up_scale, resolution.down_shift]
    }
    pub(super) fn change_resolution(&mut self, resolution: crate::ResolutionScaling) {
        Arc::make_mut(&mut self.configuration)
            .settings
            .resolution_info = resolution;
        self.current_graphics = None;
        self.pipeline_state = None;
    }
    pub(super) fn load_disk_resources(
        &mut self,
        directory: &std::path::Path,
        profile: &Arc<DeviceProfile>,
        stop: &AtomicBool,
    ) {
        let started = Instant::now();
        log::info!("loading shader and pipeline cache: {}", directory.display());
        let archive = match crate::cache::archive::EnvironmentArchive::open(
            directory,
            self.compiler.serialization(),
        ) {
            Ok(archive) => archive,
            Err(error) => {
                log::warn!("shader environment cache load: {error}");
                return;
            }
        };
        let mut total = 0;
        let (sender, receiver) = std::sync::mpsc::channel();
        archive.load(stop, |environment| {
            let graphics = FixedPipelineState::from_environment_key(&environment.key)
                .filter(|key| key.matches_profile(profile));
            let compute = ComputeKey::from_environment_key(&environment.key);
            if graphics.is_none() && compute.is_none() {
                return;
            }
            total += 1;
            let compiler = Arc::clone(&self.compiler);
            let profile = Arc::clone(profile);
            let configuration = Arc::clone(&self.configuration);
            let sender = sender.clone();
            if let Err(error) = self.compiler.queue(move |worker| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<DiskPipeline, VulkanError> {
                        let mut pools =
                            nixe_shader_recompiler::Compiler::new().map_err(compilation)?;
                        let programs = environment
                            .inputs
                            .translate(&mut pools, &configuration)
                            .map_err(compilation)?;
                        if let Some(key) = graphics {
                            Self::build_graphics(
                                &compiler,
                                &mut pools,
                                &profile,
                                &configuration,
                                key,
                                &programs,
                                Some(worker),
                            )
                            .map(DiskPipeline::Graphics)
                        } else {
                            Self::build_compute(
                                &compiler,
                                &mut pools,
                                &configuration,
                                compute.unwrap(),
                                &programs,
                                Some(worker),
                            )
                            .map(DiskPipeline::Compute)
                        }
                    },
                ))
                .unwrap_or_else(|_| Err(compilation("cached pipeline construction panicked")));
                let _ = sender.send(result);
            }) {
                log::warn!("cached pipeline queue: {error}");
            }
        });
        log::info!(
            "read and queued {total} pipeline environments in {:?}",
            started.elapsed()
        );
        self.archive = Some(archive);
        drop(sender);
        let mut loaded = 0;
        for result in receiver {
            match result {
                Ok(DiskPipeline::Graphics(pipeline)) => {
                    if !self
                        .graphics
                        .contains_key(&(self.scale_key(), pipeline.key))
                    {
                        let slot = self.graphics_pipelines.len();
                        self.graphics.insert((self.scale_key(), pipeline.key), slot);
                        self.graphics_pipelines.push(pipeline);
                        loaded += 1;
                    }
                }
                Ok(DiskPipeline::Compute(pipeline)) => {
                    if !self.compute.contains_key(&(self.scale_key(), pipeline.key)) {
                        let slot = self.compute_pipelines.len();
                        self.compute.insert((self.scale_key(), pipeline.key), slot);
                        self.compute_pipelines.push(pipeline);
                        loaded += 1;
                    }
                }
                Err(error) => log::warn!("cached pipeline construction: {error}"),
            }
        }
        self.compiler.wait_for_requests();
        match self.compiler.checkpoint() {
            Ok(size) => {
                self.flush.bytes.store(size, Ordering::Relaxed);
                self.last_flush = Some(Instant::now());
            }
            Err(error) => log::warn!("Vulkan driver cache write: {error}"),
        }
        log::info!(
            "loaded {loaded}/{total} native pipeline environments before guest execution in {:?}",
            started.elapsed()
        );
    }
    fn serialize(&mut self, key: Vec<u8>, inputs: Arc<GuestShaderCode>) {
        let Some(archive) = &self.archive else {
            return;
        };
        archive.serialize(key, inputs);
        self.pipelines_since_flush += 1;
        if self.pipelines_since_flush < 128 {
            return;
        }
        let interval = Duration::from_secs(
            (self.flush.bytes.load(Ordering::Relaxed) / (1024 * 1024)).max(30) as u64,
        );
        if self
            .last_flush
            .is_some_and(|time| time.elapsed() < interval)
            || self.flush.in_flight.swap(true, Ordering::AcqRel)
        {
            return;
        }
        self.pipelines_since_flush = 0;
        self.last_flush = Some(Instant::now());
        let flush = Arc::clone(&self.flush);
        if let Err(error) = self.compiler.queue_checkpoint(move |bytes| {
            flush.bytes.store(bytes, Ordering::Relaxed);
            flush.in_flight.store(false, Ordering::Release);
        }) {
            self.flush.in_flight.store(false, Ordering::Release);
            log::warn!("Vulkan driver cache serialization: {error}");
        }
    }
    #[allow(clippy::type_complexity)]
    fn capture_inputs(
        &mut self,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        statistics: &mut maxwell::NativeFrontendStatistics,
        requests: Vec<(
            Option<u8>,
            nixe_gpu_maxwell::recompiler::CaptureRequest,
            usize,
        )>,
    ) -> Result<(Arc<GuestShaderCode>, Vec<nixe_shader_recompiler::ProgramId>), VulkanError> {
        if self.main_compiler.is_none() {
            self.main_compiler =
                Some(nixe_shader_recompiler::Compiler::new().map_err(compilation)?);
        }
        let inputs = nixe_gpu_maxwell::recompiler::capture_program(
            self.main_compiler.as_mut().unwrap(),
            space,
            requests,
            &self.configuration,
            access,
        )
        .map_err(compilation)?;
        statistics.shader_captures += 1;
        Ok(inputs)
    }
    fn select_graphics(&mut self, index: usize) {
        if let Some(previous) = self.current_graphics {
            let key = self.graphics_pipelines[index].key;
            self.graphics_pipelines[previous].transition_keys.push(key);
            self.graphics_pipelines[previous].transitions.push(index);
        }
        self.current_graphics = Some(index);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn current_graphics(
        &mut self,
        profile: &DeviceProfile,
        statistics: &mut maxwell::NativeFrontendStatistics,
        shaders: &maxwell::GraphicsShaders,
        registers: &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
        draw: &nixe_gpu_maxwell::MaxwellThreeDDrawState,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), VulkanError> {
        self.refresh_fixed_pipeline_state(profile, shaders, registers, draw)?;
        statistics.fixed_state_refreshes += 1;
        let id = self
            .pipeline_state
            .as_ref()
            .expect("current pipeline state");
        if let Some(index) = self.current_graphics {
            let pipeline = &self.graphics_pipelines[index];
            let next = if pipeline.key == *id {
                Some(index)
            } else {
                pipeline
                    .transition_keys
                    .iter()
                    .position(|key| key == id)
                    .map(|slot| pipeline.transitions[slot])
            };
            if let Some(next) = next {
                self.current_graphics = Some(next);
                return Ok(());
            }
        }

        let index = if let Some(&index) = self.graphics.get(&(self.scale_key(), *id)) {
            index
        } else {
            let id = *id;
            let new_environment = !crate::RESOLUTION_SCALES.iter().any(|scale| {
                self.graphics
                    .contains_key(&([scale.up_scale, scale.down_shift], id))
            });
            let requests = shaders
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
                .collect::<Result<_, _>>()?;
            let (inputs, programs) = self.capture_inputs(space, access, statistics, requests)?;
            let pipeline = Self::build_graphics(
                &self.compiler,
                self.main_compiler.as_mut().unwrap(),
                profile,
                &self.configuration,
                id,
                &programs,
                None,
            )?;
            let index = self.graphics_pipelines.len();
            self.graphics_pipelines.push(pipeline);
            self.graphics.insert((self.scale_key(), id), index);
            statistics.prepared_graphics += 1;
            if new_environment {
                self.serialize(id.environment_key(), inputs);
            }
            index
        };
        self.select_graphics(index);
        Ok(())
    }
    fn build_graphics(
        compiler: &PipelineCompiler,
        pools: &mut nixe_shader_recompiler::Compiler,
        profile: &DeviceProfile,
        configuration: &nixe_gpu::shader_compilation::Configuration,
        key: FixedPipelineState,
        programs: &[nixe_shader_recompiler::ProgramId],
        worker: Option<usize>,
    ) -> Result<GraphicsPipeline, VulkanError> {
        let runtime = key.runtime().map_err(compilation)?;
        nixe_gpu::trace::event("shader_compile", 0, 0, 0);
        let shader = crate::cache::compile_guest(pools, programs, configuration, Some(&runtime))
            .map_err(compilation)?;
        let shader = Self::graphics_shader(&key, profile, shader)?;
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
        let state = key.graphics(profile, vertex_attributes)?;
        let recipe = PipelineRecipe::Graphics {
            shader: Arc::clone(&shader),
            state,
        };
        let pipeline = match worker {
            Some(worker) => compiler.build_immediate(recipe, worker),
            None => compiler.request(recipe),
        }
        .map_err(compilation)?;
        let resources = PipelineResources::new(&shader, &pipeline.layout.key)?;
        let configure = GraphicsPipeline::configure_func(&shader, &resources);
        Ok(GraphicsPipeline {
            key,
            transition_keys: Vec::new(),
            transitions: Vec::new(),
            pipeline,
            resources,
            vertex_attributes,
            configure,
        })
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

    pub(super) fn current_compute(
        &mut self,
        launch: &nixe_gpu_maxwell::MaxwellResolvedComputeLaunch,
        shader: &nixe_gpu_maxwell::recompiler::ShaderInfo,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        statistics: &mut maxwell::NativeFrontendStatistics,
    ) -> Result<usize, VulkanError> {
        let key = ComputeKey {
            shader: shader.unique_hash,
            shared_memory: launch.shared_memory_size(),
            workgroup: launch.workgroup_size(),
        };
        let slot = if let Some(&slot) = self.compute.get(&(self.scale_key(), key)) {
            slot
        } else {
            let new_environment = !crate::RESOLUTION_SCALES.iter().any(|scale| {
                self.compute
                    .contains_key(&([scale.up_scale, scale.down_shift], key))
            });
            let (inputs, programs) = self.capture_inputs(
                space,
                access,
                statistics,
                vec![(None, launch.capture_request(), shader.size_bytes)],
            )?;
            let pipeline = Self::build_compute(
                &self.compiler,
                self.main_compiler.as_mut().unwrap(),
                &self.configuration,
                key,
                &programs,
                None,
            )?;
            let slot = self.compute_pipelines.len();
            self.compute_pipelines.push(pipeline);
            self.compute.insert((self.scale_key(), key), slot);
            if new_environment {
                self.serialize(key.environment_key(), inputs);
            }
            slot
        };
        Ok(slot)
    }
    fn build_compute(
        compiler: &PipelineCompiler,
        pools: &mut nixe_shader_recompiler::Compiler,
        configuration: &nixe_gpu::shader_compilation::Configuration,
        key: ComputeKey,
        programs: &[nixe_shader_recompiler::ProgramId],
        worker: Option<usize>,
    ) -> Result<ComputePipeline, VulkanError> {
        nixe_gpu::trace::event("shader_compile", 0, 0, 0);
        let shader = crate::cache::compile_guest(pools, programs, configuration, None)
            .map_err(compilation)?;
        let recipe = PipelineRecipe::Compute(Arc::clone(&shader));
        let pipeline = match worker {
            Some(worker) => compiler.build_immediate(recipe, worker),
            None => compiler.request(recipe),
        }
        .map_err(compilation)?;
        let resources = PipelineResources::new(&shader, &pipeline.layout.key)?;
        Ok(ComputePipeline {
            key,
            pipeline,
            resources,
        })
    }
}
fn compilation(error: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}
