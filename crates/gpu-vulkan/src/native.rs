//! Concrete native execution selected at startup. Cold compilation is polled by
//! the owner; prepared references bypass content hashes and descriptor updates.
use crate::{cache::*, *};
use ash::vk;
use nixe_gpu::{
    BackendExecutionCompletion, BackendRuntimeError, BackendSubmissionToken,
    BackendVisibilityRequester, ExecutionBatch, FrontendSubmissionId, GpuExecution,
    PreparedStateId, ResidentImage,
    native::*,
    shader_identity::{ContentId, ShaderNamespace},
};
use nixe_memory::{CpuVisibilityRequest, DeviceVisibilityPoint};
use rustc_hash::FxHashMap as HashMap;
use std::{collections::VecDeque, sync::Arc, time::Duration};
mod completion;
mod images;
mod state;
use crate::residency::affine_pipeline;
use images::GuestNativeImage;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeExecutionStatistics {
    pub draws: u64,
    pub prepared_draws: u64,
    pub prepared_graphics: u64,
    pub reused_graphics: u64,
    pub reused_programs: u64,
    pub warm_draws: u64,
    pub batches: u64,
    pub submits: u64,
    pub dispatches: u64,
}
struct Drawing {
    source: Arc<GuestDrawState>,
    draw: PreparedDraw,
    push: Arc<[u8]>,
    recorded: bool,
}
enum Graphics {
    Shader {
        pending: Pending<ShaderArtifact>,
        state: Arc<GuestDrawState>,
        graphics: GraphicsState,
    },
    Pipeline {
        pending: Pending<NativePipeline>,
        state: Arc<GuestDrawState>,
    },
    Ready {
        source: Arc<GuestDrawParameters>,
        state: crate::residency::execution::PreparedGraphicsState,
        push: Arc<[u8]>,
    },
}
enum Computing {
    Shader {
        pending: Pending<ShaderArtifact>,
        source: Arc<GuestProgram>,
    },
    Pipeline {
        pending: Pending<NativePipeline>,
        source: Arc<GuestProgram>,
    },
    Ready {
        source: Arc<GuestProgram>,
        program: PreparedProgram,
        push: Box<[u8]>,
    },
}
struct Completion {
    frontend: FrontendSubmissionId,
    token: BackendSubmissionToken,
    work: Option<WorkTicket>,
    visibility: DeviceVisibilityPoint,
}
pub struct NativeGpuExecution {
    // Drop/join before residency destroys the timeline semaphore, including
    // failure paths that do not reach explicit teardown.
    completion_wakeup: Option<completion::CompletionWakeup>,
    frames: crate::presentation::Frames,
    host: Option<crate::presentation::Host>,
    residency: VulkanResidency,
    video: crate::residency::video::VideoTransfers,
    video_device: Option<[u8; 16]>,
    shaders: ShaderCompiler,
    pipelines: Arc<PipelineCompiler>,
    descriptors: DescriptorCache,
    samplers: SamplerCache,
    drawings: HashMap<PreparedStateId, Drawing>,
    graphics: HashMap<(PreparedStateId, ContentId), Graphics>,
    targets: Option<images::Targets>,
    programs: nixe_gpu::WorkingSet<(PreparedStateId, usize), Arc<PreparedProgram>>,
    next_collection: u64,
    recording_batch: Option<(Option<WorkTicket>, usize)>,
    computing: HashMap<PreparedStateId, Computing>,
    images: HashMap<images::ImageKey, Arc<GuestNativeImage>>,
    draw_arguments: Vec<Draw>,
    affine_pipelines: HashMap<vk::Format, Pending<NativePipeline>>,
    video_pipelines: HashMap<vk::Format, Pending<NativePipeline>>,
    completions: VecDeque<Completion>,
    next: u64,
    pub statistics: NativeExecutionStatistics,
}
fn runtime(error: impl std::fmt::Display) -> BackendRuntimeError {
    BackendRuntimeError::Backend(error.to_string().into())
}
impl NativeGpuExecution {
    pub fn attach_presentation(
        &mut self,
        client: crate::presentation::PresentationClient,
        vsync: bool,
    ) -> Result<(), VulkanError> {
        self.host = Some(crate::presentation::Host::new(
            self.residency.device(),
            client,
            vsync,
        )?);
        Ok(())
    }
    pub fn new(
        device: VulkanDevice,
        namespace: ShaderNamespace,
        configuration: CacheConfiguration,
        resident_bytes: u64,
    ) -> Result<Self, VulkanError> {
        let driver_pool = Arc::new(
            CompilePool::with_name(
                configuration.workers,
                configuration.pending_jobs,
                "gpu-pipeline",
            )
            .map_err(|e| VulkanError::Loader(e.to_string().into()))?,
        );
        // Long driver jobs must never starve the short source/SSA/SPIR-V jobs
        // that discover their dependencies. Both queues retain explicit bounds.
        let translation_pool = Arc::new(
            CompilePool::with_name(1, configuration.pending_jobs, "shader-source")
                .map_err(|e| VulkanError::Loader(e.to_string().into()))?,
        );
        let shaders = ShaderCompiler::new(namespace, Arc::clone(&translation_pool), &configuration);
        let pipelines = Arc::new(PipelineCompiler::new(&device, driver_pool, &configuration)?);
        let descriptors = DescriptorCache::new(&device, configuration.descriptor_sets);
        let samplers = SamplerCache::new(&device, configuration.sampler_entries);
        let mut residency = VulkanResidency::new(device, resident_bytes)?;
        residency.enable_sample_conversion(Arc::clone(&pipelines));
        let video_device = crate::residency::video::device_uuid(residency.device());
        Ok(Self {
            completion_wakeup: None,
            video: Default::default(),
            video_device,
            frames: crate::presentation::Frames::new(),
            host: None,
            residency,
            shaders,
            pipelines,
            descriptors,
            samplers,
            drawings: HashMap::default(),
            graphics: HashMap::default(),
            targets: None,
            programs: nixe_gpu::WorkingSet::new(8192, 32 * 1024 * 1024),
            next_collection: 0,
            recording_batch: None,
            computing: HashMap::default(),
            images: HashMap::default(),
            draw_arguments: Vec::with_capacity(128),
            affine_pipelines: HashMap::default(),
            video_pipelines: HashMap::default(),
            completions: VecDeque::new(),
            next: 1,
            statistics: Default::default(),
        })
    }
    pub fn device_profile(&self) -> &DeviceProfile {
        self.residency.device().profile()
    }
    pub fn recording_statistics(&self) -> RecordingStatistics {
        self.residency.device().recording_statistics()
    }
    pub fn residency_statistics(&self) -> ResidencyStatistics {
        self.residency.statistics()
    }
    fn request_shader(
        &self,
        program: &GuestProgram,
        draw: Option<&GuestDrawState>,
    ) -> Result<Pending<ShaderArtifact>, CompileError> {
        let runtime = draw
            .map(|draw| state::guest_runtime(&draw.parameters))
            .transpose()?;
        self.shaders.request_guest_with_depth(
            Arc::clone(&program.shader),
            Arc::new(self.residency.device().profile().shader_compiler.clone()),
            runtime,
            draw.and_then(|draw| draw.parameters.viewport)
                .and_then(|viewport| viewport.fragment_depth_transform()),
        )
    }

    fn graphics(&mut self, state: &Arc<GuestDrawState>) -> Result<bool, VulkanError> {
        let id = (state.parameters.identity, state.program.shader.identity);
        if !self.graphics.contains_key(&id) {
            let graphics = state::graphics(&state.parameters)?;
            let pending = match self.request_shader(&state.program, Some(state)) {
                Ok(p) => p,
                Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
            };
            self.graphics.insert(
                id,
                Graphics::Shader {
                    pending,
                    state: Arc::clone(state),
                    graphics,
                },
            );
        }
        loop {
            let ready = match &self.graphics[&id] {
                Graphics::Ready { source, .. } => {
                    if !Arc::ptr_eq(source, &state.parameters) {
                        return Err(VulkanError::InvalidResource(
                            "graphics identity was reused for different parameters",
                        ));
                    }
                    return Ok(true);
                }
                Graphics::Shader { pending, .. } => pending.poll().map(|r| r.map(Either::Shader)),
                Graphics::Pipeline { pending, .. } => {
                    pending.poll().map(|r| r.map(Either::Pipeline))
                }
            };
            let Some(ready) = ready else { return Ok(false) };
            let ready = ready.map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            let old = self.graphics.remove(&id).unwrap();
            match (old, ready) {
                (
                    Graphics::Shader {
                        state,
                        graphics,
                        pending,
                    },
                    Either::Shader(mut shader),
                ) => {
                    if let Some(tessellation) = state.parameters.tessellation
                        && matches!(
                            tessellation.control,
                            nixe_gpu::TessellationControl::DefaultLevels { .. }
                        )
                    {
                        let interface = |kind| {
                            shader.stages.iter().find(|stage| stage.stage == kind)
                            .and_then(|stage| stage.guest.as_ref()).map(|stage| &stage.interface)
                            .ok_or(VulkanError::InvalidResource("default tessellation control requires vertex and evaluation interfaces"))
                        };
                        let artifact = nixe_gpu::shader_module::synthetic::default_control(
                            interface(nixe_gpu::ShaderStage::Vertex)?,
                            interface(nixe_gpu::ShaderStage::TessellationEvaluation)?,
                            u32::from(tessellation.input_control_points),
                            tessellation.mode.domain,
                            self.residency
                                .device()
                                .profile()
                                .shader_compiler
                                .profile
                                .supported_spirv,
                        )
                        .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
                        let control = CompiledStage::from_guest(artifact)
                            .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
                        let mut stages = shader.stages.to_vec();
                        stages.insert(1, Arc::new(control));
                        shader = Arc::new(ShaderArtifact::new(stages.into_boxed_slice()));
                    }
                    if state.parameters.topology == nixe_gpu::PrimitiveTopology::Quads {
                        let geometry = {
                            let vertex = shader
                                .stages
                                .iter()
                                .find(|stage| stage.stage == nixe_gpu::ShaderStage::Vertex)
                                .and_then(|stage| stage.guest.as_ref())
                                .ok_or(VulkanError::InvalidResource(
                                    "quad assembly requires vertex interface metadata",
                                ))?;
                            let fragment = shader
                                .stages
                                .iter()
                                .find(|stage| stage.stage == nixe_gpu::ShaderStage::Fragment)
                                .and_then(|stage| stage.guest.as_ref());
                            let artifact = nixe_gpu::shader_module::synthetic::quad_geometry(
                                &vertex.interface,
                                fragment.map(|stage| &stage.interface),
                                state.parameters.provoking_first,
                                self.residency
                                    .device()
                                    .profile()
                                    .shader_compiler
                                    .profile
                                    .supported_spirv,
                            )
                            .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
                            CompiledStage::from_guest(artifact).map_err(|error| {
                                VulkanError::Compilation(error.to_string().into())
                            })?
                        };
                        let mut stages = shader.stages.to_vec();
                        stages.push(Arc::new(geometry));
                        shader = Arc::new(ShaderArtifact::new(stages.into_boxed_slice()));
                    }
                    let pending = match self.pipelines.request(PipelineRecipe::Graphics {
                        shader,
                        state: graphics.clone(),
                    }) {
                        Ok(p) => p,
                        Err(CompileError::Backpressure | CompileError::Budget) => {
                            self.graphics.insert(
                                id,
                                Graphics::Shader {
                                    pending,
                                    state,
                                    graphics,
                                },
                            );
                            return Ok(false);
                        }
                        Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
                    };
                    self.graphics
                        .insert(id, Graphics::Pipeline { pending, state });
                }
                (Graphics::Pipeline { state, .. }, Either::Pipeline(pipeline)) => {
                    let parameters = state::parameters(&state.parameters)?;
                    let push = state::guest_push(&pipeline, Some(&state.parameters))?.into();
                    let prepared = self
                        .residency
                        .prepare_graphics_state(pipeline, parameters)?;
                    self.statistics.prepared_graphics += 1;
                    self.graphics.insert(
                        id,
                        Graphics::Ready {
                            source: Arc::clone(&state.parameters),
                            state: prepared,
                            push,
                        },
                    );
                }
                _ => unreachable!("compilation phase/result match"),
            }
        }
    }
    fn drawing(&mut self, source: &Arc<GuestDrawState>) -> Result<bool, VulkanError> {
        let id = source.identity;
        if let Some(draw) = self.drawings.get(&id) {
            if !Arc::ptr_eq(&draw.source, source) {
                return Err(VulkanError::InvalidResource(
                    "prepared draw identity was reused",
                ));
            }
            return Ok(true);
        }
        let already_prepared = matches!(
            self.graphics
                .get(&(source.parameters.identity, source.program.shader.identity)),
            Some(Graphics::Ready { .. })
        );
        if !self.graphics(source)? {
            return Ok(false);
        }
        let Graphics::Ready { state, push, .. } =
            &self.graphics[&(source.parameters.identity, source.program.shader.identity)]
        else {
            unreachable!()
        };
        let pipeline = Arc::clone(state.pipeline());
        let push = Arc::clone(push);
        let program_key = (source.program.identity, Arc::as_ptr(&pipeline) as usize);
        let program = if let Some(program) = self.programs.get(&program_key) {
            self.statistics.reused_programs += 1;
            Arc::clone(program)
        } else {
            let bindings = self.bindings(&source.program, &pipeline)?;
            let program = Arc::new(self.residency.prepare_program(pipeline, &bindings)?);
            self.programs
                .insert(program_key, Arc::clone(&program), program.retained_bytes());
            program
        };
        let Graphics::Ready { state, .. } =
            &self.graphics[&(source.parameters.identity, source.program.shader.identity)]
        else {
            unreachable!()
        };
        let graphics = state.bind(program)?;
        let targets = self.targets(source)?;
        let mut vertices = Vec::new();
        for v in &source.vertices {
            let buffer = self.residency.buffer_for_deferred_import(
                v.backing.clone(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;
            vertices.push(VertexBuffer {
                binding: v.binding,
                buffer,
                offset: 0,
                size: v.backing.size(),
            });
        }
        let index = source
            .index
            .as_ref()
            .map(|index| {
                let GuestIndexBuffer::Canonical(backing, kind) = index else {
                    return Ok(None);
                };
                let buffer = self.residency.buffer_for_deferred_import(
                    backing.clone(),
                    vk::BufferUsageFlags::INDEX_BUFFER,
                )?;
                Ok::<_, VulkanError>(Some(IndexBuffer {
                    buffer,
                    offset: 0,
                    size: backing.size(),
                    kind: match kind {
                        nixe_gpu::IndexType::Uint8 => vk::IndexType::UINT8_EXT,
                        nixe_gpu::IndexType::Uint16 => vk::IndexType::UINT16,
                        nixe_gpu::IndexType::Uint32 => vk::IndexType::UINT32,
                    },
                }))
            })
            .transpose()?
            .flatten();
        let draw = if let Some(GuestIndexBuffer::Inline(bytes)) = &source.index {
            self.residency.prepare_inline_index_draw(
                graphics,
                targets,
                vertices.into_boxed_slice(),
                Arc::clone(bytes),
            )?
        } else {
            self.residency
                .prepare_draw(graphics, targets, vertices.into_boxed_slice(), index)?
        };
        self.statistics.prepared_draws += 1;
        self.statistics.reused_graphics += u64::from(already_prepared);
        self.drawings.insert(
            id,
            Drawing {
                source: Arc::clone(source),
                draw,
                push,
                recorded: false,
            },
        );
        Ok(true)
    }
    fn computing(&mut self, source: &Arc<GuestProgram>) -> Result<bool, VulkanError> {
        let id = source.identity;
        if !self.computing.contains_key(&id) {
            let pending = match self.request_shader(source, None) {
                Ok(p) => p,
                Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
            };
            self.computing.insert(
                id,
                Computing::Shader {
                    pending,
                    source: Arc::clone(source),
                },
            );
        }
        loop {
            let ready = match &self.computing[&id] {
                Computing::Ready { .. } => return Ok(true),
                Computing::Shader { pending, .. } => pending.poll().map(|r| r.map(Either::Shader)),
                Computing::Pipeline { pending, .. } => {
                    pending.poll().map(|r| r.map(Either::Pipeline))
                }
            };
            let Some(ready) = ready else { return Ok(false) };
            let ready = ready.map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            match (self.computing.remove(&id).unwrap(), ready) {
                (Computing::Shader { source, pending }, Either::Shader(shader)) => {
                    let pending = match self.pipelines.request(PipelineRecipe::Compute(shader)) {
                        Ok(p) => p,
                        Err(CompileError::Backpressure | CompileError::Budget) => {
                            self.computing
                                .insert(id, Computing::Shader { pending, source });
                            return Ok(false);
                        }
                        Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
                    };
                    self.computing
                        .insert(id, Computing::Pipeline { pending, source });
                }
                (Computing::Pipeline { source, .. }, Either::Pipeline(pipeline)) => {
                    let bindings = self.bindings(&source, &pipeline)?;
                    let push = state::guest_push(&pipeline, None)?;
                    let program = self.residency.prepare_program(pipeline, &bindings)?;
                    self.computing.insert(
                        id,
                        Computing::Ready {
                            source,
                            program,
                            push,
                        },
                    );
                }
                _ => unreachable!("compilation phase/result match"),
            }
        }
    }
    fn prepare_command(&mut self, command: &NativeCommand) -> Result<bool, VulkanError> {
        match command {
            NativeCommand::Sequence(commands) => {
                let mut ready = true;
                for c in commands {
                    ready &= self.prepare_command(c)?;
                }
                Ok(ready)
            }
            NativeCommand::Draw { state, .. } => self.drawing(state),
            NativeCommand::Dispatch { program, .. } => self.computing(program),
            NativeCommand::ConvertNv12 { images, .. } => {
                let format = images::format(images[2].description.format());
                if !self.video_pipelines.contains_key(&format) {
                    let pending = match crate::nv12_pipeline(&self.pipelines, format) {
                        Ok(pending) => pending,
                        Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                        Err(error) => {
                            return Err(VulkanError::Compilation(error.to_string().into()));
                        }
                    };
                    self.video_pipelines.insert(format, pending);
                }
                self.video_pipelines[&format]
                    .poll()
                    .map_or(Ok(false), |result| {
                        result
                            .map(|_| true)
                            .map_err(|error| VulkanError::Compilation(error.to_string().into()))
                    })
            }
            NativeCommand::Blit {
                destination,
                sampling: Some(_),
                ..
            } => {
                if self.device_profile().core.shader_int64 == 0 {
                    return Err(VulkanError::InvalidResource(
                        "exact Maxwell 32.32 affine sampling requires enabled shaderInt64",
                    ));
                }
                let format = images::format(destination.description.format());
                if !self.affine_pipelines.contains_key(&format) {
                    let pending = match affine_pipeline(&self.pipelines, format) {
                        Ok(p) => p,
                        Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                        Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
                    };
                    self.affine_pipelines.insert(format, pending);
                }
                match self.affine_pipelines[&format].poll() {
                    None => Ok(false),
                    Some(Ok(_)) => Ok(true),
                    Some(Err(e)) => Err(VulkanError::Compilation(e.to_string().into())),
                }
            }
            _ => Ok(true),
        }
    }
    fn command(&mut self, command: &NativeCommand) -> Result<(), VulkanError> {
        match command {
            NativeCommand::WaitForIdle => self.residency.order_device()?,
            NativeCommand::WriteMemory {
                target,
                offset,
                bytes,
                after_prior_work,
            } => {
                self.residency
                    .write_memory(target.clone(), *offset, bytes, *after_prior_work)?;
            }
            NativeCommand::Sequence(commands) => self.record_commands(commands)?,
            NativeCommand::Draw { state, arguments } => {
                let Drawing {
                    source,
                    draw,
                    push,
                    recorded,
                } = self
                    .drawings
                    .get_mut(&state.identity)
                    .ok_or(VulkanError::InvalidResource("draw was not prepared"))?;
                if !Arc::ptr_eq(source, state) {
                    return Err(VulkanError::InvalidResource(
                        "prepared draw identity was reused",
                    ));
                }
                self.residency
                    .draws_streamed(draw, &[state::arguments(*arguments)], push, None)?;
                self.statistics.draws += 1;
                self.statistics.warm_draws += u64::from(*recorded);
                *recorded = true;
            }
            NativeCommand::Dispatch { program, groups } => {
                let Computing::Ready { program, push, .. } = &self.computing[&program.identity]
                else {
                    return Err(VulkanError::InvalidResource(
                        "dispatch compilation is pending",
                    ));
                };
                self.residency
                    .dispatch(program, Dispatch::Direct(*groups), push)?;
                self.statistics.dispatches += 1;
            }
            NativeCommand::Clear {
                image,
                rectangle,
                value,
            } => {
                self.clear(image, *rectangle, *value)?;
            }
            NativeCommand::Blit {
                source,
                destination,
                source_rectangle,
                destination_rectangle,
                linear,
                resolve,
                sampling,
            } => self.blit(
                source,
                destination,
                *source_rectangle,
                *destination_rectangle,
                *linear,
                *resolve,
                *sampling,
            )?,
            NativeCommand::Barrier(_) => {
                self.residency.flush()?;
            }
            // Canonical CPU epochs, captured TIC/TSC/code identities and native
            // access scopes already make later consumers coherent, including
            // dirty-write flush requests. CPU visibility and explicit completion
            // demand submit pending work; cache maintenance is not an idle wait.
            NativeCommand::Ordering(_) => {}
            NativeCommand::ConvertNv12 { images, transform } => {
                let format = images::format(images[2].description.format());
                let pipeline = self.video_pipelines[&format]
                    .poll()
                    .ok_or(VulkanError::InvalidResource(
                        "video conversion pipeline is pending",
                    ))?
                    .map_err(|error| VulkanError::Compilation(error.to_string().into()))?;
                let sources = [self.video_view(&images[0])?, self.video_view(&images[1])?];
                let destination = self.image(&images[2])?.image;
                let conversion =
                    self.residency
                        .prepare_nv12(pipeline, sources, destination, 0, *transform)?;
                self.residency.convert_nv12(&conversion)?;
            }
            NativeCommand::UploadPlanes {
                images,
                bytes,
                widths,
            } => {
                for index in 0..2 {
                    let image = self.image(&images[index])?.image;
                    self.residency
                        .upload_video_plane(image, widths[index], &bytes[index])?;
                }
            }
            NativeCommand::ImportDecodedPlanes { images, frame } => {
                let planes = [self.image(&images[0])?.image, self.image(&images[1])?.image];
                self.video.import(&mut self.residency, planes, frame)?;
            }
        }
        Ok(())
    }
    fn record_commands(&mut self, commands: &[NativeCommand]) -> Result<(), VulkanError> {
        let mut at = 0;
        let mut streamed_until = 0;
        let mut inputs = None;
        let mut cpu_writes = nixe_memory::CanonicalWriteBatch::default();
        while at < commands.len() {
            if let NativeCommand::WriteMemory {
                target,
                offset,
                bytes,
                after_prior_work: false,
            } = &commands[at]
                && !bytes.is_empty()
                && offset
                    .checked_add(bytes.len() as u64)
                    .is_some_and(|end| end <= target.size())
                && self
                    .residency
                    .inline_write_is_cpu(target, *offset, bytes.len() as u64)
            {
                cpu_writes
                    .stage_with_resolver(target, *offset, bytes, &mut |_, request| {
                        self.residency
                            .materialize_cpu_page(request)
                            .map_err(|error| {
                                nixe_memory::VisibilityCoordinatorError::new(error.to_string())
                            })
                    })
                    .map_err(|error| VulkanError::CanonicalMemory(error.to_string().into()))?;
                at += 1;
                continue;
            }
            if !cpu_writes.is_empty() {
                std::mem::take(&mut cpu_writes)
                    .commit_ordered()
                    .map_err(|error| VulkanError::CanonicalMemory(error.to_string().into()))?;
            }
            if at >= streamed_until && matches!(commands[at], NativeCommand::Draw { .. }) {
                streamed_until = at
                    + commands[at..]
                        .iter()
                        .take_while(|command| matches!(command, NativeCommand::Draw { .. }))
                        .count();
                inputs = if streamed_until - at > 1 {
                    Some(self.residency.capture_draw_inputs(
                        commands[at..streamed_until].iter().map(|command| {
                            let NativeCommand::Draw { state, .. } = command else {
                                unreachable!()
                            };
                            &self.drawings[&state.identity].draw
                        }),
                    )?)
                } else {
                    None
                };
            }
            if let NativeCommand::Draw { state, arguments } = &commands[at]
                && let Some(Drawing {
                    draw,
                    push,
                    recorded,
                    ..
                }) = self.drawings.get_mut(&state.identity)
                && draw.can_group()
            {
                self.draw_arguments.clear();
                self.draw_arguments.push(state::arguments(*arguments));
                let mut end = at + 1;
                while let Some(NativeCommand::Draw {
                    state: next,
                    arguments,
                }) = commands.get(end)
                {
                    if !Arc::ptr_eq(state, next) {
                        break;
                    }
                    let next = state::arguments(*arguments);
                    if !append_instances(self.draw_arguments.last_mut().unwrap(), next) {
                        self.draw_arguments.push(next);
                    }
                    end += 1;
                }
                self.residency
                    .draws_streamed(draw, &self.draw_arguments, push, inputs.as_ref())?;
                self.statistics.draws += (end - at) as u64;
                self.statistics.warm_draws += (end - at) as u64 - u64::from(!*recorded);
                *recorded = true;
                at = end;
            } else {
                self.command(&commands[at])?;
                at += 1;
            }
        }
        if !cpu_writes.is_empty() {
            cpu_writes
                .commit_ordered()
                .map_err(|error| VulkanError::CanonicalMemory(error.to_string().into()))?;
        }
        Ok(())
    }
    fn record_batch(
        &mut self,
        batch: ExecutionBatch<'_, NativeSubmission>,
    ) -> Result<(Option<WorkTicket>, DeviceVisibilityPoint), BackendRuntimeError> {
        let _trace = nixe_gpu::trace::span(
            "backend_record",
            batch.commands.frontend.get(),
            u64::from(batch.commands.segment.get()),
            batch.commands.commands.len() as u64,
        );
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        self.record_commands(&batch.commands.commands)
            .map_err(runtime)?;
        let work = self.residency.device().preceding_work();
        let visibility = self
            .residency
            .device_mut()
            .visibility_point()
            .map_err(runtime)?;
        // A guest delivery boundary is not a host queue dependency. Publish
        // accepted ownership now; visibility, presentation and completion
        // demand flush the owner. Bound batching to keep the GPU supplied even
        // under a continuous producer that never drains its inbox.
        let commands = batch
            .commands
            .commands
            .iter()
            .map(command_count)
            .sum::<usize>();
        let pending = self
            .recording_batch
            .filter(|(previous, _)| *previous == work)
            .map_or(commands, |(_, count)| count.saturating_add(commands));
        // Releases retain their ordered visibility versions. Submit them when
        // a consumer demands progress or the recording reaches its bound.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/fence_manager.h
        if pending >= 256 {
            self.residency.flush().map_err(runtime)?;
            self.recording_batch = None;
        } else {
            self.residency.accept_recording().map_err(runtime)?;
            self.recording_batch = Some((work, pending));
        }
        if self.residency.device().submission.recording.is_none() {
            self.residency.device_mut().end_visibility_scope();
        }
        nixe_gpu::trace::detail_event(
            "backend_accept",
            batch.commands.frontend.get(),
            u64::from(batch.commands.segment.get()),
            work.map_or(0, WorkTicket::tick),
        );
        #[cfg(feature = "performance-counters")]
        {
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::BackendSubmissionNanoseconds,
                started.elapsed().as_nanos() as u64,
            );
            nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::BackendSubmissions, 1);
        }
        self.statistics.batches += 1;
        self.statistics.submits = self.residency.device().submitted_tick();
        Ok((work, visibility))
    }
    fn completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<BackendExecutionCompletion>, VulkanError> {
        if let Some(wakeup) = &self.completion_wakeup {
            wakeup.check()?;
        }
        let _trace = timeout.map(|_| nixe_gpu::trace::span("backend_completion", 1, 0, 0));
        let Some(front) = self.completions.front() else {
            return Ok(None);
        };
        let ready = if front
            .work
            .is_none_or(|work| work.tick() <= self.residency.device().completed_tick())
        {
            true
        } else if let Some(timeout) = timeout {
            let work = front.work.unwrap();
            if work.tick() > self.residency.device().submitted_tick() {
                self.residency.flush()?;
            }
            self.residency.device_mut().wait_for_work(work, timeout)?
        } else {
            self.residency.device_mut().poll_progress()? >= front.work.unwrap().tick()
        };
        if !ready {
            return Ok(None);
        };
        let completed = self.completions.pop_front().unwrap();
        Ok(Some(BackendExecutionCompletion::new(
            completed.frontend,
            completed.token,
            completed.visibility,
        )))
    }
}
enum Either {
    Shader(Arc<ShaderArtifact>),
    Pipeline(Arc<NativePipeline>),
}
impl GpuExecution for NativeGpuExecution {
    type Commands = NativeSubmission;
    type ImageRequest = NativePresentationImage;
    fn synchronize_command_sources(&mut self) -> Result<(), BackendRuntimeError> {
        self.residency.flush().map_err(runtime)?;
        if let Some(work) = self.residency.device().preceding_work() {
            self.residency
                .device_mut()
                .wait_for_work(work, Duration::MAX)
                .map_err(runtime)?;
        }
        Ok(())
    }
    fn external_image_device(&self) -> Option<[u8; 16]> {
        self.video_device
    }
    fn bind_host_wakeup(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.completion_wakeup = Some(completion::CompletionWakeup::new(
            self.residency.device(),
            Arc::clone(&wake),
        ));
        if let Some(host) = &self.host {
            host.client.bind_wakeup(wake);
        }
    }
    fn service_host(&mut self) -> Result<(), BackendRuntimeError> {
        let Some(mut host) = self.host.take() else {
            return Ok(());
        };
        let result = host.service(
            &mut self.residency,
            &mut self.frames,
            &self.pipelines,
            &mut self.descriptors,
            &mut self.samplers,
        );
        if let Err(error) = &result {
            host.client.fail(error.to_string());
        }
        self.host = Some(host);
        result.map_err(runtime)
    }
    fn host_work_pending(&self) -> bool {
        self.host.as_ref().is_some_and(|host| host.pending())
    }
    fn request_completion_wakeup(&mut self) -> Result<bool, BackendRuntimeError> {
        let Some(wakeup) = &self.completion_wakeup else {
            return Ok(false);
        };
        let demand = self.residency.cpu_readback_point().map_err(runtime)?;
        if self
            .residency
            .cpu_readback_submission_pending()
            .map_err(runtime)?
        {
            // Feed demanded downloads even while an older frontend completion
            // is pending, without an extra idle wake before their submission.
            self.residency.flush().map_err(runtime)?;
        }
        let completed = self.residency.device().completed_tick();
        let point = self
            .completions
            .front()
            .and_then(|c| c.work)
            .map(WorkTicket::tick)
            .filter(|&p| p > completed)
            .into_iter()
            .chain(demand)
            .min();
        if let Some(point) = point {
            if point > self.residency.device().submitted_tick() {
                self.residency.flush().map_err(runtime)?;
            }
            return wakeup.arm(point).map_err(runtime);
        }
        // A logical completion can have no GPU ticket, or already be complete.
        // No future timeline event can wake that owner: let it retire the ready
        // prefix instead of sleeping indefinitely waiting for another delivery.
        Ok(false)
    }
    fn prepare(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<bool, BackendRuntimeError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        let _trace = nixe_gpu::trace::span(
            "backend_prepare",
            batch.commands.frontend.get(),
            batch.commands.commands.len() as u64,
            0,
        );
        // Retirement scans are maintenance, not per-batch preparation. The
        // frontend's bounded working sets retain the reusable active states.
        if self.statistics.batches >= self.next_collection {
            self.next_collection = self.statistics.batches.saturating_add(256);
            self.drawings
                .retain(|_, drawing| Arc::strong_count(&drawing.source) > 1);
            self.graphics.retain(|_, graphics| match graphics {
                Graphics::Ready { source, .. } => Arc::strong_count(source) > 1,
                _ => true,
            });
            self.computing.retain(|_, computing| match computing {
                Computing::Ready { source, .. } => Arc::strong_count(source) > 1,
                _ => true,
            });
        }
        let mut ready = true;
        for c in &batch.commands.commands {
            ready &= self.prepare_command(c).map_err(runtime)?;
        }
        #[cfg(feature = "performance-counters")]
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::BackendSubmissionNanoseconds,
            started.elapsed().as_nanos() as u64,
        );
        Ok(ready)
    }
    fn record(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<(), BackendRuntimeError> {
        self.record_batch(batch).map(|_| ())
    }
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        let frontend = batch.commands.frontend;
        let (work, visibility) = self.record_batch(batch)?;
        let token = BackendSubmissionToken::new(
            self.residency.device().submission_instance(),
            self.next,
            1,
        );
        self.next += 1;
        self.completions.push_back(Completion {
            frontend,
            token,
            work,
            visibility,
        });
        Ok(token)
    }
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.completion(None).map_err(runtime)
    }
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.completion(Some(Duration::MAX)).map_err(runtime)
    }
    fn wait_for_completion_timeout(
        &mut self,
        t: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.completion(Some(t)).map_err(runtime)
    }
    fn bind_visibility_requester(
        &mut self,
        r: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError> {
        self.residency
            .visibility()
            .bind_requester(r)
            .map_err(runtime)
    }
    fn make_cpu_visible(
        &mut self,
        r: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError> {
        self.residency.materialize_cpu_page(r).map_err(runtime)
    }
    fn service_cpu_visibility(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Option<Box<[u8]>>, BackendRuntimeError> {
        let result = self.residency.request_cpu_page(request).map_err(runtime)?;
        if result.is_none() {
            // CPU consumption is a submission boundary, not a completion wait.
            // Do not leave its download behind unrelated inbox traffic or the
            // ordinary command batching threshold.
            if self
                .residency
                .cpu_readback_submission_pending()
                .map_err(runtime)?
            {
                self.residency.flush().map_err(runtime)?;
            }
            if let Some(wakeup) = &self.completion_wakeup
                && let Some(point) = self.residency.cpu_readback_point().map_err(runtime)?
            {
                wakeup.arm(point).map_err(runtime)?;
            }
        }
        Ok(result)
    }

    fn acquire_presentable_image(
        &mut self,
        request: NativePresentationImage,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        nixe_gpu::trace::event(
            "frame_work",
            self.statistics.draws,
            self.statistics.dispatches,
            self.statistics.batches,
        );
        nixe_gpu::trace::event(
            "frame_queue",
            self.residency.device().submitted_tick(),
            self.recording_statistics().draw_calls,
            self.residency.statistics().native_readback_bytes,
        );
        self.frames
            .capture(&mut self.residency, request, &mut self.descriptors)
            .map_err(runtime)
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        if let Some(host) = &mut self.host {
            host.shutdown(&mut self.residency).map_err(runtime)?;
        }
        #[cfg(target_os = "linux")]
        log::info!("native Vulkan video: {:?}", self.video.statistics);
        log::info!("native Vulkan residency: {:?}", self.residency.statistics());
        log::info!("native Vulkan recording: {:?}", self.recording_statistics());
        log::info!("native Vulkan preparation: {:?}", self.statistics);
        log::info!(
            "native Vulkan execution: draws={} dispatches={} prepared-draws={} batches={} submits={} descriptors={:?}",
            self.statistics.draws,
            self.statistics.dispatches,
            self.statistics.prepared_draws,
            self.statistics.batches,
            self.statistics.submits,
            self.descriptors.statistics()
        );
        self.residency.flush().map_err(runtime)?;
        while self
            .completion(Some(Duration::MAX))
            .map_err(runtime)?
            .is_some()
        {}
        self.pipelines.checkpoint().map_err(runtime)?;
        self.residency
            .device()
            .validation_monitor()
            .check()
            .map_err(runtime)?;
        Ok(())
    }
}

// The caller requires identical immutable state and read-only shader resources.
// The shader runtime retains the logical guest base instance independently of
// each host draw's firstInstance, so contiguous instances preserve shader IDs.
fn append_instances(previous: &mut Draw, next: Draw) -> bool {
    let (first, count, next_first, next_count) = match (previous, next) {
        (
            Draw::Direct {
                vertices,
                instances,
                first_vertex,
                first_instance,
            },
            Draw::Direct {
                vertices: nv,
                instances: ni,
                first_vertex: nf,
                first_instance: nb,
            },
        ) if *vertices == nv && *first_vertex == nf => (first_instance, instances, nb, ni),
        (
            Draw::Indexed {
                indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            },
            Draw::Indexed {
                indices: nv,
                instances: ni,
                first_index: nf,
                vertex_offset: no,
                first_instance: nb,
            },
        ) if *indices == nv && *first_index == nf && *vertex_offset == no => {
            (first_instance, instances, nb, ni)
        }
        _ => return false,
    };
    if first.checked_add(*count) != Some(next_first) {
        return false;
    }
    let Some(combined) = count.checked_add(next_count) else {
        return false;
    };
    *count = combined;
    true
}

fn command_count(command: &NativeCommand) -> usize {
    match command {
        NativeCommand::Sequence(commands) => commands.iter().map(command_count).sum(),
        NativeCommand::WriteMemory {
            after_prior_work: false,
            ..
        } => 0,
        _ => 1,
    }
}

#[cfg(test)]
mod tests;
