//! Concrete native execution selected at startup. Cold compilation is polled by
//! the owner; prepared references bypass content hashes and descriptor updates.
use crate::residency::Resource;
use crate::{cache::*, *};
use ash::vk;
use nixe_gpu::{
    BackendExecutionCompletion, BackendRuntimeError, BackendSubmissionToken, FrontendSubmissionId,
    GpuExecution, PreparedStateId, ResidentImage,
    native::*,
    shader_identity::{ContentId, ShaderNamespace},
};
use rustc_hash::FxHashMap as HashMap;
use std::{collections::VecDeque, sync::Arc, time::Duration};
mod completion;
mod images;
mod key;
mod state;
use crate::residency::affine_pipeline;
use key::GraphicsKey;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeExecutionStatistics {
    pub draws: u64,
    pub prepared_draws: u64,
    pub reused_vertex_selections: u64,
    pub reused_index_selections: u64,
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
    Ready {
        pipeline: Arc<NativePipeline>,
    },
}
enum Computing {
    Shader {
        pending: Pending<ShaderArtifact>,
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
}
pub struct NativeGpuExecution {
    // Drop/join before residency destroys the timeline semaphore, including
    // failure paths that do not reach explicit teardown.
    completion_wakeup: Option<completion::CompletionWakeup>,
    frames: crate::presentation::Frames,
    host: Option<crate::presentation::Host>,
    residency: VulkanResidency,
    shaders: ShaderCompiler,
    pipelines: Arc<PipelineCompiler>,
    descriptors: DescriptorCache,
    samplers: SamplerCache,
    drawings: HashMap<PreparedStateId, Drawing>,
    graphics: HashMap<GraphicsKey, Graphics>,
    targets: Option<images::Targets>,
    vertex_buffers: Vec<Option<(GuestVertexBuffer, VertexBuffer)>>,
    index_buffer: Option<(
        Arc<nixe_gpu::physical::PhysicalMapping>,
        nixe_gpu::IndexType,
        IndexBuffer,
    )>,
    programs: nixe_gpu::WorkingSet<(PreparedStateId, usize), Arc<PreparedProgram>>,
    next_collection: usize,
    computing: HashMap<PreparedStateId, Computing>,
    affine_pipelines: HashMap<vk::Format, Arc<NativePipeline>>,
    video_pipelines: HashMap<vk::Format, Arc<NativePipeline>>,
    video_inputs: Option<crate::residency::Nv12Inputs>,
    completions: VecDeque<Completion>,
    next: u64,
    draw_counter: u32,
    pub statistics: NativeExecutionStatistics,
}
fn runtime(error: VulkanError) -> BackendRuntimeError {
    match error {
        VulkanError::Memory(cause) => BackendRuntimeError::Memory(cause),
        error => BackendRuntimeError::Backend(error.to_string().into()),
    }
}
impl NativeGpuExecution {
    fn progress_accesses(&mut self) -> Result<(), BackendRuntimeError> {
        self.residency.progress_device_accesses().map_err(runtime)?;
        if self.residency.device_access_submission_pending() {
            self.residency.flush().map_err(runtime)?;
        }
        if let Some(wakeup) = &self.completion_wakeup
            && let Some(point) = self.residency.readback_point()
        {
            wakeup.arm(point).map_err(runtime)?;
        }
        Ok(())
    }
    fn service_host(&mut self) -> Result<(), BackendRuntimeError> {
        if self.host.is_none() {
            self.residency.maintain_budget(false).map_err(runtime)?;
        }
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
        memory: nixe_memory::Machine,
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
        let descriptors = DescriptorCache::new(&device);
        let samplers = SamplerCache::new(&device, configuration.sampler_entries);
        let mut residency = VulkanResidency::new(memory, device, resident_bytes)?;
        residency.enable_sample_conversion(Arc::clone(&pipelines));
        residency
            .memory()
            .enable_cpu_write_collection()
            .map_err(VulkanError::from)?;
        Ok(Self {
            completion_wakeup: None,
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
            vertex_buffers: Vec::new(),
            index_buffer: None,
            programs: nixe_gpu::WorkingSet::new(8192, 32 * 1024 * 1024),
            next_collection: 0,
            computing: HashMap::default(),
            affine_pipelines: HashMap::default(),
            video_pipelines: HashMap::default(),
            video_inputs: None,
            completions: VecDeque::new(),
            next: 1,
            draw_counter: 0,
            statistics: Default::default(),
        })
    }
    pub fn device_profile(&self) -> &DeviceProfile {
        self.residency.device().profile()
    }
    pub fn cache_monitor(&self) -> CacheMonitor {
        CacheMonitor::new([
            ("shader cache", self.shaders.cache_counters()),
            ("shader stages", self.shaders.stage_counters()),
            ("pipeline cache", self.pipelines.cache_counters()),
            ("pipeline libraries", self.pipelines.library_counters()),
            ("descriptor sets", self.residency.descriptor_counters()),
        ])
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
        let id = GraphicsKey::new(state, self.device_profile().dynamic_state.base);
        if !self.graphics.contains_key(&id) {
            let graphics = state::graphics(&state.parameters)?;
            let pending = match self.request_shader(&state.program, Some(state)) {
                Ok(p) => p,
                Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
            };
            self.graphics.insert(
                id.clone(),
                Graphics::Shader {
                    pending,
                    state: Arc::clone(state),
                    graphics,
                },
            );
        }
        let ready = match self.graphics.get_mut(&id).unwrap() {
            Graphics::Ready { .. } => return Ok(true),
            Graphics::Shader { pending, .. } => pending.poll(),
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
                mut shader,
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
                        CompiledStage::from_guest(artifact)
                            .map_err(|error| VulkanError::Compilation(error.to_string().into()))?
                    };
                    let mut stages = shader.stages.to_vec();
                    stages.push(Arc::new(geometry));
                    shader = Arc::new(ShaderArtifact::new(stages.into_boxed_slice()));
                }
                let pipeline = match self.pipelines.request(PipelineRecipe::Graphics {
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
                self.statistics.prepared_graphics += 1;
                self.graphics.insert(id, Graphics::Ready { pipeline });
                Ok(true)
            }
            _ => unreachable!("only shader preparation can be pending"),
        }
    }
    fn drawing(&mut self, source: &Arc<GuestDrawState>) -> Result<bool, VulkanError> {
        let retained = self.residency.constructing.len();
        let result = self.prepare_drawing(source);
        self.residency.constructing.truncate(retained);
        result
    }
    fn prepare_drawing(&mut self, source: &Arc<GuestDrawState>) -> Result<bool, VulkanError> {
        let id = source.identity;
        if let Some(draw) = self.drawings.get(&id) {
            if !Arc::ptr_eq(&draw.source, source) {
                return Err(VulkanError::InvalidResource(
                    "prepared draw identity was reused",
                ));
            }
            if self.residency.draw_current(&draw.draw) {
                return Ok(true);
            }
        }
        self.drawings.remove(&id);
        let graphics_key = GraphicsKey::new(source, self.device_profile().dynamic_state.base);
        let already_prepared = matches!(
            self.graphics.get(&graphics_key),
            Some(Graphics::Ready { .. })
        );
        if !self.graphics(source)? {
            return Ok(false);
        }
        let Graphics::Ready { pipeline, .. } = &self.graphics[&graphics_key] else {
            unreachable!()
        };
        let pipeline = Arc::clone(pipeline);
        let push: Arc<[u8]> = state::guest_push(&pipeline, Some(&source.parameters))?.into();
        let parameters =
            state::parameters(&source.parameters, pipeline.graphics.as_ref().unwrap())?;
        let state = self
            .residency
            .prepare_graphics_state(Arc::clone(&pipeline), parameters)?;
        let program_key = (source.program.identity, Arc::as_ptr(&pipeline) as usize);
        let retained = self.residency.constructing.len();
        // Select the complete binding set before materializing descriptor uses.
        // Buffer growth may retire an earlier selection. Eden likewise updates
        // index/vertex buffers before shader buffers and repeats after deletions.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        loop {
            self.residency.constructing.truncate(retained);
            let targets = self.targets(source)?;
            // Reuse each binding independently of pipeline and dynamic state.
            // Native cache retirement invalidates only the affected selection.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
            let index = if let Some(GuestIndexBuffer::Canonical(backing, kind)) = &source.index {
                let selected = self
                    .index_buffer
                    .as_ref()
                    .filter(|(old, old_kind, selected)| {
                        old == backing
                            && old_kind == kind
                            && self.residency.buffer_binding_current(selected.buffer)
                    });
                let index = if let Some((_, _, selected)) = selected {
                    self.statistics.reused_index_selections += 1;
                    *selected
                } else {
                    let buffer = self.residency.buffer_for_deferred_import(
                        backing.clone(),
                        vk::BufferUsageFlags::INDEX_BUFFER,
                    )?;
                    IndexBuffer {
                        buffer: buffer.buffer,
                        offset: buffer.offset,
                        size: buffer.size,
                        kind: match kind {
                            nixe_gpu::IndexType::Uint8 => vk::IndexType::UINT8_EXT,
                            nixe_gpu::IndexType::Uint16 => vk::IndexType::UINT16,
                            nixe_gpu::IndexType::Uint32 => vk::IndexType::UINT32,
                        },
                    }
                };
                self.index_buffer = Some((Arc::clone(backing), *kind, index));
                self.residency
                    .constructing
                    .push(Resource::Buffer(index.buffer));
                Some(index)
            } else {
                self.index_buffer = None;
                None
            };
            let mut vertices = Vec::with_capacity(source.vertices.len());
            for vertex in source.vertices.iter() {
                let slot = vertex.binding as usize;
                if vertex.binding >= self.device_profile().limits.max_vertex_input_bindings {
                    return Err(VulkanError::InvalidResource(
                        "vertex binding exceeds device limits",
                    ));
                }
                if self.vertex_buffers.len() <= slot {
                    self.vertex_buffers.resize_with(slot + 1, || None);
                }
                let selected = self.vertex_buffers[slot]
                    .as_ref()
                    .filter(|(old, selected)| {
                        old.backing == vertex.backing
                            && self.residency.buffer_binding_current(selected.buffer)
                    });
                let selected = if let Some((_, selected)) = selected {
                    self.statistics.reused_vertex_selections += 1;
                    *selected
                } else {
                    let buffer = self.residency.buffer_for_deferred_import(
                        vertex.backing.clone(),
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    )?;
                    VertexBuffer {
                        binding: vertex.binding,
                        buffer: buffer.buffer,
                        offset: buffer.offset,
                        size: buffer.size,
                    }
                };
                self.vertex_buffers[slot] = Some((vertex.clone(), selected));
                self.residency
                    .constructing
                    .push(Resource::Buffer(selected.buffer));
                vertices.push(selected);
            }
            let cached = self
                .programs
                .get(&program_key)
                .filter(|program| self.residency.program_current(program))
                .cloned();
            let bindings = if cached.is_none() {
                Some(self.bindings(&source.program, &pipeline)?)
            } else {
                None
            };
            if !self.residency.targets_current(&targets)
                || vertices
                    .iter()
                    .any(|v| !self.residency.buffer_binding_current(v.buffer))
                || index.is_some_and(|i| !self.residency.buffer_binding_current(i.buffer))
            {
                continue;
            }
            let program = if let Some(program) = cached {
                self.statistics.reused_programs += 1;
                program
            } else {
                let program = Arc::new(
                    self.residency
                        .prepare_program(Arc::clone(&pipeline), bindings.as_ref().unwrap())?,
                );
                self.programs
                    .insert(program_key, Arc::clone(&program), program.retained_bytes());
                program
            };
            let graphics = state.bind(program)?;
            let draw = if let Some(GuestIndexBuffer::Inline(bytes)) = &source.index {
                self.residency.prepare_inline_index_draw(
                    graphics,
                    targets,
                    vertices.into_boxed_slice(),
                    Arc::clone(bytes),
                )?
            } else {
                self.residency.prepare_draw(
                    graphics,
                    targets,
                    vertices.into_boxed_slice(),
                    index,
                )?
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
            return Ok(true);
        }
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
        if let Computing::Ready { program, .. } = &self.computing[&id] {
            if self.residency.program_current(program) {
                return Ok(true);
            }
            let pipeline = Arc::clone(program.pipeline());
            let bindings = self.bindings(source, &pipeline)?;
            let program = self
                .residency
                .prepare_program(Arc::clone(&pipeline), &bindings)?;
            let Computing::Ready { program: old, .. } = self.computing.get_mut(&id).unwrap() else {
                unreachable!()
            };
            *old = program;
            return Ok(true);
        }
        let ready = match &self.computing[&id] {
            Computing::Ready { .. } => return Ok(true),
            Computing::Shader { pending, .. } => pending.poll(),
        };
        let Some(ready) = ready else { return Ok(false) };
        let ready = ready.map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        match (self.computing.remove(&id).unwrap(), ready) {
            (Computing::Shader { source, pending }, shader) => {
                let pipeline = match self.pipelines.request(PipelineRecipe::Compute(shader)) {
                    Ok(p) => p,
                    Err(CompileError::Backpressure | CompileError::Budget) => {
                        self.computing
                            .insert(id, Computing::Shader { pending, source });
                        return Ok(false);
                    }
                    Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
                };
                let bindings = self.bindings(&source, &pipeline)?;
                let push = state::guest_push(&pipeline, None)?;
                let program = self
                    .residency
                    .prepare_program(Arc::clone(&pipeline), &bindings)?;
                self.computing.insert(
                    id,
                    Computing::Ready {
                        source,
                        program,
                        push,
                    },
                );
                Ok(true)
            }
            _ => unreachable!("only shader preparation can be pending"),
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
            NativeCommand::Draw { state, .. } => {
                if self.drawings.contains_key(&state.identity) {
                    Ok(true)
                } else {
                    self.graphics(state)
                }
            }
            NativeCommand::Dispatch { program, .. } => self.computing(program),
            NativeCommand::ConvertNv12 { destination, .. } => {
                let format = images::format(destination.description.format());
                if !self.video_pipelines.contains_key(&format) {
                    let pending = match crate::residency::nv12_pipeline(&self.pipelines, format) {
                        Ok(pending) => pending,
                        Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                        Err(error) => {
                            return Err(VulkanError::Compilation(error.to_string().into()));
                        }
                    };
                    self.video_pipelines.insert(format, pending);
                }
                Ok(true)
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
                Ok(true)
            }
            _ => Ok(true),
        }
    }
    fn command(&mut self, command: &NativeCommand) -> Result<bool, VulkanError> {
        if matches!(
            command,
            NativeCommand::ConvertNv12 { .. }
                | NativeCommand::Blit {
                    sampling: Some(_),
                    ..
                }
        ) && !self.prepare_command(command)?
        {
            return Ok(false);
        }
        match command {
            NativeCommand::CopyBuffer {
                source,
                destination,
                regions,
            } => {
                self.residency
                    .copy_physical_buffers(source, destination, regions)?;
            }

            NativeCommand::WaitForIdle => self.residency.order_device()?,
            NativeCommand::ReportMemory {
                target,
                offset,
                bytes,
                fence,
            } => {
                self.residency
                    .report_memory(target.clone(), *offset, bytes, *fence)?;
            }
            NativeCommand::Sequence(commands) => {
                if !self.prepare_command(command)? {
                    return Ok(false);
                }
                for command in commands {
                    if !self.command(command)? {
                        return Err(VulkanError::InvalidResource(
                            "ready sequence lost compilation dependency",
                        ));
                    }
                }
            }
            NativeCommand::Draw { state, arguments } => {
                if !self.drawing(state)? {
                    return Ok(false);
                }
                self.flush_work()?;
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
                    .draw(draw, state::arguments(*arguments), push)?;
                self.statistics.draws += 1;
                self.statistics.warm_draws += u64::from(*recorded);
                *recorded = true;
            }
            NativeCommand::Dispatch { program, groups } => {
                if !self.computing(program)? {
                    return Ok(false);
                }
                self.flush_work()?;
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
                self.flush_work()?;
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
            } => {
                if sampling.is_some() {
                    self.flush_work()?;
                }
                self.blit(
                    source,
                    destination,
                    *source_rectangle,
                    *destination_rectangle,
                    *linear,
                    *resolve,
                    *sampling,
                )?;
            }
            // Canonical CPU epochs, captured TIC/TSC/code identities and native
            // access scopes already make later consumers coherent, including
            // dirty-write flush requests. CPU visibility and explicit completion
            // demand submit pending work; cache maintenance is not an idle wait.
            NativeCommand::Ordering(_) => {}
            NativeCommand::ConvertNv12 {
                destination,
                source,
                transform,
            } => {
                let extent = crate::residency::Nv12Inputs::dimensions(source)?;
                if self.video_inputs.as_ref().is_none_or(|inputs| {
                    inputs.extent[0] < extent[0] || inputs.extent[1] < extent[1]
                }) {
                    let mut capacity = extent;
                    if let Some(old) = self.video_inputs.take() {
                        capacity = std::array::from_fn(|axis| capacity[axis].max(old.extent[axis]));
                    }
                    self.video_inputs = Some(crate::residency::Nv12Inputs::new(
                        &mut self.residency,
                        capacity,
                    )?);
                }
                self.video_inputs
                    .as_mut()
                    .unwrap()
                    .upload(&mut self.residency, source)?;
                let output = self.image(destination)?;
                let pipeline = Arc::clone(
                    &self.video_pipelines[&images::format(destination.description.format())],
                );
                let conversion = self.residency.prepare_nv12(
                    pipeline,
                    &mut self.descriptors,
                    self.video_inputs.as_ref().unwrap(),
                    output,
                    0,
                    *transform,
                )?;
                self.residency.convert_nv12(&conversion)?;
            }
        }
        Ok(true)
    }
    fn flush_work(&mut self) -> Result<(), VulkanError> {
        // Match FlushWork: dispatch each eight operations, and submit long
        // streams at 4096 operations without a wall-clock deadline.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
        self.draw_counter += 1;
        if self.draw_counter & 7 == 7 {
            if self.draw_counter >= 4096 {
                self.residency.flush()?;
                self.draw_counter = 0;
            } else {
                self.residency.device_mut().dispatch.dispatch_work()?;
            }
        }
        Ok(())
    }
    fn record_work(&mut self, batch: &NativeWork) -> Result<bool, BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        let _trace = nixe_gpu::trace::span("backend_record", batch.frontend.get(), 0, 1);
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        if !self.command(&batch.command).map_err(runtime)? {
            return Ok(false);
        }
        let work = self.residency.device().preceding_work();
        nixe_gpu::trace::detail_event(
            "backend_accept",
            batch.frontend.get(),
            0,
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
        Ok(true)
    }
    fn completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<BackendExecutionCompletion>, VulkanError> {
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
            self.residency.device_mut().wait_for_work(work, timeout)?
        } else {
            self.residency.device_mut().poll_work(front.work.unwrap())?
                == crate::WorkStatus::Complete
        };
        if !ready {
            return Ok(None);
        };
        self.residency
            .publish_buffer_downloads(self.residency.device().completed_tick())?;
        let completed = self.completions.pop_front().unwrap();
        Ok(Some(BackendExecutionCompletion::new(
            completed.frontend,
            completed.token,
        )))
    }
}
impl Drop for NativeGpuExecution {
    fn drop(&mut self) {
        self.residency.memory().disable_cpu_write_collection();
    }
}
impl GpuExecution for NativeGpuExecution {
    type Commands = NativeWork;
    type ImageRequest = NativePresentationImage;
    fn device_uuid(&self) -> Option<[u8; 16]> {
        Some(self.residency.device().profile().device_uuid)
    }
    fn bind_host_wakeup(
        &mut self,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), BackendRuntimeError> {
        if self.completion_wakeup.is_some() {
            return Err(runtime(VulkanError::InvalidResource(
                "GPU owner wakeup is already bound",
            )));
        }
        self.shaders.bind_wakeup(Arc::clone(&wake));
        self.pipelines.bind_wakeup(Arc::clone(&wake));
        self.completion_wakeup = Some(
            completion::CompletionWakeup::new(self.residency.device(), Arc::clone(&wake))
                .map_err(runtime)?,
        );
        if let Some(host) = &self.host {
            host.client.bind_wakeup(wake);
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), BackendRuntimeError> {
        self.draw_counter = 0;
        self.residency.flush().map_err(runtime)?;
        Ok(())
    }
    fn arm_completion_wakeup(&mut self) -> Result<(), BackendRuntimeError> {
        // Queue publication fences like InnerFence::Queue, without submitting
        // ordinary recording merely to register a status notification.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_fence_manager.cpp
        if self.residency.buffer_publications_pending() {
            self.flush()?;
        }
        let demand = self.residency.readback_point();
        let Some(wakeup) = &self.completion_wakeup else {
            return Ok(());
        };
        let point = self
            .completions
            .front()
            .and_then(|c| c.work)
            .map(WorkTicket::tick)
            .into_iter()
            .chain(demand)
            .chain(
                self.host
                    .as_ref()
                    .and_then(crate::presentation::Host::completion_point),
            )
            .min();
        if let Some(point) = point {
            // Native progress may advance after the owner's logical poll.
            // Registration must also deliver already-completed pending work.
            return wakeup.arm(point).map_err(runtime);
        }
        // The owner polls the ready logical prefix before registration. With
        // no remaining native point, there is no future timeline event to arm.
        Ok(())
    }
    fn consume(&mut self, batch: &Self::Commands) -> Result<bool, BackendRuntimeError> {
        // Retire released frontend states only after the prepared set grows. Warm
        // effects neither age pipelines nor repeatedly scan unchanged caches.
        if self.drawings.len() + self.computing.len() >= self.next_collection {
            self.drawings
                .retain(|_, drawing| Arc::strong_count(&drawing.source) > 1);
            self.computing.retain(|_, computing| match computing {
                Computing::Ready { source, .. } => Arc::strong_count(source) > 1,
                _ => true,
            });
            self.next_collection = self.drawings.len() + self.computing.len() + 256;
        }
        self.record_work(batch)
    }
    fn submit(
        &mut self,
        frontend: FrontendSubmissionId,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        self.residency.commit_buffer_downloads().map_err(runtime)?;
        if self.residency.buffer_publications_pending() {
            self.flush()?;
        }
        let work = self.residency.device().preceding_work();
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
        });
        Ok(token)
    }
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.completion(None).map_err(runtime)
    }
    fn access_memory(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_gpu::OwnerAccess, BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.residency
            .begin_access_memory(ranges, operation)
            .map_err(runtime)
    }
    unsafe fn memory_written(
        &mut self,
        machine: nixe_memory::MachineId,
        extents: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), BackendRuntimeError> {
        self.residency.check_machine(machine).map_err(runtime)?;
        unsafe { self.residency.owner_memory_written(extents) }.map_err(runtime)
    }
    fn queue_device_access(
        &mut self,
        request: nixe_memory::DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.residency
            .queue_device_access(request)
            .map_err(runtime)?;
        self.progress_accesses()
    }
    fn cancel_device_access(
        &mut self,
        request: nixe_memory::AccessRequestId,
    ) -> Result<(), BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.residency
            .cancel_device_access(request)
            .map_err(runtime)?;
        self.progress_accesses()
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.progress_accesses()?;
        self.service_host()
    }
    fn fail_device_accesses(&mut self, cause: Arc<nixe_memory::MemoryError>) {
        self.residency.fail_device_accesses(cause);
    }
    fn retire_physical_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
        extents: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.residency
            .retire_physical_memory(machine, allocation, extents)
            .map_err(runtime)
    }
    fn release_guest_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError> {
        self.residency.check_machine(machine).map_err(runtime)?;
        self.progress()
    }
    fn allocation_pressure(
        &mut self,
        machine: nixe_memory::MachineId,
        _: u64,
    ) -> Result<(), BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.residency.check_machine(machine).map_err(runtime)?;
        self.residency.flush().map_err(runtime)?;
        self.progress_accesses()
    }
    fn acquire_presentable_image(
        &mut self,
        request: NativePresentationImage,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        self.draw_counter = 0;
        self.residency.tick_uniform_cache();
        self.residency.maintain_budget(true).map_err(runtime)?;
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
        nixe_gpu::trace::event(
            "frame_memory",
            self.residency.statistics().cpu_uploaded_bytes,
            self.residency.statistics().cpu_streamed_bytes,
            self.residency.statistics().gpu_copied_bytes,
        );
        nixe_gpu::trace::event(
            "frame_buffers",
            self.residency.statistics().buffer_allocations,
            self.residency.statistics().buffer_merges,
            0,
        );
        self.frames
            .capture(&mut self.residency, request)
            .map_err(runtime)
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        let memory = self.residency.memory().clone();
        let cpu_writes = memory
            .collect_cpu_writes()
            .map_err(|error| runtime(VulkanError::from(error)))?;
        self.residency
            .consume_cpu_writes(cpu_writes)
            .map_err(runtime)?;
        if let Some(host) = &mut self.host {
            host.shutdown(&mut self.residency).map_err(runtime)?;
        }
        #[cfg(target_os = "linux")]
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
        self.residency.drain_buffer_downloads().map_err(runtime)?;
        while self
            .completion(Some(Duration::MAX))
            .map_err(runtime)?
            .is_some()
        {}
        self.pipelines
            .checkpoint()
            .map_err(|error| runtime(VulkanError::Compilation(error.to_string().into())))?;
        self.residency
            .device()
            .validation_monitor()
            .check()
            .map_err(runtime)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
