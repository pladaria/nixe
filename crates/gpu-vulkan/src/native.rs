//! Concrete native execution selected at startup. Cold compilation is polled by
//! the owner; prepared references bypass content hashes and descriptor updates.
use crate::{cache::*, *};
use ash::vk;
use nixe_gpu::{
    BackendExecutionCompletion, BackendRuntimeError, BackendSubmissionToken,
    BackendVisibilityRequester, ExecutionBatch, FrontendSubmissionId, GpuExecution,
    PreparedStateId, ResidentImage,
    native::*,
    ssa::{self, artifact::ShaderNamespace},
};
use nixe_memory::{CpuVisibilityRequest, DeviceVisibilityPoint};
use rustc_hash::FxHashMap as HashMap;
use std::{collections::VecDeque, sync::Arc, time::Duration};
mod images;
mod state;
use crate::residency::affine_pipeline;
use images::GuestNativeImage;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeExecutionStatistics {
    pub draws: u64,
    pub prepared_draws: u64,
    pub warm_draws: u64,
    pub batches: u64,
    pub submits: u64,
    pub dispatches: u64,
}
enum Drawing {
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
        source: Arc<GuestDrawState>,
        draw: PreparedDraw,
        push: Box<[u8]>,
        recorded: bool,
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
    },
}
struct Completion {
    frontend: FrontendSubmissionId,
    token: BackendSubmissionToken,
    work: WorkTicket,
}
pub struct NativeGpuExecution {
    frames: crate::presentation::Frames,
    host: Option<crate::presentation::Host>,
    residency: VulkanResidency,
    pool: Arc<CompilePool>,
    shaders: ShaderCompiler,
    pipelines: Arc<PipelineCompiler>,
    descriptors: DescriptorCache,
    samplers: SamplerCache,
    drawings: HashMap<PreparedStateId, Drawing>,
    computing: HashMap<PreparedStateId, Computing>,
    images: HashMap<images::ImageKey, Arc<GuestNativeImage>>,
    draw_arguments: Vec<Draw>,
    affine_pipelines: HashMap<vk::Format, Pending<NativePipeline>>,
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
    ) -> Result<(), VulkanError> {
        self.host = Some(crate::presentation::Host::new(
            self.residency.device(),
            client,
        )?);
        Ok(())
    }
    pub fn new(
        device: VulkanDevice,
        namespace: ShaderNamespace,
        configuration: CacheConfiguration,
        resident_bytes: u64,
    ) -> Result<Self, VulkanError> {
        let pool = Arc::new(
            CompilePool::new(configuration.workers, configuration.pending_jobs)
                .map_err(|e| VulkanError::Loader(e.to_string().into()))?,
        );
        let shaders = ShaderCompiler::new(namespace, Arc::clone(&pool), &configuration);
        let pipelines = Arc::new(PipelineCompiler::new(
            &device,
            Arc::clone(&pool),
            &configuration,
        )?);
        let descriptors = DescriptorCache::new(&device, configuration.descriptor_sets);
        let samplers = SamplerCache::new(&device, configuration.sampler_entries);
        let mut residency = VulkanResidency::new(device, resident_bytes)?;
        residency.enable_sample_conversion(Arc::clone(&pipelines));
        Ok(Self {
            frames: crate::presentation::Frames::new(),
            host: None,
            residency,
            pool,
            shaders,
            pipelines,
            descriptors,
            samplers,
            drawings: HashMap::default(),
            computing: HashMap::default(),
            images: HashMap::default(),
            draw_arguments: Vec::with_capacity(128),
            affine_pipelines: HashMap::default(),
            completions: VecDeque::new(),
            next: 1,
            statistics: Default::default(),
        })
    }
    pub fn compilation_pool(&self) -> Arc<CompilePool> {
        Arc::clone(&self.pool)
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
        let (code, options, identity) =
            state::shader_code(program, draw, self.residency.device().shader_options())
                .map_err(CompileError::from)?;
        self.shaders.request(
            identity,
            options,
            move || Ok(code),
            |code: &Arc<GuestShaderCode>| code.functions.iter().collect(),
        )
    }
    fn drawing(&mut self, state: &Arc<GuestDrawState>) -> Result<bool, VulkanError> {
        let id = state.program.identity;
        if !self.drawings.contains_key(&id) {
            let graphics = state::graphics(state)?;
            let pending = match self.request_shader(&state.program, Some(state)) {
                Ok(p) => p,
                Err(CompileError::Backpressure | CompileError::Budget) => return Ok(false),
                Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
            };
            self.drawings.insert(
                id,
                Drawing::Shader {
                    pending,
                    state: Arc::clone(state),
                    graphics,
                },
            );
        }
        loop {
            let ready = match &self.drawings[&id] {
                Drawing::Ready { .. } => return Ok(true),
                Drawing::Shader { pending, .. } => pending.poll().map(|r| r.map(Either::Shader)),
                Drawing::Pipeline { pending, .. } => {
                    pending.poll().map(|r| r.map(Either::Pipeline))
                }
            };
            let Some(ready) = ready else { return Ok(false) };
            let ready = ready.map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            let old = self.drawings.remove(&id).unwrap();
            match (old, ready) {
                (
                    Drawing::Shader {
                        state,
                        graphics,
                        pending,
                    },
                    Either::Shader(mut shader),
                ) => {
                    if state.topology == nixe_gpu::PrimitiveTopology::Quads {
                        let vertex = state
                            .program
                            .shader
                            .functions
                            .iter()
                            .find(|f| f.stage == nixe_gpu::ShaderStage::Vertex)
                            .unwrap();
                        let geometry =
                            ssa::spirv::emit_quad_geometry(vertex, state.provoking_first)
                                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
                        let mut stages = shader.stages.to_vec();
                        stages.push(CompiledStage::new(
                            nixe_gpu::ShaderStage::Geometry,
                            geometry,
                            None,
                            0,
                            None,
                        ));
                        shader = Arc::new(ShaderArtifact::new(stages.into_boxed_slice()));
                    }
                    let pending = match self.pipelines.request(PipelineRecipe::Graphics {
                        shader,
                        state: graphics.clone(),
                    }) {
                        Ok(p) => p,
                        Err(CompileError::Backpressure | CompileError::Budget) => {
                            self.drawings.insert(
                                id,
                                Drawing::Shader {
                                    pending,
                                    state,
                                    graphics,
                                },
                            );
                            return Ok(false);
                        }
                        Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
                    };
                    self.drawings
                        .insert(id, Drawing::Pipeline { pending, state });
                }
                (Drawing::Pipeline { state, .. }, Either::Pipeline(pipeline)) => {
                    let bindings = self.bindings(&state.program, &pipeline)?;
                    let program = self.residency.prepare_program(
                        pipeline,
                        &bindings,
                        &mut self.descriptors,
                    )?;
                    let parameters = state::parameters(&state)?;
                    let graphics = self.residency.prepare_graphics(program, parameters)?;
                    let targets = self.targets(&state)?;
                    let mut vertices = Vec::new();
                    for v in &state.vertices {
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
                    let index = state
                        .index
                        .as_ref()
                        .map(|(backing, kind)| {
                            let buffer = self.residency.buffer_for_deferred_import(
                                backing.clone(),
                                vk::BufferUsageFlags::INDEX_BUFFER,
                            )?;
                            Ok::<_, VulkanError>(IndexBuffer {
                                buffer,
                                offset: 0,
                                size: backing.size(),
                                kind: match kind {
                                    nixe_gpu::IndexType::Uint8 => vk::IndexType::UINT8_EXT,
                                    nixe_gpu::IndexType::Uint16 => vk::IndexType::UINT16,
                                    nixe_gpu::IndexType::Uint32 => vk::IndexType::UINT32,
                                },
                            })
                        })
                        .transpose()?;
                    let draw = self.residency.prepare_draw(
                        graphics,
                        targets,
                        vertices.into_boxed_slice(),
                        index,
                    )?;
                    let push = state::push(&state)?;
                    self.statistics.prepared_draws += 1;
                    self.drawings.insert(
                        id,
                        Drawing::Ready {
                            source: state,
                            draw,
                            push,
                            recorded: false,
                        },
                    );
                }
                _ => unreachable!("compilation phase/result match"),
            }
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
                    let program = self.residency.prepare_program(
                        pipeline,
                        &bindings,
                        &mut self.descriptors,
                    )?;
                    self.computing
                        .insert(id, Computing::Ready { source, program });
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
            NativeCommand::Sequence(commands) => {
                for c in commands {
                    self.command(c)?;
                }
            }
            NativeCommand::Draw { state, arguments } => {
                let Drawing::Ready {
                    source,
                    draw,
                    push,
                    recorded,
                } = self
                    .drawings
                    .get_mut(&state.program.identity)
                    .ok_or(VulkanError::InvalidResource("draw was not prepared"))?
                else {
                    return Err(VulkanError::InvalidResource("draw compilation is pending"));
                };
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
                let Computing::Ready { program, .. } = &self.computing[&program.identity] else {
                    return Err(VulkanError::InvalidResource(
                        "dispatch compilation is pending",
                    ));
                };
                self.residency
                    .dispatch(program, Dispatch::Direct(*groups), &[])?;
                self.statistics.dispatches += 1;
            }
            NativeCommand::Clear {
                image,
                rectangle,
                value,
            } => self.clear(image, *rectangle, *value)?,
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
            NativeCommand::ConvertNv12 { .. } => {
                return Err(VulkanError::InvalidResource(
                    "NV12 producer preparation requires native conversion state",
                ));
            }
            NativeCommand::UploadPlanes { .. } => {
                return Err(VulkanError::InvalidResource(
                    "decoded plane upload requires native plane preparation",
                ));
            }
        }
        Ok(())
    }
    fn record_commands(&mut self, commands: &[NativeCommand]) -> Result<(), VulkanError> {
        let mut at = 0;
        while at < commands.len() {
            if let NativeCommand::Draw { state, arguments } = &commands[at]
                && let Some(Drawing::Ready {
                    draw,
                    push,
                    recorded,
                    ..
                }) = self.drawings.get_mut(&state.program.identity)
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
                    self.draw_arguments.push(state::arguments(*arguments));
                    end += 1;
                }
                self.residency.draws(draw, &self.draw_arguments, push)?;
                self.statistics.draws += (end - at) as u64;
                self.statistics.warm_draws += (end - at) as u64 - u64::from(!*recorded);
                *recorded = true;
                at = end;
            } else {
                self.command(&commands[at])?;
                at += 1;
            }
        }
        Ok(())
    }
    fn completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<BackendExecutionCompletion>, VulkanError> {
        let Some(front) = self.completions.front() else {
            return Ok(None);
        };
        let ready = if let Some(timeout) = timeout {
            if front.work.tick() > self.residency.device().queue_usage().submitted_tick {
                self.residency.flush()?;
            }
            self.residency
                .device_mut()
                .wait_for_work(front.work, timeout)?
        } else {
            self.residency.device_mut().poll_progress()? >= front.work.tick()
        };
        if !ready {
            return Ok(None);
        };
        let completed = self.completions.pop_front().unwrap();
        Ok(Some(BackendExecutionCompletion::new(
            completed.frontend,
            completed.token,
            DeviceVisibilityPoint::new(completed.work.tick()),
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
    fn bind_host_wakeup(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
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
    fn prepare(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<bool, BackendRuntimeError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        self.drawings.retain(|_, drawing| match drawing {
            Drawing::Ready { source, .. } => Arc::strong_count(source) > 1,
            _ => true,
        });
        self.computing.retain(|_, computing| match computing {
            Computing::Ready { source, .. } => Arc::strong_count(source) > 1,
            _ => true,
        });
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
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        self.record_commands(&batch.commands.commands)
            .map_err(runtime)?;
        let work = self.residency.device_mut().work_ticket().map_err(runtime)?;
        if batch.commands.final_segment {
            self.residency.flush().map_err(runtime)?;
        } else {
            self.residency.accept_recording().map_err(runtime)?;
        }
        let token = BackendSubmissionToken::new(work.submission().instance(), self.next, 1);
        self.next += 1;
        self.completions.push_back(Completion {
            frontend: batch.commands.frontend,
            token,
            work,
        });
        #[cfg(feature = "performance-counters")]
        {
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::BackendSubmissionNanoseconds,
                started.elapsed().as_nanos() as u64,
            );
            nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::BackendSubmissions, 1);
        }
        self.statistics.batches += 1;
        self.statistics.submits = self.residency.device().queue_usage().submitted_tick;
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
    fn acquire_presentable_image(
        &mut self,
        request: NativePresentationImage,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        self.frames
            .capture(&mut self.residency, request, &mut self.descriptors)
            .map_err(runtime)
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        if let Some(host) = &mut self.host {
            host.shutdown(&mut self.residency).map_err(runtime)?;
        }
        log::info!("native Vulkan residency: {:?}", self.residency.statistics());
        log::info!("native Vulkan recording: {:?}", self.recording_statistics());
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

#[cfg(test)]
mod tests;
