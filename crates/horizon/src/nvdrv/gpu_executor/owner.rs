//! The command processor owns translation, canonical effects and native scheduling.
//! Worker compilation and device execution remain asynchronous. No lowered-command
//! queue separates guest memory observation from its native resource consumption.
use super::*;
use std::cell::RefCell;
use std::rc::Rc;

struct GpuWork<W: MaxwellWork> {
    clock: nixe_runtime::VirtualClock,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    execution: MaxwellBackendExecution<W>,
    reservation: Option<ReservedTimelinePoint>,
    control: Arc<Mutex<NvHostControl>>,
    _permit: GpuFrontendPermit,
}
struct PendingWork<W: MaxwellWork> {
    token: BackendSubmissionToken,
    work: GpuWork<W>,
}
struct PendingConversion {
    token: BackendSubmissionToken,
    frontend: FrontendSubmissionId,
    reply: mpsc::SyncSender<Result<(), Box<str>>>,
}

pub(super) fn run<L, E>(
    receiver: mpsc::Receiver<GpuMessage>,
    backend: Rc<RefCell<Option<E>>>,
    frontend: L,
    budget: &GpuWorkBudget,
    endpoint: &GpuMemoryEndpoint,
) -> Result<(), GpuExecutorFailure>
where
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = WorkCommands<L::Work>,
            ImageRequest = <L::Work as MaxwellWork>::ImageRequest,
        >,
{
    let mut frontend = frontend;
    let mut owner = Owner::new(&receiver, backend, budget);
    let mut result = owner.run(&mut frontend);
    let shutting_down = budget.is_stopped();
    let cause = result
        .as_ref()
        .err()
        .and_then(|error| error.memory_cause.clone())
        .unwrap_or_else(|| {
            Arc::new(MemoryError::Execution(result.as_ref().err().map_or_else(
                || "GPU owner stopped".into(),
                |error| error.to_string().into(),
            )))
        });
    if let Err(error) = &mut result {
        error.memory_cause = Some(Arc::clone(&cause));
    }
    endpoint.stop(Arc::clone(&cause));
    if let Some(backend) = owner.backend.borrow_mut().as_mut() {
        backend.fail_device_accesses(Arc::clone(&cause));
    }
    owner.cancel(&cause);
    if shutting_down { Ok(()) } else { result }
}

struct Owner<
    'a,
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
> {
    receiver: &'a mpsc::Receiver<GpuMessage>,
    backend: Rc<RefCell<Option<E>>>,
    budget: &'a GpuWorkBudget,
    deferred: VecDeque<GpuMessage>,
    sources: nixe_gpu_maxwell::MaxwellCommandSourceScratch,
    pending: VecDeque<PendingWork<W>>,
    conversions: VecDeque<PendingConversion>,
}
impl<'a, W, E> Owner<'a, W, E>
where
    W: MaxwellWork,
    E: GpuExecution<Commands = WorkCommands<W>, ImageRequest = W::ImageRequest>,
{
    fn new(
        receiver: &'a mpsc::Receiver<GpuMessage>,
        backend: Rc<RefCell<Option<E>>>,
        budget: &'a GpuWorkBudget,
    ) -> Self {
        Self {
            receiver,
            backend,
            budget,
            deferred: VecDeque::new(),
            sources: Default::default(),
            pending: VecDeque::new(),
            conversions: VecDeque::new(),
        }
    }
    fn failure(
        &self,
        frontend: FrontendSubmissionId,
        detail: impl Into<Box<str>>,
    ) -> GpuExecutorFailure {
        GpuExecutorFailure {
            memory_cause: None,
            frontend,
            boundary: None,
            detail: detail.into(),
        }
    }
    fn poll(&mut self) -> Result<(), GpuExecutorFailure> {
        let mut backend = self.backend.borrow_mut();
        let Some(runtime) = backend.as_mut() else {
            return Ok(());
        };
        runtime
            .progress_device_accesses()
            .map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))?;
        runtime
            .service_host()
            .map_err(|error| self.failure(FrontendSubmissionId::new(0), error.to_string()))?;
        while let Some(completion) = runtime
            .poll_completion()
            .map_err(|error| self.failure(FrontendSubmissionId::new(0), error.to_string()))?
        {
            if let Some(index) = self
                .conversions
                .iter()
                .position(|pending| pending.token == completion.submission())
            {
                let pending = self.conversions.remove(index).unwrap();
                if pending.frontend != completion.frontend() {
                    return Err(self.failure(
                        completion.frontend(),
                        "conversion completion has a different frontend",
                    ));
                }
                let _ = pending.reply.send(Ok(()));
            } else {
                let index = self
                    .pending
                    .iter()
                    .position(|pending| pending.token == completion.submission())
                    .ok_or_else(|| {
                        self.failure(completion.frontend(), "GPU completed an unknown delivery")
                    })?;
                let pending = self.pending.remove(index).unwrap();
                if pending.work.execution.frontend() != completion.frontend() {
                    return Err(
                        self.failure(completion.frontend(), "completion has a different frontend")
                    );
                }
                publish_completion(pending.work)?;
            }
        }
        Ok(())
    }
    fn flush_progress(&mut self) -> Result<(), GpuExecutorFailure> {
        // Each caller has polled its ready prefix. Registration also delivers
        // an immediate wake if native completion raced with that poll.
        self.backend
            .borrow_mut()
            .as_mut()
            .map_or(Ok(()), |runtime| {
                runtime
                    .request_completion_wakeup()
                    .map_err(|error| self.failure(FrontendSubmissionId::new(0), error.to_string()))
            })
    }
    fn runtime_failure(
        &self,
        frontend: FrontendSubmissionId,
        error: nixe_gpu::BackendRuntimeError,
    ) -> GpuExecutorFailure {
        let mut failure = self.failure(frontend, error.to_string());
        if let nixe_gpu::BackendRuntimeError::Memory(cause) = error {
            failure.memory_cause = Some(cause);
        }
        failure
    }
    fn service_control(&mut self, message: GpuMessage) -> Result<(), GpuExecutorFailure> {
        let result = match message {
            GpuMessage::MemoryAccess(request) => {
                if let Some(backend) = self.backend.borrow_mut().as_mut() {
                    backend.queue_device_access(request)
                } else {
                    request.finish(Err(Arc::new(MemoryError::Execution(
                        "GPU backend is unavailable".into(),
                    ))));
                    Ok(())
                }
            }
            GpuMessage::CancelMemory(request) => self
                .backend
                .borrow_mut()
                .as_mut()
                .map_or(Ok(()), |backend| backend.cancel_device_access(request)),
            GpuMessage::GuestMemoryReleased {
                machine,
                allocation,
            } => self
                .backend
                .borrow_mut()
                .as_mut()
                .map_or(Ok(()), |backend| {
                    backend.release_guest_memory(machine, allocation)
                }),
            GpuMessage::PhysicalRetired {
                machine,
                allocation,
                extents,
            } => self
                .backend
                .borrow_mut()
                .as_mut()
                .map_or(Ok(()), |backend| {
                    backend.retire_physical_memory(machine, allocation, &extents)
                }),
            GpuMessage::AllocationPressure { machine, bytes } => self
                .backend
                .borrow_mut()
                .as_mut()
                .map_or(Ok(()), |backend| {
                    backend.allocation_pressure(machine, bytes)
                }),
            GpuMessage::Wake => self
                .backend
                .borrow_mut()
                .as_mut()
                .map_or(Ok(()), GpuExecution::progress_device_accesses),
            message => {
                self.deferred.push_back(message);
                Ok(())
            }
        };
        result.map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))
    }
    fn service_ready_control(&mut self) -> Result<(), GpuExecutorFailure> {
        // Eden ticks CPU coherency requests after PrepareDraw has consumed its
        // resources, including when native compilation had to wait. A request
        // must not resolve against authority missing that pending effect.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
        while matches!(self.deferred.front(), Some(GpuMessage::MemoryAccess(_))) {
            let message = self.deferred.pop_front().unwrap();
            self.service_control(message)?;
        }
        let mut wake = false;
        while let Ok(message) = self.receiver.try_recv() {
            wake |= matches!(message, GpuMessage::Wake);
            self.service_control(message)?;
        }
        self.poll()?;
        if wake && self.budget.take_progress_request() {
            self.flush_progress()?;
        }
        Ok(())
    }
    fn run<L: MaxwellFrontend<Work = W>>(
        &mut self,
        frontend: &mut L,
    ) -> Result<(), GpuExecutorFailure> {
        // Eden's GPU thread consumes commands and resources in one ordered pass;
        // its Vulkan scheduler independently dispatches native recording work.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/gpu_thread.cpp
        let mut channels = std::collections::BTreeMap::new();
        loop {
            if self.budget.is_stopped() {
                return Ok(());
            }
            self.poll()?;
            if self.budget.take_progress_request() {
                self.flush_progress()?;
            }
            let message = self
                .deferred
                .pop_front()
                .or_else(|| self.receiver.try_recv().ok());
            let message = if let Some(message) = message {
                message
            } else {
                self.flush_progress()?;
                match self.receiver.recv() {
                    Ok(message) => message,
                    Err(_) => return Ok(()),
                }
            };
            match message {
                GpuMessage::Wake
                | GpuMessage::MemoryAccess(_)
                | GpuMessage::CancelMemory(_)
                | GpuMessage::GuestMemoryReleased { .. }
                | GpuMessage::PhysicalRetired { .. }
                | GpuMessage::AllocationPressure { .. } => self.service_control(message)?,
                message => self.execute(frontend, &mut channels, message)?,
            }
        }
    }
    fn execute<L: MaxwellFrontend<Work = W>>(
        &mut self,
        frontend: &mut L,
        channels: &mut std::collections::BTreeMap<
            nixe_gpu_maxwell::MaxwellChannelId,
            nixe_gpu_maxwell::MaxwellGpuChannel,
        >,
        message: GpuMessage,
    ) -> Result<(), GpuExecutorFailure> {
        match message {
            GpuMessage::Submission(queued) => self.submission(frontend, channels, queued),
            GpuMessage::CloseChannel(channel) => {
                channels.remove(&channel);
                Ok(())
            }
            GpuMessage::PresentImage { request, reply } => {
                let result = frontend
                    .lower_presentation(request)
                    .map_err(|error| error.to_string().into_boxed_str())
                    .and_then(|request| {
                        self.backend
                            .borrow_mut()
                            .as_mut()
                            .ok_or_else(|| Box::<str>::from("GPU backend is unavailable"))?
                            .acquire_presentable_image(request)
                            .map_err(|error| error.to_string().into_boxed_str())
                    });
                nixe_memory::trace::event("frame_ready", 0, 0, 0);
                let error = result.as_ref().err().cloned();
                let _ = reply.send(result);
                error.map_or(Ok(()), |detail| {
                    Err(self.failure(FrontendSubmissionId::new(0), detail))
                })
            }
            GpuMessage::ConvertNv12 {
                images,
                transform,
                reply,
            } => {
                let work = frontend.lower_nv12_conversion(*images, transform);
                self.conversion(work, reply)
            }
            GpuMessage::DecodedPlanes {
                images,
                data,
                reply,
            } => {
                let work = match data {
                    DecodedPlaneData::Device(frame) => {
                        frontend.lower_device_plane_uploads(*images, frame)
                    }
                    DecodedPlaneData::Linear {
                        bytes,
                        visible_widths,
                    } => frontend.lower_decoded_plane_uploads(*images, bytes, visible_widths),
                };
                self.conversion(work, reply)
            }
            message => {
                self.service_control(message)?;
                Ok(())
            }
        }
    }
    fn conversion(
        &mut self,
        work: Result<W, nixe_gpu_maxwell::MaxwellLoweringError>,
        reply: mpsc::SyncSender<Result<(), Box<str>>>,
    ) -> Result<(), GpuExecutorFailure> {
        let work = match work {
            Ok(work) => work,
            Err(error) => {
                let detail = error.to_string().into_boxed_str();
                let _ = reply.send(Err(detail.clone()));
                return Err(self.failure(FrontendSubmissionId::new(0), detail));
            }
        };
        let id = work.frontend();
        let mut batch = W::Batch::default();
        batch.work(&work);
        let segment = batch
            .finish(id, nixe_gpu::FrontendSubmissionSegment::FIRST, true)
            .map_err(|error| self.failure(id, error.to_string()))?;
        if let Err(error) = self.prepare(&segment, id) {
            let _ = reply.send(Err(error.detail.clone()));
            return Err(error);
        }
        let result = self
            .backend
            .borrow_mut()
            .as_mut()
            .ok_or_else(|| self.failure(id, "conversion requires an accelerated backend"))?
            .submit(segment.execution_batch())
            .map_err(|error| self.failure(id, error.to_string()));
        match result {
            Ok(token) => {
                self.conversions.push_back(PendingConversion {
                    token,
                    frontend: id,
                    reply,
                });
                Ok(())
            }
            Err(error) => {
                let _ = reply.send(Err(error.detail.clone()));
                Err(error)
            }
        }
    }
    fn submission<L: MaxwellFrontend<Work = W>>(
        &mut self,
        frontend: &mut L,
        channels: &mut std::collections::BTreeMap<
            nixe_gpu_maxwell::MaxwellChannelId,
            nixe_gpu_maxwell::MaxwellGpuChannel,
        >,
        queued: GpuQueuedSubmission,
    ) -> Result<(), GpuExecutorFailure> {
        let GpuQueuedSubmission { submission, clock } = queued;
        let GpuSubmission {
            descriptor,
            request,
            dispatch,
            channel,
            address_space,
            reservation,
            control,
            permit,
        } = submission;
        let id = dispatch.scheduled().frontend();
        let mut work = GpuWork {
            clock,
            descriptor,
            request,
            execution: MaxwellBackendExecution::begin(
                id,
                reservation.as_ref().map(ReservedTimelinePoint::point),
            ),
            reservation,
            control,
            _permit: permit,
        };
        let _trace = nixe_memory::trace::span("gpu_command_stream", id.get(), 0, 0);
        let mut sources = std::mem::take(&mut self.sources);
        let source_backend = Rc::clone(&self.backend);
        let result = {
            let context = channels.entry(channel.id()).or_insert_with(|| {
                nixe_gpu_maxwell::MaxwellGpuChannel::from_configuration(channel)
            });
            context.update_configuration(channel);
            stream_maxwell_frontend(
                &dispatch,
                context,
                &address_space,
                frontend,
                &mut sources,
                &mut |ranges, operation| {
                    let mut backend = source_backend.borrow_mut();
                    let backend = backend.as_mut().ok_or_else(|| {
                        Arc::new(MemoryError::Invalid(
                            "graphics memory access requires an accelerated backend",
                        ))
                    })?;
                    backend
                        .access_memory(ranges, operation)
                        .map_err(|error| match error {
                            nixe_gpu::BackendRuntimeError::Memory(cause) => cause,
                            other => Arc::new(MemoryError::Execution(other.to_string().into())),
                        })
                },
                &mut || {
                    source_backend
                        .borrow_mut()
                        .as_mut()
                        .ok_or_else(|| {
                            Box::<str>::from("command fetch requires an accelerated backend")
                        })?
                        .synchronize_command_sources()
                        .map_err(|error| error.to_string().into())
                },
                &mut |steps| {
                    work.execution.append_steps(steps);
                    self.consume(&mut work)
                        .map(|_| ())
                        .map_err(|error| error.to_string().into_boxed_str())
                },
            )
        };
        self.sources = sources;
        let tail = result.map_err(|failure| GpuExecutorFailure {
            memory_cause: None,
            frontend: id,
            detail: failure.to_string().into(),
            boundary: Some((
                descriptor,
                request,
                Box::new(MaxwellFrontendDispatchBoundary::Frontend {
                    dispatch: Box::new(dispatch),
                    failure,
                }),
            )),
        })?;
        work.execution.append_steps(tail.into_steps().into_vec());
        work.execution.seal();
        let token = self.consume(&mut work)?;
        if let Some(token) = token {
            self.pending.push_back(PendingWork { token, work });
        } else {
            publish_completion(work)?;
        }
        Ok(())
    }
    fn consume(
        &mut self,
        work: &mut GpuWork<W>,
    ) -> Result<Option<BackendSubmissionToken>, GpuExecutorFailure> {
        let id = work.execution.frontend();
        let mut completion = None;
        loop {
            if self.budget.is_stopped() {
                return Err(self.failure(id, "GPU owner is shutting down"));
            }
            let segment = {
                let mut backend = self.backend.borrow_mut();
                work.execution
                    .next_segment_with_memory(
                        nixe_gpu_maxwell::maxwell_gpu_timestamp(work.clock.scheduler_time_ns()),
                        &mut |ranges, operation| {
                            backend
                                .as_mut()
                                .ok_or_else(|| {
                                    Arc::new(MemoryError::Invalid(
                                        "graphics memory access requires an accelerated backend",
                                    ))
                                })?
                                .access_memory(ranges, operation)
                                .map_err(|error| match error {
                                    nixe_gpu::BackendRuntimeError::Memory(cause) => cause,
                                    other => {
                                        Arc::new(MemoryError::Execution(other.to_string().into()))
                                    }
                                })
                        },
                    )
                    .map_err(|error| self.failure(id, error.to_string()))?
            };
            if let Some(segment) = segment {
                self.prepare(&segment, id)?;
                {
                    let mut backend = self.backend.borrow_mut();
                    let runtime = backend.as_mut().ok_or_else(|| {
                        self.failure(id, "submission requires an accelerated GPU backend")
                    })?;
                    if segment.is_final_segment() {
                        let token = runtime
                            .submit(segment.execution_batch())
                            .map_err(|error| self.failure(id, error.to_string()))?;
                        assert!(
                            completion.replace(token).is_none(),
                            "one completion per delivery"
                        );
                    } else {
                        runtime
                            .record(segment.execution_batch())
                            .map_err(|error| self.failure(id, error.to_string()))?;
                    }
                }
                work.execution.recycle_segment(segment);
                work.execution.accept_segment();
                work.execution.retire_consumed_steps();
                if completion.is_none() {
                    self.service_ready_control()?;
                }
                continue;
            }
            work.execution.retire_consumed_steps();
            if completion.is_none() {
                self.service_ready_control()?;
            }
            if work.execution.is_finished() || work.execution.waiting_for_input() {
                return Ok(completion);
            }
            return Err(self.failure(
                id,
                "command processor retained an unsupported blocking effect",
            ));
        }
    }
    fn prepare(
        &mut self,
        segment: &<W::Batch as MaxwellWorkBatch<W>>::Segment,
        id: FrontendSubmissionId,
    ) -> Result<(), GpuExecutorFailure> {
        loop {
            if self.budget.is_stopped() {
                return Err(self.failure(id, "GPU owner stopped during host preparation"));
            }
            let ready = self
                .backend
                .borrow_mut()
                .as_mut()
                .ok_or_else(|| self.failure(id, "submission requires an accelerated GPU backend"))?
                .prepare(segment.execution_batch())
                .map_err(|error| self.failure(id, error.to_string()))?;
            if ready {
                return Ok(());
            }
            self.pump()?;
        }
    }
    fn pump(&mut self) -> Result<(), GpuExecutorFailure> {
        self.poll()?;
        if self.budget.take_progress_request() {
            self.flush_progress()?;
        }
        // Compiler completion has its own wakeup. Waiting for it does not
        // require submission or guest-visible completion of unrelated work.
        match self.receiver.recv() {
            Ok(message @ GpuMessage::MemoryAccess(_)) => {
                // Keep the existing between-effect control priority and FIFO
                // order without servicing guest memory inside preparation.
                let index = self
                    .deferred
                    .iter()
                    .position(|pending| !matches!(pending, GpuMessage::MemoryAccess(_)))
                    .unwrap_or(self.deferred.len());
                self.deferred.insert(index, message);
                Ok(())
            }
            Ok(message) => {
                self.service_control(message)?;
                Ok(())
            }
            Err(_) => Err(self.failure(
                FrontendSubmissionId::new(0),
                "GPU inbox disconnected during command consumption",
            )),
        }
    }
    fn cancel(&mut self, cause: &Arc<MemoryError>) {
        for pending in self.conversions.drain(..) {
            let _ = pending.reply.send(Err("GPU owner is shutting down".into()));
        }
        self.pending.clear();
        let deferred = self.deferred.drain(..).chain(self.receiver.try_iter());
        for message in deferred {
            match message {
                GpuMessage::MemoryAccess(request) => request.finish(Err(Arc::clone(cause))),
                GpuMessage::ConvertNv12 { reply, .. } | GpuMessage::DecodedPlanes { reply, .. } => {
                    let _ = reply.send(Err("GPU owner is shutting down".into()));
                }
                GpuMessage::PresentImage { reply, .. } => {
                    let _ = reply.send(Err("GPU owner is shutting down".into()));
                }
                _ => {}
            }
        }
    }
}
fn publish_completion<W: MaxwellWork>(work: GpuWork<W>) -> Result<(), GpuExecutorFailure> {
    let id = work.execution.frontend();
    if work.execution.completion() != work.reservation.as_ref().map(ReservedTimelinePoint::point) {
        return Err(GpuExecutorFailure {
            memory_cause: None,
            frontend: id,
            boundary: None,
            detail: "GPU completion does not match its reserved timeline".into(),
        });
    }
    if let Some(reservation) = work.reservation {
        work.control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .complete_channel_submission(work.descriptor, work.request, &reservation)
            .map_err(|error| GpuExecutorFailure {
                memory_cause: None,
                frontend: id,
                boundary: None,
                detail: format!("guest completion publication failed: {error:?}").into(),
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
