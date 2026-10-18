//! The command processor owns translation, canonical effects and native scheduling.
//! Worker compilation and device execution remain asynchronous. No lowered-command
//! queue separates guest memory observation from its native resource consumption.
use super::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

type VisibilityResult = Result<Box<[u8]>, VisibilityCoordinatorError>;
struct LocalVisibility {
    identity: Arc<()>,
    resolve: Box<dyn Fn(CpuVisibilityRequest) -> VisibilityResult>,
}
thread_local! {
    static LOCAL_VISIBILITY: RefCell<Option<LocalVisibility>> = const { RefCell::new(None) };
}
struct VisibilityScope;
impl Drop for VisibilityScope {
    fn drop(&mut self) {
        LOCAL_VISIBILITY.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
pub(super) fn local_visibility(
    identity: &Arc<()>,
    request: CpuVisibilityRequest,
) -> Option<VisibilityResult> {
    LOCAL_VISIBILITY.with(|slot| {
        let slot = slot.borrow();
        slot.as_ref()
            .filter(|local| Arc::ptr_eq(&local.identity, identity))
            .map(|local| (local.resolve)(request))
    })
}

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
    identity: Arc<()>,
) -> Result<(), GpuExecutorFailure>
where
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = WorkCommands<L::Work>,
            ImageRequest = <L::Work as MaxwellWork>::ImageRequest,
        >,
{
    let local_backend = Rc::clone(&backend);
    LOCAL_VISIBILITY.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut() = Some(LocalVisibility {
            identity,
            resolve: Box::new(move |request| {
                local_backend.try_borrow_mut()
                    .map_err(|_| VisibilityCoordinatorError::new("recursive canonical visibility while native recording owns the backend"))?
                    .as_mut().ok_or_else(|| VisibilityCoordinatorError::new("canonical GPU read requires an accelerated backend"))?
                    .make_cpu_visible(request).map_err(|error| VisibilityCoordinatorError::new(error.to_string()))
            }),
        });
    });
    let _scope = VisibilityScope;
    let mut frontend = frontend;
    let mut owner = Owner::new(&receiver, backend, budget);
    let result = owner.run(&mut frontend);
    let shutting_down = budget.is_stopped();
    owner.cancel();
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
    visibility: CpuVisibilityQueue,
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
            visibility: CpuVisibilityQueue::default(),
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
            frontend,
            boundary: None,
            detail: detail.into(),
        }
    }
    fn poll(&mut self) -> Result<(), GpuExecutorFailure> {
        let mut backend = self.backend.borrow_mut();
        self.visibility.progress(backend.as_mut());
        let Some(runtime) = backend.as_mut() else {
            return Ok(());
        };
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
        if self.pending.is_empty() && self.conversions.is_empty() && self.visibility.is_empty() {
            return Ok(());
        }
        if let Some(runtime) = self.backend.borrow_mut().as_mut() {
            runtime
                .request_completion_wakeup()
                .map_err(|error| self.failure(FrontendSubmissionId::new(0), error.to_string()))?;
        }
        Ok(())
    }
    fn service_control(&mut self, message: GpuMessage) {
        match message {
            GpuMessage::CpuVisibility { request, reply } => {
                self.visibility
                    .request(self.backend.borrow_mut().as_mut(), request, reply)
            }
            GpuMessage::Wake => self.visibility.progress(self.backend.borrow_mut().as_mut()),
            message => self.deferred.push_back(message),
        }
    }
    fn service_ready_control(&mut self) -> Result<(), GpuExecutorFailure> {
        let mut wake = false;
        while let Ok(message) = self.receiver.try_recv() {
            wake |= matches!(message, GpuMessage::Wake);
            self.service_control(message);
        }
        if wake && self.budget.take_progress_request() {
            self.flush_progress()?;
        }
        if wake || !self.visibility.is_empty() {
            self.poll()?;
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
                let pending = !self.pending.is_empty()
                    || !self.conversions.is_empty()
                    || !self.visibility.is_empty()
                    || self
                        .backend
                        .borrow()
                        .as_ref()
                        .is_some_and(GpuExecution::host_work_pending);
                if !pending {
                    match self.receiver.recv() {
                        Ok(message) => {
                            self.service_control(message);
                            continue;
                        }
                        Err(_) => return Ok(()),
                    }
                }
                match self.receiver.recv_timeout(Duration::from_millis(1)) {
                    Ok(message) => message,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                }
            };
            match message {
                GpuMessage::Wake | GpuMessage::CpuVisibility { .. } => {
                    self.service_control(message)
                }
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
                self.service_control(message);
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
                    .next_segment_with_resolver(
                        nixe_gpu_maxwell::maxwell_gpu_timestamp(work.clock.scheduler_time_ns()),
                        &mut |_, request| {
                            backend
                                .as_mut()
                                .ok_or_else(|| {
                                    VisibilityCoordinatorError::new(
                                        "canonical GPU read requires an accelerated backend",
                                    )
                                })?
                                .make_cpu_visible(request)
                                .map_err(|error| VisibilityCoordinatorError::new(error.to_string()))
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
        match self.receiver.recv_timeout(Duration::from_millis(1)) {
            Ok(message) => {
                self.service_control(message);
                Ok(())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.flush_progress()?;
                Ok(())
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(self.failure(
                FrontendSubmissionId::new(0),
                "GPU inbox disconnected during command consumption",
            )),
        }
    }
    fn cancel(&mut self) {
        self.visibility.cancel();
        for pending in self.conversions.drain(..) {
            let _ = pending.reply.send(Err("GPU owner is shutting down".into()));
        }
        self.pending.clear();
        let deferred = self.deferred.drain(..).chain(self.receiver.try_iter());
        for message in deferred {
            match message {
                GpuMessage::CpuVisibility { reply, .. } => {
                    let _ = reply.send(Err("GPU owner is shutting down".into()));
                }
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
                frontend: id,
                boundary: None,
                detail: format!("guest completion publication failed: {error:?}").into(),
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
