//! The command processor owns translation, canonical effects and native scheduling.
//! Worker compilation and device execution remain asynchronous. No lowered-command
//! queue separates guest memory observation from its native resource consumption.
use super::*;
use std::cell::RefCell;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

struct GpuWork {
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    frontend: FrontendSubmissionId,
    reservation: Option<ReservedTimelinePoint>,
    control: Arc<Mutex<NvHostControl>>,
}
struct PendingWork {
    token: BackendSubmissionToken,
    work: GpuWork,
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
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
{
    let owner = Rc::new(RefCell::new(Owner::new(
        &receiver, backend, budget, frontend,
    )));
    let mut result = Owner::drive(&owner);
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
    if let Some(backend) = owner.borrow().backend.borrow_mut().as_mut() {
        backend.fail_device_accesses(Arc::clone(&cause));
    }
    owner.borrow_mut().cancel(&cause);
    if shutting_down { Ok(()) } else { result }
}

struct Owner<
    'a,
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
> {
    receiver: &'a mpsc::Receiver<GpuMessage>,
    backend: Rc<RefCell<Option<E>>>,
    budget: &'a GpuWorkBudget,
    deferred: VecDeque<GpuMessage>,
    sources: nixe_gpu_maxwell::MaxwellCommandSourceScratch,
    frontend: Rc<RefCell<L>>,
    channels: std::collections::BTreeMap<
        nixe_gpu_maxwell::MaxwellChannelId,
        Option<nixe_gpu_maxwell::MaxwellGpuChannel>,
    >,
    blocked_effects: usize,
    recheck: bool,
    pending: VecDeque<PendingWork>,
    conversions: VecDeque<PendingConversion>,
}
impl<'a, L, E> Owner<'a, L, E>
where
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
{
    fn new(
        receiver: &'a mpsc::Receiver<GpuMessage>,
        backend: Rc<RefCell<Option<E>>>,
        budget: &'a GpuWorkBudget,
        frontend: L,
    ) -> Self {
        Self {
            receiver,
            backend,
            budget,
            deferred: VecDeque::new(),
            sources: Default::default(),
            frontend: Rc::new(RefCell::new(frontend)),
            channels: Default::default(),
            blocked_effects: 0,
            recheck: false,
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
            .progress()
            .map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))?;
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
                if pending.work.frontend != completion.frontend() {
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
                if self.blocked_effects != 0 {
                    self.deferred.push_front(GpuMessage::MemoryAccess(request));
                    return Ok(());
                }
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
            GpuMessage::Wake => {
                self.recheck = true;
                Ok(())
            }
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
        while self.blocked_effects == 0
            && matches!(self.deferred.front(), Some(GpuMessage::MemoryAccess(_)))
        {
            let message = self.deferred.pop_front().unwrap();
            self.service_control(message)?;
        }
        let mut serviced = false;
        let mut wake = false;
        while let Ok(message) = self.receiver.try_recv() {
            serviced = true;
            wake |= matches!(message, GpuMessage::Wake);
            self.service_control(message)?;
        }
        if serviced {
            self.poll()?;
        }
        if wake && self.budget.take_progress_request() {
            self.flush_progress()?;
        }
        Ok(())
    }
    fn drive(owner: &Rc<RefCell<Self>>) -> Result<(), GpuExecutorFailure> {
        // A suspended list keeps its cursor, engine registers and current effect.
        // Poll peers from here, never recursively from a dependency wait. The
        // existing inbox carries worker/timeline wakes; no async runtime or
        // thread per channel is needed. Ingress permits bound the live lists.
        type Task<'a> = Pin<Box<dyn Future<Output = Result<(), GpuExecutorFailure>> + 'a>>;
        let mut running: Vec<(Option<nixe_gpu_maxwell::MaxwellChannelId>, Task<'_>)> = Vec::new();
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if owner.borrow().budget.is_stopped() {
                return Ok(());
            }
            owner.borrow_mut().recheck = false;
            owner.borrow_mut().poll()?;
            let mut index = 0;
            while index < running.len() {
                match running[index].1.as_mut().poll(&mut context) {
                    Poll::Pending => index += 1,
                    Poll::Ready(result) => {
                        drop(running.remove(index));
                        result?;
                    }
                }
            }
            let message = {
                let mut state = owner.borrow_mut();
                if state.budget.take_progress_request() {
                    state.flush_progress()?;
                }
                state.service_ready_control()?;
                let next = state.deferred.iter().position(|message| match message {
                    GpuMessage::Submission(queued) => !running
                        .iter()
                        .any(|(channel, _)| *channel == Some(queued.submission.channel.id())),
                    GpuMessage::MemoryAccess(_) => state.blocked_effects == 0,
                    GpuMessage::CloseChannel(id) => {
                        !running.iter().any(|(channel, _)| *channel == Some(*id))
                    }
                    _ => running.is_empty(),
                });
                next.map(|index| state.deferred.remove(index).unwrap())
            };
            if let Some(message) = message {
                let channel = match &message {
                    GpuMessage::Submission(queued) => Some(queued.submission.channel.id()),
                    _ => None,
                };
                running.push((channel, Box::pin(Self::execute(Rc::clone(owner), message))));
                continue;
            }
            if owner.borrow().recheck {
                continue;
            }
            owner.borrow_mut().flush_progress()?;
            let message = owner.borrow().receiver.recv();
            match message {
                Ok(message) => owner.borrow_mut().service_control(message)?,
                Err(_) if running.is_empty() => return Ok(()),
                Err(_) => {
                    return Err(owner.borrow().failure(
                        FrontendSubmissionId::new(0),
                        "GPU inbox disconnected with suspended commands",
                    ));
                }
            }
        }
    }
    async fn execute(
        owner: Rc<RefCell<Self>>,
        message: GpuMessage,
    ) -> Result<(), GpuExecutorFailure> {
        let frontend = Rc::clone(&owner.borrow().frontend);
        let mut host = CommandHost {
            owner: Rc::clone(&owner),
            clock: Default::default(),
        };
        let conversion = match message {
            GpuMessage::Submission(queued) => return Self::submission(owner, queued).await,
            GpuMessage::CloseChannel(channel) => {
                owner.borrow_mut().channels.remove(&channel);
                frontend.borrow_mut().close_channel(channel);
                return Ok(());
            }
            GpuMessage::PresentImage { request, reply } => {
                let result = frontend
                    .borrow_mut()
                    .lower_presentation(request)
                    .map_err(|error| error.to_string().into_boxed_str())
                    .and_then(|request| {
                        owner
                            .borrow()
                            .backend
                            .borrow_mut()
                            .as_mut()
                            .ok_or_else(|| Box::<str>::from("GPU backend is unavailable"))?
                            .acquire_presentable_image(request)
                            .map_err(|error| error.to_string().into_boxed_str())
                    });
                nixe_memory::trace::event("frame_ready", 0, 0, 0);
                let error = result.as_ref().err().cloned();
                let _ = reply.send(result);
                return error.map_or(Ok(()), |detail| {
                    Err(owner.borrow().failure(FrontendSubmissionId::new(0), detail))
                });
            }
            GpuMessage::ConvertNv12 {
                images,
                transform,
                reply,
            } => (
                frontend
                    .borrow_mut()
                    .lower_nv12_conversion(*images, transform),
                reply,
            ),
            GpuMessage::DecodedPlanes {
                images,
                data,
                reply,
            } => {
                let work = match data {
                    DecodedPlaneData::Device(frame) => frontend
                        .borrow_mut()
                        .lower_device_plane_uploads(*images, frame),
                    DecodedPlaneData::Linear {
                        bytes,
                        visible_widths,
                    } => frontend.borrow_mut().lower_decoded_plane_uploads(
                        *images,
                        bytes,
                        visible_widths,
                    ),
                };
                (work, reply)
            }
            message => return owner.borrow_mut().service_control(message),
        };
        let (work, reply) = conversion;
        let result: Result<_, GpuExecutorFailure> = async {
            let work = work.map_err(|error| {
                owner
                    .borrow()
                    .failure(FrontendSubmissionId::new(0), error.to_string())
            })?;
            let id = work.frontend;
            host.emit_work(&work).await?;
            let token = owner
                .borrow()
                .backend
                .borrow_mut()
                .as_mut()
                .unwrap()
                .submit(id)
                .map_err(|error| owner.borrow().failure(id, error.to_string()))?;
            Ok((id, token))
        }
        .await;
        match result {
            Ok((frontend, token)) => {
                owner.borrow_mut().conversions.push_back(PendingConversion {
                    token,
                    frontend,
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
    async fn submission(
        owner: Rc<RefCell<Self>>,
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
        let mut host = CommandHost {
            owner: Rc::clone(&owner),
            clock: clock.clone(),
        };
        let work = GpuWork {
            descriptor,
            request,
            frontend: id,
            reservation,
            control,
        };
        let (mut sources, mut context, frontend) = {
            let mut state = owner.borrow_mut();
            // Leave the slot in place while the list owns its mutable registers.
            // The scheduler admits at most one list for each channel.
            let context = state
                .channels
                .entry(channel.id())
                .or_insert_with(|| {
                    Some(nixe_gpu_maxwell::MaxwellGpuChannel::from_configuration(
                        channel,
                    ))
                })
                .take()
                .expect("channel has only one active command list");
            (
                std::mem::take(&mut state.sources),
                context,
                Rc::clone(&state.frontend),
            )
        };
        context.update_configuration(channel);
        let result = stream_maxwell_frontend(
            &dispatch,
            &mut context,
            &address_space,
            &frontend,
            &mut sources,
            &mut host,
        )
        .await;
        let mut state = owner.borrow_mut();
        *state.channels.get_mut(&channel.id()).unwrap() = Some(context);
        state.sources.recycle(sources);
        result.map_err(|failure| GpuExecutorFailure {
            memory_cause: None,
            frontend: id,
            detail: failure.to_string().into(),
            boundary: Some((
                descriptor,
                request,
                Box::new(MaxwellFrontendDispatchBoundary {
                    dispatch: Box::new(dispatch),
                    failure,
                }),
            )),
        })?;
        drop(permit);
        let token = state
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .submit(id)
            .map_err(|error| state.failure(id, error.to_string()))?;
        state.pending.push_back(PendingWork { token, work });
        Ok(())
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
fn publish_completion(work: GpuWork) -> Result<(), GpuExecutorFailure> {
    let id = work.frontend;
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

struct CommandHost<
    'a,
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
> {
    owner: Rc<RefCell<Owner<'a, L, E>>>,
    clock: nixe_runtime::VirtualClock,
}
impl<
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
> CommandHost<'_, L, E>
{
    async fn emit_work(
        &mut self,
        command: &nixe_gpu::native::NativeWork,
    ) -> Result<(), GpuExecutorFailure> {
        let mut retained = None;
        let mut blocked = false;
        let result = poll_fn(|_| {
            let mut owner = self.owner.borrow_mut();
            if owner.budget.is_stopped() {
                return Poll::Ready(Err(
                    owner.failure(command.frontend, "GPU owner stopped during preparation")
                ));
            }
            let ready = owner
                .backend
                .borrow_mut()
                .as_mut()
                .unwrap()
                .consume(command)
                .map_err(|error| owner.failure(command.frontend, error.to_string()))?;
            if !ready {
                if retained.is_none() {
                    let mut inputs = Vec::new();
                    command
                        .command
                        .retain_inputs(&mut inputs)
                        .map_err(|error| owner.failure(command.frontend, error.to_string()))?;
                    retained = Some(inputs);
                }
                if !blocked {
                    blocked = true;
                    owner.blocked_effects += 1;
                }
                return Poll::Pending;
            }
            Poll::Ready(Ok(()))
        })
        .await;
        if blocked {
            self.owner.borrow_mut().blocked_effects -= 1;
        }
        result
    }
}
impl<
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
> nixe_gpu_maxwell::MaxwellCommandHost for CommandHost<'_, L, E>
{
    async fn emit(
        &mut self,
        work: nixe_gpu::native::NativeWork,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        self.emit_work(&work).await.map_err(|error| {
            nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
        })
    }
    unsafe fn memory_written(
        &mut self,
        machine: nixe_memory::MachineId,
        extents: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), Arc<MemoryError>> {
        unsafe {
            self.owner
                .borrow()
                .backend
                .borrow_mut()
                .as_mut()
                .unwrap()
                .memory_written(machine, extents)
        }
        .map_err(|error| match error {
            nixe_gpu::BackendRuntimeError::Memory(cause) => cause,
            other => Arc::new(MemoryError::Execution(other.to_string().into())),
        })
    }
    fn access(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, Arc<MemoryError>> {
        let result = self
            .owner
            .borrow()
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .access_memory(ranges, operation)
            .map_err(|error| match error {
                nixe_gpu::BackendRuntimeError::Memory(cause) => cause,
                other => Arc::new(MemoryError::Execution(other.to_string().into())),
            })?;
        let mut pending = match result {
            nixe_gpu::OwnerAccess::Ready(result) => return Ok(result),
            nixe_gpu::OwnerAccess::Pending(pending) => pending,
        };
        // Like Eden's synchronous ReadBlock/FlushRegion, resource capture must
        // finish the actual native producer before reading its bytes. Service
        // control while waiting, but do not recursively execute another channel.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
        self.owner.borrow_mut().blocked_effects += 1;
        let result = (|| {
            loop {
                let mut owner = self.owner.borrow_mut();
                owner
                    .poll()
                    .and_then(|()| owner.flush_progress())
                    .map_err(|error| {
                        error
                            .memory_cause
                            .unwrap_or_else(|| Arc::new(MemoryError::Execution(error.detail)))
                    })?;
                if let Some(result) = pending.poll() {
                    return result;
                }
                if owner.budget.is_stopped() {
                    return Err(Arc::new(MemoryError::Execution(
                        "GPU owner stopped during memory access".into(),
                    )));
                }
                match owner.receiver.recv() {
                    Ok(message) => owner
                        .service_control(message)
                        .map_err(|error| Arc::new(MemoryError::Execution(error.detail)))?,
                    Err(_) => {
                        return Err(Arc::new(MemoryError::Execution(
                            "GPU inbox disconnected during memory access".into(),
                        )));
                    }
                }
            }
        })();
        self.owner.borrow_mut().blocked_effects -= 1;
        result
    }

    fn control(&mut self) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        let mut owner = self.owner.borrow_mut();
        if owner.budget.is_stopped() {
            return Err(nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(
                "GPU owner is shutting down".into(),
            ));
        }
        owner.service_ready_control().map_err(|error| {
            nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
        })
    }
    fn timestamp(&self) -> u64 {
        nixe_gpu_maxwell::maxwell_gpu_timestamp(self.clock.scheduler_time_ns())
    }
}
