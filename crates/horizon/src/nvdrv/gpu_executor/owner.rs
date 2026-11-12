//! Preparation owner for ordered engine methods, resources and native scheduling.
use super::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
struct SyncpointEffect {
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    reservation: ReservedTimelinePoint,
    control: Arc<Mutex<NvHostControl>>,
}
pub(super) fn run<L, E>(
    mut receiver: nixe_gpu::ordered::Receiver<GpuMessage>,
    backend: Rc<RefCell<Option<E>>>,
    frontend: L,
    budget: &GpuWorkBudget,
    endpoint: &GpuMemoryEndpoint,
    control: Arc<Mutex<NvHostControl>>,
    failure: &Mutex<Option<GpuExecutorFailure>>,
) -> Result<(), GpuExecutorFailure>
where
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
{
    let mut owner = Owner::new(backend, budget, frontend, endpoint.sender.clone(), control);
    let mut result = owner.drive(&mut receiver);
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
        if !shutting_down {
            failure.lock().unwrap().get_or_insert_with(|| error.clone());
        }
    }
    budget.stop();
    endpoint.stop(Arc::clone(&cause));
    owner.cancel(&mut receiver, &cause);
    if shutting_down { Ok(()) } else { result }
}

struct AddressSpace {
    driver: nixe_gpu_maxwell::MaxwellGpuAddressSpace,
    memory: Rc<RefCell<nixe_gpu_maxwell::MaxwellGpuMemory>>,
}

struct ActiveSubmission {
    queued: GpuQueuedSubmission,
    context: Box<nixe_gpu_maxwell::MaxwellGpuChannel>,
    memory: Rc<RefCell<nixe_gpu_maxwell::MaxwellGpuMemory>>,
    state: nixe_gpu_maxwell::MaxwellMethodState,
    syncpoint: Option<SyncpointEffect>,
}

struct Owner<
    'a,
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
> {
    active: Option<ActiveSubmission>,
    suspended: std::collections::BTreeMap<FrontendSubmissionId, ActiveSubmission>,
    composite_sender: CommandSender,
    control: Arc<Mutex<NvHostControl>>,
    #[cfg(feature = "performance-counters")]
    split_stats: commands::Statistics,
    #[cfg(feature = "performance-counters")]
    prepare_ns: u64,
    #[cfg(feature = "performance-counters")]
    back_idle_ns: u64,
    idle: Vec<(std::time::Instant, std::time::Instant)>,
    #[cfg(feature = "performance-counters")]
    coherent_ns: u64,
    backend: Rc<RefCell<Option<E>>>,
    budget: &'a GpuWorkBudget,
    memory_managers:
        std::collections::BTreeMap<nixe_gpu_maxwell::MaxwellAddressSpaceId, AddressSpace>,
    frontend: Rc<RefCell<L>>,
    channels: std::collections::BTreeMap<
        nixe_gpu_maxwell::MaxwellChannelId,
        Option<Box<nixe_gpu_maxwell::MaxwellGpuChannel>>,
    >,
}
impl<'a, L, E> Owner<'a, L, E>
where
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
{
    fn new(
        backend: Rc<RefCell<Option<E>>>,
        budget: &'a GpuWorkBudget,
        frontend: L,
        composite_sender: CommandSender,
        control: Arc<Mutex<NvHostControl>>,
    ) -> Self {
        Self {
            #[cfg(feature = "performance-counters")]
            split_stats: Default::default(),
            #[cfg(feature = "performance-counters")]
            prepare_ns: 0,
            #[cfg(feature = "performance-counters")]
            back_idle_ns: 0,
            idle: Vec::new(),
            #[cfg(feature = "performance-counters")]
            coherent_ns: 0,
            active: None,
            suspended: Default::default(),
            composite_sender,
            control,
            backend,
            budget,
            memory_managers: Default::default(),
            frontend: Rc::new(RefCell::new(frontend)),
            channels: Default::default(),
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
        if let Some(runtime) = self.backend.borrow_mut().as_mut() {
            runtime
                .progress()
                .map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))?;
        }
        Ok(())
    }

    fn flush_progress(&mut self) -> Result<(), GpuExecutorFailure> {
        let mut backend = self.backend.borrow_mut();
        if let Some(runtime) = backend.as_mut() {
            runtime
                .flush()
                .map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))?;
        }
        Ok(())
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
            GpuMessage::RequestComposite {
                composite,
                fences,
                registration,
            } => {
                return self.request_composite(composite, fences, registration);
            }
            GpuMessage::Composite(composite) => {
                return self.compose(composite);
            }
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
            _ => unreachable!("ordered control dispatched by execute"),
        };
        result.map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))
    }
    fn service_progress(&mut self) -> Result<(), GpuExecutorFailure> {
        let requested = self.budget.take_progress_request();
        if requested & GpuWorkBudget::FLUSH != 0 {
            self.flush_progress()?;
        } else if requested & GpuWorkBudget::SUBMIT_DEFERRED != 0
            && let Some(runtime) = self.backend.borrow_mut().as_mut()
        {
            runtime
                .flush_deferred()
                .map_err(|e| self.runtime_failure(FrontendSubmissionId::new(0), e))?;
        }
        self.frontend
            .borrow_mut()
            .service_host_requests()
            .map_err(|error| self.failure(FrontendSubmissionId::new(0), error.to_string()))?;
        self.poll()
    }
    fn drive(
        &mut self,
        receiver: &mut nixe_gpu::ordered::Receiver<GpuMessage>,
    ) -> Result<(), GpuExecutorFailure> {
        while !self.budget.is_stopped() {
            match self.next_message(receiver)? {
                Some(nixe_gpu::ordered::Read::Control(message)) => self.execute(message)?,
                Some(nixe_gpu::ordered::Read::Words(mut batch)) => {
                    self.methods(Some(&mut batch))?
                }
                None => break,
            }
        }
        Ok(())
    }
    fn next_message<'r>(
        &mut self,
        receiver: &'r mut nixe_gpu::ordered::Receiver<GpuMessage>,
    ) -> Result<Option<nixe_gpu::ordered::Read<'r, GpuMessage>>, GpuExecutorFailure> {
        self.idle.clear();
        if !receiver.ready() {
            let mut start = std::time::Instant::now();
            let budget = self.budget;
            let ready = receiver.wait(
                || budget.is_stopped(),
                || {
                    budget
                        .progress_requested
                        .load(std::sync::atomic::Ordering::SeqCst)
                        != 0
                },
                || {
                    self.idle.push((start, std::time::Instant::now()));
                    self.service_progress()?;
                    if let Some(runtime) = self.backend.borrow_mut().as_mut() {
                        runtime.flush_deferred().map_err(|error| {
                            self.runtime_failure(FrontendSubmissionId::new(0), error)
                        })?;
                    }
                    // The hook is real preparation work, not pacing idle time.
                    start = std::time::Instant::now();
                    Ok::<_, GpuExecutorFailure>(())
                },
            )?;
            let end = std::time::Instant::now();
            #[cfg(feature = "performance-counters")]
            {
                self.back_idle_ns += end.duration_since(start).as_nanos() as u64;
            }
            self.idle.push((start, end));
            if !ready {
                return Ok(None);
            }
        }
        Ok(receiver.try_recv())
    }
    fn execute(&mut self, message: GpuMessage) -> Result<(), GpuExecutorFailure> {
        match message {
            GpuMessage::FrontIdle { start, end } => {
                // Pacing can be excluded only while both CPU stages were idle.
                // A waiting front does not make overlapping preparation free.
                for &(back_start, back_end) in &self.idle {
                    let start = start.max(back_start);
                    let end = end.min(back_end);
                    if start < end
                        && let Some(backend) = self.backend.borrow_mut().as_mut()
                    {
                        backend.frame_idle(start, end);
                    }
                }
            }
            #[cfg(feature = "performance-counters")]
            GpuMessage::FrontMetrics(stats) => self.split_stats = stats,
            GpuMessage::SemaphoreRead {
                frontend,
                space,
                address,
                reply,
            } => {
                let memory = &mut self
                    .memory_managers
                    .get(&space)
                    .expect("registered semaphore space")
                    .memory
                    .borrow_mut();
                let result = (|| {
                    let mut backend = self.backend.borrow_mut();
                    let access = backend.as_mut().expect("installed backend");
                    memory.flush_caching(access)?;
                    let address = memory
                        .address(address)
                        .map_err(nixe_gpu_maxwell::MaxwellGpuAccessError::Address)?;
                    let mut bytes = [0; 4];
                    memory.read_into(address, &mut bytes, access)?;
                    Ok(u32::from_le_bytes(bytes))
                })();
                let _ = reply.send((frontend, CommandReadReply::Semaphore(result)));
                self.budget.wake_command();
            }
            GpuMessage::CommandFetch { request, reply } => {
                #[cfg(feature = "performance-counters")]
                let start = std::time::Instant::now();
                let frontend = request.origin.frontend;
                let mut memory = self
                    .memory_managers
                    .get(&request.space)
                    .expect("registered GPU address space")
                    .memory
                    .borrow_mut();
                let mut backend = self.backend.borrow_mut();
                let result = request
                    .read(&mut memory, backend.as_mut().expect("installed backend"))
                    .map_err(|e| {
                        nixe_gpu_maxwell::MaxwellFrontendDispatchError::CommandSource(Box::new(e))
                    });
                #[cfg(feature = "performance-counters")]
                {
                    self.coherent_ns += start.elapsed().as_nanos() as u64;
                }
                let _ = reply.send((frontend, CommandReadReply::Commands(result)));
                self.budget.wake_command();
            }
            GpuMessage::Submission(queued) => self.begin_submission(queued)?,
            GpuMessage::SelectSubmission(id) => self.select_submission(id),
            GpuMessage::EndSubmission => self.methods(None)?,

            GpuMessage::CreateAddressSpace(space) => {
                self.frontend.borrow_mut().bind_gpu_memory(space.memory());
                let memory = Rc::new(RefCell::new(nixe_gpu_maxwell::MaxwellGpuMemory::new(
                    space.memory().clone(),
                )));
                assert!(
                    self.memory_managers
                        .insert(
                            space.id(),
                            AddressSpace {
                                driver: space,
                                memory
                            }
                        )
                        .is_none(),
                    "GPU address space already exists"
                );
            }
            GpuMessage::AddressSpaceIoctl {
                space,
                operation,
                reply,
            } => {
                let space = self
                    .memory_managers
                    .get_mut(&space)
                    .expect("registered GPU address space");
                reply.complete(operation.execute(&mut space.driver));
            }
            GpuMessage::CreateChannel(channel) => {
                assert!(
                    self.channels
                        .insert(
                            channel.id(),
                            Some(Box::new(
                                nixe_gpu_maxwell::MaxwellGpuChannel::from_configuration(channel)
                            ))
                        )
                        .is_none(),
                    "hardware channel already exists"
                );
                self.frontend.borrow_mut().create_channel(channel.id());
            }
            GpuMessage::CloseChannel(channel) => {
                self.channels.remove(&channel);
                self.frontend.borrow_mut().close_channel(channel);
            }
            GpuMessage::CloseAddressSpace(space) => {
                self.memory_managers.remove(&space);
            }
            GpuMessage::ConvertNv12 {
                destination,
                transform,
                source,
                reply,
            } => {
                if let Some(backend) = self.backend.borrow_mut().as_mut() {
                    backend.begin_frame_work();
                }
                let result = (|| {
                    let work = self
                        .frontend
                        .borrow_mut()
                        .lower_nv12_conversion(*destination, source, transform)
                        .map_err(|e| self.failure(FrontendSubmissionId::new(0), e.to_string()))?;
                    let id = work.frontend;
                    self.backend
                        .borrow_mut()
                        .as_mut()
                        .expect("installed backend")
                        .convert_nv12(&work)
                        .map_err(|e| self.runtime_failure(id, e))?;
                    let completed_reply = reply.clone();
                    let action: nixe_gpu::FenceAction = Box::new(move |result| {
                        let _ =
                            completed_reply.send(result.map_err(|cause| cause.to_string().into()));
                        Ok(())
                    });
                    self.backend
                        .borrow_mut()
                        .as_mut()
                        .expect("installed backend")
                        .signal_fence(Some(action))
                        .map_err(|e| self.runtime_failure(id, e))
                })();
                if let Err(error) = &result {
                    let _ = reply.send(Err(error.to_string().into()));
                }
                result?;
            }
            message => return self.service_control(message),
        }
        Ok(())
    }
    fn begin_submission(&mut self, queued: GpuQueuedSubmission) -> Result<(), GpuExecutorFailure> {
        let submission = &queued.submission;
        let id = submission.dispatch.scheduled().frontend();
        if let Some(backend) = self.backend.borrow_mut().as_mut() {
            backend.begin_frame_work_at(queued.started.unwrap_or_else(std::time::Instant::now));
        }
        let mut context = self
            .channels
            .get_mut(&submission.channel.id())
            .expect("registered hardware channel")
            .take()
            .expect("one active submission per channel");
        context.update_configuration(submission.channel);
        let memory = Rc::clone(
            &self
                .memory_managers
                .get(&submission.address_space)
                .expect("registered address space")
                .memory,
        );
        let state =
            nixe_gpu_maxwell::MaxwellMethodState::new(&submission.dispatch, &memory.borrow());
        let syncpoint = submission
            .reservation
            .clone()
            .map(|reservation| SyncpointEffect {
                descriptor: submission.descriptor,
                request: submission.request,
                reservation,
                control: Arc::clone(&self.control),
            });
        self.suspended.insert(
            id,
            ActiveSubmission {
                queued,
                context,
                memory,
                state,
                syncpoint,
            },
        );
        nixe_memory::trace::event(
            "gpu_command_begin",
            id.get(),
            self.suspended[&id].queued.submission.channel.id().get(),
            0,
        );
        Ok(())
    }
    fn select_submission(&mut self, id: FrontendSubmissionId) {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.queued.submission.dispatch.scheduled().frontend() == id)
        {
            return;
        }
        let next = self.suspended.remove(&id).expect("admitted submission");
        self.frontend
            .borrow_mut()
            .bind_channel(next.queued.submission.channel.id(), next.context.three_d());
        if let Some(previous) = self.active.replace(next) {
            self.suspended.insert(
                previous.queued.submission.dispatch.scheduled().frontend(),
                previous,
            );
        }
    }
    fn methods(
        &mut self,
        batch: Option<&mut nixe_gpu::ordered::Words<'_, GpuMessage>>,
    ) -> Result<(), GpuExecutorFailure> {
        let Some(mut active) = self.active.take() else {
            return Err(self.failure(FrontendSubmissionId::new(0), "methods outside a submission"));
        };
        let done = batch.is_none();
        let submission = &active.queued.submission;
        let id = submission.dispatch.scheduled().frontend();
        let frontend = Rc::clone(&self.frontend);
        #[cfg(feature = "performance-counters")]
        let start = std::time::Instant::now();
        let result = {
            let mut host = CommandHost {
                owner: self,
                clock: &active.queued.clock,
                syncpoint: active.syncpoint.as_ref(),
            };
            let mut execution = nixe_gpu_maxwell::MaxwellMethodExecution::new(
                &submission.dispatch,
                &mut active.context,
                &active.memory,
                &frontend,
                &mut host,
                &mut active.state,
            );
            match batch {
                Some(batch) => execution.consume(batch),
                None => execution.finish(&submission.dispatch),
            }
        };
        #[cfg(feature = "performance-counters")]
        {
            self.prepare_ns += start.elapsed().as_nanos() as u64;
        }
        result.map_err(|failure| GpuExecutorFailure {
            memory_cause: None,
            frontend: id,
            detail: failure.to_string().into(),
            boundary: Some((
                submission.descriptor,
                submission.request,
                Box::new(MaxwellFrontendDispatchBoundary {
                    dispatch: Box::new(submission.dispatch.clone()),
                    failure,
                }),
            )),
        })?;
        if done {
            *self.channels.get_mut(&submission.channel.id()).unwrap() = Some(active.context);
            if let Some(backend) = self.backend.borrow_mut().as_mut() {
                backend
                    .end_command_list()
                    .map_err(|e| self.runtime_failure(id, e))?;
            }
            nixe_memory::trace::event(
                "gpu_command_end",
                id.get(),
                submission.channel.id().get(),
                0,
            );
            drop(active.queued.submission.permit);
        } else {
            self.active = Some(active);
        }
        Ok(())
    }
    fn request_composite(
        &mut self,
        composite: Composite,
        fences: Box<[nixe_gpu::GuestTimelinePoint]>,
        registration: CompositeRegistration,
    ) -> Result<(), GpuExecutorFailure> {
        let result = self.register_composite(composite, &fences);
        registration.complete(result.as_ref().map(|_| ()).map_err(|e| e.detail.clone()));
        result
    }
    fn register_composite(
        &mut self,
        composite: Composite,
        fences: &[nixe_gpu::GuestTimelinePoint],
    ) -> Result<(), GpuExecutorFailure> {
        if fences.is_empty() {
            return self.compose(composite);
        }
        let pending = Arc::new(PendingComposite {
            state: Mutex::new((fences.len(), Some(composite))),
            sender: self.composite_sender.clone(),
        });
        let mut ready = Vec::new();
        {
            let mut control = self
                .control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for &point in fences {
                match control.register_guest_action(point, Arc::clone(&pending)) {
                    Ok(Some(action)) => ready.push(action),
                    Ok(None) => (),
                    Err(error) => {
                        pending.cancel();
                        return Err(self.failure(FrontendSubmissionId::new(0), error));
                    }
                }
            }
        }
        // RegisterGuestAction invokes already-reached callbacks in this sync
        // operation. No following producer can run before their composition.
        for action in ready {
            if let Some(work) = action.signal() {
                self.compose(work)?;
            }
        }
        Ok(())
    }
    fn compose(&mut self, composite: Composite) -> Result<(), GpuExecutorFailure> {
        #[cfg(feature = "performance-counters")]
        {
            let s = self.split_stats;
            let sequence = composite.sequence;
            nixe_memory::trace::event("split_front_work", sequence, s.active_ns, s.mme_ns);
            nixe_memory::trace::event("split_front_wait", sequence, s.read_ns, s.queue_ns);
            nixe_memory::trace::event(
                "split_back_work",
                sequence,
                self.prepare_ns,
                self.coherent_ns,
            );
            nixe_memory::trace::event("split_back_idle", sequence, self.back_idle_ns, 0);
            nixe_memory::trace::event("split_queue", sequence, s.occupancy, s.entries);
            nixe_memory::trace::event("split_queue_peak", sequence, s.peak, s.words);
            nixe_memory::trace::event("split_requests", sequence, s.read_count, s.batches);
        }
        let Composite {
            request,
            swap_interval,
            pacing,
            presenter,
            sequence,
            source,
            display_sequence,
        } = composite;
        let image = {
            let mut backend = self.backend.borrow_mut();
            let backend = backend.as_mut().ok_or_else(|| {
                self.failure(FrontendSubmissionId::new(0), "GPU backend is unavailable")
            })?;
            backend
                .composite(request, swap_interval, pacing)
                .map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))?
        };
        nixe_memory::trace::event(
            "frame_ready",
            image.lease().producer().slot(),
            image.lease().image().slot,
            image.lease().content_version(),
        );
        nixe_memory::trace::event("vi_latch", source, sequence, display_sequence);
        nixe_memory::trace::event(
            "frame_producer",
            sequence,
            image.lease().producer().slot(),
            image.lease().content_version(),
        );
        // The renderer owns publication. VI waits only for registration and
        // retains the acquired producer item for the reference display period.
        presenter.publish(Arc::new(nixe_video::PresentationFrame::new(
            image, sequence,
        )));
        Ok(())
    }
    fn cancel(
        &mut self,
        receiver: &mut nixe_gpu::ordered::Receiver<GpuMessage>,
        cause: &Arc<MemoryError>,
    ) {
        // This owner shares the actual SyncpointManager with its nvdrv client.
        // Drop unresolved compositions before tearing down native resources.
        self.control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancel_guest_actions();
        while let Some(message) = receiver.try_recv() {
            let nixe_gpu::ordered::Read::Control(message) = message else {
                continue;
            };
            match message {
                GpuMessage::MemoryAccess(request) => request.finish(Err(Arc::clone(cause))),
                GpuMessage::ConvertNv12 { reply, .. } => {
                    let _ = reply.send(Err("GPU owner is shutting down".into()));
                }
                GpuMessage::RequestComposite { registration, .. } => {
                    registration.complete(Err("GPU owner is shutting down".into()));
                }
                _ => {}
            }
        }
    }
}

struct CommandHost<
    'host,
    'owner,
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
> {
    owner: &'host mut Owner<'owner, L, E>,
    clock: &'host nixe_runtime::VirtualClock,
    syncpoint: Option<&'host SyncpointEffect>,
}
impl<
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
> nixe_gpu_maxwell::MaxwellCommandHost for CommandHost<'_, '_, L, E>
{
    fn signal_syncpoint(
        &mut self,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        let effect = self
            .syncpoint
            .ok_or_else(|| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(
                    "syncpoint has no channel reservation".into(),
                )
            })?
            .clone();
        // Command consumption releases guest/GPU dependencies immediately.
        // CPU events advance only after signal_fence publishes prior downloads.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/fence_manager.h#L104-L108
        let actions = effect
            .control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .consume_channel_increment(effect.descriptor, effect.request, &effect.reservation)
            .map_err(|error| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(
                    format!("syncpoint consumption failed: {error:?}").into(),
                )
            })?;
        // IncrementGuest invokes its reached actions before SignalFence, as
        // FenceManager::SignalSyncPoint does. Keep composition in command order.
        for action in actions {
            if let Some(composite) = action.signal() {
                self.owner.compose(composite).map_err(|error| {
                    nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(
                        error.to_string().into(),
                    )
                })?;
            }
        }
        let action: nixe_gpu::FenceAction = Box::new(move |result| {
            result?;
            effect
                .control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .publish_channel_increment(effect.descriptor, effect.request, &effect.reservation)
                .map(|_| ())
                .map_err(|error| {
                    Arc::new(MemoryError::Execution(
                        format!("syncpoint publication failed: {error:?}").into(),
                    ))
                })
        });
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .signal_fence(Some(action))
            .map_err(|error| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
            })
    }

    fn release_fences(&mut self) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        self.owner.poll().map_err(|error| {
            nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
        })
    }

    fn signal_reference(
        &mut self,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .signal_fence(None)
            .map_err(|error| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
            })
    }

    fn report_memory(
        &mut self,
        frontend: FrontendSubmissionId,
        target: Vec<nixe_memory::AllocationRange>,
        bytes: Arc<[u8]>,
        fence: bool,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .report_memory(frontend, target, bytes, fence)
            .map_err(|error| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
            })
    }
    fn order_device(
        &mut self,
        frontend: FrontendSubmissionId,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .order_device(frontend)
            .map_err(|error| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
            })
    }
    fn cache_maintenance(
        &mut self,
        frontend: FrontendSubmissionId,
        operation: nixe_gpu::CacheMaintenanceOperation,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .cache_maintenance(frontend, operation)
            .map_err(|error| {
                nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
            })
    }
    fn is_stopped(&self) -> bool {
        self.owner.budget.is_stopped()
    }
    fn service_progress(
        &mut self,
    ) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        let owner = &mut *self.owner;
        if owner.budget.is_stopped() {
            return Err(nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(
                "GPU owner is shutting down".into(),
            ));
        }
        if owner
            .budget
            .progress_requested
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return Ok(());
        }
        owner.service_progress().map_err(|error| {
            nixe_gpu_maxwell::MaxwellSubmissionExecutionError::Native(error.to_string().into())
        })
    }
    fn timestamp(&self) -> u64 {
        // Match Eden's default fast GPU report clock; CPU/driver time stays unscaled.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/gpu.cpp
        nixe_gpu_maxwell::maxwell_gpu_timestamp(self.clock.scheduler_time_ns()) / 256
    }
}

impl<
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
> nixe_gpu::OwnerMemory for CommandHost<'_, '_, L, E>
{
    fn invalidate_gpu_cache(&mut self) -> Result<(), Arc<MemoryError>> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .invalidate_gpu_cache()
    }

    fn flush_region(&mut self, address: u64, bytes: u64) -> Result<(), Arc<MemoryError>> {
        let owner = &*self.owner;
        let mut backend = owner.backend.borrow_mut();
        let Some(backend) = backend.as_mut() else {
            return Ok(());
        };
        backend.flush_region(address, bytes)
    }
    fn invalidate_regions(
        &mut self,
        regions: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), Arc<MemoryError>> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .invalidate_regions(regions)
    }
    fn inline_memory(
        &mut self,
        regions: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), Arc<MemoryError>> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .inline_memory(regions)
    }
    fn access_memory(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, Arc<MemoryError>> {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .access_memory(ranges, operation)
    }
}
