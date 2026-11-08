//! The command processor owns translation, canonical effects and native scheduling.
//! Worker compilation and device execution remain asynchronous. No lowered-command
//! queue separates guest memory observation from its native resource consumption.
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
    receiver: mpsc::Receiver<GpuMessage>,
    backend: Rc<RefCell<Option<E>>>,
    frontend: L,
    budget: &GpuWorkBudget,
    endpoint: &GpuMemoryEndpoint,
    control: Arc<Mutex<NvHostControl>>,
) -> Result<(), GpuExecutorFailure>
where
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
{
    let mut owner = Owner::new(
        &receiver,
        backend,
        budget,
        frontend,
        endpoint.sender.clone(),
        control,
    );
    let mut result = owner.drive();
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
    owner.cancel(&cause);
    if shutting_down { Ok(()) } else { result }
}

struct Owner<
    'a,
    L: MaxwellFrontend,
    E: GpuExecution<
            Conversion = nixe_gpu::native::NativeNv12Conversion,
            ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
        >,
> {
    receiver: &'a mpsc::Receiver<GpuMessage>,
    composite_sender: mpsc::Sender<GpuMessage>,
    control: Arc<Mutex<NvHostControl>>,
    backend: Rc<RefCell<Option<E>>>,
    budget: &'a GpuWorkBudget,
    deferred: VecDeque<GpuMessage>,
    sources: nixe_gpu_maxwell::MaxwellCommandSourceScratch,
    memory_managers: std::collections::BTreeMap<
        nixe_gpu_maxwell::MaxwellAddressSpaceId,
        Rc<RefCell<nixe_gpu_maxwell::MaxwellGpuMemory>>,
    >,
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
        receiver: &'a mpsc::Receiver<GpuMessage>,
        backend: Rc<RefCell<Option<E>>>,
        budget: &'a GpuWorkBudget,
        frontend: L,
        composite_sender: mpsc::Sender<GpuMessage>,
        control: Arc<Mutex<NvHostControl>>,
    ) -> Self {
        Self {
            receiver,
            composite_sender,
            control,
            backend,
            budget,
            deferred: VecDeque::new(),
            sources: Default::default(),
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
            GpuMessage::CancelMemory(id) => {
                if let Some(index) = self.deferred.iter().position(|message| {
                    matches!(
                        message, GpuMessage::MemoryAccess(request) if request.identity() == id
                    )
                }) && let Some(GpuMessage::MemoryAccess(request)) = self.deferred.remove(index)
                {
                    request.finish(Err(Arc::new(MemoryError::Execution(
                        "GPU memory request was cancelled".into(),
                    ))));
                }
                Ok(())
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
            GpuMessage::Wake => Ok(()),
            message => {
                self.deferred.push_back(message);
                Ok(())
            }
        };
        result.map_err(|error| self.runtime_failure(FrontendSubmissionId::new(0), error))
    }
    fn service_ready_control(&mut self) -> Result<(), GpuExecutorFailure> {
        let mut serviced = false;
        let mut wake = false;
        while let Ok(message) = self.receiver.try_recv() {
            match message {
                GpuMessage::Submission(_)
                | GpuMessage::CreateChannel(_)
                | GpuMessage::CloseChannel(_)
                | GpuMessage::CreateAddressSpace(_)
                | GpuMessage::CloseAddressSpace(_)
                | GpuMessage::ConvertNv12 { .. }
                | GpuMessage::RequestComposite { .. }
                | GpuMessage::Composite(_) => self.deferred.push_back(message),
                message => {
                    serviced = true;
                    wake |= matches!(message, GpuMessage::Wake);
                    self.service_control(message)?;
                }
            }
        }
        if serviced {
            self.poll()?;
        }
        if wake && self.budget.take_progress_request() {
            self.flush_progress()?;
        }
        Ok(())
    }
    fn drive(&mut self) -> Result<(), GpuExecutorFailure> {
        loop {
            if self.budget.is_stopped() {
                return Ok(());
            }
            self.poll()?;
            if self.budget.take_progress_request() {
                self.flush_progress()?;
            }
            let message = match self.deferred.pop_front() {
                Some(message) => message,
                None => match self.receiver.recv() {
                    Ok(message) => message,
                    Err(_) => return Ok(()),
                },
            };
            self.execute(message)?;
        }
    }
    fn execute(&mut self, message: GpuMessage) -> Result<(), GpuExecutorFailure> {
        match message {
            GpuMessage::CreateAddressSpace(handle) => {
                self.frontend.borrow_mut().bind_gpu_memory(&handle);
                assert!(
                    self.memory_managers
                        .insert(
                            handle.id(),
                            Rc::new(RefCell::new(nixe_gpu_maxwell::MaxwellGpuMemory::new(
                                handle
                            )))
                        )
                        .is_none(),
                    "GPU address space already exists"
                );
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
            GpuMessage::Submission(queued) => return self.submission(queued),
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
    fn submission(&mut self, queued: GpuQueuedSubmission) -> Result<(), GpuExecutorFailure> {
        let GpuQueuedSubmission { submission, clock } = queued;
        let GpuSubmission {
            descriptor,
            request,
            dispatch,
            channel,
            address_space,
            reservation,
            permit,
        } = submission;
        let id = dispatch.scheduled().frontend();
        if let Some(backend) = self.backend.borrow_mut().as_mut() {
            backend.begin_frame_work();
        }
        let _trace =
            nixe_memory::trace::span("gpu_command_stream", id.get(), channel.id().get(), 0);
        nixe_memory::trace::event("gpu_command_begin", id.get(), channel.id().get(), 0);
        let syncpoint = reservation.map(|reservation| SyncpointEffect {
            descriptor,
            request,
            reservation,
            control: Arc::clone(&self.control),
        });
        let mut context = self
            .channels
            .get_mut(&channel.id())
            .expect("registered hardware channel")
            .take()
            .expect("one active command list per channel");
        let memory = Rc::clone(
            self.memory_managers
                .get(&address_space)
                .expect("registered GPU address space"),
        );
        let frontend = Rc::clone(&self.frontend);
        let mut sources = std::mem::take(&mut self.sources);
        context.update_configuration(channel);
        frontend
            .borrow_mut()
            .bind_channel(channel.id(), context.three_d());
        let result = {
            let mut host = CommandHost {
                owner: self,
                clock,
                syncpoint,
            };
            stream_maxwell_frontend(
                &dispatch,
                &mut context,
                &memory,
                &frontend,
                &mut sources,
                &mut host,
            )
        };
        *self.channels.get_mut(&channel.id()).unwrap() = Some(context);
        self.sources.recycle(sources);
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
        if let Some(backend) = self.backend.borrow_mut().as_mut() {
            backend
                .end_command_list()
                .map_err(|error| self.runtime_failure(id, error))?;
        }
        drop(permit);
        nixe_memory::trace::event("gpu_command_end", id.get(), channel.id().get(), 0);
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
        let Composite {
            request,
            swap_interval,
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
            backend.begin_frame_work();
            backend
                .composite(request, swap_interval)
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
    fn cancel(&mut self, cause: &Arc<MemoryError>) {
        // This owner shares the actual SyncpointManager with its nvdrv client.
        // Drop unresolved compositions before tearing down native resources.
        self.control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancel_guest_actions();
        let deferred = self.deferred.drain(..).chain(self.receiver.try_iter());
        for message in deferred {
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
    clock: nixe_runtime::VirtualClock,
    syncpoint: Option<SyncpointEffect>,
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
            .as_ref()
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
    fn tick_work(&mut self) -> Result<(), nixe_gpu_maxwell::MaxwellSubmissionExecutionError> {
        let owner = &mut *self.owner;
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
    fn invalidate_gpu_cache(&mut self) {
        self.owner
            .backend
            .borrow_mut()
            .as_mut()
            .unwrap()
            .invalidate_gpu_cache();
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
