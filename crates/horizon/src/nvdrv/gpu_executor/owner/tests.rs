use super::*;
use nixe_gpu::BackendExecutionCompletion;
use nixe_gpu::{BackendRuntimeError, CacheMaintenanceOperation};
use nixe_gpu_vulkan::{ApiPreference, VulkanDevice, WorkTicket};
use nixe_memory::{AccessOperation, AccessResult, AllocationRange, Machine};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

mod support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gpu-vulkan/tests/support/device.rs"
    ));
}

struct ReleasePreparationOnDrop(Arc<AtomicBool>);
impl Drop for ReleasePreparationOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

struct QueueFixture {
    memory: Machine,
    effect: AllocationRange,
    device: VulkanDevice,
    pending: VecDeque<(FrontendSubmissionId, WorkTicket)>,
    preparation_ready: Arc<AtomicBool>,
    started_wait: Option<mpsc::Sender<()>>,
    completed: mpsc::Sender<BackendExecutionCompletion>,
    completion_wakeups: usize,
    recorded: Vec<FrontendSubmissionId>,
}
impl QueueFixture {
    fn complete(&mut self) -> Option<BackendExecutionCompletion> {
        let (frontend, work) = *self.pending.front()?;
        if work.tick() > self.device.queue_usage().completed_tick {
            return None;
        }
        self.pending.pop_front();
        // No resource/canonical write exists in this fixture. Real residency
        // completes its ownership effects before constructing this record.
        let completion = BackendExecutionCompletion::new(frontend, work.submission());
        self.completed.send(completion).unwrap();
        Some(completion)
    }
}
impl GpuExecution for QueueFixture {
    type Commands = nixe_gpu::native::NativeWork;
    type ImageRequest = nixe_gpu::native::NativePresentationImage;
    fn flush(&mut self) -> Result<(), BackendRuntimeError> {
        self.device
            .flush()
            .map(|_| ())
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))
    }
    fn consume(&mut self, batch: &Self::Commands) -> Result<bool, BackendRuntimeError> {
        if batch.frontend == FrontendSubmissionId::new(1)
            && !self.preparation_ready.load(Ordering::Acquire)
        {
            if let Some(started) = self.started_wait.take() {
                started.send(()).unwrap();
            }
            return Ok(false);
        }
        self.recorded.push(batch.frontend);
        if batch.frontend == FrontendSubmissionId::new(2) {
            self.preparation_ready.store(true, Ordering::Release);
        }
        self.device
            .work_ticket()
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        self.memory
            .write_resolved(std::slice::from_ref(&self.effect), &[0xa5; 16])
            .map_err(|error| BackendRuntimeError::Memory(Arc::new(error)))?;
        Ok(true)
    }
    fn submit(
        &mut self,
        frontend: FrontendSubmissionId,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        let work = self
            .device
            .work_ticket()
            .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))?;
        if self.preparation_ready.load(Ordering::Acquire) {
            self.device
                .flush()
                .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))?;
        }
        self.pending.push_back((frontend, work));
        Ok(work.submission())
    }
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.device
            .poll_progress()
            .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))?;
        Ok(self.complete())
    }
    fn arm_completion_wakeup(&mut self) -> Result<(), BackendRuntimeError> {
        self.completion_wakeups += 1;
        Ok(())
    }
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<nixe_gpu::OwnerAccess, BackendRuntimeError> {
        self.memory
            .apply_resolved_access(&ranges, &operation)
            .map(nixe_gpu::OwnerAccess::Ready)
            .map_err(|e| BackendRuntimeError::Memory(Arc::new(e)))
    }
    unsafe fn memory_written(
        &mut self,
        _: nixe_memory::MachineId,
        _: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), BackendRuntimeError> {
        unreachable!("fixture does not produce native memory")
    }
    fn queue_device_access(
        &mut self,
        request: nixe_memory::DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError> {
        let result = self
            .memory
            .apply_resolved_access(request.ranges(), request.operation())
            .map_err(Arc::new);
        request.finish(result);
        Ok(())
    }
    fn cancel_device_access(
        &mut self,
        _: nixe_memory::AccessRequestId,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn fail_device_accesses(&mut self, _: Arc<nixe_memory::MemoryError>) {}
    fn retire_physical_memory(
        &mut self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn release_guest_memory(
        &mut self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn allocation_pressure(
        &mut self,
        _: nixe_memory::MachineId,
        _: u64,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn acquire_presentable_image(
        &mut self,
        _: Self::ImageRequest,
        _: u32,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        Err(BackendRuntimeError::Backend(
            "queue fixture has no images".into(),
        ))
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        self.device
            .shutdown()
            .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))
    }
}

#[test]
fn owned_memory_access_waits_until_the_current_effect_is_consumed() {
    use super::super::super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let memory = Machine::new(4096).unwrap();
    let allocation = memory.account(4096).unwrap().allocate(4096).unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, 16).unwrap();
    memory
        .write_resolved(std::slice::from_ref(&input), &[0x5a; 16])
        .unwrap();

    let preparation_ready = Arc::new(AtomicBool::new(false));
    let _release = ReleasePreparationOnDrop(Arc::clone(&preparation_ready));
    let budget = GpuWorkBudget::default();
    let budget = Arc::new(budget);
    let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve(0) else {
        panic!();
    };
    let (sender, receiver) = mpsc::channel();
    let (started, waited) = mpsc::channel();
    let (completed_tx, completed) = mpsc::channel();
    let backend = Rc::new(RefCell::new(Some(QueueFixture {
        memory: memory.clone(),
        effect: input.clone(),
        device,
        pending: VecDeque::new(),
        preparation_ready: Arc::clone(&preparation_ready),
        started_wait: Some(started),
        completed: completed_tx,
        completion_wakeups: 0,
        recorded: Vec::new(),
    })));
    let owner = Rc::new(RefCell::new(Owner::new(
        &receiver,
        Rc::clone(&backend),
        &budget,
        nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
    )));
    let work = GpuWork {
        descriptor: NvDrvDeviceDescriptor::open(
            NvDrvFileDescriptor::new(1),
            NvDrvDeviceKind::HostGpu,
            NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
            NvDrvPermissionProfile::Application,
        ),
        request: 0xc018_481b,
        frontend: FrontendSubmissionId::new(1),
        reservation: None,
        control: Arc::new(Mutex::new(NvHostControl::default())),
    };
    let command = nixe_gpu::native::NativeWork {
        frontend: work.frontend,
        command: nixe_gpu::native::NativeCommand::Ordering(
            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
        ),
    };
    let warm_sender = sender.clone();
    let producer_memory = memory.clone();
    let producer_input = input.clone();
    let producer = thread::spawn(move || {
        waited.recv_timeout(Duration::from_secs(3)).unwrap();
        let (request, mut pending) = producer_memory
            .prepare_device_access(vec![producer_input], AccessOperation::Read)
            .unwrap();
        let (wake, completed) = mpsc::sync_channel(1);
        pending.set_waker(Arc::new(move || {
            let _ = wake.try_send(());
        }));
        sender.send(GpuMessage::MemoryAccess(request)).unwrap();
        // Compiler workers own captured inputs. Readiness must not depend on
        // servicing a later guest memory operation ahead of this command list.
        preparation_ready.store(true, Ordering::Release);
        sender.send(GpuMessage::Wake).unwrap();
        completed.recv_timeout(Duration::from_secs(3)).unwrap();
        assert!(
            matches!(pending.poll().unwrap().unwrap(), AccessResult::Read(bytes) if bytes.as_ref() == [0xa5; 16])
        );
    });
    finish(
        &owner,
        CommandHost {
            owner: Rc::clone(&owner),
            clock: Default::default(),
        }
        .emit_work(&command),
    )
    .unwrap();
    owner.borrow_mut().service_ready_control().unwrap();
    assert_eq!(
        backend.borrow().as_ref().unwrap().completion_wakeups,
        0,
        "compiler readiness and admitted RAM access must not submit unrelated work"
    );
    let (request, mut warm_reply) = memory
        .prepare_device_access(vec![input], AccessOperation::Read)
        .unwrap();
    warm_sender.send(GpuMessage::MemoryAccess(request)).unwrap();
    finish(
        &owner,
        CommandHost {
            owner: Rc::clone(&owner),
            clock: Default::default(),
        }
        .emit_work(&command),
    )
    .unwrap();
    owner.borrow_mut().service_ready_control().unwrap();
    assert!(matches!(warm_reply.poll().unwrap().unwrap(),
        AccessResult::Read(bytes) if bytes.as_ref() == [0xa5; 16]));
    let token = backend
        .borrow_mut()
        .as_mut()
        .unwrap()
        .submit(work.frontend)
        .unwrap();
    drop(permit);
    owner
        .borrow_mut()
        .pending
        .push_back(PendingWork { token, work });
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !owner.borrow().pending.is_empty() {
        owner.borrow_mut().poll().unwrap();
        assert!(std::time::Instant::now() < deadline);
        thread::yield_now();
    }
    assert_eq!(
        completed
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .submission(),
        token
    );
    assert_eq!(budget.state.lock().unwrap().active, 0);
    producer.join().unwrap();
    backend.borrow_mut().as_mut().unwrap().teardown().unwrap();
    assert!(validation.errors().is_empty(), "{:?}", validation.errors());
}

fn queued_commands(
    budget: &Arc<GpuWorkBudget>,
    memory: &Machine,
    channel_id: u64,
    frontend_id: u64,
    sync_wait_second: bool,
    build: impl FnOnce(u64, &AllocationRange) -> Vec<Vec<u32>>,
) -> (
    nixe_gpu_maxwell::MaxwellGpuAddressSpace,
    GpuQueuedSubmission,
) {
    use super::super::super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    use nixe_gpu_maxwell::*;
    let allocation = memory.account(0x4000).unwrap().allocate(0x4000).unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, 0x4000).unwrap();
    let mut space = MaxwellGpuAddressSpace::new(
        memory.clone(),
        MaxwellAddressSpaceId::new(channel_id),
        SWITCH_1_GM20B_PROFILE,
    );
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mapping = space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(channel_id),
            backing: vec![input.clone()],
            backing_offset: 0,
            size: 0x4000,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: nixe_memory::MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    let base = mapping.offset().get();
    let entries = build(base, &input);
    let mut descriptors = Vec::new();
    for (index, words) in entries.iter().enumerate() {
        let offset = 0x1000 * (index as u64 + 1);
        let data: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        memory
            .write_resolved(&[input.slice(offset, data.len() as u64).unwrap()], &data)
            .unwrap();
        let address = base + offset;
        descriptors.extend_from_slice(&(address as u32).to_le_bytes());
        descriptors.extend_from_slice(
            &(((address >> 32) as u32)
                | ((words.len() as u32) << 10)
                | (if sync_wait_second && index == 1 {
                    1 << 31
                } else {
                    0
                }))
            .to_le_bytes(),
        );
    }
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(channel_id),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    channel.bind_address_space(space.id()).unwrap();
    channel
        .bind_memory_manager(MaxwellMemoryManagerId::new(1))
        .unwrap();
    channel
        .allocate_gpfifo(128, false, nixe_gpu::GuestSyncpointId::new(1))
        .unwrap();
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        128,
        MaxwellGpfifoSubmitRequest {
            entry_count: entries.len() as u32,
            flags: 4,
            fence_id: 0,
            fence_value: 0,
        },
        &descriptors,
    )
    .unwrap();
    let retained = validate_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(frontend_id),
        decoded,
        space.memory(),
    )
    .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), retained, None, None)
        .unwrap();
    let dispatch = scheduler
        .dispatch_next(true, space.memory())
        .unwrap()
        .unwrap();
    let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve(descriptors.len()) else {
        panic!("ingress capacity")
    };
    let queued = GpuQueuedSubmission {
        clock: Default::default(),
        submission: GpuSubmission::new(
            NvDrvDeviceDescriptor::open(
                NvDrvFileDescriptor::new(channel_id as u32),
                NvDrvDeviceKind::HostGpu,
                NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
                NvDrvPermissionProfile::Application,
            ),
            0xc018_481b,
            dispatch,
            channel.configuration(),
            space.memory().clone(),
            None,
            Arc::new(Mutex::new(NvHostControl::default())),
            permit,
        ),
    };
    (space, queued)
}

#[test]
fn gpu_written_command_entry_is_consumed_after_dma_without_sync_wait() {
    command_entry_after_dma(false, false);
}
#[test]
fn gpu_written_command_entry_is_consumed_after_dma_with_sync_wait() {
    command_entry_after_dma(false, true);
}
#[test]
fn inline_upload_overwrites_gpu_written_command_entry_before_consumption() {
    command_entry_after_dma(true, false);
}
fn command_entry_after_dma(overwrite: bool, sync_wait: bool) {
    use nixe_gpu_vulkan::{NativeGpuExecution, cache::CacheConfiguration};
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let memory = Machine::new(0x10000).unwrap();
    let budget = Arc::new(GpuWorkBudget::default());
    let mut output = None;
    let (_space, queued) = queued_commands(&budget, &memory, 1, 1, sync_wait, |base, input| {
        let word = (4u32 << 29) | (0x50 / 4); // Valid SET_REFERENCE replaces an invalid header.
        memory
            .write_resolved(&[input.slice(0, 4).unwrap()], &word.to_le_bytes())
            .unwrap();
        output = Some(input.slice(0x2000, 4).unwrap());
        let mut first = Vec::new();
        let mut packet = |method: u32, args: &[u32]| {
            first.push((1 << 29) | ((args.len() as u32) << 16) | (4 << 13) | (method / 4));
            first.extend_from_slice(args);
        };
        packet(
            0,
            &[nixe_gpu_maxwell::SWITCH_1_GM20B_PROFILE
                .classes()
                .dma_copy()
                .0],
        );
        packet(
            0x400,
            &[
                (base >> 32) as u32,
                base as u32,
                ((base + 0x2000) >> 32) as u32,
                (base + 0x2000) as u32,
                4,
                4,
                4,
                1,
            ],
        );
        packet(0x300, &[0x182]); // Pitch-to-pitch, one-dimensional nonpipelined DMA.
        if overwrite {
            let destination = base + 0x2000;
            // The inline constant upload overwrites a native producer on the
            // same owner before the next command entry consumes those bytes.
            first.extend_from_slice(&[
                (1 << 29) | (1 << 16),
                nixe_gpu_maxwell::SWITCH_1_GM20B_PROFILE
                    .classes()
                    .three_d()
                    .0,
                (1 << 29) | (4 << 16) | (0x2380 / 4),
                4,
                (destination >> 32) as u32,
                destination as u32,
                0,
                (3 << 29) | (1 << 16) | (0x2390 / 4),
                (4 << 29) | (7 << 16) | (0x50 / 4),
            ]);
        }
        vec![first, vec![2 << 29]]
    });
    let mut backend = NativeGpuExecution::new(
        memory.clone(),
        device,
        nixe_gpu::shader_identity::ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    let (sender, receiver) = mpsc::channel();
    backend
        .bind_host_wakeup(Arc::new(move || {
            let _ = sender.send(GpuMessage::Wake);
        }))
        .unwrap();
    let backend = Rc::new(RefCell::new(Some(backend)));
    let owner = Rc::new(RefCell::new(Owner::new(
        &receiver,
        Rc::clone(&backend),
        &budget,
        nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
    )));
    finish(&owner, Owner::submission(Rc::clone(&owner), queued)).unwrap();
    assert_eq!(budget.state.lock().unwrap().active, 0);
    let mut actual = [0; 4];
    memory
        .capture_stream(&[output.unwrap()], &mut actual)
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(actual),
        (4 << 29) | (if overwrite { 7 << 16 } else { 0 }) | (0x50 / 4)
    );
    backend.borrow_mut().as_mut().unwrap().teardown().unwrap();
    assert!(validation.errors().is_empty(), "{:?}", validation.errors());
}

fn finish<
    L: MaxwellFrontend,
    E: GpuExecution<
            Commands = nixe_gpu::native::NativeWork,
            ImageRequest = nixe_gpu::native::NativePresentationImage,
        >,
    F: Future,
>(
    owner: &Rc<RefCell<Owner<'_, L, E>>>,
    future: F,
) -> F::Output {
    let mut future = std::pin::pin!(future);
    loop {
        match future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                let message = owner
                    .borrow()
                    .receiver
                    .recv_timeout(Duration::from_secs(3))
                    .unwrap();
                owner.borrow_mut().service_control(message).unwrap();
            }
        }
    }
}
struct ControlledPreparation {
    memory: Machine,
    outer_ready: Arc<AtomicBool>,
    inner_ready: Arc<AtomicBool>,
    inner_started: Option<mpsc::Sender<()>>,
    recorded: mpsc::Sender<FrontendSubmissionId>,
}
impl GpuExecution for ControlledPreparation {
    type Commands = nixe_gpu::native::NativeWork;
    type ImageRequest = nixe_gpu::native::NativePresentationImage;
    fn flush(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn consume(&mut self, command: &Self::Commands) -> Result<bool, BackendRuntimeError> {
        if command.frontend.get() == 1 && !self.outer_ready.load(Ordering::Acquire) {
            return Ok(false);
        }
        if command.frontend.get() == 2 && !self.inner_ready.load(Ordering::Acquire) {
            if let Some(started) = self.inner_started.take() {
                started.send(()).unwrap();
            }
            return Ok(false);
        }
        self.recorded.send(command.frontend).unwrap();
        Ok(true)
    }
    fn submit(
        &mut self,
        frontend: FrontendSubmissionId,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        Ok(BackendSubmissionToken::new(
            nixe_gpu::BackendInstanceId::new(1),
            frontend.get(),
            1,
        ))
    }
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        Ok(None)
    }
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<nixe_gpu::OwnerAccess, BackendRuntimeError> {
        self.memory
            .apply_resolved_access(&ranges, &operation)
            .map(nixe_gpu::OwnerAccess::Ready)
            .map_err(|e| BackendRuntimeError::Memory(Arc::new(e)))
    }
    unsafe fn memory_written(
        &mut self,
        _: nixe_memory::MachineId,
        _: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), BackendRuntimeError> {
        unreachable!("fixture does not produce native memory")
    }
    fn queue_device_access(
        &mut self,
        request: nixe_memory::DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError> {
        let result = self
            .memory
            .apply_resolved_access(request.ranges(), request.operation())
            .map_err(Arc::new);
        request.finish(result);
        Ok(())
    }
    fn cancel_device_access(
        &mut self,
        _: nixe_memory::AccessRequestId,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn fail_device_accesses(&mut self, _: Arc<nixe_memory::MemoryError>) {}
    fn retire_physical_memory(
        &mut self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn release_guest_memory(
        &mut self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn allocation_pressure(
        &mut self,
        _: nixe_memory::MachineId,
        _: u64,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn acquire_presentable_image(
        &mut self,
        _: Self::ImageRequest,
        _: u32,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        unreachable!()
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
}

#[test]
fn ready_channel_resumes_while_another_remains_blocked_and_preserves_fifo() {
    let memory = Machine::new(0x10000).unwrap();
    let outer_ready = Arc::new(AtomicBool::new(false));
    let inner_ready = Arc::new(AtomicBool::new(false));
    let budget = Arc::new(GpuWorkBudget::default());
    let (sender, receiver) = mpsc::channel();
    let (inner_started, waiting_inner) = mpsc::channel();
    let (recorded, records) = mpsc::channel();
    let backend = Rc::new(RefCell::new(Some(ControlledPreparation {
        memory: memory.clone(),
        outer_ready: outer_ready.clone(),
        inner_ready: inner_ready.clone(),
        inner_started: Some(inner_started),
        recorded,
    })));
    let commands = |_, _: &AllocationRange| vec![vec![(4 << 29) | (0x50 / 4)]];
    let (_first_space, first) = queued_commands(&budget, &memory, 1, 1, false, commands);
    let (_second_space, second) = queued_commands(&budget, &memory, 2, 2, false, commands);
    let (_third_space, third) = queued_commands(&budget, &memory, 1, 3, false, commands);
    for queued in [first, third, second] {
        sender.send(GpuMessage::Submission(queued)).unwrap();
    }
    let producer = thread::spawn(move || {
        waiting_inner.recv_timeout(Duration::from_secs(3)).unwrap();
        outer_ready.store(true, Ordering::Release);
        sender.send(GpuMessage::Wake).unwrap();
        let first = records.recv_timeout(Duration::from_secs(3)).ok();
        let third = records.recv_timeout(Duration::from_secs(3)).ok();
        // Release even on regression so the test cannot hang in its failure path.
        inner_ready.store(true, Ordering::Release);
        sender.send(GpuMessage::Wake).unwrap();
        let second = records.recv_timeout(Duration::from_secs(3)).ok();
        (first, third, second)
    });
    let owner = Rc::new(RefCell::new(Owner::new(
        &receiver,
        Rc::clone(&backend),
        &budget,
        nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
    )));
    Owner::drive(&owner).unwrap();
    assert_eq!(
        producer.join().unwrap(),
        (
            Some(FrontendSubmissionId::new(1)),
            Some(FrontendSubmissionId::new(3)),
            Some(FrontendSubmissionId::new(2))
        )
    );
    assert_eq!(budget.state.lock().unwrap().active, 0);
}

#[test]
fn stopping_with_suspended_channels_releases_admission_without_emitting_effects() {
    let memory = Machine::new(0x10000).unwrap();
    let budget = Arc::new(GpuWorkBudget::default());
    let (sender, receiver) = mpsc::channel();
    let (started, waiting) = mpsc::channel();
    let (recorded, records) = mpsc::channel();
    let backend = Rc::new(RefCell::new(Some(ControlledPreparation {
        memory: memory.clone(),
        outer_ready: Arc::new(AtomicBool::new(false)),
        inner_ready: Arc::new(AtomicBool::new(false)),
        inner_started: Some(started),
        recorded,
    })));
    let commands = |_, _: &AllocationRange| vec![vec![(4 << 29) | (0x50 / 4)]];
    let (_first_space, first) = queued_commands(&budget, &memory, 1, 1, false, commands);
    let (_second_space, second) = queued_commands(&budget, &memory, 2, 2, false, commands);
    sender.send(GpuMessage::Submission(first)).unwrap();
    sender.send(GpuMessage::Submission(second)).unwrap();
    let stopping_budget = Arc::clone(&budget);
    let stopper = thread::spawn(move || {
        let started = waiting.recv_timeout(Duration::from_secs(3));
        stopping_budget.stop();
        sender.send(GpuMessage::Wake).unwrap();
        started.unwrap();
    });
    let owner = Rc::new(RefCell::new(Owner::new(
        &receiver,
        backend,
        &budget,
        nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
    )));
    Owner::drive(&owner).unwrap();
    stopper.join().unwrap();
    assert!(records.try_recv().is_err());
    assert_eq!(budget.state.lock().unwrap().active, 0);
}

#[test]
fn idle_owner_publishes_explicit_fences_without_new_commands() {
    use super::super::super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    use nixe_gpu::native::{NativeCommand, NativeWork};
    use nixe_gpu_vulkan::{NativeGpuExecution, cache::CacheConfiguration};
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let memory = Machine::new(0x10000).unwrap();
    let allocation = memory.account(4096).unwrap().allocate(4096).unwrap();
    let target = AllocationRange::new(allocation.pin(), 0, 4096).unwrap();
    let budget = Arc::new(GpuWorkBudget::default());
    let (sender, receiver) = mpsc::channel();
    let (completed, observed) = mpsc::channel();
    let wake = sender.clone();
    let mut native = NativeGpuExecution::new(
        memory,
        device,
        nixe_gpu::shader_identity::ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    native
        .bind_host_wakeup(Arc::new(move || {
            let _ = completed.send(());
            let _ = wake.send(GpuMessage::Wake);
        }))
        .unwrap();
    let work = NativeWork {
        frontend: FrontendSubmissionId::new(1),
        command: NativeCommand::ReportMemory {
            fence: true,
            target: vec![target],
            offset: 64,
            bytes: Arc::from([0x37; 16]),
        },
    };
    assert!(native.consume(&work).unwrap());
    let token = native.submit(work.frontend).unwrap();

    let backend = Rc::new(RefCell::new(Some(native)));
    let owner = Rc::new(RefCell::new(Owner::new(
        &receiver,
        Rc::clone(&backend),
        &budget,
        nixe_gpu_maxwell::MaxwellNativeFrontend::new(Default::default()),
    )));
    owner.borrow_mut().pending.push_back(PendingWork {
        token,
        work: GpuWork {
            descriptor: NvDrvDeviceDescriptor::open(
                NvDrvFileDescriptor::new(1),
                NvDrvDeviceKind::HostGpu,
                NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
                NvDrvPermissionProfile::Application,
            ),
            request: 0xc018_481b,
            frontend: work.frontend,
            reservation: None,
            control: Arc::new(Mutex::new(NvHostControl::default())),
        },
    });
    let stopping = Arc::clone(&budget);
    let watchdog = thread::spawn(move || {
        let notified = observed.recv_timeout(Duration::from_secs(3)).is_ok();
        stopping.stop();
        let _ = sender.send(GpuMessage::Wake);
        notified
    });
    Owner::drive(&owner).unwrap();
    assert!(
        watchdog.join().unwrap(),
        "idle work needs a deadline, not another command or CPU request"
    );
    backend.borrow_mut().as_mut().unwrap().teardown().unwrap();
    drop(owner);
    drop(backend);
    validation.check().unwrap();
}
