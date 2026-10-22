use super::*;
use nixe_gpu::BackendExecutionCompletion;
use nixe_gpu::ExecutionBatch;
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
    type Commands = nixe_gpu::native::NativeSubmission;
    type ImageRequest = nixe_gpu::native::NativePresentationImage;
    fn prepare(
        &mut self,
        _batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<bool, BackendRuntimeError> {
        if !self.preparation_ready.load(Ordering::Acquire) {
            if let Some(started) = self.started_wait.take() {
                started.send(()).unwrap();
            }
            return Ok(false);
        }
        Ok(true)
    }
    fn record(
        &mut self,
        _batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<(), BackendRuntimeError> {
        self.device
            .work_ticket()
            .map_err(|error| BackendRuntimeError::Backend(error.to_string().into()))?;
        self.memory
            .write_resolved(std::slice::from_ref(&self.effect), &[0xa5; 16])
            .map_err(|error| BackendRuntimeError::Memory(Arc::new(error)))?;
        Ok(())
    }
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        assert!(batch.commands.commands.iter().all(|op| matches!(
            op,
            nixe_gpu::native::NativeCommand::Ordering(
                CacheMaintenanceOperation::InvalidateDeviceReadCaches
            )
        )));
        let work = self
            .device
            .work_ticket()
            .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))?;
        if self.preparation_ready.load(Ordering::Acquire) {
            self.device
                .flush()
                .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))?;
        }
        self.pending.push_back((batch.commands.frontend, work));
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
    fn request_completion_wakeup(&mut self) -> Result<(), BackendRuntimeError> {
        self.completion_wakeups += 1;
        Ok(())
    }
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<AccessResult, BackendRuntimeError> {
        self.memory
            .apply_resolved_access(&ranges, &operation)
            .map_err(|e| BackendRuntimeError::Memory(Arc::new(e)))
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
    fn progress_device_accesses(&mut self) -> Result<(), BackendRuntimeError> {
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
    let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
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
    })));
    let mut owner = Owner::new(&receiver, Rc::clone(&backend), &budget);
    let execution = MaxwellBackendExecution::<nixe_gpu::native::NativeWork>::begin(
        FrontendSubmissionId::new(1),
        None,
    );
    let mut work = GpuWork {
        clock: nixe_runtime::VirtualClock::default(),
        descriptor: NvDrvDeviceDescriptor::open(
            NvDrvFileDescriptor::new(1),
            NvDrvDeviceKind::HostGpu,
            NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
            NvDrvPermissionProfile::Application,
        ),
        request: 0xc018_481b,
        execution,
        reservation: None,
        control: Arc::new(Mutex::new(NvHostControl::default())),
        _permit: permit,
    };
    work.execution.append_steps(vec![
        nixe_gpu_maxwell::MaxwellSubmissionExecutionStep::Ordering(
            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
        ),
    ]);
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
    assert!(owner.consume(&mut work).unwrap().is_none());
    assert_eq!(
        backend.borrow().as_ref().unwrap().completion_wakeups,
        0,
        "compiler readiness and admitted RAM access must not submit unrelated work"
    );
    assert!(work.execution.waiting_for_input());
    let (request, mut warm_reply) = memory
        .prepare_device_access(vec![input], AccessOperation::Read)
        .unwrap();
    warm_sender.send(GpuMessage::MemoryAccess(request)).unwrap();
    work.execution.append_steps(vec![
        nixe_gpu_maxwell::MaxwellSubmissionExecutionStep::Ordering(
            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
        ),
    ]);
    assert!(owner.consume(&mut work).unwrap().is_none());
    assert!(matches!(warm_reply.poll().unwrap().unwrap(),
        AccessResult::Read(bytes) if bytes.as_ref() == [0xa5; 16]));
    assert!(
        !work.execution.is_finished(),
        "the next guest command has not arrived"
    );
    work.execution.seal();
    let token = owner.consume(&mut work).unwrap().unwrap();
    assert!(work.execution.is_finished());
    owner.pending.push_back(PendingWork { token, work });
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !owner.pending.is_empty() {
        owner.poll().unwrap();
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
