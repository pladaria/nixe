use super::*;
use nixe_gpu::BackendExecutionCompletion;
use nixe_gpu::ExecutionBatch;
use nixe_gpu::{BackendRuntimeError, CacheMaintenanceOperation};
use nixe_gpu_vulkan::{ApiPreference, VulkanDevice, WorkTicket};
use nixe_memory::{CanonicalPageId, DeviceVisibilityPoint, NonCpuDeviceId};
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
    device: VulkanDevice,
    pending: VecDeque<(FrontendSubmissionId, WorkTicket)>,
    preparation_ready: Arc<AtomicBool>,
    started_wait: Option<mpsc::Sender<()>>,
    completed: mpsc::Sender<BackendExecutionCompletion>,
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
        let completion = BackendExecutionCompletion::new(
            frontend,
            work.submission(),
            DeviceVisibilityPoint::new(0),
        );
        self.completed.send(completion).unwrap();
        Some(completion)
    }
    fn wait(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        if let Some(started) = self.started_wait.take() {
            started.send(()).unwrap();
        }
        if !self.preparation_ready.load(Ordering::Acquire) {
            thread::sleep(timeout);
            return Ok(None);
        }
        if let Some((_, work)) = self.pending.front() {
            self.device
                .wait_for_work(*work, timeout)
                .map_err(|e| BackendRuntimeError::Backend(e.to_string().into()))?;
        }
        Ok(self.complete())
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
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.wait(Duration::MAX)
    }
    fn wait_for_completion_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.wait(timeout)
    }
    fn bind_visibility_requester(
        &mut self,
        _: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError> {
        // An independent visibility request must be routed while preparation
        // is blocked; this fixture has no device-written canonical pages.
        assert_eq!(request.visible_at, DeviceVisibilityPoint::new(0));
        Ok(vec![0x5a; request.size].into_boxed_slice())
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
fn native_gpu_completion_and_cpu_visibility_progress_on_the_command_owner() {
    use super::super::super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
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
        device,
        pending: VecDeque::new(),
        preparation_ready: Arc::clone(&preparation_ready),
        started_wait: Some(started),
        completed: completed_tx,
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
    let producer = thread::spawn(move || {
        waited.recv_timeout(Duration::from_secs(3)).unwrap();
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .send(GpuMessage::CpuVisibility {
                request: CpuVisibilityRequest {
                    page: CanonicalPageId::new(
                        nixe_memory::BackingStoreId::new(1),
                        nixe_memory::GuestPhysicalPageId::new(1),
                    ),
                    size: 16,
                    device: NonCpuDeviceId::new(1),
                    visible_at: DeviceVisibilityPoint::new(0),
                },
                reply,
            })
            .unwrap();
        assert_eq!(
            result
                .recv_timeout(Duration::from_secs(3))
                .unwrap()
                .unwrap()
                .as_ref(),
            &[0x5a; 16]
        );
        preparation_ready.store(true, Ordering::Release);
        sender.send(GpuMessage::Wake).unwrap();
    });
    assert!(owner.consume(&mut work).unwrap().is_none());
    assert!(work.execution.waiting_for_input());
    let (reply, warm_reply) = mpsc::sync_channel(1);
    warm_sender
        .send(GpuMessage::CpuVisibility {
            request: CpuVisibilityRequest {
                page: CanonicalPageId::new(
                    nixe_memory::BackingStoreId::new(1),
                    nixe_memory::GuestPhysicalPageId::new(2),
                ),
                size: 16,
                device: NonCpuDeviceId::new(1),
                visible_at: DeviceVisibilityPoint::new(0),
            },
            reply,
        })
        .unwrap();
    work.execution.append_steps(vec![
        nixe_gpu_maxwell::MaxwellSubmissionExecutionStep::Ordering(
            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
        ),
    ]);
    assert!(owner.consume(&mut work).unwrap().is_none());
    assert_eq!(
        warm_reply.try_recv().unwrap().unwrap().as_ref(),
        &[0x5a; 16]
    );
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
