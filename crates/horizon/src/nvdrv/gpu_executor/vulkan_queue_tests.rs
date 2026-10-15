//! Real Vulkan completion through the statically dispatched Horizon owner.
//! The fixture consumes only a resource-free cache-maintenance operation; it is
//! not a production GpuExecution implementation or a partial game renderer.
use super::*;
use nixe_gpu::ExecutionBatch;
use nixe_gpu::{BackendRuntimeError, CacheMaintenanceOperation, GpuCommand};
use nixe_gpu_vulkan::{ApiPreference, VulkanDevice, WorkTicket};
use nixe_memory::{CanonicalPageId, DeviceVisibilityPoint, NonCpuDeviceId};
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
    presented: Option<mpsc::Sender<Option<FrontendSubmissionId>>>,
}
impl QueueFixture {
    fn complete(&mut self) -> Option<BackendExecutionCompletion> {
        // Keep the real submission outstanding until the ordered scanout has
        // observed it: presentation needs queue order, not a host completion wait.
        if self.presented.is_some() {
            return None;
        }
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
    type Commands = nixe_gpu::OperationSubmission;
    type ImageRequest = nixe_gpu::PresentationImageRequest;
    fn prepare(&mut self, _batch: ExecutionBatch<'_>) -> Result<bool, BackendRuntimeError> {
        if !self.preparation_ready.load(Ordering::Acquire) {
            if let Some(started) = self.started_wait.take() {
                started.send(()).unwrap();
            }
            return Ok(false);
        }
        Ok(true)
    }
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        assert!(batch.definitions.is_empty() && batch.retirements.is_empty());
        assert!(batch.commands.operations().iter().all(|op| matches!(
            op.command(),
            GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateDeviceReadCaches)
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
        self.pending.push_back((batch.commands.id(), work));
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
        _: PresentationImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        if let Some(presented) = self.presented.take() {
            presented
                .send(self.pending.back().map(|(frontend, _)| *frontend))
                .unwrap();
        }
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
fn real_gpu_final_completion_and_visibility_progress_while_preparation_is_pending() {
    use super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let preparation_ready = Arc::new(AtomicBool::new(false));
    let _release_preparation = ReleasePreparationOnDrop(Arc::clone(&preparation_ready));
    let budget = Arc::new(GpuWorkBudget::default());
    let work = |id, permit| {
        let mut execution = MaxwellBackendExecution::<nixe_gpu_maxwell::MaxwellLoweredWork>::begin(
            FrontendSubmissionId::new(id),
            None,
        );
        execution.append_steps(vec![MaxwellSubmissionExecutionStep::BackendOperation(
            nixe_gpu::GpuOperation::new(
                GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateDeviceReadCaches),
                [],
                [],
                nixe_gpu::CapabilityRequirements::none(),
            ),
        )]);
        execution.seal();
        GpuExecutorMessage::Submission(GpuWork {
            clock: nixe_runtime::VirtualClock::default(),
            descriptor: NvDrvDeviceDescriptor::open(
                NvDrvFileDescriptor::new(1),
                NvDrvDeviceKind::HostGpu,
                NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
                NvDrvPermissionProfile::Application,
            ),
            request: 0xc018_481b,
            execution: Box::new(execution),
            pending: VecDeque::new(),
            preparing: None,
            queued_chunks: VecDeque::new(),
            reservation: None,
            control: Arc::new(Mutex::new(NvHostControl::default())),
            permit,
        })
    };
    let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
        panic!()
    };
    let (sender, receiver) = mpsc::channel();
    sender.send(work(1, permit)).unwrap();
    let (started, waited) = mpsc::channel();
    let (complete, completed) = mpsc::channel();
    let (present, presented) = mpsc::channel();
    let ready = Arc::clone(&preparation_ready);
    let owner_budget = Arc::clone(&budget);
    let owner = thread::spawn(move || {
        let mut backend = Some(QueueFixture {
            device,
            pending: VecDeque::new(),
            preparation_ready: ready,
            started_wait: Some(started),
            completed: complete,
            presented: Some(present),
        });
        let result = run_gpu_owner(receiver, &mut backend, &owner_budget);
        backend.as_mut().unwrap().teardown().unwrap();
        result
    });
    // Send visibility after cold preparation has become pending. No host
    // submission is accepted until readiness, and the owner must stay responsive.
    waited.recv_timeout(Duration::from_secs(3)).unwrap();
    // A scanout request follows the accepted frontend delivery but must not
    // capture the previous contents while its first pipeline is still pending.
    let allocation = nixe_memory::CanonicalAllocation::zeroed(4096, 4096).unwrap();
    let backing = allocation
        .backing_range(nixe_memory::MemoryPermissions::READ_WRITE)
        .unwrap();
    let (reply, scanout) = mpsc::sync_channel(1);
    sender
        .send(GpuExecutorMessage::PresentImage {
            request: nixe_gpu::PresentationImageRequest {
                allow_canonical_import: true,
                cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(&backing).unwrap(),
                backing: nixe_gpu::BackingView::new(
                    nixe_gpu::GpuAllocationId::new(1),
                    nixe_gpu::GpuAllocationDescription::new(4096, 4096).unwrap(),
                    0,
                    backing,
                )
                .unwrap(),
                width: 1,
                height: 1,
                format: nixe_gpu::PresentationImageFormat::Rgba8,
                layout: nixe_gpu::ImageMemoryLayout::PitchLinear {
                    row_pitch: 4,
                    layer_stride: 4,
                },
                row_pitch: 4,
            },
            reply,
        })
        .unwrap();
    let early_scanout = scanout.recv_timeout(Duration::from_millis(30));
    // Saturate queued byte admission; independent visibility must still reach
    // this owner while frontend preparation is pending.
    let pressure = budget.handoff.reserve(64 * 1024 * 1024).unwrap();
    let (reply, visible) = mpsc::sync_channel(1);
    sender
        .send(GpuExecutorMessage::CpuVisibility {
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
    let visibility = visible.recv_timeout(Duration::from_secs(3));
    // Unblock even if an assertion fails, so teardown cannot hang the test.
    let early_completion = completed.try_recv();
    preparation_ready.store(true, Ordering::Release);

    // No next delivery or wake is sent. The owner must submit/wait its final
    // real Vulkan tick and return the matching guest completion record itself.
    let completion = completed.recv_timeout(Duration::from_secs(3));
    let captured_after = presented.recv_timeout(Duration::from_secs(3));
    budget.stop();
    sender.send(GpuExecutorMessage::Wake).unwrap();
    owner.join().unwrap().unwrap();
    drop(pressure);
    assert!(matches!(
        early_scanout,
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(captured_after.unwrap(), Some(FrontendSubmissionId::new(1)));
    assert!(
        scanout
            .recv_timeout(Duration::from_secs(3))
            .unwrap()
            .is_err()
    );
    assert!(matches!(early_completion, Err(mpsc::TryRecvError::Empty)));
    assert_eq!(visibility.unwrap().unwrap().as_ref(), &[0x5a; 16]);
    let completion = completion.unwrap();
    assert_eq!(completion.frontend(), FrontendSubmissionId::new(1));
    assert_eq!(completion.submission().slot(), 1);
    validation.check().unwrap();
}
