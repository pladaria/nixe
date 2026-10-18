use super::*;

#[test]
fn admission_bounds_real_deliveries_and_teardown_wakes_capacity_waiters() {
    let budget = Arc::new(GpuWorkBudget::default());
    let mut permits = Vec::new();
    for _ in 0..MAX_GPU_SUBMISSIONS_IN_FLIGHT {
        let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
            panic!("capacity available");
        };
        permits.push(permit);
    }
    assert!(matches!(
        budget.reserve(),
        Some(GpuSubmissionAdmission::Pending(_))
    ));
    permits.pop();
    assert!(budget.readable.is_signalled());
    let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve() else {
        panic!("released capacity");
    };
    permits.push(permit);
    budget.stop();
    assert!(budget.readable.is_signalled());
    assert!(budget.reserve().is_none());
    drop(permits);
    assert_eq!(budget.state.lock().unwrap().active, 0);
}
#[test]
fn progress_notifications_are_coalesced_without_a_capture_frontier() {
    let budget = GpuWorkBudget::default();
    assert_eq!(budget.request_progress(), Some(true));
    assert_eq!(budget.request_progress(), Some(false));
    assert!(budget.take_progress_request());
    assert!(!budget.take_progress_request());
    budget.stop();
    assert_eq!(budget.request_progress(), None);
}
#[test]
fn presentation_cannot_overtake_a_frontend_delivery() {
    use super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    use nixe_gpu::{BackingView, GpuAllocationId, ImageMemoryLayout, PresentationImageFormat};
    use nixe_gpu_maxwell::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellChannelId,
        MaxwellChannelOwner, MaxwellGpfifoSubmitRequest, MaxwellGpuChannel, MaxwellScheduler,
        SWITCH_1_GM20B_PROFILE, decode_gpfifo_submission, resolve_gpfifo_submission,
    };
    use nixe_memory::{CanonicalAllocation, CanonicalCpuWriteDependency, MemoryPermissions};

    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    channel.bind_address_space(address_space.id()).unwrap();
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        8,
        MaxwellGpfifoSubmitRequest {
            entry_count: 0,
            flags: 4,
            fence_id: 0,
            fence_value: 0,
        },
        &[],
    )
    .unwrap();
    let retained = resolve_gpfifo_submission(
        &channel,
        FrontendSubmissionId::new(1),
        decoded,
        &address_space,
    )
    .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler.enqueue(&channel, retained, None, None).unwrap();
    let dispatch = scheduler
        .dispatch_next(true, &address_space)
        .unwrap()
        .unwrap();
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let cpu_writes = CanonicalCpuWriteDependency::capture(&backing).unwrap();
    let image = PresentationImageRequest {
        allow_canonical_import: false,
        backing: BackingView::new(
            GpuAllocationId::new(1),
            nixe_gpu::GpuAllocationDescription::new(0x1000, 0x1000).unwrap(),
            0,
            backing,
        )
        .unwrap(),
        width: 1,
        height: 1,
        format: PresentationImageFormat::Rgba8,
        layout: ImageMemoryLayout::PitchLinear {
            row_pitch: 4,
            layer_stride: 4,
        },
        row_pitch: 4,
        cpu_writes,
    };
    let executor = NvDrvGpuExecutor::new();
    let GpuSubmissionAdmission::Ready(permit) = executor.reserve_submission().unwrap() else {
        panic!()
    };
    let descriptor = NvDrvDeviceDescriptor::open(
        NvDrvFileDescriptor::new(1),
        NvDrvDeviceKind::HostGpu,
        NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
        NvDrvPermissionProfile::Application,
    );
    executor
        .enqueue(
            GpuSubmission::new(
                descriptor,
                0xc018_481b,
                dispatch,
                channel.configuration(),
                address_space,
                None,
                Arc::new(Mutex::new(NvHostControl::default())),
                permit,
            ),
            &nixe_runtime::VirtualClock::default(),
        )
        .unwrap();
    let presented = executor.request_presentable_image(image).unwrap();
    // Empty work fails at the frontend boundary. Its queued presentation
    // must never reach the backend or create a false resident producer.
    assert!(
        presented
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap()
            .is_err()
    );
    assert!(executor.teardown().unwrap_err().boundary.is_some());
}
