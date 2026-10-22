use super::*;

#[test]
fn channel_acceptance_stays_ordered_until_its_submission_is_queued() {
    use super::super::{NV_SUCCESS, NvDrvInlineBuffer, NvDrvIoctlCaller, NvDrvSession};
    use std::time::{Duration, Instant};

    let session = NvDrvSession::new(crate::test_memory::machine());
    session.initialize(0);
    let nvmap = session.open(b"/dev/nvmap", 1).unwrap();
    let space = session.open(b"/dev/nvhost-as-gpu", 1).unwrap();
    let channel = session.open(b"/dev/nvhost-gpu", 1).unwrap();
    session
        .ioctl(channel, 0x4004_4801, &nvmap.raw().to_le_bytes())
        .unwrap();
    session
        .ioctl(space, 0x4004_4101, &channel.raw().to_le_bytes())
        .unwrap();
    let mut allocate = [0; 32];
    allocate[..4].copy_from_slice(&8u32.to_le_bytes());
    assert_eq!(
        session.ioctl(channel, 0xc020_481a, &allocate).unwrap().1,
        NV_SUCCESS
    );
    let (gpu, executor) = {
        let state = session.state.lock().unwrap();
        (
            state.nvhost_gpu.clone(),
            state.gpu_backend.as_ref().unwrap().clone(),
        )
    };
    // Hold the real enqueue boundary, after capacity admission but before the
    // owner can observe this submission. No GPU or timing hook is required.
    let sender = executor.sender.lock().unwrap();
    let submitting = session.clone();
    let worker = thread::spawn(move || {
        let mut input = [0; 24];
        input[12..16].copy_from_slice(&4u32.to_le_bytes());
        submitting.ioctl_inner(
            channel,
            0xc018_481b,
            &input,
            NvDrvInlineBuffer::Input(&[]),
            None,
            NvDrvIoctlCaller {
                thread_id: 1,
                clock: &nixe_runtime::VirtualClock::default(),
            },
        )
    });
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut acceptance_was_released = false;
    let mut admission_started = false;
    while Instant::now() < deadline {
        admission_started |= executor.budget.state.lock().unwrap().active != 0;
        if admission_started
            && let Ok(guard) = gpu.try_lock()
            && guard.accepted_submission_count() != 0
        {
            acceptance_was_released = true;
            break;
        }
        thread::yield_now();
    }
    drop(sender);
    let accepted = worker.join().unwrap().unwrap();
    assert!(
        matches!(accepted, super::super::NvDrvIoctlOutcome::Complete(response)
        if response.driver_result == NV_SUCCESS)
    );
    assert!(admission_started, "submission never reached admission");
    assert!(
        !acceptance_was_released,
        "another ioctl could accept a later fence before the first submission was queued"
    );
}

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
    assert!((budget.request_progress()).is_none());
}
#[test]
fn presentation_cannot_overtake_a_frontend_delivery() {
    use super::super::{
        NvDrvDescriptorOwner, NvDrvDeviceKind, NvDrvFileDescriptor, NvDrvPermissionProfile,
        NvDrvSessionId,
    };
    use nixe_gpu::physical::PhysicalMapping;
    use nixe_gpu::{BackingView, GpuAllocationId, ImageMemoryLayout, PresentationImageFormat};
    use nixe_gpu_maxwell::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellChannelId,
        MaxwellChannelOwner, MaxwellGpfifoSubmitRequest, MaxwellGpuAddressSpace, MaxwellGpuChannel,
        MaxwellScheduler, SWITCH_1_GM20B_PROFILE, decode_gpfifo_submission,
        resolve_gpfifo_submission,
    };
    use nixe_memory::{AllocationRange, Machine};

    let machine = Machine::new(4096).unwrap();
    let mut address_space = MaxwellGpuAddressSpace::new(
        machine.clone(),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
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
        &channel.configuration(),
        FrontendSubmissionId::new(1),
        decoded,
        address_space.memory(),
    )
    .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), retained, None, None)
        .unwrap();
    let dispatch = scheduler
        .dispatch_next(true, address_space.memory())
        .unwrap()
        .unwrap();
    let allocation = machine.account(4096).unwrap().allocate(4096).unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, 4096).unwrap();
    let backing = PhysicalMapping::new(std::slice::from_ref(&input)).unwrap();
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
        inputs: vec![input],
    };
    let executor = NvDrvGpuExecutor::new(machine);
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
                address_space.memory().clone(),
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
