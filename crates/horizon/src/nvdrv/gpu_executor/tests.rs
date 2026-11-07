use super::*;

#[test]
fn channel_acceptance_stays_ordered_until_its_submission_is_queued() {
    use super::super::{NV_SUCCESS, NvDrvInlineBuffer, NvDrvIoctlCaller, NvDrvSession};
    use std::time::{Duration, Instant};

    let session = NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
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
    let budget = Arc::new(GpuWorkBudget::new(&crate::test_memory::kernel()));
    let mut permits = Vec::new();
    for _ in 0..MAX_GPU_SUBMISSIONS_IN_FLIGHT {
        let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve(0) else {
            panic!("capacity available");
        };
        permits.push(permit);
    }
    assert!(matches!(
        budget.reserve(0),
        Some(GpuSubmissionAdmission::Pending(_))
    ));
    permits.pop();
    assert!(budget.readable.is_signalled());
    let Some(GpuSubmissionAdmission::Ready(permit)) = budget.reserve(0) else {
        panic!("released capacity");
    };
    permits.push(permit);
    budget.stop();
    assert!(budget.readable.is_signalled());
    assert!(budget.reserve(0).is_none());
    drop(permits);
    assert_eq!(budget.state.lock().unwrap().active, 0);
}
#[test]
fn progress_notifications_are_coalesced_without_a_capture_frontier() {
    let budget = GpuWorkBudget::new(&crate::test_memory::kernel());
    assert_eq!(budget.request_progress(), Some(true));
    assert_eq!(budget.request_progress(), Some(false));
    assert!(budget.take_progress_request());
    assert!(!budget.take_progress_request());
    budget.stop();
    assert!((budget.request_progress()).is_none());
}

#[test]
fn admission_accounts_owned_bytes_and_releases_them_before_device_completion() {
    let budget = Arc::new(GpuWorkBudget::new(&crate::test_memory::kernel()));
    let Some(GpuSubmissionAdmission::Ready(first)) = budget.reserve(MAX_GPU_INGRESS_BYTES - 8)
    else {
        panic!("initial capacity")
    };
    assert!(matches!(
        budget.reserve(9),
        Some(GpuSubmissionAdmission::Pending(_))
    ));
    let Some(GpuSubmissionAdmission::Ready(last)) = budget.reserve(8) else {
        panic!("exact remaining capacity")
    };
    assert_eq!(budget.state.lock().unwrap().bytes, MAX_GPU_INGRESS_BYTES);
    drop(first);
    assert_eq!(budget.state.lock().unwrap().bytes, 8);
    let Some(GpuSubmissionAdmission::Ready(replacement)) =
        budget.reserve(MAX_GPU_INGRESS_BYTES - 8)
    else {
        panic!("consumed descriptor storage reusable")
    };
    drop(last);
    drop(replacement);
    assert_eq!(budget.state.lock().unwrap().bytes, 0);
}
