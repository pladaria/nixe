use super::*;
use crate::test_device as support;

fn small_device(
    api: crate::ApiPreference,
    configuration: QueueConfiguration,
) -> Option<VulkanDevice> {
    let mut device = support::device_for_test(api)?;
    device.submission = QueueState::new(
        Arc::clone(&device.owner),
        device.queues().graphics_compute_family,
        configuration,
        &device.dispatch,
    )
    .unwrap();
    Some(device)
}

mod gate {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/gate.rs"
    ));
}
use gate::Gate;

#[test]
#[ignore = "Explicit diagnostic: delays the native submit thread to measure zero-time polling"]
fn profile_zero_timeout_submission_acknowledgement() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let ticket = device.enqueue_buffer_readback(&[13; 64]).unwrap();
    let queue_mutex = Arc::clone(&device.owner.queue_mutex);
    let (held, ready) = std::sync::mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _guard = queue_mutex.lock().unwrap();
            held.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(150));
        });
        ready.recv().unwrap();
        device.flush().unwrap();
        let begin = std::time::Instant::now();
        let completed = device.wait_for_work(ticket.work(), Duration::ZERO).unwrap();
        eprintln!(
            "zero-time-poll elapsed_ms={} completed={completed}",
            begin.elapsed().as_secs_f64() * 1000.0,
        );
    });
    assert!(device.wait_for_work(ticket.work(), Duration::MAX).unwrap());
    assert_eq!(device.readback(ticket).unwrap(), [13; 64]);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn queued_zero_query_and_finite_wait_do_not_drain_the_recording_worker() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let work = device.work_ticket().unwrap();
    assert_eq!(device.poll_work(work).unwrap(), WorkStatus::Pending);
    assert_eq!(device.queue_usage().scheduled_tick, 0);
    let queue_mutex = Arc::clone(&device.owner.queue_mutex);
    let (held, ready) = std::sync::mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _guard = queue_mutex.lock().unwrap();
            held.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(200));
        });
        ready.recv().unwrap();
        device.flush().unwrap();
        let begin = Instant::now();
        assert!(!device.wait_for_work(work, Duration::ZERO).unwrap());
        assert!(begin.elapsed() < Duration::from_millis(50));
        assert_eq!(device.queue_usage().scheduled_tick, work.tick());
        assert_eq!(device.queue_usage().submitted_tick, 0);
        let begin = Instant::now();
        assert!(
            !device
                .wait_for_work(work, Duration::from_millis(20))
                .unwrap()
        );
        assert!(begin.elapsed() < Duration::from_millis(100));
    });
    assert!(device.wait_for_work(work, Duration::from_secs(5)).unwrap());
    drop(device);
    validation.check().unwrap();
}

#[test]
fn finite_wait_shares_one_deadline_across_submission_and_gpu_execution() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut gate = Gate::new(&device);
    let work = device.work_ticket().unwrap();
    let queue_mutex = Arc::clone(&device.owner.queue_mutex);
    let (held, ready) = std::sync::mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _guard = queue_mutex.lock().unwrap();
            held.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(120));
        });
        ready.recv().unwrap();
        unsafe {
            device.submission.flush(
                device.queue,
                &device.dispatch,
                &[vk::SemaphoreSubmitInfo::default()
                    .semaphore(gate.handle)
                    .value(1)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)],
                &[],
            )
        }
        .unwrap();
        gate.last_use = work.tick();
        let begin = Instant::now();
        assert!(
            !device
                .wait_for_work(work, Duration::from_millis(180))
                .unwrap()
        );
        let elapsed = begin.elapsed();
        eprintln!(
            "combined-deadline elapsed_ms={}",
            elapsed.as_secs_f64() * 1000.0
        );
        assert!(elapsed >= Duration::from_millis(170));
        assert!(elapsed < Duration::from_millis(260));
        // Under contention the deadline may expire before native submission.
        // A timeout bounds the wait; it does not guarantee worker progress.
        assert_eq!(device.completed_tick(), 0);
        gate.signal();
    });
    assert!(device.wait_for_work(work, Duration::from_secs(5)).unwrap());
    assert_eq!(device.queue_usage().submitted_tick, work.tick());
    drop(gate);
    drop(device);
    validation.check().unwrap();
}

#[test]
#[ignore = "Explicit physical-GPU throughput diagnostic, outside game measurement windows"]
fn profile_native_transfer_roundtrip() {
    let Some(mut device) = support::device_for_test_with_validation(
        crate::ApiPreference::Auto,
        crate::ValidationMode::Disabled,
    ) else {
        return;
    };
    let bytes: Vec<_> = (0..1024 * 1024).map(|i| (i * 37) as u8).collect();
    assert_eq!(device.transfer_round_trip(&bytes).unwrap(), bytes);
    let begin = std::time::Instant::now();
    for _ in 0..64 {
        assert_eq!(device.transfer_round_trip(&bytes).unwrap(), bytes);
    }
    eprintln!(
        "native-roundtrip iterations=64 bytes_per_iteration={} seconds={}",
        bytes.len(),
        begin.elapsed().as_secs_f64(),
    );
}

fn exercise_reuse(api: crate::ApiPreference) {
    let configuration = QueueConfiguration {
        command_slots: 2,
        arena_bytes: 1024,
        overflow_bytes: 2048,
    };
    let Some(mut device) = small_device(api, configuration) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut gate = Gate::new(&device);
    let bytes: Vec<_> = (0..256).map(|i| (i * 19 + 3) as u8).collect();
    let first = device.enqueue_buffer_readback(&bytes).unwrap();
    let first_work = first.work();
    assert_eq!(device.queue_usage().submitted_tick, 0);
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let accepted = unsafe {
        device
            .submission
            .flush(device.queue, &device.dispatch, &waits, &[])
    }
    .unwrap()
    .unwrap();
    gate.last_use = accepted.tick();
    assert_eq!(accepted, first_work);
    assert_eq!(accepted.submission().slot(), accepted.tick());
    let second = device
        .enqueue_image_readback(8, 8, &bytes, AllocationMode::Dedicated)
        .unwrap();
    let second_work = device.flush().unwrap().unwrap();
    gate.last_use = second_work.tick();
    assert!(!device.wait_for_work(second_work, Duration::ZERO).unwrap());
    assert_eq!(device.poll_progress().unwrap(), 0);
    let usage = device.queue_usage();
    assert_eq!(
        (
            usage.command_pools,
            usage.retired_buffers,
            usage.retired_images
        ),
        (2, 1, 1)
    );
    assert!(
        matches!(device.enqueue_buffer_readback(&bytes), Err(VulkanError::QueuePressure { pending: Some(ticket), .. }) if ticket == first_work)
    );
    let pools: Vec<_> = device
        .submission
        .slots
        .iter()
        .map(|slot| slot.pool)
        .collect();
    gate.signal();
    assert!(
        device
            .wait_for_work(second_work, Duration::from_secs(5))
            .unwrap()
    );
    let usage = device.queue_usage();
    assert_eq!(
        (
            usage.retired_buffers,
            usage.retired_images,
            usage.retained_readbacks
        ),
        (0, 0, 2)
    );
    assert!(matches!(
        device.enqueue_buffer_readback(&bytes),
        Err(VulkanError::QueuePressure { pending: None, .. })
    ));
    let stale = ReadbackTicket {
        work: first.work,
        slot: first.slot,
        entry: first.entry,
    };
    assert_eq!(device.readback(first).unwrap(), bytes);
    assert_eq!(device.readback(second).unwrap(), bytes);
    for cycle in 0..12 {
        let pattern: Vec<_> = bytes.iter().map(|byte| byte.wrapping_add(cycle)).collect();
        let readback = device.enqueue_buffer_readback(&pattern).unwrap();
        assert_eq!(device.readback(readback).unwrap(), pattern);
        assert_eq!(device.queue_usage().command_pools, 2);
        assert_eq!(device.queue_usage().arena_capacity, 4096);
        assert_eq!(
            device
                .submission
                .slots
                .iter()
                .map(|slot| slot.pool)
                .collect::<Vec<_>>(),
            pools
        );
    }
    assert!(device.readback(stale).is_err());
    // Seal a final signal explicitly; waiting only observes its real completion.
    let final_work = device.work_ticket().unwrap();
    assert!(!device.wait_for_work(final_work, Duration::ZERO).unwrap());
    assert_eq!(device.flush().unwrap(), Some(final_work));
    assert!(
        device
            .wait_for_work(final_work, Duration::from_secs(5))
            .unwrap()
    );
    assert_eq!(device.queue_usage().completed_tick, final_work.tick());
    drop(gate);
    device.shutdown().unwrap();
    assert!(matches!(
        device.work_ticket(),
        Err(VulkanError::QueueStopped)
    ));
    drop(device);
    validation.check().unwrap();
}

#[test]
fn core_timeline_drives_retirement_reuse_and_a_final_fence() {
    exercise_reuse(crate::ApiPreference::Auto);
}
#[test]
fn khr_timeline_drives_retirement_reuse_and_a_final_fence() {
    exercise_reuse(crate::ApiPreference::Compatibility12);
}

#[test]
fn native_recording_waits_for_actual_slot_capacity_without_growing_the_budget() {
    let Some(mut device) = small_device(
        crate::ApiPreference::Auto,
        QueueConfiguration {
            command_slots: 1,
            arena_bytes: 512,
            overflow_bytes: 1024,
        },
    ) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut gate = Gate::new(&device);
    let first = device.work_ticket().unwrap();
    unsafe {
        device.submission.flush(
            device.queue,
            &device.dispatch,
            &[vk::SemaphoreSubmitInfo::default()
                .semaphore(gate.handle)
                .value(1)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)],
            &[],
        )
    }
    .unwrap()
    .unwrap();
    gate.last_use = first.tick();
    assert!(
        matches!(device.submission.begin(), Err(VulkanError::QueuePressure { pending: Some(work), .. }) if work == first)
    );
    let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let signal = Arc::clone(&released);
    let producer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(25));
        signal.store(true, std::sync::atomic::Ordering::Release);
        gate.signal();
    });
    let _slot = device.begin_recording().unwrap();
    assert!(
        released.load(std::sync::atomic::Ordering::Acquire),
        "capacity cannot recycle before the actual GPU dependency"
    );
    assert_eq!(device.queue_usage().command_pools, 1);
    let next = device.work_ticket().unwrap();
    assert!(next.tick() > first.tick());
    device.flush().unwrap();
    device.wait_for_work(next, Duration::from_secs(5)).unwrap();
    producer.join().unwrap();
    drop(device);
    validation.check().unwrap();
}

#[test]
fn staging_pressure_is_bounded_and_unread_bytes_survive_completion() {
    let Some(mut device) = small_device(
        crate::ApiPreference::Auto,
        QueueConfiguration {
            command_slots: 1,
            arena_bytes: 512,
            overflow_bytes: 1024,
        },
    ) else {
        return;
    };
    let validation = device.validation_monitor();
    let first = device.enqueue_buffer_readback(&[0x19; 512]).unwrap();
    let second = device.enqueue_buffer_readback(&[0xa7; 512]).unwrap();
    assert_eq!(device.queue_usage().overflow_capacity, 1024);
    assert!(matches!(
        device.enqueue_buffer_readback(&[0x00; 512]),
        Err(VulkanError::QueuePressure { .. })
    ));
    let work = device.flush().unwrap().unwrap();
    assert!(device.wait_for_work(work, Duration::from_secs(5)).unwrap());
    assert_eq!(device.queue_usage().overflow_capacity, 512);
    assert!(
        matches!(
            device.begin_recording(),
            Err(VulkanError::QueuePressure {
                pending: None,
                retained_readbacks: 2
            })
        ),
        "a GPU wait cannot release unread CPU bytes"
    );
    assert_eq!(device.readback(first).unwrap(), [0x19; 512]);
    assert!(matches!(
        device.enqueue_buffer_readback(&[0x00; 512]),
        Err(VulkanError::QueuePressure { .. })
    ));
    assert_eq!(device.readback(second).unwrap(), [0xa7; 512]);
    assert_eq!(device.queue_usage().overflow_capacity, 0);
    assert!(matches!(
        device.enqueue_buffer_readback(&[0; 2048]),
        Err(VulkanError::StagingRequestTooLarge { .. })
    ));
    assert_eq!(
        device.transfer_round_trip(&[0x5c; 257]).unwrap(),
        [0x5c; 257]
    );
    drop(device);
    validation.check().unwrap();
}

#[test]
fn shutdown_drains_accepted_work_and_cancels_unsubmitted_work() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let accepted = device.enqueue_buffer_readback(&[1; 64]).unwrap();
    let submitted = device.flush().unwrap().unwrap();
    assert!(
        device
            .submission
            .progress
            .wait_submitted(submitted.tick(), None)
            .unwrap()
    );
    let cancelled = device.enqueue_buffer_readback(&[2; 64]).unwrap();
    assert!(cancelled.work().tick() > submitted.tick());
    device.shutdown().unwrap();
    assert_eq!(device.queue_usage().submitted_tick, submitted.tick());
    assert_eq!(device.queue_usage().completed_tick, submitted.tick());
    assert!(device.readback(cancelled).is_err());
    assert!(device.discard_readback(accepted).is_err());
    assert_eq!(device.queue_usage().retained_readbacks, 0);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn foreign_tickets_and_terminal_device_errors_never_publish_completion() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let foreign = WorkTicket {
        instance: BackendInstanceId::new(u64::MAX),
        tick: 1,
    };
    assert!(device.wait_for_work(foreign, Duration::ZERO).is_err());
    // Exercise loss bookkeeping without pretending to fault-inject the driver.
    let error = device
        .submission
        .error("loss fixture", vk::Result::ERROR_DEVICE_LOST);
    assert!(
        matches!(error, VulkanError::SubmissionFailed { tick: 1, cause }
        if matches!(*cause, VulkanError::Vulkan { result: vk::Result::ERROR_DEVICE_LOST, .. }))
    );
    assert_eq!(device.queue_usage().completed_tick, 0);
    assert!(matches!(
        device.work_ticket(),
        Err(VulkanError::QueueStopped)
    ));
    drop(device);
    validation.check().unwrap();
}

#[test]
fn staging_overflow_suballocates_small_transfers_and_preserves_unread_slices() {
    let Some(mut device) = small_device(
        crate::ApiPreference::Auto,
        QueueConfiguration {
            command_slots: 1,
            arena_bytes: 4096,
            overflow_bytes: 16384,
        },
    ) else {
        return;
    };
    let validation = device.validation_monitor();
    // Fill the primary arenas; subsequent small requests must share overflow
    // arenas rather than allocating one native buffer and memory per request.
    device.submission.reserve_upload(&[0; 4096]).unwrap();
    let (_, _, unused) = device.submission.reserve_readback(4096).unwrap();
    let mut tickets = Vec::new();
    for value in 0..128u8 {
        let (index, source) = device.submission.reserve_upload(&[value; 64]).unwrap();
        let (_, destination, ticket) = device.submission.reserve_readback(64).unwrap();
        let slot = &device.submission.slots[index];
        unsafe {
            device.dispatch.cmd_copy_buffer(
                slot.command,
                slot.upload_buffer(source).handle,
                slot.readback_buffer(destination).handle,
                &[vk::BufferCopy::default()
                    .src_offset(source.offset)
                    .dst_offset(destination.offset)
                    .size(64)],
            );
        }
        tickets.push(ticket);
    }
    assert_eq!(device.submission.slots[0].upload_overflow.len(), 2);
    assert_eq!(device.submission.slots[0].readback_overflow.len(), 2);
    assert_eq!(device.queue_usage().overflow_capacity, 16384);
    device.discard_readback(unused).unwrap();
    for (value, ticket) in tickets.into_iter().enumerate() {
        assert_eq!(device.readback(ticket).unwrap(), [value as u8; 64]);
    }
    assert_eq!(device.queue_usage().overflow_capacity, 0);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn prior_native_use_retires_after_a_later_failure_without_submitting_dependent_ticks() {
    use crate::queue::progress::NativeFailure;
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let a = device.work_ticket().unwrap();
    device.flush().unwrap();
    device
        .submission
        .progress
        .wait_submitted(a.tick(), None)
        .unwrap();
    let queue_mutex = Arc::clone(&device.owner.queue_mutex);
    let held = queue_mutex.lock().unwrap();
    let b = device.work_ticket().unwrap();
    device.flush().unwrap();
    let c = device.work_ticket().unwrap();
    device.flush().unwrap();
    // Inject the scheduler outcome, without claiming to fault the native driver.
    device.submission.progress.fail(
        b.tick(),
        NativeFailure::Vulkan {
            operation: "injected native submit failure",
            result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        },
    );
    assert!(device.wait_for_work(a, Duration::from_secs(5)).unwrap());
    for work in [b, c] {
        assert!(matches!(device.wait_for_work(work, Duration::ZERO),
            Err(VulkanError::SubmissionFailed { tick, .. }) if tick == b.tick()));
    }
    drop(held);
    assert!(device.shutdown().is_err());
    assert_eq!(device.queue_usage().submitted_tick, a.tick());
    assert_eq!(device.queue_usage().completed_tick, a.tick());
    assert_eq!(device.queue_usage().command_pools, 0);
    drop(device);
    validation.check().unwrap();
}
