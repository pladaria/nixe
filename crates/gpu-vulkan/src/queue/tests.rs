use super::*;
use crate::test_device as support;

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
                &mut device.dispatch,
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
    let Some(mut device) = support::device_for_test(api) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut pending = Vec::new();
    for byte in 0..16u8 {
        let read = device.enqueue_buffer_readback(&[byte; 128]).unwrap();
        device.flush().unwrap();
        pending.push((byte, read));
    }
    let last = device.preceding_work().unwrap();
    assert!(device.wait_for_work(last, Duration::from_secs(5)).unwrap());
    for (byte, read) in pending {
        assert_eq!(device.readback(read).unwrap(), [byte; 128]);
    }
    let read = device.enqueue_buffer_readback(&[0x37; 128]).unwrap();
    device.flush().unwrap();
    assert_eq!(device.readback(read).unwrap(), [0x37; 128]);
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
    assert_eq!(device.queue_usage().command_slots, 0);
    drop(device);
    validation.check().unwrap();
}

#[cfg(feature = "performance-counters")]
#[test]
fn full_chunks_record_without_submitting() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let first = device.work_ticket().unwrap();
    let queued_chunks = || {
        nixe_gpu::metrics::snapshot()
            .into_iter()
            .find(|(name, _)| *name == "NativeQueuedChunks")
            .unwrap()
            .1
    };
    let initial_chunks = queued_chunks();
    let mutex = Arc::clone(&device.owner.queue_mutex);
    let (held, ready) = std::sync::mpsc::sync_channel(1);
    let mut reads = Vec::new();
    std::thread::scope(|scope| {
        let blocked_worker = scope.spawn(move || {
            let _lock = mutex.lock().unwrap();
            held.send(()).unwrap();
            let deadline = Instant::now() + Duration::from_secs(3);
            // The native submit is held while recording chunks continue to queue.
            while queued_chunks() < initial_chunks + 2 {
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            true
        });
        ready.recv().unwrap();
        device.flush().unwrap();
        for index in 0..300 {
            let value = (index % 251) as u8;
            reads.push((value, device.enqueue_buffer_readback(&[value; 16]).unwrap()));
        }
        assert_eq!(
            device.scheduled_tick(),
            first.tick(),
            "full recording chunks must not seal a batch"
        );
        assert!(device.queue_usage().command_slots >= 2);
        device.flush().unwrap();
        assert!(
            blocked_worker.join().unwrap(),
            "the worker gate must observe queued recording chunks"
        );
    });
    for (value, read) in reads {
        assert_eq!(device.readback(read).unwrap(), [value; 16]);
    }
    drop(device);
    validation.check().unwrap();
}

#[test]
fn recording_chunk_fullness_does_not_seal_a_native_batch() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut reads = Vec::new();
    for value in 0..64u8 {
        reads.push((value, device.enqueue_buffer_readback(&[value; 16]).unwrap()));
    }
    assert_eq!(device.scheduled_tick(), 0);
    assert_eq!(device.queue_usage().submitted_tick, 0);
    assert!(
        reads
            .iter()
            .all(|(_, ticket)| ticket.work() == reads[0].1.work())
    );
    for (value, ticket) in reads {
        assert_eq!(device.readback(ticket).unwrap(), [value; 16]);
    }
    drop(device);
    validation.check().unwrap();
}

#[test]
fn command_contexts_grow_while_the_gpu_is_busy() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut gate = gate::Gate::new(&device);
    let first = device.work_ticket().unwrap();
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    unsafe {
        device
            .submission
            .flush(device.queue, &mut device.dispatch, &waits, &[])
            .unwrap();
    }
    let mut last = first;
    for _ in 0..24 {
        last = device.work_ticket().unwrap();
        device.flush().unwrap();
    }
    assert_eq!(device.completed_tick(), 0);
    assert!(device.queue_usage().command_slots >= 25);
    assert!(!device.wait_for_work(last, Duration::ZERO).unwrap());
    gate.signal();
    assert!(device.wait_for_work(last, Duration::from_secs(5)).unwrap());
    let capacity = device.queue_usage().command_slots;
    device.work_ticket().unwrap();
    device.flush().unwrap();
    assert_eq!(device.queue_usage().command_slots, capacity);
    drop(gate);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn deferred_cpu_readbacks_do_not_pin_completed_command_contexts() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut pending = Vec::new();
    for byte in 0..24u8 {
        let read = device.enqueue_buffer_readback(&[byte; 64]).unwrap();
        device.flush().unwrap();
        assert!(
            device
                .wait_for_work(read.work(), Duration::from_secs(5))
                .unwrap()
        );
        pending.push((byte, read));
    }
    assert_eq!(device.queue_usage().retained_readbacks, pending.len());
    assert_eq!(device.queue_usage().command_slots, 2);
    for (byte, read) in pending {
        assert_eq!(device.readback(read).unwrap(), [byte; 64]);
    }
    assert_eq!(device.queue_usage().retained_readbacks, 0);
    drop(device);
    validation.check().unwrap();
}
