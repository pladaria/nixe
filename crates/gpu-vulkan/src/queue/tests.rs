use super::*;
mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

fn small_device(
    api: crate::ApiPreference,
    configuration: QueueConfiguration,
) -> Option<VulkanDevice> {
    let mut device = support::device_for_test(api)?;
    device.submission = QueueState::new(
        Arc::clone(&device.owner),
        device.queues().graphics_compute_family,
        configuration,
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
    // The final signal has no following data/message/explicit flush. Waiting
    // submits that exact recording ticket and observes its own real completion.
    let final_work = device.work_ticket().unwrap();
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
    assert!(matches!(
        error,
        VulkanError::Vulkan {
            result: vk::Result::ERROR_DEVICE_LOST,
            ..
        }
    ));
    assert_eq!(device.queue_usage().completed_tick, 0);
    assert!(matches!(
        device.work_ticket(),
        Err(VulkanError::QueueStopped)
    ));
    drop(device);
    validation.check().unwrap();
}
