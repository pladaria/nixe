use super::*;
use crate::test_device as support;

#[test]
fn consuming_an_incomplete_readback_never_submits_or_waits() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let ticket = device.enqueue_buffer_readback(&[0x37; 64]).unwrap();
    let work = ticket.work();
    assert!(ticket.completed().is_err());
    assert_eq!(device.scheduled_tick(), 0);
    assert_eq!(device.completed_tick(), 0);
    assert_eq!(device.queue_usage().retained_readbacks, 0);
    device.flush().unwrap();
    device.wait_for_work(work, Duration::MAX).unwrap();
    drop(device);
    validation.check().unwrap();
}

#[test]
fn readback_reservation_can_be_consumed_by_a_worker_after_recorder_shutdown() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let ticket = device.enqueue_buffer_readback(&[0x42; 128]).unwrap();
    device.flush().unwrap();
    device.wait_for_work(ticket.work(), Duration::MAX).unwrap();
    let (start, ready) = std::sync::mpsc::channel();
    let consumer = std::thread::spawn(move || {
        ready.recv().unwrap();
        let staging = ticket.completed().unwrap();
        assert_eq!(staging.as_ref(), [0x42; 128]);
    });
    device.shutdown().unwrap();
    drop(device);
    start.send(()).unwrap();
    consumer.join().unwrap();
    validation.check().unwrap();
}

#[test]
fn mapped_readback_owns_staging_through_reuse_and_shutdown() {
    let Some(mut device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let ticket = device.enqueue_buffer_readback(&[0x37; 64]).unwrap();
    device.flush().unwrap();
    device.wait_for_work(ticket.work(), Duration::MAX).unwrap();
    let mapped = ticket.completed().unwrap();
    for byte in 0..16 {
        let next = device.enqueue_buffer_readback(&[byte; 64]).unwrap();
        device.flush().unwrap();
        device.wait_for_work(next.work(), Duration::MAX).unwrap();
        assert_eq!(next.completed().unwrap().as_ref(), [byte; 64]);
        assert_eq!(mapped.as_ref(), [0x37; 64]);
        assert_eq!(device.queue_usage().retained_readbacks, 1);
    }
    device.shutdown().unwrap();
    drop(device);
    assert_eq!(mapped.as_ref(), [0x37; 64]);
    drop(mapped);
    validation.check().unwrap();
}

#[cfg(feature = "performance-counters")]
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
        device.flush().unwrap();
        device.wait_for_work(ticket.work(), Duration::MAX).unwrap();
        assert_eq!(ticket.completed().unwrap().as_ref(), [value; 16]);
    }
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
        device.flush().unwrap();
        device.wait_for_work(read.work(), Duration::MAX).unwrap();
        assert_eq!(read.completed().unwrap().as_ref(), [byte; 64]);
    }
    assert_eq!(device.queue_usage().retained_readbacks, 0);
    drop(device);
    validation.check().unwrap();
}
