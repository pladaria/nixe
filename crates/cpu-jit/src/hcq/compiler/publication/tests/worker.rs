use super::lifecycle::{payload, promote_at, run};
use super::negative::admit_to;
use super::*;
use crate::lifetime::background::{Outcome, workers::Workers};
use std::sync::mpsc;
use std::time::Duration;

mod races;
mod rejection;
mod sampling;

#[test]
fn real_worker_replaces_two_families_then_persists_no_op_without_recompilation() {
    let (process, memory, mut reader) = setup();
    let old = [
        promote_at(&process, &memory, &mut reader, 0x2000),
        promote_at(&process, &memory, &mut reader, 0x1000),
    ];
    let memory = Arc::new(memory);
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
    let (done, received) = mpsc::channel();
    let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
        let result = consumer(resources, work);
        done.send(result.is_ok()).unwrap();
        result
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x1000,
            0x1004,
            0x2000
        ),
        Outcome::Queued
    );
    assert!(received.recv_timeout(Duration::from_secs(10)).unwrap());
    process.try_service_links().unwrap();
    let entry = payload(&mut reader, 0x1000).unwrap();
    assert_eq!(
        entry.hcq().unwrap().family,
        payload(&mut reader, 0x2000).unwrap().hcq().unwrap().family
    );
    // Production maintenance, not a test-only collector call, releases them.
    assert_eq!(process.reclaim_units().unwrap(), 0);
    for unit in old {
        assert!(matches!(
            process.snapshot(unit),
            Err(lifetime::Error::StaleUnit)
        ));
    }
    for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
        run(&process, &memory, &mut reader, pc, expected);
    }
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x1000,
            0x1004,
            0x2000
        ),
        Outcome::Queued
    );
    assert!(received.recv_timeout(Duration::from_secs(10)).unwrap());
    assert_eq!(payload(&mut reader, 0x1000).unwrap(), entry);
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x1000,
            0x1004,
            0x2000
        ),
        Outcome::Suppressed
    );
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn real_worker_records_disconnected_boundary_without_poisoning_initial_promotion() {
    let (process, memory, mut reader) = setup();
    let memory = Arc::new(memory);
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
    let (done, received) = mpsc::channel();
    let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
        let result = consumer(resources, work);
        done.send(result.is_ok()).unwrap();
        result
    })
    .unwrap()
    .unwrap();
    // RET is deliberately not traversed by region discovery.
    let before = payload(&mut reader, 0x7000).unwrap();
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x7000,
            0x7000,
            0x2000
        ),
        Outcome::Queued
    );
    assert!(received.recv_timeout(Duration::from_secs(10)).unwrap());
    assert_eq!(payload(&mut reader, 0x7000).unwrap(), before);
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x7000,
            0x7000,
            0x2000
        ),
        Outcome::Suppressed
    );
    promote_at(&process, &memory, &mut reader, 0x7000);
    run(&process, &memory, &mut reader, 0x7000, 2);
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn real_worker_keeps_its_running_reshape_reservation_across_unrelated_stop() {
    let (process, memory, mut reader) = setup();
    promote_at(&process, &memory, &mut reader, 0x2000);
    let memory = Arc::new(memory);
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
    let (started, ready) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let (done, received) = mpsc::channel();
    let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
        started.send(()).unwrap();
        wait.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        let result = consumer(resources, work);
        done.send(result.is_ok()).unwrap();
        result
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x1000,
            0x1004,
            0x2000
        ),
        Outcome::Queued
    );
    ready.recv_timeout(Duration::from_secs(10)).unwrap();
    process.request(lifetime::Reason::LinkPatch).unwrap();
    assert!(process.try_service_links().unwrap());
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x1000,
            0x1004,
            0x2000
        ),
        Outcome::Deferred
    );
    release.send(()).unwrap();
    assert!(received.recv_timeout(Duration::from_secs(10)).unwrap());
    process.try_service_links().unwrap();
    assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_some());
    run(&process, &memory, &mut reader, 0x3000, 3);
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
    assert!(process.try_shutdown().unwrap());
}
