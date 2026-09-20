use super::*;
use crate::lifetime::background::tests::{observed_boundary, setup, snapshot};
use crate::lifetime::unit::tests::{key, publish};

mod discovery;

fn enqueue(process: &Lifetime, queue: &Queue, samples: &mut Samples, pc: u64) -> AdmissionSnapshot {
    let snapshot = snapshot(process, pc);
    assert_eq!(
        process.admit_seed(queue, samples, snapshot).unwrap(),
        Outcome::Queued
    );
    snapshot
}

fn retire(process: &Lifetime, handle: unit::UnitHandle) {
    process.retire_unit(handle).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn running_job_acquires_only_named_current_lcq_inputs_in_its_execution_context() {
    let (process, queue, mut samples) = setup(3);
    publish(&process, &AtomicU64::new(0), &[8], Tier::Hcq);
    process.try_service_links().unwrap();
    let observed = enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(work.observation(), Observation::Seed(observed));
    assert_eq!(process.lock().compilers, 1);
    let source = work.lcq(key(0)).unwrap().unwrap();
    assert_eq!(source.key, key(0));
    assert_eq!(source.version, observed.version);
    assert_eq!(source.unit.tier, Tier::Lcq);
    assert_eq!(source.unit.instructions[0].key.block_key(), key(0));
    assert!(work.lcq(key(4)).unwrap().is_some());
    assert!(work.lcq(key(8)).unwrap().is_none());
    let count = process.lock().keys.len();
    assert!(work.lcq(key(128)).unwrap().is_none());
    assert!(
        work.lcq(BlockKey {
            fp: crate::abi::FpSpecialization::Exact(0),
            ..key(0)
        })
        .unwrap()
        .is_none()
    );
    assert_eq!(process.lock().keys.len(), count);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
    drop(work);
    assert_eq!(process.lock().compilers, 0);
}

#[test]
fn hcq_graph_retains_distinct_demanded_units_and_deterministic_input_identities() {
    let (process, queue, mut samples) = setup(3);
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let first = crate::hcq::Graph::discover(&work).unwrap();
    let second = crate::hcq::Graph::discover(&work).unwrap();
    assert_eq!(first.units.len(), 3);
    assert_eq!(first.inputs.len(), 3);
    assert_eq!(first.instructions.len(), 3);
    assert_eq!(first.blocks.len(), 3);
    for (a, b) in first.inputs.iter().zip(&second.inputs) {
        assert_eq!((a.key, a.version, a.unit), (b.key, b.version, b.unit));
        assert_eq!(first.units[a.unit].id, second.units[b.unit].id);
    }
    let handle = first.units[0].registered_handle().unwrap();
    retire(&process, handle);
    assert_eq!(work.check(), Err(Error::StalePublication));
    drop(work);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(first.instructions[0].instruction.bits, 0xd503201f);
    drop(first);
    drop(second);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn compiler_references_survive_unlink_without_holding_an_execution_epoch() {
    let (process, queue, mut samples) = setup(1);
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let source = work.lcq(key(0)).unwrap().unwrap();
    let handle = source.unit.registered_handle().unwrap();
    // Closed must complete without waiting for the compiler, while its strong
    // snapshot remains readable even after the original unit is unlinked.
    retire(&process, handle);
    assert_eq!(work.check(), Err(Error::StalePublication));
    assert!(matches!(work.lcq(key(0)), Err(Error::StalePublication)));
    process.reclaim_units().unwrap();
    assert_eq!(source.unit.instructions[0].bits, 0xd503201f);
    drop(work);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    drop(source);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
}

#[test]
fn old_running_cleanup_cannot_cancel_a_new_epoch_job() {
    let (process, queue, mut samples) = setup(1);
    let observed = enqueue(&process, &queue, &mut samples, 0);
    let old = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    process.request(Reason::LinkPatch).unwrap();
    process.try_service_links().unwrap();
    assert_eq!(old.check(), Err(Error::StalePublication));
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
    let current = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(process.lock().compilers, 2);
    drop(old);
    assert_eq!(process.lock().compilers, 1);
    current.check().unwrap();
    assert!(current.lcq(key(0)).unwrap().is_some());
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
}

#[test]
fn dequeue_rejects_replaced_epoch_version_and_foreign_process_identities() {
    for replace in [false, true] {
        let (process, queue, mut samples) = setup(1);
        enqueue(&process, &queue, &mut samples, 0);
        let job = queue.wait().unwrap().unwrap();
        if replace {
            publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
        } else {
            process.request(Reason::LinkPatch).unwrap();
        }
        process.try_service_links().unwrap();
        assert!(process.accept_background(job).unwrap().is_none());
        assert_eq!(process.lock().compilers, 0);
        let current = enqueue(&process, &queue, &mut samples, 0);
        let foreign = crate::lifetime::unit::tests::process();
        assert!(matches!(
            foreign.accept_background(queue.wait().unwrap().unwrap()),
            Err(Error::InvalidUnit(
                "background job belongs to another process"
            ))
        ));
        assert_eq!(foreign.lock().compilers, 0);
        assert_eq!(
            process.admit_seed(&queue, &mut samples, current).unwrap(),
            Outcome::Queued
        );
    }
}

#[test]
fn reshape_workers_read_participating_baselines_not_optimized_bodies() {
    let (process, queue, mut samples) = setup(3);
    let source = process.reserve(key(0)).unwrap();
    let baseline = process
        .lock()
        .dispatch
        .get(source.slot)
        .unwrap()
        .snapshot()
        .lcq()
        .unwrap()
        .unit;
    for pc in [0, 4, 8] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Hcq);
    }
    process.try_service_links().unwrap();
    let observed = observed_boundary(&process, &mut samples);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), observed)
            .unwrap(),
        Outcome::Queued
    );
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(
        work.observation(),
        Observation::Reshape {
            source_block: key(0),
            snapshot: observed
        }
    );
    assert_eq!(work.lcq(key(0)).unwrap().unwrap().unit.id, baseline);
    assert_eq!(work.lcq(key(4)).unwrap().unwrap().unit.tier, Tier::Lcq);
    assert!(work.lcq(key(8)).unwrap().is_none());
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), observed)
            .unwrap(),
        Outcome::Deferred
    );
    drop(work);
    assert_eq!(process.lock().compilers, 0);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), observed)
            .unwrap(),
        Outcome::Queued
    );
}

#[test]
fn shutdown_waits_for_running_work_and_then_for_retained_input() {
    let (process, queue, mut samples) = setup(1);
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let source = work.lcq(key(0)).unwrap().unwrap();
    process.request_shutdown().unwrap();
    drop(queue.close().unwrap());
    assert!(queue.wait().unwrap().is_none());
    assert_eq!(work.check(), Err(Error::StalePublication));
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.try_finish_shutdown().unwrap());
    drop(work);
    assert!(!transition.try_finish_shutdown().unwrap());
    drop(source);
    assert!(transition.try_finish_shutdown().unwrap());
}

#[test]
fn unwinding_worker_scope_releases_compiler_count_and_exact_reservation() {
    let (process, queue, mut samples) = setup(1);
    let observed = enqueue(&process, &queue, &mut samples, 0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let work = process
            .accept_background(queue.wait().unwrap().unwrap())
            .unwrap()
            .unwrap();
        let _source = work.lcq(key(0)).unwrap().unwrap();
        panic!("worker scope unwind");
    }));
    assert!(result.is_err());
    assert_eq!(process.lock().compilers, 0);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
}

#[test]
fn waiting_workers_claim_independent_jobs_and_close_wakes_the_empty_queue() {
    use std::sync::mpsc;
    use std::time::Duration;
    let (process, queue, mut samples) = setup(2);
    let queue = Arc::new(queue);
    let (events, receiver) = mpsc::channel();
    let mut releases = Vec::new();
    let mut threads = Vec::new();
    for worker in 0..2 {
        let queue = Arc::clone(&queue);
        let process = Arc::clone(&process);
        let events = events.clone();
        let (release, wait_release) = mpsc::channel();
        releases.push(release);
        threads.push(std::thread::spawn(move || {
            events.send((worker, None)).unwrap();
            let work = process
                .accept_background(queue.wait().unwrap().unwrap())
                .unwrap()
                .unwrap();
            let Observation::Seed(snapshot) = work.observation() else {
                panic!("expected seed")
            };
            let input = work.lcq(snapshot.key).unwrap().unwrap();
            assert_eq!(input.unit.instructions[0].key.block_key(), snapshot.key);
            events.send((worker, Some(snapshot.key))).unwrap();
            wait_release.recv().unwrap();
            drop(input);
            drop(work);
            events.send((worker, None)).unwrap();
            assert!(queue.wait().unwrap().is_none());
        }));
    }
    for _ in 0..2 {
        assert!(
            receiver
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .1
                .is_none()
        );
    }
    enqueue(&process, &queue, &mut samples, 0);
    let first = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    enqueue(&process, &queue, &mut samples, 4);
    let second = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_ne!(first.0, second.0);
    assert_eq!(first.1, Some(key(0)));
    assert_eq!(second.1, Some(key(4)));
    assert_eq!(process.lock().compilers, 2);
    for release in releases {
        release.send(()).unwrap();
    }
    // Both consumers return to an empty queue. Closure must either wake a
    // sleeping consumer or be observed before it starts waiting; neither order
    // may lose the notification.
    for _ in 0..2 {
        assert!(
            receiver
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .1
                .is_none()
        );
    }
    drop(queue.close().unwrap());
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(process.lock().compilers, 0);
}
