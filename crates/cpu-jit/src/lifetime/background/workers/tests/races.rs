use super::*;
use crate::executable::{SEGMENT_BYTES, SOFT_BYTES, Tier};
use crate::lifetime::background::tests::observed_boundary;
use crate::lifetime::unit::tests::publish;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[test]
fn pressure_defers_seed_and_reshape_and_discards_already_queued_work() {
    let (process, queue, mut samples) = setup(2);
    let seed = snapshot(&process, 0);
    for _ in 0..8 {
        samples.seed(seed.key, seed.version, None, true);
    }
    let seed = samples.seed_snapshot(seed.key).unwrap().0;
    let boundary = observed_boundary(&process, &mut samples);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, seed).unwrap(),
        Outcome::Queued
    );
    let pending = queue.pop().unwrap().unwrap();
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    assert_eq!(
        process.admit_seed(&queue, &mut samples, seed).unwrap(),
        Outcome::Deferred
    );
    assert_eq!(samples.seed_snapshot(seed.key).unwrap().1, 7);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), boundary)
            .unwrap(),
        Outcome::Deferred
    );
    assert_eq!(
        samples
            .boundary_snapshot(boundary.key.source, boundary.key.target)
            .unwrap()
            .1,
        3
    );
    assert!(process.accept_background(pending).unwrap().is_none());
    assert_eq!(process.lock().compilers, 0);
    assert!(queue.pop().unwrap().is_none());
    drop(charge);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, seed).unwrap(),
        Outcome::Queued
    );
    let work = process
        .accept_background(queue.pop().unwrap().unwrap())
        .unwrap()
        .unwrap();
    assert!(work.lcq(key(0)).unwrap().is_some());
    work.check().unwrap();
}

#[test]
fn pressure_cancels_running_work_without_rejection_and_resets_unfinished_frontend() {
    let (process, _, mut samples) = setup(1);
    let (started, ready) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let (finished, done) = mpsc::channel();
    let calls = AtomicUsize::new(0);
    let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
        let call = calls.fetch_add(1, Ordering::Relaxed);
        let source = work.lcq(key(0))?.unwrap();
        // Deliberately abandon a non-finalized builder on the first attempt.
        {
            let mut builder =
                FunctionBuilder::new(&mut resources.context.func, &mut resources.frontend);
            let block = builder.create_block();
            builder.switch_to_block(block);
            builder.ins().iconst(types::I64, 42);
        }
        if call == 0 {
            started.send(()).unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        let result = work.check();
        assert_eq!(source.unit.instructions.get(0).unwrap().bits, 0xd503201f);
        drop(source);
        drop(work);
        finished.send((call, result)).unwrap();
        result?;
        // Test-only abandonment also exercises resetting a second builder.
        Err(CompileError::Cancelled)
    })
    .unwrap()
    .unwrap();
    enqueue(&process, &workers, &mut samples, 0);
    ready.recv_timeout(Duration::from_secs(10)).unwrap();
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    release.send(()).unwrap();
    assert!(matches!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        (0, Err(lifetime::Error::Capacity(_)))
    ));
    drop(charge);
    enqueue(&process, &workers, &mut samples, 0);
    assert_eq!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        (1, Ok(()))
    );
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
}

#[test]
fn replaced_running_job_cannot_clear_new_running_reservation() {
    let (process, _, mut samples) = setup(1);
    let (events, receiver) = mpsc::channel();
    let (release_old, old_wait) = mpsc::channel();
    let (release_new, new_wait) = mpsc::channel();
    let waits = [Mutex::new(old_wait), Mutex::new(new_wait)];
    let calls = AtomicUsize::new(0);
    let mut workers = Workers::start(2, Arc::clone(&process), move |_, work| {
        let call = calls.fetch_add(1, Ordering::Relaxed);
        let source = work.lcq(key(0))?.unwrap();
        events.send((call, false)).unwrap();
        waits[call]
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        let result = work.check();
        assert_eq!(source.unit.instructions.get(0).unwrap().bits, 0xd503201f);
        drop(source);
        drop(work);
        events.send((call, true)).unwrap();
        if call == 0 {
            assert_eq!(result, Err(lifetime::Error::StalePublication));
        } else {
            assert_eq!(result, Ok(()));
        }
        result?;
        Ok(())
    })
    .unwrap()
    .unwrap();
    enqueue(&process, &workers, &mut samples, 0);
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(10)).unwrap(),
        (0, false)
    );
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    process.try_service_links().unwrap();
    enqueue(&process, &workers, &mut samples, 0);
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(10)).unwrap(),
        (1, false)
    );
    release_old.send(()).unwrap();
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(10)).unwrap(),
        (0, true)
    );
    assert_eq!(
        process
            .admit_seed(workers.queue(), &mut samples, snapshot(&process, 0))
            .unwrap(),
        Outcome::Duplicate
    );
    release_new.send(()).unwrap();
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(10)).unwrap(),
        (1, true)
    );
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
}

#[test]
fn hcq_pressure_retires_reserved_families_without_waiting_for_reshape_compilation() {
    let (process, _, mut samples) = setup(2);
    for pc in [0, 4] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Hcq);
    }
    process.try_service_links().unwrap();
    let boundary = observed_boundary(&process, &mut samples);
    let (started, ready) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let mut workers = Workers::start(1, Arc::clone(&process), move |_, work| {
        let source = work.lcq(key(0))?.unwrap();
        started.send(()).unwrap();
        wait.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        assert_eq!(source.unit.tier, Tier::Lcq);
        work.check()?;
        panic!("retired reshape must be stale");
    })
    .unwrap()
    .unwrap();
    let start = Instant::now();
    loop {
        match process
            .admit_reshape(workers.queue(), &mut samples, key(0), boundary)
            .unwrap()
        {
            Outcome::Queued => break,
            Outcome::Deferred if start.elapsed() < Duration::from_secs(10) => {
                std::thread::yield_now();
            }
            outcome => panic!("unexpected admission: {outcome:?}"),
        }
    }
    ready.recv_timeout(Duration::from_secs(10)).unwrap();
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES + SEGMENT_BYTES / 2 - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.request(lifetime::Reason::Eviction).unwrap();
    let mut stop = process.try_transition().unwrap().unwrap();
    stop.wait_closed().unwrap();
    let result = stop.relieve_pressure(0, Tier::Hcq);
    assert!(matches!(result, Ok(()) | Err(lifetime::Error::Capacity(_))));
    for pc in [0, 4] {
        let state = process.lock();
        let payload = state
            .dispatch
            .get(*state.keys.get(&key(pc)).unwrap())
            .unwrap()
            .snapshot();
        assert!(payload.lcq().is_some());
        assert!(payload.hcq().is_none());
    }
    stop.batch().unwrap().complete().unwrap();
    assert!(stop.try_reopen().unwrap());
    drop(charge);
    release.send(()).unwrap();
    workers.shutdown().unwrap();
    process.reclaim_units().unwrap();
    assert!(process.background_failure().is_none());
}
