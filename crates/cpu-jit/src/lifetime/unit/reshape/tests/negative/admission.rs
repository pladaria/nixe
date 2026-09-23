use super::*;

fn install(process: &Lifetime) {
    let work = reshape(process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(frozen.prepare_unchanged().unwrap().install().unwrap());
}

#[test]
fn shared_negative_suppresses_independent_vcpu_samples_without_jobs_or_cache_growth() {
    let (process, _) = setup();
    install(&process);
    let boundary = result_key(&process).boundary;
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let before = process.cache.usage().unwrap();
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..4 {
            let process = &process;
            let queue = &queue;
            threads.push(scope.spawn(move || {
                let mut samples = Samples::new();
                let mut suppressed = 0;
                for _ in 0..128 {
                    if let Some(snapshot) = samples.boundary(boundary, true) {
                        match process
                            .admit_reshape(queue, &mut samples, key(0), snapshot)
                            .unwrap()
                        {
                            Outcome::Suppressed => suppressed += 1,
                            Outcome::Deferred => {} // Existing nonblocking mutex contention.
                            outcome => panic!("unexpected admission: {outcome:?}"),
                        }
                    }
                }
                suppressed
            }));
        }
        assert!(
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .sum::<usize>()
                > 0
        );
    });
    assert!(queue.pop().unwrap().is_none());
    assert_eq!(process.lock().compilers, 0);
    assert_eq!(process.cache.usage().unwrap().metadata, before.metadata);
    assert_eq!(process.cache.usage().unwrap().committed, before.committed);
}

#[test]
fn sample_eviction_and_new_sequence_do_not_erase_negative_but_root_change_allows_retry() {
    let (process, _) = setup();
    install(&process);
    let boundary = result_key(&process).boundary;
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let original = heat(&mut samples, boundary);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), original)
            .unwrap(),
        Outcome::Suppressed
    );
    // Find two actual colliders in this table's randomized placement; equal
    // heat and newer age evict the original through the ordinary victim rule.
    let colliders: Vec<_> = (1..)
        .map(|index| BoundaryKey {
            target: instruction(16 + index * 4),
            ..boundary
        })
        .filter(|candidate| samples.same_boundary_set(boundary, *candidate))
        .take(2)
        .collect();
    for collider in colliders {
        heat(&mut samples, collider);
    }
    assert!(
        samples
            .boundary_snapshot(boundary.source, boundary.target)
            .is_none()
    );
    let current = heat(&mut samples, boundary);
    assert!(current.sequence > original.sequence);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), current)
            .unwrap(),
        Outcome::Suppressed
    );
    assert_eq!(
        samples
            .boundary_snapshot(boundary.source, boundary.target)
            .unwrap()
            .1,
        4
    );
    // This introduces an entry at PC 20 without changing the named endpoints.
    crate::lifetime::unit::links::tests::source(&process, &AtomicU64::new(0), 128, 20);
    assert_eq!(result_key(&process).boundary, boundary);
    let retry = samples.boundary(boundary, true).unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), retry)
            .unwrap(),
        Outcome::Queued
    );
    drop(queue.pop().unwrap().unwrap());
}

#[test]
fn suppression_precedes_busy_family_and_full_queue_without_hiding_another_boundary() {
    let (process, _) = setup();
    install(&process);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let other = heat(&mut samples, boundary(&process, 16, 16, 20));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(16), other)
            .unwrap(),
        Outcome::Queued
    );
    let snapshot = heat(&mut samples, result_key(&process).boundary);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Suppressed
    );
    let job = pop(&queue);
    assert_eq!(job.snapshot, other);
    assert!(queue.pop().unwrap().is_none());
}

#[test]
fn contention_and_pressure_defer_without_destroying_the_negative() {
    let (process, _) = setup();
    install(&process);
    let result = result_key(&process);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let snapshot = heat(&mut samples, result.boundary);
    let state = process.lock();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Deferred
    );
    assert_eq!(
        samples
            .boundary_snapshot(result.boundary.source, result.boundary.target)
            .unwrap()
            .1,
        3
    );
    drop(state);
    let bytes = crate::executable::SOFT_BYTES - process.cache.usage().unwrap().total();
    let pressure = process.cache.charge_metadata(bytes, Tier::Lcq).unwrap();
    let snapshot = samples.boundary(result.boundary, true).unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Deferred
    );
    drop(pressure);
    let snapshot = samples.boundary(result.boundary, true).unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Suppressed
    );
    assert!(process.lock().units.negatives.get(result).is_some());
    assert!(queue.pop().unwrap().is_none());
}

#[test]
fn old_endpoint_generation_is_stale_and_new_generation_is_not_suppressed() {
    let (process, _) = setup();
    install(&process);
    let old = result_key(&process).boundary;
    publish_words(&process, 16, &[0xd503201f, 0x17fffffb]);
    assert!(process.try_service_links().unwrap());
    let new = result_key(&process).boundary;
    assert_ne!(old.target_version, new.target_version);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let snapshot = heat(&mut samples, old);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Stale
    );
    let snapshot = heat(&mut samples, new);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Queued
    );
    drop(queue.pop().unwrap().unwrap());
}
