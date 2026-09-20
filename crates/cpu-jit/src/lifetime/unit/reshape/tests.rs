use super::*;
use crate::lifetime::background::Job;
use crate::lifetime::unit::tests::{input, key, process, publish};
use crate::sampling::BoundaryKey;

fn instruction(pc: u64) -> InstructionKey {
    InstructionKey::new(key(pc)).unwrap()
}

fn optimize(process: &Lifetime, pcs: &[u64], entries: usize) -> UnitHandle {
    let mut candidate = input(process, pcs, Tier::Hcq);
    candidate.entries = candidate
        .entries
        .into_vec()
        .into_iter()
        .take(entries)
        .collect();
    let publications: Vec<_> = pcs[..entries]
        .iter()
        .map(|pc| process.reserve(key(*pc)).unwrap())
        .collect();
    let handle = process
        .prepare_unit(&publications, candidate, &AtomicU64::new(0))
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    handle
}

fn boundary(process: &Lifetime, block: u64, source: u64, target: u64) -> BoundaryKey {
    let state = process.lock();
    BoundaryKey {
        source: instruction(source),
        target: instruction(target),
        source_version: sampling::endpoint(&state, key(block))
            .unwrap()
            .payload
            .reachability(),
        target_version: sampling::endpoint(&state, key(target))
            .unwrap()
            .payload
            .reachability(),
        source_family: sampling::family(&state, instruction(source)).unwrap(),
        target_family: sampling::family(&state, instruction(target)).unwrap(),
    }
}

fn heat(samples: &mut Samples, key: BoundaryKey) -> ReshapeSnapshot {
    let mut snapshot = None;
    for _ in 0..4 {
        snapshot = samples.boundary(key, true);
    }
    snapshot.unwrap()
}

fn pop(queue: &Queue) -> ReshapeJob {
    let Some(Job::Reshape(job)) = queue.pop().unwrap() else {
        panic!("expected reshape job")
    };
    job
}

fn retire(process: &Lifetime, unit: UnitHandle) {
    process.retire_unit(unit).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn fourth_boundary_sample_admits_zero_one_or_two_versioned_families() {
    for count in 0..=2 {
        let process = process();
        for pc in [0, 16] {
            publish(&process, &AtomicU64::new(0), &[pc, pc + 4], Tier::Lcq);
        }
        if count >= 1 {
            optimize(&process, &[16, 20], 1);
        }
        if count == 2 {
            optimize(&process, &[0, 4], 1);
        }
        let queue = Queue::new(1, &process).unwrap().unwrap();
        let key = boundary(&process, 0, 4, 16);
        let mut samples = Samples::new();
        for _ in 0..3 {
            assert!(samples.boundary(key, true).is_none());
        }
        assert!(queue.pop().unwrap().is_none());
        let snapshot = samples.boundary(key, true).unwrap();
        assert_eq!(
            process
                .admit_reshape(
                    &queue,
                    &mut samples,
                    crate::lifetime::unit::tests::key(0),
                    snapshot
                )
                .unwrap(),
            Outcome::Queued
        );
        // A busy owner prevents a second boundary request without cooling an
        // unrelated vCPU record or replacing the first request's snapshot.
        assert_eq!(
            process
                .admit_reshape(
                    &queue,
                    &mut samples,
                    crate::lifetime::unit::tests::key(0),
                    snapshot
                )
                .unwrap(),
            Outcome::Deferred
        );
        assert_eq!(
            samples.boundary_snapshot(key.source, key.target).unwrap().1,
            3
        );
        let job = pop(&queue);
        assert_eq!(job.snapshot, snapshot);
        assert_eq!(job.source_block, crate::lifetime::unit::tests::key(0));
        assert_eq!(job.process, process.identity);
        assert_eq!(job.admission, process.lock().admission);
        assert_eq!(job.participants.iter().flatten().count(), count);
        assert_eq!(job.reservations.iter().flatten().count(), count.max(1));
        if count == 2 {
            assert!(
                job.participants[0].unwrap().identity.id < job.participants[1].unwrap().identity.id
            );
        }
        let state = process.lock();
        for endpoint in &job.endpoints {
            assert!(
                state
                    .dispatch
                    .get(endpoint.slot)
                    .unwrap()
                    .optimization
                    .pinned()
            );
            assert!(state.units.records.get(endpoint.unit.0).is_some());
        }
    }
}

#[test]
fn retained_interior_entry_reserves_the_same_family_only_once() {
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Lcq);
    optimize(&process, &[0, 4], 1);
    let mut samples = Samples::new();
    let snapshot = heat(&mut samples, boundary(&process, 0, 0, 4));
    assert_eq!(snapshot.key.source_family, snapshot.key.target_family);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Queued
    );
    let job = pop(&queue);
    assert_eq!(job.participants.iter().flatten().count(), 1);
    assert_eq!(job.reservations.iter().flatten().count(), 1);
}

#[test]
fn a_busy_second_family_rolls_back_only_the_first_claim() {
    let process = process();
    for pc in [0, 4, 8] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
    }
    let first = optimize(&process, &[0], 1);
    optimize(&process, &[4], 1);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let b_to_c = heat(&mut samples, boundary(&process, 4, 4, 8));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(4), b_to_c)
            .unwrap(),
        Outcome::Queued
    );
    let a_to_b = heat(&mut samples, boundary(&process, 0, 0, 4));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), a_to_b)
            .unwrap(),
        Outcome::Deferred
    );
    assert!(
        !process
            .lock()
            .units
            .records
            .get(first.0)
            .unwrap()
            .reshape
            .as_ref()
            .unwrap()
            .pinned()
    );
    let a_to_c = heat(&mut samples, boundary(&process, 0, 0, 8));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), a_to_c)
            .unwrap(),
        Outcome::Queued
    );
    assert_eq!(pop(&queue).snapshot, a_to_c);
    assert_eq!(pop(&queue).snapshot, b_to_c);
}

#[test]
fn queue_fullness_releases_both_family_claims_and_keeps_score_three() {
    let process = process();
    for pc in (0..40).step_by(4) {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
    }
    optimize(&process, &[32], 1);
    optimize(&process, &[36], 1);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    for pc in (0..32).step_by(4) {
        let version = process.reserve(key(pc)).unwrap().reachability;
        let mut snapshot = None;
        for _ in 0..8 {
            snapshot = samples.seed(key(pc), version, None, true);
        }
        assert_eq!(
            process
                .admit_seed(&queue, &mut samples, snapshot.unwrap())
                .unwrap(),
            Outcome::Queued
        );
    }
    let snapshot = heat(&mut samples, boundary(&process, 32, 32, 36));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(32), snapshot)
            .unwrap(),
        Outcome::Deferred
    );
    assert_eq!(
        samples
            .boundary_snapshot(instruction(32), instruction(36))
            .unwrap()
            .1,
        3
    );
    drop(queue.pop().unwrap());
    let snapshot = samples.boundary(snapshot.key, true).unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(32), snapshot)
            .unwrap(),
        Outcome::Queued
    );
    assert_eq!(pop(&queue).snapshot, snapshot);
    assert_eq!(queue.close().unwrap().len(), 7);
}

#[test]
fn stale_first_or_second_family_cannot_enqueue_or_release_a_new_epoch_claim() {
    for retired_index in [0, 1] {
        let process = process();
        for pc in [0, 4] {
            publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
        }
        let families = [optimize(&process, &[0], 1), optimize(&process, &[4], 1)];
        let snapshot = heat(&mut Samples::new(), boundary(&process, 0, 0, 4));
        let old = process
            .reserve_reshape(key(0), snapshot)
            .unwrap()
            .ok()
            .unwrap();
        let queue = Queue::new(1, &process).unwrap().unwrap();
        retire(&process, families[retired_index]);
        process.reclaim_units().unwrap();
        // Pin holds the actual family/unit registry generation after unlink,
        // without holding the removed Family object or baseline code promises.
        assert!(
            process
                .lock()
                .units
                .records
                .get(families[retired_index].0)
                .is_some()
        );
        optimize(&process, &[retired_index as u64 * 4], 1);
        let mut samples = Samples::new();
        let current = heat(&mut samples, boundary(&process, 0, 0, 4));
        assert_eq!(
            process
                .admit_reshape(&queue, &mut samples, key(0), current)
                .unwrap(),
            Outcome::Queued
        );
        assert_eq!(queue.enqueue(old).unwrap(), Outcome::Stale);
        assert_eq!(
            process
                .admit_reshape(&queue, &mut samples, key(0), current)
                .unwrap(),
            Outcome::Deferred
        );
        drop(pop(&queue));
        process.reclaim_units().unwrap();
        assert!(
            process
                .lock()
                .units
                .records
                .get(families[retired_index].0)
                .is_none()
        );
    }
}

#[test]
fn invalid_logical_source_versions_ownership_and_context_do_not_create_work() {
    let process = process();
    for pc in [0, 16] {
        publish(&process, &AtomicU64::new(0), &[pc, pc + 4], Tier::Lcq);
    }
    optimize(&process, &[16, 20], 1);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let snapshot = heat(&mut samples, boundary(&process, 0, 4, 16));
    let mut wrong = snapshot;
    wrong.key.target_family = None;
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), wrong)
            .unwrap(),
        Outcome::Stale
    );
    wrong = snapshot;
    wrong.key.source_version = crate::abi::ReachabilityVersion::new(u64::MAX).unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), wrong)
            .unwrap(),
        Outcome::Stale
    );
    wrong = snapshot;
    wrong.key.source = instruction(8); // Not an instruction of source block 0.
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), wrong)
            .unwrap(),
        Outcome::Stale
    );
    wrong = snapshot;
    wrong.key.target = InstructionKey::new(BlockKey {
        fp: crate::abi::FpSpecialization::Exact(0),
        ..key(16)
    })
    .unwrap();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), wrong)
            .unwrap(),
        Outcome::Stale
    );
    let count = process.lock().keys.len();
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(128), snapshot)
            .unwrap(),
        Outcome::Stale
    );
    assert_eq!(process.lock().keys.len(), count);
    assert!(queue.pop().unwrap().is_none());
}

#[test]
fn closing_and_shutdown_preserve_pins_until_queued_or_unqueued_cleanup() {
    for enqueue in [false, true] {
        let process = process();
        for pc in [0, 4] {
            publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
        }
        optimize(&process, &[0], 1);
        optimize(&process, &[4], 1);
        let mut samples = Samples::new();
        let snapshot = heat(&mut samples, boundary(&process, 0, 0, 4));
        let mut reserved = Some(
            process
                .reserve_reshape(key(0), snapshot)
                .unwrap()
                .ok()
                .unwrap(),
        );
        let queue = Queue::new(1, &process).unwrap().unwrap();
        if enqueue {
            assert_eq!(
                queue.enqueue(reserved.take().unwrap()).unwrap(),
                Outcome::Queued
            );
        }
        process.request_shutdown().unwrap();
        assert_eq!(
            process
                .admit_reshape(&queue, &mut samples, key(0), snapshot)
                .unwrap(),
            Outcome::Stale
        );
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        assert!(!transition.try_finish_shutdown().unwrap());
        let drained = queue.close().unwrap();
        if let Some(reserved) = reserved {
            assert_eq!(queue.enqueue(reserved).unwrap(), Outcome::Stale);
        }
        drop(drained);
        assert!(transition.try_finish_shutdown().unwrap());
    }
}

#[test]
fn a_lost_second_token_rolls_back_the_first_queued_transition() {
    let process = process();
    for pc in [0, 4] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
    }
    let first = optimize(&process, &[0], 1);
    let second = optimize(&process, &[4], 1);
    let snapshot = heat(&mut Samples::new(), boundary(&process, 0, 0, 4));
    let old = process
        .reserve_reshape(key(0), snapshot)
        .unwrap()
        .ok()
        .unwrap();
    retire(&process, second);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    // No new admission reset the surviving family's first token. Its CAS to
    // Queued succeeds, then the second fails and exact rollback must clear it.
    assert_eq!(queue.enqueue(old).unwrap(), Outcome::Stale);
    assert!(
        !process
            .lock()
            .units
            .records
            .get(first.0)
            .unwrap()
            .reshape
            .as_ref()
            .unwrap()
            .pinned()
    );
    assert!(queue.pop().unwrap().is_none());
    let mut samples = Samples::new();
    let current = heat(&mut samples, boundary(&process, 0, 0, 4));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), current)
            .unwrap(),
        Outcome::Queued
    );
}

#[test]
fn concurrent_vcpus_cannot_enqueue_two_jobs_for_a_shared_family() {
    let process = process();
    for pc in [0, 4] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
    }
    optimize(&process, &[0], 1);
    optimize(&process, &[4], 1);
    let snapshot = heat(&mut Samples::new(), boundary(&process, 0, 0, 4));
    let queue = Arc::new(Queue::new(1, &process).unwrap().unwrap());
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let process = Arc::clone(&process);
            let queue = Arc::clone(&queue);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let mut samples = Samples::new();
                barrier.wait();
                process
                    .admit_reshape(&queue, &mut samples, key(0), snapshot)
                    .unwrap()
            })
        })
        .collect();
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(
        results
            .iter()
            .filter(|&&result| result == Outcome::Queued)
            .count(),
        1
    );
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Outcome::Queued | Outcome::Deferred))
    );
    assert_eq!(pop(&queue).snapshot, snapshot);
    assert!(queue.pop().unwrap().is_none());
}
