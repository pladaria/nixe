use super::*;
use crate::lifetime::unit::tests::{key, process, publish};

pub(super) fn snapshot(process: &Lifetime, pc: u64) -> AdmissionSnapshot {
    AdmissionSnapshot {
        key: key(pc),
        version: process.reserve(key(pc)).unwrap().reachability,
        sequence: 8,
        last_edge: None,
        successors: [None; 4],
    }
}

pub(super) fn setup(count: usize) -> (Arc<Lifetime>, Queue, Samples) {
    let process = process();
    for index in 0..count {
        publish(&process, &AtomicU64::new(0), &[index as u64 * 4], Tier::Lcq);
    }
    let queue = Queue::new(1, &process).unwrap().unwrap();
    (process, queue, Samples::new())
}

fn retire(process: &Lifetime, unit: unit::UnitHandle) {
    process.retire_unit(unit).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn eighth_seed_sample_enqueues_one_immutable_exact_version() {
    let (process, queue, mut samples) = setup(1);
    let initial = snapshot(&process, 0);
    for _ in 0..7 {
        assert!(
            samples
                .seed(initial.key, initial.version, None, true)
                .is_none()
        );
    }
    assert!(queue.pop().unwrap().is_none());
    let observed = samples
        .seed(initial.key, initial.version, None, true)
        .unwrap();
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
    samples.seed(initial.key, initial.version, None, true);
    let mut job = queue.pop().unwrap().unwrap().seed();
    assert_eq!(job.snapshot, observed);
    assert_eq!(job.process, process.identity);
    assert_eq!(
        process.snapshot(job.unit).unwrap().entries[0].key,
        observed.key
    );
    assert_eq!(job.reservation.word & PHASE_MASK, QUEUED);
    assert!(job.reservation.transition(RUNNING));
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
    drop(job);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
}

#[test]
fn registry_and_queue_contention_defer_without_waiting_and_retry_only_on_new_sample() {
    let (process, queue, mut samples) = setup(1);
    let initial = snapshot(&process, 0);
    for _ in 0..7 {
        samples.seed(initial.key, initial.version, None, true);
    }
    for registry in [true, false] {
        let observed = samples
            .seed(initial.key, initial.version, None, true)
            .unwrap();
        let registry_guard = registry.then(|| process.lock());
        let queue_guard = (!registry).then(|| queue.pending.lock().unwrap());
        assert_eq!(
            process.admit_seed(&queue, &mut samples, observed).unwrap(),
            Outcome::Deferred
        );
        assert_eq!(samples.seed_snapshot(initial.key).unwrap().1, 7);
        drop(queue_guard);
        drop(registry_guard);
        assert!(queue.pop().unwrap().is_none());
    }
    let observed = samples
        .seed(initial.key, initial.version, None, true)
        .unwrap();
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
}

#[test]
fn full_queue_does_not_grow_and_releases_only_its_failed_reservation() {
    let (process, queue, mut samples) = setup(9);
    let capacity = queue.pending.lock().unwrap().jobs.capacity();
    for pc in (0..32).step_by(4) {
        assert_eq!(
            process
                .admit_seed(&queue, &mut samples, snapshot(&process, pc))
                .unwrap(),
            Outcome::Queued
        );
    }
    let next = snapshot(&process, 32);
    for _ in 0..8 {
        samples.seed(next.key, next.version, None, true);
    }
    let next = samples.seed_snapshot(next.key).unwrap().0;
    assert_eq!(
        process.admit_seed(&queue, &mut samples, next).unwrap(),
        Outcome::Deferred
    );
    assert_eq!(samples.seed_snapshot(next.key).unwrap().1, 7);
    let pending = queue.pending.lock().unwrap();
    assert_eq!(pending.jobs.len(), 8);
    assert_eq!(pending.jobs.capacity(), capacity);
    drop(pending);
    drop(queue.pop().unwrap());
    let next = samples.seed(next.key, next.version, None, true).unwrap();
    assert_eq!(
        process.admit_seed(&queue, &mut samples, next).unwrap(),
        Outcome::Queued
    );
}

#[test]
fn process_queue_removes_seven_newest_then_one_oldest_across_empty_periods() {
    let (process, _, mut samples) = setup(16);
    let queue = Queue::new(2, &process).unwrap().unwrap();
    for pc in (0..64).step_by(4) {
        assert_eq!(
            process
                .admit_seed(&queue, &mut samples, snapshot(&process, pc))
                .unwrap(),
            Outcome::Queued
        );
    }
    for expected in [60, 56, 52, 48, 44, 40, 36, 0, 32, 28, 24, 20, 16, 12, 8, 4] {
        let job = queue.pop().unwrap().unwrap().seed();
        assert_eq!(job.snapshot.key.pc.get(), expected);
    }
    assert!(queue.pop().unwrap().is_none());
    assert_eq!(queue.pending.lock().unwrap().removals, 0);
    process
        .admit_seed(&queue, &mut samples, snapshot(&process, 0))
        .unwrap();
    drop(queue.pop().unwrap());
    assert!(queue.pop().unwrap().is_none());
    assert_eq!(queue.pending.lock().unwrap().removals, 1);
}

#[test]
fn maintenance_preserves_reservation_and_stale_rollback_cannot_erase_replacement() {
    let (process, queue, mut samples) = setup(1);
    let observed = snapshot(&process, 0);
    let old = process.reserve_seed(observed).unwrap().ok().unwrap();
    process.request(Reason::LinkPatch).unwrap();
    process.try_service_links().unwrap();
    // Maintenance cannot steal a live token from the same input version.
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    process.try_service_links().unwrap();
    let observed = snapshot(&process, 0);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
    let current = queue.pop().unwrap().unwrap().seed();
    assert_ne!(
        old.reservation.word & !PHASE_MASK,
        current.reservation.word & !PHASE_MASK
    );
    assert_eq!(queue.enqueue(old).unwrap(), Outcome::Stale);
    assert_eq!(
        current.reservation.cell.0.load(Ordering::Acquire),
        current.reservation.word
    );
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
}

#[test]
fn publication_cancels_a_reserved_old_version_before_enqueue() {
    let (process, queue, mut samples) = setup(1);
    let observed = snapshot(&process, 0);
    let old = process.reserve_seed(observed).unwrap().ok().unwrap();
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    process.try_service_links().unwrap();
    assert_eq!(queue.enqueue(old).unwrap(), Outcome::Stale);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Stale
    );
    let current = snapshot(&process, 0);
    assert_eq!(
        process.admit_seed(&queue, &mut samples, current).unwrap(),
        Outcome::Queued
    );
}

#[test]
fn rejection_survives_reopen_but_not_reachability_replacement() {
    let (process, queue, mut samples) = setup(1);
    let observed = snapshot(&process, 0);
    process.admit_seed(&queue, &mut samples, observed).unwrap();
    let mut job = queue.pop().unwrap().unwrap().seed();
    assert!(job.reservation.transition(RUNNING));
    assert!(job.reservation.reject());
    process.request(Reason::LinkPatch).unwrap();
    process.try_service_links().unwrap();
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Duplicate
    );
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    process.try_service_links().unwrap();
    assert_eq!(
        process
            .admit_seed(&queue, &mut samples, snapshot(&process, 0))
            .unwrap(),
        Outcome::Queued
    );
}

#[test]
fn cancelled_unqueued_job_pins_the_actual_slot_until_exact_cleanup() {
    let (process, queue, mut samples) = setup(1);
    let observed = snapshot(&process, 0);
    let job = process.reserve_seed(observed).unwrap().ok().unwrap();
    let slot = job.slot;
    retire(&process, job.unit);
    process.reclaim_units().unwrap();
    assert!(process.lock().dispatch.get(slot).is_some());
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    let replacement = snapshot(&process, 0);
    assert_eq!(
        process
            .admit_seed(&queue, &mut samples, replacement)
            .unwrap(),
        Outcome::Queued
    );
    assert_ne!(
        match queue.pending.lock().unwrap().jobs.front().unwrap() {
            Job::Seed(job) => job.slot,
            _ => panic!("expected seed job"),
        },
        slot
    );
    drop(job);
    assert_eq!(process.collect_dispatch().unwrap(), 1);
    assert!(process.lock().dispatch.get(slot).is_none());
    assert_eq!(
        process
            .admit_seed(&queue, &mut samples, replacement)
            .unwrap(),
        Outcome::Duplicate
    );
}

#[test]
fn close_drains_pins_and_prevents_insertion_from_the_reservation_gap() {
    let (process, queue, mut samples) = setup(2);
    let first = snapshot(&process, 0);
    process.admit_seed(&queue, &mut samples, first).unwrap();
    let second = snapshot(&process, 4);
    let reserved = process.reserve_seed(second).unwrap().ok().unwrap();
    let drained = queue.close().unwrap();
    assert_eq!(drained.len(), 1);
    assert!(queue.pop().unwrap().is_none());
    assert_eq!(queue.enqueue(reserved).unwrap(), Outcome::Stale);
    drop(drained);
    assert!(
        process
            .lock()
            .dispatch
            .values()
            .all(|slot| !slot.optimization.pinned())
    );
    assert_eq!(
        process.admit_seed(&queue, &mut samples, first).unwrap(),
        Outcome::Stale
    );
}

#[test]
fn zero_workers_allocate_no_queue_and_token_exhaustion_fails_explicitly() {
    let (process, queue, mut samples) = setup(1);
    assert!(Queue::new(0, &process).unwrap().is_none());
    let observed = snapshot(&process, 0);
    process.lock().background_tokens.0 = !PHASE_MASK;
    let error = Error::Exhausted(IdentityExhausted("background reservation"));
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed),
        Err(error)
    );
    assert_eq!(process.lock().failure, Some(error));
    assert!(queue.pop().unwrap().is_none());
}

#[test]
fn concurrent_vcpus_share_the_dispatch_reservation_not_their_sample_tables() {
    let (process, queue, _) = setup(1);
    let queue = Arc::new(queue);
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let observed = snapshot(&process, 0);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let process = Arc::clone(&process);
            let queue = Arc::clone(&queue);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let mut samples = Samples::new();
                barrier.wait();
                process.admit_seed(&queue, &mut samples, observed).unwrap()
            })
        })
        .collect();
    let outcomes: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|&&outcome| outcome == Outcome::Queued)
            .count(),
        1
    );
    assert!(outcomes.iter().all(|outcome| matches!(
        outcome,
        Outcome::Queued | Outcome::Deferred | Outcome::Duplicate
    )));
    let job = queue.pop().unwrap().unwrap().seed();
    assert_eq!(job.snapshot, observed);
    assert_eq!(
        job.reservation.cell.0.load(Ordering::Acquire),
        job.reservation.word
    );
    assert!(queue.pop().unwrap().is_none());
}

#[test]
fn reservation_cell_survives_registry_growth_and_rejects_a_foreign_queue() {
    let (process, queue, mut samples) = setup(1);
    let observed = snapshot(&process, 0);
    let job = process.reserve_seed(observed).unwrap().ok().unwrap();
    let cell = Arc::as_ptr(&job.reservation.cell);
    let capacity = process.lock().dispatch.capacity();
    for index in 1..=capacity {
        publish(&process, &AtomicU64::new(0), &[index as u64 * 4], Tier::Lcq);
    }
    assert!(process.lock().dispatch.capacity() > capacity);
    assert_eq!(
        cell,
        Arc::as_ptr(
            &process
                .lock()
                .dispatch
                .get(job.slot)
                .unwrap()
                .optimization
                .cell
        )
    );
    assert_eq!(queue.enqueue(job).unwrap(), Outcome::Queued);
    let other = crate::lifetime::unit::tests::process();
    assert_eq!(
        other.admit_seed(&queue, &mut samples, observed),
        Err(Error::InvalidUnit(
            "background queue belongs to another process"
        ))
    );
}

#[test]
fn shutdown_cannot_release_registry_storage_before_an_unqueued_reservation() {
    let (process, _, _) = setup(1);
    let reserved = process
        .reserve_seed(snapshot(&process, 0))
        .unwrap()
        .ok()
        .unwrap();
    process.request_shutdown().unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.try_finish_shutdown().unwrap());
    assert!(process.lock().dispatch.get(reserved.slot).is_some());
    drop(reserved);
    assert!(transition.try_finish_shutdown().unwrap());
}

pub(super) fn observed_boundary(
    process: &Lifetime,
    samples: &mut Samples,
) -> crate::sampling::ReshapeSnapshot {
    use crate::sampling::{BoundaryKey, FamilyIdentity};
    let payload = |pc| {
        let state = process.lock();
        state
            .dispatch
            .get(*state.keys.get(&key(pc)).unwrap())
            .unwrap()
            .snapshot()
    };
    let source = payload(0);
    let target = payload(4);
    let boundary = BoundaryKey {
        source: crate::abi::InstructionKey::new(key(0)).unwrap(),
        target: crate::abi::InstructionKey::new(key(4)).unwrap(),
        source_version: source.reachability(),
        target_version: target.reachability(),
        source_family: source.hcq().map(|entry| FamilyIdentity {
            id: entry.family,
            version: entry.family_version,
        }),
        target_family: target.hcq().map(|entry| FamilyIdentity {
            id: entry.family,
            version: entry.family_version,
        }),
    };
    for _ in 0..3 {
        samples.boundary(boundary, true);
    }
    samples.boundary(boundary, true).unwrap()
}

#[test]
fn reshape_contention_rolls_back_without_waiting_for_registry_or_queue() {
    let (process, queue, mut samples) = setup(2);
    for pc in [0, 4] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Hcq);
    }
    process.try_service_links().unwrap();
    let mut snapshot = observed_boundary(&process, &mut samples);
    for registry in [true, false] {
        let registry_guard = registry.then(|| process.lock());
        let queue_guard = (!registry).then(|| queue.pending.lock().unwrap());
        assert_eq!(
            process
                .admit_reshape(&queue, &mut samples, key(0), snapshot)
                .unwrap(),
            Outcome::Deferred
        );
        assert_eq!(
            samples
                .boundary_snapshot(snapshot.key.source, snapshot.key.target)
                .unwrap()
                .1,
            3
        );
        drop(queue_guard);
        drop(registry_guard);
        assert!(queue.pop().unwrap().is_none());
        snapshot = samples.boundary(snapshot.key, true).unwrap();
    }
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Queued
    );
}

#[test]
fn zero_family_reshape_does_not_erase_or_inherit_normal_seed_rejection() {
    let (process, queue, mut samples) = setup(2);
    let seed = snapshot(&process, 0);
    process.admit_seed(&queue, &mut samples, seed).unwrap();
    let mut job = queue.pop().unwrap().unwrap().seed();
    assert!(job.reservation.transition(RUNNING));
    assert!(job.reservation.reject());
    let shape = observed_boundary(&process, &mut samples);
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), shape)
            .unwrap(),
        Outcome::Queued
    );
    assert_eq!(
        process.admit_seed(&queue, &mut samples, seed).unwrap(),
        Outcome::Duplicate
    );
    drop(queue.pop().unwrap());
    assert_eq!(
        process.admit_seed(&queue, &mut samples, seed).unwrap(),
        Outcome::Duplicate
    );
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), shape)
            .unwrap(),
        Outcome::Queued
    );
}
