use super::*;
use crate::lifetime::background::{Job, Outcome, Queue};
use crate::sampling::Samples;

fn queued(process: &Lifetime, queue: &Queue, pc: u64) -> Job {
    let boundary = boundary(process, pc);
    let mut samples = Samples::new();
    let mut snapshot = None;
    for _ in 0..4 {
        snapshot = samples.boundary(boundary.boundary, true);
    }
    assert_eq!(
        process
            .admit_reshape(queue, &mut samples, key(pc), snapshot.unwrap())
            .unwrap(),
        Outcome::Queued
    );
    queue.pop().unwrap().unwrap()
}

#[test]
fn reserved_slots_cannot_be_consumed_by_other_results_and_survive_growth() {
    let process = process();
    let key = boundary(&process, 0);
    let owner = dispatch(&process, 0);
    let mut index = storage(&process, 1, 1);
    index.reserve_record().unwrap();
    assert_eq!(index.reserved, 1);
    assert!(matches!(index.reserve_record(), Err(Error::Capacity(_))));
    let mut prepared = record(&process, key, &[owner]);
    assert!(matches!(
        insert(&mut index, &mut prepared),
        Err(Error::Capacity(_))
    ));
    assert!(prepared.is_some());
    let (records, owners) = index.reservation_growth().unwrap().unwrap();
    let mut spare = Storage::prepare(&process.cache, records, owners).unwrap();
    index.grow(&mut spare).unwrap();
    assert_eq!(index.reserved, 1);
    assert!(insert(&mut index, &mut prepared).unwrap());
    index.invalidate_all();
    assert_eq!(index.reserved, 1); // Invalidation never manufactures a spare slot.
    index.release_record();
    assert_eq!(index.reserved, 0);
}

#[test]
fn reshape_workers_reserve_before_discovery_and_reuse_storage_after_cancellation() {
    let process = process();
    for pc in [0, 4, 8, 12] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
    }
    let queue = Queue::new(2, &process).unwrap().unwrap();
    let first = process
        .accept_background(queued(&process, &queue, 0))
        .unwrap()
        .unwrap();
    let usage = process.cache.usage().unwrap().metadata;
    let second = process
        .accept_background(queued(&process, &queue, 8))
        .unwrap()
        .unwrap();
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        usage + size_of::<Record>()
    );
    {
        let state = process.lock();
        assert_eq!(state.units.negatives.reserved, 2);
        assert_eq!(state.units.negatives.heads.capacity(), 0);
        assert!(state.units.negatives.keys.is_empty());
        assert!(state.units.negatives.records.is_empty());
    }
    drop(first);
    assert_eq!(process.lock().units.negatives.reserved, 1);
    assert_eq!(process.cache.usage().unwrap().metadata, usage);
    let reused = process
        .accept_background(queued(&process, &queue, 0))
        .unwrap()
        .unwrap();
    assert_eq!(process.lock().units.negatives.reserved, 2);
    drop(reused);
    drop(second);
    assert_eq!(process.lock().units.negatives.reserved, 0);
    assert_eq!(process.lock().compilers, 0);
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        usage - size_of::<Record>()
    );
}

#[test]
fn result_header_and_index_pressure_defer_without_leaking_worker_or_reservation() {
    // First exhaust just the header; then allow the header but not the initial
    // index allocation. Both fail after the initial soft-pressure check passes.
    for available in [1, size_of::<Record>()] {
        let process = process();
        for pc in [0, 4] {
            publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
        }
        let queue = Queue::new(1, &process).unwrap().unwrap();
        let job = queued(&process, &queue, 0);
        let before = process.cache.usage().unwrap();
        let pressure = process
            .cache
            .charge_metadata(
                crate::executable::SOFT_BYTES - before.total() - available,
                Tier::Lcq,
            )
            .unwrap();
        let charged = process.cache.usage().unwrap().metadata;
        assert!(process.accept_background(job).unwrap().is_none());
        assert_eq!(process.cache.usage().unwrap().metadata, charged);
        assert_eq!(process.lock().compilers, 0);
        assert_eq!(process.lock().units.negatives.reserved, 0);
        assert_eq!(process.lock().units.negatives.records.capacity(), 0);
        drop(pressure);
        // The boundary is not marked rejected: it can retry without a version change.
        let retry = process
            .accept_background(queued(&process, &queue, 0))
            .unwrap()
            .unwrap();
        drop(retry);
    }
}

#[test]
fn shutdown_keeps_reserved_storage_until_the_worker_releases_it() {
    let process = process();
    for pc in [0, 4] {
        publish(&process, &AtomicU64::new(0), &[pc], Tier::Lcq);
    }
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let work = process
        .accept_background(queued(&process, &queue, 0))
        .unwrap()
        .unwrap();
    process.request_shutdown().unwrap();
    assert!(!process.try_shutdown().unwrap());
    assert_eq!(process.lock().units.negatives.reserved, 1);
    assert!(work.check().is_err());
    drop(work);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.lock().units.negatives.reserved, 0);
    assert_eq!(process.lock().units.negatives.records.capacity(), 0);
}

#[test]
fn seed_acceptance_never_allocates_reshape_result_storage() {
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let publication = process.reserve(key(0)).unwrap();
    let mut samples = Samples::new();
    let mut snapshot = None;
    for _ in 0..8 {
        snapshot = samples.seed(key(0), publication.reachability, None, true);
    }
    assert_eq!(
        process
            .admit_seed(&queue, &mut samples, snapshot.unwrap())
            .unwrap(),
        Outcome::Queued
    );
    let before = process.cache.usage().unwrap().metadata;
    let work = process
        .accept_background(queue.pop().unwrap().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(process.cache.usage().unwrap().metadata, before);
    assert_eq!(process.lock().units.negatives.reserved, 0);
    assert_eq!(process.lock().units.negatives.records.capacity(), 0);
    drop(work);
}

#[test]
fn stale_queued_reshape_never_reserves_result_storage() {
    let process = process();
    let source = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    publish(&process, &AtomicU64::new(0), &[4], Tier::Lcq);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let job = queued(&process, &queue, 0);
    process.retire_unit(source).unwrap();
    let before = process.cache.usage().unwrap().metadata;
    assert!(process.accept_background(job).unwrap().is_none());
    assert_eq!(process.cache.usage().unwrap().metadata, before);
    assert_eq!(process.lock().compilers, 0);
    assert_eq!(process.lock().units.negatives.reserved, 0);
    assert_eq!(process.lock().units.negatives.records.capacity(), 0);
}

#[test]
fn reserved_installation_consumes_exactly_its_own_slot() {
    let process = process();
    let a = boundary(&process, 0);
    let b = boundary(&process, 8);
    let owner = dispatch(&process, 0);
    let mut index = storage(&process, 3, 1);
    index.reserve_record().unwrap();
    index.reserve_record().unwrap();
    index.reserve_record().unwrap();
    assert!(
        index
            .insert_reserved(&mut record(&process, a, &[owner]))
            .unwrap()
    );
    assert_eq!(index.reserved, 2);
    let mut next = record(&process, b, &[owner]);
    assert!(matches!(
        insert(&mut index, &mut next),
        Err(Error::Capacity(_))
    ));
    assert!(index.insert_reserved(&mut next).unwrap());
    assert_eq!(index.reserved, 1);
    let mut duplicate = record(&process, a, &[owner]);
    assert!(!index.insert_reserved(&mut duplicate).unwrap());
    assert!(duplicate.is_some());
    assert_eq!(index.reserved, 1);
    index.release_record(); // The duplicate worker releases its unused slot.
    assert_eq!(index.reserved, 0);
    assert!(index.get(a).is_some());
    assert!(index.get(b).is_some());
}

#[test]
fn insufficient_evidence_capacity_does_not_consume_reserved_slot_or_partial_record() {
    let process = process();
    let key = boundary(&process, 0);
    let owners: Vec<_> = (0..8).map(|i| dispatch(&process, i * 4)).collect();
    let mut index = storage(&process, 1, 0);
    index.reserve_record().unwrap();
    let mut prepared = record(&process, key, &owners);
    assert!(matches!(
        index.insert_reserved(&mut prepared),
        Err(Error::Capacity(_))
    ));
    assert_eq!(index.reserved, 1);
    assert!(index.keys.is_empty());
    assert!(index.heads.is_empty());
    let (records, owners) = index
        .evidence_growth(prepared.as_ref().unwrap())
        .unwrap()
        .unwrap();
    let mut spare = Storage::prepare(&process.cache, records, owners).unwrap();
    index.grow(&mut spare).unwrap();
    assert!(index.insert_reserved(&mut prepared).unwrap());
    assert_eq!(index.reserved, 0);
}
