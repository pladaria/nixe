use super::*;
use crate::lifetime::unit::tests::{key, process, publish};

mod lifecycle;
mod reservation;

fn boundary(process: &Lifetime, pc: u64) -> Key {
    let source = process.reserve(key(pc)).unwrap();
    let target = process.reserve(key(pc + 4)).unwrap();
    Key {
        source: key(pc),
        boundary: BoundaryKey {
            source: InstructionKey::new(key(pc)).unwrap(),
            target: InstructionKey::new(key(pc + 4)).unwrap(),
            source_version: source.reachability,
            target_version: target.reachability,
            source_family: None,
            target_family: None,
        },
    }
}

fn dispatch(process: &Lifetime, pc: u64) -> Owner {
    Owner::Dispatch(process.reserve(key(pc)).unwrap().slot)
}

fn storage(process: &Lifetime, count: usize, owners: usize) -> Index {
    let mut index = Index::new();
    let mut spare = Storage::prepare(&process.cache, count, owners).unwrap();
    index.grow(&mut spare).unwrap();
    index
}

fn record(process: &Lifetime, key: Key, owners: &[Owner]) -> Option<Box<Record>> {
    Some(
        Record::prepare(
            &process.cache,
            key,
            Rejection::Unchanged,
            MemoryInvalidationCursor::new(42),
            owners.iter().copied(),
        )
        .unwrap(),
    )
}

#[test]
fn negative_identity_includes_logical_root_and_both_versions() {
    let process = process();
    let original = boundary(&process, 0);
    let owner = dispatch(&process, 0);
    let mut other_root = original;
    other_root.source = key(8);
    let other_versions = boundary(&process, 16);
    let mut source_version = original;
    source_version.boundary.source_version = other_versions.boundary.source_version;
    let mut target_version = original;
    target_version.boundary.target_version = other_versions.boundary.target_version;
    let mut index = storage(&process, 4, 1);
    for key in [original, other_root, source_version, target_version] {
        assert!(index.insert(&mut record(&process, key, &[owner])).unwrap());
    }
    assert_eq!(index.keys.len(), 4);
    for key in [original, other_root, source_version, target_version] {
        assert_eq!(index.get(key).unwrap().reason, Rejection::Unchanged);
        assert_eq!(
            index.get(key).unwrap().cursor,
            MemoryInvalidationCursor::new(42)
        );
    }
    let mut duplicate = record(&process, original, &[owner]);
    assert!(!index.insert(&mut duplicate).unwrap());
    assert!(duplicate.is_some());
    index.invalidate(owner);
    assert!(index.get(original).is_none());
    drop(index.take_removed());
}

#[test]
fn negative_evidence_is_deduplicated_charged_and_released_after_detach() {
    let process = process();
    let key = boundary(&process, 0);
    let owner = dispatch(&process, 0);
    let before = process.cache.usage().unwrap().metadata;
    let mut index = Index::new();
    assert_eq!(process.cache.usage().unwrap().metadata, before);
    let mut spare = Storage::prepare(&process.cache, 1, 1).unwrap();
    let index_bytes = spare.records.capacity() * size_of::<Slot<Box<Record>>>()
        + spare.keys.allocation_size()
        + spare.heads.allocation_size();
    index.grow(&mut spare).unwrap();
    drop(spare);
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        before + index_bytes
    );
    let mut prepared = record(&process, key, &[owner, owner, owner]);
    assert_eq!(prepared.as_ref().unwrap().associations.len(), 1);
    let record_bytes = size_of::<Record>() + size_of::<Association>();
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        before + index_bytes + record_bytes
    );
    index.insert(&mut prepared).unwrap();
    assert!(prepared.is_none());
    index.invalidate(owner);
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        before + index_bytes + record_bytes
    );
    drop(index.take_removed());
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        before + index_bytes
    );
    drop(index);
    assert_eq!(process.cache.usage().unwrap().metadata, before);
}

#[test]
fn negative_invalidation_detaches_middle_head_and_tail_associations() {
    let process = process();
    let keys: Vec<_> = (0..6).map(|i| boundary(&process, i * 8)).collect();
    let common = dispatch(&process, 0);
    let owners: Vec<_> = (0..6).map(|i| dispatch(&process, 64 + i * 4)).collect();
    let mut index = storage(&process, 6, 7);
    for i in 0..6 {
        index
            .insert(&mut record(&process, keys[i], &[common, owners[i]]))
            .unwrap();
    }
    for i in [2, 5, 0] {
        index.invalidate(owners[i]);
        assert!(index.get(keys[i]).is_none());
        for j in [1, 3, 4] {
            assert!(index.get(keys[j]).is_some());
        }
    }
    index.invalidate(common);
    assert!(index.heads.is_empty());
    assert!(index.keys.is_empty());
    assert!(index.records.is_empty());
    drop(index.take_removed());
}

#[test]
fn negative_capacity_failure_never_evicts_or_partially_attaches() {
    let process = process();
    let a = boundary(&process, 0);
    let b = boundary(&process, 8);
    let owner = dispatch(&process, 0);
    let mut index = storage(&process, 1, 1);
    index.insert(&mut record(&process, a, &[owner])).unwrap();
    let old_handle = *index.keys.iter().next().unwrap();
    let mut pending = record(&process, b, &[owner]);
    assert!(matches!(
        index.insert(&mut pending),
        Err(Error::Capacity(_))
    ));
    assert!(pending.is_some());
    assert!(index.get(a).is_some());
    assert!(index.get(b).is_none());
    index.invalidate(owner);
    // Reuse registry and hash capacity while retired storage remains charged.
    assert!(index.insert(&mut pending).unwrap());
    let new_handle = *index.keys.iter().next().unwrap();
    assert_ne!(old_handle, new_handle);
    assert!(index.records.get(old_handle).is_none());
    drop(index.take_removed());
    assert!(index.get(b).is_some());
}

#[test]
fn negative_growth_preserves_handles_links_and_rejects_stale_plans() {
    let process = process();
    let a = boundary(&process, 0);
    let b = boundary(&process, 8);
    let owner = dispatch(&process, 0);
    let mut index = storage(&process, 1, 1);
    index.insert(&mut record(&process, a, &[owner])).unwrap();
    let handle = *index.keys.iter().next().unwrap();
    let mut large = Storage::prepare(&process.cache, 8, 8).unwrap();
    index.grow(&mut large).unwrap();
    drop(large);
    assert!(index.records.get(handle).is_some());
    index.insert(&mut record(&process, b, &[owner])).unwrap();
    let mut stale = Storage::prepare(&process.cache, 2, 2).unwrap();
    assert!(matches!(index.grow(&mut stale), Err(Error::Capacity(_))));
    assert!(index.get(a).is_some());
    assert!(index.get(b).is_some());
    index.invalidate(owner);
    assert!(index.keys.is_empty());
    assert!(index.heads.is_empty());
}

#[test]
fn stale_dispatch_invalidation_cannot_remove_a_reused_generation() {
    let process = process();
    let original = process.reserve(key(0)).unwrap();
    let old = Owner::Dispatch(original.slot);
    process.retire_dispatch(original).unwrap();
    assert_eq!(process.collect_dispatch().unwrap(), 1);
    let current = dispatch(&process, 0);
    assert_ne!(old, current);
    let key = boundary(&process, 0);
    let mut index = storage(&process, 1, 1);
    index
        .insert(&mut record(&process, key, &[current]))
        .unwrap();
    index.invalidate(old);
    assert!(index.get(key).is_some());
    index.invalidate(current);
    assert!(index.get(key).is_none());
}

#[test]
fn unit_associations_and_large_garbage_chains_retain_no_code() {
    let process = process();
    let unit = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    let owners = [Owner::Unit(unit)];
    let references = Arc::strong_count(&process.lock().units.records.get(unit.0).unwrap().code);
    let mut index = storage(&process, 1024, 1);
    let mut key = boundary(&process, 0);
    for pc in 0..1024 {
        key.source = crate::lifetime::unit::tests::key(pc * 4);
        index.insert(&mut record(&process, key, &owners)).unwrap();
    }
    assert_eq!(
        Arc::strong_count(&process.lock().units.records.get(unit.0).unwrap().code),
        references
    );
    index.invalidate(owners[0]);
    assert!(index.keys.is_empty());
    // Drop a long retired list without recursive record destruction.
    drop(index.take_removed());
    assert!(index.removed.is_none());
}

#[test]
fn metadata_pressure_refuses_new_storage_without_losing_existing_results() {
    let process = process();
    let key = boundary(&process, 0);
    let owner = dispatch(&process, 0);
    let mut index = storage(&process, 1, 1);
    index.insert(&mut record(&process, key, &[owner])).unwrap();
    let before = process.cache.usage().unwrap();
    let pressure = process
        .cache
        .charge_metadata(crate::executable::HARD_BYTES - before.total(), Tier::Lcq)
        .unwrap();
    assert!(matches!(
        Storage::prepare(&process.cache, 2, 2),
        Err(Error::Capacity(_))
    ));
    assert!(matches!(
        Record::prepare(
            &process.cache,
            key,
            Rejection::Disconnected,
            MemoryInvalidationCursor::INITIAL,
            [owner]
        ),
        Err(Error::Capacity(_))
    ));
    assert!(index.get(key).is_some());
    drop(pressure);
    assert_eq!(process.cache.usage().unwrap().metadata, before.metadata);
    index.invalidate(owner);
    drop(index.take_removed());
}
