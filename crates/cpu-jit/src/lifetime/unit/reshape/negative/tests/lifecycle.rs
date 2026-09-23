//! Lifecycle fixtures inject index records directly to isolate invalidation.
//! Worker-result tests cover validated installation and admission separately.
use super::*;
use crate::lifetime::unit::tests::input;
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidationKind,
};

mod selection;

fn install(process: &Lifetime, key: Key, owners: &[Owner]) {
    let mut spare = Storage::prepare(&process.cache, 8, 16).unwrap();
    let mut prepared = record(process, key, owners);
    let mut state = process.lock();
    state.units.negatives.grow(&mut spare).unwrap();
    assert!(state.units.negatives.insert(&mut prepared).unwrap());
    drop(state);
    drop(spare);
}

fn present(process: &Lifetime, key: Key) -> bool {
    process.lock().units.negatives.get(key).is_some()
}

fn drain(process: &Lifetime) {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn retirement_invalidates_exact_unit_evidence_before_snapshot_release() {
    let process = process();
    let a = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    let b = publish(&process, &AtomicU64::new(0), &[32], Tier::Lcq);
    let a_key = boundary(&process, 0);
    let b_key = boundary(&process, 32);
    install(&process, a_key, &[Owner::Unit(a)]);
    install(&process, b_key, &[Owner::Unit(b)]);
    let snapshot = process.snapshot(a).unwrap();
    let before = process.cache.usage().unwrap().metadata;
    process.retire_unit(a).unwrap();
    assert!(!present(&process, a_key));
    assert!(present(&process, b_key));
    assert!(process.lock().units.negatives.removed.is_none());
    assert!(process.cache.usage().unwrap().metadata < before);
    assert_eq!(snapshot.registered_handle(), Some(a));
    drain(&process);
    assert!(present(&process, b_key)); // Unrelated maintenance is not evidence.
}

#[test]
fn ordinary_churn_returns_aborted_family_and_negative_record_charges() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut steady = None;
    for _ in 0..32 {
        let baseline = publish(&process, &cursor, &[0, 4], Tier::Lcq);
        let publications = [
            process.reserve(key(0)).unwrap(),
            process.reserve(key(4)).unwrap(),
        ];
        let abandoned = process
            .prepare_unit(&publications, input(&process, &[0, 4], Tier::Hcq), &cursor)
            .unwrap();
        assert!(matches!(
            process.retire_unit(baseline),
            Err(Error::PinnedBaseline)
        ));
        drop(abandoned);
        assert_eq!(
            process
                .snapshot(baseline)
                .unwrap()
                .baseline_pins
                .load(Ordering::Relaxed),
            0
        );
        let family = publish(&process, &cursor, &[0, 4], Tier::Hcq);
        let compiler = process.snapshot(family).unwrap();
        let evidence = boundary(&process, 0);
        install(
            &process,
            evidence,
            &[Owner::Unit(family), dispatch(&process, 0)],
        );
        let charged = process.cache.usage().unwrap().metadata;
        process.retire_unit(family).unwrap();
        assert!(!present(&process, evidence));
        assert!(process.lock().units.negatives.removed.is_none());
        assert!(process.cache.usage().unwrap().metadata < charged);
        drain(&process);
        process.retire_unit(baseline).unwrap();
        drain(&process);
        assert!(process.try_service_links().unwrap());
        drop(compiler);
        assert!(process.try_service_links().unwrap());
        let state = process.lock();
        assert!(state.units.records.is_empty());
        assert!(state.units.families.is_empty());
        assert!(state.units.dependencies.entries.is_empty());
        assert!(state.units.retired_tables.is_empty());
        assert!(state.dispatch.is_empty());
        assert_eq!(state.retired_dispatch.len, 0);
        assert!(state.units.negatives.records.is_empty());
        assert!(state.units.negatives.keys.is_empty());
        assert!(state.units.negatives.heads.is_empty());
        assert!(state.units.negatives.removed.is_none());
        let capacity = (
            state.units.negatives.records.capacity(),
            state.units.negatives.keys.capacity(),
            state.units.negatives.heads.capacity(),
        );
        drop(state);
        let usage = process.cache.usage().unwrap();
        // Index/registry capacity remains charged and reusable, but no live
        // record, directory snapshot or abandoned staging owner accumulates.
        assert_eq!(*steady.get_or_insert((capacity, usage)), (capacity, usage));
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}

#[test]
fn publication_invalidates_dispatch_evidence_and_replaced_lcq_units_only() {
    for tier in [Tier::Lcq, Tier::Hcq] {
        let process = process();
        let unit = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
        let original = boundary(&process, 0);
        let other = boundary(&process, 32);
        let mut unit_key = original;
        unit_key.source = key(8); // A separate record tracking an inspected input.
        let slot = dispatch(&process, 0);
        let survivor = dispatch(&process, 32);
        install(&process, original, &[slot]);
        install(&process, unit_key, &[Owner::Unit(unit)]);
        install(&process, other, &[survivor]);
        publish(&process, &AtomicU64::new(0), &[0], tier);
        assert!(!present(&process, original));
        assert_eq!(present(&process, unit_key), tier == Tier::Hcq);
        assert!(present(&process, other));
        assert!(process.lock().units.negatives.removed.is_none());
    }
}

#[test]
fn dispatch_retirement_and_reuse_do_not_keep_stale_evidence() {
    let process = process();
    let old_key = boundary(&process, 0);
    let publication = process.reserve(key(0)).unwrap();
    let old = Owner::Dispatch(publication.slot);
    install(&process, old_key, &[old]);
    process.retire_dispatch(publication).unwrap();
    assert!(!present(&process, old_key));
    process.collect_dispatch().unwrap();
    let new_key = boundary(&process, 0);
    let new = dispatch(&process, 0);
    assert_ne!(old, new);
    install(&process, new_key, &[new]);
    process.lock().units.negatives.invalidate(old);
    assert!(present(&process, new_key));
}

fn image(process: &Lifetime, pc: u64, space: u64, page: u64) -> UnitHandle {
    let mut candidate = input(process, &[pc], Tier::Lcq);
    let mut entry = key(pc);
    entry.address_space = AddressSpaceId::new(space);
    candidate.entries[0].key = entry;
    candidate.instructions[0].key = InstructionKey::new(entry).unwrap();
    candidate.faults[0].instruction = candidate.instructions[0].key;
    candidate.dependencies[0].page = GuestPhysicalPageId::new(page);
    process
        .prepare_unit(
            &[process.reserve(entry).unwrap()],
            candidate,
            &AtomicU64::new(0),
        )
        .unwrap()
        .publish()
        .unwrap()
}

#[test]
fn exact_memory_changes_remove_affected_results_not_unrelated_boundaries() {
    for change in [
        MemoryInvalidationKind::ExecutableContent {
            first: GuestPhysicalPageId::new(1),
            second: None,
        },
        MemoryInvalidationKind::Mapping {
            address_space: AddressSpaceId::new(1),
            start: GuestVirtualAddress::new(0),
            size: 4,
        },
        MemoryInvalidationKind::InstructionCache {
            address_space: AddressSpaceId::new(1),
        },
    ] {
        let process = process();
        let a = image(&process, 0, 1, 1);
        let b = image(&process, 32, 2, 2);
        let a_key = boundary(&process, 64);
        let b_key = boundary(&process, 72);
        install(&process, a_key, &[Owner::Unit(a)]);
        install(&process, b_key, &[Owner::Unit(b)]);
        let snapshot = process.snapshot(a).unwrap();
        process.invalidate_memory(&[change]).unwrap();
        assert!(!present(&process, a_key));
        assert!(present(&process, b_key));
        assert!(process.lock().units.negatives.removed.is_none());
        drain(&process);
        assert!(present(&process, b_key));
        assert_eq!(snapshot.registered_handle(), Some(a));
    }
}

#[test]
fn invalidating_a_retained_baseline_also_invalidates_participant_evidence() {
    let process = process();
    // This HCQ unit covers only 0, but its retained LCQ image also includes 4.
    let mut baseline = input(&process, &[0, 4], Tier::Lcq);
    baseline.entries = baseline.entries.into_vec().into_iter().take(1).collect();
    process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            baseline,
            &AtomicU64::new(0),
        )
        .unwrap()
        .publish()
        .unwrap();
    let family = publish(&process, &AtomicU64::new(0), &[0], Tier::Hcq);
    process.try_service_links().unwrap();
    let key = boundary(&process, 0);
    install(&process, key, &[Owner::Unit(family)]);
    process
        .invalidate_memory(&[MemoryInvalidationKind::Mapping {
            address_space: AddressSpaceId::new(1),
            start: GuestVirtualAddress::new(4),
            size: 4,
        }])
        .unwrap();
    assert!(!present(&process, key));
    assert!(process.lock().units.negatives.removed.is_none());
    drain(&process);
}

#[test]
fn unlink_invalidates_dispatch_evidence_before_reuse() {
    let process = process();
    let unit = publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    let key = boundary(&process, 0);
    install(&process, key, &[dispatch(&process, 0)]);
    process.retire_unit(unit).unwrap();
    // An endpoint-only fixture has no Unit association; unlink must invalidate
    // the dispatch evidence itself, not rely on that earlier unit notification.
    assert!(present(&process, key));
    drain(&process);
    assert!(!present(&process, key));
    assert!(process.lock().units.negatives.removed.is_none());
}

#[test]
fn history_loss_clears_all_evidence_but_keeps_reusable_index_capacity() {
    let process = process();
    let key = boundary(&process, 0);
    let owner = dispatch(&process, 0);
    install(&process, key, &[owner]);
    let capacity = process.lock().units.negatives.records.capacity();
    process.invalidate_all_memory().unwrap();
    assert!(!present(&process, key));
    assert!(process.lock().units.negatives.removed.is_none());
    assert_eq!(process.lock().units.negatives.records.capacity(), capacity);
    drain(&process);
    install(&process, key, &[owner]);
    assert!(present(&process, key));
}

#[test]
fn shutdown_releases_records_then_index_capacity() {
    for direct in [false, true] {
        let process = process();
        let key = boundary(&process, 0);
        install(&process, key, &[dispatch(&process, 0)]);
        if direct {
            process.request(Reason::Shutdown).unwrap();
        } else {
            process.request_shutdown().unwrap();
        }
        assert!(!present(&process, key));
        assert!(process.lock().units.negatives.removed.is_none());
        assert!(process.try_shutdown().unwrap());
        let state = process.lock();
        assert_eq!(state.units.negatives.records.capacity(), 0);
        assert_eq!(state.units.negatives.keys.capacity(), 0);
        assert_eq!(state.units.negatives.heads.capacity(), 0);
        assert!(state.units.negatives.charge.is_none());
    }
}

#[test]
fn static_root_changes_invalidate_only_the_target_familys_entry_evidence() {
    use crate::lifetime::unit::links;
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0, 4, 32], Tier::Lcq);
    let family = publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Hcq);
    let other_family = publish(&process, &AtomicU64::new(0), &[32], Tier::Hcq);
    let entries = boundary(&process, 0);
    let other = boundary(&process, 32);
    let mut code = entries;
    code.source = key(16);
    install(&process, entries, &[Owner::Entries(family)]);
    install(&process, code, &[Owner::Unit(family)]);
    install(&process, other, &[Owner::Entries(other_family)]);
    let source = links::tests::source(&process, &AtomicU64::new(0), 128, 4);
    assert!(!present(&process, entries));
    assert!(present(&process, code));
    assert!(present(&process, other));
    assert!(process.lock().units.negatives.removed.is_none());
    install(&process, entries, &[Owner::Entries(family)]);
    process.retire_unit(source).unwrap();
    drain(&process);
    assert!(!present(&process, entries));
    assert!(present(&process, code));
    assert!(present(&process, other));
    assert!(process.lock().units.negatives.removed.is_none());
}

#[test]
fn pic_insert_weak_reuse_and_reader_drop_invalidate_entry_evidence_but_hits_do_not() {
    use crate::lifetime::unit::{EdgeKind, dynamic};
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Lcq);
    let family = publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Hcq);
    let boundary = boundary(&process, 0);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let mut other = process.register().unwrap();
    let bridge = || {
        process
            .prepare_dynamic_bridge(source, 0, key(4))
            .unwrap()
            .unwrap()
    };
    install(&process, boundary, &[Owner::Entries(family)]);
    reader.cache_bridge(bridge()).unwrap();
    assert!(!present(&process, boundary));
    assert!(process.lock().units.negatives.removed.is_none());
    install(&process, boundary, &[Owner::Entries(family)]);
    reader.cache_bridge(bridge()).unwrap(); // Same private way: no root change.
    assert!(present(&process, boundary));
    other.cache_bridge(bridge()).unwrap(); // Weak reuse creates a new root.
    assert!(!present(&process, boundary));
    assert!(process.lock().units.negatives.removed.is_none());
    install(&process, boundary, &[Owner::Entries(family)]);
    drop(other);
    assert!(!present(&process, boundary));
    assert!(process.lock().units.negatives.removed.is_none());
}

#[test]
fn baseline_publication_inside_family_invalidates_entry_evidence_without_retiring_family() {
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Lcq);
    let family = publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Hcq);
    let snapshot = process.snapshot(family).unwrap();
    let entries = boundary(&process, 0);
    let mut code = entries;
    code.source = key(16);
    install(&process, entries, &[Owner::Entries(family)]);
    install(&process, code, &[Owner::Unit(family)]);
    publish(&process, &AtomicU64::new(0), &[4], Tier::Lcq);
    assert!(!present(&process, entries));
    assert!(present(&process, code));
    assert!(process.lock().units.negatives.removed.is_none());
    // Retirement invalidates both kinds, even while a snapshot keeps code alive.
    install(&process, entries, &[Owner::Entries(family)]);
    process.retire_unit(family).unwrap();
    assert!(!present(&process, entries));
    assert!(!present(&process, code));
    drop(snapshot);
}
