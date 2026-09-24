use super::*;
use crate::lifetime::unit::tests::{frame, input, key, process};
use crate::lifetime::{Phase, compile::Request};
use nixe_cpu::state::a64::A64State;
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration};

fn publish_image(
    process: &Lifetime,
    cursor: &AtomicU64,
    pcs: &[u64],
    space: u64,
    page: u64,
) -> UnitHandle {
    let mut candidate = input(process, pcs, Tier::Lcq);
    candidate.entries = candidate.entries.into_iter().take(1).collect();
    let change_key = |mut key: crate::abi::BlockKey| {
        key.address_space = AddressSpaceId::new(space);
        key
    };
    candidate.entries[0].key = change_key(candidate.entries[0].key);
    for instruction in &mut candidate.instructions {
        instruction.key = InstructionKey::new(change_key(instruction.key.block_key())).unwrap();
    }
    candidate.faults[0].instruction = candidate.instructions[0].key;
    candidate.dependencies[0].page = GuestPhysicalPageId::new(page);
    // Different mapping generations must not hide aliases of the same page.
    candidate.dependencies[0].mapping_generation = MappingGeneration::new(space + 1);
    let publication = process.reserve(candidate.entries[0].key).unwrap();
    process
        .prepare_unit(&[publication], candidate, cursor)
        .unwrap()
        .publish()
        .unwrap()
}

fn drain(process: &Lifetime) -> usize {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let retired = transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    retired
}

fn content(page: u64) -> MemoryInvalidationKind {
    MemoryInvalidationKind::ExecutableContent {
        first: GuestPhysicalPageId::new(page),
        second: None,
    }
}

fn mapping(space: u64, start: u64, size: u64) -> MemoryInvalidationKind {
    MemoryInvalidationKind::Mapping {
        address_space: AddressSpaceId::new(space),
        start: GuestVirtualAddress::new(start),
        size,
    }
}

fn pending_units(process: &Lifetime) -> Vec<UnitHandle> {
    let state = process.lock();
    let units = &state.units;
    let mut handles = Vec::new();
    for (head, tier) in [
        (units.retirements.hcq, Tier::Hcq),
        (units.retirements.lcq, Tier::Lcq),
    ] {
        let mut next = head;
        while let Some(handle) = next {
            let unit = UnitHandle(handle, process.identity);
            assert!(
                !handles.contains(&unit),
                "duplicate or cyclic retirement link"
            );
            handles.push(unit);
            let record = units.records.get(handle).expect("live generation");
            assert_eq!(record.code.tier, tier);
            assert!(record.retirement.is_some());
            next = record.retirement_next;
        }
    }
    assert_eq!(
        handles.len(),
        units
            .records
            .values()
            .filter(|r| r.retirement.is_some())
            .count()
    );
    assert!(
        units
            .records
            .values()
            .filter(|r| r.retirement.is_none() && !matches!(r.lifecycle, Lifecycle::Retired(_)))
            .all(|r| r.retirement_next.is_none())
    );
    handles
}

#[test]
fn retirement_lists_exclude_unrelated_units_and_reclaimed_slots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let resident: Vec<_> = (0..16)
        .map(|i| publish_image(&process, &cursor, &[i * 4], 1, i + 1))
        .collect();
    assert!(pending_units(&process).is_empty());
    process.invalidate_memory(&[]).unwrap();
    assert!(pending_units(&process).is_empty());
    assert_eq!(drain(&process), 0);

    process
        .invalidate_memory(&[content(2), content(16), content(2)])
        .unwrap();
    let pending = pending_units(&process);
    assert_eq!(pending.len(), 2);
    assert!(pending.contains(&resident[1]));
    assert!(pending.contains(&resident[15]));
    assert_eq!(drain(&process), 2);
    assert!(pending_units(&process).is_empty());
    assert_eq!(process.reclaim_units().unwrap(), 2);
    let capacity = process.lock().units.records.capacity();

    // Repeated insertion uses the freed slots with fresh generations, not
    // stale work-list nodes. The other fourteen units never enter either list.
    for _ in 0..4 {
        let replacement = publish_image(&process, &cursor, &[4], 1, 2);
        assert_ne!(replacement, resident[1]);
        assert!(pending_units(&process).is_empty());
        process
            .invalidate_memory(&[content(2), content(2)])
            .unwrap();
        assert_eq!(pending_units(&process), [replacement]);
        assert_eq!(drain(&process), 1);
        assert_eq!(process.reclaim_units().unwrap(), 1);
        assert!(pending_units(&process).is_empty());
        assert_eq!(process.lock().units.records.capacity(), capacity);
    }
}

#[test]
fn pending_queries_find_older_sequences_beyond_a_newer_list_head() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish_image(&process, &cursor, &[0], 1, 7);
    let new = publish_image(&process, &cursor, &[4], 1, 8);
    let eviction = process.retire_unit(old).unwrap();
    let first = process.invalidate_memory(&[content(7)]).unwrap();
    let second = process.invalidate_memory(&[content(8)]).unwrap();
    assert_eq!(pending_units(&process), [new, old]);
    let units = &process.lock().units;
    assert!(!units.pending_retirement(Reason::MappingChange, eviction));
    assert!(units.pending_retirement(Reason::Eviction, eviction));
    assert!(units.pending_retirement(Reason::MappingChange, first));
    assert!(units.pending_retirement(Reason::MappingChange, second));
}

#[test]
fn physical_invalidation_finds_aliases_and_both_pages_without_evicting_unrelated_code() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let first = publish_image(&process, &cursor, &[0, 4], 1, 7);
    let alias = publish_image(&process, &cursor, &[0x1000], 2, 7);
    let second = publish_image(&process, &cursor, &[0x2000], 1, 9);
    let survivor = publish_image(&process, &cursor, &[0x3000], 1, 8);
    let ticket = process
        .invalidate_memory(&[MemoryInvalidationKind::ExecutableContent {
            first: GuestPhysicalPageId::new(7),
            second: Some(GuestPhysicalPageId::new(9)),
        }])
        .unwrap();
    assert!(
        !process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, ticket)
            .unwrap()
    );
    assert_eq!(drain(&process), 3);
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, ticket)
            .unwrap()
    );
    for handle in [first, alias, second] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert!(process.snapshot(survivor).is_ok());
    assert_eq!(process.reclaim_units().unwrap(), 3);
    assert_eq!(process.lock().units.dependencies.entries.len(), 1);
}

#[test]
fn mapping_uses_all_instruction_bytes_including_overlapping_non_root_words() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let first = publish_image(&process, &cursor, &[0, 4, 8], 1, 7);
    let overlap = publish_image(&process, &cursor, &[4, 8], 1, 7);
    let other_space = publish_image(&process, &cursor, &[4, 8], 2, 7);
    let adjacent = publish_image(&process, &cursor, &[12], 1, 7);
    process.invalidate_memory(&[mapping(1, 11, 1)]).unwrap();
    assert_eq!(drain(&process), 2);
    assert!(matches!(process.snapshot(first), Err(Error::StaleUnit)));
    assert!(matches!(process.snapshot(overlap), Err(Error::StaleUnit)));
    assert!(process.snapshot(other_space).is_ok());
    assert!(process.snapshot(adjacent).is_ok());
    process.invalidate_memory(&[mapping(1, 12, 0)]).unwrap();
    assert_eq!(drain(&process), 0);
    process
        .invalidate_memory(&[MemoryInvalidationKind::InstructionCache {
            address_space: AddressSpaceId::new(2),
        }])
        .unwrap();
    assert_eq!(drain(&process), 1);
    assert!(process.snapshot(adjacent).is_ok());
}

#[test]
fn final_address_byte_does_not_wrap_the_affected_range() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let low = publish_image(&process, &cursor, &[0], 1, 7);
    publish_image(&process, &cursor, &[u64::MAX - 3], 1, 8);
    assert!(matches!(
        process.invalidate_memory(&[mapping(1, u64::MAX, 2)]),
        Err(Error::InvalidUnit(_))
    ));
    assert_eq!(process.lock().phase, Phase::Open);
    process
        .invalidate_memory(&[mapping(1, u64::MAX, 1)])
        .unwrap();
    assert_eq!(drain(&process), 1);
    assert!(process.snapshot(low).is_ok());
}

#[test]
fn duplicate_requests_and_eviction_keep_each_exact_ticket_pending_until_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let unit = publish_image(&process, &cursor, &[0], 1, 7);
    let eviction = process.retire_unit(unit).unwrap();
    let first = process.invalidate_memory(&[content(7)]).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let batch = transition.batch().unwrap();
    let second = process
        .invalidate_memory(&[content(7), content(7)])
        .unwrap();
    assert_eq!(batch.complete(), Err(Error::MaintenancePending));
    assert!(
        !process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, first)
            .unwrap()
    );
    assert!(
        !process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, second)
            .unwrap()
    );
    assert!(
        !process
            .maintenance_complete(crate::lifetime::Reason::Eviction, eviction)
            .unwrap()
    );
    assert!(!transition.try_reopen().unwrap());
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, first)
            .unwrap()
    );
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, second)
            .unwrap()
    );
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::Eviction, eviction)
            .unwrap()
    );
}

#[test]
fn memory_work_arriving_after_a_batch_snapshot_cannot_be_acknowledged_by_that_batch() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish_image(&process, &cursor, &[0], 1, 7);
    publish_image(&process, &cursor, &[4], 1, 8);
    let first = process.invalidate_memory(&[content(7)]).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    let batch = transition.batch().unwrap();
    let second = process.invalidate_memory(&[content(8)]).unwrap();
    batch.complete().unwrap();
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, first)
            .unwrap()
    );
    assert!(
        !process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, second)
            .unwrap()
    );
    assert!(!transition.try_reopen().unwrap());
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, second)
            .unwrap()
    );
}

#[test]
fn closure_cancels_compiler_claims_waiters_and_prepared_publication_even_without_readers() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut compiler = process.register().unwrap();
    let Request::Owner(claim) = compiler.claim(key(0)).unwrap() else {
        panic!()
    };
    let mut waiting = process.register().unwrap();
    let Request::Wait(waiter) = waiting.claim(key(0)).unwrap() else {
        panic!()
    };
    let prepared = process
        .prepare_unit(
            &[claim.publication().unwrap()],
            input(&process, &[0], Tier::Lcq),
            &cursor,
        )
        .unwrap();
    // No resident page association exists yet. The old admission still dies.
    process.invalidate_memory(&[content(7)]).unwrap();
    assert_eq!(waiter.wait(), Err(Error::Closed));
    assert_eq!(drain(&process), 0);
    assert_eq!(claim.validate(), Err(Error::StalePublication));
    assert!(matches!(prepared.publish(), Err(Error::StalePublication)));
    let Request::Owner(fresh) = waiting.claim(key(0)).unwrap() else {
        panic!()
    };
    drop(claim);
    fresh.validate().unwrap();
}

#[test]
fn active_fault_reader_drains_before_unlink_and_compiler_snapshot_delays_span_reuse() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish_image(&process, &cursor, &[0], 1, 7);
    let snapshot = process.snapshot(old).unwrap();
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + 12)
            .unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let process = &process;
        let closer = scope.spawn(move || {
            process.invalidate_memory(&[content(7)]).unwrap();
            send.send(()).unwrap();
            assert_eq!(drain(process), 1);
        });
        receive
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(process.lock().phase, Phase::Closing);
        assert_eq!(fault.unit.id, snapshot.id);
        assert_eq!(process.reclaim_units().unwrap(), 0);
        drop(invocation);
        closer.join().unwrap();
    });
    assert_eq!(process.reclaim_units().unwrap(), 0);
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn invalidating_a_baseline_also_removes_a_family_using_only_its_unchanged_words() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish_image(&process, &cursor, &[0, 4], 1, 7);
    let hcq = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor,
        )
        .unwrap()
        .publish()
        .unwrap();
    process.invalidate_memory(&[mapping(1, 4, 4)]).unwrap();
    assert_eq!(drain(&process), 2);
    assert!(matches!(process.snapshot(baseline), Err(Error::StaleUnit)));
    assert!(matches!(process.snapshot(hcq), Err(Error::StaleUnit)));
}

#[test]
fn family_invalidation_resolves_exact_pins_and_preserves_unrelated_families() {
    use crate::lifetime::unit::tests::publish;

    for change in [mapping(1, 20, 4), content(18)] {
        let process = process();
        let cursor = AtomicU64::new(0);
        let first = publish_image(&process, &cursor, &[0, 4], 1, 17);
        let second = publish_image(&process, &cursor, &[16, 20], 1, 18);
        let unrelated = publish_image(&process, &cursor, &[32, 36], 1, 19);
        let family = publish(&process, &cursor, &[0, 16], Tier::Hcq);
        let other_family = publish(&process, &cursor, &[32], Tier::Hcq);

        // Common tracking-only/data-mapping mutations must not withdraw code.
        process.invalidate_memory(&[]).unwrap();
        process
            .invalidate_memory(&[mapping(1, 0x1000, 4096)])
            .unwrap();
        assert!(pending_units(&process).is_empty());
        assert_eq!(drain(&process), 0);

        // The changed word/page belongs to the second pinned baseline, not
        // to the family's own image/dependencies. Repeated requests coalesce.
        process.invalidate_memory(&[change]).unwrap();
        process.invalidate_memory(&[change]).unwrap();
        assert_eq!(pending_units(&process), [family, second]);
        assert_eq!(drain(&process), 2);
        for survivor in [first, unrelated, other_family] {
            assert!(process.snapshot(survivor).is_ok());
        }
        for removed in [second, family] {
            assert!(matches!(process.snapshot(removed), Err(Error::StaleUnit)));
        }
        assert_eq!(process.reclaim_units().unwrap(), 2);
    }
}

#[test]
fn family_invalidation_uses_current_generations_in_reused_sparse_registry() {
    use crate::lifetime::unit::tests::publish;

    let process = process();
    let cursor = AtomicU64::new(0);
    // Leave vacant slots so neither CodeUnitId nor dense iteration ordinal
    // can stand in for the registered generational handle.
    for index in 0..32 {
        publish_image(&process, &cursor, &[0x1000 + index * 4], 2, 100 + index);
    }
    process
        .invalidate_memory(&[mapping(2, 0x1000, 4096)])
        .unwrap();
    assert_eq!(drain(&process), 32);
    assert_eq!(process.reclaim_units().unwrap(), 32);
    let capacity = process.lock().units.records.capacity();
    let mut previous = None;
    for _ in 0..3 {
        let baseline = publish_image(&process, &cursor, &[0, 4], 1, 17);
        let family = publish(&process, &cursor, &[0], Tier::Hcq);
        if let Some((old_baseline, old_family)) = previous {
            assert_ne!(baseline, old_baseline);
            assert_ne!(family, old_family);
            assert!(matches!(
                process.snapshot(old_baseline),
                Err(Error::StaleUnit)
            ));
            assert!(matches!(
                process.snapshot(old_family),
                Err(Error::StaleUnit)
            ));
        }
        process.invalidate_memory(&[mapping(1, 4, 4)]).unwrap();
        assert_eq!(pending_units(&process), [family, baseline]);
        assert_eq!(drain(&process), 2);
        assert_eq!(process.reclaim_units().unwrap(), 2);
        assert_eq!(process.lock().units.records.capacity(), capacity);
        previous = Some((baseline, family));
    }
}

#[test]
fn stale_inflight_family_retains_storage_without_blocking_memory_invalidation() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish_image(&process, &cursor, &[0, 4], 1, 7);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor,
        )
        .unwrap();
    process.invalidate_memory(&[mapping(1, 4, 4)]).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(matches!(prepared.publish(), Err(Error::StalePublication)));
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn instruction_cache_invalidation_includes_old_cutovers_but_preserves_other_address_spaces() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish_image(&process, &cursor, &[0, 4], 1, 7);
    let other = publish_image(&process, &cursor, &[8], 2, 8);
    let new = publish_image(&process, &cursor, &[0], 1, 7); // Queues a tier cutover.
    process
        .invalidate_memory(&[MemoryInvalidationKind::InstructionCache {
            address_space: AddressSpaceId::new(1),
        }])
        .unwrap();
    assert_eq!(drain(&process), 2);
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    assert!(matches!(process.snapshot(new), Err(Error::StaleUnit)));
    assert_eq!(process.reclaim_units().unwrap(), 2);
    assert!(process.snapshot(other).is_ok());
}
