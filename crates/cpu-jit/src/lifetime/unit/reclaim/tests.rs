use super::super::tests::{frame, input, key, process, publish};
use super::*;
use crate::executable::{HARD_BYTES, SEGMENT_BYTES, SOFT_BYTES};
use nixe_cpu::state::a64::A64State;

fn drain(process: &Lifetime) {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn ordinary_maintenance_collects_retired_storage_without_pressure_or_shutdown() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let pinned = publish(&process, &cursor, &[0], Tier::Lcq);
    let hold = process.snapshot(pinned).unwrap();
    let units: Vec<_> = (1..=70)
        .map(|i| publish(&process, &cursor, &[i * 4], Tier::Lcq))
        .collect();
    process.retire_unit(pinned).unwrap();
    for &unit in &units {
        process.retire_unit(unit).unwrap();
    }
    drain(&process);
    assert!(process.cache.usage().unwrap().total() < SOFT_BYTES);
    assert_eq!(process.lock().units.reclaim_len, 71);
    assert!(process.try_service_links().unwrap());
    let remaining = process.lock().units.reclaim_len;
    assert!((39..=40).contains(&remaining)); // At most 32 retired records visited.
    for _ in 0..3 {
        assert!(process.try_service_links().unwrap());
    }
    {
        let state = process.lock();
        assert_eq!(state.units.reclaim_len, 1);
        assert!(state.units.records.get(pinned.0).is_some());
        for unit in units {
            assert!(state.units.records.get(unit.0).is_none());
        }
    }
    drop(hold);
    assert!(process.try_service_links().unwrap());
    assert_eq!(process.lock().units.reclaim_len, 0);
    assert!(process.lock().units.records.get(pinned.0).is_none());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn ordinary_collection_preserves_the_directory_readers_second_grace_period() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let address = process.snapshot(old).unwrap().code.allocation.address();
    process.retire_unit(old).unwrap();
    drain(&process);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(4)) }
        .unwrap()
        .unwrap();
    let fault = invocation.fault(address + 12).unwrap();
    assert!(process.try_service_links().unwrap());
    assert!(invocation.fault(address + 12).is_none());
    assert_eq!(fault.unit.code.allocation.address(), address);
    assert_eq!(process.lock().units.reclaim_len, 1);
    drop(invocation);
    assert!(process.try_service_links().unwrap());
    assert!(process.lock().units.records.get(old.0).is_none());
    assert_eq!(process.lock().units.reclaim_len, 0);
}

#[test]
fn cold_pressure_waits_for_memory_authority_without_owning_its_transition() {
    use nixe_memory::ExecutionMutationObserver;
    for stop in [false, true] {
        let process = process();
        let hold = process.clone().begin(&[]).unwrap();
        let worker_process = process.clone();
        let worker = std::thread::spawn(move || worker_process.recover_capacity());
        {
            let mut state = process.lock();
            while state.pending[Reason::Eviction as usize].is_none() {
                state = process.changed.wait(state).unwrap();
            }
            assert!(!state.transition_owned);
            assert!(!worker.is_finished());
        }
        if stop {
            process.request_shutdown().unwrap();
            assert_eq!(worker.join().unwrap(), Err(Error::Shutdown));
            drop(hold);
            assert!(process.try_shutdown().unwrap());
        } else {
            drop(hold);
            worker.join().unwrap().unwrap();
            assert_eq!(process.lock().phase, Phase::Open);
            assert!(process.lock().pending.iter().all(Option::is_none));
        }
    }
}

#[test]
fn faultless_unit_directory_borrow_survives_detachment_and_span_reuse() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut candidate = input(&process, &[0], Tier::Lcq);
    candidate.faults = Box::new([]);
    candidate.states = Box::new([]);
    candidate.code.proofs.as_mut().unwrap().faults = Box::new([]);
    let address = candidate.code.allocation.address();
    let old = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], candidate, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    // Keep another span resident so reuse exercises this same segment table.
    publish(&process, &cursor, &[4], Tier::Lcq);
    process.retire_unit(old).unwrap();
    drain(&process);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let mut invocation = unsafe { reader.admit(&mut frame, key(4)) }
        .unwrap()
        .unwrap();
    let (_, lookup) = invocation.frame_and_faults();
    let unit = lookup.unit(address).unwrap();
    let id = unit.id;
    assert!(unit.faults.is_empty());
    assert!(lookup.find(address + 12).is_none());
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(lookup.unit(address).is_none());
    assert_eq!(unit.id, id); // Detached metadata remains protected by this epoch.
    drop(invocation);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    assert_eq!(
        process.snapshot(new).unwrap().code.allocation.address(),
        address
    );
    let mut invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let (_, lookup) = invocation.frame_and_faults();
    assert_ne!(lookup.unit(address).unwrap().id, id);
    assert_eq!(
        lookup.find(address + 12).unwrap().unit.id,
        lookup.unit(address).unwrap().id
    );
}

#[test]
fn snapshots_pin_actual_code_dependencies_and_slots_until_last_release() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let survivor = publish(&process, &cursor, &[4], Tier::Lcq);
    let snapshot = process.snapshot(old).unwrap();
    let address = snapshot.code.allocation.address();
    let ticket = process.retire_unit(old).unwrap();
    drain(&process);
    assert!(ticket.is_complete().unwrap());
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    let clone = snapshot.clone(); // Existing compiler ownership remains valid.
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(process.lock().units.dependencies.entries.len(), 2);
    let unrelated = input(&process, &[8], Tier::Lcq);
    assert_ne!(unrelated.code.allocation.address(), address);
    drop(unrelated);
    drop(clone);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert_eq!(process.lock().units.dependencies.entries.len(), 1);
    assert!(process.lock().units.records.get(old.0).is_none());
    assert!(process.lock().units.records.get(survivor.0).is_some());
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    assert_ne!(new, old);
    assert_eq!(
        process.snapshot(new).unwrap().code.allocation.address(),
        address
    );
    assert!(matches!(process.retire_unit(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(new).is_ok());
}

#[test]
fn newer_fault_reader_needs_a_second_grace_period_after_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let address = process.snapshot(old).unwrap().code.allocation.address();
    process.retire_unit(old).unwrap();
    drain(&process);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(4)) }
        .unwrap()
        .unwrap();
    let fault = invocation.fault(address + 12).unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(invocation.fault(address + 12).is_none());
    assert_eq!(fault.unit.code.allocation.address(), address); // Old table borrow survives.
    assert!(process.lock().units.records.get(old.0).is_some());
    drop(invocation);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn closure_waits_for_native_invocation_before_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    process.retire_unit(old).unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    std::thread::scope(|scope| {
        let (send, receive) = std::sync::mpsc::channel();
        let process = &process;
        let closer = scope.spawn(move || {
            let mut transition = process.try_transition().unwrap().unwrap();
            send.send(()).unwrap();
            transition.wait_closed().unwrap();
            transition.drain_retirements().unwrap();
            assert_eq!(process.reclaim_units().unwrap(), 1);
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
        });
        receive.recv().unwrap();
        assert_eq!(process.lock().phase, Phase::Closing);
        assert!(
            invocation
                .fault(invocation.payload().preferred().unwrap().canonical.get() + 12)
                .is_some()
        );
        drop(invocation);
        closer.join().unwrap();
    });
}

#[test]
fn empty_segment_republication_uses_same_address_but_fresh_generation() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let snapshot = process.snapshot(old).unwrap();
    let address = snapshot.code.allocation.address();
    let generation = snapshot.code.allocation.generation;
    drop(snapshot);
    process.retire_unit(old).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert_eq!(process.cache.usage().unwrap().committed, 0);
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    let snapshot = process.snapshot(new).unwrap();
    assert_eq!(snapshot.code.allocation.address(), address);
    assert_ne!(snapshot.code.allocation.generation, generation);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.fault(address + 12).unwrap().unit.id, snapshot.id);
}

#[test]
fn decommit_scan_preserves_unpublished_allocations_without_directory_tables() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let staged = input(&process, &[0], Tier::Lcq);
    let address = staged.code.allocation.address();
    let generation = staged.code.allocation.generation;
    // No unit record or directory table exists yet. The cache lease must still
    // prevent the empty-segment check from decommitting this segment.
    assert!(process.lock().units.segment_records.iter().all(|n| *n == 0));
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(process.cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(process.lock().units.decommitting.iter().all(|flag| !flag));
    let unit = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], staged, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let snapshot = process.snapshot(unit).unwrap();
    let segment = snapshot.code.allocation.segment;
    assert_eq!(process.lock().units.segment_records[segment], 1);
    assert_eq!(snapshot.code.allocation.address(), address);
    assert_eq!(snapshot.code.allocation.generation, generation);
    process.retire_unit(unit).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(process.lock().units.segment_records[segment], 1);
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert_eq!(process.lock().units.segment_records[segment], 0);
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}

#[test]
fn segment_record_counts_follow_shared_spans_and_reused_slots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let survivor = publish(&process, &cursor, &[0], Tier::Lcq);
    let segment = process.snapshot(survivor).unwrap().code.allocation.segment;
    for _ in 0..16 {
        let old = publish(&process, &cursor, &[4], Tier::Lcq);
        let retained = process.snapshot(old).unwrap();
        assert_eq!(retained.code.allocation.segment, segment);
        assert_eq!(process.lock().units.segment_records[segment], 2);
        process.retire_unit(old).unwrap();
        drain(&process);
        assert_eq!(process.reclaim_units().unwrap(), 0);
        assert_eq!(process.lock().units.segment_records[segment], 2);
        drop(retained);
        assert_eq!(process.reclaim_units().unwrap(), 1);
        assert_eq!(process.lock().units.segment_records[segment], 1);
        assert_eq!(process.cache.usage().unwrap().committed, SEGMENT_BYTES);
    }
    process.retire_unit(survivor).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(process.lock().units.segment_records.iter().all(|n| *n == 0));
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}

#[test]
fn hcq_retirement_restores_all_baselines_and_releases_family_pins() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[0, 4], Tier::Hcq);
    assert!(matches!(
        process.retire_unit(lcq),
        Err(Error::PinnedBaseline)
    ));
    let family = process
        .lock()
        .units
        .records
        .get(hcq.0)
        .unwrap()
        .family
        .unwrap();
    let compiler = process.snapshot(hcq).unwrap();
    process.retire_unit(hcq).unwrap();
    drain(&process);
    for pc in [0, 4] {
        let state = process.lock();
        let payload = state
            .dispatch
            .get(*state.keys.get(&key(pc)).unwrap())
            .unwrap()
            .snapshot();
        assert!(payload.hcq().is_none());
        assert_eq!(payload.preferred(), payload.lcq());
    }
    assert!(!process.lock().units.families.is_empty()); // Held until span returned.
    process.retire_unit(lcq).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    drop(compiler);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(process.lock().units.families.is_empty());
    assert!(process.lock().units.families.get(family).is_none());
    assert!(process.lock().dispatch.is_empty());
}

#[test]
fn in_flight_family_prevents_baseline_eviction_until_cancelled() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0], Tier::Lcq);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor,
        )
        .unwrap();
    assert!(matches!(
        process.retire_unit(lcq),
        Err(Error::PinnedBaseline)
    ));
    drop(prepared);
    process.retire_unit(lcq).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn lcq_cutover_cannot_be_acknowledged_without_draining_exact_old_owner() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(new).is_ok());
}

#[test]
fn cancelled_partial_cutover_can_be_queued_again_for_eviction() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let replacement = publish(&process, &cursor, &[0], Tier::Lcq);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(process.lock().units.retirements.next(), Some(old.0));
    // The old unit still owns PC 4, so this cutover cancels rather than unlinks.
    assert_eq!(transition.unlink_next().unwrap(), Some(false));
    assert_eq!(transition.unlink_next().unwrap(), None);
    let eviction = process.retire_unit(old).unwrap();
    assert_eq!(process.lock().units.retirements.next(), Some(old.0));
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(eviction.is_complete().unwrap());
    assert!(process.snapshot(replacement).is_ok());
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn newly_queued_hcq_is_drained_before_pending_baselines() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[0], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[0], Tier::Hcq);
    let unrelated = publish(&process, &cursor, &[4], Tier::Lcq);
    process.retire_unit(unrelated).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(transition.unlink_next().unwrap(), Some(true));

    // New work can arrive between unlinks; selection must recheck HCQ every
    // time, even though the older baseline precedes its family in the registry.
    process.invalidate_all_memory().unwrap();
    assert_eq!(process.lock().units.retirements.next(), Some(hcq.0));
    assert_eq!(transition.unlink_next().unwrap(), Some(true));
    assert_eq!(process.lock().units.retirements.next(), Some(baseline.0));
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(process.reclaim_units().unwrap(), 3);
}

#[test]
fn handles_are_process_scoped_and_exhaustion_never_partially_unlinks() {
    let process = process();
    let other = super::super::tests::process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    publish(&other, &cursor, &[0, 4], Tier::Lcq);
    assert!(matches!(other.snapshot(old), Err(Error::StaleUnit)));
    assert!(matches!(other.retire_unit(old), Err(Error::StaleUnit)));
    process.retire_unit(old).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    process.lock().reachabilities = CheckedCounter::exhausted();
    assert!(matches!(
        transition.drain_retirements(),
        Err(Error::Exhausted(_))
    ));
    let state = process.lock();
    for pc in [0, 4] {
        assert!(
            state
                .dispatch
                .get(*state.keys.get(&key(pc)).unwrap())
                .unwrap()
                .snapshot()
                .lcq()
                .is_some()
        );
    }
}

#[test]
fn closed_reclamation_makes_progress_with_no_free_metadata_budget() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    // Model unrelated live charged storage filling the real configured budget.
    let charge = process
        .cache
        .charge_metadata(
            HARD_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.retire_unit(old).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}

#[test]
fn pressure_evicts_oldest_hcq_before_any_lcq_without_waiting_on_snapshots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let first = publish(&process, &cursor, &[0], Tier::Hcq);
    let second = publish(&process, &cursor, &[4], Tier::Hcq);
    let retained = process.snapshot(first).unwrap();
    // Both HCQ units share a segment; neither eviction refunds that backing
    // while the first compiler snapshot remains. The pass must not wait.
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES + SEGMENT_BYTES / 2 - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(matches!(
        transition.relieve_pressure(0, Tier::Hcq),
        Err(Error::Capacity(_))
    ));
    assert!(process.lock().units.records.get(second.0).is_none());
    assert!(process.lock().units.records.get(lcq.0).is_some());
    assert!(process.lock().units.records.get(first.0).is_some());
    drop(retained);
    transition.relieve_pressure(0, Tier::Hcq).unwrap();
    assert!(process.lock().units.records.get(first.0).is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}

#[test]
fn pressure_leaves_headroom_for_the_next_segment_without_another_eviction() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[0], Tier::Lcq);
    let optimized = publish(&process, &cursor, &[0], Tier::Hcq);
    assert_eq!(process.cache.usage().unwrap().committed, 2 * SEGMENT_BYTES);
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.recover_capacity().unwrap();
    // Merely dropping below the trigger would evict HCQ alone. The low-water
    // target also retires the now-unpinned baseline, leaving a segment plus
    // metadata worth of headroom; no retained references are bypassed.
    assert!(process.lock().units.records.get(optimized.0).is_none());
    assert!(process.lock().units.records.get(baseline.0).is_none());
    assert!(process.cache.usage().unwrap().total() <= SOFT_BYTES - 2 * SEGMENT_BYTES);
    assert!(process.lock().units.segment_retired.iter().all(|n| *n == 0));
    let replacement = publish(&process, &cursor, &[4], Tier::Lcq);
    assert!(!process.cache.usage().unwrap().needs_reclamation());
    process.recover_capacity().unwrap();
    assert!(process.snapshot(replacement).is_ok());
    drop(charge);
}

#[test]
fn eviction_batches_keep_creation_order_after_slot_reuse_and_prioritize_hcq() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let units: Vec<_> = (0..EVICTION_BATCH + 8)
        .map(|index| publish(&process, &cursor, &[4 * index as u64], Tier::Lcq))
        .collect();
    for unit in &units[..4] {
        process.retire_unit(*unit).unwrap();
    }
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 4);
    for index in 0..4 {
        publish(&process, &cursor, &[4 * index], Tier::Lcq);
    }
    {
        let state = process.lock();
        let candidates: Vec<_> = state
            .units
            .eviction_candidates()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(candidates.len(), EVICTION_BATCH);
        assert_eq!(
            candidates,
            units[4..4 + EVICTION_BATCH]
                .iter()
                .map(|unit| unit.0)
                .collect::<Vec<_>>()
        );
    }
    let hcq = publish(&process, &cursor, &[0], Tier::Hcq);
    assert_eq!(
        process
            .lock()
            .units
            .eviction_candidates()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec![hcq.0]
    );
    process.retire_unit(hcq).unwrap();
    drain(&process);
    process.reclaim_units().unwrap();
    assert!(process.lock().units.segment_retired.iter().all(|n| *n == 0));
}

#[test]
fn shutdown_waits_for_snapshots_and_unpublished_outputs_then_unmaps() {
    let process = process();
    let cache = Arc::clone(&process.cache);
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0], Tier::Lcq);
    let snapshot = process.snapshot(lcq).unwrap();
    let staged = input(&process, &[4], Tier::Lcq);
    let reader = process.register().unwrap();
    let address = snapshot.code.allocation.address();
    let ticket = process.request(Reason::Shutdown).unwrap();
    // Identify this backing, not merely its address: another parallel test
    // may reserve the same virtual range as soon as shutdown unmaps it.
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    let backing = maps
        .lines()
        .find_map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            let (start, end) = fields[0].split_once('-').unwrap();
            let start = usize::from_str_radix(start, 16).unwrap();
            let end = usize::from_str_radix(end, 16).unwrap();
            (start <= address && address < end)
                .then(|| (fields[3].to_owned(), fields[4].to_owned()))
        })
        .unwrap();
    assert_ne!(backing.1, "0");
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.try_finish_shutdown().unwrap());
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    drop(snapshot);
    assert!(!transition.try_finish_shutdown().unwrap());
    drop(staged);
    assert!(transition.try_finish_shutdown().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(ticket.is_complete().unwrap());
    assert!(transition.try_reopen().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
    assert_eq!(process.lock().units.records.capacity(), 0);
    assert_eq!(process.lock().dispatch.capacity(), 0);
    assert_eq!(process.lock().readers.capacity(), 0);
    assert!(matches!(
        process.cache.charge_metadata(1, Tier::Lcq),
        Err(crate::executable::Error::Closed)
    ));
    for line in std::fs::read_to_string("/proc/self/maps").unwrap().lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert!(fields[3] != backing.0 || fields[4] != backing.1);
    }
    drop(reader); // Inactive registrations can outlive terminal cleanup.
    drop(transition);
    drop(process);
    assert_eq!(
        cache.usage().unwrap().metadata,
        size_of::<crate::executable::Cache>() + 2 * size_of::<usize>()
    );
}

#[test]
fn partial_lcq_replacement_retains_other_roots_until_final_cutover() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    publish(&process, &cursor, &[0], Tier::Lcq);
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(process.snapshot(old).is_ok());
    publish(&process, &cursor, &[4], Tier::Lcq);
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
}

#[test]
fn requests_arriving_during_closed_keep_their_exact_targets_pending() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let first = publish(&process, &cursor, &[0], Tier::Lcq);
    let second = publish(&process, &cursor, &[4], Tier::Lcq);
    let first_ticket = process.retire_unit(first).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    let batch = transition.batch().unwrap();
    let second_ticket = process.retire_unit(second).unwrap();
    batch.complete().unwrap();
    assert!(first_ticket.is_complete().unwrap());
    assert!(!second_ticket.is_complete().unwrap());
    assert!(!transition.try_reopen().unwrap());
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(process.reclaim_units().unwrap(), 2);
    transition.batch().unwrap().complete().unwrap();
    assert!(second_ticket.is_complete().unwrap());
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn pressure_returns_empty_compiler_reservations_without_reviving_old_handles() {
    let process = process();
    let old = process.reserve(key(0)).unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.relieve_pressure(0, Tier::Lcq).unwrap();
    assert!(process.lock().keys.is_empty());
    assert!(process.lock().dispatch.is_empty());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    let new = process.reserve(key(0)).unwrap();
    assert_ne!(old.slot, new.slot);
    assert_eq!(process.retire_dispatch(old), Err(Error::StalePublication));
}

#[test]
fn repeated_lcq_hcq_churn_reuses_bounded_metadata_and_executable_storage() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut expected = None;
    for _ in 0..128 {
        let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
        let hcq = publish(&process, &cursor, &[0, 4], Tier::Hcq);
        process.retire_unit(hcq).unwrap();
        drain(&process);
        process.retire_unit(lcq).unwrap();
        drain(&process);
        assert_eq!(process.reclaim_units().unwrap(), 2);
        let state = process.lock();
        assert!(state.units.records.is_empty());
        assert!(state.units.families.is_empty());
        assert!(state.dispatch.is_empty());
        assert!(state.keys.is_empty());
        assert!(state.units.dependencies.entries.is_empty());
        let capacities = (
            state.units.records.capacity(),
            state.units.families.capacity(),
            state.dispatch.capacity(),
        );
        drop(state);
        let usage = process.cache.usage().unwrap();
        assert_eq!(usage.committed, 0);
        assert_eq!(
            *expected.get_or_insert((capacities, usage)),
            (capacities, usage)
        );
    }
}

#[test]
fn compiler_table_snapshot_blocks_in_place_mutation_not_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(8)).unwrap()],
            input(&process, &[8], Tier::Lcq),
            &cursor,
        )
        .unwrap();
    let charge = process
        .cache
        .charge_metadata(
            HARD_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.retire_unit(old).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(process.lock().units.records.get(old.0).is_some());
    drop(prepared);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}

#[test]
fn pressure_uses_creation_order_and_lcq_reports_hard_capacity_precisely() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let first = publish(&process, &cursor, &[0], Tier::Hcq);
    let second = publish(&process, &cursor, &[4], Tier::Hcq);
    let snapshots = [
        process.snapshot(first).unwrap(),
        process.snapshot(second).unwrap(),
    ];
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES + SEGMENT_BYTES / 2 - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(matches!(
        transition.relieve_pressure(0, Tier::Hcq),
        Err(Error::Capacity(_))
    ));
    {
        let state = process.lock();
        let Lifecycle::Retired(a) = state.units.records.get(first.0).unwrap().lifecycle else {
            panic!()
        };
        let Lifecycle::Retired(b) = state.units.records.get(second.0).unwrap().lifecycle else {
            panic!()
        };
        assert!(a < b);
        assert_eq!(
            state.units.records.get(lcq.0).unwrap().lifecycle,
            Lifecycle::Published
        );
    }
    assert!(matches!(
        transition.relieve_pressure(HARD_BYTES, Tier::Lcq),
        Err(Error::Capacity("640 MiB code+metadata hard limit"))
    ));
    drop(snapshots);
    assert_eq!(process.reclaim_units().unwrap(), 2);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}
