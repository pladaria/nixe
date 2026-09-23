use super::*;
use std::time::{Duration, Instant};

#[test]
fn ordinary_collection_error_requeues_unit_and_releases_collector_ownership() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    process.retire_unit(old).unwrap();
    drain(&process);
    process.collect_tables().unwrap();
    let usage = process.cache.usage().unwrap();
    process.lock().executions = CheckedCounter::exhausted();
    assert!(matches!(
        process.try_service_links(),
        Err(Error::Exhausted(_))
    ));
    let state = process.lock();
    assert!(!state.units.collecting);
    assert_eq!(state.units.reclaim_len, 1);
    assert_eq!(state.units.reclaim_head, Some(old.0));
    let record = state.units.records.get(old.0).unwrap();
    assert!(record.detached_epoch.is_none());
    assert!(state.units.tables[record.code.code.allocation.segment].is_some());
    // Exhaustion is terminal, not permission to reuse partially detached code.
    assert!(matches!(state.failure, Some(Error::Exhausted(_))));
    drop(state);
    assert_eq!(process.cache.usage().unwrap(), usage);
}

#[test]
fn dispatch_only_maintenance_is_bounded_and_preserves_active_reader_epochs() {
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let retired: Vec<_> = (1..=70)
        .map(|i| {
            let publication = process.reserve(key(i * 4)).unwrap();
            process.retire_dispatch(publication).unwrap();
            publication
        })
        .collect();
    assert!(process.try_service_links().unwrap());
    assert_eq!(process.lock().retired_dispatch.len, 70);
    for publication in &retired {
        assert!(process.lock().dispatch.get(publication.slot).is_some());
    }
    drop(invocation);
    for remaining in [38, 6, 0] {
        assert!(process.try_service_links().unwrap());
        assert_eq!(process.lock().retired_dispatch.len, remaining);
    }
    let state = process.lock();
    let capacity = state.dispatch.capacity();
    for publication in &retired {
        assert!(state.dispatch.get(publication.slot).is_none());
    }
    drop(state);
    for publication in retired {
        let replacement = process.reserve(publication.key).unwrap();
        assert_ne!(replacement.slot, publication.slot);
        assert_eq!(
            process.retire_dispatch(publication),
            Err(Error::StalePublication)
        );
    }
    assert_eq!(process.lock().dispatch.capacity(), capacity);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn cancelled_compile_claim_leaves_retired_slot_to_ordinary_collector() {
    let process = process();
    let mut reader = process.register().unwrap();
    let crate::lifetime::compile::Request::Owner(claim) = reader.claim(key(0)).unwrap() else {
        panic!()
    };
    let publication = claim.publication().unwrap();
    process.retire_dispatch(publication).unwrap();
    drop(claim);
    assert_eq!(process.lock().compilers, 0);
    assert_eq!(process.lock().retired_dispatch.len, 1);
    assert!(process.try_service_links().unwrap());
    assert_eq!(process.lock().retired_dispatch.len, 0);
    assert!(process.lock().dispatch.is_empty());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn repeated_ordinary_maintenance_reuses_dispatch_family_and_unit_storage() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut steady = None;
    let mut addresses = None;
    let mut previous = None;
    for _ in 0..32 {
        let baseline = publish(&process, &cursor, &[0, 4], Tier::Lcq);
        let hcq = publish(&process, &cursor, &[0, 4], Tier::Hcq);
        let publication = process.reserve(key(0)).unwrap();
        let family = process
            .lock()
            .units
            .records
            .get(hcq.0)
            .unwrap()
            .family
            .unwrap();
        if let Some((old_baseline, old_hcq, old_family, old_publication)) = previous {
            assert!(matches!(
                process.snapshot(old_baseline),
                Err(Error::StaleUnit)
            ));
            assert!(matches!(process.snapshot(old_hcq), Err(Error::StaleUnit)));
            assert!(process.lock().units.families.get(old_family).is_none());
            assert_eq!(
                process.retire_dispatch(old_publication),
                Err(Error::StalePublication)
            );
        }
        previous = Some((baseline, hcq, family, publication));
        let actual_addresses = (
            process
                .snapshot(baseline)
                .unwrap()
                .code
                .allocation
                .address(),
            process.snapshot(hcq).unwrap().code.allocation.address(),
        );
        assert_eq!(*addresses.get_or_insert(actual_addresses), actual_addresses);
        let compiler = process.snapshot(hcq).unwrap();
        assert!(matches!(
            process.retire_unit(baseline),
            Err(Error::PinnedBaseline)
        ));
        process.retire_unit(hcq).unwrap();
        drain(&process);
        process.retire_unit(baseline).unwrap();
        drain(&process);
        assert!(process.try_service_links().unwrap());
        assert!(process.lock().units.records.get(hcq.0).is_some());
        drop(compiler);
        assert!(process.try_service_links().unwrap());
        let state = process.lock();
        assert!(state.units.records.is_empty());
        assert!(state.units.families.is_empty());
        assert!(state.units.dependencies.entries.is_empty());
        assert!(state.units.retired_tables.is_empty());
        assert!(state.keys.is_empty());
        assert!(
            state.dispatch.is_empty(),
            "ordinary collection left retired dispatch owners"
        );
        let capacities = (
            state.units.records.capacity(),
            state.units.families.capacity(),
            state.dispatch.capacity(),
        );
        drop(state);
        let usage = process.cache.usage().unwrap();
        // Ordinary maintenance returns spans but deliberately leaves empty
        // segments committed for reuse. Pressure/shutdown own decommit scans.
        assert_eq!(usage.committed, 2 * SEGMENT_BYTES);
        assert_eq!(
            *steady.get_or_insert((capacities, usage)),
            (capacities, usage)
        );
        assert!(matches!(process.snapshot(hcq), Err(Error::StaleUnit)));
        assert!(matches!(process.snapshot(baseline), Err(Error::StaleUnit)));
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}

#[test]
fn in_flight_ordinary_collector_excludes_pressure_and_accepts_new_retirements() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let pinned = publish(&process, &cursor, &[0], Tier::Lcq);
    let free = publish(&process, &cursor, &[4], Tier::Lcq);
    let late = publish(&process, &cursor, &[8], Tier::Lcq);
    let hold = process.snapshot(pinned).unwrap();
    process.collect_tables().unwrap();
    process.retire_unit(pinned).unwrap();
    process.retire_unit(free).unwrap();
    drain(&process);
    std::thread::scope(|scope| {
        // Block the actual collector at the cache mutex, outside JIT state.
        // The guard is dropped before scoped joins, including on assertion failure.
        let (collector, pressure) = process.cache.with_lock_held(|| {
            let collector = scope.spawn(|| process.try_service_links());
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let state = process.lock();
                if state.units.collecting && state.units.reclaim_len == 1 {
                    // The pinned head has rotated. The free unit is now owned
                    // by the collector and waiting to allocate its new table.
                    assert_eq!(state.units.reclaim_head, Some(pinned.0));
                    assert!(state.units.records.get(free.0).is_some());
                    break;
                }
                drop(state);
                assert!(
                    Instant::now() < deadline,
                    "ordinary collector never reached detachment"
                );
                std::thread::yield_now();
            }
            process.retire_unit(late).unwrap();
            drain(&process);
            assert_eq!(process.lock().units.reclaim_len, 2);
            // Neither another maintenance caller nor an exhaustive pressure
            // collector may steal a queue entry from the active owner.
            assert!(process.try_service_links().unwrap());
            assert_eq!(process.reclaim_units().unwrap(), 0);
            assert_eq!(process.lock().units.reclaim_len, 2);
            let pressure = scope.spawn(|| process.recover_capacity());
            loop {
                let state = process.lock();
                if state.phase == Phase::Closed && state.transition_owned {
                    assert!(state.units.collecting);
                    assert!(state.pending[Reason::Eviction as usize].is_some());
                    break;
                }
                drop(state);
                assert!(Instant::now() < deadline, "pressure did not join the stop");
                std::thread::yield_now();
            }
            (collector, pressure)
        });
        assert!(collector.join().unwrap().unwrap());
        pressure.join().unwrap().unwrap();
    });
    assert!(!process.lock().units.collecting);
    assert!(process.lock().units.records.get(free.0).is_none());
    // The original pass had a fixed two-record budget: the arrival after its
    // first pop is eventually consumed by ordinary production maintenance.
    assert!(process.try_service_links().unwrap());
    assert!(process.lock().units.records.get(late.0).is_none());
    assert_eq!(process.lock().units.reclaim_len, 1);
    assert!(process.lock().units.records.get(pinned.0).is_some());
    drop(hold);
    assert!(process.try_service_links().unwrap());
    assert_eq!(process.lock().units.reclaim_len, 0);
    assert!(process.lock().units.records.get(pinned.0).is_none());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn failed_ordinary_directory_allocation_rotates_head_without_starving_other_segments() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let head = publish(&process, &cursor, &[0], Tier::Lcq);
    let baseline = publish(&process, &cursor, &[4], Tier::Lcq);
    let tail = publish(&process, &cursor, &[4], Tier::Hcq);
    assert_ne!(
        process.snapshot(head).unwrap().code.allocation.segment,
        process.snapshot(tail).unwrap().code.allocation.segment
    );
    process.retire_unit(head).unwrap();
    drain(&process);
    process.retire_unit(tail).unwrap();
    drain(&process);
    process.collect_tables().unwrap();
    let charge = process
        .cache
        .charge_metadata(
            HARD_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    assert!(process.try_service_links().unwrap());
    {
        let state = process.lock();
        // The LCQ directory still has the live baseline and needs a copy;
        // the single-unit HCQ directory can be removed without allocation.
        assert!(
            state
                .units
                .records
                .get(head.0)
                .unwrap()
                .detached_epoch
                .is_none()
        );
        assert!(state.units.records.get(tail.0).is_none());
        assert_eq!(state.units.reclaim_len, 1);
        assert!(!state.units.collecting);
    }
    drop(charge);
    assert!(process.try_service_links().unwrap());
    assert!(process.lock().units.records.get(head.0).is_none());
    assert_eq!(process.lock().units.reclaim_len, 0);
    assert!(process.snapshot(baseline).is_ok());
    assert!(process.try_shutdown().unwrap());
}
