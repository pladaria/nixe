use super::*;
use crate::lifetime::Reader;
use crate::lifetime::unit::dynamic::pic::tests::cache;

// Ownership fixture with a static site and an independent dynamic source map.
// Only the real linker/PIC/publication protocol runs here, not an HCQ compiler.
fn mixed_source(process: &Lifetime, cursor: &AtomicU64) -> UnitHandle {
    let mut candidate = source_input(process, 0, 4);
    let mut state = candidate.states[0].state.clone();
    state.site.state_map = 1;
    let mut transfer = **candidate.states[0].transfer.as_ref().unwrap();
    transfer.static_target = None;
    let mut maps = candidate.states.into_vec();
    maps.push(StateRecord {
        native_offset: 8,
        state,
        exit: Some(GuestExit {
            pc: key(0).pc,
            kind: EdgeKind::Indirect,
            block_index: 0,
            instruction_index: 0,
        }),
        transfer: Some(Box::new(transfer)),
    });
    candidate.states = maps.into_boxed_slice();
    let mut backend_maps =
        std::mem::take(&mut candidate.code.proofs.as_mut().unwrap().states).into_vec();
    let mut dynamic = backend_maps[0].clone();
    dynamic.id = 1;
    backend_maps.push(dynamic);
    candidate.code.proofs.as_mut().unwrap().states = backend_maps.into_boxed_slice();
    let source = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], candidate, cursor)
        .unwrap()
        .publish()
        .unwrap();
    assert!(process.try_service_links().unwrap());
    source
}

fn cache_target(process: &Lifetime, reader: &mut Reader, source: UnitHandle) {
    cache(
        reader,
        process
            .prepare_dynamic_bridge(source, 1, key(4))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
}

fn pic_addresses(process: &Lifetime, reader: &Reader) -> Vec<usize> {
    let state = process.lock();
    let table = state.readers.get(reader.handle).unwrap().pic.native_table();
    // No native execution or competing owner in these protocol fixtures.
    (0..crate::native::pic::WAYS)
        .filter_map(|index| unsafe { (*table.add(index)).as_ref().map(|record| record.address) })
        .collect()
}

fn callable_target(process: &Lifetime, source: UnitHandle) -> Option<UnitHandle> {
    let state = process.lock();
    let link = state.units.records.get(source.0).unwrap().static_sites[0].callable?;
    Some(state.units.links.records.get(link).unwrap().target)
}

#[test]
fn hcq_withdrawal_cuts_mixed_roots_before_deferred_baseline_relink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[4, 8], Tier::Lcq);
    let source = mixed_source(&process, &cursor);
    let baseline_address = process
        .snapshot(baseline)
        .unwrap()
        .code
        .allocation
        .address();
    let mut first = process.register().unwrap();
    let mut second = process.register().unwrap();
    cache_target(&process, &mut first, source);
    assert_eq!(callable_target(&process, source), Some(baseline));
    let hcq = publish(&process, &cursor, &[4, 8], Tier::Hcq);
    // Publication queues the new static patch; its still-live baseline is
    // callable through both the old static root and the first vCPU's PIC.
    assert_eq!(callable_target(&process, source), Some(baseline));
    assert_eq!(pic_addresses(&process, &first), [baseline_address]);
    assert!(process.try_service_links().unwrap());
    let retained = process.snapshot(hcq).unwrap();
    let hcq_address = retained.code.allocation.address();
    assert_eq!(callable_target(&process, source), Some(hcq));
    cache_target(&process, &mut second, source);
    assert_eq!(pic_addresses(&process, &second), [hcq_address]);
    assert!(matches!(
        process.retire_unit(baseline),
        Err(Error::PinnedBaseline)
    ));
    process.retire_unit(hcq).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    // Model an already consumed optional-install quota, without thousands of
    // irrelevant source units. Safety unlink must still finish in this stop.
    process.lock().link_install_attempts = INSTALL_LIMIT;
    assert!(!transition.drain_links().unwrap());
    assert_eq!(callable_target(&process, source), None);
    assert!(pic_addresses(&process, &second).is_empty());
    assert_eq!(pic_addresses(&process, &first), [baseline_address]);
    {
        let state = process.lock();
        let payload = state
            .dispatch
            .get(*state.keys.get(&key(4)).unwrap())
            .unwrap()
            .snapshot();
        assert!(payload.hcq().is_none());
        assert_eq!(payload.preferred(), payload.lcq());
        assert_eq!(
            state
                .units
                .records
                .get(baseline.0)
                .unwrap()
                .code
                .baseline_pins
                .load(Ordering::Relaxed),
            0
        );
    }
    assert_eq!(process.reclaim_units().unwrap(), 0); // Snapshot, not a callable root.
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    assert!(process.try_service_links().unwrap());
    assert_eq!(callable_target(&process, source), Some(baseline));
    cache_target(&process, &mut second, source);
    assert_eq!(pic_addresses(&process, &second), [baseline_address]);
    drop(retained);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}

#[test]
fn baseline_only_mapping_invalidation_removes_hcq_and_all_mixed_roots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[4, 8], Tier::Lcq);
    let source = mixed_source(&process, &cursor);
    let mut first = process.register().unwrap();
    let mut second = process.register().unwrap();
    cache_target(&process, &mut first, source);
    let hcq = publish(&process, &cursor, &[4], Tier::Hcq);
    assert!(process.try_service_links().unwrap());
    cache_target(&process, &mut second, source);
    let old_baseline = process.snapshot(baseline).unwrap();
    let old_hcq = process.snapshot(hcq).unwrap();
    // PC 8 belongs only to LCQ, but HCQ promises that whole LCQ as its baseline.
    // The source and optimized body do not directly intersect this mapping.
    let ticket = process
        .invalidate_memory(&[nixe_memory::MemoryInvalidationKind::Mapping {
            address_space: key(8).address_space,
            start: key(8).pc,
            size: 4,
        }])
        .unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    process.lock().link_install_attempts = INSTALL_LIMIT;
    assert!(transition.drain_links().unwrap());
    assert_eq!(callable_target(&process, source), None);
    for reader in [&first, &second] {
        assert!(pic_addresses(&process, reader).is_empty());
    }
    assert_eq!(old_baseline.baseline_pins.load(Ordering::Relaxed), 0);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    assert!(
        process
            .maintenance_complete(crate::lifetime::Reason::MappingChange, ticket)
            .unwrap()
    );
    assert!(matches!(process.snapshot(baseline), Err(Error::StaleUnit)));
    assert!(matches!(process.snapshot(hcq), Err(Error::StaleUnit)));
    assert!(process.snapshot(source).is_ok());
    assert_eq!(process.reclaim_units().unwrap(), 0);
    let replacement = publish(&process, &cursor, &[4, 8], Tier::Lcq);
    assert!(process.try_service_links().unwrap());
    assert_eq!(callable_target(&process, source), Some(replacement));
    let address = process
        .snapshot(replacement)
        .unwrap()
        .code
        .allocation
        .address();
    assert_ne!(address, old_baseline.code.allocation.address());
    for reader in [&mut first, &mut second] {
        cache_target(&process, reader, source);
        assert_eq!(pic_addresses(&process, reader), [address]);
    }
    assert!(!process.try_shutdown().unwrap());
    for reader in [&first, &second] {
        assert!(pic_addresses(&process, reader).is_empty());
    }
    drop(old_baseline);
    drop(old_hcq);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}
