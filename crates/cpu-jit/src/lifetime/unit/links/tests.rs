use super::*;
use crate::executable::output::{Metadata, Output};
use crate::lifetime::unit::tests::{input, key, process, publish};
use cranelift_codegen::{ir, nixe::StateMap};

mod mixed;

#[test]
fn static_refresh_retargets_installed_links_while_the_old_baseline_stays_live() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let first = transition.refresh_static_link(src, 0).unwrap().unwrap();
    let sequence = process.lock().pending[Reason::LinkPatch as usize];
    assert_eq!(transition.refresh_static_link(src, 0).unwrap(), Some(first));
    assert_eq!(process.lock().pending[Reason::LinkPatch as usize], sequence);
    assert!(transition.drain_links().unwrap());
    assert_eq!(transition.refresh_static_link(src, 0).unwrap(), Some(first));
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());

    let hcq = publish(&process, &cursor, &[4], Tier::Hcq);
    let mut transition = stop(&process);
    let second = transition.refresh_static_link(src, 0).unwrap().unwrap();
    assert_ne!(first, second);
    {
        let state = process.lock();
        assert!(state.units.links.records.get(first.0).unwrap().installed);
        let old_target = state.units.records.get(baseline.0).unwrap();
        assert_eq!(old_target.lifecycle, Lifecycle::Published);
        assert_eq!(old_target.incoming, Some(first.0));
        let record = state.units.links.records.get(second.0).unwrap();
        assert_eq!(record.target, hcq);
        assert!(!record.installed);
        let map = &record.source_code.states[record.state_map as usize];
        let allocation = &record.source_code.code.allocation;
        let expected = crate::native::link::emit(
            record.source_code.code.metadata.abi,
            (allocation.address() + map.native_offset as usize) as u64,
            (allocation.address() + map.transfer.as_ref().unwrap().fallback_offset as usize) as u64,
            0,
        )
        .unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (allocation.address() + map.native_offset as usize) as *const u8,
                expected.patch().len(),
            )
        };
        // Publication queued a successor but retained the callable old branch.
        assert_ne!(bytes, expected.patch());
        assert_eq!(
            state.units.records.get(src.0).unwrap().static_sites[0].link,
            Some(second.0)
        );
    }
    assert!(transition.drain_links().unwrap());
    assert!(process.lock().units.links.records.get(first.0).is_none());
    assert!(
        process
            .lock()
            .units
            .records
            .get(baseline.0)
            .unwrap()
            .incoming
            .is_none()
    );
    process.retire_unit(hcq).unwrap();
    // Withdraw before draining retirement: refresh itself must restore the
    // fallback, and cannot recreate the link to the still-preferred dying HCQ.
    assert_eq!(transition.refresh_static_link(src, 0).unwrap(), None);
    assert!(
        process
            .lock()
            .units
            .records
            .get(src.0)
            .unwrap()
            .static_sites[0]
            .link
            .is_none()
    );
    transition.drain_retirements().unwrap();
    let third = transition.refresh_static_link(src, 0).unwrap().unwrap();
    assert_eq!(
        process
            .lock()
            .units
            .links
            .records
            .get(third.0)
            .unwrap()
            .target,
        baseline
    );
    assert!(transition.drain_links().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn static_refresh_retargets_pending_links_without_duplicate_queue_membership() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let first = transition.refresh_static_link(src, 0).unwrap().unwrap();
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    let hcq = publish(&process, &cursor, &[4], Tier::Hcq);
    let mut transition = stop(&process);
    let second = transition.refresh_static_link(src, 0).unwrap().unwrap();
    assert_eq!(pending(&process), vec![second]);
    assert_ne!(first, second);
    assert!(process.lock().units.links.records.get(first.0).is_none());
    assert_eq!(
        process
            .lock()
            .units
            .links
            .records
            .get(second.0)
            .unwrap()
            .target,
        hcq
    );
    assert_eq!(
        transition.refresh_static_link(src, 0).unwrap(),
        Some(second)
    );
    assert_eq!(pending(&process), vec![second]);
    assert!(transition.drain_links().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn static_resolution_waits_for_demand_then_installs_the_registered_target() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    assert!(transition.prepare_static_link(src, 0).unwrap().is_none());
    assert!(matches!(
        transition.prepare_static_link(src, 1),
        Err(Error::InvalidUnit(_))
    ));
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let mut transition = stop(&process);
    let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
    assert_eq!(prepared.target, target);
    assert_eq!(prepared.target_entry, 0);
    let handle = transition.register_link(prepared).unwrap();
    assert!(transition.install_link(handle).unwrap());
    process.retire_unit(target).unwrap();
    assert!(transition.prepare_static_link(src, 0).unwrap().is_none());
    transition.drain_retirements().unwrap();
    assert!(transition.prepare_static_link(src, 0).unwrap().is_none());
    assert!(
        process
            .lock()
            .units
            .records
            .get(src.0)
            .unwrap()
            .outgoing
            .is_none()
    );
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn static_resolution_selects_hcq_entry_then_restores_the_lcq_owner() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[4, 8], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[4, 8], Tier::Hcq);
    let src = source(&process, &cursor, 0, 8);
    let mut transition = stop(&process);
    let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
    assert_eq!(prepared.target, hcq);
    assert_eq!(prepared.target_entry, 1);
    let handle = transition.register_link(prepared).unwrap();
    assert!(transition.install_link(handle).unwrap());
    process.retire_unit(hcq).unwrap();
    // A still-preferred but withdrawing HCQ is a miss until safety unlink;
    // the linker must not silently choose a different payload/version.
    assert!(transition.prepare_static_link(src, 0).unwrap().is_none());
    transition.drain_retirements().unwrap();
    let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
    assert_eq!(prepared.target, baseline);
    assert_eq!(prepared.target_entry, 1);
    let handle = transition.register_link(prepared).unwrap();
    assert!(transition.install_link(handle).unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn static_resolution_tracks_replacement_and_reused_dispatch_slots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let first = publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let old = transition.prepare_static_link(src, 0).unwrap().unwrap();
    assert_eq!(old.target, first);
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    let second = publish(&process, &cursor, &[4], Tier::Lcq);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(matches!(
        transition.register_link(old),
        Err(Error::StalePublication)
    ));
    transition.drain_retirements().unwrap();
    let current = transition.prepare_static_link(src, 0).unwrap().unwrap();
    assert_eq!(current.target, second);
    drop(current);
    process.retire_unit(second).unwrap();
    transition.drain_retirements().unwrap();
    process.reclaim_units().unwrap();
    assert!(transition.prepare_static_link(src, 0).unwrap().is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    let third = publish(&process, &cursor, &[4], Tier::Lcq);
    let mut transition = stop(&process);
    let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
    assert_eq!(prepared.target, third);
    assert_ne!(prepared.target, second);
    let handle = transition.register_link(prepared).unwrap();
    assert!(transition.install_link(handle).unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn static_source_index_uses_full_target_keys_and_survives_target_replacement() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let first = source(&process, &cursor, 0, 4);
    let mut specialized = source_input(&process, 12, 4);
    let mut specialized_key = key(4);
    specialized_key.fp = crate::abi::FpSpecialization::Exact(0);
    specialized.states[0]
        .transfer
        .as_mut()
        .unwrap()
        .static_target = Some(specialized_key);
    let mut source_key = key(12);
    source_key.fp = specialized_key.fp;
    specialized.instructions[0].key = InstructionKey::new(source_key).unwrap();
    specialized.entries[0].key = source_key;
    let specialized = process
        .prepare_unit(
            &[process.reserve(source_key).unwrap()],
            specialized,
            &cursor,
        )
        .unwrap()
        .publish()
        .unwrap();
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    assert!(process.try_service_links().unwrap());
    assert!(
        process
            .lock()
            .units
            .records
            .get(specialized.0)
            .unwrap()
            .static_sites[0]
            .link
            .is_none()
    );
    let second = source(&process, &cursor, 8, 4);
    // Source-first, target-first and unrelated specialization all share the
    // normal publication path; discovery cannot match only the PC.
    let sources = |target| {
        process
            .lock()
            .units
            .static_sources(target)
            .collect::<Vec<_>>()
    };
    let sites = sources(key(4));
    assert_eq!(sites.len(), 2);
    assert!(
        sites
            .iter()
            .any(|site| site.source == first && site.state_map == 0 && site.island == 0)
    );
    assert!(sites.iter().any(|site| site.source == second));
    assert_eq!(sources(specialized_key)[0].source, specialized);
    let mut other_space = key(4);
    other_space.address_space = nixe_memory::AddressSpaceId::new(2);
    assert!(sources(other_space).is_empty());

    let replacement = publish(&process, &cursor, &[4], Tier::Lcq);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(sources(key(4)), sites);
    assert!(matches!(
        process
            .lock()
            .units
            .records
            .get(target.0)
            .unwrap()
            .lifecycle,
        Lifecycle::Retired(_)
    ));
    process.retire_unit(replacement).unwrap();
    transition.drain_retirements().unwrap();
    // An absent target leaves the source discoverable for future demand.
    assert_eq!(sources(key(4)), sites);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
    assert!(process.lock().units.static_sites.entries.is_empty());
    assert!(process.lock().units.static_site_storage.is_none());
}

#[test]
fn static_source_index_drops_unlinked_sources_before_compiler_pins_and_reuses_capacity() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut sources = Vec::new();
    for i in 0..48 {
        sources.push(source(&process, &cursor, i * 4, 0x1000));
    }
    let capacity = process.lock().units.static_sites.entries.capacity();
    assert_eq!(process.lock().units.static_sources(key(0x1000)).count(), 48);
    assert!(process.lock().units.static_site_storage.is_some());
    let snapshot = process.snapshot(sources[0]).unwrap();
    let old = sources[0];
    let mut transition = stop(&process);
    // Exercise interior, tail and head removal from the same target list.
    for (removed, index) in [24, 0, 47].into_iter().enumerate() {
        process.retire_unit(sources[index]).unwrap();
        transition.drain_retirements().unwrap();
        assert_eq!(
            process.lock().units.static_sources(key(0x1000)).count(),
            47 - removed
        );
    }
    for (index, handle) in sources.into_iter().enumerate() {
        if [24, 0, 47].contains(&index) {
            continue;
        }
        process.retire_unit(handle).unwrap();
    }
    // Closing sources cannot acquire new work, even before their index
    // membership is physically removed by the safety drain.
    assert_eq!(process.lock().units.static_sources(key(0x1000)).count(), 0);
    transition.drain_retirements().unwrap();
    assert!(process.lock().units.static_sites.entries.is_empty());
    process.reclaim_units().unwrap();
    assert!(process.lock().units.records.get(old.0).is_some());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(snapshot);
    process.reclaim_units().unwrap();
    for i in 0..48 {
        let handle = source(&process, &cursor, i * 4, 0x1000);
        assert_ne!(handle, old);
    }
    assert_eq!(
        process.lock().units.static_sites.entries.capacity(),
        capacity
    );
    assert_eq!(process.lock().units.static_sources(key(0x1000)).count(), 48);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn stale_source_publication_leaves_no_static_association() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            source_input(&process, 0, 4),
            &cursor,
        )
        .unwrap();
    cursor.store(1, Ordering::Release);
    assert_eq!(prepared.publish(), Err(Error::StalePublication));
    assert!(process.lock().units.static_sites.entries.is_empty());
    assert!(process.lock().units.records.is_empty());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn static_source_index_keeps_each_exit_and_island_across_index_growth() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut input = source_input(&process, 0, 4);
    let mut states = Vec::new();
    for (index, offset) in [8, 24, 32].into_iter().enumerate() {
        let mut state = input.states[0].state.clone();
        state.site.state_map = index as u32;
        let mut transfer = input.states[0].transfer.clone();
        if index == 0 {
            transfer.as_mut().unwrap().static_target = None;
        }
        states.push(StateRecord {
            native_offset: offset,
            state,
            exit: Some(GuestExit {
                pc: key(0).pc,
                kind: if index == 0 {
                    EdgeKind::Breakpoint(0)
                } else {
                    EdgeKind::Static
                },
            }),
            transfer,
        });
    }
    input.states = states.into_boxed_slice();
    input.code = {
        let mut old = input.code;
        let mut bytes = unsafe {
            std::slice::from_raw_parts(old.allocation.address() as *const u8, old.allocation.len())
        }
        .to_vec();
        bytes.resize(40, 0);
        let mut backend_states = old.metadata.states.to_vec();
        for (index, offset) in [24, 32].into_iter().enumerate() {
            let mut backend = backend_states[0].clone();
            backend.id = (index + 1) as u64;
            backend.offset = offset;
            backend_states.push(backend);
            let branch =
                crate::native::link::emit(old.metadata.abi, u64::from(offset), 16, 0).unwrap();
            bytes[offset as usize..][..branch.patch().len()].copy_from_slice(branch.patch());
        }
        old.metadata.states = backend_states.into_boxed_slice();
        process
            .cache
            .install_with_islands(
                Output {
                    bytes: bytes.into_boxed_slice(),
                    alignment: 16,
                    metadata: old.metadata,
                },
                Tier::Lcq,
                2,
                |_| None,
            )
            .unwrap()
    };
    let src = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], input, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    for i in 0..48 {
        source(&process, &cursor, 0x100 + i * 4, 0x2000 + i * 4);
    }
    let sites = process
        .lock()
        .units
        .static_sources(key(4))
        .collect::<Vec<_>>();
    assert_eq!(sites.len(), 2);
    assert!(sites.iter().all(|site| site.source == src));
    assert!(
        sites
            .iter()
            .any(|site| site.state_map == 1 && site.island == 0)
    );
    assert!(
        sites
            .iter()
            .any(|site| site.state_map == 2 && site.island == 1)
    );
    assert_eq!(process.lock().units.static_sites.entries.len(), 49);
    process.retire_unit(src).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(process.lock().units.static_sources(key(4)).count(), 0);
    assert_eq!(process.lock().units.static_sites.entries.len(), 48);
    for i in 0..48 {
        assert_eq!(
            process
                .lock()
                .units
                .static_sources(key(0x2000 + i * 4))
                .count(),
            1
        );
    }
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn publication_rejects_linkable_observations_and_missing_static_islands() {
    for case in 0..4 {
        let process = process();
        let cursor = AtomicU64::new(0);
        let mut input = source_input(&process, 0, 4);
        let expected = if case == 3 {
            let old = input.code;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    old.allocation.address() as *const u8,
                    old.allocation.len(),
                )
            }
            .to_vec()
            .into_boxed_slice();
            // Keep all terminal metadata/bytes valid but omit its reservation.
            input.code = process
                .cache
                .install(
                    Output {
                        bytes,
                        alignment: 16,
                        metadata: old.metadata,
                    },
                    Tier::Lcq,
                    |_| None,
                )
                .unwrap();
            "static exits exceed the source's reserved island capacity"
        } else {
            input.states[0].exit.as_mut().unwrap().kind = match case {
                0 => EdgeKind::Breakpoint(0),
                1 => EdgeKind::SupervisorCall(0),
                _ => EdgeKind::Indirect,
            };
            "invalid terminal transfer contract"
        };
        assert!(
            matches!(process.prepare_unit(&[process.reserve(key(0)).unwrap()], input, &cursor),
            Err(Error::InvalidUnit(detail)) if detail == expected)
        );
        assert!(process.lock().units.records.is_empty());
    }
}

pub(super) fn source(process: &Lifetime, cursor: &AtomicU64, pc: u64, target: u64) -> UnitHandle {
    let handle = process
        .prepare_unit(
            &[process.reserve(key(pc)).unwrap()],
            source_input(process, pc, target),
            cursor,
        )
        .unwrap()
        .publish()
        .unwrap();
    // These protocol fixtures inspect/register/install the initial fallback
    // explicitly. Defer automatically registered work using the real batching
    // contract; production execution normally services it before entry.
    if process.lock().phase == crate::lifetime::Phase::Closing {
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        transition
            .batch()
            .unwrap()
            .complete_with_links_deferred()
            .unwrap();
        assert!(transition.try_reopen().unwrap());
    }
    handle
}

pub(super) fn source_input(process: &Lifetime, pc: u64, target: u64) -> Input {
    let mut input = input(process, &[pc], Tier::Lcq);
    let abi = input.code.metadata.abi;
    let width = if abi == HostAbi::X86_64 { 8 } else { 4 };
    // Real aligned patch at +8, initially reaching a source-local return-42
    // fallback at +16. Installation tests redirect it only under Closed.
    let bytes = match abi {
        HostAbi::X86_64 => {
            let mut bytes = vec![0x90; 24];
            bytes[..4].copy_from_slice(&[0xf3, 0x0f, 0x1e, 0xfa]);
            bytes[8..16].copy_from_slice(crate::native::link::emit(abi, 8, 16, 0).unwrap().patch());
            bytes[16..22].copy_from_slice(&[0xb8, 42, 0, 0, 0, 0xc3]);
            bytes
        }
        HostAbi::Aarch64 => [
            0xd503245f_u32,
            0xd503201f,
            0x14000002,
            0xd503201f,
            0x52800540,
            0xd65f03c0,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect(),
    };
    input.code = process
        .cache
        .install_with_islands(
            Output {
                bytes: bytes.into_boxed_slice(),
                alignment: 16,
                metadata: Metadata {
                    abi,
                    frame_extent: crate::abi::TRANSFER_BYTES,
                    entries: Box::new([(ir::Block::from_u32(0), 0)]),
                    states: Box::new([StateMap {
                        id: 0,
                        offset: 8,
                        entry: false,
                        patch_bytes: width,
                        fault_bytes: 0,
                        poll: None,
                        values: Vec::new(),
                    }]),
                    faults: Box::new([]),
                    traps: Box::new([]),
                    relocations: Box::new([]),
                },
            },
            Tier::Lcq,
            1,
            |_| None,
        )
        .unwrap();
    input.faults = Box::new([]);
    input.states[0].native_offset = 8;
    input.states[0].exit = Some(GuestExit {
        pc: key(pc).pc,
        kind: EdgeKind::Static,
    });
    input.states[0].transfer = Some(Box::new(TerminalTransfer {
        destination: ValueLocation::Constant(u128::from(target)),
        static_target: Some(key(target)),
        completed: 1,
        patch_bytes: width,
        fallback_offset: 16,
        poll_offset: None,
    }));
    input
}

pub(super) fn stop(process: &Lifetime) -> Transition<'_> {
    process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition
}

pub(super) fn pending(process: &Lifetime) -> Vec<LinkHandle> {
    let state = process.lock();
    let mut next = state.units.links.head;
    let mut result = Vec::new();
    while let Some(handle) = next {
        result.push(LinkHandle(handle, process.identity));
        next = state
            .units
            .links
            .records
            .get(handle)
            .unwrap()
            .pending
            .unwrap()
            .next;
    }
    result
}

#[test]
fn duplicate_registration_preserves_one_record_request_and_charge() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let source = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(source, 0, target, 0, 0).unwrap();
    assert_eq!(
        prepared.source_state().site.source,
        prepared.source_code.version
    );
    assert_eq!(
        prepared.target_contract().abi,
        prepared.target_code.code.metadata.abi
    );
    let handle = transition.register_link(prepared).unwrap();
    assert_eq!(
        process
            .lock()
            .units
            .records
            .get(source.0)
            .unwrap()
            .static_sites[0]
            .link,
        Some(handle.0)
    );
    let usage = process.cache.usage().unwrap();
    let sequence = process.lock().pending[Reason::LinkPatch as usize];
    for _ in 0..8 {
        let prepared = transition.prepare_link(source, 0, target, 0, 0).unwrap();
        assert_eq!(transition.register_link(prepared).unwrap(), handle);
    }
    assert_eq!(process.cache.usage().unwrap(), usage);
    assert_eq!(process.lock().pending[Reason::LinkPatch as usize], sequence);
    assert_eq!(pending(&process), vec![handle]);
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert!(!transition.try_reopen().unwrap());
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_ne!(process.pending.load(Ordering::Acquire), 0);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.discard_pending_link(handle).unwrap();
    assert!(
        process
            .lock()
            .units
            .records
            .get(source.0)
            .unwrap()
            .static_sites[0]
            .link
            .is_none()
    );
    assert!(pending(&process).is_empty());
    // Real slot capacity is reused, but the old generation cannot name it.
    let prepared = transition.prepare_link(source, 0, target, 0, 0).unwrap();
    let reused = transition.register_link(prepared).unwrap();
    assert_eq!(
        process
            .lock()
            .units
            .records
            .get(source.0)
            .unwrap()
            .static_sites[0]
            .link,
        Some(reused.0)
    );
    assert_ne!(handle, reused);
    assert_eq!(process.cache.usage().unwrap(), usage);
    assert_eq!(
        transition.discard_pending_link(handle),
        Err(Error::StaleUnit)
    );
    transition.discard_pending_link(reused).unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn target_and_source_retirement_detach_only_their_pending_adjacency() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[100], Tier::Lcq);
    let unrelated_target = publish(&process, &cursor, &[200], Tier::Lcq);
    let sources: Vec<_> = (0..20)
        .map(|i| source(&process, &cursor, i * 4, 100))
        .collect();
    let unrelated = source(&process, &cursor, 300, 200);
    let mut transition = stop(&process);
    let mut handles = Vec::new();
    for src in &sources {
        let prepared = transition.prepare_link(*src, 0, target, 0, 0).unwrap();
        handles.push(transition.register_link(prepared).unwrap());
    }
    let prepared = transition
        .prepare_link(unrelated, 0, unrelated_target, 0, 0)
        .unwrap();
    let keep = transition.register_link(prepared).unwrap();
    // Remove middle, head and tail of an incoming list / FIFO independently.
    for index in [10, 0, 19] {
        transition.discard_pending_link(handles[index]).unwrap();
    }
    process.retire_unit(sources[5]).unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(
        transition.discard_pending_link(handles[5]),
        Err(Error::StaleUnit)
    );
    process.retire_unit(target).unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(pending(&process), vec![keep]);
    {
        let state = process.lock();
        for src in sources {
            assert!(state.units.records.get(src.0).unwrap().outgoing.is_none());
        }
        assert!(
            state
                .units
                .records
                .get(target.0)
                .unwrap()
                .incoming
                .is_none()
        );
    }
    transition.discard_pending_link(keep).unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn self_link_detaches_both_lists_and_shutdown_releases_registry_storage() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let src = source(&process, &cursor, 0, 0);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(src, 0, src, 0, 0).unwrap();
    let weak = Arc::downgrade(&prepared.source_code);
    transition.register_link(prepared).unwrap();
    process.request(Reason::Shutdown).unwrap();
    assert!(transition.try_finish_shutdown().unwrap());
    assert!(weak.upgrade().is_none());
    assert!(pending(&process).is_empty());
    assert_eq!(process.lock().units.links.records.capacity(), 0);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap()); // Shutdown remains Closed.
}

#[test]
fn prepared_work_retains_owners_but_rejects_retirement_and_changed_admission() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let source = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let old = transition.prepare_link(source, 0, target, 0, 0).unwrap();
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    let mut transition = stop(&process);
    assert_eq!(transition.register_link(old), Err(Error::StalePublication));
    let old = transition.prepare_link(source, 0, target, 0, 0).unwrap();
    let weak = Arc::downgrade(&old.target_code);
    process.retire_unit(target).unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(weak.upgrade().is_some());
    assert_eq!(transition.register_link(old), Err(Error::StaleUnit));
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(weak.upgrade().is_none());
    assert!(pending(&process).is_empty());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn rootless_collector_routes_linked_units_through_coordinated_retirement() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let src = source(&process, &cursor, 0, 0);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(src, 0, src, 0, 0).unwrap();
    let handle = transition.register_link(prepared).unwrap();
    {
        // Model a cancelled cutover which loses its remaining dispatch root
        // later. The collector previously retired this state directly.
        let mut state = process.lock();
        let record = state.units.records.get_mut(src.0).unwrap();
        record.lifecycle = Lifecycle::Superseded;
        let slot = record.slots[0];
        let reachability = state.dispatch.get(slot).unwrap().reachability();
        state
            .dispatch
            .get_mut(slot)
            .unwrap()
            .rewrite_closed(DispatchPayload::new(reachability, None, None));
    }
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(pending(&process), vec![handle]);
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert!(pending(&process).is_empty());
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn replacement_and_mapping_invalidation_cancel_old_preparations_not_other_units() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old_target = publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    assert!(matches!(
        transition.prepare_link(src, 1, old_target, 0, 0),
        Err(Error::InvalidUnit(_))
    ));
    assert!(matches!(
        transition.prepare_link(src, 0, old_target, 1, 0),
        Err(Error::InvalidUnit(_))
    ));
    assert!(matches!(
        transition.prepare_link(src, 0, old_target, 0, 1),
        Err(Error::InvalidUnit(_))
    ));
    assert!(matches!(
        transition.prepare_link(src, 0, src, 0, 0),
        Err(Error::InvalidUnit(_))
    ));
    let prepared = transition.prepare_link(src, 0, old_target, 0, 0).unwrap();
    let old = transition.register_link(prepared).unwrap();
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    let new_target = publish(&process, &cursor, &[4], Tier::Lcq);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(transition.discard_pending_link(old), Err(Error::StaleUnit));
    // Publication already replaced the uninstalled request. Retiring the old
    // target must leave the successor's roots and queue membership intact.
    let successor = pending(&process);
    assert_eq!(successor.len(), 1);
    let prepared = transition.prepare_link(src, 0, new_target, 0, 0).unwrap();
    assert_eq!(transition.register_link(prepared).unwrap(), successor[0]);
    process
        .invalidate_memory(&[nixe_memory::MemoryInvalidationKind::Mapping {
            address_space: key(4).address_space,
            start: key(4).pc,
            size: 4,
        }])
        .unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert!(pending(&process).is_empty());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());

    // All preparatory graph operations leave the source's actual native
    // fallback unchanged, including replacement of its proposed target.
    let mut reader = process.register().unwrap();
    let mut cpu = nixe_cpu::state::a64::A64State::default();
    let mut frame = crate::lifetime::unit::tests::frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    let call: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(address) };
    assert_eq!(unsafe { call() }, 42);
}
