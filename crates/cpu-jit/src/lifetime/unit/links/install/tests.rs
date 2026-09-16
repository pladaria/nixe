use super::*;
use crate::executable::output::{Metadata, Output};
#[cfg(target_arch = "aarch64")]
use crate::lifetime::unit::links::tests::source_input;
use crate::lifetime::unit::links::tests::{pending, source, stop};
use crate::lifetime::unit::tests::{frame, key, process, publish};
use nixe_cpu::state::a64::A64State;

mod batching;
mod maintenance;

fn target_binding(input: &mut Input) {
    use crate::abi::{GuestValue, RegisterClass, ValueBinding};
    input.entries[0].contract.live_in.integer.x[0] = true;
    input.entries[0].contract.bindings = Box::new([ValueBinding {
        value: GuestValue::General(0),
        location: ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 0,
        },
    }]);
}

fn nonempty_target(process: &Lifetime, cursor: &AtomicU64) -> UnitHandle {
    let mut candidate = crate::lifetime::unit::tests::input(process, &[4], Tier::Lcq);
    target_binding(&mut candidate);
    process
        .prepare_unit(&[process.reserve(key(4)).unwrap()], candidate, cursor)
        .unwrap()
        .publish()
        .unwrap()
}

fn value(transition: &mut Transition<'_>, target: UnitHandle, value: u8) {
    let bytes = if cfg!(target_arch = "x86_64") {
        vec![value]
    } else {
        (0x52800000 | (u32::from(value) << 5))
            .to_le_bytes()
            .to_vec()
    };
    let offset = if cfg!(target_arch = "x86_64") { 5 } else { 4 };
    // Modify only the synthetic target's MOV immediate, not its state/fault PCs.
    unsafe {
        transition
            .patch_unit(
                target,
                &[Write::Code {
                    offset,
                    bytes: &bytes,
                }],
            )
            .unwrap();
    }
}

fn execute(process: &Arc<Lifetime>) -> u32 {
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    let call: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(address) };
    // Both synthetic units are System-ABI leaves. The real reader epoch spans
    // A's entry, its native jump to B, and the final return, with no Rust hop.
    unsafe { call() }
}

#[test]
fn publication_retargets_callable_links_and_preserves_active_reader_epochs() {
    for tier in [Tier::Lcq, Tier::Hcq] {
        let process = process();
        let cursor = AtomicU64::new(0);
        let target = publish(&process, &cursor, &[4], Tier::Lcq);
        let src = source(&process, &cursor, 0, 4);
        let mut transition = stop(&process);
        value(&mut transition, target, 77);
        assert!(transition.drain_links().unwrap());
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        let old = process
            .lock()
            .units
            .records
            .get(src.0)
            .unwrap()
            .static_sites[0]
            .callable
            .unwrap();

        let mut reader = process.register().unwrap();
        let mut cpu = A64State::default();
        let mut frame = frame(&mut cpu);
        let invocation = unsafe { reader.admit(&mut frame, key(0)) }
            .unwrap()
            .unwrap();
        let call: unsafe extern "C" fn() -> u32 = unsafe {
            std::mem::transmute(invocation.payload().preferred().unwrap().canonical.get())
        };
        assert_eq!(unsafe { call() }, 77);
        let replacement = publish(&process, &cursor, &[4], tier);
        let pending = pending(&process);
        assert_eq!(pending.len(), 1);
        {
            let state = process.lock();
            let site = &state.units.records.get(src.0).unwrap().static_sites[0];
            assert_eq!(site.callable, Some(old));
            assert_eq!(site.link, Some(pending[0].0));
            assert_eq!(
                state.units.links.records.get(pending[0].0).unwrap().target,
                replacement
            );
            assert_eq!(
                state.units.records.get(target.0).unwrap().incoming,
                Some(old)
            );
        }
        // Closing admission does not modify the old branch under this reader.
        assert!(!process.try_service_links().unwrap());
        assert_eq!(unsafe { call() }, 77);
        drop(invocation);
        assert!(process.try_service_links().unwrap());
        {
            let state = process.lock();
            assert!(state.units.links.records.get(old).is_none());
            let site = &state.units.records.get(src.0).unwrap().static_sites[0];
            assert_eq!(site.callable, site.link);
            assert_eq!(site.link, Some(pending[0].0));
            assert!(
                state
                    .units
                    .links
                    .records
                    .get(pending[0].0)
                    .unwrap()
                    .installed
            );
            if tier == Tier::Hcq {
                let baseline = state.units.records.get(target.0).unwrap();
                assert_eq!(baseline.lifecycle, Lifecycle::Published);
                assert!(baseline.incoming.is_none());
            }
        }
        assert_eq!(execute(&process), 42);
        assert!(process.try_shutdown().unwrap());
    }
}

#[test]
fn hcq_withdrawal_relinks_live_sources_to_each_retained_baseline_entry() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[4, 8], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[4, 8], Tier::Hcq);
    let first = source(&process, &cursor, 0, 4);
    let second = source(&process, &cursor, 12, 8);
    let dying = source(&process, &cursor, 16, 4);
    let unrelated = source(&process, &cursor, 20, 100);
    let mut transition = stop(&process);
    value(&mut transition, baseline, 77);
    assert!(transition.drain_links().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    assert_eq!(execute(&process), 42); // HCQ, not the retained LCQ leaf.
    let ticket = process.retire_unit(hcq).unwrap();
    process.retire_unit(dying).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 2);
    assert_eq!(pending(&process).len(), 2);
    {
        let state = process.lock();
        for (source, entry) in [(first, 0), (second, 1)] {
            let site = &state.units.records.get(source.0).unwrap().static_sites[0];
            assert!(site.callable.is_none());
            let link = state.units.links.records.get(site.link.unwrap()).unwrap();
            assert_eq!(link.target, baseline);
            assert_eq!(link.target_entry, entry);
            assert!(!link.installed);
        }
        assert!(
            state.units.records.get(unrelated.0).unwrap().static_sites[0]
                .link
                .is_none()
        );
    }
    // Retirement alone must not acknowledge the newly registered patch work.
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert!(transition.drain_links().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(ticket.is_complete().unwrap());
    drop(transition);
    assert_eq!(execute(&process), 77);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn cancelling_a_replacement_preserves_the_callable_link_until_its_own_retirement() {
    for retire_source in [false, true] {
        let process = process();
        let cursor = AtomicU64::new(0);
        let target = publish(&process, &cursor, &[4], Tier::Lcq);
        let src = source(&process, &cursor, 0, 4);
        let mut transition = stop(&process);
        value(&mut transition, target, 77);
        assert!(transition.drain_links().unwrap());
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        let old = process
            .lock()
            .units
            .records
            .get(src.0)
            .unwrap()
            .static_sites[0]
            .callable
            .unwrap();
        let replacement = publish(&process, &cursor, &[4], Tier::Hcq);
        let next = pending(&process)[0];
        process
            .retire_unit(if retire_source { src } else { replacement })
            .unwrap();
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        transition.drain_retirements().unwrap();
        assert!(pending(&process).is_empty());
        assert!(process.lock().units.links.records.get(next.0).is_none());
        if retire_source {
            assert!(process.lock().units.links.records.get(old).is_none());
        } else {
            let state = process.lock();
            let site = &state.units.records.get(src.0).unwrap().static_sites[0];
            assert_eq!(site.link, Some(old));
            assert_eq!(site.callable, Some(old));
            assert!(state.units.links.records.get(old).unwrap().installed);
        }
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        if !retire_source {
            assert_eq!(execute(&process), 77);
            // Reuse the source after cancellation: its callable identity must
            // still be discoverable for a later preferred version.
            publish(&process, &cursor, &[4], Tier::Hcq);
            assert!(process.try_service_links().unwrap());
            assert_eq!(execute(&process), 42);
        }
        assert!(process.try_shutdown().unwrap());
        assert!(process.lock().units.links.records.is_empty());
    }
}

#[test]
fn installed_branch_executes_and_explicit_unlink_restores_the_owned_fallback() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    assert_eq!(execute(&process), 42);
    let mut transition = stop(&process);
    value(&mut transition, target, 77);
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    let handle = transition.register_link(prepared).unwrap();
    assert!(transition.install_link(handle).unwrap());
    assert!(!transition.install_link(handle).unwrap());
    assert!(pending(&process).is_empty());
    {
        let state = process.lock();
        assert!(state.units.links.records.get(handle.0).unwrap().installed);
        assert_eq!(
            state.units.records.get(src.0).unwrap().outgoing,
            Some(handle.0)
        );
        assert_eq!(
            state.units.records.get(target.0).unwrap().incoming,
            Some(handle.0)
        );
    }
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    assert_eq!(transition.register_link(prepared).unwrap(), handle);
    assert!(pending(&process).is_empty());
    assert!(matches!(
        transition.discard_pending_link(handle),
        Err(Error::InvalidUnit(_))
    ));
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 77);
    assert_eq!(transition.unlink_link(handle), Err(Error::Closed));
    let mut transition = stop(&process);
    transition.unlink_link(handle).unwrap();
    assert_eq!(transition.unlink_link(handle), Err(Error::StaleUnit));
    {
        let state = process.lock();
        assert!(state.units.records.get(src.0).unwrap().outgoing.is_none());
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
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 42);
}

#[test]
fn target_retirement_unlinks_before_reclaim_and_keeps_compiler_snapshots_alive() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let snapshot = process.snapshot(target).unwrap();
    let address = snapshot.code.allocation.address();
    let mut transition = stop(&process);
    value(&mut transition, target, 91);
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    let weak = Arc::downgrade(&prepared.target_code);
    let handle = transition.register_link(prepared).unwrap();
    transition.install_link(handle).unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 91);
    process.retire_unit(target).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(transition.unlink_link(handle), Err(Error::StaleUnit));
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(snapshot.code.allocation.address(), address);
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(weak.upgrade().is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 42);

    // Reuse the actual target address and registry slot for unrelated code.
    // A stale native jump would now return 123 instead of the source fallback.
    let reused = publish(&process, &cursor, &[8], Tier::Lcq);
    assert_eq!(
        process.snapshot(reused).unwrap().code.allocation.address(),
        address
    );
    let mut transition = stop(&process);
    value(&mut transition, reused, 123);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 42);
}

#[test]
fn replacement_invalidation_and_source_retirement_remove_installed_edges() {
    for mode in 0..3 {
        let process = process();
        let cursor = AtomicU64::new(0);
        let target = publish(&process, &cursor, &[4], Tier::Lcq);
        let src = source(&process, &cursor, 0, 4);
        let mut transition = stop(&process);
        value(&mut transition, target, 63);
        let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
        let handle = transition.register_link(prepared).unwrap();
        transition.install_link(handle).unwrap();
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        assert_eq!(execute(&process), 63);
        match mode {
            0 => {
                publish(&process, &cursor, &[4], Tier::Lcq);
            }
            1 => {
                process
                    .invalidate_memory(&[nixe_memory::MemoryInvalidationKind::Mapping {
                        address_space: key(4).address_space,
                        start: key(4).pc,
                        size: 4,
                    }])
                    .unwrap();
            }
            _ => {
                process.retire_unit(src).unwrap();
            }
        }
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        assert_eq!(transition.drain_retirements().unwrap(), 1);
        assert_eq!(transition.unlink_link(handle), Err(Error::StaleUnit));
        if mode == 0 {
            assert_eq!(pending(&process).len(), 1);
            transition
                .batch()
                .unwrap()
                .complete_with_links_deferred()
                .unwrap();
        } else {
            assert!(process.lock().units.links.records.is_empty());
            transition.batch().unwrap().complete().unwrap();
        }
        assert!(transition.try_reopen().unwrap());
        if mode != 2 {
            assert_eq!(execute(&process), 42);
        }
    }
}

#[test]
fn stale_pending_destination_is_discarded_without_patching() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = nonempty_target(&process, &cursor);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    let handle = transition.register_link(prepared).unwrap();
    process.retire_unit(target).unwrap();
    let before = process.cache.usage().unwrap();
    assert!(!transition.install_link(handle).unwrap());
    // Stale nonempty work is canceled before allocating executable storage.
    assert_eq!(process.cache.usage().unwrap(), before);
    assert!(pending(&process).is_empty());
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 42);
}

#[test]
fn retirement_and_shutdown_release_nonempty_bridges_without_retaining_dead_roots() {
    for mode in 0..6 {
        let process = process();
        let cursor = AtomicU64::new(0);
        let target = nonempty_target(&process, &cursor);
        let src = source(&process, &cursor, 0, 4);
        let mut transition = stop(&process);
        let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
        let handle = transition.register_link(prepared).unwrap();
        transition.install_link(handle).unwrap();
        let (bridge_address, bridge_bytes) = {
            let state = process.lock();
            let bridge = state
                .units
                .links
                .records
                .get(handle.0)
                .unwrap()
                .bridge
                .as_ref()
                .unwrap();
            (bridge.allocation.address(), bridge.allocation.len())
        };
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        match mode {
            0 => {
                process.retire_unit(target).unwrap();
            }
            1 => {
                process.retire_unit(src).unwrap();
            }
            2 => {
                publish(&process, &cursor, &[4], Tier::Lcq);
            }
            3 => {
                process
                    .invalidate_memory(&[nixe_memory::MemoryInvalidationKind::Mapping {
                        address_space: key(4).address_space,
                        start: key(4).pc,
                        size: 4,
                    }])
                    .unwrap();
            }
            4 => {
                process.request(Reason::Shutdown).unwrap();
            }
            _ => {
                publish(&process, &cursor, &[4], Tier::Hcq);
            }
        }
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        if mode == 4 {
            assert!(transition.try_finish_shutdown().unwrap());
            assert_eq!(process.cache.usage().unwrap().committed, 0);
        } else {
            if mode == 5 {
                assert!(transition.drain_links().unwrap());
                let state = process.lock();
                let baseline = state.units.records.get(target.0).unwrap();
                assert_eq!(baseline.lifecycle, Lifecycle::Published);
                assert!(baseline.incoming.is_none());
                assert!(state.units.links.records.get(handle.0).is_none());
            } else {
                transition.drain_retirements().unwrap();
            }
            // Retirement still retains unit snapshots/directory owners, but
            // the restored bridge has no external users and its span is free.
            let reuse = process
                .cache
                .install_with_islands(
                    Output {
                        bytes: vec![0; bridge_bytes].into_boxed_slice(),
                        alignment: 16,
                        metadata: Metadata {
                            abi: if cfg!(target_arch = "x86_64") {
                                HostAbi::X86_64
                            } else {
                                HostAbi::Aarch64
                            },
                            frame_extent: crate::abi::TRANSFER_BYTES,
                            entries: Box::new([]),
                            states: Box::new([]),
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
            assert_eq!(reuse.allocation.address(), bridge_address);
        }
        if mode == 2 {
            assert_eq!(pending(&process).len(), 1);
            transition
                .batch()
                .unwrap()
                .complete_with_links_deferred()
                .unwrap();
        } else if mode == 5 {
            {
                let state = process.lock();
                let site = &state.units.records.get(src.0).unwrap().static_sites[0];
                assert!(site.callable.is_some());
                assert_eq!(site.callable, site.link);
            }
            transition.batch().unwrap().complete().unwrap();
        } else {
            assert!(process.lock().units.links.records.is_empty());
            transition.batch().unwrap().complete().unwrap();
        }
        assert!(transition.try_reopen().unwrap());
        if mode != 1 && mode != 4 {
            assert_eq!(execute(&process), 42);
        }
    }
}

#[test]
fn shutdown_unlinks_installed_self_edge_before_releasing_its_storage() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let src = source(&process, &cursor, 0, 0);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(src, 0, src, 0, 0).unwrap();
    let weak = Arc::downgrade(&prepared.source_code);
    let handle = transition.register_link(prepared).unwrap();
    transition.install_link(handle).unwrap();
    // Never execute this unbounded synthetic self-loop; shutdown must restore
    // its fallback despite both adjacency references naming the same unit.
    process.request(Reason::Shutdown).unwrap();
    assert!(transition.try_finish_shutdown().unwrap());
    assert!(weak.upgrade().is_none());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn nonempty_transfer_storage_is_owned_until_unlink_and_then_reused() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = nonempty_target(&process, &cursor);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    let handle = transition.register_link(prepared).unwrap();
    let before = process.cache.usage().unwrap();
    assert!(transition.install_link(handle).unwrap());
    let (address, island, bytes, abi) = {
        let state = process.lock();
        let record = state.units.links.records.get(handle.0).unwrap();
        assert!(record.installed);
        let bridge = record.bridge.as_ref().unwrap();
        assert!(bridge.metadata.faults.is_empty());
        assert!(bridge.metadata.states.is_empty());
        (
            bridge.allocation.address(),
            bridge.allocation.island_address(0).unwrap(),
            bridge.allocation.len(),
            bridge.metadata.abi,
        )
    };
    assert!(pending(&process).is_empty());
    assert_eq!(
        process.cache.usage().unwrap().metadata,
        before.metadata + size_of::<crate::executable::Installed>()
    );
    assert!(!transition.install_link(handle).unwrap());
    // The synthetic source/target are System ABI leaves, so don't execute the
    // canonical-load bridge without a NativeFrame. LCQ tests cover real entry.
    transition.unlink_link(handle).unwrap();
    assert_eq!(process.cache.usage().unwrap(), before);
    let reuse = process
        .cache
        .install_with_islands(
            Output {
                bytes: vec![0; bytes].into_boxed_slice(),
                alignment: 16,
                metadata: Metadata {
                    abi,
                    frame_extent: crate::abi::TRANSFER_BYTES,
                    entries: Box::new([]),
                    states: Box::new([]),
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
    assert_eq!(reuse.allocation.address(), address);
    assert_eq!(reuse.allocation.island_address(0), Some(island));
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 42);
}

#[test]
fn failed_safety_unlink_retains_roots_and_permanently_prevents_reopening() {
    let process = process();
    let other = crate::lifetime::unit::tests::process();
    let cursor = AtomicU64::new(0);
    let target = nonempty_target(&process, &cursor);
    let foreign = publish(&other, &cursor, &[0], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    let handle = transition.register_link(prepared).unwrap();
    transition.install_link(handle).unwrap();
    // Test-only corrupt owner identity causes the real Closed writer to reject
    // restoration. No production failure-injection callback or mutable flag.
    process
        .lock()
        .units
        .links
        .records
        .get_mut(handle.0)
        .unwrap()
        .source = foreign;
    assert_eq!(transition.unlink_link(handle), Err(Error::StaleUnit));
    assert_eq!(transition.try_reopen(), Err(Error::StaleUnit));
    assert!(matches!(transition.batch(), Err(Error::StaleUnit)));
    let state = process.lock();
    let record = state.units.links.records.get(handle.0).unwrap();
    assert!(record.installed);
    assert!(record.bridge.is_some());
    assert_eq!(
        state.units.records.get(src.0).unwrap().outgoing,
        Some(handle.0)
    );
    assert_eq!(
        state.units.records.get(target.0).unwrap().incoming,
        Some(handle.0)
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn far_installed_target_executes_through_its_reserved_island_and_unlinks() {
    far_installed_target(false);
    far_installed_target(true);
}

#[cfg(target_arch = "aarch64")]
fn far_installed_target(with_bridge: bool) {
    use crate::executable::{
        SEGMENT_BYTES,
        output::{Metadata, Output},
    };
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut input = source_input(&process, 0, 4);
    if with_bridge {
        // A constant -> X0 transfer is nonempty but needs no NativeFrame, so
        // this synthetic System-ABI chain can execute the owned bridge too.
        input.states[0].state.live.integer.x[0] = true;
        input.states[0].state.dirty_live.integer.x[0] = true;
        input.states[0].state.bindings = Box::new([crate::abi::ValueBinding {
            value: crate::abi::GuestValue::General(0),
            location: ValueLocation::Constant(9),
        }]);
    }
    let src = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], input, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    // Keep an exactly bounded free candidate near A for the later bridge.
    // Otherwise best-fit may choose the target's large alignment gap instead
    // of the available bump space near A, avoiding the far tail we must test.
    let padding = |bytes| {
        process
            .cache
            .install(
                Output {
                    bytes: vec![0; bytes].into_boxed_slice(),
                    alignment: 16,
                    metadata: Metadata {
                        abi: HostAbi::Aarch64,
                        frame_extent: crate::abi::TRANSFER_BYTES,
                        entries: Box::new([]),
                        states: Box::new([]),
                        faults: Box::new([]),
                        traps: Box::new([]),
                        relocations: Box::new([]),
                    },
                },
                Tier::Lcq,
                |_| None,
            )
            .unwrap()
    };
    let near_slot = padding(64);
    let _near_guard = padding(64);
    // Real allocations, not fake PCs or allocator mutation. Segment-aligned
    // tiny spacers occupy successive segment starts; the target's identical
    // alignment then forces it outside AArch64's +/-128 MiB branch range.
    let mut spacers = Vec::new();
    for _ in 0..9 {
        spacers.push(
            process
                .cache
                .install(
                    Output {
                        bytes: Box::new([0; 16]),
                        alignment: SEGMENT_BYTES,
                        metadata: Metadata {
                            abi: HostAbi::Aarch64,
                            frame_extent: crate::abi::TRANSFER_BYTES,
                            entries: Box::new([]),
                            states: Box::new([]),
                            faults: Box::new([]),
                            traps: Box::new([]),
                            relocations: Box::new([]),
                        },
                    },
                    Tier::Lcq,
                    |_| None,
                )
                .unwrap(),
        );
    }
    let mut input = crate::lifetime::unit::tests::input(&process, &[4], Tier::Lcq);
    if with_bridge {
        target_binding(&mut input);
    }
    let old = input.code;
    let bytes = unsafe {
        std::slice::from_raw_parts(old.allocation.address() as *const u8, old.allocation.len())
    }
    .to_vec()
    .into_boxed_slice();
    input.code = process
        .cache
        .install(
            Output {
                bytes,
                alignment: SEGMENT_BYTES,
                metadata: old.metadata,
            },
            Tier::Lcq,
            |_| None,
        )
        .unwrap();
    let target = process
        .prepare_unit(&[process.reserve(key(4)).unwrap()], input, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let mut transition = stop(&process);
    value(&mut transition, target, 87);
    let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
    let source_pc = prepared.source_code.code.allocation.address() + 8;
    let target_pc = prepared.target_code.code.allocation.address();
    assert!(target_pc.abs_diff(source_pc) >= 1 << 27);
    let island_pc = prepared
        .source_code
        .code
        .allocation
        .island_address(0)
        .unwrap();
    let handle = transition.register_link(prepared).unwrap();
    drop(near_slot);
    transition.install_link(handle).unwrap();
    drop(spacers);
    let (patch_pc, island_pc) = if with_bridge {
        let state = process.lock();
        let bridge = state
            .units
            .links
            .records
            .get(handle.0)
            .unwrap()
            .bridge
            .as_ref()
            .unwrap();
        // Bridge was allocated near A; its terminal branch, rather than A's
        // patch, needs a far island to B. Check real relocated RX addresses.
        (
            bridge.allocation.address() + bridge.allocation.len() - 4,
            bridge.allocation.island_address(0).unwrap(),
        )
    } else {
        (source_pc, island_pc)
    };
    assert!(target_pc.abs_diff(patch_pc) >= 1 << 27);
    assert_eq!(
        unsafe { std::ptr::read_unaligned((island_pc + 8) as *const u64) },
        target_pc as u64
    );
    let patch = unsafe { std::ptr::read_unaligned(patch_pc as *const u32) };
    let delta = ((patch << 6) as i32 >> 6) * 4;
    assert_eq!(patch_pc.wrapping_add_signed(delta as isize), island_pc);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 87);
    let mut transition = stop(&process);
    transition.unlink_link(handle).unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 42);
}
