use super::*;
use crate::lifetime::unit::dynamic::tests::source;
use crate::lifetime::unit::tests::{key, process, publish};
use crate::native::pic::WAYS;

fn install(
    reader: &mut Reader,
    process: &Lifetime,
    source: UnitHandle,
    map: u32,
    pc: u64,
) -> PicHandle {
    let transfer = process
        .prepare_dynamic_bridge(source, map, key(pc))
        .unwrap()
        .unwrap();
    reader.cache_bridge(transfer).unwrap()
}

fn cached(process: &Lifetime, handle: PicHandle) -> Option<usize> {
    if handle.process != process.identity {
        return None;
    }
    let state = process.lock();
    let pic = &state.readers.get(handle.site.reader)?.pic;
    // These tests never run native probes concurrently with a table write.
    let native = unsafe { *pic.native.as_ptr().add(handle.site.slot) };
    let Some(bridge) = pic.way(handle.site.slot).bridge.as_ref() else {
        assert!(native.is_null());
        return None;
    };
    assert_eq!(native, &bridge.native as *const Record);
    assert_eq!(
        unsafe { &*native },
        &Record::new(bridge.key.source, bridge.key.target, bridge.native.address)
    );
    (bridge.generation == handle.generation).then_some(bridge.native.address)
}

#[test]
fn pic_native_view_survives_registry_growth_and_other_vcpu_backlink_writes() {
    use nixe_cpu::state::a64::A64State;
    use std::sync::{Barrier, atomic::AtomicBool};

    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8196, 16388], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let handle = install(&mut reader, &process, source, 0, 4);
    let (table, record, address) = {
        let state = process.lock();
        let pic = &state.readers.get(reader.handle).unwrap().pic;
        let bridge = pic.way(handle.site.slot).bridge.as_ref().unwrap();
        (
            pic.native.as_ptr() as usize,
            &bridge.native as *const Record as usize,
            bridge.native.address,
        )
    };
    let mut cpu = A64State::default();
    let mut frame = crate::lifetime::unit::tests::frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let start = Barrier::new(2);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let probe = scope.spawn(|| {
            start.wait();
            for _ in 0..100_000 {
                // SAFETY: the invocation retains this vCPU registration and
                // epoch. Only other vCPUs mutate, so our table/record stay
                // unchanged even as cold code mutably borrows Registration.
                let found = unsafe { *(table as *const *const Record).add(handle.site.slot) };
                assert_eq!(found as usize, record);
                assert_eq!(unsafe { (*found).address }, address);
                if done.load(Ordering::Acquire) {
                    break;
                }
            }
        });
        start.wait();
        let mut others: Vec<_> = (0..32).map(|_| process.register().unwrap()).collect();
        for pc in [4, 8196, 16388].into_iter().cycle().take(128) {
            // Shared source/target lists update backlinks in our active reader;
            // those ownership writes must not touch its native-readable cells.
            install(&mut others[0], &process, source, 0, pc);
        }
        drop(others);
        done.store(true, Ordering::Release);
        probe.join().unwrap();
    });
    assert_eq!(cached(&process, handle), Some(address));
    drop(invocation);
    process.retire_unit(source).unwrap();
    close_retirements(&process);
    assert!(cached(&process, handle).is_none());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn pic_retirement_waits_for_the_existing_invocation_epoch() {
    use nixe_cpu::state::a64::A64State;
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let handle = install(&mut reader, &process, source, 0, 4);
    let mut cpu = A64State::default();
    let mut frame = crate::lifetime::unit::tests::frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    process.retire_unit(target).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    // No wait under our own epoch: attempted mutation must fail without
    // cutting the edge or releasing its RX owner while the reader is active.
    assert_eq!(transition.drain_retirements(), Err(Error::Closed));
    assert!(cached(&process, handle).is_some());
    drop(invocation);
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert!(cached(&process, handle).is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn pic_target_churn_keeps_two_roots_and_a_fixed_metadata_charge() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8196, 16388], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    install(&mut reader, &process, source, 0, 4);
    install(&mut reader, &process, source, 0, 8196);
    let before = process.cache.usage().unwrap();
    for pc in [16388, 4, 8196].into_iter().cycle().take(1000) {
        install(&mut reader, &process, source, 0, pc);
        assert_eq!(process.cache.usage().unwrap(), before);
        let state = process.lock();
        let pic = &state.readers.get(reader.handle).unwrap().pic;
        assert_eq!(pic.sets.len() * 2, WAYS);
        let first = pic.head.unwrap();
        let second = pic.way(first).next.unwrap();
        assert!(pic.way(second).next.is_none());
    }
    assert!(process.try_shutdown().unwrap());
}

fn close_retirements(process: &Lifetime) {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn pic_suspended_installation_keeps_the_invocation_epoch_and_private_table() {
    use nixe_cpu::state::a64::A64State;
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8196, 16388], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let handle = reader.handle;
    let announcement = Arc::clone(&reader.announcement);
    let table = process
        .lock()
        .readers
        .get(handle)
        .unwrap()
        .pic
        .native_table();
    let mut cpu = A64State::default();
    let mut frame = crate::lifetime::unit::tests::frame(&mut cpu);
    let mut invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let (native_frame, mut lookup) = invocation.frame_and_faults();
    let epoch = native_frame.execution_epoch;
    assert_eq!(native_frame.indirect_pic, table);
    assert_ne!(epoch, 0);
    unsafe { native_frame.suspend_fp() };
    {
        // No native execution occurs while this exclusive dispatcher borrow
        // installs/replaces ways. The existing epoch remains announced.
        let mut suspended = unsafe { lookup.suspend_native() };
        for pc in [4, 8196, 16388].into_iter().cycle().take(100) {
            let prepared = process
                .prepare_dynamic_bridge(source, 0, key(pc))
                .unwrap()
                .unwrap();
            let slot = set_index(prepared.key.source, prepared.key.target) * 2;
            suspended.cache_bridge(prepared).unwrap();
            assert_eq!(announcement.load(Ordering::Acquire), epoch);
            let found = (slot..slot + 2).any(|slot| {
                let record = unsafe { *table.add(slot) };
                !record.is_null() && unsafe { (*record).pc == pc }
            });
            assert!(found);
        }
    }
    assert!(lookup.static_entry(key(4)).unwrap().is_some());
    assert_eq!(native_frame.execution_epoch, epoch);
    drop(invocation);
    assert!(frame.indirect_pic.is_null());
    assert_eq!(announcement.load(Ordering::Acquire), 0);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn pic_suspended_installation_rejects_closure_without_cutting_the_existing_way() {
    use nixe_cpu::state::a64::A64State;
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4, 8196], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Return);
    let mut reader = process.register().unwrap();
    let existing = install(&mut reader, &process, source, 0, 4);
    let prepared = process
        .prepare_dynamic_bridge(source, 0, key(8196))
        .unwrap()
        .unwrap();
    let mut cpu = A64State::default();
    let mut frame = crate::lifetime::unit::tests::frame(&mut cpu);
    let mut invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let (native_frame, mut lookup) = invocation.frame_and_faults();
    unsafe { native_frame.suspend_fp() };
    let table = native_frame.indirect_pic;
    let old = unsafe { *table.add(existing.site.slot) };
    process.retire_unit(target).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    {
        let mut suspended = unsafe { lookup.suspend_native() };
        assert_eq!(suspended.cache_bridge(prepared), Err(Error::Closed));
    }
    assert_eq!(transition.drain_retirements(), Err(Error::Closed));
    assert_eq!(unsafe { *table.add(existing.site.slot) }, old);
    assert!(cached(&process, existing).is_some());
    drop(invocation);
    assert!(frame.indirect_pic.is_null());
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert!(cached(&process, existing).is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn pic_suspended_installation_cannot_publish_a_different_process_bridge() {
    use nixe_cpu::state::a64::A64State;
    let first = process();
    let second = process();
    let cursor = AtomicU64::new(0);
    let source = source(&first, &cursor, 0, EdgeKind::Indirect);
    publish(&first, &cursor, &[4], Tier::Lcq);
    publish(&second, &cursor, &[0], Tier::Lcq);
    let prepared = first
        .prepare_dynamic_bridge(source, 0, key(4))
        .unwrap()
        .unwrap();
    let mut reader = second.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = crate::lifetime::unit::tests::frame(&mut cpu);
    let mut invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let (native_frame, mut lookup) = invocation.frame_and_faults();
    unsafe { native_frame.suspend_fp() };
    assert_eq!(
        unsafe { lookup.suspend_native() }.cache_bridge(prepared),
        Err(Error::StaleUnit)
    );
    for slot in 0..WAYS {
        assert!(unsafe { (*native_frame.indirect_pic.add(slot)).is_null() });
    }
    drop(invocation);
    // Failed admission must also clear the borrowed native pointer before the
    // frame survives or the reader registration can be destroyed/reused.
    assert!(
        unsafe { reader.admit(&mut frame, key(8)) }
            .unwrap()
            .is_none()
    );
    assert!(frame.indirect_pic.is_null());
    assert_eq!(frame.execution_epoch, 0);
    assert!(second.try_shutdown().unwrap());
    assert_eq!(
        unsafe { reader.admit(&mut frame, key(0)) }.err(),
        Some(Error::Shutdown)
    );
    assert!(frame.indirect_pic.is_null());
}

#[test]
fn pic_full_keys_two_way_replacement_and_generations() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8196, 16388], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let a = install(&mut reader, &process, source, 0, 4);
    let b = install(&mut reader, &process, source, 0, 8196);
    assert_eq!(a.site.slot / 2, b.site.slot / 2);
    assert_ne!(a.site.slot, b.site.slot);
    assert!(cached(&process, a).is_some());
    assert!(cached(&process, b).is_some());
    let before = process.cache.usage().unwrap();
    assert_eq!(install(&mut reader, &process, source, 0, 4), a);
    assert_eq!(process.cache.usage().unwrap(), before);
    let c = install(&mut reader, &process, source, 0, 16388);
    assert_eq!(c.site.slot, a.site.slot); // A hit did not write recency.
    assert_ne!(c.generation, a.generation);
    assert!(cached(&process, a).is_none());
    assert!(cached(&process, b).is_some());
    let other_map = install(&mut reader, &process, source, 1, 4);
    assert_ne!(other_map, c);
    process.retire_unit(source).unwrap();
    close_retirements(&process);
    for handle in [b, c, other_map] {
        assert!(cached(&process, handle).is_none());
    }
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn pic_retirement_detaches_only_affected_ways_across_vcpus() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    publish(&process, &cursor, &[8], Tier::Lcq);
    let a = source(&process, &cursor, 0, EdgeKind::Indirect);
    let b = source(&process, &cursor, 12, EdgeKind::Return);
    let mut first = process.register().unwrap();
    let mut second = process.register().unwrap();
    let x = install(&mut first, &process, a, 0, 4);
    let y = install(&mut second, &process, b, 0, 4);
    let unrelated = install(&mut first, &process, a, 1, 8);
    process.retire_unit(target).unwrap();
    // Requesting a stop doesn't release a still-callable PIC way.
    assert!(cached(&process, x).is_some());
    assert!(cached(&process, y).is_some());
    close_retirements(&process);
    assert!(cached(&process, x).is_none());
    assert!(cached(&process, y).is_none());
    assert!(cached(&process, unrelated).is_some());
    process.reclaim_units().unwrap();
    assert!(process.lock().units.records.get(target.0).is_none());
    assert!(process.try_shutdown().unwrap());
    assert!(cached(&process, unrelated).is_none());
}

#[test]
fn pic_vcpu_teardown_releases_roots_storage_and_cannot_revive_old_handles() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Call);
    // Prime the reusable reader registry allocation; only per-vCPU storage
    // and occupied bridge owners should subsequently appear and disappear.
    drop(process.register().unwrap());
    let before = process.cache.usage().unwrap();
    let mut reader = process.register().unwrap();
    let a = install(&mut reader, &process, source, 0, 4);
    let b = install(&mut reader, &process, source, 1, 4);
    assert!(process.cache.usage().unwrap().metadata > before.metadata);
    drop(reader);
    assert_eq!(process.cache.usage().unwrap(), before);
    assert!(cached(&process, a).is_none());
    assert!(cached(&process, b).is_none());
    {
        let state = process.lock();
        assert!(
            state
                .units
                .records
                .get(source.0)
                .unwrap()
                .pic_outgoing
                .is_none()
        );
        assert!(
            state
                .units
                .records
                .get(target.0)
                .unwrap()
                .pic_incoming
                .is_none()
        );
    }
    let mut reader = process.register().unwrap();
    let new = install(&mut reader, &process, source, 0, 4);
    assert_ne!(new, a);
    assert!(cached(&process, a).is_none());
    assert!(cached(&process, new).is_some());
    assert!(process.try_shutdown().unwrap());
    drop(reader); // Registration was already released by shutdown.
}

#[test]
fn pic_self_edge_and_stale_installation_release_both_adjacencies() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let handle = install(&mut reader, &process, source, 0, 0);
    let stale = process
        .prepare_dynamic_bridge(source, 1, key(0))
        .unwrap()
        .unwrap();
    process.retire_unit(source).unwrap();
    close_retirements(&process);
    assert!(cached(&process, handle).is_none());
    assert!(reader.cache_bridge(stale).is_err());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn pic_replacement_generation_exhaustion_preserves_existing_root() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8196], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let existing = install(&mut reader, &process, source, 0, 4);
    let transfer = process
        .prepare_dynamic_bridge(source, 0, key(8196))
        .unwrap()
        .unwrap();
    process.lock().bridge_generations = CheckedCounter::exhausted();
    assert!(matches!(
        reader.cache_bridge(transfer),
        Err(Error::Exhausted(_))
    ));
    assert!(cached(&process, existing).is_some());
    drop(reader); // Even poisoned admission must detach its roots safely.
    assert!(cached(&process, existing).is_none());
}
