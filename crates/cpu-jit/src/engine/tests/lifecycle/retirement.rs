use super::*;
use crate::lifetime::{Error, Reason};

#[test]
fn late_source_retirement_cancels_preparations_without_cutting_surviving_returns() {
    let mut first = fixture();
    let process = first.process.clone();
    let target = publish(&mut first, TARGET);
    let mut survivors = vec![target];
    for pc in [0x1004, 0x1014, 0x3000] {
        survivors.push(publish(&mut first, GuestVirtualAddress::new(pc)));
    }
    let mut second = JitThread::new(process.clone()).unwrap();
    let mut previous = None;
    for _ in 0..8 {
        let direct = publish(&mut first, PC);
        let indirect = publish(&mut first, PC.checked_add(16).unwrap());
        assert!(process.lifetime.try_service_links().unwrap());
        if let Some((old_direct, old_indirect)) = previous {
            assert_ne!(direct, old_direct);
            assert_ne!(indirect, old_indirect);
            assert!(matches!(
                process.lifetime.snapshot(old_direct),
                Err(Error::StaleUnit)
            ));
            assert!(matches!(
                process.lifetime.snapshot(old_indirect),
                Err(Error::StaleUnit)
            ));
        }
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                warm(thread, offset, 1);
            }
        }
        let direct_map = process
            .lifetime
            .snapshot(direct)
            .unwrap()
            .states
            .iter()
            .position(|map| {
                map.transfer.as_ref().is_some_and(|transfer| {
                    transfer.static_target == Some(first.key(TARGET).unwrap())
                })
            })
            .unwrap() as u32;
        let indirect_map = process
            .lifetime
            .snapshot(indirect)
            .unwrap()
            .states
            .iter()
            .position(|map| map.exit.is_some_and(|exit| exit.kind == EdgeKind::Call))
            .unwrap() as u32;
        let dynamic = process
            .lifetime
            .prepare_dynamic_bridge(indirect, indirect_map, first.key(TARGET).unwrap())
            .unwrap()
            .unwrap();
        // A captured but unfinished compile is also invalidated by this stop.
        let Request::Owner(claim) = first
            .reader
            .claim(first.key(PC.checked_add(8).unwrap()).unwrap())
            .unwrap()
        else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &*process.memory).unwrap();
        process.lifetime.request(Reason::LinkPatch).unwrap();
        let mut transition = process.lifetime.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        let static_preparation = transition
            .prepare_link(direct, direct_map, target, 0, 0)
            .unwrap();
        let batch = transition.batch().unwrap();
        // New safety work arrives after the optional link batch was captured.
        // Acknowledging that old batch must not reopen past these retirements.
        let direct_ticket = process.lifetime.retire_unit(direct).unwrap();
        let indirect_ticket = process.lifetime.retire_unit(indirect).unwrap();
        batch.complete().unwrap();
        assert!(!transition.try_reopen().unwrap());
        assert!(!direct_ticket.is_complete().unwrap());
        assert!(!indirect_ticket.is_complete().unwrap());
        assert!(transition.drain_links().unwrap());
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 0);
        assert_eq!(
            transition.register_link(static_preparation),
            Err(Error::StaleUnit)
        );
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        assert!(direct_ticket.is_complete().unwrap());
        assert!(indirect_ticket.is_complete().unwrap());
        assert!(matches!(dynamic.emit(), Err(Error::StalePublication)));
        assert!(matches!(
            first.compiler.publish(
                compilation,
                &process.lifetime,
                process.lifetime.executable_cache(),
                &*process.memory
            ),
            Err(PublishError::Lifetime(Error::StalePublication))
        ));
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 2);
        for &unit in &survivors {
            assert!(process.lifetime.snapshot(unit).is_ok());
        }
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                let mut missing = initial(offset);
                assert!(
                    thread
                        .invoke(
                            &mut ReturnStack::default(),
                            &mut NativeWorker::default(),
                            &mut missing,
                            PollBudget::new(4096, 64).unwrap(),
                            &VcpuEventState::default()
                        )
                        .unwrap()
                        .0
                        .is_none()
                );
                // Callee->continuation roots are independent of the retired
                // callers. A pending guest return still hits without Rust.
                let continuation = PC.checked_add(offset + 4).unwrap();
                let mut returns = ReturnStack::default();
                returns.entries[0] =
                    crate::rsb::Continuation::from(thread.key(continuation).unwrap());
                returns.head = 1;
                returns.depth = 1;
                let mut state = initial(offset);
                state.set_pc(TARGET.get());
                state.general_register_storage_mut()[30] = continuation.get();
                let (returned, budget) = fallback::without_resolver(
                    &mut returns,
                    thread,
                    &mut state,
                    PollBudget::new(1, 64).unwrap(),
                    &VcpuEventState::default(),
                );
                assert_eq!(returned.reason, NativeExitReason::Architectural);
                assert_eq!(budget.slice_remaining, 62);
                assert_eq!(state.pc(), continuation.get());
                assert_eq!(state.general_register_storage_mut()[0], 1);
                assert_eq!(returns.depth, 0);
            }
        }
        previous = Some((direct, indirect));
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(
        process
            .lifetime
            .executable_cache()
            .usage()
            .unwrap()
            .committed,
        0
    );
}

#[test]
fn mixed_root_pressure_reuses_segments_after_snapshot_and_bridge_release() {
    use crate::executable::{SEGMENT_BYTES, SOFT_BYTES, Tier};

    let mut first = fixture();
    let process = first.process.clone();
    let cache = process.lifetime.executable_cache();
    let mut second = JitThread::new(process.clone()).unwrap();
    let mut previous = None;
    let mut empty_usage = None;
    for _ in 0..4 {
        let target = publish(&mut first, TARGET);
        let mut units = vec![target];
        for pc in [0x1004, 0x1014, 0x1000, 0x1010] {
            units.push(publish(&mut first, GuestVirtualAddress::new(pc)));
        }
        assert!(process.lifetime.try_service_links().unwrap());
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                warm(thread, offset, 1);
            }
        }
        let retained = process.lifetime.snapshot(target).unwrap();
        let address = retained.code.allocation.address();
        let generation = retained.code.allocation.generation;
        if let Some((old_address, old_generation, old_unit)) = previous {
            assert_eq!(address, old_address); // Actual segment/span reuse.
            assert_ne!(generation, old_generation);
            assert_ne!(retained.id, old_unit);
            let mut state = initial(0);
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 64).unwrap());
            let mut invocation = unsafe { first.reader.admit(&mut frame, first.key(PC).unwrap()) }
                .unwrap()
                .unwrap();
            let (_, directory) = invocation.frame_and_faults();
            assert_eq!(directory.unit(address).unwrap().id, retained.id);
        }
        let source = *units.last().unwrap();
        let map = process
            .lifetime
            .snapshot(source)
            .unwrap()
            .states
            .iter()
            .position(|map| map.exit.is_some_and(|exit| exit.kind == EdgeKind::Call))
            .unwrap() as u32;
        // Unpublished bridge bytes and snapshots retain storage, not execution
        // authority. Both are legitimate owners during outside-lock compilation.
        let transfer = process
            .lifetime
            .prepare_dynamic_bridge(source, map, first.key(TARGET).unwrap())
            .unwrap()
            .unwrap()
            .emit()
            .unwrap();
        let committed = cache.usage().unwrap().committed;
        // Exercise real pressure accounting without allocating 512 MiB of data.
        let charge = cache
            .charge_metadata(
                SOFT_BYTES + SEGMENT_BYTES / 2 - cache.usage().unwrap().total(),
                Tier::Lcq,
            )
            .unwrap();
        process.lifetime.request(Reason::Eviction).unwrap();
        let mut transition = process.lifetime.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        transition.relieve_pressure(0, Tier::Lcq).unwrap();
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        drop(transition);
        for unit in units {
            assert!(matches!(
                process.lifetime.snapshot(unit),
                Err(Error::StaleUnit)
            ));
        }
        assert_eq!(cache.usage().unwrap().committed, committed);
        assert!(cache.usage().unwrap().total() > SOFT_BYTES);
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                let mut state = initial(offset);
                let before = state.clone();
                let (exit, budget) = thread
                    .invoke(
                        &mut ReturnStack::default(),
                        &mut NativeWorker::default(),
                        &mut state,
                        PollBudget::new(4096, 64).unwrap(),
                        &VcpuEventState::default(),
                    )
                    .unwrap();
                assert!(exit.is_none());
                assert_eq!(state, before);
                assert_eq!(budget.slice_remaining, 64);
            }
        }
        previous = Some((address, generation, retained.id));
        drop(retained);
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 0); // Bridge still pins both.
        drop(transfer);
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 2);
        assert_eq!(cache.usage().unwrap().committed, 0);
        assert!(cache.usage().unwrap().total() < SOFT_BYTES);
        drop(charge);
        let usage = cache.usage().unwrap();
        assert_eq!(*empty_usage.get_or_insert(usage), usage);
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
}
