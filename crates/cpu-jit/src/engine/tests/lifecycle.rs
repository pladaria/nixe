use super::*;
use crate::{ReturnStack, abi::NativeExitReason, lifetime::unit::UnitHandle};

const TARGET: GuestVirtualAddress = GuestVirtualAddress::new(0x2000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x4000);

mod coordination;
mod retirement;

fn fixture() -> JitThread {
    let mut memory = ExecutionMemory::new();
    for (page, words) in [
        // BL target; SVC #7; BRK; BRK; BLR X2; SVC #7.
        (
            1,
            vec![
                0x94000400_u32,
                0xd40000e1,
                0xd4200000,
                0xd4200000,
                0xd63f0040,
                0xd40000e1,
            ],
        ),
        (2, vec![0x91000400, 0xd65f03c0]), // ADD X0,X0,#1; RET.
        (3, vec![0xd4200000]),             // Unrelated resident code.
    ] {
        let physical = GuestPhysicalPageId::new(page);
        assert!(memory.add_ram_page(physical));
        memory
            .initialize_ram(
                physical,
                0,
                &words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(page * 4096),
            physical,
            MemoryPermissions::READ_EXECUTE
        ));
    }
    assert!(memory.map_page(
        SPACE,
        ALIAS,
        GuestPhysicalPageId::new(2),
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, DirectBackendPolicy::Required)
        .unwrap();
    JitThread::new(Arc::new(JitProcess::new(cpu(), Arc::new(memory)).unwrap())).unwrap()
}

fn publish(thread: &mut JitThread, pc: GuestVirtualAddress) -> UnitHandle {
    let process = thread.process.clone();
    assert!(process.lifetime.try_service_links().unwrap());
    let Request::Owner(claim) = thread.reader.claim(thread.key(pc).unwrap()).unwrap() else {
        panic!("expected an unpublished entry")
    };
    thread
        .compiler
        .publish(
            Compilation::capture(claim, &*process.memory).unwrap(),
            &process.lifetime,
            process.lifetime.executable_cache(),
            &*process.memory,
        )
        .unwrap()
}

fn initial(offset: u64) -> A64State {
    let mut state = A64State::default();
    state.set_pc(PC.get() + offset);
    state.general_register_storage_mut()[2] = TARGET.get();
    state
}

fn warm(thread: &mut JitThread, offset: u64, increment: u64) {
    let mut state = initial(offset);
    let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
        .invoke(
            &mut ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut state,
            PollBudget::new(4096, 64).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected completed call")
    };
    assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
    assert_eq!(budget.slice_remaining, 61);
    assert_eq!(state.general_register_storage_mut()[0], increment);
    let mut hot = initial(offset);
    let mut returns = ReturnStack::default();
    let (returned, budget) = fallback::without_resolver(
        &mut returns,
        thread,
        &mut hot,
        PollBudget::new(1, 64).unwrap(),
        &VcpuEventState::default(),
    );
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(budget.slice_remaining, 61);
    assert_eq!(hot, state);
    assert_eq!(returns.depth, 0);
}

#[test]
fn alias_write_and_permission_change_cut_static_and_all_vcpu_pic_roots() {
    for permission_change in [false, true] {
        let mut first = fixture();
        let process = first.process.clone();
        let old = publish(&mut first, TARGET);
        let mut survivors = Vec::new();
        for pc in [0x1004, 0x1014, 0x1000, 0x1010, 0x3000] {
            survivors.push(publish(&mut first, GuestVirtualAddress::new(pc)));
        }
        assert!(process.lifetime.try_service_links().unwrap());
        let mut second = JitThread::new(process.clone()).unwrap();
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                warm(thread, offset, 1);
            }
        }
        // Both callers and both vCPU PICs root the same callee. A compiler
        // snapshot may keep its bytes alive, but must not keep it callable.
        let retained = process.lifetime.snapshot(old).unwrap();
        let old_address = retained.code.allocation.address();
        if permission_change {
            process
                .memory
                .set_permissions(SPACE, TARGET, 4096, MemoryPermissions::READ)
                .unwrap();
        } else {
            process
                .memory
                .overwrite_mapped_ram(SPACE, ALIAS, &0x91000800_u32.to_le_bytes())
                .unwrap(); // ADD X0,X0,#2.
        }
        assert!(process.lifetime.try_service_links().unwrap());
        assert!(matches!(
            process.lifetime.snapshot(old),
            Err(lifetime::Error::StaleUnit)
        ));
        for &unit in &survivors {
            assert!(process.lifetime.snapshot(unit).is_ok());
        }
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 0);
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                let mut state = initial(offset);
                let mut returns = ReturnStack::default();
                let (returned, budget) = fallback::without_resolver(
                    &mut returns,
                    thread,
                    &mut state,
                    PollBudget::new(4096, 64).unwrap(),
                    &VcpuEventState::default(),
                );
                assert_eq!(returned.reason, NativeExitReason::Dispatch);
                assert_eq!(budget.slice_remaining, 63);
                assert_eq!(state.pc(), TARGET.get());
                assert_eq!(state.general_register_storage_mut()[0], 0);
                assert_eq!(returns.depth, 1); // Unlink retained the BL push.
            }
        }
        if permission_change {
            assert!(matches!(
                first.demand(TARGET).unwrap(),
                Demand::FetchFault(_)
            ));
            process
                .memory
                .set_permissions(SPACE, TARGET, 4096, MemoryPermissions::READ_EXECUTE)
                .unwrap();
        }
        let replacement = publish(&mut first, TARGET);
        assert!(process.lifetime.try_service_links().unwrap());
        assert_ne!(
            process
                .lifetime
                .snapshot(replacement)
                .unwrap()
                .code
                .allocation
                .address(),
            old_address
        );
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                warm(thread, offset, if permission_change { 1 } else { 2 });
            }
        }
        drop(retained);
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 1);
        // Live vCPU owners must not keep their bridge roots after shutdown.
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
}

#[test]
fn later_unit_fault_epoch_delays_mixed_root_unlink_and_alias_visibility() {
    use nixe_cpu::memory::InstructionMemory;
    use nixe_memory::MemoryInvalidationKind;
    use std::sync::mpsc;
    use std::time::Duration;

    let mut thread = fixture();
    let process = thread.process.clone();
    // LDR X0,[X1]; RET, with the load in the callee, not the admitted caller.
    process
        .memory
        .overwrite_mapped_ram(SPACE, ALIAS, &0xf9400020_u32.to_le_bytes())
        .unwrap();
    let target = publish(&mut thread, TARGET);
    for pc in [0x1004, 0x1014, 0x1000, 0x1010] {
        publish(&mut thread, GuestVirtualAddress::new(pc));
    }
    assert!(process.lifetime.try_service_links().unwrap());
    for offset in [0, 16] {
        let mut state = initial(offset);
        state.general_register_storage_mut()[1] = 0x3000;
        let (Some(invocation::Exit::Native { guest, .. }), _) = thread
            .invoke(
                &mut ReturnStack::default(),
                &mut NativeWorker::default(),
                &mut state,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected the faultable callee to return")
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
    }
    let snapshot = process.lifetime.snapshot(target).unwrap();
    let native_pc = snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize;
    let mut observer = process.lifetime.register().unwrap();
    let mut blocked_reader = process.lifetime.register().unwrap();
    let blocked_key = thread.key(GuestVirtualAddress::new(0x1010)).unwrap();
    let mut state = initial(0);
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 64).unwrap());
    let lease = process.memory.acquire_execution_lease();
    // Hold the exact guard used through native fault dispatch/retry. No code is
    // running in this interval; the protected metadata belongs to a later unit.
    let invocation = unsafe { observer.admit(&mut frame, thread.key(PC).unwrap()) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.fault(native_pc).unwrap().unit.id, snapshot.id);
    let ticket = process
        .lifetime
        .invalidate_memory(&[MemoryInvalidationKind::ExecutableContent {
            first: GuestPhysicalPageId::new(2),
            second: None,
        }])
        .unwrap();
    std::thread::scope(|scope| {
        let fault = invocation.fault(native_pc).unwrap();
        let (owned_tx, owned_rx) = mpsc::sync_channel(0);
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let process = &process;
        let writer = scope.spawn(move || {
            let mut transition = process.lifetime.try_transition().unwrap().unwrap();
            // A separate OS thread tests rejection without nesting FP owners.
            let mut blocked_state = initial(16);
            let mut blocked_frame =
                NativeFrame::new(&mut blocked_state, PollBudget::new(4096, 64).unwrap());
            assert!(matches!(
                unsafe { blocked_reader.admit(&mut blocked_frame, blocked_key) },
                Err(lifetime::Error::Closed)
            ));
            owned_tx.send(()).unwrap();
            transition.wait_closed().unwrap();
            transition.drain_retirements().unwrap();
            // Yield coordinator ownership while keeping MappingChange pending.
            // The real memory authority acquires its own hold through visibility.
            drop(transition);
            process
                .memory
                .overwrite_mapped_ram(SPACE, ALIAS, &0x91000800_u32.to_le_bytes())
                .unwrap();
            done_tx.send(()).unwrap();
        });
        owned_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!ticket.is_complete().unwrap());
        assert!(matches!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(
            process.memory.fetch32(SPACE, TARGET).unwrap().bits,
            0xf9400020
        );
        assert_eq!(invocation.fault(native_pc).unwrap().unit.id, fault.unit.id);
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 0);
        drop(invocation);
        drop(lease);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        writer.join().unwrap();
    });
    assert!(ticket.is_complete().unwrap());
    assert_eq!(
        process.memory.fetch32(SPACE, TARGET).unwrap().bits,
        0x91000800
    );
    assert!(matches!(
        process.lifetime.snapshot(target),
        Err(lifetime::Error::StaleUnit)
    ));
    assert_eq!(process.lifetime.reclaim_units().unwrap(), 0);
    // Native-PC records remain valid while retained, then disappear before the
    // span is reusable. A surviving caller protects the directory lookup.
    drop(snapshot);
    assert_eq!(process.lifetime.reclaim_units().unwrap(), 1);
    let admitted = unsafe { observer.admit(&mut frame, thread.key(PC).unwrap()) }
        .unwrap()
        .unwrap();
    assert!(admitted.fault(native_pc).is_none());
    drop(admitted);
    publish(&mut thread, TARGET);
    assert!(process.lifetime.try_service_links().unwrap());
    for offset in [0, 16] {
        warm(&mut thread, offset, 2);
    }
    assert!(process.try_shutdown().unwrap());
}
