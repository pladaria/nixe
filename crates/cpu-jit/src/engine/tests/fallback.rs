use super::*;
use crate::abi::NativeExitReason;
use crate::lifetime::{Reason, unit::UnitHandle};
use nixe_cpu::execution::{ControlRequest, VcpuEventState};

// Publish through the production compiler, then explicitly unlink the source.
// This leaves real source-local fallbacks with resident destinations and no
// pending maintenance that would immediately reinstall their fast patches.
fn fixture(words: &[u32], targets: &[u64]) -> (JitThread, UnitHandle) {
    let mut thread = budget::setup(words, false);
    let process = thread.process.clone();
    let mut source = None;
    for offset in targets.iter().copied().chain([4]) {
        let Request::Owner(claim) = thread
            .reader
            .claim(thread.key(PC.checked_add(offset).unwrap()).unwrap())
            .unwrap()
        else {
            panic!()
        };
        let unit = thread
            .compiler
            .publish(
                Compilation::capture(claim, &*process.memory).unwrap(),
                &process.lifetime,
                process.lifetime.executable_cache(),
                &*process.memory,
            )
            .unwrap();
        if offset == 4 {
            source = Some(unit);
        }
    }
    assert!(process.lifetime.try_service_links().unwrap());
    let source = source.unwrap();
    let sites = process
        .lifetime
        .snapshot(source)
        .unwrap()
        .states
        .iter()
        .filter(|state| {
            state
                .transfer
                .as_ref()
                .is_some_and(|transfer| transfer.static_target.is_some())
        })
        .count();
    process.lifetime.request(Reason::LinkPatch).unwrap();
    let mut transition = process.lifetime.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    for island in 0..sites {
        if let Some(handle) = transition.refresh_static_link(source, island).unwrap() {
            transition.unlink_link(handle).unwrap();
        }
    }
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    (thread, source)
}

// Use the same admitted code/table with no cold callback. Reaching a resident
// destination proves a native PIC hit rather than another resolver success.
// These fixtures contain no faultable guest memory instructions.
pub(super) fn without_resolver(
    returns: &mut crate::ReturnStack,
    thread: &mut JitThread,
    state: &mut A64State,
    budget: PollBudget,
    events: &VcpuEventState,
) -> (crate::native::NativeReturn, PollBudget) {
    let key = thread.key(GuestVirtualAddress::new(state.pc())).unwrap();
    let _lease = thread.process.memory.acquire_execution_lease();
    let arena = thread
        .process
        .memory
        .direct_address_space_view(SPACE)
        .unwrap();
    let mut frame = NativeFrame::new(state, budget).with_return_stack(returns);
    frame.poll_requests[1] = thread.control.pending_word_address() as *const _;
    frame.poll_requests[2] = events.pending_interrupts_address() as *const _;
    let mut invocation = unsafe { thread.reader.admit(&mut frame, key) }
        .unwrap()
        .unwrap();
    let entry = invocation.payload().preferred().unwrap();
    let returned = unsafe {
        crate::native::enter_protected(
            invocation.frame(),
            arena.base as *mut u8,
            entry.canonical.get() as *const u8,
        )
    }
    .unwrap();
    drop(invocation);
    assert!(frame.indirect_pic.is_null());
    (returned, frame.budget)
}

#[test]
fn indirect_pic_resolves_br_blr_ret_then_hits_without_a_rust_resolver() {
    for branch in [0xd61f0040, 0xd63f0040, 0xd65f03c0] {
        let (mut thread, _) = fixture(
            &[0xd4200000, branch, 0xd40000e1, 0x91000400, 0xd40000e1],
            &[12],
        );
        let mut initial = A64State::default();
        initial.set_pc(PC.get() + 4);
        initial.general_register_storage_mut()[2] = PC.get() + 12;
        initial.general_register_storage_mut()[30] = PC.get() + 12;
        let mut prediction = crate::ReturnStack::default();
        if branch == 0xd65f03c0 {
            prediction.entries[0] =
                crate::rsb::Continuation::from(thread.key(PC.checked_add(12).unwrap()).unwrap());
            prediction.head = 1;
            prediction.depth = 1;
        }
        let mut returns = prediction.clone();
        let mut cold = initial.clone();
        let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
            .invoke(
                &mut returns,
                &mut NativeWorker::default(),
                &mut cold,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected indirect resolver to reach SVC")
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
        assert_eq!(budget.slice_remaining, 62);
        assert_eq!(cold.general_register_storage_mut()[0], 1);
        assert_eq!(returns.depth, u32::from(branch == 0xd63f0040));
        assert_eq!(
            cold.general_register_storage_mut()[30],
            PC.get() + if branch == 0xd63f0040 { 8 } else { 12 }
        );
        let mut hot = initial.clone();
        returns = prediction.clone();
        let (returned, budget) = without_resolver(
            &mut returns,
            &mut thread,
            &mut hot,
            PollBudget::new(4096, 64).unwrap(),
            &VcpuEventState::default(),
        );
        assert_eq!(returned.reason, NativeExitReason::Architectural);
        assert_eq!(budget.slice_remaining, 62);
        assert_eq!(hot, cold);
        assert_eq!(returns.depth, u32::from(branch == 0xd63f0040));

        // Even a populated PIC must not run the destination after source work
        // consumes the slice. The poll's exhausted path skips the whole probe.
        let mut exhausted = initial.clone();
        returns = prediction.clone();
        let (returned, budget) = without_resolver(
            &mut returns,
            &mut thread,
            &mut exhausted,
            PollBudget::new(1, 1).unwrap(),
            &VcpuEventState::default(),
        );
        assert!(returned.poll.exhausted);
        assert_eq!(budget.slice_remaining, 0);
        assert_eq!(exhausted.pc(), PC.get() + 12);
        assert_eq!(exhausted.general_register_storage_mut()[0], 0);
        assert_eq!(returns.depth, u32::from(branch == 0xd63f0040));

        let events = VcpuEventState::default();
        events.post_interrupts(4);
        let mut interrupted = initial;
        returns = prediction;
        let (returned, budget) = without_resolver(
            &mut returns,
            &mut thread,
            &mut interrupted,
            PollBudget::new(1, 64).unwrap(),
            &events,
        );
        assert_eq!(returned.reason, NativeExitReason::Control);
        assert_eq!(budget.slice_remaining, 63);
        assert_eq!(interrupted.general_register_storage_mut()[0], 0);
        assert_eq!(events.take_pending_interrupts(), 4);
        assert_eq!(returns.depth, u32::from(branch == 0xd63f0040));
        assert!(thread.process.lifetime.try_shutdown().unwrap());
    }
}

#[test]
fn indirect_pic_native_collisions_alternate_without_hit_recency() {
    let mut memory = ExecutionMemory::new();
    let targets = [0x1010, 0x3010, 0x5010];
    for (index, target) in targets.into_iter().enumerate() {
        let page = GuestPhysicalPageId::new(index as u64 + 1);
        assert!(memory.add_ram_page(page));
        // ADD X0,X0,#(index+1); SVC #7. PCs differ by 8192, so all
        // destinations select the same set for this BR's ExitSiteKey.
        let words = [0x91000000 | ((index as u32 + 1) << 10), 0xd40000e1];
        memory
            .initialize_ram(
                page,
                16,
                &words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        if index == 0 {
            memory
                .initialize_ram(page, 0, &0xd61f0040_u32.to_le_bytes())
                .unwrap(); // BR X2
        }
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(target - 16),
            page,
            MemoryPermissions::READ_EXECUTE,
        ));
    }
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, DirectBackendPolicy::Required)
        .unwrap();
    let process = Arc::new(JitProcess::new(cpu(), Arc::new(memory)).unwrap());
    let mut thread = JitThread::new(process.clone()).unwrap();
    for pc in targets.into_iter().chain([PC.get()]) {
        assert!(matches!(
            thread.demand(GuestVirtualAddress::new(pc)).unwrap(),
            Demand::Ready
        ));
    }
    let run = |thread: &mut JitThread, index: usize, resolve: bool, hit: bool| {
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[2] = targets[index];
        let budget = PollBudget::new(4096, 64).unwrap();
        let events = VcpuEventState::default();
        let (returned, budget) = if resolve {
            let (
                Some(invocation::Exit::Native {
                    returned, guest, ..
                }),
                budget,
            ) = thread
                .invoke(
                    &mut crate::ReturnStack::default(),
                    &mut NativeWorker::default(),
                    &mut state,
                    budget,
                    &events,
                )
                .unwrap()
            else {
                panic!("expected resident indirect destination")
            };
            assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
            (returned, budget)
        } else {
            without_resolver(
                &mut crate::ReturnStack::default(),
                thread,
                &mut state,
                budget,
                &events,
            )
        };
        assert_eq!(
            returned.reason,
            if hit {
                NativeExitReason::Architectural
            } else {
                NativeExitReason::Dispatch
            }
        );
        assert_eq!(budget.slice_remaining, if hit { 62 } else { 63 });
        assert_eq!(state.pc(), targets[index] + if hit { 4 } else { 0 });
        assert_eq!(
            state.general_register_storage_mut()[0],
            if hit { index as u64 + 1 } else { 0 }
        );
    };
    run(&mut thread, 0, true, true);
    run(&mut thread, 1, true, true);
    let usage = process.lifetime.executable_cache().usage().unwrap();
    for _ in 0..8 {
        run(&mut thread, 0, false, true);
    }
    run(&mut thread, 2, true, true); // Replaces A despite its recent hits.
    run(&mut thread, 0, false, false);
    run(&mut thread, 1, false, true);
    run(&mut thread, 2, false, true);
    // Relearn an evicted target: a stale weak anchor must not select C's record.
    run(&mut thread, 0, true, true); // Replaces B, not C.
    run(&mut thread, 1, false, false);
    run(&mut thread, 2, false, true);
    run(&mut thread, 0, false, true);
    assert_eq!(process.lifetime.executable_cache().usage().unwrap(), usage);
    assert!(process.lifetime.try_shutdown().unwrap());
}

#[test]
fn indirect_pic_production_vcpus_share_a_bridge_and_survive_owner_removal() {
    // Dirty X19 crosses a BR, then the target consumes it. This requires an
    // actual transfer bridge, not just an empty bridge's target fast address.
    let (mut first, _) = fixture(
        &[0xd4200000, 0x91000413, 0xd61f0040, 0x91001e61, 0xd40000e1],
        &[12],
    ); // ADD X19,X0,#1; BR X2; ADD X1,X19,#7; SVC #7.
    let process = first.process.clone();
    let mut second = JitThread::new(process.clone()).unwrap();
    let mut initial = A64State::default();
    initial.set_pc(PC.get() + 4);
    initial.general_register_storage_mut()[0] = 41;
    initial.general_register_storage_mut()[2] = PC.get() + 12;
    let target_fast = {
        let key = first.key(PC.checked_add(12).unwrap()).unwrap();
        let mut state = initial.clone();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 64).unwrap());
        let invocation = unsafe { first.reader.admit(&mut frame, key) }
            .unwrap()
            .unwrap();
        invocation.payload().preferred().unwrap().fast.get()
    };
    // Read only under admission protection. Return integer identities, never
    // a borrow/raw pointer that a later replacement could invalidate.
    let record = |thread: &mut JitThread| {
        let key = thread.key(GuestVirtualAddress::new(initial.pc())).unwrap();
        let mut state = initial.clone();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 64).unwrap());
        let mut invocation = unsafe { thread.reader.admit(&mut frame, key) }
            .unwrap()
            .unwrap();
        let table = invocation.frame().indirect_pic;
        let mut occupied = None;
        for slot in 0..crate::native::pic::WAYS {
            // SAFETY: this vCPU is paused and its announced epoch excludes
            // Closed unlinking; every nonnull cell retains its immutable owner.
            let pointer = unsafe { *table.add(slot) };
            if !pointer.is_null() {
                assert!(occupied.is_none());
                let native = unsafe { &*pointer };
                assert_eq!(native.pc, PC.get() + 12);
                occupied = Some((pointer as usize, native.address));
            }
        }
        occupied.unwrap()
    };
    let mut expected = None;
    let mut shared = None;
    let mut usage = None;
    for thread in [&mut first, &mut second] {
        let mut state = initial.clone();
        let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut NativeWorker::default(),
                &mut state,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected resident indirect destination")
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
        assert_eq!(budget.slice_remaining, 61);
        assert_eq!(state.general_register_storage_mut()[19], 42);
        assert_eq!(state.general_register_storage_mut()[1], 49);
        let identity = record(thread);
        assert_ne!(identity.1, target_fast, "expected nonempty bridge code");
        let current_usage = process.lifetime.executable_cache().usage().unwrap();
        if let Some(shared) = shared {
            assert_eq!(identity, shared); // Same immutable owner, not just equal bytes.
            assert_eq!(Some(current_usage), usage);
            assert_eq!(Some(&state), expected.as_ref());
        } else {
            shared = Some(identity);
            usage = Some(current_usage);
            expected = Some(state);
        }
        let mut hot = initial.clone();
        let (returned, budget) = without_resolver(
            &mut crate::ReturnStack::default(),
            thread,
            &mut hot,
            PollBudget::new(4096, 64).unwrap(),
            &VcpuEventState::default(),
        );
        assert_eq!(returned.reason, NativeExitReason::Architectural);
        assert_eq!(budget.slice_remaining, 61);
        assert_eq!(Some(&hot), expected.as_ref());
    }
    drop(first);
    assert_eq!(Some(record(&mut second)), shared);
    let mut hot = initial;
    let (returned, _) = without_resolver(
        &mut crate::ReturnStack::default(),
        &mut second,
        &mut hot,
        PollBudget::new(4096, 64).unwrap(),
        &VcpuEventState::default(),
    );
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(Some(hot), expected);
    drop(second);
    assert!(process.lifetime.try_shutdown().unwrap());
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
fn indirect_pic_missing_and_misaligned_targets_remain_canonical() {
    let (mut thread, _) = fixture(
        &[0xd4200000, 0xd61f0040, 0xd40000e1, 0x91000400, 0xd40000e1],
        &[12],
    );
    for destination in [PC.get() + 24, PC.get() + 13] {
        let mut state = A64State::default();
        state.set_pc(PC.get() + 4);
        state.general_register_storage_mut()[2] = destination;
        let (
            Some(invocation::Exit::Native {
                returned, guest, ..
            }),
            budget,
        ) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut NativeWorker::default(),
                &mut state,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(returned.reason, NativeExitReason::Dispatch);
        assert_eq!(guest.kind, EdgeKind::Indirect);
        assert_eq!(budget.slice_remaining, 63);
        assert_eq!(state.pc(), destination);
        assert_eq!(state.general_register_storage_mut()[0], 0);
    }
}

#[test]
fn indirect_pic_bridge_preserves_dirty_homes_lazy_flags_and_sample_resume() {
    // ADD X5,X5,#2; ADDS X19,X0,#1; BR X2;
    // ADD X1,X19,#7; CSEL X6,X7,X8,EQ; SVC #7.
    let (mut thread, _) = fixture(
        &[
            0xd4200000, 0x910008a5, 0xb1000413, 0xd61f0040, 0x91001e61, 0x9a8800e6, 0xd40000e1,
        ],
        &[16],
    );
    for input in [u64::MAX, 0, i64::MAX as u64] {
        let mut initial = A64State::default();
        initial.set_pc(PC.get() + 4);
        initial.general_register_storage_mut()[0] = input;
        initial.general_register_storage_mut()[2] = PC.get() + 16;
        initial.general_register_storage_mut()[5] = 41;
        initial.general_register_storage_mut()[7] = 111;
        initial.general_register_storage_mut()[8] = 222;
        let mut cold = initial.clone();
        let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut NativeWorker::default(),
                &mut cold,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
        assert_eq!(budget.slice_remaining, 59);
        assert_eq!(cold.general_register_storage_mut()[5], 43);
        assert_eq!(
            cold.general_register_storage_mut()[19],
            input.wrapping_add(1)
        );
        assert_eq!(
            cold.general_register_storage_mut()[1],
            input.wrapping_add(8)
        );
        assert_eq!(
            cold.general_register_storage_mut()[6],
            if input == u64::MAX { 111 } else { 222 }
        );
        let mut hot = initial;
        let (returned, budget) = without_resolver(
            &mut crate::ReturnStack::default(),
            &mut thread,
            &mut hot,
            PollBudget::new(1, 64).unwrap(),
            &VcpuEventState::default(),
        );
        assert_eq!(returned.reason, NativeExitReason::Architectural);
        assert_eq!(budget.slice_remaining, 59);
        assert_eq!(budget.sample_remaining, 4092);
        assert_eq!(hot, cold);
    }
}

#[test]
fn indirect_pic_cannot_reenter_a_retired_target_and_relearns_its_replacement() {
    let (mut thread, source) = fixture(
        &[0xd4200000, 0xd61f0040, 0xd40000e1, 0x91000400, 0xd40000e1],
        &[12],
    );
    let process = thread.process.clone();
    assert_eq!(
        process
            .lifetime
            .snapshot(source)
            .unwrap()
            .registered_handle(),
        Some(source)
    );
    let target_key = thread.key(PC.checked_add(12).unwrap()).unwrap();
    let target = {
        let mut cpu = A64State::default();
        let mut frame = NativeFrame::new(&mut cpu, PollBudget::new(4096, 64).unwrap());
        let mut invocation = unsafe { thread.reader.admit(&mut frame, target_key) }
            .unwrap()
            .unwrap();
        let entry = invocation.payload().preferred().unwrap();
        let (_, lookup) = invocation.frame_and_faults();
        lookup
            .unit(entry.canonical.get())
            .unwrap()
            .registered_handle()
            .unwrap()
    };
    let mut initial = A64State::default();
    initial.set_pc(PC.get() + 4);
    initial.general_register_storage_mut()[2] = PC.get() + 12;
    let mut state = initial.clone();
    let _ = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut state,
            PollBudget::new(4096, 64).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap();
    assert_eq!(state.pc(), PC.get() + 16);
    process.lifetime.retire_unit(target).unwrap();
    {
        let mut transition = process.lifetime.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        transition.drain_retirements().unwrap();
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
    }
    let mut state = initial.clone();
    let (returned, budget) = without_resolver(
        &mut crate::ReturnStack::default(),
        &mut thread,
        &mut state,
        PollBudget::new(4096, 64).unwrap(),
        &VcpuEventState::default(),
    );
    assert_eq!(returned.reason, NativeExitReason::Dispatch);
    assert_eq!(budget.slice_remaining, 63);
    assert_eq!(state.pc(), PC.get() + 12);
    assert_eq!(state.general_register_storage_mut()[0], 0);
    assert!(matches!(
        thread.demand(target_key.pc).unwrap(),
        Demand::Ready
    ));
    let mut cold = initial.clone();
    let _ = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut cold,
            PollBudget::new(4096, 64).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap();
    let mut hot = initial;
    let (returned, _) = without_resolver(
        &mut crate::ReturnStack::default(),
        &mut thread,
        &mut hot,
        PollBudget::new(4096, 64).unwrap(),
        &VcpuEventState::default(),
    );
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(hot, cold);
    assert_eq!(hot.general_register_storage_mut()[0], 1);
    assert!(process.lifetime.try_shutdown().unwrap());
}

#[test]
fn indirect_pic_self_loop_is_bounded_by_native_polls_without_stack_growth() {
    let (mut thread, _) = fixture(&[0xd4200000, 0xd61f0040], &[]);
    let mut initial = A64State::default();
    initial.set_pc(PC.get() + 4);
    initial.general_register_storage_mut()[2] = PC.get() + 4;
    let mut cold = initial.clone();
    let (Some(invocation::Exit::Native { returned, .. }), budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut cold,
            PollBudget::new(3, 100_000).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!()
    };
    assert!(returned.poll.exhausted);
    assert_eq!(budget.slice_remaining, 0);
    assert_eq!(cold, initial);
    let mut hot = initial.clone();
    let (returned, hot_budget) = without_resolver(
        &mut crate::ReturnStack::default(),
        &mut thread,
        &mut hot,
        PollBudget::new(3, 100_000).unwrap(),
        &VcpuEventState::default(),
    );
    assert!(returned.poll.exhausted);
    assert_eq!(hot_budget.slice_remaining, budget.slice_remaining);
    assert_eq!(hot_budget.sample_remaining, budget.sample_remaining);
    assert_eq!(hot_budget.armed_span, budget.armed_span);
    assert_eq!(hot, initial);
    assert!(thread.process.lifetime.try_shutdown().unwrap());
}

#[test]
fn static_fallback_resolves_resident_b_bl_and_conditional_targets_in_one_invocation() {
    for (branch, x0, taken) in [
        (0x14000003, 0, true),
        (0x94000003, 0, true),
        (0xb4000060, 0, true),
        (0xb4000060, 5, false),
    ] {
        let (mut thread, _) = fixture(
            &[
                0xd4200000, branch, 0x91000821, 0xd40000e1, 0x91000400, 0xd40000e1,
            ],
            &[8, 16],
        );
        let mut state = A64State::default();
        state.set_pc(PC.get() + 4);
        state.general_register_storage_mut()[0] = x0;
        let (
            Some(invocation::Exit::Native {
                returned, guest, ..
            }),
            budget,
        ) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut NativeWorker::default(),
                &mut state,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected one invocation to reach the destination SVC")
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
        assert_eq!(guest.pc.get(), PC.get() + if taken { 20 } else { 12 });
        assert_ne!(returned.reason, NativeExitReason::Dispatch);
        // The SVC itself is completed/charged outside native protection.
        assert_eq!(budget.slice_remaining, 62);
        assert_eq!(
            state.general_register_storage_mut()[0],
            x0 + u64::from(taken)
        );
        assert_eq!(
            state.general_register_storage_mut()[1],
            if taken { 0 } else { 2 }
        );
        assert_eq!(
            state.general_register_storage_mut()[30],
            if branch == 0x94000003 {
                PC.get() + 8
            } else {
                0
            }
        );
        assert!(thread.process.lifetime.try_shutdown().unwrap());
    }
}

#[test]
fn static_fallback_loop_preserves_lazy_flags_and_poll_balances() {
    let (mut thread, _) = fixture(
        &[
            0xd4200000, 0xf1000400, 0x54ffffe1, 0xd40000e1, // SUBS X0,#1; B.NE -4; SVC #7
        ],
        &[12],
    );
    let mut state = A64State::default();
    state.set_pc(PC.get() + 4);
    state.general_register_storage_mut()[0] = 5;
    let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut state,
            PollBudget::new(3, 100).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
    assert_eq!(budget.slice_remaining, 90);
    assert_eq!(budget.sample_remaining, 4089);
    assert_eq!(state.general_register_storage_mut()[0], 0);
    assert_eq!(state.nzcv().bits(), 0x60000000);
}

#[test]
fn static_fallback_preserves_guest_fp_status_and_restores_the_caller() {
    let words = [0xd4200000, 0x1e622820, 0x14000001, 0x1e622803, 0xd40000e1];
    // FADD D0,D1,D2; B next; FADD D3,D0,D2. Both additions are inexact
    // under guest round-toward-positive-infinity, unlike the caller mode.
    let (mut thread, _) = fixture(&words, &[12]);
    let mut state = A64State::default();
    state.set_pc(PC.get() + 4);
    state.set_fpcr(1 << 22);
    state.set_fpsr(2);
    state.set_vector(1, 1.0f64.to_bits().into());
    state.set_vector(2, 2.0f64.powi(-53).to_bits().into());
    let mut expected = state.clone();
    for word in &words[1..4] {
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word).unwrap();
    }
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut state,
            PollBudget::new(4096, 100).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
    assert_eq!(budget.slice_remaining, 97);
    assert_eq!(state, expected);
    assert_eq!(state.fpsr() & 0x12, 0x12);
    let mut probe = crate::abi::HostFpState::default();
    unsafe {
        probe.begin();
        probe.finish();
    }
    assert_eq!((probe.saved_control, probe.saved_status), caller);
}

#[test]
fn static_fallback_target_fault_keeps_the_source_prefix_and_target_attribution() {
    let (mut thread, _) = fixture(
        &[0xd4200000, 0x14000002, 0xd4200000, 0xf9000020, 0xd40000e1],
        &[12],
    ); // B target; STR X0,[X1]; SVC #7
    let mut state = A64State::default();
    state.set_pc(PC.get() + 4);
    state.general_register_storage_mut()[0] = 37;
    state.general_register_storage_mut()[1] = 0x5000; // Unmapped guest address.
    let (
        Some(invocation::Exit::Memory {
            instruction,
            outcome,
            ..
        }),
        budget,
    ) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut NativeWorker::default(),
            &mut state,
            PollBudget::new(4096, 100).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected a protected fault in the resolved target")
    };
    assert!(matches!(outcome, invocation::MemoryExit::Fault(_)));
    assert_eq!(instruction.key.block_key().pc.get(), PC.get() + 12);
    assert_eq!(instruction.bits, 0xf9000020);
    assert_eq!(state.pc(), PC.get() + 12);
    assert_eq!(budget.slice_remaining, 99);
    assert_eq!(state.general_register_storage_mut()[0], 37);
    assert!(thread.process.lifetime.try_shutdown().unwrap());
}

#[test]
fn static_fallback_miss_budget_and_control_do_not_execute_the_successor() {
    for mode in 0..3 {
        let (mut thread, _) = fixture(
            &[0xd4200000, 0x94000002, 0xd4200000, 0x91000400, 0xd40000e1],
            if mode == 0 { &[] } else { &[12] },
        );
        if mode == 2 {
            thread.control.request(ControlRequest::Preempt);
        }
        let mut state = A64State::default();
        state.set_pc(PC.get() + 4);
        let (
            Some(invocation::Exit::Native {
                returned, guest, ..
            }),
            budget,
        ) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut NativeWorker::default(),
                &mut state,
                PollBudget::new(4096, if mode == 1 { 1 } else { 64 }).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(guest.kind, EdgeKind::Call);
        assert_eq!(
            returned.reason,
            if mode == 2 {
                NativeExitReason::Control
            } else {
                NativeExitReason::Dispatch
            }
        );
        assert_eq!(state.pc(), PC.get() + 12);
        assert_eq!(state.general_register_storage_mut()[0], 0);
        assert_eq!(state.general_register_storage_mut()[30], PC.get() + 8);
        assert_eq!(budget.slice_remaining, if mode == 1 { 0 } else { 63 });
    }
}
