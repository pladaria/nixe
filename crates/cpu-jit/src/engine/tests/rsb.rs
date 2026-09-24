use super::*;
use crate::{ReturnStack, abi::NativeExitReason, rsb::Continuation};

#[test]
fn recursive_and_nonlocal_guest_returns_match_the_interpreter() {
    // Preserve the outer LR in X19; recurse into the SUBS/BL loop, then unwind
    // through its shared continuation. X1=1 instead skips the pending returns
    // by restoring the original X30, as a nonlocal guest return would.
    let words = [
        0xd4200000, 0x94000002, 0xd40000e1, 0xaa1e03f3, 0xf1000400, 0x54000040, 0x97fffffe,
        0xf1000421, 0x54000040, 0xd65f03c0, 0xaa1303fe, 0xd65f03c0,
    ];
    let mut thread = budget::setup(&words, false);
    for offset in [8, 36, 40, 28, 24, 16, 12, 4] {
        assert!(thread.process.lifetime.try_service_links().unwrap());
        assert!(matches!(
            thread.demand(PC.checked_add(offset).unwrap()).unwrap(),
            Demand::Ready
        ));
    }
    assert!(thread.process.lifetime.try_service_links().unwrap());
    for (depth, unwind) in [(4, 4), (24, 24), (24, 1)] {
        let mut initial = A64State::default();
        initial.set_pc(PC.get() + 4);
        initial.general_register_storage_mut()[0] = depth;
        initial.general_register_storage_mut()[1] = unwind;
        let mut expected = initial.clone();
        let mut completed = 0;
        while expected.pc() != PC.get() + 8 {
            assert!(completed < 512);
            let word = words[((expected.pc() - PC.get()) / 4) as usize];
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word)
                .unwrap();
            completed += 1;
        }
        for _ in 0..2 {
            let mut state = initial.clone();
            let mut returns = ReturnStack::default();
            let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
                .invoke(
                    &mut returns,
                    &mut NativeWorker::default(),
                    &mut state,
                    PollBudget::new(1, 512).unwrap(),
                    &VcpuEventState::default(),
                )
                .unwrap()
            else {
                panic!("expected recursive return to SVC")
            };
            assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
            assert_eq!(budget.slice_remaining, 512 - completed);
            assert_eq!(state, expected);
            assert_eq!((returns.head, returns.depth), (0, 0));
        }
        if depth == 4 || unwind == 1 {
            let mut state = initial;
            let mut returns = ReturnStack::default();
            let (returned, budget) = fallback::without_resolver(
                &mut returns,
                &mut thread,
                &mut state,
                PollBudget::new(1, 512).unwrap(),
                &VcpuEventState::default(),
            );
            assert_eq!(
                returned.reason,
                if unwind == 1 {
                    NativeExitReason::Dispatch
                } else {
                    NativeExitReason::Architectural
                }
            );
            assert_eq!(budget.slice_remaining, 512 - completed);
            assert_eq!(state, expected);
            assert_eq!((returns.head, returns.depth), (0, 0));
        }
    }
    assert!(thread.process.lifetime.try_shutdown().unwrap());
}

#[test]
fn unpublished_call_target_is_demanded_without_repeating_the_push() {
    for call in [0x94000002, 0xd63f0040] {
        let mut thread = budget::setup(
            &[0xd4200000, call, 0xd40000e1, 0x91000400, 0xd65f03c0],
            false,
        );
        for offset in [8, 4] {
            assert!(matches!(
                thread.demand(PC.checked_add(offset).unwrap()).unwrap(),
                Demand::Ready
            ));
        }
        let mut state = A64State::default();
        state.set_pc(PC.get() + 4);
        state.general_register_storage_mut()[2] = PC.get() + 12;
        let mut returns = ReturnStack::default();
        let (
            Some(invocation::Exit::Native {
                returned, guest, ..
            }),
            budget,
        ) = thread
            .invoke(
                &mut returns,
                &mut NativeWorker::default(),
                &mut state,
                PollBudget::new(1, 64).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected demand exit")
        };
        assert_eq!(returned.reason, NativeExitReason::Dispatch);
        assert_eq!(guest.kind, EdgeKind::Call);
        assert_eq!(budget.slice_remaining, 63);
        assert_eq!(state.pc(), PC.get() + 12);
        assert_eq!((returns.head, returns.depth), (1, 1));
        assert!(matches!(
            thread.demand(PC.checked_add(12).unwrap()).unwrap(),
            Demand::Ready
        ));
        assert!(thread.process.lifetime.try_service_links().unwrap());
        let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
            .invoke(
                &mut returns,
                &mut NativeWorker::default(),
                &mut state,
                budget,
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected resumed callee and return")
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
        assert_eq!(budget.slice_remaining, 61);
        assert_eq!(state.general_register_storage_mut()[0], 1);
        assert_eq!((returns.head, returns.depth), (0, 0));
        assert!(thread.process.lifetime.try_shutdown().unwrap());
    }
}

#[test]
fn nested_native_calls_preserve_returns_beyond_the_sixteen_entry_capacity() {
    for depth in [3_u32, 20] {
        let mut words = vec![0xd4200000, 0x94000002, 0xd40000e1];
        let mut entries = vec![8, 4];
        for index in 0..depth {
            let start = 12 + u64::from(index) * 16;
            entries.extend([start, start + 8]);
            words.extend([
                0xaa1e03e0 | index,         // MOV Xi,X30
                0x94000003,                 // BL next function
                0xaa0003fe | (index << 16), // MOV X30,Xi
                0xd65f03c0,                 // RET
            ]);
        }
        entries.push(12 + u64::from(depth) * 16);
        words.extend([0x91000739, 0xd65f03c0]); // ADD X25,X25,#1; RET
        let mut thread = budget::setup(&words, false);
        for offset in entries {
            assert!(thread.process.lifetime.try_service_links().unwrap());
            assert!(matches!(
                thread.demand(PC.checked_add(offset).unwrap()).unwrap(),
                Demand::Ready
            ));
        }
        assert!(thread.process.lifetime.try_service_links().unwrap());
        let mut initial = A64State::default();
        initial.set_pc(PC.get() + 4);
        let mut expected = initial.clone();
        let mut completed = 0;
        while expected.pc() != PC.get() + 8 {
            assert!(completed < 128);
            let word = words[((expected.pc() - PC.get()) / 4) as usize];
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word)
                .unwrap();
            completed += 1;
        }
        assert_eq!(completed, u64::from(depth) * 4 + 3);
        for _ in 0..2 {
            let mut state = initial.clone();
            let mut returns = ReturnStack::default();
            let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
                .invoke(
                    &mut returns,
                    &mut NativeWorker::default(),
                    &mut state,
                    PollBudget::new(1, 512).unwrap(),
                    &VcpuEventState::default(),
                )
                .unwrap()
            else {
                panic!("expected nested return to SVC")
            };
            assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
            assert_eq!(budget.slice_remaining, 512 - completed as i64);
            assert_eq!(state, expected);
            assert_eq!((returns.head, returns.depth), (0, 0));
        }
        if depth < 16 {
            let mut state = initial;
            let mut returns = ReturnStack::default();
            let (returned, budget) = fallback::without_resolver(
                &mut returns,
                &mut thread,
                &mut state,
                PollBudget::new(1, 512).unwrap(),
                &VcpuEventState::default(),
            );
            assert_eq!(returned.reason, NativeExitReason::Architectural);
            assert_eq!(budget.slice_remaining, 512 - completed as i64);
            assert_eq!(state, expected);
            assert_eq!((returns.head, returns.depth), (0, 0));
        }
        assert!(thread.process.lifetime.try_shutdown().unwrap());
    }
}

#[test]
fn pending_native_call_follows_the_guest_to_another_vcpu_pic() {
    let mut first = budget::setup(
        &[0xd4200000, 0x94000002, 0xd40000e1, 0x91000400, 0xd65f03c0],
        false,
    );
    for offset in [8, 12, 4] {
        assert!(matches!(
            first.demand(PC.checked_add(offset).unwrap()).unwrap(),
            Demand::Ready
        ));
    }
    let process = first.process.clone();
    assert!(process.lifetime.try_service_links().unwrap());
    let mut second = JitThread::new(process.clone()).unwrap();
    let mut worker = NativeWorker::default();
    let mut returns = ReturnStack::default();
    let mut state = A64State::default();
    state.set_pc(PC.get() + 4);
    let (_, budget) = first
        .invoke(
            &mut returns,
            &mut worker,
            &mut state,
            PollBudget::new(4096, 1).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap();
    assert_eq!(budget.slice_remaining, 0);
    assert_eq!(returns.depth, 1);
    assert_eq!(state.pc(), PC.get() + 12);
    // Destination vCPU has a cold PIC. It learns its own return bridge without
    // losing the migrated guest prediction or borrowing the first vCPU's table.
    let (Some(invocation::Exit::Native { guest, .. }), _) = second
        .invoke(
            &mut returns,
            &mut worker,
            &mut state,
            PollBudget::new(4096, 64).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected migrated return")
    };
    assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
    assert_eq!(returns.depth, 0);
    for _ in 0..2 {
        state.set_pc(PC.get() + 4);
        let _ = first
            .invoke(
                &mut returns,
                &mut worker,
                &mut state,
                PollBudget::new(1, 1).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap();
        assert_eq!(returns.depth, 1);
        let (returned, _) = fallback::without_resolver(
            &mut returns,
            &mut second,
            &mut state,
            PollBudget::new(1, 64).unwrap(),
            &VcpuEventState::default(),
        );
        assert_eq!(returned.reason, NativeExitReason::Architectural);
        assert_eq!((returns.head, returns.depth), (0, 0));
    }
    assert_eq!(state.general_register_storage_mut()[0], 3);
    assert!(process.lifetime.try_shutdown().unwrap());
}

#[test]
fn linked_calls_and_returns_update_predictions_once_across_polls_and_resumption() {
    for call in [0x94000002, 0xd63f0040] {
        // BL +8 / BLR X2
        let mut thread = budget::setup(
            &[0xd4200000, call, 0xd40000e1, 0x91000400, 0xd65f03c0],
            false,
        ); // BRK; call; SVC #7; ADD X0,X0,#1; RET X30
        for offset in [8, 12, 4] {
            assert!(matches!(
                thread.demand(PC.checked_add(offset).unwrap()).unwrap(),
                Demand::Ready
            ));
        }
        assert!(thread.process.lifetime.try_service_links().unwrap());
        let continuation = Continuation::from(thread.key(PC.checked_add(8).unwrap()).unwrap());
        let mut initial = A64State::default();
        initial.set_pc(PC.get() + 4);
        initial.general_register_storage_mut()[2] = PC.get() + 12;
        let mut returns = ReturnStack::default();
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
            panic!("expected complete call and return")
        };
        assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
        assert_eq!(budget.slice_remaining, 61);
        assert_eq!((returns.head, returns.depth), (0, 0));
        assert_eq!(returns.entries[0], continuation);
        assert_eq!(cold.general_register_storage_mut()[0], 1);

        for case in 0..6 {
            let mut state = initial.clone();
            let mut returns = ReturnStack::default();
            let events = VcpuEventState::default();
            if case == 5 {
                // Pause after a call made in an earlier slice, then request
                // control specifically at RET rather than at its caller.
                state.set_pc(PC.get() + 12);
                state.general_register_storage_mut()[30] = PC.get() + 8;
                returns.entries[0] = continuation;
                returns.head = 1;
                returns.depth = 1;
            }
            if matches!(case, 3 | 5) {
                events.post_interrupts(4);
            }
            let sample = if case == 0 { 4096 } else { 1 };
            let slice = match case {
                2 => 1,
                4 => 3,
                _ => 64,
            };
            let (returned, budget) = fallback::without_resolver(
                &mut returns,
                &mut thread,
                &mut state,
                PollBudget::new(sample, slice).unwrap(),
                &events,
            );
            assert_eq!(returns.entries[0], continuation);
            assert_eq!(state.general_register_storage_mut()[30], PC.get() + 8);
            if case < 2 {
                // No resolver at all: the static/PIC call and matched RET PIC
                // both remain native, including sample-only poll resumption.
                assert_eq!(returned.reason, NativeExitReason::Architectural);
                assert_eq!(budget.slice_remaining, 61);
                assert_eq!((returns.head, returns.depth), (0, 0));
                assert_eq!(state, cold);
                continue;
            }
            assert_eq!(
                returned.reason,
                if matches!(case, 3 | 5) {
                    NativeExitReason::Control
                } else {
                    NativeExitReason::Dispatch
                }
            );
            assert_eq!(returned.poll.exhausted, matches!(case, 2 | 4));
            assert_eq!(returns.depth, u32::from(case < 4));
            assert_eq!(state.pc(), PC.get() + if case >= 4 { 8 } else { 12 });
            assert_eq!(
                state.general_register_storage_mut()[0],
                u64::from(case >= 4)
            );
            if matches!(case, 3 | 5) {
                assert_eq!(events.take_pending_interrupts(), 4);
            }
            // Resume the already-completed guest transfer, not the call/RET
            // itself. There must be no duplicate push/pop across invocations.
            let (Some(invocation::Exit::Native { guest, .. }), _) = thread
                .invoke(
                    &mut returns,
                    &mut NativeWorker::default(),
                    &mut state,
                    PollBudget::new(4096, 64).unwrap(),
                    &events,
                )
                .unwrap()
            else {
                panic!("expected continuation after a poll exit")
            };
            assert_eq!(guest.kind, EdgeKind::SupervisorCall(7));
            assert_eq!((returns.head, returns.depth), (0, 0));
            assert_eq!(state, cold);
        }
        // Even with a warm RET PIC, underflow/mismatch must take its ordinary
        // canonical miss instead of turning the cached address into authority.
        for mismatch in [false, true] {
            let mut state = initial.clone();
            state.set_pc(PC.get() + 12);
            state.general_register_storage_mut()[30] = PC.get() + 8;
            let mut returns = ReturnStack::default();
            if mismatch {
                returns.entries[0] = continuation;
                returns.entries[0].pc += 4;
                returns.head = 1;
                returns.depth = 1;
            }
            let (returned, _) = fallback::without_resolver(
                &mut returns,
                &mut thread,
                &mut state,
                PollBudget::new(4096, 64).unwrap(),
                &VcpuEventState::default(),
            );
            assert_eq!(returned.reason, NativeExitReason::Dispatch);
            assert_eq!(state.pc(), PC.get() + 8);
            assert_eq!((returns.head, returns.depth), (0, 0));
        }
        assert!(thread.process.lifetime.try_shutdown().unwrap());
    }
}
