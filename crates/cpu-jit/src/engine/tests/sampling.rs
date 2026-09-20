use super::*;
use crate::lcq::invocation::{Exit, MemoryExit};
use nixe_cpu::execution::ControlRequest;

#[test]
fn guest_migration_keeps_sample_phase_and_heat_with_each_vcpu() {
    use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot};
    struct Timer;
    impl ArchitecturalTimer for Timer {
        fn snapshot(&self) -> TimerSnapshot {
            TimerSnapshot {
                counter: 0,
                frequency: 1,
            }
        }
    }
    let first = budget::setup(&[0x14000000], false); // B self; one completed instruction.
    let second = JitThread::new(Arc::clone(&first.process)).unwrap();
    let key = first.key(PC).unwrap();
    let mut vcpus = [first, second];
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let mut returns = crate::ReturnStack::default();
    let mut worker = NativeWorker::default();
    // The same guest moves between vCPUs. Neither its heat nor its deadline
    // follows it; samples accumulate only on the vCPU that completed the work.
    for (vcpu, budget, phases, scores) in [
        (0, 4095, [1, 4096], [0, 0]),
        (1, 1, [1, 4095], [0, 0]),
        (0, 1, [4096, 4095], [1, 0]),
        (1, 4095, [4096, 4096], [1, 1]),
        (0, 4096, [4096, 4096], [2, 1]),
    ] {
        let report = vcpus[vcpu]
            .run_slice(
                &mut returns,
                &mut worker,
                &mut state,
                budget,
                &Timer,
                &VcpuEventState::default(),
            )
            .unwrap();
        assert_eq!(report.stop, CpuExit::BudgetExhausted);
        assert_eq!(report.progress, budget);
        for index in 0..2 {
            assert_eq!(vcpus[index].sample_remaining, phases[index]);
            assert_eq!(
                vcpus[index]
                    .samples
                    .seed_snapshot(key)
                    .map_or(0, |(_, score)| score),
                scores[index]
            );
        }
    }
    worker.finish().unwrap();
}

#[test]
fn resumable_samples_follow_the_later_linked_source_without_recharging_work() {
    // A: B B; B: ADDS X0,X0,#1; B B. All samples occur in B,
    // including a deadline overshot by one instruction when phase is two.
    for (phase, slice) in [(2, 10003), (3, 10003), (4096, 10003), (3, 8195)] {
        let mut thread = budget::setup(&[0x14000001, 0xb1000400, 0x17ffffff], false);
        let source = thread.key(PC.checked_add(4).unwrap()).unwrap();
        assert!(matches!(thread.demand(source.pc).unwrap(), Demand::Ready));
        assert!(thread.process.lifetime.try_service_links().unwrap());
        let mut state = A64State::default();
        state.set_pc(PC.get());
        let mut worker = NativeWorker::default();
        let (Some(Exit::Native { returned, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(phase, slice).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected linked-loop slice exit")
        };
        let expected = 1 + (slice - phase) / 4096;
        assert!(returned.poll.exhausted);
        assert_eq!(returned.poll.sample, (slice - phase) % 4096 == 0);
        assert_eq!(budget.slice_remaining, 0);
        assert_eq!(budget.sample_remaining, phase + expected * 4096 - slice);
        assert_eq!(
            state.general_register_storage_mut()[0],
            (slice as u64 - 1) / 2
        );
        assert_eq!(state.pc(), source.pc.get());
        assert_eq!(state.nzcv().bits(), 0);
        let (snapshot, score) = thread.samples.seed_snapshot(source).unwrap();
        assert_eq!(i64::from(score), expected);
        assert_eq!(snapshot.sequence, expected as u64);
        assert_eq!(
            snapshot.last_edge,
            Some(crate::sampling::ObservedEdge {
                destination: source.pc,
                kind: EdgeKind::Static,
            })
        );
        assert_eq!(i64::from(snapshot.successors[0].unwrap().count), expected);
        assert!(
            thread
                .samples
                .seed_snapshot(thread.key(PC).unwrap())
                .is_none()
        );
    }
}

#[test]
fn resumable_samples_keep_live_fp_and_software_status_across_a_native_loop() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    // A: B B; B: FADD D0,D1,D2; ADDS X0,X0,#1; B B.
    let mut thread = budget::setup(&[0x14000001, 0x1e622820, 0xb1000400, 0x17fffffe], false);
    let source = thread.key(PC.checked_add(4).unwrap()).unwrap();
    assert!(matches!(thread.demand(source.pc).unwrap(), Demand::Ready));
    assert!(thread.process.lifetime.try_service_links().unwrap());
    let mut state = A64State::default();
    state.set_pc(PC.get());
    state.set_fpsr(1 << 27);
    state.set_vector(1, u128::from(1.0f64.to_bits()));
    state.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    let mut worker = NativeWorker::default();
    let (Some(Exit::Native { returned, .. }), budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            PollBudget::new(3, 9001).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected FP loop slice exit")
    };
    assert!(returned.poll.exhausted && !returned.poll.sample);
    assert_eq!(budget.slice_remaining, 0);
    assert_eq!(budget.sample_remaining, 3290);
    assert_eq!(state.general_register_storage_mut()[0], 3000);
    assert_eq!(state.vector(0), Some(u128::from(1.0f64.to_bits())));
    assert_eq!(state.fpsr(), (1 << 27) | (1 << 4));
    assert_eq!(thread.samples.seed_snapshot(source).unwrap().1, 3);
    let mut host = crate::abi::HostFpState::default();
    unsafe {
        host.begin();
        host.finish();
    }
    assert_eq!((host.saved_control, host.saved_status), caller);
}

#[test]
fn canonical_sample_keeps_guest_sticky_status_and_restores_caller_fp() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let mut thread = budget::setup(&[0x1e622820, 0xd4200000], false); // FADD; BRK.
    let mut worker = NativeWorker::default();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    state.set_fpsr(1 << 27);
    state.set_vector(1, u128::from(1.0f64.to_bits()));
    state.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    let (
        Some(Exit::Native {
            returned, guest, ..
        }),
        budget,
    ) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            PollBudget::new(1, 20).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected native FP completion")
    };
    let mut host = crate::abi::HostFpState::default();
    unsafe {
        host.begin();
    }
    let after = (host.saved_control, host.saved_status);
    unsafe {
        host.finish();
    }
    assert_eq!(after, caller);
    assert_eq!(guest.kind, EdgeKind::Breakpoint(0));
    assert!(returned.poll.sample);
    assert_eq!(budget.slice_remaining, 19);
    assert_eq!(state.fpsr(), (1 << 27) | (1 << 4));
    assert_eq!(
        thread
            .samples
            .seed_snapshot(thread.key(PC).unwrap())
            .unwrap()
            .1,
        1
    );
}

#[test]
fn canonical_sample_attributes_the_later_linked_source_and_actual_edge() {
    // A: B B; B: ADDS X0,X0,#1; B.EQ .
    let mut thread = budget::setup(&[0x14000001, 0xb1000400, 0x54000000], false);
    let source = thread.key(PC.checked_add(4).unwrap()).unwrap();
    assert!(matches!(thread.demand(source.pc).unwrap(), Demand::Ready));
    assert!(thread.process.lifetime.try_service_links().unwrap());
    let mut worker = NativeWorker::default();
    for index in 0..8 {
        let taken = index % 2 != 0;
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[0] = if taken { u64::MAX } else { 0 };
        let (Some(Exit::Native { returned, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(3, 3).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected canonical slice exit")
        };
        assert!(returned.poll.sample && returned.poll.exhausted);
        assert_eq!(budget.sample_remaining, 4096);
        assert_eq!(budget.slice_remaining, 0);
        assert_eq!(state.nzcv().bits(), if taken { 0x60000000 } else { 0 });
        let (snapshot, score) = thread.samples.seed_snapshot(source).unwrap();
        assert_eq!(snapshot.sequence, index + 1);
        assert_eq!(u64::from(score), index + 1);
        assert_eq!(
            snapshot.last_edge,
            Some(crate::sampling::ObservedEdge {
                destination: PC.checked_add(if taken { 8 } else { 12 }).unwrap(),
                kind: if taken {
                    EdgeKind::Taken
                } else {
                    EdgeKind::NotTaken
                },
            })
        );
        assert!(
            thread
                .samples
                .seed_snapshot(thread.key(PC).unwrap())
                .is_none()
        );
    }
    let (snapshot, _) = thread.samples.seed_snapshot(source).unwrap();
    assert_eq!(snapshot.successors[0].unwrap().count, 4);
    assert_eq!(snapshot.successors[1].unwrap().count, 4);
}

#[test]
fn canonical_pre_sample_has_no_fabricated_successor_or_pending_instruction_charge() {
    for word in [0xd4200000, 0xd4000001, 0xd53b4420] {
        // BRK; SVC; MRS X0,FPSR.
        let mut thread = budget::setup(&[0xd503201f, word], false);
        let mut worker = NativeWorker::default();
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[0] = 99;
        let (_, budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(1, 20).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap();
        let (snapshot, score) = thread
            .samples
            .seed_snapshot(thread.key(PC).unwrap())
            .unwrap();
        assert_eq!((snapshot.sequence, score), (1, 1));
        assert_eq!(snapshot.last_edge, None);
        assert_eq!(snapshot.successors, [None; 4]);
        assert_eq!(budget.slice_remaining, 19);
        assert_eq!(state.pc(), PC.get() + 4);
        assert_eq!(state.general_register_storage_mut()[0], 99);
    }
}

#[test]
fn fault_prefix_samples_once_and_repair_retry_does_not_add_a_sample() {
    for valid in [false, true] {
        // NOP; STR X0,[X1]; BRK. The valid alias repairs code-page protection.
        let mut thread = budget::setup(&[0xd503201f, 0xf9000020, 0xd4200000], true);
        let mut worker = NativeWorker::default();
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[0] = 37;
        state.general_register_storage_mut()[1] = if valid { 0x3800 } else { 0x5000 };
        let (Some(exit), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(1, 20).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected an attributed exit")
        };
        assert!(if valid {
            matches!(exit, Exit::Native { .. })
        } else {
            matches!(
                exit,
                Exit::Memory {
                    outcome: MemoryExit::Fault(_),
                    ..
                }
            )
        });
        let (snapshot, score) = thread
            .samples
            .seed_snapshot(thread.key(PC).unwrap())
            .unwrap();
        assert_eq!((snapshot.sequence, score), (1, 1));
        assert_eq!(snapshot.last_edge, None);
        assert_eq!(snapshot.successors, [None; 4]);
        assert_eq!(budget.slice_remaining, if valid { 18 } else { 19 });
    }
}

#[test]
fn forced_control_discards_heat_but_preserves_sample_phase() {
    let mut thread = budget::setup(&[0xb1000400, 0x14000000], false);
    thread.control.request(ControlRequest::Preempt);
    let mut worker = NativeWorker::default();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let (Some(Exit::Native { returned, .. }), budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            PollBudget::new(1, 20).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected control exit")
    };
    assert_eq!(returned.reason, crate::abi::NativeExitReason::Control);
    assert!(!returned.poll.sample);
    assert_eq!(budget.sample_remaining, 4095);
    assert_eq!(budget.slice_remaining, 18);
    assert!(
        thread
            .samples
            .seed_snapshot(thread.key(PC).unwrap())
            .is_none()
    );
}
