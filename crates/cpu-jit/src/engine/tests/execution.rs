use super::*;
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuExit, CpuFaultKind, TimerSnapshot, VcpuEventState,
};
use nixe_memory::MemoryInvalidationSource;

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 17,
            frequency: 19,
        }
    }
}

fn state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(PC.get());
    state
}

#[test]
fn recognized_unsupported_catalog_preserves_identity_and_pre_state_on_both_platforms() {
    use nixe_cpu::decode::{DecodeSupport, a64};

    let mut worker = NativeWorker::default();
    for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
        for pattern in a64::patterns()
            .iter()
            .filter(|pattern| pattern.decoder == DecodeSupport::RecognizedUnimplemented)
        {
            let bits = pattern
                .regression_fixture
                .expect("unsupported fixture")
                .encoding
                .bits();
            let memory = memory(DirectBackendPolicy::Required);
            memory
                .overwrite_mapped_ram(SPACE, PC, &bits.to_le_bytes())
                .unwrap();
            let process =
                Arc::new(JitProcess::new(ProcessCpuContext::new(platform, SPACE), memory).unwrap());
            let mut thread = JitThread::new(process.clone()).unwrap();
            let mut state = state();
            state
                .general_register_storage_mut()
                .fill(0x1234_5678_9abc_def0);
            let before = state.clone();
            let report = thread
                .run_slice(
                    &mut worker,
                    &mut state,
                    100,
                    None,
                    &Timer,
                    &VcpuEventState::default(),
                )
                .unwrap();
            assert!(
                matches!(report.stop,
                    CpuExit::UnsupportedSemantics { coverage_id, source, .. }
                        if coverage_id == pattern.coverage_id && source.pc == PC
                ),
                "{platform:?}: {}: {:?}",
                pattern.name,
                report.stop
            );
            assert_eq!(report.progress, 0);
            assert_eq!(state, before);
            drop(thread);
            assert!(process.try_shutdown().unwrap());
        }
    }
}

#[test]
fn empty_slice_control_and_events_stop_before_native_registration_or_demand() {
    let previous_stack = worker::signal_stack();
    let mut worker = NativeWorker::default();
    let process = Arc::new(JitProcess::new(cpu(), memory(DirectBackendPolicy::Required)).unwrap());
    let mut thread = JitThread::new(process).unwrap();
    let mut state = state();
    let before = state.clone();
    let events = VcpuEventState::default();
    let control = thread.control();
    control.request(ControlRequest::Preempt);
    events.post_interrupts(5);
    for (limit, expected) in [
        (0, CpuExit::BudgetExhausted),
        (10, CpuExit::Safepoint),
        (10, CpuExit::PendingEvent { mask: 5 }),
    ] {
        let report = thread
            .run_slice(&mut worker, &mut state, limit, None, &Timer, &events)
            .unwrap();
        assert_eq!(report.stop, expected);
        assert_eq!(report.progress, 0);
        assert_eq!(report.context, Some(state.register_context()));
        assert_eq!(state, before);
        assert_eq!(worker::signal_stack(), previous_stack);
    }
    let error = thread
        .run_slice(&mut worker, &mut state, u64::MAX, None, &Timer, &events)
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::InvalidRequest);
    assert_eq!(error.progress, 0);
    assert_eq!(worker::signal_stack(), previous_stack);
}

#[test]
fn slice_demands_call_return_and_system_continuations_then_recognizes_loader_return() {
    let mut worker = NativeWorker::default();
    // BL +16; B 0x2000; padding; MRS X0,CNTPCT_EL0; ADD X0,X0,#1; RET.
    let mut thread = budget::setup(
        &[
            0x94000004, 0x140003ff, 0, 0, 0xd53be020, 0x91000400, 0xd65f03c0,
        ],
        false,
    );
    let mut state = state();
    let report = thread
        .run_slice(
            &mut worker,
            &mut state,
            100,
            Some(GuestVirtualAddress::new(0x2000)),
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(
        matches!(report.stop, CpuExit::LoaderReturn { source, result_code: 18 } if source.pc.get() == 0x2000)
    );
    assert_eq!(report.progress, 5);
    assert_eq!(state.general_register_storage_mut()[30], PC.get() + 4);
    assert_eq!(thread.sample_remaining, 4091);
    assert_eq!(report.context, Some(state.register_context()));
}

#[test]
fn backedges_obey_small_slices_and_preserve_sample_phase_between_runs() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0x91000400, 0x17ffffff], false); // ADD X0,X0,#1; B -4.
    thread.sample_remaining = 1;
    let mut state = state();
    for (limit, progress, value, phase) in [(3, 4, 2, 4093), (1, 2, 3, 4091)] {
        let report = thread
            .run_slice(
                &mut worker,
                &mut state,
                limit,
                None,
                &Timer,
                &VcpuEventState::default(),
            )
            .unwrap();
        assert_eq!(report.stop, CpuExit::BudgetExhausted);
        assert_eq!(report.progress, progress);
        assert_eq!(state.general_register_storage_mut()[0], value);
        assert_eq!(state.pc(), PC.get());
        assert_eq!(thread.sample_remaining, phase);
    }
}

#[test]
fn exhausted_prefix_finishes_only_its_cold_instruction_before_yielding() {
    let mut worker = NativeWorker::default();
    // NOP; MRS X0,CNTPCT_EL0; ADD X1,X1,#1; BRK #9.
    let mut thread = budget::setup(&[0xd503201f, 0xd53be020, 0x91000421, 0xd4200120], false);
    let mut state = state();
    let events = VcpuEventState::default();
    let report = thread
        .run_slice(&mut worker, &mut state, 1, None, &Timer, &events)
        .unwrap();
    assert_eq!(report.stop, CpuExit::BudgetExhausted);
    assert_eq!(report.progress, 2);
    assert_eq!(state.pc(), PC.get() + 8);
    assert_eq!(state.general_register_storage_mut()[0], 17);
    assert_eq!(state.general_register_storage_mut()[1], 0);
    let report = thread
        .run_slice(&mut worker, &mut state, 1, None, &Timer, &events)
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            syndrome: Some(9),
            ..
        }
    ));
    assert_eq!(report.progress, 2);
    assert_eq!(state.general_register_storage_mut()[1], 1);
    assert_eq!(state.pc(), PC.get() + 12);
}

#[test]
fn control_posted_by_completion_prevents_the_next_native_entry_and_preserves_event() {
    let mut worker = NativeWorker::default();
    struct PreemptingTimer {
        control: CpuControl,
        events: VcpuEventState,
    }
    impl ArchitecturalTimer for PreemptingTimer {
        fn snapshot(&self) -> TimerSnapshot {
            self.control.request(ControlRequest::Preempt);
            self.events.post_interrupts(2);
            Timer.snapshot()
        }
    }
    let mut thread = budget::setup(&[0xd53be020, 0x91000421, 0xd4200000], false);
    let mut state = state();
    let events = VcpuEventState::default();
    let timer = PreemptingTimer {
        control: thread.control(),
        events: events.clone(),
    };
    for (expected, progress) in [
        (CpuExit::Safepoint, 1),
        (CpuExit::PendingEvent { mask: 2 }, 0),
    ] {
        let report = thread
            .run_slice(&mut worker, &mut state, 100, None, &timer, &events)
            .unwrap();
        assert_eq!(report.stop, expected);
        assert_eq!(report.progress, progress);
        assert_eq!(state.pc(), PC.get() + 4);
        assert_eq!(state.general_register_storage_mut()[0], 17);
        assert_eq!(state.general_register_storage_mut()[1], 0);
    }
}

#[test]
fn slice_reports_precise_fetch_and_data_faults_with_completed_prefix_only() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0xd503201f, 0xf9400020], false); // NOP; LDR X0,[X1].
    let events = VcpuEventState::default();
    let mut state = state();
    state.general_register_storage_mut()[1] = 0x5000;
    let report = thread
        .run_slice(&mut worker, &mut state, 100, None, &Timer, &events)
        .unwrap();
    assert!(
        matches!(report.stop, CpuExit::DataFault { source, fault } if source.pc.get() == PC.get() + 4 && fault.address.get() == 0x5000)
    );
    assert_eq!(report.progress, 1);
    assert_eq!(state.pc(), PC.get() + 4);
    assert_eq!(report.context, Some(state.register_context()));
    for (pc, reason) in [
        (0x2000, InstructionFetchFaultReason::Unmapped),
        (0x1001, InstructionFetchFaultReason::Misaligned),
    ] {
        state.set_pc(pc);
        let report = thread
            .run_slice(&mut worker, &mut state, 100, None, &Timer, &events)
            .unwrap();
        assert!(
            matches!(report.stop, CpuExit::FetchFault { fault } if fault.address.get() == pc && fault.reason == reason)
        );
        assert_eq!(report.progress, 0);
        assert_eq!(state.pc(), pc);
    }
}

#[test]
fn invalidation_notification_acknowledges_canonical_state_then_demands_replaced_code() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0x14000000], false); // B .
    thread
        .process
        .memory
        .overwrite_mapped_ram(SPACE, PC, &0xd4200120_u32.to_le_bytes())
        .unwrap();
    // The memory authority already closed admission and unlinked the old loop.
    // A CPU notification alone must never be used to make that write safe.
    let control = thread.control();
    let cursor = thread.process.memory.invalidation_cursor();
    control.request_invalidation(cursor.get());
    let mut state = state();
    let report = thread
        .run_slice(
            &mut worker,
            &mut state,
            10,
            None,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            syndrome: Some(9),
            ..
        }
    ));
    assert_eq!(report.progress, 1);
    assert!(control.acknowledged_invalidation(cursor.get()));
}

#[test]
fn closed_admission_yields_without_acknowledging_maintenance_and_shutdown_is_terminal() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0x14000000], false); // B .
    let process = thread.process.clone();
    let ticket = process
        .lifetime
        .request(lifetime::Reason::MappingChange)
        .unwrap();
    let mut state = state();
    let before = state.clone();
    let events = VcpuEventState::default();
    let report = thread
        .run_slice(&mut worker, &mut state, 1, None, &Timer, &events)
        .unwrap();
    assert_eq!(report.stop, CpuExit::Safepoint);
    assert_eq!(report.progress, 0);
    assert_eq!(state, before);
    assert!(!ticket.is_complete().unwrap());
    process
        .lifetime
        .request(lifetime::Reason::Shutdown)
        .unwrap();
    let error = thread
        .run_slice(&mut worker, &mut state, 1, None, &Timer, &events)
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::Unavailable);
    assert_eq!(error.progress, 0);
    assert_eq!(state, before);
}
