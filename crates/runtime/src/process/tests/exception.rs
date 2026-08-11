use super::*;

#[test]
fn supervisor_calls_route_a64_a32_and_t32_with_current_runtime_context() {
    let cases = [
        (ExecutionState::A64, 0xd400_4681, 0x234),
        (ExecutionState::A32, 0xef12_3456, 0x12_3456),
        (ExecutionState::T32, 0xbf00_df7b, 0x7b),
    ];

    for (execution_state, encoding, immediate) in cases {
        let (_directory, plan) = plan();
        let mut process = reference_process_builder().build(&plan).unwrap();
        replace_entry_instruction(&mut process, encoding);
        let entry = process.entry_module().entry_address();
        if execution_state != ExecutionState::A64 {
            let mut state = match execution_state {
                ExecutionState::A32 => nixe_cpu::state::A32State::a32(),
                ExecutionState::T32 => nixe_cpu::state::A32State::t32(),
                ExecutionState::A64 => unreachable!(),
            };
            state
                .set_instruction_address(u32::try_from(entry).unwrap())
                .unwrap();
            *process.main_thread_mut().state_mut() = ThreadCpuState::A32(Box::new(state));
        }

        let report = process.run_reference(1).unwrap();
        let expected_encoding = match execution_state {
            ExecutionState::T32 => InstructionEncoding::from_u16(encoding as u16),
            ExecutionState::A64 | ExecutionState::A32 => InstructionEncoding::from_u32(encoding),
        };
        let mut dispatcher = RecordingSupervisorCallDispatcher {
            expected_encoding: Some(expected_encoding),
            observed: None,
        };
        let outcome = process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap();

        assert_eq!(outcome, crate::ExceptionHandlingResult::Suspended);
        let (request, address_space, typed_thread, vcpu, thread_id, handle) =
            dispatcher.observed.unwrap();
        assert_eq!(request.kind(), ExceptionKind::SupervisorCall);
        assert_eq!(request.syndrome(), Some(immediate));
        assert_eq!(request.source().pc.get(), entry);
        assert_eq!(request.source().execution_state, execution_state);
        assert_eq!(address_space, process.cpu_context().address_space_id());
        assert_eq!(thread_id, 1);
        assert_eq!(typed_thread, nixe_scheduler::GuestThreadId::new(1));
        assert_eq!(vcpu, nixe_scheduler::VirtualCpuId::new(0));
        assert_eq!(handle, process.main_thread().handle);
        match process.main_thread().state() {
            ThreadCpuState::A64(state) => assert_eq!(
                state.read_x(nixe_cpu::state::a64::A64Register::General(a64_register(0))),
                0xfeed_face
            ),
            ThreadCpuState::A32(state) => {
                assert_eq!(state.read_r(a32_register(0)), 0xfeed_face)
            }
        }
    }
}

struct IdentityOutcomeDispatcher {
    expected_thread: nixe_scheduler::GuestThreadId,
    expected_vcpu: nixe_scheduler::VirtualCpuId,
    outcome: Option<crate::ExceptionDispatchOutcome<&'static str>>,
}

impl crate::ExceptionDispatcher for IdentityOutcomeDispatcher {
    type Fault = &'static str;

    fn dispatch(
        &mut self,
        context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        assert_eq!(context.thread().id(), self.expected_thread);
        assert_eq!(context.thread().vcpu(), self.expected_vcpu);
        self.outcome.take().unwrap()
    }
}

#[test]
fn every_supervisor_outcome_targets_an_explicit_second_thread() {
    let cases = [
        (
            crate::ExceptionDispatchOutcome::Resume(crate::ExceptionResume::Next),
            nixe_scheduler::ThreadLifecycle::Ready,
        ),
        (
            crate::ExceptionDispatchOutcome::Suspend(crate::ExceptionResume::Retry),
            nixe_scheduler::ThreadLifecycle::Waiting,
        ),
        (
            crate::ExceptionDispatchOutcome::Reject {
                diagnostic: "guest",
            },
            nixe_scheduler::ThreadLifecycle::Ready,
        ),
        (
            crate::ExceptionDispatchOutcome::Terminate {
                scope: crate::ExceptionTerminationScope::CurrentThread,
                exit_code: 9,
                reason: crate::ExceptionTerminationReason::Requested,
            },
            nixe_scheduler::ThreadLifecycle::Exited,
        ),
        (
            crate::ExceptionDispatchOutcome::Fault("host"),
            nixe_scheduler::ThreadLifecycle::Faulted,
        ),
    ];
    for (outcome, expected_lifecycle) in cases {
        let (mut process, report, _) = process_stopped_at_svc(ExecutionState::A64);
        let second_id = nixe_scheduler::GuestThreadId::new(2);
        let mut second = process.main_thread().clone();
        second.id = second_id;
        second.object = crate::ThreadObject::new(second_id.get());
        second.lifecycle = nixe_scheduler::ThreadLifecycle::Waiting;
        process.threads.insert(second).unwrap();
        let vcpu = nixe_scheduler::VirtualCpuId::new(3);
        let mut dispatcher = IdentityOutcomeDispatcher {
            expected_thread: second_id,
            expected_vcpu: vcpu,
            outcome: Some(outcome),
        };
        process
            .route_supervisor_call_for(second_id, vcpu, &report.stop, &mut dispatcher)
            .unwrap();
        assert_eq!(
            process.thread(second_id).unwrap().lifecycle(),
            expected_lifecycle
        );
        if matches!(
            expected_lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
        ) {
            assert_eq!(
                process.lifecycle(),
                nixe_scheduler::ProcessLifecycle::Running
            );
        }
    }
}

#[test]
fn handled_supervisor_calls_advance_once_in_a64_a32_and_t32() {
    let cases = [
        (ExecutionState::A64, 4_u64),
        (ExecutionState::A32, 4_u64),
        (ExecutionState::T32, 2_u64),
    ];

    for (execution_state, width) in cases {
        let (mut process, report, entry) = process_stopped_at_svc(execution_state);
        let mut dispatcher = FixedSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
                crate::ExceptionResume::Next,
            )),
        };

        let result = process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap();

        assert_eq!(result, crate::ExceptionHandlingResult::Resumed);
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
        assert_eq!(
            instruction_address(process.main_thread_mut().state_mut()),
            entry + width
        );
        let next = process.run_reference(1).unwrap();
        assert!(!matches!(
            next.stop,
            crate::ExecutionStop::SupervisorCall { source, .. } if source.pc.get() == entry
        ));
    }
}

#[test]
fn supervisor_call_retry_is_explicit_and_reexecutes_the_source() {
    for execution_state in [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ] {
        let (mut process, report, entry) = process_stopped_at_svc(execution_state);
        let mut dispatcher = PcMutatingSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
                crate::ExceptionResume::Retry,
            )),
        };

        assert_eq!(
            process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap(),
            crate::ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            instruction_address(process.main_thread_mut().state_mut()),
            entry
        );
        let retried = process.run_reference(1).unwrap();
        assert!(matches!(
            retried.stop,
            crate::ExecutionStop::SupervisorCall { source, .. } if source.pc.get() == entry
        ));
    }
}

#[test]
fn suspended_supervisor_call_installs_continuation_without_becoming_runnable() {
    let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
    let mut dispatcher = FixedSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Suspend(
            crate::ExceptionResume::Next,
        )),
    };

    assert_eq!(
        process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Suspended
    );
    assert_eq!(
        instruction_address(process.main_thread_mut().state_mut()),
        entry + 4
    );
    assert_eq!(
        process.execution_status(),
        ProcessExecutionStatus::Suspended
    );
    assert!(matches!(
        process.run_reference(1),
        Err(crate::ProcessExecutionError::NotRunnable {
            status: ProcessExecutionStatus::Suspended,
            ..
        })
    ));
    assert!(process.resume());
    assert_eq!(
        instruction_address(process.main_thread_mut().state_mut()),
        entry + 4
    );
}

#[test]
fn faulted_supervisor_call_retains_source_and_cannot_run() {
    let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
    let mut dispatcher = PcMutatingSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::Fault(
            "svc dispatch failed",
        )),
    };

    assert_eq!(
        process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Fault("svc dispatch failed")
    );
    assert_eq!(
        instruction_address(process.main_thread_mut().state_mut()),
        entry
    );
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Faulted);
    assert!(matches!(
        process.run_reference(1),
        Err(crate::ProcessExecutionError::NotRunnable {
            status: ProcessExecutionStatus::Faulted,
            ..
        })
    ));
    assert!(!process.resume());
}

#[test]
fn supervisor_call_termination_scope_is_preserved_through_teardown() {
    let cases = [
        (
            crate::ExceptionTerminationScope::CurrentThread,
            crate::ProcessExitCause::LastThreadExited,
        ),
        (
            crate::ExceptionTerminationScope::Process,
            crate::ProcessExitCause::ProcessRequested,
        ),
    ];

    for (scope, expected_cause) in cases {
        let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
        let mut dispatcher = FixedSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Terminate {
                scope,
                exit_code: 0x55,
                reason: crate::ExceptionTerminationReason::Requested,
            }),
        };

        assert_eq!(
            process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap(),
            crate::ExceptionHandlingResult::Terminated {
                scope,
                exit_code: 0x55,
                reason: crate::ExceptionTerminationReason::Requested,
            }
        );
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Exited);
        assert_eq!(process.exit().unwrap().cause, expected_cause);
        assert_eq!(process.exit().unwrap().source.unwrap().pc.get(), entry);
        assert_eq!(process.main_thread().exit().unwrap().requested_scope, scope);
        assert!(matches!(
            process.run_reference(1),
            Err(crate::ProcessExecutionError::NotRunnable {
                status: ProcessExecutionStatus::Exited,
                ..
            })
        ));

        let teardown = process.teardown();
        assert_eq!(teardown.previous_status, ProcessExecutionStatus::Exited);
        assert_eq!(teardown.exit.unwrap().cause, expected_cause);
        assert_eq!(teardown.threads_released, 1);
    }
}

#[test]
fn detailed_instruction_trace_is_opt_in_bounded_and_persistent_across_slices() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_diagnostics(crate::DiagnosticsPolicy {
            instruction_trace: true,
            ..crate::DiagnosticsPolicy::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&mut process, 0x1400_0000); // B #0

    let first = process
        .run_reference(crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 3)
        .unwrap();
    assert!(first.trace.enabled());
    assert_eq!(
        first.trace.entries().len(),
        crate::MAX_INSTRUCTION_TRACE_ENTRIES
    );
    assert_eq!(first.trace.discarded(), 3);
    assert_eq!(first.trace.entries()[0].sequence, 3);
    assert_eq!(
        first.trace.entries().last().unwrap().sequence,
        crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 2
    );
    assert!(
        first
            .trace
            .entries()
            .iter()
            .all(|entry| entry.disassembly.as_deref() == Some("b imm=#0"))
    );
    assert!(first.trace.to_string().len() <= crate::MAX_INSTRUCTION_TRACE_EXPORT_BYTES);

    let second = process.run_reference(1).unwrap();
    assert_eq!(second.trace.discarded(), 4);
    assert_eq!(second.trace.entries()[0].sequence, 4);
    assert_eq!(
        second.trace.entries().last().unwrap().sequence,
        crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 3
    );
}

#[test]
fn sanitized_instruction_trace_omits_detailed_disassembly() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_diagnostics(crate::DiagnosticsPolicy {
            report_detail: crate::ReportDetail::Sanitized,
            instruction_trace: true,
            ..crate::DiagnosticsPolicy::default()
        })
        .build(&plan)
        .unwrap();

    let report = process.run_reference(1).unwrap();
    assert_eq!(report.trace.entries().len(), 1);
    assert!(report.trace.entries()[0].disassembly.is_none());
    assert!(!report.trace.to_string().contains("disassembly="));
}

#[test]
fn teardown_reports_resources_owned_by_the_process() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    assert!(process.terminate());
    assert_eq!(
        process.exit().unwrap().cause,
        crate::ProcessExitCause::HostRequested
    );
    assert_eq!(
        process.main_thread().exit().unwrap().requested_scope,
        crate::ExceptionTerminationScope::Process
    );

    let report = process.teardown();
    assert_eq!(
        report.previous_status,
        crate::ProcessExecutionStatus::Exited
    );
    assert_eq!(
        report.exit.unwrap().cause,
        crate::ProcessExitCause::HostRequested
    );
    assert_eq!(report.threads_released, 1);
    assert_eq!(report.modules_released, 1);
    assert!(report.mappings_released > 0);
    assert!(report.physical_pages_released > 0);
    assert_eq!(report.mounts_released, 0);
    assert_eq!(report.handles_released, 1);
}
