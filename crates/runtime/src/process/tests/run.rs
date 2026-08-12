use super::*;

#[test]
fn reference_execution_honors_budget_and_preserves_dispatch_pc() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();

    let report = process.run(1).unwrap();
    assert_eq!(report.instructions_executed, 1);
    assert_eq!(report.stop, crate::ExecutionStop::BudgetExhausted);
    assert!(report.stop.exception_dispatch_request().is_none());
    assert!(!report.trace.enabled());
    assert!(report.trace.entries().is_empty());
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Ready
    );
    let nixe_cpu::state::RegisterContext::A64(context) = &report.context else {
        panic!("homebrew fixture must report A64 context");
    };
    assert_eq!(context.pc.get(), entry + 0x80);
    assert!(report.to_string().contains("flags=N0Z0C0V0"));
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        panic!("homebrew fixture must initialize A64");
    };
    assert_eq!(state.pc(), entry + 0x80);
}

#[test]
fn reference_slices_preserve_instruction_and_supervisor_call_boundaries() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_diagnostics(crate::DiagnosticsPolicy {
            instruction_trace: true,
            ..crate::DiagnosticsPolicy::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0x9100_0400, // ADD X0,X0,#1
            0x9100_0800, // ADD X0,X0,#2
            0xd400_0841, // SVC #0x42
            0x9100_1000, // ADD X0,X0,#4
        ],
    );
    let entry = process.entry_module().entry_address();
    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        panic!("homebrew fixture must initialize A64");
    };
    state.write_x(A64Register::General(a64_register(0)), 0);

    let first = process.run(1).unwrap();
    assert_eq!(first.instructions_executed, 1);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 1);
    assert_eq!(state.pc(), entry + 4);

    let second = process.run(1).unwrap();
    assert_eq!(second.instructions_executed, 1);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 3);
    assert_eq!(state.pc(), entry + 8);

    let svc = process.run(1).unwrap();
    assert_eq!(svc.instructions_executed, 1);
    assert!(matches!(
        svc.stop,
        crate::ExecutionStop::SupervisorCall {
            source,
            immediate: 0x42,
        } if source.pc.get() == entry + 8
    ));
    let sources = svc
        .trace
        .entries()
        .iter()
        .map(|entry| entry.source.pc.get())
        .collect::<Vec<_>>();
    assert_eq!(sources, [entry, entry + 4, entry + 8]);

    let mut dispatcher = FixedSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
            crate::ExceptionResume::Next,
        )),
    };
    assert_eq!(
        process
            .route_supervisor_call(&svc.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Resumed
    );
    let resumed = process.run(1).unwrap();
    assert_eq!(resumed.instructions_executed, 1);
    assert_eq!(resumed.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 7);
    assert_eq!(state.pc(), entry + 16);
}
#[test]
fn fixed_virtual_timer_is_stable_across_reference_slices() {
    let (_directory, plan) = plan();
    let frequency = 24_000_000;
    let mut process = reference_process_builder()
        .with_config(ProcessBuildConfig {
            architectural_timer_frequency: frequency,
            ..ProcessBuildConfig::default()
        })
        .with_virtual_clock(crate::VirtualClock::new(crate::VirtualClockMode::Fixed {
            unix_seconds: 1_700_000_000,
        }))
        .build(&plan)
        .unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0xd53b_e001, // MRS X1,CNTFRQ_EL0
            0xd53b_e022, // MRS X2,CNTVCT_EL0
            0xd53b_e023, // MRS X3,CNTVCT_EL0
        ],
    );

    let first = process.run(2).unwrap();
    assert_eq!(first.instructions_executed, 2);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let second = process.run(1).unwrap();
    assert_eq!(second.instructions_executed, 1);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);

    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(
        state.read_x(A64Register::General(a64_register(1))),
        frequency
    );
    assert_eq!(state.read_x(A64Register::General(a64_register(2))), 0);
    assert_eq!(state.read_x(A64Register::General(a64_register(3))), 0);
}

#[test]
fn exclusive_monitor_persists_and_observes_generation_changes_across_slices() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0x885f_fc60, // LDAXR W0,[X3]
            0x8801_fc60, // STLXR W1,W0,[X3]
        ],
    );
    let entry = process.entry_module().entry_address();
    let data = {
        let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
            panic!("homebrew fixture must initialize A64");
        };
        let address = GuestVirtualAddress::new(
            state
                .read_x(A64Register::StackPointer)
                .checked_sub(8)
                .unwrap(),
        );
        state.write_x(A64Register::General(a64_register(3)), address.get());
        address
    };
    let address_space = process.cpu_context().address_space_id();
    process
        .memory()
        .write(
            address_space,
            data,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(7),
        )
        .unwrap();

    assert_eq!(
        process.run(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        unreachable!();
    };
    assert_eq!(state.read_w(A64Register::General(a64_register(0))), 7);
    state.write_x(A64Register::General(a64_register(0)), 9);
    assert_eq!(
        process.run(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(state.read_w(A64Register::General(a64_register(1))), 0);
    assert_eq!(
        process
            .memory()
            .read(
                address_space,
                data,
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(9)
    );

    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        unreachable!();
    };
    state.set_pc(entry);
    assert_eq!(
        process.run(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    process
        .memory()
        .write(
            address_space,
            data,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(11),
        )
        .unwrap();
    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        unreachable!();
    };
    state.write_x(A64Register::General(a64_register(0)), 13);
    assert_eq!(
        process.run(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(state.read_w(A64Register::General(a64_register(1))), 1);
    assert_eq!(
        process
            .memory()
            .read(
                address_space,
                data,
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(11)
    );
}

#[test]
fn reference_execution_observes_safepoints_before_fetch() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();
    process.request_safepoint();

    let report = process.run(10).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert_eq!(report.stop, crate::ExecutionStop::Safepoint);
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        panic!("homebrew fixture must initialize A64");
    };
    assert_eq!(state.pc(), entry);
}

#[test]
fn reference_execution_observes_pending_events_before_fetch() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();
    process.post_event(0b0001);
    process.post_event(0b0100);

    let report = process.run(10).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert_eq!(
        report.stop,
        crate::ExecutionStop::PendingEvent { mask: 0b0101 }
    );
    let next = process.run(1).unwrap();
    assert_eq!(next.instructions_executed, 1);
    assert_eq!(next.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
        unreachable!();
    };
    assert_eq!(state.pc(), entry + 0x80);
}

#[test]
fn reference_execution_reports_instruction_fetch_faults_as_a_distinct_stop() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        panic!("homebrew fixture must initialize A64");
    };
    state.set_pc(0x1000);

    let report = process.run(1).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::FetchFault { .. }
    ));
    let nixe_cpu::state::RegisterContext::A64(context) = &report.context else {
        panic!("homebrew fixture must report A64 context");
    };
    assert_eq!(context.pc.get(), 0x1000);
    assert!(report.to_string().contains("fetch-fault"));
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Faulted
    );
}

#[test]
fn unallocated_encoding_suspends_until_runtime_resumes_thread() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();

    let report = process.run(2).unwrap();
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::UnallocatedEncoding { .. }
    ));
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Suspended
    );
    assert!(matches!(
        process.run(1),
        Err(crate::ProcessExecutionError::NotRunnable {
            status: crate::ProcessExecutionStatus::Suspended,
            ..
        })
    ));
    assert!(process.resume_thread(process.main_thread_id()));
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Ready
    );
}

#[test]
fn reference_execution_distinguishes_unsupported_profile_and_unallocated_code() {
    let (_directory, plan) = plan();

    let mut unsupported = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut unsupported, 0xd503_205f); // WFE
    let report = unsupported.run(1).unwrap();
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::UnsupportedSemantics { .. }
    ));
    assert_eq!(
        unsupported.execution_status(),
        ProcessExecutionStatus::Faulted
    );
    assert!(report.to_string().contains("unsupported-semantics"));

    let mut profile_disabled = reference_process_builder()
        .with_config(ProcessBuildConfig {
            cpu_profile: GuestCpuProfile::switch_2_native(),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&mut profile_disabled, 0x4e22_1c20);
    let report = profile_disabled.run(1).unwrap();
    assert_eq!(
        report.stop.exception_dispatch_request().unwrap().kind(),
        nixe_cpu::exception::ExceptionKind::UndefinedInstruction
    );
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::ProfileDisabled { .. }
    ));
    assert!(report.to_string().contains("profile-disabled"));

    let mut unallocated = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut unallocated, 0);
    let report = unallocated.run(1).unwrap();
    assert_eq!(
        report.stop.exception_dispatch_request().unwrap().kind(),
        nixe_cpu::exception::ExceptionKind::UndefinedInstruction
    );
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::UnallocatedEncoding { .. }
    ));
    assert!(report.to_string().contains("unallocated-encoding"));
}

#[test]
fn reference_execution_distinguishes_svc_architectural_and_data_fault_stops() {
    let (_directory, plan) = plan();

    let mut svc = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut svc, 0xd400_0841); // SVC #0x42
    let report = svc.run(1).unwrap();
    let dispatch = report.stop.exception_dispatch_request().unwrap();
    assert_eq!(
        dispatch.kind(),
        nixe_cpu::exception::ExceptionKind::SupervisorCall
    );
    assert_eq!(dispatch.syndrome(), Some(0x42));
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::SupervisorCall {
            immediate: 0x42,
            ..
        }
    ));
    assert!(report.to_string().contains("supervisor-call"));

    let mut breakpoint = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut breakpoint, 0xd420_2460); // BRK #0x123
    let report = breakpoint.run(1).unwrap();
    let dispatch = report.stop.exception_dispatch_request().unwrap();
    assert_eq!(
        dispatch.kind(),
        nixe_cpu::exception::ExceptionKind::Breakpoint
    );
    assert_eq!(dispatch.syndrome(), Some(0x123));
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::ArchitecturalException {
            kind: nixe_cpu::exception::ExceptionKind::Breakpoint,
            syndrome: Some(0x123),
            ..
        }
    ));
    assert!(report.to_string().contains("architectural-exception"));

    let mut data_fault = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut data_fault, 0xf940_0020); // LDR X0,[X1]
    let ThreadCpuState::A64(state) = data_fault.main_thread_mut().state_mut() else {
        panic!("homebrew fixture must initialize A64");
    };
    state.write_x(
        nixe_cpu::state::a64::A64Register::General(a64_register(1)),
        0x1000,
    );
    let report = data_fault.run(1).unwrap();
    assert_eq!(
        report.stop.exception_dispatch_request().unwrap().kind(),
        nixe_cpu::exception::ExceptionKind::DataAbort
    );
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::DataFault { .. }
    ));
    assert!(report.to_string().contains("data-fault"));
}
