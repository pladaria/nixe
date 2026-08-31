use super::*;

#[test]
fn reference_execution_honors_budget_and_preserves_dispatch_pc() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();

    let report = process.run(1).unwrap();
    assert_eq!(report.progress, 1);
    assert_eq!(report.stop, crate::ExecutionStop::BudgetExhausted);
    assert!(report.stop.exception_dispatch_request().is_none());
    let context = &report.context;
    assert_eq!(context.pc.get(), entry + 0x80);
    assert!(report.to_string().contains("flags=N0Z0C0V0"));
    let state = process.main_thread().state();
    assert_eq!(state.pc(), entry + 0x80);
}

#[test]
fn reference_slices_preserve_instruction_and_supervisor_call_boundaries() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
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
    let state = process.main_thread_mut().state_mut();
    state.write_x(A64Register::General(a64_register(0)), 0);
    let mut cpu_thread = process
        .execution
        .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
        .unwrap();

    let first = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
    assert_eq!(first.progress, 1);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let state = process.main_thread().state();
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 1);
    assert_eq!(state.pc(), entry + 4);

    let second = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
    assert_eq!(second.progress, 1);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);
    let state = process.main_thread().state();
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 3);
    assert_eq!(state.pc(), entry + 8);

    let svc = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
    assert_eq!(svc.progress, 1);
    assert!(matches!(
        svc.stop,
        crate::ExecutionStop::SupervisorCall {
            source,
            immediate: 0x42,
        } if source.pc.get() == entry + 8
    ));
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
    let resumed = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
    assert_eq!(resumed.progress, 1);
    assert_eq!(resumed.stop, crate::ExecutionStop::BudgetExhausted);
    let state = process.main_thread().state();
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
    assert_eq!(first.progress, 2);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let second = process.run(1).unwrap();
    assert_eq!(second.progress, 1);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);

    let state = process.main_thread().state();
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
        let state = process.main_thread_mut().state_mut();
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
    let mut cpu_thread = process
        .execution
        .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
        .unwrap();

    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 1)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let state = process.main_thread_mut().state_mut();
    assert_eq!(state.read_w(A64Register::General(a64_register(0))), 7);
    state.write_x(A64Register::General(a64_register(0)), 9);
    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 1)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let state = process.main_thread().state();
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

    let state = process.main_thread_mut().state_mut();
    state.set_pc(entry);
    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 1)
            .unwrap()
            .stop,
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
    let state = process.main_thread_mut().state_mut();
    state.write_x(A64Register::General(a64_register(0)), 13);
    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 1)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let state = process.main_thread().state();
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
    assert_eq!(report.progress, 0);
    assert_eq!(report.stop, crate::ExecutionStop::Safepoint);
    let state = process.main_thread().state();
    assert_eq!(state.pc(), entry);
}

#[test]
fn reference_execution_reports_instruction_fetch_faults_as_a_distinct_stop() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let state = process.main_thread_mut().state_mut();
    state.set_pc(0x1000);

    let report = process.run(1).unwrap();
    assert_eq!(report.progress, 0);
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::FetchFault { .. }
    ));
    let context = &report.context;
    assert_eq!(context.pc.get(), 0x1000);
    assert!(report.to_string().contains("fetch-fault"));
}

#[test]
fn reference_execution_rejects_unsupported_and_invalid_code() {
    let (_directory, plan) = plan();

    let mut unsupported = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut unsupported, 0xd503_20df); // unimplemented reserved HINT #6
    let crate::ProcessExecutionError::Cpu { fault } = unsupported.run(1).unwrap_err() else {
        panic!("unsupported instruction must be a CPU fault");
    };
    assert_eq!(fault.kind, nixe_cpu::execution::CpuFaultKind::Unavailable);
    assert_eq!(fault.progress, 0);
    assert!(fault.message.contains("unsupported instruction"));

    let mut unallocated = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut unallocated, 0);
    let crate::ProcessExecutionError::Cpu { fault } = unallocated.run(1).unwrap_err() else {
        panic!("invalid instruction must be a CPU fault");
    };
    assert_eq!(
        fault.kind,
        nixe_cpu::execution::CpuFaultKind::InvalidRequest
    );
    assert_eq!(fault.progress, 0);
    assert!(fault.message.contains("invalid instruction stream"));
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
    let state = data_fault.main_thread_mut().state_mut();
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
