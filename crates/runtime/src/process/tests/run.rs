use super::*;

#[test]
fn loader_stub_executes_before_runtime_termination() {
    for split in [false, true] {
        let (_directory, plan) = plan();
        let mut process = reference_process_builder().build(&plan).unwrap();
        let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
        let stub = process.main_thread().loader_return.unwrap();
        write_code(&process, entry, &0xd65f03c0_u32.to_le_bytes()); // RET X30 to the real mapped loader stub.
        let result = 0x8123_4567_89ab_cdef;
        process
            .main_thread_mut()
            .state_mut()
            .write_x(A64Register::General(a64_register(0)), result);
        let mut cpu_thread = process
            .execution
            .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
            .unwrap();

        if split {
            let returned = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
            assert_eq!(returned.stop, crate::ExecutionStop::BudgetExhausted);
            assert_eq!(returned.progress, 1);
            assert_eq!(process.main_thread().state().pc(), stub.get());
            assert!(process.exit().is_none());
            let zero = process.run_with_cpu_thread(&mut cpu_thread, 0).unwrap();
            assert_eq!(zero.stop, crate::ExecutionStop::BudgetExhausted);
            assert_eq!(zero.progress, 0);
            cpu_thread
                .control()
                .request(nixe_cpu::execution::ControlRequest::Preempt);
            let paused = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
            assert_eq!(paused.stop, crate::ExecutionStop::Safepoint);
            assert_eq!(paused.progress, 0);
            assert_eq!(process.main_thread().state().pc(), stub.get());
            assert!(process.exit().is_none());
        }
        let report = process.run_with_cpu_thread(&mut cpu_thread, 16).unwrap();
        assert!(matches!(report.stop,
                crate::ExecutionStop::LoaderReturn { source, result_code }
                    if source.pc == stub && result_code == result));
        assert_eq!(report.progress, if split { 1 } else { 2 }); // RET, then SVC.
        assert_eq!(process.main_thread().state().pc(), stub.get());
        assert_eq!(
            report.context,
            Some(process.main_thread().state().register_context())
        );
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Exited
        );
        let exit = process.exit().unwrap();
        assert_eq!(exit.cause, ProcessExitCause::LoaderReturned);
        assert_eq!(exit.exit_code, result);
        assert_eq!(exit.source.unwrap().pc, stub);
        assert_eq!(exit.context.as_deref(), report.context.as_ref());
        assert_eq!(
            process.main_thread().exit.as_ref().unwrap().exit_code,
            result
        );

        process.request_execution_stop().unwrap();
        cpu_thread.clear_local_exclusive_reservation();
        drop(cpu_thread);
        process.complete_cpu_thread_retirement();
        process.try_teardown().unwrap();
    }
}

#[test]
fn loader_stub_recognition_requires_the_thread_address_and_exit_svc() {
    // Same SVC elsewhere, same stub without this thread's registration,
    // and another SVC at the registered address must remain ordinary calls.
    for case in 0..3 {
        let (_directory, plan) = plan();
        let mut process = reference_process_builder().build(&plan).unwrap();
        let stub = process.main_thread().loader_return.unwrap();
        let pc = if case == 0 {
            GuestVirtualAddress::new(process.entry_module().entry_address())
        } else {
            stub
        };
        let (instruction, immediate) = if case == 2 {
            (0xd4000841_u32, 0x42)
        } else {
            (HOME_BREW_EXIT_PROCESS_INSTRUCTION, 7)
        };
        write_code(&process, pc, &instruction.to_le_bytes());
        if case == 1 {
            process.main_thread_mut().loader_return = None;
        }
        process.main_thread_mut().state_mut().set_pc(pc.get());
        let mut cpu_thread = process
            .execution
            .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
            .unwrap();
        let report = process.run_with_cpu_thread(&mut cpu_thread, 1).unwrap();
        assert!(matches!(report.stop,
                crate::ExecutionStop::SupervisorCall { source, immediate: observed }
                    if source.pc == pc && observed == immediate));
        assert_eq!(report.progress, 1);
        assert_eq!(process.main_thread().state().pc(), pc.get());
        assert_eq!(
            report.context,
            Some(process.main_thread().state().register_context())
        );
        assert!(process.exit().is_none());
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Running
        );
        process.request_execution_stop().unwrap();
        cpu_thread.clear_local_exclusive_reservation();
        drop(cpu_thread);
        process.complete_cpu_thread_retirement();
        process.try_teardown().unwrap();
    }
}

#[test]
fn production_jit_uses_block_budgets_recompiles_mutated_code_and_retires() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    let mut words = vec![0x91000400_u32; 17]; // ADD X0,X0,#1.
    words.push(0x17ffffef); // B entry: -17 instructions.
    write_code(
        &process,
        entry,
        &words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    process
        .main_thread_mut()
        .state_mut()
        .write_x(A64Register::General(a64_register(0)), 0);
    let mut thread = process
        .execution
        .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
        .unwrap();
    let zero = process.run_with_cpu_thread(&mut thread, 0).unwrap();
    assert_eq!(zero.progress, 0);
    assert_eq!(zero.stop, crate::ExecutionStop::BudgetExhausted);
    let first = process.run_with_cpu_thread(&mut thread, 1).unwrap();
    assert_eq!(first.progress, 18);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    assert_eq!(process.main_thread().state().pc(), entry.get());
    assert_eq!(
        process
            .main_thread()
            .state()
            .read_x(A64Register::General(a64_register(0))),
        17
    );

    write_code(&process, entry, &0x91002400_u32.to_le_bytes()); // ADD #9.
    let second = process.run_with_cpu_thread(&mut thread, 1).unwrap();
    assert_eq!(second.progress, 18);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);
    assert_eq!(
        process
            .main_thread()
            .state()
            .read_x(A64Register::General(a64_register(0))),
        42
    );
    process.request_execution_stop().unwrap();
    thread.clear_local_exclusive_reservation();
    drop(thread);
    process.complete_cpu_thread_retirement();
    process.try_teardown().unwrap();
}

#[test]
fn reference_execution_honors_budget_and_preserves_dispatch_pc() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();

    let report = process.run(1).unwrap();
    assert_eq!(report.progress, 1);
    assert_eq!(report.stop, crate::ExecutionStop::BudgetExhausted);
    assert!(report.stop.exception_dispatch_request().is_none());
    let context = report.context.as_ref().unwrap();
    assert_eq!(context.pc.get(), entry + 0x80);
    assert!(report.to_string().contains("flags=N0Z0C0V0"));
    let state = process.main_thread().state();
    assert_eq!(state.pc(), entry + 0x80);
}

#[test]
fn native_slices_preserve_supervisor_call_boundaries_and_apply_the_continuation_once() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0x91000400, // ADD X0,#1.
            0x91000800, // ADD X0,#2.
            0xd4000841, // SVC #0x42.
            0x91001000, // ADD X0,#4.
            0xd4000861, // SVC #0x43.
        ],
    );
    let entry = process.entry_module().entry_address();
    process
        .main_thread_mut()
        .state_mut()
        .write_x(A64Register::General(a64_register(0)), 0);
    let mut cpu = process
        .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
        .unwrap();
    let first = process.run_with_cpu_thread(&mut cpu, 100).unwrap();
    assert!(
        matches!(first.stop, crate::ExecutionStop::SupervisorCall { source, immediate: 0x42 }
        if source.pc.get() == entry + 8)
    );
    assert_eq!(
        process
            .main_thread()
            .state()
            .read_x(A64Register::General(a64_register(0))),
        3
    );
    let mut dispatcher = FixedSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
            crate::ExceptionResume::Next,
        )),
    };
    assert_eq!(
        process
            .route_supervisor_call(&first.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Resumed
    );
    let second = process.run_with_cpu_thread(&mut cpu, 100).unwrap();
    assert!(
        matches!(second.stop, crate::ExecutionStop::SupervisorCall { source, immediate: 0x43 }
        if source.pc.get() == entry + 16)
    );
    assert_eq!(
        process
            .main_thread()
            .state()
            .read_x(A64Register::General(a64_register(0))),
        7
    );
}
#[test]
fn fixed_virtual_timer_is_stable_across_native_slices() {
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
            0x17fffffd,  // B to the first timer read.
        ],
    );

    let first = process.run(4).unwrap();
    assert_eq!(first.progress, 4);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let second = process.run(4).unwrap();
    assert_eq!(second.progress, 4);
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
            0x14000001,  // End the load block without clearing the monitor.
            0x8801_fc60, // STLXR W1,W0,[X3]
            0x17fffffd,  // End the store block and return to the load.
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
    assert!(matches!(
        process.memory().space().write_value(data, 4, 7).unwrap(),
        nixe_memory::Access::Ready(())
    ));
    let mut cpu_thread = process
        .execution
        .create_worker_cpu_thread(nixe_scheduler::VirtualCpuId::new(0))
        .unwrap();

    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 2)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let state = process.main_thread_mut().state_mut();
    assert_eq!(state.read_w(A64Register::General(a64_register(0))), 7);
    state.write_x(A64Register::General(a64_register(0)), 9);
    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 2)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let state = process.main_thread().state();
    assert_eq!(state.read_w(A64Register::General(a64_register(1))), 0);
    assert_eq!(
        process
            .memory()
            .space()
            .read_value(data, 4)
            .unwrap()
            .into_ready()
            .unwrap(),
        9
    );

    let state = process.main_thread_mut().state_mut();
    state.set_pc(entry);
    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 2)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    assert!(matches!(
        process.memory().space().write_value(data, 4, 11).unwrap(),
        nixe_memory::Access::Ready(())
    ));
    let state = process.main_thread_mut().state_mut();
    state.write_x(A64Register::General(a64_register(0)), 13);
    assert_eq!(
        process
            .run_with_cpu_thread(&mut cpu_thread, 2)
            .unwrap()
            .stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let state = process.main_thread().state();
    assert_eq!(state.read_w(A64Register::General(a64_register(1))), 1);
    assert_eq!(
        process
            .memory()
            .space()
            .read_value(data, 4)
            .unwrap()
            .into_ready()
            .unwrap(),
        11
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
    let context = report.context.as_ref().unwrap();
    assert_eq!(context.pc.get(), 0x1000);
    assert!(report.to_string().contains("fetch-fault"));
}

#[test]
fn native_execution_reports_an_unallocated_instruction_with_its_original_source() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut process, 0);
    let entry = process.entry_module().entry_address();
    let report = process.run(100).unwrap();
    let crate::ExecutionStop::UnallocatedEncoding { error } = report.stop else {
        panic!("unallocated encoding must stop execution")
    };
    assert_eq!(error.instruction.location.pc.get(), entry);
    assert_eq!(error.instruction.encoding.bits(), 0);
    assert_eq!(process.main_thread().state().pc(), entry);
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
