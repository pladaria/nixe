use super::*;

#[test]
fn loader_stub_executes_before_runtime_termination() {
    {
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

        let report = process.run_with_cpu_thread(&mut cpu_thread).unwrap();
        assert!(matches!(report.stop,
                crate::ExecutionStop::LoaderReturn { source, result_code }
                    if source.pc == stub && result_code == result));

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
        let report = process.run_with_cpu_thread(&mut cpu_thread).unwrap();
        assert!(matches!(report.stop,
                crate::ExecutionStop::SupervisorCall { source, immediate: observed }
                    if source.pc == pc && observed == immediate));

        assert_eq!(process.main_thread().state().pc(), pc.get());
        assert!(report.context.is_none());
        assert!(process.exit().is_none());
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Running
        );
        process.request_execution_stop().unwrap();

        drop(cpu_thread);
        process.complete_cpu_thread_retirement();
        process.try_teardown().unwrap();
    }
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
    let first = process.run_with_cpu_thread(&mut cpu).unwrap();
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
    let second = process.run_with_cpu_thread(&mut cpu).unwrap();
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
            0xd4000001,  // SVC.
        ],
    );

    let first = process.run().unwrap();

    assert!(matches!(
        first.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));
    let entry = process.entry_module().entry_address();
    process.main_thread_mut().state_mut().set_pc(entry);
    let second = process.run().unwrap();

    assert!(matches!(
        second.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));

    let state = process.main_thread().state();
    assert_eq!(
        state.read_x(A64Register::General(a64_register(1))),
        frequency
    );
    assert_eq!(state.read_x(A64Register::General(a64_register(2))), 0);
    assert_eq!(state.read_x(A64Register::General(a64_register(3))), 0);
}

#[test]
fn reference_execution_observes_safepoints_before_fetch() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();
    process.request_safepoint();

    let report = process.run().unwrap();

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

    let report = process.run().unwrap();

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
    let report = process.run().unwrap();
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
    let report = svc.run().unwrap();
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
    let report = breakpoint.run().unwrap();
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
}
