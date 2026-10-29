use super::*;

pub(super) fn scheduled(
    process: RunnableProcess,
) -> (
    crate::RuntimeCoordinator,
    nixe_scheduler::ProcessId,
    Arc<RunnableProcess>,
) {
    let core = nixe_scheduler::VirtualCpuId::new(0);
    let mut coordinator = crate::RuntimeCoordinator::new(process.scheduler().clone());
    let registration = crate::ProcessRegistration {
        priority: 44,
        ideal_vcpu: Some(core),
        affinity: coordinator.scheduler().profile().all_cores(),
    };
    let id = coordinator.register_process(process, registration).unwrap();
    let process = coordinator.process(id).unwrap().clone();
    (coordinator, id, process)
}
fn execute(coordinator: &mut crate::RuntimeCoordinator) -> ExecutionReport {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if let Some(execution) = coordinator
            .run_next(deadline.saturating_duration_since(std::time::Instant::now()))
            .unwrap()
        {
            return execution.report;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "native execution did not produce a guest boundary"
        );
    }
}

#[test]
fn loader_stub_executes_before_runtime_termination() {
    let (_directory, plan) = plan();
    let process = reference_process_builder(super::test_kernel())
        .build(&plan)
        .unwrap();
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    let stub = process.main_thread().loader_return.unwrap();
    write_code(&process, entry, &0xd65f03c0_u32.to_le_bytes());
    let result = 0x8123_4567_89ab_cdef;
    process
        .main_thread()
        .registers()
        .write_x(A64Register::General(a64_register(0)), result);
    let thread = process.main_thread();
    let (mut coordinator, _, process) = scheduled(process);
    let report = execute(&mut coordinator);
    assert!(
        matches!(report.stop, ExecutionStop::LoaderReturn { source, result_code } if source.pc == stub && result_code == result)
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
    assert_eq!(thread.exit().unwrap().exit_code, result);
    assert!(process.handles().is_empty());
    drop(thread);
    coordinator.shutdown().unwrap();
}
#[test]
fn loader_stub_recognition_requires_the_thread_address_and_exit_svc() {
    for stub_address in [false, true] {
        let (_directory, plan) = plan();
        let process = reference_process_builder(super::test_kernel())
            .build(&plan)
            .unwrap();
        let pc = if stub_address {
            process.main_thread().loader_return.unwrap()
        } else {
            GuestVirtualAddress::new(process.entry_module().entry_address())
        };
        let (instruction, immediate) = if stub_address {
            (0xd4000841_u32, 0x42)
        } else {
            (HOME_BREW_EXIT_PROCESS_INSTRUCTION, 7)
        };
        write_code(&process, pc, &instruction.to_le_bytes());
        process.main_thread().registers().set_pc(pc.get());
        let (mut coordinator, _, process) = scheduled(process);
        let report = execute(&mut coordinator);
        assert!(
            matches!(report.stop, ExecutionStop::SupervisorCall { source, immediate: observed } if source.pc == pc && observed == immediate)
        );
        assert_eq!(process.main_thread().registers().pc(), pc.get());
        assert!(report.context.is_none());
        assert!(process.exit().is_none());
        coordinator.shutdown().unwrap();
    }
}
#[test]
fn fixed_virtual_timer_is_stable_across_native_slices() {
    let (_directory, plan) = plan();
    let frequency = 24_000_000;
    let fixed = nixe_scheduler::SchedulerState::new(
        super::test_kernel().profile().clone(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed {
            unix_seconds: 1_700_000_000,
        }),
    );
    let process = reference_process_builder(fixed)
        .with_config(ProcessBuildConfig {
            architectural_timer_frequency: frequency,
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instructions(
        &process,
        &[
            0xd53b_e001, // MRS X1,CNTFRQ_EL0
            0xd53b_e022, // MRS X2,CNTVCT_EL0
            0xd53b_e023, // MRS X3,CNTVCT_EL0
            0xd4000001,  // SVC.
        ],
    );

    let (mut coordinator, _, process) = scheduled(process);
    let first = execute(&mut coordinator);

    assert!(matches!(
        first.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));
    let entry = process.entry_module().entry_address();
    process.main_thread().registers().set_pc(entry);
    coordinator
        .scheduler()
        .make_ready(process.main_thread_id())
        .unwrap();
    let second = execute(&mut coordinator);

    assert!(matches!(
        second.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));

    let state_thread = process.main_thread();

    let state = state_thread.registers();
    assert_eq!(
        state.read_x(A64Register::General(a64_register(1))),
        frequency
    );
    assert_eq!(state.read_x(A64Register::General(a64_register(2))), 0);
    assert_eq!(state.read_x(A64Register::General(a64_register(3))), 0);
    drop(state);
}

#[test]
fn reference_execution_reports_instruction_fetch_faults_as_a_distinct_stop() {
    let (_directory, plan) = plan();
    let process = reference_process_builder(super::test_kernel())
        .build(&plan)
        .unwrap();
    let state_thread = process.main_thread();
    let mut state = state_thread.registers();
    state.set_pc(0x1000);
    drop(state);

    let (mut coordinator, _, _process) = scheduled(process);
    let report = execute(&mut coordinator);

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
    let process = reference_process_builder(super::test_kernel())
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&process, 0);
    let entry = process.entry_module().entry_address();
    let (mut coordinator, _, process) = scheduled(process);
    let report = execute(&mut coordinator);
    let crate::ExecutionStop::UnallocatedEncoding { error } = report.stop else {
        panic!("unallocated encoding must stop execution")
    };
    assert_eq!(error.instruction.location.pc.get(), entry);
    assert_eq!(error.instruction.encoding.bits(), 0);
    assert_eq!(process.main_thread().registers().pc(), entry);
}

#[test]
fn reference_execution_distinguishes_svc_architectural_and_data_fault_stops() {
    let (_directory, plan) = plan();

    let svc = reference_process_builder(super::test_kernel())
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&svc, 0xd400_0841); // SVC #0x42
    let (mut svc_coordinator, _, _svc) = scheduled(svc);
    let report = execute(&mut svc_coordinator);
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

    let breakpoint = reference_process_builder(super::test_kernel())
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&breakpoint, 0xd420_2460); // BRK #0x123
    let (mut breakpoint_coordinator, _, _breakpoint) = scheduled(breakpoint);
    let report = execute(&mut breakpoint_coordinator);
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
