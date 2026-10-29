use super::*;
use crate::ThreadCreateRequest;
use crate::process::tests::synthetic_native_instruction_process;
use nixe_scheduler::MachineSchedulerProfile;

#[test]
fn shutdown_cancels_an_execution_queued_before_its_worker_enters_native_code() {
    let core = VirtualCpuId::new(7);
    let profile = MachineSchedulerProfile::new(
        vec![nixe_scheduler::VirtualCpuDescriptor::new(core, 0)],
        nixe_scheduler::PriorityRange::new(0, 63).unwrap(),
    )
    .unwrap();
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile,
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let process = coordinator
        .register_process(
            synthetic_native_instruction_process(coordinator.scheduler(), 1, &[0x14000000]),
            crate::ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(core),
                affinity: coordinator.scheduler.profile().all_cores(),
            },
        )
        .unwrap();
    let lease = coordinator.scheduler.claim(core).unwrap().unwrap();
    let runtime = coordinator.process(process).unwrap();
    let cpu_thread = WorkerCpuThreadKey {
        process,
        cpu_process: runtime.cpu_process_id(),
    };
    let execution = runtime
        .begin_thread_execution(lease.thread, core, Default::default())
        .unwrap();
    coordinator.request_process_stop(process).unwrap();
    coordinator
        .workers
        .dispatch(
            WorkerRequest {
                lease,
                cpu_thread,
                execution,
            },
            None,
        )
        .unwrap_or_else(|_| panic!("worker dispatch failed"));
    coordinator.shutdown().unwrap();
    assert_eq!(coordinator.process_count(), 0);
    assert!(!coordinator.scheduler.has_active_leases());
    assert_eq!(coordinator.scheduler.thread_count(), 0);
}

#[test]
fn a_reschedule_after_native_svc_return_is_handled_on_the_original_core_without_replaying_it() {
    let profile = MachineSchedulerProfile::new(
        vec![nixe_scheduler::VirtualCpuDescriptor::new(
            VirtualCpuId::new(7),
            0,
        )],
        nixe_scheduler::PriorityRange::new(0, 63).unwrap(),
    )
    .unwrap();
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile,
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let process = coordinator
        .register_process(
            synthetic_native_instruction_process(coordinator.scheduler(), 1, &[0xd4000201]),
            crate::ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(VirtualCpuId::new(7)),
                affinity: coordinator.scheduler.profile().all_cores(),
            },
        )
        .unwrap();
    let runtime = coordinator.process(process).unwrap();
    let main = runtime.main_thread_id();
    let tls = runtime.main_thread().tls_base;
    runtime
        .memory()
        .space()
        .write_value(tls.checked_add(0x100).unwrap(), 4, 2)
        .unwrap()
        .into_ready()
        .unwrap();
    let lease = coordinator
        .scheduler
        .claim(VirtualCpuId::new(7))
        .unwrap()
        .unwrap();
    coordinator.dispatch_worker(lease, None).unwrap();
    let result = match coordinator
        .inbox
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap()
        .unwrap()
    {
        crate::external_event::CoordinatorEvent::Completed(result) => result,
        _ => panic!("expected actual native completion"),
    };
    assert!(
        matches!(&result.outcome, Ok(report) if matches!(report.stop, ExecutionStop::SupervisorCall { immediate: 0x10, .. }))
    );
    let runtime = coordinator.process(process).unwrap();
    let request = ThreadCreateRequest {
        entry: nixe_memory::GuestVirtualAddress::new(runtime.entry_module().entry_address()),
        argument: 0,
        fpcr: 0,
        fpsr: 0,
        stack_top: runtime.main_thread().stack_top,
        priority: 10,
        ideal_vcpu: Some(VirtualCpuId::new(7)),
        affinity: coordinator.scheduler.profile().all_cores(),
    };
    let higher =
        crate::process::tests::ready(coordinator.process(process).unwrap().create_thread(request))
            .unwrap()
            .id;
    {
        let id = GuestThreadId::new(higher.get());
        let owner = coordinator.scheduler().thread(id).unwrap().process;
        coordinator
            .process(owner)
            .unwrap()
            .thread(id)
            .unwrap()
            .start()
    }
    .unwrap();
    coordinator.workers.pending_results.push_back(result);
    let execution = coordinator
        .run_next(std::time::Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(execution.lease, lease);
    assert!(matches!(
        execution.report.stop,
        ExecutionStop::SupervisorCall {
            immediate: 0x10,
            ..
        }
    ));
    assert_eq!(
        coordinator
            .scheduler
            .pinned_thread(process, VirtualCpuId::new(7))
            .unwrap(),
        Some(main)
    );
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .memory()
            .space()
            .read_value(tls.checked_add(0x102).unwrap(), 2)
            .unwrap()
            .into_ready()
            .unwrap(),
        1
    );
    assert!(!coordinator.scheduler.has_active_leases());
}

#[test]
fn shutdown_drains_a_stopped_worker_after_a_different_core_faults() {
    let cores = [VirtualCpuId::new(0), VirtualCpuId::new(1)];
    let profile = MachineSchedulerProfile::new(
        cores
            .map(|core| nixe_scheduler::VirtualCpuDescriptor::new(core, 0))
            .to_vec(),
        nixe_scheduler::PriorityRange::new(0, 63).unwrap(),
    )
    .unwrap();
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile,
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let first = coordinator
        .scheduler
        .profile()
        .core_set([cores[0]])
        .unwrap();
    let second = coordinator
        .scheduler
        .profile()
        .core_set([cores[1]])
        .unwrap();
    let process = coordinator
        .register_process(
            synthetic_native_instruction_process(coordinator.scheduler(), 1, &[0x14000000]),
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(cores[0]),
                affinity: first,
            },
        )
        .unwrap();
    let runtime = coordinator.process(process).unwrap().clone();
    let request = ThreadCreateRequest {
        entry: nixe_memory::GuestVirtualAddress::new(runtime.entry_module().entry_address()),
        argument: 0,
        fpcr: 0,
        fpsr: 0,
        stack_top: runtime.main_thread().stack_top,
        priority: 44,
        ideal_vcpu: Some(cores[1]),
        affinity: second,
    };
    let child = crate::process::tests::ready(runtime.create_thread(request)).unwrap();
    runtime.thread(child.id).unwrap().start().unwrap();
    let failed = coordinator.scheduler.claim(cores[0]).unwrap().unwrap();
    let queued = coordinator.scheduler.claim(cores[1]).unwrap().unwrap();
    let failed_state = runtime
        .begin_thread_execution(failed.thread, cores[0], Default::default())
        .unwrap();
    let queued_state = runtime
        .begin_thread_execution(queued.thread, cores[1], Default::default())
        .unwrap();
    // Exercise the production dispatch-failure reconciliation before the other
    // worker enters native code; its stop is then guaranteed to be observed.
    runtime.abort_thread_execution(failed.thread, failed.vcpu, Some(failed_state));
    coordinator.complete_failed_worker_lease(failed).unwrap();
    assert_eq!(
        runtime.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Terminating
    );
    coordinator.request_process_stop(process).unwrap();
    coordinator
        .workers
        .dispatch(
            WorkerRequest {
                lease: queued,
                cpu_thread: WorkerCpuThreadKey {
                    process,
                    cpu_process: runtime.cpu_process_id(),
                },
                execution: queued_state,
            },
            None,
        )
        .unwrap_or_else(|_| panic!("worker dispatch failed"));
    coordinator.shutdown().unwrap();
    assert_eq!(
        runtime.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Faulted
    );
    assert_eq!(coordinator.resource_counts(), Default::default());
}
