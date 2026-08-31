use super::*;
use crate::process::tests::{
    synthetic_instruction_process_for_coordinator, synthetic_process_for_coordinator,
    synthetic_svc_process_for_coordinator,
};
use nixe_cpu::execution::SchedulerRequest;
use nixe_scheduler::{PriorityRange, VirtualCpuDescriptor};

fn profile() -> MachineSchedulerProfile {
    MachineSchedulerProfile::new(
        vec![VirtualCpuDescriptor::new(VirtualCpuId::new(7), 0)],
        PriorityRange::new(0, 63).unwrap(),
        10,
    )
    .unwrap()
}

fn two_core_profile() -> MachineSchedulerProfile {
    MachineSchedulerProfile::new(
        vec![
            VirtualCpuDescriptor::new(VirtualCpuId::new(3), 0),
            VirtualCpuDescriptor::new(VirtualCpuId::new(7), 0),
        ],
        PriorityRange::new(0, 63).unwrap(),
        10,
    )
    .unwrap()
}

fn registration(coordinator: &RuntimeCoordinator) -> ProcessRegistration {
    ProcessRegistration {
        priority: 44,
        ideal_vcpu: Some(VirtualCpuId::new(7)),
        affinity: coordinator.scheduler().profile().all_cores(),
    }
}

#[test]
fn registration_removal_and_identity_retirement_are_atomic() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = synthetic_process_for_coordinator(1);
    let id = coordinator
        .register_process(process, registration(&coordinator))
        .unwrap();
    assert_eq!(coordinator.process_count(), 1);
    assert_eq!(
        coordinator
            .scheduler()
            .thread(nixe_scheduler::GuestThreadId::new(1))
            .unwrap()
            .process,
        id
    );
    let process = coordinator.remove_process(id).unwrap();
    assert_eq!(coordinator.process_count(), 0);
    let error = coordinator
        .register_process(process, registration(&coordinator))
        .unwrap_err();
    assert!(matches!(
        error,
        CoordinatorError::Execution {
            error: ProcessExecutionError::Cpu { fault },
            ..
        } if fault.kind == nixe_cpu::execution::CpuFaultKind::Unavailable
    ));
    let replacement = coordinator
        .register_process(
            synthetic_process_for_coordinator(2),
            registration(&coordinator),
        )
        .unwrap();
    assert_ne!(replacement, id);
    assert_eq!(coordinator.process_count(), 1);
}

#[test]
fn coordinator_preserves_external_event_sequence_boundaries() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let sender = coordinator.event_sender();
    sender.submit(crate::ExternalEvent::HostStop).unwrap();
    sender.submit(crate::ExternalEvent::HostStop).unwrap();
    let report = coordinator.drain_external_events().unwrap();
    assert_eq!(report.received, 2);
    assert_eq!(
        report.first_sequence.map(|sequence| sequence.get()),
        Some(1)
    );
    assert_eq!(report.last_sequence.map(|sequence| sequence.get()), Some(2));
}

#[test]
fn bounded_external_wait_expires_without_fabricating_guest_readiness() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let _process_id = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let execution = coordinator.run_next(2).unwrap().unwrap();
    let thread = execution.lease.thread;
    coordinator.register_timed_wait(thread, None).unwrap();

    assert_eq!(
        coordinator
            .wait_for_external_event_for(std::time::Duration::ZERO)
            .unwrap(),
        None
    );
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Waiting
    );
}

#[test]
fn one_slice_flows_through_a_scheduler_lease() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = synthetic_process_for_coordinator(1);
    let id = coordinator
        .register_process(process, registration(&coordinator))
        .unwrap();
    let execution = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(execution.lease.process, id);
    assert_eq!(execution.lease.vcpu, VirtualCpuId::new(7));
    assert_eq!(execution.report.progress, 1);
    assert_eq!(
        coordinator
            .scheduler()
            .thread(execution.lease.thread)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
}

#[test]
fn scheduler_hints_use_vcpu_owned_event_and_interrupt_state() {
    const WFE: u32 = 0xd503_205f;
    const WFI: u32 = 0xd503_207f;
    const SEV: u32 = 0xd503_209f;
    const SEVL: u32 = 0xd503_20bf;
    const YIELD: u32 = 0xd503_203f;

    let mut coordinator = RuntimeCoordinator::new(profile());
    let waiter = coordinator
        .register_process(
            synthetic_instruction_process_for_coordinator(1, &[WFE]),
            registration(&coordinator),
        )
        .unwrap();
    let waited = coordinator.run_next(1).unwrap().unwrap();
    assert!(matches!(
        waited.report.stop,
        ExecutionStop::Scheduled {
            request: SchedulerRequest::WaitForEvent,
            ..
        }
    ));
    assert_eq!(
        coordinator
            .scheduler()
            .thread(waited.lease.thread)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Waiting
    );

    let sender = coordinator
        .register_process(
            synthetic_instruction_process_for_coordinator(2, &[SEV]),
            registration(&coordinator),
        )
        .unwrap();
    let sent = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(sent.lease.process, sender);
    assert!(matches!(
        sent.report.stop,
        ExecutionStop::Scheduled {
            request: SchedulerRequest::SendEvent,
            ..
        }
    ));
    assert_eq!(
        coordinator
            .scheduler()
            .thread(waited.lease.thread)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
    assert!(coordinator.process(waiter).is_some());

    let mut interrupt_coordinator = RuntimeCoordinator::new(profile());
    interrupt_coordinator
        .register_process(
            synthetic_instruction_process_for_coordinator(3, &[WFI]),
            registration(&interrupt_coordinator),
        )
        .unwrap();
    let interrupted = interrupt_coordinator.run_next(1).unwrap().unwrap();
    assert!(matches!(
        interrupted.report.stop,
        ExecutionStop::Scheduled {
            request: SchedulerRequest::WaitForInterrupt,
            ..
        }
    ));
    interrupt_coordinator
        .post_vcpu_interrupt(VirtualCpuId::new(7), 0x40)
        .unwrap();
    assert_eq!(
        interrupt_coordinator
            .scheduler()
            .thread(interrupted.lease.thread)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
    let pending = interrupt_coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(
        pending.report.stop,
        ExecutionStop::PendingEvent { mask: 0x40 }
    );

    let mut local = RuntimeCoordinator::new(profile());
    local
        .register_process(
            synthetic_instruction_process_for_coordinator(4, &[SEVL, WFE, YIELD]),
            registration(&local),
        )
        .unwrap();
    let local_exit = local.run_next(3).unwrap().unwrap();
    assert_eq!(local_exit.report.progress, 3);
    assert!(matches!(
        local_exit.report.stop,
        ExecutionStop::Scheduled {
            request: SchedulerRequest::Yield,
            ..
        }
    ));
    assert_eq!(
        local
            .scheduler()
            .thread(local_exit.lease.thread)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
}

#[test]
fn adaptive_execution_budget_grows_only_across_uninterrupted_slices() {
    let mut budget = AdaptiveExecutionBudget::new(10_000);
    for expected in [20_000, 40_000, 80_000, 100_000, 100_000] {
        budget.observe(true);
        assert_eq!(budget.current, expected);
    }
    budget.observe(false);
    assert_eq!(budget.current, 10_000);
}

#[test]
fn guest_thread_ids_are_unique_across_live_processes() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let first = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let second = coordinator
        .register_process(
            synthetic_process_for_coordinator(2),
            registration(&coordinator),
        )
        .unwrap();
    let first_thread = coordinator.process(first).unwrap().main_thread_id();
    let second_thread = coordinator.process(second).unwrap().main_thread_id();
    assert_ne!(first_thread, second_thread);
    let first_object = coordinator.process(first).unwrap().main_thread().object();
    let second_object = coordinator.process(second).unwrap().main_thread().object();
    assert_eq!(first_object.thread_id(), first_thread.get());
    assert_eq!(second_object.thread_id(), second_thread.get());
    assert_ne!(first_object, second_object);
    assert_eq!(
        coordinator
            .thread_scheduling_info(first_object.thread_id())
            .unwrap()
            .thread,
        first_thread,
        "a copied thread object resolves globally instead of aliasing a local ID"
    );
    assert_eq!(
        coordinator
            .scheduler()
            .thread(first_thread)
            .unwrap()
            .process,
        first
    );
    assert_eq!(
        coordinator
            .scheduler()
            .thread(second_thread)
            .unwrap()
            .process,
        second
    );
}

#[test]
fn external_wakes_are_generation_safe_and_idempotent() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = synthetic_process_for_coordinator(1);
    let _id = coordinator
        .register_process(process, registration(&coordinator))
        .unwrap();
    let execution = coordinator.run_next(2).unwrap().unwrap();
    let thread = execution.lease.thread;
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Waiting
    );
    let token = coordinator
        .register_wait(thread, Readiness::Pending)
        .unwrap()
        .unwrap();
    let sender = coordinator.event_sender();
    let event = ExternalEvent::Wake {
        source: crate::ExternalEventSource::Device,
        token,
    };
    sender.submit(event).unwrap();
    sender.submit(event).unwrap();
    let report = coordinator.drain_external_events().unwrap();
    assert_eq!(report.woken, 1);
    assert_eq!(report.stale, 1);
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
}

struct ResumeDispatcher;

impl crate::ExceptionDispatcher for ResumeDispatcher {
    type Fault = &'static str;

    fn dispatch(
        &mut self,
        _context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        crate::ExceptionDispatchOutcome::Resume(crate::ExceptionResume::Next)
    }
}

#[test]
fn exception_resume_updates_runtime_and_scheduler_as_one_coordinator_operation() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = synthetic_svc_process_for_coordinator(1);
    let _process_id = coordinator
        .register_process(process, registration(&coordinator))
        .unwrap();
    let execution = coordinator.run_next(1).unwrap().unwrap();
    assert!(matches!(
        execution.report.stop,
        ExecutionStop::SupervisorCall { .. }
    ));
    assert_eq!(
        coordinator
            .route_supervisor_call(
                execution.lease,
                &execution.report.stop,
                &mut ResumeDispatcher,
            )
            .unwrap(),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        coordinator
            .scheduler()
            .thread(execution.lease.thread)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
}

fn valid_thread_request(
    coordinator: &RuntimeCoordinator,
    process: ProcessId,
) -> ThreadCreateRequest {
    let runtime = coordinator.process(process).unwrap();
    ThreadCreateRequest {
        entry: nixe_memory::GuestVirtualAddress::new(runtime.entry_module().entry_address()),
        argument: 0x1234,
        stack_top: runtime.main_thread().stack_top,
        priority: 20,
        ideal_vcpu: Some(VirtualCpuId::new(7)),
        affinity: coordinator.scheduler().profile().all_cores(),
    }
}

#[test]
fn thread_construction_is_created_and_failure_atomic() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let handles_before = coordinator.process(process).unwrap().handles().len();
    let threads_before = coordinator.process(process).unwrap().threads().len();

    for invalid in [
        ThreadCreateRequest {
            entry: nixe_memory::GuestVirtualAddress::new(3),
            ..valid_thread_request(&coordinator, process)
        },
        ThreadCreateRequest {
            stack_top: nixe_memory::GuestVirtualAddress::new(3),
            ..valid_thread_request(&coordinator, process)
        },
        ThreadCreateRequest {
            priority: 64,
            ..valid_thread_request(&coordinator, process)
        },
        ThreadCreateRequest {
            ideal_vcpu: Some(VirtualCpuId::new(99)),
            ..valid_thread_request(&coordinator, process)
        },
    ] {
        assert!(coordinator.create_thread(process, invalid).is_err());
        assert_eq!(
            coordinator.process(process).unwrap().handles().len(),
            handles_before
        );
        assert_eq!(
            coordinator.process(process).unwrap().threads().len(),
            threads_before
        );
    }

    let request = valid_thread_request(&coordinator, process);
    let creation = coordinator.create_thread(process, request).unwrap();
    assert_eq!(
        coordinator.process(process).unwrap().handles().len(),
        handles_before + 1
    );
    assert_eq!(
        coordinator.process(process).unwrap().threads().len(),
        threads_before + 1
    );
    assert_eq!(
        coordinator
            .scheduler()
            .thread(creation.id)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Created
    );
    let object_id = coordinator
        .process(process)
        .unwrap()
        .thread(creation.id)
        .unwrap()
        .object()
        .thread_id();
    assert_eq!(coordinator.start_thread(object_id), Ok(creation.id));
    assert_eq!(
        coordinator
            .scheduler()
            .thread(creation.id)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
    assert_eq!(
        coordinator.start_thread(object_id),
        Err(ThreadOperationError::InvalidState)
    );
}

#[test]
fn terminal_thread_reaping_reclaims_and_reuses_its_tls_slot() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let initial_used = coordinator
        .process(process)
        .unwrap()
        .memory_accounting()
        .used_non_system_user_physical_memory_size();
    let first = coordinator
        .create_thread(process, valid_thread_request(&coordinator, process))
        .unwrap();
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .memory_accounting()
            .used_non_system_user_physical_memory_size(),
        initial_used + nixe_cpu::memory::SYNTHETIC_PAGE_SIZE as u64
    );
    let tls = coordinator
        .process(process)
        .unwrap()
        .thread(first.id)
        .unwrap()
        .tls_base;
    coordinator
        .scheduler
        .apply(SchedulerCommand::Terminate {
            thread: first.id,
            faulted: false,
        })
        .unwrap();
    coordinator
        .processes
        .get_mut(&process)
        .unwrap()
        .terminate_thread_from_coordinator(
            first.id,
            0,
            crate::ExceptionTerminationScope::CurrentThread,
        );
    coordinator
        .process_mut(process)
        .unwrap()
        .handles_mut()
        .close(first.handle)
        .unwrap();
    coordinator.reap_thread(first.id.get()).unwrap();
    assert!(
        coordinator
            .process(process)
            .unwrap()
            .thread(first.id)
            .is_none()
    );
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .memory_accounting()
            .used_non_system_user_physical_memory_size(),
        initial_used
    );

    let replacement = coordinator
        .create_thread(process, valid_thread_request(&coordinator, process))
        .unwrap();
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .thread(replacement.id)
            .unwrap()
            .tls_base,
        tls
    );
    assert_eq!(
        coordinator
            .process(process)
            .unwrap()
            .memory_accounting()
            .used_non_system_user_physical_memory_size(),
        initial_used + nixe_cpu::memory::SYNTHETIC_PAGE_SIZE as u64
    );
}

struct TerminateDispatcher(crate::ExceptionTerminationScope);

impl crate::ExceptionDispatcher for TerminateDispatcher {
    type Fault = &'static str;

    fn dispatch(
        &mut self,
        _context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        crate::ExceptionDispatchOutcome::Terminate {
            scope: self.0,
            exit_code: 7,
            reason: crate::ExceptionTerminationReason::Requested,
        }
    }
}

#[test]
fn thread_exit_preserves_other_threads_and_process_exit_terminates_all() {
    for scope in [
        crate::ExceptionTerminationScope::CurrentThread,
        crate::ExceptionTerminationScope::Process,
    ] {
        let mut coordinator = RuntimeCoordinator::new(profile());
        let process_id = coordinator
            .register_process(
                synthetic_svc_process_for_coordinator(1),
                registration(&coordinator),
            )
            .unwrap();
        let creation = coordinator
            .create_thread(process_id, valid_thread_request(&coordinator, process_id))
            .unwrap();
        let object_id = coordinator
            .process(process_id)
            .unwrap()
            .thread(creation.id)
            .unwrap()
            .object()
            .thread_id();
        let join_object = coordinator
            .process(process_id)
            .unwrap()
            .thread(creation.id)
            .unwrap()
            .object();
        assert!(!join_object.is_signalled());
        coordinator.start_thread(object_id).unwrap();
        let execution = coordinator.run_next(1).unwrap().unwrap();
        assert_eq!(execution.lease.thread, creation.id);
        coordinator
            .route_supervisor_call(
                execution.lease,
                &execution.report.stop,
                &mut TerminateDispatcher(scope),
            )
            .unwrap();
        let process = coordinator.process(process_id).unwrap();
        assert_eq!(
            coordinator
                .scheduler()
                .thread(creation.id)
                .unwrap()
                .lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited
        );
        assert!(join_object.is_signalled());
        let main = process.main_thread_id();
        if scope == crate::ExceptionTerminationScope::CurrentThread {
            assert_eq!(
                process.lifecycle(),
                nixe_scheduler::ProcessLifecycle::Running
            );
            assert_eq!(
                coordinator.scheduler().thread(main).unwrap().lifecycle,
                nixe_scheduler::ThreadLifecycle::Ready
            );
        } else {
            assert_eq!(
                process.lifecycle(),
                nixe_scheduler::ProcessLifecycle::Exited
            );
            assert_eq!(
                coordinator.scheduler().thread(main).unwrap().lifecycle,
                nixe_scheduler::ThreadLifecycle::Exited
            );
        }
    }
}

#[test]
fn priority_and_affinity_updates_use_the_machine_profile() {
    let mut coordinator = RuntimeCoordinator::new(two_core_profile());
    let process_id = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(VirtualCpuId::new(7)),
                affinity: coordinator.scheduler().profile().all_cores(),
            },
        )
        .unwrap();
    let creation = coordinator
        .create_thread(
            process_id,
            ThreadCreateRequest {
                ideal_vcpu: Some(VirtualCpuId::new(3)),
                affinity: coordinator
                    .scheduler()
                    .profile()
                    .core_set([VirtualCpuId::new(3)])
                    .unwrap(),
                ..valid_thread_request(&coordinator, process_id)
            },
        )
        .unwrap();
    let object_id = coordinator
        .process(process_id)
        .unwrap()
        .thread(creation.id)
        .unwrap()
        .object()
        .thread_id();
    let main_object_id = coordinator
        .process(process_id)
        .unwrap()
        .main_thread()
        .object()
        .thread_id();
    coordinator.set_thread_priority(object_id, 5).unwrap();
    coordinator
        .inherit_thread_priority(main_object_id, object_id, 0x1000)
        .unwrap();
    let main_id = coordinator.process(process_id).unwrap().main_thread_id();
    assert_eq!(
        coordinator
            .scheduler()
            .thread(main_id)
            .unwrap()
            .effective_priority,
        5
    );
    coordinator
        .restore_thread_priority(main_object_id, 0x1000)
        .unwrap();
    let affinity = coordinator
        .scheduler()
        .profile()
        .core_set([VirtualCpuId::new(7)])
        .unwrap();
    coordinator
        .set_thread_affinity(object_id, Some(VirtualCpuId::new(7)), affinity.clone())
        .unwrap();
    let info = coordinator.thread_scheduling_info(object_id).unwrap();
    assert_eq!(info.base_priority, 5);
    assert_eq!(info.ideal_vcpu, Some(VirtualCpuId::new(7)));
    assert_eq!(info.affinity, affinity);
    assert_eq!(
        coordinator.set_thread_priority(object_id, 64),
        Err(ThreadOperationError::InvalidState)
    );
    coordinator.start_thread(object_id).unwrap();
    coordinator.set_thread_activity(object_id, true).unwrap();
    assert!(
        coordinator
            .thread_scheduling_info(object_id)
            .unwrap()
            .paused
    );
    coordinator.set_thread_activity(object_id, false).unwrap();
    assert!(
        !coordinator
            .thread_scheduling_info(object_id)
            .unwrap()
            .paused
    );
}

#[test]
fn priority_donations_are_multi_source_transitive_and_keyed() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let mut processes = Vec::new();
    for seed in 1..=3 {
        let process = coordinator
            .register_process(
                synthetic_process_for_coordinator(seed),
                registration(&coordinator),
            )
            .unwrap();
        let object = coordinator
            .process(process)
            .unwrap()
            .main_thread()
            .object()
            .thread_id();
        processes.push((process, object));
    }
    coordinator.set_thread_priority(processes[0].1, 40).unwrap();
    coordinator.set_thread_priority(processes[1].1, 30).unwrap();
    coordinator.set_thread_priority(processes[2].1, 5).unwrap();
    coordinator
        .inherit_thread_priority(processes[0].1, processes[1].1, 0x1000)
        .unwrap();
    coordinator
        .inherit_thread_priority(processes[1].1, processes[2].1, 0x2000)
        .unwrap();
    assert_eq!(
        coordinator
            .thread_scheduling_info(processes[0].1)
            .unwrap()
            .effective_priority,
        5
    );
    assert_eq!(
        coordinator
            .thread_scheduling_info(processes[1].1)
            .unwrap()
            .effective_priority,
        5
    );
    coordinator
        .restore_thread_priority(processes[1].1, 0x2000)
        .unwrap();
    assert_eq!(
        coordinator
            .thread_scheduling_info(processes[0].1)
            .unwrap()
            .effective_priority,
        30
    );
    coordinator
        .restore_thread_priority(processes[0].1, 0x1000)
        .unwrap();
    assert_eq!(
        coordinator
            .thread_scheduling_info(processes[0].1)
            .unwrap()
            .effective_priority,
        40
    );
}

#[test]
fn process_exit_terminates_created_threads_before_start() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process_id = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let creation = coordinator
        .create_thread(process_id, valid_thread_request(&coordinator, process_id))
        .unwrap();
    let join = coordinator
        .process(process_id)
        .unwrap()
        .thread(creation.id)
        .unwrap()
        .object();
    let execution = coordinator.run_next(1).unwrap().unwrap();
    coordinator
        .route_supervisor_call(
            execution.lease,
            &execution.report.stop,
            &mut TerminateDispatcher(crate::ExceptionTerminationScope::Process),
        )
        .unwrap();
    assert_eq!(
        coordinator
            .scheduler()
            .thread(creation.id)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Exited
    );
    assert!(join.is_signalled());
}

#[test]
fn virtual_deadlines_are_deterministic_and_do_not_sleep_the_host() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let _process_id = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let execution = coordinator.run_next(2).unwrap().unwrap();
    let thread = execution.lease.thread;
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Waiting
    );
    assert!(coordinator.sleep_thread(thread, 100).unwrap().is_some());
    assert_eq!(coordinator.virtual_time_ns(), 0);
    assert_eq!(coordinator.advance_virtual_time(99).unwrap(), 0);
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Waiting
    );
    assert_eq!(coordinator.advance_virtual_time(1).unwrap(), 1);
    assert_eq!(coordinator.virtual_time_ns(), 100);
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
}

#[test]
fn process_removal_cancels_deadlines_and_external_observers() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let process_id = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let execution = coordinator.run_next(2).unwrap().unwrap();
    let thread = execution.lease.thread;
    let (_, readable) = crate::EventObject::create_pair();
    coordinator
        .register_event_wait(
            thread,
            [readable],
            Some(100),
            crate::ExternalEventSource::Device,
        )
        .unwrap();
    assert_eq!(
        coordinator.resource_counts(),
        CoordinatorResourceCounts {
            processes: 1,
            scheduled_threads: 1,
            active_waits: 1,
            deadlines: 1,
            external_watcher_groups: 1,
            priority_donations: 0,
            address_waiters: 0,
        }
    );
    coordinator.remove_process(process_id).unwrap();
    assert_eq!(
        coordinator.resource_counts(),
        CoordinatorResourceCounts::default()
    );
}

#[test]
fn multiple_processes_share_fair_scheduling_and_independent_teardown() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let first = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let second = coordinator
        .register_process(
            synthetic_process_for_coordinator(2),
            registration(&coordinator),
        )
        .unwrap();
    let first_execution = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(first_execution.lease.process, first);
    coordinator
        .route_supervisor_call(
            first_execution.lease,
            &first_execution.report.stop,
            &mut TerminateDispatcher(crate::ExceptionTerminationScope::Process),
        )
        .unwrap();
    assert_eq!(
        coordinator.process(first).unwrap().lifecycle(),
        nixe_scheduler::ProcessLifecycle::Exited
    );
    assert_eq!(
        coordinator.process(second).unwrap().lifecycle(),
        nixe_scheduler::ProcessLifecycle::Running
    );
    let second_execution = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(second_execution.lease.process, second);
    coordinator.remove_process(first).unwrap();
    assert_eq!(coordinator.process_count(), 1);
    assert_eq!(coordinator.scheduler().thread_count(), 1);
}

#[test]
fn equal_virtual_deadlines_wake_in_registration_order() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let first = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    let second = coordinator
        .register_process(
            synthetic_process_for_coordinator(2),
            registration(&coordinator),
        )
        .unwrap();
    let first_thread = coordinator.run_next(2).unwrap().unwrap().lease.thread;
    let second_thread = coordinator.run_next(2).unwrap().unwrap().lease.thread;
    coordinator
        .register_timed_wait(first_thread, Some(50))
        .unwrap();
    coordinator
        .register_timed_wait(second_thread, Some(50))
        .unwrap();
    assert_eq!(coordinator.advance_virtual_time(50).unwrap(), 2);
    let selected = coordinator.run_next(1).unwrap().unwrap();
    assert_eq!(selected.lease.process, first);
    assert_eq!(
        coordinator
            .scheduler()
            .thread(second_thread)
            .unwrap()
            .process,
        second
    );
}

#[test]
fn parallel_wave_assigns_at_most_one_thread_to_each_vcpu() {
    let clock = crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 });
    let mut coordinator = RuntimeCoordinator::try_with_execution_mode(
        two_core_profile(),
        clock,
        VcpuExecutionMode::Parallel,
    )
    .unwrap();
    for (process_id, ideal_vcpu) in [(1, 3), (2, 7)] {
        let affinity = coordinator.scheduler().profile().all_cores();
        coordinator
            .register_process(
                synthetic_process_for_coordinator(process_id),
                ProcessRegistration {
                    priority: 44,
                    ideal_vcpu: Some(VirtualCpuId::new(ideal_vcpu)),
                    affinity,
                },
            )
            .unwrap();
    }
    let executions = coordinator.run_parallel_wave(1).unwrap();
    assert_eq!(executions.len(), 2);
    assert_ne!(executions[0].lease.thread, executions[1].lease.thread);
    assert_ne!(executions[0].lease.vcpu, executions[1].lease.vcpu);
    assert!(coordinator.scheduler().active_leases().next().is_none());
}

#[test]
fn parallel_wave_runs_distinct_threads_from_one_process() {
    let clock = crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 });
    let mut coordinator = RuntimeCoordinator::try_with_execution_mode(
        two_core_profile(),
        clock,
        VcpuExecutionMode::Parallel,
    )
    .unwrap();
    let process_id = coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(VirtualCpuId::new(7)),
                affinity: coordinator.scheduler().profile().all_cores(),
            },
        )
        .unwrap();
    let request = ThreadCreateRequest {
        ideal_vcpu: Some(VirtualCpuId::new(3)),
        affinity: coordinator
            .scheduler()
            .profile()
            .core_set([VirtualCpuId::new(3)])
            .unwrap(),
        ..valid_thread_request(&coordinator, process_id)
    };
    let created = coordinator.create_thread(process_id, request).unwrap();
    let object_id = coordinator
        .process(process_id)
        .unwrap()
        .thread(created.id)
        .unwrap()
        .object()
        .thread_id();
    coordinator.start_thread(object_id).unwrap();

    let executions = coordinator.run_parallel_wave(1).unwrap();
    assert_eq!(executions.len(), 2);
    assert!(
        executions
            .iter()
            .all(|execution| execution.lease.process == process_id)
    );
    assert_ne!(executions[0].lease.thread, executions[1].lease.thread);
}

#[test]
fn parallel_wave_fast_forwards_virtual_deadlines_before_reporting_idle() {
    let clock = crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 });
    let mut coordinator = RuntimeCoordinator::try_with_execution_mode(
        two_core_profile(),
        clock,
        VcpuExecutionMode::Parallel,
    )
    .unwrap();
    coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(VirtualCpuId::new(3)),
                affinity: coordinator.scheduler().profile().all_cores(),
            },
        )
        .unwrap();

    let execution = coordinator.run_parallel_wave(2).unwrap().remove(0);
    let thread = execution.lease.thread;
    assert_eq!(
        coordinator.scheduler().thread(thread).unwrap().lifecycle,
        nixe_scheduler::ThreadLifecycle::Waiting
    );
    assert!(coordinator.sleep_thread(thread, 100).unwrap().is_some());
    assert_eq!(coordinator.virtual_time_ns(), 0);

    let executions = coordinator.run_parallel_wave(1).unwrap();
    assert_eq!(coordinator.virtual_time_ns(), 100);
    assert_eq!(executions.len(), 1);
    assert_eq!(executions[0].lease.thread, thread);
}

#[test]
fn deterministic_records_reproduce_architectural_observations() {
    fn recorded_run() -> crate::ExecutionRecord {
        let mut coordinator = RuntimeCoordinator::new(profile());
        coordinator.enable_execution_recording(std::num::NonZeroUsize::new(16).unwrap());
        let registration = registration(&coordinator);
        coordinator
            .register_process(synthetic_process_for_coordinator(1), registration)
            .unwrap();
        coordinator.run_next(1).unwrap().unwrap();
        coordinator.take_execution_record().unwrap()
    }

    let expected = recorded_run();
    let observed = recorded_run();
    expected.compare(&observed).unwrap();
    assert_eq!(expected.observations().len(), 2);
}

#[test]
fn parallel_observations_replay_through_deterministic_workers() {
    fn register_pair(coordinator: &mut RuntimeCoordinator) {
        for (process_id, ideal_vcpu) in [(1, 3), (2, 7)] {
            let affinity = coordinator.scheduler().profile().all_cores();
            coordinator
                .register_process(
                    synthetic_process_for_coordinator(process_id),
                    ProcessRegistration {
                        priority: 44,
                        ideal_vcpu: Some(VirtualCpuId::new(ideal_vcpu)),
                        affinity,
                    },
                )
                .unwrap();
        }
    }

    let clock = crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 });
    let mut parallel = RuntimeCoordinator::try_with_execution_mode(
        two_core_profile(),
        clock.clone(),
        VcpuExecutionMode::Parallel,
    )
    .unwrap();
    parallel.enable_execution_recording(std::num::NonZeroUsize::new(16).unwrap());
    register_pair(&mut parallel);
    assert_eq!(parallel.run_parallel_wave(1).unwrap().len(), 2);
    let expected = parallel.take_execution_record().unwrap();

    let mut replay = RuntimeCoordinator::with_virtual_clock(two_core_profile(), clock);
    register_pair(&mut replay);
    replay.begin_differential_replay(expected).unwrap();
    replay.run_next(999).unwrap().unwrap();
    replay.run_next(999).unwrap().unwrap();
    replay.finish_differential_replay().unwrap();
}

#[test]
fn coordinator_worker_shutdown_is_idempotent() {
    let mut coordinator = RuntimeCoordinator::new(profile());
    coordinator
        .register_process(
            synthetic_process_for_coordinator(1),
            registration(&coordinator),
        )
        .unwrap();
    coordinator.shutdown().unwrap();
    assert_eq!(
        coordinator.resource_counts(),
        CoordinatorResourceCounts::default()
    );
    coordinator.shutdown().unwrap();
}
