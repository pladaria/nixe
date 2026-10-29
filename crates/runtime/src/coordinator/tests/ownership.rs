use super::*;
use std::sync::{Arc, Barrier};
use std::time::Duration;

fn request(process: &RunnableProcess) -> ThreadCreateRequest {
    ThreadCreateRequest {
        entry: nixe_memory::GuestVirtualAddress::new(process.entry_module().entry_address()),
        argument: 0x1234,
        fpcr: 0,
        fpsr: 0,
        stack_top: process.main_thread().stack_top,
        priority: 20,
        ideal_vcpu: Some(VirtualCpuId::new(7)),
        affinity: process.scheduler().profile().all_cores(),
    }
}

#[test]
fn retained_thread_handles_survive_concurrent_close_priority_and_memory_operations() {
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let id = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 1),
            registration(&coordinator),
        )
        .unwrap();
    let process = coordinator.process(id).unwrap().clone();
    let child = crate::process::tests::ready(process.create_thread(request(&process))).unwrap();
    let retained = process
        .handles()
        .lookup_as::<crate::ThreadObject>(child.handle)
        .unwrap();
    let barrier = Arc::new(Barrier::new(5));
    std::thread::scope(|scope| {
        for worker in 0..4_u64 {
            let process = process.clone();
            let retained = retained.clone();
            let barrier = barrier.clone();
            scope.spawn(move || {
                let address = process
                    .main_thread()
                    .stack_bottom
                    .checked_add(worker * 8)
                    .unwrap();
                barrier.wait();
                for iteration in 0..128 {
                    process
                        .scheduler()
                        .set_priority(child.id, 20 + (iteration % 4), None)
                        .unwrap();
                    if let Some(object) = process
                        .handles()
                        .lookup_as::<crate::ThreadObject>(child.handle)
                    {
                        assert!(Arc::ptr_eq(&retained, &object));
                    }
                    process
                        .memory()
                        .space()
                        .write_value(address, 8, u128::from(worker + 1))
                        .unwrap()
                        .into_ready()
                        .unwrap();
                    assert_eq!(
                        process
                            .memory()
                            .space()
                            .read_value(address, 8)
                            .unwrap()
                            .into_ready()
                            .unwrap(),
                        u128::from(worker + 1)
                    );
                    assert_eq!(retained.id(), child.id);
                    assert!(retained.snapshot().is_some());
                }
            });
        }
        barrier.wait();
        drop(process.handles().close(child.handle).unwrap());
    });
    assert!(process.thread(child.id).is_some());
    drop(retained);
    coordinator.run_next(Duration::from_millis(1)).unwrap();
    assert!(coordinator.scheduler().thread(child.id).is_none());
    assert!(process.thread(child.id).is_none());
    let replacement =
        crate::process::tests::ready(process.create_thread(request(&process))).unwrap();
    assert_ne!(replacement.id, child.id);
    coordinator.shutdown().unwrap();
}

#[test]
fn accepted_thread_can_start_after_its_guest_handle_closes() {
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let id = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 1),
            registration(&coordinator),
        )
        .unwrap();
    let process = coordinator.process(id).unwrap().clone();
    let created = crate::process::tests::ready(process.create_thread(request(&process))).unwrap();
    let accepted = process
        .handles()
        .lookup_as::<crate::ThreadObject>(created.handle)
        .unwrap();
    drop(process.handles().close(created.handle).unwrap());
    accepted.start().unwrap();
    assert!(process.handles().lookup(created.handle).is_none());
    assert_eq!(
        coordinator
            .scheduler()
            .thread(created.id)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
    assert!(Arc::ptr_eq(&process.thread(created.id).unwrap(), &accepted));
    coordinator.shutdown().unwrap();
}

#[test]
fn simultaneous_thread_starts_acquire_one_running_reference_and_cannot_restart_after_exit() {
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let id = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 1),
            registration(&coordinator),
        )
        .unwrap();
    let process = coordinator.process(id).unwrap().clone();
    let child = crate::process::tests::ready(process.create_thread(request(&process))).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let results = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let process = process.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    process.thread(child.id).unwrap().start()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| **result == Err(ThreadStartError::InvalidState))
            .count(),
        1
    );
    coordinator.scheduler().terminate(child.id, false).unwrap();
    process.terminate_thread_from_coordinator(
        child.id,
        0,
        crate::ExceptionTerminationScope::CurrentThread,
    );
    assert_eq!(
        process.thread(child.id).unwrap().start(),
        Err(ThreadStartError::InvalidState)
    );
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Running
    );
    coordinator.shutdown().unwrap();
}

#[test]
fn process_exit_closes_thread_creation_admission_and_retires_racing_allocations() {
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let id = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 1),
            registration(&coordinator),
        )
        .unwrap();
    let process = coordinator.process(id).unwrap().clone();
    let machine = process.memory().account().machine().clone();
    let request = request(&process);
    let barrier = Arc::new(Barrier::new(5));
    let generation = process.generation();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let process = process.clone();
            let request = request.clone();
            let barrier = barrier.clone();
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..16 {
                    match crate::process::tests::ready(process.create_thread(request.clone())) {
                        Ok(child) => {
                            // Termination may already have closed the source table.
                            drop(process.handles().close(child.handle));
                        }
                        Err(crate::ThreadCreateError::Terminating) => break,
                        Err(error) => panic!("racing creation failed: {error}"),
                    }
                }
            });
        }
        barrier.wait();
        coordinator.terminate_process(id).unwrap();
    });
    assert_ne!(process.generation(), generation);
    assert!(matches!(
        crate::process::tests::ready(process.create_thread(request)),
        Err(crate::ThreadCreateError::Terminating)
    ));
    coordinator.shutdown().unwrap();
    assert_eq!(coordinator.resource_counts().scheduled_threads, 0);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn exported_threads_keep_their_kernel_identity_after_the_owner_process_is_removed() {
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let owner = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 1),
            registration(&coordinator),
        )
        .unwrap();
    let receiver = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 2),
            registration(&coordinator),
        )
        .unwrap();
    let process = coordinator.process(owner).unwrap().clone();
    let child = crate::process::tests::ready(process.create_thread(request(&process))).unwrap();
    let foreign = coordinator
        .process(receiver)
        .unwrap()
        .handles()
        .insert_object(process.handles().lookup(child.handle).unwrap())
        .unwrap();
    let parent = Arc::downgrade(&process);
    coordinator.remove_process(owner).unwrap();
    drop(process);
    assert!(parent.upgrade().is_some());
    let view = coordinator
        .scheduler()
        .thread(child.id)
        .expect("exported identity remains retained");
    assert_eq!(view.process, owner);
    assert_eq!(view.lifecycle, nixe_scheduler::ThreadLifecycle::Exited);
    assert!(
        coordinator
            .process(receiver)
            .unwrap()
            .handles()
            .lookup_as::<crate::ThreadObject>(foreign)
            .unwrap()
            .is_signalled()
    );
    drop(
        coordinator
            .process(receiver)
            .unwrap()
            .handles()
            .close(foreign)
            .unwrap(),
    );
    coordinator.run_next(Duration::from_millis(1)).unwrap();
    assert!(coordinator.scheduler().thread(child.id).is_none());
    assert!(parent.upgrade().is_none());
    coordinator.shutdown().unwrap();
}

#[test]
fn thread_completion_observers_see_terminal_scheduling_and_restored_registers() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Exit;
    impl crate::ExceptionDispatcher for Exit {
        type Fault = ();
        fn dispatch(
            &mut self,
            _: &mut crate::ExceptionDispatchContext<'_>,
            _: crate::ExceptionDispatchRequest,
        ) -> crate::ExceptionDispatchOutcome<()> {
            crate::ExceptionDispatchOutcome::Terminate {
                scope: crate::ExceptionTerminationScope::CurrentThread,
                exit_code: 0,
                reason: crate::ExceptionTerminationReason::Requested,
            }
        }
    }
    let mut coordinator = RuntimeCoordinator::new(SchedulerState::new(
        profile(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let id = coordinator
        .register_process(
            synthetic_svc_process_for_coordinator(coordinator.scheduler(), 1),
            registration(&coordinator),
        )
        .unwrap();
    let thread = coordinator.process(id).unwrap().main_thread();
    let observed = Arc::new(AtomicBool::new(false));
    let watched = thread.clone();
    let kernel = coordinator.scheduler.clone();
    let notified = observed.clone();
    let _registration = crate::process::tests::watch_future(
        thread.readable_event().wait_async(),
        Arc::new(move || {
            assert_eq!(
                kernel.thread(watched.id()).unwrap().lifecycle,
                nixe_scheduler::ThreadLifecycle::Exited
            );
            assert!(watched.snapshot().is_some());
            notified.store(true, Ordering::Release);
        }),
    );
    let execution = coordinator
        .run_next(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    coordinator
        .route_supervisor_call(execution.lease, &execution.report.stop, &mut Exit)
        .unwrap();
    assert!(observed.load(Ordering::Acquire));
    drop(thread);
    coordinator.shutdown().unwrap();
}
