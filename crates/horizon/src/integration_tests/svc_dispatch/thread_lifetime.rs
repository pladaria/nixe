use super::*;
use nixe_scheduler::VirtualCpuId;

#[test]
fn closing_the_self_handle_invalidates_its_number_but_preserves_duplicates_and_pseudo_handles() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x0c),
        svc(0x16),
        svc(0x0c),
        svc(0x25),
        svc(0x0e),
        svc(0x0d),
        svc(0x0c),
        svc(0x25),
        svc(0x0e),
        svc(0x0c),
        svc(0x25),
        svc(0x0e),
    ]);
    let mut dispatcher = support::dispatcher(&process);
    let original = process.main_thread().handle;
    let duplicate = process.handles().duplicate(original).unwrap();
    state(&mut process).write_w(x(1), original);
    dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(read_abi_register(&process, 0), 0);
    state(&mut process).write_w(x(0), original);
    dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(read_abi_register(&process, 0), 0);
    assert!(process.handles().lookup(original).is_none());
    for (handle, expected) in [
        (original, HorizonKernelResult::INVALID_HANDLE),
        (duplicate, HorizonKernelResult::SUCCESS),
        (CURRENT_THREAD_HANDLE, HorizonKernelResult::SUCCESS),
    ] {
        for register in [1, 1, 2] {
            state(&mut process).write_w(x(register), handle);
            assert!(matches!(
                dispatch_scheduled_next(&mut process, &mut dispatcher).1,
                ExceptionHandlingResult::Resumed
            ));
            assert_eq!(read_abi_register(&process, 0), u64::from(expected.raw()));
        }
        if handle == original {
            state(&mut process).write_w(x(0), original);
            state(&mut process).write_w(x(1), 20);
            dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(read_abi_register(&process, 0), u64::from(expected.raw()));
        }
    }
    assert_eq!(read_abi_register(&process, 1), 0);
}

#[test]
fn main_thread_retirement_allows_child_execution_and_clean_host_stop_or_shutdown() {
    for close_main in [false, true] {
        for shutdown in [false, true] {
            let (_directory, mut process) = fixture_process(&[
                svc(if close_main { 0x16 } else { 0x10 }),
                svc(0x0a),
                svc(0x10),
                svc(0x0a),
            ]);
            let mut dispatcher = support::dispatcher(&process);
            let entry = instruction_address(&process);
            let main = process.main_thread_id();
            let handle = process.main_thread().handle;
            let (child, _) = super::mutex::add_thread(&mut process, entry + 8, 45);
            let process_id = process.scheduler_process_id();
            process.coordinator_mut().start_thread(child.get()).unwrap();
            state(&mut process).write_w(x(0), handle);
            dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert!(
                matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
                (id, ExceptionHandlingResult::Terminated { .. }) if id == main)
            );
            let exit = process.thread(main).unwrap().exit().unwrap();
            assert!(
                matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
                (id, ExceptionHandlingResult::Resumed) if id == child)
            );
            assert_eq!(process.thread(main).is_none(), close_main);
            assert_eq!(
                process.lifecycle(),
                nixe_scheduler::ProcessLifecycle::Running
            );
            if shutdown {
                process.coordinator_mut().shutdown().unwrap();
                assert_eq!(
                    process.coordinator_mut().resource_counts(),
                    Default::default()
                );
            } else {
                assert!(
                    process
                        .coordinator_mut()
                        .terminate_process(process_id)
                        .unwrap()
                );
                assert_eq!(
                    process.lifecycle(),
                    nixe_scheduler::ProcessLifecycle::Exited
                );
                assert_eq!(
                    process
                        .thread(child)
                        .unwrap()
                        .exit()
                        .unwrap()
                        .requested_scope,
                    nixe_runtime::ExceptionTerminationScope::Process
                );
                if !close_main {
                    assert_eq!(process.thread(main).unwrap().exit(), Some(exit));
                }
                assert!(
                    !process
                        .coordinator_mut()
                        .terminate_process(process_id)
                        .unwrap()
                );
                process.coordinator_mut().shutdown().unwrap();
            }
        }
    }
}

#[test]
fn last_started_thread_exit_ignores_created_and_retained_exited_children() {
    for close_created in [false, true] {
        let (_directory, mut process) = fixture_process(&[svc(0x0a)]);
        let mut dispatcher = support::dispatcher(&process);
        let main = process.main_thread_id();
        let entry = instruction_address(&process);
        let (created, copied) = super::mutex::add_thread(&mut process, entry, 30);
        let created_object = process.thread(created).unwrap().object();
        if close_created {
            let handle = process.thread(created).unwrap().handle;
            drop(process.handles().close(handle).unwrap());
            drop(process.handles().close(copied).unwrap());
            drop(created_object);
            process.coordinator_mut().drain_external_events().unwrap();
            assert!(process.thread(created).is_none());
        } else {
            drop(created_object);
        }
        let (exited, _) = super::mutex::add_thread(&mut process, entry, 30);
        process
            .coordinator_mut()
            .start_thread(exited.get())
            .unwrap();
        assert!(
            matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
            (id, ExceptionHandlingResult::Terminated { .. }) if id == exited)
        );
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Running
        );
        assert!(
            matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
            (id, ExceptionHandlingResult::Terminated { .. }) if id == main)
        );
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Exited
        );
        assert_eq!(
            process.exit().unwrap().cause,
            nixe_runtime::ProcessExitCause::LastThreadExited
        );
        if !close_created {
            assert!(process.thread(created).unwrap().object().is_signalled());
            assert_eq!(
                process
                    .coordinator_mut()
                    .scheduler()
                    .thread(created)
                    .unwrap()
                    .lifecycle,
                nixe_scheduler::ThreadLifecycle::Exited
            );
        }
    }
}

#[test]
fn started_waiting_or_paused_children_keep_the_process_alive() {
    for paused in [false, true] {
        let (_directory, mut process) = fixture_process(&[svc(0x0a), svc(0x0b), svc(0x0a)]);
        let mut dispatcher = support::dispatcher(&process);
        let main = process.main_thread_id();
        let entry = instruction_address(&process);
        let (child, _) = super::mutex::add_thread(&mut process, entry + 4, 30);
        process
            .thread_mut(child)
            .unwrap()
            .state_mut()
            .write_x(x(0), 1_000_000_000);
        process.coordinator_mut().start_thread(child.get()).unwrap();
        if paused {
            process
                .coordinator_mut()
                .scheduler()
                .set_activity(child, true)
                .unwrap();
        } else {
            assert!(
                matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
                (id, ExceptionHandlingResult::Resumed) if id == child)
            );
            assert_eq!(
                process
                    .coordinator_mut()
                    .scheduler()
                    .thread(child)
                    .unwrap()
                    .lifecycle,
                nixe_scheduler::ThreadLifecycle::Waiting
            );
        }
        assert!(
            matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
            (id, ExceptionHandlingResult::Terminated { .. }) if id == main)
        );
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Running
        );
        if paused {
            process
                .coordinator_mut()
                .scheduler()
                .set_activity(child, false)
                .unwrap();
            dispatch_scheduled_next(&mut process, &mut dispatcher);
        }
        assert!(
            matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
            (id, ExceptionHandlingResult::Terminated { .. }) if id == child)
        );
        assert_eq!(
            process.lifecycle(),
            nixe_scheduler::ProcessLifecycle::Exited
        );
    }
}

#[test]
fn exited_thread_handles_and_pending_operations_retain_one_canonical_object() {
    for retain_operation in [false, true] {
        let (_directory, mut process) =
            fixture_process(&[svc(0x16), svc(0x0c), svc(0x16), svc(0x10), svc(0x0a)]);
        let mut dispatcher = support::dispatcher(&process);
        let entry = instruction_address(&process);
        let (child, copied) = super::mutex::add_thread(&mut process, entry + 16, 30);
        let original = process.thread(child).unwrap().handle;
        assert!(
            process
                .handles()
                .lookup(original)
                .unwrap()
                .same_identity(&process.handles().lookup(copied).unwrap())
        );
        let retained = retain_operation.then(|| process.thread(child).unwrap().object());
        state(&mut process).write_w(x(0), copied);
        process.coordinator_mut().start_thread(child.get()).unwrap();
        assert!(
            matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Terminated { .. }) if id == child)
        );
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0);
        assert!(
            process
                .coordinator_mut()
                .scheduler()
                .thread(child)
                .is_some()
        );
        state(&mut process).write_w(x(1), original);
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0);
        assert_eq!(read_abi_register(&process, 1), 30);
        state(&mut process).write_w(x(0), original);
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0);
        process.coordinator_mut().drain_external_events().unwrap();
        assert_eq!(
            process
                .coordinator_mut()
                .scheduler()
                .thread(child)
                .is_some(),
            retain_operation
        );
        if let Some(retained) = retained {
            assert!(retained.is_signalled());
            std::thread::spawn(move || drop(retained)).join().unwrap();
            process.coordinator_mut().drain_external_events().unwrap();
        }
        assert!(
            process
                .coordinator_mut()
                .scheduler()
                .thread(child)
                .is_none()
        );
        assert!(process.thread(child).is_none());
    }
}

#[test]
fn closing_the_last_handle_finalizes_an_unstarted_thread() {
    let (_directory, mut process) = fixture_process(&[svc(0x16), svc(0x10)]);
    let mut dispatcher = support::dispatcher(&process);
    let entry = instruction_address(&process);
    let (child, copied) = super::mutex::add_thread(&mut process, entry + 4, 30);
    let original = process.thread(child).unwrap().handle;
    drop(process.handles().close(copied).unwrap());
    state(&mut process).write_w(x(0), original);
    assert!(matches!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    process.coordinator_mut().drain_external_events().unwrap();
    assert!(process.thread(child).is_none());
    assert!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(child)
            .is_none()
    );
}

#[test]
fn native_priority_queries_follow_donation_and_base_priority_restoration() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x0c),
        svc(0x0c),
        svc(0x0c),
        svc(0x1b),
        svc(0x0c),
        svc(0x0c),
        svc(0x10),
        svc(0x1a),
        svc(0x0a),
    ]);
    let mut dispatcher = support::dispatcher(&process);
    let owner = process.main_thread_id();
    let owner_handle = process.main_thread().handle;
    let mutex = process.main_thread().stack_bottom;
    let entry = instruction_address(&process);
    let (donor, tag) = super::mutex::add_thread(&mut process, entry + 28, 30);
    write_guest_bytes(&process, mutex, &(owner_handle | (1 << 30)).to_le_bytes());
    let donor_state = process.thread_mut(donor).unwrap().state_mut();
    donor_state.write_w(x(0), owner_handle);
    donor_state.write_x(x(1), mutex.get());
    donor_state.write_w(x(2), tag);
    let query = |process: &mut ScheduledProcess,
                 dispatcher: &mut HorizonSvcDispatcher,
                 expected| {
        state(process).write_w(x(1), owner_handle);
        assert!(
            matches!(dispatch_scheduled_next(process, dispatcher), (id, ExceptionHandlingResult::Resumed) if id == owner)
        );
        assert_eq!(read_abi_register(process, 0), 0);
        assert_eq!(read_abi_register(process, 1), expected);
    };
    query(&mut process, &mut dispatcher, 44);
    process.coordinator_mut().start_thread(donor.get()).unwrap();
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Suspended) if id == donor)
    );
    query(&mut process, &mut dispatcher, 30);
    process
        .coordinator_mut()
        .set_thread_priority(Some(owner), owner.get(), 52)
        .unwrap();
    query(&mut process, &mut dispatcher, 30);
    state(&mut process).write_x(x(0), mutex.get());
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Resumed) if id == owner)
    );
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Resumed) if id == donor)
    );
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Terminated { .. }) if id == donor)
    );
    query(&mut process, &mut dispatcher, 52);
    process
        .coordinator_mut()
        .set_thread_priority(Some(owner), owner.get(), 44)
        .unwrap();
    query(&mut process, &mut dispatcher, 44);
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .selection(VirtualCpuId::new(0))
            .unwrap()
            .thread,
        Some(owner)
    );
}
