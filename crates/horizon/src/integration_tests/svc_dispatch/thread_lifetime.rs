use super::*;
use nixe_scheduler::VirtualCpuId;

#[test]
fn start_thread_resolves_the_current_thread_pseudohandle_like_a_retained_thread() {
    let (_directory, mut process) = fixture_process(&[svc(0x09); 6]);
    let mut dispatcher = support::dispatcher(&process);
    let handle = process.main_thread_handle();
    let duplicate = process.handles().duplicate(handle).unwrap();
    let event = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    for (handle, expected) in [
        (handle, HorizonKernelResult::INVALID_STATE),
        (CURRENT_THREAD_HANDLE, HorizonKernelResult::INVALID_STATE),
        (duplicate, HorizonKernelResult::INVALID_STATE),
        (event, HorizonKernelResult::INVALID_HANDLE),
        (CURRENT_PROCESS_HANDLE, HorizonKernelResult::INVALID_HANDLE),
        (0, HorizonKernelResult::INVALID_HANDLE),
    ] {
        write_abi_register(&process, 0, u64::from(handle));
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(expected.raw()),
            "StartThread handle {handle:#x}"
        );
    }
}

#[test]
fn get_info_accepts_process_handles_and_rejects_other_objects_and_closed_handles() {
    let (_directory, mut process) = fixture_process(&[svc(0x29); 8]);
    let mut dispatcher = support::dispatcher(&process);
    let owner = std::sync::Arc::clone(&process);
    let handle = process
        .handles()
        .insert_object(nixe_runtime::HandleObject::from_arc(owner))
        .unwrap();
    let duplicate = process.handles().duplicate(handle).unwrap();
    let closed = process.handles().duplicate(handle).unwrap();
    process.handles().close(closed).unwrap();
    let event = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    for (handle, valid) in [
        (CURRENT_PROCESS_HANDLE, true),
        (handle, true),
        (duplicate, true),
        (event, false),
        (process.main_thread_handle(), false),
        (CURRENT_THREAD_HANDLE, false),
        (closed, false),
        (0, false),
    ] {
        for (index, value) in [(1, 4), (2, u64::from(handle)), (3, 0)] {
            write_abi_register(&process, index, value);
        }
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(
                if valid {
                    HorizonKernelResult::SUCCESS
                } else {
                    HorizonKernelResult::INVALID_HANDLE
                }
                .raw()
            )
        );
        if valid {
            assert_eq!(
                read_abi_register(&process, 1),
                process.memory_layout().heap().base().get()
            );
        }
    }
}

#[test]
fn get_info_queries_the_referenced_process_instead_of_the_caller() {
    let (directory, mut process) = fixture_process(&[svc(0x29); 17]);
    let mut dispatcher = support::dispatcher(&process);
    let plan = Launcher::build(LauncherInput::new(directory.path().join("svc.nro"))).unwrap();
    let target = support::process_builder_with_kernel(process.scheduler().clone())
        .with_config(ProcessBuildConfig {
            physical_memory_limit: 128 * 1024 * 1024,
            image_base: GuestVirtualAddress::new(0x7300_0000),
            ..Default::default()
        })
        .build(&plan)
        .unwrap();
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let id = process
        .coordinator_mut()
        .register_process(
            target,
            nixe_runtime::ProcessRegistration {
                priority: 45,
                ideal_vcpu: Some(VirtualCpuId::new(0)),
                affinity,
            },
        )
        .unwrap();
    let target = process.coordinator_mut().process(id).unwrap().clone();
    target
        .scheduler()
        .set_activity(target.main_thread_id(), true)
        .unwrap();
    support::ready(target.memory().resize_heap(0x1_0000)).unwrap();
    assert_ne!(
        target.total_user_physical_memory_size(),
        process.total_user_physical_memory_size()
    );
    assert_ne!(
        target.used_user_physical_memory_size(),
        process.used_user_physical_memory_size()
    );
    let handle = process
        .handles()
        .insert_object(nixe_runtime::HandleObject::from_arc(target.clone()))
        .unwrap();
    let layout = target.memory_layout();
    let cases = [
        (2, layout.alias().base().get()),
        (3, layout.alias().size()),
        (4, layout.heap().base().get()),
        (5, layout.heap().size()),
        (6, target.total_user_physical_memory_size()),
        (7, target.used_user_physical_memory_size()),
        (12, layout.aslr().base().get()),
        (13, layout.aslr().size()),
        (14, layout.stack().base().get()),
        (15, layout.stack().size()),
        (16, target.total_system_resource_size()),
        (17, target.used_system_resource_size()),
        (20, target.user_exception_context_address().get()),
        (21, target.total_non_system_user_physical_memory_size()),
        (22, target.used_non_system_user_physical_memory_size()),
        (28, 0),
    ];
    drop(target);
    for (kind, expected) in cases {
        for (index, value) in [(1, kind), (2, u64::from(handle)), (3, 0)] {
            write_abi_register(&process, index, value);
        }
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0, "GetInfo kind {kind}");
        assert_eq!(
            read_abi_register(&process, 1),
            expected,
            "GetInfo kind {kind}"
        );
    }
    process.coordinator_mut().remove_process(id).unwrap();
    for (index, value) in [(1, 4), (2, u64::from(handle)), (3, 0)] {
        write_abi_register(&process, index, value);
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_abi_register(&process, 1), layout.heap().base().get());
}

#[test]
fn process_id_accepts_process_thread_and_pseudo_handles() {
    let (_directory, mut process) = fixture_process(&[svc(0x24); 6]);
    let mut dispatcher = support::dispatcher(&process);
    let id = process.scheduler_process_id();
    let owner = process.coordinator_mut().process(id).unwrap().clone();
    let process_handle = owner
        .handles()
        .insert_object(nixe_runtime::HandleObject::from_arc(owner.clone()))
        .unwrap();
    let thread_handle = owner
        .handles()
        .duplicate(process.main_thread_handle())
        .unwrap();
    let event = owner
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    for (handle, valid) in [
        (CURRENT_PROCESS_HANDLE, true),
        (CURRENT_THREAD_HANDLE, true),
        (process_handle, true),
        (thread_handle, true),
        (event, false),
        (0, false),
    ] {
        process.main_thread().registers().write_w(x(1), handle);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(
                if valid {
                    HorizonKernelResult::SUCCESS
                } else {
                    HorizonKernelResult::INVALID_HANDLE
                }
                .raw()
            )
        );
        if valid {
            assert_eq!(read_abi_register(&process, 1), id.get());
        }
    }
}

#[test]
fn process_id_of_an_exported_thread_survives_removal_of_its_process() {
    use std::sync::Arc;
    let (directory, mut process) = fixture_process(&[svc(0x24)]);
    let mut dispatcher = support::dispatcher(&process);
    let plan = Launcher::build(LauncherInput::new(directory.path().join("svc.nro"))).unwrap();
    let source = support::process_builder_with_kernel(process.scheduler().clone())
        .build(&plan)
        .unwrap();
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let id = process
        .coordinator_mut()
        .register_process(
            source,
            nixe_runtime::ProcessRegistration {
                priority: 45,
                ideal_vcpu: Some(VirtualCpuId::new(0)),
                affinity,
            },
        )
        .unwrap();
    let source = process.coordinator_mut().process(id).unwrap().clone();
    let parent = Arc::downgrade(&source);
    let handle = process
        .handles()
        .insert_object(nixe_runtime::HandleObject::from_arc(source.main_thread()))
        .unwrap();
    drop(source);
    process.coordinator_mut().remove_process(id).unwrap();
    assert!(parent.upgrade().is_some());
    process.main_thread().registers().write_w(x(1), handle);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_abi_register(&process, 1), id.get());
    drop(process.handles().close(handle).unwrap());
    assert!(parent.upgrade().is_none());
}

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
    let original = process.main_thread_handle();
    let duplicate = process.handles().duplicate(original).unwrap();
    process.main_thread().registers().write_w(x(1), original);
    dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(read_abi_register(&process, 0), 0);
    process.main_thread().registers().write_w(x(0), original);
    dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(read_abi_register(&process, 0), 0);
    assert!(process.handles().lookup(original).is_none());
    for (handle, expected) in [
        (original, HorizonKernelResult::INVALID_HANDLE),
        (duplicate, HorizonKernelResult::SUCCESS),
        (CURRENT_THREAD_HANDLE, HorizonKernelResult::SUCCESS),
    ] {
        for register in [1, 1, 2] {
            process
                .main_thread()
                .registers()
                .write_w(x(register), handle);
            assert!(matches!(
                dispatch_scheduled_next(&mut process, &mut dispatcher).1,
                ExceptionHandlingResult::Resumed
            ));
            assert_eq!(read_abi_register(&process, 0), u64::from(expected.raw()));
        }
        if handle == original {
            process.main_thread().registers().write_w(x(0), original);
            process.main_thread().registers().write_w(x(1), 20);
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
            let handle = process.main_thread_handle();
            let (child, _) = super::mutex::add_thread(&mut process, entry + 8, 45);
            let process_id = process.scheduler_process_id();
            process.thread(child).unwrap().start().unwrap();
            process.main_thread().registers().write_w(x(0), handle);
            dispatch_scheduled_next(&mut process, &mut dispatcher);
            let main_object = process.main_thread();
            assert!(
                matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
                (id, ExceptionHandlingResult::Terminated { .. }) if id == main)
            );
            let exit = main_object.exit().unwrap();
            drop(main_object);
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
                let retained_child = process.thread(child).unwrap();
                let retained_main = process.thread(main);
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
                    retained_child.exit().unwrap().requested_scope,
                    nixe_runtime::ExceptionTerminationScope::Process
                );
                if !close_main {
                    assert_eq!(retained_main.unwrap().exit(), Some(exit));
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
        let (created, handle) = super::mutex::add_thread(&mut process, entry, 30);
        let copied = process.handles().duplicate(handle).unwrap();
        let created_object = process.thread(created).unwrap().clone();
        let retained_created = (!close_created).then(|| created_object.clone());
        if close_created {
            drop(process.handles().close(handle).unwrap());
            drop(process.handles().close(copied).unwrap());
            drop(created_object);
            process.coordinator_mut().drain_external_events().unwrap();
            assert!(process.thread(created).is_none());
        } else {
            drop(created_object);
        }
        let (exited, _) = super::mutex::add_thread(&mut process, entry, 30);
        process.thread(exited).unwrap().start().unwrap();
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
            assert!(retained_created.as_ref().unwrap().is_signalled());
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
            .thread(child)
            .unwrap()
            .registers()
            .write_x(x(0), 1_000_000_000);
        process.thread(child).unwrap().start().unwrap();
        if paused {
            process
                .coordinator_mut()
                .scheduler()
                .set_activity(child, true)
                .unwrap();
        } else {
            assert!(
                matches!(dispatch_scheduled_next(&mut process, &mut dispatcher),
                (id, ExceptionHandlingResult::Suspended) if id == child)
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
        process
            .coordinator_mut()
            .advance_virtual_time(1_000_000_000)
            .unwrap();
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
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
        let (child, original) = super::mutex::add_thread(&mut process, entry + 16, 30);
        let copied = process.handles().duplicate(original).unwrap();
        assert!(
            process
                .handles()
                .lookup(original)
                .unwrap()
                .same_identity(&process.handles().lookup(copied).unwrap())
        );
        let retained = retain_operation.then(|| process.thread(child).unwrap().clone());
        process.main_thread().registers().write_w(x(0), copied);
        process.thread(child).unwrap().start().unwrap();
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
        process.main_thread().registers().write_w(x(1), original);
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0);
        assert_eq!(read_abi_register(&process, 1), 30);
        process.main_thread().registers().write_w(x(0), original);
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
    let original = copied;
    let copied = process.handles().duplicate(original).unwrap();
    drop(process.handles().close(copied).unwrap());
    process.main_thread().registers().write_w(x(0), original);
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
    let owner_handle = process.main_thread_handle();
    let mutex = process.main_thread().stack_bottom;
    let entry = instruction_address(&process);
    let (donor, tag) = super::mutex::add_thread(&mut process, entry + 28, 30);
    write_guest_bytes(&process, mutex, &(owner_handle | (1 << 30)).to_le_bytes());
    let donor_state_thread = process.thread(donor).unwrap();
    let mut donor_state = donor_state_thread.registers();
    donor_state.write_w(x(0), owner_handle);
    donor_state.write_x(x(1), mutex.get());
    donor_state.write_w(x(2), tag);
    drop(donor_state);
    let query = |process: &mut ScheduledProcess,
                 dispatcher: &mut HorizonSvcDispatcher,
                 expected| {
        process
            .main_thread()
            .registers()
            .write_w(x(1), owner_handle);
        assert!(
            matches!(dispatch_scheduled_next(process, dispatcher), (id, ExceptionHandlingResult::Resumed) if id == owner)
        );
        assert_eq!(read_abi_register(process, 0), 0);
        assert_eq!(read_abi_register(process, 1), expected);
    };
    query(&mut process, &mut dispatcher, 44);
    process.thread(donor).unwrap().start().unwrap();
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Suspended) if id == donor)
    );
    query(&mut process, &mut dispatcher, 30);
    process
        .scheduler()
        .set_priority(owner, 52, Some(owner))
        .unwrap();
    query(&mut process, &mut dispatcher, 30);
    process.main_thread().registers().write_x(x(0), mutex.get());
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Resumed) if id == owner)
    );
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert_eq!(process.thread_lifecycle(donor), ThreadLifecycle::Ready);
    assert!(
        matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Terminated { .. }) if id == donor)
    );
    query(&mut process, &mut dispatcher, 52);
    process
        .scheduler()
        .set_priority(owner, 44, Some(owner))
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
