use super::*;

#[test]
fn pending_cancellation_survives_polling_and_is_consumed_by_one_wait() {
    let (_directory, mut process) = fixture_process(&[svc(0x19), svc(0x18), svc(0x18), svc(0x18)]);
    let mut dispatcher = fixed_time_dispatcher();
    state(&mut process).write_w(x(0), CURRENT_THREAD_HANDLE);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(state(&mut process).read_w(x(0)), 0);
    state(&mut process).write_x(x(1), 0);
    state(&mut process).write_w(x(2), 0);
    state(&mut process).write_x(x(3), 0);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    state(&mut process).write_x(x(3), u64::MAX);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::CANCELLED.raw()
    );
    assert_eq!(state(&mut process).read_w(x(1)), u32::MAX);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Suspended
    );
}

#[test]
fn cancellation_wakes_another_waiting_thread_and_releases_its_deadline() {
    let (_directory, mut process) = fixture_process(&[svc(0x19), svc(0x0a), svc(0x18), svc(0x0a)]);
    let entry = GuestVirtualAddress::new(state(&mut process).pc() + 8);
    let stack_top = process.main_thread().stack_top;
    let process_id = process.scheduler_process_id();
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let child = process
        .coordinator_mut()
        .create_thread(
            process_id,
            nixe_runtime::ThreadCreateRequest {
                entry,
                argument: 0,
                stack_top,
                priority: 20,
                ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
                affinity,
            },
        )
        .unwrap();
    let child_state = process.thread_mut(child.id).unwrap().state_mut();
    child_state.write_x(x(1), 0);
    child_state.write_w(x(2), 0);
    child_state.write_x(x(3), 1_000_000);
    let object_id = process.thread(child.id).unwrap().object().thread_id();
    process.coordinator_mut().start_thread(object_id).unwrap();
    let mut dispatcher = fixed_time_dispatcher();
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child.id, ExceptionHandlingResult::Suspended)
    );
    assert!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(child.id)
            .unwrap()
            .active_wait
            .is_some()
    );
    state(&mut process).write_w(x(0), child.handle);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child.id, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(
        process.thread(child.id).unwrap().state().read_w(x(0)),
        HorizonKernelResult::CANCELLED.raw()
    );
    assert_eq!(
        process
            .coordinator_mut()
            .advance_virtual_time(2_000_000)
            .unwrap(),
        0
    );
}

#[test]
fn cancellation_rejects_a_non_thread_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x19)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    state(&mut process).write_w(x(0), CURRENT_PROCESS_HANDLE);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::INVALID_HANDLE.raw()
    );
}
