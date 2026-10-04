use super::*;
use nixe_scheduler::{GuestThreadId, VirtualCpuId};

const WAITERS: u32 = 1 << 30;

fn add_thread(process: &mut ScheduledProcess, entry: u64, priority: i32) -> (GuestThreadId, u32) {
    let process_id = process.scheduler_process_id();
    let stack_top = process.main_thread().stack_top;
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let child = process
        .coordinator_mut()
        .create_thread(
            process_id,
            nixe_runtime::ThreadCreateRequest {
                entry: GuestVirtualAddress::new(entry),
                argument: 0,
                stack_top,
                priority,
                ideal_vcpu: Some(VirtualCpuId::new(0)),
                affinity,
            },
        )
        .unwrap();
    let object = process.thread(child.id).unwrap().object().clone();
    let handle = process.handles_mut().insert(object).unwrap();
    (child.id, handle)
}

fn start(process: &mut ScheduledProcess, thread: GuestThreadId) {
    let object = process.thread(thread).unwrap().object().thread_id();
    process.coordinator_mut().start_thread(object).unwrap();
}

fn lock_arguments(state: &mut A64State, owner: u32, mutex: u64, tag: u32) {
    state.write_w(x(0), owner);
    state.write_x(x(1), mutex);
    state.write_w(x(2), tag);
}

#[test]
fn mutex_owner_change_returns_without_stealing_or_waiting_on_a_stale_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x1a), svc(0x1a), svc(0x1a)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let mutex = process.main_thread().stack_bottom;
    for word in [0, 0x1234, WAITERS | 0x5678] {
        write_guest_bytes(&process, mutex, &word.to_le_bytes());
        lock_arguments(state(&mut process), 0xbad, mutex.get(), 1);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), 0);
        assert_eq!(read_guest_u32(&process, mutex), word);
        assert_eq!(process.address_waits().waiter_count(), 0);
    }
}

#[test]
fn mutex_handoff_keeps_remaining_waiters_and_transfers_priority_before_waking() {
    let (_directory, mut process) =
        fixture_process(&[svc(0x1b), svc(0x0a), svc(0x1a), svc(0x1b), svc(0x0a)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let mutex = process.main_thread().stack_bottom;
    let main = process.main_thread_id();
    let main_object = process.main_thread().object().clone();
    let owner = process.handles_mut().insert(main_object).unwrap();
    let entry = state(&mut process).pc() + 8;
    write_guest_bytes(&process, mutex, &(owner | WAITERS).to_le_bytes());
    let mut children = Vec::new();
    for priority in [40, 30] {
        let (child, handle) = add_thread(&mut process, entry, priority);
        lock_arguments(
            process.thread_mut(child).unwrap().state_mut(),
            owner,
            mutex.get(),
            handle,
        );
        start(&mut process, child);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (child, ExceptionHandlingResult::Suspended)
        );
        children.push((child, handle));
    }
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(main)
            .unwrap()
            .effective_priority,
        30
    );
    state(&mut process).write_x(x(0), mutex.get());
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (main, ExceptionHandlingResult::Resumed)
    );
    let (next, next_tag) = children[1];
    let (last, last_tag) = children[0];
    assert_eq!(read_guest_u32(&process, mutex), next_tag | WAITERS);
    assert!(process.address_waits().is_signalled(mutex.get(), next));
    assert!(!process.address_waits().is_signalled(mutex.get(), last));
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(main)
            .unwrap()
            .effective_priority,
        44
    );
    let last_object = process.thread(last).unwrap().object().thread_id();
    process
        .coordinator_mut()
        .set_thread_priority(last_object, 20)
        .unwrap();
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(next)
            .unwrap()
            .effective_priority,
        20
    );
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (next, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(process.thread(next).unwrap().state().read_w(x(0)), 0);
    assert_eq!(read_guest_u32(&process, mutex), next_tag | WAITERS);
    process
        .thread_mut(next)
        .unwrap()
        .state_mut()
        .write_x(x(0), mutex.get());
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (next, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(read_guest_u32(&process, mutex), last_tag);
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(next)
            .unwrap()
            .effective_priority,
        30
    );
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (last, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(process.thread(last).unwrap().state().read_w(x(0)), 0);
    process
        .thread_mut(last)
        .unwrap()
        .state_mut()
        .write_x(x(0), mutex.get());
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (last, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(read_guest_u32(&process, mutex), 0);
    assert_eq!(process.address_waits().waiter_count(), 0);
}

#[test]
fn condition_signal_waits_for_the_held_mutex_and_preserves_the_original_deadline() {
    let (_directory, mut process) =
        fixture_process(&[svc(0x1d), svc(0x1b), svc(0x0a), svc(0x1c), svc(0x0a)]);
    let mut dispatcher = fixed_time_dispatcher();
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let main = process.main_thread_id();
    let main_object = process.main_thread().object().clone();
    let owner = process.handles_mut().insert(main_object).unwrap();
    let entry = state(&mut process).pc() + 12;
    let (child, tag) = add_thread(&mut process, entry, 30);
    let child_state = process.thread_mut(child).unwrap().state_mut();
    child_state.write_x(x(0), mutex.get());
    child_state.write_x(x(1), key.get());
    child_state.write_w(x(2), tag);
    child_state.write_x(x(3), 2_000_000);
    write_guest_bytes(&process, mutex, &tag.to_le_bytes());
    start(&mut process, child);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child, ExceptionHandlingResult::Suspended)
    );
    assert_eq!(read_guest_u32(&process, mutex), 0);
    // Another thread takes the free mutex before signaling the condition.
    write_guest_bytes(&process, mutex, &owner.to_le_bytes());
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), 1);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (main, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(read_guest_u32(&process, mutex), owner | WAITERS);
    assert_eq!(read_guest_u32(&process, key), 0);
    assert!(!process.address_waits().is_signalled(mutex.get(), child));
    assert!(!process.address_waits().contains(key.get(), child));
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(main)
            .unwrap()
            .effective_priority,
        30
    );
    process
        .coordinator_mut()
        .advance_virtual_time(1_000_000)
        .unwrap();
    assert_eq!(
        process.thread_lifecycle(child),
        nixe_scheduler::ThreadLifecycle::Waiting
    );
    process
        .coordinator_mut()
        .advance_virtual_time(1_000_000)
        .unwrap();
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(
        process.thread(child).unwrap().state().read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert_eq!(read_guest_u32(&process, mutex), owner | WAITERS);
    assert_eq!(process.address_waits().waiter_count(), 0);
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(main)
            .unwrap()
            .effective_priority,
        44
    );
}

#[test]
fn condition_signal_claims_a_free_mutex_before_the_waiter_runs() {
    let (_directory, mut process) = fixture_process(&[svc(0x1d), svc(0x0a), svc(0x1c), svc(0x0a)]);
    let mut dispatcher = fixed_time_dispatcher();
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let entry = state(&mut process).pc() + 8;
    let (child, tag) = add_thread(&mut process, entry, 30);
    let child_state = process.thread_mut(child).unwrap().state_mut();
    child_state.write_x(x(0), mutex.get());
    child_state.write_x(x(1), key.get());
    child_state.write_w(x(2), tag);
    child_state.write_x(x(3), u64::MAX);
    write_guest_bytes(&process, mutex, &tag.to_le_bytes());
    start(&mut process, child);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child, ExceptionHandlingResult::Suspended)
    );
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), 1);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, mutex), tag);
    assert_eq!(read_guest_u32(&process, key), 0);
    assert!(process.address_waits().is_signalled(key.get(), child));
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(process.thread(child).unwrap().state().read_w(x(0)), 0);
    assert_eq!(read_guest_u32(&process, mutex), tag);
    assert_eq!(process.address_waits().waiter_count(), 0);
}

#[test]
fn condition_signals_keep_the_key_set_until_all_waiters_reacquire_the_mutex() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x1d),
        svc(0x1d),
        svc(0x1b),
        svc(0x0a),
        svc(0x1c),
        svc(0x1b),
        svc(0x0a),
    ]);
    let mut dispatcher = fixed_time_dispatcher();
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let main = process.main_thread_id();
    let main_object = process.main_thread().object().clone();
    let owner = process.handles_mut().insert(main_object).unwrap();
    let entry = state(&mut process).pc() + 16;
    let mut children = Vec::new();
    for priority in [40, 30] {
        let (child, tag) = add_thread(&mut process, entry, priority);
        let child_state = process.thread_mut(child).unwrap().state_mut();
        child_state.write_x(x(0), mutex.get());
        child_state.write_x(x(1), key.get());
        child_state.write_w(x(2), tag);
        child_state.write_x(x(3), u64::MAX);
        write_guest_bytes(&process, mutex, &tag.to_le_bytes());
        start(&mut process, child);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (child, ExceptionHandlingResult::Suspended)
        );
        children.push((child, tag));
    }
    write_guest_bytes(&process, mutex, &owner.to_le_bytes());
    for (count, expected_key) in [(1_u32, 1_u32), (u32::MAX, 0)] {
        state(&mut process).write_x(x(0), key.get());
        state(&mut process).write_w(x(1), count);
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (main, ExceptionHandlingResult::Resumed)
        );
        assert_eq!(read_guest_u32(&process, key), expected_key);
        assert_eq!(read_guest_u32(&process, mutex), owner | WAITERS);
        assert!(process.address_waits().contains(mutex.get(), children[1].0));
        assert_eq!(
            process.address_waits().contains(key.get(), children[0].0),
            expected_key != 0
        );
    }
    state(&mut process).write_x(x(0), mutex.get());
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (main, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(read_guest_u32(&process, mutex), children[1].1 | WAITERS);
    for (index, expected_word) in [(1, children[0].1), (0, 0)] {
        let (child, _) = children[index];
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (child, ExceptionHandlingResult::Resumed)
        );
        assert_eq!(process.thread(child).unwrap().state().read_w(x(0)), 0);
        process
            .thread_mut(child)
            .unwrap()
            .state_mut()
            .write_x(x(0), mutex.get());
        assert_eq!(
            dispatch_scheduled_next(&mut process, &mut dispatcher),
            (child, ExceptionHandlingResult::Resumed)
        );
        assert_eq!(read_guest_u32(&process, mutex), expected_word);
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Terminated { .. }
        ));
    }
    assert_eq!(process.address_waits().waiter_count(), 0);
}

#[test]
fn condition_signal_reports_an_invalid_mutex_owner_to_the_waiter() {
    let (_directory, mut process) = fixture_process(&[svc(0x1d), svc(0x0a), svc(0x1c), svc(0x0a)]);
    let mut dispatcher = fixed_time_dispatcher();
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let entry = state(&mut process).pc() + 8;
    let (child, tag) = add_thread(&mut process, entry, 30);
    let child_state = process.thread_mut(child).unwrap().state_mut();
    child_state.write_x(x(0), mutex.get());
    child_state.write_x(x(1), key.get());
    child_state.write_w(x(2), tag);
    child_state.write_x(x(3), u64::MAX);
    write_guest_bytes(&process, mutex, &tag.to_le_bytes());
    start(&mut process, child);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child, ExceptionHandlingResult::Suspended)
    );
    write_guest_bytes(&process, mutex, &0xbad_u32.to_le_bytes());
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), 1);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, mutex), 0xbad | WAITERS);
    assert_eq!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (child, ExceptionHandlingResult::Resumed)
    );
    assert_eq!(
        process.thread(child).unwrap().state().read_w(x(0)),
        HorizonKernelResult::INVALID_STATE.raw()
    );
    assert_eq!(process.address_waits().waiter_count(), 0);
}
