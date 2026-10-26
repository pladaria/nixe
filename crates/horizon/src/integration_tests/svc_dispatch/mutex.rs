use super::*;
use nixe_scheduler::{GuestThreadId, VirtualCpuId};

const WAITERS: u32 = 1 << 30;

fn add_thread(process: &mut ScheduledProcess, entry: u64, priority: i32) -> (GuestThreadId, u32) {
    let process_id = process.scheduler_process_id();
    let stack_top = process.main_thread().stack_top;
    // Arbitration order is a per-core contract. Keep these contenders on core
    // zero while using the same parallel coordinator as normal execution.
    let affinity = process
        .coordinator_mut()
        .scheduler()
        .profile()
        .core_set([VirtualCpuId::new(0)])
        .unwrap();
    let main = process.main_thread_id();
    process
        .coordinator_mut()
        .migrate_thread(main, Some(VirtualCpuId::new(0)), affinity.clone())
        .unwrap();
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
    let handle = process.handles().insert(object).unwrap();
    (child.id, handle)
}

fn start(process: &mut ScheduledProcess, thread: GuestThreadId) {
    let object = process.thread(thread).unwrap().object().thread_id();
    process.coordinator_mut().start_thread(object).unwrap();
}

fn lock_arguments(state: &mut nixe_runtime::GuestCpuState, owner: u32, mutex: u64, tag: u32) {
    state.write_w(x(0), owner);
    state.write_x(x(1), mutex);
    state.write_w(x(2), tag);
}

#[test]
fn pending_condition_release_precedes_a_later_mutex_acquisition() {
    pending_release_and_acquisition(true, false);
}
#[test]
fn pending_mutex_unlock_precedes_a_later_acquisition() {
    pending_release_and_acquisition(false, false);
}
#[test]
fn pending_mutex_acquisition_registers_before_condition_release() {
    pending_release_and_acquisition(true, true);
}
#[test]
fn pending_mutex_acquisition_registers_before_unlock() {
    pending_release_and_acquisition(false, true);
}

fn pending_release_and_acquisition(condition: bool, acquire_first: bool) {
    use nixe_memory::{DeviceAccessEndpoint, DeviceAccessRequest};
    use std::sync::{Arc, mpsc};

    struct Device(mpsc::Sender<DeviceAccessRequest>);
    impl DeviceAccessEndpoint for Device {
        fn submit(&self, request: DeviceAccessRequest) {
            self.0.send(request).unwrap();
        }
        fn cancel(&self, _: nixe_memory::AccessRequestId) {}
        fn allocation_released(&self, _: nixe_memory::MachineId, _: nixe_memory::AllocationId) {}
        fn physical_retired(
            &self,
            _: nixe_memory::MachineId,
            _: nixe_memory::AllocationId,
            _: Box<[nixe_memory::Extent]>,
        ) {
        }
        fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
    }
    let (_directory, mut process) = fixture_process(&[
        svc(if condition { 0x1c } else { 0x1b }),
        svc(0x0a),
        svc(0x1a),
        svc(0x0a),
    ]);
    let ingress = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_memory_completions(ingress);
    let mutex = process.main_thread().stack_bottom;
    let key = mutex.checked_add(12).unwrap();
    let owner_object = process.main_thread().object().clone();
    let owner = process.handles().insert(owner_object).unwrap();
    let entry = state(&mut process).pc() + 8;
    let (child, tag) = add_thread(&mut process, entry, 30);
    lock_arguments(
        process.thread_mut(child).unwrap().state_mut(),
        owner,
        mutex.get(),
        tag,
    );
    write_guest_bytes(&process, mutex, &(owner | WAITERS).to_le_bytes());
    state(&mut process).write_x(x(0), mutex.get());
    state(&mut process).write_x(x(1), key.get());
    state(&mut process).write_w(x(2), owner);
    state(&mut process).write_x(x(3), u64::MAX);

    let machine = process.memory().space().machine().clone();
    let ranges = process
        .memory()
        .space()
        .capture(mutex, 4, nixe_memory::MemoryPermissions::READ_WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    if acquire_first {
        start(&mut process, child);
    }
    let (selected, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(
        selected,
        if acquire_first {
            child
        } else {
            process.main_thread_id()
        }
    );
    assert!(matches!(handling, ExceptionHandlingResult::Suspended));
    let first = receive
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    if acquire_first {
        assert!(matches!(
            first.operation(),
            nixe_memory::AccessOperation::Read
        ));
    } else {
        assert!(matches!(
            first.operation(),
            nixe_memory::AccessOperation::Write(_)
        ));
        start(&mut process, child);
    }
    let (selected, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(
        selected,
        if acquire_first {
            process.main_thread_id()
        } else {
            child
        }
    );
    assert!(matches!(handling, ExceptionHandlingResult::Suspended));
    let finish = |request: DeviceAccessRequest| {
        let result = machine
            .apply_resolved_access(request.ranges(), request.operation())
            .map_err(Arc::new);
        request.finish(result);
    };
    if !acquire_first {
        // Complete any read of the old owner while release is still pending.
        while let Ok(request) = receive.try_recv() {
            finish(request);
        }
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    finish(first);
    for _ in 0..8 {
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        while let Ok(request) = receive.try_recv() {
            finish(request);
        }
    }
    machine.retire_device_pages(&ranges).unwrap();
    process.coordinator_mut().drain_external_events().unwrap();
    assert_eq!(
        read_guest_u32(&process, mutex),
        if acquire_first { tag } else { 0 }
    );
    let runnable = {
        let waits = process.address_waits().lock();
        !waits.contains(mutex.get(), child) || waits.is_signalled(mutex.get(), child)
    };
    assert!(
        runnable,
        "an acquirer must not remain blocked after the pending mutex release"
    );
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(child)
            .unwrap()
            .lifecycle,
        nixe_scheduler::ThreadLifecycle::Ready
    );
    if acquire_first {
        let (selected, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(selected, child);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    }
    assert_eq!(process.thread(child).unwrap().state().read_w(x(0)), 0);
}

#[test]
fn mutex_owner_change_returns_without_stealing_or_waiting_on_a_stale_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x1a), svc(0x1a), svc(0x1a)]);
    let mut dispatcher = support::dispatcher(&process);
    let mutex = process.main_thread().stack_bottom;
    for word in [0, 0x1234, WAITERS | 0x5678] {
        write_guest_bytes(&process, mutex, &word.to_le_bytes());
        lock_arguments(state(&mut process), 0xbad, mutex.get(), 1);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(state(&mut process).read_w(x(0)), 0);
        assert_eq!(read_guest_u32(&process, mutex), word);
        assert_eq!(process.address_waits().lock().waiter_count(), 0);
    }
}

#[test]
fn mutex_handoff_keeps_remaining_waiters_and_transfers_priority_before_waking() {
    let (_directory, mut process) =
        fixture_process(&[svc(0x1b), svc(0x0a), svc(0x1a), svc(0x1b), svc(0x0a)]);
    let mut dispatcher = support::dispatcher(&process);
    let mutex = process.main_thread().stack_bottom;
    let main = process.main_thread_id();
    let main_object = process.main_thread().object().clone();
    let owner = process.handles().insert(main_object).unwrap();
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
        {
            let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(thread, child);
            assert!(matches!(handling, ExceptionHandlingResult::Suspended));
        };
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
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, main);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    let (next, next_tag) = children[1];
    let (last, last_tag) = children[0];
    assert_eq!(read_guest_u32(&process, mutex), next_tag | WAITERS);
    assert!(
        process
            .address_waits()
            .lock()
            .is_signalled(mutex.get(), next)
    );
    assert!(
        !process
            .address_waits()
            .lock()
            .is_signalled(mutex.get(), last)
    );
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
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, next);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(process.thread(next).unwrap().state().read_w(x(0)), 0);
    assert_eq!(read_guest_u32(&process, mutex), next_tag | WAITERS);
    process
        .thread_mut(next)
        .unwrap()
        .state_mut()
        .write_x(x(0), mutex.get());
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, next);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
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
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, last);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(process.thread(last).unwrap().state().read_w(x(0)), 0);
    process
        .thread_mut(last)
        .unwrap()
        .state_mut()
        .write_x(x(0), mutex.get());
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, last);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(read_guest_u32(&process, mutex), 0);
    assert_eq!(process.address_waits().lock().waiter_count(), 0);
}

#[test]
fn condition_signal_waits_for_the_held_mutex_and_preserves_the_original_deadline() {
    let (_directory, mut process) =
        fixture_process(&[svc(0x1d), svc(0x1b), svc(0x0a), svc(0x1c), svc(0x0a)]);
    let mut dispatcher = fixed_time_dispatcher(&process);
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let main = process.main_thread_id();
    let main_object = process.main_thread().object().clone();
    let owner = process.handles().insert(main_object).unwrap();
    let entry = state(&mut process).pc() + 12;
    let (child, tag) = add_thread(&mut process, entry, 30);
    let child_state = process.thread_mut(child).unwrap().state_mut();
    child_state.write_x(x(0), mutex.get());
    child_state.write_x(x(1), key.get());
    child_state.write_w(x(2), tag);
    child_state.write_x(x(3), 2_000_000);
    write_guest_bytes(&process, mutex, &tag.to_le_bytes());
    start(&mut process, child);
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, child);
        assert!(matches!(handling, ExceptionHandlingResult::Suspended));
    };
    assert_eq!(read_guest_u32(&process, mutex), 0);
    // Another thread takes the free mutex before signaling the condition.
    write_guest_bytes(&process, mutex, &owner.to_le_bytes());
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), 1);
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, main);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(read_guest_u32(&process, mutex), owner | WAITERS);
    assert_eq!(read_guest_u32(&process, key), 0);
    assert!(
        !process
            .address_waits()
            .lock()
            .is_signalled(mutex.get(), child)
    );
    assert!(!process.address_waits().lock().contains(key.get(), child));
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
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, child);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(
        process.thread(child).unwrap().state().read_w(x(0)),
        HorizonKernelResult::TIMED_OUT.raw()
    );
    assert_eq!(read_guest_u32(&process, mutex), owner | WAITERS);
    assert_eq!(process.address_waits().lock().waiter_count(), 0);
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
    let mut dispatcher = fixed_time_dispatcher(&process);
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
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, child);
        assert!(matches!(handling, ExceptionHandlingResult::Suspended));
    };
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), 1);
    assert!(matches!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_guest_u32(&process, mutex), tag);
    assert_eq!(read_guest_u32(&process, key), 0);
    assert!(
        process
            .address_waits()
            .lock()
            .is_signalled(key.get(), child)
    );
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, child);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(process.thread(child).unwrap().state().read_w(x(0)), 0);
    assert_eq!(read_guest_u32(&process, mutex), tag);
    assert_eq!(process.address_waits().lock().waiter_count(), 0);
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
    let mut dispatcher = fixed_time_dispatcher(&process);
    let key = process.main_thread().stack_bottom;
    let mutex = key.checked_add(4).unwrap();
    let main = process.main_thread_id();
    let main_object = process.main_thread().object().clone();
    let owner = process.handles().insert(main_object).unwrap();
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
        {
            let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(thread, child);
            assert!(matches!(handling, ExceptionHandlingResult::Suspended));
        };
        children.push((child, tag));
    }
    write_guest_bytes(&process, mutex, &owner.to_le_bytes());
    for (count, expected_key) in [(1_u32, 1_u32), (u32::MAX, 0)] {
        state(&mut process).write_x(x(0), key.get());
        state(&mut process).write_w(x(1), count);
        {
            let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(thread, main);
            assert!(matches!(handling, ExceptionHandlingResult::Resumed));
        };
        assert_eq!(read_guest_u32(&process, key), expected_key);
        assert_eq!(read_guest_u32(&process, mutex), owner | WAITERS);
        assert!(
            process
                .address_waits()
                .lock()
                .contains(mutex.get(), children[1].0)
        );
        assert_eq!(
            process
                .address_waits()
                .lock()
                .contains(key.get(), children[0].0),
            expected_key != 0
        );
    }
    state(&mut process).write_x(x(0), mutex.get());
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, main);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(read_guest_u32(&process, mutex), children[1].1 | WAITERS);
    for (index, expected_word) in [(1, children[0].1), (0, 0)] {
        let (child, _) = children[index];
        {
            let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(thread, child);
            assert!(matches!(handling, ExceptionHandlingResult::Resumed));
        };
        assert_eq!(process.thread(child).unwrap().state().read_w(x(0)), 0);
        process
            .thread_mut(child)
            .unwrap()
            .state_mut()
            .write_x(x(0), mutex.get());
        {
            let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(thread, child);
            assert!(matches!(handling, ExceptionHandlingResult::Resumed));
        };
        assert_eq!(read_guest_u32(&process, mutex), expected_word);
        assert!(matches!(
            dispatch_scheduled_next(&mut process, &mut dispatcher).1,
            ExceptionHandlingResult::Terminated { .. }
        ));
    }
    assert_eq!(process.address_waits().lock().waiter_count(), 0);
}

#[test]
fn condition_signal_reports_an_invalid_mutex_owner_to_the_waiter() {
    let (_directory, mut process) = fixture_process(&[svc(0x1d), svc(0x0a), svc(0x1c), svc(0x0a)]);
    let mut dispatcher = fixed_time_dispatcher(&process);
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
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, child);
        assert!(matches!(handling, ExceptionHandlingResult::Suspended));
    };
    write_guest_bytes(&process, mutex, &0xbad_u32.to_le_bytes());
    state(&mut process).write_x(x(0), key.get());
    state(&mut process).write_x(x(1), 1);
    assert!(matches!(
        dispatch_scheduled_next(&mut process, &mut dispatcher).1,
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_guest_u32(&process, mutex), 0xbad | WAITERS);
    {
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, child);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    };
    assert_eq!(
        process.thread(child).unwrap().state().read_w(x(0)),
        HorizonKernelResult::INVALID_STATE.raw()
    );
    assert_eq!(process.address_waits().lock().waiter_count(), 0);
}
