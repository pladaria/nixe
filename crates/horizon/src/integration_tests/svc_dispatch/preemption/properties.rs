use super::*;

#[test]
fn a_registered_affinity_pin_wait_survives_sleep_until_actual_unpin() {
    let (_directory, mut process) = fixture_process(&[
        svc(0x0f),
        svc(0x10),
        0x14000000,
        0xd53bd064, // MRS x4, TPIDRRO_EL0
        0xb9400041, // LDR w1, [x2] (host gate)
        0x34ffffe1, // CBZ w1, previous instruction
        svc(0x0b),  // SleepThread while the pin is still held.
        0x7902009f, // STRH wzr, [x4, #0x100]
        svc(0x36),
        svc(0x10),
        0x14000000,
    ]);
    let entry = instruction_address(&process);
    let caller = process.main_thread_id();
    let process_id = process.scheduler_process_id();
    let gate = process.main_thread().stack_bottom;
    let (target, handle) = super::super::mutex::add_thread(&mut process, entry + 12, 52);
    let core1 = VirtualCpuId::new(1);
    let affinity = process
        .coordinator_mut()
        .scheduler()
        .profile()
        .core_set([core1])
        .unwrap();
    process
        .coordinator_mut()
        .scheduler()
        .migrate(caller, IdealCore::Core(core1), affinity)
        .unwrap();
    let tls = process.thread(target).unwrap().tls_base;
    write_guest_bytes(
        &process,
        tls.checked_add(0x100).unwrap(),
        &0x0001_0002u32.to_le_bytes(),
    );
    let registers = process.thread_mut(target).unwrap().state_mut();
    registers.write_x(x(0), 1_000_000_000);
    registers.write_x(x(2), gate.get());
    state(&mut process).write_w(x(0), handle);
    state(&mut process).write_x(x(1), 1);
    state(&mut process).write_x(x(2), 2);
    process
        .coordinator_mut()
        .start_thread(target.get())
        .unwrap();
    process
        .coordinator_mut()
        .scheduler()
        .pin_thread(VirtualCpuId::new(0), target)
        .unwrap();
    let mut dispatcher = fixed_time_dispatcher(&process)
        .with_memory_completions(process.coordinator_mut().event_sender());
    let (thread, handling) = next_call(&mut process, &mut dispatcher);
    assert_eq!(thread, caller);
    assert!(matches!(handling, ExceptionHandlingResult::Suspended));
    let token = process
        .coordinator_mut()
        .scheduler()
        .thread(caller)
        .unwrap()
        .active_wait
        .unwrap();
    write_guest_bytes(&process, gate, &1u32.to_le_bytes());
    let (thread, _) = next_call(&mut process, &mut dispatcher);
    assert_eq!(thread, target);
    let scheduler = process.coordinator_mut().scheduler();
    assert_eq!(scheduler.thread(caller).unwrap().active_wait, Some(token));
    assert!(scheduler.thread(target).unwrap().active_wait.is_some());
    assert_eq!(
        scheduler
            .pinned_thread(process_id, VirtualCpuId::new(0))
            .unwrap(),
        Some(target)
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(0x100).unwrap()),
        0x0001_0002
    );
    process
        .coordinator_mut()
        .advance_virtual_time(1_000_000_000)
        .unwrap();
    let (thread, handling) = next_call(&mut process, &mut dispatcher);
    assert_eq!(thread, target);
    assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    assert!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(caller)
            .unwrap()
            .active_wait
            .is_none()
    );
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(process.main_thread().state().pc(), entry + 4);
    assert_eq!(read_guest_u32(&process, tls.checked_add(0x100).unwrap()), 0);
    process.coordinator_mut().shutdown().unwrap();
}

#[test]
fn a_suspended_final_property_owner_keeps_its_operation_pending_until_resumed() {
    for stop in [false, true] {
        let (_directory, mut process) = fixture_process(&[
            0xd53bd060, // MRS x0, TPIDRRO_EL0
            0xb9400041, // LDR w1, [x2] (host gate)
            0x34ffffe1, // CBZ w1, previous instruction
            0x7902001f, // STRH wzr, [x0, #0x100]
            svc(0x36),
            svc(0x10),
            0x14000000,
            svc(0x32),
            0x14000000, // First pauses target.
            svc(0x32),
            0x14000000, // Second resumes target after acquiring its lock.
            svc(0x32),  // Controller pauses second.
            0xaa0403e1, // MOV x1, x4 (event-handle array)
            0x52800022, // MOV w2, #1
            0x92800003, // MOV x3, #-1
            svc(0x18),  // Controller blocks on an unsignalled event.
            0xb94000a0, // LDR w0, [x5] (second's handle)
            0x52800001, // MOV w1, #0 (resume)
            svc(0x32),
            svc(0x10),
            0x14000000,
        ]);
        let entry = instruction_address(&process);
        let target = process.main_thread_id();
        let process_id = process.scheduler_process_id();
        let gate = process.main_thread().stack_bottom;
        let buffer = gate.checked_add(0x800).unwrap();
        let tls = process.main_thread().tls_base;
        let handle = process.main_thread().handle;
        let (first, _) = super::super::mutex::add_thread(&mut process, entry + 28, 30);
        let (second, second_handle) = super::super::mutex::add_thread(&mut process, entry + 36, 20);
        let (controller, _) = super::super::mutex::add_thread(&mut process, entry + 44, 10);
        let core1 = VirtualCpuId::new(1);
        let affinity = process
            .coordinator_mut()
            .scheduler()
            .profile()
            .core_set([core1])
            .unwrap();
        let (writable, readable) = nixe_runtime::EventObject::create_pair();
        let event = process.handles().insert(readable).unwrap();
        write_guest_bytes(&process, buffer, &event.to_le_bytes());
        write_guest_bytes(
            &process,
            buffer.checked_add(4).unwrap(),
            &second_handle.to_le_bytes(),
        );
        for (thread, argument, activity) in [
            (first, handle, 1),
            (second, handle, 0),
            (controller, second_handle, 1),
        ] {
            process
                .coordinator_mut()
                .scheduler()
                .migrate(thread, IdealCore::Core(core1), affinity)
                .unwrap();
            let registers = process.thread_mut(thread).unwrap().state_mut();
            registers.write_w(x(0), argument);
            registers.write_x(x(1), activity);
            registers.write_x(x(4), buffer.get());
            registers.write_x(x(5), buffer.get() + 4);
        }
        state(&mut process).write_x(x(2), gate.get());
        write_guest_bytes(
            &process,
            tls.checked_add(0x100).unwrap(),
            &0x0001_0002u32.to_le_bytes(),
        );
        process
            .coordinator_mut()
            .scheduler()
            .pin_thread(VirtualCpuId::new(0), target)
            .unwrap();
        let mut dispatcher = fixed_time_dispatcher(&process)
            .with_memory_completions(process.coordinator_mut().event_sender());
        for (thread, blocks) in [(first, true), (second, true), (controller, false)] {
            process
                .coordinator_mut()
                .start_thread(thread.get())
                .unwrap();
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, thread);
            assert_eq!(
                matches!(handling, ExceptionHandlingResult::Suspended),
                blocks
            );
        }
        let second_token = process
            .coordinator_mut()
            .scheduler()
            .thread(second)
            .unwrap()
            .active_wait
            .unwrap();
        let (caller, handling) = next_call(&mut process, &mut dispatcher);
        assert_eq!(caller, controller);
        assert!(matches!(handling, ExceptionHandlingResult::Suspended));
        write_guest_bytes(&process, gate, &1u32.to_le_bytes());
        let (caller, handling) = next_call(&mut process, &mut dispatcher);
        assert_eq!(caller, target);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        let scheduler = process.coordinator_mut().scheduler();
        assert!(scheduler.thread(first).unwrap().active_wait.is_none());
        let second_info = scheduler.thread(second).unwrap();
        assert!(second_info.paused);
        assert_eq!(second_info.active_wait, Some(second_token));
        assert!(
            scheduler.thread(target).unwrap().paused,
            "a suspended recipient must not resume the target"
        );
        assert_eq!(scheduler.priority_donation_count(), 0);
        if stop {
            process
                .coordinator_mut()
                .terminate_process(process_id)
                .unwrap();
            dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap();
        } else {
            writable.signal();
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, controller);
            assert!(matches!(handling, ExceptionHandlingResult::Resumed));
            assert_eq!(process.thread(controller).unwrap().state().pc(), entry + 64);
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, controller);
            assert!(matches!(handling, ExceptionHandlingResult::Resumed));
            assert_eq!(process.thread(controller).unwrap().state().read_x(x(0)), 0);
            dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap();
            let scheduler = process.coordinator_mut().scheduler();
            let second_info = scheduler.thread(second).unwrap();
            assert!(!second_info.paused);
            assert!(second_info.active_wait.is_none());
            assert!(!scheduler.thread(target).unwrap().paused);
            assert_eq!(process.thread(second).unwrap().state().read_x(x(0)), 0);
            assert_eq!(process.thread(second).unwrap().state().pc(), entry + 40);
        }
        process.coordinator_mut().shutdown().unwrap();
    }
}
