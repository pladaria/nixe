use super::*;
use nixe_scheduler::IdealCore;
use nixe_scheduler::VirtualCpuId;

#[path = "preemption/properties.rs"]
mod properties;

#[test]
fn interrupted_sdk_critical_section_runs_to_its_release_before_a_higher_priority_thread() {
    let (_directory, mut process) = fixture_process(&[
        0xd53bd060, // MRS x0, TPIDRRO_EL0
        0x52800041, // MOV w1, #2 (nested preemption disable count)
        0x79020001, // STRH w1, [x0, #0x100]
        0xb9000041, // STR w1, [x2] (critical-section work pending)
        0x79420401, // LDRH w1, [x0, #0x102] (deferred interrupt flag)
        0x34ffffe1, // CBZ w1, previous instruction
        0x7902001f, // STRH wzr, [x0, #0x100]
        0xb900005f, // STR wzr, [x2] (work completed before unpin)
        svc(0x36),
        svc(0x10),
        svc(0x0a),
    ]);
    let mut dispatcher = fixed_time_dispatcher(&process);
    let main = process.main_thread_id();
    let process_id = process.scheduler_process_id();
    let entry = instruction_address(&process);
    let tls = process.main_thread().tls_base;
    let work = process.main_thread().stack_bottom;
    state(&mut process).write_x(x(2), work.get());
    process
        .coordinator_mut()
        .set_thread_priority(Some(main), main.get(), 52)
        .unwrap();
    let (higher, _) = super::mutex::add_thread(&mut process, entry + 9 * 4, 51);
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        let execution = process
            .coordinator_mut()
            .run_next(Duration::from_millis(10))
            .unwrap();
        if let Some(execution) = execution {
            assert!(matches!(
                execution.report.stop,
                nixe_runtime::ExecutionStop::Safepoint
            ));
        }
        if read_guest_u32(&process, work) == 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "native critical section did not start"
        );
    }
    process
        .coordinator_mut()
        .start_thread(higher.get())
        .unwrap();

    let interrupted = process
        .coordinator_mut()
        .run_next(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(interrupted.lease.thread, main);
    assert!(matches!(
        interrupted.report.stop,
        nixe_runtime::ExecutionStop::Safepoint
    ));
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .pinned_thread(process_id, VirtualCpuId::new(0))
            .unwrap(),
        Some(main)
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(0x100).unwrap()),
        0x0001_0002
    );
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .selection(VirtualCpuId::new(0))
            .unwrap()
            .thread,
        Some(main)
    );
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .thread(main)
            .unwrap()
            .effective_priority,
        52
    );

    let (caller, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, main);
    assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    assert_eq!(
        read_abi_register(&process, 0),
        tls.get(),
        "void SVC preserves x0"
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(0x100).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, work), 0);
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .pinned_thread(process_id, VirtualCpuId::new(0))
            .unwrap(),
        None
    );
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .selection(VirtualCpuId::new(0))
            .unwrap()
            .thread,
        Some(higher)
    );
    let (caller, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, higher);
    assert!(matches!(handling, ExceptionHandlingResult::Resumed));
}

#[test]
fn synchronize_preemption_state_without_a_pin_preserves_tls_and_all_registers() {
    let (_directory, mut process) = fixture_process(&[svc(0x36)]);
    let mut dispatcher = fixed_time_dispatcher(&process);
    let tls = process.main_thread().tls_base;
    write_guest_bytes(
        &process,
        tls.checked_add(0x100).unwrap(),
        &0x0001_0002_u32.to_le_bytes(),
    );
    for i in 0..31 {
        state(&mut process).write_x(x(i), 0x1234_5678_0000_0000 | u64::from(i));
    }
    let original = state(&mut process).register_context();
    let (_, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    let after = state(&mut process).register_context();
    assert_eq!(after.x, original.x);
    assert_eq!(after.sp, original.sp);
    assert_eq!(after.nzcv, original.nzcv);
    assert_eq!(after.pc.get(), original.pc.get() + 4);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(0x100).unwrap()),
        0x0001_0002
    );
    assert!(
        dispatcher
            .coverage()
            .iter()
            .any(|entry| entry.immediate == 0x36 && entry.support == HorizonSvcSupport::Complete)
    );
}

#[test]
fn synchronization_does_not_clear_another_threads_pin_or_interrupt_flag() {
    let (_directory, mut process) = fixture_process(&[svc(0x36), 0x14000000]);
    let main = process.main_thread_id();
    let process_id = process.scheduler_process_id();
    let entry = instruction_address(&process);
    let (peer, _) = super::mutex::add_thread(&mut process, entry + 4, 52);
    process.coordinator_mut().start_thread(peer.get()).unwrap();
    let peer_tls = process.thread(peer).unwrap().tls_base;
    write_guest_bytes(
        &process,
        peer_tls.checked_add(0x102).unwrap(),
        &1u16.to_le_bytes(),
    );
    process
        .coordinator_mut()
        .scheduler()
        .pin_thread(VirtualCpuId::new(0), peer)
        .unwrap();
    let affinity = process
        .coordinator_mut()
        .scheduler()
        .profile()
        .core_set([VirtualCpuId::new(1)])
        .unwrap();
    process
        .coordinator_mut()
        .scheduler()
        .migrate(main, IdealCore::Core(VirtualCpuId::new(1)), affinity)
        .unwrap();
    let mut dispatcher = fixed_time_dispatcher(&process);
    let (caller, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, main);
    assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .pinned_thread(process_id, VirtualCpuId::new(0))
            .unwrap(),
        Some(peer)
    );
    assert_eq!(
        read_guest_bytes(&process, peer_tls.checked_add(0x102).unwrap(), 2),
        [1, 0]
    );
    process
        .coordinator_mut()
        .scheduler()
        .unpin_thread(peer)
        .unwrap();
    assert_eq!(
        process
            .coordinator_mut()
            .scheduler()
            .pinned_thread(process_id, VirtualCpuId::new(0))
            .unwrap(),
        None
    );
}

#[test]
fn deferred_flag_clear_retains_the_pin_until_completion_and_preserves_void_registers() {
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
    for succeeds in [true, false] {
        let (_directory, mut process) = fixture_process(&[svc(0x36)]);
        let ingress = process.coordinator_mut().event_sender();
        let mut dispatcher = support::dispatcher(&process).with_memory_completions(ingress.clone());
        let main = process.main_thread_id();
        let process_id = process.scheduler_process_id();
        let flag = process.main_thread().tls_base.checked_add(0x102).unwrap();
        write_guest_bytes(&process, flag, &1u16.to_le_bytes());
        state(&mut process).write_x(x(0), 0x12345678);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let execution = loop {
            if let Some(execution) = process
                .coordinator_mut()
                .run_next(Duration::from_millis(20))
                .unwrap()
                && matches!(
                    execution.report.stop,
                    nixe_runtime::ExecutionStop::SupervisorCall { .. }
                )
            {
                break execution;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "native SVC did not arrive"
            );
        };
        process
            .coordinator_mut()
            .scheduler()
            .pin_thread(VirtualCpuId::new(0), main)
            .unwrap();
        let machine = process.memory().account().machine().clone();
        let ranges = process
            .memory()
            .space()
            .capture(flag, 2, MemoryPermissions::READ_WRITE)
            .unwrap();
        let (send, received) = mpsc::channel();
        let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
        machine.set_device_endpoint(&device);
        machine.publish_device_write(&ranges, || Ok(())).unwrap();
        let handling = dispatcher
            .route_scheduled_supervisor_call(
                process.coordinator_mut(),
                execution.lease,
                &execution.report.stop,
            )
            .unwrap();
        assert!(matches!(handling, ExceptionHandlingResult::Suspended));
        let pending = received.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(
            pending.operation(),
            nixe_memory::AccessOperation::Write(_)
        ));
        assert_eq!(
            process
                .coordinator_mut()
                .scheduler()
                .pinned_thread(process_id, VirtualCpuId::new(0))
                .unwrap(),
            Some(main)
        );
        ingress
            .submit(nixe_runtime::ExternalEvent::HostService(
                nixe_runtime::ExternalEventSource::Input,
            ))
            .unwrap();
        assert!(
            process
                .coordinator_mut()
                .run_next(Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        let completion = if succeeds {
            machine
                .apply_resolved_access(pending.ranges(), pending.operation())
                .map_err(Arc::new)
        } else {
            Err(Arc::new(nixe_memory::MemoryError::Execution(
                "flag clear failed".into(),
            )))
        };
        pending.finish(completion);
        let result = dispatcher.poll_memory_completions(process.coordinator_mut());
        if succeeds {
            result.unwrap();
            assert_eq!(read_abi_register(&process, 0), 0x12345678);
            assert!(
                matches!(machine.apply_resolved_access(&ranges, &nixe_memory::AccessOperation::Read).unwrap(), nixe_memory::AccessResult::Read(bytes) if bytes.as_ref() == [0, 0])
            );
        } else {
            assert!(matches!(
                result,
                Err(HorizonSvcFault::MemoryAccess {
                    immediate: 0x36,
                    ..
                })
            ));
        }
        assert_eq!(
            process
                .coordinator_mut()
                .scheduler()
                .pinned_thread(process_id, VirtualCpuId::new(0))
                .unwrap(),
            None
        );
        assert!(!process.coordinator_mut().scheduler().has_active_leases());
    }
}

#[test]
fn thread_activity_and_affinity_wait_for_the_targets_real_unpin() {
    for pause in [true, false] {
        let (_directory, mut process) = fixture_process(&[
            svc(if pause { 0x32 } else { 0x0f }),
            svc(0x10),
            svc(0x0a),
            0xd53bd060, // MRS x0, TPIDRRO_EL0
            0xb9400041, // LDR w1, [x2] (host gate)
            0x34ffffe1, // CBZ w1, previous instruction
            0x7902001f, // STRH wzr, [x0, #0x100]
            svc(0x36),
            svc(0x0a),
        ]);
        let mut dispatcher = fixed_time_dispatcher(&process)
            .with_memory_completions(process.coordinator_mut().event_sender());
        let main = process.main_thread_id();
        let entry = instruction_address(&process);
        let gate = process.main_thread().stack_bottom;
        let (target, handle) = super::mutex::add_thread(&mut process, entry + 12, 52);
        let core = VirtualCpuId::new(1);
        let affinity = process
            .coordinator_mut()
            .scheduler()
            .profile()
            .core_set([core])
            .unwrap();
        process
            .coordinator_mut()
            .scheduler()
            .migrate(main, IdealCore::Core(core), affinity)
            .unwrap();
        let target_tls = process.thread(target).unwrap().tls_base;
        write_guest_bytes(
            &process,
            target_tls.checked_add(0x100).unwrap(),
            &0x0001_0002u32.to_le_bytes(),
        );
        process
            .thread_mut(target)
            .unwrap()
            .state_mut()
            .write_x(x(2), gate.get());
        process
            .coordinator_mut()
            .start_thread(target.get())
            .unwrap();
        process
            .coordinator_mut()
            .scheduler()
            .pin_thread(VirtualCpuId::new(0), target)
            .unwrap();
        state(&mut process).write_x(x(0), u64::from(handle));
        state(&mut process).write_x(x(1), 1);
        state(&mut process).write_x(x(2), 2);
        let (caller, handling) = next_call(&mut process, &mut dispatcher);
        assert_eq!(caller, main);
        assert!(matches!(handling, ExceptionHandlingResult::Suspended));
        assert!(
            process
                .coordinator_mut()
                .scheduler()
                .thread(main)
                .unwrap()
                .active_wait
                .is_some()
        );
        assert_eq!(
            process
                .coordinator_mut()
                .scheduler()
                .thread(target)
                .unwrap()
                .paused,
            pause
        );
        write_guest_bytes(&process, gate, &1u32.to_le_bytes());
        let (caller, handling) = next_call(&mut process, &mut dispatcher);
        assert_eq!(caller, target);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
        assert_eq!(
            read_guest_u32(&process, target_tls.checked_add(0x100).unwrap()),
            0
        );
        let view = process.coordinator_mut().scheduler().thread(main).unwrap();
        assert!(view.active_wait.is_none());
        assert_eq!(view.lifecycle, nixe_scheduler::ThreadLifecycle::Ready);
        let target_view = process
            .coordinator_mut()
            .scheduler()
            .thread(target)
            .unwrap();
        if pause {
            assert_eq!(
                target_view.lifecycle,
                nixe_scheduler::ThreadLifecycle::Suspended
            );
        } else {
            assert_eq!(target_view.affinity, affinity);
        }
        assert_eq!(read_abi_register(&process, 0), 0);
        let (caller, handling) = next_call(&mut process, &mut dispatcher);
        assert_eq!(caller, main);
        assert!(matches!(handling, ExceptionHandlingResult::Resumed));
    }
}

#[test]
fn competing_thread_properties_serialize_through_unpin_and_publish_results_once() {
    for operation in ["resume", "context", "mask", "preserve", "terminated"] {
        for stop in [false, true] {
            let mask = matches!(operation, "mask" | "preserve");
            let context_query = matches!(operation, "context" | "terminated");
            let (_directory, mut process) = fixture_process(&[
                svc(if mask { 0x0f } else { 0x32 }),
                svc(0x10),
                0x14000000,
                0xd53bd060, // MRS x0, TPIDRRO_EL0
                0xb9400041, // LDR w1, [x2] (host gate)
                0x34ffffe1, // CBZ w1, previous instruction
                svc(0x10),  // An ordinary SVC while the SDK pin is still held.
                0xd53bd060, // MRS x0, TPIDRRO_EL0
                0xb9400041, // LDR w1, [x2]
                0x7100083f, // CMP w1, #2
                0x54ffffc1, // B.NE to LDR
                0x7902001f, // STRH wzr, [x0, #0x100]
                svc(if operation == "terminated" {
                    0x0a
                } else {
                    0x36
                }),
                svc(0x10),
                0x14000000,
                svc(match operation {
                    "context" | "terminated" => 0x33,
                    "mask" | "preserve" => 0x0f,
                    _ => 0x32,
                }),
                svc(0x10),
                0x14000000,
            ]);
            let entry = instruction_address(&process);
            let main = process.main_thread_id();
            let process_id = process.scheduler_process_id();
            let gate = process.main_thread().stack_bottom;
            let output = gate.checked_add(0x800).unwrap();
            write_guest_bytes(&process, output, &[0xaa; 0x320]);
            let (target, handle) = super::mutex::add_thread(&mut process, entry + 12, 52);
            let (second, _) = super::mutex::add_thread(&mut process, entry + 60, 20);
            let core1 = VirtualCpuId::new(1);
            let affinity = process
                .coordinator_mut()
                .scheduler()
                .profile()
                .core_set([core1])
                .unwrap();
            for thread in [main, second] {
                process
                    .coordinator_mut()
                    .scheduler()
                    .migrate(thread, IdealCore::Core(core1), affinity)
                    .unwrap();
            }
            let tls = process.thread(target).unwrap().tls_base;
            write_guest_bytes(
                &process,
                tls.checked_add(0x100).unwrap(),
                &0x0001_0002u32.to_le_bytes(),
            );
            process
                .thread_mut(target)
                .unwrap()
                .state_mut()
                .write_x(x(2), gate.get());
            state(&mut process).write_w(x(0), handle);
            state(&mut process).write_x(x(1), 1);
            state(&mut process).write_x(x(2), 2);
            let second_state = process.thread_mut(second).unwrap().state_mut();
            second_state.write_x(
                x(0),
                if context_query {
                    output.get()
                } else {
                    u64::from(handle)
                },
            );
            second_state.write_x(
                x(1),
                if context_query {
                    u64::from(handle)
                } else if operation == "preserve" {
                    -3i64 as u64
                } else {
                    0
                },
            );
            second_state.write_x(x(2), if operation == "preserve" { 3 } else { 1 });
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
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, main);
            assert!(matches!(handling, ExceptionHandlingResult::Suspended));
            process
                .coordinator_mut()
                .start_thread(second.get())
                .unwrap();
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, second);
            assert!(matches!(handling, ExceptionHandlingResult::Suspended));
            let scheduler = process.coordinator_mut().scheduler();
            for thread in [main, second] {
                assert_eq!(
                    scheduler.thread(thread).unwrap().lifecycle,
                    nixe_scheduler::ThreadLifecycle::Waiting
                );
                assert!(scheduler.thread(thread).unwrap().active_wait.is_some());
            }
            assert_eq!(scheduler.thread(main).unwrap().effective_priority, 20);
            let target_info = scheduler.thread(target).unwrap();
            assert_eq!(target_info.paused, !mask);
            assert_eq!(
                target_info.affinity,
                if mask {
                    affinity
                } else {
                    scheduler
                        .profile()
                        .core_set([VirtualCpuId::new(0)])
                        .unwrap()
                }
            );
            assert_eq!(read_guest_bytes(&process, output, 0x320), [0xaa; 0x320]);
            // Both accepted operations retain their actual target across close.
            let object = process.thread(target).unwrap().object();
            let target_lifetime = std::sync::Arc::downgrade(&object);
            drop(object);
            let original = process.thread(target).unwrap().handle;
            drop(process.handles().close(handle).unwrap());
            drop(process.handles().close(original).unwrap());
            if stop {
                process
                    .coordinator_mut()
                    .terminate_process(process_id)
                    .unwrap();
                dispatcher
                    .poll_memory_completions(process.coordinator_mut())
                    .unwrap();
                assert!(target_lifetime.upgrade().is_none());
                assert_eq!(
                    process
                        .coordinator_mut()
                        .scheduler()
                        .priority_donation_count(),
                    0
                );
                process.coordinator_mut().shutdown().unwrap();
                continue;
            }
            write_guest_bytes(&process, gate, &1u32.to_le_bytes());
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, target);
            assert!(matches!(handling, ExceptionHandlingResult::Resumed));
            for thread in [main, second] {
                assert!(
                    process
                        .coordinator_mut()
                        .scheduler()
                        .thread(thread)
                        .unwrap()
                        .active_wait
                        .is_some(),
                    "a pinned SVC must not complete {operation}"
                );
            }
            write_guest_bytes(&process, gate, &2u32.to_le_bytes());
            let (caller, handling) = next_call(&mut process, &mut dispatcher);
            assert_eq!(caller, target);
            if operation == "terminated" {
                assert!(matches!(
                    handling,
                    ExceptionHandlingResult::Terminated { .. }
                ));
            } else {
                assert!(matches!(handling, ExceptionHandlingResult::Resumed));
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while [main, second].into_iter().any(|id| {
                process
                    .coordinator_mut()
                    .scheduler()
                    .thread(id)
                    .unwrap()
                    .active_wait
                    .is_some()
            }) {
                dispatcher
                    .poll_memory_completions(process.coordinator_mut())
                    .unwrap();
                assert!(
                    std::time::Instant::now() < deadline,
                    "property completion was lost"
                );
            }
            assert_eq!(read_abi_register(&process, 0), 0);
            assert_eq!(process.thread(second).unwrap().state().read_x(x(0)), 0);
            assert_eq!(process.thread(main).unwrap().state().pc(), entry + 4);
            assert_eq!(process.thread(second).unwrap().state().pc(), entry + 64);
            let scheduler = process.coordinator_mut().scheduler();
            assert_eq!(scheduler.thread(main).unwrap().effective_priority, 44);
            let target_info = scheduler.thread(target).unwrap();
            assert_eq!(target_info.paused, context_query);
            assert!(target_info.affinity.contains(VirtualCpuId::new(0)));
            if operation == "preserve" {
                assert_eq!(target_info.ideal_vcpu, Some(core1));
                assert!(target_info.affinity.contains(core1));
            }
            if context_query {
                let bytes = read_guest_bytes(&process, output, 0x320);
                if operation == "terminated" {
                    assert_eq!(bytes, [0; 0x320]);
                } else {
                    assert_eq!(
                        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                        tls.get()
                    );
                    assert_eq!(
                        u64::from_le_bytes(bytes[0x100..0x108].try_into().unwrap()),
                        entry + 52
                    );
                }
            }
            process.coordinator_mut().shutdown().unwrap();
        }
    }
}

fn next_call(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
) -> (
    nixe_scheduler::GuestThreadId,
    ExceptionHandlingResult<HorizonSvcFault>,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        let execution = process
            .coordinator_mut()
            .run_next(Duration::from_millis(20))
            .unwrap();
        if let Some(execution) = execution {
            if matches!(
                execution.report.stop,
                nixe_runtime::ExecutionStop::SupervisorCall { .. }
            ) {
                let handling = dispatcher
                    .route_scheduled_supervisor_call(
                        process.coordinator_mut(),
                        execution.lease,
                        &execution.report.stop,
                    )
                    .unwrap();
                dispatcher
                    .poll_memory_completions(process.coordinator_mut())
                    .unwrap();
                return (execution.lease.thread, handling);
            }
            assert!(matches!(
                execution.report.stop,
                nixe_runtime::ExecutionStop::Safepoint
            ));
        }
        assert!(
            std::time::Instant::now() < deadline,
            "native caller did not reach an SVC"
        );
    }
}
