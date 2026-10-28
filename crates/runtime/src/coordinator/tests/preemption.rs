use super::*;
use crate::process::tests::synthetic_native_instruction_process;
use nixe_memory::{AccessOperation, DeviceAccessEndpoint, DeviceAccessRequest};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
    mpsc,
};

struct Device {
    machine: nixe_memory::Machine,
    scheduler: Arc<SchedulerState>,
    process: ProcessId,
    fail_on_read: Option<bool>,
    accesses: Mutex<Vec<(bool, Option<GuestThreadId>)>>,
    defer: AtomicU8,
    pending: Mutex<Vec<DeviceAccessRequest>>,
    arrived: mpsc::Sender<()>,
}
impl DeviceAccessEndpoint for Device {
    fn submit(&self, request: DeviceAccessRequest) {
        let read = matches!(
            request.operation(),
            AccessOperation::Read | AccessOperation::ReadNative
        );
        // This would deadlock if TLS access held the scheduler lock. Inspecting
        // the actual pin also checks the read -> pin -> flag publication order.
        let pinned = self
            .scheduler
            .pinned_thread(self.process, VirtualCpuId::new(7))
            .unwrap();
        self.accesses.lock().unwrap().push((read, pinned));
        if self
            .defer
            .compare_exchange(
                if read { 1 } else { 2 },
                0,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            self.pending.lock().unwrap().push(request);
            self.arrived.send(()).unwrap();
            return;
        }
        self.finish(request);
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
impl Device {
    fn finish(&self, request: DeviceAccessRequest) {
        let read = matches!(
            request.operation(),
            AccessOperation::Read | AccessOperation::ReadNative
        );
        let result = if self.fail_on_read == Some(read) {
            Err(Arc::new(nixe_memory::MemoryError::Execution(
                "TLS device access failed".into(),
            )))
        } else {
            self.machine
                .apply_resolved_access(request.ranges(), request.operation())
                .map_err(Arc::new)
        };
        request.finish(result);
    }
}

fn pending_tls(
    fail_on_read: Option<bool>,
) -> (
    RuntimeCoordinator,
    ProcessId,
    GuestThreadId,
    Arc<Device>,
    mpsc::Receiver<()>,
) {
    let mut coordinator = RuntimeCoordinator::new(two_core_profile());
    let process = coordinator
        .register_process(
            synthetic_native_instruction_process(1, &[0x14000000]),
            registration(&coordinator),
        )
        .unwrap();
    let runtime = coordinator.process(process).unwrap();
    let thread = runtime.main_thread_id();
    let tls = runtime.main_thread().tls_base.checked_add(0x100).unwrap();
    runtime
        .memory()
        .space()
        .write_value(tls, 4, 2)
        .unwrap()
        .into_ready()
        .unwrap();
    let machine = runtime.memory().space().machine().clone();
    let ranges = runtime
        .memory()
        .space()
        .capture(tls, 4, nixe_memory::MemoryPermissions::READ_WRITE)
        .unwrap();
    let (arrived, receiver) = mpsc::channel();
    let device = Arc::new(Device {
        machine: machine.clone(),
        scheduler: coordinator.scheduler.clone(),
        process,
        fail_on_read,
        accesses: Mutex::default(),
        defer: AtomicU8::new(0),
        pending: Mutex::default(),
        arrived,
    });
    let endpoint: Arc<dyn DeviceAccessEndpoint> = device.clone();
    machine.set_device_endpoint(&endpoint);
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    coordinator
        .post_vcpu_interrupt(VirtualCpuId::new(7), 1)
        .unwrap();
    (coordinator, process, thread, device, receiver)
}

fn pending_native_data(
    instructions: &[u32],
    read: bool,
    fail: bool,
) -> (
    RuntimeCoordinator,
    ProcessId,
    GuestThreadId,
    Arc<Device>,
    mpsc::Receiver<()>,
) {
    let mut coordinator = RuntimeCoordinator::new(profile());
    let mut process = synthetic_native_instruction_process(1, instructions);
    let address = process.main_thread().stack_bottom;
    use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
    let state = process.main_thread_mut().state_mut();
    state.write_x(A64Register::General(A64GeneralRegister::new(0).unwrap()), 7);
    state.write_x(
        A64Register::General(A64GeneralRegister::new(1).unwrap()),
        address.get(),
    );
    let process = coordinator
        .register_process(process, registration(&coordinator))
        .unwrap();
    let runtime = coordinator.process(process).unwrap();
    let thread = runtime.main_thread_id();
    let machine = runtime.memory().space().machine().clone();
    let ranges = runtime
        .memory()
        .space()
        .capture(address, 8, nixe_memory::MemoryPermissions::READ_WRITE)
        .unwrap();
    let (arrived, receiver) = mpsc::channel();
    let device = Arc::new(Device {
        machine: machine.clone(),
        scheduler: coordinator.scheduler.clone(),
        process,
        fail_on_read: fail.then_some(read),
        accesses: Mutex::default(),
        defer: AtomicU8::new(if read { 1 } else { 2 }),
        pending: Mutex::default(),
        arrived,
    });
    let endpoint: Arc<dyn DeviceAccessEndpoint> = device.clone();
    machine.set_device_endpoint(&endpoint);
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    (coordinator, process, thread, device, receiver)
}

#[test]
fn stopping_native_reads_writes_and_exclusive_stores_cancels_the_dependency_without_a_cpu_fault() {
    for (code, read) in [
        (&[0xf9400020, 0xd4000201][..], true),  // LDR x0, [x1]
        (&[0xf9000020, 0xd4000201][..], false), // STR x0, [x1]
        (&[0xc85f7c22, 0xc8037c20, 0xd4000201][..], false), // LDXR; STXR
    ] {
        let (mut coordinator, process, thread, device, arrived) =
            pending_native_data(code, read, false);
        assert!(
            coordinator
                .run_next(std::time::Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        arrived
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let request = device.pending.lock().unwrap().pop().unwrap();
        if code.len() == 3 {
            assert!(matches!(request.operation(), AccessOperation::Atomic(_)));
        }
        assert!(matches!(
            request.operation(),
            AccessOperation::ReadNative | AccessOperation::Write(_) | AccessOperation::Atomic(_)
        ));
        assert!(coordinator.terminate_process(process).unwrap());
        assert!(request.is_cancelled());
        assert!(!coordinator.scheduler.has_active_leases());
        assert_eq!(
            coordinator.scheduler.thread(thread).unwrap().lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited
        );
        drop(request);
        coordinator.shutdown().unwrap();
    }
}

#[test]
fn native_data_failures_completed_before_host_stop_remain_fatal() {
    for (code, read) in [
        (&[0xf9400020, 0xd4000201][..], true),
        (&[0xf9000020, 0xd4000201][..], false),
        (&[0xc85f7c22, 0xc8037c20, 0xd4000201][..], false),
    ] {
        let (mut coordinator, process, thread, device, arrived) =
            pending_native_data(code, read, true);
        assert!(
            coordinator
                .run_next(std::time::Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        arrived
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        device.finish(device.pending.lock().unwrap().pop().unwrap());
        let result = match coordinator
            .inbox
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap()
        {
            crate::external_event::CoordinatorEvent::Completed(result) => result,
            _ => panic!("expected native completion with the original data failure"),
        };
        coordinator.request_process_stop(process).unwrap();
        coordinator.workers.pending_results.push_back(result);
        assert!(
            matches!(coordinator.run_next(std::time::Duration::from_secs(2)), Err(CoordinatorError::Execution { error: ProcessExecutionError::Cpu { fault }, .. }) if fault.kind == nixe_cpu::execution::CpuFaultKind::Internal && fault.message.contains("TLS device access failed"))
        );
        assert!(!coordinator.scheduler.has_active_leases());
        assert_eq!(
            coordinator.scheduler.thread(thread).unwrap().lifecycle,
            nixe_scheduler::ThreadLifecycle::Faulted
        );
    }
}

#[test]
fn device_tls_dependencies_complete_outside_the_scheduler_lock_before_lease_release() {
    let (mut coordinator, process, thread, device, _arrived) = pending_tls(None);
    let execution = coordinator
        .run_next(std::time::Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(execution.lease.thread, thread);
    assert!(matches!(
        execution.report.stop,
        ExecutionStop::PendingEvent { mask: 1 }
    ));
    assert_eq!(
        coordinator
            .scheduler
            .pinned_thread(process, VirtualCpuId::new(7))
            .unwrap(),
        Some(thread)
    );
    assert_eq!(
        *device.accesses.lock().unwrap(),
        [(true, None), (false, Some(thread))]
    );
    let accesses = device.accesses.lock().unwrap();
    assert!(
        accesses
            .iter()
            .filter(|(read, _)| !read)
            .all(|(_, pin)| *pin == Some(thread))
    );
}

#[test]
fn failed_tls_dependency_remains_fatal_and_releases_the_native_lease() {
    for fail_read in [true, false] {
        let (mut coordinator, process, thread, _device, _arrived) = pending_tls(Some(fail_read));
        let Err(error) = coordinator.run_next(std::time::Duration::from_secs(2)) else {
            panic!("failed TLS dependency was accepted");
        };
        assert!(
            matches!(error, CoordinatorError::ThreadLocal { thread: failed, fault }
        if failed == thread && matches!(fault.as_ref(), nixe_memory::MemoryError::Execution(message) if message.as_ref() == "TLS device access failed"))
        );
        assert!(!coordinator.scheduler.has_active_leases());
        assert_eq!(
            coordinator.scheduler.thread(thread).unwrap().lifecycle,
            nixe_scheduler::ThreadLifecycle::Faulted
        );
        assert_eq!(
            coordinator
                .scheduler
                .pinned_thread(process, VirtualCpuId::new(7))
                .unwrap(),
            None
        );
    }
}

#[test]
fn deferred_tls_read_and_flag_leave_other_cores_and_host_stop_live() {
    for phase in [1, 2] {
        let (mut coordinator, process, thread, device, arrived) = pending_tls(None);
        let other = coordinator
            .register_process(
                synthetic_native_instruction_process(2, &[0xd4000201]), // SVC #0x10
                ProcessRegistration {
                    priority: 44,
                    ideal_vcpu: Some(VirtualCpuId::new(3)),
                    affinity: coordinator
                        .scheduler
                        .profile()
                        .core_set([VirtualCpuId::new(3)])
                        .unwrap(),
                },
            )
            .unwrap();
        let other_thread = coordinator.process(other).unwrap().main_thread_id();
        coordinator
            .scheduler
            .set_activity(other_thread, true)
            .unwrap();
        device.defer.store(phase, Ordering::Relaxed);
        assert!(
            coordinator
                .run_next(std::time::Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        arrived
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        assert_eq!(
            coordinator
                .scheduler
                .lease_for_vcpu(VirtualCpuId::new(7))
                .unwrap()
                .thread,
            thread
        );
        assert_eq!(
            coordinator
                .scheduler
                .pinned_thread(process, VirtualCpuId::new(7))
                .unwrap(),
            (phase == 2).then_some(thread)
        );
        if phase == 2 {
            // A known pin defers suspension. The caller must not synchronously
            // wait for the core that is publishing its pending interrupt flag.
            coordinator.scheduler().set_activity(thread, true).unwrap();
        }

        coordinator
            .scheduler
            .set_activity(other_thread, false)
            .unwrap();
        let execution = coordinator
            .run_next(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(execution.lease.vcpu, VirtualCpuId::new(3));
        assert!(matches!(
            execution.report.stop,
            ExecutionStop::SupervisorCall {
                immediate: 0x10,
                ..
            }
        ));
        coordinator
            .event_sender()
            .submit(ExternalEvent::HostStop)
            .unwrap();
        assert!(
            coordinator
                .run_next(std::time::Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        assert!(coordinator.host_stop_requested());

        let request = device.pending.lock().unwrap().pop().unwrap();
        device.finish(request);
        coordinator.quiesce().unwrap();
        assert!(!coordinator.scheduler.has_active_leases());
        assert_eq!(
            coordinator
                .scheduler
                .pinned_thread(process, VirtualCpuId::new(7))
                .unwrap(),
            Some(thread)
        );
        assert_eq!(
            *device.accesses.lock().unwrap(),
            [(true, None), (false, Some(thread))]
        );
    }
}

#[test]
fn stopping_a_core_with_deferred_tls_cancels_the_real_dependency_and_releases_ownership() {
    for phase in [1, 2] {
        let (mut coordinator, process, thread, device, arrived) = pending_tls(None);
        device.defer.store(phase, Ordering::Relaxed);
        assert!(
            coordinator
                .run_next(std::time::Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        arrived
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        coordinator.terminate_process(process).unwrap();
        assert!(!coordinator.scheduler.has_active_leases());
        assert_eq!(
            coordinator
                .scheduler
                .pinned_thread(process, VirtualCpuId::new(7))
                .unwrap(),
            None
        );
        assert_eq!(
            coordinator.scheduler.thread(thread).unwrap().lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited
        );
        // The outstanding request never needs a fabricated completion to stop.
        drop(device.pending.lock().unwrap().pop().unwrap());
    }
}

#[test]
fn host_stop_does_not_hide_a_device_failure_already_returned_by_the_interrupted_core() {
    for (phase, read) in [(1, true), (2, false)] {
        let (mut coordinator, process, _thread, device, arrived) = pending_tls(Some(read));
        device.defer.store(phase, Ordering::Relaxed);
        assert!(
            coordinator
                .run_next(std::time::Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        arrived
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let pending = device.pending.lock().unwrap().pop().unwrap();
        device.finish(pending);
        let result = match coordinator
            .inbox
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap()
        {
            crate::external_event::CoordinatorEvent::Completed(result) => result,
            _ => panic!("expected native completion with the original device failure"),
        };
        coordinator.request_process_stop(process).unwrap();
        coordinator.workers.pending_results.push_back(result);
        assert!(
            matches!(coordinator.run_next(std::time::Duration::from_secs(2)),
            Err(CoordinatorError::ThreadLocal { fault, .. }) if matches!(fault.as_ref(), nixe_memory::MemoryError::Execution(message) if message.as_ref() == "TLS device access failed"))
        );
        assert!(!coordinator.scheduler.has_active_leases());
        assert_eq!(
            coordinator
                .scheduler
                .pinned_thread(process, VirtualCpuId::new(7))
                .unwrap(),
            None
        );
    }
}
