use super::*;

use std::fs;

use crate::{Launcher, LauncherInput};

pub(crate) fn test_kernel() -> std::sync::Arc<nixe_scheduler::SchedulerState> {
    use nixe_scheduler::{
        MachineSchedulerProfile, PriorityRange, SchedulerState, VirtualCpuDescriptor, VirtualCpuId,
    };
    SchedulerState::new(
        MachineSchedulerProfile::new(
            (0..8)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}

pub(crate) fn reference_process_builder(
    kernel: std::sync::Arc<nixe_scheduler::SchedulerState>,
) -> ProcessBuilder {
    let machine = nixe_memory::Machine::switch_ram().unwrap();
    let system = machine.account(nixe_memory::DEFAULT_RAM_BYTES).unwrap();
    ProcessBuilder::new(machine, system, kernel)
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn replace_entry_instructions(process: &RunnableProcess, encodings: &[u32]) {
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    write_code(
        process,
        entry,
        &encodings
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    );
}

fn write_code(process: &RunnableProcess, address: GuestVirtualAddress, bytes: &[u8]) {
    let ranges = process
        .memory
        .space()
        .capture(address, bytes.len() as u64, MemoryPermissions::EXECUTE)
        .unwrap();
    process
        .memory
        .account()
        .machine()
        .write_resolved(&ranges, bytes)
        .unwrap();
}

fn replace_entry_instruction(process: &RunnableProcess, encoding: u32) {
    replace_entry_instructions(process, &[encoding]);
}

#[test]
fn worker_slice_moves_thread_state_out_of_the_process_until_reconciliation() {
    let process = synthetic_process_for_coordinator(&test_kernel(), 1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(3);

    let execution = process
        .begin_thread_execution(thread, vcpu, nixe_cpu::execution::VcpuEventState::default())
        .unwrap();
    assert!(process.main_thread().take_state().is_none());

    process.abort_thread_execution(thread, vcpu, Some(execution));
    assert!(process.main_thread().snapshot().is_some());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Terminating
    );
}

#[test]
fn unsynchronized_cpu_failure_discards_state_and_allows_retirement() {
    let process = synthetic_process_for_coordinator(&test_kernel(), 1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(0);
    let cpu = process.create_worker_cpu_thread(vcpu).unwrap();
    let execution = process
        .begin_thread_execution(thread, vcpu, Default::default())
        .unwrap();
    let failure = ProcessExecutionError::Cpu {
        fault: nixe_cpu::execution::CpuFault {
            backend: "dynarmic",
            kind: nixe_cpu::execution::CpuFaultKind::Internal,
            message: "native execution did not synchronize".into(),
            context: None,
        },
    };
    assert_eq!(
        process.finish_thread_execution(thread, vcpu, execution, Err(failure.clone())),
        Err(failure)
    );
    assert!(process.main_thread().take_state().is_none());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Terminating
    );

    drop(cpu);
    process.complete_cpu_thread_retirement();
}

#[test]
fn concurrent_stop_does_not_hide_an_unsynchronized_native_failure() {
    let process = synthetic_process_for_coordinator(&test_kernel(), 1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(0);
    let execution = process
        .begin_thread_execution(thread, vcpu, Default::default())
        .unwrap();
    assert!(process.terminate_from_host());
    let failure = ProcessExecutionError::Cpu {
        fault: nixe_cpu::execution::CpuFault {
            backend: "dynarmic",
            kind: nixe_cpu::execution::CpuFaultKind::Internal,
            message: "native execution did not synchronize".into(),
            context: None,
        },
    };
    assert_eq!(
        process.finish_thread_execution(thread, vcpu, execution, Err(failure.clone())),
        Err(failure)
    );
    assert!(process.main_thread().take_state().is_none());
}

#[test]
fn panicked_worker_does_not_restore_an_incoming_snapshot() {
    let process = synthetic_process_for_coordinator(&test_kernel(), 1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(0);
    let execution = process
        .begin_thread_execution(thread, vcpu, Default::default())
        .unwrap();
    drop(execution);
    process.abort_thread_execution(thread, vcpu, None);
    assert!(process.main_thread().take_state().is_none());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Terminating
    );
}

fn synthetic_nro() -> Vec<u8> {
    let mut bytes = vec![0; 0x2800];
    bytes[..4].copy_from_slice(&0x1400_0020_u32.to_le_bytes()); // B entry + 0x80
    bytes[0x80..0x84].copy_from_slice(&0xd420_0000_u32.to_le_bytes()); // BRK #0
    bytes[0x10..0x14].copy_from_slice(b"NRO0");
    put_u32(&mut bytes, 0x18, 0x2800);
    put_u32(&mut bytes, 0x20, 0);
    put_u32(&mut bytes, 0x24, 0x1000);
    put_u32(&mut bytes, 0x28, 0x1000);
    put_u32(&mut bytes, 0x2c, 0x1000);
    put_u32(&mut bytes, 0x30, 0x2000);
    put_u32(&mut bytes, 0x34, 0x800);
    put_u32(&mut bytes, 0x38, 0x800);
    bytes[0x40..0x60].fill(0x5a);
    bytes
}

fn plan() -> (tempfile::TempDir, LaunchPlan) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("synthetic.nro");
    fs::write(&path, synthetic_nro()).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    (directory, plan)
}

pub(crate) fn synthetic_process_for_coordinator(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
) -> RunnableProcess {
    let (_directory, plan) = plan();
    reference_process_builder(kernel.clone())
        .with_config(ProcessBuildConfig {
            process_id,

            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap()
}

#[test]
fn thread_creation_succeeds_after_heap_consumes_the_application_budget() {
    let (_directory, plan) = plan();
    let process = reference_process_builder(test_kernel())
        .with_config(ProcessBuildConfig {
            physical_memory_limit: 4 * 1024 * 1024,
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    let memory = process.memory().clone();
    let available = memory.account().limit() - memory.account().used_bytes();
    let mut resize = Box::pin(memory.resize_heap(available));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        std::future::Future::poll(resize.as_mut(), &mut context),
        std::task::Poll::Ready(Ok(_))
    ));
    assert_eq!(memory.account().used_bytes(), memory.account().limit());
    let kernel_before = memory.system_account().used_bytes();
    let core = nixe_scheduler::VirtualCpuId::new(0);
    let profile = nixe_scheduler::MachineSchedulerProfile::new(
        vec![nixe_scheduler::VirtualCpuDescriptor::new(core, 0)],
        nixe_scheduler::PriorityRange::new(0, 63).unwrap(),
    )
    .unwrap();
    let request = ThreadCreateRequest {
        entry: GuestVirtualAddress::new(process.entry_module().entry_address()),
        argument: 0,
        fpcr: 0,
        fpsr: 0,
        stack_top: process.main_thread().stack_top,
        priority: 44,
        ideal_vcpu: Some(core),
        affinity: profile.all_cores(),
    };
    let mut coordinator = crate::RuntimeCoordinator::new(process.scheduler().clone());
    let registration = crate::ProcessRegistration {
        priority: 44,
        ideal_vcpu: Some(core),
        affinity: coordinator.scheduler().profile().all_cores(),
    };
    let process = coordinator.register_process(process, registration).unwrap();
    for _ in 2..=10 {
        crate::process::tests::ready(
            coordinator
                .process(process)
                .unwrap()
                .create_thread(request.clone()),
        )
        .unwrap();
    }
    assert_eq!(memory.account().used_bytes(), memory.account().limit());
    assert_eq!(
        memory.system_account().used_bytes(),
        kernel_before + PAGE_SIZE
    );
}

pub(crate) fn synthetic_instruction_process_for_coordinator(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
    encodings: &[u32],
) -> RunnableProcess {
    let process = synthetic_process_for_coordinator(kernel, process_id);
    replace_entry_instructions(&process, encodings);
    process
}

pub(crate) fn synthetic_svc_process_for_coordinator(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
) -> RunnableProcess {
    let process = synthetic_process_for_coordinator(kernel, process_id);
    replace_entry_instruction(&process, 0xd400_4681);
    process
}

pub(crate) fn synthetic_memory_loop_process(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
) -> RunnableProcess {
    synthetic_native_instruction_process(
        kernel,
        process_id,
        &[0x10000001, 0xf9400020, 0xd4000021, 0x17fffffd],
    )
}

pub(crate) fn synthetic_native_instruction_process(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
    code: &[u32],
) -> RunnableProcess {
    let mut bytes = synthetic_nro();
    for (index, word) in code.iter().copied().enumerate() {
        put_u32(&mut bytes, 0x80 + index * 4, word);
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("memory-loop.nro");
    fs::write(&path, bytes).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    reference_process_builder(kernel.clone())
        .with_config(ProcessBuildConfig {
            process_id,

            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap()
}

mod run;

#[test]
fn dropping_a_process_closes_its_memory_and_handles_while_stopped_native_registers_survive() {
    let process = synthetic_svc_process_for_coordinator(&test_kernel(), 1);
    let memory = std::sync::Arc::clone(process.memory());
    let handles = std::sync::Arc::clone(process.handles());
    let (mut coordinator, _, process) = run::scheduled(process);
    let report = coordinator
        .run_next(std::time::Duration::from_secs(2))
        .unwrap()
        .unwrap()
        .report;
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));
    let state = process.main_thread().take_state().unwrap();
    assert!(state.is_resident());
    let expected = state.register_context();
    assert!(memory.account().used_bytes() != 0);
    coordinator.shutdown().unwrap();
    drop(process);
    assert_eq!(memory.account().used_bytes(), 0);
    assert_eq!(memory.account().machine().statistics().allocated_bytes, 0);
    assert!(handles.is_empty());
    assert_eq!(
        handles.insert(crate::EventObject::new(&test_kernel())),
        Err(crate::HandleError::Closed)
    );
    assert_eq!(state.register_context(), expected);
}

pub fn ready<T>(future: impl std::future::Future<Output = T>) -> T {
    match std::pin::pin!(future).poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => {
            panic!("fixture setup unexpectedly requires asynchronous progress")
        }
    }
}

pub(crate) fn poll_session(
    session: &crate::SessionObject,
    owner: crate::SessionRequestOwner,
) -> Result<Option<crate::SessionMessage>, crate::SessionError> {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };
    match Pin::new(&mut session.response(owner)).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Pending => Ok(None),
        Poll::Ready(result) => result.map(Some),
    }
}

pub(crate) fn watch_future<F: std::future::Future + Unpin>(
    mut future: F,
    callback: std::sync::Arc<dyn Fn() + Send + Sync>,
) -> F {
    use std::{
        pin::Pin,
        task::{Context, Wake, Waker},
    };
    struct Notify(std::sync::Arc<dyn Fn() + Send + Sync>);
    impl Wake for Notify {
        fn wake(self: std::sync::Arc<Self>) {
            (self.0)();
        }
    }
    let waker = Waker::from(std::sync::Arc::new(Notify(callback)));
    assert!(
        Pin::new(&mut future)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    future
}

pub(crate) fn block_thread(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    thread: nixe_scheduler::GuestThreadId,
) {
    use nixe_scheduler::{Completion, ThreadLifecycle};
    let view = kernel.thread(thread).unwrap();
    if view.lifecycle == ThreadLifecycle::Waiting {
        return;
    }
    if view.lifecycle == ThreadLifecycle::Created {
        kernel.start_thread(thread).unwrap();
    }
    kernel.set_priority(thread, 0, None).unwrap();
    let core = view
        .ideal_vcpu
        .unwrap_or_else(|| view.affinity.iter().next().unwrap());
    let lease = kernel.claim(core).unwrap().unwrap();
    assert_eq!(lease.thread, thread);
    kernel.complete(lease, Completion::Waiting).unwrap();
    kernel
        .set_priority(thread, view.base_priority, None)
        .unwrap();
}
