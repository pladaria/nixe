use super::*;

use std::fs;

use crate::{Launcher, LauncherInput};

pub(crate) fn reference_process_builder() -> ProcessBuilder {
    let machine = nixe_memory::Machine::switch_ram().unwrap();
    let system = machine.account(nixe_memory::DEFAULT_RAM_BYTES).unwrap();
    ProcessBuilder::new(machine, system)
}

struct FixedSupervisorCallDispatcher<F> {
    outcome: Option<crate::ExceptionDispatchOutcome<F>>,
}

impl<F> crate::ExceptionDispatcher for FixedSupervisorCallDispatcher<F> {
    type Fault = F;

    fn dispatch(
        &mut self,
        _context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        self.outcome.take().expect("dispatcher is called once")
    }
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn replace_entry_instructions(process: &mut RunnableProcess, encodings: &[u32]) {
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

fn replace_entry_instruction(process: &mut RunnableProcess, encoding: u32) {
    replace_entry_instructions(process, &[encoding]);
}

#[test]
fn worker_slice_moves_thread_state_out_of_the_process_until_reconciliation() {
    let mut process = synthetic_process_for_coordinator(1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(3);

    let execution = process
        .begin_thread_execution(thread, vcpu, nixe_cpu::execution::VcpuEventState::default())
        .unwrap();
    assert!(process.main_thread().state.is_none());

    process.abort_thread_execution(thread, vcpu, Some(execution));
    assert!(process.main_thread().state.is_some());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Faulted
    );
}

#[test]
fn unsynchronized_cpu_failure_discards_state_and_allows_retirement() {
    let mut process = synthetic_process_for_coordinator(1);
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
    assert!(process.main_thread().state.is_none());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Faulted
    );

    drop(cpu);
    process.complete_cpu_thread_retirement();
}

#[test]
fn concurrent_stop_does_not_hide_an_unsynchronized_native_failure() {
    let mut process = synthetic_process_for_coordinator(1);
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
    assert!(process.main_thread().state.is_none());
}

#[test]
fn panicked_worker_does_not_restore_an_incoming_snapshot() {
    let mut process = synthetic_process_for_coordinator(1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(0);
    let execution = process
        .begin_thread_execution(thread, vcpu, Default::default())
        .unwrap();
    drop(execution);
    process.abort_thread_execution(thread, vcpu, None);
    assert!(process.main_thread().state.is_none());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Faulted
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

pub(crate) fn synthetic_process_for_coordinator(process_id: u64) -> RunnableProcess {
    let (_directory, plan) = plan();
    reference_process_builder()
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
    let mut process = reference_process_builder()
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
        stack_top: process.main_thread().stack_top,
        priority: 44,
        ideal_vcpu: Some(core),
        affinity: profile.all_cores(),
    };
    for id in 2..=10 {
        process
            .commit_created_thread(nixe_scheduler::GuestThreadId::new(id), &request)
            .unwrap();
    }
    assert_eq!(memory.account().used_bytes(), memory.account().limit());
    assert_eq!(
        memory.system_account().used_bytes(),
        kernel_before + PAGE_SIZE
    );
}

pub(crate) fn synthetic_instruction_process_for_coordinator(
    process_id: u64,
    encodings: &[u32],
) -> RunnableProcess {
    let mut process = synthetic_process_for_coordinator(process_id);
    replace_entry_instructions(&mut process, encodings);
    process
}

pub(crate) fn synthetic_svc_process_for_coordinator(process_id: u64) -> RunnableProcess {
    let mut process = synthetic_process_for_coordinator(process_id);
    replace_entry_instruction(&mut process, 0xd400_4681);
    process
}

pub(crate) fn synthetic_memory_loop_process(process_id: u64) -> RunnableProcess {
    synthetic_native_instruction_process(
        process_id,
        &[0x10000001, 0xf9400020, 0xd4000021, 0x17fffffd],
    )
}

pub(crate) fn synthetic_native_instruction_process(
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
    reference_process_builder()
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
    let mut process = synthetic_svc_process_for_coordinator(1);
    let memory = std::sync::Arc::clone(process.memory());
    let handles = std::sync::Arc::clone(process.handles());
    let report = process.run().unwrap();
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));
    let state = process.main_thread_mut().take_state().unwrap();
    assert!(state.is_resident());
    let expected = state.register_context();
    assert!(memory.account().used_bytes() != 0);
    drop(process);
    assert_eq!(memory.account().used_bytes(), 0);
    assert_eq!(memory.account().machine().statistics().allocated_bytes, 0);
    assert!(handles.is_empty());
    assert_eq!(
        handles.insert(crate::EventObject::new()),
        Err(crate::HandleError::Closed)
    );
    assert_eq!(state.register_context(), expected);
}
