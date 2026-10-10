use super::*;

use std::fs;

use nixe_cpu::memory::{
    CpuMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue, SYNTHETIC_PAGE_SIZE,
};

use crate::{Launcher, LauncherInput};

fn reference_process_builder() -> ProcessBuilder {
    ProcessBuilder::default().with_cpu_backend(crate::CpuBackendConfig::Interpreter)
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
    let address_space = process.cpu.address_space_id();
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    let mapping = process.memory.mapping_info(address_space, entry).unwrap();
    let alias_page = GuestVirtualAddress::new(0x6000_0000);
    assert!(
        std::sync::Arc::get_mut(&mut process.memory)
            .unwrap()
            .map_page(
                address_space,
                alias_page,
                mapping.physical_page,
                MemoryPermissions::READ_WRITE,
            )
    );
    let page_offset = entry.get() % SYNTHETIC_PAGE_SIZE as u64;
    for (index, encoding) in encodings.iter().copied().enumerate() {
        let instruction_offset = u64::try_from(index).unwrap() * 4;
        process
            .memory
            .write(
                address_space,
                GuestVirtualAddress::new(alias_page.get() + page_offset + instruction_offset),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(encoding),
            )
            .unwrap();
    }
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
        .begin_thread_execution(
            thread,
            vcpu,
            1,
            nixe_cpu::execution::VcpuEventState::default(),
            true,
        )
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
    let mut cpu = process.create_worker_cpu_thread(vcpu).unwrap();
    let execution = process
        .begin_thread_execution(thread, vcpu, 1, Default::default(), true)
        .unwrap();
    let failure = ProcessExecutionError::Cpu {
        fault: nixe_cpu::execution::CpuFault {
            backend: "dynarmic",
            kind: nixe_cpu::execution::CpuFaultKind::Internal,
            progress: 0,
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
    process
        .cpu_thread_teardown_state()
        .prepare(&mut cpu)
        .unwrap();
}

#[test]
fn concurrent_stop_does_not_hide_an_unsynchronized_native_failure() {
    let mut process = synthetic_process_for_coordinator(1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(0);
    let execution = process
        .begin_thread_execution(thread, vcpu, 1, Default::default(), true)
        .unwrap();
    assert!(process.terminate_from_host());
    let failure = ProcessExecutionError::Cpu {
        fault: nixe_cpu::execution::CpuFault {
            backend: "dynarmic",
            kind: nixe_cpu::execution::CpuFaultKind::Internal,
            progress: 0,
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
        .begin_thread_execution(thread, vcpu, 1, Default::default(), true)
        .unwrap();
    drop(execution);
    process.abort_thread_execution(thread, vcpu, None);
    assert!(process.main_thread().state.is_none());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Faulted
    );
}

#[test]
fn runtime_mapping_mutation_requires_backend_acknowledgement_before_reentry() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let heap = process.memory_layout().heap();
    process
        .resize_memory_mapping(
            heap.base(),
            0,
            SYNTHETIC_PAGE_SIZE as u64,
            MemoryPermissions::READ_WRITE,
            MemoryMappingPurpose::Heap,
        )
        .unwrap();
    let cursor = process.memory.invalidation_cursor();
    assert!(!process.memory_invalidation_acknowledged(cursor));
    assert_eq!(
        process.run(0).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    assert!(process.memory_invalidation_acknowledged(cursor));
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
            address_space_id: AddressSpaceId::new(process_id),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap()
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

pub(crate) fn synthetic_memory_loop_process(
    process_id: u64,
    backend: crate::CpuBackendConfig,
) -> RunnableProcess {
    synthetic_backend_instruction_process(
        process_id,
        backend,
        &[0x10000001, 0xf9400020, 0xd503203f, 0x17fffffd],
    )
}

pub(crate) fn synthetic_backend_instruction_process(
    process_id: u64,
    backend: crate::CpuBackendConfig,
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
    ProcessBuilder::default()
        .with_cpu_backend(backend)
        .with_memory_backend(nixe_memory::DirectBackendPolicy::Required)
        .with_config(ProcessBuildConfig {
            process_id,
            address_space_id: AddressSpaceId::new(process_id),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap()
}

mod run;
