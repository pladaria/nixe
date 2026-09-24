use nixe_cpu::execution::{
    ArchitecturalTimer, CpuExit, CpuThreadId, MemoryBinding, TimerSnapshot, VcpuEventState,
};
use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_interpreter::{InterpreterProcess, InterpreterRunRequest};
use nixe_cpu_jit::{JitProcess, JitThread};
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidationSource,
};
use std::sync::Arc;

const SPACE: AddressSpaceId = AddressSpaceId::new(7);
const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);

struct FixedTimer;

impl ArchitecturalTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}

#[test]
fn concrete_interpreter_and_jit_match_at_an_architectural_boundary() {
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let code = [0xd503_201f_u32, 0xd420_0000];
    let mut memory = executable_memory(&code);
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: memory.as_ref(),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };

    let mut interpreter_process = InterpreterProcess::new(cpu);
    interpreter_process.bind_memory(binding).unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let jit_process = Arc::new(JitProcess::new(cpu, memory.clone()).unwrap());
    let mut jit = JitThread::new(jit_process.clone()).unwrap();

    let mut interpreter_state = a64_state();
    let mut jit_state = a64_state();
    let interpreter_report = interpreter
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            interpreter_request(&memory, &mut interpreter_state, 2),
        )
        .unwrap();
    let jit_report = jit
        .run_slice(
            &mut nixe_cpu_jit::ReturnStack::default(),
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            &mut jit_state,
            2,
            &FixedTimer,
            &VcpuEventState::default(),
        )
        .unwrap();

    assert_eq!(interpreter_state, jit_state);
    assert!(matches!(
        interpreter_report.stop,
        CpuExit::ArchitecturalException { .. }
    ));
    assert_eq!(interpreter_report.stop, jit_report.stop);

    drop(jit);
    assert!(jit_process.try_shutdown().unwrap());
}

#[test]
fn switch_1_pointer_authentication_hint_family_is_differentially_nop() {
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let hints = [
        0xd503_20ff_u32,
        0xd503_211f,
        0xd503_215f,
        0xd503_219f,
        0xd503_21df,
        0xd503_231f,
        0xd503_233f,
        0xd503_235f,
        0xd503_237f,
        0xd503_239f,
        0xd503_23bf,
        0xd503_23df,
        0xd503_23ff,
    ];
    let mut code = hints.to_vec();
    code.push(0xd420_0000);
    let mut memory = executable_memory(&code);
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: memory.as_ref(),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };

    let mut interpreter_process = InterpreterProcess::new(cpu);
    interpreter_process.bind_memory(binding).unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let jit_process = Arc::new(JitProcess::new(cpu, memory.clone()).unwrap());
    let mut jit = JitThread::new(jit_process.clone()).unwrap();

    let link_register = A64Register::General(A64GeneralRegister::new(30).unwrap());
    let signed_pointer = 0xabcd_0000_7518_7c14;
    let mut interpreter_state = a64_state();
    interpreter_state.write_x(link_register, signed_pointer);
    let mut jit_state = interpreter_state.clone();
    let interpreter_budget = hints.len() as u64 + 1;
    let interpreter_report = interpreter
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            interpreter_request(&memory, &mut interpreter_state, interpreter_budget),
        )
        .unwrap();
    let jit_report = jit
        .run_slice(
            &mut nixe_cpu_jit::ReturnStack::default(),
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            &mut jit_state,
            interpreter_budget,
            &FixedTimer,
            &VcpuEventState::default(),
        )
        .unwrap();

    assert_eq!(interpreter_state, jit_state);
    assert_eq!(interpreter_state.read_x(link_register), signed_pointer);
    assert!(matches!(
        interpreter_report.stop,
        CpuExit::ArchitecturalException { .. }
    ));
    assert_eq!(interpreter_report.stop, jit_report.stop);

    drop(jit);
    assert!(jit_process.try_shutdown().unwrap());
}

fn executable_memory(code: &[u32]) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    let bytes: Vec<_> = code.iter().copied().flat_map(u32::to_le_bytes).collect();
    memory.initialize_ram(page, 0, &bytes).unwrap();
    assert!(memory.map_page(SPACE, CODE, page, MemoryPermissions::READ_EXECUTE));
    memory
}

fn a64_state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(CODE.get());
    state
}

fn interpreter_request<'a>(
    memory: &'a ExecutionMemory,
    state: &'a mut A64State,
    instruction_budget: u64,
) -> InterpreterRunRequest<'a> {
    InterpreterRunRequest {
        memory,
        memory_lease: Some(memory.acquire_execution_lease()),
        state,
        instruction_budget,
        timer: &FixedTimer,
        events: VcpuEventState::default(),
    }
}
