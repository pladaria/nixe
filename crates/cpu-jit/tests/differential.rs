use nixe_cpu::execution::{
    ArchitecturalTimer, CpuExit, CpuProcessId, CpuThreadId, MemoryBinding, RunRequest,
    TimerSnapshot, VcpuEventState,
};
use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::{ThreadCpuState, a64::A64State};
use nixe_cpu_interpreter::InterpreterProcess;
use nixe_cpu_jit::{JitConfiguration, JitProcess};
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidationSource,
};

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
fn concrete_interpreter_and_jit_match_on_a_bounded_synthetic_process() {
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    assert!(memory.initialize_ram(page, 0, &0xd503_201f_u32.to_le_bytes()));
    assert!(memory.map_page(SPACE, CODE, page, MemoryPermissions::READ_EXECUTE));
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };

    let mut interpreter_process = InterpreterProcess::new(cpu);
    interpreter_process.bind_memory(binding).unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let mut jit_process =
        JitProcess::new(CpuProcessId::new(2), cpu, JitConfiguration::default()).unwrap();
    jit_process.bind_memory(binding).unwrap();
    let mut jit = jit_process.create_thread(CpuThreadId::new(2)).unwrap();

    let mut interpreter_state = state();
    let mut jit_state = state();
    let interpreter_report = interpreter
        .run_slice(request(cpu, &memory, &mut interpreter_state))
        .unwrap();
    let jit_report = jit
        .run_slice(request(cpu, &memory, &mut jit_state))
        .unwrap();

    assert_eq!(interpreter_state, jit_state);
    assert_eq!(interpreter_report.instructions_executed, 1);
    assert_eq!(jit_report.instructions_executed, 1);
    assert_eq!(interpreter_report.stop, CpuExit::BudgetExhausted);
    assert_eq!(jit_report.stop, CpuExit::BudgetExhausted);

    drop(jit);
    jit_process.shutdown().unwrap();
}

fn state() -> ThreadCpuState {
    let mut state = A64State::default();
    state.set_pc(CODE.get());
    ThreadCpuState::A64(Box::new(state))
}

fn request<'a>(
    cpu: ProcessCpuContext,
    memory: &'a ExecutionMemory,
    state: &'a mut ThreadCpuState,
) -> RunRequest<'a> {
    RunRequest {
        cpu,
        memory,
        state,
        instruction_budget: 1,
        loader_return: None,
        timer: &FixedTimer,
        events: VcpuEventState::default(),
    }
}
