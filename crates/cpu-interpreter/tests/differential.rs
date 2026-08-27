use std::cell::RefCell;

use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, CpuThreadId, RunRequest, TimerSnapshot};
use nixe_cpu::location::InstructionEncoding;
use nixe_cpu::memory::{MemoryPermissions, SyntheticMemory};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::{ThreadCpuState, a64::A64State};
use nixe_cpu_interpreter::{
    InstructionStep, InterpreterContext, InterpreterProcess, execute_one_with_context,
};
use nixe_memory::GuestPhysicalPageId;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

const CODE: u64 = 0x1000;

struct FixedTimer;
impl ArchitecturalTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 123,
            frequency: 19_200_000,
        }
    }
}

#[test]
fn concrete_thread_matches_direct_single_step_state_counts_and_stops() {
    for (encoding, expected_exception) in [(0x9100_0400_u32, false), (0xd400_0541, true)] {
        let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
        let memory = memory(encoding);
        let initial = state();
        let mut direct = initial.clone();
        let monitor = RefCell::new(ExclusiveMonitorState::default());
        let events = nixe_cpu::execution::VcpuEventState::default();
        let direct_outcome = execute_one_with_context(
            InterpreterContext::new(cpu, &memory, &monitor, &FixedTimer, &events),
            &mut direct,
            InstructionEncoding::from_u32(encoding),
        )
        .unwrap();

        let mut adapted = initial;
        let mut process = InterpreterProcess::new(cpu);
        let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
        let report = thread
            .run_slice(RunRequest {
                cpu,
                memory: &memory,
                state: &mut adapted,
                instruction_budget: 1,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu::execution::VcpuEventState::default(),
            })
            .unwrap();
        assert_eq!(adapted, direct);
        assert_eq!(report.instructions_executed, 1);
        assert_eq!(
            matches!(report.stop, CpuExit::SupervisorCall { .. }),
            expected_exception
        );
        assert_eq!(
            matches!(
                direct_outcome,
                InstructionStep::Exit(CpuExit::SupervisorCall { .. })
            ),
            expected_exception
        );
    }
}

fn state() -> ThreadCpuState {
    let mut state = A64State::default();
    state.set_pc(CODE);
    ThreadCpuState::A64(Box::new(state))
}

fn memory(encoding: u32) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    assert!(memory.initialize_ram(page, 0, &encoding.to_le_bytes()));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(CODE),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    memory
}
