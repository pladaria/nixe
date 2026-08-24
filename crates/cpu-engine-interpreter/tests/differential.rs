use nixe_cpu::location::InstructionEncoding;
use nixe_cpu::memory::{MemoryPermissions, SyntheticMemory};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a64::A64State};
use nixe_cpu_engine::{
    DomainRequest, EngineDomain, EngineDomainId, EngineExecutorId, EngineExit, EngineTimer,
    RunRequest, TimerSnapshot,
};
use nixe_cpu_engine_interpreter::InterpreterDomain;
use nixe_cpu_engine_interpreter::{
    InterpreterContext, InterpreterOutcome, execute_one_with_context,
};
use nixe_memory::GuestPhysicalPageId;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

const CODE: u64 = 0x1000;

struct FixedTimer;
impl EngineTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 123,
            frequency: 19_200_000,
        }
    }
}

#[test]
fn engine_domain_matches_direct_single_step_state_counts_and_stops() {
    for (encoding, expected_exception) in [(0x9100_0400_u32, false), (0xd400_0541, true)] {
        let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(1));
        let memory = memory(encoding);
        let initial = state();
        let mut direct = initial.clone();
        let direct_outcome = execute_one_with_context(
            InterpreterContext::new(cpu).with_memory(&memory),
            &mut direct,
            InstructionEncoding::from_u32(encoding),
        )
        .unwrap();

        let mut adapted = initial;
        let mut domain = InterpreterDomain::new(EngineDomainId::new(1));
        let mut executor = domain.create_executor(EngineExecutorId::new(1)).unwrap();
        let report = executor
            .run_slice(RunRequest {
                cpu,
                memory: &memory,
                state: &mut adapted,
                instruction_budget: 1,
                loader_return: None,
                timer: &FixedTimer,
            })
            .unwrap();
        assert_eq!(adapted, direct);
        assert_eq!(report.instructions_executed, 1);
        assert_eq!(
            matches!(report.stop, EngineExit::SupervisorCall { .. }),
            expected_exception
        );
        assert_eq!(
            matches!(direct_outcome, InterpreterOutcome::Exception { .. }),
            expected_exception
        );
    }
}

#[test]
fn provider_domain_creation_uses_only_the_neutral_request() {
    let request = DomainRequest {
        domain: EngineDomainId::new(4),
        cpu: ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(1)),
    };
    let domain = InterpreterDomain::new(request.domain);
    assert_eq!(domain.domain_id(), EngineDomainId::new(4));
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
