use std::cell::RefCell;

use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, CpuThreadId, MemoryBinding, TimerSnapshot};
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment,
    MemoryOrdering, MemoryPermissions, MemoryValue, SyntheticMemory,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_interpreter::{
    InstructionStep, InterpreterContext, InterpreterProcess, InterpreterRunRequest,
    execute_one_with_context,
};
use nixe_memory::GuestPhysicalPageId;
use nixe_memory::{
    AddressSpaceId, DirectBackendPolicy, GuestVirtualAddress, MemoryInvalidationSource,
};

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
            encoding,
        )
        .unwrap();

        let mut adapted = initial;
        let mut process = InterpreterProcess::new(cpu);
        let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
        let report = thread
            .run_slice(InterpreterRunRequest {
                memory: &memory,
                memory_lease: None,
                state: &mut adapted,
                instruction_budget: 1,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu::execution::VcpuEventState::default(),
            })
            .unwrap();
        assert_eq!(adapted, direct);
        assert_eq!(report.progress, 1);
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

#[test]
fn linux_direct_interpreter_stores_continue_after_the_first_protected_write() {
    const DATA: u64 = 0x8000;
    let cases = [
        (0x3900_0020_u32, MemoryAccessSize::Byte, 0xa5_u64, 0x5a_u64),
        (0x7900_0020, MemoryAccessSize::Halfword, 0xa5c3, 0x5a3c),
        (
            0xb900_0020,
            MemoryAccessSize::Word,
            0xa5c3_9678,
            0x5a3c_6987,
        ),
        (
            0xf900_0020,
            MemoryAccessSize::Doubleword,
            0xa5c3_9678_1234_fedc,
            0x5a3c_6987_edcb_0123,
        ),
    ];
    for (encoding, size, first, second) in cases {
        let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
        let memory = direct_memory(encoding, DATA);
        let binding = direct_binding(&memory);
        let mut process = InterpreterProcess::new(cpu);
        process.bind_memory(binding).unwrap();
        let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
        let mut state = direct_state(DATA, first);

        let first_report = thread
            .run_slice(direct_request(&memory, &mut state))
            .unwrap();
        assert_eq!(first_report.stop, CpuExit::BudgetExhausted);
        assert_eq!(read_value(&memory, DATA, size), first);

        state.set_pc(CODE);
        write_register(&mut state, 0, second);
        let second_report = thread
            .run_slice(direct_request(&memory, &mut state))
            .unwrap();
        assert_eq!(second_report.stop, CpuExit::BudgetExhausted);
        assert_eq!(read_value(&memory, DATA, size), second);
    }
}

#[test]
fn linux_direct_interpreter_reads_directly_mapped_values() {
    const DATA: u64 = 0x8000;
    for (encoding, size, value) in [
        (0x3940_0020_u32, MemoryAccessSize::Byte, 0xa5_u64),
        (0x7940_0020, MemoryAccessSize::Halfword, 0xa5c3),
        (0xb940_0020, MemoryAccessSize::Word, 0xa5c3_9678),
        (
            0xf940_0020,
            MemoryAccessSize::Doubleword,
            0xa5c3_9678_1234_fedc,
        ),
    ] {
        let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
        let mut memory = direct_memory(encoding, DATA);
        assert!(memory.initialize_ram(
            GuestPhysicalPageId::new(2),
            0,
            &value.to_le_bytes()[..size.bytes()],
        ));
        let binding = direct_binding(&memory);
        let mut process = InterpreterProcess::new(cpu);
        process.bind_memory(binding).unwrap();
        let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
        let mut state = direct_state(DATA, 0);

        let report = thread
            .run_slice(direct_request(&memory, &mut state))
            .unwrap();
        assert_eq!(report.stop, CpuExit::BudgetExhausted);
        assert_eq!(read_register(&state, 0), value);
    }
}

#[test]
fn linux_direct_interpreter_preserves_unaligned_ordinary_accesses() {
    const DATA: u64 = 0x8000;
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
    let memory = direct_memory(0xf900_0020, DATA);
    let binding = direct_binding(&memory);
    let mut process = InterpreterProcess::new(cpu);
    process.bind_memory(binding).unwrap();
    let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
    let address = DATA + 4;
    let value = 0xa5c3_9678_1234_fedc;
    let mut state = direct_state(address, value);

    let report = thread
        .run_slice(direct_request(&memory, &mut state))
        .unwrap();
    assert_eq!(report.stop, CpuExit::BudgetExhausted);
    let access = MemoryAccess::new(
        MemoryAccessSize::Doubleword,
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    );
    assert_eq!(
        memory
            .read(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(address),
                access
            )
            .unwrap()
            .value,
        MemoryValue::U64(value),
    );
}

#[test]
fn linux_direct_interpreter_rejects_a_replacement_arena_before_native_entry() {
    const DATA: u64 = 0x8000;
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
    let first = direct_memory(0xf940_0020, DATA);
    let replacement = direct_memory(0xf940_0020, DATA);
    let mut process = InterpreterProcess::new(cpu);
    process.bind_memory(direct_binding(&first)).unwrap();
    let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
    drop(first);
    let mut state = direct_state(DATA, 0);

    let error = thread
        .run_slice(direct_request(&replacement, &mut state))
        .unwrap_err();
    assert_eq!(error.kind, nixe_cpu::execution::CpuFaultKind::Internal);
    assert!(
        error
            .message
            .contains("differs from its immutable process binding")
    );
}

#[test]
fn linux_direct_interpreter_requires_a_live_mapping_lease() {
    const DATA: u64 = 0x8000;
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
    let memory = direct_memory(0xf940_0020, DATA);
    let mut process = InterpreterProcess::new(cpu);
    process.bind_memory(direct_binding(&memory)).unwrap();
    let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
    let mut state = direct_state(DATA, 0);

    let error = thread
        .run_slice(InterpreterRunRequest {
            memory: &memory,
            memory_lease: None,
            state: &mut state,
            instruction_budget: 1,
            loader_return: None,
            timer: &FixedTimer,
            events: nixe_cpu::execution::VcpuEventState::default(),
        })
        .unwrap_err();
    assert_eq!(error.kind, nixe_cpu::execution::CpuFaultKind::Internal);
    assert!(error.message.contains("requires its live mapping lease"));
}

#[test]
fn linux_direct_interpreter_reports_unmapped_fault_with_exact_prefault_state() {
    const DATA: u64 = 0x8000;
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &0xf940_0020_u32.to_le_bytes()));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    memory
        .bind_cpu_memory_backend(
            AddressSpaceId::new(1),
            0x1_0000,
            DirectBackendPolicy::Required,
        )
        .unwrap();
    let mut process = InterpreterProcess::new(cpu);
    process.bind_memory(direct_binding(&memory)).unwrap();
    let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
    let mut state = direct_state(DATA, 0xfeed_face_cafe_beef);
    let before = state.clone();

    let report = thread
        .run_slice(direct_request(&memory, &mut state))
        .unwrap();
    let CpuExit::DataFault { source, fault } = report.stop else {
        panic!("unmapped direct load did not produce a guest data fault");
    };
    assert_eq!(source.pc, GuestVirtualAddress::new(CODE));
    assert_eq!(fault.address, GuestVirtualAddress::new(DATA));
    assert_eq!(report.progress, 1);
    assert_eq!(state, before);
}

#[test]
fn native_store_through_an_alias_invalidates_exclusive_reservations() {
    const DATA: u64 = 0x8000;
    const ALIAS: u64 = 0x9000;
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
    let memory = direct_memory_with_alias(0x3900_0020, DATA, ALIAS, false);
    let mut process = InterpreterProcess::new(cpu);
    process.bind_memory(direct_binding(&memory)).unwrap();
    let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
    let mut state = direct_state(ALIAS, 0x11);

    thread
        .run_slice(direct_request(&memory, &mut state))
        .unwrap();
    let access = MemoryAccess::normal(MemoryAccessSize::Byte);
    let (_, reservation) = memory
        .load_exclusive(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(DATA),
            access,
        )
        .unwrap();

    state.set_pc(CODE);
    write_register(&mut state, 0, 0x22);
    thread
        .run_slice(direct_request(&memory, &mut state))
        .unwrap();
    let (_, stored) = memory
        .store_exclusive(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(DATA),
            access,
            MemoryValue::U8(0x33),
            reservation,
        )
        .unwrap();
    assert!(!stored);
    assert_eq!(read_value(&memory, DATA, MemoryAccessSize::Byte), 0x22);
}

#[test]
fn executable_physical_alias_uses_the_same_direct_store_path() {
    const DATA: u64 = 0x8000;
    const EXECUTABLE_ALIAS: u64 = 0x9000;
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, AddressSpaceId::new(1));
    let memory = direct_memory_with_alias(0x3900_0020, DATA, EXECUTABLE_ALIAS, true);
    let mut process = InterpreterProcess::new(cpu);
    process.bind_memory(direct_binding(&memory)).unwrap();
    let mut thread = process.create_thread(CpuThreadId::new(1)).unwrap();
    let mut state = direct_state(DATA, 0x44);

    thread
        .run_slice(direct_request(&memory, &mut state))
        .unwrap();
    state.set_pc(CODE);
    write_register(&mut state, 0, 0x55);
    thread
        .run_slice(direct_request(&memory, &mut state))
        .unwrap();

    assert_eq!(read_value(&memory, DATA, MemoryAccessSize::Byte), 0x55);
}

fn direct_memory(encoding: u32, data: u64) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    assert!(memory.initialize_ram(code_page, 0, &encoding.to_le_bytes()));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(data),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    assert_eq!(
        memory
            .bind_cpu_memory_backend(
                AddressSpaceId::new(1),
                0x1_0000,
                DirectBackendPolicy::Required,
            )
            .unwrap(),
        nixe_memory::CpuMemoryBackend::LinuxDirect,
    );
    memory
}

fn direct_memory_with_alias(
    encoding: u32,
    data: u64,
    alias: u64,
    executable_alias: bool,
) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    assert!(memory.initialize_ram(code_page, 0, &encoding.to_le_bytes()));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(data),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(alias),
        data_page,
        if executable_alias {
            MemoryPermissions::READ_EXECUTE
        } else {
            MemoryPermissions::READ_WRITE
        },
    ));
    memory
        .bind_cpu_memory_backend(
            AddressSpaceId::new(1),
            0x1_0000,
            DirectBackendPolicy::Required,
        )
        .unwrap();
    memory
}

fn direct_binding(memory: &ExecutionMemory) -> MemoryBinding<'_> {
    MemoryBinding {
        address_space: AddressSpaceId::new(1),
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    }
}

fn direct_state(data: u64, value: u64) -> A64State {
    let mut state = state();
    write_register(&mut state, 0, value);
    write_register(&mut state, 1, data);
    state
}

fn direct_request<'a>(
    memory: &'a ExecutionMemory,
    state: &'a mut A64State,
) -> InterpreterRunRequest<'a> {
    InterpreterRunRequest {
        memory,
        memory_lease: Some(memory.acquire_execution_lease()),
        state,
        instruction_budget: 1,
        loader_return: None,
        timer: &FixedTimer,
        events: nixe_cpu::execution::VcpuEventState::default(),
    }
}

fn write_register(state: &mut A64State, register: u8, value: u64) {
    state.write_x(
        A64Register::General(A64GeneralRegister::new(register).unwrap()),
        value,
    );
}

fn read_register(state: &A64State, register: u8) -> u64 {
    state.read_x(A64Register::General(
        A64GeneralRegister::new(register).unwrap(),
    ))
}

fn read_value(memory: &ExecutionMemory, address: u64, size: MemoryAccessSize) -> u64 {
    match memory
        .read(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(address),
            MemoryAccess::normal(size),
        )
        .unwrap()
        .value
    {
        MemoryValue::U8(value) => u64::from(value),
        MemoryValue::U16(value) => u64::from(value),
        MemoryValue::U32(value) => u64::from(value),
        MemoryValue::U64(value) => value,
        MemoryValue::U128(_) => unreachable!(),
    }
}

fn state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(CODE);
    state
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
