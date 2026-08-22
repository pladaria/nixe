use std::cell::RefCell;

use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::location::InstructionEncoding;
use nixe_cpu::memory::{
    CpuMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue, SyntheticMemory,
};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a32::A32GeneralRegister};
use nixe_cpu_engine_interpreter::{InterpreterContext, execute_one_with_context};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};

const SPACE: AddressSpaceId = AddressSpaceId::new(7);
const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(9);
const PRIMARY: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x2000);

fn register(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).unwrap()
}

#[test]
fn a32_ldrex_strex_uses_local_monitor_and_canonical_alias_generation() {
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(SPACE, PRIMARY, PAGE, MemoryPermissions::READ_WRITE));
    assert!(memory.map_page(SPACE, ALIAS, PAGE, MemoryPermissions::READ_WRITE));
    assert!(memory.initialize_ram(PAGE, 0, &7_u32.to_le_bytes()));

    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let context =
        InterpreterContext::new(ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE))
            .with_memory(&memory)
            .with_exclusive_monitor(&monitor);
    let mut state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.write_r(register(3), PRIMARY.get() as u32);

    // LDREX r0, [r3]
    execute_one_with_context(
        context,
        &mut state,
        InstructionEncoding::from_u32(0xe193_0f9f),
    )
    .unwrap();
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(register(0)), 7);
    a32.write_r(register(0), 9);

    // Any write through a physical alias invalidates the reservation.
    memory
        .write(
            SPACE,
            ALIAS,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(8),
        )
        .unwrap();

    // STREX r1, r0, [r3]
    execute_one_with_context(
        context,
        &mut state,
        InstructionEncoding::from_u32(0xe183_1f90),
    )
    .unwrap();
    let ThreadCpuState::A32(a32) = &state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(register(1)), 1);
    assert_eq!(
        memory
            .read(SPACE, PRIMARY, MemoryAccess::normal(MemoryAccessSize::Word),)
            .unwrap()
            .value,
        MemoryValue::U32(8),
    );
}
