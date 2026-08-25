use std::cell::RefCell;

use nixe_cpu::{
    decode::{self, DecodeResult, DecodeSupport, InstructionPattern},
    exclusive::ExclusiveMonitorState,
    ir::terminator::Terminator,
    location::{ExecutionState, InstructionSize, LocationDescriptor},
    memory::{MemoryPermissions, SyntheticMemory},
    profile::{GuestCpuProfile, ProcessCpuContext},
    state::{ThreadCpuState, a32::A32State},
    translate::{RegionTranslationConfig, translate_region},
};
use nixe_cpu_engine_interpreter::{InterpreterContext, execute_one_with_context};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};

const SPACE: AddressSpaceId = AddressSpaceId::new(71);
const DATA_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(72);
const CODE_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(73);
const CODE_ADDRESS: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);

fn patterns() -> impl Iterator<Item = &'static InstructionPattern> {
    decode::a64::patterns()
        .iter()
        .chain(decode::a32::patterns())
        .chain(decode::t32::patterns_16())
        .chain(decode::t32::patterns_32())
}

fn state_for(execution_state: ExecutionState) -> ThreadCpuState {
    match execution_state {
        ExecutionState::A64 => ThreadCpuState::A64(Box::default()),
        ExecutionState::A32 => ThreadCpuState::A32(Box::new(A32State::a32())),
        ExecutionState::T32 => ThreadCpuState::A32(Box::new(A32State::t32())),
    }
}

fn instruction_bytes(pattern: &InstructionPattern) -> Vec<u8> {
    let encoding = pattern.regression_fixture.unwrap().encoding;
    match (pattern.execution_state, encoding.size()) {
        (ExecutionState::T32, InstructionSize::Bits32) => {
            let bits = encoding.bits();
            [(bits >> 16) as u16, bits as u16]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect()
        }
        (_, InstructionSize::Bits16) => encoding.bits().to_le_bytes()[..2].to_vec(),
        (_, InstructionSize::Bits32) => encoding.bits().to_le_bytes().to_vec(),
    }
}

#[test]
fn every_switch1_interpreter_registry_entry_has_a_non_fallback_ir_lifter() {
    let profile = GuestCpuProfile::switch_1();
    for pattern in patterns().filter(|pattern| pattern.decoder == DecodeSupport::Ready) {
        if !profile
            .allowed_execution_states()
            .contains(pattern.execution_state)
        {
            continue;
        }
        let fixture = pattern
            .regression_fixture
            .unwrap_or_else(|| panic!("{} has no parity fixture", pattern.coverage_id));
        let location = LocationDescriptor::new(CODE_ADDRESS, pattern.execution_state, profile.id());
        match decode::decode(&profile, location, fixture.encoding) {
            DecodeResult::ProfileDisabled { .. } => continue,
            DecodeResult::Decoded(decoded) => {
                assert_eq!(decoded.instruction.coverage_id(), pattern.coverage_id)
            }
            other => panic!(
                "{} {} parity fixture was not decodable: {other:?}",
                pattern.execution_state, pattern.coverage_id
            ),
        }

        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(DATA_PAGE));
        assert!(memory.add_ram_page(CODE_PAGE));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0),
            DATA_PAGE,
            MemoryPermissions::READ_WRITE,
        ));
        assert!(memory.map_page(
            SPACE,
            CODE_ADDRESS,
            CODE_PAGE,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.initialize_ram(CODE_PAGE, 0, &instruction_bytes(pattern)));

        let monitor = RefCell::new(ExclusiveMonitorState::default());
        let context = InterpreterContext::new(ProcessCpuContext::new(profile, SPACE))
            .with_memory(&memory)
            .with_exclusive_monitor(&monitor);
        let mut state = state_for(pattern.execution_state);
        if let Err(error) = execute_one_with_context(context, &mut state, fixture.encoding) {
            panic!(
                "{} {} is registry-ready but the interpreter rejected its fixture: {error}",
                pattern.execution_state, pattern.coverage_id,
            );
        }

        let region = translate_region(
            RegionTranslationConfig {
                max_blocks: core::num::NonZeroU32::new(1).unwrap(),
                max_guest_instructions: core::num::NonZeroU32::new(1).unwrap(),
                max_guest_instructions_per_block: core::num::NonZeroU32::new(1).unwrap(),
                ..RegionTranslationConfig::default()
            },
            &profile,
            SPACE,
            location,
            &memory,
        )
        .unwrap();
        assert!(
            !matches!(
                region.entry_block().terminator,
                Terminator::InterpretOne { .. } | Terminator::UnsupportedInstruction { .. }
            ),
            "{} {} interpreter fixture lowered to {:?}",
            pattern.execution_state,
            pattern.coverage_id,
            region.entry_block().terminator
        );
    }
}
