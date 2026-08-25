#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use nixe_cpu::{
    ir::{print::print_region, verify::verify_region},
    location::{ExecutionState, LocationDescriptor},
    memory::{MemoryPermissions, SYNTHETIC_PAGE_SIZE, SyntheticMemory},
    profile::GuestCpuProfile,
    translate::{MAX_GUEST_INSTRUCTIONS_PER_REGION, RegionTranslationConfig, translate_region},
};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};

const MAX_INPUT_BYTES: usize = 2 * SYNTHETIC_PAGE_SIZE + 16;
const MAX_DIAGNOSTIC_BYTES: usize = 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut header = [0_u8; 8];
    let header_length = data.len().min(header.len());
    header[..header_length].copy_from_slice(&data[..header_length]);

    let state = match header[0] % 3 {
        0 => ExecutionState::A64,
        1 => ExecutionState::A32,
        _ => ExecutionState::T32,
    };
    let profile = if state == ExecutionState::A64 && header[1] & 1 != 0 {
        GuestCpuProfile::switch_2_native()
    } else {
        GuestCpuProfile::switch_1()
    };
    let address_space = AddressSpaceId::new(1);
    let base = 0x1000_u64;
    let offset = match (state, header[2] % 3) {
        (ExecutionState::T32, 1) => SYNTHETIC_PAGE_SIZE - 2,
        (_, 1) => SYNTHETIC_PAGE_SIZE - 4,
        (_, 2) => SYNTHETIC_PAGE_SIZE / 2,
        _ => 0,
    };
    let start_address = GuestVirtualAddress::new(base + offset as u64);

    let mut memory = SyntheticMemory::new();
    for index in 0..2_u64 {
        let page = GuestPhysicalPageId::new(index + 1);
        assert!(memory.add_ram_page(page));
        let selector = header[3 + index as usize] % 4;
        let permissions = match selector {
            0 => MemoryPermissions::NONE,
            1 => MemoryPermissions::READ,
            2 => MemoryPermissions::EXECUTE,
            _ => MemoryPermissions::READ_EXECUTE,
        };
        if header[5] & (1 << index) != 0 {
            assert!(memory.map_page(
                address_space,
                GuestVirtualAddress::new(base + index * SYNTHETIC_PAGE_SIZE as u64),
                page,
                permissions,
            ));
        }
    }

    let payload = data.get(8..).unwrap_or_default();
    let first_capacity = SYNTHETIC_PAGE_SIZE - offset;
    let first_length = payload.len().min(first_capacity);
    assert!(memory.initialize_ram(
        GuestPhysicalPageId::new(1),
        offset,
        &payload[..first_length]
    ));
    let second = &payload[first_length..];
    let second_length = second.len().min(SYNTHETIC_PAGE_SIZE);
    assert!(memory.initialize_ram(GuestPhysicalPageId::new(2), 0, &second[..second_length],));
    if header[6] & 1 != 0 {
        memory.inject_instruction_fault(address_space, start_address, "fuzzed fetch fault");
    }

    let instruction_limit = u32::from(header[7] % 64) + 1;
    let config = RegionTranslationConfig {
        max_guest_instructions: NonZeroU32::new(instruction_limit).unwrap(),
        max_guest_instructions_per_block: NonZeroU32::new(instruction_limit).unwrap(),
        ..RegionTranslationConfig::default()
    };
    let result = translate_region(
        config,
        &profile,
        address_space,
        LocationDescriptor::new(start_address, state, profile.id()),
        &memory,
    );
    assert!(memory.physical_page_count() <= 2);
    match result {
        Ok(region) => {
            let instruction_count = region.metadata.guest_instruction_count as usize;
            assert!(instruction_count <= instruction_limit as usize);
            assert!(instruction_count <= MAX_GUEST_INSTRUCTIONS_PER_REGION as usize);
            assert!(region.metadata.ir_operation_count as usize <= instruction_count * 64);
            assert!(verify_region(&region).is_ok());
            assert!(print_region(&region, Default::default()).len() <= MAX_DIAGNOSTIC_BYTES);
        }
        Err(error) => assert!(error.to_string().len() <= 4_096),
    }
});
