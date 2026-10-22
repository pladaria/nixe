//! Canonical memory fixtures for environment capture.
use crate::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::Allocation;
use nixe_memory::MemoryPermissions;
pub(crate) fn write_shader_words(address_space: &crate::MaxwellGpuMemory, writes: &[(u64, u32)]) {
    for &(address, value) in writes {
        let range = address_space
            .resolve_range(
                address_space.address(address).unwrap(),
                4,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        let inputs = address_space.capture(&range).unwrap();
        address_space
            .machine()
            .write_resolved(&inputs, &value.to_le_bytes())
            .unwrap();
    }
}

pub(crate) fn mapped_memory() -> (Allocation, MaxwellGpuAddressSpace, u64) {
    let machine = crate::test_memory::machine();
    let allocation = crate::test_memory::allocate(&machine, 0x1000);
    let mut address_space = MaxwellGpuAddressSpace::new(
        machine.clone(),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mapping = address_space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(1),
            backing: crate::test_memory::ranges(&allocation),
            backing_offset: 0,
            size: 0x1000,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    (allocation, address_space, mapping.offset().get())
}
