//! Canonical memory fixtures for environment capture.
use crate::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::{CanonicalAllocation, CanonicalWriteBatch, MemoryPermissions};
pub(crate) fn write_shader_words(address_space: &MaxwellGpuAddressSpace, writes: &[(u64, u32)]) {
    let mut batch = CanonicalWriteBatch::new();
    for &(address, value) in writes {
        let range = address_space
            .resolve_range(
                address_space.address(address).unwrap(),
                4,
                MemoryPermissions::READ,
            )
            .unwrap();
        let mut offset = 0;
        let bytes = value.to_le_bytes();
        for segment in range.segments() {
            let end = offset + segment.size() as usize;
            batch
                .stage(
                    segment.mapping().backing(),
                    segment.backing_offset(),
                    &bytes[offset..end],
                )
                .unwrap();
            offset = end;
        }
    }
    batch.commit_ordered().unwrap();
}

pub(crate) fn mapped_memory() -> (CanonicalAllocation, MaxwellGpuAddressSpace, u64) {
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mapping = address_space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(1),
            backing: allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
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
