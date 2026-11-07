use nixe_memory::{Allocation, AllocationRange, Machine, PAGE_SIZE};

pub fn allocate(machine: &Machine, bytes: u64) -> Allocation {
    machine.account(bytes).unwrap().allocate(bytes).unwrap()
}
pub fn ranges(allocation: &Allocation) -> Vec<AllocationRange> {
    let pin = allocation.pin();
    let bytes = pin.bytes();
    vec![AllocationRange::new(pin, 0, bytes).unwrap()]
}
pub fn write(allocation: &Allocation, offset: u64, bytes: &[u8]) {
    let input = AllocationRange::new(allocation.pin(), offset, bytes.len() as u64).unwrap();
    input
        .allocation()
        .machine()
        .write_resolved(std::slice::from_ref(&input), bytes)
        .unwrap();
}
pub fn address_space(machine: &Machine, id: u64) -> nixe_gpu_maxwell::MaxwellGpuAddressSpace {
    use nixe_gpu_maxwell::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellGpuAddressSpace,
        SWITCH_1_GM20B_PROFILE,
    };
    let mut space = MaxwellGpuAddressSpace::new(
        nixe_gpu_maxwell::MaxwellDeviceMemory::new(machine.clone()),
        MaxwellAddressSpaceId::new(id),
        SWITCH_1_GM20B_PROFILE,
    );
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    space
}
pub fn map(
    space: &mut nixe_gpu_maxwell::MaxwellGpuAddressSpace,
    allocation: &Allocation,
    id: u64,
) -> nixe_gpu_maxwell::MaxwellGpuMapping {
    use nixe_gpu_maxwell::{MaxwellAllocationId, MaxwellMapRequest};
    space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(id),
            backing: space
                .memory()
                .device_memory()
                .map((ranges(allocation)).into(), None)
                .unwrap(),
            backing_offset: 0,
            size: allocation.pin().bytes(),
            allocation_alignment: PAGE_SIZE as u32,
            page_size: PAGE_SIZE as u32,
            kind: 0,
            cacheable: false,

            fixed_offset: None,
        })
        .unwrap()
}
