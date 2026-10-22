use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::{
    AccessOperation, AccessResult, AllocationRange, Machine, MemoryPermissions, PAGE_SIZE,
};

fn fixture() -> (
    Machine,
    MaxwellGpuAddressSpace,
    nixe_gpu_maxwell::MaxwellGpuMapping,
) {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let account = machine.account(3 * PAGE_SIZE).unwrap();
    let first = account.allocate(PAGE_SIZE).unwrap();
    let gap = account.allocate(PAGE_SIZE).unwrap();
    let last = account.allocate(PAGE_SIZE).unwrap();
    let input = vec![
        AllocationRange::new(first.pin(), 0, PAGE_SIZE).unwrap(),
        AllocationRange::new(last.pin(), 0, PAGE_SIZE).unwrap(),
    ];
    machine
        .write_resolved(&input[..1], &vec![0x11; PAGE_SIZE as usize])
        .unwrap();
    machine
        .write_resolved(&input[1..], &vec![0x22; PAGE_SIZE as usize])
        .unwrap();
    let mut space = MaxwellGpuAddressSpace::new(
        machine.clone(),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mapping = space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(1),
            backing: input,
            backing_offset: 0,
            size: 2 * PAGE_SIZE,
            allocation_alignment: PAGE_SIZE as u32,
            page_size: PAGE_SIZE as u32,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    drop((first, last, gap));
    (machine, space, mapping)
}

#[test]
fn ordinary_scattered_command_reads_need_no_backend_or_owned_intermediate_bytes() {
    let (machine, space, mapping) = fixture();
    let memory = space.memory().clone();
    let start = memory
        .address(mapping.offset().get() + PAGE_SIZE - 7)
        .unwrap();
    let range = memory
        .resolve_range(start, 19, MemoryPermissions::READ)
        .unwrap();
    let mut bytes = [0; 19];
    memory
        .read_resolved(&range, &mut bytes, &mut |_, _| {
            panic!("ordinary RAM must be read directly")
        })
        .unwrap();
    assert_eq!(&bytes[..7], &[0x11; 7]);
    assert_eq!(&bytes[7..], &[0x22; 12]);
    drop(space);
    assert_eq!(
        machine.statistics().allocated_bytes,
        0,
        "weak resolved metadata cannot retain RAM"
    );
}

#[test]
fn device_read_owns_exact_inputs_and_releases_mapping_lock_before_entering_owner() {
    let (machine, mut space, mapping) = fixture();
    let memory = space.memory().clone();
    let start = memory
        .address(mapping.offset().get() + PAGE_SIZE - 7)
        .unwrap();
    let range = memory
        .resolve_range(start, 19, MemoryPermissions::READ)
        .unwrap();
    machine
        .publish_device_write(&mapping.physical().pin(PAGE_SIZE, 12).unwrap(), || Ok(()))
        .unwrap();
    let mut calls = 0;
    let mut bytes = [0; 19];
    memory
        .read_resolved(&range, &mut bytes, &mut |inputs, operation| {
            calls += 1;
            assert!(matches!(operation, AccessOperation::Read));
            assert_eq!(inputs.iter().map(AllocationRange::bytes).sum::<u64>(), 19);
            // This would deadlock if the direct read held the mapping lock while
            // requesting coherence. The operation must retain the original RAM.
            space.unmap(mapping.offset()).unwrap();
            assert_eq!(machine.statistics().retained_bytes, 2 * PAGE_SIZE);
            machine.write_resolved(&inputs, &[0x33; 19]).unwrap();
            Ok(AccessResult::Read(Box::new([0x33; 19])))
        })
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(bytes, [0x33; 19]);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}
