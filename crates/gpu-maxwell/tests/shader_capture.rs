use nixe_gpu::shader_compilation::Configuration;
use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    recompiler::{CaptureBindings, CaptureRequest, MemoryCache, capture_program},
};
use nixe_memory::{AllocationRange, Machine, MemoryPermissions, PAGE_SIZE};
use nixe_shader_recompiler::{Compiler, Stage, environment::EnvironmentMetadata};
use std::sync::Arc;

#[test]
fn owned_shader_capture_rechecks_code_pages_and_reuses_equal_instructions() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let exit = 0xe300_0000_0007_000fu64;
    machine
        .write_resolved(&[input.slice(8, 8).unwrap()], &exit.to_le_bytes())
        .unwrap();
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
            backing: vec![input.clone()],
            backing_offset: 0,
            size: PAGE_SIZE,
            allocation_alignment: PAGE_SIZE as u32,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    let request = || {
        [(
            None,
            CaptureRequest {
                address: mapping.offset().get(),
                metadata: EnvironmentMetadata {
                    stage: Stage::Compute,
                    workgroup_size: [1; 3],
                    ..Default::default()
                },
                bindings: CaptureBindings::default(),
            },
        )]
    };
    let mut compiler = Compiler::new().unwrap();
    let mut memory = MemoryCache::default();
    let mut access = |ranges: Vec<AllocationRange>, operation| {
        machine
            .apply_resolved_access(&ranges, &operation)
            .map_err(Arc::new)
    };
    let captured = capture_program(
        &mut compiler,
        address_space.memory(),
        request(),
        &Configuration::default(),
        &mut memory,
        &mut access,
    )
    .unwrap();
    assert!(captured.stages[0].request_is_current(&request()[0].1, address_space.memory()));
    assert_eq!(
        captured.stages[0]
            .environment
            .observations()
            .instructions
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [exit]
    );
    machine
        .write_resolved(&[input.slice(0x800, 1).unwrap()], &[1])
        .unwrap();
    assert!(!captured.stages[0].request_is_current(&request()[0].1, address_space.memory()));
    let refreshed = captured
        .refresh(&request(), address_space.memory(), &mut memory, &mut access)
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&captured.inputs, &refreshed.inputs));
    assert!(refreshed.stages[0].request_is_current(&request()[0].1, address_space.memory()));
    machine
        .write_resolved(&[input.slice(8, 8).unwrap()], &exit.to_le_bytes())
        .unwrap();
    assert!(!refreshed.stages[0].request_is_current(&request()[0].1, address_space.memory()));
    let refreshed = refreshed
        .refresh(&request(), address_space.memory(), &mut memory, &mut access)
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&captured.inputs, &refreshed.inputs));
    assert!(refreshed.stages[0].request_is_current(&request()[0].1, address_space.memory()));
    machine
        .write_resolved(&[input.slice(8, 8).unwrap()], &0u64.to_le_bytes())
        .unwrap();
    assert!(
        refreshed
            .refresh(&request(), address_space.memory(), &mut memory, &mut access)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        captured.stages[0]
            .environment
            .observations()
            .instructions
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [exit]
    );
}
