use nixe_gpu::{FrontendSubmissionId, GuestSyncpointId, native::NativeCommand};
use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellBackendExecution, MaxwellChannelId, MaxwellChannelOwner, MaxwellCommandSourceScratch,
    MaxwellGpfifoSubmitRequest, MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellMapRequest,
    MaxwellMemoryManagerId, MaxwellNativeFrontend, MaxwellScheduler, SWITCH_1_GM20B_PROFILE,
    decode_gpfifo_submission, resolve_gpfifo_submission, stream_maxwell_frontend,
};
use nixe_memory::{AllocationRange, Machine, MemoryPermissions, PAGE_SIZE};
use std::sync::Arc;

#[test]
fn inline_uploads_are_consumed_between_native_inputs_without_native_delivery() {
    let frontend = FrontendSubmissionId::new(1);
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
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
    let address = mapping.offset().get();
    let mut words = Vec::new();
    let mut packet = |subchannel: u32, method: u32, arguments: &[u32]| {
        words
            .push((1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | (method / 4));
        words.extend_from_slice(arguments);
    };
    packet(1, 0, &[SWITCH_1_GM20B_PROFILE.classes().compute().0]);
    packet(6, 0x002c, &[0x8000_0000]);
    for value in [2, 3] {
        packet(1, 0x0188, &[(address >> 32) as u32, address as u32]);
        packet(1, 0x0180, &[4, 1]);
        packet(1, 0x01b0, &[0x41]);
        packet(1, 0x01b4, &[value]);
        packet(6, 0x002c, &[0x8000_0000]);
    }
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    machine
        .write_resolved(&[input.slice(0x400, bytes.len() as u64).unwrap()], &bytes)
        .unwrap();
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    channel.bind_address_space(space.memory().id()).unwrap();
    channel
        .bind_memory_manager(MaxwellMemoryManagerId::new(1))
        .unwrap();
    channel
        .allocate_gpfifo(8, false, GuestSyncpointId::new(1))
        .unwrap();
    let pointer = address + 0x400;
    let descriptor = [
        pointer as u32,
        ((pointer >> 32) as u32) | ((words.len() as u32) << 10),
    ];
    let descriptor: Vec<_> = descriptor.iter().flat_map(|w| w.to_le_bytes()).collect();
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        8,
        MaxwellGpfifoSubmitRequest {
            entry_count: 1,
            flags: 1 << 2,
            fence_id: 0,
            fence_value: 0,
        },
        &descriptor,
    )
    .unwrap();
    let submission =
        resolve_gpfifo_submission(&channel.configuration(), frontend, decoded, space.memory())
            .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), submission, None, None)
        .unwrap();
    let dispatch = scheduler
        .dispatch_next(true, space.memory())
        .unwrap()
        .unwrap();
    let mut execution = MaxwellBackendExecution::begin(frontend, None);
    let mut observed = Vec::new();
    let mut access = |ranges: Vec<AllocationRange>, operation| {
        machine
            .apply_resolved_access(&ranges, &operation)
            .map_err(Arc::new)
    };
    let mut owner_access = |ranges: Vec<AllocationRange>, operation| {
        machine
            .apply_resolved_access(&ranges, &operation)
            .map_err(Arc::new)
    };
    let tail = stream_maxwell_frontend(
        &dispatch,
        &mut channel,
        space.memory(),
        &mut MaxwellNativeFrontend::new(Default::default()),
        &mut MaxwellCommandSourceScratch::default(),
        &mut access,
        &mut || panic!("this stream has no command-source wait"),
        &mut |steps| {
            execution.append_steps(steps);
            while let Some(segment) = execution
                .next_segment_with_memory(0, &mut owner_access)
                .unwrap()
            {
                for command in &segment.commands {
                    assert!(
                        matches!(command, NativeCommand::Ordering(_)),
                        "inline RAM writes must not become native commands"
                    );
                    // Stand in for the native consumer's capture, before acceptance.
                    let mut bytes = [0; 4];
                    machine
                        .capture_stream(&[input.slice(0, 4).unwrap()], &mut bytes)
                        .unwrap();
                    observed.push(u32::from_le_bytes(bytes));
                }
                execution.recycle_segment(segment);
                execution.accept_segment();
                execution.retire_consumed_steps();
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(observed, [0, 2, 3]);
    execution.append_steps(tail.into_steps().into_vec());
    execution.seal();
    let final_segment = execution
        .next_segment_with_memory(0, &mut access)
        .unwrap()
        .unwrap();
    assert!(final_segment.final_segment);
    assert!(final_segment.commands.is_empty());
    execution.recycle_segment(final_segment);
    execution.accept_segment();
    assert!(
        execution
            .next_segment_with_memory(0, &mut access)
            .unwrap()
            .is_none()
    );
    assert!(execution.is_finished());
}
