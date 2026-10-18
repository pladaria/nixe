use super::*;
use crate::*;
use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress};
use nixe_memory::{CanonicalAllocation, MappingGeneration, MemoryPermissions};

#[test]
fn inline_mutations_are_consumed_between_native_effects() {
    let mut space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mapping = space
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
    let address = mapping.offset().get();
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    let frontend = FrontendSubmissionId::new(2);
    let mut words = Vec::new();
    let mut packet = |subchannel: u32, method: u32, arguments: &[u32]| {
        words
            .push((1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | (method / 4));
        words.extend_from_slice(arguments);
    };
    packet(1, 0, &[SWITCH_1_GM20B_PROFILE.classes().compute().0]);
    for value in [0x1122_3344, 0x5566_7788] {
        packet(1, 0x0188, &[(address >> 32) as u32, address as u32]);
        packet(1, 0x0180, &[4, 1]);
        packet(1, 0x01b0, &[0x41]);
        packet(1, 0x01b4, &[value]);
        packet(6, 0x002c, &[0x8000_0000]);
    }
    let channel_id = channel.id();
    let mut packets = DecodedPackets::new(words.into_iter().enumerate().map(|(index, value)| {
        Ok(MaxwellPushbufferWord::new(
            value,
            MaxwellGpfifoSourceLocation {
                channel: channel_id,
                frontend,
                entry_index: 0,
                pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
                word_offset: index as u64,
                mapping: MaxwellMappingId::new(1),
                generation: MappingGeneration::new(1),
            },
        ))
    }));
    let mut cache = MaxwellNativeFrontend::new(Default::default());
    let planner = MaxwellSubmissionPlanner::new(&space, frontend, None, &mut cache);
    let mut execution = MaxwellBackendExecution::begin(frontend, None);
    let mut observed = Vec::new();
    let tail = lower_packets_streamed(&mut packets, &mut channel, planner, &mut |steps| {
        execution.append_steps(steps);
        while let Some(segment) = execution.next_segment(0).unwrap() {
            for command in &segment.commands {
                assert!(matches!(
                    command,
                    nixe_gpu::native::NativeCommand::Ordering(_)
                ));
                let mut bytes = [0; 4];
                allocation.read(0, &mut bytes).unwrap();
                observed.push(u32::from_le_bytes(bytes));
            }
            execution.recycle_segment(segment);
            execution.accept_segment();
            execution.retire_consumed_steps();
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(observed, [0x1122_3344, 0x5566_7788]);
    execution.append_steps(tail.into_steps().into_vec());
    execution.seal();
    let final_segment = execution.next_segment(0).unwrap().unwrap();
    assert!(final_segment.final_segment);
    assert!(final_segment.commands.is_empty());
    execution.accept_segment();
    assert!(execution.next_segment(0).unwrap().is_none());
    assert!(execution.is_finished());
}
