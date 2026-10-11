use super::super::binary::{MaxwellShaderInstructionBundle, MaxwellShaderMetadata};
use super::super::test_support::{mapped_memory, program_three_d};
use super::*;
use crate::{MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuChannel, SWITCH_1_GM20B_PROFILE};

fn channel(address: u64) -> MaxwellGpuChannel {
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    for (method, value) in [
        (0, SWITCH_1_GM20B_PROFILE.classes().three_d().0),
        (0x2380, 0x100),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2490, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    channel
}
fn binary() -> MaxwellShaderBinary {
    MaxwellShaderBinary {
        address: 0,
        metadata: MaxwellShaderMetadata::Compute {
            workgroup_size: [1, 1, 1],
            register_count: 8,
        },
        bundles: vec![
            MaxwellShaderInstructionBundle {
                offset: 0,
                control: 0,
                instructions: [
                    0x3820_0000_0007_0000 | (7 << 39) | (2 << 20), // unsigned min r0, r0, 2
                    0x3848_0000_0027_0003,                         // shl r3, r0, 2
                    0xef95_0000_0007_ff04,                         // unrelated B64 load into r4/r5
                ],
            },
            MaxwellShaderInstructionBundle {
                offset: 32,
                control: 0,
                instructions: [
                    0xef94_0000_0007_0303, // table r3 = c0[r3]
                    0xe250_0000_0007_030f, // relative BRX r3, base PC 56
                    0xe300_0000_0007_000f,
                ],
            },
            MaxwellShaderInstructionBundle {
                offset: 64,
                control: 0,
                instructions: [0xe300_0000_0007_000f; 3],
            },
        ]
        .into(),
        indirect_tables: Box::new([]),
        source_cpu_writes: Box::new([]),
        source_mappings: Box::new([]),
    }
}
#[test]
fn bounded_table_tracks_only_its_bytes_and_binding_identity() {
    let (allocation, space, address) = mapped_memory();
    allocation
        .write(0, &[0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0])
        .unwrap();
    let channel = channel(address);
    let writes = CanonicalWriteBatch::new();
    let mut binary = binary();
    capture(
        &mut binary,
        channel.three_d().shader_bindings(),
        Some(4),
        &MaxwellShaderMemoryView::new(&space, &writes),
    )
    .unwrap();
    let table = &binary.indirect_tables[0];
    assert_eq!(table.load_offset, 40);
    assert_eq!(
        table.read_end, 12,
        "include repeated entries, not only distinct BRX targets"
    );
    assert_eq!(
        table.targets.as_ref(),
        [
            (0, ShaderSourceLocation::new(56)),
            (16, ShaderSourceLocation::new(72))
        ]
    );
    assert!(table.current(&space));
    assert!(table.binding_current(channel.three_d().shader_bindings()));
    allocation.write(0x80, &[1]).unwrap();
    assert!(table.current(&space));
    let mut overlay = CanonicalWriteBatch::new();
    overlay
        .stage(&table.range, 0, &16_u32.to_le_bytes())
        .unwrap();
    assert!(!table.staged_current(&overlay));
    assert!(!table.binding_current(self::channel(address + 0x100).three_d().shader_bindings()));
    allocation.write(0, &16_u32.to_le_bytes()).unwrap();
    assert!(!table.current(&space));
}
#[test]
fn control_edge_cannot_bypass_the_table_index_clamp() {
    let (_allocation, space, address) = mapped_memory();
    let channel = channel(address);
    let mut binary = binary();
    // A direct edge targets the stride at PC16, bypassing the clamp at PC8.
    let displacement = (16_i32 - 80) as u32 & 0xffffff;
    binary.bundles[2].instructions[0] = 0xe240_0000_0007_000f | (u64::from(displacement) << 20);
    let writes = CanonicalWriteBatch::new();
    let error = capture(
        &mut binary,
        channel.three_d().shader_bindings(),
        Some(4),
        &MaxwellShaderMemoryView::new(&space, &writes),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not dominate"), "{error}");
}
