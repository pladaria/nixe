use nixe_gpu::{
    BackendResourceCreateInfo, FrontendSubmissionId, GpuVirtualAddress, MappingGeneration,
    ShaderStage,
};
use nixe_gpu_maxwell::*;
use nixe_memory::{CanonicalAllocation, MemoryPermissions};

fn location(word_offset: u32) -> MaxwellGpfifoSourceLocation {
    MaxwellGpfifoSourceLocation {
        channel: MaxwellChannelId::new(1),
        frontend: FrontendSubmissionId::new(2),
        entry_index: 0,
        pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
        word_offset: u64::from(word_offset),
        mapping: MaxwellMappingId::new(1),
        generation: MappingGeneration::new(1),
    }
}

fn packet(subchannel: u32, method_dword: u32, arguments: &[u32]) -> MaxwellDecodedPushbuffer {
    let mut words = Vec::with_capacity(arguments.len() + 1);
    words.push(Ok(MaxwellPushbufferWord::new(
        (1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
        location(0),
    )));
    words.extend(arguments.iter().enumerate().map(|(index, argument)| {
        Ok(MaxwellPushbufferWord::new(
            *argument,
            location(index as u32 + 1),
        ))
    }));
    decode_maxwell_pushbuffer(words).unwrap()
}

fn address_space() -> MaxwellGpuAddressSpace {
    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    address_space
}

#[test]
fn captured_shader_families_reach_neutral_draw_work_and_backend_modules() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let shader_allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut address_space = address_space();
    let mut map = |allocation: &CanonicalAllocation, id, size, kind| {
        address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(id),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap()
            .offset()
            .get()
    };
    let vertex = map(&vertex_allocation, 1, 0x4000, 0);
    let target = map(&target_allocation, 2, 0x10000, 0xfe);
    let shader = map(&shader_allocation, 3, 0x1000, 0);

    let write_program = |offset: usize, header: [u32; 20], code: &[u64]| {
        let bytes = header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(code.iter().flat_map(|word| word.to_le_bytes()))
            .collect::<Vec<_>>();
        shader_allocation.write(offset, &bytes).unwrap();
    };
    let mut vertex_header = [0_u32; 20];
    vertex_header[0] = 0x0002_0461;
    vertex_header[4] = 0x000f_f000;
    vertex_header[6] = 0x0000_0077;
    vertex_header[13] = 0x0007_f000;
    write_program(
        0,
        vertex_header,
        &[
            0x001f_b800_e420_0701,
            0xefd8_ff80_087f_ff00,
            0xefd8_7f80_0887_ff02,
            0x0103_f800_0007_f003,
            0x001c_b801_e020_18e2,
            0xeff1_ff80_0707_ff00,
            0xefd8_ff80_0907_ff00,
            0xefd8_7f80_0987_ff02,
            0x07ff_bc02_3c40_08e1,
            0xeff0_ff80_087f_ff00,
            0xeff0_7f80_0887_ff02,
            0xe300_0000_0007_000f,
        ],
    );
    let mut fragment_header = [0_u32; 20];
    fragment_header[0] = 0x0002_5462;
    fragment_header[5] = 0x8000_0000;
    fragment_header[6] = 0x0000_002a;
    fragment_header[18] = 0x0000_000f;
    write_program(
        0x200,
        fragment_header,
        &[
            0x001f_b001_e020_070f,
            0xe003_ff87_cff7_ff00,
            0x5080_0000_0047_0002,
            0x0103_f800_0007_f003,
            0x015c_8800_6840_0901,
            0xe043_ff88_0027_ff00,
            0xe043_ff88_4027_ff01,
            0xe043_ff88_8027_ff02,
            0x0000_0000_0001_ffef,
            0xe300_0000_0007_000f,
            0,
            0,
        ],
    );

    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    let mut lowering_cache = MaxwellThreeDLoweringCache::default();
    let mut dispatch = |method: u32, argument: u32| {
        let decoded = packet(0, method / 4, &[argument]);
        lower_maxwell_pushbuffer(
            &decoded,
            &mut channel,
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &mut lowering_cache,
        )
        .unwrap()
    };
    for (method, argument) in [
        (0, SWITCH_1_GM20B_PROFILE.classes().three_d().0),
        (0x0800, (target >> 32) as u32),
        (0x0804, target as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x0d6c, 64 << 16),
        (0x0d70, 32 << 16),
        (0x1c00, 0x1010),
        (0x1c04, (vertex >> 32) as u32),
        (0x1c08, vertex as u32),
        (0x1f00, (vertex >> 32) as u32),
        (0x1f04, (vertex + 0xff) as u32),
        (0x1160, 0x3820_0000),
        (0x0d74, 3),
        (0x0308, 3),
        (0x1618, 4),
        (0x1970, 4),
        (0x1608, (shader >> 32) as u32),
        (0x160c, shader as u32),
        (0x2000, 0x11),
        (0x2004, 0),
        (0x200c, 4),
        (0x2040, 0x51),
        (0x2044, 0x200),
        (0x204c, 4),
        (0x12e4, 0),
        (0x135c, 0),
        (0x121c, 1),
    ] {
        let _ = dispatch(method, argument);
    }
    let draw = dispatch(0x0d78, 3);
    let [MaxwellSubmissionExecutionStep::ThreeD(work)] = draw.steps() else {
        panic!("captured draw did not lower to one neutral 3D work item");
    };
    let shader_modules = work
        .resource_creations()
        .iter()
        .filter_map(|creation| match creation {
            BackendResourceCreateInfo::Shader {
                description,
                module,
                ..
            } => Some((description.stage, module)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(shader_modules.len(), 2);
    assert!(
        shader_modules
            .iter()
            .any(|(stage, _)| *stage == ShaderStage::Vertex)
    );
    assert!(
        shader_modules
            .iter()
            .any(|(stage, _)| *stage == ShaderStage::Fragment)
    );
    for (_, module) in shader_modules {
        let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&parsed)
        .unwrap();
    }

    shader_allocation
        .write(0, &vertex_header[0].to_le_bytes())
        .unwrap();
    let replacement = dispatch(0x0d78, 3);
    let [MaxwellSubmissionExecutionStep::ThreeD(replacement_work)] = replacement.steps() else {
        panic!("replacement draw did not lower to one neutral 3D work item");
    };
    assert!(replacement_work.resource_invalidations().is_empty());
    assert!(
        !replacement_work
            .resource_creations()
            .iter()
            .any(|creation| matches!(creation, BackendResourceCreateInfo::Shader { .. }))
    );
    assert!(
        !replacement_work
            .resource_creations()
            .iter()
            .any(|creation| matches!(creation, BackendResourceCreateInfo::Pipeline { .. }))
    );
}
