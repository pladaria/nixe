use super::super::source::prepare_maxwell_shader_translation_source;
use super::super::test_support::{
    canonical_shader_writes, mapped_memory, program_three_d, translate_maxwell_shader_programs,
    validate_wgsl,
};
use crate::{
    MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuChannel, MaxwellLoweringCache,
    SWITCH_1_GM20B_PROFILE,
};
use nixe_gpu::{ShaderIoLocation, ShaderOperation, ShaderResourceKind, lower_shader_ir_to_wgsl};

#[test]
fn enabled_vertex_and_fragment_interfaces_are_linked_before_backend_lowering() {
    let (allocation, address_space, address) = mapped_memory();
    let mut vertex_header = [0_u32; 20];
    vertex_header[0] = 0x0002_0461;
    vertex_header[6] = 0xf | (0xf << 8); // Position from attribute 0, passthrough from 2.
    vertex_header[13] = (0xf << 12) | (1 << 16) | (0xf << 24);
    let mut fragment_header = [0_u32; 20];
    fragment_header[0] = 0x0002_5462;
    fragment_header[6] = 2;
    fragment_header[18] = 1;
    let program_bytes = |header: [u32; 20]| {
        let generic = (0..32)
            .find(|index| (header[6 + index / 4] >> ((index % 4) * 8)) & 255 != 0)
            .unwrap();
        // Consume the selected header input, so linkage tests observe real reads.
        let code = [
            0_u64,
            0xe0c3_ff88_0077_ff00 | ((generic as u64 * 16) << 28),
            0xe300_0000_0007_000f,
            0,
        ];
        header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(code.into_iter().flat_map(u64::to_le_bytes))
            .collect::<Vec<_>>()
    };
    let vertex_code = [
        0_u64,
        0xeff1_ff80_0707_ff00, // AST Position, preloaded attribute 0
        0xefd9_ff80_0a07_ff00, // ALD R0-R3, Generic(2)
        0xeff1_ff80_0a07_ff00, // AST Generic(2), R0-R3
        0,
        0x0103_f800_0007_f000, // MOV R0, 1.0 for Generic(0)
        0xe300_0000_0007_000f,
        0,
    ];
    let vertex_bytes = vertex_header
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(vertex_code.into_iter().flat_map(u64::to_le_bytes))
        .collect::<Vec<_>>();
    allocation.write(0, &vertex_bytes).unwrap();
    allocation
        .write(0x100, &program_bytes(fragment_header))
        .unwrap();

    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    program_three_d(&mut channel, 0x2140, 0x50);
    for (method, argument) in [
        (0x1608, (address >> 32) as u32),
        (0x160c, address as u32),
        (0x2000, 0x11),
        (0x2004, 0),
        (0x200c, 4),
        (0x2040, 0x51),
        (0x2044, 0x100),
        (0x204c, 4),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let linked = translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert_eq!(linked.len(), 2);
    let vertex = linked[0].module.ir();
    assert!(
        vertex
            .ir()
            .inputs()
            .iter()
            .all(|i| i.location() == ShaderIoLocation::Generic(0)),
        "dead passthrough must not require vertex attribute 2"
    );
    assert!(!vertex.ir().instructions().iter().any(|i| matches!(
        i.operation(),
        ShaderOperation::LoadInput {
            location: ShaderIoLocation::Generic(2),
            ..
        }
    )));
    let wgsl = lower_shader_ir_to_wgsl(vertex).unwrap();
    validate_wgsl(&wgsl);
    assert!(!wgsl.source().contains("generic_2"));

    fragment_header[6] = 2 << 8;
    allocation
        .write(0x100, &program_bytes(fragment_header))
        .unwrap();
    let unwritten =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert!(unwritten[1].module.ir().ir().inputs().is_empty());
    // The same vertex program must keep the attribute when a new fragment
    // program consumes it. Linked inputs belong to shader cache identity.
    fragment_header[6] = 2 << 16;
    allocation
        .write(0x100, &program_bytes(fragment_header))
        .unwrap();
    let consumed =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert_ne!(linked[0].fingerprint, consumed[0].fingerprint);
    assert!(
        consumed[0]
            .module
            .ir()
            .ir()
            .inputs()
            .iter()
            .any(|i| i.location() == ShaderIoLocation::Generic(2))
    );
}

#[test]
fn unwritten_fragment_varyings_use_maxwell_defaults_and_specialize_cache_identity() {
    let (allocation, address_space, address) = mapped_memory();
    let mut vertex_header = [0_u32; 20];
    vertex_header[0] = 0x0002_0461;
    let mut fragment_header = [0_u32; 20];
    fragment_header[0] = 0x0002_5462;
    fragment_header[6] = (1 << 8) | (1 << 14); // Generic(1).x and .w, constant.
    fragment_header[18] = 3; // R0-R1 -> red/green.
    let vertex_code = [0, 0xe300_0000_0007_000f_u64, 0, 0];
    let fragment_code = [
        0,
        0xe0c3_ff89_0077_ff00_u64, // IPA.SC R0, Generic(1).x.
        0xe0c3_ff89_c077_ff01_u64, // IPA.SC R1, Generic(1).w.
        0xe300_0000_0007_000f,
    ];
    let write_program = |offset, header: [u32; 20], code: &[u64]| {
        allocation
            .write(
                offset,
                &header
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .chain(code.iter().copied().flat_map(u64::to_le_bytes))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    write_program(0, vertex_header, &vertex_code);
    write_program(0x100, fragment_header, &fragment_code);
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    program_three_d(&mut channel, 0x2140, 0x50);
    for (method, argument) in [
        (0x1608, (address >> 32) as u32),
        (0x160c, address as u32),
        (0x2000, 0x11),
        (0x2004, 0),
        (0x200c, 4),
        (0x2040, 0x51),
        (0x2044, 0x100),
        (0x204c, 4),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let defaults =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    let fragment = defaults[1].module.ir();
    assert!(fragment.ir().inputs().is_empty());
    for (register, expected) in [(0, 0), (1, 1.0_f32.to_bits())] {
        assert!(
            fragment
                .ir()
                .instructions()
                .iter()
                .any(|instruction| matches!(
                    instruction.operation(),
                    ShaderOperation::MoveImmediate32 { destination, bits, .. }
                        if destination.index() == register && *bits == expected
                ))
        );
    }
    validate_wgsl(&lower_shader_ir_to_wgsl(fragment).unwrap());
    // The fragment binary is unchanged. A different producer exporting this
    // varying must use interpolation instead of reusing the constant variant.
    for (exported_bit, expected_red, expected_green) in
        [(20, 0.25_f32, 1.0_f32), (23, 0.0_f32, 0.75_f32)]
    {
        vertex_header[13] = 1 << exported_bit;
        write_program(0, vertex_header, &vertex_code);
        let partial =
            translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
        assert_eq!(partial[1].module.ir().ir().inputs().len(), 1);
        assert_ne!(defaults[1].fingerprint(), partial[1].fingerprint());
        let inputs = nixe_gpu::ShaderEvaluationInputs::default()
            .with_interface_bits(ShaderIoLocation::Generic(1), 0, 0.25_f32.to_bits())
            .with_interface_bits(ShaderIoLocation::Generic(1), 3, 0.75_f32.to_bits());
        let result = nixe_gpu::evaluate_shader_ir(partial[1].module.ir(), &inputs, 64).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Color(0), 0),
            Some(expected_red.to_bits())
        );
        assert_eq!(
            result.output_bits(ShaderIoLocation::Color(0), 1),
            Some(expected_green.to_bits())
        );
        validate_wgsl(&lower_shader_ir_to_wgsl(partial[1].module.ir()).unwrap());
    }
    vertex_header[13] = 1 << 20;
    vertex_header[13] |= 1 << 23;
    write_program(0, vertex_header, &vertex_code);
    let exported =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert_ne!(defaults[1].fingerprint(), exported[1].fingerprint());
    assert_eq!(exported[1].module.ir().ir().inputs().len(), 2);
    validate_wgsl(&lower_shader_ir_to_wgsl(exported[1].module.ir()).unwrap());
    vertex_header[13] = 0;
    write_program(0, vertex_header, &vertex_code);
    let revisited =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert_eq!(defaults[1].fingerprint(), revisited[1].fingerprint());
}

#[test]
fn shader_cache_reuses_exact_inputs_and_retains_alternating_programs() {
    let (allocation, address_space, address) = mapped_memory();
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    let code = [0_u64, 0xe300_0000_0007_000f, 0, 0];
    let bytes = header
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(code.into_iter().flat_map(u64::to_le_bytes))
        .collect::<Vec<_>>();
    allocation.write(0, &bytes).unwrap();

    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    program_three_d(&mut channel, 0x2140, 0x50);
    program_three_d(&mut channel, 0x2040, 0x10);
    program_three_d(&mut channel, 0x1608, (address >> 32) as u32);
    program_three_d(&mut channel, 0x160c, address as u32);
    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x2004, 0);
    program_three_d(&mut channel, 0x200c, 4);

    let mut cache = MaxwellLoweringCache::default();
    let first_source = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let owned_first_source = first_source.materialize();
    assert!(first_source.matches(&owned_first_source));
    assert_eq!(
        first_source.fingerprint(),
        nixe_gpu::cache_fingerprint(&owned_first_source)
    );
    let first = cache
        .resolve_shader_translation_source(
            &first_source,
            &address_space,
            &nixe_memory::CanonicalWriteBatch::new(),
        )
        .unwrap();
    let repeated = cache
        .resolve_shader_translation_source(
            &first_source,
            &address_space,
            &nixe_memory::CanonicalWriteBatch::new(),
        )
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&repeated, &first));
    assert_eq!(cache.shader_translation_set_count(), 1);
    let state_programs = cache
        .resolve_shader_translation_for_state(
            channel.three_d(),
            &nixe_memory::CanonicalWriteBatch::new(),
            &address_space,
        )
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&state_programs, &first));
    let translated = std::sync::Arc::new(cache.stage_shader_translations(&state_programs).unwrap());
    let first_id = translated.shaders()[0].shader();
    cache.retain_translated_shader_state(&state_programs, std::sync::Arc::clone(&translated));
    let repeated_translated = cache
        .reuse_translated_shaders_for_state(
            channel.three_d(),
            &nixe_memory::CanonicalWriteBatch::new(),
            &address_space,
        )
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&repeated_translated, &translated));
    let repeated_id = cache
        .stage_shader_translations(&repeated)
        .unwrap()
        .shaders()[0]
        .shader();
    assert_eq!(repeated_id, first_id);
    assert_eq!(cache.shader_translation_count(), 1);

    program_three_d(&mut channel, 0x200c, 5);
    let changed_register_source =
        prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let changed_registers = cache
        .resolve_shader_translation_source(
            &changed_register_source,
            &address_space,
            &nixe_memory::CanonicalWriteBatch::new(),
        )
        .unwrap();
    let changed_register_id = cache
        .stage_shader_translations(&changed_registers)
        .unwrap()
        .shaders()[0]
        .shader();
    assert!(!std::sync::Arc::ptr_eq(&changed_registers, &first));
    assert_ne!(changed_register_id, first_id);
    assert_eq!(cache.shader_translation_set_count(), 2);
    assert_eq!(cache.shader_translation_count(), 2);
    program_three_d(&mut channel, 0x200c, 4);
    let restored_source = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let restored = cache
        .resolve_shader_translation_source(
            &restored_source,
            &address_space,
            &nixe_memory::CanonicalWriteBatch::new(),
        )
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&restored, &first));
    assert_eq!(
        cache
            .stage_shader_translations(&restored)
            .unwrap()
            .shaders()[0]
            .shader(),
        first_id
    );

    allocation.write(0, &bytes[..4]).unwrap();
    let after_cpu_write_source =
        prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let after_cpu_write = cache
        .resolve_shader_translation_source(
            &after_cpu_write_source,
            &address_space,
            &nixe_memory::CanonicalWriteBatch::new(),
        )
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&after_cpu_write, &first));
    assert_eq!(cache.shader_translation_set_count(), 2);
    let after_cpu_write_id = cache
        .stage_shader_translations(&after_cpu_write)
        .unwrap()
        .shaders()[0]
        .shader();
    assert_eq!(after_cpu_write_id, first_id);
    assert_eq!(cache.shader_translation_count(), 2);

    let staged = canonical_shader_writes(&address_space, &[(address, header[0])]);
    let after_staged_write_source =
        prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let after_staged_write = cache
        .resolve_shader_translation_source(&after_staged_write_source, &address_space, &staged)
        .unwrap();
    let after_staged_write_id = cache
        .stage_shader_translations(&after_staged_write)
        .unwrap()
        .shaders()[0]
        .shader();
    assert_eq!(after_staged_write_id, after_cpu_write_id);
    assert_eq!(cache.shader_translation_count(), 2);

    let ordered_forward = [(address + 4, 1), (address + 4, 2)];
    let ordered_reverse = [(address + 4, 2), (address + 4, 1)];
    let forward_source = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let forward = cache
        .resolve_shader_translation_source(
            &forward_source,
            &address_space,
            &canonical_shader_writes(&address_space, &ordered_forward),
        )
        .unwrap();
    let reverse_source = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let reverse = cache
        .resolve_shader_translation_source(
            &reverse_source,
            &address_space,
            &canonical_shader_writes(&address_space, &ordered_reverse),
        )
        .unwrap();
    let forward_id = cache.stage_shader_translations(&forward).unwrap().shaders()[0].shader();
    let reverse_id = cache.stage_shader_translations(&reverse).unwrap().shaders()[0].shader();
    assert_ne!(forward_id, reverse_id);
    assert_eq!(cache.shader_translation_count(), 4);
    // Shader writes through another GPU VA must invalidate this cache even
    // though the semantic key still names the original program address.
    let mut address_space = address_space;
    let alias = address_space
        .map(crate::MaxwellMapRequest {
            allocation: crate::MaxwellAllocationId::new(1),
            backing: allocation
                .backing_range(nixe_memory::MemoryPermissions::READ_WRITE)
                .unwrap(),
            backing_offset: 0,
            size: 0x1000,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: nixe_memory::MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    let alias_writes = canonical_shader_writes(&address_space, &[(alias.offset().get() + 4, 3)]);
    let alias_programs = cache
        .resolve_shader_translation_for_state(channel.three_d(), &alias_writes, &address_space)
        .unwrap();
    assert_ne!(
        cache
            .stage_shader_translations(&alias_programs)
            .unwrap()
            .shaders()[0]
            .shader(),
        reverse_id
    );
    let mut canonical = [0; 4];
    allocation.read(4, &mut canonical).unwrap();
    assert_eq!(canonical, 0_u32.to_le_bytes());
}

#[test]
fn shader_resources_use_reset_binding_group_zero_when_guest_omits_write() {
    let (allocation, address_space, address) = mapped_memory();
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    let mov = 0x0100_0000_0007_f000_u64 | (u64::from(1.0_f32.to_bits()) << 20);
    let fadd_cbuf = 0x4c58_0000_0007_0001_u64;
    let exit = 0xe300_0000_0007_000f_u64;
    let bytes = header
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(
            [0, mov, fadd_cbuf, exit]
                .into_iter()
                .flat_map(u64::to_le_bytes),
        )
        .collect::<Vec<_>>();
    allocation.write(0, &bytes).unwrap();

    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    program_three_d(&mut channel, 0x2140, 0x50);
    program_three_d(&mut channel, 0x2040, 0x10);
    program_three_d(&mut channel, 0x1608, (address >> 32) as u32);
    program_three_d(&mut channel, 0x160c, address as u32);
    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x2004, 0);
    program_three_d(&mut channel, 0x200c, 4);

    let translated =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert_eq!(translated.len(), 1);
    assert_eq!(translated[0].bind_group(), Some(0));
    assert!(translated[0].resources().iter().any(|resource| {
        resource.binding() == 0 && resource.kind() == ShaderResourceKind::ConstantBuffer
    }));
}

#[test]
fn reset_stage_groups_receive_distinct_neutral_resource_bindings() {
    let (allocation, address_space, address) = mapped_memory();
    let mut vertex_header = [0_u32; 20];
    vertex_header[0] = 0x0002_0461;
    vertex_header[13] = 1; // Position.x makes the vertex buffer load observable.
    let mut fragment_header = [0_u32; 20];
    fragment_header[0] = 0x0002_5462;
    fragment_header[18] = 1; // R0 -> red.
    let mov = 0x0100_0000_0007_f000_u64 | (u64::from(1.0_f32.to_bits()) << 20);
    let fadd_cbuf = 0x4c58_0000_0007_0000_u64;
    let exit = 0xe300_0000_0007_000f_u64;
    let program_bytes = |header: [u32; 20]| {
        header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(
                [
                    0,
                    mov,
                    fadd_cbuf,
                    if header[13] == 1 {
                        0xeff0_7f80_0707_ff00
                    } else {
                        exit
                    },
                    0,
                    exit,
                    0,
                    0,
                ]
                .into_iter()
                .flat_map(u64::to_le_bytes),
            )
            .collect::<Vec<_>>()
    };
    allocation.write(0, &program_bytes(vertex_header)).unwrap();
    allocation
        .write(0x100, &program_bytes(fragment_header))
        .unwrap();

    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    for (method, argument) in [
        (0x1608, (address >> 32) as u32),
        (0x160c, address as u32),
        (0x2040, 0x11),
        (0x2044, 0),
        (0x204c, 4),
        (0x2140, 0x51),
        (0x2144, 0x100),
        (0x214c, 4),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let translated =
        translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert_eq!(translated.len(), 2);
    assert_eq!(translated[0].bind_group(), Some(0));
    assert_eq!(translated[1].bind_group(), Some(4));
    assert_eq!(translated[0].resources()[0].binding(), 0);
    assert_eq!(translated[1].resources()[0].binding(), 1);
    // Required byte extents follow the linked bindings, including a stage
    // whose local c0 was remapped to neutral binding 1.
    assert_eq!(translated[0].constant_buffer_read_end(0), 4);
    assert_eq!(translated[1].constant_buffer_read_end(1), 4);
    assert!(
        lower_shader_ir_to_wgsl(translated[0].module().ir())
            .unwrap()
            .source()
            .contains("@binding(0)")
    );
    assert!(
        lower_shader_ir_to_wgsl(translated[1].module().ir())
            .unwrap()
            .source()
            .contains("@binding(1)")
    );

    let lowered = MaxwellLoweringCache::default()
        .stage_shader_translations(&translated)
        .unwrap();
    assert_eq!(
        lowered.resources()[0].role(),
        crate::MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 }
    );
    assert_eq!(
        lowered.resources()[1].role(),
        crate::MaxwellThreeDResourceRole::ConstantBuffer { group: 4, slot: 0 }
    );

    // Once the result no longer reaches rasterization, neither its load nor
    // the declaration may demand a guest buffer or produce a zero-sized import.
    vertex_header[13] = 0;
    allocation.write(0, &program_bytes(vertex_header)).unwrap();
    let pruned = translate_maxwell_shader_programs(channel.three_d(), &address_space, &[]).unwrap();
    assert!(pruned[0].resources().is_empty());
    let lowered = MaxwellLoweringCache::default()
        .stage_shader_translations(&pruned)
        .unwrap();
    assert_eq!(lowered.resources().len(), 1);
    assert_eq!(
        lowered.resources()[0].role(),
        crate::MaxwellThreeDResourceRole::ConstantBuffer { group: 4, slot: 0 }
    );
}

#[test]
fn graphics_ldg_uses_distinct_stage_bindings_and_coexists_with_textures() {
    let (allocation, space, address) = mapped_memory();
    let ldc = 0xef95_0000_0007_ff00 | (0x110 << 20); // R0:R1 = c0[0x110].
    let ldg = 0xeed4_a000_0007_0000; // Aliased B32 load R0,[R0:R1].
    let exit = 0xe300_0000_0007_000f;
    let mut vh = [0_u32; 20];
    vh[0] = 0x0402_0461;
    vh[13] = 1;
    let mut ph = [0_u32; 20];
    ph[0] = 0x0402_5462;
    ph[18] = 1;
    for (offset, header, code) in [
        (0, vh, [0, ldc, ldg, 0xeff0_7f80_0707_ff00, 0, exit, 0, 0]),
        (
            0x100,
            ph,
            [0, ldc, ldg, 0xda50_1a40_2077_0604, 0, exit, 0, 0],
        ),
    ] {
        allocation
            .write(
                offset,
                &header
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .chain(code.into_iter().flat_map(u64::to_le_bytes))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    for (method, argument) in [
        (0x1608, (address >> 32) as u32),
        (0x160c, address as u32),
        (0x2040, 0x11),
        (0x2044, 0),
        (0x204c, 8),
        (0x2140, 0x51),
        (0x2144, 0x100),
        (0x214c, 8),
        (0x2608, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let translated = translate_maxwell_shader_programs(channel.three_d(), &space, &[]).unwrap();
    let global_bindings: Vec<_> = translated
        .iter()
        .map(|p| {
            p.resources()
                .iter()
                .find(|r| r.kind() == ShaderResourceKind::StorageBuffer)
                .unwrap()
                .binding()
        })
        .collect();
    assert_ne!(global_bindings[0], global_bindings[1]);
    assert!(
        translated[0]
            .resources()
            .iter()
            .all(|r| r.kind() != ShaderResourceKind::ConstantBuffer),
        "dead pointer arithmetic must not bind a CB"
    );
    for program in &translated {
        validate_wgsl(&lower_shader_ir_to_wgsl(program.module().ir()).unwrap());
    }
    let pixel = &translated[1];
    let image_binding = pixel
        .resources()
        .iter()
        .find(|r| r.kind() == ShaderResourceKind::SampledImage)
        .unwrap()
        .binding();
    assert_ne!(global_bindings[1], image_binding);
    let staged = MaxwellLoweringCache::default()
        .stage_shader_translations(&translated)
        .unwrap();
    let globals: Vec<_> = staged
        .resources()
        .iter()
        .map(|r| r.role())
        .filter(|r| matches!(r, crate::MaxwellThreeDResourceRole::StorageBuffer { .. }))
        .collect();
    assert_eq!(
        globals,
        [
            crate::MaxwellThreeDResourceRole::StorageBuffer {
                group: 0,
                slot: 0,
                byte_offset: 0x110
            },
            crate::MaxwellThreeDResourceRole::StorageBuffer {
                group: 4,
                slot: 0,
                byte_offset: 0x110
            },
        ]
    );
}
