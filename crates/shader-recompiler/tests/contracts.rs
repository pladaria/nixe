use nixe_shader_recompiler::{
    artifact::*,
    diagnostics::FailureCategory,
    environment::*,
    ir::{Attribute, Type},
    maxwell::header::ProgramHeader,
    profile::*,
    types::*,
};

#[test]
fn header_decodes_little_endian_bytes_and_overlapping_stage_maps() {
    let mut bytes = [0; 80];
    bytes[0..4].copy_from_slice(&0xda97_e4a3u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&0x1200_1234u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&0x3400_0002u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&0xf700_0040u32.to_le_bytes());
    bytes[16..20].copy_from_slice(&0xab5a_b321u32.to_le_bytes());
    bytes[23] = 0xa5;
    bytes[24] = 0xe4;
    bytes[39] = 0x81;
    bytes[42] = 0x83;
    bytes[43] = 0xf7;
    bytes[53] = 0x5a;
    bytes[54] = 0x96;
    bytes[69] = 0xa3;
    bytes[72..76].copy_from_slice(&0x8000_0105u32.to_le_bytes());
    bytes[76..80].copy_from_slice(&0x4000_0003u32.to_le_bytes());
    let header = ProgramHeader::from_bytes(&bytes);
    assert_eq!(header.to_bytes(), bytes);
    assert_eq!(header.sph_type(), 3);
    assert_eq!(header.version(), 5);
    assert_eq!(header.shader_type(), 9);
    assert_eq!(header.local_memory_size(), 0x200_1234);
    assert_eq!(header.per_patch_attribute_count(), 0x12);
    assert_eq!(header.threads_per_input_primitive(), 0x34);
    assert_eq!(header.shader_local_memory_crs_size(), 0x40);
    assert_eq!(
        header.output_topology().unwrap(),
        OutputTopology::TriangleStrip
    );
    assert_eq!(header.max_output_vertices(), 0x321);
    assert_eq!(header.store_req_start(), 0xab);
    assert_eq!(header.store_req_end(), 0xab);
    assert_eq!(header.input_generic(0), [false, false, true, false]);
    assert_eq!(header.input_generic(1), [false, true, true, true]);
    assert_eq!(header.input_generic(31), [false, false, false, true]);
    assert_eq!(header.output_generic(0), [false, true, true, false]);
    assert_eq!(header.output_generic(31), [false, true, false, true]);
    assert_eq!(header.vtg_input_system_b().0, 0xa5);
    assert_eq!(header.vtg_output_system_b().0, 0x5a);
    assert_eq!(header.vtg_input_system_c().0, 0xf783);
    assert_eq!(header.vtg_output_system_c().0, 0x0105);
    assert_eq!(
        header.generic_input_map(0),
        [
            PixelImap::Unused,
            PixelImap::Constant,
            PixelImap::Perspective,
            PixelImap::ScreenLinear
        ]
    );
    assert_eq!(
        header.enabled_output_components(0),
        [true, false, true, false]
    );
    assert!(!header.has_output_components(1));
    assert!(header.has_output_components(2));
    assert_eq!(
        header.enabled_output_components(7),
        [false, false, false, true]
    );
    assert!(header.fragment_sample_mask());
    assert!(header.fragment_depth());
    assert_eq!(header.fragment_output_reserved(), 0x1000_0000);
    assert!(ProgramHeader::try_from(&bytes[..79]).is_err());
    assert!(ProgramHeader::default().output_topology().is_err());
}

#[test]
fn memory_and_entry_offsets_are_byte_addressed() {
    let mut bytes = [0; 80];
    bytes[4..8].copy_from_slice(&0x123u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&0x25u32.to_le_bytes());
    let header = ProgramHeader::from_bytes(&bytes);
    assert_eq!(
        EnvironmentMetadata::graphics_local_memory_size(&header).unwrap(),
        0x148
    );
    assert_eq!(
        EnvironmentMetadata::compute_local_memory_size(0x123, 0x25).unwrap(),
        0x148
    );
    let mut metadata = EnvironmentMetadata {
        start_address: ProgramOffset(0x100),
        ..Default::default()
    };
    assert_eq!(metadata.entry_address().unwrap(), ProgramOffset(0x150));
    metadata.stage = Stage::Compute;
    assert_eq!(metadata.entry_address().unwrap(), ProgramOffset(0x100));
    metadata.stage = Stage::Fragment;
    metadata.start_address = ProgramOffset(u32::MAX - 79);
    assert!(metadata.entry_address().is_err());
    bytes[8..12].copy_from_slice(&0x100u32.to_le_bytes());
    assert!(
        EnvironmentMetadata::graphics_local_memory_size(&ProgramHeader::from_bytes(&bytes))
            .is_err()
    );
}

#[test]
fn capability_policy_fills_only_zero_limits() {
    let mut info = HostTranslateInfo {
        min_ssbo_alignment: 256,
        max_descriptor_set_samplers: 17,
        ..Default::default()
    };
    info.apply_descriptor_limit_policy();
    assert_eq!(info.min_ssbo_alignment, 256);
    assert_eq!(info.max_descriptor_set_samplers, 17);
    assert_eq!(info.max_per_stage_descriptor_sampled_images, 1024);
    assert_eq!(info.max_per_stage_resources, 1024);
    assert_eq!(info.max_descriptor_set_uniform_buffers, 1024);
    assert_eq!(info.max_descriptor_set_uniform_buffers_dynamic, 1024);
    assert_eq!(info.max_descriptor_set_storage_buffers, 1024);
    assert_eq!(info.max_descriptor_set_storage_buffers_dynamic, 1024);
    assert_eq!(info.max_descriptor_set_sampled_images, 1024);
    assert_eq!(info.max_descriptor_set_storage_images, 1024);
    assert_eq!(info.max_descriptor_set_input_attachements, 1024);
    let once = info.clone();
    info.apply_descriptor_limit_policy();
    assert_eq!(info, once);
    let mut zero = HostTranslateInfo::default();
    zero.apply_descriptor_limit_policy();
    assert_eq!(zero.min_ssbo_alignment, 1);
    assert!(!zero.support_float16);
    let mut profile = Profile::default();
    assert_eq!(profile.supported_spirv, 0x0001_0000);
    for stage in 0..7 {
        assert!(profile.supports_subgroup_stage(Stage::try_from(stage).unwrap()));
    }
    profile.supported_subgroup_stages = 1 << Stage::Compute as u32;
    assert!(profile.supports_subgroup_stage(Stage::Compute));
    assert!(!profile.supports_subgroup_stage(Stage::VertexA));
    assert_eq!(
        RuntimeInfo::default().generic_input_types,
        [AttributeType::Float; 32]
    );
}

#[test]
fn reference_enum_discriminants_are_preserved() {
    // Compare against the normative declarations, including explicit gaps. This
    // catches omissions/reordering without making the implementation its oracle.
    fn check<T: TryFrom<u32> + std::fmt::Debug>(name: &str) {
        let spec = include_str!("data/contract-enums.txt");
        let declaration = spec.rsplit(&format!("enum class {name} ")).next().unwrap();
        let body = declaration
            .split_once('{')
            .unwrap()
            .1
            .split_once("};")
            .unwrap()
            .0;
        let body = body
            .lines()
            .map(|line| line.split("//").next().unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let mut next = 0;
        let mut discriminants = Vec::new();
        for item in body
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            let (variant, value) = if let Some((variant, value)) = item.split_once('=') {
                (variant.trim(), value.trim().parse::<u32>().unwrap())
            } else {
                (item, next)
            };
            let actual = T::try_from(value).ok().unwrap();
            assert_eq!(format!("{actual:?}"), variant, "{name} at {value}");
            discriminants.push(value);
            next = value + 1;
        }
        for value in 0..=next {
            assert_eq!(
                T::try_from(value).is_ok(),
                discriminants.contains(&value),
                "{name} at {value}"
            );
        }
    }
    check::<Stage>("Stage");
    check::<OutputTopology>("OutputTopology");
    check::<PixelImap>("PixelImap");
    check::<AttributeType>("AttributeType");
    check::<InputTopology>("InputTopology");
    check::<CompareFunction>("CompareFunction");
    check::<TessPrimitive>("TessPrimitive");
    check::<TessSpacing>("TessSpacing");
    check::<ReplaceConstant>("ReplaceConstant");
    check::<TextureType>("TextureType");
    check::<TexturePixelFormat>("TexturePixelFormat");
    check::<ImageFormat>("ImageFormat");
    check::<Interpolation>("Interpolation");
}

#[test]
fn varying_masks_cover_high_slots_and_legacy_attributes() {
    let mut state = VaryingState::default();
    assert!(state.is_uniform(Attribute::Generic31X));
    for component in 0..4 {
        state.set(Attribute::generic(31, component).unwrap(), true);
    }
    assert!(state.generic(31));
    assert!(state.all_components(Attribute::Generic31X));
    state.set(Attribute::Generic31Y, false);
    assert!(!state.is_uniform(Attribute::Generic31X));
    state.set(Attribute::ClipDistance7, true);
    assert!(state.clip_distances());
    assert!(!state.legacy());
    state.set(Attribute::FixedFncTexture9Q, true);
    assert!(state.fixed_function_texture());
    assert!(state.legacy());
    state.set(Attribute::DrawID, true);
    assert!(state.get(Attribute::DrawID));
    let last = Attribute::from_index(511).unwrap();
    state.set(last, true);
    assert_eq!(state.mask[7], 1 << 63);
    assert!(Attribute::from_index(512).is_none());
    assert_eq!(ShaderInfo::default().emulated_layer.index(), 0);
    assert_eq!(ShaderInfo::default().used_constant_buffer_types, Type::Void);
}

#[test]
fn descriptor_limits_count_arrays_without_limiting_texture_vector_growth() {
    let mut info = ShaderInfo::default();
    for index in 0..MAX_CBUFS {
        info.constant_buffer_descriptors
            .push(ConstantBufferDescriptor {
                index: CbufIndex(index as u32),
                count: 2,
            })
            .unwrap();
    }
    assert_eq!(num_descriptors(&info.constant_buffer_descriptors), 36);
    assert_eq!(
        info.constant_buffer_descriptors
            .push(ConstantBufferDescriptor::default())
            .unwrap_err()
            .category,
        FailureCategory::ResourceLimit
    );
    for _ in 0..MAX_SSBOS {
        info.storage_buffers_descriptors
            .push(StorageBufferDescriptor::default())
            .unwrap();
    }
    assert!(
        info.storage_buffers_descriptors
            .push(StorageBufferDescriptor::default())
            .is_err()
    );
    for _ in 0..100 {
        info.texture_descriptors.push(TextureDescriptor::default());
    }
    assert_eq!(info.texture_descriptors.len(), 100);
}

#[test]
fn snapshots_capture_only_consumed_facts_and_detach_from_source() {
    let key = CbufWordKey {
        index: CbufIndex(3),
        offset: ByteOffset(12),
    };
    let handle = TextureHandle(0x5678);
    let source_data = Observations {
        texture_handles: Default::default(),
        instructions: [(ProgramOffset(0x58), 0x1234)].into(),
        cbuf_words: [(key, 0xdead_beef)].into(),
        texture_types: [(handle, TextureType::ColorArray2D)].into(),
        texture_pixel_formats: [(handle, TexturePixelFormat::R32_UINT)].into(),
        texture_integer_formats: [(handle, true)].into(),
        viewport_transform_state: Some(1),
        replacements: [(key, None)].into(),
    };
    let mut source = SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage: Stage::Fragment,
            ..Default::default()
        },
        source_data,
    );
    let mut recorder = RecordingEnvironment::new(&mut source);
    assert_eq!(
        recorder.read_instruction(ProgramOffset(0x58)).unwrap(),
        0x1234
    );
    assert_eq!(
        recorder.read_cbuf_value(key.index, key.offset).unwrap(),
        0xdead_beef
    );
    assert_eq!(
        recorder.read_cbuf_value(key.index, key.offset).unwrap(),
        0xdead_beef
    );
    assert_eq!(
        recorder.read_texture_type(handle).unwrap(),
        TextureType::ColorArray2D
    );
    assert_eq!(
        recorder
            .get_replace_const_buffer(key.index, key.offset)
            .unwrap(),
        None
    );
    assert_eq!(recorder.observations().cbuf_words.len(), 1);
    assert!(recorder.observations().texture_pixel_formats.is_empty());
    let mut captured = recorder.into_snapshot();
    drop(source);
    assert_eq!(
        captured.read_cbuf_value(key.index, key.offset).unwrap(),
        0xdead_beef
    );
    assert_eq!(
        captured
            .read_cbuf_value(key.index, ByteOffset(3))
            .unwrap_err()
            .category,
        FailureCategory::MissingObservation
    );
    assert!(captured.read_texture_pixel_format(handle).is_err());
    assert!(captured.is_texture_pixel_format_integer(handle).is_err());
    assert!(captured.read_viewport_transform_state().is_err());
    assert!(
        captured
            .get_replace_const_buffer(CbufIndex(4), key.offset)
            .is_err()
    );
    let error = captured.read_instruction(ProgramOffset(0x60)).unwrap_err();
    assert_eq!(error.stage, Some(Stage::Fragment));
    assert_eq!(error.pc, Some(ProgramOffset(0x60)));
    assert!(error.to_string().contains("0x60"));
}
