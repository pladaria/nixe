use super::*;
use crate::engines::MaxwellShaderBufferSize;

#[test]
fn three_d_embedded_inline_upload_covers_captured_block_linear_gob_row() {
    let mut channel = three_d_channel();

    dispatch_incrementing(&mut channel, 0x0180 / 4, &[0x40, 1, 0, 0x0409_3000]).unwrap();
    program_three_d(&mut channel, 0x01b0, 0);

    let words = [
        0x7017_ff8f,
        0x00f5_0000,
        0x0060_0005,
        0x0007_0008,
        0xea80_000f,
        0x80ff_000f,
        0x0300_0000,
        0,
        0x0002_6412,
        0x0000_0351,
        0x00f0_0000,
        0,
        0,
        0,
        0,
        0,
    ];
    let data = non_incrementing_packet_on_subchannel(0, 0x01b4 / 4, &words);
    let (writes, _, result) = super::bulk::transfer(&mut channel, &data, false);
    result.unwrap();
    assert_eq!(
        writes,
        words
            .into_iter()
            .enumerate()
            .map(|(index, value)| (0x0409_3000 + index as u64 * 4, value))
            .collect::<Vec<_>>()
    );
    let inline = channel.three_d().inline_to_memory();
    assert_eq!(inline.line_length().value(), Some(&0x40));
    assert_eq!(inline.line_count().value(), Some(&1));
    assert_eq!(
        inline.launch().value().map(|launch| launch.layout()),
        Some(MaxwellThreeDInlineToMemoryLayout::BlockLinear)
    );
    assert_eq!(inline.last_data().value(), Some(&0));
}

#[test]
fn three_d_embedded_inline_upload_rejects_unrepresented_block_linear_rows() {
    let mut channel = three_d_channel();
    dispatch_incrementing(&mut channel, 0x0180 / 4, &[0x44, 1, 0, 0x0409_3000]).unwrap();

    let launch = packet(0x01b0 / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &launch),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "LAUNCH_DMA",
            reason: "only one contiguous row of the first block-linear GOB is implemented",
            ..
        })
    ));
}

#[test]
fn three_d_embedded_inline_upload_retains_pitch_flush_only_launch() {
    for raw in [0x11, 0x1011] {
        let mut channel = three_d_channel();
        dispatch_incrementing(&mut channel, 0x0180 / 4, &[4, 1, 0, 0x0409_3000]).unwrap();
        program_three_d(&mut channel, 0x01b0, raw);

        let register = channel.three_d().inline_to_memory().launch();
        assert_eq!(register.raw(), Some(raw));
        let launch = register.value().unwrap();
        assert_eq!(launch.layout(), MaxwellThreeDInlineToMemoryLayout::Pitch);
        assert_eq!(
            launch.completion(),
            MaxwellThreeDInlineToMemoryCompletion::FlushOnly
        );
        assert!(!launch.system_memory_barrier_disabled());
    }
}

#[test]
fn three_d_embedded_inline_upload_still_rejects_active_unsupported_launch_modes() {
    for raw in [
        0x1021, // One-word semaphore release.
        0x1031, // Undefined completion type.
        0x1111, // Interrupt enabled.
        0x1013, // Reduction enabled.
        0x1091, // Undefined bit 7.
    ] {
        let mut channel = three_d_channel();
        dispatch_incrementing(&mut channel, 0x0180 / 4, &[4, 1, 0, 0x0409_3000]).unwrap();
        let before = channel.three_d().inline_to_memory().clone();
        assert!(matches!(
            dispatch_first(&mut channel, &packet(0x01b0 / 4, raw)),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                method_name: "LAUNCH_DMA",
                ..
            })
        ));
        assert_eq!(channel.three_d().inline_to_memory(), &before);
    }
}

#[test]
fn non_indexed_resource_scope_ignores_unrelated_partial_index_shadow_state() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x17c8, 5);
    let address_space = resource_address_space();

    assert!(matches!(
        fixture_bindings(
            channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::IncompleteState {
            role: MaxwellThreeDResourceRole::IndexBuffer,
        })
    ));
    let resources = fixture_roles(
        channel.three_d(),
        address_space.memory(),
        &[],
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert!(
        resources
            .resources()
            .all(|resource| resource.role() != MaxwellThreeDResourceRole::IndexBuffer)
    );
    assert!(matches!(
        fixture_roles(
            channel.three_d(),
            address_space.memory(),
            &[MaxwellThreeDResourceRole::IndexBuffer],
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::IncompleteState {
            role: MaxwellThreeDResourceRole::IndexBuffer,
        })
    ));
}

#[test]
fn operation_resource_scope_ignores_only_unconsumed_partial_color_targets() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0810, 0xd5);
    let address_space = resource_address_space();

    assert!(matches!(
        fixture_bindings(
            channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::IncompleteState {
            role: MaxwellThreeDResourceRole::ColorTarget(0),
        })
    ));
    let resources = fixture_roles(
        channel.three_d(),
        address_space.memory(),
        &[],
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert!(resources.resources().next().is_none());
    assert!(matches!(
        fixture_roles(
            channel.three_d(),
            address_space.memory(),
            &[MaxwellThreeDResourceRole::ColorTarget(0)],
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::IncompleteState {
            role: MaxwellThreeDResourceRole::ColorTarget(0),
        })
    ));
}

#[test]
fn operation_resource_scope_ignores_an_unreferenced_contradictory_vertex_stream() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    for (method, argument) in [
        (0x1c00, 0x1018),
        (0x1c04, 1),
        (0x1c08, 0x2000),
        (0x1f00, 1),
        (0x1f04, 0x1000),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let address_space = resource_address_space();

    let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
    let operation = &dispatch.operations()[0];
    let mut roles = Vec::new();
    operation
        .trigger()
        .append_resource_roles(operation.state(), &mut roles);
    assert!(roles.is_empty());
    let resources = fixture_roles(
        operation.state(),
        address_space.memory(),
        &roles,
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert!(resources.resources().next().is_none());

    let mut consumed = channel;
    program_three_d(&mut consumed, 0x1160, 0x3840_0000);
    let dispatch = dispatch_method(&mut consumed, 0x0d78 / 4, 3).unwrap();
    let operation = &dispatch.operations()[0];
    let mut roles = Vec::new();
    operation
        .trigger()
        .append_resource_roles(operation.state(), &mut roles);
    assert_eq!(roles, [MaxwellThreeDResourceRole::VertexStream(0)]);
    assert!(matches!(
        fixture_roles(
            operation.state(),
            address_space.memory(),
            &roles,
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::ContradictoryState {
            role: MaxwellThreeDResourceRole::VertexStream(0),
        })
    ));
}

#[test]
fn rgba16_float_targets_preserve_texel_width_and_reject_incompatible_kinds() {
    for kind in [0xfe, 0xe9, 0xdb, 0xe0] {
        let mut address_space = resource_address_space();
        let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x10000);
        let mapping = map_resource(
            &mut address_space,
            crate::test_memory::ranges(&allocation),
            14,
            kind,
        );
        let mut channel = three_d_channel();
        program_color_target(&mut channel, 0, mapping.offset().get(), 0xca);
        program_three_d(&mut channel, 0x15d0, 0);
        let roles = [MaxwellThreeDResourceRole::ColorTarget(0)];
        let result = fixture_roles(
            channel.three_d(),
            address_space.memory(),
            &roles,
            &mut crate::test_memory::access(address_space.memory().machine()),
        );
        if matches!(kind, 0xdb | 0xe0) {
            assert!(
                matches!(result, Err(MaxwellThreeDResourceError::UnsupportedKind { expected: 0xe9, actual, .. }) if actual == kind)
            );
            continue;
        }
        let resources = result.unwrap();
        let MaxwellThreeDResolvedResource::Image(image) = resources.resources().next().unwrap()
        else {
            panic!("image expected")
        };
        assert_eq!(image.description().format(), ImageFormat::Rgba16Float);
        assert_eq!(
            image.description().format().plane_bytes_per_texel(0),
            Some(8)
        );
        assert_eq!(image.guest_layout().pte_kind(), kind);
        assert_eq!(image.source().size(), 64 * 32 * 8);
        // Adding the single-sample format must not accept a 64-bit MSAA image
        // using the hard-coded C32_MS4 storage family.
        program_three_d(&mut channel, 0x15d0, 2);
        assert!(matches!(
            fixture_roles(
                channel.three_d(),
                address_space.memory(),
                &roles,
                &mut crate::test_memory::access(address_space.memory().machine())
            ),
            Err(MaxwellThreeDResourceError::UnsupportedImageLayout { .. })
        ));
    }
}

#[test]
fn captured_array_color_target_uses_the_verified_zero_layer_reset() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x40_0000);
    let backing = crate::test_memory::ranges(&allocation);
    let mapping = map_resource(&mut address_space, backing, 12, 0xfe);
    let address = mapping.offset().get();
    let mut channel = three_d_channel();

    // deko_basic programs the complete attachment description but relies on
    // SET_COLOR_TARGET_LAYER(0)'s architectural zero reset.
    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 1280),
        (0x080c, 720),
        (0x0810, 0xd5),
        (0x0814, 0x40),
        (0x0818, 1),
        (0x081c, 0x000f_0000),
        (0x15d0, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let target = &channel.three_d().render_targets().color()[0];

    assert_eq!(target.layer().value(), Some(&0));
    let resources = fixture_roles(
        channel.three_d(),
        address_space.memory(),
        &[MaxwellThreeDResourceRole::ColorTarget(0)],
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert_eq!(resources.resources().count(), 1);
    assert_eq!(
        resources.resources().next().unwrap().role(),
        MaxwellThreeDResourceRole::ColorTarget(0)
    );
}

#[test]
fn captured_deko_basic_color_target_accepts_its_matching_c32_pte_kind() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x40_0000);
    let backing = crate::test_memory::ranges(&allocation);
    let mapping = map_resource(&mut address_space, backing, 13, 0xdb);
    let address = mapping.offset().get();
    let mut channel = three_d_channel();

    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 1280),
        (0x080c, 720),
        (0x0810, 0xd5),
        (0x0814, 0x40),
        (0x0818, 1),
        (0x081c, 0x000f_0000),
        (0x15d0, 0),
        (0x19e0, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let resources = fixture_roles(
        channel.three_d(),
        address_space.memory(),
        &[MaxwellThreeDResourceRole::ColorTarget(0)],
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    let image = resources
        .resources()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image) => Some(image),
            MaxwellThreeDResolvedResource::Buffer(_) => None,
        })
        .expect("captured color target must resolve as an image");

    assert_eq!(image.guest_layout().pte_kind(), 0xdb);
    assert!(image.guest_layout().requires_materialization());
    assert!(!image.guest_layout().has_direct_canonical_representation());
}

#[test]
fn three_d_depth_layer_selects_one_array_subresource() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x10000);
    let backing = crate::test_memory::ranges(&allocation);
    let mapping = map_resource(&mut address_space, backing, 14, 0xfe);
    let address = mapping.offset().get();
    let mut channel = three_d_channel();

    for (method, argument) in [
        (0x0fe0, (address >> 32) as u32),
        (0x0fe4, address as u32),
        (0x0fe8, 0x13),
        (0x0fec, 0),
        (0x0ff0, 0x1000),
        (0x1228, 64),
        (0x122c, 32),
        (0x1230, 2),
        (0x1538, 1),
        (0x179c, 1),
        (0x15d0, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let resolved = fixture_bindings(
        channel.three_d(),
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    let depth = resolved
        .resources()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .expect("depth/stencil target must resolve as an image");
    assert_eq!(depth.description().array_layers(), 2);
    assert_eq!(
        depth.description().dimension(),
        nixe_gpu::ImageDimension::Two
    );
    assert_eq!(depth.description().extent().depth, 1);
    assert_eq!(depth.view().bindings().len(), 1);
    assert_eq!(
        depth.view().bindings()[0].subresources(),
        nixe_gpu::ImageSubresourceRange {
            plane: 0,
            mip_level: 0,
            base_layer: 1,
            layer_count: 1,
        }
    );
    assert_eq!(depth.source().size(), 0x4000);

    program_three_d(&mut channel, 0x179c, 2);
    assert!(matches!(
        fixture_bindings(
            channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::ContradictoryState {
            role: MaxwellThreeDResourceRole::DepthStencilTarget,
        })
    ));

    program_three_d(&mut channel, 0x1230, 0x0001_0002);
    program_three_d(&mut channel, 0x179c, 1);
    assert!(matches!(
        fixture_bindings(
            channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::ContradictoryState {
            role: MaxwellThreeDResourceRole::DepthStencilTarget,
        })
    ));
    program_three_d(&mut channel, 0x179c, 0);
    for raw in [1, 0x10000, 0x10001, 0x10002, 0x1ffff] {
        program_three_d(&mut channel, 0x1230, raw);
        let resolved = fixture_bindings(
            channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine()),
        )
        .unwrap();
        let depth = resolved
            .resources()
            .find_map(|resource| match resource {
                MaxwellThreeDResolvedResource::Image(image)
                    if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
                {
                    Some(image)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(
            depth.description().dimension(),
            nixe_gpu::ImageDimension::Two
        );
        assert_eq!(depth.description().extent().depth, 1);
        assert_eq!(depth.description().array_layers(), 1);
        assert_eq!(depth.view().bindings()[0].subresources().base_layer, 0);
    }
    program_three_d(&mut channel, 0x1230, 0);
    assert!(matches!(
        fixture_bindings(
            channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::ContradictoryState {
            role: MaxwellThreeDResourceRole::DepthStencilTarget
        })
    ));
}

#[test]
fn three_d_stencil8_z24_preserves_guest_packing_with_neutral_depth_stencil_semantics() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x10000);
    let mapping = map_resource(
        &mut address_space,
        crate::test_memory::ranges(&allocation),
        15,
        0xfe,
    );
    let address = mapping.offset().get();
    let mut channel = three_d_channel();

    for (method, argument) in [
        (0x0fe0, (address >> 32) as u32),
        (0x0fe4, address as u32),
        (0x0fe8, 0x16),
        (0x0fec, 0),
        (0x0ff0, 0x2000),
        (0x1228, 64),
        (0x122c, 32),
        (0x1230, 1),
        (0x1538, 1),
        (0x179c, 0),
        (0x15d0, 0),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let resolved = fixture_bindings(
        channel.three_d(),
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    let depth = resolved
        .resources()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .expect("depth/stencil target must resolve as an image");
    assert_eq!(
        depth.description().format(),
        ImageFormat::Depth24UnormStencil8Uint
    );
    assert_eq!(
        depth.guest_format(),
        MaxwellThreeDGuestImageFormat::DepthStencil(MaxwellThreeDDepthStencilFormat::Stencil8Z24)
    );
    assert_eq!(depth.source().size(), 0x8000);

    program_three_d(&mut channel, 0x0fe8, 0x14);
    let resolved = fixture_bindings(
        channel.three_d(),
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    let depth = resolved
        .resources()
        .find_map(|resource| match resource {
            MaxwellThreeDResolvedResource::Image(image)
                if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                Some(image)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(
        depth.description().format(),
        ImageFormat::Depth24UnormStencil8Uint
    );
    assert_eq!(
        depth.guest_format(),
        MaxwellThreeDGuestImageFormat::DepthStencil(MaxwellThreeDDepthStencilFormat::Z24Stencil8)
    );
}

#[test]
fn compressed_depth_kinds_preserve_format_and_reject_other_packing_msaa_and_color_kinds() {
    for (format, expected_kind) in [(0x13, 0x07), (0x16, 0x17), (0x14, 0x51)] {
        for kind in [0x07, 0x17, 0x51, 0x18, 0x52, 0xdb, 0xfe] {
            let mut address_space = resource_address_space();
            let allocation =
                crate::test_memory::allocate(address_space.memory().machine(), 0x10000);
            let mapping = map_resource(
                &mut address_space,
                crate::test_memory::ranges(&allocation),
                16,
                kind,
            );
            let address = mapping.offset().get();
            for compression in [None, Some(0), Some(1)] {
                let mut channel = three_d_channel();
                for (method, argument) in [
                    (0x0fe0, (address >> 32) as u32),
                    (0x0fe4, address as u32),
                    (0x0fe8, format),
                    (0x0fec, 0),
                    (0x0ff0, 0x2000),
                    (0x1228, 64),
                    (0x122c, 32),
                    (0x1230, 1),
                    (0x1538, 1),
                    (0x179c, 0),
                    (0x15d0, 0),
                ] {
                    program_three_d(&mut channel, method, argument);
                }
                if let Some(compression) = compression {
                    program_three_d(&mut channel, 0x19cc, compression);
                }
                let result = fixture_bindings(
                    channel.three_d(),
                    address_space.memory(),
                    &mut crate::test_memory::access(address_space.memory().machine()),
                );
                if kind == expected_kind || (kind == 0xfe && compression != Some(1)) {
                    let resolved = result.unwrap();
                    let depth = resolved
                        .resources()
                        .find_map(|resource| match resource {
                            MaxwellThreeDResolvedResource::Image(image)
                                if image.role()
                                    == MaxwellThreeDResourceRole::DepthStencilTarget =>
                            {
                                Some(image)
                            }
                            _ => None,
                        })
                        .unwrap();
                    assert_eq!(depth.guest_layout().pte_kind(), kind);
                    assert_eq!(
                        depth.guest_layout().has_direct_canonical_representation(),
                        kind == 0xfe
                    );
                    if format == 0x13 {
                        assert_eq!(depth.description().format(), ImageFormat::Depth16Unorm);
                    }
                } else {
                    assert!(
                        matches!(result,
                            Err(MaxwellThreeDResourceError::UnsupportedKind {
                                role: MaxwellThreeDResourceRole::DepthStencilTarget, expected, actual,
                            }) if expected == expected_kind && actual == kind
                        ),
                        "format={format:x}, kind={kind:x}: {result:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn three_d_surface_clip_pair_preserves_sources_and_origin_extent() {
    let mut channel = three_d_channel();
    let dispatch =
        dispatch_incrementing(&mut channel, 0x0ff4 / 4, &[0x0500_0000, 0x02d0_0000]).unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_SURFACE_CLIP_HORIZONTAL"
    );
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "SET_SURFACE_CLIP_VERTICAL"
    );
    let state = channel.three_d().fixed_function();
    let horizontal = state.surface_clip_horizontal();
    let vertical = state.surface_clip_vertical();
    assert_eq!(horizontal.raw(), Some(0x0500_0000));
    assert_eq!(horizontal.value().unwrap().origin(), 0);
    assert_eq!(horizontal.value().unwrap().extent(), 1280);

    assert_eq!(vertical.raw(), Some(0x02d0_0000));
    assert_eq!(vertical.value().unwrap().origin(), 0);
    assert_eq!(vertical.value().unwrap().extent(), 720);
}

#[test]
fn standalone_inline_to_memory_pitch_upload_tracks_state_words_and_ordered_effects() {
    let mut channel = inline_to_memory_channel();
    let address = 0x04_082b_30c0_u64;
    let setup = incrementing_packet_on_subchannel(
        2,
        0x0180 / 4,
        &[8, 1, (address >> 32) as u32, address as u32, 8],
    );
    let setup_dispatch = dispatch_first(&mut channel, &setup).unwrap();
    assert_eq!(
        setup_dispatch.methods()[2].metadata().method_name(),
        "OFFSET_OUT_UPPER"
    );
    assert_eq!(channel.inline_to_memory().address_upper().value(), Some(&4));
    assert_eq!(channel.inline_to_memory().pitch().value(), Some(&8));

    let launch = packet_on_subchannel(2, 0x01b0 / 4, 0x1001);
    dispatch_first(&mut channel, &launch).unwrap();
    let launch = channel.inline_to_memory().launch();
    assert_eq!(launch.raw(), Some(0x1001));
    assert_eq!(
        launch.value().unwrap().semaphore_structure_size(),
        MaxwellInlineToMemorySemaphoreStructureSize::OneWord
    );
    assert!(!launch.value().unwrap().system_memory_barrier_disabled());
    let data = non_incrementing_packet_on_subchannel(2, 0x01b4 / 4, &[0x1122_3344, 0x5566_7788]);
    let (writes, _, result) = super::bulk::transfer(&mut channel, &data, false);
    result.unwrap();

    assert!(channel.inline_to_memory().pending().is_none());
    assert_eq!(
        channel.inline_to_memory().last_data().value(),
        Some(&0x5566_7788)
    );
    assert_eq!(writes, [(address, 0x1122_3344), (address + 4, 0x5566_7788)]);
}

#[test]
fn standalone_inline_to_memory_rejects_invalid_sequences_atomically() {
    let mut channel = inline_to_memory_channel();

    let before = channel.clone();
    let data = packet_on_subchannel(2, 0x01b4 / 4, 0x1122_3344);
    assert!(matches!(
        dispatch_first(&mut channel, &data),
        Err(
            MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
                method_name: "LOAD_INLINE_DATA",
                ..
            }
        )
    ));
    assert_eq!(channel, before);

    let setup = incrementing_packet_on_subchannel(2, 0x0180 / 4, &[4, 2, 4, 0x082b_30c0, 3]);
    dispatch_first(&mut channel, &setup).unwrap();
    let before = channel.clone();
    let unsupported_completion = packet_on_subchannel(2, 0x01b0 / 4, 0x21);
    assert!(matches!(
        dispatch_first(&mut channel, &unsupported_completion),
        Err(
            MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
                method_name: "LAUNCH_DMA",
                reason: "interrupt, reduction, and semaphore release inline uploads are not implemented",
                ..
            }
        )
    ));
    assert_eq!(channel, before);

    let launch = packet_on_subchannel(2, 0x01b0 / 4, 0x41);
    assert!(matches!(
        dispatch_first(&mut channel, &launch),
        Err(
            MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
                method_name: "LAUNCH_DMA",
                reason: "destination pitch is smaller than the line length",
                ..
            }
        )
    ));
    assert_eq!(channel, before);
}

#[test]
fn constant_buffer_inline_load_tracks_typed_cursor_and_upload_effects() {
    let mut channel = three_d_channel();
    dispatch_incrementing(&mut channel, 0x2380 / 4, &[0x10, 0, 0x82c_3000]).unwrap();

    let dispatch =
        dispatch_incrementing(&mut channel, 0x238c / 4, &[0, 0x1122_3344, 0x5566_7788]).unwrap();
    assert_eq!(dispatch.methods().len(), 3);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "LOAD_CONSTANT_BUFFER_OFFSET"
    );
    let state = channel.three_d().shader_bindings().constant_buffer_load();
    assert_eq!(state.offset().value(), Some(&0));

    assert_eq!(state.next_offset(), Some(8));
    assert_eq!(state.last_data().value(), Some(&0x5566_7788));

    for index in 0..2 {
        let method = dispatch.methods()[index + 1];
        assert_eq!(method.metadata().method_name(), "LOAD_CONSTANT_BUFFER");
    }
}

#[test]
fn constant_buffer_inline_load_validates_fields_sequence_bounds_and_atomicity() {
    let mut channel = three_d_channel();

    let before = channel.three_d().clone();
    let data_without_selector = packet(0x2390 / 4, 1);
    assert!(matches!(
        dispatch_first(&mut channel, &data_without_selector),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "LOAD_CONSTANT_BUFFER",
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    let invalid_offset = packet(0x238c / 4, 0x1_0000);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_offset),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "LOAD_CONSTANT_BUFFER_OFFSET",
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    for (method, argument) in [(0x2380, 4), (0x2384, 0), (0x2388, 0x1000)] {
        program_three_d(&mut channel, method, argument);
    }
    let before_packet = channel.three_d().clone();
    let overflow = increment_once_packet(0x238c / 4, &[0, 1, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &overflow),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "LOAD_CONSTANT_BUFFER",
            ..
        }) if source.argument() == 2
    ));
    assert_ne!(channel.three_d(), &before_packet);
}

#[test]
fn conditional_constant_buffer_inline_load_is_an_explicit_host_boundary() {
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x2380, 4),
        (0x2384, 0),
        (0x2388, 0x1000),
        (0x238c, 0),
        (0x030c, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let before = channel.three_d().clone();
    let data = packet(0x2390 / 4, 0xfeed_beef);
    assert!(matches!(
        dispatch_first(&mut channel, &data),
        Err(MaxwellEngineDispatchError::UnsupportedConditionalConstantBufferLoad {
            source,
        }) if source.argument() == 0xfeed_beef
    ));
    assert_eq!(channel.three_d(), &before);

    program_three_d(&mut channel, 0x030c, 0);
    program_three_d(&mut channel, 0x2390, 0xfeed_beef);
    assert_eq!(
        channel
            .three_d()
            .shader_bindings()
            .constant_buffer_load()
            .next_offset(),
        Some(4)
    );
}

#[test]
fn incomplete_constant_buffer_binding_is_rejected() {
    let mut channel = three_d_channel();
    let before = channel.three_d().clone();
    let bind = packet(0x2410 / 4, 1);
    assert!(matches!(
        dispatch_first(&mut channel, &bind),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn resource_snapshot_resolves_buffers_aliases_and_content_generations() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x4000);
    let backing = crate::test_memory::ranges(&allocation);
    let first = map_resource(&mut address_space, backing.clone(), 11, 0);
    let alias = map_resource(&mut address_space, backing, 11, 0);
    let mut channel = three_d_channel();

    let vertex = first.offset().get();
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, (vertex >> 32) as u32),
        (0x1c08, vertex as u32),
        (0x1f00, (vertex >> 32) as u32),
        (0x1f04, (vertex + 0xff) as u32),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let constant = alias.offset().get();
    for (method, argument) in [
        (0x2380, 0x100),
        (0x2384, (constant >> 32) as u32),
        (0x2388, constant as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    crate::test_memory::write(&allocation, 0, &[0x5a]).unwrap();
    let resolved = fixture_bindings(
        channel.three_d(),
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert_eq!(resolved.resources().count(), 2);
    let MaxwellThreeDResolvedResource::Buffer(vertex) = resolved.resources().next().unwrap() else {
        panic!("vertex stream must resolve as a buffer");
    };
    assert_eq!(vertex.role(), MaxwellThreeDResourceRole::VertexStream(0));
    assert_eq!(vertex.view().size(), 0x100);
    assert_eq!(
        vertex.view().backing().mapping().spans()[0].allocation,
        allocation.pin().identity()
    );
    address_space.unmap(first.offset()).unwrap();
    assert!(matches!(
        resolved.validate_mappings(address_space.memory()),
        Err(MaxwellThreeDResourceError::StaleMapping { mapping, .. })
            if mapping == first.id()
    ));
}

#[test]
fn resource_snapshot_preserves_block_linear_targets_and_tracks_dirty_images() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x10000);
    let backing = crate::test_memory::ranges(&allocation);
    let mapping = map_resource(&mut address_space, backing, 12, 0xfe);
    let address = mapping.offset().get();
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x121c, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let resolved = fixture_bindings(
        channel.three_d(),
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert_eq!(resolved.resources().count(), 1);
    let MaxwellThreeDResolvedResource::Image(image) = resolved.resources().next().unwrap() else {
        panic!("color target must resolve as an image");
    };
    assert_eq!(
        image.description().format(),
        nixe_gpu::ImageFormat::Rgba8Unorm
    );
    assert_eq!(image.source().size(), 0x2000);
    assert_eq!(
        image.guest_layout().layout(),
        nixe_gpu::ImageMemoryLayout::BlockLinear(nixe_gpu::BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            layer_stride: 0x2000,
        })
    );
}

#[test]
fn resource_resolution_ignores_unused_descriptor_memory_and_rejects_consumed_images() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x10000);
    let backing = crate::test_memory::ranges(&allocation);
    let mapping = map_resource(&mut address_space, backing, 13, 0);
    let address = mapping.offset().get();
    let mut state_channel = channel();
    bind_three_d(&mut state_channel);
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, (address >> 32) as u32),
        (0x1c08, address as u32),
        (0x1f00, (address >> 32) as u32),
        (0x1f04, (address + 0xff) as u32),
        (0x1574, 0),
        (0x1578, 0x4000),
        (0x157c, 1),
    ] {
        program_three_d(&mut state_channel, method, argument);
    }
    let state = state_channel.three_d().clone();

    let resolved = fixture_bindings(
        &state,
        address_space.memory(),
        &mut crate::test_memory::access(address_space.memory().machine()),
    )
    .unwrap();
    assert_eq!(resolved.resources().count(), 1);
    assert_eq!(
        resolved.resources().next().unwrap().role(),
        MaxwellThreeDResourceRole::VertexStream(0)
    );
    assert_eq!(state_channel.three_d(), &state);

    let mut image_channel = channel();
    bind_three_d(&mut image_channel);
    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x121c, 1),
    ] {
        program_three_d(&mut image_channel, method, argument);
    }
    assert!(matches!(
        fixture_bindings(
            image_channel.three_d(),
            address_space.memory(),
            &mut crate::test_memory::access(address_space.memory().machine())
        ),
        Err(MaxwellThreeDResourceError::UnsupportedKind {
            role: MaxwellThreeDResourceRole::ColorTarget(0),
            expected: 0xdb,
            actual: 0,
        })
    ));
}

#[test]
fn three_d_register_writes_do_not_mutate_compute_state() {
    let mut channel = three_d_channel();
    let compute_before = channel.compute().clone();
    dispatch_method(&mut channel, 0x1518 / 4, 0x3f80_0000).unwrap();
    assert_eq!(channel.compute(), &compute_before);
    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(0x3f80_0000)
    );
}

#[test]
fn taxonomy_separates_unsupported_invalid_capability_and_unknown_methods() {
    let mut channel = three_d_channel();
    let cases = [
        (0x10c, 1, "known Maxwell method is not implemented"),
        (
            0x124,
            4,
            "argument sets bits outside its verified field mask",
        ),
        (0x304, 1, "requires an unavailable execution capability"),
        (0x2ffc, 0, "unknown Maxwell class method"),
    ];
    for (method, argument, expected) in cases {
        let error = dispatch_method(&mut channel, method / 4, argument).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn ct_mrt_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDSeparateFragmentData::Disabled),
        (1, MaxwellThreeDSeparateFragmentData::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0fac / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let register = channel.three_d().render_targets().separate_fragment_data();

        assert_eq!(method.metadata().method_name(), "SET_CT_MRT_ENABLE");

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0fac / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CT_MRT_ENABLE",
                reason: "undefined boolean encoding or reserved bits",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x0fac / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0fb0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn compute_shader_memory_state_is_typed_source_preserving_and_atomic() {
    let mut channel = compute_channel();
    let three_d_before = channel.three_d().clone();
    let two_d_before = channel.two_d().clone();

    let upper = packet_on_subchannel(1, 0x0790 / 4, 4);
    let dispatch = dispatch_first(&mut channel, &upper).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_SHADER_LOCAL_MEMORY_A"
    );

    let memory = channel.compute().shader_local_memory();
    assert_eq!(memory.address(), None);
    assert_eq!(memory.address_upper().raw(), Some(4));
    assert_eq!(memory.address_upper().value(), Some(&4));
    assert_eq!(
        memory.address_upper().origin(),
        MaxwellComputeRegisterOrigin::Programmed
    );
    assert!(memory.address_upper().source().is_some());

    let lower = packet_on_subchannel(1, 0x0794 / 4, 0x0008_0000);
    dispatch_first(&mut channel, &lower).unwrap();
    assert_eq!(
        channel.compute().shader_local_memory().address(),
        Some(MaxwellComputeAddress::new(4, 0x0008_0000))
    );
    assert_eq!(
        channel
            .compute()
            .shader_local_memory()
            .address()
            .unwrap()
            .get(),
        0x0000_0004_0008_0000
    );

    for (method, arguments) in [
        (0x02e4, [0, 0x0040_8000, 0xff]),
        (0x02f0, [0, 0x0040_8000, 0xff]),
    ] {
        let decoded = incrementing_packet_on_subchannel(1, method / 4, &arguments);
        dispatch_first(&mut channel, &decoded).unwrap();
    }
    let memory = channel.compute().shader_local_memory();
    for allocation in [memory.non_throttled(), memory.throttled()] {
        assert_eq!(allocation.size(), Some(0x0040_8000));
        assert_eq!(allocation.max_sm_count().value().unwrap().get(), 0xff);
        assert!(allocation.size_upper().source().is_some());
        assert!(allocation.size_lower().source().is_some());
        assert!(allocation.max_sm_count().source().is_some());
    }

    for (method, argument, name) in [
        (0x077c, 0xff00_0000, "SET_SHADER_LOCAL_MEMORY_WINDOW"),
        (0x0214, 0xfe00_0000, "SET_SHADER_SHARED_MEMORY_WINDOW"),
    ] {
        let decoded = packet_on_subchannel(1, method / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
    }
    let memory = channel.compute().shader_local_memory();
    assert_eq!(memory.local_window_base().value(), Some(&0xff00_0000));
    assert_eq!(memory.shared_window_base().value(), Some(&0xfe00_0000));
    assert_eq!(channel.three_d(), &three_d_before);
    assert_eq!(channel.two_d(), &two_d_before);

    for (method, argument, mask) in [
        (0x0790, 0x100, 0xff),
        (0x02e4, 0x100, 0xff),
        (0x02ec, 0x200, 0x1ff),
        (0x02f0, 0x100, 0xff),
        (0x02f8, 0x200, 0x1ff),
    ] {
        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let decoded = packet_on_subchannel(1, method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let decoded = incrementing_packet_on_subchannel(1, 0x02e4 / 4, &[1, 2, 0x200]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x02ec)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.compute(), &compute_before);
}

#[test]
fn compute_shader_exceptions_preserve_configuration_without_emitting_work() {
    let mut channel = compute_channel();
    let three_d_before = channel.three_d().clone();
    let two_d_before = channel.two_d().clone();
    let register = channel.compute().shader_exceptions_enable();
    assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Unset);
    assert_eq!(register.raw(), None);
    assert_eq!(register.value(), None);

    // The captured initialization disables exceptions. Enabling and disabling
    // again are valid configuration writes, not launches or completion events.
    for raw in [0, 1, 0] {
        let packet = packet_on_subchannel(1, 0x1528 / 4, raw);
        let dispatch = dispatch_first(&mut channel, &packet).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SHADER_EXCEPTIONS"
        );
        assert!(dispatch.ordered_operations().is_empty());
        let register = channel.compute().shader_exceptions_enable();
        assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(raw));
        assert_eq!(register.value(), Some(&(raw != 0)));
    }

    assert_eq!(channel.three_d(), &three_d_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(
        compute_channel()
            .compute()
            .shader_exceptions_enable()
            .value(),
        None
    );
}

#[test]
fn compute_shader_exceptions_reject_reserved_bits_without_mutation() {
    let mut channel = compute_channel();
    dispatch_first(&mut channel, &packet_on_subchannel(1, 0x1528 / 4, 1)).unwrap();
    for raw in [2, 3, 0x8000_0000, u32::MAX] {
        let before = channel.clone();
        let packet = packet_on_subchannel(1, 0x1528 / 4, raw);
        assert!(matches!(
            dispatch_first(&mut channel, &packet),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source, metadata, defined_mask: 1,
            }) if source.method() == GpuMethodId(0x1528)
                && source.argument() == raw
                && metadata.method_name() == "SET_SHADER_EXCEPTIONS"
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn compute_program_state_is_typed_source_preserving_and_atomic() {
    let mut channel = compute_channel();

    let lower = packet_on_subchannel(1, 0x160c / 4, 0x0123_4000);
    let dispatch = dispatch_first(&mut channel, &lower).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_B"
    );

    assert_eq!(channel.compute().program().region_address(), None);

    let upper = packet_on_subchannel(1, 0x1608 / 4, 4);
    dispatch_first(&mut channel, &upper).unwrap();
    let program = channel.compute().program();
    assert_eq!(
        program.region_address(),
        Some(MaxwellComputeAddress::new(4, 0x0123_4000))
    );
    assert_eq!(
        program.region_address().unwrap().get(),
        0x0000_0004_0123_4000
    );
    assert_eq!(program.region_address_upper().raw(), Some(4));
    assert_eq!(program.region_address_lower().raw(), Some(0x0123_4000));
    assert_eq!(
        program.region_address_upper().origin(),
        MaxwellComputeRegisterOrigin::Programmed
    );
    assert!(program.region_address_upper().source().is_some());
    assert!(program.region_address_lower().source().is_some());

    let spa = packet_on_subchannel(1, 0x0310 / 4, 0x0400);
    let dispatch = dispatch_first(&mut channel, &spa).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_SPA_VERSION"
    );
    let version = *channel.compute().program().spa_version().value().unwrap();
    assert_eq!(version.major(), 4);
    assert_eq!(version.minor(), 0);
    assert_eq!(version.raw(), 0x0400);
    assert_eq!(
        channel.compute().program().spa_version().raw(),
        Some(0x0400)
    );
    assert!(channel.compute().program().spa_version().source().is_some());

    for (method, argument, mask) in [(0x1608, 0x100, 0xff), (0x0310, 0x1_0000, 0xffff)] {
        let compute_before = channel.compute().clone();
        let decoded = packet_on_subchannel(1, method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let decoded = non_incrementing_packet_on_subchannel(1, 0x0310 / 4, &[0x0501, 0x1_0000]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x0310)
                && source.argument() == 0x1_0000
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.compute(), &compute_before);
}

#[test]
fn compute_descriptor_pools_are_typed_source_preserving_and_atomic() {
    let mut channel = compute_channel();

    let header_lower = packet_on_subchannel(1, 0x1578 / 4, 0x0410_0000);
    dispatch_first(&mut channel, &header_lower).unwrap();
    assert_eq!(channel.compute().texture_headers().address(), None);

    let header_upper = packet_on_subchannel(1, 0x1574 / 4, 4);
    let dispatch = dispatch_first(&mut channel, &header_upper).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_TEX_HEADER_POOL_A"
    );

    let header_maximum = packet_on_subchannel(1, 0x157c / 4, 0x003f_ffff);
    dispatch_first(&mut channel, &header_maximum).unwrap();
    let headers = channel.compute().texture_headers();
    assert_eq!(
        headers.address(),
        Some(MaxwellComputeAddress::new(4, 0x0410_0000))
    );
    assert_eq!(headers.address().unwrap().get(), 0x0000_0004_0410_0000);
    assert_eq!(headers.maximum_index().value(), Some(&0x003f_ffff));
    assert!(headers.address_upper().source().is_some());
    assert!(headers.address_lower().source().is_some());
    assert!(headers.maximum_index().source().is_some());

    let sampler = incrementing_packet_on_subchannel(1, 0x155c / 4, &[4, 0x0411_0000, 0x000f_ffff]);
    let dispatch = dispatch_first(&mut channel, &sampler).unwrap();
    assert_eq!(dispatch.methods().len(), 3);
    assert_eq!(
        dispatch.methods()[2].metadata().method_name(),
        "SET_TEX_SAMPLER_POOL_C"
    );
    let samplers = channel.compute().samplers();
    assert_eq!(
        samplers.address(),
        Some(MaxwellComputeAddress::new(4, 0x0411_0000))
    );
    assert_eq!(samplers.maximum_index().value(), Some(&0x000f_ffff));
    assert!(samplers.address_upper().source().is_some());
    assert!(samplers.address_lower().source().is_some());
    assert!(samplers.maximum_index().source().is_some());

    for (method, argument, mask) in [
        (0x155c, 0x100, 0xff),
        (0x1564, 0x0010_0000, 0x000f_ffff),
        (0x1574, 0x100, 0xff),
        (0x157c, 0x0040_0000, 0x003f_ffff),
    ] {
        let compute_before = channel.compute().clone();
        let decoded = packet_on_subchannel(1, method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let invalid_header =
        incrementing_packet_on_subchannel(1, 0x1574 / 4, &[5, 0x1234_0000, 0x0040_0000]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_header),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x157c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.compute(), &compute_before);
}

#[test]
fn compute_bindless_texture_slot_is_typed_source_preserving_and_atomic() {
    let mut channel = compute_channel();
    assert_eq!(
        channel
            .compute()
            .bindless_texture_constant_buffer_slot()
            .origin(),
        MaxwellComputeRegisterOrigin::Unset
    );

    let slots = non_incrementing_packet_on_subchannel(1, 0x2608 / 4, &[0, 7]);
    let dispatch = dispatch_first(&mut channel, &slots).unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "SET_BINDLESS_TEXTURE"
    );

    let slot = channel.compute().bindless_texture_constant_buffer_slot();
    assert_eq!(slot.value().unwrap().get(), 7);
    assert_eq!(slot.raw(), Some(7));
    assert_eq!(slot.origin(), MaxwellComputeRegisterOrigin::Programmed);
    assert_eq!(slot.source().unwrap().method(), GpuMethodId(0x2608));

    let compute_before = channel.compute().clone();
    let invalid = packet_on_subchannel(1, 0x2608 / 4, 8);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_0007,
            ..
        }) if source.argument() == 8
    ));
    assert_eq!(channel.compute(), &compute_before);

    let frontend_before = channel.frontend();
    let invalid_packet = non_incrementing_packet_on_subchannel(1, 0x2608 / 4, &[6, 8]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_packet),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.argument() == 8
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.compute(), &compute_before);
}

#[test]
fn compute_inline_pitch_completion_modes_preserve_split_transfer_state() {
    for (raw, flush, barrier_disabled) in [
        (0x01, false, false),
        (0x11, true, false),
        (0x41, false, true),
        (0x51, true, true),
    ] {
        let mut channel = compute_channel();
        let setup = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[8, 1, 0, 0x0408_3000]);
        dispatch_first(&mut channel, &setup).unwrap();
        let launch = packet_on_subchannel(1, 0x01b0 / 4, raw);
        let dispatch = dispatch_first(&mut channel, &launch).unwrap();
        assert!(dispatch.ordered_operations().is_empty());
        let inline = channel.compute().inline_to_memory();
        assert_eq!(inline.launch().raw(), Some(raw));
        assert_eq!(inline.launch().source().unwrap().argument(), raw);
        let config = *inline.launch().value().unwrap();
        assert_eq!(config.layout(), MaxwellComputeInlineToMemoryLayout::Pitch);
        assert_eq!(config.flush_on_completion(), flush);
        assert_eq!(config.system_memory_barrier_disabled(), barrier_disabled);

        for (index, value) in [0x1122_3344, 0x5566_7788].into_iter().enumerate() {
            let data = packet_on_subchannel(1, 0x01b4 / 4, value);
            let dispatch = dispatch_first(&mut channel, &data).unwrap();
            let [MaxwellEngineOperation::ComputeInlineToMemory(upload)] =
                dispatch.ordered_operations()
            else {
                panic!("one inline word must emit exactly one write, not a completion signal");
            };
            assert_eq!(upload.address().get(), 0x0408_3000);
            assert_eq!(upload.offset(), index as u32 * 4);
            assert_eq!(upload.value(), value);
            assert_eq!(upload.source().argument(), value);
            let inline = channel.compute().inline_to_memory();
            assert_eq!(*inline.launch().value().unwrap(), config);
            assert_eq!(
                inline.pending().map(|pending| pending.next_offset()),
                if index == 0 { Some(4) } else { None }
            );
            if index == 0 {
                let before = channel.compute().clone();
                assert!(matches!(
                    dispatch_first(&mut channel, &launch),
                    Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
                        reason: "cannot replace an incomplete inline upload",
                        ..
                    })
                ));
                assert_eq!(channel.compute(), &before);
            }
        }
    }
}

#[test]
fn compute_inline_pitch_unsupported_launch_modes_do_not_arm_transfers() {
    let mut channel = compute_channel();
    let setup = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[4, 1, 0, 0x0408_3000]);
    dispatch_first(&mut channel, &setup).unwrap();
    // Block-linear, semaphore release, reserved completion selector, reduction,
    // interrupt and semaphore structure configuration remain unsupported.
    for raw in [0x10, 0x21, 0x31, 0x13, 0x15, 0x2011, 0x111, 0x1011] {
        let before = channel.clone();
        assert!(matches!(
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x01b0 / 4, raw)),
            Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
                method_name: "LAUNCH_DMA", source, ..
            }) if source.argument() == raw
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn compute_inline_to_memory_pitch_upload_is_typed_ordered_and_atomic() {
    let mut channel = compute_channel();

    let empty_before = channel.compute().clone();
    let unconfigured_launch = packet_on_subchannel(1, 0x01b0 / 4, 0x41);
    assert!(matches!(
        dispatch_first(&mut channel, &unconfigured_launch),
        Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
            method_name: "LAUNCH_DMA",
            reason: "launch requires a complete destination address",
            ..
        })
    ));
    assert_eq!(channel.compute(), &empty_before);

    let address = incrementing_packet_on_subchannel(1, 0x0188 / 4, &[0, 0x082b_30c0]);
    dispatch_first(&mut channel, &address).unwrap();
    let dimensions = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[0x40, 1]);
    dispatch_first(&mut channel, &dimensions).unwrap();

    let data = [0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 3, 0, 2, 1, 3, 1];
    let mut launch_arguments = Vec::with_capacity(data.len() + 1);
    launch_arguments.push(0x41);
    launch_arguments.extend_from_slice(&data);
    let launch = increment_once_packet_on_subchannel(1, 0x01b0 / 4, &launch_arguments);
    let dispatch = dispatch_first(&mut channel, &launch).unwrap();

    assert_eq!(dispatch.methods().len(), 17);
    assert_eq!(dispatch.methods()[0].metadata().method_name(), "LAUNCH_DMA");

    for index in 0..data.len() {
        let method = dispatch.methods()[index + 1];
        assert_eq!(method.metadata().method_name(), "LOAD_INLINE_DATA");
    }
    let inline = channel.compute().inline_to_memory();
    assert_eq!(inline.address().unwrap().get(), 0x082b_30c0);
    assert_eq!(inline.line_length().value(), Some(&0x40));
    assert_eq!(inline.line_count().value(), Some(&1));
    assert_eq!(inline.launch().raw(), Some(0x41));
    assert_eq!(inline.last_data().value(), Some(&1));
    assert_eq!(inline.pending(), None);
    assert!(inline.address_upper().source().is_some());
    assert!(inline.address_lower().source().is_some());

    let short = packet_on_subchannel(1, 0x0180 / 4, 4);
    dispatch_first(&mut channel, &short).unwrap();
    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let excessive =
        increment_once_packet_on_subchannel(1, 0x01b0 / 4, &[0x41, 0x1122_3344, 0x5566_7788]);
    assert!(matches!(
        dispatch_first(&mut channel, &excessive),
        Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
            source,
            method_name: "LOAD_INLINE_DATA",
            ..
        }) if source.argument() == 0x5566_7788
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.compute(), &compute_before);

    let after_prefix = channel.compute().clone();
    let reserved = packet_on_subchannel(1, 0x01b0 / 4, 0x80);
    assert!(matches!(
        dispatch_first(&mut channel, &reserved),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_f37f,
            ..
        }) if source.argument() == 0x80
    ));
    assert_eq!(channel.compute(), &after_prefix);
}

#[test]
fn compute_shader_cache_invalidation_is_typed_ordered_and_atomic() {
    let mut channel = compute_channel();

    let address = incrementing_packet_on_subchannel(1, 0x0188 / 4, &[0, 0x082b_30c0]);
    dispatch_first(&mut channel, &address).unwrap();
    let dimensions = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[4, 1]);
    dispatch_first(&mut channel, &dimensions).unwrap();
    let upload = increment_once_packet_on_subchannel(1, 0x01b0 / 4, &[0x41, 0xdead_beef]);
    dispatch_first(&mut channel, &upload).unwrap();
    let compute_before = channel.compute().clone();
    assert_eq!(
        compute_before.inline_to_memory().last_data().value(),
        Some(&0xdead_beef)
    );

    let invalidations = non_incrementing_packet_on_subchannel(1, 0x1698 / 4, &[0x1000, 0x1011]);
    let dispatch = dispatch_first(&mut channel, &invalidations).unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(dispatch.compute_operations().len(), 2);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "INVALIDATE_SHADER_CACHES_NO_WFI"
    );
    for (index, expected) in [
        MaxwellComputeShaderCacheInvalidation::new(false, false, true),
        MaxwellComputeShaderCacheInvalidation::new(true, true, true),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            lower_maxwell_compute_synchronization(dispatch.compute_operations()[index], true),
            MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches: expected }
        );
    }
    let captured = match dispatch.compute_operations()[0] {
        MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi { caches, .. } => caches,
        _ => unreachable!(),
    };
    assert!(!captured.instruction());
    assert!(!captured.global_data());
    assert!(captured.constant());
    assert_eq!(channel.compute(), &compute_before);

    let frontend_before = channel.frontend();
    let invalid = non_incrementing_packet_on_subchannel(1, 0x1698 / 4, &[0x1000, 0x1002]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_1011,
            ..
        }) if source.argument() == 0x1002
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
}

#[test]
fn compute_shader_cache_invalidation_with_wfi_preserves_selectors_and_rejects_unsupported_flags() {
    let mut channel = compute_channel();
    let before = channel.compute().clone();
    for raw in [0, 1, 0x10, 0x1000, 0x1011] {
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x021c / 4, raw)).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "INVALIDATE_SHADER_CACHES"
        );
        let operation = dispatch.compute_operations()[0];
        assert_eq!(channel.compute(), &before);
        assert_eq!(operation.source().argument(), raw);
        let expected = MaxwellComputeShaderCacheInvalidation::new(
            raw & 1 != 0,
            raw & 0x10 != 0,
            raw & 0x1000 != 0,
        );
        for pending in [false, true] {
            assert_eq!(
                lower_maxwell_compute_synchronization(operation, pending),
                MaxwellComputeSynchronizationPlan::InvalidateShaderCaches {
                    caches: expected,
                    prior_work_pending: pending,
                }
            );
        }
    }
    for (raw, detail) in [
        (2, "shader cache lock invalidation is not implemented"),
        (4, "shader data cache flush is not implemented"),
    ] {
        assert!(
            matches!(dispatch_first(&mut channel, &packet_on_subchannel(1, 0x021c / 4, raw)),
            Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding { reason, .. }) if reason == detail)
        );
    }
    assert!(matches!(
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x021c / 4, 8)),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            defined_mask: 0x1017,
            ..
        })
    ));
    assert_eq!(channel.compute(), &before);
}

#[test]
fn compute_cwd_reference_counter_bank_is_typed_source_preserving_and_atomic() {
    let mut channel = compute_channel();

    let arguments = (0..MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT)
        .rev()
        .map(|index| (0x0380 << 8) | index as u32)
        .collect::<Vec<_>>();
    let counters = non_incrementing_packet_on_subchannel(1, 0x0248 / 4, arguments.as_slice());
    let dispatch = dispatch_first(&mut channel, &counters).unwrap();
    assert_eq!(
        dispatch.methods().len(),
        MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT
    );
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_CWD_REF_COUNTER"
    );

    let bank = channel.compute().cwd_reference_counters();
    assert_eq!(bank.entries().len(), MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT);
    for index in 0..MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT {
        let index = MaxwellComputeCwdRefCounterIndex::new(index as u8).unwrap();
        let register = bank.entry(index);
        assert_eq!(register.value().unwrap().get(), 0x0380);
        assert_eq!(register.raw(), Some((0x0380 << 8) | u32::from(index.get())));
        assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Programmed);
    }
    assert_eq!(MaxwellComputeCwdRefCounterIndex::new(64), None);

    let overwrite = packet_on_subchannel(1, 0x0248 / 4, 0xffff << 8);
    dispatch_first(&mut channel, &overwrite).unwrap();
    let bank = channel.compute().cwd_reference_counters();
    assert_eq!(
        bank.entry(MaxwellComputeCwdRefCounterIndex::new(0).unwrap())
            .value(),
        Some(&MaxwellComputeCwdRefCounterValue::new(0xffff))
    );
    assert_eq!(
        bank.entry(MaxwellComputeCwdRefCounterIndex::new(1).unwrap())
            .value(),
        Some(&MaxwellComputeCwdRefCounterValue::new(0x0380))
    );

    for argument in [0x0000_0040, 0x0100_0000] {
        let compute_before = channel.compute().clone();
        let invalid = packet_on_subchannel(1, 0x0248 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &invalid),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x00ff_ff3f,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let invalid_packet = non_incrementing_packet_on_subchannel(1, 0x0248 / 4, &[0x0012_343f, 0x40]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_packet),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.argument() == 0x40
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.compute(), &compute_before);
}

#[test]
fn compute_wait_for_idle_is_an_ordered_neutral_operation() {
    let mut channel = compute_channel();
    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let three_d_before = channel.three_d().clone();
    let two_d_before = channel.two_d().clone();

    let waits = non_incrementing_packet_on_subchannel(1, 0x0110 / 4, &[0, 0xfeed_beef]);
    let dispatch = dispatch_first(&mut channel, &waits).unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(dispatch.compute_operations().len(), 2);
    assert!(dispatch.operations().is_empty());
    for (index, value) in [0, 0xfeed_beef].into_iter().enumerate() {
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "WAIT_FOR_IDLE"
        );

        let operation = dispatch.compute_operations()[index];
        assert!(matches!(
            operation,
            MaxwellComputeOperationTrigger::WaitForIdle {
                value: actual,
                source,
            } if actual == value && source.argument() == value
        ));
        assert_eq!(channel.compute(), &compute_before);
        assert_eq!(
            lower_maxwell_compute_synchronization(operation, false),
            MaxwellComputeSynchronizationPlan::WaitForIdle {
                prior_work_pending: false,
            }
        );
        assert_eq!(
            lower_maxwell_compute_synchronization(operation, true),
            MaxwellComputeSynchronizationPlan::WaitForIdle {
                prior_work_pending: true,
            }
        );
    }
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
    assert_eq!(channel.three_d(), &three_d_before);
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn render_target_discard_is_a_validated_non_destructive_hint() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    // Already-bound targets must not be unbound, cleared, or invalidated.
    program_three_d(&mut channel, 0x0800, 5);
    program_three_d(&mut channel, 0x0804, 0x0511_0000);
    program_three_d(&mut channel, 0x0fe0, 5);
    program_three_d(&mut channel, 0x0fe4, 0x0800_0000);
    let before = channel.clone();
    let draw = channel.three_d().draw_state_identity();
    let shaders = channel.three_d().shader_state_identity();
    let resources = channel
        .three_d()
        .resource_state_identity(MaxwellThreeDResourceRole::DepthStencilTarget);
    for color in 0..8 {
        for depth_stencil in 0..2 {
            let raw = (color << 4) | depth_stencil;
            let dispatch = dispatch_method(&mut channel, 0x0f78 / 4, raw).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            let method = dispatch.methods()[0];
            assert_eq!(method.metadata().method_name(), "DISCARD_RENDER_TARGET");
            assert_eq!(method.method().source().argument(), raw);
            assert_eq!(method.method().source().method(), GpuMethodId(0x0f78));
            assert_eq!(channel, before);
            assert!(draw.matches(channel.three_d()));
            assert!(shaders.matches_code(channel.three_d()));
            assert!(resources.matches(channel.three_d()));
        }
    }
    for raw in [2, 4, 8, 0x80, 0x100, u32::MAX] {
        assert!(matches!(
            dispatch_method(&mut channel, 0x0f78 / 4, raw),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                defined_mask: 0x71,
                ..
            })
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn tiled_cache_barrier_preserves_source_and_draw_state() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let before = channel.clone();
    let dispatch = dispatch_method(&mut channel, 0x0f7c / 4, 0).unwrap();
    assert!(dispatch.operations().is_empty());
    assert_eq!(dispatch.synchronization_operations().len(), 1);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "TILED_CACHE_BARRIER"
    );
    let operation = &dispatch.synchronization_operations()[0];
    assert!(matches!(operation.trigger(),
        MaxwellThreeDSynchronizationTrigger::TiledCacheBarrier { source }
            if source.method() == GpuMethodId(0x0f7c) && source.argument() == 0));
    assert_eq!(operation.state(), before.three_d());
    for pending in [false, true] {
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, pending),
            Ok(MaxwellThreeDSynchronizationPlan::TiledCacheBarrier),
        );
    }
    assert_eq!(channel, before);
    for raw in [1, 2, 0x8000_0000, u32::MAX] {
        assert!(matches!(
            dispatch_method(&mut channel, 0x0f7c / 4, raw),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                defined_mask: 0,
                ..
            })
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn pixel_shader_barriers_preserve_source_and_do_not_mutate_draw_state() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let before = channel.clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x0de0 / 4, &[0, 1]);
    let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
    assert!(dispatch.operations().is_empty());
    assert_eq!(dispatch.synchronization_operations().len(), 2);
    for (index, operation) in dispatch.synchronization_operations().iter().enumerate() {
        let expected = index != 0;
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "PIXEL_SHADER_BARRIER"
        );
        assert!(matches!(operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::PixelShaderBarrier { system_memory_barrier, source }
            if system_memory_barrier == expected && source.argument() == index as u32
                && source.method() == GpuMethodId(0x0de0)));
        assert_eq!(operation.state(), before.three_d());
        for pending in [false, true] {
            assert_eq!(
                lower_maxwell_three_d_synchronization(operation, None, pending),
                Ok(MaxwellThreeDSynchronizationPlan::PixelShaderBarrier {
                    system_memory_barrier: expected
                })
            );
        }
    }
    assert_eq!(channel, before);
    for raw in [2, 3, 0x8000_0000, u32::MAX] {
        assert!(
            matches!(dispatch_first(&mut channel, &packet(0x0de0 / 4, raw)),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { defined_mask: 1, source, .. })
                if source.argument() == raw)
        );
        assert_eq!(channel, before);
    }
}

#[test]
fn three_d_wait_for_idle_is_an_ordered_neutral_operation() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let frontend_before = channel.frontend();
    let compute_before = channel.compute().clone();
    let three_d_before = channel.three_d().clone();
    let two_d_before = channel.two_d().clone();

    let waits = non_incrementing_packet_on_subchannel(0, 0x0110 / 4, &[0, 0xfeed_beef]);
    let dispatch = dispatch_first(&mut channel, &waits).unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(dispatch.synchronization_operations().len(), 2);
    assert!(dispatch.operations().is_empty());
    for (index, value) in [0, 0xfeed_beef].into_iter().enumerate() {
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "WAIT_FOR_IDLE"
        );

        let operation = &dispatch.synchronization_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::WaitForIdle {
                value: actual,
                source,
            } if actual == value && source.argument() == value
        ));
        assert_eq!(operation.state(), &three_d_before);
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, false),
            Ok(MaxwellThreeDSynchronizationPlan::WaitForIdle {
                prior_work_pending: false,
            })
        );
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, true),
            Ok(MaxwellThreeDSynchronizationPlan::WaitForIdle {
                prior_work_pending: true,
            })
        );
    }
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.compute(), &compute_before);
    assert_eq!(channel.three_d(), &three_d_before);
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn decompress_surface_is_ordered_for_reset_or_programmed_uncompressed_targets() {
    let mut uncompressed = channel();
    bind_three_d(&mut uncompressed);
    program_three_d(&mut uncompressed, 0x19e0 + 5 * 4, 0);

    let decoded = packet(0x02e0 / 4, (0x1234 << 4) | 5);
    let dispatch = dispatch_first(&mut uncompressed, &decoded).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "DECOMPRESS_SURFACE"
    );
    assert!(dispatch.operations().is_empty());
    let operation = &dispatch.synchronization_operations()[0];
    let request = match operation.trigger() {
        MaxwellThreeDSynchronizationTrigger::DecompressSurface { request, source } => {
            assert_eq!(source.method(), GpuMethodId(0x02e0));
            request
        }
        other => panic!("unexpected synchronization trigger: {other:?}"),
    };
    assert_eq!(request.color_target(), 5);
    assert_eq!(request.array_index(), 0x1234);
    assert_eq!(request.raw(), (0x1234 << 4) | 5);
    assert_eq!(
        lower_maxwell_three_d_synchronization(operation, None, true),
        Ok(
            MaxwellThreeDSynchronizationPlan::DecompressUncompressedSurface {
                request,
                prior_work_pending: true,
            }
        )
    );

    let mut compressed = channel();
    bind_three_d(&mut compressed);
    program_three_d(&mut compressed, 0x19e0, 1);
    let decoded = packet(0x02e0 / 4, 0);
    let dispatch = dispatch_first(&mut compressed, &decoded).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_synchronization(
            dispatch.synchronization_operations()[0],
            None,
            false
        ),
        Err(MaxwellThreeDSynchronizationError::CompressedSurfaceDecompressionUnavailable {
            request,
            ..
        }) if request.color_target() == 0 && request.array_index() == 0
    ));

    let mut reset = channel();
    bind_three_d(&mut reset);
    let dispatch = dispatch_first(&mut reset, &decoded).unwrap();
    let operation = &dispatch.synchronization_operations()[0];
    let request = match operation.trigger() {
        MaxwellThreeDSynchronizationTrigger::DecompressSurface { request, .. } => request,
        other => panic!("unexpected synchronization trigger: {other:?}"),
    };
    assert_eq!(
        lower_maxwell_three_d_synchronization(operation, None, false),
        Ok(
            MaxwellThreeDSynchronizationPlan::DecompressUncompressedSurface {
                request,
                prior_work_pending: false,
            }
        )
    );
}

#[test]
fn decompress_surface_rejects_reserved_bits_atomically() {
    let mut channel = three_d_channel();

    for raw in [8, 1 << 20, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x02e0 / 4, raw);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x000f_fff7,
                ..
            }) if source.argument() == raw
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn three_d_flush_and_syncpoint_increment_are_ordered_completion_operations() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let three_d_before = channel.three_d().clone();

    let flushes = non_incrementing_packet_on_subchannel(0, 0x1144 / 4, &[0, 1]);
    let flush_dispatch = dispatch_first(&mut channel, &flushes).unwrap();
    assert_eq!(flush_dispatch.methods().len(), 2);
    assert_eq!(flush_dispatch.synchronization_operations().len(), 2);
    assert!(flush_dispatch.operations().is_empty());
    for (index, expected) in [false, true].into_iter().enumerate() {
        assert_eq!(
            flush_dispatch.methods()[index].metadata().method_name(),
            "FLUSH_PENDING_WRITES"
        );
        let operation = &flush_dispatch.synchronization_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::FlushPendingWrites { request, source }
                if request.sm_does_global_store() == expected
                    && source.argument() == u32::from(expected)
                    && source.method() == GpuMethodId(0x1144)
        ));
        assert_eq!(operation.state(), &three_d_before);
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, false),
            Ok(MaxwellThreeDSynchronizationPlan::FlushPendingWrites {
                request: MaxwellThreeDFlushPendingWrites::new(expected),
            })
        );
    }

    let increments =
        non_incrementing_packet_on_subchannel(0, 0x02c8 / 4, &[1, (1 << 20) | (1 << 16) | 1]);
    let increment_dispatch = dispatch_first(&mut channel, &increments).unwrap();
    assert_eq!(increment_dispatch.synchronization_operations().len(), 2);
    let stream_out = &increment_dispatch.synchronization_operations()[0];
    assert!(matches!(
        stream_out.trigger(),
        MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, source }
            if request.syncpoint() == GuestSyncpointId::new(1)
                && !request.clean_l2()
                && request.condition() == MaxwellThreeDSyncpointCondition::StreamOutWritesDone
                && source.method() == GpuMethodId(0x02c8)
    ));
    let rop = &increment_dispatch.synchronization_operations()[1];
    assert!(matches!(
        rop.trigger(),
        MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, .. }
            if request.syncpoint() == GuestSyncpointId::new(1)
                && request.clean_l2()
                && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
    ));

    let owner = TimelineOwnerId::new(9);
    let mut timeline = GuestTimeline::new(
        GuestSyncpointId::new(1),
        TimelineInstanceId::new(4),
        owner,
        GuestSyncpointValue::new(0),
    );
    let reservation = timeline.reserve(owner, 1).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, Some(&reservation), false),
        Ok(MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
            request,
            completion,
        }) if request.clean_l2()
            && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
            && completion == reservation.point()
    ));
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, None, false),
        Err(MaxwellThreeDSynchronizationError::MissingCompletionReservation {
            requested,
            ..
        }) if requested == GuestSyncpointId::new(1)
    ));

    let mut wrong_timeline = GuestTimeline::new(
        GuestSyncpointId::new(2),
        TimelineInstanceId::new(5),
        owner,
        GuestSyncpointValue::new(0),
    );
    let wrong_reservation = wrong_timeline.reserve(owner, 1).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_synchronization(rop, Some(&wrong_reservation), false),
        Err(MaxwellThreeDSynchronizationError::WrongCompletionSyncpoint {
            requested,
            reserved,
            ..
        }) if requested == GuestSyncpointId::new(1)
            && reserved == GuestSyncpointId::new(2)
    ));
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn tiled_cache_flush_decodes_allocated_modes_and_lowers_only_verified_flush() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let three_d_before = channel.three_d().clone();

    let flushes = non_incrementing_packet_on_subchannel(0, 0x0f80 / 4, &[0, 1]);
    let dispatch = dispatch_first(&mut channel, &flushes).unwrap();
    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(dispatch.synchronization_operations().len(), 2);
    assert!(dispatch.operations().is_empty());
    assert_eq!(channel.three_d(), &three_d_before);

    for (index, mode) in [
        MaxwellThreeDTiledCacheFlushMode::Flush,
        MaxwellThreeDTiledCacheFlushMode::Alternate,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "TILED_CACHE_FLUSH"
        );
        let operation = &dispatch.synchronization_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::TiledCacheFlush {
                mode: actual,
                source,
            } if actual == mode
                && source.argument() == mode.raw()
                && source.method() == GpuMethodId(0x0f80)
        ));
        assert_eq!(operation.state(), &three_d_before);
    }

    assert_eq!(
        lower_maxwell_three_d_synchronization(dispatch.synchronization_operations()[0], None, true,),
        Ok(MaxwellThreeDSynchronizationPlan::TiledCacheFlush {
            mode: MaxwellThreeDTiledCacheFlushMode::Flush,
            maintenance: nixe_gpu::CacheMaintenanceOperation::FlushDirtyDeviceWrites,
        })
    );
    assert!(matches!(
        lower_maxwell_three_d_synchronization(
            dispatch.synchronization_operations()[1],
            None,
            true,
        ),
        Err(MaxwellThreeDSynchronizationError::UnsupportedTiledCacheFlushMode {
            source,
            mode: MaxwellThreeDTiledCacheFlushMode::Alternate,
        }) if source.method() == GpuMethodId(0x0f80) && source.argument() == 1
    ));

    let before_invalid = channel.clone();
    let invalid = packet(0x0f80 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.method() == GpuMethodId(0x0f80) && source.argument() == 2
    ));
    assert_eq!(channel, before_invalid);
}

#[test]
fn three_d_texture_cache_invalidation_family_preserves_target_and_scope_without_waiting() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let three_d_before = channel.three_d().clone();
    let tag = 0x002a_55aa;
    for (method, name, target, maintenance) in [
        (
            0x1288,
            "INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI",
            MaxwellThreeDTextureCacheTarget::Data,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureReadCaches,
        ),
        (
            0x1424,
            "INVALIDATE_SAMPLER_CACHE_NO_WFI",
            MaxwellThreeDTextureCacheTarget::Sampler,
            nixe_gpu::CacheMaintenanceOperation::InvalidateSamplerCaches,
        ),
        (
            0x1428,
            "INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI",
            MaxwellThreeDTextureCacheTarget::Header,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureHeaderCaches,
        ),
    ] {
        let invalidations =
            non_incrementing_packet_on_subchannel(0, method / 4, &[0, (tag << 4) | 1]);
        let dispatch = dispatch_first(&mut channel, &invalidations).unwrap();

        assert_eq!(dispatch.synchronization_operations().len(), 2);
        for (index, expected_lines) in [
            MaxwellThreeDTextureCacheLines::All,
            MaxwellThreeDTextureCacheLines::One,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dispatch.methods()[index].metadata().method_name(), name);
            let operation = &dispatch.synchronization_operations()[index];
            let expected_tag = if index == 0 { 0 } else { tag };
            assert!(matches!(
                operation.trigger(),
                MaxwellThreeDSynchronizationTrigger::InvalidateTextureCacheNoWfi {
                    request,
                    source,
                } if request.target() == target
                    && request.lines() == expected_lines
                    && request.tag() == expected_tag
                    && source.method() == GpuMethodId(method)
            ));
            assert_eq!(operation.state(), &three_d_before);
            assert_eq!(
                lower_maxwell_three_d_synchronization(operation, None, false),
                Ok(
                    MaxwellThreeDSynchronizationPlan::InvalidateTextureCacheNoWfi {
                        request: MaxwellThreeDTextureCacheInvalidation::new(
                            target,
                            expected_lines,
                            expected_tag,
                        ),
                        maintenance,
                    }
                )
            );
        }

        let before = channel.clone();
        let invalid = non_incrementing_packet_on_subchannel(0, method / 4, &[0, 1 << 1]);
        assert!(matches!(
            dispatch_first(&mut channel, &invalid),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x03ff_fff1,
                ..
            }) if source.argument() == 1 << 1
        ));
        assert_eq!(channel, before);
    }
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_waiting_texture_cache_invalidation_family_drains_prior_work() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let three_d_before = channel.three_d().clone();
    let tag = 0x0012_3456;
    for (method, name, target, maintenance) in [
        (
            0x1330,
            "INVALIDATE_SAMPLER_CACHE",
            MaxwellThreeDTextureCacheTarget::Sampler,
            nixe_gpu::CacheMaintenanceOperation::InvalidateSamplerCaches,
        ),
        (
            0x1334,
            "INVALIDATE_TEXTURE_HEADER_CACHE",
            MaxwellThreeDTextureCacheTarget::Header,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureHeaderCaches,
        ),
        (
            0x1338,
            "INVALIDATE_TEXTURE_DATA_CACHE",
            MaxwellThreeDTextureCacheTarget::Data,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureReadCaches,
        ),
        (
            0x0f74,
            "INVALIDATE_TEXTURE_DATA_CACHE_TILED",
            MaxwellThreeDTextureCacheTarget::Data,
            nixe_gpu::CacheMaintenanceOperation::InvalidateTextureReadCaches,
        ),
    ] {
        let invalidations =
            non_incrementing_packet_on_subchannel(0, method / 4, &[0, (tag << 4) | 1]);
        let dispatch = dispatch_first(&mut channel, &invalidations).unwrap();

        for (index, expected_lines) in [
            MaxwellThreeDTextureCacheLines::All,
            MaxwellThreeDTextureCacheLines::One,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dispatch.methods()[index].metadata().method_name(), name);
            let operation = &dispatch.synchronization_operations()[index];
            let expected_tag = if index == 0 { 0 } else { tag };
            assert!(matches!(
                operation.trigger(),
                MaxwellThreeDSynchronizationTrigger::InvalidateTextureCache {
                    request,
                    source,
                } if request.target() == target
                    && request.lines() == expected_lines
                    && request.tag() == expected_tag
                    && source.method() == GpuMethodId(method)
            ));
            assert_eq!(operation.state(), &three_d_before);
            assert_eq!(
                lower_maxwell_three_d_synchronization(operation, None, true),
                Ok(MaxwellThreeDSynchronizationPlan::InvalidateTextureCache {
                    request: MaxwellThreeDTextureCacheInvalidation::new(
                        target,
                        expected_lines,
                        expected_tag,
                    ),
                    maintenance,
                    prior_work_pending: true,
                })
            );
        }

        let before = channel.clone();
        let invalid = packet_on_subchannel(0, method / 4, 1 << 2);
        assert!(matches!(
            dispatch_first(&mut channel, &invalid),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                defined_mask: 0x03ff_fff1,
                ..
            })
        ));
        assert_eq!(channel, before);
    }
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_shader_cache_invalidation_covers_every_selector_combination_atomically() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let three_d_before = channel.three_d().clone();
    let arguments = [0, 1, 0x10, 0x11, 0x1000, 0x1001, 0x1010, 0x1011];
    let invalidations = non_incrementing_packet_on_subchannel(0, 0x0da4 / 4, &arguments);
    let dispatch = dispatch_first(&mut channel, &invalidations).unwrap();

    assert_eq!(dispatch.synchronization_operations().len(), arguments.len());
    for (index, argument) in arguments.into_iter().enumerate() {
        assert_eq!(
            dispatch.methods()[index].metadata().method_name(),
            "INVALIDATE_SHADER_CACHES_NO_WFI"
        );
        let expected = MaxwellShaderCacheInvalidation::new(
            argument & 1 != 0,
            argument & 0x10 != 0,
            argument & 0x1000 != 0,
        );
        let operation = &dispatch.synchronization_operations()[index];
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDSynchronizationTrigger::InvalidateShaderCachesNoWfi {
                caches,
                source,
            } if caches == expected
                && source.argument() == argument
                && source.method() == GpuMethodId(0x0da4)
        ));
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, false),
            Ok(
                MaxwellThreeDSynchronizationPlan::InvalidateShaderCachesNoWfi {
                    caches: expected,
                    maintenance: nixe_gpu::CacheMaintenanceOperation::InvalidateShaderCaches {
                        instruction: expected.instruction(),
                        global_data: expected.global_data(),
                        constant: expected.constant(),
                    },
                }
            )
        );
    }
    assert_eq!(channel.three_d(), &three_d_before);

    let before = channel.clone();
    let invalid = non_incrementing_packet_on_subchannel(0, 0x0da4 / 4, &[0x1011, 1 << 1]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_1011,
            ..
        }) if source.argument() == 1 << 1
    ));
    assert_eq!(channel, before);
}

#[test]
fn ordered_three_d_shader_cache_invalidation_preserves_controls_and_prior_work() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    let state_before = channel.three_d().clone();
    let dispatch = dispatch_method(&mut channel, 0x021c / 4, 0x1011).unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "INVALIDATE_SHADER_CACHES"
    );
    let operation = &dispatch.synchronization_operations()[0];
    let (request, source) = match operation.trigger() {
        MaxwellThreeDSynchronizationTrigger::InvalidateShaderCaches { request, source } => {
            (request, source)
        }
        other => panic!("unexpected trigger: {other:?}"),
    };
    assert!(request.caches().instruction());
    assert!(request.caches().global_data());
    assert!(request.caches().constant());
    assert!(!request.locks());
    assert!(!request.flush_data());
    assert_eq!(source.method(), GpuMethodId(0x021c));
    assert_eq!(source.argument(), 0x1011);
    assert_eq!(operation.state(), &state_before);
    assert_eq!(channel.three_d(), &state_before);

    for prior_work_pending in [false, true] {
        assert_eq!(
            lower_maxwell_three_d_synchronization(operation, None, prior_work_pending),
            Ok(MaxwellThreeDSynchronizationPlan::InvalidateShaderCaches {
                request,
                maintenance: nixe_gpu::CacheMaintenanceOperation::InvalidateShaderCaches {
                    instruction: true,
                    global_data: true,
                    constant: true,
                },
                prior_work_pending,
            })
        );
    }
}

#[test]
fn ordered_shader_cache_special_controls_fail_during_lowering_and_reserved_bits_are_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (1 << 1, "locks"),
        (1 << 2, "flush"),
        ((1 << 1) | (1 << 2), "locks"),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x021c / 4, argument).unwrap();
        let operation = &dispatch.synchronization_operations()[0];
        assert!(match (
            expected,
            lower_maxwell_three_d_synchronization(operation, None, true)
        ) {
            (
                "locks",
                Err(MaxwellThreeDSynchronizationError::UnsupportedShaderCacheLockInvalidation {
                    source,
                }),
            ) => source.argument() == argument,
            (
                "flush",
                Err(MaxwellThreeDSynchronizationError::UnsupportedShaderDataCacheFlush { source }),
            ) => source.argument() == argument,
            _ => false,
        });
    }

    for argument in [1 << 3, 1 << 5, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x021c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_1017,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn three_d_completion_reserved_bits_fail_without_register_state_writes() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);

    for (method, valid, invalid, mask) in [
        (0x1144, 0, 2, 0x0000_0001),
        (0x02c8, 0x0011_0001, 0x0011_1001, 0x0011_0fff),
    ] {
        let frontend_before = channel.frontend();
        let three_d_before = channel.three_d().clone();
        let packet = non_incrementing_packet_on_subchannel(0, method / 4, &[valid, invalid]);
        assert!(matches!(
            dispatch_first(&mut channel, &packet),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == invalid && defined_mask == mask
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn known_compute_class_distinguishes_missing_method_coverage() {
    let mut channel = compute_channel();
    let method = packet_on_subchannel(1, 0x100 / 4, 0);
    let error = dispatch_first(&mut channel, &method).unwrap_err();
    assert!(matches!(
        error,
        MaxwellEngineDispatchError::UnknownMethod {
            class_name: "MAXWELL_COMPUTE_B",
            ..
        }
    ));
}

#[test]
fn standalone_inline_upload_addresses_block_linear_rows_and_nonzero_origins() {
    for (block_height, width, height, x, y, line_length, line_count, expected) in [
        (
            1,
            64,
            16,
            0,
            0,
            64,
            16,
            vec![(0, 0), (4, 4), (16, 32), (32, 256), (64, 16), (512, 512)],
        ),
        (
            3,
            2048,
            64,
            0,
            63,
            2048,
            1,
            vec![(0, 3792), (16, 3824), (64, 7888)],
        ),
        (
            0,
            128,
            16,
            16,
            8,
            64,
            2,
            vec![(0, 1056), (16, 1280), (48, 1536), (64, 1072)],
        ),
    ] {
        let mut channel = inline_to_memory_channel();
        dispatch_first(
            &mut channel,
            &incrementing_packet_on_subchannel(
                2,
                0x180 / 4,
                &[line_length, line_count, 4, 0x1000, 0],
            ),
        )
        .unwrap();
        let geometry = incrementing_packet_on_subchannel(
            2,
            0x194 / 4,
            &[block_height << 4, width, height, 1, 0, x, y],
        );
        let registers = dispatch_first(&mut channel, &geometry).unwrap();
        assert_eq!(
            channel.inline_to_memory().origin_y().source(),
            Some(registers.methods()[6].method().source())
        );
        dispatch_first(&mut channel, &packet_on_subchannel(2, 0x1b0 / 4, 0x1000)).unwrap();
        let data: Vec<u32> = (0..line_length * line_count / 4).collect();
        let (writes, _, result) = super::bulk::transfer(
            &mut channel,
            &non_incrementing_packet_on_subchannel(2, 0x1b4 / 4, &data),
            false,
        );
        result.unwrap();
        for (linear, physical) in expected {
            assert_eq!(
                writes[linear / 4],
                (0x4_00001000 + physical as u64, (linear / 4) as u32)
            );
        }
        assert!(channel.inline_to_memory().pending().is_none());
    }
}

#[test]
fn standalone_inline_pitch_rows_use_pitch_and_ignore_block_geometry() {
    let mut channel = inline_to_memory_channel();
    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(2, 0x180 / 4, &[4, 2, 4, 0x1000, 8]),
    )
    .unwrap();
    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(2, 0x194 / 4, &[0xfff, 0, 0, 0, 7, 3, 9]),
    )
    .unwrap();
    dispatch_first(&mut channel, &packet_on_subchannel(2, 0x1b0 / 4, 0x11)).unwrap();
    let dispatch = dispatch_first(
        &mut channel,
        &non_incrementing_packet_on_subchannel(2, 0x1b4 / 4, &[11, 22]),
    )
    .unwrap();
    for (index, offset) in [0, 8].into_iter().enumerate() {
        assert!(
            matches!(dispatch.ordered_operations()[index], MaxwellEngineOperation::InlineToMemory(upload) if upload.offset() == offset)
        );
    }
    for (method, invalid) in [(0x194, 0x1000), (0x1a8, 0x200000), (0x1ac, 0x20000)] {
        assert!(
            dispatch_first(&mut channel, &packet_on_subchannel(2, method / 4, invalid)).is_err()
        );
    }
    // Invalid geometry must be rejected when a block-linear launch consumes it.
    assert!(dispatch_first(&mut channel, &packet_on_subchannel(2, 0x1b0 / 4, 0x1000)).is_err());
}

#[test]
fn explicit_srgb_surface_format_survives_legacy_srgb_write_state() {
    let mut address_space = resource_address_space();
    let allocation = crate::test_memory::allocate(address_space.memory().machine(), 0x4000);
    let backing = crate::test_memory::ranges(&allocation);
    let address = map_resource(&mut address_space, backing, 12, 0xfe)
        .offset()
        .get();
    let mut channel = three_d_channel();
    assert_eq!(
        channel.three_d().render_targets().srgb_write().value(),
        Some(&false)
    );
    for (method, value) in [
        (0x800, (address >> 32) as u32),
        (0x804, address as u32),
        (0x808, 16),
        (0x80c, 16),
        (0x814, 0x40),
        (0x818, 1),
        (0x81c, 0),
        (0x15d0, 0),
    ] {
        program_three_d(&mut channel, method, value);
    }
    for (guest_format, expected) in [
        (0xd6, nixe_gpu::ImageFormat::Rgba8Srgb),
        (0xd0, nixe_gpu::ImageFormat::Bgra8Srgb),
        (0xd5, nixe_gpu::ImageFormat::Rgba8Unorm),
    ] {
        program_three_d(&mut channel, 0x810, guest_format);
        for enabled in [false, true] {
            let _dispatch = dispatch_method(&mut channel, 0x15b8 / 4, u32::from(enabled)).unwrap();

            let resources = fixture_roles(
                channel.three_d(),
                address_space.memory(),
                &[MaxwellThreeDResourceRole::ColorTarget(0)],
                &mut crate::test_memory::access(address_space.memory().machine()),
            )
            .unwrap();
            let MaxwellThreeDResolvedResource::Image(image) = resources.resources().next().unwrap()
            else {
                panic!("expected attachment")
            };
            assert_eq!(image.description().format(), expected);
        }
    }
    assert!(dispatch_method(&mut channel, 0x15b8 / 4, 2).is_err());
}

#[test]
fn constant_buffer_import_uses_shader_extent_and_cache_tracks_extent_changes() {
    let mut space = resource_address_space();
    let allocation = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let mapping = map_resource(&mut space, crate::test_memory::ranges(&allocation), 91, 0);
    let address = mapping.offset().get();
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x2380, 0x10000),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let role = MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 };
    let mut cache = super::super::threed::MaxwellThreeDResolvedResourceCache::default();
    let first = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &[(role, MaxwellShaderBufferSize::Proven(0x40))],
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    for (iteration, end) in [0x40_u64, 0x80, 0x40].into_iter().enumerate() {
        let resolved = cache
            .resolve_with_buffer_sizes(
                channel.three_d(),
                space.memory(),
                &[role],
                &[(role, MaxwellShaderBufferSize::Proven(end))],
                8,
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap()
            .clone();
        assert_eq!(resolved.resources().count(), 1);
        let Some(MaxwellThreeDResolvedResource::Buffer(buffer)) = resolved.resources().next()
        else {
            panic!("expected one constant buffer")
        };
        assert_eq!(buffer.view().size(), end);
        if iteration == 0 {
            assert!(first.same_bindings(&resolved));
        }
    }
    assert!(
        cache
            .resolve_with_buffer_sizes(
                channel.three_d(),
                space.memory(),
                &[role],
                &[(role, MaxwellShaderBufferSize::Declared)],
                8,
                &mut crate::test_memory::access(space.memory().machine())
            )
            .is_err(),
        "an unbounded indexed access must retain the invalid declared domain"
    );
    assert!(
        cache
            .resolve_with_buffer_sizes(
                channel.three_d(),
                space.memory(),
                &[role],
                &[(role, MaxwellShaderBufferSize::Proven(0x1004))],
                8,
                &mut crate::test_memory::access(space.memory().machine())
            )
            .is_err(),
        "consumed unmapped bytes must still fail"
    );
    assert!(
        cache
            .resolve(
                channel.three_d(),
                space.memory(),
                &[role],
                8,
                &mut crate::test_memory::access(space.memory().machine())
            )
            .is_err(),
        "complete role inspection has no shader proof"
    );
}

#[test]
fn graphics_storage_descriptor_tracks_pointer_size_committed_writes_and_remapping() {
    let mut space = resource_address_space();
    let cb = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let data = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let cb_mapping = map_resource(&mut space, crate::test_memory::ranges(&cb), 92, 0);
    let data_mapping = map_resource(&mut space, crate::test_memory::ranges(&data), 93, 0);
    let descriptor = |address: u64, size: u32| {
        address
            .to_le_bytes()
            .into_iter()
            .chain(size.to_le_bytes())
            .chain([0; 4])
            .collect::<Vec<_>>()
    };
    crate::test_memory::write(&cb, 0x110, &descriptor(data_mapping.offset().get(), 128)).unwrap();
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x2380, 0x200),
        (0x2384, (cb_mapping.offset().get() >> 32) as u32),
        (0x2388, cb_mapping.offset().get() as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let role = MaxwellThreeDResourceRole::StorageBuffer {
        group: 0,
        slot: 0,
        byte_offset: 0x110,
    };
    let sizes = [(role, MaxwellShaderBufferSize::Declared)];
    let mut cache = super::super::threed::MaxwellThreeDResolvedResourceCache::default();
    let first = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    let repeated = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert!(first.same_bindings(&repeated));
    let buffer = |resources: &MaxwellThreeDResolvedResources| {
        assert_eq!(resources.resources().count(), 1);
        let Some(MaxwellThreeDResolvedResource::Buffer(buffer)) = resources.resources().next()
        else {
            panic!("expected one storage buffer")
        };
        (buffer.source().offset().get(), buffer.view().size())
    };
    assert_eq!(buffer(&first), (data_mapping.offset().get(), 128));
    // A different word on the descriptor page leaves the consumed bytes equal,
    // so the already prepared bindings remain reusable.
    crate::test_memory::write(&cb, 0x800, &[7]).unwrap();
    let refreshed = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert!(first.same_bindings(&refreshed));
    let clean = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert!(first.same_bindings(&clean));
    crate::test_memory::write(
        &cb,
        0x110,
        &descriptor(data_mapping.offset().get() + 64, 32),
    )
    .unwrap();
    let changed = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert_eq!(buffer(&changed), (data_mapping.offset().get() + 64, 32));
    assert_eq!(buffer(&first), (data_mapping.offset().get(), 128));
    crate::test_memory::write(&cb, 0x110, &descriptor(data_mapping.offset().get(), 128)).unwrap();
    let restored = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert_eq!(buffer(&restored), (data_mapping.offset().get(), 128));
    let bounded = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &[(role, MaxwellShaderBufferSize::Proven(4))],
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert_eq!(buffer(&bounded), (data_mapping.offset().get(), 4));
    crate::test_memory::write(&cb, 0x118, &0_u32.to_le_bytes()).unwrap();
    assert!(
        cache
            .resolve_with_buffer_sizes(
                channel.three_d(),
                space.memory(),
                &[role],
                &sizes,
                8,
                &mut crate::test_memory::access(space.memory().machine())
            )
            .is_err()
    );
    let before_remap = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &[(role, MaxwellShaderBufferSize::Proven(4))],
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    // Replacing the CB mapping must invalidate the pointer dependency even
    // when neither the old canonical bytes nor the storage target was changed.
    let replacement = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    crate::test_memory::write(
        &replacement,
        0x110,
        &descriptor(data_mapping.offset().get() + 128, 64),
    )
    .unwrap();
    space.unmap(cb_mapping.offset()).unwrap();
    space
        .reserve(1, 0x1000, 1, cb_mapping.offset().get())
        .unwrap();
    space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(94),
            backing: crate::test_memory::ranges(&replacement),
            backing_offset: 0,
            size: 0x1000,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: true,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: Some(cb_mapping.offset()),
        })
        .unwrap();
    let remapped_bounded = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &[(role, MaxwellShaderBufferSize::Proven(4))],
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert!(!before_remap.same_bindings(&remapped_bounded));
    assert_eq!(
        buffer(&remapped_bounded),
        (data_mapping.offset().get() + 128, 4)
    );
    let remapped = cache
        .resolve_with_buffer_sizes(
            channel.three_d(),
            space.memory(),
            &[role],
            &sizes,
            8,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap()
        .clone();
    assert_eq!(buffer(&remapped), (data_mapping.offset().get() + 128, 64));
}
