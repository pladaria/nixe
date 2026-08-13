use super::*;

#[test]
fn z_compression_selector_is_typed_depth_state_without_an_operation() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let color_before = channel.three_d().render_targets().color().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .compression()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDZCompressionMode::Disabled),
        (1, MaxwellThreeDZCompressionMode::Enabled),
    ] {
        let decoded = packet(0x19cc / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .compression();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_Z_COMPRESSION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::DepthCompression {
                    value: expected,
                    source,
                }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(channel.three_d().render_targets().color(), &color_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    assert!(resources.resources().is_empty());
}

#[test]
fn invalid_z_compression_values_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for argument in [2, 3, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x19cc / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_Z_COMPRESSION",
                reason: "reserved bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn enabled_z_compression_without_a_depth_target_does_not_block_draw_preflight() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let compression = packet(0x19cc / 4, 1);
    let compression_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &compression.packets()[0],
    )
    .unwrap();
    program_three_d(&mut channel, 0x121c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: compression_dispatch.methods()[0].method().source(),
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn color_compression_selectors_are_typed_and_isolated_per_target() {
    for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let depth_before = channel.three_d().render_targets().depth_stencil().clone();
        let two_d_before = channel.two_d().clone();

        for (argument, expected) in [
            (0, MaxwellThreeDColorCompressionMode::Disabled),
            (1, MaxwellThreeDColorCompressionMode::Enabled),
        ] {
            let method = 0x19e0 + u32::from(target) * 4;
            let decoded = packet(method / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let targets = channel.three_d().render_targets().color();
            let register = targets[target as usize].compression();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_COLOR_COMPRESSION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                    MaxwellThreeDRenderTargetWrite::ColorCompression {
                        target,
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            for (other, state) in targets.iter().enumerate() {
                if other != target as usize {
                    assert_eq!(
                        state.compression().origin(),
                        MaxwellThreeDRegisterOrigin::Unset
                    );
                }
            }
            assert_eq!(
                channel.three_d().render_targets().depth_stencil(),
                &depth_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        assert!(resources.resources().is_empty());
    }
}

#[test]
fn invalid_color_compression_values_are_rejected_atomically() {
    for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
        for argument in [2, 3, u32::MAX] {
            let mut channel = channel();
            bind_three_d(&mut channel);
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let method = 0x19e0 + u32::from(target) * 4;
            let decoded = packet(method / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_COLOR_COMPRESSION",
                    reason: "reserved bits are set",
                }) if source.argument() == argument && source.method() == GpuMethodId(method)
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }
}

#[test]
fn compressed_color_full_clear_materializes_without_importing_guest_bytes() {
    let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let mapping = map_resource(
        &mut address_space,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        41,
        0xfe,
    );
    let address = mapping.offset().get();
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        (0x19e0, 1),
        (0x121c, 1),
        (0x12e4, 0),
        (0x135c, 0),
        (0x0d6c, (32 << 16) | 0),
        (0x0d70, (16 << 16) | 0),
        (0x0d80, 0x3f80_0000),
        (0x0d84, 0x3f00_0000),
        (0x0d88, 0),
        (0x0d8c, 0x3f80_0000),
        (0x10f8, 0x10),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    assert!(
        resources
            .resources()
            .iter()
            .any(|resource| { resource.role() == MaxwellThreeDResourceRole::ColorTarget(0) })
    );
    let mut cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();

    let draw_source = channel.three_d().render_targets().color()[0]
        .compression()
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: draw_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::CompressedColorImportRequired { target: 0 })
    ));
    assert_eq!(cache, cache_before);

    let clear = packet(0x19d0 / 4, 0x3c);
    let clear_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &clear_dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::CompressedColorImportRequired { target: 0 })
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x10f8, 0);
    let full_clear = packet(0x19d0 / 4, 0x3c);
    let full_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &full_clear.packets()[0],
    )
    .unwrap();
    let full = &full_dispatch.operations()[0];
    let full_resources = resolve_maxwell_three_d_resources(full.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        full.state(),
        &full_resources,
        full.trigger(),
        None,
        FrontendSubmissionId::new(12),
        Vec::new(),
        &lowering_capabilities(BackendFeatures::CLEAR),
        &cache,
    )
    .unwrap();
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Image { view: None, .. }
    )));
    assert!(matches!(
        plan.submission().operations()[0].command(),
        GpuCommand::Clear(nixe_gpu::ClearOperation::Image {
            value: nixe_gpu::ClearValue::Color(_),
            ..
        })
    ));
    plan.commit_cache(&mut cache).unwrap();

    let partial_after_materialization = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        None,
        FrontendSubmissionId::new(13),
        Vec::new(),
        &lowering_capabilities(BackendFeatures::CLEAR),
        &cache,
    )
    .unwrap();
    assert!(
        partial_after_materialization
            .resource_creations()
            .is_empty()
    );
}

#[test]
fn color_compression_does_not_block_a_different_clear_target() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x19e4, 1);
    program_three_d(&mut channel, 0x121c, 1);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &MaxwellThreeDLoweringCache::default(),
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn color_target_selection_retains_all_fields_for_counts_zero_through_eight() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let targets = [7, 0, 6, 1, 5, 2, 4, 3];

    for count in 0..=8 {
        let argument = color_target_selection_raw(count, targets);
        let decoded = packet(0x121c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().render_targets().color_target_selection();
        let selection = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CT_SELECT"
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(selection.target_count(), count);
        assert_eq!(selection.targets(), targets);
        assert_eq!(selection.active_targets(), &targets[..usize::from(count)]);
        assert_eq!(selection.raw(), argument);
    }
}

#[test]
fn render_target_layer_is_typed_source_preserving_and_conditionally_dependent() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let neutral_dependencies = channel.three_d().pipeline_dependencies(&[0]);

    for (argument, layer, control, affects_draw_layering) in [
        (0, 0, MaxwellThreeDRenderTargetLayerControl::Fixed, false),
        (
            0xffff,
            u16::MAX,
            MaxwellThreeDRenderTargetLayerControl::Fixed,
            true,
        ),
        (
            0x0001_0000,
            0,
            MaxwellThreeDRenderTargetLayerControl::GeometryShader,
            true,
        ),
        (
            0x0001_ffff,
            u16::MAX,
            MaxwellThreeDRenderTargetLayerControl::GeometryShader,
            true,
        ),
    ] {
        let decoded = packet(0x15cc / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDRenderTargetLayer::new(layer, control);
        let register = channel.three_d().render_targets().render_target_layer();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RT_LAYER"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::RenderTargetLayer { value, source }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(value.layer(), layer);
        assert_eq!(value.control(), control);
        assert_eq!(value.raw(), argument);
        assert_eq!(value.affects_draw_layering(), affects_draw_layering);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[0]) == neutral_dependencies,
            !affects_draw_layering
        );
    }
}

#[test]
fn render_target_layer_reserved_bits_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x15cc, 0);

    for argument in [0x0002_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x15cc / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_RT_LAYER",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x15cc / 4, &[1, 0x10]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_ANTI_ALIAS",
            ..
        }) if source.method() == GpuMethodId(0x15d0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn malformed_color_target_selection_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let valid = color_target_selection_raw(2, [1, 0, 7, 6, 5, 4, 3, 2]);
    program_three_d(&mut channel, 0x121c, valid);

    for argument in [9, 15, 0x1000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x121c / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CT_SELECT",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x121c / 4, &[1, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1220)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn draw_rejects_missing_disabled_incomplete_and_duplicate_color_routes() {
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let address_space = resource_address_space();
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (argument, expected) in [
        (
            color_target_selection_raw(1, [3, 0, 0, 0, 0, 0, 0, 0]),
            MaxwellThreeDLoweringError::ColorTargetRouteUnprogrammed { slot: 0, target: 3 },
        ),
        (
            color_target_selection_raw(2, [3, 3, 0, 0, 0, 0, 0, 0]),
            MaxwellThreeDLoweringError::DuplicateColorTargetRoute { target: 3 },
        ),
    ] {
        program_three_d(&mut channel, 0x121c, argument);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let source = channel
            .three_d()
            .render_targets()
            .color_target_selection()
            .source()
            .unwrap();
        let result = preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("invalid color-target route unexpectedly lowered"),
        };
        assert_eq!(error.to_string(), expected.to_string());
    }

    program_three_d(&mut channel, 0x0810, 0);
    program_three_d(&mut channel, 0x121c, 1);
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let source = channel
        .three_d()
        .render_targets()
        .color_target_selection()
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ColorTargetRouteDisabled { slot: 0, target: 0 })
    ));

    program_three_d(&mut channel, 0x0810, 0xd5);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ColorTargetRouteIncomplete { slot: 0, target: 0 })
    ));
}

#[test]
fn three_d_register_write_uses_private_candidate_then_atomic_commit() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let decoded = packet(0x1518 / 4, 0x3fc0_0000);
    let before = channel.three_d().clone();

    let prepared = preflight_maxwell_engine_packet(
        &channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    assert_eq!(
        channel.three_d().raster().point_size().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );
    let after = prepared.three_d_after().clone();
    assert_eq!(
        after.raster().point_size().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(after.raster().point_size().raw(), Some(0x3fc0_0000));
    assert_eq!(
        after.raster().point_size().value().copied(),
        Some(MaxwellThreeDPointSize::from_bits(0x3fc0_0000))
    );
    assert_eq!(
        prepared.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::PointSize {
            value: MaxwellThreeDPointSize::from_bits(0x3fc0_0000),
            source: prepared.methods()[0].method().source(),
        })
    );

    commit_maxwell_engine_packet(&mut channel, &prepared).unwrap();
    assert_ne!(channel.three_d(), &before);
    assert_eq!(channel.three_d(), &after);
}

#[test]
fn enumerated_register_values_are_checked_before_state_changes() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let valid = packet(0x0d7c / 4, 1);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &valid.packets()[0],
    )
    .unwrap();
    assert_eq!(
        channel.three_d().viewport().z_clip_range().value().copied(),
        Some(MaxwellThreeDViewportZClipRange::ZeroToPositiveW)
    );

    let before = channel.three_d().clone();
    let invalid = packet(0x0d7c / 4, 2);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            defined_mask: 1,
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn render_target_state_distinguishes_unset_disabled_ready_and_profile_unavailable() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Unprogrammed
    );

    let disabled = packet(0x0810 / 4, 0);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &disabled.packets()[0],
    )
    .unwrap();
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Disabled
    );

    let complete = incrementing_packet(0x0800 / 4, &[0, 0x0080_0000, 1280, 720, 0xd5, 0, 1, 0, 0]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &complete.packets()[0],
    )
    .unwrap();
    let target = &channel.three_d().render_targets().color()[0];
    assert_eq!(
        target.readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );
    assert_eq!(
        target.readiness(false),
        MaxwellThreeDAttachmentReadiness::ProfileUnavailable
    );
    assert_eq!(target.address_lower().value(), Some(&0x0080_0000));
    assert_eq!(target.format().raw(), Some(0xd5));
}

#[test]
fn render_target_encodings_and_cross_register_contradictions_reject_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    let malformed_layout = packet(0x0814 / 4, 0x1001);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &malformed_layout.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);

    let three_dimensional = packet(0x0814 / 4, 0x1_0000);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &three_dimensional.packets()[0],
    )
    .unwrap();
    let before_layer = channel.three_d().clone();
    let nonzero_layer = packet(0x0820 / 4, 1);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &nonzero_layer.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::ContradictoryState { .. })
    ));
    assert_eq!(channel.three_d(), &before_layer);
}

#[test]
fn clear_and_fixed_function_state_preserve_typed_values_and_sources() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    for (method, argument) in [
        (0x0d6c, (640_u32 << 16) | 10),
        (0x0d70, (480_u32 << 16) | 20),
        (0x0d80, 0x3f80_0000),
        (0x0d90, 0x3f00_0000),
        (0x0da0, 0x7f),
        (0x19d0, 0x3c),
        (0x12cc, 1),
        (0x130c, 0x203),
        (0x1380, 1),
        (0x1384, 0x1e00),
        (0x1390, 0x207),
        (0x1598, 0x1e01),
        (0x15a4, 0x201),
        (0x133c, 1),
        (0x1340, 0x8006),
        (0x1344, 0x4302),
        (0x1e00, 1),
        (0x1e04, 0x8006),
        (0x1e08, 0x4302),
        (0x1918, 1),
        (0x191c, 0x901),
        (0x1920, 0x405),
        (0x1a00, 0x1101),
    ] {
        let decoded = packet(method / 4, argument);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    let state = channel.three_d();
    let clear = state.render_targets().clear();
    assert_eq!(clear.horizontal().value().unwrap().min, 10);
    assert_eq!(clear.horizontal().value().unwrap().max, 640);
    assert_eq!(clear.last_surface().value().unwrap().color_mask(), 0xf);
    assert_eq!(
        clear.last_surface().source().unwrap().method(),
        GpuMethodId(0x19d0)
    );
    assert_eq!(
        state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::DepthCompare)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::Compare(
            MaxwellThreeDCompareOp::LessEqual
        ))
    );
    assert_eq!(
        state.fixed_function().color_mask()[0].value(),
        Some(&MaxwellThreeDColorMask {
            red: true,
            green: false,
            blue: true,
            alpha: true,
        })
    );
    assert_eq!(
        state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::FrontStencilFail)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::StencilOp(
            MaxwellThreeDStencilOp::Keep
        ))
    );
    assert_eq!(
        state.fixed_function().per_target_blend()[0][1].value(),
        Some(&MaxwellThreeDFixedFunctionValue::BlendOp(
            MaxwellThreeDBlendOp::Add
        ))
    );
}

#[test]
fn clear_surface_control_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();

    for combination in 0_u32..16 {
        let argument = (combination & 1)
            | ((combination & 2) << 3)
            | ((combination & 4) << 6)
            | ((combination & 8) << 9);
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        let decoded = packet(0x10f8 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().render_targets().clear().surface_control();
        let value = register.value().copied().unwrap();

        assert_eq!(method.metadata().method_name(), "SET_CLEAR_SURFACE_CONTROL");
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::ClearSurfaceControl { value, source }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(value.raw(), argument);
        assert_eq!(value.respect_stencil_mask(), combination & 1 != 0);
        assert_eq!(value.use_clear_rect(), combination & 2 != 0);
        assert_eq!(value.use_scissor_zero(), combination & 4 != 0);
        assert_eq!(value.use_viewport_clip_zero(), combination & 8 != 0);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn clear_surface_control_reserved_bits_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x10f8, 0x1111);

    for argument in [2, 0x20, 0x200, 0x2000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x10f8 / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CLEAR_SURFACE_CONTROL",
                reason: "reserved control bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10f8 / 4, &[0, 0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1100)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn unsupported_clear_surface_modifiers_fail_only_when_consumed() {
    let preflight = |control, surface| {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10f8, control);
        let clear = packet(0x19d0 / 4, surface);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space())
                .unwrap();
        match preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &MaxwellThreeDLoweringCache::default(),
        ) {
            Ok(_) => panic!("unsupported clear modifier unexpectedly lowered"),
            Err(error) => error,
        }
    };

    assert!(matches!(
        preflight(0x0100, 0x3c),
        MaxwellThreeDLoweringError::UnsupportedClearScissorSemantics
    ));
    assert!(matches!(
        preflight(0x1000, 0x3c),
        MaxwellThreeDLoweringError::UnsupportedClearViewportClipSemantics
    ));
    assert!(matches!(
        preflight(0x0001, 0x03),
        MaxwellThreeDLoweringError::UnsupportedClearStencilMaskSemantics
    ));
    assert!(!matches!(
        preflight(0x0001, 0x3c),
        MaxwellThreeDLoweringError::UnsupportedClearStencilMaskSemantics
    ));
}

#[test]
fn malformed_scissor_suffix_discards_the_whole_candidate() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let decoded = incrementing_packet(0x0e00 / 4, &[1, (100 << 16) | 5, (10 << 16) | 20]);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn window_clip_type_is_typed_and_rejects_unknown_values_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (argument, expected) in [
        (0, MaxwellThreeDWindowClipType::Inclusive),
        (1, MaxwellThreeDWindowClipType::Exclusive),
        (2, MaxwellThreeDWindowClipType::ClipAll),
    ] {
        let decoded = packet(0x1950 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipType);

        assert_eq!(expected.raw(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::WindowClipType(expected))
        );
        assert_eq!(register.source(), Some(source));
    }

    for argument in [3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1950 / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_WINDOW_CLIP_TYPE",
                reason: "unknown window clip type",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn window_clip_packet_programs_all_eight_typed_source_preserving_pairs() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x1950, 0);
    program_three_d(&mut channel, 0x194c, 0);
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipType)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::WindowClipType(
            MaxwellThreeDWindowClipType::Inclusive
        ))
    );
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let arguments = std::array::from_fn::<_, 16, _>(|word| {
        let region = word / 2;
        let minimum = (region * 10 + word % 2) as u32;
        let maximum = minimum + 100;
        (maximum << 16) | minimum
    });
    let decoded = incrementing_packet(0x0d00 / 4, &arguments);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 16);
    assert!(dispatch.operations().is_empty());
    for (region_index, region) in channel
        .three_d()
        .fixed_function()
        .window_clip()
        .iter()
        .enumerate()
    {
        for (vertical, register, word) in [
            (false, region.horizontal(), region_index * 2),
            (true, region.vertical(), region_index * 2 + 1),
        ] {
            let method = dispatch.methods()[word];
            let source = method.method().source();
            let expected = MaxwellThreeDRectangle {
                min: (region_index * 10 + usize::from(vertical)) as u16,
                max: (region_index * 10 + usize::from(vertical) + 100) as u16,
            };
            let method_name = if vertical {
                "SET_WINDOW_CLIP_VERTICAL"
            } else {
                "SET_WINDOW_CLIP_HORIZONTAL"
            };

            assert_eq!(method.metadata().method_name(), method_name);
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::WindowClipRectangle {
                        region: region_index as u8,
                        vertical,
                        value: expected,
                        source,
                    }
                ))
            );
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(arguments[word]));
            assert_eq!(register.value(), Some(&expected));
            assert_eq!(register.source(), Some(source));
        }
    }
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn malformed_window_clip_rectangle_discards_the_whole_packet() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let valid = incrementing_packet(0x0d00 / 4, &[0; 16]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &valid.packets()[0],
    )
    .unwrap();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let mut arguments = [0; 16];
    arguments[9] = (1 << 16) | 2;
    let malformed = incrementing_packet(0x0d00 / 4, &arguments);

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &malformed.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_WINDOW_CLIP_VERTICAL",
            reason: "rectangle minimum exceeds maximum",
        }) if source.method() == GpuMethodId(0x0d24)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn window_clip_dependencies_and_draw_error_follow_enable_while_clear_is_independent() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1950, 0);
    program_three_d(&mut channel, 0x194c, 0);
    let dependencies_disabled = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x0d00, (100 << 16) | 10);
    program_three_d(&mut channel, 0x1950, 1);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_disabled
    );

    program_three_d(&mut channel, 0x194c, 1);
    let dependencies_enabled = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(dependencies_enabled, dependencies_disabled);
    program_three_d(&mut channel, 0x0d04, (200 << 16) | 20);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_enabled
    );

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
    let source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable)
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedWindowClipSemantics)
    ));

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn clip_id_test_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();
    let window_clip_before = channel.three_d().fixed_function().window_clip().to_owned();
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDClipIdTestEnable::Disabled),
        (1, MaxwellThreeDClipIdTestEnable::Enabled),
    ] {
        let decoded = packet(0x197c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable);
        let value = MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(expected);

        assert_eq!(method.metadata().method_name(), "SET_CLIP_ID_TEST");
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::Register {
                    register: MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable,
                    value,
                    source,
                }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().fixed_function().window_clip(),
            &window_clip_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x197c / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_CLIP_ID_TEST",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x197c / 4, &[0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1980)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn clip_id_test_only_blocks_draw_when_enabled_and_never_blocks_clear() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x197c, 0);
    let dependencies_disabled = channel.three_d().pipeline_dependencies(&[]);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
    let disabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: disabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x197c, 1);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_disabled
    );
    let enabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: enabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedClipIdTestSemantics)
    ));

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn viewport_scale_offset_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (argument, expected) in [
        (0, MaxwellThreeDViewportScaleOffsetEnable::Disabled),
        (1, MaxwellThreeDViewportScaleOffsetEnable::Enabled),
    ] {
        let decoded = packet(0x192c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable);
        let value = MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(expected);

        assert_eq!(method.metadata().method_name(), "SET_VIEWPORT_SCALE_OFFSET");
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::Register {
                    register: MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable,
                    value,
                    source,
                }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x192c / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_VIEWPORT_SCALE_OFFSET",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x192c / 4, &[0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1930)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn viewport_scale_offset_dependencies_and_draw_validation_follow_enable_only() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x192c, 0);
    let dependencies_disabled = channel.three_d().pipeline_dependencies(&[]);

    program_three_d(&mut channel, 0x0a00, 1.0_f32.to_bits());
    program_three_d(&mut channel, 0x0a0c, 2.0_f32.to_bits());
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_disabled
    );

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();
    let disabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: disabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x192c, 1);
    let dependencies_enabled = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(dependencies_enabled, dependencies_disabled);
    for (method, value) in [
        (0x0a00, 3.0_f32),
        (0x0a04, -4.0_f32),
        (0x0a08, 0.5_f32),
        (0x0a0c, 2.0_f32),
        (0x0a10, 4.0_f32),
        (0x0a14, 0.5_f32),
    ] {
        program_three_d(&mut channel, method, value.to_bits());
    }
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_enabled
    );

    let enabled_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: enabled_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
}

#[test]
fn mme_ram_loads_capture_typed_programs_with_sources_and_auto_advance() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    let start_packet = incrementing_packet(0x011c / 4, &[5, 7]);
    let start_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &start_packet.packets()[0],
    )
    .unwrap();
    let start_pointer_source = start_dispatch.methods()[0].method().source();
    let start_source = start_dispatch.methods()[1].method().source();
    assert_eq!(
        start_dispatch.methods()[0].metadata().method_name(),
        "LOAD_MME_START_ADDRESS_RAM_POINTER"
    );
    assert_eq!(
        start_dispatch.methods()[1].metadata().method_name(),
        "LOAD_MME_START_ADDRESS_RAM"
    );
    assert_eq!(
        start_dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Mme(
            MaxwellThreeDMmeStateWrite::StartAddressPointer {
                value: MaxwellThreeDMmeRamAddress::new(5),
                source: start_pointer_source,
            }
        ))
    );
    assert_eq!(
        start_dispatch.methods()[1].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Mme(
            MaxwellThreeDMmeStateWrite::StartAddress {
                index: MaxwellThreeDMmeRamAddress::new(5),
                address: MaxwellThreeDMmeRamAddress::new(7),
                source: start_source,
            }
        ))
    );

    let instruction_words = [0x0000_0301, 0x0000_0211, 0x0588_0021];
    let instruction_packet = increment_once_packet(
        0x0114 / 4,
        &[
            7,
            instruction_words[0],
            instruction_words[1],
            instruction_words[2],
        ],
    );
    let instruction_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &instruction_packet.packets()[0],
    )
    .unwrap();
    let mme = channel.three_d().mme();

    assert!(start_dispatch.operations().is_empty());
    assert!(instruction_dispatch.operations().is_empty());
    assert_eq!(mme.instruction_pointer().raw(), Some(7));
    assert_eq!(
        mme.instruction_pointer().value(),
        Some(&MaxwellThreeDMmeRamAddress::new(7))
    );
    assert_eq!(
        mme.next_instruction_address(),
        Some(MaxwellThreeDMmeRamAddress::new(10))
    );
    assert_eq!(mme.instruction_count(), 3);
    for (word, expected) in instruction_words.into_iter().enumerate() {
        let address = MaxwellThreeDMmeRamAddress::new(7 + word as u32);
        let register = mme.instruction(address).unwrap();
        let source = instruction_dispatch.methods()[word + 1].method().source();
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(expected));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDMmeInstruction::new(expected))
        );
        assert_eq!(register.source(), Some(source));
    }
    assert_eq!(mme.start_address_pointer().raw(), Some(5));
    assert_eq!(
        mme.next_start_address_index(),
        Some(MaxwellThreeDMmeRamAddress::new(6))
    );
    let start = mme
        .start_address(MaxwellThreeDMmeRamAddress::new(5))
        .unwrap();
    assert_eq!(start.raw(), Some(7));
    assert_eq!(start.value(), Some(&MaxwellThreeDMmeRamAddress::new(7)));
    assert_eq!(start.source(), Some(start_source));
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn mme_ram_load_failures_discard_the_whole_packet() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (method, ram) in [
        (0x0118, MaxwellThreeDMmeRam::Instruction),
        (0x0120, MaxwellThreeDMmeRam::StartAddress),
    ] {
        let before = channel.three_d().clone();
        let decoded = packet(method / 4, 0);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: actual,
                error: MaxwellThreeDMmeLoadError::PointerUnset,
                ..
            }) if actual == ram
        ));
        assert_eq!(channel.three_d(), &before);
    }

    for (method, ram) in [
        (0x0114, MaxwellThreeDMmeRam::Instruction),
        (0x011c, MaxwellThreeDMmeRam::StartAddress),
    ] {
        let before = channel.three_d().clone();
        let decoded = incrementing_packet(method / 4, &[u32::MAX, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: actual,
                error: MaxwellThreeDMmeLoadError::PointerOverflow,
                ..
            }) if actual == ram
        ));
        assert_eq!(channel.three_d(), &before);
    }

    let before = channel.three_d().clone();
    let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2);
    arguments.push(0);
    arguments.resize(
        MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2,
        0x0000_0201,
    );
    let decoded = increment_once_packet(0x0114 / 4, &arguments);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeRamLoad {
            ram: MaxwellThreeDMmeRam::Instruction,
            error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                limit: MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2);
    arguments.push(0);
    arguments.resize(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2, 0);
    let decoded = increment_once_packet(0x011c / 4, &arguments);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeRamLoad {
            ram: MaxwellThreeDMmeRam::StartAddress,
            error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                limit: MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn mme_macro_executes_captured_code_and_emits_validated_methods() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let macro_index = 5;
    let point_size_method_dword = 0x1518 / 4;
    let set_method = 1 | (2 << 4) | (point_size_method_dword << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[set_method, send_parameter_and_exit, 0x11],
    );

    let argument = 2.5_f32.to_bits();
    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, argument);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &call.packets()[0],
    )
    .unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "CALL_MME_MACRO"
    );
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::MmeMacroCall {
            macro_index,
            parameter_count: 1,
            report: MaxwellThreeDMmeExecutionReport {
                instructions: 3,
                emitted_methods: 1,
            },
        }
    );
    let point_size = channel.three_d().raster().point_size();
    assert_eq!(point_size.raw(), Some(argument));
    let source = point_size.source().unwrap();
    assert_eq!(
        source.location(),
        dispatch.methods()[0].method().source().location()
    );
    assert_eq!(source.method(), GpuMethodId(0x1518));
    assert_eq!(source.argument(), argument);
    assert_eq!(
        channel
            .three_d()
            .raw_register(GpuMethodId(0x1518))
            .and_then(MaxwellThreeDRegister::raw),
        Some(argument)
    );
}

#[test]
fn mme_call_data_supplies_additional_parameters() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let macro_index = 2;
    let fetch_second_parameter = 1 | (2 << 8);
    let point_size_method_dword = 0x1518 / 4;
    let set_method = 1 | (2 << 4) | (point_size_method_dword << 14);
    let send_second_parameter_and_exit = (4 << 4) | (1 << 7) | (2 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            fetch_second_parameter,
            set_method,
            send_second_parameter_and_exit,
            0x11,
        ],
    );

    let argument = 4.0_f32.to_bits();
    let call = incrementing_packet(
        (0x3800 + u32::from(macro_index) * 8) / 4,
        &[0xdead_beef, argument],
    );
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &call.packets()[0],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "CALL_MME_DATA"
    );
    assert_eq!(
        dispatch.methods()[1].effect(),
        MaxwellEngineMethodEffect::MmeMacroData { macro_index }
    );
    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(argument)
    );
}

#[test]
fn mme_reads_polygon_mode_reset_bits_until_guest_programming_overrides_them() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (method, register) in [
        (0x0dac, MaxwellThreeDFixedFunctionRegister::FrontPolygonMode),
        (0x0db0, MaxwellThreeDFixedFunctionRegister::BackPolygonMode),
    ] {
        let raw = channel.three_d().raw_register(GpuMethodId(method)).unwrap();
        assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
        assert_eq!(raw.raw(), Some(0));
        assert_eq!(raw.value(), Some(&0));
        assert_eq!(raw.source(), None);

        let typed = channel.three_d().fixed_function().register(register);
        assert_eq!(typed.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
        assert_eq!(typed.raw(), Some(0));
        assert_eq!(typed.value(), None);
        assert_eq!(typed.source(), None);
    }

    let macro_index = 3;
    let read_front = 5 | (1 << 4) | (2 << 8) | (0x036b << 14);
    let read_back_and_exit = 5 | (1 << 4) | (1 << 7) | (3 << 8) | (0x036c << 14);
    load_mme_program(
        &mut channel,
        macro_index,
        &[read_front, read_back_and_exit, 0x11],
    );
    let before_call = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &call.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::MmeMacroCall {
            macro_index,
            parameter_count: 1,
            report: MaxwellThreeDMmeExecutionReport {
                instructions: 3,
                emitted_methods: 0,
            },
        }
    );
    assert_eq!(channel.three_d(), &before_call);

    program_three_d(&mut channel, 0x0dac, 0x1b02);
    let raw = channel.three_d().raw_register(GpuMethodId(0x0dac)).unwrap();
    assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(raw.raw(), Some(0x1b02));
    assert!(raw.source().is_some());
    let typed = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::FrontPolygonMode);
    assert_eq!(typed.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(
        typed.value(),
        Some(&MaxwellThreeDFixedFunctionValue::PolygonMode(
            MaxwellThreeDPolygonMode::Fill,
        ))
    );
    assert_eq!(
        channel
            .three_d()
            .raw_register(GpuMethodId(0x0db0))
            .unwrap()
            .origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );
}

#[test]
fn mme_reads_all_pipeline_shader_reset_headers_and_writes_override_one_slot() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
        let method = 0x2000 + pipeline as u32 * 0x40;
        let raw = channel.three_d().raw_register(GpuMethodId(method)).unwrap();
        assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
        assert_eq!(raw.raw(), Some(0));
        assert_eq!(raw.value(), Some(&0));
        assert_eq!(raw.source(), None);

        let binding = &channel.three_d().shader_bindings().pipeline()[pipeline];
        assert_eq!(
            binding.enabled().origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
        assert_eq!(binding.enabled().raw(), Some(0));
        assert_eq!(binding.enabled().value(), Some(&false));
        assert_eq!(binding.enabled().source(), None);
        assert_eq!(
            binding.stage().origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
        assert_eq!(binding.stage().raw(), Some(0));
        assert_eq!(
            binding.stage().value(),
            Some(&MaxwellThreeDShaderStage::VertexCullBeforeFetch)
        );
        assert_eq!(binding.stage().source(), None);
        assert_eq!(binding.group().origin(), MaxwellThreeDRegisterOrigin::Unset);
    }

    let macro_index = 4;
    let read_pipeline_three = 5 | (1 << 4) | (2 << 8) | (0x0830 << 14);
    let read_pipeline_four_and_exit = 5 | (1 << 4) | (1 << 7) | (3 << 8) | (0x0840 << 14);
    load_mme_program(
        &mut channel,
        macro_index,
        &[read_pipeline_three, read_pipeline_four_and_exit, 0x11],
    );
    let before_call = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &call.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::MmeMacroCall {
            macro_index,
            parameter_count: 1,
            report: MaxwellThreeDMmeExecutionReport {
                instructions: 3,
                emitted_methods: 0,
            },
        }
    );
    assert_eq!(channel.three_d(), &before_call);

    let before_invalid = channel.three_d().clone();
    let invalid = packet(0x20c0 / 4, 2);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_PIPELINE_SHADER",
            ..
        }) if source.method() == GpuMethodId(0x20c0)
    ));
    assert_eq!(channel.three_d(), &before_invalid);

    program_three_d(&mut channel, 0x20c0, 0x41);
    let raw = channel.three_d().raw_register(GpuMethodId(0x20c0)).unwrap();
    assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(raw.raw(), Some(0x41));
    assert!(raw.source().is_some());
    let binding = &channel.three_d().shader_bindings().pipeline()[3];
    assert_eq!(
        binding.enabled().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(binding.enabled().value(), Some(&true));
    assert_eq!(
        binding.stage().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(
        binding.stage().value(),
        Some(&MaxwellThreeDShaderStage::Geometry)
    );
    assert_eq!(
        channel
            .three_d()
            .raw_register(GpuMethodId(0x2100))
            .unwrap()
            .origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );
}

#[test]
fn mme_emitted_draw_keeps_the_exact_candidate_snapshot() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let macro_index = 6;
    let draw_method_dword = 0x0d78 / 4;
    let set_method = 1 | (2 << 4) | (draw_method_dword << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[set_method, send_parameter_and_exit, 0x11],
    );

    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 3);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &call.packets()[0],
    )
    .unwrap();
    assert_eq!(dispatch.operations().len(), 1);
    let operation = &dispatch.operations()[0];
    assert_eq!(operation.state(), channel.three_d());
    assert!(matches!(
        operation.trigger(),
        MaxwellThreeDOperationTrigger::DrawVertexArray {
            source,
            vertex_count: 3,
        } if source.method() == GpuMethodId(0x0d78)
            && source.location() == dispatch.methods()[0].method().source().location()
    ));
}

#[test]
fn mme_execution_errors_and_partial_emissions_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    let data = packet(0x3804 / 4, 0);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &data.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
            ..
        })
    ));

    let missing = packet(0x3800 / 4, 0);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &missing.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index: 0 },
            ..
        })
    ));

    let macro_index = 1;
    let point_size_method_dword = 0x1518 / 4;
    let set_point_size = 1 | (2 << 4) | (point_size_method_dword << 14);
    let send_parameter = (4 << 4) | (1 << 11);
    let set_recursive_method = 1 | (2 << 4) | (0x0e00 << 14);
    let send_recursive_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            set_point_size,
            send_parameter,
            set_recursive_method,
            send_recursive_and_exit,
            0x11,
        ],
    );
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 1.0_f32.to_bits());
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall {
                method_dword: 0x0e00,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn mme_register_reads_and_execution_limit_fail_typed_and_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    let read_macro = 3;
    let read_unset_register = 5 | (1 << 4) | (2 << 8) | (0x036d << 14);
    load_mme_program(&mut channel, read_macro, &[read_unset_register]);
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(read_macro) * 8) / 4, 0);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable {
                method_dword: 0x036d,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);

    let loop_macro = 4;
    let branch_to_self_without_delay = 7 | (1 << 5);
    load_mme_program(&mut channel, loop_macro, &[branch_to_self_without_delay]);
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(loop_macro) * 8) / 4, 0);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::InstructionLimitExceeded {
                limit: MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
            },
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn vertex_assembly_controls_are_typed_source_preserving_and_isolated() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let input_before = channel.three_d().vertex_input().clone();
    let two_d_before = channel.two_d().clone();

    let decoded = packet(0x1610 / 4, 0x0e);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    let defaults_source = dispatch.methods()[0].method().source();
    let defaults = channel
        .three_d()
        .vertex_input()
        .assembly()
        .attribute_defaults();
    let value = *defaults.value().unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_ATTRIBUTE_DEFAULT"
    );
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
            MaxwellThreeDVertexInputWrite::AttributeDefaults {
                value,
                source: defaults_source,
            }
        ))
    );
    assert_eq!(defaults.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(defaults.raw(), Some(0x0e));
    assert_eq!(defaults.source(), Some(defaults_source));
    assert_eq!(
        value.color_front_diffuse(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.color_front_specular(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.generic_vector(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.fixed_function_texture(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.dx9_color0(),
        MaxwellThreeDAttributeDefaultVector::Vector0001
    );
    assert_eq!(
        value.dx9_color1_to_color15(),
        MaxwellThreeDAttributeDefaultVector::Vector0000
    );
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .assembly()
            .vertex_id_uses_array_start(),
        input_before.assembly().vertex_id_uses_array_start()
    );

    let decoded = packet(0x164c / 4, 0x1000);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    let vertex_id_source = dispatch.methods()[0].method().source();
    let input = channel.three_d().vertex_input();
    let vertex_id = input.assembly().vertex_id_uses_array_start();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_DA_OUTPUT"
    );
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
            MaxwellThreeDVertexInputWrite::VertexIdUsesArrayStart {
                value: MaxwellThreeDVertexIdUsesArrayStart::Enabled,
                source: vertex_id_source,
            }
        ))
    );
    assert_eq!(vertex_id.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(vertex_id.raw(), Some(0x1000));
    assert_eq!(
        vertex_id.value(),
        Some(&MaxwellThreeDVertexIdUsesArrayStart::Enabled)
    );
    assert_eq!(vertex_id.source(), Some(vertex_id_source));
    assert_eq!(input.streams(), input_before.streams());
    assert_eq!(input.attributes(), input_before.attributes());
    assert_eq!(input.index(), input_before.index());
    assert_eq!(input.primitive(), input_before.primitive());
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn vertex_assembly_controls_are_draw_dependencies() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let unset = channel.three_d().pipeline_dependencies(&[]);

    program_three_d(&mut channel, 0x1610, 0x0e);
    let defaults = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(defaults, unset);

    program_three_d(&mut channel, 0x164c, 0x1000);
    let vertex_id = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(vertex_id, defaults);

    program_three_d(&mut channel, 0x1610, 0x3f);
    assert_ne!(channel.three_d().pipeline_dependencies(&[]), vertex_id);
}

#[test]
fn invalid_vertex_assembly_controls_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x1610, 0x0e);
    program_three_d(&mut channel, 0x164c, 0x1000);

    for (method, method_name, arguments) in [
        (
            0x1610,
            "SET_ATTRIBUTE_DEFAULT",
            [0x40, 0x100, 0x8000_0000, u32::MAX],
        ),
        (0x164c, "SET_DA_OUTPUT", [1, 0x2000, 0x1001, u32::MAX]),
    ] {
        for argument in arguments {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: actual,
                    ..
                }) if actual == method_name && source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1648 / 4, &[0xdead_beef, 1]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_DA_OUTPUT",
            ..
        }) if source.argument() == 1
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_stream_attributes_and_begin_state_remain_unresolved_and_typed() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, 0),
        (0x1c08, 0x2000),
        (0x1c0c, 1),
        (0x1f00, 0),
        (0x1f04, 0x20ff),
        (0x1160, 0x3820_0000),
        (0x1164, 0x40),
        (0x0d74, 7),
        (0x1618, 4),
        (0x1970, 4),
    ] {
        let decoded = packet(method / 4, argument);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    let input = channel.three_d().vertex_input();
    let stream = &input.streams()[0];
    assert_eq!(stream.address().unwrap().get(), 0x2000);
    assert_eq!(stream.limit().unwrap().get(), 0x20ff);
    assert_eq!(stream.format().value().unwrap().stride(), 16);
    assert!(stream.format().value().unwrap().enabled());
    let attribute = input.attributes()[0].value().unwrap();
    assert!(attribute.enabled());
    assert_eq!(attribute.stream(), 0);
    assert_eq!(attribute.component_widths().unwrap().byte_size(), 16);
    assert!(!input.attributes()[1].value().unwrap().enabled());
    assert_eq!(input.primitive().vertex_array_start().value(), Some(&7));
    assert_eq!(input.primitive().begin().value().unwrap().topology(), 4);
    assert_eq!(input.primitive().topology().value().unwrap().raw(), 4);
}

#[test]
fn end_closes_the_active_begin_and_preserves_sequence_provenance_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    program_three_d(&mut channel, 0x1618, 4);
    let primitive = channel.three_d().vertex_input().primitive();
    assert!(primitive.is_open());
    assert_eq!(primitive.active_begin().unwrap().topology(), 4);
    let begin_source = primitive.begin().source().unwrap();
    let open_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(open_dependencies, dependencies_before);

    let decoded = packet(0x1614 / 4, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    let source = dispatch.methods()[0].method().source();
    assert_eq!(dispatch.methods()[0].metadata().method_name(), "END");
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
            MaxwellThreeDVertexInputWrite::End {
                value: false,
                source,
            }
        ))
    );
    assert!(dispatch.operations().is_empty());

    let primitive = channel.three_d().vertex_input().primitive();
    assert!(!primitive.is_open());
    assert_eq!(primitive.active_begin(), None);
    assert_eq!(primitive.begin().value().unwrap().topology(), 4);
    assert_eq!(primitive.begin().source(), Some(begin_source));
    assert_eq!(
        primitive.end().origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(primitive.end().raw(), Some(0));
    assert_eq!(primitive.end().value(), Some(&false));
    assert_eq!(primitive.end().source(), Some(source));
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1610 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "END",
            ..
        }) if source.method() == GpuMethodId(0x1614) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    let decoded = incrementing_packet(0x1614 / 4, &[1, 5]);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    let primitive = channel.three_d().vertex_input().primitive();
    assert!(primitive.is_open());
    assert_eq!(primitive.end().value(), Some(&true));
    assert_eq!(primitive.active_begin().unwrap().topology(), 5);
}

#[test]
fn malformed_vertex_suffix_and_index_relationships_reject_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let malformed = incrementing_packet(0x1c00 / 4, &[0x1010, 0, 0x2000, 1, 0x2000]);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &malformed.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);

    for (method, argument) in [(0x17c8, 0), (0x17cc, 0x1000), (0x17d0, 0), (0x17d4, 0x100e)] {
        let decoded = packet(method / 4, argument);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }
    let before_size = channel.three_d().clone();
    let size = packet(0x17d8 / 4, 2);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &size.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::ContradictoryState { .. })
    ));
    assert_eq!(channel.three_d(), &before_size);
}

#[test]
fn shader_bindings_snapshot_selectors_and_preserve_stage_visibility() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    for (method, argument) in [
        (0x2000, 0x11),
        (0x2010, 2),
        (0x2380, 0x100),
        (0x2384, 0),
        (0x2388, 0x4000),
        (0x2450, 0x31),
        (0x1574, 0),
        (0x1578, 0x8000),
        (0x157c, 3),
        (0x155c, 0),
        (0x1560, 0xa000),
        (0x1564, 7),
        (0x1234, 1),
        (0x2608, 3),
    ] {
        let decoded = packet(method / 4, argument);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    let bindings = channel.three_d().shader_bindings();
    let constant = bindings.groups()[2].constant_buffers()[3].unwrap();
    assert!(constant.enabled());
    assert_eq!(constant.address().unwrap().get(), 0x4000);
    assert_eq!(constant.size(), Some(0x100));
    assert!(bindings.stage_visibility(2)[MaxwellThreeDShaderStage::Vertex as usize]);
    assert_eq!(bindings.texture_headers().address().unwrap().get(), 0x8000);
    assert_eq!(bindings.samplers().maximum_index().value(), Some(&7));
    assert_eq!(
        bindings.sampler_binding().value(),
        Some(&MaxwellThreeDSamplerBindingMode::ViaTextureHeader)
    );
}

#[test]
fn program_region_is_source_preserving_and_only_active_for_shader_pipelines() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();
    let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);

    let lower = packet(0x160c / 4, 0);
    let lower_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &lower.packets()[0],
    )
    .unwrap();
    let lower_source = lower_dispatch.methods()[0].method().source();
    let region = channel.three_d().shader_bindings().program_region();
    assert_eq!(
        lower_dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_B"
    );
    assert_eq!(
        lower_dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderBinding(
            MaxwellThreeDShaderBindingWrite::ProgramRegionAddressLower {
                value: 0,
                source: lower_source,
            }
        ))
    );
    assert!(region.address().is_none());
    assert_eq!(region.address_lower().raw(), Some(0));
    assert_eq!(region.address_lower().source(), Some(lower_source));
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );

    let upper = packet(0x1608 / 4, 4);
    let upper_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &upper.packets()[0],
    )
    .unwrap();
    let upper_source = upper_dispatch.methods()[0].method().source();
    let region = channel.three_d().shader_bindings().program_region();
    assert_eq!(
        upper_dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_A"
    );
    assert_eq!(region.address_upper().raw(), Some(4));
    assert_eq!(region.address_upper().source(), Some(upper_source));
    assert_eq!(region.address().unwrap().get(), 0x0000_0004_0000_0000);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );
    assert_eq!(channel.two_d(), &two_d_before);

    program_three_d(&mut channel, 0x2000, 0x11);
    let active_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x160c, 0x1000);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        active_dependencies
    );
}

#[test]
fn invalid_program_region_upper_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x1608, 4);
    program_three_d(&mut channel, 0x160c, 0);

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1608 / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PROGRAM_REGION_A",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1608 / 4, &[5, 0x1000, 0x40]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_ATTRIBUTE_DEFAULT",
            ..
        }) if source.method() == GpuMethodId(0x1610)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_stream_substitute_address_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    let lower = packet(0x0f88 / 4, 0x082c_3000);
    let lower_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &lower.packets()[0],
    )
    .unwrap();
    let lower_method = lower_dispatch.methods()[0];
    let lower_source = lower_method.method().source();
    assert_eq!(
        lower_method.metadata().method_name(),
        "SET_VERTEX_STREAM_SUBSTITUTE_B"
    );
    assert_eq!(
        lower_method.effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
            MaxwellThreeDVertexInputWrite::StreamSubstituteAddressLower {
                value: 0x082c_3000,
                source: lower_source,
            }
        ))
    );
    let substitute = channel.three_d().vertex_input().stream_substitute();
    assert!(substitute.address().is_none());
    assert_eq!(substitute.address_lower().raw(), Some(0x082c_3000));
    assert_eq!(substitute.address_lower().source(), Some(lower_source));

    let upper = packet(0x0f84 / 4, 0x7f);
    let upper_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &upper.packets()[0],
    )
    .unwrap();
    let upper_method = upper_dispatch.methods()[0];
    let upper_source = upper_method.method().source();
    assert_eq!(
        upper_method.metadata().method_name(),
        "SET_VERTEX_STREAM_SUBSTITUTE_A"
    );
    assert_eq!(
        upper_method.effect(),
        MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
            MaxwellThreeDVertexInputWrite::StreamSubstituteAddressUpper {
                value: 0x7f,
                source: upper_source,
            }
        ))
    );
    let substitute = channel.three_d().vertex_input().stream_substitute();
    assert_eq!(substitute.address_upper().raw(), Some(0x7f));
    assert_eq!(substitute.address_upper().source(), Some(upper_source));
    assert_eq!(substitute.address().unwrap().get(), 0x7f_082c_3000);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn invalid_vertex_stream_substitute_upper_and_packet_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x0f84, 4);
    program_three_d(&mut channel, 0x0f88, 0x1000);

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0f84 / 4, &[argument, 0x082c_3000]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_VERTEX_STREAM_SUBSTITUTE_A",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x0f84 / 4, &[0, 0x082c_3000, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0f8c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn active_shader_pipeline_requires_complete_program_region_but_clear_does_not() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x160c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
    let source = channel
        .three_d()
        .shader_bindings()
        .program_region()
        .address_lower()
        .source()
        .unwrap();

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_PROGRAM_REGION_A/B"
        ))
    ));

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));

    program_three_d(&mut channel, 0x1608, 4);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn pipeline_program_offsets_and_register_counts_are_indexed_and_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
        let program_method = 0x2004 + pipeline as u32 * 0x40;
        let count_method = 0x200c + pipeline as u32 * 0x40;
        let offset = 0x1000 + pipeline as u32 * 0x80;
        let count = 4 + pipeline as u32;

        let program = packet(program_method / 4, offset);
        let program_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &program.packets()[0],
        )
        .unwrap();
        let program_source = program_dispatch.methods()[0].method().source();
        assert_eq!(
            program_dispatch.methods()[0].metadata().method_name(),
            "SET_PIPELINE_PROGRAM"
        );
        assert_eq!(
            program_dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderBinding(
                MaxwellThreeDShaderBindingWrite::PipelineProgram {
                    pipeline: pipeline as u8,
                    offset,
                    source: program_source,
                }
            ))
        );

        let count_packet = packet(count_method / 4, count);
        let count_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &count_packet.packets()[0],
        )
        .unwrap();
        let count_source = count_dispatch.methods()[0].method().source();
        assert_eq!(
            count_dispatch.methods()[0].metadata().method_name(),
            "SET_PIPELINE_REGISTER_COUNT"
        );

        let binding = &channel.three_d().shader_bindings().pipeline()[pipeline];
        assert_eq!(binding.program_offset().raw(), Some(offset));
        assert_eq!(binding.program_offset().value(), Some(&offset));
        assert_eq!(binding.program_offset().source(), Some(program_source));
        assert_eq!(binding.register_count().raw(), Some(count));
        assert_eq!(binding.register_count().value(), Some(&(count as u8)));
        assert_eq!(binding.register_count().source(), Some(count_source));
    }

    for argument in [0x100, 0x1ff, u32::MAX] {
        let before = channel.clone();
        let invalid = packet(0x200c / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PIPELINE_REGISTER_COUNT",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn program_offset_and_register_count_dependencies_require_an_enabled_slot() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let reset_dependencies = channel.three_d().pipeline_dependencies(&[]);

    program_three_d(&mut channel, 0x2044, 0x7f730);
    program_three_d(&mut channel, 0x204c, 4);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        reset_dependencies
    );

    program_three_d(&mut channel, 0x2040, 0x11);
    let enabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(enabled_dependencies, reset_dependencies);

    program_three_d(&mut channel, 0x2044, 0x7f830);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        enabled_dependencies
    );
    let offset_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x204c, 5);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        offset_dependencies
    );

    program_three_d(&mut channel, 0x2040, 0x10);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x2044, 0x7f930);
    program_three_d(&mut channel, 0x204c, 6);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );
}

#[test]
fn tessellation_lod_family_is_source_preserving_and_stage_conditional() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let levels = [
        MaxwellThreeDTessellationLod::OuterU0OrDensity,
        MaxwellThreeDTessellationLod::OuterV0OrDetail,
        MaxwellThreeDTessellationLod::OuterU1OrW0,
        MaxwellThreeDTessellationLod::OuterV1,
        MaxwellThreeDTessellationLod::InnerU,
        MaxwellThreeDTessellationLod::InnerV,
    ];
    let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);

    for (index, level) in levels.into_iter().enumerate() {
        let method = 0x0324 + index as u32 * 4;
        let argument = if index == 5 {
            u32::MAX
        } else {
            0x3f80_0000 + index as u32
        };
        let decoded = packet(method / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TESSELLATION_LOD"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderBinding(
                MaxwellThreeDShaderBindingWrite::TessellationLod {
                    level,
                    value: argument,
                    source,
                }
            ))
        );
        let register = channel.three_d().shader_bindings().tessellation_lod(level);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&argument));
        assert_eq!(register.source(), Some(source));
    }
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );

    program_three_d(&mut channel, 0x2080, 0x21);
    let tessellation_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(tessellation_dependencies, inactive_dependencies);
    program_three_d(&mut channel, 0x0324, 0x4000_0000);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        tessellation_dependencies
    );

    program_three_d(&mut channel, 0x2080, 0x20);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x0324, 0x4040_0000);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );
}

#[test]
fn render_target_index_offset_is_typed_source_preserving_and_conditionally_dependent() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();
    let neutral_dependencies = channel.three_d().pipeline_dependencies(&[0]);

    for (argument, expected, enabled) in [
        (0, MaxwellThreeDRenderTargetIndexOffset::Disabled, false),
        (
            1,
            MaxwellThreeDRenderTargetIndexOffset::ByViewportIndex,
            true,
        ),
    ] {
        let decoded = packet(0x11f0 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .render_targets()
            .render_target_index_offset();

        assert_eq!(
            method.metadata().method_name(),
            "SET_OFFSET_RENDER_TARGET_INDEX"
        );
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::RenderTargetIndexOffset {
                    value: expected,
                    source,
                }
            ))
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.enabled(), enabled);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[0]) == neutral_dependencies,
            !enabled
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x11f0 / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_OFFSET_RENDER_TARGET_INDEX",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}
