use std::sync::Arc;

use super::*;

#[test]
fn raw_words_and_shadow_words_preserve_values() {
    let mut channel = three_d_channel();
    let reset = channel
        .three_d_mut()
        .raw_register_value(GpuMethodId(0x0dac))
        .unwrap();

    assert_eq!(Some(reset), Some(0x1b02));

    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x0dac))
            .is_none()
    );
    for method in [0x3401, 0x37fc, 0x4000, u32::MAX] {
        assert!(
            channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(method))
                .is_none()
        );
        assert!(
            channel
                .three_d_mut()
                .mme()
                .shadow_value(GpuMethodId(method))
                .is_none()
        );
    }

    // Both ends of scratch RAM accept all 32-bit words, including zero. Source
    // offsets differ even when the final value repeats an earlier write.
    for method in [0x3400, 0x37fc] {
        let decoded =
            non_incrementing_packet_on_subchannel(0, method / 4, &[u32::MAX, 0, u32::MAX]);
        let applied = dispatch_first(&mut channel, &decoded).unwrap();
        let source = applied.methods().last().unwrap().method().source();
        let raw = channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(method))
            .unwrap();
        assert_eq!(Some(raw), Some(u32::MAX));
        assert_eq!(Some(&raw), Some(&u32::MAX));
        assert_eq!(
            channel
                .three_d_mut()
                .mme()
                .shadow_value(GpuMethodId(method)),
            Some(u32::MAX)
        );

        let retained = channel.three_d_mut().clone();
        let rewrite = dispatch_method(&mut channel, method / 4, u32::MAX).unwrap();
        let new_source = rewrite.methods()[0].method().source();
        assert_ne!(source, new_source);
        assert_eq!(retained.raw_register_value(GpuMethodId(method)), Some(raw));
        assert_eq!(
            retained.mme().shadow_value(GpuMethodId(method)),
            Some(u32::MAX)
        );
        let rewritten = channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(method))
            .unwrap();
        assert_eq!(Some(rewritten), Some(raw));
        assert_eq!(
            channel
                .three_d_mut()
                .mme()
                .shadow_value(GpuMethodId(method)),
            Some(u32::MAX)
        );
    }
}

#[test]
fn notification_address_state_arms_a_single_following_no_operation() {
    let mut channel = three_d_channel();
    for (method, value) in [(0x0104, 0x12), (0x0108, 0x3456_7890)] {
        let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
        assert!(dispatch.ordered_operations().is_empty());
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(method)),
            Some(value)
        );
    }
    assert!(
        dispatch_method(&mut channel, 0x010c / 4, 0)
            .unwrap()
            .ordered_operations()
            .is_empty()
    );
    let dispatch = dispatch_method(&mut channel, 0x0100 / 4, 0x1234).unwrap();
    assert!(matches!(
        dispatch.ordered_operations(),
        [MaxwellEngineOperation::Notification {
            address: 0x0012_3456_7890,
            ..
        }]
    ));
    assert!(
        dispatch_method(&mut channel, 0x0100 / 4, 0)
            .unwrap()
            .ordered_operations()
            .is_empty()
    );
    assert!(dispatch_method(&mut channel, 0x010c / 4, 1).is_err());
    assert!(dispatch_method(&mut channel, 0x0104 / 4, 0x100).is_err());
}

#[test]
fn pending_notification_rejects_unimplemented_completions_and_context_switches() {
    let mut channel = three_d_channel();
    assert!(dispatch_method(&mut channel, 0x010c / 4, 0).is_err());
    dispatch_first(
        &mut channel,
        &packet_on_subchannel(1, 0, SWITCH_1_GM20B_PROFILE.classes().compute().0),
    )
    .unwrap();
    dispatch_method(&mut channel, 0x0104 / 4, 0).unwrap();
    dispatch_method(&mut channel, 0x0108 / 4, 0x1000).unwrap();
    dispatch_method(&mut channel, 0x010c / 4, 0).unwrap();
    assert!(dispatch_method(&mut channel.clone(), 0x010c / 4, 0).is_err());
    assert!(dispatch_method(&mut channel.clone(), 0x0108 / 4, 0x2000).is_err());
    assert!(dispatch_method(&mut channel.clone(), 0x0f7c / 4, 0).is_err());
    assert!(
        dispatch_method(
            &mut channel.clone(),
            0,
            SWITCH_1_GM20B_PROFILE.classes().three_d().0
        )
        .is_err()
    );
    assert!(dispatch_first(&mut channel, &packet_on_subchannel(1, 0x0100 / 4, 0)).is_err());
}

#[test]
fn z_compression_selector_is_typed_depth_state_without_an_operation() {
    let mut channel = three_d_channel();
    let color_before = channel.three_d().render_targets().color().clone();
    let two_d_before = channel.two_d().clone();

    for (argument, expected) in [
        (0, MaxwellThreeDZCompressionMode::Disabled),
        (1, MaxwellThreeDZCompressionMode::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x19cc / 4, argument).unwrap();
        let register = channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .compression();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_Z_COMPRESSION"
        );

        assert!(dispatch.operations().is_empty());

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));

        assert_eq!(expected.raw(), argument);
        assert_eq!(channel.three_d().render_targets().color(), &color_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let resources = fixture_bindings(
        channel.three_d(),
        resource_address_space().memory(),
        &mut crate::test_memory::apply,
    )
    .unwrap();
    assert!(resources.resources().next().is_none());
}

#[test]
fn invalid_z_compression_values_are_rejected_atomically() {
    let mut channel = three_d_channel();

    for argument in [2, 3, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x19cc / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
fn color_compression_selectors_are_typed_and_isolated_per_target() {
    for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
        let mut channel = three_d_channel();
        let depth_before = channel.three_d().render_targets().depth_stencil().clone();
        let two_d_before = channel.two_d().clone();

        for state in channel.three_d().render_targets().color().iter() {
            let compression = state.compression();

            assert_eq!(compression.raw(), Some(0));
            assert_eq!(
                compression.value().copied(),
                Some(MaxwellThreeDColorCompressionMode::Disabled)
            );
        }
        for index in 0..MAXWELL_COLOR_TARGET_COUNT {
            let raw = channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(0x19e0 + index as u32 * 4))
                .expect("color compression reset must be visible to MME");

            assert_eq!(Some(raw), Some(0));
            assert_eq!(Some(&raw).copied(), Some(0));
        }

        for (argument, expected) in [
            (0, MaxwellThreeDColorCompressionMode::Disabled),
            (1, MaxwellThreeDColorCompressionMode::Enabled),
        ] {
            let method = 0x19e0 + u32::from(target) * 4;
            let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
            let targets = channel.three_d().render_targets().color();
            let register = targets[target as usize].compression();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_COLOR_COMPRESSION"
            );

            assert!(dispatch.operations().is_empty());

            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));

            assert_eq!(expected.raw(), argument);
            for (other, state) in targets.iter().enumerate() {
                if other != target as usize {
                    let compression = state.compression();

                    assert_eq!(compression.raw(), Some(0));
                    assert_eq!(
                        compression.value().copied(),
                        Some(MaxwellThreeDColorCompressionMode::Disabled)
                    );
                }
            }
            assert_eq!(
                channel.three_d().render_targets().depth_stencil(),
                &depth_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let resources = fixture_bindings(
            channel.three_d(),
            resource_address_space().memory(),
            &mut crate::test_memory::apply,
        )
        .unwrap();
        assert!(resources.resources().next().is_none());
    }
}

#[test]
fn invalid_color_compression_values_are_rejected_atomically() {
    for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
        for argument in [2, 3, u32::MAX] {
            let mut channel = three_d_channel();
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let method = 0x19e0 + u32::from(target) * 4;
            let decoded = packet(method / 4, argument);

            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
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
fn zero_bandwidth_clear_masks_preserve_independent_state_without_gpu_work() {
    let mut channel = three_d_channel();
    let targets = channel.three_d().render_targets();
    assert_eq!(targets.color_zero_bandwidth_clear().value(), None);
    assert_eq!(targets.depth_zero_bandwidth_clear().value(), None);
    let draw_identity = channel.three_d().draw_state_identity();
    let resource_identities = [
        MaxwellThreeDResourceRole::ColorTarget(0),
        MaxwellThreeDResourceRole::DepthStencilTarget,
    ]
    .map(|role| channel.three_d().resource_state_identity(role));
    let two_d = channel.two_d().clone();
    for (method, name) in [
        (0x07a4, "SET_COLOR_ZERO_BANDWIDTH_CLEAR"),
        (0x07a8, "SET_Z_ZERO_BANDWIDTH_CLEAR"),
    ] {
        for mask in [0x7ff8, 0, 0x7fff, 1, 0x4000] {
            let before = channel.three_d().render_targets().clone();
            let dispatch = dispatch_method(&mut channel, method / 4, mask).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
            let targets = channel.three_d().render_targets();
            let register = if method == 0x07a4 {
                assert_eq!(
                    targets.depth_zero_bandwidth_clear(),
                    before.depth_zero_bandwidth_clear()
                );
                targets.color_zero_bandwidth_clear()
            } else {
                assert_eq!(
                    targets.color_zero_bandwidth_clear(),
                    before.color_zero_bandwidth_clear()
                );
                targets.depth_zero_bandwidth_clear()
            };
            assert_eq!(register.raw(), Some(mask));
            assert_eq!(register.value(), Some(&(mask as u16)));

            assert!(draw_identity.matches(channel.three_d()));
            assert!(
                resource_identities
                    .iter()
                    .all(|identity| identity.matches(channel.three_d()))
            );
            assert_eq!(channel.two_d(), &two_d);
        }
    }
}

#[test]
fn compression_threshold_is_typed_source_preserving_nonsemantic_policy() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for (argument, expected, samples) in [
        (0, MaxwellThreeDCompressionThreshold::Samples0, 0),
        (1, MaxwellThreeDCompressionThreshold::Samples1, 1),
        (2, MaxwellThreeDCompressionThreshold::Samples2, 2),
        (3, MaxwellThreeDCompressionThreshold::Samples4, 4),
        (4, MaxwellThreeDCompressionThreshold::Samples8, 8),
        (5, MaxwellThreeDCompressionThreshold::Samples16, 16),
        (6, MaxwellThreeDCompressionThreshold::Samples32, 32),
        (7, MaxwellThreeDCompressionThreshold::Samples64, 64),
        (8, MaxwellThreeDCompressionThreshold::Samples128, 128),
        (9, MaxwellThreeDCompressionThreshold::Samples256, 256),
        (10, MaxwellThreeDCompressionThreshold::Samples512, 512),
        (11, MaxwellThreeDCompressionThreshold::Samples1024, 1024),
        (12, MaxwellThreeDCompressionThreshold::Samples2048, 2048),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1220 / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let register = channel.three_d().render_targets().compression_threshold();

        assert_eq!(method.metadata().method_name(), "SET_COMPRESSION_THRESHOLD");

        assert!(dispatch.operations().is_empty());

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));

        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.sample_count(), samples);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let resources = fixture_bindings(
        channel.three_d(),
        resource_address_space().memory(),
        &mut crate::test_memory::apply,
    )
    .unwrap();
    assert!(resources.resources().next().is_none());
}

#[test]
fn color_target_selection_retains_all_fields_for_counts_zero_through_eight() {
    let mut channel = three_d_channel();
    let targets = [7, 0, 6, 1, 5, 2, 4, 3];

    for count in 0..=8 {
        let argument = color_target_selection_raw(count, targets);
        let dispatch = dispatch_method(&mut channel, 0x121c / 4, argument).unwrap();
        let register = channel.three_d().render_targets().color_target_selection();
        let selection = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CT_SELECT"
        );
        assert!(dispatch.operations().is_empty());

        assert_eq!(register.raw(), Some(argument));

        assert_eq!(selection.target_count(), count);
        assert_eq!(selection.targets(), targets);
        assert_eq!(selection.active_targets(), &targets[..usize::from(count)]);
        assert_eq!(selection.raw(), argument);
    }
}

#[test]
fn render_target_layer_is_typed_source_preserving_and_consumer_scoped() {
    let mut channel = three_d_channel();

    for (argument, layer, control, fixed_layering, geometry_layering) in [
        (
            0,
            0,
            MaxwellThreeDRenderTargetLayerControl::Fixed,
            false,
            false,
        ),
        (
            0xffff,
            u16::MAX,
            MaxwellThreeDRenderTargetLayerControl::Fixed,
            true,
            true,
        ),
        (
            0x0001_0000,
            0,
            MaxwellThreeDRenderTargetLayerControl::GeometryShader,
            false,
            true,
        ),
        (
            0x0001_ffff,
            u16::MAX,
            MaxwellThreeDRenderTargetLayerControl::GeometryShader,
            false,
            true,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x15cc / 4, argument).unwrap();
        let value = MaxwellThreeDRenderTargetLayer::new(layer, control);
        let register = channel.three_d().render_targets().render_target_layer();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RT_LAYER"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.layer(), layer);
        assert_eq!(value.control(), control);
        assert_eq!(value.raw(), argument);
        assert_eq!(value.affects_draw_layering(false), fixed_layering);
        assert_eq!(value.affects_draw_layering(true), geometry_layering);

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
    }

    program_three_d(&mut channel, 0x15cc, 0);
    program_three_d(&mut channel, 0x20c0, 0x41);
    program_three_d(&mut channel, 0x15cc, 0x0001_0000);
}

#[test]
fn render_target_layer_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x15cc, 0);

    for argument in [0x0002_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x15cc / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_ANTI_ALIAS",
            ..
        }) if source.method() == GpuMethodId(0x15d0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn malformed_color_target_selection_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    let valid = color_target_selection_raw(2, [1, 0, 7, 6, 5, 4, 3, 2]);
    program_three_d(&mut channel, 0x121c, valid);

    for argument in [9, 15, 0x1000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x121c / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
    let decoded = incrementing_packet(0x121c / 4, &[1, 13]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_COMPRESSION_THRESHOLD",
            ..
        }) if source.method() == GpuMethodId(0x1220)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_register_write_applies_immediately() {
    let mut channel = three_d_channel();
    let decoded = packet(0x1518 / 4, 0x3fc0_0000);
    let before = channel.three_d().clone();

    dispatch_first(&mut channel, &decoded).unwrap();

    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(0x3fc0_0000)
    );
    assert_eq!(
        channel.three_d().raster().point_size().value().copied(),
        Some(MaxwellThreeDPointSize::from_bits(0x3fc0_0000))
    );

    assert_ne!(channel.three_d(), &before);
}

#[test]
fn enumerated_register_values_are_checked_before_state_changes() {
    let mut channel = three_d_channel();
    dispatch_method(&mut channel, 0x0d7c / 4, 1).unwrap();
    assert_eq!(
        channel.three_d().viewport().z_clip_range().value().copied(),
        Some(MaxwellThreeDViewportZClipRange::ZeroToPositiveW)
    );

    let before = channel.three_d().clone();
    let invalid = packet(0x0d7c / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            defined_mask: 1,
            ..
        })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn render_target_state_distinguishes_unset_disabled_ready_and_profile_unavailable() {
    let mut channel = three_d_channel();
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Unprogrammed
    );

    dispatch_method(&mut channel, 0x0810 / 4, 0).unwrap();
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Disabled
    );

    dispatch_incrementing(
        &mut channel,
        0x0800 / 4,
        &[0, 0x0080_0000, 1280, 720, 0xd5, 0, 1, 0, 0],
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
fn malformed_render_target_layout_is_rejected_before_application() {
    let mut channel = three_d_channel();

    let malformed_layout = packet(0x0814 / 4, 0x1001);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_first(&mut channel, &malformed_layout),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_eq!(channel.three_d(), &before);
}

#[test]
fn clear_and_fixed_function_state_preserve_typed_values_and_sources() {
    let mut channel = three_d_channel();
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
        dispatch_method(&mut channel, method / 4, argument).unwrap();
    }

    let state = channel.three_d();
    let clear = state.render_targets().clear();
    assert_eq!(clear.horizontal().value().unwrap().min, 10);
    assert_eq!(clear.horizontal().value().unwrap().max, 640);
    assert_eq!(clear.last_surface().value().unwrap().color_mask(), 0xf);

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
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for combination in 0_u32..16 {
        let argument = (combination & 1)
            | ((combination & 2) << 3)
            | ((combination & 4) << 6)
            | ((combination & 8) << 9);
        let dispatch = dispatch_method(&mut channel, 0x10f8 / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let register = channel.three_d().render_targets().clear().surface_control();
        let value = register.value().copied().unwrap();

        assert_eq!(method.metadata().method_name(), "SET_CLEAR_SURFACE_CONTROL");

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.raw(), argument);
        assert_eq!(value.respect_stencil_mask(), combination & 1 != 0);
        assert_eq!(value.use_clear_rect(), combination & 2 != 0);
        assert_eq!(value.use_scissor_zero(), combination & 4 != 0);
        assert_eq!(value.use_viewport_clip_zero(), combination & 8 != 0);

        assert_eq!(register.raw(), Some(argument));

        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn clear_surface_control_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10f8, 0x1111);

    for argument in [2, 0x20, 0x200, 0x2000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x10f8 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1100)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn malformed_scissor_suffix_keeps_valid_prefix_before_failure() {
    let mut channel = three_d_channel();
    let decoded = incrementing_packet(0x0e00 / 4, &[1, (100 << 16) | 5, (10 << 16) | 20]);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_ne!(channel.three_d(), &before);
}

#[test]
fn window_clip_type_is_typed_and_rejects_unknown_values_atomically() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDWindowClipType::Inclusive),
        (1, MaxwellThreeDWindowClipType::Exclusive),
        (2, MaxwellThreeDWindowClipType::ClipAll),
    ] {
        dispatch_method(&mut channel, 0x1950 / 4, argument).unwrap();
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
    }

    for argument in [3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1950 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
    let mut channel = three_d_channel();
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
    let arguments = std::array::from_fn::<_, 16, _>(|word| {
        let region = word / 2;
        let minimum = (region * 10 + word % 2) as u32;
        let maximum = minimum + 100;
        (maximum << 16) | minimum
    });
    let dispatch = dispatch_incrementing(&mut channel, 0x0d00 / 4, &arguments).unwrap();

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

            assert_eq!(register.raw(), Some(arguments[word]));
            assert_eq!(register.value(), Some(&expected));
        }
    }
}

#[test]
fn malformed_window_clip_rectangle_keeps_valid_prefix_before_failure() {
    let mut channel = three_d_channel();
    let valid = incrementing_packet(0x0d00 / 4, &[0; 16]);
    dispatch_first(&mut channel, &valid).unwrap();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let mut arguments = [0; 16];
    arguments[9] = (1 << 16) | 2;
    let malformed = incrementing_packet(0x0d00 / 4, &arguments);

    assert!(matches!(
        dispatch_first(&mut channel, &malformed),
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
fn clip_id_test_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let window_clip_before = channel.three_d().fixed_function().window_clip().to_owned();

    for (argument, expected) in [
        (0, MaxwellThreeDClipIdTestEnable::Disabled),
        (1, MaxwellThreeDClipIdTestEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x197c / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable);
        let value = MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(expected);

        assert_eq!(method.metadata().method_name(), "SET_CLIP_ID_TEST");

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));

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
            dispatch_first(&mut channel, &decoded),
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
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1980)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn viewport_scale_offset_enable_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDViewportScaleOffsetEnable::Disabled),
        (1, MaxwellThreeDViewportScaleOffsetEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x192c / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable);
        let value = MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(expected);

        assert_eq!(method.metadata().method_name(), "SET_VIEWPORT_SCALE_OFFSET");

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x192c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1930)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn mme_ram_loads_capture_typed_programs_with_sources_and_auto_advance() {
    let mut channel = three_d_channel();

    let start_dispatch = dispatch_incrementing(&mut channel, 0x011c / 4, &[5, 7]).unwrap();
    assert_eq!(
        start_dispatch.methods()[0].metadata().method_name(),
        "LOAD_MME_START_ADDRESS_RAM_POINTER"
    );
    assert_eq!(
        start_dispatch.methods()[1].metadata().method_name(),
        "LOAD_MME_START_ADDRESS_RAM"
    );

    let instruction_words = [0x0000_0301, 0x0000_0211, 0x0588_0021];
    let instruction_dispatch = dispatch_increment_once(
        &mut channel,
        0x0114 / 4,
        &[
            7,
            instruction_words[0],
            instruction_words[1],
            instruction_words[2],
        ],
    )
    .unwrap();
    let mme = channel.three_d_mut().mme();

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

        assert_eq!(register.raw(), Some(expected));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDMmeInstruction::new(expected))
        );
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
}

#[test]
fn mme_ram_load_failures_preserve_only_valid_prefixes() {
    let mut channel = three_d_channel();

    for (method, ram) in [
        (0x0118, MaxwellThreeDMmeRam::Instruction),
        (0x0120, MaxwellThreeDMmeRam::StartAddress),
    ] {
        let before = channel.three_d_mut().clone();
        let decoded = packet(method / 4, 0);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: actual,
                error: MaxwellThreeDMmeLoadError::PointerUnset,
                ..
            }) if actual == ram
        ));
        assert_eq!(channel.three_d_mut(), &before);
    }

    for (method, ram) in [
        (0x0114, MaxwellThreeDMmeRam::Instruction),
        (0x011c, MaxwellThreeDMmeRam::StartAddress),
    ] {
        let before = channel.three_d_mut().clone();
        let decoded = incrementing_packet(method / 4, &[u32::MAX, 0]);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: actual,
                error: MaxwellThreeDMmeLoadError::PointerOverflow,
                ..
            }) if actual == ram
        ));
        assert_ne!(channel.three_d_mut(), &before);
    }

    let before = channel.three_d_mut().clone();
    let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2);
    arguments.push(0);
    arguments.resize(
        MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2,
        0x0000_0201,
    );
    let decoded = increment_once_packet(0x0114 / 4, &arguments);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MmeRamLoad {
            ram: MaxwellThreeDMmeRam::Instruction,
            error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                limit: MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS,
            },
            ..
        })
    ));
    assert_ne!(channel.three_d_mut(), &before);

    let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2);
    arguments.push(0);
    arguments.resize(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2, 0);
    let decoded = increment_once_packet(0x011c / 4, &arguments);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MmeRamLoad {
            ram: MaxwellThreeDMmeRam::StartAddress,
            error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                limit: MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
            },
            ..
        })
    ));
    assert_ne!(channel.three_d_mut(), &before);
}

#[test]
fn mme_macro_executes_captured_code_and_emits_validated_methods() {
    let mut channel = three_d_channel();
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
    let dispatch = dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        argument,
    )
    .unwrap();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "CALL_MME_MACRO"
    );

    let point_size = channel.three_d().raster().point_size();
    assert_eq!(point_size.raw(), Some(argument));

    assert_eq!(
        channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(0x1518)),
        Some(argument)
    );
}

#[test]
fn mme_emitted_methods_track_replay_and_bypass_the_same_shadow_registers_as_pushbuffers() {
    for mode in [0, 1, 2, 3] {
        let mut channel = three_d_channel();
        let macro_index = 5;
        let method_dword = 0x192c / 4;
        load_mme_program(
            &mut channel,
            macro_index,
            &[
                1 | (2 << 4) | (method_dword << 14),
                (4 << 4) | (1 << 7) | (1 << 11),
                0x11,
            ],
        );
        program_three_d(&mut channel, 0x192c, 1);
        program_three_d(&mut channel, 0x0124, mode);
        dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable);
        assert_eq!(register.raw(), Some(u32::from(mode == 3)), "mode {mode}");
        let shadow = channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x192c));
        assert_eq!(shadow, Some(u32::from(mode >= 2)), "mode {mode}");
        program_three_d(&mut channel, 0x0124, 3);
        program_three_d(&mut channel, 0x192c, 0);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
                .raw(),
            shadow
        );
    }
}

#[test]
fn mme_replay_restores_firmware_color_target_selection_after_temporary_override() {
    let mut channel = three_d_channel();
    let selection = channel.three_d().render_targets().color_target_selection();

    assert_eq!(selection.raw(), Some(0x0fac_6881));
    assert_eq!(selection.value().unwrap().raw(), 0x0fac_6881);
    let macro_index = 5;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            1 | (2 << 4) | ((0x121c / 4) << 14),
            (4 << 4) | (1 << 7) | (1 << 11),
            0x11,
        ],
    );
    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x0124, 3);
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .color_target_selection()
            .raw(),
        Some(0x0fac_6881)
    );
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_target_count()
            .raw(),
        Some(1)
    );
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_target_count()
            .value(),
        Some(&MaxwellThreeDDepthTargetCount::One)
    );
    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x1538, 0);
    program_three_d(&mut channel, 0x0124, 3);
    program_three_d(&mut channel, 0x1538, 0);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_target_count()
            .raw(),
        Some(1)
    );
}

#[test]
fn shadow_replay_restores_each_color_descriptor_field_to_verified_firmware_values() {
    let mut channel = three_d_channel();
    for target in 0..8 {
        for (offset, expected) in [
            (0, 0),
            (4, 0),
            (8, 0x400),
            (0x0c, 0x300),
            (0x10, if target == 0 { 0xcf } else { 0 }),
            (0x14, 0x40),
            (0x18, 1),
            (0x1c, 0),
            (0x20, 0),
        ] {
            let method = 0x0800 + target * 0x40 + offset;
            program_three_d(&mut channel, 0x0124, 2);
            program_three_d(&mut channel, method, 0);
            program_three_d(&mut channel, 0x0124, 3);
            program_three_d(&mut channel, method, 0);
            assert_eq!(
                channel
                    .three_d_mut()
                    .raw_register_value(GpuMethodId(method)),
                Some(expected),
                "{method:#x}"
            );
        }
    }
}

#[test]
fn mme_execution_crosses_instruction_windows_and_preserves_sparse_addresses() {
    for start in [63, u32::MAX - 4] {
        let mut channel = three_d_channel();
        let macro_index = 5;
        let set_method = 1 | (2 << 4) | ((0x1518 / 4) << 14);
        let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
        dispatch_incrementing(&mut channel, 0x011c / 4, &[macro_index, start]).unwrap();
        dispatch_increment_once(
            &mut channel,
            0x0114 / 4,
            &[start, set_method, send_parameter_and_exit, 0x11],
        )
        .unwrap();
        let before = channel.three_d_mut().mme().clone();
        let value = 2.5_f32.to_bits();
        dispatch_method(&mut channel, (0x3800 + macro_index * 8) / 4, value).unwrap();
        assert_eq!(channel.three_d().raster().point_size().raw(), Some(value));
        let mme = channel.three_d_mut().mme();
        assert_eq!(mme.instruction_count(), 3);
        assert!(
            mme.instruction(MaxwellThreeDMmeRamAddress::new(start - 1))
                .is_none()
        );
        assert!(
            mme.instruction(MaxwellThreeDMmeRamAddress::new(start + 3))
                .is_none()
        );
        dispatch_incrementing(&mut channel, 0x0114 / 4, &[start, 0x11]).unwrap();
        assert_eq!(
            before
                .instruction(MaxwellThreeDMmeRamAddress::new(start))
                .unwrap()
                .raw(),
            Some(set_method)
        );
        assert_eq!(channel.three_d_mut().mme().instruction_count(), 3);
        // Replacing an uploaded instruction must replace its executable form
        // too, while an older retained program still owns its original words.
        let send_zero_and_exit = (4 << 4) | (1 << 7);
        dispatch_increment_once(
            &mut channel,
            0x0114 / 4,
            &[start, set_method, send_zero_and_exit],
        )
        .unwrap();
        let next = 4.25_f32.to_bits();
        dispatch_method(&mut channel, (0x3800 + macro_index * 8) / 4, next).unwrap();
        assert_eq!(channel.three_d().raster().point_size().raw(), Some(0));
        assert_eq!(
            before
                .instruction(MaxwellThreeDMmeRamAddress::new(start + 1))
                .unwrap()
                .raw(),
            Some(send_parameter_and_exit)
        );
    }
}

#[test]
fn mme_call_data_supplies_additional_parameters() {
    let mut channel = three_d_channel();
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
    let dispatch = dispatch_incrementing(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        &[0xdead_beef, argument],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "CALL_MME_DATA"
    );

    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(argument)
    );
}

#[test]
fn mme_reads_polygon_mode_reset_bits_until_guest_programming_overrides_them() {
    let mut channel = three_d_channel();

    for (method, register) in [
        (0x0dac, MaxwellThreeDFixedFunctionRegister::FrontPolygonMode),
        (0x0db0, MaxwellThreeDFixedFunctionRegister::BackPolygonMode),
    ] {
        let raw = channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(method))
            .unwrap();

        assert_eq!(Some(raw), Some(0x1b02));
        assert_eq!(Some(&raw), Some(&0x1b02));

        let typed = channel.three_d().fixed_function().register(register);

        assert_eq!(typed.raw(), Some(0x1b02));
        assert_eq!(
            typed.value(),
            Some(&MaxwellThreeDFixedFunctionValue::PolygonMode(
                MaxwellThreeDPolygonMode::Fill
            ))
        );
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
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();

    assert_eq!(channel.three_d(), &before_call);

    program_three_d(&mut channel, 0x0dac, 0x1b02);
    let raw = channel
        .three_d_mut()
        .raw_register_value(GpuMethodId(0x0dac))
        .unwrap();
    assert_eq!(Some(raw), Some(0x1b02));
    let typed = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::FrontPolygonMode);

    assert_eq!(
        typed.value(),
        Some(&MaxwellThreeDFixedFunctionValue::PolygonMode(
            MaxwellThreeDPolygonMode::Fill,
        ))
    );
}

#[test]
fn mme_reads_pipeline_header_and_binding_resets_and_writes_override_one_slot() {
    let mut channel = three_d_channel();

    for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
        let method = 0x2000 + pipeline as u32 * 0x40;
        let header = [0, 0x11, 0x20, 0x30, 0x40, 0x51][pipeline];
        let group = [0_u8, 0, 1, 2, 3, 4][pipeline];
        let stage = [
            MaxwellShaderStage::VertexCullBeforeFetch,
            MaxwellShaderStage::Vertex,
            MaxwellShaderStage::TessellationInit,
            MaxwellShaderStage::Tessellation,
            MaxwellShaderStage::Geometry,
            MaxwellShaderStage::Pixel,
        ][pipeline];
        let raw = channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(method))
            .unwrap();

        assert_eq!(Some(raw), Some(header));
        assert_eq!(Some(&raw), Some(&header));

        let binding = &channel.three_d().shader_bindings().pipeline()[pipeline];

        assert_eq!(binding.enabled().raw(), Some(header));
        assert_eq!(binding.enabled().value(), Some(&(header & 1 != 0)));

        assert_eq!(binding.stage().raw(), Some(header));
        assert_eq!(binding.stage().value(), Some(&stage));

        assert_eq!(binding.group().raw(), Some(u32::from(group)));
        assert_eq!(binding.group().value(), Some(&group));

        let raw_binding = channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(method + 0x10))
            .unwrap();

        assert_eq!(Some(raw_binding), Some(u32::from(group)));
        assert_eq!(Some(&raw_binding), Some(&u32::from(group)));
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
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();

    assert_eq!(channel.three_d(), &before_call);

    let before_invalid = channel.three_d().clone();
    let invalid = packet(0x20c0 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_PIPELINE_SHADER",
            ..
        }) if source.method() == GpuMethodId(0x20c0)
    ));
    assert_eq!(channel.three_d(), &before_invalid);

    program_three_d(&mut channel, 0x20c0, 0x41);
    let raw = channel
        .three_d_mut()
        .raw_register_value(GpuMethodId(0x20c0))
        .unwrap();
    assert_eq!(Some(raw), Some(0x41));
    let binding = &channel.three_d().shader_bindings().pipeline()[3];

    assert_eq!(binding.enabled().value(), Some(&true));

    assert_eq!(binding.stage().value(), Some(&MaxwellShaderStage::Geometry));
    assert_eq!(binding.group().value(), Some(&2));
    assert_eq!(binding.effective_group(), Some(2));
    assert!(
        channel.three_d().shader_bindings().stage_visibility(2)
            [MaxwellShaderStage::Geometry as usize]
    );
    program_three_d(&mut channel, 0x20d0, 6);
    let binding = &channel.three_d().shader_bindings().pipeline()[3];

    assert_eq!(binding.effective_group(), Some(6));
}

#[test]
fn mme_shadow_ram_control_tracks_bypasses_replays_and_is_atomic() {
    let mut channel = three_d_channel();

    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().value(),
        Some(&MaxwellThreeDMmeShadowRamControl::MethodTrack)
    );

    program_three_d(&mut channel, 0x0124, 0);
    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().value(),
        Some(&MaxwellThreeDMmeShadowRamControl::MethodTrack)
    );
    program_three_d(&mut channel, 0x0dac, 0x1b02);
    let tracked = channel
        .three_d_mut()
        .mme()
        .shadow_value(GpuMethodId(0x0dac))
        .unwrap();
    assert_eq!(tracked, 0x1b02);

    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x0dac, 0x1b01);
    assert_eq!(
        channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x0dac)),
        Some(0x1b02)
    );
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(0x0dac)),
        Some(0x1b01)
    );

    program_three_d(&mut channel, 0x0124, 3);
    let dispatch = dispatch_method(&mut channel, 0x0dac / 4, 0xffff_ffff).unwrap();
    let replayed_source = dispatch.methods()[0].method().source();
    assert_eq!(replayed_source.argument(), 0x1b02);
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(0x0dac)),
        Some(0x1b02)
    );

    // The control method itself consumes the non-shadowed argument, allowing
    // the stream to leave replay mode.
    program_three_d(&mut channel, 0x0124, 1);
    assert_eq!(
        channel.three_d_mut().mme().shadow_ram_control().value(),
        Some(&MaxwellThreeDMmeShadowRamControl::MethodTrackWithFilter)
    );
    program_three_d(&mut channel, 0x0dac, 0x1b00);
    assert_eq!(
        channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x0dac)),
        Some(0x1b00)
    );

    let before_invalid = channel.clone();
    let invalid = packet(0x0124 / 4, 4);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            defined_mask: 3,
            ..
        })
    ));
    assert_eq!(channel, before_invalid);
}

#[test]
fn depth_layer_reset_is_shared_by_live_state_and_mme_shadow_replay() {
    let mut channel = three_d_channel();
    let reset_state = channel.three_d().clone();
    let layer = reset_state.render_targets().depth_stencil().layer();

    assert_eq!(layer.raw(), Some(0));
    assert_eq!(layer.value(), Some(&0));

    let raw = channel
        .three_d_mut()
        .raw_register_value(GpuMethodId(0x179c))
        .unwrap();

    assert_eq!(Some(raw), Some(0));
    // Shadow storage is sparse: an unwritten entry resolves through the shared
    // reset table, as exercised by replay below, without allocating an entry.
    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x179c))
            .is_none()
    );
    // The reset is not a target binding and does not initialize other fields.

    assert!(
        fixture_bindings(
            &reset_state,
            resource_address_space().memory(),
            &mut crate::test_memory::apply
        )
        .unwrap()
        .resources()
        .next()
        .is_none()
    );

    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x179c, 3);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&3)
    );
    program_three_d(&mut channel, 0x0124, 3);
    let replay = dispatch_method(&mut channel, 0x179c / 4, u32::MAX).unwrap();
    assert_eq!(replay.methods()[0].method().source().argument(), 0);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&0)
    );

    // An explicitly tracked layer still overrides the reset on later replay.
    program_three_d(&mut channel, 0x0124, 0);
    program_three_d(&mut channel, 0x179c, 2);
    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x179c, 4);
    program_three_d(&mut channel, 0x0124, 3);
    program_three_d(&mut channel, 0x179c, u32::MAX);
    assert_eq!(
        channel
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&2)
    );
    assert_eq!(
        reset_state.render_targets().depth_stencil().layer().value(),
        Some(&0)
    );
    assert_eq!(
        three_d_channel()
            .three_d()
            .render_targets()
            .depth_stencil()
            .layer()
            .value(),
        Some(&0)
    );
}

#[test]
fn mme_reset_tracking_preserves_a_complete_color_target_across_passthrough_and_replay() {
    let mut channel = three_d_channel();
    let initial = [
        (0x0800, 5),
        (0x0804, 0x00f0_0000),
        (0x0808, 1280),
        (0x080c, 720),
        (0x0810, 0xd5),
        (0x0814, 0x40),
        (0x0818, 1),
        (0x081c, 0x000f_0000),
    ];
    for (method, argument) in initial {
        program_three_d(&mut channel, method, argument);
    }
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );
    let layer = channel.three_d().render_targets().color()[0].layer();

    assert_eq!(layer.value(), Some(&0));
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(0x0820)),
        Some(0)
    );

    program_three_d(&mut channel, 0x0124, 2);
    for (method, _) in initial {
        program_three_d(&mut channel, method, 0);
    }
    assert_ne!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );

    program_three_d(&mut channel, 0x0124, 3);
    for (method, _) in initial {
        program_three_d(&mut channel, method, u32::MAX);
    }
    assert_eq!(
        channel.three_d().render_targets().color()[0].readiness(true),
        MaxwellThreeDAttachmentReadiness::Ready
    );
    for (method, argument) in initial {
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(method)),
            Some(argument)
        );
    }
}

#[test]
fn mme_shadow_replay_without_a_tracked_value_fails_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0124, 3);
    let before = channel.clone();
    let decoded = packet(0x0db4 / 4, 1);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MmeShadowRam {
            error: MaxwellThreeDMmeShadowRamError::ReplayRegisterUnavailable {
                method_dword: 0x036d,
            },
            ..
        })
    ));
    assert_eq!(channel, before);
}

#[test]
fn mme_shadow_replay_uses_the_verified_window_origin_reset() {
    let mut channel = three_d_channel();

    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x13ac))
            .is_none()
    );
    let reset = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowOrigin);

    assert_eq!(reset.raw(), Some(0));

    program_three_d(&mut channel, 0x0124, 3);
    let dispatch = dispatch_method(&mut channel, 0x13ac / 4, 0x10).unwrap();

    let source = dispatch.methods()[0].method().source();
    assert_eq!(source.argument(), 0);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_WINDOW_ORIGIN"
    );
    let window_origin = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowOrigin);

    assert_eq!(window_origin.raw(), Some(0));
    assert_eq!(
        window_origin.value(),
        Some(&MaxwellThreeDFixedFunctionValue::Mask(0))
    );

    // Replay consumes the immutable reset without turning it into a tracked
    // guest write.
    assert!(
        channel
            .three_d_mut()
            .mme()
            .shadow_value(GpuMethodId(0x13ac))
            .is_none()
    );
}

#[test]
fn mme_scratch_reads_verified_initial_values_without_defaulting_unknown_registers() {
    for index in [0_u32, 40, 127, 128, 255] {
        let mut channel = three_d_channel();
        let method = 0x3400 + index * 4;
        let read_and_exit = 5 | (1 << 4) | (1 << 7) | (2 << 8) | ((method / 4) << 14);
        load_mme_program(&mut channel, 3, &[read_and_exit, 0x11]);
        let before = channel.three_d().clone();
        let result = dispatch_method(&mut channel, 0x3818 / 4, 0);
        if index < 128 {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(MaxwellEngineDispatchError::MmeExecution {
                    error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable { .. },
                    ..
                })
            ));
        }
        assert_eq!(channel.three_d(), &before);
        program_three_d(&mut channel, method, 0xcafe_babe);
        dispatch_method(&mut channel, 0x3818 / 4, 0).unwrap();
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(method)),
            Some(0xcafe_babe)
        );
    }
}

#[test]
fn mme_shadow_scratch_family_is_indexed_readable_and_keeps_valid_prefix() {
    let mut channel = three_d_channel();

    for (index, value) in [(0_u8, 0_u32), (1, 0xfeed_beef), (u8::MAX, u32::MAX)] {
        let method = 0x3400 + u32::from(index) * 4;
        let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_MME_SHADOW_SCRATCH"
        );

        let scratch = channel
            .three_d_mut()
            .mme()
            .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(index))
            .unwrap();

        assert_eq!(scratch.raw(), Some(value));
        assert_eq!(scratch.value(), Some(&value));

        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(method)),
            Some(value)
        );
    }
    assert_eq!(channel.three_d_mut().mme().shadow_scratch_count(), 3);
    assert_eq!(MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT, 256);

    let macro_index = 5;
    let read_scratch_and_exit = 5 | (1 << 4) | (1 << 7) | (2 << 8) | ((0x3400 / 4) << 14);
    load_mme_program(&mut channel, macro_index, &[read_scratch_and_exit, 0x11]);
    let before_call = channel.three_d().clone();
    dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 0).unwrap();

    assert_eq!(channel.three_d(), &before_call);

    let before_invalid = channel.clone();
    let invalid_suffix = incrementing_packet(0x37fc / 4, &[0x1234_5678, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_suffix),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index: 0 },
            ..
        })
    ));
    assert_ne!(channel, before_invalid);
}

#[test]
fn falcon_call_four_applies_masked_pgraph_writes_and_signals_completion() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x3404, 0x00f0_5500);
    program_three_d(&mut channel, 0x3408, 0x00ff_0f00);

    let dispatch = dispatch_method(&mut channel, 0x2310 / 4, 0x0041_8800).unwrap();
    let source = dispatch.methods()[0].method().source();
    let address = MaxwellThreeDFalconRegisterAddress::try_new(0x0041_8800).unwrap();
    let expected =
        MaxwellThreeDFalconMaskedRegisterWrite::new(address, 0x00f0_5500, 0x00ff_0f00, source);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_FALCON04"
    );

    assert_eq!(
        channel.three_d().falcon().last_masked_write(),
        Some(expected)
    );
    let register = channel.three_d().falcon().register(address).unwrap();
    assert_eq!(register.known_mask(), 0x00ff_0f00);
    assert_eq!(register.value(), 0x00f0_0500);

    let completion = channel
        .three_d_mut()
        .mme()
        .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(0))
        .unwrap();
    assert_eq!(completion.raw(), Some(1));

    assert_eq!(
        channel
            .three_d_mut()
            .raw_register_value(GpuMethodId(0x3400)),
        Some(1)
    );

    program_three_d(&mut channel, 0x3404, 0xaa00_00cc);
    program_three_d(&mut channel, 0x3408, 0xff00_00ff);
    program_three_d(&mut channel, 0x2310, address.raw());
    let register = channel.three_d().falcon().register(address).unwrap();
    assert_eq!(register.known_mask(), 0xffff_0fff);
    assert_eq!(register.value(), 0xaaf0_05cc);
}

#[test]
fn captured_deko_write_hardware_register_macro_reaches_falcon_call_four() {
    let mut channel = three_d_channel();
    let macro_index = 2;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            0x0011_0071,
            0x0740_0251,
            0x0000_0331,
            0x0000_1041,
            0x0000_1841,
            0x0231_0021,
            0x0000_0841,
            0x0340_0115,
            0xffff_c911,
            0xffff_8817,
            0x0010_00f1,
            0x0000_0011,
        ],
    );

    let dispatch = dispatch_increment_once(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        &[0x0041_8800, 0, 0x0180_0000],
    )
    .unwrap();

    assert_eq!(dispatch.synchronization_operations().len(), 1);
    let address = MaxwellThreeDFalconRegisterAddress::try_new(0x0041_8800).unwrap();
    let write = channel.three_d().falcon().last_masked_write().unwrap();
    assert_eq!(write.address(), address);
    assert_eq!(write.value(), 0);
    assert_eq!(write.mask(), 0x0180_0000);
    assert_eq!(
        channel
            .three_d_mut()
            .mme()
            .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(0))
            .and_then(MaxwellThreeDRegister::raw),
        Some(1)
    );
}

#[test]
fn falcon_call_four_rejects_incomplete_or_unaligned_transactions_atomically() {
    let mut channel = three_d_channel();

    let missing = packet(0x2310 / 4, 0x0041_8800);
    let before_missing = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &missing),
        Err(MaxwellEngineDispatchError::FalconFirmware {
            error: MaxwellThreeDFalconError::MissingFirmwareArgument { index: 1 },
            ..
        })
    ));
    assert_eq!(channel, before_missing);

    program_three_d(&mut channel, 0x3404, 1);
    program_three_d(&mut channel, 0x3408, 1);
    let unaligned = packet(0x2310 / 4, 0x0041_8801);
    let before_unaligned = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &unaligned),
        Err(MaxwellEngineDispatchError::FalconFirmware {
            error: MaxwellThreeDFalconError::UnalignedRegisterAddress {
                address: 0x0041_8801
            },
            ..
        })
    ));
    assert_eq!(channel, before_unaligned);

    let invalid_suffix = incrementing_packet(0x2310 / 4, &[0x0041_8800, 0]);
    let before_suffix = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_suffix),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x2314)
    ));
    assert_ne!(channel, before_suffix);
}

#[test]
fn mme_emitted_draw_keeps_the_exact_method_state_snapshot() {
    let mut channel = three_d_channel();
    let macro_index = 6;
    let draw_method_dword = 0x0d78 / 4;
    let set_method = 1 | (2 << 4) | (draw_method_dword << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[set_method, send_parameter_and_exit, 0x11],
    );

    let dispatch =
        dispatch_method(&mut channel, (0x3800 + u32::from(macro_index) * 8) / 4, 3).unwrap();
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
    let mut channel = three_d_channel();

    let data = packet(0x3804 / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &data),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
            ..
        })
    ));

    let missing = packet(0x3800 / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &missing),
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
        dispatch_first(&mut channel, &call),
        Err(MaxwellEngineDispatchError::MmeExecution {
            error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall {
                method_dword: 0x0e00,
            },
            ..
        })
    ));
    assert_ne!(channel.three_d(), &before);
}

#[test]
fn mme_register_reads_and_execution_limit_fail_typed_and_atomically() {
    let mut channel = three_d_channel();

    let read_macro = 3;
    let read_unset_register = 5 | (1 << 4) | (2 << 8) | (0x036d << 14);
    load_mme_program(&mut channel, read_macro, &[read_unset_register]);
    let before = channel.three_d().clone();
    let call = packet((0x3800 + u32::from(read_macro) * 8) / 4, 0);
    assert!(matches!(
        dispatch_first(&mut channel, &call),
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
        dispatch_first(&mut channel, &call),
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
fn mme_finite_loop_may_retire_more_instructions_than_the_method_output_limit() {
    let mut channel = three_d_channel();

    let decrement_r1 = 1 | (1 << 4) | (1 << 8) | (1 << 11) | ((-1_i32 as u32) << 14);
    let branch_to_decrement_while_r1_nonzero =
        7 | (1 << 4) | (1 << 5) | (1 << 11) | ((-1_i32 as u32) << 14);
    let exit = 1 | (1 << 4) | (1 << 7);
    let delay_slot = 1 | (1 << 4);
    let macro_index = 14;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            decrement_r1,
            branch_to_decrement_while_r1_nonzero,
            exit,
            delay_slot,
        ],
    );
    dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        20_000,
    )
    .unwrap();
}

#[test]
fn mme_finite_loop_may_emit_more_than_the_old_host_budget() {
    let mut channel = three_d_channel();

    let set_nop_method = 1 | (2 << 4) | ((0x0100 / 4) << 14);
    let decrement_r1_and_send = 1 | (4 << 4) | (1 << 8) | (1 << 11) | ((-1_i32 as u32) << 14);
    let branch_to_decrement_while_r1_nonzero =
        7 | (1 << 4) | (1 << 5) | (1 << 11) | ((-1_i32 as u32) << 14);
    let exit = 1 | (1 << 4) | (1 << 7);
    let delay_slot = 1 | (1 << 4);
    let macro_index = 15;
    load_mme_program(
        &mut channel,
        macro_index,
        &[
            set_nop_method,
            decrement_r1_and_send,
            branch_to_decrement_while_r1_nonzero,
            exit,
            delay_slot,
        ],
    );
    dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        5_000,
    )
    .unwrap();
}

#[test]
fn vertex_assembly_controls_are_typed_source_preserving_and_isolated() {
    let mut channel = three_d_channel();
    let input_before = channel.three_d().vertex_input().clone();
    let two_d_before = channel.two_d().clone();

    let dispatch = dispatch_method(&mut channel, 0x1610 / 4, 0x0e).unwrap();
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

    assert_eq!(defaults.raw(), Some(0x0e));

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

    let dispatch = dispatch_method(&mut channel, 0x164c / 4, 0x1000).unwrap();
    let input = channel.three_d().vertex_input();
    let vertex_id = input.assembly().vertex_id_uses_array_start();

    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_DA_OUTPUT"
    );

    assert_eq!(vertex_id.raw(), Some(0x1000));
    assert_eq!(
        vertex_id.value(),
        Some(&MaxwellThreeDVertexIdUsesArrayStart::Enabled)
    );

    assert_eq!(input.streams(), input_before.streams());
    assert_eq!(input.attributes(), input_before.attributes());
    assert_eq!(input.index(), input_before.index());
    assert_eq!(input.primitive(), input_before.primitive());
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn vertex_assembly_controls_update_typed_state() {
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x1610, 0x0e);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .assembly()
            .attribute_defaults()
            .raw(),
        Some(0x0e)
    );

    program_three_d(&mut channel, 0x164c, 0x1000);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .assembly()
            .vertex_id_uses_array_start()
            .raw(),
        Some(0x1000)
    );

    program_three_d(&mut channel, 0x1610, 0x3f);
}

#[test]
fn vertex_stream_attributes_and_begin_state_remain_unresolved_and_typed() {
    let mut channel = three_d_channel();
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
        dispatch_method(&mut channel, method / 4, argument).unwrap();
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
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x1618, 4);
    let primitive = channel.three_d().vertex_input().primitive();
    assert!(primitive.is_open());
    assert_eq!(primitive.active_begin().unwrap().topology(), 4);

    let dispatch = dispatch_method(&mut channel, 0x1614 / 4, 0).unwrap();
    assert_eq!(dispatch.methods()[0].metadata().method_name(), "END");

    assert!(dispatch.operations().is_empty());

    let primitive = channel.three_d().vertex_input().primitive();
    assert!(!primitive.is_open());
    assert_eq!(primitive.active_begin(), None);
    assert_eq!(primitive.begin().value().unwrap().topology(), 4);

    assert_eq!(primitive.end().raw(), Some(0));
    assert_eq!(primitive.end().value(), Some(&false));

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1610 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "END",
            ..
        }) if source.method() == GpuMethodId(0x1614) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    dispatch_incrementing(&mut channel, 0x1614 / 4, &[1, 5]).unwrap();
    let primitive = channel.three_d().vertex_input().primitive();
    assert!(primitive.is_open());
    assert_eq!(primitive.end().value(), Some(&true));
    assert_eq!(primitive.active_begin().unwrap().topology(), 5);
}

#[test]
fn begin_instance_modes_reset_advance_and_preserve_the_instance_sequence() {
    let mut channel = three_d_channel();

    // NVIDIA's public BEGIN layout defines FIRST=0, SUBSEQUENT=1 and
    // UNCHANGED=2 in bits 27:26.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3045-L3048
    program_three_d(&mut channel, 0x1618, 4);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        0
    );

    program_three_d(&mut channel, 0x1618, 4 | (1 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        1
    );
    program_three_d(&mut channel, 0x1618, 4 | (1 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        2
    );

    program_three_d(&mut channel, 0x1618, 4 | (2 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        2
    );

    program_three_d(&mut channel, 0x1618, 4);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        0
    );
}

#[test]
fn malformed_vertex_suffix_preserves_applied_prefix() {
    let mut channel = three_d_channel();
    let malformed = incrementing_packet(0x1c00 / 4, &[0x1010, 0, 0x2000, 1, 0x2000]);
    let before = channel.three_d().clone();
    assert!(matches!(
        dispatch_first(&mut channel, &malformed),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
    ));
    assert_ne!(channel.three_d(), &before);
}

#[test]
fn vertex_stream_ranges_may_be_reprogrammed_across_packets_before_consumption() {
    let mut channel = three_d_channel();

    for (method, argument) in [
        (0x1c00, 0x1018),
        (0x1c04, 0),
        (0x1c08, 0x093f_5000),
        (0x1f00, 0),
        (0x1f04, 0x093f_6fdf),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    // The lower address half is commonly written before the replacement
    // limit. A move in the opposite direction therefore creates a legal,
    // short-lived address > old-limit snapshot between these packets.
    dispatch_incrementing(&mut channel, 0x1c00 / 4, &[0x1018, 0, 0x0941_3000]).unwrap();

    dispatch_incrementing(&mut channel, 0x1f00 / 4, &[0, 0x0941_6fbf]).unwrap();

    let stream = &channel.three_d().vertex_input().streams()[0];
    assert_eq!(
        stream.address().map(|address| address.get()),
        Some(0x0941_3000)
    );
    assert_eq!(stream.limit().map(|limit| limit.get()), Some(0x0941_6fbf));
}

#[test]
fn shader_bindings_snapshot_selectors_and_preserve_stage_visibility() {
    let mut channel = three_d_channel();
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
        dispatch_method(&mut channel, method / 4, argument).unwrap();
    }

    let bindings = channel.three_d().shader_bindings();
    let constant = bindings.groups()[2].constant_buffers()[3].unwrap();
    assert!(constant.enabled());
    assert_eq!(constant.address().unwrap().get(), 0x4000);
    assert_eq!(constant.size(), Some(0x100));
    assert!(bindings.stage_visibility(2)[MaxwellShaderStage::Vertex as usize]);
    assert_eq!(bindings.texture_headers().address().unwrap().get(), 0x8000);
    assert_eq!(bindings.samplers().maximum_index().value(), Some(&7));
    assert_eq!(
        bindings.sampler_binding().value(),
        Some(&MaxwellThreeDSamplerBindingMode::ViaTextureHeader)
    );
}

#[test]
fn program_region_is_source_preserving_and_only_active_for_shader_pipelines() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    let lower_dispatch = dispatch_method(&mut channel, 0x160c / 4, 0).unwrap();
    let region = channel.three_d().shader_bindings().program_region();
    assert_eq!(
        lower_dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_B"
    );

    assert!(region.address().is_none());
    assert_eq!(region.address_lower().raw(), Some(0));

    let upper_dispatch = dispatch_method(&mut channel, 0x1608 / 4, 4).unwrap();
    let region = channel.three_d().shader_bindings().program_region();
    assert_eq!(
        upper_dispatch.methods()[0].metadata().method_name(),
        "SET_PROGRAM_REGION_A"
    );
    assert_eq!(region.address_upper().raw(), Some(4));

    assert_eq!(region.address().unwrap().get(), 0x0000_0004_0000_0000);
    assert_eq!(channel.two_d(), &two_d_before);

    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x160c, 0x1000);
}

#[test]
fn spa_version_is_shared_source_preserving_and_only_active_for_shader_pipelines() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    let dispatch = dispatch_method(&mut channel, 0x0310 / 4, 0x0503).unwrap();
    let method = dispatch.methods()[0];
    let register = channel
        .three_d()
        .shader_bindings()
        .program_region()
        .spa_version();
    let value = *register.value().unwrap();
    let _: MaxwellComputeSpaVersion = value;

    assert_eq!(method.metadata().method_name(), "SET_SPA_VERSION");

    assert!(dispatch.operations().is_empty());
    assert_eq!(value.major(), 5);
    assert_eq!(value.minor(), 3);
    assert_eq!(value.raw(), 0x0503);

    assert_eq!(register.raw(), Some(0x0503));

    assert_eq!(channel.two_d(), &two_d_before);

    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x0310, 0x0400);
}

#[test]
fn spa_version_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0310, 0x0503);

    for argument in [0x1_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0310 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_SPA_VERSION",
                reason: "reserved bits are set",
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x0310 / 4, &[0x0503, 0x1_0000]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. })
            if source.method() == GpuMethodId(0x0310)
                && source.argument() == 0x1_0000
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_program_region_upper_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1608, 4);
    program_three_d(&mut channel, 0x160c, 0);

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1608 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_ATTRIBUTE_DEFAULT",
            ..
        }) if source.method() == GpuMethodId(0x1610)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_stream_substitute_address_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    let lower_dispatch = dispatch_method(&mut channel, 0x0f88 / 4, 0x082c_3000).unwrap();
    let lower_method = lower_dispatch.methods()[0];
    assert_eq!(
        lower_method.metadata().method_name(),
        "SET_VERTEX_STREAM_SUBSTITUTE_B"
    );

    let substitute = channel.three_d().vertex_input().stream_substitute();
    assert!(substitute.address().is_none());
    assert_eq!(substitute.address_lower().raw(), Some(0x082c_3000));

    let upper_dispatch = dispatch_method(&mut channel, 0x0f84 / 4, 0x7f).unwrap();
    let upper_method = upper_dispatch.methods()[0];
    assert_eq!(
        upper_method.metadata().method_name(),
        "SET_VERTEX_STREAM_SUBSTITUTE_A"
    );

    let substitute = channel.three_d().vertex_input().stream_substitute();
    assert_eq!(substitute.address_upper().raw(), Some(0x7f));

    assert_eq!(substitute.address().unwrap().get(), 0x7f_082c_3000);
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn invalid_vertex_stream_substitute_upper_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f84, 4);
    program_three_d(&mut channel, 0x0f88, 0x1000);

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0f84 / 4, &[argument, 0x082c_3000]);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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
    let decoded = incrementing_packet(0x0f84 / 4, &[0, 0x082c_3000, 0, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0f94)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn program_offset_and_register_count_updates_preserve_slot_state() {
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x2044, 0x7f730);
    program_three_d(&mut channel, 0x204c, 4);

    program_three_d(&mut channel, 0x2040, 0x11);

    program_three_d(&mut channel, 0x2044, 0x7f830);
    program_three_d(&mut channel, 0x204c, 5);

    program_three_d(&mut channel, 0x2040, 0x10);
    program_three_d(&mut channel, 0x2044, 0x7f930);
    program_three_d(&mut channel, 0x204c, 6);
}

#[test]
fn tessellation_lod_family_is_source_preserving() {
    let mut channel = three_d_channel();
    let levels = [
        MaxwellThreeDTessellationLod::OuterU0OrDensity,
        MaxwellThreeDTessellationLod::OuterV0OrDetail,
        MaxwellThreeDTessellationLod::OuterU1OrW0,
        MaxwellThreeDTessellationLod::OuterV1,
        MaxwellThreeDTessellationLod::InnerU,
        MaxwellThreeDTessellationLod::InnerV,
    ];

    for (index, level) in levels.into_iter().enumerate() {
        let method = 0x0324 + index as u32 * 4;
        let argument = if index == 5 {
            u32::MAX
        } else {
            0x3f80_0000 + index as u32
        };
        let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TESSELLATION_LOD"
        );

        let register = channel.three_d().shader_bindings().tessellation_lod(level);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&argument));
    }

    program_three_d(&mut channel, 0x2080, 0x21);
    program_three_d(&mut channel, 0x0324, 0x4000_0000);

    program_three_d(&mut channel, 0x2080, 0x20);
    program_three_d(&mut channel, 0x0324, 0x4040_0000);
}

#[test]
fn render_target_index_offset_is_typed_source_preserving_and_conditionally_dependent() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for (argument, expected, enabled) in [
        (0, MaxwellThreeDRenderTargetIndexOffset::Disabled, false),
        (
            1,
            MaxwellThreeDRenderTargetIndexOffset::ByViewportIndex,
            true,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x11f0 / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let register = channel
            .three_d()
            .render_targets()
            .render_target_index_offset();

        assert_eq!(
            method.metadata().method_name(),
            "SET_OFFSET_RENDER_TARGET_INDEX"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.enabled(), enabled);

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));

        assert_eq!(channel.two_d(), &two_d_before);
    }

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x11f0 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
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

#[test]
fn opportunistic_early_z_threshold_retains_sources_and_rejects_reserved_encodings() {
    let mut channel = three_d_channel();
    for raw in (0..=19).chain([31]) {
        let dispatch = dispatch_method(&mut channel, 0x204 / 4, raw).unwrap();
        let register = channel
            .three_d()
            .shader_execution()
            .opportunistic_early_z_hysteresis();
        assert_eq!(register.value(), Some(&(raw as u8)));

        assert!(dispatch.ordered_operations().is_empty());
    }
    for raw in [20, 30, 32, u32::MAX] {
        assert!(dispatch_method(&mut channel, 0x204 / 4, raw).is_err());
    }
}

#[test]
fn viewport_firmware_defaults_are_shared_by_live_state_and_shadow_replay() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    for (method, register, raw) in [
        (
            0x192c,
            MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable,
            1,
        ),
        (
            0x193c,
            MaxwellThreeDFixedFunctionRegister::ViewportClipControl,
            0x2c1c,
        ),
    ] {
        let value = channel.three_d().fixed_function().register(register);
        assert_eq!(value.raw(), Some(raw));

        assert!(
            channel
                .three_d_mut()
                .mme()
                .shadow_value(GpuMethodId(method))
                .is_none()
        );
    }
    program_three_d(&mut channel, 0x0124, 3);
    for (method, raw) in [(0x192c, 1), (0x193c, 0x2c1c)] {
        let dispatch = dispatch_method(&mut channel, method / 4, 0).unwrap();
        assert_eq!(dispatch.methods()[0].method().source().argument(), raw);
    }
    program_three_d(&mut channel, 0x0124, 0);
    program_three_d(&mut channel, 0x192c, 0);
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
            .value(),
        Some(&MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(
            MaxwellThreeDViewportScaleOffsetEnable::Disabled
        ))
    );
}

struct NativeDrawFixture {
    space: MaxwellGpuAddressSpace,
    channel: MaxwellGpuChannel,
    shaders: Allocation,
    vertex: u64,
    fragment_header: [u32; 20],
}

fn native_draw_fixture() -> NativeDrawFixture {
    let mut space = resource_address_space();
    let vertices = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let color = crate::test_memory::allocate(space.memory().machine(), 0x10000);
    let shaders = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let mut map = |allocation: &Allocation, id, kind| {
        map_resource(&mut space, crate::test_memory::ranges(allocation), id, kind)
            .offset()
            .get()
    };
    let vertex = map(&vertices, 1, 0);
    let target = map(&color, 2, 0xfe);
    let code = map(&shaders, 3, 0);
    let exit = 0xe300_0000_0007_000f_u64;
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    crate::test_memory::write(
        &shaders,
        0,
        &header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain([exit, 0, 0, 0].into_iter().flat_map(u64::to_le_bytes))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    header = [0; 20];
    header[0] = 0x0002_5462;
    header[4] = 0x000f_f000;
    header[6] = 0xaa;
    header[13] = 0;
    header[18] = 1;
    crate::test_memory::write(
        &shaders,
        0x100,
        &header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(
                [0xe0c3_ff88_0077_ff00, exit, 0, 0]
                    .into_iter()
                    .flat_map(u64::to_le_bytes),
            )
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    for (method, value) in [
        (0x1608, (code >> 32) as u32),
        (0x160c, code as u32),
        (0x2004, 0),
        (0x200c, 4),
        (0x2044, 0x100),
        (0x204c, 4),
    ] {
        program_three_d(&mut channel, method, value);
    }
    NativeDrawFixture {
        space,
        channel,
        shaders,
        vertex,
        fragment_header: header,
    }
}

#[test]
fn native_draw_retains_shader_outputs_with_constant_color_registers_programmed() {
    use crate::execution::MaxwellFrontend;
    use nixe_gpu::native::NativeCommand;
    let NativeDrawFixture {
        space, mut channel, ..
    } = native_draw_fixture();
    let mut frontend = crate::MaxwellNativeFrontend::new(Default::default());
    let mut previous = None;
    for enabled in [false, true, false] {
        for (method, value) in [
            (0x0f40, u32::from(enabled)),
            (0x0f44, 1.0_f32.to_bits()),
            (0x0f48, 0),
            (0x0f4c, 0),
            (0x0f50, 1.0_f32.to_bits()),
        ] {
            program_three_d(&mut channel, method, value);
        }
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let draw = &dispatch.operations()[0];
        let work = frontend
            .lower_three_d(
                draw.state(),
                draw.trigger(),
                space.memory(),
                FrontendSubmissionId::new(1),
                None,
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Draw { state, .. } = work.command else {
            panic!("expected draw")
        };
        if let Some(previous) = previous {
            assert!(Arc::ptr_eq(&previous, &state));
        }
        previous = Some(state);
    }
    assert_eq!(frontend.statistics.prepared_draws, 1);
    assert_eq!(frontend.statistics.warm_draws, 2);
}

#[test]
fn native_draw_early_fragment_tests_follow_consumed_state_changes() {
    use crate::execution::MaxwellFrontend;
    use nixe_gpu::native::NativeCommand;
    let NativeDrawFixture {
        space, mut channel, ..
    } = native_draw_fixture();
    let mut frontend = crate::MaxwellNativeFrontend::new(Default::default());
    let mut previous: Option<Arc<nixe_gpu::native::GuestDrawState>> = None;
    for enabled in [false, true, true, false] {
        program_three_d(&mut channel, 0x0210, u32::from(enabled));
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let draw = &dispatch.operations()[0];
        let work = frontend
            .lower_three_d(
                draw.state(),
                draw.trigger(),
                space.memory(),
                FrontendSubmissionId::new(1),
                None,
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Draw { state, .. } = work.command else {
            panic!("expected draw")
        };
        assert_eq!(state.parameters.shader_state.force_early_z, enabled);
        if let Some(previous) = previous {
            assert_eq!(
                Arc::ptr_eq(&previous.parameters, &state.parameters),
                previous.parameters.shader_state.force_early_z == enabled
            );
        }
        previous = Some(state);
    }
}

#[test]
#[ignore = "Explicit command/resource CPU-stage saturation diagnostic; not game FPS"]
fn profile_native_frontend_draw_consumption() {
    use crate::execution::MaxwellFrontend;
    let NativeDrawFixture {
        space, mut channel, ..
    } = native_draw_fixture();
    let mut frontend = crate::MaxwellNativeFrontend::new(Default::default());
    for (phase, iterations) in [("warmup", 1000), ("sample", 100_000)] {
        let begin = std::time::Instant::now();
        for index in 0..iterations {
            program_three_d(&mut channel, 0x0d74, index % 32);
            let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
            let draw = &dispatch.operations()[0];
            std::hint::black_box(
                frontend
                    .lower_three_d(
                        draw.state(),
                        draw.trigger(),
                        space.memory(),
                        FrontendSubmissionId::new(u64::from(index) + 1),
                        None,
                        &mut crate::test_memory::access(space.memory().machine()),
                    )
                    .unwrap(),
            );
        }
        eprintln!(
            "command-resource-stage phase={phase} draws={iterations} seconds={}",
            begin.elapsed().as_secs_f64(),
        );
    }
    assert_eq!(frontend.statistics.draws, 101_000);
    assert_eq!(frontend.statistics.shader_captures, 1);
}

#[test]
fn native_frontend_reuses_immutable_draw_state_and_only_changes_arguments() {
    use crate::execution::MaxwellFrontend;
    use nixe_gpu::native::NativeCommand;
    let NativeDrawFixture {
        mut space,
        mut channel,
        shaders,
        vertex,
        fragment_header: mut header,
    } = native_draw_fixture();
    let exit = 0xe300_0000_0007_000f_u64;
    let mut frontend = crate::MaxwellNativeFrontend::new(Default::default());
    let mut previous = None;
    for first in 0..32 {
        program_three_d(&mut channel, 0x0d74, first);
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let draw = &dispatch.operations()[0];
        let work = frontend
            .lower_three_d(
                draw.state(),
                draw.trigger(),
                space.memory(),
                FrontendSubmissionId::new(u64::from(first) + 1),
                (first == 31).then_some((2, 5)),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Draw { state, arguments } = work.command else {
            panic!("expected native draw")
        };
        assert!(
            matches!(arguments,nixe_gpu::DrawArguments::NonIndexed {first_vertex, first_instance, instance_count, ..}
                if first_vertex == first && first_instance == if first == 31 { 2 } else { 0 }
                && instance_count == if first == 31 { 5 } else { 1 })
        );
        if let Some(old) = &previous {
            assert!(Arc::ptr_eq(old, &state));
        }
        previous = Some(state);
    }
    assert_eq!(frontend.statistics.prepared_draws, 1);
    assert_eq!(frontend.statistics.shader_captures, 1);
    assert_eq!(frontend.statistics.warm_draws, 31);
    let original = previous.unwrap();
    // Reconstructing equal image metadata must preserve draw lookup identity.
    let mut reconstructed = (*original).clone();
    for image in Arc::make_mut(&mut reconstructed.colors) {
        image.planes = Arc::from(image.planes.to_vec());
    }
    assert_eq!(original.as_ref(), &reconstructed);
    let mut lookup = nixe_gpu::WorkingSet::new(2, 4096);
    lookup.insert(Arc::clone(&original), 1, 512);
    assert_eq!(lookup.get(&reconstructed), Some(&1));

    // Every streamed vertex window is a distinct physical binding, while its
    // vertex layout and rasterization semantics retain one prepared identity.
    program_three_d(&mut channel, 0x1f04, (vertex + 0xfff) as u32);
    for offset in (16..=256).step_by(16) {
        program_three_d(&mut channel, 0x1c08, vertex as u32 + offset);
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let draw = &dispatch.operations()[0];
        let work = frontend
            .lower_three_d(
                draw.state(),
                draw.trigger(),
                space.memory(),
                FrontendSubmissionId::new(200 + u64::from(offset)),
                None,
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Draw { state, .. } = work.command else {
            panic!("expected draw")
        };
        assert!(!Arc::ptr_eq(&original, &state));
        assert!(Arc::ptr_eq(&original.parameters, &state.parameters));
        assert!(Arc::ptr_eq(&original.program, &state.program));
        assert!(Arc::ptr_eq(&original.colors, &state.colors));
        assert_ne!(
            original.vertices[0].backing.as_ref(),
            state.vertices[0].backing.as_ref()
        );
    }
    assert_eq!(frontend.statistics.prepared_parameters, 1);
    assert_eq!(frontend.statistics.reused_parameters, 16);
    assert_eq!(frontend.statistics.reused_programs, 16);
    // Intermediate state changes restored before consumption do not create a
    // new semantic preparation, even though their monotonic revisions advanced.
    program_three_d(&mut channel, 0x1160, 0x3820_0000 | (8 << 7));
    program_three_d(&mut channel, 0x1160, 0x3820_0000);
    let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
    let draw = &dispatch.operations()[0];
    let work = frontend
        .lower_three_d(
            draw.state(),
            draw.trigger(),
            space.memory(),
            FrontendSubmissionId::new(500),
            None,
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap();
    let NativeCommand::Draw { state, .. } = work.command else {
        panic!("expected draw")
    };
    assert!(Arc::ptr_eq(&original.parameters, &state.parameters));
    assert_eq!(frontend.statistics.prepared_parameters, 1);
    for phase in 0..2 {
        // Stream offsets affect vertex binding, but not shader input types.
        program_three_d(&mut channel, 0x1160, 0x3820_0000 | (((phase + 1) * 8) << 7));
        if phase == 1 {
            // An identical CPU write refreshes the memory proof while keeping
            // the immutable capture and compiled code reusable.
            crate::test_memory::write(&shaders, 0x50, &exit.to_le_bytes()).unwrap();
        }
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let draw = &dispatch.operations()[0];
        let work = frontend
            .lower_three_d(
                draw.state(),
                draw.trigger(),
                space.memory(),
                FrontendSubmissionId::new(100 + u64::from(phase)),
                None,
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Draw { state, .. } = work.command else {
            panic!("expected draw")
        };
        assert!(!Arc::ptr_eq(&original, &state));
        assert!(!Arc::ptr_eq(&original.parameters, &state.parameters));
        assert!(Arc::ptr_eq(&original.program.shader, &state.program.shader));
        assert_eq!(frontend.statistics.shader_captures, 1);
    }

    // Two live uniform bindings exercise both complete reuse and a changed
    // suffix after an equal prefix. Data epochs cannot change binding identity.
    let uniforms = crate::test_memory::allocate(space.memory().machine(), 0x1000);
    let uniform_address = map_resource(&mut space, crate::test_memory::ranges(&uniforms), 4, 0)
        .offset()
        .get();
    header[18] = 3;
    crate::test_memory::write(
        &shaders,
        0x100,
        &header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(
                [0x4c98_0784_0047_0000, 0x4c98_0788_0047_0001, 0, exit]
                    .into_iter()
                    .flat_map(u64::to_le_bytes),
            )
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (method, value) in [
        (0x2380, 0x100),
        (0x2384, (uniform_address >> 32) as u32),
        (0x2388, uniform_address as u32),
        (0x2430, 0x11),
        (0x2430, 0x21),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let draw = |channel: &mut MaxwellGpuChannel, frontend: &mut crate::MaxwellNativeFrontend| {
        let dispatch = dispatch_method(channel, 0x0d78 / 4, 3).unwrap();
        let trigger = &dispatch.operations()[0];
        let work = frontend
            .lower_three_d(
                trigger.state(),
                trigger.trigger(),
                space.memory(),
                FrontendSubmissionId::new(600),
                None,
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let NativeCommand::Draw { state, .. } = work.command else {
            panic!("expected draw")
        };
        state
    };
    let before = draw(&mut channel, &mut frontend);
    assert_eq!(before.program.bindings.len(), 2);
    crate::test_memory::write(&uniforms, 16, &1.0_f32.to_le_bytes()).unwrap();
    let rewritten = draw(&mut channel, &mut frontend);
    assert!(Arc::ptr_eq(&before.program, &rewritten.program));
    program_three_d(&mut channel, 0x2388, (uniform_address + 0x100) as u32);
    program_three_d(&mut channel, 0x2430, 0x21);
    let changed = draw(&mut channel, &mut frontend);
    assert!(!Arc::ptr_eq(&before.program, &changed.program));
    let mapping = |binding: &nixe_gpu::native::GuestBinding| {
        let nixe_gpu::native::GuestBinding::Buffer(range) = binding else {
            panic!("expected buffer")
        };
        range.clone()
    };
    assert_eq!(
        mapping(&before.program.bindings[0].1),
        mapping(&changed.program.bindings[0].1)
    );
    assert_ne!(
        mapping(&before.program.bindings[1].1),
        mapping(&changed.program.bindings[1].1)
    );
    program_three_d(&mut channel, 0x1c08, vertex as u32 + 32);
    let rebound_vertex = draw(&mut channel, &mut frontend);
    assert!(Arc::ptr_eq(&changed.program, &rebound_vertex.program));
}

#[test]
fn direct_mme_program_cache_follows_instruction_uploads_and_start_address_changes() {
    let mut channel = three_d_channel();
    let index = 5u8;
    let set_method = 1 | (2 << 4) | ((0x1518 / 4) << 14);
    let send_parameter = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(&mut channel, index, &[set_method, send_parameter, 0x11]);
    let call = (0x3800 + u32::from(index) * 8) / 4;
    dispatch_method(&mut channel, call, 2.5_f32.to_bits()).unwrap();
    assert_eq!(
        channel.three_d().raster().point_size().raw(),
        Some(2.5_f32.to_bits())
    );
    let constant = |value: u32| 1 | (4 << 4) | (1 << 7) | (value << 14);
    load_mme_program(&mut channel, index, &[set_method, constant(4), 0x11]);
    dispatch_method(&mut channel, call, 3.5_f32.to_bits()).unwrap();
    assert_eq!(channel.three_d().raster().point_size().raw(), Some(4));
    load_mme_program(&mut channel, index + 1, &[set_method, constant(8), 0x11]);
    dispatch_method(&mut channel, call, 3.5_f32.to_bits()).unwrap();
    assert_eq!(channel.three_d().raster().point_size().raw(), Some(4));
    dispatch_incrementing(
        &mut channel,
        0x011c / 4,
        &[u32::from(index), 0x100 + u32::from(index + 1) * 0x10],
    )
    .unwrap();
    dispatch_method(&mut channel, call, 3.5_f32.to_bits()).unwrap();
    assert_eq!(channel.three_d().raster().point_size().raw(), Some(8));
}
