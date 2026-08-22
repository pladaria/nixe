use super::*;

#[test]
fn three_d_no_operation_is_named_and_implemented() {
    let mut channel = three_d_channel();
    let dispatch = dispatch_method(&mut channel, 0x100 / 4, 0xfeed_beef).unwrap();

    assert_eq!(dispatch.methods().len(), 1);
    assert_eq!(dispatch.methods()[0].metadata().class_name(), "MAXWELL_B");
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "NO_OPERATION"
    );
}

#[test]
fn pipe_nop_accepts_its_full_payload_without_state_or_execution_effects() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);

    for argument in [0, 1, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let dispatch = dispatch_method(&mut channel, 0x1a2c / 4, argument).unwrap();

        assert_eq!(dispatch.methods().len(), 1);
        assert_eq!(dispatch.methods()[0].metadata().class_name(), "MAXWELL_B");
        assert_eq!(dispatch.methods()[0].metadata().method_name(), "PIPE_NOP");

        assert!(dispatch.operations().is_empty());
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn instrumentation_header_and_data_are_source_preserving_pipeline_neutral_annotations() {
    let mut channel = three_d_channel();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let dispatch =
        dispatch_incrementing(&mut channel, 0x0150 / 4, &[0x4900_0000, 0x4900_0001]).unwrap();

    assert_eq!(
        dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        [
            "SET_INSTRUMENTATION_METHOD_HEADER",
            "SET_INSTRUMENTATION_METHOD_DATA"
        ]
    );
    assert!(dispatch.operations().is_empty());

    let header_source = dispatch.methods()[0].method().source();
    let header = MaxwellThreeDInstrumentationValue::from_bits(0x4900_0000);

    let header_register = channel.three_d().instrumentation().header();
    assert_eq!(
        header_register.origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(header_register.raw(), Some(0x4900_0000));
    assert_eq!(header_register.value().copied(), Some(header));
    assert_eq!(header_register.source(), Some(header_source));

    let data_source = dispatch.methods()[1].method().source();
    let data = MaxwellThreeDInstrumentationValue::from_bits(0x4900_0001);

    let data_register = channel.three_d().instrumentation().data();
    assert_eq!(data_register.raw(), Some(0x4900_0001));
    assert_eq!(data_register.value().copied(), Some(data));
    assert_eq!(data_register.source(), Some(data_source));
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn instrumentation_accepts_all_word_bits_and_invalid_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();

    for (method, argument) in [(0x0150, u32::MAX), (0x0154, 0)] {
        program_three_d(&mut channel, method, argument);
        let register = if method == 0x0150 {
            channel.three_d().instrumentation().header()
        } else {
            channel.three_d().instrumentation().data()
        };
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().map(|value| value.bits()), Some(argument));
    }

    let before = channel.clone();
    let decoded = incrementing_packet(0x0150 / 4, &[1, 2, 3]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0158)
    ));
    assert_ne!(channel, before);
}

#[test]
fn tiled_cache_initialization_family_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for (method, argument) in [
        (0x0f60, 1),
        (0x0f64, 0x0080_0080),
        (0x0f68, 0x0000_1109),
        (0x0f6c, 0x0808_0202),
        (0x1108, 0x0000_001f),
        (0x0f70, 0x0008_0001),
    ] {
        program_three_d(&mut channel, method, argument);
    }

    let state = channel.three_d().tiled_cache();
    assert_eq!(state.enabled().value(), Some(&true));
    assert_eq!(state.enabled().raw(), Some(1));
    let tile_size = state.tile_size().value().copied().unwrap();
    assert_eq!(tile_size.width(), 128);
    assert_eq!(tile_size.height(), 128);
    assert_eq!(tile_size.raw(), 0x0080_0080);
    for (index, expected) in [0x0000_1109, 0x0808_0202, 0x0000_001f, 0x0008_0001]
        .into_iter()
        .enumerate()
    {
        let register = state.unknown_config(index as u8).unwrap();
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(expected));
        assert_eq!(register.value().map(|value| value.bits()), Some(expected));
        assert!(register.source().is_some());
    }
    assert!(state.unknown_config(4).is_none());
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn tiled_cache_enable_rejects_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f60, 1);

    let before = channel.clone();
    let decoded = incrementing_packet(0x0f60 / 4, &[0, 0x0040_0020, 3, 4, 5, 6]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0f74)
    ));
    assert_ne!(channel, before);

    let after_prefix = channel.clone();
    let invalid_enable = packet(0x0f60 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_enable),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x0f60)
    ));
    assert_eq!(channel, after_prefix);
}

#[test]
fn mutable_method_control_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    assert_eq!(
        channel.three_d().mme().mutable_method_control().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDMutableMethodControl::Lightweight),
        (1, MaxwellThreeDMutableMethodControl::Heavyweight),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1134 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_MUTABLE_METHOD_CONTROL"
        );

        let register = channel.three_d().mme().mutable_method_control();
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
    }

    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn mutable_method_control_rejects_reserved_bits_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1134, 1);
    let before = channel.clone();

    let decoded = packet(0x1134 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x1134)
    ));
    assert_eq!(channel, before);
}

#[test]
fn constant_color_rendering_family_is_typed_and_conditionally_pipeline_relevant() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f40, 0);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);

    for (method, component, bits) in [
        (
            0x0f44,
            MaxwellThreeDConstantColorComponent::Red,
            0x3f80_0000,
        ),
        (
            0x0f48,
            MaxwellThreeDConstantColorComponent::Green,
            0x4000_0000,
        ),
        (
            0x0f4c,
            MaxwellThreeDConstantColorComponent::Blue,
            0x4040_0000,
        ),
        (
            0x0f50,
            MaxwellThreeDConstantColorComponent::Alpha,
            0x4080_0000,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, method / 4, bits).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .constant_color_rendering()
            .component(component);
        assert_eq!(register.raw(), Some(bits));
        assert_eq!(register.value().map(|value| value.bits()), Some(bits));
        assert_eq!(register.source(), Some(source));
    }
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );

    program_three_d(&mut channel, 0x0f40, 1);
    let enabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(enabled_dependencies, disabled_dependencies);
    for raw in [1, 0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000] {
        assert!(enabled_dependencies.contains(&Some(raw)));
    }
}

#[test]
fn constant_color_rendering_rejects_reserved_enable_bits_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f40, 0);
    let before = channel.clone();

    let decoded = incrementing_packet(0x0f40 / 4, &[1, 2, 3, 4, 5, 0x100]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. })
            if source.method() == GpuMethodId(0x0f54)
    ));
    assert_ne!(channel, before);

    let after_prefix = channel.clone();
    let invalid_enable = packet(0x0f40 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_enable),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x0f40)
    ));
    assert_eq!(channel, after_prefix);
}

#[test]
fn api_mandated_early_z_is_typed_and_source_preserving() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x2000, 0x11);

    program_three_d(&mut channel, 0x0210, 0);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    let disabled = channel.three_d().shader_execution().api_mandated_early_z();
    assert_eq!(
        disabled.value(),
        Some(&MaxwellThreeDApiMandatedEarlyZ::Disabled)
    );
    assert_eq!(disabled.raw(), Some(0));
    assert!(disabled.source().is_some());

    program_three_d(&mut channel, 0x0210, 1);
    let enabled = channel.three_d().shader_execution().api_mandated_early_z();
    assert_eq!(
        enabled.value(),
        Some(&MaxwellThreeDApiMandatedEarlyZ::Enabled)
    );
    assert_eq!(enabled.raw(), Some(1));
    assert!(enabled.source().is_some());
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );
}

#[test]
fn api_mandated_early_z_rejects_reserved_bits_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0210, 0);
    let before = channel.clone();

    let decoded = packet(0x0210 / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x0210)
    ));
    assert_eq!(channel, before);
}

#[test]
fn post_z_pixel_shader_imask_is_typed_and_only_active_state_is_pipeline_relevant() {
    let mut channel = three_d_channel();

    program_three_d(&mut channel, 0x0f1c, 0);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    let disabled = channel.three_d().coverage().post_z_pixel_shader_imask();
    assert_eq!(
        disabled.value(),
        Some(&MaxwellThreeDPostZPixelShaderImask::Disabled)
    );
    assert_eq!(disabled.raw(), Some(0));
    assert!(disabled.source().is_some());

    program_three_d(&mut channel, 0x0f1c, 1);
    let enabled = channel.three_d().coverage().post_z_pixel_shader_imask();
    assert_eq!(
        enabled.value(),
        Some(&MaxwellThreeDPostZPixelShaderImask::Enabled)
    );
    assert_eq!(enabled.raw(), Some(1));
    assert!(enabled.source().is_some());
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );
}

#[test]
fn post_z_pixel_shader_imask_rejects_reserved_bits_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f1c, 0);
    let before = channel.clone();

    let decoded = packet(0x0f1c / 4, 2);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x0f1c)
    ));
    assert_eq!(channel, before);
}

#[test]
fn pixel_shader_interlock_control_decodes_all_allocated_field_combinations() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x2000, 0x11);

    for raw in 0..=0x0f {
        if raw & 3 == 3 {
            continue;
        }
        program_three_d(&mut channel, 0x1224, raw);
        let value = channel
            .three_d()
            .shader_execution()
            .pixel_shader_interlock_control()
            .value()
            .copied()
            .unwrap();
        assert_eq!(value.raw(), raw);
        assert_eq!(
            value.tile_size(),
            if raw & 4 == 0 {
                MaxwellThreeDPixelShaderInterlockTileSize::Tile16x16
            } else {
                MaxwellThreeDPixelShaderInterlockTileSize::Tile8x8
            }
        );
        assert_eq!(
            value.fragment_order(),
            if raw & 8 == 0 {
                MaxwellThreeDPixelShaderInterlockFragmentOrder::Ordered
            } else {
                MaxwellThreeDPixelShaderInterlockFragmentOrder::Unordered
            }
        );
        assert_eq!(value.conflict_detection_enabled(), raw & 3 != 0);
    }

    program_three_d(&mut channel, 0x1224, 0);
    let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x1224, 1);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );
}

#[test]
fn pixel_shader_interlock_control_rejects_reserved_encodings_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1224, 0);

    for raw in [3, 7, 0x0b, 0x0f, 0x10, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x1224 / 4, raw);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.method() == GpuMethodId(0x1224)
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn global_draw_indices_preserve_full_width_values_without_changing_pipeline_identity() {
    let mut channel = three_d_channel();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for raw in [0, 1, i32::MAX as u32, u32::MAX] {
        let dispatch = dispatch_method(&mut channel, 0x1434 / 4, raw).unwrap();
        let register = channel
            .three_d()
            .vertex_input()
            .assembly()
            .global_base_vertex_index();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_GLOBAL_BASE_VERTEX_INDEX"
        );
        assert_eq!(register.raw(), Some(raw));
        assert_eq!(register.value(), Some(&raw));
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    for raw in [0, 1, u32::MAX] {
        let dispatch = dispatch_method(&mut channel, 0x1438 / 4, raw).unwrap();
        let register = channel
            .three_d()
            .vertex_input()
            .assembly()
            .global_base_instance_index();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_GLOBAL_BASE_INSTANCE_INDEX"
        );
        assert_eq!(register.raw(), Some(raw));
        assert_eq!(register.value(), Some(&raw));
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
}

#[test]
fn shader_exceptions_enable_is_typed_source_preserving_diagnostic_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    assert_eq!(
        channel
            .three_d()
            .shader_execution()
            .shader_exceptions_enable()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDShaderExceptionsEnable::Disabled),
        (1, MaxwellThreeDShaderExceptionsEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1528 / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .shader_execution()
            .shader_exceptions_enable();

        assert_eq!(method.metadata().method_name(), "SET_SHADER_EXCEPTIONS");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.enabled(), argument != 0);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn shader_exceptions_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1528, 1);

    for argument in [2, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1528 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
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
    let decoded = non_incrementing_packet_on_subchannel(0, 0x1528 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.method() == GpuMethodId(0x1528) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn alpha_fraction_is_typed_source_preserving_raster_state() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let viewport_before = channel.three_d().viewport().clone();
    let point_size_before = *channel.three_d().raster().point_size();
    let two_d_before = channel.two_d().clone();
    let mut previous_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_eq!(
        channel.three_d().raster().alpha_fraction().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for argument in [0, 0x3f, 0xff] {
        let dispatch = dispatch_method(&mut channel, 0x074c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDAlphaFraction::new(argument as u8);
        let register = channel.three_d().raster().alpha_fraction();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALPHA_FRACTION"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument as u8);
        assert_eq!(channel.three_d().raster().point_size(), &point_size_before);
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().render_targets(), &render_targets_before);
        assert_eq!(channel.three_d().viewport(), &viewport_before);
        assert_eq!(channel.two_d(), &two_d_before);

        let dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(dependencies, previous_dependencies);
        previous_dependencies = dependencies;
    }
}

#[test]
fn invalid_alpha_fraction_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x074c, 0x3f);

    for argument in [0x0000_0100, 0x0000_ff00, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x074c / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_00ff,
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
    let decoded = incrementing_packet(0x074c / 4, &[0xff, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0750)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_report_semaphore_setup_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    let dispatch =
        dispatch_incrementing(&mut channel, 0x1b00 / 4, &[0x12, 0x3456_7890, 0xcafe_babe]).unwrap();

    assert_eq!(
        dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        [
            "SET_REPORT_SEMAPHORE_A",
            "SET_REPORT_SEMAPHORE_B",
            "SET_REPORT_SEMAPHORE_C",
        ]
    );
    let state = channel.three_d().report_semaphore();
    assert_eq!(
        state.address().map(MaxwellThreeDUnresolvedAddress::get),
        Some(0x12_3456_7890)
    );
    assert_eq!(state.address_upper().raw(), Some(0x12));
    assert_eq!(state.address_lower().raw(), Some(0x3456_7890));
    assert_eq!(state.payload().raw(), Some(0xcafe_babe));
    assert_eq!(
        state.address_upper().source(),
        Some(dispatch.methods()[0].method().source())
    );
    assert_eq!(
        state.address_lower().source(),
        Some(dispatch.methods()[1].method().source())
    );
    assert_eq!(
        state.payload().source(),
        Some(dispatch.methods()[2].method().source())
    );

    let before = channel.clone();
    let invalid = packet(0x1b00 / 4, 0x100);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0xff,
            ..
        }) if source.argument() == 0x100
    ));
    assert_eq!(channel, before);
}

#[test]
fn three_d_report_semaphore_trigger_is_an_explicit_observable_boundary() {
    let mut channel = three_d_channel();
    for (method, argument) in [(0x1b00, 0x12), (0x1b04, 0x3456_7890), (0x1b08, 0xcafe_babe)] {
        program_three_d(&mut channel, method, argument);
    }

    let dispatch = dispatch_method(&mut channel, 0x1b0c / 4, 0x1000_f010).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_REPORT_SEMAPHORE_D"
    );
    let operation = &dispatch.synchronization_operations()[0];
    let control = match operation.trigger() {
        MaxwellThreeDSynchronizationTrigger::ReportSemaphore { control, .. } => control,
        other => panic!("unexpected trigger: {other:?}"),
    };
    assert_eq!(control.raw(), 0x1000_f010);
    assert_eq!(
        control.operation(),
        MaxwellThreeDReportSemaphoreOperation::Release
    );
    assert_eq!(
        control.pipeline_location(),
        MaxwellThreeDReportSemaphorePipelineLocation::All
    );
    assert_eq!(
        control.structure_size(),
        MaxwellThreeDReportSemaphoreStructureSize::OneWord
    );
    assert!(matches!(
        lower_maxwell_three_d_synchronization(operation, None, true),
        Ok(MaxwellThreeDSynchronizationPlan::ReportSemaphoreRelease(release))
            if release.address().get() == 0x12_3456_7890
                && release.payload() == 0xcafe_babe
                && release.prior_work_pending()
    ));

    let dispatch = dispatch_method(&mut channel, 0x1b0c / 4, 0).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_synchronization(
            dispatch.synchronization_operations()[0],
            None,
            false
        ),
        Err(MaxwellThreeDSynchronizationError::UnsupportedReportSemaphoreControl {
            control,
            ..
        }) if control.raw() == 0
    ));

    let before = channel.clone();
    let unallocated = packet(0x1b0c / 4, 3 << 12);
    assert!(matches!(
        dispatch_first(&mut channel, &unallocated),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "SET_REPORT_SEMAPHORE_D",
            ..
        })
    ));
    assert_eq!(channel, before);

    let invalid = packet(0x1b0c / 4, 1 << 30);
    assert!(matches!(
        dispatch_first(&mut channel, &invalid),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x1fb7_ffff,
            ..
        }) if source.argument() == 1 << 30
    ));
    assert_eq!(channel, before);
}

#[test]
fn raster_bounding_box_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let dependencies = channel.three_d().pipeline_dependencies(&[]);
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel.three_d().raster().bounding_box().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, mode, pad) in [
        (0, MaxwellThreeDRasterBoundingBoxMode::BoundingBox, 0),
        (1, MaxwellThreeDRasterBoundingBoxMode::FullViewport, 0),
        (0x60, MaxwellThreeDRasterBoundingBoxMode::BoundingBox, 6),
        (
            0x0ff1,
            MaxwellThreeDRasterBoundingBoxMode::FullViewport,
            u8::MAX,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x02ec / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDRasterBoundingBox::new(mode, pad);
        let register = channel.three_d().raster().bounding_box();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RASTER_BOUNDING_BOX"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.mode(), mode);
        assert_eq!(value.pad(), pad);
        assert_eq!(value.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d().pipeline_dependencies(&[]), dependencies);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let macro_index = 8;
    let method_dword = 0x02ec / 4;
    let set_method = 1 | (2 << 4) | (method_dword << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        macro_index,
        &[set_method, send_parameter_and_exit, 0x11],
    );
    let dispatch = dispatch_method(
        &mut channel,
        (0x3800 + u32::from(macro_index) * 8) / 4,
        0x60,
    )
    .unwrap();

    let register = channel.three_d().raster().bounding_box();
    let source = register.source().unwrap();
    assert_eq!(register.raw(), Some(0x60));
    assert_eq!(source.method(), GpuMethodId(0x02ec));
    assert_eq!(
        source.location(),
        dispatch.methods()[0].method().source().location()
    );
    assert_eq!(channel.three_d().pipeline_dependencies(&[]), dependencies);
}

#[test]
fn invalid_raster_bounding_box_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x02ec, 0x60);

    for argument in [2, 4, 8, 0x1000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x02ec / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_0ff1,
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
    let decoded = incrementing_packet(0x02ec / 4, &[0x60, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x02f0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn sph_version_check_is_typed_profile_validated_and_state_neutral() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    for (argument, current, oldest_supported) in [
        (0x0003_0003, 3, 3),
        (0x0003_0004, 4, 3),
        (0x0002_0003, 3, 2),
        (0x0002_0004, 4, 2),
    ] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let dispatch = dispatch_method(&mut channel, 0x16a8 / 4, argument).unwrap();
        let requested = MaxwellShaderProgramHeaderVersionRange::new(
            MaxwellShaderProgramHeaderVersion::new(current),
            MaxwellShaderProgramHeaderVersion::new(oldest_supported),
        );

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CHECK_SPH_VERSION"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(requested.raw(), argument);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
        assert_eq!(channel.frontend(), frontend_before);
    }
}

#[test]
fn malformed_incompatible_and_suffixed_sph_checks_are_rejected_atomically() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet(0x16a8 / 4, 0x0003_0002);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "CHECK_SPH_VERSION",
            ..
        }) if source.argument() == 0x0003_0002
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    for argument in [0x0002_0002, 0x0004_0004] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x16a8 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(
                MaxwellEngineDispatchError::IncompatibleShaderProgramHeaderVersion {
                    source,
                    requested,
                    supported,
                }
            ) if source.argument() == argument
                && requested.raw() == argument
                && supported.raw() == 0x0003_0003
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x16a8 / 4, &[0x0003_0003, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x16ac)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn aam_version_check_is_typed_profile_validated_and_state_neutral() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);
    for (argument, current, oldest_supported) in [
        (0x0002_0002, 2, 2),
        (0x0002_0003, 3, 2),
        (0x0001_0002, 2, 1),
        (0x0001_0003, 3, 1),
    ] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let dispatch = dispatch_method(&mut channel, 0x1794 / 4, argument).unwrap();
        let requested = MaxwellAamVersionRange::new(
            MaxwellAamVersion::new(current),
            MaxwellAamVersion::new(oldest_supported),
        );

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CHECK_AAM_VERSION"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(requested.raw(), argument);
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn malformed_incompatible_and_suffixed_aam_checks_are_rejected_atomically() {
    let mut channel = three_d_channel();
    use_mme_shadow_passthrough(&mut channel);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet(0x1794 / 4, 0x0003_0002);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "CHECK_AAM_VERSION",
            ..
        }) if source.argument() == 0x0003_0002
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    for argument in [0x0001_0001, 0x0003_0003] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1794 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::IncompatibleAamVersion {
                source,
                requested,
                supported,
            }) if source.argument() == argument
                && requested.raw() == argument
                && supported.raw() == 0x0002_0002
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1794 / 4, &[0x0002_0002, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1798)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn rop_l2_cache_controls_are_typed_source_preserving_independent_state() {
    let mut channel = three_d_channel();
    let methods = [
        (
            0x0218,
            "SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS",
            MaxwellThreeDRopL2CacheRequest::PrefetchRead,
        ),
        (
            0x10fc,
            "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS",
            MaxwellThreeDRopL2CacheRequest::NoninterlockedRead,
        ),
        (
            0x1290,
            "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS",
            MaxwellThreeDRopL2CacheRequest::InterlockedRead,
        ),
        (
            0x12d8,
            "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS",
            MaxwellThreeDRopL2CacheRequest::NoninterlockedWrite,
        ),
        (
            0x12dc,
            "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS",
            MaxwellThreeDRopL2CacheRequest::InterlockedWrite,
        ),
    ];
    let requests = methods.map(|(_, _, request)| request);
    let two_d_before = channel.two_d().clone();
    let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for request in requests {
        assert_eq!(
            channel.three_d().l2_cache().rop_policy(request).origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );
    }

    for (method, method_name, request) in methods {
        for (argument, value) in [
            (0x00, MaxwellThreeDL2CacheEvictionPolicy::EvictFirst),
            (0x10, MaxwellThreeDL2CacheEvictionPolicy::EvictNormal),
            (0x20, MaxwellThreeDL2CacheEvictionPolicy::EvictLast),
        ] {
            let registers_before =
                requests.map(|other| *channel.three_d().l2_cache().rop_policy(other));
            let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().l2_cache().rop_policy(request);

            assert_eq!(dispatch.methods()[0].metadata().method_name(), method_name);

            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.encoded(), argument);

            for (index, other) in requests.into_iter().enumerate() {
                if other != request {
                    assert_eq!(
                        channel.three_d().l2_cache().rop_policy(other),
                        &registers_before[index]
                    );
                }
            }
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                pipeline_dependencies_before
            );
        }
    }
}

#[test]
fn vaf_l2_cache_control_preserves_volatility_policy_source_and_isolation() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let rop_before = [
        MaxwellThreeDRopL2CacheRequest::PrefetchRead,
        MaxwellThreeDRopL2CacheRequest::NoninterlockedRead,
        MaxwellThreeDRopL2CacheRequest::InterlockedRead,
        MaxwellThreeDRopL2CacheRequest::NoninterlockedWrite,
        MaxwellThreeDRopL2CacheRequest::InterlockedWrite,
    ]
    .map(|request| *channel.three_d().l2_cache().rop_policy(request));
    assert_eq!(
        channel.three_d().l2_cache().vaf_control().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, system_memory, policy) in [
        (
            0x00,
            MaxwellThreeDSystemMemoryVolatile::Stable,
            MaxwellThreeDL2CacheEvictionPolicy::EvictFirst,
        ),
        (
            0x01,
            MaxwellThreeDSystemMemoryVolatile::Volatile,
            MaxwellThreeDL2CacheEvictionPolicy::EvictFirst,
        ),
        (
            0x10,
            MaxwellThreeDSystemMemoryVolatile::Stable,
            MaxwellThreeDL2CacheEvictionPolicy::EvictNormal,
        ),
        (
            0x11,
            MaxwellThreeDSystemMemoryVolatile::Volatile,
            MaxwellThreeDL2CacheEvictionPolicy::EvictNormal,
        ),
        (
            0x20,
            MaxwellThreeDSystemMemoryVolatile::Stable,
            MaxwellThreeDL2CacheEvictionPolicy::EvictLast,
        ),
        (
            0x21,
            MaxwellThreeDSystemMemoryVolatile::Volatile,
            MaxwellThreeDL2CacheEvictionPolicy::EvictLast,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1000 / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let source = method.method().source();
        let value = channel
            .three_d()
            .l2_cache()
            .vaf_control()
            .value()
            .copied()
            .unwrap();

        assert_eq!(
            method.metadata().method_name(),
            "SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.system_memory(), system_memory);
        assert_eq!(value.policy(), policy);
        assert_eq!(value.raw(), argument);
        assert_eq!(system_memory.volatile(), argument & 1 != 0);
        let register = channel.three_d().l2_cache().vaf_control();
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            [
                MaxwellThreeDRopL2CacheRequest::PrefetchRead,
                MaxwellThreeDRopL2CacheRequest::NoninterlockedRead,
                MaxwellThreeDRopL2CacheRequest::InterlockedRead,
                MaxwellThreeDRopL2CacheRequest::NoninterlockedWrite,
                MaxwellThreeDRopL2CacheRequest::InterlockedWrite,
            ]
            .map(|request| *channel.three_d().l2_cache().rop_policy(request)),
            rop_before
        );
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_vaf_l2_cache_controls_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1000, 0x10);

    for argument in [0x02, 0x30, 0x31, 0x40, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1000 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x31,
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
    let decoded = non_incrementing_packet_on_subchannel(0, 0x1000 / 4, &[0x11, 0x30]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x31,
            ..
        }) if source.method() == GpuMethodId(0x1000) && source.argument() == 0x30
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_rop_l2_cache_policies_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    let methods = [0x0218, 0x10fc, 0x1290, 0x12d8, 0x12dc];
    for method in methods {
        program_three_d(&mut channel, method, 0x10);
    }

    for method in methods {
        for argument in [0x30, 0x01, 0x40, 0x8000_0010, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);

            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x0000_0030,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x0218 / 4, &[0x20, 1 << 3]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x0000_1017,
            ..
        }) if source.method() == GpuMethodId(0x021c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_notify_address_upper_is_bounded_state_without_notification_effects() {
    let mut channel = two_d_channel();
    let render_enable_before = channel.two_d().render_enable().clone();
    let pixels_from_memory_before = channel.two_d().pixels_from_memory().clone();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().notify().address_upper().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );
    assert_eq!(
        MaxwellTwoDNotifyAddressUpper::new(
            MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX.saturating_add(1)
        ),
        None
    );

    for argument in [0, MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX] {
        let decoded = packet_on_subchannel(3, 0x0104 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDNotifyAddressUpper::new(argument).unwrap();
        let register = channel.two_d().notify().address_upper();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NOTIFY_A"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.get(), argument);
        assert_eq!(channel.two_d().render_enable(), &render_enable_before);
        assert_eq!(
            channel.two_d().pixels_from_memory(),
            &pixels_from_memory_before
        );
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn invalid_two_d_notify_address_upper_rejects_atomically() {
    let mut channel = two_d_channel();

    for argument in [0x0200_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0104 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn two_d_notify_address_lower_accepts_its_complete_bit_domain() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().notify().address_lower().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for argument in [0, 0x0820_2010, u32::MAX] {
        let decoded = packet_on_subchannel(3, 0x0108 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDNotifyAddressLower::new(argument);
        let register = channel.two_d().notify().address_lower();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NOTIFY_B"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.get(), argument);
        assert_eq!(
            channel.two_d().notify().address_upper().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn incrementing_notify_address_fragments_commit_together() {
    let mut channel = two_d_channel();
    let decoded = incrementing_packet_on_subchannel(3, 0x0104 / 4, &[1, 0x0820_2010]);
    let dispatch = dispatch_first(&mut channel, &decoded).unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_NOTIFY_A"
    );
    assert_eq!(
        dispatch.methods()[1].metadata().method_name(),
        "SET_NOTIFY_B"
    );
    assert!(dispatch.operations().is_empty());
    assert_eq!(channel.two_d().notify().address_upper().raw(), Some(1));
    assert_eq!(
        channel.two_d().notify().address_lower().raw(),
        Some(0x0820_2010)
    );
    assert_eq!(
        channel.two_d().notify().address_upper().source(),
        Some(dispatch.methods()[0].method().source())
    );
    assert_eq!(
        channel.two_d().notify().address_lower().source(),
        Some(dispatch.methods()[1].method().source())
    );
}

#[test]
fn unsupported_notify_trigger_keeps_both_address_fragments() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x0104 / 4, &[1, 0x0820_2010, 0]);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x010c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_processing_cluster_values_are_typed_and_retain_their_source() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().processing_clusters().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellTwoDProcessingClusters::All),
        (1, MaxwellTwoDProcessingClusters::One),
    ] {
        let decoded = packet_on_subchannel(3, 0x0260 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().processing_clusters();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NUM_PROCESSING_CLUSTERS"
        );

        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn two_d_render_enable_modes_are_typed_state_without_condition_evaluation() {
    let mut channel = two_d_channel();
    let pixels_from_memory_before = channel.two_d().pixels_from_memory().clone();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().render_enable().mode().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellTwoDRenderEnableMode::Disabled),
        (1, MaxwellTwoDRenderEnableMode::Enabled),
        (2, MaxwellTwoDRenderEnableMode::Conditional),
        (3, MaxwellTwoDRenderEnableMode::RenderIfEqual),
        (4, MaxwellTwoDRenderEnableMode::RenderIfNotEqual),
    ] {
        let decoded = packet_on_subchannel(3, 0x026c / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().render_enable().mode();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_C"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.two_d().pixels_from_memory(),
            &pixels_from_memory_before
        );
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn invalid_two_d_render_enable_modes_are_rejected_atomically() {
    let mut channel = two_d_channel();

    for argument in [5, 6, 7, 8, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x026c / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 7,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn unsupported_render_enable_address_methods_remain_typed_fatal_errors() {
    let mut channel = two_d_channel();

    for method in [0x0264, 0x0268] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, method / 4, 0);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "FERMI_TWOD_A",
            }) if source.method() == GpuMethodId(method)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn two_d_beta_registers_preserve_complete_bit_domains_without_execution() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().beta().beta1().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );
    assert_eq!(
        channel.two_d().beta().beta4().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for argument in [0, 0x7f80_0000, u32::MAX] {
        let decoded = packet_on_subchannel(3, 0x02a4 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDBeta1::new(argument);
        let register = channel.two_d().beta().beta1();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_BETA1");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d(), &three_d_before);
    }

    for argument in [0, 0x1122_3344, u32::MAX] {
        let decoded = packet_on_subchannel(3, 0x02a8 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDBeta4::from_raw(argument);
        let register = channel.two_d().beta().beta4();

        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_BETA4");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.blue(), argument as u8);
        assert_eq!(value.green(), (argument >> 8) as u8);
        assert_eq!(value.red(), (argument >> 16) as u8);
        assert_eq!(value.alpha(), (argument >> 24) as u8);
        assert_eq!(value.raw(), argument);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn incrementing_two_d_beta_pair_commits_in_method_order() {
    let mut channel = two_d_channel();
    let decoded = incrementing_packet_on_subchannel(3, 0x02a4 / 4, &[0x7f80_0000, 0x1122_3344]);
    let dispatch = dispatch_first(&mut channel, &decoded).unwrap();

    assert_eq!(dispatch.methods().len(), 2);
    assert_eq!(
        dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        ["SET_BETA1", "SET_BETA4"]
    );
    assert!(dispatch.operations().is_empty());
    assert_eq!(
        channel.two_d().beta().beta1().value().copied(),
        Some(MaxwellTwoDBeta1::new(0x7f80_0000))
    );
    assert_eq!(
        channel.two_d().beta().beta4().value().copied(),
        Some(MaxwellTwoDBeta4::from_raw(0x1122_3344))
    );
}

#[test]
fn two_d_operation_values_are_typed_state_without_execution() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().operation().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellTwoDOperation::SourceCopyAnd),
        (1, MaxwellTwoDOperation::RasterOperationAnd),
        (2, MaxwellTwoDOperation::BlendAnd),
        (3, MaxwellTwoDOperation::SourceCopy),
        (4, MaxwellTwoDOperation::RasterOperation),
        (5, MaxwellTwoDOperation::SourceCopyPremultiplied),
        (6, MaxwellTwoDOperation::BlendPremultiplied),
    ] {
        let decoded = packet_on_subchannel(3, 0x02ac / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().operation();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_OPERATION"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn two_d_clip_enable_values_are_typed_state_without_execution() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().clip_enable().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellTwoDClipEnable::Disabled),
        (1, MaxwellTwoDClipEnable::Enabled),
    ] {
        let decoded = packet_on_subchannel(3, 0x0290 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().clip_enable();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CLIP_ENABLE"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn invalid_two_d_clip_enable_rejects_without_mutating_channel_state() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x0290 / 4, 2);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_color_key_enable_values_are_typed_state_without_execution() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().color_key_enable().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );
    assert_eq!(
        channel.two_d().clip_enable().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellTwoDColorKeyEnable::Disabled),
        (1, MaxwellTwoDColorKeyEnable::Enabled),
    ] {
        let decoded = packet_on_subchannel(3, 0x029c / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().color_key_enable();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_COLOR_KEY_ENABLE"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.two_d().clip_enable().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn invalid_two_d_color_key_enable_rejects_without_mutating_channel_state() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x029c / 4, 2);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn unsupported_method_after_color_key_enable_keeps_the_packet_prefix() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x029c / 4, &[1, 0]);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x02a0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_corral_size_is_bounded_source_preserving_state_without_execution() {
    let mut channel = two_d_channel();
    let state_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().pixels_from_memory().corral_size().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for argument in [0, 0x3f, u32::from(MAXWELL_TWO_D_CORRAL_SIZE_MAX)] {
        let decoded = packet_on_subchannel(3, 0x0884 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDPixelsFromMemoryCorralSize::new(argument as u16).unwrap();
        let register = channel.two_d().pixels_from_memory().corral_size();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PIXELS_FROM_MEMORY_CORRAL_SIZE"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.get(), argument as u16);
        assert_eq!(channel.two_d().clip_enable(), state_before.clip_enable());
        assert_eq!(
            channel.two_d().color_key_enable(),
            state_before.color_key_enable()
        );
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn invalid_two_d_corral_size_rejects_without_mutating_channel_state() {
    let mut channel = two_d_channel();

    for argument in [0x0400, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0884 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x03ff,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn two_d_safe_overlap_values_are_typed_state_without_execution() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().pixels_from_memory().safe_overlap().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellTwoDPixelsFromMemorySafeOverlap::Disabled),
        (1, MaxwellTwoDPixelsFromMemorySafeOverlap::Enabled),
    ] {
        let decoded = packet_on_subchannel(3, 0x0888 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().pixels_from_memory().safe_overlap();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.two_d().pixels_from_memory().corral_size().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn invalid_two_d_safe_overlap_rejects_without_mutating_channel_state() {
    let mut channel = two_d_channel();

    for argument in [2, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0888 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn unsupported_pixels_from_memory_method_remains_a_typed_fatal_error() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x0880 / 4, 0);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x0880)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_two_d_operation_values_are_rejected_atomically() {
    let mut channel = two_d_channel();

    for argument in [7, 8] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x02ac / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 7,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn unsupported_method_after_two_d_operation_keeps_the_packet_prefix() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x02ac / 4, &[3, 0]);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x02b0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_two_d_value_rejects_without_mutating_any_channel_state() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x0260 / 4, 2);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn unsupported_two_d_suffix_keeps_the_valid_packet_prefix() {
    let mut channel = two_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x0260 / 4, &[1, 0]);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x0264)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_ne!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_packet_mutates_only_its_engine_state() {
    let mut channel = two_d_channel();
    let three_d_before = channel.three_d().clone();
    let first = packet_on_subchannel(3, 0x0260 / 4, 0);
    dispatch_first(&mut channel, &first).unwrap();
    assert_eq!(channel.two_d().processing_clusters().raw(), Some(0));
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn three_d_render_enable_modes_are_typed_and_engine_owned() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel.three_d().render_enable().mode().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDRenderEnableMode::Disabled),
        (1, MaxwellThreeDRenderEnableMode::Enabled),
        (2, MaxwellThreeDRenderEnableMode::Conditional),
        (3, MaxwellThreeDRenderEnableMode::RenderIfEqual),
        (4, MaxwellThreeDRenderEnableMode::RenderIfNotEqual),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1558 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().render_enable().mode();

        assert_eq!(dispatch.methods()[0].metadata().class(), threed::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_C"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn render_enable_control_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let mode_before = *channel.three_d().render_enable().mode();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for (argument, value) in [
        (0, MaxwellThreeDConditionalLoadConstantBuffer::Disabled),
        (1, MaxwellThreeDConditionalLoadConstantBuffer::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x030c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .render_enable()
            .conditional_load_constant_buffer();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_CONTROL"
        );

        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(channel.three_d().render_enable().mode(), &mode_before);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_render_enable_controls_are_rejected_atomically() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x030c, 0);

    for argument in [2, 3, 0x10, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x030c / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn enabled_conditional_load_stops_before_neutral_lowering() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1558, 1);
    let disabled = dispatch_method(&mut channel, 0x030c / 4, 0).unwrap();
    assert_eq!(
        disabled.methods()[0].metadata().method_name(),
        "SET_RENDER_ENABLE_CONTROL"
    );
    let clear_dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let disabled_triggered = &clear_dispatch.operations()[0];
    let resources =
        resolve_maxwell_three_d_resources(disabled_triggered.state(), &resource_address_space())
            .unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            disabled_triggered.state(),
            &resources,
            disabled_triggered.trigger(),
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));

    let dispatch = dispatch_method(&mut channel, 0x030c / 4, 1).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_RENDER_ENABLE_CONTROL"
    );
    let clear_dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let enabled_triggered = &clear_dispatch.operations()[0];
    let resources =
        resolve_maxwell_three_d_resources(enabled_triggered.state(), &resource_address_space())
            .unwrap();
    let cache_before = cache.clone();

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            enabled_triggered.state(),
            &resources,
            enabled_triggered.trigger(),
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedConditionalLoadConstantBufferSemantics)
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn invalid_three_d_render_enable_modes_are_rejected_atomically() {
    let mut channel = three_d_channel();

    for argument in [5, 6, 7, 8, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1558 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 7,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn unsupported_three_d_render_enable_address_methods_remain_typed_fatal_errors() {
    let mut channel = three_d_channel();

    for method in [0x1550, 0x1554] {
        let frontend_before = channel.frontend();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(method / 4, 0);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "MAXWELL_B",
            }) if source.method() == GpuMethodId(method)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn non_enabled_three_d_render_modes_stop_before_neutral_lowering() {
    for (argument, expected) in [
        (0, MaxwellThreeDRenderEnableMode::Disabled),
        (2, MaxwellThreeDRenderEnableMode::Conditional),
        (3, MaxwellThreeDRenderEnableMode::RenderIfEqual),
        (4, MaxwellThreeDRenderEnableMode::RenderIfNotEqual),
    ] {
        let mut channel = three_d_channel();
        let dispatch = dispatch_method(&mut channel, 0x1558 / 4, argument).unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::ClearSurface {
                    source: dispatch.methods()[0].method().source(),
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedRenderEnableMode(mode))
                if mode == expected
        ));
        assert_eq!(cache, cache_before);
    }
}

#[test]
fn l1_configuration_is_typed_source_preserving_shader_memory_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let visible_call_before = channel
        .three_d()
        .shader_execution()
        .visible_call_limit()
        .to_owned();
    let pipeline_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_eq!(
        channel
            .three_d()
            .shader_execution()
            .l1_configuration()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected, bytes) in [
        (
            1,
            MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
            16 * 1024,
        ),
        (
            3,
            MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
            48 * 1024,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0308 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().shader_execution().l1_configuration();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_L1_CONFIGURATION"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.bytes(), bytes);
        assert_eq!(
            channel.three_d().shader_execution().visible_call_limit(),
            &visible_call_before
        );
        assert_eq!(channel.two_d(), &two_d_before);

        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            pipeline_dependencies
        );
    }

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    let resources =
        resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn invalid_l1_configurations_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0308, 1);

    for argument in [0, 2, 4, 5, 6, 7, 8, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0308 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x07,
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
    let decoded = incrementing_packet(0x0308 / 4, &[3, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.method() == GpuMethodId(0x030c) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn shader_local_memory_block_is_typed_source_preserving_and_shader_scoped() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);

    let region_dispatch = dispatch_incrementing(
        &mut channel,
        0x0790 / 4,
        &[0x04, 0x0008_0000, 0, 0x0408_0000, 0],
    )
    .unwrap();
    assert_eq!(
        region_dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        [
            "SET_SHADER_LOCAL_MEMORY_A",
            "SET_SHADER_LOCAL_MEMORY_B",
            "SET_SHADER_LOCAL_MEMORY_C",
            "SET_SHADER_LOCAL_MEMORY_D",
            "SET_SHADER_LOCAL_MEMORY_E",
        ]
    );

    let window_dispatch = dispatch_method(&mut channel, 0x077c / 4, 0xff00_0000).unwrap();
    assert_eq!(
        window_dispatch.methods()[0].metadata().method_name(),
        "SET_SHADER_LOCAL_MEMORY_WINDOW"
    );

    let local = channel.three_d().shader_execution().shader_local_memory();
    assert_eq!(local.address().unwrap().get(), 0x04_0008_0000);
    assert_eq!(local.size(), Some(0x0408_0000));
    assert_eq!(local.address_upper().raw(), Some(4));
    assert_eq!(
        local.address_upper().source(),
        Some(region_dispatch.methods()[0].method().source())
    );
    assert_eq!(local.address_lower().raw(), Some(0x0008_0000));
    assert_eq!(local.size_upper().raw(), Some(0));
    assert_eq!(local.size_lower().raw(), Some(0x0408_0000));
    let per_warp = local.default_size_per_warp();
    assert_eq!(per_warp.raw(), Some(0));
    assert_eq!(per_warp.value().unwrap().bytes(), 0);
    assert_eq!(
        per_warp.source(),
        Some(region_dispatch.methods()[4].method().source())
    );
    assert_eq!(local.window_base_address().raw(), Some(0xff00_0000));
    assert_eq!(
        local.window_base_address().source(),
        Some(window_dispatch.methods()[0].method().source())
    );
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );
    assert_eq!(channel.two_d(), &two_d_before);

    program_three_d(&mut channel, 0x2000, 0x11);
    let active_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x077c, 0xfe00_0000);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        active_dependencies
    );
}

#[test]
fn shader_local_memory_fields_are_atomic_and_cross_register_ranges_defer() {
    let mut channel = three_d_channel();

    for (method, argument, mask) in [
        (0x0790, 0x100, 0xff),
        (0x0798, 0x40, 0x3f),
        (
            0x07a0,
            MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX + 1,
            MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
        ),
    ] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask,
                ..
            }) if source.argument() == argument && defined_mask == mask
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let invalid_suffix = incrementing_packet(
        0x0790 / 4,
        &[
            4,
            0x0008_0000,
            0,
            0x0408_0000,
            MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX + 1,
        ],
    );
    assert!(matches!(
        dispatch_first(&mut channel, &invalid_suffix),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x07a0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    program_three_d(&mut channel, 0x0790, 0xff);
    program_three_d(&mut channel, 0x0794, u32::MAX);
    dispatch_incrementing(&mut channel, 0x0798 / 4, &[0, 2]).unwrap();
    let error = channel.three_d().validate_cross_registers().unwrap_err();
    assert_eq!(
        error.reason,
        "shader-local-memory region exceeds the 40-bit GPU address space"
    );
    assert_eq!(error.source.unwrap().method(), GpuMethodId(0x079c));
}

#[test]
fn active_shader_local_memory_blocks_only_draws_before_effects() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x0790, 4);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let partial_source = channel
        .three_d()
        .shader_execution()
        .shader_local_memory()
        .address_upper()
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: partial_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(9),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_SHADER_LOCAL_MEMORY_A-D"
        ))
    ));

    for (method, argument) in [
        (0x0794, 0x0008_0000),
        (0x0798, 0),
        (0x079c, 0x0408_0000),
        (0x07a0, 0),
        (0x077c, 0xff00_0000),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let source = channel
        .three_d()
        .shader_execution()
        .shader_local_memory()
        .default_size_per_warp()
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
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x07a0, 0x100);
    let cache_before = cache.clone();
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
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedShaderLocalMemorySemantics {
            default_size_per_warp,
        }) if default_size_per_warp.bytes() == 0x100
    ));
    assert_eq!(cache, cache_before);

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
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
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn visible_call_limit_is_typed_source_preserving_execution_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let timeout_before = channel
        .three_d()
        .shader_execution()
        .sm_timeout_counter_bit()
        .to_owned();
    assert_eq!(
        channel
            .three_d()
            .shader_execution()
            .visible_call_limit()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected, limit) in [
        (0, MaxwellThreeDVisibleCallLimit::Calls0, Some(0)),
        (1, MaxwellThreeDVisibleCallLimit::Calls1, Some(1)),
        (2, MaxwellThreeDVisibleCallLimit::Calls2, Some(2)),
        (3, MaxwellThreeDVisibleCallLimit::Calls4, Some(4)),
        (4, MaxwellThreeDVisibleCallLimit::Calls8, Some(8)),
        (5, MaxwellThreeDVisibleCallLimit::Calls16, Some(16)),
        (6, MaxwellThreeDVisibleCallLimit::Calls32, Some(32)),
        (7, MaxwellThreeDVisibleCallLimit::Calls64, Some(64)),
        (8, MaxwellThreeDVisibleCallLimit::Calls128, Some(128)),
        (15, MaxwellThreeDVisibleCallLimit::NoCheck, None),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0d64 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().shader_execution().visible_call_limit();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_API_VISIBLE_CALL_LIMIT"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(expected.limit(), limit);
        assert_eq!(
            channel
                .three_d()
                .shader_execution()
                .sm_timeout_counter_bit(),
            &timeout_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_visible_call_limits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0d64, 8);

    for argument in [9, 10, 11, 12, 13, 14, 0x10, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0d64 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0f,
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
    let decoded = incrementing_packet(0x0d64 / 4, &[15, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0d68)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn finite_visible_call_limits_defer_draw_validation_until_t10_evidence() {
    let mut channel = three_d_channel();
    let address_space = resource_address_space();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();
    program_three_d(&mut channel, 0x121c, 0);

    for argument in [0, 8] {
        program_three_d(&mut channel, 0x0d64, argument);
        let source = channel
            .three_d()
            .shader_execution()
            .visible_call_limit()
            .source()
            .unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let cache_before = cache.clone();

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
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));
        assert_eq!(cache, cache_before);
    }

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let cache_before = cache.clone();
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
fn visible_call_no_check_does_not_invent_a_draw_limit() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0d64, 15);
    program_three_d(&mut channel, 0x121c, 0);
    let source = channel
        .three_d()
        .shader_execution()
        .visible_call_limit()
        .source()
        .unwrap();
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
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
        &lowering_capabilities(BackendFeatures::empty()),
        &cache,
    );

    assert!(matches!(
        result,
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn active_zcull_region_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let stats_before = *channel.three_d().zcull().stats_enable();
    assert_eq!(
        channel.three_d().zcull().active_region().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for argument in [0, 1, 0x3f] {
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        let dispatch = dispatch_method(&mut channel, 0x1590 / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().zcull().active_region();
        let value = register.value().copied().unwrap();

        assert_eq!(method.metadata().method_name(), "SET_ACTIVE_ZCULL_REGION");

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.id(), argument as u8);
        assert_eq!(value.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d().zcull().stats_enable(), &stats_before);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn active_zcull_region_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1590, 0x3f);

    for argument in [0x40, 0x80, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1590 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x3f,
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
    let decoded = incrementing_packet(0x1590 / 4, &[0, 1, 0, 0, 0, 1, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x15a8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn two_sided_stencil_test_is_typed_source_preserving_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();

    for (argument, expected) in [(0, false), (1, true)] {
        let dispatch = dispatch_method(&mut channel, 0x1594 / 4, argument).unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable);

        assert_eq!(
            method.metadata().method_name(),
            "SET_TWO_SIDED_STENCIL_TEST"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::Boolean(expected))
        );
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_two_sided_stencil_test_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1594, 1);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1594 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_TWO_SIDED_STENCIL_TEST",
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
    let decoded = non_incrementing_packet_on_subchannel(0, 0x1594 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_TWO_SIDED_STENCIL_TEST",
            ..
        }) if source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn two_sided_stencil_state_affects_only_enabled_stencil_draws() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1380, 0);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);

    program_three_d(&mut channel, 0x1594, 1);
    program_three_d(&mut channel, 0x1598, 0x1e00);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );

    program_three_d(&mut channel, 0x1380, 1);
    let two_sided_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(two_sided_dependencies, disabled_dependencies);
    program_three_d(&mut channel, 0x1598, 0x1e01);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        two_sided_dependencies
    );

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();
    let source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable)
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
        Err(MaxwellThreeDLoweringError::UnsupportedStencilTestSemantics { two_sided: true })
    ));

    program_three_d(&mut channel, 0x1594, 0);
    let one_sided_source = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable)
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: one_sided_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedStencilTestSemantics { two_sided: false })
    ));

    program_three_d(&mut channel, 0x1380, 0);
    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
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
fn active_zcull_region_without_region_storage_does_not_change_draw_or_clear_semantics() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1590, 0x3f);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
    let source = channel.three_d().zcull().active_region().source().unwrap();

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
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
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
fn zcull_stats_enable_is_typed_source_preserving_isolated_three_d_state() {
    let mut channel = three_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let coverage_before = channel.three_d().coverage().clone();
    assert_eq!(
        channel.three_d().zcull().stats_enable().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDZCullStatsEnable::Disabled),
        (1, MaxwellThreeDZCullStatsEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x151c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().zcull().stats_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ZCULL_STATS"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d().coverage(), &coverage_before);
    }
}

#[test]
fn zpass_pixel_count_enable_is_typed_source_preserving_pipeline_neutral_state() {
    let mut channel = three_d_channel();
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    assert_eq!(
        channel
            .three_d()
            .counters()
            .zpass_pixel_count_enable()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDZPassPixelCountEnable::Disabled),
        (1, MaxwellThreeDZPassPixelCountEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1514 / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().counters().zpass_pixel_count_enable();

        assert_eq!(method.metadata().method_name(), "SET_ZPASS_PIXEL_COUNT");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
    }
}

#[test]
fn invalid_zpass_pixel_count_values_and_related_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1514, 1);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let channel_before = channel.clone();
        let decoded = packet(0x1514 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, channel_before);
    }

    let channel_before = channel.clone();
    let decoded = incrementing_packet(0x1514 / 4, &[0, 0x3f80_0000, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.method() == GpuMethodId(0x151c) && source.argument() == 2
    ));
    assert_ne!(channel, channel_before);
}

#[test]
fn invalid_zcull_stats_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x151c, 1);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x151c / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
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
    let decoded = incrementing_packet(0x151c / 4, &[0, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1524)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn zcull_stats_are_preserved_instrumentation_policy_without_draw_semantics() {
    let mut channel = three_d_channel();
    let address_space = resource_address_space();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();

    program_three_d(&mut channel, 0x151c, 1);
    let source = channel.three_d().zcull().stats_enable().source().unwrap();
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let cache_before = cache.clone();
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
    assert_eq!(
        result.err(),
        Some(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))
    );
    assert_eq!(
        channel.three_d().zcull().stats_enable().value(),
        Some(&MaxwellThreeDZCullStatsEnable::Enabled)
    );
    assert_eq!(cache, cache_before);

    let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
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

    program_three_d(&mut channel, 0x151c, 0);
    let source = channel.three_d().zcull().stats_enable().source().unwrap();
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let result = preflight_maxwell_three_d_operation(
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
    );
    assert_eq!(
        result.err(),
        Some(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))
    );
    assert_eq!(
        channel.three_d().zcull().stats_enable().value(),
        Some(&MaxwellThreeDZCullStatsEnable::Disabled)
    );
    assert_eq!(cache, cache_before);
}

#[test]
fn balanced_primitive_workload_is_typed_source_preserving_nonsemantic_policy() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for argument in [0, 1, 0x10, 0x11] {
        let dispatch = dispatch_method(&mut channel, 0x0374 / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let source = method.method().source();
        let value = channel
            .three_d()
            .vertex_input()
            .primitive()
            .balanced_workload()
            .value()
            .copied()
            .unwrap();

        assert_eq!(
            method.metadata().method_name(),
            "SET_BALANCED_PRIMITIVE_WORKLOAD"
        );

        assert_eq!(value.raw(), argument);
        assert_eq!(value.in_unpartitioned_mode(), argument & 1 != 0);
        assert_eq!(value.in_timesliced_mode(), argument & 0x10 != 0);
        let register = channel
            .three_d()
            .vertex_input()
            .primitive()
            .balanced_workload();
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert!(dispatch.operations().is_empty());
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn balanced_primitive_workload_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0374, 0x11);

    for argument in [2, 4, 8, 0x20, 0x8000_0000, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x0374 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_BALANCED_PRIMITIVE_WORKLOAD",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x0374 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. })
            if source.argument() == 2
    ));
    assert_ne!(channel, before);
}

#[test]
fn subtiling_perf_knobs_are_typed_source_preserving_nonsemantic_policy() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let dispatch = dispatch_incrementing(&mut channel, 0x0360 / 4, &[0x2016_4010, 0x20]).unwrap();

    assert_eq!(
        dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        ["SET_SUBTILING_PERF_KNOB_A", "SET_SUBTILING_PERF_KNOB_B"]
    );
    assert!(dispatch.operations().is_empty());

    let a_source = dispatch.methods()[0].method().source();
    let a = MaxwellThreeDSubtilingPerfKnobA::parse(0x2016_4010);
    assert_eq!(a.register_file(), 0x10);
    assert_eq!(a.pixel_output_buffer(), 0x40);
    assert_eq!(a.triangle_ram(), 0x16);
    assert_eq!(a.max_quads(), 0x20);
    assert_eq!(a.raw(), 0x2016_4010);

    let a_register = channel.three_d().shader_execution().subtiling_perf_knob_a();
    assert_eq!(a_register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(a_register.raw(), Some(0x2016_4010));
    assert_eq!(a_register.value().copied(), Some(a));
    assert_eq!(a_register.source(), Some(a_source));

    let b_source = dispatch.methods()[1].method().source();
    let b = MaxwellThreeDSubtilingPerfKnobB::new(0x20).unwrap();
    assert_eq!(b.max_primitives(), 0x20);
    assert_eq!(b.raw(), 0x20);

    let b_register = channel.three_d().shader_execution().subtiling_perf_knob_b();
    assert_eq!(b_register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(b_register.raw(), Some(0x20));
    assert_eq!(b_register.value().copied(), Some(b));
    assert_eq!(b_register.source(), Some(b_source));

    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn subtiling_perf_knob_b_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0360, 0x2016_4010);
    program_three_d(&mut channel, 0x0364, 0x20);

    for argument in [0x0000_0100, 0x0001_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0364 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0xff,
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
    let decoded = incrementing_packet(0x0360 / 4, &[u32::MAX, 0x100]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0xff,
            ..
        }) if source.method() == GpuMethodId(0x0364) && source.argument() == 0x100
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn shader_watermark_family_is_typed_source_preserving_nonsemantic_policy() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for (method, method_name, target, argument) in [
        (
            0x0f98,
            "SET_VTG_WARP_WATERMARKS",
            MaxwellThreeDShaderWatermarkTarget::VertexTessellationGeometryWarps,
            0x0001_ffff,
        ),
        (
            0x1450,
            "SET_PS_WARP_WATERMARKS",
            MaxwellThreeDShaderWatermarkTarget::PixelWarps,
            0x0080_0008,
        ),
        (
            0x1454,
            "SET_PS_REGISTER_WATERMARKS",
            MaxwellThreeDShaderWatermarkTarget::PixelRegisters,
            u32::MAX,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
        let dispatched = &dispatch.methods()[0];
        let source = dispatched.method().source();
        let value = MaxwellThreeDShaderWatermarkRange::parse(argument);
        let state = channel.three_d().shader_execution();
        let register = match target {
            MaxwellThreeDShaderWatermarkTarget::VertexTessellationGeometryWarps => {
                state.vtg_warp_watermarks()
            }
            MaxwellThreeDShaderWatermarkTarget::PixelWarps => state.ps_warp_watermarks(),
            MaxwellThreeDShaderWatermarkTarget::PixelRegisters => state.ps_register_watermarks(),
        };

        assert_eq!(dispatched.metadata().method_name(), method_name);

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.low(), argument as u16);
        assert_eq!(value.high(), (argument >> 16) as u16);
        assert_eq!(value.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn shader_watermark_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1450, 0x0080_0008);
    program_three_d(&mut channel, 0x1454, 0x0040_0004);
    let before = channel.clone();
    let decoded = incrementing_packet(0x1450 / 4, &[0x0010_0001, 0x0020_0002, 0]);

    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1458)
    ));
    assert_ne!(channel, before);
}

#[test]
fn zcull_enable_and_bounds_are_typed_source_preserving_pipeline_neutral_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let dispatch = dispatch_incrementing(&mut channel, 0x1968 / 4, &[0x11, 0]).unwrap();

    assert_eq!(
        dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        ["SET_ZCULL", "SET_ZCULL_BOUNDS"]
    );
    assert!(dispatch.operations().is_empty());

    let enable_source = dispatch.methods()[0].method().source();
    let enable = MaxwellThreeDZCullEnable::parse(0x11).unwrap();
    assert!(enable.depth());
    assert!(enable.stencil());
    assert_eq!(enable.raw(), 0x11);

    let enable_register = channel.three_d().zcull().enable();
    assert_eq!(
        enable_register.origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(enable_register.raw(), Some(0x11));
    assert_eq!(enable_register.value().copied(), Some(enable));
    assert_eq!(enable_register.source(), Some(enable_source));

    let bounds_source = dispatch.methods()[1].method().source();
    let bounds = MaxwellThreeDZCullBounds::parse(0).unwrap();
    assert!(!bounds.minimum_unbounded());
    assert!(!bounds.maximum_unbounded());
    assert_eq!(bounds.raw(), 0);

    let bounds_register = channel.three_d().zcull().bounds();
    assert_eq!(
        bounds_register.origin(),
        MaxwellThreeDRegisterOrigin::Programmed
    );
    assert_eq!(bounds_register.raw(), Some(0));
    assert_eq!(bounds_register.value().copied(), Some(bounds));
    assert_eq!(bounds_register.source(), Some(bounds_source));

    for raw in [0, 1, 0x10, 0x11] {
        assert_eq!(MaxwellThreeDZCullEnable::parse(raw).unwrap().raw(), raw);
        assert_eq!(MaxwellThreeDZCullBounds::parse(raw).unwrap().raw(), raw);
    }
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn zcull_criterion_is_typed_source_preserving_pipeline_neutral_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
    let dispatch = dispatch_method(&mut channel, 0x0dd8 / 4, 0xff00_0005).unwrap();

    assert_eq!(dispatch.methods().len(), 1);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_ZCULL_CRITERION"
    );
    assert!(dispatch.operations().is_empty());

    let source = dispatch.methods()[0].method().source();
    let criterion = MaxwellThreeDZCullCriterion::parse(0xff00_0005).unwrap();
    assert_eq!(
        criterion.stencil_function(),
        MaxwellThreeDZCullStencilFunction::NotEqual
    );
    assert!(!criterion.no_invalidate());
    assert!(!criterion.force_match());
    assert_eq!(criterion.stencil_reference(), 0);
    assert_eq!(criterion.stencil_mask(), 0xff);
    assert_eq!(criterion.raw(), 0xff00_0005);

    let register = channel.three_d().zcull().criterion();
    assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
    assert_eq!(register.raw(), Some(0xff00_0005));
    assert_eq!(register.value().copied(), Some(criterion));
    assert_eq!(register.source(), Some(source));
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_before
    );
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn zcull_criterion_decodes_the_complete_documented_field_family() {
    for raw in 0..=7 {
        let criterion = MaxwellThreeDZCullCriterion::parse(raw).unwrap();
        assert_eq!(criterion.stencil_function().raw(), raw as u8);
        assert_eq!(criterion.raw(), raw);
    }

    let criterion = MaxwellThreeDZCullCriterion::parse(0xa55a_0307).unwrap();
    assert_eq!(
        criterion.stencil_function(),
        MaxwellThreeDZCullStencilFunction::Always
    );
    assert!(criterion.no_invalidate());
    assert!(criterion.force_match());
    assert_eq!(criterion.stencil_reference(), 0x5a);
    assert_eq!(criterion.stencil_mask(), 0xa5);
    assert_eq!(criterion.raw(), 0xa55a_0307);
}

#[test]
fn zcull_criterion_invalid_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0dd8, 0xff00_0005);

    for argument in [8, 0x0000_0400, 0x0000_f000, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x0dd8 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0xffff_03ff,
                ..
            }) if source.method() == GpuMethodId(0x0dd8) && source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = incrementing_packet(0x0dd8 / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0ddc)
    ));
    assert_ne!(channel, before);
}

#[test]
fn zcull_control_reserved_bits_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1968, 0x11);
    program_three_d(&mut channel, 0x196c, 0);

    for method in [0x1968, 0x196c] {
        for argument in [2, 8, 0x20, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);

            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x11,
                    ..
                }) if source.method() == GpuMethodId(method) && source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x1968 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x11,
            ..
        }) if source.method() == GpuMethodId(0x196c) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}
