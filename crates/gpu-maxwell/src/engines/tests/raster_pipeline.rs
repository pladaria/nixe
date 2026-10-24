use super::*;

#[test]
fn disabled_depth_bias_parameters_preserve_float_bits_without_triggering_work() {
    for (method, register, name) in [
        (
            0x156c,
            MaxwellThreeDFixedFunctionRegister::SlopeScaleDepthBias,
            "SET_SLOPE_SCALE_DEPTH_BIAS",
        ),
        (
            0x15bc,
            MaxwellThreeDFixedFunctionRegister::DepthBias,
            "SET_DEPTH_BIAS",
        ),
        (
            0x187c,
            MaxwellThreeDFixedFunctionRegister::DepthBiasClamp,
            "SET_DEPTH_BIAS_CLAMP",
        ),
    ] {
        let mut channel = three_d_channel();
        for bits in [
            0,
            (-2.5_f32).to_bits(),
            f32::INFINITY.to_bits(),
            0x7fc0_1234,
        ] {
            let dispatch = dispatch_method(&mut channel, method / 4, bits).unwrap();
            assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
            assert!(dispatch.operations().is_empty());
            let value = channel.three_d().fixed_function().register(register);
            assert_eq!(
                value.value(),
                Some(&MaxwellThreeDFixedFunctionValue::FloatBits(
                    MaxwellThreeDRawValue::new(bits)
                ))
            );
            assert_eq!(
                value.source(),
                Some(dispatch.methods()[0].method().source())
            );
        }
    }
}

#[test]
fn sm_timeout_counter_bit_is_bounded_source_preserving_state() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .shader_execution()
            .sm_timeout_counter_bit()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for argument in [0, 0x17, MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX] {
        let dispatch = dispatch_method(&mut channel, 0x0de4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDSmTimeoutCounterBit::new(argument).unwrap();
        let register = channel
            .three_d()
            .shader_execution()
            .sm_timeout_counter_bit();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SM_TIMEOUT_INTERVAL"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(u32::from(value.get()), argument);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn sm_timeout_reserved_bits_are_rejected_atomically() {
    let mut channel = three_d_channel();

    for argument in [0x40, 0x80, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0de4 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }
}

#[test]
fn csaa_enable_is_typed_source_preserving_state_isolated_from_multisample() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel.three_d().coverage().csaa_enable().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDCsaaEnable::Disabled),
        (1, MaxwellThreeDCsaaEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x15b4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().csaa_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CSAA_ENABLE"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn sample_location_family_preserves_all_coordinates_and_keeps_valid_prefix_on_failure() {
    let mut channel = three_d_channel();
    let arguments = [0x8888_8888, 0x0123_4567, 0x89ab_cdef, 0xfedc_ba98];
    let dispatch = dispatch_incrementing(&mut channel, 0x11e0 / 4, &arguments).unwrap();

    assert_eq!(
        dispatch.methods().len(),
        MAXWELL_SAMPLE_LOCATION_GROUP_COUNT
    );
    for (group, argument) in arguments.into_iter().enumerate() {
        let source = dispatch.methods()[group].method().source();
        let register = &channel.three_d().coverage().sample_locations()[group];
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[group].metadata().method_name(),
            [
                "SAMPLE_LOCATIONS(0)",
                "SAMPLE_LOCATIONS(1)",
                "SAMPLE_LOCATIONS(2)",
                "SAMPLE_LOCATIONS(3)",
            ][group]
        );

        assert_eq!(value.raw(), argument);
        for (index, location) in value.locations().iter().enumerate() {
            assert_eq!(location.x(), ((argument >> (index * 8)) & 0xf) as u8);
            assert_eq!(location.y(), ((argument >> (index * 8 + 4)) & 0xf) as u8);
        }
        assert_eq!(value.is_centered(), argument == 0x8888_8888);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
    }
    assert!(dispatch.operations().is_empty());

    let before = channel.clone();
    let decoded = incrementing_packet(0x11e0 / 4, &[0x8888_8888; 5]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_OFFSET_RENDER_TARGET_INDEX",
            ..
        }) if source.method() == GpuMethodId(0x11f0)
    ));
    assert_ne!(channel, before);
}

#[test]
fn ps_output_sample_mask_usage_is_typed_source_preserving_coverage_state() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let csaa_before = *channel.three_d().coverage().csaa_enable();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .coverage()
            .ps_output_sample_mask_usage()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for argument in 0..=3 {
        let dispatch = dispatch_method(&mut channel, 0x0300 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().ps_output_sample_mask_usage();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PS_OUTPUT_SAMPLE_MASK_USAGE"
        );

        assert_eq!(value.enabled(), argument & 1 != 0);
        assert_eq!(value.qualify_by_anti_alias_enable(), argument & 2 != 0);
        assert_eq!(value.effective(Some(false)), Some(matches!(argument, 1)));
        assert_eq!(value.effective(Some(true)), Some(matches!(argument, 1 | 3)));
        assert_eq!(
            value.effective(None),
            match argument {
                1 => Some(true),
                3 => None,
                _ => Some(false),
            }
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d().coverage().csaa_enable(), &csaa_before);
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert!(dispatch.operations().is_empty());
    }
}

#[test]
fn primitive_circular_buffer_throttle_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .circular_buffer_throttle()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for argument in [0, 1, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX] {
        let dispatch = dispatch_method(&mut channel, 0x02d0 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .vertex_input()
            .primitive()
            .circular_buffer_throttle();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PRIM_CIRCULAR_BUFFER_THROTTLE"
        );

        assert_eq!(value.primitive_area(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.two_d(), &two_d_before);
        assert!(dispatch.operations().is_empty());
    }
}

#[test]
fn unorm8_color_reduction_thresholds_are_typed_source_preserving_and_isolated() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .color_reduction()
            .thresholds_unorm8()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, all_hit_once, all_covered) in [
        (0, 0, 0),
        (0x0000_00ff, 0xff, 0),
        (0x00ff_0000, 0, 0xff),
        (0x00ff_00ff, 0xff, 0xff),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x10cc / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDColorReductionThresholdsUnorm8::new(
            MaxwellThreeDUnorm8::new(all_hit_once),
            MaxwellThreeDUnorm8::new(all_covered),
        );
        let register = channel.three_d().color_reduction().thresholds_unorm8();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_REDUCE_COLOR_THRESHOLDS_UNORM8"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
        assert_eq!(value.all_covered().raw(), all_covered);
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().render_targets(), &render_targets_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
    program_three_d(&mut channel, 0x10cc, 0x0056_0078);
    assert_eq!(
        channel.three_d().color_reduction().thresholds_unorm10(),
        &unorm10_before
    );
}

#[test]
fn unorm10_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .color_reduction()
            .thresholds_unorm10()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, all_hit_once, all_covered) in [
        (0, 0, 0),
        (0x0000_00ff, 0xff, 0),
        (0x00ff_0000, 0, 0xff),
        (0x00ff_00ff, 0xff, 0xff),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x10e0 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDColorReductionThresholdsUnorm10::new(
            MaxwellThreeDUnorm8::new(all_hit_once),
            MaxwellThreeDUnorm8::new(all_covered),
        );
        let register = channel.three_d().color_reduction().thresholds_unorm10();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_REDUCE_COLOR_THRESHOLDS_UNORM10"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
        assert_eq!(value.all_covered().raw(), all_covered);
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm8(),
            &unorm8_before
        );
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().render_targets(), &render_targets_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn unorm16_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    program_three_d(&mut channel, 0x10e0, 0x0056_0078);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .color_reduction()
            .thresholds_unorm16()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, all_hit_once, all_covered) in [
        (0, 0, 0),
        (0x0000_00ff, 0xff, 0),
        (0x00ff_0000, 0, 0xff),
        (0x00ff_00ff, 0xff, 0xff),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x10e4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDColorReductionThresholdsUnorm16::new(
            MaxwellThreeDUnorm8::new(all_hit_once),
            MaxwellThreeDUnorm8::new(all_covered),
        );
        let register = channel.three_d().color_reduction().thresholds_unorm16();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_REDUCE_COLOR_THRESHOLDS_UNORM16"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
        assert_eq!(value.all_covered().raw(), all_covered);
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm8(),
            &unorm8_before
        );
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm10(),
            &unorm10_before
        );
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().render_targets(), &render_targets_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
    program_three_d(&mut channel, 0x10e0, 0x009a_00bc);
    assert_eq!(
        channel.three_d().color_reduction().thresholds_unorm16(),
        &unorm16_before
    );
}

#[test]
fn fp16_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    program_three_d(&mut channel, 0x10e0, 0x0056_0078);
    program_three_d(&mut channel, 0x10e4, 0x009a_00bc);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
    let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .color_reduction()
            .thresholds_fp16()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, all_hit_once, all_covered) in [
        (0, 0, 0),
        (0x0000_00ff, 0xff, 0),
        (0x00ff_0000, 0, 0xff),
        (0x00ff_00ff, 0xff, 0xff),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x10ec / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDColorReductionThresholdsFp16::new(
            MaxwellThreeDColorReductionFp16Threshold::new(all_hit_once),
            MaxwellThreeDColorReductionFp16Threshold::new(all_covered),
        );
        let register = channel.three_d().color_reduction().thresholds_fp16();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_REDUCE_COLOR_THRESHOLDS_FP16"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
        assert_eq!(value.all_covered().raw(), all_covered);
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm8(),
            &unorm8_before
        );
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm10(),
            &unorm10_before
        );
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm16(),
            &unorm16_before
        );
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().render_targets(), &render_targets_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn srgb8_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    program_three_d(&mut channel, 0x10e0, 0x0056_0078);
    program_three_d(&mut channel, 0x10e4, 0x009a_00bc);
    program_three_d(&mut channel, 0x10ec, 0x00de_00f0);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
    let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
    let fp16_before = *channel.three_d().color_reduction().thresholds_fp16();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .color_reduction()
            .thresholds_srgb8()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, all_hit_once, all_covered) in [
        (0, 0, 0),
        (0x0000_00ff, 0xff, 0),
        (0x00ff_0000, 0, 0xff),
        (0x00ff_00ff, 0xff, 0xff),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x10f0 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDColorReductionThresholdsSrgb8::new(
            MaxwellThreeDColorReductionSrgb8Threshold::new(all_hit_once),
            MaxwellThreeDColorReductionSrgb8Threshold::new(all_covered),
        );
        let register = channel.three_d().color_reduction().thresholds_srgb8();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_REDUCE_COLOR_THRESHOLDS_SRGB8"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
        assert_eq!(value.all_covered().raw(), all_covered);
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm8(),
            &unorm8_before
        );
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm10(),
            &unorm10_before
        );
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm16(),
            &unorm16_before
        );
        assert_eq!(
            channel.three_d().color_reduction().thresholds_fp16(),
            &fp16_before
        );
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().render_targets(), &render_targets_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_csaa_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x15b4, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x15b4 / 4, argument);

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
    let decoded = incrementing_packet(0x15b4 / 4, &[1, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. })
            if source.method() == GpuMethodId(0x15b8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_ps_output_sample_mask_usage_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0300, 3);

    for argument in [4, 7, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0300 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 3,
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
    let decoded = incrementing_packet(0x0300 / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::MissingCapability { source, .. })
            if source.method() == GpuMethodId(0x0304)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_primitive_circular_buffer_throttle_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x02d0, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX);

    for argument in [0x0040_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x02d0 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PRIM_CIRCULAR_BUFFER_THROTTLE",
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
    let decoded = incrementing_packet(0x02d0 / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x02d4)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn patch_size_is_typed_source_preserving_and_reserved_bits_are_atomic() {
    let mut channel = three_d_channel();

    for argument in [0, 1, 3, 32, 0xff] {
        let dispatch = dispatch_method(&mut channel, 0x0dcc / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDPatchSize::new(argument as u8);
        let register = channel.three_d().vertex_input().primitive().patch_size();

        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_PATCH");

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.control_points(), argument as u8);
        assert_eq!(value.raw(), argument);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&value));
        assert_eq!(register.source(), Some(source));
    }

    for argument in [0x100, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0dcc / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PATCH",
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
    let decoded = incrementing_packet(0x0dcc / 4, &[3, 0, 0, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0ddc)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn iterated_blend_family_is_typed_source_preserving_and_disabled_pipeline_neutral() {
    let mut channel = three_d_channel();
    let ordinary_blend_before = *channel.three_d().fixed_function().single_rop_control();
    let dispatch = dispatch_incrementing(&mut channel, 0x0dd0 / 4, &[0, 5]).unwrap();

    assert_eq!(
        dispatch
            .methods()
            .iter()
            .map(|method| method.metadata().method_name())
            .collect::<Vec<_>>(),
        ["SET_ITERATED_BLEND", "SET_ITERATED_BLEND_PASS"]
    );
    assert!(dispatch.operations().is_empty());
    let controls = channel.three_d().fixed_function().blend_controls();
    let value = controls.iterated_blend().value().copied().unwrap();
    assert!(!value.color_enabled());
    assert!(!value.alpha_enabled());
    assert!(!value.enabled());
    assert_eq!(value.raw(), 0);
    assert_eq!(controls.iterated_blend().raw(), Some(0));
    assert_eq!(
        controls.iterated_blend().source(),
        Some(dispatch.methods()[0].method().source())
    );
    let pass_count = controls
        .iterated_blend_pass_count()
        .value()
        .copied()
        .unwrap();
    assert_eq!(pass_count.pass_count(), 5);
    assert_eq!(pass_count.raw(), 5);
    assert_eq!(controls.iterated_blend_pass_count().raw(), Some(5));
    assert_eq!(
        controls.iterated_blend_pass_count().source(),
        Some(dispatch.methods()[1].method().source())
    );
    assert_eq!(
        channel.three_d().fixed_function().single_rop_control(),
        &ordinary_blend_before
    );
}

#[test]
fn edge_flag_state_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDEdgeFlag::Disabled),
        (1, MaxwellThreeDEdgeFlag::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x15e4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().raster().edge_flag();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_EDGE_FLAG"
        );
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.raw(), Some(expected.raw()));
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.source(), Some(source));
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet(0x15e4 / 4, 2);
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

    let decoded = incrementing_packet(0x15e4 / 4, &[0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding { source, method_name: "INLINE_INDEX", .. })
            if source.method() == GpuMethodId(0x15e8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_array_restart_is_typed_source_preserving_and_isolated_from_indexed_restart() {
    let mut channel = three_d_channel();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .vertex_array_restart_enabled()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    program_three_d(&mut channel, 0x1644, 1);
    program_three_d(&mut channel, 0x1648, 0xfeed_beef);
    let indexed_enable_before = channel
        .three_d()
        .vertex_input()
        .primitive()
        .restart_enabled()
        .to_owned();
    let indexed_index_before = channel
        .three_d()
        .vertex_input()
        .primitive()
        .restart_index()
        .to_owned();

    for (argument, expected) in [
        (0, MaxwellThreeDVertexArrayPrimitiveRestartEnable::Disabled),
        (1, MaxwellThreeDVertexArrayPrimitiveRestartEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0de8 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let primitive = channel.three_d().vertex_input().primitive();
        let register = primitive.vertex_array_restart_enabled();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(primitive.restart_enabled(), &indexed_enable_before);
        assert_eq!(primitive.restart_index(), &indexed_index_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let vertex_array_before = channel
        .three_d()
        .vertex_input()
        .primitive()
        .vertex_array_restart_enabled()
        .to_owned();
    program_three_d(&mut channel, 0x1644, 0);
    program_three_d(&mut channel, 0x1648, 7);
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .vertex_array_restart_enabled(),
        &vertex_array_before
    );
}

#[test]
fn invalid_vertex_array_restart_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0de8, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0de8 / 4, argument);

        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY",
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
    let decoded = incrementing_packet(0x0de8 / 4, &[1, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0dec)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn shade_mode_is_typed_and_source_preserving() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x12cc, 0);
    let depth_test_before = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable)
        .to_owned();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0x1d00, MaxwellThreeDShadeMode::Flat),
        (0x1d01, MaxwellThreeDShadeMode::Smooth),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x12d4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ShadeMode);

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SHADE_MODE"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value().copied(),
            Some(MaxwellThreeDFixedFunctionValue::ShadeMode(expected))
        );
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable),
            &depth_test_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_shade_modes_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x12d4, 0x1d00);

    for argument in [0, 1, 0x1cff, 0x1d02, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x12d4 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_SHADE_MODE",
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
    // 0x12e0 is now decoded; use its reserved footprint encoding to retain
    // the valid-prefix failure test rather than relying on an unknown method.
    let decoded = incrementing_packet(0x12d4 / 4, &[0x1d01, 0, 0, 3]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, defined_mask: 0xf, .. })
            if source.method() == GpuMethodId(0x12e0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn blend_controls_are_typed_source_preserving_and_family_isolated() {
    let mut channel = three_d_channel();
    let common_before = *channel.three_d().fixed_function().single_rop_control();
    let per_target_enable_before = *channel.three_d().fixed_function().blend_enable();
    let per_target_state_before = *channel.three_d().fixed_function().per_target_blend();
    let two_d_before = channel.two_d().clone();

    for (argument, value) in [
        (0, MaxwellThreeDBlendPerFormatEnable::Disabled),
        (0x10, MaxwellThreeDBlendPerFormatEnable::Enabled),
    ] {
        let pixel_kill_before = *channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .float_pixel_kill_enable();
        let zero_product_before = *channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .zero_times_anything_is_zero();
        let dispatch = dispatch_method(&mut channel, 0x1140 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_controls().per_format_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_PER_FORMAT_ENABLE"
        );

        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(
            fixed.blend_controls().float_pixel_kill_enable(),
            &pixel_kill_before
        );
        assert_eq!(
            fixed.blend_controls().zero_times_anything_is_zero(),
            &zero_product_before
        );
    }

    for (argument, value) in [
        (0, MaxwellThreeDBlendFloatPixelKillEnable::Disallowed),
        (1, MaxwellThreeDBlendFloatPixelKillEnable::Allowed),
    ] {
        let per_format_before = *channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .per_format_enable();
        let zero_product_before = *channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .zero_times_anything_is_zero();
        let dispatch = dispatch_method(&mut channel, 0x0fdc / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_controls().float_pixel_kill_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_OPT_CONTROL"
        );

        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(
            fixed.blend_controls().per_format_enable(),
            &per_format_before
        );
        assert_eq!(
            fixed.blend_controls().zero_times_anything_is_zero(),
            &zero_product_before
        );
    }

    for (argument, value) in [
        (0, MaxwellThreeDBlendZeroTimesAnythingIsZero::Disabled),
        (1, MaxwellThreeDBlendZeroTimesAnythingIsZero::Enabled),
    ] {
        let per_format_before = *channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .per_format_enable();
        let pixel_kill_before = *channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .float_pixel_kill_enable();
        let dispatch = dispatch_method(&mut channel, 0x19c0 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_controls().zero_times_anything_is_zero();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_FLOAT_OPTION"
        );

        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));
        assert_eq!(register.source(), Some(source));
        assert_eq!(value.raw(), argument);
        assert_eq!(
            fixed.blend_controls().per_format_enable(),
            &per_format_before
        );
        assert_eq!(
            fixed.blend_controls().float_pixel_kill_enable(),
            &pixel_kill_before
        );
    }

    let fixed = channel.three_d().fixed_function();
    assert_eq!(fixed.single_rop_control(), &common_before);
    assert_eq!(fixed.blend_enable(), &per_target_enable_before);
    assert_eq!(fixed.per_target_blend(), &per_target_state_before);
    assert_eq!(channel.two_d(), &two_d_before);
    program_three_d(&mut channel, 0x12e4, 0);
    program_three_d(&mut channel, 0x135c, 0);
    program_three_d(&mut channel, 0x1140, 0);
    program_three_d(&mut channel, 0x0fdc, 0);
    program_three_d(&mut channel, 0x19c0, 0);

    program_three_d(&mut channel, 0x1140, 0x10);
    program_three_d(&mut channel, 0x0fdc, 1);
    program_three_d(&mut channel, 0x19c0, 1);

    program_three_d(&mut channel, 0x1140, 0);
    program_three_d(&mut channel, 0x0fdc, 0);
    program_three_d(&mut channel, 0x19c0, 0);
    program_three_d(&mut channel, 0x135c, 1);

    program_three_d(&mut channel, 0x0fdc, 1);
    program_three_d(&mut channel, 0x1140, 0x10);
    program_three_d(&mut channel, 0x19c0, 1);
}

#[test]
fn single_rop_control_is_typed_source_preserving_and_family_isolated() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x12e4, 1);
    program_three_d(&mut channel, 0x1360, 1);
    program_three_d(&mut channel, 0x1e00, 0);
    let per_target_mode_before = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable)
        .to_owned();
    let per_target_enable_before = *channel.three_d().fixed_function().blend_enable();
    let per_target_state_before = *channel.three_d().fixed_function().per_target_blend();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .single_rop_control()
            .origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDSingleRopControl::Disabled),
        (1, MaxwellThreeDSingleRopControl::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x135c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.single_rop_control();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SINGLE_ROP_CONTROL"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(
            fixed.register(MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable),
            &per_target_mode_before
        );
        assert_eq!(fixed.blend_enable(), &per_target_enable_before);
        assert_eq!(fixed.per_target_blend(), &per_target_state_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    let common_before = channel
        .three_d()
        .fixed_function()
        .single_rop_control()
        .to_owned();
    program_three_d(&mut channel, 0x12e4, 0);
    program_three_d(&mut channel, 0x1360, 0);
    program_three_d(&mut channel, 0x1e00, 1);
    assert_eq!(
        channel.three_d().fixed_function().single_rop_control(),
        &common_before
    );
}

#[test]
fn invalid_single_rop_control_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x135c, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x135c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_SINGLE_ROP_CONTROL",
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
    let decoded = incrementing_packet(0x135c / 4, &[1, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_BLEND",
            ..
        }) if source.method() == GpuMethodId(0x1360)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn aliased_line_width_selector_is_typed_source_preserving_isolated_state() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let coverage_before = channel.three_d().coverage().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .line()
            .aliased_line_width_enable()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDAliasedLineWidthEnable::Disabled),
        (1, MaxwellThreeDAliasedLineWidthEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x020c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().line().aliased_line_width_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALIASED_LINE_WIDTH_ENABLE"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().coverage(), &coverage_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn polygon_clip_generated_edge_is_typed_source_preserving_line_state() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let two_d_before = channel.two_d().clone();
    assert_eq!(
        channel
            .three_d()
            .line()
            .polygon_clip_generated_edge()
            .origin(),
        MaxwellThreeDRegisterOrigin::VerifiedReset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDPolygonClipGeneratedEdge::DrawLine),
        (1, MaxwellThreeDPolygonClipGeneratedEdge::DoNotDrawLine),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0f8c / 4, argument).unwrap();
        let method = &dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().line().polygon_clip_generated_edge();

        assert_eq!(
            method.metadata().method_name(),
            "SET_LINE_MODE_POLYGON_CLIP"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_aliased_line_width_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x020c, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x020c / 4, argument);

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
    let decoded = incrementing_packet(0x020c / 4, &[1, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0214)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn tir_mode_and_controls_are_typed_source_preserving_state() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDTirMode::Disabled),
        (1, MaxwellThreeDTirMode::RasterNTargetM),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0fb4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().tir_mode();

        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_TIR");

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
    }

    for argument in [0, 1, 2, 3, 0x10, 0x11, 0x12, 0x13] {
        let dispatch = dispatch_method(&mut channel, 0x1130 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().tir_control();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TIR_CONTROL"
        );
        assert_eq!(value.raw(), argument);
        assert_eq!(
            value.z_pass_pixel_count_uses_raster_samples(),
            argument & 1 != 0
        );
        assert_eq!(value.reduce_coverage(), argument & 2 != 0);
        assert_eq!(
            value.alpha_to_coverage_uses_raster_samples(),
            argument & 0x10 != 0
        );
        assert_eq!(register.source(), Some(source));
    }

    for (argument, expected) in [
        (0, MaxwellThreeDTirModulationComponentSelect::NoModulation),
        (1, MaxwellThreeDTirModulationComponentSelect::ModulateRgb),
        (
            2,
            MaxwellThreeDTirModulationComponentSelect::ModulateAlphaOnly,
        ),
        (3, MaxwellThreeDTirModulationComponentSelect::ModulateRgba),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0fd4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().tir_modulation();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TIR_MODULATION"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
    }

    for (argument, expected) in [
        (0, MaxwellThreeDTirModulationFunction::Linear),
        (1, MaxwellThreeDTirModulationFunction::Table),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x0fd8 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().tir_modulation_function();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TIR_MODULATION_FUNCTION"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
    }
}

#[test]
fn coverage_output_controls_are_typed_source_preserving_state() {
    let mut channel = three_d_channel();

    for color_target in 0..8 {
        for enabled in [false, true] {
            let argument = (color_target << 4) | u32::from(enabled);
            let dispatch = dispatch_method(&mut channel, 0x11f8 / 4, argument).unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().coverage().coverage_to_color();
            let value = register.value().copied().unwrap();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_COVERAGE_TO_COLOR"
            );
            assert_eq!(value.enabled(), enabled);
            assert_eq!(value.color_target(), color_target as u8);
            assert_eq!(value.raw(), argument);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.source(), Some(source));
        }
    }

    for argument in 0..4 {
        let dispatch = dispatch_method(&mut channel, 0x16b4 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().alpha_to_coverage_override();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALPHA_TO_COVERAGE_OVERRIDE"
        );
        assert_eq!(value.qualify_by_anti_alias_enable(), argument & 1 != 0);
        assert_eq!(
            value.qualify_by_pixel_shader_sample_mask(),
            argument & 2 != 0
        );
        assert_eq!(value.raw(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
    }
}
