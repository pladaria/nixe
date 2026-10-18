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
fn hybrid_anti_alias_control_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let csaa_before = *channel.three_d().coverage().csaa_enable();

    for argument in [0, 1, 0x10, 0x20, 0x3f] {
        let dispatch = dispatch_method(&mut channel, 0x0754 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().hybrid_anti_alias_control();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_HYBRID_ANTI_ALIAS_CONTROL"
        );

        assert!(dispatch.operations().is_empty());
        assert_eq!(value.passes(), argument as u8 & 0xf);
        assert_eq!(
            value.centroid(),
            if argument & 0x10 == 0 {
                MaxwellThreeDHybridAntiAliasCentroid::PerFragment
            } else {
                MaxwellThreeDHybridAntiAliasCentroid::PerPass
            }
        );
        assert_eq!(value.passes_extended(), argument & 0x20 != 0);
        assert_eq!(value.raw(), argument);
        assert_eq!(value.is_single_pass_per_fragment(), argument == 1);
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(channel.three_d().coverage().csaa_enable(), &csaa_before);
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
    }

    for argument in [0x40, 0x80, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x0754 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x3f,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = incrementing_packet(0x0754 / 4, &[1, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0758)
    ));
    assert_ne!(channel, before);
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
fn invalid_color_reduction_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x10cc, 0x0000_00ff);

    program_three_d(&mut channel, 0x10e0, 0x00ff_0000);
    program_three_d(&mut channel, 0x10e4, 0x0000_00ff);
    program_three_d(&mut channel, 0x10ec, 0x0000_00ff);
    program_three_d(&mut channel, 0x10f0, 0x0000_00ff);

    for method in [0x10cc, 0x10e0, 0x10e4, 0x10ec, 0x10f0] {
        for argument in [0x0000_0100, 0x0000_ff00, 0x0100_0000, 0xff00_0000] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);

            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x00ff_00ff,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    for argument in [2, 3, u32::MAX] {
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0d9c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel.three_d(), &three_d_before);
    }

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10cc / 4, &[0x00ff_00ff, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x10d0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10e0 / 4, &[0x00ff_00ff, 0x0000_0100]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x00ff_00ff,
            ..
        }) if source.method() == GpuMethodId(0x10e4)
            && source.argument() == 0x0000_0100
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10e4 / 4, &[0x00ff_00ff, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x10e8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10ec / 4, &[0x00ff_00ff, 0x0000_0100]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x00ff_00ff,
            ..
        }) if source.method() == GpuMethodId(0x10f0)
            && source.argument() == 0x0000_0100
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10f0 / 4, &[0x00ff_00ff, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x10f4)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
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
fn iterated_blend_values_are_complete_and_invalid_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();

    for raw in 0..=3 {
        program_three_d(&mut channel, 0x0dd0, raw);
        let value = channel
            .three_d()
            .fixed_function()
            .blend_controls()
            .iterated_blend()
            .value()
            .copied()
            .unwrap();
        assert_eq!(value.raw(), raw);
        assert_eq!(value.color_enabled(), raw & 1 != 0);
        assert_eq!(value.alpha_enabled(), raw & 2 != 0);
    }
    for raw in [0, 0xff] {
        program_three_d(&mut channel, 0x0dd4, raw);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .iterated_blend_pass_count()
                .value()
                .map(|value| value.pass_count()),
            Some(raw as u8)
        );
    }

    for (method, argument, defined_mask) in [
        (0x0dd0, 4, 0x3),
        (0x0dd0, u32::MAX, 0x3),
        (0x0dd4, 0x100, 0xff),
        (0x0dd4, u32::MAX, 0xff),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: actual_mask,
                ..
            }) if source.method() == GpuMethodId(method)
                && source.argument() == argument
                && actual_mask == defined_mask
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = incrementing_packet(0x0dd0 / 4, &[0, 5, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0ddc)
    ));
    assert_ne!(channel, before);
}

#[test]
fn point_rasterization_state_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, r_mode, origin, texture_mask) in [
        (
            0,
            MaxwellThreeDPointSpriteRMode::Zero,
            MaxwellThreeDPointSpriteOrigin::Bottom,
            0,
        ),
        (
            0x000d,
            MaxwellThreeDPointSpriteRMode::FromR,
            MaxwellThreeDPointSpriteOrigin::Top,
            1,
        ),
        (
            0x1ffe,
            MaxwellThreeDPointSpriteRMode::FromS,
            MaxwellThreeDPointSpriteOrigin::Top,
            0x03ff,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1604 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().raster().point_sprite_select();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_POINT_SPRITE_SELECT"
        );
        assert_eq!(value.r_mode(), r_mode);
        assert_eq!(value.origin(), origin);
        assert_eq!(value.generated_texture_mask(), texture_mask);
        assert_eq!(value.raw(), argument);
        for texture in 0..10 {
            assert_eq!(
                value.generates_texture(texture),
                texture_mask & (1 << texture) != 0
            );
        }
        assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
    }

    for (argument, mode) in [
        (0, MaxwellThreeDPointCenterMode::OpenGl),
        (1, MaxwellThreeDPointCenterMode::Direct3D),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x165c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().raster().point_center_mode();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_POINT_CENTER_MODE"
        );
        assert_eq!(register.value(), Some(&mode));
        assert_eq!(register.raw(), Some(mode.raw()));
        assert_eq!(register.source(), Some(source));
    }

    for (method, argument, mask) in [
        (0x1604, 0x1fff, 0x1fff),
        (0x1604, 0x2000, 0x1fff),
        (0x165c, 2, 1),
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
    let decoded = incrementing_packet(0x1604 / 4, &[0, 0x100]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_PROGRAM_REGION_A",
            ..
        }) if source.method() == GpuMethodId(0x1608)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
}

#[test]
fn point_size_source_sprite_and_antialias_controls_are_typed_and_atomic() {
    let mut channel = three_d_channel();

    for argument in [0, 1, 0x0101, 0x0ff1] {
        let dispatch = dispatch_method(&mut channel, 0x1910 / 4, argument).unwrap();
        let register = channel.three_d().raster().attribute_point_size();
        let value = register.value().copied().unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ATTRIBUTE_POINT_SIZE"
        );
        assert_eq!(value.enabled(), argument & 1 != 0);
        assert_eq!(value.slot(), ((argument >> 4) & 0xff) as u8);
        assert_eq!(value.raw(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    for (method, name) in [
        (0x1520, "SET_POINT_SPRITE"),
        (0x1658, "SET_ANTI_ALIASED_POINT"),
    ] {
        for argument in [0, 1] {
            let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
            let register = if method == 0x1520 {
                channel.three_d().raster().point_sprite_enable()
            } else {
                channel.three_d().raster().anti_aliased_point_enable()
            };
            assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&(argument != 0)));
            assert_eq!(
                register.source(),
                Some(dispatch.methods()[0].method().source())
            );
        }
    }

    for (method, argument, name) in [
        (0x1910, 0x2, "SET_ATTRIBUTE_POINT_SIZE"),
        (0x1910, 0x1000, "SET_ATTRIBUTE_POINT_SIZE"),
        (0x1520, 2, "SET_POINT_SPRITE"),
        (0x1658, 2, "SET_ANTI_ALIASED_POINT"),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: actual,
                ..
            }) if source.argument() == argument && actual == name
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn triangle_fill_and_conservative_raster_controls_are_typed_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDFillViaTriangleMode::Disabled),
        (1, MaxwellThreeDFillViaTriangleMode::FillAll),
        (2, MaxwellThreeDFillViaTriangleMode::FillBoundingBox),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x113c / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().raster().fill_via_triangle();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_FILL_VIA_TRIANGLE"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
    }

    for (argument, expected) in [
        (0, MaxwellThreeDConservativeRasterEnable::Disabled),
        (1, MaxwellThreeDConservativeRasterEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1148 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().raster().conservative_raster();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CONSERVATIVE_RASTER"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
    }

    for (method, argument, name) in [
        (0x113c, 3, "SET_FILL_VIA_TRIANGLE"),
        (0x113c, u32::MAX, "SET_FILL_VIA_TRIANGLE"),
        (0x1148, 2, "SET_CONSERVATIVE_RASTER"),
        (0x1148, u32::MAX, "SET_CONSERVATIVE_RASTER"),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: actual,
                ..
            }) if source.argument() == argument && actual == name
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn polygon_smoothing_and_stipple_family_is_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (method, name, register) in [
        (0x0db4, "SET_POLY_SMOOTH", 0),
        (0x168c, "SET_POLYGON_STIPPLE", 1),
    ] {
        for argument in [0, 1] {
            let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
            let state_register = if register == 0 {
                channel.three_d().raster().polygon_smooth_enable()
            } else {
                channel.three_d().raster().polygon_stipple_enable()
            };
            assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
            assert_eq!(state_register.raw(), Some(argument));
            assert_eq!(state_register.value(), Some(&(argument != 0)));
            assert_eq!(
                state_register.source(),
                Some(dispatch.methods()[0].method().source())
            );
        }
    }

    for word in 0..MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT {
        let argument = 0xa5a5_0000 | word as u32;
        let method = 0x1700 + word as u32 * 4;
        let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = &channel.three_d().raster().polygon_stipple_pattern()[word];
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_POLYGON_STIPPLE_PATTERN"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&argument));
        assert_eq!(register.source(), Some(source));
    }

    for (method, name) in [(0x0db4, "SET_POLY_SMOOTH"), (0x168c, "SET_POLYGON_STIPPLE")] {
        for argument in [2, u32::MAX] {
            let before = channel.clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: actual,
                    ..
                }) if source.argument() == argument && actual == name
            ));
            assert_eq!(channel, before);
        }
    }
}

#[test]
fn viewport_pixel_center_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDViewportPixelCenter::HalfIntegers),
        (1, MaxwellThreeDViewportPixelCenter::Integers),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1924 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().viewport().pixel_center();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_VIEWPORT_PIXEL"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.source(), Some(source));
    }

    for argument in [2, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x1924 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_VIEWPORT_PIXEL",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn viewport_coordinate_swizzles_are_indexed_typed_and_atomic() {
    let mut channel = three_d_channel();

    for viewport in 0..MAXWELL_VIEWPORT_COUNT {
        let method = 0x0a18 + viewport as u32 * 0x20;
        let dispatch = dispatch_method(&mut channel, method / 4, 0x6420).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().fixed_function().viewport()[viewport].coordinate_swizzle();
        let value = register.value().copied().unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_VIEWPORT_COORDINATE_SWIZZLE"
        );
        assert_eq!(
            value.components(),
            [
                MaxwellThreeDViewportSwizzleComponent::PositiveX,
                MaxwellThreeDViewportSwizzleComponent::PositiveY,
                MaxwellThreeDViewportSwizzleComponent::PositiveZ,
                MaxwellThreeDViewportSwizzleComponent::PositiveW,
            ]
        );
        assert!(value.is_identity());
        assert_eq!(value.raw(), 0x6420);
        assert_eq!(register.raw(), Some(0x6420));
        assert_eq!(register.source(), Some(source));
    }

    dispatch_method(&mut channel, 0x0a18 / 4, 0x1357).unwrap();
    let value = channel.three_d().fixed_function().viewport()[0]
        .coordinate_swizzle()
        .value()
        .copied()
        .unwrap();
    assert_eq!(
        value.components(),
        [
            MaxwellThreeDViewportSwizzleComponent::NegativeW,
            MaxwellThreeDViewportSwizzleComponent::NegativeZ,
            MaxwellThreeDViewportSwizzleComponent::NegativeY,
            MaxwellThreeDViewportSwizzleComponent::NegativeX,
        ]
    );

    for argument in [0x0008, 0x8000, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x0a18 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_VIEWPORT_COORDINATE_SWIZZLE",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
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
fn invalid_blend_controls_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    for (method, argument) in [(0x1140, 0x10), (0x0fdc, 1), (0x19c0, 1)] {
        program_three_d(&mut channel, method, argument);
    }

    for (method, method_name, arguments) in [
        (
            0x1140,
            "SET_BLEND_PER_FORMAT_ENABLE",
            [1, 0x20, 0x11, u32::MAX],
        ),
        (
            0x0fdc,
            "SET_BLEND_OPT_CONTROL",
            [2, 0x10, 0x8000_0000, u32::MAX],
        ),
        (
            0x19c0,
            "SET_BLEND_FLOAT_OPTION",
            [2, 0x10, 0x8000_0000, u32::MAX],
        ),
    ] {
        for argument in arguments {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
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
    let decoded = incrementing_packet(0x1140 / 4, &[0, 0, 0, 0]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x114c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_ne!(channel.three_d(), &three_d_before);
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
fn logic_operation_family_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for argument in [0, 1] {
        let dispatch = dispatch_method(&mut channel, 0x19c4 / 4, argument).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LogicOpEnable);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_LOGIC_OP"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::Boolean(argument != 0))
        );
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
        assert!(dispatch.operations().is_empty());
    }

    for argument in [2, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x19c4 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_LOGIC_OP",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let functions = [
        MaxwellThreeDLogicOp::Clear,
        MaxwellThreeDLogicOp::And,
        MaxwellThreeDLogicOp::AndReverse,
        MaxwellThreeDLogicOp::Copy,
        MaxwellThreeDLogicOp::AndInverted,
        MaxwellThreeDLogicOp::Noop,
        MaxwellThreeDLogicOp::Xor,
        MaxwellThreeDLogicOp::Or,
        MaxwellThreeDLogicOp::Nor,
        MaxwellThreeDLogicOp::Equiv,
        MaxwellThreeDLogicOp::Invert,
        MaxwellThreeDLogicOp::OrReverse,
        MaxwellThreeDLogicOp::CopyInverted,
        MaxwellThreeDLogicOp::OrInverted,
        MaxwellThreeDLogicOp::Nand,
        MaxwellThreeDLogicOp::Set,
    ];
    for (index, expected) in functions.into_iter().enumerate() {
        let argument = 0x1500 + index as u32;
        let dispatch = dispatch_method(&mut channel, 0x19c8 / 4, argument).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LogicOpFunction);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_LOGIC_OP_FUNC"
        );
        assert_eq!(expected.raw(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::LogicOp(expected))
        );
    }

    for argument in [0, 0x14ff, 0x1510, u32::MAX] {
        let before = channel.clone();
        let decoded = incrementing_packet(0x19c4 / 4, &[1, argument]);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_LOGIC_OP_FUNC",
                reason: "unknown logic operation",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn color_write_control_and_all_target_masks_are_typed_source_preserving_state() {
    let mut channel = three_d_channel();

    for argument in [0, 1] {
        let dispatch = dispatch_method(&mut channel, 0x0f90 / 4, argument).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::SingleColorTargetWriteControl);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SINGLE_CT_WRITE_CONTROL"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::Boolean(argument != 0))
        );
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
        assert!(dispatch.operations().is_empty());
    }

    for argument in [2, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x0f90 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_SINGLE_CT_WRITE_CONTROL",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    for target in 0..8_u8 {
        let argument = 1_u32 << ((target % 4) * 4);
        let dispatch =
            dispatch_method(&mut channel, (0x1a00 + u32::from(target) * 4) / 4, argument).unwrap();
        let register = &channel.three_d().fixed_function().color_mask()[usize::from(target)];
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CT_WRITE"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().unwrap().raw(), argument);
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    let before = channel.clone();
    let decoded = packet(0x1a1c / 4, 0x2);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_CT_WRITE",
            reason: "reserved color-mask bits are set",
        }) if source.argument() == 0x2
    ));
    assert_eq!(channel, before);
}

#[test]
fn alpha_test_family_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for argument in [0, 1] {
        let dispatch = dispatch_method(&mut channel, 0x12ec / 4, argument).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALPHA_TEST"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::Boolean(argument != 0))
        );
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    for argument in [2, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x12ec / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_ALPHA_TEST",
                reason: "expected boolean 0 or 1",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let reference_bits = 0x3f00_0000;
    let dispatch = dispatch_method(&mut channel, 0x1310 / 4, reference_bits).unwrap();
    let reference = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestReference);
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_ALPHA_REF"
    );
    assert_eq!(reference.raw(), Some(reference_bits));
    assert_eq!(
        reference.value(),
        Some(&MaxwellThreeDFixedFunctionValue::FloatBits(
            MaxwellThreeDRawValue::new(reference_bits)
        ))
    );

    let functions = [
        MaxwellThreeDCompareOp::Never,
        MaxwellThreeDCompareOp::Less,
        MaxwellThreeDCompareOp::Equal,
        MaxwellThreeDCompareOp::LessEqual,
        MaxwellThreeDCompareOp::Greater,
        MaxwellThreeDCompareOp::NotEqual,
        MaxwellThreeDCompareOp::GreaterEqual,
        MaxwellThreeDCompareOp::Always,
    ];
    for (index, expected) in functions.into_iter().enumerate() {
        for argument in [1 + index as u32, 0x200 + index as u32] {
            dispatch_method(&mut channel, 0x1314 / 4, argument).unwrap();
            let function = channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::AlphaTestFunction);
            assert_eq!(function.raw(), Some(argument));
            assert_eq!(
                function.value(),
                Some(&MaxwellThreeDFixedFunctionValue::Compare(expected))
            );
        }
    }

    for argument in [0, 9, 0x1ff, 0x208, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x1314 / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_ALPHA_FUNC",
                reason: "unknown compare operation",
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }
}

#[test]
fn provoking_vertex_and_two_sided_light_are_typed_source_preserving_state() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDProvokingVertex::First),
        (1, MaxwellThreeDProvokingVertex::Last),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1684 / 4, argument).unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ProvokingVertex);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PROVOKING_VERTEX"
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::ProvokingVertex(expected))
        );
        assert_eq!(register.source(), Some(source));
        assert_eq!(expected.raw(), argument);
    }

    for argument in [0, 1] {
        let dispatch = dispatch_method(&mut channel, 0x1688 / 4, argument).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::TwoSidedLightEnable);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TWO_SIDED_LIGHT"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::Boolean(argument != 0))
        );
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    for (method, method_name) in [
        (0x1684, "SET_PROVOKING_VERTEX"),
        (0x1688, "SET_TWO_SIDED_LIGHT"),
    ] {
        for argument in [2, u32::MAX] {
            let before = channel.clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_first(&mut channel, &decoded),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: actual,
                    ..
                }) if source.argument() == argument && actual == method_name
            ));
            assert_eq!(channel, before);
        }
    }
}

#[test]
fn color_clamp_and_pixel_shader_saturate_are_typed_source_preserving_state() {
    let mut channel = three_d_channel();

    for argument in [0, 1] {
        let dispatch = dispatch_method(&mut channel, 0x2600 / 4, argument).unwrap();
        let register = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ColorClampEnable);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_COLOR_CLAMP"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(
            register.value(),
            Some(&MaxwellThreeDFixedFunctionValue::Boolean(argument != 0))
        );
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    let raw = 0x3000_0201;
    let dispatch = dispatch_method(&mut channel, 0x13a8 / 4, raw).unwrap();
    let register = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::PixelShaderSaturate);
    let value = match register.value() {
        Some(MaxwellThreeDFixedFunctionValue::PixelShaderSaturate(value)) => *value,
        other => panic!("unexpected SET_PS_SATURATE value: {other:?}"),
    };
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_PS_SATURATE"
    );
    assert_eq!(register.raw(), Some(raw));
    assert_eq!(
        register.source(),
        Some(dispatch.methods()[0].method().source())
    );
    assert_eq!(value.raw(), raw);
    assert_eq!(value.output_enabled(0), Some(true));
    assert_eq!(
        value.clamp_range(0),
        Some(MaxwellThreeDPixelShaderClampRange::ZeroToOne)
    );
    assert_eq!(value.output_enabled(2), Some(false));
    assert_eq!(
        value.clamp_range(2),
        Some(MaxwellThreeDPixelShaderClampRange::MinusOneToOne)
    );
    assert_eq!(value.output_enabled(7), Some(true));
    assert_eq!(
        value.clamp_range(7),
        Some(MaxwellThreeDPixelShaderClampRange::MinusOneToOne)
    );
    assert_eq!(value.output_enabled(8), None);
    assert_eq!(value.clamp_range(8), None);

    for (method, argument, method_name) in [
        (0x2600, 2, "SET_COLOR_CLAMP"),
        (0x2600, u32::MAX, "SET_COLOR_CLAMP"),
        (0x13a8, 0x4, "SET_PS_SATURATE"),
        (0x13a8, 0x8000_0000, "SET_PS_SATURATE"),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: actual,
                ..
            }) if source.argument() == argument && actual == method_name
        ));
        assert_eq!(channel, before);
    }
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
fn invalid_polygon_clip_generated_edge_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0f8c, 0);

    for argument in [2, 0x8000_0000, u32::MAX] {
        let before = channel.clone();
        let decoded = packet(0x0f8c / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_LINE_MODE_POLYGON_CLIP",
                ..
            }) if source.argument() == argument
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x0f8c / 4, &[1, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_LINE_MODE_POLYGON_CLIP",
            ..
        }) if source.method() == GpuMethodId(0x0f8c) && source.argument() == 2
    ));
    assert_ne!(channel, before);
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
fn line_smoothing_width_and_stipple_family_is_typed_source_preserving_and_atomic() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDAntiAliasedLineEnable::Disabled),
        (1, MaxwellThreeDAntiAliasedLineEnable::Enabled),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1570 / 4, argument).unwrap();
        let register = channel.three_d().line().anti_aliased_line_enable();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ANTI_ALIASED_LINE"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }

    let width = 0x3f80_0000;
    program_three_d(&mut channel, 0x13b4, width);
    assert_eq!(
        channel.three_d().line().aliased_line_width().raw(),
        Some(width)
    );
    assert_eq!(
        channel.three_d().line().aliased_line_width().value(),
        Some(&MaxwellThreeDRawValue::new(width))
    );

    program_three_d(&mut channel, 0x166c, 1);
    assert_eq!(
        channel.three_d().line().stipple_enable().value(),
        Some(&true)
    );
    let parameters_raw = 0xabcd04;
    program_three_d(&mut channel, 0x1680, parameters_raw);
    let parameters = channel
        .three_d()
        .line()
        .stipple_parameters()
        .value()
        .copied()
        .unwrap();
    assert_eq!(parameters.raw(), parameters_raw);
    assert_eq!(parameters.factor(), 4);
    assert_eq!(parameters.pattern(), 0xabcd);

    for (method, argument, method_name) in [
        (0x1570, 2, "SET_ANTI_ALIASED_LINE"),
        (0x166c, 2, "SET_LINE_STIPPLE"),
        (0x1680, 0x0100_0000, "SET_LINE_STIPPLE_PARAMETERS"),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: actual,
                ..
            }) if source.argument() == argument && actual == method_name
        ));
        assert_eq!(channel, before);
    }
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

#[test]
fn invalid_coverage_output_controls_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();

    for (method, argument, defined_mask) in [
        (0x11f8, 2, 0x71),
        (0x11f8, 8, 0x71),
        (0x11f8, 0x80, 0x71),
        (0x16b4, 4, 3),
        (0x16b4, u32::MAX, 3),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: actual_mask,
                ..
            }) if source.argument() == argument && actual_mask == defined_mask
        ));
        assert_eq!(channel, before);
    }

    for (method, arguments) in [(0x11f8, [0x71, 2]), (0x16b4, [3, 4])] {
        let before = channel.clone();
        let decoded = non_incrementing_packet_on_subchannel(0, method / 4, &arguments);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.argument() == arguments[1]
        ));
        assert_ne!(channel, before);
    }
}

#[test]
fn invalid_tir_values_and_failed_packet_keeps_valid_prefix() {
    let mut channel = three_d_channel();

    for (method, argument, defined_mask) in [
        (0x0fb4, 2, 0x3),
        (0x0fb4, 3, 0x3),
        (0x0fd4, 4, 0x3),
        (0x0fd4, u32::MAX, 0x3),
        (0x0fd8, 2, 0x1),
        (0x0fd8, u32::MAX, 0x1),
        (0x1130, 4, 0x13),
        (0x1130, 8, 0x13),
        (0x1130, 0x20, 0x13),
    ] {
        let before = channel.clone();
        let decoded = packet(method / 4, argument);
        assert!(matches!(
            dispatch_first(&mut channel, &decoded),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: actual_mask,
                ..
            }) if source.argument() == argument && actual_mask == defined_mask
        ));
        assert_eq!(channel, before);
    }

    let before = channel.clone();
    let decoded = non_incrementing_packet_on_subchannel(0, 0x1130 / 4, &[0x13, 4]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.argument() == 4
    ));
    assert_ne!(channel, before);

    let before = channel.clone();
    let decoded = incrementing_packet(0x0fd4 / 4, &[3, 2]);
    assert!(matches!(
        dispatch_first(&mut channel, &decoded),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.method() == GpuMethodId(0x0fd8) && source.argument() == 2
    ));
    assert_ne!(channel, before);
}
