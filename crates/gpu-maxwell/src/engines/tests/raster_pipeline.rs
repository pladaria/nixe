use super::*;

#[test]
fn sm_timeout_counter_bit_is_bounded_source_preserving_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x0de4 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderExecution(
                MaxwellThreeDShaderExecutionStateWrite::SmTimeoutCounterBit { value, source }
            ))
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
    let mut channel = channel();
    bind_three_d(&mut channel);

    for argument in [0x40, 0x80, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0de4 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
fn programmed_sm_timeout_stops_shader_execution_before_cache_publication() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let decoded = packet(0x0de4 / 4, 0x17);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: dispatch.methods()[0].method().source(),
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedSmTimeoutIntervalSemantics(value))
            if value.get() == 0x17
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn csaa_enable_is_typed_source_preserving_state_isolated_from_multisample() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x15b4 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().csaa_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CSAA_ENABLE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Coverage(
                MaxwellThreeDCoverageStateWrite::CsaaEnable {
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
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn ps_output_sample_mask_usage_is_typed_source_preserving_coverage_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x0300 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().coverage().ps_output_sample_mask_usage();
        let value = register.value().copied().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PS_OUTPUT_SAMPLE_MASK_USAGE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Coverage(
                MaxwellThreeDCoverageStateWrite::PsOutputSampleMaskUsage { value, source }
            ))
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
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        let decoded = packet(0x02d0 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::PrimitiveCircularBufferThrottle { value, source }
            ))
        );
        assert_eq!(value.primitive_area(), argument);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
        assert!(dispatch.operations().is_empty());
    }
}

#[test]
fn unorm8_color_reduction_thresholds_are_typed_source_preserving_and_isolated() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
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
        let decoded = packet(0x10cc / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm8 { value, source }
            ))
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
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            pipeline_dependencies_before
        );
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
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
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
        let decoded = packet(0x10e0 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm10 { value, source }
            ))
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
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            pipeline_dependencies_before
        );
    }
}

#[test]
fn unorm16_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    program_three_d(&mut channel, 0x10e0, 0x0056_0078);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
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
        let decoded = packet(0x10e4 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm16 { value, source }
            ))
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
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            pipeline_dependencies_before
        );
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
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x10cc, 0x0012_0034);
    program_three_d(&mut channel, 0x10e0, 0x0056_0078);
    program_three_d(&mut channel, 0x10e4, 0x009a_00bc);
    let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
    let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
    let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
    let fixed_function_before = channel.three_d().fixed_function().clone();
    let render_targets_before = channel.three_d().render_targets().clone();
    let two_d_before = channel.two_d().clone();
    let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
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
        let decoded = packet(0x10ec / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                MaxwellThreeDColorReductionStateWrite::ThresholdsFp16 { value, source }
            ))
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
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            pipeline_dependencies_before
        );
    }
}

#[test]
fn srgb8_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
    let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
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
        let decoded = packet(0x10f0 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                MaxwellThreeDColorReductionStateWrite::ThresholdsSrgb8 { value, source }
            ))
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
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            pipeline_dependencies_before
        );
    }
}

#[test]
fn invalid_color_reduction_values_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
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
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x10d0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10e0 / 4, &[0x00ff_00ff, 0x0000_0100]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x00ff_00ff,
            ..
        }) if source.method() == GpuMethodId(0x10e4)
            && source.argument() == 0x0000_0100
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10e4 / 4, &[0x00ff_00ff, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x10e8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10ec / 4, &[0x00ff_00ff, 0x0000_0100]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 0x00ff_00ff,
            ..
        }) if source.method() == GpuMethodId(0x10f0)
            && source.argument() == 0x0000_0100
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet(0x10f0 / 4, &[0x00ff_00ff, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x10f4)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_csaa_values_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x15b4, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x15b4 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let decoded = incrementing_packet(0x15b4 / 4, &[1, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x15b8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_ps_output_sample_mask_usage_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x0300, 3);

    for argument in [4, 7, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0300 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::MissingCapability { source, .. })
            if source.method() == GpuMethodId(0x0304)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_primitive_circular_buffer_throttle_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x02d0, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX);

    for argument in [0x0040_0000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x02d0 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x02d4)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn patch_size_is_typed_source_preserving_and_reserved_bits_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for argument in [0, 1, 3, 32, 0xff] {
        let decoded = packet(0x0dcc / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDPatchSize::new(argument as u8);
        let register = channel.three_d().vertex_input().primitive().patch_size();

        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_PATCH");
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::PatchSize { value, source }
            ))
        );
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
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let decoded = incrementing_packet(0x0dcc / 4, &[3, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0dd0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn point_rasterization_state_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

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
        let decoded = packet(0x1604 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        let decoded = packet(0x165c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_PROGRAM_REGION_A",
            ..
        }) if source.method() == GpuMethodId(0x1608)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn edge_flag_state_is_typed_source_preserving_and_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

    for (argument, expected) in [
        (0, MaxwellThreeDEdgeFlag::Disabled),
        (1, MaxwellThreeDEdgeFlag::Enabled),
    ] {
        let decoded = packet(0x15e4 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x15e8)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn vertex_array_restart_is_typed_source_preserving_and_isolated_from_indexed_restart() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x0de8 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let primitive = channel.three_d().vertex_input().primitive();
        let register = primitive.vertex_array_restart_enabled();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::VertexArrayPrimitiveRestartEnable {
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
fn invalid_vertex_array_restart_values_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x0de8, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0de8 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0dec)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn shade_mode_is_typed_source_preserving_and_part_of_pipeline_identity() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x12cc, 0);
    let depth_test_before = channel
        .three_d()
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable)
        .to_owned();
    let two_d_before = channel.two_d().clone();
    let unset_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_eq!(
        channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    let mut previous_dependencies = unset_dependencies;
    for (argument, expected) in [
        (0x1d00, MaxwellThreeDShadeMode::Flat),
        (0x1d01, MaxwellThreeDShadeMode::Smooth),
    ] {
        let decoded = packet(0x12d4 / 4, argument);
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
            .register(MaxwellThreeDFixedFunctionRegister::ShadeMode);

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SHADE_MODE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::Register {
                    register: MaxwellThreeDFixedFunctionRegister::ShadeMode,
                    value: MaxwellThreeDFixedFunctionValue::ShadeMode(expected),
                    source,
                }
            ))
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

        let dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(dependencies, previous_dependencies);
        previous_dependencies = dependencies;
    }
}

#[test]
fn invalid_shade_modes_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x12d4, 0x1d00);

    for argument in [0, 1, 0x1cff, 0x1d02, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x12d4 / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let decoded = incrementing_packet(0x12d4 / 4, &[0x1d01, 0, 0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x12e0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn shade_mode_is_consumed_only_by_draws_before_cache_or_backend_effects() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let address_space = resource_address_space();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();

    for (argument, expected) in [
        (0x1d00, MaxwellThreeDShadeMode::Flat),
        (0x1d01, MaxwellThreeDShadeMode::Smooth),
    ] {
        program_three_d(&mut channel, 0x12d4, argument);
        let source = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
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
            Err(MaxwellThreeDLoweringError::UnsupportedShadeModeSemantics(mode))
                if mode == expected
        ));
        assert_eq!(cache, cache_before);
    }

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
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
fn blend_controls_are_typed_source_preserving_and_family_isolated() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let common_before = *channel.three_d().fixed_function().blend_enable_common();
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
        let decoded = packet(0x1140 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_controls().per_format_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_PER_FORMAT_ENABLE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::BlendPerFormatEnable { value, source }
            ))
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
        let decoded = packet(0x0fdc / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_controls().float_pixel_kill_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_OPT_CONTROL"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::BlendFloatPixelKillEnable { value, source }
            ))
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
        let decoded = packet(0x19c0 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_controls().zero_times_anything_is_zero();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_FLOAT_OPTION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::BlendZeroTimesAnythingIsZero { value, source }
            ))
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
    assert_eq!(fixed.blend_enable_common(), &common_before);
    assert_eq!(fixed.blend_enable(), &per_target_enable_before);
    assert_eq!(fixed.per_target_blend(), &per_target_state_before);
    assert_eq!(channel.two_d(), &two_d_before);
}

#[test]
fn blend_control_pipeline_dependencies_are_semantic_and_ignore_optimization_permission() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x12e4, 0);
    program_three_d(&mut channel, 0x135c, 0);
    program_three_d(&mut channel, 0x1140, 0);
    program_three_d(&mut channel, 0x0fdc, 0);
    program_three_d(&mut channel, 0x19c0, 0);

    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[0]);
    program_three_d(&mut channel, 0x1140, 0x10);
    program_three_d(&mut channel, 0x0fdc, 1);
    program_three_d(&mut channel, 0x19c0, 1);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[0]),
        disabled_dependencies
    );

    program_three_d(&mut channel, 0x1140, 0);
    program_three_d(&mut channel, 0x0fdc, 0);
    program_three_d(&mut channel, 0x19c0, 0);
    program_three_d(&mut channel, 0x135c, 1);
    let enabled_dependencies = channel.three_d().pipeline_dependencies(&[0]);

    program_three_d(&mut channel, 0x0fdc, 1);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[0]),
        enabled_dependencies
    );
    program_three_d(&mut channel, 0x1140, 0x10);
    let per_format_dependencies = channel.three_d().pipeline_dependencies(&[0]);
    assert_ne!(per_format_dependencies, enabled_dependencies);
    program_three_d(&mut channel, 0x19c0, 1);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[0]),
        per_format_dependencies
    );
}

#[test]
fn invalid_blend_controls_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
    let decoded = incrementing_packet(0x1140 / 4, &[0, 0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1148)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn common_blend_enable_is_typed_source_preserving_and_family_isolated() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
            .blend_enable_common()
            .origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for (argument, expected) in [
        (0, MaxwellThreeDBlendEnableCommon::Disabled),
        (1, MaxwellThreeDBlendEnableCommon::Enabled),
    ] {
        let decoded = packet(0x135c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let fixed = channel.three_d().fixed_function();
        let register = fixed.blend_enable_common();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_BLEND_ENABLE_COMMON"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                MaxwellThreeDFixedFunctionWrite::BlendEnableCommon {
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
        .blend_enable_common()
        .to_owned();
    program_three_d(&mut channel, 0x12e4, 0);
    program_three_d(&mut channel, 0x1360, 0);
    program_three_d(&mut channel, 0x1e00, 1);
    assert_eq!(
        channel.three_d().fixed_function().blend_enable_common(),
        &common_before
    );
}

#[test]
fn invalid_common_blend_enable_values_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x135c, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x135c / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_BLEND_ENABLE_COMMON",
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "SET_BLEND",
            ..
        }) if source.method() == GpuMethodId(0x1360)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn draw_resolves_common_and_per_target_blend_state_before_effects() {
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let target = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        73,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_color_target(&mut channel, 0, target, 0xd5);
    program_three_d(&mut channel, 0x15d0, 0);
    program_three_d(&mut channel, 0x121c, 1);
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let source = channel
        .three_d()
        .render_targets()
        .color_target_selection()
        .source()
        .unwrap();

    let preflight = |channel: &MaxwellGpuChannel| {
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
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
        )
    };

    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: None,
            field: "SET_BLEND_STATE_PER_TARGET"
        })
    ));
    program_three_d(&mut channel, 0x12e4, 0);
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: None,
            field: "SET_BLEND_ENABLE_COMMON"
        })
    ));
    program_three_d(&mut channel, 0x135c, 0);
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x135c, 1);
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: None,
            field: "SET_BLEND_SEPARATE_FOR_ALPHA"
        })
    ));
    for (method, argument) in [(0x133c, 1), (0x1340, 1), (0x1344, 1), (0x1348, 1)] {
        program_three_d(&mut channel, method, argument);
    }
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: None,
            field: "SET_BLEND_OP_ALPHA"
        })
    ));
    for (method, argument) in [(0x134c, 1), (0x1350, 1), (0x1358, 1)] {
        program_three_d(&mut channel, method, argument);
    }
    let cache_before = cache.clone();
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::UnsupportedBlendSemantics { target: None })
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x12e4, 1);
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: Some(0),
            field: "SET_BLEND(i)"
        })
    ));
    program_three_d(&mut channel, 0x1360, 0);
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
    program_three_d(&mut channel, 0x1360, 1);
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: Some(0),
            field: "SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA"
        })
    ));
    for (method, argument) in [(0x1e00, 1), (0x1e04, 1), (0x1e08, 1), (0x1e0c, 1)] {
        program_three_d(&mut channel, method, argument);
    }
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::IncompleteBlendState {
            target: Some(0),
            field: "SET_BLEND_PER_TARGET_OP_ALPHA"
        })
    ));
    for (method, argument) in [(0x1e10, 1), (0x1e14, 1), (0x1e18, 1)] {
        program_three_d(&mut channel, method, argument);
    }
    assert!(matches!(
        preflight(&channel),
        Err(MaxwellThreeDLoweringError::UnsupportedBlendSemantics { target: Some(0) })
    ));
    assert_eq!(cache, cache_before);

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
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
}

#[test]
fn vertex_array_restart_is_consumed_only_by_non_indexed_draws() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());

    program_three_d(&mut channel, 0x0de8, 0);
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .vertex_array_restart_enabled()
        .source()
        .unwrap();
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
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

    program_three_d(&mut channel, 0x0de8, 1);
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .vertex_array_restart_enabled()
        .source()
        .unwrap();
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
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedVertexArrayPrimitiveRestartSemantics)
    ));
    assert_eq!(cache, cache_before);

    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let clear_resources =
        resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space()).unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &clear_resources,
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
fn aliased_line_width_selector_is_typed_source_preserving_isolated_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x020c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().line().aliased_line_width_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALIASED_LINE_WIDTH_ENABLE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Line(
                MaxwellThreeDLineStateWrite::AliasedLineWidthEnable {
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
        assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
        assert_eq!(channel.three_d().coverage(), &coverage_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn invalid_aliased_line_width_values_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x020c, 0);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x020c / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let decoded = incrementing_packet(0x020c / 4, &[1, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0210)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn aliased_line_width_is_consumed_only_by_line_rasterization() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1618, 1);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .begin()
        .source()
        .unwrap();

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 2,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_ALIASED_LINE_WIDTH_ENABLE"
        ))
    ));

    program_three_d(&mut channel, 0x020c, 0);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 2,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_LINE_WIDTH_FLOAT"
        ))
    ));
    program_three_d(&mut channel, 0x13b0, 0x3f80_0000);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 2,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x020c, 1);
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 2,
            },
            None,
            FrontendSubmissionId::new(13),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedAliasedLineWidthSemantics)
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x1618, 4);
    program_three_d(&mut channel, 0x0dac, 0x1b02);
    program_three_d(&mut channel, 0x0db0, 0x1b02);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(14),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x1618, 0);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 1,
            },
            None,
            FrontendSubmissionId::new(15),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x1618, 4);
    program_three_d(&mut channel, 0x0dac, 0x1b01);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(16),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedAliasedLineWidthSemantics)
    ));
    assert_eq!(cache, cache_before);

    // Polygon mode does not turn point primitives into line primitives.
    program_three_d(&mut channel, 0x1618, 0);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 1,
            },
            None,
            FrontendSubmissionId::new(17),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
}

#[test]
fn aliased_line_width_selector_does_not_change_clear_semantics() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x020c, 1);
    let clear = packet(0x19d0 / 4, 0x3c);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources =
        resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space()).unwrap();

    assert!(matches!(
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(18),
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
fn csaa_only_blocks_draws_when_explicitly_enabled_and_never_blocks_clear() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();

    program_three_d(&mut channel, 0x15b4, 0);
    let source = channel.three_d().coverage().csaa_enable().source().unwrap();
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

    program_three_d(&mut channel, 0x15b4, 1);
    let source = channel.three_d().coverage().csaa_enable().source().unwrap();
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
        Err(MaxwellThreeDLoweringError::UnsupportedCsaaSemantics)
    ));
    assert_eq!(cache, cache_before);

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
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn ps_output_sample_mask_usage_obeys_aa_and_never_blocks_clear() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();

    program_three_d(&mut channel, 0x1534, 0);
    program_three_d(&mut channel, 0x0300, 3);
    let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);
    let source = channel
        .three_d()
        .coverage()
        .ps_output_sample_mask_usage()
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

    program_three_d(&mut channel, 0x0300, 0);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );

    program_three_d(&mut channel, 0x0300, 3);
    program_three_d(&mut channel, 0x1534, 1);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        inactive_dependencies
    );
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
        Err(MaxwellThreeDLoweringError::UnsupportedPsOutputSampleMaskSemantics)
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x1534, 0);
    program_three_d(&mut channel, 0x0300, 1);
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
        Err(MaxwellThreeDLoweringError::UnsupportedPsOutputSampleMaskSemantics)
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
            FrontendSubmissionId::new(13),
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
fn primitive_circular_buffer_throttle_does_not_change_draw_or_clear_semantics() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x02d0, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let cache_before = cache.clone();
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .circular_buffer_throttle()
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
    assert_eq!(cache, cache_before);
}

#[test]
fn patch_size_is_consumed_only_by_patch_draws_and_never_by_clears() {
    let mut missing_channel = channel();
    bind_three_d(&mut missing_channel);
    program_three_d(&mut missing_channel, 0x121c, 0);
    program_three_d(&mut missing_channel, 0x1618, 14);
    let missing = missing_channel.three_d().clone();

    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x2080, 0x20);
    program_three_d(&mut channel, 0x20c0, 0x30);
    program_three_d(&mut channel, 0x1618, 4);
    let dependencies_without_patch = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x0dcc, 3);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        dependencies_without_patch
    );
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .patch_size()
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
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x1618, 14);
    let patch_three_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x0dcc, 4);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        patch_three_dependencies
    );
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .begin()
        .source()
        .unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 4,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedPatchSemantics(size))
            if size.control_points() == 4
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x0dcc, 0);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 4,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::InvalidPatchSize(size))
            if size.control_points() == 0
    ));

    // A fresh channel proves that patch topology without SET_PATCH is
    // incomplete rather than silently assuming a control-point count.
    let missing_source = missing.vertex_input().primitive().begin().source().unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            &missing,
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: missing_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(13),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteDraw("SET_PATCH"))
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
            FrontendSubmissionId::new(14),
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
fn point_rasterization_state_is_consumed_only_by_point_draws_and_never_by_clears() {
    let mut passthrough = channel();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1618, 4);
    let triangle_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x1604, 0x000d);
    program_three_d(&mut channel, 0x165c, 1);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        triangle_dependencies
    );

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .begin()
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
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x1618, 0);
    let generated_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x1604, 0);
    let passthrough_dependencies = channel.three_d().pipeline_dependencies(&[]);
    assert_ne!(generated_dependencies, passthrough_dependencies);
    program_three_d(&mut channel, 0x1604, 0x000d);
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .begin()
        .source()
        .unwrap();
    let cache_before = cache.clone();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 1,
            },
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedPointSpriteCoordinatesSemantics(
            select
        )) if select.generated_texture_mask() == 1
            && select.r_mode() == MaxwellThreeDPointSpriteRMode::FromR
            && select.origin() == MaxwellThreeDPointSpriteOrigin::Top
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x1604, 0);
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 1,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedPointCenterSemantics(
            MaxwellThreeDPointCenterMode::Direct3D
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
            FrontendSubmissionId::new(13),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);

    bind_three_d(&mut passthrough);
    program_three_d(&mut passthrough, 0x121c, 0);
    program_three_d(&mut passthrough, 0x1618, 0);
    program_three_d(&mut passthrough, 0x1604, 0);
    let resources =
        resolve_maxwell_three_d_resources(passthrough.three_d(), &resource_address_space())
            .unwrap();
    let source = passthrough
        .three_d()
        .vertex_input()
        .primitive()
        .begin()
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            passthrough.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 1,
            },
            None,
            FrontendSubmissionId::new(14),
            Vec::new(),
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
}

#[test]
fn edge_flag_is_consumed_only_by_non_fill_polygon_draws_and_never_by_clears() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x1618, 4);
    program_three_d(&mut channel, 0x0dac, 0x1b02);
    program_three_d(&mut channel, 0x0db0, 0x1b02);
    let fill_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x15e4, 0);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        fill_dependencies
    );

    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let source = channel
        .three_d()
        .vertex_input()
        .primitive()
        .begin()
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
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    program_three_d(&mut channel, 0x0dac, 0x1b01);
    let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
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
            &capabilities,
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::UnsupportedEdgeFlagSemantics(
            MaxwellThreeDEdgeFlag::Disabled
        ))
    ));
    assert_eq!(cache, cache_before);

    program_three_d(&mut channel, 0x15e4, 1);
    assert_ne!(
        channel.three_d().pipeline_dependencies(&[]),
        disabled_dependencies
    );

    program_three_d(&mut channel, 0x1618, 0);
    program_three_d(&mut channel, 0x15e4, 0);
    let point_disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
    program_three_d(&mut channel, 0x15e4, 1);
    assert_eq!(
        channel.three_d().pipeline_dependencies(&[]),
        point_disabled_dependencies
    );

    program_three_d(&mut channel, 0x15e4, 0);
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
fn color_reduction_only_blocks_draws_when_explicitly_enabled_and_never_blocks_clear() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x10cc, 0x0000_00ff);
    program_three_d(&mut channel, 0x10e0, 0x0000_00ff);
    program_three_d(&mut channel, 0x10e4, 0x0000_00ff);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let cache = MaxwellThreeDLoweringCache::default();
    let threshold_source = channel
        .three_d()
        .color_reduction()
        .thresholds_unorm8()
        .source()
        .unwrap();
    assert!(matches!(
        preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: threshold_source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(9),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));

    for (argument, expected) in [
        (0, MaxwellThreeDColorReductionThresholdsEnable::Disabled),
        (1, MaxwellThreeDColorReductionThresholdsEnable::Enabled),
    ] {
        let decoded = packet(0x0d9c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().color_reduction().enable();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_REDUCE_COLOR_THRESHOLDS_ENABLE"
        );
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));
        assert_eq!(register.source(), Some(source));
        assert!(dispatch.operations().is_empty());
    }

    program_three_d(&mut channel, 0x0d9c, 0);
    let source = channel
        .three_d()
        .color_reduction()
        .enable()
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

    program_three_d(&mut channel, 0x0d9c, 1);
    let source = channel
        .three_d()
        .color_reduction()
        .enable()
        .source()
        .unwrap();
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
        Err(MaxwellThreeDLoweringError::UnsupportedColorReductionSemantics)
    ));
    assert_eq!(cache, cache_before);

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
            &cache,
        ),
        Err(MaxwellThreeDLoweringError::IncompleteClear(
            "horizontal rectangle"
        ))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn coverage_line_and_vertex_array_restart_selectors_separate_pipeline_identity() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        71,
        0,
    )
    .offset()
    .get();
    let target = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        72,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    let shaders = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let mut cache = MaxwellThreeDLoweringCache::default();
    let draw = packet(0x0d78 / 4, 3);

    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(20),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    plan.commit_cache(&mut cache).unwrap();
    let pipeline_count = cache.pipeline_count();

    program_three_d(&mut channel, 0x15b4, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(21),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    assert!(!plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count + 1);

    let pipeline_count = cache.pipeline_count();
    program_three_d(&mut channel, 0x020c, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(22),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    assert!(!plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count + 1);

    let pipeline_count = cache.pipeline_count();
    program_three_d(&mut channel, 0x0de8, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(23),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    assert!(!plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count + 1);
}

#[test]
fn disabled_blend_pipeline_identity_tracks_only_effective_active_selectors() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        74,
        0,
    )
    .offset()
    .get();
    let target = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        75,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target, 0xd5);
    program_three_d(&mut channel, 0x121c, 1);
    let shaders = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let mut cache = MaxwellThreeDLoweringCache::default();
    let draw = packet(0x0d78 / 4, 3);

    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(30),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap()
    .commit_cache(&mut cache)
    .unwrap();
    let pipeline_count = cache.pipeline_count();

    // Per-target state and common equations are inactive while common
    // blending is selected and explicitly disabled.
    for (method, argument) in [
        (0x1364, 1),
        (0x1e20, 1),
        (0x133c, 1),
        (0x1340, 1),
        (0x1344, 1),
        (0x1348, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(31),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(!plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count);

    // Selecting per-target state and explicitly disabling the active
    // target changes the effective pipeline state.
    program_three_d(&mut channel, 0x12e4, 1);
    program_three_d(&mut channel, 0x1360, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(32),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count + 1);
    let pipeline_count = cache.pipeline_count();

    // Common and unselected target selectors are now inactive.
    program_three_d(&mut channel, 0x135c, 1);
    program_three_d(&mut channel, 0x1364, 0);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &draw.packets()[0],
    )
    .unwrap();
    let triggered = &dispatch.operations()[0];
    let resources = resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
    let plan = preflight_maxwell_three_d_operation(
        triggered.state(),
        &resources,
        triggered.trigger(),
        Some(&shaders),
        FrontendSubmissionId::new(33),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    assert!(!plan.resource_creations().iter().any(|creation| matches!(
        creation,
        nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
    )));
    plan.commit_cache(&mut cache).unwrap();
    assert_eq!(cache.pipeline_count(), pipeline_count);
}
