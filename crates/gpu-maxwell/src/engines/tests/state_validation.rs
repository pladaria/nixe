use super::*;

#[test]
fn three_d_no_operation_is_named_and_implemented() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let decoded = packet(0x100 / 4, 0xfeed_beef);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();

    assert_eq!(dispatch.methods().len(), 1);
    assert_eq!(dispatch.methods()[0].metadata().class_name(), "MAXWELL_B");
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "NO_OPERATION"
    );
    assert_eq!(
        dispatch.methods()[0].effect(),
        MaxwellEngineMethodEffect::NoOperation
    );
}

#[test]
fn alpha_fraction_is_typed_source_preserving_raster_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x074c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDAlphaFraction::new(argument as u8);
        let register = channel.three_d().raster().alpha_fraction();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALPHA_FRACTION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::AlphaFraction {
                value,
                source,
            })
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
fn invalid_alpha_fraction_values_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x074c, 0x3f);

    for argument in [0x0000_0100, 0x0000_ff00, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x074c / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0750)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn raster_bounding_box_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x02ec / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellThreeDRasterBoundingBox::new(mode, pad);
        let register = channel.three_d().raster().bounding_box();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RASTER_BOUNDING_BOX"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RasterBoundingBox {
                value,
                source
            })
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
    let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 0x60);
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
                emitted_methods: 1,
            },
        }
    );
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
fn invalid_raster_bounding_box_values_and_packet_suffix_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x02ec, 0x60);

    for argument in [2, 4, 8, 0x1000, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x02ec / 4, argument);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x02f0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn sph_version_check_is_typed_profile_validated_and_state_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let supported = channel.profile().shader().sph_versions();

    for (argument, current, oldest_supported) in [
        (0x0003_0003, 3, 3),
        (0x0003_0004, 4, 3),
        (0x0002_0003, 3, 2),
        (0x0002_0004, 4, 2),
    ] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x16a8 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let requested = MaxwellShaderProgramHeaderVersionRange::new(
            MaxwellShaderProgramHeaderVersion::new(current),
            MaxwellShaderProgramHeaderVersion::new(oldest_supported),
        );

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CHECK_SPH_VERSION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ShaderProgramHeaderCompatibilityCheck {
                requested,
                supported,
            }
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
    let mut channel = channel();
    bind_three_d(&mut channel);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet(0x16a8 / 4, 0x0003_0002);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
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
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x16ac)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn aam_version_check_is_typed_profile_validated_and_state_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let supported = channel.profile().aam_versions();

    for (argument, current, oldest_supported) in [
        (0x0002_0002, 2, 2),
        (0x0002_0003, 3, 2),
        (0x0001_0002, 2, 1),
        (0x0001_0003, 3, 1),
    ] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1794 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let requested = MaxwellAamVersionRange::new(
            MaxwellAamVersion::new(current),
            MaxwellAamVersion::new(oldest_supported),
        );

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CHECK_AAM_VERSION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::AamCompatibilityCheck {
                requested,
                supported,
            }
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
    let mut channel = channel();
    bind_three_d(&mut channel);

    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet(0x1794 / 4, 0x0003_0002);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
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
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1798)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn rop_l2_cache_controls_are_typed_source_preserving_independent_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
            channel.three_d().rop_l2_cache().policy(request).origin(),
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
                requests.map(|other| *channel.three_d().rop_l2_cache().policy(other));
            let decoded = packet(method / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().rop_l2_cache().policy(request);

            assert_eq!(dispatch.methods()[0].metadata().method_name(), method_name);
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RopL2Cache(
                    MaxwellThreeDRopL2CacheStateWrite::Policy {
                        request,
                        value,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.encoded(), argument);

            for (index, other) in requests.into_iter().enumerate() {
                if other != request {
                    assert_eq!(
                        channel.three_d().rop_l2_cache().policy(other),
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
fn invalid_rop_l2_cache_policies_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
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
    let decoded = incrementing_packet(0x0218 / 4, &[0x20, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x021c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_notify_address_upper_is_bounded_state_without_notification_effects() {
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDNotifyAddressUpper::new(argument).unwrap();
        let register = channel.two_d().notify().address_upper();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NOTIFY_A"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::Notify(
                MaxwellTwoDNotifyStateWrite::AddressUpper { value, source }
            ))
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
    let mut channel = channel();
    bind_two_d(&mut channel);

    for argument in [0x0200_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0104 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let mut channel = channel();
    bind_two_d(&mut channel);
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().notify().address_lower().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for argument in [0, 0x0820_2010, u32::MAX] {
        let decoded = packet_on_subchannel(3, 0x0108 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDNotifyAddressLower::new(argument);
        let register = channel.two_d().notify().address_lower();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NOTIFY_B"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::Notify(
                MaxwellTwoDNotifyStateWrite::AddressLower { value, source }
            ))
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
    let mut channel = channel();
    bind_two_d(&mut channel);
    let decoded = incrementing_packet_on_subchannel(3, 0x0104 / 4, &[1, 0x0820_2010]);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();

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
fn unsupported_notify_trigger_discards_both_address_fragments() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x0104 / 4, &[1, 0x0820_2010, 0]);

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x010c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_processing_cluster_values_are_typed_and_retain_their_source() {
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().processing_clusters();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NUM_PROCESSING_CLUSTERS"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::ProcessingClusters {
                value: expected,
                source,
            })
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
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().render_enable().mode();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_C"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::RenderEnable(
                MaxwellTwoDRenderEnableStateWrite::Mode {
                    value: expected,
                    source,
                }
            ))
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
    let mut channel = channel();
    bind_two_d(&mut channel);

    for argument in [5, 6, 7, 8, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x026c / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let mut channel = channel();
    bind_two_d(&mut channel);

    for method in [0x0264, 0x0268] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, method / 4, 0);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
fn two_d_operation_values_are_typed_state_without_execution() {
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().operation();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_OPERATION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::Operation {
                value: expected,
                source,
            })
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
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().clip_enable();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CLIP_ENABLE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::ClipEnable {
                value: expected,
                source,
            })
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
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x0290 / 4, 2);

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
}

#[test]
fn two_d_color_key_enable_values_are_typed_state_without_execution() {
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().color_key_enable();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_COLOR_KEY_ENABLE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::ColorKeyEnable {
                value: expected,
                source,
            })
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
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x029c / 4, 2);

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
}

#[test]
fn unsupported_method_after_color_key_enable_discards_the_packet_prefix() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x029c / 4, &[1, 0]);

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x02a0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn two_d_corral_size_is_bounded_source_preserving_state_without_execution() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let state_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    assert_eq!(
        channel.two_d().pixels_from_memory().corral_size().origin(),
        MaxwellTwoDRegisterOrigin::Unset
    );

    for argument in [0, 0x3f, u32::from(MAXWELL_TWO_D_CORRAL_SIZE_MAX)] {
        let decoded = packet_on_subchannel(3, 0x0884 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let value = MaxwellTwoDPixelsFromMemoryCorralSize::new(argument as u16).unwrap();
        let register = channel.two_d().pixels_from_memory().corral_size();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PIXELS_FROM_MEMORY_CORRAL_SIZE"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::PixelsFromMemory(
                MaxwellTwoDPixelsFromMemoryStateWrite::CorralSize { value, source }
            ))
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
    let mut channel = channel();
    bind_two_d(&mut channel);

    for argument in [0x0400, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0884 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let mut channel = channel();
    bind_two_d(&mut channel);
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
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.two_d().pixels_from_memory().safe_overlap();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::PixelsFromMemory(
                MaxwellTwoDPixelsFromMemoryStateWrite::SafeOverlap {
                    value: expected,
                    source,
                }
            ))
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
    let mut channel = channel();
    bind_two_d(&mut channel);

    for argument in [2, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0888 / 4, argument);

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
}

#[test]
fn unsupported_pixels_from_memory_method_remains_a_typed_fatal_error() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x0880 / 4, 0);

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
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
    let mut channel = channel();
    bind_two_d(&mut channel);

    for argument in [7, 8] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x02ac / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
fn unsupported_method_after_two_d_operation_discards_the_packet_prefix() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x02ac / 4, &[3, 0]);

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x02b0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn invalid_two_d_value_rejects_without_mutating_any_channel_state() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = packet_on_subchannel(3, 0x0260 / 4, 2);

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
}

#[test]
fn unsupported_two_d_suffix_discards_the_valid_packet_prefix() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let decoded = incrementing_packet_on_subchannel(3, 0x0260 / 4, &[1, 0]);

    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: "FERMI_TWOD_A",
        }) if source.method() == GpuMethodId(0x0264)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn commit_rejects_intervening_two_d_state_without_partial_publish() {
    let mut channel = channel();
    bind_two_d(&mut channel);
    let first = packet_on_subchannel(3, 0x0260 / 4, 0);
    let prepared = preflight_maxwell_engine_packet(
        &channel,
        FrontendSubmissionId::new(3),
        &first.packets()[0],
    )
    .unwrap();
    let intervening = packet_on_subchannel(3, 0x0260 / 4, 1);
    dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &intervening.packets()[0],
    )
    .unwrap();
    let committed_two_d = channel.two_d().clone();
    let committed_three_d = channel.three_d().clone();

    assert!(matches!(
        commit_maxwell_engine_packet(&mut channel, &prepared),
        Err(MaxwellEngineDispatchError::EngineStateChanged { .. })
    ));
    assert_eq!(channel.two_d(), &committed_two_d);
    assert_eq!(channel.three_d(), &committed_three_d);
}

#[test]
fn three_d_render_enable_modes_are_typed_and_engine_owned() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x1558 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().render_enable().mode();

        assert_eq!(dispatch.methods()[0].metadata().class(), threed::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_C"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderEnable(
                MaxwellThreeDRenderEnableStateWrite::Mode {
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
        assert_eq!(channel.two_d(), &two_d_before);
    }
}

#[test]
fn render_enable_control_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let mode_before = *channel.three_d().render_enable().mode();
    let two_d_before = channel.two_d().clone();
    let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

    for (argument, value) in [
        (0, MaxwellThreeDConditionalLoadConstantBuffer::Disabled),
        (1, MaxwellThreeDConditionalLoadConstantBuffer::Enabled),
    ] {
        let decoded = packet(0x030c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel
            .three_d()
            .render_enable()
            .conditional_load_constant_buffer();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_CONTROL"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderEnable(
                MaxwellThreeDRenderEnableStateWrite::ConditionalLoadConstantBuffer {
                    value,
                    source,
                }
            ))
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
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x030c, 0);

    for argument in [2, 3, 0x10, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x030c / 4, argument);

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
}

#[test]
fn enabled_conditional_load_stops_before_neutral_lowering() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x1558, 1);
    let decoded = packet(0x030c / 4, 0);
    let disabled = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    assert_eq!(
        disabled.methods()[0].metadata().method_name(),
        "SET_RENDER_ENABLE_CONTROL"
    );
    let clear = packet(0x19d0 / 4, 0x3c);
    let clear_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
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

    let decoded = packet(0x030c / 4, 1);
    let dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &decoded.packets()[0],
    )
    .unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_RENDER_ENABLE_CONTROL"
    );
    let clear_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &clear.packets()[0],
    )
    .unwrap();
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
    let mut channel = channel();
    bind_three_d(&mut channel);

    for argument in [5, 6, 7, 8, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1558 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let mut channel = channel();
    bind_three_d(&mut channel);

    for method in [0x1550, 0x1554] {
        let frontend_before = channel.frontend();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(method / 4, 0);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        let mut channel = channel();
        bind_three_d(&mut channel);
        let decoded = packet(0x1558 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
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
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();
    let visible_call_before = channel
        .three_d()
        .shader_execution()
        .visible_call_limit()
        .to_owned();
    let mut previous_dependencies = channel.three_d().pipeline_dependencies(&[]);
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
        let decoded = packet(0x0308 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().shader_execution().l1_configuration();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_L1_CONFIGURATION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderExecution(
                MaxwellThreeDShaderExecutionStateWrite::L1Configuration {
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
        assert_eq!(expected.bytes(), bytes);
        assert_eq!(
            channel.three_d().shader_execution().visible_call_limit(),
            &visible_call_before
        );
        assert_eq!(channel.two_d(), &two_d_before);

        let dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(dependencies, previous_dependencies);
        previous_dependencies = dependencies;
    }

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
fn invalid_l1_configurations_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x0308, 1);

    for argument in [0, 2, 4, 5, 6, 7, 8, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0308 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            defined_mask: 1,
            ..
        }) if source.method() == GpuMethodId(0x030c) && source.argument() == 2
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn shader_local_memory_block_is_typed_source_preserving_and_shader_scoped() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let two_d_before = channel.two_d().clone();
    let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);

    let region = incrementing_packet(0x0790 / 4, &[0x04, 0x0008_0000, 0, 0x0408_0000, 0]);
    let region_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &region.packets()[0],
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

    let window = packet(0x077c / 4, 0xff00_0000);
    let window_dispatch = dispatch_maxwell_engine_packet(
        &mut channel,
        FrontendSubmissionId::new(3),
        &window.packets()[0],
    )
    .unwrap();
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
fn shader_local_memory_fields_ranges_and_packets_are_atomic() {
    let mut channel = channel();
    bind_three_d(&mut channel);

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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalid_suffix.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
            if source.method() == GpuMethodId(0x07a0)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);

    program_three_d(&mut channel, 0x0790, 0xff);
    program_three_d(&mut channel, 0x0794, u32::MAX);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let three_d_before = channel.three_d().clone();
    let overflowing_size = incrementing_packet(0x0798 / 4, &[0, 2]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &overflowing_size.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::ContradictoryState {
            source: Some(source),
            reason: "shader-local-memory region exceeds the 40-bit GPU address space",
        }) if source.method() == GpuMethodId(0x079c)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn active_shader_local_memory_blocks_only_draws_before_effects() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
fn visible_call_limit_is_typed_source_preserving_execution_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x0d64 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().shader_execution().visible_call_limit();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_API_VISIBLE_CALL_LIMIT"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderExecution(
                MaxwellThreeDShaderExecutionStateWrite::VisibleCallLimit {
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
fn invalid_visible_call_limits_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x0d64, 8);

    for argument in [9, 10, 11, 12, 13, 14, 0x10, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x0d64 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x0d68)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn active_visible_call_limits_block_only_draws_before_cache_effects() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let address_space = resource_address_space();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();

    for (argument, expected) in [
        (0, MaxwellThreeDVisibleCallLimit::Calls0),
        (8, MaxwellThreeDVisibleCallLimit::Calls128),
    ] {
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
            Err(MaxwellThreeDLoweringError::UnsupportedVisibleCallLimitSemantics(limit))
                if limit == expected
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
fn visible_call_no_check_does_not_invent_a_draw_limit() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x0d64, 15);
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

    assert!(!matches!(
        result,
        Err(MaxwellThreeDLoweringError::UnsupportedVisibleCallLimitSemantics(_))
    ));
    assert_eq!(cache, cache_before);
}

#[test]
fn active_zcull_region_is_typed_source_preserving_and_pipeline_neutral() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let frontend_before = channel.frontend();
    let two_d_before = channel.two_d().clone();
    let stats_before = *channel.three_d().zcull().stats_enable();
    assert_eq!(
        channel.three_d().zcull().active_region().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );

    for argument in [0, 1, 0x3f] {
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        let decoded = packet(0x1590 / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let method = dispatch.methods()[0];
        let source = method.method().source();
        let register = channel.three_d().zcull().active_region();
        let value = register.value().copied().unwrap();

        assert_eq!(method.metadata().method_name(), "SET_ACTIVE_ZCULL_REGION");
        assert_eq!(
            method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ZCull(
                MaxwellThreeDZCullStateWrite::ActiveRegion { value, source }
            ))
        );
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
fn active_zcull_region_reserved_bits_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x1590, 0x3f);

    for argument in [0x40, 0x80, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1590 / 4, argument);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
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
    let decoded = incrementing_packet(0x1590 / 4, &[0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1594)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn active_zcull_region_without_region_storage_does_not_change_draw_or_clear_semantics() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
fn zcull_stats_enable_is_typed_source_preserving_isolated_three_d_state() {
    let mut channel = channel();
    bind_three_d(&mut channel);
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
        let decoded = packet(0x151c / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let source = dispatch.methods()[0].method().source();
        let register = channel.three_d().zcull().stats_enable();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ZCULL_STATS"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ZCull(
                MaxwellThreeDZCullStateWrite::StatsEnable {
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
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d().coverage(), &coverage_before);
    }
}

#[test]
fn invalid_zcull_stats_values_and_packet_suffix_are_rejected_atomically() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    program_three_d(&mut channel, 0x151c, 1);

    for argument in [2, 3, 0x8000_0000, u32::MAX] {
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x151c / 4, argument);

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
    let decoded = incrementing_packet(0x151c / 4, &[0, 0]);
    assert!(matches!(
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0]
        ),
        Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
            if source.method() == GpuMethodId(0x1520)
    ));
    assert_eq!(channel.frontend(), frontend_before);
    assert_eq!(channel.two_d(), &two_d_before);
    assert_eq!(channel.three_d(), &three_d_before);
}

#[test]
fn enabled_zcull_stats_block_only_draws_before_cache_effects() {
    let mut channel = channel();
    bind_three_d(&mut channel);
    let address_space = resource_address_space();
    let capabilities = lowering_capabilities(BackendFeatures::empty());
    let cache = MaxwellThreeDLoweringCache::default();

    program_three_d(&mut channel, 0x151c, 1);
    let source = channel.three_d().zcull().stats_enable().source().unwrap();
    let resources = resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
    let cache_before = cache.clone();
    let error = match preflight_maxwell_three_d_operation(
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
    ) {
        Ok(_) => panic!("enabled Z-cull statistics unexpectedly allowed a draw"),
        Err(error) => error,
    };
    assert!(matches!(
        &error,
        MaxwellThreeDLoweringError::UnsupportedZCullStatsSemantics
    ));
    assert_eq!(
        error.to_string(),
        "MAXWELL_B enabled Z-cull statistics have no implemented counter accumulation, visibility, or reporting semantics"
    );
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
    assert!(!matches!(
        result,
        Err(MaxwellThreeDLoweringError::UnsupportedZCullStatsSemantics)
    ));
    assert_eq!(cache, cache_before);
}
