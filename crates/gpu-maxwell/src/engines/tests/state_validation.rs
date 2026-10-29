use super::*;

#[test]
fn viewport_clips_preserve_origins_and_extents_independently() {
    let mut channel = three_d_channel();
    for (viewport, raw) in [(0, 0x00f0_01e0), (15, u32::MAX), (7, 0x0000_ffff)] {
        for vertical in [false, true] {
            program_three_d(
                &mut channel,
                0x0c00 + viewport * 16 + u32::from(vertical) * 4,
                raw,
            );
            let clip = &channel
                .three_d()
                .fixed_function()
                .viewport(viewport as usize);
            let register = if vertical {
                clip.clip_vertical()
            } else {
                clip.clip_horizontal()
            };
            assert_eq!(register.raw(), Some(raw));
            assert_eq!(register.value().unwrap().origin(), raw as u16);
            assert_eq!(register.value().unwrap().extent(), (raw >> 16) as u16);
        }
    }
}

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
fn mutable_method_control_is_typed_and_source_preserving() {
    let mut channel = three_d_channel();

    for (argument, expected) in [
        (0, MaxwellThreeDMutableMethodControl::Lightweight),
        (1, MaxwellThreeDMutableMethodControl::Heavyweight),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1134 / 4, argument).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_MUTABLE_METHOD_CONTROL"
        );

        let register = channel.three_d_mut().mme().mutable_method_control();
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value(), Some(&expected));
    }
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
        Ok(MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(release))
            if release.address().get() == 0x12_3456_7890
                && release.payload() == 0xcafe_babe
                && release.prior_work_pending()
    ));

    let dispatch = dispatch_method(&mut channel, 0x1b0c / 4, 0xf002).unwrap();
    assert!(matches!(lower_maxwell_three_d_synchronization(
        dispatch.synchronization_operations()[0], None, true,
    ), Ok(MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(report))
        if report.payload() == 0xcafe_babe
            && report.structure_size() == MaxwellThreeDReportSemaphoreStructureSize::FourWords
            && report.prior_work_pending()));
    // Short RELEASE is the completion boundary. Timestamped releases and
    // REPORT_ONLY do not commit preceding asynchronous reports.
    for (raw, fence) in [
        (0x1000_f000, true),
        (0x0000_f000, false),
        (0x1000_f002, false),
        (0x0000_f002, false),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x1b0c / 4, raw).unwrap();
        let Ok(MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(report)) =
            lower_maxwell_three_d_synchronization(
                dispatch.synchronization_operations()[0],
                None,
                true,
            )
        else {
            panic!("expected payload report");
        };
        assert_eq!(report.is_fence(), fence, "control={raw:#x}");
    }
    for raw in [0xf001, 0xf003, 0x0080_f002, 0x0010_f002, 0x0002_f002] {
        let dispatch = dispatch_method(&mut channel, 0x1b0c / 4, raw).unwrap();
        assert!(matches!(
            lower_maxwell_three_d_synchronization(
                dispatch.synchronization_operations()[0],
                None,
                false,
            ),
            Err(MaxwellThreeDSynchronizationError::UnsupportedReportSemaphoreControl { .. })
        ));
    }

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
        let register = channel.two_d().processing_clusters();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NUM_PROCESSING_CLUSTERS"
        );

        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));

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
        MaxwellTwoDRegisterOrigin::ContextDefault
    );
    assert_eq!(
        channel.two_d().render_enable().mode().value(),
        Some(&MaxwellTwoDRenderEnableMode::Enabled)
    );
    assert_eq!(channel.two_d().render_enable().mode().raw(), Some(1));
    assert_eq!(channel.two_d().render_enable().mode().source(), None);

    for (argument, expected) in [
        (0, MaxwellTwoDRenderEnableMode::Disabled),
        (1, MaxwellTwoDRenderEnableMode::Enabled),
        (2, MaxwellTwoDRenderEnableMode::Conditional),
        (3, MaxwellTwoDRenderEnableMode::RenderIfEqual),
        (4, MaxwellTwoDRenderEnableMode::RenderIfNotEqual),
    ] {
        let decoded = packet_on_subchannel(3, 0x026c / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
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
        let value = MaxwellTwoDBeta1::new(argument);
        let register = channel.two_d().beta().beta1();

        assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_BETA1");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));

        assert_eq!(channel.three_d(), &three_d_before);
    }

    for argument in [0, 0x1122_3344, u32::MAX] {
        let decoded = packet_on_subchannel(3, 0x02a8 / 4, argument);
        let dispatch = dispatch_first(&mut channel, &decoded).unwrap();
        let value = MaxwellTwoDBeta4::from_raw(argument);
        let register = channel.two_d().beta().beta4();

        assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_BETA4");

        assert!(dispatch.operations().is_empty());
        assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(value));

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
        MaxwellTwoDRegisterOrigin::ContextDefault
    );
    assert_eq!(
        channel.two_d().color_key_enable().value(),
        Some(&MaxwellTwoDColorKeyEnable::Disabled)
    );
    assert_eq!(channel.two_d().color_key_enable().raw(), Some(0));
    assert_eq!(channel.two_d().color_key_enable().source(), None);
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
        assert!(
            matches!(dispatch.ordered_operations(), [MaxwellEngineOperation::ThreeDRenderCondition(observed)] if *observed==source)
        );

        assert_eq!(register.raw(), Some(argument));
        assert_eq!(register.value().copied(), Some(expected));

        assert_eq!(channel.two_d(), &two_d_before);
    }
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
fn render_enable_address_fragments_preserve_the_complete_condition_address() {
    let mut channel = three_d_channel();
    let mode = channel.three_d().render_enable().mode();
    program_three_d(&mut channel, 0x1550, 5);
    program_three_d(&mut channel, 0x1554, 0x1234_5678);
    assert_eq!(
        channel.three_d().render_enable().address(),
        Some(MaxwellThreeDUnresolvedAddress::new(5, 0x1234_5678))
    );
    assert_eq!(channel.three_d().render_enable().mode(), mode);
    program_three_d(&mut channel, 0x1550, 0x100);
    assert!(channel.three_d().render_enable().address().is_none());
}
