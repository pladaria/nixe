use super::*;

#[test]
fn alpha_to_coverage_dither_is_typed_validated_and_source_preserving() {
    let mut channel = three_d_channel();
    assert!(
        channel
            .three_d()
            .coverage()
            .alpha_to_coverage_dither()
            .value()
            .is_none()
    );
    for (raw, expected) in [
        (0, MaxwellThreeDAlphaToCoverageDither::Pixels1x1),
        (1, MaxwellThreeDAlphaToCoverageDither::Pixels2x2),
        (
            2,
            MaxwellThreeDAlphaToCoverageDither::Pixels1x1VirtualSamples,
        ),
    ] {
        let dispatch = dispatch_method(&mut channel, 0x12e0 / 4, raw).unwrap();
        let source = dispatch.methods()[0].method().source();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ALPHA_TO_COVERAGE_DITHER_CONTROL"
        );
        assert!(dispatch.operations().is_empty());
        let register = channel.three_d().coverage().alpha_to_coverage_dither();
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.raw(), Some(raw));
        assert_eq!(register.source(), Some(source));
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register(GpuMethodId(0x12e0))
                .unwrap()
                .value(),
            Some(&raw)
        );
    }
    for raw in [3, 15, 16, u32::MAX] {
        let before = channel.three_d().clone();
        assert!(matches!(
            dispatch_method(&mut channel, 0x12e0 / 4, raw),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                defined_mask: 0xf,
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);
    }
}

#[test]
fn dither_shadow_replay_and_new_channel_do_not_leak_state() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x124, 0); // Track.
    program_three_d(&mut channel, 0x12e0, 1);
    program_three_d(&mut channel, 0x124, 2); // Passthrough.
    program_three_d(&mut channel, 0x12e0, 2);
    program_three_d(&mut channel, 0x124, 3); // Replay.
    program_three_d(&mut channel, 0x12e0, u32::MAX);
    assert_eq!(
        channel
            .three_d()
            .coverage()
            .alpha_to_coverage_dither()
            .raw(),
        Some(1)
    );
    assert_eq!(
        channel
            .three_d_mut()
            .raw_register(GpuMethodId(0x12e0))
            .unwrap()
            .value(),
        Some(&1)
    );
    assert!(
        three_d_channel()
            .three_d()
            .coverage()
            .alpha_to_coverage_dither()
            .value()
            .is_none()
    );
}

#[test]
fn active_alpha_coverage_rejects_draws_but_does_not_affect_clears() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    program_three_d(&mut channel, 0x12e0, 1);
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    let mut cache = MaxwellThreeDLoweringCache::default();
    for control in [1, 0x10, 0x11] {
        program_three_d(&mut channel, 0x153c, control);
        let dispatch = dispatch_method(&mut channel, 0x0d78 / 4, 3).unwrap();
        let triggered = &dispatch.operations()[0];
        assert!(matches!(lower_maxwell_three_d_operation(
            triggered.state(), &resources, triggered.trigger(), None,
            FrontendSubmissionId::new(1), vec![],
            &lowering_capabilities(BackendFeatures::empty()), &mut cache,
        ), Err(MaxwellThreeDLoweringError::UnsupportedAntiAliasAlphaControl { alpha_to_coverage, alpha_to_one })
            if alpha_to_coverage == (control & 1 != 0) && alpha_to_one == (control & 0x10 != 0)));
        let dispatch = dispatch_method(&mut channel, 0x19d0 / 4, 0x3c).unwrap();
        let triggered = &dispatch.operations()[0];
        assert!(matches!(
            lower_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(2),
                vec![],
                &lowering_capabilities(BackendFeatures::empty()),
                &mut cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
    }
}
