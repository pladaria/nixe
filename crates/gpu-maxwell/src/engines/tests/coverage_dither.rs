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
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "STATE_REGISTER"
        );
        assert!(dispatch.operations().is_empty());
        let register = channel.three_d().coverage().alpha_to_coverage_dither();
        assert_eq!(register.value(), Some(&expected));
        assert_eq!(register.raw(), Some(raw));

        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(GpuMethodId(0x12e0))
                .as_ref(),
            Some(&raw)
        );
    }
    for raw in [3, 15, 16, u32::MAX] {
        program_three_d(&mut channel, 0x12e0, raw);
        assert!(
            channel
                .three_d()
                .coverage()
                .alpha_to_coverage_dither()
                .value()
                .is_none()
        );
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
            .raw_register_value(GpuMethodId(0x12e0))
            .as_ref(),
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
