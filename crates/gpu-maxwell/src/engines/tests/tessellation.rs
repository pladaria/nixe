use super::super::threed::tessellation::{draw_state, validate_default_level_inputs};
use super::*;
use nixe_gpu::{
    TessellationControl, TessellationDomain, TessellationOutput, TessellationSpacing,
    TessellationWinding,
};

fn patch_channel() -> MaxwellGpuChannel {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1618, 14);
    program_three_d(&mut channel, 0x0dcc, 4);
    program_three_d(&mut channel, 0x0320, 0x201);
    program_three_d(&mut channel, 0x2040, 0x11);
    program_three_d(&mut channel, 0x2080, 0x21);
    program_three_d(&mut channel, 0x20c0, 0x31);
    channel
}

#[test]
fn tessellation_mode_domain_spacing_and_winding_are_not_dump_labels() {
    for domain in 0..3 {
        for spacing in 0..3 {
            let flags = if domain == 0 { 0x100 } else { 0x200 };
            let mode = MaxwellThreeDTessellationMode::new(domain | spacing << 4 | flags)
                .lower()
                .unwrap();
            assert_eq!(
                mode.domain,
                [
                    TessellationDomain::Isolines,
                    TessellationDomain::Triangles,
                    TessellationDomain::Quads
                ][domain as usize]
            );
            assert_eq!(
                mode.spacing,
                [
                    TessellationSpacing::Equal,
                    TessellationSpacing::FractionalOdd,
                    TessellationSpacing::FractionalEven
                ][spacing as usize]
            );
            assert_eq!(
                mode.output,
                if domain == 0 {
                    TessellationOutput::Lines
                } else {
                    TessellationOutput::Triangles(TessellationWinding::CounterClockwise)
                }
            );
            assert_eq!(
                MaxwellThreeDTessellationMode::new(domain | spacing << 4)
                    .lower()
                    .unwrap()
                    .output,
                TessellationOutput::Points
            );
        }
    }
    assert_eq!(
        MaxwellThreeDTessellationMode::new(0x301)
            .lower()
            .unwrap()
            .output,
        TessellationOutput::Triangles(TessellationWinding::Clockwise)
    );
}

#[test]
fn tessellation_shadow_replay_preserves_effective_source_and_channel_isolation() {
    let mut channel = three_d_channel();
    let dispatch = dispatch_method(&mut channel, 0x320 / 4, 0x201).unwrap();
    assert_eq!(
        dispatch.methods()[0].metadata().method_name(),
        "SET_TESSELLATION_PARAMETERS"
    );
    assert_eq!(
        channel
            .three_d()
            .shader_bindings()
            .tessellation_mode()
            .source(),
        Some(dispatch.methods()[0].method().source())
    );
    program_three_d(&mut channel, 0x124, 2);
    program_three_d(&mut channel, 0x320, 0x322);
    program_three_d(&mut channel, 0x124, 3);
    let replay = dispatch_method(&mut channel, 0x320 / 4, u32::MAX).unwrap();
    let register = channel.three_d().shader_bindings().tessellation_mode();
    assert_eq!(register.value().unwrap().raw(), 0x201);
    assert_eq!(
        register.source(),
        Some(replay.methods()[0].method().source())
    );
    assert_eq!(register.source().unwrap().argument(), 0x201);
    let fresh = three_d_channel();
    assert!(
        fresh
            .three_d()
            .shader_bindings()
            .tessellation_mode()
            .value()
            .is_none()
    );
    assert!(
        fresh
            .three_d()
            .shader_bindings()
            .tessellation_lod(MaxwellThreeDTessellationLod::InnerU)
            .value()
            .is_none()
    );
}

#[test]
fn tessellation_draw_state_invalidates_without_retranslating_guest_code() {
    let mut channel = patch_channel();
    let shader = channel.three_d().shader_state_identity();
    let initial = channel.three_d().draw_state_identity();
    program_three_d(&mut channel, 0x320, 0x201);
    assert!(initial.matches(channel.three_d()));
    program_three_d(&mut channel, 0x320, 0x211);
    let mode_changed = channel.three_d().draw_state_identity();
    assert!(!initial.matches(channel.three_d()));
    program_three_d(&mut channel, 0x324, 0x7fc0_0042);
    assert!(!mode_changed.matches(channel.three_d()));
    assert!(shader.matches(channel.three_d()));
}

#[test]
fn tessellation_modes_fail_at_consumption_with_method_provenance() {
    let mut channel = patch_channel();
    for (raw, expected) in [
        (4, MaxwellTessellationModeError::ReservedBits),
        (3, MaxwellTessellationModeError::ReservedDomain),
        (0x231, MaxwellTessellationModeError::ReservedSpacing),
        (
            0x200,
            MaxwellTessellationModeError::UnsupportedIsolineConnectivity,
        ),
    ] {
        program_three_d(&mut channel, 0x320, raw);
        assert!(
            matches!(draw_state(channel.three_d()), Err(MaxwellThreeDLoweringError::TessellationMode { reason, source: Some(source), .. }) if reason == expected && source.argument() == raw)
        );
    }
    program_three_d(&mut channel, 0x320, 0x201);
    let state = draw_state(channel.three_d()).unwrap().unwrap();
    assert_eq!(state.input_control_points, 4);
    assert_eq!(state.control, TessellationControl::Shader);
    program_three_d(&mut channel, 0x20c0, 0x30);
    assert!(matches!(
        draw_state(channel.three_d()),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "patch draw requires tessellation evaluation shader"
        ))
    ));
    program_three_d(&mut channel, 0x20c0, 0x31);
    program_three_d(&mut channel, 0x2100, 0x41);
    assert!(matches!(
        draw_state(channel.three_d()),
        Err(MaxwellThreeDLoweringError::UnsupportedShaderStage(
            MaxwellThreeDShaderStage::Geometry
        ))
    ));
    program_three_d(&mut channel, 0x1618, 4);
    assert!(matches!(
        draw_state(channel.three_d()),
        Err(MaxwellThreeDLoweringError::TessellationStageTopology)
    ));
}

#[test]
fn default_levels_preserve_bits_and_only_require_domain_consumed_lanes() {
    let mut channel = patch_channel();
    program_three_d(&mut channel, 0x2080, 0x20);
    assert!(matches!(
        draw_state(channel.three_d()),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(_))
    ));
    for (method, bits) in [
        (0x324, 0x8000_0000),
        (0x328, 0x7fc0_0042),
        (0x32c, 0xff80_0000),
        (0x334, 0x7f80_0000),
    ] {
        program_three_d(&mut channel, method, bits);
    }
    assert_eq!(
        draw_state(channel.three_d()).unwrap().unwrap().control,
        TessellationControl::DefaultLevels {
            outer: [0x8000_0000, 0x7fc0_0042, 0xff80_0000, 0],
            inner: [0x7f80_0000, 0],
            defined: 0b01_0111
        }
    );
    program_three_d(&mut channel, 0x320, 0x202);
    assert!(matches!(
        draw_state(channel.three_d()),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(_))
    ));
}

#[test]
fn evaluation_reads_cannot_consume_unprogrammed_domain_unused_levels() {
    let control = TessellationControl::DefaultLevels {
        outer: [0; 4],
        inner: [0; 2],
        defined: 0b01_0111,
    };
    let ir = nixe_gpu::ShaderIr::new(
        ShaderStage::TessellationEvaluation,
        vec![
            nixe_gpu::ShaderInterfaceElement::new(
                nixe_gpu::ShaderIoLocation::TessLevelInner,
                1,
                nixe_gpu::ShaderScalarType::Float32,
                None,
            )
            .unwrap(),
        ],
        vec![],
        vec![],
        vec![],
    );
    assert!(matches!(
        validate_default_level_inputs(control, &ir),
        Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "default tessellation level consumed by evaluation shader"
        ))
    ));
    validate_default_level_inputs(
        TessellationControl::DefaultLevels {
            outer: [0; 4],
            inner: [0; 2],
            defined: 0x3f,
        },
        &ir,
    )
    .unwrap();
    validate_default_level_inputs(TessellationControl::Shader, &ir).unwrap();
}
