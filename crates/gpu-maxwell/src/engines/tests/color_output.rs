use super::*;
use nixe_gpu::{
    BlendComponent, BlendFactor as F, BlendOperation as O, ColorBlendState, ColorOutputState,
    ColorWriteMask,
};

#[test]
fn deko_blending_consumes_initial_separate_alpha_without_selector_write() {
    let mut address_space = resource_address_space();
    let vertex = map_resource(
        &mut address_space,
        CanonicalAllocation::zeroed(0x4000, 0x1000)
            .unwrap()
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        71,
        0,
    )
    .offset()
    .get();
    let target = map_resource(
        &mut address_space,
        CanonicalAllocation::zeroed(0x10000, 0x1000)
            .unwrap()
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        72,
        0xfe,
    )
    .offset()
    .get();
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 0, target, 0xd5);
    for (method, value) in [
        (0x121c, color_target_selection_raw(1, [0; 8])),
        (0x12e4, 1),
        (0x1360, 1),
        // Captured demo packet starts at 0x1e04, never writes 0x1e00.
        (0x1e04, 1),
        (0x1e08, 5),
        (0x1e0c, 6),
        (0x1e10, 1),
        (0x1e14, 2),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let (shaders, mut cache) = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let mut serial = 0;
    let mut lower = |channel: &mut MaxwellGpuChannel| {
        serial += 1;
        let dispatch = dispatch_method(channel, 0x0d78 / 4, 3).unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(serial),
            Vec::new(),
            &capabilities,
            &mut cache,
        )?;
        Ok::<_, MaxwellLoweringError>(
            plan.submission()
                .operations()
                .iter()
                .find_map(|op| {
                    if let GpuCommand::Draw(draw) = op.command() {
                        Some(draw.prepared.clone())
                    } else {
                        None
                    }
                })
                .unwrap(),
        )
    };
    // Only the selector has a verified initial value: missing consumed factors
    // still fail instead of inheriting fabricated blending defaults.
    assert!(matches!(
        lower(&mut channel),
        Err(MaxwellLoweringError::IncompleteBlendState {
            target: Some(0),
            field: "SET_BLEND_PER_TARGET_COEFF_DESTINATION_ALPHA"
        })
    ));
    program_three_d(&mut channel, 0x1e18, 6);
    let separate = lower(&mut channel).unwrap();
    let blend = separate.color_outputs[0].blend.unwrap();
    assert_eq!(
        blend,
        ColorBlendState {
            color: BlendComponent {
                operation: O::Add,
                source: F::SourceAlpha,
                destination: F::OneMinusSourceAlpha
            },
            alpha: BlendComponent {
                operation: O::Add,
                source: F::One,
                destination: F::OneMinusSourceAlpha
            },
        }
    );
    let repeated = lower(&mut channel).unwrap();
    assert_eq!(repeated.color_outputs, separate.color_outputs);
    assert_eq!(repeated.pipeline, separate.pipeline);
    // deko_examples2 explicitly binds default multisampling: the stored 2x2
    // dither footprint is inactive because alpha-to-coverage is disabled.
    program_three_d(&mut channel, 0x153c, 0);
    for footprint in [0, 1, 2] {
        program_three_d(&mut channel, 0x12e0, footprint);
        let inactive = lower(&mut channel).unwrap();
        assert_eq!(inactive.color_outputs, separate.color_outputs);
        assert_eq!(inactive.pipeline, separate.pipeline);
    }
    // Enabling after a successful cached draw must not reuse that draw silently.
    program_three_d(&mut channel, 0x153c, 1);
    assert!(matches!(
        lower(&mut channel),
        Err(MaxwellLoweringError::UnsupportedAntiAliasAlphaControl {
            alpha_to_coverage: true,
            alpha_to_one: false
        })
    ));
    program_three_d(&mut channel, 0x153c, 0);
    assert_eq!(lower(&mut channel).unwrap().pipeline, separate.pipeline);
    program_three_d(&mut channel, 0x1e00, 0);
    let shared = lower(&mut channel).unwrap();
    assert_eq!(
        shared.color_outputs[0].blend,
        Some(ColorBlendState {
            color: blend.color,
            alpha: blend.color
        })
    );
    assert_eq!(shared.pipeline, separate.pipeline);
    program_three_d(&mut channel, 0x1e00, 1);
    let restored = lower(&mut channel).unwrap();
    assert_eq!(restored.color_outputs, separate.color_outputs);
    assert_eq!(restored.pipeline, separate.pipeline);
}

#[test]
fn separate_alpha_initial_state_is_consistent_across_targets_and_shadow_replay() {
    for _ in 0..2 {
        let mut channel = three_d_channel();
        for target in 0..8 {
            let register = &channel.three_d().fixed_function().per_target_blend()[target][0];
            assert_eq!(
                register.origin(),
                MaxwellThreeDRegisterOrigin::VerifiedReset
            );
            assert_eq!(register.raw(), Some(1));
            assert_eq!(
                register.value(),
                Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
            );
            assert_eq!(register.source(), None);
            let raw = channel
                .three_d_mut()
                .raw_register(GpuMethodId(0x1e00 + target as u32 * 0x20))
                .unwrap();
            assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
            assert_eq!(raw.value(), Some(&1));
            assert_eq!(raw.source(), None);
        }
        // Replay an unprogrammed selector: the supplied invalid argument must
        // be replaced by the initial shadow value, independently for each RT.
        program_three_d(&mut channel, 0x124, 3);
        for target in 0..8 {
            program_three_d(&mut channel, 0x1e00 + target * 0x20, u32::MAX);
            assert_eq!(
                channel.three_d().fixed_function().per_target_blend()[target as usize][0].raw(),
                Some(1)
            );
        }
        // Passthrough changes live state, not the initial shadow value.
        program_three_d(&mut channel, 0x124, 2);
        program_three_d(&mut channel, 0x1e00, 0);
        program_three_d(&mut channel, 0x124, 3);
        program_three_d(&mut channel, 0x1e00, 0);
        assert_eq!(
            channel.three_d().fixed_function().per_target_blend()[0][0].raw(),
            Some(1)
        );
        // A tracked write must subsequently override that initial value.
        program_three_d(&mut channel, 0x124, 0);
        program_three_d(&mut channel, 0x1e00, 0);
        program_three_d(&mut channel, 0x124, 3);
        program_three_d(&mut channel, 0x1e00, 1);
        assert_eq!(
            channel.three_d().fixed_function().per_target_blend()[0][0].raw(),
            Some(0)
        );
    }
}

#[test]
fn consumed_color_state_routes_physical_targets_and_invalidates_prepared_draws() {
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
    let mut channel = three_d_channel();
    program_basic_draw_state(&mut channel, vertex);
    program_color_target(&mut channel, 1, target, 0xd5);
    for (method, value) in [
        (
            0x121c,
            color_target_selection_raw(1, [1, 0, 0, 0, 0, 0, 0, 0]),
        ),
        (0x0f90, 0),
        (0x1a04, 0x1011),
        (0x135c, 1),
        (0x133c, 1),
        (0x1340, 0x8006),
        (0x1344, 0x4302),
        (0x1348, 0x4303),
        (0x134c, 1),
        (0x1350, 2),
        (0x1358, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let (shaders, mut cache) = translated_graphics_shaders();
    let capabilities =
        lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
    let mut serial = 0;
    let mut lower = |channel: &mut MaxwellGpuChannel| {
        serial += 1;
        let dispatch = dispatch_method(channel, 0x0d78 / 4, 3).unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = lower_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(serial),
            Vec::new(),
            &capabilities,
            &mut cache,
        )?;
        Ok::<_, MaxwellLoweringError>(
            plan.submission()
                .operations()
                .iter()
                .find_map(|op| {
                    if let GpuCommand::Draw(draw) = op.command() {
                        Some(draw.prepared.clone())
                    } else {
                        None
                    }
                })
                .expect("prepared draw"),
        )
    };
    let component = |operation, source, destination| BlendComponent {
        operation,
        source,
        destination,
    };
    let common = lower(&mut channel).unwrap();
    assert_eq!(
        common.color_outputs[0],
        ColorOutputState {
            blend: Some(ColorBlendState {
                color: component(O::Add, F::SourceAlpha, F::OneMinusSourceAlpha),
                alpha: component(O::Add, F::One, F::Zero),
            }),
            write_mask: ColorWriteMask::new(true, true, false, true),
        }
    );
    assert_eq!(common.color_outputs[1], ColorOutputState::REPLACE);

    // Per-target state is physical target 1, not fragment-output slot 0.
    // Min/max ignore unset coefficients and separate-alpha=false ignores its op.
    for (method, value) in [(0x12e4, 1), (0x1364, 1), (0x1e20, 0), (0x1e24, 0x8008)] {
        program_three_d(&mut channel, method, value);
    }
    let per_target = lower(&mut channel).unwrap();
    assert_eq!(per_target.pipeline, common.pipeline);
    let max = component(O::Max, F::One, F::One);
    assert_eq!(
        per_target.color_outputs[0].blend,
        Some(ColorBlendState {
            color: max,
            alpha: max
        })
    );

    program_three_d(&mut channel, 0x0f90, 1);
    program_three_d(&mut channel, 0x1a00, 0x0100);
    assert_eq!(
        lower(&mut channel).unwrap().color_outputs[0].write_mask,
        ColorWriteMask::new(false, false, true, false)
    );

    // Constants and dual-source factors remain explicit unsupported boundaries.
    for value in [0xc001, 0xc900, 0x0c] {
        program_three_d(&mut channel, 0x1e24, 1);
        program_three_d(&mut channel, 0x1e28, value);
        program_three_d(&mut channel, 0x1e2c, 2);
        assert!(
            matches!(lower(&mut channel), Err(MaxwellLoweringError::UnsupportedBlendFactor { target: Some(1), value: actual }) if actual == value)
        );
    }
    // A disabled equation does not consume those factors, and state changes do
    // not require recompiling shaders or a second neutral pipeline family.
    program_three_d(&mut channel, 0x1364, 0);
    let disabled = lower(&mut channel).unwrap();
    assert_eq!(disabled.pipeline, common.pipeline);
    assert_eq!(disabled.color_outputs[0].blend, None);

    program_three_d(&mut channel, 0x12e4, 0);
    program_three_d(&mut channel, 0x0f90, 0);
    assert_eq!(
        lower(&mut channel).unwrap().color_outputs,
        common.color_outputs
    );
}
