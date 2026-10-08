use super::*;

// Configuration emitted by deko3d when binding rotating cube's depth target.
// These are just public register commands, independent of the local dump file.
const CUBE_REGION: &[(u32, u32, &str)] = &[
    (0x07e0, 0x07f8_0000, "SET_ZCULL_REGION_LOCATION"),
    (0x07e4, 0x07f8, "SET_ZCULL_REGION_ALIQUOTS"),
    (0x15c8, 2, "SET_ZCULL_REGION_FORMAT"),
    (0x07c0, 1920, "SET_ZCULL_REGION_SIZE_A"),
    (0x07c4, 1088, "SET_ZCULL_REGION_SIZE_B"),
    (0x07c8, 1, "SET_ZCULL_REGION_SIZE_C"),
    (0x15fc, 0, "SET_ZCULL_REGION_PIXEL_OFFSET_A"),
    (0x1600, 0, "SET_ZCULL_REGION_PIXEL_OFFSET_B"),
    (0x07cc, 0, "SET_ZCULL_REGION_PIXEL_OFFSET_C"),
    (0x02e8, 0, "SET_ZCULL_SUBREGION"),
    (0x0dbc, 0x0001_0000, "SET_ZCULL_DIR_FORMAT"),
];

#[test]
fn cube_zcull_configuration_is_typed_source_preserving_and_cache_neutral() {
    use MaxwellThreeDZCullAxis::{Depth, Height, Width};
    let mut channel = three_d_channel();
    let before = channel.three_d().clone();
    let frontend = channel.frontend();
    let two_d = channel.two_d().clone();
    let draw = before.fixed_draw_identity();
    let shaders = before.shader_state_identity();
    let resources =
        before.resource_state_identity(&[MaxwellThreeDResourceRole::DepthStencilTarget], false);
    for &(method, argument, name) in CUBE_REGION {
        let dispatch = dispatch_method(&mut channel, method / 4, argument).unwrap();
        assert!(dispatch.ordered_operations().is_empty());
        let dispatched = dispatch.methods()[0];
        assert_eq!(dispatched.metadata().method_name(), name);
        let state = channel.three_d().zcull();
        let (raw, source) = match method {
            0x07e0 => (
                state.region_location().raw(),
                state.region_location().source(),
            ),
            0x07e4 => (
                state.region_aliquots().raw(),
                state.region_aliquots().source(),
            ),
            0x15c8 => (state.region_format().raw(), state.region_format().source()),
            0x02e8 => (state.subregion().raw(), state.subregion().source()),
            0x0dbc => (
                state.direction_format().raw(),
                state.direction_format().source(),
            ),
            _ => {
                let register = match method {
                    0x07c0 => state.region_size(Width),
                    0x07c4 => state.region_size(Height),
                    0x07c8 => state.region_size(Depth),
                    0x15fc => state.region_pixel_offset(Width),
                    0x1600 => state.region_pixel_offset(Height),
                    0x07cc => state.region_pixel_offset(Depth),
                    _ => unreachable!(),
                };
                (register.raw(), register.source())
            }
        };
        assert_eq!(raw, Some(argument));
        assert_eq!(source, Some(dispatched.method().source()));
        assert!(draw.matches(channel.three_d()));
        assert!(shaders.matches(channel.three_d()));
        assert!(resources.matches(channel.three_d()));
    }
    let state = channel.three_d().zcull();
    let location = state.region_location().value().unwrap();
    assert_eq!(location.start_aliquot(), 0);
    assert_eq!(location.aliquot_count(), 2040);
    assert_eq!(state.region_location().raw(), Some(0x07f8_0000));
    assert_eq!(
        state.region_location().source().unwrap().method(),
        GpuMethodId(0x07e0)
    );
    assert_eq!(state.region_aliquots().value(), Some(&2040));
    assert_eq!(
        state.region_format().value(),
        Some(&MaxwellThreeDZCullRegionFormat::Z4x2)
    );
    for (axis, value) in [(Width, 1920), (Height, 1088), (Depth, 1)] {
        assert_eq!(state.region_size(axis).value(), Some(&value));
        assert_eq!(state.region_pixel_offset(axis).value(), Some(&0));
        assert_eq!(
            state.region_size(axis).origin(),
            MaxwellThreeDRegisterOrigin::Programmed
        );
    }
    assert!(!state.subregion().value().unwrap().enabled());
    assert_eq!(state.subregion().value().unwrap().normalized_aliquots(), 0);
    let direction = state.direction_format().value().unwrap();
    assert!(!direction.greater());
    assert_eq!(direction.format(), MaxwellThreeDZCullDepthFormat::Float);
    assert_eq!(
        before.zcull().region_location().origin(),
        MaxwellThreeDRegisterOrigin::Unset
    );
    assert_eq!(channel.three_d().render_targets(), before.render_targets());
    assert_eq!(channel.frontend(), frontend);
    assert_eq!(channel.two_d(), &two_d);
}

#[test]
fn zcull_region_fields_cover_documented_values_without_inventing_geometry_constraints() {
    let mut channel = three_d_channel();
    for value in [0, 0x07f8_0000, 0xffff_ffff, 0x1234_5678] {
        dispatch_method(&mut channel, 0x07e0 / 4, value).unwrap();
        let location = *channel.three_d().zcull().region_location().value().unwrap();
        assert_eq!(location.start_aliquot(), value as u16);
        assert_eq!(location.aliquot_count(), (value >> 16) as u16);
        assert_eq!(location.raw(), value);
    }
    for value in 0..=12 {
        dispatch_method(&mut channel, 0x15c8 / 4, value).unwrap();
        assert_eq!(
            channel
                .three_d()
                .zcull()
                .region_format()
                .value()
                .unwrap()
                .raw(),
            value
        );
    }
    for value in [0, 1, 0x0fff_fff0, 0x0fff_fff1] {
        dispatch_method(&mut channel, 0x02e8 / 4, value).unwrap();
        let subregion = *channel.three_d().zcull().subregion().value().unwrap();
        assert_eq!(subregion.enabled(), value & 1 != 0);
        assert_eq!(subregion.normalized_aliquots(), value >> 4);
        assert_eq!(subregion.raw(), value);
    }
    for format in 0..=2 {
        for direction in 0..=1 {
            let value = (format << 16) | direction;
            dispatch_method(&mut channel, 0x0dbc / 4, value).unwrap();
            assert_eq!(
                channel
                    .three_d()
                    .zcull()
                    .direction_format()
                    .value()
                    .unwrap()
                    .raw(),
                value
            );
        }
    }
    for method in [0x07c0, 0x07c4, 0x07c8, 0x07cc, 0x07e4, 0x15fc, 0x1600] {
        for value in [0, 1, 0xffff] {
            dispatch_method(&mut channel, method / 4, value).unwrap();
        }
    }
}

#[test]
fn invalid_zcull_fields_fail_without_mutating_state_and_keep_valid_packet_prefix() {
    let mut channel = three_d_channel();
    for &(method, value, _) in CUBE_REGION {
        dispatch_method(&mut channel, method / 4, value).unwrap();
    }
    for (method, value) in [
        (0x07c0, 0x10000),
        (0x07c4, 0x10000),
        (0x07c8, 0x10000),
        (0x07cc, 0x10000),
        (0x07e4, 0x10000),
        (0x15fc, 0x10000),
        (0x1600, 0x10000),
        (0x15c8, 13),
        (0x15c8, 14),
        (0x15c8, 15),
        (0x15c8, 0x10),
        (0x02e8, 2),
        (0x02e8, 4),
        (0x02e8, 8),
        (0x02e8, 1 << 28),
        (0x0dbc, 2),
        (0x0dbc, 0x30000),
        (0x0dbc, u32::MAX),
    ] {
        let before = channel.three_d().clone();
        let error = dispatch_method(&mut channel, method / 4, value).unwrap_err();
        let source = match error {
            MaxwellEngineDispatchError::InvalidMethodValue { source, .. }
            | MaxwellEngineDispatchError::InvalidMethodEncoding { source, .. } => source,
            other => panic!("{other:?}"),
        };
        assert_eq!(source.method(), GpuMethodId(method));
        assert_eq!(source.argument(), value);
        assert_eq!(channel.three_d(), &before);
    }
    let before = channel.three_d().clone();
    assert!(dispatch_incrementing(&mut channel, 0x07c0 / 4, &[640, 0x10000]).is_err());
    assert_eq!(
        channel
            .three_d()
            .zcull()
            .region_size(MaxwellThreeDZCullAxis::Width)
            .value(),
        Some(&640)
    );
    assert_eq!(
        channel
            .three_d()
            .zcull()
            .region_size(MaxwellThreeDZCullAxis::Height),
        before.zcull().region_size(MaxwellThreeDZCullAxis::Height)
    );
}

#[test]
fn zcull_configuration_participates_in_shadow_replay_and_preserves_snapshots() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x0124, 0);
    program_three_d(&mut channel, 0x07e0, 0x07f8_0000);
    program_three_d(&mut channel, 0x0124, 2);
    program_three_d(&mut channel, 0x07e0, 0x1234_5678);
    let snapshot = channel.three_d().clone();
    program_three_d(&mut channel, 0x0124, 3);
    program_three_d(&mut channel, 0x07e0, 0);
    assert_eq!(
        channel.three_d().zcull().region_location().raw(),
        Some(0x07f8_0000)
    );
    assert_eq!(snapshot.zcull().region_location().raw(), Some(0x1234_5678));
}

#[test]
fn zcull_maintenance_never_clears_the_attachment_or_invalidates_prepared_draws() {
    let mut channel = three_d_channel();
    for &(method, value, _) in CUBE_REGION {
        dispatch_method(&mut channel, method / 4, value).unwrap();
    }
    let before = channel.three_d().clone();
    let draw = before.fixed_draw_identity();
    for (method, values) in [(0x12c8, &[0, 0x19, 0x1f_ffff][..]), (0x1958, &[0][..])] {
        for &value in values {
            let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            assert_eq!(channel.three_d().zcull(), before.zcull());
            assert_eq!(channel.three_d().render_targets(), before.render_targets());
            assert!(draw.matches(channel.three_d()));
        }
    }
    for (method, value) in [(0x12c8, 1 << 21), (0x1958, 1), (0x1958, u32::MAX)] {
        assert!(matches!(
            dispatch_method(&mut channel, method / 4, value),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { .. })
        ));
    }
}

#[test]
fn mme_can_clear_zcull_conservatively_without_emitting_an_attachment_clear() {
    let mut channel = three_d_channel();
    // A minimal macro emits the same method/argument as deko3d's
    // ConditionalZcullInvalidate after detecting a changed depth target.
    let set_method = 1 | (2 << 4) | ((0x12c8 / 4) << 14);
    let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
    load_mme_program(
        &mut channel,
        8,
        &[set_method, send_parameter_and_exit, 0x11],
    );
    let before = channel.three_d().clone();
    let draw = before.fixed_draw_identity();
    let dispatch = dispatch_method(&mut channel, 0x3840 / 4, 0x19).unwrap();
    assert!(dispatch.ordered_operations().is_empty());
    assert_eq!(channel.three_d().render_targets(), before.render_targets());
    assert_eq!(channel.three_d().zcull(), before.zcull());
    assert!(draw.matches(channel.three_d()));
}

#[test]
fn zcull_geometry_does_not_fabricate_draw_resources_or_counter_operations() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x121c, 0);
    for &(method, value, _) in CUBE_REGION {
        dispatch_method(&mut channel, method / 4, value).unwrap();
    }
    let resources =
        resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space()).unwrap();
    assert!(matches!(
        lower_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source: channel
                    .three_d()
                    .zcull()
                    .region_location()
                    .source()
                    .unwrap(),
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &mut MaxwellLoweringCache::default(),
        ),
        Err(MaxwellLoweringError::ShaderTranslationRequired)
    ));
}

#[test]
fn zcull_storage_retains_addresses_without_creating_attachment_operations() {
    let mut channel = three_d_channel();
    let before = channel.three_d().clone();
    for (index, (method, value)) in [
        (0x7e8, 0xab),
        (0x7ec, 0x1234_0000),
        (0x7f0, 0xab),
        (0x7f4, 0x1234_ffff),
    ]
    .into_iter()
    .enumerate()
    {
        let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
        assert!(dispatch.ordered_operations().is_empty());
        let register = channel.three_d().zcull().storage_word(index).unwrap();
        assert_eq!(register.raw(), Some(value));
        assert_eq!(
            register.source(),
            Some(dispatch.methods()[0].method().source())
        );
    }
    assert!(channel.three_d().fixed_draw_identity().matches(&before));
    assert!(
        before
            .resource_state_identity(&[MaxwellThreeDResourceRole::DepthStencilTarget], false)
            .matches(channel.three_d())
    );
    assert!(dispatch_method(&mut channel, 0x7e8 / 4, 0x100).is_err());
    assert_eq!(
        channel.three_d().zcull().storage_word(0).unwrap().raw(),
        Some(0xab)
    );
}

#[test]
fn zcull_serialization_does_not_change_depth_resources_or_draw_state() {
    let mut channel = three_d_channel();
    let before = channel.three_d().clone();
    for method in [0x1464, 0x1500] {
        for value in [0, 1] {
            let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            assert!(before.fixed_draw_identity().matches(channel.three_d()));
            assert!(
                before
                    .resource_state_identity(
                        &[MaxwellThreeDResourceRole::DepthStencilTarget],
                        false
                    )
                    .matches(channel.three_d())
            );
        }
        assert!(dispatch_method(&mut channel, method / 4, 2).is_err());
    }
}

#[test]
fn zcull_allocation_decodes_documented_formats_and_rejects_reserved_values() {
    let mut channel = three_d_channel();
    for format in (0..=12).chain([15]) {
        let raw = 0x0012_34ab | format << 24;
        dispatch_method(&mut channel, 0x2f8 / 4, raw).unwrap();
        let register = channel.three_d().zcull().subregion_allocation();
        assert_eq!(register.raw(), Some(raw));
        let allocation = register.value().unwrap();
        assert_eq!(allocation.id(), 0xab);
        assert_eq!(allocation.aliquots(), 0x1234);
        assert_eq!(
            allocation.format().map(u32::from),
            if format == 15 { None } else { Some(format) }
        );
    }
    for raw in [13 << 24, 14 << 24, 1 << 28] {
        assert!(dispatch_method(&mut channel, 0x2f8 / 4, raw).is_err());
    }
    for value in [0, 1] {
        dispatch_method(&mut channel, 0x2fc / 4, value).unwrap();
        assert_eq!(
            channel.three_d().zcull().subregion_algorithm().raw(),
            Some(value)
        );
    }
    assert!(dispatch_method(&mut channel, 0x2fc / 4, 2).is_err());
}

#[test]
fn zcull_report_configuration_does_not_emit_a_counter_report() {
    let mut channel = three_d_channel();
    for value in [0, 1, 0x0ff1] {
        let dispatch = dispatch_method(&mut channel, 0x36c / 4, value).unwrap();
        assert!(dispatch.ordered_operations().is_empty());
        assert_eq!(
            channel.three_d().zcull().report_selection().raw(),
            Some(value)
        );
    }
    for kind in 0..4 {
        for enabled in [0, 1] {
            let value = kind << 4 | enabled;
            let dispatch = dispatch_method(&mut channel, 0x370 / 4, value).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            assert_eq!(channel.three_d().zcull().report_type().raw(), Some(value));
        }
    }
    assert!(dispatch_method(&mut channel, 0x36c / 4, 2).is_err());
    assert!(dispatch_method(&mut channel, 0x370 / 4, 0x41).is_err());
}

#[test]
fn clearing_zcull_statistics_does_not_clear_depth_or_other_counters() {
    let mut channel = three_d_channel();
    let before = channel.three_d().clone();
    let dispatch = dispatch_method(&mut channel, 0x1530 / 4, 2).unwrap();
    assert!(dispatch.ordered_operations().is_empty());
    assert_eq!(channel.three_d().zcull(), before.zcull());
    assert!(
        before
            .resource_state_identity(&[MaxwellThreeDResourceRole::DepthStencilTarget], false)
            .matches(channel.three_d())
    );
    assert!(dispatch_method(&mut channel, 0x1530 / 4, 1).is_err());
}
