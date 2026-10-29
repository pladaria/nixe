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
fn zcull_maintenance_never_clears_the_attachment_or_invalidates_prepared_draws() {
    let mut channel = three_d_channel();
    for &(method, value, _) in CUBE_REGION {
        dispatch_method(&mut channel, method / 4, value).unwrap();
    }
    let before = channel.three_d().clone();
    let draw = before.draw_state_identity();
    for (method, values) in [(0x12c8, &[0, 0x19, 0x1f_ffff][..]), (0x1958, &[0][..])] {
        for &value in values {
            let dispatch = dispatch_method(&mut channel, method / 4, value).unwrap();
            assert!(dispatch.ordered_operations().is_empty());
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
    let draw = before.draw_state_identity();
    let dispatch = dispatch_method(&mut channel, 0x3840 / 4, 0x19).unwrap();
    assert!(dispatch.ordered_operations().is_empty());
    assert_eq!(channel.three_d().render_targets(), before.render_targets());
    assert!(draw.matches(channel.three_d()));
}

#[test]
fn clearing_zcull_statistics_does_not_clear_depth_or_other_counters() {
    let mut channel = three_d_channel();
    let before = channel.three_d().clone();
    let dispatch = dispatch_method(&mut channel, 0x1530 / 4, 2).unwrap();
    assert!(dispatch.ordered_operations().is_empty());
    assert!(
        before
            .resource_state_identity(MaxwellThreeDResourceRole::DepthStencilTarget)
            .matches(channel.three_d())
    );
    assert!(dispatch_method(&mut channel, 0x1530 / 4, 1).is_err());
}
