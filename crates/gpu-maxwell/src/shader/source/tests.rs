use super::super::test_support::program_three_d;
use super::*;
use crate::{MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuChannel, SWITCH_1_GM20B_PROFILE};
use nixe_gpu::{ShaderIoLocation, ShaderScalarType};
use std::collections::BTreeMap;

#[test]
fn vertex_integer_formats_define_shader_input_scalar_types() {
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    // Attribute 1: stream 0, active, offset 2, R8_G8, NUM_UINT.
    program_three_d(&mut channel, 0x1164, 0x2300_0100);

    let types = maxwell_vertex_input_types_iter(channel.three_d()).collect::<BTreeMap<_, _>>();
    assert_eq!(
        types.get(&ShaderIoLocation::Generic(1)),
        Some(&ShaderScalarType::Unsigned32)
    );
}

#[test]
fn window_origin_changes_shader_source_identity_and_facing_flip_does_not_change_y_direction() {
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    for (method, value) in [
        (0, SWITCH_1_GM20B_PROFILE.classes().three_d().0),
        (0x2040, 0x10),
        (0x1608, 0),
        (0x160c, 0x80000),
        (0x2140, 0x51),
        (0x2144, 0),
        (0x214c, 6),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let original = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    let original_identity = channel.three_d().shader_state_identity();
    assert!(!original.y_negate);
    program_three_d(&mut channel, 0x13ac, 1);
    let reversed = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    assert!(reversed.y_negate);
    assert_ne!(original.fingerprint(), reversed.fingerprint());
    assert!(!original_identity.matches(channel.three_d()));
    assert!(!original.matches(&reversed.materialize()));
    assert_eq!(
        reversed.fingerprint(),
        nixe_gpu::cache_fingerprint(&reversed.materialize())
    );
    program_three_d(&mut channel, 0x13ac, 0x10);
    let facing_flip = prepare_maxwell_shader_translation_source(channel.three_d()).unwrap();
    assert!(!facing_flip.y_negate);
    assert_eq!(original.fingerprint(), facing_flip.fingerprint());
    assert!(original.matches(&facing_flip.materialize()));
}
