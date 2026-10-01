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
