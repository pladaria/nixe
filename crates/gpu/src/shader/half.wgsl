// IEEE binary16 conversion. Integer rounding fixes the mode and preserves
// subnormals independently of optional host half-arithmetic capabilities.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-cvt
fn nixe_unpack_half(bits: u32) -> u32 {
    let sign = (bits & 0x8000u) << 16u;
    let exponent = (bits >> 10u) & 31u;
    let fraction = bits & 1023u;
    if (exponent == 0u) {
        return sign | bitcast<u32>(f32(fraction) * 0.000000059604644775390625);
    }
    if (exponent == 31u) {
        return sign | 0x7f800000u | (fraction << 13u) | select(0u, 0x00400000u, fraction != 0u);
    }
    return sign | ((exponent + 112u) << 23u) | (fraction << 13u);
}
fn nixe_pack_half(bits: u32) -> u32 {
    let sign = (bits >> 16u) & 0x8000u;
    let magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return sign | 0x7e00u | ((magnitude >> 13u) & 1023u);
    }
    if (magnitude >= 0x477ff000u) { return sign | 0x7c00u; }
    if (magnitude < 0x33000000u) { return sign; }
    if (magnitude < 0x38800000u) {
        let shift = 126u - (magnitude >> 23u);
        let fraction = (magnitude & 0x007fffffu) | 0x00800000u;
        let truncated = fraction >> shift;
        let remainder = fraction & ((1u << shift) - 1u);
        let midpoint = 1u << (shift - 1u);
        return sign | (truncated + select(0u, 1u,
            remainder > midpoint || (remainder == midpoint && (truncated & 1u) != 0u)));
    }
    let rounded = magnitude + 0xfffu + ((magnitude >> 13u) & 1u);
    return sign | ((rounded >> 13u) - 0x1c000u);
}
