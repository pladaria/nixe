//! Reference conversion keeps every F64 rounding bit. The half crate's F16C
//! path converts through F32 and can double-round, so it is unsuitable here.
pub(super) fn round_half(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0x7ff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let unbiased = exponent - 1023;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -25 {
        return sign;
    }
    let significand = fraction | (1_u64 << 52);
    let shift = 42 + (-14 - unbiased).max(0) as u32;
    let truncated = significand >> shift;
    let remainder = significand & ((1_u64 << shift) - 1);
    let midpoint = 1_u64 << (shift - 1);
    let rounded = truncated
        + u64::from(remainder > midpoint || (remainder == midpoint && truncated & 1 != 0));
    sign | if unbiased < -14 {
        rounded as u16
    } else {
        (((unbiased + 14) as u16) << 10) + rounded as u16
    }
}
