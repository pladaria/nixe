// Directed U32/S32 -> F32: assemble the truncated significand, then round
// away from zero when the requested direction and discarded bits require it.
// Integer arithmetic also handles U32_MAX and I32_MIN without conversion overflow.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
fn nixe_integer_to_float(bits: u32, signed: bool, mode: u32) -> u32 {
    let negative = signed && (bits & 0x80000000u) != 0u;
    let magnitude = select(bits, 0u - bits, negative);
    let exponent = 31u - countLeadingZeros(magnitude | 1u);
    let discarded = u32(max(i32(exponent) - 23, 0));
    let shift_left = u32(max(23 - i32(exponent), 0));
    let significand = (magnitude >> discarded) << shift_left;
    var result = ((exponent + 127u) << 23u) | (significand & 0x7fffffu);
    let remainder = magnitude & ((1u << discarded) - 1u);
    let away = (mode == 1u && negative) || (mode == 2u && !negative);
    result += u32(away && remainder != 0u);
    result = select(result, 0u, magnitude == 0u);
    return result | select(0u, 0x80000000u, negative);
}
