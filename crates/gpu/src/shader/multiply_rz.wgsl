// Exact binary32 multiplication with RZ, DAZ and FTZ using a 48-bit product.
// No host floating-point rounding or denormal behavior enters this operation.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul
fn nixe_multiply_rz_ftz(a: u32, b: u32, zero_absorbs: bool) -> u32 {
    let sign = (a ^ b) & 0x80000000u;
    let ea = (a >> 23u) & 255u;
    let eb = (b >> 23u) & 255u;
    if (zero_absorbs && (ea == 0u || eb == 0u)) { return 0u; }
    if (ea == 255u && (a & 0x007fffffu) != 0u) { return a | 0x00400000u; }
    if (eb == 255u && (b & 0x007fffffu) != 0u) { return b | 0x00400000u; }
    if (ea == 255u || eb == 255u) {
        if (ea == 0u || eb == 0u) { return 0x7fc00000u; }
        return sign | 0x7f800000u;
    }
    if (ea == 0u || eb == 0u) { return sign; }
    let ma = (a & 0x007fffffu) | 0x00800000u;
    let mb = (b & 0x007fffffu) | 0x00800000u;
    let p0 = (ma & 65535u) * (mb & 65535u);
    let p1 = (ma >> 16u) * (mb & 65535u);
    let p2 = (ma & 65535u) * (mb >> 16u);
    let middle = (p0 >> 16u) + (p1 & 65535u) + (p2 & 65535u);
    let low = (p0 & 65535u) | (middle << 16u);
    let high = (ma >> 16u) * (mb >> 16u) + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u);
    let top = (high & 32768u) != 0u;
    let exponent = i32(ea) + i32(eb) - 127 + select(0, 1, top);
    if (exponent <= 0) { return sign; }
    if (exponent >= 255) { return sign | 0x7f7fffffu; }
    let mantissa = select((high << 9u) | (low >> 23u), (high << 8u) | (low >> 24u), top);
    return sign | (u32(exponent) << 23u) | (mantissa & 0x007fffffu);
}
