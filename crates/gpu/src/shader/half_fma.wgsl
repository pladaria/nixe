// The product of two finite binary16 significands fits exactly in binary32.
// Only a binary32 sum exactly at a half midpoint can double-round. Integer
// significand alignment with guard/round/sticky bits determines which side of
// that midpoint the exact sum lies on; move it one F32 ULP in that direction.
// This avoids algebraic reassociation of an FP residual and handles 65520 overflow.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma
fn nixe_fma_half(a: u32, b: u32, c: u32, saturate: bool) -> u32 {
    let product = bitcast<f32>(a) * bitcast<f32>(b);
    let addend = bitcast<f32>(c);
    let product_bits = bitcast<u32>(product);
    let big_product = (product_bits & 0x7fffffffu) >= (c & 0x7fffffffu);
    let big = select(addend, product, big_product);
    let small = select(product, addend, big_product);
    let sum = big + small;
    var bits = bitcast<u32>(sum);
    var magnitude = bits & 0x7fffffffu;
    if (saturate) {
        if ((bits & 0x80000000u) != 0u || magnitude > 0x7f800000u) { return 0u; }
        if (magnitude >= 0x3f800000u) { return 0x3c00u; }
    }
    if (magnitude < 0x33000000u || magnitude > 0x477ff000u) { return nixe_pack_half(bits); }
    var midpoint = (magnitude & 0x1fffu) == 0x1000u;
    if (magnitude < 0x38800000u) {
        let shift = 126u - (magnitude >> 23u);
        let fraction = (magnitude & 0x007fffffu) | 0x00800000u;
        midpoint = (fraction & ((1u << shift) - 1u)) == (1u << (shift - 1u));
    }
    if (midpoint) {
        let big_bits = select(c, product_bits, big_product);
        let small_bits = select(product_bits, c, big_product);
        let big_exponent = (big_bits >> 23u) & 255u;
        let small_exponent = (small_bits >> 23u) & 255u;
        let sum_exponent = magnitude >> 23u;
        // Cancellation by more than one exponent is exact (Sterbenz's lemma).
        if (sum_exponent + 1u < big_exponent) { return nixe_pack_half(bits); }
        let distance = big_exponent - small_exponent;
        let big_significand = ((big_bits & 0x007fffffu) | 0x00800000u) << 3u;
        var small_significand = ((small_bits & 0x007fffffu) | 0x00800000u) << 3u;
        if ((small_bits & 0x7fffffffu) == 0u) {
            small_significand = 0u;
        } else if (distance >= 32u) {
            small_significand = 1u;
        } else {
            let sticky = (small_significand & ((1u << distance) - 1u)) != 0u;
            small_significand = (small_significand >> distance) | select(0u, 1u, sticky);
        }
        let exact_significand = select(big_significand - small_significand,
            big_significand + small_significand, ((big_bits ^ small_bits) & 0x80000000u) == 0u);
        let sum_significand = ((magnitude & 0x007fffffu) | 0x00800000u)
            << (sum_exponent + 3u - big_exponent);
        if (exact_significand != sum_significand) {
            magnitude = select(magnitude - 1u, magnitude + 1u, exact_significand > sum_significand);
            bits = (bits & 0x80000000u) | magnitude;
        }
    }
    return nixe_pack_half(bits);
}
