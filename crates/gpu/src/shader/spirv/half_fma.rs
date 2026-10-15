//! Exact binary16 FMA without Float16/FMA extensions. Binary16 products fit
//! exactly in normal binary32. At a half midpoint, aligned integer significands
//! with guard/round/sticky bits remove the intermediate F32 double rounding.
//! https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma
use super::*;
impl Math {
    pub(crate) fn half_fma(&mut self, a: u32, b: u32, c: u32, saturate: bool) -> Result<u32> {
        self.require_float(ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            true,
            true,
            false,
        ))?;
        let af = self.b.bitcast(self.float, None, a)?;
        let bf = self.b.bitcast(self.float, None, b)?;
        let cf = self.b.bitcast(self.float, None, c)?;
        let product = self.b.f_mul(self.float, None, af, bf)?;
        self.b.decorate(product, spv::Decoration::NoContraction, []);
        let product_bits = self.b.bitcast(self.uint, None, product)?;
        let magnitude_mask = self.constant(0x7fffffff);
        let product_abs = self
            .b
            .bitwise_and(self.uint, None, product_bits, magnitude_mask)?;
        let c_abs = self.b.bitwise_and(self.uint, None, c, magnitude_mask)?;
        let big_product = self
            .b
            .u_greater_than_equal(self.boolean, None, product_abs, c_abs)?;
        let big = self.b.select(self.float, None, big_product, product, cf)?;
        let small = self.b.select(self.float, None, big_product, cf, product)?;
        let sum = self.b.f_add(self.float, None, big, small)?;
        self.b.decorate(sum, spv::Decoration::NoContraction, []);
        let bits = self.b.bitcast(self.uint, None, sum)?;
        let magnitude = self.b.bitwise_and(self.uint, None, bits, magnitude_mask)?;
        let zero = self.constant(0);
        let one = self.constant(1);
        let twenty_three = self.constant(23);
        let sign_mask = self.constant(0x80000000);
        let sign = self.b.bitwise_and(self.uint, None, bits, sign_mask)?;
        let exponent = self
            .b
            .shift_right_logical(self.uint, None, magnitude, twenty_three)?;
        let fraction_mask = self.constant(0x7fffff);
        let hidden = self.constant(0x800000);
        let fraction = self
            .b
            .bitwise_and(self.uint, None, magnitude, fraction_mask)?;
        let significand = self.b.bitwise_or(self.uint, None, fraction, hidden)?;
        let midpoint_mask = self.constant(0x1fff);
        let midpoint_value = self.constant(0x1000);
        let low = self
            .b
            .bitwise_and(self.uint, None, magnitude, midpoint_mask)?;
        let normal_midpoint = self.b.i_equal(self.boolean, None, low, midpoint_value)?;
        let tiny_limit = self.constant(0x38800000);
        let tiny = self
            .b
            .u_less_than(self.boolean, None, magnitude, tiny_limit)?;
        let bias126 = self.constant(126);
        let tiny_shift = self.b.i_sub(self.uint, None, bias126, exponent)?;
        let thirty_one = self.constant(31);
        let tiny_shift = self
            .b
            .bitwise_and(self.uint, None, tiny_shift, thirty_one)?;
        let tiny_unit = self
            .b
            .shift_left_logical(self.uint, None, one, tiny_shift)?;
        let tiny_mask = self.b.i_sub(self.uint, None, tiny_unit, one)?;
        let tiny_half = self
            .b
            .shift_right_logical(self.uint, None, tiny_unit, one)?;
        let tiny_remainder = self
            .b
            .bitwise_and(self.uint, None, significand, tiny_mask)?;
        let tiny_midpoint = self
            .b
            .i_equal(self.boolean, None, tiny_remainder, tiny_half)?;
        let midpoint = self
            .b
            .select(self.boolean, None, tiny, tiny_midpoint, normal_midpoint)?;
        let lower = self.constant(0x33000000);
        let upper = self.constant(0x477ff000);
        let above = self
            .b
            .u_greater_than_equal(self.boolean, None, magnitude, lower)?;
        let below = self
            .b
            .u_less_than_equal(self.boolean, None, magnitude, upper)?;
        let range = self.b.logical_and(self.boolean, None, above, below)?;
        let repair = self.b.logical_and(self.boolean, None, range, midpoint)?;
        let original = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, spv::SelectionControl::NONE)?;
        self.b.branch_conditional(repair, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;
        let big_bits = self
            .b
            .select(self.uint, None, big_product, product_bits, c)?;
        let small_bits = self
            .b
            .select(self.uint, None, big_product, c, product_bits)?;
        let exponent_mask = self.constant(255);
        let big_exp = self
            .b
            .shift_right_logical(self.uint, None, big_bits, twenty_three)?;
        let big_exp = self
            .b
            .bitwise_and(self.uint, None, big_exp, exponent_mask)?;
        let small_exp = self
            .b
            .shift_right_logical(self.uint, None, small_bits, twenty_three)?;
        let small_exp = self
            .b
            .bitwise_and(self.uint, None, small_exp, exponent_mask)?;
        let distance = self.b.i_sub(self.uint, None, big_exp, small_exp)?;
        let three = self.constant(3);
        let big_sig = self
            .b
            .bitwise_and(self.uint, None, big_bits, fraction_mask)?;
        let big_sig = self.b.bitwise_or(self.uint, None, big_sig, hidden)?;
        let big_sig = self.b.shift_left_logical(self.uint, None, big_sig, three)?;
        let small_sig = self
            .b
            .bitwise_and(self.uint, None, small_bits, fraction_mask)?;
        let small_sig = self.b.bitwise_or(self.uint, None, small_sig, hidden)?;
        let small_sig = self
            .b
            .shift_left_logical(self.uint, None, small_sig, three)?;
        let masked_distance = self.b.bitwise_and(self.uint, None, distance, thirty_one)?;
        let lost_mask = self
            .b
            .shift_left_logical(self.uint, None, one, masked_distance)?;
        let lost_mask = self.b.i_sub(self.uint, None, lost_mask, one)?;
        let lost = self.b.bitwise_and(self.uint, None, small_sig, lost_mask)?;
        let sticky = self.b.i_not_equal(self.boolean, None, lost, zero)?;
        let sticky = self.b.select(self.uint, None, sticky, one, zero)?;
        let shifted = self
            .b
            .shift_right_logical(self.uint, None, small_sig, masked_distance)?;
        let shifted = self.b.bitwise_or(self.uint, None, shifted, sticky)?;
        let thirty_two = self.constant(32);
        let far = self
            .b
            .u_greater_than_equal(self.boolean, None, distance, thirty_two)?;
        let shifted = self.b.select(self.uint, None, far, one, shifted)?;
        let small_abs = self
            .b
            .bitwise_and(self.uint, None, small_bits, magnitude_mask)?;
        let small_zero = self.b.i_equal(self.boolean, None, small_abs, zero)?;
        let shifted = self.b.select(self.uint, None, small_zero, zero, shifted)?;
        let xor = self.b.bitwise_xor(self.uint, None, big_bits, small_bits)?;
        let xor = self.b.bitwise_and(self.uint, None, xor, sign_mask)?;
        let same_sign = self.b.i_equal(self.boolean, None, xor, zero)?;
        let plus = self.b.i_add(self.uint, None, big_sig, shifted)?;
        let minus = self.b.i_sub(self.uint, None, big_sig, shifted)?;
        let exact_sig = self.b.select(self.uint, None, same_sign, plus, minus)?;
        let sum_shift = self.b.i_add(self.uint, None, exponent, three)?;
        let sum_shift = self.b.i_sub(self.uint, None, sum_shift, big_exp)?;
        let sum_shift = self.b.bitwise_and(self.uint, None, sum_shift, thirty_one)?;
        let sum_sig = self
            .b
            .shift_left_logical(self.uint, None, significand, sum_shift)?;
        let inexact = self.b.i_not_equal(self.boolean, None, exact_sig, sum_sig)?;
        let exponent_plus_one = self.b.i_add(self.uint, None, exponent, one)?;
        let exact_cancellation =
            self.b
                .u_less_than(self.boolean, None, exponent_plus_one, big_exp)?;
        let not_exact = self.b.logical_not(self.boolean, None, exact_cancellation)?;
        let needs = self.b.logical_and(self.boolean, None, inexact, not_exact)?;
        let up = self
            .b
            .u_greater_than(self.boolean, None, exact_sig, sum_sig)?;
        let incremented = self.b.i_add(self.uint, None, magnitude, one)?;
        let decremented = self.b.i_sub(self.uint, None, magnitude, one)?;
        let adjusted = self
            .b
            .select(self.uint, None, up, incremented, decremented)?;
        let adjusted = self.b.select(self.uint, None, needs, adjusted, magnitude)?;
        let adjusted = self.b.bitwise_or(self.uint, None, adjusted, sign)?;
        let body_end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        let corrected = self
            .b
            .phi(self.uint, None, [(bits, original), (adjusted, body_end)])?;
        let packed = self.pack_half(corrected)?;
        if !saturate {
            return Ok(packed);
        }
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let infinity = self.constant(0x7f800000);
        let nan = self
            .b
            .u_greater_than(self.boolean, None, magnitude, infinity)?;
        let zero_result = self.b.logical_or(self.boolean, None, negative, nan)?;
        let one_float = self.constant(0x3f800000);
        let above_one = self
            .b
            .u_greater_than_equal(self.boolean, None, magnitude, one_float)?;
        let half_one = self.constant(0x3c00);
        let saturated = self
            .b
            .select(self.uint, None, above_one, half_one, packed)?;
        Ok(self
            .b
            .select(self.uint, None, zero_result, zero, saturated)?)
    }
}
