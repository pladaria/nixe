//! Exact binary32 RTZ multiplication with DAZ/FTZ. A native 24x24-bit integer
//! product avoids dependence on host float rounding, overflow or denormal modes.
//! https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul
use super::*;

impl Math {
    pub(super) fn multiply_rz_ftz(&mut self, a: u32, b: u32) -> Result<u32> {
        let zero = self.constant(0);
        let shift23 = self.constant(23);
        let shift24 = self.constant(24);
        let shift8 = self.constant(8);
        let shift9 = self.constant(9);
        let max_exp = self.constant(255);
        let hidden = self.constant(0x800000);
        let fraction = self.constant(0x7fffff);
        let sign_mask = self.constant(0x80000000);
        let xor = self.b.bitwise_xor(self.uint, None, a, b)?;
        let sign = self.b.bitwise_and(self.uint, None, xor, sign_mask)?;
        let ea = self.b.shift_right_logical(self.uint, None, a, shift23)?;
        let eb = self.b.shift_right_logical(self.uint, None, b, shift23)?;
        let ea = self.b.bitwise_and(self.uint, None, ea, max_exp)?;
        let eb = self.b.bitwise_and(self.uint, None, eb, max_exp)?;
        let fa = self.b.bitwise_and(self.uint, None, a, fraction)?;
        let fb = self.b.bitwise_and(self.uint, None, b, fraction)?;
        let ma = self.b.bitwise_or(self.uint, None, fa, hidden)?;
        let mb = self.b.bitwise_or(self.uint, None, fb, hidden)?;
        let pair = self.b.type_struct([self.uint, self.uint]);
        let product = self.b.u_mul_extended(pair, None, ma, mb)?;
        let low = self.b.composite_extract(self.uint, None, product, [0])?;
        let high = self.b.composite_extract(self.uint, None, product, [1])?;
        let top_mask = self.constant(0x8000);
        let top = self.b.bitwise_and(self.uint, None, high, top_mask)?;
        let top = self.b.i_not_equal(self.boolean, None, top, zero)?;
        let one = self.constant(1);
        let normalizer = self.b.select(self.uint, None, top, one, zero)?;
        let exponent = self.b.i_add(self.uint, None, ea, eb)?;
        let exponent = self.b.i_add(self.uint, None, exponent, normalizer)?;
        let bias = self.constant(127);
        let exponent = self.b.i_sub(self.uint, None, exponent, bias)?;
        let exponent = self.b.bitcast(self.int, None, exponent)?;
        let izero = self.b.constant_bit32(self.int, 0);
        let imax = self.b.constant_bit32(self.int, 255);
        let under = self
            .b
            .s_less_than_equal(self.boolean, None, exponent, izero)?;
        let over = self
            .b
            .s_greater_than_equal(self.boolean, None, exponent, imax)?;
        let left_shift = self.b.select(self.uint, None, top, shift8, shift9)?;
        let right_shift = self.b.select(self.uint, None, top, shift24, shift23)?;
        let hi = self
            .b
            .shift_left_logical(self.uint, None, high, left_shift)?;
        let lo = self
            .b
            .shift_right_logical(self.uint, None, low, right_shift)?;
        let significand = self.b.bitwise_or(self.uint, None, hi, lo)?;
        let significand = self.b.bitwise_and(self.uint, None, significand, fraction)?;
        let exponent = self.b.bitcast(self.uint, None, exponent)?;
        let exponent = self
            .b
            .shift_left_logical(self.uint, None, exponent, shift23)?;
        let magnitude = self.b.bitwise_or(self.uint, None, exponent, significand)?;
        let maximum = self.constant(0x7f7fffff);
        let magnitude = self.b.select(self.uint, None, over, maximum, magnitude)?;
        let magnitude = self.b.select(self.uint, None, under, zero, magnitude)?;
        let mut result = self.b.bitwise_or(self.uint, None, sign, magnitude)?;
        let a_zero = self.b.i_equal(self.boolean, None, ea, zero)?;
        let b_zero = self.b.i_equal(self.boolean, None, eb, zero)?;
        let any_zero = self.b.logical_or(self.boolean, None, a_zero, b_zero)?;
        result = self.b.select(self.uint, None, any_zero, sign, result)?;
        let a_special = self.b.i_equal(self.boolean, None, ea, max_exp)?;
        let b_special = self.b.i_equal(self.boolean, None, eb, max_exp)?;
        let any_special = self
            .b
            .logical_or(self.boolean, None, a_special, b_special)?;
        let infinity = self.constant(0x7f800000);
        let infinity = self.b.bitwise_or(self.uint, None, sign, infinity)?;
        let nan = self.constant(0x7fc00000);
        let special = self.b.select(self.uint, None, any_zero, nan, infinity)?;
        result = self
            .b
            .select(self.uint, None, any_special, special, result)?;
        let a_nan = self.b.i_not_equal(self.boolean, None, fa, zero)?;
        let b_nan = self.b.i_not_equal(self.boolean, None, fb, zero)?;
        let a_nan = self.b.logical_and(self.boolean, None, a_special, a_nan)?;
        let b_nan = self.b.logical_and(self.boolean, None, b_special, b_nan)?;
        let quiet = self.constant(0x400000);
        let qa = self.b.bitwise_or(self.uint, None, a, quiet)?;
        let qb = self.b.bitwise_or(self.uint, None, b, quiet)?;
        result = self.b.select(self.uint, None, b_nan, qb, result)?;
        self.b
            .select(self.uint, None, a_nan, qa, result)
            .map_err(Into::into)
    }
}
