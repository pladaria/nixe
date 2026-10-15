//! Bit-exact scalar rounding/conversion. No host FP conversion, denormal mode
//! or out-of-range conversion result enters the guest contract.
//! https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
use super::*;
impl Math {
    pub(crate) fn round_integral_bits(
        &mut self,
        bits: u32,
        mode: ShaderRoundingMode,
        flush: bool,
    ) -> Result<u32> {
        let bits = if flush {
            self.flush_denormal(bits)?
        } else {
            bits
        };
        let sign_mask = self.constant(0x80000000);
        let magnitude_mask = self.constant(0x7fffffff);
        let sign = self.b.bitwise_and(self.uint, None, bits, sign_mask)?;
        let magnitude = self.b.bitwise_and(self.uint, None, bits, magnitude_mask)?;
        let zero = self.constant(0);
        let one = self.constant(1);
        let mask = self.constant(255);
        let shift23 = self.constant(23);
        let exponent = self
            .b
            .shift_right_logical(self.uint, None, magnitude, shift23)?;
        let exponent = self.b.bitwise_and(self.uint, None, exponent, mask)?;
        let limit = self.constant(150);
        let large = self
            .b
            .u_greater_than_equal(self.boolean, None, exponent, limit)?;
        let count = self.b.i_sub(self.uint, None, limit, exponent)?;
        let too_many = self.b.u_greater_than(self.boolean, None, count, shift23)?;
        let count = self.b.select(self.uint, None, too_many, shift23, count)?;
        let count = self.b.select(self.uint, None, large, zero, count)?;
        let unit = self.b.shift_left_logical(self.uint, None, one, count)?;
        let lost_mask = self.b.i_sub(self.uint, None, unit, one)?;
        let lost = self.b.bitwise_and(self.uint, None, magnitude, lost_mask)?;
        let keep = self.b.not(self.uint, None, lost_mask)?;
        let truncated = self.b.bitwise_and(self.uint, None, magnitude, keep)?;
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let inexact = self.b.i_not_equal(self.boolean, None, lost, zero)?;
        let increment = match mode {
            ShaderRoundingMode::TowardZero => self.false_value,
            ShaderRoundingMode::TowardNegative => {
                self.b.logical_and(self.boolean, None, negative, inexact)?
            }
            ShaderRoundingMode::TowardPositive => {
                let positive = self.b.logical_not(self.boolean, None, negative)?;
                self.b.logical_and(self.boolean, None, positive, inexact)?
            }
            ShaderRoundingMode::NearestEven => {
                let half = self.b.shift_right_logical(self.uint, None, unit, one)?;
                let above = self.b.u_greater_than(self.boolean, None, lost, half)?;
                let equal = self.b.i_equal(self.boolean, None, lost, half)?;
                let bit = self.b.bitwise_and(self.uint, None, truncated, unit)?;
                let odd = self.b.i_not_equal(self.boolean, None, bit, zero)?;
                let tie = self.b.logical_and(self.boolean, None, equal, odd)?;
                self.b.logical_or(self.boolean, None, above, tie)?
            }
        };
        let addition = self.b.select(self.uint, None, increment, unit, zero)?;
        let regular = self.b.i_add(self.uint, None, truncated, addition)?;
        let one_float = self.constant(0x3f800000);
        let small = self
            .b
            .u_less_than(self.boolean, None, magnitude, one_float)?;
        let nonzero = self.b.i_not_equal(self.boolean, None, magnitude, zero)?;
        let small_up = match mode {
            ShaderRoundingMode::TowardZero => self.false_value,
            ShaderRoundingMode::TowardNegative => {
                self.b.logical_and(self.boolean, None, negative, nonzero)?
            }
            ShaderRoundingMode::TowardPositive => {
                let positive = self.b.logical_not(self.boolean, None, negative)?;
                self.b.logical_and(self.boolean, None, positive, nonzero)?
            }
            ShaderRoundingMode::NearestEven => {
                let half = self.constant(0x3f000000);
                self.b.u_greater_than(self.boolean, None, magnitude, half)?
            }
        };
        let rounded_small = self.b.select(self.uint, None, small_up, one_float, zero)?;
        let rounded = self
            .b
            .select(self.uint, None, small, rounded_small, regular)?;
        let rounded = self.b.bitwise_or(self.uint, None, rounded, sign)?;
        Ok(self.b.select(self.uint, None, large, bits, rounded)?)
    }
    pub(crate) fn float_to_integer_bits(
        &mut self,
        bits: u32,
        signed: bool,
        width: u32,
        mode: ShaderRoundingMode,
        flush: bool,
    ) -> Result<u32> {
        if !matches!(width, 8 | 16 | 32) {
            return Err(self.unsupported("integer conversion width must be 8/16/32"));
        }
        let rounded = self.round_integral_bits(bits, mode, flush)?;
        let mask = self.constant(0x7fffffff);
        let sign_mask = self.constant(0x80000000);
        let magnitude = self.b.bitwise_and(self.uint, None, rounded, mask)?;
        let sign = self.b.bitwise_and(self.uint, None, rounded, sign_mask)?;
        let zero = self.constant(0);
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let twenty_three = self.constant(23);
        let exponent = self
            .b
            .shift_right_logical(self.uint, None, magnitude, twenty_three)?;
        let frac_mask = self.constant(0x7fffff);
        let fraction = self.b.bitwise_and(self.uint, None, magnitude, frac_mask)?;
        let hidden = self.constant(0x800000);
        let significand = self.b.bitwise_or(self.uint, None, fraction, hidden)?;
        let bias = self.constant(150);
        let right = self.b.i_sub(self.uint, None, bias, exponent)?;
        let left = self.b.i_sub(self.uint, None, exponent, bias)?;
        let thirty_one = self.constant(31);
        let right = self.b.bitwise_and(self.uint, None, right, thirty_one)?;
        let left = self.b.bitwise_and(self.uint, None, left, thirty_one)?;
        let r = self
            .b
            .shift_right_logical(self.uint, None, significand, right)?;
        let l = self
            .b
            .shift_left_logical(self.uint, None, significand, left)?;
        let wide = self
            .b
            .u_greater_than_equal(self.boolean, None, exponent, bias)?;
        let integer = self.b.select(self.uint, None, wide, l, r)?;
        let one_float = self.constant(0x3f800000);
        let small = self
            .b
            .u_less_than(self.boolean, None, magnitude, one_float)?;
        let integer = self.b.select(self.uint, None, small, zero, integer)?;
        let exponent_limit = 127 + width - u32::from(signed);
        let float_limit = self.constant(exponent_limit << 23);
        let overflow = self
            .b
            .u_greater_than_equal(self.boolean, None, magnitude, float_limit)?;
        let mut result = if signed {
            let minimum = self.constant(1 << (width - 1));
            let maximum = self.constant((1u32 << (width - 1)) - 1);
            let negated = self.b.i_sub(self.uint, None, zero, integer)?;
            let value = self.b.select(self.uint, None, negative, negated, integer)?;
            let min = self.b.i_sub(self.uint, None, zero, minimum)?;
            let clamp = self.b.select(self.uint, None, negative, min, maximum)?;
            self.b.select(self.uint, None, overflow, clamp, value)?
        } else {
            let maximum = self.constant(if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            });
            let value = self.b.select(self.uint, None, overflow, maximum, integer)?;
            self.b.select(self.uint, None, negative, zero, value)?
        };
        let infinity = self.constant(0x7f800000);
        let nan = self
            .b
            .u_greater_than(self.boolean, None, magnitude, infinity)?;
        result = self.b.select(self.uint, None, nan, zero, result)?;
        Ok(result)
    }
}
