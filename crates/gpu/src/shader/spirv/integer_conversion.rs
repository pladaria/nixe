//! Exact directed integer conversion, matching integer_conversion.wgsl.
//! https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
use super::*;

impl Emitter {
    pub(super) fn directed_integer_to_float(
        &mut self,
        bits: u32,
        source_type: ShaderScalarType,
        rounding: ShaderRoundingMode,
    ) -> Result<u32> {
        let zero = self.constant(0);
        let one = self.constant(1);
        let twenty_three = self.constant(23);
        let sign_mask = self.constant(0x80000000);
        let sign = if source_type == ShaderScalarType::Signed32 {
            self.b.bitwise_and(self.uint, None, bits, sign_mask)?
        } else {
            zero
        };
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let negated = self.b.i_sub(self.uint, None, zero, bits)?;
        let magnitude = self.b.select(self.uint, None, negative, negated, bits)?;
        let nonzero = self.b.bitwise_or(self.uint, None, magnitude, one)?;
        let set = *self
            .glsl_import
            .get_or_insert_with(|| self.b.ext_inst_import("GLSL.std.450"));
        // FindUMsb returns the bit index; OR 1 makes the zero case well-defined.
        let exponent = self.b.ext_inst(
            self.uint,
            None,
            set,
            spv::GlslStd450Op::FindUMsb as u32,
            [rspirv::dr::Operand::IdRef(nonzero)],
        )?;
        let wide = self
            .b
            .u_greater_than(self.boolean, None, exponent, twenty_three)?;
        let excess = self.b.i_sub(self.uint, None, exponent, twenty_three)?;
        let discarded = self.b.select(self.uint, None, wide, excess, zero)?;
        let left = self.b.i_sub(self.uint, None, twenty_three, exponent)?;
        let left = self.b.select(self.uint, None, wide, zero, left)?;
        let fraction = self
            .b
            .shift_right_logical(self.uint, None, magnitude, discarded)?;
        let fraction = self.b.shift_left_logical(self.uint, None, fraction, left)?;
        let mask = self.constant(0x7fffff);
        let fraction = self.b.bitwise_and(self.uint, None, fraction, mask)?;
        let bias = self.constant(127);
        let biased = self.b.i_add(self.uint, None, exponent, bias)?;
        let biased = self
            .b
            .shift_left_logical(self.uint, None, biased, twenty_three)?;
        let truncated = self.b.bitwise_or(self.uint, None, biased, fraction)?;
        let mut result = truncated;
        if rounding != ShaderRoundingMode::TowardZero {
            let lost_mask = self.b.shift_left_logical(self.uint, None, one, discarded)?;
            let lost_mask = self.b.i_sub(self.uint, None, lost_mask, one)?;
            let lost = self.b.bitwise_and(self.uint, None, magnitude, lost_mask)?;
            let inexact = self.b.i_not_equal(self.boolean, None, lost, zero)?;
            let away = if rounding == ShaderRoundingMode::TowardNegative {
                negative
            } else {
                self.b.logical_not(self.boolean, None, negative)?
            };
            let increment = self.b.logical_and(self.boolean, None, inexact, away)?;
            let increment = self.b.select(self.uint, None, increment, one, zero)?;
            result = self.b.i_add(self.uint, None, result, increment)?;
        }
        let is_zero = self.b.i_equal(self.boolean, None, magnitude, zero)?;
        let result = self.b.select(self.uint, None, is_zero, zero, result)?;
        Ok(self.b.bitwise_or(self.uint, None, result, sign)?)
    }
}
