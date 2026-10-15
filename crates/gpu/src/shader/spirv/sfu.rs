//! Gradual-underflow support for explicitly approximate guest SFUs. Tiny
//! sin/cos/exp2 inputs have exact rounded identities. sqrt/log2 normalize a
//! subnormal by an exact power of two; exp2 normalizes tiny results before an
//! integer nearest-even downscale. Normal-data evaluation remains native.
//! https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision
use super::*;
impl Math {
    pub(super) fn sfu_normalized_input(&mut self, bits: u32) -> Result<u32> {
        let fraction = self.masked(bits, 0x7fffff)?;
        let fraction = self.b.convert_u_to_f(self.float, None, fraction)?;
        let scale = self
            .b
            .constant_bit32(self.float, 2_f32.powi(-125).to_bits());
        let magnitude = self.b.f_mul(self.float, None, fraction, scale)?;
        self.exact_step(magnitude);
        let magnitude = self.b.bitcast(self.uint, None, magnitude)?;
        let sign = self.masked(bits, 0x80000000)?;
        let signed = self.b.bitwise_or(self.uint, None, magnitude, sign)?;
        Ok(self.b.bitcast(self.float, None, signed)?)
    }
    fn downscale24(&mut self, value: u32) -> Result<u32> {
        let bits = self.b.bitcast(self.uint, None, value)?;
        let exponent = self.masked(bits, 0x7f800000)?;
        let shift = self.constant(23);
        let e = self
            .b
            .shift_right_logical(self.uint, None, exponent, shift)?;
        let limit = self.constant(24);
        let normal = self.b.u_greater_than(self.boolean, None, e, limit)?;
        let bias = self.constant(24 << 23);
        let normalized = self.b.i_sub(self.uint, None, bits, bias)?;
        let fraction = self.masked(bits, 0x7fffff)?;
        let hidden = self.constant(0x800000);
        let sig = self.b.bitwise_or(self.uint, None, fraction, hidden)?;
        let twenty_five = self.constant(25);
        let count = self.b.i_sub(self.uint, None, twenty_five, e)?;
        let thirty_one = self.constant(31);
        let count = self.b.bitwise_and(self.uint, None, count, thirty_one)?;
        let floor = self.b.shift_right_logical(self.uint, None, sig, count)?;
        let one = self.constant(1);
        let zero = self.constant(0);
        let mask = self.b.shift_left_logical(self.uint, None, one, count)?;
        let mask = self.b.i_sub(self.uint, None, mask, one)?;
        let remainder = self.b.bitwise_and(self.uint, None, sig, mask)?;
        let half_shift = self.b.i_sub(self.uint, None, count, one)?;
        let half_shift = self
            .b
            .bitwise_and(self.uint, None, half_shift, thirty_one)?;
        let half = self
            .b
            .shift_left_logical(self.uint, None, one, half_shift)?;
        let above = self.b.u_greater_than(self.boolean, None, remainder, half)?;
        let equal = self.b.i_equal(self.boolean, None, remainder, half)?;
        let parity = self.b.bitwise_and(self.uint, None, floor, one)?;
        let odd = self.b.i_not_equal(self.boolean, None, parity, zero)?;
        let tie = self.b.logical_and(self.boolean, None, equal, odd)?;
        let increment = self.b.logical_or(self.boolean, None, above, tie)?;
        let increment = self.b.select(self.uint, None, increment, one, zero)?;
        let tiny = self.b.i_add(self.uint, None, floor, increment)?;
        let flushed = self.b.i_equal(self.boolean, None, e, zero)?;
        let tiny = self.b.select(self.uint, None, flushed, zero, tiny)?;
        Ok(self.b.select(self.uint, None, normal, normalized, tiny)?)
    }
    pub(crate) fn repair_sfu(
        &mut self,
        value: u32,
        input: u32,
        operation: spv::GlslStd450Op,
    ) -> Result<u32> {
        use spv::GlslStd450Op as G;
        let bits = self.b.bitcast(self.uint, None, input)?;
        let native = self.b.bitcast(self.uint, None, value)?;
        let magnitude = self.masked(bits, 0x7fffffff)?;
        if matches!(operation, G::Sin | G::Cos) {
            // For |x| <= 2^-12, sin(x) rounds to x and cos(x) rounds to 1.
            let bound = self.constant(2_f32.powi(-12).to_bits());
            let small = self
                .b
                .u_less_than_equal(self.boolean, None, magnitude, bound)?;
            let exact = if operation == G::Sin {
                bits
            } else {
                self.constant(1_f32.to_bits())
            };
            let bits = self.b.select(self.uint, None, small, exact, native)?;
            return Ok(self.b.bitcast(self.float, None, bits)?);
        }
        let subnormal = self.nonzero_subnormal(bits)?;
        let repair = if operation == G::Exp2 {
            let exp = self.masked_exp(bits)?;
            let special = self.constant(255);
            let finite = self.b.i_not_equal(self.boolean, None, exp, special)?;
            let tiny = self.tiny(native)?;
            let tiny = self.b.logical_and(self.boolean, None, tiny, finite)?;
            self.b.logical_or(self.boolean, None, subnormal, tiny)?
        } else {
            subnormal
        };
        let header = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        self.b
            .selection_merge(merge, spv::SelectionControl::DONT_FLATTEN)?;
        self.b.branch_conditional(repair, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;
        let set = *self
            .glsl_import
            .get_or_insert_with(|| self.b.ext_inst_import("GLSL.std.450"));
        let repaired = if operation == G::Exp2 {
            let twenty_four = self.b.constant_bit32(self.float, 24_f32.to_bits());
            let normalized = self.b.f_add(self.float, None, input, twenty_four)?;
            self.exact_step(normalized);
            let evaluated = self.b.ext_inst(
                self.float,
                None,
                set,
                operation as u32,
                [rspirv::dr::Operand::IdRef(normalized)],
            )?;
            let scaled = self.downscale24(evaluated)?;
            let one = self.constant(1_f32.to_bits());
            self.b.select(self.uint, None, subnormal, one, scaled)?
        } else {
            let normalized = self.sfu_normalized_input(bits)?;
            let evaluated = self.b.ext_inst(
                self.float,
                None,
                set,
                operation as u32,
                [rspirv::dr::Operand::IdRef(normalized)],
            )?;
            let result = if operation == G::Sqrt {
                let scale = self.b.constant_bit32(self.float, 2_f32.powi(-12).to_bits());
                self.b.f_mul(self.float, None, evaluated, scale)?
            } else {
                let twenty_four = self.b.constant_bit32(self.float, 24_f32.to_bits());
                self.b.f_sub(self.float, None, evaluated, twenty_four)?
            };
            self.exact_step(result);
            self.b.bitcast(self.uint, None, result)?
        };
        let end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        let bits = self
            .b
            .phi(self.uint, None, [(native, header), (repaired, end)])?;
        Ok(self.b.bitcast(self.float, None, bits)?)
    }
}
