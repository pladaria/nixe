//! Exact gradual-underflow lowering on hosts without DenormPreserve32. Native
//! arithmetic remains the normal-data path. Only subnormal operands or possibly
//! flushed results take FP64; every finite intermediate is normal FP64.
//!
//! Float32 products have at most 48 significant bits, hence are exact in FP64.
//! TwoSum retains an FMA sum's rounding residual. Float32 midpoint decisions use
//! that residual, preventing double rounding. Tiny outputs are packed with integer
//! RNE instead of a host float32 conversion which could flush them.
//! https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision
//! https://doi.org/10.1137/0304010
use super::*;
impl Math {
    fn wide_bits(&mut self, value: u32) -> Result<[u32; 2]> {
        let pair = self.b.type_vector(self.uint, 2);
        let words = self.b.bitcast(pair, None, value)?;
        Ok([
            self.b.composite_extract(self.uint, None, words, [0])?,
            self.b.composite_extract(self.uint, None, words, [1])?,
        ])
    }
    fn wide_value(&mut self, wide: u32, low: u32, high: u32) -> Result<u32> {
        let pair = self.b.type_vector(self.uint, 2);
        let words = self.b.composite_construct(pair, None, [low, high])?;
        Ok(self.b.bitcast(wide, None, words)?)
    }
    fn exact_wide_operand(&mut self, value: u32, wide: u32) -> Result<u32> {
        let bits = self.b.bitcast(self.uint, None, value)?;
        let sign = self.masked(bits, 0x80000000)?;
        let fraction = self.masked(bits, 0x7fffff)?;
        let exp = self.masked(bits, 0x7f800000)?;
        let shift = self.constant(23);
        let exp = self.b.shift_right_logical(self.uint, None, exp, shift)?;
        let zero = self.constant(0);
        let subnormal = self.b.i_equal(self.boolean, None, exp, zero)?;
        let hidden = self.constant(0x800000);
        let normal = self.b.bitwise_or(self.uint, None, fraction, hidden)?;
        let significand = self
            .b
            .select(self.uint, None, subnormal, fraction, normal)?;
        let significand = self.b.convert_u_to_f(wide, None, significand)?;
        // Finite input = significand * 2^(e-150); subnormal input uses e=1.
        let one = self.constant(1);
        let exp = self.b.select(self.uint, None, subnormal, one, exp)?;
        let bias = self.constant(1023 - 150);
        let exp = self.b.i_add(self.uint, None, exp, bias)?;
        let shift = self.constant(20);
        let exp = self.b.shift_left_logical(self.uint, None, exp, shift)?;
        let scale = self.wide_value(wide, zero, exp)?;
        let finite = self.b.f_mul(wide, None, significand, scale)?;
        self.exact_step(finite);
        let negative = self.b.f_negate(wide, None, finite)?;
        let neg = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let finite = self.b.select(wide, None, neg, negative, finite)?;
        let max = self.constant(255);
        let input_exp = self.masked_exp(bits)?;
        let special = self.b.i_equal(self.boolean, None, input_exp, max)?;
        let converted = self.b.f_convert(wide, None, value)?;
        Ok(self.b.select(wide, None, special, converted, finite)?)
    }
    pub(super) fn masked_exp(&mut self, bits: u32) -> Result<u32> {
        let exp = self.masked(bits, 0x7f800000)?;
        let shift = self.constant(23);
        Ok(self.b.shift_right_logical(self.uint, None, exp, shift)?)
    }
    pub(super) fn nonzero_subnormal(&mut self, bits: u32) -> Result<u32> {
        let tiny = self.tiny(bits)?;
        let magnitude = self.masked(bits, 0x7fffffff)?;
        let zero = self.constant(0);
        let nonzero = self.b.i_not_equal(self.boolean, None, magnitude, zero)?;
        Ok(self.b.logical_and(self.boolean, None, tiny, nonzero)?)
    }
    fn round_wide32(&mut self, sum: u32, residual: Option<u32>, wide: u32) -> Result<u32> {
        let zero64 = self.b.constant_bit64(wide, 0);
        let min = self.b.constant_bit64(wide, (2_f64.powi(-126)).to_bits());
        let [_, high] = self.wide_bits(sum)?;
        let sign = self.masked(high, 0x80000000)?;
        let zero = self.constant(0);
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let negated = self.b.f_negate(wide, None, sum)?;
        let absolute = self.b.select(wide, None, negative, negated, sum)?;
        let tiny = self.b.f_ord_less_than(self.boolean, None, absolute, min)?;
        let small = self.b.id();
        let normal = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, spv::SelectionControl::NONE)?;
        self.b.branch_conditional(tiny, small, normal, [])?;
        self.block = self.b.begin_block(Some(small))?;
        let scale = self.b.constant_bit64(wide, 2_f64.powi(149).to_bits());
        let scaled = self.b.f_mul(wide, None, absolute, scale)?;
        self.exact_step(scaled);
        let floor = self.b.convert_f_to_u(self.uint, None, scaled)?;
        let integral = self.b.convert_u_to_f(wide, None, floor)?;
        let fraction = self.b.f_sub(wide, None, scaled, integral)?;
        self.exact_step(fraction);
        let half = self.b.constant_bit64(wide, 0.5_f64.to_bits());
        let above = self
            .b
            .f_ord_greater_than(self.boolean, None, fraction, half)?;
        let tie = self.b.f_ord_equal(self.boolean, None, fraction, half)?;
        let (above, tie) = if let Some(residual) = residual {
            let neg_residual = self.b.f_negate(wide, None, residual)?;
            let signed = self
                .b
                .select(wide, None, negative, neg_residual, residual)?;
            let up = self
                .b
                .f_ord_greater_than(self.boolean, None, signed, zero64)?;
            let exactly = self.b.f_ord_equal(self.boolean, None, signed, zero64)?;
            let up_tie = self.b.logical_and(self.boolean, None, tie, up)?;
            (
                self.b.logical_or(self.boolean, None, above, up_tie)?,
                self.b.logical_and(self.boolean, None, tie, exactly)?,
            )
        } else {
            (above, tie)
        };
        let one = self.constant(1);
        let parity = self.b.bitwise_and(self.uint, None, floor, one)?;
        let odd = self.b.i_not_equal(self.boolean, None, parity, zero)?;
        let tie_odd = self.b.logical_and(self.boolean, None, tie, odd)?;
        let round = self.b.logical_or(self.boolean, None, above, tie_odd)?;
        let increment = self.b.select(self.uint, None, round, one, zero)?;
        let mantissa = self.b.i_add(self.uint, None, floor, increment)?;
        let tiny_bits = self.b.bitwise_or(self.uint, None, sign, mantissa)?;
        let small_end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(normal))?;
        let sum = if let Some(residual) = residual {
            // A nonzero residual can change RNE32 only at a float32 midpoint.
            // Move one FP64 ULP in its direction; no blanket epsilon is applied.
            let [low, high] = self.wide_bits(sum)?;
            let tail = self.masked(low, 0x1fffffff)?;
            let midpoint = self.constant(0x10000000);
            let midpoint = self.b.i_equal(self.boolean, None, tail, midpoint)?;
            let nonzero = self
                .b
                .f_ord_not_equal(self.boolean, None, residual, zero64)?;
            let adjust = self.b.logical_and(self.boolean, None, midpoint, nonzero)?;
            let exp = self.masked(high, 0x7ff00000)?;
            let decrement = self.constant(52 << 20);
            let ulp_high = self.b.i_sub(self.uint, None, exp, decrement)?;
            let ulp = self.wide_value(wide, zero, ulp_high)?;
            let negative_residual = self
                .b
                .f_ord_less_than(self.boolean, None, residual, zero64)?;
            let negative_ulp = self.b.f_negate(wide, None, ulp)?;
            let ulp = self
                .b
                .select(wide, None, negative_residual, negative_ulp, ulp)?;
            let adjusted = self.b.f_add(wide, None, sum, ulp)?;
            self.exact_step(adjusted);
            self.b.select(wide, None, adjust, adjusted, sum)?
        } else {
            sum
        };
        let converted = self.b.f_convert(self.float, None, sum)?;
        let normal_bits = self.b.bitcast(self.uint, None, converted)?;
        let normal_end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        Ok(self.b.phi(
            self.uint,
            None,
            [(tiny_bits, small_end), (normal_bits, normal_end)],
        )?)
    }
    pub(crate) fn repair_preserving(
        &mut self,
        value: u32,
        left: u32,
        right: u32,
        addend: Option<u32>,
        multiply: bool,
    ) -> Result<u32> {
        let native = self.b.bitcast(self.uint, None, value)?;
        let a = self.b.bitcast(self.uint, None, left)?;
        let b = self.b.bitcast(self.uint, None, right)?;
        let am = self.masked(a, 0x7fffffff)?;
        let bm = self.masked(b, 0x7fffffff)?;
        let sa = self.nonzero_subnormal(a)?;
        let sb = self.nonzero_subnormal(b)?;
        let mut subnormal = self.b.logical_or(self.boolean, None, sa, sb)?;
        if let Some(c) = addend {
            let cb = self.b.bitcast(self.uint, None, c)?;
            let sc = self.nonzero_subnormal(cb)?;
            subnormal = self.b.logical_or(self.boolean, None, subnormal, sc)?;
        }
        let tiny = self.tiny(native)?;
        let possible = if multiply {
            let zero = self.constant(0);
            let an = self.b.i_not_equal(self.boolean, None, am, zero)?;
            let bn = self.b.i_not_equal(self.boolean, None, bm, zero)?;
            let nonzero = self.b.logical_and(self.boolean, None, an, bn)?;
            if addend.is_some() {
                let ea = self.masked(a, 0x7f800000)?;
                let eb = self.masked(b, 0x7f800000)?;
                let sum = self.b.i_add(self.uint, None, ea, eb)?;
                let bound = self.constant(175 << 23);
                let small = self.b.u_less_than(self.boolean, None, sum, bound)?;
                self.b.logical_and(self.boolean, None, nonzero, small)?
            } else {
                nonzero
            }
        } else {
            self.b.i_not_equal(self.boolean, None, am, bm)?
        };
        let tiny = self.b.logical_and(self.boolean, None, tiny, possible)?;
        let repair = self.b.logical_or(self.boolean, None, subnormal, tiny)?;
        let wide = self.require_repair_float()?;
        let header = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        self.b
            .selection_merge(merge, spv::SelectionControl::DONT_FLATTEN)?;
        self.b.branch_conditional(repair, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;
        let repaired = if let Some(c) = addend {
            self.repair_call(
                math::Repair::PreservingFma,
                [left, right, c],
                |m, [a, b, c]| m.preserving_body(a, b, Some(c), true, wide),
            )?
        } else {
            let kind = if multiply {
                math::Repair::PreservingMultiply
            } else {
                math::Repair::PreservingAdd
            };
            self.repair_call(kind, [left, right], |m, [a, b]| {
                m.preserving_body(a, b, None, multiply, wide)
            })?
        };
        let body_end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        let bits = self
            .b
            .phi(self.uint, None, [(native, header), (repaired, body_end)])?;
        Ok(self.b.bitcast(self.float, None, bits)?)
    }
    fn preserving_body(
        &mut self,
        left: u32,
        right: u32,
        addend: Option<u32>,
        multiply: bool,
        wide: u32,
    ) -> Result<u32> {
        let a = self.exact_wide_operand(left, wide)?;
        let b = self.exact_wide_operand(right, wide)?;
        let product = if multiply {
            self.b.f_mul(wide, None, a, b)?
        } else {
            self.b.f_add(wide, None, a, b)?
        };
        self.exact_step(product);
        let (sum, residual) = if let Some(c) = addend {
            let c = self.exact_wide_operand(c, wide)?;
            let sum = self.b.f_add(wide, None, product, c)?;
            let v = self.b.f_sub(wide, None, sum, product)?;
            let u = self.b.f_sub(wide, None, sum, v)?;
            let w = self.b.f_sub(wide, None, c, v)?;
            let z = self.b.f_sub(wide, None, product, u)?;
            let residual = self.b.f_add(wide, None, z, w)?;
            for step in [sum, v, u, w, z, residual] {
                self.exact_step(step);
            }
            (sum, Some(residual))
        } else {
            (product, None)
        };
        self.round_wide32(sum, residual, wide)
    }
}
impl Math {
    /// Reciprocal and reciprocal sqrt are declared approximate guest operations.
    /// Wide evaluation preserves tiny input/output values rather than accepting
    /// a host flush. The normal-data result still uses the selected native SFU.
    pub(crate) fn repair_inverse(
        &mut self,
        value: u32,
        input: u32,
        square_root: bool,
    ) -> Result<u32> {
        if square_root {
            return self.repair_rsqrt(value, input);
        }
        let bits = self.b.bitcast(self.uint, None, input)?;
        let subnormal = self.nonzero_subnormal(bits)?;
        let result_bits = self.b.bitcast(self.uint, None, value)?;
        let tiny = self.tiny(result_bits)?;
        let exp = self.masked_exp(bits)?;
        let special = self.constant(255);
        let finite = self.b.i_not_equal(self.boolean, None, exp, special)?;
        let tiny = self.b.logical_and(self.boolean, None, tiny, finite)?;
        let repair = self.b.logical_or(self.boolean, None, subnormal, tiny)?;
        let wide = self.require_repair_float()?;
        let header = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        self.b
            .selection_merge(merge, spv::SelectionControl::DONT_FLATTEN)?;
        self.b.branch_conditional(repair, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;
        let input = self.exact_wide_operand(input, wide)?;
        let result = if square_root {
            let set = *self
                .glsl_import
                .get_or_insert_with(|| self.b.ext_inst_import("GLSL.std.450"));
            self.b.ext_inst(
                wide,
                None,
                set,
                spv::GlslStd450Op::InverseSqrt as u32,
                [rspirv::dr::Operand::IdRef(input)],
            )?
        } else {
            let one = self.b.constant_bit64(wide, 1_f64.to_bits());
            self.b.f_div(wide, None, one, input)?
        };
        self.exact_step(result);
        let repaired = self.round_wide32(result, None, wide)?;
        let end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        let bits = self
            .b
            .phi(self.uint, None, [(result_bits, header), (repaired, end)])?;
        Ok(self.b.bitcast(self.float, None, bits)?)
    }
}
impl Math {
    // rsqrt(x) = rsqrt(x * 2^24) * 2^12. Both arithmetic steps and
    // the native SFU result stay normal FP32 for every nonzero subnormal x.
    // Power-of-two scaling is exact; it does not widen the SFU's error bound.
    fn repair_rsqrt(&mut self, value: u32, input: u32) -> Result<u32> {
        let native = self.b.bitcast(self.uint, None, value)?;
        let bits = self.b.bitcast(self.uint, None, input)?;
        let repair = self.nonzero_subnormal(bits)?;
        let header = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        self.b
            .selection_merge(merge, spv::SelectionControl::DONT_FLATTEN)?;
        self.b.branch_conditional(repair, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;
        let scaled = self.sfu_normalized_input(bits)?;
        let set = *self
            .glsl_import
            .get_or_insert_with(|| self.b.ext_inst_import("GLSL.std.450"));
        let inverse = self.b.ext_inst(
            self.float,
            None,
            set,
            spv::GlslStd450Op::InverseSqrt as u32,
            [rspirv::dr::Operand::IdRef(scaled)],
        )?;
        let factor = self.b.constant_bit32(self.float, 2_f32.powi(12).to_bits());
        let inverse = self.b.f_mul(self.float, None, inverse, factor)?;
        self.exact_step(inverse);
        let repaired = self.b.bitcast(self.uint, None, inverse)?;
        let end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        let bits = self
            .b
            .phi(self.uint, None, [(native, header), (repaired, end)])?;
        Ok(self.b.bitcast(self.float, None, bits)?)
    }
}
