//! Exact DAZ+FTZ repair when float32 denorm preservation is unavailable.
//!
//! Vulkan permits pre-rounding flushing, including a result that RNE would round
//! up to MIN_POSITIVE. Merely masking the native result is therefore insufficient.
//! https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision
//!
//! Normal arithmetic stays float32. Only possibly tiny products use float64. A
//! product of two DAZ float32 values is exact in float64 (48 significant bits,
//! exponents within [-252, 256]). TwoSum retains the low part of FMA's addition:
//! converting a rounded float64 sum to float32 would permit double rounding.
//! All nonzero intermediates are normal float64, so no DenormPreserve64 is needed.
//! Error-free TwoSum: https://doi.org/10.1137/0304010

use super::*;

impl Emitter {
    fn masked(&mut self, value: u32, mask: u32) -> Result<u32> {
        let mask = self.constant(mask);
        Ok(self.b.bitwise_and(self.uint, None, value, mask)?)
    }

    fn tiny(&mut self, bits: u32) -> Result<u32> {
        let exponent = self.masked(bits, 0x7f80_0000)?;
        let zero = self.constant(0);
        Ok(self.b.i_equal(self.boolean, None, exponent, zero)?)
    }

    pub(super) fn repair_binary_underflow(
        &mut self,
        value: u32,
        left: u32,
        right: u32,
        multiply: bool,
    ) -> Result<u32> {
        if multiply {
            return self.repair_product_underflow(value, left, right, None);
        }
        // A sum of two normal float32 operands is a multiple of 2^-149.
        // It cannot fall between the largest subnormal and MIN_POSITIVE.
        // Only the sign of a flushed nonzero cancellation needs reconstruction.
        let bits = self.b.bitcast(self.uint, None, value)?;
        let left = self.b.bitcast(self.uint, None, left)?;
        let right = self.b.bitcast(self.uint, None, right)?;
        let a = self.masked(left, 0x7fff_ffff)?;
        let b = self.masked(right, 0x7fff_ffff)?;
        let unequal = self.b.i_not_equal(self.boolean, None, a, b)?;
        let larger = self.b.u_greater_than(self.boolean, None, a, b)?;
        let largest = self.b.select(self.uint, None, larger, left, right)?;
        let sign = self.masked(largest, 0x8000_0000)?;
        let tiny = self.tiny(bits)?;
        let repair = self.b.logical_and(self.boolean, None, tiny, unequal)?;
        let result = self.b.select(self.uint, None, repair, sign, bits)?;
        Ok(self.b.bitcast(self.float, None, result)?)
    }

    fn require_repair_float(&mut self) -> Result<u32> {
        if let Some(ty) = self.repair_float {
            return Ok(ty);
        }
        let caps = self.options.float64;
        if !caps.enabled || !caps.rounding_mode_rte || !caps.signed_zero_inf_nan_preserve {
            return Err(self.unsupported("DAZ/FTZ multiplication without DenormPreserve32 requires enabled float64 with RTE and SignedZeroInfNanPreserve for exact underflow repair"));
        }
        self.b.capability(spv::Capability::Float64);
        // require_float already declared these capabilities/extension for 32-bit.
        self.b
            .execution_mode(self.entry, spv::ExecutionMode::RoundingModeRTE, [64]);
        self.b.execution_mode(
            self.entry,
            spv::ExecutionMode::SignedZeroInfNanPreserve,
            [64],
        );
        let ty = self.b.type_float(64, None);
        self.repair_float = Some(ty);
        Ok(ty)
    }

    fn exact_step(&mut self, value: u32) -> u32 {
        self.b.decorate(value, spv::Decoration::NoContraction, []);
        value
    }

    pub(super) fn repair_product_underflow(
        &mut self,
        value: u32,
        left: u32,
        right: u32,
        addend: Option<u32>,
    ) -> Result<u32> {
        let wide = self.require_repair_float()?;
        let bits = self.b.bitcast(self.uint, None, value)?;
        let mut repair = self.tiny(bits)?;
        let a = self.b.bitcast(self.uint, None, left)?;
        let b = self.b.bitcast(self.uint, None, right)?;
        let a = self.masked(a, 0x7f80_0000)?;
        let b = self.masked(b, 0x7f80_0000)?;
        // Exact zero products already have native signed-zero semantics. Do not
        // send common zero coordinates through wide arithmetic.
        let zero = self.constant(0);
        let a_nonzero = self.b.i_not_equal(self.boolean, None, a, zero)?;
        let b_nonzero = self.b.i_not_equal(self.boolean, None, b, zero)?;
        let nonzero = self
            .b
            .logical_and(self.boolean, None, a_nonzero, b_nonzero)?;
        repair = self.b.logical_and(self.boolean, None, repair, nonzero)?;
        if addend.is_some() {
            // With biased operand exponents ea, eb the exact product's quantum
            // is 2^(ea+eb-300). If ea+eb >= 175, neither a nonzero product nor
            // cancellation with a float32 addend can be subnormal. This keeps
            // ordinary exact cancellation (including geometry zeros) off FP64.
            let sum = self.b.i_add(self.uint, None, a, b)?;
            let bound = self.constant(175 << 23);
            let small = self.b.u_less_than(self.boolean, None, sum, bound)?;
            repair = self.b.logical_and(self.boolean, None, repair, small)?;
        }
        let header = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        self.b
            .selection_merge(merge, spv::SelectionControl::DONT_FLATTEN)?;
        self.b.branch_conditional(repair, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;

        let a = self.b.f_convert(wide, None, left)?;
        let b = self.b.f_convert(wide, None, right)?;
        let product = self.b.f_mul(wide, None, a, b)?;
        self.exact_step(product);
        let (sum, residual) = if let Some(addend) = addend {
            let c = self.b.f_convert(wide, None, addend)?;
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
        // Extract the sign without requiring shaderInt64, including negative zero.
        let pair = self.b.type_vector(self.uint, 2);
        let words = self.b.bitcast(pair, None, sum)?;
        let high = self.b.composite_extract(self.uint, None, words, [1])?;
        let sign = self.masked(high, 0x8000_0000)?;
        let zero = self.constant(0);
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let negated = self.b.f_negate(wide, None, sum)?;
        let magnitude = self.b.select(wide, None, negative, negated, sum)?;
        // Halfway from largest subnormal to MIN_POSITIVE. Ties round UP (even).
        // Exactly representable in binary64; do not convert a rounded sum to f32.
        let midpoint = self.b.constant_bit64(wide, 0x380f_ffff_e000_0000);
        let mut normal =
            self.b
                .f_ord_greater_than_equal(self.boolean, None, magnitude, midpoint)?;
        if let Some(residual) = residual {
            let negated = self.b.f_negate(wide, None, residual)?;
            let residual = self.b.select(wide, None, negative, negated, residual)?;
            let wide_zero = self.b.constant_bit64(wide, 0);
            let below = self
                .b
                .f_ord_less_than(self.boolean, None, residual, wide_zero)?;
            let tie = self
                .b
                .f_ord_equal(self.boolean, None, magnitude, midpoint)?;
            let below_tie = self.b.logical_and(self.boolean, None, tie, below)?;
            let not_below = self.b.logical_not(self.boolean, None, below_tie)?;
            normal = self.b.logical_and(self.boolean, None, normal, not_below)?;
        }
        let minimum = self.constant(0x0080_0000);
        let magnitude = self.b.select(self.uint, None, normal, minimum, zero)?;
        let repaired = self.b.bitwise_or(self.uint, None, sign, magnitude)?;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        let result = self
            .b
            .phi(self.uint, None, [(bits, header), (repaired, body)])?;
        Ok(self.b.bitcast(self.float, None, result)?)
    }
}
