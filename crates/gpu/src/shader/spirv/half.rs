//! Binary16 conversions use native Float16 only with explicit preservation and
//! rounding guarantees. Other hosts use exact integer conversion. NaN payloads
//! and quieting are reconstructed by bits on both host capability paths.
use super::*;

impl Math {
    pub(crate) fn unpack_half(&mut self, packed: u32, high: bool) -> Result<u32> {
        if self.options.native_half_conversions {
            return self.unpack_half_native(packed, high);
        }
        let sixteen = self.constant(16);
        let bits = if high {
            self.b
                .shift_right_logical(self.uint, None, packed, sixteen)?
        } else {
            packed
        };
        let sign_mask = self.constant(0x8000);
        let sign = self.b.bitwise_and(self.uint, None, bits, sign_mask)?;
        let sign = self.b.shift_left_logical(self.uint, None, sign, sixteen)?;
        let ten = self.constant(10);
        let exponent = self.b.shift_right_logical(self.uint, None, bits, ten)?;
        let mask = self.constant(31);
        let exponent = self.b.bitwise_and(self.uint, None, exponent, mask)?;
        let fraction_mask = self.constant(1023);
        let fraction = self.b.bitwise_and(self.uint, None, bits, fraction_mask)?;
        let zero = self.constant(0);
        let subnormal = self.b.i_equal(self.boolean, None, exponent, zero)?;
        let nonfinite = self.b.i_equal(self.boolean, None, exponent, mask)?;
        let nan = self.b.i_not_equal(self.boolean, None, fraction, zero)?;
        let thirteen = self.constant(13);
        let mantissa = self
            .b
            .shift_left_logical(self.uint, None, fraction, thirteen)?;
        let bias = self.constant(112);
        let normal_exponent = self.b.i_add(self.uint, None, exponent, bias)?;
        let twenty_three = self.constant(23);
        let normal_exponent =
            self.b
                .shift_left_logical(self.uint, None, normal_exponent, twenty_three)?;
        let normal = self
            .b
            .bitwise_or(self.uint, None, normal_exponent, mantissa)?;
        let special_exponent = self.constant(0x7f800000);
        let special = self
            .b
            .bitwise_or(self.uint, None, special_exponent, mantissa)?;
        let quiet_mask = self.constant(0x00400000);
        let quiet = self.b.select(self.uint, None, nan, quiet_mask, zero)?;
        let special = self.b.bitwise_or(self.uint, None, special, quiet)?;
        // Every binary16 subnormal becomes a normal binary32 value. The integer
        // conversion and power-of-two multiplication are both exact.
        let fraction_float = self.b.convert_u_to_f(self.float, None, fraction)?;
        let scale = self.b.constant_bit32(self.float, 0x3380_0000);
        let tiny = self.b.f_mul(self.float, None, fraction_float, scale)?;
        let tiny = self.b.bitcast(self.uint, None, tiny)?;
        let value = self.b.select(self.uint, None, nonfinite, special, normal)?;
        let value = self.b.select(self.uint, None, subnormal, tiny, value)?;
        Ok(self.b.bitwise_or(self.uint, None, value, sign)?)
    }

    pub(crate) fn pack_half(&mut self, bits: u32) -> Result<u32> {
        if self.options.native_half_conversions {
            return self.pack_half_native(bits);
        }
        let zero = self.constant(0);
        let one = self.constant(1);
        let thirteen = self.constant(13);
        let sixteen = self.constant(16);
        let twenty_three = self.constant(23);
        let sign = self.b.shift_right_logical(self.uint, None, bits, sixteen)?;
        let sign_mask = self.constant(0x8000);
        let sign = self.b.bitwise_and(self.uint, None, sign, sign_mask)?;
        let abs_mask = self.constant(0x7fffffff);
        let magnitude = self.b.bitwise_and(self.uint, None, bits, abs_mask)?;
        let infinity = self.constant(0x7f800000);
        let nan = self
            .b
            .u_greater_than(self.boolean, None, magnitude, infinity)?;
        let overflow_limit = self.constant(0x477ff000);
        let overflow =
            self.b
                .u_greater_than_equal(self.boolean, None, magnitude, overflow_limit)?;
        let zero_limit = self.constant(0x33000000);
        let underflow = self
            .b
            .u_less_than(self.boolean, None, magnitude, zero_limit)?;
        let normal_limit = self.constant(0x38800000);
        let subnormal = self
            .b
            .u_less_than(self.boolean, None, magnitude, normal_limit)?;
        let retained = self
            .b
            .shift_right_logical(self.uint, None, magnitude, thirteen)?;
        let low_bit = self.b.bitwise_and(self.uint, None, retained, one)?;
        let round_mask = self.constant(0xfff);
        let round = self.b.i_add(self.uint, None, round_mask, low_bit)?;
        let normal = self.b.i_add(self.uint, None, magnitude, round)?;
        let normal = self
            .b
            .shift_right_logical(self.uint, None, normal, thirteen)?;
        let bias = self.constant(0x1c000);
        let normal = self.b.i_sub(self.uint, None, normal, bias)?;
        let exponent = self
            .b
            .shift_right_logical(self.uint, None, magnitude, twenty_three)?;
        let bias = self.constant(126);
        let shift = self.b.i_sub(self.uint, None, bias, exponent)?;
        let max_shift = self.constant(24);
        let too_large = self
            .b
            .u_greater_than(self.boolean, None, shift, max_shift)?;
        let shift = self
            .b
            .select(self.uint, None, too_large, max_shift, shift)?;
        let too_small = self.b.u_less_than(self.boolean, None, shift, one)?;
        let shift = self.b.select(self.uint, None, too_small, one, shift)?;
        let fraction_mask = self.constant(0x007fffff);
        let fraction = self
            .b
            .bitwise_and(self.uint, None, magnitude, fraction_mask)?;
        let hidden = self.constant(0x00800000);
        let fraction = self.b.bitwise_or(self.uint, None, fraction, hidden)?;
        let truncated = self
            .b
            .shift_right_logical(self.uint, None, fraction, shift)?;
        let remainder_mask = self.b.shift_left_logical(self.uint, None, one, shift)?;
        let remainder_mask = self.b.i_sub(self.uint, None, remainder_mask, one)?;
        let remainder = self
            .b
            .bitwise_and(self.uint, None, fraction, remainder_mask)?;
        let midpoint_shift = self.b.i_sub(self.uint, None, shift, one)?;
        let midpoint = self
            .b
            .shift_left_logical(self.uint, None, one, midpoint_shift)?;
        let above = self
            .b
            .u_greater_than(self.boolean, None, remainder, midpoint)?;
        let equal = self.b.i_equal(self.boolean, None, remainder, midpoint)?;
        let odd = self.b.bitwise_and(self.uint, None, truncated, one)?;
        let odd = self.b.i_not_equal(self.boolean, None, odd, zero)?;
        let tie = self.b.logical_and(self.boolean, None, equal, odd)?;
        let increment = self.b.logical_or(self.boolean, None, above, tie)?;
        let increment = self.b.select(self.uint, None, increment, one, zero)?;
        let tiny = self.b.i_add(self.uint, None, truncated, increment)?;
        let value = self.b.select(self.uint, None, subnormal, tiny, normal)?;
        let value = self.b.select(self.uint, None, underflow, zero, value)?;
        let half_infinity = self.constant(0x7c00);
        let value = self
            .b
            .select(self.uint, None, overflow, half_infinity, value)?;
        let nan_mask = self.constant(1023);
        let payload = self.b.bitwise_and(self.uint, None, retained, nan_mask)?;
        let quiet_nan = self.constant(0x7e00);
        let payload = self.b.bitwise_or(self.uint, None, payload, quiet_nan)?;
        let value = self.b.select(self.uint, None, nan, payload, value)?;
        Ok(self.b.bitwise_or(self.uint, None, value, sign)?)
    }
}

impl Math {
    // OpFConvert obeys the destination width's RTE and preservation modes.
    // https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision
    pub(crate) fn native_half_type(&mut self) -> Result<u32> {
        if let Some(half) = self.native_half {
            return Ok(half);
        }
        self.b.extension("SPV_KHR_float_controls");
        self.b.capability(spv::Capability::RoundingModeRTE);
        self.b.capability(spv::Capability::SignedZeroInfNanPreserve);
        self.b.capability(spv::Capability::Float16);
        self.b.capability(spv::Capability::DenormPreserve);
        for mode in [
            spv::ExecutionMode::DenormPreserve,
            spv::ExecutionMode::RoundingModeRTE,
            spv::ExecutionMode::SignedZeroInfNanPreserve,
        ] {
            self.b.execution_mode(self.entry, mode, [16]);
        }
        let half = self.b.type_float(16, None);
        self.native_half = Some(half);
        Ok(half)
    }
    pub(crate) fn half_value(&mut self, bits: u32) -> Result<u32> {
        let half = self.native_half_type()?;
        let pair = self.b.type_vector(half, 2);
        let pair = self.b.bitcast(pair, None, bits)?;
        Ok(self.b.composite_extract(half, None, pair, [0])?)
    }
    pub(crate) fn half_bits(&mut self, value: u32) -> Result<u32> {
        let half = self.native_half_type()?;
        let zero = self.b.constant_bit32(half, 0);
        let pair = self.b.type_vector(half, 2);
        let pair = self.b.composite_construct(pair, None, [value, zero])?;
        Ok(self.b.bitcast(self.uint, None, pair)?)
    }
    fn unpack_half_native(&mut self, packed: u32, high: bool) -> Result<u32> {
        let bits = if high {
            let shift = self.constant(16);
            self.b.shift_right_logical(self.uint, None, packed, shift)?
        } else {
            packed
        };
        let value = self.half_value(bits)?;
        self.widen_half_native(value, bits)
    }
    pub(crate) fn widen_half_native(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.require_repaired_float(ShaderFloatControl::PRECISE)?;
        let value = self.b.f_convert(self.float, None, value)?;
        let native = self.b.bitcast(self.uint, None, value)?;
        let magnitude = self.masked(bits, 0x7fff)?;
        let infinity = self.constant(0x7c00);
        let nan = self
            .b
            .u_greater_than(self.boolean, None, magnitude, infinity)?;
        let sign = self.masked(bits, 0x8000)?;
        let shift = self.constant(16);
        let sign = self.b.shift_left_logical(self.uint, None, sign, shift)?;
        let payload = self.masked(bits, 0x3ff)?;
        let shift = self.constant(13);
        let payload = self.b.shift_left_logical(self.uint, None, payload, shift)?;
        let quiet = self.constant(0x7fc00000);
        let payload = self.b.bitwise_or(self.uint, None, payload, quiet)?;
        let payload = self.b.bitwise_or(self.uint, None, payload, sign)?;
        Ok(self.b.select(self.uint, None, nan, payload, native)?)
    }
    fn pack_half_native(&mut self, bits: u32) -> Result<u32> {
        let value = self.narrow_half_native(bits)?;
        self.half_bits(value)
    }
    pub(crate) fn narrow_half_native(&mut self, bits: u32) -> Result<u32> {
        let half = self.native_half_type()?;
        let value = self.b.bitcast(self.float, None, bits)?;
        let native = self.b.f_convert(half, None, value)?;
        let magnitude = self.masked(bits, 0x7fffffff)?;
        let infinity = self.constant(0x7f800000);
        let nan = self
            .b
            .u_greater_than(self.boolean, None, magnitude, infinity)?;
        let shift = self.constant(13);
        let payload = self.b.shift_right_logical(self.uint, None, bits, shift)?;
        let payload = self.masked(payload, 0x3ff)?;
        let quiet = self.constant(0x7e00);
        let payload = self.b.bitwise_or(self.uint, None, payload, quiet)?;
        let shift = self.constant(16);
        let sign = self.b.shift_right_logical(self.uint, None, bits, shift)?;
        let sign = self.masked(sign, 0x8000)?;
        let payload = self.b.bitwise_or(self.uint, None, payload, sign)?;
        let payload = self.half_value(payload)?;
        Ok(self.b.select(half, None, nan, payload, native)?)
    }
}

impl Math {
    /// The caller proves all three FP32 carriers represent binary16 values.
    /// Promoted/mixed-precision arithmetic must keep its FP32 FMA lowering.
    pub(crate) fn half_fma_native(&mut self, args: [u32; 3], saturate: bool) -> Result<u32> {
        let half = self.native_half_type()?;
        self.enable_fma();
        let mut inputs = [0; 3];
        for (result, bits) in inputs.iter_mut().zip(args) {
            let value = self.b.bitcast(self.float, None, bits)?;
            *result = self.b.f_convert(half, None, value)?;
        }
        let value = self
            .b
            .fma_khr(half, None, inputs[0], inputs[1], inputs[2])?;
        self.b.decorate(value, spv::Decoration::NoContraction, []);
        let zero_half = self.b.constant_bit32(half, 0);
        let pair = self.b.type_vector(half, 2);
        let pair = self.b.composite_construct(pair, None, [value, zero_half])?;
        let bits = self.b.bitcast(self.uint, None, pair)?;
        if !saturate {
            return Ok(bits);
        }
        let magnitude = self.masked(bits, 0x7fff)?;
        let infinity = self.constant(0x7c00);
        let nan = self
            .b
            .u_greater_than(self.boolean, None, magnitude, infinity)?;
        let sign = self.masked(bits, 0x8000)?;
        let zero = self.constant(0);
        let negative = self.b.i_not_equal(self.boolean, None, sign, zero)?;
        let clear = self.b.logical_or(self.boolean, None, negative, nan)?;
        let one = self.constant(0x3c00);
        let above = self.b.u_greater_than(self.boolean, None, magnitude, one)?;
        let value = self.b.select(self.uint, None, above, one, bits)?;
        Ok(self.b.select(self.uint, None, clear, zero, value)?)
    }
}
