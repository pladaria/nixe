//! Typed binary16 values remain native scalars until a word ABI consumes them.
//! Hosts without preserving Float16 use low-lane integer bits for this logical
//! type and the exact conversion/arithmetic lowerings; no 16-bit storage ABI.
use super::*;

impl Emitter<'_> {
    pub(super) fn half_type(&mut self) -> Result<u32, Error> {
        if self.options.shader.native_half_conversions {
            self.math
                .native_half_type()
                .map_err(|e| self.error(e.to_string()))
        } else {
            Ok(self.math.uint)
        }
    }
    pub(super) fn half_from_bits(&mut self, bits: u32) -> Result<u32, Error> {
        if !self.options.shader.native_half_conversions {
            return Ok(bits);
        }
        self.math
            .half_value(bits)
            .map_err(|e| self.error(e.to_string()))
    }
    pub(super) fn half_pack(&mut self, value: u32) -> Result<u32, Error> {
        if !self.options.shader.native_half_conversions {
            let mask = self.math.constant(0xffff);
            return self
                .math
                .b
                .bitwise_and(self.math.uint, None, value, mask)
                .map_err(|e| self.error(e.to_string()));
        }
        self.math
            .half_bits(value)
            .map_err(|e| self.error(e.to_string()))
    }

    pub(super) fn half_unpack(&mut self, bits: u32) -> Result<u32, Error> {
        let mask = self.math.constant(0xffff);
        let bits = self
            .math
            .b
            .bitwise_and(self.math.uint, None, bits, mask)
            .map_err(|e| self.error(e.to_string()))?;
        let magnitude = self.math.constant(0x7fff);
        let magnitude = self
            .math
            .b
            .bitwise_and(self.math.uint, None, bits, magnitude)
            .map_err(|e| self.error(e.to_string()))?;
        let infinity = self.math.constant(0x7c00);
        let nan = self
            .math
            .b
            .u_greater_than(self.math.boolean, None, magnitude, infinity)
            .map_err(|e| self.error(e.to_string()))?;
        let quiet = self.math.constant(0x200);
        let quiet = self
            .math
            .b
            .bitwise_or(self.math.uint, None, bits, quiet)
            .map_err(|e| self.error(e.to_string()))?;
        let bits = self
            .math
            .b
            .select(self.math.uint, None, nan, quiet, bits)
            .map_err(|e| self.error(e.to_string()))?;
        self.half_from_bits(bits)
    }
    pub(super) fn half_narrow(&mut self, value: u32) -> Result<u32, Error> {
        let bits = self
            .math
            .b
            .bitcast(self.math.uint, None, value)
            .map_err(|e| self.error(e.to_string()))?;
        if self.options.shader.native_half_conversions {
            self.math
                .narrow_half_native(bits)
                .map_err(|e| self.error(e.to_string()))
        } else {
            self.math
                .pack_half(bits)
                .map_err(|e| self.error(e.to_string()))
        }
    }
    pub(super) fn half_widen(&mut self, value: u32) -> Result<u32, Error> {
        let bits = self.half_pack(value)?;
        let bits = if self.options.shader.native_half_conversions {
            self.math.widen_half_native(value, bits)
        } else {
            self.math.unpack_half(bits, false)
        }
        .map_err(|e| self.error(e.to_string()))?;
        self.math
            .b
            .bitcast(self.math.float, None, bits)
            .map_err(|e| self.error(e.to_string()))
    }
    pub(super) fn half_fma(&mut self, args: [u32; 3], saturate: bool) -> Result<u32, Error> {
        if self.options.shader.native_half_conversions && self.options.shader.native_half_fma {
            let half = self.half_type()?;
            self.math.enable_fma();
            let result = self
                .math
                .b
                .fma_khr(half, None, args[0], args[1], args[2])
                .map_err(|e| self.error(e.to_string()))?;
            self.math
                .b
                .decorate(result, spv::Decoration::NoContraction, []);
            return self.half_saturate(result, saturate);
        }
        let mut wide = [0; 3];
        for (output, input) in wide.iter_mut().zip(args) {
            let value = self.half_widen(input)?;
            *output = self
                .math
                .b
                .bitcast(self.math.uint, None, value)
                .map_err(|e| self.error(e.to_string()))?;
        }
        let packed = self
            .math
            .half_fma(wide[0], wide[1], wide[2], saturate)
            .map_err(|e| self.error(e.to_string()))?;
        self.half_from_bits(packed)
    }
    pub(super) fn half_binary(
        &mut self,
        op: Binary,
        a: u32,
        b: u32,
        control: ShaderFloatControl,
    ) -> Result<u32, Error> {
        if !matches!(op, Binary::Add | Binary::Multiply)
            || control.rounding != ShaderRoundingMode::NearestEven
            || control.denormals_are_zero
            || control.flush_denormals_to_zero
            || control.nan_mode != crate::ShaderNanMode::Propagate
        {
            return Err(self.error("binary16 arithmetic requires preserving RTE add/multiply"));
        }
        if self.options.shader.native_half_conversions {
            let half = self.half_type()?;
            let value = if op == Binary::Add {
                self.math.b.f_add(half, None, a, b)
            } else {
                self.math.b.f_mul(half, None, a, b)
            }
            .map_err(|e| self.error(e.to_string()))?;
            self.math
                .b
                .decorate(value, spv::Decoration::NoContraction, []);
            return self.half_saturate(value, control.saturate);
        }
        let a = self.half_widen(a)?;
        let b = self.half_widen(b)?;
        let a = self
            .math
            .b
            .bitcast(self.math.uint, None, a)
            .map_err(|e| self.error(e.to_string()))?;
        let b = self
            .math
            .b
            .bitcast(self.math.uint, None, b)
            .map_err(|e| self.error(e.to_string()))?;
        let bits = self
            .math
            .float_binary_known_normal(a, b, control, op == Binary::Multiply, true)
            .map_err(|e| self.error(e.to_string()))?;
        self.math
            .pack_half(bits)
            .map_err(|e| self.error(e.to_string()))
    }
    fn half_saturate(&mut self, value: u32, saturate: bool) -> Result<u32, Error> {
        if !saturate {
            return Ok(value);
        }
        let bits = self.half_pack(value)?;
        let magnitude = self.math.constant(0x7fff);
        let magnitude = self
            .math
            .b
            .bitwise_and(self.math.uint, None, bits, magnitude)
            .map_err(|e| self.error(e.to_string()))?;
        let infinity = self.math.constant(0x7c00);
        let nan = self
            .math
            .b
            .u_greater_than(self.math.boolean, None, magnitude, infinity)
            .map_err(|e| self.error(e.to_string()))?;
        let sign = self.math.constant(0x8000);
        let negative = self
            .math
            .b
            .u_greater_than_equal(self.math.boolean, None, bits, sign)
            .map_err(|e| self.error(e.to_string()))?;
        let clear = self
            .math
            .b
            .logical_or(self.math.boolean, None, nan, negative)
            .map_err(|e| self.error(e.to_string()))?;
        let one = self.math.constant(0x3c00);
        let above = self
            .math
            .b
            .u_greater_than(self.math.boolean, None, magnitude, one)
            .map_err(|e| self.error(e.to_string()))?;
        let bits = self
            .math
            .b
            .select(self.math.uint, None, above, one, bits)
            .map_err(|e| self.error(e.to_string()))?;
        let zero = self.math.constant(0);
        let bits = self
            .math
            .b
            .select(self.math.uint, None, clear, zero, bits)
            .map_err(|e| self.error(e.to_string()))?;
        self.half_from_bits(bits)
    }
}
