//! Shared native FP predicates and value lowering, independent of execution boundaries.

use crate::jit_error::Error;
use crate::simd_lowering::{SimdLowering, bitcast_flags};
use cranelift_codegen::ir::{
    InstBuilder, Value,
    condcodes::{FloatCC, IntCC},
    types,
};

pub(crate) trait FpLowering<'a>: SimdLowering<'a> {
    /// Advanced SIMD SCVTF/UCVTF in an active guest FP environment. Inactive
    /// lanes are zeroed before conversion so they cannot contribute IXC.
    /// Single-element encodings use scalar CLIF instead of converting unused
    /// lanes and repacking them, particularly expensive for baseline x86 i64.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--vector---Signed-integer-Convert-to-Floating-point--vector--
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--vector---Unsigned-integer-Convert-to-Floating-point--vector--
    fn vector_integer_to_fp_value(
        &mut self,
        value: Value,
        lane_64: bool,
        vector_bits: u32,
        signed: bool,
    ) -> Value {
        let lane_bits = if lane_64 { 64 } else { 32 };
        if vector_bits == lane_bits {
            let bits = self.vector_as(value, types::I128);
            let integer_ty = if lane_64 { types::I64 } else { types::I32 };
            let bits = self.builder().ins().ireduce(types::I64, bits);
            let result = self.integer_to_fp_value(bits, lane_64, lane_64, signed);
            let bits = self
                .builder()
                .ins()
                .bitcast(integer_ty, bitcast_flags(), result);
            let bits = self.builder().ins().uextend(types::I128, bits);
            return self.vector_as(bits, types::I8X16);
        }
        let value = self.mask_vector(value, vector_bits);
        let value = self.vector_as(value, if lane_64 { types::I64X2 } else { types::I32X4 });
        let float_ty = if lane_64 { types::F64X2 } else { types::F32X4 };
        let result = if signed {
            self.builder().ins().fcvt_from_sint(float_ty, value)
        } else {
            self.builder().ins().fcvt_from_uint(float_ty, value)
        };
        let result = self.vector_as(result, types::I8X16);
        self.mask_vector(result, vector_bits)
    }

    /// Native FCVTZS/FCVTZU domain, checked without touching host FP status.
    /// Admit normal in-range inputs and either signed zero. The signed minimum
    /// is inclusive; positive bounds are exclusive powers of two. Negative
    /// unsigned inputs and just-below-minimum fractions use exact completion.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-toward-Zero--scalar--
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-toward-Zero--scalar--
    fn fp_to_integer_domain(
        &mut self,
        bits: Value,
        width: u32,
        destination_64: bool,
        signed: bool,
    ) -> Value {
        let (fraction, bias) = if width == 32 { (23, 127) } else { (52, 1023) };
        let integer_bits = if destination_64 { 64 } else { 32 };
        let limit = (bias + integer_bits - u64::from(signed)) << fraction;
        let sign = 1u64 << (width - 1);
        let magnitude = self.builder().ins().band_imm_u(bits, (sign - 1) as i64);
        let ordered = if signed { magnitude } else { bits };
        let normal = self.builder().ins().icmp_imm_u(
            IntCC::UnsignedGreaterThanOrEqual,
            ordered,
            1i64 << fraction,
        );
        let below = self
            .builder()
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThan, ordered, limit as i64);
        let in_range = self.builder().ins().band(normal, below);
        let zero = self.builder().ins().icmp_imm_s(IntCC::Equal, magnitude, 0);
        let mut direct = self.builder().ins().bor(in_range, zero);
        if signed {
            let minimum =
                self.builder()
                    .ins()
                    .icmp_imm_u(IntCC::Equal, bits, (sign | limit) as i64);
            direct = self.builder().ins().bor(direct, minimum);
        }
        direct
    }

    /// Convert only after the domain/FPCR guard and guest FP activation. The
    /// backend's saturating sequence is used inside its ordinary valid domain;
    /// out-of-domain guest saturation/status/traps belong to exact completion.
    /// W destinations clear their upper half, including signed conversions.
    fn fp_to_integer_value(
        &mut self,
        bits: Value,
        source_64: bool,
        destination_64: bool,
        signed: bool,
    ) -> Value {
        let float_ty = if source_64 { types::F64 } else { types::F32 };
        let integer_ty = if destination_64 {
            types::I64
        } else {
            types::I32
        };
        let value = self
            .builder()
            .ins()
            .bitcast(float_ty, bitcast_flags(), bits);
        let result = if signed {
            self.builder().ins().fcvt_to_sint_sat(integer_ty, value)
        } else {
            self.builder().ins().fcvt_to_uint_sat(integer_ty, value)
        };
        if destination_64 {
            result
        } else {
            self.builder().ins().uextend(types::I64, result)
        }
    }

    /// W/X integer to S/D in an active guest FP segment. W sources discard
    /// their upper bits before signedness is interpreted; only the final
    /// conversion rounds and contributes inexact status.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar--integer---Signed-integer-Convert-to-Floating-point--scalar--
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--scalar--integer---Unsigned-integer-Convert-to-Floating-point--scalar--
    fn integer_to_fp_value(
        &mut self,
        value: Value,
        source_64: bool,
        destination_64: bool,
        signed: bool,
    ) -> Value {
        let value = if source_64 {
            value
        } else {
            self.builder().ins().ireduce(types::I32, value)
        };
        let ty = if destination_64 {
            types::F64
        } else {
            types::F32
        };
        if signed {
            self.builder().ins().fcvt_from_sint(ty, value)
        } else {
            self.builder().ins().fcvt_from_uint(ty, value)
        }
    }

    /// x86 FMA tininess guard, with normal/zero operands checked separately.
    /// A zero product leaves the normal/zero addend. With a zero addend require
    /// a non-tiny product. Otherwise require both exact product and addend to
    /// lie on the minimum-normal lattice: even cancellation then cannot be
    /// subnormal. This conservative exponent-only bound excludes rare small
    /// domains without computing or rounding an intermediate FP product.
    fn fp_fused_domain(&mut self, first: Value, second: Value, third: Value, width: u32) -> Value {
        let mask = ((1u64 << (width - 1)) - 1) as i64;
        let a = self.builder().ins().band_imm_u(first, mask);
        let b = self.builder().ins().band_imm_u(second, mask);
        let c = self.builder().ins().band_imm_u(third, mask);
        let az = self.builder().ins().icmp_imm_s(IntCC::Equal, a, 0);
        let bz = self.builder().ins().icmp_imm_s(IntCC::Equal, b, 0);
        let cz = self.builder().ins().icmp_imm_s(IntCC::Equal, c, 0);
        let product_zero = self.builder().ins().bor(az, bz);
        let (fraction, bias) = if width == 32 { (23, 127) } else { (52, 1023) };
        let ae = self.builder().ins().ushr_imm_u(a, fraction);
        let be = self.builder().ins().ushr_imm_u(b, fraction);
        let ce = self.builder().ins().ushr_imm_u(c, fraction);
        let sum = self.builder().ins().iadd(ae, be);
        let product_normal =
            self.builder()
                .ins()
                .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, sum, bias + 1);
        let no_addend = self.builder().ins().band(cz, product_normal);
        // Product precision is 2*(fraction+1); addend precision is fraction+1.
        let product_lattice = self.builder().ins().icmp_imm_u(
            IntCC::UnsignedGreaterThanOrEqual,
            sum,
            bias + 2 * fraction + 1,
        );
        let addend_lattice =
            self.builder()
                .ins()
                .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, ce, fraction + 1);
        let lattice = self.builder().ins().band(product_lattice, addend_lattice);
        let safe = self.builder().ins().bor(lattice, no_addend);
        self.builder().ins().bor(safe, product_zero)
    }

    /// Apply architectural input sign changes before one fused rounding.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMADD--Floating-point-fused-Multiply-Add-
    fn fp_fused_value(
        &mut self,
        first: Value,
        second: Value,
        third: Value,
        operation: nixe_cpu::decode::a64::fp_simd::FloatFusedMultiplyOperation,
    ) -> Value {
        use nixe_cpu::decode::a64::fp_simd::FloatFusedMultiplyOperation as Op;
        let first = if matches!(operation, Op::MultiplySubtract | Op::NegatedMultiplyAdd) {
            self.builder().ins().fneg(first)
        } else {
            first
        };
        let third = if matches!(
            operation,
            Op::NegatedMultiplyAdd | Op::NegatedMultiplySubtract
        ) {
            self.builder().ins().fneg(third)
        } else {
            third
        };
        self.builder().ins().fma(first, second, third)
    }

    /// Normal/zero operands are checked separately. On x86, conservatively
    /// exclude possible tiny products before status production: a normal
    /// product has exponent e1+e2 or e1+e2+1, and the biased exponent sum
    /// being at least bias+1 guarantees it is not tiny. Either zero is safe.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMUL--scalar---Floating-point-Multiply--scalar--
    fn fp_multiply_domain(&mut self, first: Value, second: Value, width: u32) -> Value {
        let mask = ((1u64 << (width - 1)) - 1) as i64;
        let first = self.builder().ins().band_imm_u(first, mask);
        let second = self.builder().ins().band_imm_u(second, mask);
        let first_zero = self.builder().ins().icmp_imm_s(IntCC::Equal, first, 0);
        let second_zero = self.builder().ins().icmp_imm_s(IntCC::Equal, second, 0);
        let zero = self.builder().ins().bor(first_zero, second_zero);
        let (fraction, threshold) = if width == 32 { (23, 128) } else { (52, 1024) };
        let first = self.builder().ins().ushr_imm_u(first, fraction);
        let second = self.builder().ins().ushr_imm_u(second, fraction);
        let sum = self.builder().ins().iadd(first, second);
        let non_tiny =
            self.builder()
                .ins()
                .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, sum, threshold);
        self.builder().ins().bor(zero, non_tiny)
    }

    /// FNMUL negates the rounded product, not an input: moving the negation
    /// across FMUL would change directed rounding and NaN sign semantics.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMUL--Floating-point-Negated-Multiply--scalar--
    fn fp_multiply_value(
        &mut self,
        first: Value,
        second: Value,
        operation: nixe_cpu::decode::a64::fp_simd::FloatMultiplyOperation,
    ) -> Value {
        let result = self.builder().ins().fmul(first, second);
        if matches!(
            operation,
            nixe_cpu::decode::a64::fp_simd::FloatMultiplyOperation::NegatedMultiply
        ) {
            self.builder().ins().fneg(result)
        } else {
            result
        }
    }

    /// Combine with finite/normal-or-zero operand guards. A zero divisor uses
    /// exact completion. On x86, also exclude possible tiny quotients before
    /// producing status: Arm uses pre-rounding tininess and FZ raises UFC only.
    /// For normal operands the quotient exponent is e1-e2 or e1-e2-1, so
    /// e1-e2 >= Emin+1 guarantees a non-tiny result. Zero numerators are safe.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--Floating-point-Divide--scalar--
    fn fp_divide_domain(
        &mut self,
        first: Value,
        second: Value,
        width: u32,
        abi: crate::abi::HostAbi,
    ) -> Value {
        let magnitude_mask = (1u64 << (width - 1)) - 1;
        let second = self
            .builder()
            .ins()
            .band_imm_u(second, magnitude_mask as i64);
        let nonzero = self.builder().ins().icmp_imm_s(IntCC::NotEqual, second, 0);
        if abi == crate::abi::HostAbi::Aarch64 {
            return nonzero;
        }
        let first = self
            .builder()
            .ins()
            .band_imm_u(first, magnitude_mask as i64);
        let zero = self.builder().ins().icmp_imm_s(IntCC::Equal, first, 0);
        let (fraction_bits, minimum_difference) =
            if width == 32 { (23, -125) } else { (52, -1021) };
        let first_exponent = self.builder().ins().ushr_imm_u(first, fraction_bits);
        let second_exponent = self.builder().ins().ushr_imm_u(second, fraction_bits);
        let difference = self.builder().ins().isub(first_exponent, second_exponent);
        let non_tiny = self.builder().ins().icmp_imm_s(
            IntCC::SignedGreaterThanOrEqual,
            difference,
            minimum_difference,
        );
        let safe = self.builder().ins().bor(zero, non_tiny);
        self.builder().ins().band(nonzero, safe)
    }

    /// Eligible scalar division inside the active guest FP segment.
    fn fp_divide_value(&mut self, first: Value, second: Value) -> Value {
        self.builder().ins().fdiv(first, second)
    }

    /// Nonnegative inputs and signed zero; combine with finite/normal guards.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSQRT--Floating-point-Square-Root--scalar--
    fn fp_sqrt_domain(&mut self, bits: Value, width: u32) -> Value {
        let sign_mask = 1u64 << (width - 1);
        let sign = self.builder().ins().band_imm_u(bits, sign_mask as i64);
        let positive = self.builder().ins().icmp_imm_s(IntCC::Equal, sign, 0);
        let magnitude = self
            .builder()
            .ins()
            .band_imm_u(bits, (sign_mask - 1) as i64);
        let zero = self.builder().ins().icmp_imm_s(IntCC::Equal, magnitude, 0);
        self.builder().ins().bor(positive, zero)
    }

    /// Arm detects tiny results before rounding. x86 may round a tiny D->S
    /// result up to the minimum normal without UFC, or add IXC when FTZ flushes
    /// an exact tiny result. Keep that narrow input domain on the exact edge.
    fn fp_demote_domain(&mut self, bits: Value) -> Value {
        let magnitude = self.builder().ins().band_imm_u(bits, 0x7fff_ffff_ffff_ffff);
        let normal = self.builder().ins().icmp_imm_u(
            IntCC::UnsignedGreaterThanOrEqual,
            magnitude,
            (897u64 << 52) as i64,
        );
        let zero = self.builder().ins().icmp_imm_s(IntCC::Equal, magnitude, 0);
        self.builder().ins().bor(normal, zero)
    }

    /// Result-only CLIF for the typed unary operations, run inside an eligible
    /// native FP segment. Exception/status policy belongs to the input guards.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVT--Floating-point-Convert-precision--scalar--
    fn fp_unary_value(&mut self, input: Value, kind: crate::abi::FpUnaryKind) -> Value {
        use crate::abi::FpUnaryKind;
        use nixe_cpu::decode::a64::fp_simd::FloatConversion;
        match kind {
            FpUnaryKind::SquareRoot { .. } => self.builder().ins().sqrt(input),
            FpUnaryKind::Convert(FloatConversion::SingleToDouble) => {
                self.builder().ins().fpromote(types::F64, input)
            }
            FpUnaryKind::Convert(FloatConversion::DoubleToSingle) => {
                self.builder().ins().fdemote(types::F32, input)
            }
        }
    }
    /// x86 FTZ raises PE as well as UE for an exact tiny cancellation; Arm FZ
    /// requires UFC alone. Exclude that domain before producing host status.
    /// Normal/zero inputs with effective opposite signs can have a subnormal
    /// difference only below exponent-field precision+1 (including the binade
    /// boundary's half-ULP spacing). All other guarded adds remain native.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/Shared-Pseudocode/shared.functions.float.fpadd.FPAdd
    fn fp_add_status_compatible(
        &mut self,
        first: Value,
        second: Value,
        width: u32,
        operation: nixe_cpu::decode::a64::fp_simd::FloatAddOperation,
        fpcr: Value,
        abi: crate::abi::HostAbi,
    ) -> Value {
        if abi == crate::abi::HostAbi::Aarch64 {
            return self.builder().ins().iconst(types::I8, 1);
        }
        let sign_mask = 1u64 << (width - 1);
        let threshold = if width == 32 {
            25u64 << 23
        } else {
            54u64 << 52
        };
        let signs = self.builder().ins().bxor(first, second);
        let signs = self.builder().ins().band_imm_u(signs, sign_mask as i64);
        let opposite = self.builder().ins().icmp_imm_s(
            if matches!(
                operation,
                nixe_cpu::decode::a64::fp_simd::FloatAddOperation::Add
            ) {
                IntCC::NotEqual
            } else {
                IntCC::Equal
            },
            signs,
            0,
        );
        let first = self
            .builder()
            .ins()
            .band_imm_u(first, (sign_mask - 1) as i64);
        let second = self
            .builder()
            .ins()
            .band_imm_u(second, (sign_mask - 1) as i64);
        let small_first =
            self.builder()
                .ins()
                .icmp_imm_u(IntCC::UnsignedLessThan, first, threshold as i64);
        let small_second =
            self.builder()
                .ins()
                .icmp_imm_u(IntCC::UnsignedLessThan, second, threshold as i64);
        let small = self.builder().ins().band(small_first, small_second);
        let cancellation = self.builder().ins().band(small, opposite);
        let fz = self.builder().ins().band_imm_u(fpcr, 1 << 24);
        let fz = self.builder().ins().icmp_imm_s(IntCC::NotEqual, fz, 0);
        let exact = self.builder().ins().band(cancellation, fz);
        self.builder().ins().icmp_imm_s(IntCC::Equal, exact, 0)
    }

    /// Native result in an active, eligible guest FP segment. The owner retains
    /// sticky host status until observation or canonical exit.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FADD--Floating-point-Add--scalar--
    fn float_add_values(
        &mut self,
        first: Value,
        second: Value,
        operation: nixe_cpu::decode::a64::fp_simd::FloatAddOperation,
    ) -> Value {
        use nixe_cpu::decode::a64::fp_simd::FloatAddOperation;
        match operation {
            FloatAddOperation::Add => self.builder().ins().fadd(first, second),
            FloatAddOperation::Subtract => self.builder().ins().fsub(first, second),
        }
    }
    /// AArch64's FRINTN/P/M/Z do not set IXC. Call only on finite normal/zero
    /// inputs under the shared GuardedExact policy; x86 CLIF rounding may leak
    /// precision status and must not use this path.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FRINTN--Floating-point-Round-to-Integral--to-nearest-with-ties-to-even--scalar--
    fn native_scalar_round(
        &mut self,
        bits: Value,
        width: u32,
        operation: nixe_cpu::decode::a64::fp_simd::FloatRoundOperation,
    ) -> Value {
        use nixe_cpu::decode::a64::fp_simd::FloatRoundOperation;
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let value = self.builder().ins().bitcast(ty, bitcast_flags(), bits);
        let result = match operation {
            FloatRoundOperation::NearestEven => self.builder().ins().nearest(value),
            FloatRoundOperation::TowardPositive => self.builder().ins().ceil(value),
            FloatRoundOperation::TowardNegative => self.builder().ins().floor(value),
            FloatRoundOperation::TowardZero => self.builder().ins().trunc(value),
            FloatRoundOperation::NearestAway
            | FloatRoundOperation::Exact
            | FloatRoundOperation::CurrentMode => unreachable!("exact-only rounding"),
        };
        let integer_ty = if width == 32 { types::I32 } else { types::I64 };
        let result = self
            .builder()
            .ins()
            .bitcast(integer_ty, bitcast_flags(), result);
        let result = self.builder().ins().uextend(types::I128, result);
        self.vector_as(result, types::I8X16)
    }

    fn scalar_fp_bits(&mut self, register: u8, width: u32) -> Result<Value, Error> {
        let value = self.read_vector_as(register, types::I128)?;
        Ok(self
            .builder()
            .ins()
            .ireduce(if width == 32 { types::I32 } else { types::I64 }, value))
    }

    /// True for zero and finite normal values. NaNs, infinities and denormal
    /// inputs use the exact edge because their payload/status contracts differ
    /// between Arm and the host FP ISA.
    fn fp_finite_or_zero(&mut self, bits: Value, width: u32) -> Value {
        let (exponent_mask, magnitude_mask) = if width == 32 {
            (0x7f80_0000_u64, 0x7fff_ffff_u64)
        } else {
            (0x7ff0_0000_0000_0000, 0x7fff_ffff_ffff_ffff)
        };
        let exponent = self.builder().ins().band_imm_u(bits, exponent_mask as i64);
        let exponent_nonzero = self
            .builder()
            .ins()
            .icmp_imm_s(IntCC::NotEqual, exponent, 0);
        let exponent_finite =
            self.builder()
                .ins()
                .icmp_imm_s(IntCC::NotEqual, exponent, exponent_mask as i64);
        let normal = self.builder().ins().band(exponent_nonzero, exponent_finite);
        let magnitude = self.builder().ins().band_imm_u(bits, magnitude_mask as i64);
        let zero = self.builder().ins().icmp_imm_s(IntCC::Equal, magnitude, 0);
        self.builder().ins().bor(zero, normal)
    }

    fn fp_vector_nonfinite_or_subnormal_lanes(&mut self, value: Value, lane_bits: u32) -> Value {
        let (exponent_mask, fraction_mask) = if lane_bits == 32 {
            (0x7f80_0000_u64, 0x007f_ffff_u64)
        } else {
            (0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff)
        };
        let ty = self.builder().func.dfg.value_type(value);
        let exponent_mask = self.fp_vector_lane_constant(ty, lane_bits, exponent_mask);
        let fraction_mask = self.fp_vector_lane_constant(ty, lane_bits, fraction_mask);
        let zero = self.fp_vector_lane_constant(ty, lane_bits, 0);
        let exponent = self.builder().ins().band(value, exponent_mask);
        let fraction = self.builder().ins().band(value, fraction_mask);
        let exponent_zero = self.builder().ins().icmp(IntCC::Equal, exponent, zero);
        let exponent_ones = self
            .builder()
            .ins()
            .icmp(IntCC::Equal, exponent, exponent_mask);
        let fraction_nonzero = self.builder().ins().icmp(IntCC::NotEqual, fraction, zero);
        let subnormal = self.builder().ins().band(exponent_zero, fraction_nonzero);
        self.builder().ins().bor(exponent_ones, subnormal)
    }

    /// Normalize packed FDIV operands without executing FP. CLIF has no F32X2:
    /// use exact 0/1 in its inactive lanes, never 0/0 or poisoned guest bits.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--vector---Floating-point-Divide--vector--
    fn fp_vector_divide_operands(
        &mut self,
        first: Value,
        second: Value,
        lane_bits: u32,
        vector_bits: u32,
    ) -> (Value, Value) {
        let first = self.mask_vector(first, vector_bits);
        let mut second = self.mask_vector(second, vector_bits);
        if vector_bits == 64 {
            let ones = self.vector_constant(0x3f80_0000_3f80_0000_0000_0000_0000_0000);
            second = self.builder().ins().bor(second, ones);
        }
        let ty = if lane_bits == 32 {
            types::I32X4
        } else {
            types::I64X2
        };
        (self.vector_as(first, ty), self.vector_as(second, ty))
    }

    /// Packed counterpart of the scalar FDIV domain. All guards use integer
    /// instructions before activation; keep normal/zero inputs, nonzero
    /// divisors, and on x86 exclude potentially tiny results (Arm tininess/FZ
    /// status differs). Reduce lane predicates once instead of scalarizing.
    fn fp_vector_divide_domain(
        &mut self,
        first: Value,
        second: Value,
        lane_bits: u32,
        abi: crate::abi::HostAbi,
    ) -> Value {
        let invalid_first = self.fp_vector_nonfinite_or_subnormal_lanes(first, lane_bits);
        let invalid_second = self.fp_vector_nonfinite_or_subnormal_lanes(second, lane_bits);
        let mut invalid = self.builder().ins().bor(invalid_first, invalid_second);
        let ty = self.builder().func.dfg.value_type(first);
        let magnitude_mask = (1u64 << (lane_bits - 1)) - 1;
        let magnitude_mask = self.fp_vector_lane_constant(ty, lane_bits, magnitude_mask);
        let zero = self.fp_vector_lane_constant(ty, lane_bits, 0);
        let second = self.builder().ins().band(second, magnitude_mask);
        let zero_divisor = self.builder().ins().icmp(IntCC::Equal, second, zero);
        invalid = self.builder().ins().bor(invalid, zero_divisor);
        if abi == crate::abi::HostAbi::X86_64 {
            let first = self.builder().ins().band(first, magnitude_mask);
            let first_nonzero = self.builder().ins().icmp(IntCC::NotEqual, first, zero);
            let (fraction, minimum) = if lane_bits == 32 {
                (23, (-125i32) as u32 as u64)
            } else {
                (52, (-1021i64) as u64)
            };
            let first_exp = self.builder().ins().ushr_imm_u(first, fraction);
            let second_exp = self.builder().ins().ushr_imm_u(second, fraction);
            let difference = self.builder().ins().isub(first_exp, second_exp);
            let minimum = self.fp_vector_lane_constant(ty, lane_bits, minimum);
            let tiny = self
                .builder()
                .ins()
                .icmp(IntCC::SignedLessThan, difference, minimum);
            let tiny = self.builder().ins().band(tiny, first_nonzero);
            invalid = self.builder().ins().bor(invalid, tiny);
        }
        let invalid = self.vector_as(invalid, types::I128);
        self.builder().ins().icmp_imm_s(IntCC::Equal, invalid, 0)
    }

    /// One packed divide, with no scalar lane loop or per-lane helper calls.
    fn fp_vector_divide_value(
        &mut self,
        first: Value,
        second: Value,
        lane_bits: u32,
        vector_bits: u32,
    ) -> Value {
        let ty = if lane_bits == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let first = self.vector_as(first, ty);
        let second = self.vector_as(second, ty);
        let result = self.builder().ins().fdiv(first, second);
        let result = self.vector_as(result, types::I8X16);
        self.mask_vector(result, vector_bits)
    }

    /// FMUL by element selects Rm's lane before masking the destination shape.
    /// Inactive numerator lanes are zero; the guarded finite multiplier makes
    /// their products exact zeros, whose sign is cleared by the output mask.
    /// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMUL--by-element---Floating-point-Multiply--by-element--
    fn fp_vector_multiply_element_operands(
        &mut self,
        first: Value,
        second: Value,
        lane_bits: u32,
        vector_bits: u32,
        lane: u8,
    ) -> (Value, Value) {
        let ty = if lane_bits == 32 {
            types::I32X4
        } else {
            types::I64X2
        };
        let first = self.mask_vector(first, vector_bits);
        let first = self.vector_as(first, ty);
        let second = self.vector_as(second, ty);
        let element = self.builder().ins().extractlane(second, lane);
        let second = self.builder().ins().splat(ty, element);
        (first, second)
    }

    /// Packed scalar-FMUL domain: combine integer lane predicates once. x86
    /// excludes potential tiny products because Arm detects tininess before
    /// rounding and FZ flushing need not raise IXC. Zero products remain native.
    fn fp_vector_multiply_domain(
        &mut self,
        first: Value,
        second: Value,
        lane_bits: u32,
        abi: crate::abi::HostAbi,
    ) -> Value {
        let first_bad = self.fp_vector_nonfinite_or_subnormal_lanes(first, lane_bits);
        let second_bad = self.fp_vector_nonfinite_or_subnormal_lanes(second, lane_bits);
        let mut invalid = self.builder().ins().bor(first_bad, second_bad);
        if abi == crate::abi::HostAbi::X86_64 {
            let ty = self.builder().func.dfg.value_type(first);
            let mask = self.fp_vector_lane_constant(ty, lane_bits, (1u64 << (lane_bits - 1)) - 1);
            let zero = self.fp_vector_lane_constant(ty, lane_bits, 0);
            let first = self.builder().ins().band(first, mask);
            let second = self.builder().ins().band(second, mask);
            let first_nonzero = self.builder().ins().icmp(IntCC::NotEqual, first, zero);
            let second_nonzero = self.builder().ins().icmp(IntCC::NotEqual, second, zero);
            let nonzero = self.builder().ins().band(first_nonzero, second_nonzero);
            let (fraction, threshold) = if lane_bits == 32 {
                (23, 128)
            } else {
                (52, 1024)
            };
            let first_exp = self.builder().ins().ushr_imm_u(first, fraction);
            let second_exp = self.builder().ins().ushr_imm_u(second, fraction);
            let sum = self.builder().ins().iadd(first_exp, second_exp);
            let threshold = self.fp_vector_lane_constant(ty, lane_bits, threshold);
            // Exponent sums are nonnegative and at most 4094. Signed SIMD
            // comparison avoids the unsigned-order adjustment on baseline x86.
            let tiny = self
                .builder()
                .ins()
                .icmp(IntCC::SignedLessThan, sum, threshold);
            let tiny = self.builder().ins().band(tiny, nonzero);
            invalid = self.builder().ins().bor(invalid, tiny);
        }
        let invalid = self.vector_as(invalid, types::I128);
        self.builder().ins().icmp_imm_s(IntCC::Equal, invalid, 0)
    }

    /// One packed FMUL inside the active native FP segment.
    fn fp_vector_multiply_value(
        &mut self,
        first: Value,
        second: Value,
        lane_bits: u32,
        vector_bits: u32,
    ) -> Value {
        let ty = if lane_bits == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let first = self.vector_as(first, ty);
        let second = self.vector_as(second, ty);
        let result = self.builder().ins().fmul(first, second);
        let result = self.vector_as(result, types::I8X16);
        self.mask_vector(result, vector_bits)
    }

    fn fp_vector_lane_constant(
        &mut self,
        ty: cranelift_codegen::ir::Type,
        lane_bits: u32,
        lane_value: u64,
    ) -> Value {
        let mut bits = 0_u128;
        for offset in (0..128).step_by(lane_bits as usize) {
            bits |= u128::from(lane_value) << offset;
        }
        let value = self.vector_constant(bits);
        self.vector_as(value, ty)
    }

    // Ordered S/D comparison: input guards exclude NaNs and subnormals, so
    // this path neither consumes rounding control nor produces FP status.
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCMP--Floating-point-Compare--scalar--
    fn ordered_fp_compare(&mut self, first_bits: Value, second_bits: Value, width: u32) -> Value {
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self
            .builder()
            .ins()
            .bitcast(ty, bitcast_flags(), first_bits);
        let second = self
            .builder()
            .ins()
            .bitcast(ty, bitcast_flags(), second_bits);
        let equal = self.builder().ins().fcmp(FloatCC::Equal, first, second);
        let less = self.builder().ins().fcmp(FloatCC::LessThan, first, second);
        let equal_flags = self.builder().ins().iconst(types::I32, 0x6000_0000);
        let less_flags = self
            .builder()
            .ins()
            .iconst(types::I32, 0x8000_0000_u32 as i64);
        let greater_flags = self.builder().ins().iconst(types::I32, 0x2000_0000);
        let ordered = self.builder().ins().select(less, less_flags, greater_flags);
        self.builder().ins().select(equal, equal_flags, ordered)
    }
}
