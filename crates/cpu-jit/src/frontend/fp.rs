use super::*;
use crate::abi::FpFusedOperation;
use crate::abi::IntegerToFpOperation;
use crate::abi::{
    FpAddOperation, FpCompareOperation, FpDivideOperation, FpMultiplyOperation, FpRoundOperation,
    FpToIntegerOperation, FpUnaryKind, FpUnaryOperation,
};
use crate::fp_policy::{FpLoweringDisposition, fp_lowering_for_host};
use crate::simd_lowering::scalar_width;
use nixe_cpu::decode::a64::fp_simd::Instruction;

impl Translator<'_> {
    pub(crate) fn fp(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        match instruction {
            Instruction::CompareRegister(_)
            | Instruction::CompareZero(_)
            | Instruction::ConditionalCompare(_) => self.fp_compare(pc, instruction, flags),
            Instruction::ScalarFloatRound(_) => self.fp_round(pc, instruction, flags),
            Instruction::ScalarFloatAdd(_) => self.fp_add(pc, instruction, flags),
            Instruction::ScalarFloatMaxNumber(_)
            | Instruction::ScalarFloatMinNumber(_)
            | Instruction::ScalarFloatMax(_)
            | Instruction::ScalarFloatMin(_) => self.fp_min_max(pc, instruction, flags),
            Instruction::VectorFloatAdd(_) => self.vector_fp_add(pc, instruction, flags),
            Instruction::ScalarFloatDivide(_) => self.fp_divide(pc, instruction, flags),
            Instruction::VectorFloatDivide(_) => self.vector_fp_divide(pc, instruction, flags),
            Instruction::VectorFloatMultiply(_) | Instruction::VectorFloatMultiplyElement(_) => {
                self.vector_fp_multiply(pc, instruction, flags)
            }
            Instruction::ScalarFloatMultiply(_) | Instruction::ScalarFloatMultiplyElement(_) => {
                self.fp_multiply(pc, instruction, flags)
            }
            Instruction::ScalarFloatFusedMultiplyAdd(_)
            | Instruction::ScalarFloatFusedElement(_) => self.fp_fused(pc, instruction, flags),
            Instruction::VectorFloatFusedElement(_) | Instruction::VectorFloatFused(_) => {
                self.vector_fp_fused(pc, instruction, flags)
            }
            Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
                self.integer_to_fp(pc, instruction, flags)
            }
            Instruction::VectorSignedIntToFloat(_)
            | Instruction::VectorUnsignedIntToFloat(_)
            | Instruction::ScalarVectorSignedIntToFloat(_)
            | Instruction::ScalarVectorUnsignedIntToFloat(_) => {
                self.vector_integer_to_fp(pc, instruction, flags)
            }
            Instruction::VectorFloatConvertNarrow(_) => {
                self.fp_convert_narrow(pc, instruction, flags)
            }
            Instruction::VectorFloatConvertLong(_) => self.fp_convert_long(pc, instruction, flags),
            Instruction::ScalarFloatSquareRoot(_) | Instruction::ScalarFloatConvert(_) => {
                self.fp_unary(pc, instruction, flags)
            }
            Instruction::VectorFloatToSignedInt(_) | Instruction::VectorFloatToUnsignedInt(_) => {
                self.vector_fp_to_integer(pc, instruction, flags)
            }
            Instruction::FloatToSignedInt(_)
            | Instruction::FloatToUnsignedInt(_)
            | Instruction::ScalarVectorFloatToSignedInt(_)
            | Instruction::ScalarVectorFloatToUnsignedInt(_) => {
                self.fp_to_integer(pc, instruction, flags)
            }
            _ => unreachable!("unported FP lowering rejected before builder creation"),
        }
    }

    // Normal, in-range lanes lower natively; exceptional values use the exact
    // typed completion, including saturation, cumulative status and traps.
    // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85 (pp. 1330-1332, 1344-1346)
    fn vector_fp_to_integer(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let width = scalar_width(f.opc)?;
        let vector_bits = if f.vector_128 { 128 } else { 64 };
        let operation = crate::abi::VectorFpToIntegerOperation {
            rn: f.rn,
            rd: f.rd,
            lane_64: width == 64,
            vector_bits: vector_bits as u8,
            signed: matches!(instruction, Instruction::VectorFloatToSignedInt(_)),
        };
        let source = self.read_vector(f.rn)?;
        let lanes = self.vector_as(
            source,
            if width == 64 {
                types::I64X2
            } else {
                types::I32X4
            },
        );
        let mut direct = self.builder.ins().iconst(types::I8, 1);
        for lane in 0..vector_bits / width {
            let bits = self.builder.ins().extractlane(lanes, lane as u8);
            let eligible =
                self.fp_to_integer_domain(bits, width, operation.lane_64, operation.signed, 0);
            direct = self.builder.ins().band(direct, eligible);
        }
        self.native_fp_path(pc, EdgeKind::VectorFpToInteger(operation), direct, flags)?;
        let source = self.read_vector(f.rn)?;
        let source = self.mask_vector(source, vector_bits);
        let result = if width == 32 {
            let value = self.vector_as(source, types::F32X4);
            if operation.signed {
                self.builder.ins().fcvt_to_sint_sat(types::I32X4, value)
            } else {
                // Stock x86 packed unsigned saturation subtracts 2^31 in
                // every lane, including lanes that never use that result,
                // leaking host inexact flags. Subtract only in upper-half
                // lanes; Sterbenz guarantees that subtraction is exact.
                let bits = self.vector_as(value, types::I32X4);
                let magnitude = self.builder.ins().iconst(types::I32, 0x7fff_ffff);
                let magnitude = self.builder.ins().splat(types::I32X4, magnitude);
                let bits = self.builder.ins().band(bits, magnitude); // unsigned -0 is also zero
                let bound = self.builder.ins().iconst(types::I32, 0x4f00_0000); // 2^31
                let bound = self.builder.ins().splat(types::I32X4, bound);
                let high = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, bits, bound);
                let subtract = self.builder.ins().band(high, bound);
                let subtract = self.vector_as(subtract, types::F32X4);
                let adjusted = self.builder.ins().fsub(value, subtract);
                let converted = self.builder.ins().fcvt_to_sint_sat(types::I32X4, adjusted);
                let sign = self.builder.ins().iconst(types::I32, i64::from(i32::MIN));
                let sign = self.builder.ins().splat(types::I32X4, sign);
                let sign = self.builder.ins().band(high, sign);
                self.builder.ins().bor(converted, sign)
            }
        } else {
            // Baseline x86 has no packed double-to-i64 conversion.
            let lanes = self.vector_as(source, types::I64X2);
            let zero = self.builder.ins().iconst(types::I64, 0);
            let mut result = self.builder.ins().splat(types::I64X2, zero);
            for lane in 0..2 {
                let bits = self.builder.ins().extractlane(lanes, lane);
                let value = self.fp_to_integer_value(bits, true, true, operation.signed);
                result = self.builder.ins().insertlane(result, value, lane);
            }
            result
        };
        let result = self.vector_as(result, types::I8X16);
        let result = self.mask_vector(result, vector_bits);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn fp_to_integer(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let vector_destination = matches!(
            instruction,
            Instruction::ScalarVectorFloatToSignedInt(_)
                | Instruction::ScalarVectorFloatToUnsignedInt(_)
        );
        let operation = FpToIntegerOperation {
            vector_destination,
            rn: f.rn,
            rd: f.rd,
            source_64: f.opc == 1,
            destination_64: if vector_destination {
                f.opc == 1
            } else {
                f.size & 2 != 0
            },
            signed: matches!(
                instruction,
                Instruction::FloatToSignedInt(_) | Instruction::ScalarVectorFloatToSignedInt(_)
            ),
            rounding: f
                .float_to_integer_rounding
                .expect("normalized FCVT rounding"),
            fractional_bits: f.fixed_point_fraction_bits.unwrap_or(0),
        };
        let kind = EdgeKind::FpToInteger(operation);
        if fp_lowering_for_host(instruction, self.abi).is_exact() {
            self.constant_exit(pc, pc, kind, NativeExitReason::Architectural, flags)?;
            return Ok(true);
        }
        let width = scalar_width(f.opc)?;
        let bits = self.scalar_fp_bits(f.rn, width)?;
        let direct = self.fp_to_integer_domain(
            bits,
            width,
            operation.destination_64,
            operation.signed,
            operation.fractional_bits,
        );
        self.native_fp_path(pc, kind, direct, flags)?;
        // The activation continuation defines fresh SSA inputs.
        let bits = self.scalar_fp_bits(f.rn, width)?;
        let bits = self.scale_fp_to_integer_bits(bits, width, operation.fractional_bits);
        let result = self.fp_to_integer_value(
            bits,
            operation.source_64,
            operation.destination_64,
            operation.signed,
        );
        if vector_destination {
            let result = self.builder.ins().uextend(types::I128, result);
            let result = self.vector_as(result, types::I8X16);
            self.write_vector(f.rd, result);
        } else {
            self.write_register(f.rd, result);
        }
        Ok(false)
    }

    // Scalar min/max's normal/zero domain needs only integer ordering, so it cannot
    // modify host FP status and does not need to activate a host FP epoch.
    // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85
    fn fp_min_max(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let width = scalar_width(f.opc)?;
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let first_ok = self.fp_finite_or_zero(first, width);
        let second_ok = self.fp_finite_or_zero(second, width);
        let direct = self.builder.ins().band(first_ok, second_ok);
        let native = self.builder.create_block();
        let exact = self.builder.create_block();
        self.builder.set_cold_block(exact);
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(exact);
        self.constant_exit(
            pc,
            pc,
            EdgeKind::FpMinMax(crate::abi::FpMinMaxOperation {
                minimum: matches!(
                    instruction,
                    Instruction::ScalarFloatMinNumber(_) | Instruction::ScalarFloatMin(_)
                ),
                number: matches!(
                    instruction,
                    Instruction::ScalarFloatMaxNumber(_) | Instruction::ScalarFloatMinNumber(_)
                ),
                rn: f.rn,
                rm: f.rm,
                rd: f.rd,
                width_64: width == 64,
            }),
            NativeExitReason::Architectural,
            flags,
        )?;
        self.builder.switch_to_block(native);
        let sign = self.builder.ins().iconst(
            if width == 32 { types::I32 } else { types::I64 },
            (1_u64 << (width - 1)) as i64,
        );
        let mut keys = [first, second];
        for key in &mut keys {
            let negative = self
                .builder
                .ins()
                .icmp_imm_s(IntCC::SignedLessThan, *key, 0);
            let inverted = self.builder.ins().bnot(*key);
            let positive = self.builder.ins().bor(*key, sign);
            *key = self.builder.ins().select(negative, inverted, positive);
        }
        let comparison = if matches!(
            instruction,
            Instruction::ScalarFloatMinNumber(_) | Instruction::ScalarFloatMin(_)
        ) {
            IntCC::UnsignedLessThan
        } else {
            IntCC::UnsignedGreaterThan
        };
        let selected = self.builder.ins().icmp(comparison, keys[0], keys[1]);
        let result = self.builder.ins().select(selected, first, second);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn fp_add(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let width = scalar_width(f.opc)?;
        let operation = FpAddOperation {
            rn: f.rn,
            rm: f.rm,
            rd: f.rd,
            width_64: width == 64,
            operation: f.float_add_operation.expect("normalized FP add operation"),
        };
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let first_ok = self.fp_finite_or_zero(first, width);
        let second_ok = self.fp_finite_or_zero(second, width);
        let direct = self.builder.ins().band(first_ok, second_ok);
        let fpcr = self.system_value(GuestValue::Fpcr)?;
        let compatible = self.fp_add_status_compatible(
            first,
            second,
            width,
            operation.operation,
            fpcr,
            self.abi,
        );
        let direct = self.builder.ins().band(direct, compatible);
        self.native_fp_path(pc, EdgeKind::FpAdd(operation), direct, flags)?;
        // Activation defines new SSA inputs; reload through the continuation.
        let first = self.fp_element_value(f.rn, width, None)?;
        let second = self.fp_element_value(f.rm, width, None)?;
        let result = self.float_add_values(first, second, operation.operation);
        self.write_fp_scalar(f.rd, result);
        Ok(false)
    }

    fn vector_fp_add(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let lane_bits = scalar_width(f.opc)?;
        let vector_bits = if f.vector_128 { 128 } else { 64 };
        let operation = crate::abi::VectorFpAddOperation {
            rn: f.rn,
            rm: f.rm,
            rd: f.rd,
            lane_64: lane_bits == 64,
            vector_128: f.vector_128,
            operation: f.float_add_operation.expect("normalized vector FP add"),
        };
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) =
            self.fp_vector_binary_operands(first, second, lane_bits, vector_bits, None);
        let first_bad = self.fp_vector_nonfinite_or_subnormal_lanes(first, lane_bits);
        let second_bad = self.fp_vector_nonfinite_or_subnormal_lanes(second, lane_bits);
        let bad = self.builder.ins().bor(first_bad, second_bad);
        let bad = self.vector_as(bad, types::I128);
        let direct = self.builder.ins().icmp_imm_s(IntCC::Equal, bad, 0);
        let fpcr = self.system_value(GuestValue::Fpcr)?;
        let compatible = self.fp_add_status_compatible(
            first,
            second,
            lane_bits,
            operation.operation,
            fpcr,
            self.abi,
        );
        let direct = self.builder.ins().band(direct, compatible);
        self.native_fp_path(pc, EdgeKind::VectorFpAdd(operation), direct, flags)?;
        // Activation starts fresh SSA inputs; mask inactive lanes before status production.
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) =
            self.fp_vector_binary_operands(first, second, lane_bits, vector_bits, None);
        let ty = if lane_bits == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let first = self.vector_as(first, ty);
        let second = self.vector_as(second, ty);
        let result = self.float_add_values(first, second, operation.operation);
        let result = self.vector_as(result, types::I8X16);
        let result = self.mask_vector(result, vector_bits);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn fp_divide(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let width = scalar_width(f.opc)?;
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let first_ok = self.fp_finite_or_zero(first, width);
        let second_ok = self.fp_finite_or_zero(second, width);
        let direct = self.builder.ins().band(first_ok, second_ok);
        let compatible = self.fp_divide_domain(first, second, width, self.abi);
        let direct = self.builder.ins().band(direct, compatible);
        self.native_fp_path(
            pc,
            EdgeKind::FpDivide(FpDivideOperation {
                rn: f.rn,
                rm: f.rm,
                rd: f.rd,
                width_64: width == 64,
            }),
            direct,
            flags,
        )?;
        // The activation continuation owns fresh SSA bindings.
        let first = self.fp_element_value(f.rn, width, None)?;
        let second = self.fp_element_value(f.rm, width, None)?;
        let result = self.fp_divide_value(first, second);
        self.write_fp_scalar(f.rd, result);
        Ok(false)
    }

    fn vector_fp_divide(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let lane_bits = scalar_width(f.opc)?;
        let vector_bits = if f.vector_128 { 128 } else { 64 };
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) = self.fp_vector_divide_operands(first, second, lane_bits, vector_bits);
        let direct = self.fp_vector_divide_domain(first, second, lane_bits, self.abi);
        self.native_fp_path(
            pc,
            EdgeKind::VectorFpDivide(crate::abi::VectorFpDivideOperation {
                rn: f.rn,
                rm: f.rm,
                rd: f.rd,
                lane_64: lane_bits == 64,
                vector_128: f.vector_128,
            }),
            direct,
            flags,
        )?;
        // Activation starts a fresh SSA entry, including the vector operands.
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) = self.fp_vector_divide_operands(first, second, lane_bits, vector_bits);
        let result = self.fp_vector_divide_value(first, second, lane_bits, vector_bits);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn vector_fp_multiply(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let lane = matches!(instruction, Instruction::VectorFloatMultiplyElement(_))
            .then_some(f.fp_element_lane);
        let lane_bits = scalar_width(f.opc)?;
        let vector_bits = if f.vector_128 { 128 } else { 64 };
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) =
            self.fp_vector_binary_operands(first, second, lane_bits, vector_bits, lane);
        let direct = self.fp_vector_multiply_domain(first, second, lane_bits, self.abi);
        self.native_fp_path(
            pc,
            EdgeKind::VectorFpMultiply(crate::abi::VectorFpMultiplyOperation {
                rn: f.rn,
                rm: f.rm,
                rd: f.rd,
                lane_64: lane_bits == 64,
                vector_128: f.vector_128,
                lane,
            }),
            direct,
            flags,
        )?;
        // Re-read operands after activation establishes fresh SSA inputs.
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) =
            self.fp_vector_binary_operands(first, second, lane_bits, vector_bits, lane);
        let result = self.fp_vector_multiply_value(first, second, lane_bits, vector_bits);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn vector_fp_fused(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let lane_bits = scalar_width(f.opc)?;
        let vector_bits = if f.vector_128 { 128 } else { 64 };
        let lane = matches!(instruction, Instruction::VectorFloatFusedElement(_))
            .then_some(f.fp_element_lane);
        let edge = EdgeKind::VectorFpFused(crate::abi::VectorFpFusedOperation {
            rn: f.rn,
            rm: f.rm,
            rd: f.rd,
            lane_64: lane_bits == 64,
            vector_128: f.vector_128,
            lane,
            subtract: f.subtract,
        });
        if !self.native_fma {
            self.constant_exit(pc, pc, edge, NativeExitReason::Architectural, flags)?;
            return Ok(true);
        }
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let third = self.read_vector(f.rd)?;
        let (first, second) =
            self.fp_vector_binary_operands(first, second, lane_bits, vector_bits, lane);
        let third = self.mask_vector(third, vector_bits);
        let ty = self.builder.func.dfg.value_type(first);
        let third = self.vector_as(third, ty);
        let direct = self.fp_vector_fused_domain(first, second, third, lane_bits);
        self.native_fp_path(pc, edge, direct, flags)?;
        // FP-region activation starts fresh SSA inputs, including Rd's accumulator.
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let third = self.read_vector(f.rd)?;
        let (first, second) =
            self.fp_vector_binary_operands(first, second, lane_bits, vector_bits, lane);
        let third = self.mask_vector(third, vector_bits);
        let ty = if lane_bits == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let first = self.vector_as(first, ty);
        let second = self.vector_as(second, ty);
        let third = self.vector_as(third, ty);
        use nixe_cpu::decode::a64::fp_simd::FloatFusedMultiplyOperation as Op;
        let result = self.fp_fused_value(
            first,
            second,
            third,
            if f.subtract {
                Op::MultiplySubtract
            } else {
                Op::MultiplyAdd
            },
        );
        let result = self.vector_as(result, types::I8X16);
        let result = self.mask_vector(result, vector_bits);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    // Extract arithmetic operands as FP values directly. Extracting integer
    // bits and bitcasting afterward crosses register files on baseline x86;
    // the integer extraction remains separate for exception-domain guards.
    fn fp_element_value(
        &mut self,
        register: u8,
        width: u32,
        lane: Option<u8>,
    ) -> Result<ir::Value, Error> {
        let ty = if width == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let vector = self.read_vector_as(register, ty)?;
        Ok(self.builder.ins().extractlane(vector, lane.unwrap_or(0)))
    }

    fn fp_element_bits(
        &mut self,
        register: u8,
        width: u32,
        lane: Option<u8>,
    ) -> Result<ir::Value, Error> {
        if lane.unwrap_or(0) == 0 {
            return self.scalar_fp_bits(register, width);
        }
        let ty = if width == 32 {
            types::I32X4
        } else {
            types::I64X2
        };
        let vector = self.read_vector_as(register, ty)?;
        Ok(self
            .builder
            .ins()
            .extractlane(vector, lane.expect("nonzero lane")))
    }

    fn fp_multiply(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let width = scalar_width(f.opc)?;
        let lane = matches!(instruction, Instruction::ScalarFloatMultiplyElement(_))
            .then_some(f.fp_element_lane);
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.fp_element_bits(f.rm, width, lane)?;
        let first_ok = self.fp_finite_or_zero(first, width);
        let second_ok = self.fp_finite_or_zero(second, width);
        let mut direct = self.builder.ins().band(first_ok, second_ok);
        if self.abi == HostAbi::X86_64 {
            let compatible = self.fp_multiply_domain(first, second, width);
            direct = self.builder.ins().band(direct, compatible);
        }
        self.native_fp_path(
            pc,
            EdgeKind::FpMultiply(FpMultiplyOperation {
                lane,
                rn: f.rn,
                rm: f.rm,
                rd: f.rd,
                width_64: width == 64,
                operation: f
                    .float_multiply_operation
                    .expect("normalized FP multiply operation"),
            }),
            direct,
            flags,
        )?;
        // The activation continuation owns fresh SSA bindings.
        let first = self.fp_element_value(f.rn, width, None)?;
        let second = self.fp_element_value(f.rm, width, lane)?;
        let result = self.fp_multiply_value(
            first,
            second,
            f.float_multiply_operation
                .expect("normalized FP multiply operation"),
        );
        self.write_fp_scalar(f.rd, result);
        Ok(false)
    }

    fn fp_fused(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let width = scalar_width(f.opc)?;
        let lane = matches!(instruction, Instruction::ScalarFloatFusedElement(_))
            .then_some(f.fp_element_lane);
        let operation = FpFusedOperation {
            lane,
            rn: f.rn,
            rm: f.rm,
            ra: f.ra,
            rd: f.rd,
            width_64: width == 64,
            operation: f
                .float_fused_multiply_operation
                .expect("normalized fused operation"),
        };
        if !self.native_fma {
            // The baseline x86 CLIF lowering is a libcall, forbidden inside
            // this frameless ABI. Complete the same typed operation outside it.
            self.constant_exit(
                pc,
                pc,
                EdgeKind::FpFused(operation),
                NativeExitReason::Architectural,
                flags,
            )?;
            return Ok(true);
        }
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.fp_element_bits(f.rm, width, lane)?;
        let third = self.scalar_fp_bits(f.ra, width)?;
        let first_ok = self.fp_finite_or_zero(first, width);
        let second_ok = self.fp_finite_or_zero(second, width);
        let third_ok = self.fp_finite_or_zero(third, width);
        let direct = self.builder.ins().band(first_ok, second_ok);
        let mut direct = self.builder.ins().band(direct, third_ok);
        if self.abi == HostAbi::X86_64 {
            let compatible = self.fp_fused_domain(first, second, third, width);
            direct = self.builder.ins().band(direct, compatible);
        }
        self.native_fp_path(pc, EdgeKind::FpFused(operation), direct, flags)?;
        let first = self.fp_element_value(f.rn, width, None)?;
        let second = self.fp_element_value(f.rm, width, lane)?;
        let third = self.fp_element_value(f.ra, width, None)?;
        let result = self.fp_fused_value(first, second, third, operation.operation);
        self.write_fp_scalar(f.rd, result);
        Ok(false)
    }

    fn vector_integer_to_fp(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let lane_bits = scalar_width(f.opc)?;
        let scalar = matches!(
            instruction,
            Instruction::ScalarVectorSignedIntToFloat(_)
                | Instruction::ScalarVectorUnsignedIntToFloat(_)
        );
        let vector_bits = if scalar {
            lane_bits
        } else if f.vector_128 {
            128
        } else {
            64
        };
        let signed = matches!(
            instruction,
            Instruction::VectorSignedIntToFloat(_) | Instruction::ScalarVectorSignedIntToFloat(_)
        );
        let operation = crate::abi::VectorIntegerToFpOperation {
            rn: f.rn,
            rd: f.rd,
            lane_64: lane_bits == 64,
            vector_bits: vector_bits as u8,
            signed,
        };
        let eligible = self.builder.ins().iconst(types::I8, 1);
        self.native_fp_path(pc, EdgeKind::VectorIntegerToFp(operation), eligible, flags)?;
        let source = self.read_vector(f.rn)?;
        let result =
            self.vector_integer_to_fp_value(source, operation.lane_64, vector_bits, signed);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn integer_to_fp(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let operation = IntegerToFpOperation {
            rn: f.rn,
            rd: f.rd,
            source_64: f.size & 2 != 0,
            destination_64: f.opc == 1,
            signed: matches!(instruction, Instruction::SignedIntToFloat(_)),
            fractional_bits: f.fixed_point_fraction_bits.unwrap_or(0),
        };
        let eligible = self.builder.ins().iconst(types::I8, 1);
        self.native_fp_path(pc, EdgeKind::IntegerToFp(operation), eligible, flags)?;
        // Read after activation, whose continuation owns new SSA bindings.
        let value = self.read_register(f.rn, false)?;
        let result = self.integer_to_fp_value(
            value,
            operation.source_64,
            operation.destination_64,
            operation.signed,
            operation.fractional_bits,
        );
        self.write_fp_scalar(f.rd, result);
        Ok(false)
    }

    fn native_fp_path(
        &mut self,
        pc: GuestVirtualAddress,
        kind: EdgeKind,
        direct: ir::Value,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<(), Error> {
        let fpcr = self.system_value(GuestValue::Fpcr)?;
        let unsupported = self
            .builder
            .ins()
            .band_imm_u(fpcr, i64::from(!crate::fp_policy::NATIVE_FPCR_MASK));
        let supported = self.builder.ins().icmp_imm_s(IntCC::Equal, unsupported, 0);
        let direct = self.builder.ins().band(direct, supported);
        let native = self.builder.create_block();
        let exact = self.builder.create_block();
        self.builder.set_cold_block(exact);
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(exact);
        self.constant_exit(pc, pc, kind, NativeExitReason::Architectural, flags)?;
        self.builder.switch_to_block(native);
        self.ensure_fp(pc, flags);
        Ok(())
    }

    fn write_fp_scalar(&mut self, rd: u8, result: ir::Value) {
        let width = self.builder.func.dfg.value_type(result).bits();
        // Scalar Arm FP writes clear the remaining vector bits. Keep the
        // value in the SIMD register file instead of splitting an I128 value
        // into GPR halves and reconstructing it after every arithmetic result.
        // Switch 1 has no FEAT_AFP merging mode; see FADD's 128-bit result:
        // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85 (p. 1188)
        let result = self.builder.ins().scalar_to_vector(
            if width == 32 {
                types::F32X4
            } else {
                types::F64X2
            },
            result,
        );
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(rd, result);
    }

    // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85 (FCVTN pp. 1281–1282)
    fn fp_convert_narrow(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let mut direct = None;
        for lane in 0..2 {
            let bits = self.fp_element_bits(f.rn, 64, Some(lane))?;
            let mut eligible = self.fp_finite_or_zero(bits, 64);
            if self.abi == HostAbi::X86_64 {
                let normal = self.fp_demote_domain(bits);
                eligible = self.builder.ins().band(eligible, normal);
            }
            direct = Some(if let Some(previous) = direct {
                self.builder.ins().band(previous, eligible)
            } else {
                eligible
            });
        }
        let kind = FpUnaryKind::ConvertNarrow {
            upper: f.vector_128,
        };
        self.native_fp_path(
            pc,
            EdgeKind::FpUnary(FpUnaryOperation {
                rn: f.rn,
                rd: f.rd,
                kind,
            }),
            direct.expect("two conversion lanes"),
            flags,
        )?;
        let source = self.read_vector_as(f.rn, types::F64X2)?;
        let result = self.fp_unary_value(source, kind);
        let result = self.vector_as(result, types::I8X16);
        let result = if f.vector_128 {
            let old = self.read_vector(f.rd)?;
            self.shuffle_bytes(
                old,
                result,
                [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
            )
        } else {
            result
        };
        self.write_vector(f.rd, result);
        Ok(false)
    }

    // Native widening for normal/zero lanes, with the exact typed edge for
    // NaNs and denormals. FP ownership/exception traps use the shared guard.
    // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85 (FCVTL pp. 1261–1262)
    fn fp_convert_long(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let first = if f.vector_128 { 2 } else { 0 };
        let low = self.fp_element_bits(f.rn, 32, Some(first))?;
        let high = self.fp_element_bits(f.rn, 32, Some(first + 1))?;
        let low = self.fp_finite_or_zero(low, 32);
        let high = self.fp_finite_or_zero(high, 32);
        let direct = self.builder.ins().band(low, high);
        let kind = FpUnaryKind::ConvertLong {
            upper: f.vector_128,
        };
        self.native_fp_path(
            pc,
            EdgeKind::FpUnary(FpUnaryOperation {
                rn: f.rn,
                rd: f.rd,
                kind,
            }),
            direct,
            flags,
        )?;
        let source = self.read_vector(f.rn)?;
        let source = if f.vector_128 {
            self.shuffle_bytes(
                source,
                source,
                [8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15],
            )
        } else {
            source
        };
        let source = self.vector_as(source, types::F32X4);
        let result = self.fp_unary_value(source, kind);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn fp_unary(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        use nixe_cpu::decode::a64::fp_simd::FloatConversion;
        let f = instruction.operands();
        let (width, kind) = match instruction {
            Instruction::ScalarFloatSquareRoot(_) => (
                scalar_width(f.opc)?,
                FpUnaryKind::SquareRoot {
                    width_64: f.opc == 1,
                },
            ),
            Instruction::ScalarFloatConvert(_) => {
                let conversion = f.float_conversion.expect("normalized scalar conversion");
                (
                    if conversion == FloatConversion::SingleToDouble {
                        32
                    } else {
                        64
                    },
                    FpUnaryKind::Convert(conversion),
                )
            }
            _ => unreachable!(),
        };
        let bits = self.scalar_fp_bits(f.rn, width)?;
        let mut direct = self.fp_finite_or_zero(bits, width);
        let extra = match kind {
            FpUnaryKind::SquareRoot { .. } => Some(self.fp_sqrt_domain(bits, width)),
            FpUnaryKind::Convert(FloatConversion::DoubleToSingle)
                if self.abi == HostAbi::X86_64 =>
            {
                Some(self.fp_demote_domain(bits))
            }
            _ => None,
        };
        if let Some(extra) = extra {
            direct = self.builder.ins().band(direct, extra);
        }
        self.native_fp_path(
            pc,
            EdgeKind::FpUnary(FpUnaryOperation {
                rn: f.rn,
                rd: f.rd,
                kind,
            }),
            direct,
            flags,
        )?;
        let value = self.fp_element_value(f.rn, width, None)?;
        let result = self.fp_unary_value(value, kind);
        self.write_fp_scalar(f.rd, result);
        Ok(false)
    }

    fn fp_round(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let operation = FpRoundOperation {
            rn: f.rn,
            rd: f.rd,
            width_64: f.opc == 1,
            rounding: f.float_round_operation.expect("normalized FRINT operation"),
        };
        if fp_lowering_for_host(instruction, self.abi).is_exact() {
            self.constant_exit(
                pc,
                pc,
                EdgeKind::FpRound(operation),
                NativeExitReason::Architectural,
                flags,
            )?;
            return Ok(true);
        }
        debug_assert_eq!(
            fp_lowering_for_host(instruction, self.abi),
            FpLoweringDisposition::GuardedExact
        );
        let width = scalar_width(f.opc)?;
        let bits = self.scalar_fp_bits(f.rn, width)?;
        let direct = self.fp_finite_or_zero(bits, width);
        let native = self.builder.create_block();
        let exact = self.builder.create_block();
        self.builder.set_cold_block(exact);
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(exact);
        self.constant_exit(
            pc,
            pc,
            EdgeKind::FpRound(operation),
            NativeExitReason::Architectural,
            flags,
        )?;
        self.builder.switch_to_block(native);
        let result = self.native_scalar_round(bits, width, operation.rounding);
        self.write_vector(f.rd, result);
        Ok(false)
    }

    fn fp_compare(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let operation = FpCompareOperation {
            rn: f.rn,
            rm: (!matches!(instruction, Instruction::CompareZero(_))).then_some(f.rm),
            width_64: f.opc == 1,
            signaling: f.signaling_compare,
            condition: matches!(instruction, Instruction::ConditionalCompare(_))
                .then_some((Condition::from_encoding(f.condition), f.nzcv_immediate)),
        };
        // The false condition must not observe FP inputs or raise exceptions.
        // Select harmless zeros before the ordered comparison, including when
        // the unused inputs are signaling NaNs and invalid traps are enabled.
        // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCCMP--Floating-point-Conditional-Compare--scalar--
        // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCCMPE--Floating-point-Conditional-Compare--scalar--
        let condition = operation
            .condition
            .map(|(condition, _)| self.emit_condition(condition, flags));
        let width = scalar_width(f.opc)?;
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = if let Some(rm) = operation.rm {
            self.scalar_fp_bits(rm, width)?
        } else {
            self.builder
                .ins()
                .iconst(if width == 32 { types::I32 } else { types::I64 }, 0)
        };
        let (first, second) = if let Some(condition) = condition {
            let zero = self
                .builder
                .ins()
                .iconst(if width == 32 { types::I32 } else { types::I64 }, 0);
            (
                self.builder.ins().select(condition, first, zero),
                self.builder.ins().select(condition, second, zero),
            )
        } else {
            (first, second)
        };
        let first_direct = self.fp_finite_or_zero(first, width);
        let second_direct = self.fp_finite_or_zero(second, width);
        let direct = self.builder.ins().band(first_direct, second_direct);
        let native = self.builder.create_block();
        let exact = self.builder.create_block();
        self.builder.set_cold_block(exact);
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(exact);
        self.constant_exit(
            pc,
            pc,
            EdgeKind::FpCompare(operation),
            NativeExitReason::Architectural,
            flags,
        )?;
        self.builder.switch_to_block(native);
        let ordered = self.ordered_fp_compare(first, second, width);
        let result = if let Some(condition) = condition {
            let (_, literal) = operation.condition.unwrap();
            let literal = self
                .builder
                .ins()
                .iconst(types::I32, i64::from(literal) << 28);
            self.builder.ins().select(condition, ordered, literal)
        } else {
            ordered
        };
        *flags = LazyFlags::Packed(result);
        self.dirty.nzcv = crate::analysis::NZCV;
        Ok(false)
    }
}
