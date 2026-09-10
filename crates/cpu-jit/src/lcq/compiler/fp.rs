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
    pub(super) fn fp(
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
            Instruction::ScalarFloatDivide(_) => self.fp_divide(pc, instruction, flags),
            Instruction::VectorFloatDivide(_) => self.vector_fp_divide(pc, instruction, flags),
            Instruction::VectorFloatMultiplyElement(_) => {
                self.vector_fp_multiply_element(pc, instruction, flags)
            }
            Instruction::ScalarFloatMultiply(_) => self.fp_multiply(pc, instruction, flags),
            Instruction::ScalarFloatFusedMultiplyAdd(_) => self.fp_fused(pc, instruction, flags),
            Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
                self.integer_to_fp(pc, instruction, flags)
            }
            Instruction::VectorSignedIntToFloat(_)
            | Instruction::VectorUnsignedIntToFloat(_)
            | Instruction::ScalarVectorSignedIntToFloat(_)
            | Instruction::ScalarVectorUnsignedIntToFloat(_) => {
                self.vector_integer_to_fp(pc, instruction, flags)
            }
            Instruction::ScalarFloatSquareRoot(_) | Instruction::ScalarFloatConvert(_) => {
                self.fp_unary(pc, instruction, flags)
            }
            Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
                self.fp_to_integer(pc, instruction, flags)
            }
            _ => unreachable!("unported FP lowering rejected before builder creation"),
        }
    }

    fn fp_to_integer(
        &mut self,
        pc: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        let f = instruction.operands();
        let operation = FpToIntegerOperation {
            rn: f.rn,
            rd: f.rd,
            source_64: f.opc == 1,
            destination_64: f.size & 2 != 0,
            signed: matches!(instruction, Instruction::FloatToSignedInt(_)),
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
        let direct =
            self.fp_to_integer_domain(bits, width, operation.destination_64, operation.signed);
        self.native_fp_path(pc, kind, direct, flags)?;
        // The activation continuation defines fresh SSA inputs.
        let bits = self.scalar_fp_bits(f.rn, width)?;
        let result = self.fp_to_integer_value(
            bits,
            operation.source_64,
            operation.destination_64,
            operation.signed,
        );
        self.write_register(f.rd, result)?;
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
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), first);
        let second = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), second);
        let result = self.float_add_values(first, second, operation.operation);
        self.write_fp_scalar(f.rd, result)?;
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
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), first);
        let second = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), second);
        let result = self.fp_divide_value(first, second);
        self.write_fp_scalar(f.rd, result)?;
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
        self.write_vector(f.rd, result)?;
        Ok(false)
    }

    fn vector_fp_multiply_element(
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
        let (first, second) = self.fp_vector_multiply_element_operands(
            first,
            second,
            lane_bits,
            vector_bits,
            f.fp_element_lane,
        );
        let direct = self.fp_vector_multiply_domain(first, second, lane_bits, self.abi);
        self.native_fp_path(
            pc,
            EdgeKind::VectorFpMultiplyElement(crate::abi::VectorFpMultiplyElementOperation {
                rn: f.rn,
                rm: f.rm,
                rd: f.rd,
                lane_64: lane_bits == 64,
                vector_128: f.vector_128,
                lane: f.fp_element_lane,
            }),
            direct,
            flags,
        )?;
        // Re-read operands after activation establishes fresh SSA inputs.
        let first = self.read_vector(f.rn)?;
        let second = self.read_vector(f.rm)?;
        let (first, second) = self.fp_vector_multiply_element_operands(
            first,
            second,
            lane_bits,
            vector_bits,
            f.fp_element_lane,
        );
        let result = self.fp_vector_multiply_value(first, second, lane_bits, vector_bits);
        self.write_vector(f.rd, result)?;
        Ok(false)
    }

    fn fp_multiply(
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
        let mut direct = self.builder.ins().band(first_ok, second_ok);
        if self.abi == HostAbi::X86_64 {
            let compatible = self.fp_multiply_domain(first, second, width);
            direct = self.builder.ins().band(direct, compatible);
        }
        self.native_fp_path(
            pc,
            EdgeKind::FpMultiply(FpMultiplyOperation {
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
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), first);
        let second = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), second);
        let result = self.fp_multiply_value(
            first,
            second,
            f.float_multiply_operation
                .expect("normalized FP multiply operation"),
        );
        self.write_fp_scalar(f.rd, result)?;
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
        let operation = FpFusedOperation {
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
        let second = self.scalar_fp_bits(f.rm, width)?;
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
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = self.scalar_fp_bits(f.rm, width)?;
        let third = self.scalar_fp_bits(f.ra, width)?;
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), first);
        let second = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), second);
        let third = self
            .builder
            .ins()
            .bitcast(ty, crate::simd_lowering::bitcast_flags(), third);
        let result = self.fp_fused_value(first, second, third, operation.operation);
        self.write_fp_scalar(f.rd, result)?;
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
        self.write_vector(f.rd, result)?;
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
        );
        self.write_fp_scalar(f.rd, result)?;
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
        self.ensure_fp(flags);
        Ok(())
    }

    fn write_fp_scalar(&mut self, rd: u8, result: ir::Value) -> Result<(), Error> {
        let width = self.builder.func.dfg.value_type(result).bits();
        let result = self.builder.ins().bitcast(
            if width == 32 { types::I32 } else { types::I64 },
            crate::simd_lowering::bitcast_flags(),
            result,
        );
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(rd, result)
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
        let bits = self.scalar_fp_bits(f.rn, width)?;
        let value = self.builder.ins().bitcast(
            if width == 32 { types::F32 } else { types::F64 },
            crate::simd_lowering::bitcast_flags(),
            bits,
        );
        let result = self.fp_unary_value(value, kind);
        self.write_fp_scalar(f.rd, result)?;
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
        self.write_vector(f.rd, result)?;
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
        if operation.condition.is_some() {
            // Matches the shared Exact policy for FCCMP/FCCMPE. No successor
            // has been captured; success demands PC+4, failure retains PC.
            self.constant_exit(
                pc,
                pc,
                EdgeKind::FpCompare(operation),
                NativeExitReason::Architectural,
                flags,
            )?;
            return Ok(true);
        }
        let width = scalar_width(f.opc)?;
        let first = self.scalar_fp_bits(f.rn, width)?;
        let second = if let Some(rm) = operation.rm {
            self.scalar_fp_bits(rm, width)?
        } else {
            self.builder
                .ins()
                .iconst(if width == 32 { types::I32 } else { types::I64 }, 0)
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
        *flags = LazyFlags::Packed(self.ordered_fp_compare(first, second, width));
        Ok(false)
    }
}
