//! Shared register-only SIMD lowering. Bit/lane operations do not acquire FP ownership.

use crate::{abi::LazyFlags, jit_error::Error, lowering::IntegerLowering};
use cranelift_codegen::ir::{
    ConstantData, Endianness, InstBuilder, MemFlagsData, Value, condcodes::IntCC,
    immediates::Ieee128, types,
};
use nixe_cpu::decode::a64::fp_simd::{
    BitwiseOperation, Instruction, IntegerComparison, Operands, PairwiseOperation, PermuteOperation,
};
use nixe_cpu::semantics::conditions::Condition;

/// Register-only operations whose semantics do not consume or update FP status.
pub(crate) fn is_register_simd(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::DuplicateGeneral(_)
            | Instruction::DuplicateElement(_)
            | Instruction::ModifiedImmediate(_)
            | Instruction::UnsignedMoveToGeneral(_)
            | Instruction::InsertElement(_)
            | Instruction::InsertGeneral(_)
            | Instruction::MoveToGeneral(_)
            | Instruction::MoveFromGeneral(_)
            | Instruction::ScalarMove(_)
            | Instruction::ScalarAbsolute(_)
            | Instruction::ScalarNegate(_)
            | Instruction::VectorFloatAbsolute(_)
            | Instruction::VectorFloatNegate(_)
            | Instruction::Integer(_)
            | Instruction::Bitwise(_)
            | Instruction::IntegerCompare(_)
            | Instruction::IntegerPairwise(_)
            | Instruction::IntegerMinMax(_)
            | Instruction::PermuteTwoSource(_)
            | Instruction::Extract(_)
            | Instruction::ShiftRightNarrow(_)
            | Instruction::ExtractNarrow(_)
            | Instruction::ScalarShiftRightImmediate(_)
            | Instruction::VectorShiftRightImmediate(_)
            | Instruction::ScalarShiftLeftImmediate(_)
            | Instruction::VectorShiftLeftImmediate(_)
            | Instruction::ShiftLeftLong(_)
            | Instruction::VectorSignedShiftRegister(_)
            | Instruction::VectorUnsignedShiftRegister(_)
            | Instruction::CountBits(_)
            | Instruction::AddAcrossVector(_)
            | Instruction::ScalarFloatImmediate(_)
            | Instruction::VectorFloatImmediate(_)
            | Instruction::ScalarFloatConditionalSelect(_)
    )
}

pub(crate) trait SimdLowering<'a>: IntegerLowering<'a> {
    fn read_vector(&mut self, index: u8) -> Result<Value, Error>;
    fn write_vector(&mut self, index: u8, value: Value) -> Result<(), Error>;
    /// Whether CLIF shuffle lowers without a backend libcall. The legacy
    /// module can resolve libcalls; frameless native units cannot.
    fn use_clif_shuffle(&self) -> bool {
        true
    }

    // Arm DDI 0602: scalar bit transfers and Advanced SIMD lane operations.
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions
    fn emit_register_simd(
        &mut self,
        instruction: Instruction,
        flags: &LazyFlags<Value>,
    ) -> Result<(), Error> {
        let fields = instruction.operands();
        match instruction {
            Instruction::DuplicateGeneral(_) => {
                let lane_bits = 8_u32 << fields.immediate_5.trailing_zeros();
                let lane = integer_lane_type(lane_bits)?;
                let value = self.read_register(fields.rn, false)?;
                let value = cast_integer(self.builder(), value, lane, false);
                let vector_ty = vector_type(lane, lane_bits)?;
                let value = self.builder().ins().splat(vector_ty, value);
                let value = self.finish_vector(value, fields.vector_128);
                self.write_vector(fields.rd, value)
            }
            Instruction::DuplicateElement(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let lane_index = fields.immediate_5 >> (shift + 1);
                let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
                let source = self.read_vector_as(fields.rn, vector_ty)?;
                let lane = self.builder().ins().extractlane(source, lane_index);
                let value = self.builder().ins().splat(vector_ty, lane);
                let value = self.finish_vector(value, fields.vector_128);
                self.write_vector(fields.rd, value)
            }
            Instruction::ModifiedImmediate(_) => {
                let immediate = expand_modified_immediate(
                    fields.cmode,
                    fields.immediate_8,
                    fields.operation_bit,
                )?;
                let bits = u128::from(immediate) | (u128::from(immediate) << 64);
                let immediate = self.vector_constant(bits);
                let value = if fields.cmode <= 11 && fields.cmode & 1 != 0 {
                    let previous = self.read_vector(fields.rd)?;
                    if fields.operation_bit {
                        self.builder().ins().band(previous, immediate)
                    } else {
                        self.builder().ins().bor(previous, immediate)
                    }
                } else {
                    immediate
                };
                let value = self.mask_vector(value, if fields.vector_128 { 128 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            Instruction::UnsignedMoveToGeneral(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let lane_index = fields.immediate_5 >> (shift + 1);
                let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
                let source = self.read_vector_as(fields.rn, vector_ty)?;
                let value = self.builder().ins().extractlane(source, lane_index);
                let value = if fields.vector_128 {
                    cast_integer(self.builder(), value, types::I64, false)
                } else {
                    let value = cast_integer(self.builder(), value, types::I32, false);
                    self.builder().ins().uextend(types::I64, value)
                };
                self.write_register(fields.rd, value)
            }
            Instruction::InsertElement(_) | Instruction::InsertGeneral(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let destination_lane = fields.immediate_5 >> (shift + 1);
                let lane = integer_lane_type(lane_bits)?;
                let vector_ty = vector_type(lane, lane_bits)?;
                let previous = self.read_vector_as(fields.rd, vector_ty)?;
                let value = if matches!(instruction, Instruction::InsertElement(_)) {
                    let source_lane = fields.immediate_4 >> shift;
                    let source = self.read_vector_as(fields.rn, vector_ty)?;
                    self.builder().ins().extractlane(source, source_lane)
                } else {
                    let source = self.read_register(fields.rn, false)?;
                    cast_integer(self.builder(), source, lane, false)
                };
                let value = self
                    .builder()
                    .ins()
                    .insertlane(previous, value, destination_lane);
                let value = self.vector_as(value, types::I8X16);
                self.write_vector(fields.rd, value)
            }
            Instruction::MoveToGeneral(_) => self.emit_move_to_general(fields),
            Instruction::MoveFromGeneral(_) => self.emit_move_from_general(fields),
            Instruction::ScalarMove(_) => {
                let width = scalar_width(fields.opc)?;
                let value = self.read_vector(fields.rn)?;
                let value = self.mask_vector(value, width);
                self.write_vector(fields.rd, value)
            }
            Instruction::ScalarAbsolute(_) | Instruction::ScalarNegate(_) => {
                let width = scalar_width(fields.opc)?;
                let source = self.read_vector(fields.rn)?;
                let source = self.mask_vector(source, width);
                let sign = self.vector_constant(1_u128 << (width - 1));
                let value = if matches!(instruction, Instruction::ScalarNegate(_)) {
                    self.builder().ins().bxor(source, sign)
                } else {
                    let sign = self.builder().ins().bnot(sign);
                    self.builder().ins().band(source, sign)
                };
                self.write_vector(fields.rd, value)
            }
            Instruction::VectorFloatAbsolute(_) | Instruction::VectorFloatNegate(_) => {
                let lane_bits = if fields.opc & 1 == 0 { 32 } else { 64 };
                let vector_bits = if fields.vector_128 { 128 } else { 64 };
                let mut sign_bits = 0_u128;
                for offset in (0..vector_bits).step_by(lane_bits as usize) {
                    sign_bits |= 1_u128 << (offset + lane_bits - 1);
                }
                let sign = self.vector_constant(sign_bits);
                let source = self.read_vector(fields.rn)?;
                let source = self.mask_vector(source, vector_bits);
                let value = if matches!(instruction, Instruction::VectorFloatNegate(_)) {
                    self.builder().ins().bxor(source, sign)
                } else {
                    let sign = self.builder().ins().bnot(sign);
                    self.builder().ins().band(source, sign)
                };
                self.write_vector(fields.rd, value)
            }
            Instruction::Integer(_) => self.emit_integer_vector(fields),
            Instruction::Bitwise(_) => self.emit_bitwise(fields),
            Instruction::IntegerCompare(_) => self.emit_integer_compare(fields),
            Instruction::IntegerPairwise(_) => self.emit_integer_pairwise(fields),
            Instruction::IntegerMinMax(_) => self.emit_integer_min_max(fields),
            Instruction::PermuteTwoSource(_) => self.emit_permute(fields),
            Instruction::Extract(_) => self.emit_vector_extract(fields),
            Instruction::ShiftRightNarrow(_) | Instruction::ExtractNarrow(_) => {
                self.emit_narrow(instruction, fields)
            }
            Instruction::ScalarShiftRightImmediate(_)
            | Instruction::VectorShiftRightImmediate(_)
            | Instruction::ScalarShiftLeftImmediate(_)
            | Instruction::VectorShiftLeftImmediate(_) => {
                self.emit_immediate_shift(instruction, fields)
            }
            Instruction::ShiftLeftLong(_) => self.emit_shift_left_long(fields),
            Instruction::VectorSignedShiftRegister(_)
            | Instruction::VectorUnsignedShiftRegister(_) => {
                self.emit_register_shift(instruction, fields)
            }
            Instruction::CountBits(_) => {
                let source = self.read_vector(fields.rn)?;
                let value = self.builder().ins().popcnt(source);
                let value = self.mask_vector(value, if fields.vector_128 { 128 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            Instruction::AddAcrossVector(_) => self.emit_add_across(fields),
            Instruction::ScalarFloatImmediate(_) | Instruction::VectorFloatImmediate(_) => {
                self.emit_float_immediate(instruction, fields)
            }
            Instruction::ScalarFloatConditionalSelect(_) => {
                let predicate =
                    self.emit_condition(Condition::from_encoding(fields.condition), flags);
                let first = self.read_vector(fields.rn)?;
                let second = self.read_vector(fields.rm)?;
                let selected = self.builder().ins().select(predicate, first, second);
                let value = self.mask_vector(selected, if fields.opc == 0 { 32 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            _ => Err(Error::internal("instruction is not register-only SIMD")),
        }
    }

    fn vector_as(&mut self, value: Value, ty: cranelift_codegen::ir::Type) -> Value {
        if self.builder().func.dfg.value_type(value) == ty {
            value
        } else {
            self.builder().ins().bitcast(ty, bitcast_flags(), value)
        }
    }

    fn read_vector_as(
        &mut self,
        index: u8,
        ty: cranelift_codegen::ir::Type,
    ) -> Result<Value, Error> {
        let value = self.read_vector(index)?;
        Ok(self.vector_as(value, ty))
    }

    fn vector_constant(&mut self, value: u128) -> Value {
        let constant = self
            .builder()
            .func
            .dfg
            .constants
            .insert(Ieee128::with_bits(value).into());
        self.builder().ins().vconst(types::I8X16, constant)
    }

    fn shuffle_bytes(&mut self, first: Value, second: Value, mask: [u8; 16]) -> Value {
        if !self.use_clif_shuffle() {
            // x86 SSE2 has no PSHUFB. Constant lane extraction/insertion keeps
            // this operation native without raising the host requirement to
            // SSSE3 or allowing a backend-created call across the Nixe ABI.
            let mut result = self.vector_constant(0);
            for (index, source) in mask.into_iter().enumerate() {
                if source < 32 {
                    let lane = self
                        .builder()
                        .ins()
                        .extractlane(if source < 16 { first } else { second }, source % 16);
                    result = self.builder().ins().insertlane(result, lane, index as u8);
                }
            }
            return result;
        }
        let mask = self
            .builder()
            .func
            .dfg
            .immediates
            .push(ConstantData::from(mask.as_slice()));
        self.builder().ins().shuffle(first, second, mask)
    }

    fn mask_vector(&mut self, value: Value, bits: u32) -> Value {
        if bits == 128 {
            return value;
        }
        let mask = self.vector_constant((1_u128 << bits) - 1);
        self.builder().ins().band(value, mask)
    }

    fn finish_vector(&mut self, value: Value, full_width: bool) -> Value {
        let value = self.vector_as(value, types::I8X16);
        if full_width {
            value
        } else {
            self.mask_vector(value, 64)
        }
    }

    fn emit_integer_vector(&mut self, fields: Operands) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let rhs = self.read_vector_as(fields.rm, vector_ty)?;
        let result = if fields.subtract {
            self.builder().ins().isub(lhs, rhs)
        } else {
            self.builder().ins().iadd(lhs, rhs)
        };
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_bitwise(&mut self, fields: Operands) -> Result<(), Error> {
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let result = match fields
            .bitwise_operation
            .expect("normalized SIMD bitwise operation")
        {
            BitwiseOperation::And => self.builder().ins().band(first, second),
            BitwiseOperation::BitClear => {
                let not_second = self.builder().ins().bnot(second);
                self.builder().ins().band(first, not_second)
            }
            BitwiseOperation::Or => self.builder().ins().bor(first, second),
            BitwiseOperation::OrNot => {
                let not_second = self.builder().ins().bnot(second);
                self.builder().ins().bor(first, not_second)
            }
            BitwiseOperation::ExclusiveOr => self.builder().ins().bxor(first, second),
            BitwiseOperation::Select => {
                let destination = self.read_vector(fields.rd)?;
                self.builder().ins().bitselect(destination, first, second)
            }
            BitwiseOperation::InsertIfTrue => {
                let destination = self.read_vector(fields.rd)?;
                self.builder().ins().bitselect(second, first, destination)
            }
            BitwiseOperation::InsertIfFalse => {
                let destination = self.read_vector(fields.rd)?;
                self.builder().ins().bitselect(second, destination, first)
            }
        };
        let result = self.mask_vector(result, if fields.vector_128 { 128 } else { 64 });
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_compare(&mut self, fields: Operands) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let zero = self.builder().ins().iconst(lane, 0);
        let zero = self.builder().ins().splat(vector_ty, zero);
        let rhs = if fields.compare_with_zero {
            zero
        } else {
            self.read_vector_as(fields.rm, vector_ty)?
        };
        let comparison = fields
            .integer_comparison
            .expect("normalized SIMD comparison");
        let result = match comparison {
            IntegerComparison::NonzeroBitTest => {
                let bits = self.builder().ins().band(lhs, rhs);
                self.builder().ins().icmp(IntCC::NotEqual, bits, zero)
            }
            comparison => {
                let condition = match comparison {
                    IntegerComparison::SignedGreaterThan => IntCC::SignedGreaterThan,
                    IntegerComparison::UnsignedGreaterThan => IntCC::UnsignedGreaterThan,
                    IntegerComparison::SignedGreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
                    IntegerComparison::UnsignedGreaterThanOrEqual => {
                        IntCC::UnsignedGreaterThanOrEqual
                    }
                    IntegerComparison::SignedLessThan => IntCC::SignedLessThan,
                    IntegerComparison::SignedLessThanOrEqual => IntCC::SignedLessThanOrEqual,
                    IntegerComparison::Equal => IntCC::Equal,
                    IntegerComparison::NonzeroBitTest => unreachable!(),
                };
                self.builder().ins().icmp(condition, lhs, rhs)
            }
        };
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_pairwise(&mut self, fields: Operands) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let lanes = (if fields.vector_128 { 128 } else { 64 }) / lane_bits;
        let lane_bytes = lane_bits / 8;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let mut left_mask = [0_u8; 16];
        let mut right_mask = [0_u8; 16];
        for destination in 0..lanes {
            let (source_base, source_lane) = if destination < lanes / 2 {
                (0, destination * 2)
            } else {
                (16, (destination - lanes / 2) * 2)
            };
            for byte in 0..lane_bytes {
                let output = (destination * lane_bytes + byte) as usize;
                left_mask[output] = (source_base + source_lane * lane_bytes + byte) as u8;
                right_mask[output] = left_mask[output] + lane_bytes as u8;
            }
        }
        let left = self.shuffle_bytes(first, second, left_mask);
        let right = self.shuffle_bytes(first, second, right_mask);
        let left = self.vector_as(left, vector_ty);
        let right = self.vector_as(right, vector_ty);
        let operation = fields
            .pairwise_operation
            .expect("normalized pairwise operation");
        let result = self.select_pairwise_vector(left, right, operation);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_min_max(&mut self, fields: Operands) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let rhs = self.read_vector_as(fields.rm, vector_ty)?;
        let operation = fields
            .pairwise_operation
            .expect("normalized min/max operation");
        let result = self.select_pairwise_vector(lhs, rhs, operation);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn select_pairwise_vector(
        &mut self,
        lhs: Value,
        rhs: Value,
        operation: PairwiseOperation,
    ) -> Value {
        match operation {
            PairwiseOperation::Add => self.builder().ins().iadd(lhs, rhs),
            operation => {
                let condition = match operation {
                    PairwiseOperation::SignedMaximum => IntCC::SignedGreaterThanOrEqual,
                    PairwiseOperation::SignedMinimum => IntCC::SignedLessThanOrEqual,
                    PairwiseOperation::UnsignedMaximum => IntCC::UnsignedGreaterThanOrEqual,
                    PairwiseOperation::UnsignedMinimum => IntCC::UnsignedLessThanOrEqual,
                    PairwiseOperation::Add => unreachable!(),
                };
                let mask = self.builder().ins().icmp(condition, lhs, rhs);
                self.builder().ins().bitselect(mask, lhs, rhs)
            }
        }
    }

    fn emit_permute(&mut self, fields: Operands) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let lane_count = (if fields.vector_128 { 128 } else { 64 }) / lane_bits;
        let lane_bytes = lane_bits / 8;
        let half = lane_count / 2;
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let operation = fields
            .permute_operation
            .expect("normalized SIMD permutation");
        let mut mask = [0_u8; 16];
        for destination in 0..lane_count {
            let (source_base, lane) = match operation {
                PermuteOperation::UnzipPrimary | PermuteOperation::UnzipSecondary => {
                    let odd = u32::from(matches!(operation, PermuteOperation::UnzipSecondary));
                    if destination < half {
                        (0, destination * 2 + odd)
                    } else {
                        (16, (destination - half) * 2 + odd)
                    }
                }
                PermuteOperation::TransposePrimary | PermuteOperation::TransposeSecondary => {
                    let odd = u32::from(matches!(operation, PermuteOperation::TransposeSecondary));
                    (
                        if destination & 1 == 0 { 0 } else { 16 },
                        (destination / 2) * 2 + odd,
                    )
                }
                PermuteOperation::ZipPrimary | PermuteOperation::ZipSecondary => {
                    let upper = u32::from(matches!(operation, PermuteOperation::ZipSecondary));
                    (
                        if destination & 1 == 0 { 0 } else { 16 },
                        destination / 2 + upper * half,
                    )
                }
            };
            for byte in 0..lane_bytes {
                mask[(destination * lane_bytes + byte) as usize] =
                    (source_base + lane * lane_bytes + byte) as u8;
            }
        }
        let result = self.shuffle_bytes(first, second, mask);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_vector_extract(&mut self, fields: Operands) -> Result<(), Error> {
        let count = if fields.vector_128 { 16 } else { 8 };
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let mut mask = [0_u8; 16];
        for destination in 0..count {
            let source = destination + u32::from(fields.immediate_4);
            mask[destination as usize] = if source < count {
                source as u8
            } else {
                (16 + source - count) as u8
            };
        }
        let result = self.shuffle_bytes(first, second, mask);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_narrow(&mut self, instruction: Instruction, fields: Operands) -> Result<(), Error> {
        let (destination_bits, shift) = if matches!(instruction, Instruction::ShiftRightNarrow(_)) {
            let high = u32::from(fields.shift_immediate >> 3);
            let destination = 8_u32 << (31 - high.leading_zeros());
            (
                destination,
                destination * 2 - u32::from(fields.shift_immediate),
            )
        } else {
            (8_u32 << fields.opc, 0)
        };
        let source_bits = destination_bits * 2;
        let lane_count = 128 / source_bits;
        let source_ty = vector_type(integer_lane_type(source_bits)?, source_bits)?;
        let mut source = self.read_vector_as(fields.rn, source_ty)?;
        if shift != 0 {
            source = self.builder().ins().ushr_imm_u(source, i64::from(shift));
        }
        let source = self.vector_as(source, types::I8X16);
        let source_bytes = source_bits / 8;
        let destination_bytes = destination_bits / 8;
        let mut packed_mask = [0_u8; 16];
        for lane in 0..lane_count {
            for byte in 0..destination_bytes {
                packed_mask[(lane * destination_bytes + byte) as usize] =
                    (lane * source_bytes + byte) as u8;
            }
        }
        let packed = self.shuffle_bytes(source, source, packed_mask);
        let result = if fields.vector_128 {
            let previous = self.read_vector(fields.rd)?;
            let mut upper_mask = [0_u8; 16];
            for byte in 0..8 {
                upper_mask[byte] = byte as u8;
                upper_mask[byte + 8] = (16 + byte) as u8;
            }
            self.shuffle_bytes(previous, packed, upper_mask)
        } else {
            self.mask_vector(packed, 64)
        };
        self.write_vector(fields.rd, result)
    }

    fn emit_immediate_shift(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), Error> {
        let immediate = u32::from(fields.shift_immediate);
        let high = immediate >> 3;
        let lane_bits = 8_u32 << (31 - high.leading_zeros());
        let right = matches!(
            instruction,
            Instruction::ScalarShiftRightImmediate(_) | Instruction::VectorShiftRightImmediate(_)
        );
        let scalar = matches!(
            instruction,
            Instruction::ScalarShiftRightImmediate(_) | Instruction::ScalarShiftLeftImmediate(_)
        );
        let shift = if right {
            2 * lane_bits - immediate
        } else {
            immediate - lane_bits
        };
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let source = self.read_vector_as(fields.rn, vector_ty)?;
        let result = if right && shift == lane_bits && !fields.operation_bit {
            self.builder()
                .ins()
                .sshr_imm_u(source, i64::from(lane_bits - 1))
        } else if right && shift == lane_bits {
            let zero = self.builder().ins().iconst(lane, 0);
            self.builder().ins().splat(vector_ty, zero)
        } else if right && !fields.operation_bit {
            self.builder().ins().sshr_imm_u(source, i64::from(shift))
        } else if right {
            self.builder().ins().ushr_imm_u(source, i64::from(shift))
        } else {
            self.builder().ins().ishl_imm_u(source, i64::from(shift))
        };
        let result = self.finish_vector(result, !scalar && fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_shift_left_long(&mut self, fields: Operands) -> Result<(), Error> {
        let immediate = u32::from(fields.shift_immediate);
        let high = immediate >> 3;
        let source_bits = 8_u32 << (31 - high.leading_zeros());
        let destination_bits = source_bits * 2;
        let shift = immediate - source_bits;
        let lane_count = 64 / source_bits;
        let source_lane = integer_lane_type(source_bits)?;
        let destination_lane = integer_lane_type(destination_bits)?;
        let source_ty = vector_type(source_lane, source_bits)?;
        let destination_ty = vector_type(destination_lane, destination_bits)?;
        let source = self.read_vector_as(fields.rn, source_ty)?;
        let zero = self.builder().ins().iconst(destination_lane, 0);
        let mut result = self.builder().ins().splat(destination_ty, zero);
        let first = if fields.vector_128 { lane_count } else { 0 };
        for index in 0..lane_count {
            let value = self
                .builder()
                .ins()
                .extractlane(source, (first + index) as u8);
            let value = if fields.operation_bit {
                self.builder().ins().uextend(destination_lane, value)
            } else {
                self.builder().ins().sextend(destination_lane, value)
            };
            let value = if shift == 0 {
                value
            } else {
                self.builder().ins().ishl_imm_u(value, i64::from(shift))
            };
            result = self.builder().ins().insertlane(result, value, index as u8);
        }
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)
    }

    fn emit_register_shift(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let values = self.read_vector_as(fields.rn, vector_ty)?;
        let mut distance = self.read_vector_as(fields.rm, vector_ty)?;
        let zero = self.builder().ins().iconst(lane, 0);
        let zero = self.builder().ins().splat(vector_ty, zero);
        if lane_bits > 8 {
            let low_byte = self.builder().ins().iconst(lane, 0xff);
            let low_byte = self.builder().ins().splat(vector_ty, low_byte);
            distance = self.builder().ins().band(distance, low_byte);
            distance = self
                .builder()
                .ins()
                .ishl_imm_u(distance, i64::from(lane_bits - 8));
            distance = self
                .builder()
                .ins()
                .sshr_imm_u(distance, i64::from(lane_bits - 8));
        }
        let nonnegative =
            self.builder()
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, distance, zero);
        let negative = self.builder().ins().ineg(distance);
        let magnitude = self
            .builder()
            .ins()
            .bitselect(nonnegative, distance, negative);
        let signed = matches!(instruction, Instruction::VectorSignedShiftRegister(_));
        let mut left = values;
        let mut right = values;
        let mut amount = 1_u32;
        while amount < lane_bits {
            let bit = self.builder().ins().iconst(lane, i64::from(amount));
            let bit = self.builder().ins().splat(vector_ty, bit);
            let selected = self.builder().ins().band(magnitude, bit);
            let selected = self.builder().ins().icmp(IntCC::NotEqual, selected, zero);
            let shifted_left = self.builder().ins().ishl_imm_u(left, i64::from(amount));
            let shifted_right = if signed {
                self.builder().ins().sshr_imm_u(right, i64::from(amount))
            } else {
                self.builder().ins().ushr_imm_u(right, i64::from(amount))
            };
            left = self.builder().ins().bitselect(selected, shifted_left, left);
            right = self
                .builder()
                .ins()
                .bitselect(selected, shifted_right, right);
            amount *= 2;
        }
        let width = self.builder().ins().iconst(lane, i64::from(lane_bits));
        let width = self.builder().ins().splat(vector_ty, width);
        let out = self
            .builder()
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, magnitude, width);
        let fill = if signed {
            self.builder()
                .ins()
                .sshr_imm_u(values, i64::from(lane_bits - 1))
        } else {
            zero
        };
        left = self.builder().ins().bitselect(out, zero, left);
        right = self.builder().ins().bitselect(out, fill, right);
        let result = self.builder().ins().bitselect(nonnegative, left, right);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_add_across(&mut self, fields: Operands) -> Result<(), Error> {
        let lane_bits = 8_u32 << fields.opc;
        let lane_count = (if fields.vector_128 { 128 } else { 64 }) / lane_bits;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let mut value = self.read_vector_as(fields.rn, vector_ty)?;
        let lane_bytes = lane_bits / 8;
        let mut distance = lane_count / 2;
        while distance != 0 {
            let bytes = self.vector_as(value, types::I8X16);
            let mut mask = [0_u8; 16];
            for destination in 0..lane_count {
                let source = if destination < distance {
                    destination + distance
                } else {
                    destination
                };
                for byte in 0..lane_bytes {
                    mask[(destination * lane_bytes + byte) as usize] =
                        (source * lane_bytes + byte) as u8;
                }
            }
            let paired = self.shuffle_bytes(bytes, bytes, mask);
            let paired = self.vector_as(paired, vector_ty);
            value = self.builder().ins().iadd(value, paired);
            distance /= 2;
        }
        let result = self.builder().ins().extractlane(value, 0);
        let result = self.builder().ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)
    }

    fn emit_float_immediate(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), Error> {
        let value = if matches!(instruction, Instruction::ScalarFloatImmediate(_)) {
            let (exponent, fraction) = match fields.opc {
                0 => (8, 23),
                1 => (11, 52),
                3 => (5, 10),
                _ => return Err(Error::invalid("invalid scalar FP immediate width")),
            };
            u128::from(expand_vfp_immediate(
                fields.fp_immediate_8,
                exponent,
                fraction,
            ))
        } else {
            let (lane, bits) = if fields.operation_bit {
                (expand_vfp_immediate(fields.immediate_8, 11, 52), 64)
            } else {
                (expand_vfp_immediate(fields.immediate_8, 8, 23), 32)
            };
            if bits == 64 {
                u128::from(lane) | (u128::from(lane) << 64)
            } else {
                let lane = u128::from(lane as u32);
                lane | lane << 32 | lane << 64 | lane << 96
            }
        };
        let value = self.vector_constant(value);
        let value = self.mask_vector(
            value,
            if fields.vector_128 || matches!(instruction, Instruction::ScalarFloatImmediate(_)) {
                128
            } else {
                64
            },
        );
        self.write_vector(fields.rd, value)
    }

    fn emit_move_to_general(&mut self, fields: Operands) -> Result<(), Error> {
        let vector = self.read_vector_as(fields.rn, types::I128)?;
        let (width, value) = match (fields.size & 2 != 0, fields.opc) {
            (false, 0) => (32, self.builder().ins().ireduce(types::I32, vector)),
            (false, 3) => (32, self.builder().ins().ireduce(types::I16, vector)),
            (true, 1) => (64, self.builder().ins().ireduce(types::I64, vector)),
            (true, 2) => {
                let value = self.builder().ins().ushr_imm_u(vector, 64);
                (64, self.builder().ins().ireduce(types::I64, value))
            }
            _ => return Err(Error::invalid("invalid FMOV general width")),
        };
        let value = cast_integer(self.builder(), value, types::I64, false);
        let _ = width;
        self.write_register(fields.rd, value)
    }

    fn emit_move_from_general(&mut self, fields: Operands) -> Result<(), Error> {
        let value = self.read_register(fields.rn, false)?;
        let general_64 = fields.size & 2 != 0;
        let value = if general_64 {
            value
        } else {
            self.builder().ins().ireduce(types::I32, value)
        };
        let value = cast_integer(self.builder(), value, types::I128, false);
        let value = match (general_64, fields.opc) {
            (false, 0) => value,
            (false, 3) => {
                let value = self.builder().ins().ireduce(types::I16, value);
                self.builder().ins().uextend(types::I128, value)
            }
            (true, 1) => value,
            (true, 2) => {
                let previous = self.read_vector_as(fields.rd, types::I128)?;
                let low = self.builder().ins().ireduce(types::I64, previous);
                let low = self.builder().ins().uextend(types::I128, low);
                let high = self.builder().ins().ishl_imm_u(value, 64);
                self.builder().ins().bor(low, high)
            }
            _ => return Err(Error::invalid("invalid FMOV general width")),
        };
        let value = self.vector_as(value, types::I8X16);
        self.write_vector(fields.rd, value)
    }
}

pub(crate) fn bitcast_flags() -> MemFlagsData {
    MemFlagsData::new().with_endianness(Endianness::Little)
}

pub(crate) fn integer_lane_type(bits: u32) -> Result<cranelift_codegen::ir::Type, Error> {
    match bits {
        8 => Ok(types::I8),
        16 => Ok(types::I16),
        32 => Ok(types::I32),
        64 => Ok(types::I64),
        _ => Err(Error::invalid("invalid SIMD lane width")),
    }
}

pub(crate) fn vector_type(
    lane: cranelift_codegen::ir::Type,
    lane_bits: u32,
) -> Result<cranelift_codegen::ir::Type, Error> {
    lane.by(128 / lane_bits)
        .ok_or_else(|| Error::unsupported("host CLIF lacks required SIMD shape"))
}

pub(crate) fn cast_integer(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    ty: cranelift_codegen::ir::Type,
    signed: bool,
) -> Value {
    let from = builder.func.dfg.value_type(value);
    if from == ty {
        value
    } else if from.bits() > ty.bits() {
        builder.ins().ireduce(ty, value)
    } else if signed {
        builder.ins().sextend(ty, value)
    } else {
        builder.ins().uextend(ty, value)
    }
}

pub(crate) fn scalar_width(opc: u8) -> Result<u32, Error> {
    match opc {
        0 => Ok(32),
        1 => Ok(64),
        3 => Ok(16),
        _ => Err(Error::invalid("invalid scalar FP width")),
    }
}

fn expand_modified_immediate(cmode: u8, immediate: u8, operation_bit: bool) -> Result<u64, Error> {
    let immediate = u64::from(immediate);
    let value = match cmode {
        0..=7 => {
            let lane = immediate << ((cmode >> 1) * 8);
            lane | lane << 32
        }
        8..=11 => {
            let lane = immediate << (((cmode >> 1) & 1) * 8);
            lane | lane << 16 | lane << 32 | lane << 48
        }
        12 => {
            let lane = immediate << 8 | 0xff;
            lane | lane << 32
        }
        13 => {
            let lane = immediate << 16 | 0xffff;
            lane | lane << 32
        }
        14 if !operation_bit => immediate * 0x0101_0101_0101_0101,
        14 => {
            let mut result = 0;
            for bit in 0..8 {
                if immediate & (1 << bit) != 0 {
                    result |= 0xff << (bit * 8);
                }
            }
            result
        }
        _ => return Err(Error::invalid("invalid SIMD modified immediate")),
    };
    Ok(if operation_bit && cmode != 14 {
        !value
    } else {
        value
    })
}

fn expand_vfp_immediate(immediate: u8, exponent_bits: u32, fraction_bits: u32) -> u64 {
    let sign = u64::from(immediate >> 7);
    let control = u64::from((immediate >> 6) & 1);
    let tail = u64::from((immediate >> 4) & 3);
    let fraction = u64::from(immediate & 0xf);
    let sign_shift = exponent_bits + fraction_bits;
    let repeated = if control == 0 {
        0
    } else {
        (1_u64 << (exponent_bits - 3)) - 1
    };
    sign << sign_shift
        | (control ^ 1) << (sign_shift - 1)
        | repeated << (fraction_bits + 2)
        | tail << fraction_bits
        | fraction << (fraction_bits - 4)
}
