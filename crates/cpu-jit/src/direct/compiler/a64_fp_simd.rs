use std::mem::offset_of;

use cranelift_codegen::ir::{
    AbiParam, ConstantData, Endianness, InstBuilder, MemFlagsData, Signature, Value,
    condcodes::IntCC, immediates::Ieee128, types,
};
use nixe_cpu::decode::a64::fp_simd::{
    BitwiseOperation, FloatAddOperation, FloatConversion, FloatFusedMultiplyOperation,
    FloatMultiplyOperation, FloatRoundOperation, FloatToIntegerRounding, Instruction,
    IntegerComparison, Operands, PairwiseOperation, PermuteOperation,
};
use nixe_cpu::memory::{MemoryAccessSize, MemoryOrdering};
use nixe_cpu::semantics::conditions::Condition;
use nixe_memory::GuestVirtualAddress;

use super::{CraneliftTranslator, LazyFlags, a64_memory::MemoryOperation};
use crate::direct::slow;
use crate::direct::{DirectJitError, NativeContext};

impl CraneliftTranslator<'_, '_> {
    // A64 Advanced SIMD and floating-point behavior follows Arm DDI 0602
    // (2025-12). Pure lane and bit operations are emitted directly; only FP
    // behavior whose FPCR/FPSR contract is not represented by CLIF crosses an
    // exact typed slow boundary.
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions
    pub(super) fn emit_fp_simd(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let fields = instruction.operands();
        match instruction {
            Instruction::DuplicateGeneral(_) => {
                let lane_bits = 8_u32 << fields.immediate_5.trailing_zeros();
                let lane = integer_lane_type(lane_bits)?;
                let value = self.read_register(fields.rn, false)?;
                let value = cast_integer(&mut self.builder, value, lane, false);
                let vector_ty = vector_type(lane, lane_bits)?;
                let value = self.builder.ins().splat(vector_ty, value);
                let value = self.finish_vector(value, fields.vector_128);
                self.write_vector(fields.rd, value)
            }
            Instruction::DuplicateElement(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let lane_index = fields.immediate_5 >> (shift + 1);
                let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
                let source = self.read_vector_as(fields.rn, vector_ty)?;
                let lane = self.builder.ins().extractlane(source, lane_index);
                let value = self.builder.ins().splat(vector_ty, lane);
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
                        self.builder.ins().band(previous, immediate)
                    } else {
                        self.builder.ins().bor(previous, immediate)
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
                let value = self.builder.ins().extractlane(source, lane_index);
                let value = if fields.vector_128 {
                    cast_integer(&mut self.builder, value, types::I64, false)
                } else {
                    let value = cast_integer(&mut self.builder, value, types::I32, false);
                    self.builder.ins().uextend(types::I64, value)
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
                    self.builder.ins().extractlane(source, source_lane)
                } else {
                    let source = self.read_register(fields.rn, false)?;
                    cast_integer(&mut self.builder, source, lane, false)
                };
                let value = self
                    .builder
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
                    self.builder.ins().bxor(source, sign)
                } else {
                    let sign = self.builder.ins().bnot(sign);
                    self.builder.ins().band(source, sign)
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
                    self.builder.ins().bxor(source, sign)
                } else {
                    let sign = self.builder.ins().bnot(sign);
                    self.builder.ins().band(source, sign)
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
                let value = self.builder.ins().popcnt(source);
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
                let selected = self.builder.ins().select(predicate, first, second);
                let value = self.mask_vector(selected, if fields.opc == 0 { 32 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            Instruction::MemoryUnsigned(_)
            | Instruction::MemoryUnscaled(_)
            | Instruction::MemoryPostIndex(_)
            | Instruction::MemoryPreIndex(_)
            | Instruction::MemoryRegister(_)
            | Instruction::MemoryPair(_)
            | Instruction::MemoryMultipleStructures(_)
            | Instruction::MemoryMultipleStructuresPostIndex(_)
            | Instruction::MemorySingleStructure(_)
            | Instruction::MemorySingleStructurePostIndex(_) => {
                self.emit_vector_memory(source, instruction, fields, flags)
            }
            _ => self.emit_exact_fp(source, instruction, fields, flags),
        }
    }

    fn read_vector(&mut self, index: u8) -> Result<Value, DirectJitError> {
        let variable = self.vector_registers[usize::from(index)].ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT vector V{index} was not planned"))
        })?;
        Ok(self.builder.use_var(variable))
    }

    fn write_vector(&mut self, index: u8, value: Value) -> Result<(), DirectJitError> {
        let variable = self.vector_registers[usize::from(index)].ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT vector V{index} was not planned"))
        })?;
        self.builder.def_var(variable, value);
        self.block_dirty_vector_registers[usize::from(index)] = true;
        Ok(())
    }

    fn vector_as(&mut self, value: Value, ty: cranelift_codegen::ir::Type) -> Value {
        if self.builder.func.dfg.value_type(value) == ty {
            value
        } else {
            self.builder.ins().bitcast(ty, bitcast_flags(), value)
        }
    }

    fn read_vector_as(
        &mut self,
        index: u8,
        ty: cranelift_codegen::ir::Type,
    ) -> Result<Value, DirectJitError> {
        let value = self.read_vector(index)?;
        Ok(self.vector_as(value, ty))
    }

    fn vector_constant(&mut self, value: u128) -> Value {
        let constant = self
            .builder
            .func
            .dfg
            .constants
            .insert(Ieee128::with_bits(value).into());
        self.builder.ins().vconst(types::I8X16, constant)
    }

    fn shuffle_bytes(&mut self, first: Value, second: Value, mask: [u8; 16]) -> Value {
        let mask = self
            .builder
            .func
            .dfg
            .immediates
            .push(ConstantData::from(mask.as_slice()));
        self.builder.ins().shuffle(first, second, mask)
    }

    fn mask_vector(&mut self, value: Value, bits: u32) -> Value {
        if bits == 128 {
            return value;
        }
        let mask = self.vector_constant((1_u128 << bits) - 1);
        self.builder.ins().band(value, mask)
    }

    fn finish_vector(&mut self, value: Value, full_width: bool) -> Value {
        let value = self.vector_as(value, types::I8X16);
        if full_width {
            value
        } else {
            self.mask_vector(value, 64)
        }
    }

    fn emit_integer_vector(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let rhs = self.read_vector_as(fields.rm, vector_ty)?;
        let result = if fields.subtract {
            self.builder.ins().isub(lhs, rhs)
        } else {
            self.builder.ins().iadd(lhs, rhs)
        };
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_bitwise(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let destination = self.read_vector(fields.rd)?;
        let result = match fields
            .bitwise_operation
            .expect("normalized SIMD bitwise operation")
        {
            BitwiseOperation::And => self.builder.ins().band(first, second),
            BitwiseOperation::BitClear => {
                let not_second = self.builder.ins().bnot(second);
                self.builder.ins().band(first, not_second)
            }
            BitwiseOperation::Or => self.builder.ins().bor(first, second),
            BitwiseOperation::OrNot => {
                let not_second = self.builder.ins().bnot(second);
                self.builder.ins().bor(first, not_second)
            }
            BitwiseOperation::ExclusiveOr => self.builder.ins().bxor(first, second),
            BitwiseOperation::Select => self.builder.ins().bitselect(destination, first, second),
            BitwiseOperation::InsertIfTrue => {
                self.builder.ins().bitselect(second, first, destination)
            }
            BitwiseOperation::InsertIfFalse => {
                self.builder.ins().bitselect(second, destination, first)
            }
        };
        let result = self.mask_vector(result, if fields.vector_128 { 128 } else { 64 });
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_compare(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let zero = self.builder.ins().iconst(lane, 0);
        let zero = self.builder.ins().splat(vector_ty, zero);
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
                let bits = self.builder.ins().band(lhs, rhs);
                self.builder.ins().icmp(IntCC::NotEqual, bits, zero)
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
                self.builder.ins().icmp(condition, lhs, rhs)
            }
        };
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_pairwise(&mut self, fields: Operands) -> Result<(), DirectJitError> {
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

    fn emit_integer_min_max(&mut self, fields: Operands) -> Result<(), DirectJitError> {
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
            PairwiseOperation::Add => self.builder.ins().iadd(lhs, rhs),
            operation => {
                let condition = match operation {
                    PairwiseOperation::SignedMaximum => IntCC::SignedGreaterThanOrEqual,
                    PairwiseOperation::SignedMinimum => IntCC::SignedLessThanOrEqual,
                    PairwiseOperation::UnsignedMaximum => IntCC::UnsignedGreaterThanOrEqual,
                    PairwiseOperation::UnsignedMinimum => IntCC::UnsignedLessThanOrEqual,
                    PairwiseOperation::Add => unreachable!(),
                };
                let mask = self.builder.ins().icmp(condition, lhs, rhs);
                self.builder.ins().bitselect(mask, lhs, rhs)
            }
        }
    }

    fn emit_permute(&mut self, fields: Operands) -> Result<(), DirectJitError> {
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

    fn emit_vector_extract(&mut self, fields: Operands) -> Result<(), DirectJitError> {
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

    fn emit_narrow(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
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
            source = self.builder.ins().ushr_imm_u(source, i64::from(shift));
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
    ) -> Result<(), DirectJitError> {
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
            self.builder
                .ins()
                .sshr_imm_u(source, i64::from(lane_bits - 1))
        } else if right && shift == lane_bits {
            let zero = self.builder.ins().iconst(lane, 0);
            self.builder.ins().splat(vector_ty, zero)
        } else if right && !fields.operation_bit {
            self.builder.ins().sshr_imm_u(source, i64::from(shift))
        } else if right {
            self.builder.ins().ushr_imm_u(source, i64::from(shift))
        } else {
            self.builder.ins().ishl_imm_u(source, i64::from(shift))
        };
        let result = self.finish_vector(result, !scalar && fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_shift_left_long(&mut self, fields: Operands) -> Result<(), DirectJitError> {
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
        let zero = self.builder.ins().iconst(destination_lane, 0);
        let mut result = self.builder.ins().splat(destination_ty, zero);
        let first = if fields.vector_128 { lane_count } else { 0 };
        for index in 0..lane_count {
            let value = self
                .builder
                .ins()
                .extractlane(source, (first + index) as u8);
            let value = if fields.operation_bit {
                self.builder.ins().uextend(destination_lane, value)
            } else {
                self.builder.ins().sextend(destination_lane, value)
            };
            let value = if shift == 0 {
                value
            } else {
                self.builder.ins().ishl_imm_u(value, i64::from(shift))
            };
            result = self.builder.ins().insertlane(result, value, index as u8);
        }
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)
    }

    fn emit_register_shift(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let values = self.read_vector_as(fields.rn, vector_ty)?;
        let mut distance = self.read_vector_as(fields.rm, vector_ty)?;
        let zero = self.builder.ins().iconst(lane, 0);
        let zero = self.builder.ins().splat(vector_ty, zero);
        if lane_bits > 8 {
            let low_byte = self.builder.ins().iconst(lane, 0xff);
            let low_byte = self.builder.ins().splat(vector_ty, low_byte);
            distance = self.builder.ins().band(distance, low_byte);
            distance = self
                .builder
                .ins()
                .ishl_imm_u(distance, i64::from(lane_bits - 8));
            distance = self
                .builder
                .ins()
                .sshr_imm_u(distance, i64::from(lane_bits - 8));
        }
        let nonnegative = self
            .builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, distance, zero);
        let negative = self.builder.ins().ineg(distance);
        let magnitude = self
            .builder
            .ins()
            .bitselect(nonnegative, distance, negative);
        let signed = matches!(instruction, Instruction::VectorSignedShiftRegister(_));
        let mut left = values;
        let mut right = values;
        let mut amount = 1_u32;
        while amount < lane_bits {
            let bit = self.builder.ins().iconst(lane, i64::from(amount));
            let bit = self.builder.ins().splat(vector_ty, bit);
            let selected = self.builder.ins().band(magnitude, bit);
            let selected = self.builder.ins().icmp(IntCC::NotEqual, selected, zero);
            let shifted_left = self.builder.ins().ishl_imm_u(left, i64::from(amount));
            let shifted_right = if signed {
                self.builder.ins().sshr_imm_u(right, i64::from(amount))
            } else {
                self.builder.ins().ushr_imm_u(right, i64::from(amount))
            };
            left = self.builder.ins().bitselect(selected, shifted_left, left);
            right = self.builder.ins().bitselect(selected, shifted_right, right);
            amount *= 2;
        }
        let width = self.builder.ins().iconst(lane, i64::from(lane_bits));
        let width = self.builder.ins().splat(vector_ty, width);
        let out = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, magnitude, width);
        let fill = if signed {
            self.builder
                .ins()
                .sshr_imm_u(values, i64::from(lane_bits - 1))
        } else {
            zero
        };
        left = self.builder.ins().bitselect(out, zero, left);
        right = self.builder.ins().bitselect(out, fill, right);
        let result = self.builder.ins().bitselect(nonnegative, left, right);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_add_across(&mut self, fields: Operands) -> Result<(), DirectJitError> {
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
            value = self.builder.ins().iadd(value, paired);
            distance /= 2;
        }
        let result = self.builder.ins().extractlane(value, 0);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)
    }

    fn emit_float_immediate(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
        let value = if matches!(instruction, Instruction::ScalarFloatImmediate(_)) {
            let (exponent, fraction) = match fields.opc {
                0 => (8, 23),
                1 => (11, 52),
                3 => (5, 10),
                _ => return Err(DirectJitError::invalid("invalid scalar FP immediate width")),
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

    fn emit_move_to_general(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let vector = self.read_vector_as(fields.rn, types::I128)?;
        let (width, value) = match (fields.size & 2 != 0, fields.opc) {
            (false, 0) => (32, self.builder.ins().ireduce(types::I32, vector)),
            (false, 3) => (32, self.builder.ins().ireduce(types::I16, vector)),
            (true, 1) => (64, self.builder.ins().ireduce(types::I64, vector)),
            (true, 2) => {
                let value = self.builder.ins().ushr_imm_u(vector, 64);
                (64, self.builder.ins().ireduce(types::I64, value))
            }
            _ => return Err(DirectJitError::invalid("invalid FMOV general width")),
        };
        let value = cast_integer(&mut self.builder, value, types::I64, false);
        let _ = width;
        self.write_register(fields.rd, value)
    }

    fn emit_move_from_general(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let value = self.read_register(fields.rn, false)?;
        let general_64 = fields.size & 2 != 0;
        let value = if general_64 {
            value
        } else {
            self.builder.ins().ireduce(types::I32, value)
        };
        let value = cast_integer(&mut self.builder, value, types::I128, false);
        let value = match (general_64, fields.opc) {
            (false, 0) => value,
            (false, 3) => {
                let value = self.builder.ins().ireduce(types::I16, value);
                self.builder.ins().uextend(types::I128, value)
            }
            (true, 1) => value,
            (true, 2) => {
                let previous = self.read_vector_as(fields.rd, types::I128)?;
                let low = self.builder.ins().ireduce(types::I64, previous);
                let low = self.builder.ins().uextend(types::I128, low);
                let high = self.builder.ins().ishl_imm_u(value, 64);
                self.builder.ins().bor(low, high)
            }
            _ => return Err(DirectJitError::invalid("invalid FMOV general width")),
        };
        let value = self.vector_as(value, types::I8X16);
        self.write_vector(fields.rd, value)
    }

    fn emit_vector_memory(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        match instruction {
            Instruction::MemoryPair(_) => self.emit_vector_pair(source, fields, flags),
            Instruction::MemoryMultipleStructures(_)
            | Instruction::MemoryMultipleStructuresPostIndex(_) => {
                self.emit_multiple_structures(source, instruction, fields, flags)
            }
            Instruction::MemorySingleStructure(_)
            | Instruction::MemorySingleStructurePostIndex(_) => {
                self.emit_single_structure(source, instruction, fields, flags)
            }
            _ => {
                let size = vector_access_size(fields)?;
                let base = self.read_register(fields.rn, true)?;
                let offset = match instruction {
                    Instruction::MemoryUnsigned(_) => {
                        u64::from(fields.immediate_12) * size.bytes() as u64
                    }
                    Instruction::MemoryUnscaled(_)
                    | Instruction::MemoryPostIndex(_)
                    | Instruction::MemoryPreIndex(_) => {
                        sign_extend(u64::from(fields.immediate_9), 9) as u64
                    }
                    Instruction::MemoryRegister(_) => {
                        let offset = self.read_register(fields.rm, false)?;
                        let offset = match fields.option {
                            2 => {
                                let offset = self.builder.ins().ireduce(types::I32, offset);
                                self.builder.ins().uextend(types::I64, offset)
                            }
                            3 => offset,
                            6 => {
                                let offset = self.builder.ins().ireduce(types::I32, offset);
                                self.builder.ins().sextend(types::I64, offset)
                            }
                            7 => offset,
                            _ => {
                                return Err(DirectJitError::invalid(
                                    "invalid SIMD register-offset extension",
                                ));
                            }
                        };
                        let offset = if fields.scaled {
                            self.builder
                                .ins()
                                .ishl_imm_u(offset, i64::from(size.bytes().trailing_zeros()))
                        } else {
                            offset
                        };
                        let address = self.builder.ins().iadd(base, offset);
                        return self.emit_vector_transfer(
                            source,
                            fields,
                            address,
                            size,
                            Some(0),
                            flags,
                        );
                    }
                    _ => unreachable!(),
                };
                let address = if matches!(instruction, Instruction::MemoryPostIndex(_)) {
                    base
                } else {
                    self.builder.ins().iadd_imm_u(base, offset as i64)
                };
                let direct = matches!(
                    instruction,
                    Instruction::MemoryUnsigned(_) | Instruction::MemoryUnscaled(_)
                );
                self.emit_vector_transfer(
                    source,
                    fields,
                    address,
                    size,
                    direct.then_some(0),
                    flags,
                )?;
                if matches!(
                    instruction,
                    Instruction::MemoryPostIndex(_) | Instruction::MemoryPreIndex(_)
                ) {
                    let updated = self.builder.ins().iadd_imm_u(base, offset as i64);
                    self.write_register_with_sp(fields.rn, true, updated)?;
                }
                Ok(())
            }
        }
    }

    fn emit_vector_transfer(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        address: Value,
        size: MemoryAccessSize,
        direct_element: Option<u8>,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let direct = direct_element.is_some();
        let element_index = direct_element.unwrap_or(0);
        if fields.load {
            let value = self.memory_read(
                source,
                address,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, direct)
                    .with_element_index(element_index),
                flags,
            )?;
            let value = if size == MemoryAccessSize::Quadword {
                value
            } else {
                self.builder.ins().uextend(types::I128, value)
            };
            let value = self.vector_as(value, types::I8X16);
            self.write_vector(fields.rd, value)
        } else {
            let value = self.read_vector_as(fields.rd, types::I128)?;
            let value = reduce_integer(&mut self.builder, value, size);
            self.memory_write(
                source,
                address,
                value,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, direct)
                    .with_element_index(element_index),
                flags,
            )
        }
    }

    fn emit_vector_pair(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = match fields.size {
            0 => MemoryAccessSize::Word,
            1 => MemoryAccessSize::Doubleword,
            2 => MemoryAccessSize::Quadword,
            _ => return Err(DirectJitError::invalid("invalid SIMD pair size")),
        };
        let base = self.read_register(fields.rn, true)?;
        let offset = sign_extend(u64::from(fields.immediate_7), 7) * size.bytes() as i64;
        let updated = self.builder.ins().iadd_imm_s(base, offset);
        let first = if matches!(fields.mode, 2 | 3) {
            updated
        } else {
            base
        };
        let second = self.builder.ins().iadd_imm_u(first, size.bytes() as i64);
        if fields.load {
            let first_value = self.memory_read(
                source,
                first,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, true),
                flags,
            )?;
            let second_value = self.memory_read(
                source,
                second,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, true).with_element_index(1),
                flags,
            )?;
            let first_value = if size == MemoryAccessSize::Quadword {
                first_value
            } else {
                self.builder.ins().uextend(types::I128, first_value)
            };
            let second_value = if size == MemoryAccessSize::Quadword {
                second_value
            } else {
                self.builder.ins().uextend(types::I128, second_value)
            };
            let first_value = self.vector_as(first_value, types::I8X16);
            let second_value = self.vector_as(second_value, types::I8X16);
            self.write_vector(fields.rd, first_value)?;
            self.write_vector(fields.rt2, second_value)?;
        } else {
            for (element_index, (register, address)) in [(fields.rd, first), (fields.rt2, second)]
                .into_iter()
                .enumerate()
            {
                let mut transfer = fields;
                transfer.rd = register;
                self.emit_vector_transfer(
                    source,
                    transfer,
                    address,
                    size,
                    Some(element_index as u8),
                    flags,
                )?;
            }
        }
        if matches!(fields.mode, 1 | 3) {
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_multiple_structures(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let count: u8 = match fields.structure_opcode {
            0b0010 => 4,
            0b0110 => 3,
            0b1010 => 2,
            0b0111 => 1,
            _ => return Err(DirectJitError::invalid("invalid LD1/ST1 register count")),
        };
        let size = if fields.vector_128 {
            MemoryAccessSize::Quadword
        } else {
            MemoryAccessSize::Doubleword
        };
        let base = self.read_register(fields.rn, true)?;
        for index in 0..count {
            let displacement = usize::from(index) * size.bytes();
            let address = self.builder.ins().iadd_imm_u(base, displacement as i64);
            let mut transfer = fields;
            transfer.rd = fields.rd.wrapping_add(index) & 31;
            self.emit_vector_transfer(source, transfer, address, size, None, flags)?;
        }
        if matches!(
            instruction,
            Instruction::MemoryMultipleStructuresPostIndex(_)
        ) {
            let offset = if fields.rm == 31 {
                self.builder
                    .ins()
                    .iconst(types::I64, (usize::from(count) * size.bytes()) as i64)
            } else {
                self.read_register(fields.rm, false)?
            };
            let updated = self.builder.ins().iadd(base, offset);
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_single_structure(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let (size, lane) = single_structure_shape(fields)?;
        let address = self.read_register(fields.rn, true)?;
        let lane_bits = (size.bytes() * 8) as u32;
        let lane_ty = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane_ty, lane_bits)?;
        if fields.load {
            let value = self.memory_read(
                source,
                address,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, false),
                flags,
            )?;
            let previous = self.read_vector_as(fields.rd, vector_ty)?;
            let value = cast_integer(&mut self.builder, value, lane_ty, false);
            let result = self.builder.ins().insertlane(previous, value, lane);
            let result = self.vector_as(result, types::I8X16);
            self.write_vector(fields.rd, result)?;
        } else {
            let vector = self.read_vector_as(fields.rd, vector_ty)?;
            let value = self.builder.ins().extractlane(vector, lane);
            self.memory_write(
                source,
                address,
                value,
                MemoryOperation::new(size, MemoryOrdering::Relaxed, false),
                flags,
            )?;
        }
        if matches!(instruction, Instruction::MemorySingleStructurePostIndex(_)) {
            let offset = if fields.rm == 31 {
                self.builder.ins().iconst(types::I64, size.bytes() as i64)
            } else {
                self.read_register(fields.rm, false)?
            };
            let updated = self.builder.ins().iadd(address, offset);
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_exact_fp(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        self.publish_fp_inputs(instruction, fields, flags)?;
        let constant = |translator: &mut Self, value: u64| {
            translator.builder.ins().iconst(types::I64, value as i64)
        };
        let rd = constant(self, u64::from(fields.rd));
        let rn = constant(self, u64::from(fields.rn));
        let rm = constant(self, u64::from(fields.rm));
        let ra = constant(self, u64::from(fields.ra));
        let opc = constant(self, u64::from(fields.opc));
        let q = constant(self, u64::from(fields.vector_128));
        let size = constant(self, u64::from(fields.size));
        let fraction = constant(
            self,
            fields.fixed_point_fraction_bits.map_or(u64::MAX, u64::from),
        );
        let (function, arguments): (usize, Vec<Value>) = match instruction {
            Instruction::SignedIntToFloat(_) => (
                slow::fp_signed_integer_to_float as *const () as usize,
                vec![rd, rn, opc, size, fraction],
            ),
            Instruction::UnsignedIntToFloat(_) => (
                slow::fp_unsigned_integer_to_float as *const () as usize,
                vec![rd, rn, opc, size, fraction],
            ),
            Instruction::VectorSignedIntToFloat(_) => (
                slow::fp_vector_signed_integer_to_float as *const () as usize,
                vec![rd, rn, rm, opc, q],
            ),
            Instruction::VectorUnsignedIntToFloat(_) => (
                slow::fp_vector_unsigned_integer_to_float as *const () as usize,
                vec![rd, rn, rm, opc, q],
            ),
            Instruction::ScalarVectorSignedIntToFloat(_) => (
                slow::fp_scalar_vector_signed_integer_to_float as *const () as usize,
                vec![rd, rn, rm, opc, q],
            ),
            Instruction::ScalarVectorUnsignedIntToFloat(_) => (
                slow::fp_scalar_vector_unsigned_integer_to_float as *const () as usize,
                vec![rd, rn, rm, opc, q],
            ),
            Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
                let rounding = fields
                    .float_to_integer_rounding
                    .expect("normalized FP conversion rounding");
                let rounding = match rounding {
                    FloatToIntegerRounding::NearestEven => 0,
                    FloatToIntegerRounding::NearestAway => 1,
                    FloatToIntegerRounding::TowardPositive => 2,
                    FloatToIntegerRounding::TowardNegative => 3,
                    FloatToIntegerRounding::TowardZero => 4,
                };
                let shape = constant(self, u64::from(fields.size) | (rounding << 8));
                let function = if matches!(instruction, Instruction::FloatToSignedInt(_)) {
                    slow::fp_float_to_signed_integer as *const () as usize
                } else {
                    slow::fp_float_to_unsigned_integer as *const () as usize
                };
                (function, vec![rd, rn, opc, shape, fraction])
            }
            Instruction::VectorFloatDivide(_) => (
                slow::fp_vector_divide as *const () as usize,
                vec![rd, rn, rm, opc, q],
            ),
            Instruction::ScalarFloatConvert(_) => {
                let function = match fields.float_conversion.expect("normalized FP conversion") {
                    FloatConversion::SingleToDouble => {
                        slow::fp_convert_single_double as *const () as usize
                    }
                    FloatConversion::DoubleToSingle => {
                        slow::fp_convert_double_single as *const () as usize
                    }
                };
                (function, vec![rd, rn, opc])
            }
            Instruction::ScalarFloatDivide(_) => {
                (slow::fp_divide as *const () as usize, vec![rd, rn, rm, opc])
            }
            Instruction::ScalarFloatRound(_) => {
                let function = match fields.float_round_operation.expect("normalized FP round") {
                    FloatRoundOperation::NearestEven => {
                        slow::fp_round_nearest_even as *const () as usize
                    }
                    FloatRoundOperation::TowardPositive => {
                        slow::fp_round_positive as *const () as usize
                    }
                    FloatRoundOperation::TowardNegative => {
                        slow::fp_round_negative as *const () as usize
                    }
                    FloatRoundOperation::TowardZero => slow::fp_round_zero as *const () as usize,
                    FloatRoundOperation::NearestAway => {
                        slow::fp_round_nearest_away as *const () as usize
                    }
                    FloatRoundOperation::Exact => slow::fp_round_exact as *const () as usize,
                    FloatRoundOperation::CurrentMode => {
                        slow::fp_round_current as *const () as usize
                    }
                };
                (function, vec![rd, rn, opc])
            }
            Instruction::ScalarFloatAdd(_) => {
                let function = match fields.float_add_operation.expect("normalized FP add") {
                    FloatAddOperation::Add => slow::fp_add as *const () as usize,
                    FloatAddOperation::Subtract => slow::fp_subtract as *const () as usize,
                };
                (function, vec![rd, rn, rm, opc])
            }
            Instruction::ScalarFloatMultiply(_) => {
                let function = match fields
                    .float_multiply_operation
                    .expect("normalized FP multiply")
                {
                    FloatMultiplyOperation::Multiply => slow::fp_multiply as *const () as usize,
                    FloatMultiplyOperation::NegatedMultiply => {
                        slow::fp_negated_multiply as *const () as usize
                    }
                };
                (function, vec![rd, rn, rm, opc])
            }
            Instruction::ScalarFloatFusedMultiplyAdd(_) => {
                let function = match fields
                    .float_fused_multiply_operation
                    .expect("normalized FP fused operation")
                {
                    FloatFusedMultiplyOperation::MultiplyAdd => {
                        slow::fp_multiply_add as *const () as usize
                    }
                    FloatFusedMultiplyOperation::MultiplySubtract => {
                        slow::fp_multiply_subtract as *const () as usize
                    }
                    FloatFusedMultiplyOperation::NegatedMultiplyAdd => {
                        slow::fp_negated_multiply_add as *const () as usize
                    }
                    FloatFusedMultiplyOperation::NegatedMultiplySubtract => {
                        slow::fp_negated_multiply_subtract as *const () as usize
                    }
                };
                (function, vec![rd, rn, rm, ra, opc])
            }
            Instruction::ScalarFloatSquareRoot(_) => (
                slow::fp_square_root as *const () as usize,
                vec![rd, rn, opc],
            ),
            Instruction::CompareRegister(_) | Instruction::CompareZero(_) => {
                let signaling = constant(self, u64::from(fields.signaling_compare));
                let function = if matches!(instruction, Instruction::CompareZero(_)) {
                    slow::fp_compare_zero as *const () as usize
                } else {
                    slow::fp_compare as *const () as usize
                };
                (function, vec![rn, rm, opc, signaling])
            }
            Instruction::ConditionalCompare(_) => {
                let signaling = constant(self, u64::from(fields.signaling_compare));
                let condition = constant(self, u64::from(fields.condition));
                let nzcv = constant(self, u64::from(fields.nzcv_immediate));
                (
                    slow::fp_conditional_compare as *const () as usize,
                    vec![rn, rm, opc, signaling, condition, nzcv],
                )
            }
            _ => {
                return Err(DirectJitError::unsupported(
                    "A64 FP instruction lacks an exact typed boundary",
                ));
            }
        };
        self.call_exact_fp(function, &arguments, source, flags)?;
        self.reload_fp_outputs(instruction, fields, flags)
    }

    fn publish_fp_inputs(
        &mut self,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let mut vectors = Vec::new();
        match instruction {
            Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
                self.store_general(fields.rn)?
            }
            Instruction::ScalarFloatFusedMultiplyAdd(_) => {
                vectors.extend([fields.rn, fields.rm, fields.ra])
            }
            Instruction::VectorFloatDivide(_)
            | Instruction::ScalarFloatDivide(_)
            | Instruction::ScalarFloatAdd(_)
            | Instruction::ScalarFloatMultiply(_)
            | Instruction::CompareRegister(_)
            | Instruction::ConditionalCompare(_) => vectors.extend([fields.rn, fields.rm]),
            _ => vectors.push(fields.rn),
        }
        vectors.sort_unstable();
        vectors.dedup();
        for register in vectors {
            self.store_vector(register)?;
        }
        if matches!(instruction, Instruction::ConditionalCompare(_)) {
            let packed = self.packed_flags(flags);
            self.builder
                .ins()
                .store(super::trusted_flags(), packed, self.nzcv, 0);
        }
        let fpsr = self.builder.use_var(self.fpsr_state);
        self.builder
            .ins()
            .store(super::trusted_flags(), fpsr, self.fpsr, 0);
        Ok(())
    }

    fn store_general(&mut self, index: u8) -> Result<(), DirectJitError> {
        if index == 31 {
            return Ok(());
        }
        let value = self.read_register(index, false)?;
        self.builder
            .ins()
            .store(super::trusted_flags(), value, self.x, i32::from(index) * 8);
        Ok(())
    }

    fn store_vector(&mut self, index: u8) -> Result<(), DirectJitError> {
        let value = self.read_vector(index)?;
        self.builder.ins().store(
            super::trusted_flags(),
            value,
            self.vector,
            i32::from(index) * 16,
        );
        Ok(())
    }

    fn call_exact_fp(
        &mut self,
        function: usize,
        arguments: &[Value],
        source: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let callee = self.builder.ins().iconst(types::I64, function as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.push(AbiParam::new(types::I64));
        for _ in arguments {
            signature.params.push(AbiParam::new(types::I64));
        }
        let signature = self.builder.import_signature(signature);
        let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
        call_arguments.push(self.context);
        call_arguments.extend_from_slice(arguments);
        self.builder
            .ins()
            .call_indirect(signature, callee, &call_arguments);
        let status = self.load_context(types::I32, offset_of!(NativeContext, slow_status))?;
        let success = self.builder.create_block();
        let failed = self.cold_block();
        let succeeded = self.builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
        self.builder
            .ins()
            .brif(succeeded, success, &[], failed, &[]);
        self.builder.switch_to_block(failed);
        self.commit_state(source, flags)?;
        self.dispatch_fp_failure(status, source);
        self.builder.switch_to_block(success);
        Ok(())
    }

    fn reload_fp_outputs(
        &mut self,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let fpsr = self
            .builder
            .ins()
            .load(types::I32, super::trusted_flags(), self.fpsr, 0);
        self.builder.def_var(self.fpsr_state, fpsr);
        self.block_dirty_fpsr = true;
        match instruction {
            Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
                if fields.rd != 31 {
                    let value = self.builder.ins().load(
                        types::I64,
                        super::trusted_flags(),
                        self.x,
                        i32::from(fields.rd) * 8,
                    );
                    self.write_register(fields.rd, value)?;
                }
            }
            Instruction::CompareRegister(_)
            | Instruction::CompareZero(_)
            | Instruction::ConditionalCompare(_) => {
                let packed =
                    self.builder
                        .ins()
                        .load(types::I32, super::trusted_flags(), self.nzcv, 0);
                *flags = LazyFlags::Packed(packed);
            }
            _ => {
                let value = self.builder.ins().load(
                    types::I8X16,
                    super::trusted_flags(),
                    self.vector,
                    i32::from(fields.rd) * 16,
                );
                self.write_vector(fields.rd, value)?;
            }
        }
        Ok(())
    }
}

fn bitcast_flags() -> MemFlagsData {
    MemFlagsData::new().with_endianness(Endianness::Little)
}

fn integer_lane_type(bits: u32) -> Result<cranelift_codegen::ir::Type, DirectJitError> {
    match bits {
        8 => Ok(types::I8),
        16 => Ok(types::I16),
        32 => Ok(types::I32),
        64 => Ok(types::I64),
        _ => Err(DirectJitError::invalid("invalid SIMD lane width")),
    }
}

fn vector_type(
    lane: cranelift_codegen::ir::Type,
    lane_bits: u32,
) -> Result<cranelift_codegen::ir::Type, DirectJitError> {
    lane.by(128 / lane_bits)
        .ok_or_else(|| DirectJitError::unsupported("host CLIF lacks required SIMD shape"))
}

fn cast_integer(
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

fn reduce_integer(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    size: MemoryAccessSize,
) -> Value {
    let ty = match size {
        MemoryAccessSize::Byte => types::I8,
        MemoryAccessSize::Halfword => types::I16,
        MemoryAccessSize::Word => types::I32,
        MemoryAccessSize::Doubleword => types::I64,
        MemoryAccessSize::Quadword => types::I128,
    };
    if builder.func.dfg.value_type(value) == ty {
        value
    } else {
        builder.ins().ireduce(ty, value)
    }
}

fn scalar_width(opc: u8) -> Result<u32, DirectJitError> {
    match opc {
        0 => Ok(32),
        1 => Ok(64),
        3 => Ok(16),
        _ => Err(DirectJitError::invalid("invalid scalar FP width")),
    }
}

fn vector_access_size(fields: Operands) -> Result<MemoryAccessSize, DirectJitError> {
    match fields.size + ((fields.opc & 2) << 1) {
        0 => Ok(MemoryAccessSize::Byte),
        1 => Ok(MemoryAccessSize::Halfword),
        2 => Ok(MemoryAccessSize::Word),
        3 => Ok(MemoryAccessSize::Doubleword),
        4 => Ok(MemoryAccessSize::Quadword),
        _ => Err(DirectJitError::invalid("invalid SIMD transfer size")),
    }
}

fn single_structure_shape(fields: Operands) -> Result<(MemoryAccessSize, u8), DirectJitError> {
    let opcode = fields.structure_opcode >> 1;
    let s = fields.structure_opcode & 1;
    let q = u8::from(fields.vector_128);
    match opcode {
        0 => Ok((
            MemoryAccessSize::Byte,
            (q << 3) | (s << 2) | fields.element_size,
        )),
        2 => Ok((
            MemoryAccessSize::Halfword,
            (q << 2) | (s << 1) | (fields.element_size >> 1),
        )),
        4 if fields.element_size == 0 => Ok((MemoryAccessSize::Word, (q << 1) | s)),
        4 => Ok((MemoryAccessSize::Doubleword, q)),
        _ => Err(DirectJitError::invalid(
            "invalid SIMD single-structure shape",
        )),
    }
}

fn expand_modified_immediate(
    cmode: u8,
    immediate: u8,
    operation_bit: bool,
) -> Result<u64, DirectJitError> {
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
        _ => return Err(DirectJitError::invalid("invalid SIMD modified immediate")),
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

const fn sign_extend(value: u64, bits: u8) -> i64 {
    ((value << (64 - bits)) as i64) >> (64 - bits)
}
