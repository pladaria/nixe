use crate::{
    address::GuestVirtualAddress,
    decode::{
        DecodedOpcode,
        a64::fp_simd::{BitwiseOperation, Instruction, IntegerComparison, PairwiseOperation},
    },
    location::DecodedInstruction,
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    semantics::{
        bits::{BitWidth, replicate},
        vector::{LaneWidth, VectorArrangement, extract_lane},
    },
    state::a64::A64State,
};

use super::{advance, read, register_offset_address, resume, sign_extend};
use crate::interpreter::{InterpreterContext, InterpreterError, InterpreterOutcome};

type MemoryStep = Result<(), DataAccessFault>;

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    let fields = instruction.operands();
    let result = match instruction {
        Instruction::DuplicateGeneral(_) => {
            duplicate_general(state, fields);
            None
        }
        Instruction::ModifiedImmediate(_) => {
            modified_immediate(state, fields);
            None
        }
        Instruction::UnsignedMoveToGeneral(_) => {
            unsigned_move_to_general(state, fields);
            None
        }
        Instruction::InsertElement(_) => {
            insert_element(state, fields);
            None
        }
        Instruction::InsertGeneral(_) => {
            insert_general(state, fields);
            None
        }
        Instruction::MoveToGeneral(_) => {
            floating_move_to_general(state, fields);
            None
        }
        Instruction::MoveFromGeneral(_) => {
            floating_move_from_general(state, fields);
            None
        }
        Instruction::Integer(_) => {
            integer_add_sub(state, fields);
            None
        }
        Instruction::Bitwise(_) => {
            bitwise(
                state,
                fields,
                fields
                    .bitwise_operation
                    .expect("normalized SIMD bitwise operation"),
            );
            None
        }
        Instruction::IntegerCompare(_) => {
            integer_compare(
                state,
                fields,
                fields
                    .integer_comparison
                    .expect("normalized SIMD integer comparison"),
            );
            None
        }
        Instruction::IntegerPairwise(_) => {
            integer_pairwise(
                state,
                fields,
                fields
                    .pairwise_operation
                    .expect("normalized SIMD pairwise operation"),
            );
            None
        }
        Instruction::MemoryUnsigned(_)
        | Instruction::MemoryUnscaled(_)
        | Instruction::MemoryPostIndex(_)
        | Instruction::MemoryPreIndex(_) => {
            let Some(memory) = context.memory() else {
                return Err(super::super::unsupported(decoded));
            };
            let address_space = context.process().address_space_id();
            let size = vector_access_size(fields)?;
            let base = read(state, fields.rn, 64, true);
            let offset = match instruction {
                Instruction::MemoryUnsigned(_) => {
                    u64::from(fields.immediate_12) * size.bytes() as u64
                }
                Instruction::MemoryUnscaled(_)
                | Instruction::MemoryPostIndex(_)
                | Instruction::MemoryPreIndex(_) => {
                    sign_extend(u64::from(fields.immediate_9), 9) as u64
                }
                _ => unreachable!(),
            };
            let address = if matches!(instruction, Instruction::MemoryPostIndex(_)) {
                base
            } else {
                base.wrapping_add(offset)
            };
            let result = vector_transfer(
                memory,
                address_space,
                state,
                fields,
                GuestVirtualAddress::new(address),
                size,
            );
            if result.is_ok()
                && matches!(
                    instruction,
                    Instruction::MemoryPostIndex(_) | Instruction::MemoryPreIndex(_)
                )
            {
                super::write(state, fields.rn, 64, true, base.wrapping_add(offset));
            }
            Some(result)
        }
        Instruction::MemoryPair(_) => {
            let Some(memory) = context.memory() else {
                return Err(super::super::unsupported(decoded));
            };
            Some(vector_pair(
                memory,
                context.process().address_space_id(),
                state,
                fields,
            ))
        }
        Instruction::MemoryRegister(_) => {
            let Some(memory) = context.memory() else {
                return Err(super::super::unsupported(decoded));
            };
            let size = vector_access_size(fields)?;
            let Some(address) = register_offset_address(
                state,
                fields.rn,
                fields.rm,
                fields.option,
                fields.scaled,
                size.bytes().trailing_zeros(),
            ) else {
                return Err(super::super::unsupported(decoded));
            };
            Some(vector_transfer(
                memory,
                context.process().address_space_id(),
                state,
                fields,
                address,
                size,
            ))
        }
        Instruction::MemoryMultipleStructures(_)
        | Instruction::MemoryMultipleStructuresPostIndex(_) => {
            let Some(memory) = context.memory() else {
                return Err(super::super::unsupported(decoded));
            };
            let Some(register_count) = ld1_st1_register_count(fields.structure_opcode) else {
                return Err(super::super::unsupported(decoded));
            };
            Some(vector_multiple_structures(
                memory,
                context.process().address_space_id(),
                state,
                fields,
                register_count,
                matches!(
                    instruction,
                    Instruction::MemoryMultipleStructuresPostIndex(_)
                ),
            ))
        }
        _ => return Err(super::super::unsupported(decoded)),
    };
    if let Some(Err(fault)) = result {
        return Ok(InterpreterOutcome::DataAbort {
            source: decoded.location,
            fault,
        });
    }
    advance(state);
    Ok(resume(state, decoded))
}

fn pair_access_size(size: u8) -> MemoryAccessSize {
    match size {
        0 => MemoryAccessSize::Word,
        1 => MemoryAccessSize::Doubleword,
        2 => MemoryAccessSize::Quadword,
        _ => unreachable!("allocation validation rejects invalid SIMD pair sizes"),
    }
}

fn vector_pair(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> MemoryStep {
    let size = pair_access_size(fields.size);
    let base = read(state, fields.rn, 64, true);
    let offset = sign_extend(u64::from(fields.immediate_7), 7) * size.bytes() as i64;
    let transfer_base = if matches!(fields.mode, 2 | 3) {
        base.wrapping_add_signed(offset)
    } else {
        base
    };
    let first = GuestVirtualAddress::new(transfer_base);
    let second = first.wrapping_add(size.bytes() as u64);
    if fields.load {
        let first_value = read_vector(memory, address_space, first, size)?;
        let second_value = read_vector(memory, address_space, second, size)?;
        assert!(state.set_vector(fields.rd, first_value));
        assert!(state.set_vector(fields.rt2, second_value));
    } else {
        write_vector(memory, address_space, first, size, state, fields.rd)?;
        write_vector(memory, address_space, second, size, state, fields.rt2)?;
    }
    if matches!(fields.mode, 1 | 3) {
        super::write(state, fields.rn, 64, true, base.wrapping_add_signed(offset));
    }
    Ok(())
}

// LD1/ST1 multiple-structures register-list semantics, Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/LD1--multiple-structures---Load-multiple-single-element-structures-to-one--two--three--or-four-registers-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ST1--multiple-structures---Store-multiple-single-element-structures-from-one--two--three--or-four-registers-
fn ld1_st1_register_count(opcode: u8) -> Option<u8> {
    match opcode {
        0b0010 => Some(4),
        0b0110 => Some(3),
        0b1010 => Some(2),
        0b0111 => Some(1),
        _ => None,
    }
}

fn vector_multiple_structures(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    register_count: u8,
    post_index: bool,
) -> MemoryStep {
    let size = if fields.vector_128 {
        MemoryAccessSize::Quadword
    } else {
        MemoryAccessSize::Doubleword
    };
    let base = read(state, fields.rn, 64, true);
    let mut address = GuestVirtualAddress::new(base);
    for index in 0..register_count {
        let register = fields.rd.wrapping_add(index) & 31;
        if fields.load {
            let value = read_vector(memory, address_space, address, size)?;
            assert!(state.set_vector(register, value));
        } else {
            write_vector(memory, address_space, address, size, state, register)?;
        }
        address = address.wrapping_add(size.bytes() as u64);
    }
    if post_index {
        let offset = if fields.rm == 31 {
            u64::from(register_count) * size.bytes() as u64
        } else {
            read(state, fields.rm, 64, false)
        };
        super::write(state, fields.rn, 64, true, base.wrapping_add(offset));
    }
    Ok(())
}

fn duplicate_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let lane_bits = 8_u8 << fields.immediate_5.trailing_zeros();
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let source = read(state, fields.rn, lane_bits, false);
    let value = replicate(
        source.into(),
        BitWidth::new(lane_bits).expect("allocated SIMD lane width"),
        BitWidth::new(vector_bits).expect("allocated SIMD vector width"),
    )
    .expect("allocated SIMD lane arrangement");
    assert!(state.set_vector(fields.rd, value));
}

fn modified_immediate(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let immediate =
        expand_modified_immediate(fields.cmode, fields.immediate_8, fields.operation_bit);
    let replicated = u128::from(immediate) | (u128::from(immediate) << 64);
    let active_mask = if fields.vector_128 {
        u128::MAX
    } else {
        u128::from(u64::MAX)
    };
    let previous = state
        .vector(fields.rd)
        .expect("normalized modified-immediate destination register");
    let value = if fields.cmode <= 11 && fields.cmode & 1 != 0 {
        if fields.operation_bit {
            previous & replicated
        } else {
            previous | replicated
        }
    } else {
        replicated
    };
    assert!(state.set_vector(fields.rd, value & active_mask));
}

fn expand_modified_immediate(cmode: u8, immediate: u8, operation_bit: bool) -> u64 {
    let immediate = u64::from(immediate);
    let value = match cmode {
        0..=7 => {
            let lane = immediate << ((cmode >> 1) * 8);
            lane | (lane << 32)
        }
        8..=11 => {
            let lane = immediate << (((cmode >> 1) & 1) * 8);
            lane | (lane << 16) | (lane << 32) | (lane << 48)
        }
        12 => {
            let lane = (immediate << 8) | 0xff;
            lane | (lane << 32)
        }
        13 => {
            let lane = (immediate << 16) | 0xffff;
            lane | (lane << 32)
        }
        14 if !operation_bit => immediate * 0x0101_0101_0101_0101,
        14 => {
            let mut result = 0_u64;
            for bit in 0..8 {
                if immediate & (1 << bit) != 0 {
                    result |= 0xff << (bit * 8);
                }
            }
            result
        }
        _ => unreachable!("allocation validation excludes floating-point immediates"),
    };
    if operation_bit && cmode != 14 {
        !value
    } else {
        value
    }
}

fn unsigned_move_to_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let size_shift = fields.immediate_5.trailing_zeros() as u8;
    let lane_width = match size_shift {
        0 => LaneWidth::Bits8,
        1 => LaneWidth::Bits16,
        2 => LaneWidth::Bits32,
        3 => LaneWidth::Bits64,
        _ => unreachable!("allocation validation rejects invalid UMOV element sizes"),
    };
    let lane = fields.immediate_5 >> (size_shift + 1);
    let arrangement =
        VectorArrangement::new(128, lane_width).expect("allocated UMOV vector arrangement");
    let vector = state
        .vector(fields.rn)
        .expect("normalized UMOV vector register");
    let value = extract_lane(vector, arrangement, lane).expect("allocated UMOV lane index");
    super::write(
        state,
        fields.rd,
        if fields.vector_128 { 64 } else { 32 },
        false,
        value,
    );
}

fn insert_element(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (lane_bits, destination_lane) = insert_lane_shape(fields.immediate_5);
    let size_shift = lane_bits.trailing_zeros() - 3;
    let source_lane = fields.immediate_4 >> size_shift;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized INS vector source register");
    let lane = (source >> (u32::from(source_lane) * lane_bits)) & lane_mask;
    insert_lane(state, fields.rd, destination_lane, lane_bits, lane);
}

fn insert_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (lane_bits, destination_lane) = insert_lane_shape(fields.immediate_5);
    let lane = read(state, fields.rn, lane_bits as u8, false);
    insert_lane(
        state,
        fields.rd,
        destination_lane,
        lane_bits,
        u128::from(lane),
    );
}

fn insert_lane_shape(immediate_5: u8) -> (u32, u8) {
    let size_shift = immediate_5.trailing_zeros();
    let lane_bits = 8_u32 << size_shift;
    let lane = immediate_5 >> (size_shift + 1);
    (lane_bits, lane)
}

fn insert_lane(state: &mut A64State, register: u8, lane: u8, lane_bits: u32, value: u128) {
    let shift = u32::from(lane) * lane_bits;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let previous = state
        .vector(register)
        .expect("normalized INS destination register");
    let result = (previous & !(lane_mask << shift)) | ((value & lane_mask) << shift);
    assert!(state.set_vector(register, result));
}

// FMOV between general-purpose and SIMD&FP registers copies the bit pattern
// without numeric conversion. Scalar destinations clear the rest of the
// SIMD&FP register; the Vd.D[1] form is the exception and preserves Dd.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--general---Floating-point-Move-to-or-from-general-purpose-register-without-conversion-
fn floating_move_to_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let vector = state
        .vector(fields.rn)
        .expect("normalized FMOV SIMD&FP source register");
    let (width, value) = match (fields.size & 2 != 0, fields.opc) {
        (false, 0) => (32, vector as u64),        // FMOV Wd, Sn
        (false, 3) => (32, vector as u16 as u64), // FMOV Wd, Hn
        (true, 1) => (64, vector as u64),         // FMOV Xd, Dn
        (true, 2) => (64, (vector >> 64) as u64), // FMOV Xd, Vn.D[1]
        _ => unreachable!("allocation validation rejects invalid FMOV register widths"),
    };
    super::write(state, fields.rd, width, false, value);
}

fn floating_move_from_general(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let general_64 = fields.size & 2 != 0;
    let width = if general_64 { 64 } else { 32 };
    let value = read(state, fields.rn, width, false);
    let vector = match (general_64, fields.opc) {
        (false, 0) => u128::from(value as u32), // FMOV Sd, Wn
        (false, 3) => u128::from(value as u16), // FMOV Hd, Wn
        (true, 1) => u128::from(value),         // FMOV Dd, Xn
        (true, 2) => {
            let previous = state
                .vector(fields.rd)
                .expect("normalized FMOV SIMD&FP destination register");
            (previous & u128::from(u64::MAX)) | (u128::from(value) << 64)
        } // FMOV Vd.D[1], Xn
        _ => unreachable!("allocation validation rejects invalid FMOV register widths"),
    };
    assert!(state.set_vector(fields.rd, vector));
}

fn integer_add_sub(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD source register");
    let rhs = state
        .vector(fields.rm)
        .expect("normalized SIMD source register");
    let mut result = 0_u128;
    for shift in (0..vector_bits).step_by(usize::from(lane_bits)) {
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        let lane = if fields.subtract {
            lhs_lane.wrapping_sub(rhs_lane)
        } else {
            lhs_lane.wrapping_add(rhs_lane)
        } & lane_mask;
        result |= lane << shift;
    }
    assert!(state.set_vector(fields.rd, result));
}

// The lower half of the result reduces adjacent pairs from Vn and the upper
// half reduces adjacent pairs from Vm. ADDP wraps at the element width, while
// the minimum/maximum forms compare signed or unsigned elements as specified
// by Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ADDP--vector---Add-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMAXP--Signed-Maximum-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SMINP--Signed-Minimum-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMAXP--Unsigned-Maximum-Pairwise--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UMINP--Unsigned-Minimum-Pairwise--vector--
fn integer_pairwise(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: PairwiseOperation,
) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let first = state
        .vector(fields.rn)
        .expect("normalized SIMD first source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized SIMD second source register");
    let lanes_per_source = vector_bits / u32::from(lane_bits);
    let mut result = 0_u128;

    for (source_index, source) in [first, second].into_iter().enumerate() {
        for pair in 0..(lanes_per_source / 2) {
            let first_shift = pair * 2 * u32::from(lane_bits);
            let second_shift = first_shift + u32::from(lane_bits);
            let lhs = (source >> first_shift) & lane_mask;
            let rhs = (source >> second_shift) & lane_mask;
            let reduced = match operation {
                PairwiseOperation::Add => lhs.wrapping_add(rhs) & lane_mask,
                PairwiseOperation::SignedMaximum => {
                    if sign_extend(lhs as u64, lane_bits) >= sign_extend(rhs as u64, lane_bits) {
                        lhs
                    } else {
                        rhs
                    }
                }
                PairwiseOperation::SignedMinimum => {
                    if sign_extend(lhs as u64, lane_bits) <= sign_extend(rhs as u64, lane_bits) {
                        lhs
                    } else {
                        rhs
                    }
                }
                PairwiseOperation::UnsignedMaximum => lhs.max(rhs),
                PairwiseOperation::UnsignedMinimum => lhs.min(rhs),
            };
            let destination_lane = source_index as u32 * (lanes_per_source / 2) + pair;
            result |= reduced << (destination_lane * u32::from(lane_bits));
        }
    }
    assert!(state.set_vector(fields.rd, result));
}

// Whole-vector operation and destination-mask rules for the Advanced SIMD
// bitwise family, Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/AND--vector---Bitwise-AND--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIC--vector---Bitwise-bit-Clear--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORR--vector---Bitwise-OR--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ORN--vector---Bitwise-inclusive-OR-NOT--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/EOR--vector---Bitwise-exclusive-OR--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BSL--Bitwise-Select-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIT--Bitwise-Insert-if-True-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/BIF--Bitwise-Insert-if-False-
fn bitwise(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: BitwiseOperation,
) {
    let first = state
        .vector(fields.rn)
        .expect("normalized SIMD source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized SIMD source register");
    let destination = state
        .vector(fields.rd)
        .expect("normalized SIMD destination register");
    let result = match operation {
        BitwiseOperation::And => first & second,
        BitwiseOperation::BitClear => first & !second,
        BitwiseOperation::Or => first | second,
        BitwiseOperation::OrNot => first | !second,
        BitwiseOperation::ExclusiveOr => first ^ second,
        BitwiseOperation::Select => (destination & first) | (!destination & second),
        BitwiseOperation::InsertIfTrue => (destination & !second) | (first & second),
        BitwiseOperation::InsertIfFalse => (destination & second) | (first & !second),
    };
    let active_mask = if fields.vector_128 {
        u128::MAX
    } else {
        u128::from(u64::MAX)
    };
    assert!(state.set_vector(fields.rd, result & active_mask));
}

// Per-lane result and signedness rules for the Advanced SIMD register
// comparisons, Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGT--register---Compare-signed-greater-than--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGE--register---Compare-signed-greater-than-or-equal--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMHI--register---Compare-unsigned-higher--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMHS--register---Compare-unsigned-higher-or-same--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMEQ--register---Compare-bitwise-equal--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMTST--Compare-bitwise-test-bits-nonzero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGT--zero---Compare-signed-greater-than-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMGE--zero---Compare-signed-greater-than-or-equal-to-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMEQ--zero---Compare-bitwise-equal-to-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMLE--Compare-signed-less-than-or-equal-to-zero--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/CMLT--Compare-signed-less-than-zero--vector--
fn integer_compare(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    comparison: IntegerComparison,
) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD source register");
    let rhs = if fields.compare_with_zero {
        0
    } else {
        state
            .vector(fields.rm)
            .expect("normalized SIMD source register")
    };
    let mut result = 0_u128;
    for shift in (0..vector_bits).step_by(usize::from(lane_bits)) {
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        let matches = match comparison {
            IntegerComparison::SignedGreaterThan => {
                sign_extend(lhs_lane as u64, lane_bits) > sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::UnsignedGreaterThan => lhs_lane > rhs_lane,
            IntegerComparison::SignedGreaterThanOrEqual => {
                sign_extend(lhs_lane as u64, lane_bits) >= sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::UnsignedGreaterThanOrEqual => lhs_lane >= rhs_lane,
            IntegerComparison::SignedLessThan => {
                sign_extend(lhs_lane as u64, lane_bits) < sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::SignedLessThanOrEqual => {
                sign_extend(lhs_lane as u64, lane_bits) <= sign_extend(rhs_lane as u64, lane_bits)
            }
            IntegerComparison::NonzeroBitTest => lhs_lane & rhs_lane != 0,
            IntegerComparison::Equal => lhs_lane == rhs_lane,
        };
        if matches {
            result |= lane_mask << shift;
        }
    }
    assert!(state.set_vector(fields.rd, result));
}

fn vector_access_size(
    fields: crate::decode::a64::fp_simd::Operands,
) -> Result<MemoryAccessSize, InterpreterError> {
    Ok(match fields.size + ((fields.opc & 2) << 1) {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        4 => MemoryAccessSize::Quadword,
        _ => unreachable!("allocation validation rejects invalid SIMD transfer sizes"),
    })
}

fn vector_transfer(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
) -> MemoryStep {
    if fields.load {
        let value = read_vector(memory, address_space, address, size)?;
        assert!(state.set_vector(fields.rd, value));
    } else {
        write_vector(memory, address_space, address, size, state, fields.rd)?;
    }
    Ok(())
}

fn read_vector(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
) -> Result<u128, DataAccessFault> {
    let value = memory
        .read(address_space, address, vector_access(size))?
        .value;
    Ok(match value {
        MemoryValue::U8(value) => u128::from(value),
        MemoryValue::U16(value) => u128::from(value),
        MemoryValue::U32(value) => u128::from(value),
        MemoryValue::U64(value) => u128::from(value),
        MemoryValue::U128(value) => value,
    })
}

fn write_vector(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
    state: &A64State,
    register: u8,
) -> Result<(), DataAccessFault> {
    let value = state.vector(register).expect("normalized vector register");
    let value = match size {
        MemoryAccessSize::Byte => MemoryValue::U8(value as u8),
        MemoryAccessSize::Halfword => MemoryValue::U16(value as u16),
        MemoryAccessSize::Word => MemoryValue::U32(value as u32),
        MemoryAccessSize::Doubleword => MemoryValue::U64(value as u64),
        MemoryAccessSize::Quadword => MemoryValue::U128(value),
    };
    memory
        .write(address_space, address, vector_access(size), value)
        .map(|_| ())
}

fn vector_access(size: MemoryAccessSize) -> MemoryAccess {
    MemoryAccess::new(
        size,
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    )
}
