use crate::{
    address::GuestVirtualAddress,
    decode::{
        DecodedOpcode,
        a64::fp_simd::{
            BitwiseOperation, FloatAddOperation, FloatConversion, FloatMultiplyOperation,
            FloatRoundOperation, FloatToIntegerRounding, Instruction, IntegerComparison,
            PairwiseOperation, PermuteOperation,
        },
    },
    location::DecodedInstruction,
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    semantics::{
        bits::{BitWidth, replicate},
        floating_point::{FpFormat, FpRoundingMode, FpStatus},
        vector::{LaneWidth, VectorArrangement, extract_lane},
    },
    state::a64::{A64State, Nzcv},
};

use super::{advance, read, register_offset_address, resume, sign_extend, write};
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
        Instruction::IntegerMinMax(_) => {
            integer_min_max(
                state,
                fields,
                fields
                    .pairwise_operation
                    .expect("normalized SIMD minimum/maximum operation"),
            );
            None
        }
        Instruction::PermuteTwoSource(_) => {
            permute_two_source(
                state,
                fields,
                fields
                    .permute_operation
                    .expect("normalized SIMD two-source permute operation"),
            );
            None
        }
        Instruction::Extract(_) => {
            extract_vector(state, fields);
            None
        }
        Instruction::ShiftRightNarrow(_) => {
            shift_right_narrow(state, fields);
            None
        }
        Instruction::VectorSignedIntToFloat(_) | Instruction::VectorUnsignedIntToFloat(_) => {
            let signed = matches!(instruction, Instruction::VectorSignedIntToFloat(_));
            let (value, inexact) = vector_integer_to_float(state, fields, signed);
            // FPCR.IXE requests an architectural trap for an inexact result.
            // Trap delivery is not implemented yet, so preserve precise state
            // and retain the typed unsupported-semantics boundary.
            if inexact && state.fpcr() & (1 << 12) != 0 {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, value));
            if inexact {
                state.set_fpsr(state.fpsr() | (1 << 4));
            }
            None
        }
        Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
            let signed = matches!(instruction, Instruction::SignedIntToFloat(_));
            let (value, inexact) = scalar_integer_to_float(state, fields, signed);
            // FPCR.IXE requests an architectural trap for an inexact result.
            // Trap delivery is not implemented, so commit neither destination
            // nor cumulative status when the conversion would trap.
            if inexact && state.fpcr() & (1 << 12) != 0 {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, u128::from(value)));
            if inexact {
                state.set_fpsr(state.fpsr() | (1 << 4));
            }
            None
        }
        Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
            let signed = matches!(instruction, Instruction::FloatToSignedInt(_));
            let outcome = scalar_float_to_integer(
                state,
                fields,
                signed,
                fields
                    .float_to_integer_rounding
                    .expect("normalized scalar floating-point to integer rounding mode"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            write(state, fields.rd, outcome.width, false, outcome.value);
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            None
        }
        Instruction::VectorFloatDivide(_) => {
            let (value, status) = vector_float_divide(state, fields);
            if fp_status_traps(status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, value));
            state.set_fpsr(state.fpsr() | fp_status_bits(status));
            None
        }
        Instruction::ScalarFloatImmediate(_) => {
            scalar_float_immediate(state, fields);
            None
        }
        Instruction::ScalarFloatConvert(_) => {
            let outcome = scalar_float_convert(
                state,
                fields,
                fields
                    .float_conversion
                    .expect("normalized scalar floating-point conversion"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            None
        }
        Instruction::ScalarFloatDivide(_) => {
            let outcome = scalar_float_divide(state, fields);
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            None
        }
        Instruction::ScalarFloatRound(_) => {
            let outcome = scalar_float_round(
                state,
                fields,
                fields
                    .float_round_operation
                    .expect("normalized scalar floating-point round operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            None
        }
        Instruction::ScalarFloatAdd(_) => {
            let outcome = scalar_float_add(
                state,
                fields,
                fields
                    .float_add_operation
                    .expect("normalized scalar floating-point add operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            None
        }
        Instruction::ScalarFloatMultiply(_) => {
            let outcome = scalar_float_multiply(
                state,
                fields,
                fields
                    .float_multiply_operation
                    .expect("normalized scalar floating-point multiply operation"),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            assert!(state.set_vector(fields.rd, u128::from(outcome.bits)));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
            None
        }
        Instruction::CompareRegister(_) | Instruction::CompareZero(_) => {
            let outcome = scalar_float_compare(
                state,
                fields,
                matches!(instruction, Instruction::CompareZero(_)),
            );
            if fp_status_traps(outcome.status, state.fpcr()) {
                return Err(super::super::unsupported(decoded));
            }
            state.set_nzcv(Nzcv::from_bits(outcome.nzcv));
            state.set_fpsr(state.fpsr() | fp_status_bits(outcome.status));
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
        Instruction::MemorySingleStructure(_) | Instruction::MemorySingleStructurePostIndex(_) => {
            let Some(memory) = context.memory() else {
                return Err(super::super::unsupported(decoded));
            };
            Some(vector_single_structure(
                memory,
                context.process().address_space_id(),
                state,
                fields,
                matches!(instruction, Instruction::MemorySingleStructurePostIndex(_)),
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

// LD1/ST1 transfer one element between memory and one vector lane. Loads
// preserve every other lane, and post-index writeback occurs only after a
// successful memory access. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/LD1--single-structure---Load-one-single-element-structure-to-one-lane-of-one-register-
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ST1--single-structure---Store-one-single-element-structure-from-one-lane-of-one-register-
fn vector_single_structure(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    post_index: bool,
) -> MemoryStep {
    let (size, lane) = single_structure_shape(fields);
    let base = read(state, fields.rn, 64, true);
    let address = GuestVirtualAddress::new(base);
    if fields.load {
        let value = read_vector(memory, address_space, address, size)?;
        insert_lane(state, fields.rd, lane, size.bytes() as u32 * 8, value);
    } else {
        let vector = state
            .vector(fields.rd)
            .expect("normalized single-structure source register");
        let lane_bits = size.bytes() as u32 * 8;
        let lane_mask = (1_u128 << lane_bits) - 1;
        let value = (vector >> (u32::from(lane) * lane_bits)) & lane_mask;
        write_lane(memory, address_space, address, size, value)?;
    }
    if post_index {
        let offset = if fields.rm == 31 {
            size.bytes() as u64
        } else {
            read(state, fields.rm, 64, false)
        };
        super::write(state, fields.rn, 64, true, base.wrapping_add(offset));
    }
    Ok(())
}

fn single_structure_shape(fields: crate::decode::a64::fp_simd::Operands) -> (MemoryAccessSize, u8) {
    let opcode = fields.structure_opcode >> 1;
    let s = fields.structure_opcode & 1;
    let q = u8::from(fields.vector_128);
    match opcode {
        0 => (
            MemoryAccessSize::Byte,
            (q << 3) | (s << 2) | fields.element_size,
        ),
        2 => (
            MemoryAccessSize::Halfword,
            (q << 2) | (s << 1) | (fields.element_size >> 1),
        ),
        4 if fields.element_size == 0 => (MemoryAccessSize::Word, (q << 1) | s),
        4 => (MemoryAccessSize::Doubleword, q),
        _ => unreachable!("allocation validation rejects other single-structure opcodes"),
    }
}

fn write_lane(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
    value: u128,
) -> MemoryStep {
    let value = match size {
        MemoryAccessSize::Byte => MemoryValue::U8(value as u8),
        MemoryAccessSize::Halfword => MemoryValue::U16(value as u16),
        MemoryAccessSize::Word => MemoryValue::U32(value as u32),
        MemoryAccessSize::Doubleword => MemoryValue::U64(value as u64),
        MemoryAccessSize::Quadword => unreachable!("single-structure lanes are at most 64 bits"),
    };
    memory
        .write(address_space, address, vector_access(size), value)
        .map(|_| ())
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

// FMOV (scalar, immediate) uses VFPExpandImm to construct an exact IEEE bit
// pattern. It performs no floating-point arithmetic, ignores FPCR, and does
// not update FPSR. A scalar SIMD&FP write clears all bits above the result.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--scalar--immediate---Floating-point-Move-immediate--scalar--
fn scalar_float_immediate(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let (exponent_bits, fraction_bits) = match fields.opc {
        0 => (8, 23),
        1 => (11, 52),
        3 => (5, 10),
        _ => unreachable!("allocation rejects the unallocated FMOV immediate type"),
    };
    let value = expand_vfp_immediate(fields.fp_immediate_8, exponent_bits, fraction_bits);
    assert!(state.set_vector(fields.rd, u128::from(value)));
}

fn expand_vfp_immediate(immediate: u8, exponent_bits: u32, fraction_bits: u32) -> u64 {
    let sign = u64::from(immediate >> 7);
    let exponent_control = u64::from((immediate >> 6) & 1);
    let exponent_tail = u64::from((immediate >> 4) & 3);
    let fraction_head = u64::from(immediate & 0xf);
    let sign_shift = exponent_bits + fraction_bits;
    let repeated_count = exponent_bits - 3;
    let repeated = if exponent_control == 0 {
        0
    } else {
        (1_u64 << repeated_count) - 1
    };
    (sign << sign_shift)
        | ((exponent_control ^ 1) << (sign_shift - 1))
        | (repeated << (fraction_bits + 2))
        | (exponent_tail << fraction_bits)
        | (fraction_head << (fraction_bits - 4))
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

// SCVTF and UCVTF convert each integer lane independently using FPCR.RMode,
// clear inactive destination bits, and accumulate FPSR.IXC when any lane is
// inexact. The conversion below constructs IEEE-754 bits directly so host
// floating-point state cannot affect guest rounding. Arm ARM DDI 0602
// (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--vector---Signed-integer-Convert-to-Floating-point--vector--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--vector---Unsigned-integer-Convert-to-Floating-point--vector--
fn vector_integer_to_float(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
) -> (u128, bool) {
    let lane_bits = if fields.opc == 0 { 32_u32 } else { 64_u32 };
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64_u32 };
    let lane_mask = if lane_bits == 64 {
        u128::from(u64::MAX)
    } else {
        u128::from(u32::MAX)
    };
    let rounding = fpcr_rounding_mode(state.fpcr());
    let source = state
        .vector(fields.rn)
        .expect("normalized SIMD conversion source register");
    let mut result = 0_u128;
    let mut any_inexact = false;
    for shift in (0..vector_bits).step_by(lane_bits as usize) {
        let lane = ((source >> shift) & lane_mask) as u64;
        let (negative, magnitude) = integer_sign_and_magnitude(lane, lane_bits, signed);
        let (converted, inexact) =
            integer_magnitude_to_ieee(magnitude, negative, lane_bits, rounding);
        result |= u128::from(converted) << shift;
        any_inexact |= inexact;
    }
    (result, any_inexact)
}

// SCVTF and UCVTF scalar integer forms use FPCR.RMode, set FPSR.IXC for a
// rounded result, and clear all bits above the scalar destination. Constructing
// the IEEE result directly avoids dependence on host floating-point state.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SCVTF--scalar--integer---Signed-integer-Convert-to-Floating-point--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UCVTF--scalar--integer---Unsigned-integer-Convert-to-Floating-point--scalar--
fn scalar_integer_to_float(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
) -> (u64, bool) {
    let source_bits = if fields.size & 2 != 0 { 64_u32 } else { 32_u32 };
    let destination_bits = match fields.opc {
        0 => 32_u32,
        1 => 64_u32,
        _ => unreachable!("decoder only accepts scalar S/D conversion destinations"),
    };
    let source = read(state, fields.rn, source_bits as u8, false);
    let (negative, magnitude) = integer_sign_and_magnitude(source, source_bits, signed);
    integer_magnitude_to_ieee(
        magnitude,
        negative,
        destination_bits,
        fpcr_rounding_mode(state.fpcr()),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IntegerConversionOutcome {
    value: u64,
    width: u8,
    status: FpStatus,
}

// FCVT{N,P,M,A,Z}{S,U} convert a scalar S or D operand to W or X using their
// encoded rounding direction. Invalid conversions produce the architecturally
// saturated result; finite discarded fractions set IXC. The conversion uses
// only the guest IEEE bit pattern so host floating-point behavior cannot alter
// rounding, saturation, or exception status. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZS--scalar--integer---Floating-point-Convert-to-Signed-integer--rounding-toward-Zero--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVTZU--scalar--integer---Floating-point-Convert-to-Unsigned-integer--rounding-toward-Zero--scalar--
fn scalar_float_to_integer(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    signed: bool,
    rounding: FloatToIntegerRounding,
) -> IntegerConversionOutcome {
    let width = if fields.size & 2 != 0 { 64_u8 } else { 32_u8 };
    let format = match fields.opc {
        0 => BinaryFormat::new(FpFormat::Binary32),
        1 => BinaryFormat::new(FpFormat::Binary64),
        _ => unreachable!("decoder only accepts scalar S/D conversion sources"),
    };
    let source_mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let source = state
        .vector(fields.rn)
        .expect("normalized scalar floating-point conversion source") as u64
        & source_mask;
    float_bits_to_integer(source, format, width, signed, rounding, state.fpcr())
}

fn float_bits_to_integer(
    source: u64,
    format: BinaryFormat,
    width: u8,
    signed: bool,
    rounding: FloatToIntegerRounding,
    fpcr: u32,
) -> IntegerConversionOutcome {
    let sign = source & format.sign_mask() != 0;
    let exponent = (source & format.exponent_mask()) >> format.fraction_bits;
    let fraction = source & format.fraction_mask();
    let mut status = FpStatus::default();
    let positive_limit = if signed {
        (1_u128 << (width - 1)) - 1
    } else {
        (1_u128 << width) - 1
    };
    let negative_limit = if signed { 1_u128 << (width - 1) } else { 0 };
    let saturated = |negative: bool| {
        if signed {
            if negative {
                1_u64 << (width - 1)
            } else if width == 64 {
                i64::MAX as u64
            } else {
                u64::from(i32::MAX as u32)
            }
        } else if negative {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            u64::from(u32::MAX)
        }
    };

    if exponent == format.exponent_max() {
        status.invalid_operation = true;
        return IntegerConversionOutcome {
            value: if fraction != 0 { 0 } else { saturated(sign) },
            width,
            status,
        };
    }

    if exponent == 0 && fraction != 0 && fpcr & (1 << 24) != 0 {
        status.input_denormal = true;
        return IntegerConversionOutcome {
            value: 0,
            width,
            status,
        };
    }

    let (significand, unbiased_exponent) = if exponent == 0 {
        (u128::from(fraction), 1_i32 - format.exponent_bias)
    } else {
        (
            u128::from((1_u64 << format.fraction_bits) | fraction),
            exponent as i32 - format.exponent_bias,
        )
    };
    let shift = unbiased_exponent - format.fraction_bits as i32;
    let (magnitude, discarded, relation_to_half) = if shift >= 0 {
        let shift = shift as u32;
        if shift >= 128 || significand > (u128::MAX >> shift) {
            (u128::MAX, false, core::cmp::Ordering::Less)
        } else {
            (significand << shift, false, core::cmp::Ordering::Less)
        }
    } else {
        let right = shift.unsigned_abs();
        if right >= 128 {
            (0, significand != 0, core::cmp::Ordering::Less)
        } else {
            let mask = (1_u128 << right) - 1;
            let remainder = significand & mask;
            (
                significand >> right,
                remainder != 0,
                remainder.cmp(&(1_u128 << (right - 1))),
            )
        }
    };
    let increment = discarded
        && match rounding {
            FloatToIntegerRounding::NearestEven => {
                relation_to_half.is_gt() || (relation_to_half.is_eq() && magnitude & 1 != 0)
            }
            FloatToIntegerRounding::NearestAway => !relation_to_half.is_lt(),
            FloatToIntegerRounding::TowardPositive => !sign,
            FloatToIntegerRounding::TowardNegative => sign,
            FloatToIntegerRounding::TowardZero => false,
        };
    let magnitude = magnitude.saturating_add(u128::from(increment));

    let out_of_range = if sign {
        if signed {
            magnitude > negative_limit
        } else {
            magnitude != 0
        }
    } else {
        magnitude > positive_limit
    };
    if out_of_range {
        status.invalid_operation = true;
        return IntegerConversionOutcome {
            value: saturated(sign),
            width,
            status,
        };
    }

    status.inexact = discarded;
    let value = if sign && magnitude != 0 {
        (0_u64).wrapping_sub(magnitude as u64)
    } else {
        magnitude as u64
    };
    IntegerConversionOutcome {
        value,
        width,
        status,
    }
}

fn fpcr_rounding_mode(fpcr: u32) -> FpRoundingMode {
    match (fpcr >> 22) & 3 {
        0 => FpRoundingMode::TiesToEven,
        1 => FpRoundingMode::TowardPositive,
        2 => FpRoundingMode::TowardNegative,
        3 => FpRoundingMode::TowardZero,
        _ => unreachable!(),
    }
}

fn integer_sign_and_magnitude(value: u64, width: u32, signed: bool) -> (bool, u64) {
    if !signed {
        return (false, value);
    }
    if width == 32 {
        let value = value as u32 as i32;
        (value.is_negative(), u64::from(value.unsigned_abs()))
    } else {
        let value = value as i64;
        (value.is_negative(), value.unsigned_abs())
    }
}

fn integer_magnitude_to_ieee(
    magnitude: u64,
    negative: bool,
    format_bits: u32,
    rounding: FpRoundingMode,
) -> (u64, bool) {
    if magnitude == 0 {
        return (0, false);
    }
    let (fraction_bits, exponent_bias) = if format_bits == 32 {
        (23_u32, 127_u64)
    } else {
        (52_u32, 1023_u64)
    };
    let leading_bit = 63 - magnitude.leading_zeros();
    let mut exponent = u64::from(leading_bit) + exponent_bias;
    let (mut significand, inexact) = if leading_bit <= fraction_bits {
        (magnitude << (fraction_bits - leading_bit), false)
    } else {
        let discarded_bits = leading_bit - fraction_bits;
        let retained = magnitude >> discarded_bits;
        let discarded_mask = (1_u64 << discarded_bits) - 1;
        let discarded = magnitude & discarded_mask;
        let halfway = 1_u64 << (discarded_bits - 1);
        let increment = match rounding {
            FpRoundingMode::TiesToEven => {
                discarded > halfway || (discarded == halfway && retained & 1 != 0)
            }
            FpRoundingMode::TowardPositive => !negative && discarded != 0,
            FpRoundingMode::TowardNegative => negative && discarded != 0,
            FpRoundingMode::TowardZero => false,
            FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
                unreachable!("FPCR cannot select this rounding mode")
            }
        };
        (retained + u64::from(increment), discarded != 0)
    };
    let precision = fraction_bits + 1;
    if significand == 1_u64 << precision {
        significand >>= 1;
        exponent += 1;
    }
    let sign = u64::from(negative) << (format_bits - 1);
    let fraction_mask = (1_u64 << fraction_bits) - 1;
    (
        sign | (exponent << fraction_bits) | (significand & fraction_mask),
        inexact,
    )
}

// FCVT converts between the base scalar S and D formats using FPCR controls,
// writes a scalar result (therefore clearing the remaining destination bits),
// and accumulates architectural exception status. Integer-only packing keeps
// guest rounding and NaN payload conversion independent of host FP behavior.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCVT--scalar---Floating-point-Convert-precision--scalar--
fn scalar_float_convert(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    conversion: FloatConversion,
) -> FpLaneOutcome {
    let (source_format, destination_format) = match conversion {
        FloatConversion::SingleToDouble => (FpFormat::Binary32, FpFormat::Binary64),
        FloatConversion::DoubleToSingle => (FpFormat::Binary64, FpFormat::Binary32),
    };
    let source_mask = match source_format {
        FpFormat::Binary32 => u64::from(u32::MAX),
        FpFormat::Binary64 => u64::MAX,
        FpFormat::Binary16 => unreachable!(),
    };
    let source = state
        .vector(fields.rn)
        .expect("normalized scalar floating-point conversion source") as u64
        & source_mask;
    convert_ieee_format(
        source,
        source_format,
        destination_format,
        FpConvertControl::from_fpcr(state.fpcr()),
    )
}

#[derive(Clone, Copy)]
struct FpConvertControl {
    rounding: FpRoundingMode,
    default_nan: bool,
    flush_to_zero: bool,
}

impl FpConvertControl {
    fn from_fpcr(fpcr: u32) -> Self {
        Self {
            rounding: fpcr_rounding_mode(fpcr),
            default_nan: fpcr & (1 << 25) != 0,
            flush_to_zero: fpcr & (1 << 24) != 0,
        }
    }
}

fn convert_ieee_format(
    source_bits: u64,
    source_fp_format: FpFormat,
    destination_fp_format: FpFormat,
    control: FpConvertControl,
) -> FpLaneOutcome {
    let source_format = BinaryFormat::new(source_fp_format);
    let destination_format = BinaryFormat::new(destination_fp_format);
    let mut source = DecodedFloat::new(source_bits, source_format);
    let mut status = FpStatus::default();

    if source.is_nan(source_format) {
        status.invalid_operation = source.is_signaling_nan(source_format);
        return FpLaneOutcome {
            bits: convert_nan(
                source,
                source_format,
                destination_format,
                control.default_nan,
            ),
            status,
        };
    }

    let sign_bits = u64::from(source.sign) << (destination_format.total_bits - 1);
    if control.flush_to_zero && source.is_subnormal() {
        status.input_denormal = true;
        source.significand = 0;
        source.fraction = 0;
    }
    if source.is_zero() {
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    if source.is_infinite(source_format) {
        return FpLaneOutcome {
            bits: sign_bits | destination_format.exponent_mask(),
            status,
        };
    }

    convert_finite_format(source, sign_bits, destination_format, control, status)
}

fn convert_nan(
    source: DecodedFloat,
    source_format: BinaryFormat,
    destination_format: BinaryFormat,
    default_nan: bool,
) -> u64 {
    if default_nan {
        return destination_format.default_nan();
    }
    let source_payload = source.fraction | source_format.quiet_nan_bit();
    let payload = if source_format.fraction_bits > destination_format.fraction_bits {
        source_payload >> (source_format.fraction_bits - destination_format.fraction_bits)
    } else {
        source_payload << (destination_format.fraction_bits - source_format.fraction_bits)
    };
    let sign = u64::from(source.sign) << (destination_format.total_bits - 1);
    sign | destination_format.exponent_mask()
        | (payload & destination_format.fraction_mask())
        | destination_format.quiet_nan_bit()
}

fn convert_finite_format(
    source: DecodedFloat,
    sign_bits: u64,
    destination_format: BinaryFormat,
    control: FpConvertControl,
    mut status: FpStatus,
) -> FpLaneOutcome {
    let source_top = 63 - source.significand.leading_zeros() as i32;
    let mut exponent = source.scale + source_top;
    let minimum_normal = 1 - destination_format.exponent_bias;
    let maximum_normal = destination_format.exponent_bias;

    if exponent >= minimum_normal {
        let destination_precision = destination_format.fraction_bits + 1;
        let source_precision = source_top as u32 + 1;
        let (mut significand, inexact) = if source_precision > destination_precision {
            round_shift_right(
                source.significand,
                source_precision - destination_precision,
                source.sign,
                control.rounding,
            )
        } else {
            (
                source.significand << (destination_precision - source_precision),
                false,
            )
        };
        if significand == 1_u64 << destination_precision {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > maximum_normal {
            status.overflow = true;
            status.inexact = true;
            return FpLaneOutcome {
                bits: overflow_result(sign_bits, destination_format, control.rounding),
                status,
            };
        }
        status.inexact = inexact;
        return FpLaneOutcome {
            bits: sign_bits
                | (((exponent + destination_format.exponent_bias) as u64)
                    << destination_format.fraction_bits)
                | (significand & destination_format.fraction_mask()),
            status,
        };
    }

    let minimum_subnormal = minimum_normal - destination_format.fraction_bits as i32;
    let scale_difference = source.scale - minimum_subnormal;
    let (fraction, inexact) = if scale_difference >= 0 {
        (source.significand << scale_difference, false)
    } else {
        round_shift_right(
            source.significand,
            (-scale_difference) as u32,
            source.sign,
            control.rounding,
        )
    };
    status.inexact = inexact;
    status.underflow = inexact;
    if fraction == 1_u64 << destination_format.fraction_bits {
        return FpLaneOutcome {
            bits: sign_bits | (1_u64 << destination_format.fraction_bits),
            status,
        };
    }
    if control.flush_to_zero && fraction != 0 {
        status.underflow = true;
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    FpLaneOutcome {
        bits: sign_bits | fraction,
        status,
    }
}

fn round_shift_right(
    value: u64,
    shift: u32,
    negative: bool,
    rounding: FpRoundingMode,
) -> (u64, bool) {
    if shift == 0 {
        return (value, false);
    }
    let retained = if shift < 64 { value >> shift } else { 0 };
    let discarded = if shift < 64 {
        value & ((1_u64 << shift) - 1)
    } else {
        value
    };
    if discarded == 0 {
        return (retained, false);
    }
    let increment = match rounding {
        FpRoundingMode::TiesToEven if shift < 64 => {
            let halfway = 1_u64 << (shift - 1);
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        FpRoundingMode::TiesToEven if shift == 64 => discarded > (1_u64 << 63),
        FpRoundingMode::TiesToEven => false,
        FpRoundingMode::TowardPositive => !negative,
        FpRoundingMode::TowardNegative => negative,
        FpRoundingMode::TowardZero => false,
        FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
            unreachable!("FPCR cannot select this rounding mode")
        }
    };
    (retained + u64::from(increment), true)
}

// FDIV applies the scalar IEEE operation independently to every active vector
// lane. Integer arithmetic keeps guest rounding, NaN propagation, denormal
// controls, and cumulative exception state independent of host FP behavior.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--vector---Floating-point-Divide--vector--
fn vector_float_divide(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> (u128, FpStatus) {
    let format = if fields.opc & 1 == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let lane_bits = u32::from(format.bits());
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64_u32 };
    let lane_mask = if lane_bits == 64 {
        u128::from(u64::MAX)
    } else {
        u128::from(u32::MAX)
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD first division operand");
    let rhs = state
        .vector(fields.rm)
        .expect("normalized SIMD second division operand");
    let control = FpDivideControl::from_fpcr(state.fpcr());
    let mut result = 0_u128;
    let mut status = FpStatus::default();
    for shift in (0..vector_bits).step_by(lane_bits as usize) {
        let lhs_lane = ((lhs >> shift) & lane_mask) as u64;
        let rhs_lane = ((rhs >> shift) & lane_mask) as u64;
        let outcome = divide_ieee_lane(lhs_lane, rhs_lane, format, control);
        result |= u128::from(outcome.bits) << shift;
        merge_fp_status(&mut status, outcome.status);
    }
    (result, status)
}

// Scalar FDIV shares the exact integer-only IEEE lane implementation with
// vector FDIV, but writes only the selected S/D scalar and clears all upper
// destination bits. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FDIV--scalar---Floating-point-Divide--scalar--
fn scalar_float_divide(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
) -> FpLaneOutcome {
    let format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let mask = match format {
        FpFormat::Binary32 => u64::from(u32::MAX),
        FpFormat::Binary64 => u64::MAX,
        FpFormat::Binary16 => unreachable!("scalar FDIV pattern excludes half precision"),
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized scalar first division operand") as u64
        & mask;
    let rhs = state
        .vector(fields.rm)
        .expect("normalized scalar second division operand") as u64
        & mask;
    divide_ieee_lane(lhs, rhs, format, FpDivideControl::from_fpcr(state.fpcr()))
}

// FMUL and FNMUL share one exact integer-significand product. FNMUL negates
// the completed product, including its encoded sign, as specified by Arm.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMUL--scalar---Floating-point-Multiply--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FNMUL--Floating-point-Negated-Multiply--scalar--
fn scalar_float_multiply(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatMultiplyOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let lhs = state
        .vector(fields.rn)
        .expect("normalized scalar multiplication first operand") as u64
        & mask;
    let rhs = state
        .vector(fields.rm)
        .expect("normalized scalar multiplication second operand") as u64
        & mask;
    let mut outcome = multiply_ieee_lane(lhs, rhs, fp_format, state.fpcr());
    if matches!(operation, FloatMultiplyOperation::NegatedMultiply) {
        outcome.bits ^= format.sign_mask();
    }
    outcome
}

fn multiply_ieee_lane(
    lhs_bits: u64,
    rhs_bits: u64,
    fp_format: FpFormat,
    fpcr: u32,
) -> FpLaneOutcome {
    let format = BinaryFormat::new(fp_format);
    let control = FpAddControl::from_fpcr(fpcr);
    let mut status = FpStatus::default();
    let mut lhs = DecodedFloat::new(lhs_bits, format);
    let mut rhs = DecodedFloat::new(rhs_bits, format);

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = lhs.is_signaling_nan(format) || rhs.is_signaling_nan(format);
        return FpLaneOutcome {
            bits: propagate_nan(lhs, rhs, format, control.default_nan),
            status,
        };
    }
    for operand in [&mut lhs, &mut rhs] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
        }
    }

    let negative = lhs.sign ^ rhs.sign;
    let sign_bits = u64::from(negative) << (format.total_bits - 1);
    if (lhs.is_infinite(format) && rhs.is_zero()) || (rhs.is_infinite(format) && lhs.is_zero()) {
        status.invalid_operation = true;
        return FpLaneOutcome {
            bits: format.default_nan(),
            status,
        };
    }
    if lhs.is_infinite(format) || rhs.is_infinite(format) {
        return FpLaneOutcome {
            bits: sign_bits | format.exponent_mask(),
            status,
        };
    }
    if lhs.is_zero() || rhs.is_zero() {
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }

    let magnitude = u128::from(lhs.significand) * u128::from(rhs.significand);
    pack_float_sum(
        magnitude,
        lhs.scale + rhs.scale,
        negative,
        format,
        control,
        status,
    )
}

// FADD and FSUB share one exact integer-significand implementation. Three
// low-order guard/round/sticky bits retain every distinction needed for the
// architectural rounding modes without consulting host floating-point state.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FADD--scalar---Floating-point-Add--scalar--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FSUB--scalar---Floating-point-Subtract--scalar--
fn scalar_float_add(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatAddOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let mut lhs = DecodedFloat::new(
        state
            .vector(fields.rn)
            .expect("normalized scalar addition first operand") as u64
            & mask,
        format,
    );
    let mut rhs = DecodedFloat::new(
        state
            .vector(fields.rm)
            .expect("normalized scalar addition second operand") as u64
            & mask,
        format,
    );
    if matches!(operation, FloatAddOperation::Subtract) {
        rhs.bits ^= format.sign_mask();
        rhs.sign = !rhs.sign;
    }
    let control = FpAddControl::from_fpcr(state.fpcr());
    let mut status = FpStatus::default();

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = lhs.is_signaling_nan(format) || rhs.is_signaling_nan(format);
        return FpLaneOutcome {
            bits: propagate_nan(lhs, rhs, format, control.default_nan),
            status,
        };
    }
    for operand in [&mut lhs, &mut rhs] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
        }
    }

    if lhs.is_infinite(format) || rhs.is_infinite(format) {
        if lhs.is_infinite(format) && rhs.is_infinite(format) && lhs.sign != rhs.sign {
            status.invalid_operation = true;
            return FpLaneOutcome {
                bits: format.default_nan(),
                status,
            };
        }
        return FpLaneOutcome {
            bits: if lhs.is_infinite(format) {
                lhs.bits
            } else {
                rhs.bits
            },
            status,
        };
    }
    if lhs.is_zero() && rhs.is_zero() {
        let negative = if lhs.sign == rhs.sign {
            lhs.sign
        } else {
            control.rounding == FpRoundingMode::TowardNegative
        };
        return FpLaneOutcome {
            bits: u64::from(negative) << (format.total_bits - 1),
            status,
        };
    }
    if lhs.is_zero() {
        return FpLaneOutcome {
            bits: rhs.bits,
            status,
        };
    }
    if rhs.is_zero() {
        return FpLaneOutcome {
            bits: lhs.bits,
            status,
        };
    }

    let common_scale = lhs.scale.max(rhs.scale);
    let lhs_magnitude = shift_right_jam(
        u128::from(lhs.significand) << 3,
        (common_scale - lhs.scale) as u32,
    );
    let rhs_magnitude = shift_right_jam(
        u128::from(rhs.significand) << 3,
        (common_scale - rhs.scale) as u32,
    );
    let (negative, magnitude) = if lhs.sign == rhs.sign {
        (lhs.sign, lhs_magnitude + rhs_magnitude)
    } else if lhs_magnitude > rhs_magnitude {
        (lhs.sign, lhs_magnitude - rhs_magnitude)
    } else if rhs_magnitude > lhs_magnitude {
        (rhs.sign, rhs_magnitude - lhs_magnitude)
    } else {
        let negative = control.rounding == FpRoundingMode::TowardNegative;
        return FpLaneOutcome {
            bits: u64::from(negative) << (format.total_bits - 1),
            status,
        };
    };
    pack_float_sum(
        magnitude,
        common_scale - 3,
        negative,
        format,
        control,
        status,
    )
}

#[derive(Clone, Copy)]
struct FpAddControl {
    rounding: FpRoundingMode,
    default_nan: bool,
    flush_to_zero: bool,
}

impl FpAddControl {
    fn from_fpcr(fpcr: u32) -> Self {
        Self {
            rounding: fpcr_rounding_mode(fpcr),
            default_nan: fpcr & (1 << 25) != 0,
            flush_to_zero: fpcr & (1 << 24) != 0,
        }
    }
}

fn shift_right_jam(value: u128, distance: u32) -> u128 {
    if distance == 0 {
        value
    } else if distance < 128 {
        (value >> distance) | u128::from(value & ((1_u128 << distance) - 1) != 0)
    } else {
        u128::from(value != 0)
    }
}

fn pack_float_sum(
    magnitude: u128,
    scale: i32,
    negative: bool,
    format: BinaryFormat,
    control: FpAddControl,
    mut status: FpStatus,
) -> FpLaneOutcome {
    let sign_bits = u64::from(negative) << (format.total_bits - 1);
    let top = 127 - magnitude.leading_zeros() as i32;
    let exponent = scale + top;
    let minimum_normal = 1 - format.exponent_bias;
    let maximum_normal = format.exponent_bias;

    if exponent >= minimum_normal {
        let shift = top - format.fraction_bits as i32;
        let (mut significand, remainder, denominator) = scale_integer_for_pack(magnitude, shift);
        let inexact = remainder != 0;
        if should_round(
            significand,
            remainder,
            denominator,
            negative,
            control.rounding,
        ) {
            significand += 1;
        }
        let mut rounded_exponent = exponent;
        if significand == 1_u64 << (format.fraction_bits + 1) {
            significand >>= 1;
            rounded_exponent += 1;
        }
        if rounded_exponent > maximum_normal {
            status.overflow = true;
            status.inexact = true;
            return FpLaneOutcome {
                bits: overflow_result(sign_bits, format, control.rounding),
                status,
            };
        }
        status.inexact = inexact;
        return FpLaneOutcome {
            bits: sign_bits
                | (((rounded_exponent + format.exponent_bias) as u64) << format.fraction_bits)
                | (significand & format.fraction_mask()),
            status,
        };
    }

    let minimum_subnormal = minimum_normal - format.fraction_bits as i32;
    let shift = minimum_subnormal - scale;
    let (mut fraction, remainder, denominator) = scale_integer_for_pack(magnitude, shift);
    let inexact = remainder != 0;
    if should_round(fraction, remainder, denominator, negative, control.rounding) {
        fraction += 1;
    }
    status.inexact = inexact;
    status.underflow = inexact;
    if fraction == 1_u64 << format.fraction_bits {
        return FpLaneOutcome {
            bits: sign_bits | (1_u64 << format.fraction_bits),
            status,
        };
    }
    if control.flush_to_zero && fraction != 0 {
        status.underflow = true;
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    FpLaneOutcome {
        bits: sign_bits | fraction,
        status,
    }
}

fn scale_integer_for_pack(value: u128, shift: i32) -> (u64, u128, u128) {
    if shift <= 0 {
        return ((value << (-shift) as u32) as u64, 0, 1);
    }
    let shift = shift as u32;
    if shift < 128 {
        let denominator = 1_u128 << shift;
        return (
            (value >> shift) as u64,
            value & (denominator - 1),
            denominator,
        );
    }
    (0, value, u128::MAX)
}

// FRINTN/P/M/Z/A/X/I round one scalar S/D value to an integral floating-point
// value. The implementation operates on the IEEE fields directly so guest
// rounding is independent of the host floating-point environment. FRINTX is
// the sole member that reports an inexact result; FRINTX and FRINTI select
// their direction from FPCR.RMode. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FRINTN--FRINTP--FRINTM--FRINTZ--FRINTA--FRINTX--FRINTI--Floating-point-Round-to-Integer--scalar--
fn scalar_float_round(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: FloatRoundOperation,
) -> FpLaneOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let source_bits = state
        .vector(fields.rn)
        .expect("normalized scalar floating-point round source") as u64
        & mask;
    let mut source = DecodedFloat::new(source_bits, format);
    let mut status = FpStatus::default();

    if source.is_nan(format) {
        status.invalid_operation = source.is_signaling_nan(format);
        let bits = if state.fpcr() & (1 << 25) != 0 {
            format.default_nan()
        } else {
            source.bits | format.quiet_nan_bit()
        };
        return FpLaneOutcome { bits, status };
    }
    if source.is_infinite(format) || source.is_zero() {
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }
    if state.fpcr() & (1 << 24) != 0 && source.is_subnormal() {
        status.input_denormal = true;
        source = DecodedFloat::new(source.bits & format.sign_mask(), format);
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }

    let exponent = source.exponent_field as i32 - format.exponent_bias;
    if exponent >= format.fraction_bits as i32 {
        return FpLaneOutcome {
            bits: source.bits,
            status,
        };
    }

    let (rounding, exact) = match operation {
        FloatRoundOperation::NearestEven => (FpRoundingMode::TiesToEven, false),
        FloatRoundOperation::TowardPositive => (FpRoundingMode::TowardPositive, false),
        FloatRoundOperation::TowardNegative => (FpRoundingMode::TowardNegative, false),
        FloatRoundOperation::TowardZero => (FpRoundingMode::TowardZero, false),
        FloatRoundOperation::NearestAway => (FpRoundingMode::TiesAway, false),
        FloatRoundOperation::Exact => (fpcr_rounding_mode(state.fpcr()), true),
        FloatRoundOperation::CurrentMode => (fpcr_rounding_mode(state.fpcr()), false),
    };
    let shift = (format.fraction_bits as i32 - exponent) as u32;
    let (magnitude, inexact) =
        round_integral_magnitude(source.significand, shift, source.sign, rounding);
    status.inexact = exact && inexact;
    let bits = if magnitude == 0 {
        source.bits & format.sign_mask()
    } else {
        integer_magnitude_to_ieee(magnitude, source.sign, format.total_bits, rounding).0
    };
    FpLaneOutcome { bits, status }
}

fn round_integral_magnitude(
    significand: u64,
    shift: u32,
    negative: bool,
    rounding: FpRoundingMode,
) -> (u64, bool) {
    let retained = if shift < 64 { significand >> shift } else { 0 };
    let discarded = if shift < 64 {
        significand & ((1_u64 << shift) - 1)
    } else {
        significand
    };
    if discarded == 0 {
        return (retained, false);
    }
    let increment = match rounding {
        FpRoundingMode::TiesToEven if shift < 64 => {
            let halfway = 1_u64 << (shift - 1);
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        FpRoundingMode::TiesAway if shift < 64 => discarded >= 1_u64 << (shift - 1),
        FpRoundingMode::TiesToEven | FpRoundingMode::TiesAway => false,
        FpRoundingMode::TowardPositive => !negative,
        FpRoundingMode::TowardNegative => negative,
        FpRoundingMode::TowardZero => false,
        FpRoundingMode::ToOdd => unreachable!("FRINT does not select round-to-odd"),
    };
    (retained + u64::from(increment), true)
}

#[derive(Clone, Copy)]
struct FpCompareOutcome {
    nzcv: u32,
    status: FpStatus,
}

// FCMP and FCMPE compare scalar S/D operands without using host floating-point
// ordering. FCMPE reports every NaN as invalid, while FCMP reports only a
// signaling NaN. Both produce Arm's unordered NZCV value for NaNs.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FCMP--FCMPE---Floating-point-Compare--scalar--
fn scalar_float_compare(
    state: &A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    compare_with_zero: bool,
) -> FpCompareOutcome {
    let fp_format = if fields.opc == 0 {
        FpFormat::Binary32
    } else {
        FpFormat::Binary64
    };
    let format = BinaryFormat::new(fp_format);
    let mask = if format.total_bits == 32 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let mut lhs = DecodedFloat::new(
        state
            .vector(fields.rn)
            .expect("normalized scalar comparison first operand") as u64
            & mask,
        format,
    );
    let mut rhs = DecodedFloat::new(
        if compare_with_zero {
            0
        } else {
            state
                .vector(fields.rm)
                .expect("normalized scalar comparison second operand") as u64
                & mask
        },
        format,
    );
    let mut status = FpStatus::default();

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = fields.signaling_compare
            || lhs.is_signaling_nan(format)
            || rhs.is_signaling_nan(format);
        return FpCompareOutcome {
            nzcv: (1 << 29) | (1 << 28),
            status,
        };
    }

    if state.fpcr() & (1 << 24) != 0 {
        for operand in [&mut lhs, &mut rhs] {
            if operand.is_subnormal() {
                status.input_denormal = true;
                *operand = DecodedFloat::new(operand.bits & format.sign_mask(), format);
            }
        }
    }

    let equal = (lhs.is_zero() && rhs.is_zero()) || lhs.bits == rhs.bits;
    let less = if equal {
        false
    } else if lhs.sign != rhs.sign {
        lhs.sign
    } else if lhs.sign {
        (lhs.bits & !format.sign_mask()) > (rhs.bits & !format.sign_mask())
    } else {
        lhs.bits < rhs.bits
    };
    let nzcv = if equal {
        (1 << 30) | (1 << 29)
    } else if less {
        1 << 31
    } else {
        1 << 29
    };
    FpCompareOutcome { nzcv, status }
}

#[derive(Clone, Copy)]
struct FpDivideControl {
    rounding: FpRoundingMode,
    default_nan: bool,
    flush_to_zero: bool,
}

impl FpDivideControl {
    fn from_fpcr(fpcr: u32) -> Self {
        let rounding = match (fpcr >> 22) & 3 {
            0 => FpRoundingMode::TiesToEven,
            1 => FpRoundingMode::TowardPositive,
            2 => FpRoundingMode::TowardNegative,
            3 => FpRoundingMode::TowardZero,
            _ => unreachable!(),
        };
        Self {
            rounding,
            default_nan: fpcr & (1 << 25) != 0,
            flush_to_zero: fpcr & (1 << 24) != 0,
        }
    }
}

#[derive(Clone, Copy)]
struct FpLaneOutcome {
    bits: u64,
    status: FpStatus,
}

#[derive(Clone, Copy)]
struct BinaryFormat {
    total_bits: u32,
    fraction_bits: u32,
    exponent_bits: u32,
    exponent_bias: i32,
}

impl BinaryFormat {
    fn new(format: FpFormat) -> Self {
        match format {
            FpFormat::Binary32 => Self {
                total_bits: 32,
                fraction_bits: 23,
                exponent_bits: 8,
                exponent_bias: 127,
            },
            FpFormat::Binary64 => Self {
                total_bits: 64,
                fraction_bits: 52,
                exponent_bits: 11,
                exponent_bias: 1023,
            },
            FpFormat::Binary16 => unreachable!("FDIV vector has no half-precision encoding"),
        }
    }

    fn sign_mask(self) -> u64 {
        1_u64 << (self.total_bits - 1)
    }

    fn fraction_mask(self) -> u64 {
        (1_u64 << self.fraction_bits) - 1
    }

    fn exponent_mask(self) -> u64 {
        ((1_u64 << self.exponent_bits) - 1) << self.fraction_bits
    }

    fn exponent_max(self) -> u64 {
        (1_u64 << self.exponent_bits) - 1
    }

    fn quiet_nan_bit(self) -> u64 {
        1_u64 << (self.fraction_bits - 1)
    }

    fn default_nan(self) -> u64 {
        self.exponent_mask() | self.quiet_nan_bit()
    }
}

#[derive(Clone, Copy)]
struct DecodedFloat {
    bits: u64,
    sign: bool,
    exponent_field: u64,
    fraction: u64,
    significand: u64,
    scale: i32,
}

impl DecodedFloat {
    fn new(bits: u64, format: BinaryFormat) -> Self {
        let sign = bits & format.sign_mask() != 0;
        let exponent_field = (bits & format.exponent_mask()) >> format.fraction_bits;
        let fraction = bits & format.fraction_mask();
        let (significand, scale) = if exponent_field == 0 {
            (
                fraction,
                1 - format.exponent_bias - format.fraction_bits as i32,
            )
        } else {
            (
                (1_u64 << format.fraction_bits) | fraction,
                exponent_field as i32 - format.exponent_bias - format.fraction_bits as i32,
            )
        };
        Self {
            bits,
            sign,
            exponent_field,
            fraction,
            significand,
            scale,
        }
    }

    fn is_zero(self) -> bool {
        self.exponent_field == 0 && self.fraction == 0
    }

    fn is_subnormal(self) -> bool {
        self.exponent_field == 0 && self.fraction != 0
    }

    fn is_infinite(self, format: BinaryFormat) -> bool {
        self.exponent_field == format.exponent_max() && self.fraction == 0
    }

    fn is_nan(self, format: BinaryFormat) -> bool {
        self.exponent_field == format.exponent_max() && self.fraction != 0
    }

    fn is_signaling_nan(self, format: BinaryFormat) -> bool {
        self.is_nan(format) && self.fraction & format.quiet_nan_bit() == 0
    }
}

fn divide_ieee_lane(
    lhs_bits: u64,
    rhs_bits: u64,
    fp_format: FpFormat,
    control: FpDivideControl,
) -> FpLaneOutcome {
    let format = BinaryFormat::new(fp_format);
    let mut status = FpStatus::default();
    let mut lhs = DecodedFloat::new(lhs_bits, format);
    let mut rhs = DecodedFloat::new(rhs_bits, format);

    if lhs.is_nan(format) || rhs.is_nan(format) {
        status.invalid_operation = lhs.is_signaling_nan(format) || rhs.is_signaling_nan(format);
        let bits = propagate_nan(lhs, rhs, format, control.default_nan);
        return FpLaneOutcome { bits, status };
    }

    for operand in [&mut lhs, &mut rhs] {
        if control.flush_to_zero && operand.is_subnormal() {
            status.input_denormal = true;
            operand.bits &= format.sign_mask();
            operand.exponent_field = 0;
            operand.fraction = 0;
            operand.significand = 0;
        }
    }

    let sign = lhs.sign ^ rhs.sign;
    let sign_bits = u64::from(sign) << (format.total_bits - 1);
    if (lhs.is_zero() && rhs.is_zero()) || (lhs.is_infinite(format) && rhs.is_infinite(format)) {
        status.invalid_operation = true;
        return FpLaneOutcome {
            bits: format.default_nan(),
            status,
        };
    }
    if lhs.is_infinite(format) {
        return FpLaneOutcome {
            bits: sign_bits | format.exponent_mask(),
            status,
        };
    }
    if rhs.is_infinite(format) || lhs.is_zero() {
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    if rhs.is_zero() {
        status.divide_by_zero = true;
        return FpLaneOutcome {
            bits: sign_bits | format.exponent_mask(),
            status,
        };
    }

    divide_finite(lhs, rhs, sign_bits, format, control, status)
}

fn propagate_nan(
    lhs: DecodedFloat,
    rhs: DecodedFloat,
    format: BinaryFormat,
    default_nan: bool,
) -> u64 {
    if default_nan {
        return format.default_nan();
    }
    let selected = if lhs.is_signaling_nan(format) {
        lhs
    } else if rhs.is_signaling_nan(format) {
        rhs
    } else if lhs.is_nan(format) {
        lhs
    } else {
        rhs
    };
    selected.bits | format.quiet_nan_bit()
}

fn divide_finite(
    lhs: DecodedFloat,
    rhs: DecodedFloat,
    sign_bits: u64,
    format: BinaryFormat,
    control: FpDivideControl,
    mut status: FpStatus,
) -> FpLaneOutcome {
    let lhs_top = 63 - lhs.significand.leading_zeros() as i32;
    let rhs_top = 63 - rhs.significand.leading_zeros() as i32;
    let mut ratio_exponent = lhs_top - rhs_top;
    if (u128::from(lhs.significand) << rhs_top) < (u128::from(rhs.significand) << lhs_top) {
        ratio_exponent -= 1;
    }
    let exponent = lhs.scale - rhs.scale + ratio_exponent;
    let minimum_normal = 1 - format.exponent_bias;
    let maximum_normal = format.exponent_bias;

    if exponent >= minimum_normal {
        let shift = rhs.scale - lhs.scale + exponent - format.fraction_bits as i32;
        let (mut significand, remainder, denominator) =
            scaled_quotient(lhs.significand, rhs.significand, shift);
        let inexact = remainder != 0;
        if should_round(
            significand,
            remainder,
            denominator,
            sign_bits != 0,
            control.rounding,
        ) {
            significand += 1;
        }
        let precision_limit = 1_u64 << (format.fraction_bits + 1);
        let mut rounded_exponent = exponent;
        if significand == precision_limit {
            significand >>= 1;
            rounded_exponent += 1;
        }
        if rounded_exponent > maximum_normal {
            status.overflow = true;
            status.inexact = true;
            return FpLaneOutcome {
                bits: overflow_result(sign_bits, format, control.rounding),
                status,
            };
        }
        status.inexact = inexact;
        return FpLaneOutcome {
            bits: sign_bits
                | (((rounded_exponent + format.exponent_bias) as u64) << format.fraction_bits)
                | (significand & format.fraction_mask()),
            status,
        };
    }

    let minimum_subnormal = minimum_normal - format.fraction_bits as i32;
    let shift = rhs.scale - lhs.scale + minimum_subnormal;
    let (mut fraction, remainder, denominator) =
        scaled_quotient(lhs.significand, rhs.significand, shift);
    let inexact = remainder != 0;
    if should_round(
        fraction,
        remainder,
        denominator,
        sign_bits != 0,
        control.rounding,
    ) {
        fraction += 1;
    }
    status.inexact = inexact;
    status.underflow = inexact;
    if fraction == 1_u64 << format.fraction_bits {
        return FpLaneOutcome {
            bits: sign_bits | (1_u64 << format.fraction_bits),
            status,
        };
    }
    if control.flush_to_zero && fraction != 0 {
        status.underflow = true;
        return FpLaneOutcome {
            bits: sign_bits,
            status,
        };
    }
    FpLaneOutcome {
        bits: sign_bits | fraction,
        status,
    }
}

// Returns floor((numerator / denominator) * 2^-shift), plus an exact
// remainder/denominator pair used for one final architectural rounding step.
fn scaled_quotient(numerator: u64, denominator: u64, shift: i32) -> (u64, u128, u128) {
    if shift >= 0 {
        let denominator_bits = 64 - denominator.leading_zeros() as i32;
        if denominator_bits + shift > 127 {
            // The exact scaled value is far below one half. Preserve a
            // nonzero sticky remainder for directed rounding without forming
            // an integer wider than the software helper's u128 bound.
            return (0, 1, 3);
        }
        let scaled_denominator = u128::from(denominator) << shift;
        let numerator = u128::from(numerator);
        return (
            (numerator / scaled_denominator) as u64,
            numerator % scaled_denominator,
            scaled_denominator,
        );
    }
    debug_assert!(64 - numerator.leading_zeros() as i32 - shift <= 127);
    let scaled_numerator = u128::from(numerator) << (-shift);
    let denominator = u128::from(denominator);
    (
        (scaled_numerator / denominator) as u64,
        scaled_numerator % denominator,
        denominator,
    )
}

fn should_round(
    retained: u64,
    remainder: u128,
    denominator: u128,
    negative: bool,
    rounding: FpRoundingMode,
) -> bool {
    if remainder == 0 {
        return false;
    }
    match rounding {
        FpRoundingMode::TiesToEven => {
            let twice = remainder << 1;
            twice > denominator || (twice == denominator && retained & 1 != 0)
        }
        FpRoundingMode::TowardPositive => !negative,
        FpRoundingMode::TowardNegative => negative,
        FpRoundingMode::TowardZero => false,
        FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
            unreachable!("FPCR cannot select this rounding mode")
        }
    }
}

fn overflow_result(sign_bits: u64, format: BinaryFormat, rounding: FpRoundingMode) -> u64 {
    let infinity = sign_bits | format.exponent_mask();
    let maximum_finite =
        sign_bits | ((format.exponent_max() - 1) << format.fraction_bits) | format.fraction_mask();
    match rounding {
        FpRoundingMode::TiesToEven => infinity,
        FpRoundingMode::TowardPositive if sign_bits == 0 => infinity,
        FpRoundingMode::TowardNegative if sign_bits != 0 => infinity,
        FpRoundingMode::TowardPositive
        | FpRoundingMode::TowardNegative
        | FpRoundingMode::TowardZero => maximum_finite,
        FpRoundingMode::TiesAway | FpRoundingMode::ToOdd => {
            unreachable!("FPCR cannot select this rounding mode")
        }
    }
}

fn merge_fp_status(destination: &mut FpStatus, source: FpStatus) {
    destination.invalid_operation |= source.invalid_operation;
    destination.divide_by_zero |= source.divide_by_zero;
    destination.overflow |= source.overflow;
    destination.underflow |= source.underflow;
    destination.inexact |= source.inexact;
    destination.input_denormal |= source.input_denormal;
}

fn fp_status_bits(status: FpStatus) -> u32 {
    u32::from(status.invalid_operation)
        | (u32::from(status.divide_by_zero) << 1)
        | (u32::from(status.overflow) << 2)
        | (u32::from(status.underflow) << 3)
        | (u32::from(status.inexact) << 4)
        | (u32::from(status.input_denormal) << 7)
}

fn fp_status_traps(status: FpStatus, fpcr: u32) -> bool {
    let status = fp_status_bits(status);
    let enables = ((fpcr >> 8) & 0x1f) | (((fpcr >> 15) & 1) << 7);
    status & enables != 0
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

fn integer_min_max(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: PairwiseOperation,
) {
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_bits = 8_u8 << fields.opc;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let lhs = state
        .vector(fields.rn)
        .expect("normalized SIMD first source register");
    let rhs = state
        .vector(fields.rm)
        .expect("normalized SIMD second source register");
    let mut result = 0_u128;

    for shift in (0..vector_bits).step_by(usize::from(lane_bits)) {
        let lhs_lane = (lhs >> shift) & lane_mask;
        let rhs_lane = (rhs >> shift) & lane_mask;
        let selected = match operation {
            PairwiseOperation::SignedMaximum => {
                if sign_extend(lhs_lane as u64, lane_bits)
                    >= sign_extend(rhs_lane as u64, lane_bits)
                {
                    lhs_lane
                } else {
                    rhs_lane
                }
            }
            PairwiseOperation::SignedMinimum => {
                if sign_extend(lhs_lane as u64, lane_bits)
                    <= sign_extend(rhs_lane as u64, lane_bits)
                {
                    lhs_lane
                } else {
                    rhs_lane
                }
            }
            PairwiseOperation::UnsignedMaximum => lhs_lane.max(rhs_lane),
            PairwiseOperation::UnsignedMinimum => lhs_lane.min(rhs_lane),
            PairwiseOperation::Add => unreachable!("ADD is not a minimum/maximum operation"),
        };
        result |= selected << shift;
    }
    assert!(state.set_vector(fields.rd, result));
}

// UZP1/2, TRN1/2, and ZIP1/2 select lanes from both source vectors before
// writing the destination, which also permits either source to alias Vd.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ZIP1--vector---Zip-vectors--primary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/ZIP2--vector---Zip-vectors--secondary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/TRN1--Transpose-vectors--primary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/TRN2--Transpose-vectors--secondary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UZP1--Unzip-vectors--primary--
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/UZP2--Unzip-vectors--secondary--
fn permute_two_source(
    state: &mut A64State,
    fields: crate::decode::a64::fp_simd::Operands,
    operation: PermuteOperation,
) {
    let vector_bits = if fields.vector_128 { 128_u32 } else { 64 };
    let lane_bits = 8_u32 << fields.opc;
    let lane_count = vector_bits / lane_bits;
    let half = lane_count / 2;
    let lane_mask = (1_u128 << lane_bits) - 1;
    let first = state
        .vector(fields.rn)
        .expect("normalized SIMD first source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized SIMD second source register");
    let mut result = 0_u128;

    for destination_lane in 0..lane_count {
        let (source, source_lane) = match operation {
            PermuteOperation::UnzipPrimary | PermuteOperation::UnzipSecondary => {
                let odd = u32::from(matches!(operation, PermuteOperation::UnzipSecondary));
                if destination_lane < half {
                    (first, destination_lane * 2 + odd)
                } else {
                    (second, (destination_lane - half) * 2 + odd)
                }
            }
            PermuteOperation::TransposePrimary | PermuteOperation::TransposeSecondary => {
                let odd = u32::from(matches!(operation, PermuteOperation::TransposeSecondary));
                let source = if destination_lane & 1 == 0 {
                    first
                } else {
                    second
                };
                (source, (destination_lane / 2) * 2 + odd)
            }
            PermuteOperation::ZipPrimary | PermuteOperation::ZipSecondary => {
                let upper = u32::from(matches!(operation, PermuteOperation::ZipSecondary));
                let source = if destination_lane & 1 == 0 {
                    first
                } else {
                    second
                };
                (source, destination_lane / 2 + upper * half)
            }
        };
        let lane = (source >> (source_lane * lane_bits)) & lane_mask;
        result |= lane << (destination_lane * lane_bits);
    }
    assert!(state.set_vector(fields.rd, result));
}

// EXT treats the two source vectors as one byte string and copies a vector-
// sized window beginning at the encoded byte offset. Reading both operands
// before committing the destination preserves architectural alias behavior.
// Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/EXT--Extract-vector-from-pair-of-vectors-
fn extract_vector(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let byte_count = if fields.vector_128 { 16_u32 } else { 8_u32 };
    let offset = u32::from(fields.immediate_4);
    let first = state
        .vector(fields.rn)
        .expect("normalized EXT first source register");
    let second = state
        .vector(fields.rm)
        .expect("normalized EXT second source register");
    let mut result = 0_u128;
    for destination_byte in 0..byte_count {
        let source_byte = destination_byte + offset;
        let byte = if source_byte < byte_count {
            first >> (source_byte * 8)
        } else {
            second >> ((source_byte - byte_count) * 8)
        } & 0xff;
        result |= byte << (destination_byte * 8);
    }
    assert!(state.set_vector(fields.rd, result));
}

// SHRN writes eight, four, or two narrowed lanes into one 64-bit half.
// SHRN clears the upper half while SHRN2 preserves the lower half and replaces
// the upper half. Arm ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/SHRN--SHRN2--Shift-right-narrow--immediate--
fn shift_right_narrow(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let bits = fields.helper_token.helper_abi_value();
    let immediate_high = (bits >> 19) & 0xf;
    let immediate_low = (bits >> 16) & 7;
    let destination_lane_bits = 8_u32 << (31 - immediate_high.leading_zeros());
    let source_lane_bits = destination_lane_bits * 2;
    let immediate = (immediate_high << 3) | immediate_low;
    let shift = source_lane_bits - immediate;
    let lane_count = 128 / source_lane_bits;
    let destination_mask = (1_u128 << destination_lane_bits) - 1;
    let source = state
        .vector(fields.rn)
        .expect("normalized SHRN source register");
    let mut narrowed = 0_u128;
    for lane in 0..lane_count {
        let value = source >> (lane * source_lane_bits);
        narrowed |= ((value >> shift) & destination_mask) << (lane * destination_lane_bits);
    }
    let result = if fields.vector_128 {
        let previous = state
            .vector(fields.rd)
            .expect("normalized SHRN2 destination register");
        (previous & u128::from(u64::MAX)) | (narrowed << 64)
    } else {
        narrowed
    };
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
