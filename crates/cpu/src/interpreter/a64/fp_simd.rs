use crate::{
    address::GuestVirtualAddress,
    decode::{DecodedOpcode, a64::fp_simd::Instruction},
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
        Instruction::MoveImmediate32(_) => {
            move_immediate_32(state, fields);
            None
        }
        Instruction::UnsignedMoveToGeneral(_) => {
            unsigned_move_to_general(state, fields);
            None
        }
        Instruction::Integer(_) => {
            integer_add_sub(state, fields);
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

fn move_immediate_32(state: &mut A64State, fields: crate::decode::a64::fp_simd::Operands) {
    let shift = (fields.cmode >> 1) * 8;
    let lane = u128::from(fields.immediate_8) << shift;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let value = replicate(
        lane,
        BitWidth::new(32).expect("32-bit SIMD immediate lane"),
        BitWidth::new(vector_bits).expect("allocated SIMD vector width"),
    )
    .expect("allocated SIMD immediate arrangement");
    assert!(state.set_vector(fields.rd, value));
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
