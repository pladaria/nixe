use nixe_cpu::{
    decode::{DecodedOpcode, a64::fp_simd::Instruction},
    location::DecodedInstruction,
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    state::a64::A64State,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use super::{advance, read, register_offset_address, sign_extend};
use crate::interpreter::{InstructionStep, InterpreterContext, InterpreterError};

type MemoryStep = Result<(), DataAccessFault>;

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InstructionStep, InterpreterError> {
    let fields = instruction.operands();
    match nixe_cpu::semantics::a64_fp_simd::execute(state, instruction) {
        Ok(()) => {
            advance(state);
            return Ok(InstructionStep::Continue);
        }
        Err(nixe_cpu::semantics::a64_fp_simd::A64FpSimdError::Trap) => {
            return Ok(InstructionStep::architectural_exception(
                decoded.location,
                nixe_cpu::exception::ExceptionKind::FloatingPoint,
                None,
            ));
        }
        Err(nixe_cpu::semantics::a64_fp_simd::A64FpSimdError::Unsupported) => {}
    }

    let result = match instruction {
        Instruction::MemoryUnsigned(_)
        | Instruction::MemoryUnscaled(_)
        | Instruction::MemoryPostIndex(_)
        | Instruction::MemoryPreIndex(_) => {
            let memory = context.memory();
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
            let memory = context.memory();
            Some(vector_pair(
                memory,
                context.process().address_space_id(),
                state,
                fields,
            ))
        }
        Instruction::MemoryRegister(_) => {
            let memory = context.memory();
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
            let memory = context.memory();
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
            let memory = context.memory();
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
        return Ok(InstructionStep::data_fault(decoded.location, fault));
    }
    advance(state);
    Ok(InstructionStep::Continue)
}

// FMOV (register) copies the scalar bit pattern without floating-point
// processing. This preserves every NaN payload, subnormal, and signed zero,
// clears the remaining destination bits, and leaves FPCR/FPSR unchanged. Arm
// ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--register---Floating-point-Move-register--
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
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

fn insert_lane(state: &mut A64State, register: u8, lane: u8, lane_bits: u32, value: u128) {
    let shift = u32::from(lane) * lane_bits;
    let mask = ((1_u128 << lane_bits) - 1) << shift;
    let previous = state
        .vector(register)
        .expect("normalized single-structure destination register");
    assert!(state.set_vector(register, (previous & !mask) | ((value << shift) & mask)));
}

fn single_structure_shape(
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> (MemoryAccessSize, u8) {
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
    address_space: AddressSpaceId,
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

fn vector_access_size(
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
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
    address_space: AddressSpaceId,
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
    address_space: AddressSpaceId,
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
