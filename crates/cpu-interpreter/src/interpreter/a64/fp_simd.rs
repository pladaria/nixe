use nixe_cpu::{
    decode::{DecodedOpcode, a64::fp_simd::Instruction},
    location::DecodedInstruction,
    memory::{
        MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering,
        MemoryValue,
    },
    semantics::a64::{
        SimdMemoryMode, SimdMemoryShape, simd_memory_access_size, simd_multiple_structure_shape,
        simd_pair_access_size, simd_single_structure_shape,
    },
    state::a64::A64State,
};
use nixe_memory::GuestVirtualAddress;

use super::{
    advance,
    memory::{MemoryStepError, ordinary_read, ordinary_write},
    read, register_offset_address, sign_extend,
};
use crate::interpreter::{InstructionStep, InterpreterContext, InterpreterError};

type MemoryStep = Result<(), MemoryStepError>;

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
            let size = simd_memory_access_size(fields.size, fields.opc)
                .expect("allocation validation rejects invalid SIMD transfer sizes");
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
                context,
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
        Instruction::MemoryPair(_) => Some(vector_pair(context, state, fields)),
        Instruction::MemoryRegister(_) => {
            let size = simd_memory_access_size(fields.size, fields.opc)
                .expect("allocation validation rejects invalid SIMD transfer sizes");
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
            Some(vector_transfer(context, state, fields, address, size))
        }
        Instruction::MemoryMultipleStructures(_)
        | Instruction::MemoryMultipleStructuresPostIndex(_) => {
            let Some(shape) = simd_multiple_structure_shape(fields) else {
                return Err(super::super::unsupported(decoded));
            };
            Some(vector_multiple_structures(
                context,
                state,
                fields,
                shape,
                matches!(
                    instruction,
                    Instruction::MemoryMultipleStructuresPostIndex(_)
                ),
            ))
        }
        Instruction::MemorySingleStructure(_) | Instruction::MemorySingleStructurePostIndex(_) => {
            let Some(shape) = simd_single_structure_shape(fields) else {
                return Err(super::super::unsupported(decoded));
            };
            Some(vector_single_structure(
                context,
                state,
                fields,
                shape,
                matches!(instruction, Instruction::MemorySingleStructurePostIndex(_)),
            ))
        }
        _ => return Err(super::super::unsupported(decoded)),
    };
    if let Some(Err(error)) = result {
        return match error {
            MemoryStepError::Data(fault) => {
                Ok(InstructionStep::data_fault(decoded.location, fault))
            }
            MemoryStepError::Direct(detail) => Err(InterpreterError::DirectMemory {
                source: decoded.location,
                detail,
            }),
        };
    }
    advance(state);
    Ok(InstructionStep::Continue)
}

// FMOV (register) copies the scalar bit pattern without floating-point
// processing. This preserves every NaN payload, subnormal, and signed zero,
// clears the remaining destination bits, and leaves FPCR/FPSR unchanged. Arm
// ARM DDI 0602 (2025-12):
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/FMOV--register---Floating-point-Move-register--
fn vector_pair(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> MemoryStep {
    let size = simd_pair_access_size(fields.size)
        .expect("allocation validation rejects invalid SIMD pair sizes");
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
        let first_value = read_vector(context, first, size)?;
        let second_value = read_vector(context, second, size)?;
        assert!(state.set_vector(fields.rd, first_value));
        assert!(state.set_vector(fields.rt2, second_value));
    } else {
        write_vector(context, first, size, state, fields.rd)?;
        write_vector(context, second, size, state, fields.rt2)?;
    }
    if matches!(fields.mode, 1 | 3) {
        super::write(state, fields.rn, 64, true, base.wrapping_add_signed(offset));
    }
    Ok(())
}

fn vector_multiple_structures(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
    shape: SimdMemoryShape,
    post_index: bool,
) -> MemoryStep {
    let vector_size = if shape.vector_bytes == 16 {
        MemoryAccessSize::Quadword
    } else {
        MemoryAccessSize::Doubleword
    };
    let base = read(state, fields.rn, 64, true);
    let mut address = GuestVirtualAddress::new(base);
    if shape.structure_registers == 1 {
        for repetition in 0..shape.repetitions {
            let register = fields.rd.wrapping_add(repetition) & 31;
            if fields.load {
                let value = read_vector(context, address, vector_size)?;
                assert!(state.set_vector(register, value));
            } else {
                write_vector(context, address, vector_size, state, register)?;
            }
            address = address.wrapping_add(u64::from(shape.vector_bytes));
        }
    } else {
        let lane_bits = shape.element_size.bytes() as u32 * 8;
        for lane in 0..shape.elements_per_register {
            for register_offset in 0..shape.structure_registers {
                let register = fields.rd.wrapping_add(register_offset) & 31;
                if fields.load {
                    let value = read_vector(context, address, shape.element_size)?;
                    insert_lane(state, register, lane, lane_bits, value);
                } else {
                    write_lane(
                        context,
                        address,
                        shape.element_size,
                        vector_lane(state, register, lane, lane_bits),
                    )?;
                }
                address = address.wrapping_add(shape.element_size.bytes() as u64);
            }
        }
        if !fields.vector_128 && fields.load {
            for register_offset in 0..shape.structure_registers {
                let register = fields.rd.wrapping_add(register_offset) & 31;
                let value = state
                    .vector(register)
                    .expect("normalized multiple-structure destination register")
                    & u128::from(u64::MAX);
                assert!(state.set_vector(register, value));
            }
        }
    }
    if post_index {
        let offset = if fields.rm == 31 {
            u64::from(shape.immediate_post_index)
        } else {
            read(state, fields.rm, 64, false)
        };
        super::write(state, fields.rn, 64, true, base.wrapping_add(offset));
    }
    Ok(())
}

fn vector_single_structure(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
    shape: SimdMemoryShape,
    post_index: bool,
) -> MemoryStep {
    let base = read(state, fields.rn, 64, true);
    let mut address = GuestVirtualAddress::new(base);
    let lane_bits = shape.element_size.bytes() as u32 * 8;
    for register_offset in 0..shape.structure_registers {
        let register = fields.rd.wrapping_add(register_offset) & 31;
        match shape.mode {
            SimdMemoryMode::Lane(lane) if fields.load => {
                let value = read_vector(context, address, shape.element_size)?;
                insert_lane(state, register, lane, lane_bits, value);
            }
            SimdMemoryMode::Lane(lane) => {
                write_lane(
                    context,
                    address,
                    shape.element_size,
                    vector_lane(state, register, lane, lane_bits),
                )?;
            }
            SimdMemoryMode::Replicate => {
                let value = read_vector(context, address, shape.element_size)?;
                let mut vector = 0_u128;
                for lane in 0..shape.elements_per_register {
                    vector |= value << (u32::from(lane) * lane_bits);
                }
                assert!(state.set_vector(register, vector));
            }
            SimdMemoryMode::Multiple => unreachable!("single-structure shape has a single mode"),
        }
        address = address.wrapping_add(shape.element_size.bytes() as u64);
    }
    if post_index {
        let offset = if fields.rm == 31 {
            u64::from(shape.immediate_post_index)
        } else {
            read(state, fields.rm, 64, false)
        };
        super::write(state, fields.rn, 64, true, base.wrapping_add(offset));
    }
    Ok(())
}

fn vector_lane(state: &A64State, register: u8, lane: u8, lane_bits: u32) -> u128 {
    let vector = state
        .vector(register)
        .expect("normalized single-structure source register");
    (vector >> (u32::from(lane) * lane_bits)) & ((1_u128 << lane_bits) - 1)
}

fn insert_lane(state: &mut A64State, register: u8, lane: u8, lane_bits: u32, value: u128) {
    let shift = u32::from(lane) * lane_bits;
    let mask = ((1_u128 << lane_bits) - 1) << shift;
    let previous = state
        .vector(register)
        .expect("normalized single-structure destination register");
    assert!(state.set_vector(register, (previous & !mask) | ((value << shift) & mask)));
}

fn write_lane(
    context: InterpreterContext<'_>,
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
    ordinary_write(context, address, value, vector_access(size))
}

fn vector_transfer(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
) -> MemoryStep {
    if fields.load {
        let value = read_vector(context, address, size)?;
        assert!(state.set_vector(fields.rd, value));
    } else {
        write_vector(context, address, size, state, fields.rd)?;
    }
    Ok(())
}

fn read_vector(
    context: InterpreterContext<'_>,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
) -> Result<u128, MemoryStepError> {
    let value = ordinary_read(context, address, vector_access(size))?;
    Ok(match value {
        MemoryValue::U8(value) => u128::from(value),
        MemoryValue::U16(value) => u128::from(value),
        MemoryValue::U32(value) => u128::from(value),
        MemoryValue::U64(value) => u128::from(value),
        MemoryValue::U128(value) => value,
    })
}

fn write_vector(
    context: InterpreterContext<'_>,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
    state: &A64State,
    register: u8,
) -> Result<(), MemoryStepError> {
    let value = state.vector(register).expect("normalized vector register");
    let value = match size {
        MemoryAccessSize::Byte => MemoryValue::U8(value as u8),
        MemoryAccessSize::Halfword => MemoryValue::U16(value as u16),
        MemoryAccessSize::Word => MemoryValue::U32(value as u32),
        MemoryAccessSize::Doubleword => MemoryValue::U64(value as u64),
        MemoryAccessSize::Quadword => MemoryValue::U128(value),
    };
    ordinary_write(context, address, value, vector_access(size))
}

fn vector_access(size: MemoryAccessSize) -> MemoryAccess {
    MemoryAccess::new(
        size,
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    )
}
