//! Encoding-independent AArch32 reference semantics shared by A32 and T32.

use crate::{
    address::{AddressSpaceId, GuestVirtualAddress},
    decode::aarch32::{
        DataOperation, DataProcessing, MemoryOffset, MemorySize, MultipleTransfer, ShiftAmount,
        ShifterOperand, SingleTransfer,
    },
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    semantics::{
        arithmetic::{add_with_carry, subtract_with_carry},
        bits::BitWidth,
        shifts::a32_shift_with_carry,
    },
    state::a32::{A32GeneralRegister, A32State, Cpsr},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SemanticControl {
    Continue,
    Branch,
}

pub(super) fn read_register(state: &A32State, index: u8, align_pc: bool) -> u32 {
    if index == 15 {
        let pc = state.read_pc();
        if align_pc { pc & !3 } else { pc }
    } else {
        state.read_r(A32GeneralRegister::new(index).expect("normalized AArch32 register"))
    }
}

pub(super) fn write_register(
    state: &mut A32State,
    index: u8,
    value: u32,
) -> Result<SemanticControl, crate::state::a32::InvalidBranchTarget> {
    if index == 15 {
        state.branch_exchange(value)?;
        Ok(SemanticControl::Branch)
    } else {
        state.write_r(
            A32GeneralRegister::new(index).expect("normalized AArch32 register"),
            value,
        );
        Ok(SemanticControl::Continue)
    }
}

pub(super) fn execute_data_processing(
    state: &mut A32State,
    instruction: DataProcessing,
) -> Result<SemanticControl, crate::state::a32::InvalidBranchTarget> {
    let carry_in = state.cpsr().carry();
    let (operand2, shifter_carry) = shifter_operand(state, instruction.operand2, carry_in);
    let lhs = read_register(state, instruction.rn, false);
    let width = BitWidth::new(32).expect("constant width");
    let mut arithmetic = None;
    let result = match instruction.operation {
        DataOperation::And | DataOperation::Test => lhs & operand2,
        DataOperation::ExclusiveOr | DataOperation::TestExclusiveOr => lhs ^ operand2,
        DataOperation::Subtract | DataOperation::Compare => {
            let r = subtract_with_carry(lhs.into(), operand2.into(), true, width);
            arithmetic = Some(r);
            r.result as u32
        }
        DataOperation::ReverseSubtract => {
            let r = subtract_with_carry(operand2.into(), lhs.into(), true, width);
            arithmetic = Some(r);
            r.result as u32
        }
        DataOperation::Add | DataOperation::CompareNegative => {
            let r = add_with_carry(lhs.into(), operand2.into(), false, width);
            arithmetic = Some(r);
            r.result as u32
        }
        DataOperation::AddCarry => {
            let r = add_with_carry(lhs.into(), operand2.into(), carry_in, width);
            arithmetic = Some(r);
            r.result as u32
        }
        DataOperation::SubtractCarry => {
            let r = subtract_with_carry(lhs.into(), operand2.into(), carry_in, width);
            arithmetic = Some(r);
            r.result as u32
        }
        DataOperation::ReverseSubtractCarry => {
            let r = subtract_with_carry(operand2.into(), lhs.into(), carry_in, width);
            arithmetic = Some(r);
            r.result as u32
        }
        DataOperation::Or => lhs | operand2,
        DataOperation::Move => operand2,
        DataOperation::BitClear => lhs & !operand2,
        DataOperation::MoveNot => !operand2,
    };
    if instruction.set_flags {
        let (carry, overflow) = arithmetic.map_or((shifter_carry, state.cpsr().overflow()), |r| {
            (r.carry_out, r.overflow)
        });
        set_nzcv(state, result, carry, overflow);
    }
    if instruction.operation.is_test() {
        Ok(SemanticControl::Continue)
    } else {
        write_register(state, instruction.rd, result)
    }
}

fn shifter_operand(state: &A32State, operand: ShifterOperand, carry_in: bool) -> (u32, bool) {
    match operand {
        ShifterOperand::Immediate { value, rotation } => (
            value,
            if rotation == 0 {
                carry_in
            } else {
                value & 0x8000_0000 != 0
            },
        ),
        ShifterOperand::Register { rm, shift } => {
            let value = read_register(state, rm, false);
            let amount = match shift.amount {
                ShiftAmount::Immediate(value) => u32::from(value),
                ShiftAmount::Register(rs) => read_register(state, rs, false) & 0xff,
            };
            let result = a32_shift_with_carry(value, shift.kind, amount, carry_in)
                .expect("normalized AArch32 shift");
            (result.result as u32, result.carry_out)
        }
    }
}

pub(super) fn execute_multiply(
    state: &mut A32State,
    instruction: crate::decode::aarch32::Multiply,
) -> Result<SemanticControl, crate::state::a32::InvalidBranchTarget> {
    let product = read_register(state, instruction.rm, false).wrapping_mul(read_register(
        state,
        instruction.rs,
        false,
    ));
    let result = if instruction.accumulate {
        product.wrapping_add(read_register(state, instruction.rn, false))
    } else {
        product
    };
    if instruction.set_flags {
        let cpsr = state.cpsr();
        set_nzcv(state, result, cpsr.carry(), cpsr.overflow());
    }
    write_register(state, instruction.rd, result)
}

fn set_nzcv(state: &mut A32State, result: u32, carry: bool, overflow: bool) {
    let old = state.cpsr().bits();
    let flags = if result & 0x8000_0000 != 0 {
        Cpsr::N
    } else {
        0
    } | if result == 0 { Cpsr::Z } else { 0 }
        | if carry { Cpsr::C } else { 0 }
        | if overflow { Cpsr::V } else { 0 };
    state.set_cpsr(Cpsr::from_bits(
        (old & !(Cpsr::N | Cpsr::Z | Cpsr::C | Cpsr::V)) | flags,
    ));
}

fn access_size(size: MemorySize) -> MemoryAccessSize {
    match size {
        MemorySize::Byte => MemoryAccessSize::Byte,
        MemorySize::Halfword => MemoryAccessSize::Halfword,
        MemorySize::Word => MemoryAccessSize::Word,
    }
}
fn access(size: MemorySize) -> MemoryAccess {
    MemoryAccess::new(
        access_size(size),
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    )
}

pub(super) fn execute_single(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A32State,
    instruction: SingleTransfer,
) -> Result<SemanticControl, DataAccessFault> {
    let base = read_register(state, instruction.rn, instruction.rn == 15);
    let raw_offset = match instruction.offset {
        MemoryOffset::Immediate(value) => value,
        MemoryOffset::Register { rm, shift } => {
            let value = read_register(state, rm, false);
            let amount = match shift.amount {
                ShiftAmount::Immediate(value) => u32::from(value),
                ShiftAmount::Register(rs) => read_register(state, rs, false) & 0xff,
            };
            a32_shift_with_carry(value, shift.kind, amount, state.cpsr().carry())
                .expect("normalized memory shift")
                .result as u32
        }
    };
    let offset_address = if instruction.add {
        base.wrapping_add(raw_offset)
    } else {
        base.wrapping_sub(raw_offset)
    };
    let address = if instruction.pre_index {
        offset_address
    } else {
        base
    };
    let address = GuestVirtualAddress::new(u64::from(address));
    let control = if instruction.load {
        let value = memory
            .read(address_space, address, access(instruction.size))?
            .value;
        let value = match (instruction.size, value, instruction.signed) {
            (MemorySize::Byte, MemoryValue::U8(value), true) => i32::from(value as i8) as u32,
            (MemorySize::Byte, MemoryValue::U8(value), false) => u32::from(value),
            (MemorySize::Halfword, MemoryValue::U16(value), true) => i32::from(value as i16) as u32,
            (MemorySize::Halfword, MemoryValue::U16(value), false) => u32::from(value),
            (MemorySize::Word, MemoryValue::U32(value), _) => value,
            _ => unreachable!("memory contract returns the requested width"),
        };
        write_register(state, instruction.rt, value).map_err(|_| {
            DataAccessFault::new(
                address_space,
                address,
                crate::memory::DataAccessKind::Read,
                crate::memory::DataAccessFaultReason::Injected("invalid loaded PC".into()),
            )
        })?
    } else {
        let value = read_register(state, instruction.rt, false);
        let value = match instruction.size {
            MemorySize::Byte => MemoryValue::U8(value as u8),
            MemorySize::Halfword => MemoryValue::U16(value as u16),
            MemorySize::Word => MemoryValue::U32(value),
        };
        memory.write(address_space, address, access(instruction.size), value)?;
        SemanticControl::Continue
    };
    if instruction.writeback {
        write_register(state, instruction.rn, offset_address)
            .expect("writeback base cannot be PC in MVP");
    }
    Ok(control)
}

pub(super) fn execute_multiple(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A32State,
    instruction: MultipleTransfer,
) -> Result<SemanticControl, DataAccessFault> {
    let count = instruction.registers.count_ones();
    let base = read_register(state, instruction.rn, false);
    let start = match (instruction.increment, instruction.before) {
        (true, false) => base,
        (true, true) => base.wrapping_add(4),
        (false, true) => base.wrapping_sub(count * 4),
        (false, false) => base.wrapping_sub((count.saturating_sub(1)) * 4),
    };
    let mut control = SemanticControl::Continue;
    let mut address = start;
    for register in 0..16_u8 {
        if instruction.registers & (1 << register) == 0 {
            continue;
        }
        let guest = GuestVirtualAddress::new(u64::from(address));
        if instruction.load {
            let MemoryValue::U32(value) = memory
                .read(address_space, guest, access(MemorySize::Word))?
                .value
            else {
                unreachable!()
            };
            control = write_register(state, register, value).map_err(|_| {
                DataAccessFault::new(
                    address_space,
                    guest,
                    crate::memory::DataAccessKind::Read,
                    crate::memory::DataAccessFaultReason::Injected("invalid loaded PC".into()),
                )
            })?;
        } else {
            memory.write(
                address_space,
                guest,
                access(MemorySize::Word),
                MemoryValue::U32(read_register(state, register, false)),
            )?;
        }
        address = address.wrapping_add(4);
    }
    if instruction.writeback {
        let final_base = if instruction.increment {
            base.wrapping_add(count * 4)
        } else {
            base.wrapping_sub(count * 4)
        };
        write_register(state, instruction.rn, final_base)
            .expect("multiple-transfer base is an ordinary register");
    }
    Ok(control)
}
