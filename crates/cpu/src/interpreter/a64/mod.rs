//! A64 reference interpretation split by architectural instruction family.

mod control;
mod fp_simd;
mod integer;
mod memory;
mod system;

use crate::{
    address::GuestVirtualAddress,
    decode::{DecodedOpcode, a64::A64Instruction},
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    state::a64::{A64GeneralRegister, A64Register, A64State},
};

use super::{InterpreterContext, InterpreterError, InterpreterOutcome};

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    match crate::decode::a64::normalize(&decoded.instruction, decoded.encoding) {
        A64Instruction::Control(instruction) => control::execute(state, decoded, instruction),
        A64Instruction::System(instruction) => {
            system::execute(context, state, decoded, instruction)
        }
        A64Instruction::Integer(instruction) => integer::execute(state, decoded, instruction),
        A64Instruction::Memory(instruction) => {
            memory::execute(context, state, decoded, instruction)
        }
        A64Instruction::FpSimd(instruction) => {
            fp_simd::execute(context, state, decoded, instruction)
        }
        A64Instruction::RecognizedFallback { .. } => Err(super::unsupported(decoded)),
    }
}

fn resume(state: &A64State, decoded: &DecodedInstruction<DecodedOpcode>) -> InterpreterOutcome {
    InterpreterOutcome::Resume(LocationDescriptor::new(
        GuestVirtualAddress::new(state.pc()),
        ExecutionState::A64,
        decoded.location.profile_id,
    ))
}

fn advance(state: &mut A64State) {
    state.set_pc(state.pc().wrapping_add(4));
}

fn zero_register(index: u8) -> A64Register {
    A64GeneralRegister::new(index).map_or(A64Register::Zero, A64Register::General)
}

fn stack_pointer_register(index: u8) -> A64Register {
    A64GeneralRegister::new(index).map_or(A64Register::StackPointer, A64Register::General)
}

fn read(state: &A64State, index: u8, width: u8, register31_is_sp: bool) -> u64 {
    let register = if register31_is_sp {
        stack_pointer_register(index)
    } else {
        zero_register(index)
    };
    if width == 64 {
        state.read_x(register)
    } else {
        u64::from(state.read_w(register))
    }
}

fn write(state: &mut A64State, index: u8, width: u8, register31_is_sp: bool, value: u64) {
    let register = if register31_is_sp {
        stack_pointer_register(index)
    } else {
        zero_register(index)
    };
    if width == 64 {
        state.write_x(register, value);
    } else {
        state.write_w(register, value as u32);
    }
}

fn sign_extend(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}
