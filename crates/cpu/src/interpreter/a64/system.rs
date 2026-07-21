use core::sync::atomic::{Ordering, fence};

use crate::{
    decode::{
        DecodedOpcode,
        a64::system::{Instruction, Operands},
    },
    location::{DecodedInstruction, LocationDescriptor},
    state::a64::{A64State, Nzcv},
};

use super::{advance, read, resume, write};
use crate::interpreter::{InterpreterError, InterpreterOutcome};

pub(super) fn execute(
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    let fields = instruction.operands();
    let outcome = match instruction {
        Instruction::Hint(_) => execute_hint(state, decoded.location, fields),
        Instruction::ReadRegister(_) => execute_mrs(state, fields),
        Instruction::WriteRegister(_) => execute_msr(state, fields),
        Instruction::Barrier(_) => execute_barrier(fields),
        Instruction::System(_) => false,
    };
    if !outcome {
        return Err(super::super::unsupported(decoded));
    }
    advance(state);
    Ok(resume(state, decoded))
}

fn execute_hint(_state: &mut A64State, _source: LocationDescriptor, fields: Operands) -> bool {
    match fields.hint {
        0 => true,
        // YIELD/WFE/WFI/SEV/SEVL require scheduler/event callbacks. Treating
        // them as no-ops would make this reference engine an invalid oracle.
        1..=5 => false,
        _ => false,
    }
}

fn execute_mrs(state: &mut A64State, fields: Operands) -> bool {
    let value = match fields.system_key {
        0xd53b_4200 => u64::from(state.nzcv().bits()),
        0xd53b_4400 => u64::from(state.fpcr()),
        0xd53b_4420 => u64::from(state.fpsr()),
        0xd53b_d040 => state.tpidr_el0(),
        0xd53b_d060 => state.tpidrro_el0(),
        _ => return false,
    };
    write(state, fields.rt, 64, false, value);
    true
}

fn execute_msr(state: &mut A64State, fields: Operands) -> bool {
    let value = read(state, fields.rt, 64, false);
    match fields.system_key {
        0xd51b_4200 => state.set_nzcv(Nzcv::from_bits(value as u32)),
        0xd51b_4400 => state.set_fpcr(value as u32),
        0xd51b_4420 => state.set_fpsr(value as u32),
        0xd51b_d040 => state.set_tpidr_el0(value),
        // TPIDRRO_EL0 is runtime-owned and architecturally read-only here.
        0xd51b_d060 => return false,
        _ => return false,
    }
    true
}

fn execute_barrier(fields: Operands) -> bool {
    match fields.barrier_opcode {
        4 | 5 if valid_barrier_option(fields.barrier_option) => {
            // This supplies the local reference engine's host ordering. The
            // guest multicore scheduler/memory-model contract remains Phase 4.
            fence(Ordering::SeqCst);
            true
        }
        6 if fields.barrier_option == 15 => true,
        _ => false,
    }
}

fn valid_barrier_option(option: u8) -> bool {
    option & 3 != 0 && option >> 2 <= 3
}
