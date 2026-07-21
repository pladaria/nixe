//! Modular A32 reference interpreter.

mod control;
mod fp_simd;
mod integer;
mod memory;

use super::{InterpreterContext, InterpreterError, InterpreterOutcome};
use crate::{
    address::GuestVirtualAddress,
    decode::{
        DecodedOpcode,
        a32::{A32Instruction, normalize},
    },
    location::{DecodedInstruction, LocationDescriptor},
    semantics::conditions::evaluate_a32,
    state::a32::A32State,
};

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    let normalized = normalize(&decoded.instruction, decoded.encoding);
    if !evaluate_a32(normalized.condition, state.cpsr().bits()) {
        advance(state, decoded)?;
        return Ok(resume(decoded, state));
    }
    let control = match normalized.instruction {
        A32Instruction::Control(instruction) => {
            return control::execute(state, decoded, instruction);
        }
        A32Instruction::Integer(instruction) => integer::execute(state, decoded, instruction)?,
        A32Instruction::Memory(instruction) => {
            match memory::execute(context, state, decoded, instruction)? {
                memory::Execution::Control(control) => control,
                memory::Execution::Fault(fault) => {
                    return Ok(InterpreterOutcome::DataAbort {
                        source: decoded.location,
                        fault,
                    });
                }
            }
        }
        A32Instruction::FpSimd(instruction) => {
            match fp_simd::execute(context, state, decoded, instruction)? {
                fp_simd::Execution::Control(control) => control,
                fp_simd::Execution::Fault(fault) => {
                    return Ok(InterpreterOutcome::DataAbort {
                        source: decoded.location,
                        fault,
                    });
                }
            }
        }
    };
    if control == super::aarch32::SemanticControl::Continue {
        advance(state, decoded)?;
    }
    Ok(resume(decoded, state))
}

fn advance(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<(), InterpreterError> {
    state
        .set_instruction_address(state.instruction_address().wrapping_add(4))
        .map_err(|error| InterpreterError::ContextMismatch {
            source: decoded.location,
            reason: error.to_string().into(),
        })
}

pub(super) fn resume(
    decoded: &DecodedInstruction<DecodedOpcode>,
    state: &A32State,
) -> InterpreterOutcome {
    InterpreterOutcome::Resume(LocationDescriptor::new(
        GuestVirtualAddress::new(u64::from(state.instruction_address())),
        state.execution_state(),
        decoded.location.profile_id,
    ))
}

fn branch_error(
    decoded: &DecodedInstruction<DecodedOpcode>,
    error: crate::state::a32::InvalidBranchTarget,
) -> InterpreterError {
    InterpreterError::ContextMismatch {
        source: decoded.location,
        reason: error.to_string().into(),
    }
}
