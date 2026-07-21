//! Modular T32 reference interpreter.

mod control;
mod integer;
mod memory;

use super::{InterpreterContext, InterpreterError, InterpreterOutcome};
use crate::{
    address::GuestVirtualAddress,
    decode::{
        DecodedOpcode,
        t32::{T32Instruction, normalize},
    },
    location::{DecodedInstruction, LocationDescriptor},
    semantics::conditions::evaluate_a32,
    state::a32::{A32State, ItState},
};

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    let instruction = normalize(&decoded.instruction, decoded.encoding);
    if let T32Instruction::Control(crate::decode::t32::control::Instruction::It {
        first_condition,
        mask,
    }) = instruction
    {
        return execute_it(state, decoded, first_condition, mask);
    }
    let it_state = state.cpsr().it_state();
    let executes = it_state
        .current_condition()
        .is_none_or(|condition| evaluate_a32(condition, state.cpsr().bits()));
    state.set_cpsr(state.cpsr().with_it_state(it_state.advance()));
    if !executes {
        advance(state, decoded)?;
        return Ok(resume(decoded, state));
    }
    let control = match instruction {
        T32Instruction::Control(instruction) => {
            return control::execute(state, decoded, instruction);
        }
        T32Instruction::Integer(instruction) => {
            integer::execute(state, decoded, instruction, it_state.is_active())?
        }
        T32Instruction::Memory(instruction) => {
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
    };
    if control == super::aarch32::SemanticControl::Continue {
        advance(state, decoded)?;
    }
    Ok(resume(decoded, state))
}

fn execute_it(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    first_condition: u8,
    mask: u8,
) -> Result<InterpreterOutcome, InterpreterError> {
    if state.cpsr().it_state().is_active() {
        return Ok(InterpreterOutcome::Exception {
            source: decoded.location,
            kind: crate::ir::terminator::ExceptionKind::UndefinedInstruction,
            syndrome: None,
        });
    }
    let Some(it_state) = ItState::from_encoding(first_condition, mask) else {
        return Ok(InterpreterOutcome::Exception {
            source: decoded.location,
            kind: crate::ir::terminator::ExceptionKind::UndefinedInstruction,
            syndrome: None,
        });
    };
    state.set_cpsr(state.cpsr().with_it_state(it_state));
    advance(state, decoded)?;
    Ok(resume(decoded, state))
}

fn advance(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<(), InterpreterError> {
    state
        .set_instruction_address(
            state
                .instruction_address()
                .wrapping_add(u32::from(decoded.encoding.size().bytes())),
        )
        .map_err(|error| branch_error(decoded, error))
}
fn resume(decoded: &DecodedInstruction<DecodedOpcode>, state: &A32State) -> InterpreterOutcome {
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
