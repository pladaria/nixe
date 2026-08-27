//! Modular T32 reference interpreter.

mod control;
mod integer;
mod memory;

use super::{InstructionStep, InterpreterContext, InterpreterError};
use nixe_cpu::{
    decode::{
        DecodedOpcode,
        t32::{T32Instruction, normalize},
    },
    location::DecodedInstruction,
    semantics::conditions::evaluate_a32,
    state::a32::{A32State, ItState},
};
pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InstructionStep, InterpreterError> {
    let instruction = normalize(&decoded.instruction, decoded.encoding);
    if let T32Instruction::Control(nixe_cpu::decode::t32::control::Instruction::It {
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
        return Ok(InstructionStep::Continue);
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
                    return Ok(InstructionStep::data_fault(decoded.location, fault));
                }
            }
        }
    };
    if control == super::aarch32::SemanticControl::Continue {
        advance(state, decoded)?;
    }
    Ok(InstructionStep::Continue)
}

fn execute_it(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    first_condition: u8,
    mask: u8,
) -> Result<InstructionStep, InterpreterError> {
    if state.cpsr().it_state().is_active() {
        return Ok(InstructionStep::architectural_exception(
            decoded.location,
            nixe_cpu::exception::ExceptionKind::UndefinedInstruction,
            None,
        ));
    }
    let Some(it_state) = ItState::from_encoding(first_condition, mask) else {
        return Ok(InstructionStep::architectural_exception(
            decoded.location,
            nixe_cpu::exception::ExceptionKind::UndefinedInstruction,
            None,
        ));
    };
    state.set_cpsr(state.cpsr().with_it_state(it_state));
    advance(state, decoded)?;
    Ok(InstructionStep::Continue)
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
fn branch_error(
    decoded: &DecodedInstruction<DecodedOpcode>,
    error: nixe_cpu::state::a32::InvalidBranchTarget,
) -> InterpreterError {
    InterpreterError::ContextMismatch {
        source: decoded.location,
        reason: error.to_string().into(),
    }
}
