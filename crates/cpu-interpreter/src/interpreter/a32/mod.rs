//! Modular A32 reference interpreter.

mod control;
mod fp_simd;
mod integer;
mod memory;

use super::{InstructionStep, InterpreterContext, InterpreterError};
use nixe_cpu::{
    decode::{
        DecodedOpcode,
        a32::{A32Instruction, normalize},
    },
    location::DecodedInstruction,
    semantics::conditions::evaluate_a32,
    state::a32::A32State,
};
pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InstructionStep, InterpreterError> {
    let normalized = normalize(&decoded.instruction, decoded.encoding);
    if !evaluate_a32(normalized.condition, state.cpsr().bits()) {
        advance(state, decoded)?;
        return Ok(InstructionStep::Continue);
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
                    return Ok(InstructionStep::data_fault(decoded.location, fault));
                }
            }
        }
        A32Instruction::FpSimd(instruction) => {
            match fp_simd::execute(context, state, decoded, instruction)? {
                fp_simd::Execution::Control(control) => control,
                fp_simd::Execution::Fault(fault) => {
                    return Ok(InstructionStep::data_fault(decoded.location, fault));
                }
                fp_simd::Execution::FloatingPointException => {
                    return Ok(InstructionStep::architectural_exception(
                        decoded.location,
                        nixe_cpu::exception::ExceptionKind::FloatingPoint,
                        None,
                    ));
                }
            }
        }
    };
    if control == super::aarch32::SemanticControl::Continue {
        advance(state, decoded)?;
    }
    Ok(InstructionStep::Continue)
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

fn branch_error(
    decoded: &DecodedInstruction<DecodedOpcode>,
    error: nixe_cpu::state::a32::InvalidBranchTarget,
) -> InterpreterError {
    InterpreterError::ContextMismatch {
        source: decoded.location,
        reason: error.to_string().into(),
    }
}
