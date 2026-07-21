use super::{InterpreterContext, InterpreterError};
use crate::interpreter::aarch32::{SemanticControl, execute_multiple, execute_single};
use crate::{
    decode::{DecodedOpcode, t32::memory::Instruction},
    location::DecodedInstruction,
    state::a32::A32State,
};

pub(super) enum Execution {
    Control(SemanticControl),
    Fault(crate::memory::DataAccessFault),
}
pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<Execution, InterpreterError> {
    let Some(memory) = context.memory() else {
        return Err(super::super::unsupported(decoded));
    };
    let address_space = context.process().address_space_id();
    let result = match instruction {
        Instruction::Single(instruction) => {
            execute_single(memory, address_space, state, instruction)
        }
        Instruction::Multiple(instruction) => {
            execute_multiple(memory, address_space, state, instruction)
        }
    };
    Ok(match result {
        Ok(control) => Execution::Control(control),
        Err(fault) => Execution::Fault(fault),
    })
}
