use super::{InterpreterError, branch_error};
use crate::interpreter::aarch32::{SemanticControl, execute_data_processing, execute_multiply};
use crate::{
    decode::{DecodedOpcode, t32::integer::Instruction},
    location::DecodedInstruction,
    state::a32::{A32GeneralRegister, A32State},
};

pub(super) fn execute(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
    in_it_block: bool,
) -> Result<SemanticControl, InterpreterError> {
    match instruction {
        Instruction::DataProcessing(mut instruction) => {
            if in_it_block && !instruction.operation.is_test() {
                instruction.set_flags = false;
            }
            execute_data_processing(state, instruction).map_err(|e| branch_error(decoded, e))
        }
        Instruction::Multiply(mut instruction) => {
            if in_it_block {
                instruction.set_flags = false;
            }
            execute_multiply(state, instruction).map_err(|e| branch_error(decoded, e))
        }
        Instruction::MoveWide { rd, immediate, top } => {
            let register =
                A32GeneralRegister::new(rd).ok_or_else(|| InterpreterError::ContextMismatch {
                    source: decoded.location,
                    reason: "MOVW/MOVT destination cannot be PC".into(),
                })?;
            let value = if top {
                (state.read_r(register) & 0xffff) | (u32::from(immediate) << 16)
            } else {
                u32::from(immediate)
            };
            state.write_r(register, value);
            Ok(SemanticControl::Continue)
        }
    }
}
