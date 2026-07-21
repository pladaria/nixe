use super::{InterpreterError, branch_error};
use crate::interpreter::aarch32::{SemanticControl, execute_data_processing, execute_multiply};
use crate::{
    decode::{DecodedOpcode, a32::integer::Instruction},
    location::DecodedInstruction,
    state::a32::{A32GeneralRegister, A32State},
};

pub(super) fn execute(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<SemanticControl, InterpreterError> {
    match instruction {
        Instruction::DataProcessing(instruction) => {
            execute_data_processing(state, instruction).map_err(|e| branch_error(decoded, e))
        }
        Instruction::Multiply(instruction) => {
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
