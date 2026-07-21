use super::{InterpreterError, InterpreterOutcome, branch_error, resume};
use crate::{
    decode::{DecodedOpcode, t32::control::Instruction},
    ir::{op::Condition, terminator::ExceptionKind},
    location::{DecodedInstruction, InstructionSize},
    semantics::conditions::evaluate_a32,
    state::a32::{A32GeneralRegister, A32State},
};

pub(super) fn execute(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    match instruction {
        Instruction::Nop | Instruction::Hint { operation: 0 } => advance(state, decoded)?,
        Instruction::Hint { .. } => return Err(super::super::unsupported(decoded)),
        Instruction::It { .. } => unreachable!("IT is handled before ordinary predication"),
        Instruction::Branch {
            condition,
            displacement,
        } => {
            if condition.is_some_and(|encoded| {
                encoded >= 14
                    || !evaluate_a32(Condition::from_encoding(encoded), state.cpsr().bits())
            }) {
                advance(state, decoded)?;
            } else {
                state
                    .set_instruction_address(state.read_pc().wrapping_add_signed(displacement))
                    .map_err(|e| branch_error(decoded, e))?;
            }
        }
        Instruction::Exchange { link, rm } => {
            let target = if rm == 15 {
                state.read_pc()
            } else {
                state.read_r(A32GeneralRegister::new(rm).unwrap())
            };
            if link {
                state.branch_link_exchange(target, InstructionSize::Bits16)
            } else {
                state.branch_exchange(target)
            }
            .map_err(|e| branch_error(decoded, e))?;
        }
        Instruction::BranchLink { displacement } => {
            state.write_r(
                A32GeneralRegister::new(14).unwrap(),
                state.instruction_address().wrapping_add(4) | 1,
            );
            state
                .set_instruction_address(state.read_pc().wrapping_add_signed(displacement))
                .map_err(|e| branch_error(decoded, e))?;
        }
        Instruction::Svc { immediate } => {
            return Ok(InterpreterOutcome::Exception {
                source: decoded.location,
                kind: ExceptionKind::SupervisorCall,
                syndrome: Some(u64::from(immediate)),
            });
        }
        Instruction::Breakpoint { immediate } => {
            return Ok(InterpreterOutcome::Exception {
                source: decoded.location,
                kind: ExceptionKind::Breakpoint,
                syndrome: Some(u64::from(immediate)),
            });
        }
    }
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
        .map_err(|e| branch_error(decoded, e))
}
