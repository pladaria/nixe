use super::{InterpreterError, InterpreterOutcome, branch_error, resume};
use crate::{
    decode::{DecodedOpcode, a32::control::Instruction},
    exception::ExceptionKind,
    location::{DecodedInstruction, InstructionSize},
    state::a32::{A32GeneralRegister, A32State},
};

pub(super) fn execute(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    match instruction {
        Instruction::Nop => state
            .set_instruction_address(state.instruction_address().wrapping_add(4))
            .map_err(|e| branch_error(decoded, e))?,
        Instruction::Branch { link, displacement } => {
            if link {
                state.write_r(
                    A32GeneralRegister::new(14).unwrap(),
                    state.instruction_address().wrapping_add(4),
                );
            }
            state
                .set_instruction_address(state.read_pc().wrapping_add_signed(displacement))
                .map_err(|e| branch_error(decoded, e))?;
        }
        Instruction::Exchange { link, rm } => {
            let target = if rm == 15 {
                state.read_pc()
            } else {
                state.read_r(A32GeneralRegister::new(rm).unwrap())
            };
            if link {
                state.branch_link_exchange(target, InstructionSize::Bits32)
            } else {
                state.branch_exchange(target)
            }
            .map_err(|e| branch_error(decoded, e))?;
        }
        Instruction::BlxImmediate { displacement } => state
            .branch_link_exchange_immediate(
                state.read_pc().wrapping_add_signed(displacement),
                InstructionSize::Bits32,
            )
            .map_err(|e| branch_error(decoded, e))?,
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
