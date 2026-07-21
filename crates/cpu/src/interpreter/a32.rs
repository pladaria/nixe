//! A32 reference interpretation.

use crate::{
    address::GuestVirtualAddress,
    decode::{DecodedOpcode, OperandId, OperandValue},
    ir::op::Condition,
    location::{DecodedInstruction, InstructionSize, LocationDescriptor},
    semantics::conditions::evaluate_a32,
    state::a32::A32State,
};

use super::{InterpreterError, InterpreterOutcome};

pub(super) fn execute(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    let name = decoded.instruction.pattern().name;
    let condition = decoded_condition(decoded);
    if !evaluate_a32(condition, state.cpsr().bits()) {
        return advance(state, decoded, InstructionSize::Bits32);
    }
    match name {
        "nop" | "unconditional-space-nop-alias" => advance(state, decoded, InstructionSize::Bits32),
        "b" => {
            let displacement = signed_immediate(decoded)?;
            let target = decoded.location.pc.wrapping_offset(8 + displacement);
            state
                .set_instruction_address(target.get() as u32)
                .map_err(|error| InterpreterError::ContextMismatch {
                    source: decoded.location,
                    reason: error.to_string().into(),
                })?;
            Ok(resume(decoded, state))
        }
        "blx-immediate" => {
            let bits = decoded.encoding.bits();
            let immediate = ((bits & 0x00ff_ffff) << 2) | ((bits >> 23) & 2);
            let displacement = i64::from(((immediate << 6) as i32) >> 6);
            let target = decoded.location.pc.wrapping_offset(8 + displacement);
            state
                .branch_link_exchange_immediate(target.get() as u32, InstructionSize::Bits32)
                .map_err(|error| InterpreterError::ContextMismatch {
                    source: decoded.location,
                    reason: error.to_string().into(),
                })?;
            Ok(resume(decoded, state))
        }
        _ => Err(super::unsupported(decoded)),
    }
}

fn decoded_condition(decoded: &DecodedInstruction<DecodedOpcode>) -> Condition {
    match decoded.instruction.operands().get(OperandId::Condition) {
        Some(OperandValue::Unsigned(value)) => Condition::from_encoding(value as u8),
        None if decoded.instruction.pattern().name == "blx-immediate"
            || decoded.instruction.pattern().name == "unconditional-space-nop-alias" =>
        {
            Condition::Al
        }
        _ => Condition::Nv,
    }
}

fn signed_immediate(decoded: &DecodedInstruction<DecodedOpcode>) -> Result<i64, InterpreterError> {
    match decoded.instruction.operands().get(OperandId::Immediate) {
        Some(OperandValue::Signed(value)) => Ok(value),
        _ => Err(InterpreterError::ContextMismatch {
            source: decoded.location,
            reason: "decoded A32 branch has no signed displacement".into(),
        }),
    }
}

fn advance(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    size: InstructionSize,
) -> Result<InterpreterOutcome, InterpreterError> {
    let next = state
        .instruction_address()
        .wrapping_add(u32::from(size.bytes()));
    state
        .set_instruction_address(next)
        .map_err(|error| InterpreterError::ContextMismatch {
            source: decoded.location,
            reason: error.to_string().into(),
        })?;
    Ok(resume(decoded, state))
}

fn resume(decoded: &DecodedInstruction<DecodedOpcode>, state: &A32State) -> InterpreterOutcome {
    InterpreterOutcome::Resume(LocationDescriptor::new(
        GuestVirtualAddress::new(u64::from(state.instruction_address())),
        state.execution_state(),
        decoded.location.profile_id,
    ))
}
