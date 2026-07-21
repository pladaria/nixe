//! T32 reference interpretation.

use crate::{
    address::GuestVirtualAddress,
    decode::{DecodedOpcode, OperandId, OperandValue},
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    semantics::conditions::evaluate_a32,
    state::a32::{A32GeneralRegister, A32State, Cpsr, ItState},
};

use super::{InterpreterError, InterpreterOutcome};

pub(super) fn execute(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    let name = decoded.instruction.pattern().name;
    if name == "it" {
        return execute_it(state, decoded);
    }

    let mut cpsr = state.cpsr();
    let it_state = cpsr.it_state();
    let executes = it_state
        .current_condition()
        .is_none_or(|condition| evaluate_a32(condition, cpsr.bits()));
    cpsr = cpsr.with_it_state(it_state.advance());
    state.set_cpsr(cpsr);

    if executes {
        match name {
            "nop" | "nop.w" => {}
            "movs" => execute_movs(state, decoded, !it_state.is_active()),
            "b" => {
                let displacement = match decoded.instruction.operands().get(OperandId::Immediate) {
                    Some(OperandValue::Signed(value)) => value,
                    _ => {
                        return Err(InterpreterError::ContextMismatch {
                            source: decoded.location,
                            reason: "decoded T32 branch has no signed displacement".into(),
                        });
                    }
                };
                let target = decoded.location.pc.wrapping_offset(4 + displacement);
                state
                    .set_instruction_address(target.get() as u32)
                    .map_err(|error| InterpreterError::ContextMismatch {
                        source: decoded.location,
                        reason: error.to_string().into(),
                    })?;
                return Ok(resume(decoded, state));
            }
            _ => return Err(super::unsupported(decoded)),
        }
    }

    let next = state
        .instruction_address()
        .wrapping_add(u32::from(decoded.encoding.size().bytes()));
    state
        .set_instruction_address(next)
        .map_err(|error| InterpreterError::ContextMismatch {
            source: decoded.location,
            reason: error.to_string().into(),
        })?;
    Ok(resume(decoded, state))
}

fn execute_it(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    let immediate = match decoded.instruction.operands().get(OperandId::Immediate) {
        Some(OperandValue::Unsigned(value)) => value as u8,
        _ => return Err(super::unsupported(decoded)),
    };
    let Some(it_state) = ItState::from_encoding(immediate >> 4, immediate & 0xf) else {
        return Ok(InterpreterOutcome::Exception {
            source: decoded.location,
            kind: crate::ir::terminator::ExceptionKind::UndefinedInstruction,
            syndrome: None,
        });
    };
    state.set_cpsr(state.cpsr().with_it_state(it_state));
    let next = state.instruction_address().wrapping_add(2);
    state
        .set_instruction_address(next)
        .map_err(|error| InterpreterError::ContextMismatch {
            source: decoded.location,
            reason: error.to_string().into(),
        })?;
    Ok(resume(decoded, state))
}

fn execute_movs(
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    set_flags: bool,
) {
    let register = match decoded.instruction.operands().get(OperandId::Destination) {
        Some(OperandValue::Register { index, .. }) => A32GeneralRegister::new(index).unwrap(),
        _ => unreachable!("MOVS decoder validates the destination register"),
    };
    let immediate = match decoded.instruction.operands().get(OperandId::Immediate) {
        Some(OperandValue::Unsigned(value)) => value as u32,
        _ => unreachable!("MOVS decoder validates the immediate"),
    };
    state.write_r(register, immediate);
    if !set_flags {
        return;
    }
    let old = state.cpsr().bits();
    let flags = (if immediate & (1 << 31) != 0 {
        Cpsr::N
    } else {
        0
    }) | (if immediate == 0 { Cpsr::Z } else { 0 });
    state.set_cpsr(Cpsr::from_bits((old & !(Cpsr::N | Cpsr::Z)) | flags));
}

fn resume(decoded: &DecodedInstruction<DecodedOpcode>, state: &A32State) -> InterpreterOutcome {
    InterpreterOutcome::Resume(LocationDescriptor::new(
        GuestVirtualAddress::new(u64::from(state.instruction_address())),
        ExecutionState::T32,
        decoded.location.profile_id,
    ))
}
