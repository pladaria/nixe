//! A32-to-IR translation.

use crate::{
    decode::{DecodedOpcode, OperandId, OperandValue},
    ir::{
        builder::IrBuilder,
        op::{Condition, FlagOperation, OperationKind, StateRegister},
        terminator::ControlTarget,
        types::IrType,
        value::Operand,
    },
    location::{DecodedInstruction, ExecutionState},
};

use super::block::{LiftOutcome, conditional_terminator, direct_branch_target};

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    match decoded.instruction.pattern().name {
        "nop" => LiftOutcome::Continue,
        "b" => lift_branch(builder, decoded),
        _ => LiftOutcome::Interpret(decoded.instruction.coverage_id()),
    }
}

fn lift_branch(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    let target = direct_branch_target(decoded)
        .expect("validated A32 branch displacement always produces an aligned target");
    let condition = decoded_condition(decoded);
    if condition == Condition::Al {
        return LiftOutcome::Terminate(super::block::direct_branch(target));
    }
    if condition == Condition::Nv {
        return LiftOutcome::Interpret(decoded.instruction.coverage_id());
    }
    let condition = evaluate_condition(builder, decoded, condition);
    let fallthrough = ControlTarget::Direct {
        pc: decoded.location.pc.wrapping_offset(4),
        execution_state: ExecutionState::A32,
    };
    LiftOutcome::Terminate(conditional_terminator(condition, target, fallthrough))
}

fn decoded_condition(decoded: &DecodedInstruction<DecodedOpcode>) -> Condition {
    match decoded.instruction.operands().get(OperandId::Condition) {
        Some(OperandValue::Unsigned(value)) => Condition::from_encoding(value as u8),
        _ => unreachable!("A32 conditional pattern has a validated condition operand"),
    }
}

fn evaluate_condition(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    condition: Condition,
) -> Operand {
    let packed = builder
        .emit(
            decoded.location,
            &[IrType::I32],
            OperationKind::ReadState(StateRegister::A32Cpsr),
        )
        .expect("A32 flag reads form valid IR")
        .iter()
        .next()
        .unwrap();
    let flags = builder
        .emit(
            decoded.location,
            &[IrType::Flags],
            OperationKind::Flags(FlagOperation::FromPacked {
                value: packed.into(),
            }),
        )
        .expect("packed A32 flags form valid IR")
        .iter()
        .next()
        .unwrap();
    builder
        .emit(
            decoded.location,
            &[IrType::I1],
            OperationKind::Flags(FlagOperation::Evaluate {
                flags: flags.into(),
                condition,
            }),
        )
        .expect("A32 condition evaluation forms valid IR")
        .iter()
        .next()
        .unwrap()
        .into()
}
