//! A32-to-IR translation.

use crate::{
    decode::{
        DecodedOpcode,
        a32::{A32Instruction, control::Instruction as ControlInstruction, normalize},
    },
    ir::{
        builder::IrBuilder,
        op::{Condition, FlagOperation, OperationKind, StateRegister},
        terminator::ControlTarget,
        types::IrType,
        value::Operand,
    },
    location::{DecodedInstruction, ExecutionState},
};

use super::block::{LiftOutcome, conditional_terminator};

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    let normalized = normalize(&decoded.instruction, decoded.encoding);
    match normalized.instruction {
        A32Instruction::Control(ControlInstruction::Nop) => LiftOutcome::Continue,
        A32Instruction::Control(ControlInstruction::Branch {
            link: false,
            displacement,
        }) => lift_branch(builder, decoded, normalized.condition, displacement),
        _ => LiftOutcome::Interpret(decoded.instruction.coverage_id()),
    }
}

fn lift_branch(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    condition: Condition,
    displacement: i32,
) -> LiftOutcome {
    let target = ControlTarget::Direct {
        pc: crate::address::GuestVirtualAddress::new(u64::from(
            (decoded.location.pc.get() as u32)
                .wrapping_add(8)
                .wrapping_add_signed(displacement),
        )),
        execution_state: ExecutionState::A32,
    };
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
