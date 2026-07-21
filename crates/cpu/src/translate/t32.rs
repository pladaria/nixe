//! T32-to-IR translation.

use crate::{
    decode::{DecodedOpcode, OperandId, OperandValue},
    ir::{
        builder::{BuildError, IrBuilder},
        op::{
            FlagOperation, IntegerBinaryKind, IntegerPredicate, OperationKind, ScalarOperation,
            ShiftKind, StateRegister,
        },
        terminator::ControlTarget,
        types::IrType,
        value::{Immediate, Operand, Value},
    },
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    state::a32::{Cpsr, ItState},
};

use super::block::{LiftOutcome, conditional_terminator, direct_branch_target};

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    lift_inner(builder, decoded).expect("T32 semantic construction must produce valid IR")
}

fn lift_inner(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    match decoded.instruction.pattern().name {
        "it" => lift_it(builder, decoded),
        "nop" | "nop.w" => {
            advance_it_state(builder, decoded.location)?;
            Ok(LiftOutcome::Continue)
        }
        "b" => lift_branch(builder, decoded),
        _ => Ok(LiftOutcome::Interpret(decoded.instruction.coverage_id())),
    }
}

fn lift_it(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    let immediate = match decoded.instruction.operands().get(OperandId::Immediate) {
        Some(OperandValue::Unsigned(value)) => value as u8,
        _ => unreachable!("T32 IT has a validated immediate operand"),
    };
    let Some(it_state) = ItState::from_encoding(immediate >> 4, immediate & 0xf) else {
        return Ok(LiftOutcome::Interpret(decoded.instruction.coverage_id()));
    };

    let cpsr = read_cpsr(builder, decoded.location)?;
    let packed = pack_it_state(
        builder,
        decoded.location,
        cpsr.into(),
        Immediate::I32(u32::from(it_state.bits())).into(),
    )?;
    write_cpsr(builder, decoded.location, packed.into())?;
    Ok(LiftOutcome::Continue)
}

fn lift_branch(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    let target = direct_branch_target(decoded)
        .expect("validated T32 branch displacement always produces an aligned target");
    let condition = current_it_condition(builder, decoded.location)?;
    let fallthrough = ControlTarget::Direct {
        pc: decoded.location.pc.wrapping_offset(2),
        execution_state: ExecutionState::T32,
    };
    Ok(LiftOutcome::Terminate(conditional_terminator(
        condition,
        target,
        fallthrough,
    )))
}

/// Returns the current IT predicate and advances ITSTATE before any observable
/// effect of the predicated instruction is emitted.
fn current_it_condition(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
) -> Result<Operand, BuildError> {
    let cpsr = read_cpsr(builder, source)?;
    let it_state = unpack_it_state(builder, source, cpsr.into())?;
    let encoded = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: it_state.into(),
            amount: Immediate::I32(4).into(),
        },
    )?;
    let inactive = emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Scalar(ScalarOperation::Compare {
            predicate: IntegerPredicate::Equal,
            lhs: it_state.into(),
            rhs: Immediate::I32(0).into(),
        }),
    )?;
    let condition = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(ScalarOperation::Select {
            condition: inactive.into(),
            when_true: Immediate::I32(14).into(),
            when_false: encoded.into(),
        }),
    )?;
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked { value: cpsr.into() }),
    )?;
    let predicate = emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Flags(FlagOperation::EvaluateEncoded {
            flags: flags.into(),
            condition: condition.into(),
            nv_is_unconditional: false,
        }),
    )?;
    let advanced = advance_it_value(builder, source, it_state.into())?;
    let packed = pack_it_state(builder, source, cpsr.into(), advanced.into())?;
    write_cpsr(builder, source, packed.into())?;
    Ok(predicate.into())
}

fn advance_it_state(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<(), BuildError> {
    let cpsr = read_cpsr(builder, source)?;
    let it_state = unpack_it_state(builder, source, cpsr.into())?;
    let advanced = advance_it_value(builder, source, it_state.into())?;
    let packed = pack_it_state(builder, source, cpsr.into(), advanced.into())?;
    write_cpsr(builder, source, packed.into())
}

fn advance_it_value(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    it_state: Operand,
) -> Result<Value, BuildError> {
    let low_three = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        it_state,
        Immediate::I32(7).into(),
    )?;
    let is_last = emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Scalar(ScalarOperation::Compare {
            predicate: IntegerPredicate::Equal,
            lhs: low_three.into(),
            rhs: Immediate::I32(0).into(),
        }),
    )?;
    let top = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        it_state,
        Immediate::I32(0xe0).into(),
    )?;
    let shifted = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: it_state,
            amount: Immediate::I32(1).into(),
        },
    )?;
    let low = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        shifted.into(),
        Immediate::I32(0x1f).into(),
    )?;
    let next = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        top.into(),
        low.into(),
    )?;
    emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(ScalarOperation::Select {
            condition: is_last.into(),
            when_true: Immediate::I32(0).into(),
            when_false: next.into(),
        }),
    )
}

fn unpack_it_state(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    cpsr: Operand,
) -> Result<Value, BuildError> {
    let low = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: cpsr,
            amount: Immediate::I32(25).into(),
        },
    )?;
    let low = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        low.into(),
        Immediate::I32(3).into(),
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: cpsr,
            amount: Immediate::I32(10).into(),
        },
    )?;
    let high = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        high.into(),
        Immediate::I32(0x3f).into(),
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: high.into(),
            amount: Immediate::I32(2).into(),
        },
    )?;
    scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        high.into(),
        low.into(),
    )
}

fn pack_it_state(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    cpsr: Operand,
    it_state: Operand,
) -> Result<Value, BuildError> {
    let cleared = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        cpsr,
        Immediate::I32(!Cpsr::IT_MASK).into(),
    )?;
    let low = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        it_state,
        Immediate::I32(3).into(),
    )?;
    let low = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: low.into(),
            amount: Immediate::I32(25).into(),
        },
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: it_state,
            amount: Immediate::I32(2).into(),
        },
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: high.into(),
            amount: Immediate::I32(10).into(),
        },
    )?;
    let packed = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        cleared.into(),
        low.into(),
    )?;
    scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        packed.into(),
        high.into(),
    )
}

fn read_cpsr(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<Value, BuildError> {
    emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A32Cpsr),
    )
}

fn write_cpsr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
) -> Result<(), BuildError> {
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A32Cpsr,
            value,
        },
    )?;
    Ok(())
}

fn scalar(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    operation: ScalarOperation,
) -> Result<Value, BuildError> {
    emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(operation),
    )
}

fn scalar_binary(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    kind: IntegerBinaryKind,
    lhs: Operand,
    rhs: Operand,
) -> Result<Value, BuildError> {
    scalar(builder, source, ScalarOperation::Binary { kind, lhs, rhs })
}

fn emit_one(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    ty: IrType,
    kind: OperationKind,
) -> Result<Value, BuildError> {
    Ok(builder
        .emit(source, &[ty], kind)?
        .iter()
        .next()
        .expect("one result was requested"))
}
