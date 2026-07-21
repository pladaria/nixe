use super::*;

use crate::{
    decode::{
        DecodedOpcode,
        a64::control::{Instruction as ControlInstruction, Operands as ControlOperands},
    },
    ir::builder::{BuildError, IrBuilder},
    location::DecodedInstruction,
};

use super::{LiftOutcome, direct_target, emit_call, next_pc, sign_extend};

pub(super) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: ControlInstruction,
) -> Result<LiftOutcome, BuildError> {
    let fields = instruction.operands();
    let source = decoded.location;
    Ok(match instruction {
        ControlInstruction::Nop(_) => LiftOutcome::Continue,
        ControlInstruction::BranchImmediate(_) => {
            LiftOutcome::Terminate(super::super::block::direct_branch(direct_target(
                source,
                sign_extend(u64::from(fields.immediate_26), 26) << 2,
            )))
        }
        ControlInstruction::BranchLinkImmediate(_) => {
            let target =
                direct_target(source, sign_extend(u64::from(fields.immediate_26), 26) << 2);
            LiftOutcome::Terminate(emit_call(builder, source, target, next_pc(source))?)
        }
        ControlInstruction::BranchRegister(_) => lift_branch_register(builder, decoded, fields)?,
        ControlInstruction::ConditionalBranch(_) => {
            lift_conditional_branch(builder, decoded, fields)?
        }
        ControlInstruction::CompareBranch(_) => lift_compare_branch(builder, decoded, fields)?,
        ControlInstruction::TestBranch(_) => lift_test_branch(builder, decoded, fields)?,
        ControlInstruction::SupervisorCall(_) => LiftOutcome::Terminate(exception(
            source,
            crate::ir::terminator::ExceptionKind::SupervisorCall,
            Some(u64::from(fields.immediate_16)),
        )),
        ControlInstruction::Breakpoint(_) => LiftOutcome::Terminate(exception(
            source,
            crate::ir::terminator::ExceptionKind::Breakpoint,
            Some(u64::from(fields.immediate_16)),
        )),
    })
}

fn lift_branch_register(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: ControlOperands,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    let masked = fields.branch_register_key;
    let rn = fields.rn;
    if !matches!(masked, 0xd61f_0000 | 0xd63f_0000 | 0xd65f_0000) {
        return Ok(LiftOutcome::Interpret(crate::coverage::CoverageId::new(5)));
    }
    let address_bits = read_gpr(builder, source, rn, IrType::I64, Register31::Zero)?;
    let address = bitcast(builder, source, address_bits, IrType::Address)?;
    let target = indirect_target(address, ExecutionState::A64);
    Ok(match masked {
        0xd61f_0000 => LiftOutcome::Terminate(Terminator::Indirect { target }),
        0xd63f_0000 => LiftOutcome::Terminate(emit_call(builder, source, target, next_pc(source))?),
        0xd65f_0000 => LiftOutcome::Terminate(Terminator::Return { target }),
        _ => unreachable!(),
    })
}

fn lift_conditional_branch(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: ControlOperands,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    let cond = evaluate_condition(builder, source, condition(u32::from(fields.condition)))?;
    let displacement = sign_extend(u64::from(fields.immediate_19), 19) << 2;
    Ok(LiftOutcome::Terminate(conditional_terminator(
        cond,
        direct_target(source, displacement),
        direct_target(source, 4),
    )))
}

fn lift_compare_branch(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: ControlOperands,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    let width = if fields.width_64 {
        IrType::I64
    } else {
        IrType::I32
    };
    let value = read_gpr(builder, source, fields.rd, width, Register31::Zero)?;
    let zero = if width == IrType::I64 {
        Immediate::I64(0)
    } else {
        Immediate::I32(0)
    };
    let predicate = if !fields.nonzero {
        IntegerPredicate::Equal
    } else {
        IntegerPredicate::NotEqual
    };
    let condition = scalar(
        builder,
        source,
        IrType::I1,
        ScalarOperation::Compare {
            predicate,
            lhs: value,
            rhs: zero.into(),
        },
    )?;
    let displacement = sign_extend(u64::from(fields.immediate_19), 19) << 2;
    Ok(LiftOutcome::Terminate(conditional_terminator(
        condition,
        direct_target(source, displacement),
        direct_target(source, 4),
    )))
}

fn lift_test_branch(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: ControlOperands,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    let bit_index = u32::from(fields.bit_index);
    let value = read_gpr(builder, source, fields.rd, IrType::I64, Register31::Zero)?;
    let tested = binary(
        builder,
        source,
        IntegerBinaryKind::And,
        value,
        Immediate::I64(1_u64 << bit_index).into(),
    )?;
    let predicate = if !fields.nonzero {
        IntegerPredicate::Equal
    } else {
        IntegerPredicate::NotEqual
    };
    let condition = scalar(
        builder,
        source,
        IrType::I1,
        ScalarOperation::Compare {
            predicate,
            lhs: tested,
            rhs: Immediate::I64(0).into(),
        },
    )?;
    let displacement = sign_extend(u64::from(u32::from(fields.immediate_14)), 14) << 2;
    Ok(LiftOutcome::Terminate(conditional_terminator(
        condition,
        direct_target(source, displacement),
        direct_target(source, 4),
    )))
}

fn exception(source: LocationDescriptor, kind: ExceptionKind, syndrome: Option<u64>) -> Terminator {
    Terminator::Exception {
        source,
        kind,
        syndrome,
    }
}
