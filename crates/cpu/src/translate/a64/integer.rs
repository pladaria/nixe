use super::*;

use crate::{
    decode::{
        DecodedOpcode,
        a64::integer::{Instruction as IntegerInstruction, Operands as IntegerOperands},
    },
    ir::builder::{BuildError, IrBuilder},
    location::DecodedInstruction,
    semantics::{a64::shift_kind, shifts::ShiftKind as SemanticShiftKind},
};

use super::LiftOutcome;

pub(super) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: IntegerInstruction,
) -> Result<LiftOutcome, BuildError> {
    let fields = instruction.operands();
    match instruction {
        IntegerInstruction::MoveWide(_) => lift_move_wide(builder, decoded, fields),
        IntegerInstruction::AddSubImmediate(_) => lift_add_sub_immediate(builder, decoded, fields),
        IntegerInstruction::AddSubShifted(_) => lift_add_sub_shifted(builder, decoded, fields),
        IntegerInstruction::AddSubExtended(_) => lift_add_sub_extended(builder, decoded, fields),
        IntegerInstruction::AddSubCarry(_) => lift_add_sub_carry(builder, decoded, fields),
        IntegerInstruction::LogicalImmediate(_) => lift_logical_immediate(builder, decoded, fields),
        IntegerInstruction::LogicalShifted(_) => lift_logical_shifted(builder, decoded, fields),
        IntegerInstruction::Bitfield(_) => lift_bitfield(builder, decoded, fields),
        IntegerInstruction::Extract(_) => lift_extract(builder, decoded, fields),
        IntegerInstruction::TwoSource(_) => lift_two_source(builder, decoded, fields),
        IntegerInstruction::ConditionalCompareRegister(_)
        | IntegerInstruction::ConditionalCompareImmediate(_) => {
            lift_conditional_compare(builder, decoded, fields)
        }
        IntegerInstruction::ConditionalSelect(_) => {
            lift_conditional_select(builder, decoded, fields)
        }
        IntegerInstruction::ThreeSource(_) => lift_three_source(builder, decoded, fields),
        IntegerInstruction::OneSource(_) => lift_one_source(builder, decoded, fields),
        IntegerInstruction::Adr(_) => lift_adr(builder, decoded, fields, false),
        IntegerInstruction::Adrp(_) => lift_adr(builder, decoded, fields, true),
    }
}

pub(super) fn integer_width(fields: IntegerOperands) -> IrType {
    if fields.width_64 {
        IrType::I64
    } else {
        IrType::I32
    }
}

fn immediate_for(width: IrType, value: u64) -> Immediate {
    if width == IrType::I64 {
        Immediate::I64(value)
    } else {
        Immediate::I32(value as u32)
    }
}

fn lift_move_wide(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let hw = u32::from(fields.opcode_2);
    if width == IrType::I32 && hw >= 2 {
        return Ok(unsupported(decoded));
    }
    let shift = hw * 16;
    let imm = u64::from(u32::from(fields.immediate_16)) << shift;
    let opc = u32::from((fields.subtract as u8) * 2 + fields.set_flags as u8);
    let value: Operand = match opc {
        0 => immediate_for(width, !imm).into(), // MOVN, truncated by the immediate type
        2 => immediate_for(width, imm).into(),  // MOVZ
        3 => {
            let old = read_gpr(
                builder,
                decoded.location,
                fields.rd,
                width,
                Register31::Zero,
            )?;
            scalar(
                builder,
                decoded.location,
                width,
                ScalarOperation::InsertBits {
                    destination: old,
                    source: immediate_for(width, u64::from(fields.immediate_16)).into(),
                    source_lsb: 0,
                    destination_lsb: shift as u8,
                    width: 16,
                },
            )?
        }
        _ => return Ok(unsupported(decoded)),
    };
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

struct AddSubSpec {
    subtract: bool,
    set_flags: bool,
    destination: u8,
    destination_register31: Register31,
}

fn emit_add_sub(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    lhs: Operand,
    rhs: Operand,
    spec: AddSubSpec,
) -> Result<(), BuildError> {
    if spec.set_flags
        && spec.destination == 31
        && matches!(spec.destination_register31, Register31::Zero)
    {
        let operation = if spec.subtract {
            FlagOperation::Subtract {
                lhs,
                rhs,
                result: None,
            }
        } else {
            FlagOperation::Add {
                lhs,
                rhs,
                result: None,
            }
        };
        let flags = emit_one(
            builder,
            source,
            IrType::Flags,
            OperationKind::Flags(operation),
        )?;
        return write_flags(builder, source, flags.into());
    }
    let result = binary(
        builder,
        source,
        if spec.subtract {
            IntegerBinaryKind::Subtract
        } else {
            IntegerBinaryKind::Add
        },
        lhs,
        rhs,
    )?;
    write_gpr(
        builder,
        source,
        spec.destination,
        result,
        spec.destination_register31,
    )?;
    if spec.set_flags {
        let operation = if spec.subtract {
            FlagOperation::Subtract {
                lhs,
                rhs,
                result: Some(result),
            }
        } else {
            FlagOperation::Add {
                lhs,
                rhs,
                result: Some(result),
            }
        };
        let flags = emit_one(
            builder,
            source,
            IrType::Flags,
            OperationKind::Flags(operation),
        )?;
        write_flags(builder, source, flags.into())?;
    }
    Ok(())
}

fn lift_add_sub_immediate(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::StackPointer,
    )?;
    let shift = if fields.n { 12 } else { 0 };
    let rhs = immediate_for(width, u64::from(u32::from(fields.immediate_12)) << shift).into();
    let set_flags = fields.set_flags;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        rhs,
        AddSubSpec {
            subtract: fields.subtract,
            set_flags,
            destination: fields.rd,
            destination_register31: if set_flags {
                Register31::Zero
            } else {
                Register31::StackPointer
            },
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn shifted_register(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    fields: IntegerOperands,
    width: IrType,
    index: u8,
) -> Result<Option<Operand>, BuildError> {
    let amount = u32::from(fields.shift_amount);
    if width == IrType::I32 && amount >= 32 {
        return Ok(None);
    }
    let Some(kind) = shift_kind(fields.shift_kind, true) else {
        return Ok(None);
    };
    let kind = match kind {
        SemanticShiftKind::LogicalLeft => ShiftKind::LogicalLeft,
        SemanticShiftKind::LogicalRight => ShiftKind::LogicalRight,
        SemanticShiftKind::ArithmeticRight => ShiftKind::ArithmeticRight,
        SemanticShiftKind::RotateRight => ShiftKind::RotateRight,
    };
    let value = read_gpr(builder, source, index, width, Register31::Zero)?;
    if amount == 0 {
        Ok(Some(value))
    } else {
        Ok(Some(scalar(
            builder,
            source,
            width,
            ScalarOperation::ShiftImmediate {
                kind,
                value,
                amount: amount as u8,
            },
        )?))
    }
}

fn lift_add_sub_shifted(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    if u32::from(fields.shift_kind) == 3 {
        return Ok(unsupported(decoded));
    }
    let width = integer_width(fields);
    let Some(rhs) = shifted_register(builder, decoded.location, fields, width, fields.rm)? else {
        return Ok(unsupported(decoded));
    };
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        rhs,
        AddSubSpec {
            subtract: fields.subtract,
            set_flags: fields.set_flags,
            destination: fields.rd,
            destination_register31: Register31::Zero,
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_add_sub_extended(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let shift = u32::from(fields.small_shift);
    if shift > 4 {
        return Ok(unsupported(decoded));
    }
    let rm = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let extension = u32::from(fields.extension);
    let source_width = match extension & 3 {
        0 => IrType::I8,
        1 => IrType::I16,
        2 => IrType::I32,
        3 => IrType::I64,
        _ => unreachable!(),
    };
    let narrowed = if source_width.bit_width() < width.bit_width() {
        scalar(
            builder,
            decoded.location,
            source_width,
            ScalarOperation::Truncate {
                value: rm,
                to: source_width,
            },
        )?
    } else {
        rm
    };
    let extended = if source_width.bit_width() < width.bit_width() {
        scalar(
            builder,
            decoded.location,
            width,
            if extension & 4 == 0 {
                ScalarOperation::ZeroExtend {
                    value: narrowed,
                    to: width,
                }
            } else {
                ScalarOperation::SignExtend {
                    value: narrowed,
                    to: width,
                }
            },
        )?
    } else {
        narrowed
    };
    let result = if shift == 0 {
        extended
    } else {
        scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ShiftImmediate {
                kind: ShiftKind::LogicalLeft,
                value: extended,
                amount: shift as u8,
            },
        )?
    };
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::StackPointer,
    )?;
    let set_flags = fields.set_flags;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        result,
        AddSubSpec {
            subtract: fields.subtract,
            set_flags,
            destination: fields.rd,
            destination_register31: if set_flags {
                Register31::Zero
            } else {
                Register31::StackPointer
            },
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn carry_in(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<Operand, BuildError> {
    let flags = read_flags(builder, source)?;
    Ok(emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Flags(FlagOperation::EvaluateBit {
            flags,
            bit: FlagBit::Carry,
        }),
    )?
    .into())
}

fn lift_add_sub_carry(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let rhs = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let carry = carry_in(builder, decoded.location)?;
    if fields.set_flags && fields.rd == 31 {
        let flag_operation = if fields.subtract {
            FlagOperation::SubtractCarry {
                lhs,
                rhs,
                carry_in: carry,
                result: None,
            }
        } else {
            FlagOperation::AddCarry {
                lhs,
                rhs,
                carry_in: carry,
                result: None,
            }
        };
        let flags = emit_one(
            builder,
            decoded.location,
            IrType::Flags,
            OperationKind::Flags(flag_operation),
        )?;
        write_flags(builder, decoded.location, flags.into())?;
        return Ok(LiftOutcome::Continue);
    }
    let scalar_operation = if fields.subtract {
        ScalarOperation::SubtractCarry {
            lhs,
            rhs,
            carry_in: carry,
        }
    } else {
        ScalarOperation::AddCarry {
            lhs,
            rhs,
            carry_in: carry,
        }
    };
    let result = scalar(builder, decoded.location, width, scalar_operation)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    if fields.set_flags {
        let flag_operation = if fields.subtract {
            FlagOperation::SubtractCarry {
                lhs,
                rhs,
                carry_in: carry,
                result: Some(result),
            }
        } else {
            FlagOperation::AddCarry {
                lhs,
                rhs,
                carry_in: carry,
                result: Some(result),
            }
        };
        let flags = emit_one(
            builder,
            decoded.location,
            IrType::Flags,
            OperationKind::Flags(flag_operation),
        )?;
        write_flags(builder, decoded.location, flags.into())?;
    }
    Ok(LiftOutcome::Continue)
}

fn logical_result(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    opc: u32,
    lhs: Operand,
    rhs: Operand,
) -> Result<Operand, BuildError> {
    binary(
        builder,
        source,
        match opc {
            0 | 3 => IntegerBinaryKind::And,
            1 => IntegerBinaryKind::Or,
            2 => IntegerBinaryKind::Xor,
            _ => unreachable!(),
        },
        lhs,
        rhs,
    )
}

fn lift_logical_immediate(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let size = if width == IrType::I64 { 64 } else { 32 };
    let Ok(immediate) = decode_a64_logical_immediate(
        fields.n,
        (u32::from(fields.immediate_6_high)) as u8,
        (u32::from(fields.shift_amount)) as u8,
        size,
    ) else {
        return Ok(unsupported(decoded));
    };
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let opc = u32::from((fields.subtract as u8) * 2 + fields.set_flags as u8);
    let rhs = immediate_for(width, immediate).into();
    if opc == 3 && fields.rd == 31 {
        logical_flags(builder, decoded.location, lhs, rhs, None)?;
        return Ok(LiftOutcome::Continue);
    }
    let result = logical_result(builder, decoded.location, opc, lhs, rhs)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    if opc == 3 {
        logical_flags(builder, decoded.location, lhs, rhs, Some(result))?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_logical_shifted(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let Some(mut rhs) = shifted_register(builder, decoded.location, fields, width, fields.rm)?
    else {
        return Ok(unsupported(decoded));
    };
    if fields.invert {
        rhs = scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::Not { value: rhs },
        )?;
    }
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let opc = u32::from((fields.subtract as u8) * 2 + fields.set_flags as u8);
    if opc == 3 && fields.rd == 31 {
        logical_flags(builder, decoded.location, lhs, rhs, None)?;
        return Ok(LiftOutcome::Continue);
    }
    let result = logical_result(builder, decoded.location, opc, lhs, rhs)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    if opc == 3 {
        logical_flags(builder, decoded.location, lhs, rhs, Some(result))?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_bitfield(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let n = fields.n;
    let bits = width.bit_width().expect("A64 integer width is fixed") as u8;
    let imm_r = fields.immediate_6_high;
    let imm_s = u32::from(fields.shift_amount) as u8;
    if n != (width == IrType::I64) || imm_r >= bits || imm_s >= bits {
        return Ok(unsupported(decoded));
    }
    let opc = u32::from((fields.subtract as u8) * 2 + fields.set_flags as u8);
    if opc == 3 {
        return Ok(unsupported(decoded));
    }
    let source_value = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let value = match (opc, imm_r <= imm_s) {
        (0, true) => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ExtractBits {
                value: source_value,
                lsb: imm_r,
                width: imm_s - imm_r + 1,
                signed: true,
            },
        )?,
        (0, false) => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::SignedInsertBits {
                source: source_value,
                destination_lsb: bits - imm_r,
                width: imm_s + 1,
            },
        )?,
        (1, true) if imm_r == 0 && imm_s + 1 == bits => source_value,
        (1, non_wrapping) => {
            let destination = read_gpr(
                builder,
                decoded.location,
                fields.rd,
                width,
                Register31::Zero,
            )?;
            let (source_lsb, destination_lsb, field_width) = if non_wrapping {
                (imm_r, 0, imm_s - imm_r + 1)
            } else {
                (0, bits - imm_r, imm_s + 1)
            };
            scalar(
                builder,
                decoded.location,
                width,
                ScalarOperation::InsertBits {
                    destination,
                    source: source_value,
                    source_lsb,
                    destination_lsb,
                    width: field_width,
                },
            )?
        }
        (2, true) => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ExtractBits {
                value: source_value,
                lsb: imm_r,
                width: imm_s - imm_r + 1,
                signed: false,
            },
        )?,
        (2, false) => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::InsertBits {
                destination: immediate_for(width, 0).into(),
                source: source_value,
                source_lsb: 0,
                destination_lsb: bits - imm_r,
                width: imm_s + 1,
            },
        )?,
        _ => unreachable!(),
    };
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_extract(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lsb = u32::from(fields.shift_amount);
    if (fields.n) != (width == IrType::I64) || (width == IrType::I32 && lsb >= 32) {
        return Ok(unsupported(decoded));
    }
    let second = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let value = if lsb == 0 {
        second
    } else if fields.rn == fields.rm {
        scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ShiftImmediate {
                kind: ShiftKind::RotateRight,
                value: second,
                amount: lsb as u8,
            },
        )?
    } else {
        let first = read_gpr(
            builder,
            decoded.location,
            fields.rn,
            width,
            Register31::Zero,
        )?;
        scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ExtractConcat {
                high: first,
                low: second,
                lsb: lsb as u8,
            },
        )?
    };
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_two_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let opcode = u32::from(fields.shift_amount);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let rhs = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let operation = match opcode {
        2 => ScalarOperation::Divide {
            signedness: IntegerSignedness::Unsigned,
            lhs,
            rhs,
        },
        3 => ScalarOperation::Divide {
            signedness: IntegerSignedness::Signed,
            lhs,
            rhs,
        },
        8 => ScalarOperation::ShiftMasked {
            kind: ShiftKind::LogicalLeft,
            value: lhs,
            amount: rhs,
        },
        9 => ScalarOperation::ShiftMasked {
            kind: ShiftKind::LogicalRight,
            value: lhs,
            amount: rhs,
        },
        10 => ScalarOperation::ShiftMasked {
            kind: ShiftKind::ArithmeticRight,
            value: lhs,
            amount: rhs,
        },
        11 => ScalarOperation::ShiftMasked {
            kind: ShiftKind::RotateRight,
            value: lhs,
            amount: rhs,
        },
        _ => return Ok(unsupported(decoded)),
    };
    let value = scalar(builder, decoded.location, width, operation)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn proposed_compare_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    lhs: Operand,
    rhs: Operand,
    subtract: bool,
) -> Result<Operand, BuildError> {
    let operation = if subtract {
        FlagOperation::Subtract {
            lhs,
            rhs,
            result: None,
        }
    } else {
        FlagOperation::Add {
            lhs,
            rhs,
            result: None,
        }
    };
    Ok(emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(operation),
    )?
    .into())
}

fn lift_conditional_compare(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let rhs = if fields.immediate_form {
        immediate_for(width, u64::from(u32::from(fields.rm))).into()
    } else {
        read_gpr(
            builder,
            decoded.location,
            fields.rm,
            width,
            Register31::Zero,
        )?
    };
    let proposed = proposed_compare_flags(builder, decoded.location, lhs, rhs, fields.subtract)?;
    let fallback = emit_one(
        builder,
        decoded.location,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked {
            value: Immediate::I32(u32::from(fields.nzcv) << 28).into(),
        }),
    )?;
    let cond = evaluate_condition(
        builder,
        decoded.location,
        Condition::from_encoding(fields.condition),
    )?;
    let selected = emit_one(
        builder,
        decoded.location,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::Select {
            condition: cond,
            when_true: proposed,
            when_false: fallback.into(),
        }),
    )?;
    write_flags(builder, decoded.location, selected.into())?;
    Ok(LiftOutcome::Continue)
}

fn lift_conditional_select(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let true_value = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let false_value = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let cond = evaluate_condition(
        builder,
        decoded.location,
        Condition::from_encoding(fields.condition),
    )?;
    let operation = match (fields.subtract, fields.bit10) {
        (false, false) => ScalarOperation::Select {
            condition: cond,
            when_true: true_value,
            when_false: false_value,
        },
        (false, true) => ScalarOperation::SelectTransformed {
            condition: cond,
            when_true: true_value,
            when_false: false_value,
            transform: SelectTransform::Increment,
        },
        (true, false) => ScalarOperation::SelectTransformed {
            condition: cond,
            when_true: true_value,
            when_false: false_value,
            transform: SelectTransform::Invert,
        },
        (true, true) => ScalarOperation::SelectTransformed {
            condition: cond,
            when_true: true_value,
            when_false: false_value,
            transform: SelectTransform::Negate,
        },
    };
    let result = scalar(builder, decoded.location, width, operation)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_three_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let opcode = u32::from(fields.opcode_3);
    let (result_width, operation) = match opcode {
        0 => {
            let lhs = read_gpr(
                builder,
                decoded.location,
                fields.rn,
                width,
                Register31::Zero,
            )?;
            let rhs = read_gpr(
                builder,
                decoded.location,
                fields.rm,
                width,
                Register31::Zero,
            )?;
            let addend = read_gpr(
                builder,
                decoded.location,
                fields.ra,
                width,
                Register31::Zero,
            )?;
            (
                width,
                ScalarOperation::MultiplyAdd {
                    lhs,
                    rhs,
                    addend,
                    subtract_product: fields.subtract_product,
                },
            )
        }
        1 | 5 => {
            let lhs = read_gpr(
                builder,
                decoded.location,
                fields.rn,
                IrType::I32,
                Register31::Zero,
            )?;
            let rhs = read_gpr(
                builder,
                decoded.location,
                fields.rm,
                IrType::I32,
                Register31::Zero,
            )?;
            let addend = read_gpr(
                builder,
                decoded.location,
                fields.ra,
                IrType::I64,
                Register31::Zero,
            )?;
            (
                IrType::I64,
                ScalarOperation::WideningMultiplyAdd {
                    signedness: if opcode == 1 {
                        IntegerSignedness::Signed
                    } else {
                        IntegerSignedness::Unsigned
                    },
                    lhs,
                    rhs,
                    addend,
                    subtract_product: fields.subtract_product,
                },
            )
        }
        2 | 6 if fields.ra == 31 && !fields.subtract_product => {
            let lhs = read_gpr(
                builder,
                decoded.location,
                fields.rn,
                IrType::I64,
                Register31::Zero,
            )?;
            let rhs = read_gpr(
                builder,
                decoded.location,
                fields.rm,
                IrType::I64,
                Register31::Zero,
            )?;
            (
                IrType::I64,
                ScalarOperation::MultiplyHigh {
                    signedness: if opcode == 2 {
                        IntegerSignedness::Signed
                    } else {
                        IntegerSignedness::Unsigned
                    },
                    lhs,
                    rhs,
                },
            )
        }
        _ => return Ok(unsupported(decoded)),
    };
    let result = scalar(builder, decoded.location, result_width, operation)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_one_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let opcode = u32::from(fields.shift_amount);
    let input = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let value = match opcode {
        0 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ReverseBits { value: input },
        )?,
        1..=3 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ReverseBytes {
                value: input,
                container: match opcode {
                    1 => ByteReverseWidth::Bits16,
                    2 => ByteReverseWidth::Bits32,
                    3 => ByteReverseWidth::Full,
                    _ => unreachable!(),
                },
            },
        )?,
        4 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::CountLeadingZeros { value: input },
        )?,
        5 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::CountLeadingSignBits { value: input },
        )?,
        _ => return Ok(unsupported(decoded)),
    };
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_adr(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: IntegerOperands,
    page_relative: bool,
) -> Result<LiftOutcome, BuildError> {
    let immediate = sign_extend(u64::from(fields.adr_immediate), 21);
    let address = if page_relative {
        GuestVirtualAddress::new(decoded.location.pc.get() & !0xfff)
            .wrapping_offset(immediate << 12)
    } else {
        decoded.location.pc.wrapping_offset(immediate)
    };
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        Immediate::I64(address.get()).into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}
