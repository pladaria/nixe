use super::*;

use crate::{
    decode::{
        DecodedOpcode,
        a64::{A64Instruction, IntegerOperation},
    },
    ir::builder::{BuildError, IrBuilder},
    location::DecodedInstruction,
};

use super::LiftOutcome;

pub(super) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: A64Instruction,
    operation: IntegerOperation,
) -> Result<LiftOutcome, BuildError> {
    let fields = instruction.fields;
    match operation {
        IntegerOperation::MoveWide => lift_move_wide(builder, decoded, fields),
        IntegerOperation::AddSubImmediate => lift_add_sub_immediate(builder, decoded, fields),
        IntegerOperation::AddSubShifted => lift_add_sub_shifted(builder, decoded, fields),
        IntegerOperation::AddSubExtended => lift_add_sub_extended(builder, decoded, fields),
        IntegerOperation::AddSubCarry => lift_add_sub_carry(builder, decoded, fields),
        IntegerOperation::LogicalImmediate => lift_logical_immediate(builder, decoded, fields),
        IntegerOperation::LogicalShifted => lift_logical_shifted(builder, decoded, fields),
        IntegerOperation::Bitfield => lift_bitfield(builder, decoded, fields),
        IntegerOperation::Extract => lift_extract(builder, decoded, fields),
        IntegerOperation::TwoSource => lift_two_source(builder, decoded, fields),
        IntegerOperation::ConditionalCompareRegister
        | IntegerOperation::ConditionalCompareImmediate => {
            lift_conditional_compare(builder, decoded, fields)
        }
        IntegerOperation::ConditionalSelect => lift_conditional_select(builder, decoded, fields),
        IntegerOperation::ThreeSource => lift_three_source(builder, decoded, fields),
        IntegerOperation::OneSource => lift_one_source(builder, decoded, fields),
        IntegerOperation::Adr => lift_adr(builder, decoded, fields, false),
        IntegerOperation::Adrp => lift_adr(builder, decoded, fields, true),
    }
}

pub(super) fn integer_width(fields: A64Fields) -> IrType {
    if fields.bit31 {
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
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let hw = u32::from(fields.field_21_2);
    if width == IrType::I32 && hw >= 2 {
        return Ok(interpret(decoded));
    }
    let shift = hw * 16;
    let imm = u64::from(u32::from(fields.immediate_16)) << shift;
    let opc = u32::from((fields.bit30 as u8) * 2 + fields.bit29 as u8);
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
            let mask = !(0xffff_u64 << shift);
            let retained = binary(
                builder,
                decoded.location,
                IntegerBinaryKind::And,
                old,
                immediate_for(width, mask).into(),
            )?;
            binary(
                builder,
                decoded.location,
                IntegerBinaryKind::Or,
                retained,
                immediate_for(width, imm).into(),
            )?
        }
        _ => return Ok(interpret(decoded)),
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
    width: IrType,
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
    let rhs = if spec.subtract {
        binary(
            builder,
            source,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(spec.width, u64::MAX).into(),
        )?
    } else {
        rhs
    };
    let results: Vec<_> = builder
        .emit(
            source,
            &[spec.width, IrType::I1, IrType::I1],
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in: Immediate::I1(spec.subtract).into(),
            }),
        )?
        .iter()
        .collect();
    write_gpr(
        builder,
        source,
        spec.destination,
        results[0].into(),
        spec.destination_register31,
    )?;
    if spec.set_flags {
        arithmetic_flags(
            builder,
            source,
            results[0].into(),
            results[1].into(),
            results[2].into(),
        )?;
    }
    Ok(())
}

fn lift_add_sub_immediate(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::StackPointer,
    )?;
    let shift = if fields.bit22 { 12 } else { 0 };
    let rhs = immediate_for(width, u64::from(u32::from(fields.field_10_12)) << shift).into();
    let set_flags = fields.bit29;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        rhs,
        AddSubSpec {
            width,
            subtract: fields.bit30,
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
    fields: A64Fields,
    width: IrType,
    index: u8,
) -> Result<Option<Operand>, BuildError> {
    let amount = u32::from(fields.field_10_6);
    if width == IrType::I32 && amount >= 32 {
        return Ok(None);
    }
    let kind = match u32::from(fields.field_22_2) {
        0 => ShiftKind::LogicalLeft,
        1 => ShiftKind::LogicalRight,
        2 => ShiftKind::ArithmeticRight,
        3 => ShiftKind::RotateRight,
        _ => unreachable!(),
    };
    let value = read_gpr(builder, source, index, width, Register31::Zero)?;
    if amount == 0 {
        Ok(Some(value))
    } else {
        Ok(Some(scalar(
            builder,
            source,
            width,
            ScalarOperation::Shift {
                kind,
                value,
                amount: immediate_for(width, u64::from(amount)).into(),
            },
        )?))
    }
}

fn lift_add_sub_shifted(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    if u32::from(fields.field_22_2) == 3 {
        return Ok(interpret(decoded));
    }
    let width = integer_width(fields);
    let Some(rhs) = shifted_register(builder, decoded.location, fields, width, fields.rm)? else {
        return Ok(interpret(decoded));
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
            width,
            subtract: fields.bit30,
            set_flags: fields.bit29,
            destination: fields.rd,
            destination_register31: Register31::Zero,
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_add_sub_extended(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let shift = u32::from(fields.field_10_3);
    if shift > 4 {
        return Ok(interpret(decoded));
    }
    let rm = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let extension = (u32::from(fields.field_13_3)) as u64;
    let result = helper(
        builder,
        decoded.location,
        "a64.extend-register",
        vec![
            rm,
            Immediate::I8(extension as u8).into(),
            Immediate::I8(shift as u8).into(),
        ],
        &[width],
        OperationEffects::default(),
    )?[0];
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::StackPointer,
    )?;
    let set_flags = fields.bit29;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        result.into(),
        AddSubSpec {
            width,
            subtract: fields.bit30,
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
    evaluate_condition(builder, source, Condition::Cs)
}

fn lift_add_sub_carry(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let mut rhs = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let subtract = fields.bit30;
    if subtract {
        rhs = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    let carry = carry_in(builder, decoded.location)?;
    let values: Vec<_> = builder
        .emit(
            decoded.location,
            &[width, IrType::I1, IrType::I1],
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in: carry,
            }),
        )?
        .iter()
        .collect();
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        values[0].into(),
        Register31::Zero,
    )?;
    if fields.bit29 {
        arithmetic_flags(
            builder,
            decoded.location,
            values[0].into(),
            values[1].into(),
            values[2].into(),
        )?;
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
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let size = if width == IrType::I64 { 64 } else { 32 };
    let Ok(immediate) = decode_a64_logical_immediate(
        fields.bit22,
        (u32::from(fields.field_16_6)) as u8,
        (u32::from(fields.field_10_6)) as u8,
        size,
    ) else {
        return Ok(interpret(decoded));
    };
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let opc = u32::from((fields.bit30 as u8) * 2 + fields.bit29 as u8);
    let result = logical_result(
        builder,
        decoded.location,
        opc,
        lhs,
        immediate_for(width, immediate).into(),
    )?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    if opc == 3 {
        logical_flags(builder, decoded.location, result)?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_logical_shifted(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let Some(mut rhs) = shifted_register(builder, decoded.location, fields, width, fields.rm)?
    else {
        return Ok(interpret(decoded));
    };
    if fields.bit21 {
        rhs = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let opc = u32::from((fields.bit30 as u8) * 2 + fields.bit29 as u8);
    let result = logical_result(builder, decoded.location, opc, lhs, rhs)?;
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result,
        Register31::Zero,
    )?;
    if opc == 3 {
        logical_flags(builder, decoded.location, result)?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_bitfield(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let n = fields.bit22;
    if n != (width == IrType::I64) || (width == IrType::I32 && fields.bit15) {
        return Ok(interpret(decoded));
    }
    let opc = u32::from((fields.bit30 as u8) * 2 + fields.bit29 as u8);
    if opc == 3 {
        return Ok(interpret(decoded));
    }
    let source_value = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let destination = read_gpr(
        builder,
        decoded.location,
        fields.rd,
        width,
        Register31::Zero,
    )?;
    let name = match opc {
        0 => "a64.sbfm",
        1 => "a64.bfm",
        2 => "a64.ubfm",
        _ => unreachable!(),
    };
    let value = helper(
        builder,
        decoded.location,
        name,
        vec![
            destination,
            source_value,
            Immediate::I8(fields.field_16_6).into(),
            Immediate::I8((u32::from(fields.field_10_6)) as u8).into(),
        ],
        &[width],
        OperationEffects::default(),
    )?[0];
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value.into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_extract(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lsb = u32::from(fields.field_10_6);
    if (fields.bit22) != (width == IrType::I64) || (width == IrType::I32 && lsb >= 32) {
        return Ok(interpret(decoded));
    }
    let first = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let second = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let value = helper(
        builder,
        decoded.location,
        "a64.extr",
        vec![first, second, Immediate::I8(lsb as u8).into()],
        &[width],
        OperationEffects::default(),
    )?[0];
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        value.into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_two_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let opcode = u32::from(fields.field_10_6);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let mut rhs = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    if matches!(opcode, 8..=11) {
        rhs = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::And,
            rhs,
            immediate_for(width, if width == IrType::I64 { 63 } else { 31 }).into(),
        )?;
    }
    let operation = match opcode {
        2 => ScalarOperation::Binary {
            kind: IntegerBinaryKind::UnsignedDivide,
            lhs,
            rhs,
        },
        3 => ScalarOperation::Binary {
            kind: IntegerBinaryKind::SignedDivide,
            lhs,
            rhs,
        },
        8 => ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: lhs,
            amount: rhs,
        },
        9 => ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: lhs,
            amount: rhs,
        },
        10 => ScalarOperation::Shift {
            kind: ShiftKind::ArithmeticRight,
            value: lhs,
            amount: rhs,
        },
        11 => ScalarOperation::Shift {
            kind: ShiftKind::RotateRight,
            value: lhs,
            amount: rhs,
        },
        _ => return Ok(interpret(decoded)),
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
    width: IrType,
    lhs: Operand,
    mut rhs: Operand,
    subtract: bool,
) -> Result<Operand, BuildError> {
    if subtract {
        rhs = binary(
            builder,
            source,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    let values: Vec<_> = builder
        .emit(
            source,
            &[width, IrType::I1, IrType::I1],
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in: Immediate::I1(subtract).into(),
            }),
        )?
        .iter()
        .collect();
    Ok(emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromArithmetic {
            result: values[0].into(),
            carry: values[1].into(),
            overflow: values[2].into(),
        }),
    )?
    .into())
}

fn lift_conditional_compare(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let lhs = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let rhs = if fields.bit11 {
        immediate_for(width, u64::from(u32::from(fields.field_16_5))).into()
    } else {
        read_gpr(
            builder,
            decoded.location,
            fields.rm,
            width,
            Register31::Zero,
        )?
    };
    let proposed =
        proposed_compare_flags(builder, decoded.location, width, lhs, rhs, fields.bit30)?;
    let fallback = emit_one(
        builder,
        decoded.location,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked {
            value: Immediate::I32(u32::from(fields.low4) << 28).into(),
        }),
    )?;
    let cond = evaluate_condition(
        builder,
        decoded.location,
        condition(u32::from(fields.field_12_4)),
    )?;
    let selected = scalar(
        builder,
        decoded.location,
        IrType::Flags,
        ScalarOperation::Select {
            condition: cond,
            when_true: proposed,
            when_false: fallback.into(),
        },
    )?;
    write_flags(builder, decoded.location, selected)?;
    Ok(LiftOutcome::Continue)
}

fn lift_conditional_select(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let true_value = read_gpr(
        builder,
        decoded.location,
        fields.rn,
        width,
        Register31::Zero,
    )?;
    let mut false_value = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        width,
        Register31::Zero,
    )?;
    let op = fields.bit30;
    let op2 = fields.bit10;
    if op {
        false_value = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Xor,
            false_value,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    if op2 {
        false_value = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            false_value,
            immediate_for(width, 1).into(),
        )?;
    }
    let cond = evaluate_condition(
        builder,
        decoded.location,
        condition(u32::from(fields.field_12_4)),
    )?;
    let result = scalar(
        builder,
        decoded.location,
        width,
        ScalarOperation::Select {
            condition: cond,
            when_true: true_value,
            when_false: false_value,
        },
    )?;
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
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let opcode = u32::from(fields.field_21_3);
    if opcode != 0 {
        if matches!(opcode, 2 | 6) && (u32::from(fields.ra)) != 31 {
            return Ok(interpret(decoded));
        }
        let name = match (opcode, fields.bit15) {
            (1, false) => "a64.smaddl",
            (1, true) => "a64.smsubl",
            (2, false) => "a64.smulh",
            (5, false) => "a64.umaddl",
            (5, true) => "a64.umsubl",
            (6, false) => "a64.umulh",
            _ => return Ok(interpret(decoded)),
        };
        let operand_width = if matches!(opcode, 1 | 5) {
            IrType::I32
        } else {
            IrType::I64
        };
        let mut values = vec![
            read_gpr(
                builder,
                decoded.location,
                fields.rn,
                operand_width,
                Register31::Zero,
            )?,
            read_gpr(
                builder,
                decoded.location,
                fields.rm,
                operand_width,
                Register31::Zero,
            )?,
        ];
        if matches!(opcode, 1 | 5) {
            values.push(read_gpr(
                builder,
                decoded.location,
                fields.ra,
                IrType::I64,
                Register31::Zero,
            )?);
        }
        let result = helper(
            builder,
            decoded.location,
            name,
            values,
            &[IrType::I64],
            OperationEffects::default(),
        )?[0];
        write_gpr(
            builder,
            decoded.location,
            fields.rd,
            result.into(),
            Register31::Zero,
        )?;
        return Ok(LiftOutcome::Continue);
    }
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
    let product = binary(
        builder,
        decoded.location,
        IntegerBinaryKind::Multiply,
        lhs,
        rhs,
    )?;
    let result = binary(
        builder,
        decoded.location,
        if fields.bit15 {
            IntegerBinaryKind::Subtract
        } else {
            IntegerBinaryKind::Add
        },
        addend,
        product,
    )?;
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
    fields: A64Fields,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(fields);
    let opcode = u32::from(fields.field_10_6);
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
        4 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::CountLeadingZeros { value: input },
        )?,
        1 | 2 | 3 | 5 => {
            let name = match opcode {
                1 => "a64.rev16",
                2 => "a64.rev32",
                3 => "a64.rev",
                5 => "a64.cls",
                _ => unreachable!(),
            };
            helper(
                builder,
                decoded.location,
                name,
                vec![input],
                &[width],
                OperationEffects::default(),
            )?[0]
                .into()
        }
        _ => return Ok(interpret(decoded)),
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
    fields: A64Fields,
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
