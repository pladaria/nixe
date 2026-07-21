use super::*;

use crate::{
    decode::{
        DecodedOpcode,
        a64::fp_simd::{Instruction as FpSimdInstruction, Operands as FpSimdOperands},
    },
    ir::builder::{BuildError, IrBuilder},
    location::DecodedInstruction,
};

use super::LiftOutcome;

pub(super) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: FpSimdInstruction,
) -> Result<LiftOutcome, BuildError> {
    let fields = instruction.operands();
    match instruction {
        FpSimdInstruction::Bitwise(_)
        | FpSimdInstruction::Integer(_)
        | FpSimdInstruction::ScalarTwoSource(_)
        | FpSimdInstruction::ScalarMove(_)
        | FpSimdInstruction::CompareRegister(_)
        | FpSimdInstruction::CompareZero(_) => {
            lift_fp_simd_compute(builder, decoded, fields, instruction)
        }
        FpSimdInstruction::SignedIntToFloat(_)
        | FpSimdInstruction::UnsignedIntToFloat(_)
        | FpSimdInstruction::FloatToSignedInt(_)
        | FpSimdInstruction::FloatToUnsignedInt(_)
        | FpSimdInstruction::MoveToGeneral(_)
        | FpSimdInstruction::MoveFromGeneral(_) => {
            lift_fp_conversion(builder, decoded, fields, instruction)
        }
        FpSimdInstruction::MemoryUnsigned(_)
        | FpSimdInstruction::MemoryUnscaled(_)
        | FpSimdInstruction::MemoryPostIndex(_)
        | FpSimdInstruction::MemoryPreIndex(_)
        | FpSimdInstruction::MemoryRegister(_)
        | FpSimdInstruction::MemoryLiteral(_) => {
            lift_fp_simd_memory(builder, decoded, fields, instruction)
        }
    }
}

fn vector_read(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::V128,
        OperationKind::ReadState(StateRegister::A64V(
            crate::ir::op::RegisterIndex::new(index).unwrap(),
        )),
    )?
    .into())
}

fn vector_write(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    value: Operand,
) -> Result<(), BuildError> {
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64V(crate::ir::op::RegisterIndex::new(index).unwrap()),
            value,
        },
    )?;
    Ok(())
}

fn lift_fp_simd_compute(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
    operation: FpSimdInstruction,
) -> Result<LiftOutcome, BuildError> {
    let first = vector_read(builder, decoded.location, fields.rn)?;
    if matches!(
        operation,
        FpSimdInstruction::Bitwise(_)
            | FpSimdInstruction::Integer(_)
            | FpSimdInstruction::ScalarMove(_)
    ) {
        let mut arguments = vec![first];
        if !matches!(operation, FpSimdInstruction::ScalarMove(_)) {
            arguments.push(vector_read(builder, decoded.location, fields.rm)?);
            arguments.push(vector_read(builder, decoded.location, fields.rd)?);
        }
        arguments.push(Immediate::I32(fields.helper_token.helper_abi_value()).into());
        let name = match operation {
            FpSimdInstruction::Bitwise(_) => "a64.simd.bitwise",
            FpSimdInstruction::Integer(_) => "a64.simd.integer-arithmetic-compare",
            FpSimdInstruction::ScalarMove(_) => "a64.fp.scalar-move",
            _ => unreachable!(),
        };
        let result = helper(
            builder,
            decoded.location,
            name,
            arguments,
            &[IrType::V128],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0];
        vector_write(builder, decoded.location, fields.rd, result.into())?;
        return Ok(LiftOutcome::Continue);
    }

    let compare = matches!(
        operation,
        FpSimdInstruction::CompareRegister(_) | FpSimdInstruction::CompareZero(_)
    );
    let second = if matches!(operation, FpSimdInstruction::CompareZero(_)) {
        Immediate::V128(0).into()
    } else {
        vector_read(builder, decoded.location, fields.rm)?
    };
    let fpcr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpcr),
    )?;
    let fpsr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpsr),
    )?;
    let result_types: &[IrType] = if compare {
        &[IrType::Flags, IrType::I32]
    } else {
        &[IrType::V128, IrType::I32]
    };
    let results = helper(
        builder,
        decoded.location,
        if compare {
            "a64.fp.scalar-compare"
        } else {
            "a64.fp.scalar-arithmetic"
        },
        vec![
            first,
            second,
            fpcr.into(),
            fpsr.into(),
            Immediate::I32(fields.helper_token.helper_abi_value()).into(),
        ],
        result_types,
        OperationEffects::new(
            EffectSet::READ_FPCR
                .union(EffectSet::WRITE_FPSR)
                .union(EffectSet::HELPER),
            false,
        ),
    )?;
    if compare {
        write_flags(builder, decoded.location, results[0].into())?;
    } else {
        vector_write(builder, decoded.location, fields.rd, results[0].into())?;
    }
    builder.emit(
        decoded.location,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Fpsr,
            value: results[1].into(),
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_fp_conversion(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
    operation: FpSimdInstruction,
) -> Result<LiftOutcome, BuildError> {
    if u32::from(fields.opc) > 1 {
        return Ok(interpret(decoded));
    }
    let width = if fields.size & 2 != 0 {
        IrType::I64
    } else {
        IrType::I32
    };
    let rn = fields.rn;
    let rd = fields.rd;

    if matches!(operation, FpSimdInstruction::MoveToGeneral(_)) {
        let vector = vector_read(builder, decoded.location, rn)?;
        let result = helper(
            builder,
            decoded.location,
            "a64.fp.move-to-general",
            vec![
                vector,
                Immediate::I32(fields.helper_token.helper_abi_value()).into(),
            ],
            &[width],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0];
        write_gpr(
            builder,
            decoded.location,
            rd,
            result.into(),
            Register31::Zero,
        )?;
        return Ok(LiftOutcome::Continue);
    }
    if matches!(operation, FpSimdInstruction::MoveFromGeneral(_)) {
        let integer = read_gpr(builder, decoded.location, rn, width, Register31::Zero)?;
        let result = helper(
            builder,
            decoded.location,
            "a64.fp.move-from-general",
            vec![
                integer,
                Immediate::I32(fields.helper_token.helper_abi_value()).into(),
            ],
            &[IrType::V128],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0];
        vector_write(builder, decoded.location, rd, result.into())?;
        return Ok(LiftOutcome::Continue);
    }

    let fpcr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpcr),
    )?;
    let fpsr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpsr),
    )?;
    let effects = OperationEffects::new(
        EffectSet::READ_FPCR
            .union(EffectSet::WRITE_FPSR)
            .union(EffectSet::HELPER),
        false,
    );
    let int_to_float = matches!(
        operation,
        FpSimdInstruction::SignedIntToFloat(_) | FpSimdInstruction::UnsignedIntToFloat(_)
    );
    let results = if int_to_float {
        let integer = read_gpr(builder, decoded.location, rn, width, Register31::Zero)?;
        helper(
            builder,
            decoded.location,
            if matches!(operation, FpSimdInstruction::SignedIntToFloat(_)) {
                "a64.fp.signed-int-to-float"
            } else {
                "a64.fp.unsigned-int-to-float"
            },
            vec![
                integer,
                fpcr.into(),
                fpsr.into(),
                Immediate::I32(fields.helper_token.helper_abi_value()).into(),
            ],
            &[IrType::V128, IrType::I32],
            effects,
        )?
    } else {
        let vector = vector_read(builder, decoded.location, rn)?;
        helper(
            builder,
            decoded.location,
            if matches!(operation, FpSimdInstruction::FloatToSignedInt(_)) {
                "a64.fp.float-to-signed-int"
            } else {
                "a64.fp.float-to-unsigned-int"
            },
            vec![
                vector,
                fpcr.into(),
                fpsr.into(),
                Immediate::I32(fields.helper_token.helper_abi_value()).into(),
            ],
            &[width, IrType::I32],
            effects,
        )?
    };
    if int_to_float {
        vector_write(builder, decoded.location, rd, results[0].into())?;
    } else {
        write_gpr(
            builder,
            decoded.location,
            rd,
            results[0].into(),
            Register31::Zero,
        )?;
    }
    builder.emit(
        decoded.location,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Fpsr,
            value: results[1].into(),
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_fp_simd_memory(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
    operation: FpSimdInstruction,
) -> Result<LiftOutcome, BuildError> {
    let literal = matches!(operation, FpSimdInstruction::MemoryLiteral(_));
    let size = if literal {
        match fields.size {
            0 => MemoryAccessSize::Word,
            1 => MemoryAccessSize::Doubleword,
            2 => MemoryAccessSize::Quadword,
            _ => return Ok(interpret(decoded)),
        }
    } else if fields.quad {
        MemoryAccessSize::Quadword
    } else {
        memory::size_from_bits(u32::from(fields.size))
    };
    let rn = fields.rn;
    let mut writeback = None;
    let address = if literal {
        let target = decoded
            .location
            .pc
            .wrapping_offset(sign_extend(u64::from(fields.immediate_19), 19) << 2);
        Immediate::Address(target).into()
    } else {
        let base = memory::base_address(builder, decoded.location, rn)?;
        if matches!(operation, FpSimdInstruction::MemoryRegister(_)) {
            let option = u32::from(fields.option);
            if option & 2 == 0 {
                return Ok(interpret(decoded));
            }
            let raw_offset = read_gpr(
                builder,
                decoded.location,
                fields.rm,
                IrType::I64,
                Register31::Zero,
            )?;
            let shift = if fields.scaled {
                size.bytes().trailing_zeros() as u8
            } else {
                0
            };
            let offset = helper(
                builder,
                decoded.location,
                "a64.load-store-register-offset",
                vec![
                    raw_offset,
                    Immediate::I8(option as u8).into(),
                    Immediate::I8(shift).into(),
                ],
                &[IrType::I64],
                OperationEffects::default(),
            )?[0];
            guest_address_offset(builder, decoded.location, base, offset.into())?
        } else {
            let offset = if matches!(operation, FpSimdInstruction::MemoryUnsigned(_)) {
                i64::from(u32::from(fields.immediate_12)) * size.bytes() as i64
            } else {
                sign_extend(u64::from(u32::from(fields.immediate_9)), 9)
            };
            let transfer_base = if matches!(operation, FpSimdInstruction::MemoryPostIndex(_)) {
                base
            } else {
                guest_address_offset(
                    builder,
                    decoded.location,
                    base,
                    Immediate::I64(offset as u64).into(),
                )?
            };
            if matches!(
                operation,
                FpSimdInstruction::MemoryPreIndex(_) | FpSimdInstruction::MemoryPostIndex(_)
            ) {
                let updated_address = guest_address_offset(
                    builder,
                    decoded.location,
                    base,
                    Immediate::I64(offset as u64).into(),
                )?;
                writeback = Some(guest_address_to_integer(
                    builder,
                    decoded.location,
                    updated_address,
                )?);
            }
            transfer_base
        }
    };
    let descriptor = memory::descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal);
    let rt = fields.rd;
    if literal || fields.load {
        let raw = emit_one(
            builder,
            decoded.location,
            descriptor.value_type(),
            OperationKind::Memory(MemoryOperation::Load {
                address,
                descriptor,
            }),
        )?;
        let vector = helper(
            builder,
            decoded.location,
            "a64.simd.zero-extend-load",
            vec![raw.into()],
            &[IrType::V128],
            OperationEffects::default(),
        )?[0];
        vector_write(builder, decoded.location, rt, vector.into())?;
    } else {
        let vector = vector_read(builder, decoded.location, rt)?;
        let raw = helper(
            builder,
            decoded.location,
            "a64.simd.low-bits",
            vec![vector],
            &[descriptor.value_type()],
            OperationEffects::default(),
        )?[0];
        builder.emit(
            decoded.location,
            &[],
            OperationKind::Memory(MemoryOperation::Store {
                address,
                value: raw.into(),
                descriptor,
            }),
        )?;
    }
    if let Some(updated) = writeback {
        write_gpr(
            builder,
            decoded.location,
            rn,
            updated,
            Register31::StackPointer,
        )?;
    }
    Ok(LiftOutcome::Continue)
}
