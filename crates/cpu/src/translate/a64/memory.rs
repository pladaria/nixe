use super::*;

use crate::{
    decode::{
        DecodedOpcode,
        a64::memory::{Instruction as A64MemoryInstruction, Operands as MemoryOperands},
    },
    ir::builder::{BuildError, IrBuilder},
    location::DecodedInstruction,
};

use super::LiftOutcome;

pub(super) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: A64MemoryInstruction,
) -> Result<LiftOutcome, BuildError> {
    let fields = instruction.operands();
    match instruction {
        A64MemoryInstruction::Literal(_) => lift_literal_load(builder, decoded, fields),
        A64MemoryInstruction::Unsigned(_) => lift_load_store_unsigned(builder, decoded, fields),
        A64MemoryInstruction::Unscaled(_)
        | A64MemoryInstruction::PostIndex(_)
        | A64MemoryInstruction::PreIndex(_) => {
            lift_load_store_indexed(builder, decoded, fields, instruction)
        }
        A64MemoryInstruction::Register(_) => lift_load_store_register(builder, decoded, fields),
        A64MemoryInstruction::Pair(_) => lift_load_store_pair(builder, decoded, fields),
        A64MemoryInstruction::LoadAcquire(_) | A64MemoryInstruction::StoreRelease(_) => {
            lift_acquire_release(builder, decoded, fields, instruction)
        }
        A64MemoryInstruction::LoadExclusive(_) | A64MemoryInstruction::StoreExclusive(_) => {
            lift_exclusive(builder, decoded, fields, instruction)
        }
    }
}

pub(super) fn descriptor(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    class: MemoryAccessClass,
) -> MemoryDescriptor {
    MemoryDescriptor {
        access: MemoryAccess::new(size, MemoryAlignment::Unaligned, ordering, class),
        byte_order: ByteOrder::Little,
        volatility: Volatility::NonVolatile,
    }
}

fn aligned_descriptor(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    class: MemoryAccessClass,
) -> MemoryDescriptor {
    let mut descriptor = descriptor(size, ordering, class);
    descriptor.access.alignment = MemoryAlignment::Natural;
    descriptor
}

pub(super) fn size_from_bits(size: u32) -> MemoryAccessSize {
    match size {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        _ => unreachable!(),
    }
}

fn address_add(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    base: Operand,
    offset: i64,
) -> Result<Operand, BuildError> {
    let raw = binary(
        builder,
        source,
        IntegerBinaryKind::Add,
        base,
        Immediate::I64(offset as u64).into(),
    )?;
    bitcast(builder, source, raw, IrType::Address)
}

pub(super) fn base_address(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    rn: u8,
) -> Result<Operand, BuildError> {
    read_gpr(builder, source, rn, IrType::I64, Register31::StackPointer)
}

fn memory_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    fields: MemoryOperands,
    address: Operand,
    descriptor: MemoryDescriptor,
) -> Result<bool, BuildError> {
    let opc = u32::from(fields.opc);
    let rt = fields.rt;
    if opc == 0 {
        let value = read_gpr(
            builder,
            source,
            rt,
            descriptor.value_type(),
            Register31::Zero,
        )?;
        builder.emit(
            source,
            &[],
            OperationKind::Memory(MemoryOperation::Store {
                address,
                value,
                descriptor,
            }),
        )?;
        return Ok(true);
    }
    if (opc >= 2 && descriptor.access.size == MemoryAccessSize::Doubleword)
        || (opc == 3 && descriptor.access.size == MemoryAccessSize::Word)
    {
        return Ok(false);
    }
    let loaded = emit_one(
        builder,
        source,
        descriptor.value_type(),
        OperationKind::Memory(MemoryOperation::Load {
            address,
            descriptor,
        }),
    )?;
    let destination_width = if opc == 2 || descriptor.access.size == MemoryAccessSize::Doubleword {
        IrType::I64
    } else {
        IrType::I32
    };
    let mut value: Operand = loaded.into();
    if descriptor.value_type() != destination_width {
        value = scalar(
            builder,
            source,
            destination_width,
            if matches!(opc, 2 | 3) {
                ScalarOperation::SignExtend {
                    value,
                    to: destination_width,
                }
            } else {
                ScalarOperation::ZeroExtend {
                    value,
                    to: destination_width,
                }
            },
        )?;
    }
    write_gpr(builder, source, rt, value, Register31::Zero)?;
    Ok(true)
}

fn lift_literal_load(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
) -> Result<LiftOutcome, BuildError> {
    let opc = u32::from(fields.size);
    if opc == 3 {
        return Ok(interpret(decoded)); // PRFM literal
    }
    let size = match opc {
        0 | 2 => MemoryAccessSize::Word,
        1 => MemoryAccessSize::Doubleword,
        3 => return Ok(interpret(decoded)), // PRFM literal
        _ => unreachable!(),
    };
    let address = bitcast(
        builder,
        decoded.location,
        Immediate::I64(
            decoded
                .location
                .pc
                .wrapping_offset(sign_extend(u64::from(fields.immediate_19), 19) << 2)
                .get(),
        )
        .into(),
        IrType::Address,
    )?;
    let descriptor = descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal);
    let loaded = emit_one(
        builder,
        decoded.location,
        descriptor.value_type(),
        OperationKind::Memory(MemoryOperation::Load {
            address,
            descriptor,
        }),
    )?;
    let value = if opc == 2 {
        scalar(
            builder,
            decoded.location,
            IrType::I64,
            ScalarOperation::SignExtend {
                value: loaded.into(),
                to: IrType::I64,
            },
        )?
    } else {
        loaded.into()
    };
    write_gpr(
        builder,
        decoded.location,
        fields.rt,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_unsigned(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(u32::from(fields.size));
    let base = base_address(builder, decoded.location, fields.rn)?;
    let address = address_add(
        builder,
        decoded.location,
        base,
        i64::from(u32::from(fields.immediate_12)) * size.bytes() as i64,
    )?;
    if !memory_transfer(
        builder,
        decoded.location,
        fields,
        address,
        descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal),
    )? {
        return Ok(interpret(decoded));
    }
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_indexed(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
    instruction: A64MemoryInstruction,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(u32::from(fields.size));
    let rn = fields.rn;
    let rt = fields.rt;
    if !matches!(instruction, A64MemoryInstruction::Unscaled(_)) && rn != 31 && rn == rt {
        return Ok(interpret(decoded));
    }
    let base = base_address(builder, decoded.location, rn)?;
    let offset = sign_extend(u64::from(u32::from(fields.immediate_9)), 9);
    let address = if matches!(instruction, A64MemoryInstruction::PreIndex(_)) {
        address_add(builder, decoded.location, base, offset)?
    } else {
        bitcast(builder, decoded.location, base, IrType::Address)?
    };
    if !memory_transfer(
        builder,
        decoded.location,
        fields,
        address,
        descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal),
    )? {
        return Ok(interpret(decoded));
    }
    if !matches!(instruction, A64MemoryInstruction::Unscaled(_)) {
        let updated = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            base,
            Immediate::I64(offset as u64).into(),
        )?;
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

fn lift_load_store_register(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(u32::from(fields.size));
    let base = base_address(builder, decoded.location, fields.rn)?;
    let offset = read_gpr(
        builder,
        decoded.location,
        fields.rm,
        IrType::I64,
        Register31::Zero,
    )?;
    let option = u32::from(fields.option);
    if option & 2 == 0 {
        return Ok(interpret(decoded));
    }
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
            offset,
            Immediate::I8(option as u8).into(),
            Immediate::I8(shift).into(),
        ],
        &[IrType::I64],
        OperationEffects::default(),
    )?[0];
    let raw = binary(
        builder,
        decoded.location,
        IntegerBinaryKind::Add,
        base,
        offset.into(),
    )?;
    let address = bitcast(builder, decoded.location, raw, IrType::Address)?;
    if !memory_transfer(
        builder,
        decoded.location,
        fields,
        address,
        descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal),
    )? {
        return Ok(interpret(decoded));
    }
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_pair(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
) -> Result<LiftOutcome, BuildError> {
    let opc = u32::from(fields.size);
    if opc == 3 {
        return Ok(interpret(decoded));
    }
    let size = if matches!(opc, 0 | 1) {
        MemoryAccessSize::Word
    } else {
        MemoryAccessSize::Doubleword
    };
    let width = if size == MemoryAccessSize::Word {
        IrType::I32
    } else {
        IrType::I64
    };
    let rn = fields.rn;
    let rt = fields.rt;
    let rt2 = fields.rt2;
    let mode = u32::from(fields.mode);
    let load = fields.load;
    if (load && rt == rt2) || (matches!(mode, 1 | 3) && rn != 31 && (rn == rt || rn == rt2)) {
        return Ok(interpret(decoded));
    }
    let base = base_address(builder, decoded.location, rn)?;
    let offset = sign_extend(u64::from(fields.immediate_7), 7) * size.bytes() as i64;
    let transfer_base = if mode == 3 {
        binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            base,
            Immediate::I64(offset as u64).into(),
        )?
    } else {
        base
    };
    let first_address = bitcast(builder, decoded.location, transfer_base, IrType::Address)?;
    let second_address = address_add(
        builder,
        decoded.location,
        transfer_base,
        size.bytes() as i64,
    )?;
    if opc == 1 && !load {
        return Ok(interpret(decoded));
    }
    let descriptor = descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal);
    for (rt, address) in [(fields.rt, first_address), (fields.rt2, second_address)] {
        if load {
            let mut value: Operand = emit_one(
                builder,
                decoded.location,
                width,
                OperationKind::Memory(MemoryOperation::Load {
                    address,
                    descriptor,
                }),
            )?
            .into();
            if opc == 1 {
                value = scalar(
                    builder,
                    decoded.location,
                    IrType::I64,
                    ScalarOperation::SignExtend {
                        value,
                        to: IrType::I64,
                    },
                )?;
            }
            write_gpr(builder, decoded.location, rt, value, Register31::Zero)?;
        } else {
            let value = read_gpr(builder, decoded.location, rt, width, Register31::Zero)?;
            builder.emit(
                decoded.location,
                &[],
                OperationKind::Memory(MemoryOperation::Store {
                    address,
                    value,
                    descriptor,
                }),
            )?;
        }
    }
    if matches!(mode, 1 | 3) {
        let updated = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            base,
            Immediate::I64(offset as u64).into(),
        )?;
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

fn lift_acquire_release(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
    instruction: A64MemoryInstruction,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(u32::from(fields.size));
    let base = base_address(builder, decoded.location, fields.rn)?;
    let address = bitcast(builder, decoded.location, base, IrType::Address)?;
    let load = matches!(instruction, A64MemoryInstruction::LoadAcquire(_));
    let ordering = if load {
        MemoryOrdering::Acquire
    } else {
        MemoryOrdering::Release
    };
    let descriptor = aligned_descriptor(size, ordering, MemoryAccessClass::Normal);
    let rt = fields.rt;
    if load {
        let value = emit_one(
            builder,
            decoded.location,
            descriptor.value_type(),
            OperationKind::Memory(MemoryOperation::Load {
                address,
                descriptor,
            }),
        )?;
        write_gpr(
            builder,
            decoded.location,
            rt,
            value.into(),
            Register31::Zero,
        )?;
    } else {
        let value = read_gpr(
            builder,
            decoded.location,
            rt,
            descriptor.value_type(),
            Register31::Zero,
        )?;
        builder.emit(
            decoded.location,
            &[],
            OperationKind::Memory(MemoryOperation::Store {
                address,
                value,
                descriptor,
            }),
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_exclusive(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: MemoryOperands,
    instruction: A64MemoryInstruction,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(u32::from(fields.size));
    let base = base_address(builder, decoded.location, fields.rn)?;
    let address = bitcast(builder, decoded.location, base, IrType::Address)?;
    let ordered = fields.ordered;
    let ordering = match (instruction, ordered) {
        (A64MemoryInstruction::LoadExclusive(_), true) => MemoryOrdering::Acquire,
        (A64MemoryInstruction::StoreExclusive(_), true) => MemoryOrdering::Release,
        (_, false) => MemoryOrdering::Relaxed,
        _ => unreachable!(),
    };
    let descriptor = aligned_descriptor(size, ordering, MemoryAccessClass::Exclusive);
    if matches!(instruction, A64MemoryInstruction::LoadExclusive(_)) {
        let value = emit_one(
            builder,
            decoded.location,
            descriptor.value_type(),
            OperationKind::Exclusive(ExclusiveOperation::Load {
                address,
                descriptor,
            }),
        )?;
        write_gpr(
            builder,
            decoded.location,
            fields.rt,
            value.into(),
            Register31::Zero,
        )?;
    } else {
        let value = read_gpr(
            builder,
            decoded.location,
            fields.rt,
            descriptor.value_type(),
            Register31::Zero,
        )?;
        let succeeded = emit_one(
            builder,
            decoded.location,
            IrType::I1,
            OperationKind::Exclusive(ExclusiveOperation::Store {
                address,
                value,
                descriptor,
            }),
        )?;
        let status = scalar(
            builder,
            decoded.location,
            IrType::I32,
            ScalarOperation::Select {
                condition: succeeded.into(),
                when_true: Immediate::I32(0).into(),
                when_false: Immediate::I32(1).into(),
            },
        )?;
        write_gpr(
            builder,
            decoded.location,
            fields.rm,
            status,
            Register31::Zero,
        )?;
    }
    Ok(LiftOutcome::Continue)
}
