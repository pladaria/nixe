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
        FpSimdInstruction::DuplicateGeneral(_)
        | FpSimdInstruction::DuplicateElement(_)
        | FpSimdInstruction::ModifiedImmediate(_)
        | FpSimdInstruction::InsertElement(_)
        | FpSimdInstruction::InsertGeneral(_)
        | FpSimdInstruction::PermuteTwoSource(_)
        | FpSimdInstruction::Extract(_)
        | FpSimdInstruction::ExtractNarrow(_)
        | FpSimdInstruction::IntegerCompare(_)
        | FpSimdInstruction::IntegerPairwise(_)
        | FpSimdInstruction::IntegerMinMax(_)
        | FpSimdInstruction::VectorSignedShiftRegister(_)
        | FpSimdInstruction::VectorUnsignedShiftRegister(_)
        | FpSimdInstruction::VectorSignedIntToFloat(_)
        | FpSimdInstruction::VectorUnsignedIntToFloat(_)
        | FpSimdInstruction::ScalarVectorSignedIntToFloat(_)
        | FpSimdInstruction::ScalarVectorUnsignedIntToFloat(_)
        | FpSimdInstruction::VectorFloatDivide(_)
        | FpSimdInstruction::VectorFloatImmediate(_)
        | FpSimdInstruction::VectorFloatAbsolute(_)
        | FpSimdInstruction::VectorFloatNegate(_)
        | FpSimdInstruction::ScalarFloatImmediate(_)
        | FpSimdInstruction::ScalarFloatConvert(_)
        | FpSimdInstruction::ScalarFloatDivide(_)
        | FpSimdInstruction::ScalarFloatRound(_)
        | FpSimdInstruction::ScalarFloatAdd(_)
        | FpSimdInstruction::ScalarFloatMultiply(_)
        | FpSimdInstruction::ScalarFloatFusedMultiplyAdd(_)
        | FpSimdInstruction::ScalarFloatSquareRoot(_)
        | FpSimdInstruction::ScalarFloatConditionalSelect(_)
        | FpSimdInstruction::ScalarAbsolute(_)
        | FpSimdInstruction::ScalarNegate(_)
        | FpSimdInstruction::ShiftRightNarrow(_)
        | FpSimdInstruction::ScalarShiftRightImmediate(_)
        | FpSimdInstruction::VectorShiftRightImmediate(_)
        | FpSimdInstruction::ScalarShiftLeftImmediate(_)
        | FpSimdInstruction::VectorShiftLeftImmediate(_)
        | FpSimdInstruction::CountBits(_)
        | FpSimdInstruction::AddAcrossVector(_) => {
            lift_semantic_vector_helper(builder, decoded, fields, instruction)
        }
        FpSimdInstruction::ConditionalCompare(_) => {
            lift_semantic_compare_helper(builder, decoded, fields)
        }
        FpSimdInstruction::UnsignedMoveToGeneral(_) => {
            lift_semantic_general_helper(builder, decoded, fields)
        }
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
        FpSimdInstruction::MemoryPair(_)
        | FpSimdInstruction::MemoryMultipleStructures(_)
        | FpSimdInstruction::MemoryMultipleStructuresPostIndex(_)
        | FpSimdInstruction::MemorySingleStructure(_)
        | FpSimdInstruction::MemorySingleStructurePostIndex(_) => {
            lift_fp_simd_complex_memory(builder, decoded, fields, instruction)
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

fn lift_fp_simd_complex_memory(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
    operation: FpSimdInstruction,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    let base = memory::base_address(builder, source, fields.rn)?;
    let pair = matches!(operation, FpSimdInstruction::MemoryPair(_));
    let multiple = matches!(
        operation,
        FpSimdInstruction::MemoryMultipleStructures(_)
            | FpSimdInstruction::MemoryMultipleStructuresPostIndex(_)
    );
    let post_index = matches!(
        operation,
        FpSimdInstruction::MemoryMultipleStructuresPostIndex(_)
            | FpSimdInstruction::MemorySingleStructurePostIndex(_)
    );
    let offset_register = if !pair && post_index && fields.rm != 31 {
        read_gpr(builder, source, fields.rm, IrType::I64, Register31::Zero)?
    } else {
        Immediate::I64(0).into()
    };
    let register_count = if pair {
        2
    } else if multiple {
        match fields.structure_opcode {
            0b0010 => 4,
            0b0110 => 3,
            0b1010 => 2,
            0b0111 => 1,
            _ => return Ok(unsupported(decoded)),
        }
    } else {
        1
    };
    let writeback = match operation {
        FpSimdInstruction::MemoryPair(_) => matches!(fields.mode, 1 | 3),
        FpSimdInstruction::MemoryMultipleStructuresPostIndex(_)
        | FpSimdInstruction::MemorySingleStructurePostIndex(_) => true,
        _ => false,
    };
    let updated_address = if writeback {
        let offset = if pair {
            let bytes = match fields.size {
                0 => MemoryAccessSize::Word.bytes(),
                1 => MemoryAccessSize::Doubleword.bytes(),
                2 => MemoryAccessSize::Quadword.bytes(),
                _ => return Ok(unsupported(decoded)),
            };
            Immediate::I64((sign_extend(u64::from(fields.immediate_7), 7) * bytes as i64) as u64)
                .into()
        } else if fields.rm != 31 {
            offset_register
        } else if matches!(
            operation,
            FpSimdInstruction::MemoryMultipleStructuresPostIndex(_)
        ) {
            let count = match fields.structure_opcode {
                0b0010 => 4,
                0b0110 => 3,
                0b1010 => 2,
                0b0111 => 1,
                _ => return Ok(unsupported(decoded)),
            };
            let bytes = if fields.vector_128 { 16 } else { 8 };
            Immediate::I64(count * bytes).into()
        } else {
            let bytes = match fields.structure_opcode >> 1 {
                0 => 1,
                2 => 2,
                4 if fields.element_size == 0 => 4,
                4 => 8,
                _ => return Ok(unsupported(decoded)),
            };
            Immediate::I64(bytes).into()
        };
        Some(guest_address_offset(builder, source, base, offset)?)
    } else {
        None
    };
    let mut arguments = Vec::with_capacity(register_count + 6);
    arguments.push(base);
    if pair {
        arguments.push(Immediate::I8(fields.size).into());
        arguments.push(Immediate::I1(fields.load).into());
        arguments.push(Immediate::I8(fields.mode).into());
        arguments.push(Immediate::I8(fields.immediate_7).into());
    } else {
        arguments.push(Immediate::I1(fields.vector_128).into());
        arguments.push(Immediate::I1(fields.load).into());
        arguments.push(Immediate::I8(fields.structure_opcode).into());
        if !multiple {
            arguments.push(Immediate::I8(fields.element_size).into());
        }
    }
    for offset in 0..register_count {
        let register = if pair && offset == 1 {
            fields.rt2
        } else {
            fields.rd.wrapping_add(offset as u8) & 31
        };
        arguments.push(vector_read(builder, source, register)?);
    }

    let mut result_types = Vec::with_capacity(if fields.load { register_count } else { 0 });
    if fields.load {
        result_types.resize(register_count, IrType::V128);
    }
    let results = helper(
        builder,
        source,
        match (pair, multiple) {
            (true, _) => "a64.simd.pair-memory",
            (false, true) => "a64.simd.multiple-structure-memory",
            (false, false) => "a64.simd.single-structure-memory",
        },
        arguments,
        &result_types,
        OperationEffects::new(
            EffectSet::HELPER.union(if fields.load {
                EffectSet::READ_MEMORY
            } else {
                EffectSet::WRITE_MEMORY
            }),
            true,
        ),
    )?;
    if fields.load {
        for (offset, result) in results.iter().take(register_count).enumerate() {
            let register = if pair && offset == 1 {
                fields.rt2
            } else {
                fields.rd.wrapping_add(offset as u8) & 31
            };
            vector_write(builder, source, register, (*result).into())?;
        }
    }

    if let Some(updated_address) = updated_address {
        let updated = guest_address_to_integer(builder, source, updated_address)?;
        write_gpr(
            builder,
            source,
            fields.rn,
            updated,
            Register31::StackPointer,
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_semantic_vector_helper(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
    instruction: FpSimdInstruction,
) -> Result<LiftOutcome, BuildError> {
    let fp = matches!(
        instruction,
        FpSimdInstruction::VectorSignedIntToFloat(_)
            | FpSimdInstruction::VectorUnsignedIntToFloat(_)
            | FpSimdInstruction::ScalarVectorSignedIntToFloat(_)
            | FpSimdInstruction::ScalarVectorUnsignedIntToFloat(_)
            | FpSimdInstruction::VectorFloatDivide(_)
            | FpSimdInstruction::VectorFloatImmediate(_)
            | FpSimdInstruction::VectorFloatAbsolute(_)
            | FpSimdInstruction::VectorFloatNegate(_)
            | FpSimdInstruction::ScalarFloatImmediate(_)
            | FpSimdInstruction::ScalarFloatConvert(_)
            | FpSimdInstruction::ScalarFloatDivide(_)
            | FpSimdInstruction::ScalarFloatRound(_)
            | FpSimdInstruction::ScalarFloatAdd(_)
            | FpSimdInstruction::ScalarFloatMultiply(_)
            | FpSimdInstruction::ScalarFloatFusedMultiplyAdd(_)
            | FpSimdInstruction::ScalarFloatSquareRoot(_)
            | FpSimdInstruction::ScalarFloatConditionalSelect(_)
    );
    let source = decoded.location;
    let fpcr = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpcr),
    )?;
    let fpsr = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpsr),
    )?;
    let rn = vector_read(builder, source, fields.rn)?;
    let rm = vector_read(builder, source, fields.rm)?;
    let ra = vector_read(builder, source, fields.ra)?;
    let rd = vector_read(builder, source, fields.rd)?;
    let general_rn = read_gpr(builder, source, fields.rn, IrType::I64, Register31::Zero)?;
    let flags = read_flags(builder, source)?;
    let results = helper(
        builder,
        source,
        "a64.fp-simd.semantic-vector",
        vec![
            rn,
            rm,
            ra,
            rd,
            general_rn,
            flags,
            fpcr.into(),
            fpsr.into(),
            Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
        ],
        &[IrType::V128, IrType::I32],
        OperationEffects::new(
            if fp {
                EffectSet::HELPER
                    .union(EffectSet::READ_FPCR)
                    .union(EffectSet::WRITE_FPSR)
            } else {
                EffectSet::HELPER
            },
            fp,
        ),
    )?;
    vector_write(builder, source, fields.rd, results[0].into())?;
    if fp {
        builder.emit(
            source,
            &[],
            OperationKind::WriteState {
                register: StateRegister::A64Fpsr,
                value: results[1].into(),
            },
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_semantic_compare_helper(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    let fpcr = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpcr),
    )?;
    let fpsr = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpsr),
    )?;
    let rn = vector_read(builder, source, fields.rn)?;
    let rm = vector_read(builder, source, fields.rm)?;
    let flags = read_flags(builder, source)?;
    let results = helper(
        builder,
        source,
        "a64.fp.semantic-conditional-compare",
        vec![
            rn,
            rm,
            flags,
            fpcr.into(),
            fpsr.into(),
            Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
        ],
        &[IrType::Flags, IrType::I32],
        OperationEffects::new(
            EffectSet::HELPER
                .union(EffectSet::READ_FPCR)
                .union(EffectSet::WRITE_FPSR),
            true,
        ),
    )?;
    write_flags(builder, source, results[0].into())?;
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Fpsr,
            value: results[1].into(),
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_semantic_general_helper(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: FpSimdOperands,
) -> Result<LiftOutcome, BuildError> {
    let rn = vector_read(builder, decoded.location, fields.rn)?;
    let result = helper(
        builder,
        decoded.location,
        "a64.simd.unsigned-move-to-general",
        vec![
            rn,
            Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
        ],
        &[if fields.vector_128 {
            IrType::I64
        } else {
            IrType::I32
        }],
        OperationEffects::new(EffectSet::HELPER, false),
    )?[0];
    write_gpr(
        builder,
        decoded.location,
        fields.rd,
        result.into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
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
        arguments.push(Immediate::I64(fields.helper_token.semantic_abi_value()).into());
        let name = match operation {
            FpSimdInstruction::Bitwise(_) => "a64.simd.bitwise",
            FpSimdInstruction::Integer(_) => "a64.simd.integer-add-sub",
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
            Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
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
        return Ok(unsupported(decoded));
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
                Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
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
        let previous = vector_read(builder, decoded.location, rd)?;
        let result = helper(
            builder,
            decoded.location,
            "a64.fp.move-from-general",
            vec![
                integer,
                previous,
                Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
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
                Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
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
                Immediate::I64(fields.helper_token.semantic_abi_value()).into(),
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
            _ => return Ok(unsupported(decoded)),
        }
    } else if fields.quad {
        MemoryAccessSize::Quadword
    } else {
        crate::semantics::a64::memory_size(fields.size)
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
                return Ok(unsupported(decoded));
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
