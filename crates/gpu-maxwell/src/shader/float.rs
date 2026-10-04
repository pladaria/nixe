//! Maxwell floating-point arithmetic and operand modifiers.

use super::decode::{allocate_shader_temporary, decode_predicate_fields, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderFloatComparison, ShaderFloatControl, ShaderNanMode, ShaderOperation,
    ShaderPredicateSetOperation, ShaderRegister, ShaderRoundingMode, ShaderScalarType,
};

pub(super) const fn is_float_multiply(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfffa == 0x5c68
        || opcode & 0xfffa == 0x4c68
        || opcode & 0xfefa == 0x3868
        || encoding >> 56 == 0x1e
}

pub(super) const fn is_float_min_max(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c60 || opcode & 0xfff8 == 0x4c60 || opcode & 0xfef8 == 0x3860
}

pub(super) const fn is_float_fused_multiply_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode & 0xff80, 0x5980 | 0x4980 | 0x5180) || opcode & 0xfe80 == 0x3280
}

pub(super) const fn is_float_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c58 || opcode & 0xfff8 == 0x4c58 || opcode & 0xfefa == 0x3858
}

pub(super) const fn is_float_set_predicate(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff0 == 0x5bb0 || opcode & 0xfff0 == 0x4bb0 || opcode & 0xfef0 == 0x36b0
}

pub(super) struct DecodedFloatMultiply {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedFloatFusedMultiplyAdd {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedFloatAdd {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedFloatMinMax {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedFloatSetPredicate {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_float_multiply(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatMultiply, MaxwellShaderTranslationError> {
    // SM50 FTZ and DNZ are distinct modes, not independent output/input flags.
    // DNZ implies FTZ and additionally makes +/-0 absorb even Inf/NaN to +0.
    // NAK selects DNZ for fmulz, not ordinary multiplication; FTZ+DNZ is invalid:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/a3fcccb47bfbaf49a5d1ffa56547973462e70ab0/src/nouveau/compiler/nak/from_nir.rs
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul
    // FMUL32I carries every immediate bit at 20..51 and moves the modifier
    // fields above it. Negation is encoded in the immediate's sign bit; there
    // is no rounding or PDIV field in this form.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp (emitFMUL)
    let full_immediate = encoding >> 56 == 0x1e;
    let dnz = encoding & (1 << if full_immediate { 54 } else { 45 }) != 0;
    let ftz = encoding & (1 << if full_immediate { 53 } else { 44 }) != 0;
    if encoding & (1 << if full_immediate { 52 } else { 47 }) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMUL condition-code write",
        });
    }
    if dnz && ftz {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMUL combined FTZ and DNZ modes",
        });
    }
    let destination = (encoding & 0xff) as u8;
    let left = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(stage, offset, encoding, left, 1, register_count)?;
    if !full_immediate && (encoding >> 41) & 0x7 != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "FMUL encodes reserved PDIV bits",
        ));
    }
    if !full_immediate && encoding & (1 << 48) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMUL source negation",
        });
    }
    // SAT clamps the rounded result to [0, 1], including NaN -> +0.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul
    let saturate = encoding & (1 << if full_immediate { 55 } else { 50 }) != 0;
    let rounding = match if full_immediate {
        0
    } else {
        (encoding >> 39) & 0x3
    } {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let float_control = ShaderFloatControl::new(
        rounding,
        ShaderNanMode::Propagate,
        ftz || dnz,
        ftz || dnz,
        saturate,
    );
    let opcode = (encoding >> 48) as u16;
    let (right, preparation, constant_buffer_binding) = if opcode & 0xfffa == 0x5c68 {
        let right = ((encoding >> 20) & 0xff) as u8;
        validate_register_range(stage, offset, encoding, right, 1, register_count)?;
        (ShaderRegister::new(u16::from(right)), None, None)
    } else {
        if *next_temporary >= 256 {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "FMUL temporary register overflow",
            ));
        }
        let temporary = ShaderRegister::new(*next_temporary);
        *next_temporary += 1;
        if opcode & 0xfffa == 0x4c68 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            (
                temporary,
                Some(ShaderOperation::LoadConstantBuffer32 {
                    destination: temporary,
                    binding,
                    byte_offset,
                    scalar_type: ShaderScalarType::Float32,
                }),
                Some(binding),
            )
        } else {
            let bits = if full_immediate {
                (encoding >> 20) as u32
            } else {
                ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                    | if encoding & (1 << 56) != 0 {
                        1 << 31
                    } else {
                        0
                    }
            };
            (
                temporary,
                Some(ShaderOperation::MoveImmediate32 {
                    destination: temporary,
                    bits,
                    scalar_type: ShaderScalarType::Float32,
                }),
                None,
            )
        }
    };
    let mut operations = Vec::with_capacity(2);
    if let Some(preparation) = preparation {
        operations.push(preparation);
    }
    operations.push(if dnz {
        ShaderOperation::FloatMultiplyZero32 {
            destination: ShaderRegister::new(u16::from(destination)),
            left: ShaderRegister::new(u16::from(left)),
            right,
            float_control,
        }
    } else {
        ShaderOperation::Multiply32 {
            destination: ShaderRegister::new(u16::from(destination)),
            left: ShaderRegister::new(u16::from(left)),
            right,
            scalar_type: ShaderScalarType::Float32,
            float_control,
        }
    });
    Ok(DecodedFloatMultiply {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_float_min_max(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatMinMax, MaxwellShaderTranslationError> {
    // Operand forms and modifier fields follow Mesa NAK's pinned SM50 FMNMX
    // encoder and envytools' pinned GM107 disassembler table:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L605-L637
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c#L2005
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FMNMX condition-code output",
        });
    }
    let mut operations = Vec::with_capacity(5);
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 46) != 0,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let opcode = (encoding >> 48) as u16;
    let (right, constant_buffer_binding) = if opcode & 0xfff8 == 0x5c60 {
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                ((encoding >> 20) & 0xff) as u8,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FMNMX temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff8 == 0x4c60 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            let right = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                temporary,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                next_temporary,
                &mut operations,
            )?;
            (right, Some(binding))
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            (temporary, None)
        }
    };
    let ftz = encoding & (1 << 44) != 0;
    operations.push(ShaderOperation::FloatMinMax32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left,
        right,
        minimum: decode_predicate_fields(encoding, 39, 42),
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            ftz,
            ftz,
            false,
        ),
    });
    Ok(DecodedFloatMinMax {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_float_add(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatAdd, MaxwellShaderTranslationError> {
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FADD condition-code output",
        });
    }
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let saturate = encoding & (1 << 50) != 0;
    let rounding = match (encoding >> 39) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let float_control = ShaderFloatControl::new(
        rounding,
        ShaderNanMode::Propagate,
        encoding & (1 << 44) != 0,
        // FADD.FTZ also flushes subnormal inputs, not only the rounded sum.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add
        encoding & (1 << 44) != 0,
        saturate,
    );
    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(5);
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 46) != 0,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let (right, constant_buffer_binding) = if opcode & 0xfff8 == 0x5c58 {
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                ((encoding >> 20) & 0xff) as u8,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FADD temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff8 == 0x4c58 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            let modified = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                temporary,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                next_temporary,
                &mut operations,
            )?;
            (modified, Some(binding))
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::Add32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left,
        right,
        scalar_type: ShaderScalarType::Float32,
        float_control,
    });
    Ok(DecodedFloatAdd {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_float_set_predicate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatSetPredicate, MaxwellShaderTranslationError> {
    // Field locations and opcode forms follow Mesa NAK's pinned SM50 FSETP
    // encoder: https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L866-L898
    if encoding & 0x7 != 0x7 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FSETP secondary predicate destination",
        });
    }
    let destination = ((encoding >> 3) & 0x7) as u8;
    if destination == 7 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FSETP discarded primary predicate destination",
        });
    }
    let comparison = match (encoding >> 48) & 0xf {
        1 => ShaderFloatComparison::OrderedLess,
        2 => ShaderFloatComparison::OrderedEqual,
        3 => ShaderFloatComparison::OrderedLessOrEqual,
        4 => ShaderFloatComparison::OrderedGreater,
        5 => ShaderFloatComparison::OrderedNotEqual,
        6 => ShaderFloatComparison::OrderedGreaterOrEqual,
        7 => ShaderFloatComparison::IsNumber,
        8 => ShaderFloatComparison::IsNan,
        9 => ShaderFloatComparison::UnorderedLess,
        10 => ShaderFloatComparison::UnorderedEqual,
        11 => ShaderFloatComparison::UnorderedLessOrEqual,
        12 => ShaderFloatComparison::UnorderedGreater,
        13 => ShaderFloatComparison::UnorderedNotEqual,
        14 => ShaderFloatComparison::UnorderedGreaterOrEqual,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "invalid FSETP comparison",
            ));
        }
    };
    let set_operation = match (encoding >> 45) & 0x3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "reserved FSETP boolean operation",
            ));
        }
    };
    let accumulator = decode_predicate_fields(encoding, 39, 42);
    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(6);
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        encoding & (1 << 7) != 0,
        encoding & (1 << 43) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let (right, constant_buffer_binding) = if opcode & 0xfff0 == 0x5bb0 {
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                ((encoding >> 20) & 0xff) as u8,
                encoding & (1 << 44) != 0,
                encoding & (1 << 6) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FSETP temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff0 == 0x4bb0 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            let right = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                temporary,
                encoding & (1 << 44) != 0,
                encoding & (1 << 6) != 0,
                next_temporary,
                &mut operations,
            )?;
            (right, Some(binding))
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::SetPredicateFloat32 {
        destination,
        left,
        right,
        comparison,
        accumulator,
        set_operation,
        flush_denormals_to_zero: encoding & (1 << 47) != 0,
    });
    Ok(DecodedFloatSetPredicate {
        operations,
        constant_buffer_binding,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_float_register_source(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    raw: u8,
    absolute: bool,
    negate: bool,
    register_count: u8,
    next_temporary: &mut u16,
    operations: &mut Vec<ShaderOperation>,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    let source = if raw == 0xff {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "floating RZ temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: temporary,
            bits: 0.0_f32.to_bits(),
            scalar_type: ShaderScalarType::Float32,
        });
        temporary
    } else {
        validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
        ShaderRegister::new(u16::from(raw))
    };
    apply_float_source_modifiers(
        stage,
        offset,
        encoding,
        source,
        absolute,
        negate,
        next_temporary,
        operations,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_float_source_modifiers(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    mut source: ShaderRegister,
    absolute: bool,
    negate: bool,
    next_temporary: &mut u16,
    operations: &mut Vec<ShaderOperation>,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    if absolute {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "floating modifier temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::FloatAbsolute32 {
            destination,
            source,
        });
        source = destination;
    }
    if negate {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "floating modifier temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::FloatNegate32 {
            destination,
            source,
        });
        source = destination;
    }
    Ok(source)
}

pub(super) fn decode_float_fused_multiply_add(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatFusedMultiplyAdd, MaxwellShaderTranslationError> {
    // A floating CC write must not leave a preceding integer carry live.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp (emitFFMA)
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FFMA condition-code output",
        });
    }
    // FTZ flushes all three inputs and the fused result. DNZ is a different
    // multiplication mode, not the IR's independent denormals-are-zero flag.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L532-L602
    if encoding & (1 << 54) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FFMA DNZ zero-multiply semantics",
        });
    }
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let saturate = encoding & (1 << 50) != 0;
    let rounding = match (encoding >> 51) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let float_control = ShaderFloatControl::new(
        rounding,
        ShaderNanMode::Propagate,
        encoding & (1 << 53) != 0,
        encoding & (1 << 53) != 0,
        saturate,
    );
    let opcode_class = ((encoding >> 48) as u16) & 0xff80;
    let mut operations = Vec::with_capacity(6);
    let mut constant_buffer_binding = None;
    // SM50 FFMA has no per-source absolute modifiers. Bit 48 negates the
    // multiplication result (equivalently either multiplicand) and bit 49
    // negates src2. Field locations follow Mesa NAK's pinned SM50 encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L532-L602
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        ((encoding >> 8) & 0xff) as u8,
        false,
        encoding & (1 << 48) != 0,
        register_count,
        next_temporary,
        &mut operations,
    )?;

    let (right, raw_addend) = if opcode_class == 0x5980 {
        let right = ((encoding >> 20) & 0xff) as u8;
        let addend = ((encoding >> 39) & 0xff) as u8;
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                right,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                addend,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
        )
    } else if opcode_class == 0x5180 {
        let right = ((encoding >> 39) & 0xff) as u8;
        let addend = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FFMA temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        constant_buffer_binding = Some(binding);
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: addend,
            binding,
            byte_offset,
            scalar_type: ShaderScalarType::Float32,
        });
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                right,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            addend,
        )
    } else {
        let right = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "FFMA temporary register overflow",
            next_temporary,
        )?;
        if opcode_class == 0x4980 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: right,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
        } else {
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: right,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
        }
        let addend = ((encoding >> 39) & 0xff) as u8;
        (
            right,
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                addend,
                false,
                false,
                register_count,
                next_temporary,
                &mut operations,
            )?,
        )
    };
    let addend = apply_float_source_modifiers(
        stage,
        offset,
        encoding,
        raw_addend,
        false,
        encoding & (1 << 49) != 0,
        next_temporary,
        &mut operations,
    )?;
    operations.push(ShaderOperation::FusedMultiplyAdd32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left,
        right,
        addend,
        float_control,
    });
    Ok(DecodedFloatFusedMultiplyAdd {
        operations,
        constant_buffer_binding,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod control_tests;
