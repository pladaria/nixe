//! Maxwell range reduction and special-function instruction pairs.

use super::decode::{allocate_shader_temporary, decode_predicate, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use super::float::{apply_float_source_modifiers, prepare_float_register_source};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderFloatControl, ShaderMathAccuracy, ShaderNanMode, ShaderOperation, ShaderPredicate,
    ShaderRegister, ShaderRoundingMode, ShaderScalarType, ShaderSourceLocation,
    ShaderSpecialFunction,
};

pub(super) const fn is_mufu(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0x5080
}

pub(super) const fn is_range_reduction(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c90 || opcode & 0xfff8 == 0x4c90 || opcode & 0xfef8 == 0x3890
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaxwellRangeReduction {
    SinCos,
    Exp2,
}

pub(super) struct PendingRangeReduction {
    pub(super) offset: u32,
    pub(super) encoding: u64,
    pub(super) source: ShaderSourceLocation,
    pub(super) predicate: ShaderPredicate,
    destination: u8,
    input: ShaderRegister,
    mode: MaxwellRangeReduction,
    pub(super) preparation: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_range_reduction(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    predicate: ShaderPredicate,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<PendingRangeReduction, MaxwellShaderTranslationError> {
    // RRO source forms, modifiers, and the SINCOS/EX2 selector follow Mesa
    // NAK's pinned SM50 encoder and envytools' pinned public GM107 table:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L708-L741
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c#L2000
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let opcode = (encoding >> 48) as u16;
    let mut preparation = Vec::with_capacity(3);
    let mut constant_buffer_binding = None;
    let input = if opcode & 0xfff8 == 0x5c90 {
        prepare_float_register_source(
            stage,
            offset,
            encoding,
            ((encoding >> 20) & 0xff) as u8,
            encoding & (1 << 49) != 0,
            encoding & (1 << 45) != 0,
            register_count,
            next_temporary,
            &mut preparation,
        )?
    } else {
        let source = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "RRO source temporary register overflow",
            next_temporary,
        )?;
        if opcode & 0xfff8 == 0x4c90 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            constant_buffer_binding = Some(binding);
            preparation.push(ShaderOperation::LoadConstantBuffer32 {
                destination: source,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Float32,
            });
            apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                source,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                next_temporary,
                &mut preparation,
            )?
        } else {
            if encoding & ((1 << 45) | (1 << 49)) != 0 {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "immediate RRO encodes source modifiers",
                ));
            }
            let bits = ((((encoding >> 20) & 0x7ffff) as u32) << 12)
                | if encoding & (1 << 56) != 0 {
                    1 << 31
                } else {
                    0
                };
            preparation.push(ShaderOperation::MoveImmediate32 {
                destination: source,
                bits,
                scalar_type: ShaderScalarType::Float32,
            });
            source
        }
    };

    Ok(PendingRangeReduction {
        offset,
        encoding,
        source: ShaderSourceLocation::new(offset),
        predicate,
        destination,
        input,
        mode: if encoding & (1 << 39) == 0 {
            MaxwellRangeReduction::SinCos
        } else {
            MaxwellRangeReduction::Exp2
        },
        preparation,
        constant_buffer_binding,
    })
}

pub(super) fn is_compatible_mufu(
    range_reduction: &PendingRangeReduction,
    encoding: u64,
    predicate: ShaderPredicate,
) -> bool {
    if !is_mufu(encoding)
        || predicate != range_reduction.predicate
        || ((encoding >> 8) & 0xff) as u8 != range_reduction.destination
        || encoding & ((1 << 46) | (1 << 48)) != 0
    {
        return false;
    }
    matches!(
        (range_reduction.mode, ((encoding >> 20) & 0xf) as u8),
        (MaxwellRangeReduction::Exp2, 2) | (MaxwellRangeReduction::SinCos, 0 | 1)
    )
}

pub(super) fn decode_range_reduced_mufu(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    range_reduction: &PendingRangeReduction,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    if !is_compatible_mufu(range_reduction, encoding, decode_predicate(encoding)) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: range_reduction.offset,
            encoding: range_reduction.encoding,
            detail: "RRO result is not consumed by an adjacent compatible MUFU",
        });
    }
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let function = match ((encoding >> 20) & 0xf) as u8 {
        0 => ShaderSpecialFunction::Cosine,
        1 => ShaderSpecialFunction::Sine,
        2 => ShaderSpecialFunction::Exp2,
        _ => unreachable!("compatibility check bounds the MUFU operation"),
    };
    Ok(ShaderOperation::SpecialFunction32 {
        destination: ShaderRegister::new(u16::from(destination)),
        source: range_reduction.input,
        function,
        accuracy: ShaderMathAccuracy::Approximate,
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            false,
            false,
            false,
        ),
    })
}

pub(super) fn decode_mufu(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let mut operations = Vec::with_capacity(4);
    let source = prepare_float_register_source(
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
    let float_control = ShaderFloatControl::new(
        ShaderRoundingMode::NearestEven,
        ShaderNanMode::Propagate,
        false,
        false,
        false,
    );
    let mufu_operation = ((encoding >> 20) & 0xf) as u8;
    let operation = match mufu_operation {
        0..=3 | 8 => ShaderOperation::SpecialFunction32 {
            destination: ShaderRegister::new(u16::from(destination)),
            source,
            function: match mufu_operation {
                0 => ShaderSpecialFunction::Cosine,
                1 => ShaderSpecialFunction::Sine,
                2 => ShaderSpecialFunction::Exp2,
                3 => ShaderSpecialFunction::Log2,
                8 => ShaderSpecialFunction::SquareRoot,
                _ => unreachable!(),
            },
            accuracy: ShaderMathAccuracy::Approximate,
            float_control,
        },
        4 => ShaderOperation::Reciprocal32 {
            destination: ShaderRegister::new(u16::from(destination)),
            source,
            accuracy: ShaderMathAccuracy::Approximate,
            float_control,
        },
        5 => ShaderOperation::ReciprocalSqrt32 {
            destination: ShaderRegister::new(u16::from(destination)),
            source,
            accuracy: ShaderMathAccuracy::Approximate,
            float_control,
        },
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "64-bit MUFU operation",
            });
        }
    };
    operations.push(operation);
    Ok(operations)
}

#[cfg(test)]
mod tests;
