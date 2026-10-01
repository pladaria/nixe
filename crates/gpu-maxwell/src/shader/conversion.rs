//! Maxwell integer/float conversion instructions and rounding modes.

use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use super::float::{apply_float_source_modifiers, prepare_float_register_source};
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderOperation, ShaderRegister, ShaderRoundingMode, ShaderScalarType};

pub(super) const fn is_integer_to_float(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5cb8 | 0x4cb8) || opcode & 0xfeff == 0x38b8
}

pub(super) const fn is_float_to_float(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5ca8 | 0x4ca8) || opcode & 0xfeff == 0x38a8
}

pub(super) const fn is_float_to_integer(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5cb0 | 0x4cb0) || opcode & 0xfeff == 0x38b0
}

pub(super) struct DecodedIntegerToFloat {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedFloatToFloat {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedFloatToInteger {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_float_to_integer(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatToInteger, MaxwellShaderTranslationError> {
    // Operand forms, destination type, rounding, and FTZ follow Mesa NAK's
    // pinned SM50 F2I encoder. Range clamping and NaN/FTZ behavior follow
    // NVIDIA's public PTX conversion semantics:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1802-L1840
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let destination_bits = match (encoding >> 8) & 0x3 {
        0 => 8,
        1 => 16,
        2 => 32,
        3 => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "F2I 64-bit destination",
            });
        }
        _ => unreachable!(),
    };
    if (encoding >> 10) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2I source width other than F32",
        });
    }
    if encoding & (1 << 41) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "F2I F32 source selects half swizzle",
        ));
    }
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2I condition-code output",
        });
    }
    let rounding = match (encoding >> 39) & 0x3 {
        0 => ShaderRoundingMode::NearestEven,
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };
    let destination_type = if encoding & (1 << 12) != 0 {
        ShaderScalarType::Signed32
    } else {
        ShaderScalarType::Unsigned32
    };

    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(4);
    let (source, constant_buffer_binding) = if opcode == 0x5cb0 {
        let source = ((encoding >> 20) & 0xff) as u8;
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                source,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else if opcode == 0x4cb0 {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "F2I operand temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: temporary,
            binding,
            byte_offset,
            scalar_type: ShaderScalarType::Float32,
        });
        let source = apply_float_source_modifiers(
            stage,
            offset,
            encoding,
            temporary,
            encoding & (1 << 49) != 0,
            encoding & (1 << 45) != 0,
            next_temporary,
            &mut operations,
        )?;
        (source, Some(binding))
    } else {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2I immediate operand",
        });
    };
    operations.push(ShaderOperation::ConvertFloat32ToInteger {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        destination_type,
        destination_bits,
        rounding,
        flush_denormals_to_zero: encoding & (1 << 44) != 0,
    });
    Ok(DecodedFloatToInteger {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_float_to_float(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedFloatToFloat, MaxwellShaderTranslationError> {
    // Operand forms and conversion controls follow Mesa NAK's pinned SM50
    // F2F encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1756-L1800
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 8) & 0x3 != 2 || (encoding >> 10) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F width other than F32-to-F32",
        });
    }
    if encoding & (1 << 41) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "F2F F32 source selects half swizzle",
        ));
    }
    if encoding & (1 << 42) == 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F without integral rounding",
        });
    }
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F condition-code output",
        });
    }
    if encoding & (1 << 50) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F saturation",
        });
    }
    let rounding = match (encoding >> 39) & 0x3 {
        0 => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "F2F nearest-even integral rounding",
            });
        }
        1 => ShaderRoundingMode::TowardNegative,
        2 => ShaderRoundingMode::TowardPositive,
        3 => ShaderRoundingMode::TowardZero,
        _ => unreachable!(),
    };

    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(4);
    let (source, constant_buffer_binding) = if opcode == 0x5ca8 {
        let source = ((encoding >> 20) & 0xff) as u8;
        (
            prepare_float_register_source(
                stage,
                offset,
                encoding,
                source,
                encoding & (1 << 49) != 0,
                encoding & (1 << 45) != 0,
                register_count,
                next_temporary,
                &mut operations,
            )?,
            None,
        )
    } else if opcode == 0x4ca8 {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "F2F operand temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: temporary,
            binding,
            byte_offset,
            scalar_type: ShaderScalarType::Float32,
        });
        let source = apply_float_source_modifiers(
            stage,
            offset,
            encoding,
            temporary,
            encoding & (1 << 49) != 0,
            encoding & (1 << 45) != 0,
            next_temporary,
            &mut operations,
        )?;
        (source, Some(binding))
    } else {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "F2F immediate operand",
        });
    };
    operations.push(ShaderOperation::RoundFloat32ToIntegral {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        rounding,
        flush_denormals_to_zero: encoding & (1 << 44) != 0,
    });
    Ok(DecodedFloatToFloat {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_integer_to_float(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerToFloat, MaxwellShaderTranslationError> {
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F condition-code output",
        });
    }
    // Operand forms and type/modifier fields follow Mesa NAK's pinned SM50
    // I2F encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1842-L1880
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 8) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F destination width other than F32",
        });
    }
    if (encoding >> 10) & 0x3 != 2 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F source width other than 32 bits",
        });
    }
    if (encoding >> 39) & 0x3 != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F directed rounding mode",
        });
    }
    if (encoding >> 41) & 0x3 != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "I2F encodes a reserved sub-operation",
        ));
    }
    if encoding & (1 << 45) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F integer source negation",
        });
    }
    if encoding & (1 << 49) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "I2F integer source absolute value",
        });
    }

    let source_type = if encoding & (1 << 13) != 0 {
        ShaderScalarType::Signed32
    } else {
        ShaderScalarType::Unsigned32
    };
    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(2);
    let (source, constant_buffer_binding) = if opcode == 0x5cb8 {
        let source = ((encoding >> 20) & 0xff) as u8;
        validate_register_range(stage, offset, encoding, source, 1, register_count)?;
        (ShaderRegister::new(u16::from(source)), None)
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "I2F operand temporary register overflow",
            next_temporary,
        )?;
        if opcode == 0x4cb8 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: source_type,
            });
            (temporary, Some(binding))
        } else {
            let low = ((encoding >> 20) & 0x7ffff) as u32;
            let bits = low
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits,
                scalar_type: source_type,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::ConvertIntegerToFloat32 {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        source_type,
    });
    Ok(DecodedIntegerToFloat {
        operations,
        constant_buffer_binding,
    })
}

#[cfg(test)]
mod tests;
