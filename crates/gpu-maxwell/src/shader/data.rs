//! Maxwell register moves and constant-buffer loads.

use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderOperation, ShaderRegister, ShaderScalarType};

pub(super) const fn is_move_immediate(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfff0 == 0x0100
}

pub(super) const fn is_move(encoding: u64) -> bool {
    matches!((encoding >> 48) as u16, 0x5c98 | 0x4c98)
}

pub(super) const fn is_constant_buffer_load(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfff8 == 0xef90
}

pub(super) struct DecodedMove {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) struct DecodedConstantBufferLoad {
    pub(super) operation: ShaderOperation,
    pub(super) constant_buffer_binding: u8,
}

pub(super) fn decode_constant_buffer_load(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<DecodedConstantBufferLoad, MaxwellShaderTranslationError> {
    // Field locations follow Mesa NAK's pinned SM50 LDC encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L2618-L2649
    let destination = (encoding & 0xff) as u8;
    let dynamic_byte_offset = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(
        stage,
        offset,
        encoding,
        dynamic_byte_offset,
        1,
        register_count,
    )?;
    let memory_type = ((encoding >> 48) & 0x7) as u8;
    if memory_type != 4 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "LDC element width other than B32",
        });
    }
    let address_mode = ((encoding >> 44) & 0x3) as u8;
    if address_mode != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "LDC addressing mode other than indexed",
        });
    }
    let binding = ((encoding >> 36) & 0x1f) as u8;
    let base_byte_offset = ((encoding >> 20) & 0xffff) as u16 as i16 as i32;
    Ok(DecodedConstantBufferLoad {
        operation: ShaderOperation::LoadConstantBufferIndexed32 {
            destination: ShaderRegister::new(u16::from(destination)),
            binding,
            base_byte_offset,
            dynamic_byte_offset: ShaderRegister::new(u16::from(dynamic_byte_offset)),
            scalar_type: ShaderScalarType::Unsigned32,
        },
        constant_buffer_binding: binding,
    })
}

pub(super) fn decode_move_immediate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 12) & 0xf != 0xf {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "MOV32I does not select all quad lanes",
        ));
    }
    Ok(ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister::new(u16::from(destination)),
        bits: ((encoding >> 20) & 0xffff_ffff) as u32,
        scalar_type: ShaderScalarType::Unsigned32,
    })
}

pub(super) fn decode_move(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedMove, MaxwellShaderTranslationError> {
    // Field locations and opcode forms follow Mesa NAK's pinned SM50 MOV
    // encoder: https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1927-L1950
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if (encoding >> 39) & 0xf != 0xf {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "MOV partial quad-lane mask",
        });
    }
    let destination = ShaderRegister::new(u16::from(destination));
    let opcode = (encoding >> 48) as u16;
    if opcode == 0x5c98 {
        let source = ((encoding >> 20) & 0xff) as u8;
        let operation = if source == 0xff {
            ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            }
        } else {
            validate_register_range(stage, offset, encoding, source, 1, register_count)?;
            ShaderOperation::Move32 {
                destination,
                source: ShaderRegister::new(u16::from(source)),
                scalar_type: ShaderScalarType::Unsigned32,
            }
        };
        Ok(DecodedMove {
            operations: vec![operation],
            constant_buffer_binding: None,
        })
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "MOV constant-buffer temporary register overflow",
            next_temporary,
        )?;
        let binding = ((encoding >> 34) & 0x1f) as u8;
        let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
        Ok(DecodedMove {
            operations: vec![
                ShaderOperation::LoadConstantBuffer32 {
                    destination: temporary,
                    binding,
                    byte_offset,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
                ShaderOperation::Move32 {
                    destination,
                    source: temporary,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            ],
            constant_buffer_binding: Some(binding),
        })
    }
}

#[cfg(test)]
mod tests;
