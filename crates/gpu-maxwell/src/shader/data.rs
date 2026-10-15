//! Maxwell register moves and constant-buffer loads.

use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use crate::shader::decoded::Operation as ShaderOperation;
use crate::shader::decoded::Register as ShaderRegister;
use nixe_gpu::ShaderScalarType;

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
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: u8,
}

pub(super) fn decode_constant_buffer_load(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u32,
) -> Result<DecodedConstantBufferLoad, MaxwellShaderTranslationError> {
    let ConstantBufferLoadFields {
        destination,
        dynamic_byte_offset,
        words,
        binding,
        base_byte_offset,
    } = decode_constant_buffer_load_fields(stage, offset, encoding, register_count)?;
    if dynamic_byte_offset == 255 && base_byte_offset >= 0 {
        return Ok(DecodedConstantBufferLoad {
            operations: (0..words)
                .map(|word| ShaderOperation::LoadConstantBuffer32 {
                    destination: ShaderRegister::new(
                        u32::from(destination) + u32::from(u16::from(word)),
                    ),
                    binding: binding.into(),
                    byte_offset: base_byte_offset as u32 + u32::from(word) * 4,
                    scalar_type: ShaderScalarType::Unsigned32,
                })
                .collect(),
            constant_buffer_binding: binding,
        });
    }
    let mut operations = Vec::with_capacity(usize::from(words) + 1);
    let mut dynamic_register = ShaderRegister::new(u32::from(dynamic_byte_offset));
    if dynamic_byte_offset == 255 {
        dynamic_register = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "LDC zero address temporary overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: dynamic_register,
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    if words > 1 && (destination..destination + words).contains(&dynamic_byte_offset) {
        // Every word uses the address from before any destination is written.
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "LDC address temporary register overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::Move32 {
            destination: temporary,
            source: dynamic_register,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        dynamic_register = temporary;
    }
    for word in 0..words {
        operations.push(ShaderOperation::LoadConstantBufferIndexed32 {
            destination: ShaderRegister::new(u32::from(destination + word)),
            binding: binding.into(),
            base_byte_offset: base_byte_offset + i32::from(word) * 4,
            dynamic_byte_offset: dynamic_register,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    Ok(DecodedConstantBufferLoad {
        operations,
        constant_buffer_binding: binding,
    })
}

pub(super) fn decode_move_immediate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let (destination, bits) =
        decode_move_immediate_fields(stage, offset, encoding, register_count)?;
    Ok(ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister::new(u32::from(destination)),
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    })
}

// Both compiler paths consume the same checked ISA fields. This decoder does
// not construct the legacy register IR or depend on either host emitter.
pub(super) fn decode_move_immediate_fields(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<(u8, u32), MaxwellShaderTranslationError> {
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
    Ok((destination, ((encoding >> 20) & 0xffff_ffff) as u32))
}

pub(super) fn decode_move(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u32,
) -> Result<DecodedMove, MaxwellShaderTranslationError> {
    let fields = decode_move_fields(stage, offset, encoding, register_count)?;
    let destination = fields.destination;
    let destination = ShaderRegister::new(u32::from(destination));
    match fields.source {
        ScalarSource::Immediate(bits) => Ok(DecodedMove {
            operations: vec![ShaderOperation::MoveImmediate32 {
                destination,
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            }],
            constant_buffer_binding: None,
        }),
        ScalarSource::Register(source) => Ok(DecodedMove {
            operations: vec![ShaderOperation::Move32 {
                destination,
                source: ShaderRegister::new(u32::from(source)),
                scalar_type: ShaderScalarType::Unsigned32,
            }],
            constant_buffer_binding: None,
        }),
        ScalarSource::ConstantBuffer {
            binding,
            byte_offset,
        } => {
            let temporary = allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "MOV constant-buffer temporary register overflow",
                next_temporary,
            )?;
            Ok(DecodedMove {
                operations: vec![
                    ShaderOperation::LoadConstantBuffer32 {
                        destination: temporary,
                        binding: binding.into(),
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
}

#[derive(Clone, Copy)]
pub(super) enum ScalarSource {
    Register(u8),
    Immediate(u32),
    ConstantBuffer { binding: u8, byte_offset: u32 },
}
pub(super) struct MoveFields {
    pub(super) destination: u8,
    pub(super) source: ScalarSource,
}

pub(super) fn decode_move_fields(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<MoveFields, MaxwellShaderTranslationError> {
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
    let source = if (encoding >> 48) as u16 == 0x5c98 {
        let source = ((encoding >> 20) & 0xff) as u8;
        if source == 0xff {
            ScalarSource::Immediate(0)
        } else {
            validate_register_range(stage, offset, encoding, source, 1, register_count)?;
            ScalarSource::Register(source)
        }
    } else {
        ScalarSource::ConstantBuffer {
            binding: ((encoding >> 34) & 0x1f) as u8,
            byte_offset: (((encoding >> 20) & 0x3fff) as u32) * 4,
        }
    };
    Ok(MoveFields {
        destination,
        source,
    })
}

pub(super) struct ConstantBufferLoadFields {
    pub(super) destination: u8,
    pub(super) dynamic_byte_offset: u8,
    pub(super) words: u8,
    pub(super) binding: u8,
    pub(super) base_byte_offset: i32,
}
pub(super) fn decode_constant_buffer_load_fields(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ConstantBufferLoadFields, MaxwellShaderTranslationError> {
    // Field locations follow Mesa NAK's pinned SM50 LDC encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L2618-L2649
    let destination = (encoding & 0xff) as u8;
    let dynamic_byte_offset = ((encoding >> 8) & 0xff) as u8;
    let memory_type = ((encoding >> 48) & 0x7) as u8;
    // Mesa's set_mem_type maps B32/B64/B128 to 4/5/6. Wider loads write
    // consecutive 32-bit GPRs, starting with the least significant word.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L2386-L2399
    let words = match memory_type {
        4 => 1,
        5 => 2,
        6 => 4,
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "LDC element width other than B32/B64/B128",
            });
        }
    };
    validate_register_range(stage, offset, encoding, destination, words, register_count)?;
    if dynamic_byte_offset != 255 {
        validate_register_range(
            stage,
            offset,
            encoding,
            dynamic_byte_offset,
            1,
            register_count,
        )?;
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
    Ok(ConstantBufferLoadFields {
        destination,
        dynamic_byte_offset,
        words,
        binding,
        base_byte_offset,
    })
}

#[cfg(test)]
mod tests;
