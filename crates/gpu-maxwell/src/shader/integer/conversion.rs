//! Integer conversions use scalar bit operations shared by every host backend.
use super::*;
use nixe_gpu::ShaderPredicate;

pub(in crate::shader) const fn is_integer_to_integer(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff9 == 0x5ce0 || opcode & 0xfff9 == 0x4ce0 || opcode & 0xfef9 == 0x38e0
}

pub(in crate::shader) fn decode_integer_to_integer(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
    flags: &mut IntegerConditionCodes,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // Operand fields: Mesa SM50 OpI2I and UAM GM107 emitI2I.
    // Narrow destination values are zero-extended, even for signed destinations;
    // saturation interprets the modified source with its original signedness.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/integer_to_integer_conversion.cpp
    let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
        stage,
        instruction_offset: offset,
        encoding,
        detail,
    };
    let destination_bits = 8_u32 << ((encoding >> 8) & 3);
    let source_bits = 8_u32 << ((encoding >> 10) & 3);
    if destination_bits == 64 || source_bits == 64 {
        return Err(error("I2I 64-bit operand"));
    }
    let selector = ((encoding >> 41) & 7) as u32;
    if selector * 8 + source_bits > 32 || (source_bits == 16 && !selector.is_multiple_of(2)) {
        return Err(error("I2I source selector is incompatible with its width"));
    }
    let source_signed = encoding & (1 << 13) != 0;
    let destination_signed = encoding & (1 << 12) != 0;
    let mut temp =
        || allocate_shader_temporary(stage, offset, encoding, "I2I temporary overflow", next);
    let mut ops = Vec::new();
    let zero = immediate(&mut ops, &mut temp, 0)?;
    let opcode = (encoding >> 48) as u16 & 0xfef9;
    let binding = (opcode == 0x4ce0).then_some(((encoding >> 34) & 31) as u8);
    let source = match opcode {
        0x5ce0 => {
            let raw = (encoding >> 20) as u8;
            if raw == 255 {
                zero
            } else {
                validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
                ShaderRegister::new(u16::from(raw))
            }
        }
        0x4ce0 => {
            let source = temp()?;
            ops.push(ShaderOperation::LoadConstantBuffer32 {
                destination: source,
                binding: binding.unwrap(),
                byte_offset: ((encoding >> 20) & 0x3fff) as u32 * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            source
        }
        _ => immediate(
            &mut ops,
            &mut temp,
            ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
                } else {
                    0
                },
        )?,
    };
    let position = immediate(&mut ops, &mut temp, selector * 8)?;
    let count = immediate(&mut ops, &mut temp, source_bits)?;
    let value = temp()?;
    ops.push(ShaderOperation::ExtractBitfield32 {
        destination: value,
        value: source,
        offset: position,
        count,
        signed: source_signed,
        reverse: false,
    });
    let ones = immediate(&mut ops, &mut temp, u32::MAX)?;
    let one = immediate(&mut ops, &mut temp, 1)?;
    let thirty_one = immediate(&mut ops, &mut temp, 31)?;
    if encoding & (1 << 49) != 0 {
        let sign = temp()?;
        let sign_bit = temp()?;
        ops.extend([
            ShaderOperation::ExtractBitfield32 {
                destination: sign,
                value,
                offset: thirty_one,
                count: one,
                signed: true,
                reverse: false,
            },
            ShaderOperation::Bitwise32 {
                destination: value,
                left: value,
                right: sign,
                operation: ShaderBitwiseOperation::Xor,
            },
            ShaderOperation::Bitwise32 {
                destination: sign_bit,
                left: sign,
                right: one,
                operation: ShaderBitwiseOperation::And,
            },
            ShaderOperation::Add32 {
                destination: value,
                left: value,
                right: sign_bit,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            },
        ]);
    }
    if encoding & (1 << 45) != 0 {
        ops.extend([
            ShaderOperation::Bitwise32 {
                destination: value,
                left: value,
                right: ones,
                operation: ShaderBitwiseOperation::Xor,
            },
            ShaderOperation::Add32 {
                destination: value,
                left: value,
                right: one,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            },
        ]);
    }
    if encoding & (1 << 50) != 0 {
        let min = if source_signed && destination_signed {
            u32::MAX << (destination_bits - 1)
        } else {
            0
        };
        let max = if destination_signed {
            u32::MAX >> (33 - destination_bits)
        } else {
            u32::MAX >> (32 - destination_bits)
        };
        if source_signed && !destination_signed {
            clamp(&mut ops, &mut temp, value, zero, true, true, ones)?;
        }
        for (bits, minimum) in [(min, true), (max, false)] {
            let limit = temp()?;
            ops.push(ShaderOperation::MoveImmediate32 {
                destination: limit,
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            clamp(
                &mut ops,
                &mut temp,
                value,
                limit,
                source_signed && destination_signed,
                minimum,
                ones,
            )?;
        }
    }
    let raw = encoding as u8;
    let destination = if raw == 255 {
        temp()?
    } else {
        validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
        ShaderRegister::new(u16::from(raw))
    };
    // Saturation retains the signed 32-bit clamp result; wrapping conversions
    // truncate to the destination width and zero-extend the stored register.
    if encoding & (1 << 50) != 0 || destination_bits == 32 {
        ops.push(ShaderOperation::Move32 {
            destination,
            source: value,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    } else {
        let bits = temp()?;
        ops.push(ShaderOperation::MoveImmediate32 {
            destination: bits,
            bits: destination_bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        ops.push(ShaderOperation::ExtractBitfield32 {
            destination,
            value,
            offset: zero,
            count: bits,
            signed: false,
            reverse: false,
        });
    }
    if encoding & (1 << 47) != 0 {
        // I2I.CC tests the converted word (even when discarded to RZ),
        // then clears carry/overflow. All operations inherit ISA predication.
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/integer_to_integer_conversion.cpp
        let z = flag_register(&mut flags.zero, &mut temp)?;
        let s = flag_register(&mut flags.sign, &mut temp)?;
        let c = flag_register(&mut flags.carry, &mut temp)?;
        let o = flag_register(&mut flags.overflow, &mut temp)?;
        ops.extend([
            ShaderOperation::CompareInteger32 {
                destination: z,
                float_result: false,
                left: destination,
                right: zero,
                signed: false,
                comparison: ShaderIntegerComparison::Equal,
                accumulator: ShaderPredicate::Always,
                set_operation: ShaderPredicateSetOperation::And,
            },
            ShaderOperation::Bitwise32 {
                destination: z,
                left: z,
                right: one,
                operation: ShaderBitwiseOperation::And,
            },
            ShaderOperation::ExtractBitfield32 {
                destination: s,
                value: destination,
                offset: thirty_one,
                count: one,
                signed: false,
                reverse: false,
            },
            ShaderOperation::Move32 {
                destination: c,
                source: zero,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::Move32 {
                destination: o,
                source: zero,
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ]);
    }
    Ok(DecodedIntegerOperation {
        operations: ops,
        constant_buffer_binding: binding,
    })
}

fn clamp(
    ops: &mut Vec<ShaderOperation>,
    temp: &mut impl FnMut() -> Result<ShaderRegister, MaxwellShaderTranslationError>,
    value: ShaderRegister,
    limit: ShaderRegister,
    signed: bool,
    minimum: bool,
    ones: ShaderRegister,
) -> Result<(), MaxwellShaderTranslationError> {
    let mask = temp()?;
    let inverse = temp()?;
    let a = temp()?;
    let b = temp()?;
    ops.extend([
        ShaderOperation::CompareInteger32 {
            destination: mask,
            float_result: false,
            left: value,
            right: limit,
            signed,
            comparison: if minimum {
                ShaderIntegerComparison::Greater
            } else {
                ShaderIntegerComparison::Less
            },
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination: inverse,
            left: mask,
            right: ones,
            operation: ShaderBitwiseOperation::Xor,
        },
        ShaderOperation::Bitwise32 {
            destination: a,
            left: value,
            right: mask,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination: b,
            left: limit,
            right: inverse,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination: value,
            left: a,
            right: b,
            operation: ShaderBitwiseOperation::Or,
        },
    ]);
    Ok(())
}

fn immediate(
    ops: &mut Vec<ShaderOperation>,
    temp: &mut impl FnMut() -> Result<ShaderRegister, MaxwellShaderTranslationError>,
    bits: u32,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    let destination = temp()?;
    ops.push(ShaderOperation::MoveImmediate32 {
        destination,
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    });
    Ok(destination)
}

fn flag_register(
    flag: &mut Option<ShaderRegister>,
    temp: &mut impl FnMut() -> Result<ShaderRegister, MaxwellShaderTranslationError>,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    if let Some(register) = *flag {
        return Ok(register);
    }
    let register = temp()?;
    *flag = Some(register);
    Ok(register)
}
