//! Three-input integer addition, with unsigned half extraction before negation.
//! https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/integer_add_three_input.cpp
use super::*;

pub(in crate::shader) const fn is_add_three(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode & 0xfff0, 0x5cc0 | 0x4cc0) || opcode & 0xfef0 == 0x38c0
}

pub(in crate::shader) fn decode_add_three(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    if encoding & ((1 << 47) | (1 << 48)) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "IADD3 condition codes or extended carry",
        });
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode & 0xfff0 == 0x5cc0;
    let constant = opcode & 0xfff0 == 0x4cc0;
    let operand_mask = if register {
        (255_u64 << 20) | (255 << 31)
    } else {
        0x7ffff_u64 << 20
    };
    if encoding & !(0xffff_0000_000f_ffff | operand_mask | (255 << 39)) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "IADD3 reserved field is nonzero",
        ));
    }
    let shift = if register { (encoding >> 37) & 3 } else { 0 };
    let halves = if register {
        [
            ((encoding >> 35) & 3),
            ((encoding >> 33) & 3),
            ((encoding >> 31) & 3),
        ]
    } else {
        [0; 3]
    };
    if shift == 3 || halves.contains(&3) {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "IADD3 reserved half or shift selector",
        ));
    }
    let mut temporary =
        || allocate_shader_temporary(stage, offset, encoding, "IADD3 temporary overflow", next);
    let zero = temporary()?;
    let mut operations = vec![ShaderOperation::MoveImmediate32 {
        destination: zero,
        bits: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    }];
    let reg = |raw: u8| -> Result<_, MaxwellShaderTranslationError> {
        if raw == 255 {
            Ok(zero)
        } else {
            validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
            Ok(ShaderRegister::new(u16::from(raw)))
        }
    };
    let a = reg((encoding >> 8) as u8)?;
    let c = reg((encoding >> 39) as u8)?;
    let mut constant_buffer_binding = None;
    let b = if register {
        reg((encoding >> 20) as u8)?
    } else {
        let destination = temporary()?;
        if constant {
            let binding = ((encoding >> 34) & 31) as u8;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset: ((encoding >> 20) & 0x3fff) as u32 * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        } else {
            let bits = ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff80000
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        }
        destination
    };
    let mut operands = [a, b, c];
    let mut sixteen = None;
    let mut ones = None;
    for (index, value) in operands.iter_mut().enumerate() {
        let half = halves[index];
        if half != 0 {
            let count = if let Some(count) = sixteen {
                count
            } else {
                let count = temporary()?;
                sixteen = Some(count);
                operations.push(ShaderOperation::MoveImmediate32 {
                    destination: count,
                    bits: 16,
                    scalar_type: ShaderScalarType::Unsigned32,
                });
                count
            };
            let destination = temporary()?;
            operations.push(ShaderOperation::ExtractBitfield32 {
                destination,
                value: *value,
                offset: if half == 1 { zero } else { count },
                count,
                signed: false,
                reverse: false,
            });
            *value = destination;
        }
        if encoding & (1 << (51 - index)) != 0 {
            let minus_one = if let Some(ones) = ones {
                ones
            } else {
                let ones_reg = temporary()?;
                ones = Some(ones_reg);
                operations.push(ShaderOperation::MoveImmediate32 {
                    destination: ones_reg,
                    bits: u32::MAX,
                    scalar_type: ShaderScalarType::Unsigned32,
                });
                ones_reg
            };
            let destination = temporary()?;
            operations.push(ShaderOperation::Multiply32 {
                destination,
                left: *value,
                right: minus_one,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            });
            *value = destination;
        }
    }
    let sum = temporary()?;
    if shift == 1 {
        // RS shifts the full unsigned 33-bit A+B sum, including its carry.
        let carry = temporary()?;
        operations.push(ShaderOperation::AddCarry32 {
            destination: sum,
            carry_out: carry,
            left: operands[0],
            right: operands[1],
            carry_in: None,
        });
        let count = if let Some(count) = sixteen {
            count
        } else {
            let count = temporary()?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: count,
                bits: 16,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            count
        };
        operations.extend([
            ShaderOperation::ShiftRight32 {
                destination: sum,
                value: sum,
                amount: count,
                signed: false,
                reverse: false,
                wrap: false,
            },
            ShaderOperation::ShiftLeft32 {
                destination: carry,
                value: carry,
                amount: count,
                wrap: false,
            },
            ShaderOperation::Add32 {
                destination: sum,
                left: sum,
                right: carry,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            },
        ]);
    } else {
        operations.push(ShaderOperation::Add32 {
            destination: sum,
            left: operands[0],
            right: operands[1],
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        });
        if shift == 2 {
            let count = if let Some(count) = sixteen {
                count
            } else {
                let count = temporary()?;
                operations.push(ShaderOperation::MoveImmediate32 {
                    destination: count,
                    bits: 16,
                    scalar_type: ShaderScalarType::Unsigned32,
                });
                count
            };
            operations.push(ShaderOperation::ShiftLeft32 {
                destination: sum,
                value: sum,
                amount: count,
                wrap: false,
            });
        }
    }
    let destination = if encoding as u8 == 255 {
        temporary()?
    } else {
        reg(encoding as u8)?
    };
    operations.push(ShaderOperation::Add32 {
        destination,
        left: sum,
        right: operands[2],
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    });
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}
