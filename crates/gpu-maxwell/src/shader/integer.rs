//! Maxwell integer instruction decoding. ISA fields follow the public GM107
//! emitter, not instruction samples:
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2084-L2130
use super::decode::{allocate_shader_temporary, decode_predicate_fields, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderBitwiseOperation, ShaderFloatControl, ShaderIntegerComparison, ShaderNanMode,
    ShaderOperation, ShaderPredicateSetOperation, ShaderRegister, ShaderRoundingMode,
    ShaderScalarType,
};

pub(super) const fn is_shift_left(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5c48 | 0x4c48) || opcode & 0xfeff == 0x3848
}

pub(super) struct DecodedShiftLeft {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_shift_left(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedShiftLeft, MaxwellShaderTranslationError> {
    // Operand forms and the wrap-count bit follow Mesa NAK's pinned SM50 SHL
    // encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1695-L1722
    let destination = (encoding & 0xff) as u8;
    let value = ((encoding >> 8) & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(stage, offset, encoding, value, 1, register_count)?;
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "SHL condition-code write",
        });
    }
    if encoding & (1 << 43) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "SHL extended carry input",
        });
    }

    let opcode = (encoding >> 48) as u16;
    let mut operations = Vec::with_capacity(2);
    let (amount, constant_buffer_binding) = if opcode == 0x5c48 {
        let amount = ((encoding >> 20) & 0xff) as u8;
        validate_register_range(stage, offset, encoding, amount, 1, register_count)?;
        (ShaderRegister::new(u16::from(amount)), None)
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "SHL operand temporary register overflow",
            next_temporary,
        )?;
        if opcode == 0x4c48 {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            let byte_offset = (((encoding >> 20) & 0x3fff) as u32) * 4;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination: temporary,
                binding,
                byte_offset,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            (temporary, Some(binding))
        } else {
            let low = ((encoding >> 20) & 0x7ffff) as u32;
            let sign = if encoding & (1 << 56) != 0 {
                0xfff8_0000
            } else {
                0
            };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: temporary,
                bits: sign | low,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            (temporary, None)
        }
    };
    operations.push(ShaderOperation::ShiftLeft32 {
        destination: ShaderRegister::new(u16::from(destination)),
        value: ShaderRegister::new(u16::from(value)),
        amount,
        wrap: encoding & (1 << 39) != 0,
    });
    Ok(DecodedShiftLeft {
        operations,
        constant_buffer_binding,
    })
}

pub(super) const fn is_bitwise(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c40 || opcode & 0xfff8 == 0x4c40 || opcode & 0xfef8 == 0x3840
}

pub(super) struct DecodedIntegerOperation {
    pub operations: Vec<ShaderOperation>,
    pub constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_bitwise(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // LOP register/cbuf/signed-20-bit immediate forms and operand complements.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L1568-L1645
    let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
        stage,
        instruction_offset: offset,
        encoding,
        detail,
    };
    if encoding & (1 << 47) != 0 {
        return Err(error("LOP condition-code write"));
    }
    if encoding & (1 << 43) != 0 {
        return Err(error("LOP extended condition-code input"));
    }
    if (encoding >> 48) & 7 != 7 || encoding & (3 << 44) != 0 {
        return Err(error("LOP predicate output"));
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode & 0xfff8 == 0x5c40;
    let constant = opcode & 0xfff8 == 0x4c40;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let allowed = 0xffff_0000_000f_ffff | operand_mask | (0xf << 39);
    if encoding & !allowed != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "LOP reserved field is nonzero",
        ));
    }
    let mut operations = Vec::with_capacity(6);
    let mut temporary = || {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "LOP temporary register overflow",
            next_temporary,
        )
    };
    let destination = encoding as u8;
    let destination = if destination == 0xff {
        temporary()?
    } else {
        validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
        ShaderRegister::new(u16::from(destination))
    };
    let mut operand = |raw: u8,
                       operations: &mut Vec<ShaderOperation>|
     -> Result<_, MaxwellShaderTranslationError> {
        if raw == 0xff {
            let destination = temporary()?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            Ok(destination)
        } else {
            validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
            Ok(ShaderRegister::new(u16::from(raw)))
        }
    };
    let selector = (encoding >> 41) & 3;
    // PASS_B does not read the first operand.
    let left = if selector == 3 {
        None
    } else {
        Some(operand((encoding >> 8) as u8, &mut operations)?)
    };
    let mut constant_buffer_binding = None;
    let right = if register {
        operand((encoding >> 20) as u8, &mut operations)?
    } else {
        let destination = temporary()?;
        if constant {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset: (((encoding >> 20) & 0x3fff) as u32) * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        } else {
            let bits = ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
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
    let mut complement = |value, inverted: bool| -> Result<_, MaxwellShaderTranslationError> {
        if !inverted {
            return Ok(value);
        }
        let mask = temporary()?;
        let result = temporary()?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: mask,
            bits: u32::MAX,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        operations.push(ShaderOperation::Bitwise32 {
            destination: result,
            left: value,
            right: mask,
            operation: ShaderBitwiseOperation::Xor,
        });
        Ok(result)
    };
    let left = left
        .map(|left| complement(left, encoding & (1 << 39) != 0))
        .transpose()?;
    let right = complement(right, encoding & (1 << 40) != 0)?;
    operations.push(if let Some(left) = left {
        ShaderOperation::Bitwise32 {
            destination,
            left,
            right,
            operation: match selector {
                0 => ShaderBitwiseOperation::And,
                1 => ShaderBitwiseOperation::Or,
                2 => ShaderBitwiseOperation::Xor,
                _ => unreachable!(),
            },
        }
    } else {
        ShaderOperation::Move32 {
            destination,
            source: right,
            scalar_type: ShaderScalarType::Unsigned32,
        }
    });
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

pub(super) const fn is_set_predicate(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff0 == 0x5b60 || opcode & 0xfff0 == 0x4b60 || opcode & 0xfef0 == 0x3660
}

pub(super) const fn is_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c10 || opcode & 0xfff8 == 0x4c10 || opcode & 0xfef8 == 0x3810
}

pub(super) const fn is_shift_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfffc == 0x5c18 || opcode & 0xfffc == 0x4c18 || opcode & 0xfefc == 0x3818
}

pub(super) fn decode_add(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
    carry_register: &mut Option<ShaderRegister>,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // IADD and ISCADD share modulo-2^32 operands; ISCADD shifts the first
    // operand by an immediate 0..31 before adding, without a carry input.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L1648-L1685
    let shifted = is_shift_add(encoding);
    for (mask, detail) in [
        (
            if shifted { 1 << 47 } else { 0 },
            if shifted {
                "ISCADD condition-code write"
            } else {
                "IADD condition-code write"
            },
        ),
        (1 << 50, "IADD signed saturation"),
        (
            3 << 48,
            if shifted {
                "ISCADD negated operands"
            } else {
                "IADD negated operands"
            },
        ),
    ] {
        if encoding & mask != 0 {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail,
            });
        }
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode == 0x5c10 || opcode == 0x5c18;
    let constant = opcode == 0x4c10 || opcode == 0x4c18;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let shift_mask = if shifted {
        0x1f << 39
    } else {
        (1 << 47) | (1 << 43)
    };
    if encoding & !(0xffff_0000_000f_ffff | operand_mask | shift_mask) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            if shifted {
                "ISCADD reserved field is nonzero"
            } else {
                "IADD reserved field is nonzero"
            },
        ));
    }
    let mut operations = Vec::with_capacity(3);
    let mut temporary = || {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "IADD temporary register overflow",
            next_temporary,
        )
    };
    let destination = encoding as u8;
    let destination = if destination == 0xff {
        temporary()?
    } else {
        validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
        ShaderRegister::new(u16::from(destination))
    };
    let mut operand = |raw: u8,
                       operations: &mut Vec<ShaderOperation>|
     -> Result<_, MaxwellShaderTranslationError> {
        if raw == 0xff {
            let destination = temporary()?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            Ok(destination)
        } else {
            validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
            Ok(ShaderRegister::new(u16::from(raw)))
        }
    };
    let left = operand((encoding >> 8) as u8, &mut operations)?;
    let mut constant_buffer_binding = None;
    let right = if register {
        operand((encoding >> 20) as u8, &mut operations)?
    } else {
        let destination = temporary()?;
        if constant {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset: (((encoding >> 20) & 0x3fff) as u32) * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        } else {
            let bits = ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
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
    let left = if shifted && (encoding >> 39) & 31 != 0 {
        let amount = temporary()?;
        let destination = temporary()?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: amount,
            bits: ((encoding >> 39) & 31) as u32,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        operations.push(ShaderOperation::ShiftLeft32 {
            destination,
            value: left,
            amount,
            wrap: false,
        });
        destination
    } else {
        left
    };
    let carry_input = !shifted && encoding & (1 << 43) != 0;
    let carry_output = !shifted && encoding & (1 << 47) != 0;
    // IADD.X reads the unsigned carry from the previous CC definition. Do not
    // fabricate reset flags. Predication of the resulting IR preserves the
    // previous carry on a false path, and the verifier checks reaching defs.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp (emitIADD)
    if carry_input || carry_output {
        let carry_in = if carry_input {
            Some(carry_register.ok_or(
                MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage,
                    instruction_offset: offset,
                    encoding,
                    detail: "IADD.X has no translated carry definition",
                },
            )?)
        } else {
            None
        };
        let carry_out = if carry_output {
            if let Some(register) = carry_register {
                *register
            } else {
                let register = temporary()?;
                *carry_register = Some(register);
                register
            }
        } else {
            temporary()?
        };
        operations.push(ShaderOperation::AddCarry32 {
            destination,
            carry_out,
            left,
            right,
            carry_in,
        });
    } else {
        operations.push(ShaderOperation::Add32 {
            destination,
            left,
            right,
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::new(
                ShaderRoundingMode::NearestEven,
                ShaderNanMode::Propagate,
                false,
                false,
                false,
            ),
        });
    }
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_set_predicate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    if encoding & (1 << 43) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "ISETP.X condition-code input",
        });
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode & 0xfff0 == 0x5b60;
    let constant = opcode & 0xfff0 == 0x4b60;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let allowed = 0xffff_0000_000f_ff3f_u64 | operand_mask | (0xf << 39) | (3 << 45);
    if encoding & !allowed != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "ISETP reserved field is nonzero",
        ));
    }
    let destinations = [((encoding >> 3) & 7) as u8, (encoding & 7) as u8]
        .map(|value| (value != 7).then_some(value));
    if destinations[0].is_some() && destinations[0] == destinations[1] {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "ISETP aliased predicate destinations",
        });
    }
    let set_operation = match (encoding >> 45) & 3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "ISETP reserved boolean operation",
            ));
        }
    };
    let comparison = match (encoding >> 49) & 7 {
        0 => ShaderIntegerComparison::False,
        1 => ShaderIntegerComparison::Less,
        2 => ShaderIntegerComparison::Equal,
        3 => ShaderIntegerComparison::LessOrEqual,
        4 => ShaderIntegerComparison::Greater,
        5 => ShaderIntegerComparison::NotEqual,
        6 => ShaderIntegerComparison::GreaterOrEqual,
        7 => ShaderIntegerComparison::True,
        _ => unreachable!(),
    };
    let mut operations = Vec::with_capacity(3);
    let mut source_register = |raw: u8,
                               operations: &mut Vec<ShaderOperation>|
     -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        if raw == 0xff {
            let destination = allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "ISETP zero operand temporary overflow",
                next_temporary,
            )?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            Ok(destination)
        } else {
            validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
            Ok(ShaderRegister::new(u16::from(raw)))
        }
    };
    let left = source_register((encoding >> 8) as u8, &mut operations)?;
    let mut constant_buffer_binding = None;
    let right = if register {
        source_register((encoding >> 20) as u8, &mut operations)?
    } else {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "ISETP right operand temporary overflow",
            next_temporary,
        )?;
        if constant {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset: (((encoding >> 20) & 0x3fff) as u32) * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        } else {
            let bits = ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
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
    operations.push(ShaderOperation::SetPredicateInteger32 {
        destinations,
        left,
        right,
        signed: encoding & (1 << 48) != 0,
        comparison,
        accumulator: decode_predicate_fields(encoding, 39, 42),
        set_operation,
    });
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

#[cfg(test)]
mod tests;
