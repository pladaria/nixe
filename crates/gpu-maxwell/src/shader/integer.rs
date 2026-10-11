//! Maxwell integer instruction decoding. ISA fields follow the public GM107
//! emitter, not instruction samples:
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2084-L2130
mod conversion;
pub(super) use conversion::{decode_integer_to_integer, is_integer_to_integer};

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

pub(super) const fn is_shift_right(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode & 0xfffe, 0x5c28 | 0x4c28) || opcode & 0xfefe == 0x3828
}

pub(super) fn decode_shift_right(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedShift, MaxwellShaderTranslationError> {
    decode_shift(
        stage,
        offset,
        encoding,
        register_count,
        next_temporary,
        true,
    )
}

pub(super) struct DecodedShift {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_shift_left(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedShift, MaxwellShaderTranslationError> {
    decode_shift(
        stage,
        offset,
        encoding,
        register_count,
        next_temporary,
        false,
    )
}

fn decode_shift(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
    right: bool,
) -> Result<DecodedShift, MaxwellShaderTranslationError> {
    // Operand forms and the wrap-count bit follow Mesa NAK's pinned SM50 SHL
    // encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1695-L1722
    let destination = (encoding & 0xff) as u8;
    let value = ((encoding >> 8) & 0xff) as u8;
    let mut operations = Vec::with_capacity(4);
    let destination = if destination == 255 {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "shift discarded destination overflow",
            next_temporary,
        )?
    } else {
        validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
        ShaderRegister::new(u16::from(destination))
    };
    let value = if value == 255 {
        let zero = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "shift zero source overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: zero,
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        zero
    } else {
        validate_register_range(stage, offset, encoding, value, 1, register_count)?;
        ShaderRegister::new(u16::from(value))
    };
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: if right {
                "SHR condition-code write"
            } else {
                "SHL condition-code write"
            },
        });
    }
    if encoding & (1 << 43) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: if right {
                "SHR extended carry input"
            } else {
                "SHL extended carry input"
            },
        });
    }

    let opcode = ((encoding >> 48) as u16) & if right { 0xfffe } else { 0xffff };
    let (amount, constant_buffer_binding) = if opcode == if right { 0x5c28 } else { 0x5c48 } {
        let amount = ((encoding >> 20) & 0xff) as u8;
        let amount = if amount == 255 {
            let zero = allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "shift zero count overflow",
                next_temporary,
            )?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: zero,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            zero
        } else {
            validate_register_range(stage, offset, encoding, amount, 1, register_count)?;
            ShaderRegister::new(u16::from(amount))
        };
        (amount, None)
    } else {
        let temporary = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "shift operand temporary register overflow",
            next_temporary,
        )?;
        if opcode == if right { 0x4c28 } else { 0x4c48 } {
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
    // SHR's signed and BREV modifiers apply before the shift. Clamped
    // arithmetic counts >=32 retain the sign; wrapped counts use the low 5 bits.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1725
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/integer_shift_right.cpp
    let wrap = encoding & (1 << 39) != 0;
    operations.push(if right {
        ShaderOperation::ShiftRight32 {
            destination,
            value,
            amount,
            wrap,
            signed: encoding & (1 << 48) != 0,
            reverse: encoding & (1 << 40) != 0,
        }
    } else {
        ShaderOperation::ShiftLeft32 {
            destination,
            value,
            amount,
            wrap,
        }
    });
    Ok(DecodedShift {
        operations,
        constant_buffer_binding,
    })
}

pub(super) const fn is_bitwise(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c40
        || opcode & 0xfff8 == 0x4c40
        || opcode & 0xfef8 == 0x3840
        || opcode & 0xfc00 == 0x0400
}

pub(super) struct DecodedIntegerOperation {
    pub operations: Vec<ShaderOperation>,
    pub constant_buffer_binding: Option<u8>,
}

pub(super) const fn is_bitfield_extract(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfffe == 0x5c00 || opcode & 0xfffe == 0x4c00 || opcode & 0xfefe == 0x3800
}

pub(super) fn decode_bitfield_extract(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // Encoding: UAM GM107 emitBFE; boundary/sign extension behavior is
    // documented in PTX and agrees with Eden's Maxwell BFE implementation.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-bfe
    if encoding & (1 << 47) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "BFE condition-code write is not implemented",
        ));
    }
    // Bits 40/48 select bit reversal/sign; bits 39 and 41..46 are reserved.
    if encoding & ((1 << 39) | (0x3f << 41)) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "BFE reserved field is nonzero",
        ));
    }
    let reg = |index: u8| -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        validate_register_range(stage, offset, encoding, index, 1, register_count)?;
        Ok(ShaderRegister::new(u16::from(index)))
    };
    let mut temp =
        || allocate_shader_temporary(stage, offset, encoding, "BFE temporary overflow", next);
    let destination = if encoding as u8 == 255 {
        temp()?
    } else {
        reg(encoding as u8)?
    };
    let value = reg((encoding >> 8) as u8)?;
    let [control, position, count, mask, eight] = [temp()?, temp()?, temp()?, temp()?, temp()?];
    let mut operations = vec![];
    let opcode = (encoding >> 48) as u16;
    let constant_buffer_binding = if opcode & 0xfffe == 0x5c00 {
        operations.push(ShaderOperation::Move32 {
            destination: control,
            source: reg((encoding >> 20) as u8)?,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        None
    } else if opcode & 0xfffe == 0x4c00 {
        let binding = ((encoding >> 34) & 31) as u8;
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination: control,
            binding,
            byte_offset: ((encoding >> 20) & 0x3fff) as u32 * 4,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        Some(binding)
    } else {
        let bits = ((encoding >> 20) & 0x7ffff) as u32
            | if encoding & (1 << 56) != 0 {
                0xfff80000
            } else {
                0
            };
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: control,
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        None
    };
    operations.extend([
        ShaderOperation::MoveImmediate32 {
            destination: mask,
            bits: 255,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::MoveImmediate32 {
            destination: eight,
            bits: 8,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::Bitwise32 {
            destination: position,
            left: control,
            right: mask,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::ShiftRight32 {
            destination: count,
            value: control,
            amount: eight,
            wrap: false,
            signed: false,
            reverse: false,
        },
        ShaderOperation::Bitwise32 {
            destination: count,
            left: count,
            right: mask,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::ExtractBitfield32 {
            destination,
            value,
            offset: position,
            count,
            signed: encoding & (1 << 48) != 0,
            reverse: encoding & (1 << 40) != 0,
        },
    ]);
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

pub(super) const fn is_bitfield_insert(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5bf0 | 0x4bf0) || opcode & 0xfef8 == 0x36f0
}

pub(super) fn decode_bitfield_insert(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // Packed control: low eight bits are the position, next eight the width.
    // Insertion truncates at bit 32; an out-of-range position changes no bits.
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c
    if encoding & (1 << 47) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "BFI condition-code write is not implemented",
        ));
    }
    let mut temp =
        || allocate_shader_temporary(stage, offset, encoding, "BFI temporary overflow", next);
    let [
        zero,
        mask8,
        eight,
        ones,
        control,
        position,
        count,
        shifted_mask,
        inverse_mask,
        shifted_value,
        inserted,
        retained,
    ] = std::array::from_fn::<_, 12, _>(|_| temp())
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .unwrap();
    let mut ops = vec![];
    for (destination, bits) in [(zero, 0), (mask8, 255), (eight, 8), (ones, u32::MAX)] {
        ops.push(ShaderOperation::MoveImmediate32 {
            destination,
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
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
    let destination = if encoding as u8 == 255 {
        temp()?
    } else {
        reg(encoding as u8)?
    };
    let opcode = (encoding >> 48) as u16;
    let mut binding = None;
    if opcode == 0x5bf0 {
        ops.push(ShaderOperation::Move32 {
            destination: control,
            source: reg((encoding >> 20) as u8)?,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    } else if opcode == 0x4bf0 {
        let slot = ((encoding >> 34) & 31) as u8;
        binding = Some(slot);
        ops.push(ShaderOperation::LoadConstantBuffer32 {
            destination: control,
            binding: slot,
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
        ops.push(ShaderOperation::MoveImmediate32 {
            destination: control,
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    ops.extend([
        ShaderOperation::Bitwise32 {
            destination: position,
            left: control,
            right: mask8,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::ShiftRight32 {
            destination: count,
            value: control,
            amount: eight,
            wrap: false,
            signed: false,
            reverse: false,
        },
        ShaderOperation::Bitwise32 {
            destination: count,
            left: count,
            right: mask8,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::ShiftLeft32 {
            destination: shifted_mask,
            value: ones,
            amount: count,
            wrap: false,
        },
        ShaderOperation::Bitwise32 {
            destination: shifted_mask,
            left: shifted_mask,
            right: ones,
            operation: ShaderBitwiseOperation::Xor,
        },
        ShaderOperation::ShiftLeft32 {
            destination: shifted_mask,
            value: shifted_mask,
            amount: position,
            wrap: false,
        },
        ShaderOperation::Bitwise32 {
            destination: inverse_mask,
            left: shifted_mask,
            right: ones,
            operation: ShaderBitwiseOperation::Xor,
        },
        ShaderOperation::ShiftLeft32 {
            destination: shifted_value,
            value: a,
            amount: position,
            wrap: false,
        },
        ShaderOperation::Bitwise32 {
            destination: inserted,
            left: shifted_value,
            right: shifted_mask,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination: retained,
            left: c,
            right: inverse_mask,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination,
            left: retained,
            right: inserted,
            operation: ShaderBitwiseOperation::Or,
        },
    ]);
    Ok(DecodedIntegerOperation {
        operations: ops,
        constant_buffer_binding: binding,
    })
}

pub(super) const fn is_integer_min_max(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    matches!(opcode, 0x5c20 | 0x4c20) || opcode & 0xfef8 == 0x3820
}
pub(super) fn decode_integer_min_max(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c (IMNMX)
    if encoding & ((1 << 47) | (3 << 43)) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "IMNMX condition codes or extended width are not implemented",
        ));
    }
    let mut temp =
        || allocate_shader_temporary(stage, offset, encoding, "IMNMX temporary overflow", next);
    let [zero, ones, b, mask, inverse, selected_a, selected_b] =
        std::array::from_fn::<_, 7, _>(|_| temp())
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap();
    let mut ops = vec![
        ShaderOperation::MoveImmediate32 {
            destination: zero,
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::MoveImmediate32 {
            destination: ones,
            bits: u32::MAX,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    ];
    let reg = |raw: u8| -> Result<_, MaxwellShaderTranslationError> {
        if raw == 255 {
            Ok(zero)
        } else {
            validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
            Ok(ShaderRegister::new(u16::from(raw)))
        }
    };
    let a = reg((encoding >> 8) as u8)?;
    let opcode = (encoding >> 48) as u16;
    let mut binding = None;
    if opcode == 0x5c20 {
        ops.push(ShaderOperation::Move32 {
            destination: b,
            source: reg((encoding >> 20) as u8)?,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    } else if opcode == 0x4c20 {
        let slot = ((encoding >> 34) & 31) as u8;
        binding = Some(slot);
        ops.push(ShaderOperation::LoadConstantBuffer32 {
            destination: b,
            binding: slot,
            byte_offset: ((encoding >> 20) & 0x3fff) as u32 * 4,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    } else {
        ops.push(ShaderOperation::MoveImmediate32 {
            destination: b,
            bits: ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff80000
                } else {
                    0
                },
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    let selector = match decode_predicate_fields(encoding, 39, 42) {
        nixe_gpu::ShaderPredicate::Always => nixe_gpu::ShaderPredicate::Never,
        nixe_gpu::ShaderPredicate::Never => nixe_gpu::ShaderPredicate::Always,
        nixe_gpu::ShaderPredicate::Register { register, inverted } => {
            nixe_gpu::ShaderPredicate::Register {
                register,
                inverted: !inverted,
            }
        }
    };
    let destination = if encoding as u8 == 255 {
        temp()?
    } else {
        reg(encoding as u8)?
    };
    ops.extend([
        ShaderOperation::CompareInteger32 {
            destination: mask,
            float_result: false,
            left: a,
            right: b,
            signed: encoding & (1 << 48) != 0,
            comparison: ShaderIntegerComparison::Less,
            accumulator: selector,
            set_operation: ShaderPredicateSetOperation::Xor,
        },
        ShaderOperation::Bitwise32 {
            destination: inverse,
            left: mask,
            right: ones,
            operation: ShaderBitwiseOperation::Xor,
        },
        ShaderOperation::Bitwise32 {
            destination: selected_a,
            left: a,
            right: mask,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination: selected_b,
            left: b,
            right: inverse,
            operation: ShaderBitwiseOperation::And,
        },
        ShaderOperation::Bitwise32 {
            destination,
            left: selected_a,
            right: selected_b,
            operation: ShaderBitwiseOperation::Or,
        },
    ]);
    Ok(DecodedIntegerOperation {
        operations: ops,
        constant_buffer_binding: binding,
    })
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
    let opcode = (encoding >> 48) as u16;
    let full_immediate = opcode & 0xfc00 == 0x0400;
    if !full_immediate && encoding & (1 << 47) != 0 {
        return Err(error("LOP condition-code write"));
    }
    if !full_immediate && encoding & (1 << 43) != 0 {
        return Err(error("LOP extended condition-code input"));
    }
    if full_immediate && encoding & ((1 << 52) | (1 << 57)) != 0 {
        return Err(error(
            "LOP32I condition codes or extended input are not implemented",
        ));
    }
    let register = opcode & 0xfff8 == 0x5c40;
    let constant = opcode & 0xfff8 == 0x4c40;
    let operand_mask = if full_immediate {
        0xffff_ffff_u64 << 20
    } else if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let allowed = 0xffff_0000_000f_ffff | operand_mask | (0xf << 39) | (3 << 44);
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
    let selector = (encoding >> if full_immediate { 53 } else { 41 }) & 3;
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
            let bits = if full_immediate {
                (encoding >> 20) as u32
            } else {
                ((encoding >> 20) & 0x7ffff) as u32
                    | if encoding & (1 << 56) != 0 {
                        0xfff8_0000
                    } else {
                        0
                    }
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
        .map(|left| {
            complement(
                left,
                encoding & (1 << if full_immediate { 55 } else { 39 }) != 0,
            )
        })
        .transpose()?;
    let right = complement(
        right,
        encoding & (1 << if full_immediate { 56 } else { 40 }) != 0,
    )?;
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
    let predicate_destination = ((encoding >> 48) & 7) as u8;
    if predicate_destination != 7 && !full_immediate {
        // LOP can publish false/true or test the computed word for zero.
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/logic_operation.cpp
        let zero = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "LOP predicate temporary overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: zero,
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        operations.push(ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(predicate_destination), None],
            left: destination,
            right: zero,
            signed: false,
            comparison: match (encoding >> 44) & 3 {
                0 => ShaderIntegerComparison::False,
                1 => ShaderIntegerComparison::True,
                2 => ShaderIntegerComparison::Equal,
                3 => ShaderIntegerComparison::NotEqual,
                _ => unreachable!(),
            },
            accumulator: nixe_gpu::ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        });
    }
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
    opcode & 0xfff8 == 0x5c10
        || opcode & 0xfff8 == 0x4c10
        || opcode & 0xfef8 == 0x3810
        || opcode & 0xfe00 == 0x1c00
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
    let long = (encoding >> 48) & 0xfe00 == 0x1c00;
    if encoding & (1 << if long { 54 } else { 50 }) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "IADD signed saturation",
        });
    }
    let opcode = (encoding >> 48) as u16;
    let register = if shifted {
        opcode & 0xfffc == 0x5c18
    } else {
        opcode & 0xfff8 == 0x5c10
    };
    let constant = if shifted {
        opcode & 0xfffc == 0x4c18
    } else {
        opcode & 0xfff8 == 0x4c10
    };
    let operand_mask = if long {
        u64::from(u32::MAX) << 20
    } else if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let shift_mask = if long {
        (1 << 56) | (1 << 53) | (1 << 52)
    } else if shifted {
        (0x1f << 39) | (1 << 47)
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
    let mut left = operand((encoding >> 8) as u8, &mut operations)?;
    let mut constant_buffer_binding = None;
    let mut right = if register {
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
            let bits = if long {
                (encoding >> 20) as u32
            } else {
                ((encoding >> 20) & 0x7ffff) as u32
                    | if encoding & (1 << 56) != 0 {
                        0xfff8_0000
                    } else {
                        0
                    }
            };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        }
        destination
    };
    let carry_input = !shifted && encoding & (1 << if long { 53 } else { 43 }) != 0;
    let carry_output = encoding & (1 << if long { 52 } else { 47 }) != 0;
    let neg_left = encoding & (1 << if long { 56 } else { 49 }) != 0;
    let neg_right = !long && encoding & (1 << 48) != 0;
    let plus_one = if long {
        (encoding >> 55) & 3 == 3
    } else {
        neg_left && neg_right
    };
    // IADD.PO negates A and adds one; ISCADD.PO and IADD32I.PO keep A.
    // .X uses one's complement instead of arithmetic negation, allowing
    // multiword subtraction with the previous carry as the low +1 bit.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L1291
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/integer_add.cpp
    if !shifted && plus_one && (carry_input || carry_output) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "IADD.PO with condition codes or extended carry",
        });
    }
    for (value, negated) in [
        (&mut left, neg_left && (!plus_one || (!shifted && !long))),
        (&mut right, neg_right && !plus_one),
    ] {
        if negated {
            let ones = temporary()?;
            let adjusted = temporary()?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: ones,
                bits: u32::MAX,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            operations.push(if carry_input {
                ShaderOperation::Bitwise32 {
                    destination: adjusted,
                    left: *value,
                    right: ones,
                    operation: ShaderBitwiseOperation::Xor,
                }
            } else {
                ShaderOperation::Multiply32 {
                    destination: adjusted,
                    left: *value,
                    right: ones,
                    scalar_type: ShaderScalarType::Unsigned32,
                    float_control: ShaderFloatControl::PRECISE,
                }
            });
            *value = adjusted;
        }
    }
    let mut po_carry = None;
    if plus_one {
        let one = temporary()?;
        let adjusted = temporary()?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: one,
            bits: 1,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        if carry_output {
            let carry = temporary()?;
            operations.push(ShaderOperation::AddCarry32 {
                destination: adjusted,
                carry_out: carry,
                left: right,
                right: one,
                carry_in: None,
            });
            po_carry = Some(carry);
        } else {
            operations.push(ShaderOperation::Add32 {
                destination: adjusted,
                left: right,
                right: one,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            });
        }
        right = adjusted;
    }
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
        if let Some(previous) = po_carry {
            operations.push(ShaderOperation::Bitwise32 {
                destination: carry_out,
                left: carry_out,
                right: previous,
                operation: ShaderBitwiseOperation::Or,
            });
        }
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

pub(super) const fn is_set(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff0 == 0x5b50 || opcode & 0xfff0 == 0x4b50 || opcode & 0xfef0 == 0x3650
}

pub(super) fn decode_set(
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
            detail: "ISET.X condition-code input",
        });
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode & 0xfff0 == 0x5b50;
    let constant = opcode & 0xfff0 == 0x4b50;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    if encoding & (1 << 47) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "ISET condition-code write",
        });
    }
    let allowed = 0xffff_0000_000f_ffff_u64 | operand_mask | (0xf << 39) | (1 << 44) | (3 << 45);
    if encoding & !allowed != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "ISET reserved field is nonzero",
        ));
    }
    let raw = encoding as u8;
    let destination = if raw == 255 {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "ISET temporary overflow",
            next_temporary,
        )?
    } else {
        validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
        ShaderRegister::new(u16::from(raw))
    };
    let set_operation = match (encoding >> 45) & 3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "ISET reserved boolean operation",
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
                "ISET zero operand temporary overflow",
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
            "ISET right operand temporary overflow",
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
    operations.push(ShaderOperation::CompareInteger32 {
        destination,
        float_result: encoding & (1 << 44) != 0,
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
