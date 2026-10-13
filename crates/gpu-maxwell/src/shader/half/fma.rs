//! Half FMA operands are promoted together when an operand selects F32.
use super::*;

pub(in crate::shader) const fn is_half_fused_multiply_add(word: u64) -> bool {
    word & 0xfff8_0000_0000_0000 == 0x5d00_0000_0000_0000
        || matches!(
            word & 0xf880_0000_0000_0000,
            0x7000_0000_0000_0000 | 0x7080_0000_0000_0000 | 0x6080_0000_0000_0000
        )
        || word & 0xfe00_0000_0000_0000 == 0x2800_0000_0000_0000
}

pub(in crate::shader) fn decode_half_fused_multiply_add(
    stage: MaxwellShaderStage,
    offset: u32,
    word: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<DecodedHalfBinary, MaxwellShaderTranslationError> {
    // Encoding, operand promotion, negation and merge modes:
    // https://github.com/envytools/envytools/blob/master/envydis/gm107.c
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_fused_multiply_add.cpp
    let register = word & 0xfff8_0000_0000_0000 == 0x5d00_0000_0000_0000;
    let full = word & 0xfe00_0000_0000_0000 == 0x2800_0000_0000_0000;
    let family = word & 0xf880_0000_0000_0000;
    let cbuf_b = family == 0x7080_0000_0000_0000;
    let cbuf_c = family == 0x6080_0000_0000_0000;
    let precision = (word
        >> if register {
            37
        } else if full {
            55
        } else {
            57
        })
        & 3;
    if precision != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding: word,
            detail: "HFMA2 FTZ/FMZ mode is not implemented",
        });
    }
    let swizzles = [
        (word >> if full { 53 } else { 47 }) & 3,
        if register {
            (word >> 28) & 3
        } else if cbuf_b {
            1
        } else if cbuf_c {
            (word >> 53) & 3
        } else {
            0
        },
        if register {
            (word >> 35) & 3
        } else if cbuf_c {
            1
        } else if full {
            0
        } else {
            (word >> 53) & 3
        },
    ];
    let promoted = swizzles.contains(&1);
    let raw_dst = word as u8;
    validate_register_range(stage, offset, word, raw_dst, 1, register_count)?;
    let destination = ShaderRegister::new(u16::from(raw_dst));
    let mut operations = Vec::new();
    let temporary = |next: &mut u16| {
        allocate_shader_temporary(stage, offset, word, "HFMA2 temporary overflow", next)
    };
    let mut sources = [destination; 3];
    sources[0] = prepare_float_register_source(
        stage,
        offset,
        word,
        (word >> 8) as u8,
        false,
        false,
        register_count,
        next,
        &mut operations,
    )?;
    let constant_buffer_binding = (cbuf_b || cbuf_c).then_some(((word >> 34) & 31) as u8);
    for (index, raw, cbuf) in [
        (1, (word >> if cbuf_c { 39 } else { 20 }) as u8, cbuf_b),
        (2, if full { raw_dst } else { (word >> 39) as u8 }, cbuf_c),
    ] {
        sources[index] = if cbuf {
            let destination = temporary(next)?;
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding: constant_buffer_binding.unwrap(),
                byte_offset: (((word >> 20) & 0x3fff) as u32) * 4,
                scalar_type: ShaderScalarType::Float32,
            });
            destination
        } else if index == 1 && !register && !cbuf_c {
            let destination = temporary(next)?;
            let bits = if full {
                (word >> 20) as u32
            } else {
                (((word >> 20) as u32 & 0x1ff) << 6)
                    | (((word >> 29) as u32 & 1) << 15)
                    | (((word >> 30) as u32 & 0x1ff) << 22)
                    | (((word >> 56) as u32 & 1) << 31)
            };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            destination
        } else {
            prepare_float_register_source(
                stage,
                offset,
                word,
                raw,
                false,
                false,
                register_count,
                next,
                &mut operations,
            )?
        };
    }
    let negations = [
        false,
        if register {
            word & (1 << 31) != 0
        } else {
            (cbuf_b || cbuf_c) && word & (1 << 56) != 0
        },
        word & (1
            << if register {
                30
            } else if full {
                52
            } else {
                51
            })
            != 0,
    ];
    let merge = if full { 0 } else { (word >> 49) & 3 };
    let lanes: &[u8] = match merge {
        0 => &[0, 1],
        3 => &[1],
        _ => &[0],
    };
    let saturate = !full && word & (1 << if register { 32 } else { 52 }) != 0;
    let mut results = Vec::with_capacity(lanes.len());
    for &lane in lanes {
        let mut operands = [destination; 3];
        for index in 0..3 {
            operands[index] = prepare_half_lane(
                stage,
                offset,
                word,
                sources[index],
                swizzles[index],
                lane,
                false,
                negations[index],
                next,
                &mut operations,
            )?;
        }
        let rounded = if promoted {
            let fused = temporary(next)?;
            operations.push(ShaderOperation::FusedMultiplyAdd32 {
                destination: fused,
                left: operands[0],
                right: operands[1],
                addend: operands[2],
                float_control: ShaderFloatControl::new(
                    nixe_gpu::ShaderRoundingMode::NearestEven,
                    nixe_gpu::ShaderNanMode::Propagate,
                    false,
                    false,
                    saturate,
                ),
            });
            let rounded = temporary(next)?;
            operations.push(ShaderOperation::PackHalf32 {
                destination: rounded,
                source: fused,
            });
            rounded
        } else {
            let rounded = temporary(next)?;
            operations.push(ShaderOperation::FusedMultiplyAddHalf32 {
                destination: rounded,
                left: operands[0],
                right: operands[1],
                addend: operands[2],
                saturate,
            });
            rounded
        };
        results.push(rounded);
    }
    merge_half_results(
        stage,
        offset,
        word,
        destination,
        merge,
        &results,
        next,
        &mut operations,
    )?;
    Ok(DecodedHalfBinary {
        operations,
        constant_buffer_binding,
    })
}
