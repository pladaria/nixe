//! Packed half arithmetic with explicit operand swizzles and result conversion.
mod fma;
pub(super) use fma::{decode_half_fused_multiply_add, is_half_fused_multiply_add};

use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::MaxwellShaderTranslationError;
use super::float::{apply_float_source_modifiers, prepare_float_register_source};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderBitwiseOperation, ShaderFloatControl, ShaderOperation, ShaderRegister, ShaderScalarType,
};

pub(super) const fn is_half_multiply(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0x5d08_0000_0000_0000
        || encoding & 0xfe80_0000_0000_0000 == 0x7800_0000_0000_0000
        || encoding & 0xfe80_0000_0000_0000 == 0x7880_0000_0000_0000
        || encoding & 0xfe00_0000_0000_0000 == 0x2a00_0000_0000_0000
}

pub(super) const fn is_half_add(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0x5d10_0000_0000_0000
        || encoding & 0xfe80_0000_0000_0000 == 0x7a00_0000_0000_0000
        || encoding & 0xfe80_0000_0000_0000 == 0x7a80_0000_0000_0000
        || encoding & 0xfe00_0000_0000_0000 == 0x2c00_0000_0000_0000
}

pub(super) struct DecodedHalfBinary {
    pub(super) operations: Vec<ShaderOperation>,
    pub(super) constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_half_add(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedHalfBinary, MaxwellShaderTranslationError> {
    decode_half_binary(
        stage,
        offset,
        encoding,
        register_count,
        next_temporary,
        true,
    )
}

pub(super) fn decode_half_multiply(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedHalfBinary, MaxwellShaderTranslationError> {
    decode_half_binary(
        stage,
        offset,
        encoding,
        register_count,
        next_temporary,
        false,
    )
}

fn decode_half_binary(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
    add: bool,
) -> Result<DecodedHalfBinary, MaxwellShaderTranslationError> {
    // HMUL2 register encoding and float/half swizzle/result modes:
    // https://github.com/envytools/envytools/blob/master/envydis/gm107.c#L1133-L1162
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_multiply.cpp
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.cpp
    let immediate32 = encoding & 0xfe00_0000_0000_0000
        == if add {
            0x2c00_0000_0000_0000
        } else {
            0x2a00_0000_0000_0000
        };
    let immediate = encoding & 0xfe80_0000_0000_0000
        == if add {
            0x7a00_0000_0000_0000
        } else {
            0x7800_0000_0000_0000
        };
    let constant_buffer = encoding & 0xfe80_0000_0000_0000
        == if add {
            0x7a80_0000_0000_0000
        } else {
            0x7880_0000_0000_0000
        };
    let precision = (encoding >> if immediate32 { 55 } else { 39 }) & if add { 1 } else { 3 };
    let saturate = encoding
        & (1 << if immediate || immediate32 || constant_buffer {
            52
        } else {
            32
        })
        != 0;
    if precision != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "half arithmetic FTZ/FMZ mode is not implemented",
        });
    }
    let raw_destination = encoding as u8;
    validate_register_range(stage, offset, encoding, raw_destination, 1, register_count)?;
    let destination = ShaderRegister::new(u16::from(raw_destination));
    let mut operations = Vec::new();
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        (encoding >> 8) as u8,
        false,
        false,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let constant_buffer_binding = constant_buffer.then_some(((encoding >> 34) & 0x1f) as u8);
    let right = if let Some(binding) = constant_buffer_binding {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "half constant buffer temporary overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::LoadConstantBuffer32 {
            destination,
            binding,
            byte_offset: (((encoding >> 20) & 0x3fff) as u32) * 4,
            scalar_type: ShaderScalarType::Float32,
        });
        destination
    } else if immediate || immediate32 {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "HADD2 immediate temporary overflow",
            next_temporary,
        )?;
        // The compact immediate stores two signed half values with six zero
        // mantissa bits; HADD2_32I carries the complete packed half pair.
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_add.cpp
        let bits = if immediate32 {
            (encoding >> 20) as u32
        } else {
            (((encoding >> 20) as u32 & 0x1ff) << 6)
                | (((encoding >> 29) as u32 & 1) << 15)
                | (((encoding >> 30) as u32 & 0x1ff) << 22)
                | (((encoding >> 56) as u32 & 1) << 31)
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
            encoding,
            (encoding >> 20) as u8,
            false,
            false,
            register_count,
            next_temporary,
            &mut operations,
        )?
    };
    let temporary = |next: &mut u16| {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "HMUL2 temporary register overflow",
            next,
        )
    };
    let merge = if immediate32 { 0 } else { (encoding >> 49) & 3 };
    let lanes: &[u8] = match merge {
        0 => &[0, 1],
        3 => &[1],
        _ => &[0],
    };
    let mut results = [ShaderRegister::new(0); 2];
    for (result_index, &lane) in lanes.iter().enumerate() {
        let mut operands = [ShaderRegister::new(0); 2];
        for (operand_index, (source, swizzle, absolute, negate)) in [
            (
                left,
                if immediate32 {
                    (encoding >> 53) & 3
                } else {
                    (encoding >> 47) & 3
                },
                !immediate32 && encoding & (1 << 44) != 0,
                (add || immediate || constant_buffer)
                    && encoding & (1 << if immediate32 { 56 } else { 43 }) != 0,
            ),
            (
                right,
                if constant_buffer {
                    1
                } else if immediate || immediate32 {
                    0
                } else {
                    (encoding >> 28) & 3
                },
                !immediate
                    && !immediate32
                    && encoding & (1 << if constant_buffer { 54 } else { 30 }) != 0,
                !immediate
                    && !immediate32
                    && (add || !constant_buffer)
                    && encoding & (1 << if constant_buffer { 56 } else { 31 }) != 0,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            operands[operand_index] = prepare_half_lane(
                stage,
                offset,
                encoding,
                source,
                swizzle,
                lane,
                absolute,
                negate,
                next_temporary,
                &mut operations,
            )?;
        }
        let product = temporary(next_temporary)?;
        operations.push(if add {
            ShaderOperation::Add32 {
                destination: product,
                left: operands[0],
                right: operands[1],
                scalar_type: ShaderScalarType::Float32,
                float_control: ShaderFloatControl::new(
                    nixe_gpu::ShaderRoundingMode::NearestEven,
                    nixe_gpu::ShaderNanMode::Propagate,
                    false,
                    false,
                    saturate,
                ),
            }
        } else {
            ShaderOperation::Multiply32 {
                destination: product,
                left: operands[0],
                right: operands[1],
                scalar_type: ShaderScalarType::Float32,
                float_control: ShaderFloatControl::new(
                    nixe_gpu::ShaderRoundingMode::NearestEven,
                    nixe_gpu::ShaderNanMode::Propagate,
                    false,
                    false,
                    saturate,
                ),
            }
        });
        let rounded = temporary(next_temporary)?;
        // Two binary16 significands have an exact binary32 product; mixed F32
        // input uses F32 multiplication before the same half result rounding.
        operations.push(ShaderOperation::PackHalf32 {
            destination: rounded,
            source: product,
        });
        results[result_index] = rounded;
    }
    merge_half_results(
        stage,
        offset,
        encoding,
        destination,
        merge,
        &results[..lanes.len()],
        next_temporary,
        &mut operations,
    )?;
    Ok(DecodedHalfBinary {
        operations,
        constant_buffer_binding,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_half_lane(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    source: ShaderRegister,
    swizzle: u64,
    lane: u8,
    absolute: bool,
    negate: bool,
    next: &mut u16,
    operations: &mut Vec<ShaderOperation>,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    let value = if swizzle == 1 {
        source
    } else {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "half operand temporary overflow",
            next,
        )?;
        operations.push(ShaderOperation::UnpackHalf32 {
            destination,
            source,
            high: match swizzle {
                0 => lane == 1,
                2 => false,
                3 => true,
                _ => unreachable!(),
            },
        });
        destination
    };
    apply_float_source_modifiers(
        stage, offset, encoding, value, absolute, negate, next, operations,
    )
}

#[allow(clippy::too_many_arguments)]
fn merge_half_results(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    destination: ShaderRegister,
    merge: u64,
    results: &[ShaderRegister],
    next_temporary: &mut u16,
    operations: &mut Vec<ShaderOperation>,
) -> Result<(), MaxwellShaderTranslationError> {
    let temporary = |next: &mut u16| {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "half merge temporary overflow",
            next,
        )
    };
    if merge == 1 {
        operations.push(ShaderOperation::UnpackHalf32 {
            destination,
            source: results[0],
            high: false,
        });
    } else {
        let shift = temporary(next_temporary)?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: shift,
            bits: 16,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        let high = temporary(next_temporary)?;
        if merge == 0 || merge == 3 {
            operations.push(ShaderOperation::ShiftLeft32 {
                destination: high,
                value: results[results.len() - 1],
                amount: shift,
                wrap: false,
            });
        }
        let (low, high) = if merge == 0 {
            (results[0], high)
        } else {
            let mask = temporary(next_temporary)?;
            let preserved = temporary(next_temporary)?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: mask,
                bits: if merge == 2 { 0xffff_0000 } else { 0xffff },
                scalar_type: ShaderScalarType::Unsigned32,
            });
            operations.push(ShaderOperation::Bitwise32 {
                destination: preserved,
                left: destination,
                right: mask,
                operation: ShaderBitwiseOperation::And,
            });
            if merge == 2 {
                (results[0], preserved)
            } else {
                (preserved, high)
            }
        };
        operations.push(ShaderOperation::Bitwise32 {
            destination,
            left: low,
            right: high,
            operation: ShaderBitwiseOperation::Or,
        });
    }
    Ok(())
}

pub(super) const fn is_half_set_predicate(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0x5d20_0000_0000_0000
}

pub(super) fn decode_half_set_predicate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    use nixe_gpu::{ShaderFloatComparison as C, ShaderPredicate, ShaderPredicateSetOperation};
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_set_predicate.cpp
    let destinations = [((encoding >> 3) & 7) as u8, (encoding & 7) as u8];
    if encoding & ((1 << 49) | (1 << 6)) != 0
        || (destinations[0] == destinations[1] && destinations[0] != 7)
    {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "HSETP2 lane conjunction, half FTZ or aliased predicate destinations are not implemented",
        });
    }
    let comparison = match (encoding >> 35) & 15 {
        1 => C::OrderedLess,
        2 => C::OrderedEqual,
        3 => C::OrderedLessOrEqual,
        4 => C::OrderedGreater,
        5 => C::OrderedNotEqual,
        6 => C::OrderedGreaterOrEqual,
        7 => C::IsNumber,
        8 => C::IsNan,
        9 => C::UnorderedLess,
        10 => C::UnorderedEqual,
        11 => C::UnorderedLessOrEqual,
        12 => C::UnorderedGreater,
        13 => C::UnorderedNotEqual,
        14 => C::UnorderedGreaterOrEqual,
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "HSETP2 constant comparison is not implemented",
            });
        }
    };
    let set_operation = match (encoding >> 45) & 3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(super::error::malformed(
                stage,
                offset,
                encoding,
                "HSETP2 reserved boolean operation",
            ));
        }
    };
    let accumulator = super::decode::decode_predicate_fields(encoding, 39, 42);
    let mut operations = Vec::new();
    let mut operands = [[ShaderRegister::new(0); 2]; 2];
    for (lane, lane_operands) in operands.iter_mut().enumerate() {
        for (index, (raw, swizzle, absolute, negate)) in [
            (
                (encoding >> 8) as u8,
                (encoding >> 47) & 3,
                encoding & (1 << 44) != 0,
                encoding & (1 << 43) != 0,
            ),
            (
                (encoding >> 20) as u8,
                (encoding >> 28) & 3,
                encoding & (1 << 30) != 0,
                encoding & (1 << 31) != 0,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let source = prepare_float_register_source(
                stage,
                offset,
                encoding,
                raw,
                false,
                false,
                register_count,
                next,
                &mut operations,
            )?;
            let value = if swizzle == 1 {
                source
            } else {
                let destination = allocate_shader_temporary(
                    stage,
                    offset,
                    encoding,
                    "HSETP2 temporary overflow",
                    next,
                )?;
                operations.push(ShaderOperation::UnpackHalf32 {
                    destination,
                    source,
                    high: match swizzle {
                        0 => lane == 1,
                        2 => false,
                        3 => true,
                        _ => unreachable!(),
                    },
                });
                destination
            };
            lane_operands[index] = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                value,
                absolute,
                negate,
                next,
                &mut operations,
            )?;
        }
    }
    // Freeze both comparisons before publishing either predicate. This also
    // preserves the instruction guard when a destination aliases that guard.
    let mut results = [ShaderRegister::new(0); 2];
    for lane in 0..2 {
        results[lane] = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "HSETP2 result temporary overflow",
            next,
        )?;
        operations.push(ShaderOperation::CompareFloat32 {
            destination: results[lane],
            float_result: false,
            left: operands[lane][0],
            right: operands[lane][1],
            comparison,
            accumulator,
            set_operation,
            flush_denormals_to_zero: false,
        });
    }
    let zero = allocate_shader_temporary(
        stage,
        offset,
        encoding,
        "HSETP2 zero temporary overflow",
        next,
    )?;
    operations.push(ShaderOperation::MoveImmediate32 {
        destination: zero,
        bits: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    });
    let guard = super::decode::decode_predicate(encoding);
    let order = if matches!(guard, ShaderPredicate::Register {register,..} if register == destinations[0])
    {
        [1, 0]
    } else {
        [0, 1]
    };
    for lane in order {
        if destinations[lane] == 7 {
            continue;
        }
        operations.push(ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(destinations[lane]), None],
            left: results[lane],
            right: zero,
            signed: false,
            comparison: nixe_gpu::ShaderIntegerComparison::NotEqual,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        });
    }
    Ok(operations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{
        ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderPredicate,
        ShaderSourceLocation, ShaderStage, VerifiedShaderIr,
    };

    #[test]
    fn constant_buffer_half_arithmetic_reads_f32_and_preserves_merged_lane() {
        for (word, input, constant, expected, add) in [
            (
                0x7a85_0804_0027_ff00,
                0x4000_0000,
                1.25_f32,
                0x4000_3d00,
                true,
            ),
            (
                0x78c5_1804_0027_0000,
                0x4000_bc00,
                -2.0_f32,
                0x4000_c000,
                false,
            ),
        ] {
            let decode = if add {
                decode_half_add
            } else {
                decode_half_multiply
            };
            let decoded = decode(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4).unwrap();
            assert_eq!(decoded.constant_buffer_binding, Some(1));
            let mut ops = vec![ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(0),
                bits: input,
                scalar_type: ShaderScalarType::Unsigned32,
            }];
            ops.extend(decoded.operations);
            ops.push(ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(0)].into(),
                location: ShaderIoLocation::Color(0),
                first_component: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            ops.push(ShaderOperation::Exit);
            let ir = VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Fragment,
                vec![],
                vec![
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        0,
                        ShaderScalarType::Unsigned32,
                        None,
                    )
                    .unwrap(),
                ],
                vec![
                    nixe_gpu::ShaderResourceAccess::new(
                        1,
                        nixe_gpu::ShaderResourceKind::ConstantBuffer,
                        true,
                        false,
                    )
                    .unwrap(),
                ],
                ops.into_iter()
                    .enumerate()
                    .map(|(i, op)| {
                        ShaderInstruction::new(
                            ShaderSourceLocation::new(i as u32 * 8),
                            ShaderPredicate::Always,
                            op,
                        )
                    })
                    .collect(),
            ))
            .unwrap();
            let result = nixe_gpu::evaluate_shader_ir(
                &ir,
                &nixe_gpu::ShaderEvaluationInputs::default().with_constant_buffer_bits(
                    1,
                    8,
                    constant.to_bits(),
                ),
                128,
            )
            .unwrap();
            assert_eq!(
                result.output_bits(ShaderIoLocation::Color(0), 0),
                Some(expected)
            );
            super::super::test_support::validate_wgsl(
                &nixe_gpu::lower_shader_ir_to_wgsl(&ir).unwrap(),
            );

            // The instruction translator must declare the consumed uniform binding.
            let mut header = [0; 20];
            header[0] = 0x0002_0461;
            header[4] = 0x000f_f000;
            header[6] = 0x0000_0077;
            header[13] = 0x0007_f000;
            let translated = super::super::test_support::translated_fixture(
                MaxwellShaderStage::Vertex,
                header,
                &[0, word, 0xe300_0000_0007_000f],
            );
            assert_eq!(translated.ir().resources(), ir.ir().resources());
            super::super::test_support::validate_wgsl(
                &nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap(),
            );
        }
    }
    #[test]
    fn hmul2_float_output_retains_half_result_rounding_and_operand_aliasing() {
        let mut code = vec![
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(1),
                bits: 1.23456_f32.to_bits(),
                scalar_type: ShaderScalarType::Float32,
            },
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(4),
                bits: 0xbe00_3a00,
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ];
        // F32 left, broadcast low half right, F32 result; destination aliases left.
        code.extend(
            decode_half_multiply(
                MaxwellShaderStage::Pixel,
                8,
                0x5d0a_8000_2047_0101,
                5,
                &mut 5,
            )
            .unwrap()
            .operations,
        );
        code.push(ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister::new(1)].into(),
            location: ShaderIoLocation::Color(0),
            first_component: 0,
            scalar_type: ShaderScalarType::Float32,
        });
        code.push(ShaderOperation::Exit);
        let ir = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            Vec::new(),
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Color(0),
                    0,
                    ShaderScalarType::Float32,
                    None,
                )
                .unwrap(),
            ],
            Vec::new(),
            code.into_iter()
                .enumerate()
                .map(|(i, op)| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(i as u32 * 8),
                        ShaderPredicate::Always,
                        op,
                    )
                })
                .collect(),
        ))
        .unwrap();
        let result =
            nixe_gpu::evaluate_shader_ir(&ir, &nixe_gpu::ShaderEvaluationInputs::default(), 64)
                .unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Color(0), 0),
            Some(0x3f6d_0000)
        );
        let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(&ir).unwrap();
        super::super::test_support::validate_wgsl(&wgsl);
        assert!(
            decode_half_multiply(
                MaxwellShaderStage::Pixel,
                8,
                0x5d0a_8080_2047_0101,
                5,
                &mut 5
            )
            .is_err()
        );
    }
    #[test]
    fn saturated_half_add_preserves_clamp_nan_infinity_and_half_rounding() {
        // Captured compact-immediate HADD2.SAT.F32: 1.0 - R7, F32 output.
        let word = 0x7a12883c0f070711;
        for (input, expected) in [
            (-2.0, 1.0),
            (2.0, 0.0),
            (0.25, 0.75),
            (f32::NAN, 0.0),
            (f32::INFINITY, 0.0),
            (f32::NEG_INFINITY, 1.0),
            (0.33333, f32::from_bits(0x3f2aa000)),
        ] {
            let mut ops = vec![ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(7),
                bits: input.to_bits(),
                scalar_type: ShaderScalarType::Float32,
            }];
            ops.extend(
                decode_half_add(MaxwellShaderStage::Pixel, 8, word, 32, &mut 32)
                    .unwrap()
                    .operations,
            );
            ops.push(ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(17)].into_boxed_slice(),
                location: ShaderIoLocation::Color(0),
                first_component: 0,
                scalar_type: ShaderScalarType::Float32,
            });
            ops.push(ShaderOperation::Exit);
            let ir = VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Fragment,
                vec![],
                vec![
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        0,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap(),
                ],
                vec![],
                ops.into_iter()
                    .enumerate()
                    .map(|(i, op)| {
                        ShaderInstruction::new(
                            ShaderSourceLocation::new(i as u32 * 8),
                            ShaderPredicate::Always,
                            op,
                        )
                    })
                    .collect(),
            ))
            .unwrap();
            let result = nixe_gpu::evaluate_shader_ir(
                &ir,
                &nixe_gpu::ShaderEvaluationInputs::default(),
                128,
            )
            .unwrap();
            assert_eq!(
                result.output_bits(ShaderIoLocation::Color(0), 0),
                Some(expected.to_bits()),
                "input={input}"
            );
            super::super::test_support::validate_wgsl(
                &nixe_gpu::lower_shader_ir_to_wgsl(&ir).unwrap(),
            );
        }
    }
}
