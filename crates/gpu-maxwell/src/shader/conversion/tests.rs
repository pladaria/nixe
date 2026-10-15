use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{translated_fixture_with_register_count, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderOperation, ShaderRoundingMode, ShaderScalarType, lower_shader_ir_to_wgsl};

#[test]
fn captured_signed_i2f_reaches_verified_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_1000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_ff80_2f87_ff00,
            0x5cb8_0000_0007_2a00,
            0xe300_0000_0007_000f,
        ],
        4,
    );

    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::ConvertIntegerToFloat32 {
                    destination,
                    source,
                    source_type: ShaderScalarType::Signed32,
                    rounding: nixe_gpu::ShaderRoundingMode::NearestEven,
                } if destination.index() == 0 && source.index() == 0
            ))
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(
        module
            .source()
            .contains("registers[0] = bitcast<u32>(f32(bitcast<i32>(registers[0])))")
    );
    validate_wgsl(&module);
}

#[test]
fn captured_f2f_floor_ftz_reaches_verified_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_1000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_ff80_2f87_ff00,
            0x5ca8_1480_0007_0a03,
            0xe300_0000_0007_000f,
        ],
        4,
    );

    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::RoundFloat32ToIntegral {
                    destination,
                    source,
                    rounding: ShaderRoundingMode::TowardNegative,
                    flush_denormals_to_zero: true,
                } if destination.index() == 3 && source.index() == 0
            ))
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains(
        "registers[3] = bitcast<u32>(floor(bitcast<f32>(nixe_flush_denormal(registers[0]))))"
    ));
    validate_wgsl(&module);
}

#[test]
fn f2f_absolute_register_and_constant_sources_are_recognized_and_evaluated() {
    let mut header = [0; 20];
    header[0] = 0x0002_5462;
    header[18] = 1;
    for encoding in [0x5caa_1480_0047_0a0a, 0x4caa_1480_0007_0a0a] {
        assert!(is_float_to_float(encoding));
        let translated = translated_fixture_with_register_count(
            MaxwellShaderStage::Pixel,
            header,
            &[
                0,
                0x0100_0000_0007_f004 | (u64::from((-1.75_f32).to_bits()) << 20),
                encoding,
                0x5c98_0780_00a7_0000,
                0,
                0xe300_0000_0007_000f,
                0,
                0,
            ],
            16,
        );
        let inputs = nixe_gpu::ShaderEvaluationInputs::default().with_constant_buffer_bits(
            0,
            0,
            (-1.75_f32).to_bits(),
        );
        let result = nixe_gpu::evaluate_shader_ir(&translated, &inputs, 32).unwrap();
        assert_eq!(
            result.output_bits(nixe_gpu::ShaderIoLocation::Color(0), 0),
            Some(1.0_f32.to_bits())
        );
        validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
    }
}

#[test]
fn captured_f2i_u16_nearest_ftz_reaches_verified_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_1000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_ff80_2f87_ff00,
            0x5cb0_1000_0007_0900,
            0xe300_0000_0007_000f,
        ],
        4,
    );

    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::ConvertFloat32ToInteger {
                    destination,
                    source,
                    destination_type: ShaderScalarType::Unsigned32,
                    destination_bits: 16,
                    rounding: ShaderRoundingMode::NearestEven,
                    flush_denormals_to_zero: true,
                } if destination.index() == 0 && source.index() == 0
            ))
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("nixe_round_ties_even"));
    assert!(module.source().contains("0x0000ffffu"));
    validate_wgsl(&module);
}

#[test]
fn float_to_integer_covers_width_sign_rounding_and_cbuf_forms() {
    for (width_field, expected_bits) in [(0_u64, 8_u8), (1, 16), (2, 32)] {
        for (rounding_field, expected_rounding) in [
            (0_u64, ShaderRoundingMode::NearestEven),
            (1, ShaderRoundingMode::TowardNegative),
            (2, ShaderRoundingMode::TowardPositive),
            (3, ShaderRoundingMode::TowardZero),
        ] {
            let mut next_temporary = 8;
            let encoding =
                0x5cb0_0000_0037_0802 | (width_field << 8) | (rounding_field << 39) | (1 << 12);
            let decoded = decode_float_to_integer(
                MaxwellShaderStage::Vertex,
                16,
                encoding,
                8,
                &mut next_temporary,
            )
            .unwrap();
            assert!(matches!(
                decoded.operations.as_slice(),
                [crate::shader::decoded::Operation::ConvertFloat32ToInteger {
                    destination_type: ShaderScalarType::Signed32,
                    destination_bits,
                    rounding,
                    ..
                }] if *destination_bits == expected_bits && *rounding == expected_rounding
            ));
        }
    }

    let mut next_temporary = 8;
    let constant = decode_float_to_integer(
        MaxwellShaderStage::Vertex,
        16,
        0x4cb0_0088_0037_0902,
        8,
        &mut next_temporary,
    )
    .unwrap();
    assert_eq!(constant.constant_buffer_binding, Some(2));
    assert!(matches!(
        constant.operations.as_slice(),
        [
            crate::shader::decoded::Operation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                ..
            },
            crate::shader::decoded::Operation::ConvertFloat32ToInteger {
                destination_bits: 16,
                rounding: ShaderRoundingMode::TowardNegative,
                ..
            }
        ]
    ));
}

#[test]
fn float_to_float_integral_rounding_covers_directed_modes_and_cbuf() {
    for (field, expected) in [
        (1_u64, ShaderRoundingMode::TowardNegative),
        (2, ShaderRoundingMode::TowardPositive),
        (3, ShaderRoundingMode::TowardZero),
    ] {
        let mut next_temporary = 8;
        let decoded = decode_float_to_float(
            MaxwellShaderStage::Vertex,
            16,
            0x5ca8_0400_0037_0a02 | (field << 39),
            8,
            &mut next_temporary,
        )
        .unwrap();
        assert_eq!(decoded.constant_buffer_binding, None);
        assert!(matches!(
            decoded.operations.last(),
            Some(crate::shader::decoded::Operation::RoundFloat32ToIntegral {
                destination,
                source,
                rounding,
                ..
            }) if destination.index() == 2 && source.index() == 3 && *rounding == expected
        ));
    }

    let mut next_temporary = 8;
    let constant = decode_float_to_float(
        MaxwellShaderStage::Vertex,
        16,
        0x4ca8_0488_0037_0a02,
        8,
        &mut next_temporary,
    )
    .unwrap();
    assert_eq!(constant.constant_buffer_binding, Some(2));
    assert!(matches!(
        constant.operations.as_slice(),
        [
            crate::shader::decoded::Operation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                ..
            },
            crate::shader::decoded::Operation::RoundFloat32ToIntegral {
                rounding: ShaderRoundingMode::TowardNegative,
                ..
            }
        ]
    ));
}

#[test]
fn integer_to_float_decodes_register_immediate_and_constant_buffer_forms() {
    let mut next_temporary = 8;
    let register = decode_integer_to_float(
        MaxwellShaderStage::Vertex,
        16,
        0x5cb8_0000_0037_0a02,
        8,
        &mut next_temporary,
    )
    .unwrap();
    assert_eq!(register.constant_buffer_binding, None);
    assert!(matches!(
        register.operations.as_slice(),
        [crate::shader::decoded::Operation::ConvertIntegerToFloat32 {
            destination,
            source,
            source_type: ShaderScalarType::Unsigned32,
            rounding: nixe_gpu::ShaderRoundingMode::NearestEven,
        }] if destination.index() == 2 && source.index() == 3
    ));

    let immediate = decode_integer_to_float(
        MaxwellShaderStage::Vertex,
        16,
        0x39b8_0000_0037_2a02,
        8,
        &mut next_temporary,
    )
    .unwrap();
    assert_eq!(immediate.constant_buffer_binding, None);
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            crate::shader::decoded::Operation::MoveImmediate32 {
                bits: 0xfff8_0003,
                scalar_type: ShaderScalarType::Signed32,
                ..
            },
            crate::shader::decoded::Operation::ConvertIntegerToFloat32 {
                source_type: ShaderScalarType::Signed32,
                ..
            }
        ]
    ));

    let constant = decode_integer_to_float(
        MaxwellShaderStage::Vertex,
        16,
        0x4cb8_0008_0037_0a02,
        8,
        &mut next_temporary,
    )
    .unwrap();
    assert_eq!(constant.constant_buffer_binding, Some(2));
    assert!(matches!(
        constant.operations.as_slice(),
        [
            crate::shader::decoded::Operation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                scalar_type: ShaderScalarType::Unsigned32,
                ..
            },
            crate::shader::decoded::Operation::ConvertIntegerToFloat32 {
                source_type: ShaderScalarType::Unsigned32,
                ..
            }
        ]
    ));
}

#[test]
fn integer_to_float_rejects_unrepresented_width_and_negation() {
    assert!(matches!(
        decode_integer_to_float(
            MaxwellShaderStage::Vertex,
            16,
            0x5cb8_0000_0037_0a02 | (1 << 45),
            8,
            &mut 8
        ),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "I2F integer source negation",
            ..
        })
    ));
    for (encoding, detail) in [
        (
            0x5cb8_0000_0037_0902,
            "I2F destination width other than F32",
        ),
        (0x5cb8_0000_0037_0e02, "I2F 64-bit source"),
    ] {
        let mut next_temporary = 8;
        assert!(matches!(
            decode_integer_to_float(
                MaxwellShaderStage::Vertex,
                16,
                encoding,
                8,
                &mut next_temporary,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: actual,
                ..
            }) if actual == detail
        ));
    }
}

#[test]
fn integer_to_float_directed_rounding_preserves_inexact_large_values() {
    use nixe_gpu::*;
    for (mode, rounding) in [
        (1, ShaderRoundingMode::TowardNegative),
        (2, ShaderRoundingMode::TowardPositive),
        (3, ShaderRoundingMode::TowardZero),
    ] {
        for (signed, bits, expected) in [
            (
                false,
                u32::MAX,
                if mode == 2 { 0x4f800000 } else { 0x4f7fffff },
            ),
            (
                true,
                0x80000001,
                if mode == 1 { 0xcf000000 } else { 0xceffffff },
            ),
            (true, 0x80000000, 0xcf000000),
            (false, 0, 0),
            (
                false,
                0x01000001,
                if mode == 2 { 0x4b800001 } else { 0x4b800000 },
            ),
        ] {
            let word = 0x5cb8_0000_0007_0a00 | (mode << 39) | (u64::from(signed) << 13);
            let decoded =
                decode_integer_to_float(MaxwellShaderStage::Pixel, 8, word, 1, &mut 1).unwrap();
            assert!(
                matches!(decoded.operations.last(), Some(crate::shader::decoded::Operation::ConvertIntegerToFloat32 { rounding: r, .. }) if *r == rounding)
            );
            let mut operations = vec![ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(0),
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            }];
            operations.extend(decoded.operations.into_iter().map(Into::into));
            operations.push(ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(0)].into(),
                location: ShaderIoLocation::Color(0),
                first_component: 0,
                scalar_type: ShaderScalarType::Float32,
            });
            operations.push(ShaderOperation::Exit);
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
                operations
                    .into_iter()
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
            validate_wgsl(&lower_shader_ir_to_wgsl(&ir).unwrap());
            assert_eq!(
                evaluate_shader_ir(&ir, &ShaderEvaluationInputs::default(), 16)
                    .unwrap()
                    .output_bits(ShaderIoLocation::Color(0), 0),
                Some(expected),
                "{mode} {bits:#x}"
            );
        }
    }
}

#[test]
fn integer_to_float_sources_preserve_selectors_sign_extension_and_absolute_value() {
    use nixe_gpu::*;
    for width in [8_u32, 16, 32] {
        for selector in 0..4_u32 {
            for signed in [false, true] {
                for absolute in [false, true] {
                    let base = 0x5cb8_0000_0007_0200
                        | (u64::from(width.trailing_zeros() - 3) << 10)
                        | (u64::from(selector) << 41)
                        | (u64::from(signed) << 13);
                    if selector * 8 + width > 32 || (width == 16 && selector % 2 != 0) {
                        assert!(
                            decode_integer_to_float(MaxwellShaderStage::Pixel, 8, base, 1, &mut 1)
                                .is_err()
                        );
                        continue;
                    }
                    for input in [
                        0_u32,
                        1,
                        0x7fff_ffff,
                        0x8000_0000,
                        0xffff_8080,
                        0x8000_8000,
                        0x0123_4567,
                        0x89ab_cdef,
                        u32::MAX,
                    ] {
                        for form in [0x5cb8_u64, 0x4cb8, 0x38b8] {
                            let word = (base & !(0xffff << 48))
                                | (form << 48)
                                | (u64::from(absolute) << 49)
                                | if form == 0x4cb8 {
                                    2 << 34
                                } else if form == 0x38b8 {
                                    (u64::from(input & 0x7ffff) << 20)
                                        | (u64::from(input & 0x80000 != 0) << 56)
                                } else {
                                    0
                                };
                            assert!(is_integer_to_float(word));
                            if absolute && !signed && width < 32 {
                                assert!(
                                    decode_integer_to_float(
                                        MaxwellShaderStage::Pixel,
                                        8,
                                        word,
                                        1,
                                        &mut 1
                                    )
                                    .is_err()
                                );
                                continue;
                            }
                            let raw = if form == 0x38b8 {
                                ((input as i32) << 12 >> 12) as u32
                            } else {
                                input
                            };
                            let extracted = raw >> (selector * 8);
                            let expected = if signed {
                                let value = (extracted << (32 - width)) as i32 >> (32 - width);
                                let minimum = if width == 32 {
                                    i32::MIN
                                } else {
                                    -(1_i32 << (width - 1))
                                };
                                (if absolute && value != minimum {
                                    value.abs()
                                } else {
                                    value
                                }) as f32
                            } else {
                                (extracted & (u32::MAX >> (32 - width))) as f32
                            };
                            let decoded = decode_integer_to_float(
                                MaxwellShaderStage::Pixel,
                                8,
                                word,
                                1,
                                &mut 1,
                            )
                            .unwrap();
                            let mut operations = vec![ShaderOperation::MoveImmediate32 {
                                destination: ShaderRegister::new(0),
                                bits: input,
                                scalar_type: ShaderScalarType::Unsigned32,
                            }];
                            operations.extend(decoded.operations.into_iter().map(Into::into));
                            operations.push(ShaderOperation::StoreOutput {
                                sources: vec![ShaderRegister::new(0)].into(),
                                location: ShaderIoLocation::Color(0),
                                first_component: 0,
                                scalar_type: ShaderScalarType::Float32,
                            });
                            operations.push(ShaderOperation::Exit);
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
                                decoded
                                    .constant_buffer_binding
                                    .into_iter()
                                    .map(|binding| {
                                        ShaderResourceAccess::new(
                                            binding,
                                            ShaderResourceKind::ConstantBuffer,
                                            true,
                                            false,
                                        )
                                        .unwrap()
                                    })
                                    .collect(),
                                operations
                                    .into_iter()
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
                            let result = evaluate_shader_ir(
                                &ir,
                                &ShaderEvaluationInputs::default()
                                    .with_constant_buffer_bits(2, 0, input),
                                32,
                            )
                            .unwrap();
                            assert_eq!(
                                result.output_bits(ShaderIoLocation::Color(0), 0),
                                Some(expected.to_bits()),
                                "width={width}, selector={selector}, word={word:016x}, input={input:08x}"
                            );
                            validate_wgsl(&lower_shader_ir_to_wgsl(&ir).unwrap());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn i2f_absolute_captured_vertex_reaches_wgsl() {
    let word = 0x5cba_0000_0197_2a19;
    assert!(is_integer_to_float(word));
    let mut header = [0; 20];
    header[0] = 0x0006_0461;
    header[13] = 0xf << 12;
    let shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[0, word, 0xeff0_7f80_0707_ff19, 0xe300_0000_0007_000f],
        32,
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
}
