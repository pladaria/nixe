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

    assert!(matches!(
        translated.ir().instructions()[2].operation(),
        ShaderOperation::ConvertIntegerToFloat32 {
            destination,
            source,
            source_type: ShaderScalarType::Signed32,
        } if destination.index() == 0 && source.index() == 0
    ));
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

    assert!(matches!(
        translated.ir().instructions()[2].operation(),
        ShaderOperation::RoundFloat32ToIntegral {
            destination,
            source,
            rounding: ShaderRoundingMode::TowardNegative,
            flush_denormals_to_zero: true,
        } if destination.index() == 3 && source.index() == 0
    ));
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

    assert!(matches!(
        translated.ir().instructions()[2].operation(),
        ShaderOperation::ConvertFloat32ToInteger {
            destination,
            source,
            destination_type: ShaderScalarType::Unsigned32,
            destination_bits: 16,
            rounding: ShaderRoundingMode::NearestEven,
            flush_denormals_to_zero: true,
        } if destination.index() == 0 && source.index() == 0
    ));
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
                [ShaderOperation::ConvertFloat32ToInteger {
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
            ShaderOperation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                ..
            },
            ShaderOperation::ConvertFloat32ToInteger {
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
            Some(ShaderOperation::RoundFloat32ToIntegral {
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
            ShaderOperation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                ..
            },
            ShaderOperation::RoundFloat32ToIntegral {
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
        [ShaderOperation::ConvertIntegerToFloat32 {
            destination,
            source,
            source_type: ShaderScalarType::Unsigned32,
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
            ShaderOperation::MoveImmediate32 {
                bits: 0xfff8_0003,
                scalar_type: ShaderScalarType::Signed32,
                ..
            },
            ShaderOperation::ConvertIntegerToFloat32 {
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
            ShaderOperation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                scalar_type: ShaderScalarType::Unsigned32,
                ..
            },
            ShaderOperation::ConvertIntegerToFloat32 {
                source_type: ShaderScalarType::Unsigned32,
                ..
            }
        ]
    ));
}

#[test]
fn integer_to_float_rejects_unrepresented_width_rounding_and_modifiers() {
    for (modifier, detail) in [
        (1_u64 << 39, "I2F directed rounding mode"),
        (1_u64 << 45, "I2F integer source negation"),
        (1_u64 << 49, "I2F integer source absolute value"),
    ] {
        let mut next_temporary = 8;
        assert!(matches!(
            decode_integer_to_float(
                MaxwellShaderStage::Vertex,
                16,
                0x5cb8_0000_0037_0a02 | modifier,
                8,
                &mut next_temporary,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: actual,
                ..
            }) if actual == detail
        ));
    }
    for (encoding, detail) in [
        (
            0x5cb8_0000_0037_0902,
            "I2F destination width other than F32",
        ),
        (0x5cb8_0000_0037_0602, "I2F source width other than 32 bits"),
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
