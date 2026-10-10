use super::super::decode::decode_predicate;
use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{
    translated_fixture, translated_fixture_with_register_count, validate_wgsl,
};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderOperation, ShaderPredicate, ShaderResourceKind, ShaderScalarType, lower_shader_ir_to_wgsl,
};

#[test]
fn captured_indexed_constant_buffer_load_reaches_verified_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_8000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_7f80_2fc7_ff00,
            0x3848_0000_0047_0007,
            0xef94_0010_0307_0700,
            0,
            0xe300_0000_0007_000f,
            0,
            0,
        ],
        8,
    );

    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 1 && resource.kind() == ShaderResourceKind::ConstantBuffer
    }));
    assert!(matches!(
        translated.ir().instructions()[3].operation(),
        ShaderOperation::LoadConstantBufferIndexed32 {
            destination,
            binding: 1,
            base_byte_offset: 0x30,
            dynamic_byte_offset,
            scalar_type: ShaderScalarType::Unsigned32,
        } if destination.index() == 0 && dynamic_byte_offset.index() == 7
    ));
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(
        module
            .source()
            .contains("constant_buffer_1[(registers[7] + 0x00000030u) >> 2u]")
    );
    validate_wgsl(&module);
}

#[test]
fn indexed_constant_buffer_load_rejects_unrepresented_widths_and_modes() {
    let captured = 0xef94_0010_0307_0700_u64;
    assert!(matches!(
        decode_constant_buffer_load(
            MaxwellShaderStage::Vertex,
            24,
            (captured & !(0x7 << 48)) | (0x2 << 48),
            8,
            &mut 8,
        ),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "LDC element width other than B32/B64/B128",
            ..
        })
    ));
    assert!(matches!(
        decode_constant_buffer_load(
            MaxwellShaderStage::Vertex,
            24,
            captured | (1 << 44),
            8,
            &mut 8
        ),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "LDC addressing mode other than indexed",
            ..
        })
    ));
}

fn move_bits(destination: u8, bits: u32) -> u64 {
    0x0100_0000_0007_f000 | (u64::from(bits) << 20) | u64::from(destination)
}

fn pixel_header(mask: u32) -> [u32; 20] {
    let mut header = [0; 20];
    header[0] = 0x0002_5462;
    header[18] = mask;
    header
}

fn spirv_options() -> nixe_gpu::SpirvShaderOptions {
    nixe_gpu::SpirvShaderOptions {
        depth_clip_negative_one_to_one: false,
        input_control_points: 0,
        tessellation_mode: None,
        float32: Default::default(),
        float64: Default::default(),
    }
}

#[test]
fn captured_celeste_ldc64_loads_both_words_and_lowers_to_wgsl_and_spirv() {
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        pixel_header(3),
        &[
            0,
            move_bits(15, 32),
            0xef95_0010_0007_0f08,
            0x5c98_0780_0087_0000, // MOV R0,R8.
            0,
            0x5c98_0780_0097_0001, // MOV R1,R9.
            0xe300_0000_0007_000f,
            0,
        ],
        16,
    );
    let inputs = nixe_gpu::ShaderEvaluationInputs::default()
        .with_constant_buffer_bits(1, 32, 0x89ab_cdef)
        .with_constant_buffer_bits(1, 36, 0x0123_4567);
    let evaluated = nixe_gpu::evaluate_shader_ir(&translated, &inputs, 32).unwrap();
    assert_eq!(
        evaluated.output_bits(nixe_gpu::ShaderIoLocation::Color(0), 0),
        Some(0x89ab_cdef)
    );
    assert_eq!(
        evaluated.output_bits(nixe_gpu::ShaderIoLocation::Color(0), 1),
        Some(0x0123_4567)
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
    nixe_gpu::lower_shader_ir_to_spirv(&translated, spirv_options()).unwrap();
}

#[test]
fn multiword_ldc_preserves_overlapping_address_signed_offset_and_predicate() {
    let values = [0x8000_0000, 0x7fc0_0123, 0x0123_4567, 0xffff_ffff];
    for (memory_type, words) in [(5_u64, 2_u8), (6, 4)] {
        for address_register in 0..words {
            for skipped in [false, true] {
                let mut code = vec![0];
                let mut original = [0x11, 0x22, 0x33, 0x44];
                original[usize::from(address_register)] = 32;
                for (register, bits) in original.iter().enumerate() {
                    if code.len() % 4 == 0 {
                        code.push(0); // Scheduling word.
                    }
                    code.push(move_bits(register as u8, *bits));
                }
                // LDC B64/B128,R0,c[1][address_register-16], optionally @!PT.
                let ldc = 0xef90_0010_0007_0000
                    | (memory_type << 48)
                    | (0xfff0 << 20)
                    | (u64::from(address_register) << 8)
                    | (u64::from(skipped) << 19);
                code.push(ldc);
                while code.len() % 4 != 0 {
                    code.push(0xe300_0000_0007_000f);
                }
                // The first EXIT above terminates the program, after the load.
                let translated = translated_fixture_with_register_count(
                    MaxwellShaderStage::Pixel,
                    pixel_header((1 << words) - 1),
                    &code,
                    4,
                );
                let mut inputs = nixe_gpu::ShaderEvaluationInputs::default();
                for (word, value) in values.iter().enumerate() {
                    inputs = inputs.with_constant_buffer_bits(1, 16 + word as u32 * 4, *value);
                }
                let result = nixe_gpu::evaluate_shader_ir(&translated, &inputs, 32).unwrap();
                for word in 0..words {
                    assert_eq!(
                        result.output_bits(nixe_gpu::ShaderIoLocation::Color(0), word),
                        Some(if skipped {
                            original[usize::from(word)]
                        } else {
                            values[usize::from(word)]
                        }),
                        "type={memory_type}, address=R{address_register}, skipped={skipped}, word={word}"
                    );
                }
                validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
                nixe_gpu::lower_shader_ir_to_spirv(&translated, spirv_options()).unwrap();
            }
        }
    }
}

#[test]
fn multiword_ldc_validates_the_entire_destination_range() {
    for (memory_type, destination) in [(5_u64, 3_u64), (6, 1)] {
        assert!(matches!(
            decode_constant_buffer_load(
                MaxwellShaderStage::Pixel,
                0x148,
                0xef90_0010_0007_0000 | (memory_type << 48) | destination,
                4,
                &mut 4,
            ),
            Err(MaxwellShaderTranslationError::MalformedInstruction { .. })
        ));
    }
}

#[test]
fn captured_predicated_mov_and_constant_buffer_form_decode() {
    let captured = 0x5c98_0780_0078_0005_u64;
    let mut temporary = 8;
    let decoded = decode_move(
        MaxwellShaderStage::Vertex,
        0x2b0,
        captured,
        8,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(
        decode_predicate(captured),
        ShaderPredicate::Register {
            register: 0,
            inverted: true,
        }
    );
    assert!(matches!(
        decoded.operations.as_slice(),
        [ShaderOperation::Move32 {
            destination,
            source,
            scalar_type: ShaderScalarType::Unsigned32,
        }] if destination.index() == 5 && source.index() == 7
    ));

    let constant = decode_move(
        MaxwellShaderStage::Vertex,
        8,
        0x4c98_0788_0047_0001,
        8,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(constant.constant_buffer_binding, Some(2));
    assert!(matches!(
        constant.operations.as_slice(),
        [
            ShaderOperation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 16,
                ..
            },
            ShaderOperation::Move32 {
                destination,
                source,
                ..
            }
        ] if destination.index() == 1 && source.index() == 8
    ));

    let zero = decode_move(
        MaxwellShaderStage::Vertex,
        16,
        0x5c98_0780_0ff7_0002,
        8,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        zero.operations.as_slice(),
        [ShaderOperation::MoveImmediate32 {
            destination,
            bits: 0,
            ..
        }] if destination.index() == 2
    ));

    assert!(matches!(
        decode_move(
            MaxwellShaderStage::Vertex,
            24,
            captured & !(0xf << 39),
            8,
            &mut temporary,
        ),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "MOV partial quad-lane mask",
            ..
        })
    ));
}

#[test]
fn generated_valid_mov32i_encodings_decode_by_family() {
    let mut seed = 0x4d59_5df4_d0f3_3173_u64;
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    for destination in 0..4_u8 {
        for _ in 0..32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let immediate = seed as u32;
            let encoding = 0x0100_0000_0000_0000_u64
                | (u64::from(immediate) << 20)
                | (7 << 16)
                | (0xf << 12)
                | u64::from(destination);
            let translated = translated_fixture(
                MaxwellShaderStage::Vertex,
                header,
                &[0, encoding, 0xe300_0000_0007_000f, 0],
            );
            assert!(translated.ir().instructions().iter().any(|instruction| {
                matches!(
                    instruction.operation(),
                    ShaderOperation::MoveImmediate32 {
                        destination: decoded_destination,
                        bits,
                        ..
                    } if decoded_destination.index() == u16::from(destination)
                        && *bits == immediate
                )
            }));
        }
    }
}
