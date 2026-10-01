use super::super::error::MaxwellShaderTranslationError;
use super::super::link::finalize_shader_ir;
use super::super::test_support::{
    translated_fixture, translated_fixture_with_register_count, validate_wgsl,
};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderOperation, ShaderRegister, ShaderResourceKind, ShaderTextureSampleOutput,
    lower_shader_ir_to_wgsl,
};
use std::collections::BTreeMap;

#[test]
fn tlds_level_zero_preserves_coordinates_channels_and_samplerless_binding() {
    let mut bindings = BTreeMap::new();
    let captured = 0xda50_1a40_2077_0600;
    let operation = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x30,
        captured,
        8,
        &mut bindings,
    )
    .unwrap();
    assert_eq!(
        operation,
        ShaderOperation::LoadTexture2D {
            outputs: (0..4)
                .map(|component| ShaderTextureSampleOutput::new(
                    ShaderRegister::new(u16::from(component)),
                    component
                )
                .unwrap())
                .collect(),
            coordinates: [ShaderRegister::new(6), ShaderRegister::new(7)],
            image_binding: 32,
            mip_level: 0,
        }
    );
    assert_eq!(bindings[&420].constant_buffer_byte_offset, 1680);
    assert_eq!(bindings[&420].sampler_binding, None);
    // The same descriptor can later be filtered: reserve its sampler once.
    let sample = (captured & !(0x1f << 53)) | (1 << 53);
    decode_texture_access_simplified(MaxwellShaderStage::Pixel, 0x38, sample, 8, &mut bindings)
        .unwrap();
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[&420].sampler_binding, Some(33));
    decode_texture_access_simplified(MaxwellShaderStage::Pixel, 0x40, captured, 8, &mut bindings)
        .unwrap();
    assert_eq!(bindings[&420].sampler_binding, Some(33));
    for word in [
        captured & !(1 << 59),
        captured ^ (1 << 53),
        captured | (1 << 55),
    ] {
        assert!(matches!(
            decode_texture_access_simplified(
                MaxwellShaderStage::Pixel,
                0x30,
                word,
                8,
                &mut BTreeMap::new()
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { .. })
        ));
    }
}

#[test]
fn tlds_reaches_wgsl_without_declaring_a_sampler() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0xf;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0037_f006,
            0x0100_0000_0057_f007,
            0xda50_1a40_2077_0600,
            0,
            0xe300_0000_0007_000f,
        ],
        8,
    );
    assert_eq!(translated.ir().resources().len(), 1);
    assert_eq!(
        translated.ir().resources()[0].kind(),
        ShaderResourceKind::SampledImage
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("textureLoad(sampled_image_32"));
    assert!(!module.source().contains("textureSample("));
    assert!(!module.source().contains("var sampler_"));
    validate_wgsl(&module);
    let mapping = BTreeMap::from([(32, 4)]);
    let remapped = finalize_shader_ir(
        translated.ir().clone(),
        MaxwellShaderStage::Pixel,
        &mapping,
        &[],
    )
    .unwrap();
    assert!(
        remapped
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::LoadTexture2D {
                    image_binding: 4,
                    ..
                }
            ))
    );
}

#[test]
fn texs_2d_implicit_lod_decodes_captured_split_rgba_operands() {
    let encoding = 0xd830_0080_2007_0100;
    let mut bindings = BTreeMap::new();
    let operation = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x2a8,
        encoding,
        4,
        &mut bindings,
    )
    .unwrap();

    assert_eq!(
        operation,
        ShaderOperation::SampleTexture2D {
            outputs: (0..4)
                .map(|component| {
                    ShaderTextureSampleOutput::new(
                        ShaderRegister::new(u16::from(component)),
                        component,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            coordinates: [ShaderRegister::new(1), ShaderRegister::new(0)],
            image_binding: 32,
            sampler_binding: 33,
        }
    );
    assert_eq!(
        bindings.get(&8),
        Some(&MaxwellTextureResourceBinding {
            constant_buffer_byte_offset: 32,
            image_binding: 32,
            sampler_binding: Some(33),
            image_kind: ShaderResourceKind::SampledImage,
        })
    );
}

#[test]
fn texs_2d_array_implicit_lod_decodes_packed_layer_and_coordinates() {
    let encoding = 0xd8e0_1a4f_f027_0003;
    let mut bindings = BTreeMap::new();
    let operation = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x30,
        encoding,
        4,
        &mut bindings,
    )
    .unwrap();

    assert_eq!(
        operation,
        ShaderOperation::SampleTexture2DArray {
            outputs: vec![ShaderTextureSampleOutput::new(ShaderRegister::new(3), 0).unwrap()]
                .into_boxed_slice(),
            coordinates: [ShaderRegister::new(1), ShaderRegister::new(2)],
            array_index: ShaderRegister::new(0),
            image_binding: 32,
            sampler_binding: 33,
        }
    );
    assert_eq!(
        bindings.get(&420),
        Some(&MaxwellTextureResourceBinding {
            constant_buffer_byte_offset: 1680,
            image_binding: 32,
            sampler_binding: Some(33),
            image_kind: ShaderResourceKind::SampledImage2DArray,
        })
    );
}

#[test]
fn texs_2d_array_lowers_to_an_array_texture_and_unsigned_u16_layer() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0x0000_000f;
    let translated = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0007_f000_u64 | (7_u64 << 20),
            0x0100_0000_0007_f001_u64 | (u64::from(0.25_f32.to_bits()) << 20),
            0x0100_0000_0007_f002_u64 | (u64::from(0.75_f32.to_bits()) << 20),
            0,
            0xd8e0_1a4f_f027_0003,
            0xe300_0000_0007_000f,
            0,
        ],
    );

    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 32 && resource.kind() == ShaderResourceKind::SampledImage2DArray
    }));
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("texture_2d_array<f32>"));
    assert!(module.source().contains("i32(registers[0] & 0xffffu)"));
    validate_wgsl(&module);
}

#[test]
fn texs_2d_implicit_lod_translates_to_verified_sample_resources_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0x0000_000f;
    let move_x = 0x0100_0000_0007_f000_u64 | (u64::from(0.25_f32.to_bits()) << 20);
    let move_y = 0x0100_0000_0007_f001_u64 | (u64::from(0.75_f32.to_bits()) << 20);
    let translated = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            move_x,
            move_y,
            0xd830_0080_2007_0100,
            0,
            0xe300_0000_0007_000f,
            0,
            0,
        ],
    );

    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 32 && resource.kind() == ShaderResourceKind::SampledImage
    }));
    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 33 && resource.kind() == ShaderResourceKind::Sampler
    }));
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("textureSample"));
    validate_wgsl(&module);
}
