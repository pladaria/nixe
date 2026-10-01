use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{translated_fixture, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderInterfaceElement, ShaderInterpolation, ShaderIoLocation, ShaderOperation, ShaderRegister,
    ShaderScalarType, ShaderStage, lower_shader_ir_to_wgsl,
};
use std::collections::BTreeMap;

#[test]
fn captured_vertex_system_value_ald_family_reaches_verified_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_1000;
    for (encoding, location) in [
        (0xefd8_7f80_2f87_ff00, ShaderIoLocation::InstanceId),
        (0xefd8_7f80_2fc7_ff00, ShaderIoLocation::VertexId),
    ] {
        let translated = translated_fixture(
            MaxwellShaderStage::Vertex,
            header,
            &[0, encoding, 0xe300_0000_0007_000f, 0],
        );
        let ir = translated.ir();

        assert!(ir.inputs().iter().any(|input| {
            input.location() == location
                && input.component() == 0
                && input.scalar_type() == ShaderScalarType::Unsigned32
        }));
        assert!(matches!(
            ir.instructions()[0].operation(),
            ShaderOperation::LoadInput {
                location: decoded_location,
                first_component: 0,
                scalar_type: ShaderScalarType::Unsigned32,
                ..
            } if *decoded_location == location
        ));
        validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
    }
}

#[test]
fn vertex_system_value_ald_spans_adjacent_instance_and_vertex_id_slots() {
    let operations = decode_attribute_load(
        MaxwellShaderStage::Vertex,
        8,
        0xefd8_ff80_2f87_ff00,
        4,
        &BTreeMap::new(),
    )
    .unwrap();
    assert!(matches!(
        operations.as_slice(),
        [
            ShaderOperation::LoadInput {
                destinations: instance,
                location: ShaderIoLocation::InstanceId,
                first_component: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::LoadInput {
                destinations: vertex,
                location: ShaderIoLocation::VertexId,
                first_component: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            }
        ] if instance[0].index() == 0 && vertex[0].index() == 1
    ));
    assert!(matches!(
        decode_attribute_load(
            MaxwellShaderStage::Pixel,
            8,
            0xefd8_7f80_2fc7_ff00,
            4,
            &BTreeMap::new(),
        ),
        Err(MaxwellShaderTranslationError::MalformedInstruction {
            reason: "vertex system-value ALD is used outside a vertex shader",
            ..
        })
    ));
}

#[test]
fn attribute_load_spans_generic_vectors_without_losing_register_order() {
    let operations = decode_attribute_load(
        MaxwellShaderStage::Vertex,
        8,
        0xefd8_ff80_08c7_ff02,
        8,
        &BTreeMap::new(),
    )
    .unwrap();
    assert!(matches!(
        operations.as_slice(),
        [
            ShaderOperation::LoadInput {
                destinations: first,
                location: ShaderIoLocation::Generic(0),
                first_component: 3,
                ..
            },
            ShaderOperation::LoadInput {
                destinations: second,
                location: ShaderIoLocation::Generic(1),
                first_component: 0,
                ..
            }
        ] if first[0].index() == 2 && second[0].index() == 3
    ));
}

#[test]
fn captured_fragment_ipa_reciprocal_and_color_output_translate() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[5] = 0x8000_0000;
    header[6] = 0x0000_002a;
    header[18] = 0x0000_000f;
    let code = [
        0x001f_b001_e020_070f,
        0xe003_ff87_cff7_ff00,
        0x5080_0000_0047_0002,
        0x0103_f800_0007_f003,
        0x015c_8800_6840_0901,
        0xe043_ff88_0027_ff00,
        0xe043_ff88_4027_ff01,
        0xe043_ff88_8027_ff02,
        0x0000_0000_0001_ffef,
        0xe300_0000_0007_000f,
        0,
        0,
    ];
    let translated = translated_fixture(MaxwellShaderStage::Pixel, header, &code);
    let ir = translated.ir();

    assert_eq!(ir.stage(), ShaderStage::Fragment);
    assert_eq!(
        ir.inputs()
            .iter()
            .filter(|element| element.location() == ShaderIoLocation::Generic(0))
            .count(),
        3
    );
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::Reciprocal32 {
            destination,
            source,
            ..
        } if destination.index() == 2 && source.index() == 0
    )));
    assert_eq!(
        ir.instructions()
            .iter()
            .filter(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::StoreOutput {
                    location: ShaderIoLocation::Color(0),
                    ..
                }
            ))
            .count(),
        4
    );
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("input.generic_0.x"));
    assert!(!module.source().contains("input.generic_0.x *"));
    validate_wgsl(&module);
}

#[test]
fn ipa_constant_and_sc_modes_preserve_declared_interpolation_and_reject_unmodeled_bits() {
    let captured = 0xe083_ff89_0ff7_ff00;
    let constant_input = ShaderInterfaceElement::new(
        ShaderIoLocation::Generic(1),
        0,
        ShaderScalarType::Float32,
        Some(ShaderInterpolation::Constant),
    )
    .unwrap();
    assert_eq!(
        decode_interpolate(
            MaxwellShaderStage::Pixel,
            0x38,
            captured,
            1,
            &[constant_input],
        )
        .unwrap(),
        ShaderOperation::InterpolateInput {
            destination: ShaderRegister::new(0),
            location: ShaderIoLocation::Generic(1),
            component: 0,
            interpolation: ShaderInterpolation::Constant,
        }
    );

    let screen_input = ShaderInterfaceElement::new(
        ShaderIoLocation::Generic(1),
        0,
        ShaderScalarType::Float32,
        Some(ShaderInterpolation::ScreenLinear),
    )
    .unwrap();
    let sc = (captured & !(3_u64 << 54)) | (3_u64 << 54);
    assert!(matches!(
        decode_interpolate(MaxwellShaderStage::Pixel, 0x38, sc, 1, &[screen_input],),
        Ok(ShaderOperation::InterpolateInput {
            interpolation: ShaderInterpolation::ScreenLinear,
            ..
        })
    ));

    for unsupported in [
        captured | (1_u64 << 38),
        captured | (1_u64 << 51),
        captured | (1_u64 << 52),
    ] {
        assert!(matches!(
            decode_interpolate(
                MaxwellShaderStage::Pixel,
                0x38,
                unsupported,
                1,
                &[constant_input],
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { .. })
        ));
    }
}
