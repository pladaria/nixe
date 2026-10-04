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
fn perspective_pass_exposes_the_unnormalized_barycentric_numerator() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[6] = 2; // Perspective generic 0.x, without an explicit Position.w load.
    header[18] = 1;
    let shader = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[0, 0xe003_ff88_0ff7_ff00, 0xe300_0000_0007_000f, 0],
    );
    // Vertex values [0, 1, 1], clip W [1, 2, 4], barycentrics [1/4, 1/4, 1/2]:
    // numerator = 1/4; interpolated 1/w = 1/2; logical host attribute = 1/2.
    let inputs = nixe_gpu::ShaderEvaluationInputs::default()
        .with_interface_bits(ShaderIoLocation::Generic(0), 0, 0.5_f32.to_bits())
        .with_interface_bits(ShaderIoLocation::Position, 3, 0.5_f32.to_bits());
    let result = nixe_gpu::evaluate_shader_ir(&shader, &inputs, 64).unwrap();
    assert_eq!(
        result.output_bits(ShaderIoLocation::Color(0), 0),
        Some(0.25_f32.to_bits())
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
    nixe_gpu::lower_shader_ir_to_spirv(
        &shader,
        nixe_gpu::SpirvShaderOptions {
            input_control_points: 0,
            tessellation_mode: None,
            float32: nixe_gpu::SpirvFloat32Capabilities {
                denorm_preserve: false,
                rounding_mode_rte: true,
                signed_zero_inf_nan_preserve: true,
                fused_multiply_add: true,
            },
            float64: nixe_gpu::SpirvFloat64Capabilities {
                enabled: true,
                rounding_mode_rte: true,
                signed_zero_inf_nan_preserve: true,
            },
        },
    )
    .unwrap();
}

#[test]
fn perspective_pass_mul_w_uses_its_register_even_when_it_aliases_destination() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[6] = 2;
    header[18] = 1;
    for factor in [2.0_f32, 3.0] {
        let shader = translated_fixture(
            MaxwellShaderStage::Pixel,
            header,
            &[
                0,
                0x0100_0000_0007_f000 | (u64::from(factor.to_bits()) << 20),
                0xe043_ff88_0007_ff00,
                0xe300_0000_0007_000f,
            ],
        );
        let inputs = nixe_gpu::ShaderEvaluationInputs::default()
            .with_interface_bits(ShaderIoLocation::Generic(0), 0, 0.5_f32.to_bits())
            .with_interface_bits(ShaderIoLocation::Position, 3, 0.5_f32.to_bits());
        let result = nixe_gpu::evaluate_shader_ir(&shader, &inputs, 64).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Color(0), 0),
            Some((0.25 * factor).to_bits())
        );
        validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
    }
}

#[test]
fn perspective_normalization_proof_tracks_register_overwrites_and_predication() {
    let make = |operation| {
        ShaderInstruction::new(
            ShaderSourceLocation::new(8),
            ShaderPredicate::Always,
            operation,
        )
    };
    let mut instructions = vec![
        make(ShaderOperation::LoadInput {
            destinations: vec![ShaderRegister::new(0)].into_boxed_slice(),
            location: ShaderIoLocation::Position,
            first_component: 3,
            scalar_type: ShaderScalarType::Float32,
        }),
        make(ShaderOperation::Reciprocal32 {
            destination: ShaderRegister::new(2),
            source: ShaderRegister::new(0),
            accuracy: nixe_gpu::ShaderMathAccuracy::Approximate,
            float_control: ShaderFloatControl::PRECISE,
        }),
    ];
    assert!(is_reciprocal_fragment_w(
        &instructions,
        ShaderRegister::new(2)
    ));
    // The reciprocal retains its value after its source register is reused.
    instructions.push(make(ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister::new(0),
        bits: 3_f32.to_bits(),
        scalar_type: ShaderScalarType::Float32,
    }));
    assert!(is_reciprocal_fragment_w(
        &instructions,
        ShaderRegister::new(2)
    ));
    instructions.push(make(ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister::new(2),
        bits: 3_f32.to_bits(),
        scalar_type: ShaderScalarType::Float32,
    }));
    assert!(!is_reciprocal_fragment_w(
        &instructions,
        ShaderRegister::new(2)
    ));
    instructions.truncate(1);
    instructions.push(ShaderInstruction::new(
        ShaderSourceLocation::new(16),
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        },
        ShaderOperation::Reciprocal32 {
            destination: ShaderRegister::new(2),
            source: ShaderRegister::new(0),
            accuracy: nixe_gpu::ShaderMathAccuracy::Approximate,
            float_control: ShaderFloatControl::PRECISE,
        },
    ));
    assert!(!is_reciprocal_fragment_w(
        &instructions,
        ShaderRegister::new(2)
    ));
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
            &mut vec![constant_input],
            &mut 8,
            false,
        )
        .unwrap(),
        vec![ShaderOperation::InterpolateInput {
            destination: ShaderRegister::new(0),
            location: ShaderIoLocation::Generic(1),
            component: 0,
            interpolation: ShaderInterpolation::Constant,
        }]
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
        decode_interpolate(MaxwellShaderStage::Pixel, 0x38, sc, 1, &mut vec![screen_input], &mut 8, false),
        Ok(operations) if matches!(operations.as_slice(), [ShaderOperation::InterpolateInput { interpolation: ShaderInterpolation::ScreenLinear, .. }])
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
                &mut vec![constant_input],
                &mut 8,
                false,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { .. })
        ));
    }
}
