use super::super::decode::decode_predicate;
use super::super::test_support::{translated_fixture, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderMathAccuracy, ShaderOperation, ShaderPredicate, ShaderSpecialFunction};

#[test]
fn captured_mufu_rsq_applies_absolute_before_approximate_reciprocal_sqrt() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let mov = |destination: u8, bits: u32| {
        0x0100_0000_0000_0000_u64
            | (u64::from(bits) << 20)
            | (7 << 16)
            | (0xf << 12)
            | u64::from(destination)
    };
    let translated = translated_fixture(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            mov(0, 1.0_f32.to_bits()),
            mov(1, (-4.0_f32).to_bits()),
            mov(2, 3.0_f32.to_bits()),
            0,
            mov(3, 1.0_f32.to_bits()),
            0x5080_4000_0057_0101,
            0xe300_0000_0007_000f,
        ],
    );
    let ir = translated.ir();

    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::FloatAbsolute32 {
            destination,
            source,
        } if destination.index() == 4 && source.index() == 1
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::ReciprocalSqrt32 {
            destination,
            source,
            accuracy: ShaderMathAccuracy::Approximate,
            ..
        } if destination.index() == 1 && source.index() == 4
    )));
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("inverseSqrt"));
    validate_wgsl(&module);
}

#[test]
fn captured_predicated_mufu_sqrt_and_scalar_special_functions_decode() {
    let captured = 0x5080_0000_0080_0007_u64;
    let mut temporary = 8;
    let decoded = decode_mufu(
        MaxwellShaderStage::Vertex,
        0x278,
        captured,
        8,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(
        decode_predicate(captured),
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        }
    );
    assert!(matches!(
        decoded.as_slice(),
        [ShaderOperation::SpecialFunction32 {
            destination,
            source,
            function: ShaderSpecialFunction::SquareRoot,
            accuracy: ShaderMathAccuracy::Approximate,
            ..
        }] if destination.index() == 7 && source.index() == 0
    ));

    for (operation, expected) in [
        (0, ShaderSpecialFunction::Cosine),
        (1, ShaderSpecialFunction::Sine),
        (2, ShaderSpecialFunction::Exp2),
        (3, ShaderSpecialFunction::Log2),
        (8, ShaderSpecialFunction::SquareRoot),
    ] {
        let encoding = 0x5080_0000_0007_0100_u64 | (operation << 20);
        let decoded =
            decode_mufu(MaxwellShaderStage::Vertex, 8, encoding, 8, &mut temporary).unwrap();
        assert!(matches!(
            decoded.as_slice(),
            [ShaderOperation::SpecialFunction32 { function, .. }] if *function == expected
        ));
    }
}

#[test]
fn captured_rro_ex2_fuses_with_the_matching_mufu() {
    let captured = 0x5c90_0080_00a7_000a_u64;
    let mut temporary = 16;
    let range_reduction = decode_range_reduction(
        MaxwellShaderStage::Pixel,
        0x338,
        captured,
        decode_predicate(captured),
        16,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(range_reduction.destination, 10);
    assert_eq!(range_reduction.input.index(), 10);
    assert_eq!(range_reduction.mode, MaxwellRangeReduction::Exp2);
    assert!(range_reduction.preparation.is_empty());

    let mufu = 0x5080_0000_0027_0a0a_u64;
    assert!(is_compatible_mufu(
        &range_reduction,
        mufu,
        decode_predicate(mufu)
    ));
    assert!(matches!(
        decode_range_reduced_mufu(
            MaxwellShaderStage::Pixel,
            0x348,
            mufu,
            16,
            &range_reduction,
        )
        .unwrap(),
        ShaderOperation::SpecialFunction32 {
            destination,
            source,
            function: ShaderSpecialFunction::Exp2,
            accuracy: ShaderMathAccuracy::Approximate,
            ..
        } if destination.index() == 10 && source.index() == 10
    ));
}

#[test]
fn rro_register_constant_and_immediate_sources_decode() {
    let mut temporary = 16;
    let register = 0x5c90_0000_0027_0001_u64 | (2 << 20) | (1 << 45) | (1 << 49);
    let register = decode_range_reduction(
        MaxwellShaderStage::Pixel,
        8,
        register,
        ShaderPredicate::Always,
        16,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(register.mode, MaxwellRangeReduction::SinCos);
    assert!(matches!(
        register.preparation.as_slice(),
        [
            ShaderOperation::FloatAbsolute32 { source, .. },
            ShaderOperation::FloatNegate32 { .. }
        ] if source.index() == 2
    ));

    let constant = 0x4c90_0000_0007_0001_u64 | (3 << 34) | (5 << 20) | (1 << 39);
    let constant = decode_range_reduction(
        MaxwellShaderStage::Pixel,
        16,
        constant,
        ShaderPredicate::Always,
        16,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(constant.mode, MaxwellRangeReduction::Exp2);
    assert_eq!(constant.constant_buffer_binding, Some(3));
    assert!(matches!(
        constant.preparation.as_slice(),
        [ShaderOperation::LoadConstantBuffer32 {
            binding: 3,
            byte_offset: 20,
            ..
        }]
    ));

    let immediate_bits = (-2.0_f32).to_bits();
    let immediate = 0x3890_0000_0007_0001_u64
        | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
        | (u64::from(immediate_bits >> 31) << 56)
        | (1 << 39);
    let immediate = decode_range_reduction(
        MaxwellShaderStage::Pixel,
        24,
        immediate,
        ShaderPredicate::Always,
        16,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        immediate.preparation.as_slice(),
        [ShaderOperation::MoveImmediate32 { bits, .. }] if *bits == immediate_bits
    ));
}

#[test]
fn rro_fusion_rejects_mismatched_modes_predicates_and_modifiers() {
    let encoding = 0x5c90_0080_0017_0101_u64;
    let mut temporary = 4;
    let range_reduction = decode_range_reduction(
        MaxwellShaderStage::Pixel,
        8,
        encoding,
        ShaderPredicate::Always,
        4,
        &mut temporary,
    )
    .unwrap();
    let cosine = 0x5080_0000_0007_0101_u64;
    assert!(!is_compatible_mufu(
        &range_reduction,
        cosine,
        ShaderPredicate::Always
    ));
    let exp2 = cosine | (2 << 20);
    assert!(!is_compatible_mufu(
        &range_reduction,
        exp2,
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        }
    ));
    assert!(!is_compatible_mufu(
        &range_reduction,
        exp2 | (1 << 46),
        ShaderPredicate::Always
    ));
}

#[test]
fn adjacent_rro_mufu_pair_lowers_as_one_high_level_special_function() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let rro = 0x5c90_0000_0007_0001_u64 | (1 << 20) | (1 << 39);
    let mufu = 0x5080_0000_0007_0101_u64 | (2 << 20);
    let translated = translated_fixture(
        MaxwellShaderStage::Vertex,
        header,
        &[0, rro, mufu, 0xe300_0000_0007_000f],
    );
    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::SpecialFunction32 {
                    destination,
                    source,
                    function: ShaderSpecialFunction::Exp2,
                    ..
                } if destination.index() == 1 && source.index() == 1
            ))
    );
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("exp2"));
    validate_wgsl(&module);
}
