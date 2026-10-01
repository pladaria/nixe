use super::super::test_support::{translated_fixture, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderFloatComparison, ShaderOperation, ShaderPredicate, ShaderPredicateSetOperation,
    ShaderResourceAccess, ShaderResourceKind, ShaderScalarType,
};

#[test]
fn captured_fmul_ftz_reads_the_declared_constant_buffer_word() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let translated = translated_fixture(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_ff80_087f_ff00,
            0x4c68_1000_0007_0002,
            0xe300_0000_0007_000f,
        ],
    );
    let ir = translated.ir();

    assert_eq!(
        ir.resources(),
        [ShaderResourceAccess::new(0, ShaderResourceKind::ConstantBuffer, true, false,).unwrap()]
    );
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::LoadConstantBuffer32 {
            destination,
            binding: 0,
            byte_offset: 0,
            scalar_type: ShaderScalarType::Float32,
        } if destination.index() == 4
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::Multiply32 {
            destination,
            left,
            right,
            scalar_type: ShaderScalarType::Float32,
            float_control,
        } if destination.index() == 2
            && left.index() == 0
            && right.index() == 4
            && float_control.flush_denormals_to_zero()
            && float_control.denormals_are_zero()
    )));
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(
        module
            .source()
            .contains("@group(0) @binding(0) var<storage, read> constant_buffer_0: array<u32>;")
    );
    assert!(module.source().contains("nixe_flush_denormal"));
    validate_wgsl(&module);
}

#[test]
fn fmul_register_and_compact_immediate_forms_decode_consistently() {
    let mut temporary = 4;
    let register = decode_float_multiply(
        MaxwellShaderStage::Vertex,
        8,
        0x5c68_0000_0017_0102,
        4,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(register.constant_buffer_binding, None);
    assert!(matches!(
        register.operations.as_slice(),
        [ShaderOperation::Multiply32 {
            destination,
            left,
            right,
            ..
        }] if destination.index() == 2 && left.index() == 1 && right.index() == 1
    ));

    let immediate_bits = 2.0_f32.to_bits();
    let immediate_encoding =
        0x3868_0000_0007_0102_u64 | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20);
    let immediate = decode_float_multiply(
        MaxwellShaderStage::Vertex,
        16,
        immediate_encoding,
        4,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits, .. },
            ShaderOperation::Multiply32 { right, .. },
        ] if *bits == immediate_bits && right.index() == 4
    ));
}

#[test]
fn captured_ffma_ftz_uses_one_rounding_and_constant_buffer_offset() {
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
            mov(1, 2.0_f32.to_bits()),
            mov(2, 3.0_f32.to_bits()),
            0,
            mov(3, 1.0_f32.to_bits()),
            0x49a0_0100_0047_0102,
            0xe300_0000_0007_000f,
        ],
    );
    let ir = translated.ir();

    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::LoadConstantBuffer32 {
            destination,
            binding: 0,
            byte_offset: 16,
            scalar_type: ShaderScalarType::Float32,
        } if destination.index() == 4
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::FusedMultiplyAdd32 {
            destination,
            left,
            right,
            addend,
            float_control,
        } if destination.index() == 2
            && left.index() == 1
            && right.index() == 4
            && addend.index() == 2
            && float_control.flush_denormals_to_zero()
    )));
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("bitcast<u32>(fma("));
    validate_wgsl(&module);
}

#[test]
fn ffma_register_immediate_and_constant_addend_forms_decode() {
    let mut temporary = 4;
    let register_encoding = 0x5980_0000_0007_0100_u64 | (2 << 20) | (3 << 39);
    let register = decode_float_fused_multiply_add(
        MaxwellShaderStage::Vertex,
        8,
        register_encoding,
        4,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        register.operations.as_slice(),
        [ShaderOperation::FusedMultiplyAdd32 {
            destination,
            left,
            right,
            addend,
            ..
        }] if destination.index() == 0
            && left.index() == 1
            && right.index() == 2
            && addend.index() == 3
    ));

    let immediate_bits = 2.0_f32.to_bits();
    let immediate_encoding =
        0x3280_0000_0007_0100_u64 | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20) | (3 << 39);
    let immediate = decode_float_fused_multiply_add(
        MaxwellShaderStage::Vertex,
        16,
        immediate_encoding,
        4,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits, .. },
            ShaderOperation::FusedMultiplyAdd32 { right, addend, .. },
        ] if *bits == immediate_bits && right.index() == 4 && addend.index() == 3
    ));

    let constant_addend_encoding = 0x5180_0000_0007_0100_u64 | (2 << 20) | (2 << 39);
    let constant_addend = decode_float_fused_multiply_add(
        MaxwellShaderStage::Vertex,
        24,
        constant_addend_encoding,
        4,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(constant_addend.constant_buffer_binding, Some(0));
    assert!(matches!(
        constant_addend.operations.as_slice(),
        [
            ShaderOperation::LoadConstantBuffer32 {
                byte_offset: 8,
                ..
            },
            ShaderOperation::FusedMultiplyAdd32 { right, addend, .. },
        ] if right.index() == 2 && addend.index() == 5
    ));
}

#[test]
fn captured_pixel_ffma_decodes_product_and_addend_sign_modifiers() {
    let mut temporary = 16;
    let captured = decode_float_fused_multiply_add(
        MaxwellShaderStage::Pixel,
        0xf8,
        0x59a2_0500_0057_0605,
        16,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        captured.operations.as_slice(),
        [
            ShaderOperation::FloatNegate32 {
                destination: negated_addend,
                source,
            },
            ShaderOperation::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                ..
            },
        ] if source.index() == 10
            && negated_addend.index() == 16
            && destination.index() == 5
            && left.index() == 6
            && right.index() == 5
            && addend == negated_addend
    ));

    let mut temporary = 8;
    let product_negated = decode_float_fused_multiply_add(
        MaxwellShaderStage::Pixel,
        0x100,
        0x5981_0000_0007_0201_u64 | (3 << 20) | (4 << 39),
        8,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        product_negated.operations.as_slice(),
        [
            ShaderOperation::FloatNegate32 {
                destination: negated_left,
                source,
            },
            ShaderOperation::FusedMultiplyAdd32 {
                left,
                right,
                addend,
                ..
            },
        ] if source.index() == 2
            && negated_left.index() == 8
            && left == negated_left
            && right.index() == 3
            && addend.index() == 4
    ));
}

#[test]
fn captured_fadd_ftz_reads_the_expected_constant_buffer_word() {
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
            mov(1, 2.0_f32.to_bits()),
            mov(2, 3.0_f32.to_bits()),
            0,
            mov(3, 1.0_f32.to_bits()),
            0x4c58_1000_00c7_0201,
            0xe300_0000_0007_000f,
        ],
    );
    let ir = translated.ir();

    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::LoadConstantBuffer32 {
            destination,
            binding: 0,
            byte_offset: 48,
            scalar_type: ShaderScalarType::Float32,
        } if destination.index() == 4
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::Add32 {
            destination,
            left,
            right,
            scalar_type: ShaderScalarType::Float32,
            float_control,
        } if destination.index() == 1
            && left.index() == 2
            && right.index() == 4
            && float_control.flush_denormals_to_zero()
    )));
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains(" + bitcast<f32>"));
    validate_wgsl(&module);
}

#[test]
fn fadd_register_and_compact_immediate_forms_decode() {
    let mut temporary = 4;
    let register = decode_float_add(
        MaxwellShaderStage::Vertex,
        8,
        0x5c58_0000_0027_0100,
        4,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        register.operations.as_slice(),
        [ShaderOperation::Add32 {
            destination,
            left,
            right,
            ..
        }] if destination.index() == 0 && left.index() == 1 && right.index() == 2
    ));

    let immediate_bits = (-2.0_f32).to_bits();
    let immediate_encoding = 0x3858_0000_0007_0100_u64
        | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
        | (u64::from(immediate_bits >> 31) << 56);
    let immediate = decode_float_add(
        MaxwellShaderStage::Vertex,
        16,
        immediate_encoding,
        4,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits, .. },
            ShaderOperation::Add32 { right, .. },
        ] if *bits == immediate_bits && right.index() == 4
    ));
}

#[test]
fn captured_fadd_accepts_rz_and_preserves_both_negate_modifiers() {
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
            mov(1, 2.0_f32.to_bits()),
            mov(2, 3.0_f32.to_bits()),
            0,
            mov(3, 1.0_f32.to_bits()),
            0x5c59_3000_0017_ff02,
            0xe300_0000_0007_000f,
        ],
    );
    let ir = translated.ir();

    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::MoveImmediate32 {
            destination,
            bits: 0,
            scalar_type: ShaderScalarType::Float32,
        } if destination.index() == 4
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::FloatNegate32 {
            destination,
            source,
        } if destination.index() == 5 && source.index() == 4
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::FloatNegate32 {
            destination,
            source,
        } if destination.index() == 6 && source.index() == 1
    )));
    assert!(ir.instructions().iter().any(|instruction| matches!(
        instruction.operation(),
        ShaderOperation::Add32 {
            destination,
            left,
            right,
            float_control,
            ..
        } if destination.index() == 2
            && left.index() == 5
            && right.index() == 6
            && float_control.flush_denormals_to_zero()
    )));
    let module = nixe_gpu::lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("^ 0x80000000u"));
    validate_wgsl(&module);
}

#[test]
fn captured_fsetp_lt_ftz_writes_p0_from_rz_and_r0() {
    let mut temporary = 8;
    let decoded = decode_float_set_predicate(
        MaxwellShaderStage::Vertex,
        0x270,
        0x5bb1_8380_0007_ff07,
        8,
        &mut temporary,
    )
    .unwrap();

    assert_eq!(decoded.constant_buffer_binding, None);
    assert!(matches!(
        decoded.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 {
                destination: zero,
                bits: 0,
                ..
            },
            ShaderOperation::SetPredicateFloat32 {
                destination: 0,
                left,
                right,
                comparison: ShaderFloatComparison::OrderedLess,
                accumulator: ShaderPredicate::Always,
                set_operation: ShaderPredicateSetOperation::And,
                flush_denormals_to_zero: true,
            }
        ] if left == zero && right.index() == 0
    ));
}

#[test]
fn fsetp_register_immediate_and_constant_buffer_forms_decode() {
    let mut temporary = 8;
    let register_encoding =
        0x5bb2_0000_0007_0107_u64 | (2 << 3) | (3 << 20) | (4 << 39) | (1 << 42) | (1 << 45);
    let register = decode_float_set_predicate(
        MaxwellShaderStage::Vertex,
        8,
        register_encoding,
        8,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        register.operations.as_slice(),
        [ShaderOperation::SetPredicateFloat32 {
            destination: 2,
            left,
            right,
            comparison: ShaderFloatComparison::OrderedEqual,
            accumulator: ShaderPredicate::Register {
                register: 4,
                inverted: true,
            },
            set_operation: ShaderPredicateSetOperation::Or,
            ..
        }] if left.index() == 1 && right.index() == 3
    ));

    let immediate_bits = (-2.0_f32).to_bits();
    let immediate_encoding = 0x36b5_0000_0007_0107_u64
        | (u64::from((immediate_bits >> 12) & 0x7ffff) << 20)
        | (u64::from(immediate_bits >> 31) << 56);
    let immediate = decode_float_set_predicate(
        MaxwellShaderStage::Vertex,
        16,
        immediate_encoding,
        8,
        &mut temporary,
    )
    .unwrap();
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits, .. },
            ShaderOperation::SetPredicateFloat32 {
                comparison: ShaderFloatComparison::OrderedNotEqual,
                ..
            }
        ] if *bits == immediate_bits
    ));

    let constant_encoding = 0x4bb6_0000_0007_0107_u64 | (4 << 20) | (2 << 34);
    let constant = decode_float_set_predicate(
        MaxwellShaderStage::Vertex,
        24,
        constant_encoding,
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
            ShaderOperation::SetPredicateFloat32 {
                comparison: ShaderFloatComparison::OrderedGreaterOrEqual,
                ..
            }
        ]
    ));
}

#[test]
fn fmnmx_register_immediate_and_constant_forms_decode_with_minimum_selector() {
    let captured = 0x5c60_1780_0ff7_0a0a;
    let mut temporary = 16;
    let register = decode_float_min_max(
        MaxwellShaderStage::Pixel,
        0x318,
        captured,
        16,
        &mut temporary,
    )
    .unwrap();
    assert_eq!(register.constant_buffer_binding, None);
    assert!(matches!(
        register.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits: 0, .. },
            ShaderOperation::FloatMinMax32 {
                destination,
                left,
                minimum: ShaderPredicate::Never,
                float_control,
                ..
            }
        ] if destination.index() == 10
            && left.index() == 10
            && float_control.flush_denormals_to_zero()
            && float_control.denormals_are_zero()
    ));

    let immediate_bits = 1.5_f32.to_bits();
    let immediate = 0x3860_0000_0007_0100_u64 | (u64::from(immediate_bits >> 12) << 20) | (7 << 39);
    let immediate =
        decode_float_min_max(MaxwellShaderStage::Pixel, 8, immediate, 16, &mut temporary).unwrap();
    assert_eq!(immediate.constant_buffer_binding, None);
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits, .. },
            ShaderOperation::FloatMinMax32 {
                minimum: ShaderPredicate::Always,
                ..
            }
        ] if *bits == immediate_bits
    ));

    let constant = 0x4c60_0000_0007_0100_u64 | (3 << 34) | (4 << 20) | (7 << 39);
    let constant =
        decode_float_min_max(MaxwellShaderStage::Pixel, 8, constant, 16, &mut temporary).unwrap();
    assert_eq!(constant.constant_buffer_binding, Some(3));
    assert!(matches!(
        constant.operations.as_slice(),
        [
            ShaderOperation::LoadConstantBuffer32 {
                binding: 3,
                byte_offset: 16,
                ..
            },
            ShaderOperation::FloatMinMax32 {
                minimum: ShaderPredicate::Always,
                ..
            }
        ]
    ));
}
