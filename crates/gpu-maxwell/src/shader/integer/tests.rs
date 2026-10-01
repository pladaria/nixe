use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{translated_fixture_with_register_count, validate_wgsl};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderOperation,
    ShaderPredicate, ShaderResourceAccess, ShaderResourceKind, ShaderScalarType,
    ShaderSourceLocation, ShaderStage, VerifiedShaderIr, lower_shader_ir_to_wgsl,
};

#[test]
fn captured_iadd_reaches_verified_unsigned_add_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_8000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_7f80_2fc7_ff01,
            0x3810_0000_0017_0101,
            0xe300_0000_0007_000f,
        ],
        4,
    );
    assert!(matches!(
        translated.ir().instructions()[1].operation(),
        ShaderOperation::MoveImmediate32 { bits: 1, .. }
    ));
    assert!(
        matches!(translated.ir().instructions()[2].operation(), ShaderOperation::Add32 { destination, left, scalar_type: ShaderScalarType::Unsigned32, .. } if destination.index() == 1 && left.index() == 1)
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(
        module
            .source()
            .contains("registers[1] = registers[1] + registers[4]")
    );
    validate_wgsl(&module);
}

#[test]
fn captured_lop_and_vertex_id_reaches_verified_ir_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_8000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_7f80_2fc7_ff00,
            0x3847_0000_0027_0000,
            0xe300_0000_0007_000f,
        ],
        4,
    );
    assert!(matches!(
        translated.ir().instructions()[1].operation(),
        ShaderOperation::MoveImmediate32 { bits: 2, .. }
    ));
    assert!(matches!(translated.ir().instructions()[2].operation(),
        ShaderOperation::Bitwise32 { destination, left, operation: nixe_gpu::ShaderBitwiseOperation::And, .. }
            if destination.index() == 0 && left.index() == 0));
    validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
}

#[test]
fn predicated_lop_complement_keeps_conditionally_defined_sources_guarded() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_8000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0xefd8_7f80_2fc7_ff00, // R0 = vertex ID
            0x5b64_0380_0ff7_0007, // P0 = R0 == 0
            0x5c98_0780_0000_0001, // @P0 R1 = R0
            0,
            0x3847_0080_0020_0101, // @P0 R1 = ~R1 & 2
            0xe300_0000_0007_000f,
        ],
        4,
    );
    let expanded = translated
        .ir()
        .instructions()
        .iter()
        .filter(|instruction| instruction.source().byte_offset() == 40)
        .collect::<Vec<_>>();
    assert_eq!(expanded.len(), 4);
    assert!(expanded.iter().all(|instruction| instruction.predicate()
        == ShaderPredicate::Register {
            register: 0,
            inverted: false
        }));
    validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
}

#[test]
fn captured_immediate_shift_left_reaches_verified_ir_and_wgsl() {
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
            0xe300_0000_0007_000f,
        ],
        8,
    );

    assert!(matches!(
        translated.ir().instructions()[1].operation(),
        ShaderOperation::MoveImmediate32 {
            bits: 4,
            scalar_type: ShaderScalarType::Unsigned32,
            ..
        }
    ));
    assert!(matches!(
        translated.ir().instructions()[2].operation(),
        ShaderOperation::ShiftLeft32 {
            destination,
            value,
            wrap: false,
            ..
        } if destination.index() == 7 && value.index() == 0
    ));
    validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
}

#[test]
fn shift_left_decodes_register_immediate_and_constant_buffer_forms() {
    let mut next_temporary = 16;
    let register = decode_shift_left(
        MaxwellShaderStage::Vertex,
        8,
        0x5c48_0080_0017_0002,
        16,
        &mut next_temporary,
    )
    .unwrap();
    assert_eq!(register.constant_buffer_binding, None);
    assert!(matches!(
        register.operations.as_slice(),
        [ShaderOperation::ShiftLeft32 {
            destination,
            value,
            amount,
            wrap: true,
        }] if destination.index() == 2 && value.index() == 0 && amount.index() == 1
    ));

    let immediate = decode_shift_left(
        MaxwellShaderStage::Vertex,
        8,
        0x3948_0000_0037_0002,
        16,
        &mut next_temporary,
    )
    .unwrap();
    assert!(matches!(
        immediate.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 {
                bits: 0xfff8_0003,
                ..
            },
            ShaderOperation::ShiftLeft32 { wrap: false, .. }
        ]
    ));

    let constant = decode_shift_left(
        MaxwellShaderStage::Vertex,
        8,
        0x4c48_0008_0037_0002,
        16,
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
            ShaderOperation::ShiftLeft32 { wrap: false, .. }
        ]
    ));
}

#[test]
fn shift_left_rejects_unimplemented_condition_code_and_carry_modes() {
    for (modifier, detail) in [
        (1_u64 << 47, "SHL condition-code write"),
        (1_u64 << 43, "SHL extended carry input"),
    ] {
        let mut next_temporary = 16;
        assert!(matches!(
            decode_shift_left(
                MaxwellShaderStage::Vertex,
                8,
                0x3848_0000_0047_0002 | modifier,
                16,
                &mut next_temporary,
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: actual,
                ..
            }) if actual == detail
        ));
    }
}

const STAGE: MaxwellShaderStage = MaxwellShaderStage::TessellationInit;

#[test]
fn iadd_extended_consumes_and_preserves_carry_until_another_cc_write() {
    for (left, right, expected) in [
        (u32::MAX, 1, 2),
        (u32::MAX, 0, 1),
        (0x8000_0000, 0x8000_0000, 2),
    ] {
        let mut next = 4;
        let mut carry = None;
        let mut decoded =
            decode_add(STAGE, 8, 0x5c10_8000_0017_0000, 4, &mut next, &mut carry).unwrap();
        let original = carry;
        for _ in 0..2 {
            let high =
                decode_add(STAGE, 16, 0x3810_0800_0017_ff00, 4, &mut next, &mut carry).unwrap();
            decoded.operations.extend(high.operations);
            assert_eq!(
                carry, original,
                "IADD.X without CC must not replace the flag"
            );
        }
        assert_eq!(evaluate_integer(decoded, left, right), expected);
    }
}

#[test]
fn iadd_constant_buffer_pointer_pair_carries_across_the_four_gib_boundary() {
    for offset in [0_u32, 3, 4, 8192, u32::MAX] {
        let mut next = 4;
        let mut carry = None;
        let mut low =
            decode_add(STAGE, 8, 0x4c10_8008_0037_0000, 4, &mut next, &mut carry).unwrap();
        let high = decode_add(STAGE, 16, 0x4c10_0808_0047_ff01, 4, &mut next, &mut carry).unwrap();
        assert_eq!(low.constant_buffer_binding, Some(2));
        assert_eq!(high.constant_buffer_binding, Some(2));
        low.operations.extend(high.operations);
        let expected = 0x0000_0004_ffff_fffc_u64 + u64::from(offset);
        let high_operations = low.operations.clone();
        assert_eq!(evaluate_integer(low, offset, 0xffff_fffc), expected as u32);
        let mut high = DecodedIntegerOperation {
            operations: high_operations,
            constant_buffer_binding: Some(2),
        };
        high.operations.push(ShaderOperation::Move32 {
            destination: ShaderRegister::new(0),
            source: ShaderRegister::new(1),
            scalar_type: ShaderScalarType::Unsigned32,
        });
        assert_eq!(
            evaluate_integer(high, offset, 0xffff_fffc),
            (expected >> 32) as u32
        );
    }
}

#[test]
fn iscadd_preserves_wrapping_shift_add_and_aliased_destination() {
    for shift in 0..32 {
        for (left, right) in [
            (0, 0),
            (1, 7),
            (0xffff_ffff_u32, 1),
            (0x8123_4567, 0xfedc_ba98),
        ] {
            let encoding = 0x5c18_0000_0017_0000 | (u64::from(shift) << 39);
            assert!(is_shift_add(encoding));
            let decoded = decode_add(STAGE, 8, encoding, 4, &mut 4, &mut None).unwrap();
            assert_eq!(
                evaluate_integer(decoded, left, right),
                left.wrapping_shl(shift).wrapping_add(right)
            );
        }
    }
}

#[test]
fn iscadd_signed_immediate_constant_buffer_and_zero_register() {
    // ISCADD R0, R0, -1, 5 and ISCADD R0, R0, c[2][12], 5.
    for (encoding, right) in [
        (0x3918_02ff_fff7_0000, u32::MAX),
        (0x4c18_0288_0037_0000, 0x8765_4321),
    ] {
        assert!(is_shift_add(encoding));
        let decoded = decode_add(STAGE, 8, encoding, 4, &mut 4, &mut None).unwrap();
        assert_eq!(
            evaluate_integer(decoded, 17, right),
            (17_u32 << 5).wrapping_add(right)
        );
    }
    let decoded = decode_add(STAGE, 8, 0x5c18_0280_0017_ff00, 4, &mut 4, &mut None).unwrap();
    assert_eq!(evaluate_integer(decoded, 17, 42), 42);
}

#[test]
fn iscadd_rejects_untranslated_flags_and_reserved_fields() {
    for flag in [1 << 47, 1 << 48, 1 << 49, 1 << 44, 1 << 28] {
        assert!(decode_add(STAGE, 8, 0x5c18_0280_0017_0000 | flag, 4, &mut 4, &mut None).is_err());
    }
}

fn evaluate_lop(encoding: u64, left: u32, right: u32) -> u32 {
    let decoded = decode_bitwise(STAGE, 8, encoding, 4, &mut 4).unwrap();
    evaluate_integer(decoded, left, right)
}

fn evaluate_iadd(encoding: u64, left: u32, right: u32) -> u32 {
    assert!(is_add(encoding));
    let decoded = decode_add(STAGE, 8, encoding, 4, &mut 4, &mut None).unwrap();
    evaluate_integer(decoded, left, right)
}

fn evaluate_integer(decoded: DecodedIntegerOperation, left: u32, right: u32) -> u32 {
    let resources = decoded
        .constant_buffer_binding
        .into_iter()
        .map(|binding| {
            ShaderResourceAccess::new(binding, ShaderResourceKind::ConstantBuffer, true, false)
                .unwrap()
        })
        .collect();
    let mut operations = vec![
        ShaderOperation::MoveImmediate32 {
            destination: ShaderRegister::new(0),
            bits: left,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::MoveImmediate32 {
            destination: ShaderRegister::new(1),
            bits: right,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    ];
    operations.extend(decoded.operations);
    operations.extend([
        ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister::new(0)].into(),
            location: ShaderIoLocation::Color(0),
            first_component: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::Exit,
    ]);
    let shader = VerifiedShaderIr::verify(ShaderIr::new(
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
        resources,
        operations
            .into_iter()
            .enumerate()
            .map(|(i, operation)| {
                ShaderInstruction::new(
                    ShaderSourceLocation::new(i as u32 * 8),
                    ShaderPredicate::Always,
                    operation,
                )
            })
            .collect::<Vec<_>>(),
    ))
    .unwrap();
    nixe_gpu::evaluate_shader_ir(
        &shader,
        &nixe_gpu::ShaderEvaluationInputs::default()
            .with_constant_buffer_bits(2, 12, right)
            .with_constant_buffer_bits(2, 16, 4),
        32,
    )
    .unwrap()
    .output_bits(ShaderIoLocation::Color(0), 0)
    .unwrap()
}

#[test]
fn iadd_forms_preserve_wrapping_addition_and_signed_immediates() {
    for left in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
        for right in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
            for word in [0x5c10_0000_0017_0000, 0x4c10_0000_0037_0000 | (2 << 34)] {
                assert_eq!(evaluate_iadd(word, left, right), left.wrapping_add(right));
            }
        }
        for (word, right) in [
            (0x3810_0000_0017_0000, 1),
            (0x3810_007f_fff7_0000, 0x0007_ffff),
            (0x3910_007f_fff7_0000, u32::MAX),
            (0x3910_0000_0007_0000, 0xfff8_0000),
        ] {
            assert_eq!(evaluate_iadd(word, left, 0), left.wrapping_add(right));
        }
    }
    assert_eq!(evaluate_iadd(0x3810_0000_0017_ff00, 42, 0), 1);
    assert_eq!(evaluate_iadd(0x5c10_0000_0ff7_0000, 42, 0), 42);
    assert_eq!(evaluate_iadd(0x3810_0000_0017_00ff, 42, 0), 42);
}

#[test]
fn iadd_rejects_unrepresented_modifiers_and_invalid_registers() {
    let captured = 0x3810_0000_0017_0101_u64;
    for (modifier, detail) in [
        (1 << 43, "IADD.X has no translated carry definition"),
        (1 << 50, "IADD signed saturation"),
        (1 << 48, "IADD negated operands"),
        (1 << 49, "IADD negated operands"),
    ] {
        assert!(is_add(captured | modifier));
        assert!(
            matches!(decode_add(STAGE, 0x48, captured | modifier, 4, &mut 4, &mut None),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: actual, instruction_offset: 0x48, .. }) if actual == detail)
        );
    }
    for word in [
        captured | (1 << 39),
        captured | (1 << 46),
        captured | 4,
        captured | (4 << 8),
        0x5c10_0000_0047_0000,
        0x5c10_0000_1017_0000,
    ] {
        assert!(
            decode_add(STAGE, 8, word, 4, &mut 4, &mut None).is_err(),
            "{word:016x}"
        );
    }
    assert!(!is_add(0x5c18_0000_0017_0000)); // ISCADD is a distinct instruction.
}

#[test]
fn lop_forms_complements_and_pass_b_match_integer_semantics() {
    let left = 0xa5a5_1234_u32;
    for (encoding, right) in [
        (0x5c47_0000_0017_0000_u64, 0x8000_2468_u32),
        (0x4c47_0000_0037_0000 | (2 << 34), 0x8000_2468),
        (0x3847_0000_0027_0000, 2),
        (0x3947_007f_fff7_0000, u32::MAX),
        (0x3947_0000_0007_0000, 0xfff8_0000),
    ] {
        assert!(is_bitwise(encoding));
        for selector in 0..4 {
            for inversion in 0..4 {
                let a = if inversion & 1 != 0 { !left } else { left };
                let b = if inversion & 2 != 0 { !right } else { right };
                let expected = match selector {
                    0 => a & b,
                    1 => a | b,
                    2 => a ^ b,
                    _ => b,
                };
                let word = encoding | (selector << 41) | (inversion << 39);
                assert_eq!(evaluate_lop(word, left, right), expected, "{word:016x}");
            }
        }
    }
    // RZ is zero on reads and discards writes; PASS_B does not require A.
    assert_eq!(evaluate_lop(0x3847_0000_0027_ff00, left, 0), 0);
    assert_eq!(evaluate_lop(0x5c47_0000_0ff7_0000, left, 0), 0);
    assert_eq!(evaluate_lop(0x3847_0000_0027_00ff, left, 0), left);
    assert_eq!(evaluate_lop(0x3847_0600_0027_fe00, left, 0), 2);
}

#[test]
fn lop_rejects_flags_predicate_outputs_reserved_bits_and_bad_registers() {
    let captured = 0x3847_0000_0027_0000_u64;
    for (word, detail) in [
        (captured | (1 << 47), "LOP condition-code write"),
        (captured | (1 << 43), "LOP extended condition-code input"),
        (captured & !(7 << 48), "LOP predicate output"),
        (captured | (1 << 44), "LOP predicate output"),
    ] {
        assert!(matches!(decode_bitwise(STAGE, 8, word, 4, &mut 4),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: actual, .. }) if actual == detail));
    }
    for word in [
        captured | (1 << 46),
        0x5c47_0000_1017_0000,
        captured | 4,
        captured | (4 << 8),
    ] {
        assert!(
            decode_bitwise(STAGE, 8, word, 4, &mut 4).is_err(),
            "{word:016x}"
        );
    }
}

#[test]
fn isetp_register_immediate_and_constant_forms() {
    let captured = 0x5b64_0380_0ff7_0007;
    let decoded = decode_set_predicate(STAGE, 0x10, captured, 5, &mut 5).unwrap();
    assert!(matches!(
        decoded.operations.as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits: 0, .. },
            ShaderOperation::SetPredicateInteger32 {
                destinations: [Some(0), None],
                signed: false,
                comparison: ShaderIntegerComparison::Equal,
                accumulator: ShaderPredicate::Always,
                set_operation: ShaderPredicateSetOperation::And,
                ..
            }
        ]
    ));
    for (encoding, bits) in [
        (0x3664_0380_0017_0007, 1),
        (0x3764_03ff_fff7_0007, u32::MAX),
        (0x3764_0380_0007_0007, 0xfff8_0000),
    ] {
        assert!(is_set_predicate(encoding));
        let decoded = decode_set_predicate(STAGE, 8, encoding, 5, &mut 5).unwrap();
        assert!(
            matches!(decoded.operations[0], ShaderOperation::MoveImmediate32 { bits: actual, .. } if actual == bits)
        );
    }
    let encoding = 0x4b65_0380_0037_0007 | (2 << 34);
    let decoded = decode_set_predicate(STAGE, 8, encoding, 5, &mut 5).unwrap();
    assert_eq!(decoded.constant_buffer_binding, Some(2));
    assert!(matches!(
        decoded.operations[0],
        ShaderOperation::LoadConstantBuffer32 {
            binding: 2,
            byte_offset: 12,
            ..
        }
    ));
    assert!(matches!(
        decoded.operations[1],
        ShaderOperation::SetPredicateInteger32 { signed: true, .. }
    ));
    for bits in [1 << 6, 1 << 7, 1 << 28, 1 << 38, 1 << 44, 1 << 47, 3 << 45] {
        assert!(
            decode_set_predicate(STAGE, 8, captured | bits, 5, &mut 5).is_err(),
            "reserved bits {bits:x}"
        );
    }
    assert!(matches!(
        decode_set_predicate(STAGE, 8, captured | (1 << 43), 5, &mut 5),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "ISETP.X condition-code input",
            ..
        })
    ));
}
