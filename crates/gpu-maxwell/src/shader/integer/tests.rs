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
    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::MoveImmediate32 { bits: 1, .. }
            ))
    );
    assert!(
        translated.ir().instructions().iter().any(|instruction| matches!(instruction.operation(), ShaderOperation::Add32 { destination, left, scalar_type: ShaderScalarType::Unsigned32, .. } if destination.index() == 1 && left.index() == 1)
    ));
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
    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::MoveImmediate32 { bits: 2, .. }
            ))
    );
    assert!(translated.ir().instructions().iter().any(|instruction| matches!(instruction.operation(),
        ShaderOperation::Bitwise32 { destination, left, operation: nixe_gpu::ShaderBitwiseOperation::And, .. }
            if destination.index() == 0 && left.index() == 0)));
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

    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::MoveImmediate32 {
                    bits: 4,
                    scalar_type: ShaderScalarType::Unsigned32,
                    ..
                }
            ))
    );
    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::ShiftLeft32 {
                    destination,
                    value,
                    wrap: false,
                    ..
                } if destination.index() == 7 && value.index() == 0
            ))
    );
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
    for flag in [1 << 45, 1 << 44, 1 << 28] {
        assert!(decode_add(STAGE, 8, 0x5c18_0280_0017_0000 | flag, 4, &mut 4, &mut None).is_err());
    }
}

#[test]
fn iscadd_negation_and_plus_one_preserve_wrapping_aliases_and_carry() {
    for shift in 0..32 {
        for (a, b) in [
            (0_u32, u32::MAX),
            (0x80000001, 17),
            (0xffffffff, 0xfffffffe),
            (7, 3),
        ] {
            for (neg_a, neg_b) in [(false, true), (true, false), (true, true)] {
                let encoding = 0x5c18_0000_0017_0000
                    | (u64::from(shift) << 39)
                    | (u64::from(neg_a) << 49)
                    | (u64::from(neg_b) << 48);
                let decoded = decode_add(STAGE, 8, encoding, 4, &mut 4, &mut None).unwrap();
                let (left, right) = if neg_a && neg_b {
                    (a, b.wrapping_add(1))
                } else {
                    (
                        if neg_a { a.wrapping_neg() } else { a },
                        if neg_b { b.wrapping_neg() } else { b },
                    )
                };
                assert_eq!(
                    evaluate_integer(decoded, a, b),
                    left.wrapping_shl(shift).wrapping_add(right)
                );
            }
        }
    }
    // .PO.CC must retain the carry from B+1 even if the final addition has none.
    let mut next = 4;
    let mut carry = None;
    let mut decoded =
        decode_add(STAGE, 8, 0x5c1b_8000_0017_0000, 4, &mut next, &mut carry).unwrap();
    decoded.operations.push(ShaderOperation::Move32 {
        destination: ShaderRegister::new(0),
        source: carry.unwrap(),
        scalar_type: ShaderScalarType::Unsigned32,
    });
    assert_eq!(evaluate_integer(decoded, 0, u32::MAX), 1);
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
    evaluate_integer_with_third(decoded, left, right, 0)
}

fn evaluate_integer_with_third(
    decoded: DecodedIntegerOperation,
    left: u32,
    right: u32,
    third: u32,
) -> u32 {
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
    operations.push(ShaderOperation::MoveImmediate32 {
        destination: ShaderRegister::new(2),
        bits: third,
        scalar_type: ShaderScalarType::Unsigned32,
    });
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
fn lop_rejects_unimplemented_flags_reserved_bits_and_bad_registers() {
    let captured = 0x3847_0000_0027_0000_u64;
    for (word, detail) in [
        (captured | (1 << 47), "LOP condition-code write"),
        (captured | (1 << 43), "LOP extended condition-code input"),
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

#[test]
fn shift_right_forms_modifiers_boundaries_and_aliased_destination() {
    for value in [0_u32, 1, 0x8000_0000, 0x7654_3210, u32::MAX] {
        for count in [0_u32, 1, 7, 31, 32, 33, 63, u32::MAX] {
            for flags in 0..8_u64 {
                let signed = flags & 1 != 0;
                let reverse = flags & 2 != 0;
                let wrap = flags & 4 != 0;
                let modifiers =
                    ((flags & 1) << 48) | (((flags >> 1) & 1) << 40) | (((flags >> 2) & 1) << 39);
                let base = if reverse { value.reverse_bits() } else { value };
                let expected = if signed {
                    ((base as i32) >> if wrap { count & 31 } else { count.min(31) }) as u32
                } else if !wrap && count >= 32 {
                    0
                } else {
                    base >> (count & 31)
                };
                let immediate = (u64::from(count & 0x7ffff) << 20)
                    | if count & 0x80000 != 0 { 1 << 56 } else { 0 };
                for word in [
                    0x5c28_0000_0017_0000,
                    0x4c28_0008_0037_0000,
                    0x3828_0000_0007_0000 | immediate,
                ] {
                    let word = word | modifiers;
                    assert!(is_shift_right(word));
                    let decoded = decode_shift_right(STAGE, 8, word, 4, &mut 4).unwrap();
                    let result = evaluate_integer(
                        DecodedIntegerOperation {
                            operations: decoded.operations,
                            constant_buffer_binding: decoded.constant_buffer_binding,
                        },
                        value,
                        count,
                    );
                    assert_eq!(
                        result, expected,
                        "value={value:08x}, count={count}, word={word:016x}"
                    );
                }
            }
        }
    }
    for word in [
        0x5c28_0000_0ff7_0000,
        0x5c28_0000_0017_ff00,
        0x3828_0000_0017_00ff,
    ] {
        let decoded = decode_shift_right(STAGE, 8, word, 4, &mut 4).unwrap();
        let result = evaluate_integer(
            DecodedIntegerOperation {
                operations: decoded.operations,
                constant_buffer_binding: decoded.constant_buffer_binding,
            },
            16,
            1,
        );
        assert_eq!(
            result,
            if word as u8 == 255 {
                16
            } else if (word >> 8) as u8 == 255 {
                0
            } else {
                16
            }
        );
    }
    for flag in [1 << 47, 1 << 43] {
        assert!(decode_shift_right(STAGE, 8, 0x3828_0000_0007_0000 | flag, 4, &mut 4).is_err());
    }
}

#[test]
fn iadd_negation_plus_one_and_extended_complement() {
    for left in [0_u32, 1, 0x8000_0000, u32::MAX] {
        for right in [0_u32, 1, 0x7654_3210, u32::MAX] {
            for flags in 1..4_u64 {
                let expected = match flags {
                    1 => left.wrapping_sub(right),
                    2 => right.wrapping_sub(left),
                    _ => right.wrapping_sub(left).wrapping_add(1),
                };
                for word in [0x5c10_0000_0017_0000, 0x4c10_0008_0037_0000] {
                    assert_eq!(evaluate_iadd(word | (flags << 48), left, right), expected);
                }
            }
            for carry in 0..=1_u32 {
                for word in [
                    0x5c11_0800_0017_0000,
                    0x5c12_0800_0017_0000,
                    0x1d20_0000_0007_0000 | (u64::from(right) << 20),
                ] {
                    let mut decoded =
                        decode_add(STAGE, 8, word, 4, &mut 4, &mut Some(ShaderRegister::new(2)))
                            .unwrap();
                    decoded.operations.insert(
                        0,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(2),
                            bits: carry,
                            scalar_type: ShaderScalarType::Unsigned32,
                        },
                    );
                    let expected = if word >> 48 == 0x5c11 {
                        left.wrapping_add(!right)
                    } else {
                        (!left).wrapping_add(right)
                    }
                    .wrapping_add(carry);
                    assert_eq!(
                        evaluate_integer(decoded, left, right),
                        expected,
                        "word={word:016x}"
                    );
                }
            }
        }
        assert_eq!(
            evaluate_iadd(0x1d80_0000_0057_0000, left, 0),
            left.wrapping_add(6)
        );
        assert_eq!(
            evaluate_iadd(0x1d00_0000_0057_0000, left, 0),
            5_u32.wrapping_sub(left)
        );
    }
    for flag in [1 << 47, 1 << 43] {
        assert!(decode_add(STAGE, 8, 0x5c13_0000_0017_0000 | flag, 4, &mut 4, &mut None).is_err());
    }
}

#[test]
fn psetp_truth_tables_negations_aliases_and_instruction_predicate() {
    let combine = |a: bool, b: bool, op| match op {
        0 => a && b,
        1 => a || b,
        2 => a ^ b,
        _ => unreachable!(),
    };
    for sources in [[0_u64, 1, 2], [7, 1, 2], [0, 7, 7]] {
        for state in 0..8_u32 {
            for invert in 0..8_u64 {
                for first in 0..3_u64 {
                    for second in 0..3_u64 {
                        for destinations in [[0_u64, 1], [1, 0], [2, 0], [7, 0], [0, 7]] {
                            for guard in [
                                ShaderPredicate::Always,
                                ShaderPredicate::Register {
                                    register: 0,
                                    inverted: false,
                                },
                                ShaderPredicate::Register {
                                    register: 0,
                                    inverted: true,
                                },
                            ] {
                                let word = 0x5090_0000_0007_0000
                                    | (destinations[0] << 3)
                                    | destinations[1]
                                    | (sources[0] << 12)
                                    | ((invert & 1) << 15)
                                    | (first << 24)
                                    | (sources[1] << 29)
                                    | (((invert >> 1) & 1) << 32)
                                    | (sources[2] << 39)
                                    | (((invert >> 2) & 1) << 42)
                                    | (second << 45);
                                let decoded = decode_predicate_boolean(STAGE, 8, word).unwrap();
                                let old = [state & 1 != 0, state & 2 != 0, state & 4 != 0];
                                let mut ops = vec![(
                                    ShaderPredicate::Always,
                                    ShaderOperation::MoveImmediate32 {
                                        destination: ShaderRegister::new(0),
                                        bits: 0,
                                        scalar_type: ShaderScalarType::Unsigned32,
                                    },
                                )];
                                for (index, value) in old.iter().enumerate() {
                                    ops.push((
                                        ShaderPredicate::Always,
                                        ShaderOperation::SetPredicateInteger32 {
                                            destinations: [Some(index as u8), None],
                                            left: ShaderRegister::new(0),
                                            right: ShaderRegister::new(0),
                                            signed: false,
                                            comparison: if *value {
                                                ShaderIntegerComparison::True
                                            } else {
                                                ShaderIntegerComparison::False
                                            },
                                            accumulator: ShaderPredicate::Always,
                                            set_operation: ShaderPredicateSetOperation::And,
                                        },
                                    ));
                                }
                                ops.push((guard, decoded));
                                for index in 0..3_u8 {
                                    ops.push((
                                        ShaderPredicate::Always,
                                        ShaderOperation::CompareInteger32 {
                                            destination: ShaderRegister::new(0),
                                            float_result: false,
                                            left: ShaderRegister::new(0),
                                            right: ShaderRegister::new(0),
                                            signed: false,
                                            comparison: ShaderIntegerComparison::True,
                                            accumulator: ShaderPredicate::Register {
                                                register: index,
                                                inverted: false,
                                            },
                                            set_operation: ShaderPredicateSetOperation::And,
                                        },
                                    ));
                                    ops.push((
                                        ShaderPredicate::Always,
                                        ShaderOperation::StoreOutput {
                                            sources: vec![ShaderRegister::new(0)].into(),
                                            location: ShaderIoLocation::Color(0),
                                            first_component: index,
                                            scalar_type: ShaderScalarType::Unsigned32,
                                        },
                                    ));
                                }
                                ops.push((ShaderPredicate::Always, ShaderOperation::Exit));
                                let shader = VerifiedShaderIr::verify(ShaderIr::new(
                                    ShaderStage::Fragment,
                                    vec![],
                                    (0..3)
                                        .map(|component| {
                                            ShaderInterfaceElement::new(
                                                ShaderIoLocation::Color(0),
                                                component,
                                                ShaderScalarType::Unsigned32,
                                                None,
                                            )
                                            .unwrap()
                                        })
                                        .collect(),
                                    vec![],
                                    ops.into_iter()
                                        .enumerate()
                                        .map(|(i, (predicate, operation))| {
                                            ShaderInstruction::new(
                                                ShaderSourceLocation::new(i as u32 * 8),
                                                predicate,
                                                operation,
                                            )
                                        })
                                        .collect::<Vec<_>>(),
                                ))
                                .unwrap();
                                let mut expected = old;
                                let enabled = match guard {
                                    ShaderPredicate::Always => true,
                                    ShaderPredicate::Register { inverted, .. } => old[0] ^ inverted,
                                    _ => unreachable!(),
                                };
                                if enabled {
                                    let values = sources
                                        .map(|p| if p == 7 { true } else { old[p as usize] });
                                    let [a, b, c] = std::array::from_fn(|i| {
                                        values[i] ^ (invert & (1 << i) != 0)
                                    });
                                    for (i, destination) in destinations.into_iter().enumerate() {
                                        if destination != 7 {
                                            expected[destination as usize] = combine(
                                                combine(if i == 0 { a } else { !a }, b, first),
                                                c,
                                                second,
                                            );
                                        }
                                    }
                                }
                                let result = nixe_gpu::evaluate_shader_ir(
                                    &shader,
                                    &nixe_gpu::ShaderEvaluationInputs::default(),
                                    32,
                                )
                                .unwrap();
                                for (i, value) in expected.iter().enumerate() {
                                    assert_eq!(
                                        result.output_bits(ShaderIoLocation::Color(0), i as u8),
                                        Some(if *value { u32::MAX } else { 0 }),
                                        "{word:016x} state={state} guard={guard:?}"
                                    );
                                }
                                if state == 0
                                    && invert == 0
                                    && sources == [0, 1, 2]
                                    && destinations == [0, 1]
                                {
                                    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn psetp_captured_encoding_and_reserved_fields() {
    let word = 0x5090_0380_e007_9007;
    assert!(is_predicate_boolean(word));
    assert!(matches!(
        decode_predicate_boolean(STAGE, 0x18, word).unwrap(),
        ShaderOperation::SetPredicateBoolean {
            destinations: [Some(0), None],
            inputs: [
                ShaderPredicate::Register {
                    register: 1,
                    inverted: true
                },
                ShaderPredicate::Always,
                ShaderPredicate::Always
            ],
            operations: [
                ShaderPredicateSetOperation::And,
                ShaderPredicateSetOperation::And
            ],
        }
    ));
    for field in [
        1 << 6,
        1 << 8,
        1 << 20,
        1 << 27,
        1 << 33,
        1 << 43,
        1 << 47,
        3 << 24,
        3 << 45,
    ] {
        assert!(
            decode_predicate_boolean(STAGE, 8, word | field).is_err(),
            "{field:x}"
        );
    }
    assert!(decode_predicate_boolean(STAGE, 8, word & !7).is_err());
}

#[test]
fn psetp_vertex_program_reaches_verified_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0006_0461;
    header[13] = 0x0000_8000;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0x5b64_0380_0ff7_000f,
            0x5090_0380_e007_9007,
            0xe300_0000_0007_000f,
        ],
        4,
    );
    assert!(
        translated
            .ir()
            .instructions()
            .iter()
            .any(|i| matches!(i.operation(), ShaderOperation::SetPredicateBoolean { .. }))
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&translated).unwrap());
}

#[test]
fn r2p_selected_bytes_masks_preservation_and_predication() {
    for byte in 0..4_u64 {
        for mask in 0..128_u64 {
            for bits in [
                0_u32,
                u32::MAX,
                0x12345678,
                0xa5a5a5a5,
                0x01020408,
                0x80402010,
            ] {
                for old in [0_u8, 0x7f, 0x55] {
                    for inverted in [None, Some(false), Some(true)] {
                        let word = 0x38f0_0000_0007_0000 | (mask << 20) | (byte << 41);
                        let decoded =
                            decode_register_to_predicate(STAGE, 0x238, word, 2, &mut 2).unwrap();
                        let guard = inverted.map_or(ShaderPredicate::Always, |inverted| {
                            ShaderPredicate::Register {
                                register: 0,
                                inverted,
                            }
                        });
                        let mut ops = vec![(
                            ShaderPredicate::Always,
                            ShaderOperation::MoveImmediate32 {
                                destination: ShaderRegister::new(0),
                                bits,
                                scalar_type: ShaderScalarType::Unsigned32,
                            },
                        )];
                        for index in 0..7_u8 {
                            ops.push((
                                ShaderPredicate::Always,
                                ShaderOperation::SetPredicateInteger32 {
                                    destinations: [Some(index), None],
                                    left: ShaderRegister::new(0),
                                    right: ShaderRegister::new(0),
                                    signed: false,
                                    comparison: if old & (1 << index) != 0 {
                                        ShaderIntegerComparison::True
                                    } else {
                                        ShaderIntegerComparison::False
                                    },
                                    accumulator: ShaderPredicate::Always,
                                    set_operation: ShaderPredicateSetOperation::And,
                                },
                            ));
                        }
                        ops.extend(decoded.into_iter().map(|op| (guard, op)));
                        for index in 0..7_u8 {
                            ops.push((
                                ShaderPredicate::Always,
                                ShaderOperation::CompareInteger32 {
                                    destination: ShaderRegister::new(1),
                                    float_result: false,
                                    left: ShaderRegister::new(0),
                                    right: ShaderRegister::new(0),
                                    signed: false,
                                    comparison: ShaderIntegerComparison::True,
                                    accumulator: ShaderPredicate::Register {
                                        register: index,
                                        inverted: false,
                                    },
                                    set_operation: ShaderPredicateSetOperation::And,
                                },
                            ));
                            ops.push((
                                ShaderPredicate::Always,
                                ShaderOperation::StoreOutput {
                                    sources: vec![ShaderRegister::new(1)].into(),
                                    location: ShaderIoLocation::Color(index / 4),
                                    first_component: index % 4,
                                    scalar_type: ShaderScalarType::Unsigned32,
                                },
                            ));
                        }
                        ops.push((ShaderPredicate::Always, ShaderOperation::Exit));
                        let shader = VerifiedShaderIr::verify(ShaderIr::new(
                            ShaderStage::Fragment,
                            vec![],
                            (0..7_u8)
                                .map(|i| {
                                    ShaderInterfaceElement::new(
                                        ShaderIoLocation::Color(i / 4),
                                        i % 4,
                                        ShaderScalarType::Unsigned32,
                                        None,
                                    )
                                    .unwrap()
                                })
                                .collect(),
                            vec![],
                            ops.into_iter()
                                .enumerate()
                                .map(|(i, (predicate, operation))| {
                                    ShaderInstruction::new(
                                        ShaderSourceLocation::new(i as u32 * 8),
                                        predicate,
                                        operation,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        ))
                        .unwrap();
                        let enabled = inverted.is_none_or(|invert| (old & 1 != 0) ^ invert);
                        let mask = if enabled { mask as u8 } else { 0 };
                        let expected = (old & !mask) | (((bits >> (byte * 8)) as u8) & mask);
                        let result = nixe_gpu::evaluate_shader_ir(
                            &shader,
                            &nixe_gpu::ShaderEvaluationInputs::default(),
                            64,
                        )
                        .unwrap();
                        for i in 0..7_u8 {
                            assert_eq!(
                                result.output_bits(ShaderIoLocation::Color(i / 4), i % 4),
                                Some(if expected & (1 << i) != 0 {
                                    u32::MAX
                                } else {
                                    0
                                }),
                                "{word:016x} bits={bits:08x} old={old:02x} guard={guard:?}"
                            );
                        }
                        if mask == 127 && bits == 0 && old == 0 {
                            validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn r2p_captured_zero_source_and_reserved_fields() {
    let word = 0x38f0_0000_0037_0700;
    assert!(is_register_to_predicate(word));
    assert!(
        matches!(decode_register_to_predicate(STAGE, 0x238, word, 8, &mut 8).unwrap().as_slice(), [ShaderOperation::SetPredicatesFromBits { source, first_bit: 0, write_mask: 3 }] if source.index() == 7)
    );
    assert!(matches!(
        decode_register_to_predicate(STAGE, 8, word | 0xff00, 8, &mut 8)
            .unwrap()
            .as_slice(),
        [
            ShaderOperation::MoveImmediate32 { bits: 0, .. },
            ShaderOperation::SetPredicatesFromBits { .. }
        ]
    ));
    for bits in [1, 1 << 43, 1 << 47] {
        assert!(decode_register_to_predicate(STAGE, 8, word | bits, 8, &mut 8).is_err());
    }
    assert!(decode_register_to_predicate(STAGE, 8, word | (1 << 40), 8, &mut 8).is_err());
    // Higher immediate bits, including sign extension, cannot write PT or general registers.
    assert!(matches!(
        decode_register_to_predicate(STAGE, 8, word | (0x7ffff << 20) | (1 << 56), 8, &mut 8)
            .unwrap()[0],
        ShaderOperation::SetPredicatesFromBits {
            write_mask: 127,
            ..
        }
    ));
    let mut header = [0; 20];
    header[0] = 0x0006_0461;
    header[13] = 0xf << 12;
    let shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[0, word, 0xe300_0000_0007_000f],
        8,
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
}

#[test]
fn icmp_signed_unsigned_truth_tables_aliases_and_all_operand_forms() {
    let compare = |value: u32, signed: bool, mode| {
        let value = if signed {
            i64::from(value as i32)
        } else {
            i64::from(value)
        };
        match mode {
            0 => false,
            1 => value < 0,
            2 => value == 0,
            3 => value <= 0,
            4 => value > 0,
            5 => value != 0,
            6 => value >= 0,
            7 => true,
            _ => unreachable!(),
        }
    };
    for (opcode, operand) in [
        (0x5b40_u64, 1 << 20),
        (0x3640, 0x7ffff << 20 | 1 << 56),
        (0x4b40, 2 << 34 | 3 << 20),
        (0x5340, 2 << 34 | 3 << 20),
    ] {
        for mode in 0..8_u64 {
            for signed in [false, true] {
                for tested in [0_u32, 1, u32::MAX, 0x80000000, 0x7fffffff] {
                    for destination in [0_u8, 1, 2, 255] {
                        let left = 0x7fc01234; // Preserve raw NaN payload bits, without float comparison.
                        let right = if opcode == 0x5340 { tested } else { 0x81234567 };
                        let third = if opcode == 0x5340 { 0x89abcdef } else { tested };
                        let word = opcode << 48
                            | (mode << 49)
                            | (u64::from(signed) << 48)
                            | (2 << 39)
                            | operand
                            | (7 << 16)
                            | u64::from(destination);
                        assert!(is_integer_compare(word));
                        let mut decoded =
                            decode_integer_compare(STAGE, 8, word, 4, &mut 4).unwrap();
                        if destination != 0 && destination != 255 {
                            decoded.operations.push(ShaderOperation::Move32 {
                                destination: ShaderRegister::new(0),
                                source: ShaderRegister::new(u16::from(destination)),
                                scalar_type: ShaderScalarType::Unsigned32,
                            });
                        }
                        let no = match opcode {
                            0x3640 => u32::MAX,
                            0x5340 => third,
                            _ => right,
                        };
                        let expected = if destination == 255 || compare(tested, signed, mode) {
                            left
                        } else {
                            no
                        };
                        assert_eq!(
                            evaluate_integer_with_third(decoded, left, right, third),
                            expected,
                            "{word:016x} tested={tested:x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn icmp_captured_zero_source_reaches_gpu_selection() {
    let word = 0x364d_0b00_0017_ff19;
    let decoded = decode_integer_compare(STAGE, 0x508, word, 32, &mut 32).unwrap();
    assert!(matches!(
        &decoded.operations[decoded.operations.len() - 2],
        ShaderOperation::CompareInteger32 {
            signed: true,
            comparison: ShaderIntegerComparison::GreaterOrEqual,
            ..
        }
    ));
    assert!(
        matches!(decoded.operations.last().unwrap(), ShaderOperation::Select32 { destination, .. } if destination.index() == 25)
    );
    assert!(decode_integer_compare(STAGE, 8, word | (1 << 47), 32, &mut 32).is_err());
    assert!(
        decode_integer_compare(STAGE, 8, 0x5b40_0000_0017_0000 | (1 << 28), 32, &mut 32).is_err()
    );
    let mut header = [0; 20];
    header[0] = 0x0006_0461;
    header[13] = 0xf << 12;
    let shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[0, word, 0xeff0_7f80_0707_ff19, 0xe300_0000_0007_000f],
        32,
    );
    let module = lower_shader_ir_to_wgsl(&shader).unwrap();
    assert!(module.source().contains("= select(registers["));
    validate_wgsl(&module);
}

#[test]
fn iadd3_halves_negations_shifts_and_aliases_preserve_wrapping_sums() {
    let half = |v: u32, h| match h {
        0 => v,
        1 => v & 65535,
        2 => v >> 16,
        _ => unreachable!(),
    };
    for ha in 0..3_u64 {
        for hb in 0..3_u64 {
            for hc in 0..3_u64 {
                for shift in 0..3_u64 {
                    for negations in 0..8_u64 {
                        for values in [
                            [0_u32; 3],
                            [u32::MAX, u32::MAX, 0],
                            [0x12345678, 0x89abcdef, 0x80000000],
                            [1, 2, u32::MAX],
                            [0xffff0000, 0xffff, 0x76543210],
                        ] {
                            for destination in [0_u8, 1, 2, 255] {
                                let word = 0x5cc0_0000_0017_0000
                                    | (2 << 39)
                                    | (ha << 35)
                                    | (hb << 33)
                                    | (hc << 31)
                                    | (shift << 37)
                                    | ((negations & 1) << 51)
                                    | (((negations >> 1) & 1) << 50)
                                    | (((negations >> 2) & 1) << 49)
                                    | u64::from(destination);
                                let mut decoded =
                                    decode_add_three(STAGE, 8, word, 4, &mut 4).unwrap();
                                if destination != 0 && destination != 255 {
                                    decoded.operations.push(ShaderOperation::Move32 {
                                        destination: ShaderRegister::new(0),
                                        source: ShaderRegister::new(u16::from(destination)),
                                        scalar_type: ShaderScalarType::Unsigned32,
                                    });
                                }
                                let parts = [ha, hb, hc]
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, h)| {
                                        let v = half(values[i], h);
                                        if negations & (1 << i) != 0 {
                                            v.wrapping_neg()
                                        } else {
                                            v
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                let sum = u64::from(parts[0]) + u64::from(parts[1]);
                                let sum = match shift {
                                    0 => sum as u32,
                                    1 => (sum >> 16) as u32,
                                    2 => (sum as u32) << 16,
                                    _ => unreachable!(),
                                };
                                let expected = if destination == 255 {
                                    values[0]
                                } else {
                                    sum.wrapping_add(parts[2])
                                };
                                assert_eq!(
                                    evaluate_integer_with_third(
                                        decoded, values[0], values[1], values[2]
                                    ),
                                    expected,
                                    "{word:016x} values={values:x?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn iadd3_immediate_constant_zero_sources_and_captured_negated_sum() {
    for (word, rhs) in [
        (0x38c0_0100_0057_0000, 5_u32),
        (0x39c0_0100_0057_0000, 0xfff80005),
        (0x4cc0_0108_0037_0000, 77),
    ] {
        assert!(is_add_three(word));
        let decoded = decode_add_three(STAGE, 8, word, 4, &mut 4).unwrap();
        assert_eq!(
            evaluate_integer_with_third(decoded, 17, if word >> 48 == 0x4cc0 { 77 } else { 0 }, 42),
            17_u32.wrapping_add(rhs).wrapping_add(42)
        );
    }
    let word = 0x5cc6_0c80_0227_ff18;
    let decoded = decode_add_three(STAGE, 0x570, word, 40, &mut 40).unwrap();
    assert!(matches!(
        decoded.operations[0],
        ShaderOperation::MoveImmediate32 { bits: 0, .. }
    ));
    for bits in [1 << 47, 1 << 48, 1 << 28, 3 << 35, 3 << 37] {
        assert!(decode_add_three(STAGE, 8, word | bits, 40, &mut 40).is_err());
    }
    let mut header = [0; 20];
    header[0] = 0x0006_0461;
    header[13] = 0xf << 12;
    let shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[0, word, 0xeff0_7f80_0707_ff18, 0xe300_0000_0007_000f],
        40,
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
}
