//! Synthetic ISA regressions, independent of application shader binaries.
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::*;

fn evaluate(
    operations: Vec<ShaderOperation>,
    inputs: &[u32],
    extra: Vec<ShaderInstruction>,
) -> u32 {
    let r = ShaderRegister::new;
    let mut instructions = inputs
        .iter()
        .enumerate()
        .map(|(i, &bits)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(8 + i as u32 * 8),
                ShaderPredicate::Always,
                ShaderOperation::MoveImmediate32 {
                    destination: r(i as u16),
                    bits,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            )
        })
        .collect::<Vec<_>>();
    instructions.extend(operations.into_iter().map(|operation| {
        ShaderInstruction::new(
            ShaderSourceLocation::new(80),
            ShaderPredicate::Always,
            operation,
        )
    }));
    instructions.extend(extra);
    instructions.extend([
        ShaderInstruction::new(
            ShaderSourceLocation::new(88),
            ShaderPredicate::Always,
            ShaderOperation::StoreOutput {
                sources: vec![r(0)].into(),
                location: ShaderIoLocation::Color(0),
                first_component: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ),
        ShaderInstruction::new(
            ShaderSourceLocation::new(96),
            ShaderPredicate::Always,
            ShaderOperation::Exit,
        ),
    ]);
    let ir = VerifiedShaderIr::verify(ShaderIr::new(
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
        vec![],
        instructions,
    ))
    .unwrap();
    let wgsl = lower_shader_ir_to_wgsl(&ir).unwrap();
    test_support::validate_wgsl(&wgsl);
    evaluate_shader_ir(&ir, &ShaderEvaluationInputs::default(), 256)
        .unwrap()
        .output_bits(ShaderIoLocation::Color(0), 0)
        .unwrap()
}

#[test]
fn xmad_selected_halves_signed_products_addend_modes_and_merge() {
    let [a, b, c] = [0x8123_fffe_u32, 0xfedc_8001, 0x3456_789a];
    for ah in [false, true] {
        for bh in [false, true] {
            for sa in [false, true] {
                for sb in [false, true] {
                    for mode in [0_u64, 1, 2, 4] {
                        for psl in [false, true] {
                            for merge in [false, true] {
                                let word = 0x5b00_0000_0027_0100
                                    | (3 << 39)
                                    | (u64::from(ah) << 53)
                                    | (u64::from(bh) << 35)
                                    | (u64::from(sa) << 48)
                                    | (u64::from(sb) << 49)
                                    | (mode << 50)
                                    | (u64::from(psl) << 36)
                                    | (u64::from(merge) << 37);
                                let ops = xmad::decode_xmad(
                                    MaxwellShaderStage::Pixel,
                                    8,
                                    word,
                                    4,
                                    &mut 4,
                                )
                                .unwrap()
                                .operations;
                                let half = |v: u32, high, signed| {
                                    let v = (if high { v >> 16 } else { v }) & 0xffff;
                                    if signed { (v as i16 as i32) as u32 } else { v }
                                };
                                let mut product = half(a, ah, sa).wrapping_mul(half(b, bh, sb));
                                if psl {
                                    product <<= 16;
                                }
                                let addend = match mode {
                                    0 => c,
                                    1 => c & 0xffff,
                                    2 => c >> 16,
                                    4 => c.wrapping_add(b << 16),
                                    _ => unreachable!(),
                                };
                                let mut expected = product.wrapping_add(addend);
                                if merge {
                                    expected = (expected & 0xffff) | (b << 16);
                                }
                                assert_eq!(
                                    evaluate(ops, &[0, a, b, c], vec![]),
                                    expected,
                                    "{word:#018x}"
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
fn iadd32i_preserves_every_immediate_bit_and_negates_before_wrapping_add() {
    for left in [0_u32, 1, u32::MAX, 0x8000_0000] {
        for immediate in [0_u32, 0x8000_0001, 0xffff_ffff, 0x1234_5678] {
            for negate in [false, true] {
                let word = 0x1c00_0000_0007_0000
                    | (u64::from(immediate) << 20)
                    | (u64::from(negate) << 56);
                let ops =
                    integer::decode_add(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4, &mut None)
                        .unwrap()
                        .operations;
                assert_eq!(
                    evaluate(ops, &[left], vec![]),
                    (if negate { left.wrapping_neg() } else { left }).wrapping_add(immediate)
                );
            }
        }
    }
}

#[test]
fn fset_writes_float_or_bit_masks_without_clobbering_predicates() {
    for (left, right, ordered, unordered) in [
        (1.0_f32, 2.0_f32, true, true),
        (2.0, 1.0, false, false),
        (f32::NAN, 2.0, false, true),
    ] {
        for (comparison, expected) in [(1, ordered), (9, unordered)] {
            for float_result in [false, true] {
                let word =
                    0x5800_0380_0017_0000 | (comparison << 48) | (u64::from(float_result) << 52);
                let ops = float::decode_float_set(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
                    .unwrap()
                    .operations;
                assert_eq!(
                    evaluate(ops, &[left.to_bits(), right.to_bits()], vec![]),
                    if expected {
                        if float_result {
                            1.0_f32.to_bits()
                        } else {
                            u32::MAX
                        }
                    } else {
                        0
                    }
                );
            }
        }
    }
}

#[test]
fn iset_compares_signed_edges_and_zero_register() {
    for signed in [false, true] {
        let word = 0x5b52_0380_0ff7_0000 | (u64::from(signed) << 48);
        let ops = integer::decode_set(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
            .unwrap()
            .operations;
        assert_eq!(
            evaluate(ops, &[u32::MAX], vec![]),
            if signed { u32::MAX } else { 0 }
        );
    }
}

#[test]
fn lop_predicate_tests_the_computed_result_even_when_data_destination_is_discarded() {
    for value in [0_u32, 3] {
        for (mode, expected) in [(0_u64, false), (1, true), (2, value == 0), (3, value != 0)] {
            let word = 0x5c40_0000_0ff7_00ff | (mode << 44) | (1 << 41);
            let ops = integer::decode_bitwise(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
                .unwrap()
                .operations;
            let extra = vec![ShaderInstruction::new(
                ShaderSourceLocation::new(84),
                ShaderPredicate::Register {
                    register: 0,
                    inverted: false,
                },
                ShaderOperation::MoveImmediate32 {
                    destination: ShaderRegister::new(0),
                    bits: 1,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            )];
            assert_eq!(
                evaluate(ops, &[value], extra),
                if expected { 1 } else { value }
            );
        }
    }
}

#[test]
fn hadd2_register_and_full_immediate_preserve_both_half_lanes_and_aliases() {
    let left = 0x4000_3c00; // (1, 2)
    let right = 0x4400_4200; // (3, 4)
    for word in [
        0x5d10_0000_0017_0000,
        0x2c00_0000_0007_0000 | (u64::from(right) << 20),
    ] {
        let ops = half::decode_half_add(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
            .unwrap()
            .operations;
        assert_eq!(evaluate(ops, &[left, right], vec![]), 0x4600_4400); // (4, 6)
    }
}

#[test]
fn hmul2_immediates_preserve_both_half_lanes_signs_and_aliases() {
    let compact = 0x7800_0000_0007_0000 | (0x108 << 20) | (0x110 << 30);
    for word in [compact, 0x2a00_0000_0007_0000 | (0x4400_4200_u64 << 20)] {
        assert!(half::is_half_multiply(word));
        let ops = half::decode_half_multiply(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
            .unwrap()
            .operations;
        assert_eq!(evaluate(ops, &[0x4000_3c00], vec![]), 0x4800_4200); // (3, 8)
    }
    let ops = half::decode_half_multiply(
        MaxwellShaderStage::Pixel,
        8,
        compact | (1 << 29) | (1 << 56),
        4,
        &mut 4,
    )
    .unwrap()
    .operations;
    assert_eq!(evaluate(ops, &[0x4000_3c00], vec![]), 0xc800_c200);

    // Captured F32 input/output variant, with source/destination aliased to R0.
    let captured = 0x7802_8034_0d07_0000;
    let ops = half::decode_half_multiply(MaxwellShaderStage::Pixel, 8, captured, 4, &mut 4)
        .unwrap()
        .operations;
    assert_eq!(
        evaluate(ops, &[1.0_f32.to_bits()], vec![]),
        0.25_f32.to_bits()
    );
    let ops = half::decode_half_multiply(
        MaxwellShaderStage::Pixel,
        8,
        captured | (1 << 43),
        4,
        &mut 4,
    )
    .unwrap()
    .operations;
    assert_eq!(
        evaluate(ops, &[1.0_f32.to_bits()], vec![]),
        (-0.25_f32).to_bits()
    );
}

#[test]
fn fmul_scale_precedes_multiply_and_source_negation() {
    let left = f32::MAX;
    for (scale, value) in [
        (1_u64, 0.5_f32),
        (2, 0.25),
        (3, 0.125),
        (4, 8.0),
        (5, 4.0),
        (6, 2.0),
    ] {
        let word = 0x5c68_1000_0017_0000 | (scale << 41) | (1 << 48);
        let ops = float::decode_float_multiply(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
            .unwrap()
            .operations;
        assert_eq!(
            evaluate(ops, &[left.to_bits(), 2.0_f32.to_bits()], vec![]),
            ((left * value) * -2.0).to_bits()
        );
    }
}

#[test]
fn hfma2_promoted_operands_keep_fusion_merge_and_destination_aliases() {
    // Captured compact immediate, R0 destination/F32 input, low R1 half addend.
    let captured = (0x7044_8fb4_0d07_1120 & !(0xffff | (0xff << 39))) | (1 << 39);
    let decoded =
        half::decode_half_fused_multiply_add(MaxwellShaderStage::Pixel, 8, captured, 4, &mut 4)
            .unwrap();
    assert_eq!(
        evaluate(decoded.operations, &[1.0_f32.to_bits(), 0x3c00], vec![]),
        0x3f80_3d00
    );
    let full = 0x2820_0000_0007_0000 | (0x4200_4000_u64 << 20);
    let decoded =
        half::decode_half_fused_multiply_add(MaxwellShaderStage::Pixel, 8, full, 4, &mut 4)
            .unwrap();
    assert_eq!(
        evaluate(decoded.operations, &[1.0_f32.to_bits()], vec![]),
        0x44e0_4000
    );
    // Fused result is a tiny negative number, rounded to half -0. Splitting
    // multiply/add would round the product to one first and produce +0.
    let register = 0x5d00_0000_0007_0000
        | (1 << 20)
        | (2 << 39)
        | (1 << 47)
        | (1 << 28)
        | (1 << 35)
        | (1 << 49);
    let decoded =
        half::decode_half_fused_multiply_add(MaxwellShaderStage::Pixel, 8, register, 4, &mut 4)
            .unwrap();
    assert_eq!(
        evaluate(
            decoded.operations,
            &[0x3f80_0001, 0x3f7f_fffe, (-1.0_f32).to_bits()],
            vec![]
        ),
        0x8000_0000
    );

    let mut header = [0; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let ir = test_support::translated_fixture(
        MaxwellShaderStage::Vertex,
        header,
        &[0, register, 0xe300_0000_0007_000f],
    );
    test_support::validate_wgsl(&lower_shader_ir_to_wgsl(&ir).unwrap());
}

#[test]
fn hfma2_half_operands_round_directly_at_midpoints_and_overflow() {
    let word = 0x5d00_0000_0007_0000 | (1 << 20) | (2 << 39);
    for (a, b, c, expected) in [
        (0x3e00, 0x3c01, 0x8001, 0x3e01),
        (0x3e00, 0x3c03, 0x0001, 0x3e05),
        (0x4b80, 0x6c44, 0x8001, 0x7bff),
        (0x4b80, 0x6c44, 0x0001, 0x7c00),
        (0x0001, 0x3800, 0x0000, 0x0000),
        (0x0001, 0x3800, 0x0001, 0x0002),
    ] {
        let decoded =
            half::decode_half_fused_multiply_add(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4)
                .unwrap();
        assert_eq!(evaluate(decoded.operations, &[a, b, c], vec![]), expected);
    }
}

#[test]
fn scheduled_rro_freezes_original_input_and_serves_sine_and_cosine() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_0461;
    header[4] = 0x000f_f000;
    header[6] = 0x0000_0077;
    header[13] = 0x0007_f000;
    let ir = test_support::translated_fixture_with_register_count(
        MaxwellShaderStage::Vertex,
        header,
        &[
            0,
            0x5c90_0000_0007_0002,
            0x0100_0000_0007_f000,
            0x5080_0000_0007_0201,
            0,
            0x5080_0000_0017_0202,
            0xe300_0000_0007_000f,
            0x50b0_0000_0007_0f00,
        ],
        4,
    );
    let snapshot = ir
        .ir()
        .instructions()
        .iter()
        .find_map(|i| match i.operation() {
            ShaderOperation::Move32 {
                destination,
                source,
                ..
            } if source.index() == 0 && destination.index() >= 4 => Some(*destination),
            _ => None,
        })
        .unwrap();
    let functions = ir
        .ir()
        .instructions()
        .iter()
        .filter_map(|i| match i.operation() {
            ShaderOperation::SpecialFunction32 {
                source, function, ..
            } => {
                assert_eq!(*source, snapshot);
                Some(*function)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        functions,
        [ShaderSpecialFunction::Cosine, ShaderSpecialFunction::Sine]
    );
    test_support::validate_wgsl(&lower_shader_ir_to_wgsl(&ir).unwrap());
}

#[test]
fn virtual_registers_above_hardware_limit_evaluate_and_lower_without_truncation() {
    let result = evaluate(
        vec![
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(513),
                bits: 0x1234_abcd,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::Move32 {
                destination: ShaderRegister::new(0),
                source: ShaderRegister::new(513),
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ],
        &[0],
        vec![],
    );
    assert_eq!(result, 0x1234_abcd);
}

#[test]
fn hsetp2_compares_each_lane_and_preserves_an_aliased_accumulator() {
    let r = ShaderRegister::new;
    for (left, right, expected) in [
        (0x3c00_4000, 0x4000_3c00, 1),
        (0x4000_3c00, 0x3c00_4000, 2),
        (0x7e00_7e00, 0, 0),
    ] {
        let mut ops = vec![ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(0), None],
            left: r(0),
            right: r(0),
            signed: false,
            comparison: ShaderIntegerComparison::True,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        }];
        // HSETP2.LT.AND P0,P1,R0,R1,P0; both outputs read old P0.
        ops.extend(
            half::decode_half_set_predicate(
                MaxwellShaderStage::Pixel,
                8,
                0x5d20_0008_0017_0001,
                4,
                &mut 4,
            )
            .unwrap(),
        );
        ops.push(ShaderOperation::MoveImmediate32 {
            destination: r(0),
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        let extra = [(0, 2), (1, 1)].map(|(register, _bits)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(84),
                ShaderPredicate::Register {
                    register,
                    inverted: false,
                },
                ShaderOperation::Bitwise32 {
                    destination: r(0),
                    left: r(0),
                    right: r(2 + register as u16),
                    operation: ShaderBitwiseOperation::Or,
                },
            )
        });
        assert_eq!(evaluate(ops, &[left, right, 2, 1], extra.into()), expected);
    }
}

#[test]
fn hsetp2_preserves_both_results_when_it_overwrites_its_instruction_guard() {
    let r = ShaderRegister::new;
    let encoding = 0x5d20_0388_0010_0001; // @P0 HSETP2.LT.AND P0,P1,R0,R1,PT
    let guard = ShaderPredicate::Register {
        register: 0,
        inverted: false,
    };
    let init = ShaderOperation::SetPredicateInteger32 {
        destinations: [Some(0), None],
        left: r(0),
        right: r(0),
        signed: false,
        comparison: ShaderIntegerComparison::True,
        accumulator: ShaderPredicate::Always,
        set_operation: ShaderPredicateSetOperation::And,
    };
    let mut extra =
        half::decode_half_set_predicate(MaxwellShaderStage::Pixel, 8, encoding, 4, &mut 4)
            .unwrap()
            .into_iter()
            .map(|op| ShaderInstruction::new(ShaderSourceLocation::new(80), guard, op))
            .collect::<Vec<_>>();
    extra.push(ShaderInstruction::new(
        ShaderSourceLocation::new(84),
        ShaderPredicate::Always,
        ShaderOperation::MoveImmediate32 {
            destination: r(0),
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    ));
    for (register, value) in [(0, 2), (1, 3)] {
        extra.push(ShaderInstruction::new(
            ShaderSourceLocation::new(86),
            ShaderPredicate::Register {
                register,
                inverted: false,
            },
            ShaderOperation::Bitwise32 {
                destination: r(0),
                left: r(0),
                right: r(value),
                operation: ShaderBitwiseOperation::Or,
            },
        ));
    }
    // The low lane is false, the high lane true. Publishing P0 first would
    // incorrectly suppress P1, since P0 also guards the original instruction.
    let initialize_secondary = ShaderOperation::SetPredicateInteger32 {
        destinations: [Some(1), None],
        left: r(0),
        right: r(0),
        signed: false,
        comparison: ShaderIntegerComparison::False,
        accumulator: ShaderPredicate::Always,
        set_operation: ShaderPredicateSetOperation::And,
    };
    assert_eq!(
        evaluate(
            vec![init, initialize_secondary],
            &[0x3c00_4000, 0x4000_3c00, 2, 1],
            extra
        ),
        1
    );
}

#[test]
fn bitfield_insert_clamps_position_and_count_and_preserves_unselected_bits() {
    for position in [0u32, 4, 16, 31, 32, 255] {
        for count in [0u32, 1, 2, 16, 32, 255] {
            let control = position | count << 8;
            // Register form, destination aliases the retained third operand.
            let encoding = 0x5bf0000000070002 | (1 << 20) | (2u64 << 39);
            let ops =
                integer::decode_bitfield_insert(MaxwellShaderStage::Pixel, 8, encoding, 4, &mut 4)
                    .unwrap()
                    .operations;
            let mut extra = vec![ShaderInstruction::new(
                ShaderSourceLocation::new(84),
                ShaderPredicate::Always,
                ShaderOperation::Move32 {
                    destination: ShaderRegister::new(0),
                    source: ShaderRegister::new(2),
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            )];
            let mask = if position >= 32 {
                0
            } else {
                u32::MAX
                    .checked_shr(32 - count.min(32 - position))
                    .unwrap_or(0)
                    << position
            };
            let a = 0x81234567u32;
            let c = 0xa5a5a5a5;
            let expected = (c & !mask) | (a.checked_shl(position).unwrap_or(0) & mask);
            assert_eq!(
                evaluate(ops, &[a, control, c, 0], std::mem::take(&mut extra)),
                expected
            );
        }
    }
}
#[test]
fn lop32i_retains_full_immediates_and_complement_modifiers() {
    let immediate = 0xf1234567u32;
    for selector in 0..4u64 {
        for invert_a in [false, true] {
            for invert_b in [false, true] {
                let encoding = 0x0400000000070000
                    | (u64::from(immediate) << 20)
                    | (selector << 53)
                    | ((invert_a as u64) << 55)
                    | ((invert_b as u64) << 56);
                let ops =
                    integer::decode_bitwise(MaxwellShaderStage::Pixel, 8, encoding, 4, &mut 4)
                        .unwrap()
                        .operations;
                let a = if invert_a { !0x98765432u32 } else { 0x98765432 };
                let b = if invert_b { !immediate } else { immediate };
                let expected = match selector {
                    0 => a & b,
                    1 => a | b,
                    2 => a ^ b,
                    _ => b,
                };
                assert_eq!(evaluate(ops, &[0x98765432], vec![]), expected);
            }
        }
    }
}
#[test]
fn unsigned_integer_min_max_clamps_without_float_precision_loss() {
    for (left, expected) in [(0u32, 0u32), (1, 1), (2, 2), (3, 2), (u32::MAX, 2)] {
        let ops = integer::decode_integer_min_max(
            MaxwellShaderStage::Pixel,
            8,
            0x3820038000270000,
            4,
            &mut 4,
        )
        .unwrap()
        .operations;
        assert_eq!(evaluate(ops, &[left], vec![]), expected);
    }
}

#[test]
fn bitfield_extract_preserves_sign_reversal_boundaries_and_operand_aliases() {
    for value in [0_u32, 0x8000_1234, 0x7fff_fedc] {
        for position in [0_u32, 8, 31, 32, 255] {
            for count in [0_u32, 1, 4, 32, 255] {
                for signed in [false, true] {
                    for reverse in [false, true] {
                        let control = position | count << 8;
                        let base = if reverse { value.reverse_bits() } else { value };
                        // Independent bit-by-bit oracle following the documented
                        // NVIDIA extraction/sign-padding rule, including pos >= 32.
                        let sign = signed
                            && count != 0
                            && (base >> (position + count - 1).min(31)) & 1 != 0;
                        let mut expected = 0;
                        for bit in 0..32 {
                            let set = if bit < count && position + bit < 32 {
                                (base >> (position + bit)) & 1 != 0
                            } else {
                                sign
                            };
                            expected |= u32::from(set) << bit;
                        }
                        for form in [0x3800_u64, 0x5c00, 0x4c00] {
                            let mut encoding = (form << 48)
                                | 0x70000
                                | u64::from(signed) << 48
                                | u64::from(reverse) << 40;
                            encoding |= if form == 0x3800 {
                                u64::from(control) << 20
                            } else {
                                1 << 20
                            };
                            let mut decoded = integer::decode_bitfield_extract(
                                MaxwellShaderStage::Pixel,
                                8,
                                encoding,
                                2,
                                &mut 2,
                            )
                            .unwrap();
                            if form == 0x4c00 {
                                assert_eq!(decoded.constant_buffer_binding, Some(0));
                                let ShaderOperation::LoadConstantBuffer32 {
                                    destination,
                                    byte_offset,
                                    ..
                                } = decoded.operations[0]
                                else {
                                    panic!("constant operand expected")
                                };
                                assert_eq!(byte_offset, 4);
                                decoded.operations[0] = ShaderOperation::MoveImmediate32 {
                                    destination,
                                    bits: control,
                                    scalar_type: ShaderScalarType::Unsigned32,
                                };
                            }
                            assert_eq!(
                                evaluate(decoded.operations, &[value, control], vec![]),
                                expected,
                                "value={value:x} pos={position} count={count} signed={signed} reverse={reverse} form={form:x}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn i2i_converts_selected_lanes_modifiers_and_saturates_without_aliasing() {
    // Expected raw register bits include narrow wrapping conversions and the
    // full signed saturation result, as defined by Maxwell rather than Rust casts.
    for (src, selector, src_signed, dst, dst_signed, abs, neg, sat, input, expected) in [
        (0, 0, true, 2, true, false, false, false, 0x80, 0xffff_ff80),
        (0, 0, true, 0, true, false, false, false, 0x80, 0x80),
        (0, 3, true, 1, false, true, false, false, 0x8000_007f, 0x80),
        (
            1,
            2,
            true,
            2,
            true,
            false,
            false,
            false,
            0xff80_00aa,
            0xffff_ff80,
        ),
        (
            1,
            0,
            false,
            2,
            false,
            false,
            false,
            false,
            0xabcd_8001,
            0x8001,
        ),
        (2, 0, true, 0, true, false, false, false, 200, 200),
        (2, 0, true, 0, true, false, false, true, 200, 127),
        (
            2,
            0,
            true,
            0,
            true,
            false,
            false,
            true,
            0xffff_ff7f,
            0xffff_ff80,
        ),
        (2, 0, true, 0, false, false, false, true, u32::MAX, 0),
        (2, 0, false, 0, true, false, false, true, u32::MAX, 127),
        (2, 0, false, 1, false, false, false, true, u32::MAX, 0xffff),
        (
            2,
            0,
            false,
            2,
            true,
            false,
            false,
            true,
            u32::MAX,
            0x7fff_ffff,
        ),
        (0, 0, true, 1, true, false, false, false, 0xff, 0xffff),
        (0, 0, true, 1, true, false, false, true, 0xff, u32::MAX),
        (1, 0, true, 1, true, true, false, false, 0x8000, 0x8000),
        (
            2,
            0,
            true,
            2,
            true,
            true,
            false,
            false,
            0x8000_0000,
            0x8000_0000,
        ),
        (2, 0, false, 2, false, true, false, false, u32::MAX, 1),
        (
            2,
            0,
            true,
            2,
            true,
            true,
            true,
            false,
            0xffff_fff6,
            0xffff_fff6,
        ),
        (
            0,
            1,
            false,
            2,
            false,
            false,
            true,
            false,
            0x12ab,
            0xffff_ffee,
        ),
        (2, 0, true, 0, false, true, true, true, 10, 0),
    ] {
        let fields = (dst << 8)
            | (src << 10)
            | (u64::from(dst_signed) << 12)
            | (u64::from(src_signed) << 13)
            | (selector << 41)
            | (u64::from(neg) << 45)
            | (u64::from(abs) << 49)
            | (u64::from(sat) << 50);
        let word = 0x5ce0_0000_0017_0000 | fields;
        let ops = integer::decode_integer_to_integer(
            MaxwellShaderStage::Pixel,
            8,
            word,
            2,
            &mut 2,
            &mut integer::IntegerConditionCodes::default(),
        )
        .unwrap()
        .operations;
        assert_eq!(
            evaluate(ops, &[0, input], vec![]),
            expected,
            "{word:#018x} {input:#x}"
        );
        // CC is based on the converted result, including narrow truncation
        // and saturation. Preexisting C/O are cleared rather than preserved.
        let mut flags = integer::IntegerConditionCodes {
            zero: Some(ShaderRegister::new(2)),
            sign: Some(ShaderRegister::new(3)),
            carry: Some(ShaderRegister::new(4)),
            overflow: Some(ShaderRegister::new(5)),
        };
        let cc = integer::decode_integer_to_integer(
            MaxwellShaderStage::Pixel,
            8,
            word | (1 << 47),
            2,
            &mut 6,
            &mut flags,
        )
        .unwrap();
        for (flag, value) in [
            (flags.zero.unwrap(), u32::from(expected == 0)),
            (flags.sign.unwrap(), expected >> 31),
            (flags.carry.unwrap(), 0),
            (flags.overflow.unwrap(), 0),
        ] {
            let read_flag = ShaderInstruction::new(
                ShaderSourceLocation::new(84),
                ShaderPredicate::Always,
                ShaderOperation::Move32 {
                    destination: ShaderRegister::new(0),
                    source: flag,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            );
            assert_eq!(
                evaluate(
                    cc.operations.clone(),
                    &[0, input, 1, 1, 1, 1],
                    vec![read_flag.clone()]
                ),
                value
            );
            let mut disabled = cc
                .operations
                .iter()
                .cloned()
                .map(|operation| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(80),
                        ShaderPredicate::Never,
                        operation,
                    )
                })
                .collect::<Vec<_>>();
            disabled.push(read_flag);
            assert_eq!(evaluate(vec![], &[0, input, 1, 1, 1, 1], disabled), 1);
        }
        // In-place source/destination conversion must consume the old value.
        let alias = 0x5ce0_0000_0007_0000 | fields;
        let ops = integer::decode_integer_to_integer(
            MaxwellShaderStage::Pixel,
            8,
            alias,
            1,
            &mut 1,
            &mut integer::IntegerConditionCodes::default(),
        )
        .unwrap()
        .operations;
        assert_eq!(evaluate(ops, &[input], vec![]), expected);
    }
}

#[test]
fn i2i_immediate_constant_buffer_and_zero_operands_decode_by_family() {
    for (word, expected) in [
        (0x38e0_0000_0077_3a00, 7),
        (0x39e0_007f_fff7_3a00, u32::MAX),
        (0x5ce0_0000_0ff7_3a00, 0),
    ] {
        assert!(integer::is_integer_to_integer(word));
        let decoded = integer::decode_integer_to_integer(
            MaxwellShaderStage::Pixel,
            8,
            word,
            1,
            &mut 1,
            &mut integer::IntegerConditionCodes::default(),
        )
        .unwrap();
        assert_eq!(decoded.constant_buffer_binding, None);
        assert_eq!(evaluate(decoded.operations, &[42], vec![]), expected);
    }
    let decoded = integer::decode_integer_to_integer(
        MaxwellShaderStage::Pixel,
        8,
        0x4ce0_0008_0047_3a00,
        1,
        &mut 1,
        &mut integer::IntegerConditionCodes::default(),
    )
    .unwrap();
    assert_eq!(decoded.constant_buffer_binding, Some(2));
    assert!(decoded.operations.iter().any(|op| matches!(
        op,
        ShaderOperation::LoadConstantBuffer32 {
            binding: 2,
            byte_offset: 16,
            ..
        }
    )));
    for (fields, detail) in [
        (3 << 8, "I2I 64-bit operand"),
        (
            1 << 41,
            "I2I source selector is incompatible with its width",
        ),
    ] {
        assert!(matches!(integer::decode_integer_to_integer(
            MaxwellShaderStage::Pixel, 8, 0x5ce0_0000_0007_3a00 | fields, 1, &mut 1, &mut integer::IntegerConditionCodes::default(),
        ), Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: actual, .. }) if actual == detail));
    }
}

#[test]
fn i2i_cc_discarded_destination_resets_the_shared_extended_add_carry() {
    let mut flags = integer::IntegerConditionCodes::default();
    let mut next = 2;
    let mut ops = integer::decode_add(
        MaxwellShaderStage::Pixel,
        8,
        0x5c10_8000_0017_0000,
        2,
        &mut next,
        &mut flags.carry,
    )
    .unwrap()
    .operations;
    let carry = flags.carry;
    let cc = integer::decode_integer_to_integer(
        MaxwellShaderStage::Pixel,
        16,
        0x5ce0_8000_0017_0aff,
        2,
        &mut next,
        &mut flags,
    )
    .unwrap();
    assert_eq!(flags.carry, carry);
    ops.extend(cc.operations);
    ops.extend(
        integer::decode_add(
            MaxwellShaderStage::Pixel,
            24,
            0x3810_0800_0007_ff00,
            2,
            &mut next,
            &mut flags.carry,
        )
        .unwrap()
        .operations,
    );
    assert_eq!(evaluate(ops, &[u32::MAX, 1], vec![]), 0);
    // The observed instruction uses RZ as destination and R8 as source.
    let mut input = [0; 9];
    for value in [0, 1, 0x8000_0000, u32::MAX] {
        input[8] = value;
        let mut flags = integer::IntegerConditionCodes::default();
        let decoded = integer::decode_integer_to_integer(
            MaxwellShaderStage::Vertex,
            0x1870,
            0x5ce0_8000_0087_0aff,
            9,
            &mut 9,
            &mut flags,
        )
        .unwrap();
        for (flag, expected) in [
            (flags.zero.unwrap(), u32::from(value == 0)),
            (flags.sign.unwrap(), value >> 31),
        ] {
            assert_eq!(
                evaluate(
                    decoded.operations.clone(),
                    &input,
                    vec![ShaderInstruction::new(
                        ShaderSourceLocation::new(84),
                        ShaderPredicate::Always,
                        ShaderOperation::Move32 {
                            destination: ShaderRegister::new(0),
                            source: flag,
                            scalar_type: ShaderScalarType::Unsigned32
                        },
                    )]
                ),
                expected
            );
        }
    }
}

#[test]
fn csetp_condition_truth_tables_boolean_combination_and_aliased_accumulator() {
    let r = ShaderRegister::new;
    let flags = integer::IntegerConditionCodes {
        zero: Some(r(0)),
        sign: Some(r(1)),
        carry: Some(r(2)),
        overflow: Some(r(3)),
    };
    for bits in 0..16 {
        let z = bits & 1 != 0;
        let s = bits & 2 != 0;
        let c = bits & 4 != 0;
        let o = bits & 8 != 0;
        let expected = [
            false,
            (s && !z) ^ o,
            !s && z,
            s ^ (z || o),
            (!s ^ o) && !z,
            !z,
            !(s ^ o),
            !s || !z,
            s && z,
            s ^ o,
            z,
            (s ^ o) || z,
            !s ^ (z || o),
            s || !z,
            (!s || z) ^ o,
            true,
            !o,
            !c,
            !s,
            z || !c,
            c && !z,
            s,
            c,
            o,
        ];
        for test in (0..24).chain([30, 31]) {
            let condition = match test {
                30 => s || z,
                31 => !s && !z,
                _ => expected[test as usize],
            };
            for combine in 0..3u64 {
                for accumulator in [false, true] {
                    for inverted in [false, true] {
                        let word = 0x50a0_0000_0007_0000
                            | (test << 8)
                            | (2 << 3)
                            | 1
                            | (2 << 39)
                            | (u64::from(inverted) << 42)
                            | (combine << 45);
                        assert!(integer::is_condition_set_predicate(word));
                        let decoded = integer::decode_condition_set_predicate(
                            MaxwellShaderStage::Pixel,
                            8,
                            word,
                            &mut 6,
                            &flags,
                        )
                        .unwrap();
                        let mut ops = vec![ShaderOperation::SetPredicateInteger32 {
                            destinations: [Some(2), None],
                            left: r(4),
                            right: r(5),
                            signed: false,
                            comparison: ShaderIntegerComparison::NotEqual,
                            accumulator: ShaderPredicate::Always,
                            set_operation: ShaderPredicateSetOperation::And,
                        }];
                        ops.extend(decoded.operations);
                        let accumulator = accumulator ^ inverted;
                        for (predicate, condition) in [(2, condition), (1, !condition)] {
                            let expected = match combine {
                                0 => condition && accumulator,
                                1 => condition || accumulator,
                                _ => condition ^ accumulator,
                            };
                            let read = ShaderInstruction::new(
                                ShaderSourceLocation::new(84),
                                ShaderPredicate::Always,
                                ShaderOperation::CompareInteger32 {
                                    destination: r(0),
                                    left: r(5),
                                    right: r(5),
                                    signed: false,
                                    float_result: false,
                                    comparison: ShaderIntegerComparison::True,
                                    accumulator: ShaderPredicate::Register {
                                        register: predicate,
                                        inverted: false,
                                    },
                                    set_operation: ShaderPredicateSetOperation::And,
                                },
                            );
                            assert_eq!(
                                evaluate(
                                    ops.clone(),
                                    &[
                                        u32::from(z),
                                        u32::from(s),
                                        u32::from(c),
                                        u32::from(o),
                                        u32::from(accumulator ^ inverted),
                                        0,
                                    ],
                                    vec![read]
                                ),
                                if expected { u32::MAX } else { 0 },
                                "bits={bits} test={test} combine={combine} inverted={inverted} predicate={predicate}"
                            );
                        }
                    }
                }
            }
        }
    }
    let absent = integer::IntegerConditionCodes::default();
    for (test, detail) in [
        (5, "CSETP has no translated Z flag definition"),
        (21, "CSETP has no translated S flag definition"),
        (22, "CSETP has no translated C flag definition"),
        (23, "CSETP has no translated O flag definition"),
        (28, "CSETP CSM/FCSM flow test"),
    ] {
        assert!(matches!(integer::decode_condition_set_predicate(
            MaxwellShaderStage::Vertex, 8, 0x50a0_0380_0007_000f | (test << 8),
            &mut 4, &absent,
        ), Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: observed, .. }) if observed == detail));
    }
    // The observed NEU condition reads only S/Z, not carry or overflow.
    let only_zs = integer::IntegerConditionCodes {
        zero: flags.zero,
        sign: flags.sign,
        ..Default::default()
    };
    assert!(
        integer::decode_condition_set_predicate(
            MaxwellShaderStage::Vertex,
            0x18a8,
            0x50a0_0380_0007_0d0f,
            &mut 4,
            &only_zs,
        )
        .is_ok()
    );
    for modifier in [1 << 14, 3 << 45] {
        assert!(
            integer::decode_condition_set_predicate(
                MaxwellShaderStage::Vertex,
                8,
                0x50a0_0380_0007_0d0f | modifier,
                &mut 4,
                &flags,
            )
            .is_err()
        );
    }
}

#[test]
fn sel_preserves_operand_bits_forms_predicates_zero_and_register_aliases() {
    let r = ShaderRegister::new;
    for (left, right) in [(0, u32::MAX), (0x8000_0000, 0), (0x7fc0_1234, 0xdead_beef)] {
        for form in [0x5ca0u64, 0x4ca0, 0x38a0, 0x39a0] {
            for raw_a in [0, 255] {
                for raw_b in [1, 255] {
                    for raw_destination in [0, 1, 255] {
                        for selector in [0, 7] {
                            for predicate in [false, true] {
                                for inverted in [false, true] {
                                    let payload = match form {
                                        0x5ca0 => raw_b << 20,
                                        0x4ca0 => (2 << 34) | (3 << 20),
                                        _ => 0x7ffff << 20,
                                    };
                                    let word = (form << 48)
                                        | 0x70000
                                        | raw_destination
                                        | (raw_a << 8)
                                        | payload
                                        | (selector << 39)
                                        | (u64::from(inverted) << 42);
                                    assert!(integer::is_select(word));
                                    let mut decoded = integer::decode_select(
                                        MaxwellShaderStage::Pixel,
                                        8,
                                        word,
                                        4,
                                        &mut 4,
                                    )
                                    .unwrap();
                                    if form == 0x4ca0 {
                                        assert_eq!(decoded.constant_buffer_binding, Some(2));
                                        let load = decoded
                                            .operations
                                            .iter_mut()
                                            .find(|op| {
                                                matches!(
                                                    op,
                                                    ShaderOperation::LoadConstantBuffer32 { .. }
                                                )
                                            })
                                            .unwrap();
                                        let ShaderOperation::LoadConstantBuffer32 {
                                            destination,
                                            byte_offset,
                                            ..
                                        } = *load
                                        else {
                                            unreachable!()
                                        };
                                        assert_eq!(byte_offset, 12);
                                        *load = ShaderOperation::MoveImmediate32 {
                                            destination,
                                            bits: right,
                                            scalar_type: ShaderScalarType::Unsigned32,
                                        };
                                    }
                                    let mut ops = vec![ShaderOperation::SetPredicateInteger32 {
                                        destinations: [Some(0), None],
                                        left: r(2),
                                        right: r(3),
                                        signed: false,
                                        comparison: ShaderIntegerComparison::NotEqual,
                                        accumulator: ShaderPredicate::Always,
                                        set_operation: ShaderPredicateSetOperation::And,
                                    }];
                                    let ShaderOperation::Select32 { destination, .. } =
                                        *decoded.operations.last().unwrap()
                                    else {
                                        panic!("SEL must use neutral select")
                                    };
                                    ops.extend(decoded.operations);
                                    ops.push(ShaderOperation::Move32 {
                                        destination: r(0),
                                        source: destination,
                                        scalar_type: ShaderScalarType::Unsigned32,
                                    });
                                    let selected_a = (selector == 7 || predicate) ^ inverted;
                                    let a = if raw_a == 255 { 0 } else { left };
                                    let b = match form {
                                        0x5ca0 if raw_b == 255 => 0,
                                        0x38a0 => 0x7ffff,
                                        0x39a0 => u32::MAX,
                                        _ => right,
                                    };
                                    assert_eq!(
                                        evaluate(
                                            ops,
                                            &[left, right, u32::from(predicate), 0],
                                            vec![]
                                        ),
                                        if selected_a { a } else { b },
                                        "{word:016x}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let captured = integer::decode_select(
        MaxwellShaderStage::Vertex,
        0x1c38,
        0x39a0_00ff_fff7_1e17,
        31,
        &mut 31,
    )
    .unwrap();
    assert_eq!(captured.constant_buffer_binding, None);
    assert!(
        captured
            .operations
            .iter()
            .any(|op| matches!(op, ShaderOperation::MoveImmediate32 { bits: u32::MAX, .. }))
    );
    assert!(
        integer::decode_select(
            MaxwellShaderStage::Vertex,
            8,
            0x5ca0_0380_0017_0000 | (1 << 47),
            4,
            &mut 4
        )
        .is_err()
    );
}
