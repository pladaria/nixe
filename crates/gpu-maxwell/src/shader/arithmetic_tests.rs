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
        let ops = half::decode_half_add(MaxwellShaderStage::Pixel, 8, word, 4, &mut 4).unwrap();
        assert_eq!(evaluate(ops, &[left, right], vec![]), 0x4600_4400); // (4, 6)
    }
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
