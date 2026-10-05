use super::integer::{compare, initial_state};

#[test]
fn table_lookup_matches_all_indices_table_lengths_and_aliases() {
    for count in 1..=4_u32 {
        for full in [false, true] {
            for extend in [false, true] {
                for rd in [0, 1, 2, 4, 31] {
                    for batch in 0..16_u8 {
                        let mut initial = initial_state();
                        for register in 0..32_u8 {
                            initial.set_vector(
                                register,
                                u128::from_le_bytes(std::array::from_fn(|i| {
                                    register.wrapping_mul(17).wrapping_add(i as u8)
                                })),
                            );
                        }
                        initial.set_vector(
                            4,
                            u128::from_le_bytes(std::array::from_fn(|i| batch * 16 + i as u8)),
                        );
                        initial.set_fpsr(0x0800_009f);
                        compare(
                            0x0e04_03e0
                                | ((count - 1) << 13)
                                | (u32::from(full) << 30)
                                | (u32::from(extend) << 12)
                                | rd,
                            initial,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn integer_sign_matches_wrapping_extremes_and_aliases() {
    for scalar in [false, true] {
        for size in 0..4 {
            if scalar && size != 3 {
                continue;
            }
            for full in [false, true] {
                if scalar && !full || !scalar && size == 3 && !full {
                    continue;
                }
                for negate in [false, true] {
                    for (rd, rn) in [(0, 1), (1, 1), (31, 31)] {
                        let mut initial = initial_state();
                        initial.set_vector(rd, u128::MAX);
                        initial.set_vector(rn, 0x8000_0000_0000_0000_807f_0100_ffff_8000);
                        initial.set_fpsr(0x0800_009f);
                        let base = if scalar { 0x5e20_b800 } else { 0x0e20_b800 };
                        compare(
                            base | (size << 22)
                                | (u32::from(full) << 30)
                                | (u32::from(negate) << 29)
                                | (u32::from(rn) << 5)
                                | u32::from(rd),
                            initial,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn integer_add_wide_matches_signed_unsigned_source_halves_and_aliases() {
    for size in 0..3 {
        for unsigned in [false, true] {
            for upper in [false, true] {
                for (rd, rn, rm) in [(0, 1, 2), (1, 1, 2), (2, 1, 2), (31, 31, 31)] {
                    let mut initial = initial_state();
                    initial.set_vector(rd, u128::MAX);
                    initial.set_vector(rn, 0xffff_ffff_0000_0001_7fff_ffff_8000_0000);
                    initial.set_vector(rm, 0x8000_0001_ffff_ffff_807f_0203_fefd_0001);
                    initial.set_fpsr(0x0800_009f);
                    compare(
                        0x0e20_1000
                            | (size << 22)
                            | (u32::from(unsigned) << 29)
                            | (u32::from(upper) << 30)
                            | (u32::from(rm) << 16)
                            | (u32::from(rn) << 5)
                            | u32::from(rd),
                        initial,
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_integer_comparisons_match_masks_signedness_and_aliases() {
    for base in [
        0x5ee0_3400,
        0x7ee0_3400,
        0x5ee0_3c00,
        0x7ee0_3c00,
        0x5ee0_8c00,
        0x7ee0_8c00,
    ] {
        for (rd, rn, rm) in [(0, 1, 2), (1, 2, 1), (31, 31, 31)] {
            for (lhs, rhs) in [
                (0_u64, 0_u64),
                (0, 1),
                (u64::MAX, 1),
                (1 << 63, 0),
                (0x1234, 0x1234),
                (0xff00, 0x00ff),
            ] {
                let mut initial = initial_state();
                initial.set_vector(rd, u128::MAX);
                initial.set_vector(rn, u128::from(lhs) | (u128::from(rhs) << 64));
                initial.set_vector(rm, u128::from(rhs) | (u128::from(lhs) << 64));
                initial.set_fpsr(0x0800_009f);
                compare(
                    base | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd),
                    initial,
                );
            }
        }
    }
}

#[test]
fn integer_min_max_across_matches_active_arrangements_and_aliases() {
    for base in [0x0e30_a800, 0x0e31_a800, 0x2e30_a800, 0x2e31_a800] {
        for size in 0..3 {
            for full in [false, true] {
                if size == 2 && !full {
                    continue;
                }
                for (rd, rn) in [(0, 1), (31, 31)] {
                    for value in [0, u128::MAX, 0x8000_0001_7fff_ffff_807f_0001_ffff_0203] {
                        let mut initial = initial_state();
                        initial.set_vector(rd, u128::MAX);
                        initial.set_vector(rn, value);
                        initial.set_fpsr(0x0800_009f);
                        compare(
                            base | (size << 22)
                                | (u32::from(full) << 30)
                                | (u32::from(rn) << 5)
                                | u32::from(rd),
                            initial,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn shift_right_accumulate_matches_signed_unsigned_full_shifts_and_aliases() {
    for size in 0..4 {
        let width = 8 << size;
        for scalar in [false, true] {
            for full in [false, true] {
                if (scalar && size != 3) || (!scalar && !full && size == 3) {
                    continue;
                }
                for unsigned in [false, true] {
                    for shift in [1, width / 2, width] {
                        for (rd, rn) in [(0, 1), (31, 31)] {
                            let mut initial = initial_state();
                            initial.set_vector(rd, u128::MAX);
                            initial.set_vector(rn, 0x8000_0001_ffff_ffff_807f_0102_fefd_0304);
                            initial.set_fpsr(0x0800_009f);
                            let word = (if scalar { 0x5f00_1400 } else { 0x0f00_1400 })
                                | (u32::from(unsigned) << 29)
                                | (u32::from(full || scalar) << 30)
                                | ((width * 2 - shift) << 16)
                                | (u32::from(rn) << 5)
                                | u32::from(rd);
                            compare(word, initial);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn vector_integer_multiply_matches_lane_widths_wrapping_and_aliases() {
    for size in 0..3 {
        for full in [false, true] {
            for (rd, rn, rm) in [(0, 1, 2), (31, 31, 2), (2, 31, 2), (31, 31, 31)] {
                for (lhs, rhs) in [
                    (0, u128::MAX),
                    (u128::MAX, u128::MAX),
                    (
                        0x8000_ffff_7fff_0002_0102_0304_0506_0708,
                        0x1234_5678_9abc_def0_fedc_ba98_7654_3210,
                    ),
                ] {
                    let mut initial = initial_state();
                    initial.set_vector(rd, u128::MAX);
                    initial.set_vector(rn, lhs);
                    initial.set_vector(rm, rhs);
                    initial.set_fpsr(0x0800_009f);
                    initial.set_fpcr(0x07c0_0000);
                    compare(
                        0x0e20_9c00
                            | (size << 22)
                            | (u32::from(full) << 30)
                            | (u32::from(rm) << 16)
                            | (u32::from(rn) << 5)
                            | u32::from(rd),
                        initial,
                    );
                }
            }
        }
    }
}

#[test]
fn vector_not_native_lowering_matches_active_widths_and_aliases() {
    for full in [false, true] {
        for (rd, rn) in [(0, 0), (31, 31), (2, 31), (31, 2)] {
            let mut initial = initial_state();
            initial.set_vector(rd, u128::MAX);
            initial.set_vector(rn, 0x1234_5678_9abc_def0_0123_4567_89ab_cdef);
            initial.set_fpsr(0x0800_009f);
            initial.set_fpcr(0x07c0_0000);
            compare(
                0x2e20_5800 | (u32::from(full) << 30) | (u32::from(rn) << 5) | u32::from(rd),
                initial,
            );
        }
    }
}

#[test]
fn scalar_dup_native_lowering_matches_all_lanes_and_register_aliases() {
    for size in 0..4_u32 {
        for index in 0..16 >> size {
            for (rd, rn) in [(0, 0), (31, 31), (2, 31), (31, 2)] {
                let mut initial = initial_state();
                initial.set_vector(rd, u128::MAX);
                initial.set_vector(rn, 0x8f8e_8d8c_8b8a_8988_8786_8584_8382_8180);
                initial.set_fpsr(0x0800_009f);
                initial.set_fpcr(0x07c0_0000);
                let imm5 = (index << (size + 1)) | (1 << size);
                compare(
                    0x5e00_0400 | (imm5 << 16) | (u32::from(rn) << 5) | u32::from(rd),
                    initial,
                );
            }
        }
    }
}

#[test]
fn rev64_native_lowering_matches_all_arrangements_and_register_aliases() {
    for size in 0..3 {
        for full in [false, true] {
            for (rd, rn) in [(0, 0), (31, 31), (2, 31), (31, 2)] {
                let mut initial = initial_state();
                initial.set_vector(rd, u128::MAX);
                initial.set_vector(rn, 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100);
                initial.set_fpsr(0x0800_009f);
                compare(
                    0x0e20_0800
                        | (size << 22)
                        | (u32::from(full) << 30)
                        | (u32::from(rn) << 5)
                        | u32::from(rd),
                    initial,
                );
            }
        }
    }
}

#[test]
fn rev32_native_lowering_matches_all_arrangements_and_register_aliases() {
    for size in 0..2 {
        for full in [false, true] {
            for (rd, rn) in [(0, 0), (31, 31), (2, 31), (31, 2)] {
                let mut initial = initial_state();
                initial.set_vector(rd, u128::MAX);
                initial.set_vector(rn, 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100);
                initial.set_fpsr(0x0800_009f);
                compare(
                    0x2e20_0800
                        | (size << 22)
                        | (u32::from(full) << 30)
                        | (u32::from(rn) << 5)
                        | u32::from(rd),
                    initial,
                );
            }
        }
    }
}

#[test]
fn uaddlv_native_lowering_matches_all_unsigned_arrangements_and_aliases() {
    for word in [
        0x2e30_3800_u32,
        0x6e30_3800,
        0x2e70_3800,
        0x6e70_3800,
        0x6eb0_3800,
    ] {
        for (rd, rn) in [(0, 0), (31, 31), (2, 31), (31, 2)] {
            for value in [
                0,
                u128::MAX,
                0xffff_ffff_ffff_ffff_0807_0605_0403_0201,
                0xffff_ffff_ffff_ffff_0000_0000_0000_0000,
            ] {
                let mut initial = initial_state();
                initial.set_vector(rd, u128::MAX);
                initial.set_vector(rn, value);
                initial.set_fpsr(0x0800_009f);
                compare(word | (u32::from(rn) << 5) | u32::from(rd), initial);
            }
        }
    }
}

#[test]
fn simd_integer_moves_permutations_and_shifts_match_the_interpreter() {
    let cases = [
        0x4e01_0c20_u32, // DUP V0.16B,W1
        0x4e22_1c20,     // AND V0.16B,V1.16B,V2.16B
        0x4e22_8420,     // ADD V0.16B,V1.16B,V2.16B
        0x4e22_bc20,     // ADDP V0.16B,V1.16B,V2.16B
        0x4e21_34a3,     // CMGT V3.16B,V5.16B,V1.16B
        0x4e02_1823,     // UZP1 V3.16B,V1.16B,V2.16B
        0x6e02_4023,     // EXT V3.16B,V1.16B,V2.16B,#8
        0x0f0c_8400,     // SHRN V0.8B,V0.8H,#4
        0x2f0f_0420,     // USHR V0.8B,V1.8B,#1
        0x0e22_4420,     // SSHL V0.8B,V1.8B,V2.8B
        0x4e20_5862,     // CNT V2.16B,V3.16B
        0x4e31_b862,     // ADDV B2,V3.16B
    ];
    for encoding in cases {
        let mut initial = initial_state();
        for register in 0_u8..32 {
            let byte = register.wrapping_mul(7).wrapping_add(3);
            let value = u128::from_le_bytes([byte; 16]) ^ 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
            assert!(initial.set_vector(register, value));
        }
        compare(encoding, initial);
    }
}

#[test]
fn compact_simd_emitters_match_the_interpreter_across_shapes_and_operations() {
    let cases = [
        // Bitwise operations, including destination-as-mask forms.
        0x4e22_1c20_u32,
        0x4e62_1c20,
        0x4ea2_1c20,
        0x4ee2_1c20,
        0x6e22_1c20,
        0x6e62_1c20,
        0x6ea2_1c20,
        0x6ee2_1c20,
        // Pairwise integer operations.
        0x4e22_bc20,
        0x4e22_a420,
        0x4e22_ac20,
        0x6e22_a420,
        0x6e22_ac20,
        0x0e62_bc20,
        0x6ea2_a420,
        0x4ee2_bc20,
        // Element-wise min/max and comparisons.
        0x0ebf_6fdf,
        0x4e21_34a3,
        0x6e21_34a3,
        0x4e21_3ca3,
        0x6e21_3ca3,
        0x4e21_8ca3,
        0x6e21_8ca3,
        0x4e20_8823,
        0x6e20_8823,
        0x4e20_9823,
        0x4ee1_34a3,
        0x2e21_3ca3,
        // Permutations and byte extracts.
        0x4e02_1823,
        0x4e02_5824,
        0x4e02_2825,
        0x4e02_6826,
        0x0e02_3827,
        0x4e42_6828,
        0x4e82_5829,
        0x4ec2_782a,
        0x6e02_4023,
        0x2e0b_3949,
        0x6e1f_43ff,
        // Narrowing and immediate shifts.
        0x0f0c_8400,
        0x4f0c_8420,
        0x0e21_2820,
        0x0e61_2862,
        0x0ea1_28a4,
        0x4e21_28e6,
        0x4e61_2928,
        0x4ea1_296a,
        0x2f0f_0420,
        0x4f0f_0630,
        0x2f1f_04a4,
        0x6f10_04e6,
        0x2f3f_0528,
        0x6f20_056a,
        0x5f60_57de,
        0x0f20_a7fe,
        // Signed and unsigned variable shifts across lane widths and aliases.
        0x0e22_4420,
        0x2e24_4462,
        0x0ebd_47fd,
        0x0e62_4420,
        0x0ea2_4420,
        0x4ee2_4420,
        // Across-vector reductions.
        0x0e31_bbde,
        0x4e31_b862,
        0x0e71_b8a4,
        0x4e71_b8e6,
        0x4eb1_b928,
    ];
    for encoding in cases {
        let mut initial = initial_state();
        for register in 0_u8..32 {
            let low = u64::from(register)
                .wrapping_mul(0x0102_0304_0506_0708)
                .wrapping_add(0x807f_01ff_55aa_cc33);
            let high = low.rotate_left(u32::from(register));
            assert!(initial.set_vector(register, u128::from(low) | (u128::from(high) << 64)));
        }
        compare(encoding, initial);
    }
}

#[test]
fn rev32_reverses_elements_within_each_word_and_clears_inactive_bytes() {
    let input = u128::from_le_bytes(std::array::from_fn(|byte| byte as u8));
    for wide in [false, true] {
        for halfwords in [false, true] {
            for register in [7_u8, 31] {
                let word = 0x2e20_0800
                    | (u32::from(wide) << 30)
                    | (u32::from(halfwords) << 22)
                    | (u32::from(register) << 5)
                    | u32::from(register);
                let mut initial = initial_state();
                initial.set_vector(register, input);
                let mut expected = if halfwords {
                    [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]
                } else {
                    [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
                };
                if !wide {
                    expected[8..].fill(0);
                }
                let mut reference = initial.clone();
                nixe_cpu_interpreter::execute_one(
                    &nixe_cpu::platform::TargetPlatform::Switch1,
                    &mut reference,
                    word,
                )
                .unwrap();
                assert_eq!(
                    reference.vector(register),
                    Some(u128::from_le_bytes(expected))
                );
                compare(word, initial);
            }
        }
    }
}
