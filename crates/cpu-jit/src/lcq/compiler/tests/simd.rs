use super::integer::{compare, initial_state};

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
