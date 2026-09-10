use super::*;

pub(super) fn initial_state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(PC);
    for (index, register) in state.general_register_storage_mut().iter_mut().enumerate() {
        *register = 0x8000_0123_4567_89ab ^ (index as u64 * 0x1111_1111);
    }
    *state.stack_pointer_storage_mut() = 0x1234_5670;
    state.set_nzcv(Nzcv::from_bits(0xb000_0000));
    state
}

pub(super) fn compare(word: u32, mut expected: A64State) {
    let mut actual = expected.clone();
    nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let (reason, exit) = execute(&[word, 0xd420_0000], &mut actual);
    assert_eq!(reason, NativeExitReason::Architectural);
    assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
    assert_eq!(actual, expected, "instruction {word:08x}");
}

#[test]
fn scalar_subencodings_preserve_pre_cutover_semantic_coverage() {
    for word in [
        0xf2d5_79a0, // MOVK X0,#0xabcd,LSL#32
        0xab02_0c20, // ADDS X0,X1,X2,LSL#3
        0x6b82_1c20, // SUBS W0,W1,W2,ASR#7
        0x8b21_cbe0, // ADD X0,SP,W1,SXTW#2
        0xba02_0020, // ADCS X0,X1,X2
        0x7a02_0020, // SBCS W0,W1,W2
        0x9208_9c20, // AND X0,X1,#0xff00ff00ff00ff00
        0x6ae2_1420, // BICS W0,W1,W2,ROR#5
        0x331b_0c20, // BFI W0,W1,#5,#4
        0x3300_1020, // BFXIL W0,W1,#0,#5
        0x93c2_3420, // EXTR X0,X1,X2,#13
        0x9ac2_0820, // UDIV X0,X1,X2
        0x9ac2_0c20, // SDIV X0,X1,X2
        0x9ac2_2420, // LSRV X0,X1,X2
        0x9ac2_2820, // ASRV X0,X1,X2
        0x9ac2_2c20, // RORV X0,X1,X2
        0xfa42_102a, // CCMP X1,X2,#10,NE
        0xfa47_0825, // CCMP X1,#7,#5,EQ
        0x9a82_b420, // CSINC X0,X1,X2,LT
        0xda82_a020, // CSINV X0,X1,X2,GE
        0xda82_8420, // CSNEG X0,X1,X2,HI
        0x9b02_0c20, // MADD X0,X1,X2,X3
        0x1b02_8c20, // MSUB W0,W1,W2,W3
        0x9b22_0c20, // SMADDL X0,W1,W2,X3
        0x9b22_8c20, // SMSUBL X0,W1,W2,X3
        0x9b42_7c20, // SMULH X0,X1,X2
        0x9ba2_0c20, // UMADDL X0,W1,W2,X3
        0x9ba2_8c20, // UMSUBL X0,W1,W2,X3
        0x9bc2_7c20, // UMULH X0,X1,X2
        0xdac0_0020, // RBIT X0,X1
        0xdac0_0420, // REV16 X0,X1
        0xdac0_0820, // REV32 X0,X1
        0xdac0_0c20, // REV X0,X1
        0xdac0_1020, // CLZ X0,X1
        0xdac0_1420, // CLS X0,X1
    ] {
        compare(word, initial_state());
    }
}

#[test]
fn division_zero_and_signed_overflow_match_both_guest_widths() {
    for wide in [false, true] {
        for signed in [false, true] {
            let word = 0x1ac2_0820 | (u32::from(wide) << 31) | (u32::from(signed) << 10);
            let mut state = initial_state();
            state.general_register_storage_mut()[2] = 0;
            compare(word, state);
        }
        let mut state = initial_state();
        state.general_register_storage_mut()[1] = if wide { 1 << 63 } else { 1 << 31 };
        state.general_register_storage_mut()[2] = u64::MAX;
        compare(0x1ac2_0c20 | (u32::from(wide) << 31), state);
    }
}

#[test]
fn all_conditions_match_every_nzcv_combination() {
    for condition in 0..16 {
        for flags in 0..16 {
            let mut state = initial_state();
            state.set_nzcv(Nzcv::from_bits(flags << 28));
            compare(0x9a82_0020 | (condition << 12), state); // CSEL X0,X1,X2,cond
        }
    }
}
