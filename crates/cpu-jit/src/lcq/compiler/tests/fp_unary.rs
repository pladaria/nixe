use super::*;
use crate::lcq::fp::{CompletionError, complete_unary};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

// ADDS and FMOV supply dirty lazy flags and vector input before the operation;
// ADC and FMOV after it consume preserved carry and the newly produced value.
fn check(encoding: u32, bits: u64, fpcr: u32) -> EdgeKind {
    let words = [
        0xb100_0400,
        0x9e67_0081,
        encoding,
        0x9a1f_0063,
        0x9e66_0005,
        0xd420_0000,
    ];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpcr(fpcr);
    actual.set_fpsr(1 << 27);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.general_register_storage_mut()[4] = bits;
    actual.set_vector(0, u128::MAX);
    actual.set_vector(1, u128::MAX);
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    match exit.kind {
        EdgeKind::FpUnary(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 8);
            match complete_unary(operation, &mut actual) {
                Ok(()) => {
                    assert_eq!(reference, InstructionStep::Continue);
                    assert_eq!(
                        actual, expected,
                        "helper {encoding:08x}, {bits:x}, FPCR {fpcr:x}"
                    );
                    execute_memory(&memory, 3, &mut actual);
                }
                Err(CompletionError::Trap(_)) => {
                    assert!(matches!(reference, InstructionStep::Exit(_)));
                    assert_eq!(actual, expected);
                    return exit.kind;
                }
                Err(error) => panic!("{error:?}"),
            }
        }
        EdgeKind::Breakpoint(0) => assert_eq!(reference, InstructionStep::Continue),
        other => panic!("{other:?}"),
    }
    for &word in &words[3..5] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    assert_eq!(
        actual, expected,
        "native {encoding:08x}, {bits:x}, FPCR {fpcr:x}"
    );
    exit.kind
}

#[test]
fn sqrt_and_precision_conversions_match_exact_results_and_status() {
    for (word, wide) in [
        (0x1e21_c020, false),
        (0x1e61_c020, true),
        (0x1e22_c020, false),
        (0x1e62_4020, true),
    ] {
        let samples = if wide {
            [
                0,
                1 << 63,
                2.0f64.to_bits(),
                4.0f64.to_bits(),
                (-1.0f64).to_bits(),
                f64::MIN_POSITIVE.to_bits(),
                f64::MAX.to_bits(),
                1,
                f64::INFINITY.to_bits(),
                0x7ff8_0000_0000_0042,
                0x7ff0_0000_0000_0001,
            ]
        } else {
            [
                0,
                1 << 31,
                2.0f32.to_bits() as u64,
                4.0f32.to_bits() as u64,
                (-1.0f32).to_bits() as u64,
                f32::MIN_POSITIVE.to_bits() as u64,
                f32::MAX.to_bits() as u64,
                1,
                f32::INFINITY.to_bits() as u64,
                0x7fc0_0042,
                0x7f80_0001,
            ]
        };
        for mode in 0..16 {
            for bits in samples {
                check(
                    word,
                    if wide {
                        bits
                    } else {
                        bits | 0xffff_ffff_0000_0000
                    },
                    mode << 22,
                );
            }
        }
        for fpcr in [1 << 8, 1 << 12, (1 << 24) | (1 << 15)] {
            for bits in samples {
                check(word, bits, fpcr);
            }
        }
        assert_eq!(check(word, samples[2], 0), EdgeKind::Breakpoint(0));
        check((word & !31) | 1, samples[2], 0); // Aliased source/destination.
        check((word & !31) | 31, samples[2], 0); // V31 is not XZR.
    }
}

#[test]
fn demotion_handles_tiny_rounding_boundaries_before_native_status() {
    let min_normal = (f32::MIN_POSITIVE as f64).to_bits();
    for bits in [
        min_normal - 1,
        min_normal,
        min_normal + 1,
        (f32::from_bits(1) as f64).to_bits(),
        0x3ff0_0000_1000_0000, // 1 + 2^-24, halfway between adjacent singles.
        (f32::MAX as f64).to_bits() + 1,
    ] {
        for mode in 0..16 {
            check(0x1e62_4020, bits, mode << 22);
            check(0x1e62_4020, bits | (1 << 63), mode << 22);
        }
        for fpcr in [1 << 11, 1 << 12] {
            check(0x1e62_4020, bits, fpcr);
        }
    }
}

#[test]
fn unary_boundaries_use_final_maps_and_keep_pending_status() {
    let words = [0x1e62_2823, 0x1e61_c0a0, 0xd420_0000]; // FADD; FSQRT D0,D5
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.output.metadata.entries.len(), 2); // One activation only.
        let exact = lowered
            .states
            .iter()
            .find(|s| matches!(s.exit.map(|e| e.kind), Some(EdgeKind::FpUnary(_))))
            .unwrap();
        assert!(exact.state.host_fpsr_pending && exact.state.dirty_live.fpsr);
        assert!(exact.state.dirty_live.vector[3]);
        assert!(!exact.state.dirty_live.vector[0]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_vector(1, u128::from(1.0f64.to_bits()));
    actual.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    actual.set_vector(5, u128::from((-1.0f64).to_bits()));
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(actual, expected);
    let EdgeKind::FpUnary(operation) = exit.kind else {
        panic!("{exit:?}");
    };
    complete_unary(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x11);
}
