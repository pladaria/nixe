use super::*;
use crate::fp_policy::fp_lowering_disposition;
use crate::lcq::fp::{CompletionError, complete_round, complete_to_integer};
use nixe_cpu::decode::a64::fp_simd::Instruction as FpInstruction;
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::execution::CpuExit;
use nixe_cpu_interpreter::{InstructionStep, execute_one};

const ROUND: [u32; 7] = [
    0x1e64_4020, // FRINTN D0,D1
    0x1e64_c020, // FRINTP
    0x1e65_4020, // FRINTM
    0x1e65_c020, // FRINTZ
    0x1e66_4020, // FRINTA
    0x1e67_4020, // FRINTX
    0x1e67_c020, // FRINTI
];

fn instruction(encoding: u32) -> FpInstruction {
    let fragment = Fragment::capture(&memory(&[encoding, 0xd420_0000]), key()).unwrap();
    let DecodeResult::Decoded(decoded) = &fragment.instructions[0] else {
        panic!("not a supported instruction: {encoding:08x}");
    };
    let A64Instruction::FpSimd(instruction) =
        decode::a64::normalize(&decoded.instruction, decoded.encoding)
    else {
        panic!();
    };
    instruction
}

// ADDS supplies dirty lazy flags and an integer destination; FMOV dirties the
// vector destination as well. Helpers must see precisely that PRE-state, not
// an entry checkpoint. ADC after the operation consumes preserved NZCV.
fn check(encoding: u32, bits: u64, fpcr: u32) {
    let words = [0xb100_0484, 0x9e67_0080, encoding, 0x9a1f_0063, 0xd420_0000];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpcr(fpcr);
    actual.set_fpsr(1 << 27);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.general_register_storage_mut()[4] = u64::MAX;
    actual.set_vector(0, u128::MAX);
    // Poison every inactive bit, including the high half of a scalar S source.
    let value = if encoding & (1 << 22) == 0 {
        u128::from(bits as u32) | (u128::MAX << 32)
    } else {
        u128::from(bits) | (u128::MAX << 64)
    };
    actual.set_vector(1, value);
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, encoding).unwrap();
    let count = if fp_lowering_disposition(instruction(encoding)).is_exact() {
        3
    } else {
        5
    };
    let (_, exit) = execute_memory(&memory, count, &mut actual);
    let completion = match exit.kind {
        EdgeKind::FpRound(operation) => {
            assert_eq!(actual, prestate);
            complete_round(operation, &mut actual)
        }
        EdgeKind::FpToInteger(operation) => {
            assert_eq!(actual, prestate);
            complete_to_integer(operation, &mut actual)
        }
        EdgeKind::Breakpoint(0) => {
            assert_eq!(reference, InstructionStep::Continue);
            execute_one(&TargetPlatform::Switch1, &mut expected, words[3]).unwrap();
            assert_eq!(
                actual, expected,
                "native {encoding:08x}, input {bits:x}, FPCR {fpcr:x}"
            );
            return;
        }
        other => panic!("unexpected exit {other:?}"),
    };
    assert_eq!(exit.pc.get(), PC + 8);
    match completion {
        Ok(()) => {
            assert_eq!(reference, InstructionStep::Continue);
            assert_eq!(
                actual, expected,
                "completion {encoding:08x}, input {bits:x}, FPCR {fpcr:x}"
            );
            execute_memory(&memory, 2, &mut actual);
            execute_one(&TargetPlatform::Switch1, &mut expected, words[3]).unwrap();
            assert_eq!(actual, expected);
        }
        Err(CompletionError::Trap(_)) => {
            assert!(matches!(
                reference,
                InstructionStep::Exit(CpuExit::ArchitecturalException {
                    kind: ExceptionKind::FloatingPoint,
                    ..
                })
            ));
            assert_eq!(actual, prestate);
            assert_eq!(actual, expected);
        }
        Err(error) => panic!("{error:?}"),
    }
}

#[test]
fn scalar_rounding_preserves_flags_status_and_scalar_write_semantics() {
    for wide in [false, true] {
        let samples = if wide {
            [
                0,
                1 << 63,
                2.5f64.to_bits(),
                (-1.75f64).to_bits(),
                1,
                f64::INFINITY.to_bits(),
                0x7ff8_0000_0000_0042,
                0x7ff0_0000_0000_0001,
            ]
        } else {
            [
                0,
                1 << 31,
                2.5f32.to_bits() as u64,
                (-1.75f32).to_bits() as u64,
                1,
                f32::INFINITY.to_bits() as u64,
                0x7fc0_0042,
                0x7f80_0001,
            ]
        };
        for encoding in ROUND {
            let encoding = if wide {
                encoding
            } else {
                encoding & !(1 << 22)
            };
            for bits in samples {
                check(encoding, bits, 0);
            }
        }
    }
    for encoding in ROUND {
        for fpcr in [
            1 << 22,
            2 << 22,
            3 << 22,
            1 << 24,
            1 << 25,
            1 << 8,
            1 << 12,
            (1 << 24) | (1 << 15),
        ] {
            for bits in [(-1.75f64).to_bits(), 1, 0x7ff0_0000_0000_0001] {
                check(encoding, bits, fpcr);
            }
        }
        // Aliased source/destination and V31 are ordinary vector registers.
        check((encoding & !31) | 1, 2.5f64.to_bits(), 0);
        check((encoding & !31) | 31, (-1.75f64).to_bits(), 0);
    }
}

#[test]
fn exact_directional_and_fixed_conversions_match_interpreter() {
    for base in [0x1e20_0020, 0x1e24_0020, 0x1e28_0020, 0x1e30_0020] {
        for wide_source in [false, true] {
            for wide_destination in [false, true] {
                for unsigned in [false, true] {
                    let encoding = base
                        | ((wide_source as u32) << 22)
                        | ((wide_destination as u32) << 31)
                        | ((unsigned as u32) << 16);
                    for number in [-3.75f64, 2.5, 1.0e30] {
                        let bits = if wide_source {
                            number.to_bits()
                        } else {
                            (number as f32).to_bits() as u64
                        };
                        check(encoding, bits, 0);
                    }
                }
            }
        }
    }
    for encoding in [0x1e19_e020, 0x1e58_f820, 0x9e59_c020, 0x9e58_0020] {
        let bits = if encoding & (1 << 22) != 0 {
            (-3.75f64).to_bits()
        } else {
            (-3.75f32).to_bits() as u64
        };
        check(encoding, bits, 0);
        check(encoding, bits, 1 << 12);
    }
    // Invalid saturation/NaN and discarded WZR/XZR still update status or trap.
    for encoding in [0x1e20_003f, 0x9e60_003f, 0x9e59_c03f] {
        let nan = if encoding & (1 << 22) != 0 {
            0x7ff0_0000_0000_0001
        } else {
            0x7f80_0001
        };
        for fpcr in [0, 1 << 8, (1 << 24) | (1 << 15)] {
            check(encoding, nan, fpcr);
            check(encoding, 1, fpcr);
        }
    }
}

#[test]
fn round_and_conversion_complete_after_pending_native_status_is_merged() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for encoding in [ROUND[0], ROUND[5], 0x9e60_0020] {
        let words = [encoding, 0xd420_0000];
        let memory = memory(&words);
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.set_vector(1, u128::from(2.5f64.to_bits()));
        actual.set_fpsr(1 << 27);
        let mut expected = actual.clone();
        expected.set_fpsr((1 << 27) | 2); // Pending native divide-by-zero.
        execute_one(&TargetPlatform::Switch1, &mut expected, encoding).unwrap();
        let exact = fp_lowering_disposition(instruction(encoding)).is_exact();
        let (_, exit) = execute_with_fp(
            &memory,
            if exact { 1 } else { 2 },
            &mut actual,
            Compiler::new(native_abi()).unwrap(),
            true,
        );
        match exit.kind {
            EdgeKind::FpRound(operation) => {
                assert_eq!(actual.fpsr(), (1 << 27) | 2);
                complete_round(operation, &mut actual).unwrap();
            }
            EdgeKind::FpToInteger(operation) => {
                assert_eq!(actual.fpsr(), (1 << 27) | 2);
                complete_to_integer(operation, &mut actual).unwrap();
            }
            EdgeKind::Breakpoint(0) => assert!(!exact),
            other => panic!("unexpected exit {other:?}"),
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn round_lowering_uses_selected_isa_and_retains_exact_prestate_maps() {
    for encoding in ROUND {
        let fragment =
            Fragment::capture(&memory(&[0x9e67_0080, encoding, 0xd420_0000]), key()).unwrap();
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let lowered = Compiler::new(abi)
                .unwrap()
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            let native = abi == HostAbi::Aarch64 && ROUND[..4].contains(&encoding);
            assert_eq!(lowered.states.len(), if native { 2 } else { 1 });
            assert!(matches!(
                lowered.states[0].exit.unwrap().kind,
                EdgeKind::FpRound(_)
            ));
            assert!(lowered.states[0].state.dirty_live.vector[0]);
            assert!(!lowered.states[0].state.host_fpsr_pending);
            assert!(lowered.output.metadata.faults.is_empty());
            if native {
                assert!(
                    lowered.output.bytes.chunks_exact(4).any(|bytes| {
                        u32::from_le_bytes(bytes.try_into().unwrap()) & !0x3ff == encoding & !0x3ff
                    }),
                    "missing native FRINT {encoding:08x}"
                );
            }
        }
    }
}
