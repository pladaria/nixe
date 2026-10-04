use super::*;
use crate::lcq::fp::{CompletionError, complete_compare};
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::execution::CpuExit;
use nixe_cpu_interpreter::{InstructionStep, execute_one};

#[test]
fn native_fp_comparisons_and_exact_edges_match_interpreter() {
    for wide in [false, true] {
        let cases: &[u64] = if wide {
            &[
                0,
                1 << 63,
                1.5f64.to_bits(),
                (-2.0f64).to_bits(),
                f64::INFINITY.to_bits(),
                0x7ff8_0000_0000_0001,
                0x7ff0_0000_0000_0001,
                1,
            ]
        } else {
            &[
                0,
                1 << 31,
                1.5f32.to_bits() as u64,
                (-2.0f32).to_bits() as u64,
                f32::INFINITY.to_bits() as u64,
                0x7fc0_0001,
                0x7f80_0001,
                1,
            ]
        };
        for &first in cases {
            for &second in &[cases[0], cases[2], cases[5]] {
                for signaling in [false, true] {
                    let compare = 0x1e22_2020 | ((wide as u32) << 22) | ((signaling as u32) << 4);
                    check(
                        &[0xb100_0400, compare, 0x9a1f_0063, 0xd420_0000],
                        first,
                        second,
                        0,
                    );
                }
            }
        }
        for &first in cases {
            let compare_zero = 0x1e20_2028 | ((wide as u32) << 22);
            check(
                &[0xb100_0400, compare_zero, 0x9a1f_0063, 0xd420_0000],
                first,
                0,
                1 << 24,
            );
        }
    }
}

fn check(words: &[u32], first: u64, second: u64, fpcr: u32) -> EdgeKind {
    let memory = memory(words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.set_vector(1, u128::from(first) | (u128::from(u64::MAX) << 64));
    actual.set_vector(2, u128::from(second) | (u128::from(u64::MAX) << 64));
    actual.set_fpcr(fpcr);
    actual.set_fpsr(1 << 27);
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let before_compare = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    if let EdgeKind::FpCompare(operation) = exit.kind {
        assert_eq!(exit.pc.get(), PC + 4);
        assert_eq!(actual, before_compare);
        match complete_compare(operation, &mut actual) {
            Ok(()) => {
                assert_eq!(reference, InstructionStep::Continue);
                assert_eq!(actual, expected);
                // Resume through an ordinary demanded canonical entry. This
                // consumes the helper's NZCV without rerunning the comparison.
                execute_memory(&memory, 2, &mut actual);
            }
            Err(CompletionError::Trap(status)) => {
                assert!(status.invalid_operation || status.input_denormal);
                assert!(matches!(
                    reference,
                    InstructionStep::Exit(CpuExit::ArchitecturalException {
                        kind: ExceptionKind::FloatingPoint,
                        ..
                    })
                ));
                assert_eq!(actual, before_compare);
                assert_eq!(actual, expected);
                return exit.kind;
            }
            Err(error) => panic!("{error:?}"),
        }
    } else {
        assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
        assert_eq!(reference, InstructionStep::Continue);
    }
    execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
    assert_eq!(
        actual, expected,
        "{:08x?}: {first:x}, {second:x}, FPCR {fpcr:x}",
        words
    );
    exit.kind
}

#[test]
fn exact_comparison_traps_preserve_prestate_and_conditional_false_does_not_trap() {
    for fpcr in [1 << 8, (1 << 15) | (1 << 24)] {
        for first in [0x7ff0_0000_0000_0001, 1] {
            check(
                &[0xb100_0400, 0x1e62_2020, 0x9a1f_0063, 0xd420_0000],
                first,
                0,
                fpcr,
            );
        }
    }
    // ADDS produces Z=1. EQ evaluates true; NE must select the literal without
    // observing the signaling NaN, including when invalid exceptions are enabled.
    for condition in [0, 1] {
        for signaling in [false, true] {
            let compare = 0x1e62_042a | (condition << 12) | ((signaling as u32) << 4);
            for fpcr in [0, 1 << 8] {
                check(
                    &[0xb100_0400, compare, 0x9a1f_0063, 0xd420_0000],
                    0x7ff0_0000_0000_0001,
                    0,
                    fpcr,
                );
            }
        }
    }
}

#[test]
fn exact_comparison_consumes_dirty_vector_prestate() {
    // FMOV D1, X4; FCMPE D1, D2; ADC X3, X3, XZR; BRK.
    let words = [0x9e67_0081, 0x1e62_2030, 0x9a1f_0063, 0xd420_0000];
    let memory = memory(&words);
    for fpcr in [0, 1 << 8] {
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.set_fpcr(fpcr);
        actual.general_register_storage_mut()[4] = 0x7ff0_0000_0000_0001;
        actual.set_vector(1, u128::MAX);
        let mut expected = actual.clone();
        execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
        let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
        let EdgeKind::FpCompare(operation) = exit.kind else {
            panic!("expected exact comparison, got {exit:?}");
        };
        assert_eq!(actual, expected);
        let reference = execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
        let completion = complete_compare(operation, &mut actual);
        assert_eq!(actual, expected);
        if fpcr == 0 {
            completion.unwrap();
            assert_eq!(reference, InstructionStep::Continue);
            execute_memory(&memory, 2, &mut actual);
            execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
            assert_eq!(actual, expected);
        } else {
            assert!(matches!(completion, Err(CompletionError::Trap(_))));
            assert!(matches!(reference, InstructionStep::Exit(_)));
        }
    }
}

#[test]
fn comparison_final_maps_retain_prestate_and_normal_path_has_no_helper() {
    let words = [0xb100_0400, 0x1e62_2020, 0x9a1f_0063, 0xd420_0000];
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.states.len(), 2);
        assert!(matches!(
            lowered.states[0].exit.unwrap().kind,
            EdgeKind::FpCompare(_)
        ));
        assert!(matches!(
            lowered.states[0].state.nzcv,
            NzcvLocation::Deferred(_)
        ));
        // The later ADC reads X3: its inherited value is observable even at
        // the earlier FP guard, before the ADC has executed.
        assert!(lowered.states[0].state.dirty_live.integer.x.contains(3));
        assert!(lowered.states[1].state.dirty_live.integer.x.contains(3));
        assert!(lowered.output.metadata.faults.is_empty());
    }
    let mut state = A64State::default();
    state.set_pc(PC);
    state.set_vector(1, u128::from(1.5f64.to_bits()));
    state.set_vector(2, u128::from(2.0f64.to_bits()));
    let (_, exit) = execute(&words, &mut state);
    assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
}

#[test]
fn conditional_fp_comparison_keeps_lazy_flags_and_never_observes_false_inputs() {
    use nixe_cpu::semantics::conditions::evaluate_a64;
    for wide in [false, true] {
        for condition in 0..16 {
            for nzcv in 0..16 {
                for signaling in [false, true] {
                    let first = if wide {
                        1.5f64.to_bits()
                    } else {
                        u64::from(1.5f32.to_bits())
                    };
                    let snan = if wide {
                        0x7ff0_0000_0000_0001
                    } else {
                        0x7f80_0001
                    };
                    let compare = 0x1e22_042a
                        | (u32::from(wide) << 22)
                        | (condition << 12)
                        | (u32::from(signaling) << 4);
                    for input in [first, snan] {
                        let words = [compare, 0xd420_0000];
                        let memory = memory(&words);
                        let mut actual = A64State::default();
                        actual.set_pc(PC);
                        actual.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(nzcv << 28));
                        actual.set_fpcr(1 << 8);
                        actual.set_fpsr(1 << 27);
                        actual.set_vector(1, u128::from(input));
                        actual.set_vector(2, 0);
                        let mut expected = actual.clone();
                        let reference =
                            execute_one(&TargetPlatform::Switch1, &mut expected, compare).unwrap();
                        let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
                        if input == first
                            || !evaluate_a64(
                                nixe_cpu::semantics::conditions::Condition::from_encoding(
                                    condition as u8,
                                ),
                                nzcv << 28,
                            )
                        {
                            assert_eq!(
                                exit.kind,
                                EdgeKind::Breakpoint(0),
                                "condition={condition} NZCV={nzcv:x}"
                            );
                            assert_eq!(reference, InstructionStep::Continue);
                            assert_eq!(actual, expected);
                        } else {
                            let EdgeKind::FpCompare(operation) = exit.kind else {
                                panic!("{exit:?}")
                            };
                            assert!(matches!(
                                complete_compare(operation, &mut actual),
                                Err(CompletionError::Trap(_))
                            ));
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }
}
