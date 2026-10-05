use super::*;
use crate::lcq::fp::{CompletionError, complete_add};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

#[test]
fn scalar_add_rounding_overflow_cancellation_and_zero_match_interpreter() {
    for wide in [false, true] {
        let (sign, one, half_ulp, tiny, largest) = if wide {
            (
                1u64 << 63,
                1.0f64.to_bits(),
                2.0f64.powi(-53).to_bits(),
                f64::MIN_POSITIVE.to_bits(),
                f64::MAX.to_bits(),
            )
        } else {
            (
                1u64 << 31,
                1.0f32.to_bits() as u64,
                2.0f32.powi(-24).to_bits() as u64,
                f32::MIN_POSITIVE.to_bits() as u64,
                f32::MAX.to_bits() as u64,
            )
        };
        for subtract in [false, true] {
            let word = 0x1e22_2823 | ((wide as u32) << 22) | ((subtract as u32) << 12);
            for mode in 0..16 {
                for (first, second) in [
                    (0, sign),
                    (one, half_ulp),
                    (one | sign, half_ulp),
                    (tiny, tiny | sign),
                    (tiny + 1, tiny | sign),
                    (largest, largest),
                    (tiny, tiny),
                    (tiny, (tiny + 1) | sign),
                ] {
                    let mut actual = A64State::default();
                    actual.set_pc(PC);
                    actual.set_fpcr(mode << 22);
                    let poison = if wide {
                        u128::MAX << 64
                    } else {
                        u128::MAX << 32
                    };
                    actual.set_vector(1, u128::from(first) | poison);
                    actual.set_vector(2, u128::from(second) | poison);
                    actual.set_vector(3, u128::MAX);
                    let mut expected = actual.clone();
                    execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
                    let (_, exit) = execute(&[word, 0xd420_0000], &mut actual);
                    match exit.kind {
                        EdgeKind::FpAdd(operation) => complete_add(operation, &mut actual).unwrap(),
                        EdgeKind::Breakpoint(0) => (),
                        other => panic!("{other:?}"),
                    }
                    assert_eq!(
                        actual, expected,
                        "{word:08x}, {first:x}, {second:x}, mode {mode}"
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_add_activates_fp_and_preserves_lazy_flags() {
    let words = [
        0xb100_0400,
        0x1e62_2823,
        0x1e62_3864,
        0x9a1f_00a5,
        0xd420_0000,
    ];
    for mode in 0..16 {
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.set_fpcr(mode << 22);
        actual.set_fpsr(1 << 27);
        actual.general_register_storage_mut()[0] = u64::MAX;
        actual.set_vector(1, u128::from(1.0f64.to_bits()));
        actual.set_vector(2, u128::from((2.0f64.powi(-53)).to_bits()));
        let mut expected = actual.clone();
        for &word in &words[..4] {
            execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
        }
        let (_, exit) = execute(&words, &mut actual);
        assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
        assert_eq!(actual, expected, "mode {mode}");
    }
}

#[test]
fn scalar_add_special_inputs_and_traps_keep_exact_prestate() {
    let words = [0xb100_0400, 0x1e62_2823, 0x9a1f_00a5, 0xd420_0000];
    for fpcr in [0, 1 << 8, 1 << 12, (1 << 15) | (1 << 24)] {
        for first in [
            0x7ff0_0000_0000_0001,
            f64::INFINITY.to_bits(),
            1,
            1.0f64.to_bits(),
        ] {
            let mut actual = A64State::default();
            actual.set_pc(PC);
            actual.set_fpcr(fpcr);
            actual.general_register_storage_mut()[0] = u64::MAX;
            actual.set_vector(1, u128::from(first));
            actual.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
            let mut expected = actual.clone();
            execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
            let prestate = expected.clone();
            let reference = execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
            let (_, exit) = execute(&words, &mut actual);
            match exit.kind {
                EdgeKind::FpAdd(operation) => {
                    assert_eq!(actual, prestate);
                    match complete_add(operation, &mut actual) {
                        Ok(()) => {
                            assert_eq!(reference, InstructionStep::Continue);
                            assert_eq!(actual, expected);
                            execute_memory(&memory(&words), 2, &mut actual);
                        }
                        Err(CompletionError::Trap(_)) => {
                            assert!(matches!(reference, InstructionStep::Exit(_)));
                            assert_eq!(actual, expected);
                            continue;
                        }
                        Err(error) => panic!("{error:?}"),
                    }
                }
                EdgeKind::Breakpoint(0) => assert_eq!(reference, InstructionStep::Continue),
                other => panic!("{other:?}"),
            }
            execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn activation_transfers_allocated_spills_vectors_and_carry_recipes() {
    let mut words: Vec<u32> = (0..31).map(|r| 0x9100_0400 | (r << 5) | r).collect();
    words.push(0xba01_0000); // ADCS X0,X0,X1: carry input is part of the recipe.
    words.extend((0..32).map(|r| 0x6e3f_1c00 | (r << 5) | r));
    words.extend([0x1e62_2823, 0x1e62_2864, 0x9a1f_00a5, 0xd420_0000]);
    let fragment = Fragment::capture(&memory(&words), key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        let internal = lowered
            .states
            .iter()
            .find(|state| state.exit.is_none())
            .unwrap();
        assert_eq!(internal.state.dirty_live.integer.x, StateSet::ALL.integer.x);
        assert!(internal.state.dirty_live.vector == StateSet::ALL.vector);
        assert!(matches!(internal.state.nzcv, NzcvLocation::Deferred(_)));
        assert!(
            internal
                .state
                .bindings
                .iter()
                .any(|binding| matches!(binding.location, crate::abi::ValueLocation::Spill { .. }))
        );
        assert_eq!(lowered.output.metadata.entries.len(), 2);
        assert!(
            lowered
                .states
                .iter()
                .any(|state| state.state.host_fpsr_pending && state.state.dirty_live.fpsr)
        );
    }
    for initial in [0, u64::MAX] {
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.set_nzcv(Nzcv::from_bits(0x2000_0000));
        actual.general_register_storage_mut().fill(initial);
        for index in 0..31 {
            actual.set_vector(
                index,
                u128::from(1.0f64.to_bits()) | (u128::from(index) << 64),
            );
        }
        actual.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
        let mut expected = actual.clone();
        for &word in &words[..words.len() - 1] {
            execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
        }
        let (_, exit) = execute(&words, &mut actual);
        assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
        assert_eq!(actual, expected);
    }
}

#[test]
fn native_to_exact_to_native_preserves_accumulated_status() {
    let words = [0x1e62_2823, 0x1e61_28a4, 0x1e61_2826, 0xd420_0000];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpsr(1 << 27);
    actual.set_vector(1, u128::from(1.0f64.to_bits()));
    actual.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    actual.set_vector(5, 0x7ff0_0000_0000_0001);
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(
        actual, expected,
        "native inexact must merge before the exact operation"
    );
    let EdgeKind::FpAdd(operation) = exit.kind else {
        panic!("{exit:?}");
    };
    execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
    complete_add(operation, &mut actual).unwrap();
    assert_eq!(actual, expected);
    execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
    execute_memory(&memory, 2, &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), (1 << 27) | 0x11);
}

#[test]
fn activation_keeps_an_already_active_segment_and_restores_the_caller() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let mut caller = crate::abi::HostFpState::default();
    unsafe {
        caller.begin();
        caller.finish();
    }
    let words = [0x1e62_2823, 0xd420_0000];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_vector(1, u128::from(1.0f64.to_bits()));
    actual.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    let mut expected = actual.clone();
    expected.set_fpsr(2); // Seeded native divide-by-zero before the activation leaf.
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    execute_with_fp(
        &memory,
        2,
        &mut actual,
        Compiler::new(native_abi()).unwrap(),
        true,
    );
    assert_eq!(actual, expected);
    let mut restored = crate::abi::HostFpState::default();
    unsafe {
        restored.begin();
        restored.finish();
    }
    assert_eq!(
        (restored.saved_control, restored.saved_status),
        (caller.saved_control, caller.saved_status)
    );
}

#[test]
fn fabd_scalar_and_vector_rounding_special_values_aliases_and_traps() {
    for vector in [false, true] {
        for wide in [false, true] {
            for full in [false, true] {
                if !vector && !full || vector && wide && !full {
                    continue;
                }
                for alias in [false, true] {
                    for fpcr in [
                        0,
                        1 << 22,
                        2 << 22,
                        3 << 22,
                        1 << 24,
                        1 << 25,
                        1 << 8,
                        1 << 11,
                        1 << 12,
                        1 << 15,
                    ] {
                        let samples = if wide {
                            [
                                0,
                                1 << 63,
                                1,
                                1.0_f64.to_bits(),
                                f64::MAX.to_bits(),
                                f64::INFINITY.to_bits(),
                                0xfff8_0000_0000_0123,
                                0xfff0_0000_0000_0123,
                            ]
                        } else {
                            [
                                0,
                                1 << 31,
                                1,
                                u64::from(1.0_f32.to_bits()),
                                u64::from(f32::MAX.to_bits()),
                                u64::from(f32::INFINITY.to_bits()),
                                0xffc0_0123,
                                0xff80_0123,
                            ]
                        };
                        for sample in samples {
                            let rd = if alias { 1 } else { 0 };
                            let width = if wide { 64 } else { 32 };
                            let mut actual = A64State::default();
                            actual.set_pc(PC);
                            actual.set_fpcr(fpcr);
                            actual.set_fpsr(1 << 27);
                            actual.set_vector(rd as u8, u128::MAX);
                            actual
                                .set_vector(1, u128::from(sample) | (u128::from(sample) << width));
                            actual.set_vector(
                                2,
                                u128::from(samples[3]) | (u128::from(samples[3]) << width),
                            );
                            let pre = actual.clone();
                            let mut expected = actual.clone();
                            let base = if vector { 0x2ea2_d420 } else { 0x7ea2_d420 };
                            let word =
                                base | (u32::from(wide) << 22) | (u32::from(full) << 30) | rd;
                            let reference =
                                execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
                            let (_, exit) = execute(&[word, 0xd420_0000], &mut actual);
                            let result = match exit.kind {
                                EdgeKind::FpAdd(op) => {
                                    assert_eq!(actual, pre);
                                    complete_add(op, &mut actual)
                                }
                                EdgeKind::VectorFpAdd(op) => {
                                    assert_eq!(actual, pre);
                                    crate::lcq::fp::complete_vector_add(op, &mut actual)
                                }
                                EdgeKind::Breakpoint(0) => {
                                    assert_eq!(reference, InstructionStep::Continue);
                                    Ok(())
                                }
                                other => panic!("{other:?}"),
                            };
                            if let Err(error) = result {
                                assert!(matches!(error, CompletionError::Trap(_)));
                                assert!(matches!(reference, InstructionStep::Exit(_)));
                            }
                            assert_eq!(
                                actual, expected,
                                "FABD {word:x} sample={sample:x} FPCR={fpcr:x}"
                            );
                        }
                    }
                }
            }
        }
    }
}
