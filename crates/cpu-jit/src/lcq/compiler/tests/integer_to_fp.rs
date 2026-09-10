use super::*;
use crate::lcq::fp::{CompletionError, complete_from_integer};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit, state::a64::A64Register};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

#[test]
fn overwritten_integer_to_fp_result_retains_inexact_status() {
    let words = [0x9e22_0020, 0x9e67_03e0, 0xd420_0000]; // SCVTF S0,X1; FMOV D0,XZR
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.general_register_storage_mut()[1] = (1 << 24) + 1;
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    execute(&words, &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 1 << 4);
}

fn check(word: u32, source: u64, fpcr: u32) -> EdgeKind {
    // Dirty W/X source and lazy carry cross activation; the continuation reads
    // both the scalar result and preserved carry. Register 31 must not use SP.
    let words = [
        0xb100_0400,
        0x9100_0481,
        word,
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
    actual.general_register_storage_mut()[4] = source.wrapping_sub(1);
    actual.write_x(A64Register::StackPointer, u64::MAX);
    actual.set_vector(0, u128::MAX);
    actual.set_vector(31, u128::MAX);
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    match exit.kind {
        EdgeKind::IntegerToFp(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 8);
            match complete_from_integer(operation, &mut actual) {
                Ok(()) => {
                    assert_eq!(reference, InstructionStep::Continue);
                    assert_eq!(actual, expected);
                    let (_, exit) = execute_memory(&memory, 3, &mut actual);
                    assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
                }
                Err(CompletionError::Trap(_)) => {
                    assert!(
                        matches!(
                            reference,
                            InstructionStep::Exit(CpuExit::ArchitecturalException {
                                kind: ExceptionKind::FloatingPoint,
                                ..
                            })
                        ),
                        "{word:08x}, source {source:x}, FPCR {fpcr:x}: {reference:?}"
                    );
                    assert_eq!(actual, prestate);
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
        "integer-to-FP {word:08x}: {source:x}, FPCR {fpcr:x}"
    );
    exit.kind
}

#[test]
fn scalar_integer_to_fp_rounding_widths_and_exceptions_match_interpreter() {
    for source_64 in [false, true] {
        for destination_64 in [false, true] {
            for signed in [false, true] {
                let word = 0x1e22_0020
                    | ((source_64 as u32) << 31)
                    | ((destination_64 as u32) << 22)
                    | ((!signed as u32) << 16);
                let samples = [
                    0,
                    1,
                    u64::MAX,
                    0x8000_0000,
                    0xffff_ffff_8000_0001,
                    (1u64 << 24) + 1,
                    (1u64 << 53) + 1,
                    1u64 << 63,
                    (1u64 << 63) + 1,
                    u64::MAX - 1023,
                ];
                for &source in &samples {
                    for mode in 0..4 {
                        check(word, source, mode << 22);
                    }
                    check(word, source, 1 << 12); // IXE: trap only for an inexact result.
                }
                for mode in 4..16 {
                    for source in [0, u64::MAX, (1 << 53) + 1] {
                        check(word, source, mode << 22);
                    }
                }
                assert_eq!(check(word, 7, 0), EdgeKind::Breakpoint(0));
                check((word & !31) | 31, u64::MAX, 0); // V31 is a real destination.
                check((word & !31) | 1, u64::MAX, 1 << 12); // W/X1 and V1 do not alias.
                for fpcr in [0, 1 << 12] {
                    check((word & !(31 << 5)) | (31 << 5), u64::MAX, fpcr);
                }
            }
        }
    }
}

#[test]
fn integer_to_fp_activation_and_exact_maps_preserve_pending_status() {
    // The first conversion activates FP and sets IXC; the exact comparison
    // must expose its result and pending status before evaluating a NaN.
    let words = [0x9e63_0020, 0x1e64_2000, 0xd420_0000]; // UCVTF D0,X1; FCMP D0,D4
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.output.metadata.entries.len(), 2);
        let exit = lowered
            .states
            .iter()
            .find(|s| {
                s.exit
                    .is_some_and(|e| matches!(e.kind, EdgeKind::FpCompare(_)))
            })
            .unwrap();
        assert!(exit.state.host_fpsr_pending && exit.state.dirty_live.fpsr);
        assert!(exit.state.dirty_live.vector[0]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.general_register_storage_mut()[1] = u64::MAX;
    actual.set_vector(4, 0x7ff0_0000_0000_0001);
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let (_, exit) = execute_memory(&memory, 3, &mut actual);
    assert_eq!(actual, expected);
    let EdgeKind::FpCompare(operation) = exit.kind else {
        panic!("{exit:?}");
    };
    crate::lcq::fp::complete_compare(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x11);
}
