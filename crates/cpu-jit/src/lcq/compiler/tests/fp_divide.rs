use super::*;
use crate::lcq::fp::{CompletionError, complete_divide};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

fn check(word: u32, first: u64, second: u64, fpcr: u32) -> EdgeKind {
    // Dirty operands and lazy carry cross activation; the continuation consumes
    // carry and the division result after either native or exact execution.
    let words = [
        0xb100_0400,
        0x9e67_0081,
        0x9e67_00a2,
        word,
        0x9a1f_0063,
        0x9e66_0006,
        0xd420_0000,
    ];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpcr(fpcr);
    actual.set_fpsr(1 << 27);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.general_register_storage_mut()[4] = first;
    actual.general_register_storage_mut()[5] = second;
    actual.set_vector(0, u128::MAX);
    let mut expected = actual.clone();
    for &instruction in &words[..3] {
        execute_one(&TargetPlatform::Switch1, &mut expected, instruction).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    match exit.kind {
        EdgeKind::FpDivide(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 12);
            match complete_divide(operation, &mut actual) {
                Ok(()) => {
                    assert_eq!(reference, InstructionStep::Continue);
                    assert_eq!(actual, expected);
                    let (_, continuation) = execute_memory(&memory, 3, &mut actual);
                    assert_eq!(continuation.kind, EdgeKind::Breakpoint(0));
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
                    return exit.kind;
                }
                Err(error) => panic!("{error:?}"),
            }
        }
        EdgeKind::Breakpoint(0) => assert_eq!(reference, InstructionStep::Continue),
        other => panic!("{other:?}"),
    }
    for &instruction in &words[4..6] {
        execute_one(&TargetPlatform::Switch1, &mut expected, instruction).unwrap();
    }
    assert_eq!(
        actual, expected,
        "FDIV {word:08x}: {first:x}/{second:x}, FPCR {fpcr:x}"
    );
    exit.kind
}

#[test]
fn scalar_division_matches_rounding_special_inputs_and_traps() {
    for wide in [false, true] {
        let bits = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                (v as f32).to_bits() as u64
            }
        };
        let word = 0x1e22_1820 | ((wide as u32) << 22);
        let snan = if wide {
            0x7ff0_0000_0000_0001
        } else {
            0x7f80_0001
        };
        let max = if wide {
            f64::MAX.to_bits()
        } else {
            f32::MAX.to_bits() as u64
        };
        for (first, second) in [
            (bits(6.0), bits(2.0)),
            (bits(1.0), bits(3.0)),
            (bits(-1.0), bits(3.0)),
            (bits(0.0), bits(-2.0)),
            (bits(-0.0), bits(-2.0)),
            (max, bits(0.5)),
            (bits(1.0), bits(0.0)),
            (bits(0.0), bits(0.0)),
            (bits(f64::INFINITY), bits(f64::INFINITY)),
            (bits(2.0), bits(f64::INFINITY)),
            (bits(f64::NAN), snan),
            (snan, bits(f64::NAN)),
            (1, bits(1.0)),
            (bits(1.0), 1),
        ] {
            for mode in 0..16 {
                let poison = if wide { 0 } else { 0xffff_ffff_0000_0000 };
                check(word, first | poison, second | poison, mode << 22);
            }
            for fpcr in [1 << 8, 1 << 9, 1 << 10, 1 << 12, (1 << 24) | (1 << 15)] {
                check(word, first, second, fpcr);
            }
        }
        assert_eq!(
            check(word, bits(6.0), bits(2.0), 0),
            EdgeKind::Breakpoint(0)
        );
        for rd in [1, 2, 31] {
            // Aliasing either operand and V31 writes.
            check((word & !31) | rd, bits(6.0), bits(2.0), 0);
            check((word & !31) | rd, snan, bits(2.0), 0);
        }
    }
}

#[test]
fn scalar_division_tiny_results_match_arm_status_and_rounding() {
    for wide in [false, true] {
        let word = 0x1e22_1820 | ((wide as u32) << 22);
        let (minimum, two, three, max, sign) = if wide {
            (
                f64::MIN_POSITIVE.to_bits(),
                2.0f64.to_bits(),
                3.0f64.to_bits(),
                f64::MAX.to_bits(),
                1 << 63,
            )
        } else {
            (
                f32::MIN_POSITIVE.to_bits() as u64,
                2.0f32.to_bits() as u64,
                3.0f32.to_bits() as u64,
                f32::MAX.to_bits() as u64,
                1 << 31,
            )
        };
        for (first, second) in [
            (minimum, two),
            (minimum + 1, three),
            (2 * minimum - 1, two),
            (minimum, max),
        ] {
            for mode in 0..16 {
                for sign in [0, sign] {
                    check(word, first | sign, second, mode << 22);
                }
            }
            for fpcr in [
                1 << 11,
                1 << 12,
                (1 << 24) | (1 << 11),
                (1 << 24) | (1 << 12),
            ] {
                check(word, first, second, fpcr);
            }
        }
    }
}

#[test]
fn division_exact_boundary_preserves_pending_status_and_final_maps() {
    let words = [0x1e62_1823, 0x1e64_18a0, 0xd420_0000]; // D3=D1/D2; D0=D5/D4
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.output.metadata.entries.len(), 2);
        let exact = lowered
            .states
            .iter()
            .find(|s| {
                s.exit.is_some_and(|e| {
                    e.pc.get() == PC + 4 && matches!(e.kind, EdgeKind::FpDivide(_))
                })
            })
            .unwrap();
        assert!(exact.state.host_fpsr_pending && exact.state.dirty_live.fpsr);
        assert!(exact.state.dirty_live.vector[3]);
        assert!(!exact.state.dirty_live.vector[0]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_vector(1, u128::from(1.0f64.to_bits()));
    actual.set_vector(2, u128::from(3.0f64.to_bits()));
    actual.set_vector(5, u128::from(1.0f64.to_bits()));
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(actual, expected);
    let EdgeKind::FpDivide(operation) = exit.kind else {
        panic!("{exit:?}");
    };
    complete_divide(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x12); // IXC from native 1/3 and DZC from exact 1/0.
}
