use super::*;
use crate::lcq::fp::{CompletionError, complete_vector_add};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

fn pack(lanes: [u64; 4], wide: bool) -> u128 {
    if wide {
        u128::from(lanes[0]) | (u128::from(lanes[1]) << 64)
    } else {
        lanes
            .into_iter()
            .enumerate()
            .fold(0, |v, (i, x)| v | (u128::from(x as u32) << (32 * i)))
    }
}

fn check(word: u32, first: u128, second: u128, fpcr: u32) -> EdgeKind {
    let words = [0xb100_0400, word, 0x9a1f_00a5, 0x9e66_03e6, 0xd420_0000];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpcr(fpcr);
    actual.set_fpsr(1 << 27);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.set_vector(0, u128::MAX);
    actual.set_vector(31, first);
    actual.set_vector(23, second);
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    match exit.kind {
        EdgeKind::VectorFpAdd(operation) => {
            assert_eq!(actual, prestate);
            match complete_vector_add(operation, &mut actual) {
                Ok(()) => {
                    assert_eq!(reference, InstructionStep::Continue);
                    assert_eq!(actual, expected);
                    let (_, exit) = execute_memory(&memory, 3, &mut actual);
                    assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
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
    for &word in &words[2..4] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    assert_eq!(
        actual, expected,
        "{word:08x}, {first:x} / {second:x}, FPCR={fpcr:x}"
    );
    exit.kind
}

#[test]
fn vector_add_shapes_rounding_aliases_and_atomic_traps() {
    for (wide, full) in [(false, false), (false, true), (true, true)] {
        let bits = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            }
        };
        let sign = 1u64 << if wide { 63 } else { 31 };
        let tiny = bits(if wide {
            f64::MIN_POSITIVE
        } else {
            f64::from(f32::MIN_POSITIVE)
        });
        let largest = bits(if wide { f64::MAX } else { f64::from(f32::MAX) });
        let snan = bits(f64::INFINITY) | 1;
        let half_ulp = bits(2.0f64.powi(if wide { -53 } else { -24 }));
        for subtract in [false, true] {
            let word = 0x0e37_d7ff
                | (u32::from(full) << 30)
                | (u32::from(wide) << 22)
                | (u32::from(subtract) << 23);
            let inputs = |a, b| {
                let mut first = [bits(1.5), a, bits(-3.0), bits(4.5)];
                let mut second = [bits(2.0), b, bits(7.0), bits(-1.5)];
                if !wide && !full {
                    first[2..].copy_from_slice(&[snan, 1]);
                    second[2..].copy_from_slice(&[snan, 1]);
                }
                (pack(first, wide), pack(second, wide))
            };
            for (a, b) in [
                (bits(1.0), half_ulp),
                (0, sign),
                (largest, largest),
                (tiny + 1, tiny | sign),
                (tiny, tiny + 1),
                (tiny, tiny | sign),
                (1, bits(1.0)),
                (bits(f64::INFINITY), bits(f64::NEG_INFINITY)),
                (snan, bits(f64::NAN)),
                (bits(f64::NAN), snan),
                (bits(f64::NAN), 1),
                (1, snan),
            ] {
                let (first, second) = inputs(a, b);
                for mode in 0..16 {
                    check(word, first, second, mode << 22);
                }
                for fpcr in [1 << 8, 1 << 10, 1 << 11, 1 << 12, (1 << 24) | (1 << 15)] {
                    check(word, first, second, fpcr);
                }
            }
            for rd in [0, 23, 31] {
                let alias = (word & !31) | rd;
                let (first, second) = inputs(bits(3.0), bits(2.0));
                assert_eq!(
                    check(alias, first, second, 0),
                    EdgeKind::Breakpoint(0),
                    "ordinary vector adds must remain native"
                );
                assert_eq!(
                    check(alias, first, second, 1 << 24),
                    EdgeKind::Breakpoint(0)
                );
                assert_eq!(check(alias, 0, 0, 1 << 24), EdgeKind::Breakpoint(0));
                let (first, second) = inputs(snan, bits(2.0));
                check(alias, first, second, 1 << 8);
            }
        }
    }
}
