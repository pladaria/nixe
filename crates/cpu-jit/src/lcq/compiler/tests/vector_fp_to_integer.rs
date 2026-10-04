use super::*;
use crate::lcq::fp::{CompletionError, complete_vector_to_integer};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

fn check(word: u32, source: u128, fpcr: u32, baseline: bool) -> EdgeKind {
    let words = [
        0xb100_0400,
        0x4ea4_1c81,
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
    actual.set_vector(0, u128::MAX);
    actual.set_vector(1, u128::MAX);
    actual.set_vector(4, source);
    actual.set_vector(31, source);
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let mut compiler = Compiler::new(native_abi()).unwrap();
    if baseline {
        compiler.isa = isa::lookup(compiler.isa.triple().clone())
            .unwrap()
            .finish(compiler.isa.flags().clone())
            .unwrap();
    }
    let (_, exit) = execute_compiler(&memory, words.len(), &mut actual, compiler);
    match exit.kind {
        EdgeKind::VectorFpToInteger(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 8);
            match complete_vector_to_integer(operation, &mut actual) {
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
    for &word in &words[3..5] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    assert_eq!(
        actual, expected,
        "{word:08x}, {source:x}, FPCR={fpcr:x}, baseline={baseline}"
    );
    exit.kind
}

#[test]
fn vector_fcvtzs_fcvtzu_match_native_exact_aliases_status_and_traps() {
    for shape in [0x0ea1_b820, 0x4ea1_b820, 0x4ee1_b820] {
        let wide = shape & (1 << 22) != 0;
        let encode = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            }
        };
        let pack = |values: [u64; 4]| -> u128 {
            values
                .into_iter()
                .take(if wide { 2 } else { 4 })
                .enumerate()
                .fold(0, |acc, (i, v)| {
                    acc | (u128::from(v) << (i * if wide { 64 } else { 32 }))
                })
        };
        let bound = encode(2.0f64.powi(if wide { 63 } else { 31 }));
        for unsigned in [false, true] {
            for lanes in [
                [0; 4],
                [encode(-0.0); 4],
                [encode(1.75), encode(100.125), encode(-1.75), encode(0.5)],
                [bound, encode(1.5), bound, encode(-1.5)],
                [encode(1.0), 1, encode(2.0), 1],
                [
                    encode(1.0),
                    encode(f64::INFINITY),
                    encode(f64::NEG_INFINITY),
                    encode(f64::NAN),
                ],
                [encode(f64::INFINITY) | 1; 4],
                [encode(2.0f64.powi(if wide { 64 } else { 32 })); 4],
                // Upper inactive lanes must not contribute flags/traps.
                [encode(1.0), encode(2.0), encode(f64::NAN), 1],
            ] {
                for rd in [0, 1, 31] {
                    let word = shape | ((unsigned as u32) << 29) | rd;
                    for fpcr in [0, 1 << 24, 1 << 8, 1 << 12, 1 << 15, 3 << 22] {
                        check(word, pack(lanes), fpcr, false);
                    }
                    check(word, pack(lanes), 0, true);
                }
            }
        }
    }
}

#[test]
fn vector_fcvt_normal_lanes_stay_native_and_clear_inactive_bits() {
    let source = u128::from(1.75f32.to_bits())
        | (u128::from(2.5f32.to_bits()) << 32)
        | (u128::from(f32::NAN.to_bits()) << 64);
    assert_eq!(
        check(0x0ea1_b820, source, 0, false),
        EdgeKind::Breakpoint(0)
    );
    assert_eq!(check(0x2ea1_b820, source, 0, true), EdgeKind::Breakpoint(0));
}
