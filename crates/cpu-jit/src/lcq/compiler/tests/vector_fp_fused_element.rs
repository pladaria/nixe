use super::*;
use crate::lcq::fp::{CompletionError, complete_vector_fused_element};
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

fn encoding(wide: bool, full: bool, lane: u32) -> u32 {
    0x0f82_1020
        | ((full as u32) << 30)
        | ((wide as u32) << 22)
        | if wide {
            lane << 11
        } else {
            ((lane & 1) << 21) | ((lane >> 1) << 11)
        }
}

fn check(
    word: u32,
    first: u128,
    second: u128,
    accumulator: u128,
    fpcr: u32,
    baseline: bool,
) -> EdgeKind {
    let words = [
        0xb100_0400,
        0x4ea4_1c81,
        0x4ea5_1ca2,
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
    actual.set_vector(0, accumulator);
    actual.set_vector(23, first);
    actual.set_vector(25, second);
    actual.set_vector(27, accumulator);
    actual.set_vector(4, first);
    actual.set_vector(5, second);
    actual.set_vector(31, first);
    let mut expected = actual.clone();
    for &word in &words[..3] {
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
        EdgeKind::VectorFpFusedElement(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 12);
            match complete_vector_fused_element(operation, &mut actual) {
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
    for &word in &words[4..6] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    assert_eq!(
        actual, expected,
        "{word:08x}, {first:x} * element({second:x}), FPCR={fpcr:x}"
    );
    exit.kind
}

#[test]
fn fused_element_matches_shapes_rounding_cancellation_aliases_and_traps() {
    for (wide, full) in [(false, false), (false, true), (true, true)] {
        let bits = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            }
        };
        let minimum = bits(if wide {
            f64::MIN_POSITIVE
        } else {
            f64::from(f32::MIN_POSITIVE)
        });
        let maximum = bits(if wide { f64::MAX } else { f64::from(f32::MAX) });
        let snan = bits(f64::INFINITY) | 1;
        let last = if wide { 1 } else { 3 };
        for subtract in [false, true] {
            let word = encoding(wide, full, last) | (u32::from(subtract) << 14);
            for (a, b, c) in [
                (bits(1.1), bits(-1.1), bits(0.75)),
                (bits(1.0) + 1, bits(1.0) - 2, bits(-1.0)),
                (0, bits(-2.0), bits(-0.0)),
                (
                    maximum,
                    bits(2.0),
                    maximum | (1 << if wide { 63 } else { 31 }),
                ),
                (minimum, bits(0.5), 0),
                (minimum + 1, bits(0.5), 0),
                (minimum, minimum, bits(-0.0)),
                (bits(f64::INFINITY), 0, bits(1.0)),
                (snan, bits(f64::NAN), snan + 1),
                (bits(f64::NAN), snan, bits(f64::NAN)),
                (bits(2.0), bits(2.0), snan),
                (1, bits(1.0), 0),
                (bits(1.0), 1, minimum),
            ] {
                let mut first = [bits(2.0), a, bits(3.0), bits(4.0)];
                let mut accumulator = [bits(0.5), c, bits(1.0), bits(-2.0)];
                if !wide && !full {
                    first[2..].copy_from_slice(&[snan, 1]);
                    accumulator[2..].copy_from_slice(&[snan, 1]);
                }
                let mut second = [snan; 4];
                second[last as usize] = b;
                for mode in 0..16 {
                    check(
                        word,
                        pack(first, wide),
                        pack(second, wide),
                        pack(accumulator, wide),
                        mode << 22,
                        false,
                    );
                }
                for fpcr in [1 << 8, 1 << 10, 1 << 11, 1 << 12, (1 << 24) | (1 << 15)] {
                    check(
                        word,
                        pack(first, wide),
                        pack(second, wide),
                        pack(accumulator, wide),
                        fpcr,
                        false,
                    );
                }
            }
            for lane in 0..=last {
                let word = encoding(wide, full, lane) | (u32::from(subtract) << 14);
                let mut second = [snan; 4];
                second[lane as usize] = bits(1.1);
                let edge = check(
                    word,
                    pack([bits(1.1); 4], wide),
                    pack(second, wide),
                    pack([bits(0.5); 4], wide),
                    0,
                    false,
                );
                #[cfg(target_arch = "x86_64")]
                let native_fma =
                    std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma");
                #[cfg(target_arch = "aarch64")]
                let native_fma = true;
                if native_fma {
                    assert_eq!(
                        edge,
                        EdgeKind::Breakpoint(0),
                        "ordinary FMA must stay native"
                    );
                }
                check(
                    word,
                    pack([bits(1.1); 4], wide),
                    pack(second, wide),
                    pack([bits(0.5); 4], wide),
                    0,
                    true,
                );
            }
            for alias in [
                (word & !31) | 1,
                (word & !31) | 2,
                (word & !(31 << 5)),
                (word & !(31 << 16)),
                (word & !31) | 31,
            ] {
                check(
                    alias,
                    pack([bits(1.1); 4], wide),
                    pack([bits(-1.1); 4], wide),
                    pack([bits(0.5); 4], wide),
                    0,
                    false,
                );
                check(
                    alias,
                    pack([snan; 4], wide),
                    pack([bits(1.0); 4], wide),
                    pack([snan; 4], wide),
                    1 << 8,
                    false,
                );
            }
        }
    }
}

#[test]
fn rotating_cube_fmla_encoding_stays_native() {
    check(
        0x4f99_12fb,
        pack([1.5f32.to_bits() as u64; 4], false),
        pack([2.0f32.to_bits() as u64; 4], false),
        pack([0.5f32.to_bits() as u64; 4], false),
        0,
        false,
    );
}
