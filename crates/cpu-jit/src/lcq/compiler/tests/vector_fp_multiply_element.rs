use super::*;
use crate::lcq::fp::{CompletionError, complete_vector_multiply_element};
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
    0x0f82_9020
        | ((full as u32) << 30)
        | ((wide as u32) << 22)
        | if wide {
            lane << 11
        } else {
            ((lane & 1) << 21) | ((lane >> 1) << 11)
        }
}

fn check(word: u32, first: u128, second: u128, fpcr: u32, baseline: bool) -> EdgeKind {
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
    actual.set_vector(0, u128::MAX);
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
        EdgeKind::VectorFpMultiplyElement(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 12);
            match complete_vector_multiply_element(operation, &mut actual) {
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
fn multiply_element_matches_shapes_rounding_tiny_results_and_traps() {
    for (wide, full) in [(false, false), (false, true), (true, true)] {
        let bits = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                (v as f32).to_bits() as u64
            }
        };
        let minimum = bits(if wide {
            f64::MIN_POSITIVE
        } else {
            f32::MIN_POSITIVE as f64
        });
        let maximum = bits(if wide { f64::MAX } else { f32::MAX as f64 });
        let snan = bits(f64::INFINITY) | 1;
        let last = if wide { 1 } else { 3 };
        let word = encoding(wide, full, last);
        for (index, (a, b)) in [
            (bits(1.1), bits(1.1)),
            (bits(-1.1), bits(1.1)),
            (0, bits(-2.0)),
            (bits(-0.0), bits(-2.0)),
            (maximum, bits(2.0)),
            (minimum, bits(0.5)),
            (minimum + 1, bits(0.5)),
            (2 * minimum - 1, bits(0.5)),
            (minimum, minimum),
            (bits(f64::INFINITY), 0),
            (bits(2.0), bits(f64::INFINITY)),
            (snan, bits(f64::NAN)),
            (bits(f64::NAN), snan),
            (1, bits(1.0)),
            (bits(1.0), 1),
        ]
        .into_iter()
        .enumerate()
        {
            let mut first = [bits(2.0); 4];
            first[index % if wide || !full { 2 } else { 4 }] = a;
            if !wide && !full {
                first[2..].copy_from_slice(&[snan, 1]);
            }
            // Unselected elements are exceptional; they must not affect native eligibility.
            let mut second = [snan; 4];
            second[last as usize] = b;
            let (first, second) = (pack(first, wide), pack(second, wide));
            for mode in 0..16 {
                check(word, first, second, mode << 22, false);
            }
            for fpcr in [1 << 8, 1 << 10, 1 << 11, 1 << 12, (1 << 24) | (1 << 15)] {
                check(word, first, second, fpcr, false);
            }
        }
        for lane in 0..=last {
            let word = encoding(wide, full, lane);
            let mut first = [bits(1.1); 4];
            if !wide && !full {
                first[2..].copy_from_slice(&[snan, 1]);
            }
            let mut second = [snan; 4];
            second[lane as usize] = bits(-1.1);
            for mode in 0..4 {
                assert_eq!(
                    check(
                        word,
                        pack(first, wide),
                        pack(second, wide),
                        mode << 22,
                        true
                    ),
                    EdgeKind::Breakpoint(0)
                );
            }
            check(word, pack(first, wide), pack(second, wide), 1 << 12, false);
        }
        for alias in [
            (word & !31) | 1,
            (word & !31) | 2,
            (word & !31) | 31,
            (word & !(31 << 5)) | (31 << 5),
            (word & !(31 << 16)) | (31 << 16),
        ] {
            check(
                alias,
                pack([bits(1.1); 4], wide),
                pack([bits(-1.1); 4], wide),
                0,
                false,
            );
            check(
                alias,
                pack([snan; 4], wide),
                pack([bits(2.0); 4], wide),
                0,
                false,
            );
        }
    }
}

#[test]
fn overwritten_multiply_element_keeps_status_and_precise_maps() {
    // Native inexact product is overwritten before a second multiply's exact 0*Inf.
    let words = [
        encoding(false, true, 3),
        0x6e20_1c00,
        (encoding(false, true, 0) & !((31 << 5) | (31 << 16))) | (5 << 5) | (4 << 16),
        0xd420_0000,
    ];
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
                    e.pc.get() == PC + 8 && matches!(e.kind, EdgeKind::VectorFpMultiplyElement(_))
                })
            })
            .unwrap();
        assert!(exact.state.host_fpsr_pending && exact.state.dirty_live.fpsr);
        assert!(exact.state.dirty_live.vector.contains(0));
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_vector(1, pack([1.1f32.to_bits() as u64; 4], false));
    actual.set_vector(2, pack([1.1f32.to_bits() as u64; 4], false));
    actual.set_vector(4, pack([f32::INFINITY.to_bits() as u64; 4], false));
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 1 << 4);
    let EdgeKind::VectorFpMultiplyElement(operation) = exit.kind else {
        panic!("{exit:?}")
    };
    complete_vector_multiply_element(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x11);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn multiply_element_exact_semantics_match_arm_instructions() {
    use nixe_cpu::semantics::a64_fp_simd::{exact_vector_float_multiply_element, fp_status_bits};
    macro_rules! arm {
        ($instruction:literal, $first:expr, $second:expr, $fpcr:expr) => {{
            let mut result = 0u128;
            let status: u64;
            // Exceptions masked; save/restore the complete caller FP environment.
            unsafe { std::arch::asm!(
                "mrs {saved_control}, fpcr", "mrs {saved_status}, fpsr",
                "msr fpcr, {control}", "msr fpsr, xzr",
                "ldr q1, [{first}]", "ldr q2, [{second}]", $instruction,
                "str q0, [{result}]", "mrs {status}, fpsr",
                "msr fpcr, {saved_control}", "msr fpsr, {saved_status}",
                saved_control = out(reg) _, saved_status = out(reg) _,
                control = in(reg) u64::from($fpcr), status = out(reg) status,
                first = in(reg) &$first, second = in(reg) &$second,
                result = in(reg) &mut result, out("v0") _, out("v1") _, out("v2") _, options(nostack),
            ); }
            (result, status as u32)
        }};
    }
    for (lane_bits, vector_bits) in [(32, 64), (32, 128), (64, 128)] {
        let wide = lane_bits == 64;
        let bits = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                (v as f32).to_bits() as u64
            }
        };
        let minimum = bits(if wide {
            f64::MIN_POSITIVE
        } else {
            f32::MIN_POSITIVE as f64
        });
        let snan = bits(f64::INFINITY) | 1;
        let lane = if wide { 1 } else { 3 };
        for (a, b) in [
            (bits(1.1), bits(-1.1)),
            (minimum, bits(0.5)),
            (minimum + 1, bits(0.5)),
            (2 * minimum - 1, bits(0.5)),
            (bits(-0.0), bits(-2.0)),
            (0, bits(f64::INFINITY)),
            (snan, bits(f64::NAN)),
            (bits(f64::NAN), snan),
            (1, bits(1.0)),
        ] {
            let first = pack([bits(2.0), a, snan, minimum], wide);
            let mut second = [snan; 4];
            second[lane as usize] = b;
            let second = pack(second, wide);
            for mode in 0u32..16 {
                let fpcr = mode << 22;
                let actual = match (lane_bits, vector_bits) {
                    (32, 64) => arm!("fmul v0.2s, v1.2s, v2.s[3]", first, second, fpcr),
                    (32, 128) => arm!("fmul v0.4s, v1.4s, v2.s[3]", first, second, fpcr),
                    (64, 128) => arm!("fmul v0.2d, v1.2d, v2.d[1]", first, second, fpcr),
                    _ => unreachable!(),
                };
                let expected = exact_vector_float_multiply_element(
                    first,
                    second,
                    lane_bits,
                    vector_bits,
                    lane,
                    fpcr,
                );
                assert_eq!(
                    actual,
                    (expected.bits, fp_status_bits(expected.status)),
                    "{first:x}*element({second:x}), lane_bits={lane_bits}, vector_bits={vector_bits}, FPCR={fpcr:x}"
                );
            }
        }
    }
}
