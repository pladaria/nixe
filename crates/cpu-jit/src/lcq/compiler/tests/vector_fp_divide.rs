use super::*;
use crate::lcq::fp::{CompletionError, complete_vector_divide};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

fn pack(lanes: [u64; 4], wide: bool) -> u128 {
    if wide {
        u128::from(lanes[0]) | (u128::from(lanes[1]) << 64)
    } else {
        lanes
            .into_iter()
            .enumerate()
            .fold(0, |v, (i, lane)| v | (u128::from(lane as u32) << (i * 32)))
    }
}

fn check(word: u32, first: u128, second: u128, fpcr: u32, baseline: bool) -> EdgeKind {
    // Dirty vector sources and deferred carry cross activation and completion.
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
        EdgeKind::VectorFpDivide(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 12);
            match complete_vector_divide(operation, &mut actual) {
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
        "{word:08x}, {first:x}/{second:x}, FPCR={fpcr:x}, baseline={baseline}"
    );
    exit.kind
}

#[test]
fn packed_division_matches_active_lanes_modes_tiny_results_and_traps() {
    for word in [0x2e22_fc20, 0x6e22_fc20, 0x6e62_fc20] {
        let wide = word & (1 << 22) != 0;
        let lanes = if wide || word & (1 << 30) == 0 { 2 } else { 4 };
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
        for (index, (a, b)) in [
            (bits(1.0), bits(3.0)),
            (bits(-1.0), bits(3.0)),
            (bits(-0.0), bits(-2.0)),
            (maximum, bits(0.5)),
            (bits(1.0), bits(0.0)),
            (0, 0),
            (bits(f64::INFINITY), bits(f64::INFINITY)),
            (bits(2.0), bits(f64::INFINITY)),
            (bits(f64::NAN), snan),
            (snan, bits(f64::NAN)),
            (1, bits(1.0)),
            (bits(1.0), 1),
            (minimum, bits(2.0)),
            (minimum + 1, bits(3.0)),
            (2 * minimum - 1, bits(2.0)),
            (minimum, maximum),
        ]
        .into_iter()
        .enumerate()
        {
            let mut first = [bits(6.0); 4];
            let mut second = [bits(2.0); 4];
            first[index % lanes] = a;
            second[index % lanes] = b;
            if !wide && lanes == 2 {
                // Inactive sNaN/subnormal/zero must not affect guards or FPSR.
                first[2..].copy_from_slice(&[snan, 1]);
                second[2..].copy_from_slice(&[0, snan]);
            }
            let (first, second) = (pack(first, wide), pack(second, wide));
            for mode in 0..16 {
                check(word, first, second, mode << 22, false);
            }
            for fpcr in [
                1 << 8,
                1 << 9,
                1 << 10,
                1 << 11,
                1 << 12,
                (1 << 24) | (1 << 15),
            ] {
                check(word, first, second, fpcr, false);
            }
        }
        for mode in 0..4 {
            let first = if !wide && lanes == 2 {
                [bits(1.0), bits(1.0), snan, 1]
            } else {
                [bits(1.0); 4]
            };
            let second = if !wide && lanes == 2 {
                [bits(3.0), bits(3.0), 0, snan]
            } else {
                [bits(3.0); 4]
            };
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
        for alias in [
            (word & !31) | 1,
            (word & !31) | 2,
            (word & !31) | 31,
            (word & !(31 << 5)) | (31 << 5),
            (word & !(31 << 16)) | (31 << 16),
        ] {
            check(
                alias,
                pack([bits(6.0); 4], wide),
                pack([bits(2.0); 4], wide),
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
fn overwritten_packed_divide_keeps_status_and_exact_maps() {
    // Native 1/3 is overwritten; the next packed divide exits before 1/0.
    let words = [0x6e22_fc20, 0x6e20_1c00, 0x6e24_fca0, 0xd420_0000];
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
                    e.pc.get() == PC + 8 && matches!(e.kind, EdgeKind::VectorFpDivide(_))
                })
            })
            .unwrap();
        assert!(exact.state.host_fpsr_pending && exact.state.dirty_live.fpsr);
        assert!(exact.state.dirty_live.vector[0]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_vector(1, pack([1.0f32.to_bits() as u64; 4], false));
    actual.set_vector(2, pack([3.0f32.to_bits() as u64; 4], false));
    actual.set_vector(5, pack([1.0f32.to_bits() as u64; 4], false));
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 1 << 4);
    let EdgeKind::VectorFpDivide(operation) = exit.kind else {
        panic!("{exit:?}")
    };
    complete_vector_divide(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x12);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn packed_divide_exact_semantics_match_arm_instructions() {
    use nixe_cpu::semantics::a64_fp_simd::{exact_vector_float_divide, fp_status_bits};
    macro_rules! arm {
        ($instruction:literal, $first:expr, $second:expr, $fpcr:expr) => {{
            let mut result = 0u128;
            let status: u64;
            // Exceptions masked; preserve the complete caller FP environment.
            unsafe { std::arch::asm!(
                "mrs {saved_control}, fpcr", "mrs {saved_status}, fpsr",
                "msr fpcr, {control}", "msr fpsr, xzr",
                "ldr q1, [{first}]", "ldr q2, [{second}]", $instruction,
                "str q0, [{result}]", "mrs {status}, fpsr",
                "msr fpcr, {saved_control}", "msr fpsr, {saved_status}",
                saved_control = out(reg) _, saved_status = out(reg) _,
                control = in(reg) u64::from($fpcr), status = out(reg) status,
                first = in(reg) &$first, second = in(reg) &$second,
                result = in(reg) &mut result, out("v0") _, out("v1") _, out("v2") _,
                options(nostack),
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
        for (a, b) in [
            (bits(1.0), bits(3.0)),
            (minimum, bits(2.0)),
            (minimum + 1, bits(3.0)),
            (2 * minimum - 1, bits(2.0)),
            (bits(-0.0), bits(-2.0)),
            (0, 0),
            (bits(1.0), 0),
            (snan, bits(f64::NAN)),
            (bits(f64::NAN), snan),
            (1, bits(1.0)),
        ] {
            let first = pack([bits(6.0), a, snan, minimum], wide);
            let second = pack([bits(2.0), b, 0, bits(3.0)], wide);
            for mode in 0u32..16 {
                let fpcr = mode << 22;
                let actual = match (lane_bits, vector_bits) {
                    (32, 64) => arm!("fdiv v0.2s, v1.2s, v2.2s", first, second, fpcr),
                    (32, 128) => arm!("fdiv v0.4s, v1.4s, v2.4s", first, second, fpcr),
                    (64, 128) => arm!("fdiv v0.2d, v1.2d, v2.2d", first, second, fpcr),
                    _ => unreachable!(),
                };
                let expected =
                    exact_vector_float_divide(first, second, lane_bits, vector_bits, fpcr);
                assert_eq!(
                    actual,
                    (expected.bits, fp_status_bits(expected.status)),
                    "{first:x}/{second:x}, lane_bits={lane_bits}, vector_bits={vector_bits}, FPCR={fpcr:x}"
                );
            }
        }
    }
}
