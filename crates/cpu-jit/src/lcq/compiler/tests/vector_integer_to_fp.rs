use super::*;
use crate::lcq::fp::{CompletionError, complete_from_vector_integer};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

#[test]
fn unsigned_vector_zero_stays_positive_under_round_down() {
    let words = [0x6e61_d820, 0xd420_0000]; // UCVTF V0.2D,V1.2D
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpcr(2 << 22);
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    execute(&words, &mut actual);
    assert_eq!(actual, expected);
}

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
        EdgeKind::VectorIntegerToFp(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 8);
            match complete_from_vector_integer(operation, &mut actual) {
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
fn simd_integer_conversions_match_active_lanes_rounding_and_traps() {
    for shape in [
        0x0e21_d820,
        0x4e21_d820,
        0x4e61_d820,
        0x5e21_d820,
        0x5e61_d820,
    ] {
        let lane_64 = shape & (1 << 22) != 0;
        let precision = if lane_64 { 53 } else { 24 };
        let pack = |values: [u64; 4]| -> u128 {
            if lane_64 {
                u128::from(values[0]) | (u128::from(values[1]) << 64)
            } else {
                values
                    .into_iter()
                    .enumerate()
                    .fold(0, |acc, (i, v)| acc | (u128::from(v as u32) << (i * 32)))
            }
        };
        let inexact = (1 << precision) + 1;
        for unsigned in [false, true] {
            let word = shape | ((unsigned as u32) << 29);
            for lanes in [
                [0; 4],
                [1; 4],
                [u64::MAX; 4],
                [inexact, 1, 0, u64::MAX],
                [0, inexact, inexact, inexact],
                [1 << 63, (1 << 63) + 1, 0x8000_0000, 0x8000_0001],
                [1, 1, inexact, inexact],
            ] {
                for mode in 0..16 {
                    assert_eq!(
                        check(word, pack(lanes), mode << 22, false),
                        EdgeKind::Breakpoint(0)
                    );
                }
                check(word, pack(lanes), 1 << 12, false); // IXE: all-or-nothing lane commit.
            }
            // Selected-ISA fallback sequences must work without optional x86
            // features; caller hardware features must not leak into emission.
            for mode in 0..4 {
                check(
                    word,
                    pack([0, inexact, u64::MAX, inexact]),
                    mode << 22,
                    true,
                );
            }
            for alias in [
                (word & !31) | 1,
                (word & !31) | 31,
                (word & !(31 << 5)) | (31 << 5),
            ] {
                check(alias, pack([inexact; 4]), 0, false);
                check(alias, pack([inexact; 4]), 1 << 12, false);
            }
        }
    }
}

#[test]
fn overwritten_vector_conversion_keeps_status_at_precise_exit() {
    // The packed result is overwritten before an exact scalar conversion.
    // IXC must still be visible at its PRE-state, alongside dirty lazy NZCV.
    let words = [
        0xb100_0400,
        0x4e21_d820, // SCVTF V0.4S,V1.4S
        0x6e20_1c00, // EOR V0.16B,V0.16B,V0.16B
        0x9e78_0043, // FCVTZS X3,D2
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
        let exit = lowered
            .states
            .iter()
            .find(|s| {
                s.exit.is_some_and(|e| {
                    e.pc.get() == PC + 12 && matches!(e.kind, EdgeKind::FpToInteger(_))
                })
            })
            .unwrap();
        assert!(exit.state.host_fpsr_pending && exit.state.dirty_live.fpsr);
        assert!(exit.state.dirty_live.integer.x[0]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.set_vector(1, (1 << 24) + 1);
    actual.set_vector(2, u128::from(f64::NAN.to_bits()));
    let mut expected = actual.clone();
    for &word in &words[..3] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 1 << 4);
    let EdgeKind::FpToInteger(operation) = exit.kind else {
        panic!("{exit:?}")
    };
    crate::lcq::fp::complete_to_integer(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[3]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x11);
}

// The interpreter and exact completion share a provider. Check it against Arm
// instructions too, including inactive lanes and all supported rounding modes.
#[cfg(target_arch = "aarch64")]
#[test]
fn simd_integer_conversion_exact_semantics_match_arm_instructions() {
    use nixe_cpu::semantics::a64_fp_simd::exact_vector_integer_to_float;
    macro_rules! arm {
        ($instruction:literal, $source:expr, $fpcr:expr) => {{
            let mut result = 0u128;
            let status: u64;
            // Exceptions are masked; restore both caller FP registers.
            unsafe { std::arch::asm!(
                "mrs {saved_control}, fpcr", "mrs {saved_status}, fpsr",
                "msr fpcr, {control}", "msr fpsr, xzr",
                "ldr q1, [{source}]", $instruction, "str q0, [{result}]",
                "mrs {status}, fpsr", "msr fpcr, {saved_control}", "msr fpsr, {saved_status}",
                saved_control = out(reg) _, saved_status = out(reg) _,
                control = in(reg) u64::from($fpcr), status = out(reg) status,
                source = in(reg) &$source, result = in(reg) &mut result,
                out("v0") _, out("v1") _, options(nostack),
            ); }
            (result, status as u32)
        }};
    }
    for (lane_bits, vector_bits) in [(32, 32), (32, 64), (32, 128), (64, 64), (64, 128)] {
        for source in [
            0u128,
            u128::MAX,
            0x8000_0001_8000_0000_0000_0001_0100_0001,
            0x0020_0000_0000_0001_8000_0000_0000_0000,
            0x0020_0000_0000_0001_0000_0000_0000_0001,
            0x0100_0001_0100_0001_0000_0001_0000_0001,
        ] {
            for signed in [false, true] {
                for mode in 0u32..16 {
                    let fpcr = mode << 22;
                    let actual = match (lane_bits, vector_bits, signed) {
                        (32, 32, true) => arm!("scvtf s0, s1", source, fpcr),
                        (32, 32, false) => arm!("ucvtf s0, s1", source, fpcr),
                        (32, 64, true) => arm!("scvtf v0.2s, v1.2s", source, fpcr),
                        (32, 64, false) => arm!("ucvtf v0.2s, v1.2s", source, fpcr),
                        (32, 128, true) => arm!("scvtf v0.4s, v1.4s", source, fpcr),
                        (32, 128, false) => arm!("ucvtf v0.4s, v1.4s", source, fpcr),
                        (64, 64, true) => arm!("scvtf d0, d1", source, fpcr),
                        (64, 64, false) => arm!("ucvtf d0, d1", source, fpcr),
                        (64, 128, true) => arm!("scvtf v0.2d, v1.2d", source, fpcr),
                        (64, 128, false) => arm!("ucvtf v0.2d, v1.2d", source, fpcr),
                        _ => unreachable!(),
                    };
                    let (value, inexact) =
                        exact_vector_integer_to_float(source, lane_bits, vector_bits, signed, fpcr);
                    assert_eq!(
                        actual,
                        (value, u32::from(inexact) << 4),
                        "{source:x}, lane_bits={lane_bits}, vector_bits={vector_bits}, signed={signed}, FPCR={fpcr:x}"
                    );
                }
            }
        }
    }
}
