use super::*;
use crate::lcq::fp::{CompletionError, complete_to_integer};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

fn check(word: u32, bits: u64, fpcr: u32) -> EdgeKind {
    // A dirty vector operand and lazy carry cross FP activation; conversion
    // writes must not confuse the scalar vector source with a same-index GPR.
    let words = [
        0xb100_0484,
        0x9e67_00a1,
        word,
        0x9a1f_0063,
        0x9100_0406,
        0xd420_0000,
    ];
    let memory = memory(&words);
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_fpcr(fpcr);
    actual.set_fpsr(1 << 27);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.general_register_storage_mut()[4] = u64::MAX;
    actual.general_register_storage_mut()[5] = bits;
    actual.set_vector(1, u128::MAX);
    actual.set_vector(31, u128::from(bits) | (u128::MAX << 64));
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    match exit.kind {
        EdgeKind::FpToInteger(operation) => {
            assert_eq!(
                actual, prestate,
                "PRE-state {word:08x}, {bits:x}, FPCR={fpcr:x}"
            );
            assert_eq!(exit.pc.get(), PC + 8);
            match complete_to_integer(operation, &mut actual) {
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
    assert_eq!(actual, expected, "{word:08x}, {bits:x}, FPCR={fpcr:x}");
    exit.kind
}

#[test]
fn truncating_fp_to_integer_matches_widths_boundaries_modes_and_traps() {
    for source_64 in [false, true] {
        let encode = |v: f64| {
            if source_64 {
                v.to_bits()
            } else {
                (v as f32).to_bits() as u64
            }
        };
        let sign = if source_64 { 1 << 63 } else { 1 << 31 };
        let normal = if source_64 {
            f64::MIN_POSITIVE.to_bits()
        } else {
            f32::MIN_POSITIVE.to_bits() as u64
        };
        let qnan = encode(f64::INFINITY) | if source_64 { 1 << 51 } else { 1 << 22 } | 0x42;
        let snan = encode(f64::INFINITY) | 0x42;
        for destination_64 in [false, true] {
            for signed in [false, true] {
                let word = 0x1e38_0020
                    | ((source_64 as u32) << 22)
                    | ((destination_64 as u32) << 31)
                    | ((!signed as u32) << 16);
                let limit =
                    encode(2.0f64.powi(if destination_64 { 64 } else { 32 } - i32::from(signed)));
                for bits in [
                    0,
                    sign,
                    encode(1.75),
                    encode(-1.75),
                    encode(-0.75),
                    normal,
                    normal - 1,
                    1,
                    limit - 1,
                    limit,
                    limit + 1,
                    sign | (limit - 1),
                    sign | limit,
                    sign | (limit + 1),
                    qnan,
                    snan,
                    encode(f64::INFINITY),
                    encode(f64::NEG_INFINITY),
                ] {
                    for mode in 0..16 {
                        check(word, bits, mode << 22);
                    }
                }
                for fpcr in [1 << 8, 1 << 12, (1 << 24) | (1 << 15)] {
                    for bits in [encode(2.0), encode(1.75), 1, limit, snan] {
                        check(word, bits, fpcr);
                        check(word | 31, bits, fpcr); // WZR/XZR still traps and contributes status.
                    }
                }
                for bits in [0, sign, encode(1.75), limit - 1] {
                    assert_eq!(check(word, bits, 0), EdgeKind::Breakpoint(0));
                }
                assert_eq!(check(word | 31, encode(1.75), 0), EdgeKind::Breakpoint(0));
                if !source_64 {
                    check(word, 0xdead_beef_0000_0000 | encode(1.75), 0);
                }
                if signed {
                    assert_eq!(check(word, sign | limit, 0), EdgeKind::Breakpoint(0));
                }
                // Source V31 is real; destination W/X1 does not alias V1.
                check((word & !(31 << 5)) | (31 << 5), encode(1.75), 0);
                check((word & !31) | 1, encode(-1.75), 0);
                // D -> W has fractional values immediately below INT_MIN.
                if source_64 && !destination_64 && signed {
                    check(word, (-2147483648.5f64).to_bits(), 0);
                }
            }
        }
    }
}

// Compare the shared exact provider with actual Arm instructions as well, so
// agreement between two users of that provider is not our only oracle.
#[cfg(target_arch = "aarch64")]
#[test]
fn truncating_fp_to_integer_exact_semantics_match_arm_instructions() {
    use nixe_cpu::{
        decode::a64::fp_simd::FloatToIntegerRounding,
        semantics::a64_fp_simd::{exact_float_to_integer, fp_status_bits},
    };
    macro_rules! arm {
        ($instruction:literal, $bits:expr, $fpcr:expr) => {{
            let result: u64;
            let status: u64;
            // Exceptions are masked; save/restore the complete host FP state.
            unsafe { std::arch::asm!(
                "mrs {saved_control}, fpcr", "mrs {saved_status}, fpsr",
                "msr fpcr, {control}", "msr fpsr, xzr", $instruction,
                "mrs {status}, fpsr", "msr fpcr, {saved_control}", "msr fpsr, {saved_status}",
                saved_control = out(reg) _, saved_status = out(reg) _,
                control = in(reg) u64::from($fpcr), status = out(reg) status,
                in("v1") f64::from_bits($bits), out("x0") result, options(nostack),
            ); }
            (result, status as u32)
        }};
    }
    for source_64 in [false, true] {
        let bits = |v: f64| {
            if source_64 {
                v.to_bits()
            } else {
                (v as f32).to_bits() as u64
            }
        };
        let snan = bits(f64::INFINITY) | 1;
        for input in [
            0,
            bits(-0.0),
            bits(1.75),
            bits(-0.75),
            bits(-1.75),
            1,
            bits(-2147483648.5),
            bits(2147483648.0),
            bits(4294967296.0),
            bits(2.0f64.powi(63)) - 1,
            bits(2.0f64.powi(63)),
            bits(2.0f64.powi(64)),
            bits(f64::INFINITY),
            bits(f64::NEG_INFINITY),
            bits(f64::NAN),
            snan,
        ] {
            for destination_64 in [false, true] {
                for signed in [false, true] {
                    for mode in 0u32..16 {
                        let fpcr = mode << 22;
                        let actual = match (source_64, destination_64, signed) {
                            (false, false, true) => arm!("fcvtzs w0, s1", input, fpcr),
                            (false, false, false) => arm!("fcvtzu w0, s1", input, fpcr),
                            (false, true, true) => arm!("fcvtzs x0, s1", input, fpcr),
                            (false, true, false) => arm!("fcvtzu x0, s1", input, fpcr),
                            (true, false, true) => arm!("fcvtzs w0, d1", input, fpcr),
                            (true, false, false) => arm!("fcvtzu w0, d1", input, fpcr),
                            (true, true, true) => arm!("fcvtzs x0, d1", input, fpcr),
                            (true, true, false) => arm!("fcvtzu x0, d1", input, fpcr),
                        };
                        let expected = exact_float_to_integer(
                            input,
                            if source_64 { 64 } else { 32 },
                            if destination_64 { 64 } else { 32 },
                            signed,
                            FloatToIntegerRounding::TowardZero,
                            0,
                            fpcr,
                        );
                        let value = if destination_64 {
                            expected.value
                        } else {
                            u64::from(expected.value as u32)
                        };
                        assert_eq!(
                            actual,
                            (value, fp_status_bits(expected.status)),
                            "input={input:x}, S64={source_64}, D64={destination_64}, signed={signed}, FPCR={fpcr:x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn discarded_fp_to_integer_results_keep_status_and_precise_exact_maps() {
    // Discarded FCVTZS sets IXC; a following exact conversion sees that status
    // before its own IOC. Dirty X0/lazy flags remain live across activation.
    let words = [0xb100_0400, 0x9e78_003f, 0x9e78_0043, 0xd420_0000];
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
                    e.pc.get() == PC + 8 && matches!(e.kind, EdgeKind::FpToInteger(_))
                })
            })
            .unwrap();
        assert!(exit.state.host_fpsr_pending && exit.state.dirty_live.fpsr);
        assert!(exit.state.dirty_live.integer.x[0]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.general_register_storage_mut()[0] = u64::MAX;
    actual.set_vector(1, u128::from(1.75f64.to_bits()));
    actual.set_vector(2, u128::from(f64::NAN.to_bits()));
    let mut expected = actual.clone();
    for &word in &words[..2] {
        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    }
    let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 1 << 4);
    let EdgeKind::FpToInteger(operation) = exit.kind else {
        panic!("{exit:?}")
    };
    complete_to_integer(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[2]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x11);
    // An overwritten (not just XZR) result also has observable FP effects.
    let words = [0x9e78_0020, 0xd280_0000, 0xd420_0000];
    actual.set_pc(PC);
    actual.set_fpsr(0);
    execute(&words, &mut actual);
    assert_eq!(actual.fpsr(), 1 << 4);
}
