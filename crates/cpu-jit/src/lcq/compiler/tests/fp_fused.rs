use super::*;
use crate::lcq::fp::{CompletionError, complete_fused};
use nixe_cpu::{exception::ExceptionKind, execution::CpuExit};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

fn host_native_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
}

// This independent Arm execution checks the shared software oracle, including
// NaNs that deliberately never reach the guarded JIT native path.
#[cfg(target_arch = "aarch64")]
#[test]
fn exact_fused_semantics_match_arm_instructions() {
    use nixe_cpu::{
        decode::a64::fp_simd::FloatFusedMultiplyOperation as Op,
        semantics::a64_fp_simd::{exact_scalar_float_fused_multiply_add, fp_status_bits},
    };
    macro_rules! arm {
        ($inst:literal, $a:expr, $b:expr, $c:expr, $fpcr:expr) => {{
            let result: f64;
            let status: u64;
            // No exception enables. Save/restore the caller's complete FP state.
            unsafe { std::arch::asm!(
                "mrs {saved_control}, fpcr", "mrs {saved_status}, fpsr",
                "msr fpcr, {control}", "msr fpsr, xzr", $inst,
                "mrs {status}, fpsr", "msr fpcr, {saved_control}", "msr fpsr, {saved_status}",
                saved_control = out(reg) _, saved_status = out(reg) _,
                control = in(reg) u64::from($fpcr), status = out(reg) status,
                in("v1") f64::from_bits($a), in("v2") f64::from_bits($b), in("v3") f64::from_bits($c),
                lateout("v0") result, options(nostack),
            ); }
            (result.to_bits(), status as u32)
        }};
    }
    let q = 0x7ff8_0000_0000_0000;
    let s = 0x7ff0_0000_0000_0000;
    let inf = f64::INFINITY.to_bits();
    let one = 1.0f64.to_bits();
    for (a, b, c) in [
        (q | 1, q | 2, q | 3),
        (s | 1, one, q | 3),
        (s | 1, s | 2, s | 3),
        (inf, 0, q | 3),
        (inf, 0, s | 3),
        (inf, 1, q | 3),
        (1, one, q | 3),
        (one + 1, one - 2, one | (1 << 63)),
        (f64::MIN_POSITIVE.to_bits() + 1, 0.5f64.to_bits(), 0),
    ] {
        for fpcr in [0u32, 1 << 24, 1 << 25, 1 << 22, 2 << 22, 3 << 22] {
            for op in [
                Op::MultiplyAdd,
                Op::MultiplySubtract,
                Op::NegatedMultiplyAdd,
                Op::NegatedMultiplySubtract,
            ] {
                let actual = match op {
                    Op::MultiplyAdd => arm!("fmadd d0, d1, d2, d3", a, b, c, fpcr),
                    Op::MultiplySubtract => arm!("fmsub d0, d1, d2, d3", a, b, c, fpcr),
                    Op::NegatedMultiplyAdd => arm!("fnmadd d0, d1, d2, d3", a, b, c, fpcr),
                    Op::NegatedMultiplySubtract => arm!("fnmsub d0, d1, d2, d3", a, b, c, fpcr),
                };
                let expected = exact_scalar_float_fused_multiply_add(a, b, c, 64, op, fpcr);
                assert_eq!(
                    actual,
                    (expected.bits as u64, fp_status_bits(expected.status)),
                    "{op:?} {a:x} {b:x} {c:x} FPCR {fpcr:x}"
                );
            }
        }
    }
}

fn check(word: u32, first: u64, second: u64, third: u64, fpcr: u32) -> EdgeKind {
    // Dirty operands and lazy carry cross activation; the continuation consumes
    // carry and the multiplication result after either native or exact execution.
    let words = [
        0xb100_0400,
        0x9e67_0081,
        0x9e67_00a2,
        0x9e67_00e3,
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
    actual.general_register_storage_mut()[7] = third;
    actual.set_vector(0, u128::MAX);
    let mut expected = actual.clone();
    for &instruction in &words[..4] {
        execute_one(&TargetPlatform::Switch1, &mut expected, instruction).unwrap();
    }
    let prestate = expected.clone();
    let reference = execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
    let (_, exit) = execute_memory(
        &memory,
        if host_native_fma() { words.len() } else { 5 },
        &mut actual,
    );
    match exit.kind {
        EdgeKind::FpFused(operation) => {
            assert_eq!(actual, prestate);
            assert_eq!(exit.pc.get(), PC + 16);
            match complete_fused(operation, &mut actual) {
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
    for &instruction in &words[5..7] {
        execute_one(&TargetPlatform::Switch1, &mut expected, instruction).unwrap();
    }
    assert_eq!(
        actual, expected,
        "FMA {word:08x}: {first:x}*{second:x}+{third:x}, FPCR {fpcr:x}"
    );
    exit.kind
}

#[test]
fn scalar_fused_variants_preserve_single_rounding_and_exact_edges() {
    for wide in [false, true] {
        let bits = |v: f64| {
            if wide {
                v.to_bits()
            } else {
                (v as f32).to_bits() as u64
            }
        };
        let min = if wide {
            f64::MIN_POSITIVE.to_bits()
        } else {
            f32::MIN_POSITIVE.to_bits() as u64
        };
        let max = if wide {
            f64::MAX.to_bits()
        } else {
            f32::MAX.to_bits() as u64
        };
        let sign = if wide { 1 << 63 } else { 1 << 31 };
        let one = bits(1.0);
        for (base, np, na) in [
            (0x1f02_0c20, false, false),
            (0x1f02_8c20, true, false),
            (0x1f22_0c20, true, true),
            (0x1f22_8c20, false, true),
        ] {
            let word = base | ((wide as u32) << 22);
            if host_native_fma() {
                assert_eq!(
                    check(word, bits(2.0), bits(3.0), one, 0),
                    EdgeKind::Breakpoint(0)
                );
            }
            let cancel_sign = if np == na { sign } else { 0 };
            for (a, b, c) in [
                (bits(1.5), one + 1, bits(0.25)),
                (one + 1, one - 2, one | cancel_sign), // Fused cancellation, not zero.
                (max, bits(2.0), max | cancel_sign),   // Intermediate product would overflow.
                (0, sign, 0),
                (0, sign, sign),
                (min, bits(0.5), 0),
                (2 * min - 1, bits(0.5), 0),
                (min + 1, bits(1.5), min | cancel_sign),
            ] {
                for mode in 0..8 {
                    let poison = if wide { 0 } else { 0xffff_ffff_0000_0000 };
                    check(word, a | poison, b | poison, c | poison, mode << 22);
                }
            }
            let qnan = if wide {
                0x7ff8_0000_0000_0042
            } else {
                0x7fc0_0042
            };
            let snan = if wide {
                0x7ff0_0000_0000_0001
            } else {
                0x7f80_0001
            };
            for (a, b, c) in [
                (qnan, one, snan),
                (bits(f64::INFINITY), 0, qnan),
                (1, one, one),
                (one, one, bits(f64::INFINITY)),
            ] {
                for fpcr in [0, 1 << 25, 1 << 24, 1 << 8, 1 << 12, (1 << 24) | (1 << 15)] {
                    check(word, a, b, c, fpcr);
                }
            }
            for rd in [1, 2, 3, 31] {
                check((word & !31) | rd, bits(2.0), bits(3.0), one, 0);
                check((word & !31) | rd, snan, one, one, 0);
            }
            for fpcr in [1 << 10, 1 << 11, 1 << 12, (1 << 24) | (1 << 11)] {
                check(word, max, max, one, fpcr);
                check(word, min + 1, bits(0.5), 0, fpcr);
            }
        }
    }
}

#[test]
fn fused_capabilities_and_pending_status_use_real_boundaries() {
    let words = [0x1e62_1826, 0x1f42_0c20, 0xd420_0000]; // FDIV D6,D1,D2; FMADD D0,D1,D2,D3
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::new(abi).unwrap();
        if abi == HostAbi::X86_64 {
            let mut isa = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
            isa.set("has_avx", "true").unwrap();
            isa.set("has_fma", "true").unwrap();
            compiler.isa = isa.finish(compiler.isa.flags().clone()).unwrap();
        }
        let lowered = compiler
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.output.metadata.entries.len(), 2);
        let exact = lowered
            .states
            .iter()
            .find(|s| {
                s.exit
                    .is_some_and(|e| matches!(e.kind, EdgeKind::FpFused(_)))
            })
            .unwrap();
        assert!(exact.state.host_fpsr_pending && exact.state.dirty_live.fpsr);
        assert!(exact.state.dirty_live.vector[6]);
    }
    let mut actual = A64State::default();
    actual.set_pc(PC);
    actual.set_vector(1, u128::from(1.0f64.to_bits()));
    actual.set_vector(2, u128::from(3.0f64.to_bits()));
    actual.set_vector(3, 0x7ff0_0000_0000_0001);
    let mut expected = actual.clone();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
    let (_, exit) = execute_memory(&memory, if host_native_fma() { 3 } else { 2 }, &mut actual);
    assert_eq!(actual, expected);
    let EdgeKind::FpFused(operation) = exit.kind else {
        panic!("{exit:?}");
    };
    complete_fused(operation, &mut actual).unwrap();
    execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr(), 0x11);

    // Baseline x86 must never lower CLIF fma to a forbidden frameless libcall.
    let words = [0x1f42_0c20, 0xd420_0000];
    let memory = super::memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    let mut compiler = Compiler::new(HostAbi::X86_64).unwrap();
    compiler.isa = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
        .unwrap()
        .finish(compiler.isa.flags().clone())
        .unwrap();
    let lowered = compiler
        .lower(&fragment, CodeVersion::new(1).unwrap())
        .unwrap();
    assert_eq!(lowered.output.metadata.entries.len(), 1);
    assert!(lowered.states.iter().any(|s| {
        s.exit
            .is_some_and(|e| matches!(e.kind, EdgeKind::FpFused(_)))
    }));
    if cfg!(target_arch = "x86_64") {
        actual.set_pc(PC);
        actual.set_fpcr(0);
        actual.set_vector(3, u128::from(2.0f64.to_bits()));
        expected = actual.clone();
        let (_, exit) = execute_compiler(
            &memory,
            if host_native_fma() { 2 } else { 1 },
            &mut actual,
            compiler,
        );
        assert_eq!(actual, expected);
        let EdgeKind::FpFused(operation) = exit.kind else {
            panic!("{exit:?}");
        };
        complete_fused(operation, &mut actual).unwrap();
        execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
        assert_eq!(actual, expected);
    }
}
