use super::*;
use nixe_cpu_interpreter::{InstructionStep, execute_one};

// Also exercise the optimized allocator/egraph contract without changing the
// production LCQ compilation settings.
fn compiler(opt: &str, allocator: &str) -> Compiler {
    let mut compiler = Compiler::new(native_abi()).unwrap();
    let mut flags = settings::builder();
    for flag in compiler.isa.flags().iter() {
        flags.set(flag.name, &flag.value_string()).unwrap();
    }
    flags.set("opt_level", opt).unwrap();
    flags.set("regalloc_algorithm", allocator).unwrap();
    let mut target = cranelift_native::builder().unwrap();
    if native_abi() == HostAbi::Aarch64 {
        target.set("use_bti", "true").unwrap();
    }
    compiler.isa = target.finish(settings::Flags::new(flags)).unwrap();
    compiler
}

#[test]
fn overwritten_native_fp_results_retain_status_with_both_optimization_modes() {
    // All operands are inside their native domain. Each first operation sets
    // IXC, but its entire vector result is then overwritten by FMOV D0,XZR.
    let cases = [
        (0x1e62_2820, 1.0, 2.0f64.powi(-53)), // FADD D0,D1,D2
        (0x1e62_3820, 1.0, 2.0f64.powi(-54)), // FSUB
        (0x1e62_0820, 1.5, f64::from_bits(1.0f64.to_bits() + 1)), // FMUL
        (0x1e62_1820, 1.0, 3.0),              // FDIV
        (0x1e61_c020, 2.0, 0.0),              // FSQRT D0,D1
        (0x1e62_4020, 1.0 + 2.0f64.powi(-24), 0.0), // FCVT S0,D1
    ];
    for (opt, allocator) in [("none", "single_pass"), ("speed", "backtracking")] {
        for (word, first, second) in cases {
            let words = [word, 0x9e67_03e0, 0xd420_0000];
            let memory = memory(&words);
            for rounding in 0..4 {
                let mut actual = A64State::default();
                actual.set_pc(PC);
                actual.set_fpcr(rounding << 22);
                actual.set_fpsr(1 << 27);
                actual.set_vector(1, u128::from(first.to_bits()));
                actual.set_vector(2, u128::from(second.to_bits()));
                let mut expected = actual.clone();
                for &word in &words[..2] {
                    assert_eq!(
                        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap(),
                        InstructionStep::Continue
                    );
                }
                let (_, exit) = execute_compiler(&memory, 3, &mut actual, compiler(opt, allocator));
                assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
                assert_eq!(actual, expected, "{word:08x}, {opt}, rounding={rounding}");
                assert_eq!(actual.fpsr(), (1 << 27) | (1 << 4));
            }
        }
    }
}

#[test]
fn constant_integer_to_fp_uses_guest_rounding_and_retains_discarded_status() {
    // Activate with an exact conversion first so the constant producer and
    // inexact conversions occupy the same native continuation/egraph region.
    // MOVZ/MOVK X1 forms 2^24+1. One SCVTF result survives and another is dead.
    let words = [
        0x9e22_03e7, // SCVTF S7,XZR
        0xd280_0021, // MOVZ X1,#1
        0xf2a0_2001, // MOVK X1,#0x100,LSL#16
        0x9e22_0020, // SCVTF S0,X1
        0x9e22_0022, // SCVTF S2,X1
        0x9e67_03e2, // FMOV D2,XZR
        0xd420_0000,
    ];
    let memory = memory(&words);
    for (opt, allocator) in [("none", "single_pass"), ("speed", "backtracking")] {
        for rounding in 0..4 {
            let mut actual = A64State::default();
            actual.set_pc(PC);
            actual.set_fpcr(rounding << 22);
            let mut expected = actual.clone();
            for &word in &words[..6] {
                assert_eq!(
                    execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap(),
                    InstructionStep::Continue
                );
            }
            let (_, exit) =
                execute_compiler(&memory, words.len(), &mut actual, compiler(opt, allocator));
            assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
            assert_eq!(actual, expected, "{opt}, rounding={rounding}");
            assert_eq!(actual.fpsr(), 1 << 4);
        }
    }
}
