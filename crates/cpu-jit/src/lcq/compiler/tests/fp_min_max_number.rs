use super::*;
use crate::lcq::fp::{CompletionError, complete_min_max_number};
use nixe_cpu_interpreter::{InstructionStep, execute_one};

#[test]
fn fminmaxnm_native_and_exact_paths_preserve_results_status_and_traps() {
    for minimum in [false, true] {
        for wide in [false, true] {
            let bits = |v: f64| {
                if wide {
                    v.to_bits()
                } else {
                    u64::from((v as f32).to_bits())
                }
            };
            let sign = 1_u64 << if wide { 63 } else { 31 };
            let quiet = 1_u64 << if wide { 51 } else { 22 };
            let snan = bits(f64::INFINITY) | 0x123;
            let qnan = snan | quiet | sign;
            for (a, b) in [
                (bits(1.0), bits(2.0)),
                (bits(-1.0), bits(-2.0)),
                (sign, 0),
                (0, sign),
                (sign, sign),
                (1, sign | 1),
                (qnan, bits(2.0)),
                (bits(2.0), qnan),
                (snan, qnan),
                (qnan, snan),
                (qnan, qnan),
                (bits(f64::NEG_INFINITY), bits(f64::INFINITY)),
            ] {
                for fpcr in [0, 3 << 22, 1 << 24, 1 << 25, 1 << 8, (1 << 24) | (1 << 15)] {
                    // Destination aliases a source and both source register 31 and
                    // pre-existing flags/status must survive both execution paths.
                    let word = 0x1e3f_681f | (u32::from(wide) << 22) | (u32::from(minimum) << 12);
                    let mut actual = A64State::default();
                    actual.set_pc(PC);
                    actual.set_fpcr(fpcr);
                    actual.set_fpsr(1 << 27);
                    actual.set_nzcv(Nzcv::from_bits(0xb000_0000));
                    actual.set_vector(0, u128::from(a) | (u128::MAX << 64));
                    actual.set_vector(31, u128::from(b) | (u128::MAX << 64));
                    let before = actual.clone();
                    let mut expected = actual.clone();
                    let reference =
                        execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
                    let (_, exit) = execute(&[word, 0xd420_0000], &mut actual);
                    match exit.kind {
                        EdgeKind::FpMinMaxNumber(op) => {
                            assert_eq!(actual, before);
                            match complete_min_max_number(op, &mut actual) {
                                Ok(()) => assert_eq!(reference, InstructionStep::Continue),
                                Err(CompletionError::Trap(_)) => {
                                    assert!(matches!(reference, InstructionStep::Exit(_)));
                                    assert_eq!(actual, before);
                                }
                                Err(error) => panic!("{error:?}"),
                            }
                        }
                        EdgeKind::Breakpoint(0) => assert_eq!(reference, InstructionStep::Continue),
                        other => panic!("{other:?}"),
                    }
                    assert_eq!(actual, expected, "{word:x} {a:x} {b:x} fpcr={fpcr:x}");
                    if a == bits(1.0) && b == bits(2.0) {
                        assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
                    }
                }
            }
        }
    }
}

#[test]
fn fminmaxnm_exact_exit_preserves_pending_native_fp_status_for_both_hosts() {
    for minimum in [false, true] {
        let words = [
            0x1e62_1823,
            0x1e64_68a0 | (u32::from(minimum) << 12),
            0xd420_0000,
        ];
        let memory = memory(&words);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let lowered = Compiler::new(abi)
                .unwrap()
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            let exact = lowered
                .states
                .iter()
                .find(|s| {
                    s.exit.is_some_and(|e| {
                        e.pc.get() == PC + 4 && matches!(e.kind, EdgeKind::FpMinMaxNumber(_))
                    })
                })
                .unwrap();
            assert!(exact.state.host_fpsr_pending && exact.state.dirty_live.fpsr);
            assert!(exact.state.dirty_live.vector.contains(3));
            assert!(!exact.state.dirty_live.vector.contains(0));
        }
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.set_vector(1, u128::from(1.0f64.to_bits()));
        actual.set_vector(2, u128::from(3.0f64.to_bits()));
        actual.set_vector(5, 0x7ff0_0000_0000_0001);
        let mut expected = actual.clone();
        execute_one(&TargetPlatform::Switch1, &mut expected, words[0]).unwrap();
        let (_, exit) = execute_memory(&memory, words.len(), &mut actual);
        assert_eq!(actual, expected);
        let EdgeKind::FpMinMaxNumber(op) = exit.kind else {
            panic!("{exit:?}")
        };
        complete_min_max_number(op, &mut actual).unwrap();
        execute_one(&TargetPlatform::Switch1, &mut expected, words[1]).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.fpsr(), 0x11);
    }
}

#[test]
fn fminnm_captured_encoding_executes_natively_with_overlapping_destination() {
    let mut state = A64State::default();
    state.set_pc(PC);
    state.set_vector(0, u128::from(3.0_f32.to_bits()));
    state.set_vector(2, u128::from(8.0_f32.to_bits()) | (u128::MAX << 32));
    super::integer::compare(0x1e20_7842, state);
}
