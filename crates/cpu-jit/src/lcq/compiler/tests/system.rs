use super::*;
use crate::abi::FpSystemOperation;
use crate::lcq::system::{CompletionError, RuntimeServices, complete_runtime};
use nixe_cpu::exclusive::{ExclusiveMonitorState, ExclusiveReservation};
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::MemoryValue;
use nixe_cpu_interpreter::{InstructionStep, InterpreterContext, execute_one_with_context};
use std::cell::{Cell, RefCell};

struct CountingTimer(Cell<u32>);
impl ArchitecturalTimer for CountingTimer {
    fn snapshot(&self) -> TimerSnapshot {
        self.0.set(self.0.get() + 1);
        TimerSnapshot {
            counter: 0x1234_5678_9abc_def0,
            frequency: 19_200_000,
        }
    }
}

#[test]
fn runtime_system_helpers_match_interpreter_after_protected_native_exit() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for word in [
        0xd53b_e000u32,
        0xd53b_e020,
        0xd53b_e03f, // timer reads, including XZR
        0xd503_203f,
        0xd503_205f,
        0xd503_207f,
        0xd503_209f,
        0xd503_20bf, // hints
        0xd503_3bbf,
        0xd503_3f9f,
        0xd503_3fdf, // DMB ISH, DSB SY, ISB
        0xd503_3f5f, // CLREX
        0xd508_751f,
        0xd50b_7520,
        0xd508_7620,
        0xd50b_7b20,
        // CIVAC has native-probe coverage in lifetime::memory::tests::data_cache.
    ] {
        for event_mode in 0..3 {
            // SUBS creates dirty helper input/NZCV with C=1; ADC consumes it after a
            // successful completion through a newly demanded continuation.
            let words = [0xf100_1000, word, 0x9a1f_0063, 0xd420_0000];
            let memory = memory(&words);
            let timer = CountingTimer(Cell::new(0));
            let events = VcpuEventState::default();
            let expected_events = VcpuEventState::default();
            if event_mode == 1 {
                events.signal_event();
                expected_events.signal_event();
            }
            if event_mode == 2 {
                events.post_interrupts(8);
                expected_events.post_interrupts(8);
            }
            let mut exclusive = ExclusiveMonitorState::default();
            exclusive.reserve(ExclusiveReservation {
                page: GuestPhysicalPageId::new(1),
                byte_offset: 0,
                access_size: 8,
                expected: MemoryValue::U64(7),
            });
            let expected_exclusive = RefCell::new(exclusive);
            let mut actual = A64State::default();
            actual.set_pc(PC);
            actual.general_register_storage_mut()[0] = PC + 4;
            actual.set_fpsr(1 << 27);
            let mut expected = actual.clone();
            expected.set_fpsr((1 << 27) | 2); // Seeded real host FP contribution.
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, words[0])
                .unwrap();
            let (reason, exit) = execute_with_fp(
                &memory,
                2,
                &mut actual,
                Compiler::new(native_abi()).unwrap(),
                true,
            );
            assert_eq!(reason, NativeExitReason::Architectural);
            assert_eq!(actual, expected); // Helper has not run, FP completion has.
            assert_eq!(timer.0.get(), 0);
            assert_eq!(exit.pc.get(), PC + 4);
            let EdgeKind::RuntimeSystem(operation) = exit.kind else {
                panic!("{word:08x}: {exit:?}")
            };
            let mut services = RuntimeServices {
                address_space: SPACE,
                memory: &memory,
                timer: &timer,
                events: &events,
                exclusive: &mut exclusive,
            };
            let actual_schedule = complete_runtime(operation, &mut actual, &mut services).unwrap();
            let context = InterpreterContext::new(
                ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                &memory,
                &expected_exclusive,
                &timer,
                &expected_events,
            );
            let expected_schedule =
                match execute_one_with_context(context, &mut expected, word).unwrap() {
                    InstructionStep::Continue => None,
                    InstructionStep::Exit(CpuExit::Scheduled { request, source }) => {
                        assert_eq!(source.pc.get(), exit.pc.get());
                        Some(request)
                    }
                    other => panic!("unexpected {word:08x} outcome: {other:?}"),
                };
            assert_eq!(
                actual_schedule, expected_schedule,
                "{word:08x}, events {event_mode}"
            );
            assert_eq!(actual, expected, "{word:08x}");
            assert_eq!(exclusive, *expected_exclusive.borrow());
            assert_eq!(events.consume_event(), expected_events.consume_event());
            assert_eq!(
                events.take_pending_interrupts(),
                expected_events.take_pending_interrupts()
            );
            assert_eq!(
                timer.0.get(),
                if word & 0xffff_ffc0 == 0xd53b_e000 {
                    2
                } else {
                    0
                }
            );
            if actual_schedule.is_none() {
                let (_, continuation) = execute_memory(&memory, 2, &mut actual);
                assert_eq!(continuation.kind, EdgeKind::Breakpoint(0));
                nixe_cpu_interpreter::execute_one(
                    &TargetPlatform::Switch1,
                    &mut expected,
                    words[2],
                )
                .unwrap();
                assert_eq!(actual, expected, "helper continuation {word:08x}");
            }
        }
    }
}

#[test]
fn cache_completion_publishes_one_invalidation_and_preserves_fault_identity() {
    for address in [PC, 0xdead_0000] {
        let words = [0xb100_1000, 0xd50b_7520]; // ADDS then IC IVAU,X0
        let memory = memory(&words);
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.general_register_storage_mut()[0] = address - 4;
        let (reason, exit) = execute_memory(&memory, words.len(), &mut actual);
        assert_eq!(reason, NativeExitReason::Architectural);
        let before = actual.clone();
        let cursor = memory.invalidation_cursor();
        let timer = CountingTimer(Cell::new(0));
        let events = VcpuEventState::default();
        let mut exclusive = ExclusiveMonitorState::default();
        let mut services = RuntimeServices {
            address_space: SPACE,
            memory: &memory,
            timer: &timer,
            events: &events,
            exclusive: &mut exclusive,
        };
        let EdgeKind::RuntimeSystem(operation) = exit.kind else {
            panic!()
        };
        let result = complete_runtime(operation, &mut actual, &mut services);
        let mut invalidations = Vec::new();
        memory
            .read_invalidations_since(cursor, &mut invalidations)
            .unwrap();
        if address == PC {
            assert_eq!(result.unwrap(), None);
            assert_eq!(actual.pc(), PC + 8);
            assert_eq!(invalidations.len(), 1);
        } else {
            let Err(CompletionError::Memory(fault)) = result else {
                panic!("expected cache address fault")
            };
            assert_eq!(fault.address, GuestVirtualAddress::new(address));
            assert_eq!(exit.pc.get(), PC + 4);
            assert_eq!(actual, before);
            assert!(invalidations.is_empty());
        }
    }
}

#[test]
fn inline_system_registers_preserve_ssa_and_mask_reserved_nzcv_bits() {
    let words = [
        0xd51b_4200, // MSR NZCV,X0
        0xd53b_4201, // MRS X1,NZCV: reserved bits must already be zero.
        0xd51b_d041, // MSR TPIDR_EL0,X1
        0xd53b_d042, // MRS X2,TPIDR_EL0: sees the new SSA value.
        0xd53b_d063, // MRS X3,TPIDRRO_EL0
        0xd53b_4404, // MRS X4,FPCR
        0xd53b_0025, // MRS X5,CTR_EL0
        0xd53b_00e6, // MRS X6,DCZID_EL0
        0xd503_20ff, // XPACLRI: no-op for Switch 1, not all profiles.
        0xd420_0000,
    ];
    for bits in [0, u64::MAX, 0xface_beef_afff_ffff, 0x1234_5678] {
        let mut expected = A64State::default();
        expected.set_pc(PC);
        expected.general_register_storage_mut()[0] = bits;
        expected.set_tpidr_el0(0x1234);
        expected.set_tpidrro_el0_from_runtime(0x8000_0000_1234_5678);
        expected.set_fpcr(0xdead_beef);
        let mut actual = expected.clone();
        for word in &words[..words.len() - 1] {
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word)
                .unwrap();
        }
        execute(&words, &mut actual);
        assert_eq!(actual, expected);
    }
    let memory = memory(&words);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(
                &Fragment::capture(&memory, key()).unwrap(),
                CodeVersion::new(1).unwrap(),
            )
            .unwrap();
        assert!(lowered.states[0].state.dirty_live.tpidr_el0);
        assert!(!lowered.states[0].state.dirty_live.fpcr);
    }
}

#[test]
fn lazy_flags_feed_system_reads_and_zero_register_system_accesses() {
    let words = [
        0xf100_041f, // CMP X0,#1
        0xd53b_4201, // MRS X1,NZCV from the subtraction recipe.
        0xd51b_421f, // MSR NZCV,XZR
        0xd53b_4202, // MRS X2,NZCV
        0xd51b_d05f, // MSR TPIDR_EL0,XZR
        0xd53b_d05f, // MRS XZR,TPIDR_EL0
        0xd420_0000,
    ];
    for x0 in [0, 1, u64::MAX] {
        let mut expected = A64State::default();
        expected.set_pc(PC);
        expected.general_register_storage_mut()[0] = x0;
        expected.set_tpidr_el0(u64::MAX);
        let mut actual = expected.clone();
        for word in &words[..words.len() - 1] {
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word)
                .unwrap();
        }
        execute(&words, &mut actual);
        assert_eq!(actual, expected);
    }
}

#[test]
fn fp_observation_and_replacement_complete_after_real_gateway_fp_merge() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for (word, operation) in [
        (0xd53b_4420, FpSystemOperation::ReadStatus { rt: 0 }),
        (0xd53b_443f, FpSystemOperation::ReadStatus { rt: 31 }),
        (0xd51b_4400, FpSystemOperation::WriteControl { rt: 0 }),
        (0xd51b_441f, FpSystemOperation::WriteControl { rt: 31 }),
        (0xd51b_4420, FpSystemOperation::WriteStatus { rt: 0 }),
        (0xd51b_443f, FpSystemOperation::WriteStatus { rt: 31 }),
    ] {
        // A dirty source register and NZCV precede the PRE-instruction exit.
        let words = [0xb100_0400, word]; // ADDS X0,X0,#1
        let memory = memory(&words);
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.general_register_storage_mut()[0] = 0x1234_003f_ffff;
        actual.set_fpsr(1 << 27);
        let mut expected = actual.clone();
        expected.set_fpsr((1 << 27) | 2); // Real native divide-by-zero below.
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, words[0])
            .unwrap();
        let (reason, exit) = execute_with_fp(
            &memory,
            2,
            &mut actual,
            Compiler::new(native_abi()).unwrap(),
            true,
        );
        assert_eq!(reason, NativeExitReason::Architectural);
        assert_eq!(exit.kind, EdgeKind::FpSystem(operation));
        assert_eq!(exit.pc.get(), PC + 4);
        assert_eq!(
            actual, expected,
            "the system instruction must not execute before FP completion"
        );
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
        crate::lcq::system::complete_fp(operation, &mut actual).unwrap();
        assert_eq!(actual, expected, "{word:08x}");
    }
}

#[test]
fn unsupported_system_operands_exit_at_the_captured_instruction() {
    let mut expected = A64State::default();
    expected.set_pc(PC);
    nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, 0xb100_0400)
        .unwrap();
    let mut actual = A64State::default();
    actual.set_pc(PC);
    let (reason, exit) = execute(&[0xb100_0400, 0xd53b_0000], &mut actual);
    assert_eq!(reason, NativeExitReason::Unsupported);
    assert_eq!(exit.kind, EdgeKind::Unsupported);
    assert_eq!(exit.pc.get(), PC + 4);
    assert_eq!(actual, expected);
}
