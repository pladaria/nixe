use super::*;
use nixe_cpu::memory::{
    CpuMemory, DirectFaultResolution, ExecutionMemory, MemoryAccess, MemoryAccessClass,
    MemoryAccessSize, MemoryAlignment, MemoryOrdering, MemoryValue,
};
use nixe_cpu_direct_memory::{InvocationOutcome, NativeInvocation, WorkerFaultContext};
use nixe_memory::{CanonicalRangeTranslator, DirectBackendPolicy};

mod store_pair;

fn load(size: u32, acquire: bool, rn: u32, rt: u32) -> u32 {
    0x085f_7c00 | (size << 30) | (u32::from(acquire) << 15) | (rn << 5) | rt
}

fn pair_w(acquire: bool, rn: u32, rt: u32, rt2: u32) -> u32 {
    0x887f_0000 | (u32::from(acquire) << 15) | (rt2 << 10) | (rn << 5) | rt
}

fn store(size: u32, release: bool, rn: u32, rt: u32, status: u32) -> u32 {
    0x0800_7c00 | (size << 30) | (u32::from(release) << 15) | (status << 16) | (rn << 5) | rt
}

#[test]
fn native_exclusive_store_matches_interpreter_and_consumes_once() {
    for size in 0..4 {
        for release in [false, true] {
            for (rn, rt, status) in [(1, 2, 3), (31, 2, 3), (1, 31, 3), (31, 2, 31), (1, 1, 3)] {
                let words = [
                    0xf100_0529, // SUBS: keep carry lazy across both exclusives.
                    load(size, true, rn, 0),
                    store(size, release, rn, rt, status),
                    0x9a1f_014a,                   // ADC consumes that carry.
                    store(size, release, 5, 2, 4), // consumed: no access to invalid X5
                    0xd420_0000,
                ];
                // Unmodified backing starts read-only for write tracking. The
                // STXR CAS must fault/retry after consuming its reservation.
                let memory = setup_with_value(&words, None);
                let reference = setup_with_value(&words, None);
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = 0x3020;
                *state.stack_pointer_storage_mut() = 0x3020;
                state.general_register_storage_mut()[2] = 0x1234_5678_9876_5432;
                state.general_register_storage_mut()[3] = u64::MAX;
                state.general_register_storage_mut()[4] = u64::MAX;
                state.general_register_storage_mut()[5] = u64::MAX;
                state.general_register_storage_mut()[9] = 1;
                let mut expected = state.clone();
                let expected_monitor = RefCell::new(ExclusiveMonitorState::default());
                let events = VcpuEventState::default();
                let context = InterpreterContext::new(
                    ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                    &reference,
                    &expected_monitor,
                    &Timer,
                    &events,
                );
                for &word in &words[..5] {
                    assert_eq!(
                        execute_one_with_context(context, &mut expected, word).unwrap(),
                        InstructionStep::Continue
                    );
                }
                let mut monitor = ExclusiveMonitorState::default();
                let (resolution, physical, cold) =
                    execute_exit(&memory, &mut state, &mut monitor, |_| {});
                assert_eq!(resolution, Some(DirectFaultResolution::Retry));
                assert!(physical.is_none() && cold.is_none());
                assert_eq!(
                    state, expected,
                    "size={size}, release={release}, rn={rn}, rt={rt}, status={status}"
                );
                assert_eq!(monitor, expected_monitor.into_inner());
                assert_eq!(monitor.reservation(), None);
                let read = |memory: &ExecutionMemory| {
                    memory
                        .read(
                            SPACE,
                            GuestVirtualAddress::new(0x3020),
                            MemoryAccess::normal(MemoryAccessSize::Doubleword),
                        )
                        .unwrap()
                        .value
                };
                assert_eq!(read(&memory), read(&reference));
            }
        }
    }
}

#[test]
fn native_exclusive_store_value_mismatch_fails_and_a_new_load_replaces_consumed_state() {
    for size in 0..4 {
        for reload in [false, true] {
            let mut words = vec![
                load(size, false, 1, 0),
                0x3900_0026 | (size << 30), // STR[B/H/W/X] X6, [X1]
                store(size, true, 1, 2, 3),
            ];
            if reload {
                words.push(load(size, false, 1, 7));
            }
            words.push(0xd420_0000);
            let memory = setup(&words);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x3020;
            state.general_register_storage_mut()[2] = 99;
            state.general_register_storage_mut()[3] = u64::MAX;
            state.general_register_storage_mut()[6] = 7;
            let mut monitor = ExclusiveMonitorState::default();
            let (_, physical, cold) = execute_exit(&memory, &mut state, &mut monitor, |_| {});
            assert!(physical.is_none() && cold.is_none());
            assert_eq!(state.general_register_storage_mut()[3], 1);
            if reload {
                assert_eq!(monitor.reservation().unwrap().expected.bits(), 7);
                assert_eq!(state.general_register_storage_mut()[7], 7);
            } else {
                assert_eq!(monitor.reservation(), None);
            }
            let size = nixe_cpu::semantics::a64::memory_size(size as u8);
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(0x3020),
                        MemoryAccess::normal(size)
                    )
                    .unwrap()
                    .value
                    .bits(),
                7
            );
        }
    }
}

#[test]
fn native_exclusive_store_permission_fault_consumes_monitor_but_keeps_status_pre() {
    for size in 0..4 {
        for release in [false, true] {
            let memory = setup(&[
                load(size, false, 1, 0),
                0xf100_0529,
                store(size, release, 1, 2, 3),
                0xd420_0000,
            ]);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x2020; // read-only alias
            state.general_register_storage_mut()[2] = 99;
            state.general_register_storage_mut()[3] = 0xfeed_face_dead_beef;
            state.general_register_storage_mut()[9] = 1;
            let mut monitor = ExclusiveMonitorState::default();
            let (resolution, physical, cold) =
                execute_exit(&memory, &mut state, &mut monitor, |_| {});
            let Some(DirectFaultResolution::Fault(fault)) = resolution else {
                panic!("{resolution:?}")
            };
            assert_eq!(
                fault.reason,
                nixe_cpu::memory::DataAccessFaultReason::WritePermissionDenied
            );
            assert_eq!(fault.kind, nixe_cpu::memory::DataAccessKind::Write);
            assert_eq!(fault.address.get(), 0x2020);
            assert!(physical.is_none() && cold.is_none());
            assert_eq!(state.pc(), PC + 8);
            assert_eq!(
                state.general_register_storage_mut()[3],
                0xfeed_face_dead_beef
            );
            assert_eq!(state.nzcv().bits(), Nzcv::Z | Nzcv::C);
            assert_eq!(monitor.reservation(), None);
        }
    }
}

#[test]
fn exclusive_store_followed_by_a_faulting_load_does_not_restore_consumed_monitor() {
    let memory = setup(&[
        load(3, true, 1, 0),
        store(3, true, 1, 2, 3),
        load(3, false, 5, 4),
        0xd420_0000,
    ]);
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[1] = 0x3020;
    state.general_register_storage_mut()[2] = 99;
    state.general_register_storage_mut()[3] = u64::MAX;
    state.general_register_storage_mut()[4] = 77;
    state.general_register_storage_mut()[5] = 0x4000;
    let mut monitor = ExclusiveMonitorState::default();
    let (resolution, physical, cold) = execute_exit(&memory, &mut state, &mut monitor, |_| {});
    assert!(matches!(resolution, Some(DirectFaultResolution::Fault(_))));
    assert!(physical.is_none() && cold.is_none());
    assert_eq!(state.pc(), PC + 8);
    assert_eq!(state.general_register_storage_mut()[3], 0);
    assert_eq!(state.general_register_storage_mut()[4], 77);
    assert_eq!(monitor.reservation(), None);
}

#[test]
fn exclusive_store_physical_exit_rejects_device_without_callbacks_and_consumes_monitor() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut memory = setup(&[load(3, false, 1, 0), store(3, true, 2, 3, 4), 0xd420_0000]);
    let calls = Arc::new(AtomicUsize::new(0));
    assert!(memory.add_mmio_page(
        GuestPhysicalPageId::new(3),
        authority::Device(calls.clone())
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0),
        GuestPhysicalPageId::new(3),
        MemoryPermissions::READ_WRITE
    ));
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[1] = 0x2020;
    state.general_register_storage_mut()[3] = 99;
    state.general_register_storage_mut()[4] = u64::MAX;
    let mut monitor = ExclusiveMonitorState::default();
    let (resolution, physical, cold) = execute_exit(&memory, &mut state, &mut monitor, |_| {});
    assert_eq!(resolution, None);
    assert!(cold.is_none());
    let fault = physical
        .unwrap()
        .complete(&mut state, &memory, SPACE, &mut monitor)
        .unwrap_err();
    assert_eq!(
        fault.reason,
        nixe_cpu::memory::DataAccessFaultReason::MixedRegions
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(state.pc(), PC + 4);
    assert_eq!(state.general_register_storage_mut()[4], u64::MAX);
    assert_eq!(monitor.reservation(), None);
}

#[test]
fn exclusive_store_physical_exit_preserves_alias_identity_after_native_owners_drop() {
    for size in 0..4 {
        for remap in [false, true] {
            let mut memory = setup(&[
                load(size, true, 1, 0),
                store(size, true, 2, 3, 4),
                0xd420_0000,
            ]);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x2020;
            state.general_register_storage_mut()[2] = 0x3020;
            state.general_register_storage_mut()[3] = 99;
            state.general_register_storage_mut()[4] = u64::MAX;
            let mut monitor = ExclusiveMonitorState::default();
            let (resolution, physical, cold) =
                execute_exit(&memory, &mut state, &mut monitor, |_| {});
            assert_eq!(resolution, None);
            assert!(cold.is_none());
            let operation = physical.unwrap();
            assert_eq!(state.pc(), PC + 4);
            assert_eq!(state.general_register_storage_mut()[4], u64::MAX);
            assert_eq!(
                monitor.reservation().unwrap().page,
                GuestPhysicalPageId::new(2)
            );
            if remap {
                memory
                    .resize_zeroed_mapping(
                        SPACE,
                        GuestVirtualAddress::new(0x3000),
                        4096,
                        0,
                        MemoryPermissions::READ_WRITE,
                        nixe_cpu::memory::MemoryMappingPurpose::Normal,
                    )
                    .unwrap();
                assert!(memory.map_page(
                    SPACE,
                    GuestVirtualAddress::new(0x3000),
                    GuestPhysicalPageId::new(1),
                    MemoryPermissions::READ_WRITE
                ));
                memory
                    .write(
                        SPACE,
                        GuestVirtualAddress::new(0x3020),
                        MemoryAccess::normal(operation.size),
                        monitor.reservation().unwrap().expected,
                    )
                    .unwrap();
            }
            operation
                .complete(&mut state, &memory, SPACE, &mut monitor)
                .unwrap();
            assert_eq!(state.pc(), PC + 8);
            assert_eq!(state.general_register_storage_mut()[4], u64::from(remap));
            assert_eq!(monitor.reservation(), None);
            let value = memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(0x3020),
                    MemoryAccess::normal(operation.size),
                )
                .unwrap()
                .value;
            assert_eq!(
                value.bits(),
                if remap {
                    MemoryValue::from_bits(operation.size, 0x8123_4567_89ab_cdef).bits()
                } else {
                    99
                }
            );
        }
    }
}

#[test]
fn exclusive_store_incoming_or_absent_monitor_and_cold_fault_consume_once() {
    for case in 0..4 {
        let memory = setup(&[store(3, true, 1, 2, 3), 0xd420_0000]);
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[1] = match case {
            0 => u64::MAX,
            1 => 0x3020,
            2 => 0x2020,
            _ => 0x3021,
        };
        state.general_register_storage_mut()[2] = 99;
        state.general_register_storage_mut()[3] = u64::MAX;
        let mut monitor = ExclusiveMonitorState::default();
        if case != 0 {
            let (_, reservation) = memory
                .load_exclusive(
                    SPACE,
                    GuestVirtualAddress::new(0x2020),
                    MemoryAccess::new(
                        MemoryAccessSize::Doubleword,
                        MemoryAlignment::Natural,
                        MemoryOrdering::Acquire,
                        MemoryAccessClass::Exclusive,
                    ),
                )
                .unwrap();
            monitor.reserve(reservation);
        }
        let (resolution, physical, cold) = execute_exit(&memory, &mut state, &mut monitor, |_| {});
        assert_eq!(resolution, None);
        assert!(cold.is_none());
        assert_eq!(state.pc(), PC);
        let result = physical
            .unwrap()
            .complete(&mut state, &memory, SPACE, &mut monitor);
        assert_eq!(monitor.reservation(), None);
        if case < 2 {
            result.unwrap();
            assert_eq!(
                state.general_register_storage_mut()[3],
                u64::from(case == 0)
            );
            assert_eq!(state.pc(), PC + 4);
        } else {
            let fault = result.unwrap_err();
            assert_eq!(
                fault.reason,
                if case == 2 {
                    nixe_cpu::memory::DataAccessFaultReason::WritePermissionDenied
                } else {
                    nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                        required_alignment: 8,
                    }
                }
            );
            assert_eq!(state.general_register_storage_mut()[3], u64::MAX);
            assert_eq!(state.pc(), PC);
        }
    }
}

#[test]
fn native_exclusive_store_has_native_cas_pre_fault_maps_and_rejects_status_overlap() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for size in 0..4 {
            let memory = super::super::memory(&[
                load(size, false, 1, 0),
                store(size, true, 1, 2, 3),
                0xd420_0000,
            ]);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            let stores: Vec<_> = lowered
                .faults
                .iter()
                .filter(|fault| fault.access == crate::lifetime::unit::Access::Atomic)
                .collect();
            assert!(matches!(stores.len(), 1 | 3));
            for fault in stores {
                assert_eq!(fault.bytes, 1 << size);
                assert_eq!(fault.commit_stage, 0);
                assert_eq!(fault.subaccess, 0);
                assert!(fault.completed_read.is_none());
                lowered.states[fault.state_map as usize]
                    .state
                    .validate()
                    .unwrap();
            }
            let clif = compiler.context.func.display().to_string();
            assert_eq!(clif.matches("atomic_cas").count(), 1, "{clif}");
            assert!(!clif.contains("call"), "{clif}");
            for (rn, rt, status) in [(1, 2, 2), (1, 2, 1), (31, 31, 31)] {
                let memory =
                    super::super::memory(&[store(size, false, rn, rt, status), 0xd420_0000]);
                let fragment = Fragment::capture(&memory, key()).unwrap();
                assert!(
                    compiler
                        .lower(&fragment, CodeVersion::new(1).unwrap())
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("constrained-unpredictable")
                );
            }
        }
    }
}

#[test]
fn native_exclusive_pair_x_reads_read_only_ram_and_hands_off_all_128_bits() {
    let bits = 0xfedc_ba98_7654_3210_8123_4567_89ab_cdef_u128;
    for acquire in [false, true] {
        for (rn, rt, rt2) in [
            (1, 0, 2),
            (31, 0, 2),
            (1, 1, 2),
            (1, 2, 1),
            (1, 31, 2),
            (1, 2, 31),
        ] {
            let instruction = pair_w(acquire, rn, rt, rt2) | (1 << 30);
            let memory = setup(&[0xf100_0529, instruction, 0x9a1f_014a, 0xd420_0000]);
            let access = MemoryAccess::new(
                MemoryAccessSize::Quadword,
                MemoryAlignment::Natural,
                MemoryOrdering::AcquireRelease,
                MemoryAccessClass::Exclusive,
            );
            memory
                .write(
                    SPACE,
                    GuestVirtualAddress::new(0x3020),
                    MemoryAccess::normal(MemoryAccessSize::Quadword),
                    MemoryValue::from_bits(MemoryAccessSize::Quadword, bits),
                )
                .unwrap();
            let protection = memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x2020));
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x2020;
            state.general_register_storage_mut()[9] = 1;
            *state.stack_pointer_storage_mut() = 0x2020;
            let mut monitor = ExclusiveMonitorState::default();
            assert_eq!(execute(&memory, &mut state, &mut monitor, |_| {}), None);
            if rt != 31 {
                assert_eq!(
                    state.general_register_storage_mut()[rt as usize],
                    bits as u64
                );
            }
            if rt2 != 31 {
                assert_eq!(
                    state.general_register_storage_mut()[rt2 as usize],
                    (bits >> 64) as u64
                );
            }
            assert_eq!(state.nzcv().bits(), Nzcv::Z | Nzcv::C);
            assert_eq!(state.general_register_storage_mut()[10], 1);
            assert_eq!(state.pc(), PC + 12);
            assert_eq!(
                memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x2020)),
                protection
            );
            let reservation = monitor.reservation().unwrap();
            assert_eq!(reservation.expected.bits(), bits);
            assert_eq!(reservation.access_size, 16);
            assert_eq!(reservation.page, GuestPhysicalPageId::new(2));
            assert!(
                memory
                    .store_exclusive(
                        SPACE,
                        GuestVirtualAddress::new(0x3020),
                        access,
                        MemoryValue::from_bits(MemoryAccessSize::Quadword, 123),
                        reservation
                    )
                    .unwrap()
                    .1
            );
        }
    }
}

#[test]
fn native_exclusive_pair_x_alignment_fault_keeps_destinations_and_prior_monitor() {
    for acquire in [false, true] {
        for (rt, rt2) in [(3, 4), (2, 3), (3, 2), (31, 3)] {
            let memory = setup(&[
                load(3, false, 1, 0),
                0x9100_0442,
                pair_w(acquire, 2, rt, rt2) | (1 << 30),
                0xd420_0000,
            ]);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x2020;
            state.general_register_storage_mut()[2] = 0x2027;
            state.general_register_storage_mut()[3] = 55;
            state.general_register_storage_mut()[4] = 66;
            let mut monitor = ExclusiveMonitorState::default();
            let Some(DirectFaultResolution::Fault(fault)) =
                execute(&memory, &mut state, &mut monitor, |_| {})
            else {
                panic!()
            };
            assert_eq!(
                fault.reason,
                nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                    required_alignment: 16
                }
            );
            assert_eq!(fault.address.get(), 0x2028);
            assert_eq!(state.pc(), PC + 8);
            assert_eq!(state.general_register_storage_mut()[2], 0x2028);
            assert_eq!(state.general_register_storage_mut()[3], 55);
            assert_eq!(state.general_register_storage_mut()[4], 66);
            assert_eq!(
                monitor.reservation().unwrap().expected,
                MemoryValue::U64(0x8123_4567_89ab_cdef)
            );
        }
    }
}

#[test]
fn native_exclusive_pair_x_maps_keep_each_read_and_need_no_cmpxchg16b() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let memory = super::super::memory(&[pair_w(true, 1, 1, 2) | (1 << 30), 0xd420_0000]);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
        if abi == HostAbi::X86_64 {
            let mut target = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
            target.set("has_cmpxchg16b", "false").unwrap();
            compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
        }
        let lowered = compiler
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.faults.len(), 2);
        for (index, fault) in lowered.faults.iter().enumerate() {
            assert_eq!(fault.bytes, 8);
            assert_eq!(fault.subaccess, index as u16);
            assert_eq!(fault.commit_stage, 0);
            assert_eq!(fault.completed_read.is_some(), index == 1);
            assert_eq!(fault.access, crate::lifetime::unit::Access::Read);
            lowered.states[fault.state_map as usize]
                .state
                .validate()
                .unwrap();
        }
        let clif = compiler.context.func.display().to_string();
        assert_eq!(clif.matches("atomic_load.i64").count(), 2, "{clif}");
        assert!(
            !clif.contains("call") && !clif.contains("atomic_cas"),
            "{clif}"
        );
        let after = clif.rsplit_once("nixe_fault_end").unwrap().1;
        assert_eq!(after.matches("store ").count(), 4, "{clif}");
    }
}

#[test]
fn native_exclusive_pair_w_matches_interpreter_and_full_width_reservation() {
    for acquire in [false, true] {
        for (rn, rt, rt2) in [
            (1, 0, 2),
            (31, 0, 2),
            (1, 1, 2),
            (1, 2, 1),
            (1, 31, 2),
            (1, 2, 31),
            (31, 30, 0),
        ] {
            let words = [
                0xf100_0529,
                pair_w(acquire, rn, rt, rt2),
                0x9a1f_014a,
                0xd420_0000,
            ];
            let memory = setup(&words);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut().fill(u64::MAX);
            state.general_register_storage_mut()[1] = DATA as u64 + 32;
            state.general_register_storage_mut()[9] = 1;
            *state.stack_pointer_storage_mut() = DATA as u64 + 32;
            let mut expected = state.clone();
            let expected_monitor = RefCell::new(ExclusiveMonitorState::default());
            let events = VcpuEventState::default();
            let context = InterpreterContext::new(
                ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                &memory,
                &expected_monitor,
                &Timer,
                &events,
            );
            for &word in &words[..3] {
                assert_eq!(
                    execute_one_with_context(context, &mut expected, word).unwrap(),
                    InstructionStep::Continue
                );
            }
            let mut monitor = ExclusiveMonitorState::default();
            assert_eq!(execute(&memory, &mut state, &mut monitor, |_| {}), None);
            assert_eq!(
                state, expected,
                "acquire={acquire}, rn={rn}, rt={rt}, rt2={rt2}"
            );
            if rt != 31 {
                assert_eq!(
                    state.general_register_storage_mut()[rt as usize],
                    0x89ab_cdef
                );
            }
            if rt2 != 31 {
                assert_eq!(
                    state.general_register_storage_mut()[rt2 as usize],
                    0x8123_4567
                );
            }
            assert_eq!(monitor, expected_monitor.into_inner());
            let reservation = monitor.reservation().unwrap();
            assert_eq!(
                reservation.expected,
                MemoryValue::U64(0x8123_4567_89ab_cdef)
            );
            assert_eq!(reservation.access_size, 8);
            assert!(
                memory
                    .store_exclusive(
                        SPACE,
                        GuestVirtualAddress::new(0x3020),
                        MemoryAccess::new(
                            MemoryAccessSize::Doubleword,
                            MemoryAlignment::Natural,
                            MemoryOrdering::Release,
                            MemoryAccessClass::Exclusive
                        ),
                        MemoryValue::U64(123),
                        reservation
                    )
                    .unwrap()
                    .1
            );
        }
    }
}

#[test]
fn native_exclusive_pair_w_fault_preserves_both_dirty_destinations_and_reservation() {
    for acquire in [false, true] {
        for address in [0x2004, 0x4000, u64::MAX] {
            for (rt, rt2) in [(3, 4), (2, 3), (3, 2), (31, 3), (3, 31)] {
                let words = [
                    load(3, false, 1, 0),
                    0xf100_0529,
                    0x9100_0442,
                    0x9100_0463,
                    0x9100_0484,
                    pair_w(acquire, 2, rt, rt2),
                    0xd420_0000,
                ];
                let memory = setup(&words);
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = DATA as u64 + 32;
                state.general_register_storage_mut()[2] = address - 1;
                state.general_register_storage_mut()[3] = u64::MAX - 1;
                state.general_register_storage_mut()[4] = 88;
                state.general_register_storage_mut()[9] = 1;
                let mut monitor = ExclusiveMonitorState::default();
                let Some(DirectFaultResolution::Fault(fault)) =
                    execute(&memory, &mut state, &mut monitor, |_| {})
                else {
                    panic!()
                };
                assert_eq!(fault.address.get(), address);
                if address == 0x2004 {
                    assert_eq!(
                        fault.reason,
                        nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                            required_alignment: 8
                        }
                    );
                }
                assert_eq!(state.pc(), PC + 20);
                assert_eq!(state.general_register_storage_mut()[2], address);
                assert_eq!(state.general_register_storage_mut()[3], u64::MAX);
                assert_eq!(state.general_register_storage_mut()[4], 89);
                assert_eq!(state.nzcv().bits(), Nzcv::Z | Nzcv::C);
                assert_eq!(
                    monitor.reservation().unwrap().expected,
                    MemoryValue::U64(0x8123_4567_89ab_cdef)
                );
            }
        }
    }
}

#[test]
fn native_exclusive_pair_w_maps_describe_one_atomic_read_and_reject_overlap() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for acquire in [false, true] {
            let memory = super::super::memory(&[pair_w(acquire, 1, 0, 2), 0xd420_0000]);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(lowered.faults.len(), 1);
            let fault = &lowered.faults[0];
            assert_eq!(fault.bytes, 8);
            assert_eq!(fault.access, crate::lifetime::unit::Access::Read);
            assert_eq!((fault.subaccess, fault.commit_stage), (0, 0));
            assert!(fault.completed_read.is_none());
            let clif = compiler.context.func.display().to_string();
            assert_eq!(clif.matches("atomic_load.i64").count(), 1, "{clif}");
            assert!(
                !clif.contains("call") && !clif.contains("atomic_cas"),
                "{clif}"
            );
            for register in [0, 31] {
                let memory =
                    super::super::memory(&[pair_w(acquire, 1, register, register), 0xd420_0000]);
                let fragment = Fragment::capture(&memory, key()).unwrap();
                let error = compiler
                    .lower(&fragment, CodeVersion::new(2).unwrap())
                    .err()
                    .unwrap();
                assert!(error.to_string().contains("overlapping destinations"));
            }
        }
    }
}

fn setup(words: &[u32]) -> ExecutionMemory {
    setup_with_value(words, Some(0x8123_4567_89ab_cdef))
}

fn setup_with_value(words: &[u32], initial: Option<u64>) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    for id in 1..=2 {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(id)));
    }
    let code: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    memory
        .initialize_ram(GuestPhysicalPageId::new(1), 0, &code)
        .unwrap();
    if let Some(initial) = initial {
        memory
            .initialize_ram(GuestPhysicalPageId::new(2), 32, &initial.to_le_bytes())
            .unwrap();
    }
    for (address, page, permissions) in [
        (PC, 1, MemoryPermissions::READ_EXECUTE),
        (DATA as u64, 2, MemoryPermissions::READ),
        (0x3000, 2, MemoryPermissions::READ_WRITE),
    ] {
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(address),
            GuestPhysicalPageId::new(page),
            permissions
        ));
    }
    memory
        .bind_cpu_memory_backend(SPACE, ARENA as u64, DirectBackendPolicy::Required)
        .unwrap();
    memory
}

// Exercise the real gateway, fault authority, PRE reconstruction and exit
// handoff under one mapping lease; no interpreter participates in this path.
fn execute(
    memory: &ExecutionMemory,
    state: &mut A64State,
    monitor: &mut ExclusiveMonitorState,
    before_handoff: impl FnOnce(&ExecutionMemory),
) -> Option<DirectFaultResolution> {
    execute_exit(memory, state, monitor, before_handoff).0
}

fn execute_exit(
    memory: &ExecutionMemory,
    state: &mut A64State,
    monitor: &mut ExclusiveMonitorState,
    before_handoff: impl FnOnce(&ExecutionMemory),
) -> (
    Option<DirectFaultResolution>,
    Option<crate::abi::ExclusiveStoreOperation>,
    Option<crate::lcq::fault::cold::Completion>,
) {
    let demanded = key().at(GuestVirtualAddress::new(state.pc())).unwrap();
    let arena = memory.direct_address_space_view(SPACE).unwrap();
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let Request::Owner(claim) = reader.claim(demanded).unwrap() else {
        panic!()
    };
    let handle = Compiler::for_arena(native_abi(), ARENA)
        .unwrap()
        .publish(
            Compilation::capture(claim, memory).unwrap(),
            &process,
            &cache,
            memory,
        )
        .unwrap();
    let snapshot = process.snapshot(handle).unwrap();
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, 1000).unwrap());
    let _lease = memory.acquire_execution_lease();
    let mut invocation = unsafe { reader.admit(&mut frame, demanded) }
        .unwrap()
        .unwrap();
    let entry = invocation.payload().preferred().unwrap().canonical.get();
    let (frame, lookup) = invocation.frame_and_faults();
    let mut dispatcher = authority::Dispatch {
        frame: std::ptr::from_ref(frame).cast(),
        lookup,
        memory,
        resolution: None,
        count: 0,
    };
    let mut call = CapturedEntry {
        frame,
        arena: arena.base as *mut u8,
        result: None,
    };
    let mut worker = WorkerFaultContext::register().unwrap();
    let outcome = unsafe {
        worker.invoke_captured(
            arena,
            [
                call.frame.host_fp.saved_control,
                call.frame.host_fp.saved_status,
            ],
            authority::dispatch,
            std::ptr::from_mut(&mut dispatcher).cast(),
            NativeInvocation {
                gateway: captured_entry,
                context: std::ptr::from_mut(&mut call).cast(),
                entry,
            },
        )
    }
    .unwrap();
    let mut physical = None;
    let mut cold = None;
    if outcome == InvocationOutcome::Escaped {
        let captured = worker.escaped_fault().unwrap();
        let fault = dispatcher.lookup.find(captured.native_pc()).unwrap();
        let reconstructed =
            unsafe { crate::lcq::fault::reconstruct(call.frame, &captured, &fault) }.unwrap();
        if dispatcher.resolution == Some(DirectFaultResolution::Cold) {
            cold = Some(
                unsafe {
                    crate::lcq::fault::cold::Completion::prepare(
                        call.frame,
                        &fault,
                        reconstructed.completed_read,
                    )
                }
                .unwrap(),
            );
        }
    } else {
        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(
            call.result.unwrap().unwrap().reason,
            NativeExitReason::Architectural
        );
        if let EdgeKind::ExclusiveStore(operation) = snapshot.states
            [call.frame.exit_state_map as usize]
            .exit
            .unwrap()
            .kind
        {
            physical = Some(operation);
        }
    }
    before_handoff(memory);
    call.frame
        .finish_exclusive_load(memory, SPACE, monitor)
        .unwrap();
    let finished = *monitor;
    call.frame
        .finish_exclusive_load(memory, SPACE, monitor)
        .unwrap();
    assert_eq!(*monitor, finished, "handoff is consumed once");
    (dispatcher.resolution, physical, cold)
}

#[test]
fn native_exclusive_load_matches_interpreter_and_reserves_discarded_results() {
    for size in 0..4 {
        for acquire in [false, true] {
            for (rn, rt) in [(1, 0), (31, 0), (1, 1), (1, 31)] {
                let words = [
                    0xf100_0529,
                    load(size, acquire, rn, rt),
                    0x9a1f_014a,
                    0xd420_0000,
                ];
                let memory = setup(&words);
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = DATA as u64 + 32;
                state.general_register_storage_mut()[9] = 1;
                *state.stack_pointer_storage_mut() = DATA as u64 + 32;
                let mut expected = state.clone();
                let expected_monitor = RefCell::new(ExclusiveMonitorState::default());
                let events = VcpuEventState::default();
                let context = InterpreterContext::new(
                    ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                    &memory,
                    &expected_monitor,
                    &Timer,
                    &events,
                );
                for &word in &words[..3] {
                    assert_eq!(
                        execute_one_with_context(context, &mut expected, word).unwrap(),
                        InstructionStep::Continue
                    );
                }
                let mut monitor = ExclusiveMonitorState::default();
                assert_eq!(execute(&memory, &mut state, &mut monitor, |_| {}), None);
                assert_eq!(
                    state, expected,
                    "size={size}, acquire={acquire}, rn={rn}, rt={rt}"
                );
                assert_eq!(monitor, expected_monitor.into_inner());
                assert_eq!(
                    monitor.reservation().unwrap().page,
                    GuestPhysicalPageId::new(2)
                );
                assert_eq!(monitor.reservation().unwrap().byte_offset, 32);
                let reservation = monitor.reservation().unwrap();
                assert!(
                    memory
                        .store_exclusive(
                            SPACE,
                            GuestVirtualAddress::new(0x3020),
                            MemoryAccess::new(
                                reservation.expected.size(),
                                MemoryAlignment::Natural,
                                MemoryOrdering::Release,
                                MemoryAccessClass::Exclusive
                            ),
                            MemoryValue::from_bits(reservation.expected.size(), 17),
                            reservation
                        )
                        .unwrap()
                        .1
                );
            }
        }
    }
}

#[test]
fn native_exclusive_load_fault_preserves_last_reservation_and_observed_value() {
    // The second load faults after a successful first load. The alias write
    // after escape must not replace the bits recorded by the first load.
    let words = [
        load(3, false, 1, 0),
        0xf100_0529,
        0x9100_0442,
        load(3, true, 2, 3),
        0xd420_0000,
    ];
    for address in [0x2001, 0x4000, u64::MAX] {
        let memory = setup(&words);
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[1] = DATA as u64 + 32;
        state.general_register_storage_mut()[2] = address - 1;
        state.general_register_storage_mut()[3] = 77;
        state.general_register_storage_mut()[9] = 1;
        let mut monitor = ExclusiveMonitorState::default();
        let result = execute(&memory, &mut state, &mut monitor, |memory| {
            memory
                .write(
                    SPACE,
                    GuestVirtualAddress::new(0x3020),
                    MemoryAccess::normal(MemoryAccessSize::Doubleword),
                    MemoryValue::U64(99),
                )
                .unwrap();
        });
        assert!(matches!(result, Some(DirectFaultResolution::Fault(_))));
        assert_eq!(state.pc(), PC + 12);
        assert_eq!(state.general_register_storage_mut()[3], 77);
        assert_eq!(state.nzcv().bits(), Nzcv::Z | Nzcv::C);
        let reservation = monitor.reservation().unwrap();
        assert_eq!(
            reservation.expected,
            MemoryValue::U64(0x8123_4567_89ab_cdef)
        );
        assert!(
            !memory
                .store_exclusive(
                    SPACE,
                    GuestVirtualAddress::new(0x3020),
                    MemoryAccess::new(
                        MemoryAccessSize::Doubleword,
                        MemoryAlignment::Natural,
                        MemoryOrdering::Release,
                        MemoryAccessClass::Exclusive
                    ),
                    MemoryValue::U64(88),
                    reservation
                )
                .unwrap()
                .1
        );
        // A later fragment's first load faults: it must leave the already
        // resolved per-thread reservation intact as well.
        let old = monitor;
        execute(&memory, &mut state, &mut monitor, |_| {});
        assert_eq!(monitor, old);
    }
}

#[test]
fn native_exclusive_load_last_success_survives_exit_and_clrex_clears_it() {
    use crate::abi::RuntimeSystemOperation;
    use crate::lcq::system::{RuntimeServices, complete_runtime};
    let memory = setup(&[
        load(3, false, 1, 0),
        load(3, true, 2, 3),
        0xd503_3f5f,
        0xd420_0000,
    ]);
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x3028),
            MemoryAccess::normal(MemoryAccessSize::Doubleword),
            MemoryValue::U64(123),
        )
        .unwrap();
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[1] = DATA as u64 + 32;
    state.general_register_storage_mut()[2] = 0x3028;
    let mut monitor = ExclusiveMonitorState::default();
    execute(&memory, &mut state, &mut monitor, |_| {});
    assert_eq!(state.pc(), PC + 8);
    let reservation = monitor.reservation().unwrap();
    assert_eq!(reservation.byte_offset, 40);
    assert_eq!(reservation.expected, MemoryValue::U64(123));
    let events = VcpuEventState::default();
    complete_runtime(
        RuntimeSystemOperation::ClearExclusive,
        &mut state,
        &mut RuntimeServices {
            address_space: SPACE,
            memory: &memory,
            timer: &Timer,
            events: &events,
            exclusive: &mut monitor,
        },
    )
    .unwrap();
    assert!(monitor.reservation().is_none());
    execute(&memory, &mut state, &mut monitor, |_| {});
    assert!(
        monitor.reservation().is_none(),
        "a new frame must not resurrect the load before CLREX"
    );
}

#[test]
fn native_exclusive_load_maps_are_precise_and_have_no_helper() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for size in 0..4 {
            let memory = super::super::memory(&[load(size, true, 1, 31), 0xd420_0000]);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(lowered.faults.len(), 1);
            assert_eq!(lowered.faults[0].bytes, 1 << size);
            assert_eq!(
                lowered.faults[0].access,
                crate::lifetime::unit::Access::Read
            );
            let clif = compiler.context.func.display().to_string();
            assert_eq!(clif.matches("atomic_load").count(), 1);
            assert!(!clif.contains("call"), "{clif}");
            // Ignore the compiler's unreachable allocation-root block; only
            // the successful side of this memory boundary updates the frame.
            let after = clif.split_once("nixe_fault_end").unwrap().1;
            assert_eq!(after.matches("get_pinned_reg").count(), 1, "{clif}");
            assert_eq!(after.matches("store ").count(), 3, "{clif}");
        }
    }
}

#[test]
fn native_exclusive_load_retries_gpu_visibility_before_recording_the_value() {
    for variant in 0..3 {
        let pair = variant != 0;
        let wide = variant == 2;
        let instruction = if wide {
            pair_w(true, 1, 0, 2) | (1 << 30)
        } else if pair {
            pair_w(true, 1, 0, 2)
        } else {
            load(3, true, 1, 0)
        };
        let memory = setup(&[instruction, 0xd420_0000]);
        let range = memory
            .translate_canonical_range(
                SPACE,
                GuestVirtualAddress::new(0x3000),
                4096,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let declaration = nixe_memory::DeviceAccessDeclaration::write(
            nixe_memory::NonCpuDeviceId::new(1),
            nixe_memory::DeviceVisibilityPoint::new(1),
            nixe_memory::DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        let coordinator: Arc<dyn nixe_memory::VisibilityCoordinator> =
            Arc::new(authority::Writeback);
        range
            .prepare_device_access(declaration, coordinator.clone())
            .unwrap();
        range
            .publish_device_write(declaration, coordinator)
            .unwrap();
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[1] = DATA as u64 + 32;
        let mut monitor = ExclusiveMonitorState::default();
        assert_eq!(
            execute(&memory, &mut state, &mut monitor, |_| {}),
            Some(DirectFaultResolution::Retry)
        );
        assert_eq!(
            state.general_register_storage_mut()[0],
            if pair && !wide {
                0x5a5a_5a5a
            } else {
                0x5a5a_5a5a_5a5a_5a5a
            }
        );
        if pair {
            assert_eq!(
                state.general_register_storage_mut()[2],
                if wide {
                    0x5a5a_5a5a_5a5a_5a5a
                } else {
                    0x5a5a_5a5a
                }
            );
        }
        assert_eq!(
            monitor.reservation().unwrap().expected,
            MemoryValue::from_bits(
                if wide {
                    MemoryAccessSize::Quadword
                } else {
                    MemoryAccessSize::Doubleword
                },
                0x5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a
            )
        );
    }
}

#[test]
fn cold_exclusive_load_revalidates_memory_and_commits_monitor_only_on_success() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for size in 0..6 {
        let pair = size >= 4;
        let wide = size == 5;
        for acquire in [false, true] {
            for remap in [false, true] {
                let instruction = if wide {
                    pair_w(acquire, 1, 0, 2) | (1 << 30)
                } else if pair {
                    pair_w(acquire, 1, 0, 2)
                } else {
                    load(size, acquire, 1, 0)
                };
                let mut memory = setup(&[instruction, 0xd420_0000]);
                let calls = Arc::new(AtomicUsize::new(0));
                let device = GuestPhysicalPageId::new(3);
                assert!(memory.add_mmio_page(device, authority::Device(calls.clone())));
                assert!(memory.map_page(
                    SPACE,
                    GuestVirtualAddress::new(0),
                    device,
                    MemoryPermissions::READ_WRITE
                ));
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[0] = u64::MAX;
                state.general_register_storage_mut()[2] = u64::MAX;
                state.general_register_storage_mut()[1] = 32;
                let access_size = if wide {
                    MemoryAccessSize::Quadword
                } else if pair {
                    MemoryAccessSize::Doubleword
                } else {
                    nixe_cpu::semantics::a64::memory_size(size as u8)
                };
                let access = MemoryAccess::new(
                    access_size,
                    MemoryAlignment::Natural,
                    if acquire {
                        MemoryOrdering::Acquire
                    } else {
                        MemoryOrdering::Relaxed
                    },
                    MemoryAccessClass::Exclusive,
                );
                let (_, old) = memory
                    .load_exclusive(SPACE, GuestVirtualAddress::new(0x2020), access)
                    .unwrap();
                let mut monitor = ExclusiveMonitorState::default();
                monitor.reserve(old);
                let completion = authority::prepare_cold(&memory, &mut state);
                let before = state.clone();
                if remap {
                    memory
                        .resize_zeroed_mapping(
                            SPACE,
                            GuestVirtualAddress::new(0),
                            4096,
                            0,
                            MemoryPermissions::READ_WRITE,
                            nixe_cpu::memory::MemoryMappingPurpose::Normal,
                        )
                        .unwrap();
                    assert!(memory.map_page(
                        SPACE,
                        GuestVirtualAddress::new(0),
                        GuestPhysicalPageId::new(2),
                        MemoryPermissions::READ
                    ));
                    memory
                        .write(
                            SPACE,
                            GuestVirtualAddress::new(0x3020),
                            MemoryAccess::normal(access_size),
                            MemoryValue::from_bits(
                                access_size,
                                0x8123_4567_ffff_0000_aabb_ccdd_eeff_9182,
                            ),
                        )
                        .unwrap();
                }
                let result = completion.complete(&mut state, &memory, &mut monitor);
                if remap {
                    result.unwrap();
                    let (read, expected) = memory
                        .load_exclusive(SPACE, GuestVirtualAddress::new(32), access)
                        .unwrap();
                    assert_eq!(monitor.reservation(), Some(expected));
                    assert_eq!(
                        state.general_register_storage_mut()[0],
                        if pair && !wide {
                            u64::from(read.value.bits() as u32)
                        } else {
                            read.value.bits() as u64
                        }
                    );
                    if pair {
                        assert_eq!(
                            state.general_register_storage_mut()[2],
                            (read.value.bits() >> if wide { 64 } else { 32 }) as u64
                        );
                    }
                    assert_eq!(state.pc(), PC + 4);
                } else {
                    let Err(crate::lcq::fault::cold::Error::Data(fault)) = result else {
                        panic!()
                    };
                    assert_eq!(
                        fault.reason,
                        nixe_cpu::memory::DataAccessFaultReason::MixedRegions
                    );
                    assert_eq!(monitor.reservation(), Some(old));
                    assert_eq!(state, before);
                }
                assert_eq!(calls.load(Ordering::Relaxed), 0);
            }
        }
    }
}
