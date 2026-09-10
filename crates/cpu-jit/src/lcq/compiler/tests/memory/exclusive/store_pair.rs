use super::*;

fn store_pair(wide: bool, release: bool, rn: u32, rt: u32, rt2: u32, status: u32) -> u32 {
    0x8820_0000
        | (u32::from(wide) << 30)
        | (u32::from(release) << 15)
        | (status << 16)
        | (rt2 << 10)
        | (rn << 5)
        | rt
}

fn load_pair(wide: bool, acquire: bool, rn: u32) -> u32 {
    pair_w(acquire, rn, 0, 8) | (u32::from(wide) << 30)
}

#[test]
fn exclusive_pair_store_matches_interpreter_through_tracking_retry_and_consumption() {
    for wide in [false, true] {
        for release in [false, true] {
            // Source/source and source/base aliases are valid; status aliases
            // are constrained-unpredictable and tested separately below.
            for (rn, rt, rt2, status) in [
                (1, 2, 7, 3),
                (31, 2, 7, 3),
                (1, 31, 7, 3),
                (1, 2, 31, 3),
                (1, 2, 2, 3),
                (1, 1, 7, 3),
                (31, 2, 7, 31),
            ] {
                let words = [
                    0xf100_0529,
                    load_pair(wide, true, rn),
                    store_pair(wide, release, rn, rt, rt2, status),
                    0x9a1f_014a,
                    store_pair(wide, release, 5, 2, 7, 6), // consumed, invalid VA
                    0xd420_0000,
                ];
                let memory = setup_with_value(&words, None);
                let reference = setup_with_value(&words, None);
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = 0x3020;
                *state.stack_pointer_storage_mut() = 0x3020;
                state.general_register_storage_mut()[2] = 0x1234_5678_89ab_cdef;
                state.general_register_storage_mut()[7] = 0xfedc_ba98_7654_3210;
                state.general_register_storage_mut()[3] = u64::MAX;
                state.general_register_storage_mut()[6] = u64::MAX;
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
                    "wide={wide}, release={release}, rn={rn}, rt={rt}, rt2={rt2}, status={status}"
                );
                assert_eq!(monitor, expected_monitor.into_inner());
                assert_eq!(monitor.reservation(), None);
                let size = if wide {
                    MemoryAccessSize::Quadword
                } else {
                    MemoryAccessSize::Doubleword
                };
                let read = |memory: &ExecutionMemory| {
                    memory
                        .read(
                            SPACE,
                            GuestVirtualAddress::new(0x3020),
                            MemoryAccess::normal(size),
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
fn exclusive_pair_store_compares_both_halves_without_partial_replacement() {
    for wide in [false, true] {
        for high in [false, true] {
            let offset = if high { if wide { 8 } else { 4 } } else { 0 };
            // Change just one byte of the selected half after the exclusive load.
            let words = [
                load_pair(wide, false, 1),
                0x3900_0024 | (offset << 10),
                store_pair(wide, true, 1, 2, 7, 3),
                0xd420_0000,
            ];
            let memory = setup_with_value(&words, None);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x3020;
            state.general_register_storage_mut()[2] = u64::MAX;
            state.general_register_storage_mut()[7] = u64::MAX;
            state.general_register_storage_mut()[4] = 42;
            let mut monitor = ExclusiveMonitorState::default();
            let (resolution, physical, cold) =
                execute_exit(&memory, &mut state, &mut monitor, |_| {});
            assert_eq!(resolution, Some(DirectFaultResolution::Retry));
            assert!(physical.is_none() && cold.is_none());
            assert_eq!(state.general_register_storage_mut()[3], 1);
            assert_eq!(monitor.reservation(), None);
            let size = if wide {
                MemoryAccessSize::Quadword
            } else {
                MemoryAccessSize::Doubleword
            };
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
                42u128 << (offset * 8)
            );
        }
    }
}

#[test]
fn exclusive_pair_store_fault_preserves_status_flags_and_all_memory() {
    for wide in [false, true] {
        for release in [false, true] {
            let memory = setup(&[
                load_pair(wide, false, 1),
                0xf100_0529,
                store_pair(wide, release, 1, 2, 7, 3),
                0xd420_0000,
            ]);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x2020;
            state.general_register_storage_mut()[2] = 99;
            state.general_register_storage_mut()[7] = 88;
            state.general_register_storage_mut()[3] = u64::MAX;
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
            assert_eq!(fault.address.get(), 0x2020);
            assert_eq!(fault.kind, nixe_cpu::memory::DataAccessKind::Write);
            assert!(physical.is_none() && cold.is_none());
            assert_eq!(state.general_register_storage_mut()[3], u64::MAX);
            assert_eq!(state.pc(), PC + 8);
            assert_eq!(state.nzcv().bits(), Nzcv::Z | Nzcv::C);
            assert_eq!(monitor.reservation(), None);
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(0x2020),
                        MemoryAccess::normal(MemoryAccessSize::Quadword)
                    )
                    .unwrap()
                    .value
                    .bits(),
                0x8123_4567_89ab_cdef
            );
        }
    }
}

#[test]
fn exclusive_pair_store_physical_exit_keeps_pair_bits_and_revalidates_aliases() {
    for wide in [false, true] {
        for case in 0..4 {
            let mut memory = setup(&[
                load_pair(wide, true, 1),
                store_pair(wide, true, 2, 3, 7, 4),
                0xd420_0000,
            ]);
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = 0x2020;
            state.general_register_storage_mut()[2] = 0x3020 + u64::from(case == 3) * 4;
            state.general_register_storage_mut()[3] = 0x1234_5678_89ab_cdef;
            state.general_register_storage_mut()[7] = 0xfedc_ba98_7654_3210;
            state.general_register_storage_mut()[4] = u64::MAX;
            let mut monitor = ExclusiveMonitorState::default();
            let (resolution, physical, cold) =
                execute_exit(&memory, &mut state, &mut monitor, |_| {});
            assert_eq!(resolution, None);
            assert!(cold.is_none());
            let operation = physical.unwrap();
            let reserved = monitor.reservation().unwrap();
            assert_eq!(reserved.page, GuestPhysicalPageId::new(2));
            assert_eq!(operation.second, Some(7));
            if case == 1 {
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
                        reserved.expected,
                    )
                    .unwrap();
            } else if case == 2 {
                // Completion consumes a monitor left empty by CLREX; no store.
                monitor.clear();
            }
            let result = operation.complete(&mut state, &memory, SPACE, &mut monitor);
            assert_eq!(monitor.reservation(), None);
            if case == 3 {
                assert_eq!(
                    result.unwrap_err().reason,
                    nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                        required_alignment: if wide { 16 } else { 8 }
                    }
                );
                assert_eq!(state.pc(), PC + 4);
                assert_eq!(state.general_register_storage_mut()[4], u64::MAX);
            } else {
                result.unwrap();
                assert_eq!(state.pc(), PC + 8);
                assert_eq!(
                    state.general_register_storage_mut()[4],
                    u64::from(case != 0)
                );
                let expected = if case != 0 {
                    reserved.expected.bits()
                } else if wide {
                    0xfedc_ba98_7654_3210_1234_5678_89ab_cdef
                } else {
                    0x7654_3210_89ab_cdef
                };
                assert_eq!(
                    memory
                        .read(
                            SPACE,
                            GuestVirtualAddress::new(0x3020),
                            MemoryAccess::normal(operation.size)
                        )
                        .unwrap()
                        .value
                        .bits(),
                    expected
                );
            }
        }
    }
}

#[test]
fn exclusive_pair_store_native_shape_fault_maps_and_status_constraints() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for wide in [false, true] {
            let memory = super::super::super::memory(&[
                load_pair(wide, true, 1),
                store_pair(wide, true, 1, 2, 7, 3),
                0xd420_0000,
            ]);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            if abi == HostAbi::X86_64 {
                // Cross-target construction on Arm has baseline x86 features.
                // Exercise the supported CAS128 backend, then reject its
                // missing-feature variant explicitly below.
                let mut target = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
                target.set("has_cmpxchg16b", "true").unwrap();
                compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
            }
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
                assert_eq!(fault.bytes, if wide { 16 } else { 8 });
                assert_eq!(fault.subaccess, 0);
                assert_eq!(fault.commit_stage, 0);
                assert!(fault.completed_read.is_none());
                lowered.states[fault.state_map as usize]
                    .state
                    .validate()
                    .unwrap();
            }
            let clif = compiler.context.func.display().to_string();
            assert_eq!(clif.matches("atomic_cas").count(), 1, "{clif}");
            assert!(!clif.contains("call"), "{clif}");
            for (rn, rt, rt2, status) in [(1, 2, 7, 2), (1, 2, 7, 7), (1, 2, 7, 1), (31, 2, 31, 31)]
            {
                let memory = super::super::super::memory(&[
                    store_pair(wide, false, rn, rt, rt2, status),
                    0xd420_0000,
                ]);
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
            if abi == HostAbi::X86_64 {
                let mut target = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
                target.set("has_cmpxchg16b", "false").unwrap();
                compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
                let result = compiler.lower(&fragment, CodeVersion::new(1).unwrap());
                if wide {
                    assert!(result.err().unwrap().to_string().contains("CMPXCHG16B"));
                } else {
                    assert!(result.is_ok());
                }
            }
        }
    }
}
