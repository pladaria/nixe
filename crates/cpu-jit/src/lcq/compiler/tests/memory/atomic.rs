use super::*;
use crate::lifetime::unit::Access;

pub(super) fn rmw(size: u32, opcode: u32, ordering: u32, rn: u32, rs: u32, rt: u32) -> u32 {
    0x3820_0000 | (size << 30) | (ordering << 22) | (rs << 16) | (opcode << 12) | (rn << 5) | rt
}

#[test]
fn scalar_rmw_matches_all_operations_widths_aliases_and_lazy_flags() {
    for size in 0..4 {
        for opcode in 0..9 {
            for ordering in 0..4 {
                for (rn, rs, rt) in [
                    (1, 2, 3),
                    (31, 2, 3),
                    (1, 31, 3),
                    (1, 2, 31),
                    (1, 1, 3),
                    (1, 2, 2),
                    (1, 2, 1),
                ] {
                    let word = rmw(size, opcode, ordering, rn, rs, rt);
                    let words = [0xf100_04a5, word, 0x9a1f_00c6, 0xd420_0000];
                    // Both signs, ties and overflow; high source bits deliberately
                    // differ from the access width to expose narrow min/max bugs.
                    let initial = [0x8080_8080_8080_8080u64, 0x7f7f_7f7f_7f7f_7f7f, 0, u64::MAX]
                        [ordering as usize];
                    let operand =
                        [0x7171_7171_7171_7171u64, 0x9292_9292_9292_9292, 0, 1][ordering as usize];
                    let mut memory = super::super::memory(&words);
                    let page = GuestPhysicalPageId::new(2);
                    assert!(memory.add_ram_page(page));
                    assert!(memory.initialize_ram(page, 0, &initial.to_le_bytes()));
                    assert!(memory.map_page(
                        SPACE,
                        GuestVirtualAddress::new(DATA as u64),
                        page,
                        MemoryPermissions::READ_WRITE
                    ));
                    let mut arena = vec![0; ARENA];
                    arena[DATA..DATA + 8].copy_from_slice(&initial.to_le_bytes());
                    let mut state = A64State::default();
                    state.set_pc(PC);
                    state.general_register_storage_mut()[1] = DATA as u64;
                    state.general_register_storage_mut()[2] = operand;
                    state.general_register_storage_mut()[3] = u64::MAX;
                    state.general_register_storage_mut()[5] = 1;
                    state.general_register_storage_mut()[6] = 19;
                    *state.stack_pointer_storage_mut() = DATA as u64;
                    let mut expected = state.clone();
                    let monitor = RefCell::new(ExclusiveMonitorState::default());
                    let events = VcpuEventState::default();
                    for word in &words[..3] {
                        assert_eq!(
                            execute_one_with_context(
                                InterpreterContext::new(
                                    ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                                    &memory,
                                    &monitor,
                                    &Timer,
                                    &events
                                ),
                                &mut expected,
                                *word,
                            )
                            .unwrap(),
                            InstructionStep::Continue
                        );
                    }
                    run(&words, &mut state, &mut arena);
                    assert_eq!(state, expected, "{word:08x}");
                    let mut expected_bytes = [0; 8];
                    memory
                        .read_bytes(
                            SPACE,
                            GuestVirtualAddress::new(DATA as u64),
                            &mut expected_bytes,
                        )
                        .unwrap();
                    assert_eq!(arena[DATA..DATA + 8], expected_bytes, "{word:08x}");
                }
            }
        }
    }
}

#[test]
fn scalar_rmw_maps_cover_native_and_loop_accesses_without_committing() {
    for (abi, lse) in [
        (HostAbi::X86_64, false),
        (HostAbi::Aarch64, false),
        (HostAbi::Aarch64, true),
    ] {
        for size in 0..4 {
            for opcode in 0..9 {
                let memory = super::super::memory(&[
                    0xf100_04a5,
                    rmw(size, opcode, 3, 1, 2, 3),
                    0xd420_0000,
                ]);
                let fragment = Fragment::capture(&memory, key()).unwrap();
                let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
                if abi == HostAbi::Aarch64 {
                    let mut target =
                        isa::lookup("aarch64-unknown-linux-gnu".parse().unwrap()).unwrap();
                    target.set("use_bti", "true").unwrap();
                    target
                        .set("has_lse", if lse { "true" } else { "false" })
                        .unwrap();
                    compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
                }
                let lowered = compiler
                    .lower(&fragment, CodeVersion::new(1).unwrap())
                    .unwrap();
                let single = if abi == HostAbi::Aarch64 {
                    lse
                } else {
                    matches!(opcode, 0 | 8)
                };
                assert_eq!(
                    lowered.faults.len(),
                    if single { 1 } else { 2 },
                    "{abi:?} {size} {opcode}"
                );
                for fault in &lowered.faults {
                    assert_eq!(fault.access, Access::Atomic);
                    assert_eq!(fault.bytes, 1 << size);
                    assert_eq!(fault.subaccess, 0);
                    assert_eq!(fault.commit_stage, 0);
                    assert!(fault.completed_read.is_none());
                    let state = &lowered.states[fault.state_map as usize].state;
                    assert!(!state.dirty_live.integer.x[3]);
                    assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
                    state.validate().unwrap();
                    if abi == HostAbi::Aarch64 {
                        assert_eq!(fault.native_end - fault.native_start, 4);
                    }
                }
                let clif = compiler.context.func.display().to_string();
                assert_eq!(clif.matches("atomic_rmw").count(), 1);
                assert_eq!(clif.matches("nixe_fault_start").count(), 1);
                assert!(!clif.contains("call"));
                assert!(!clif.contains("store"));
            }
        }
    }
}

fn cas(size: u32, acquire: bool, release: bool, rn: u32, rs: u32, rt: u32) -> u32 {
    0x08a0_7c00
        | (size << 30)
        | (u32::from(acquire) << 22)
        | (u32::from(release) << 15)
        | (rs << 16)
        | (rn << 5)
        | rt
}

#[test]
fn delivered_scalar_rmw_retry_escape_and_alignment_preserve_pre_state() {
    for size in 0..4 {
        for opcode in 0..9 {
            for (escape, misaligned) in [(false, false), (true, false), (true, true)] {
                if misaligned && size == 0 {
                    continue;
                }
                // Dirty source/destination and lazy NZCV before the atomic.
                let words = [
                    0xf100_04a5,
                    0x9100_0442,
                    rmw(size, opcode, 3, 1, 2, 2),
                    0xd420_0000,
                ];
                let mut arena = vec![0; ARENA];
                arena[DATA..DATA + 8].copy_from_slice(&1u64.to_le_bytes());
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = DATA as u64 + u64::from(misaligned);
                state.general_register_storage_mut()[5] = 1;
                let mut expected = state.clone();
                expected.set_pc(PC + 8);
                expected.general_register_storage_mut()[2] = 1;
                expected.general_register_storage_mut()[5] = 0;
                expected.set_nzcv(Nzcv::from_bits(Nzcv::Z | Nzcv::C));
                let reconstructed = run_memory_case(
                    &words,
                    &mut state,
                    &mut arena,
                    Some((if misaligned { ARENA } else { DATA }, 0, 0)),
                    escape,
                    false,
                );
                if escape {
                    let (reconstructed, access) = reconstructed.unwrap();
                    assert!(reconstructed.completed_read.is_none());
                    if misaligned {
                        assert_eq!(
                            access.guest_fault(SPACE).unwrap().reason,
                            nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                                required_alignment: 1 << size
                            }
                        );
                    }
                    assert_eq!(arena[DATA..DATA + 8], 1u64.to_le_bytes());
                } else {
                    expected.set_pc(PC + 12);
                    let new: u64 = match opcode {
                        0 => 2,
                        1 | 2 => 0,
                        _ => 1,
                    };
                    assert_eq!(arena[DATA..DATA + 8], new.to_le_bytes());
                }
                assert_eq!(
                    state, expected,
                    "size={size} opcode={opcode} escape={escape} aligned={}",
                    !misaligned
                );
            }
        }
    }
}

#[test]
fn scalar_cas_matches_success_failure_aliases_and_lazy_flags() {
    for size in 0..4 {
        for acquire in [false, true] {
            for release in [false, true] {
                for (rn, rs, rt) in [
                    (1, 2, 3),
                    (31, 2, 3),
                    (1, 31, 3),
                    (1, 2, 31),
                    (1, 1, 3),
                    (1, 2, 2),
                    (1, 2, 1),
                ] {
                    for success in [false, true] {
                        let word = cas(size, acquire, release, rn, rs, rt);
                        let words = [0xf100_04a5, word, 0x9a1f_00c6, 0xd420_0000];
                        let mut memory = super::super::memory(&words);
                        let page = GuestPhysicalPageId::new(2);
                        assert!(memory.add_ram_page(page));
                        assert!(memory.initialize_ram(page, 0, &[0x92; 4096]));
                        assert!(memory.map_page(
                            SPACE,
                            GuestVirtualAddress::new(DATA as u64),
                            page,
                            MemoryPermissions::READ_WRITE
                        ));
                        let mut arena = vec![0; ARENA];
                        arena[DATA..DATA + 4096].fill(0x92);
                        let mut state = A64State::default();
                        state.set_pc(PC);
                        state.general_register_storage_mut()[1] = DATA as u64;
                        state.general_register_storage_mut()[2] =
                            if success { 0x9292_9292_9292_9292 } else { 0 };
                        state.general_register_storage_mut()[3] = 0x0123_4567_89ab_cdef;
                        state.general_register_storage_mut()[5] = 1;
                        state.general_register_storage_mut()[6] = 19;
                        *state.stack_pointer_storage_mut() = DATA as u64;
                        let mut expected = state.clone();
                        let monitor = RefCell::new(ExclusiveMonitorState::default());
                        let events = VcpuEventState::default();
                        for word in &words[..3] {
                            assert_eq!(
                                execute_one_with_context(
                                    InterpreterContext::new(
                                        ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                                        &memory,
                                        &monitor,
                                        &Timer,
                                        &events,
                                    ),
                                    &mut expected,
                                    *word
                                )
                                .unwrap(),
                                InstructionStep::Continue
                            );
                        }
                        run(&words, &mut state, &mut arena);
                        assert_eq!(state, expected, "{word:08x} success={success}");
                        let mut expected_bytes = [0; 8];
                        memory
                            .read_bytes(
                                SPACE,
                                GuestVirtualAddress::new(DATA as u64),
                                &mut expected_bytes,
                            )
                            .unwrap();
                        assert_eq!(arena[DATA..DATA + 8], expected_bytes);
                    }
                }
            }
        }
    }
}

#[test]
fn scalar_cas_maps_cover_lse_and_exclusive_loop_without_committing() {
    for (abi, lse) in [
        (HostAbi::X86_64, false),
        (HostAbi::Aarch64, false),
        (HostAbi::Aarch64, true),
    ] {
        for size in 0..4 {
            let memory =
                super::super::memory(&[0xf100_04a5, cas(size, true, true, 1, 2, 3), 0xd420_0000]);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            if abi == HostAbi::Aarch64 {
                let mut target = isa::lookup("aarch64-unknown-linux-gnu".parse().unwrap()).unwrap();
                target.set("use_bti", "true").unwrap();
                target
                    .set("has_lse", if lse { "true" } else { "false" })
                    .unwrap();
                compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
            }
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(
                lowered.faults.len(),
                if abi == HostAbi::Aarch64 && !lse {
                    3
                } else {
                    1
                }
            );
            for fault in &lowered.faults {
                assert_eq!(fault.access, Access::Atomic);
                assert_eq!(fault.bytes, 1 << size);
                assert_eq!(fault.subaccess, 0);
                assert_eq!(fault.commit_stage, 0);
                assert!(fault.completed_read.is_none());
                let state = &lowered.states[fault.state_map as usize].state;
                // CAS must retain the incoming compare value before commit.
                assert!(state.dirty_live.integer.x[2]);
                assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
                state.validate().unwrap();
                if abi == HostAbi::Aarch64 {
                    assert_eq!(fault.native_end - fault.native_start, 4);
                }
            }
            let clif = compiler.context.func.display().to_string();
            assert_eq!(clif.matches("atomic_cas").count(), 1);
            assert_eq!(clif.matches("nixe_fault_start").count(), 1);
            assert!(!clif.contains("call"));
            assert!(!clif.contains("store"));
        }
    }
}

#[test]
fn delivered_scalar_cas_alignment_fault_keeps_pre_state() {
    for size in 1..4 {
        let words = [cas(size, true, true, 1, 2, 3), 0xd420_0000];
        let mut arena = vec![0x92; ARENA];
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[1] = DATA as u64 + 1;
        state.general_register_storage_mut()[2] = 0x9292_9292_9292_9292;
        state.general_register_storage_mut()[3] = 19;
        let expected = state.clone();
        let (_, access) = run_memory_case(
            &words,
            &mut state,
            &mut arena,
            Some((ARENA, 0, 0)),
            true,
            false,
        )
        .unwrap();
        let fault = access.guest_fault(SPACE).unwrap();
        assert_eq!(fault.kind, nixe_cpu::memory::DataAccessKind::Read);
        assert_eq!(
            fault.reason,
            nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                required_alignment: 1 << size
            }
        );
        assert_eq!(state, expected);
        assert_eq!(arena[DATA..DATA + 16], [0x92; 16]);
    }
}

#[test]
fn delivered_scalar_cas_retry_and_escape_preserve_pre_state() {
    for size in 0..4 {
        for escape in [false, true] {
            // Dirty both NZCV and the compare/result register before the fault.
            let words = [
                0xf100_04a5,
                0x9100_0442,
                cas(size, true, true, 1, 2, 3),
                0xd420_0000,
            ];
            let mut arena = vec![0; ARENA];
            arena[DATA..DATA + 8].copy_from_slice(&1u64.to_le_bytes());
            let mut state = A64State::default();
            state.set_pc(PC);
            state.general_register_storage_mut()[1] = DATA as u64;
            state.general_register_storage_mut()[3] = 0x0123_4567_89ab_cdef;
            state.general_register_storage_mut()[5] = 1;
            let mut expected = state.clone();
            let memory = super::super::memory(&words);
            let monitor = RefCell::new(ExclusiveMonitorState::default());
            let events = VcpuEventState::default();
            for word in &words[..2] {
                execute_one_with_context(
                    InterpreterContext::new(
                        ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                        &memory,
                        &monitor,
                        &Timer,
                        &events,
                    ),
                    &mut expected,
                    *word,
                )
                .unwrap();
            }
            let reconstructed = run_memory_case(
                &words,
                &mut state,
                &mut arena,
                Some((DATA, 0, 0)),
                escape,
                false,
            );
            if escape {
                assert!(reconstructed.unwrap().0.completed_read.is_none());
                assert_eq!(arena[DATA..DATA + 8], 1u64.to_le_bytes());
            } else {
                expected.set_pc(PC + 12);
                let width = 1 << size;
                assert_eq!(
                    &arena[DATA..DATA + width],
                    &0x0123_4567_89ab_cdefu64.to_le_bytes()[..width]
                );
            }
            assert_eq!(state, expected);
        }
    }
}
