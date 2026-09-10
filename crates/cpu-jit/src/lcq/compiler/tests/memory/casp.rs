use super::*;
use crate::lifetime::unit::Access;

#[test]
fn casp_x_without_cmpxchg16b_reports_the_missing_host_capability() {
    let memory = super::super::memory(&[word(3, 1, 2, 4) | (1 << 30), 0xd420_0000]);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    let mut compiler = Compiler::for_arena(HostAbi::X86_64, ARENA).unwrap();
    let mut target = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
    target.set("has_cmpxchg16b", "false").unwrap();
    compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
    let error = compiler
        .lower(&fragment, CodeVersion::new(1).unwrap())
        .err()
        .unwrap();
    assert!(error.to_string().contains("CMPXCHG16B"));
}

#[test]
fn delivered_casp_x_retry_escape_and_alignment_preserve_both_pre_registers() {
    for ordering in 0..4 {
        for success in [false, true] {
            for (escape, misaligned) in [(false, false), (true, false), (true, true)] {
                let words = [
                    0xf100_0529,
                    0x9100_0442,
                    0x9100_0463,
                    word(ordering, 1, 2, 4) | (1 << 30),
                    0xd420_0000,
                ];
                let initial: u128 = if success {
                    0xeeee_eeee_0000_0002_ffff_ffff_0000_0001
                } else {
                    0x0123_4567_89ab_cdef_9876_5432_8123_4567
                };
                let mut arena = vec![0; ARENA];
                arena[DATA..DATA + 16].copy_from_slice(&initial.to_le_bytes());
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = DATA as u64 + 8 * u64::from(misaligned);
                state.general_register_storage_mut()[2] = 0xffff_ffff_0000_0000;
                state.general_register_storage_mut()[3] = 0xeeee_eeee_0000_0001;
                state.general_register_storage_mut()[4] = 0xdead_beef_7654_3210;
                state.general_register_storage_mut()[5] = 0xfeed_cafe_fedc_ba98;
                state.general_register_storage_mut()[9] = 1;
                let mut expected = state.clone();
                expected.general_register_storage_mut()[2] += 1;
                expected.general_register_storage_mut()[3] += 1;
                expected.general_register_storage_mut()[9] = 0;
                expected.set_nzcv(Nzcv::from_bits(Nzcv::Z | Nzcv::C));
                expected.set_pc(PC + 12);
                let result = run_memory_case(
                    &words,
                    &mut state,
                    &mut arena,
                    Some((if misaligned { ARENA } else { DATA }, 0, 0)),
                    escape,
                );
                if escape {
                    let (reconstructed, access) = result.unwrap();
                    assert!(reconstructed.completed_read.is_none());
                    if misaligned {
                        assert_eq!(
                            access.guest_fault(SPACE).unwrap().reason,
                            nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                                required_alignment: 16
                            }
                        );
                    }
                } else {
                    expected.general_register_storage_mut()[2] = initial as u64;
                    expected.general_register_storage_mut()[3] = (initial >> 64) as u64;
                    expected.set_pc(PC + 16);
                }
                assert_eq!(state, expected);
                let wanted = if !escape && success {
                    0xfeed_cafe_fedc_ba98_dead_beef_7654_3210
                } else {
                    initial
                };
                assert_eq!(arena[DATA..DATA + 16], wanted.to_le_bytes());
            }
        }
    }
}

pub(super) fn word(ordering: u32, rn: u32, rs: u32, rt: u32) -> u32 {
    0x0820_7c00 | ((ordering & 1) << 22) | ((ordering >> 1) << 15) | (rs << 16) | (rn << 5) | rt
}

#[test]
fn casp_matches_pairs_aliases_and_lazy_flags() {
    for (size, ordering) in (0..2).flat_map(|size| (0..4).map(move |ordering| (size, ordering))) {
        for (rn, rs, rt) in [
            (1, 2, 4),
            (31, 2, 4),
            (2, 2, 4),
            (3, 2, 4),
            (4, 2, 4),
            (5, 2, 4),
            (1, 2, 2),
            (1, 30, 4),
            (1, 2, 30),
            (31, 30, 30),
        ] {
            for success in [false, true] {
                let word = word(ordering, rn, rs, rt) | (size << 30);
                let words = [0xf100_0529, word, 0x9a1f_014a, 0xd420_0000]; // SUBS X9; CASP; ADC X10
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[2] = 0xdead_beef_8123_4567;
                state.general_register_storage_mut()[3] = 0xfeed_cafe_9876_5432;
                state.general_register_storage_mut()[4] = 0xffff_ffff_7654_3210;
                state.general_register_storage_mut()[5] = 0xffff_ffff_fedc_ba98;
                state.general_register_storage_mut()[30] = 0xffff_ffff_8765_4321;
                state.general_register_storage_mut()[9] = 1;
                if rn != 31 {
                    state.general_register_storage_mut()[rn as usize] = DATA as u64;
                }
                *state.stack_pointer_storage_mut() = DATA as u64;
                let mut read = |r: u32| {
                    if r == 31 {
                        0
                    } else {
                        let value = state.general_register_storage_mut()[r as usize];
                        u128::from(if size == 0 {
                            value as u32 as u64
                        } else {
                            value
                        })
                    }
                };
                let initial = (read(rs) | (read(rs + 1) << (32 << size)))
                    ^ if success { 0 } else { 1 << ((64 << size) - 1) };
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
                arena[DATA..DATA + 16].copy_from_slice(&initial.to_le_bytes());
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
                assert_eq!(state, expected, "{word:08x} success={success}");
                let mut wanted = [0; 16];
                memory
                    .read_bytes(SPACE, GuestVirtualAddress::new(DATA as u64), &mut wanted)
                    .unwrap();
                assert_eq!(arena[DATA..DATA + 16], wanted);
            }
        }
    }
}

#[test]
fn casp_maps_describe_one_transaction_and_both_pre_destinations() {
    for (size, abi, lse) in [
        (HostAbi::X86_64, false),
        (HostAbi::Aarch64, false),
        (HostAbi::Aarch64, true),
    ]
    .into_iter()
    .flat_map(|(abi, lse)| (0..2).map(move |size| (size, abi, lse)))
    {
        let memory =
            super::super::memory(&[0xf100_0529, word(3, 1, 2, 4) | (size << 30), 0xd420_0000]);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
        if abi == HostAbi::Aarch64 {
            let mut target = isa::lookup("aarch64-unknown-linux-gnu".parse().unwrap()).unwrap();
            target.set("use_bti", "true").unwrap();
            target
                .set("has_lse", if lse { "true" } else { "false" })
                .unwrap();
            compiler.isa = target.finish(compiler.isa.flags().clone()).unwrap();
        } else {
            let mut target = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
            target.set("has_cmpxchg16b", "true").unwrap();
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
            assert_eq!(fault.bytes, 8 << size);
            assert_eq!((fault.subaccess, fault.commit_stage), (0, 0));
            assert!(fault.completed_read.is_none());
            let state = &lowered.states[fault.state_map as usize].state;
            assert!(!state.dirty_live.integer.x[2]);
            assert!(!state.dirty_live.integer.x[3]);
            assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
            state.validate().unwrap();
        }
        let clif = compiler.context.func.display().to_string();
        assert_eq!(clif.matches("atomic_cas").count(), 1);
        assert_eq!(clif.matches("nixe_fault_start").count(), 1);
        assert!(!clif.contains("call"));
        assert!(!clif.contains("store"));
        if size == 0 {
            assert!(!clif.contains("i128"));
        }
    }
}

#[test]
fn delivered_casp_w_retry_escape_and_alignment_preserve_both_pre_registers() {
    for ordering in 0..4 {
        for success in [false, true] {
            for (escape, misaligned) in [(false, false), (true, false), (true, true)] {
                let words = [
                    0xf100_0529,
                    0x9100_0442,
                    0x9100_0463,
                    word(ordering, 1, 2, 4),
                    0xd420_0000,
                ];
                let initial: u64 = if success {
                    0x0000_0002_0000_0001
                } else {
                    0x9876_5432_8123_4567
                };
                let mut arena = vec![0; ARENA];
                arena[DATA..DATA + 8].copy_from_slice(&initial.to_le_bytes());
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = DATA as u64 + u64::from(misaligned);
                state.general_register_storage_mut()[2] = 0xffff_ffff_0000_0000;
                state.general_register_storage_mut()[3] = 0xeeee_eeee_0000_0001;
                state.general_register_storage_mut()[4] = 0xdead_beef_7654_3210;
                state.general_register_storage_mut()[5] = 0xfeed_cafe_fedc_ba98;
                state.general_register_storage_mut()[9] = 1;
                let mut expected = state.clone();
                expected.general_register_storage_mut()[2] += 1;
                expected.general_register_storage_mut()[3] += 1;
                expected.general_register_storage_mut()[9] = 0;
                expected.set_nzcv(Nzcv::from_bits(Nzcv::Z | Nzcv::C));
                expected.set_pc(PC + 12);
                let result = run_memory_case(
                    &words,
                    &mut state,
                    &mut arena,
                    Some((if misaligned { ARENA } else { DATA }, 0, 0)),
                    escape,
                );
                if escape {
                    let (reconstructed, access) = result.unwrap();
                    assert!(reconstructed.completed_read.is_none());
                    if misaligned {
                        assert_eq!(
                            access.guest_fault(SPACE).unwrap().reason,
                            nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                                required_alignment: 8
                            }
                        );
                    }
                } else {
                    expected.general_register_storage_mut()[2] = initial as u32 as u64;
                    expected.general_register_storage_mut()[3] = initial >> 32;
                    expected.set_pc(PC + 16);
                }
                assert_eq!(state, expected);
                let wanted = if !escape && success {
                    0xfedc_ba98_7654_3210
                } else {
                    initial
                };
                assert_eq!(arena[DATA..DATA + 8], wanted.to_le_bytes());
            }
        }
    }
}
