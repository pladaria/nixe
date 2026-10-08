//! Fast inputs may be newer than their canonical homes even when only read.
use super::*;
use crate::abi::{ExitStateMap, ValueBinding, ValueLocation};

#[test]
fn fast_inputs_survive_read_only_use_and_fp_activation() {
    crate::native::check_host().unwrap();
    for fp in [false, true] {
        let mut words = vec![
            0x9a14_0260, // ADC X0,X19,X20 (reads C without defining NZCV).
            0x9100_03e1, // MOV X1,SP.
            0xd53b_d042, // MRS X2,TPIDR_EL0.
            0x4ea2_1c20, // ORR V0.16B,V1.16B,V2.16B.
        ];
        if fp {
            words.push(0x1e62_2823); // FADD D3,D1,D2, through FP activation.
        }
        words.push(0xd420_0000);
        let memory = memory(&words);
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        let fragment = &compilation.fragment;
        let abi = native_abi();
        let mut lowered = Compiler::new(abi)
            .unwrap()
            .lower(fragment, compilation.identity.version())
            .unwrap();
        assert_eq!(lowered.entry.live_in.nzcv, crate::analysis::C);
        let mut initial = integer::initial_state();
        initial.set_tpidr_el0(0x1234_5678_9abc_def0);
        initial.set_vector(1, u128::from(1.0f64.to_bits()));
        initial.set_vector(2, u128::from(2.0f64.to_bits()));
        let mut expected = initial.clone();
        for &word in &words[..words.len() - 1] {
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word)
                .unwrap();
        }
        let mut actual = initial.clone();
        let bindings = lowered
            .entry
            .bindings
            .iter()
            .map(|binding| {
                let value = match binding.value {
                    GuestValue::General(index) => {
                        actual.general_register_storage_mut()[usize::from(index)] = 0;
                        u128::from(initial.general_register_storage_mut()[usize::from(index)])
                    }
                    GuestValue::Sp => {
                        *actual.stack_pointer_storage_mut() = 0;
                        u128::from(*initial.stack_pointer_storage_mut())
                    }
                    GuestValue::Vector(index) => {
                        actual.set_vector(index, 0);
                        initial.vector(index).unwrap()
                    }
                    GuestValue::TpidrEl0 => {
                        actual.set_tpidr_el0(0);
                        u128::from(initial.tpidr_el0())
                    }
                    GuestValue::Fpcr => u128::from(initial.fpcr()),
                    other => panic!("unexpected input {other:?}"),
                };
                ValueBinding {
                    value: binding.value,
                    location: ValueLocation::constant(value),
                }
            })
            .collect();
        let mut source = ExitStateMap {
            site: ExitSiteKey {
                source: CodeVersion::new(2).unwrap(),
                state_map: 0,
            },
            abi,
            live: lowered.entry.live_in,
            dirty_live: lowered.entry.live_in,
            bindings,
            nzcv: NzcvLocation::Packed(ValueLocation::constant(u128::from(initial.nzcv().bits()))),
            host_fpsr_pending: false,
        };
        source.dirty_live.fpcr = false;
        actual.set_nzcv(Nzcv::from_bits(initial.nzcv().bits() ^ (1 << 29)));
        // A test-owned native predecessor installs newer values directly into
        // the compiled fast contract. No canonical ingress may repair the test.
        let mut ingress = landing(abi);
        ingress.extend(crate::native::emit_fast_transfer(&source, &lowered.entry).unwrap());
        while !ingress.len().is_multiple_of(8) {
            ingress.extend(nop(abi));
        }
        let jump = ingress.len();
        ingress.resize(jump + 8, 0);
        let mut bytes = lowered.output.bytes.into_vec();
        let start = append(&mut bytes, &ingress);
        StateMap {
            id: 0,
            offset: (start + jump) as u32,
            entry: false,
            patch_bytes: if abi == HostAbi::X86_64 { 8 } else { 4 },
            fault_bytes: 0,
            poll: None,
            subtract_flags: false,
            values: vec![],
        }
        .patch_exit(&mut bytes, 0, u64::from(lowered.fast))
        .unwrap();
        lowered.output.bytes = bytes.into_boxed_slice();
        lowered.canonical = start as u32;
        Compiler::publish_lowered(compilation, lowered, &process, &cache, &memory).unwrap();
        {
            let mut frame = NativeFrame::new(&mut actual, PollBudget::new(4096, 1000).unwrap());
            let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
            let address = invocation.payload().preferred().unwrap().canonical.get();
            unsafe {
                crate::native::enter_protected(
                    invocation.frame(),
                    std::ptr::null_mut(),
                    address as *const u8,
                )
                .unwrap();
            }
            drop(invocation);
            assert_eq!(frame.execution_epoch, 0);
        }
        assert_eq!(actual, expected, "FP activation: {fp}");
    }
}

#[test]
fn prefault_maps_keep_inputs_needed_only_after_the_fault() {
    // X19 and C are read after the potentially faulting load. Their incoming
    // values must already be recoverable at the load, not only after first use.
    let memory = memory(&[0xf940_0020, 0x9a1f_0262, 0xd420_0000]);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::for_arena(abi, 1 << 20)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        let state = &lowered.states[lowered.faults[0].state_map as usize].state;
        // Integer-only code permits inherited status without activating FP or
        // introducing a physical software-FPSR operand/canonical store.
        assert_eq!(lowered.output.metadata.entries.len(), 1);
        for record in &lowered.states {
            assert!(record.state.host_fpsr_pending && record.state.dirty_live.fpsr);
            assert!(
                record
                    .state
                    .bindings
                    .iter()
                    .all(|b| b.value != GuestValue::Fpsr)
            );
        }
        assert!(state.dirty_live.integer.x.contains(19));
        assert!(state.dirty_live.integer.x.contains(1));
        assert!(!state.dirty_live.integer.x.contains(0));
        assert_eq!(state.dirty_live.nzcv, crate::analysis::C);
        assert!(matches!(
            state.nzcv,
            NzcvLocation::Deferred(LazyFlags::Canonical(_))
        ));
        state.validate().unwrap();
    }
}

#[test]
fn coordinated_entries_carry_unused_values_without_a_state_adapter() {
    // A produces X19 and V1. B does not read either; C eventually consumes both.
    // Allocation constraints must survive real codegen on both host backends.
    let words = [
        0x9100_0673, // ADD X19,X19,#1
        0x4ea3_1c61, // ORR V1.16B,V3.16B,V3.16B
        0x1400_0001, // B B
        0xd503_201f, // NOP
        0x1400_0001, // B C
        0x9100_0260, // ADD X0,X19,#0
        0x4ea1_1c20, // ORR V0.16B,V1.16B,V1.16B
        0xd420_0000,
    ];
    let memory = memory(&words);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::new(abi).unwrap();
        let a = compiler
            .lower(
                &Fragment::capture(&memory, key()).unwrap(),
                CodeVersion::new(1).unwrap(),
            )
            .unwrap();
        let source = &a.states[0].state;
        let plan = crate::frontend::entry::Plan::from_exit(source);
        let b = compiler
            .lower_with_plan(
                &Fragment::capture(
                    &memory,
                    key().at(GuestVirtualAddress::new(PC + 12)).unwrap(),
                )
                .unwrap(),
                CodeVersion::new(2).unwrap(),
                &plan,
            )
            .unwrap();
        assert!(b.entry.live_in.integer.x.contains(19));
        assert!(b.entry.live_in.vector.contains(1));
        assert!(
            crate::native::emit_chain_transfer(source, &b.entry)
                .unwrap()
                .is_empty()
        );
        let c = compiler
            .lower_with_plan(
                &Fragment::capture(
                    &memory,
                    key().at(GuestVirtualAddress::new(PC + 20)).unwrap(),
                )
                .unwrap(),
                CodeVersion::new(3).unwrap(),
                &crate::frontend::entry::Plan::from_exit(&b.states[0].state),
            )
            .unwrap();
        assert!(
            crate::native::emit_chain_transfer(&b.states[0].state, &c.entry)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn unused_faulting_destination_keeps_its_old_canonical_home() {
    let memory = memory(&[0x9100_0400, 0x1400_0001, 0xf940_0020, 0xd420_0000]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::for_arena(abi, 1 << 20).unwrap();
        let a = compiler
            .lower(
                &Fragment::capture(&memory, key()).unwrap(),
                CodeVersion::new(1).unwrap(),
            )
            .unwrap();
        let plan = crate::frontend::entry::Plan::from_exit(&a.states[0].state);
        let b = compiler
            .lower_with_plan(
                &Fragment::capture(&memory, key().at(GuestVirtualAddress::new(PC + 8)).unwrap())
                    .unwrap(),
                CodeVersion::new(2).unwrap(),
                &plan,
            )
            .unwrap();
        let pre = &b.states[b.faults[0].state_map as usize].state;
        assert!(!b.entry.live_in.integer.x.contains(0));
        assert!(!b.entry.discard.integer.x.contains(0));
        assert!(!pre.dirty_live.integer.x.contains(0));
        assert!(
            !crate::native::emit_chain_transfer(&a.states[0].state, &b.entry)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn discarded_inputs_are_overwritten_before_every_precise_observation() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for (words, discard_x0) in [
            (vec![0xd28000e0, 0xd4200000], true),  // MOVZ X0,#7
            (vec![0xf9400020, 0xd4200000], false), // LDR X0,[X1] can fault PRE
            (vec![0xd53b4420, 0xd4200000], false), // MRS X0,FPSR exits PRE
            (vec![0xd28000e0, 0xf9400021, 0xd4200000], true),
            (vec![0xf9400021, 0xd28000e0, 0xd4200000], false),
            (vec![0xf28000e0, 0xd4200000], false), // MOVK reads old destination
        ] {
            let memory = memory(&words);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let lowered = Compiler::for_arena(abi, 1 << 20)
                .unwrap()
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(
                lowered.entry.discard.integer.x.contains(0),
                discard_x0,
                "{abi:?}: {words:x?}"
            );
            assert!(
                lowered
                    .entry
                    .live_in
                    .intersection(lowered.entry.discard)
                    .is_empty()
            );
        }
    }
}
