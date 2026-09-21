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
        let fragment = Fragment::capture(&memory, key()).unwrap();
        let abi = native_abi();
        let mut lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
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
                    location: ValueLocation::Constant(value),
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
            nzcv: NzcvLocation::Packed(ValueLocation::Constant(u128::from(initial.nzcv().bits()))),
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
            values: vec![],
        }
        .patch_exit(&mut bytes, 0, u64::from(lowered.fast))
        .unwrap();
        lowered.output.bytes = bytes.into_boxed_slice();
        let cache = Cache::new().unwrap();
        let owner = cache.install(lowered.output, Tier::Lcq, |_| None).unwrap();
        {
            let mut frame = NativeFrame::new(&mut actual, PollBudget::new(4096, 1000).unwrap());
            // Isolated copy/state proof: the test owns the complete allocation
            // until execution and reconstruction finish; no published link.
            frame.execution_epoch = 1;
            unsafe {
                frame.begin_fp();
                crate::native::enter_protected(
                    &mut frame,
                    std::ptr::null_mut(),
                    (owner.allocation.address() + start) as *const u8,
                )
                .unwrap();
            }
            frame.execution_epoch = 0;
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
        assert!(state.dirty_live.integer.x[19]);
        assert!(state.dirty_live.integer.x[1]);
        assert!(!state.dirty_live.integer.x[0]);
        assert_eq!(state.dirty_live.nzcv, crate::analysis::C);
        assert!(matches!(
            state.nzcv,
            NzcvLocation::Deferred(LazyFlags::Canonical(_))
        ));
        state.validate().unwrap();
    }
}
