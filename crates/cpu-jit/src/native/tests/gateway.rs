use super::*;
use canonical::native;
use nixe_cpu::state::a64::A64State;

pub(super) fn landing(abi: HostAbi) -> Vec<u8> {
    match abi {
        HostAbi::X86_64 => vec![0xf3, 0x0f, 0x1e, 0xfa],
        HostAbi::Aarch64 => 0xd50324dfu32.to_le_bytes().to_vec(), // BTI jc
    }
}

#[test]
fn canonical_exit_validates_pc_and_reason() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, _) = contracts(abi, &[]);
        for pc in [
            integer(abi.reserved().frame),
            spill(0, 8),
            ValueLocation::constant(u128::MAX),
        ] {
            assert!(emit_canonical_exit(&source, pc, NativeExitReason::Dispatch, 0).is_err());
        }
        assert!(
            emit_canonical_exit(
                &source,
                ValueLocation::constant(0),
                NativeExitReason::None,
                0
            )
            .is_err()
        );
    }
}

#[test]
fn canonical_exit_preserves_dynamic_pc_until_writeback_finishes() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for pc in [
            integer(0),
            vector(0),
            spill(2048, 8),
            ValueLocation::constant(0x123456789abcdef0),
        ] {
            if !native(abi) {
                continue;
            }
            let (source, entry) = contracts(abi, &[(GuestValue::General(0), pc, pc)]);
            let owner = published::Published::new(|process, cache| {
                // Constant exit sources are valid, ingress destinations aren't.
                let ingress = if matches!(pc, ValueLocation::Constant(_)) {
                    contracts(abi, &[]).1
                } else {
                    entry
                };
                published::synthetic(
                    process.begin_unit(crate::executable::Tier::Lcq).unwrap(),
                    cache,
                    ingress,
                    source,
                    (vec![], None),
                    pc,
                    NativeExitReason::Control,
                )
            });
            let mut state = A64State::default();
            state.general_register_storage_mut()[0] = 0x123456789abcdef0;
            {
                let mut frame = NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
                let mut reader = owner.process.register().unwrap();
                let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
                    .unwrap()
                    .unwrap();
                let address =
                    invocation.payload().preferred().unwrap().canonical.get() as *const u8;
                let epoch = invocation.frame().execution_epoch;
                let result =
                    unsafe { enter_protected(invocation.frame(), std::ptr::null_mut(), address) }
                        .unwrap();
                assert_eq!(
                    result.poll,
                    PollOutcome {
                        sample: false,
                        exhausted: false
                    }
                );
                assert_eq!(invocation.frame().exit_pc, 0x123456789abcdef0);
                assert_eq!(invocation.frame().budget.slice_remaining, 11);
                assert_eq!(invocation.frame().execution_epoch, epoch);
                drop(invocation);
                assert_eq!(frame.execution_epoch, 0);
            }
            assert_eq!(state.pc(), 0x123456789abcdef0);
            owner.shutdown();
        }
    }
}

#[test]
fn polled_exits_share_writeback_and_preserve_each_cold_entry_contract() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for flags in 0..3 {
            let (mut source, mut entry) = canonical::complete(abi);
            if flags == 1 {
                source.nzcv = NzcvLocation::Host {
                    carry_inverted: true,
                };
                entry.nzcv = source.nzcv.clone();
            } else if flags == 2 {
                source.nzcv = NzcvLocation::Deferred(LazyFlags::Subtract {
                    lhs: integer(0),
                    rhs: integer(1),
                    result: integer(2),
                    width: 64,
                });
            }
            for pc in [
                integer(0),
                vector(0),
                spill(2048, 8),
                ValueLocation::constant(0x123456789abcdef0),
            ] {
                let (shared, _) =
                    super::super::canonical::emit_polled_exit(&source, pc, &[], false).unwrap();
                let dispatch =
                    super::super::canonical::emit_dispatch_fallback(&source, pc, 0).unwrap();
                let exit = emit_canonical_exit(&source, pc, NativeExitReason::Control, 0).unwrap();
                assert!(shared.len() < dispatch.len() + exit.len());
                if !native(abi) {
                    continue;
                }
                for reason in [NativeExitReason::Dispatch, NativeExitReason::Control] {
                    let mut results = Vec::new();
                    for use_shared in [false, true] {
                        let owner = published::Published::new(|process, cache| {
                            let identity =
                                process.begin_unit(crate::executable::Tier::Lcq).unwrap();
                            source.site = ExitSiteKey {
                                source: identity.version(),
                                state_map: 0,
                            };
                            let (shared, [_, control]) =
                                super::super::canonical::emit_polled_exit(&source, pc, &[], false)
                                    .unwrap();
                            let mut bytes = landing(abi);
                            bytes.extend(emit_canonical_entry(&entry).unwrap());
                            // Initialize the explicit spill PC even when this complete
                            // map keeps guest X0 in a register rather than that slot.
                            let mut initialize = Emitter::new(abi);
                            initialize.copy(Copy {
                                source: ValueLocation::constant(0xabcdef),
                                destination: spill(2048, 8),
                                bytes: 8,
                            });
                            bytes.extend(initialize.finish());
                            let exit_offset = bytes.len() as u32;
                            if use_shared {
                                if reason == NativeExitReason::Control {
                                    if abi == HostAbi::X86_64 {
                                        bytes.push(0xe9);
                                        bytes.extend((control as i32).to_le_bytes());
                                    } else {
                                        bytes.extend(
                                            (0x14000000 | ((control as u32 + 4) / 4)).to_le_bytes(),
                                        );
                                    }
                                }
                                bytes.extend_from_slice(&shared);
                            } else {
                                bytes.extend(emit_canonical_exit(&source, pc, reason, 0).unwrap());
                            }
                            published::encoded(
                                identity,
                                cache,
                                bytes,
                                crate::lifetime::unit::Entry {
                                    key: published::key(),
                                    canonical_offset: 0,
                                    fast_offset: 0,
                                    contract: entry.clone(),
                                },
                                vec![crate::lifetime::unit::StateRecord {
                                    exit: None,
                                    transfer: None,
                                    native_offset: exit_offset,
                                    state: source.clone(),
                                }],
                                Box::new([]),
                            )
                        });
                        let mut reader = owner.process.register().unwrap();
                        let (mut state, _) = canonical::pattern(&entry);
                        state.set_fpcr(0);
                        state.set_fpsr(0);
                        {
                            let mut frame =
                                NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                            let mut invocation =
                                unsafe { reader.admit(&mut frame, published::key()) }
                                    .unwrap()
                                    .unwrap();
                            let address = invocation.payload().preferred().unwrap().canonical.get()
                                as *const u8;
                            let epoch = invocation.frame().execution_epoch;
                            let frame = invocation.frame();
                            let result =
                                unsafe { enter_protected(frame, std::ptr::null_mut(), address) }
                                    .unwrap();
                            assert_eq!(result.reason, reason);
                            assert_eq!(frame.exit_source_version, source.site.source.get());
                            assert_eq!(frame.exit_state_map, source.site.state_map);
                            assert_eq!(frame.execution_epoch, epoch);
                            assert_eq!(frame.budget.slice_remaining, 1000);
                            assert_eq!(frame.budget.sample_remaining, 77);
                        }
                        results.push(state);
                        drop(reader);
                        owner.shutdown();
                    }
                    assert_eq!(
                        results[0], results[1],
                        "flags={flags}, pc={pc:?}, reason={reason:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_native_budget_restores_fp_without_announcing_quiescence() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, entry) = contracts(abi, &[]);
        let mut invalid = moves::Emitter::new(abi);
        invalid.constant(abi.reserved().poll, 8, 8); // Armed span is only seven.
        let body = invalid.finish();
        if !native(abi) {
            continue;
        }
        let owner = published::Published::new(|process, cache| {
            published::synthetic(
                process.begin_unit(crate::executable::Tier::Lcq).unwrap(),
                cache,
                entry,
                source,
                (body, None),
                ValueLocation::constant(4),
                NativeExitReason::Internal,
            )
        });
        let mut state = A64State::default();
        {
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
            let mut reader = owner.process.register().unwrap();
            let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
                .unwrap()
                .unwrap();
            let address = invocation.payload().preferred().unwrap().canonical.get() as *const u8;
            let epoch = invocation.frame().execution_epoch;
            let result = unsafe {
                invocation.frame().ensure_fp().unwrap();
                crate::fp_env::tests::divide_by_zero();
                enter_protected(invocation.frame(), std::ptr::null_mut(), address)
            };
            assert_eq!(
                result,
                Err(NativeReturnError::Budget(BudgetError::InvalidDeadline))
            );
            assert_eq!(
                (
                    invocation.frame().host_fp.saved,
                    invocation.frame().host_fp.active
                ),
                (0, 0)
            );
            assert_eq!(unsafe { *invocation.frame().canonical.fpsr }, 2);
            assert_eq!(
                (
                    invocation.frame().budget.sample_remaining,
                    invocation.frame().budget.slice_remaining
                ),
                (7, 11)
            );
            assert_eq!(invocation.frame().execution_epoch, epoch);
            drop(invocation);
            assert_eq!(frame.execution_epoch, 0);
        }
        owner.shutdown();
    }
}
