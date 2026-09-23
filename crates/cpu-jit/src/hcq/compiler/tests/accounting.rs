use super::*;
use crate::abi::{NativeFrame, PollBudget};
use nixe_cpu::state::a64::{A64State, Nzcv};

#[test]
fn hcq_external_checkpoints_charge_the_executed_path_once_including_overshoot() {
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    crate::native::check_host().unwrap();
    for terminal in [0x14000004, 0x54000080, 0x94000004, RET] {
        let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[0xd503201f, terminal])]);
        let entries = [0, block(&graph, 64)];
        let (process, owner) = published(&graph, &entries, abi);
        let mut reader = process.register().unwrap();
        for entry in owner.entries.iter() {
            let cost = if entry.key.pc.get() == 0 { 4 } else { 2 };
            for balance in [1, cost, 1000] {
                for zero in [false, true] {
                    let mut state = A64State::default();
                    state.set_pc(entry.key.pc.get());
                    state.general_register_storage_mut()[1] = if zero { u64::MAX } else { 0 };
                    state.general_register_storage_mut()[30] = 128;
                    state.set_nzcv(Nzcv::from_bits(if zero { Nzcv::Z } else { 0 }));
                    {
                        let mut frame = NativeFrame::new(
                            &mut state,
                            PollBudget::new(balance, balance).unwrap(),
                        );
                        let mut invocation = unsafe { reader.admit(&mut frame, entry.key) }
                            .unwrap()
                            .unwrap();
                        let address = invocation.payload().preferred().unwrap().canonical.get();
                        let frame = invocation.frame();
                        let result = unsafe {
                            crate::native::enter_protected(
                                frame,
                                std::ptr::null_mut(),
                                address as *const u8,
                            )
                        }
                        .unwrap();
                        assert_eq!(frame.budget.slice_remaining, balance - cost);
                        assert_eq!(result.poll.exhausted, balance <= cost);
                        assert_eq!(result.poll.sample, balance <= cost);
                        // Slice exhaustion uses the same source-local canonical
                        // fallback; the caller consumes `poll.exhausted`.
                        assert_eq!(result.reason, NativeExitReason::Dispatch);
                        drop(invocation);
                    }
                    let pc = if terminal == RET {
                        128
                    } else if terminal == 0x54000080 && !zero {
                        72
                    } else {
                        84
                    };
                    assert_eq!(state.pc(), pc);
                    if terminal == 0x94000004 {
                        assert_eq!(state.general_register_storage_mut()[30], 72);
                    }
                }
            }
        }
    }
}

#[test]
fn hcq_prefault_and_preexit_prefixes_are_source_local_at_public_entries() {
    let graph = graph(&[
        (0, &[ADDS, 0x1400000f]),
        (64, &[0xd503201f, 0xa9400c22, 0xd4200000]), // NOP; LDP; BRK
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for entries in [vec![0], vec![0, block(&graph, 64)]] {
            let (context, body) = emitted(&graph, &entries, abi);
            let mut states = Vec::new();
            let faults = memory::records(
                abi,
                CodeVersion::new(1).unwrap(),
                graph.blocks[0].key,
                context.compiled_code().unwrap(),
                &context.func,
                &body.faults,
                &mut states,
            )
            .unwrap();
            assert_eq!(faults.len(), 2);
            for fault in faults {
                assert_eq!(fault.instruction.block_key().pc.get(), 68);
                assert_eq!(fault.completed, 1); // not array ordinal 3 or PC distance 17
            }
            assert_eq!(body.exits.len(), 1);
            assert_eq!(body.exits[0].guest.pc.get(), 72);
            assert_eq!(body.exits[0].completed, 2); // BRK is PRE, excludes itself
        }
    }
}

#[test]
fn hcq_external_terminals_include_the_branch_but_fp_cold_exits_remain_pre() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for terminal in [0x14000004, 0x54000080, 0x94000004, RET] {
            let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[0xd503201f, terminal])]);
            let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            assert!(!body.exits.is_empty());
            for exit in body.exits {
                assert_eq!(exit.guest.pc.get(), 68);
                assert_eq!(exit.reason, NativeExitReason::Dispatch);
                // Driver-owned branches charge before selecting the edge;
                // call/return terminals still charge at the exit checkpoint.
                assert_eq!(
                    exit.completed,
                    if matches!(terminal, 0x94000004 | RET) {
                        2
                    } else {
                        0
                    }
                );
            }
        }
        let graph = graph(&[
            (0, &[ADDS, 0x1400000f]),
            (64, &[0xd503201f, 0x1e222820, RET]),
        ]);
        let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
        assert!(
            body.exits
                .iter()
                .any(|exit| exit.reason != NativeExitReason::Dispatch)
        );
        for exit in body.exits {
            let dispatch = exit.reason == NativeExitReason::Dispatch;
            assert_eq!(exit.guest.pc.get(), if dispatch { 72 } else { 68 });
            assert_eq!(exit.completed, if dispatch { 3 } else { 1 });
        }
    }
}
