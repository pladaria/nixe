use super::*;

const FADD: u32 = 0x1e222820;
const NOP: u32 = 0xd503201f;

fn activation_pcs(body: &Body) -> Vec<u64> {
    body.fp_activations
        .iter()
        .map(|pending| pending.pc.get())
        .collect()
}

#[test]
fn hcq_fp_straight_paths_share_one_activation_but_public_entries_need_their_own() {
    let graph = graph(&[(0, &[FADD, FADD, 0x14000002]), (16, &[FADD, RET])]);
    for entries in [vec![0], vec![0, block(&graph, 16)]] {
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (context, body) = emitted(&graph, &entries, abi);
            assert_eq!(
                activation_pcs(&body),
                if entries.len() == 1 {
                    vec![0]
                } else {
                    vec![0, 16]
                }
            );
            assert_eq!(
                context.compiled_code().unwrap().buffer.nixe_entries.len(),
                entries.len() + body.fp_activations.len()
            );
            // Three eligibility failures and the actual RET, not activation exits.
            assert_eq!(body.exits.len(), 4);
            let ids: std::collections::HashSet<_> = context
                .compiled_code()
                .unwrap()
                .buffer
                .nixe_states
                .iter()
                .map(|map| map.id)
                .collect();
            assert_eq!(
                ids.len(),
                context.compiled_code().unwrap().buffer.nixe_states.len()
            );
        }
    }
}

#[test]
fn hcq_fp_mixed_and_all_active_diamonds_consume_definite_path_proof() {
    for right in [NOP, FADD] {
        let graph = graph(&[
            (0, &[0x54000080]),
            (4, &[FADD, 0x14000004]),
            (16, &[right, 0x14000001]),
            (24, &[FADD, RET]),
        ]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0], abi);
            assert_eq!(
                activation_pcs(&body),
                if right == NOP {
                    vec![4, 24]
                } else {
                    vec![4, 16]
                }
            );
            assert!(body.exits.iter().all(|exit| exit.state.dirty.fpsr));
        }
    }
}

#[test]
fn hcq_fp_loop_keeps_initial_activation_without_reactivating_the_internal_body() {
    let graph = graph(&[(0, &[FADD, 0x14000001]), (8, &[FADD, 0x17fffffd])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0], abi);
        assert_eq!(activation_pcs(&body), vec![0]);
        assert_eq!(body.exits.len(), 2); // only eligibility failures
    }
}

#[test]
fn hcq_fp_irreducible_cycle_does_not_hide_the_first_inactive_path() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[FADD, 0x14000002]),
        (16, &[0x54000020]),
        (20, &[FADD, 0x17fffffb]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0], abi);
        assert_eq!(activation_pcs(&body), vec![4, 20]);
    }
}

#[test]
fn hcq_fp_disconnected_public_entries_never_inherit_emission_order_state() {
    let graph = graph(&[(0, &[FADD, RET]), (64, &[FADD, RET])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
        assert_eq!(activation_pcs(&body), vec![0, 64]);
    }
}

#[test]
fn hcq_fp_comparison_does_not_activate_and_its_cold_exit_keeps_old_flags() {
    let graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[0x1e222020, FADD, RET])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0], abi);
        assert_eq!(activation_pcs(&body), vec![12]);
        assert!(matches!(
            body.exits[0].state.flags,
            Some(LazyFlags::Add { .. })
        ));
        assert!(matches!(
            body.exits[1].state.flags,
            Some(LazyFlags::Packed(_))
        ));
        assert_eq!(body.exits[0].guest.pc.get(), 8);
    }
}

#[test]
fn hcq_fp_exact_and_status_boundaries_keep_pending_state_without_new_activation() {
    // FCCMP completes outside native execution, as do reads of accumulated FPSR.
    for word in [0x1e62042a, 0xd53b4420] {
        let graph = graph(&[(0, &[FADD, ADDS, 0x1400000e]), (64, &[word])]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            assert_eq!(activation_pcs(&body), vec![0]);
            assert_eq!(body.exits.len(), 2);
            let exit = &body.exits[1];
            assert_eq!(exit.guest.pc.get(), 64);
            assert_eq!(exit.reason, NativeExitReason::Architectural);
            assert!(exit.state.dirty.vector[0]);
            assert!(exit.state.dirty.integer.x[0]);
            assert_eq!(exit.state.dirty.nzcv, crate::analysis::NZCV);
            assert!(exit.state.dirty.fpsr);
            assert!(matches!(exit.state.flags, Some(LazyFlags::Packed(_))));
        }
    }
}

#[test]
fn hcq_fp_continuations_transport_typed_lazy_operands_and_subsequent_fault_state() {
    // ADCS captures an I8 carry; a public successor must activate independently.
    let graph = graph(&[
        (0, &[0xba020020, FADD, 0x1400000e]),
        (64, &[FADD, 0xf9400020, RET]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
        assert_eq!(activation_pcs(&body), vec![4, 64]);
        assert_eq!(body.faults.len(), 1);
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
        assert_eq!(faults[0].instruction.block_key().pc.get(), 68);
        assert!(states[0].state.dirty_live.integer.x[0]);
        assert!(states[0].state.dirty_live.vector[0]);
        assert!(states[0].state.host_fpsr_pending);
        assert!(matches!(
            body.exits[0].state.flags,
            Some(LazyFlags::AddCarry { .. })
        ));
    }
}
