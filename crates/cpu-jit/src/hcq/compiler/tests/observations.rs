use super::*;
use crate::abi::{LazyFlags, NzcvLocation};
use crate::lifetime::unit::{Access, FaultRecord, StateRecord};

fn faults(
    abi: HostAbi,
    graph: &Graph,
    context: &Context,
    body: &Body,
) -> (Box<[FaultRecord]>, Vec<StateRecord>) {
    let mut states = Vec::new();
    let records = memory::records(
        abi,
        CodeVersion::new(1).unwrap(),
        graph.blocks[0].key,
        context.compiled_code().unwrap(),
        &context.func,
        &body.faults,
        &mut states,
    )
    .unwrap();
    (records, states)
}

#[test]
fn hcq_memory_faults_keep_noncontiguous_source_and_incoming_lazy_state() {
    // Carry a dirty load destination and lazy ADDS from a different block.
    let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[0xf8408420, RET])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0], abi);
        let (faults, states) = faults(abi, &graph, &context, &body);
        assert_eq!(faults.len(), 1);
        let fault = &faults[0];
        assert_eq!(fault.instruction.block_key().pc.get(), 64);
        assert_eq!(fault.completed, 0); // source block, not the unit's third word
        assert_eq!(fault.access, Access::Read);
        assert_eq!(fault.bytes, 8);
        assert!(fault.native_end > fault.native_start);
        let state = &states[fault.state_map as usize].state;
        assert!(state.dirty_live.integer.x[0]); // old destination, not the load result
        assert!(state.dirty_live.integer.x[1]); // PRE post-index base
        assert!(matches!(
            state.nzcv,
            NzcvLocation::Deferred(LazyFlags::Add { .. })
        ));
        assert_eq!(body.exits.len(), 1);
    }
}

#[test]
fn hcq_memory_public_join_preserves_bypass_values_before_overwrite() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[0xd2800020, 0x14000004]),
        (16, &[0xd503201f, 0x14000001]),
        (24, &[0xf9400020, RET]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0, block(&graph, 24)], abi);
        let (faults, states) = faults(abi, &graph, &context, &body);
        assert!(
            body.ssa.blocks[block(&graph, 24)]
                .operands
                .contains(&GuestValue::General(0))
        );
        assert!(
            states[faults[0].state_map as usize]
                .state
                .dirty_live
                .integer
                .x[0]
        );
        assert_eq!(faults[0].instruction.block_key().pc.get(), 24);
    }
}

#[test]
fn hcq_pair_faults_keep_deferred_reads_and_partial_store_stages() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for (pair, vector, load) in [
            (0xa9400c20, false, true),
            (0xa9000c20, false, false),
            (0xad400c20, true, true),
            (0xad000c20, true, false),
        ] {
            let producers = if vector {
                [0x4f00e400, 0x4f00e403, 0x1400000e]
            } else {
                [0xd28000e0, 0xd2800103, 0x1400000e]
            };
            let graph = graph(&[(0, &producers), (64, &[pair, RET])]);
            let (context, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            let (faults, states) = faults(abi, &graph, &context, &body);
            assert_eq!(faults.len(), 2);
            for (i, fault) in faults.iter().enumerate() {
                assert_eq!(fault.instruction.block_key().pc.get(), 64);
                assert_eq!(fault.completed, 0); // neither pair element commits an instruction
                assert_eq!(fault.subaccess, i as u16);
                assert_eq!(fault.commit_stage, if load { 0 } else { i as u16 });
                assert_eq!(fault.completed_read.is_some(), load && i == 1);
                let state = &states[fault.state_map as usize].state;
                let dirty = if vector {
                    &state.dirty_live.vector[..]
                } else {
                    &state.dirty_live.integer.x[..]
                };
                assert!(dirty[0] && dirty[3]);
                assert!(state.host_fpsr_pending && state.dirty_live.fpsr);
                if let Some(location) = fault.completed_read {
                    assert!(location.valid(abi, fault.bytes));
                }
            }
        }
    }
}

#[test]
fn hcq_vector_writeback_and_cache_probe_use_shared_fault_boundaries() {
    let graph = graph(&[
        (0, &[0x4f00e400, 0x1400000f]),
        (64, &[0x3cc10420, 0x3c810c20, 0xd50b7e21, RET]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0], abi);
        let (faults, states) = faults(abi, &graph, &context, &body);
        assert_eq!(faults.len(), 3);
        for (i, fault) in faults.iter().enumerate() {
            assert_eq!(fault.instruction.block_key().pc.get(), 64 + i as u64 * 4);
            assert_eq!(fault.completed, i as u16);
            let state = &states[fault.state_map as usize].state;
            assert!(state.dirty_live.vector[0] && state.dirty_live.integer.x[1]);
        }
        assert_eq!(faults[2].access, Access::CacheProbe);
        assert_eq!(faults[2].bytes, 1);
        assert_eq!(body.exits.len(), 1);
    }
}

#[test]
fn hcq_system_read_completion_does_not_lose_the_old_destination_at_public_join() {
    // FPSR is merged outside native execution; a timer read also completes cold.
    for word in [0xd53b4420, 0xd53be020] {
        let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[word])]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            let exit = &body.exits[0];
            assert_eq!(exit.guest.pc.get(), 64);
            assert_eq!(exit.reason, NativeExitReason::Architectural);
            assert!(
                exit.state.dirty.integer.x[0],
                "cold completion has not written X0"
            );
            assert!(
                body.ssa.blocks[block(&graph, 64)]
                    .operands
                    .contains(&GuestValue::General(0))
            );
        }
    }
}

#[test]
fn hcq_system_cold_exits_preserve_pre_state_without_nominal_writes() {
    for word in [0xd51b4400, 0xd51b4420, 0xd503205f, 0xd5033bbf, 0xd5033f5f] {
        let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[word])]);
        let analysis = Analysis::build(&graph, &[0, block(&graph, 64)]);
        let last = graph.instructions.len() - 1;
        assert_eq!(
            analysis.native.instructions[last].dirty_before,
            analysis.native.instructions[last].dirty_after
        );
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            assert_eq!(body.exits.len(), 1);
            let exit = &body.exits[0];
            assert_eq!(exit.reason, NativeExitReason::Architectural);
            assert_eq!(exit.guest.pc.get(), 64);
            assert!(exit.state.dirty.integer.x[0]);
            assert_eq!(exit.state.dirty.nzcv, crate::analysis::NZCV);
            assert!(exit.state.dirty.fpsr);
            assert!(!exit.state.dirty.fpcr);
            assert!(body.faults.is_empty());
        }
    }
}

#[test]
fn hcq_atomic_and_exclusive_maps_stay_precise_across_internal_edges() {
    for words in [
        vec![0xc8a27c23, RET],             // CAS X2,X3,[X1]
        vec![0xf8220023, RET],             // LDADD X2,X3,[X1]
        vec![0xc85f7c20, 0xc803fc22, RET], // LDXR X0,[X1]; STLXR W3,X2,[X1]
    ] {
        // Dirty status/return registers must survive the CAS or slow PRE exit.
        let graph = graph(&[(0, &[0xd2800023, ADDS, 0x1400000e]), (64, &words)]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (context, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            let (faults, states) = faults(abi, &graph, &context, &body);
            assert!(!faults.is_empty());
            for fault in &faults {
                assert!(matches!(fault.instruction.block_key().pc.get(), 64 | 68));
                assert_eq!(fault.bytes, 8);
                assert!(matches!(fault.access, Access::Read | Access::Atomic));
                let state = &states[fault.state_map as usize].state;
                assert!(state.dirty_live.integer.x[3]);
                assert_eq!(state.dirty_live.nzcv, crate::analysis::NZCV);
            }
            if words.len() == 3 {
                assert_eq!(body.exits.len(), 2);
                let exit = &body.exits[0];
                assert!(matches!(exit.guest.kind, EdgeKind::ExclusiveStore(_)));
                assert_eq!(exit.guest.pc.get(), 68);
                assert!(exit.state.dirty.integer.x[3]);
            } else {
                assert_eq!(body.exits.len(), 1);
            }
        }
    }
}

#[test]
fn hcq_inline_system_values_and_flags_cross_internal_edges_without_exits() {
    let graph = graph(&[
        (0, &[ADDS, 0xd53b4201, 0xd51bd041, 0x1400000d]),
        (64, &[0xd53bd042, 0xd51b4202, 0xd53b4203, RET]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
        assert_eq!(body.exits.len(), 1);
        assert!(body.faults.is_empty());
        assert!(body.exits[0].state.dirty.tpidr_el0);
        assert_eq!(body.exits[0].state.dirty.nzcv, crate::analysis::NZCV);
        assert!(matches!(
            body.exits[0].state.flags,
            Some(LazyFlags::Packed(_))
        ));
    }
}
