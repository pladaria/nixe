use super::*;
use crate::abi::{CodeVersion, GuestValue};
use crate::hcq::{
    flow::tests::{block, graph},
    ssa::tests::target,
};
use crate::native::AllocatedBoundary;
use cranelift_codegen::control::ControlPlane;

const RET: u32 = 0xd65f03c0;
const ADDS: u32 = 0xb1000420;
const SUBS: u32 = 0xf1000420;

fn emitted(graph: &Graph, entries: &[usize], abi: HostAbi) -> (Context, Body) {
    let analysis = Analysis::build(graph, entries);
    let isa = target(if abi == HostAbi::X86_64 {
        "x86_64-unknown-linux-gnu"
    } else {
        "aarch64-unknown-linux-gnu"
    });
    let mut context = Context::new();
    let mut frontend = FunctionBuilderContext::new();
    let body = emit(
        abi,
        &*isa,
        Some(0x10000),
        &mut context,
        &mut frontend,
        graph,
        &analysis,
        entries,
    )
    .unwrap();
    context
        .compile(&*isa, &mut ControlPlane::default())
        .unwrap();
    let code = context.compiled_code().unwrap();
    for (index, exit) in body.exits.iter().enumerate() {
        let map = code
            .buffer
            .nixe_states
            .iter()
            .find(|map| !map.entry && map.id == index as u64 + 1)
            .unwrap();
        let allocated = AllocatedBoundary::new(abi, code, map).unwrap();
        exit.state
            .allocate(abi, CodeVersion::new(1).unwrap(), index as u32, &allocated)
            .unwrap();
    }
    memory::records(
        abi,
        CodeVersion::new(1).unwrap(),
        graph.blocks[0].key,
        code,
        &context.func,
        &body.faults,
        &mut Vec::new(),
    )
    .unwrap();
    for (index, pending) in body.fp_activations.iter().enumerate() {
        let (source, adapter, continuation, state) = pending
            .adapter(
                abi,
                code,
                crate::abi::ExitSiteKey {
                    source: CodeVersion::new(1).unwrap(),
                    state_map: (body.exits.len() + index) as u32,
                },
            )
            .unwrap();
        assert!(!adapter.is_empty());
        assert!(!source.entry);
        assert!(
            code.buffer
                .nixe_entries
                .contains(&(pending.entry, continuation))
        );
        state.state.validate().unwrap();
    }
    (context, body)
}

fn staged(graph: &Graph, entries: &[usize], abi: HostAbi, version: CodeVersion) -> stage::Staged {
    let compiler = backend::Compiler::new(abi, 0x10000).unwrap();
    let mut context = Context::new();
    let analysis = Analysis::build(graph, entries);
    let body = compiler
        .emit(
            &mut context,
            &mut FunctionBuilderContext::new(),
            graph,
            &analysis,
            entries,
        )
        .unwrap();
    assert_eq!(
        body.polls.len(),
        analysis.backedges.iter().flatten().filter(|&&v| v).count()
    );
    compiler.finish(&mut context, body, graph, version).unwrap()
}

// Exercise native fragments under real dispatch protection, without binding a
// resolver: external edges retain their production canonical fallback.
fn published(
    graph: &Graph,
    entries: &[usize],
    abi: HostAbi,
) -> (
    std::sync::Arc<crate::lifetime::Lifetime>,
    crate::lifetime::unit::Snapshot,
) {
    use crate::executable::{Cache, Tier};
    use crate::lifetime::{Lifetime, unit::Input};
    use std::sync::{Arc, atomic::AtomicU64};
    crate::native::check_host().unwrap();
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    // HCQ replaces resident LCQ images; do not bypass that production invariant.
    use nixe_cpu::memory::{MemoryPermissions, SyntheticMemory};
    use nixe_memory::{AddressSpaceId, GuestPhysicalPageId};
    let mut memory = SyntheticMemory::new();
    let base = graph.instructions[0].instruction.key.block_key().pc.get() & !4095;
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    for word in &graph.instructions {
        assert!(memory.initialize_ram(
            page,
            (word.instruction.key.block_key().pc.get() - base) as usize,
            &word.instruction.bits.to_le_bytes()
        ));
    }
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(base),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    let mut reader = process.register().unwrap();
    let mut lcq = crate::lcq::compiler::Compiler::new(abi).unwrap();
    for block in &graph.blocks {
        let crate::lifetime::compile::Request::Owner(claim) = reader.claim(block.key).unwrap()
        else {
            panic!()
        };
        lcq.publish(
            crate::lcq::Compilation::capture(claim, &memory).unwrap(),
            &process,
            &cache,
            &memory,
        )
        .unwrap();
        process.try_service_links().unwrap();
    }
    let identity = process.begin_unit(Tier::Hcq).unwrap();
    let image = staged(graph, entries, abi, identity.version());
    let reservations: Vec<_> = image
        .entries
        .iter()
        .map(|entry| process.reserve(entry.key).unwrap())
        .collect();
    let islands = image
        .states
        .iter()
        .filter(|state| {
            state
                .transfer
                .as_ref()
                .is_some_and(|t| t.static_target.is_some())
        })
        .count();
    let code = cache
        .install_with_islands(image.output, Tier::Hcq, islands, |_| None)
        .unwrap();
    let handle = process
        .prepare_unit(
            &reservations,
            Input {
                identity,
                code,
                tier: Tier::Hcq,
                instructions: graph
                    .instructions
                    .iter()
                    .map(|word| word.instruction)
                    .collect(),
                entries: image.entries,
                dependencies: Box::new([]),
                cursor: nixe_memory::MemoryInvalidationCursor::INITIAL,
                states: image.states,
                faults: image.faults,
            },
            &AtomicU64::new(0),
        )
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    let snapshot = process.snapshot(handle).unwrap();
    (process, snapshot)
}

mod accounting;
mod entries;
mod exits;
mod fp;
mod observations;
mod polling;
mod runtime;
mod staging;

#[test]
fn hcq_body_internal_edges_do_not_create_native_exit_records() {
    let graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[SUBS, RET])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0, block(&graph, 8)], abi);
        assert_eq!(body.exits.len(), 1);
        assert_eq!(body.exits[0].guest.pc.get(), 12);
        assert_eq!(body.exits[0].guest.kind, EdgeKind::Return);
        assert_eq!(
            context.compiled_code().unwrap().buffer.nixe_entries.len(),
            2
        );
    }
}

#[test]
fn hcq_body_mixed_conditional_edges_have_only_the_real_external_observations() {
    // B.EQ, CBZ W1, CBNZ X1, TBZ X1,#40 and TBNZ X1,#40, all targeting 16.
    for branch in [0x54000080, 0x34000081, 0xb5000081, 0xb6400081, 0xb7400081] {
        let graph = graph(&[(0, &[branch]), (4, &[0xd2800020, RET])]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0], abi);
            assert_eq!(body.exits.len(), 2);
            assert_eq!(body.exits[0].guest.kind, EdgeKind::Taken);
            assert_eq!(body.exits[0].static_target.unwrap().get(), 16);
            assert_eq!(body.exits[1].guest.kind, EdgeKind::Return);
        }
    }
}

#[test]
fn hcq_body_simd_partial_registers_and_overlapping_public_entries_share_lowering() {
    let graph = graph(&[(0, &[0x4e181c20, 0x14000001]), (8, &[0x4e221c20, RET])]); // INS; AND
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0, block(&graph, 8)], abi);
        assert!(body.ssa.blocks[0].operands.contains(&GuestValue::Vector(0)));
        assert!(body.exits[0].state.dirty.vector.contains(0));
        assert!(body.exits[0].state.dirty.vector.contains(1));
        assert!(body.exits[0].state.flags.is_none());
        assert!(body.exits[0].state.dirty.fpsr); // invocation-owned, not a vector input
    }
}

#[test]
fn hcq_body_packed_join_does_not_invent_dirtiness_for_dead_nzcv_bits() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[ADDS, 0x14000004]),
        (16, &[SUBS, 0x14000001]),
        (24, &[0x9a820020, ADDS, RET]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, body) = emitted(&graph, &[0], abi);
        assert_eq!(
            body.ssa.blocks[block(&graph, 24)].flag_mask,
            crate::analysis::Z
        );
        assert_eq!(body.exits[0].state.dirty.nzcv, crate::analysis::NZCV); // actual ADDS writes all
    }
}

#[test]
fn hcq_body_read_only_packed_inputs_keep_only_inherited_dirty_bits() {
    for (word, mask) in [
        (0x9a020020, crate::analysis::C),
        (0x9a820020, crate::analysis::Z),
    ] {
        let graph = graph(&[(0, &[word, RET])]); // ADC or CSEL EQ, neither writes NZCV
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0], abi);
            assert_eq!(body.exits[0].state.dirty.nzcv, mask);
            assert!(matches!(
                body.exits[0].state.flags,
                Some(LazyFlags::Packed(_))
            ));
        }
    }
}

#[test]
fn hcq_body_calls_and_architectural_exits_keep_shared_lr_and_source_contracts() {
    for (words, kind, dirty_lr) in [
        (vec![0x94000004], EdgeKind::Call, true),
        (vec![0xd63f03c0], EdgeKind::Call, true), // BLR X30: target must precede LR write
        (vec![0xd4000001], EdgeKind::SupervisorCall(0), false),
        (vec![0xd4200000], EdgeKind::Breakpoint(0), false),
    ] {
        let graph = graph(&[(0, &words)]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let (_, body) = emitted(&graph, &[0], abi);
            assert_eq!(body.exits[0].guest.kind, kind);
            assert_eq!(body.exits[0].state.dirty.integer.x.contains(30), dirty_lr);
            assert_eq!(body.exits[0].guest.pc.get(), 0);
        }
    }
}

#[test]
fn hcq_body_failed_memory_lowering_does_not_leave_a_partial_function() {
    let isa = target("x86_64-unknown-linux-gnu");
    let mut context = Context::new();
    let mut frontend = FunctionBuilderContext::new();
    for arena in [None, Some(0), Some(1)] {
        let graph = graph(&[(0, &[0xf9400020, RET])]);
        let analysis = Analysis::build(&graph, &[0]);
        assert!(
            emit(
                HostAbi::X86_64,
                &*isa,
                arena,
                &mut context,
                &mut frontend,
                &graph,
                &analysis,
                &[0]
            )
            .is_err()
        );
        assert!(context.func.layout.entry_block().is_none());
    }
    let graph = graph(&[(0, &[ADDS, RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert!(
        emit(
            HostAbi::X86_64,
            &*isa,
            None,
            &mut context,
            &mut frontend,
            &graph,
            &analysis,
            &[0]
        )
        .is_ok()
    );
}
