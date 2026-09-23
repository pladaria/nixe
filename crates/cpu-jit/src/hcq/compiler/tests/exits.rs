use super::*;
use crate::abi::{ExitSiteKey, NzcvLocation};
use crate::frontend::exit;

#[test]
fn subtraction_terminals_export_host_flags_without_a_result_recipe() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for cmp in [0x6b01_001f, 0xeb01_001f] {
            let graph = graph(&[(0, &[cmp, 0x1400_0010])]);
            let (context, body) = emitted(&graph, &[0], abi);
            let code = context.compiled_code().unwrap();
            assert_eq!(body.exits.len(), 1);
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| map.id == 1 && !map.entry)
                .unwrap();
            assert!(map.subtract_flags);
            assert!(body.exits[0].state.flags.is_none());
            let allocated = crate::native::AllocatedBoundary::new(abi, code, map).unwrap();
            let state = body.exits[0]
                .state
                .allocate(abi, CodeVersion::new(1).unwrap(), 0, &allocated)
                .unwrap();
            assert_eq!(
                state.nzcv,
                NzcvLocation::Host {
                    carry_inverted: abi == HostAbi::X86_64
                }
            );
            assert!(
                code.buffer
                    .nixe_states
                    .iter()
                    .filter(|map| map.entry)
                    .all(|map| !map.subtract_flags)
            );
            assert!(
                code.buffer
                    .nixe_faults
                    .iter()
                    .all(|map| !map.subtract_flags)
            );
        }
    }
}

// One entry, one finite path: ADDS; B 64; terminal. Three instructions really
// execute, despite the noncontiguous PCs. The root is charged in the body;
// terminal checkpoints own only the remaining source-local work.
pub(super) fn terminal(abi: HostAbi, word: u32, charged: bool) -> (Graph, Context, Body) {
    let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[word])]);
    let analysis = Analysis::build(&graph, &[0]);
    let isa = target(if abi == HostAbi::X86_64 {
        "x86_64-unknown-linux-gnu"
    } else {
        "aarch64-unknown-linux-gnu"
    });
    let mut context = Context::new();
    let body = emit(
        abi,
        &*isa,
        None,
        &mut context,
        &mut FunctionBuilderContext::new(),
        &graph,
        &analysis,
        &[0],
    )
    .unwrap();
    assert_eq!(body.exits.len(), 1);
    if charged {
        context
            .func
            .nixe_exit_costs
            .insert(1, body.exits[0].completed);
    }
    context
        .compile(&*isa, &mut ControlPlane::default())
        .unwrap();
    (graph, context, body)
}

fn site() -> ExitSiteKey {
    ExitSiteKey {
        source: CodeVersion::new(1).unwrap(),
        state_map: 0,
    }
}

#[test]
fn hcq_external_edges_share_charged_static_pic_and_rsb_adapters() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for (word, kind, target_pc) in [
            (0x14000004, EdgeKind::Static, Some(80)), // B 80
            (0x94000004, EdgeKind::Call, Some(80)),   // BL 80
            (0xd61f0020, EdgeKind::Indirect, None),   // BR X1
            (0xd63f0020, EdgeKind::Call, None),       // BLR X1
            (RET, EdgeKind::Return, None),
        ] {
            let (graph, mut context, body) = terminal(abi, word, true);
            let code = context.take_compiled_code().unwrap();
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| map.id == 1)
                .unwrap();
            let poll_offset = map.poll.unwrap().offset;
            let body_bytes = code.code_buffer().len();
            let staged = body
                .stage(abi, code, &context.func, &graph, site().source)
                .unwrap();
            let record = &staged.states[0];
            record.state.validate().unwrap();
            let source = record.exit.unwrap();
            assert_eq!(source.pc.get(), 64);
            assert_eq!(source.kind, kind);
            let transfer = record.transfer.as_ref().unwrap();
            assert_eq!(
                transfer.completed,
                if kind == EdgeKind::Static { 0 } else { 1 }
            );
            assert_eq!(transfer.static_target.map(|key| key.pc.get()), target_pc);
            assert_eq!(transfer.poll_offset, Some(poll_offset));
            assert!(transfer.fallback_offset as usize >= body_bytes);
            assert!((transfer.fallback_offset as usize) < staged.output.bytes.len());
            assert_eq!(record.state.site, site());
        }
    }
}

#[test]
fn hcq_indirect_adapters_reject_missing_terminal_charge() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for word in [0xd61f0020, 0xd63f0020, RET] {
            let (graph, context, body) = terminal(abi, word, false);
            let code = context.compiled_code().unwrap();
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| map.id == 1)
                .unwrap();
            let allocated = AllocatedBoundary::new(abi, code, map).unwrap();
            let error = exit::prepare(
                abi,
                graph.blocks[0].key,
                &allocated,
                &body.exits[0],
                site(),
                body.exits[0].completed,
            )
            .err()
            .expect("uncharged indirect probe must not be installed");
            assert!(error.to_string().contains("no charged terminal checkpoint"));
        }
    }
}

#[test]
fn hcq_exit_adapters_reject_cost_disagreement_with_backend_checkpoint() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (graph, context, body) = terminal(abi, RET, true);
        let code = context.compiled_code().unwrap();
        let map = code
            .buffer
            .nixe_states
            .iter()
            .find(|map| map.id == 1)
            .unwrap();
        let allocated = AllocatedBoundary::new(abi, code, map).unwrap();
        let error = exit::prepare(
            abi,
            graph.blocks[0].key,
            &allocated,
            &body.exits[0],
            site(),
            17,
        )
        .err()
        .expect("PC distance is not executed work");
        assert!(error.to_string().contains("cost/shape mismatch"));
    }
}
