use super::*;
use crate::hcq::{
    Builder,
    tests::{key, words},
};

const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

pub(super) fn graph(inputs: &[(u64, &[u32])]) -> Graph {
    let mut builder = Builder::new(key(inputs[0].0));
    for &(pc, bits) in inputs {
        builder.merge(key(pc), &words(pc, bits)).unwrap();
    }
    let (instructions, blocks) = builder.finish().unwrap();
    Graph {
        units: Vec::new(),
        inputs: Vec::new(),
        instructions,
        blocks,
    }
}

pub(super) fn block(graph: &Graph, pc: u64) -> usize {
    graph
        .blocks
        .iter()
        .position(|block| block.key == key(pc))
        .unwrap()
}

fn point<'a>(graph: &Graph, analysis: &'a Analysis, pc: u64) -> &'a Point {
    &analysis.instructions[graph
        .instructions
        .iter()
        .position(|word| word.instruction.key.block_key() == key(pc))
        .unwrap()]
}

#[test]
fn hcq_liveness_diamond_preserves_bypass_values_at_a_public_join() {
    let graph = graph(&[
        (0, &[0x54000080]),             // B.EQ 16
        (4, &[0xd2800020, 0x14000004]), // MOVZ X0,#1; B 24
        (16, &[NOP, 0x14000001]),       // B 24
        (24, &[RET]),
    ]);
    let join = block(&graph, 24);
    let analysis = Analysis::build(&graph, &[0, join]);
    assert!(analysis.blocks[0].live_in.integer.x[0]);
    assert!(!analysis.blocks[block(&graph, 4)].live_in.integer.x[0]);
    assert!(analysis.blocks[join].live_in.integer.x[0]);
    assert!(analysis.backedges.iter().flatten().all(|&edge| !edge));
}

#[test]
fn hcq_liveness_fault_observes_old_destination_and_keeps_earlier_producers() {
    let graph = graph(&[
        (0, &[0xd2800020, 0x14000001]), // MOVZ X0,#1; B 8
        (8, &[0xf9400020, RET]),        // LDR X0,[X1]
    ]);
    let analysis = Analysis::build(&graph, &[0, block(&graph, 8)]);
    let load = point(&graph, &analysis, 8);
    assert_eq!(load.live_before, StateSet::ALL);
    assert!(!analysis.blocks[0].live_in.integer.x[0]);
    assert!(analysis.blocks[block(&graph, 8)].live_in.integer.x[0]);
    let graph = self::graph(&[(0, &[0xf9400020, RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert!(!analysis.native.instructions[0].dirty_before.integer.x[0]);
    assert!(analysis.native.instructions[0].dirty_after.integer.x[0]);
}

#[test]
fn hcq_liveness_partial_integer_and_vector_writes_keep_preserved_parts() {
    for (bits, vector, preserved) in [
        (0xf2800020, false, true),  // MOVK X0
        (0x52800020, false, false), // MOVZ W0 kills all X0
        (0x4e181c20, true, true),   // INS V0.D[1],X1
        (0x1e270020, true, false),  // FMOV S0,W1 kills upper V0
    ] {
        let graph = graph(&[(0, &[bits, RET])]);
        let analysis = Analysis::build(&graph, &[0]);
        let live = analysis.blocks[0].live_in;
        assert_eq!(
            if vector {
                live.vector[0]
            } else {
                live.integer.x[0]
            },
            preserved
        );
    }
}

#[test]
fn hcq_liveness_flags_and_fp_status_are_observable_not_discarded_register_results() {
    let graph = graph(&[
        (0, &[0xab020020, 0x14000001]), // ADDS X0,X1,X2; B 8
        (8, &[0x1e222820, RET]),        // FADD S0,S1,S2; can trap PRE
    ]);
    let analysis = Analysis::build(&graph, &[0, block(&graph, 8)]);
    assert_eq!(analysis.blocks[0].live_in.nzcv, 0);
    assert_eq!(
        analysis.blocks[block(&graph, 8)].live_in.nzcv,
        analysis::NZCV
    );
    let fp = point(&graph, &analysis, 8);
    assert!(fp.live_before.fpcr && fp.live_before.fpsr);
    assert!(fp.live_after.fpsr);
    let graph = self::graph(&[(0, &[0xd51b4400])]); // MSR FPCR,X0; POST boundary
    let analysis = Analysis::build(&graph, &[0]);
    assert!(!analysis.blocks[0].live_in.fpcr);
    assert!(analysis.instructions[0].live_after.fpcr);
}

#[test]
fn hcq_liveness_invalid_and_unsupported_boundaries_never_commit_nominal_writes() {
    let graph = graph(&[(0, &[0])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(analysis.instructions[0].live_before, StateSet::ALL);
    assert!(analysis.native.instructions[0].dirty_after.is_empty());
    // Model a decoded encoding rejected before execution by its boundary.
    let mut graph = self::graph(&[(0, &[0xd2800020])]);
    graph.blocks[0].exit = Exit::Boundary(End::Unsupported);
    let analysis = Analysis::build(&graph, &[0]);
    assert!(analysis.instructions[0].live_before.integer.x[0]);
    assert!(analysis.native.instructions[0].dirty_after.is_empty());
}

#[test]
fn hcq_liveness_closed_loop_keeps_post_values_for_its_control_poll() {
    let graph = graph(&[(0, &[0xd2800020, 0x17ffffff])]); // MOVZ X0,#1; B 0
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(analysis.backedges, vec![[true, false]]);
    assert!(!analysis.blocks[0].live_in.integer.x[0]);
    assert_eq!(analysis.blocks[0].live_out, StateSet::ALL);
    assert!(point(&graph, &analysis, 4).live_before.integer.x[0]);
    assert!(analysis.native.instructions[0].dirty_before.integer.x[0]);
}

#[test]
fn hcq_cycle_checks_cover_irreducible_components_without_marking_forward_edges() {
    let targets = [
        [Some(Target::Internal(1)), Some(Target::Internal(2))],
        [Some(Target::Internal(2)), None],
        [Some(Target::Internal(1)), Some(Target::Internal(3))],
        [Some(Target::Internal(2)), None],
        [Some(Target::Internal(4)), None], // disconnected sampled loop
    ];
    let checks = backedges(&targets, &[0, 4]);
    assert_eq!(
        checks,
        vec![
            [false, false],
            [false, false],
            [true, false],
            [true, false],
            [true, false]
        ]
    );
    // Removing all marked edges must leave a DAG (Kahn's algorithm).
    let mut incoming = vec![0; targets.len()];
    for (node, edges) in targets.iter().enumerate() {
        for (ordinal, edge) in edges.iter().enumerate() {
            if let Some(Target::Internal(target)) = edge
                && !checks[node][ordinal]
            {
                incoming[*target] += 1;
            }
        }
    }
    let mut ready: Vec<_> = incoming
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (n == 0).then_some(i))
        .collect();
    let mut visited = 0;
    while let Some(node) = ready.pop() {
        visited += 1;
        for (ordinal, edge) in targets[node].iter().enumerate() {
            if let Some(Target::Internal(target)) = edge
                && !checks[node][ordinal]
            {
                incoming[*target] -= 1;
                if incoming[*target] == 0 {
                    ready.push(*target);
                }
            }
        }
    }
    assert_eq!(visited, targets.len());
    assert_eq!(
        backedges(&[[Some(Target::Internal(1)), None], [None, None]], &[0]),
        vec![[false; 2]; 2]
    );
}
