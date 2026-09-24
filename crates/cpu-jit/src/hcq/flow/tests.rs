use super::*;
use crate::hcq::{
    Builder,
    tests::{key, words},
};

const RET: u32 = 0xd65f03c0;

pub(in crate::hcq) fn graph(inputs: &[(u64, &[u32])]) -> Graph {
    let mut builder = Builder::new(key(inputs[0].0));
    for &(pc, bits) in inputs {
        builder.merge(key(pc), words(pc, bits)).unwrap();
    }
    let (instructions, blocks) = builder.finish().unwrap();
    Graph {
        discovery: None,
        units: Vec::new(),
        inputs: Vec::new(),
        instructions,
        blocks,
    }
}

pub(in crate::hcq) fn block(graph: &Graph, pc: u64) -> usize {
    graph
        .blocks
        .iter()
        .position(|block| block.key == key(pc))
        .unwrap()
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
        let live = analysis.native.blocks[0].live_in;
        assert_eq!(
            if vector {
                live.vector.contains(0)
            } else {
                live.integer.x.contains(0)
            },
            preserved
        );
    }
}

#[test]
fn hcq_liveness_invalid_and_unsupported_boundaries_never_commit_nominal_writes() {
    let graph = graph(&[(0, &[0])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert!(analysis.native.instructions[0].dirty_after.is_empty());
    // Model a decoded encoding rejected before execution by its boundary.
    let mut graph = self::graph(&[(0, &[0xd2800020])]);
    graph.blocks[0].exit = Exit::Boundary(End::Unsupported);
    let analysis = Analysis::build(&graph, &[0]);
    assert!(analysis.native.instructions[0].dirty_after.is_empty());
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
