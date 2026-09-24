use super::*;
use crate::hcq::flow::tests::{block, graph};

const ADDS: u32 = 0xb1000420;
const SUBS: u32 = 0xf1000420;
const RET: u32 = 0xd65f03c0;
const NOP: u32 = 0xd503201f;

fn shape(word: u32) -> LazyFlags<()> {
    let graph = graph(&[(0, &[word, RET])]);
    Analysis::build(&graph, &[0]).flags.blocks[0]
        .output
        .clone()
        .unwrap()
}

fn diamond(left: u32, right: u32) -> Graph {
    graph(&[
        (0, &[0x54000080]),       // B.EQ 16
        (4, &[left, 0x14000004]), // B 24
        (16, &[right, 0x14000001]),
        (24, &[RET]),
    ])
}

#[test]
fn flag_flow_same_recipe_carries_operands_even_when_producers_use_different_registers() {
    // ADDS immediate and ADDS shifted register produce the same 64-bit recipe.
    let graph = diamond(ADDS, 0xab0700c5);
    let analysis = Analysis::build(&graph, &[0]);
    let join = block(&graph, 24);
    assert_eq!(analysis.flags.blocks[join].input, Some(shape(ADDS)));
    for source in [4, 16] {
        assert_eq!(
            analysis.flags.blocks[block(&graph, source)].output,
            analysis.flags.blocks[join].input
        );
    }
}

#[test]
fn flag_flow_different_kind_width_or_carry_requires_explicit_ssa_merge() {
    for other in [SUBS, 0x31000420, 0xba020020, 0xfa020020, 0xea0703e5] {
        let graph = diamond(ADDS, other);
        let analysis = Analysis::build(&graph, &[0]);
        let join = block(&graph, 24);
        assert_eq!(
            analysis.flags.blocks[join].input,
            Some(LazyFlags::Packed(()))
        );
        for source in [4, 16] {
            assert_ne!(
                analysis.flags.blocks[block(&graph, source)].output,
                analysis.flags.blocks[join].input
            );
        }
    }
}

#[test]
fn flag_flow_carry_and_conditional_recipes_keep_captured_operands_and_literals() {
    for word in [0xba020020, 0xfa020020, 0xfa42102a] {
        let graph = diamond(word, word);
        let analysis = Analysis::build(&graph, &[0]);
        let join = block(&graph, 24);
        let input = analysis.flags.blocks[join].input.as_ref().unwrap();
        assert_eq!(input, &shape(word));
        let mut count = 0;
        let rebound = input
            .try_map(&mut |_| {
                count += 1;
                Ok::<_, ()>(count)
            })
            .unwrap();
        assert_eq!(count, 4); // lhs/rhs/result + carry or predicate
        assert_eq!(rebound.shape(), *input);
        assert_eq!(
            analysis.flags.blocks[block(&graph, 4)].output,
            analysis.flags.blocks[join].input
        );
    }
    // Different predicates remain operand values; a different literal changes shape.
    for (other, packed) in [(0xfa42002a, false), (0xfa42102b, true)] {
        let graph = diamond(0xfa42102a, other);
        let analysis = Analysis::build(&graph, &[0]);
        assert_eq!(
            analysis.flags.blocks[block(&graph, 24)].input == Some(LazyFlags::Packed(())),
            packed
        );
    }
}

#[test]
fn flag_flow_public_join_cannot_inherit_a_private_predecessor_recipe() {
    let graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[RET])]);
    let join = block(&graph, 8);
    let internal = Analysis::build(&graph, &[0]);
    assert_eq!(internal.flags.blocks[join].input, Some(shape(ADDS)));
    let public = Analysis::build(&graph, &[0, join]);
    assert_eq!(public.flags.blocks[join].input, Some(LazyFlags::Packed(())));
    assert_eq!(
        LazyFlags::Canonical(42).shape(),
        LazyFlags::Packed(7).shape()
    );
}

#[test]
fn flag_flow_bypass_packed_and_fp_or_system_definitions_mix_without_home_roundtrip() {
    for other in [NOP, 0x1e222020, 0xd51b4206] {
        let graph = diamond(ADDS, other);
        let analysis = Analysis::build(&graph, &[0]);
        let join = block(&graph, 24);
        assert_eq!(
            analysis.flags.blocks[block(&graph, 4)].output,
            Some(shape(ADDS))
        );
        assert_eq!(
            analysis.flags.blocks[block(&graph, 16)].output,
            analysis.flags.blocks[join].input
        );
        assert_eq!(
            analysis.flags.blocks[join].input,
            Some(LazyFlags::Packed(()))
        );
    }
}

#[test]
fn flag_flow_loop_fixed_point_preserves_compatible_recipes_and_handles_conflicts() {
    for body in [NOP, ADDS, SUBS] {
        let graph = graph(&[
            (0, &[ADDS, 0x14000001]),
            (8, &[0x54000040]), // B.EQ 16; fallthrough 12
            (12, &[RET]),
            (16, &[body, 0x17fffffd]), // B 8
        ]);
        let analysis = Analysis::build(&graph, &[0]);
        let header = block(&graph, 8);
        assert_eq!(
            analysis.flags.blocks[header].input,
            Some(if body == SUBS {
                LazyFlags::Packed(())
            } else {
                shape(ADDS)
            })
        );
        let public = Analysis::build(&graph, &[0, header]);
        assert_eq!(
            public.flags.blocks[header].input,
            Some(LazyFlags::Packed(()))
        );
    }
}

#[test]
fn flag_flow_dead_inputs_do_not_create_parameters_or_pack_edges() {
    let graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[SUBS, RET])]);
    let analysis = Analysis::build(&graph, &[0, block(&graph, 8)]);
    assert_eq!(analysis.flags.blocks[0], FlagBlock::default());
    assert_eq!(analysis.flags.blocks[block(&graph, 8)].input, None);
}

#[test]
fn flag_flow_fault_keeps_old_recipe_and_rejected_producer_does_not_replace_it() {
    let mut graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[SUBS])]);
    let rejected = block(&graph, 8);
    graph.blocks[rejected].exit = Exit::Boundary(End::Unsupported);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(
        analysis.flags.blocks[block(&graph, 8)].input,
        Some(shape(ADDS))
    );
    assert_eq!(
        analysis.flags.blocks[block(&graph, 8)].output,
        Some(shape(ADDS))
    );
    let graph = self::graph(&[(0, &[ADDS, 0x14000001]), (8, &[0xf9400020, SUBS, RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(
        analysis.flags.blocks[block(&graph, 8)].input,
        Some(shape(ADDS))
    );
    assert_eq!(
        analysis.flags.blocks[block(&graph, 8)].output,
        Some(shape(SUBS))
    );
}

#[test]
fn flag_flow_explicit_merge_retains_bit_precise_demand_and_disconnected_contracts() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[ADDS, 0x14000004]),
        (16, &[SUBS, 0x14000001]),
        (24, &[0x9a820020, ADDS, RET]), // CSEL X0,X1,X2,EQ; overwrite flags
        (48, &[0xba020020, RET]),       // independent entry needs carry only
    ]);
    let join = block(&graph, 24);
    let independent = block(&graph, 48);
    let analysis = Analysis::build(&graph, &[0, independent]);
    assert_eq!(analysis.native.blocks[join].live_in.nzcv, analysis::Z);
    assert_eq!(
        analysis.flags.blocks[join].input,
        Some(LazyFlags::Packed(()))
    );
    assert_eq!(
        analysis.native.blocks[independent].live_in.nzcv,
        analysis::C
    );
    assert_eq!(
        analysis.flags.blocks[independent].input,
        Some(LazyFlags::Packed(()))
    );
}

#[test]
fn flag_flow_irreducible_and_unrooted_cycles_have_stable_total_contracts() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[ADDS, 0x14000002]), // B 16
        (16, &[0x54000020]),
        (20, &[SUBS, 0x17fffffb]), // B 4
        (40, &[0x54ffffe0]),       // B.EQ 36, external fallthrough
        (36, &[0x14000001]),       // B 40, unreachable SCC
    ]);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(
        analysis.flags.blocks[block(&graph, 16)].input,
        Some(LazyFlags::Packed(()))
    );
    assert_eq!(
        analysis.flags.blocks[block(&graph, 36)].input,
        Some(LazyFlags::Packed(()))
    );
    for (index, node) in graph.blocks.iter().enumerate() {
        for target in successors(&node.exit).iter().flatten() {
            if let Target::Internal(target) = target {
                let source = &analysis.flags.blocks[index].output;
                let destination = &analysis.flags.blocks[*target].input;
                assert!(
                    destination.is_none()
                        || source == destination
                        || (source.is_some() && *destination == Some(LazyFlags::Packed(())))
                );
            }
        }
    }
}
