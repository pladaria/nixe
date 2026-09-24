use super::*;
use crate::hcq::flow::tests::{block, graph};

const FADD: u32 = 0x1e222820;
const RET: u32 = 0xd65f03c0;
const NOP: u32 = 0xd503201f;

fn point<'a>(graph: &Graph, analysis: &'a Analysis, pc: u64) -> &'a FpPoint {
    let ordinal = graph
        .instructions
        .iter()
        .position(|word| word.instruction.key.block_key().pc.get() == pc)
        .unwrap();
    &analysis.fp.instructions[ordinal]
}

#[test]
fn fp_flow_activates_only_the_first_native_operation_on_a_straight_path() {
    let graph = graph(&[(0, &[FADD, FADD, RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(
        *point(&graph, &analysis, 0),
        FpPoint {
            active_before: false,
            activate: true,
            active_after: true
        }
    );
    assert_eq!(
        *point(&graph, &analysis, 4),
        FpPoint {
            active_before: true,
            activate: false,
            active_after: true
        }
    );
    assert!(!point(&graph, &analysis, 8).activate);
}

#[test]
fn fp_flow_mixed_diamond_requires_activation_but_two_active_paths_do_not() {
    for right in [NOP, FADD] {
        let graph = graph(&[
            (0, &[0x54000080]),       // B.EQ 16
            (4, &[FADD, 0x14000004]), // B 24
            (16, &[right, 0x14000001]),
            (24, &[FADD, RET]),
        ]);
        let analysis = Analysis::build(&graph, &[0]);
        assert_eq!(point(&graph, &analysis, 24).activate, right == NOP);
        assert_eq!(point(&graph, &analysis, 24).active_before, right == FADD);
        assert!(point(&graph, &analysis, 24).active_after);
    }
}

#[test]
fn fp_flow_public_join_cannot_assume_its_internal_predecessor_activated_fp() {
    let graph = graph(&[(0, &[FADD, 0x14000001]), (8, &[FADD, RET])]);
    let internal = Analysis::build(&graph, &[0]);
    assert!(!point(&graph, &internal, 8).activate);
    let public = Analysis::build(&graph, &[0, block(&graph, 8)]);
    assert!(point(&graph, &public, 8).activate);
    assert!(!point(&graph, &public, 8).active_before);
}

#[test]
fn fp_flow_loop_header_keeps_the_first_visit_check_and_the_body_inherits_it() {
    let graph = graph(&[(0, &[FADD, 0x14000001]), (8, &[FADD, 0x17fffffd])]); // B 0
    let analysis = Analysis::build(&graph, &[0]);
    assert!(point(&graph, &analysis, 0).activate);
    assert!(!point(&graph, &analysis, 8).activate);
    assert!(point(&graph, &analysis, 8).active_before);
    assert_eq!(analysis.backedges[block(&graph, 8)], [true, false]);
    let public = Analysis::build(&graph, &[0, block(&graph, 8)]);
    assert!(point(&graph, &public, 8).activate);
}

#[test]
fn fp_flow_entry_path_into_irreducible_loop_is_not_hidden_by_an_active_backedge() {
    let graph = graph(&[
        (0, &[0x54000080]),        // B.EQ 16
        (4, &[FADD, 0x14000002]),  // B 16
        (16, &[0x54000020]),       // B.EQ 20, fallthrough also 20
        (20, &[FADD, 0x17fffffb]), // B 4
    ]);
    let analysis = Analysis::build(&graph, &[0]);
    assert!(!point(&graph, &analysis, 16).active_before);
    assert!(point(&graph, &analysis, 20).activate);
    assert!(point(&graph, &analysis, 4).activate);
}

#[test]
fn fp_flow_comparison_and_simd_bits_do_not_establish_native_fp_ownership() {
    // FCMP S1,S2 uses the guarded exact policy, AND uses only SIMD bits.
    for word in [0x1e222020, 0x4e221c20] {
        let graph = graph(&[(0, &[word, FADD, RET])]);
        let analysis = Analysis::build(&graph, &[0]);
        assert_eq!(*point(&graph, &analysis, 0), FpPoint::default());
        assert!(point(&graph, &analysis, 4).activate);
    }
}

#[test]
fn fp_flow_disconnected_entry_and_unrooted_cycle_do_not_inherit_other_activation() {
    let graph = graph(&[(0, &[FADD, RET]), (16, &[FADD, 0x17ffffff])]); // B 16
    for entries in [vec![0], vec![0, block(&graph, 16)]] {
        let analysis = Analysis::build(&graph, &entries);
        assert!(point(&graph, &analysis, 16).activate);
        assert!(!point(&graph, &analysis, 16).active_before);
    }
}

#[test]
fn fp_flow_rejected_encoding_does_not_activate_its_nominal_fp_operation() {
    let mut graph = graph(&[(0, &[FADD])]);
    graph.blocks[0].exit = Exit::Boundary(End::Unsupported);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(analysis.fp.instructions[0], FpPoint::default());
}
