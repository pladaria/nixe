use super::*;
use crate::hcq::flow::tests::{block, graph};

const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

fn x(registers: &[usize]) -> StateSet {
    let mut values = StateSet::default();
    for &register in registers {
        values.integer.x[register] = true;
    }
    values
}

#[test]
fn native_inputs_do_not_load_clean_homes_for_full_fault_or_exit_observations() {
    for (words, expected) in [
        (vec![NOP, RET], x(&[30])),
        (vec![0xd2800020, RET], x(&[30])),    // MOVZ X0,#1
        (vec![0xf9400020, RET], x(&[1, 30])), // LDR X0,[X1]
    ] {
        let graph = graph(&[(0, &words)]);
        let analysis = Analysis::build(&graph, &[0]);
        assert_eq!(analysis.native.blocks[0].live_in, expected);
        assert_ne!(analysis.blocks[0].live_in, expected);
        assert!(!analysis.native.instructions[0].dirty_before.integer.x[0]);
        for (native, semantic) in analysis
            .native
            .instructions
            .iter()
            .zip(&analysis.instructions)
        {
            assert!(native.live_before.without(semantic.live_before).is_empty());
            assert!(native.live_after.without(semantic.live_after).is_empty());
        }
    }
}

#[test]
fn native_inputs_preserve_read_only_fast_values_until_overwrite() {
    // Read X1, fault, then overwrite X1, with another fault after replacement.
    let graph = graph(&[(0, &[0x91000420, 0xf9400043, 0xd2800021, 0xf9400043, RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(analysis.native.blocks[0].live_in, x(&[1, 2, 30]));
    let points = &analysis.native.instructions;
    assert!(points[1].dirty_before.integer.x[1]);
    assert!(
        points[1].live_before.integer.x[1],
        "the prefault map must carry inherited X1"
    );
    assert!(
        !points[2].live_before.integer.x[1],
        "MOVZ kills the inherited value"
    );
    assert!(
        points[3].live_before.integer.x[1],
        "the later fault sees the replacement"
    );
}

#[test]
fn native_inputs_public_diamond_join_keeps_the_clean_bypass_value() {
    let graph = graph(&[
        (0, &[0x54000080]),             // B.EQ 16
        (4, &[0xd2800020, 0x14000004]), // MOVZ X0,#1; B 24
        (16, &[NOP, 0x14000001]),
        (24, &[RET]),
    ]);
    let join = block(&graph, 24);
    let writer = block(&graph, 4);
    let bypass = block(&graph, 16);
    let analysis = Analysis::build(&graph, &[0, writer, join]);
    assert!(analysis.native.blocks[0].live_in.integer.x[0]);
    assert!(!analysis.native.blocks[writer].live_in.integer.x[0]);
    assert!(analysis.native.blocks[bypass].live_in.integer.x[0]);
    assert!(analysis.native.blocks[join].live_in.integer.x[0]);
    assert!(
        analysis.native.instructions[graph.blocks[join].instructions.start]
            .dirty_before
            .integer
            .x[0]
    );
    assert!(!analysis.native.blocks[0].live_in.integer.x[17]);
}

#[test]
fn native_inputs_selected_fault_entry_supplies_its_own_old_destination() {
    let graph = graph(&[(0, &[0xd2800020, 0x14000001]), (8, &[0xf9400020, RET])]);
    let load = block(&graph, 8);
    let analysis = Analysis::build(&graph, &[0, load]);
    assert_eq!(analysis.native.blocks[0].live_in, x(&[1, 30]));
    assert_eq!(analysis.native.blocks[load].live_in, x(&[0, 1, 30]));
    // The common prefault map cannot assume the producer in block 0 ran.
    assert!(
        analysis.native.instructions[graph.blocks[load].instructions.start]
            .live_before
            .integer
            .x[0]
    );
}

#[test]
fn native_inputs_reach_a_joint_fixed_point_across_independent_entry_paths() {
    let graph = graph(&[
        (0, &[0x910004a1, 0x14000007]), // ADD X1,X5,#1; B 32
        (16, &[NOP, 0x14000003]),       // B 32
        (32, &[0xf9400040, RET]),       // LDR X0,[X2]
    ]);
    let bypass = block(&graph, 16);
    let analysis = Analysis::build(&graph, &[0, bypass]);
    assert_eq!(analysis.native.blocks[0].live_in, x(&[2, 5, 30]));
    // X5 is never locally written, but reading it through fast entry 0 can
    // inherit a stale home. The common fault map must preserve it on BOTH paths.
    assert_eq!(analysis.native.blocks[bypass].live_in, x(&[1, 2, 5, 30]));
    let fault = &analysis.native.instructions[graph.blocks[block(&graph, 32)].instructions.start];
    assert!(fault.dirty_before.integer.x[5] && fault.live_before.integer.x[5]);
    assert!(!fault.live_before.integer.x[0]);
}

#[test]
fn native_inputs_do_not_mix_dirty_state_between_disconnected_entries() {
    let graph = graph(&[(0, &[0xf9400040, RET]), (16, &[0x910004a1, RET])]);
    let second = block(&graph, 16);
    let analysis = Analysis::build(&graph, &[0, second]);
    assert_eq!(analysis.native.blocks[0].live_in, x(&[2, 30]));
    assert_eq!(analysis.native.blocks[second].live_in, x(&[5, 30]));
    assert!(!analysis.native.instructions[0].dirty_before.integer.x[5]);
}

#[test]
fn native_inputs_loop_fixed_point_keeps_updates_but_not_killed_initial_values() {
    for (word, initial) in [(0xd2800020, false), (0x91000400, true)] {
        let graph = graph(&[(0, &[word, 0x17ffffff])]); // MOVZ/ADD X0; B 0
        let analysis = Analysis::build(&graph, &[0]);
        assert_eq!(analysis.native.blocks[0].live_in.integer.x[0], initial);
        assert_eq!(analysis.native.blocks[0].live_out, x(&[0]));
        assert!(analysis.native.instructions[1].live_after.integer.x[0]);
        assert!(analysis.native.instructions[1].dirty_after.integer.x[0]);
        assert!(
            analysis.native.blocks[0]
                .live_in
                .without(x(&[0]))
                .is_empty()
        );
    }
}

#[test]
fn native_inputs_partial_writes_and_flags_use_shared_bit_precise_effects() {
    let graph = graph(&[(0, &[0xf2800020, 0x4e181c20, 0xab020020, RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    let input = analysis.native.blocks[0].live_in;
    assert!(input.integer.x[0] && input.integer.x[1] && input.integer.x[2]);
    assert!(input.vector[0]);
    assert_eq!(input.nzcv, 0, "ADDS replaces all incoming flags");
    assert_eq!(
        analysis.native.instructions.last().unwrap().live_after.nzcv,
        analysis::NZCV
    );
    let graph = self::graph(&[(0, &[0x54000040]), (4, &[RET]), (8, &[RET])]);
    let analysis = Analysis::build(&graph, &[0]);
    assert_eq!(analysis.native.blocks[0].live_in.nzcv, analysis::Z);
    assert_eq!(
        analysis.native.instructions[0].dirty_before.nzcv,
        analysis::Z
    );
}

#[test]
fn native_inputs_leave_fpsr_invocation_owned_and_read_only_homes_clean() {
    let graph = graph(&[(0, &[0x1e222820, 0xd53bd060, RET])]); // FADD; MRS X0,TPIDRRO_EL0
    let analysis = Analysis::build(&graph, &[0]);
    let input = analysis.native.blocks[0].live_in;
    assert!(input.fpcr && input.tpidrro_el0);
    assert!(!input.fpsr);
    assert!(analysis.instructions[0].live_before.fpsr);
    for point in &analysis.native.instructions {
        assert!(!point.dirty_before.fpcr && !point.dirty_after.fpcr);
        assert!(!point.dirty_before.tpidrro_el0 && !point.dirty_after.tpidrro_el0);
        assert!(!point.live_before.fpsr && !point.live_after.fpsr);
    }
}
