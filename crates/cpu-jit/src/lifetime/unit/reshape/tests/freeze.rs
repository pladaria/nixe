use super::discovery::{NOP, RET, owned, owned_entries, reshape};
use super::*;
use crate::hcq::Graph;
use crate::lifetime::background::{Frozen, workers::CompileError};
use crate::lifetime::unit::dynamic::pic::tests::cache;
use crate::lifetime::unit::{dynamic, links, tests::publish_words};

fn entries(frozen: &Frozen<'_, '_>) -> Vec<u64> {
    frozen
        .entries()
        .iter()
        .map(|&index| frozen.graph().blocks[index].key.pc.get())
        .collect()
}

#[test]
fn reshape_freeze_exports_root_and_required_target_without_exporting_coverage() {
    for owners in 0..=2 {
        let process = process();
        publish_words(&process, 0, &[NOP, 0x14000003]); // B 16.
        publish_words(&process, 16, &[NOP, NOP, RET]);
        publish_words(&process, 20, &[NOP, RET]); // Demanded, but no external root.
        if owners >= 1 {
            owned(&process, &[(16, NOP), (20, NOP), (24, RET)]);
        }
        if owners == 2 {
            owned(&process, &[(0, NOP), (4, 0x14000003)]);
        }
        let work = reshape(&process, 0, 4, 16);
        let slots = process.lock().keys.len();
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(entries(&frozen), [0, 16]);
        assert_eq!(frozen.graph().instructions.len(), 5);
        assert_eq!(process.lock().keys.len(), slots);
        assert!(!process.lock().keys.contains_key(&key(4)));
        assert!(!process.lock().keys.contains_key(&key(24)));
        frozen.check().unwrap();
        let analysis = frozen.analyze().unwrap();
        assert_eq!(analysis.native.instructions.len(), 5);
    }
}

#[test]
fn reshape_freeze_exports_a_same_family_demanded_interior_entry() {
    let process = process();
    publish_words(&process, 0, &[0xd61f0000]); // BR X0 observed to 16.
    publish_words(&process, 16, &[RET]);
    owned(&process, &[(0, 0xd61f0000), (16, RET)]);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(entries(&frozen), [0, 16]);
    // The old family exports only 0: this changes entries, not membership.
    assert!(!frozen.unchanged());
    frozen.analyze().unwrap();
}

#[test]
fn reshape_freeze_includes_external_static_roots() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    publish_words(&process, 20, &[RET]);
    owned(&process, &[(16, NOP), (20, RET)]);
    links::tests::source(&process, &AtomicU64::new(0), 128, 20);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(entries(&frozen), [0, 16, 20]);
}

#[test]
fn reshape_freeze_finds_indirect_and_return_roots_on_either_tier() {
    for optimized in [false, true] {
        for kind in [EdgeKind::Indirect, EdgeKind::Return] {
            let process = process();
            publish_words(&process, 0, &[0x14000004]);
            publish_words(&process, 16, &[NOP, NOP, RET]);
            publish_words(&process, 20, &[NOP, RET]);
            publish_words(&process, 24, &[RET]);
            if optimized {
                owned_entries(&process, &[(16, NOP), (20, NOP), (24, RET)], 2);
            }
            let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, kind);
            let mut reader = process.register().unwrap();
            cache(
                &mut reader,
                process
                    .prepare_dynamic_bridge(source, 0, key(20))
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
            let work = reshape(&process, 0, 0, 16);
            let frozen = work
                .reserve_candidate(Graph::discover(&work).unwrap())
                .unwrap()
                .freeze()
                .unwrap();
            // Unit adjacency must not export 24 merely because it shares HCQ
            // storage with the actual PIC destination 20.
            assert_eq!(entries(&frozen), [0, 16, 20]);
            frozen.check().unwrap();
        }
    }
}

#[test]
fn reshape_freeze_cancels_an_uncaptured_late_external_entry() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    owned(&process, &[(16, NOP), (20, RET)]);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let work = reshape(&process, 0, 0, 16);
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    publish_words(&process, 20, &[RET]);
    cache(
        &mut reader,
        process
            .prepare_dynamic_bridge(source, 0, key(20))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    work.check().unwrap();
    assert!(matches!(candidate.freeze(), Err(CompileError::Cancelled)));
    // Cancellation releases the whole candidate, but not its valid family job.
    work.reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap()
        .check()
        .unwrap();
}

#[test]
fn reshape_freeze_does_not_change_entries_for_a_later_pic_root() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    publish_words(&process, 20, &[RET]);
    owned(&process, &[(16, NOP), (20, RET)]);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Return);
    let mut reader = process.register().unwrap();
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(entries(&frozen), [0, 16]);
    cache(
        &mut reader,
        process
            .prepare_dynamic_bridge(source, 0, key(20))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    frozen.check().unwrap();
    assert_eq!(entries(&frozen), [0, 16]);
    // The newly observed PIC root stays on LCQ: labels are frozen before analysis.
}

#[test]
fn reshape_freeze_identifies_unchanged_sets_independent_of_root_order() {
    for root in [0, 16] {
        let process = process();
        publish_words(&process, 0, &[0x14000004]); // B 16.
        publish_words(&process, 16, &[0x17fffffc]); // B 0.
        owned_entries(&process, &[(0, 0x14000004), (16, 0x17fffffc)], 2);
        let work = reshape(&process, root, root, 16 - root);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(entries(&frozen), [0, 16]);
        assert!(frozen.unchanged());
        frozen.validate_unchanged_locked(&process.lock()).unwrap();
        assert!(matches!(frozen.analyze(), Err(CompileError::Deferred)));
        frozen.check().unwrap();
    }
}

#[test]
fn no_op_evidence_rechecks_late_static_and_pic_entries_without_changing_frozen_labels() {
    for static_root in [false, true] {
        let process = process();
        publish_words(&process, 0, &[0x14000004]);
        publish_words(&process, 16, &[NOP, RET]);
        publish_words(&process, 20, &[RET]);
        owned_entries(&process, &[(0, 0x14000004), (16, NOP), (20, RET)], 2);
        let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Return);
        let mut reader = process.register().unwrap();
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(frozen.unchanged());
        frozen.validate_unchanged_locked(&process.lock()).unwrap();
        if static_root {
            links::tests::source(&process, &AtomicU64::new(0), 256, 20);
        } else {
            cache(
                &mut reader,
                process
                    .prepare_dynamic_bridge(source, 0, key(20))
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
        }
        // Neither an unrelated static-source publication nor a PIC insertion
        // invalidates the captured code. They change only the no-op entry proof.
        frozen.check().unwrap();
        assert_eq!(entries(&frozen), [0, 16]);
        assert_eq!(
            frozen.validate_unchanged_locked(&process.lock()),
            Err(Error::StalePublication)
        );
    }
}

#[test]
fn no_op_evidence_keeps_a_published_entry_after_its_pic_root_is_removed() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    publish_words(&process, 20, &[RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, NOP), (20, RET)], 3);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    cache(
        &mut reader,
        process
            .prepare_dynamic_bridge(source, 0, key(20))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(frozen.unchanged());
    assert_eq!(entries(&frozen), [0, 16, 20]);
    frozen.validate_unchanged_locked(&process.lock()).unwrap();
    drop(reader);
    frozen.check().unwrap();
    frozen.validate_unchanged_locked(&process.lock()).unwrap();
}

#[test]
fn no_op_evidence_detects_new_demand_and_root_at_a_previously_uncaptured_interior_pc() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, NOP), (20, RET)], 2);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    frozen.validate_unchanged_locked(&process.lock()).unwrap();
    assert!(
        !frozen
            .graph()
            .inputs
            .iter()
            .any(|input| input.key == key(20))
    );
    publish_words(&process, 20, &[RET]);
    cache(
        &mut reader,
        process
            .prepare_dynamic_bridge(source, 0, key(20))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    frozen.check().unwrap();
    assert_eq!(
        frozen.validate_unchanged_locked(&process.lock()),
        Err(Error::StalePublication)
    );
}

#[test]
fn reshape_discovery_rejects_equal_size_replacement_that_loses_membership() {
    use crate::hcq::{DiscoveryError, StructuralReason};
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[0x14000008]);
    publish_words(&process, 32, &[RET]);
    publish_words(&process, 48, &[RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, 0x14000008), (32, RET)], 2);
    let work = reshape(&process, 0, 0, 16);
    let Err(DiscoveryError::Structural(rejected)) = Graph::discover(&work) else {
        panic!("membership loss must be rejected");
    };
    assert_eq!(rejected.reason(), StructuralReason::PartitionLoss);
}

#[test]
fn reshape_freeze_preserves_public_entries_without_recompiling_unchanged_body() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    publish_words(&process, 20, &[RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, NOP), (20, RET)], 3);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.graph().instructions.len(), 3);
    assert_eq!(entries(&frozen), [0, 16, 20]);
    assert!(frozen.unchanged());
    assert!(matches!(frozen.analyze(), Err(CompileError::Deferred)));
}

#[test]
fn reshape_freeze_revalidates_a_participant_before_analysis() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    let owner = owned(&process, &[(16, RET)]);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    process.retire_unit(owner).unwrap();
    assert_eq!(frozen.check(), Err(Error::StalePublication));
    assert!(matches!(frozen.analyze(), Err(CompileError::Cancelled)));
}

#[test]
fn reshape_freeze_rejects_output_which_differs_from_its_captured_words() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(matches!(
        frozen.prepare(input(&process, &[0, 16], Tier::Hcq), &AtomicU64::new(0)),
        Err(Error::InvalidUnit(
            "HCQ output differs from its frozen candidate"
        ))
    ));
    frozen.check().unwrap();
}
