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

fn fallbacks(frozen: &Frozen<'_, '_>) -> Vec<u64> {
    frozen
        .replacement()
        .fallbacks
        .iter()
        .map(|entry| entry.key.pc.get())
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
    assert!(fallbacks(&frozen).is_empty());
}

#[test]
fn reshape_freeze_captures_covered_and_dropped_predecessor_entries() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    publish_words(&process, 20, &[RET]);
    let outside = publish_words(&process, 32, &[RET]);
    let old = owned_entries(
        &process,
        &[(0, 0x14000004), (16, NOP), (20, RET), (32, RET)],
        4,
    );
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(entries(&frozen), [0, 16, 20]);
    assert_eq!(fallbacks(&frozen), [32]);
    assert!(frozen.graph().contains(instruction(20)));
    assert!(!frozen.graph().contains(instruction(32)));
    let replacement = frozen.replacement();
    assert_eq!(replacement.predecessors.iter().flatten().count(), 1);
    for (fallback, baseline) in replacement.fallbacks.iter().zip([outside]) {
        assert_eq!(fallback.previous.unit, old);
        assert_eq!(fallback.baseline.registered_handle(), Some(baseline));
        let state = process.lock();
        let payload = state.dispatch.get(fallback.slot).unwrap().snapshot();
        // Freeze has not changed any live entry or active membership.
        assert!(payload.hcq().is_some());
        assert_eq!(payload.lcq().unwrap().unit, fallback.baseline.id);
    }
    assert!(!frozen.unchanged());
    frozen.analyze().unwrap();
}

#[test]
fn reshape_freeze_merges_two_families_and_captures_both_fallback_sets() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    for pc in [8, 16, 24] {
        publish_words(&process, pc, &[RET]);
    }
    let first = owned_entries(&process, &[(0, 0x14000004), (8, RET)], 2);
    let second = owned_entries(&process, &[(16, RET), (24, RET)], 2);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(entries(&frozen), [0, 16]);
    assert_eq!(fallbacks(&frozen), [8, 24]);
    assert_eq!(
        frozen
            .replacement()
            .predecessors
            .iter()
            .flatten()
            .map(|p| p.registered_handle().unwrap())
            .collect::<Vec<_>>(),
        [first, second]
    );
    assert!(!frozen.unchanged());
    frozen.analyze().unwrap();
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
        assert!(fallbacks(&frozen).is_empty());
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
fn reshape_freeze_does_not_confuse_equal_sizes_with_equal_membership() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[0x14000008]); // B 48.
    publish_words(&process, 32, &[RET]);
    publish_words(&process, 48, &[RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, 0x14000008), (32, RET)], 2);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(entries(&frozen), [0, 16]);
    assert!(fallbacks(&frozen).is_empty());
    assert_eq!(frozen.graph().instructions.len(), 3);
    assert!(!frozen.unchanged());
    frozen.analyze().unwrap();
}

#[test]
fn reshape_freeze_revalidates_fallbacks_outside_the_candidate() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    publish_words(&process, 32, &[RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, RET), (32, RET)], 3);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(fallbacks(&frozen), [32]);
    // Isolate the fallback guard from endpoint/participant cancellation: make
    // only the saved LCQ owner unavailable while leaving the family untouched.
    let slot = frozen.replacement().fallbacks[0].slot;
    let owner = process.lock().dispatch.get_mut(slot).unwrap().owners[0].take();
    work.check().unwrap();
    assert_eq!(frozen.check(), Err(Error::StalePublication));
    assert!(matches!(frozen.analyze(), Err(CompileError::Cancelled)));
    process.lock().dispatch.get_mut(slot).unwrap().owners[0] = owner;
    frozen.check().unwrap();
    let baseline = frozen.replacement().fallbacks[0]
        .baseline
        .registered_handle()
        .unwrap();
    process
        .lock()
        .units
        .records
        .get_mut(baseline.0)
        .unwrap()
        .lifecycle = Lifecycle::Invalidating;
    work.check().unwrap();
    assert_eq!(frozen.check(), Err(Error::StalePublication));
    process
        .lock()
        .units
        .records
        .get_mut(baseline.0)
        .unwrap()
        .lifecycle = Lifecycle::Published;
    frozen.check().unwrap();
}

#[test]
fn reshape_freeze_rejects_partial_fallback_capture_and_releases_claims() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    for pc in [16, 32, 48] {
        publish_words(&process, pc, &[RET]);
    }
    owned_entries(
        &process,
        &[(0, 0x14000004), (16, RET), (32, RET), (48, RET)],
        4,
    );
    let work = reshape(&process, 0, 0, 16);
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    let slot = *process.lock().keys.get(&key(48)).unwrap();
    let owner = process.lock().dispatch.get_mut(slot).unwrap().owners[0].take();
    // Capture retains 32 first, then fails at 48. Snapshot/claim destruction
    // must happen outside the state guard and leave this job reusable.
    assert!(matches!(candidate.freeze(), Err(CompileError::Cancelled)));
    process.lock().dispatch.get_mut(slot).unwrap().owners[0] = owner;
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(fallbacks(&frozen), [32, 48]);
    frozen.check().unwrap();
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
    assert!(fallbacks(&frozen).is_empty());
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
