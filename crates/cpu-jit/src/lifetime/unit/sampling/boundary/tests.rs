use super::*;
use crate::lifetime::unit::tests::{input, key, process, publish};
use nixe_memory::GuestVirtualAddress;

fn instruction(pc: u64) -> InstructionKey {
    InstructionKey::new(key(pc)).unwrap()
}

fn edge(pc: u64) -> ObservedEdge {
    ObservedEdge {
        destination: GuestVirtualAddress::new(pc),
        kind: EdgeKind::Static,
    }
}

fn hcq(process: &Lifetime, cursor: &AtomicU64, pcs: &[u64], entries: usize) -> UnitHandle {
    let mut candidate = input(process, pcs, Tier::Hcq);
    candidate.entries = candidate
        .entries
        .into_vec()
        .into_iter()
        .take(entries)
        .collect();
    let publications: Vec<_> = pcs[..entries]
        .iter()
        .map(|pc| process.reserve(key(*pc)).unwrap())
        .collect();
    let handle = process
        .prepare_unit(&publications, candidate, cursor)
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    handle
}

fn retire(process: &Lifetime, handle: UnitHandle) {
    process.retire_unit(handle).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn uncovered_transfers_heat_the_root_without_demanding_destinations() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let handle = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    publish(&process, &cursor, &[16], Tier::Lcq);
    let unit = process.snapshot(handle).unwrap();
    let slots = process.lock().keys.len();
    let mut samples = Samples::new();
    for target in [16, 32, 33] {
        process
            .sample_lcq(&unit, &mut samples, Some(edge(target)))
            .unwrap();
        assert!(
            samples
                .boundary_snapshot(instruction(4), instruction(target & !3))
                .is_none()
        );
    }
    let (snapshot, score) = samples.seed_snapshot(key(0)).unwrap();
    assert_eq!(score, 3);
    assert_eq!(snapshot.last_edge, Some(edge(33)));
    assert_eq!(snapshot.successors.iter().flatten().count(), 2);
    assert!(samples.seed_snapshot(key(4)).is_none());
    assert_eq!(process.lock().keys.len(), slots);
}

#[test]
fn boundary_uses_actual_instruction_reachabilities_and_both_family_identities() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    publish(&process, &cursor, &[16], Tier::Lcq);
    publish(&process, &cursor, &[32], Tier::Lcq);
    hcq(&process, &cursor, &[16], 1);
    let mut samples = Samples::new();
    let baseline = process.snapshot(baseline).unwrap();
    for _ in 0..4 {
        process
            .sample_lcq(&baseline, &mut samples, Some(edge(16)))
            .unwrap();
    }
    let (observed, score) = samples
        .boundary_snapshot(instruction(4), instruction(16))
        .unwrap();
    assert_eq!(score, 4);
    let state = process.lock();
    let source = endpoint(&state, key(0)).unwrap();
    let target = endpoint(&state, key(16)).unwrap();
    assert_eq!(observed.key.source_version, source.payload.reachability());
    assert_eq!(observed.key.target_version, target.payload.reachability());
    assert_eq!(observed.key.source_family, None);
    assert_eq!(observed.key.target_family, target.family);
    assert!(target.family.is_some());
    drop(state);
    assert!(samples.seed_snapshot(key(0)).is_none());

    let optimized = hcq(&process, &cursor, &[0, 4], 1);
    let optimized = process.snapshot(optimized).unwrap();
    for target in [16, 32] {
        process
            .sample_transfer(
                &optimized,
                key(0),
                instruction(4),
                &mut samples,
                edge(target),
            )
            .unwrap();
        let (observed, score) = samples
            .boundary_snapshot(instruction(4), instruction(target))
            .unwrap();
        let state = process.lock();
        assert_eq!(score, 1); // Changed source ownership resets the old observation.
        assert_eq!(
            observed.key.source_family,
            endpoint(&state, key(0)).unwrap().family
        );
        assert_eq!(
            observed.key.target_family,
            endpoint(&state, key(target)).unwrap().family
        );
        assert_eq!(observed.key.target_family.is_some(), target == 16);
    }
}

#[test]
fn internal_hcq_edges_are_not_boundaries_but_retained_lcq_entries_are() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[0, 4, 8], Tier::Lcq);
    let optimized = hcq(&process, &cursor, &[0, 4, 8], 2);
    let optimized = process.snapshot(optimized).unwrap();
    let mut samples = Samples::new();
    // Logical source block 4 is not the family's first public entry.
    process
        .sample_transfer(&optimized, key(4), instruction(4), &mut samples, edge(0))
        .unwrap();
    assert!(
        samples
            .boundary_snapshot(instruction(4), instruction(0))
            .is_none()
    );
    process
        .sample_transfer(&optimized, key(4), instruction(4), &mut samples, edge(8))
        .unwrap();
    let (observed, _) = samples
        .boundary_snapshot(instruction(4), instruction(8))
        .unwrap();
    assert!(observed.key.source_family.is_some());
    assert_eq!(observed.key.source_family, observed.key.target_family);
    assert!(
        endpoint(&process.lock(), key(8))
            .unwrap()
            .payload
            .hcq()
            .is_none()
    );
    // A retained baseline body can still be the actual executing source.
    process
        .sample_lcq(
            &process.snapshot(baseline).unwrap(),
            &mut samples,
            Some(edge(0)),
        )
        .unwrap();
    assert!(
        samples
            .boundary_snapshot(instruction(8), instruction(0))
            .is_some()
    );
    assert!(samples.seed_snapshot(key(0)).is_none());
}

#[test]
fn interior_ownership_suppresses_seed_and_completion_even_without_hcq_entry() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let optimized = hcq(&process, &cursor, &[0, 4], 1);
    let interior = publish(&process, &cursor, &[4], Tier::Lcq);
    process.try_service_links().unwrap();
    let interior = process.snapshot(interior).unwrap();
    let mut samples = Samples::new();
    assert!(process.completion_sample(&interior).unwrap().is_none());
    process.sample_lcq(&interior, &mut samples, None).unwrap();
    assert!(samples.seed_snapshot(key(4)).is_none());
    process
        .sample_lcq(&interior, &mut samples, Some(edge(0)))
        .unwrap();
    let (observed, _) = samples
        .boundary_snapshot(instruction(4), instruction(0))
        .unwrap();
    assert!(observed.key.source_family.is_some());
    assert_eq!(observed.key.source_family, observed.key.target_family);
    retire(&process, optimized);
    process.sample_lcq(&interior, &mut samples, None).unwrap();
    assert_eq!(samples.seed_snapshot(key(4)).unwrap().1, 1);
}

#[test]
fn missing_dispatch_identity_is_not_fabricated_from_instruction_ownership() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut candidate = input(&process, &[0, 4], Tier::Lcq);
    candidate.entries = candidate.entries.into_vec().into_iter().take(1).collect();
    process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], candidate, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let optimized = hcq(&process, &cursor, &[0, 4], 1);
    let optimized = process.snapshot(optimized).unwrap();
    let mut samples = Samples::new();
    process
        .sample_transfer(&optimized, key(0), instruction(0), &mut samples, edge(4))
        .unwrap();
    process
        .sample_transfer(&optimized, key(4), instruction(4), &mut samples, edge(0))
        .unwrap();
    assert!(
        samples
            .boundary_snapshot(instruction(0), instruction(4))
            .is_none()
    );
    assert!(
        samples
            .boundary_snapshot(instruction(4), instruction(0))
            .is_none()
    );
    let state = process.lock();
    assert_eq!(state.keys.len(), 1);
    assert!(state.units.family_owners.get(instruction(4)).is_some());
}

#[test]
fn contention_closure_and_retired_sources_do_not_add_boundary_heat() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let handle = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    hcq(&process, &cursor, &[4], 1);
    let unit = process.snapshot(handle).unwrap();
    let mut samples = Samples::new();
    let guard = process.lock();
    process
        .sample_lcq(&unit, &mut samples, Some(edge(4)))
        .unwrap();
    assert!(
        samples
            .boundary_snapshot(instruction(0), instruction(4))
            .is_none()
    );
    drop(guard);
    process
        .sample_lcq(&unit, &mut samples, Some(edge(4)))
        .unwrap();
    let before = samples
        .boundary_snapshot(instruction(0), instruction(4))
        .unwrap();
    process.request(Reason::LinkPatch).unwrap();
    process
        .sample_lcq(&unit, &mut samples, Some(edge(4)))
        .unwrap();
    assert_eq!(
        samples
            .boundary_snapshot(instruction(0), instruction(4))
            .unwrap(),
        before
    );
    process.try_service_links().unwrap();
    retire(&process, handle);
    process
        .sample_lcq(&unit, &mut samples, Some(edge(4)))
        .unwrap();
    assert_eq!(
        samples
            .boundary_snapshot(instruction(0), instruction(4))
            .unwrap(),
        before
    );
}

#[test]
fn target_family_replacement_resets_heat_and_rejects_pinned_old_hcq_source() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let old = hcq(&process, &cursor, &[4], 1);
    let old_pin = process.snapshot(old).unwrap();
    let baseline = process.snapshot(baseline).unwrap();
    let mut samples = Samples::new();
    for _ in 0..3 {
        process
            .sample_lcq(&baseline, &mut samples, Some(edge(4)))
            .unwrap();
    }
    let (before, score) = samples
        .boundary_snapshot(instruction(0), instruction(4))
        .unwrap();
    assert_eq!(score, 3);
    retire(&process, old);
    hcq(&process, &cursor, &[4], 1);
    process
        .sample_lcq(&baseline, &mut samples, Some(edge(4)))
        .unwrap();
    let (after, score) = samples
        .boundary_snapshot(instruction(0), instruction(4))
        .unwrap();
    assert_eq!(score, 1);
    assert_ne!(before.key.target_family, after.key.target_family);
    assert_ne!(before.key.target_version, after.key.target_version);
    process
        .sample_transfer(&old_pin, key(4), instruction(4), &mut samples, edge(0))
        .unwrap();
    assert!(
        samples
            .boundary_snapshot(instruction(4), instruction(0))
            .is_none()
    );
}

#[test]
fn source_instruction_must_belong_to_the_exact_lcq_image_and_context() {
    let process = process();
    let handle = publish(&process, &AtomicU64::new(0), &[0, 4], Tier::Lcq);
    let unit = process.snapshot(handle).unwrap();
    let mut samples = Samples::new();
    assert_eq!(
        process.sample_transfer(&unit, key(0), instruction(8), &mut samples, edge(0)),
        Err(Error::InvalidUnit(
            "sample instruction is absent from LCQ source"
        ))
    );
    let other = InstructionKey::new(BlockKey {
        fp: crate::abi::FpSpecialization::Exact(0),
        ..key(4)
    })
    .unwrap();
    assert_eq!(
        process.sample_transfer(&unit, key(0), other, &mut samples, edge(0)),
        Err(Error::InvalidUnit("sample source mixes execution contexts"))
    );
    assert!(samples.seed_snapshot(key(0)).is_none());
}
