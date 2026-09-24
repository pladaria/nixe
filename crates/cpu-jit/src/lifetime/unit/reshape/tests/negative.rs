//! Lifetime-side installation tests; real memory authority is exercised by the
//! HCQ publication tests. These do not bypass the validated result installer.
use super::discovery::{owned_entries, reshape};
use super::*;
use crate::hcq::Graph;
use crate::lifetime::unit::dynamic::pic::tests::cache;
use crate::lifetime::unit::reshape::negative::Key;
use crate::lifetime::unit::{dynamic, tests::publish_words};

mod admission;
mod backend;
mod frontier;

fn setup() -> (Arc<Lifetime>, UnitHandle) {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[0xd503201f, 0x17fffffb]);
    publish_words(&process, 20, &[0x17fffffb]);
    let family = owned_entries(
        &process,
        &[(0, 0x14000004), (16, 0xd503201f), (20, 0x17fffffb)],
        2,
    );
    (process, family)
}

fn result_key(process: &Lifetime) -> Key {
    Key {
        source: key(0),
        boundary: boundary(process, 0, 0, 16),
    }
}

#[test]
fn unchanged_installation_transfers_budget_suppresses_repeats_and_retains_no_code() {
    let (process, family) = setup();
    let key = result_key(&process);
    let snapshot = process.snapshot(family).unwrap();
    let references = Arc::strong_count(&process.lock().units.records.get(family.0).unwrap().code);
    let mut resident_metadata = None;
    for inserted in [true, false] {
        if !inserted {
            assert_suppressed(&process, key.source, key.boundary);
            assert_eq!(
                Some(process.cache.usage().unwrap().metadata),
                resident_metadata
            );
            continue;
        }
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let code_bytes = process.cache.usage().unwrap().committed;
        assert_eq!(
            frozen.prepare_unchanged().unwrap().install().unwrap(),
            inserted
        );
        assert_eq!(process.cache.usage().unwrap().committed, code_bytes);
        drop(frozen);
        drop(work);
        let metadata = process.cache.usage().unwrap().metadata;
        resident_metadata = Some(metadata);
        assert_eq!(
            Arc::strong_count(&process.lock().units.records.get(family.0).unwrap().code),
            references
        );
        let state = process.lock();
        assert!(state.units.negatives.get(key).is_some());
    }
    process.retire_unit(family).unwrap();
    assert!(process.lock().units.negatives.get(key).is_none());
    drop(snapshot);
}

#[test]
fn late_entry_root_rejects_prepared_negative_and_allows_retry_after_cancellation() {
    let (process, _) = setup();
    let result_key = result_key(&process);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let prepared = frozen.prepare_unchanged().unwrap();
    cache(
        &mut reader,
        process
            .prepare_dynamic_bridge(source, 0, key(20))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    frozen.check().unwrap();
    assert_eq!(prepared.install(), Err(Error::StalePublication));
    assert!(process.lock().units.negatives.get(result_key).is_none());
    drop(frozen);
    drop(work);
    drop(reader);
    let retry = reshape(&process, 0, 0, 16);
    let frozen = retry
        .reserve_candidate(Graph::discover(&retry).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(frozen.prepare_unchanged().unwrap().install().unwrap());
}

#[test]
fn cancelled_prepared_negative_cannot_install_after_retirement_or_shutdown() {
    for shutdown in [false, true] {
        let (process, family) = setup();
        let key = result_key(&process);
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let prepared = frozen.prepare_unchanged().unwrap();
        if shutdown {
            process.request_shutdown().unwrap();
        } else {
            process.retire_unit(family).unwrap();
        }
        assert!(matches!(
            prepared.install(),
            Err(Error::Closed | Error::Shutdown | Error::StalePublication)
        ));
        assert!(process.lock().units.negatives.get(key).is_none());
        drop(frozen);
        drop(work);
        if shutdown {
            assert!(process.try_shutdown().unwrap());
        }
        assert_eq!(process.lock().compilers, 0);
    }
}

#[test]
fn evidence_pressure_and_abandoned_preparation_leave_no_negative_and_allow_retry() {
    for pressure in [false, true] {
        let (process, _) = setup();
        let key = result_key(&process);
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let charge = pressure.then(|| {
            let usage = process.cache.usage().unwrap();
            process
                .cache
                .charge_metadata(crate::executable::SOFT_BYTES - usage.total() - 1, Tier::Lcq)
                .unwrap()
        });
        let prepared = frozen.prepare_unchanged();
        if pressure {
            assert!(matches!(prepared, Err(Error::Capacity(_))));
        } else {
            drop(prepared.unwrap());
        }
        assert!(process.lock().units.negatives.get(key).is_none());
        drop(charge);
        drop(frozen);
        drop(work);
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(frozen.prepare_unchanged().unwrap().install().unwrap());
    }
}

#[test]
fn temporary_competitor_cannot_turn_a_trimmed_candidate_into_a_persistent_no_op() {
    let process = process();
    for pc in [0, 16, 32] {
        publish_words(&process, pc, &[0x14000004]);
    }
    publish_words(&process, 48, &[0xd65f03c0]);
    owned_entries(&process, &[(0, 0x14000004), (16, 0x14000004)], 2);
    let work = reshape(&process, 0, 0, 16);
    let graph = Graph::discover(&work).unwrap();
    assert_eq!(graph.instructions.len(), 4);
    assert_eq!(graph.discovery.as_ref().unwrap().len(), 4);
    let competitor = reshape(&process, 32, 32, 48);
    let claims = competitor
        .reserve_candidate(Graph::discover(&competitor).unwrap())
        .unwrap();
    let frozen = work.reserve_candidate(graph).unwrap().freeze().unwrap();
    assert_eq!(frozen.graph().instructions.len(), 2);
    assert_eq!(frozen.graph().units.len(), 2);
    assert_eq!(frozen.graph().discovery.as_ref().unwrap().len(), 4);
    assert!(frozen.unchanged());
    assert!(matches!(
        frozen.prepare_unchanged(),
        Err(Error::StalePublication)
    ));
    assert!(matches!(
        frozen.prepare_backend_negative(),
        Err(Error::StalePublication)
    ));
    // Releasing the competitor makes the larger useful candidate available.
    drop(claims);
    drop(competitor);
    drop(frozen);
    let next = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(!next.unchanged());
    let key = result_key(&process);
    assert!(process.lock().units.negatives.get(key).is_none());
}

#[test]
fn discovery_distinguishes_missing_inputs_from_stable_foreign_frontiers() {
    for foreign in [false, true] {
        let process = process();
        publish_words(&process, 0, &[0x14000004]); // B 16.
        publish_words(&process, 16, &[0x14000004]); // B 32.
        owned_entries(&process, &[(0, 0x14000004), (16, 0x14000004)], 2);
        let foreign_owner = foreign.then(|| {
            publish_words(&process, 32, &[0xd65f03c0]);
            owned_entries(&process, &[(32, 0xd65f03c0)], 1)
        });
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(frozen.unchanged());
        frozen.check().unwrap(); // The positive candidate contract is unaffected.
        let key = result_key(&process);
        let prepared = frozen.prepare_unchanged();
        if foreign {
            assert!(prepared.unwrap().install().unwrap());
            assert!(process.lock().units.negatives.get(key).is_some());
        } else {
            assert!(matches!(prepared, Err(Error::StalePublication)));
            assert!(process.lock().units.negatives.get(key).is_none());
        }
        drop(frozen);
        drop(work);
        if let Some(owner) = foreign_owner {
            retire(&process, owner);
            assert!(process.lock().units.negatives.get(key).is_none());
        } else {
            publish_words(&process, 32, &[0xd65f03c0]);
        }
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(!frozen.unchanged());
        assert_eq!(frozen.graph().instructions.len(), 3);
        frozen.check().unwrap();
    }
}

#[test]
fn discovery_evidence_is_charged_weak_and_rejects_retired_input() {
    let (process, _) = setup();
    let work = reshape(&process, 0, 0, 16);
    let mut graph = Graph::discover(&work).unwrap();
    let evidence = graph.discovery.take().unwrap();
    assert_eq!(evidence.len(), 3);
    evidence.validate_no_op(&process.lock(), &graph).unwrap();
    let input = work.lcq(key(20)).unwrap().unwrap();
    let handle = input.unit.registered_handle().unwrap();
    let references = Arc::strong_count(&process.lock().units.records.get(handle.0).unwrap().code);
    let mut inspections = Vec::with_capacity(8);
    let mut leaders = Vec::new();
    inspections.push(input.inspection(&work.extent(&input).unwrap(), &mut leaders));
    let before = process.cache.usage().unwrap().metadata;
    let weak = work
        .discovery_evidence(inspections, Vec::new(), leaders, true)
        .unwrap();
    assert!(process.cache.usage().unwrap().metadata > before);
    assert_eq!(
        Arc::strong_count(&process.lock().units.records.get(handle.0).unwrap().code),
        references
    );
    weak.validate_no_op(&process.lock(), &graph).unwrap();
    drop(weak);
    assert_eq!(process.cache.usage().unwrap().metadata, before);
    // Still pinned by both graph and input: retirement, not deallocation, ends validity.
    process.retire_unit(handle).unwrap();
    assert_eq!(
        evidence.validate_no_op(&process.lock(), &graph),
        Err(Error::StalePublication)
    );
}

#[test]
fn discovery_evidence_pressure_defers_without_leaking_storage_or_code_pins() {
    let (process, _) = setup();
    let work = reshape(&process, 0, 0, 16);
    let input = work.lcq(key(20)).unwrap().unwrap();
    let handle = input.unit.registered_handle().unwrap();
    let references = Arc::strong_count(&process.lock().units.records.get(handle.0).unwrap().code);
    let extent = work.extent(&input).unwrap();
    let usage = process.cache.usage().unwrap();
    let pressure = process
        .cache
        .charge_metadata(crate::executable::SOFT_BYTES - usage.total() - 1, Tier::Lcq)
        .unwrap();
    let charged = process.cache.usage().unwrap().metadata;
    let mut leaders = Vec::new();
    assert!(matches!(
        work.discovery_evidence(
            vec![input.inspection(&extent, &mut leaders)],
            Vec::new(),
            leaders,
            true
        ),
        Err(Error::Capacity(_))
    ));
    assert_eq!(process.cache.usage().unwrap().metadata, charged);
    assert_eq!(
        Arc::strong_count(&process.lock().units.records.get(handle.0).unwrap().code),
        references
    );
    drop(pressure);
    let mut leaders = Vec::new();
    let evidence = work
        .discovery_evidence(
            vec![input.inspection(&extent, &mut leaders)],
            Vec::new(),
            leaders,
            true,
        )
        .unwrap();
    drop(evidence);
    assert_eq!(process.cache.usage().unwrap().metadata, usage.metadata);
}
