use super::*;
use crate::lifetime::unit::reshape::tests::discovery::{NOP, RET};

fn partial() -> (Arc<Lifetime>, UnitHandle) {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[NOP, NOP, RET]);
    publish_words(&process, 20, &[NOP, RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, NOP)], 2);
    let foreign = owned_entries(&process, &[(20, NOP), (24, RET)], 1);
    (process, foreign)
}

#[test]
fn partial_input_no_op_watches_foreign_membership_without_pinning_its_code() {
    let (process, foreign) = partial();
    let key = result_key(&process);
    let references = Arc::strong_count(&process.lock().units.records.get(foreign.0).unwrap().code);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(frozen.unchanged());
    assert_eq!(frozen.graph().instructions.len(), 2);
    assert!(frozen.prepare_unchanged().unwrap().install().unwrap());
    drop(frozen);
    drop(work);
    assert_eq!(
        Arc::strong_count(&process.lock().units.records.get(foreign.0).unwrap().code),
        references
    );
    assert!(process.lock().units.negatives.get(key).is_some());
    retire(&process, foreign);
    assert!(process.lock().units.negatives.get(key).is_none());
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(!frozen.unchanged());
    assert_eq!(frozen.graph().instructions.len(), 4);
}

#[test]
fn retiring_blocker_rejects_completion_before_membership_is_detached() {
    let (process, foreign) = partial();
    let key = result_key(&process);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let prepared = frozen.prepare_unchanged().unwrap();
    process.retire_unit(foreign).unwrap();
    {
        let state = process.lock();
        // The ownership table is detached later, during maintenance. It is not
        // sufficient proof of an active blocker after retirement is queued.
        assert!(state.units.family_owners.get(instruction(20)).is_some());
        assert!(state.units.active_family_owner(instruction(20)).is_none());
    }
    // Unrelated maintenance no longer cancels Work: the foreign proof itself
    // must reject the retired blocker, even while its index node still exists.
    assert_eq!(
        frozen
            .graph()
            .discovery
            .as_ref()
            .unwrap()
            .validate_no_op(&process.lock(), frozen.graph()),
        Err(Error::StalePublication)
    );
    assert!(matches!(prepared.install(), Err(Error::StalePublication)));
    assert!(process.lock().units.negatives.get(key).is_none());
}

#[test]
fn foreign_membership_is_evidence_even_without_a_dispatch_entry() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[0x14000005]); // B 36.
    publish_words(&process, 32, &[NOP, RET]);
    owned_entries(&process, &[(0, 0x14000004), (16, 0x14000005)], 2);
    let foreign = owned_entries(&process, &[(32, NOP), (36, RET)], 1);
    assert!(!process.lock().keys.contains_key(&key(36)));
    let result_key = result_key(&process);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(frozen.prepare_unchanged().unwrap().install().unwrap());
    assert!(!process.lock().keys.contains_key(&key(36))); // No fabricated demand.
    drop(frozen);
    drop(work);
    retire(&process, foreign);
    assert!(process.lock().units.negatives.get(result_key).is_none());
    // Removing ownership does not invent the missing LCQ entry; now this is
    // unavailable input, not another persistent structural/no-op rejection.
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(matches!(
        frozen.prepare_unchanged(),
        Err(Error::StalePublication)
    ));
}

#[test]
fn frontier_storage_is_charged_and_pressure_does_not_leave_a_negative() {
    let (process, _) = partial();
    let key = result_key(&process);
    let work = reshape(&process, 0, 0, 16);
    let mut blocked = Vec::with_capacity(8);
    blocked.push(work.blocker(super::key(20)).unwrap().unwrap());
    let bytes = blocked.capacity() * size_of_val(&blocked[0]);
    let before = process.cache.usage().unwrap();
    let evidence = work
        .discovery_evidence(Vec::new(), blocked, Vec::new(), true)
        .unwrap();
    let charged = process.cache.usage().unwrap().metadata - before.metadata;
    assert_eq!(charged, bytes);
    assert_eq!(evidence.owner_capacity(), 1); // Capacity is charged, only actual owners indexed.
    drop(evidence);
    assert_eq!(process.cache.usage().unwrap().metadata, before.metadata);
    let pressure = process
        .cache
        .charge_metadata(
            crate::executable::SOFT_BYTES - before.total() - 1,
            Tier::Lcq,
        )
        .unwrap();
    assert!(matches!(
        work.discovery_evidence(
            Vec::new(),
            vec![work.blocker(super::key(20)).unwrap().unwrap()],
            Vec::new(),
            true
        ),
        Err(Error::Capacity(_))
    ));
    assert!(process.lock().units.negatives.get(key).is_none());
    drop(pressure);
    assert_eq!(process.cache.usage().unwrap().metadata, before.metadata);
}
