//! Exact lifetime proof; real executable-memory checks live in publication tests.
use super::*;
use crate::lifetime::unit::reshape::negative::{Key, Rejection};

fn result_key(process: &Lifetime) -> Key {
    Key {
        source: key(0),
        boundary: boundary(process, 0, 0x4000, 0x8000),
    }
}

#[test]
fn cap_installation_suppresses_repeats_releases_worker_storage_and_keeps_only_weak_evidence() {
    let process = process();
    let inputs = chain(&process, false);
    let result_key = result_key(&process);
    let references: Vec<_> = inputs
        .iter()
        .map(|input| Arc::strong_count(&process.lock().units.records.get(input.0).unwrap().code))
        .collect();
    let committed = process.cache.usage().unwrap().committed;
    let mut metadata = None;
    for expected in [true, false] {
        if !expected {
            assert_suppressed(&process, result_key.source, result_key.boundary);
            assert_eq!(Some(process.cache.usage().unwrap().metadata), metadata);
            continue;
        }
        let work = reshape(&process, 0, 0x4000, 0x8000);
        let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
            panic!()
        };
        assert_eq!(
            result
                .prepare(MemoryInvalidationCursor::new(42))
                .unwrap()
                .install()
                .unwrap(),
            expected
        );
        drop(result);
        drop(work);
        for (input, &references) in inputs.iter().zip(&references) {
            assert_eq!(
                Arc::strong_count(&process.lock().units.records.get(input.0).unwrap().code),
                references
            );
        }
        let usage = process.cache.usage().unwrap();
        assert_eq!(usage.committed, committed);
        metadata = Some(usage.metadata);
        let state = process.lock();
        let record = state.units.negatives.get(result_key).unwrap();
        assert_eq!(record.reason, Rejection::InstructionLimit);
        assert_eq!(record.cursor, MemoryInvalidationCursor::new(42));
    }
    process.retire_unit(inputs[6]).unwrap(); // Cap-excluded input, not an endpoint.
    assert!(process.lock().units.negatives.get(result_key).is_none());
}

#[test]
fn new_interior_leader_rejects_prepared_cap_result_without_changing_endpoint_versions() {
    let process = process();
    chain(&process, false);
    let result_key = result_key(&process);
    let work = reshape(&process, 0, 0x4000, 0x8000);
    let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!()
    };
    let prepared = result.prepare(MemoryInvalidationCursor::INITIAL).unwrap();
    let mut words = vec![NOP; 511];
    words[510] = branch(0x57fc, 0x4000);
    publish_words(&process, 0x5004, &words);
    work.check().unwrap(); // Admission/endpoint checks alone cannot detect this.
    assert_eq!(result.check(), Err(Error::StalePublication));
    assert_eq!(prepared.install(), Err(Error::StalePublication));
    assert!(process.lock().units.negatives.get(result_key).is_none());
}

#[test]
fn installed_cap_result_watches_pages_not_every_instruction_or_unrelated_demand() {
    for ownership in [false, true] {
        let process = process();
        chain(&process, false);
        let result_key = result_key(&process);
        {
            let work = reshape(&process, 0, 0x4000, 0x8000);
            let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
                panic!()
            };
            assert!(
                result
                    .prepare(MemoryInvalidationCursor::INITIAL)
                    .unwrap()
                    .install()
                    .unwrap()
            );
        }
        publish_words(&process, 0xa000, &[RET]);
        assert!(process.lock().units.negatives.get(result_key).is_some());
        if ownership {
            // No demand slot exists here: new ownership alone changes eligibility.
            owned_entries(&process, &[(0xa000, RET), (0x5004, NOP)], 1);
        } else {
            publish_words(&process, 0x5004, &[NOP, RET]);
        }
        assert!(process.lock().units.negatives.get(result_key).is_none());
    }
}

#[test]
fn structural_installation_pressure_and_shutdown_leave_no_partial_result() {
    let process = process();
    chain(&process, false);
    let result_key = result_key(&process);
    {
        let work = reshape(&process, 0, 0x4000, 0x8000);
        let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
            panic!()
        };
        let prepared = result.prepare(MemoryInvalidationCursor::INITIAL).unwrap();
        let usage = process.cache.usage().unwrap();
        let pressure = process
            .cache
            .charge_metadata(crate::executable::SOFT_BYTES - usage.total(), Tier::Lcq)
            .unwrap();
        assert!(matches!(prepared.install(), Err(Error::Capacity(_))));
        assert!(process.lock().units.negatives.get(result_key).is_none());
        drop(pressure);
    }
    let work = reshape(&process, 0, 0x4000, 0x8000);
    let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!()
    };
    let prepared = result.prepare(MemoryInvalidationCursor::INITIAL).unwrap();
    process.request_shutdown().unwrap();
    assert!(prepared.install().is_err());
    assert!(process.lock().units.negatives.get(result_key).is_none());
}
