use super::*;

fn changed(owned: bool) -> (Arc<Lifetime>, UnitHandle) {
    let process = process();
    let source = publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[0xd503201f, 0x17fffffb]);
    publish_words(&process, 20, &[0x17fffffb]);
    if owned {
        owned_entries(&process, &[(0, 0x14000004)], 1);
    }
    (process, source)
}

#[test]
fn backend_negative_is_boundary_scoped_weak_and_suppresses_repeats() {
    for owned in [false, true] {
        let (process, source) = changed(owned);
        let key = result_key(&process);
        let references =
            Arc::strong_count(&process.lock().units.records.get(source.0).unwrap().code);
        let mut metadata = None;
        for expected in [true, false] {
            if !expected {
                assert_suppressed(&process, key.source, key.boundary);
                assert_eq!(Some(process.cache.usage().unwrap().metadata), metadata);
                continue;
            }
            let work = reshape(&process, 0, 0, 16);
            let frozen = work
                .reserve_candidate(Graph::discover(&work).unwrap())
                .unwrap()
                .freeze()
                .unwrap();
            assert!(!frozen.unchanged());
            let committed = process.cache.usage().unwrap().committed;
            assert_eq!(
                frozen
                    .prepare_backend_negative(MemoryInvalidationCursor::new(42))
                    .unwrap()
                    .install()
                    .unwrap(),
                expected
            );
            drop(frozen);
            drop(work);
            assert_eq!(process.cache.usage().unwrap().committed, committed);
            assert_eq!(
                Arc::strong_count(&process.lock().units.records.get(source.0).unwrap().code),
                references
            );
            let usage = process.cache.usage().unwrap().metadata;
            metadata = Some(usage);
            assert_eq!(
                process.lock().units.negatives.get(key).unwrap().reason,
                Rejection::BackendRejected
            );
        }
        let retired = if owned {
            process
                .lock()
                .units
                .active_family_owner(crate::abi::InstructionKey::new(super::key(0)).unwrap())
                .unwrap()
        } else {
            source
        };
        process.retire_unit(retired).unwrap();
        assert!(process.lock().units.negatives.get(key).is_none());
    }
}

#[test]
fn late_pic_root_cancels_backend_negative_with_unchanged_code_and_claims() {
    let (process, _) = changed(false);
    let key = result_key(&process);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let prepared = frozen
        .prepare_backend_negative(MemoryInvalidationCursor::INITIAL)
        .unwrap();
    reader
        .cache_bridge(
            process
                .prepare_dynamic_bridge(source, 0, super::key(20))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    frozen.check().unwrap();
    assert_eq!(prepared.install(), Err(Error::StalePublication));
    assert!(process.lock().units.negatives.get(key).is_none());
}

#[test]
fn backend_entry_page_watch_catches_lcq_only_pic_changes_but_not_hits_or_other_pages() {
    let (process, _) = changed(false);
    let key = result_key(&process);
    publish_words(&process, 0x2000, &[0xd65f03c0]);
    let source = dynamic::tests::source(&process, &AtomicU64::new(0), 128, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let install = || {
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(
            frozen
                .prepare_backend_negative(MemoryInvalidationCursor::INITIAL)
                .unwrap()
                .install()
                .unwrap()
        );
    };
    install();
    reader
        .cache_bridge(
            process
                .prepare_dynamic_bridge(source, 0, super::key(0x2000))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    assert!(process.lock().units.negatives.get(key).is_some());
    reader
        .cache_bridge(
            process
                .prepare_dynamic_bridge(source, 0, super::key(20))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    assert!(process.lock().units.negatives.get(key).is_none());
    install(); // This candidate now includes PC 20 as a public entry.
    reader
        .cache_bridge(
            process
                .prepare_dynamic_bridge(source, 0, super::key(20))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    assert!(process.lock().units.negatives.get(key).is_some());
    drop(reader);
    assert!(process.lock().units.negatives.get(key).is_none());
}

#[test]
fn backend_negative_missing_input_and_pressure_defer_without_consuming_family_state() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[0x14000004]); // B 32, not yet demanded.
    let key = result_key(&process);
    {
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(matches!(
            frozen.prepare_backend_negative(MemoryInvalidationCursor::INITIAL),
            Err(Error::StalePublication)
        ));
        assert!(process.lock().units.negatives.get(key).is_none());
    }
    publish_words(&process, 32, &[0xd65f03c0]);
    {
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let prepared = frozen
            .prepare_backend_negative(MemoryInvalidationCursor::INITIAL)
            .unwrap();
        let usage = process.cache.usage().unwrap();
        let pressure = process
            .cache
            .charge_metadata(crate::executable::SOFT_BYTES - usage.total(), Tier::Lcq)
            .unwrap();
        assert!(matches!(prepared.install(), Err(Error::Capacity(_))));
        assert!(process.lock().units.negatives.get(key).is_none());
        drop(pressure);
    }
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert!(
        frozen
            .prepare_backend_negative(MemoryInvalidationCursor::INITIAL)
            .unwrap()
            .install()
            .unwrap()
    );
}

#[test]
fn backend_entry_page_watch_tracks_static_roots_without_an_hcq_owner() {
    let (process, _) = changed(false);
    let key = result_key(&process);
    let install = || {
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(
            frozen
                .prepare_backend_negative(MemoryInvalidationCursor::INITIAL)
                .unwrap()
                .install()
                .unwrap()
        );
    };
    install();
    let source = crate::lifetime::unit::links::tests::source(&process, &AtomicU64::new(0), 128, 20);
    assert!(process.lock().units.negatives.get(key).is_none());
    install();
    retire(&process, source);
    assert!(process.lock().units.negatives.get(key).is_none());
}

#[test]
fn backend_negative_can_record_a_complete_capped_candidate_and_watches_excluded_input() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[0x140003fc]); // B 0x1000.
    let mut excluded = None;
    for pc in [0x1000, 0x2000, 0x3000, 0x4000] {
        let mut words = vec![0xd503201f; 512];
        words[511] = if pc == 0x4000 { 0xd65f03c0 } else { 0x14000201 };
        excluded = Some(publish_words(&process, pc, &words));
    }
    let key = result_key(&process);
    {
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(frozen.graph().instructions.len(), 1538);
        assert_eq!(frozen.graph().discovery.as_ref().unwrap().len(), 6);
        assert!(
            frozen
                .prepare_backend_negative(MemoryInvalidationCursor::INITIAL)
                .unwrap()
                .install()
                .unwrap()
        );
    }
    process.retire_unit(excluded.unwrap()).unwrap();
    assert!(process.lock().units.negatives.get(key).is_none());
}
