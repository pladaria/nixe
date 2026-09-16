use super::*;

#[test]
fn link_service_yields_to_readers_and_resumes_deferred_work_without_waiting() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut reader = process.register().unwrap();
    let mut transition = stop(&process);
    let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
    let handle = transition.register_link(prepared).unwrap();
    let ticket = process.request(Reason::LinkPatch).unwrap();
    assert!(!process.try_service_links().unwrap()); // Existing transition owner.
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert!(!process.try_service_links().unwrap());
    assert_eq!(process.lock().phase, crate::lifetime::Phase::Closing);
    assert!(!ticket.is_complete().unwrap());
    assert!(
        !process
            .lock()
            .units
            .links
            .records
            .get(handle.0)
            .unwrap()
            .installed
    );
    drop(invocation);
    assert!(process.try_service_links().unwrap());
    assert!(ticket.is_complete().unwrap());
    assert!(
        process
            .lock()
            .units
            .links
            .records
            .get(handle.0)
            .unwrap()
            .installed
    );
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn link_service_never_acknowledges_memory_pressure_or_shutdown_work() {
    for foreign in [Reason::MappingChange, Reason::Eviction, Reason::Shutdown] {
        let process = process();
        let cursor = AtomicU64::new(0);
        publish(&process, &cursor, &[4], Tier::Lcq);
        let src = source(&process, &cursor, 0, 4);
        let mut transition = stop(&process);
        let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
        transition.register_link(prepared).unwrap();
        drop(transition);
        let ticket = process.request(foreign).unwrap();
        if foreign == Reason::Shutdown {
            assert_eq!(process.try_service_links(), Err(Error::Shutdown));
        } else {
            assert!(!process.try_service_links().unwrap());
        }
        assert!(!ticket.is_complete().unwrap());
        assert!(!pending(&process).is_empty());
        assert!(process.try_shutdown().unwrap());
    }
}

#[test]
fn link_service_completes_replacement_retirement_without_an_external_owner() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let src = source(&process, &cursor, 0, 4);
    let mut transition = stop(&process);
    let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
    let handle = transition.register_link(prepared).unwrap();
    drop(transition);
    assert!(process.try_service_links().unwrap());
    let new = publish(&process, &cursor, &[4], Tier::Lcq);
    assert!(process.try_service_links().unwrap());
    assert!(process.lock().units.links.records.get(handle.0).is_none());
    let transition = stop(&process);
    assert_eq!(
        transition
            .prepare_static_link(src, 0)
            .unwrap()
            .unwrap()
            .target,
        new
    );
    drop(transition);
    assert!(process.try_service_links().unwrap());
    assert!(process.try_shutdown().unwrap());
}
