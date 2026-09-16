use super::*;

fn execute_at(process: &Arc<Lifetime>, pc: u64) -> u32 {
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(pc)) }
        .unwrap()
        .unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    // The synthetic System-ABI source and target share this real reader epoch.
    let call: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(address) };
    unsafe { call() }
}

#[test]
fn installed_retargeting_defers_excess_work_without_losing_the_old_callable_edge() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let sources: Vec<_> = (0..INSTALL_LIMIT + 1)
        .map(|index| source(&process, &cursor, 8 + index as u64 * 4, 4))
        .collect();
    let mut transition = stop(&process);
    value(&mut transition, target, 77);
    assert!(!transition.drain_links().unwrap());
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    assert!(process.try_service_links().unwrap());
    assert!(pending(&process).is_empty());
    let hcq = publish(&process, &cursor, &[4], Tier::Hcq);
    assert_eq!(pending(&process).len(), INSTALL_LIMIT + 1);
    // Publication visits the intrusive target bucket, independent of the
    // original source insertion order. Remember its last queued successor.
    let last = *pending(&process).last().unwrap();
    let (src, pc, old) = {
        let state = process.lock();
        let record = state.units.links.records.get(last.0).unwrap();
        let source = state.units.records.get(record.source.0).unwrap();
        (
            record.source,
            source.code.entries[0].key.pc.get(),
            source.static_sites[0].callable.unwrap(),
        )
    };
    assert!(process.try_service_links().unwrap());
    assert_eq!(pending(&process), vec![last]);
    {
        let state = process.lock();
        let site = &state.units.records.get(src.0).unwrap().static_sites[0];
        assert_eq!(site.link, Some(last.0));
        assert_eq!(site.callable, Some(old));
        assert!(state.units.links.records.get(old).unwrap().installed);
        assert!(!state.units.links.records.get(last.0).unwrap().installed);
    }
    assert_eq!(execute_at(&process, pc), 77);
    assert!(process.try_service_links().unwrap());
    assert!(pending(&process).is_empty());
    assert!(process.lock().units.links.records.get(old).is_none());
    assert_eq!(execute_at(&process, pc), 42);
    for src in sources {
        let state = process.lock();
        let site = &state.units.records.get(src.0).unwrap().static_sites[0];
        assert_eq!(site.link, site.callable);
        assert!(
            state
                .units
                .links
                .records
                .get(site.link.unwrap())
                .unwrap()
                .installed
        );
    }
    // Withdrawing HCQ must restore every unsafe branch even though only the
    // first 4096 baseline links can be installed during this stop.
    process.retire_unit(hcq).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.drain_links().unwrap());
    let deferred = pending(&process);
    assert_eq!(deferred.len(), 1);
    let pc = {
        let state = process.lock();
        let link = state.units.links.records.get(deferred[0].0).unwrap();
        assert_eq!(link.target, target);
        let source = state.units.records.get(link.source.0).unwrap();
        assert!(source.static_sites[0].callable.is_none());
        source.code.entries[0].key.pc.get()
    };
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    assert_eq!(execute_at(&process, pc), 42); // Correct source fallback, not HCQ.
    assert!(process.try_service_links().unwrap());
    assert_eq!(execute_at(&process, pc), 77); // Reinstalled LCQ baseline.
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn install_limit_spans_batches_and_owners_but_never_limits_safety_unlinks() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    // Use the real 4096-record boundary, actual published patchpoints and
    // island reservations, not a test-only limit or fabricated registry state.
    let pcs: Vec<_> = (0..INSTALL_LIMIT + 3)
        .map(|index| if index == 0 { 0 } else { 8 + index as u64 * 4 })
        .collect();
    let sources: Vec<_> = pcs
        .iter()
        .map(|pc| source(&process, &cursor, *pc, 4))
        .collect();
    let mut transition = stop(&process);
    let handles: Vec<_> = sources
        .iter()
        .map(|src| {
            let prepared = transition.prepare_link(*src, 0, target, 0, 0).unwrap();
            transition.register_link(prepared).unwrap()
        })
        .collect();
    let ticket = process.request(Reason::LinkPatch).unwrap();
    value(&mut transition, target, 77);
    // Explicit installation shares the drain's quota; neither entry point may
    // install record 4097 before the stop ends.
    assert!(transition.install_link(handles[0]).unwrap());
    assert!(!transition.install_link(handles[0]).unwrap());
    assert!(!transition.drain_links().unwrap());
    assert_eq!(pending(&process), handles[INSTALL_LIMIT..]);
    assert_eq!(process.lock().link_install_attempts, INSTALL_LIMIT);
    assert!(!transition.install_link(handles[INSTALL_LIMIT]).unwrap());
    assert!(!transition.drain_links().unwrap());
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert!(!ticket.is_complete().unwrap());

    drop(transition);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.drain_links().unwrap());
    assert!(!transition.install_link(handles[INSTALL_LIMIT]).unwrap());
    assert_eq!(pending(&process), handles[INSTALL_LIMIT..]);
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();

    // A requester joins AFTER deferral was acknowledged. Reopening cannot
    // ignore it, even though installation already exhausted the stop budget.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                process.retire_unit(sources[1]).unwrap();
            })
            .join()
            .unwrap();
    });
    assert!(!transition.try_reopen().unwrap());
    assert!(!transition.drain_links().unwrap());
    assert!(
        process
            .lock()
            .units
            .links
            .records
            .get(handles[1].0)
            .is_none()
    );
    assert_eq!(pending(&process), handles[INSTALL_LIMIT..]);
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(!ticket.is_complete().unwrap());
    assert_eq!(execute_at(&process, pcs[0]), 77);
    assert_eq!(execute_at(&process, pcs[INSTALL_LIMIT]), 42);
    drop(transition);

    // The retained request, without a new registration/request sequence,
    // starts the next stop through the canonical execution service. FIFO
    // leftovers now install with fresh capacity, and that service reopens.
    assert!(process.try_service_links().unwrap());
    assert!(pending(&process).is_empty());
    assert!(ticket.is_complete().unwrap());
    {
        let state = process.lock();
        assert_eq!(
            handles
                .iter()
                .filter(|handle| {
                    state
                        .units
                        .links
                        .records
                        .get(handle.0)
                        .is_some_and(|record| record.installed)
                })
                .count(),
            INSTALL_LIMIT + 2
        );
    }
    process.retire_unit(target).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    // More than 4096 installed incoming edges must ALL be restored, not
    // deferred. Safety drains do not debit the performance-attempt counter.
    assert!(transition.drain_links().unwrap());
    assert!(process.lock().units.links.records.is_empty());
    assert_eq!(process.lock().link_install_attempts, 0);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute_at(&process, pcs[0]), 42);
    assert_eq!(execute_at(&process, pcs[INSTALL_LIMIT]), 42);
}

#[test]
fn canonical_link_service_defers_after_the_real_install_limit() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let sources: Vec<_> = (0..INSTALL_LIMIT + 1)
        .map(|index| source(&process, &cursor, 8 + index as u64 * 4, 4))
        .collect();
    let mut transition = stop(&process);
    for src in sources {
        let prepared = transition.prepare_static_link(src, 0).unwrap().unwrap();
        transition.register_link(prepared).unwrap();
    }
    let ticket = process.request(Reason::LinkPatch).unwrap();
    drop(transition);
    assert!(process.try_service_links().unwrap());
    assert_eq!(pending(&process).len(), 1);
    assert!(!ticket.is_complete().unwrap());
    assert_eq!(process.lock().phase, crate::lifetime::Phase::Open);
    assert!(process.try_service_links().unwrap());
    assert!(pending(&process).is_empty());
    assert!(ticket.is_complete().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn drain_prioritizes_retirement_and_explicit_stale_attempts_are_counted() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let first = source(&process, &cursor, 0, 4);
    let second = source(&process, &cursor, 8, 4);
    let mut transition = stop(&process);
    let mut handles = Vec::new();
    for src in [first, second] {
        let prepared = transition.prepare_link(src, 0, target, 0, 0).unwrap();
        handles.push(transition.register_link(prepared).unwrap());
    }
    process.retire_unit(target).unwrap();
    // A direct stale preparation consumes one record attempt, but the drain
    // cancels the other through safety adjacency before attempting to emit it.
    assert!(!transition.install_link(handles[0]).unwrap());
    assert_eq!(process.lock().link_install_attempts, 1);
    assert!(transition.drain_links().unwrap());
    assert_eq!(process.lock().link_install_attempts, 1);
    assert!(pending(&process).is_empty());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute_at(&process, 0), 42);
    assert_eq!(transition.drain_links(), Err(Error::Closed));
}
