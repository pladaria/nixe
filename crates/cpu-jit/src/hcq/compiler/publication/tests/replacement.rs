use super::lifecycle::{demand, payload, promote_at, run};
use super::negative::reshape;
use super::*;

const ROUTES: [(u64, u64); 4] = [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)];

fn publish(
    frozen: &Frozen<'_, '_>,
    memory: &(impl ExecutableMemory + MemoryInvalidationSource),
) -> Result<UnitHandle, Failure> {
    Compiler::new(host(), 0x10000).unwrap().publish(
        &mut Context::new(),
        &mut FunctionBuilderContext::new(),
        frozen,
        memory,
    )
}

#[test]
fn real_replacement_merges_zero_one_two_families_and_executes_all_ingress_paths() {
    for count in 0..=2 {
        let (process, memory, mut first) = setup();
        let mut second = process.register().unwrap();
        let baselines = [0x1000, 0x2000].map(|pc| payload(&mut first, pc).unwrap().lcq());
        let mut predecessors = Vec::new();
        // Optimize the target first so the source remains a separate family.
        if count >= 1 {
            predecessors.push(promote_at(&process, &memory, &mut first, 0x2000));
        }
        if count == 2 {
            predecessors.push(promote_at(&process, &memory, &mut first, 0x1000));
        }
        let pins: Vec<_> = predecessors
            .iter()
            .map(|&unit| process.snapshot(unit).unwrap())
            .collect();
        for _ in 0..2 {
            for reader in [&mut first, &mut second] {
                for (pc, expected) in ROUTES {
                    run(&process, &memory, reader, pc, expected);
                }
            }
        }
        let work = reshape(&process, &mut first, 0x1000, 0x1004, 0x2000);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(frozen.graph().instructions.len(), 4);
        assert_eq!(frozen.entries().len(), 2);
        let successor = publish(&frozen, &memory).unwrap();
        process.try_service_links().unwrap();
        let code = process.snapshot(successor).unwrap();
        for (index, pc) in [0x1000, 0x2000].into_iter().enumerate() {
            let entry = payload(&mut first, pc).unwrap();
            assert_eq!(entry.hcq().unwrap().entry.unit, code.id);
            assert_eq!(entry.lcq(), baselines[index]);
        }
        drop(frozen);
        drop(work);
        assert_eq!(process.reclaim_units().unwrap(), 0); // Compiler snapshots pin both old bodies.
        drop(pins);
        assert_eq!(process.reclaim_units().unwrap(), count);
        for predecessor in predecessors {
            assert!(matches!(
                process.snapshot(predecessor),
                Err(lifetime::Error::StaleUnit)
            ));
        }
        // Old executable spans have actually been released, not merely hidden
        // from dispatch. Warm PIC/return ways on both readers must be safe now.
        for _ in 0..2 {
            for reader in [&mut first, &mut second] {
                for (pc, expected) in ROUTES {
                    run(&process, &memory, reader, pc, expected);
                }
            }
        }
        drop(code);
        assert!(process.try_shutdown().unwrap());
    }
}

#[test]
fn real_replacement_preserves_prefix_and_old_hcq_ingress_when_observed_from_interior() {
    let (process, memory, mut first) = setup();
    let mut second = process.register().unwrap();
    let predecessor = promote_at(&process, &memory, &mut first, 0x1000);
    let retained = process.snapshot(predecessor).unwrap();
    let outer = promote_at(&process, &memory, &mut first, 0x3000);
    let outer_entry = payload(&mut first, 0x3000).unwrap();
    let baseline = payload(&mut first, 0x1000).unwrap().lcq();
    for _ in 0..2 {
        for reader in [&mut first, &mut second] {
            for (pc, expected) in ROUTES {
                run(&process, &memory, reader, pc, expected);
            }
        }
    }
    // An interior observation grows across BR X2 without discarding the
    // predecessor's reachable prefix or its existing native ingress.
    let work = reshape(&process, &mut first, 0x2000, 0x2004, 0x5000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.graph().instructions.len(), 5);
    let successor = publish(&frozen, &memory).unwrap();
    process.try_service_links().unwrap();
    let retained_entry = payload(&mut first, 0x1000).unwrap();
    assert_eq!(
        retained_entry.hcq().unwrap().entry.unit,
        process.snapshot(successor).unwrap().id
    );
    assert_eq!(retained_entry.lcq(), baseline);
    assert_eq!(payload(&mut first, 0x3000).unwrap(), outer_entry);
    assert!(process.snapshot(outer).is_ok());
    assert_eq!(
        payload(&mut first, 0x2000)
            .unwrap()
            .hcq()
            .unwrap()
            .entry
            .unit,
        process.snapshot(successor).unwrap().id
    );
    drop(frozen);
    drop(work);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    drop(retained);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    for _ in 0..2 {
        for reader in [&mut first, &mut second] {
            for (pc, expected) in ROUTES {
                run(&process, &memory, reader, pc, expected);
            }
        }
    }
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn real_replacement_retains_predecessors_until_the_last_announced_reader_exits() {
    let (process, memory, mut reader) = setup();
    let old = promote_at(&process, &memory, &mut reader, 0x2000);
    let old_code = process.snapshot(old).unwrap();
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, key(0x2000)) }
        .unwrap()
        .unwrap();
    // The compiler runs on a different OS thread, never carrying this vCPU's
    // execution epoch or host FP environment through backend work.
    let successor = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let successor = publish(&frozen, &memory).unwrap();
                assert!(!process.try_service_links().unwrap());
                successor
            })
            .join()
            .unwrap()
    });
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(invocation.payload().hcq().unwrap().entry.unit, old_code.id);
    // The old invocation can still attribute its native PC while publication
    // has already exposed a new body and requested closure.
    let (_, lookup) = invocation.frame_and_faults();
    assert_eq!(
        lookup.unit(old_code.code.allocation.address()).unwrap().id,
        old_code.id
    );
    drop(invocation);
    assert!(process.try_service_links().unwrap());
    drop(frozen);
    drop(work);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    drop(old_code);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert_eq!(
        payload(&mut reader, 0x2000)
            .unwrap()
            .hcq()
            .unwrap()
            .entry
            .unit,
        process.snapshot(successor).unwrap().id
    );
    run(&process, &memory, &mut reader, 0x3000, 3);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn real_replacement_cancels_when_code_changes_after_staging() {
    let (process, memory, mut reader) = setup();
    let old = promote_at(&process, &memory, &mut reader, 0x2000);
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let change = || {
        memory
            .overwrite_mapped_ram(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x2000),
                &0x91000c00u32.to_le_bytes(),
            )
            .unwrap(); // ADD X0,X0,#3
    };
    let view = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((4, &change)),
    };
    assert!(matches!(publish(&frozen, &view), Err(Failure::Cancelled)));
    assert!(view.validations.load(Ordering::Relaxed) >= 5);
    drop(frozen);
    drop(work);
    assert!(matches!(
        process.snapshot(old),
        Err(lifetime::Error::StaleUnit)
    ));
    process.reclaim_units().unwrap();
    demand(&process, &memory, &mut reader, 0x2000);
    assert!(payload(&mut reader, 0x2000).unwrap().hcq().is_none());
    for (pc, expected) in [(0x3000, 4), (0x4000, 3), (0x6000, 4), (0x7000, 3)] {
        run(&process, &memory, &mut reader, pc, expected);
    }
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn real_replacement_prepared_output_is_released_on_pressure_or_shutdown() {
    for shutdown in [false, true] {
        let (process, memory, mut reader) = setup();
        let predecessors = [
            promote_at(&process, &memory, &mut reader, 0x2000),
            promote_at(&process, &memory, &mut reader, 0x1000),
        ];
        let original = [0x1000, 0x2000].map(|pc| payload(&mut reader, pc).unwrap());
        let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let cache = process.executable_cache();
        let pressure = Mutex::new(None);
        let interrupt = || {
            if shutdown {
                process.request_shutdown().unwrap();
                assert!(!process.try_shutdown().unwrap());
            } else {
                *pressure.lock().unwrap() = Some(
                    cache
                        .charge_metadata(
                            crate::executable::SOFT_BYTES - cache.usage().unwrap().total(),
                            Tier::Lcq,
                        )
                        .unwrap(),
                );
            }
        };
        let view = Observed {
            memory: &memory,
            runs: Mutex::new(Vec::new()),
            validations: AtomicUsize::new(0),
            // Both capture runs and post-backend checks have completed;
            // code and replacement metadata are now prepared but unpublished.
            during_validation: Some((4, &interrupt)),
        };
        let mut context = Context::new();
        let mut frontend = FunctionBuilderContext::new();
        let compiler = Compiler::new(host(), 0x10000).unwrap();
        let result = compiler.publish(&mut context, &mut frontend, &frozen, &view);
        if shutdown {
            assert!(matches!(result, Err(Failure::Cancelled)), "{result:?}");
        } else {
            assert!(matches!(result, Err(Failure::Deferred)), "{result:?}");
        }
        assert!(view.validations.load(Ordering::Relaxed) >= 5);
        assert!(context.compiled_code().is_none());
        assert!(context.func.layout.blocks().next().is_none());
        drop(pressure.lock().unwrap().take());
        drop(frozen);
        drop(work);
        if !shutdown {
            for (pc, expected) in [0x1000, 0x2000].into_iter().zip(original) {
                assert_eq!(payload(&mut reader, pc).unwrap(), expected);
            }
            assert_eq!(process.reclaim_units().unwrap(), 0);
            run(&process, &memory, &mut reader, 0x3000, 3);
            // Pressure leaves no negative/reservation behind and does not
            // poison the reusable backend scratch or retire either predecessor.
            let retry = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
            let frozen = retry
                .reserve_candidate(Graph::discover(&retry).unwrap())
                .unwrap()
                .freeze()
                .unwrap();
            compiler
                .publish(&mut context, &mut frontend, &frozen, &memory)
                .unwrap();
            drop(frozen);
            drop(retry);
            process.try_service_links().unwrap();
            // Normal cutover maintenance already reclaimed both predecessors.
            assert_eq!(process.reclaim_units().unwrap(), 0);
            for predecessor in predecessors {
                assert!(matches!(
                    process.snapshot(predecessor),
                    Err(lifetime::Error::StaleUnit)
                ));
            }
            run(&process, &memory, &mut reader, 0x3000, 3);
        }
        assert!(process.try_shutdown().unwrap());
        assert_eq!(cache.usage().unwrap().committed, 0);
    }
}

#[test]
fn real_replacement_mapping_change_joins_pending_cutover_and_invalidates_both_versions() {
    let (process, memory, mut reader) = setup();
    let old = promote_at(&process, &memory, &mut reader, 0x2000);
    for (pc, expected) in ROUTES {
        run(&process, &memory, &mut reader, pc, expected);
    }
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let successor = publish(&frozen, &memory).unwrap();
    // Do not service replacement's pending stop first. The memory authority
    // must join it and withdraw both versions, including newly queued roots.
    memory
        .overwrite_mapped_ram(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            &0x91000c00u32.to_le_bytes(),
        )
        .unwrap();
    process.try_service_links().unwrap();
    for handle in [old, successor] {
        assert!(matches!(
            process.snapshot(handle),
            Err(lifetime::Error::StaleUnit)
        ));
    }
    assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_none());
    assert!(payload(&mut reader, 0x2000).is_none());
    drop(frozen);
    drop(work);
    assert!(process.reclaim_units().unwrap() >= 2);
    demand(&process, &memory, &mut reader, 0x2000);
    for _ in 0..2 {
        for (pc, expected) in [(0x3000, 4), (0x4000, 3), (0x6000, 4), (0x7000, 3)] {
            run(&process, &memory, &mut reader, pc, expected);
        }
    }
    assert!(process.try_shutdown().unwrap());
}
