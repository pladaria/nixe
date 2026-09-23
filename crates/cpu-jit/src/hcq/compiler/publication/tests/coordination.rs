use super::lifecycle::{demand, payload, promote_at, run};
use super::negative::{real_backend_limit, reshape};
use super::*;
use crate::lifetime::{Error, Reason};
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn real_hcq_publication_crosses_closing_and_late_safety_work_without_losing_its_inputs() {
    for replacement in [false, true] {
        let (process, memory, mut reader) = setup();
        let unrelated = promote_at(&process, &memory, &mut reader, 0x7000);
        if replacement {
            promote_at(&process, &memory, &mut reader, 0x2000);
        }
        let original = [0x1000, 0x2000].map(|pc| payload(&mut reader, pc).unwrap());
        let work = if replacement {
            reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000)
        } else {
            work(&process, &mut reader)
        };
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
        let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
            .unwrap()
            .unwrap();
        process.request(Reason::LinkPatch).unwrap();
        let mut transition = process.try_transition().unwrap().unwrap();
        let successor = std::thread::scope(|scope| {
            let (prepared, ready) = mpsc::channel();
            let (finished, result) = mpsc::channel();
            let (memory, frozen) = (&memory, &frozen);
            let publisher = scope.spawn(move || {
                let notify = || prepared.send(()).unwrap();
                let observed = Observed {
                    memory,
                    runs: Mutex::new(Vec::new()),
                    validations: AtomicUsize::new(0),
                    // Two capture checks, two post-backend checks, then the
                    // prepared output's final memory check, while still Closing.
                    during_validation: Some((4, &notify)),
                };
                let outcome = Compiler::new(host(), 0x10000).unwrap().publish(
                    &mut Context::new(),
                    &mut FunctionBuilderContext::new(),
                    frozen,
                    &observed,
                );
                finished.send(outcome).unwrap();
            });
            ready.recv_timeout(Duration::from_secs(10)).unwrap();
            assert!(matches!(result.try_recv(), Err(mpsc::TryRecvError::Empty)));
            assert_eq!(invocation.payload(), &original[0]);
            drop(invocation);
            transition.wait_closed().unwrap();
            let old_batch = transition.batch().unwrap();
            let late = process.retire_unit(unrelated).unwrap();
            old_batch.complete().unwrap();
            assert!(
                !process
                    .maintenance_complete(crate::lifetime::Reason::Eviction, late)
                    .unwrap()
            );
            assert!(!transition.try_reopen().unwrap());
            assert!(matches!(result.try_recv(), Err(mpsc::TryRecvError::Empty)));
            assert!(transition.drain_links().unwrap());
            transition.batch().unwrap().complete().unwrap();
            assert!(
                process
                    .maintenance_complete(crate::lifetime::Reason::Eviction, late)
                    .unwrap()
            );
            assert!(transition.try_reopen().unwrap());
            drop(transition);
            let successor = result
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
            publisher.join().unwrap();
            successor
        });
        drop(frozen);
        drop(work);
        process.try_service_links().unwrap();
        let id = process.snapshot(successor).unwrap().id;
        for (pc, old) in [0x1000, 0x2000].into_iter().zip(original) {
            let current = payload(&mut reader, pc).unwrap();
            assert_eq!(current.lcq(), old.lcq());
            assert_eq!(current.hcq().unwrap().entry.unit, id);
        }
        for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
            run(&process, &memory, &mut reader, pc, expected);
        }
        assert!(process.try_shutdown().unwrap());
        assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
    }
}

#[test]
fn real_hcq_result_and_dynamic_preparation_cannot_revive_a_retired_source_or_clear_new_work() {
    for negative in [false, true] {
        let (process, memory, mut reader) = setup();
        let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let source = frozen
            .graph()
            .units
            .iter()
            .find(|unit| unit.instructions.get(0).unwrap().key.block_key() == key(0x2000))
            .unwrap();
        let source_handle = source.registered_handle().unwrap();
        let map = source
            .states
            .iter()
            .position(|map| map.exit.is_some_and(|exit| exit.kind == EdgeKind::Indirect))
            .unwrap() as u32;
        let bridge = || {
            process
                .prepare_dynamic_bridge(source_handle, map, key(0x5000))
                .unwrap()
                .unwrap()
        };
        let unbuilt = bridge();
        let emitted = bridge().emit().unwrap();
        let next_reader = Mutex::new(process.register().unwrap());
        let next_work = Mutex::new(None);
        let invalidate = || {
            // An eviction must preserve this pinned baseline. A real code
            // mutation instead revokes execution authority despite those pins.
            memory
                .overwrite_mapped_ram(
                    AddressSpaceId::new(1),
                    GuestVirtualAddress::new(0x2000),
                    &0x91000c00u32.to_le_bytes(), // ADD X0,X0,#3
                )
                .unwrap();
            // Replace the logical root as well, so a newer reservation can
            // belong to the same PC before the old Work is destroyed.
            memory
                .overwrite_mapped_ram(
                    AddressSpaceId::new(1),
                    GuestVirtualAddress::new(0x1000),
                    &0x91000400u32.to_le_bytes(),
                )
                .unwrap();
            assert!(process.try_service_links().unwrap());
            let mut reader = next_reader.lock().unwrap();
            demand(&process, &memory, &mut reader, 0x1000);
            demand(&process, &memory, &mut reader, 0x2000);
            *next_work.lock().unwrap() =
                Some(reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000));
        };
        let observed = Observed {
            memory: &memory,
            runs: Mutex::new(Vec::new()),
            validations: AtomicUsize::new(0),
            // Both paths have prepared their result before this callback.
            during_validation: Some((if negative { 2 } else { 4 }, &invalidate)),
        };
        let result = if negative {
            record_backend_rejection(&frozen, real_backend_limit(&frozen), &observed).map(|_| ())
        } else {
            Compiler::new(host(), 0x10000)
                .unwrap()
                .publish(
                    &mut Context::new(),
                    &mut FunctionBuilderContext::new(),
                    &frozen,
                    &observed,
                )
                .map(|_| ())
        };
        assert!(matches!(result, Err(Failure::Cancelled)), "{result:?}");
        assert!(matches!(unbuilt.emit(), Err(Error::StalePublication)));
        assert!(matches!(
            process.prepare_dynamic_bridge(source_handle, map, key(0x5000)),
            Err(Error::StaleUnit)
        ));
        assert!(payload(&mut reader, 0x2000).unwrap().hcq().is_none());
        drop(frozen);
        drop(work); // Exact-token cleanup must not release the newer job.
        assert_eq!(process.reclaim_units().unwrap(), 1); // Old root; bridge still pins its source.
        drop(emitted);
        assert_eq!(process.reclaim_units().unwrap(), 1);
        let next_work = next_work.into_inner().unwrap().unwrap();
        let next = next_work
            .reserve_candidate(Graph::discover(&next_work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        Compiler::new(host(), 0x10000)
            .unwrap()
            .publish(
                &mut Context::new(),
                &mut FunctionBuilderContext::new(),
                &next,
                &memory,
            )
            .unwrap();
        drop(next);
        drop(next_work);
        process.try_service_links().unwrap();
        for (pc, expected) in [(0x3000, 4), (0x4000, 3), (0x6000, 4), (0x7000, 3)] {
            run(&process, &memory, &mut reader, pc, expected);
        }
        assert!(process.try_shutdown().unwrap());
        assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
    }
}

#[test]
fn simultaneous_real_hcq_publishers_refresh_the_directory_without_losing_either_unit() {
    let (process, memory, mut reader) = setup();
    let original = [0x1000, 0x2000].map(|pc| payload(&mut reader, pc).unwrap().lcq());
    let first_work = work(&process, &mut reader);
    let graph = Graph::discover(&first_work).unwrap();
    let second_work = work_at(&process, &mut reader, 0x2000);
    let second = second_work
        .reserve_candidate(Graph::discover(&second_work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let first = first_work
        .reserve_candidate(graph)
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(first.entries().len(), 1);
    assert_eq!(second.entries().len(), 1);
    let staged = std::sync::Barrier::new(2);
    let publish = |frozen: &Frozen<'_, '_>| {
        let meet = || {
            // Both native outputs and directory snapshots exist before either
            // publisher can replace the shared segment's directory table.
            staged.wait();
        };
        let observed = Observed {
            memory: &memory,
            runs: Mutex::new(Vec::new()),
            validations: AtomicUsize::new(0),
            during_validation: Some((2, &meet)),
        };
        Compiler::new(host(), 0x10000)
            .unwrap()
            .publish(
                &mut Context::new(),
                &mut FunctionBuilderContext::new(),
                frozen,
                &observed,
            )
            .unwrap()
    };
    let handles = std::thread::scope(|scope| {
        let a = scope.spawn(|| publish(&first));
        let b = scope.spawn(|| publish(&second));
        [a.join().unwrap(), b.join().unwrap()]
    });
    drop(first);
    drop(second);
    drop(first_work);
    drop(second_work);
    process.try_service_links().unwrap();
    let units = handles.map(|handle| process.snapshot(handle).unwrap());
    assert_ne!(units[0].id, units[1].id);
    assert_eq!(
        units[0].code.allocation.segment,
        units[1].code.allocation.segment
    );
    for ((pc, baseline), unit) in [0x1000, 0x2000].into_iter().zip(original).zip(&units) {
        let entry = payload(&mut reader, pc).unwrap();
        assert_eq!(entry.lcq(), baseline);
        assert_eq!(entry.hcq().unwrap().entry.unit, unit.id);
    }
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    let (_, directory) = invocation.frame_and_faults();
    for unit in &units {
        assert_eq!(
            directory.unit(unit.code.allocation.address()).unwrap().id,
            unit.id
        );
    }
    drop(invocation);
    drop(units);
    for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
        run(&process, &memory, &mut reader, pc, expected);
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}
