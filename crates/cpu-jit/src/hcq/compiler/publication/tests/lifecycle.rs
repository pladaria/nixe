use super::*;
use crate::abi::DispatchPayload;

#[test]
fn process_owned_staged_hcq_cancels_during_shutdown_with_a_memory_hold() {
    use crate::lifetime::background::workers::{CompileError, Resources};
    crate::engine::tests::background::staged_worker_shutdown(|memory, ready, wait| {
        let compiler = Compiler::new(host(), 0x10000).unwrap();
        let wait = Mutex::new(wait);
        let calls = AtomicUsize::new(0);
        move |resources: &mut Resources, work: Work<'_>| {
            assert_eq!(
                calls.fetch_add(1, Ordering::Relaxed),
                0,
                "stop must drain queued work"
            );
            let frozen = work
                .reserve_candidate(Graph::discover(&work).unwrap())?
                .freeze()?;
            let pause = || {
                ready.send(()).unwrap();
                wait.lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(30))
                    .unwrap();
            };
            let observed = Observed {
                memory: &memory,
                runs: Mutex::new(Vec::new()),
                validations: AtomicUsize::new(0),
                // One captured run: capture, pre-install validation, then the
                // final validation after actual W^X/directory preparation.
                during_validation: Some((2, &pause)),
            };
            let result = compiler.publish(
                &mut resources.context,
                &mut resources.frontend,
                &frozen,
                &observed,
            );
            assert!(matches!(result, Err(Failure::Cancelled)), "{result:?}");
            Err(CompileError::Cancelled)
        }
    });
}

pub(super) fn payload(reader: &mut Reader, pc: u64) -> Option<DispatchPayload> {
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    unsafe { reader.admit(&mut frame, key(pc)) }
        .unwrap()
        .map(|invocation| invocation.payload().clone())
}

fn promote(process: &Lifetime, memory: &ExecutionMemory, reader: &mut Reader) -> UnitHandle {
    promote_at(process, memory, reader, 0x1000)
}

pub(super) fn promote_at(
    process: &Lifetime,
    memory: &ExecutionMemory,
    reader: &mut Reader,
    pc: u64,
) -> UnitHandle {
    let work = work_at(process, reader, pc);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let handle = Compiler::new(host(), 0x10000)
        .unwrap()
        .publish(
            &mut Context::new(),
            &mut FunctionBuilderContext::new(),
            &frozen,
            memory,
        )
        .unwrap();
    drop(frozen);
    drop(work);
    process.try_service_links().unwrap();
    handle
}

pub(super) fn retire(process: &Lifetime, handle: UnitHandle) {
    process.retire_unit(handle).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(transition.drain_links().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

pub(super) fn demand(process: &Lifetime, memory: &ExecutionMemory, reader: &mut Reader, pc: u64) {
    let Request::Owner(claim) = reader.claim(key(pc)).unwrap() else {
        panic!("expected a real LCQ demand")
    };
    Lcq::for_arena(host(), 0x10000)
        .unwrap()
        .publish(
            Compilation::capture(claim, memory).unwrap(),
            process,
            process.executable_cache(),
            memory,
        )
        .unwrap();
    process.try_service_links().unwrap();
}

pub(super) fn run(
    process: &Lifetime,
    memory: &ExecutionMemory,
    reader: &mut Reader,
    pc: u64,
    expected: u64,
) {
    run_to(process, memory, reader, pc, expected, (0x5000, 7));
}

pub(super) fn run_to(
    process: &Lifetime,
    memory: &ExecutionMemory,
    reader: &mut Reader,
    pc: u64,
    expected: u64,
    stop: (u64, u16),
) {
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut state = A64State::default();
    state.set_pc(pc);
    state.general_register_storage_mut()[2] = 0x5000;
    state.general_register_storage_mut()[3] = 0x1000;
    state.general_register_storage_mut()[30] = 0x2000;
    let mut returns = crate::ReturnStack::default();
    if pc == 0x7000 {
        // Model the guest-thread continuation retained from an earlier call.
        // RET must consume the prediction, not merely use indirect dispatch.
        returns.entries[0] = key(0x2000).into();
        returns.head = 1;
        returns.depth = 1;
    }
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap())
        .with_return_stack(&mut returns);
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            reader,
            &mut frame,
            memory,
            &mut worker,
            &mut ExclusiveMonitorState::default(),
            key(pc),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Native { guest, .. } = exit else {
        panic!("expected a native breakpoint exit")
    };
    assert_eq!(guest.pc.get(), stop.0);
    assert_eq!(guest.kind, EdgeKind::Breakpoint(stop.1));
    assert_eq!(state.general_register_storage_mut()[0], expected);
    assert_eq!(returns.depth, 0);
    process.try_service_links().unwrap();
}

#[test]
fn hcq_publication_late_interior_demand_keeps_a_separate_callable_baseline() {
    let (process, memory, mut reader) = setup();
    let handle = promote(&process, &memory, &mut reader);
    let promoted = payload(&mut reader, 0x1000).unwrap();
    assert!(payload(&mut reader, 0x1004).is_none());
    demand(&process, &memory, &mut reader, 0x1004);
    let late = payload(&mut reader, 0x1004).unwrap();
    assert!(late.hcq().is_none());
    assert!(late.lcq().is_some());
    assert_eq!(payload(&mut reader, 0x1000).unwrap(), promoted);
    assert_eq!(process.snapshot(handle).unwrap().entries.len(), 2);
    run(&process, &memory, &mut reader, 0x1004, 2);
    run(&process, &memory, &mut reader, 0x1000, 3);

    retire(&process, handle);
    assert_eq!(payload(&mut reader, 0x1004).unwrap().lcq(), late.lcq());
    run(&process, &memory, &mut reader, 0x1004, 2);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}

#[test]
fn hcq_publication_withdrawal_restores_real_static_pic_and_return_paths() {
    let (process, memory, mut first) = setup();
    let mut second = process.register().unwrap();
    let baselines = [0x1000, 0x2000].map(|pc| payload(&mut first, pc).unwrap().lcq());
    let routes = [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)];
    // Warm old baseline roots before promotion, then optimized roots on both
    // readers. Repeat to execute cached PIC/return ways as well as resolution.
    for (pc, expected) in routes {
        run(&process, &memory, &mut first, pc, expected);
    }
    let handle = promote(&process, &memory, &mut first);
    let retained = process.snapshot(handle).unwrap();
    for _ in 0..2 {
        for reader in [&mut first, &mut second] {
            for (pc, expected) in routes {
                run(&process, &memory, reader, pc, expected);
            }
        }
    }
    for (index, pc) in [0x1000, 0x2000].into_iter().enumerate() {
        let entry = payload(&mut first, pc).unwrap();
        assert!(entry.hcq().is_some());
        assert_eq!(entry.lcq(), baselines[index]);
    }
    retire(&process, handle);
    assert!(matches!(
        process.snapshot(handle),
        Err(lifetime::Error::StaleUnit)
    ));
    for (index, pc) in [0x1000, 0x2000].into_iter().enumerate() {
        let entry = payload(&mut first, pc).unwrap();
        assert!(entry.hcq().is_none());
        assert_eq!(entry.preferred(), baselines[index]);
    }
    assert_eq!(process.reclaim_units().unwrap(), 0);
    drop(retained);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    // The optimized span has now actually been freed. Stale static/PIC/return
    // addresses cannot be made harmless merely by retaining its executable bytes.
    for reader in [&mut first, &mut second] {
        for (pc, expected) in routes {
            run(&process, &memory, reader, pc, expected);
        }
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}

#[test]
fn hcq_publication_pressure_defers_without_rejecting_seed_or_losing_baselines() {
    let (process, memory, mut reader) = setup();
    let original = payload(&mut reader, 0x1000).unwrap();
    let work = work(&process, &mut reader);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let cache = process.executable_cache();
    let before = cache.usage().unwrap();
    // Capacity disappears after backend staging, not at initial admission.
    // Exercise real accounting without allocating hundreds of MiB.
    let pressure = Mutex::new(None);
    let fill = || {
        *pressure.lock().unwrap() = Some(
            cache
                .charge_metadata(crate::executable::SOFT_BYTES - before.total(), Tier::Lcq)
                .unwrap(),
        );
    };
    let view = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((2, &fill)),
    };
    let compiler = Compiler::new(host(), 0x10000).unwrap();
    let mut context = Context::new();
    let mut frontend = FunctionBuilderContext::new();
    let result = compiler.publish(&mut context, &mut frontend, &frozen, &view);
    assert!(matches!(result, Err(Failure::Deferred)), "{result:?}");
    assert_eq!(view.validations.load(Ordering::Relaxed), 4);
    assert_eq!(
        cache.usage().unwrap().total(),
        crate::executable::SOFT_BYTES
    );
    assert!(context.compiled_code().is_none());
    assert_eq!(context.func.layout.blocks().count(), 0);
    assert_eq!(payload(&mut reader, 0x1000).unwrap(), original);
    drop(pressure.lock().unwrap().take());
    assert_eq!(cache.usage().unwrap(), before);
    frozen.check().unwrap();
    run(&process, &memory, &mut reader, 0x1000, 3);
    // Deferral neither permanently rejects the version nor poisons scratch.
    compiler
        .publish(&mut context, &mut frontend, &frozen, &memory)
        .unwrap();
    drop(frozen);
    drop(work);
    process.try_service_links().unwrap();
    assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_some());
    run(&process, &memory, &mut reader, 0x1000, 3);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
}

#[test]
fn hcq_publication_memory_invalidation_retargets_real_ingress_to_new_lcq_bytes() {
    let (process, memory, mut reader) = setup();
    let handle = promote(&process, &memory, &mut reader);
    let retained = process.snapshot(handle).unwrap();
    for _ in 0..2 {
        for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
            run(&process, &memory, &mut reader, pc, expected);
        }
    }
    memory
        .overwrite_mapped_ram(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            &0x91000c00u32.to_le_bytes(), // ADD X0,X0,#3
        )
        .unwrap();
    process.try_service_links().unwrap();
    assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_none());
    assert!(payload(&mut reader, 0x2000).is_none());
    assert!(matches!(
        process.snapshot(handle),
        Err(lifetime::Error::StaleUnit)
    ));
    // Remove the last compiler snapshot before installing the new code so the
    // test cannot accidentally execute stale HCQ from a retained native span.
    drop(retained);
    assert!(process.reclaim_units().unwrap() > 0);
    demand(&process, &memory, &mut reader, 0x2000);
    for (pc, expected) in [(0x3000, 4), (0x4000, 3), (0x6000, 4), (0x7000, 3)] {
        run(&process, &memory, &mut reader, pc, expected);
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}

#[test]
fn hcq_publication_shutdown_after_preparation_cancels_and_releases_all_storage() {
    let (process, memory, mut reader) = setup();
    let work = work(&process, &mut reader);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let shutdown = || {
        process.request_shutdown().unwrap();
        assert!(!process.try_shutdown().unwrap()); // The compiler still owns inputs.
    };
    let view = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((4, &shutdown)),
    };
    let mut context = Context::new();
    let result = Compiler::new(host(), 0x10000).unwrap().publish(
        &mut context,
        &mut FunctionBuilderContext::new(),
        &frozen,
        &view,
    );
    assert!(matches!(result, Err(Failure::Cancelled)), "{result:?}");
    assert!(context.compiled_code().is_none());
    assert_eq!(context.func.layout.blocks().count(), 0);
    drop(frozen);
    drop(work);
    assert!(process.try_shutdown().unwrap());
    let cache = process.executable_cache().clone();
    assert_eq!(cache.usage().unwrap().committed, 0);
    drop(reader);
    drop(memory);
    assert_eq!(Arc::strong_count(&process), 1);
    drop(process);
    assert_eq!(Arc::strong_count(&cache), 1);
    // The retained Cache itself is charged until this last Arc is dropped.
    assert_eq!(
        cache.usage().unwrap().metadata,
        size_of::<Cache>() + 2 * size_of::<usize>()
    );
}
