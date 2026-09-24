use super::*;

fn sample(thread: &mut JitThread, native: &mut NativeWorker) {
    sample_at(thread, native, PC);
}

fn sample_at(thread: &mut JitThread, native: &mut NativeWorker, pc: GuestVirtualAddress) {
    let mut state = A64State::default();
    state.set_pc(pc.get());
    let result = thread.invoke(
        &mut crate::ReturnStack::default(),
        native,
        &mut state,
        PollBudget::new(1, 2).unwrap(),
        &VcpuEventState::default(),
    );
    let (exit, _) = match result {
        Err(invocation::Error::Lifetime(lifetime::Error::Closed)) => return,
        result => result.unwrap(),
    };
    assert!(matches!(exit, Some(invocation::Exit::Native { .. })));
}

fn promoted(thread: &mut JitThread) -> bool {
    promoted_at(thread, PC)
}

fn promoted_at(thread: &mut JitThread, pc: GuestVirtualAddress) -> bool {
    thread.process.lifetime.try_service_links().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 10).unwrap());
    let key = thread.key(pc).unwrap();
    match unsafe { thread.reader.admit(&mut frame, key) } {
        Ok(Some(invocation)) => invocation.payload().hcq().is_some(),
        Err(lifetime::Error::Closed) => false,
        other => panic!("unexpected admission: {:?}", other.err()),
    }
}

#[test]
fn native_sampling_automatically_promotes_only_after_the_seed_threshold() {
    let process = Arc::new(
        JitProcess::with_workers(cpu(), memory(DirectBackendPolicy::Required), 1).unwrap(),
    );
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut native = NativeWorker::default();
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    for score in 1..8 {
        sample(&mut thread, &mut native);
        assert_eq!(
            thread
                .samples
                .seed_snapshot(thread.key(PC).unwrap())
                .unwrap()
                .1,
            score
        );
        assert!(!promoted(&mut thread));
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while !promoted(&mut thread) {
        assert!(
            Instant::now() < deadline,
            "automatic HCQ promotion timed out"
        );
        // Contended try-lock admission may defer an observation. Only further
        // native samples retry; the test never enqueues or publishes a unit.
        match thread.process.lifetime.try_service_links() {
            Ok(_) => sample(&mut thread, &mut native),
            Err(error) => panic!("maintenance: {error}"),
        }
        std::thread::yield_now();
    }
    breakpoint(&mut thread, &mut native, 1);
    assert!(process.try_shutdown().unwrap());
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Joined
    ));
    drop(thread);
    native.finish().unwrap();
}

#[test]
fn zero_worker_process_samples_without_creating_compiler_or_queue() {
    let built = std::cell::Cell::new(false);
    let process = Arc::new(
        JitProcess::with_compiler(cpu(), memory(DirectBackendPolicy::Required), 0, |_, _| {
            built.set(true);
            Ok(|_: &mut Resources, _: Work<'_>| -> Result<(), CompileError> { unreachable!() })
        })
        .unwrap(),
    );
    assert!(!built.get());
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Joined
    ));
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut native = NativeWorker::default();
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    for _ in 0..16 {
        sample(&mut thread, &mut native);
        assert!(!promoted(&mut thread));
    }
    assert_eq!(
        thread
            .samples
            .seed_snapshot(thread.key(PC).unwrap())
            .unwrap()
            .1,
        8
    );
    assert!(process.try_shutdown().unwrap());
    drop(thread);
    native.finish().unwrap();
}

#[test]
fn failed_memory_binding_joins_the_new_pool_without_stopping_the_bound_process() {
    let memory = memory(DirectBackendPolicy::Required);
    let first = JitProcess::with_workers(cpu(), memory.clone(), 1).unwrap();
    let captures = Arc::new(());
    let retained = captures.clone();
    let failed = JitProcess::with_compiler(cpu(), memory, 2, |_, _| {
        Ok(
            move |_: &mut Resources, _: Work<'_>| -> Result<(), CompileError> {
                let _keep = &retained;
                panic!("construction exposes no jobs")
            },
        )
    });
    assert!(failed.is_err());
    assert_eq!(Arc::strong_count(&captures), 1); // All spawned consumers joined/dropped.
    let first = Arc::new(first);
    let mut thread = JitThread::new(first.clone()).unwrap();
    let mut native = NativeWorker::default();
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    breakpoint(&mut thread, &mut native, 1);
    assert!(first.try_shutdown().unwrap());
    drop(thread);
    native.finish().unwrap();
}

#[test]
fn independent_native_seeds_use_two_workers_while_cold_lcq_demand_continues() {
    use crate::lifetime::background::Observation;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let memory = memory(DirectBackendPolicy::Required);
    for offset in [16, 32] {
        memory
            .overwrite_mapped_ram(
                SPACE,
                PC.checked_add(offset).unwrap(),
                &[0x1f, 0x20, 0x03, 0xd5, 0x20, 0, 0x20, 0xd4],
            )
            .unwrap();
    }
    let (started, entered) = mpsc::channel();
    let (release_a, wait_a) = mpsc::channel();
    let (release_b, wait_b) = mpsc::channel();
    let process = Arc::new(
        JitProcess::with_compiler(cpu(), memory, 2, |size, memory| {
            let compile = crate::hcq::worker::consumer(
                if cfg!(target_arch = "x86_64") {
                    HostAbi::X86_64
                } else {
                    HostAbi::Aarch64
                },
                size,
                memory,
            )?;
            let ordinal = AtomicUsize::new(0);
            let waits = [Mutex::new(wait_a), Mutex::new(wait_b)];
            Ok(move |resources: &mut Resources, work: Work<'_>| {
                let index = ordinal.fetch_add(1, Ordering::Relaxed);
                if index < 2 {
                    let Observation::Seed(seed) = work.observation() else {
                        panic!("seed expected")
                    };
                    started
                        .send((
                            seed.key.pc,
                            std::ptr::from_mut(&mut resources.context) as usize,
                        ))
                        .unwrap();
                    waits[index]
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(10))
                        .map_err(|error| {
                            Error::internal(format!("test worker rendezvous: {error}"))
                        })?;
                }
                compile(resources, work)
            })
        })
        .unwrap(),
    );
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut native = NativeWorker::default();
    let seeds = [PC, PC.checked_add(16).unwrap()];
    for pc in seeds {
        assert!(matches!(thread.demand(pc).unwrap(), Demand::Ready));
    }
    let mut contexts = Vec::new();
    for pc in seeds {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(
                Instant::now() < deadline,
                "worker did not accept sampled seed"
            );
            sample_at(&mut thread, &mut native, pc);
            if let Ok((accepted, context)) = entered.try_recv() {
                assert_eq!(accepted, pc);
                contexts.push(context);
                break;
            }
            std::thread::yield_now();
        }
    }
    assert_ne!(contexts[0], contexts[1]);
    // Both consumers are paused with accepted Work. This required LCQ miss
    // completes without either background compiler being released.
    let cold = PC.checked_add(32).unwrap();
    assert!(matches!(thread.demand(cold).unwrap(), Demand::Ready));
    sample_at(&mut thread, &mut native, cold);
    release_a.send(()).unwrap();
    release_b.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "parallel seeds did not promote");
        let mut complete = true;
        for pc in seeds {
            if !promoted_at(&mut thread, pc) {
                complete = false;
                sample_at(&mut thread, &mut native, pc);
            }
        }
        if complete {
            break;
        }
        // A publication-table race may cancel one attempt. Further samples
        // retry through production admission, never a test-side queue write.
        std::thread::yield_now();
    }
    for pc in seeds {
        sample_at(&mut thread, &mut native, pc);
    }
    assert!(process.try_shutdown().unwrap());
    drop(thread);
    native.finish().unwrap();
}
