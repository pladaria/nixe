use super::*;
use crate::abi::InstructionKey;
use crate::lifetime::background::Observation;
use std::time::Instant;

fn sample(
    reader: &mut Reader,
    memory: &ExecutionMemory,
    samples: &mut Samples,
    pc: u64,
    target: u64,
) {
    let mut state = A64State::default();
    state.set_pc(pc);
    state.general_register_storage_mut()[2] = target;
    state.general_register_storage_mut()[30] = target;
    // Take a real native boundary sample with a short slice. Native fragments
    // may overshoot that slice, but cannot run an unbounded indirect loop.
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(1, 2).unwrap());
    let result = unsafe {
        invocation::run(
            samples,
            reader,
            &mut frame,
            memory,
            &mut WorkerFaultContext::register().unwrap(),
            &mut ExclusiveMonitorState::default(),
            key(pc),
        )
    };
    // Concurrent publication may close admission or the native continuation.
    // A skipped observation is retried by the caller's next real execution.
    if matches!(
        result,
        Err(invocation::Error::Lifetime(lifetime::Error::Closed))
    ) {
        return;
    }
    let exit = result.unwrap().unwrap();
    assert!(matches!(exit, invocation::Exit::Native { .. }));
}

#[test]
fn native_boundary_waits_for_profitable_heat_then_exports_late_interior_entry() {
    for interior in [false, true] {
        let (process, memory, mut reader) = setup();
        let (root, source, target) = if interior {
            // The executing source is inside a larger HCQ body, not its first
            // entry. Demand a previously internal instruction after promotion.
            promote_at(&process, &memory, &mut reader, 0x1000);
            super::super::lifecycle::demand(&process, &memory, &mut reader, 0x2004);
            (0x2000, 0x2004, 0x2004)
        } else {
            promote_at(&process, &memory, &mut reader, 0x2000);
            promote_at(&process, &memory, &mut reader, 0x1000);
            (0x1000, 0x1004, 0x2000)
        };
        let old = payload(&mut reader, root).unwrap().hcq().unwrap().family;
        // Without a pool the same native observations saturate heat, but do
        // not create work, replace code or disable the existing direct path.
        let mut no_workers = Samples::new();
        for _ in 0..8 {
            sample(&mut reader, &memory, &mut no_workers, root, target);
        }
        assert_eq!(
            payload(&mut reader, root).unwrap().hcq().unwrap().family,
            old
        );
        assert_eq!(
            no_workers
                .boundary_snapshot(
                    InstructionKey::new(key(source)).unwrap(),
                    InstructionKey::new(key(target)).unwrap()
                )
                .unwrap()
                .1,
            4
        );
        let memory = Arc::new(memory);
        let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
        let (finished, done) = mpsc::channel();
        let (started, accepted) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(Some(wait));
        let mut workers = Workers::start(
            1,
            Arc::clone(&process),
            move |resources, work| {
                let Observation::Reshape {
                    source_block,
                    snapshot,
                } = work.observation()
                else {
                    panic!("native boundary must admit a reshape, not a seed");
                };
                assert_eq!(source_block, key(root));
                assert_eq!(
                    snapshot.key.source,
                    InstructionKey::new(key(source)).unwrap()
                );
                assert_eq!(
                    snapshot.key.target,
                    InstructionKey::new(key(target)).unwrap()
                );
                started.send(()).unwrap();
                if let Some(wait) = wait.lock().unwrap().take() {
                    wait.recv_timeout(Duration::from_secs(10)).unwrap();
                }
                let result = consumer(resources, work);
                finished.send(result.is_ok()).unwrap();
                result
            },
            None,
        )
        .unwrap()
        .unwrap();
        let mut samples = Samples::new();
        for score in 1..4 {
            sample(&mut reader, &memory, &mut samples, root, target);
            assert_eq!(
                samples
                    .boundary_snapshot(
                        InstructionKey::new(key(source)).unwrap(),
                        InstructionKey::new(key(target)).unwrap()
                    )
                    .unwrap()
                    .1,
                score
            );
            assert!(matches!(
                accepted.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
            assert_eq!(
                payload(&mut reader, root).unwrap().hcq().unwrap().family,
                old
            );
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            // Retries, if admission loses a nonblocking race, are driven only
            // by actual native observations. No test-side admission/queue call.
            sample(&mut reader, &memory, &mut samples, root, target);
            match accepted.try_recv() {
                Ok(()) => break,
                Err(mpsc::TryRecvError::Empty) => {}
                Err(error) => panic!("worker disconnected: {error}"),
            }
            process.try_service_links().unwrap();
            assert!(Instant::now() < deadline, "sampled replacement timed out");
            std::thread::yield_now();
        }
        release.send(()).unwrap();
        let mut completed = done.recv_timeout(Duration::from_secs(10)).unwrap();
        while !completed {
            sample(&mut reader, &memory, &mut samples, root, target);
            process.try_service_links().unwrap();
            if let Ok(result) = done.try_recv() {
                completed = result;
            }
            assert!(
                Instant::now() < deadline,
                "profitable replacement timed out"
            );
            std::thread::yield_now();
        }
        process.try_service_links().unwrap();
        let new = payload(&mut reader, root).unwrap().hcq().unwrap().family;
        assert_ne!(new, old);
        assert_eq!(
            payload(&mut reader, target).unwrap().hcq().unwrap().family,
            new
        );
        if interior {
            // A real interior observation must keep the reachable prefix,
            // rather than shrink here and grow it back on the next sample.
            assert_eq!(
                payload(&mut reader, 0x1000).unwrap().hcq().unwrap().family,
                new
            );
        }
        // Exercise the replacement through both prefix and interior ingress.
        run(&process, &memory, &mut reader, 0x3000, 3);
        run(&process, &memory, &mut reader, 0x4000, 2);
        workers.shutdown().unwrap();
        assert!(process.background_failure().is_none());
        assert!(process.try_shutdown().unwrap());
    }
}

#[test]
fn optimized_returns_stay_linked_without_creating_reshape_jobs() {
    let (process, memory, mut reader) = setup();
    promote_at(&process, &memory, &mut reader, 0x2000);
    promote_at(&process, &memory, &mut reader, 0x7000);
    let before = payload(&mut reader, 0x7000).unwrap();
    let memory = Arc::new(memory);
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
    let (finished, done) = mpsc::channel();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let mut workers = Workers::start(
        1,
        Arc::clone(&process),
        move |resources, work| {
            observed.fetch_add(1, Ordering::Relaxed);
            let result = consumer(resources, work);
            assert!(result.is_ok(), "{result:?}");
            finished.send(()).unwrap();
            result
        },
        None,
    )
    .unwrap()
    .unwrap();
    // External returns do not grow regions into arbitrary callers. Internal
    // return edges are inferred from captured calls; neither native polls
    // nor canonical exits should submit an optimizer job for these fallbacks.
    let mut other = process.register().unwrap();
    for reader in [&mut reader, &mut other] {
        let mut samples = Samples::new();
        for _ in 0..12 {
            sample(reader, &memory, &mut samples, 0x7000, 0x2000);
        }
        assert!(
            samples
                .boundary_snapshot(
                    InstructionKey::new(key(0x7000)).unwrap(),
                    InstructionKey::new(key(0x2000)).unwrap()
                )
                .is_none()
        );
        assert_eq!(payload(reader, 0x7000).unwrap(), before);
        run(&process, &memory, reader, 0x7000, 2);
    }
    workers.shutdown().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(done.try_recv().is_err());
    assert!(process.background_failure().is_none());
    assert!(process.try_shutdown().unwrap());
}
