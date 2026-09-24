use super::*;
use nixe_memory::{ExecutionMutationObserver, MemoryInvalidationKind};

#[test]
fn shutdown_does_not_join_a_compiler_waiting_on_the_callers_execution_lease() {
    use nixe_cpu::memory::ExecutableMemory;
    let memory = memory(DirectBackendPolicy::Required);
    let (started, running) = mpsc::channel();
    let (capture, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let process = Arc::new(
        JitProcess::with_compiler(cpu(), memory.clone(), 1, |_, memory| {
            Ok(move |_: &mut Resources, work: Work<'_>| {
                started.send(()).unwrap();
                wait.lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(30))
                    .unwrap();
                let image = memory.capture_instructions(
                    SPACE,
                    PC,
                    std::num::NonZeroU16::new(1).unwrap(),
                    &|_, _| false,
                );
                assert_eq!(image.words().len(), 1);
                assert_eq!(image.words()[0].bits, 0xd503201f);
                assert_eq!(work.check(), Err(lifetime::Error::StalePublication));
                Err(CompileError::Cancelled)
            })
        })
        .unwrap(),
    );
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut native = NativeWorker::default();
    enqueue(&mut thread, &mut native);
    running.recv_timeout(Duration::from_secs(30)).unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 10).unwrap());
    let lease = memory.acquire_execution_lease();
    let key = thread.key(PC).unwrap();
    let invocation = unsafe { thread.reader.admit(&mut frame, key) }
        .unwrap()
        .unwrap();
    let (waiting, blocked) = mpsc::channel();
    memory.set_transition_notifier(Some(Arc::new(move || {
        waiting.send(()).unwrap();
    })));
    capture.send(()).unwrap();
    // The actual gate, not a sleep, proves that instruction capture is waiting
    // for this caller's lease. A join here would make releasing it impossible.
    blocked.recv_timeout(Duration::from_secs(30)).unwrap();
    let joining = process.clone();
    let (finished, done) = mpsc::channel();
    let join = std::thread::spawn(move || finished.send(joining.try_shutdown()).unwrap());
    let pending = done.recv_timeout(Duration::from_secs(30));
    // Release before checking the watchdog so a regressed join can still exit
    // and the test never leaves the compiler blocked on a leaked lease.
    drop(invocation);
    drop(lease);
    join.join().unwrap();
    assert_eq!(pending.unwrap(), Ok(false));
    assert!(process.try_shutdown().unwrap());
    assert!(process.lifetime.background_failure().is_none());
    assert_eq!(
        process
            .lifetime
            .executable_cache()
            .usage()
            .unwrap()
            .committed,
        0
    );
    memory.set_transition_notifier(None);
    drop(thread);
    native.finish().unwrap();
}

#[test]
fn failed_owned_worker_preserves_error_and_last_owner_releases_failed_process_storage() {
    for panic in [false, true] {
        let (started, running) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let captures = Arc::new(());
        let retained = captures.clone();
        let process = process(2, move |_, work| {
            let _capture = &retained;
            let crate::lifetime::background::Observation::Seed(observed) = work.observation()
            else {
                panic!("expected seed")
            };
            let source = work.lcq(observed.key).unwrap().unwrap();
            started.send(source.unit.clone()).unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(30))
                .unwrap();
            assert_eq!(source.unit.instructions.get(0).unwrap().bits, 0xd503201f);
            if panic {
                panic!("shutdown race: HCQ compiler failed");
            }
            Err(Error::internal("shutdown race: HCQ compiler failed").into())
        });
        let weak_process = Arc::downgrade(&process);
        let weak_lifetime = Arc::downgrade(&process.lifetime);
        let weak_cache = Arc::downgrade(process.lifetime.executable_cache());
        let mut thread = JitThread::new(process.clone()).unwrap();
        let mut native = NativeWorker::default();
        enqueue(&mut thread, &mut native);
        let hold = running.recv_timeout(Duration::from_secs(30)).unwrap();
        process.request_stop().unwrap();
        release.send(()).unwrap();
        let original = process.try_shutdown().unwrap_err();
        assert!(
            original
                .detail
                .contains("shutdown race: HCQ compiler failed")
        );
        assert_eq!(process.try_shutdown(), Err(original.clone()));
        assert_eq!(process.request_stop(), Err(original.clone()));
        assert_eq!(JitThread::new(process.clone()).err(), Some(original));
        assert!(matches!(
            *process.background.lock().unwrap(),
            Background::Joined
        ));
        assert_eq!(Arc::strong_count(&captures), 1);
        drop(thread);
        native.finish().unwrap();
        drop(process);
        assert!(weak_process.upgrade().is_none());
        // Failure stays terminal, rather than clearing it to run healthy
        // maintenance. An external compiler snapshot still retains old code,
        // but not the process or the complete foundation.
        assert!(weak_lifetime.upgrade().is_none());
        assert!(weak_cache.upgrade().unwrap().usage().unwrap().committed > 0);
        drop(hold);
        // Last-reference cleanup must dispose of the failed registries and
        // their actual executable mappings, not leave a process/pool cycle.
        assert!(weak_lifetime.upgrade().is_none());
        assert!(weak_cache.upgrade().is_none());
    }
}

#[test]
fn process_shutdown_preserves_in_flight_collection_and_memory_authority() {
    let process = process(0, |_, _| panic!("no compiler jobs"));
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut units = Vec::new();
    for pc in [PC, PC.checked_add(4).unwrap()] {
        let key = thread.key(pc).unwrap();
        let Request::Owner(claim) = thread.reader.claim(key).unwrap() else {
            panic!("expected cold entry")
        };
        units.push(
            thread
                .compiler
                .publish(
                    Compilation::capture(claim, &*process.memory).unwrap(),
                    &process.lifetime,
                    process.lifetime.executable_cache(),
                    &*process.memory,
                )
                .unwrap(),
        );
        process.lifetime.try_service_links().unwrap();
    }
    process.lifetime.collect_tables().unwrap();
    process.lifetime.retire_unit(units[0]).unwrap();
    let mut transition = process.lifetime.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    let cache = process.lifetime.executable_cache();
    std::thread::scope(|scope| {
        let (collector, hold) = cache.with_lock_held(|| {
            let collector = scope.spawn(|| process.lifetime.try_service_links());
            let deadline = Instant::now() + Duration::from_secs(30);
            while !process.lifetime.collection_in_flight() {
                assert!(
                    Instant::now() < deadline,
                    "collector never acquired ownership"
                );
                std::thread::yield_now();
            }
            // The retired unit shares its directory with a live unit. Its
            // replacement table allocation/destruction is held at the cache
            // mutex, while JIT state must remain available to terminal stop.
            let hold = process
                .lifetime
                .clone()
                .begin(&[MemoryInvalidationKind::Mapping {
                    address_space: SPACE,
                    start: GuestVirtualAddress::new(0x8000),
                    size: 4096,
                }])
                .unwrap();
            process.request_stop().unwrap();
            assert!(!process.try_shutdown().unwrap());
            assert!(!process.try_shutdown().unwrap());
            assert!(process.lifetime.collection_in_flight());
            assert!(!collector.is_finished());
            (collector, hold)
        });
        collector.join().unwrap().unwrap();
        assert!(!process.lifetime.collection_in_flight());
        assert!(!process.try_shutdown().unwrap());
        drop(hold);
    });
    assert!(process.try_shutdown().unwrap());
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
    assert!(matches!(
        thread.demand(PC),
        Err(PublishError::Lifetime(lifetime::Error::Shutdown))
    ));
}

// The HCQ publication test supplies its real compiler with a paused final
// validation. Keep process ownership and join assertions here, rather than
// exposing engine internals or adding a production publication hook.
pub(crate) fn staged_worker_shutdown<F>(
    make_compiler: impl FnOnce(Arc<ExecutionMemory>, mpsc::Sender<()>, mpsc::Receiver<()>) -> F,
) where
    F: Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync + 'static,
{
    let memory = memory(DirectBackendPolicy::Required);
    let queued_pc = PC.checked_add(16).unwrap();
    memory
        .overwrite_mapped_ram(
            SPACE,
            queued_pc,
            &[0x1f, 0x20, 0x03, 0xd5, 0x20, 0, 0x20, 0xd4], // NOP; BRK #1.
        )
        .unwrap();
    let (ready, staged) = mpsc::channel();
    let (resume, wait) = mpsc::channel();
    let compile = make_compiler(memory.clone(), ready, wait);
    let process =
        Arc::new(JitProcess::with_compiler(cpu(), memory.clone(), 1, |_, _| Ok(compile)).unwrap());
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut native = NativeWorker::default();
    enqueue(&mut thread, &mut native);
    staged.recv_timeout(Duration::from_secs(30)).unwrap();
    // The only worker is paused with prepared code. A second real admission
    // must be drained from the queue by stop, never passed to the compiler.
    enqueue_at(&mut thread, &mut native, queued_pc);
    let cache = process.lifetime.executable_cache();
    // LCQ and the unpublished HCQ really own separate executable segments.
    assert_eq!(
        cache.usage().unwrap().committed,
        2 * crate::executable::SEGMENT_BYTES
    );
    let hold = process
        .lifetime
        .clone()
        .begin(&[MemoryInvalidationKind::Mapping {
            address_space: SPACE,
            start: GuestVirtualAddress::new(0x8000),
            size: 4096,
        }])
        .unwrap();
    process.request_stop().unwrap();
    process.request_stop().unwrap();
    assert!(!process.try_shutdown().unwrap());
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Running(_)
    ));
    // The caller must regain control to release its memory authority before
    // any compiler join can await capture on the same execution gate.
    drop(hold);
    let joining = process.clone();
    let (finished, done) = mpsc::channel();
    let join = std::thread::spawn(move || finished.send(joining.try_shutdown()).unwrap());
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(owner) = process.background.try_lock()
            && matches!(*owner, Background::Joining)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "process never started joining its pool"
        );
        std::thread::yield_now();
    }
    // Neither a second teardown nor terminal admission waits for this worker
    // or steals the first caller's join responsibility.
    assert!(!process.try_shutdown().unwrap());
    assert!(matches!(done.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(
        thread.demand(PC),
        Err(PublishError::Lifetime(lifetime::Error::Shutdown))
    ));
    assert!(JitThread::new(process.clone()).is_err());
    assert_eq!(
        process.lifetime.try_service_links(),
        Err(lifetime::Error::Shutdown)
    );
    resume.send(()).unwrap();
    assert_eq!(
        done.recv_timeout(Duration::from_secs(30)).unwrap(),
        Ok(true)
    );
    join.join().unwrap();
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Joined
    ));
    assert!(process.lifetime.background_failure().is_none());
    // Both the prepared output and the queued job are now gone.
    assert!(process.try_shutdown().unwrap());
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
    drop(thread);
    native.finish().unwrap();
}
