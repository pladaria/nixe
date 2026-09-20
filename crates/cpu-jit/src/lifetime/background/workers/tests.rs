use super::*;
use crate::lifetime::background::{
    Observation, Outcome,
    tests::{setup, snapshot},
};
use crate::lifetime::unit::tests::key;
use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
use cranelift_frontend::FunctionBuilder;
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

mod races;

fn enqueue(process: &Lifetime, workers: &Workers, samples: &mut crate::sampling::Samples, pc: u64) {
    let observed = snapshot(process, pc);
    let start = Instant::now();
    loop {
        match process
            .admit_seed(workers.queue(), samples, observed)
            .unwrap()
        {
            Outcome::Queued => return,
            Outcome::Deferred if start.elapsed() < Duration::from_secs(10) => {
                std::thread::yield_now()
            }
            outcome => panic!("unexpected admission: {outcome:?}"),
        }
    }
}

#[test]
fn policy_reserves_two_logical_cpus_and_caps_at_four_workers() {
    for (cpus, expected) in [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 2),
        (7, 2),
        (8, 3),
        (9, 3),
        (10, 4),
        (11, 4),
        (128, 4),
        (usize::MAX, 4),
    ] {
        assert_eq!(count(cpus), expected);
    }
}

#[test]
fn zero_workers_create_no_pool_or_consumer_and_invalid_count_fails() {
    let (process, _, _) = setup(1);
    assert!(
        Workers::start(0, Arc::clone(&process), |_, _| panic!(
            "zero worker consumer"
        ))
        .unwrap()
        .is_none()
    );
    assert!(Workers::start(5, process, |_, _| panic!("invalid worker consumer")).is_err());
}

#[test]
fn fixed_workers_have_private_reusable_compiler_and_decoder_storage() {
    let (process, _, mut samples) = setup(3);
    let (events, receiver) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let config = cranelift_native::builder()
        .unwrap()
        .finish(cranelift_codegen::settings::Flags::new(
            cranelift_codegen::settings::builder(),
        ))
        .unwrap()
        .frontend_config();
    let mut workers = Workers::start(2, Arc::clone(&process), move |resources, work| {
        assert!(resources.context.func.layout.entry_block().is_none());
        assert!(resources.decoded.is_empty());
        assert!(resources.decoded.capacity() >= MAX_INSTRUCTIONS);
        let Observation::Seed(observed) = work.observation() else {
            panic!("expected seed")
        };
        let input = work.lcq(observed.key).map_err(fail)?.unwrap();
        for instruction in &input.unit.instructions {
            let key = instruction.key.block_key();
            resources.decoded.push(nixe_cpu::decode::decode(
                key.platform,
                nixe_cpu::location::LocationDescriptor::new(key.pc, key.profile),
                instruction.bits.into(),
            ));
        }
        // Exercise reusable Cranelift frontend state, not a pretend HCQ backend.
        resources
            .context
            .func
            .signature
            .returns
            .push(AbiParam::new(types::I64));
        let mut builder =
            FunctionBuilder::new(&mut resources.context.func, &mut resources.frontend);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        let value = builder
            .ins()
            .iconst(types::I64, observed.key.pc.get() as i64);
        builder.ins().return_(&[value]);
        builder.finalize(config);
        events
            .send((
                observed.key,
                &resources.context as *const Context as usize,
                resources.decoded.as_ptr() as usize,
                std::thread::current().id(),
            ))
            .unwrap();
        if observed.key.pc.get() < 8 {
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        Ok(())
    })
    .unwrap()
    .unwrap();
    for pc in [0, 4] {
        enqueue(&process, &workers, &mut samples, pc);
    }
    let first = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    let second = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_ne!(first.0, second.0);
    assert_ne!(first.1, second.1);
    assert_ne!(first.2, second.2);
    assert_ne!(first.3, second.3);
    assert_eq!(workers.threads.len(), 2);
    release.send(()).unwrap();
    release.send(()).unwrap();
    enqueue(&process, &workers, &mut samples, 8);
    let reused = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(reused.0, key(8));
    assert!(
        [first, second]
            .iter()
            .any(|old| (old.1, old.2, old.3) == (reused.1, reused.2, reused.3))
    );
    workers.shutdown().unwrap();
    assert_eq!(process.lock().compilers, 0);
    assert!(workers.threads.is_empty());
    workers.shutdown().unwrap();
}

#[test]
fn shutdown_drains_pending_jobs_and_waits_for_the_running_owner() {
    let (process, _, mut samples) = setup(2);
    let (events, receiver) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let workers = Workers::start(1, Arc::clone(&process), move |_, work| {
        assert!(work.lcq(key(0)).map_err(fail)?.is_some());
        events.send(()).unwrap();
        wait.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        Ok(())
    })
    .unwrap()
    .unwrap();
    enqueue(&process, &workers, &mut samples, 0);
    receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    enqueue(&process, &workers, &mut samples, 4);
    // Close while the one worker is busy: pending work must not be consumed.
    drop(workers.queue.close().unwrap());
    release.send(()).unwrap();
    drop(workers); // RAII joins; no detached process/compiler references.
    assert_eq!(process.lock().compilers, 0);
    for pc in [0, 4] {
        let state = process.lock();
        let slot = state
            .dispatch
            .get(*state.keys.get(&key(pc)).unwrap())
            .unwrap();
        assert!(!slot.optimization.pinned());
    }
}

#[test]
fn worker_errors_and_panics_close_admission_and_are_returned_by_join() {
    for panic in [false, true] {
        let (process, _, mut samples) = setup(1);
        let (events, receiver) = mpsc::channel();
        let mut workers = Workers::start(2, Arc::clone(&process), move |_, _work| {
            events.send(()).unwrap();
            if panic {
                panic!("injected compiler panic");
            }
            Err(Error::internal("injected compiler failure").into())
        })
        .unwrap()
        .unwrap();
        enqueue(&process, &workers, &mut samples, 0);
        receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        let error = workers.shutdown().unwrap_err();
        assert!(error.to_string().contains(if panic {
            "injected compiler panic"
        } else {
            "injected compiler failure"
        }));
        assert_eq!(process.lock().compilers, 0);
        assert!(process.lock().healthy().is_err());
        assert!(workers.threads.is_empty());
    }
}

#[test]
fn poisoned_queue_cleanup_still_drains_and_joins_sleeping_workers() {
    let (process, _, _) = setup(1);
    let mut workers = Workers::start(2, Arc::clone(&process), |_, _| panic!("no job"))
        .unwrap()
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = workers.queue.pending.lock().unwrap();
        panic!("injected queue poison");
    }));
    assert!(result.is_err());
    assert!(workers.shutdown().is_err());
    assert!(workers.threads.is_empty());
    assert_eq!(process.lock().compilers, 0);
}

#[test]
fn partial_startup_releases_threads_queue_storage_and_consumer_captures() {
    let (process, _, _) = setup(1);
    let captures = Arc::new(());
    let owners = Arc::strong_count(&process);
    let metadata = process.cache.usage().unwrap().metadata;
    for index in 0..4 {
        let captured = Arc::clone(&captures);
        let error = Workers::start_inner(
            4,
            Arc::clone(&process),
            move |_, _| {
                let _keep = &captured;
                panic!("no job during startup");
            },
            Some(index),
        )
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains(&format!("start HCQ worker {index}"))
        );
        assert!(
            error
                .to_string()
                .contains("injected HCQ thread creation failure")
        );
        assert_eq!(Arc::strong_count(&captures), 1);
        assert_eq!(Arc::strong_count(&process), owners);
        assert_eq!(process.cache.usage().unwrap().metadata, metadata);
        assert!(process.lock().background_queue.upgrade().is_none());
        process.lock().healthy().unwrap();
    }
    let pool = Workers::start(1, Arc::clone(&process), |_, _| panic!("no jobs"))
        .unwrap()
        .unwrap();
    drop(pool);
    assert_eq!(Arc::strong_count(&process), owners);
}

#[test]
fn duplicate_pool_registration_cannot_replace_or_close_the_existing_queue() {
    let (process, _, _) = setup(1);
    let mut first = Workers::start(1, Arc::clone(&process), |_, _| panic!("no job"))
        .unwrap()
        .unwrap();
    let error = Workers::start(1, Arc::clone(&process), |_, _| panic!("duplicate consumer"))
        .err()
        .unwrap();
    assert!(error.to_string().contains("already registered"));
    assert!(Arc::ptr_eq(
        &process.lock().background_queue.upgrade().unwrap(),
        &first.queue
    ));
    assert!(!first.queue.pending.lock().unwrap().closed);
    process.request_shutdown().unwrap();
    first.shutdown().unwrap();
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn process_shutdown_cancels_running_work_drains_jobs_and_blocks_late_enqueue() {
    for ticket in [false, true] {
        let (process, _, mut samples) = setup(3);
        let (started, running) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let mut workers = Workers::start(1, Arc::clone(&process), move |_, work| {
            let source = work.lcq(key(0)).map_err(fail)?.unwrap();
            started.send(()).unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            assert_eq!(work.check(), Err(lifetime::Error::StalePublication));
            assert_eq!(source.unit.instructions[0].bits, 0xd503201f);
            Ok(())
        })
        .unwrap()
        .unwrap();
        enqueue(&process, &workers, &mut samples, 0);
        running.recv_timeout(Duration::from_secs(10)).unwrap();
        enqueue(&process, &workers, &mut samples, 4);
        let late = process
            .reserve_seed(snapshot(&process, 8))
            .unwrap()
            .unwrap();
        if ticket {
            process.request(lifetime::Reason::Shutdown).unwrap();
        } else {
            process.request_shutdown().unwrap();
        }
        assert_eq!(process.lock().compilers, 1);
        assert!(workers.queue.pending.lock().unwrap().jobs.is_empty());
        assert!(workers.queue.pending.lock().unwrap().closed);
        assert_eq!(workers.queue.enqueue(late).unwrap(), Outcome::Stale);
        for pc in [4, 8] {
            let state = process.lock();
            let slot = state
                .dispatch
                .get(*state.keys.get(&key(pc)).unwrap())
                .unwrap();
            assert!(!slot.optimization.pinned());
        }
        assert!(!process.try_shutdown().unwrap());
        release.send(()).unwrap();
        workers.shutdown().unwrap();
        assert!(process.try_shutdown().unwrap());
        process.request_shutdown().unwrap();
    }
}

#[test]
fn terminal_process_rejects_startup_without_retaining_resources() {
    let (process, _, _) = setup(1);
    process.request_shutdown().unwrap();
    let before = process.cache.usage().unwrap().metadata;
    assert!(Workers::start(2, Arc::clone(&process), |_, _| panic!("closed consumer")).is_err());
    assert!(process.lock().background_queue.upgrade().is_none());
    assert_eq!(process.cache.usage().unwrap().metadata, before);
}

#[test]
fn failed_process_stop_still_wakes_registered_sleepers() {
    let (process, _, _) = setup(1);
    let mut workers = Workers::start(2, Arc::clone(&process), |_, _| panic!("no jobs"))
        .unwrap()
        .unwrap();
    process.fail(&mut process.lock(), lifetime::Error::CacheFailed);
    assert_eq!(
        process.request_shutdown(),
        Err(lifetime::Error::CacheFailed)
    );
    assert!(workers.queue.pending.lock().unwrap().closed);
    workers.shutdown().unwrap();
}

#[test]
fn first_worker_diagnostic_survives_peer_failure_and_repeated_join() {
    let (process, _, _) = setup(1);
    let mut workers = Workers::start(2, Arc::clone(&process), |_, _| panic!("no jobs"))
        .unwrap()
        .unwrap();
    let original = Error::internal("original compiler error");
    process.background_failed(original.clone());
    process.background_failed(Error::internal("secondary cancellation error"));
    assert_eq!(
        process.lock().healthy(),
        Err(lifetime::Error::BackgroundWorker)
    );
    assert_eq!(process.background_failure(), Some(original.clone()));
    assert_eq!(workers.shutdown(), Err(original.clone()));
    assert_eq!(workers.shutdown(), Err(original.clone()));
    drop(workers);
    assert_eq!(process.background_failure(), Some(original));
}

#[test]
fn worker_diagnostic_cannot_replace_an_earlier_lifecycle_failure() {
    let (process, _, _) = setup(1);
    let mut workers = Workers::start(1, Arc::clone(&process), |_, _| panic!("no jobs"))
        .unwrap()
        .unwrap();
    process.fail(&mut process.lock(), lifetime::Error::CacheFailed);
    process.background_failed(Error::internal("secondary worker failure"));
    assert_eq!(process.lock().healthy(), Err(lifetime::Error::CacheFailed));
    assert_eq!(process.background_failure(), None);
    assert!(workers.queue.pending.lock().unwrap().closed);
    assert_eq!(
        process
            .diagnostic(lifetime::Error::CacheFailed)
            .detail
            .as_ref(),
        "JIT executable cache has failed"
    );
    workers.shutdown().unwrap();
}
