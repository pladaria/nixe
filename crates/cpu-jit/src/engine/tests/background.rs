use super::*;
use crate::engine::background::Background;
use crate::lifetime::background::{
    Outcome, Work,
    workers::{CompileError, Resources},
};
use std::sync::mpsc;
use std::time::{Duration, Instant};

mod production;

fn process(
    count: usize,
    compile: impl Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync + 'static,
) -> Arc<JitProcess> {
    // Deterministic worker count in tests, regardless of the test host's CPUs.
    Arc::new(
        JitProcess::with_compiler(
            cpu(),
            memory(DirectBackendPolicy::Required),
            count,
            |_, _| Ok(compile),
        )
        .unwrap(),
    )
}

fn enqueue(thread: &mut JitThread, worker: &mut NativeWorker) {
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    thread.process.lifetime.try_service_links().unwrap();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    thread
        .invoke(
            &mut crate::ReturnStack::default(),
            worker,
            &mut state,
            PollBudget::new(1, 2).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap();
    let observed = thread
        .samples
        .seed_snapshot(thread.key(PC).unwrap())
        .unwrap()
        .0;
    let started = Instant::now();
    loop {
        let owner = thread.process.background.lock().unwrap();
        let Background::Running(workers) = &*owner else {
            panic!("expected owned pool")
        };
        let outcome = thread
            .process
            .lifetime
            .admit_seed(workers.queue(), &mut thread.samples, observed)
            .unwrap();
        drop(owner);
        match outcome {
            Outcome::Queued => break,
            Outcome::Deferred if started.elapsed() < Duration::from_secs(10) => {
                std::thread::yield_now()
            }
            other => panic!("unexpected admission {other:?}"),
        }
    }
}

#[test]
fn process_joins_idle_workers_and_zero_worker_policy_without_a_second_start() {
    for count in [0, 2] {
        let mut process = process(count, |_, _| panic!("no jobs"));
        process.request_stop().unwrap();
        assert!(process.try_shutdown().unwrap());
        assert!(matches!(
            *process.background.lock().unwrap(),
            Background::Joined
        ));
        assert!(process.try_shutdown().unwrap());
        assert!(
            Arc::get_mut(&mut process)
                .unwrap()
                .start_background(count, |_, _| panic!("restart"))
                .is_err()
        );
    }
}

#[test]
fn process_join_releases_owner_lock_and_cannot_be_mistaken_for_completed_teardown() {
    let (started, running) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let captures = Arc::new(());
    let retained = Arc::clone(&captures);
    let process = process(2, move |_, work| {
        let _keep = &retained;
        let source = match work.observation() {
            crate::lifetime::background::Observation::Seed(snapshot) => {
                work.lcq(snapshot.key).unwrap().unwrap()
            }
            _ => panic!("expected seed"),
        };
        started.send(()).unwrap();
        wait.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        assert_eq!(work.check(), Err(lifetime::Error::StalePublication));
        assert_eq!(source.unit.instructions.get(0).unwrap().bits, 0xd503201f);
        Ok(())
    });
    let mut thread = JitThread::new(Arc::clone(&process)).unwrap();
    let mut native = NativeWorker::default();
    enqueue(&mut thread, &mut native);
    running.recv_timeout(Duration::from_secs(10)).unwrap();
    process.request_stop().unwrap(); // Must not join the blocked consumer.
    let joining = Arc::clone(&process);
    let (finished, done) = mpsc::channel();
    let shutdown = std::thread::spawn(move || finished.send(joining.try_shutdown()).unwrap());
    let start = Instant::now();
    loop {
        if let Ok(owner) = process.background.try_lock()
            && matches!(*owner, Background::Joining)
        {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "join retained the owner lock"
        );
        std::thread::yield_now();
    }
    assert!(done.try_recv().is_err());
    assert!(!process.try_shutdown().unwrap());
    assert!(
        process
            .lifetime
            .executable_cache()
            .usage()
            .unwrap()
            .committed
            > 0
    );
    release.send(()).unwrap();
    assert_eq!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        Ok(true)
    );
    shutdown.join().unwrap();
    assert_eq!(Arc::strong_count(&captures), 1);
    assert_eq!(
        process
            .lifetime
            .executable_cache()
            .usage()
            .unwrap()
            .committed,
        0
    );
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Joined
    ));
    drop(thread);
    native.finish().unwrap();
}

#[test]
fn failed_owned_worker_is_joined_even_when_stop_returns_its_failure() {
    let (started, running) = mpsc::channel();
    let process = process(2, move |_, _| {
        started.send(()).unwrap();
        Err(Error::internal("owned compiler failure").into())
    });
    let mut thread = JitThread::new(Arc::clone(&process)).unwrap();
    let mut native = NativeWorker::default();
    enqueue(&mut thread, &mut native);
    running.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(
        process.try_shutdown(),
        Err(Error::internal("owned compiler failure"))
    );
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Joined
    ));
    assert_eq!(
        process.try_shutdown(),
        Err(Error::internal("owned compiler failure"))
    );
    drop(thread);
    native.finish().unwrap();
}

#[test]
fn last_process_owner_closes_and_joins_workers_without_an_arc_cycle() {
    let captures = Arc::new(());
    let retained = Arc::clone(&captures);
    let process = process(2, move |_, _| {
        let _keep = &retained;
        panic!("no jobs")
    });
    let weak = Arc::downgrade(&process);
    let lifetime = Arc::clone(&process.lifetime);
    drop(process);
    assert!(weak.upgrade().is_none());
    assert_eq!(Arc::strong_count(&captures), 1);
    assert!(lifetime.try_shutdown().unwrap());
}

#[test]
fn owned_real_hcq_consumer_promotes_and_executes_before_joining() {
    let memory = memory(DirectBackendPolicy::Required);
    let compile = crate::hcq::worker::consumer(
        if cfg!(target_arch = "x86_64") {
            HostAbi::X86_64
        } else {
            HostAbi::Aarch64
        },
        0x10000,
        memory.clone(),
    )
    .unwrap();
    let (finished, done) = mpsc::channel();
    let process = JitProcess::with_compiler(cpu(), memory.clone(), 1, |_, _| {
        Ok(move |resources: &mut Resources, work: Work<'_>| {
            let result = compile(resources, work);
            finished
                .send(result.as_ref().copied().map_err(|e| format!("{e:?}")))
                .unwrap();
            result
        })
    })
    .unwrap();
    let process = Arc::new(process);
    let weak = Arc::downgrade(&process);
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut native = NativeWorker::default();
    enqueue(&mut thread, &mut native);
    done.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    process.lifetime.try_service_links().unwrap();
    {
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 10).unwrap());
        let invocation = unsafe { thread.reader.admit(&mut frame, thread.key(PC).unwrap()) }
            .unwrap()
            .unwrap();
        assert!(invocation.payload().hcq().is_some());
    }
    breakpoint(&mut thread, &mut native, 1);
    process.request_stop().unwrap();
    assert!(process.try_shutdown().unwrap());
    assert!(matches!(
        *process.background.lock().unwrap(),
        Background::Joined
    ));
    drop(thread);
    drop(process);
    assert!(weak.upgrade().is_none());
    native.finish().unwrap();
}
