use super::*;
use crate::lifetime::background::{Outcome, workers::Workers};
use crate::sampling::{AdmissionSnapshot, Samples};
use std::sync::{Mutex, atomic::AtomicUsize, atomic::Ordering, mpsc};
use std::time::Instant;

#[test]
fn mapping_change_cancels_running_compilation_without_waiting_and_allows_retry() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let block = key(0x1000);
    let snapshot = || AdmissionSnapshot {
        key: block,
        version: process.reserve(block).unwrap().reachability,
        sequence: 8,
        last_edge: None,
        successors: [None; 4],
    };
    let observed = snapshot();
    let (started, ready) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let (finished, done) = mpsc::channel();
    let calls = AtomicUsize::new(0);
    let mut workers = Workers::start(1, Arc::clone(&process), move |_, work| {
        let call = calls.fetch_add(1, Ordering::Relaxed);
        let source = work.lcq(block)?.unwrap();
        if call == 0 {
            started.send(()).unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        let result = work.check();
        // The immutable old image survives unlink/republication until this
        // compiler releases it. It does not hold an execution epoch open.
        assert_eq!(source.unit.instructions[0].bits, 0xf9400020);
        drop(source);
        drop(work);
        finished.send((call, result)).unwrap();
        result?;
        Ok(())
    })
    .unwrap()
    .unwrap();
    let mut samples = Samples::new();
    let mut enqueue = |observed| {
        let start = Instant::now();
        loop {
            match process
                .admit_seed(workers.queue(), &mut samples, observed)
                .unwrap()
            {
                Outcome::Queued => break,
                Outcome::Deferred if start.elapsed() < Duration::from_secs(10) => {
                    std::thread::yield_now();
                }
                outcome => panic!("unexpected admission: {outcome:?}"),
            }
        }
    };
    enqueue(observed);
    ready.recv_timeout(Duration::from_secs(10)).unwrap();
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::READ,
        )
        .unwrap();
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    publish(&process, &memory, 0x1000);
    let replacement = snapshot();
    assert_ne!(replacement.version, observed.version);
    release.send(()).unwrap();
    assert_eq!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        (0, Err(Error::StalePublication))
    );
    enqueue(replacement);
    assert_eq!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        (1, Ok(()))
    );
    workers.shutdown().unwrap();
    assert_eq!(process.lock().compilers, 0);
    assert!(process.background_failure().is_none());
}
