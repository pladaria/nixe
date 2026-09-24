use super::*;
use crate::executable::{SOFT_BYTES, Tier};
use crate::lifetime::background::workers::CompileError;
use std::time::Instant;

fn enqueue(
    process: &Lifetime,
    workers: &Workers,
    reader: &mut Reader,
    root: u64,
    source: u64,
    target: u64,
) {
    let start = Instant::now();
    loop {
        match admit_to(process, workers.queue(), reader, root, source, target) {
            Outcome::Queued => return,
            Outcome::Deferred if start.elapsed() < Duration::from_secs(10) => {
                std::thread::yield_now();
            }
            outcome => panic!("unexpected reshape admission: {outcome:?}"),
        }
    }
}

#[test]
fn disjoint_real_reshapes_use_two_workers_without_sharing_compiler_scratch() {
    let (process, memory, mut reader) = setup();
    let memory = Arc::new(memory);
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
    let (started, ready) = mpsc::channel();
    let (first_release, first_wait) = mpsc::channel();
    let (second_release, second_wait) = mpsc::channel();
    let waits = [Mutex::new(first_wait), Mutex::new(second_wait)];
    let (finished, done) = mpsc::channel();
    let mut workers = Workers::start(2, Arc::clone(&process), move |resources, work| {
        let root = work.observation().root().0;
        let index = usize::from(root == key(0x6000));
        started
            .send((root, &resources.context as *const _ as usize))
            .unwrap();
        waits[index]
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        let result = consumer(resources, work);
        assert!(result.is_ok(), "{root:?}: {result:?}");
        finished.send(root).unwrap();
        result
    })
    .unwrap()
    .unwrap();
    enqueue(&process, &workers, &mut reader, 0x1000, 0x1004, 0x2000);
    let first = ready.recv_timeout(Duration::from_secs(10)).unwrap();
    enqueue(&process, &workers, &mut reader, 0x6000, 0x6000, 0x7000);
    let second = ready.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!((first.0, second.0), (key(0x1000), key(0x6000)));
    assert_ne!(first.1, second.1);
    // The second worker must compile and publish while the first still owns
    // its independent reservation and compiler lease, not just after it drains.
    second_release.send(()).unwrap();
    assert_eq!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        key(0x6000)
    );
    process.try_service_links().unwrap();
    let second_family = payload(&mut reader, 0x6000).unwrap().hcq().unwrap().family;
    assert_eq!(
        payload(&mut reader, 0x7000).unwrap().hcq().unwrap().family,
        second_family
    );
    assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_none());
    first_release.send(()).unwrap();
    assert_eq!(
        done.recv_timeout(Duration::from_secs(10)).unwrap(),
        key(0x1000)
    );
    process.try_service_links().unwrap();
    assert_ne!(
        payload(&mut reader, 0x1000).unwrap().hcq().unwrap().family,
        second_family
    );
    // X3 takes the indirect region's external alternative; RET enters the
    // independently compiled first region through its selected second entry.
    run(&process, &memory, &mut reader, 0x6000, 3);
    run(&process, &memory, &mut reader, 0x7000, 2);
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn running_real_reshape_cancels_on_code_change_or_shutdown_and_defers_on_pressure() {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Interruption {
        Code,
        Pressure,
        Shutdown,
    }
    for interruption in [
        Interruption::Code,
        Interruption::Pressure,
        Interruption::Shutdown,
    ] {
        let (process, memory, mut reader) = setup();
        promote_at(&process, &memory, &mut reader, 0x2000);
        let memory = Arc::new(memory);
        let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
        let (started, ready) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let (finished, done) = mpsc::channel();
        let calls = AtomicUsize::new(0);
        let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
            let call = calls.fetch_add(1, Ordering::Relaxed);
            // Hold actual baseline storage across the interruption. Storage
            // lifetime cannot make the old input or participant valid again.
            let baseline = work.lcq(key(0x2000))?.unwrap();
            if call == 0 {
                started.send(()).unwrap();
                wait.lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            let result = consumer(resources, work);
            if call == 0 {
                match interruption {
                    Interruption::Pressure => {
                        assert!(matches!(result, Err(CompileError::Deferred)))
                    }
                    _ => assert!(matches!(result, Err(CompileError::Cancelled))),
                }
            } else {
                assert!(result.is_ok(), "{result:?}");
            }
            drop(baseline);
            finished.send(()).unwrap();
            result
        })
        .unwrap()
        .unwrap();
        enqueue(&process, &workers, &mut reader, 0x1000, 0x1004, 0x2000);
        ready.recv_timeout(Duration::from_secs(10)).unwrap();
        let before = payload(&mut reader, 0x2000).unwrap();
        let cache = process.executable_cache();
        let pressure = match interruption {
            Interruption::Code => {
                memory
                    .overwrite_mapped_ram(
                        AddressSpaceId::new(1),
                        GuestVirtualAddress::new(0x2000),
                        &0x91000c00u32.to_le_bytes(), // ADD X0,X0,#3
                    )
                    .unwrap();
                None
            }
            Interruption::Pressure => Some(
                cache
                    .charge_metadata(SOFT_BYTES - cache.usage().unwrap().total(), Tier::Lcq)
                    .unwrap(),
            ),
            Interruption::Shutdown => {
                process.request_shutdown().unwrap();
                assert!(!process.try_shutdown().unwrap());
                None
            }
        };
        release.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(10)).unwrap();
        drop(pressure);
        if interruption != Interruption::Shutdown {
            process.try_service_links().unwrap();
            assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_none());
            if interruption == Interruption::Code {
                assert!(payload(&mut reader, 0x2000).is_none());
                super::super::lifecycle::demand(&process, &memory, &mut reader, 0x2000);
            } else {
                assert_eq!(payload(&mut reader, 0x2000).unwrap(), before);
            }
            // Neither stale work nor pressure installs a permanent negative
            // or leaks the surviving endpoint/family reservation.
            enqueue(&process, &workers, &mut reader, 0x1000, 0x1004, 0x2000);
            done.recv_timeout(Duration::from_secs(10)).unwrap();
            process.try_service_links().unwrap();
            run(
                &process,
                &memory,
                &mut reader,
                0x3000,
                if interruption == Interruption::Code {
                    4
                } else {
                    3
                },
            );
        }
        workers.shutdown().unwrap();
        assert!(process.background_failure().is_none());
        assert!(process.try_shutdown().unwrap());
        assert_eq!(cache.usage().unwrap().committed, 0);
    }
}

#[test]
fn queued_real_reshape_is_discarded_before_consumer_on_invalidation_or_shutdown() {
    for shutdown in [false, true] {
        let (process, memory, mut reader) = setup();
        promote_at(&process, &memory, &mut reader, 0x2000);
        let memory = Arc::new(memory);
        let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let called = Arc::clone(&calls);
        let (started, ready) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let (finished, done) = mpsc::channel();
        let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
            let call = called.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                assert_eq!(work.observation().root().0, key(0x6000));
                started.send(()).unwrap();
                wait.lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            } else {
                assert!(!shutdown);
                assert_eq!(call, 1);
                assert_eq!(work.observation().root().0, key(0x1000));
                // Only the readmitted generation reaches the consumer.
                assert_eq!(
                    work.lcq(key(0x2000))?
                        .unwrap()
                        .unit
                        .instructions
                        .get(0)
                        .unwrap()
                        .bits,
                    0x91000c00
                );
            }
            let result = consumer(resources, work);
            if shutdown {
                assert!(matches!(result, Err(CompileError::Cancelled)));
            } else {
                assert!(result.is_ok(), "{result:?}");
            }
            finished.send(()).unwrap();
            result
        })
        .unwrap()
        .unwrap();
        // Occupy the only worker with an independent, valid reshape. The
        // affected job cannot leave the queue until after the interruption.
        enqueue(&process, &workers, &mut reader, 0x6000, 0x6000, 0x7000);
        ready.recv_timeout(Duration::from_secs(10)).unwrap();
        enqueue(&process, &workers, &mut reader, 0x1000, 0x1004, 0x2000);
        if shutdown {
            process.request_shutdown().unwrap();
            assert!(!process.try_shutdown().unwrap());
        } else {
            memory
                .overwrite_mapped_ram(
                    AddressSpaceId::new(1),
                    GuestVirtualAddress::new(0x2000),
                    &0x91000c00u32.to_le_bytes(),
                )
                .unwrap();
        }
        release.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(10)).unwrap();
        if !shutdown {
            process.try_service_links().unwrap();
            super::super::lifecycle::demand(&process, &memory, &mut reader, 0x2000);
            enqueue(&process, &workers, &mut reader, 0x1000, 0x1004, 0x2000);
            done.recv_timeout(Duration::from_secs(10)).unwrap();
            process.try_service_links().unwrap();
            run(&process, &memory, &mut reader, 0x3000, 4);
        }
        workers.shutdown().unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), if shutdown { 1 } else { 2 });
        assert!(process.background_failure().is_none());
        assert!(process.try_shutdown().unwrap());
        assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
    }
}

#[test]
fn queued_reshape_pressure_releases_reservations_before_real_worker_retry() {
    let (process, memory, mut reader) = setup();
    promote_at(&process, &memory, &mut reader, 0x2000);
    let queue = Queue::new(1, &process).unwrap().unwrap();
    assert_eq!(
        admit_to(&process, &queue, &mut reader, 0x1000, 0x1004, 0x2000),
        Outcome::Queued
    );
    let cache = process.executable_cache();
    let pressure = cache
        .charge_metadata(SOFT_BYTES - cache.usage().unwrap().total(), Tier::Lcq)
        .unwrap();
    // Control dequeue directly to exercise the pool's exact acceptance boundary
    // deterministically, without a sleep or a test hook in its dispatch loop.
    assert!(
        process
            .accept_background(queue.pop().unwrap().unwrap())
            .unwrap()
            .is_none()
    );
    assert!(queue.pop().unwrap().is_none());
    drop(pressure);
    drop(queue);
    let memory = Arc::new(memory);
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
    let (finished, done) = mpsc::channel();
    let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
        let result = consumer(resources, work);
        assert!(result.is_ok(), "{result:?}");
        finished.send(()).unwrap();
        result
    })
    .unwrap()
    .unwrap();
    enqueue(&process, &workers, &mut reader, 0x1000, 0x1004, 0x2000);
    done.recv_timeout(Duration::from_secs(10)).unwrap();
    process.try_service_links().unwrap();
    run(&process, &memory, &mut reader, 0x3000, 3);
    workers.shutdown().unwrap();
    assert!(process.background_failure().is_none());
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
}
