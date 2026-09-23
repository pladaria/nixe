use super::*;
use crate::hcq::compiler::backend::tests::with_extra_slots;
use crate::lifetime::background::workers::CompileError;

#[test]
fn real_worker_backend_limit_preserves_participants_and_does_not_reject_seed() {
    for participants in 0..=2 {
        let (process, memory, mut reader) = setup();
        if participants >= 1 {
            promote_at(&process, &memory, &mut reader, 0x2000);
        }
        if participants == 2 {
            promote_at(&process, &memory, &mut reader, 0x1000);
        }
        let before = [0x1000, 0x2000].map(|pc| payload(&mut reader, pc).unwrap());
        let committed = process.executable_cache().usage().unwrap().committed;
        let memory = Arc::new(memory);
        let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::clone(&memory)).unwrap();
        let (finished, done) = mpsc::channel();
        let calls = AtomicUsize::new(0);
        let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
            let call = calls.fetch_add(1, Ordering::Relaxed);
            let result = if call == 0 {
                // Legal individual slots, but their aggregate exceeds the
                // fixed frame. The real backend emits ImplLimitExceeded.
                with_extra_slots(8192, || consumer(resources, work))
            } else {
                assert_eq!(call, 1);
                consumer(resources, work)
            };
            assert!(result.is_ok(), "{result:?}");
            assert!(resources.context.compiled_code().is_none());
            assert!(resources.context.func.layout.blocks().next().is_none());
            assert!(resources.context.func.nixe_exit_costs.is_empty());
            finished.send(call).unwrap();
            result
        })
        .unwrap()
        .unwrap();
        assert_eq!(
            admit_to(
                &process,
                workers.queue(),
                &mut reader,
                0x1000,
                0x1004,
                0x2000
            ),
            Outcome::Queued
        );
        assert_eq!(done.recv_timeout(Duration::from_secs(10)).unwrap(), 0);
        assert_eq!(
            process.executable_cache().usage().unwrap().committed,
            committed
        );
        for (pc, original) in [0x1000, 0x2000].into_iter().zip(before) {
            assert_eq!(payload(&mut reader, pc).unwrap(), original);
        }
        assert_eq!(
            admit_to(
                &process,
                workers.queue(),
                &mut reader,
                0x1000,
                0x1004,
                0x2000
            ),
            Outcome::Suppressed
        );
        run(&process, &memory, &mut reader, 0x3000, 3);

        // Same source for a zero-family boundary; otherwise an unoptimized
        // ingress into the still-live participants. Reuse this same worker.
        let pc = if participants == 0 { 0x1000 } else { 0x3000 };
        let snapshot = AdmissionSnapshot {
            key: key(pc),
            version: payload(&mut reader, pc).unwrap().reachability(),
            sequence: 8,
            last_edge: None,
            successors: [None; 4],
        };
        assert_eq!(
            process
                .admit_seed(workers.queue(), &mut Samples::new(), snapshot)
                .unwrap(),
            Outcome::Queued
        );
        assert_eq!(done.recv_timeout(Duration::from_secs(10)).unwrap(), 1);
        process.try_service_links().unwrap();
        assert!(payload(&mut reader, pc).unwrap().hcq().is_some());
        run(&process, &memory, &mut reader, 0x3000, 3);
        workers.shutdown().unwrap();
        assert!(process.background_failure().is_none());
        assert!(process.try_shutdown().unwrap());
        assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
    }
}

#[test]
fn real_worker_unsupported_backend_shape_is_a_failure_not_a_reshape_negative() {
    let (process, memory, mut reader) = setup();
    let consumer = crate::hcq::worker::consumer(host(), 0x10000, Arc::new(memory)).unwrap();
    let (finished, done) = mpsc::channel();
    let mut workers = Workers::start(1, Arc::clone(&process), move |resources, work| {
        // An individually illegal slot yields Unsupported, not the aggregate
        // implementation limit exercised above. Do not fabricate a backend error.
        let result = with_extra_slots(cranelift_codegen::nixe::FRAME_BYTES + 1, || {
            consumer(resources, work)
        });
        let Err(CompileError::Failed(error)) = &result else {
            panic!("backend failure must remain visible: {result:?}");
        };
        assert!(error.to_string().contains("Unsupported"));
        assert!(resources.context.func.layout.blocks().next().is_none());
        finished.send(()).unwrap();
        result
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        admit_to(
            &process,
            workers.queue(),
            &mut reader,
            0x1000,
            0x1004,
            0x2000
        ),
        Outcome::Queued
    );
    done.recv_timeout(Duration::from_secs(10)).unwrap();
    let error = workers.shutdown().unwrap_err();
    assert!(error.to_string().contains("Unsupported"));
    assert!(
        process
            .background_failure()
            .unwrap()
            .to_string()
            .contains("Unsupported")
    );
}
