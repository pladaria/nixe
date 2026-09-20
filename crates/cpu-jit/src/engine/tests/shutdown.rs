use super::*;
use nixe_cpu::execution::{ArchitecturalTimer, CpuFaultKind, TimerSnapshot, VcpuEventState};

#[test]
fn process_stop_wakes_background_workers_without_starting_production_compilation() {
    let thread = budget::setup(&[0xd4200120], false);
    let process = Arc::clone(&thread.process);
    let mut pool =
        lifetime::background::workers::Workers::start(2, Arc::clone(&process.lifetime), |_, _| {
            panic!("no production HCQ consumer in Task 5")
        })
        .unwrap()
        .unwrap();
    process.request_stop().unwrap();
    assert!(pool.queue().wait().unwrap().is_none());
    pool.shutdown().unwrap();
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn process_close_reclaims_published_native_code_and_remains_terminal_and_idempotent() {
    let mut thread = budget::setup(&[0xd4200120], false); // BRK #9.
    let process = thread.process.clone();
    let cache = process.lifetime.executable_cache();
    assert!(cache.usage().unwrap().committed > 0);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
    process.request_stop().unwrap();
    assert!(process.try_shutdown().unwrap());
    assert!(matches!(
        thread.demand(PC),
        Err(PublishError::Lifetime(lifetime::Error::Shutdown))
    ));
    assert!(
        JitThread::new(process.clone())
            .err()
            .unwrap()
            .detail
            .contains("shutting down")
    );
}

#[test]
fn close_does_not_wait_for_its_own_or_another_active_invocation() {
    let thread = budget::setup(&[0xd4200120], false);
    let process = thread.process.clone();
    let mut reader = process.lifetime.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1).unwrap());
    // SAFETY: the frame and reader are local, no other FP owner is active, and
    // this test only holds admission; it does not execute or modify guest state.
    let invocation = unsafe { reader.admit(&mut frame, thread.key(PC).unwrap()) }
        .unwrap()
        .unwrap();
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
    drop(invocation);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn stopping_inside_a_cold_completion_prevents_native_continuation() {
    let mut worker = NativeWorker::default();
    struct StoppingTimer(Arc<JitProcess>);
    impl ArchitecturalTimer for StoppingTimer {
        fn snapshot(&self) -> TimerSnapshot {
            self.0.request_stop().unwrap();
            TimerSnapshot {
                counter: 17,
                frequency: 19,
            }
        }
    }
    // MRS X0,CNTPCT_EL0; ADD X1,X1,#1; BRK.
    let mut thread = budget::setup(&[0xd53be020, 0x91000421, 0xd4200000], false);
    let process = thread.process.clone();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let error = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            100,
            &StoppingTimer(process.clone()),
            &VcpuEventState::default(),
        )
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::Unavailable);
    assert_eq!(error.progress, 1);
    assert_eq!(state.pc(), PC.get() + 4);
    assert_eq!(state.general_register_storage_mut()[0], 17);
    assert_eq!(state.general_register_storage_mut()[1], 0);
    assert_eq!(*error.context, state.register_context());
    assert!(process.try_shutdown().unwrap());
    // Per-process vCPU retirement leaves the OS worker's registration alive.
    drop(thread);
    worker.finish().unwrap();
}

#[test]
fn background_worker_failure_reaches_vcpu_and_process_apis_with_original_detail() {
    use crate::lifetime::background::{Outcome, workers::Workers};
    use crate::sampling::Samples;
    use std::time::{Duration, Instant};
    struct Timer;
    impl ArchitecturalTimer for Timer {
        fn snapshot(&self) -> TimerSnapshot {
            panic!("no guest instruction may execute")
        }
    }
    for panic in [false, true] {
        let mut thread = budget::setup(&[0x91000421, 0x17ffffff], false);
        let process = Arc::clone(&thread.process);
        process.lifetime.try_service_links().unwrap();
        let key = thread.key(PC).unwrap();
        let mut worker = NativeWorker::default();
        let mut warmup = A64State::default();
        warmup.set_pc(PC.get());
        thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut warmup,
                PollBudget::new(1, 2).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap();
        let observed = thread.samples.seed_snapshot(key).unwrap().0;
        let (started, wait) = std::sync::mpsc::channel();
        let mut pool = Workers::start(2, Arc::clone(&process.lifetime), move |_, _| {
            started.send(()).unwrap();
            if panic {
                panic!("backend failed at test IR instruction 17");
            }
            Err(Error::internal("backend failed at test IR instruction 17").into())
        })
        .unwrap()
        .unwrap();
        let mut samples = Samples::new();
        let start = Instant::now();
        loop {
            match process
                .lifetime
                .admit_seed(pool.queue(), &mut samples, observed)
                .unwrap()
            {
                Outcome::Queued => break,
                Outcome::Deferred if start.elapsed() < Duration::from_secs(10) => {
                    std::thread::yield_now()
                }
                outcome => panic!("unexpected admission {outcome:?}"),
            }
        }
        wait.recv_timeout(Duration::from_secs(10)).unwrap();
        let original = pool.shutdown().unwrap_err();
        assert!(
            original
                .detail
                .contains("backend failed at test IR instruction 17")
        );
        assert_eq!(pool.shutdown(), Err(original.clone()));
        assert_eq!(process.request_stop(), Err(original.clone()));
        assert_eq!(process.try_shutdown(), Err(original.clone()));
        assert_eq!(
            JitThread::new(Arc::clone(&process)).err(),
            Some(original.clone())
        );
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[1] = 99;
        let before = state.register_context();
        let fault = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                100,
                &Timer,
                &VcpuEventState::default(),
            )
            .unwrap_err();
        assert_eq!(fault.kind, CpuFaultKind::Internal);
        assert_eq!(fault.message, original.detail);
        assert_eq!(fault.progress, 0);
        assert_eq!(*fault.context, before);
        assert_eq!(state.register_context(), before);
        drop(thread);
        worker.finish().unwrap();
    }
}

#[test]
fn background_failure_during_completion_preserves_completed_instruction_and_progress() {
    struct FailingTimer(Arc<JitProcess>);
    impl ArchitecturalTimer for FailingTimer {
        fn snapshot(&self) -> TimerSnapshot {
            self.0
                .lifetime
                .background_failed(Error::internal("HCQ verifier: missing state map"));
            TimerSnapshot {
                counter: 17,
                frequency: 19,
            }
        }
    }
    let mut thread = budget::setup(&[0xd53be020, 0x91000421, 0xd4200000], false);
    let process = Arc::clone(&thread.process);
    let mut worker = NativeWorker::default();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let fault = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            100,
            &FailingTimer(process),
            &VcpuEventState::default(),
        )
        .unwrap_err();
    assert_eq!(fault.kind, CpuFaultKind::Internal);
    assert_eq!(&*fault.message, "HCQ verifier: missing state map");
    assert_eq!(fault.progress, 1);
    assert_eq!(state.pc(), PC.get() + 4);
    assert_eq!(state.general_register_storage_mut()[0], 17);
    assert_eq!(state.general_register_storage_mut()[1], 0);
    assert_eq!(*fault.context, state.register_context());
    drop(thread);
    worker.finish().unwrap();
}
