use super::*;
use nixe_cpu::execution::{ArchitecturalTimer, CpuFaultKind, TimerSnapshot, VcpuEventState};

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
            &mut worker,
            &mut state,
            100,
            None,
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
