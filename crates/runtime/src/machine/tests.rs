use super::*;
use crate::process::tests::synthetic_instruction_process;
use crate::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatchRequest, ExceptionResume,
    ExceptionTerminationReason, ExceptionTerminationScope,
};
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_scheduler::{MachineSchedulerProfile, PriorityRange, VirtualCpuDescriptor};
use std::{
    sync::{Condvar, mpsc},
    time::Instant,
};

fn kernel(cores: u32) -> Arc<SchedulerState> {
    SchedulerState::new(
        MachineSchedulerProfile::new(
            (0..cores)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}
fn registration(runtime: &Runtime, core: u32) -> ProcessRegistration {
    ProcessRegistration {
        priority: 44,
        ideal_vcpu: Some(VirtualCpuId::new(core)),
        affinity: runtime
            .scheduler()
            .profile()
            .core_set([VirtualCpuId::new(core)])
            .unwrap(),
    }
}
fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn exit() -> ExceptionDispatchOutcome<std::io::Error> {
    ExceptionDispatchOutcome::Terminate {
        scope: ExceptionTerminationScope::Process,
        exit_code: 0,
        reason: ExceptionTerminationReason::Requested,
    }
}
fn wait_exited(runtime: &mut Runtime, process: ProcessId) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while runtime.process(process).unwrap().lifecycle() != nixe_scheduler::ProcessLifecycle::Exited
    {
        assert!(Instant::now() < deadline, "autonomous process did not exit");
        runtime
            .wait_events(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
    }
}

struct ResumeOnce {
    calls: Mutex<Vec<(u32, u64, bool)>>,
    arrived: mpsc::Sender<()>,
    release: (Mutex<bool>, Condvar),
}
impl ExceptionDispatcher for ResumeOnce {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        self.calls.lock().unwrap().push((
            request.syndrome().unwrap() as u32,
            context.thread().state().read_x(x(0)),
            context.thread().state().is_resident(),
        ));
        if request.syndrome() == Some(0x42) {
            return ExceptionDispatchOutcome::Resume(ExceptionResume::Next);
        }
        self.arrived.send(()).unwrap();
        drop(
            self.release
                .1
                .wait_while(self.release.0.lock().unwrap(), |ready| !*ready)
                .unwrap(),
        );
        exit()
    }
}
#[test]
fn ordinary_calls_resume_locally_without_a_supervisor_message() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let process = synthetic_instruction_process(
        runtime.scheduler(),
        1,
        &[0x91000400, 0xd4000841, 0x91000800, 0xd4000861],
    );
    process.main_thread().registers().write_x(x(0), 0);
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let (send, receive) = mpsc::channel();
    let dispatcher = Arc::new(ResumeOnce {
        calls: Mutex::default(),
        arrived: send,
        release: (Mutex::new(false), Condvar::new()),
    });
    runtime.start(dispatcher.clone()).unwrap();
    receive.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(
        runtime.inbox.try_recv().unwrap().is_none(),
        "ordinary resume published a supervisor message"
    );
    assert_eq!(
        *dispatcher.calls.lock().unwrap(),
        [(0x42, 1, true), (0x43, 3, true)]
    );
    *dispatcher.release.0.lock().unwrap() = true;
    dispatcher.release.1.notify_one();
    wait_exited(&mut runtime, id);
    runtime.shutdown().unwrap();
}

struct Terminate;
impl ExceptionDispatcher for Terminate {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        _: &mut ExceptionDispatchContext<'_>,
        _: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        exit()
    }
}
#[test]
fn loader_return_captures_the_real_result_and_retires_native_contexts() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd65f03c0]);
    let thread = process.main_thread();
    let stub = thread.loader_return.unwrap();
    thread.registers().write_x(x(0), 0x8123456789abcdef);
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    runtime.start(Arc::new(Terminate)).unwrap();
    wait_exited(&mut runtime, id);
    let exit = runtime.process(id).unwrap().exit().unwrap();
    assert_eq!(exit.cause, crate::ProcessExitCause::LoaderReturned);
    assert_eq!(exit.exit_code, 0x8123456789abcdef);
    assert_eq!(exit.source.unwrap().pc, stub);
    assert!(thread.snapshot().is_some());
    assert!(!runtime.terminate_process(id).unwrap());
    assert_eq!(runtime.process(id).unwrap().exit().unwrap(), exit);
    let report = runtime.remove_process(id).unwrap();
    assert!(report.physical_pages_released > 0);
    runtime.shutdown().unwrap();
}

struct Migrate {
    calls: Mutex<Vec<u32>>,
}
impl ExceptionDispatcher for Migrate {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        self.calls
            .lock()
            .unwrap()
            .push(context.thread().vcpu().get());
        if request.syndrome() == Some(0x42) {
            let target = VirtualCpuId::new(1);
            let kernel = context.process().scheduler();
            kernel
                .migrate(
                    context.thread().id(),
                    nixe_scheduler::IdealCore::Core(target),
                    kernel.profile().core_set([target]).unwrap(),
                )
                .unwrap();
            context.thread_mut().state_mut().write_x(x(0), 0x1234);
            ExceptionDispatchOutcome::Resume(ExceptionResume::Next)
        } else {
            assert_eq!(context.thread().state().read_x(x(0)), 0x1234);
            exit()
        }
    }
}
#[test]
fn migration_releases_the_source_context_before_claim_on_the_destination() {
    let mut runtime = Runtime::try_new(kernel(2)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841, 0xd4000861]);
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let dispatcher = Arc::new(Migrate {
        calls: Mutex::default(),
    });
    runtime.start(dispatcher.clone()).unwrap();
    wait_exited(&mut runtime, id);
    assert_eq!(*dispatcher.calls.lock().unwrap(), [0, 1]);
    runtime.shutdown().unwrap();
}

#[test]
fn pause_acknowledges_native_ownership_and_freezes_the_saved_context() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0x91000400, 0x17ffffff]);
    let thread = process.main_thread();
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    runtime.start(Arc::new(Terminate)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while !runtime.scheduler().has_active_leases() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    runtime.set_paused(true).unwrap();
    assert!(!runtime.scheduler().has_active_leases());
    let context = thread.snapshot().unwrap().register_context();
    std::thread::sleep(Duration::from_millis(2));
    assert_eq!(thread.snapshot().unwrap().register_context(), context);
    runtime.set_paused(false).unwrap();
    runtime.terminate_process(id).unwrap();
    runtime.shutdown().unwrap();
}

#[test]
fn unallocated_native_instruction_reports_its_source_without_a_transport_error() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0]);
    let entry = process.entry_module().entry_address();
    runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    runtime.start(Arc::new(Terminate)).unwrap();
    let error = runtime.wait_events(Duration::from_secs(3)).unwrap_err();
    assert!(error.to_string().contains("unallocated"), "{error}");
    assert!(
        error.to_string().contains(&format!("{entry:016x}")),
        "{error}"
    );
    runtime.shutdown().unwrap();
}

struct BlockedCall {
    queue: Arc<KernelQueue>,
    caller: Arc<crate::ThreadObject>,
    source: nixe_cpu::location::LocationDescriptor,
    wait: Arc<nixe_scheduler::KernelWait>,
    ready: AtomicBool,
    cancelled: AtomicBool,
    consumed: AtomicBool,
    resumed_on: AtomicU64,
    failure: Mutex<Option<&'static str>>,
    delay_cancel: bool,
}
impl BlockedCall {
    fn complete(self: &Arc<Self>) {
        self.ready.store(true, Ordering::Release);
        self.queue.enqueue(self.clone());
    }
}
impl crate::ThreadWait for BlockedCall {
    fn cancel(self: Arc<Self>) {
        self.cancelled.store(true, Ordering::Release);
        if !self.delay_cancel {
            self.complete();
        }
    }
    fn caller(&self) -> &Arc<crate::ThreadObject> {
        &self.caller
    }
    fn source(&self) -> nixe_cpu::location::LocationDescriptor {
        self.source
    }
    fn token(&self) -> Option<nixe_scheduler::WakeToken> {
        Some(self.wait.token())
    }
    fn poll(self: Arc<Self>) -> std::task::Poll<Result<crate::WaitCompletion, DispatchFault>> {
        if !self.ready.load(Ordering::Acquire) {
            return std::task::Poll::Pending;
        }
        if let Some(message) = self.failure.lock().unwrap().take() {
            self.consumed.store(true, Ordering::Release);
            return std::task::Poll::Ready(Err(Box::new(std::io::Error::other(message))));
        }
        if self.cancelled.load(Ordering::Acquire) {
            self.consumed.store(true, Ordering::Release);
            return std::task::Poll::Ready(Ok(crate::WaitCompletion::Cancelled));
        }
        if self.consumed.swap(true, Ordering::AcqRel) {
            return std::task::Poll::Pending;
        }
        std::task::Poll::Ready(Ok(crate::WaitCompletion::Resume))
    }
    fn resume(
        self: Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<DispatchFault> {
        assert_eq!(
            context
                .process()
                .scheduler()
                .lease_for_vcpu(context.thread().vcpu())
                .map(|lease| lease.thread),
            Some(context.thread().id()),
            "a continuation must execute under its selected thread's ownership"
        );
        self.resumed_on.store(
            u64::from(context.thread().vcpu().get()) + 1,
            Ordering::Release,
        );
        context.thread_mut().state_mut().write_x(x(0), 0x1234);
        ExceptionDispatchOutcome::Resume(ExceptionResume::Next)
    }
}
struct Deferred {
    complete_inline: bool,
    delay_cancel: bool,
    pending: Mutex<Option<Arc<BlockedCall>>>,
    accepted: mpsc::Sender<()>,
}
impl ExceptionDispatcher for Deferred {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        if request.syndrome() == Some(0x43) {
            assert_eq!(context.thread().state().read_x(x(0)), 0x1235);
            return exit();
        }
        let wait = context
            .process()
            .scheduler()
            .register_wait(context.thread().id())
            .unwrap();
        wait.commit();
        let operation = Arc::new(BlockedCall {
            queue: context.completion_queue().clone(),
            caller: context.thread().object().clone(),
            source: request.source(),
            wait,
            ready: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            consumed: AtomicBool::new(false),
            resumed_on: AtomicU64::new(0),
            failure: Mutex::default(),
            delay_cancel: self.delay_cancel,
        });
        operation.caller.set_wait(operation.clone());
        *self.pending.lock().unwrap() = Some(operation.clone());
        if self.complete_inline {
            operation.complete();
        }
        self.accepted.send(()).unwrap();
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Next)
    }
}
#[test]
fn a_blocked_call_completes_once_even_when_notified_before_the_core_parks() {
    for complete_inline in [false, true] {
        let mut runtime = Runtime::try_new(kernel(1)).unwrap();
        let process = synthetic_instruction_process(
            runtime.scheduler(),
            1,
            &[0xd4000841, 0x91000400, 0xd4000861],
        );
        let id = runtime
            .register_process(process, registration(&runtime, 0))
            .unwrap();
        let (send, receive) = mpsc::channel();
        let dispatcher = Arc::new(Deferred {
            complete_inline,
            delay_cancel: false,
            pending: Mutex::default(),
            accepted: send,
        });
        runtime.start(dispatcher.clone()).unwrap();
        receive.recv_timeout(Duration::from_secs(3)).unwrap();
        if !complete_inline {
            dispatcher
                .pending
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .complete();
        }
        wait_exited(&mut runtime, id);
        runtime.shutdown().unwrap();
    }
}
#[test]
fn a_completed_call_stays_paused_and_continues_on_its_selected_core() {
    let mut runtime = Runtime::try_new(kernel(2)).unwrap();
    let process = synthetic_instruction_process(
        runtime.scheduler(),
        1,
        &[0xd4000841, 0x91000400, 0xd4000861],
    );
    let thread = process.main_thread();
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let (send, receive) = mpsc::channel();
    let dispatcher = Arc::new(Deferred {
        complete_inline: false,
        delay_cancel: false,
        pending: Mutex::default(),
        accepted: send,
    });
    runtime.start(dispatcher.clone()).unwrap();
    receive.recv_timeout(Duration::from_secs(3)).unwrap();
    runtime.set_paused(true).unwrap();
    let core = VirtualCpuId::new(1);
    runtime
        .scheduler()
        .migrate(
            thread.id(),
            nixe_scheduler::IdealCore::Core(core),
            runtime.scheduler().profile().core_set([core]).unwrap(),
        )
        .unwrap();
    let operation = dispatcher.pending.lock().unwrap().as_ref().unwrap().clone();
    operation.complete();
    let deadline = Instant::now() + Duration::from_secs(3);
    while runtime.scheduler().thread(thread.id()).unwrap().lifecycle
        != nixe_scheduler::ThreadLifecycle::Ready
    {
        assert!(
            Instant::now() < deadline,
            "completed call was not made ready"
        );
        std::thread::yield_now();
    }
    assert_eq!(operation.resumed_on.load(Ordering::Acquire), 0);
    assert!(thread.pending_wait().is_some());
    // A duplicate completion must not consume a ready continuation.
    operation.complete();
    runtime.set_paused(false).unwrap();
    wait_exited(&mut runtime, id);
    assert_eq!(operation.resumed_on.load(Ordering::Acquire), 2);
    runtime.shutdown().unwrap();
}

#[test]
fn cancellation_discards_a_stale_result_and_drains_its_owned_call() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841, 0xd4000861]);
    let thread = process.main_thread();
    thread.registers().write_x(x(0), 0x5678);
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let (send, receive) = mpsc::channel();
    let dispatcher = Arc::new(Deferred {
        complete_inline: false,
        delay_cancel: false,
        pending: Mutex::default(),
        accepted: send,
    });
    runtime.start(dispatcher.clone()).unwrap();
    receive.recv_timeout(Duration::from_secs(3)).unwrap();
    runtime.remove_process(id).unwrap();
    assert!(thread.pending_wait().is_none());
    assert_eq!(thread.snapshot().unwrap().read_x(x(0)), 0x5678);
    assert!(
        dispatcher
            .pending
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .cancelled
            .load(Ordering::Acquire)
    );
    runtime.shutdown().unwrap();
}
struct SlowService {
    entered: mpsc::Sender<()>,
    release: (Mutex<bool>, Condvar),
    other_calls: AtomicU64,
}
impl ExceptionDispatcher for SlowService {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        _: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        if request.syndrome() == Some(0x42) {
            self.entered.send(()).unwrap();
            drop(
                self.release
                    .1
                    .wait_while(self.release.0.lock().unwrap(), |released| !*released)
                    .unwrap(),
            );
            exit()
        } else {
            self.other_calls.fetch_add(1, Ordering::Relaxed);
            ExceptionDispatchOutcome::Resume(ExceptionResume::Retry)
        }
    }
}
#[test]
fn a_slow_service_and_process_termination_do_not_serialize_unrelated_cores() {
    let mut runtime = Runtime::try_new(kernel(2)).unwrap();
    let slow = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841]);
    let other = synthetic_instruction_process(runtime.scheduler(), 2, &[0xd4000861]);
    let slow_id = runtime
        .register_process(slow, registration(&runtime, 0))
        .unwrap();
    let other_id = runtime
        .register_process(other, registration(&runtime, 1))
        .unwrap();
    let (send, receive) = mpsc::channel();
    let dispatcher = Arc::new(SlowService {
        entered: send,
        release: (Mutex::new(false), Condvar::new()),
        other_calls: AtomicU64::new(0),
    });
    runtime.start(dispatcher.clone()).unwrap();
    receive.recv_timeout(Duration::from_secs(3)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while dispatcher.other_calls.load(Ordering::Relaxed) < 10 {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    runtime.remove_process(other_id).unwrap();
    assert!(runtime.process(slow_id).is_some());
    *dispatcher.release.0.lock().unwrap() = true;
    dispatcher.release.1.notify_one();
    wait_exited(&mut runtime, slow_id);
    runtime.shutdown().unwrap();
}

struct PanicAfterAcceptance(Arc<Deferred>);
impl ExceptionDispatcher for PanicAfterAcceptance {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        let _ = self.0.dispatch(context, request);
        panic!("injected native owner failure after acceptance");
    }
}

#[test]
fn termination_drains_an_accepted_effect_even_after_its_physical_core_panics() {
    for panic in [false, true] {
        let mut runtime = Runtime::try_new(kernel(1)).unwrap();
        let process =
            synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841, 0xd4000861]);
        let thread = process.main_thread();
        thread.registers().write_x(x(0), 0x9876);
        let id = runtime
            .register_process(process, registration(&runtime, 0))
            .unwrap();
        let (accepted, receive) = mpsc::channel();
        let dispatcher = Arc::new(Deferred {
            complete_inline: false,
            delay_cancel: true,
            pending: Mutex::default(),
            accepted,
        });
        if panic {
            runtime
                .start(Arc::new(PanicAfterAcceptance(dispatcher.clone())))
                .unwrap();
        } else {
            runtime.start(dispatcher.clone()).unwrap();
        }
        receive.recv_timeout(Duration::from_secs(3)).unwrap();
        let operation = dispatcher.pending.lock().unwrap().as_ref().unwrap().clone();
        let finish = operation.clone();
        let effect = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            while !finish.cancelled.load(Ordering::Acquire)
                || finish.caller.owner().unwrap().has_native_threads()
            {
                assert!(
                    Instant::now() < deadline,
                    "accepted operation was not cancelled"
                );
                std::thread::yield_now();
            }
            finish.complete();
        });
        runtime.remove_process(id).unwrap();
        effect.join().unwrap();
        assert!(thread.pending_wait().is_none());
        assert!(operation.consumed.load(Ordering::Acquire));
        assert_eq!(thread.snapshot().unwrap().read_x(x(0)), 0x9876);
        let shutdown = runtime.shutdown();
        if panic {
            assert!(matches!(shutdown, Err(RuntimeError::CorePanicked(_))));
        } else {
            shutdown.unwrap();
        }
    }
}

struct FixedCounter(Mutex<Vec<(u64, u64)>>);
impl ExceptionDispatcher for FixedCounter {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        let state = context.thread().state();
        let value = (state.read_x(x(0)), state.read_x(x(1)));
        let mut calls = self.0.lock().unwrap();
        calls.push(value);
        if calls.len() == 1 {
            context
                .process()
                .scheduler()
                .clock()
                .advance_scheduler_to(1_000_000_000);
            ExceptionDispatchOutcome::Resume(ExceptionResume::Next)
        } else {
            exit()
        }
    }
}
#[test]
fn fixed_time_diagnostics_use_the_production_core_and_local_svc_path() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    // MRS X0,CNTPCT_EL0; MRS X1,CNTFRQ_EL0; SVC; B back to MRS.
    let process = synthetic_instruction_process(
        runtime.scheduler(),
        1,
        &[0xd53be020, 0xd53be001, 0xd4000841, 0x17fffffd],
    );
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let dispatcher = Arc::new(FixedCounter(Mutex::default()));
    runtime.start(dispatcher.clone()).unwrap();
    wait_exited(&mut runtime, id);
    let calls = dispatcher.0.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, 0);
    assert!(calls[0].1 > 0);
    assert_eq!(calls[1], (calls[0].1, calls[0].1));
    drop(calls);
    runtime.shutdown().unwrap();
}

struct CodeChange {
    observed: mpsc::Sender<u64>,
}
impl ExceptionDispatcher for CodeChange {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        let value = context.thread().state().read_x(x(0));
        if value == 2 {
            self.observed.send(value).unwrap();
            exit()
        } else {
            if self.observed.send(value).is_err() {
                return exit();
            }
            ExceptionDispatchOutcome::Resume(ExceptionResume::Next)
        }
    }
}
#[test]
fn a_running_native_core_observes_host_code_invalidation_before_reentry() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    // MOV X0,#1; SVC; B back to MOV.
    let process = synthetic_instruction_process(
        runtime.scheduler(),
        1,
        &[0xd2800020, 0xd4000841, 0x17fffffe],
    );
    let entry = process.entry_module().entry_address();
    let memory = process.memory().clone();
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let (observed, receive) = mpsc::channel();
    runtime.start(Arc::new(CodeChange { observed })).unwrap();
    assert_eq!(receive.recv_timeout(Duration::from_secs(3)).unwrap(), 1);
    let ranges = memory
        .space()
        .capture(
            nixe_memory::GuestVirtualAddress::new(entry),
            4,
            nixe_memory::MemoryPermissions::EXECUTE,
        )
        .unwrap();
    memory
        .account()
        .machine()
        .write_resolved(&ranges, &0xd2800040_u32.to_le_bytes())
        .unwrap();
    wait_exited(&mut runtime, id);
    assert_eq!(receive.try_iter().last(), Some(2));
    runtime.shutdown().unwrap();
}

struct RacingExits {
    entered: (Mutex<bool>, Condvar),
    release: (Mutex<bool>, Condvar),
}
impl ExceptionDispatcher for RacingExits {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        _: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        let late = request.syndrome() == Some(0x42);
        if late {
            *self.entered.0.lock().unwrap() = true;
            self.entered.1.notify_one();
            drop(
                self.release
                    .1
                    .wait_while(self.release.0.lock().unwrap(), |ready| !*ready)
                    .unwrap(),
            );
        } else {
            drop(
                self.entered
                    .1
                    .wait_while(self.entered.0.lock().unwrap(), |entered| !*entered)
                    .unwrap(),
            );
        }
        ExceptionDispatchOutcome::Terminate {
            scope: ExceptionTerminationScope::Process,
            exit_code: if late { 2 } else { 1 },
            reason: ExceptionTerminationReason::Break {
                reason: 0,
                info: if late { 0x2000 } else { 0x1000 },
                size: 0,
                payload: None,
            },
        }
    }
}
#[test]
fn the_first_process_exit_survives_a_late_svc_from_another_core() {
    let mut runtime = Runtime::try_new(kernel(2)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841, 0xd4000861]);
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let process = runtime.process(id).unwrap();
    let main = process.main_thread();
    let core = VirtualCpuId::new(1);
    let mut create = Box::pin(process.create_thread(crate::ThreadCreateRequest {
        entry: nixe_memory::GuestVirtualAddress::new(main.registers().pc() + 4),
        argument: 0,
        fpcr: 0,
        fpsr: 0,
        stack_top: main.stack_top,
        priority: 44,
        ideal_vcpu: Some(core),
        affinity: runtime.scheduler().profile().core_set([core]).unwrap(),
    }));
    let std::task::Poll::Ready(Ok(created)) = std::future::Future::poll(
        create.as_mut(),
        &mut std::task::Context::from_waker(std::task::Waker::noop()),
    ) else {
        panic!("RAM thread creation did not complete");
    };
    process.thread(created.id).unwrap().start().unwrap();
    let dispatcher = Arc::new(RacingExits {
        entered: Default::default(),
        release: Default::default(),
    });
    runtime.start(dispatcher.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while process.generation() == 1 {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    *dispatcher.release.0.lock().unwrap() = true;
    dispatcher.release.1.notify_one();
    wait_exited(&mut runtime, id);
    assert_eq!(process.exit().unwrap().exit_code, 1);
    assert!(matches!(
        process.exit().unwrap().cause,
        crate::ProcessExitCause::GuestBreak { info: 0x1000, .. }
    ));
    runtime.shutdown().unwrap();
}

struct FatalCall {
    sibling_calls: AtomicU64,
    independent_calls: AtomicU64,
}
impl ExceptionDispatcher for FatalCall {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        _context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        match request.syndrome() {
            Some(0x42) => {
                let deadline = Instant::now() + Duration::from_secs(3);
                while self.sibling_calls.load(Ordering::Acquire) == 0 {
                    assert!(Instant::now() < deadline, "sibling thread did not run");
                    std::thread::yield_now();
                }
                ExceptionDispatchOutcome::Fault(std::io::Error::other("injected fatal SVC"))
            }
            Some(0x43) => {
                self.sibling_calls.fetch_add(1, Ordering::Release);
                ExceptionDispatchOutcome::Resume(ExceptionResume::Retry)
            }
            Some(0x44) => {
                self.independent_calls.fetch_add(1, Ordering::Release);
                ExceptionDispatchOutcome::Resume(ExceptionResume::Retry)
            }
            _ => unreachable!("fixture SVC"),
        }
    }
}

#[test]
fn a_fatal_svc_stops_its_process_before_reporting_and_leaves_other_processes_running() {
    let mut runtime = Runtime::try_new(kernel(3)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841, 0xd4000861]);
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let process = runtime.process(id).unwrap();
    let main = process.main_thread();
    let core = VirtualCpuId::new(1);
    let mut create = Box::pin(process.create_thread(crate::ThreadCreateRequest {
        entry: nixe_memory::GuestVirtualAddress::new(main.registers().pc() + 4),
        argument: 0,
        fpcr: 0,
        fpsr: 0,
        stack_top: main.stack_top,
        priority: 44,
        ideal_vcpu: Some(core),
        affinity: runtime.scheduler().profile().core_set([core]).unwrap(),
    }));
    let std::task::Poll::Ready(Ok(created)) = std::future::Future::poll(
        create.as_mut(),
        &mut std::task::Context::from_waker(Waker::noop()),
    ) else {
        panic!("RAM thread creation did not complete")
    };
    process.thread(created.id).unwrap().start().unwrap();
    let independent = synthetic_instruction_process(runtime.scheduler(), 2, &[0xd4000881]);
    let independent_id = runtime
        .register_process(independent, registration(&runtime, 2))
        .unwrap();
    let dispatcher = Arc::new(FatalCall {
        sibling_calls: AtomicU64::new(0),
        independent_calls: AtomicU64::new(0),
    });
    runtime.start(dispatcher.clone()).unwrap();
    let fault = runtime.wait_events(Duration::from_secs(3)).unwrap_err();
    assert!(fault.to_string().contains("injected fatal SVC"), "{fault}");
    let terminal_claimed = matches!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Terminating | nixe_scheduler::ProcessLifecycle::Faulted
    );
    let deadline = Instant::now() + Duration::from_secs(3);
    while process.lifecycle() == nixe_scheduler::ProcessLifecycle::Terminating
        && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    let autonomously_faulted = process.lifecycle() == nixe_scheduler::ProcessLifecycle::Faulted;
    let stopped_sibling = dispatcher.sibling_calls.load(Ordering::Acquire);
    let before = dispatcher.independent_calls.load(Ordering::Acquire);
    let deadline = Instant::now() + Duration::from_secs(3);
    while dispatcher.independent_calls.load(Ordering::Acquire) <= before {
        assert!(Instant::now() < deadline, "an unrelated process stopped");
        std::thread::yield_now();
    }
    let sibling_stayed_stopped =
        dispatcher.sibling_calls.load(Ordering::Acquire) == stopped_sibling;
    let report = runtime.remove_process(id).unwrap();
    runtime.remove_process(independent_id).unwrap();
    runtime.shutdown().unwrap();
    assert!(
        terminal_claimed,
        "fatal SVC left its process running until supervisor removal"
    );
    assert!(
        autonomously_faulted,
        "process termination required supervisor progress"
    );
    assert!(
        sibling_stayed_stopped,
        "sibling thread continued after process termination"
    );
    assert_eq!(
        report.previous_lifecycle,
        nixe_scheduler::ProcessLifecycle::Faulted
    );
}

#[test]
fn a_failure_while_paused_reaches_the_supervisor_without_a_resume_request() {
    for queued_before_wait in [false, true] {
        let mut runtime = Runtime::try_new(kernel(1)).unwrap();
        let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841]);
        let id = runtime
            .register_process(process, registration(&runtime, 0))
            .unwrap();
        let (accepted, receive) = mpsc::channel();
        let dispatcher = Arc::new(Deferred {
            complete_inline: false,
            delay_cancel: false,
            pending: Mutex::default(),
            accepted,
        });
        runtime.start(dispatcher.clone()).unwrap();
        receive.recv_timeout(Duration::from_secs(3)).unwrap();
        runtime.set_paused(true).unwrap();
        let events = runtime.event_sender();
        let operation = dispatcher.pending.lock().unwrap().as_ref().unwrap().clone();
        *operation.failure.lock().unwrap() = Some("injected paused device failure");
        if queued_before_wait {
            operation.complete();
            let deadline = Instant::now() + Duration::from_secs(3);
            while runtime.shared.faults.lock().unwrap().is_empty() {
                assert!(
                    Instant::now() < deadline,
                    "core did not publish its failure"
                );
                std::thread::yield_now();
            }
            // A cleanup/status consumer may already have handled the wakeup.
            // The terminal error remains authoritative until it is reported.
            while let Some(event) = runtime.inbox.try_recv().unwrap() {
                runtime.handle_event(event).unwrap();
            }
        }
        let (finished, completed) = mpsc::channel();
        let supervisor = std::thread::spawn(move || {
            let result = runtime.wait_while_host_paused();
            finished.send(()).unwrap();
            (runtime, result)
        });
        if !queued_before_wait {
            operation.complete();
        }
        let returned_without_resume = completed.recv_timeout(Duration::from_secs(3)).is_ok();
        if !returned_without_resume {
            // Bound cleanup of the failing implementation.
            events
                .submit(crate::ExternalEvent::HostPause(false))
                .unwrap();
        }
        let (mut runtime, result) = supervisor.join().unwrap();
        let stayed_paused = runtime.host_paused();
        runtime.remove_process(id).unwrap();
        let shutdown = runtime.shutdown();
        assert!(
            returned_without_resume && stayed_paused,
            "paused supervision ignored the core's failure"
        );
        assert!(
            matches!(result, Err(RuntimeError::Execution { diagnostic, .. })
            if diagnostic.contains("injected paused device failure"))
        );
        shutdown.unwrap();
    }
}

struct CompletionBatch {
    failed: Arc<Deferred>,
    healthy: Arc<Deferred>,
    gate: Arc<SlowService>,
}
impl ExceptionDispatcher for CompletionBatch {
    type Fault = std::io::Error;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        match context.process().process_id() {
            1 => self.failed.dispatch(context, request),
            2 => self.healthy.dispatch(context, request),
            _ => self.gate.dispatch(context, request),
        }
    }
}

#[test]
fn a_completed_failure_cannot_strand_another_process_in_the_same_ready_batch() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let mut ids = Vec::new();
    for identity in 1..=3 {
        let process = synthetic_instruction_process(
            runtime.scheduler(),
            identity,
            &[0xd4000841, 0x91000400, 0xd4000861],
        );
        ids.push(
            runtime
                .register_process(process, registration(&runtime, 0))
                .unwrap(),
        );
    }
    let (failed_accepted, failed_call) = mpsc::channel();
    let (healthy_accepted, healthy_call) = mpsc::channel();
    let (gate_entered, gate_ready) = mpsc::channel();
    let dispatcher = Arc::new(CompletionBatch {
        failed: Arc::new(Deferred {
            complete_inline: false,
            delay_cancel: true,
            pending: Mutex::default(),
            accepted: failed_accepted,
        }),
        healthy: Arc::new(Deferred {
            complete_inline: false,
            delay_cancel: false,
            pending: Mutex::default(),
            accepted: healthy_accepted,
        }),
        gate: Arc::new(SlowService {
            entered: gate_entered,
            release: (Mutex::new(false), Condvar::new()),
            other_calls: AtomicU64::new(0),
        }),
    });
    runtime.start(dispatcher.clone()).unwrap();
    failed_call.recv_timeout(Duration::from_secs(3)).unwrap();
    healthy_call.recv_timeout(Duration::from_secs(3)).unwrap();
    gate_ready.recv_timeout(Duration::from_secs(3)).unwrap();
    runtime.terminate_process(ids[0]).unwrap();
    let failed = dispatcher
        .failed
        .pending
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    let healthy = dispatcher
        .healthy
        .pending
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    *failed.failure.lock().unwrap() = Some("injected ready-batch failure");
    failed.complete();
    healthy.complete();
    *dispatcher.gate.release.0.lock().unwrap() = true;
    dispatcher.gate.release.1.notify_one();

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut failure = None;
    while !matches!(
        runtime.process(ids[1]).unwrap().lifecycle(),
        nixe_scheduler::ProcessLifecycle::Exited | nixe_scheduler::ProcessLifecycle::Faulted
    ) && Instant::now() < deadline
    {
        if let Err(error) = runtime.wait_events(deadline.saturating_duration_since(Instant::now()))
        {
            failure = Some(error);
        }
    }
    let progressed =
        runtime.process(ids[1]).unwrap().lifecycle() == nixe_scheduler::ProcessLifecycle::Exited;
    if failure.is_none() {
        failure = runtime.drain_external_events().err();
    }
    runtime.shutdown().unwrap();
    assert!(
        progressed,
        "a ready completion waited for an unrelated new notification"
    );
    assert_eq!(healthy.resumed_on.load(Ordering::Acquire), 1);
    assert!(
        matches!(failure, Some(RuntimeError::Execution { diagnostic, .. })
        if diagnostic.contains("injected ready-batch failure"))
    );
}

#[test]
fn supervisor_waits_report_a_lost_host_notification_without_resuming_the_guest() {
    for paused in [true, false] {
        let mut runtime = Runtime::try_new(kernel(1)).unwrap();
        runtime.set_paused(paused).unwrap();
        let events = runtime.event_sender();
        for _ in 0..=1024 {
            events.notify_host_service(ExternalEventSource::Input);
        }
        let (finished, completed) = mpsc::channel();
        let supervisor = std::thread::spawn(move || {
            let result = if paused {
                runtime.wait_while_host_paused()
            } else {
                runtime.wait_events(Duration::ZERO)
            };
            finished.send(()).unwrap();
            (runtime, result)
        });
        let returned = completed.recv_timeout(Duration::from_secs(3)).is_ok();
        if !returned {
            events.submit(ExternalEvent::HostPause(false)).unwrap();
        }
        let (mut runtime, result) = supervisor.join().unwrap();
        let remained_paused = runtime.host_paused() == paused;
        let shutdown = runtime.shutdown();
        assert!(
            returned && remained_paused,
            "lost notification was hidden by host wait"
        );
        assert!(
            matches!(
                result,
                Err(RuntimeError::Host(crate::ExternalEventSendError::Full))
            ),
            "{result:?}"
        );
        shutdown.unwrap();
    }
}

#[test]
fn shutdown_drains_an_accepted_effect_before_reporting_a_lost_host_notification() {
    let mut runtime = Runtime::try_new(kernel(1)).unwrap();
    let process = synthetic_instruction_process(runtime.scheduler(), 1, &[0xd4000841]);
    let thread = process.main_thread();
    let id = runtime
        .register_process(process, registration(&runtime, 0))
        .unwrap();
    let (accepted, receive) = mpsc::channel();
    let dispatcher = Arc::new(Deferred {
        complete_inline: false,
        delay_cancel: true,
        pending: Mutex::default(),
        accepted,
    });
    runtime.start(dispatcher.clone()).unwrap();
    receive.recv_timeout(Duration::from_secs(3)).unwrap();
    let operation = dispatcher.pending.lock().unwrap().as_ref().unwrap().clone();
    let events = runtime.event_sender();
    for _ in 0..=1024 {
        events.notify_host_service(ExternalEventSource::Input);
    }
    let finish = operation.clone();
    let effect = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !finish.cancelled.load(Ordering::Acquire)
            || finish.caller.owner().unwrap().has_native_threads()
        {
            assert!(
                Instant::now() < deadline,
                "accepted effect was not cancelled"
            );
            std::thread::yield_now();
        }
        finish.complete();
    });
    let result = runtime.shutdown();
    effect.join().unwrap();
    assert!(operation.consumed.load(Ordering::Acquire));
    assert!(thread.pending_wait().is_none());
    assert!(runtime.process(id).is_none());
    assert!(
        matches!(
            result,
            Err(RuntimeError::Host(crate::ExternalEventSendError::Full))
        ),
        "{result:?}"
    );
}
