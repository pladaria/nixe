use super::*;
use crate::integration_test_support as support;
use nixe_runtime::{ExceptionDispatchRequest, ExceptionDispatcher, ProcessRegistration, Runtime};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

struct ObserveIpc {
    horizon: Arc<HorizonSvcDispatcher>,
    replies: mpsc::Sender<u64>,
}
impl ExceptionDispatcher for ObserveIpc {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        if request.syndrome() == Some(0x10) {
            self.replies
                .send(read_register(context.thread().state(), 0))
                .unwrap();
        }
        self.horizon.dispatch(context, request)
    }
}

#[test]
fn queued_ipc_callers_complete_on_autonomous_cores_without_being_cancelled() {
    let svc = |id: u32| 0xd4000001 | (id << 5);
    let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
        &[svc(0x21), svc(0x10), svc(0x0a)],
        Default::default(),
        support::process_builder(),
    );
    let kernel = process.scheduler().clone();
    let core = VirtualCpuId::new(0);
    let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
    let id = runtime
        .register_process(
            process,
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(core),
                affinity: kernel.profile().core_set([core]).unwrap(),
            },
        )
        .unwrap();
    let process = runtime.process(id).unwrap();
    let main = process.main_thread();
    let entry = GuestVirtualAddress::new(main.registers().pc());
    let mut threads = vec![main.clone()];
    for index in 1..8 {
        let core = VirtualCpuId::new(index % 4);
        let created = support::ready(process.create_thread(nixe_runtime::ThreadCreateRequest {
            entry,
            argument: 0,
            fpcr: 0,
            fpsr: 0,
            stack_top: main.stack_top,
            priority: 44,
            ideal_vcpu: Some(core),
            affinity: kernel.profile().core_set([core]).unwrap(),
        }))
        .unwrap();
        let thread = process.thread(created.id).unwrap();
        thread.start().unwrap();
        threads.push(thread);
    }
    let handle = process
        .handles()
        .insert(nixe_horizon::ServiceSession::new(
            crate::HorizonIpcObject::SemanticService(crate::IpcSession::new(
                crate::IpcService::FileSystem,
            )),
        ))
        .unwrap();
    let horizon = support::dispatcher(&process);
    let mut command = [0u8; 0x100];
    for (offset, value) in [(0, 4u32), (4, 8), (16, 0x49434653), (24, 1005)] {
        command[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for thread in &threads {
        process
            .memory()
            .space()
            .write(thread.tls_base, &command)
            .unwrap()
            .into_ready()
            .unwrap();
        write_register(&mut thread.registers(), 0, u64::from(handle));
    }
    let (replies, receive) = mpsc::channel();
    runtime
        .start(Arc::new(ObserveIpc { horizon, replies }))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    for _ in 0..threads.len() {
        assert_eq!(receive.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
    }
    while process.lifecycle() != nixe_scheduler::ProcessLifecycle::Exited {
        assert!(Instant::now() < deadline);
        runtime.wait_events(Duration::from_millis(50)).unwrap();
    }
    assert_eq!(runtime.rejected_svc_kinds(), 0);
    runtime.shutdown().unwrap();
}

#[test]
fn releasing_a_completion_queue_drops_its_retired_calls() {
    let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
        &[0xd4000001],
        Default::default(),
        support::process_builder(),
    );
    let process = Arc::new(process);
    let caller = process.main_thread();
    let queue = nixe_runtime::KernelQueue::new(Waker::noop().clone());
    let weak_queue = Arc::downgrade(&queue);
    let pc = GuestVirtualAddress::new(caller.registers().pc());
    let operation = Arc::new(SvcWait {
        queued: AtomicBool::new(true),
        queue: Arc::downgrade(&queue),
        dispatcher: support::dispatcher(&process),
        owner: process,
        source: nixe_cpu::location::LocationDescriptor::new(
            pc,
            nixe_cpu::profile::CpuProfileId::new(1),
        ),
        caller,
        process_generation: 1,
        state: Mutex::default(),
    });
    let retired = Arc::downgrade(&operation);
    queue.enqueue(operation);
    drop(queue);
    assert!(
        weak_queue.upgrade().is_none(),
        "retired calls retained their executor queue"
    );
    assert!(
        retired.upgrade().is_none(),
        "the queue retained retired request resources"
    );
}

struct PanickingService {
    horizon: Arc<HorizonSvcDispatcher>,
    lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
    ready: Arc<AtomicBool>,
    wake: Arc<Mutex<Option<Waker>>>,
}

struct SlowAcceptedService {
    horizon: Arc<HorizonSvcDispatcher>,
    lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
    ready: AtomicBool,
    wake: Mutex<Option<Waker>>,
    polling: mpsc::Sender<()>,
    release: (Mutex<bool>, std::sync::Condvar),
}
impl ExceptionDispatcher for SlowAcceptedService {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        self.lifetime.begin().unwrap();
        let service = self.clone();
        self.horizon.start_owned_svc(
            context,
            self.lifetime.clone(),
            None,
            std::future::poll_fn(move |context| {
                if !service.ready.load(Ordering::Acquire) {
                    *service.wake.lock().unwrap() = Some(context.waker().clone());
                    return Poll::Pending;
                }
                service.polling.send(()).unwrap();
                drop(
                    service
                        .release
                        .1
                        .wait_while(service.release.0.lock().unwrap(), |released| !*released)
                        .unwrap(),
                );
                Poll::Ready(Ok(SvcCompletion::success(Some(0xbeef))))
            }),
        )
    }
}

#[test]
fn termination_can_cancel_a_polling_service_and_waits_for_its_accepted_effect() {
    let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
        &[0xd4000841],
        Default::default(),
        support::process_builder(),
    );
    let kernel = process.scheduler().clone();
    let core = VirtualCpuId::new(0);
    let caller = process.main_thread();
    write_register(&mut caller.registers(), 0, 0x1234);
    let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
    let id = runtime
        .register_process(
            process,
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(core),
                affinity: kernel.profile().core_set([core]).unwrap(),
            },
        )
        .unwrap();
    let process = runtime.process(id).unwrap();
    let (polling, receive) = mpsc::channel();
    let dispatcher = Arc::new(SlowAcceptedService {
        horizon: support::dispatcher(&process),
        lifetime: Arc::default(),
        ready: AtomicBool::new(false),
        wake: Mutex::default(),
        polling,
        release: Default::default(),
    });
    runtime.start(dispatcher.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while caller.pending_wait().is_none() {
        assert!(Instant::now() < deadline, "service did not suspend");
        std::thread::yield_now();
    }
    dispatcher.ready.store(true, Ordering::Release);
    dispatcher.wake.lock().unwrap().take().unwrap().wake();
    receive.recv_timeout(Duration::from_secs(5)).unwrap();
    let termination = std::thread::spawn(move || {
        runtime.remove_process(id).unwrap();
        runtime.shutdown().unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while dispatcher.lifetime.check().is_ok() && Instant::now() < deadline {
        std::thread::yield_now();
    }
    let cancelled = dispatcher.lifetime.check().is_err();
    let retained = caller.pending_wait().is_some();
    let draining = !termination.is_finished();
    *dispatcher.release.0.lock().unwrap() = true;
    dispatcher.release.1.notify_one();
    termination.join().unwrap();
    assert!(cancelled, "cancellation waited for the service's poll lock");
    assert!(
        retained && draining,
        "an executing effect was released before completion"
    );
    assert!(caller.pending_wait().is_none());
    assert_eq!(caller.snapshot().unwrap().register_context().x[0], 0x1234);
}
impl ExceptionDispatcher for PanickingService {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        let ready = self.ready.clone();
        let wake = self.wake.clone();
        let lifetime = self.lifetime.clone();
        lifetime.begin().unwrap();
        self.horizon.start_owned_svc(
            context,
            lifetime,
            None,
            std::future::poll_fn(move |context| {
                assert!(!ready.load(Ordering::Acquire), "injected service panic");
                *wake.lock().unwrap() = Some(context.waker().clone());
                Poll::Pending
            }),
        )
    }
}

#[test]
fn a_panicking_suspended_service_releases_its_wait_without_aborting_the_host() {
    let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
        &[0xd4000841],
        Default::default(),
        support::process_builder(),
    );
    let kernel = process.scheduler().clone();
    let core = VirtualCpuId::new(0);
    let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
    let id = runtime
        .register_process(
            process,
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(core),
                affinity: kernel.profile().core_set([core]).unwrap(),
            },
        )
        .unwrap();
    let process = runtime.process(id).unwrap();
    let caller = process.main_thread();
    let dispatcher = Arc::new(PanickingService {
        horizon: support::dispatcher(&process),
        lifetime: Arc::default(),
        ready: Arc::new(AtomicBool::new(false)),
        wake: Arc::default(),
    });
    runtime.start(dispatcher.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while caller.pending_wait().is_none() {
        assert!(Instant::now() < deadline, "service did not suspend");
        std::thread::yield_now();
    }
    dispatcher.ready.store(true, Ordering::Release);
    dispatcher.wake.lock().unwrap().take().unwrap().wake();
    let mut core_panicked = false;
    while process.lifecycle() == nixe_scheduler::ProcessLifecycle::Running {
        assert!(Instant::now() < deadline, "physical core did not fail");
        if let Err(error) = runtime.wait_events(Duration::from_millis(50)) {
            assert!(matches!(error, nixe_runtime::RuntimeError::CorePanicked(_)));
            core_panicked = true;
        }
    }
    runtime.remove_process(id).unwrap();
    assert!(caller.pending_wait().is_none());
    assert!(dispatcher.lifetime.check().is_err());
    if let Err(error) = runtime.shutdown() {
        assert!(matches!(error, nixe_runtime::RuntimeError::CorePanicked(_)));
        core_panicked = true;
    }
    assert!(core_panicked, "physical-core failure was not reported");
}

struct AdmissionRace {
    horizon: Arc<HorizonSvcDispatcher>,
    lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
    started: bool,
    drained: AtomicBool,
    entered: mpsc::Sender<()>,
    release: (Mutex<bool>, std::sync::Condvar),
}
impl ExceptionDispatcher for AdmissionRace {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        let service = self.clone();
        let mut first = true;
        self.horizon.start_owned_svc(
            context,
            self.lifetime.clone(),
            None,
            std::future::poll_fn(move |_| {
                if first {
                    first = false;
                    if service.started {
                        service.lifetime.begin().unwrap();
                    }
                    service.entered.send(()).unwrap();
                    drop(
                        service
                            .release
                            .1
                            .wait_while(service.release.0.lock().unwrap(), |released| !*released)
                            .unwrap(),
                    );
                }
                if service.lifetime.check().is_err() {
                    service.drained.store(true, Ordering::Release);
                    return Poll::Ready(Ok(SvcCompletion::success(None)));
                }
                Poll::Pending
            }),
        )
    }
}

#[test]
fn termination_during_wait_registration_is_cancellation_instead_of_a_fatal_service_error() {
    for started in [false, true] {
        let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
            &[0xd4000841],
            Default::default(),
            support::process_builder(),
        );
        let kernel = process.scheduler().clone();
        let core = VirtualCpuId::new(0);
        let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
        let id = runtime
            .register_process(
                process,
                ProcessRegistration {
                    priority: 44,
                    ideal_vcpu: Some(core),
                    affinity: kernel.profile().core_set([core]).unwrap(),
                },
            )
            .unwrap();
        let process = runtime.process(id).unwrap();
        let caller = process.main_thread();
        let (entered, receive) = mpsc::channel();
        let dispatcher = Arc::new(AdmissionRace {
            horizon: support::dispatcher(&process),
            lifetime: Arc::default(),
            started,
            drained: AtomicBool::new(false),
            entered,
            release: Default::default(),
        });
        runtime.start(dispatcher.clone()).unwrap();
        receive.recv_timeout(Duration::from_secs(5)).unwrap();
        let termination = std::thread::spawn(move || {
            runtime.remove_process(id).unwrap();
            runtime.shutdown()
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while kernel.selection(core).unwrap().thread.is_some() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let closed = kernel.selection(core).unwrap().thread.is_none();
        *dispatcher.release.0.lock().unwrap() = true;
        dispatcher.release.1.notify_one();
        let result = termination.join().unwrap();
        assert!(closed, "termination did not close native admission");
        assert!(caller.pending_wait().is_none());
        assert!(dispatcher.lifetime.check().is_err());
        assert_eq!(dispatcher.drained.load(Ordering::Acquire), started);
        result.expect("normal cancellation was reported as a fatal service failure");
    }
}

struct PassiveEventService {
    horizon: Arc<HorizonSvcDispatcher>,
    lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
    event: nixe_runtime::ReadableEventObject,
    timeout: Option<Duration>,
}
impl ExceptionDispatcher for PassiveEventService {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        self.lifetime.begin().unwrap();
        let service = self.clone();
        self.horizon
            .start_owned_svc(context, self.lifetime.clone(), None, async move {
                crate::ipc_wire::operation::ServiceEvent::new(
                    service.horizon.resources.system.hid.kernel(),
                    service.event.clone(),
                    service.timeout,
                    &service.lifetime,
                )
                .await
                .map_err(|error| HorizonSvcFault::Ipc {
                    immediate: 0x21,
                    fault: Box::new(HorizonIpcFault::from_wire(error)),
                })?;
                Ok(SvcCompletion::success(None))
            })
    }
}

#[test]
fn terminating_a_passive_service_wait_does_not_require_a_device_signal_or_timeout() {
    for timeout in [None, Some(Duration::from_secs(60))] {
        let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
            &[0xd4000841],
            Default::default(),
            support::process_builder(),
        );
        let kernel = process.scheduler().clone();
        let core = VirtualCpuId::new(0);
        let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
        let id = runtime
            .register_process(
                process,
                ProcessRegistration {
                    priority: 44,
                    ideal_vcpu: Some(core),
                    affinity: kernel.profile().core_set([core]).unwrap(),
                },
            )
            .unwrap();
        let process = runtime.process(id).unwrap();
        let caller = process.main_thread();
        let (signal, event) = nixe_runtime::EventObject::create_pair(&kernel);
        let dispatcher = Arc::new(PassiveEventService {
            horizon: support::dispatcher(&process),
            lifetime: Arc::default(),
            event,
            timeout,
        });
        runtime.start(dispatcher.clone()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while caller.pending_wait().is_none() {
            assert!(Instant::now() < deadline, "service did not suspend");
            std::thread::yield_now();
        }
        let termination = std::thread::spawn(move || {
            runtime.remove_process(id).unwrap();
            runtime.shutdown().unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(3);
        while !termination.is_finished() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let completed_without_signal = termination.is_finished();
        // Permit bounded cleanup even if the regression returns.
        signal.signal();
        termination.join().unwrap();
        assert!(dispatcher.lifetime.check().is_err());
        assert!(caller.pending_wait().is_none());
        assert!(
            completed_without_signal,
            "process exit waited for an unrelated device event"
        );
        assert_eq!(kernel.deadline_count(), 0);
    }
}

struct FailedDevice(mpsc::Sender<nixe_memory::DeviceAccessRequest>);
impl nixe_memory::DeviceAccessEndpoint for FailedDevice {
    fn submit(&self, request: nixe_memory::DeviceAccessRequest) {
        self.0.send(request).unwrap();
    }
    fn cancel(&self, _: nixe_memory::AccessRequestId) {}
    fn allocation_released(&self, _: nixe_memory::MachineId, _: nixe_memory::AllocationId) {}
    fn physical_retired(
        &self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: Box<[nixe_memory::Extent]>,
    ) {
    }
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
}
struct DeviceReadBeforeCancel {
    horizon: Arc<HorizonSvcDispatcher>,
    address: GuestVirtualAddress,
    entered: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}
impl ExceptionDispatcher for DeviceReadBeforeCancel {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        if request.syndrome() == Some(0x42) {
            return self
                .horizon
                .read_svc_input(context, 0x42, self.address, 8, |_, _, _| resume());
        }
        self.entered.send(()).unwrap();
        let _ = self.release.lock().unwrap().recv();
        terminate(ExceptionTerminationScope::CurrentThread)
    }
}

struct DuplicateReadyWake {
    horizon: Arc<HorizonSvcDispatcher>,
    ready: Arc<AtomicBool>,
    wake: Arc<Mutex<Option<Waker>>>,
    queue: Mutex<Option<Arc<nixe_runtime::KernelQueue>>>,
    resumed: Arc<AtomicBool>,
}
impl ExceptionDispatcher for DuplicateReadyWake {
    type Fault = HorizonSvcFault;
    fn dispatch(
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        _: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        *self.queue.lock().unwrap() = Some(context.completion_queue().clone());
        let ready = self.ready.clone();
        let wake = self.wake.clone();
        let resumed = self.resumed.clone();
        self.horizon.start_owned_svc(
            context,
            Arc::default(),
            None,
            std::future::poll_fn(move |context| {
                if !ready.load(Ordering::Acquire) {
                    *wake.lock().unwrap() = Some(context.waker().clone());
                    return Poll::Pending;
                }
                context.waker().wake_by_ref();
                let resumed = resumed.clone();
                Poll::Ready(Ok(SvcCompletion::kernel(move |_, _| {
                    resumed.store(true, Ordering::Release);
                    resume()
                })))
            }),
        )
    }
}

#[test]
fn a_duplicate_ready_wake_cannot_prevent_later_cancellation_of_a_paused_call() {
    let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
        &[0xd4000841],
        Default::default(),
        support::process_builder(),
    );
    let kernel = process.scheduler().clone();
    let core = VirtualCpuId::new(0);
    let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
    let id = runtime
        .register_process(
            process,
            ProcessRegistration {
                priority: 44,
                ideal_vcpu: Some(core),
                affinity: kernel.profile().core_set([core]).unwrap(),
            },
        )
        .unwrap();
    let process = runtime.process(id).unwrap();
    let caller = process.main_thread();
    let dispatcher = Arc::new(DuplicateReadyWake {
        horizon: support::dispatcher(&process),
        ready: Arc::new(AtomicBool::new(false)),
        wake: Arc::default(),
        queue: Mutex::default(),
        resumed: Arc::new(AtomicBool::new(false)),
    });
    runtime.start(dispatcher.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while caller.pending_wait().is_none() {
        assert!(Instant::now() < deadline, "service did not suspend");
        std::thread::yield_now();
    }
    runtime.set_paused(true).unwrap();
    dispatcher.ready.store(true, Ordering::Release);
    dispatcher.wake.lock().unwrap().take().unwrap().wake();
    while kernel.thread(caller.id()).unwrap().lifecycle != nixe_scheduler::ThreadLifecycle::Ready {
        assert!(Instant::now() < deadline, "service did not become ready");
        std::thread::yield_now();
    }
    // Two further acknowledgements let the owner consume the self-wake after
    // completing the request, without running its paused continuation.
    runtime.set_paused(true).unwrap();
    runtime.set_paused(true).unwrap();
    let operation = caller.pending_wait().unwrap();
    let queue = dispatcher.queue.lock().unwrap().as_ref().unwrap().clone();
    let termination = std::thread::spawn(move || {
        runtime.remove_process(id).unwrap();
        runtime.shutdown().unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(3);
    while !termination.is_finished() && Instant::now() < deadline {
        std::thread::yield_now();
    }
    let completed = termination.is_finished();
    if !completed {
        // Bound cleanup when reproducing a lost notification.
        queue.enqueue(operation);
    }
    termination.join().unwrap();
    assert!(!dispatcher.resumed.load(Ordering::Acquire));
    assert!(caller.pending_wait().is_none());
    assert!(
        completed,
        "a consumed duplicate left cancellation marked as already queued"
    );
}

#[test]
#[ignore = "Explicit native IPC throughput diagnostic, outside normal correctness runs"]
fn profile_native_service_roundtrips() {
    struct Profile {
        horizon: Arc<HorizonSvcDispatcher>,
        started: Mutex<Option<Instant>>,
        measured: mpsc::Sender<Duration>,
    }
    impl ExceptionDispatcher for Profile {
        type Fault = HorizonSvcFault;
        fn dispatch(
            self: &Arc<Self>,
            context: &mut ExceptionDispatchContext<'_>,
            request: ExceptionDispatchRequest,
        ) -> ExceptionDispatchOutcome<Self::Fault> {
            if request.syndrome() == Some(0x10) {
                let mut started = self.started.lock().unwrap();
                if let Some(started) = started.take() {
                    self.measured.send(started.elapsed()).unwrap();
                } else {
                    *started = Some(Instant::now());
                }
            }
            self.horizon.dispatch(context, request)
        }
    }
    const CALLS: u64 = 20_000;
    let svc = |id: u32| 0xd4000001 | (id << 5);
    // Restore the request words overwritten by each reply, then issue one RPC.
    let body = [
        0xf9000295,
        0xf9000a96,
        0xf9000e97,
        0xaa1303e0,
        svc(0x21),
        0x71000718,
        0x54ffff41,
    ];
    let mut code = body.to_vec();
    code.extend([svc(0x10), 0x2a1903f8]); // mark start; mov w24,w25
    code.extend(body);
    code.extend([svc(0x10), svc(0x0a)]);
    for sample in 0..3 {
        let (_directory, process) = crate::svc_integration_tests::fixture_runnable_with_builder(
            &code,
            Default::default(),
            support::process_builder(),
        );
        let kernel = process.scheduler().clone();
        let core = VirtualCpuId::new(0);
        let mut runtime = Runtime::try_new(kernel.clone()).unwrap();
        let id = runtime
            .register_process(
                process,
                ProcessRegistration {
                    priority: 44,
                    ideal_vcpu: Some(core),
                    affinity: kernel.profile().core_set([core]).unwrap(),
                },
            )
            .unwrap();
        let process = runtime.process(id).unwrap();
        let main = process.main_thread();
        let handle = process
            .handles()
            .insert(crate::ServiceSession::new(
                crate::HorizonIpcObject::SemanticService(crate::IpcSession::new(
                    crate::IpcService::FileSystem,
                )),
            ))
            .unwrap();
        for (index, value) in [
            (19, u64::from(handle)),
            (20, main.tls_base.get()),
            (21, 0x0000000800000004),
            (22, 0x49434653),
            (23, 1005),
            (24, 1000),
            (25, CALLS),
        ] {
            write_register(&mut main.registers(), index, value);
        }
        let horizon = support::dispatcher(&process);
        let (measured, receive) = mpsc::channel();
        runtime
            .start(Arc::new(Profile {
                horizon: horizon.clone(),
                started: Mutex::default(),
                measured,
            }))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while process.lifecycle() == nixe_runtime::ProcessLifecycle::Running {
            assert!(Instant::now() < deadline, "native RPC diagnostic stalled");
            runtime.wait_events(Duration::from_millis(10)).unwrap();
        }
        let elapsed = receive.recv_timeout(Duration::from_secs(1)).unwrap();
        runtime.shutdown().unwrap();
        assert_eq!(horizon.svc_calls(), CALLS + 1003);
        eprintln!(
            "native_ipc_profile sample={sample} calls={CALLS} elapsed_ns={} ns_per_call={:.1}",
            elapsed.as_nanos(),
            elapsed.as_nanos() as f64 / CALLS as f64
        );
    }
}
