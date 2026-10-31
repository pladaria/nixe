pub mod synthetic_packages;

use nixe_horizon::{HorizonSvcDispatcher, HorizonSvcFault, switch_1_scheduler_profile};
use nixe_runtime::{
    ExceptionDispatcher, ExceptionHandlingResult, ExecutionReport, ExecutionStop,
    ProcessRegistration, RunnableProcess, Runtime,
};
use nixe_scheduler::{Completion, GuestThreadId, Lease, ProcessId, VirtualCpuId};
use std::{collections::BTreeMap, ops::Deref, sync::Arc, time::Duration};

thread_local! {
    static RECOVERY: std::cell::RefCell<nixe_cpu_direct_memory::NativeWorker> = std::cell::RefCell::default();
}

/// Unit fixture for the public kernel/exception contracts. Native runtime
/// scheduling and recovery are covered separately through Runtime::start.
pub struct SvcTestDriver {
    runtime: Runtime,
    registered: std::collections::BTreeSet<ProcessId>,
    native: BTreeMap<(ProcessId, VirtualCpuId), (Arc<RunnableProcess>, nixe_cpu_jit::JitThread)>,
    completions: Arc<nixe_runtime::KernelQueue>,
    completed: BTreeMap<GuestThreadId, ExceptionHandlingResult<HorizonSvcFault>>,
}
pub struct ObservedCall {
    pub lease: Lease,
    pub report: ExecutionReport,
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TestResourceCounts {
    pub processes: usize,
    pub scheduled_threads: usize,
    pub active_waits: usize,
    pub deadlines: usize,
    pub priority_donations: usize,
    pub address_waiters: usize,
}
impl SvcTestDriver {
    pub fn new(kernel: Arc<nixe_scheduler::SchedulerState>) -> Self {
        Self {
            runtime: Runtime::try_new(kernel).unwrap(),
            registered: Default::default(),
            native: BTreeMap::new(),
            completions: nixe_runtime::KernelQueue::new(std::task::Waker::noop().clone()),
            completed: BTreeMap::new(),
        }
    }
    pub fn scheduler(&self) -> &Arc<nixe_scheduler::SchedulerState> {
        self.runtime.scheduler()
    }
    pub fn register_process(
        &mut self,
        process: RunnableProcess,
        registration: ProcessRegistration,
    ) -> Result<ProcessId, String> {
        let id = self
            .runtime
            .register_process(process, registration)
            .map_err(|error| error.to_string())?;
        self.registered.insert(id);
        Ok(id)
    }
    pub fn advance_virtual_time(&mut self, nanoseconds: u64) -> Result<usize, String> {
        Ok(self.scheduler().advance_time(
            self.scheduler()
                .clock()
                .scheduler_time_ns()
                .saturating_add(nanoseconds),
        ))
    }
    pub fn resource_counts(&self) -> TestResourceCounts {
        TestResourceCounts {
            processes: self.registered.len(),
            scheduled_threads: self.scheduler().thread_count(),
            active_waits: self.scheduler().active_wait_count(),
            deadlines: self.scheduler().deadline_count(),
            priority_donations: self.scheduler().priority_donation_count(),
            address_waiters: self
                .registered
                .iter()
                .filter_map(|id| self.process(*id))
                .map(|process| process.address_waits().lock().waiter_count())
                .sum(),
        }
    }
    pub fn process(&self, id: ProcessId) -> Option<Arc<RunnableProcess>> {
        self.runtime.process(id)
    }
    pub fn event_sender(&self) -> nixe_runtime::ExternalEventSender {
        self.runtime.event_sender()
    }
    pub fn drain_external_events(&mut self) -> Result<(), String> {
        self.runtime
            .drain_external_events()
            .map_err(|error| error.to_string())
    }
    pub fn run_native(&mut self, _timeout: Duration) -> Result<Option<ObservedCall>, String> {
        let core = self
            .scheduler()
            .profile()
            .vcpus()
            .iter()
            .filter_map(|core| {
                let thread = self.scheduler().selection(core.id()).unwrap().thread?;
                let view = self.scheduler().thread(thread)?;
                (view.lifecycle == nixe_scheduler::ThreadLifecycle::Ready)
                    .then_some((view.effective_priority, core.id()))
            })
            .min();
        let lease = core.and_then(|(_, core)| self.scheduler().claim(core).unwrap());
        let Some(lease) = lease else {
            return Ok(None);
        };
        let process = self.process(lease.process).unwrap();
        let thread = process.thread(lease.thread).unwrap();
        let mut state = thread.take_state().unwrap();
        let kernel = self.scheduler().clone();
        let clock = kernel.clock().clone();
        let (_, cpu) = self
            .native
            .entry((lease.process, lease.vcpu))
            .or_insert_with(|| (process.clone(), *process.create_core(lease.vcpu).unwrap()));
        struct Timer(nixe_runtime::VirtualClock);
        impl nixe_cpu::execution::ArchitecturalTimer for Timer {
            fn snapshot(&self) -> nixe_cpu::execution::TimerSnapshot {
                nixe_cpu::execution::TimerSnapshot {
                    counter: (self.0.elapsed().as_nanos() * 19_200_000 / 1_000_000_000) as u64,
                    frequency: 19_200_000,
                }
            }
        }
        let timer = Timer(clock);
        let report = RECOVERY.with_borrow_mut(|recovery| {
            cpu.run_slice(
                recovery,
                nixe_cpu_jit::JitRunRequest {
                    state: &mut state,
                    timer: &timer,
                },
            )
        });
        if report
            .as_ref()
            .is_ok_and(|report| matches!(report.stop, ExecutionStop::Safepoint))
            || kernel.selection(lease.vcpu).unwrap().thread != Some(lease.thread)
        {
            process
                .handle_interrupt(&thread, lease.vcpu, cpu)
                .map_err(|error| error.to_string())?;
        }
        if state.is_available() {
            state.leave_core();
            thread.restore_state(state);
        }
        if report
            .as_ref()
            .is_ok_and(|report| matches!(report.stop, ExecutionStop::SupervisorCall { .. }))
        {
            // The actual native owner remains runnable until the operation blocks.
        } else {
            kernel.complete(lease, Completion::Ready).unwrap();
        }
        if report.as_ref().is_ok_and(|report| {
            matches!(
                report.stop,
                ExecutionStop::FetchFault { .. }
                    | ExecutionStop::UnsupportedSemantics { .. }
                    | ExecutionStop::UnallocatedEncoding { .. }
            )
        }) {
            process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
            self.runtime
                .terminate_process(lease.process)
                .map_err(|error| error.to_string())?;
        }
        report
            .map(|report| Some(ObservedCall { lease, report }))
            .map_err(|error| error.to_string())
    }
    pub fn call_local(
        &mut self,
        lease: Lease,
        stop: &ExecutionStop,
        dispatcher: &Arc<HorizonSvcDispatcher>,
    ) -> Result<ExceptionHandlingResult<HorizonSvcFault>, String> {
        let request = stop
            .exception_dispatch_request()
            .ok_or("not an exception")?;
        let process = self.process(lease.process).unwrap();
        let thread = process.thread(lease.thread).unwrap();
        let mut state = thread.take_state().unwrap();
        let caller = nixe_runtime::ExceptionThreadContext::new(
            thread.id(),
            lease.vcpu,
            &thread,
            &mut state,
            thread.tls_base,
        );
        let mut context = nixe_runtime::ExceptionDispatchContext::new(
            request.source(),
            &process,
            caller,
            &self.completions,
            process.generation(),
        );
        let outcome = dispatcher.dispatch(&mut context, request);
        let result =
            process.apply_supervisor_call_outcome(&thread, request.source(), &mut state, outcome);
        thread.restore_state(state);
        self.finish(&process, &thread, Some(lease), &result)?;
        Ok(result)
    }
    fn finish<F>(
        &mut self,
        process: &RunnableProcess,
        thread: &Arc<nixe_runtime::ThreadObject>,
        lease: Option<Lease>,
        result: &ExceptionHandlingResult<F>,
    ) -> Result<(), String> {
        match result {
            ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_) => {
                if let Some(lease) = lease {
                    self.scheduler()
                        .resume_kernel(lease)
                        .map_err(|error| error.to_string())?;
                    self.scheduler()
                        .complete(lease, Completion::Ready)
                        .map_err(|error| error.to_string())?;
                    Ok(())
                } else {
                    self.scheduler()
                        .make_ready(thread.id())
                        .map_err(|error| error.to_string())
                }
            }
            ExceptionHandlingResult::Suspended => {
                if let Some(lease) = lease {
                    self.scheduler()
                        .complete(lease, Completion::Waiting)
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            }
            _ => {
                if let Some(lease) = lease {
                    self.scheduler()
                        .complete(lease, Completion::Ready)
                        .map_err(|error| error.to_string())?;
                }
                self.scheduler().cancel_thread_property(thread.id());
                process.address_waits().lock().release_thread(thread.id());
                while let Some((owner, waiter)) = self.scheduler().take_held_waiter(thread.id()) {
                    if let Some(owner) = self.process(owner) {
                        owner.address_waits().lock().cancel_held_waiter(waiter);
                    }
                }
                self.scheduler()
                    .terminate(
                        thread.id(),
                        matches!(result, ExceptionHandlingResult::Fault(_)),
                    )
                    .map_err(|error| error.to_string())?;
                thread.signal();
                if process.lifecycle() == nixe_scheduler::ProcessLifecycle::Terminating {
                    self.runtime
                        .terminate_process(ProcessId::new(process.process_id()))
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            }
        }
    }
    pub fn drive_pending(&mut self) -> Result<(), nixe_runtime::DispatchFault> {
        for _ in 0..self.completions.begin_batch() {
            let operation = self.completions.pop().unwrap();
            let thread = operation.caller().clone();
            let process = thread.owner().unwrap().clone();
            let source = operation.source();
            let token = operation.token();
            let completed = match operation.clone().poll() {
                std::task::Poll::Pending => continue,
                std::task::Poll::Ready(result) => result,
            };
            if !thread
                .pending_wait()
                .is_some_and(|pending| Arc::ptr_eq(&pending, &operation))
                || process.lifecycle() != nixe_scheduler::ProcessLifecycle::Running
            {
                let current = thread
                    .pending_wait()
                    .is_some_and(|pending| Arc::ptr_eq(&pending, &operation));
                if current {
                    thread.clear_wait(&operation);
                }
                if let Err(error) = completed {
                    return Err(error);
                }
                continue;
            }
            thread.clear_wait(&operation);
            if matches!(completed, Ok(nixe_runtime::WaitCompletion::Cancelled)) {
                continue;
            }
            let token = token.unwrap();
            if let Ok(nixe_runtime::WaitCompletion::Return { x0, x1 }) = completed {
                use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
                {
                    let mut state = thread.registers();
                    state.write_x(
                        A64Register::General(A64GeneralRegister::new(0).unwrap()),
                        x0,
                    );
                    if let Some(value) = x1 {
                        state.write_x(
                            A64Register::General(A64GeneralRegister::new(1).unwrap()),
                            value,
                        );
                    }
                }
                self.scheduler().finish_wait(token).unwrap();
                self.completed
                    .insert(thread.id(), ExceptionHandlingResult::Resumed);
                continue;
            }
            self.scheduler().continue_wait(token).unwrap();
            let mut state = thread.take_state().unwrap();
            let core = self
                .scheduler()
                .thread(thread.id())
                .unwrap()
                .ideal_vcpu
                .unwrap_or(VirtualCpuId::new(0));
            let caller = nixe_runtime::ExceptionThreadContext::new(
                thread.id(),
                core,
                &thread,
                &mut state,
                thread.tls_base,
            );
            let mut context = nixe_runtime::ExceptionDispatchContext::new(
                source,
                &process,
                caller,
                &self.completions,
                process.generation(),
            );
            let outcome = match completed {
                Ok(nixe_runtime::WaitCompletion::Resume) => operation.resume(&mut context),
                Err(fault) => nixe_runtime::ExceptionDispatchOutcome::Fault(fault),
                _ => unreachable!("direct return completed above"),
            };
            let result =
                process.apply_supervisor_call_outcome(&thread, source, &mut state, outcome);
            thread.restore_state(state);
            self.finish(&process, &thread, None, &result).unwrap();
            let observed = match &result {
                ExceptionHandlingResult::Resumed => ExceptionHandlingResult::Resumed,
                ExceptionHandlingResult::Suspended => ExceptionHandlingResult::Suspended,
                ExceptionHandlingResult::Rejected(error) => ExceptionHandlingResult::Rejected(
                    error
                        .downcast_ref::<HorizonSvcFault>()
                        .expect("Horizon result")
                        .clone(),
                ),
                ExceptionHandlingResult::Fault(error) => {
                    ExceptionHandlingResult::Fault(match error.downcast_ref::<HorizonSvcFault>() {
                        Some(fault) => fault.clone(),
                        None => {
                            return Err(match result {
                                ExceptionHandlingResult::Fault(error) => error,
                                _ => unreachable!(),
                            });
                        }
                    })
                }
                ExceptionHandlingResult::Terminated {
                    scope,
                    exit_code,
                    reason,
                } => ExceptionHandlingResult::Terminated {
                    scope: *scope,
                    exit_code: *exit_code,
                    reason: *reason,
                },
            };
            self.completed.insert(thread.id(), observed);
            if let ExceptionHandlingResult::Fault(error) = result {
                return Err(error);
            }
        }
        Ok(())
    }
    pub fn await_service(
        &mut self,
        thread: GuestThreadId,
    ) -> ExceptionHandlingResult<HorizonSvcFault> {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let result = self.drive_pending();
            if let Some(completed) = self.completed.remove(&thread) {
                return completed;
            }
            result.unwrap();
            assert!(
                std::time::Instant::now() < deadline,
                "service did not complete"
            );
            std::thread::yield_now();
        }
    }
    pub fn terminate_process(&mut self, id: ProcessId) -> Result<bool, String> {
        self.runtime
            .terminate_process(id)
            .map_err(|error| error.to_string())
    }
    pub fn remove_process(
        &mut self,
        id: ProcessId,
    ) -> Result<nixe_runtime::ProcessTeardownReport, String> {
        self.terminate_process(id)?;
        self.retire_native(id);
        self.drive_pending().map_err(|error| error.to_string())?;
        self.registered.remove(&id);
        self.runtime
            .remove_process(id)
            .map_err(|error| error.to_string())
    }
    fn retire_native(&mut self, process: ProcessId) {
        let keys: Vec<_> = self
            .native
            .keys()
            .copied()
            .filter(|(id, _)| *id == process)
            .collect();
        for key in keys {
            let (process, cpu) = self.native.remove(&key).unwrap();
            drop(cpu);
            process.retire_core(key.1);
        }
    }
    pub fn shutdown(&mut self) -> Result<(), String> {
        let leases: Vec<_> = self.scheduler().active_leases().collect();
        for lease in leases {
            self.scheduler()
                .complete(lease, Completion::Ready)
                .map_err(|error| error.to_string())?;
        }
        let ids: Vec<_> = self.registered.iter().copied().collect();
        for id in ids {
            self.remove_process(id)?;
        }
        self.runtime.shutdown().map_err(|error| error.to_string())
    }
}
impl Drop for SvcTestDriver {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

pub trait DispatchFixture {
    fn drive_pending(&self, driver: &mut SvcTestDriver) -> Result<(), nixe_runtime::DispatchFault>;
    fn call_local(
        &self,
        driver: &mut SvcTestDriver,
        lease: Lease,
        stop: &ExecutionStop,
    ) -> Result<ExceptionHandlingResult<HorizonSvcFault>, String>;
}
impl DispatchFixture for Arc<HorizonSvcDispatcher> {
    fn drive_pending(&self, driver: &mut SvcTestDriver) -> Result<(), nixe_runtime::DispatchFault> {
        driver.drive_pending()
    }
    fn call_local(
        &self,
        driver: &mut SvcTestDriver,
        lease: Lease,
        stop: &ExecutionStop,
    ) -> Result<ExceptionHandlingResult<HorizonSvcFault>, String> {
        driver.call_local(lease, stop, self)
    }
}

pub struct ScheduledProcess {
    driver: SvcTestDriver,
    process: Arc<RunnableProcess>,
    process_id: ProcessId,
    last_execution: Option<ObservedCall>,
    main_thread_handle: u32,
}
impl ScheduledProcess {
    pub fn new(process: RunnableProcess) -> Self {
        let main_thread_handle = initial_thread_handle(&process);
        let mut driver = SvcTestDriver::new(process.scheduler().clone());
        let id = driver
            .register_process(
                process,
                ProcessRegistration {
                    priority: 44,
                    ideal_vcpu: Some(VirtualCpuId::new(0)),
                    affinity: driver.scheduler().profile().all_cores(),
                },
            )
            .unwrap();
        Self {
            process: driver.process(id).unwrap(),
            driver,
            process_id: id,
            last_execution: None,
            main_thread_handle,
        }
    }
    pub fn thread_lifecycle(&self, thread: GuestThreadId) -> nixe_scheduler::ThreadLifecycle {
        self.scheduler().thread(thread).unwrap().lifecycle
    }
    pub fn main_thread_lifecycle(&self) -> nixe_scheduler::ThreadLifecycle {
        self.thread_lifecycle(self.main_thread_id())
    }
    pub fn main_thread_handle(&self) -> u32 {
        self.main_thread_handle
    }
    pub fn run_slice(&mut self) -> Result<ExecutionReport, String> {
        let execution = self
            .driver
            .run_native(Duration::from_secs(2))?
            .ok_or("no runnable fixture thread")?;
        let report = execution.report.clone();
        self.last_execution = Some(execution);
        Ok(report)
    }
    pub fn call_local(
        &mut self,
        stop: &ExecutionStop,
        dispatcher: &Arc<HorizonSvcDispatcher>,
    ) -> Result<ExceptionHandlingResult<HorizonSvcFault>, String> {
        let lease = self.last_execution.as_ref().unwrap().lease;
        self.driver.call_local(lease, stop, dispatcher)
    }
    pub fn terminate(&mut self) -> bool {
        self.driver
            .terminate_process(self.process_id)
            .unwrap_or(false)
    }
    pub fn teardown(mut self) -> nixe_runtime::ProcessTeardownReport {
        self.driver.remove_process(self.process_id).unwrap()
    }
    pub fn scheduler_process_id(&self) -> ProcessId {
        self.process_id
    }
    pub fn driver_mut(&mut self) -> &mut SvcTestDriver {
        &mut self.driver
    }
}
impl Deref for ScheduledProcess {
    type Target = Arc<RunnableProcess>;
    fn deref(&self) -> &Self::Target {
        &self.process
    }
}

pub fn initial_thread_handle(process: &RunnableProcess) -> u32 {
    let registers = process.main_thread().snapshot().unwrap().register_context();
    if process.mounts().homebrew_executable().is_some() {
        process
            .memory()
            .space()
            .read_value(nixe_memory::GuestVirtualAddress::new(registers.x[0] + 8), 8)
            .unwrap()
            .into_ready()
            .unwrap() as u32
    } else {
        registers.x[1] as u32
    }
}

pub fn process_builder() -> nixe_runtime::ProcessBuilder {
    process_builder_with_kernel(nixe_scheduler::SchedulerState::new(
        switch_1_scheduler_profile(),
        nixe_runtime::VirtualClock::new(nixe_runtime::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ))
}
pub fn process_builder_with_kernel(
    kernel: std::sync::Arc<nixe_scheduler::SchedulerState>,
) -> nixe_runtime::ProcessBuilder {
    let machine = nixe_memory::Machine::new(4 * 1024 * 1024 * 1024).unwrap();
    let system = machine.account(512 * 1024 * 1024).unwrap();
    nixe_runtime::ProcessBuilder::new(machine, system, kernel)
}

pub fn dispatcher(process: &RunnableProcess) -> Arc<nixe_horizon::HorizonSvcDispatcher> {
    dispatcher_with_mode(process, nixe_horizon::OperationMode::default())
}

pub fn dispatcher_with_mode(
    process: &RunnableProcess,
    mode: nixe_horizon::OperationMode,
) -> Arc<nixe_horizon::HorizonSvcDispatcher> {
    let account = process.memory().system_account();
    nixe_horizon::HorizonSvcDispatcher::new(
        process.memory().space().machine().clone(),
        mode,
        nixe_horizon::TimeEnvironment::new(account, process.scheduler().clock().clone(), "UTC")
            .unwrap(),
        nixe_horizon::HidSystem::new(account, process.scheduler()).unwrap(),
        nixe_horizon::SharedFontSession::new(account, None).unwrap(),
    )
}

pub fn ready<T>(future: impl std::future::Future<Output = T>) -> T {
    match std::pin::pin!(future).poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => {
            panic!("fixture setup unexpectedly requires asynchronous progress")
        }
    }
}

pub(crate) fn poll_session(
    session: &nixe_runtime::SessionObject,
    owner: nixe_runtime::SessionRequestOwner,
) -> Result<Option<nixe_runtime::SessionMessage>, nixe_runtime::SessionError> {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };
    match Pin::new(&mut session.response(owner)).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Pending => Ok(None),
        Poll::Ready(result) => result.map(Some),
    }
}

pub fn block_thread(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    thread: nixe_scheduler::GuestThreadId,
) {
    use nixe_scheduler::{Completion, ThreadLifecycle};
    let view = kernel.thread(thread).unwrap();
    if view.lifecycle == ThreadLifecycle::Waiting {
        return;
    }
    if view.lifecycle == ThreadLifecycle::Created {
        kernel.start_thread(thread).unwrap();
    }
    kernel.set_priority(thread, 0, None).unwrap();
    let core = view
        .ideal_vcpu
        .unwrap_or_else(|| view.affinity.iter().next().unwrap());
    let lease = kernel.claim(core).unwrap().unwrap();
    assert_eq!(lease.thread, thread);
    kernel.complete(lease, Completion::Waiting).unwrap();
    kernel
        .set_priority(thread, view.base_priority, None)
        .unwrap();
}
