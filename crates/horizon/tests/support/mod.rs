pub mod synthetic_packages;

use std::ops::Deref;

use nixe_horizon::switch_1_scheduler_profile;
use nixe_runtime::{
    CoordinatorExecution, ExceptionHandlingResult, ExecutionReport, ExecutionStop,
    ProcessRegistration, RunnableProcess, RuntimeCoordinator,
};
use nixe_scheduler::{ProcessId, VirtualCpuId};

/// Test-only owner which drives a process exclusively through the production
/// scheduler/coordinator path while still allowing fixture setup between slices.
pub struct ScheduledProcess {
    coordinator: RuntimeCoordinator,
    process: ProcessId,
    last_execution: Option<CoordinatorExecution>,
    main_thread_handle: u32,
}

impl ScheduledProcess {
    pub fn thread_lifecycle(
        &self,
        thread: nixe_scheduler::GuestThreadId,
    ) -> nixe_scheduler::ThreadLifecycle {
        self.coordinator
            .scheduler()
            .thread(thread)
            .expect("scheduled process owns the requested thread")
            .lifecycle
    }

    pub fn main_thread_lifecycle(&self) -> nixe_scheduler::ThreadLifecycle {
        self.thread_lifecycle(self.main_thread_id())
    }

    pub fn new(process: RunnableProcess) -> Self {
        let main_thread_handle = initial_thread_handle(&process);
        let mut coordinator = RuntimeCoordinator::new(process.scheduler().clone());
        let registration = ProcessRegistration {
            priority: 44,
            ideal_vcpu: Some(VirtualCpuId::new(0)),
            affinity: coordinator.scheduler().profile().all_cores(),
        };
        let process = coordinator
            .register_process(process, registration)
            .expect("test process registers with the production scheduler");
        Self {
            coordinator,
            process,
            last_execution: None,
            main_thread_handle,
        }
    }

    pub fn main_thread_handle(&self) -> u32 {
        self.main_thread_handle
    }

    pub fn run_slice(&mut self) -> Result<ExecutionReport, String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let execution = loop {
            if let Some(execution) = self
                .coordinator
                .run_next(deadline.saturating_duration_since(std::time::Instant::now()))
                .map_err(|error| error.to_string())?
            {
                break execution;
            }
            if std::time::Instant::now() >= deadline {
                return Err("scheduler found no runnable test thread".to_owned());
            }
        };
        let report = execution.report.clone();
        self.last_execution = Some(execution);
        Ok(report)
    }

    pub fn route_supervisor_call(
        &mut self,
        stop: &ExecutionStop,
        dispatcher: &mut nixe_horizon::HorizonSvcDispatcher,
    ) -> Result<
        ExceptionHandlingResult<nixe_horizon::HorizonSvcFault>,
        nixe_horizon::HorizonScheduledDispatchError,
    > {
        let lease = self
            .last_execution
            .as_ref()
            .expect("a supervisor exit is routed after its scheduler slice")
            .lease;
        dispatcher.route_scheduled_supervisor_call(&mut self.coordinator, lease, stop)
    }

    pub fn terminate(&mut self) -> bool {
        self.coordinator
            .terminate_process(self.process)
            .unwrap_or(false)
    }

    pub fn teardown(mut self) -> nixe_runtime::ProcessTeardownReport {
        self.coordinator
            .remove_process(self.process)
            .expect("test process shuts down after native retirement")
    }

    pub const fn scheduler_process_id(&self) -> ProcessId {
        self.process
    }

    pub fn coordinator_mut(&mut self) -> &mut RuntimeCoordinator {
        &mut self.coordinator
    }
}

impl Deref for ScheduledProcess {
    type Target = std::sync::Arc<RunnableProcess>;

    fn deref(&self) -> &Self::Target {
        self.coordinator
            .process(self.process)
            .expect("scheduled test process remains registered")
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

pub fn dispatcher(process: &RunnableProcess) -> nixe_horizon::HorizonSvcDispatcher {
    dispatcher_with_mode(process, nixe_horizon::OperationMode::default())
}

pub fn dispatcher_with_mode(
    process: &RunnableProcess,
    mode: nixe_horizon::OperationMode,
) -> nixe_horizon::HorizonSvcDispatcher {
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
