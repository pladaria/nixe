pub mod synthetic_packages;

use std::ops::{Deref, DerefMut};

use nixe_horizon::switch_1_scheduler_profile;
use nixe_runtime::{
    CoordinatorExecution, CoordinatorRouteError, ExceptionDispatcher, ExceptionHandlingResult,
    ExecutionReport, ExecutionStop, ProcessRegistration, RunnableProcess, RuntimeCoordinator,
};
use nixe_scheduler::{ProcessId, VirtualCpuId};

/// Test-only owner which drives a process exclusively through the production
/// scheduler/coordinator path while still allowing fixture setup between slices.
pub struct ScheduledProcess {
    coordinator: RuntimeCoordinator,
    process: ProcessId,
    last_execution: Option<CoordinatorExecution>,
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
        let mut coordinator = RuntimeCoordinator::new(switch_1_scheduler_profile());
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
        }
    }

    pub fn run_slice(&mut self, instruction_budget: u64) -> Result<ExecutionReport, String> {
        let execution = self
            .coordinator
            .run_next(instruction_budget)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "scheduler found no runnable test thread".to_owned())?;
        let report = execution.report.clone();
        self.last_execution = Some(execution);
        Ok(report)
    }

    pub fn route_supervisor_call<D: ExceptionDispatcher>(
        &mut self,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, CoordinatorRouteError> {
        let lease = self
            .last_execution
            .as_ref()
            .expect("a supervisor exit is routed after its scheduler slice")
            .lease;
        self.coordinator
            .route_supervisor_call(lease, stop, dispatcher)
    }

    pub fn resume(&mut self) -> bool {
        let thread = self.deref().main_thread_id();
        self.coordinator.make_thread_ready(thread).is_ok()
    }

    pub fn terminate(&mut self) -> bool {
        self.coordinator
            .terminate_process(self.process)
            .unwrap_or(false)
    }

    pub fn teardown(mut self) -> nixe_runtime::ProcessTeardownReport {
        self.coordinator
            .remove_process(self.process)
            .expect("test process is not in flight")
            .try_teardown()
            .expect("test engine shuts down")
    }

    pub const fn scheduler_process_id(&self) -> ProcessId {
        self.process
    }

    pub fn coordinator_mut(&mut self) -> &mut RuntimeCoordinator {
        &mut self.coordinator
    }
}

impl Deref for ScheduledProcess {
    type Target = RunnableProcess;

    fn deref(&self) -> &Self::Target {
        self.coordinator
            .process(self.process)
            .expect("scheduled test process remains registered")
    }
}

impl DerefMut for ScheduledProcess {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.coordinator
            .process_mut(self.process)
            .expect("scheduled test process remains registered")
    }
}
