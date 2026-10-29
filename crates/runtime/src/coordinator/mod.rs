use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use nixe_cpu::execution::VcpuEventState;
use nixe_scheduler::{
    Completion, CoreSet, GuestThreadId, Lease, ProcessId, ScheduledThreadConfig, SchedulerError,
    SchedulerState, VirtualCpuId, WakeToken,
};

use crate::{
    ExceptionDispatcher, ExceptionHandlingResult, ExceptionRouteError, ExecutionReport,
    ExecutionStop, ExternalEvent, ExternalEventInbox, ExternalEventSendError, ExternalEventSender,
    ProcessExecutionError, RunnableProcess,
};

mod execution;
mod identity;
mod thread;
mod wait;
pub(crate) mod worker;

use identity::ProcessIdAllocator;
pub use worker::WorkerFailure;
use worker::{VcpuWorkerPool, WorkerCpuThreadKey, WorkerRequest, WorkerRunFailure};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessRegistration {
    pub priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoordinatorResourceCounts {
    pub processes: usize,
    pub scheduled_threads: usize,
    pub active_waits: usize,
    pub deadlines: usize,
    pub priority_donations: usize,
    pub address_waiters: usize,
}

pub struct RuntimeCoordinator {
    scheduler: std::sync::Arc<SchedulerState>,
    processes: BTreeMap<ProcessId, std::sync::Arc<RunnableProcess>>,
    process_ids: ProcessIdAllocator,
    workers: VcpuWorkerPool,
    inbox: ExternalEventInbox,
    host_stop_requested: bool,
    virtual_clock: crate::VirtualClock,
    vcpu_events: BTreeMap<VirtualCpuId, VcpuEventState>,
    completed_executions: VecDeque<CoordinatorExecution>,
    deferred_events: VecDeque<crate::SequencedExternalEvent>,
}

impl RuntimeCoordinator {
    #[must_use]
    pub fn new(scheduler: std::sync::Arc<SchedulerState>) -> Self {
        Self::try_new(scheduler).expect("kernel worker construction failed")
    }

    pub fn try_new(scheduler: std::sync::Arc<SchedulerState>) -> Result<Self, CoordinatorError> {
        let profile = scheduler.profile();
        let virtual_clock = scheduler.clock().clone();
        let vcpu_events = profile
            .vcpus()
            .iter()
            .map(|descriptor| (descriptor.id(), VcpuEventState::default()))
            .collect();
        let inbox = ExternalEventInbox::bounded(1_024, profile.vcpus().len())
            .expect("the default event capacity is non-zero");
        scheduler
            .start_timers(std::task::Waker::from(std::sync::Arc::new(inbox.sender())))
            .map_err(|error| CoordinatorError::TimerStartup(error.kind()))?;
        let workers = match VcpuWorkerPool::start(
            scheduler
                .profile()
                .vcpus()
                .iter()
                .map(|descriptor| descriptor.id()),
            inbox.worker_sender(),
            scheduler.clone(),
        ) {
            Ok(workers) => workers,
            Err(error) => {
                let _ = scheduler.stop_timers();
                return Err(CoordinatorError::WorkerStartup(error.kind()));
            }
        };
        Ok(Self {
            scheduler,
            processes: BTreeMap::new(),
            process_ids: ProcessIdAllocator::new(),
            workers,
            inbox,
            host_stop_requested: false,
            virtual_clock,
            vcpu_events,
            completed_executions: VecDeque::new(),
            deferred_events: VecDeque::new(),
        })
    }

    /// Quiesces workers and releases every process, CPU backend, wait, and
    /// canonical address-space resource. Repeated shutdown calls are harmless.
    pub fn shutdown(&mut self) -> Result<(), CoordinatorError> {
        self.finalize_retired_threads()?;
        let processes: Vec<_> = self.processes.keys().copied().collect();
        for &process in &processes {
            self.request_process_stop(process)?;
        }
        self.quiesce()?;
        for process_id in processes {
            self.remove_process(process_id)?;
        }
        self.finalize_retired_threads()?;
        self.host_stop_requested = true;
        let workers = self.workers.shutdown().map_err(CoordinatorError::Worker);
        let timers = self
            .scheduler
            .stop_timers()
            .map_err(|_| CoordinatorError::TimerWorkerFailed);
        workers.and(timers)
    }

    #[must_use]
    pub fn scheduler(&self) -> &std::sync::Arc<SchedulerState> {
        &self.scheduler
    }

    #[must_use]
    pub fn process(&self, id: ProcessId) -> Option<&std::sync::Arc<RunnableProcess>> {
        self.processes.get(&id)
    }

    pub fn register_process(
        &mut self,
        mut process: RunnableProcess,
        registration: ProcessRegistration,
    ) -> Result<ProcessId, CoordinatorError> {
        if !std::sync::Arc::ptr_eq(process.scheduler(), &self.scheduler) {
            return Err(CoordinatorError::ForeignKernel);
        }
        let (id, next_process_id) = self
            .process_ids
            .candidate()
            .ok_or(CoordinatorError::ProcessIdExhausted)?;
        process.assign_process_id(id);
        let thread = self.scheduler.create_thread(ScheduledThreadConfig {
            process: id,
            base_priority: registration.priority,
            ideal_vcpu: registration.ideal_vcpu,
            affinity: registration.affinity,
        })?;
        if let Err(error) = process.assign_main_thread_id(thread) {
            self.scheduler.unregister(thread)?;
            return Err(error.into());
        }
        if let Err(error) = self.scheduler.make_ready(thread) {
            self.scheduler
                .unregister(thread)
                .expect("registration rollback removes an unleased created thread");
            return Err(error.into());
        }
        if let Err(error) = self.install_process_cpu_threads(id, &process, thread) {
            self.scheduler
                .unregister(thread)
                .expect("CPU thread installation rollback removes an unleased ready thread");
            return Err(error);
        }
        process.address_waits().bind(id, self.scheduler.clone());
        process
            .events
            .set(self.inbox.sender())
            .expect("process registers once");
        let process = std::sync::Arc::new(process);
        process.main_thread().bind_owner(&process);
        process
            .main_thread()
            .bind_retirement(id, self.inbox.sender());
        let replaced = self.processes.insert(id, process);
        debug_assert!(replaced.is_none());
        self.process_ids.commit(next_process_id);
        Ok(id)
    }

    /// Retires native contexts and closes a process. Exported kernel objects
    /// retain their identities until their last handle or pending operation ends.
    pub fn remove_process(
        &mut self,
        id: ProcessId,
    ) -> Result<crate::ProcessTeardownReport, CoordinatorError> {
        self.finalize_retired_threads()?;
        self.terminate_process(id)?;
        self.completed_executions
            .retain(|execution| execution.lease.process != id);
        let threads: Vec<_> = self
            .processes
            .get(&id)
            .ok_or(CoordinatorError::UnknownProcess(id))?
            .threads()
            .ids();
        for thread in &threads {
            let view = self
                .scheduler
                .thread(*thread)
                .ok_or(SchedulerError::UnknownThread(*thread))?;
            if view.lifecycle == nixe_scheduler::ThreadLifecycle::Running {
                return Err(SchedulerError::ThreadLeased(*thread).into());
            }
        }
        self.retire_process_cpu_threads(id)?;
        for thread in threads {
            self.finalize_thread_waits(id, thread);
        }
        let process = self
            .processes
            .remove(&id)
            .expect("process existed throughout atomic scheduler removal");
        let report = process
            .try_teardown()
            .map_err(|failure| CoordinatorError::ProcessTeardown {
                process: id,
                failure,
            });
        self.finalize_retired_threads()?;
        report
    }

    /// Terminates every thread in a registered process through the scheduler
    /// and closes owned handles. Native/memory retirement remains separate.
    pub fn terminate_process(&mut self, id: ProcessId) -> Result<bool, CoordinatorError> {
        let terminated = self.request_process_stop(id)?;
        self.quiesce_process(id)?;
        self.finish_process_termination(id)?;
        Ok(terminated)
    }

    fn request_process_stop(&mut self, id: ProcessId) -> Result<bool, CoordinatorError> {
        let process = self
            .processes
            .get(&id)
            .ok_or(CoordinatorError::UnknownProcess(id))?;
        let terminated = process.terminate_from_host();
        process
            .request_execution_stop()
            .map_err(|fault| CoordinatorError::Execution {
                process: id,
                thread: process.main_thread_id(),
                error: ProcessExecutionError::Cpu { fault },
            })?;
        Ok(terminated)
    }

    fn install_process_cpu_threads(
        &mut self,
        process_id: ProcessId,
        process: &RunnableProcess,
        thread: GuestThreadId,
    ) -> Result<(), CoordinatorError> {
        let key = WorkerCpuThreadKey {
            process: process_id,
            cpu_process: process.cpu_process_id(),
        };
        let vcpus: Vec<_> = self
            .scheduler
            .profile()
            .vcpus()
            .iter()
            .map(|descriptor| descriptor.id())
            .collect();
        let mut installed = Vec::new();
        for vcpu in vcpus {
            let cpu_thread = match process.create_worker_cpu_thread(vcpu) {
                Ok(cpu_thread) => cpu_thread,
                Err(fault) => {
                    self.rollback_process_cpu_threads(process_id, process, &installed);
                    return Err(CoordinatorError::Execution {
                        process: process_id,
                        thread,
                        error: ProcessExecutionError::Cpu { fault },
                    });
                }
            };
            if let Err(failure) = self.workers.install_cpu_thread(vcpu, key, cpu_thread) {
                self.rollback_process_cpu_threads(process_id, process, &installed);
                return Err(CoordinatorError::Worker(failure));
            }
            installed.push(vcpu);
        }
        Ok(())
    }

    fn rollback_process_cpu_threads(
        &mut self,
        process_id: ProcessId,
        process: &RunnableProcess,
        installed: &[VirtualCpuId],
    ) {
        let _ = process.request_execution_stop();
        for &vcpu in installed {
            let _ = self.workers.retire_process(vcpu, process_id);
        }
        process.complete_cpu_thread_retirement();
    }

    fn retire_process_cpu_threads(
        &mut self,
        process_id: ProcessId,
    ) -> Result<(), CoordinatorError> {
        let main_thread = self
            .processes
            .get(&process_id)
            .ok_or(CoordinatorError::UnknownProcess(process_id))?
            .main_thread_id();
        self.processes
            .get(&process_id)
            .expect("the validated process remains registered")
            .request_execution_stop()
            .map_err(|fault| CoordinatorError::Execution {
                process: process_id,
                thread: main_thread,
                error: ProcessExecutionError::Cpu { fault },
            })?;
        let vcpus: Vec<_> = self
            .scheduler
            .profile()
            .vcpus()
            .iter()
            .map(|descriptor| descriptor.id())
            .collect();
        for vcpu in vcpus {
            self.workers
                .retire_process(vcpu, process_id)
                .map_err(CoordinatorError::Worker)?;
        }
        self.processes
            .get(&process_id)
            .expect("the stopped process remains registered")
            .complete_cpu_thread_retirement();
        Ok(())
    }

    /// Publishes an interrupt to one physical emulated CPU. The interrupt
    /// remains pending until a backend observes it at a bounded safepoint.
    pub fn post_vcpu_interrupt(
        &mut self,
        vcpu: VirtualCpuId,
        mask: u32,
    ) -> Result<(), CoordinatorError> {
        let events = self
            .vcpu_events
            .get(&vcpu)
            .ok_or(SchedulerError::UnknownVirtualCpu(vcpu))?;
        if mask == 0 {
            return Ok(());
        }
        events.post_interrupts(mask);
        if let Some(lease) = self
            .scheduler
            .active_leases()
            .find(|lease| lease.vcpu == vcpu)
        {
            self.processes[&lease.process].request_vcpu_safepoint(vcpu);
        }
        Ok(())
    }

    pub fn route_supervisor_call<D: ExceptionDispatcher>(
        &mut self,
        lease: Lease,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, CoordinatorRouteError> {
        let thread =
            self.scheduler
                .thread(lease.thread)
                .ok_or(CoordinatorRouteError::Scheduler(
                    SchedulerError::UnknownThread(lease.thread),
                ))?;
        if thread.lifecycle != nixe_scheduler::ThreadLifecycle::Waiting {
            return Err(CoordinatorRouteError::Route(
                ExceptionRouteError::ProcessNotSuspended {
                    lifecycle: thread.lifecycle,
                },
            ));
        }
        let process = self
            .processes
            .get(&lease.process)
            .ok_or(CoordinatorRouteError::UnknownProcess(lease.process))?;
        let (thread, result) = process
            .route_supervisor_call_for(lease.thread, lease.vcpu, stop, dispatcher)
            .map_err(CoordinatorRouteError::Route)?;
        self.commit_exception_result(lease.process, thread, result)
    }

    /// Resumes only the current owned kernel operation. It consumes the old
    /// wait generation without making guest code runnable between kernel phases.
    pub fn resume_supervisor_call<F>(
        &mut self,
        token: WakeToken,
        vcpu: VirtualCpuId,
        source: nixe_cpu::location::LocationDescriptor,
        continuation: impl FnOnce(
            &mut crate::ExceptionDispatchContext<'_>,
        ) -> crate::ExceptionDispatchOutcome<F>,
    ) -> Result<Option<ExceptionHandlingResult<F>>, CoordinatorRouteError> {
        let Some(thread) = self.scheduler.thread(token.thread).filter(|thread| {
            thread.active_wait == Some(token)
                && thread.lifecycle == nixe_scheduler::ThreadLifecycle::Waiting
        }) else {
            return Ok(None);
        };
        let process_id = thread.process;
        self.scheduler
            .continue_wait(token)
            .map_err(CoordinatorRouteError::Scheduler)?;
        let (thread, result) = self
            .processes
            .get(&process_id)
            .ok_or(CoordinatorRouteError::UnknownProcess(process_id))?
            .resume_supervisor_call_for(token.thread, vcpu, source, continuation)
            .map_err(CoordinatorRouteError::Route)?;
        self.commit_exception_result(process_id, thread, result)
            .map(Some)
    }

    fn commit_exception_result<F>(
        &mut self,
        process_id: ProcessId,
        thread: std::sync::Arc<crate::ThreadObject>,
        result: ExceptionHandlingResult<F>,
    ) -> Result<ExceptionHandlingResult<F>, CoordinatorRouteError> {
        let thread_id = thread.id();
        let terminal = matches!(
            result,
            ExceptionHandlingResult::Terminated { .. } | ExceptionHandlingResult::Fault(_)
        );
        if terminal {
            self.finalize_thread_waits(process_id, thread_id);
        }
        match &result {
            ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_) => {
                self.scheduler.make_ready(thread_id)
            }
            ExceptionHandlingResult::Suspended => Ok(()),
            ExceptionHandlingResult::Terminated { .. } => {
                self.scheduler.terminate(thread_id, false)
            }
            ExceptionHandlingResult::Fault(_) => self.scheduler.terminate(thread_id, true),
        }
        .map_err(CoordinatorRouteError::Scheduler)?;
        if terminal {
            // Publish completion after native ownership and the architectural
            // continuation have been released, as FinishTermination does:
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L428-L447
            thread.signal();
        }
        if terminal {
            if self.processes[&process_id].lifecycle()
                == nixe_scheduler::ProcessLifecycle::Terminating
            {
                self.quiesce_process(process_id)
                    .map_err(|error| CoordinatorRouteError::Coordinator(Box::new(error)))?;
            }
            self.finish_process_termination(process_id)
                .map_err(|error| CoordinatorRouteError::Coordinator(Box::new(error)))?;
        }
        Ok(result)
    }

    #[must_use]
    pub fn process_count(&self) -> usize {
        self.processes.len()
    }

    #[must_use]
    pub fn resource_counts(&self) -> CoordinatorResourceCounts {
        CoordinatorResourceCounts {
            processes: self.processes.len(),
            scheduled_threads: self.scheduler.thread_count(),
            active_waits: self.scheduler.active_wait_count(),
            deadlines: self.scheduler.deadline_count(),
            priority_donations: self.scheduler.priority_donation_count(),
            address_waiters: self
                .processes
                .values()
                .map(|process| process.address_waits().lock().waiter_count())
                .sum(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CoordinatorError {
    ForeignKernel,
    MissingContinuation(GuestThreadId),
    TimerStartup(std::io::ErrorKind),
    TimerWorkerFailed,
    UnknownProcess(ProcessId),
    UnknownThread {
        process: ProcessId,
        thread: GuestThreadId,
    },
    Scheduler(SchedulerError),
    ThreadLocal {
        thread: GuestThreadId,
        fault: std::sync::Arc<nixe_memory::MemoryError>,
    },
    InFlightLease(Lease),
    ProcessIdExhausted,
    ThreadTable(crate::ThreadTableError),
    Execution {
        process: ProcessId,
        thread: GuestThreadId,
        error: ProcessExecutionError,
    },
    ExternalEvent(ExternalEventSendError),
    WorkerStartup(std::io::ErrorKind),
    Worker(WorkerFailure),
    ProcessTeardown {
        process: ProcessId,
        failure: crate::ProcessTeardownFailure,
    },
}

impl From<crate::ThreadTableError> for CoordinatorError {
    fn from(value: crate::ThreadTableError) -> Self {
        Self::ThreadTable(value)
    }
}

impl From<SchedulerError> for CoordinatorError {
    fn from(value: SchedulerError) -> Self {
        Self::Scheduler(value)
    }
}

impl From<ExternalEventSendError> for CoordinatorError {
    fn from(value: ExternalEventSendError) -> Self {
        Self::ExternalEvent(value)
    }
}

impl Display for CoordinatorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "runtime coordinator rejected operation: {self:?}"
        )
    }
}

impl Error for CoordinatorError {}

impl Drop for RuntimeCoordinator {
    fn drop(&mut self) {
        if self.shutdown().is_err() {
            for process in self.processes.values() {
                process.request_execution_safepoint();
            }
            let _ = self.workers.shutdown();
            let _ = self.scheduler.stop_timers();
            for process in self.processes.values() {
                let _ = process.try_teardown();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum CoordinatorRouteError {
    Coordinator(Box<CoordinatorError>),
    UnknownProcess(ProcessId),
    Route(ExceptionRouteError),
    Scheduler(SchedulerError),
}

impl Display for CoordinatorRouteError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "runtime coordinator could not route exception: {self:?}"
        )
    }
}

impl Error for CoordinatorRouteError {}

pub struct CoordinatorExecution {
    pub lease: Lease,
    pub report: ExecutionReport,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoordinatorDrainReport {
    pub received: usize,
    pub first_sequence: Option<crate::ExternalEventSequence>,
    pub last_sequence: Option<crate::ExternalEventSequence>,
}

#[cfg(test)]
mod tests;
