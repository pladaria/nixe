use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use nixe_cpu::execution::VcpuEventState;
use nixe_scheduler::{
    Completion, CoreSet, GuestThreadId, Lease, MachineSchedulerProfile, ProcessId,
    ScheduledThreadConfig, SchedulerError, SchedulerState, VirtualCpuId, WakeToken,
};

use crate::{
    ExceptionDispatcher, ExceptionHandlingResult, ExceptionRouteError, ExecutionReport,
    ExecutionStop, ExternalEvent, ExternalEventInbox, ExternalEventSendError, ExternalEventSender,
    ProcessExecutionError, RunnableProcess, ThreadCreateError, ThreadCreateRequest, ThreadCreation,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadOperationError {
    InvalidHandle,
    InvalidState,
    Internal,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoordinatorResourceCounts {
    pub processes: usize,
    pub scheduled_threads: usize,
    pub active_waits: usize,
    pub deadlines: usize,
    pub external_watcher_groups: usize,
    pub priority_donations: usize,
    pub address_waiters: usize,
}

pub struct RuntimeCoordinator {
    scheduler: std::sync::Arc<SchedulerState>,
    processes: BTreeMap<ProcessId, RunnableProcess>,
    process_ids: ProcessIdAllocator,
    workers: VcpuWorkerPool,
    inbox: ExternalEventInbox,
    host_stop_requested: bool,
    virtual_clock: crate::VirtualClock,
    deadlines: BTreeMap<(u64, u64), WakeToken>,
    next_deadline_sequence: u64,
    vcpu_events: BTreeMap<VirtualCpuId, VcpuEventState>,
    next_preemption_ns: Option<u64>,
    completed_executions: VecDeque<CoordinatorExecution>,
    deferred_events: VecDeque<crate::SequencedExternalEvent>,
}

impl RuntimeCoordinator {
    #[must_use]
    pub fn new(profile: MachineSchedulerProfile) -> Self {
        Self::with_virtual_clock(
            profile,
            crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
        )
    }

    #[must_use]
    pub fn with_virtual_clock(
        profile: MachineSchedulerProfile,
        virtual_clock: crate::VirtualClock,
    ) -> Self {
        Self::try_with_virtual_clock(profile, virtual_clock)
            .expect("vCPU worker construction failed")
    }

    pub fn try_with_virtual_clock(
        profile: MachineSchedulerProfile,
        virtual_clock: crate::VirtualClock,
    ) -> Result<Self, CoordinatorError> {
        let vcpu_events = profile
            .vcpus()
            .iter()
            .map(|descriptor| (descriptor.id(), VcpuEventState::default()))
            .collect();
        let inbox = ExternalEventInbox::bounded(1_024, profile.vcpus().len())
            .expect("the default event capacity is non-zero");
        let next_preemption_ns = profile
            .preemption_interval_ns()
            .map(|interval| virtual_clock.scheduler_time_ns().saturating_add(interval));
        let scheduler = std::sync::Arc::new(SchedulerState::new(profile, virtual_clock.clone()));
        let workers = VcpuWorkerPool::start(
            scheduler
                .profile()
                .vcpus()
                .iter()
                .map(|descriptor| descriptor.id()),
            inbox.worker_sender(),
            scheduler.clone(),
        )
        .map_err(|error| CoordinatorError::WorkerStartup(error.kind()))?;
        Ok(Self {
            scheduler,
            processes: BTreeMap::new(),
            process_ids: ProcessIdAllocator::new(),
            workers,
            inbox,
            host_stop_requested: false,
            virtual_clock,
            deadlines: BTreeMap::new(),
            next_deadline_sequence: 1,
            vcpu_events,
            next_preemption_ns,
            completed_executions: VecDeque::new(),
            deferred_events: VecDeque::new(),
        })
    }

    /// Quiesces workers and releases every process, CPU backend, wait, and
    /// canonical address-space resource. Repeated shutdown calls are harmless.
    pub fn shutdown(&mut self) -> Result<(), CoordinatorError> {
        let processes: Vec<_> = self.processes.keys().copied().collect();
        for &process in &processes {
            self.request_process_stop(process)?;
        }
        self.quiesce()?;
        for process_id in processes {
            let process = self.remove_process(process_id)?;
            process
                .try_teardown()
                .map_err(|failure| CoordinatorError::ProcessTeardown {
                    process: process_id,
                    failure,
                })?;
        }
        let waiting_threads: Vec<_> = self
            .scheduler
            .active_waits()
            .map(|token| token.thread)
            .collect();
        for thread in waiting_threads {
            self.release_wait_resources(thread);
        }
        self.deadlines.clear();
        self.host_stop_requested = true;
        self.workers.shutdown().map_err(CoordinatorError::Worker)
    }

    #[must_use]
    pub fn scheduler(&self) -> &SchedulerState {
        &self.scheduler
    }

    #[must_use]
    pub fn process(&self, id: ProcessId) -> Option<&RunnableProcess> {
        self.processes.get(&id)
    }

    pub fn process_mut(&mut self, id: ProcessId) -> Option<&mut RunnableProcess> {
        self.processes.get_mut(&id)
    }

    pub fn register_process(
        &mut self,
        mut process: RunnableProcess,
        registration: ProcessRegistration,
    ) -> Result<ProcessId, CoordinatorError> {
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
        if let Err(error) = self.install_process_cpu_threads(id, &mut process, thread) {
            self.scheduler
                .unregister(thread)
                .expect("CPU thread installation rollback removes an unleased ready thread");
            return Err(error);
        }
        process.address_waits().bind(id, self.scheduler.clone());
        process
            .main_thread()
            .object()
            .bind_retirement(id, self.inbox.sender());
        process
            .scheduler
            .set(self.scheduler.clone())
            .expect("process is registered once");
        let replaced = self.processes.insert(id, process);
        debug_assert!(replaced.is_none());
        self.process_ids.commit(next_process_id);
        Ok(id)
    }

    /// Stops a process execution domain, quiesces and drops every worker-owned
    /// CPU thread, then transfers the process solely for final resource teardown.
    /// A removed process is terminal and cannot be registered again.
    pub fn remove_process(&mut self, id: ProcessId) -> Result<RunnableProcess, CoordinatorError> {
        self.terminate_process(id)?;
        self.completed_executions
            .retain(|execution| execution.lease.process != id);
        let threads: Vec<_> = self
            .processes
            .get(&id)
            .ok_or(CoordinatorError::UnknownProcess(id))?
            .threads()
            .iter()
            .map(|(thread, _)| *thread)
            .collect();
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
            self.scheduler.unregister(thread)?;
        }
        let process = self
            .processes
            .remove(&id)
            .expect("process existed throughout atomic scheduler removal");
        Ok(process)
    }

    /// Terminates every thread in a registered process through the scheduler
    /// and records a host-requested process exit without releasing resources.
    pub fn terminate_process(&mut self, id: ProcessId) -> Result<bool, CoordinatorError> {
        let terminated = self.request_process_stop(id)?;
        self.quiesce()?;
        self.completed_executions
            .retain(|execution| execution.lease.process != id);
        let threads: Vec<_> = self
            .processes
            .get(&id)
            .ok_or(CoordinatorError::UnknownProcess(id))?
            .threads()
            .iter()
            .map(|(thread, _)| *thread)
            .collect();
        for thread in threads {
            if self.scheduler.thread(thread).is_some_and(|view| {
                matches!(
                    view.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited
                        | nixe_scheduler::ThreadLifecycle::Faulted
                )
            }) {
                continue;
            }
            self.processes
                .get_mut(&id)
                .unwrap()
                .terminate_thread_from_coordinator(
                    thread,
                    0,
                    crate::ExceptionTerminationScope::Process,
                );
            self.finalize_thread_waits(id, thread);
            self.scheduler.terminate(thread, false)?;
        }
        Ok(terminated)
    }

    fn request_process_stop(&mut self, id: ProcessId) -> Result<bool, CoordinatorError> {
        let process = self
            .processes
            .get_mut(&id)
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
        process: &mut RunnableProcess,
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
        process: &mut RunnableProcess,
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
            .get_mut(&process_id)
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
            .get_mut(&process_id)
            .expect("the stopped process remains registered")
            .complete_cpu_thread_retirement();
        Ok(())
    }

    /// Makes one waiting guest thread ready.
    pub fn make_thread_ready(&mut self, thread: GuestThreadId) -> Result<(), CoordinatorError> {
        self.scheduler.make_ready(thread)?;
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
            .get_mut(&lease.process)
            .ok_or(CoordinatorRouteError::UnknownProcess(lease.process))?;
        let result = process
            .route_supervisor_call_for(lease.thread, lease.vcpu, stop, dispatcher)
            .map_err(CoordinatorRouteError::Route)?;
        self.commit_exception_result(lease.process, lease.thread, result)
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
        self.release_wait_resources(token.thread);
        self.scheduler
            .continue_wait(token)
            .map_err(CoordinatorRouteError::Scheduler)?;
        let result = self
            .processes
            .get_mut(&process_id)
            .ok_or(CoordinatorRouteError::UnknownProcess(process_id))?
            .resume_supervisor_call_for(token.thread, vcpu, source, continuation)
            .map_err(CoordinatorRouteError::Route)?;
        self.commit_exception_result(process_id, token.thread, result)
            .map(Some)
    }

    fn commit_exception_result<F>(
        &mut self,
        process_id: ProcessId,
        thread_id: GuestThreadId,
        result: ExceptionHandlingResult<F>,
    ) -> Result<ExceptionHandlingResult<F>, CoordinatorRouteError> {
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
        if let ExceptionHandlingResult::Terminated { exit_code, .. } = &result
            && self.processes[&process_id].lifecycle() == nixe_scheduler::ProcessLifecycle::Exited
        {
            self.terminate_remaining_process_threads(process_id, thread_id, *exit_code)?;
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
            deadlines: self.deadlines.len(),
            external_watcher_groups: self.inbox.sender().watcher_group_count(),
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
    DeadlineSequenceExhausted,
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
        let _ = self.shutdown();
        if !self.host_stop_requested {
            for process in self.processes.values() {
                process.request_execution_safepoint();
            }
            let _ = self.workers.shutdown();
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
    pub woken: usize,
    pub cancelled: usize,
    pub stale: usize,
    pub first_sequence: Option<crate::ExternalEventSequence>,
    pub last_sequence: Option<crate::ExternalEventSequence>,
}

#[cfg(test)]
mod tests;
