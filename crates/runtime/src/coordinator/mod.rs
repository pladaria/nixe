use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use nixe_cpu::execution::{SchedulerRequest, VcpuEventState};
use nixe_scheduler::{
    Completion, CoreSet, GuestThreadId, Lease, MachineSchedulerProfile, ProcessId, Readiness,
    ScheduledThreadConfig, SchedulerCommand, SchedulerDecision, SchedulerError, SchedulerState,
    VirtualCpuId, WakeToken,
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
mod worker;

use identity::{GuestThreadIdAllocator, ProcessIdAllocator};
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PriorityDonation {
    owner: GuestThreadId,
    waiter: GuestThreadId,
    key: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CpuWait {
    vcpu: VirtualCpuId,
    request: SchedulerRequest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AdaptiveExecutionBudget {
    baseline: u64,
    current: u64,
    ceiling: u64,
}

impl AdaptiveExecutionBudget {
    const DEFAULT_CEILING: u64 = 100_000;

    const fn new(baseline: u64) -> Self {
        Self {
            baseline,
            current: baseline,
            ceiling: if baseline > Self::DEFAULT_CEILING {
                baseline
            } else {
                Self::DEFAULT_CEILING
            },
        }
    }

    fn observe(&mut self, uninterrupted: bool) {
        self.current = if uninterrupted {
            self.current.saturating_mul(2).min(self.ceiling)
        } else {
            self.baseline
        };
    }
}

/// System-level owner of process lookup and the pure scheduler state machine.
/// CPU backend state remains encapsulated by its registered process runtime.
pub struct RuntimeCoordinator {
    scheduler: SchedulerState,
    processes: BTreeMap<ProcessId, RunnableProcess>,
    process_ids: ProcessIdAllocator,
    workers: VcpuWorkerPool,
    thread_ids: GuestThreadIdAllocator,
    inbox: ExternalEventInbox,
    host_stop_requested: bool,
    virtual_clock: crate::VirtualClock,
    deadlines: BTreeMap<(u64, u64), WakeToken>,
    next_deadline_sequence: u64,
    priority_donations: BTreeSet<PriorityDonation>,
    vcpu_events: BTreeMap<VirtualCpuId, VcpuEventState>,
    cpu_waits: BTreeMap<GuestThreadId, CpuWait>,
    budgets: BTreeMap<VirtualCpuId, AdaptiveExecutionBudget>,
    next_preemption_ns: Option<u64>,
    completed_executions: VecDeque<CoordinatorExecution>,
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
        let adaptive_budget =
            AdaptiveExecutionBudget::new(profile.default_timeslice_instructions());
        let budgets = profile
            .vcpus()
            .iter()
            .map(|descriptor| (descriptor.id(), adaptive_budget))
            .collect();
        let workers =
            VcpuWorkerPool::start(profile.vcpus().iter().map(|descriptor| descriptor.id()))
                .map_err(|error| CoordinatorError::WorkerStartup(error.kind()))?;
        let next_preemption_ns = profile
            .preemption_interval_ns()
            .map(|interval| virtual_clock.scheduler_time_ns().saturating_add(interval));
        Ok(Self {
            scheduler: SchedulerState::new(profile),
            processes: BTreeMap::new(),
            process_ids: ProcessIdAllocator::new(),
            workers,
            thread_ids: GuestThreadIdAllocator::new(),
            inbox: ExternalEventInbox::bounded(1_024)
                .expect("the default external event capacity is non-zero"),
            host_stop_requested: false,
            virtual_clock,
            deadlines: BTreeMap::new(),
            next_deadline_sequence: 1,
            priority_donations: BTreeSet::new(),
            vcpu_events,
            cpu_waits: BTreeMap::new(),
            budgets,
            next_preemption_ns,
            completed_executions: VecDeque::new(),
        })
    }

    /// Quiesces workers and releases every process, CPU backend, wait, and
    /// canonical address-space resource. Repeated shutdown calls are harmless.
    pub fn shutdown(&mut self) -> Result<(), CoordinatorError> {
        self.quiesce()?;
        let processes: Vec<_> = self.processes.keys().copied().collect();
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
        self.cpu_waits.clear();
        self.deadlines.clear();
        self.host_stop_requested = true;
        self.workers.shutdown().map_err(CoordinatorError::Worker)
    }

    #[must_use]
    pub const fn scheduler(&self) -> &SchedulerState {
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
        let (thread, next_thread_id) = self
            .thread_ids
            .candidate()
            .ok_or(CoordinatorError::ThreadIdExhausted)?;
        process.assign_main_thread_id(thread)?;
        self.scheduler
            .apply(SchedulerCommand::Register(ScheduledThreadConfig {
                process: id,
                thread,
                base_priority: registration.priority,
                effective_priority: registration.priority,
                ideal_vcpu: registration.ideal_vcpu,
                affinity: registration.affinity,
            }))?;
        if let Err(error) = self.scheduler.apply(SchedulerCommand::MakeReady(thread)) {
            self.scheduler
                .apply(SchedulerCommand::Unregister(thread))
                .expect("registration rollback removes an unleased created thread");
            return Err(error.into());
        }
        if let Err(error) = self.install_process_cpu_threads(id, &mut process, thread) {
            self.scheduler
                .apply(SchedulerCommand::Unregister(thread))
                .expect("CPU thread installation rollback removes an unleased ready thread");
            return Err(error);
        }
        let replaced = self.processes.insert(id, process);
        debug_assert!(replaced.is_none());
        self.thread_ids.commit(next_thread_id);
        self.process_ids.commit(next_process_id);
        Ok(id)
    }

    /// Stops a process execution domain, quiesces and drops every worker-owned
    /// CPU thread, then transfers the process solely for final resource teardown.
    /// A removed process is terminal and cannot be registered again.
    pub fn remove_process(&mut self, id: ProcessId) -> Result<RunnableProcess, CoordinatorError> {
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
        let removed_threads: BTreeSet<_> = threads.iter().copied().collect();
        self.priority_donations.retain(|donation| {
            !removed_threads.contains(&donation.owner)
                && !removed_threads.contains(&donation.waiter)
        });
        for thread in threads {
            self.cpu_waits.remove(&thread);
            self.release_wait_resources(thread);
            self.scheduler.apply(SchedulerCommand::Unregister(thread))?;
        }
        let process = self
            .processes
            .remove(&id)
            .expect("process existed throughout atomic scheduler removal");
        self.recompute_effective_priorities()?;
        Ok(process)
    }

    /// Terminates every thread in a registered process through the scheduler
    /// and records a host-requested process exit without releasing resources.
    pub fn terminate_process(&mut self, id: ProcessId) -> Result<bool, CoordinatorError> {
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
        let terminated = self
            .processes
            .get_mut(&id)
            .expect("the process was validated")
            .terminate_from_host();
        if terminated {
            for thread in threads {
                self.cpu_waits.remove(&thread);
                if self.scheduler.thread(thread).is_some_and(|view| {
                    matches!(
                        view.lifecycle,
                        nixe_scheduler::ThreadLifecycle::Exited
                            | nixe_scheduler::ThreadLifecycle::Faulted
                    )
                }) {
                    continue;
                }
                self.scheduler.apply(SchedulerCommand::Terminate {
                    thread,
                    faulted: false,
                })?;
            }
        }
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
        self.scheduler.apply(SchedulerCommand::MakeReady(thread))?;
        self.cpu_waits.remove(&thread);
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
        self.wake_cpu_waiters(vcpu, true)?;
        Ok(())
    }

    fn send_event(&mut self) -> Result<(), CoordinatorError> {
        let vcpus: Vec<_> = self.vcpu_events.keys().copied().collect();
        for events in self.vcpu_events.values() {
            events.signal_event();
        }
        for vcpu in vcpus {
            self.wake_cpu_waiters(vcpu, false)?;
        }
        Ok(())
    }

    fn wake_cpu_waiters(
        &mut self,
        vcpu: VirtualCpuId,
        interrupt: bool,
    ) -> Result<(), CoordinatorError> {
        let threads: Vec<_> = self
            .cpu_waits
            .iter()
            .filter_map(|(thread, wait)| {
                (wait.vcpu == vcpu
                    && (wait.request == SchedulerRequest::WaitForEvent
                        || (interrupt && wait.request == SchedulerRequest::WaitForInterrupt)))
                    .then_some(*thread)
            })
            .collect();
        if !interrupt && !threads.is_empty() {
            let _ = self
                .vcpu_events
                .get(&vcpu)
                .expect("a recorded CPU wait references a configured vCPU")
                .consume_event();
        }
        for thread in threads {
            self.cpu_waits.remove(&thread);
            self.scheduler.apply(SchedulerCommand::MakeReady(thread))?;
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
        let other_live = self.other_live_threads(lease.process, lease.thread)?;
        let process = self
            .processes
            .get_mut(&lease.process)
            .ok_or(CoordinatorRouteError::UnknownProcess(lease.process))?;
        let result = process
            .route_supervisor_call_for(lease.thread, lease.vcpu, stop, dispatcher, other_live)
            .map_err(CoordinatorRouteError::Route)?;
        self.commit_exception_result(lease.process, lease.thread, result)
    }

    fn other_live_threads(
        &self,
        process_id: ProcessId,
        thread_id: GuestThreadId,
    ) -> Result<bool, CoordinatorRouteError> {
        Ok(self
            .processes
            .get(&process_id)
            .ok_or(CoordinatorRouteError::UnknownProcess(process_id))?
            .threads()
            .iter()
            .filter(|(id, _)| **id != thread_id)
            .any(|(id, _)| {
                self.scheduler.thread(*id).is_some_and(|thread| {
                    !matches!(
                        thread.lifecycle,
                        nixe_scheduler::ThreadLifecycle::Exited
                            | nixe_scheduler::ThreadLifecycle::Faulted
                    )
                })
            }))
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
        let other_live = self.other_live_threads(process_id, token.thread)?;
        self.release_wait_resources(token.thread);
        self.scheduler
            .apply(SchedulerCommand::ContinueWait(token))
            .map_err(CoordinatorRouteError::Scheduler)?;
        let result = self
            .processes
            .get_mut(&process_id)
            .ok_or(CoordinatorRouteError::UnknownProcess(process_id))?
            .resume_supervisor_call_for(token.thread, vcpu, source, other_live, continuation)
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
        let command = match &result {
            ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_) => {
                Some(SchedulerCommand::MakeReady(thread_id))
            }
            ExceptionHandlingResult::Suspended => None,
            ExceptionHandlingResult::Terminated { .. } => Some(SchedulerCommand::Terminate {
                thread: thread_id,
                faulted: false,
            }),
            ExceptionHandlingResult::Fault(_) => Some(SchedulerCommand::Terminate {
                thread: thread_id,
                faulted: true,
            }),
        };
        let terminal = matches!(
            result,
            ExceptionHandlingResult::Terminated { .. } | ExceptionHandlingResult::Fault(_)
        );
        if let Some(command) = command {
            self.scheduler
                .apply(command)
                .map_err(CoordinatorRouteError::Scheduler)?;
        }
        if terminal {
            self.release_wait_resources(thread_id);
            self.processes
                .get_mut(&process_id)
                .expect("the terminal thread retains its owning process")
                .address_waits()
                .lock()
                .release_thread(thread_id);
            self.priority_donations
                .retain(|donation| donation.owner != thread_id && donation.waiter != thread_id);
            self.recompute_effective_priorities()
                .map_err(CoordinatorRouteError::Scheduler)?;
            let reap_unreferenced = self
                .processes
                .get(&process_id)
                .and_then(|process| {
                    (process.main_thread_id() != thread_id)
                        .then(|| process.thread(thread_id))
                        .flatten()
                })
                .is_some_and(|thread| thread.object_identity_reference_count() == 1);
            if reap_unreferenced {
                self.scheduler
                    .apply(SchedulerCommand::Unregister(thread_id))
                    .map_err(CoordinatorRouteError::Scheduler)?;
                self.processes
                    .get_mut(&process_id)
                    .expect("the terminated thread retains its owning process")
                    .reap_exited_thread(thread_id)
                    .expect("an unreferenced terminal thread can be reaped");
            }
        }
        if let ExceptionHandlingResult::Terminated {
            scope: crate::ExceptionTerminationScope::Process,
            exit_code,
            ..
        } = &result
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
            active_waits: self
                .scheduler
                .active_wait_count()
                .saturating_add(self.cpu_waits.len()),
            deadlines: self.deadlines.len(),
            external_watcher_groups: self.inbox.sender().watcher_group_count(),
            priority_donations: self.priority_donations.len(),
            address_waiters: self
                .processes
                .values()
                .map(|process| process.address_waits().lock().waiter_count())
                .sum(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoordinatorError {
    UnknownProcess(ProcessId),
    UnknownThread {
        process: ProcessId,
        thread: GuestThreadId,
    },
    Scheduler(SchedulerError),
    InFlightLease(Lease),
    ThreadIdExhausted,
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
