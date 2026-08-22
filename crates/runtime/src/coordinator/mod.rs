use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

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
mod vcpu;
mod wait;
mod worker;

use identity::{GuestThreadIdAllocator, ProcessIdAllocator};
use vcpu::RuntimeVcpuSlot;
pub use worker::WorkerFailure;
use worker::{VcpuWorkerPool, WorkerExecutorKey, WorkerRequest, WorkerRunFailure};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum VcpuExecutionMode {
    #[default]
    Deterministic,
    Parallel,
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadSchedulingInfo {
    pub id: GuestThreadId,
    pub priority: i32,
    pub effective_priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
    pub lifecycle: nixe_scheduler::ThreadLifecycle,
    pub last_vcpu: Option<VirtualCpuId>,
    pub paused: bool,
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

/// System-level owner of process lookup and the pure scheduler state machine.
/// Engine domains remain encapsulated by their registered process runtime.
pub struct RuntimeCoordinator {
    scheduler: SchedulerState,
    processes: BTreeMap<ProcessId, RunnableProcess>,
    process_ids: ProcessIdAllocator,
    in_flight: BTreeMap<VirtualCpuId, Lease>,
    vcpu_slots: BTreeMap<VirtualCpuId, RuntimeVcpuSlot>,
    workers: VcpuWorkerPool,
    execution_mode: VcpuExecutionMode,
    execution_record: Option<crate::ExecutionRecord>,
    next_record_sequence: u64,
    record_dispatch_sequences: BTreeMap<VirtualCpuId, u64>,
    replay_expected: Option<crate::ExecutionRecord>,
    replay_dispatches: VecDeque<(u64, Lease, u64)>,
    thread_ids: GuestThreadIdAllocator,
    inbox: ExternalEventInbox,
    host_stop_requested: bool,
    virtual_clock: crate::VirtualClock,
    deadlines: BTreeMap<(u64, u64), WakeToken>,
    next_deadline_sequence: u64,
    active_waits: BTreeMap<GuestThreadId, WakeToken>,
    priority_donations: BTreeSet<PriorityDonation>,
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
        Self::try_with_execution_mode(profile, virtual_clock, VcpuExecutionMode::Deterministic)
            .expect("vCPU worker construction failed")
    }

    pub fn try_with_execution_mode(
        profile: MachineSchedulerProfile,
        virtual_clock: crate::VirtualClock,
        execution_mode: VcpuExecutionMode,
    ) -> Result<Self, CoordinatorError> {
        let vcpu_slots = profile
            .vcpus()
            .iter()
            .map(|descriptor| (descriptor.id(), RuntimeVcpuSlot::default()))
            .collect();
        let workers = VcpuWorkerPool::start(
            profile.vcpus().iter().map(|descriptor| descriptor.id()),
            execution_mode == VcpuExecutionMode::Deterministic,
        )
        .map_err(|error| CoordinatorError::WorkerStartup(error.kind()))?;
        Ok(Self {
            scheduler: SchedulerState::new(profile),
            processes: BTreeMap::new(),
            process_ids: ProcessIdAllocator::new(),
            in_flight: BTreeMap::new(),
            vcpu_slots,
            workers,
            execution_mode,
            execution_record: None,
            next_record_sequence: 1,
            record_dispatch_sequences: BTreeMap::new(),
            replay_expected: None,
            replay_dispatches: VecDeque::new(),
            thread_ids: GuestThreadIdAllocator::new(),
            inbox: ExternalEventInbox::bounded(1_024)
                .expect("the default external event capacity is non-zero"),
            host_stop_requested: false,
            virtual_clock,
            deadlines: BTreeMap::new(),
            next_deadline_sequence: 1,
            active_waits: BTreeMap::new(),
            priority_donations: BTreeSet::new(),
        })
    }

    #[must_use]
    pub const fn execution_mode(&self) -> VcpuExecutionMode {
        self.execution_mode
    }

    pub fn enable_execution_recording(&mut self, capacity: std::num::NonZeroUsize) {
        self.execution_record = Some(crate::ExecutionRecord::new(capacity));
        self.next_record_sequence = 1;
        self.record_dispatch_sequences.clear();
    }

    pub fn enable_sanitized_execution_recording(&mut self, capacity: std::num::NonZeroUsize) {
        self.execution_record = Some(crate::ExecutionRecord::sanitized(capacity));
        self.next_record_sequence = 1;
        self.record_dispatch_sequences.clear();
    }

    #[must_use]
    pub fn execution_record(&self) -> Option<&crate::ExecutionRecord> {
        self.execution_record.as_ref()
    }

    pub fn take_execution_record(&mut self) -> Option<crate::ExecutionRecord> {
        self.record_dispatch_sequences.clear();
        self.execution_record.take()
    }

    /// Installs a parallel observation record as deterministic dispatch input.
    /// The scheduler must independently reproduce every recorded thread lease.
    pub fn begin_differential_replay(
        &mut self,
        expected: crate::ExecutionRecord,
    ) -> Result<(), CoordinatorError> {
        if self.execution_mode != VcpuExecutionMode::Deterministic {
            return Err(CoordinatorError::ReplayRequiresDeterministicMode);
        }
        let observed = if expected.retains_architectural_context() {
            crate::ExecutionRecord::new(expected.capacity())
        } else {
            crate::ExecutionRecord::sanitized(expected.capacity())
        };
        self.replay_dispatches = expected.dispatches().into();
        self.replay_expected = Some(expected);
        self.execution_record = Some(observed);
        self.next_record_sequence = 1;
        self.record_dispatch_sequences.clear();
        Ok(())
    }

    pub fn finish_differential_replay(
        &mut self,
    ) -> Result<crate::ExecutionRecord, CoordinatorError> {
        if !self.replay_dispatches.is_empty() {
            return Err(CoordinatorError::ReplayIncomplete {
                remaining_dispatches: self.replay_dispatches.len(),
            });
        }
        let expected = self
            .replay_expected
            .take()
            .ok_or(CoordinatorError::ReplayNotActive)?;
        let observed = self
            .take_execution_record()
            .ok_or(CoordinatorError::ReplayNotActive)?;
        expected
            .compare(&observed)
            .map_err(CoordinatorError::ReplayMismatch)?;
        Ok(observed)
    }

    /// Stops worker activity after requesting a bounded engine safepoint.
    /// Repeated shutdown calls are harmless.
    pub fn shutdown(&mut self) -> Result<(), CoordinatorError> {
        if let Some(lease) = self.in_flight.values().next().copied() {
            return Err(CoordinatorError::ShutdownWithOutstandingLease(lease));
        }
        for process in self.processes.values() {
            process.request_execution_safepoint();
        }
        let waiting_threads: Vec<_> = self.active_waits.keys().copied().collect();
        for thread in waiting_threads {
            self.release_wait_resources(thread);
        }
        self.deadlines.clear();
        self.host_stop_requested = true;
        self.workers.shutdown().map_err(CoordinatorError::Worker)
    }

    fn record_dispatch(&mut self, lease: Lease, instruction_budget: u64) {
        let sequence = self.next_record_sequence;
        self.next_record_sequence = self.next_record_sequence.saturating_add(1);
        self.record_dispatch_sequences.insert(lease.vcpu, sequence);
        if let Some(record) = &mut self.execution_record {
            record.push(crate::ExecutionObservation::Dispatch {
                sequence,
                lease,
                instruction_budget,
            });
        }
    }

    fn record_completion(&mut self, lease: Lease, report: &ExecutionReport) {
        let sequence = self
            .record_dispatch_sequences
            .remove(&lease.vcpu)
            .unwrap_or_default();
        if let Some(record) = &mut self.execution_record {
            record.push(crate::ExecutionObservation::Completion {
                sequence,
                lease,
                instructions_executed: report.instructions_executed,
                stop: recorded_stop(&report.stop),
                context: record
                    .retains_architectural_context()
                    .then(|| Box::new(report.context.clone())),
            });
        }
    }

    pub(super) fn record_external_event(&mut self, event: crate::SequencedExternalEvent) {
        if let Some(record) = &mut self.execution_record {
            record.push(crate::ExecutionObservation::External {
                sequence: event.sequence,
                event: event.event,
            });
        }
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
        if self.execution_mode == VcpuExecutionMode::Parallel {
            let descriptor = process.engine_descriptor();
            let capabilities = descriptor.capabilities;
            let required = process.engine_requirements(true, self.vcpu_slots.len());
            if !capabilities.contains(required)
                || !capabilities.supports_profile(process.cpu_context().profile(), required)
            {
                return Err(CoordinatorError::ParallelEngineUnsupported {
                    engine: descriptor.id,
                });
            }
        }
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
        if let Err(error) = self.install_process_executors(id, &mut process, thread) {
            self.scheduler
                .apply(SchedulerCommand::Unregister(thread))
                .expect("executor installation rollback removes an unleased ready thread");
            return Err(error);
        }
        let replaced = self.processes.insert(id, process);
        debug_assert!(replaced.is_none());
        self.thread_ids.commit(next_thread_id);
        self.process_ids.commit(next_process_id);
        Ok(id)
    }

    pub fn remove_process(&mut self, id: ProcessId) -> Result<RunnableProcess, CoordinatorError> {
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
        self.retire_process_executors(id)?;
        let removed_threads: BTreeSet<_> = threads.iter().copied().collect();
        self.priority_donations.retain(|donation| {
            !removed_threads.contains(&donation.owner)
                && !removed_threads.contains(&donation.waiter)
        });
        for thread in threads {
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
        if let Some(lease) = self
            .in_flight
            .values()
            .find(|lease| lease.process == id)
            .copied()
        {
            return Err(CoordinatorError::InFlightLease(lease));
        }
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
                self.scheduler.apply(SchedulerCommand::Terminate {
                    thread,
                    faulted: false,
                })?;
            }
        }
        Ok(terminated)
    }

    fn install_process_executors(
        &mut self,
        process_id: ProcessId,
        process: &mut RunnableProcess,
        thread: GuestThreadId,
    ) -> Result<(), CoordinatorError> {
        let key = WorkerExecutorKey {
            process: process_id,
            domain: process.engine_domain_id(),
        };
        let fallback_key = process
            .fallback_engine_domain_id()
            .map(|domain| WorkerExecutorKey {
                process: process_id,
                domain,
            });
        let vcpus: Vec<_> = self.vcpu_slots.keys().copied().collect();
        let mut installed = Vec::new();
        for vcpu in vcpus {
            let executor = process.take_worker_executor(vcpu).map_err(|fault| {
                CoordinatorError::Execution {
                    process: process_id,
                    thread,
                    error: ProcessExecutionError::Engine { fault },
                }
            })?;
            if let Err(failure) = self.workers.install_executor(vcpu, key, executor) {
                for installed_vcpu in installed {
                    if let Ok(executor) = self.workers.remove_executor(installed_vcpu, key) {
                        process.restore_worker_executor(installed_vcpu, executor);
                    }
                }
                return Err(CoordinatorError::Worker(failure));
            }
            if let Some(fallback_key) = fallback_key {
                let fallback = match process.take_worker_fallback_executor(vcpu) {
                    Ok(Some(executor)) => executor,
                    Ok(None) => unreachable!("a configured fallback domain creates executors"),
                    Err(fault) => {
                        if let Ok(executor) = self.workers.remove_executor(vcpu, key) {
                            process.restore_worker_executor(vcpu, executor);
                        }
                        for installed_vcpu in installed {
                            if let Ok(executor) =
                                self.workers.remove_executor(installed_vcpu, fallback_key)
                            {
                                process.restore_worker_fallback_executor(installed_vcpu, executor);
                            }
                            if let Ok(executor) = self.workers.remove_executor(installed_vcpu, key)
                            {
                                process.restore_worker_executor(installed_vcpu, executor);
                            }
                        }
                        return Err(CoordinatorError::Execution {
                            process: process_id,
                            thread,
                            error: ProcessExecutionError::Engine { fault },
                        });
                    }
                };
                if let Err(failure) = self.workers.install_executor(vcpu, fallback_key, fallback) {
                    if let Ok(executor) = self.workers.remove_executor(vcpu, key) {
                        process.restore_worker_executor(vcpu, executor);
                    }
                    for installed_vcpu in installed {
                        if let Ok(executor) =
                            self.workers.remove_executor(installed_vcpu, fallback_key)
                        {
                            process.restore_worker_fallback_executor(installed_vcpu, executor);
                        }
                        if let Ok(executor) = self.workers.remove_executor(installed_vcpu, key) {
                            process.restore_worker_executor(installed_vcpu, executor);
                        }
                    }
                    return Err(CoordinatorError::Worker(failure));
                }
            }
            installed.push(vcpu);
        }
        Ok(())
    }

    fn retire_process_executors(&mut self, process_id: ProcessId) -> Result<(), CoordinatorError> {
        self.processes
            .get(&process_id)
            .ok_or(CoordinatorError::UnknownProcess(process_id))?;
        let vcpus: Vec<_> = self.vcpu_slots.keys().copied().collect();
        for vcpu in vcpus {
            if self
                .workers
                .retire_process(vcpu, process_id)
                .map_err(CoordinatorError::Worker)?
                == 0
            {
                return Err(CoordinatorError::Worker(
                    WorkerFailure::ExecutorUnavailable {
                        process: process_id,
                        vcpu,
                    },
                ));
            }
        }
        Ok(())
    }

    /// Makes one waiting guest thread ready in both scheduler and process state.
    pub fn make_thread_ready(&mut self, thread: GuestThreadId) -> Result<(), CoordinatorError> {
        let view = self
            .scheduler
            .thread(thread)
            .ok_or(SchedulerError::UnknownThread(thread))?;
        let runtime_lifecycle = self
            .processes
            .get(&view.process)
            .ok_or(CoordinatorError::UnknownProcess(view.process))?
            .thread(thread)
            .ok_or(CoordinatorError::UnknownThread {
                process: view.process,
                thread,
            })?
            .lifecycle();
        self.scheduler.apply(SchedulerCommand::MakeReady(thread))?;
        if runtime_lifecycle == nixe_scheduler::ThreadLifecycle::Waiting {
            let resumed = self
                .processes
                .get_mut(&view.process)
                .expect("the owning process was validated")
                .resume_thread(thread);
            debug_assert!(resumed);
        }
        Ok(())
    }

    pub fn route_supervisor_call<D: ExceptionDispatcher>(
        &mut self,
        lease: Lease,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, CoordinatorRouteError> {
        let process = self
            .processes
            .get_mut(&lease.process)
            .ok_or(CoordinatorRouteError::UnknownProcess(lease.process))?;
        let result = process
            .route_supervisor_call_for(lease.thread, lease.vcpu, stop, dispatcher)
            .map_err(CoordinatorRouteError::Route)?;
        let command = match &result {
            ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_) => {
                Some(SchedulerCommand::MakeReady(lease.thread))
            }
            ExceptionHandlingResult::Suspended => None,
            ExceptionHandlingResult::Terminated { .. } => Some(SchedulerCommand::Terminate {
                thread: lease.thread,
                faulted: false,
            }),
            ExceptionHandlingResult::Fault(_) => Some(SchedulerCommand::Terminate {
                thread: lease.thread,
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
            self.release_wait_resources(lease.thread);
            self.processes
                .get_mut(&lease.process)
                .expect("the terminal thread retains its owning process")
                .address_waits_mut()
                .release_thread(lease.thread);
            self.priority_donations.retain(|donation| {
                donation.owner != lease.thread && donation.waiter != lease.thread
            });
            self.recompute_effective_priorities()
                .map_err(CoordinatorRouteError::Scheduler)?;
            let reap_unreferenced = self
                .processes
                .get(&lease.process)
                .and_then(|process| {
                    (process.main_thread_id() != lease.thread)
                        .then(|| process.thread(lease.thread))
                        .flatten()
                })
                .is_some_and(|thread| thread.object_identity_reference_count() == 1);
            if reap_unreferenced {
                self.scheduler
                    .apply(SchedulerCommand::Unregister(lease.thread))
                    .map_err(CoordinatorRouteError::Scheduler)?;
                self.processes
                    .get_mut(&lease.process)
                    .expect("the terminated thread retains its owning process")
                    .reap_exited_thread(lease.thread)
                    .expect("an unreferenced terminal thread can be reaped");
            }
        }
        if let ExceptionHandlingResult::Terminated {
            scope: crate::ExceptionTerminationScope::Process,
            exit_code,
            ..
        } = &result
        {
            self.terminate_remaining_process_threads(lease.process, lease.thread, *exit_code)?;
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
            active_waits: self.active_waits.len(),
            deadlines: self.deadlines.len(),
            external_watcher_groups: self.inbox.sender().watcher_group_count(),
            priority_donations: self.priority_donations.len(),
            address_waiters: self
                .processes
                .values()
                .map(|process| process.address_waits().waiter_count())
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
    ParallelModeRequired,
    ParallelEngineUnsupported {
        engine: nixe_cpu_engine::EngineId,
    },
    EngineUnavailable(nixe_cpu_engine::CapabilityReport),
    ShutdownWithOutstandingLease(Lease),
    ReplayRequiresDeterministicMode,
    ReplayIncomplete {
        remaining_dispatches: usize,
    },
    ReplayNotActive,
    ReplayLeaseMismatch {
        sequence: u64,
        expected: Lease,
        observed: Lease,
    },
    ReplayMismatch(crate::ReplayMismatch),
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
        for process in self.processes.values() {
            process.request_execution_safepoint();
        }
        let _ = self.workers.shutdown();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinatorRouteError {
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

fn completion_for_stop(stop: &ExecutionStop) -> Completion {
    match stop {
        ExecutionStop::BudgetExhausted
        | ExecutionStop::Safepoint
        | ExecutionStop::PendingEvent { .. } => Completion::Ready,
        ExecutionStop::LoaderReturn { .. } => Completion::Exited,
        ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. } => {
            Completion::Faulted
        }
        _ => Completion::Waiting,
    }
}

fn recorded_stop(stop: &ExecutionStop) -> crate::RecordedStop {
    match stop {
        ExecutionStop::InterpretOne { .. } => crate::RecordedStop::InterpretOne,
        ExecutionStop::BudgetExhausted => crate::RecordedStop::BudgetExhausted,
        ExecutionStop::Safepoint => crate::RecordedStop::Safepoint,
        ExecutionStop::PendingEvent { .. } => crate::RecordedStop::PendingEvent,
        ExecutionStop::Scheduled { .. } => crate::RecordedStop::Scheduled,
        ExecutionStop::ArchitecturalException { .. } => crate::RecordedStop::ArchitecturalException,
        ExecutionStop::SupervisorCall { .. } => crate::RecordedStop::SupervisorCall,
        ExecutionStop::DataFault { .. } => crate::RecordedStop::DataFault,
        ExecutionStop::LoaderReturn { .. } => crate::RecordedStop::LoaderReturn,
        ExecutionStop::FetchFault { .. } => crate::RecordedStop::FetchFault,
        ExecutionStop::UnsupportedSemantics { .. } => crate::RecordedStop::UnsupportedSemantics,
        ExecutionStop::ProfileDisabled { .. } => crate::RecordedStop::ProfileDisabled,
        ExecutionStop::UnallocatedEncoding { .. } => crate::RecordedStop::UnallocatedEncoding,
    }
}

#[cfg(test)]
mod tests;
