use std::collections::{BTreeMap, BTreeSet};
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

mod identity;
mod thread;
mod vcpu;
mod wait;

use identity::{GuestThreadIdAllocator, ProcessIdAllocator};
use vcpu::RuntimeVcpuSlot;

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
    in_flight: Option<Lease>,
    vcpu_slots: BTreeMap<VirtualCpuId, RuntimeVcpuSlot>,
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
        let vcpu_slots = profile
            .vcpus()
            .iter()
            .map(|descriptor| (descriptor.id(), RuntimeVcpuSlot::default()))
            .collect();
        Self {
            scheduler: SchedulerState::new(profile),
            processes: BTreeMap::new(),
            process_ids: ProcessIdAllocator::new(),
            in_flight: None,
            vcpu_slots,
            thread_ids: GuestThreadIdAllocator::new(),
            inbox: ExternalEventInbox::bounded(1_024)
                .expect("the default external event capacity is non-zero"),
            host_stop_requested: false,
            virtual_clock,
            deadlines: BTreeMap::new(),
            next_deadline_sequence: 1,
            active_waits: BTreeMap::new(),
            priority_donations: BTreeSet::new(),
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

    /// Executes at most one deterministic slice and returns its scheduler lease.
    pub fn run_next(
        &mut self,
        instruction_budget: u64,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        if let Some(lease) = self.in_flight {
            return Err(CoordinatorError::InFlightLease(lease));
        }
        let SchedulerDecision::Selected(lease) =
            self.scheduler.apply(SchedulerCommand::SelectNext)?
        else {
            unreachable!("select commands always produce a selected decision")
        };
        let lease = match lease {
            Some(lease) => lease,
            None => {
                if !self.fast_forward_to_next_deadline()? {
                    return Ok(None);
                }
                let SchedulerDecision::Selected(lease) =
                    self.scheduler.apply(SchedulerCommand::SelectNext)?
                else {
                    unreachable!("select commands always produce a selected decision")
                };
                let Some(lease) = lease else {
                    return Ok(None);
                };
                lease
            }
        };
        self.in_flight = Some(lease);
        let slot = self
            .vcpu_slots
            .get_mut(&lease.vcpu)
            .expect("scheduler leases only configured vCPUs");
        slot.begin(lease);
        let result = match self.processes.get_mut(&lease.process) {
            Some(process) => process.run_thread(lease.thread, lease.vcpu, instruction_budget),
            None => {
                self.scheduler.apply(SchedulerCommand::Complete {
                    lease,
                    outcome: Completion::Faulted,
                })?;
                self.in_flight = None;
                self.vcpu_slots
                    .get_mut(&lease.vcpu)
                    .expect("leased vCPU slot remains configured")
                    .finish(lease);
                return Err(CoordinatorError::UnknownProcess(lease.process));
            }
        };
        let completion = match &result {
            Ok(report) => completion_for_stop(&report.stop),
            Err(_) => Completion::Faulted,
        };
        let completion_result = self.scheduler.apply(SchedulerCommand::Complete {
            lease,
            outcome: completion,
        });
        self.in_flight = None;
        self.vcpu_slots
            .get_mut(&lease.vcpu)
            .expect("leased vCPU slot remains configured")
            .finish(lease);
        completion_result?;
        result
            .map(|report| Some(CoordinatorExecution { lease, report }))
            .map_err(|error| CoordinatorError::Execution {
                process: lease.process,
                thread: lease.thread,
                error,
            })
    }

    /// Synchronizes a compatibility API which resumed one waiting thread.
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
    pub worker_results: usize,
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

#[cfg(test)]
mod tests;
