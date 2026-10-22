use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub use crate::identity::LeaseGeneration;
use crate::{
    CoreSet, GuestThreadId, MachineSchedulerProfile, ProcessId, SchedulerSequence, ThreadLifecycle,
    VirtualCpuId, transition_thread,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledThreadConfig {
    pub process: ProcessId,
    pub thread: GuestThreadId,
    pub base_priority: i32,
    pub effective_priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledThreadView {
    pub process: ProcessId,
    pub thread: GuestThreadId,
    pub lifecycle: ThreadLifecycle,
    pub base_priority: i32,
    pub effective_priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
    pub last_vcpu: Option<VirtualCpuId>,
    pub paused: bool,
    pub active_wait: Option<WakeToken>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Lease {
    pub process: ProcessId,
    pub thread: GuestThreadId,
    pub vcpu: VirtualCpuId,
    pub generation: LeaseGeneration,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WakeToken {
    pub thread: GuestThreadId,
    pub generation: crate::WakeGeneration,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Readiness {
    Ready,
    Pending,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Completion {
    Ready,
    Preempted,
    Waiting,
    /// Kernel entry; runnable membership survives until an actual wait.
    KernelCall,
    Exited,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MigrationEffect {
    None,
    ClearOldLocalExclusive { old_vcpu: VirtualCpuId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerCommand {
    Register(ScheduledThreadConfig),
    Unregister(GuestThreadId),
    MakeReady(GuestThreadId),
    Yield {
        thread: GuestThreadId,
        migration: YieldMigration,
    },
    Rotate,
    SelectNext,
    Select(VirtualCpuId),
    Complete {
        lease: Lease,
        outcome: Completion,
    },
    RegisterWait {
        thread: GuestThreadId,
        readiness: Readiness,
    },
    Wake(WakeToken),
    CancelWait(WakeToken),
    ContinueWait(WakeToken),
    Terminate {
        thread: GuestThreadId,
        faulted: bool,
    },
    Migrate {
        thread: GuestThreadId,
        ideal_vcpu: Option<VirtualCpuId>,
        affinity: CoreSet,
    },
    SetPriority {
        thread: GuestThreadId,
        priority: i32,
    },
    SetEffectivePriority {
        thread: GuestThreadId,
        priority: i32,
    },
    SetActivity {
        thread: GuestThreadId,
        paused: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YieldMigration {
    None,
    Suggested,
    Any,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerDecision {
    Rotated,
    Yielded(GuestThreadId),
    Registered(GuestThreadId),
    Unregistered(GuestThreadId),
    Enqueued {
        thread: GuestThreadId,
        sequence: SchedulerSequence,
    },
    Selected(Option<Lease>),
    Completed(GuestThreadId),
    WaitRegistered(WakeToken),
    ReadyImmediately(GuestThreadId),
    Woken(GuestThreadId),
    WaitCancelled(GuestThreadId),
    WaitContinued(GuestThreadId),
    Terminated(GuestThreadId),
    Migrated {
        thread: GuestThreadId,
        effect: MigrationEffect,
    },
    PriorityChanged(GuestThreadId),
    ActivityChanged(GuestThreadId),
}

// Scheduled and suggested FIFO lists are independent: rotating a scheduled
// list does not move the thread in other cores' suggestion lists.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_priority_queue.h
#[derive(Clone, Debug, Default)]
struct PriorityQueue(BTreeMap<i32, VecDeque<GuestThreadId>>);

impl PriorityQueue {
    fn front(&self) -> Option<GuestThreadId> {
        self.0
            .first_key_value()
            .and_then(|(_, queue)| queue.front().copied())
    }

    fn at_priority(&self, priority: i32) -> Option<GuestThreadId> {
        self.0
            .get(&priority)
            .and_then(|queue| queue.front().copied())
    }

    fn iter(&self) -> impl Iterator<Item = GuestThreadId> + '_ {
        self.0.values().flat_map(|queue| queue.iter().copied())
    }

    fn insert(&mut self, priority: i32, thread: GuestThreadId, front: bool) {
        let queue = self.0.entry(priority).or_default();
        debug_assert!(!queue.contains(&thread));
        if front {
            queue.push_front(thread);
        } else {
            queue.push_back(thread);
        }
    }

    fn remove(&mut self, priority: i32, thread: GuestThreadId) {
        let queue = self.0.get_mut(&priority).expect("queued priority");
        let index = queue
            .iter()
            .position(|id| *id == thread)
            .expect("queued thread");
        queue.remove(index);
        if queue.is_empty() {
            self.0.remove(&priority);
        }
    }

    fn move_back(&mut self, priority: i32, thread: GuestThreadId) {
        let queue = self.0.get_mut(&priority).expect("queued priority");
        let index = queue
            .iter()
            .position(|id| *id == thread)
            .expect("queued thread");
        queue.remove(index);
        queue.push_back(thread);
    }
}

#[derive(Clone, Debug)]
struct ScheduledThread {
    config: ScheduledThreadConfig,
    lifecycle: ThreadLifecycle,
    ready_sequence: Option<SchedulerSequence>,
    active_core: Option<VirtualCpuId>,
    last_scheduled: u64,
    active_lease: Option<Lease>,
    active_wait: Option<WakeToken>,
    last_vcpu: Option<VirtualCpuId>,
}

#[derive(Clone, Debug)]
struct VirtualCpuSlot {
    lease: Option<Lease>,
    current: Option<GuestThreadId>,
    scheduled: PriorityQueue,
    suggested: PriorityQueue,
}

#[derive(Debug)]
pub struct SchedulerState {
    profile: MachineSchedulerProfile,
    threads: BTreeMap<GuestThreadId, ScheduledThread>,
    needs_rebalance: bool,
    scheduling_tick: u64,
    vcpus: BTreeMap<VirtualCpuId, VirtualCpuSlot>,
    next_sequence: u64,
    next_lease_generation: u64,
    next_wake_generation: u64,
}

impl SchedulerState {
    #[must_use]
    pub fn new(profile: MachineSchedulerProfile) -> Self {
        let vcpus = profile
            .vcpus()
            .iter()
            .map(|descriptor| {
                (
                    descriptor.id(),
                    VirtualCpuSlot {
                        lease: None,
                        current: None,
                        scheduled: PriorityQueue::default(),
                        suggested: PriorityQueue::default(),
                    },
                )
            })
            .collect();
        Self {
            profile,
            threads: BTreeMap::new(),
            needs_rebalance: false,
            scheduling_tick: 0,
            vcpus,
            next_sequence: 1,
            next_lease_generation: 1,
            next_wake_generation: 1,
        }
    }

    #[must_use]
    pub const fn profile(&self) -> &MachineSchedulerProfile {
        &self.profile
    }

    pub fn apply(
        &mut self,
        command: SchedulerCommand,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let decision = match command {
            SchedulerCommand::Register(config) => self.register(config),
            SchedulerCommand::Unregister(thread) => self.unregister(thread),
            SchedulerCommand::MakeReady(thread) => self.make_ready(thread),
            SchedulerCommand::Yield { thread, migration } => self.yield_thread(thread, migration),
            SchedulerCommand::Rotate => {
                for index in 0..self.profile.vcpus().len() {
                    let descriptor = &self.profile.vcpus()[index];
                    if let Some(priority) = descriptor.rotation_priority() {
                        self.rotate(descriptor.id(), priority);
                    }
                }
                self.needs_rebalance = true;
                Ok(SchedulerDecision::Rotated)
            }
            SchedulerCommand::SelectNext => self.select_next(),
            SchedulerCommand::Select(vcpu) => self.select(vcpu),
            SchedulerCommand::Complete { lease, outcome } => self.complete(lease, outcome),
            SchedulerCommand::RegisterWait { thread, readiness } => {
                self.register_wait(thread, readiness)
            }
            SchedulerCommand::Wake(token) => self.finish_wait(token, false),
            SchedulerCommand::CancelWait(token) => self.finish_wait(token, true),
            SchedulerCommand::ContinueWait(token) => {
                let thread = self
                    .threads
                    .get_mut(&token.thread)
                    .ok_or(SchedulerError::UnknownThread(token.thread))?;
                if thread.active_wait != Some(token) {
                    return Err(SchedulerError::StaleWake(token));
                }
                thread.active_wait = None;
                Ok(SchedulerDecision::WaitContinued(token.thread))
            }
            SchedulerCommand::Terminate { thread, faulted } => self.terminate(thread, faulted),
            SchedulerCommand::Migrate {
                thread,
                ideal_vcpu,
                affinity,
            } => self.migrate(thread, ideal_vcpu, affinity),
            SchedulerCommand::SetPriority { thread, priority } => {
                self.set_priority(thread, priority)
            }
            SchedulerCommand::SetEffectivePriority { thread, priority } => {
                self.set_effective_priority(thread, priority)
            }
            SchedulerCommand::SetActivity { thread, paused } => self.set_activity(thread, paused),
        };
        if decision.is_ok() {
            self.rebalance();
        }
        decision
    }

    pub fn thread(&self, id: GuestThreadId) -> Option<ScheduledThreadView> {
        self.threads.get(&id).map(|thread| ScheduledThreadView {
            process: thread.config.process,
            thread: thread.config.thread,
            lifecycle: thread.lifecycle,
            base_priority: thread.config.base_priority,
            effective_priority: thread.config.effective_priority,
            ideal_vcpu: thread.config.ideal_vcpu,
            affinity: thread.config.affinity.clone(),
            last_vcpu: thread.last_vcpu,
            paused: thread.lifecycle == ThreadLifecycle::Suspended,
            active_wait: thread.active_wait,
        })
    }

    #[must_use]
    pub fn lease_for_vcpu(&self, id: VirtualCpuId) -> Option<Lease> {
        self.vcpus.get(&id).and_then(|slot| slot.lease)
    }

    pub fn idle_vcpus(&self) -> impl Iterator<Item = VirtualCpuId> + '_ {
        self.vcpus
            .iter()
            .filter_map(|(id, slot)| slot.lease.is_none().then_some(*id))
    }

    pub fn active_leases(&self) -> impl Iterator<Item = Lease> + '_ {
        self.vcpus.values().filter_map(|slot| slot.lease)
    }

    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    #[must_use]
    pub fn active_wait_count(&self) -> usize {
        self.threads
            .values()
            .filter(|thread| thread.active_wait.is_some())
            .count()
    }

    pub fn active_waits(&self) -> impl Iterator<Item = WakeToken> + '_ {
        self.threads
            .values()
            .filter_map(|thread| thread.active_wait)
    }

    fn register(
        &mut self,
        config: ScheduledThreadConfig,
    ) -> Result<SchedulerDecision, SchedulerError> {
        if self.threads.contains_key(&config.thread) {
            return Err(SchedulerError::DuplicateThread(config.thread));
        }
        self.validate_policy(&config)?;
        let id = config.thread;
        self.threads.insert(
            id,
            ScheduledThread {
                lifecycle: ThreadLifecycle::Created,
                ready_sequence: None,
                active_core: config.ideal_vcpu.or_else(|| config.affinity.iter().next()),
                last_scheduled: 0,
                config,
                active_lease: None,
                active_wait: None,
                last_vcpu: None,
            },
        );
        Ok(SchedulerDecision::Registered(id))
    }

    fn unregister(&mut self, id: GuestThreadId) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.active_lease.is_some() {
            return Err(SchedulerError::ThreadLeased(id));
        }
        self.remove_queued(id);
        self.threads.remove(&id);
        Ok(SchedulerDecision::Unregistered(id))
    }

    fn make_ready(&mut self, id: GuestThreadId) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        let mut lifecycle = thread.lifecycle;
        transition_thread(&mut lifecycle, ThreadLifecycle::Ready).map_err(|_| {
            SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            }
        })?;
        if let Some(sequence) = thread.ready_sequence {
            self.threads.get_mut(&id).unwrap().lifecycle = lifecycle;
            return Ok(SchedulerDecision::Enqueued {
                thread: id,
                sequence,
            });
        }
        let sequence = self.allocate_sequence()?;
        self.remove_queued(id);
        self.threads.get_mut(&id).unwrap().lifecycle = lifecycle;
        self.enqueue_existing_with_sequence(id, sequence);
        Ok(SchedulerDecision::Enqueued {
            thread: id,
            sequence,
        })
    }

    fn select(&mut self, vcpu: VirtualCpuId) -> Result<SchedulerDecision, SchedulerError> {
        let slot = self
            .vcpus
            .get(&vcpu)
            .ok_or(SchedulerError::UnknownVirtualCpu(vcpu))?;
        if slot.lease.is_some() {
            return Err(SchedulerError::VirtualCpuBusy(vcpu));
        }
        self.rebalance();
        let Some(id) = self.vcpus[&vcpu].scheduled.front() else {
            return Ok(SchedulerDecision::Selected(None));
        };
        let thread = &self.threads[&id];
        // A migrated register owner must return before its destination can run it.
        if thread.active_lease.is_some() || thread.lifecycle != ThreadLifecycle::Ready {
            return Ok(SchedulerDecision::Selected(None));
        }
        let generation = self.allocate_lease_generation()?;
        let slot = self.vcpus.get_mut(&vcpu).unwrap();
        let thread = self.threads.get_mut(&id).unwrap();
        transition_thread(&mut thread.lifecycle, ThreadLifecycle::Running).unwrap();
        let lease = Lease {
            process: thread.config.process,
            thread: id,
            vcpu,
            generation,
        };
        thread.active_lease = Some(lease);
        slot.lease = Some(lease);
        Ok(SchedulerDecision::Selected(Some(lease)))
    }

    fn select_next(&mut self) -> Result<SchedulerDecision, SchedulerError> {
        self.rebalance();
        let candidate = self
            .vcpus
            .iter()
            .filter(|(_, slot)| slot.lease.is_none())
            .filter_map(|(core, slot)| slot.scheduled.front().map(|id| (*core, &self.threads[&id])))
            .filter(|(_, thread)| {
                thread.lifecycle == ThreadLifecycle::Ready && thread.active_lease.is_none()
            })
            .min_by_key(|(core, thread)| {
                (
                    thread.config.effective_priority,
                    thread.ready_sequence,
                    *core,
                )
            })
            .map(|(core, _)| core);
        match candidate {
            Some(core) => self.select(core),
            None => Ok(SchedulerDecision::Selected(None)),
        }
    }

    fn complete(
        &mut self,
        lease: Lease,
        outcome: Completion,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&lease.thread)
            .ok_or(SchedulerError::UnknownThread(lease.thread))?;
        if thread.active_lease != Some(lease) {
            return Err(SchedulerError::StaleLease(lease));
        }
        if self.vcpus.get(&lease.vcpu).and_then(|slot| slot.lease) != Some(lease) {
            return Err(SchedulerError::StaleLease(lease));
        }
        self.vcpus
            .get_mut(&lease.vcpu)
            .expect("lease vCPU exists")
            .lease = None;
        let thread = self
            .threads
            .get_mut(&lease.thread)
            .expect("lease thread exists");
        thread.active_lease = None;
        thread.last_vcpu = Some(lease.vcpu);
        match outcome {
            Completion::Ready => {
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Ready)
                    .expect("running thread can become ready");
            }
            Completion::Preempted => {
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Preempted)
                    .expect("running thread can be preempted");
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Ready)
                    .expect("preempted thread can become ready");
            }
            Completion::Waiting | Completion::KernelCall => {
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Waiting)
                    .expect("running thread can wait");
            }
            Completion::Exited => {
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Terminating)
                    .expect("running thread can terminate");
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Exited)
                    .expect("terminating thread can exit");
            }
            Completion::Faulted => {
                transition_thread(&mut thread.lifecycle, ThreadLifecycle::Faulted)
                    .expect("running thread can fault");
            }
        }
        if matches!(
            outcome,
            Completion::Waiting | Completion::Exited | Completion::Faulted
        ) {
            self.remove_queued(lease.thread);
        }
        Ok(SchedulerDecision::Completed(lease.thread))
    }

    fn enqueue_existing_with_sequence(&mut self, id: GuestThreadId, sequence: SchedulerSequence) {
        self.threads.get_mut(&id).unwrap().ready_sequence = Some(sequence);
        self.insert_queued(id, false);
    }

    fn insert_queued(&mut self, id: GuestThreadId, front: bool) {
        let thread = &self.threads[&id];
        for core in thread.config.affinity.iter() {
            let slot = self.vcpus.get_mut(&core).unwrap();
            let queue = if Some(core) == thread.active_core {
                &mut slot.scheduled
            } else {
                &mut slot.suggested
            };
            queue.insert(
                thread.config.effective_priority,
                id,
                front && Some(core) == thread.active_core,
            );
        }
        self.needs_rebalance = true;
    }

    fn remove_queued(&mut self, id: GuestThreadId) {
        let thread = self.threads.get_mut(&id).unwrap();
        if thread.ready_sequence.take().is_none() {
            return;
        }
        for core in thread.config.affinity.iter() {
            let slot = self.vcpus.get_mut(&core).unwrap();
            let queue = if Some(core) == thread.active_core {
                &mut slot.scheduled
            } else {
                &mut slot.suggested
            };
            queue.remove(thread.config.effective_priority, id);
        }
        self.needs_rebalance = true;
    }

    fn change_core(&mut self, id: GuestThreadId, core: Option<VirtualCpuId>, front: bool) {
        let thread = self.threads.get_mut(&id).unwrap();
        let old = thread.active_core;
        if old == core {
            return;
        }
        thread.active_core = core;
        let priority = thread.config.effective_priority;
        if let Some(old) = old {
            let slot = self.vcpus.get_mut(&old).unwrap();
            slot.scheduled.remove(priority, id);
            slot.suggested.insert(priority, id, false);
        }
        if let Some(core) = core {
            let slot = self.vcpus.get_mut(&core).unwrap();
            slot.suggested.remove(priority, id);
            slot.scheduled.insert(priority, id, front);
        }
        self.needs_rebalance = true;
    }

    fn source_top(&self, id: GuestThreadId) -> Option<GuestThreadId> {
        self.threads[&id]
            .active_core
            .and_then(|core| self.vcpus[&core].scheduled.front())
    }

    fn migration_allowed(&self, top: Option<GuestThreadId>) -> bool {
        top.is_none_or(|id| {
            self.profile
                .migration_priority()
                .is_none_or(|limit| self.threads[&id].config.effective_priority >= limit)
        })
    }

    // Preserve each core's local front. An idle core takes a non-front suggestion
    // first; taking another core's front requires a replacement on that core.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_scheduler.cpp#L209-L315
    pub fn rebalance(&mut self) {
        if !self.needs_rebalance {
            return;
        }
        for index in 0..self.profile.vcpus().len() {
            let core = self.profile.vcpus()[index].id();
            if self.vcpus[&core].scheduled.front().is_some() {
                continue;
            }
            let mut blocked = false;
            let candidate = self.vcpus[&core].suggested.iter().find(|id| {
                let top = self.source_top(*id);
                if top == Some(*id) {
                    return false;
                }
                blocked = !self.migration_allowed(top);
                true
            });
            if let Some(id) = candidate {
                if !blocked {
                    self.change_core(id, Some(core), false);
                }
                continue;
            }
            let candidate = self.vcpus[&core].suggested.iter().find(|id| {
                self.threads[id]
                    .active_core
                    .is_some_and(|source| self.vcpus[&source].scheduled.iter().nth(1).is_some())
            });
            if let Some(id) = candidate {
                self.change_core(id, Some(core), false);
            }
        }
        self.scheduling_tick += 1;
        for slot in self.vcpus.values_mut() {
            let next = slot.scheduled.front();
            if slot.current != next {
                if let Some(previous) = slot.current.and_then(|id| self.threads.get_mut(&id)) {
                    previous.last_scheduled = self.scheduling_tick;
                }
                slot.current = next;
            }
        }
        self.needs_rebalance = false;
    }

    /// Register owners whose core now selects a different thread must be interrupted.
    pub fn preemptions(&self) -> impl Iterator<Item = Lease> + '_ {
        self.vcpus.values().filter_map(|slot| {
            slot.lease
                .filter(|lease| slot.scheduled.front() != Some(lease.thread))
        })
    }

    fn migrate_suggestion(
        &mut self,
        core: VirtualCpuId,
        priority: i32,
        next: Option<GuestThreadId>,
        previous: Option<GuestThreadId>,
    ) {
        let mut candidate = None;
        for id in self.vcpus[&core].suggested.iter() {
            let top = self.source_top(id);
            if top == Some(id) {
                continue;
            }
            let thread = &self.threads[&id];
            if thread.config.effective_priority > priority
                || thread.config.effective_priority == priority
                    && next != previous
                    && next.is_some_and(|next| {
                        self.threads[&next].last_scheduled < thread.last_scheduled
                    })
            {
                break;
            }
            if self.migration_allowed(top) {
                candidate = Some(id);
                break;
            }
        }
        if let Some(id) = candidate {
            self.change_core(id, Some(core), true);
        }
    }

    fn rotate(&mut self, core: VirtualCpuId, priority: i32) {
        let previous = self.vcpus[&core].scheduled.at_priority(priority);
        if let Some(id) = previous {
            self.vcpus
                .get_mut(&core)
                .unwrap()
                .scheduled
                .move_back(priority, id);
        }
        let next = self.vcpus[&core].scheduled.at_priority(priority);
        let mut candidate = None;
        if let Some(queue) = self.vcpus[&core].suggested.0.get(&priority) {
            for id in queue.iter().copied() {
                let top = self.source_top(id);
                if top == Some(id) {
                    continue;
                }
                if next != previous
                    && next.is_some_and(|next| {
                        self.threads[&next].last_scheduled < self.threads[&id].last_scheduled
                    })
                {
                    break;
                }
                if self.migration_allowed(top) {
                    candidate = Some(id);
                    break;
                }
            }
        }
        if let Some(id) = candidate {
            self.change_core(id, Some(core), true);
        }
        if let Some(best) = self.vcpus[&core].scheduled.front() {
            let best_priority = self.threads[&best].config.effective_priority;
            if best_priority >= priority {
                let candidate = self.vcpus[&core]
                    .suggested
                    .iter()
                    .take_while(|id| self.threads[id].config.effective_priority < best_priority)
                    .find(|id| {
                        let top = self.source_top(*id);
                        top != Some(*id) && self.migration_allowed(top)
                    });
                if let Some(id) = candidate {
                    self.change_core(id, Some(core), true);
                }
            }
        }
    }

    fn yield_thread(
        &mut self,
        id: GuestThreadId,
        migration: YieldMigration,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.active_lease.is_some() || thread.ready_sequence.is_none() {
            return Err(SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            });
        }
        let core = thread.active_core;
        let priority = thread.config.effective_priority;
        if thread.lifecycle == ThreadLifecycle::Waiting {
            self.threads.get_mut(&id).unwrap().lifecycle = ThreadLifecycle::Ready;
        }
        if let Some(core) = core {
            match migration {
                YieldMigration::None | YieldMigration::Suggested => {
                    self.vcpus
                        .get_mut(&core)
                        .unwrap()
                        .scheduled
                        .move_back(priority, id);
                    if migration == YieldMigration::Suggested {
                        let next = self.vcpus[&core].scheduled.at_priority(priority);
                        self.migrate_suggestion(core, priority, next, Some(id));
                    }
                }
                YieldMigration::Any => {
                    self.change_core(id, None, false);
                    if self.vcpus[&core].scheduled.front().is_none() {
                        let candidate = self.vcpus[&core]
                            .suggested
                            .iter()
                            .find(|id| self.source_top(*id) != Some(*id));
                        if let Some(id) = candidate
                            && self.migration_allowed(self.source_top(id))
                        {
                            self.change_core(id, Some(core), false);
                        }
                    }
                }
            }
        }
        self.needs_rebalance = true;
        Ok(SchedulerDecision::Yielded(id))
    }

    fn register_wait(
        &mut self,
        id: GuestThreadId,
        readiness: Readiness,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.lifecycle != ThreadLifecycle::Waiting {
            return Err(SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            });
        }
        if thread.active_wait.is_some() {
            return Err(SchedulerError::DuplicateWait(id));
        }
        if readiness == Readiness::Ready {
            self.make_ready(id)?;
            return Ok(SchedulerDecision::ReadyImmediately(id));
        }
        let token = WakeToken {
            thread: id,
            generation: self.allocate_wake_generation()?,
        };
        self.remove_queued(id);
        let thread = self.threads.get_mut(&id).expect("thread exists");
        thread.active_wait = Some(token);
        Ok(SchedulerDecision::WaitRegistered(token))
    }

    fn finish_wait(
        &mut self,
        token: WakeToken,
        cancelled: bool,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&token.thread)
            .ok_or(SchedulerError::UnknownThread(token.thread))?;
        if thread.active_wait != Some(token) {
            return Err(SchedulerError::StaleWake(token));
        }
        let sequence = self.allocate_sequence()?;
        let thread = self
            .threads
            .get_mut(&token.thread)
            .expect("wait token thread exists");
        thread.active_wait = None;
        transition_thread(&mut thread.lifecycle, ThreadLifecycle::Ready)
            .expect("a registered waiter is waiting");
        self.enqueue_existing_with_sequence(token.thread, sequence);
        Ok(if cancelled {
            SchedulerDecision::WaitCancelled(token.thread)
        } else {
            SchedulerDecision::Woken(token.thread)
        })
    }

    fn migrate(
        &mut self,
        id: GuestThreadId,
        ideal_vcpu: Option<VirtualCpuId>,
        affinity: CoreSet,
    ) -> Result<SchedulerDecision, SchedulerError> {
        if let Some(ideal) = ideal_vcpu
            && !affinity.contains(ideal)
        {
            return Err(SchedulerError::IdealOutsideAffinity { thread: id, ideal });
        }
        if let Some(vcpu) = affinity.iter().find(|vcpu| !self.profile.contains(*vcpu)) {
            return Err(SchedulerError::UnknownVirtualCpu(vcpu));
        }
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.active_lease.is_some() {
            return Err(SchedulerError::ThreadLeased(id));
        }
        let effect = match thread.last_vcpu {
            Some(old_vcpu) if !affinity.contains(old_vcpu) => {
                MigrationEffect::ClearOldLocalExclusive { old_vcpu }
            }
            _ => MigrationEffect::None,
        };
        let sequence = thread.ready_sequence;
        self.remove_queued(id);
        let thread = self.threads.get_mut(&id).unwrap();
        if thread
            .active_core
            .is_some_and(|core| !affinity.contains(core))
        {
            thread.active_core = ideal_vcpu.or_else(|| affinity.iter().last());
        }
        thread.config.ideal_vcpu = ideal_vcpu;
        thread.config.affinity = affinity;
        if let Some(sequence) = sequence {
            self.enqueue_existing_with_sequence(id, sequence);
        }
        Ok(SchedulerDecision::Migrated { thread: id, effect })
    }

    fn set_priority(
        &mut self,
        id: GuestThreadId,
        priority: i32,
    ) -> Result<SchedulerDecision, SchedulerError> {
        if !self.profile.priorities().contains(priority) {
            return Err(SchedulerError::InvalidPriority(priority));
        }
        self.threads
            .get_mut(&id)
            .ok_or(SchedulerError::UnknownThread(id))?
            .config
            .base_priority = priority;
        self.set_effective_priority(id, priority)
    }

    fn set_effective_priority(
        &mut self,
        id: GuestThreadId,
        priority: i32,
    ) -> Result<SchedulerDecision, SchedulerError> {
        if !self.profile.priorities().contains(priority) {
            return Err(SchedulerError::InvalidPriority(priority));
        }
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.config.effective_priority == priority {
            return Ok(SchedulerDecision::PriorityChanged(id));
        }
        let sequence = thread.ready_sequence;
        let running = thread.active_lease.is_some()
            || thread.lifecycle == ThreadLifecycle::Waiting && sequence.is_some();
        self.remove_queued(id);
        let thread = self.threads.get_mut(&id).unwrap();
        thread.config.effective_priority = priority;
        thread.ready_sequence = sequence;
        if sequence.is_some() {
            self.insert_queued(id, running);
        }
        Ok(SchedulerDecision::PriorityChanged(id))
    }

    fn set_activity(
        &mut self,
        id: GuestThreadId,
        paused: bool,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.active_lease.is_some()
            || (paused && thread.lifecycle == ThreadLifecycle::Suspended)
            || (!paused && thread.lifecycle == ThreadLifecycle::Ready)
        {
            return Err(SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            });
        }
        if paused && thread.lifecycle != ThreadLifecycle::Ready {
            return Err(SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            });
        }
        if !paused && thread.lifecycle != ThreadLifecycle::Suspended {
            return Err(SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            });
        }
        if paused {
            self.remove_queued(id);
            let thread = self.threads.get_mut(&id).expect("thread was validated");
            transition_thread(&mut thread.lifecycle, ThreadLifecycle::Suspended)
                .expect("a validated ready thread can be suspended");
        } else {
            let sequence = self.allocate_sequence()?;
            transition_thread(
                &mut self
                    .threads
                    .get_mut(&id)
                    .expect("thread was validated")
                    .lifecycle,
                ThreadLifecycle::Ready,
            )
            .expect("a validated suspended thread can become ready");
            self.enqueue_existing_with_sequence(id, sequence);
        }
        Ok(SchedulerDecision::ActivityChanged(id))
    }

    fn terminate(
        &mut self,
        id: GuestThreadId,
        faulted: bool,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.active_lease.is_some() {
            return Err(SchedulerError::ThreadLeased(id));
        }
        if matches!(
            thread.lifecycle,
            ThreadLifecycle::Exited | ThreadLifecycle::Faulted
        ) {
            return Err(SchedulerError::InvalidThreadState {
                thread: id,
                state: thread.lifecycle,
            });
        }
        let mut lifecycle = thread.lifecycle;
        if faulted {
            transition_thread(&mut lifecycle, ThreadLifecycle::Faulted).map_err(|_| {
                SchedulerError::InvalidThreadState {
                    thread: id,
                    state: thread.lifecycle,
                }
            })?;
        } else {
            transition_thread(&mut lifecycle, ThreadLifecycle::Terminating).map_err(|_| {
                SchedulerError::InvalidThreadState {
                    thread: id,
                    state: thread.lifecycle,
                }
            })?;
            transition_thread(&mut lifecycle, ThreadLifecycle::Exited)
                .expect("a terminating thread can exit");
        }
        self.remove_queued(id);
        let thread = self.threads.get_mut(&id).expect("thread was validated");
        thread.active_wait = None;
        thread.lifecycle = lifecycle;
        Ok(SchedulerDecision::Terminated(id))
    }

    fn validate_policy(&self, config: &ScheduledThreadConfig) -> Result<(), SchedulerError> {
        if let Some(vcpu) = config
            .affinity
            .iter()
            .find(|vcpu| !self.profile.contains(*vcpu))
        {
            return Err(SchedulerError::UnknownVirtualCpu(vcpu));
        }
        if !self.profile.priorities().contains(config.base_priority) {
            return Err(SchedulerError::InvalidPriority(config.base_priority));
        }
        if !self
            .profile
            .priorities()
            .contains(config.effective_priority)
        {
            return Err(SchedulerError::InvalidPriority(config.effective_priority));
        }
        if let Some(ideal) = config.ideal_vcpu
            && !config.affinity.contains(ideal)
        {
            return Err(SchedulerError::IdealOutsideAffinity {
                thread: config.thread,
                ideal,
            });
        }
        Ok(())
    }

    fn allocate_sequence(&mut self) -> Result<SchedulerSequence, SchedulerError> {
        let value = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SchedulerError::SequenceExhausted)?;
        Ok(SchedulerSequence::new(value))
    }

    fn allocate_lease_generation(&mut self) -> Result<LeaseGeneration, SchedulerError> {
        let value = self.next_lease_generation;
        self.next_lease_generation = self
            .next_lease_generation
            .checked_add(1)
            .ok_or(SchedulerError::LeaseGenerationExhausted)?;
        Ok(LeaseGeneration::new(value))
    }

    fn allocate_wake_generation(&mut self) -> Result<crate::WakeGeneration, SchedulerError> {
        let value = self.next_wake_generation;
        self.next_wake_generation = self
            .next_wake_generation
            .checked_add(1)
            .ok_or(SchedulerError::WakeGenerationExhausted)?;
        Ok(crate::WakeGeneration::new(value))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    DuplicateThread(GuestThreadId),
    UnknownThread(GuestThreadId),
    UnknownVirtualCpu(VirtualCpuId),
    InvalidPriority(i32),
    IdealOutsideAffinity {
        thread: GuestThreadId,
        ideal: VirtualCpuId,
    },
    InvalidThreadState {
        thread: GuestThreadId,
        state: ThreadLifecycle,
    },
    ThreadLeased(GuestThreadId),
    VirtualCpuBusy(VirtualCpuId),
    StaleLease(Lease),
    DuplicateWait(GuestThreadId),
    StaleWake(WakeToken),
    SequenceExhausted,
    LeaseGenerationExhausted,
    WakeGenerationExhausted,
}

impl Display for SchedulerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scheduler rejected command: {self:?}")
    }
}

impl Error for SchedulerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PriorityRange, VirtualCpuDescriptor};

    fn profile() -> MachineSchedulerProfile {
        MachineSchedulerProfile::new(
            (0..3)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
            100,
        )
        .unwrap()
    }

    fn config(scheduler: &SchedulerState, id: u64, priority: i32) -> ScheduledThreadConfig {
        ScheduledThreadConfig {
            process: ProcessId::new(1),
            thread: GuestThreadId::new(id),
            base_priority: priority,
            effective_priority: priority,
            ideal_vcpu: Some(VirtualCpuId::new(0)),
            affinity: scheduler
                .profile()
                .core_set([VirtualCpuId::new(0)])
                .unwrap(),
        }
    }

    fn policy_scheduler() -> SchedulerState {
        SchedulerState::new(
            MachineSchedulerProfile::new(
                (0..4)
                    .map(|id| {
                        VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0)
                            .with_rotation_priority(if id == 3 { 63 } else { 59 })
                    })
                    .collect(),
                PriorityRange::new(0, 63).unwrap(),
                10_000,
            )
            .unwrap()
            .with_preemption(10_000_000, 2)
            .unwrap(),
        )
    }

    fn add(s: &mut SchedulerState, id: u64, priority: i32, core: u32, affinity: &[u32]) {
        let mut thread = config(s, id, priority);
        thread.ideal_vcpu = Some(VirtualCpuId::new(core));
        thread.affinity = s
            .profile()
            .core_set(affinity.iter().copied().map(VirtualCpuId::new))
            .unwrap();
        s.apply(SchedulerCommand::Register(thread)).unwrap();
        s.apply(SchedulerCommand::MakeReady(GuestThreadId::new(id)))
            .unwrap();
    }

    fn selected(s: &mut SchedulerState, core: u32) -> Lease {
        let SchedulerDecision::Selected(Some(lease)) = s
            .apply(SchedulerCommand::Select(VirtualCpuId::new(core)))
            .unwrap()
        else {
            panic!("expected thread on core {core}")
        };
        lease
    }

    #[test]
    fn internal_boundaries_retain_turn_and_core() {
        let mut s = policy_scheduler();
        add(&mut s, 1, 40, 2, &[0, 1, 2, 3]);
        for outcome in [
            Completion::Ready,
            Completion::Preempted,
            Completion::KernelCall,
        ] {
            let lease = selected(&mut s, 2);
            s.apply(SchedulerCommand::Complete { lease, outcome })
                .unwrap();
            if outcome == Completion::KernelCall {
                s.apply(SchedulerCommand::MakeReady(lease.thread)).unwrap();
            }
            for core in [0, 1, 3] {
                assert_eq!(
                    s.apply(SchedulerCommand::Select(VirtualCpuId::new(core)))
                        .unwrap(),
                    SchedulerDecision::Selected(None)
                );
            }
        }
        add(&mut s, 2, 40, 2, &[2]);
        let lease = selected(&mut s, 2);
        s.apply(SchedulerCommand::Complete {
            lease,
            outcome: Completion::Ready,
        })
        .unwrap();
        assert_eq!(selected(&mut s, 2).thread, lease.thread);
    }

    #[test]
    fn periodic_rotation_only_rotates_the_designated_priority_and_preempts_owners() {
        for core in 0..4 {
            for priority in [40, 59, 63] {
                let mut s = policy_scheduler();
                add(&mut s, 1, priority, core, &[core]);
                add(&mut s, 2, priority, core, &[core]);
                let lease = selected(&mut s, core);
                s.apply(SchedulerCommand::Rotate).unwrap();
                let rotates = priority == if core == 3 { 63 } else { 59 };
                assert_eq!(s.preemptions().count(), usize::from(rotates));
                s.apply(SchedulerCommand::Complete {
                    lease,
                    outcome: Completion::Ready,
                })
                .unwrap();
                assert_eq!(
                    selected(&mut s, core).thread,
                    GuestThreadId::new(if rotates { 2 } else { 1 })
                );
            }
        }
    }

    #[test]
    fn idle_core_migration_requires_a_donor_and_respects_the_priority_limit() {
        let mut s = policy_scheduler();
        add(&mut s, 1, 40, 2, &[0, 2]);
        add(&mut s, 2, 50, 2, &[2]);
        // Only the front is eligible for core 0; core 2 can replace it.
        assert_eq!(selected(&mut s, 0).thread, GuestThreadId::new(1));
        assert_eq!(selected(&mut s, 2).thread, GuestThreadId::new(2));
        for priority in [1, 2] {
            let mut s = policy_scheduler();
            add(&mut s, 1, priority, 2, &[2]);
            add(&mut s, 2, 40, 2, &[0, 2]);
            s.rebalance();
            assert_eq!(
                s.vcpus[&VirtualCpuId::new(0)].scheduled.front().is_some(),
                priority == 2
            );
        }
    }

    #[test]
    fn migrated_running_owner_must_return_before_destination_dispatch() {
        let mut s = policy_scheduler();
        add(&mut s, 1, 40, 2, &[0, 2]);
        let lease = selected(&mut s, 2);
        add(&mut s, 2, 50, 2, &[2]);
        assert_eq!(
            s.apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
                .unwrap(),
            SchedulerDecision::Selected(None)
        );
        assert_eq!(s.preemptions().collect::<Vec<_>>(), [lease]);
        s.apply(SchedulerCommand::Complete {
            lease,
            outcome: Completion::Ready,
        })
        .unwrap();
        assert_eq!(selected(&mut s, 0).thread, lease.thread);
        assert_eq!(selected(&mut s, 2).thread, GuestThreadId::new(2));
    }

    #[test]
    fn explicit_yields_rotate_or_relinquish_core_without_rotating_suggestions() {
        for migration in [
            YieldMigration::None,
            YieldMigration::Suggested,
            YieldMigration::Any,
        ] {
            let mut s = policy_scheduler();
            add(&mut s, 1, 40, 0, &[0]);
            add(&mut s, 2, 40, 0, &[0]);
            let lease = selected(&mut s, 0);
            s.apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::KernelCall,
            })
            .unwrap();
            s.apply(SchedulerCommand::Yield {
                thread: lease.thread,
                migration,
            })
            .unwrap();
            assert_eq!(selected(&mut s, 0).thread, GuestThreadId::new(2));
        }
        let mut s = policy_scheduler();
        add(&mut s, 1, 40, 0, &[0]);
        add(&mut s, 2, 30, 1, &[1]);
        add(&mut s, 3, 40, 1, &[0, 1]);
        let lease = selected(&mut s, 0);
        s.apply(SchedulerCommand::Complete {
            lease,
            outcome: Completion::KernelCall,
        })
        .unwrap();
        s.apply(SchedulerCommand::Yield {
            thread: lease.thread,
            migration: YieldMigration::Suggested,
        })
        .unwrap();
        assert_eq!(selected(&mut s, 0).thread, GuestThreadId::new(3));
    }

    #[test]
    fn resumed_turn_respects_priority_and_ends_at_a_real_wait() {
        let mut scheduler = SchedulerState::new(profile());
        let first = GuestThreadId::new(1);
        let peer = GuestThreadId::new(2);
        let urgent = GuestThreadId::new(3);
        for (id, priority) in [(1, 40), (2, 40), (3, 20)] {
            scheduler
                .apply(SchedulerCommand::Register(config(&scheduler, id, priority)))
                .unwrap();
        }
        for thread in [first, peer] {
            scheduler
                .apply(SchedulerCommand::MakeReady(thread))
                .unwrap();
        }
        let select = |scheduler: &mut SchedulerState| {
            let SchedulerDecision::Selected(Some(lease)) = scheduler
                .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
                .unwrap()
            else {
                panic!("expected runnable thread")
            };
            lease
        };
        let lease = select(&mut scheduler);
        assert_eq!(lease.thread, first);
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::KernelCall,
            })
            .unwrap();
        scheduler.apply(SchedulerCommand::MakeReady(first)).unwrap();
        scheduler
            .apply(SchedulerCommand::MakeReady(urgent))
            .unwrap();
        let lease = select(&mut scheduler);
        assert_eq!(
            lease.thread, urgent,
            "retaining a turn must not override priority"
        );
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Exited,
            })
            .unwrap();
        let lease = select(&mut scheduler);
        assert_eq!(lease.thread, first);
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::KernelCall,
            })
            .unwrap();
        let SchedulerDecision::WaitRegistered(token) = scheduler
            .apply(SchedulerCommand::RegisterWait {
                thread: first,
                readiness: Readiness::Pending,
            })
            .unwrap()
        else {
            panic!("expected wait token")
        };
        scheduler
            .apply(SchedulerCommand::ContinueWait(token))
            .unwrap();
        scheduler.apply(SchedulerCommand::MakeReady(first)).unwrap();
        assert_eq!(
            select(&mut scheduler).thread,
            peer,
            "a real wait ends the old turn"
        );
    }

    #[test]
    fn priority_changes_preserve_the_running_lease_and_reorder_the_next_dispatch() {
        let mut scheduler = SchedulerState::new(profile());
        for (id, priority) in [(1, 10), (2, 20)] {
            scheduler
                .apply(SchedulerCommand::Register(config(&scheduler, id, priority)))
                .unwrap();
            scheduler
                .apply(SchedulerCommand::MakeReady(GuestThreadId::new(id)))
                .unwrap();
        }
        let SchedulerDecision::Selected(Some(lease)) = scheduler
            .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
            .unwrap()
        else {
            unreachable!()
        };
        scheduler
            .apply(SchedulerCommand::SetPriority {
                thread: lease.thread,
                priority: 30,
            })
            .unwrap();
        scheduler
            .apply(SchedulerCommand::SetEffectivePriority {
                thread: lease.thread,
                priority: 25,
            })
            .unwrap();
        assert_eq!(scheduler.lease_for_vcpu(lease.vcpu), Some(lease));
        let view = scheduler.thread(lease.thread).unwrap();
        assert_eq!(view.lifecycle, ThreadLifecycle::Running);
        assert_eq!(view.base_priority, 30);
        assert_eq!(view.effective_priority, 25);
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Preempted,
            })
            .unwrap();
        let SchedulerDecision::Selected(Some(next)) = scheduler
            .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(next.thread, GuestThreadId::new(2));
    }

    #[test]
    fn priority_then_fifo_order_is_stable() {
        fn decisions() -> Vec<GuestThreadId> {
            let mut scheduler = SchedulerState::new(profile());
            for (id, priority) in [(1, 20), (2, 10), (3, 10), (4, 30)] {
                scheduler
                    .apply(SchedulerCommand::Register(config(&scheduler, id, priority)))
                    .unwrap();
                scheduler
                    .apply(SchedulerCommand::MakeReady(GuestThreadId::new(id)))
                    .unwrap();
            }
            let mut result = Vec::new();
            for _ in 0..4 {
                let SchedulerDecision::Selected(Some(lease)) = scheduler
                    .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
                    .unwrap()
                else {
                    unreachable!()
                };
                result.push(lease.thread);
                scheduler
                    .apply(SchedulerCommand::Complete {
                        lease,
                        outcome: Completion::Exited,
                    })
                    .unwrap();
            }
            result
        }
        assert_eq!(
            decisions(),
            [
                GuestThreadId::new(2),
                GuestThreadId::new(3),
                GuestThreadId::new(1),
                GuestThreadId::new(4)
            ]
        );
        assert_eq!(decisions(), decisions());
    }

    #[test]
    fn duplicate_commands_no_runnable_and_stale_leases_are_typed() {
        let mut scheduler = SchedulerState::new(profile());
        let config = config(&scheduler, 1, 10);
        scheduler
            .apply(SchedulerCommand::Register(config.clone()))
            .unwrap();
        assert_eq!(
            scheduler.apply(SchedulerCommand::Register(config)),
            Err(SchedulerError::DuplicateThread(GuestThreadId::new(1)))
        );
        assert_eq!(
            scheduler
                .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
                .unwrap(),
            SchedulerDecision::Selected(None)
        );
        scheduler
            .apply(SchedulerCommand::MakeReady(GuestThreadId::new(1)))
            .unwrap();
        assert!(matches!(
            scheduler.apply(SchedulerCommand::MakeReady(GuestThreadId::new(1))),
            Err(SchedulerError::InvalidThreadState { .. })
        ));
        let SchedulerDecision::Selected(Some(lease)) = scheduler
            .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
            .unwrap()
        else {
            unreachable!()
        };
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Ready,
            })
            .unwrap();
        assert_eq!(
            scheduler.apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Ready
            }),
            Err(SchedulerError::StaleLease(lease))
        );
    }

    #[test]
    fn placement_honors_ideal_affinity_and_single_owner_leases() {
        let mut scheduler = SchedulerState::new(profile());
        let affinity = scheduler
            .profile()
            .core_set([VirtualCpuId::new(1), VirtualCpuId::new(2)])
            .unwrap();
        let mut thread = config(&scheduler, 5, 10);
        thread.ideal_vcpu = Some(VirtualCpuId::new(2));
        thread.affinity = affinity;
        scheduler.apply(SchedulerCommand::Register(thread)).unwrap();
        scheduler
            .apply(SchedulerCommand::MakeReady(GuestThreadId::new(5)))
            .unwrap();
        let SchedulerDecision::Selected(Some(lease)) =
            scheduler.apply(SchedulerCommand::SelectNext).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(lease.vcpu, VirtualCpuId::new(2));
        assert_eq!(scheduler.lease_for_vcpu(VirtualCpuId::new(2)), Some(lease));
        assert_eq!(
            scheduler.apply(SchedulerCommand::Select(VirtualCpuId::new(2))),
            Err(SchedulerError::VirtualCpuBusy(VirtualCpuId::new(2)))
        );
        assert_eq!(
            scheduler.apply(SchedulerCommand::Migrate {
                thread: GuestThreadId::new(5),
                ideal_vcpu: Some(VirtualCpuId::new(1)),
                affinity: scheduler
                    .profile()
                    .core_set([VirtualCpuId::new(1)])
                    .unwrap(),
            }),
            Err(SchedulerError::ThreadLeased(GuestThreadId::new(5)))
        );
    }

    #[test]
    fn migration_explicitly_clears_old_vcpu_local_exclusive() {
        let mut scheduler = SchedulerState::new(profile());
        scheduler
            .apply(SchedulerCommand::Register(config(&scheduler, 8, 10)))
            .unwrap();
        scheduler
            .apply(SchedulerCommand::MakeReady(GuestThreadId::new(8)))
            .unwrap();
        let SchedulerDecision::Selected(Some(lease)) = scheduler
            .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
            .unwrap()
        else {
            unreachable!()
        };
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Waiting,
            })
            .unwrap();
        let affinity = scheduler
            .profile()
            .core_set([VirtualCpuId::new(1)])
            .unwrap();
        assert_eq!(
            scheduler
                .apply(SchedulerCommand::Migrate {
                    thread: GuestThreadId::new(8),
                    ideal_vcpu: Some(VirtualCpuId::new(1)),
                    affinity,
                })
                .unwrap(),
            SchedulerDecision::Migrated {
                thread: GuestThreadId::new(8),
                effect: MigrationEffect::ClearOldLocalExclusive {
                    old_vcpu: VirtualCpuId::new(0)
                }
            }
        );
    }

    #[test]
    fn continuing_a_wait_consumes_its_generation_without_scheduling_guest_code() {
        let (mut scheduler, thread) = waiting_scheduler();
        let SchedulerDecision::WaitRegistered(first) = scheduler
            .apply(SchedulerCommand::RegisterWait {
                thread,
                readiness: Readiness::Pending,
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            scheduler
                .apply(SchedulerCommand::ContinueWait(first))
                .unwrap(),
            SchedulerDecision::WaitContinued(thread)
        );
        assert_eq!(
            scheduler.thread(thread).unwrap().lifecycle,
            ThreadLifecycle::Waiting
        );
        assert_eq!(scheduler.thread(thread).unwrap().active_wait, None);
        assert!(matches!(
            scheduler.apply(SchedulerCommand::SelectNext).unwrap(),
            SchedulerDecision::Selected(None)
        ));
        let SchedulerDecision::WaitRegistered(second) = scheduler
            .apply(SchedulerCommand::RegisterWait {
                thread,
                readiness: Readiness::Pending,
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_ne!(first, second);
        assert!(matches!(
            scheduler.apply(SchedulerCommand::ContinueWait(first)),
            Err(SchedulerError::StaleWake(_))
        ));
        assert_eq!(scheduler.thread(thread).unwrap().active_wait, Some(second));
        assert_eq!(
            scheduler.apply(SchedulerCommand::Wake(second)).unwrap(),
            SchedulerDecision::Woken(thread)
        );
    }

    fn waiting_scheduler() -> (SchedulerState, GuestThreadId) {
        let mut scheduler = SchedulerState::new(profile());
        let thread = GuestThreadId::new(20);
        scheduler
            .apply(SchedulerCommand::Register(config(&scheduler, 20, 10)))
            .unwrap();
        scheduler
            .apply(SchedulerCommand::MakeReady(thread))
            .unwrap();
        let SchedulerDecision::Selected(Some(lease)) = scheduler
            .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
            .unwrap()
        else {
            unreachable!()
        };
        scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Waiting,
            })
            .unwrap();
        (scheduler, thread)
    }

    #[test]
    fn wait_registration_handles_signal_before_and_after_registration() {
        let (mut scheduler, thread) = waiting_scheduler();
        assert_eq!(
            scheduler
                .apply(SchedulerCommand::RegisterWait {
                    thread,
                    readiness: Readiness::Ready,
                })
                .unwrap(),
            SchedulerDecision::ReadyImmediately(thread)
        );

        let (mut scheduler, thread) = waiting_scheduler();
        let SchedulerDecision::WaitRegistered(token) = scheduler
            .apply(SchedulerCommand::RegisterWait {
                thread,
                readiness: Readiness::Pending,
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            scheduler.apply(SchedulerCommand::Wake(token)).unwrap(),
            SchedulerDecision::Woken(thread)
        );
        assert_eq!(
            scheduler.apply(SchedulerCommand::Wake(token)),
            Err(SchedulerError::StaleWake(token))
        );
    }

    #[test]
    fn timeout_signal_cancellation_and_close_races_have_one_winner() {
        for cancel_first in [false, true] {
            let (mut scheduler, thread) = waiting_scheduler();
            let SchedulerDecision::WaitRegistered(token) = scheduler
                .apply(SchedulerCommand::RegisterWait {
                    thread,
                    readiness: Readiness::Pending,
                })
                .unwrap()
            else {
                unreachable!()
            };
            let first = if cancel_first {
                SchedulerCommand::CancelWait(token)
            } else {
                SchedulerCommand::Wake(token)
            };
            scheduler.apply(first).unwrap();
            let second = if cancel_first {
                SchedulerCommand::Wake(token)
            } else {
                SchedulerCommand::CancelWait(token)
            };
            assert_eq!(
                scheduler.apply(second),
                Err(SchedulerError::StaleWake(token))
            );
        }
    }

    #[test]
    fn rejected_commands_do_not_consume_ready_queue_sequence() {
        let mut scheduler = SchedulerState::new(profile());
        assert_eq!(
            scheduler.apply(SchedulerCommand::MakeReady(GuestThreadId::new(99))),
            Err(SchedulerError::UnknownThread(GuestThreadId::new(99)))
        );
        scheduler
            .apply(SchedulerCommand::Register(config(&scheduler, 1, 10)))
            .unwrap();
        assert_eq!(
            scheduler
                .apply(SchedulerCommand::MakeReady(GuestThreadId::new(1)))
                .unwrap(),
            SchedulerDecision::Enqueued {
                thread: GuestThreadId::new(1),
                sequence: SchedulerSequence::new(1),
            }
        );
    }

    #[test]
    fn wait_sequence_exhaustion_is_atomic_and_internal_completion_needs_no_sequence() {
        let (mut scheduler, thread) = waiting_scheduler();
        let SchedulerDecision::WaitRegistered(token) = scheduler
            .apply(SchedulerCommand::RegisterWait {
                thread,
                readiness: Readiness::Pending,
            })
            .unwrap()
        else {
            unreachable!()
        };
        scheduler.next_sequence = u64::MAX;
        assert_eq!(
            scheduler.apply(SchedulerCommand::Wake(token)),
            Err(SchedulerError::SequenceExhausted)
        );
        assert_eq!(
            scheduler.threads.get(&thread).unwrap().active_wait,
            Some(token)
        );
        assert_eq!(
            scheduler.thread(thread).unwrap().lifecycle,
            ThreadLifecycle::Waiting
        );

        let mut scheduler = SchedulerState::new(profile());
        scheduler
            .apply(SchedulerCommand::Register(config(&scheduler, 2, 10)))
            .unwrap();
        scheduler
            .apply(SchedulerCommand::MakeReady(GuestThreadId::new(2)))
            .unwrap();
        let SchedulerDecision::Selected(Some(lease)) = scheduler
            .apply(SchedulerCommand::Select(VirtualCpuId::new(0)))
            .unwrap()
        else {
            unreachable!()
        };
        scheduler.next_sequence = u64::MAX;
        assert_eq!(
            scheduler.apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Ready,
            }),
            Ok(SchedulerDecision::Completed(lease.thread))
        );
        assert_eq!(scheduler.lease_for_vcpu(lease.vcpu), None);
        assert_eq!(
            scheduler.thread(lease.thread).unwrap().lifecycle,
            ThreadLifecycle::Ready
        );
    }
}
