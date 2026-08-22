use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerDecision {
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
    Terminated(GuestThreadId),
    Migrated {
        thread: GuestThreadId,
        effect: MigrationEffect,
    },
    PriorityChanged(GuestThreadId),
    ActivityChanged(GuestThreadId),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReadyKey {
    priority: i32,
    sequence: SchedulerSequence,
    thread: GuestThreadId,
}

#[derive(Clone, Debug)]
struct ScheduledThread {
    config: ScheduledThreadConfig,
    lifecycle: ThreadLifecycle,
    ready_key: Option<ReadyKey>,
    active_lease: Option<Lease>,
    active_wait: Option<WakeToken>,
    last_vcpu: Option<VirtualCpuId>,
}

#[derive(Clone, Debug)]
struct VirtualCpuSlot {
    lease: Option<Lease>,
}

#[derive(Debug)]
pub struct SchedulerState {
    profile: MachineSchedulerProfile,
    threads: BTreeMap<GuestThreadId, ScheduledThread>,
    ready: BTreeSet<ReadyKey>,
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
            .map(|descriptor| (descriptor.id(), VirtualCpuSlot { lease: None }))
            .collect();
        Self {
            profile,
            threads: BTreeMap::new(),
            ready: BTreeSet::new(),
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
        match command {
            SchedulerCommand::Register(config) => self.register(config),
            SchedulerCommand::Unregister(thread) => self.unregister(thread),
            SchedulerCommand::MakeReady(thread) => self.make_ready(thread),
            SchedulerCommand::SelectNext => self.select_next(),
            SchedulerCommand::Select(vcpu) => self.select(vcpu),
            SchedulerCommand::Complete { lease, outcome } => self.complete(lease, outcome),
            SchedulerCommand::RegisterWait { thread, readiness } => {
                self.register_wait(thread, readiness)
            }
            SchedulerCommand::Wake(token) => self.finish_wait(token, false),
            SchedulerCommand::CancelWait(token) => self.finish_wait(token, true),
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
        }
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
                config,
                lifecycle: ThreadLifecycle::Created,
                ready_key: None,
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
        if let Some(key) = thread.ready_key {
            self.ready.remove(&key);
        }
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
        let sequence = self.allocate_sequence()?;
        let thread = self.threads.get_mut(&id).expect("thread was validated");
        thread.lifecycle = lifecycle;
        let key = ReadyKey {
            priority: thread.config.effective_priority,
            sequence,
            thread: id,
        };
        thread.ready_key = Some(key);
        let inserted = self.ready.insert(key);
        debug_assert!(inserted);
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
        let Some(key) = self.ready.iter().copied().find(|key| {
            self.threads
                .get(&key.thread)
                .is_some_and(|thread| thread.config.affinity.contains(vcpu))
        }) else {
            return Ok(SchedulerDecision::Selected(None));
        };
        let generation = self.allocate_lease_generation()?;
        let removed = self.ready.remove(&key);
        debug_assert!(removed);
        let thread = self
            .threads
            .get_mut(&key.thread)
            .expect("ready key is live");
        transition_thread(&mut thread.lifecycle, ThreadLifecycle::Running)
            .expect("only ready threads are queued");
        thread.ready_key = None;
        let lease = Lease {
            process: thread.config.process,
            thread: key.thread,
            vcpu,
            generation,
        };
        thread.active_lease = Some(lease);
        self.vcpus.get_mut(&vcpu).expect("vCPU was validated").lease = Some(lease);
        Ok(SchedulerDecision::Selected(Some(lease)))
    }

    fn select_next(&mut self) -> Result<SchedulerDecision, SchedulerError> {
        let Some(key) = self.ready.iter().copied().find(|key| {
            self.threads.get(&key.thread).is_some_and(|thread| {
                thread.config.affinity.iter().any(|vcpu| {
                    self.vcpus
                        .get(&vcpu)
                        .is_some_and(|slot| slot.lease.is_none())
                })
            })
        }) else {
            return Ok(SchedulerDecision::Selected(None));
        };
        let thread = self.threads.get(&key.thread).expect("ready key is live");
        let ideal = thread.config.ideal_vcpu.filter(|vcpu| {
            self.vcpus
                .get(vcpu)
                .is_some_and(|slot| slot.lease.is_none())
        });
        let local = thread.last_vcpu.filter(|vcpu| {
            thread.config.affinity.contains(*vcpu)
                && self
                    .vcpus
                    .get(vcpu)
                    .is_some_and(|slot| slot.lease.is_none())
        });
        let target = ideal.or(local).or_else(|| {
            thread.config.affinity.iter().find(|vcpu| {
                self.vcpus
                    .get(vcpu)
                    .is_some_and(|slot| slot.lease.is_none())
            })
        });
        self.select(target.expect("a ready thread was proven to have an idle eligible vCPU"))
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
        let ready_sequence = matches!(outcome, Completion::Ready | Completion::Preempted)
            .then(|| self.allocate_sequence())
            .transpose()?;
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
            Completion::Waiting => {
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
        if let Some(sequence) = ready_sequence {
            self.enqueue_existing_with_sequence(lease.thread, sequence);
        }
        Ok(SchedulerDecision::Completed(lease.thread))
    }

    fn enqueue_existing_with_sequence(&mut self, id: GuestThreadId, sequence: SchedulerSequence) {
        let thread = self.threads.get_mut(&id).expect("thread exists");
        let key = ReadyKey {
            priority: thread.config.effective_priority,
            sequence,
            thread: id,
        };
        thread.ready_key = Some(key);
        let inserted = self.ready.insert(key);
        debug_assert!(inserted);
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
            let sequence = self.allocate_sequence()?;
            let thread = self.threads.get_mut(&id).expect("thread exists");
            transition_thread(&mut thread.lifecycle, ThreadLifecycle::Ready)
                .expect("a waiting thread can become ready");
            self.enqueue_existing_with_sequence(id, sequence);
            return Ok(SchedulerDecision::ReadyImmediately(id));
        }
        let token = WakeToken {
            thread: id,
            generation: self.allocate_wake_generation()?,
        };
        self.threads
            .get_mut(&id)
            .expect("thread exists")
            .active_wait = Some(token);
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
            .get_mut(&id)
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
        thread.config.ideal_vcpu = ideal_vcpu;
        thread.config.affinity = affinity;
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
        let thread = self
            .threads
            .get(&id)
            .ok_or(SchedulerError::UnknownThread(id))?;
        if thread.active_lease.is_some() {
            return Err(SchedulerError::ThreadLeased(id));
        }
        let old_key = thread.ready_key;
        if let Some(key) = old_key {
            self.ready.remove(&key);
        }
        let thread = self.threads.get_mut(&id).expect("thread was validated");
        thread.config.base_priority = priority;
        thread.config.effective_priority = priority;
        if let Some(old_key) = old_key {
            let key = ReadyKey {
                priority,
                ..old_key
            };
            thread.ready_key = Some(key);
            let inserted = self.ready.insert(key);
            debug_assert!(inserted);
        }
        Ok(SchedulerDecision::PriorityChanged(id))
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
        if thread.active_lease.is_some() {
            return Err(SchedulerError::ThreadLeased(id));
        }
        let old_key = thread.ready_key;
        if let Some(key) = old_key {
            self.ready.remove(&key);
        }
        let thread = self.threads.get_mut(&id).expect("thread was validated");
        thread.config.effective_priority = priority;
        if let Some(old_key) = old_key {
            let key = ReadyKey {
                priority,
                ..old_key
            };
            thread.ready_key = Some(key);
            let inserted = self.ready.insert(key);
            debug_assert!(inserted);
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
        let old_key = thread.ready_key;
        if paused {
            if let Some(key) = old_key {
                self.ready.remove(&key);
            }
            let thread = self.threads.get_mut(&id).expect("thread was validated");
            thread.ready_key = None;
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
        let ready_key = thread.ready_key;
        if let Some(key) = ready_key {
            self.ready.remove(&key);
        }
        let thread = self.threads.get_mut(&id).expect("thread was validated");
        thread.ready_key = None;
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
            affinity: scheduler.profile().all_cores(),
        }
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
    fn sequence_exhaustion_leaves_completion_and_wait_transitions_uncommitted() {
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
            Err(SchedulerError::SequenceExhausted)
        );
        assert_eq!(scheduler.lease_for_vcpu(lease.vcpu), Some(lease));
        assert_eq!(
            scheduler.thread(lease.thread).unwrap().lifecycle,
            ThreadLifecycle::Running
        );
    }
}
