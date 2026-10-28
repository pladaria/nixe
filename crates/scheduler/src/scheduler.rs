use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Mutex;

pub use crate::identity::LeaseGeneration;
use crate::{
    CoreSet, GuestThreadId, MachineSchedulerProfile, ProcessId, ThreadLifecycle, VirtualCpuId,
};
use wait_tree::{WaitNode, WaitTree};

// Zero denotes an empty link, keeping a link pair to eight bytes without raw
// pointers into a Vec that may move when a thread is created.
type Index = std::num::NonZeroU32;
fn slot_index(index: Index) -> usize {
    (index.get() - 1) as usize
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledThreadConfig {
    pub process: ProcessId,
    pub base_priority: i32,
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

/// Exclusive ownership of the architectural context until the worker returns it.
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
pub enum Completion {
    Ready,
    Waiting,
    /// Kernel entry retains queue membership until an operation actually blocks.
    KernelCall {
        interrupt_handled: bool,
    },
    Exited,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YieldMigration {
    None,
    Suggested,
    Any,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Selection {
    pub thread: Option<GuestThreadId>,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Links {
    previous: Option<Index>,
    next: Option<Index>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Ends {
    first: Option<Index>,
    last: Option<Index>,
}

// One link per thread/core, shared between that core's scheduled/suggested lists.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_priority_queue.h
#[derive(Debug)]
struct Queue {
    priorities: [Ends; 64],
    nonempty: u64,
    len: usize,
}

impl Default for Queue {
    fn default() -> Self {
        Self {
            priorities: [Ends::default(); 64],
            nonempty: 0,
            len: 0,
        }
    }
}

impl Queue {
    fn front(&self) -> Option<Index> {
        (self.nonempty != 0)
            .then(|| self.priorities[self.nonempty.trailing_zeros() as usize].first)
            .flatten()
    }

    fn next(&self, slots: &Slots, core: usize, id: Index) -> Option<Index> {
        slots[id].links[core].next.or_else(|| {
            let priority = slots[id].priority;
            let remaining =
                self.nonempty & !u64::MAX.checked_shr(63 - priority as u32).unwrap_or(0);
            (remaining != 0)
                .then(|| self.priorities[remaining.trailing_zeros() as usize].first)
                .flatten()
        })
    }

    fn push(&mut self, slots: &mut Slots, core: usize, id: Index, front: bool) {
        let priority = slots[id].priority as usize;
        let ends = &mut self.priorities[priority];
        let (previous, next) = if front {
            (None, ends.first)
        } else {
            (ends.last, None)
        };
        slots[id].links[core] = Links { previous, next };
        if let Some(previous) = previous {
            slots[previous].links[core].next = Some(id);
        } else {
            ends.first = Some(id);
        }
        if let Some(next) = next {
            slots[next].links[core].previous = Some(id);
        } else {
            ends.last = Some(id);
        }
        self.nonempty |= 1 << priority;
        self.len += 1;
    }

    fn remove(&mut self, slots: &mut Slots, core: usize, id: Index) {
        let priority = slots[id].priority as usize;
        let ends = &mut self.priorities[priority];
        let Links { previous, next } = slots[id].links[core];
        if let Some(previous) = previous {
            slots[previous].links[core].next = next;
        } else {
            ends.first = next;
        }
        if let Some(next) = next {
            slots[next].links[core].previous = previous;
        } else {
            ends.last = previous;
        }
        slots[id].links[core] = Links::default();
        if ends.first.is_none() {
            self.nonempty &= !(1 << priority);
        }
        self.len -= 1;
    }

    fn move_back(&mut self, slots: &mut Slots, core: usize, id: Index) -> Option<Index> {
        let priority = slots[id].priority as usize;
        if self.priorities[priority].last != Some(id) {
            self.remove(slots, core, id);
            self.push(slots, core, id, false);
        }
        self.priorities[priority].first
    }
}

/// Guest address keys and kernel lock keys have distinct pinning semantics.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LockKey {
    User(u64),
    ThreadProperty(GuestThreadId),
}

/// Ideal-core update resolved atomically with the target's affinity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdealCore {
    Any,
    Core(VirtualCpuId),
    Preserve,
}
impl LockKey {
    fn is_kernel(self) -> bool {
        !matches!(self, Self::User(_))
    }
}

#[derive(Debug)]
struct Thread {
    id: GuestThreadId,
    config: ScheduledThreadConfig,
    priority: i32,
    lifecycle: ThreadLifecycle,
    paused: bool,
    queued: bool,
    initialized: bool,
    active_core: Option<usize>,
    pinned_core: Option<usize>,
    links: Box<[Links]>,
    last_scheduled: u64,
    yield_generation: Option<u64>,
    lease: Option<Lease>,
    wait: Option<WakeToken>,
    last_vcpu: Option<VirtualCpuId>,
    waiting_on: Option<waits::Membership>,
    held: Option<Index>,
    kernel_waiters: usize,
    pin_waiters: Ends,
    wait_node: WaitNode,
    property_owner: Option<Index>,
    property_target: Option<Index>,
    kernel_waker: Option<std::task::Waker>,
}

impl Thread {
    fn suspended(&self) -> bool {
        self.paused && self.pinned_core.is_none() && self.kernel_waiters == 0
    }
}

#[derive(Debug)]
struct Slot {
    generation: u32,
    thread: Option<Thread>,
    next_free: Option<Index>,
}

#[derive(Debug, Default)]
struct Slots {
    entries: Vec<Slot>,
    free: Option<Index>,
    len: usize,
}

impl Slots {
    fn index(&self, id: GuestThreadId) -> Result<Index, SchedulerError> {
        let index = Index::new(id.get() as u32);
        index
            .filter(|&index| {
                self.entries.get(slot_index(index)).is_some_and(|slot| {
                    slot.generation == (id.get() >> 32) as u32 && slot.thread.is_some()
                })
            })
            .ok_or(SchedulerError::UnknownThread(id))
    }

    fn insert(&mut self, mut thread: Thread) -> Result<GuestThreadId, SchedulerError> {
        let index = if let Some(index) = self.free {
            self.free = self.entries[slot_index(index)].next_free;
            index
        } else {
            let index = u32::try_from(self.entries.len())
                .ok()
                .and_then(|index| index.checked_add(1))
                .and_then(Index::new)
                .ok_or(SchedulerError::ThreadIdExhausted)?;
            self.entries.push(Slot {
                generation: 0,
                thread: None,
                next_free: None,
            });
            index
        };
        let slot = &mut self.entries[slot_index(index)];
        let id = GuestThreadId::new((u64::from(slot.generation) << 32) | u64::from(index.get()));
        thread.id = id;
        slot.thread = Some(thread);
        self.len += 1;
        Ok(id)
    }

    fn remove(&mut self, index: Index) {
        let slot = &mut self.entries[slot_index(index)];
        slot.thread = None;
        // Retire a generation-exhausted slot rather than accepting stale IDs.
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            slot.next_free = self.free;
            self.free = Some(index);
        }
        self.len -= 1;
    }
}

impl std::ops::Index<Index> for Slots {
    type Output = Thread;
    fn index(&self, index: Index) -> &Thread {
        self.entries[slot_index(index)]
            .thread
            .as_ref()
            .expect("live queue member")
    }
}
impl std::ops::IndexMut<Index> for Slots {
    fn index_mut(&mut self, index: Index) -> &mut Thread {
        self.entries[slot_index(index)]
            .thread
            .as_mut()
            .expect("live queue member")
    }
}

#[derive(Debug, Default)]
struct Core {
    lease: Option<Lease>,
    selected: Option<Index>,
    generation: u64,
    scheduled: Queue,
    suggested: Queue,
}

#[derive(Debug)]
struct Process {
    generation: u64,
    threads: usize,
    pinned: Box<[Option<Index>]>,
    ordered: [WaitTree; 2],
}

#[derive(Debug)]
struct State {
    slots: Slots,
    cores: Box<[Core]>,
    processes: BTreeMap<ProcessId, Process>,
    needs_selection: bool,
    wake_mask: u64,
    clock: crate::VirtualClock,
    lease_generation: u64,
    wake_generation: u64,
    locks: Vec<waits::LockSlot>,
    free_lock: Option<Index>,
    deferred_wakers: Vec<(std::task::Waker, bool)>,
    #[cfg(feature = "performance-counters")]
    statistics: SchedulerStatistics,
}

/// Canonical scheduling metadata. Native registers and kernel continuations are
/// owned separately and keyed by the same generational thread ID. No external
/// locks, callbacks, guest-memory accesses or wake notifications run under this mutex.
#[derive(Debug)]
pub struct SchedulerState {
    profile: MachineSchedulerProfile,
    state: Mutex<State>,
}

impl SchedulerState {
    #[must_use]
    pub fn new(profile: MachineSchedulerProfile, clock: crate::VirtualClock) -> Self {
        let cores = (0..profile.vcpus().len())
            .map(|_| Core::default())
            .collect();
        Self {
            profile,
            state: Mutex::new(State {
                slots: Slots::default(),
                cores,
                processes: BTreeMap::new(),
                needs_selection: false,
                wake_mask: 0,
                clock,
                lease_generation: 0,
                wake_generation: 0,
                locks: Vec::new(),
                free_lock: None,
                deferred_wakers: Vec::new(),
                #[cfg(feature = "performance-counters")]
                statistics: SchedulerStatistics::default(),
            }),
        }
    }

    #[must_use]
    pub const fn profile(&self) -> &MachineSchedulerProfile {
        &self.profile
    }

    fn edit<T>(
        &self,
        operation: impl FnOnce(&mut State) -> Result<T, SchedulerError>,
    ) -> Result<T, SchedulerError> {
        #[cfg(feature = "performance-counters")]
        let waiting = std::time::Instant::now();
        let mut state = self.state.lock().expect("scheduler lock poisoned");
        #[cfg(feature = "performance-counters")]
        let acquired = std::time::Instant::now();
        let result = operation(&mut state);
        state.select_threads(&self.profile);
        #[cfg(feature = "performance-counters")]
        {
            state.statistics.operations += 1;
            state.statistics.lock_wait_ns += acquired.duration_since(waiting).as_nanos() as u64;
            state.statistics.lock_held_ns += acquired.elapsed().as_nanos() as u64;
        }
        let wakers = std::mem::take(&mut state.deferred_wakers);
        drop(state);
        for (waker, notify) in wakers {
            if notify {
                waker.wake();
            }
        }
        result
    }

    fn core(&self, id: VirtualCpuId) -> Result<usize, SchedulerError> {
        self.profile
            .vcpus()
            .binary_search_by_key(&id, |core| core.id())
            .map_err(|_| SchedulerError::UnknownVirtualCpu(id))
    }

    fn validate(
        &self,
        priority: i32,
        ideal: Option<VirtualCpuId>,
        affinity: CoreSet,
    ) -> Result<(), SchedulerError> {
        if !self.profile.priorities().contains(priority) {
            return Err(SchedulerError::InvalidPriority(priority));
        }
        if affinity.is_empty() {
            return Err(SchedulerError::EmptyAffinity);
        }
        for core in affinity.iter() {
            self.core(core)?;
        }
        if let Some(ideal) = ideal
            && !affinity.contains(ideal)
        {
            return Err(SchedulerError::IdealOutsideAffinity(Some(ideal)));
        }
        Ok(())
    }

    pub fn create_thread(
        &self,
        config: ScheduledThreadConfig,
    ) -> Result<GuestThreadId, SchedulerError> {
        self.validate(config.base_priority, config.ideal_vcpu, config.affinity)?;
        self.edit(|state| {
            let process = config.process;
            let priority = config.base_priority;
            let id = state.slots.insert(Thread {
                id: GuestThreadId::new(0),
                config,
                priority,
                lifecycle: ThreadLifecycle::Created,
                paused: false,
                queued: false,
                initialized: false,
                active_core: None,
                pinned_core: None,
                links: vec![Links::default(); state.cores.len()].into_boxed_slice(),
                last_scheduled: 0,
                yield_generation: None,
                lease: None,
                wait: None,
                last_vcpu: None,
                waiting_on: None,
                held: None,
                kernel_waiters: 0,
                pin_waiters: Ends::default(),
                wait_node: WaitNode::default(),
                property_owner: None,
                property_target: None,
                kernel_waker: None,
            })?;
            state
                .processes
                .entry(process)
                .or_insert_with(|| Process {
                    generation: 0,
                    threads: 0,
                    pinned: vec![None; state.cores.len()].into_boxed_slice(),
                    ordered: Default::default(),
                })
                .threads += 1;
            Ok(id)
        })
    }

    pub fn unregister(&self, id: GuestThreadId) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            if state.slots[index].lease.is_some() {
                return Err(SchedulerError::ThreadLeased(id));
            }
            state.cancel_property(index, &self.profile, true);
            while let Some(owner) = state.slots[index].property_owner {
                state.cancel_property(owner, &self.profile, true);
            }
            state.remove_queued(index, &self.profile);
            state.detach_donations(index, &self.profile);
            state.unpin(index, &self.profile);
            let process = state.slots[index].config.process;
            let owner = state.processes.get_mut(&process).unwrap();
            owner.threads -= 1;
            state.slots.remove(index);
            if state.processes[&process].threads == 0 {
                state.processes.remove(&process);
            }
            Ok(())
        })
    }

    pub fn thread(&self, id: GuestThreadId) -> Option<ScheduledThreadView> {
        let state = self.state.lock().expect("scheduler lock poisoned");
        let index = state.slots.index(id).ok()?;
        let thread = &state.slots[index];
        Some(ScheduledThreadView {
            process: thread.config.process,
            thread: id,
            lifecycle: if thread.suspended() && thread.lifecycle == ThreadLifecycle::Ready {
                ThreadLifecycle::Suspended
            } else {
                thread.lifecycle
            },
            base_priority: thread.config.base_priority,
            effective_priority: thread.priority,
            ideal_vcpu: thread.config.ideal_vcpu,
            affinity: thread.config.affinity,
            last_vcpu: thread.last_vcpu,
            paused: thread.paused,
            active_wait: thread.wait,
        })
    }

    pub fn make_ready(&self, id: GuestThreadId) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            let thread = &state.slots[index];
            if thread.lease.is_some()
                || thread.wait.is_some()
                || matches!(
                    thread.lifecycle,
                    ThreadLifecycle::Exited | ThreadLifecycle::Faulted
                )
            {
                return Err(state.invalid(index));
            }
            state.slots[index].lifecycle = ThreadLifecycle::Ready;
            if !state.slots[index].suspended() && !state.slots[index].queued {
                state.enqueue(index, &self.profile);
            }
            Ok(())
        })
    }

    pub fn claim(&self, vcpu: VirtualCpuId) -> Result<Option<Lease>, SchedulerError> {
        let core = self.core(vcpu)?;
        self.edit(|state| {
            if state.cores[core].lease.is_some() {
                return Err(SchedulerError::VirtualCpuBusy(vcpu));
            }
            let Some(index) = state.cores[core].selected else {
                return Ok(None);
            };
            let thread = &state.slots[index];
            if thread.lease.is_some()
                || thread.lifecycle != ThreadLifecycle::Ready
                || thread.suspended()
            {
                return Ok(None);
            }
            state.lease_generation = state
                .lease_generation
                .checked_add(1)
                .ok_or(SchedulerError::LeaseGenerationExhausted)?;
            let lease = Lease {
                process: thread.config.process,
                thread: thread.id,
                vcpu,
                generation: LeaseGeneration::new(state.lease_generation),
            };
            state.slots[index].lease = Some(lease);
            state.slots[index].lifecycle = ThreadLifecycle::Running;
            state.cores[core].lease = Some(lease);
            Ok(Some(lease))
        })
    }

    /// False keeps ownership until the core handles an interrupt racing an SVC.
    pub fn complete(&self, lease: Lease, outcome: Completion) -> Result<bool, SchedulerError> {
        let core = self.core(lease.vcpu)?;
        self.edit(|state| {
            let index = state.slots.index(lease.thread)?;
            if state.slots[index].lease != Some(lease) || state.cores[core].lease != Some(lease) {
                return Err(SchedulerError::StaleLease(lease));
            }
            // An SVC may beat its rescheduling interrupt out of native code.
            // Keep its core/context owned until that core protects SDK TLS.
            if matches!(
                outcome,
                Completion::KernelCall {
                    interrupt_handled: false
                }
            ) && state.cores[core].selected != Some(index)
            {
                return Ok(false);
            }
            state.cores[core].lease = None;
            state.slots[index].lease = None;
            state.slots[index].last_vcpu = Some(lease.vcpu);
            state.slots[index].lifecycle = match outcome {
                Completion::Ready => ThreadLifecycle::Ready,
                Completion::Waiting | Completion::KernelCall { .. } => ThreadLifecycle::Waiting,
                Completion::Exited => ThreadLifecycle::Exited,
                Completion::Faulted => ThreadLifecycle::Faulted,
            };
            state.notify_context_release(index, &self.profile);
            if matches!(
                outcome,
                Completion::Waiting | Completion::Exited | Completion::Faulted
            ) || state.slots[index].suspended()
            {
                state.remove_queued(index, &self.profile);
            }
            if matches!(outcome, Completion::Ready)
                && !state.slots[index].suspended()
                && !state.slots[index].queued
            {
                state.enqueue(index, &self.profile);
            }
            if matches!(outcome, Completion::Exited | Completion::Faulted) {
                state.cancel_property(index, &self.profile, true);
                state.detach_donations(index, &self.profile);
                state.unpin(index, &self.profile);
            }
            if let Some(destination) = state.slots[index].active_core
                && destination != core
                && state.cores[destination].selected == Some(index)
            {
                state.wake_mask |= 1 << self.profile.vcpus()[destination].id().get();
            }
            Ok(true)
        })
    }

    pub fn lease_for_vcpu(&self, core: VirtualCpuId) -> Option<Lease> {
        let index = self.core(core).ok()?;
        self.state.lock().expect("scheduler lock poisoned").cores[index].lease
    }

    pub fn active_leases(&self) -> impl Iterator<Item = Lease> + '_ {
        self.profile
            .vcpus()
            .iter()
            .filter_map(|core| self.lease_for_vcpu(core.id()))
    }

    pub fn idle_vcpus(&self) -> impl ExactSizeIterator<Item = VirtualCpuId> + use<> {
        let state = self.state.lock().expect("scheduler lock poisoned");
        let bits =
            state
                .cores
                .iter()
                .zip(self.profile.vcpus())
                .fold(0, |bits, (core, descriptor)| {
                    bits | if core.lease.is_none() {
                        1 << descriptor.id().get()
                    } else {
                        0
                    }
                });
        CoreSet { bits }.iter()
    }

    pub fn has_active_leases(&self) -> bool {
        self.state
            .lock()
            .expect("scheduler lock poisoned")
            .cores
            .iter()
            .any(|core| core.lease.is_some())
    }

    pub fn thread_count(&self) -> usize {
        self.state
            .lock()
            .expect("scheduler lock poisoned")
            .slots
            .len
    }

    pub fn active_waits(&self) -> impl Iterator<Item = WakeToken> + use<> {
        self.state
            .lock()
            .expect("scheduler lock poisoned")
            .slots
            .entries
            .iter()
            .filter_map(|slot| slot.thread.as_ref().and_then(|thread| thread.wait))
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn active_wait_count(&self) -> usize {
        self.state
            .lock()
            .expect("scheduler lock poisoned")
            .slots
            .entries
            .iter()
            .filter(|slot| {
                slot.thread
                    .as_ref()
                    .is_some_and(|thread| thread.wait.is_some())
            })
            .count()
    }

    pub fn selection(&self, core: VirtualCpuId) -> Result<Selection, SchedulerError> {
        let core = self.core(core)?;
        let state = self.state.lock().expect("scheduler lock poisoned");
        Ok(Selection {
            thread: state.cores[core]
                .selected
                .map(|index| state.slots[index].id),
            generation: state.cores[core].generation,
        })
    }

    /// Consume targeted notifications. Callers send native interrupts only after
    /// this function has released the scheduling lock.
    pub fn take_reschedule_requests(&self) -> CoreSet {
        let mut state = self.state.lock().expect("scheduler lock poisoned");
        CoreSet {
            bits: std::mem::take(&mut state.wake_mask),
        }
    }

    pub fn register_wait(&self, id: GuestThreadId) -> Result<WakeToken, SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            state.begin_wait(index, &self.profile)
        })
    }

    pub fn finish_wait(&self, token: WakeToken) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state
                .slots
                .index(token.thread)
                .map_err(|_| SchedulerError::StaleWake(token))?;
            if state.slots[index].wait != Some(token) {
                return Err(SchedulerError::StaleWake(token));
            }
            debug_assert!(
                state.slots[index].kernel_waker.is_none(),
                "kernel results must be installed before guest wake"
            );
            state.slots[index].wait = None;
            state.slots[index].lifecycle = ThreadLifecycle::Ready;
            if !state.slots[index].suspended() {
                state.enqueue(index, &self.profile);
            }
            Ok(())
        })
    }

    pub fn continue_wait(&self, token: WakeToken) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(token.thread)?;
            if state.slots[index].wait != Some(token) {
                return Err(SchedulerError::StaleWake(token));
            }
            state.slots[index].wait = None;
            Ok(())
        })
    }

    pub fn migrate(
        &self,
        id: GuestThreadId,
        ideal: IdealCore,
        affinity: CoreSet,
    ) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            // Resolve preservation with the target's current metadata under the
            // same lock as the affinity update.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L631-L640
            let ideal = match ideal {
                IdealCore::Any => None,
                IdealCore::Core(core) => Some(core),
                IdealCore::Preserve => Some(
                    state.slots[index]
                        .config
                        .ideal_vcpu
                        .ok_or(SchedulerError::IdealOutsideAffinity(None))?,
                ),
            };
            self.validate(self.profile.priorities().highest(), ideal, affinity)?;
            if state.slots[index].config.affinity == affinity
                && state.slots[index].config.ideal_vcpu == ideal
            {
                return Ok(());
            }
            if state.slots[index].pinned_core.is_some()
                || state.slots[index].config.affinity == affinity
            {
                state.slots[index].config.affinity = affinity;
                state.slots[index].config.ideal_vcpu = ideal;
                return Ok(());
            }
            let queued = state.slots[index].queued;
            if queued {
                state.remove_memberships(index, &self.profile);
            }
            state.slots[index].config.affinity = affinity;
            state.slots[index].config.ideal_vcpu = ideal;
            if state.slots[index]
                .active_core
                .is_some_and(|core| !affinity.contains(self.profile.vcpus()[core].id()))
            {
                state.slots[index].active_core = Some(state.affinity_core(index, &self.profile));
            }
            if queued {
                state.insert_memberships(index, &self.profile, false);
            }
            if queued {
                state.changed(index);
            }
            Ok(())
        })
    }

    pub fn set_activity(&self, id: GuestThreadId, paused: bool) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            if state.slots[index].paused == paused
                || matches!(
                    state.slots[index].lifecycle,
                    ThreadLifecycle::Created | ThreadLifecycle::Exited | ThreadLifecycle::Faulted
                )
            {
                return Err(state.invalid(index));
            }
            state.slots[index].paused = paused;
            state.update_suspension(index, &self.profile);
            Ok(())
        })
    }

    pub fn terminate(&self, id: GuestThreadId, faulted: bool) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            if state.slots[index].lease.is_some() {
                return Err(SchedulerError::ThreadLeased(id));
            }
            if matches!(
                state.slots[index].lifecycle,
                ThreadLifecycle::Exited | ThreadLifecycle::Faulted
            ) {
                return Err(state.invalid(index));
            }
            state.remove_queued(index, &self.profile);
            state.slots[index].wait = None;
            state.slots[index].lifecycle = if faulted {
                ThreadLifecycle::Faulted
            } else {
                ThreadLifecycle::Exited
            };
            state.cancel_property(index, &self.profile, true);
            state.detach_donations(index, &self.profile);
            state.unpin(index, &self.profile);
            state.changed(index);
            Ok(())
        })
    }

    pub fn pinned_thread(
        &self,
        process: ProcessId,
        core: VirtualCpuId,
    ) -> Result<Option<GuestThreadId>, SchedulerError> {
        let core = self.core(core)?;
        let state = self.state.lock().expect("scheduler lock poisoned");
        Ok(state
            .processes
            .get(&process)
            .and_then(|process| process.pinned[core])
            .map(|index| state.slots[index].id))
    }

    pub fn pin_thread(&self, core: VirtualCpuId, id: GuestThreadId) -> Result<(), SchedulerError> {
        let core = self.core(core)?;
        self.edit(|state| {
            let index = state.slots.index(id)?;
            let process = state.slots[index].config.process;
            if state.processes[&process].pinned[core].is_some()
                || state.slots[index].pinned_core.is_some()
            {
                return Err(SchedulerError::InvalidPin);
            }
            if matches!(
                state.slots[index].lifecycle,
                ThreadLifecycle::Created | ThreadLifecycle::Exited | ThreadLifecycle::Faulted
            ) {
                return Err(state.invalid(index));
            }
            let queued = state.slots[index].queued;
            let changed_affinity = state.physical_affinity(index, &self.profile).bits
                != 1 << self.profile.vcpus()[core].id().get()
                || state.slots[index].active_core != Some(core);
            if queued && changed_affinity {
                state.remove_memberships(index, &self.profile);
            }
            state.slots[index].pinned_core = Some(core);
            state.slots[index].active_core = Some(core);
            if queued && changed_affinity {
                state.insert_memberships(index, &self.profile, false);
                state.increment_scheduled_count(index);
            } else if !queued && state.slots[index].lifecycle == ThreadLifecycle::Ready {
                state.enqueue(index, &self.profile);
            }
            state.processes.get_mut(&process).unwrap().pinned[core] = Some(index);
            state.needs_selection = true;
            Ok(())
        })
    }

    pub fn unpin_thread(&self, id: GuestThreadId) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            state.unpin(index, &self.profile);
            Ok(())
        })
    }
}

impl State {
    fn update_selected(
        &mut self,
        core: usize,
        next: Option<Index>,
        profile: &MachineSchedulerProfile,
    ) {
        if self.cores[core].selected == next {
            return;
        }
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.core_reselections += 1;
        }
        if let Some(previous) = self.cores[core]
            .selected
            .filter(|&index| self.slots.entries[slot_index(index)].thread.is_some())
        {
            // Sample each outgoing transition, preserving actual clock ties.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_scheduler.cpp#L183-L194
            self.slots[previous].last_scheduled = self.clock.scheduler_time_ns();
            self.increment_scheduled_count(previous);
        }
        let slot = &mut self.cores[core];
        slot.selected = next;
        slot.generation = slot.generation.wrapping_add(1);
        self.wake_mask |= 1 << profile.vcpus()[core].id().get();
    }

    fn unpin(&mut self, index: Index, profile: &MachineSchedulerProfile) {
        let Some(core) = self.slots[index].pinned_core else {
            return;
        };
        let queued = self.slots[index].queued;
        let changed_membership = self.slots[index].config.affinity.bits
            != 1 << profile.vcpus()[core].id().get()
            || (self.slots[index].paused && self.slots[index].kernel_waiters == 0);
        if queued && changed_membership {
            self.remove_memberships(index, profile);
        }
        self.slots[index].pinned_core = None;
        let process = self.slots[index].config.process;
        self.processes.get_mut(&process).unwrap().pinned[core] = None;
        if !self.slots[index]
            .config
            .affinity
            .contains(profile.vcpus()[core].id())
        {
            self.slots[index].active_core = Some(self.affinity_core(index, profile));
        }
        if queued && changed_membership {
            if !self.slots[index].suspended() {
                self.insert_memberships(index, profile, false);
            } else {
                self.slots[index].queued = false;
            }
            self.increment_scheduled_count(index);
        }
        self.notify_pin_waiters(index, profile);
        self.needs_selection = true;
    }

    fn invalid(&self, index: Index) -> SchedulerError {
        SchedulerError::InvalidThreadState {
            thread: self.slots[index].id,
            state: self.slots[index].lifecycle,
        }
    }

    // Kernel dependencies must finish before a requested suspension takes effect.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L797-L808
    fn update_suspension(&mut self, index: Index, profile: &MachineSchedulerProfile) {
        if self.slots[index].suspended() {
            self.remove_queued(index, profile);
        } else if matches!(
            self.slots[index].lifecycle,
            ThreadLifecycle::Ready | ThreadLifecycle::Running
        ) && !self.slots[index].queued
        {
            self.enqueue(index, profile);
        }
        self.wake_kernel_operation(index);
    }

    fn increment_scheduled_count(&mut self, index: Index) {
        let process = self
            .processes
            .get_mut(&self.slots[index].config.process)
            .unwrap();
        process.generation = process.generation.wrapping_add(1);
    }

    fn changed(&mut self, index: Index) {
        self.increment_scheduled_count(index);
        self.needs_selection = true;
    }

    fn physical_affinity(&self, index: Index, profile: &MachineSchedulerProfile) -> CoreSet {
        self.slots[index]
            .pinned_core
            .map_or(self.slots[index].config.affinity, |core| CoreSet {
                bits: 1 << profile.vcpus()[core].id().get(),
            })
    }

    fn affinity_core(&self, index: Index, profile: &MachineSchedulerProfile) -> usize {
        let config = &self.slots[index].config;
        let core = config
            .ideal_vcpu
            .unwrap_or_else(|| config.affinity.iter().last().expect("nonempty affinity"));
        profile
            .vcpus()
            .binary_search_by_key(&core, |core| core.id())
            .unwrap()
    }

    fn choose_core(&self, index: Index, profile: &MachineSchedulerProfile) -> usize {
        let thread = &self.slots[index];
        if let Some(ideal) = thread.config.ideal_vcpu {
            return profile
                .vcpus()
                .binary_search_by_key(&ideal, |core| core.id())
                .unwrap();
        }
        self.cores
            .iter()
            .enumerate()
            .filter(|(core, _)| thread.config.affinity.contains(profile.vcpus()[*core].id()))
            .min_by_key(|(core, slot)| (slot.scheduled.len, *core))
            .map(|(core, _)| core)
            .expect("nonempty affinity")
    }

    fn enqueue(&mut self, index: Index, profile: &MachineSchedulerProfile) {
        debug_assert!(!self.slots[index].queued);
        if !self.slots[index].initialized {
            self.slots[index].active_core = Some(self.choose_core(index, profile));
            self.slots[index].initialized = true;
        }
        self.slots[index].queued = true;
        self.insert_memberships(index, profile, false);
        self.changed(index);
    }

    fn insert_memberships(&mut self, index: Index, profile: &MachineSchedulerProfile, front: bool) {
        for (core, descriptor) in profile.vcpus().iter().enumerate() {
            if self
                .physical_affinity(index, profile)
                .contains(descriptor.id())
            {
                let scheduled = self.slots[index].active_core == Some(core);
                let queue = if scheduled {
                    &mut self.cores[core].scheduled
                } else {
                    &mut self.cores[core].suggested
                };
                queue.push(&mut self.slots, core, index, front && scheduled);
            }
        }
    }

    fn remove_memberships(&mut self, index: Index, profile: &MachineSchedulerProfile) {
        for (core, descriptor) in profile.vcpus().iter().enumerate() {
            if self
                .physical_affinity(index, profile)
                .contains(descriptor.id())
            {
                let queue = if self.slots[index].active_core == Some(core) {
                    &mut self.cores[core].scheduled
                } else {
                    &mut self.cores[core].suggested
                };
                queue.remove(&mut self.slots, core, index);
            }
        }
    }

    fn remove_queued(&mut self, index: Index, profile: &MachineSchedulerProfile) {
        if self.slots[index].queued {
            self.remove_memberships(index, profile);
            self.slots[index].queued = false;
            self.changed(index);
        }
    }

    fn change_core(&mut self, index: Index, destination: Option<usize>, front: bool) {
        let old = self.slots[index].active_core;
        if old == destination {
            return;
        }
        debug_assert!(self.slots[index].queued);
        if let Some(core) = old {
            self.cores[core]
                .scheduled
                .remove(&mut self.slots, core, index);
            self.cores[core]
                .suggested
                .push(&mut self.slots, core, index, false);
        }
        if let Some(core) = destination {
            self.cores[core]
                .suggested
                .remove(&mut self.slots, core, index);
            self.cores[core]
                .scheduled
                .push(&mut self.slots, core, index, front);
        }
        self.slots[index].active_core = destination;
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.migrations += 1;
        }
    }

    fn source_top(&self, index: Index) -> Option<Index> {
        self.slots[index]
            .active_core
            .and_then(|core| self.cores[core].scheduled.front())
    }

    fn migration_allowed(&self, top: Option<Index>, profile: &MachineSchedulerProfile) -> bool {
        top.is_none_or(|index| {
            profile
                .migration_priority()
                .is_none_or(|limit| self.slots[index].priority >= limit)
        })
    }

    // Pinning is applied before donor selection; idle means no scheduled front,
    // not a core made unavailable by a non-runnable pinned thread.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_scheduler.cpp#L209-L315
    fn select_threads(&mut self, profile: &MachineSchedulerProfile) {
        if !self.needs_selection {
            return;
        }
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.selection_passes += 1;
        }
        let mut tops = [None; 64];
        let mut idle = 0u64;
        for (core, selected) in tops.iter_mut().enumerate().take(self.cores.len()) {
            let mut top = self.cores[core].scheduled.front();
            if let Some(index) = top {
                let thread = &self.slots[index];
                if thread.kernel_waiters == 0
                    && let Some(pinned) = self.processes[&thread.config.process].pinned[core]
                    && pinned != index
                {
                    let pinned_thread = &self.slots[pinned];
                    top = (pinned_thread.queued && !pinned_thread.suspended()).then_some(pinned);
                }
            } else {
                idle |= 1 << core;
            }
            *selected = top;
            self.update_selected(core, top, profile);
        }
        while idle != 0 {
            let core = idle.trailing_zeros() as usize;
            idle &= idle - 1;
            let mut donors = [0usize; 64];
            let mut donor_count = 0;
            let mut suggestion = self.cores[core].suggested.front();
            while let Some(index) = suggestion {
                let source = self.slots[index].active_core;
                let top = source.and_then(|core| tops[core]);
                if top != Some(index) {
                    if self.migration_allowed(top, profile) {
                        self.change_core(index, Some(core), false);
                        tops[core] = Some(index);
                        self.update_selected(core, Some(index), profile);
                    }
                    break;
                }
                if let Some(source) = source {
                    donors[donor_count] = source;
                    donor_count += 1;
                }
                suggestion = self.cores[core].suggested.next(&self.slots, core, index);
            }
            if suggestion.is_none() {
                for &source in &donors[..donor_count] {
                    let old_top = tops[source].expect("donor selected top");
                    if let Some(replacement) =
                        self.cores[source]
                            .scheduled
                            .next(&self.slots, source, old_top)
                    {
                        tops[source] = Some(replacement);
                        self.update_selected(source, Some(replacement), profile);
                        self.change_core(old_top, Some(core), false);
                        tops[core] = Some(old_top);
                        self.update_selected(core, Some(old_top), profile);
                        break;
                    }
                }
            }
        }
        self.needs_selection = false;
    }

    fn rotate(&mut self, core: usize, priority: i32, profile: &MachineSchedulerProfile) {
        let previous = self.cores[core].scheduled.priorities[priority as usize].first;
        let next = previous.and_then(|index| {
            self.cores[core]
                .scheduled
                .move_back(&mut self.slots, core, index)
        });
        if let Some(previous) = previous
            && next != Some(previous)
        {
            self.changed(previous);
            if let Some(next) = next {
                self.changed(next);
            }
        }
        let mut suggestion = self.cores[core].suggested.priorities[priority as usize].first;
        while let Some(index) = suggestion {
            let top = self.source_top(index);
            if top != Some(index) {
                if next != previous
                    && next.is_some_and(|next| {
                        self.slots[next].last_scheduled < self.slots[index].last_scheduled
                    })
                {
                    break;
                }
                if self.migration_allowed(top, profile) {
                    self.change_core(index, Some(core), true);
                    self.increment_scheduled_count(index);
                    break;
                }
            }
            suggestion = self.slots[index].links[core].next;
        }
        // Periodic rotation runs outside a guest thread, like Eden's host timer.
        // A core's live register owner is not the caller of this operation.
        let best = self.cores[core].scheduled.front();
        if let Some(best) = best
            && self.slots[best].priority >= priority
        {
            let mut suggestion = self.cores[core].suggested.front();
            while let Some(index) = suggestion {
                if self.slots[index].priority >= self.slots[best].priority {
                    break;
                }
                let top = self.source_top(index);
                if top != Some(index) && self.migration_allowed(top, profile) {
                    self.change_core(index, Some(core), true);
                    self.increment_scheduled_count(index);
                    break;
                }
                suggestion = self.cores[core].suggested.next(&self.slots, core, index);
            }
        }
        self.needs_selection = true;
    }
}

impl SchedulerState {
    pub fn rotate(&self) {
        self.edit(|state| {
            for (core, descriptor) in self.profile.vcpus().iter().enumerate() {
                if let Some(priority) = descriptor.rotation_priority() {
                    state.rotate(core, priority, &self.profile);
                }
            }
            Ok(())
        })
        .expect("rotation cannot fail");
    }

    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_scheduler.cpp#L677-L878
    pub fn yield_thread(
        &self,
        id: GuestThreadId,
        migration: YieldMigration,
    ) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(id)?;
            if state.slots[index].lease.is_some() || !state.slots[index].queued {
                return Err(state.invalid(index));
            }
            state.slots[index].lifecycle = ThreadLifecycle::Ready;
            let process = state.slots[index].config.process;
            if state.slots[index].yield_generation == Some(state.processes[&process].generation) {
                #[cfg(feature = "performance-counters")]
                {
                    state.statistics.cached_yields += 1;
                }
                return Ok(());
            }
            let Some(core) = state.slots[index].active_core else {
                return Ok(());
            };
            let priority = state.slots[index].priority;
            let mut recheck = false;
            let mut effect;
            match migration {
                YieldMigration::None | YieldMigration::Suggested => {
                    let next = state.cores[core]
                        .scheduled
                        .move_back(&mut state.slots, core, index);
                    state.increment_scheduled_count(index);
                    effect = next != Some(index);
                    if migration == YieldMigration::Suggested {
                        let mut suggestion = state.cores[core].suggested.front();
                        while let Some(candidate) = suggestion {
                            let top = state.slots[candidate]
                                .active_core
                                .and_then(|source| state.cores[source].selected);
                            if top != Some(candidate) {
                                let candidate_priority = state.slots[candidate].priority;
                                if candidate_priority > priority
                                    || candidate_priority == priority
                                        && effect
                                        && next.is_some_and(|next| {
                                            state.slots[next].last_scheduled
                                                < state.slots[candidate].last_scheduled
                                        })
                                {
                                    break;
                                }
                                if state.migration_allowed(top, &self.profile) {
                                    state.change_core(candidate, Some(core), true);
                                    state.increment_scheduled_count(candidate);
                                    effect = true;
                                    break;
                                }
                                recheck = true;
                            }
                            suggestion =
                                state.cores[core]
                                    .suggested
                                    .next(&state.slots, core, candidate);
                        }
                    }
                }
                YieldMigration::Any => {
                    state.change_core(index, None, false);
                    state.increment_scheduled_count(index);
                    if state.cores[core].scheduled.front().is_none() {
                        let mut suggestion = state.cores[core].suggested.front();
                        while let Some(candidate) = suggestion {
                            let top = state.source_top(candidate);
                            if top != Some(candidate) {
                                if state.migration_allowed(top, &self.profile) {
                                    state.change_core(candidate, Some(core), false);
                                    state.increment_scheduled_count(candidate);
                                }
                                break;
                            }
                            suggestion =
                                state.cores[core]
                                    .suggested
                                    .next(&state.slots, core, candidate);
                        }
                        effect = suggestion != Some(index);
                    } else {
                        effect = true;
                    }
                }
            }
            if effect {
                state.needs_selection = true;
            } else if !recheck {
                state.slots[index].yield_generation = Some(state.processes[&process].generation);
            }
            Ok(())
        })
    }
}

mod property;
mod wait_tree;
mod waits;
pub use property::{ThreadPropertyGuard, ThreadPropertyLock};
pub use waits::{LockHandoff, OrderedWaitKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    UnknownThread(GuestThreadId),
    UnknownVirtualCpu(VirtualCpuId),
    InvalidPriority(i32),
    IdealOutsideAffinity(Option<VirtualCpuId>),
    InvalidThreadState {
        thread: GuestThreadId,
        state: ThreadLifecycle,
    },
    ThreadLeased(GuestThreadId),
    VirtualCpuBusy(VirtualCpuId),
    StaleLease(Lease),
    DuplicateWait(GuestThreadId),
    StaleWake(WakeToken),
    ThreadIdExhausted,
    LockIdExhausted,
    LeaseGenerationExhausted,
    WakeGenerationExhausted,
    InvalidPin,
    EmptyAffinity,
    PriorityCycle(GuestThreadId),
}

impl Display for SchedulerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scheduler operation failed: {self:?}")
    }
}
impl Error for SchedulerError {}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(feature = "performance-counters")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerStatistics {
    pub operations: u64,
    pub lock_wait_ns: u64,
    pub lock_held_ns: u64,
    pub selection_passes: u64,
    pub migrations: u64,
    pub core_reselections: u64,
    pub cached_yields: u64,
}

#[cfg(feature = "performance-counters")]
impl SchedulerState {
    pub fn statistics(&self) -> SchedulerStatistics {
        self.state
            .lock()
            .expect("scheduler lock poisoned")
            .statistics
    }
}
