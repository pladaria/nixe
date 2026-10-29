//! Kernel-owned ordered wait nodes and transferable priority-inheritance locks.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderedWaitKind {
    ConditionVariable,
    AddressArbiter,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LockHandoff {
    pub next: Option<GuestThreadId>,
    pub more_waiters: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Membership {
    Mutex(Index),
    Ordered(OrderedWaitKind, u64),
    Pinned(Index),
    ContextRelease(Index),
}

#[derive(Debug)]
struct Lock {
    process: ProcessId,
    key: LockKey,
    owner: Index,
    previous: Option<Index>,
    next: Option<Index>,
    waiters: WaitTree,
}

#[derive(Debug)]
pub(super) struct LockSlot {
    value: Option<Lock>,
    next_free: Option<Index>,
}

impl State {
    pub(super) fn add_waiter(
        &mut self,
        process: ProcessId,
        key: LockKey,
        owner: Index,
        waiter: Index,
        profile: &MachineSchedulerProfile,
        current: Option<GuestThreadId>,
    ) -> Result<(), SchedulerError> {
        let existing = self.find_held_lock(process, key, owner);
        if let Some(lock) = existing
            && self.slots[waiter].waiting_on == Some(Membership::Mutex(lock))
        {
            return Ok(());
        }
        self.check_donation(owner, waiter)?;
        self.remove_waiter(waiter, profile, current);
        let lock = if let Some(lock) = existing {
            lock
        } else {
            self.allocate_lock(process, key, owner)?
        };
        let membership = Membership::Mutex(lock);
        self.slots[waiter].waiting_on = Some(membership);
        self.insert_wait_node(waiter, membership);
        if key.is_kernel() {
            self.slots[owner].kernel_waiters += 1;
            self.needs_selection = true;
        }
        if self.slots[waiter].priority < self.slots[owner].priority {
            self.restore_priority(owner, profile, current);
        }
        if key.is_kernel() {
            self.update_suspension(owner, profile);
        }
        Ok(())
    }
    pub(super) fn release_lock(
        &mut self,
        process: ProcessId,
        key: LockKey,
        owner: Index,
        profile: &MachineSchedulerProfile,
    ) -> LockHandoff {
        let Some(lock) = self.find_held_lock(process, key, owner) else {
            return LockHandoff::default();
        };
        let count = self.lock(lock).waiters.len;
        let next = self
            .lock(lock)
            .waiters
            .first(&self.slots)
            .expect("contended lock");
        let priority = self.slots[next].priority;
        self.unlink_wait_node(next, Membership::Mutex(lock));
        self.slots[next].waiting_on = None;
        let more = self.lock(lock).waiters.len != 0;
        if key.is_kernel() {
            self.slots[owner].kernel_waiters -= count;
            self.slots[next].kernel_waiters += count - 1;
            self.update_suspension(owner, profile);
            self.update_suspension(next, profile);
            self.needs_selection = true;
        }
        if more {
            self.unlink_held(lock);
            self.link_held(lock, next);
        } else {
            self.free_lock(lock);
        }
        if self.slots[owner].priority == priority
            && self.slots[owner].priority < self.slots[owner].config.base_priority
        {
            self.restore_priority(owner, profile, Some(self.slots[owner].id));
        }
        LockHandoff {
            next: Some(self.slots[next].id),
            more_waiters: more,
        }
    }

    pub(super) fn insert_ordered_wait(
        &mut self,
        index: Index,
        kind: OrderedWaitKind,
        address: u64,
    ) -> Result<(), SchedulerError> {
        if self.slots[index].waiting_on.is_some() {
            return Err(SchedulerError::DuplicateWait(self.slots[index].id));
        }
        let membership = Membership::Ordered(kind, address);
        self.slots[index].waiting_on = Some(membership);
        self.insert_wait_node(index, membership);
        Ok(())
    }

    pub(super) fn begin_wait(
        &mut self,
        index: Index,
        profile: &MachineSchedulerProfile,
        kernel: std::sync::Weak<SchedulerState>,
    ) -> Result<Arc<KernelWait>, SchedulerError> {
        if self.closed {
            return Err(SchedulerError::KernelStopped);
        }
        if self.slots[index].lifecycle != ThreadLifecycle::Waiting {
            return Err(self.invalid(index));
        }
        if self.slots[index].wait.is_some() {
            return Err(SchedulerError::DuplicateWait(self.slots[index].id));
        }
        self.wake_generation = self
            .wake_generation
            .checked_add(1)
            .ok_or(SchedulerError::WakeGenerationExhausted)?;
        let token = WakeToken {
            thread: self.slots[index].id,
            generation: crate::WakeGeneration::new(self.wake_generation),
        };
        self.remove_queued(index, profile);
        #[cfg(feature = "performance-counters")]
        {
            self.slots[index].resolved_at_ns = None;
        }
        let wait = Arc::new(KernelWait::new(token, kernel));
        self.slots[index].wait = Some(wait.clone());
        Ok(wait)
    }
    fn lock(&self, index: Index) -> &Lock {
        self.locks[slot_index(index)]
            .value
            .as_ref()
            .expect("live lock")
    }
    fn lock_mut(&mut self, index: Index) -> &mut Lock {
        self.locks[slot_index(index)]
            .value
            .as_mut()
            .expect("live lock")
    }
    fn find_held_lock(&self, process: ProcessId, key: LockKey, owner: Index) -> Option<Index> {
        let mut held = self.slots[owner].held;
        while let Some(index) = held {
            let lock = self.lock(index);
            if lock.process == process && lock.key == key {
                return Some(index);
            }
            held = lock.next;
        }
        None
    }
    fn link_held(&mut self, index: Index, owner: Index) {
        let next = self.slots[owner].held;
        self.lock_mut(index).owner = owner;
        self.lock_mut(index).previous = None;
        self.lock_mut(index).next = next;
        if let Some(next) = next {
            self.lock_mut(next).previous = Some(index);
        }
        self.slots[owner].held = Some(index);
    }
    fn unlink_held(&mut self, index: Index) {
        let lock = self.lock(index);
        let (owner, previous, next) = (lock.owner, lock.previous, lock.next);
        if let Some(previous) = previous {
            self.lock_mut(previous).next = next;
        } else {
            self.slots[owner].held = next;
        }
        if let Some(next) = next {
            self.lock_mut(next).previous = previous;
        }
    }
    fn allocate_lock(
        &mut self,
        process: ProcessId,
        key: LockKey,
        owner: Index,
    ) -> Result<Index, SchedulerError> {
        let index = if let Some(index) = self.free_lock {
            self.free_lock = self.locks[slot_index(index)].next_free;
            index
        } else {
            let index = u32::try_from(self.locks.len())
                .ok()
                .and_then(|v| v.checked_add(1))
                .and_then(Index::new)
                .ok_or(SchedulerError::LockIdExhausted)?;
            self.locks.push(LockSlot {
                value: None,
                next_free: None,
            });
            index
        };
        self.locks[slot_index(index)].value = Some(Lock {
            process,
            key,
            owner,
            previous: None,
            next: None,
            waiters: WaitTree::default(),
        });
        self.link_held(index, owner);
        Ok(index)
    }
    fn free_lock(&mut self, index: Index) -> Lock {
        self.unlink_held(index);
        let slot = &mut self.locks[slot_index(index)];
        let lock = slot.value.take().expect("live lock");
        slot.next_free = self.free_lock;
        self.free_lock = Some(index);
        lock
    }
    fn unlink_wait_node(&mut self, index: Index, membership: Membership) {
        let thread = &self.slots[index];
        let process = thread.config.process;
        match membership {
            Membership::ContextRelease(_) => {}
            Membership::Pinned(owner) => {
                let links = self.slots[index].wait_node;
                if let Some(previous) = links.left {
                    self.slots[previous].wait_node.right = links.right;
                } else {
                    self.slots[owner].pin_waiters.first = links.right;
                }
                if let Some(next) = links.right {
                    self.slots[next].wait_node.left = links.left;
                } else {
                    self.slots[owner].pin_waiters.last = links.left;
                }
                self.slots[index].wait_node = WaitNode::default();
            }
            Membership::Mutex(lock) => {
                self.locks[slot_index(lock)]
                    .value
                    .as_mut()
                    .expect("live lock")
                    .waiters
                    .remove(&mut self.slots, index);
            }
            Membership::Ordered(kind, _) => {
                self.processes.get_mut(&process).unwrap().ordered[kind as usize]
                    .remove(&mut self.slots, index);
            }
        }
    }
    pub(super) fn insert_wait_node(&mut self, index: Index, membership: Membership) {
        if matches!(membership, Membership::ContextRelease(_)) {
            return;
        }
        if let Membership::Pinned(owner) = membership {
            let previous = self.slots[owner].pin_waiters.last;
            self.slots[index].wait_node = WaitNode {
                left: previous,
                ..WaitNode::default()
            };
            if let Some(previous) = previous {
                self.slots[previous].wait_node.right = Some(index);
            } else {
                self.slots[owner].pin_waiters.first = Some(index);
            }
            self.slots[owner].pin_waiters.last = Some(index);
            return;
        }
        match membership {
            Membership::Pinned(_) | Membership::ContextRelease(_) => unreachable!(),
            Membership::Mutex(lock) => {
                self.locks[slot_index(lock)]
                    .value
                    .as_mut()
                    .expect("live lock")
                    .waiters
                    .insert(&mut self.slots, index);
            }
            Membership::Ordered(kind, _) => {
                let process = self.slots[index].config.process;
                self.processes.get_mut(&process).unwrap().ordered[kind as usize]
                    .insert(&mut self.slots, index);
            }
        }
    }
    fn lock_owner(&self, thread: Index) -> Option<Index> {
        match self.slots[thread].waiting_on {
            Some(Membership::Mutex(lock)) => Some(self.lock(lock).owner),
            _ => None,
        }
    }
    pub(super) fn restore_priority(
        &mut self,
        mut index: Index,
        profile: &MachineSchedulerProfile,
        current: Option<GuestThreadId>,
    ) {
        loop {
            let old_priority = self.slots[index].priority;
            let mut priority = self.slots[index].config.base_priority;
            let mut held = self.slots[index].held;
            while let Some(lock) = held {
                let record = self.lock(lock);
                if let Some(donor) = record.waiters.first(&self.slots) {
                    priority = priority.min(self.slots[donor].priority);
                }
                held = record.next;
            }
            if priority == old_priority {
                return;
            }
            let membership = self.slots[index].waiting_on.filter(|membership| {
                matches!(membership, Membership::Mutex(_) | Membership::Ordered(..))
            });
            let owner = self.lock_owner(index);
            if let Some(membership) = membership {
                self.unlink_wait_node(index, membership);
            }
            let queued = self.slots[index].queued;
            if queued {
                self.remove_memberships(index, profile);
            }
            self.slots[index].priority = priority;
            if queued {
                self.insert_memberships(index, profile, Some(self.slots[index].id) == current);
                self.changed(index);
            }
            if let Some(membership) = membership {
                self.insert_wait_node(index, membership);
            }
            if let Some(owner) = owner {
                index = owner;
            } else {
                return;
            }
        }
    }
    fn check_donation(&self, mut owner: Index, waiter: Index) -> Result<(), SchedulerError> {
        loop {
            if owner == waiter {
                return Err(SchedulerError::PriorityCycle(self.slots[waiter].id));
            }
            if let Some(next) = self.lock_owner(owner) {
                owner = next;
            } else {
                return Ok(());
            }
        }
    }
    pub(super) fn remove_waiter(
        &mut self,
        index: Index,
        profile: &MachineSchedulerProfile,
        current: Option<GuestThreadId>,
    ) {
        let Some(membership) = self.slots[index].waiting_on.take() else {
            return;
        };
        self.unlink_wait_node(index, membership);
        if let Membership::Mutex(lock) = membership {
            let record = self.lock(lock);
            let (owner, kernel, empty) = (
                record.owner,
                record.key.is_kernel(),
                record.waiters.len == 0,
            );
            if kernel {
                self.slots[owner].kernel_waiters -= 1;
                self.update_suspension(owner, profile);
                self.needs_selection = true;
            }
            if empty {
                self.free_lock(lock);
            }
            if self.slots[owner].priority == self.slots[index].priority
                && self.slots[owner].priority < self.slots[owner].config.base_priority
            {
                self.restore_priority(owner, profile, current);
            }
        }
    }
    pub(super) fn detach_donations(&mut self, index: Index, profile: &MachineSchedulerProfile) {
        self.remove_waiter(index, profile, None);
        while let Some(lock) = self.slots[index].held {
            let mut record = self.free_lock(lock);
            while let Some(waiter) = record.waiters.first(&self.slots) {
                record.waiters.remove(&mut self.slots, waiter);
                self.slots[waiter].waiting_on = None;
            }
        }
        self.slots[index].kernel_waiters = 0;
    }
}

impl SchedulerState {
    /// Finalization visits the owner's actual lock nodes before retiring them.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L352-L380
    pub fn take_held_waiter(&self, owner: GuestThreadId) -> Option<(ProcessId, GuestThreadId)> {
        self.edit(|state| {
            let Ok(owner) = state.slots.index(owner) else {
                return Ok(None);
            };
            let Some(lock) = state.slots[owner].held else {
                return Ok(None);
            };
            debug_assert!(
                !state.lock(lock).key.is_kernel(),
                "terminal owner retains kernel waiters"
            );
            let waiter = state
                .lock(lock)
                .waiters
                .first(&state.slots)
                .expect("contended held lock");
            state.unlink_wait_node(waiter, Membership::Mutex(lock));
            state.slots[waiter].waiting_on = None;
            if state.lock(lock).waiters.len == 0 {
                state.free_lock(lock);
            }
            // A terminating owner needs no priority restoration. Cancellation
            // consumes the detached node, as in KThread::Finalize.
            Ok(Some((
                state.slots[waiter].config.process,
                state.slots[waiter].id,
            )))
        })
        .expect("detaching a terminal owner's waiter cannot fail")
    }

    pub fn set_priority(
        &self,
        id: GuestThreadId,
        priority: i32,
        current: Option<GuestThreadId>,
    ) -> Result<(), SchedulerError> {
        if !self.profile.priorities().contains(priority) {
            return Err(SchedulerError::InvalidPriority(priority));
        }
        self.edit(|state| {
            let index = state.slots.index(id)?;
            state.slots[index].config.base_priority = priority;
            state.restore_priority(index, &self.profile, current);
            Ok(())
        })
    }
    pub fn add_waiter(
        &self,
        process: ProcessId,
        key: LockKey,
        owner: GuestThreadId,
        waiter: GuestThreadId,
        current: Option<GuestThreadId>,
    ) -> Result<(), SchedulerError> {
        self.edit(|state| {
            let owner = state.slots.index(owner)?;
            let waiter = state.slots.index(waiter)?;
            state.add_waiter(process, key, owner, waiter, &self.profile, current)
        })
    }
    /// Reuses the existing lock record and remaining ordered nodes on handoff.
    pub fn release_lock(
        &self,
        process: ProcessId,
        key: LockKey,
        owner: GuestThreadId,
    ) -> Result<(LockHandoff, Option<WaitEffect>), SchedulerError> {
        self.edit(|state| {
            let owner = state.slots.index(owner)?;
            let handoff = state.release_lock(process, key, owner, &self.profile);
            let effect = handoff
                .next
                .and_then(|thread| state.slots.index(thread).ok())
                .and_then(|index| state.claim_wait_effect(index));
            Ok((handoff, effect))
        })
    }
    pub fn remove_waiter(&self, thread: GuestThreadId) -> bool {
        self.edit(|state| {
            let Ok(index) = state.slots.index(thread) else {
                return Ok(false);
            };
            let present = state.slots[index].waiting_on.is_some();
            state.remove_waiter(index, &self.profile, None);
            Ok(present)
        })
        .expect("removing a waiter cannot fail")
    }
    pub fn begin_mutex_wait(
        self: &Arc<Self>,
        process: ProcessId,
        key: LockKey,
        owner: GuestThreadId,
        thread: GuestThreadId,
    ) -> Result<Arc<KernelWait>, SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(thread)?;
            let owner = state.slots.index(owner)?;
            let wait = if let Some(wait) = &state.slots[index].wait {
                wait.clone()
            } else {
                state.begin_wait(index, &self.profile, Arc::downgrade(self))?
            };
            state.add_waiter(process, key, owner, index, &self.profile, Some(thread))?;
            Ok(wait)
        })
    }
    pub fn begin_ordered_wait(
        self: &Arc<Self>,
        thread: GuestThreadId,
        kind: OrderedWaitKind,
        address: u64,
        atomic_effect: bool,
    ) -> Result<(Arc<KernelWait>, Option<WaitEffect>), SchedulerError> {
        self.edit(|state| {
            let index = state.slots.index(thread)?;
            let wait = if let Some(wait) = &state.slots[index].wait {
                wait.clone()
            } else {
                state.begin_wait(index, &self.profile, Arc::downgrade(self))?
            };
            state.insert_ordered_wait(index, kind, address)?;
            let effect = atomic_effect.then(|| {
                state
                    .claim_wait_effect(index)
                    .expect("unresolved atomic wait")
            });
            Ok((wait, effect))
        })
    }
    pub fn first_ordered_waiter(
        &self,
        process: ProcessId,
        kind: OrderedWaitKind,
        address: u64,
    ) -> Option<GuestThreadId> {
        let state = self.state.lock().expect("scheduler lock poisoned");
        let queues = &state.processes.get(&process)?.ordered[kind as usize];
        queues
            .lower_bound(&state.slots, address)
            .filter(|&index| {
                state.slots[index].waiting_on == Some(Membership::Ordered(kind, address))
            })
            .map(|index| state.slots[index].id)
    }
    pub fn ordered_waiters(
        &self,
        process: ProcessId,
        kind: OrderedWaitKind,
        address: u64,
        count: usize,
    ) -> Vec<GuestThreadId> {
        let state = self.state.lock().expect("scheduler lock poisoned");
        let Some(process) = state.processes.get(&process) else {
            return Vec::new();
        };
        std::iter::successors(
            process.ordered[kind as usize].lower_bound(&state.slots, address),
            |&index| WaitTree::next(&state.slots, index),
        )
        .take_while(|&index| {
            state.slots[index].waiting_on == Some(Membership::Ordered(kind, address))
        })
        .take(count)
        .map(|index| state.slots[index].id)
        .collect()
    }
    pub fn claim_ordered_waiters(
        &self,
        process: ProcessId,
        kind: OrderedWaitKind,
        address: u64,
        count: usize,
        unlink: bool,
    ) -> (Vec<(GuestThreadId, WaitEffect)>, bool) {
        self.edit(|state| {
            let mut claims = Vec::new();
            let mut node = state.processes.get(&process).and_then(|process| {
                process.ordered[kind as usize].lower_bound(&state.slots, address)
            });
            while let Some(index) = node.filter(|&index| {
                state.slots[index].waiting_on == Some(Membership::Ordered(kind, address))
            }) {
                if claims.len() == count {
                    break;
                }
                node = WaitTree::next(&state.slots, index);
                if let Some(effect) = state.claim_wait_effect(index) {
                    claims.push((state.slots[index].id, effect));
                    if unlink {
                        state.remove_waiter(index, &self.profile, None);
                    }
                }
            }
            let more = node.is_some_and(|index| {
                state.slots[index].waiting_on == Some(Membership::Ordered(kind, address))
            });
            Ok((claims, more))
        })
        .expect("claiming ordered waiters")
    }

    pub fn priority_donation_count(&self) -> usize {
        self.state
            .lock()
            .expect("scheduler lock poisoned")
            .slots
            .entries
            .iter()
            .filter(|slot| {
                slot.thread
                    .as_ref()
                    .is_some_and(|thread| matches!(thread.waiting_on, Some(Membership::Mutex(_))))
            })
            .count()
    }
}

#[cfg(test)]
mod tests;
