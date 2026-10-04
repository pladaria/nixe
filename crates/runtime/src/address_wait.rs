//! Console-neutral ownership for address-keyed wait queues.

use std::collections::{BTreeMap, VecDeque};

use nixe_scheduler::GuestThreadId;

use crate::{EventObject, ReadableEventObject, WritableEventObject};

mod priority;
pub use priority::{AddressWaitResult, PriorityAddressWaitQueue};

#[derive(Clone, Debug)]
struct AddressWaiter {
    thread: GuestThreadId,
    value: u32,
    lock_address: Option<u64>,
    priority: i32,
    completion: AddressWaitCompletion,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

/// Completion payload for a mutex or condition-variable address wait.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressWaitCompletion {
    Success,
    InvalidOwner,
    InvalidMemory,
    TimedOut,
}

/// Process-owned address wait queues and mutex ownership records.
#[derive(Debug, Default)]
pub struct AddressWaitRegistry {
    waiters: BTreeMap<u64, VecDeque<AddressWaiter>>,
    owners: BTreeMap<u64, GuestThreadId>,
    priority_waits: PriorityAddressWaitQueue,
}

impl AddressWaitRegistry {
    /// Priority-ordered address signals have an independent queue namespace
    /// from mutex ownership and condition-variable keys.
    pub const fn priority_waits(&self) -> &PriorityAddressWaitQueue {
        &self.priority_waits
    }

    pub const fn priority_waits_mut(&mut self) -> &mut PriorityAddressWaitQueue {
        &mut self.priority_waits
    }

    #[must_use]
    pub fn contains(&self, address: u64, thread: GuestThreadId) -> bool {
        self.waiters
            .get(&address)
            .is_some_and(|waiters| waiters.iter().any(|waiter| waiter.thread == thread))
    }

    pub fn enqueue(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
    ) -> ReadableEventObject {
        self.enqueue_with_lock(address, thread, value, None)
    }

    pub fn enqueue_condition(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
        lock: u64,
    ) -> ReadableEventObject {
        self.enqueue_with_lock(address, thread, value, Some(lock))
    }

    fn enqueue_with_lock(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
        lock_address: Option<u64>,
    ) -> ReadableEventObject {
        let (writable, readable) = EventObject::create_pair();
        self.waiters
            .entry(address)
            .or_default()
            .push_back(AddressWaiter {
                thread,
                value,
                lock_address,
                priority: i32::MAX,
                completion: AddressWaitCompletion::Success,
                writable,
                readable: readable.clone(),
            });
        readable
    }

    #[must_use]
    pub fn is_signalled(&self, address: u64, thread: GuestThreadId) -> bool {
        self.waiters.get(&address).is_some_and(|waiters| {
            waiters
                .iter()
                .find(|waiter| waiter.thread == thread)
                .is_some_and(|waiter| waiter.readable.is_signalled())
        })
    }

    #[must_use]
    pub fn value(&self, address: u64, thread: GuestThreadId) -> Option<u32> {
        self.waiters
            .get(&address)?
            .iter()
            .find(|waiter| waiter.thread == thread)
            .map(|waiter| waiter.value)
    }

    pub fn remove(&mut self, address: u64, thread: GuestThreadId) {
        if let Some(waiters) = self.waiters.get_mut(&address)
            && let Some(index) = waiters.iter().position(|waiter| waiter.thread == thread)
        {
            waiters.remove(index);
        }
        if self.waiters.get(&address).is_some_and(VecDeque::is_empty) {
            self.waiters.remove(&address);
        }
    }

    pub fn change_priority(&mut self, thread: GuestThreadId, priority: i32) {
        self.priority_waits.change_priority(thread, priority);
        for waiters in self.waiters.values_mut() {
            for waiter in waiters.iter_mut().filter(|waiter| waiter.thread == thread) {
                waiter.priority = priority;
            }
        }
    }

    /// Initializes newly registered mutex/condition waiters without scanning
    /// the process thread table or allocating a temporary priority snapshot.
    pub fn refresh_mutex_priorities(&mut self, mut priority: impl FnMut(GuestThreadId) -> i32) {
        for waiter in self.waiters.values_mut().flatten() {
            waiter.priority = priority(waiter.thread);
        }
    }

    pub fn unsignalled_count(&self, address: u64) -> usize {
        self.waiters.get(&address).map_or(0, |waiters| {
            waiters
                .iter()
                .filter(|waiter| waiter.lock_address.is_none() && !waiter.readable.is_signalled())
                .count()
        })
    }

    /// Select current effective priority, then FIFO among equal priorities.
    pub fn next_waiter(&self, address: u64) -> Option<(GuestThreadId, u32)> {
        self.waiters
            .get(&address)?
            .iter()
            .enumerate()
            .filter(|(_, waiter)| waiter.lock_address.is_none() && !waiter.readable.is_signalled())
            .min_by_key(|(order, waiter)| (waiter.priority, *order))
            .map(|(_, waiter)| (waiter.thread, waiter.value))
    }

    pub fn next_condition(&self, address: u64) -> Option<(GuestThreadId, u32, u64)> {
        self.waiters
            .get(&address)?
            .iter()
            .enumerate()
            .filter(|(_, waiter)| waiter.lock_address.is_some() && !waiter.readable.is_signalled())
            .min_by_key(|(order, waiter)| (waiter.priority, *order))
            .map(|(_, waiter)| (waiter.thread, waiter.value, waiter.lock_address.unwrap()))
    }

    pub fn pending_threads(&self, address: u64) -> impl Iterator<Item = GuestThreadId> + '_ {
        self.waiters
            .get(&address)
            .into_iter()
            .flatten()
            .filter(|waiter| waiter.lock_address.is_none() && !waiter.readable.is_signalled())
            .map(|waiter| waiter.thread)
    }

    pub fn completion(&self, address: u64, thread: GuestThreadId) -> Option<AddressWaitCompletion> {
        self.waiters
            .get(&address)?
            .iter()
            .find(|waiter| waiter.thread == thread)
            .map(|waiter| waiter.completion)
    }

    pub fn signal_result(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        result: AddressWaitCompletion,
    ) {
        if let Some(waiter) = self
            .waiters
            .get_mut(&address)
            .and_then(|waiters| waiters.iter_mut().find(|waiter| waiter.thread == thread))
        {
            waiter.completion = result;
            waiter.writable.signal();
        }
    }

    /// Latch expiry before a ready thread runs again. A later signal cannot
    /// reacquire a mutex for a wait whose timer has already completed.
    pub fn expire_thread_wait(&mut self, thread: GuestThreadId) -> bool {
        for waiters in self.waiters.values_mut() {
            if let Some(waiter) = waiters.iter_mut().find(|waiter| waiter.thread == thread)
                && !waiter.readable.is_signalled()
            {
                waiter.completion = AddressWaitCompletion::TimedOut;
                waiter.writable.signal();
                return true;
            }
        }
        false
    }

    /// Preserve the same event and deadline while a condition waiter reacquires
    /// its mutex. The runtime must not wake it until ownership is transferred.
    pub fn move_waiter(&mut self, from: u64, to: u64, thread: GuestThreadId) {
        let waiters = self
            .waiters
            .get_mut(&from)
            .expect("registered condition waiter");
        let index = waiters
            .iter()
            .position(|waiter| waiter.thread == thread)
            .unwrap();
        let mut waiter = waiters.remove(index).unwrap();
        waiter.lock_address = None;
        if waiters.is_empty() {
            self.waiters.remove(&from);
        }
        self.waiters.entry(to).or_default().push_back(waiter);
    }

    pub fn owner(&self, address: u64) -> Option<GuestThreadId> {
        self.owners.get(&address).copied()
    }

    pub fn set_owner(&mut self, address: u64, thread: GuestThreadId) {
        self.owners.insert(address, thread);
    }

    pub fn remove_owner(&mut self, address: u64) -> Option<GuestThreadId> {
        self.owners.remove(&address)
    }

    /// Removes one terminating thread and cancels all waits on its mutexes.
    /// No ownership can be transferred without a guest memory handoff.
    /// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_thread.cpp#L390-L415
    pub fn release_thread(&mut self, thread: GuestThreadId) {
        self.priority_waits.remove(thread);
        let addresses: Vec<_> = self.waiters.keys().copied().collect();
        for address in addresses {
            self.remove(address, thread);
        }
        let owned: Vec<_> = self
            .owners
            .iter()
            .filter_map(|(address, owner)| (*owner == thread).then_some(*address))
            .collect();
        for address in owned {
            self.owners.remove(&address);
            while let Some((waiter, _)) = self.next_waiter(address) {
                self.signal_result(address, waiter, AddressWaitCompletion::InvalidOwner);
            }
        }
    }

    #[must_use]
    pub fn waiter_count(&self) -> usize {
        self.waiters.values().map(VecDeque::len).sum::<usize>() + self.priority_waits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_exit_cancels_all_waiters_without_granting_ownership() {
        let owner = GuestThreadId::new(1);
        let waiter = GuestThreadId::new(2);
        let mut registry = AddressWaitRegistry::default();
        registry.set_owner(0x1000, owner);
        let wake = registry.enqueue(0x1000, waiter, 0x1234);
        let other = GuestThreadId::new(3);
        let other_wake = registry.enqueue(0x1000, other, 0x5678);
        assert!(!wake.is_signalled());
        registry.release_thread(owner);
        assert!(wake.is_signalled());
        assert!(other_wake.is_signalled());
        assert_eq!(
            registry.completion(0x1000, waiter),
            Some(AddressWaitCompletion::InvalidOwner)
        );
        assert_eq!(
            registry.completion(0x1000, other),
            Some(AddressWaitCompletion::InvalidOwner)
        );
        assert_eq!(registry.owner(0x1000), None);
        assert_eq!(registry.value(0x1000, waiter), Some(0x1234));
        registry.release_thread(waiter);
        registry.release_thread(other);
        assert_eq!(registry.waiter_count(), 0);
    }
}
