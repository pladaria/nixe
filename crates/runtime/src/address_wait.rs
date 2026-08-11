//! Console-neutral ownership for address-keyed wait queues.

use std::collections::{BTreeMap, VecDeque};

use nixe_scheduler::GuestThreadId;

use crate::{EventObject, ReadableEventObject, WritableEventObject};

#[derive(Clone, Debug)]
struct AddressWaiter {
    thread: GuestThreadId,
    value: u32,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

/// Process-owned address wait queues and mutex ownership records.
#[derive(Debug, Default)]
pub struct AddressWaitRegistry {
    waiters: BTreeMap<u64, VecDeque<AddressWaiter>>,
    owners: BTreeMap<u64, GuestThreadId>,
}

impl AddressWaitRegistry {
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
        let (writable, readable) = EventObject::create_pair();
        self.waiters
            .entry(address)
            .or_default()
            .push_back(AddressWaiter {
                thread,
                value,
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

    pub fn signal(&self, address: u64, count: usize) {
        if let Some(waiters) = self.waiters.get(&address) {
            for waiter in waiters.iter().take(count) {
                waiter.writable.signal();
            }
        }
    }

    pub fn signal_one(&self, address: u64) {
        self.signal(address, 1);
    }

    pub fn set_owner(&mut self, address: u64, thread: GuestThreadId) {
        self.owners.insert(address, thread);
    }

    pub fn remove_owner(&mut self, address: u64) -> Option<GuestThreadId> {
        self.owners.remove(&address)
    }

    /// Removes one terminating thread and wakes the next waiter for every mutex
    /// it owned. The caller separately removes scheduler wait tokens.
    pub fn release_thread(&mut self, thread: GuestThreadId) {
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
            self.signal_one(address);
        }
    }

    #[must_use]
    pub fn waiter_count(&self) -> usize {
        self.waiters.values().map(VecDeque::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_exit_removes_its_waits_and_hands_off_once() {
        let owner = GuestThreadId::new(1);
        let waiter = GuestThreadId::new(2);
        let mut registry = AddressWaitRegistry::default();
        registry.set_owner(0x1000, owner);
        let wake = registry.enqueue(0x1000, waiter, 0x1234);
        assert!(!wake.is_signalled());
        registry.release_thread(owner);
        assert!(wake.is_signalled());
        assert_eq!(registry.value(0x1000, waiter), Some(0x1234));
        registry.release_thread(waiter);
        assert_eq!(registry.waiter_count(), 0);
    }
}
