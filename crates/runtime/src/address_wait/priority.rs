//! Priority-ordered, process-local address waits with explicit completion.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use nixe_scheduler::GuestThreadId;

use crate::{EventObject, ReadableEventObject, WritableEventObject};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressWaitResult {
    Signalled,
    TimedOut,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signals_choose_priority_then_fifo_and_never_consume_a_wait_twice() {
        let mut queue = PriorityAddressWaitQueue::default();
        let low = GuestThreadId::new(1);
        let first = GuestThreadId::new(2);
        let second = GuestThreadId::new(3);
        let low_event = queue.enqueue(0x1000, low, 40, None).unwrap();
        let first_event = queue.enqueue(0x1000, first, 20, None).unwrap();
        let second_event = queue.enqueue(0x1000, second, 20, None).unwrap();
        queue.signal(0x1000, 1);
        queue.signal(0x1000, 1);
        assert!(first_event.is_signalled());
        assert!(second_event.is_signalled());
        assert!(!low_event.is_signalled());
        assert_eq!(queue.waiting_count(0x1000), 1);
        assert_eq!(queue.take_result(first), Some(AddressWaitResult::Signalled));
        assert_eq!(
            queue.take_result(second),
            Some(AddressWaitResult::Signalled)
        );
        queue.signal(0x1000, usize::MAX);
        assert_eq!(queue.take_result(low), Some(AddressWaitResult::Signalled));
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_changes_reinsert_the_waiter_at_the_new_priority() {
        let mut queue = PriorityAddressWaitQueue::default();
        let first = GuestThreadId::new(1);
        let second = GuestThreadId::new(2);
        let first_event = queue.enqueue(0x1000, first, 20, Some(100)).unwrap();
        let second_event = queue.enqueue(0x1000, second, 40, Some(100)).unwrap();
        queue.change_priority(second, 10);
        queue.signal(0x1000, 1);
        assert!(second_event.is_signalled());
        assert!(!first_event.is_signalled());
        queue.change_priority(second, 50); // Completion is no longer queued.
        queue.expire(100);
        assert_eq!(queue.take_result(first), Some(AddressWaitResult::TimedOut));
        assert_eq!(
            queue.take_result(second),
            Some(AddressWaitResult::Signalled)
        );
        assert!(queue.is_empty());
    }

    #[test]
    fn expiry_precedes_signals_and_completed_waiters_do_not_affect_the_count() {
        let mut queue = PriorityAddressWaitQueue::default();
        let expired = GuestThreadId::new(1);
        let active = GuestThreadId::new(2);
        let expired_event = queue.enqueue(0x1000, expired, 10, Some(10)).unwrap();
        let active_event = queue.enqueue(0x1000, active, 20, Some(20)).unwrap();
        queue.expire(10);
        assert!(expired_event.is_signalled());
        assert_eq!(queue.waiting_count(0x1000), 1);
        queue.signal(0x1000, 1);
        queue.expire(20);
        assert!(active_event.is_signalled());
        assert_eq!(
            queue.take_result(expired),
            Some(AddressWaitResult::TimedOut)
        );
        assert_eq!(
            queue.take_result(active),
            Some(AddressWaitResult::Signalled)
        );
        assert!(queue.is_empty());
    }

    #[test]
    fn removal_cancels_deadlines_and_allows_a_new_wait_on_another_address() {
        let mut queue = PriorityAddressWaitQueue::default();
        let thread = GuestThreadId::new(1);
        let old_event = queue.enqueue(0x1000, thread, 20, Some(10)).unwrap();
        assert!(queue.enqueue(0x1000, thread, 20, None).is_none());
        queue.remove(thread);
        let new_event = queue.enqueue(0x2000, thread, 20, None).unwrap();
        queue.expire(10);
        queue.signal(0x1000, usize::MAX);
        assert!(!old_event.is_signalled());
        assert!(!new_event.is_signalled());
        queue.signal(0x2000, 1);
        queue.remove(thread);
        assert!(queue.is_empty());
    }
}

#[derive(Debug)]
struct Waiter {
    thread: GuestThreadId,
    deadline: Option<u64>,
    event: WritableEventObject,
}

/// Lower numeric priorities wake first; equal priorities retain queue order.
/// Completed waits leave the active queues immediately, even if their thread
/// has not yet run its continuation. Deadlines use the runtime virtual clock.
#[derive(Debug, Default)]
pub struct PriorityAddressWaitQueue {
    queues: BTreeMap<u64, BTreeMap<i32, VecDeque<Waiter>>>,
    positions: BTreeMap<GuestThreadId, (u64, i32)>,
    deadlines: BTreeSet<(u64, GuestThreadId)>,
    completed: BTreeMap<GuestThreadId, AddressWaitResult>,
}

impl PriorityAddressWaitQueue {
    /// Returns `None` if this thread already owns a pending continuation.
    pub fn enqueue(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        priority: i32,
        deadline: Option<u64>,
    ) -> Option<ReadableEventObject> {
        if self.contains(thread) {
            return None;
        }
        let (event, readable) = EventObject::create_pair();
        self.queues
            .entry(address)
            .or_default()
            .entry(priority)
            .or_default()
            .push_back(Waiter {
                thread,
                deadline,
                event,
            });
        self.positions.insert(thread, (address, priority));
        if let Some(deadline) = deadline {
            self.deadlines.insert((deadline, thread));
        }
        Some(readable)
    }

    pub fn contains(&self, thread: GuestThreadId) -> bool {
        self.positions.contains_key(&thread) || self.completed.contains_key(&thread)
    }

    pub fn take_result(&mut self, thread: GuestThreadId) -> Option<AddressWaitResult> {
        self.completed.remove(&thread)
    }

    pub fn waiting_count(&self, address: u64) -> usize {
        self.queues
            .get(&address)
            .map_or(0, |priorities| priorities.values().map(VecDeque::len).sum())
    }

    /// Selects each active waiter once and removes it before publishing wakeup.
    pub fn signal(&mut self, address: u64, count: usize) {
        for _ in 0..count {
            let Some(priorities) = self.queues.get_mut(&address) else {
                break;
            };
            let mut entry = priorities.first_entry().expect("address queue is nonempty");
            let waiter = entry
                .get_mut()
                .pop_front()
                .expect("priority queue is nonempty");
            if entry.get().is_empty() {
                entry.remove();
            }
            if priorities.is_empty() {
                self.queues.remove(&address);
            }
            self.positions.remove(&waiter.thread);
            self.complete(waiter, AddressWaitResult::Signalled);
        }
    }

    /// Expire before signalling so an overdue thread cannot consume a signal
    /// while it is ready but has not yet executed its timeout continuation.
    pub fn expire(&mut self, now: u64) {
        while let Some(&(deadline, thread)) = self.deadlines.first() {
            if deadline > now {
                break;
            }
            let waiter = self
                .remove_active(thread)
                .expect("deadline owns an active wait");
            self.complete(waiter, AddressWaitResult::TimedOut);
        }
    }

    pub fn change_priority(&mut self, thread: GuestThreadId, priority: i32) {
        let Some(&(address, old_priority)) = self.positions.get(&thread) else {
            return;
        };
        if old_priority == priority {
            return;
        }
        let waiter = self
            .remove_active(thread)
            .expect("position owns an active wait");
        if let Some(deadline) = waiter.deadline {
            self.deadlines.insert((deadline, thread));
        }
        self.queues
            .entry(address)
            .or_default()
            .entry(priority)
            .or_default()
            .push_back(waiter);
        self.positions.insert(thread, (address, priority));
    }

    pub fn remove(&mut self, thread: GuestThreadId) {
        self.remove_active(thread);
        self.completed.remove(&thread);
    }

    pub fn len(&self) -> usize {
        self.positions.len() + self.completed.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn complete(&mut self, waiter: Waiter, result: AddressWaitResult) {
        if let Some(deadline) = waiter.deadline {
            self.deadlines.remove(&(deadline, waiter.thread));
        }
        self.completed.insert(waiter.thread, result);
        waiter.event.signal();
    }

    fn remove_active(&mut self, thread: GuestThreadId) -> Option<Waiter> {
        let (address, priority) = self.positions.remove(&thread)?;
        let priorities = self
            .queues
            .get_mut(&address)
            .expect("position owns an address queue");
        let queue = priorities
            .get_mut(&priority)
            .expect("position owns a priority queue");
        let index = queue
            .iter()
            .position(|waiter| waiter.thread == thread)
            .expect("position owns a waiter");
        let waiter = queue.remove(index).expect("waiter index exists");
        if queue.is_empty() {
            priorities.remove(&priority);
        }
        if priorities.is_empty() {
            self.queues.remove(&address);
        }
        if let Some(deadline) = waiter.deadline {
            self.deadlines.remove(&(deadline, thread));
        }
        Some(waiter)
    }
}
