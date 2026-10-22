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
        let (selected, _) = queue.signal_targets(0x1000, 1);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        let (selected, _) = queue.signal_targets(0x1000, 1);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        assert!(first_event.is_signalled());
        assert!(second_event.is_signalled());
        assert!(!low_event.is_signalled());
        assert_eq!(queue.waiting_count(0x1000), 1);
        assert_eq!(queue.take_result(first), Some(AddressWaitResult::Signalled));
        assert_eq!(
            queue.take_result(second),
            Some(AddressWaitResult::Signalled)
        );
        let (selected, _) = queue.signal_targets(0x1000, usize::MAX);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
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
        let (selected, _) = queue.signal_targets(0x1000, 1);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        assert!(second_event.is_signalled());
        assert!(!first_event.is_signalled());
        queue.change_priority(second, 50); // Completion is no longer queued.
        queue.expire(100, |_| false);
        for wake in queue.take_wakes() {
            wake.signal();
        }
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
        queue.expire(10, |_| false);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        assert!(expired_event.is_signalled());
        assert_eq!(queue.waiting_count(0x1000), 1);
        let (selected, _) = queue.signal_targets(0x1000, 1);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        queue.expire(20, |_| false);
        for wake in queue.take_wakes() {
            wake.signal();
        }
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
        queue.expire(10, |_| false);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        let (selected, _) = queue.signal_targets(0x1000, usize::MAX);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        assert!(!old_event.is_signalled());
        assert!(!new_event.is_signalled());
        let (selected, _) = queue.signal_targets(0x2000, 1);
        queue.complete_signal(&selected);
        for wake in queue.take_wakes() {
            wake.signal();
        }
        queue.remove(thread);
        assert!(queue.is_empty());
    }
}

#[derive(Debug)]
pub(super) struct Waiter {
    thread: GuestThreadId,
    deadline: Option<u64>,
    event: WritableEventObject,
    checking: bool,
    timeout_deferred: bool,
}

#[derive(Debug)]
pub struct AddressSignalTargets(Vec<(GuestThreadId, WritableEventObject)>);
impl AddressSignalTargets {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
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
    wakes: Vec<WritableEventObject>,
    retired: Vec<Waiter>,
}

impl PriorityAddressWaitQueue {
    pub(super) fn take_wakes(&mut self) -> Vec<WritableEventObject> {
        std::mem::take(&mut self.wakes)
    }
    pub(super) fn take_retired(&mut self) -> Vec<Waiter> {
        std::mem::take(&mut self.retired)
    }

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
                checking: false,
                timeout_deferred: false,
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

    pub fn begin_check(&mut self, thread: GuestThreadId) {
        if let Some(waiter) = self.waiter_mut(thread) {
            waiter.checking = true;
        }
    }
    pub fn finish_check(&mut self, thread: GuestThreadId, now: u64, zero_timeout: bool) {
        let Some(waiter) = self.waiter_mut(thread) else {
            return;
        };
        waiter.checking = false;
        if zero_timeout
            || waiter.timeout_deferred
            || waiter.deadline.is_some_and(|deadline| deadline <= now)
        {
            let waiter = self.remove_active(thread).unwrap();
            self.complete(waiter, AddressWaitResult::TimedOut);
        }
    }
    /// Capture current priority/FIFO selection before its conditional word
    /// operation. Late priority changes cannot select a different generation.
    pub fn signal_targets(&self, address: u64, count: usize) -> (AddressSignalTargets, bool) {
        let mut eligible = self
            .queues
            .get(&address)
            .into_iter()
            .flat_map(|priorities| priorities.values())
            .flatten()
            .filter(|waiter| !waiter.checking);
        let selected = eligible
            .by_ref()
            .take(count)
            .map(|waiter| (waiter.thread, waiter.event.clone()))
            .collect();
        (AddressSignalTargets(selected), eligible.next().is_some())
    }
    pub fn complete_signal(&mut self, targets: &AddressSignalTargets) {
        for (thread, event) in &targets.0 {
            if self
                .waiter_mut(*thread)
                .is_some_and(|waiter| !waiter.checking && waiter.event.same_event(event))
            {
                let waiter = self.remove_active(*thread).unwrap();
                self.complete(waiter, AddressWaitResult::Signalled);
            }
        }
    }
    /// Timer delivery during an ordered word operation records expiry without
    /// allowing guest code to run ahead of that operation's result.
    pub(super) fn defer_timeout(
        &mut self,
        thread: GuestThreadId,
        locked: impl FnOnce(u64) -> bool,
    ) -> Option<u64> {
        let &(address, _) = self.positions.get(&thread)?;
        let locked = locked(address);
        let waiter = self.waiter_mut(thread).unwrap();
        if waiter.checking || locked {
            waiter.timeout_deferred = true;
            if let Some(deadline) = waiter.deadline {
                self.deadlines.remove(&(deadline, thread));
            }
            Some(address)
        } else {
            None
        }
    }
    pub fn expire(&mut self, now: u64, mut locked: impl FnMut(u64) -> bool) {
        while let Some(&(deadline, thread)) = self.deadlines.first() {
            if deadline > now {
                break;
            }
            self.deadlines.remove(&(deadline, thread));
            if self.defer_timeout(thread, &mut locked).is_none() {
                let waiter = self
                    .remove_active(thread)
                    .expect("deadline owns an active wait");
                self.complete(waiter, AddressWaitResult::TimedOut);
            }
        }
    }
    pub(super) fn publish_deferred_timeouts(&mut self, address: u64) {
        let expired: Vec<_> = self
            .queues
            .get(&address)
            .into_iter()
            .flat_map(|priorities| priorities.values())
            .flatten()
            .filter(|waiter| !waiter.checking && waiter.timeout_deferred)
            .map(|waiter| waiter.thread)
            .collect();
        for thread in expired {
            let waiter = self.remove_active(thread).unwrap();
            self.complete(waiter, AddressWaitResult::TimedOut);
        }
    }
    fn waiter_mut(&mut self, thread: GuestThreadId) -> Option<&mut Waiter> {
        let &(address, priority) = self.positions.get(&thread)?;
        self.queues
            .get_mut(&address)?
            .get_mut(&priority)?
            .iter_mut()
            .find(|waiter| waiter.thread == thread)
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
        if let Some(deadline) = waiter.deadline
            && !waiter.timeout_deferred
        {
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
        if let Some(waiter) = self.remove_active(thread) {
            self.retired.push(waiter);
        }
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
        self.wakes.push(waiter.event);
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
