//! Address-arbiter event/deadline ownership; ordering uses the kernel's wait nodes.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressWaitResult {
    Signalled,
    TimedOut,
}

#[derive(Debug)]
pub(super) struct Waiter {
    address: u64,
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

impl AddressWaitState {
    /// The accepted wait is not signalable until its guest-word check completes.
    pub fn enqueue_address_wait(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        deadline: Option<u64>,
    ) -> Option<ReadableEventObject> {
        if self.contains_address_wait(thread) {
            return None;
        }
        let (event, readable) = EventObject::create_pair();
        self.arbiter_waiters.insert(
            thread,
            Waiter {
                address,
                deadline,
                event,
                checking: true,
                timeout_deferred: false,
            },
        );
        if let Some(deadline) = deadline {
            self.arbiter_deadlines.insert((deadline, thread));
        }
        Some(readable)
    }
    pub fn contains_address_wait(&self, thread: GuestThreadId) -> bool {
        self.arbiter_waiters.contains_key(&thread) || self.arbiter_completed.contains_key(&thread)
    }
    pub fn take_address_result(&mut self, thread: GuestThreadId) -> Option<AddressWaitResult> {
        self.arbiter_completed.remove(&thread)
    }
    pub fn finish_address_check(
        &mut self,
        thread: GuestThreadId,
        now: u64,
        zero_timeout: bool,
    ) -> Result<(), SchedulerError> {
        let Some(waiter) = self.arbiter_waiters.get(&thread) else {
            return Ok(());
        };
        if zero_timeout
            || waiter.timeout_deferred
            || waiter.deadline.is_some_and(|deadline| deadline <= now)
        {
            let waiter = self.remove_active_address_wait(thread).unwrap();
            self.complete_address_wait(thread, waiter, AddressWaitResult::TimedOut);
        } else {
            self.kernel().1.register_ordered_wait(
                thread,
                OrderedWaitKind::AddressArbiter,
                waiter.address,
            )?;
            self.arbiter_waiters.get_mut(&thread).unwrap().checking = false;
        }
        Ok(())
    }
    pub fn signal_targets(&self, address: u64, count: usize) -> (AddressSignalTargets, bool) {
        let (process, scheduler) = self.kernel();
        let mut ids = scheduler.ordered_waiters(
            *process,
            OrderedWaitKind::AddressArbiter,
            address,
            count.saturating_add(1),
        );
        let more = ids.len() > count;
        if more {
            ids.pop();
        }
        let selected = ids
            .into_iter()
            .map(|id| (id, self.arbiter_waiters[&id].event.clone()))
            .collect();
        (AddressSignalTargets(selected), more)
    }
    pub fn complete_signal(&mut self, targets: &AddressSignalTargets) {
        for (thread, event) in &targets.0 {
            if self
                .arbiter_waiters
                .get(thread)
                .is_some_and(|w| !w.checking && w.event.same_event(event))
            {
                let waiter = self.remove_active_address_wait(*thread).unwrap();
                self.complete_address_wait(*thread, waiter, AddressWaitResult::Signalled);
            }
        }
    }
    pub(super) fn defer_arbiter_timeout(&mut self, thread: GuestThreadId) -> Option<u64> {
        let waiter = self.arbiter_waiters.get_mut(&thread)?;
        if waiter.checking || self.operations.contains_key(&waiter.address) {
            waiter.timeout_deferred = true;
            self.arbiter_deferred.insert((waiter.address, thread));
            if let Some(deadline) = waiter.deadline {
                self.arbiter_deadlines.remove(&(deadline, thread));
            }
            Some(waiter.address)
        } else {
            None
        }
    }
    pub fn expire_address_waits(&mut self, now: u64) {
        while let Some(&(deadline, thread)) = self.arbiter_deadlines.first() {
            if deadline > now {
                break;
            }
            self.arbiter_deadlines.remove(&(deadline, thread));
            if let Some(address) = self.defer_arbiter_timeout(thread) {
                self.deferred_addresses.insert(address);
            } else {
                let waiter = self
                    .remove_active_address_wait(thread)
                    .expect("deadline owns an active wait");
                self.complete_address_wait(thread, waiter, AddressWaitResult::TimedOut);
            }
        }
    }
    pub(super) fn publish_deferred_timeouts(&mut self, address: u64) {
        let expired: Vec<_> = self
            .arbiter_deferred
            .range((address, GuestThreadId::new(0))..=(address, GuestThreadId::new(u64::MAX)))
            .filter_map(|&(_, thread)| (!self.arbiter_waiters[&thread].checking).then_some(thread))
            .collect();
        for thread in expired {
            let waiter = self.remove_active_address_wait(thread).unwrap();
            self.complete_address_wait(thread, waiter, AddressWaitResult::TimedOut);
        }
    }
    pub fn remove_address_wait(&mut self, thread: GuestThreadId) {
        if let Some(waiter) = self.remove_active_address_wait(thread) {
            self.arbiter_retired.push(waiter);
        }
        self.arbiter_completed.remove(&thread);
    }
    fn remove_active_address_wait(&mut self, thread: GuestThreadId) -> Option<Waiter> {
        let waiter = self.arbiter_waiters.remove(&thread)?;
        self.kernel().1.remove_waiter(thread);
        if let Some(deadline) = waiter.deadline {
            self.arbiter_deadlines.remove(&(deadline, thread));
        }
        self.arbiter_deferred.remove(&(waiter.address, thread));
        Some(waiter)
    }
    fn complete_address_wait(
        &mut self,
        thread: GuestThreadId,
        waiter: Waiter,
        result: AddressWaitResult,
    ) {
        self.arbiter_completed.insert(thread, result);
        self.wakes.push(waiter.event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address_wait::test_registry;

    #[test]
    fn canonical_priority_changes_and_fifo_determine_the_real_signal_event() {
        let registry = test_registry();
        let (first, second, donor) = (
            GuestThreadId::new(1),
            GuestThreadId::new(2),
            GuestThreadId::new(3),
        );
        let (a, b) = {
            let mut state = registry.lock();
            let a = state.enqueue_address_wait(0x1000, first, None).unwrap();
            state.finish_address_check(first, 0, false).unwrap();
            let b = state.enqueue_address_wait(0x1000, second, None).unwrap();
            state.finish_address_check(second, 0, false).unwrap();
            state.kernel().1.set_priority(donor, 10, None).unwrap();
            // This is the actual mutex insertion, not a priority-refresh message.
            state.enqueue(0x2000, second, donor, 3).unwrap();
            let (targets, more) = state.signal_targets(0x1000, 1);
            assert!(more);
            state.complete_signal(&targets);
            (a, b)
        };
        assert!(!a.is_signalled());
        assert!(b.is_signalled());
        let mut state = registry.lock();
        assert_eq!(
            state.take_address_result(second),
            Some(AddressWaitResult::Signalled)
        );
        state.remove(0x2000, donor);
        assert_eq!(
            state.kernel().1.thread(second).unwrap().effective_priority,
            20
        );
        let (targets, more) = state.signal_targets(0x1000, usize::MAX);
        assert!(!more);
        state.complete_signal(&targets);
        state.complete_signal(&targets);
        drop(state);
        assert!(a.is_signalled());
        assert_eq!(registry.lock().waiter_count(), 1);
    }

    #[test]
    fn removing_a_wait_cancels_its_deadline_and_stale_signal_generation() {
        let registry = test_registry();
        let thread = GuestThreadId::new(1);
        let (old, next) = {
            let mut state = registry.lock();
            let old = state
                .enqueue_address_wait(0x1000, thread, Some(10))
                .unwrap();
            assert!(state.enqueue_address_wait(0x1000, thread, None).is_none());
            state.finish_address_check(thread, 0, false).unwrap();
            let (targets, _) = state.signal_targets(0x1000, 1);
            state.remove_address_wait(thread);
            let next = state
                .enqueue_address_wait(0x1000, thread, Some(20))
                .unwrap();
            state.finish_address_check(thread, 0, false).unwrap();
            state.complete_signal(&targets);
            state.expire_address_waits(10);
            (old, next)
        };
        assert!(!old.is_signalled());
        assert!(!next.is_signalled());
        registry.lock().expire_address_waits(20);
        assert!(next.is_signalled());
        assert_eq!(
            registry.lock().take_address_result(thread),
            Some(AddressWaitResult::TimedOut)
        );
        assert_eq!(registry.lock().waiter_count(), 0);
    }

    #[test]
    fn expired_waiters_leave_canonical_selection_before_later_signals() {
        let registry = test_registry();
        let (expired, active) = (GuestThreadId::new(1), GuestThreadId::new(2));
        let (a, b) = {
            let mut state = registry.lock();
            let a = state
                .enqueue_address_wait(0x1000, expired, Some(10))
                .unwrap();
            state.finish_address_check(expired, 0, false).unwrap();
            let b = state
                .enqueue_address_wait(0x1000, active, Some(20))
                .unwrap();
            state.finish_address_check(active, 0, false).unwrap();
            state.expire_address_waits(10);
            let (targets, more) = state.signal_targets(0x1000, 1);
            assert!(!more);
            assert_eq!(targets.len(), 1);
            state.complete_signal(&targets);
            state.expire_address_waits(20);
            (a, b)
        };
        assert!(a.is_signalled() && b.is_signalled());
        let mut state = registry.lock();
        assert_eq!(
            state.take_address_result(expired),
            Some(AddressWaitResult::TimedOut)
        );
        assert_eq!(
            state.take_address_result(active),
            Some(AddressWaitResult::Signalled)
        );
        assert_eq!(state.waiter_count(), 0);
    }
}
