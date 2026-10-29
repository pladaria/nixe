//! Address arbiter payloads share the canonical wait generation and ordered kernel nodes.
use super::*;
#[derive(Debug)]
pub struct AddressSignalTargets(Vec<(GuestThreadId, WaitEffect)>);
impl AddressSignalTargets {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
impl AddressWaitState {
    pub fn enqueue_address_wait(
        &mut self,
        address: u64,
        thread: GuestThreadId,
    ) -> Result<Arc<KernelWait>, SchedulerError> {
        let (wait, _) = self.kernel().1.begin_ordered_wait(
            thread,
            OrderedWaitKind::AddressArbiter,
            address,
            false,
        )?;
        let previous = self.arbiter_waiters.insert(thread, wait.clone());
        debug_assert!(previous.is_none());
        Ok(wait)
    }

    pub fn signal_targets(&self, address: u64, count: usize) -> (AddressSignalTargets, bool) {
        let (process, kernel) = self.kernel();
        let (claims, more) = kernel.claim_ordered_waiters(
            *process,
            OrderedWaitKind::AddressArbiter,
            address,
            count,
            false,
        );
        (AddressSignalTargets(claims), more)
    }
    pub fn complete_signal(&mut self, targets: &AddressSignalTargets) {
        for (thread, effect) in &targets.0 {
            if self
                .arbiter_waiters
                .get(thread)
                .is_some_and(|waiter| Arc::ptr_eq(waiter, effect.registration()))
            {
                let (_, notifications) = self
                    .kernel()
                    .1
                    .resolve_wait_deferred(effect.registration().token(), WaitOutcome::Ready);
                self.notifications.push(notifications);
            }
        }
    }
    pub fn remove_address_wait(&mut self, thread: GuestThreadId) {
        if let Some(waiter) = self.arbiter_waiters.remove(&thread) {
            self.kernel().1.remove_waiter(thread);
            drop(waiter);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::address_wait::test_registry;
    #[test]
    fn priority_and_fifo_choose_the_real_signal_generation() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        let first = GuestThreadId::new(1);
        let second = GuestThreadId::new(2);
        let a = registry.lock().enqueue_address_wait(0x1000, first).unwrap();
        let b = registry
            .lock()
            .enqueue_address_wait(0x1000, second)
            .unwrap();
        kernel.set_priority(second, 10, None).unwrap();
        let (targets, more) = registry.lock().signal_targets(0x1000, 1);
        assert!(more);
        registry.lock().complete_signal(&targets);
        drop(targets);
        assert_eq!(a.outcome(), None);
        assert_eq!(b.outcome(), Some(WaitOutcome::Ready));
    }
    #[test]
    fn a_failed_conditional_signal_preserves_the_ordered_wait() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        let thread = GuestThreadId::new(1);
        let wait = registry
            .lock()
            .enqueue_address_wait(0x1000, thread)
            .unwrap();
        let (targets, _) = registry.lock().signal_targets(0x1000, 1);
        // The word compare failed; no completion is published.
        drop(targets);
        assert_eq!(wait.outcome(), None);
        assert_eq!(
            kernel.first_ordered_waiter(ProcessId::new(1), OrderedWaitKind::AddressArbiter, 0x1000),
            Some(thread)
        );
    }
    #[test]
    fn timeout_uses_the_shared_heap_and_removes_the_ordered_node() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        let thread = GuestThreadId::new(1);
        let wait = registry
            .lock()
            .enqueue_address_wait(0x1000, thread)
            .unwrap();
        kernel.arm_deadline(&wait, 100);
        kernel.advance_time(100);
        assert_eq!(wait.outcome(), Some(WaitOutcome::TimedOut));
        registry.lock().remove_address_wait(thread);
        assert!(
            registry
                .lock()
                .signal_targets(0x1000, usize::MAX)
                .0
                .is_empty()
        );
        assert_eq!(registry.lock().waiter_count(), 0);
    }
}
