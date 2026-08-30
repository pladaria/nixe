//! Fair shared execution and exclusive transition gate.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

#[derive(Debug)]
struct ExecutionGateState {
    active_shared: usize,
    transition_pending: bool,
    epoch: u64,
}

struct ExecutionGateInner {
    state: Mutex<ExecutionGateState>,
    changed: Condvar,
    transition_notifier: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl fmt::Debug for ExecutionGateInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionGateInner")
            .field("state", &self.state)
            .field("changed", &self.changed)
            .field(
                "has_transition_notifier",
                &self
                    .transition_notifier
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .is_some(),
            )
            .finish()
    }
}

/// Cloneable gate shared by CPU execution and canonical memory observers.
#[derive(Clone, Debug)]
pub struct ExecutionGate {
    inner: Arc<ExecutionGateInner>,
}

impl Default for ExecutionGate {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionGate {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ExecutionGateInner {
                state: Mutex::new(ExecutionGateState {
                    active_shared: 0,
                    transition_pending: false,
                    epoch: 1,
                }),
                changed: Condvar::new(),
                transition_notifier: Mutex::new(None),
            }),
        }
    }

    /// Installs the cold callback used to request prompt CPU safepoints when
    /// an external transition closes admission. The callback must be bounded;
    /// a panic is isolated so it cannot leave the transition gate closed.
    pub fn set_transition_notifier(&self, notifier: Option<Arc<dyn Fn() + Send + Sync>>) {
        *self
            .inner
            .transition_notifier
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = notifier;
    }

    /// Stable identity used to prove that an execution lease belongs to the
    /// same canonical memory owner as a direct CPU slice.
    #[must_use]
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.inner).addr()
    }

    /// Admits one bounded CPU execution slice.
    pub fn acquire_shared(&self) -> ExecutionSharedGuard {
        let mut state = self.lock_state();
        while state.transition_pending {
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.active_shared = state
            .active_shared
            .checked_add(1)
            .expect("shared execution holders are bounded by host workers");
        let epoch = state.epoch;
        ExecutionSharedGuard {
            gate: self.clone(),
            epoch,
        }
    }

    /// Closes admission and waits until every bounded CPU slice reaches its
    /// safepoint. Pending transitions cannot be overtaken by new readers.
    pub fn acquire_exclusive(&self) -> ExecutionTransitionGuard<'_> {
        let mut state = self.lock_state();
        while state.transition_pending {
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.transition_pending = true;
        let notify = state.active_shared != 0;
        drop(state);
        if notify {
            self.notify_transition();
        }
        let mut state = self.lock_state();
        while state.active_shared != 0 {
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        ExecutionTransitionGuard {
            gate: self,
            state: Some(state),
            committed: false,
        }
    }

    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.lock_state().epoch
    }

    #[must_use]
    pub fn transition_pending(&self) -> bool {
        self.lock_state().transition_pending
    }

    fn lock_state(&self) -> MutexGuard<'_, ExecutionGateState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn notify_transition(&self) {
        let notifier = self
            .inner
            .transition_notifier
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let Some(notifier) = notifier else {
            return;
        };
        let _ = catch_unwind(AssertUnwindSafe(|| notifier()));
    }
}

/// RAII proof that external mapping and ownership transitions are stable.
pub struct ExecutionSharedGuard {
    gate: ExecutionGate,
    epoch: u64,
}

impl ExecutionSharedGuard {
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl Drop for ExecutionSharedGuard {
    fn drop(&mut self) {
        let mut state = self.gate.lock_state();
        state.active_shared = state
            .active_shared
            .checked_sub(1)
            .expect("a shared execution guard is released exactly once");
        if state.active_shared == 0 {
            self.gate.inner.changed.notify_all();
        }
    }
}

/// Exclusive transition ownership. Calling [`Self::commit`] advances the
/// epoch; dropping an uncommitted preflight leaves it unchanged.
pub struct ExecutionTransitionGuard<'a> {
    gate: &'a ExecutionGate,
    state: Option<MutexGuard<'a, ExecutionGateState>>,
    committed: bool,
}

impl ExecutionTransitionGuard<'_> {
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for ExecutionTransitionGuard<'_> {
    fn drop(&mut self) {
        let mut state = self
            .state
            .take()
            .expect("an exclusive transition guard is dropped once");
        if self.committed {
            state.epoch = state
                .epoch
                .checked_add(1)
                .expect("execution transition epoch cannot exhaust in one host run");
        }
        state.transition_pending = false;
        self.gate.inner.changed.notify_all();
        drop(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn pending_transition_closes_admission_and_advances_only_when_committed() {
        let gate = ExecutionGate::new();
        let active = gate.acquire_shared();
        let worker_gate = gate.clone();
        let acquired = Arc::new(AtomicBool::new(false));
        let worker_acquired = Arc::clone(&acquired);
        let worker = std::thread::spawn(move || {
            let mut transition = worker_gate.acquire_exclusive();
            worker_acquired.store(true, Ordering::Release);
            transition.commit();
        });
        while !gate.transition_pending() {
            std::thread::yield_now();
        }
        let follower_gate = gate.clone();
        let follower = std::thread::spawn(move || follower_gate.acquire_shared().epoch());
        std::thread::sleep(Duration::from_millis(5));
        assert!(!acquired.load(Ordering::Acquire));
        drop(active);
        worker.join().unwrap();
        assert_eq!(follower.join().unwrap(), 2);
        assert_eq!(gate.epoch(), 2);
    }

    #[test]
    fn abandoned_transition_does_not_advance_the_epoch() {
        let gate = ExecutionGate::new();
        drop(gate.acquire_exclusive());
        assert_eq!(gate.epoch(), 1);
        assert!(!gate.transition_pending());
    }

    #[test]
    fn pending_transition_requests_one_prompt_safepoint_before_waiting() {
        let gate = ExecutionGate::new();
        let active = gate.acquire_shared();
        let (notified_tx, notified_rx) = std::sync::mpsc::channel();
        gate.set_transition_notifier(Some(Arc::new(move || {
            notified_tx.send(()).unwrap();
        })));
        let worker_gate = gate.clone();
        let worker = std::thread::spawn(move || {
            let mut transition = worker_gate.acquire_exclusive();
            transition.commit();
        });

        while !gate.transition_pending() {
            std::thread::yield_now();
        }
        notified_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(active);
        worker.join().unwrap();
    }
}
