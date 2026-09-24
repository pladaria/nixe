//! Fair shared execution and exclusive transition gate.

use std::fmt;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use crate::MemoryInvalidationKind;

/// Cold engine handshake for semantic mutations, not read-only captures.
pub trait ExecutionMutationObserver: Send + Sync {
    /// Close engine admission, register exact targets and drain execution and
    /// unlinks before returning. No gate or memory mutex is held by the caller.
    fn begin(
        self: Arc<Self>,
        changes: &[MemoryInvalidationKind],
    ) -> Result<Box<dyn ExecutionMutation>, ExecutionMutationError>;
}

/// Holds engine admission closed until the memory authority has released its
/// locks and published the mutation's invalidation records. Drop releases the
/// hold, including abandoned preflights. A completion failure must disable the
/// engine; it must never silently reopen admission to stale code.
pub trait ExecutionMutation {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionMutationError(pub Box<str>);

impl fmt::Display for ExecutionMutationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
impl std::error::Error for ExecutionMutationError {}

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
    mutation_observer: Mutex<Option<Arc<dyn ExecutionMutationObserver>>>,
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
                mutation_observer: Mutex::new(None),
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

    /// Bind once before execution. An installed safety handshake cannot be
    /// replaced or removed while the memory owner remains usable.
    pub fn set_mutation_observer(
        &self,
        observer: Arc<dyn ExecutionMutationObserver>,
    ) -> Result<(), ExecutionMutationError> {
        let state = self.lock_state();
        let mut installed = self
            .inner
            .mutation_observer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.transition_pending || state.active_shared != 0 || installed.is_some() {
            return Err(ExecutionMutationError(
                "memory mutation observer requires an idle, unbound gate".into(),
            ));
        }
        *installed = Some(observer);
        Ok(())
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
        let transition = self.close_admission();
        self.wait_shared();
        transition
    }

    /// Mandatory pre/post handoff for a semantic mutation. Unlike a capture,
    /// this notifies the engine even when no execution lease is active. The
    /// caller must not own a lease/engine epoch or hold any memory/cache lock.
    pub fn acquire_mutation(
        &self,
        changes: &[MemoryInvalidationKind],
    ) -> Result<ExecutionMutationGuard<'_>, ExecutionMutationError> {
        let transition = self.close_admission();
        self.coordinate_mutation(transition, Some(changes))
    }

    /// Copy instructions under memory exclusion. A read-only attempt may
    /// discover that dirty tracking needs rearming; release all memory locks
    /// and this hold, then retry with `arm_tracking` to drain the engine too.
    /// Already-armed pages need no protection mutation or compiler cancellation.
    pub fn acquire_capture(
        &self,
        arm_tracking: bool,
    ) -> Result<ExecutionMutationGuard<'_>, ExecutionMutationError> {
        let transition = self.close_admission();
        self.coordinate_mutation(transition, arm_tracking.then_some(&[]))
    }

    /// Discover executable write targets with capture admission closed, before
    /// waiting for executing readers. Ordinary data needs only memory exclusion.
    /// Discovery must be bounded and must not acquire this gate recursively.
    /// The caller must not own an execution lease/epoch or memory/cache lock.
    pub fn acquire_write(
        &self,
        targets: impl FnOnce() -> Vec<MemoryInvalidationKind>,
    ) -> Result<ExecutionMutationGuard<'_>, ExecutionMutationError> {
        let transition = self.close_admission();
        let changes = targets();
        self.coordinate_mutation(
            transition,
            (!changes.is_empty()).then_some(changes.as_slice()),
        )
    }

    /// Instruction-cache invalidation changes derived code, not memory bytes
    /// or mappings. Without a bound code owner, the caller only publishes its
    /// log record and needs no exclusive memory lease. With an owner, discovery
    /// and quiescence follow the ordinary mutation protocol. Bind the owner
    /// before sharing memory; a bound caller must release its own epoch/lease.
    pub fn acquire_code_invalidation(
        &self,
        targets: impl FnOnce() -> Vec<MemoryInvalidationKind>,
    ) -> Result<Option<ExecutionMutationGuard<'_>>, ExecutionMutationError> {
        if self
            .inner
            .mutation_observer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_none()
        {
            return Ok(None);
        }
        self.acquire_write(targets).map(Some)
    }

    fn coordinate_mutation<'a>(
        &'a self,
        transition: ExecutionTransitionGuard<'a>,
        changes: Option<&[MemoryInvalidationKind]>,
    ) -> Result<ExecutionMutationGuard<'a>, ExecutionMutationError> {
        let observer = self
            .inner
            .mutation_observer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let participant = observer
            .zip(changes)
            .map(|(observer, changes)| observer.begin(changes))
            .transpose()?;
        self.wait_shared();
        Ok(ExecutionMutationGuard {
            participant,
            transition,
            mutation: changes.is_some(),
        })
    }

    fn close_admission(&self) -> ExecutionTransitionGuard<'_> {
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
        ExecutionTransitionGuard {
            gate: self,
            committed: false,
            marker: PhantomData,
        }
    }

    fn wait_shared(&self) {
        let mut state = self.lock_state();
        while state.active_shared != 0 {
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
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
        // Only an exclusive transition waits for readers to drain. Its pending
        // flag and this last-reader check share the condvar's predicate mutex.
        if state.active_shared == 0 && state.transition_pending {
            self.gate.inner.changed.notify_all();
        }
    }
}

/// Exclusive transition ownership. Calling [`Self::commit`] advances the
/// epoch; dropping an uncommitted preflight leaves it unchanged.
pub struct ExecutionTransitionGuard<'a> {
    gate: &'a ExecutionGate,
    committed: bool,
    // Preserve the guard's thread affinity without holding the gate mutex
    // while memory work or engine coordination runs.
    marker: PhantomData<MutexGuard<'a, ExecutionGateState>>,
}

impl ExecutionTransitionGuard<'_> {
    /// Whether this guard excludes execution on the specified backing store.
    #[must_use]
    pub fn protects(&self, gate: &ExecutionGate) -> bool {
        self.gate.identity() == gate.identity()
    }

    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for ExecutionTransitionGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.lock_state();
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

/// Field order releases the engine hold before reopening memory admission.
/// The mutation's memory locks must be declared after (and drop before) this.
pub struct ExecutionMutationGuard<'a> {
    participant: Option<Box<dyn ExecutionMutation>>,
    transition: ExecutionTransitionGuard<'a>,
    mutation: bool,
}

impl ExecutionMutationGuard<'_> {
    pub(crate) fn permits_tracking(&self) -> bool {
        self.mutation
    }

    /// Proves that this hold excludes execution on the backing store's gate.
    pub fn protects(&self, gate: &ExecutionGate) -> bool {
        self.transition.protects(gate)
    }

    pub fn commit(&mut self) {
        self.transition.commit();
    }
}

impl Drop for ExecutionMutationGuard<'_> {
    fn drop(&mut self) {
        drop(self.participant.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    struct Observer {
        gate: std::sync::Weak<ExecutionGateInner>,
        began: std::sync::mpsc::Sender<usize>,
        finished: Arc<AtomicBool>,
        reject: bool,
    }

    impl ExecutionMutationObserver for Observer {
        fn begin(
            self: Arc<Self>,
            changes: &[MemoryInvalidationKind],
        ) -> Result<Box<dyn ExecutionMutation>, ExecutionMutationError> {
            // Both calls acquire gate state. The callback must run unlocked.
            let gate = ExecutionGate {
                inner: self.gate.upgrade().unwrap(),
            };
            assert!(gate.transition_pending());
            assert_eq!(gate.epoch(), 1);
            self.began.send(changes.len()).unwrap();
            if self.reject {
                return Err(ExecutionMutationError("engine refused mutation".into()));
            }
            Ok(Box::new(Participant(self)))
        }
    }

    struct Participant(Arc<Observer>);
    impl ExecutionMutation for Participant {}
    impl Drop for Participant {
        fn drop(&mut self) {
            let gate = ExecutionGate {
                inner: self.0.gate.upgrade().unwrap(),
            };
            assert!(gate.transition_pending());
            assert_eq!(gate.epoch(), 1);
            self.0.finished.store(true, Ordering::Release);
        }
    }

    #[test]
    fn idle_semantic_mutation_has_an_unlocked_handshake_but_read_only_capture_does_not() {
        let gate = ExecutionGate::new();
        let (send, receive) = std::sync::mpsc::channel();
        let finished = Arc::new(AtomicBool::new(false));
        let observer = Arc::new(Observer {
            gate: Arc::downgrade(&gate.inner),
            began: send,
            finished: finished.clone(),
            reject: false,
        });
        gate.set_mutation_observer(observer.clone()).unwrap();
        assert!(gate.set_mutation_observer(observer).is_err());
        drop(gate.acquire_exclusive());
        assert!(receive.try_recv().is_err());
        let mut mutation = gate
            .acquire_mutation(&[MemoryInvalidationKind::InstructionCache {
                address_space: crate::AddressSpaceId::new(1),
            }])
            .unwrap();
        assert_eq!(receive.recv().unwrap(), 1);
        assert!(!finished.load(Ordering::Acquire));
        mutation.commit();
        drop(mutation);
        assert!(finished.load(Ordering::Acquire));
        assert_eq!(gate.epoch(), 2);
        assert!(!gate.transition_pending());
    }

    #[test]
    fn semantic_handshake_starts_before_shared_drain_and_error_never_exposes_a_mutation_guard() {
        for reject in [false, true] {
            let gate = ExecutionGate::new();
            let (send, receive) = std::sync::mpsc::channel();
            let finished = Arc::new(AtomicBool::new(false));
            gate.set_mutation_observer(Arc::new(Observer {
                gate: Arc::downgrade(&gate.inner),
                began: send,
                finished: finished.clone(),
                reject,
            }))
            .unwrap();
            let active = gate.acquire_shared();
            let mutated = Arc::new(AtomicBool::new(false));
            let writer_gate = gate.clone();
            let writer_mutated = mutated.clone();
            let worker = std::thread::spawn(move || {
                let mutation = writer_gate.acquire_mutation(&[]);
                if reject {
                    assert!(
                        matches!(mutation, Err(ExecutionMutationError(detail)) if &*detail == "engine refused mutation")
                    );
                } else {
                    let _mutation = mutation.unwrap();
                    writer_mutated.store(true, Ordering::Release);
                }
            });
            receive.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(!mutated.load(Ordering::Acquire));
            drop(active);
            worker.join().unwrap();
            assert_eq!(mutated.load(Ordering::Acquire), !reject);
            assert_eq!(finished.load(Ordering::Acquire), !reject);
            assert!(!gate.transition_pending());
            assert_eq!(gate.epoch(), 1);
        }
    }

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
    fn transition_waits_for_the_last_shared_reader() {
        let gate = ExecutionGate::new();
        let first = gate.acquire_shared();
        let last = gate.acquire_shared();
        let (pending_tx, pending_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        gate.set_transition_notifier(Some(Arc::new(move || {
            pending_tx.send(()).unwrap();
        })));
        let worker_gate = gate.clone();
        let worker = std::thread::spawn(move || {
            let _transition = worker_gate.acquire_exclusive();
            done_tx.send(()).unwrap();
        });
        pending_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(first);
        assert_eq!(gate.lock_state().active_shared, 1);
        assert_eq!(
            done_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        );
        drop(last);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        // Once the writer reopens admission, ordinary reader release remains
        // sufficient for the next writer even without an outstanding waiter.
        drop(gate.acquire_shared());
        drop(gate.acquire_exclusive());
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
