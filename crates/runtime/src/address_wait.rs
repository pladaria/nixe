//! Console-neutral ownership for address-keyed wait queues.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};

use nixe_scheduler::{
    GuestThreadId, LockKey, OrderedWaitKind, ProcessId, SchedulerError, SchedulerState,
};

use crate::{EventObject, ReadableEventObject, WritableEventObject};

mod arbiter;
pub use arbiter::{AddressSignalTargets, AddressWaitResult};

#[derive(Clone, Debug)]
struct AddressWaiter {
    address: u64,
    value: u32,
    lock_address: Option<u64>,
    phase: WaitPhase,
    lock_memory: Option<nixe_memory::AllocationRange>,
    writable: WritableEventObject,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitPhase {
    Registering,
    RegisteringTimedOut,
    ConditionSignalled,
    ConditionSignalledTimedOut,
    Waiting,
    Handoff,
    HandoffTimedOut,
    Resolved(AddressWaitCompletion),
}
impl WaitPhase {
    fn completion(self) -> Option<AddressWaitCompletion> {
        if let Self::Resolved(result) = self {
            Some(result)
        } else {
            None
        }
    }
}

/// Retained identity of an accepted condition wait during its unlock effects.
#[derive(Debug)]
pub struct ConditionWait {
    key: u64,
    thread: GuestThreadId,
    event: ReadableEventObject,
}
impl ConditionWait {
    pub fn event(&self) -> ReadableEventObject {
        self.event.clone()
    }
}
#[derive(Debug)]
pub struct ConditionClaim {
    key: u64,
    pub thread: GuestThreadId,
    pub tag: u32,
    pub mutex: u64,
    pub memory: nixe_memory::AllocationRange,
    event: WritableEventObject,
}
#[derive(Debug)]
pub enum ConditionSignal {
    PendingRelease,
    Reacquire(ConditionClaim),
}
#[derive(Clone, Copy, Debug)]
pub enum ConditionAcquire {
    Acquired,
    Contended(GuestThreadId),
    InvalidOwner,
    InvalidMemory,
}

/// Completion payload for a mutex or condition-variable address wait.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressWaitCompletion {
    Success,
    InvalidOwner,
    InvalidMemory,
    TimedOut,
}

/// A selected ownership handoff retains the actual wait event as its token.
/// Closing/reusing a wait on the same address cannot retarget its completion.
#[derive(Debug)]
pub struct MutexHandoff {
    address: u64,
    next: Option<(GuestThreadId, WritableEventObject)>,
    pub next_tag: Option<u32>,
    pub more_waiters: bool,
}

#[derive(Debug)]
struct AddressOperation {
    thread: GuestThreadId,
    event: Option<(WritableEventObject, ReadableEventObject)>,
}
/// An ordered operation for one process-local arbiter word. No host lock is
/// held while its physical access is pending; other addresses keep progressing.
pub struct AddressOperationGuard {
    registry: Arc<AddressWaitRegistry>,
    address: u64,
    thread: GuestThreadId,
}
impl Drop for AddressOperationGuard {
    fn drop(&mut self) {
        let retired = {
            let mut state = self.registry.lock();
            if state
                .operations
                .get(&self.address)
                .is_none_or(|operation| operation.thread != self.thread)
            {
                return;
            }
            let retired = state.operations.remove(&self.address).unwrap();
            if state.deferred_addresses.remove(&self.address) {
                state.publish_deferred_timeouts(self.address);
            }
            retired
        };
        if let Some((signal, _)) = retired.event {
            signal.signal();
        }
    }
}

/// Shared kernel wait ownership. Completed state is committed under the lock;
/// event notifications and retired observers are released after unlocking.
#[derive(Debug, Default)]
pub struct AddressWaitRegistry {
    state: Mutex<AddressWaitState>,
}
pub struct AddressWaitGuard<'a> {
    state: Option<MutexGuard<'a, AddressWaitState>>,
}
impl AddressWaitRegistry {
    pub(crate) fn bind(&self, process: ProcessId, scheduler: Arc<SchedulerState>) {
        self.lock()
            .kernel
            .set((process, scheduler))
            .expect("address waits already bound");
    }

    pub async fn address_operation(
        self: &Arc<Self>,
        address: u64,
        thread: GuestThreadId,
        clock: &crate::VirtualClock,
    ) -> AddressOperationGuard {
        loop {
            let now = clock.scheduler_time_ns();
            let pending = {
                let mut state = self.lock();
                if let Some(operation) = state.operations.get_mut(&address) {
                    let (_, event) = operation.event.get_or_insert_with(EventObject::create_pair);
                    Some(event.clone())
                } else {
                    state.expire_address_waits(now);
                    state.operations.insert(
                        address,
                        AddressOperation {
                            thread,
                            event: None,
                        },
                    );
                    None
                }
            };
            if let Some(event) = pending {
                event.wait_async().await;
            } else {
                return AddressOperationGuard {
                    registry: self.clone(),
                    address,
                    thread,
                };
            }
        }
    }
    pub fn lock(&self) -> AddressWaitGuard<'_> {
        AddressWaitGuard {
            state: Some(self.state.lock().unwrap()),
        }
    }
}
impl Deref for AddressWaitGuard<'_> {
    type Target = AddressWaitState;
    fn deref(&self) -> &Self::Target {
        self.state.as_deref().unwrap()
    }
}
impl DerefMut for AddressWaitGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state.as_deref_mut().unwrap()
    }
}
impl Drop for AddressWaitGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.state.take().unwrap();
        let wakes = std::mem::take(&mut state.wakes);
        let retired = std::mem::take(&mut state.retired);
        let arbiter_retired = std::mem::take(&mut state.arbiter_retired);
        drop(state);
        for event in wakes {
            event.signal();
        }
        drop(retired);
        drop(arbiter_retired);
    }
}

/// Process-owned payloads for the canonical kernel address wait queues.
#[derive(Debug, Default)]
pub struct AddressWaitState {
    kernel: OnceLock<(ProcessId, Arc<SchedulerState>)>,
    waiters: BTreeMap<GuestThreadId, AddressWaiter>,
    arbiter_waiters: BTreeMap<GuestThreadId, arbiter::Waiter>,
    arbiter_deadlines: BTreeSet<(u64, GuestThreadId)>,
    arbiter_deferred: BTreeSet<(u64, GuestThreadId)>,
    arbiter_completed: BTreeMap<GuestThreadId, AddressWaitResult>,
    arbiter_retired: Vec<arbiter::Waiter>,
    operations: BTreeMap<u64, AddressOperation>,
    deferred_addresses: BTreeSet<u64>,
    wakes: Vec<WritableEventObject>,
    retired: Vec<AddressWaiter>,
}

impl AddressWaitState {
    fn kernel(&self) -> &(ProcessId, Arc<SchedulerState>) {
        self.kernel
            .get()
            .expect("process address waits must be bound before execution")
    }
    fn first_ordered(&self, kind: OrderedWaitKind, address: u64) -> Option<GuestThreadId> {
        let (process, scheduler) = self.kernel();
        scheduler.first_ordered_waiter(*process, kind, address)
    }
    pub fn contains(&self, address: u64, thread: GuestThreadId) -> bool {
        self.waiters
            .get(&thread)
            .is_some_and(|waiter| waiter.address == address)
    }
    pub fn enqueue(
        &mut self,
        address: u64,
        owner: GuestThreadId,
        thread: GuestThreadId,
        value: u32,
    ) -> Result<ReadableEventObject, SchedulerError> {
        let (process, scheduler) = self.kernel();
        scheduler.add_waiter(
            *process,
            LockKey::User(address),
            owner,
            thread,
            Some(thread),
        )?;
        Ok(self.insert_waiter(address, thread, value, None))
    }
    pub fn enqueue_condition(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
        lock: u64,
        memory: nixe_memory::AllocationRange,
    ) -> Result<ConditionWait, SchedulerError> {
        self.kernel().1.register_ordered_wait(
            thread,
            OrderedWaitKind::ConditionVariable,
            address,
        )?;
        let event = self.insert_waiter(address, thread, value, Some(lock));
        let waiter = self.waiters.get_mut(&thread).unwrap();
        waiter.phase = WaitPhase::Registering;
        waiter.lock_memory = Some(memory);
        Ok(ConditionWait {
            key: address,
            thread,
            event,
        })
    }
    fn insert_waiter(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
        lock_address: Option<u64>,
    ) -> ReadableEventObject {
        let (writable, readable) = EventObject::create_pair();
        let previous = self.waiters.insert(
            thread,
            AddressWaiter {
                address,
                value,
                lock_address,
                phase: WaitPhase::Waiting,
                lock_memory: None,
                writable,
            },
        );
        debug_assert!(previous.is_none());
        readable
    }
    pub fn is_signalled(&self, address: u64, thread: GuestThreadId) -> bool {
        self.completion(address, thread).is_some()
    }
    pub fn remove(&mut self, address: u64, thread: GuestThreadId) {
        if self.contains(address, thread) {
            self.kernel().1.remove_waiter(thread);
            self.retired.push(self.waiters.remove(&thread).unwrap());
        }
    }
    pub fn has_condition_waiters(&self, address: u64) -> bool {
        self.first_ordered(OrderedWaitKind::ConditionVariable, address)
            .is_some()
    }
    pub fn completion(&self, address: u64, thread: GuestThreadId) -> Option<AddressWaitCompletion> {
        self.waiters
            .get(&thread)
            .filter(|w| w.address == address)
            .and_then(|w| w.phase.completion())
    }
    pub fn signal_result(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        result: AddressWaitCompletion,
    ) {
        if self.contains(address, thread) && self.waiters[&thread].phase.completion().is_none() {
            self.kernel().1.remove_waiter(thread);
            self.publish_result(thread, result);
        }
    }
    fn publish_result(&mut self, thread: GuestThreadId, result: AddressWaitCompletion) {
        let waiter = self.waiters.get_mut(&thread).unwrap();
        waiter.phase = WaitPhase::Resolved(result);
        self.wakes.push(waiter.writable.clone());
    }
    /// Accepted guest-word effects finish before a timeout can resume guest code.
    pub fn defer_timeout(&mut self, thread: GuestThreadId) -> bool {
        if let Some(address) = self.defer_arbiter_timeout(thread) {
            self.deferred_addresses.insert(address);
            return true;
        }
        let Some(waiter) = self.waiters.get_mut(&thread) else {
            return false;
        };
        waiter.phase = match waiter.phase {
            WaitPhase::Registering | WaitPhase::RegisteringTimedOut => {
                WaitPhase::RegisteringTimedOut
            }
            WaitPhase::ConditionSignalled | WaitPhase::ConditionSignalledTimedOut => {
                WaitPhase::ConditionSignalledTimedOut
            }
            WaitPhase::Handoff | WaitPhase::HandoffTimedOut => WaitPhase::HandoffTimedOut,
            _ => return false,
        };
        // A condition timed out during release must no longer be signalable.
        if waiter.phase == WaitPhase::RegisteringTimedOut {
            self.kernel().1.remove_waiter(thread);
        }
        true
    }
    pub fn expire_thread_wait(&mut self, thread: GuestThreadId) -> bool {
        let Some(waiter) = self.waiters.get(&thread) else {
            return false;
        };
        if waiter.phase != WaitPhase::Waiting {
            return false;
        }
        self.signal_result(waiter.address, thread, AddressWaitCompletion::TimedOut);
        true
    }
    fn condition_claim(&self, key: u64, thread: GuestThreadId) -> ConditionClaim {
        let waiter = &self.waiters[&thread];
        ConditionClaim {
            key,
            thread,
            tag: waiter.value,
            mutex: waiter.lock_address.expect("condition word"),
            memory: waiter
                .lock_memory
                .as_ref()
                .expect("accepted condition word")
                .clone(),
            event: waiter.writable.clone(),
        }
    }
    pub fn claim_condition(&mut self, key: u64) -> Option<ConditionSignal> {
        let thread = self.first_ordered(OrderedWaitKind::ConditionVariable, key)?;
        self.kernel().1.remove_waiter(thread);
        let waiter = self.waiters.get_mut(&thread).unwrap();
        if waiter.phase == WaitPhase::Registering {
            waiter.phase = WaitPhase::ConditionSignalled;
            Some(ConditionSignal::PendingRelease)
        } else {
            waiter.phase = WaitPhase::Handoff;
            Some(ConditionSignal::Reacquire(
                self.condition_claim(key, thread),
            ))
        }
    }
    pub fn finish_condition_release(
        &mut self,
        registration: &ConditionWait,
    ) -> Option<ConditionClaim> {
        let waiter = self.waiters.get_mut(&registration.thread)?;
        if waiter.address != registration.key || !registration.event.same_event(&waiter.writable) {
            return None;
        }
        match waiter.phase {
            WaitPhase::Registering => {
                waiter.phase = WaitPhase::Waiting;
                None
            }
            WaitPhase::RegisteringTimedOut => {
                self.signal_result(
                    registration.key,
                    registration.thread,
                    AddressWaitCompletion::TimedOut,
                );
                None
            }
            WaitPhase::ConditionSignalled | WaitPhase::ConditionSignalledTimedOut => {
                waiter.phase = if waiter.phase == WaitPhase::ConditionSignalledTimedOut {
                    WaitPhase::HandoffTimedOut
                } else {
                    WaitPhase::Handoff
                };
                Some(self.condition_claim(registration.key, registration.thread))
            }
            _ => None,
        }
    }
    pub fn finish_condition_acquire(
        &mut self,
        claim: &ConditionClaim,
        outcome: ConditionAcquire,
        current: GuestThreadId,
    ) -> Result<(), SchedulerError> {
        let Some(waiter) = self.waiters.get(&claim.thread) else {
            return Ok(());
        };
        if waiter.address != claim.key
            || !waiter.writable.same_event(&claim.event)
            || !matches!(
                waiter.phase,
                WaitPhase::Handoff | WaitPhase::HandoffTimedOut
            )
        {
            return Ok(());
        }
        let expired = waiter.phase == WaitPhase::HandoffTimedOut;
        match outcome {
            ConditionAcquire::Contended(owner) if !expired => {
                let (process, scheduler) = self.kernel();
                scheduler.add_waiter(
                    *process,
                    LockKey::User(claim.mutex),
                    owner,
                    claim.thread,
                    Some(current),
                )?;
                let waiter = self.waiters.get_mut(&claim.thread).unwrap();
                waiter.address = claim.mutex;
                waiter.lock_address = None;
                waiter.phase = WaitPhase::Waiting;
            }
            ConditionAcquire::Contended(_) => {
                let waiter = self.waiters.get_mut(&claim.thread).unwrap();
                waiter.address = claim.mutex;
                waiter.lock_address = None;
                self.signal_result(claim.mutex, claim.thread, AddressWaitCompletion::TimedOut);
            }
            outcome => self.signal_result(
                claim.key,
                claim.thread,
                match outcome {
                    ConditionAcquire::Acquired => AddressWaitCompletion::Success,
                    ConditionAcquire::Contended(_) => unreachable!(),
                    ConditionAcquire::InvalidOwner => AddressWaitCompletion::InvalidOwner,
                    ConditionAcquire::InvalidMemory => AddressWaitCompletion::InvalidMemory,
                },
            ),
        }
        Ok(())
    }
    pub fn claim_mutex(
        &mut self,
        address: u64,
        owner: GuestThreadId,
    ) -> Result<MutexHandoff, SchedulerError> {
        let (process, scheduler) = self.kernel();
        let handoff = scheduler.release_lock(*process, LockKey::User(address), owner)?;
        let next = handoff.next.map(|thread| {
            let waiter = self
                .waiters
                .get_mut(&thread)
                .expect("selected mutex waiter");
            waiter.phase = WaitPhase::Handoff;
            (thread, waiter.writable.clone())
        });
        Ok(MutexHandoff {
            address,
            next_tag: next.as_ref().map(|(thread, _)| self.waiters[thread].value),
            next,
            more_waiters: handoff.more_waiters,
        })
    }
    pub fn finish_mutex_handoff(&mut self, claim: &MutexHandoff, result: AddressWaitCompletion) {
        let Some((thread, event)) = &claim.next else {
            return;
        };
        if self.waiters.get(thread).is_some_and(|w| {
            w.address == claim.address
                && matches!(w.phase, WaitPhase::Handoff | WaitPhase::HandoffTimedOut)
                && w.writable.same_event(event)
        }) {
            self.signal_result(claim.address, *thread, result);
        }
    }
    pub(crate) fn release_thread(&mut self, thread: GuestThreadId) {
        self.remove_address_wait(thread);
        if let Some(waiter) = self.waiters.get(&thread) {
            self.remove(waiter.address, thread);
        }
    }
    pub(crate) fn cancel_held_waiter(&mut self, thread: GuestThreadId) {
        // The canonical node was detached before entering its process registry.
        // A timeout/cancellation may resolve or remove its payload in between.
        if self.waiters.get(&thread).is_some_and(|waiter| {
            waiter.lock_address.is_none() && waiter.phase == WaitPhase::Waiting
        }) {
            self.publish_result(thread, AddressWaitCompletion::InvalidOwner);
        }
    }
    pub fn waiter_count(&self) -> usize {
        self.waiters.len() + self.arbiter_waiters.len() + self.arbiter_completed.len()
    }
}

#[cfg(test)]
fn test_registry() -> AddressWaitRegistry {
    use nixe_scheduler::{
        MachineSchedulerProfile, PriorityRange, ScheduledThreadConfig, VirtualCpuDescriptor,
        VirtualCpuId,
    };
    let core = VirtualCpuId::new(0);
    let scheduler = Arc::new(SchedulerState::new(
        MachineSchedulerProfile::new(
            vec![VirtualCpuDescriptor::new(core, 0)],
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    ));
    let registry = AddressWaitRegistry::default();
    registry.bind(ProcessId::new(1), scheduler.clone());
    for id in 1..=4 {
        let thread = scheduler
            .create_thread(ScheduledThreadConfig {
                process: ProcessId::new(1),
                base_priority: 20,
                ideal_vcpu: Some(core),
                affinity: scheduler.profile().core_set([core]).unwrap(),
            })
            .unwrap();
        assert_eq!(thread, GuestThreadId::new(id));
    }
    registry
}

#[cfg(test)]
mod finalization_tests {
    use super::*;

    #[test]
    fn finalization_preserves_a_timeout_or_removal_that_won_after_detachment() {
        for removed in [false, true] {
            let registry = test_registry();
            let owner = GuestThreadId::new(1);
            let thread = GuestThreadId::new(2);
            let event = registry.lock().enqueue(0x1000, owner, thread, 7).unwrap();
            let scheduler = registry.lock().kernel().1.clone();
            assert_eq!(
                scheduler.take_held_waiter(owner),
                Some((ProcessId::new(1), thread))
            );
            let mut state = registry.lock();
            if removed {
                state.remove(0x1000, thread);
            } else {
                state.signal_result(0x1000, thread, AddressWaitCompletion::TimedOut);
            }
            state.cancel_held_waiter(thread);
            assert_eq!(
                state.completion(0x1000, thread),
                (!removed).then_some(AddressWaitCompletion::TimedOut)
            );
            assert_eq!(state.waiter_count(), usize::from(!removed));
            drop(state);
            if !removed {
                assert!(event.is_signalled());
            }
        }
    }
}

#[cfg(test)]
mod notification_tests {
    use super::*;
    use std::{
        future::Future,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        task::{Context, Wake, Waker},
    };
    struct Reenter {
        registry: Arc<AddressWaitRegistry>,
        notifications: AtomicUsize,
    }
    impl Wake for Reenter {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            assert!(
                self.registry.state.try_lock().is_ok(),
                "kernel wait notification ran under its state lock"
            );
            assert_eq!(self.registry.lock().waiter_count(), 1);
            self.notifications.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn terminal_claim_is_visible_before_waking_and_wakers_run_after_unlock() {
        let registry = Arc::new(test_registry());
        let thread = GuestThreadId::new(1);
        let event = registry
            .lock()
            .enqueue(0x1000, GuestThreadId::new(2), thread, 7)
            .unwrap();
        let wake = Arc::new(Reenter {
            registry: registry.clone(),
            notifications: AtomicUsize::new(0),
        });
        let waker = Waker::from(wake.clone());
        let mut wait = std::pin::pin!(event.wait_async());
        assert!(
            wait.as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        {
            let mut state = registry.lock();
            state.signal_result(0x1000, thread, AddressWaitCompletion::Success);
            state.signal_result(0x1000, thread, AddressWaitCompletion::InvalidOwner);
            assert_eq!(
                state.completion(0x1000, thread),
                Some(AddressWaitCompletion::Success)
            );
            assert_eq!(wake.notifications.load(Ordering::SeqCst), 0);
        }
        assert_eq!(wake.notifications.load(Ordering::SeqCst), 1);
        assert!(
            wait.as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_ready()
        );
    }
}

#[cfg(test)]
mod handoff_tests {
    use super::*;
    #[test]
    fn a_final_handoff_allows_an_unobserved_fast_release_and_a_new_owner() {
        let registry = test_registry();
        let owner = GuestThreadId::new(1);
        let next = GuestThreadId::new(2);
        let fast_owner = GuestThreadId::new(3);
        let final_waiter = GuestThreadId::new(4);
        let (event, claim) = {
            let mut state = registry.lock();
            let event = state.enqueue(0x1000, owner, next, 0x22).unwrap();
            let claim = state.claim_mutex(0x1000, owner).unwrap();
            assert!(!claim.more_waiters);
            (event, claim)
        };
        registry
            .lock()
            .finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        assert!(event.is_signalled());
        let mut state = registry.lock();
        state.remove(0x1000, next);
        // The selected thread drops the lock in user code; another thread
        // acquires it there. The next kernel wait must use that actual owner.
        let event = state
            .enqueue(0x1000, fast_owner, final_waiter, 0x44)
            .unwrap();
        let claim = state.claim_mutex(0x1000, fast_owner).unwrap();
        assert_eq!(claim.next_tag, Some(0x44));
        state.finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        drop(state);
        assert!(event.is_signalled());
    }
    #[test]
    fn delayed_handoff_completion_cannot_complete_a_new_wait_on_the_same_word() {
        let registry = test_registry();
        let owner = GuestThreadId::new(1);
        let thread = GuestThreadId::new(2);
        let claim = {
            let mut state = registry.lock();
            let _old = state.enqueue(0x1000, owner, thread, 0x22).unwrap();
            state.claim_mutex(0x1000, owner).unwrap()
        };
        let next = {
            let mut state = registry.lock();
            assert!(!state.expire_thread_wait(thread));
            state.remove(0x1000, thread);
            state.enqueue(0x1000, owner, thread, 0x44).unwrap()
        };
        registry
            .lock()
            .finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        assert!(!next.is_signalled());
        assert_eq!(registry.lock().completion(0x1000, thread), None);
        assert_eq!(registry.lock().waiters[&thread].value, 0x44);
    }
}

#[cfg(test)]
mod condition_tests {
    use super::*;
    fn memory() -> nixe_memory::AllocationRange {
        let machine = nixe_memory::Machine::new(nixe_memory::PAGE_SIZE).unwrap();
        let allocation = machine
            .account(nixe_memory::PAGE_SIZE)
            .unwrap()
            .allocate(nixe_memory::PAGE_SIZE)
            .unwrap();
        nixe_memory::AllocationRange::new(allocation.pin(), 0, 4).unwrap()
    }
    fn register(registry: &AddressWaitRegistry) -> ConditionWait {
        registry
            .lock()
            .enqueue_condition(0x2000, GuestThreadId::new(2), 0x22, 0x1000, memory())
            .unwrap()
    }
    #[test]
    fn a_signal_during_release_is_consumed_once_and_reacquires_after_publication() {
        let registry = test_registry();
        let wait = register(&registry);
        assert!(matches!(
            registry.lock().claim_condition(0x2000),
            Some(ConditionSignal::PendingRelease)
        ));
        assert!(registry.lock().claim_condition(0x2000).is_none());
        assert!(!wait.event().is_signalled());
        let claim = registry.lock().finish_condition_release(&wait).unwrap();
        assert!(!wait.event().is_signalled());
        assert_eq!(claim.memory.bytes(), 4);
        registry
            .lock()
            .finish_condition_acquire(&claim, ConditionAcquire::Acquired, GuestThreadId::new(1))
            .unwrap();
        assert!(wait.event().is_signalled());
        assert_eq!(
            registry.lock().completion(0x2000, GuestThreadId::new(2)),
            Some(AddressWaitCompletion::Success)
        );
    }
    #[test]
    fn timeout_during_release_waits_for_the_effect_and_does_not_consume_a_later_signal() {
        let registry = test_registry();
        let wait = register(&registry);
        assert!(registry.lock().defer_timeout(GuestThreadId::new(2)));
        assert!(!wait.event().is_signalled());
        assert!(registry.lock().claim_condition(0x2000).is_none());
        assert!(registry.lock().finish_condition_release(&wait).is_none());
        assert!(wait.event().is_signalled());
        assert_eq!(
            registry.lock().completion(0x2000, GuestThreadId::new(2)),
            Some(AddressWaitCompletion::TimedOut)
        );
    }
    #[test]
    fn a_signal_before_timeout_finishes_its_atomic_attempt_before_resolving_the_target() {
        for acquired in [true, false] {
            let registry = test_registry();
            let wait = register(&registry);
            assert!(registry.lock().finish_condition_release(&wait).is_none());
            let Some(ConditionSignal::Reacquire(claim)) = registry.lock().claim_condition(0x2000)
            else {
                panic!("waiting condition was not selected");
            };
            assert!(registry.lock().defer_timeout(GuestThreadId::new(2)));
            assert!(!wait.event().is_signalled());
            let owner = GuestThreadId::new(3);
            registry
                .lock()
                .finish_condition_acquire(
                    &claim,
                    if acquired {
                        ConditionAcquire::Acquired
                    } else {
                        ConditionAcquire::Contended(owner)
                    },
                    GuestThreadId::new(1),
                )
                .unwrap();
            assert!(wait.event().is_signalled());
            if acquired {
                assert_eq!(
                    registry.lock().completion(0x2000, GuestThreadId::new(2)),
                    Some(AddressWaitCompletion::Success)
                );
            } else {
                assert_eq!(
                    registry.lock().completion(0x1000, GuestThreadId::new(2)),
                    Some(AddressWaitCompletion::TimedOut)
                );
            }
        }
    }
    #[test]
    fn a_stale_condition_claim_cannot_set_the_owner_or_complete_a_replacement_wait() {
        let registry = test_registry();
        let wait = register(&registry);
        registry.lock().finish_condition_release(&wait);
        let Some(ConditionSignal::Reacquire(claim)) = registry.lock().claim_condition(0x2000)
        else {
            panic!("waiting condition was not selected");
        };
        registry.lock().remove(0x2000, GuestThreadId::new(2));
        let next = register(&registry);
        registry
            .lock()
            .finish_condition_acquire(&claim, ConditionAcquire::Acquired, GuestThreadId::new(1))
            .unwrap();
        assert!(!next.event().is_signalled());
        assert_eq!(
            registry.lock().completion(0x2000, GuestThreadId::new(2)),
            None
        );
    }
}

#[cfg(test)]
mod address_operation_tests {
    use super::*;
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    fn clock() -> crate::VirtualClock {
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 })
    }
    #[test]
    fn same_word_operations_wait_without_a_host_lock_and_disjoint_words_progress() {
        let registry = Arc::new(test_registry());
        let clock = clock();
        let mut first = Box::pin(registry.address_operation(0x1000, GuestThreadId::new(1), &clock));
        let mut context = Context::from_waker(Waker::noop());
        let Poll::Ready(held) = first.as_mut().poll(&mut context) else {
            panic!("uncontended word operation blocked");
        };
        let mut next = Box::pin(registry.address_operation(0x1000, GuestThreadId::new(2), &clock));
        assert!(next.as_mut().poll(&mut context).is_pending());
        let mut disjoint =
            Box::pin(registry.address_operation(0x2000, GuestThreadId::new(3), &clock));
        let Poll::Ready(disjoint) = disjoint.as_mut().poll(&mut context) else {
            panic!("an unrelated word was blocked");
        };
        assert!(registry.state.try_lock().is_ok());
        drop(disjoint);
        drop(held);
        assert!(next.as_mut().poll(&mut context).is_ready());
    }
    #[test]
    fn a_conditional_signal_publishes_its_result_before_deferred_word_timeouts() {
        for succeeds in [true, false] {
            let registry = Arc::new(test_registry());
            let clock = clock();
            let target = GuestThreadId::new(2);
            let event = registry
                .lock()
                .enqueue_address_wait(0x1000, target, Some(50))
                .unwrap();
            registry
                .lock()
                .finish_address_check(target, 0, false)
                .unwrap();
            let mut operation =
                Box::pin(registry.address_operation(0x1000, GuestThreadId::new(1), &clock));
            let mut context = Context::from_waker(Waker::noop());
            let Poll::Ready(held) = operation.as_mut().poll(&mut context) else {
                panic!("uncontended word operation blocked");
            };
            let (selected, more) = registry.lock().signal_targets(0x1000, 1);
            assert!(!more);
            clock.advance_scheduler_to(50);
            registry.lock().expire_address_waits(50);
            assert!(!event.is_signalled());
            if succeeds {
                registry.lock().complete_signal(&selected);
            }
            drop(held);
            assert!(event.is_signalled());
            assert_eq!(
                registry.lock().take_address_result(target),
                Some(if succeeds {
                    AddressWaitResult::Signalled
                } else {
                    AddressWaitResult::TimedOut
                })
            );
        }
    }
    #[test]
    fn queued_word_checks_are_not_signalled_or_resumed_before_their_accepted_access() {
        let registry = test_registry();
        let target = GuestThreadId::new(2);
        let event = {
            let mut state = registry.lock();
            state
                .enqueue_address_wait(0x1000, target, Some(50))
                .unwrap()
        };
        assert!(
            registry
                .lock()
                .signal_targets(0x1000, usize::MAX)
                .0
                .is_empty()
        );
        assert!(registry.lock().defer_timeout(target));
        assert!(!event.is_signalled());
        registry
            .lock()
            .finish_address_check(target, 50, false)
            .unwrap();
        assert!(event.is_signalled());
        assert_eq!(
            registry.lock().take_address_result(target),
            Some(AddressWaitResult::TimedOut)
        );
    }
}
