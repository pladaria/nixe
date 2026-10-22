//! Console-neutral ownership for address-keyed wait queues.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard},
};

use nixe_scheduler::GuestThreadId;

use crate::{EventObject, ReadableEventObject, WritableEventObject};

mod priority;
pub use priority::{AddressSignalTargets, AddressWaitResult, PriorityAddressWaitQueue};

#[derive(Clone, Debug)]
struct AddressWaiter {
    thread: GuestThreadId,
    value: u32,
    lock_address: Option<u64>,
    priority: i32,
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
                state.priority_waits.publish_deferred_timeouts(self.address);
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
        let priority_wakes = state.priority_waits.take_wakes();
        let retired = std::mem::take(&mut state.retired);
        let priority_retired = state.priority_waits.take_retired();
        drop(state);
        for event in wakes.into_iter().chain(priority_wakes) {
            event.signal();
        }
        drop(retired);
        drop(priority_retired);
    }
}

/// Process-owned address wait queues and mutex ownership records.
#[derive(Debug, Default)]
pub struct AddressWaitState {
    waiters: BTreeMap<u64, VecDeque<AddressWaiter>>,
    owners: BTreeMap<u64, GuestThreadId>,
    priority_waits: PriorityAddressWaitQueue,
    operations: BTreeMap<u64, AddressOperation>,
    deferred_addresses: BTreeSet<u64>,
    wakes: Vec<WritableEventObject>,
    retired: Vec<AddressWaiter>,
}

impl AddressWaitState {
    pub fn expire_address_waits(&mut self, now: u64) {
        let operations = &self.operations;
        let deferred = &mut self.deferred_addresses;
        self.priority_waits.expire(now, |address| {
            if operations.contains_key(&address) {
                deferred.insert(address);
                true
            } else {
                false
            }
        });
    }
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
        memory: nixe_memory::AllocationRange,
    ) -> ConditionWait {
        let event = self.enqueue_with_lock(address, thread, value, Some(lock));
        let waiter = self.waiters.get_mut(&address).unwrap().back_mut().unwrap();
        waiter.phase = WaitPhase::Registering;
        waiter.lock_memory = Some(memory);
        ConditionWait {
            key: address,
            thread,
            event,
        }
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
                phase: WaitPhase::Waiting,
                lock_memory: None,
                writable,
            });
        readable
    }

    #[must_use]
    pub fn is_signalled(&self, address: u64, thread: GuestThreadId) -> bool {
        self.waiters.get(&address).is_some_and(|waiters| {
            waiters
                .iter()
                .find(|waiter| waiter.thread == thread)
                .is_some_and(|waiter| waiter.phase.completion().is_some())
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
            self.retired.push(waiters.remove(index).unwrap());
        }
        if self.waiters.get(&address).is_some_and(VecDeque::is_empty) {
            self.waiters.remove(&address);
        }
        self.retire_uncontended_owner(address);
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
                .filter(|waiter| {
                    waiter.lock_address.is_none() && waiter.phase == WaitPhase::Waiting
                })
                .count()
        })
    }

    /// Select current effective priority, then FIFO among equal priorities.
    pub fn next_waiter(&self, address: u64) -> Option<(GuestThreadId, u32)> {
        self.waiters
            .get(&address)?
            .iter()
            .enumerate()
            .filter(|(_, waiter)| {
                waiter.lock_address.is_none() && waiter.phase == WaitPhase::Waiting
            })
            .min_by_key(|(order, waiter)| (waiter.priority, *order))
            .map(|(_, waiter)| (waiter.thread, waiter.value))
    }

    pub fn next_condition(&self, address: u64) -> Option<(GuestThreadId, u32, u64)> {
        self.waiters
            .get(&address)?
            .iter()
            .enumerate()
            .filter(|(_, waiter)| {
                waiter.lock_address.is_some()
                    && matches!(waiter.phase, WaitPhase::Waiting | WaitPhase::Registering)
            })
            .min_by_key(|(order, waiter)| (waiter.priority, *order))
            .map(|(_, waiter)| (waiter.thread, waiter.value, waiter.lock_address.unwrap()))
    }

    pub fn pending_threads(&self, address: u64) -> impl Iterator<Item = GuestThreadId> + '_ {
        self.waiters
            .get(&address)
            .into_iter()
            .flatten()
            .filter(|waiter| waiter.lock_address.is_none() && waiter.phase == WaitPhase::Waiting)
            .map(|waiter| waiter.thread)
    }

    pub fn completion(&self, address: u64, thread: GuestThreadId) -> Option<AddressWaitCompletion> {
        self.waiters
            .get(&address)?
            .iter()
            .find(|waiter| waiter.thread == thread)
            .and_then(|waiter| waiter.phase.completion())
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
            && waiter.phase.completion().is_none()
        {
            waiter.phase = WaitPhase::Resolved(result);
            self.wakes.push(waiter.writable.clone());
        }
        self.retire_uncontended_owner(address);
    }

    /// A pending kernel word effect keeps its target blocked until that effect
    /// completes. Record the timer in that same wait phase rather than waking
    /// guest code ahead of an accepted ownership write.
    pub fn defer_timeout(&mut self, thread: GuestThreadId) -> bool {
        let operations = &self.operations;
        if let Some(address) = self
            .priority_waits
            .defer_timeout(thread, |address| operations.contains_key(&address))
        {
            self.deferred_addresses.insert(address);
            return true;
        }
        for waiter in self
            .waiters
            .values_mut()
            .flatten()
            .filter(|waiter| waiter.thread == thread)
        {
            let phase = match waiter.phase {
                WaitPhase::Registering | WaitPhase::RegisteringTimedOut => {
                    WaitPhase::RegisteringTimedOut
                }
                WaitPhase::ConditionSignalled | WaitPhase::ConditionSignalledTimedOut => {
                    WaitPhase::ConditionSignalledTimedOut
                }
                WaitPhase::Handoff | WaitPhase::HandoffTimedOut => WaitPhase::HandoffTimedOut,
                _ => continue,
            };
            waiter.phase = phase;
            return true;
        }
        false
    }
    pub fn expire_thread_wait(&mut self, thread: GuestThreadId) -> bool {
        let expired = self.waiters.iter_mut().find_map(|(address, waiters)| {
            let waiter = waiters
                .iter_mut()
                .find(|waiter| waiter.thread == thread && waiter.phase == WaitPhase::Waiting)?;
            waiter.phase = WaitPhase::Resolved(AddressWaitCompletion::TimedOut);
            self.wakes.push(waiter.writable.clone());
            Some(*address)
        });
        if let Some(address) = expired {
            self.retire_uncontended_owner(address);
        }
        expired.is_some()
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

    fn condition_claim(&self, key: u64, thread: GuestThreadId) -> ConditionClaim {
        let waiter = self
            .waiters
            .get(&key)
            .unwrap()
            .iter()
            .find(|waiter| waiter.thread == thread)
            .unwrap();
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
        let (thread, _, _) = self.next_condition(key)?;
        let waiter = self
            .waiters
            .get_mut(&key)
            .unwrap()
            .iter_mut()
            .find(|waiter| waiter.thread == thread)
            .unwrap();
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
        let waiter = self
            .waiters
            .get_mut(&registration.key)?
            .iter_mut()
            .find(|waiter| waiter.thread == registration.thread)?;
        if !registration.event.same_event(&waiter.writable) {
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
    pub fn finish_condition_acquire(&mut self, claim: &ConditionClaim, outcome: ConditionAcquire) {
        let Some(waiter) = self
            .waiters
            .get(&claim.key)
            .and_then(|waiters| waiters.iter().find(|waiter| waiter.thread == claim.thread))
        else {
            return;
        };
        if !waiter.writable.same_event(&claim.event)
            || !matches!(
                waiter.phase,
                WaitPhase::Handoff | WaitPhase::HandoffTimedOut
            )
        {
            return;
        }
        let expired = waiter.phase == WaitPhase::HandoffTimedOut;
        match outcome {
            ConditionAcquire::Acquired => {
                self.signal_result(claim.key, claim.thread, AddressWaitCompletion::Success);
            }
            ConditionAcquire::Contended(owner) => {
                self.set_owner(claim.mutex, owner);
                self.move_waiter(claim.key, claim.mutex, claim.thread);
                let waiter = self
                    .waiters
                    .get_mut(&claim.mutex)
                    .unwrap()
                    .iter_mut()
                    .find(|waiter| waiter.thread == claim.thread)
                    .unwrap();
                waiter.phase = WaitPhase::Waiting;
                if expired {
                    self.signal_result(claim.mutex, claim.thread, AddressWaitCompletion::TimedOut);
                }
            }
            ConditionAcquire::InvalidOwner => {
                self.signal_result(claim.key, claim.thread, AddressWaitCompletion::InvalidOwner)
            }
            ConditionAcquire::InvalidMemory => self.signal_result(
                claim.key,
                claim.thread,
                AddressWaitCompletion::InvalidMemory,
            ),
        }
    }

    pub fn claim_mutex(&mut self, address: u64, owner: GuestThreadId) -> MutexHandoff {
        let next = if self.owner(address) == Some(owner) {
            self.next_waiter(address)
        } else {
            None
        };
        let claim = next.map(|(thread, tag)| {
            let waiter = self
                .waiters
                .get_mut(&address)
                .unwrap()
                .iter_mut()
                .find(|waiter| waiter.thread == thread)
                .unwrap();
            waiter.phase = WaitPhase::Handoff;
            (thread, tag, waiter.writable.clone())
        });
        if self.owner(address) == Some(owner) {
            if let Some((thread, _, _)) = &claim {
                self.set_owner(address, *thread);
            } else {
                self.remove_owner(address);
            }
        }
        self.retire_uncontended_owner(address);
        MutexHandoff {
            address,
            next_tag: claim.as_ref().map(|(_, tag, _)| *tag),
            more_waiters: self.unsignalled_count(address) != 0,
            next: claim.map(|(thread, _, event)| (thread, event)),
        }
    }
    pub fn finish_mutex_handoff(&mut self, claim: &MutexHandoff, result: AddressWaitCompletion) {
        let Some((thread, event)) = &claim.next else {
            return;
        };
        if let Some(waiter) = self
            .waiters
            .get_mut(&claim.address)
            .and_then(|waiters| waiters.iter_mut().find(|waiter| waiter.thread == *thread))
            && matches!(
                waiter.phase,
                WaitPhase::Handoff | WaitPhase::HandoffTimedOut
            )
            && waiter.writable.same_event(event)
        {
            self.signal_result(claim.address, *thread, result);
        }
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

    fn retire_uncontended_owner(&mut self, address: u64) {
        // Guest code can release and reacquire an uncontended mutex without an
        // SVC. Only blocked waiters keep kernel ownership/donation metadata.
        if self.unsignalled_count(address) == 0 {
            self.owners.remove(&address);
        }
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
        let owner_registry = AddressWaitRegistry::default();
        let mut registry = owner_registry.lock();
        registry.set_owner(0x1000, owner);
        let wake = registry.enqueue(0x1000, waiter, 0x1234);
        let other = GuestThreadId::new(3);
        let other_wake = registry.enqueue(0x1000, other, 0x5678);
        assert!(!wake.is_signalled());
        registry.release_thread(owner);
        drop(registry);
        assert!(wake.is_signalled());
        assert!(other_wake.is_signalled());
        let mut registry = owner_registry.lock();
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
        let registry = Arc::new(AddressWaitRegistry::default());
        let thread = GuestThreadId::new(1);
        let event = registry.lock().enqueue(0x1000, thread, 7);
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
        let registry = AddressWaitRegistry::default();
        let owner = GuestThreadId::new(1);
        let next = GuestThreadId::new(2);
        let fast_owner = GuestThreadId::new(3);
        let final_waiter = GuestThreadId::new(4);
        let (event, claim) = {
            let mut state = registry.lock();
            state.set_owner(0x1000, owner);
            let event = state.enqueue(0x1000, next, 0x22);
            let claim = state.claim_mutex(0x1000, owner);
            assert!(!claim.more_waiters);
            (event, claim)
        };
        registry
            .lock()
            .finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        assert!(event.is_signalled());
        let mut state = registry.lock();
        assert_eq!(state.owner(0x1000), None);
        state.remove(0x1000, next);
        // The selected thread drops the lock in user code; another thread
        // acquires it there. The next kernel wait must use that actual owner.
        state.set_owner(0x1000, fast_owner);
        let event = state.enqueue(0x1000, final_waiter, 0x44);
        let claim = state.claim_mutex(0x1000, fast_owner);
        assert_eq!(claim.next_tag, Some(0x44));
        state.finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        drop(state);
        assert!(event.is_signalled());
    }
    #[test]
    fn selected_owner_is_reserved_until_publication_even_if_the_unlocking_thread_exits() {
        let registry = AddressWaitRegistry::default();
        let owner = GuestThreadId::new(1);
        let first = GuestThreadId::new(2);
        let second = GuestThreadId::new(3);
        let (first_event, second_event, claim) = {
            let mut state = registry.lock();
            state.set_owner(0x1000, owner);
            let first_event = state.enqueue(0x1000, first, 0x22);
            let second_event = state.enqueue(0x1000, second, 0x33);
            let claim = state.claim_mutex(0x1000, owner);
            assert_eq!(claim.next_tag, Some(0x22));
            assert!(claim.more_waiters);
            assert_eq!(state.owner(0x1000), Some(first));
            assert_eq!(state.next_waiter(0x1000), Some((second, 0x33)));
            assert_eq!(state.pending_threads(0x1000).collect::<Vec<_>>(), [second]);
            (first_event, second_event, claim)
        };
        assert!(!first_event.is_signalled());
        registry.lock().release_thread(owner);
        assert!(!first_event.is_signalled());
        registry
            .lock()
            .finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        assert!(first_event.is_signalled());
        assert!(!second_event.is_signalled());
        assert_eq!(
            registry.lock().completion(0x1000, first),
            Some(AddressWaitCompletion::Success)
        );
    }
    #[test]
    fn delayed_handoff_completion_cannot_complete_a_new_wait_on_the_same_word() {
        let registry = AddressWaitRegistry::default();
        let owner = GuestThreadId::new(1);
        let thread = GuestThreadId::new(2);
        let claim = {
            let mut state = registry.lock();
            state.set_owner(0x1000, owner);
            let _old = state.enqueue(0x1000, thread, 0x22);
            state.claim_mutex(0x1000, owner)
        };
        let next = {
            let mut state = registry.lock();
            assert!(!state.expire_thread_wait(thread));
            state.remove(0x1000, thread);
            state.enqueue(0x1000, thread, 0x44)
        };
        registry
            .lock()
            .finish_mutex_handoff(&claim, AddressWaitCompletion::Success);
        assert!(!next.is_signalled());
        assert_eq!(registry.lock().completion(0x1000, thread), None);
        assert_eq!(registry.lock().value(0x1000, thread), Some(0x44));
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
    }
    #[test]
    fn a_signal_during_release_is_consumed_once_and_reacquires_after_publication() {
        let registry = AddressWaitRegistry::default();
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
            .finish_condition_acquire(&claim, ConditionAcquire::Acquired);
        assert!(wait.event().is_signalled());
        assert_eq!(
            registry.lock().completion(0x2000, GuestThreadId::new(2)),
            Some(AddressWaitCompletion::Success)
        );
        assert_eq!(registry.lock().owner(0x1000), None);
    }
    #[test]
    fn timeout_during_release_waits_for_the_effect_and_does_not_consume_a_later_signal() {
        let registry = AddressWaitRegistry::default();
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
            let registry = AddressWaitRegistry::default();
            let wait = register(&registry);
            assert!(registry.lock().finish_condition_release(&wait).is_none());
            let Some(ConditionSignal::Reacquire(claim)) = registry.lock().claim_condition(0x2000)
            else {
                panic!("waiting condition was not selected");
            };
            assert!(registry.lock().defer_timeout(GuestThreadId::new(2)));
            assert!(!wait.event().is_signalled());
            let owner = GuestThreadId::new(3);
            registry.lock().finish_condition_acquire(
                &claim,
                if acquired {
                    ConditionAcquire::Acquired
                } else {
                    ConditionAcquire::Contended(owner)
                },
            );
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
                assert_eq!(registry.lock().owner(0x1000), None);
                assert_eq!(registry.lock().pending_threads(0x1000).count(), 0);
            }
        }
    }
    #[test]
    fn a_stale_condition_claim_cannot_set_the_owner_or_complete_a_replacement_wait() {
        let registry = AddressWaitRegistry::default();
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
            .finish_condition_acquire(&claim, ConditionAcquire::Acquired);
        assert!(!next.event().is_signalled());
        assert_eq!(registry.lock().owner(0x1000), None);
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
        let registry = Arc::new(AddressWaitRegistry::default());
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
            let registry = Arc::new(AddressWaitRegistry::default());
            let clock = clock();
            let target = GuestThreadId::new(2);
            let event = registry
                .lock()
                .priority_waits_mut()
                .enqueue(0x1000, target, 20, Some(50))
                .unwrap();
            let mut operation =
                Box::pin(registry.address_operation(0x1000, GuestThreadId::new(1), &clock));
            let mut context = Context::from_waker(Waker::noop());
            let Poll::Ready(held) = operation.as_mut().poll(&mut context) else {
                panic!("uncontended word operation blocked");
            };
            let (selected, more) = registry.lock().priority_waits().signal_targets(0x1000, 1);
            assert!(!more);
            clock.advance_scheduler_to(50);
            registry.lock().expire_address_waits(50);
            assert!(!event.is_signalled());
            if succeeds {
                registry
                    .lock()
                    .priority_waits_mut()
                    .complete_signal(&selected);
            }
            drop(held);
            assert!(event.is_signalled());
            assert_eq!(
                registry.lock().priority_waits_mut().take_result(target),
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
        let registry = AddressWaitRegistry::default();
        let target = GuestThreadId::new(2);
        let event = {
            let mut state = registry.lock();
            let queue = state.priority_waits_mut();
            let event = queue.enqueue(0x1000, target, 20, Some(50)).unwrap();
            queue.begin_check(target);
            event
        };
        assert!(
            registry
                .lock()
                .priority_waits()
                .signal_targets(0x1000, usize::MAX)
                .0
                .is_empty()
        );
        assert!(registry.lock().defer_timeout(target));
        assert!(!event.is_signalled());
        registry
            .lock()
            .priority_waits_mut()
            .finish_check(target, 50, false);
        assert!(event.is_signalled());
        assert_eq!(
            registry.lock().priority_waits_mut().take_result(target),
            Some(AddressWaitResult::TimedOut)
        );
    }
}
