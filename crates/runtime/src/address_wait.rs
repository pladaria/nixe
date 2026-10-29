//! Process-owned address wait payloads; generations and timeout arbitration belong to the kernel.
use crate::{EventObject, ReadableEventObject, WritableEventObject};
use nixe_scheduler::{
    EventNotifications, GuestThreadId, KernelWait, LockKey, OrderedWaitKind, ProcessId,
    SchedulerError, SchedulerState, WaitEffect, WaitOutcome,
};
use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};
mod arbiter;
pub use arbiter::AddressSignalTargets;

#[derive(Debug)]
struct AddressWaiter {
    address: u64,
    value: u32,
    lock_address: Option<u64>,
    phase: WaitPhase,
    lock_memory: Option<nixe_memory::AllocationRange>,
    wait: Arc<KernelWait>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitPhase {
    Registering,
    ConditionSignalled,
    Waiting,
    Handoff,
    Resolved(AddressWaitCompletion),
}

#[derive(Debug)]
pub struct ConditionWait {
    key: u64,
    thread: GuestThreadId,
    wait: Arc<KernelWait>,
    effect: Option<WaitEffect>,
}
impl ConditionWait {
    pub fn registration(&self) -> &Arc<KernelWait> {
        &self.wait
    }
}
#[derive(Debug)]
pub struct ConditionClaim {
    key: u64,
    pub thread: GuestThreadId,
    pub tag: u32,
    pub mutex: u64,
    pub memory: nixe_memory::AllocationRange,
    wait: Arc<KernelWait>,
    _effect: WaitEffect,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressWaitCompletion {
    Success,
    InvalidOwner,
    InvalidMemory,
    TimedOut,
}
#[derive(Debug)]
pub struct MutexHandoff {
    address: u64,
    next: Option<(GuestThreadId, Arc<KernelWait>)>,
    pub next_tag: Option<u32>,
    pub more_waiters: bool,
    _effect: Option<WaitEffect>,
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
            state.operations.remove(&self.address).unwrap()
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
    ) -> AddressOperationGuard {
        loop {
            let pending = {
                let mut state = self.lock();
                let kernel = state.kernel().1.clone();
                if let Some(operation) = state.operations.get_mut(&address) {
                    let (_, event) = operation
                        .event
                        .get_or_insert_with(|| EventObject::create_pair(&kernel));
                    Some(event.clone())
                } else {
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
        let notifications = std::mem::take(&mut state.notifications);
        let retired = std::mem::take(&mut state.retired);
        let effects = std::mem::take(&mut state.effects);
        drop(state);
        for notification in notifications {
            notification.notify();
        }
        drop(retired);
        drop(effects);
    }
}

/// Payloads live here; no second timer registry or event generation is maintained.
#[derive(Debug, Default)]
pub struct AddressWaitState {
    kernel: OnceLock<(ProcessId, Arc<SchedulerState>)>,
    waiters: BTreeMap<GuestThreadId, AddressWaiter>,
    arbiter_waiters: BTreeMap<GuestThreadId, Arc<KernelWait>>,
    operations: BTreeMap<u64, AddressOperation>,
    notifications: Vec<EventNotifications>,
    retired: Vec<AddressWaiter>,
    effects: Vec<WaitEffect>,
}
impl AddressWaitState {
    fn effects_retired_push(&mut self, effect: WaitEffect) {
        self.effects.push(effect);
    }
    fn kernel(&self) -> &(ProcessId, Arc<SchedulerState>) {
        self.kernel.get().expect("registered process kernel")
    }
    fn first_ordered(&self, kind: OrderedWaitKind, address: u64) -> Option<GuestThreadId> {
        let (process, kernel) = self.kernel();
        kernel.first_ordered_waiter(*process, kind, address)
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
    ) -> Result<Arc<KernelWait>, SchedulerError> {
        let (process, kernel) = self.kernel();
        let wait = kernel.begin_mutex_wait(*process, LockKey::User(address), owner, thread)?;
        self.insert_waiter(address, thread, value, None, wait.clone());
        Ok(wait)
    }
    pub fn enqueue_condition(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
        lock: u64,
        memory: nixe_memory::AllocationRange,
    ) -> Result<ConditionWait, SchedulerError> {
        let (wait, effect) = self.kernel().1.begin_ordered_wait(
            thread,
            OrderedWaitKind::ConditionVariable,
            address,
            true,
        )?;
        self.insert_waiter(address, thread, value, Some(lock), wait.clone());
        let waiter = self.waiters.get_mut(&thread).unwrap();
        waiter.phase = WaitPhase::Registering;
        waiter.lock_memory = Some(memory);
        Ok(ConditionWait {
            key: address,
            thread,
            wait,
            effect,
        })
    }
    fn insert_waiter(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        value: u32,
        lock_address: Option<u64>,
        wait: Arc<KernelWait>,
    ) {
        let previous = self.waiters.insert(
            thread,
            AddressWaiter {
                address,
                value,
                lock_address,
                phase: WaitPhase::Waiting,
                lock_memory: None,
                wait,
            },
        );
        debug_assert!(previous.is_none());
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
        let waiter = self
            .waiters
            .get(&thread)
            .filter(|waiter| waiter.address == address)?;
        match waiter.wait.outcome() {
            Some(WaitOutcome::TimedOut) => Some(AddressWaitCompletion::TimedOut),
            Some(WaitOutcome::Cancelled | WaitOutcome::Terminated) => {
                Some(AddressWaitCompletion::InvalidOwner)
            }
            _ => {
                if let WaitPhase::Resolved(result) = waiter.phase {
                    Some(result)
                } else {
                    None
                }
            }
        }
    }
    pub fn signal_result(
        &mut self,
        address: u64,
        thread: GuestThreadId,
        result: AddressWaitCompletion,
    ) {
        if self.contains(address, thread) {
            self.publish_result(thread, result);
        }
    }
    fn publish_result(&mut self, thread: GuestThreadId, result: AddressWaitCompletion) {
        let waiter = self.waiters.get_mut(&thread).unwrap();
        if matches!(waiter.phase, WaitPhase::Resolved(_)) {
            return;
        }
        waiter.phase = WaitPhase::Resolved(result);
        let token = waiter.wait.token();
        let outcome = if result == AddressWaitCompletion::TimedOut {
            WaitOutcome::TimedOut
        } else {
            WaitOutcome::Ready
        };
        let (_, notifications) = self.kernel().1.resolve_wait_deferred(token, outcome);
        self.notifications.push(notifications);
    }
    fn condition_claim(
        &self,
        key: u64,
        thread: GuestThreadId,
        effect: WaitEffect,
    ) -> ConditionClaim {
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
            wait: waiter.wait.clone(),
            _effect: effect,
        }
    }
    pub fn claim_condition(&mut self, key: u64) -> Option<ConditionSignal> {
        let (process, kernel) = self.kernel();
        let (thread, effect) = kernel
            .claim_ordered_waiters(*process, OrderedWaitKind::ConditionVariable, key, 1, true)
            .0
            .pop()?;
        let waiter = self.waiters.get_mut(&thread).unwrap();
        if waiter.phase == WaitPhase::Registering {
            waiter.phase = WaitPhase::ConditionSignalled;
            // Its accepted release already retains exclusion through ConditionWait.
            self.effects_retired_push(effect);
            Some(ConditionSignal::PendingRelease)
        } else {
            waiter.phase = WaitPhase::Handoff;
            Some(ConditionSignal::Reacquire(
                self.condition_claim(key, thread, effect),
            ))
        }
    }
    pub fn finish_condition_release(
        &mut self,
        registration: &mut ConditionWait,
    ) -> Option<ConditionClaim> {
        let waiter = self.waiters.get_mut(&registration.thread)?;
        if waiter.address != registration.key || !Arc::ptr_eq(&waiter.wait, &registration.wait) {
            return None;
        }
        match waiter.phase {
            WaitPhase::Registering => {
                waiter.phase = WaitPhase::Waiting;
                None
            }
            WaitPhase::ConditionSignalled => {
                waiter.phase = WaitPhase::Handoff;
                registration.effect.take().map(|effect| {
                    self.condition_claim(registration.key, registration.thread, effect)
                })
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
            || !Arc::ptr_eq(&waiter.wait, &claim.wait)
            || waiter.phase != WaitPhase::Handoff
        {
            return Ok(());
        }
        match outcome {
            ConditionAcquire::Contended(owner) => {
                let (process, kernel) = self.kernel();
                kernel.add_waiter(
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
            outcome => self.signal_result(
                claim.key,
                claim.thread,
                match outcome {
                    ConditionAcquire::Acquired => AddressWaitCompletion::Success,
                    ConditionAcquire::InvalidOwner => AddressWaitCompletion::InvalidOwner,
                    ConditionAcquire::InvalidMemory => AddressWaitCompletion::InvalidMemory,
                    ConditionAcquire::Contended(_) => unreachable!(),
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
        let (process, kernel) = self.kernel();
        let (handoff, effect) = kernel.release_lock(*process, LockKey::User(address), owner)?;
        let next = handoff.next.map(|thread| {
            let waiter = self
                .waiters
                .get_mut(&thread)
                .expect("selected mutex waiter");
            waiter.phase = WaitPhase::Handoff;
            (thread, waiter.wait.clone())
        });
        Ok(MutexHandoff {
            address,
            next_tag: next.as_ref().map(|(thread, _)| self.waiters[thread].value),
            next,
            more_waiters: handoff.more_waiters,
            _effect: effect,
        })
    }
    pub fn finish_mutex_handoff(&mut self, claim: &MutexHandoff, result: AddressWaitCompletion) {
        let Some((thread, wait)) = &claim.next else {
            return;
        };
        if self.waiters.get(thread).is_some_and(|waiter| {
            waiter.address == claim.address
                && waiter.phase == WaitPhase::Handoff
                && Arc::ptr_eq(&waiter.wait, wait)
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
        if self.waiters.get(&thread).is_some_and(|waiter| {
            waiter.lock_address.is_none() && waiter.phase == WaitPhase::Waiting
        }) {
            self.publish_result(thread, AddressWaitCompletion::InvalidOwner);
        }
    }
    pub fn waiter_count(&self) -> usize {
        self.waiters.len() + self.arbiter_waiters.len()
    }
}

#[cfg(test)]
fn test_registry() -> Arc<AddressWaitRegistry> {
    use nixe_scheduler::{
        Completion, MachineSchedulerProfile, PriorityRange, ScheduledThreadConfig,
        VirtualCpuDescriptor, VirtualCpuId,
    };
    let core = VirtualCpuId::new(0);
    let kernel = SchedulerState::new(
        MachineSchedulerProfile::new(
            vec![VirtualCpuDescriptor::new(core, 0)],
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    );
    let registry = Arc::new(AddressWaitRegistry::default());
    registry.bind(ProcessId::new(1), kernel.clone());
    for id in 1..=4 {
        let thread = kernel
            .create_thread(ScheduledThreadConfig {
                process: ProcessId::new(1),
                base_priority: 20,
                ideal_vcpu: Some(core),
                affinity: kernel.profile().all_cores(),
            })
            .unwrap();
        assert_eq!(thread, GuestThreadId::new(id));
        kernel.make_ready(thread).unwrap();
        let lease = kernel.claim(core).unwrap().unwrap();
        kernel.complete(lease, Completion::Waiting).unwrap();
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    fn thread(id: u64) -> GuestThreadId {
        GuestThreadId::new(id)
    }
    fn memory() -> nixe_memory::AllocationRange {
        let machine = nixe_memory::Machine::new(4096).unwrap();
        let allocation = machine.account(4096).unwrap().allocate(4096).unwrap();
        nixe_memory::AllocationRange::new(allocation.pin(), 0, 4).unwrap()
    }
    fn register(registry: &AddressWaitRegistry) -> ConditionWait {
        registry
            .lock()
            .enqueue_condition(0x1000, thread(2), 7, 0x2000, memory())
            .unwrap()
    }
    #[test]
    fn ownership_transfer_uses_the_actual_wait_and_restores_donation() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        kernel.set_priority(thread(2), 10, None).unwrap();
        let wait = registry
            .lock()
            .enqueue(0x1000, thread(1), thread(2), 7)
            .unwrap();
        assert_eq!(kernel.thread(thread(1)).unwrap().effective_priority, 10);
        let handoff = registry.lock().claim_mutex(0x1000, thread(1)).unwrap();
        assert_eq!(handoff.next_tag, Some(7));
        assert_eq!(wait.outcome(), None);
        registry
            .lock()
            .finish_mutex_handoff(&handoff, AddressWaitCompletion::Success);
        drop(handoff);
        assert_eq!(wait.outcome(), Some(WaitOutcome::Ready));
        assert_eq!(kernel.thread(thread(1)).unwrap().effective_priority, 20);
        assert_eq!(
            registry.lock().completion(0x1000, thread(2)),
            Some(AddressWaitCompletion::Success)
        );
    }
    #[test]
    fn expired_waiters_are_removed_before_a_later_ownership_transfer() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        let wait = registry
            .lock()
            .enqueue(0x1000, thread(1), thread(2), 7)
            .unwrap();
        kernel.arm_deadline(&wait, 100);
        kernel.advance_time(100);
        assert_eq!(wait.outcome(), Some(WaitOutcome::TimedOut));
        assert_eq!(kernel.priority_donation_count(), 0);
        assert_eq!(
            registry
                .lock()
                .claim_mutex(0x1000, thread(1))
                .unwrap()
                .next_tag,
            None
        );
    }
    #[test]
    fn accepted_handoff_finishes_its_write_before_timeout_can_resume_guest_code() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        let wait = registry
            .lock()
            .enqueue(0x1000, thread(1), thread(2), 7)
            .unwrap();
        let handoff = registry.lock().claim_mutex(0x1000, thread(1)).unwrap();
        kernel.arm_deadline(&wait, 100);
        kernel.advance_time(100);
        assert_eq!(wait.outcome(), None);
        registry
            .lock()
            .finish_mutex_handoff(&handoff, AddressWaitCompletion::Success);
        let mut completed = std::pin::pin!(wait.wait());
        assert_eq!(
            completed
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        );
        drop(handoff);
        assert_eq!(
            completed
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(WaitOutcome::Ready)
        );
    }
    #[test]
    fn timeout_during_condition_release_does_not_consume_a_later_signal() {
        let registry = test_registry();
        let kernel = registry.lock().kernel().1.clone();
        let mut registration = register(&registry);
        let wait = registration.registration().clone();
        kernel.arm_deadline(&wait, 100);
        kernel.advance_time(100);
        assert_eq!(wait.outcome(), None);
        assert!(
            registry
                .lock()
                .finish_condition_release(&mut registration)
                .is_none()
        );
        drop(registration);
        assert_eq!(wait.outcome(), Some(WaitOutcome::TimedOut));
        assert!(registry.lock().claim_condition(0x1000).is_none());
    }
    #[test]
    fn signal_during_release_is_claimed_once_and_reacquires_only_after_release() {
        let registry = test_registry();
        let mut registration = register(&registry);
        assert!(matches!(
            registry.lock().claim_condition(0x1000),
            Some(ConditionSignal::PendingRelease)
        ));
        assert!(registry.lock().claim_condition(0x1000).is_none());
        let wait = registration.registration().clone();
        let claim = registry
            .lock()
            .finish_condition_release(&mut registration)
            .unwrap();
        registry
            .lock()
            .finish_condition_acquire(&claim, ConditionAcquire::Acquired, thread(1))
            .unwrap();
        drop(claim);
        drop(registration);
        assert_eq!(wait.outcome(), Some(WaitOutcome::Ready));
    }
    #[test]
    fn same_word_operations_suspend_without_holding_a_host_lock() {
        let registry = test_registry();
        let mut context = Context::from_waker(Waker::noop());
        let first = match std::pin::pin!(registry.address_operation(0x1000, thread(1)))
            .poll(&mut context)
        {
            Poll::Ready(guard) => guard,
            _ => panic!("free word"),
        };
        let mut second = std::pin::pin!(registry.address_operation(0x1000, thread(2)));
        assert!(second.as_mut().poll(&mut context).is_pending());
        let other = match std::pin::pin!(registry.address_operation(0x2000, thread(3)))
            .poll(&mut context)
        {
            Poll::Ready(guard) => guard,
            _ => panic!("disjoint word"),
        };
        drop(first);
        assert!(second.as_mut().poll(&mut context).is_ready());
        drop(other);
    }
}
