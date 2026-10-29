//! Kernel event state shares the run-queue lock; notifications leave that lock.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_synchronization_object.cpp
use super::*;
use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};

#[must_use]
#[derive(Debug, Default)]
pub struct EventNotifications {
    first: Option<(Waker, bool)>,
    rest: Vec<(Waker, bool)>,
}
impl EventNotifications {
    pub(super) fn push(&mut self, notification: (Waker, bool)) {
        if self.first.is_none() {
            self.first = Some(notification);
        } else {
            self.rest.push(notification);
        }
    }
    pub(super) fn extend(&mut self, notifications: impl IntoIterator<Item = (Waker, bool)>) {
        for notification in notifications {
            self.push(notification);
        }
    }
    pub fn notify(self) {
        for (waker, notify) in self.first.into_iter().chain(self.rest) {
            if notify {
                waker.wake();
            }
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct EventTable {
    entries: Vec<EventSlot>,
    free: Option<Index>,
}

#[derive(Debug)]
struct EventSlot {
    data: Option<EventData>,
    next: Option<Index>,
}

#[derive(Debug, Default)]
struct EventData {
    first: Option<EventWaiter>,
    last: Option<EventWaiter>,
    observers: BTreeMap<u64, Observer>,
    next_observer: u64,
}

#[derive(Debug)]
struct Observer {
    wake: Option<Waker>,
    signalled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EventWaiter {
    thread: Index,
    argument: u32,
}

#[derive(Debug)]
pub(super) struct EventWaitNode {
    event: ReadableEventObject,
    previous: Option<EventWaiter>,
    next: Option<EventWaiter>,
}

#[derive(Debug)]
pub enum WaitStart {
    Ready(WaitOutcome),
    Pending(Arc<KernelWait>),
}

impl EventTable {
    fn insert(&mut self) -> Index {
        let id = if let Some(id) = self.free {
            self.free = self.entries[slot_index(id)].next;
            id
        } else {
            let id = self
                .entries
                .len()
                .checked_add(1)
                .and_then(|value| u32::try_from(value).ok())
                .and_then(Index::new)
                .expect("kernel event identity exhausted");
            self.entries.push(EventSlot {
                data: None,
                next: None,
            });
            id
        };
        self.entries[slot_index(id)].data = Some(EventData::default());
        id
    }

    fn get(&self, id: Index) -> &EventData {
        self.entries[slot_index(id)]
            .data
            .as_ref()
            .expect("retained kernel event")
    }

    fn get_mut(&mut self, id: Index) -> &mut EventData {
        self.entries[slot_index(id)]
            .data
            .as_mut()
            .expect("retained kernel event")
    }

    fn remove(&mut self, id: Index) -> EventData {
        let slot = &mut self.entries[slot_index(id)];
        let data = slot.data.take().expect("retained kernel event");
        slot.next = self.free;
        self.free = Some(id);
        data
    }
}

struct EventReference {
    kernel: Arc<SchedulerState>,
    id: Index,
    // The event owns its only signal state. Kernel changes still share the
    // scheduler lock; host readiness queries need no scheduling transition.
    signalled: AtomicBool,
    auto_reset: bool,
}

impl Drop for EventReference {
    fn drop(&mut self) {
        let removed = self
            .kernel
            .edit(|state| Ok(state.events.remove(self.id)))
            .expect("retained kernel event");
        drop(removed);
    }
}

/// One event identity retained by guest handles and accepted kernel waits.
#[derive(Clone)]
pub struct EventObject {
    reference: Arc<EventReference>,
}

impl std::fmt::Debug for EventObject {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EventObject")
            .field("id", &self.reference.id)
            .finish()
    }
}

impl EventObject {
    pub fn new(kernel: &Arc<SchedulerState>) -> Self {
        Self::with_reset(kernel, false)
    }

    fn with_reset(kernel: &Arc<SchedulerState>, auto_reset: bool) -> Self {
        let id = kernel
            .edit(|state| Ok(state.events.insert()))
            .expect("kernel event allocation");
        Self {
            reference: Arc::new(EventReference {
                kernel: kernel.clone(),
                id,
                signalled: AtomicBool::new(false),
                auto_reset,
            }),
        }
    }

    pub fn create_pair(kernel: &Arc<SchedulerState>) -> (WritableEventObject, ReadableEventObject) {
        let event = Self::new(kernel);
        (
            WritableEventObject(event.clone()),
            ReadableEventObject(event),
        )
    }

    pub fn create_auto_reset_pair(
        kernel: &Arc<SchedulerState>,
    ) -> (WritableEventObject, ReadableEventObject) {
        let event = Self::with_reset(kernel, true);
        (
            WritableEventObject(event.clone()),
            ReadableEventObject(event),
        )
    }

    pub fn kernel(&self) -> &Arc<SchedulerState> {
        &self.reference.kernel
    }

    pub fn readable(&self) -> ReadableEventObject {
        ReadableEventObject(self.clone())
    }

    pub fn is_signalled(&self) -> bool {
        self.reference.signalled.load(Ordering::Acquire)
    }

    pub fn signal(&self) {
        self.publish_readiness(true).notify();
    }

    /// Publish readiness while the object's state is locked, then notify after unlocking it.
    pub fn publish_readiness(&self, ready: bool) -> EventNotifications {
        let (result, notifications) = self.reference.kernel.edit_deferred(|state| {
            if self.reference.signalled.load(Ordering::Relaxed) == ready {
                return Ok(());
            }
            self.reference.signalled.store(ready, Ordering::Release);
            if !ready {
                return Ok(());
            }
            let event = state.events.get_mut(self.reference.id);
            if self.reference.auto_reset {
                let mut candidate = event.first;
                let mut cursor = event.first;
                while let Some(node) = cursor {
                    if candidate.is_none_or(|best| {
                        state.slots[node.thread].priority < state.slots[best.thread].priority
                    }) {
                        candidate = Some(node);
                    }
                    cursor = state.slots[node.thread].wait_nodes[node.argument as usize].next;
                }
                if let Some(waiter) = candidate {
                    state.notify_event_waiter(
                        self.reference.id,
                        waiter,
                        &self.reference.kernel.profile,
                    );
                    self.reference.signalled.store(false, Ordering::Release);
                } else {
                    let event = state.events.get_mut(self.reference.id);
                    if let Some(observer) = event
                        .observers
                        .values_mut()
                        .find(|observer| !observer.signalled)
                    {
                        observer.signalled = true;
                        self.reference.signalled.store(false, Ordering::Release);
                        if let Some(waker) = observer.wake.take() {
                            state.deferred_wakers.push((waker, true));
                        }
                    }
                }
            } else {
                let observers = std::mem::take(&mut event.observers);
                state.deferred_wakers.extend(
                    observers
                        .into_values()
                        .filter_map(|observer| observer.wake)
                        .map(|waker| (waker, true)),
                );
                while let Some(waiter) = state.events.get(self.reference.id).first {
                    state.notify_event_waiter(
                        self.reference.id,
                        waiter,
                        &self.reference.kernel.profile,
                    );
                }
            }
            Ok(())
        });
        result.expect("retained kernel event");
        notifications
    }

    pub fn clear(&self) {
        self.reference
            .kernel
            .edit(|_| {
                self.reference.signalled.store(false, Ordering::Release);
                Ok(())
            })
            .expect("retained kernel event");
    }
}

impl SchedulerState {
    pub fn wait_on_objects<'a>(
        self: &Arc<Self>,
        thread: GuestThreadId,
        objects: impl IntoIterator<Item = &'a ReadableEventObject, IntoIter: Clone + ExactSizeIterator>,
        timeout_ns: Option<u64>,
        cancellable: bool,
    ) -> Result<WaitStart, SchedulerError> {
        let objects = objects.into_iter();
        if objects
            .clone()
            .any(|event| !Arc::ptr_eq(self, &event.0.reference.kernel))
        {
            return Err(SchedulerError::ForeignSynchronizationObject);
        }
        let deadline =
            timeout_ns.map(|timeout| self.clock.scheduler_time_ns().saturating_add(timeout));
        let result = self.edit(|state| {
            let index = state.slots.index(thread)?;
            if state.closed
                || matches!(
                    state.slots[index].lifecycle,
                    ThreadLifecycle::Exited | ThreadLifecycle::Faulted
                )
            {
                return Ok(WaitStart::Ready(WaitOutcome::Terminated));
            }
            if let Some((argument, object)) = objects
                .clone()
                .enumerate()
                .find(|(_, event)| event.is_signalled())
            {
                if object.0.reference.auto_reset {
                    object.0.reference.signalled.store(false, Ordering::Release);
                }
                return Ok(WaitStart::Ready(WaitOutcome::Signalled(argument as u32)));
            }
            if timeout_ns == Some(0) {
                return Ok(WaitStart::Ready(WaitOutcome::TimedOut));
            }
            if cancellable && state.slots[index].pending_cancel {
                state.slots[index].pending_cancel = false;
                return Ok(WaitStart::Ready(WaitOutcome::Cancelled));
            }
            let wait = state.begin_wait(index, &self.profile, Arc::downgrade(self))?;
            state.slots[index].wait_nodes = objects
                .clone()
                .map(|event| EventWaitNode {
                    event: event.clone(),
                    previous: None,
                    next: None,
                })
                .collect();
            for argument in 0..objects.len() {
                state.link_event_wait(EventWaiter {
                    thread: index,
                    argument: argument as u32,
                });
            }
            state.slots[index].cancellable = cancellable;
            wait.commit();
            Ok(WaitStart::Pending(wait))
        })?;
        if let WaitStart::Pending(wait) = &result
            && let Some(deadline) = deadline
        {
            self.arm_deadline(wait, deadline);
        }
        Ok(result)
    }
}

impl State {
    fn notify_event_waiter(
        &mut self,
        event: Index,
        waiter: EventWaiter,
        profile: &MachineSchedulerProfile,
    ) {
        let argument = self.slots[waiter.thread]
            .wait_nodes
            .iter()
            .position(|node| node.event.0.reference.id == event)
            .expect("linked event waiter");
        self.resolve_wait(
            waiter.thread,
            WaitOutcome::Signalled(argument as u32),
            profile,
        );
    }
    fn link_event_wait(&mut self, node: EventWaiter) {
        let id = self.slots[node.thread].wait_nodes[node.argument as usize]
            .event
            .0
            .reference
            .id;
        let previous = self.events.get(id).last;
        self.slots[node.thread].wait_nodes[node.argument as usize].previous = previous;
        if let Some(previous) = previous {
            self.slots[previous.thread].wait_nodes[previous.argument as usize].next = Some(node);
        } else {
            self.events.get_mut(id).first = Some(node);
        }
        self.events.get_mut(id).last = Some(node);
    }
    pub(super) fn unlink_event_wait(&mut self, thread: Index) {
        for argument in 0..self.slots[thread].wait_nodes.len() {
            let node = &self.slots[thread].wait_nodes[argument];
            let (id, previous, next) = (node.event.0.reference.id, node.previous, node.next);
            if let Some(previous) = previous {
                self.slots[previous.thread].wait_nodes[previous.argument as usize].next = next;
            } else {
                self.events.get_mut(id).first = next;
            }
            if let Some(next) = next {
                self.slots[next.thread].wait_nodes[next.argument as usize].previous = previous;
            } else {
                self.events.get_mut(id).last = previous;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct WritableEventObject(EventObject);

impl WritableEventObject {
    pub fn publish_readiness(&self, ready: bool) -> EventNotifications {
        self.0.publish_readiness(ready)
    }

    pub fn is_signalled(&self) -> bool {
        self.0.is_signalled()
    }
    pub fn signal(&self) {
        self.0.signal();
    }
    pub fn clear(&self) {
        self.0.clear();
    }
}

#[derive(Clone, Debug)]
pub struct ReadableEventObject(EventObject);

impl ReadableEventObject {
    pub fn kernel(&self) -> &Arc<SchedulerState> {
        self.0.kernel()
    }
    pub fn is_signalled(&self) -> bool {
        self.0.is_signalled()
    }
    pub fn clear(&self) {
        self.0.clear();
    }
    pub fn wait_async(&self) -> EventFuture {
        EventFuture {
            event: self.clone(),
            observer: None,
        }
    }
}

/// A host continuation observing the same event as guest synchronization.
pub struct EventFuture {
    event: ReadableEventObject,
    observer: Option<u64>,
}

impl Future for EventFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let reference = &this.event.0.reference;
        if this.observer.is_none() && !reference.auto_reset && this.event.is_signalled() {
            return Poll::Ready(());
        }
        reference
            .kernel
            .edit(|state| {
                let event = state.events.get_mut(reference.id);
                let claimed = this.observer.is_some_and(|id| {
                    event
                        .observers
                        .get(&id)
                        .is_some_and(|observer| observer.signalled)
                });
                if claimed || reference.signalled.load(Ordering::Acquire) {
                    if reference.auto_reset && !claimed {
                        reference.signalled.store(false, Ordering::Release);
                    }
                    if let Some(id) = this.observer.take()
                        && let Some(observer) = event.observers.remove(&id)
                        && let Some(waker) = observer.wake
                    {
                        state.deferred_wakers.push((waker, false));
                    }
                    return Ok(Poll::Ready(()));
                }
                let id = *this.observer.get_or_insert_with(|| {
                    event.next_observer = event
                        .next_observer
                        .checked_add(1)
                        .expect("kernel event observer identity exhausted");
                    event.next_observer
                });
                if event
                    .observers
                    .get(&id)
                    .and_then(|observer| observer.wake.as_ref())
                    .is_none_or(|old| !old.will_wake(context.waker()))
                    && let Some(old) = event.observers.insert(
                        id,
                        Observer {
                            wake: Some(context.waker().clone()),
                            signalled: false,
                        },
                    )
                    && let Some(old) = old.wake
                {
                    state.deferred_wakers.push((old, false));
                }
                Ok(Poll::Pending)
            })
            .expect("retained kernel event")
    }
}

impl Drop for EventFuture {
    fn drop(&mut self) {
        if let Some(id) = self.observer {
            let reference = &self.event.0.reference;
            let removed = reference
                .kernel
                .edit(|state| Ok(state.events.get_mut(reference.id).observers.remove(&id)))
                .expect("retained kernel event");
            drop(removed);
        }
    }
}

#[cfg(test)]
mod tests;
