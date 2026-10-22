//! Ownership and wakeups for suspended built-in service calls.
use super::IpcWireError;
use nixe_memory::{AllocationId, MachineId};
use nixe_runtime::{EventFuture, HandleObject, ReadableEventObject};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

/// Cancellation suppresses the reply. Once service effects may have started,
/// the owner finishes the operation before releasing its serialization turn.
#[derive(Default)]
pub(crate) struct CallLifetime {
    cancelled: AtomicBool,
    started: AtomicBool,
}
impl CallLifetime {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }
    pub fn check(&self) -> Result<(), IpcWireError> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(IpcWireError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub fn begin(&self) -> Result<(), IpcWireError> {
        self.check()?;
        self.started.store(true, Ordering::Release);
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ServiceKey {
    Session(HandleObject),
    Shared(MachineId, AllocationId),
    Clock(MachineId, AllocationId, u8),
}
struct Waiter {
    id: u64,
    wake: Option<Waker>,
}
#[derive(Default)]
struct Scheduling {
    next: u64,
    queues: HashMap<ServiceKey, VecDeque<Waiter>>,
    timers: BTreeMap<(Instant, u64), Waker>,
}
impl Scheduling {
    fn next(&mut self) -> Result<u64, IpcWireError> {
        self.next = self.next.checked_add(1).ok_or(IpcWireError::Internal(
            "service continuation identity exhausted",
        ))?;
        Ok(self.next)
    }
}
#[derive(Default)]
pub(crate) struct ServiceScheduling(Mutex<Scheduling>);
impl std::fmt::Debug for ServiceScheduling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.0.lock().unwrap();
        f.debug_struct("ServiceScheduling")
            .field("owners", &state.queues.len())
            .field("timers", &state.timers.len())
            .finish()
    }
}
impl ServiceScheduling {
    pub fn enter(self: &Arc<Self>, key: ServiceKey) -> Result<ServiceTurn, IpcWireError> {
        let id = {
            let mut state = self.0.lock().unwrap();
            let id = state.next()?;
            state
                .queues
                .entry(key.clone())
                .or_default()
                .push_back(Waiter { id, wake: None });
            id
        };
        Ok(ServiceTurn {
            scheduler: self.clone(),
            key,
            id,
        })
    }
    pub fn wait(
        self: &Arc<Self>,
        event: ReadableEventObject,
        timeout: Option<Duration>,
    ) -> ServiceEvent {
        ServiceEvent {
            scheduler: self.clone(),
            event: event.wait_async(),
            deadline: timeout.and_then(|timeout| Instant::now().checked_add(timeout)),
            timer: None,
        }
    }
    pub fn poll_deadlines(&self) {
        let now = Instant::now();
        loop {
            let due = {
                let mut state = self.0.lock().unwrap();
                if state
                    .timers
                    .first_key_value()
                    .is_none_or(|((deadline, _), _)| *deadline > now)
                {
                    break;
                }
                state.timers.pop_first().map(|(_, waker)| waker)
            };
            if let Some(waker) = due {
                waker.wake();
            }
        }
    }
    pub fn next_deadline(&self) -> Option<Duration> {
        self.0
            .lock()
            .unwrap()
            .timers
            .first_key_value()
            .map(|((deadline, _), _)| deadline.saturating_duration_since(Instant::now()))
    }
}

/// The request keeps this turn until its response has been written. A dropped
/// queued request relinquishes its place without executing service effects.
pub(crate) struct ServiceTurn {
    scheduler: Arc<ServiceScheduling>,
    key: ServiceKey,
    id: u64,
}
impl Future for &ServiceTurn {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = *self;
        let old = {
            let mut state = this.scheduler.0.lock().unwrap();
            let queue = state
                .queues
                .get_mut(&this.key)
                .expect("registered service turn");
            if queue.front().is_some_and(|waiter| waiter.id == this.id) {
                return Poll::Ready(());
            }
            let waiter = queue
                .iter_mut()
                .find(|waiter| waiter.id == this.id)
                .expect("queued service turn");
            waiter.wake.replace(context.waker().clone())
        };
        drop(old);
        Poll::Pending
    }
}
impl Drop for ServiceTurn {
    fn drop(&mut self) {
        let (removed, next, key) = {
            let mut state = self.scheduler.0.lock().unwrap();
            let queue = state
                .queues
                .get_mut(&self.key)
                .expect("registered service turn");
            let index = queue
                .iter()
                .position(|waiter| waiter.id == self.id)
                .expect("registered service waiter");
            let removed = queue.remove(index);
            let next = if index == 0 {
                queue.front_mut().and_then(|waiter| waiter.wake.take())
            } else {
                None
            };
            let key = if queue.is_empty() {
                state.queues.remove_entry(&self.key)
            } else {
                None
            };
            (removed, next, key)
        };
        drop(removed);
        drop(key);
        if let Some(waker) = next {
            waker.wake();
        }
    }
}

pub(crate) struct ServiceEvent {
    scheduler: Arc<ServiceScheduling>,
    event: EventFuture,
    deadline: Option<Instant>,
    timer: Option<(Instant, u64)>,
}
impl Future for ServiceEvent {
    type Output = Result<(), IpcWireError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if Pin::new(&mut this.event).poll(context).is_ready()
            || this
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Poll::Ready(Ok(()));
        }
        if let Some(deadline) = this.deadline {
            let old = {
                let mut state = this.scheduler.0.lock().unwrap();
                let key = match this.timer {
                    Some(key) => key,
                    None => {
                        let key = (deadline, state.next()?);
                        this.timer = Some(key);
                        key
                    }
                };
                state.timers.insert(key, context.waker().clone())
            };
            drop(old);
        }
        Poll::Pending
    }
}
impl Drop for ServiceEvent {
    fn drop(&mut self) {
        if let Some(key) = self.timer.take() {
            let removed = self.scheduler.0.lock().unwrap().timers.remove(&key);
            drop(removed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    #[derive(Default)]
    struct Wakes(AtomicUsize);
    impl Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn ready(turn: &ServiceTurn, context: &mut Context<'_>) -> bool {
        Pin::new(&mut &*turn).poll(context).is_ready()
    }
    fn key() -> ServiceKey {
        ServiceKey::Session(HandleObject::new(nixe_runtime::ProcessObject::new(9)))
    }

    #[test]
    fn operations_keep_fifo_order_and_cancelled_head_wakes_the_next_request() {
        let scheduler = Arc::new(ServiceScheduling::default());
        let key = key();
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(wakes.clone());
        let mut context = Context::from_waker(&waker);
        let first = scheduler.enter(key.clone()).unwrap();
        let second = scheduler.enter(key.clone()).unwrap();
        let cancelled = scheduler.enter(key.clone()).unwrap();
        let last = scheduler.enter(key).unwrap();
        assert!(ready(&first, &mut context));
        assert!(!ready(&second, &mut context));
        assert!(!ready(&cancelled, &mut context));
        assert!(!ready(&last, &mut context));
        drop(cancelled);
        assert_eq!(wakes.0.load(Ordering::Relaxed), 0);
        drop(first);
        assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
        assert!(ready(&second, &mut context));
        assert!(!ready(&last, &mut context));
        drop(second);
        assert_eq!(wakes.0.load(Ordering::Relaxed), 2);
        assert!(ready(&last, &mut context));
        drop(last);
        assert!(scheduler.0.lock().unwrap().queues.is_empty());
    }

    #[test]
    fn a_waiting_service_does_not_prevent_an_unrelated_owner_from_running() {
        let scheduler = Arc::new(ServiceScheduling::default());
        let first_key = key();
        let _first = scheduler.enter(first_key.clone()).unwrap();
        let blocked = scheduler.enter(first_key).unwrap();
        let independent = scheduler.enter(key()).unwrap();
        let mut context = Context::from_waker(Waker::noop());
        assert!(!ready(&blocked, &mut context));
        assert!(ready(&independent, &mut context));
    }

    #[test]
    fn event_and_deadline_waits_release_their_registrations_on_completion_or_cancel() {
        let scheduler = Arc::new(ServiceScheduling::default());
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(wakes.clone());
        let mut context = Context::from_waker(&waker);
        let (writer, reader) = nixe_runtime::EventObject::create_pair();
        let mut wait = scheduler.wait(reader.clone(), Some(Duration::from_secs(60)));
        assert!(Pin::new(&mut wait).poll(&mut context).is_pending());
        assert!(scheduler.next_deadline().is_some());
        writer.signal();
        assert!(Pin::new(&mut wait).poll(&mut context).is_ready());
        drop(wait);
        assert!(scheduler.next_deadline().is_none());
        reader.clear();
        let mut cancelled = scheduler.wait(reader.clone(), Some(Duration::from_secs(60)));
        assert!(Pin::new(&mut cancelled).poll(&mut context).is_pending());
        drop(cancelled);
        let before = wakes.0.load(Ordering::Relaxed);
        writer.signal();
        assert_eq!(wakes.0.load(Ordering::Relaxed), before);
        assert!(scheduler.next_deadline().is_none());
        reader.clear();
        let mut deadline = scheduler.wait(reader, Some(Duration::from_millis(1)));
        assert!(Pin::new(&mut deadline).poll(&mut context).is_pending());
        std::thread::sleep(Duration::from_millis(2));
        scheduler.poll_deadlines();
        assert!(Pin::new(&mut deadline).poll(&mut context).is_ready());
        assert!(scheduler.next_deadline().is_none());
    }
}
