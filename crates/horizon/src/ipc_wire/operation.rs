//! Ownership and wakeups for suspended built-in service calls.
use super::IpcWireError;
use nixe_memory::{AllocationId, MachineId};
use nixe_runtime::{EventFuture, HandleObject, ReadableEventObject};
use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
    task::{Context, Poll, Waker},
    time::Duration,
};

const STARTED: u8 = 1;
const OUTPUT: u8 = 2;
const CANCELLED: u8 = 4;

// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1445-L1525
/// Service effects finish before relinquishing their turn. A suspended response
/// write has no executing service effect and can be dropped on cancellation.
#[derive(Default)]
pub(crate) struct CallLifetime {
    state: AtomicU8,
}
impl CallLifetime {
    /// Returns whether an already executing effect must still be drained.
    pub fn cancel(&self) -> bool {
        self.state.fetch_or(CANCELLED, Ordering::AcqRel) & (STARTED | OUTPUT) == STARTED
    }
    pub fn check(&self) -> Result<(), IpcWireError> {
        if self.state.load(Ordering::Acquire) & CANCELLED != 0 {
            Err(IpcWireError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub fn begin(&self) -> Result<(), IpcWireError> {
        self.state
            .compare_exchange(0, STARTED, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| IpcWireError::Cancelled)
    }
    pub fn begin_output(&self) -> Result<(), IpcWireError> {
        self.state
            .compare_exchange(
                STARTED,
                STARTED | OUTPUT,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| IpcWireError::Cancelled)
    }
    pub fn end_output(&self) {
        self.state.fetch_and(!OUTPUT, Ordering::Release);
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
}
impl Scheduling {
    fn next(&mut self) -> Result<u64, IpcWireError> {
        self.next = self.next.checked_add(1).ok_or(IpcWireError::Internal(
            "service continuation identity exhausted",
        ))?;
        Ok(self.next)
    }
}
pub(crate) struct ServiceScheduling {
    state: Mutex<Scheduling>,
    kernel: Arc<nixe_scheduler::SchedulerState>,
}
impl std::fmt::Debug for ServiceScheduling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("ServiceScheduling")
            .field("owners", &state.queues.len())
            .finish()
    }
}
impl ServiceScheduling {
    pub fn new(kernel: &Arc<nixe_scheduler::SchedulerState>) -> Self {
        Self {
            state: Mutex::default(),
            kernel: kernel.clone(),
        }
    }
    pub fn enter(self: &Arc<Self>, key: ServiceKey) -> Result<ServiceTurn, IpcWireError> {
        let id = {
            let mut state = self.state.lock().unwrap();
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
            event: event.wait_async(),
            deadline: timeout.map(|timeout| {
                self.kernel.deadline(
                    self.kernel
                        .clock()
                        .scheduler_time_ns()
                        .saturating_add(u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX)),
                )
            }),
        }
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
            let mut state = this.scheduler.state.lock().unwrap();
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
            let mut state = self.scheduler.state.lock().unwrap();
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
    event: EventFuture,
    deadline: Option<nixe_scheduler::Deadline>,
}
impl Future for ServiceEvent {
    type Output = Result<(), IpcWireError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if Pin::new(&mut this.event).poll(context).is_ready() {
            return Poll::Ready(Ok(()));
        }
        if let Some(deadline) = &mut this.deadline {
            match Pin::new(deadline).poll(context) {
                Poll::Ready(Ok(())) => return Poll::Ready(Ok(())),
                Poll::Ready(Err(_)) => return Poll::Ready(Err(IpcWireError::Cancelled)),
                Poll::Pending => {}
            }
        }
        Poll::Pending
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
        ServiceKey::Session(HandleObject::new(nixe_runtime::EventObject::new(
            &crate::test_memory::kernel(),
        )))
    }

    #[test]
    fn operations_keep_fifo_order_and_cancelled_head_wakes_the_next_request() {
        let scheduler = Arc::new(ServiceScheduling::new(&crate::test_memory::kernel()));
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
        assert!(scheduler.state.lock().unwrap().queues.is_empty());
    }

    #[test]
    fn a_waiting_service_does_not_prevent_an_unrelated_owner_from_running() {
        let scheduler = Arc::new(ServiceScheduling::new(&crate::test_memory::kernel()));
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
        let scheduler = Arc::new(ServiceScheduling::new(&crate::test_memory::kernel()));
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(wakes.clone());
        let mut context = Context::from_waker(&waker);
        let (writer, reader) =
            nixe_runtime::EventObject::create_pair(&crate::test_memory::kernel());
        let mut wait = scheduler.wait(reader.clone(), Some(Duration::from_secs(60)));
        assert!(Pin::new(&mut wait).poll(&mut context).is_pending());
        assert!(scheduler.kernel.deadline_count() != 0);
        writer.signal();
        assert!(Pin::new(&mut wait).poll(&mut context).is_ready());
        drop(wait);
        assert!(scheduler.kernel.deadline_count() == 0);
        reader.clear();
        let mut cancelled = scheduler.wait(reader.clone(), Some(Duration::from_secs(60)));
        assert!(Pin::new(&mut cancelled).poll(&mut context).is_pending());
        drop(cancelled);
        let before = wakes.0.load(Ordering::Relaxed);
        writer.signal();
        assert_eq!(wakes.0.load(Ordering::Relaxed), before);
        assert!(scheduler.kernel.deadline_count() == 0);
        reader.clear();
        let mut deadline = scheduler.wait(reader, Some(Duration::from_millis(1)));
        assert!(Pin::new(&mut deadline).poll(&mut context).is_pending());
        scheduler.kernel.advance_time(2_000_000);
        assert!(Pin::new(&mut deadline).poll(&mut context).is_ready());
        assert!(scheduler.kernel.deadline_count() == 0);
    }
}
