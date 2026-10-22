use super::*;

/// A minimal event object with state shared by duplicated handles.
#[derive(Clone, Debug)]
pub struct EventObject {
    state: Arc<EventState>,
}

struct EventState {
    signalled: std::sync::atomic::AtomicBool,
    source: Option<crate::ExternalEventSource>,
    generation: Mutex<EventGeneration>,
    changed: Condvar,
}

struct EventGeneration {
    value: u64,
    next_watcher_id: u64,
    watchers: BTreeMap<u64, Arc<dyn Fn() + Send + Sync>>,
}

impl Debug for EventState {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let generation = self
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        formatter
            .debug_struct("EventState")
            .field("signalled", &self.signalled)
            .field("generation", &generation.value)
            .field("watchers", &generation.watchers.len())
            .finish()
    }
}

/// Cancellation handle for one non-blocking event observer.
pub(crate) struct EventWatchRegistration {
    state: Weak<EventState>,
    id: u64,
}

impl Drop for EventWatchRegistration {
    fn drop(&mut self) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let removed = state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .watchers
            .remove(&self.id);
        drop(removed);
    }
}

/// Host scheduling result from sleeping on one runtime event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EventWaitOutcome {
    Signalled,
    TimedOut,
}

impl Default for EventObject {
    fn default() -> Self {
        Self::new()
    }
}

impl EventObject {
    #[must_use]
    pub fn new() -> Self {
        Self::with_source(None)
    }

    fn with_source(source: Option<crate::ExternalEventSource>) -> Self {
        Self {
            state: Arc::new(EventState {
                signalled: std::sync::atomic::AtomicBool::new(false),
                source,
                generation: Mutex::new(EventGeneration {
                    value: 0,
                    next_watcher_id: 1,
                    watchers: BTreeMap::new(),
                }),
                changed: Condvar::new(),
            }),
        }
    }

    /// Creates the writable/readable handle views returned by Horizon's
    /// `CreateEvent` without duplicating the underlying signal state.
    #[must_use]
    pub fn create_pair() -> (WritableEventObject, ReadableEventObject) {
        let event = Self::new();
        (
            WritableEventObject(event.clone()),
            ReadableEventObject(event),
        )
    }

    /// Creates an event pair carrying its device-ingress classification all
    /// the way to the runtime coordinator.
    #[must_use]
    pub fn create_pair_with_source(
        source: crate::ExternalEventSource,
    ) -> (WritableEventObject, ReadableEventObject) {
        let event = Self::with_source(Some(source));
        (
            WritableEventObject(event.clone()),
            ReadableEventObject(event),
        )
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.state
            .signalled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn signal(&self) {
        self.state
            .signalled
            .store(true, std::sync::atomic::Ordering::Release);
        let mut generation = self
            .state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        generation.value = generation.value.wrapping_add(1);
        let watchers = std::mem::take(&mut generation.watchers);
        self.state.changed.notify_all();
        drop(generation);
        for watcher in watchers.into_values() {
            watcher();
        }
    }

    pub fn clear(&self) {
        self.state
            .signalled
            .store(false, std::sync::atomic::Ordering::Release);
    }

    fn wait_until(&self, deadline: Option<Instant>) -> EventWaitOutcome {
        if self.is_signalled() {
            return EventWaitOutcome::Signalled;
        }
        let mut generation = self
            .state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let observed = generation.value;
        loop {
            if self.is_signalled() || generation.value != observed {
                return EventWaitOutcome::Signalled;
            }
            generation = match deadline {
                None => self
                    .state
                    .changed
                    .wait(generation)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return EventWaitOutcome::TimedOut;
                    }
                    let (next, timeout) = self
                        .state
                        .changed
                        .wait_timeout(generation, deadline.saturating_duration_since(now))
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if timeout.timed_out() && Instant::now() >= deadline {
                        return EventWaitOutcome::TimedOut;
                    }
                    next
                }
            };
        }
    }

    fn watch(&self, watcher: Arc<dyn Fn() + Send + Sync>) -> EventWatchRegistration {
        let mut generation = self
            .state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = generation.next_watcher_id;
        generation.next_watcher_id = generation.next_watcher_id.wrapping_add(1).max(1);
        let notify_now = self.is_signalled();
        if !notify_now {
            generation.watchers.insert(id, Arc::clone(&watcher));
        }
        drop(generation);
        if notify_now {
            watcher();
        }
        EventWatchRegistration {
            state: Arc::downgrade(&self.state),
            id,
        }
    }
}

/// Writable side of a kernel event pair.
#[derive(Clone, Debug)]
pub struct WritableEventObject(pub(super) EventObject);

impl WritableEventObject {
    pub(crate) fn same_event(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0.state, &other.0.state)
    }
    #[must_use]
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

/// Readable synchronization side of a kernel event pair.
#[derive(Clone, Debug)]
pub struct ReadableEventObject(pub(super) EventObject);

impl ReadableEventObject {
    pub(crate) fn same_event(&self, writer: &WritableEventObject) -> bool {
        Arc::ptr_eq(&self.0.state, &writer.0.state)
    }
    #[must_use]
    pub fn source(&self) -> Option<crate::ExternalEventSource> {
        self.0.state.source
    }
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.0.is_signalled()
    }

    pub fn clear(&self) {
        self.0.clear();
    }

    pub(crate) fn watch(&self, watcher: Arc<dyn Fn() + Send + Sync>) -> EventWatchRegistration {
        self.0.watch(watcher)
    }

    /// Waits without occupying a host worker. The future retains the event,
    /// and dropping it removes its observer.
    pub fn wait_async(&self) -> EventFuture {
        EventFuture {
            event: self.clone(),
            registration: None,
        }
    }

    /// Sleeps until this event is signalled or the relative timeout expires.
    /// `None` represents an infinite wait.
    #[must_use]
    pub fn wait(&self, timeout: Option<Duration>) -> EventWaitOutcome {
        let deadline = timeout.and_then(|duration| Instant::now().checked_add(duration));
        self.0.wait_until(deadline)
    }
}

/// An owned level-triggered event wait, safe to retain across guest suspension.
pub struct EventFuture {
    event: ReadableEventObject,
    registration: Option<EventWatchRegistration>,
}
impl std::future::Future for EventFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        let this = self.get_mut();
        if this.event.is_signalled() {
            this.registration = None;
            return std::task::Poll::Ready(());
        }
        // signal() consumes observers. Replace the registration on every poll
        // so a cleared level followed by another signal cannot lose a wake.
        this.registration = None;
        let waker = context.waker().clone();
        this.registration = Some(this.event.watch(Arc::new(move || waker.wake_by_ref())));
        if this.event.is_signalled() {
            this.registration = None;
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;
    use std::{
        future::Future,
        pin::Pin,
        sync::atomic::{AtomicUsize, Ordering},
        task::{Context, Poll, Wake, Waker},
    };
    #[derive(Default)]
    struct Wakes(AtomicUsize);
    impl Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[test]
    fn async_wait_rearms_after_a_signal_is_cleared_and_releases_on_cancel() {
        let (writer, reader) = EventObject::create_pair();
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(wakes.clone());
        let mut context = Context::from_waker(&waker);
        let mut wait = reader.wait_async();
        assert_eq!(Pin::new(&mut wait).poll(&mut context), Poll::Pending);
        writer.signal();
        assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
        reader.clear();
        assert_eq!(Pin::new(&mut wait).poll(&mut context), Poll::Pending);
        writer.signal();
        assert_eq!(wakes.0.load(Ordering::Relaxed), 2);
        assert_eq!(Pin::new(&mut wait).poll(&mut context), Poll::Ready(()));
        reader.clear();
        let mut cancelled = reader.wait_async();
        assert_eq!(Pin::new(&mut cancelled).poll(&mut context), Poll::Pending);
        drop(cancelled);
        writer.signal();
        assert_eq!(wakes.0.load(Ordering::Relaxed), 2);
        let mut already_ready = reader.wait_async();
        assert_eq!(
            Pin::new(&mut already_ready).poll(&mut context),
            Poll::Ready(())
        );
    }
    #[test]
    fn async_wait_replaces_its_waker_without_retaining_the_old_executor() {
        let (writer, reader) = EventObject::create_pair();
        let first = Arc::new(Wakes::default());
        let second = Arc::new(Wakes::default());
        let mut wait = reader.wait_async();
        assert!(
            Pin::new(&mut wait)
                .poll(&mut Context::from_waker(&Waker::from(first.clone())))
                .is_pending()
        );
        assert!(
            Pin::new(&mut wait)
                .poll(&mut Context::from_waker(&Waker::from(second.clone())))
                .is_pending()
        );
        writer.signal();
        assert_eq!(first.0.load(Ordering::Relaxed), 0);
        assert_eq!(second.0.load(Ordering::Relaxed), 1);
    }
}
