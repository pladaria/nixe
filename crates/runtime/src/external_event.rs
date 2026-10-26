use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nixe_scheduler::WakeToken;

use crate::ReadableEventObject;
use crate::handle::EventWatchRegistration;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalEventSource {
    Timer,
    Device,
    GpuCompletion,
    Display,
    Input,
    Ipc,
    Worker,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExternalEventSequence(u64);

impl ExternalEventSequence {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalEvent {
    HostStop,
    /// Suspends/resumes host emulation without changing guest thread activity.
    HostPause(bool),
    /// Wakes the host coordinator without making a guest thread runnable.
    HostService(ExternalEventSource),
    Wake {
        source: ExternalEventSource,
        token: WakeToken,
    },
    CancelWait(WakeToken),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequencedExternalEvent {
    pub sequence: ExternalEventSequence,
    pub event: ExternalEvent,
}

#[derive(Debug)]
struct EventPublication {
    sender: SyncSender<CoordinatorEvent>,
    next_sequence: Mutex<u64>,
    pending: AtomicUsize,
    capacity: usize,
}

impl EventPublication {
    fn publish(&self, event: ExternalEvent) -> Result<(), ExternalEventSendError> {
        let mut sequence = self
            .next_sequence
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.pending.load(Ordering::Acquire) >= self.capacity {
            return Err(ExternalEventSendError::Full);
        }
        let next = sequence
            .checked_add(1)
            .ok_or(ExternalEventSendError::SequenceExhausted)?;
        self.pending.fetch_add(1, Ordering::Relaxed);
        if let Err(error) =
            self.sender
                .try_send(CoordinatorEvent::External(SequencedExternalEvent {
                    sequence: ExternalEventSequence(*sequence),
                    event,
                }))
        {
            self.pending.fetch_sub(1, Ordering::Release);
            return Err(match error {
                TrySendError::Full(_) => ExternalEventSendError::Full,
                TrySendError::Disconnected(_) => ExternalEventSendError::Closed,
            });
        }
        *sequence = next;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ExternalEventSender {
    publication: Arc<EventPublication>,
    wait_groups: Arc<Mutex<HashMap<WakeToken, Arc<ExternalWaitGroup>>>>,
    overflowed: Arc<AtomicBool>,
}

struct ExternalWaitGroup {
    won: AtomicBool,
    registrations: Mutex<Vec<EventWatchRegistration>>,
}

impl ExternalWaitGroup {
    fn new() -> Self {
        Self {
            won: AtomicBool::new(false),
            registrations: Mutex::new(Vec::new()),
        }
    }

    fn add(&self, registration: EventWatchRegistration) {
        let mut registrations = self
            .registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !self.won.load(Ordering::Acquire) {
            registrations.push(registration);
        }
    }

    fn cancel(&self) {
        self.won.store(true, Ordering::Release);
        self.registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

impl std::fmt::Debug for ExternalWaitGroup {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalWaitGroup")
            .field("won", &self.won)
            .finish_non_exhaustive()
    }
}

impl ExternalEventSender {
    /// Coalescing is owned by the producer's mailbox. Overflow is reported by
    /// the coordinator instead of silently losing a host service notification.
    pub fn notify_host_service(&self, source: ExternalEventSource) {
        if let Err(error) = self.submit(ExternalEvent::HostService(source))
            && matches!(
                error,
                ExternalEventSendError::Full | ExternalEventSendError::SequenceExhausted
            )
        {
            self.overflowed.store(true, Ordering::Release);
        }
    }

    pub fn submit(&self, event: ExternalEvent) -> Result<(), ExternalEventSendError> {
        self.publication.publish(event)
    }

    pub fn wake(
        &self,
        source: ExternalEventSource,
        token: WakeToken,
    ) -> Result<(), ExternalEventSendError> {
        self.submit(ExternalEvent::Wake { source, token })
    }

    pub fn gpu_completion(&self, token: WakeToken) -> Result<(), ExternalEventSendError> {
        self.wake(ExternalEventSource::GpuCompletion, token)
    }

    pub fn display_event(&self, token: WakeToken) -> Result<(), ExternalEventSendError> {
        self.wake(ExternalEventSource::Display, token)
    }

    pub fn input_event(&self, token: WakeToken) -> Result<(), ExternalEventSendError> {
        self.wake(ExternalEventSource::Input, token)
    }

    /// Registers a non-blocking observer which publishes readiness directly to
    /// the bounded inbox. Deadlines belong to the runtime coordinator.
    pub fn watch_readable_event(
        &self,
        event: ReadableEventObject,
        timeout: Option<Duration>,
        token: WakeToken,
        source: ExternalEventSource,
    ) -> Result<(), ExternalEventSendError> {
        if timeout.is_some() {
            return Err(ExternalEventSendError::HostDeadlineUnsupported);
        }
        let source = event.source().unwrap_or(source);
        let group = {
            let mut groups = self
                .wait_groups
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            groups
                .entry(token)
                .or_insert_with(|| Arc::new(ExternalWaitGroup::new()))
                .clone()
        };
        let publication = Arc::clone(&self.publication);
        let overflowed = Arc::clone(&self.overflowed);
        let callback_group = Arc::clone(&group);
        let registration = event.watch(Arc::new(move || {
            if callback_group.won.swap(true, Ordering::AcqRel) {
                return;
            }
            if let Err(error) = publication.publish(ExternalEvent::Wake { source, token })
                && matches!(
                    error,
                    ExternalEventSendError::Full | ExternalEventSendError::SequenceExhausted
                )
            {
                overflowed.store(true, Ordering::Release);
            }
        }));
        group.add(registration);
        Ok(())
    }

    pub(crate) fn cancel_watchers(&self, token: WakeToken) {
        if let Some(group) = self
            .wait_groups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&token)
        {
            group.cancel();
        }
    }

    pub(crate) fn watcher_group_count(&self) -> usize {
        self.wait_groups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "bounded ingress transfers CPU state inline without allocating each completion"
)]
pub(crate) enum CoordinatorEvent {
    External(SequencedExternalEvent),
    Completed(crate::coordinator::worker::WorkerResult),
}

pub(crate) struct ExternalEventInbox {
    sender: ExternalEventSender,
    receiver: Receiver<CoordinatorEvent>,
}

impl ExternalEventInbox {
    pub fn bounded(capacity: usize, cpu_workers: usize) -> Result<Self, ExternalEventSendError> {
        if capacity == 0 {
            return Err(ExternalEventSendError::ZeroCapacity);
        }
        // A native completion must remain publishable when host ingress is full:
        // synchronous core-install/retirement replies can depend on that worker.
        // Leases limit CPU completions to one per core; host events have their
        // own quota, retained while a drain temporarily defers their handling.
        let (sender, receiver) = sync_channel(capacity + cpu_workers);
        Ok(Self {
            sender: ExternalEventSender {
                publication: Arc::new(EventPublication {
                    sender,
                    next_sequence: Mutex::new(1),
                    pending: AtomicUsize::new(0),
                    capacity,
                }),
                wait_groups: Arc::new(Mutex::new(HashMap::new())),
                overflowed: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        })
    }

    #[must_use]
    pub fn sender(&self) -> ExternalEventSender {
        self.sender.clone()
    }

    pub(crate) fn worker_sender(&self) -> SyncSender<CoordinatorEvent> {
        self.sender.publication.sender.clone()
    }

    pub(crate) fn consume_external(&self) {
        self.sender
            .publication
            .pending
            .fetch_sub(1, Ordering::Release);
    }

    pub(crate) fn try_recv(&self) -> Result<Option<CoordinatorEvent>, ExternalEventSendError> {
        if self.sender.overflowed.swap(false, Ordering::AcqRel) {
            return Err(ExternalEventSendError::Full);
        }
        match self.receiver.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ExternalEventSendError::Closed),
        }
    }

    pub(crate) fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<CoordinatorEvent>, ExternalEventSendError> {
        match self.receiver.recv_timeout(timeout) {
            Ok(event) => Ok(Some(event)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(ExternalEventSendError::Closed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalEventSendError {
    ZeroCapacity,
    Full,
    Closed,
    HostDeadlineUnsupported,
    SequenceExhausted,
}

impl Display for ExternalEventSendError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "external event inbox rejected event: {self:?}")
    }
}

impl Error for ExternalEventSendError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventObject;
    use nixe_scheduler::{GuestThreadId, WakeGeneration};
    use std::sync::Barrier;
    use std::thread;

    fn external(
        inbox: &ExternalEventInbox,
        event: Option<CoordinatorEvent>,
    ) -> Option<SequencedExternalEvent> {
        event.map(|event| match event {
            CoordinatorEvent::External(event) => {
                inbox.consume_external();
                event
            }
            CoordinatorEvent::Completed(_) => panic!("this fixture has no CPU workers"),
        })
    }

    #[test]
    fn host_service_overflow_is_reported_to_the_consumer() {
        let inbox = ExternalEventInbox::bounded(1, 0).unwrap();
        let sender = inbox.sender();
        sender.notify_host_service(ExternalEventSource::Input);
        sender.notify_host_service(ExternalEventSource::Input);
        assert_eq!(
            inbox.try_recv().map(|event| external(&inbox, event)),
            Err(ExternalEventSendError::Full)
        );
        assert_eq!(
            inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap()
                .unwrap()
                .event,
            ExternalEvent::HostService(ExternalEventSource::Input)
        );
    }

    #[test]
    fn bounded_inbox_reports_backpressure_without_blocking() {
        let inbox = ExternalEventInbox::bounded(1, 0).unwrap();
        let sender = inbox.sender();
        sender.submit(ExternalEvent::HostStop).unwrap();
        assert_eq!(
            sender.submit(ExternalEvent::HostStop),
            Err(ExternalEventSendError::Full)
        );
        assert_eq!(
            inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap()
                .map(|event| event.event),
            Some(ExternalEvent::HostStop)
        );
        assert_eq!(
            inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap(),
            None
        );
    }

    #[test]
    fn event_observers_publish_without_host_threads_and_cancel_atomically() {
        let inbox = ExternalEventInbox::bounded(4, 0).unwrap();
        let token = WakeToken {
            thread: GuestThreadId::new(7),
            generation: WakeGeneration::new(3),
        };
        let (writable, readable) = crate::EventObject::create_pair();
        inbox
            .sender()
            .watch_readable_event(readable, None, token, ExternalEventSource::Device)
            .unwrap();
        writable.signal();
        assert_eq!(
            inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap()
                .map(|event| event.event),
            Some(ExternalEvent::Wake {
                source: ExternalEventSource::Device,
                token,
            })
        );

        let cancelled = WakeToken {
            thread: GuestThreadId::new(8),
            generation: WakeGeneration::new(4),
        };
        let (writable, readable) = crate::EventObject::create_pair();
        let sender = inbox.sender();
        sender
            .watch_readable_event(readable, None, cancelled, ExternalEventSource::Ipc)
            .unwrap();
        sender.cancel_watchers(cancelled);
        writable.signal();
        assert_eq!(
            inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap(),
            None
        );
    }

    #[test]
    fn bounded_receive_distinguishes_a_host_deadline_from_an_external_event() {
        let inbox = ExternalEventInbox::bounded(1, 0).unwrap();
        assert_eq!(
            inbox
                .recv_timeout(Duration::ZERO)
                .map(|event| external(&inbox, event))
                .unwrap(),
            None
        );

        inbox.sender().submit(ExternalEvent::HostStop).unwrap();
        assert_eq!(
            inbox
                .recv_timeout(Duration::from_secs(1))
                .map(|event| external(&inbox, event))
                .unwrap()
                .map(|event| event.event),
            Some(ExternalEvent::HostStop)
        );
    }
    #[test]
    fn concurrent_device_producers_are_published_in_one_monotonic_order() {
        const PRODUCERS: usize = 24;
        let inbox = ExternalEventInbox::bounded(PRODUCERS, 0).unwrap();
        let sender = inbox.sender();
        let start = Arc::new(Barrier::new(PRODUCERS + 1));
        let workers: Vec<_> = (0..PRODUCERS)
            .map(|index| {
                let sender = sender.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    let source = match index % 3 {
                        0 => ExternalEventSource::GpuCompletion,
                        1 => ExternalEventSource::Display,
                        _ => ExternalEventSource::Input,
                    };
                    let token = WakeToken {
                        thread: GuestThreadId::new(index as u64 + 1),
                        generation: WakeGeneration::new(1),
                    };
                    match source {
                        ExternalEventSource::GpuCompletion => sender.gpu_completion(token),
                        ExternalEventSource::Display => sender.display_event(token),
                        ExternalEventSource::Input => sender.input_event(token),
                        _ => unreachable!(),
                    }
                    .unwrap();
                })
            })
            .collect();
        start.wait();
        for worker in workers {
            worker.join().unwrap();
        }

        let mut previous = 0;
        for _ in 0..PRODUCERS {
            let event = inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap()
                .unwrap();
            assert!(event.sequence.get() > previous);
            previous = event.sequence.get();
        }
        assert_eq!(previous, PRODUCERS as u64);
    }

    #[test]
    fn typed_event_object_overrides_the_generic_wait_fallback_source() {
        let inbox = ExternalEventInbox::bounded(1, 0).unwrap();
        let token = WakeToken {
            thread: GuestThreadId::new(1),
            generation: WakeGeneration::new(1),
        };
        let (writable, readable) =
            EventObject::create_pair_with_source(ExternalEventSource::Display);
        inbox
            .sender()
            .watch_readable_event(readable, None, token, ExternalEventSource::Device)
            .unwrap();
        writable.signal();
        assert_eq!(
            inbox
                .try_recv()
                .map(|event| external(&inbox, event))
                .unwrap()
                .unwrap()
                .event,
            ExternalEvent::Wake {
                source: ExternalEventSource::Display,
                token,
            }
        );
    }
}
