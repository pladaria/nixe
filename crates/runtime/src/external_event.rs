use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
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
    sender: SyncSender<SequencedExternalEvent>,
    next_sequence: u64,
}

impl EventPublication {
    fn publish(&mut self, event: ExternalEvent) -> Result<(), ExternalEventSendError> {
        let sequence = self.next_sequence;
        let next = sequence
            .checked_add(1)
            .ok_or(ExternalEventSendError::SequenceExhausted)?;
        self.sender
            .try_send(SequencedExternalEvent {
                sequence: ExternalEventSequence(sequence),
                event,
            })
            .map_err(|error| match error {
                TrySendError::Full(_) => ExternalEventSendError::Full,
                TrySendError::Disconnected(_) => ExternalEventSendError::Closed,
            })?;
        self.next_sequence = next;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ExternalEventSender {
    publication: Arc<Mutex<EventPublication>>,
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
    pub fn submit(&self, event: ExternalEvent) -> Result<(), ExternalEventSendError> {
        self.publication
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .publish(event)
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
            if let Err(error) = publication
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .publish(ExternalEvent::Wake { source, token })
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

#[derive(Debug)]
pub struct ExternalEventInbox {
    sender: ExternalEventSender,
    receiver: Receiver<SequencedExternalEvent>,
}

impl ExternalEventInbox {
    pub fn bounded(capacity: usize) -> Result<Self, ExternalEventSendError> {
        if capacity == 0 {
            return Err(ExternalEventSendError::ZeroCapacity);
        }
        let (sender, receiver) = sync_channel(capacity);
        Ok(Self {
            sender: ExternalEventSender {
                publication: Arc::new(Mutex::new(EventPublication {
                    sender,
                    next_sequence: 1,
                })),
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

    pub fn try_recv_sequenced(
        &self,
    ) -> Result<Option<SequencedExternalEvent>, ExternalEventSendError> {
        if self.sender.overflowed.swap(false, Ordering::AcqRel) {
            return Err(ExternalEventSendError::Full);
        }
        match self.receiver.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ExternalEventSendError::Closed),
        }
    }

    pub(crate) fn recv_sequenced_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<SequencedExternalEvent>, ExternalEventSendError> {
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
    use nixe_scheduler::{GuestThreadId, WakeGeneration};

    #[test]
    fn bounded_inbox_reports_backpressure_without_blocking() {
        let inbox = ExternalEventInbox::bounded(1).unwrap();
        let sender = inbox.sender();
        sender.submit(ExternalEvent::HostStop).unwrap();
        assert_eq!(
            sender.submit(ExternalEvent::HostStop),
            Err(ExternalEventSendError::Full)
        );
        assert_eq!(
            inbox.try_recv_sequenced().unwrap().map(|event| event.event),
            Some(ExternalEvent::HostStop)
        );
        assert_eq!(inbox.try_recv_sequenced().unwrap(), None);
    }

    #[test]
    fn event_observers_publish_without_host_threads_and_cancel_atomically() {
        let inbox = ExternalEventInbox::bounded(4).unwrap();
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
            inbox.try_recv_sequenced().unwrap().map(|event| event.event),
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
        assert_eq!(inbox.try_recv_sequenced().unwrap(), None);
    }

    #[test]
    fn bounded_receive_distinguishes_a_host_deadline_from_an_external_event() {
        let inbox = ExternalEventInbox::bounded(1).unwrap();
        assert_eq!(inbox.recv_sequenced_timeout(Duration::ZERO).unwrap(), None);

        inbox.sender().submit(ExternalEvent::HostStop).unwrap();
        assert_eq!(
            inbox
                .recv_sequenced_timeout(Duration::from_secs(1))
                .unwrap()
                .map(|event| event.event),
            Some(ExternalEvent::HostStop)
        );
    }
}
