use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    /// The last reference to a canonical thread queued its actual finalization.
    ThreadRetirement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequencedExternalEvent {
    pub sequence: ExternalEventSequence,
    pub event: ExternalEvent,
}

#[derive(Debug)]
pub(crate) struct RetiredThread {
    pub process: nixe_scheduler::ProcessId,
    pub thread: nixe_scheduler::GuestThreadId,
    pub owner: std::sync::Weak<crate::RunnableProcess>,
}

#[derive(Debug)]
struct EventPublication {
    sender: SyncSender<CoordinatorEvent>,
    next_sequence: Mutex<u64>,
    pending: AtomicUsize,
    capacity: usize,
    retired_threads: Mutex<VecDeque<RetiredThread>>,
    retirement_pending: AtomicBool,
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
    overflowed: Arc<AtomicBool>,
}

impl ExternalEventSender {
    // Destruction queues actual finalization, not a guessed reference-count test.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_auto_object.h#L151-L163
    pub(crate) fn retire_thread(
        &self,
        process: nixe_scheduler::ProcessId,
        thread: nixe_scheduler::GuestThreadId,
        owner: std::sync::Weak<crate::RunnableProcess>,
    ) {
        let first = {
            let mut retired = self.publication.retired_threads.lock().unwrap();
            let first = retired.is_empty();
            retired.push_back(RetiredThread {
                process,
                thread,
                owner,
            });
            self.publication
                .retirement_pending
                .store(true, Ordering::Release);
            first
        };
        if first {
            match self.submit(ExternalEvent::ThreadRetirement) {
                Ok(()) | Err(ExternalEventSendError::Full | ExternalEventSendError::Closed) => {}
                Err(_) => self.overflowed.store(true, Ordering::Release),
            }
            // Full ingress already guarantees a wake; the actual work remains
            // queued and is consumed on that ingress, rather than being lost.
        }
    }

    pub(crate) fn take_retired_thread(&self) -> Option<RetiredThread> {
        if !self.publication.retirement_pending.load(Ordering::Acquire) {
            return None;
        }
        let mut retired = self.publication.retired_threads.lock().unwrap();
        let next = retired.pop_front();
        if retired.is_empty() {
            self.publication
                .retirement_pending
                .store(false, Ordering::Release);
        }
        next
    }

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
}

impl std::task::Wake for ExternalEventSender {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.notify_host_service(ExternalEventSource::Timer);
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
    pub(crate) sender: ExternalEventSender,
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
                    retired_threads: Mutex::default(),
                    retirement_pending: AtomicBool::new(false),
                }),
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

    pub(crate) fn recv(&self) -> Result<CoordinatorEvent, ExternalEventSendError> {
        self.receiver
            .recv()
            .map_err(|_| ExternalEventSendError::Closed)
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
    #[test]
    fn canonical_destruction_coalesces_and_preserves_every_finalization_when_ingress_is_full() {
        use super::*;
        use nixe_scheduler::{GuestThreadId, ProcessId};
        for full in [false, true] {
            let inbox = ExternalEventInbox::bounded(1, 1).unwrap();
            let sender = inbox.sender();
            if full {
                sender
                    .submit(ExternalEvent::HostService(ExternalEventSource::Input))
                    .unwrap();
            }
            std::thread::scope(|scope| {
                for worker in 0..4 {
                    let sender = sender.clone();
                    scope.spawn(move || {
                        for i in 0..512 {
                            sender.retire_thread(
                                ProcessId::new(7),
                                GuestThreadId::new(worker * 512 + i + 1),
                                std::sync::Weak::new(),
                            );
                        }
                    });
                }
            });
            let Some(CoordinatorEvent::External(event)) = inbox.try_recv().unwrap() else {
                panic!("missing coalesced ingress");
            };
            assert_eq!(
                event.event,
                if full {
                    ExternalEvent::HostService(ExternalEventSource::Input)
                } else {
                    ExternalEvent::ThreadRetirement
                }
            );
            inbox.consume_external();
            assert!(inbox.try_recv().unwrap().is_none());
            let mut retired = std::collections::BTreeSet::new();
            while let Some(item) = sender.take_retired_thread() {
                assert_eq!(item.process, ProcessId::new(7));
                assert!(retired.insert(item.thread));
            }
            assert_eq!(retired.len(), 2048);
            sender.retire_thread(
                ProcessId::new(7),
                GuestThreadId::new(2049),
                std::sync::Weak::new(),
            );
            assert!(
                matches!(inbox.try_recv().unwrap(), Some(CoordinatorEvent::External(event)) if event.event == ExternalEvent::ThreadRetirement)
            );
            assert_eq!(
                sender
                    .take_retired_thread()
                    .map(|item| (item.process, item.thread)),
                Some((ProcessId::new(7), GuestThreadId::new(2049)))
            );
        }
    }

    use super::*;

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
        let inbox = ExternalEventInbox::bounded(512, 0).unwrap();
        std::thread::scope(|scope| {
            for source in [
                ExternalEventSource::GpuCompletion,
                ExternalEventSource::Display,
                ExternalEventSource::Input,
            ] {
                let sender = inbox.sender();
                scope.spawn(move || {
                    for _ in 0..100 {
                        sender.submit(ExternalEvent::HostService(source)).unwrap();
                    }
                });
            }
        });
        for expected in 1..=300 {
            let event = external(&inbox, inbox.try_recv().unwrap()).unwrap();
            assert_eq!(event.sequence.get(), expected);
        }
        assert!(inbox.try_recv().unwrap().is_none());
    }
}
