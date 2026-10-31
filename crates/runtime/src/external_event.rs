use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalEventSource {
    Timer,
    Input,
    Worker,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalEvent {
    HostStop,
    /// Suspends/resumes host emulation without changing guest thread activity.
    HostPause(bool),
    /// Wakes the host supervisor without making a guest thread runnable.
    HostService(ExternalEventSource),
    /// The last reference to a canonical thread queued its actual finalization.
    ThreadRetirement,
}

#[derive(Debug)]
pub(crate) struct RetiredThread {
    pub process: nixe_scheduler::ProcessId,
    pub thread: nixe_scheduler::GuestThreadId,
    pub owner: std::sync::Weak<crate::RunnableProcess>,
}

#[derive(Debug)]
struct EventPublication {
    sender: SyncSender<ExternalEvent>,
    overflowed: AtomicBool,
    retired_threads: Mutex<VecDeque<RetiredThread>>,
    retirement_pending: AtomicBool,
}

#[derive(Clone, Debug)]
pub struct ExternalEventSender {
    publication: Arc<EventPublication>,
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
            let _ = self.submit(ExternalEvent::ThreadRetirement);
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
    /// the supervisor instead of silently losing a host service notification.
    pub fn notify_host_service(&self, source: ExternalEventSource) {
        if let Err(ExternalEventSendError::Full) = self.submit(ExternalEvent::HostService(source)) {
            self.publication.overflowed.store(true, Ordering::Release);
        }
    }

    pub fn submit(&self, event: ExternalEvent) -> Result<(), ExternalEventSendError> {
        self.publication
            .sender
            .try_send(event)
            .map_err(|error| match error {
                TrySendError::Full(_) => ExternalEventSendError::Full,
                TrySendError::Disconnected(_) => ExternalEventSendError::Closed,
            })
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

pub(crate) struct ExternalEventInbox {
    pub(crate) sender: ExternalEventSender,
    receiver: Receiver<ExternalEvent>,
}

impl ExternalEventInbox {
    pub fn bounded(capacity: usize) -> Result<Self, ExternalEventSendError> {
        if capacity == 0 {
            return Err(ExternalEventSendError::ZeroCapacity);
        }
        let (sender, receiver) = sync_channel(capacity);
        Ok(Self {
            sender: ExternalEventSender {
                publication: Arc::new(EventPublication {
                    sender,
                    overflowed: AtomicBool::new(false),
                    retired_threads: Mutex::default(),
                    retirement_pending: AtomicBool::new(false),
                }),
            },
            receiver,
        })
    }

    #[must_use]
    pub fn sender(&self) -> ExternalEventSender {
        self.sender.clone()
    }

    // Supervision reports lost notifications; teardown must still drain accepted work.
    pub(crate) fn check_error(&self) -> Result<(), ExternalEventSendError> {
        if self
            .sender
            .publication
            .overflowed
            .swap(false, Ordering::AcqRel)
        {
            Err(ExternalEventSendError::Full)
        } else {
            Ok(())
        }
    }

    pub(crate) fn try_recv(&self) -> Result<Option<ExternalEvent>, ExternalEventSendError> {
        self.check_error()?;
        match self.receiver.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ExternalEventSendError::Closed),
        }
    }

    pub(crate) fn recv(&self) -> Result<ExternalEvent, ExternalEventSendError> {
        self.receiver
            .recv()
            .map_err(|_| ExternalEventSendError::Closed)
    }

    pub(crate) fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<ExternalEvent>, ExternalEventSendError> {
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
            let inbox = ExternalEventInbox::bounded(1).unwrap();
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
            let Some(event) = inbox.try_recv().unwrap() else {
                panic!("missing coalesced ingress");
            };
            assert_eq!(
                event,
                if full {
                    ExternalEvent::HostService(ExternalEventSource::Input)
                } else {
                    ExternalEvent::ThreadRetirement
                }
            );
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
            assert!(matches!(
                inbox.try_recv().unwrap(),
                Some(ExternalEvent::ThreadRetirement)
            ));
            assert_eq!(
                sender
                    .take_retired_thread()
                    .map(|item| (item.process, item.thread)),
                Some((ProcessId::new(7), GuestThreadId::new(2049)))
            );
        }
    }

    use super::*;

    #[test]
    fn host_service_overflow_is_reported_to_the_consumer() {
        let inbox = ExternalEventInbox::bounded(1).unwrap();
        let sender = inbox.sender();
        sender.notify_host_service(ExternalEventSource::Input);
        sender.notify_host_service(ExternalEventSource::Input);
        assert_eq!(inbox.try_recv(), Err(ExternalEventSendError::Full));
        assert_eq!(
            inbox.try_recv().unwrap().unwrap(),
            ExternalEvent::HostService(ExternalEventSource::Input)
        );
    }

    #[test]
    fn bounded_inbox_reports_backpressure_without_blocking() {
        let inbox = ExternalEventInbox::bounded(1).unwrap();
        let sender = inbox.sender();
        sender.submit(ExternalEvent::HostStop).unwrap();
        assert_eq!(
            sender.submit(ExternalEvent::HostStop),
            Err(ExternalEventSendError::Full)
        );
        assert_eq!(inbox.try_recv().unwrap(), Some(ExternalEvent::HostStop));
        assert_eq!(inbox.try_recv().unwrap(), None);
    }

    #[test]
    fn bounded_receive_distinguishes_a_host_deadline_from_an_external_event() {
        let inbox = ExternalEventInbox::bounded(1).unwrap();
        assert_eq!(inbox.recv_timeout(Duration::ZERO).unwrap(), None);

        inbox.sender().submit(ExternalEvent::HostStop).unwrap();
        assert_eq!(
            inbox.recv_timeout(Duration::from_secs(1)).unwrap(),
            Some(ExternalEvent::HostStop)
        );
    }
}
