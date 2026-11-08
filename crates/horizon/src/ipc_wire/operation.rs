//! Ownership and wakeups for suspended built-in service calls.
use super::IpcWireError;
use nixe_runtime::{EventFuture, ReadableEventObject};
use std::{
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
const SERVER: u8 = 8;
const REPLIED: u8 = 16;
const FAILURE_REPORTED: u8 = 32;

// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1445-L1525
/// Service effects finish before relinquishing their turn. A suspended response
/// write has no executing service effect and can be dropped on cancellation.
pub(crate) struct CallLifetime {
    state: AtomicU8,
    client: nixe_runtime::KernelRequest,
}
impl CallLifetime {
    pub fn for_client(context: &nixe_runtime::ExceptionDispatchContext<'_>) -> Self {
        Self {
            state: AtomicU8::new(0),
            client: context.retain_request(),
        }
    }
    pub fn client(&self) -> &nixe_runtime::KernelRequest {
        &self.client
    }
    pub fn report_failure(&self, error: &IpcWireError) {
        let client = &self.client;
        if (self.state.load(Ordering::Acquire) & CANCELLED != 0 || client.is_retired())
            && super::HorizonIpcFault::from_wire(error.clone())
                .guest_result()
                .is_none()
            && self.state.fetch_or(FAILURE_REPORTED, Ordering::AcqRel) & FAILURE_REPORTED == 0
        {
            client.report_failure(format!(
                "accepted IPC service failed: {}",
                super::HorizonIpcFault::from_wire(error.clone())
            ));
        }
    }
    /// Returns whether an already executing effect must still be drained.
    pub fn cancel(&self) -> bool {
        let state = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                Some(if state & REPLIED == 0 {
                    state | CANCELLED
                } else {
                    state
                })
            })
            .unwrap();
        state & SERVER != 0 || state & (STARTED | OUTPUT) == STARTED
    }
    /// The caller retains the server result until its terminal error is consumed.
    pub fn delegate(&self) -> Result<(), IpcWireError> {
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & CANCELLED == 0).then_some(state | SERVER)
            })
            .map(|_| ())
            .map_err(|_| IpcWireError::Cancelled)
    }
    pub fn consume_reply(&self) {
        self.state.fetch_and(!SERVER, Ordering::Release);
    }
    pub fn must_drain_effect(&self) -> bool {
        self.state.load(Ordering::Acquire) & (STARTED | OUTPUT) == STARTED
    }
    /// Reply publication and cancellation have one atomic winner.
    pub fn commit_reply(&self) -> Result<(), IpcWireError> {
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & CANCELLED == 0).then_some(state | REPLIED)
            })
            .map(|_| ())
            .map_err(|_| IpcWireError::Cancelled)
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
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & (STARTED | CANCELLED) == 0).then_some(state | STARTED)
            })
            .map(|_| ())
            .map_err(|_| IpcWireError::Cancelled)
    }
    pub fn begin_output(&self) -> Result<(), IpcWireError> {
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & (STARTED | OUTPUT | CANCELLED) == STARTED).then_some(state | OUTPUT)
            })
            .map(|_| ())
            .map_err(|_| IpcWireError::Cancelled)
    }
    pub fn end_output(&self) {
        self.state.fetch_and(!OUTPUT, Ordering::Release);
    }
}

/// Serializes publication into one shared-memory object across suspended writes.
#[derive(Debug, Default)]
pub(crate) struct Publication(Mutex<Publications>);
#[derive(Debug, Default)]
struct Publications {
    next: u64,
    turns: std::collections::BTreeMap<u64, Option<Waker>>,
}
impl Publication {
    pub fn enter(self: &Arc<Self>) -> PublicationTurn {
        let mut state = self.0.lock().unwrap();
        state.next = state
            .next
            .checked_add(1)
            .expect("publication identity exhausted");
        let id = state.next;
        state.turns.insert(id, None);
        PublicationTurn {
            owner: self.clone(),
            id,
        }
    }
}
pub(crate) struct PublicationTurn {
    owner: Arc<Publication>,
    id: u64,
}
impl Future for &PublicationTurn {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let mut state = self.owner.0.lock().unwrap();
        if state
            .turns
            .first_key_value()
            .is_some_and(|(id, _)| *id == self.id)
        {
            return Poll::Ready(());
        }
        *state
            .turns
            .get_mut(&self.id)
            .expect("registered publication") = Some(context.waker().clone());
        Poll::Pending
    }
}
impl Drop for PublicationTurn {
    fn drop(&mut self) {
        let wake = {
            let mut state = self.owner.0.lock().unwrap();
            let head = state
                .turns
                .first_key_value()
                .is_some_and(|(id, _)| *id == self.id);
            state.turns.remove(&self.id);
            if head {
                state
                    .turns
                    .first_entry()
                    .and_then(|mut entry| entry.get_mut().take())
            } else {
                None
            }
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}

pub(crate) struct ServiceEvent<'a> {
    lifetime: &'a CallLifetime,
    event: EventFuture,
    deadline: Option<nixe_scheduler::Deadline>,
}
impl<'a> ServiceEvent<'a> {
    pub fn new(
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        event: ReadableEventObject,
        timeout: Option<Duration>,
        lifetime: &'a CallLifetime,
    ) -> Self {
        Self {
            lifetime,
            event: event.wait_async(),
            deadline: timeout.map(|timeout| {
                kernel.deadline(
                    kernel
                        .clock()
                        .scheduler_time_ns()
                        .saturating_add(u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX)),
                )
            }),
        }
    }
}
impl Future for ServiceEvent<'_> {
    type Output = Result<(), IpcWireError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        // Client termination cancels passive waits without waiting for the device.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1430-L1443
        this.lifetime.check()?;
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
