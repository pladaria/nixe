use super::*;

/// Endpoint role of one process-local session pair.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SessionEndpoint {
    Server,
    Client,
}

/// Maximum number of requests retained by one session before back-pressure.
pub const MAX_SESSION_REQUESTS: usize = 0x40;

// Session request/reply and peer-close behavior follows the public
// implementation in Atmosphère's kernel:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_client_session.cpp
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_server_session.cpp

/// Owned message transferred between normal or light Horizon session endpoints.
#[derive(Debug, Eq, PartialEq)]
pub enum SessionMessage {
    /// Captured emitted bytes, excluding unused source-buffer capacity.
    Buffer(Vec<u8>),
    /// The same captured extent with accepted special-header objects.
    TransportedBuffer {
        bytes: Vec<u8>,
        copy_handles: Vec<Option<HandleObject>>,
        move_handles: Vec<Option<HandleObject>>,
    },
    Light([u32; 7]),
}

impl SessionMessage {
    #[must_use]
    pub const fn is_light(&self) -> bool {
        matches!(self, Self::Light(_))
    }
}

#[derive(Debug)]
struct QueuedRequest {
    id: u64,
    owner: SessionRequestOwner,
    message: SessionMessage,
}

// Once delivered, the message and handles belong to the server. Only the
// reply/cancellation identity remains, as with KSessionRequest.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_session_request.h
#[derive(Debug)]
struct ActiveRequest {
    id: u64,
    owner: SessionRequestOwner,
    cancelled: bool,
    phase: RequestPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestPhase {
    Receiving,
    AwaitingReply,
    Replying,
}

/// Process/thread identity of one synchronous session participant.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SessionRequestOwner {
    pub process_id: u64,
    pub thread_id: u64,
}

#[derive(Debug)]
struct PendingResponse {
    id: u64,
    wake: Option<std::task::Waker>,
}

#[derive(Debug)]
struct SessionState {
    server_open: bool,
    client_open: bool,
    next_request_id: u64,
    queued: VecDeque<QueuedRequest>,
    current: Option<ActiveRequest>,
    light_server: Option<SessionRequestOwner>,
    pending_by_owner: BTreeMap<SessionRequestOwner, PendingResponse>,
    responses: BTreeMap<u64, Result<SessionMessage, SessionError>>,
    owning_port: Option<Weak<Mutex<PortState>>>,
}

impl SessionState {
    fn take_request(&mut self, phase: RequestPhase) -> Result<QueuedRequest, SessionError> {
        if self.current.is_some() {
            return Err(SessionError::ReplyPending);
        }
        let Some(request) = self.queued.pop_front() else {
            return Err(if self.client_open {
                SessionError::NoRequest
            } else {
                SessionError::PeerClosed
            });
        };
        self.current = Some(ActiveRequest {
            id: request.id,
            owner: request.owner,
            cancelled: false,
            phase,
        });
        Ok(request)
    }
}

#[derive(Debug)]
struct SessionIdentity {
    state: Mutex<SessionState>,
    wake_event: EventObject,
}

impl SessionIdentity {
    fn publish_server(&self, state: &SessionState) -> EventNotifications {
        self.wake_event.publish_readiness(
            !state.client_open || (state.current.is_none() && !state.queued.is_empty()),
        )
    }

    fn finish_receive(&self, id: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(request) = state.current.as_mut().filter(|request| request.id == id) else {
            return;
        };
        debug_assert_eq!(request.phase, RequestPhase::Receiving);
        request.phase = RequestPhase::AwaitingReply;
    }

    fn finish_request(
        &self,
        id: u64,
        response: Result<SessionMessage, SessionError>,
    ) -> Result<(), SessionError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state
            .current
            .as_ref()
            .is_none_or(|request| request.id != id)
        {
            return Err(SessionError::PeerClosed);
        }
        let request = state.current.take().expect("accepted request");
        let cancelled = request.cancelled || !state.client_open;
        if !cancelled {
            state.responses.insert(request.id, response);
        }
        let wake = state
            .pending_by_owner
            .get_mut(&request.owner)
            .and_then(|pending| pending.wake.take());
        let notifications = self.publish_server(&state);
        drop(state);
        notifications.notify();
        if let Some(wake) = wake {
            wake.wake();
        }
        if cancelled {
            Err(SessionError::PeerClosed)
        } else {
            Ok(())
        }
    }
}

/// Retains the accepted endpoint and request generation across a kernel phase.
#[derive(Debug)]
pub struct SessionRequest {
    lease: Arc<SessionEndpointLease>,
    id: u64,
    pending: bool,
}
impl SessionRequest {
    fn complete_receive(mut self) {
        self.pending = false;
        self.lease.identity.finish_receive(self.id);
    }
    pub fn reply(mut self, message: SessionMessage) -> Result<(), SessionError> {
        if message.is_light() {
            return Err(SessionError::MessageKindMismatch);
        }
        self.pending = false;
        self.lease.identity.finish_request(self.id, Ok(message))
    }
    pub fn fail(mut self, error: SessionError) -> Result<(), SessionError> {
        self.pending = false;
        self.lease.identity.finish_request(self.id, Err(error))
    }
}
impl Drop for SessionRequest {
    fn drop(&mut self) {
        if self.pending {
            let _ = self
                .lease
                .identity
                .finish_request(self.id, Err(SessionError::PeerClosed));
        }
    }
}

/// Owns an incoming normal request until its bytes and handles are delivered.
/// A failed or cancelled receive releases the active request and completes its
/// client; a reply cannot overtake the receive's publication.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1158-L1218
#[derive(Debug)]
pub struct SessionReceive {
    message: Option<SessionMessage>,
    request: SessionRequest,
}
impl SessionReceive {
    pub fn take_message(&mut self) -> SessionMessage {
        self.message.take().expect("one incoming message")
    }
    pub fn complete(self) {
        drop(self.message);
        self.request.complete_receive();
    }
    pub fn fail(self, error: SessionError) {
        drop(self.message);
        // A client cancelled during receive needs no further completion.
        let _ = self.request.fail(error);
    }
}

#[derive(Debug)]
struct SessionEndpointLease {
    identity: Arc<SessionIdentity>,
    endpoint: SessionEndpoint,
}

impl Drop for SessionEndpointLease {
    fn drop(&mut self) {
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            SessionEndpoint::Server => state.server_open = false,
            SessionEndpoint::Client => state.client_open = false,
        }
        // Completed results belong to the waiting client. Server closure only
        // cancels requests which have not completed, as CleanupRequests does.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1373-L1443
        let responses = if self.endpoint == SessionEndpoint::Client {
            Some((
                std::mem::take(&mut state.pending_by_owner),
                std::mem::take(&mut state.responses),
            ))
        } else {
            None
        };
        let requests = (
            std::mem::take(&mut state.queued),
            state.current.take(),
            responses,
        );
        state.light_server = None;
        let wakes: Vec<_> = state
            .pending_by_owner
            .values_mut()
            .filter_map(|pending| pending.wake.take())
            .collect();
        let notifications = self.identity.publish_server(&state);
        drop(state);
        // Transported handles may finalize shared/transfer memory or another
        // session. Finalizers never run under the session state lock.
        drop(requests);
        notifications.notify();
        for wake in wakes {
            wake.wake();
        }
    }
}

/// Deterministic session transport failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SessionError {
    WrongEndpoint,
    PeerClosed,
    QueueFull,
    NoRequest,
    ReplyPending,
    MessageKindMismatch,
    TransferFailed(u32),
}

/// One endpoint of a paired, bounded synchronous session transport.
#[derive(Clone, Debug)]
pub struct SessionObject {
    identity: Arc<SessionIdentity>,
    endpoint: SessionEndpoint,
    _lease: Arc<SessionEndpointLease>,
    is_light: bool,
}

impl SessionObject {
    #[must_use]
    pub fn create_pair(kernel: &Arc<nixe_scheduler::SchedulerState>) -> (Self, Self) {
        Self::create_pair_with_kind(kernel, false, None)
    }

    #[must_use]
    pub fn create_light_pair(kernel: &Arc<nixe_scheduler::SchedulerState>) -> (Self, Self) {
        Self::create_pair_with_kind(kernel, true, None)
    }

    pub(super) fn create_pair_with_kind(
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        is_light: bool,
        owning_port: Option<Weak<Mutex<PortState>>>,
    ) -> (Self, Self) {
        let identity = Arc::new(SessionIdentity {
            state: Mutex::new(SessionState {
                server_open: true,
                client_open: true,
                next_request_id: 1,
                queued: VecDeque::new(),
                current: None,
                light_server: None,
                pending_by_owner: BTreeMap::new(),
                responses: BTreeMap::new(),
                owning_port,
            }),
            wake_event: EventObject::new(kernel),
        });
        let server_lease = Arc::new(SessionEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: SessionEndpoint::Server,
        });
        let client_lease = Arc::new(SessionEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: SessionEndpoint::Client,
        });
        (
            Self {
                identity: identity.clone(),
                endpoint: SessionEndpoint::Server,
                _lease: server_lease,
                is_light,
            },
            Self {
                identity,
                endpoint: SessionEndpoint::Client,
                _lease: client_lease,
                is_light,
            },
        )
    }

    #[must_use]
    pub const fn endpoint(&self) -> SessionEndpoint {
        self.endpoint
    }

    #[must_use]
    pub const fn is_light(&self) -> bool {
        self.is_light
    }

    #[must_use]
    pub fn readable_event(&self) -> ReadableEventObject {
        self.identity.wake_event.readable()
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        let state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            // A queued request can be received only after the active reply.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1361
            SessionEndpoint::Server => {
                !state.client_open || (state.current.is_none() && !state.queued.is_empty())
            }
            SessionEndpoint::Client => !state.responses.is_empty() || !state.server_open,
        }
    }

    pub fn request(
        &self,
        owner: SessionRequestOwner,
        message: SessionMessage,
    ) -> Result<(), SessionError> {
        if self.endpoint != SessionEndpoint::Client {
            return Err(SessionError::WrongEndpoint);
        }
        if self.is_light != message.is_light() {
            return Err(SessionError::MessageKindMismatch);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.server_open {
            return Err(SessionError::PeerClosed);
        }
        // A blocked caller owns one accepted request. Sending never polls or
        // consumes that request's reply; completion belongs to its continuation.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_client_session.cpp
        if state.pending_by_owner.contains_key(&owner) {
            return Err(SessionError::ReplyPending);
        }
        if state.pending_by_owner.len() >= MAX_SESSION_REQUESTS {
            return Err(SessionError::QueueFull);
        }
        let id = state.next_request_id;
        state.next_request_id = state.next_request_id.saturating_add(1);
        state
            .pending_by_owner
            .insert(owner, PendingResponse { id, wake: None });
        state.queued.push_back(QueuedRequest { id, owner, message });
        let notifications = self.identity.publish_server(&state);
        drop(state);
        notifications.notify();
        Ok(())
    }

    pub fn response(&self, owner: SessionRequestOwner) -> SessionResponse {
        SessionResponse {
            session: self.clone(),
            owner,
        }
    }

    fn poll_response(
        &self,
        owner: SessionRequestOwner,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<SessionMessage, SessionError>> {
        use std::task::Poll;
        if self.endpoint != SessionEndpoint::Client {
            return Poll::Ready(Err(SessionError::WrongEndpoint));
        }
        let mut state = self.identity.state.lock().unwrap();
        let Some(pending) = state.pending_by_owner.get(&owner) else {
            return Poll::Ready(Err(SessionError::NoRequest));
        };
        let id = pending.id;
        if let Some(response) = state.responses.remove(&id) {
            let retired = state.pending_by_owner.remove(&owner);
            drop(state);
            drop(retired);
            return Poll::Ready(response);
        }
        if !state.server_open {
            let retired = state.pending_by_owner.remove(&owner);
            drop(state);
            drop(retired);
            return Poll::Ready(Err(SessionError::PeerClosed));
        }
        let pending = state.pending_by_owner.get_mut(&owner).unwrap();
        let old = if pending
            .wake
            .as_ref()
            .is_none_or(|wake| !wake.will_wake(context.waker()))
        {
            pending.wake.replace(context.waker().clone())
        } else {
            None
        };
        drop(state);
        drop(old);
        Poll::Pending
    }

    /// Cancels one client's accepted request. A server-owned current request
    /// remains current until its reply acknowledges cancellation; it cannot be
    /// confused with the next request from another client thread.
    pub fn cancel_request(&self, owner: SessionRequestOwner) -> Result<bool, SessionError> {
        if self.endpoint != SessionEndpoint::Client {
            return Err(SessionError::WrongEndpoint);
        }
        let (queued, response) = {
            let mut state = self
                .identity
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(pending) = state.pending_by_owner.remove(&owner) else {
                return Ok(false);
            };
            let id = pending.id;
            let index = state.queued.iter().position(|request| request.id == id);
            let queued = index.and_then(|index| state.queued.remove(index));
            if let Some(current) = state.current.as_mut().filter(|request| request.id == id) {
                current.cancelled = true;
            }
            let response = state.responses.remove(&id);
            let notifications = self.identity.publish_server(&state);
            (queued, (response, pending, notifications))
        };
        drop(queued);
        let (response, pending, notifications) = response;
        drop(response);
        drop(pending);
        notifications.notify();
        Ok(true)
    }

    pub fn receive(&self) -> Result<SessionReceive, SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        if self.is_light {
            return Err(SessionError::MessageKindMismatch);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let request = state.take_request(RequestPhase::Receiving)?;
        let notifications = self.identity.publish_server(&state);
        drop(state);
        notifications.notify();
        Ok(SessionReceive {
            request: SessionRequest {
                lease: Arc::clone(&self._lease),
                id: request.id,
                pending: true,
            },
            message: Some(request.message),
        })
    }

    /// Light sessions have one receiving server, which also owns the reply.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_light_server_session.cpp#L123-L207
    pub fn receive_light(
        &self,
        server: SessionRequestOwner,
    ) -> Result<SessionMessage, SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        if !self.is_light {
            return Err(SessionError::MessageKindMismatch);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.light_server.is_some_and(|owner| owner != server) {
            return Err(SessionError::ReplyPending);
        }
        let result = state.take_request(RequestPhase::AwaitingReply);
        if result.is_ok() || matches!(result, Err(SessionError::NoRequest)) {
            state.light_server = Some(server);
        }
        let notifications = self.identity.publish_server(&state);
        drop(state);
        notifications.notify();
        result.map(|request| request.message)
    }

    /// Releases only a pending light receive, never an already delivered reply.
    pub fn cancel_light_receive(&self, server: SessionRequestOwner) {
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.current.is_none() && state.light_server == Some(server) {
            state.light_server = None;
        }
    }

    /// Claims the active request before capturing reply bytes or handles.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1225-L1265
    pub fn reply(&self) -> Result<SessionRequest, SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        if self.is_light {
            return Err(SessionError::MessageKindMismatch);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.client_open {
            return Err(SessionError::PeerClosed);
        }
        let request = state.current.as_mut().ok_or(SessionError::NoRequest)?;
        if request.phase != RequestPhase::AwaitingReply {
            return Err(SessionError::ReplyPending);
        }
        request.phase = RequestPhase::Replying;
        let id = request.id;
        drop(state);
        Ok(SessionRequest {
            lease: Arc::clone(&self._lease),
            id,
            pending: true,
        })
    }

    pub fn reply_light(
        &self,
        server: SessionRequestOwner,
        words: [u32; 7],
    ) -> Result<(), SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        if !self.is_light {
            return Err(SessionError::MessageKindMismatch);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.client_open {
            return Err(SessionError::PeerClosed);
        }
        if state.light_server != Some(server) {
            return Err(SessionError::ReplyPending);
        }
        let request = state.current.take().ok_or(SessionError::NoRequest)?;
        state.light_server = None;
        // A terminated light client needs no response; the receiving server
        // still completes its reply and can accept the next request.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_light_server_session.cpp#L141-L152
        if !request.cancelled {
            state
                .responses
                .insert(request.id, Ok(SessionMessage::Light(words)));
        }
        let wake = state
            .pending_by_owner
            .get_mut(&request.owner)
            .and_then(|pending| pending.wake.take());
        let notifications = self.identity.publish_server(&state);
        drop(state);
        notifications.notify();
        if let Some(wake) = wake {
            wake.wake();
        }
        Ok(())
    }
}

pub struct SessionResponse {
    session: SessionObject,
    owner: SessionRequestOwner,
}
impl std::future::Future for SessionResponse {
    type Output = Result<SessionMessage, SessionError>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.session.poll_response(self.owner, context)
    }
}

impl Drop for SessionIdentity {
    fn drop(&mut self) {
        let owning_port = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .owning_port
            .take();
        if let Some(port) = owning_port.and_then(|port| port.upgrade()) {
            let mut port = port
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            port.active_sessions = port.active_sessions.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use std::sync::atomic::Ordering;
    fn owner(thread_id: u64) -> SessionRequestOwner {
        SessionRequestOwner {
            process_id: 1,
            thread_id,
        }
    }

    #[test]
    fn closing_the_server_preserves_completed_replies_and_disconnects_outstanding_requests() {
        for light in [false, true] {
            let (server, client) = if light {
                SessionObject::create_light_pair(&crate::process::tests::test_kernel())
            } else {
                SessionObject::create_pair(&crate::process::tests::test_kernel())
            };
            for id in [1, 2] {
                client
                    .request(
                        owner(id),
                        if light {
                            SessionMessage::Light([id as u32; 7])
                        } else {
                            SessionMessage::Buffer(vec![id as u8])
                        },
                    )
                    .unwrap();
            }
            if light {
                server.receive_light(owner(99)).unwrap();
                server.reply_light(owner(99), [3; 7]).unwrap();
            } else {
                server.receive().unwrap().complete();
                server
                    .reply()
                    .unwrap()
                    .reply(SessionMessage::Buffer(vec![3]))
                    .unwrap();
            }
            drop(server);
            assert_eq!(
                crate::process::tests::poll_session(&client, owner(1)),
                Ok(Some(if light {
                    SessionMessage::Light([3; 7])
                } else {
                    SessionMessage::Buffer(vec![3])
                }))
            );
            assert_eq!(
                crate::process::tests::poll_session(&client, owner(2)),
                Err(SessionError::PeerClosed)
            );
        }
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        client
            .request(owner(1), SessionMessage::Buffer(vec![1]))
            .unwrap();
        server
            .receive()
            .unwrap()
            .fail(SessionError::TransferFailed(0xd201));
        drop(server);
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Err(SessionError::TransferFailed(0xd201))
        );
    }

    #[test]
    fn normal_delivery_and_reply_claims_preserve_one_active_request_and_its_result() {
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        for id in [1, 2] {
            client
                .request(owner(id), SessionMessage::Buffer(vec![id as u8]))
                .unwrap();
        }
        let receive = server.receive().unwrap();
        assert_eq!(server.reply().unwrap_err(), SessionError::ReplyPending);
        assert!(!server.is_signalled());
        receive.complete();
        let reply = server.reply().unwrap();
        assert_eq!(server.reply().unwrap_err(), SessionError::ReplyPending);
        assert_eq!(server.receive().unwrap_err(), SessionError::ReplyPending);
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Ok(None)
        );
        reply.fail(SessionError::TransferFailed(0xd201)).unwrap();
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Err(SessionError::TransferFailed(0xd201))
        );
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Err(SessionError::NoRequest)
        );
        assert!(server.is_signalled());
        server.receive().unwrap().complete();
        drop(server.reply().unwrap());
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(2)),
            Err(SessionError::PeerClosed)
        );
        client
            .request(owner(3), SessionMessage::Buffer(vec![3]))
            .unwrap();
        server.receive().unwrap().complete();
        server
            .reply()
            .unwrap()
            .reply(SessionMessage::Buffer(vec![4]))
            .unwrap();
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(3)),
            Ok(Some(SessionMessage::Buffer(vec![4])))
        );
    }

    #[test]
    fn light_receive_cancellation_preserves_the_active_server_and_next_request() {
        let (server, client) =
            SessionObject::create_light_pair(&crate::process::tests::test_kernel());
        assert_eq!(
            server.receive_light(owner(10)),
            Err(SessionError::NoRequest)
        );
        assert_eq!(
            server.receive_light(owner(11)),
            Err(SessionError::ReplyPending)
        );
        server.cancel_light_receive(owner(11));
        assert_eq!(
            server.receive_light(owner(11)),
            Err(SessionError::ReplyPending)
        );
        server.cancel_light_receive(owner(10));
        client
            .request(owner(1), SessionMessage::Light([1; 7]))
            .unwrap();
        assert_eq!(
            server.receive().unwrap_err(),
            SessionError::MessageKindMismatch
        );
        assert_eq!(
            server.receive_light(owner(11)),
            Ok(SessionMessage::Light([1; 7]))
        );
        server.cancel_light_receive(owner(11));
        assert_eq!(
            server.reply_light(owner(10), [2; 7]),
            Err(SessionError::ReplyPending)
        );
        assert_eq!(
            server.reply().unwrap_err(),
            SessionError::MessageKindMismatch
        );
        assert!(client.cancel_request(owner(1)).unwrap());
        assert_eq!(server.reply_light(owner(11), [2; 7]), Ok(()));
        client
            .request(owner(2), SessionMessage::Light([3; 7]))
            .unwrap();
        assert_eq!(
            server.receive_light(owner(10)),
            Ok(SessionMessage::Light([3; 7]))
        );
        server.reply_light(owner(10), [4; 7]).unwrap();
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(2)),
            Ok(Some(SessionMessage::Light([4; 7])))
        );
        assert_eq!(
            server.receive_light(owner(10)),
            Err(SessionError::NoRequest)
        );
        drop(client);
        assert_eq!(
            server.receive_light(owner(11)),
            Err(SessionError::PeerClosed)
        );
    }

    #[test]
    fn a_busy_server_becomes_ready_only_after_its_current_reply_is_acknowledged() {
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        client
            .request(owner(1), SessionMessage::Buffer(vec![1]))
            .unwrap();
        server.receive().unwrap().complete();
        client
            .request(owner(2), SessionMessage::Buffer(vec![2]))
            .unwrap();
        assert!(!server.is_signalled());
        client.cancel_request(owner(1)).unwrap();
        assert!(!server.is_signalled());
        assert_eq!(
            server
                .reply()
                .unwrap()
                .reply(SessionMessage::Buffer(vec![3])),
            Err(SessionError::PeerClosed)
        );
        assert!(server.is_signalled());
        let mut receive = server.receive().unwrap();
        assert_eq!(receive.take_message(), SessionMessage::Buffer(vec![2]));
        receive.complete();
        assert!(!server.is_signalled());
        server
            .reply()
            .unwrap()
            .reply(SessionMessage::Buffer(vec![4]))
            .unwrap();
        assert!(!server.is_signalled());
        drop(client);
        assert!(server.is_signalled());
    }

    #[test]
    fn cancelling_a_current_request_preserves_server_reply_ownership() {
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        client
            .request(owner(1), SessionMessage::Buffer(vec![1]))
            .unwrap();
        let mut receive = server.receive().unwrap();
        assert_eq!(receive.take_message(), SessionMessage::Buffer(vec![1]));
        receive.complete();
        assert!(client.cancel_request(owner(1)).unwrap());
        client
            .request(owner(2), SessionMessage::Buffer(vec![2]))
            .unwrap();
        assert_eq!(server.receive().unwrap_err(), SessionError::ReplyPending);
        assert_eq!(
            server
                .reply()
                .unwrap()
                .reply(SessionMessage::Buffer(vec![3])),
            Err(SessionError::PeerClosed)
        );
        let mut receive = server.receive().unwrap();
        assert_eq!(receive.take_message(), SessionMessage::Buffer(vec![2]));
        receive.complete();
        server
            .reply()
            .unwrap()
            .reply(SessionMessage::Buffer(vec![4]))
            .unwrap();
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Err(SessionError::NoRequest)
        );
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(2)).unwrap(),
            Some(SessionMessage::Buffer(vec![4]))
        );
    }

    #[derive(Debug)]
    struct Finalizer {
        session: SessionObject,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl Drop for Finalizer {
        fn drop(&mut self) {
            let _signalled = self.session.is_signalled();
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn abandoned_receive_releases_its_handles_and_completes_its_client() {
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        client
            .request(
                owner(1),
                SessionMessage::TransportedBuffer {
                    bytes: vec![0; 0x100],
                    copy_handles: vec![Some(HandleObject::new(Finalizer {
                        session: client.clone(),
                        calls: calls.clone(),
                    }))],
                    move_handles: vec![],
                },
            )
            .unwrap();
        drop(server.receive().unwrap());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Err(SessionError::PeerClosed)
        );
        client
            .request(owner(2), SessionMessage::Buffer(vec![2]))
            .unwrap();
        server.receive().unwrap().complete();
        server
            .reply()
            .unwrap()
            .reply(SessionMessage::Buffer(vec![3]))
            .unwrap();
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(2)),
            Ok(Some(SessionMessage::Buffer(vec![3])))
        );
    }

    #[test]
    fn delivered_request_handles_are_released_before_the_server_replies() {
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        client
            .request(
                owner(1),
                SessionMessage::TransportedBuffer {
                    bytes: vec![7; 0x100],
                    copy_handles: vec![Some(HandleObject::new(Finalizer {
                        session: client.clone(),
                        calls: calls.clone(),
                    }))],
                    move_handles: vec![],
                },
            )
            .unwrap();
        let mut receive = server.receive().unwrap();
        let received = receive.take_message();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        drop(received);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(server.receive().unwrap_err(), SessionError::ReplyPending);
        receive.complete();
        server
            .reply()
            .unwrap()
            .reply(SessionMessage::Buffer(vec![8]))
            .unwrap();
        assert_eq!(
            crate::process::tests::poll_session(&client, owner(1)),
            Ok(Some(SessionMessage::Buffer(vec![8])))
        );
    }

    #[test]
    fn cancelling_a_queued_request_destroys_transferred_objects_outside_the_session_lock() {
        let (server, client) = SessionObject::create_pair(&crate::process::tests::test_kernel());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let object = HandleObject::new(Finalizer {
            session: client.clone(),
            calls: calls.clone(),
        });
        client
            .request(
                owner(1),
                SessionMessage::TransportedBuffer {
                    bytes: vec![],
                    copy_handles: vec![Some(object)],
                    move_handles: vec![],
                },
            )
            .unwrap();
        assert!(client.cancel_request(owner(1)).unwrap());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(server.receive().unwrap_err(), SessionError::NoRequest);
        assert!(!client.cancel_request(owner(1)).unwrap());
    }
}
