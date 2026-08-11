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

/// Message transported by a normal or light Horizon session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionMessage {
    Buffer(Vec<u8>),
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

#[derive(Clone, Debug)]
struct SessionRequest {
    id: u64,
    owner: SessionRequestOwner,
    message: SessionMessage,
}

/// Process/thread identity of one synchronous session request owner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SessionRequestOwner {
    pub process_id: u64,
    pub thread_id: u64,
}

#[derive(Debug)]
struct SessionState {
    server_open: bool,
    client_open: bool,
    next_request_id: u64,
    queued: VecDeque<SessionRequest>,
    current: Option<SessionRequest>,
    pending_by_owner: BTreeMap<SessionRequestOwner, u64>,
    responses: BTreeMap<u64, SessionMessage>,
    owning_port: Option<Weak<Mutex<PortState>>>,
}

#[derive(Debug)]
struct SessionIdentity {
    state: Mutex<SessionState>,
    wake_generation: AtomicU64,
    wake_event: EventObject,
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
        state.queued.clear();
        state.current = None;
        state.responses.clear();
        state.pending_by_owner.clear();
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
    }
}

/// Result of submitting or polling one synchronous client request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionRequestResult {
    Submitted,
    Waiting,
    Response(SessionMessage),
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
    pub fn create_pair() -> (Self, Self) {
        Self::create_pair_with_kind(false, None)
    }

    #[must_use]
    pub fn create_light_pair() -> (Self, Self) {
        Self::create_pair_with_kind(true, None)
    }

    pub(super) fn create_pair_with_kind(
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
                pending_by_owner: BTreeMap::new(),
                responses: BTreeMap::new(),
                owning_port,
            }),
            wake_generation: AtomicU64::new(0),
            wake_event: EventObject::new(),
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
    pub fn same_session(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }

    #[must_use]
    pub const fn is_light(&self) -> bool {
        self.is_light
    }

    #[must_use]
    pub fn wake_generation(&self) -> u64 {
        self.identity.wake_generation.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn readable_event(&self) -> ReadableEventObject {
        ReadableEventObject(self.identity.wake_event.clone())
    }

    #[must_use]
    pub fn is_signalled(&self) -> bool {
        let state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            SessionEndpoint::Server => !state.queued.is_empty() || !state.client_open,
            SessionEndpoint::Client => !state.responses.is_empty() || !state.server_open,
        }
    }

    pub fn request(
        &self,
        owner: SessionRequestOwner,
        message: SessionMessage,
    ) -> Result<SessionRequestResult, SessionError> {
        if self.endpoint != SessionEndpoint::Client {
            return Err(SessionError::WrongEndpoint);
        }
        if self.is_light != message.is_light() {
            return Err(SessionError::MessageKindMismatch);
        }
        if let Some(result) = self.poll_request(owner)? {
            return Ok(result);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.server_open {
            return Err(SessionError::PeerClosed);
        }
        if state.pending_by_owner.len() >= MAX_SESSION_REQUESTS {
            return Err(SessionError::QueueFull);
        }
        let id = state.next_request_id;
        state.next_request_id = state.next_request_id.saturating_add(1);
        state.pending_by_owner.insert(owner, id);
        state
            .queued
            .push_back(SessionRequest { id, owner, message });
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
        Ok(SessionRequestResult::Submitted)
    }

    /// Polls a previously submitted request without enqueueing another message.
    pub fn poll_request(
        &self,
        owner: SessionRequestOwner,
    ) -> Result<Option<SessionRequestResult>, SessionError> {
        if self.endpoint != SessionEndpoint::Client {
            return Err(SessionError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(request_id) = state.pending_by_owner.get(&owner).copied() else {
            return Ok(None);
        };
        if let Some(response) = state.responses.remove(&request_id) {
            state.pending_by_owner.remove(&owner);
            drop(state);
            self.identity
                .wake_generation
                .fetch_add(1, Ordering::Release);
            self.identity.wake_event.signal();
            return Ok(Some(SessionRequestResult::Response(response)));
        }
        if !state.server_open {
            state.pending_by_owner.remove(&owner);
            drop(state);
            self.identity
                .wake_generation
                .fetch_add(1, Ordering::Release);
            self.identity.wake_event.signal();
            return Err(SessionError::PeerClosed);
        }
        Ok(Some(SessionRequestResult::Waiting))
    }

    pub fn receive(&self) -> Result<SessionMessage, SessionError> {
        if self.endpoint != SessionEndpoint::Server {
            return Err(SessionError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.current.is_some() {
            return Err(SessionError::ReplyPending);
        }
        let request = state.queued.pop_front().ok_or_else(|| {
            if state.client_open {
                SessionError::NoRequest
            } else {
                SessionError::PeerClosed
            }
        })?;
        let message = request.message.clone();
        state.current = Some(request);
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
        Ok(message)
    }

    pub fn reply(&self, message: SessionMessage) -> Result<(), SessionError> {
        if self.endpoint != SessionEndpoint::Server {
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
        if !state.client_open {
            return Err(SessionError::PeerClosed);
        }
        let request = state.current.take().ok_or(SessionError::NoRequest)?;
        debug_assert_eq!(
            state.pending_by_owner.get(&request.owner),
            Some(&request.id)
        );
        state.responses.insert(request.id, message);
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
        Ok(())
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
