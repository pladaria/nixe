use super::*;

/// Endpoint role of a paired port.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PortEndpoint {
    Server,
    Client,
}

// Port connection queues and endpoint-close state follow:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_port.cpp

#[derive(Debug)]
pub(super) struct PortState {
    server_open: bool,
    client_open: bool,
    max_sessions: usize,
    pub(super) active_sessions: usize,
    is_light: bool,
    pending: VecDeque<SessionObject>,
}

#[derive(Debug)]
struct PortIdentity {
    state: Arc<Mutex<PortState>>,
    wake_generation: AtomicU64,
    wake_event: EventObject,
}

#[derive(Debug)]
struct PortEndpointLease {
    identity: Arc<PortIdentity>,
    endpoint: PortEndpoint,
}

impl Drop for PortEndpointLease {
    fn drop(&mut self) {
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.endpoint {
            PortEndpoint::Server => state.server_open = false,
            PortEndpoint::Client => state.client_open = false,
        }
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
    }
}

/// One endpoint of a bounded Horizon port.
#[derive(Clone, Debug)]
pub struct PortObject {
    identity: Arc<PortIdentity>,
    endpoint: PortEndpoint,
    _lease: Arc<PortEndpointLease>,
}

impl PortObject {
    #[must_use]
    pub fn create_pair(max_sessions: usize, is_light: bool) -> (Self, Self) {
        let identity = Arc::new(PortIdentity {
            state: Arc::new(Mutex::new(PortState {
                server_open: true,
                client_open: true,
                max_sessions,
                active_sessions: 0,
                is_light,
                pending: VecDeque::new(),
            })),
            wake_generation: AtomicU64::new(0),
            wake_event: EventObject::new(),
        });
        let server_lease = Arc::new(PortEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: PortEndpoint::Server,
        });
        let client_lease = Arc::new(PortEndpointLease {
            identity: Arc::clone(&identity),
            endpoint: PortEndpoint::Client,
        });
        (
            Self {
                identity: Arc::clone(&identity),
                endpoint: PortEndpoint::Server,
                _lease: server_lease,
            },
            Self {
                identity,
                endpoint: PortEndpoint::Client,
                _lease: client_lease,
            },
        )
    }

    #[must_use]
    pub const fn endpoint(&self) -> PortEndpoint {
        self.endpoint
    }

    #[must_use]
    pub fn server_is_open(&self) -> bool {
        self.identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .server_open
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
        self.endpoint == PortEndpoint::Server && (!state.pending.is_empty() || !state.client_open)
    }

    pub fn connect(&self) -> Result<SessionObject, PortError> {
        if self.endpoint != PortEndpoint::Client {
            return Err(PortError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.server_open {
            return Err(PortError::PeerClosed);
        }
        if state.active_sessions >= state.max_sessions {
            return Err(PortError::SessionLimit);
        }
        let (server, client) = SessionObject::create_pair_with_kind(
            state.is_light,
            Some(Arc::downgrade(&self.identity.state)),
        );
        state.active_sessions += 1;
        state.pending.push_back(server);
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
        Ok(client)
    }

    pub fn accept(&self) -> Result<SessionObject, PortError> {
        if self.endpoint != PortEndpoint::Server {
            return Err(PortError::WrongEndpoint);
        }
        let mut state = self
            .identity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let session = state.pending.pop_front().ok_or_else(|| {
            if state.client_open {
                PortError::NoPendingSession
            } else {
                PortError::PeerClosed
            }
        })?;
        drop(state);
        self.identity
            .wake_generation
            .fetch_add(1, Ordering::Release);
        self.identity.wake_event.signal();
        Ok(session)
    }
}

/// Deterministic port operation failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PortError {
    WrongEndpoint,
    PeerClosed,
    SessionLimit,
    NoPendingSession,
}
