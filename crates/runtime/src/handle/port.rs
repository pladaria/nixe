use super::*;

/// Endpoint role of a paired port.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PortEndpoint {
    Server,
    Client,
}

// Queue ownership, readiness and cleanup follow KServerPort:
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_port.cpp

#[derive(Debug)]
pub(super) struct PortState {
    server_open: bool,
    max_sessions: usize,
    pub(super) active_sessions: usize,
    is_light: bool,
    pending: VecDeque<SessionObject>,
}

#[derive(Debug)]
struct PortIdentity {
    state: Arc<Mutex<PortState>>,
    wake_event: EventObject,
}

#[derive(Debug)]
struct PortEndpointLease {
    identity: Arc<PortIdentity>,
    endpoint: PortEndpoint,
}

impl Drop for PortEndpointLease {
    fn drop(&mut self) {
        if self.endpoint != PortEndpoint::Server {
            return;
        }
        let pending = {
            let mut state = self
                .identity
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.server_open = false;
            std::mem::take(&mut state.pending)
        };
        // Session finalizers and peer wakeups must run outside the port lock.
        drop(pending);
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
    pub fn create_pair(
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        max_sessions: usize,
        is_light: bool,
    ) -> (Self, Self) {
        let identity = Arc::new(PortIdentity {
            state: Arc::new(Mutex::new(PortState {
                server_open: true,
                max_sessions,
                active_sessions: 0,
                is_light,
                pending: VecDeque::new(),
            })),
            wake_event: EventObject::new(kernel),
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
        self.endpoint == PortEndpoint::Server && !state.pending.is_empty()
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
            self.identity.wake_event.kernel(),
            state.is_light,
            Some(Arc::downgrade(&self.identity.state)),
        );
        state.active_sessions += 1;
        let notify = state.pending.is_empty();
        state.pending.push_back(server);
        let notifications = notify.then(|| self.identity.wake_event.publish_readiness(true));
        drop(state);
        if let Some(notifications) = notifications {
            notifications.notify();
        }
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
        let session = state
            .pending
            .pop_front()
            .ok_or(PortError::NoPendingSession)?;
        let notifications = self
            .identity
            .wake_event
            .publish_readiness(!state.pending.is_empty());
        drop(state);
        notifications.notify();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn server_close_disconnects_unaccepted_sessions_and_preserves_accepted_sessions() {
        for light in [false, true] {
            let (server, client) =
                PortObject::create_pair(&crate::process::tests::test_kernel(), 2, light);
            let accepted_client = client.connect().unwrap();
            let accepted_server = server.accept().unwrap();
            let pending_client = client.connect().unwrap();
            let owner = SessionRequestOwner {
                process_id: 1,
                thread_id: 1,
            };
            let message = || {
                if light {
                    SessionMessage::Light([7; 7])
                } else {
                    SessionMessage::Buffer(vec![7])
                }
            };
            pending_client.request(owner, message()).unwrap();

            let unlocked = Arc::new(AtomicBool::new(false));
            let observed = unlocked.clone();
            let state = client.identity.state.clone();
            let _watch = crate::process::tests::watch_future(
                pending_client.response(owner),
                Arc::new(move || {
                    observed.store(state.try_lock().is_ok(), Ordering::Release);
                }),
            );

            drop(server);

            assert_eq!(
                crate::process::tests::poll_session(&pending_client, owner),
                Err(SessionError::PeerClosed)
            );
            assert!(unlocked.load(Ordering::Acquire));
            assert!(matches!(client.connect(), Err(PortError::PeerClosed)));
            assert_eq!(accepted_client.request(owner, message()), Ok(()));
            let received = if light {
                accepted_server.receive_light(SessionRequestOwner {
                    process_id: 1,
                    thread_id: 2,
                })
            } else {
                accepted_server.receive().map(|mut receive| {
                    let message = receive.take_message();
                    receive.complete();
                    message
                })
            };
            assert_eq!(received, Ok(message()));
        }
    }

    #[test]
    fn client_port_close_does_not_signal_a_server_without_pending_sessions() {
        let (server, client) =
            PortObject::create_pair(&crate::process::tests::test_kernel(), 1, false);
        drop(client);
        assert!(!server.is_signalled());
        assert!(matches!(server.accept(), Err(PortError::NoPendingSession)));
    }
}
