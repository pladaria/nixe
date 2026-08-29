//! Shared state for Horizon's BSD socket service sessions.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use nixe_runtime::TransferMemoryObject;

/// Socket-buffer configuration supplied by `bsdInitialize`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BsdClientConfig {
    pub(crate) version: u32,
    pub(crate) tcp_tx_buffer_size: u32,
    pub(crate) tcp_rx_buffer_size: u32,
    pub(crate) tcp_tx_buffer_max_size: u32,
    pub(crate) tcp_rx_buffer_max_size: u32,
    pub(crate) udp_tx_buffer_size: u32,
    pub(crate) udp_rx_buffer_size: u32,
    pub(crate) socket_buffer_efficiency: u32,
}

#[derive(Clone, Debug)]
struct BsdClient {
    config: BsdClientConfig,
    transfer_memory: TransferMemoryObject,
    monitoring: bool,
}

#[derive(Debug, Default)]
struct BsdState {
    clients: BTreeMap<u64, BsdClient>,
}

/// Produces related BSD sessions without retaining their resources itself.
///
/// `sm:` must let consecutive service acquisitions join the same BSD service
/// instance, but it must not keep transfer memory alive after the client closes
/// every BSD session. A weak reference provides both lifetime properties.
#[derive(Clone, Debug, Default)]
pub(crate) struct BsdServiceRegistry {
    current: Arc<Mutex<Weak<Mutex<BsdState>>>>,
}

impl BsdServiceRegistry {
    pub(crate) fn open_session(&self) -> BsdSession {
        let mut current = self
            .current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let state = current.upgrade().unwrap_or_else(|| {
            let state = Arc::new(Mutex::new(BsdState::default()));
            *current = Arc::downgrade(&state);
            state
        });
        BsdSession::new(BsdSystem { state })
    }
}

/// Registry shared by all BSD sessions opened through one `sm:` session.
#[derive(Clone, Debug, Default)]
pub(crate) struct BsdSystem {
    state: Arc<Mutex<BsdState>>,
}

impl BsdSystem {
    pub(crate) fn register_client(
        &self,
        process_id: u64,
        config: BsdClientConfig,
        transfer_memory: TransferMemoryObject,
    ) -> Result<(), BsdRegistrationError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.clients.contains_key(&process_id) {
            return Err(BsdRegistrationError::AlreadyRegistered);
        }
        state.clients.insert(
            process_id,
            BsdClient {
                config,
                transfer_memory,
                monitoring: false,
            },
        );
        Ok(())
    }

    pub(crate) fn start_monitoring(&self, process_id: u64) -> Result<(), BsdMonitoringError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(client) = state.clients.get_mut(&process_id) else {
            return Err(BsdMonitoringError::UnknownClient);
        };
        client.monitoring = true;
        log::debug!(
            "bsd:u started monitoring process {process_id} (config version {}, transfer memory {:#x} bytes)",
            client.config.version,
            client.transfer_memory.size(),
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BsdSession {
    system: BsdSystem,
    domain: Arc<AtomicBool>,
}

impl BsdSession {
    pub(crate) fn new(system: BsdSystem) -> Self {
        Self {
            system,
            domain: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) const fn system(&self) -> &BsdSystem {
        &self.system
    }

    pub(crate) fn is_domain(&self) -> bool {
        self.domain.load(Ordering::Acquire)
    }

    pub(crate) fn convert_to_domain(&self) -> u32 {
        self.domain.store(true, Ordering::Release);
        1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BsdRegistrationError {
    AlreadyRegistered,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BsdMonitoringError {
    UnknownClient,
}

#[cfg(test)]
mod tests {
    use nixe_cpu::memory::MemoryPermissions;
    use nixe_memory::{CanonicalAllocation, GuestVirtualAddress};

    use super::*;

    const CONFIG: BsdClientConfig = BsdClientConfig {
        version: 1,
        tcp_tx_buffer_size: 0x8000,
        tcp_rx_buffer_size: 0x10000,
        tcp_tx_buffer_max_size: 0x40000,
        tcp_rx_buffer_max_size: 0x40000,
        udp_tx_buffer_size: 0x2400,
        udp_rx_buffer_size: 0xa500,
        socket_buffer_efficiency: 4,
    };

    fn transfer_memory() -> TransferMemoryObject {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        TransferMemoryObject::new(
            GuestVirtualAddress::new(0x1000),
            0x1000,
            MemoryPermissions::NONE,
            backing,
        )
    }

    #[test]
    fn sessions_share_registration_and_monitoring_state() {
        let registry = BsdServiceRegistry::default();
        let register_session = registry.open_session();
        let monitor_session = registry.open_session();

        register_session
            .system()
            .register_client(7, CONFIG, transfer_memory())
            .unwrap();

        assert_eq!(monitor_session.system().start_monitoring(7), Ok(()));
    }

    #[test]
    fn domain_conversion_is_local_to_one_service_session() {
        let registry = BsdServiceRegistry::default();
        let plain = registry.open_session();
        let domain = registry.open_session();
        let domain_clone = domain.clone();

        assert!(!plain.is_domain());
        assert!(!domain.is_domain());
        assert_eq!(domain.convert_to_domain(), 1);
        assert!(domain_clone.is_domain());
        assert!(!plain.is_domain());
    }

    #[test]
    fn monitoring_rejects_an_unregistered_process() {
        let system = BsdSystem::default();
        assert_eq!(
            system.start_monitoring(7),
            Err(BsdMonitoringError::UnknownClient)
        );
        system
            .register_client(7, CONFIG, transfer_memory())
            .unwrap();
        assert_eq!(system.start_monitoring(7), Ok(()));
        assert_eq!(
            system.start_monitoring(8),
            Err(BsdMonitoringError::UnknownClient)
        );
        assert_eq!(
            system.register_client(7, CONFIG, transfer_memory()),
            Err(BsdRegistrationError::AlreadyRegistered)
        );
    }

    #[test]
    fn registry_does_not_retain_closed_session_resources() {
        let registry = BsdServiceRegistry::default();
        let session = registry.open_session();
        session
            .system()
            .register_client(7, CONFIG, transfer_memory())
            .unwrap();
        drop(session);

        let replacement = registry.open_session();
        assert_eq!(
            replacement
                .system()
                .register_client(7, CONFIG, transfer_memory()),
            Ok(())
        );
    }
}
