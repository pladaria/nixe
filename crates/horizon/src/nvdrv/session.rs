use nixe_gpu_maxwell::{MaxwellGpuAddressSpace, MaxwellGpuChannel};

use super::{NvDrvDeviceDescriptor, NvDrvFileDescriptor, NvDrvSession, NvDrvSessionId};

impl NvDrvSession {
    /// Returns the stable identity of this service connection.
    #[must_use]
    pub const fn connection_id(&self) -> NvDrvSessionId {
        self.connection_id
    }

    /// Creates a distinct service connection sharing this client's state.
    ///
    /// This is deliberately separate from [`Clone::clone`]: the latter is
    /// used for host-side handle lookups and must not mutate guest state.
    pub(crate) fn clone_connection(&self) -> Option<Self> {
        let connection_id = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let connection_id = NvDrvSessionId::new(state.next_session_id);
            state.next_session_id = state.next_session_id.checked_add(1)?;
            connection_id
        };
        Some(Self {
            connection_id,
            state: self.state.clone(),
        })
    }

    /// Returns pointer-free descriptor state for diagnostics and focused tests.
    #[must_use]
    pub fn device_descriptor(&self, fd: NvDrvFileDescriptor) -> Option<NvDrvDeviceDescriptor> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .devices
            .get(&fd)
            .copied()
    }

    /// Returns the profile-bound address space owned by an as-gpu descriptor.
    #[must_use]
    pub fn gpu_address_space(&self, fd: NvDrvFileDescriptor) -> Option<MaxwellGpuAddressSpace> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .gpu_address_spaces
            .get(&fd)
            .cloned()
    }

    /// Returns pointer-free Maxwell channel state for diagnostics and tests.
    #[must_use]
    pub fn gpu_channel(&self, fd: NvDrvFileDescriptor) -> Option<MaxwellGpuChannel> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .nvhost_gpu
            .channel(fd)
    }
}
