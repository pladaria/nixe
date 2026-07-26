use std::fmt::{Display, Formatter};

/// Stable NVIDIA service file-descriptor identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NvDrvFileDescriptor(u32);

impl NvDrvFileDescriptor {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl Display for NvDrvFileDescriptor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "nvfd:{:#010x}", self.0)
    }
}

/// Stable identity of one service connection which can own descriptors.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NvDrvSessionId(u64);

impl NvDrvSessionId {
    pub(super) const ROOT: Self = Self(1);

    pub(super) const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// NVIDIA device node represented by one descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NvDrvDeviceKind {
    NvMap,
    HostControl,
    HostControlGpu,
    HostAddressSpaceGpu,
}

impl NvDrvDeviceKind {
    pub const fn path(self) -> &'static str {
        match self {
            Self::NvMap => "/dev/nvmap",
            Self::HostControl => "/dev/nvhost-ctrl",
            Self::HostControlGpu => "/dev/nvhost-ctrl-gpu",
            Self::HostAddressSpaceGpu => "/dev/nvhost-as-gpu",
        }
    }
}

/// Horizon service permission selected when the client connection is created.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NvDrvPermissionProfile {
    Application,
}

/// Observable lifecycle of one NVIDIA device descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NvDrvDescriptorLifecycle {
    Open,
}

/// Ownership recorded for an open device descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvDrvDescriptorOwner {
    session: NvDrvSessionId,
    process_id: u64,
}

impl NvDrvDescriptorOwner {
    pub(super) const fn new(session: NvDrvSessionId, process_id: u64) -> Self {
        Self {
            session,
            process_id,
        }
    }

    pub const fn session(self) -> NvDrvSessionId {
        self.session
    }

    pub const fn process_id(self) -> u64 {
        self.process_id
    }
}

/// Persistent semantic state associated with one guest-visible fd.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvDrvDeviceDescriptor {
    fd: NvDrvFileDescriptor,
    kind: NvDrvDeviceKind,
    owner: NvDrvDescriptorOwner,
    permission: NvDrvPermissionProfile,
    lifecycle: NvDrvDescriptorLifecycle,
}

impl NvDrvDeviceDescriptor {
    pub(super) const fn open(
        fd: NvDrvFileDescriptor,
        kind: NvDrvDeviceKind,
        owner: NvDrvDescriptorOwner,
        permission: NvDrvPermissionProfile,
    ) -> Self {
        Self {
            fd,
            kind,
            owner,
            permission,
            lifecycle: NvDrvDescriptorLifecycle::Open,
        }
    }

    pub const fn fd(self) -> NvDrvFileDescriptor {
        self.fd
    }

    pub const fn kind(self) -> NvDrvDeviceKind {
        self.kind
    }

    pub const fn owner(self) -> NvDrvDescriptorOwner {
        self.owner
    }

    pub const fn permission(self) -> NvDrvPermissionProfile {
        self.permission
    }

    pub const fn lifecycle(self) -> NvDrvDescriptorLifecycle {
        self.lifecycle
    }
}
