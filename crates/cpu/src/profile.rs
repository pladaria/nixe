//! Immutable process CPU selection.

use core::fmt;

use nixe_memory::AddressSpaceId;

use crate::platform::{PlatformDecoder, TargetPlatform};

/// Stable identity of one supported platform CPU definition.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CpuProfileId(u64);

impl CpuProfileId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CpuProfileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "profile=0x{:016x}", self.0)
    }
}

/// Immutable CPU inputs shared by every thread in a guest process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessCpuContext {
    platform: TargetPlatform,
    address_space_id: AddressSpaceId,
}

impl ProcessCpuContext {
    #[must_use]
    pub const fn new(platform: TargetPlatform, address_space_id: AddressSpaceId) -> Self {
        Self {
            platform,
            address_space_id,
        }
    }

    #[must_use]
    pub const fn for_platform(platform: TargetPlatform, address_space_id: AddressSpaceId) -> Self {
        Self::new(platform, address_space_id)
    }

    #[must_use]
    pub const fn decoder(self) -> PlatformDecoder {
        PlatformDecoder::new(self.platform)
    }

    #[must_use]
    pub const fn platform(self) -> TargetPlatform {
        self.platform
    }

    #[must_use]
    pub const fn profile_id(self) -> CpuProfileId {
        self.platform.profile_id()
    }

    #[must_use]
    pub const fn address_space_id(self) -> AddressSpaceId {
        self.address_space_id
    }
}
