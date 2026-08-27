//! Closed platform selection and immutable A64 decoder ownership.

use crate::profile::CpuProfileId;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TargetPlatform {
    #[default]
    Switch1,
    Switch2,
}

impl TargetPlatform {
    #[must_use]
    pub const fn profile_id(self) -> CpuProfileId {
        match self {
            Self::Switch1 => CpuProfileId::new(1),
            Self::Switch2 => CpuProfileId::new(2),
        }
    }

    #[must_use]
    pub const fn data_zero_block_bytes(self) -> u32 {
        match self {
            Self::Switch1 | Self::Switch2 => 64,
        }
    }

    #[must_use]
    pub const fn user_cache_maintenance_prohibited(self) -> bool {
        matches!(self, Self::Switch1)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlatformDecoder {
    platform: TargetPlatform,
}

impl PlatformDecoder {
    #[must_use]
    pub const fn new(platform: TargetPlatform) -> Self {
        Self { platform }
    }

    #[must_use]
    pub const fn platform(self) -> TargetPlatform {
        self.platform
    }
}

impl From<TargetPlatform> for PlatformDecoder {
    fn from(platform: TargetPlatform) -> Self {
        Self::new(platform)
    }
}

impl From<&TargetPlatform> for PlatformDecoder {
    fn from(platform: &TargetPlatform) -> Self {
        Self::new(*platform)
    }
}
