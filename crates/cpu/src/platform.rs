//! Closed platform selection and immutable A64 decoder ownership.

use crate::profile::CpuProfileId;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TargetPlatform {
    #[default]
    Switch1,
    Switch2,
}

impl TargetPlatform {
    /// The guest architectural counter is independent of CPU execution speed.
    /// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/common/cpu_features.h
    pub const fn counter_frequency(self) -> u32 {
        match self {
            Self::Switch1 => 19_200_000,
            Self::Switch2 => 31_250_000,
        }
    }

    /// Cache geometry exposed by Cortex-A57 and Cortex-A78C respectively.
    /// https://documentation-service.arm.com/static/6193c9bef45f0b1fbf3a85dd
    pub const fn cache_type(self) -> u32 {
        match self {
            Self::Switch1 => 0x8444_c004,
            Self::Switch2 => 0x9444_c004,
        }
    }

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
