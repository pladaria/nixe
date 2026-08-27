//! Closed target-platform selection and immutable decoder binding.

use crate::profile::{GuestCpuProfile, InstructionFeature};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TargetPlatform {
    #[default]
    Switch1,
    Switch2,
}

impl TargetPlatform {
    #[must_use]
    pub const fn from_profile(profile: GuestCpuProfile) -> Self {
        if profile.id().get() == GuestCpuProfile::SWITCH_2_NATIVE_ID.get() {
            Self::Switch2
        } else {
            Self::Switch1
        }
    }

    #[must_use]
    pub const fn profile(self) -> GuestCpuProfile {
        match self {
            Self::Switch1 => GuestCpuProfile::switch_1(),
            Self::Switch2 => GuestCpuProfile::switch_2_native(),
        }
    }

    #[must_use]
    pub const fn supports(self, feature: InstructionFeature) -> bool {
        match self {
            Self::Switch1 => matches!(feature, InstructionFeature::AdvancedSimd),
            Self::Switch2 => false,
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

impl From<&GuestCpuProfile> for PlatformDecoder {
    fn from(profile: &GuestCpuProfile) -> Self {
        Self::new(TargetPlatform::from_profile(*profile))
    }
}
