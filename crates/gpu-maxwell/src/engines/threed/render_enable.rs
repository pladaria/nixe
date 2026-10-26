//! Typed `MAXWELL_B` render-enable state.

use crate::MaxwellMethodSource;

use super::{MaxwellThreeDRegister, MaxwellThreeDUnresolvedAddress};

/// Whether render-enable evaluation may conditionally load a constant buffer.
///
/// NVIDIA publishes the selector but not the load target, ordering, or
/// visibility rules. Disabled is therefore neutral, while enabled state is
/// retained for explicit rejection at the operation boundary.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L393-L396>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDConditionalLoadConstantBuffer {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDConditionalLoadConstantBuffer {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Render-enable mode programmed through `MAXWELL_B::SET_RENDER_ENABLE_C`.
///
/// This type is intentionally distinct from the identically encoded Fermi 2D
/// register: class state and future execution rules remain engine-owned.
/// NVIDIA defines the A/B address fragments and all five C mode encodings at:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2759-L2771>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDRenderEnableMode {
    Disabled = 0,
    Enabled = 1,
    Conditional = 2,
    RenderIfEqual = 3,
    RenderIfNotEqual = 4,
}

impl MaxwellThreeDRenderEnableMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            2 => Some(Self::Conditional),
            3 => Some(Self::RenderIfEqual),
            4 => Some(Self::RenderIfNotEqual),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Override of the programmed render-enable condition.
/// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3427-L3431
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDRenderEnableOverride {
    UseRenderEnable = 0,
    AlwaysRender = 1,
    NeverRender = 2,
}

impl MaxwellThreeDRenderEnableOverride {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::UseRenderEnable),
            1 => Some(Self::AlwaysRender),
            2 => Some(Self::NeverRender),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// One validated 3D render-enable transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDRenderEnableStateWrite {
    AddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    AddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    Override {
        value: MaxwellThreeDRenderEnableOverride,
        source: MaxwellMethodSource,
    },
    Mode {
        value: MaxwellThreeDRenderEnableMode,
        source: MaxwellMethodSource,
    },
    ConditionalLoadConstantBuffer {
        value: MaxwellThreeDConditionalLoadConstantBuffer,
        source: MaxwellMethodSource,
    },
}

/// Persistent render-enable configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDRenderEnableState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    override_mode: MaxwellThreeDRegister<MaxwellThreeDRenderEnableOverride>,
    mode: MaxwellThreeDRegister<MaxwellThreeDRenderEnableMode>,
    conditional_load_constant_buffer:
        MaxwellThreeDRegister<MaxwellThreeDConditionalLoadConstantBuffer>,
}

impl MaxwellThreeDRenderEnableState {
    /// NVIDIA's SET_RENDER_ENABLE_A/B form one 40-bit condition address.
    /// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }
    #[must_use]
    pub const fn override_mode(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRenderEnableOverride> {
        &self.override_mode
    }

    #[must_use]
    pub const fn mode(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRenderEnableMode> {
        &self.mode
    }

    #[must_use]
    pub const fn conditional_load_constant_buffer(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDConditionalLoadConstantBuffer> {
        &self.conditional_load_constant_buffer
    }

    pub(in crate::engines) fn execution_mode(&self) -> Option<MaxwellThreeDRenderEnableMode> {
        match self.override_mode.value() {
            Some(MaxwellThreeDRenderEnableOverride::AlwaysRender) => {
                Some(MaxwellThreeDRenderEnableMode::Enabled)
            }
            Some(MaxwellThreeDRenderEnableOverride::NeverRender) => {
                Some(MaxwellThreeDRenderEnableMode::Disabled)
            }
            _ => self.mode.value().copied(),
        }
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDRenderEnableStateWrite) {
        match write {
            MaxwellThreeDRenderEnableStateWrite::AddressUpper { value, source: _ } => {
                self.address_upper = MaxwellThreeDRegister::decoded(u32::from(value), value);
            }
            MaxwellThreeDRenderEnableStateWrite::AddressLower { value, source: _ } => {
                self.address_lower = MaxwellThreeDRegister::decoded(value, value);
            }
            MaxwellThreeDRenderEnableStateWrite::Override { value, source: _ } => {
                self.override_mode = MaxwellThreeDRegister::decoded(value.raw(), value);
            }
            MaxwellThreeDRenderEnableStateWrite::Mode { value, source: _ } => {
                self.mode = MaxwellThreeDRegister::decoded(value.raw(), value);
            }
            MaxwellThreeDRenderEnableStateWrite::ConditionalLoadConstantBuffer {
                value,
                source: _,
            } => {
                self.conditional_load_constant_buffer =
                    MaxwellThreeDRegister::decoded(value.raw(), value);
            }
        }
    }
}
