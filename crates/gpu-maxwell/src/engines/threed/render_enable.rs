//! Typed `MAXWELL_B` render-enable state.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

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

/// One validated 3D render-enable transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDRenderEnableStateWrite {
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
    mode: MaxwellThreeDRegister<MaxwellThreeDRenderEnableMode>,
    conditional_load_constant_buffer:
        MaxwellThreeDRegister<MaxwellThreeDConditionalLoadConstantBuffer>,
}

impl MaxwellThreeDRenderEnableState {
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
        self.mode.value().copied()
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDRenderEnableStateWrite) {
        match write {
            MaxwellThreeDRenderEnableStateWrite::Mode { value, source } => {
                self.mode = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDRenderEnableStateWrite::ConditionalLoadConstantBuffer {
                value,
                source,
            } => {
                self.conditional_load_constant_buffer =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
