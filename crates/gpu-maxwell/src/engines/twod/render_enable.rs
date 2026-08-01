//! Typed `FERMI_TWOD_A` render-enable state.

use crate::MaxwellMethodSource;

use super::MaxwellTwoDRegister;

/// Render-enable mode programmed through `SET_RENDER_ENABLE_C`.
///
/// Conditional modes require additional state when a later operation consumes
/// them. Programming the selector remains valid regardless of register-write
/// order and does not itself perform the condition test.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellTwoDRenderEnableMode {
    Disabled = 0,
    Enabled = 1,
    Conditional = 2,
    RenderIfEqual = 3,
    RenderIfNotEqual = 4,
}

impl MaxwellTwoDRenderEnableMode {
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

/// One validated render-enable register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDRenderEnableStateWrite {
    Mode {
        value: MaxwellTwoDRenderEnableMode,
        source: MaxwellMethodSource,
    },
}

/// Persistent render-enable configuration on one Fermi 2D channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDRenderEnableState {
    mode: MaxwellTwoDRegister<MaxwellTwoDRenderEnableMode>,
}

impl MaxwellTwoDRenderEnableState {
    #[must_use]
    pub const fn mode(&self) -> &MaxwellTwoDRegister<MaxwellTwoDRenderEnableMode> {
        &self.mode
    }

    pub(super) fn apply(&mut self, write: MaxwellTwoDRenderEnableStateWrite) {
        match write {
            MaxwellTwoDRenderEnableStateWrite::Mode { value, source } => {
                self.mode = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
