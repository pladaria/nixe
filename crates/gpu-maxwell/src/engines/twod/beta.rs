//! Typed `FERMI_TWOD_A` beta blend-constant state.

use crate::MaxwellMethodSource;

use super::MaxwellTwoDRegister;

/// Exact 32-bit value programmed through `SET_BETA1`.
///
/// NVIDIA publishes the complete bit domain but no additional field
/// interpretation, so this type deliberately preserves the guest bits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellTwoDBeta1(u32);

impl MaxwellTwoDBeta1 {
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Four byte-wide color components programmed through `SET_BETA4`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellTwoDBeta4 {
    blue: u8,
    green: u8,
    red: u8,
    alpha: u8,
}

impl MaxwellTwoDBeta4 {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            blue: raw as u8,
            green: (raw >> 8) as u8,
            red: (raw >> 16) as u8,
            alpha: (raw >> 24) as u8,
        }
    }

    #[must_use]
    pub const fn blue(self) -> u8 {
        self.blue
    }

    #[must_use]
    pub const fn green(self) -> u8 {
        self.green
    }

    #[must_use]
    pub const fn red(self) -> u8 {
        self.red
    }

    #[must_use]
    pub const fn alpha(self) -> u8 {
        self.alpha
    }

    #[must_use]
    pub fn raw(self) -> u32 {
        u32::from(self.blue)
            | (u32::from(self.green) << 8)
            | (u32::from(self.red) << 16)
            | (u32::from(self.alpha) << 24)
    }
}

/// One validated beta blend-constant register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTwoDBetaStateWrite {
    Beta1 {
        value: MaxwellTwoDBeta1,
        source: MaxwellMethodSource,
    },
    Beta4 {
        value: MaxwellTwoDBeta4,
        source: MaxwellMethodSource,
    },
}

/// Persistent beta blend-constant configuration on one Fermi 2D channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDBetaState {
    beta1: MaxwellTwoDRegister<MaxwellTwoDBeta1>,
    beta4: MaxwellTwoDRegister<MaxwellTwoDBeta4>,
}

impl MaxwellTwoDBetaState {
    #[must_use]
    pub const fn beta1(&self) -> &MaxwellTwoDRegister<MaxwellTwoDBeta1> {
        &self.beta1
    }

    #[must_use]
    pub const fn beta4(&self) -> &MaxwellTwoDRegister<MaxwellTwoDBeta4> {
        &self.beta4
    }

    pub(super) fn apply(&mut self, write: MaxwellTwoDBetaStateWrite) {
        match write {
            MaxwellTwoDBetaStateWrite::Beta1 { value, source } => {
                self.beta1 = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
            MaxwellTwoDBetaStateWrite::Beta4 { value, source } => {
                self.beta4 = MaxwellTwoDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
