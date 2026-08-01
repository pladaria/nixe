//! Typed `MAXWELL_B` line-rasterization selection state.
//!
//! Selecting the aliased-width path is distinct from programming either line
//! width and from anti-aliasing or multisample state. This module therefore
//! retains the selector without inventing execution semantics for that path.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Which line-width register family later line rasterization selects.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDAliasedLineWidthEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDAliasedLineWidthEnable {
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

/// One validated line-rasterization register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDLineStateWrite {
    AliasedLineWidthEnable {
        value: MaxwellThreeDAliasedLineWidthEnable,
        source: MaxwellMethodSource,
    },
}

/// Persistent line-rasterization configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDLineState {
    aliased_line_width_enable: MaxwellThreeDRegister<MaxwellThreeDAliasedLineWidthEnable>,
}

impl MaxwellThreeDLineState {
    #[must_use]
    pub const fn aliased_line_width_enable(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDAliasedLineWidthEnable> {
        &self.aliased_line_width_enable
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDLineStateWrite) {
        match write {
            MaxwellThreeDLineStateWrite::AliasedLineWidthEnable { value, source } => {
                self.aliased_line_width_enable =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
