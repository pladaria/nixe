//! Typed `MAXWELL_B` coverage-sampling configuration.
//!
//! Coverage-sampled antialiasing is deliberately modeled independently from
//! ordinary multisample state. Identifying a selector does not establish
//! separate coverage/color sample counts, evaluation, resolve, or coherency
//! semantics.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Whether coverage-sampled antialiasing is selected for later draws.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDCsaaEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDCsaaEnable {
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

/// One validated coverage-sampling register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDCoverageStateWrite {
    CsaaEnable {
        value: MaxwellThreeDCsaaEnable,
        source: MaxwellMethodSource,
    },
}

/// Persistent coverage-sampling configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDCoverageState {
    csaa_enable: MaxwellThreeDRegister<MaxwellThreeDCsaaEnable>,
}

impl MaxwellThreeDCoverageState {
    #[must_use]
    pub const fn csaa_enable(&self) -> &MaxwellThreeDRegister<MaxwellThreeDCsaaEnable> {
        &self.csaa_enable
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDCoverageStateWrite) {
        match write {
            MaxwellThreeDCoverageStateWrite::CsaaEnable { value, source } => {
                self.csaa_enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
