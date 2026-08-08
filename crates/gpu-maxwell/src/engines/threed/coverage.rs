//! Typed `MAXWELL_B` coverage-sampling configuration.
//!
//! Coverage-sampled antialiasing is deliberately modeled independently from
//! ordinary multisample state. Identifying a selector does not establish
//! separate coverage/color sample counts, evaluation, resolve, or coherency
//! semantics.
//!
//! NVIDIA's pinned public header leaves address `0x15b4` unnamed; the pinned
//! envytools register database identifies it as `CSAA_ENABLE` and publishes
//! its boolean values. Neither source establishes enabled execution semantics:
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2864-L2888>
//! <https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/rnndb/graph/gf100_3d.xml#L831-L838>

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
