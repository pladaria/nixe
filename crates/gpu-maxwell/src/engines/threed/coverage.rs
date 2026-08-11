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

/// Whether the fragment shader's sample-mask output participates in coverage.
///
/// The two fields and their encodings are defined by NVIDIA's public
/// `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L377-L383>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDPsOutputSampleMaskUsage {
    enabled: bool,
    qualify_by_anti_alias_enable: bool,
}

impl MaxwellThreeDPsOutputSampleMaskUsage {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x3 != 0 {
            return None;
        }
        Some(Self {
            enabled: raw & 1 != 0,
            qualify_by_anti_alias_enable: raw & 2 != 0,
        })
    }

    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }

    #[must_use]
    pub const fn qualify_by_anti_alias_enable(self) -> bool {
        self.qualify_by_anti_alias_enable
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.enabled as u32 | ((self.qualify_by_anti_alias_enable as u32) << 1)
    }

    /// Returns the effective selection, retaining an unknown AA dependency.
    #[must_use]
    pub const fn effective(self, anti_alias_enable: Option<bool>) -> Option<bool> {
        if !self.enabled {
            Some(false)
        } else if !self.qualify_by_anti_alias_enable {
            Some(true)
        } else {
            anti_alias_enable
        }
    }
}

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
    PsOutputSampleMaskUsage {
        value: MaxwellThreeDPsOutputSampleMaskUsage,
        source: MaxwellMethodSource,
    },
    CsaaEnable {
        value: MaxwellThreeDCsaaEnable,
        source: MaxwellMethodSource,
    },
}

/// Persistent coverage-sampling configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDCoverageState {
    ps_output_sample_mask_usage: MaxwellThreeDRegister<MaxwellThreeDPsOutputSampleMaskUsage>,
    csaa_enable: MaxwellThreeDRegister<MaxwellThreeDCsaaEnable>,
}

impl MaxwellThreeDCoverageState {
    #[must_use]
    pub const fn ps_output_sample_mask_usage(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDPsOutputSampleMaskUsage> {
        &self.ps_output_sample_mask_usage
    }

    #[must_use]
    pub const fn csaa_enable(&self) -> &MaxwellThreeDRegister<MaxwellThreeDCsaaEnable> {
        &self.csaa_enable
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDCoverageStateWrite) {
        match write {
            MaxwellThreeDCoverageStateWrite::PsOutputSampleMaskUsage { value, source } => {
                self.ps_output_sample_mask_usage =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDCoverageStateWrite::CsaaEnable { value, source } => {
                self.csaa_enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
