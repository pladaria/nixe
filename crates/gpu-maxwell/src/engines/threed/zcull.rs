//! Typed `MAXWELL_B` Z-cull state.
//!
//! These 3D-engine registers are deliberately separate from the channel
//! Z-cull binding and from immutable GPU-profile capabilities. Selecting a
//! region does not by itself establish its storage or geometry, and enabling
//! statistics does not establish counter accumulation, visibility, or
//! reporting semantics.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Identifier selected for later Z-cull work.
///
/// NVIDIA publishes `SET_ACTIVE_ZCULL_REGION` and its six-bit `ID` field in
/// the pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2799-L2800>
///
/// The selector remains pipeline-neutral until the selected region has
/// modeled storage and geometry that a draw can actually consume.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullRegionId(u8);

impl MaxwellThreeDZCullRegionId {
    #[must_use]
    pub const fn new(raw: u32) -> Option<Self> {
        if raw <= 0x3f {
            Some(Self(raw as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn id(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// Whether later 3D work accumulates Z-cull statistics.
///
/// NVIDIA publishes `SET_ZCULL_STATS`, its one-bit `ENABLE` field, and both
/// boolean encodings in the pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2699-L2710>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDZCullStatsEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDZCullStatsEnable {
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

/// One validated Z-cull register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDZCullStateWrite {
    ActiveRegion {
        value: MaxwellThreeDZCullRegionId,
        source: MaxwellMethodSource,
    },
    StatsEnable {
        value: MaxwellThreeDZCullStatsEnable,
        source: MaxwellMethodSource,
    },
}

/// Persistent Z-cull configuration on one `MAXWELL_B` engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDZCullState {
    active_region: MaxwellThreeDRegister<MaxwellThreeDZCullRegionId>,
    stats_enable: MaxwellThreeDRegister<MaxwellThreeDZCullStatsEnable>,
}

impl MaxwellThreeDZCullState {
    #[must_use]
    pub const fn active_region(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullRegionId> {
        &self.active_region
    }

    #[must_use]
    pub const fn stats_enable(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullStatsEnable> {
        &self.stats_enable
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDZCullStateWrite) {
        match write {
            MaxwellThreeDZCullStateWrite::ActiveRegion { value, source } => {
                self.active_region = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::StatsEnable { value, source } => {
                self.stats_enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
