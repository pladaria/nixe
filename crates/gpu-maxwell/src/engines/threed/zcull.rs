//! Typed `MAXWELL_B` Z-cull statistics state.
//!
//! This 3D-engine selector is deliberately separate from the channel Z-cull
//! binding and from immutable GPU-profile capabilities. Enabling statistics
//! does not by itself establish counter storage, accumulation, visibility, or
//! reporting semantics.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

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

/// One validated Z-cull statistics register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDZCullStateWrite {
    StatsEnable {
        value: MaxwellThreeDZCullStatsEnable,
        source: MaxwellMethodSource,
    },
}

/// Persistent Z-cull statistics configuration on one `MAXWELL_B` engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDZCullState {
    stats_enable: MaxwellThreeDRegister<MaxwellThreeDZCullStatsEnable>,
}

impl MaxwellThreeDZCullState {
    #[must_use]
    pub const fn stats_enable(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullStatsEnable> {
        &self.stats_enable
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDZCullStateWrite) {
        match write {
            MaxwellThreeDZCullStateWrite::StatsEnable { value, source } => {
                self.stats_enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
