//! Typed state controlling guest-visible `MAXWELL_B` performance counters.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Whether later raster work contributes to the Z-pass pixel counter.
///
/// This controls instrumentation, not raster output. Neutral draws can execute
/// without a host counter implementation, but a future guest-visible report or
/// reset must provide verified counter semantics rather than fabricate data.
///
/// NVIDIA publishes `SET_ZPASS_PIXEL_COUNT`, its one-bit `ENABLE` field, and
/// both boolean encodings in the pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2699-L2702>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDZPassPixelCountEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDZPassPixelCountEnable {
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

/// One validated performance-counter state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDCounterStateWrite {
    ZPassPixelCountEnable {
        value: MaxwellThreeDZPassPixelCountEnable,
        source: MaxwellMethodSource,
    },
}

/// Persistent performance-counter configuration on one `MAXWELL_B` engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDCounterState {
    zpass_pixel_count_enable: MaxwellThreeDRegister<MaxwellThreeDZPassPixelCountEnable>,
}

impl MaxwellThreeDCounterState {
    #[must_use]
    pub const fn zpass_pixel_count_enable(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDZPassPixelCountEnable> {
        &self.zpass_pixel_count_enable
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDCounterStateWrite) {
        match write {
            MaxwellThreeDCounterStateWrite::ZPassPixelCountEnable { value, source } => {
                self.zpass_pixel_count_enable =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
