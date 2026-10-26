//! Source-preserving constant-color rendering state for `MAXWELL_B`.

use super::MaxwellThreeDRegister;
use crate::MaxwellMethodSource;

/// One component of the retained constant-color register family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDConstantColorComponent {
    Red = 0,
    Green = 1,
    Blue = 2,
    Alpha = 3,
}

impl MaxwellThreeDConstantColorComponent {
    const fn index(self) -> usize {
        self as usize
    }
}

/// Exact IEEE-754 component bits written by the guest.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDConstantColorValue(u32);

impl MaxwellThreeDConstantColorValue {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Constant-color rendering registers retained independently of shader outputs.
///
/// The reference renderer records these five words but does not replace
/// translated fragment outputs with their values:
/// <https://github.com/eden-emulator/mirror/blob/master/src/video_core/engines/maxwell_3d.h>
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDConstantColorRenderingState {
    enabled: MaxwellThreeDRegister<bool>,
    components: [MaxwellThreeDRegister<MaxwellThreeDConstantColorValue>; 4],
}

impl MaxwellThreeDConstantColorRenderingState {
    #[must_use]
    pub const fn enabled(&self) -> &MaxwellThreeDRegister<bool> {
        &self.enabled
    }

    #[must_use]
    pub const fn component(
        &self,
        component: MaxwellThreeDConstantColorComponent,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDConstantColorValue> {
        &self.components[component.index()]
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDConstantColorRenderingStateWrite) {
        match write {
            MaxwellThreeDConstantColorRenderingStateWrite::Enable { value, source: _ } => {
                self.enabled = MaxwellThreeDRegister::decoded(u32::from(value), value);
            }
            MaxwellThreeDConstantColorRenderingStateWrite::Component {
                component,
                value,
                source: _,
            } => {
                self.components[component.index()] =
                    MaxwellThreeDRegister::decoded(value.bits(), value);
            }
        }
    }
}

/// One validated constant-color rendering transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDConstantColorRenderingStateWrite {
    Enable {
        value: bool,
        source: MaxwellMethodSource,
    },
    Component {
        component: MaxwellThreeDConstantColorComponent,
        value: MaxwellThreeDConstantColorValue,
        source: MaxwellMethodSource,
    },
}
