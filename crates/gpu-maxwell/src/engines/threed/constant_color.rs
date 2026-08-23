//! Source-preserving constant-color rendering state for `MAXWELL_B`.

use super::{MaxwellThreeDRegister, state::PipelineDependencySink};
use crate::MaxwellMethodSource;

/// One component of the color that replaces shader color output when enabled.
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

/// Constant-color rendering registers published by NVIDIA.
///
/// The enable and four full-width component fields are defined together in
/// the public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1229-L1244>
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

    pub(super) fn append_pipeline_dependencies(
        &self,
        dependencies: &mut impl PipelineDependencySink,
    ) {
        match self.enabled.value() {
            Some(false) => {}
            Some(true) => {
                dependencies.push(self.enabled.raw());
                dependencies.extend(self.components.iter().map(MaxwellThreeDRegister::raw));
            }
            None => dependencies.push(self.enabled.raw()),
        }
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDConstantColorRenderingStateWrite) {
        match write {
            MaxwellThreeDConstantColorRenderingStateWrite::Enable { value, source } => {
                self.enabled = MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDConstantColorRenderingStateWrite::Component {
                component,
                value,
                source,
            } => {
                self.components[component.index()] =
                    MaxwellThreeDRegister::programmed(value.bits(), value, source);
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
