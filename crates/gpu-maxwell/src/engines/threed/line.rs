//! Typed `MAXWELL_B` line-rasterization selection state.
//!
//! Selecting the aliased-width path is distinct from programming either line
//! width and from anti-aliasing or multisample state. This module therefore
//! retains the selector without inventing execution semantics for that path.
//!
//! NVIDIA publishes `SET_ALIASED_LINE_WIDTH_ENABLE` and its boolean values;
//! the pinned envytools database independently calls the selector
//! `LINE_WIDTH_SEPARATE`, which does not establish broader raster semantics:
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L244-L247>
//! <https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/rnndb/graph/gf100_3d.xml#L60-L68>

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
