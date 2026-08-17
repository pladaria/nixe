//! Typed `MAXWELL_B` Z-cull state.
//!
//! These 3D-engine registers are deliberately separate from the channel
//! Z-cull binding and from immutable GPU-profile capabilities. Selecting a
//! region does not by itself establish its storage or geometry, and enabling
//! statistics does not establish counter accumulation, visibility, or
//! reporting semantics.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Stencil comparison used by Maxwell's internal Z-cull criterion.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDZCullStencilFunction {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
}

impl MaxwellThreeDZCullStencilFunction {
    const fn parse(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Never),
            1 => Some(Self::Less),
            2 => Some(Self::Equal),
            3 => Some(Self::LessOrEqual),
            4 => Some(Self::Greater),
            5 => Some(Self::NotEqual),
            6 => Some(Self::GreaterOrEqual),
            7 => Some(Self::Always),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self as u8
    }
}

/// Early-stencil criterion retained by Maxwell's Z-cull unit.
///
/// This is optimization state rather than the ordinary stencil-test state.
/// A backend may preserve rendering semantics by using its normal late
/// depth/stencil path, so the criterion deliberately remains pipeline-neutral.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1060-L1077>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullCriterion {
    stencil_function: MaxwellThreeDZCullStencilFunction,
    no_invalidate: bool,
    force_match: bool,
    stencil_reference: u8,
    stencil_mask: u8,
}

impl MaxwellThreeDZCullCriterion {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & !0xffff_03ff != 0 {
            return None;
        }
        let Some(stencil_function) = MaxwellThreeDZCullStencilFunction::parse((raw & 0xff) as u8)
        else {
            return None;
        };
        Some(Self {
            stencil_function,
            no_invalidate: raw & (1 << 8) != 0,
            force_match: raw & (1 << 9) != 0,
            stencil_reference: ((raw >> 16) & 0xff) as u8,
            stencil_mask: (raw >> 24) as u8,
        })
    }

    #[must_use]
    pub const fn stencil_function(self) -> MaxwellThreeDZCullStencilFunction {
        self.stencil_function
    }

    #[must_use]
    pub const fn no_invalidate(self) -> bool {
        self.no_invalidate
    }

    #[must_use]
    pub const fn force_match(self) -> bool {
        self.force_match
    }

    #[must_use]
    pub const fn stencil_reference(self) -> u8 {
        self.stencil_reference
    }

    #[must_use]
    pub const fn stencil_mask(self) -> u8 {
        self.stencil_mask
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.stencil_function.raw() as u32
            | (self.no_invalidate as u32) << 8
            | (self.force_match as u32) << 9
            | (self.stencil_reference as u32) << 16
            | (self.stencil_mask as u32) << 24
    }
}

/// Early depth/stencil rejection domains enabled on Maxwell.
///
/// Z-cull is an implementation optimization: a backend may retain this state
/// while using its ordinary depth/stencil path without changing rendering.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3453-L3459>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullEnable {
    depth: bool,
    stencil: bool,
}

impl MaxwellThreeDZCullEnable {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x11 != 0 {
            return None;
        }
        Some(Self {
            depth: raw & 1 != 0,
            stencil: raw & 0x10 != 0,
        })
    }

    #[must_use]
    pub const fn depth(self) -> bool {
        self.depth
    }

    #[must_use]
    pub const fn stencil(self) -> bool {
        self.stencil
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.depth as u32 | (self.stencil as u32) << 4
    }
}

/// Whether either Z-cull depth bound is treated as unbounded.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3461-L3467>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullBounds {
    minimum_unbounded: bool,
    maximum_unbounded: bool,
}

impl MaxwellThreeDZCullBounds {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x11 != 0 {
            return None;
        }
        Some(Self {
            minimum_unbounded: raw & 1 != 0,
            maximum_unbounded: raw & 0x10 != 0,
        })
    }

    #[must_use]
    pub const fn minimum_unbounded(self) -> bool {
        self.minimum_unbounded
    }

    #[must_use]
    pub const fn maximum_unbounded(self) -> bool {
        self.maximum_unbounded
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.minimum_unbounded as u32 | (self.maximum_unbounded as u32) << 4
    }
}

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
/// This is source-preserving instrumentation policy rather than raster output
/// state. Neutral draws may proceed without accumulating counters; a future
/// guest-visible counter query must provide verified accumulation and reporting
/// semantics instead of synthesizing results from this enable bit.
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
    Criterion {
        value: MaxwellThreeDZCullCriterion,
        source: MaxwellMethodSource,
    },
    Enable {
        value: MaxwellThreeDZCullEnable,
        source: MaxwellMethodSource,
    },
    Bounds {
        value: MaxwellThreeDZCullBounds,
        source: MaxwellMethodSource,
    },
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
    criterion: MaxwellThreeDRegister<MaxwellThreeDZCullCriterion>,
    enable: MaxwellThreeDRegister<MaxwellThreeDZCullEnable>,
    bounds: MaxwellThreeDRegister<MaxwellThreeDZCullBounds>,
    active_region: MaxwellThreeDRegister<MaxwellThreeDZCullRegionId>,
    stats_enable: MaxwellThreeDRegister<MaxwellThreeDZCullStatsEnable>,
}

impl MaxwellThreeDZCullState {
    #[must_use]
    pub const fn criterion(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullCriterion> {
        &self.criterion
    }

    #[must_use]
    pub const fn enable(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullEnable> {
        &self.enable
    }

    #[must_use]
    pub const fn bounds(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullBounds> {
        &self.bounds
    }

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
            MaxwellThreeDZCullStateWrite::Criterion { value, source } => {
                self.criterion = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::Enable { value, source } => {
                self.enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::Bounds { value, source } => {
                self.bounds = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::ActiveRegion { value, source } => {
                self.active_region = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::StatsEnable { value, source } => {
                self.stats_enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
