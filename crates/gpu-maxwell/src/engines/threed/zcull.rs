//! Typed `MAXWELL_B` Z-cull state.
//!
//! These 3D-engine registers are deliberately separate from the channel
//! Z-cull binding and from immutable GPU-profile capabilities. Region geometry
//! describes the hardware's hierarchical depth/stencil cache, not the depth
//! attachment. Neutral backends perform ordinary depth/stencil testing and do
//! not consume this cache layout or serialize hardware culling metadata.
//! Guest-visible counter reports still require their own accumulation semantics.
//!
//! Register fields and enumerants:
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h>
//! deko3d programs this block when binding a depth attachment:
//! <https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_3d_base.cpp#L282-L297>

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Axis of a Z-cull region's size or pixel offset (unsigned 16-bit fields).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDZCullAxis {
    Width,
    Height,
    Depth,
}

/// Allocation within the hardware Z-cull cache, in aliquots, not GPU addresses.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullRegionLocation {
    start_aliquot: u16,
    aliquot_count: u16,
}

impl MaxwellThreeDZCullRegionLocation {
    #[must_use]
    pub const fn parse(raw: u32) -> Self {
        Self {
            start_aliquot: raw as u16,
            aliquot_count: (raw >> 16) as u16,
        }
    }

    #[must_use]
    pub const fn start_aliquot(self) -> u16 {
        self.start_aliquot
    }

    #[must_use]
    pub const fn aliquot_count(self) -> u16 {
        self.aliquot_count
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.start_aliquot as u32 | (self.aliquot_count as u32) << 16
    }
}

/// Cache format, independent of the depth attachment's pixel format.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDZCullRegionFormat {
    Z4x4 = 0,
    Zs4x4 = 1,
    Z4x2 = 2,
    Z2x4 = 3,
    Z16x8Block4x4 = 4,
    Z8x8Block4x2 = 5,
    Z8x8Block2x4 = 6,
    Z16x16Block4x8 = 7,
    Z4x8Block2x2 = 8,
    Zs16x8Block4x2 = 9,
    Zs16x8Block2x4 = 10,
    Zs8x8Block2x2 = 11,
    Z4x8Block1x1 = 12,
}

impl MaxwellThreeDZCullRegionFormat {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::Z4x4,
            1 => Self::Zs4x4,
            2 => Self::Z4x2,
            3 => Self::Z2x4,
            4 => Self::Z16x8Block4x4,
            5 => Self::Z8x8Block4x2,
            6 => Self::Z8x8Block2x4,
            7 => Self::Z16x16Block4x8,
            8 => Self::Z4x8Block2x2,
            9 => Self::Zs16x8Block4x2,
            10 => Self::Zs16x8Block2x4,
            11 => Self::Zs8x8Block2x2,
            12 => Self::Z4x8Block1x1,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Subregion optimization policy, not an allocation or report operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullSubregion {
    enabled: bool,
    normalized_aliquots: u32,
}

impl MaxwellThreeDZCullSubregion {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x0fff_fff1 != 0 {
            return None;
        }
        Some(Self {
            enabled: raw & 1 != 0,
            normalized_aliquots: raw >> 4,
        })
    }

    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }

    #[must_use]
    pub const fn normalized_aliquots(self) -> u32 {
        self.normalized_aliquots
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.enabled as u32 | self.normalized_aliquots << 4
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u16)]
pub enum MaxwellThreeDZCullDepthFormat {
    MostSignificantBits = 0,
    Float = 1,
    ZTrick = 2,
}

/// Z-cull's depth ordering/representation, not SET_DEPTH_FUNC or SET_ZT_FORMAT.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDZCullDirectionFormat {
    greater: bool,
    format: MaxwellThreeDZCullDepthFormat,
}

impl MaxwellThreeDZCullDirectionFormat {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & 0xffff > 1 {
            return None;
        }
        let format = match raw >> 16 {
            0 => MaxwellThreeDZCullDepthFormat::MostSignificantBits,
            1 => MaxwellThreeDZCullDepthFormat::Float,
            2 => MaxwellThreeDZCullDepthFormat::ZTrick,
            _ => return None,
        };
        Some(Self {
            greater: raw & 1 != 0,
            format,
        })
    }

    #[must_use]
    pub const fn greater(self) -> bool {
        self.greater
    }

    #[must_use]
    pub const fn format(self) -> MaxwellThreeDZCullDepthFormat {
        self.format
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.greater as u32 | (self.format as u32) << 16
    }
}

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
/// The selector is pipeline-neutral: neutral draws do not consume Maxwell's
/// hierarchical cache, regardless of its programmed geometry.
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

/// Allocation policy for one hierarchical-cache subregion. Format 15 selects
/// NONE; formats 13 and 14 are reserved by the public class header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDZCullSubregionAllocation {
    id: u8,
    aliquots: u16,
    format: Option<u8>,
}

impl MaxwellThreeDZCullSubregionAllocation {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & 0xf000_0000 != 0 {
            return None;
        }
        let format = match raw >> 24 {
            15 => None,
            value @ 0..=12 => Some(value as u8),
            _ => return None,
        };
        Some(Self {
            id: raw as u8,
            aliquots: (raw >> 8) as u16,
            format,
        })
    }
    #[must_use]
    pub const fn id(self) -> u8 {
        self.id
    }
    #[must_use]
    pub const fn aliquots(self) -> u16 {
        self.aliquots
    }
    #[must_use]
    pub const fn format(self) -> Option<u8> {
        self.format
    }
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.id as u32
            | (self.aliquots as u32) << 8
            | match self.format {
                Some(f) => f as u32,
                None => 15,
            } << 24
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDZCullSubregionAlgorithm {
    Static = 0,
    Adaptive = 1,
}

/// One validated Z-cull register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDZCullStateWrite {
    ReportSelection {
        value: u32,
        source: MaxwellMethodSource,
    },
    ReportType {
        value: u32,
        source: MaxwellMethodSource,
    },
    SubregionAllocation {
        value: MaxwellThreeDZCullSubregionAllocation,
        source: MaxwellMethodSource,
    },
    SubregionAlgorithm {
        value: MaxwellThreeDZCullSubregionAlgorithm,
        source: MaxwellMethodSource,
    },
    Storage {
        index: usize,
        value: u32,
        source: MaxwellMethodSource,
    },
    RegionLocation {
        value: MaxwellThreeDZCullRegionLocation,
        source: MaxwellMethodSource,
    },
    RegionAliquots {
        value: u16,
        source: MaxwellMethodSource,
    },
    RegionFormat {
        value: MaxwellThreeDZCullRegionFormat,
        source: MaxwellMethodSource,
    },
    RegionSize {
        axis: MaxwellThreeDZCullAxis,
        value: u16,
        source: MaxwellMethodSource,
    },
    RegionPixelOffset {
        axis: MaxwellThreeDZCullAxis,
        value: u16,
        source: MaxwellMethodSource,
    },
    Subregion {
        value: MaxwellThreeDZCullSubregion,
        source: MaxwellMethodSource,
    },
    DirectionFormat {
        value: MaxwellThreeDZCullDirectionFormat,
        source: MaxwellMethodSource,
    },
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
    report_selection: MaxwellThreeDRegister<u32>,
    report_type: MaxwellThreeDRegister<u32>,
    subregion_allocation: MaxwellThreeDRegister<MaxwellThreeDZCullSubregionAllocation>,
    subregion_algorithm: MaxwellThreeDRegister<MaxwellThreeDZCullSubregionAlgorithm>,
    storage: [MaxwellThreeDRegister<u32>; 4],
    region_location: MaxwellThreeDRegister<MaxwellThreeDZCullRegionLocation>,
    region_aliquots: MaxwellThreeDRegister<u16>,
    region_format: MaxwellThreeDRegister<MaxwellThreeDZCullRegionFormat>,
    region_size: [MaxwellThreeDRegister<u16>; 3],
    region_pixel_offset: [MaxwellThreeDRegister<u16>; 3],
    subregion: MaxwellThreeDRegister<MaxwellThreeDZCullSubregion>,
    direction_format: MaxwellThreeDRegister<MaxwellThreeDZCullDirectionFormat>,
    criterion: MaxwellThreeDRegister<MaxwellThreeDZCullCriterion>,
    enable: MaxwellThreeDRegister<MaxwellThreeDZCullEnable>,
    bounds: MaxwellThreeDRegister<MaxwellThreeDZCullBounds>,
    active_region: MaxwellThreeDRegister<MaxwellThreeDZCullRegionId>,
    stats_enable: MaxwellThreeDRegister<MaxwellThreeDZCullStatsEnable>,
}

impl MaxwellThreeDZCullState {
    /// Configuration for a later semaphore counter report; programming these
    /// registers does not itself query a counter or write a result to memory.
    #[must_use]
    pub const fn report_selection(&self) -> &MaxwellThreeDRegister<u32> {
        &self.report_selection
    }
    #[must_use]
    pub const fn report_type(&self) -> &MaxwellThreeDRegister<u32> {
        &self.report_type
    }

    #[must_use]
    pub const fn subregion_allocation(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDZCullSubregionAllocation> {
        &self.subregion_allocation
    }
    #[must_use]
    pub const fn subregion_algorithm(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDZCullSubregionAlgorithm> {
        &self.subregion_algorithm
    }

    /// Address upper/lower followed by limit-address upper/lower. Backends
    /// use their native depth cache rather than this guest hardware backing;
    /// no storage transfer is triggered by programming these registers.
    #[must_use]
    pub fn storage_word(&self, index: usize) -> Option<&MaxwellThreeDRegister<u32>> {
        self.storage.get(index)
    }

    #[must_use]
    pub const fn region_location(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDZCullRegionLocation> {
        &self.region_location
    }

    #[must_use]
    pub const fn region_aliquots(&self) -> &MaxwellThreeDRegister<u16> {
        &self.region_aliquots
    }

    #[must_use]
    pub const fn region_format(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullRegionFormat> {
        &self.region_format
    }

    #[must_use]
    pub const fn region_size(&self, axis: MaxwellThreeDZCullAxis) -> &MaxwellThreeDRegister<u16> {
        &self.region_size[axis as usize]
    }

    #[must_use]
    pub const fn region_pixel_offset(
        &self,
        axis: MaxwellThreeDZCullAxis,
    ) -> &MaxwellThreeDRegister<u16> {
        &self.region_pixel_offset[axis as usize]
    }

    #[must_use]
    pub const fn subregion(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCullSubregion> {
        &self.subregion
    }

    #[must_use]
    pub const fn direction_format(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDZCullDirectionFormat> {
        &self.direction_format
    }

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
            MaxwellThreeDZCullStateWrite::ReportSelection { value, source } => {
                self.report_selection = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDZCullStateWrite::ReportType { value, source } => {
                self.report_type = MaxwellThreeDRegister::programmed(value, value, source);
            }

            MaxwellThreeDZCullStateWrite::SubregionAllocation { value, source } => {
                self.subregion_allocation =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::SubregionAlgorithm { value, source } => {
                self.subregion_algorithm =
                    MaxwellThreeDRegister::programmed(value as u32, value, source);
            }

            MaxwellThreeDZCullStateWrite::Storage {
                index,
                value,
                source,
            } => {
                self.storage[index] = MaxwellThreeDRegister::programmed(value, value, source);
            }

            MaxwellThreeDZCullStateWrite::RegionLocation { value, source } => {
                self.region_location =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::RegionAliquots { value, source } => {
                self.region_aliquots =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDZCullStateWrite::RegionFormat { value, source } => {
                self.region_format = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::RegionSize {
                axis,
                value,
                source,
            } => {
                self.region_size[axis as usize] =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDZCullStateWrite::RegionPixelOffset {
                axis,
                value,
                source,
            } => {
                self.region_pixel_offset[axis as usize] =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDZCullStateWrite::Subregion { value, source } => {
                self.subregion = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDZCullStateWrite::DirectionFormat { value, source } => {
                self.direction_format =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
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
