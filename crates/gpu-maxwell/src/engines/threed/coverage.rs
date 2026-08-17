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

/// Target-independent rasterization mode selected by `SET_TIR`.
///
/// NVIDIA publishes the mode and the related `SET_TIR_CONTROL` bit fields in
/// the pinned `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1302-L1305>
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1801-L1810>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDTirMode {
    Disabled = 0,
    RasterNTargetM = 1,
}

impl MaxwellThreeDTirMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::RasterNTargetM),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Validated `SET_TIR_CONTROL` coverage and query policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDTirControl {
    z_pass_pixel_count_uses_raster_samples: bool,
    reduce_coverage: bool,
    alpha_to_coverage_uses_raster_samples: bool,
}

impl MaxwellThreeDTirControl {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x13 != 0 {
            return None;
        }
        Some(Self {
            z_pass_pixel_count_uses_raster_samples: raw & 1 != 0,
            reduce_coverage: raw & 2 != 0,
            alpha_to_coverage_uses_raster_samples: raw & 0x10 != 0,
        })
    }

    #[must_use]
    pub const fn z_pass_pixel_count_uses_raster_samples(self) -> bool {
        self.z_pass_pixel_count_uses_raster_samples
    }

    #[must_use]
    pub const fn reduce_coverage(self) -> bool {
        self.reduce_coverage
    }

    #[must_use]
    pub const fn alpha_to_coverage_uses_raster_samples(self) -> bool {
        self.alpha_to_coverage_uses_raster_samples
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.z_pass_pixel_count_uses_raster_samples as u32
            | ((self.reduce_coverage as u32) << 1)
            | ((self.alpha_to_coverage_uses_raster_samples as u32) << 4)
    }
}

pub const MAXWELL_SAMPLE_LOCATION_GROUP_COUNT: usize = 4;
pub const MAXWELL_SAMPLE_LOCATIONS_PER_GROUP: usize = 4;

/// One four-bit Maxwell sample coordinate pair.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDSampleLocation {
    x: u8,
    y: u8,
}

impl MaxwellThreeDSampleLocation {
    #[must_use]
    pub const fn x(self) -> u8 {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> u8 {
        self.y
    }

    #[must_use]
    pub const fn is_centered(self) -> bool {
        self.x == 8 && self.y == 8
    }
}

/// Four sample locations packed into one member of the Maxwell register array.
///
/// The register names and bit fields are established by pinned envytools data.
/// Nouveau independently programs `0x88888888` in all four registers and
/// documents that pattern as centered sample locations.
/// <https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/rnndb/graph/gf100_3d.xml#L579-L588>
/// <https://lists.debian.org/debian-x/2017/06/msg00079.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDSampleLocationGroup {
    locations: [MaxwellThreeDSampleLocation; MAXWELL_SAMPLE_LOCATIONS_PER_GROUP],
}

impl MaxwellThreeDSampleLocationGroup {
    pub(super) const fn parse(raw: u32) -> Self {
        let mut locations =
            [MaxwellThreeDSampleLocation { x: 0, y: 0 }; MAXWELL_SAMPLE_LOCATIONS_PER_GROUP];
        let mut index = 0;
        while index < MAXWELL_SAMPLE_LOCATIONS_PER_GROUP {
            let shift = index * 8;
            locations[index] = MaxwellThreeDSampleLocation {
                x: ((raw >> shift) & 0xf) as u8,
                y: ((raw >> (shift + 4)) & 0xf) as u8,
            };
            index += 1;
        }
        Self { locations }
    }

    #[must_use]
    pub const fn locations(
        &self,
    ) -> &[MaxwellThreeDSampleLocation; MAXWELL_SAMPLE_LOCATIONS_PER_GROUP] {
        &self.locations
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        let mut raw = 0;
        let mut index = 0;
        while index < MAXWELL_SAMPLE_LOCATIONS_PER_GROUP {
            raw |= (self.locations[index].x as u32) << (index * 8);
            raw |= (self.locations[index].y as u32) << (index * 8 + 4);
            index += 1;
        }
        raw
    }

    #[must_use]
    pub const fn is_centered(self) -> bool {
        let mut index = 0;
        while index < MAXWELL_SAMPLE_LOCATIONS_PER_GROUP {
            if !self.locations[index].is_centered() {
                return false;
            }
            index += 1;
        }
        true
    }
}

/// Where Maxwell evaluates the centroid for hybrid antialiasing passes.
///
/// NVIDIA publishes both encodings in the pinned `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L581-L586>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDHybridAntiAliasCentroid {
    PerFragment = 0,
    PerPass = 1,
}

/// Validated `SET_HYBRID_ANTI_ALIAS_CONTROL` state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDHybridAntiAliasControl {
    passes: u8,
    centroid: MaxwellThreeDHybridAntiAliasCentroid,
    passes_extended: bool,
}

impl MaxwellThreeDHybridAntiAliasControl {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x3f != 0 {
            return None;
        }
        Some(Self {
            passes: (raw & 0xf) as u8,
            centroid: if raw & 0x10 == 0 {
                MaxwellThreeDHybridAntiAliasCentroid::PerFragment
            } else {
                MaxwellThreeDHybridAntiAliasCentroid::PerPass
            },
            passes_extended: raw & 0x20 != 0,
        })
    }

    #[must_use]
    pub const fn passes(self) -> u8 {
        self.passes
    }

    #[must_use]
    pub const fn centroid(self) -> MaxwellThreeDHybridAntiAliasCentroid {
        self.centroid
    }

    #[must_use]
    pub const fn passes_extended(self) -> bool {
        self.passes_extended
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.passes as u32 | ((self.centroid as u32) << 4) | ((self.passes_extended as u32) << 5)
    }

    /// The captured configuration introduces no additional coverage pass or
    /// per-pass centroid behavior beyond ordinary per-fragment sampling.
    #[must_use]
    pub const fn is_single_pass_per_fragment(self) -> bool {
        self.passes == 1
            && matches!(
                self.centroid,
                MaxwellThreeDHybridAntiAliasCentroid::PerFragment
            )
            && !self.passes_extended
    }
}

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
    TirMode {
        value: MaxwellThreeDTirMode,
        source: MaxwellMethodSource,
    },
    TirControl {
        value: MaxwellThreeDTirControl,
        source: MaxwellMethodSource,
    },
    SampleLocations {
        group: u8,
        value: MaxwellThreeDSampleLocationGroup,
        source: MaxwellMethodSource,
    },
    HybridAntiAliasControl {
        value: MaxwellThreeDHybridAntiAliasControl,
        source: MaxwellMethodSource,
    },
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
    tir_mode: MaxwellThreeDRegister<MaxwellThreeDTirMode>,
    tir_control: MaxwellThreeDRegister<MaxwellThreeDTirControl>,
    sample_locations: [MaxwellThreeDRegister<MaxwellThreeDSampleLocationGroup>;
        MAXWELL_SAMPLE_LOCATION_GROUP_COUNT],
    hybrid_anti_alias_control: MaxwellThreeDRegister<MaxwellThreeDHybridAntiAliasControl>,
    ps_output_sample_mask_usage: MaxwellThreeDRegister<MaxwellThreeDPsOutputSampleMaskUsage>,
    csaa_enable: MaxwellThreeDRegister<MaxwellThreeDCsaaEnable>,
}

impl MaxwellThreeDCoverageState {
    #[must_use]
    pub const fn tir_mode(&self) -> &MaxwellThreeDRegister<MaxwellThreeDTirMode> {
        &self.tir_mode
    }

    #[must_use]
    pub const fn tir_control(&self) -> &MaxwellThreeDRegister<MaxwellThreeDTirControl> {
        &self.tir_control
    }

    #[must_use]
    pub const fn sample_locations(
        &self,
    ) -> &[MaxwellThreeDRegister<MaxwellThreeDSampleLocationGroup>;
         MAXWELL_SAMPLE_LOCATION_GROUP_COUNT] {
        &self.sample_locations
    }

    #[must_use]
    pub const fn hybrid_anti_alias_control(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDHybridAntiAliasControl> {
        &self.hybrid_anti_alias_control
    }

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
            MaxwellThreeDCoverageStateWrite::TirMode { value, source } => {
                self.tir_mode = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDCoverageStateWrite::TirControl { value, source } => {
                self.tir_control = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDCoverageStateWrite::SampleLocations {
                group,
                value,
                source,
            } => {
                self.sample_locations[group as usize] =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDCoverageStateWrite::HybridAntiAliasControl { value, source } => {
                self.hybrid_anti_alias_control =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
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
