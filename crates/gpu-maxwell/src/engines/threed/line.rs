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

use super::{MaxwellThreeDRawValue, MaxwellThreeDRegister};

/// Treatment of edges generated when polygon clipping precedes line-mode
/// rasterization.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1261-L1264>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPolygonClipGeneratedEdge {
    DrawLine = 0,
    DoNotDrawLine = 1,
}

impl MaxwellThreeDPolygonClipGeneratedEdge {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::DrawLine),
            1 => Some(Self::DoNotDrawLine),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether rasterized lines use Maxwell's smooth antialiasing path.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDAntiAliasedLineEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDAntiAliasedLineEnable {
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

/// Repeat factor and 16-bit pattern used by stippled lines.
///
/// NVIDIA publishes both fields in the pinned `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3119-L3121>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDLineStippleParameters {
    factor: u8,
    pattern: u16,
}

impl MaxwellThreeDLineStippleParameters {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw <= 0x00ff_ffff {
            Some(Self {
                factor: raw as u8,
                pattern: (raw >> 8) as u16,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn factor(self) -> u8 {
        self.factor
    }

    #[must_use]
    pub const fn pattern(self) -> u16 {
        self.pattern
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.factor as u32 | ((self.pattern as u32) << 8)
    }
}

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

/// Persistent line-rasterization configuration on one `MAXWELL_B` channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDLineState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDLineState<'_> {
    #[must_use]
    pub fn polygon_clip_generated_edge(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDPolygonClipGeneratedEdge> {
        self.registers.read(0x0f8c, |raw| {
            MaxwellThreeDPolygonClipGeneratedEdge::parse(raw)
        })
    }
    #[must_use]
    pub fn aliased_line_width_enable(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDAliasedLineWidthEnable> {
        self.registers.read(0x020c, |raw| {
            MaxwellThreeDAliasedLineWidthEnable::parse(raw)
        })
    }
    #[must_use]
    pub fn anti_aliased_line_enable(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDAntiAliasedLineEnable> {
        self.registers
            .read(0x1570, MaxwellThreeDAntiAliasedLineEnable::parse)
    }
    #[must_use]
    pub fn aliased_line_width(&self) -> MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        self.registers
            .read(0x13b4, |raw| Some(MaxwellThreeDRawValue::new(raw)))
    }
    #[must_use]
    pub fn stipple_enable(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x166c, |raw| (raw <= 1).then_some(raw != 0))
    }
    #[must_use]
    pub fn stipple_parameters(&self) -> MaxwellThreeDRegister<MaxwellThreeDLineStippleParameters> {
        self.registers
            .read(0x1680, MaxwellThreeDLineStippleParameters::parse)
    }
}
