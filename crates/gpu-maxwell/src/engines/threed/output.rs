//! Typed fixed-function output state for `MAXWELL_B`.

use crate::MaxwellMethodSource;

use super::render_targets::{MaxwellThreeDRawValue, MaxwellThreeDRectangle};
use super::state::{
    MAXWELL_THREE_D_POLYGON_MODE_RESET, MAXWELL_THREE_D_WINDOW_ORIGIN_RESET, MaxwellThreeDRegister,
};

pub const MAXWELL_VIEWPORT_COUNT: usize = 16;
pub const MAXWELL_SCISSOR_COUNT: usize = 16;
pub const MAXWELL_WINDOW_CLIP_COUNT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDCompareOp {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl MaxwellThreeDCompareOp {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        let normalized = match raw {
            0x200..=0x207 => raw - 0x200,
            1..=8 => raw - 1,
            _ => return None,
        };
        Some(match normalized {
            0 => Self::Never,
            1 => Self::Less,
            2 => Self::Equal,
            3 => Self::LessEqual,
            4 => Self::Greater,
            5 => Self::NotEqual,
            6 => Self::GreaterEqual,
            7 => Self::Always,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDPolygonMode {
    Point,
    Line,
    Fill,
}

/// Provoking-vertex interpolation mode selected by `SET_SHADE_MODE`.
///
/// Encodings are defined by NVIDIA's public `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDShadeMode {
    Flat = 0x1d00,
    Smooth = 0x1d01,
}

/// Vertex whose `flat` outputs supply the value for an entire primitive.
///
/// NVIDIA publishes the selector and its two encodings in the pinned
/// `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3123-L3126>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDProvokingVertex {
    First = 0,
    Last = 1,
}

impl MaxwellThreeDProvokingVertex {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::First),
            1 => Some(Self::Last),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Clamp interval selected for one saturated pixel-shader output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDPixelShaderClampRange {
    ZeroToOne,
    MinusOneToOne,
}

/// Saturation controls for all eight pixel-shader color outputs.
///
/// Each nibble carries an enable bit and a signed-range bit, as published in
/// NVIDIA's pinned `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2549-L2597>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDPixelShaderSaturate(u32);

impl MaxwellThreeDPixelShaderSaturate {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x3333_3333 == 0 {
            Some(Self(raw))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub fn output_enabled(self, output: u8) -> Option<bool> {
        if output < 8 {
            Some(self.0 & (1 << (u32::from(output) * 4)) != 0)
        } else {
            None
        }
    }

    #[must_use]
    pub fn clamp_range(self, output: u8) -> Option<MaxwellThreeDPixelShaderClampRange> {
        if output >= 8 {
            return None;
        }
        if self.0 & (2 << (u32::from(output) * 4)) == 0 {
            Some(MaxwellThreeDPixelShaderClampRange::ZeroToOne)
        } else {
            Some(MaxwellThreeDPixelShaderClampRange::MinusOneToOne)
        }
    }

    #[must_use]
    pub fn first_enabled_output(self) -> Option<u8> {
        let mut output = 0;
        while output < 8 {
            if self.output_enabled(output) == Some(true) {
                return Some(output);
            }
            output += 1;
        }
        None
    }
}

/// Combination rule applied to the programmed window-clip rectangles.
///
/// NVIDIA publishes all three encodings in the pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3443-L3447>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDWindowClipType {
    Inclusive = 0,
    Exclusive = 1,
    ClipAll = 2,
}

impl MaxwellThreeDWindowClipType {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Inclusive),
            1 => Some(Self::Exclusive),
            2 => Some(Self::ClipAll),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether later rasterization applies the surface clip-ID test.
///
/// NVIDIA publishes `SET_CLIP_ID_TEST`, its one-bit `ENABLE` field, and both
/// boolean encodings in the pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3500-L3503>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDClipIdTestEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDClipIdTestEnable {
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

impl MaxwellThreeDShadeMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x1d00 => Some(Self::Flat),
            0x1d01 => Some(Self::Smooth),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}
impl MaxwellThreeDPolygonMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x1b00 => Some(Self::Point),
            0x1b01 => Some(Self::Line),
            0x1b02 => Some(Self::Fill),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDFrontFace {
    Clockwise,
    CounterClockwise,
}
impl MaxwellThreeDFrontFace {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x900 => Some(Self::Clockwise),
            0x901 => Some(Self::CounterClockwise),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDCullFace {
    Front,
    Back,
    FrontAndBack,
}
impl MaxwellThreeDCullFace {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x404 => Some(Self::Front),
            0x405 => Some(Self::Back),
            0x408 => Some(Self::FrontAndBack),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDSampleMode {
    Samples1x1,
    Samples2x1,
    Samples2x2,
    Samples4x2,
    Samples4x2D3D,
    Samples2x1D3D,
    Samples4x4,
    Coverage2x2x4,
    Coverage2x2x12,
    Coverage4x2x8,
    Coverage4x2x24,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDStencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}
impl MaxwellThreeDStencilOp {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x1e00 | 1 => Some(Self::Keep),
            0 | 2 => Some(Self::Zero),
            0x1e01 | 3 => Some(Self::Replace),
            0x1e02 | 4 => Some(Self::IncrementClamp),
            0x1e03 | 5 => Some(Self::DecrementClamp),
            0x150a | 6 => Some(Self::Invert),
            0x8507 | 7 => Some(Self::IncrementWrap),
            0x8508 | 8 => Some(Self::DecrementWrap),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDBlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Bitwise operation applied between fragment and color-target values.
///
/// All sixteen encodings are published in NVIDIA's pinned `MAXWELL_B` header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3521-L3543>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDLogicOp {
    Clear = 0x1500,
    And = 0x1501,
    AndReverse = 0x1502,
    Copy = 0x1503,
    AndInverted = 0x1504,
    Noop = 0x1505,
    Xor = 0x1506,
    Or = 0x1507,
    Nor = 0x1508,
    Equiv = 0x1509,
    Invert = 0x150a,
    OrReverse = 0x150b,
    CopyInverted = 0x150c,
    OrInverted = 0x150d,
    Nand = 0x150e,
    Set = 0x150f,
}

impl MaxwellThreeDLogicOp {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        Some(match raw {
            0x1500 => Self::Clear,
            0x1501 => Self::And,
            0x1502 => Self::AndReverse,
            0x1503 => Self::Copy,
            0x1504 => Self::AndInverted,
            0x1505 => Self::Noop,
            0x1506 => Self::Xor,
            0x1507 => Self::Or,
            0x1508 => Self::Nor,
            0x1509 => Self::Equiv,
            0x150a => Self::Invert,
            0x150b => Self::OrReverse,
            0x150c => Self::CopyInverted,
            0x150d => Self::OrInverted,
            0x150e => Self::Nand,
            0x150f => Self::Set,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether the common blend state is active for later color-target draws.
///
/// NVIDIA's public class header defines the boolean selector but does not
/// establish a neutral host blend-state contract:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDBlendEnableCommon {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDBlendEnableCommon {
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

/// Color and alpha participation in Maxwell's iterated-blend path.
///
/// Iterated blending performs additional blend passes and therefore changes
/// output semantics when either bit is enabled. A disabled value is neutral;
/// enabled values must remain visible to draw lowering rather than being
/// mistaken for ordinary single-pass blending.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cla297.h#L1065-L1074>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDIteratedBlend {
    color_enabled: bool,
    alpha_enabled: bool,
}

impl MaxwellThreeDIteratedBlend {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x3 != 0 {
            return None;
        }
        Some(Self {
            color_enabled: raw & 1 != 0,
            alpha_enabled: raw & 2 != 0,
        })
    }

    #[must_use]
    pub const fn color_enabled(self) -> bool {
        self.color_enabled
    }

    #[must_use]
    pub const fn alpha_enabled(self) -> bool {
        self.alpha_enabled
    }

    #[must_use]
    pub const fn enabled(self) -> bool {
        self.color_enabled || self.alpha_enabled
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.color_enabled as u32 | (self.alpha_enabled as u32) << 1
    }
}

/// Eight-bit pass count paired with `SET_ITERATED_BLEND`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDIteratedBlendPassCount(u8);

impl MaxwellThreeDIteratedBlendPassCount {
    #[must_use]
    pub const fn new(pass_count: u8) -> Self {
        Self(pass_count)
    }

    #[must_use]
    pub const fn pass_count(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// Whether format-specific blend handling is selected for
/// `SNORM8`/`UNORM16`/`SNORM16` color targets.
///
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1828-L1831>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDBlendPerFormatEnable {
    Disabled = 0,
    Enabled = 0x10,
}

impl MaxwellThreeDBlendPerFormatEnable {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            0x10 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether hardware may apply floating-point pixel-kill optimization.
///
/// This is an optimization permission, not output arithmetic or coherency.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1345-L1348>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDBlendFloatPixelKillEnable {
    Disallowed = 0,
    Allowed = 1,
}

impl MaxwellThreeDBlendFloatPixelKillEnable {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disallowed),
            1 => Some(Self::Allowed),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether blend arithmetic defines zero multiplied by any value as zero.
///
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3516-L3519>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDBlendZeroTimesAnythingIsZero {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDBlendZeroTimesAnythingIsZero {
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

/// Format, arithmetic, and optimization controls shared by blend paths.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDBlendControlState {
    per_format_enable: MaxwellThreeDRegister<MaxwellThreeDBlendPerFormatEnable>,
    float_pixel_kill_enable: MaxwellThreeDRegister<MaxwellThreeDBlendFloatPixelKillEnable>,
    zero_times_anything_is_zero: MaxwellThreeDRegister<MaxwellThreeDBlendZeroTimesAnythingIsZero>,
    iterated_blend: MaxwellThreeDRegister<MaxwellThreeDIteratedBlend>,
    iterated_blend_pass_count: MaxwellThreeDRegister<MaxwellThreeDIteratedBlendPassCount>,
}

impl MaxwellThreeDBlendControlState {
    #[must_use]
    pub const fn per_format_enable(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDBlendPerFormatEnable> {
        &self.per_format_enable
    }

    #[must_use]
    pub const fn float_pixel_kill_enable(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDBlendFloatPixelKillEnable> {
        &self.float_pixel_kill_enable
    }

    #[must_use]
    pub const fn zero_times_anything_is_zero(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDBlendZeroTimesAnythingIsZero> {
        &self.zero_times_anything_is_zero
    }

    #[must_use]
    pub const fn iterated_blend(&self) -> &MaxwellThreeDRegister<MaxwellThreeDIteratedBlend> {
        &self.iterated_blend
    }

    #[must_use]
    pub const fn iterated_blend_pass_count(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDIteratedBlendPassCount> {
        &self.iterated_blend_pass_count
    }
}

impl MaxwellThreeDBlendOp {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x8006 | 1 => Some(Self::Add),
            0x800a | 2 => Some(Self::Subtract),
            0x800b | 3 => Some(Self::ReverseSubtract),
            0x8007 | 4 => Some(Self::Min),
            0x8008 | 5 => Some(Self::Max),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDBlendFactor(u32);
impl MaxwellThreeDBlendFactor {
    pub(super) fn parse(raw: u32) -> Option<Self> {
        const OGL: &[u32] = &[
            0x4000, 0x4001, 0x4300, 0x4301, 0x4302, 0x4303, 0x4304, 0x4305, 0x4306, 0x4307, 0x4308,
            0xc001, 0xc002, 0xc003, 0xc004, 0xc900, 0xc901, 0xc902, 0xc903,
        ];
        ((1..=0x13).contains(&raw) || OGL.contains(&raw)).then_some(Self(raw))
    }
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}
impl MaxwellThreeDSampleMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Samples1x1),
            1 => Some(Self::Samples2x1),
            2 => Some(Self::Samples2x2),
            3 => Some(Self::Samples4x2),
            4 => Some(Self::Samples4x2D3D),
            5 => Some(Self::Samples2x1D3D),
            6 => Some(Self::Samples4x4),
            8 => Some(Self::Coverage2x2x4),
            9 => Some(Self::Coverage2x2x12),
            10 => Some(Self::Coverage4x2x8),
            11 => Some(Self::Coverage4x2x24),
            _ => None,
        }
    }
}

/// Per-component color-write mask for one render target.
///
/// NVIDIA's public `MAXWELL_B` header exposes eight masks and a separate
/// `SET_SINGLE_CT_WRITE_CONTROL` selector. Keeping both pieces of state lets
/// draw validation select the effective mask without discarding guest intent.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1266-L1269>
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3580-L3592>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorMask {
    pub red: bool,
    pub green: bool,
    pub blue: bool,
    pub alpha: bool,
}
impl MaxwellThreeDColorMask {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x1111 != 0 {
            return None;
        }
        Some(Self {
            red: raw & 1 != 0,
            green: raw & 0x10 != 0,
            blue: raw & 0x100 != 0,
            alpha: raw & 0x1000 != 0,
        })
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.red as u32
            | ((self.green as u32) << 4)
            | ((self.blue as u32) << 8)
            | ((self.alpha as u32) << 12)
    }

    #[must_use]
    pub const fn all_enabled(self) -> bool {
        self.red && self.green && self.blue && self.alpha
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDViewportClipControl(u32);
impl MaxwellThreeDViewportClipControl {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        let geometry_clip = (raw >> 11) & 7;
        if raw & !0x3c9f == 0 && geometry_clip <= 6 && ((raw >> 1) & 3) <= 2 {
            Some(Self(raw))
        } else {
            None
        }
    }
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Selects whether the programmed viewport scale and offset transform is active.
///
/// NVIDIA defines this as the sole bit of `SET_VIEWPORT_SCALE_OFFSET` in its
/// pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3367-L3370>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDViewportScaleOffsetEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDViewportScaleOffsetEnable {
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDFixedFunctionRegister {
    RasterEnable,
    FrontPolygonMode,
    BackPolygonMode,
    CullEnable,
    FrontFace,
    CullFace,
    LineWidth,
    DepthTestEnable,
    ShadeMode,
    DepthWriteEnable,
    DepthCompare,
    StencilTestEnable,
    TwoSidedStencilTestEnable,
    BlendPerTargetEnable,
    AntiAliasEnable,
    AlphaToCoverageEnable,
    AlphaToOneEnable,
    SampleMode,
    RasterSampleMode,
    SampleMaskControl,
    SampleMask0,
    SampleMask1,
    SampleMask2,
    SampleMask3,
    UserClipEnable,
    ViewportScaleOffsetEnable,
    ViewportClipControl,
    DepthBoundsEnable,
    DepthBoundsMin,
    PolygonOffsetPointEnable,
    PolygonOffsetLineEnable,
    PolygonOffsetFillEnable,
    FrontStencilFail,
    FrontStencilDepthFail,
    FrontStencilPass,
    FrontStencilCompare,
    FrontStencilReference,
    FrontStencilCompareMask,
    FrontStencilWriteMask,
    BackStencilFail,
    BackStencilDepthFail,
    BackStencilPass,
    BackStencilCompare,
    BackStencilReference,
    BackStencilCompareMask,
    BackStencilWriteMask,
    BlendSeparateAlpha,
    BlendColorOp,
    BlendColorSource,
    BlendColorDestination,
    BlendAlphaOp,
    BlendAlphaSource,
    BlendAlphaDestination,
    BlendConstantRed,
    BlendConstantGreen,
    BlendConstantBlue,
    BlendConstantAlpha,
    WindowOffsetX,
    WindowOffsetY,
    WindowOrigin,
    UserClipOperation,
    WindowClipEnable,
    WindowClipType,
    ClipIdTestEnable,
    DepthBoundsMax,
    LogicOpEnable,
    LogicOpFunction,
    SingleColorTargetWriteControl,
    AlphaTestEnable,
    AlphaTestReference,
    AlphaTestFunction,
    ProvokingVertex,
    TwoSidedLightEnable,
    ColorClampEnable,
    PixelShaderSaturate,
}

impl MaxwellThreeDFixedFunctionRegister {
    const COUNT: usize = Self::PixelShaderSaturate as usize + 1;
    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDFixedFunctionValue {
    Boolean(bool),
    PolygonMode(MaxwellThreeDPolygonMode),
    ShadeMode(MaxwellThreeDShadeMode),
    FrontFace(MaxwellThreeDFrontFace),
    CullFace(MaxwellThreeDCullFace),
    Compare(MaxwellThreeDCompareOp),
    StencilOp(MaxwellThreeDStencilOp),
    BlendOp(MaxwellThreeDBlendOp),
    BlendFactor(MaxwellThreeDBlendFactor),
    LogicOp(MaxwellThreeDLogicOp),
    ProvokingVertex(MaxwellThreeDProvokingVertex),
    PixelShaderSaturate(MaxwellThreeDPixelShaderSaturate),
    FloatBits(MaxwellThreeDRawValue),
    SampleMode(MaxwellThreeDSampleMode),
    Mask(u32),
    ViewportScaleOffsetEnable(MaxwellThreeDViewportScaleOffsetEnable),
    ClipControl(MaxwellThreeDViewportClipControl),
    WindowClipType(MaxwellThreeDWindowClipType),
    ClipIdTestEnable(MaxwellThreeDClipIdTestEnable),
    AlphaControl {
        alpha_to_coverage: bool,
        alpha_to_one: bool,
    },
}

/// One axis of the global surface clip, encoded as origin plus extent.
///
/// Unlike viewport, scissor, and window-clip rectangles, these registers use
/// an origin and size rather than minimum and maximum coordinates.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1386-L1392>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDSurfaceClipAxis {
    origin: u16,
    extent: u16,
}

impl MaxwellThreeDSurfaceClipAxis {
    pub(super) const fn parse(raw: u32) -> Self {
        Self {
            origin: raw as u16,
            extent: (raw >> 16) as u16,
        }
    }

    #[must_use]
    pub const fn origin(self) -> u16 {
        self.origin
    }

    #[must_use]
    pub const fn extent(self) -> u16 {
        self.extent
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDViewportTransformState {
    scale: [MaxwellThreeDRegister<MaxwellThreeDRawValue>; 3],
    offset: [MaxwellThreeDRegister<MaxwellThreeDRawValue>; 3],
    clip_horizontal: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    clip_vertical: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    clip_min_z: MaxwellThreeDRegister<MaxwellThreeDRawValue>,
    clip_max_z: MaxwellThreeDRegister<MaxwellThreeDRawValue>,
    coordinate_swizzle: MaxwellThreeDRegister<MaxwellThreeDViewportCoordinateSwizzle>,
}

/// One signed source component selected by viewport coordinate swizzling.
///
/// NVIDIA publishes the same three-bit selector for each output component in
/// its pinned public `MAXWELL_B` class header.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L799-L835>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDViewportSwizzleComponent {
    PositiveX = 0,
    NegativeX = 1,
    PositiveY = 2,
    NegativeY = 3,
    PositiveZ = 4,
    NegativeZ = 5,
    PositiveW = 6,
    NegativeW = 7,
}

impl MaxwellThreeDViewportSwizzleComponent {
    const fn parse(raw: u32) -> Self {
        match raw & 7 {
            0 => Self::PositiveX,
            1 => Self::NegativeX,
            2 => Self::PositiveY,
            3 => Self::NegativeY,
            4 => Self::PositiveZ,
            5 => Self::NegativeZ,
            6 => Self::PositiveW,
            _ => Self::NegativeW,
        }
    }
}

/// Signed component mapping applied before one viewport transform.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDViewportCoordinateSwizzle {
    components: [MaxwellThreeDViewportSwizzleComponent; 4],
}

impl MaxwellThreeDViewportCoordinateSwizzle {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x7777 != 0 {
            return None;
        }
        Some(Self {
            components: [
                MaxwellThreeDViewportSwizzleComponent::parse(raw),
                MaxwellThreeDViewportSwizzleComponent::parse(raw >> 4),
                MaxwellThreeDViewportSwizzleComponent::parse(raw >> 8),
                MaxwellThreeDViewportSwizzleComponent::parse(raw >> 12),
            ],
        })
    }

    #[must_use]
    pub const fn components(self) -> [MaxwellThreeDViewportSwizzleComponent; 4] {
        self.components
    }

    #[must_use]
    pub const fn is_identity(self) -> bool {
        matches!(
            self.components,
            [
                MaxwellThreeDViewportSwizzleComponent::PositiveX,
                MaxwellThreeDViewportSwizzleComponent::PositiveY,
                MaxwellThreeDViewportSwizzleComponent::PositiveZ,
                MaxwellThreeDViewportSwizzleComponent::PositiveW,
            ]
        )
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.components[0] as u32
            | ((self.components[1] as u32) << 4)
            | ((self.components[2] as u32) << 8)
            | ((self.components[3] as u32) << 12)
    }
}

impl MaxwellThreeDViewportTransformState {
    #[must_use]
    pub const fn scale(&self) -> &[MaxwellThreeDRegister<MaxwellThreeDRawValue>; 3] {
        &self.scale
    }
    #[must_use]
    pub const fn offset(&self) -> &[MaxwellThreeDRegister<MaxwellThreeDRawValue>; 3] {
        &self.offset
    }
    #[must_use]
    pub const fn clip_horizontal(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.clip_horizontal
    }
    #[must_use]
    pub const fn clip_vertical(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.clip_vertical
    }
    #[must_use]
    pub const fn clip_min_z(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        &self.clip_min_z
    }
    #[must_use]
    pub const fn clip_max_z(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        &self.clip_max_z
    }
    #[must_use]
    pub const fn coordinate_swizzle(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDViewportCoordinateSwizzle> {
        &self.coordinate_swizzle
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDScissorState {
    enable: MaxwellThreeDRegister<bool>,
    horizontal: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    vertical: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
}

/// One source-preserving horizontal/vertical window-clip rectangle pair.
///
/// The eight pairs and their packed 16-bit minimum/maximum fields are defined
/// by NVIDIA's pinned public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L855-L861>
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDWindowClipState {
    horizontal: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    vertical: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
}

impl MaxwellThreeDWindowClipState {
    #[must_use]
    pub const fn horizontal(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.horizontal
    }

    #[must_use]
    pub const fn vertical(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.vertical
    }
}
impl MaxwellThreeDScissorState {
    #[must_use]
    pub const fn enable(&self) -> &MaxwellThreeDRegister<bool> {
        &self.enable
    }
    #[must_use]
    pub const fn horizontal(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.horizontal
    }
    #[must_use]
    pub const fn vertical(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.vertical
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDFixedFunctionState {
    surface_clip_horizontal: MaxwellThreeDRegister<MaxwellThreeDSurfaceClipAxis>,
    surface_clip_vertical: MaxwellThreeDRegister<MaxwellThreeDSurfaceClipAxis>,
    viewport: [MaxwellThreeDViewportTransformState; MAXWELL_VIEWPORT_COUNT],
    scissor: [MaxwellThreeDScissorState; MAXWELL_SCISSOR_COUNT],
    window_clip: [MaxwellThreeDWindowClipState; MAXWELL_WINDOW_CLIP_COUNT],
    registers: [MaxwellThreeDRegister<MaxwellThreeDFixedFunctionValue>;
        MaxwellThreeDFixedFunctionRegister::COUNT],
    blend_enable_common: MaxwellThreeDRegister<MaxwellThreeDBlendEnableCommon>,
    blend_enable: [MaxwellThreeDRegister<bool>; 8],
    color_mask: [MaxwellThreeDRegister<MaxwellThreeDColorMask>; 8],
    per_target_blend: [[MaxwellThreeDRegister<MaxwellThreeDFixedFunctionValue>; 7]; 8],
    blend_controls: MaxwellThreeDBlendControlState,
}
impl Default for MaxwellThreeDFixedFunctionState {
    fn default() -> Self {
        let mut registers = std::array::from_fn(|_| Default::default());
        for register in [
            MaxwellThreeDFixedFunctionRegister::FrontPolygonMode,
            MaxwellThreeDFixedFunctionRegister::BackPolygonMode,
        ] {
            registers[register.index()] =
                MaxwellThreeDRegister::verified_reset(MAXWELL_THREE_D_POLYGON_MODE_RESET, None);
        }
        registers[MaxwellThreeDFixedFunctionRegister::WindowOrigin.index()] =
            MaxwellThreeDRegister::verified_reset(
                MAXWELL_THREE_D_WINDOW_ORIGIN_RESET,
                Some(MaxwellThreeDFixedFunctionValue::Mask(
                    MAXWELL_THREE_D_WINDOW_ORIGIN_RESET,
                )),
            );
        Self {
            surface_clip_horizontal: Default::default(),
            surface_clip_vertical: Default::default(),
            viewport: std::array::from_fn(|_| Default::default()),
            scissor: std::array::from_fn(|_| Default::default()),
            window_clip: std::array::from_fn(|_| Default::default()),
            registers,
            blend_enable_common: Default::default(),
            blend_enable: Default::default(),
            color_mask: Default::default(),
            per_target_blend: std::array::from_fn(|_| std::array::from_fn(|_| Default::default())),
            blend_controls: Default::default(),
        }
    }
}
impl MaxwellThreeDFixedFunctionState {
    #[must_use]
    pub const fn surface_clip_horizontal(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSurfaceClipAxis> {
        &self.surface_clip_horizontal
    }

    #[must_use]
    pub const fn surface_clip_vertical(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSurfaceClipAxis> {
        &self.surface_clip_vertical
    }

    #[must_use]
    pub const fn viewport(&self) -> &[MaxwellThreeDViewportTransformState; MAXWELL_VIEWPORT_COUNT] {
        &self.viewport
    }
    #[must_use]
    pub const fn scissor(&self) -> &[MaxwellThreeDScissorState; MAXWELL_SCISSOR_COUNT] {
        &self.scissor
    }
    #[must_use]
    pub const fn window_clip(&self) -> &[MaxwellThreeDWindowClipState; MAXWELL_WINDOW_CLIP_COUNT] {
        &self.window_clip
    }
    #[must_use]
    pub const fn register(
        &self,
        register: MaxwellThreeDFixedFunctionRegister,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDFixedFunctionValue> {
        &self.registers[register.index()]
    }
    #[must_use]
    pub const fn blend_enable(&self) -> &[MaxwellThreeDRegister<bool>; 8] {
        &self.blend_enable
    }
    #[must_use]
    pub const fn blend_enable_common(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDBlendEnableCommon> {
        &self.blend_enable_common
    }
    #[must_use]
    pub const fn color_mask(&self) -> &[MaxwellThreeDRegister<MaxwellThreeDColorMask>; 8] {
        &self.color_mask
    }
    #[must_use]
    pub const fn per_target_blend(
        &self,
    ) -> &[[MaxwellThreeDRegister<MaxwellThreeDFixedFunctionValue>; 7]; 8] {
        &self.per_target_blend
    }
    #[must_use]
    pub const fn blend_controls(&self) -> &MaxwellThreeDBlendControlState {
        &self.blend_controls
    }

    pub(super) fn append_pipeline_dependencies(
        &self,
        dependencies: &mut Vec<Option<u32>>,
        active_color_targets: &[u8],
    ) {
        dependencies.push(self.surface_clip_horizontal.raw());
        dependencies.push(self.surface_clip_vertical.raw());
        let viewport_scale_offset_enable =
            self.register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable);
        let scale_offset_may_be_effective = viewport_scale_offset_enable.value()
            != Some(&MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(
                MaxwellThreeDViewportScaleOffsetEnable::Disabled,
            ));
        for viewport in &self.viewport {
            if scale_offset_may_be_effective {
                dependencies.extend(viewport.scale.iter().map(MaxwellThreeDRegister::raw));
                dependencies.extend(viewport.offset.iter().map(MaxwellThreeDRegister::raw));
            }
            dependencies.push(viewport.clip_horizontal.raw());
            dependencies.push(viewport.clip_vertical.raw());
            dependencies.push(viewport.clip_min_z.raw());
            dependencies.push(viewport.clip_max_z.raw());
            dependencies.push(viewport.coordinate_swizzle.raw());
        }
        for scissor in &self.scissor {
            dependencies.push(scissor.enable.raw());
            dependencies.push(scissor.horizontal.raw());
            dependencies.push(scissor.vertical.raw());
        }
        // Successful lowering currently requires effective blending to be
        // disabled. Retain only state that selects that effective result;
        // inactive equation families and unselected targets cannot affect the
        // pipeline and therefore must not invalidate its identity.
        let conditionally_effective_registers = [
            MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable,
            MaxwellThreeDFixedFunctionRegister::BlendSeparateAlpha,
            MaxwellThreeDFixedFunctionRegister::BlendColorOp,
            MaxwellThreeDFixedFunctionRegister::BlendColorSource,
            MaxwellThreeDFixedFunctionRegister::BlendColorDestination,
            MaxwellThreeDFixedFunctionRegister::BlendAlphaOp,
            MaxwellThreeDFixedFunctionRegister::BlendAlphaSource,
            MaxwellThreeDFixedFunctionRegister::BlendAlphaDestination,
            MaxwellThreeDFixedFunctionRegister::BlendConstantRed,
            MaxwellThreeDFixedFunctionRegister::BlendConstantGreen,
            MaxwellThreeDFixedFunctionRegister::BlendConstantBlue,
            MaxwellThreeDFixedFunctionRegister::BlendConstantAlpha,
            MaxwellThreeDFixedFunctionRegister::WindowClipType,
            MaxwellThreeDFixedFunctionRegister::LogicOpFunction,
            MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable,
            MaxwellThreeDFixedFunctionRegister::FrontStencilFail,
            MaxwellThreeDFixedFunctionRegister::FrontStencilDepthFail,
            MaxwellThreeDFixedFunctionRegister::FrontStencilPass,
            MaxwellThreeDFixedFunctionRegister::FrontStencilCompare,
            MaxwellThreeDFixedFunctionRegister::FrontStencilReference,
            MaxwellThreeDFixedFunctionRegister::FrontStencilCompareMask,
            MaxwellThreeDFixedFunctionRegister::FrontStencilWriteMask,
            MaxwellThreeDFixedFunctionRegister::BackStencilFail,
            MaxwellThreeDFixedFunctionRegister::BackStencilDepthFail,
            MaxwellThreeDFixedFunctionRegister::BackStencilPass,
            MaxwellThreeDFixedFunctionRegister::BackStencilCompare,
            MaxwellThreeDFixedFunctionRegister::BackStencilReference,
            MaxwellThreeDFixedFunctionRegister::BackStencilCompareMask,
            MaxwellThreeDFixedFunctionRegister::BackStencilWriteMask,
        ];
        dependencies.extend(
            self.registers
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    !conditionally_effective_registers
                        .iter()
                        .any(|register| register.index() == *index)
                })
                .map(|(_, register)| register.raw()),
        );
        let stencil_test_enable =
            self.register(MaxwellThreeDFixedFunctionRegister::StencilTestEnable);
        if stencil_test_enable.value() != Some(&MaxwellThreeDFixedFunctionValue::Boolean(false)) {
            for register in [
                MaxwellThreeDFixedFunctionRegister::FrontStencilFail,
                MaxwellThreeDFixedFunctionRegister::FrontStencilDepthFail,
                MaxwellThreeDFixedFunctionRegister::FrontStencilPass,
                MaxwellThreeDFixedFunctionRegister::FrontStencilCompare,
                MaxwellThreeDFixedFunctionRegister::FrontStencilReference,
                MaxwellThreeDFixedFunctionRegister::FrontStencilCompareMask,
                MaxwellThreeDFixedFunctionRegister::FrontStencilWriteMask,
            ] {
                dependencies.push(self.register(register).raw());
            }
            let two_sided =
                self.register(MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable);
            dependencies.push(two_sided.raw());
            if two_sided.value() != Some(&MaxwellThreeDFixedFunctionValue::Boolean(false)) {
                for register in [
                    MaxwellThreeDFixedFunctionRegister::BackStencilFail,
                    MaxwellThreeDFixedFunctionRegister::BackStencilDepthFail,
                    MaxwellThreeDFixedFunctionRegister::BackStencilPass,
                    MaxwellThreeDFixedFunctionRegister::BackStencilCompare,
                    MaxwellThreeDFixedFunctionRegister::BackStencilReference,
                    MaxwellThreeDFixedFunctionRegister::BackStencilCompareMask,
                    MaxwellThreeDFixedFunctionRegister::BackStencilWriteMask,
                ] {
                    dependencies.push(self.register(register).raw());
                }
            }
        }
        let window_clip_enable =
            self.register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable);
        if window_clip_enable.value() != Some(&MaxwellThreeDFixedFunctionValue::Boolean(false)) {
            dependencies.push(
                self.register(MaxwellThreeDFixedFunctionRegister::WindowClipType)
                    .raw(),
            );
            for region in &self.window_clip {
                dependencies.push(region.horizontal.raw());
                dependencies.push(region.vertical.raw());
            }
        }
        let mut blending_enabled = false;
        if !active_color_targets.is_empty() {
            let selection = self.register(MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable);
            dependencies.push(selection.raw());
            if selection.value() == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true)) {
                dependencies.extend(
                    active_color_targets
                        .iter()
                        .map(|target| self.blend_enable[*target as usize].raw()),
                );
                blending_enabled = active_color_targets
                    .iter()
                    .any(|target| self.blend_enable[*target as usize].value() == Some(&true));
            } else {
                dependencies.push(self.blend_enable_common.raw());
                blending_enabled = self.blend_enable_common.value()
                    == Some(&MaxwellThreeDBlendEnableCommon::Enabled);
            }
        }
        if blending_enabled {
            dependencies.push(self.blend_controls.per_format_enable.raw());
            dependencies.push(self.blend_controls.zero_times_anything_is_zero.raw());
        }
        if !active_color_targets.is_empty()
            && self
                .blend_controls
                .iterated_blend
                .value()
                .is_some_and(|value| value.enabled())
        {
            dependencies.push(self.blend_controls.iterated_blend.raw());
            dependencies.push(self.blend_controls.iterated_blend_pass_count.raw());
        }
        if self
            .register(MaxwellThreeDFixedFunctionRegister::LogicOpEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
        {
            dependencies.push(
                self.register(MaxwellThreeDFixedFunctionRegister::LogicOpFunction)
                    .raw(),
            );
        }
        if self
            .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
        {
            dependencies.push(
                self.register(MaxwellThreeDFixedFunctionRegister::AlphaTestReference)
                    .raw(),
            );
            dependencies.push(
                self.register(MaxwellThreeDFixedFunctionRegister::AlphaTestFunction)
                    .raw(),
            );
        }
        match self
            .register(MaxwellThreeDFixedFunctionRegister::SingleColorTargetWriteControl)
            .value()
        {
            Some(MaxwellThreeDFixedFunctionValue::Boolean(true)) => {
                dependencies.push(self.color_mask[0].raw());
            }
            Some(MaxwellThreeDFixedFunctionValue::Boolean(false)) => {
                dependencies.extend(
                    active_color_targets
                        .iter()
                        .map(|target| self.color_mask[*target as usize].raw()),
                );
            }
            _ => dependencies.extend(self.color_mask.iter().map(MaxwellThreeDRegister::raw)),
        }
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDFixedFunctionWrite) {
        let source = write.source();
        let raw = source.argument();
        match write {
            MaxwellThreeDFixedFunctionWrite::SurfaceClip {
                vertical, value, ..
            } => {
                let register = if vertical {
                    &mut self.surface_clip_vertical
                } else {
                    &mut self.surface_clip_horizontal
                };
                *register = MaxwellThreeDRegister::programmed(raw, value, source);
            }
            MaxwellThreeDFixedFunctionWrite::ViewportFloat {
                viewport,
                field,
                value,
                ..
            } => {
                let target = &mut self.viewport[viewport as usize];
                match field {
                    0..=2 => {
                        target.scale[field as usize] =
                            MaxwellThreeDRegister::programmed(raw, value, source)
                    }
                    3..=5 => {
                        target.offset[(field - 3) as usize] =
                            MaxwellThreeDRegister::programmed(raw, value, source)
                    }
                    _ => unreachable!(),
                }
            }
            MaxwellThreeDFixedFunctionWrite::ViewportRectangle {
                viewport,
                vertical,
                value,
                ..
            } => {
                if vertical {
                    self.viewport[viewport as usize].clip_vertical =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                } else {
                    self.viewport[viewport as usize].clip_horizontal =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                }
            }
            MaxwellThreeDFixedFunctionWrite::ViewportDepth {
                viewport,
                maximum,
                value,
                ..
            } => {
                if maximum {
                    self.viewport[viewport as usize].clip_max_z =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                } else {
                    self.viewport[viewport as usize].clip_min_z =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                }
            }
            MaxwellThreeDFixedFunctionWrite::ViewportCoordinateSwizzle {
                viewport, value, ..
            } => {
                self.viewport[viewport as usize].coordinate_swizzle =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::ScissorEnable { scissor, value, .. } => {
                self.scissor[scissor as usize].enable =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::ScissorRectangle {
                scissor,
                vertical,
                value,
                ..
            } => {
                if vertical {
                    self.scissor[scissor as usize].vertical =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                } else {
                    self.scissor[scissor as usize].horizontal =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                }
            }
            MaxwellThreeDFixedFunctionWrite::WindowClipRectangle {
                region,
                vertical,
                value,
                ..
            } => {
                if vertical {
                    self.window_clip[region as usize].vertical =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                } else {
                    self.window_clip[region as usize].horizontal =
                        MaxwellThreeDRegister::programmed(raw, value, source)
                }
            }
            MaxwellThreeDFixedFunctionWrite::Register {
                register, value, ..
            } => {
                self.registers[register.index()] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::BlendEnableCommon { value, .. } => {
                self.blend_enable_common = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::BlendEnable { target, value, .. } => {
                self.blend_enable[target as usize] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::ColorMask { target, value, .. } => {
                self.color_mask[target as usize] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::BlendState {
                target,
                field,
                value,
                ..
            } => {
                self.per_target_blend[target as usize][field as usize] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::BlendPerFormatEnable { value, .. } => {
                self.blend_controls.per_format_enable =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::BlendFloatPixelKillEnable { value, .. } => {
                self.blend_controls.float_pixel_kill_enable =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::BlendZeroTimesAnythingIsZero { value, .. } => {
                self.blend_controls.zero_times_anything_is_zero =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::IteratedBlend { value, .. } => {
                self.blend_controls.iterated_blend =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDFixedFunctionWrite::IteratedBlendPassCount { value, .. } => {
                self.blend_controls.iterated_blend_pass_count =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDFixedFunctionWrite {
    SurfaceClip {
        vertical: bool,
        value: MaxwellThreeDSurfaceClipAxis,
        source: MaxwellMethodSource,
    },
    ViewportFloat {
        viewport: u8,
        field: u8,
        value: MaxwellThreeDRawValue,
        source: MaxwellMethodSource,
    },
    ViewportRectangle {
        viewport: u8,
        vertical: bool,
        value: MaxwellThreeDRectangle,
        source: MaxwellMethodSource,
    },
    ViewportDepth {
        viewport: u8,
        maximum: bool,
        value: MaxwellThreeDRawValue,
        source: MaxwellMethodSource,
    },
    ViewportCoordinateSwizzle {
        viewport: u8,
        value: MaxwellThreeDViewportCoordinateSwizzle,
        source: MaxwellMethodSource,
    },
    ScissorEnable {
        scissor: u8,
        value: bool,
        source: MaxwellMethodSource,
    },
    ScissorRectangle {
        scissor: u8,
        vertical: bool,
        value: MaxwellThreeDRectangle,
        source: MaxwellMethodSource,
    },
    WindowClipRectangle {
        region: u8,
        vertical: bool,
        value: MaxwellThreeDRectangle,
        source: MaxwellMethodSource,
    },
    Register {
        register: MaxwellThreeDFixedFunctionRegister,
        value: MaxwellThreeDFixedFunctionValue,
        source: MaxwellMethodSource,
    },
    BlendEnableCommon {
        value: MaxwellThreeDBlendEnableCommon,
        source: MaxwellMethodSource,
    },
    BlendEnable {
        target: u8,
        value: bool,
        source: MaxwellMethodSource,
    },
    ColorMask {
        target: u8,
        value: MaxwellThreeDColorMask,
        source: MaxwellMethodSource,
    },
    BlendState {
        target: u8,
        field: u8,
        value: MaxwellThreeDFixedFunctionValue,
        source: MaxwellMethodSource,
    },
    BlendPerFormatEnable {
        value: MaxwellThreeDBlendPerFormatEnable,
        source: MaxwellMethodSource,
    },
    BlendFloatPixelKillEnable {
        value: MaxwellThreeDBlendFloatPixelKillEnable,
        source: MaxwellMethodSource,
    },
    BlendZeroTimesAnythingIsZero {
        value: MaxwellThreeDBlendZeroTimesAnythingIsZero,
        source: MaxwellMethodSource,
    },
    IteratedBlend {
        value: MaxwellThreeDIteratedBlend,
        source: MaxwellMethodSource,
    },
    IteratedBlendPassCount {
        value: MaxwellThreeDIteratedBlendPassCount,
        source: MaxwellMethodSource,
    },
}
impl MaxwellThreeDFixedFunctionWrite {
    pub(super) const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::SurfaceClip { source, .. }
            | Self::ViewportFloat { source, .. }
            | Self::ViewportRectangle { source, .. }
            | Self::ViewportDepth { source, .. }
            | Self::ViewportCoordinateSwizzle { source, .. }
            | Self::ScissorEnable { source, .. }
            | Self::ScissorRectangle { source, .. }
            | Self::WindowClipRectangle { source, .. }
            | Self::Register { source, .. }
            | Self::BlendEnableCommon { source, .. }
            | Self::BlendEnable { source, .. }
            | Self::ColorMask { source, .. }
            | Self::BlendState { source, .. }
            | Self::BlendPerFormatEnable { source, .. }
            | Self::BlendFloatPixelKillEnable { source, .. }
            | Self::BlendZeroTimesAnythingIsZero { source, .. }
            | Self::IteratedBlend { source, .. }
            | Self::IteratedBlendPassCount { source, .. } => source,
        }
    }
}
