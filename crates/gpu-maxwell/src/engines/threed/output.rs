//! Typed fixed-function output state for `MAXWELL_B`.

use crate::MaxwellMethodSource;

use super::render_targets::{MaxwellThreeDRawValue, MaxwellThreeDRectangle};
use super::state::{MAXWELL_THREE_D_POLYGON_MODE_RESET, MaxwellThreeDRegister};

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
}

impl MaxwellThreeDFixedFunctionRegister {
    const COUNT: usize = Self::DepthBoundsMax as usize + 1;
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDViewportTransformState {
    scale: [MaxwellThreeDRegister<MaxwellThreeDRawValue>; 3],
    offset: [MaxwellThreeDRegister<MaxwellThreeDRawValue>; 3],
    clip_horizontal: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    clip_vertical: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    clip_min_z: MaxwellThreeDRegister<MaxwellThreeDRawValue>,
    clip_max_z: MaxwellThreeDRegister<MaxwellThreeDRawValue>,
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
        Self {
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
        dependencies.extend(self.color_mask.iter().map(MaxwellThreeDRegister::raw));
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDFixedFunctionWrite) {
        let source = write.source();
        let raw = source.argument();
        match write {
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
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDFixedFunctionWrite {
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
}
impl MaxwellThreeDFixedFunctionWrite {
    pub(super) const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::ViewportFloat { source, .. }
            | Self::ViewportRectangle { source, .. }
            | Self::ViewportDepth { source, .. }
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
            | Self::BlendZeroTimesAnythingIsZero { source, .. } => source,
        }
    }
}
