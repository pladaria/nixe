//! Typed fixed-function output state for `MAXWELL_B`.

use super::render_targets::{MaxwellThreeDRawValue, MaxwellThreeDRectangle};
use super::state::MaxwellThreeDRegister;

pub const MAXWELL_VIEWPORT_COUNT: usize = 16;
pub const MAXWELL_SCISSOR_COUNT: usize = 16;
pub const MAXWELL_WINDOW_CLIP_COUNT: usize = 8;

pub(super) const BLEND_SEPARATE_ALPHA_BASE: u32 = 0x1e00;
pub(super) const BLEND_TARGET_STRIDE: u32 = 0x20;
/// Initial channel context, not a draw-time fallback. NVIDIA's GM20B method
/// initialization (shared with GM200) sets the common selector (0x133c) and
/// all eight per-target B197 selectors (0x1e00 + i*0x20) to one.
/// The installed table's SHA-256 is
/// 6372e2f6f547d7bd086beb066ce46ac379795f95c38eeeedd9e13c7b32515b24.
/// No firmware bytes are embedded here. Nouveau's public Maxwell context table
/// independently documents the same eight writes:
/// https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/nouveau/nvkm/engine/gr/ctxgm107.c#L360-L366
/// GM20B table provenance and address/class decoding:
/// https://gitlab.com/kernel-firmware/linux-firmware/-/blob/main/WHENCE
/// https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/nouveau/nvkm/engine/gr/gk20a.c#L110-L149
/// Encoding:
/// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3309-L3312
pub(super) const BLEND_SEPARATE_ALPHA_RESET: u32 = 1;

/// Initial channel depth-clip interval for each of the sixteen viewports.
/// GM200's public method initialization writes 0x0c08 + i*0x10 = 0 and
/// 0x0c0c + i*0x10 = 0x3f800000. The table hash and decoding references above
/// apply to these entries too; no firmware contents are embedded here.
/// https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/310c1f50beb77fc5c6f9075029973161d4e51a4a/src/video_core/engines/maxwell_3d.cpp#L51-L57
pub(super) const VIEWPORT_CLIP_MIN_Z_RESET: u32 = 0;
pub(super) const VIEWPORT_CLIP_MAX_Z_RESET: u32 = 0x3f80_0000;

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

/// Whether `SET_BLEND(0)` supplies the blend enable for every color target.
/// Disabled selects each physical target's own `SET_BLEND(i)` instead. This
/// selector is independent of common versus per-target blend equations.
///
/// NVIDIA's `SET_SINGLE_ROP_CONTROL` (0x135c):
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDSingleRopControl {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDSingleRopControl {
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDBlendControlState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDBlendControlState<'_> {
    #[must_use]
    pub fn per_format_enable(&self) -> MaxwellThreeDRegister<MaxwellThreeDBlendPerFormatEnable> {
        self.registers
            .read(0x1140, MaxwellThreeDBlendPerFormatEnable::parse)
    }
    #[must_use]
    pub fn float_pixel_kill_enable(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDBlendFloatPixelKillEnable> {
        self.registers.read(0x0fdc, |raw| {
            MaxwellThreeDBlendFloatPixelKillEnable::parse(raw)
        })
    }
    #[must_use]
    pub fn zero_times_anything_is_zero(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDBlendZeroTimesAnythingIsZero> {
        self.registers.read(0x19c0, |raw| {
            MaxwellThreeDBlendZeroTimesAnythingIsZero::parse(raw)
        })
    }
    #[must_use]
    pub fn iterated_blend(&self) -> MaxwellThreeDRegister<MaxwellThreeDIteratedBlend> {
        self.registers
            .read(0x0dd0, MaxwellThreeDIteratedBlend::parse)
    }
    #[must_use]
    pub fn iterated_blend_pass_count(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDIteratedBlendPassCount> {
        self.registers.read(0x0dd4, |raw| {
            (raw <= 0xff).then_some(MaxwellThreeDIteratedBlendPassCount::new(raw as u8))
        })
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
    pub(in crate::engines) fn parse(raw: u32) -> Option<Self> {
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
    SlopeScaleDepthBias,
    DepthBias,
    DepthBiasClamp,
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

/// One axis of a surface or viewport clip, encoded as origin plus extent.
///
/// Unlike scissor and window-clip rectangles, these registers use
/// an origin and size rather than minimum and maximum coordinates.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1386-L1392>
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L841-L847>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDClipAxis {
    origin: u16,
    extent: u16,
}

impl MaxwellThreeDClipAxis {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDViewportTransformState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    index: usize,
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

impl MaxwellThreeDViewportTransformState<'_> {
    #[must_use]
    pub fn scale(&self, axis: usize) -> MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        self.registers.read(
            (0x0a00 + self.index as u32 * 0x20) + axis as u32 * 4,
            |raw| Some(MaxwellThreeDRawValue::new(raw)),
        )
    }
    #[must_use]
    pub fn offset(&self, axis: usize) -> MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        self.registers.read(
            0x0a00 + self.index as u32 * 0x20 + 12 + axis as u32 * 4,
            |raw| Some(MaxwellThreeDRawValue::new(raw)),
        )
    }
    #[must_use]
    pub fn clip_horizontal(&self) -> MaxwellThreeDRegister<MaxwellThreeDClipAxis> {
        self.registers
            .read(0x0c00 + self.index as u32 * 0x10, |raw| {
                Some(MaxwellThreeDClipAxis::parse(raw))
            })
    }
    #[must_use]
    pub fn clip_vertical(&self) -> MaxwellThreeDRegister<MaxwellThreeDClipAxis> {
        self.registers
            .read(0x0c00 + self.index as u32 * 0x10 + 4, |raw| {
                Some(MaxwellThreeDClipAxis::parse(raw))
            })
    }
    #[must_use]
    pub fn clip_min_z(&self) -> MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        self.registers
            .read(0x0c00 + self.index as u32 * 0x10 + 8, |raw| {
                Some(MaxwellThreeDRawValue::new(raw))
            })
    }
    #[must_use]
    pub fn clip_max_z(&self) -> MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        self.registers
            .read(0x0c00 + self.index as u32 * 0x10 + 12, |raw| {
                Some(MaxwellThreeDRawValue::new(raw))
            })
    }
    #[must_use]
    pub fn coordinate_swizzle(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDViewportCoordinateSwizzle> {
        self.registers
            .read(0x0a18 + self.index as u32 * 0x20, |raw| {
                MaxwellThreeDViewportCoordinateSwizzle::parse(raw)
            })
    }
    #[must_use]
    pub fn snap_grid_precision(&self) -> MaxwellThreeDRegister<[u8; 2]> {
        self.registers
            .read(0x0a1c + self.index as u32 * 0x20, |raw| {
                (raw & !0x1f1f == 0).then_some([(raw & 31) as u8, ((raw >> 8) & 31) as u8])
            })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDScissorState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    index: usize,
}
impl MaxwellThreeDScissorState<'_> {
    #[must_use]
    pub fn enable(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x0e00 + self.index as u32 * 0x10, |raw| {
                (raw <= 1).then_some(raw != 0)
            })
    }
    #[must_use]
    pub fn horizontal(&self) -> MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        self.registers
            .read(0x0e00 + self.index as u32 * 0x10 + 4, |raw| {
                super::rectangle(raw)
            })
    }
    #[must_use]
    pub fn vertical(&self) -> MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        self.registers
            .read(0x0e00 + self.index as u32 * 0x10 + 8, |raw| {
                super::rectangle(raw)
            })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDWindowClipState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    index: usize,
}
impl MaxwellThreeDWindowClipState<'_> {
    #[must_use]
    pub fn horizontal(&self) -> MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        self.registers
            .read(0x0d00 + self.index as u32 * 8, super::rectangle)
    }
    #[must_use]
    pub fn vertical(&self) -> MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        self.registers
            .read(0x0d00 + self.index as u32 * 8 + 4, |raw| {
                super::rectangle(raw)
            })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDFixedFunctionState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl<'a> MaxwellThreeDFixedFunctionState<'a> {
    #[must_use]
    pub fn surface_clip_horizontal(&self) -> MaxwellThreeDRegister<MaxwellThreeDClipAxis> {
        self.registers
            .read(0x0ff4, |raw| Some(MaxwellThreeDClipAxis::parse(raw)))
    }
    #[must_use]
    pub fn surface_clip_vertical(&self) -> MaxwellThreeDRegister<MaxwellThreeDClipAxis> {
        self.registers
            .read(0x0ff8, |raw| Some(MaxwellThreeDClipAxis::parse(raw)))
    }
    #[must_use]
    pub fn viewport(&self, index: usize) -> MaxwellThreeDViewportTransformState<'a> {
        MaxwellThreeDViewportTransformState {
            registers: self.registers,
            index,
        }
    }
    #[must_use]
    pub fn scissor(&self, index: usize) -> MaxwellThreeDScissorState<'a> {
        MaxwellThreeDScissorState {
            registers: self.registers,
            index,
        }
    }
    #[must_use]
    pub fn window_clip(&self, index: usize) -> MaxwellThreeDWindowClipState<'a> {
        MaxwellThreeDWindowClipState {
            registers: self.registers,
            index,
        }
    }
    #[must_use]
    pub fn register(
        &self,
        register: MaxwellThreeDFixedFunctionRegister,
    ) -> MaxwellThreeDRegister<MaxwellThreeDFixedFunctionValue> {
        use MaxwellThreeDFixedFunctionRegister as R;
        use MaxwellThreeDFixedFunctionValue as V;
        match register {
            R::RasterEnable => self
                .registers
                .read(0x037c, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::FrontPolygonMode => self.registers.read(0x0dac, |raw| {
                MaxwellThreeDPolygonMode::parse(raw).map(V::PolygonMode)
            }),
            R::BackPolygonMode => self.registers.read(0x0db0, |raw| {
                MaxwellThreeDPolygonMode::parse(raw).map(V::PolygonMode)
            }),
            R::PolygonOffsetPointEnable => self
                .registers
                .read(0x0dc0, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::PolygonOffsetLineEnable => self
                .registers
                .read(0x0dc4, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::PolygonOffsetFillEnable => self
                .registers
                .read(0x0dc8, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::WindowOffsetX => self
                .registers
                .read(0x0df8, |raw| (raw & !0x1ffff == 0).then_some(V::Mask(raw))),
            R::WindowOffsetY => self
                .registers
                .read(0x0dfc, |raw| (raw & !0x3ffff == 0).then_some(V::Mask(raw))),
            R::BackStencilReference => self
                .registers
                .read(0x0f54, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::BackStencilWriteMask => self
                .registers
                .read(0x0f58, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::BackStencilCompareMask => self
                .registers
                .read(0x0f5c, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::SingleColorTargetWriteControl => self
                .registers
                .read(0x0f90, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::DepthBoundsMin => self.registers.read(0x0f9c, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::DepthBoundsMax => self.registers.read(0x0fa0, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::SampleMaskControl => self
                .registers
                .read(0x0fa4, |raw| (raw & !0x11 == 0).then_some(V::Mask(raw))),
            R::RasterSampleMode => self.registers.read(0x0fb8, |raw| {
                MaxwellThreeDSampleMode::parse(raw)
                    .map(V::SampleMode)
                    .filter(|value| {
                        matches!(
                            value,
                            V::SampleMode(
                                MaxwellThreeDSampleMode::Samples1x1
                                    | MaxwellThreeDSampleMode::Samples2x2
                                    | MaxwellThreeDSampleMode::Samples4x2D3D
                                    | MaxwellThreeDSampleMode::Samples2x1D3D
                                    | MaxwellThreeDSampleMode::Samples4x4
                            )
                        )
                    })
            }),
            R::SampleMask0 => self
                .registers
                .read(0x0fbc, |raw| (raw & !0xffff == 0).then_some(V::Mask(raw))),
            R::SampleMask1 => self
                .registers
                .read(0x0fc0, |raw| (raw & !0xffff == 0).then_some(V::Mask(raw))),
            R::SampleMask2 => self
                .registers
                .read(0x0fc4, |raw| (raw & !0xffff == 0).then_some(V::Mask(raw))),
            R::SampleMask3 => self
                .registers
                .read(0x0fc8, |raw| (raw & !0xffff == 0).then_some(V::Mask(raw))),
            R::DepthTestEnable => self
                .registers
                .read(0x12cc, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::ShadeMode => self.registers.read(0x12d4, |raw| {
                MaxwellThreeDShadeMode::parse(raw).map(V::ShadeMode)
            }),
            R::BlendPerTargetEnable => self
                .registers
                .read(0x12e4, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::DepthWriteEnable => self
                .registers
                .read(0x12e8, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::AlphaTestEnable => self
                .registers
                .read(0x12ec, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::DepthCompare => self.registers.read(0x130c, |raw| {
                MaxwellThreeDCompareOp::parse(raw).map(V::Compare)
            }),
            R::AlphaTestReference => self.registers.read(0x1310, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::AlphaTestFunction => self.registers.read(0x1314, |raw| {
                MaxwellThreeDCompareOp::parse(raw).map(V::Compare)
            }),
            R::BlendConstantRed => self.registers.read(0x131c, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::BlendConstantGreen => self.registers.read(0x1320, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::BlendConstantBlue => self.registers.read(0x1324, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::BlendConstantAlpha => self.registers.read(0x1328, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::BlendSeparateAlpha => self
                .registers
                .read(0x133c, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::BlendColorOp => self.registers.read(0x1340, |raw| {
                MaxwellThreeDBlendOp::parse(raw).map(V::BlendOp)
            }),
            R::BlendColorSource => self.registers.read(0x1344, |raw| {
                MaxwellThreeDBlendFactor::parse(raw).map(V::BlendFactor)
            }),
            R::BlendColorDestination => self.registers.read(0x1348, |raw| {
                MaxwellThreeDBlendFactor::parse(raw).map(V::BlendFactor)
            }),
            R::BlendAlphaOp => self.registers.read(0x134c, |raw| {
                MaxwellThreeDBlendOp::parse(raw).map(V::BlendOp)
            }),
            R::BlendAlphaSource => self.registers.read(0x1350, |raw| {
                MaxwellThreeDBlendFactor::parse(raw).map(V::BlendFactor)
            }),
            R::BlendAlphaDestination => self.registers.read(0x1358, |raw| {
                MaxwellThreeDBlendFactor::parse(raw).map(V::BlendFactor)
            }),
            R::StencilTestEnable => self
                .registers
                .read(0x1380, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::FrontStencilFail => self.registers.read(0x1384, |raw| {
                MaxwellThreeDStencilOp::parse(raw).map(V::StencilOp)
            }),
            R::FrontStencilDepthFail => self.registers.read(0x1388, |raw| {
                MaxwellThreeDStencilOp::parse(raw).map(V::StencilOp)
            }),
            R::FrontStencilPass => self.registers.read(0x138c, |raw| {
                MaxwellThreeDStencilOp::parse(raw).map(V::StencilOp)
            }),
            R::FrontStencilCompare => self.registers.read(0x1390, |raw| {
                MaxwellThreeDCompareOp::parse(raw).map(V::Compare)
            }),
            R::FrontStencilReference => self
                .registers
                .read(0x1394, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::FrontStencilCompareMask => self
                .registers
                .read(0x1398, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::FrontStencilWriteMask => self
                .registers
                .read(0x139c, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::PixelShaderSaturate => self.registers.read(0x13a8, |raw| {
                MaxwellThreeDPixelShaderSaturate::parse(raw).map(V::PixelShaderSaturate)
            }),
            R::WindowOrigin => self
                .registers
                .read(0x13ac, |raw| (raw & !0x11 == 0).then_some(V::Mask(raw))),
            R::LineWidth => self.registers.read(0x13b0, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::UserClipEnable => self
                .registers
                .read(0x1510, |raw| (raw & !0xff == 0).then_some(V::Mask(raw))),
            R::AntiAliasEnable => self
                .registers
                .read(0x1534, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::SlopeScaleDepthBias => self.registers.read(0x156c, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::TwoSidedStencilTestEnable => self
                .registers
                .read(0x1594, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::BackStencilFail => self.registers.read(0x1598, |raw| {
                MaxwellThreeDStencilOp::parse(raw).map(V::StencilOp)
            }),
            R::BackStencilDepthFail => self.registers.read(0x159c, |raw| {
                MaxwellThreeDStencilOp::parse(raw).map(V::StencilOp)
            }),
            R::BackStencilPass => self.registers.read(0x15a0, |raw| {
                MaxwellThreeDStencilOp::parse(raw).map(V::StencilOp)
            }),
            R::BackStencilCompare => self.registers.read(0x15a4, |raw| {
                MaxwellThreeDCompareOp::parse(raw).map(V::Compare)
            }),
            R::DepthBias => self.registers.read(0x15bc, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::SampleMode => self.registers.read(0x15d0, |raw| {
                MaxwellThreeDSampleMode::parse(raw).map(V::SampleMode)
            }),
            R::ProvokingVertex => self.registers.read(0x1684, |raw| {
                MaxwellThreeDProvokingVertex::parse(raw).map(V::ProvokingVertex)
            }),
            R::TwoSidedLightEnable => self
                .registers
                .read(0x1688, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::DepthBiasClamp => self.registers.read(0x187c, |raw| {
                Some(V::FloatBits(MaxwellThreeDRawValue::new(raw)))
            }),
            R::CullEnable => self
                .registers
                .read(0x1918, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::FrontFace => self.registers.read(0x191c, |raw| {
                MaxwellThreeDFrontFace::parse(raw).map(V::FrontFace)
            }),
            R::CullFace => self.registers.read(0x1920, |raw| {
                MaxwellThreeDCullFace::parse(raw).map(V::CullFace)
            }),
            R::ViewportScaleOffsetEnable => self.registers.read(0x192c, |raw| {
                MaxwellThreeDViewportScaleOffsetEnable::parse(raw).map(V::ViewportScaleOffsetEnable)
            }),
            R::ViewportClipControl => self.registers.read(0x193c, |raw| {
                MaxwellThreeDViewportClipControl::parse(raw).map(V::ClipControl)
            }),
            R::UserClipOperation => self.registers.read(0x1940, |raw| {
                (raw & !0x11111111 == 0).then_some(V::Mask(raw))
            }),
            R::WindowClipEnable => self
                .registers
                .read(0x194c, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::WindowClipType => self.registers.read(0x1950, |raw| {
                MaxwellThreeDWindowClipType::parse(raw).map(V::WindowClipType)
            }),
            R::ClipIdTestEnable => self.registers.read(0x197c, |raw| {
                MaxwellThreeDClipIdTestEnable::parse(raw).map(V::ClipIdTestEnable)
            }),
            R::DepthBoundsEnable => self
                .registers
                .read(0x19bc, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::LogicOpEnable => self
                .registers
                .read(0x19c4, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::LogicOpFunction => self.registers.read(0x19c8, |raw| {
                MaxwellThreeDLogicOp::parse(raw).map(V::LogicOp)
            }),
            R::ColorClampEnable => self
                .registers
                .read(0x2600, |raw| (raw <= 1).then_some(V::Boolean(raw != 0))),
            R::AlphaToCoverageEnable => self.registers.read(0x153c, |raw| {
                (raw & !0x11 == 0).then_some(V::AlphaControl {
                    alpha_to_coverage: raw & 1 != 0,
                    alpha_to_one: raw & 0x10 != 0,
                })
            }),
        }
    }
    #[must_use]
    pub fn blend_enable(&self, target: usize) -> MaxwellThreeDRegister<bool> {
        self.registers.read(0x1360 + target as u32 * 4, |raw| {
            (raw <= 1).then_some(raw != 0)
        })
    }
    #[must_use]
    pub fn color_mask(&self, target: usize) -> MaxwellThreeDRegister<MaxwellThreeDColorMask> {
        self.registers.read(0x1a00 + target as u32 * 4, |raw| {
            MaxwellThreeDColorMask::parse(raw)
        })
    }
    #[must_use]
    pub fn single_rop_control(&self) -> MaxwellThreeDRegister<MaxwellThreeDSingleRopControl> {
        self.registers
            .read(0x135c, MaxwellThreeDSingleRopControl::parse)
    }
    #[must_use]
    pub fn per_target_blend(
        &self,
        target: usize,
        field: usize,
    ) -> MaxwellThreeDRegister<MaxwellThreeDFixedFunctionValue> {
        use MaxwellThreeDFixedFunctionValue as V;
        self.registers.read(
            0x1e00 + target as u32 * 0x20 + field as u32 * 4,
            |raw| match field {
                0 => (raw <= 1).then_some(V::Boolean(raw != 0)),
                1 | 4 => MaxwellThreeDBlendOp::parse(raw).map(V::BlendOp),
                2 | 3 | 5 | 6 => MaxwellThreeDBlendFactor::parse(raw).map(V::BlendFactor),
                _ => None,
            },
        )
    }
    #[must_use]
    pub fn blend_controls(&self) -> MaxwellThreeDBlendControlState<'a> {
        MaxwellThreeDBlendControlState {
            registers: self.registers,
        }
    }
}
