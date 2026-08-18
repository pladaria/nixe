//! Typed, source-preserving `MAXWELL_B` register state.
//!
//! Register writes remain separate from their later draw-time interpretation.
//! In particular, an undocumented hardware reset must stay `Unset`: zero is
//! not a reset value unless a pinned public source establishes that fact.

use std::collections::BTreeMap;

use nixe_gpu::GpuMethodId;

use crate::MaxwellMethodSource;

use super::{
    MAXWELL_PIPELINE_SHADER_COUNT, MaxwellThreeDColorReductionState,
    MaxwellThreeDColorReductionStateWrite, MaxwellThreeDConstantColorRenderingState,
    MaxwellThreeDConstantColorRenderingStateWrite, MaxwellThreeDCounterState,
    MaxwellThreeDCounterStateWrite, MaxwellThreeDCoverageState, MaxwellThreeDCoverageStateWrite,
    MaxwellThreeDFalconMaskedRegisterWrite, MaxwellThreeDFalconState,
    MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDFixedFunctionWrite,
    MaxwellThreeDInstrumentationState, MaxwellThreeDInstrumentationStateWrite,
    MaxwellThreeDL2CacheState, MaxwellThreeDL2CacheStateWrite, MaxwellThreeDLineState,
    MaxwellThreeDLineStateWrite, MaxwellThreeDMmeShadowScratchIndex, MaxwellThreeDMmeState,
    MaxwellThreeDMmeStateWrite, MaxwellThreeDPolygonMode, MaxwellThreeDRenderEnableState,
    MaxwellThreeDRenderEnableStateWrite, MaxwellThreeDRenderTargetState,
    MaxwellThreeDRenderTargetWrite, MaxwellThreeDReportSemaphoreState,
    MaxwellThreeDReportSemaphoreStateWrite, MaxwellThreeDShaderBindingState,
    MaxwellThreeDShaderBindingWrite, MaxwellThreeDShaderExecutionState,
    MaxwellThreeDShaderExecutionStateWrite, MaxwellThreeDTiledCacheState,
    MaxwellThreeDTiledCacheStateWrite, MaxwellThreeDTirMode, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexInputWrite, MaxwellThreeDZCullState, MaxwellThreeDZCullStateWrite,
};

pub const MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT: usize = 32;

/// Byte-addressed polygon-mode registers read by the Switch graphics macros.
pub(super) const MAXWELL_THREE_D_FRONT_POLYGON_MODE_METHOD: u32 = 0x0dac;
pub(super) const MAXWELL_THREE_D_BACK_POLYGON_MODE_METHOD: u32 = 0x0db0;
pub(super) const MAXWELL_THREE_D_WINDOW_ORIGIN_METHOD: u32 = 0x13ac;
pub(super) const MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD: u32 = 0x2000;
pub(super) const MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE: u32 = 0x40;

/// Raw reset bits observed by MME register reads before either method is set.
///
/// yuzu explicitly zero-initializes the complete Maxwell 3D register file, and
/// Ryujinx independently exposes its zero-initialized unmanaged class state
/// directly to MME reads. The value is intentionally retained as raw-only:
/// zero is not one of NVIDIA's published polygon-mode enum encodings.
/// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/engines/maxwell_3d.cpp#L37-L42>
/// <https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Device/DeviceState.cs#L16-L30>
pub(super) const MAXWELL_THREE_D_POLYGON_MODE_RESET: u32 = 0;

/// Raw reset header shared by the six `SET_PIPELINE_SHADER(i)` slots.
///
/// The same pinned register-file sources above establish zero before guest
/// programming. Unlike polygon mode, this is also a valid typed header:
/// disabled, with the zero-valued `VERTEX_CULL_BEFORE_FETCH` type field.
pub(super) const MAXWELL_THREE_D_PIPELINE_SHADER_RESET: u32 = 0;

/// Reset binding group for every pipeline slot.
///
/// The pinned zero-initialized Maxwell register-file implementations cited
/// above cover `SET_PIPELINE_BINDING` as well as the pipeline header. Group
/// zero is a valid typed value and is observable when a guest relies on reset
/// state rather than redundantly programming the method.
pub(super) const MAXWELL_THREE_D_PIPELINE_BINDING_RESET: u32 = 0;

/// Reset value of `SET_WINDOW_ORIGIN`: upper-left with no Y flip.
///
/// NVIDIA publishes those encodings as zero. yuzu initializes the live class
/// state first and then copies it into the shadow register file; Ryujinx
/// independently initializes its live and shadow arrays through the same
/// default-state routine. Neither overrides `SET_WINDOW_ORIGIN`, so both
/// retain the zero-initialized class value in shadow RAM:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2599-L2605>
/// <https://ni.4a.si/anonymous/yuzu/tree/src/video_core/engines/maxwell_3d.cpp?id=9705094a576e6594e359cc0256b63385ac05de3f#n29>
/// <https://www.git.axenov.dev/Museum/ryujinx/src/commit/4594c3b31014655fb0b37f1305598fc3cbafdc73/Ryujinx.Graphics.Gpu/State/GpuState.cs#L48-L67>
pub(super) const MAXWELL_THREE_D_WINDOW_ORIGIN_RESET: u32 = 0;

/// Returns a raw class-register reset verified by pinned public sources.
///
/// The live and MME shadow register files share this lookup so replay starts
/// from the same architectural state. Registers absent from this set remain
/// unknown rather than being silently fabricated as zero.
pub(super) const fn verified_raw_register_reset(method: GpuMethodId) -> Option<u32> {
    match method.0 {
        MAXWELL_THREE_D_FRONT_POLYGON_MODE_METHOD | MAXWELL_THREE_D_BACK_POLYGON_MODE_METHOD => {
            Some(MAXWELL_THREE_D_POLYGON_MODE_RESET)
        }
        MAXWELL_THREE_D_WINDOW_ORIGIN_METHOD => Some(MAXWELL_THREE_D_WINDOW_ORIGIN_RESET),
        raw if raw >= MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD => {
            let offset = raw - MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD;
            let pipeline = offset / MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE;
            let register = offset % MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE;
            if pipeline < MAXWELL_PIPELINE_SHADER_COUNT as u32 {
                match register {
                    0 => Some(MAXWELL_THREE_D_PIPELINE_SHADER_RESET),
                    0x10 => Some(MAXWELL_THREE_D_PIPELINE_BINDING_RESET),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// How a modeled Maxwell register acquired its current value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDRegisterOrigin {
    /// No verified reset or method write establishes a value.
    Unset,
    /// Pinned public sources establish the modeled profile's reset value.
    VerifiedReset,
    /// A validated guest method programmed the register.
    Programmed,
}

/// One typed register with explicit validity and optional write provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDRegister<T> {
    origin: MaxwellThreeDRegisterOrigin,
    raw: Option<u32>,
    value: Option<T>,
    source: Option<MaxwellMethodSource>,
}

impl<T> MaxwellThreeDRegister<T> {
    /// Returns whether the value is absent, sourced from a verified reset, or
    /// explicitly programmed. Callers must not treat `Unset` as zero.
    #[must_use]
    pub const fn origin(&self) -> MaxwellThreeDRegisterOrigin {
        self.origin
    }

    /// Exact method/reset bits retained before later semantic interpretation.
    #[must_use]
    pub const fn raw(&self) -> Option<u32> {
        self.raw
    }

    /// Typed value, available only when the register has a valid origin.
    #[must_use]
    pub const fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Source of a programmed value. Reset and unset states have no method.
    #[must_use]
    pub const fn source(&self) -> Option<MaxwellMethodSource> {
        self.source
    }
}

impl<T> Default for MaxwellThreeDRegister<T> {
    fn default() -> Self {
        Self {
            origin: MaxwellThreeDRegisterOrigin::Unset,
            raw: None,
            value: None,
            source: None,
        }
    }
}

impl<T> MaxwellThreeDRegister<T> {
    pub(super) const fn verified_reset(raw: u32, value: Option<T>) -> Self {
        Self {
            origin: MaxwellThreeDRegisterOrigin::VerifiedReset,
            raw: Some(raw),
            value,
            source: None,
        }
    }

    pub(super) const fn programmed(raw: u32, value: T, source: MaxwellMethodSource) -> Self {
        Self {
            origin: MaxwellThreeDRegisterOrigin::Programmed,
            raw: Some(raw),
            value: Some(value),
            source: Some(source),
        }
    }
}

/// Exact IEEE-754 bits written to `SET_POINT_SIZE`.
///
/// Draw-time validation deliberately happens in the later semantic snapshot;
/// preserving the register bits here does not claim that every bit pattern is
/// a usable rasterizer point size.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDPointSize(u32);

impl MaxwellThreeDPointSize {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Selects a shader-output slot as the source of rasterized point size.
///
/// NVIDIA publishes the enable bit and eight-bit slot in the pinned
/// `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3340-L3344>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDAttributePointSize {
    enabled: bool,
    slot: u8,
}

impl MaxwellThreeDAttributePointSize {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x0ff1 == 0 {
            Some(Self {
                enabled: raw & 1 != 0,
                slot: ((raw >> 4) & 0xff) as u8,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }

    #[must_use]
    pub const fn slot(self) -> u8 {
        self.slot
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.enabled as u32 | ((self.slot as u32) << 4)
    }
}

/// Source component selected for generated point-sprite R coordinates.
///
/// The encodings and fields below come from NVIDIA's pinned public
/// `MAXWELL_B` class header.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2957-L2994>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPointSpriteRMode {
    Zero = 0,
    FromR = 1,
    FromS = 2,
}

impl MaxwellThreeDPointSpriteRMode {
    const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Zero),
            1 => Some(Self::FromR),
            2 => Some(Self::FromS),
            _ => None,
        }
    }
}

/// Vertical origin used when point-sprite texture coordinates are generated.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPointSpriteOrigin {
    Bottom = 0,
    Top = 1,
}

/// Typed `SET_POINT_SPRITE_SELECT` texture-coordinate selection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDPointSpriteSelect {
    r_mode: MaxwellThreeDPointSpriteRMode,
    origin: MaxwellThreeDPointSpriteOrigin,
    generated_texture_mask: u16,
}

impl MaxwellThreeDPointSpriteSelect {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        let Some(r_mode) = MaxwellThreeDPointSpriteRMode::parse(raw & 3) else {
            return None;
        };
        Some(Self {
            r_mode,
            origin: if raw & 4 == 0 {
                MaxwellThreeDPointSpriteOrigin::Bottom
            } else {
                MaxwellThreeDPointSpriteOrigin::Top
            },
            generated_texture_mask: ((raw >> 3) & 0x03ff) as u16,
        })
    }

    #[must_use]
    pub const fn r_mode(self) -> MaxwellThreeDPointSpriteRMode {
        self.r_mode
    }

    #[must_use]
    pub const fn origin(self) -> MaxwellThreeDPointSpriteOrigin {
        self.origin
    }

    #[must_use]
    pub const fn generated_texture_mask(self) -> u16 {
        self.generated_texture_mask
    }

    #[must_use]
    pub const fn generates_texture(self, texture: u8) -> bool {
        texture < 10 && self.generated_texture_mask & (1 << texture) != 0
    }

    #[must_use]
    pub const fn affects_point_coordinates(self) -> bool {
        self.generated_texture_mask != 0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.r_mode as u32
            | ((self.origin as u32) << 2)
            | ((self.generated_texture_mask as u32) << 3)
    }
}

/// Pixel-center convention selected for rasterized point primitives.
///
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3097-L3100>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPointCenterMode {
    OpenGl = 0,
    Direct3D = 1,
}

/// Current polygon edge flag programmed by `SET_EDGE_FLAG`.
///
/// NVIDIA defines this as a strict boolean in its pinned public `MAXWELL_B`
/// class header. A disabled flag can suppress polygon boundary rasterization
/// in non-fill polygon modes; an enabled flag is the neutral all-edges case.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2927-L2930>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDEdgeFlag {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDEdgeFlag {
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

impl MaxwellThreeDPointCenterMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::OpenGl),
            1 => Some(Self::Direct3D),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Opaque eight-bit value written to `SET_ALPHA_FRACTION`.
///
/// NVIDIA publishes the register field width but not its numerical
/// interpretation, so the frontend retains the encoded byte without assuming
/// a scale or transfer function.
///
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L578-L579>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDAlphaFraction(u8);

impl MaxwellThreeDAlphaFraction {
    #[must_use]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

/// Raster work-domain selection programmed by `SET_RASTER_BOUNDING_BOX`.
///
/// NVIDIA defines both mode encodings in its pinned public `MAXWELL_B` class
/// header. This controls how the guest GPU bounds internal raster work; it
/// does not clip primitives or change the API-visible viewport.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L337-L341>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDRasterBoundingBoxMode {
    BoundingBox = 0,
    FullViewport = 1,
}

/// Typed `SET_RASTER_BOUNDING_BOX` mode and eight-bit padding field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDRasterBoundingBox {
    mode: MaxwellThreeDRasterBoundingBoxMode,
    pad: u8,
}

impl MaxwellThreeDRasterBoundingBox {
    #[must_use]
    pub const fn new(mode: MaxwellThreeDRasterBoundingBoxMode, pad: u8) -> Self {
        Self { mode, pad }
    }

    pub(super) const fn parse(raw: u32) -> Self {
        Self::new(
            if raw & 1 == 0 {
                MaxwellThreeDRasterBoundingBoxMode::BoundingBox
            } else {
                MaxwellThreeDRasterBoundingBoxMode::FullViewport
            },
            ((raw >> 4) & 0xff) as u8,
        )
    }

    #[must_use]
    pub const fn mode(self) -> MaxwellThreeDRasterBoundingBoxMode {
        self.mode
    }

    #[must_use]
    pub const fn pad(self) -> u8 {
        self.pad
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.mode as u32 | ((self.pad as u32) << 4)
    }
}

/// Selects the specialized triangle-based fill path.
///
/// NVIDIA publishes all three encodings in its pinned public `MAXWELL_B`
/// class header. The frontend retains the selection without pretending that
/// the neutral backend can reproduce either effective mode.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1822-L1826>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDFillViaTriangleMode {
    Disabled = 0,
    FillAll = 1,
    FillBoundingBox = 2,
}

impl MaxwellThreeDFillViaTriangleMode {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::FillAll),
            2 => Some(Self::FillBoundingBox),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Enables conservative rasterization for covered primitives.
///
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1837-L1840>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDConservativeRasterEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDConservativeRasterEnable {
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

/// Raw rasterization registers whose derived combinations are validated later.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDRasterState {
    point_size: MaxwellThreeDRegister<MaxwellThreeDPointSize>,
    attribute_point_size: MaxwellThreeDRegister<MaxwellThreeDAttributePointSize>,
    point_sprite_enable: MaxwellThreeDRegister<bool>,
    anti_aliased_point_enable: MaxwellThreeDRegister<bool>,
    point_sprite_select: MaxwellThreeDRegister<MaxwellThreeDPointSpriteSelect>,
    point_center_mode: MaxwellThreeDRegister<MaxwellThreeDPointCenterMode>,
    edge_flag: MaxwellThreeDRegister<MaxwellThreeDEdgeFlag>,
    alpha_fraction: MaxwellThreeDRegister<MaxwellThreeDAlphaFraction>,
    bounding_box: MaxwellThreeDRegister<MaxwellThreeDRasterBoundingBox>,
    fill_via_triangle: MaxwellThreeDRegister<MaxwellThreeDFillViaTriangleMode>,
    conservative_raster: MaxwellThreeDRegister<MaxwellThreeDConservativeRasterEnable>,
    polygon_smooth_enable: MaxwellThreeDRegister<bool>,
    polygon_stipple_enable: MaxwellThreeDRegister<bool>,
    polygon_stipple_pattern:
        [MaxwellThreeDRegister<u32>; MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT],
}

impl MaxwellThreeDRasterState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            point_size: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            attribute_point_size: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            point_sprite_enable: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            anti_aliased_point_enable: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            point_sprite_select: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            point_center_mode: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            edge_flag: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            alpha_fraction: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            bounding_box: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            fill_via_triangle: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            conservative_raster: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            polygon_smooth_enable: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            polygon_stipple_enable: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            polygon_stipple_pattern: [const {
                MaxwellThreeDRegister {
                    origin: MaxwellThreeDRegisterOrigin::Unset,
                    raw: None,
                    value: None,
                    source: None,
                }
            }; MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT],
        }
    }

    #[must_use]
    pub const fn point_size(&self) -> &MaxwellThreeDRegister<MaxwellThreeDPointSize> {
        &self.point_size
    }

    #[must_use]
    pub const fn attribute_point_size(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDAttributePointSize> {
        &self.attribute_point_size
    }

    #[must_use]
    pub const fn point_sprite_enable(&self) -> &MaxwellThreeDRegister<bool> {
        &self.point_sprite_enable
    }

    #[must_use]
    pub const fn anti_aliased_point_enable(&self) -> &MaxwellThreeDRegister<bool> {
        &self.anti_aliased_point_enable
    }

    #[must_use]
    pub const fn point_sprite_select(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDPointSpriteSelect> {
        &self.point_sprite_select
    }

    #[must_use]
    pub const fn point_center_mode(&self) -> &MaxwellThreeDRegister<MaxwellThreeDPointCenterMode> {
        &self.point_center_mode
    }

    #[must_use]
    pub const fn edge_flag(&self) -> &MaxwellThreeDRegister<MaxwellThreeDEdgeFlag> {
        &self.edge_flag
    }

    #[must_use]
    pub const fn alpha_fraction(&self) -> &MaxwellThreeDRegister<MaxwellThreeDAlphaFraction> {
        &self.alpha_fraction
    }

    #[must_use]
    pub const fn bounding_box(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRasterBoundingBox> {
        &self.bounding_box
    }

    #[must_use]
    pub const fn fill_via_triangle(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDFillViaTriangleMode> {
        &self.fill_via_triangle
    }

    #[must_use]
    pub const fn conservative_raster(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDConservativeRasterEnable> {
        &self.conservative_raster
    }

    #[must_use]
    pub const fn polygon_smooth_enable(&self) -> &MaxwellThreeDRegister<bool> {
        &self.polygon_smooth_enable
    }

    #[must_use]
    pub const fn polygon_stipple_enable(&self) -> &MaxwellThreeDRegister<bool> {
        &self.polygon_stipple_enable
    }

    #[must_use]
    pub const fn polygon_stipple_pattern(
        &self,
    ) -> &[MaxwellThreeDRegister<u32>; MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT] {
        &self.polygon_stipple_pattern
    }
}

/// Clip-space Z range selected before viewport transformation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDViewportZClipRange {
    NegativeWToPositiveW = 0,
    ZeroToPositiveW = 1,
}

/// Pixel-center convention applied by viewport rasterization.
///
/// NVIDIA publishes both encodings in its pinned public `MAXWELL_B` class
/// header. Half-integer centers match the neutral pipeline convention;
/// integer centers require an explicit lowering path.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3362-L3365>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDViewportPixelCenter {
    HalfIntegers = 0,
    Integers = 1,
}

impl MaxwellThreeDViewportPixelCenter {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::HalfIntegers),
            1 => Some(Self::Integers),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

impl MaxwellThreeDViewportZClipRange {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::NegativeWToPositiveW),
            1 => Some(Self::ZeroToPositiveW),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Raw viewport registers whose complete combinations are validated later.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDViewportState {
    z_clip_range: MaxwellThreeDRegister<MaxwellThreeDViewportZClipRange>,
    pixel_center: MaxwellThreeDRegister<MaxwellThreeDViewportPixelCenter>,
}

impl MaxwellThreeDViewportState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            z_clip_range: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
            pixel_center: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
        }
    }

    #[must_use]
    pub const fn z_clip_range(&self) -> &MaxwellThreeDRegister<MaxwellThreeDViewportZClipRange> {
        &self.z_clip_range
    }

    #[must_use]
    pub const fn pixel_center(&self) -> &MaxwellThreeDRegister<MaxwellThreeDViewportPixelCenter> {
        &self.pixel_center
    }
}

/// Complete currently modeled state of one channel's `MAXWELL_B` engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDState {
    raw_registers: BTreeMap<u32, MaxwellThreeDRegister<u32>>,
    render_targets: Box<MaxwellThreeDRenderTargetState>,
    fixed_function: Box<MaxwellThreeDFixedFunctionState>,
    vertex_input: Box<MaxwellThreeDVertexInputState>,
    shader_bindings: Box<MaxwellThreeDShaderBindingState>,
    raster: MaxwellThreeDRasterState,
    viewport: MaxwellThreeDViewportState,
    render_enable: MaxwellThreeDRenderEnableState,
    shader_execution: MaxwellThreeDShaderExecutionState,
    color_reduction: MaxwellThreeDColorReductionState,
    constant_color_rendering: MaxwellThreeDConstantColorRenderingState,
    coverage: MaxwellThreeDCoverageState,
    line: MaxwellThreeDLineState,
    zcull: MaxwellThreeDZCullState,
    l2_cache: MaxwellThreeDL2CacheState,
    report_semaphore: MaxwellThreeDReportSemaphoreState,
    counters: MaxwellThreeDCounterState,
    falcon: MaxwellThreeDFalconState,
    instrumentation: MaxwellThreeDInstrumentationState,
    tiled_cache: MaxwellThreeDTiledCacheState,
    mme: MaxwellThreeDMmeState,
}

impl Default for MaxwellThreeDState {
    fn default() -> Self {
        let mut raw_registers = BTreeMap::new();
        for method in [
            MAXWELL_THREE_D_FRONT_POLYGON_MODE_METHOD,
            MAXWELL_THREE_D_BACK_POLYGON_MODE_METHOD,
            MAXWELL_THREE_D_WINDOW_ORIGIN_METHOD,
        ] {
            let reset = verified_raw_register_reset(GpuMethodId(method))
                .expect("listed Maxwell register reset must be verified");
            raw_registers.insert(
                method,
                MaxwellThreeDRegister::verified_reset(reset, Some(reset)),
            );
        }
        for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
            let method = MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD
                + pipeline as u32 * MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE;
            for register in [method, method + 0x10] {
                let reset = verified_raw_register_reset(GpuMethodId(register))
                    .expect("listed Maxwell pipeline reset must be verified");
                raw_registers.insert(
                    register,
                    MaxwellThreeDRegister::verified_reset(reset, Some(reset)),
                );
            }
        }
        Self {
            raw_registers,
            render_targets: Default::default(),
            fixed_function: Default::default(),
            vertex_input: Default::default(),
            shader_bindings: Default::default(),
            raster: Default::default(),
            viewport: Default::default(),
            render_enable: Default::default(),
            shader_execution: Default::default(),
            color_reduction: Default::default(),
            constant_color_rendering: Default::default(),
            coverage: Default::default(),
            line: Default::default(),
            zcull: Default::default(),
            l2_cache: Default::default(),
            report_semaphore: Default::default(),
            counters: Default::default(),
            falcon: Default::default(),
            instrumentation: Default::default(),
            tiled_cache: Default::default(),
            mme: Default::default(),
        }
    }
}

impl MaxwellThreeDState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn render_targets(&self) -> &MaxwellThreeDRenderTargetState {
        &self.render_targets
    }

    #[must_use]
    pub fn fixed_function(&self) -> &MaxwellThreeDFixedFunctionState {
        &self.fixed_function
    }

    #[must_use]
    pub fn vertex_input(&self) -> &MaxwellThreeDVertexInputState {
        &self.vertex_input
    }

    #[must_use]
    pub fn shader_bindings(&self) -> &MaxwellThreeDShaderBindingState {
        &self.shader_bindings
    }

    #[must_use]
    pub const fn raster(&self) -> &MaxwellThreeDRasterState {
        &self.raster
    }

    #[must_use]
    pub const fn viewport(&self) -> &MaxwellThreeDViewportState {
        &self.viewport
    }

    #[must_use]
    pub const fn render_enable(&self) -> &MaxwellThreeDRenderEnableState {
        &self.render_enable
    }

    #[must_use]
    pub const fn shader_execution(&self) -> &MaxwellThreeDShaderExecutionState {
        &self.shader_execution
    }

    #[must_use]
    pub const fn color_reduction(&self) -> &MaxwellThreeDColorReductionState {
        &self.color_reduction
    }

    #[must_use]
    pub const fn constant_color_rendering(&self) -> &MaxwellThreeDConstantColorRenderingState {
        &self.constant_color_rendering
    }

    #[must_use]
    pub const fn coverage(&self) -> &MaxwellThreeDCoverageState {
        &self.coverage
    }

    #[must_use]
    pub const fn line(&self) -> &MaxwellThreeDLineState {
        &self.line
    }

    #[must_use]
    pub const fn zcull(&self) -> &MaxwellThreeDZCullState {
        &self.zcull
    }

    #[must_use]
    pub const fn l2_cache(&self) -> &MaxwellThreeDL2CacheState {
        &self.l2_cache
    }

    #[must_use]
    pub const fn report_semaphore(&self) -> &MaxwellThreeDReportSemaphoreState {
        &self.report_semaphore
    }

    #[must_use]
    pub const fn counters(&self) -> &MaxwellThreeDCounterState {
        &self.counters
    }

    #[must_use]
    pub const fn falcon(&self) -> &MaxwellThreeDFalconState {
        &self.falcon
    }

    #[must_use]
    pub const fn instrumentation(&self) -> &MaxwellThreeDInstrumentationState {
        &self.instrumentation
    }

    #[must_use]
    pub const fn tiled_cache(&self) -> &MaxwellThreeDTiledCacheState {
        &self.tiled_cache
    }

    #[must_use]
    pub const fn mme(&self) -> &MaxwellThreeDMmeState {
        &self.mme
    }

    pub(super) const fn mme_mut(&mut self) -> &mut MaxwellThreeDMmeState {
        &mut self.mme
    }

    /// Returns the last validated raw value for one byte-addressed class method.
    #[must_use]
    pub fn raw_register(&self, method: GpuMethodId) -> Option<&MaxwellThreeDRegister<u32>> {
        self.raw_registers.get(&method.0)
    }

    pub(super) fn record_raw_register(&mut self, source: MaxwellMethodSource) {
        self.raw_registers.insert(
            source.method().0,
            MaxwellThreeDRegister::programmed(source.argument(), source.argument(), source),
        );
    }

    pub(crate) fn pipeline_dependencies(&self, active_color_targets: &[u8]) -> Box<[Option<u32>]> {
        let mut dependencies = Vec::new();
        self.fixed_function
            .append_pipeline_dependencies(&mut dependencies, active_color_targets);
        self.vertex_input
            .append_pipeline_dependencies(&mut dependencies);
        self.shader_bindings
            .append_pipeline_dependencies(&mut dependencies);
        dependencies.push(self.shader_execution.l1_configuration().raw());
        if self.shader_bindings.has_enabled_pipeline() {
            self.shader_execution
                .append_shader_pipeline_dependencies(&mut dependencies);
        }
        if self
            .vertex_input
            .primitive()
            .active_begin()
            .is_some_and(|begin| begin.topology() == 0)
        {
            dependencies.push(self.raster.attribute_point_size.raw());
            if !self
                .raster
                .attribute_point_size
                .value()
                .is_some_and(|value| value.enabled())
            {
                dependencies.push(self.raster.point_size.raw());
            }
            dependencies.push(self.raster.point_sprite_enable.raw());
            dependencies.push(self.raster.anti_aliased_point_enable.raw());
            if self
                .raster
                .point_sprite_select
                .value()
                .is_some_and(|value| value.affects_point_coordinates())
            {
                dependencies.push(self.raster.point_sprite_select.raw());
            }
            dependencies.push(self.raster.point_center_mode.raw());
        }
        if self.edge_flag_affects_draw() {
            dependencies.push(self.raster.edge_flag.raw());
        }
        if self
            .vertex_input
            .primitive()
            .active_begin()
            .is_some_and(|begin| matches!(begin.topology(), 4..=6))
        {
            dependencies.push(self.raster.polygon_smooth_enable.raw());
            dependencies.push(self.raster.polygon_stipple_enable.raw());
            if self.raster.polygon_stipple_enable.value() == Some(&true) {
                dependencies.extend(
                    self.raster
                        .polygon_stipple_pattern
                        .iter()
                        .map(MaxwellThreeDRegister::raw),
                );
            }
        }
        dependencies.push(self.raster.alpha_fraction.raw());
        self.constant_color_rendering
            .append_pipeline_dependencies(&mut dependencies);
        dependencies.push(self.raster.fill_via_triangle.raw());
        dependencies.push(self.raster.conservative_raster.raw());
        dependencies.push(self.viewport.z_clip_range.raw());
        dependencies.push(self.viewport.pixel_center.raw());
        dependencies.push(self.render_targets.color_target_selection().raw());
        dependencies.push(self.render_targets.depth_target_count().raw());
        if self
            .render_targets
            .render_target_layer()
            .value()
            .is_some_and(|value| value.affects_draw_layering())
        {
            dependencies.push(self.render_targets.render_target_layer().raw());
        }
        if self
            .render_targets
            .render_target_index_offset()
            .value()
            .is_some_and(|value| value.enabled())
        {
            dependencies.push(self.render_targets.render_target_index_offset().raw());
        }
        if active_color_targets.len() > 1 {
            dependencies.push(self.render_targets.separate_fragment_data().raw());
        }
        dependencies.push(self.coverage.csaa_enable().raw());
        if self.coverage.post_z_pixel_shader_imask().value()
            == Some(&super::MaxwellThreeDPostZPixelShaderImask::Enabled)
        {
            dependencies.push(self.coverage.post_z_pixel_shader_imask().raw());
        }
        dependencies.push(self.coverage.tir_mode().raw());
        if self.coverage.tir_mode().value() == Some(&MaxwellThreeDTirMode::RasterNTargetM) {
            dependencies.push(self.coverage.tir_control().raw());
            dependencies.push(self.coverage.tir_modulation().raw());
            dependencies.push(self.coverage.tir_modulation_function().raw());
        }
        if self
            .coverage
            .coverage_to_color()
            .value()
            .is_some_and(|value| value.enabled())
        {
            dependencies.push(self.coverage.coverage_to_color().raw());
        }
        if self
            .coverage
            .alpha_to_coverage_override()
            .value()
            .is_some_and(|value| value.raw() != 0)
        {
            dependencies.push(self.coverage.alpha_to_coverage_override().raw());
        }
        dependencies.push(self.coverage.hybrid_anti_alias_control().raw());
        dependencies.extend(
            self.coverage
                .sample_locations()
                .iter()
                .map(MaxwellThreeDRegister::raw),
        );
        if self
            .coverage
            .ps_output_sample_mask_usage()
            .value()
            .is_some()
            && self.ps_output_sample_mask_effective() != Some(false)
        {
            dependencies.push(self.coverage.ps_output_sample_mask_usage().raw());
        }
        self.line.append_pipeline_dependencies(&mut dependencies);
        dependencies.into_boxed_slice()
    }

    pub(crate) fn ps_output_sample_mask_effective(&self) -> Option<bool> {
        let usage = self.coverage.ps_output_sample_mask_usage().value()?;
        let anti_alias_enable = self
            .fixed_function
            .register(MaxwellThreeDFixedFunctionRegister::AntiAliasEnable)
            .value()
            .and_then(|value| match value {
                MaxwellThreeDFixedFunctionValue::Boolean(value) => Some(*value),
                _ => None,
            });
        usage.effective(anti_alias_enable)
    }

    pub(crate) fn edge_flag_affects_draw(&self) -> bool {
        let polygon_topology = self
            .vertex_input
            .primitive()
            .active_begin()
            .is_some_and(|begin| matches!(begin.topology(), 4..=6));
        let non_fill_polygon_mode = [
            MaxwellThreeDFixedFunctionRegister::FrontPolygonMode,
            MaxwellThreeDFixedFunctionRegister::BackPolygonMode,
        ]
        .into_iter()
        .any(|register| {
            matches!(
                self.fixed_function.register(register).value(),
                Some(MaxwellThreeDFixedFunctionValue::PolygonMode(
                    MaxwellThreeDPolygonMode::Point | MaxwellThreeDPolygonMode::Line
                ))
            )
        });
        polygon_topology
            && non_fill_polygon_mode
            && self.raster.edge_flag.value() == Some(&MaxwellThreeDEdgeFlag::Disabled)
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDStateWrite) {
        match write {
            MaxwellThreeDStateWrite::PointSize { value, source } => {
                self.raster.point_size =
                    MaxwellThreeDRegister::programmed(value.bits(), value, source);
            }
            MaxwellThreeDStateWrite::AttributePointSize { value, source } => {
                self.raster.attribute_point_size =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::PointSpriteEnable { value, source } => {
                self.raster.point_sprite_enable =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDStateWrite::AntiAliasedPointEnable { value, source } => {
                self.raster.anti_aliased_point_enable =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDStateWrite::PointSpriteSelect { value, source } => {
                self.raster.point_sprite_select =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::PointCenterMode { value, source } => {
                self.raster.point_center_mode =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::EdgeFlag { value, source } => {
                self.raster.edge_flag =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::AlphaFraction { value, source } => {
                self.raster.alpha_fraction =
                    MaxwellThreeDRegister::programmed(u32::from(value.raw()), value, source);
            }
            MaxwellThreeDStateWrite::RasterBoundingBox { value, source } => {
                self.raster.bounding_box =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::FillViaTriangle { value, source } => {
                self.raster.fill_via_triangle =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::ConservativeRaster { value, source } => {
                self.raster.conservative_raster =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::PolygonSmoothEnable { value, source } => {
                self.raster.polygon_smooth_enable =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDStateWrite::PolygonStippleEnable { value, source } => {
                self.raster.polygon_stipple_enable =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDStateWrite::PolygonStipplePattern {
                word,
                value,
                source,
            } => {
                self.raster.polygon_stipple_pattern[word as usize] =
                    MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDStateWrite::ViewportZClip { value, source } => {
                self.viewport.z_clip_range =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::ViewportPixelCenter { value, source } => {
                self.viewport.pixel_center =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::RenderTarget(write) => self.render_targets.apply(write),
            MaxwellThreeDStateWrite::FixedFunction(write) => self.fixed_function.apply(write),
            MaxwellThreeDStateWrite::VertexInput(write) => self.vertex_input.apply(write),
            MaxwellThreeDStateWrite::ShaderBinding(write) => self.shader_bindings.apply(write),
            MaxwellThreeDStateWrite::RenderEnable(write) => self.render_enable.apply(write),
            MaxwellThreeDStateWrite::ShaderExecution(write) => self.shader_execution.apply(write),
            MaxwellThreeDStateWrite::ColorReduction(write) => self.color_reduction.apply(write),
            MaxwellThreeDStateWrite::ConstantColorRendering(write) => {
                self.constant_color_rendering.apply(write);
            }
            MaxwellThreeDStateWrite::Coverage(write) => self.coverage.apply(write),
            MaxwellThreeDStateWrite::Line(write) => self.line.apply(write),
            MaxwellThreeDStateWrite::ZCull(write) => self.zcull.apply(write),
            MaxwellThreeDStateWrite::L2Cache(write) => self.l2_cache.apply(write),
            MaxwellThreeDStateWrite::ReportSemaphore(write) => self.report_semaphore.apply(write),
            MaxwellThreeDStateWrite::Counter(write) => self.counters.apply(write),
            MaxwellThreeDStateWrite::Instrumentation(write) => self.instrumentation.apply(write),
            MaxwellThreeDStateWrite::TiledCache(write) => self.tiled_cache.apply(write),
            MaxwellThreeDStateWrite::FalconMaskedRegister(write) => {
                self.falcon.apply(write);
                let completion = MaxwellThreeDMmeStateWrite::ShadowScratch {
                    index: MaxwellThreeDMmeShadowScratchIndex::new(0),
                    value: 1,
                    source: write.source(),
                };
                self.mme.apply(completion);
                self.raw_registers.insert(
                    0x3400,
                    MaxwellThreeDRegister::programmed(1, 1, write.source()),
                );
            }
            MaxwellThreeDStateWrite::Mme(write) => self.mme.apply(write),
        }
    }

    pub(in crate::engines) fn validate_cross_registers(
        &self,
    ) -> Result<(), MaxwellThreeDStateValidationError> {
        for target in self.render_targets.color() {
            if target.kind().value() == Some(&super::MaxwellThreeDImageKind::ThreeDimensional)
                && target.layer().value().is_some_and(|layer| *layer != 0)
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: target.layer().source().or_else(|| target.kind().source()),
                    reason: "a three-dimensional color target cannot select an array layer",
                });
            }
        }
        for stream in self.vertex_input.streams() {
            if let (Some(address), Some(limit)) = (stream.address(), stream.limit())
                && address.get() > limit.get()
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: stream
                        .limit_lower()
                        .source()
                        .or_else(|| stream.limit_upper().source()),
                    reason: "a vertex stream limit precedes its start address",
                });
            }
            if stream.instanced().value() == Some(&true) && stream.frequency().value() == Some(&0) {
                return Err(MaxwellThreeDStateValidationError {
                    source: stream
                        .frequency()
                        .source()
                        .or_else(|| stream.instanced().source()),
                    reason: "an instanced vertex stream cannot have zero frequency",
                });
            }
        }
        let local_memory = self.shader_execution.shader_local_memory();
        if let (Some(address), Some(size)) = (local_memory.address(), local_memory.size())
            && address
                .get()
                .checked_add(size)
                .is_none_or(|end| end > (1_u64 << 40))
        {
            return Err(MaxwellThreeDStateValidationError {
                source: local_memory
                    .size_lower()
                    .source()
                    .or_else(|| local_memory.size_upper().source())
                    .or_else(|| local_memory.address_lower().source())
                    .or_else(|| local_memory.address_upper().source()),
                reason: "shader-local-memory region exceeds the 40-bit GPU address space",
            });
        }
        for attribute in self.vertex_input.attributes() {
            let Some(format) = attribute.value().filter(|format| format.enabled()) else {
                continue;
            };
            if self.vertex_input.streams()[format.stream() as usize]
                .format()
                .value()
                .is_some_and(|stream| !stream.enabled())
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: attribute.source(),
                    reason: "an enabled vertex attribute references an explicitly disabled stream",
                });
            }
            let stream = &self.vertex_input.streams()[format.stream() as usize];
            if let (Some(address), Some(limit), Some(component_widths)) =
                (stream.address(), stream.limit(), format.component_widths())
            {
                let required =
                    u64::from(format.offset()).checked_add(u64::from(component_widths.byte_size()));
                let available = limit
                    .get()
                    .checked_sub(address.get())
                    .and_then(|distance| distance.checked_add(1));
                if required.is_none() || required > available {
                    return Err(MaxwellThreeDStateValidationError {
                        source: attribute.source(),
                        reason: "a vertex attribute format exceeds its stream range",
                    });
                }
            }
        }
        let index = self.vertex_input.index();
        if let (Some(upper), Some(lower), Some(limit_upper), Some(limit_lower)) = (
            index.address_upper().value(),
            index.address_lower().value(),
            index.limit_upper().value(),
            index.limit_lower().value(),
        ) {
            let address = (u64::from(*upper) << 32) | u64::from(*lower);
            let limit = (u64::from(*limit_upper) << 32) | u64::from(*limit_lower);
            if address > limit {
                return Err(MaxwellThreeDStateValidationError {
                    source: index.limit_lower().source(),
                    reason: "the index-buffer limit precedes its start address",
                });
            }
            if let Some(size) = index.element_size().value()
                && (address % size.bytes() != 0
                    || limit
                        .checked_add(1)
                        .is_some_and(|end| end % size.bytes() != 0))
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: index.element_size().source(),
                    reason: "the index-buffer range is not aligned to its element size",
                });
            }
            if let (Some(size), Some(first)) = (index.element_size().value(), index.first().value())
            {
                let first_address = u64::from(*first)
                    .checked_mul(size.bytes())
                    .and_then(|offset| address.checked_add(offset));
                if first_address.is_none_or(|first_address| first_address > limit) {
                    return Err(MaxwellThreeDStateValidationError {
                        source: index.first().source(),
                        reason: "the first index lies outside the index-buffer range",
                    });
                }
            }
        }
        let bindings = self.shader_bindings();
        let mut stages = [false; 6];
        for pipeline in bindings.pipeline() {
            if pipeline.enabled().value() != Some(&true) {
                continue;
            }
            let Some(stage) = pipeline.stage().value() else {
                continue;
            };
            let stage_index = *stage as usize;
            if stages[stage_index] {
                return Err(MaxwellThreeDStateValidationError {
                    source: pipeline.stage().source(),
                    reason: "two enabled pipeline slots expose the same shader stage",
                });
            }
            stages[stage_index] = true;
        }
        for pool in [bindings.texture_headers(), bindings.samplers()] {
            if let (Some(address), Some(maximum_index)) =
                (pool.address(), pool.maximum_index().value())
            {
                let byte_count = u64::from(*maximum_index)
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(32));
                if address.get() & 31 != 0
                    || byte_count
                        .and_then(|size| address.get().checked_add(size))
                        .is_none_or(|end| end > (1_u64 << 40))
                {
                    return Err(MaxwellThreeDStateValidationError {
                        source: pool
                            .maximum_index()
                            .source()
                            .or_else(|| pool.address_lower().source()),
                        reason: "a descriptor pool address/range is misaligned or overflows",
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::engines) struct MaxwellThreeDStateValidationError {
    pub source: Option<MaxwellMethodSource>,
    pub reason: &'static str,
}

/// One checked `MAXWELL_B` register transition ready for direct application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDStateWrite {
    PointSize {
        value: MaxwellThreeDPointSize,
        source: MaxwellMethodSource,
    },
    AttributePointSize {
        value: MaxwellThreeDAttributePointSize,
        source: MaxwellMethodSource,
    },
    PointSpriteEnable {
        value: bool,
        source: MaxwellMethodSource,
    },
    AntiAliasedPointEnable {
        value: bool,
        source: MaxwellMethodSource,
    },
    PointSpriteSelect {
        value: MaxwellThreeDPointSpriteSelect,
        source: MaxwellMethodSource,
    },
    PointCenterMode {
        value: MaxwellThreeDPointCenterMode,
        source: MaxwellMethodSource,
    },
    EdgeFlag {
        value: MaxwellThreeDEdgeFlag,
        source: MaxwellMethodSource,
    },
    AlphaFraction {
        value: MaxwellThreeDAlphaFraction,
        source: MaxwellMethodSource,
    },
    RasterBoundingBox {
        value: MaxwellThreeDRasterBoundingBox,
        source: MaxwellMethodSource,
    },
    FillViaTriangle {
        value: MaxwellThreeDFillViaTriangleMode,
        source: MaxwellMethodSource,
    },
    ConservativeRaster {
        value: MaxwellThreeDConservativeRasterEnable,
        source: MaxwellMethodSource,
    },
    PolygonSmoothEnable {
        value: bool,
        source: MaxwellMethodSource,
    },
    PolygonStippleEnable {
        value: bool,
        source: MaxwellMethodSource,
    },
    PolygonStipplePattern {
        word: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    ViewportZClip {
        value: MaxwellThreeDViewportZClipRange,
        source: MaxwellMethodSource,
    },
    ViewportPixelCenter {
        value: MaxwellThreeDViewportPixelCenter,
        source: MaxwellMethodSource,
    },
    RenderTarget(MaxwellThreeDRenderTargetWrite),
    FixedFunction(MaxwellThreeDFixedFunctionWrite),
    VertexInput(MaxwellThreeDVertexInputWrite),
    ShaderBinding(MaxwellThreeDShaderBindingWrite),
    RenderEnable(MaxwellThreeDRenderEnableStateWrite),
    ShaderExecution(MaxwellThreeDShaderExecutionStateWrite),
    ColorReduction(MaxwellThreeDColorReductionStateWrite),
    ConstantColorRendering(MaxwellThreeDConstantColorRenderingStateWrite),
    Coverage(MaxwellThreeDCoverageStateWrite),
    Line(MaxwellThreeDLineStateWrite),
    ZCull(MaxwellThreeDZCullStateWrite),
    L2Cache(MaxwellThreeDL2CacheStateWrite),
    ReportSemaphore(MaxwellThreeDReportSemaphoreStateWrite),
    Counter(MaxwellThreeDCounterStateWrite),
    Instrumentation(MaxwellThreeDInstrumentationStateWrite),
    TiledCache(MaxwellThreeDTiledCacheStateWrite),
    FalconMaskedRegister(MaxwellThreeDFalconMaskedRegisterWrite),
    Mme(MaxwellThreeDMmeStateWrite),
}
