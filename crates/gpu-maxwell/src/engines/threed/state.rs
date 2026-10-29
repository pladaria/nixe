//! Typed, source-preserving `MAXWELL_B` register state.
//!
//! Register writes remain separate from their later draw-time interpretation.
//! In particular, an undocumented hardware reset must stay `Unset`: zero is
//! not a reset value unless a pinned public source establishes that fact.

use std::sync::Arc;

use nixe_gpu::GpuMethodId;

use crate::MaxwellMethodSource;

use super::output::{
    BLEND_SEPARATE_ALPHA_BASE, BLEND_SEPARATE_ALPHA_RESET, BLEND_TARGET_STRIDE,
    VIEWPORT_CLIP_MAX_Z_RESET, VIEWPORT_CLIP_MIN_Z_RESET,
};

use super::{
    MAXWELL_COLOR_TARGET_COUNT, MAXWELL_PIPELINE_SHADER_COUNT, MaxwellThreeDColorReductionState,
    MaxwellThreeDCoverageState, MaxwellThreeDFalconMaskedRegisterWrite, MaxwellThreeDFalconState,
    MaxwellThreeDFixedFunctionState, MaxwellThreeDInlineToMemoryState,
    MaxwellThreeDInlineToMemoryStateWrite, MaxwellThreeDLineState,
    MaxwellThreeDMmeShadowScratchIndex, MaxwellThreeDMmeState, MaxwellThreeDMmeStateWrite,
    MaxwellThreeDRenderEnableState, MaxwellThreeDRenderTargetState,
    MaxwellThreeDReportSemaphoreState, MaxwellThreeDResourceRole, MaxwellThreeDShaderBindingState,
    MaxwellThreeDShaderBindingWrite, MaxwellThreeDShaderExecutionState,
    MaxwellThreeDVertexInputState, MaxwellThreeDVertexInputWrite,
    render_targets::{
        MAXWELL_THREE_D_COLOR_COMPRESSION_BASE_METHOD, MAXWELL_THREE_D_COLOR_COMPRESSION_RESET,
        MAXWELL_THREE_D_COLOR_COMPRESSION_STRIDE, MAXWELL_THREE_D_COLOR_TARGET_BASE_METHOD,
        MAXWELL_THREE_D_COLOR_TARGET_LAYER_OFFSET, MAXWELL_THREE_D_COLOR_TARGET_LAYER_RESET,
        MAXWELL_THREE_D_COLOR_TARGET_STRIDE, MAXWELL_THREE_D_DEPTH_TARGET_LAYER_METHOD,
        MAXWELL_THREE_D_DEPTH_TARGET_LAYER_RESET,
    },
};

pub const MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT: usize = 32;

/// Byte-addressed polygon-mode registers read by the Switch graphics macros.
pub(super) const MAXWELL_THREE_D_FRONT_POLYGON_MODE_METHOD: u32 = 0x0dac;
pub(super) const MAXWELL_THREE_D_BACK_POLYGON_MODE_METHOD: u32 = 0x0db0;
pub(super) const MAXWELL_THREE_D_WINDOW_ORIGIN_METHOD: u32 = 0x13ac;
pub(super) const MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD: u32 = 0x2000;
pub(super) const MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE: u32 = 0x40;

/// Initial GM20B/GM200 channel context selects filled polygons. NVIDIA's
/// public method initialization writes 0x1b02 to both 0x0dac and 0x0db0.
/// Table hash and class/address decoding are documented in output.rs beside
/// BLEND_SEPARATE_ALPHA_RESET. This is the initialized context presented to
/// applications, rather than the register file before firmware initialization.
/// https://gitlab.com/kernel-firmware/linux-firmware/-/blob/main/WHENCE
/// https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/nouveau/nvkm/engine/gr/gk20a.c#L110-L149
pub(super) const MAXWELL_THREE_D_POLYGON_MODE_RESET: u32 = 0x1b02;

/// Pipeline headers and constant-buffer groups in NVIDIA's initialized
/// context. Vertex B and pixel stages are enabled; the remaining stage slots
/// retain their matching type fields while disabled. The public method table
/// provenance, hash and decoding are documented beside the polygon reset.
/// https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/nouveau/nvkm/engine/gr/gk20a.c#L110-L149
pub(super) const MAXWELL_THREE_D_PIPELINE_SHADER_RESET: [u32; 6] =
    [0, 0x11, 0x20, 0x30, 0x40, 0x51];
pub(super) const MAXWELL_THREE_D_PIPELINE_BINDING_RESET: [u32; 6] = [0, 0, 1, 2, 3, 4];

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
        // GM20B method initialization explicitly clears scratch words 0..127.
        // Table SHA-256: 6372e2f6f547d7bd086beb066ce46ac379795f95c38eeeedd9e13c7b32515b24.
        // Each entry encodes (method_dword << 16) | class, followed by value;
        // B197 entries 0x0d00..0x0d7f are all zero. The upper half is not
        // covered by this table and remains unknown until programmed.
        // https://gitlab.com/kernel-firmware/linux-firmware/-/blob/main/WHENCE
        // https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/nouveau/nvkm/engine/gr/gk20a.c#L110-L149
        // The same GM20B table disables sparse depth (B197 dword 0x482).
        0x1208 | 0x15b8 | 0x190..=0x1ac => Some(0),
        raw @ 0x3400..=0x35fc if raw.is_multiple_of(4) => Some(0),
        raw if raw >= 0x0c00 && raw < 0x0d00 && (raw - 0x0c00) % 0x10 == 8 => {
            Some(VIEWPORT_CLIP_MIN_Z_RESET)
        }
        raw if raw >= 0x0c00 && raw < 0x0d00 && (raw - 0x0c00) % 0x10 == 12 => {
            Some(VIEWPORT_CLIP_MAX_Z_RESET)
        }
        raw if raw >= BLEND_SEPARATE_ALPHA_BASE
            && raw
                < BLEND_SEPARATE_ALPHA_BASE
                    + MAXWELL_COLOR_TARGET_COUNT as u32 * BLEND_TARGET_STRIDE
            && (raw - BLEND_SEPARATE_ALPHA_BASE).is_multiple_of(BLEND_TARGET_STRIDE) =>
        {
            Some(BLEND_SEPARATE_ALPHA_RESET)
        }
        // The common selector is initialized alongside the per-target ones.
        // Firmware table provenance and hash are documented in output.rs.
        0x133c => Some(BLEND_SEPARATE_ALPHA_RESET),
        // The same pinned GM20B table writes (0x04d7b197, 0) at byte 0x2780:
        // SET_SINGLE_ROP_CONTROL starts with independent target enables.
        0x135c => Some(0),
        // The adjacent entry at table offset 0x2788 is (0x03e4b197, 0):
        // SET_SINGLE_CT_WRITE_CONTROL selects each target's own color mask.
        0x0f90 => Some(0),
        // GM20B sw_method_init.bin, SHA-256 6372e2f6f547d7bd086beb066ce46ac379795f95c38eeeedd9e13c7b32515b24:
        // byte offsets 0x2490/0x2498 contain (0x064bb197, 1) and (0x064fb197, 0x2c1c).
        // These defaults must also be visible to MME shadow-register reads.
        // https://gitlab.com/kernel-firmware/linux-firmware/-/blob/main/nvidia/gm20b/gr/sw_method_init.bin
        // https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/nouveau/nvkm/engine/gr/gk20a.c#L110-L149
        0x192c => Some(1),
        0x193c => Some(0x2c1c),
        // SET_VIEWPORT_Z_CLIP, table byte offset 0x24a0.
        0x0d7c => Some(0),
        // GM20B table at byte offset 0x1c90: target count one, identity mapping
        // for all eight selectors (B197 dword 0x487).
        0x121c => Some(0x0fac_6881),
        // Adjacent table entry at 0x1ca0 enables the single depth target.
        0x1538 => Some(1),
        // Auxiliary blit/raster state restored by MME. Each value is verified
        // in the same GM20B context table, not inferred from a submitted replay
        // payload: 0x24c8/0x24d0 (surface clip), 0x1d10/0x1d50 (AA controls),
        // 0x1d28 (AA mode), 0x1430 (scissor), 0x1c78 (render override), and
        // raster/test selectors
        // selectors at 0x22f0, 0x2450, 0x21e0, 0x2758, 0x25b0, 0x21e8,
        // 0x2408, 0x2400, 0x25b8, 0x25d8, 0x2690, 0x25e0, 0x25f8,
        // 0x2678 and 0x2770. Unknown methods still have no reset.
        0x0210 | 0x0744 | 0x0dc8 | 0x0dd0 | 0x0e00 | 0x0f1c | 0x0f40 | 0x0fb4 | 0x0ff4 | 0x0ff8
        | 0x1148 | 0x11f8 | 0x1224 | 0x12cc | 0x12ec | 0x1380 | 0x1534 | 0x153c | 0x15d0
        | 0x1944 | 0x19bc | 0x19c4 => Some(0),
        // Raster enable (0x23f0), hybrid AA (0x1d18), attribute format
        // (0x10b0), statistics (0x1c88), sample masks (0x1d68..0x1d80), positions
        // (0x1d30..0x1d48), viewport clip bounds (0x11b0/0x1230).
        0x037c | 0x0754 => Some(1),
        0x0a18 => Some(0x6420),
        0x0d68 | 0x0fbc | 0x0fc0 | 0x0fc4 | 0x0fc8 => Some(0xffff),
        0x11e0 | 0x11e4 | 0x11e8 | 0x11ec => Some(0x8888_8888),
        raw if raw >= 0x0c00
            && raw < 0x0d00
            && (raw - 0x0c00) % 0x10 < 8
            && raw.is_multiple_of(4) =>
        {
            Some(0)
        }
        // GM20B table offsets 0x0000..0x0238 initialize all eight color-target
        // descriptors (only target zero has a nonzero format); MME restores a
        // temporarily overridden attachment through shadow replay.
        raw if raw >= 0x0800 && raw < 0x0a00 && raw.is_multiple_of(4) => {
            match (raw - 0x0800) % 0x40 {
                0 | 4 | 0x1c | 0x20 => Some(0),
                8 => Some(0x400),
                0x0c => Some(0x300),
                0x10 => Some(if raw == 0x0810 { 0xcf } else { 0 }),
                0x14 => Some(0x40),
                0x18 => Some(1),
                _ => None,
            }
        }
        // GM200 method initialization: disabled culling, clockwise front
        // faces, back-face cull selector, and target-zero RGBA write enable.
        // The same public table provenance as the polygon context above.
        0x1918 => Some(0),
        0x191c => Some(0x900),
        0x1920 => Some(0x405),
        0x1a00 => Some(0x1111),
        raw @ 0x1360..=0x137c if raw.is_multiple_of(4) => Some(0),
        raw if raw >= 0x1a04 && raw <= 0x1a1c && raw.is_multiple_of(4) => Some(0),
        // Typed line-state defaults and sources live in line.rs.
        0x0f8c | 0x166c => Some(0),
        MAXWELL_THREE_D_FRONT_POLYGON_MODE_METHOD | MAXWELL_THREE_D_BACK_POLYGON_MODE_METHOD => {
            Some(MAXWELL_THREE_D_POLYGON_MODE_RESET)
        }
        MAXWELL_THREE_D_WINDOW_ORIGIN_METHOD => Some(MAXWELL_THREE_D_WINDOW_ORIGIN_RESET),
        MAXWELL_THREE_D_DEPTH_TARGET_LAYER_METHOD => Some(MAXWELL_THREE_D_DEPTH_TARGET_LAYER_RESET),
        raw if raw >= MAXWELL_THREE_D_COLOR_COMPRESSION_BASE_METHOD
            && raw
                < MAXWELL_THREE_D_COLOR_COMPRESSION_BASE_METHOD
                    + MAXWELL_COLOR_TARGET_COUNT as u32
                        * MAXWELL_THREE_D_COLOR_COMPRESSION_STRIDE
            && (raw - MAXWELL_THREE_D_COLOR_COMPRESSION_BASE_METHOD)
                .is_multiple_of(MAXWELL_THREE_D_COLOR_COMPRESSION_STRIDE) =>
        {
            Some(MAXWELL_THREE_D_COLOR_COMPRESSION_RESET)
        }
        raw if raw >= MAXWELL_THREE_D_COLOR_TARGET_BASE_METHOD
            && raw
                < MAXWELL_THREE_D_COLOR_TARGET_BASE_METHOD
                    + MAXWELL_COLOR_TARGET_COUNT as u32 * MAXWELL_THREE_D_COLOR_TARGET_STRIDE =>
        {
            let register = (raw - MAXWELL_THREE_D_COLOR_TARGET_BASE_METHOD)
                % MAXWELL_THREE_D_COLOR_TARGET_STRIDE;
            if register == MAXWELL_THREE_D_COLOR_TARGET_LAYER_OFFSET {
                Some(MAXWELL_THREE_D_COLOR_TARGET_LAYER_RESET)
            } else {
                None
            }
        }
        raw if raw >= MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD => {
            let offset = raw - MAXWELL_THREE_D_PIPELINE_SHADER_BASE_METHOD;
            let pipeline = offset / MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE;
            let register = offset % MAXWELL_THREE_D_PIPELINE_SHADER_STRIDE;
            if pipeline < MAXWELL_PIPELINE_SHADER_COUNT as u32 {
                match register {
                    0 => Some(MAXWELL_THREE_D_PIPELINE_SHADER_RESET[pipeline as usize]),
                    0x10 => Some(MAXWELL_THREE_D_PIPELINE_BINDING_RESET[pipeline as usize]),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Decoded register bits and value. Command provenance stays at execution boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDRegister<T> {
    raw: Option<u32>,
    value: Option<T>,
}

impl<T> MaxwellThreeDRegister<T> {
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
}

impl<T> Default for MaxwellThreeDRegister<T> {
    fn default() -> Self {
        Self {
            raw: None,
            value: None,
        }
    }
}

impl<T> MaxwellThreeDRegister<T> {
    pub(super) const fn verified_reset(raw: u32, value: Option<T>) -> Self {
        Self {
            raw: Some(raw),
            value,
        }
    }

    pub(super) const fn decoded(raw: u32, value: T) -> Self {
        Self {
            raw: Some(raw),
            value: Some(value),
        }
    }
}

/// The fixed class register file keeps consumed raw words dense.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MaxwellThreeDRawRegisterFile {
    values: Box<[u32]>,
    valid: [u64; 64],
}

impl Default for MaxwellThreeDRawRegisterFile {
    fn default() -> Self {
        let mut registers = Self {
            values: vec![0; 0x1000].into_boxed_slice(),
            valid: [0; 64],
        };
        for index in 0..4096 {
            if let Some(value) = verified_raw_register_reset(GpuMethodId(index * 4)) {
                registers.set_reset(GpuMethodId(index * 4), value);
            }
        }
        registers
    }
}

impl MaxwellThreeDRawRegisterFile {
    pub(super) fn raw(&self, method: GpuMethodId) -> Option<u32> {
        if !method.0.is_multiple_of(4) {
            return None;
        }
        let index = (method.0 / 4) as usize;
        (index < self.values.len() && self.valid[index / 64] & (1 << (index % 64)) != 0)
            .then(|| self.values[index])
    }

    #[inline]
    pub(super) fn read<T>(
        &self,
        method: u32,
        decode: impl FnOnce(u32) -> Option<T>,
    ) -> MaxwellThreeDRegister<T> {
        let raw = self.raw(GpuMethodId(method));
        let value = raw.and_then(decode);
        MaxwellThreeDRegister { raw, value }
    }

    fn set_reset(&mut self, method: GpuMethodId, value: u32) {
        let index = (method.0 / 4) as usize;
        self.values[index] = value;
        self.valid[index / 64] |= 1 << (index % 64);
    }

    pub(super) fn write(&mut self, method: GpuMethodId, value: u32) {
        let index = (method.0 / 4) as usize;
        self.values[index] = value;
        self.valid[index / 64] |= 1 << (index % 64);
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

/// Direct read-only interpretation of the authoritative raster registers.
#[derive(Clone, Copy)]
pub struct MaxwellThreeDRasterState<'a> {
    registers: &'a MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDRasterState<'_> {
    pub fn point_size(&self) -> MaxwellThreeDRegister<MaxwellThreeDPointSize> {
        self.registers
            .read(0x1518, |raw| Some(MaxwellThreeDPointSize::from_bits(raw)))
    }
    pub fn attribute_point_size(&self) -> MaxwellThreeDRegister<MaxwellThreeDAttributePointSize> {
        self.registers
            .read(0x1910, MaxwellThreeDAttributePointSize::parse)
    }
    pub fn point_sprite_enable(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x1520, |raw| (raw <= 1).then_some(raw != 0))
    }
    pub fn anti_aliased_point_enable(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x1658, |raw| (raw <= 1).then_some(raw != 0))
    }
    pub fn point_sprite_select(&self) -> MaxwellThreeDRegister<MaxwellThreeDPointSpriteSelect> {
        self.registers
            .read(0x1604, MaxwellThreeDPointSpriteSelect::parse)
    }
    pub fn point_center_mode(&self) -> MaxwellThreeDRegister<MaxwellThreeDPointCenterMode> {
        self.registers
            .read(0x165c, MaxwellThreeDPointCenterMode::parse)
    }
    pub fn edge_flag(&self) -> MaxwellThreeDRegister<MaxwellThreeDEdgeFlag> {
        self.registers.read(0x15e4, MaxwellThreeDEdgeFlag::parse)
    }
    pub fn alpha_fraction(&self) -> MaxwellThreeDRegister<MaxwellThreeDAlphaFraction> {
        self.registers.read(0x074c, |raw| {
            (raw <= 255).then_some(MaxwellThreeDAlphaFraction::new(raw as u8))
        })
    }
    pub fn bounding_box(&self) -> MaxwellThreeDRegister<MaxwellThreeDRasterBoundingBox> {
        self.registers.read(0x02ec, |raw| {
            (raw & !0xff1 == 0).then_some(MaxwellThreeDRasterBoundingBox::parse(raw))
        })
    }
    pub fn fill_via_triangle(&self) -> MaxwellThreeDRegister<MaxwellThreeDFillViaTriangleMode> {
        self.registers
            .read(0x113c, MaxwellThreeDFillViaTriangleMode::parse)
    }
    pub fn conservative_raster(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDConservativeRasterEnable> {
        self.registers
            .read(0x1148, MaxwellThreeDConservativeRasterEnable::parse)
    }
    pub fn polygon_smooth_enable(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x0db4, |raw| (raw <= 1).then_some(raw != 0))
    }
    pub fn polygon_stipple_enable(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x168c, |raw| (raw <= 1).then_some(raw != 0))
    }
    pub fn polygon_stipple_word(&self, index: usize) -> MaxwellThreeDRegister<u32> {
        assert!(index < MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT);
        self.registers.read(0x1700 + index as u32 * 4, Some)
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDViewportState<'a> {
    pub(super) registers: &'a MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDViewportState<'_> {
    #[must_use]
    pub fn z_clip_range(&self) -> MaxwellThreeDRegister<MaxwellThreeDViewportZClipRange> {
        self.registers
            .read(0x0d7c, MaxwellThreeDViewportZClipRange::parse)
    }
    #[must_use]
    pub fn pixel_center(&self) -> MaxwellThreeDRegister<MaxwellThreeDViewportPixelCenter> {
        self.registers
            .read(0x1924, MaxwellThreeDViewportPixelCenter::parse)
    }
}

/// Authoritative class registers, renderer dirty domains and operational state.
/// Consumers interpret only the fields required by the operation they execute.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDState {
    registers: MaxwellThreeDRawRegisterFile,
    dirty_domains: u16,
    dirty_vertices: u32,
    dirty_colors: u8,
    inline_index: Option<Arc<[u8]>>,
    // Revisions are local to one register-file lifetime. Retain a namespace
    // across snapshots so shared caches cannot confuse two channels whose
    // independent revision counters happen to have the same numeric value.
    cache_scope: Arc<()>,
    draw_state_revision: u64,
    draw_parameters_revision: u64,
    color_target_revisions: [u64; 8],
    depth_target_revision: u64,
    vertex_slots: [u64; 32],
    index_revision: u64,
    constant_slots: [[u64; 32]; 8],
    constant_buffer_revision: u64,
    resource_group_revisions: [u64; 4],
    texture_revision: u64,
    texture_header_revision: u64,
    sampler_revision: u64,
    sample_mode_revision: u64,
    shader_revision: u64,
    primitive_sequence: super::vertex::MaxwellThreeDPrimitiveSequence,
    shader_resources: super::bindings::MaxwellThreeDShaderResources,
    falcon: Box<MaxwellThreeDFalconState>,
    inline_transfer: Option<super::inline_to_memory::PendingTransfer>,
}

/// Retained identity of the state domains which determine resource bindings.
///
/// State domains unrelated to resource resolution deliberately do not advance
/// the retained revisions or invalidate this identity.
#[derive(Clone, Debug)]
pub(crate) struct MaxwellThreeDResourceStateIdentity {
    scope: Arc<()>,
    role: MaxwellThreeDResourceRole,
    revision: u64,
    constant_buffers: [Option<(u8, u8, u64)>; 2],
    sample_mode: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct MaxwellThreeDResourceGroupIdentity {
    scope: Arc<()>,
    revision: u64,
}

impl MaxwellThreeDResourceGroupIdentity {
    pub(crate) fn matches(&self, state: &MaxwellThreeDState, group: usize) -> bool {
        Arc::ptr_eq(&self.scope, &state.cache_scope)
            && self.revision == state.resource_group_revisions[group]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MaxwellThreeDShaderStateIdentity {
    scope: Arc<()>,
    revision: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct MaxwellThreeDDrawStateIdentity {
    scope: Arc<()>,
    revision: u64,
    parameters_revision: u64,
}

impl MaxwellThreeDDrawStateIdentity {
    pub(crate) fn matches_parameters(&self, state: &MaxwellThreeDState) -> bool {
        Arc::ptr_eq(&self.scope, &state.cache_scope)
            && self.parameters_revision == state.draw_parameters_revision
    }
    pub(crate) fn matches(&self, state: &MaxwellThreeDState) -> bool {
        Arc::ptr_eq(&self.scope, &state.cache_scope) && self.revision == state.draw_state_revision
    }
}

/// Primitive class consumed by raster state, independent of assembly/winding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::engines) enum GeneratedPrimitive {
    Points,
    Lines,
    Triangles,
}

impl MaxwellThreeDShaderStateIdentity {
    pub(crate) fn matches_code(&self, state: &MaxwellThreeDState) -> bool {
        Arc::ptr_eq(&self.scope, &state.cache_scope) && self.revision == state.shader_revision
    }
}

impl MaxwellThreeDResourceStateIdentity {
    pub(crate) fn matches(&self, state: &MaxwellThreeDState) -> bool {
        Arc::ptr_eq(&self.scope, &state.cache_scope)
            && self.revision == state.binding_revision(self.role)
            && self
                .constant_buffers
                .iter()
                .flatten()
                .all(|&(group, slot, rev)| {
                    state.constant_slots[group as usize][slot as usize] == rev
                })
            && self
                .sample_mode
                .is_none_or(|revision| revision == state.sample_mode_revision)
    }
}

/// Complete currently modeled live state of one channel's `MAXWELL_B` engine.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub(crate) struct MaxwellThreeDFrontendState {
    pub(super) instance_run: Option<super::draw::InstanceRun>,
    deferred_draw_end: Option<MaxwellThreeDVertexInputWrite>,
    inline_indices: Vec<u8>,
    inline_end: Option<MaxwellMethodSource>,
    pub(in crate::engines) pending_notification: Option<(u64, MaxwellMethodSource)>,
    operation: MaxwellThreeDState,
    mme: MaxwellThreeDMmeState,
}

impl MaxwellThreeDFrontendState {
    pub(super) fn has_inline_indices(&self) -> bool {
        !self.inline_indices.is_empty()
    }

    pub(super) fn append_inline_indices(
        &mut self,
        source: MaxwellMethodSource,
    ) -> Result<(), super::MaxwellEngineDispatchError> {
        self.append_index_values(source, std::iter::once(source.argument()))
    }

    pub(super) fn append_index_values(
        &mut self,
        source: MaxwellMethodSource,
        words: impl ExactSizeIterator<Item = u32>,
    ) -> Result<(), super::MaxwellEngineDispatchError> {
        if self
            .operation
            .vertex_input()
            .primitive()
            .active_begin()
            .is_none()
        {
            return Err(super::invalid_encoding(
                source,
                "INLINE_INDEX",
                "inline indices require an active BEGIN",
            ));
        }
        let (lanes, bits) = match source.method().0 {
            0x1300 => (4, 8),
            0x15ec => (2, 16),
            0x15e8 => (1, 32),
            _ => unreachable!("inline index aperture"),
        };
        let additional = words
            .len()
            .checked_mul(lanes)
            .ok_or(super::MaxwellEngineDispatchError::ResourceExhausted)?;
        if additional > u32::MAX as usize
            || self.inline_indices.len() / 4 > u32::MAX as usize - additional
        {
            return Err(super::MaxwellEngineDispatchError::ResourceExhausted);
        }
        self.inline_indices
            .try_reserve(
                additional
                    .checked_mul(4)
                    .ok_or(super::MaxwellEngineDispatchError::ResourceExhausted)?,
            )
            .map_err(|_| super::MaxwellEngineDispatchError::ResourceExhausted)?;
        let mask = u32::MAX >> (32 - bits);
        for word in words {
            for lane in 0..lanes {
                let value = (word >> (lane * bits)) & mask;
                self.inline_indices.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    }

    pub(super) fn publish_inline_indices(
        &mut self,
        source: MaxwellMethodSource,
    ) -> Result<u32, super::MaxwellEngineDispatchError> {
        if self
            .operation
            .vertex_input()
            .primitive()
            .restart_enabled()
            .value()
            == Some(&true)
        {
            let restart = self
                .operation
                .vertex_input()
                .primitive()
                .restart_index()
                .raw()
                .ok_or_else(|| {
                    super::invalid_encoding(
                        source,
                        "INLINE_INDEX",
                        "primitive restart requires a programmed restart index",
                    )
                })?;
            for word in self.inline_indices.chunks_exact_mut(4) {
                if u32::from_le_bytes(word.try_into().unwrap()) == restart {
                    word.copy_from_slice(&u32::MAX.to_le_bytes());
                }
            }
        }
        let count = (self.inline_indices.len() / 4) as u32;
        self.operation.inline_index = Some(Arc::from(self.inline_indices.as_slice()));
        self.inline_end = Some(source);
        Ok(count)
    }

    pub(in crate::engines) fn complete_inline_draw(&mut self) {
        if let Some(source) = self.inline_end.take() {
            self.operation.inline_index = None;
            self.inline_indices.clear();
            self.apply(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::End {
                    value: source.argument() != 0,
                    source,
                },
            ));
        }
    }
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn operation_state(&self) -> &MaxwellThreeDState {
        &self.operation
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
    pub(in crate::engines) fn raw_register_value(&self, method: GpuMethodId) -> Option<u32> {
        self.operation.registers.raw(method)
    }

    pub(super) fn write_register(&mut self, method: GpuMethodId, argument: u32) {
        if self.operation.registers.raw(method) == Some(argument) {
            return;
        }
        self.operation.registers.write(method, argument);
        let domains = super::registers::domains(method);
        self.operation.dirty_domains |= domains;
        if domains & super::registers::VERTEX != 0 {
            let index = if method.0 >= 0x1f00 {
                (method.0 - 0x1f00) / 8
            } else {
                (method.0 - 0x1c00) / 16
            };
            self.operation.dirty_vertices |= 1 << index;
        }
        if domains & super::registers::COLOR != 0 {
            self.operation.dirty_colors |= if method.0 == 0x15b8 {
                0xff
            } else if method.0 >= 0x19e0 {
                1 << ((method.0 - 0x19e0) / 4)
            } else {
                1 << ((method.0 - 0x0800) / 0x40)
            };
        }
    }
    pub(super) fn record_raw_register(&mut self, source: MaxwellMethodSource) {
        self.write_register(source.method(), source.argument());
    }

    pub(super) fn flush_dirty(&mut self) {
        use super::registers as D;
        let state = &mut self.operation;
        let domains = std::mem::take(&mut state.dirty_domains);
        if domains & D::DRAW != 0 {
            advance_revision(&mut state.draw_state_revision);
        }
        if domains & D::PARAMETERS != 0 {
            advance_revision(&mut state.draw_parameters_revision);
        }
        if domains & D::SHADERS != 0 {
            advance_revision(&mut state.shader_revision);
        }
        if domains & (D::VERTEX | D::INDEX) != 0 {
            advance_revision(&mut state.resource_group_revisions[0]);
        }
        let mut vertices = std::mem::take(&mut state.dirty_vertices);
        while vertices != 0 {
            let index = vertices.trailing_zeros() as usize;
            vertices &= vertices - 1;
            advance_revision(&mut state.vertex_slots[index]);
        }
        if domains & D::INDEX != 0 {
            advance_revision(&mut state.index_revision);
        }
        if domains & (D::COLOR | D::DEPTH | D::SAMPLE_MODE) != 0 {
            advance_revision(&mut state.resource_group_revisions[3]);
        }
        let mut colors = std::mem::take(&mut state.dirty_colors);
        while colors != 0 {
            let index = colors.trailing_zeros() as usize;
            colors &= colors - 1;
            advance_revision(&mut state.color_target_revisions[index]);
        }
        if domains & D::DEPTH != 0 {
            advance_revision(&mut state.depth_target_revision);
        }
        if domains & D::SAMPLE_MODE != 0 {
            advance_revision(&mut state.sample_mode_revision);
        }
        if domains & (D::HEADERS | D::SAMPLERS) != 0 {
            advance_revision(&mut state.texture_revision);
            advance_revision(&mut state.resource_group_revisions[2]);
        }
        if domains & D::HEADERS != 0 {
            advance_revision(&mut state.texture_header_revision);
        }
        if domains & D::SAMPLERS != 0 {
            advance_revision(&mut state.sampler_revision);
        }
    }
}

impl MaxwellThreeDState {
    pub(crate) fn inline_index(&self) -> Option<&Arc<[u8]>> {
        self.inline_index.as_ref()
    }
    pub(crate) fn draw_state_identity(&self) -> MaxwellThreeDDrawStateIdentity {
        MaxwellThreeDDrawStateIdentity {
            scope: Arc::clone(&self.cache_scope),
            revision: self.draw_state_revision,
            parameters_revision: self.draw_parameters_revision,
        }
    }

    fn binding_revision(&self, role: MaxwellThreeDResourceRole) -> u64 {
        use MaxwellThreeDResourceRole as R;
        match role {
            R::VertexStream(slot) => self.vertex_slots[slot as usize],
            R::IndexBuffer => self.index_revision,
            R::ConstantBuffer { group, slot } | R::StorageBuffer { group, slot, .. } => {
                self.constant_slots[group as usize][slot as usize]
            }
            R::ColorTarget(slot) => self.color_target_revisions[slot as usize],
            R::DepthStencilTarget => self.depth_target_revision,
            R::SampledImage { .. } | R::TextureHeaders => self.texture_header_revision,
            R::Sampler(_) | R::Samplers => self.sampler_revision,
            R::BlitSource | R::BlitDestination => 0,
        }
    }
    pub(crate) fn resource_state_identity(
        &self,
        role: MaxwellThreeDResourceRole,
    ) -> MaxwellThreeDResourceStateIdentity {
        use MaxwellThreeDResourceRole as R;
        let constant_buffers = match role {
            R::SampledImage { texture, .. } | R::Sampler(texture) => {
                let binding = |slot: u8| {
                    (
                        texture.group,
                        slot,
                        self.constant_slots[texture.group as usize][slot as usize],
                    )
                };
                [
                    Some(binding(texture.constant_buffer_slot)),
                    texture.secondary.map(|(slot, _, _)| binding(slot)),
                ]
            }
            _ => [None; 2],
        };
        MaxwellThreeDResourceStateIdentity {
            scope: Arc::clone(&self.cache_scope),
            role,
            revision: self.binding_revision(role),
            constant_buffers,
            sample_mode: matches!(role, R::ColorTarget(_) | R::DepthStencilTarget)
                .then_some(self.sample_mode_revision),
        }
    }

    pub(crate) fn resource_group_identity(
        &self,
        group: usize,
    ) -> MaxwellThreeDResourceGroupIdentity {
        MaxwellThreeDResourceGroupIdentity {
            scope: Arc::clone(&self.cache_scope),
            revision: self.resource_group_revisions[group],
        }
    }

    pub(crate) fn shader_state_identity(&self) -> MaxwellThreeDShaderStateIdentity {
        MaxwellThreeDShaderStateIdentity {
            scope: Arc::clone(&self.cache_scope),
            revision: self.shader_revision,
        }
    }

    #[must_use]
    pub fn render_targets(&self) -> MaxwellThreeDRenderTargetState<'_> {
        MaxwellThreeDRenderTargetState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn fixed_function(&self) -> MaxwellThreeDFixedFunctionState<'_> {
        MaxwellThreeDFixedFunctionState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn vertex_input(&self) -> MaxwellThreeDVertexInputState<'_> {
        MaxwellThreeDVertexInputState {
            registers: &self.registers,
            sequence: &self.primitive_sequence,
        }
    }

    pub fn shader_bindings(&self) -> MaxwellThreeDShaderBindingState<'_> {
        MaxwellThreeDShaderBindingState {
            registers: &self.registers,
            resources: &self.shader_resources,
        }
    }

    #[must_use]
    pub fn raster(&self) -> MaxwellThreeDRasterState<'_> {
        MaxwellThreeDRasterState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn viewport(&self) -> MaxwellThreeDViewportState<'_> {
        MaxwellThreeDViewportState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn render_enable(&self) -> MaxwellThreeDRenderEnableState<'_> {
        MaxwellThreeDRenderEnableState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn shader_execution(&self) -> MaxwellThreeDShaderExecutionState<'_> {
        MaxwellThreeDShaderExecutionState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn color_reduction(&self) -> MaxwellThreeDColorReductionState<'_> {
        MaxwellThreeDColorReductionState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn coverage(&self) -> MaxwellThreeDCoverageState<'_> {
        MaxwellThreeDCoverageState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn line(&self) -> MaxwellThreeDLineState<'_> {
        MaxwellThreeDLineState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn report_semaphore(&self) -> MaxwellThreeDReportSemaphoreState<'_> {
        MaxwellThreeDReportSemaphoreState {
            registers: &self.registers,
        }
    }

    #[must_use]
    pub fn falcon(&self) -> &MaxwellThreeDFalconState {
        self.falcon.as_ref()
    }

    #[must_use]
    pub fn inline_to_memory(&self) -> MaxwellThreeDInlineToMemoryState<'_> {
        MaxwellThreeDInlineToMemoryState {
            registers: &self.registers,
            pending: &self.inline_transfer,
        }
    }

    /// Patches consume the tessellator's output primitive, not their input topology.
    pub(in crate::engines) fn generated_primitive(&self) -> Option<GeneratedPrimitive> {
        use GeneratedPrimitive as Output;
        match self.vertex_input().primitive().active_begin()?.topology() {
            0 => Some(Output::Points),
            1 | 3 => Some(Output::Lines),
            4..=7 => Some(Output::Triangles),
            14 => self
                .shader_bindings()
                .tessellation_mode()
                .value()?
                .lower()
                .ok()
                .map(|mode| match mode.output {
                    nixe_gpu::TessellationOutput::Points => Output::Points,
                    nixe_gpu::TessellationOutput::Lines => Output::Lines,
                    nixe_gpu::TessellationOutput::Triangles(_) => Output::Triangles,
                }),
            _ => None,
        }
    }
}

impl MaxwellThreeDFrontendState {
    fn apply_primitive(&mut self, write: MaxwellThreeDVertexInputWrite) {
        match write {
            MaxwellThreeDVertexInputWrite::Begin { value, source } => self
                .operation
                .primitive_sequence
                .begin(source.argument(), value),
            MaxwellThreeDVertexInputWrite::End { value, source } => self
                .operation
                .primitive_sequence
                .end(source.argument(), value),
        }
    }
    pub(super) fn complete_draw_end(&mut self) {
        if let Some(write) = self.deferred_draw_end.take() {
            self.apply_primitive(write);
        }
    }
    pub(super) fn apply(&mut self, write: MaxwellThreeDStateWrite) {
        // END terminates the pending draw, whose capture still consumes its
        // validated BEGIN. Apply that termination after the draw consumer
        // or before the next BEGIN; raw/shadow register writes remain immediate.
        if self.instance_run.is_some()
            && let MaxwellThreeDStateWrite::VertexInput(
                end @ MaxwellThreeDVertexInputWrite::End { .. },
            ) = write
        {
            self.deferred_draw_end = Some(end);
            return;
        }
        match write {
            MaxwellThreeDStateWrite::InlineToMemory(write) => match write {
                MaxwellThreeDInlineToMemoryStateWrite::Launch {
                    address,
                    byte_length,
                    ..
                } => {
                    self.operation.inline_transfer =
                        (byte_length != 0).then_some(super::inline_to_memory::PendingTransfer {
                            address,
                            byte_length,
                            next_offset: 0,
                        });
                }
                MaxwellThreeDInlineToMemoryStateWrite::Data { next_offset, .. } => {
                    let pending = self
                        .operation
                        .inline_transfer
                        .as_mut()
                        .expect("validated inline transfer");
                    pending.next_offset = next_offset;
                    if next_offset >= pending.byte_length {
                        self.operation.inline_transfer = None;
                    }
                }
            },
            MaxwellThreeDStateWrite::VertexInput(write) => self.apply_primitive(write),
            MaxwellThreeDStateWrite::ShaderBinding(write) => {
                if let MaxwellThreeDShaderBindingWrite::BindConstantBuffer {
                    group,
                    slot,
                    enabled,
                    address,
                    size,
                    ..
                } = write
                {
                    let previous = self.operation.shader_bindings().groups()[group as usize]
                        .constant_buffers()[slot as usize];
                    if previous.is_none_or(|p| {
                        p.enabled() != enabled || p.address() != address || p.size() != size
                    }) {
                        advance_revision(
                            &mut self.operation.constant_slots[group as usize][slot as usize],
                        );
                        advance_revision(&mut self.operation.constant_buffer_revision);
                        advance_revision(&mut self.operation.resource_group_revisions[1]);
                        advance_revision(&mut self.operation.resource_group_revisions[2]);
                    }
                }
                self.operation.shader_resources.apply(write);
            }
            MaxwellThreeDStateWrite::FalconMaskedRegister(write) => {
                self.operation.falcon.apply(write);
                self.mme.apply(MaxwellThreeDMmeStateWrite::ShadowScratch {
                    index: MaxwellThreeDMmeShadowScratchIndex::new(0),
                    value: 1,
                    source: write.source(),
                });
                self.operation.registers.write(GpuMethodId(0x3400), 1);
            }
            MaxwellThreeDStateWrite::Mme(write) => self.mme.apply(write),
        }
    }
}

fn advance_revision(revision: &mut u64) {
    *revision = revision
        .checked_add(1)
        .expect("Maxwell semantic revision exhausted");
}

/// One checked `MAXWELL_B` register transition ready for direct application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDStateWrite {
    InlineToMemory(MaxwellThreeDInlineToMemoryStateWrite),
    VertexInput(MaxwellThreeDVertexInputWrite),
    ShaderBinding(MaxwellThreeDShaderBindingWrite),
    FalconMaskedRegister(MaxwellThreeDFalconMaskedRegisterWrite),
    Mme(MaxwellThreeDMmeStateWrite),
}

#[cfg(test)]
mod tests {
    use super::MaxwellThreeDFrontendState;

    #[test]
    fn viewport_depth_reset_is_shared_by_typed_live_state_and_mme_shadow_reads() {
        let frontend = MaxwellThreeDFrontendState::default();
        for index in 0..16 {
            let viewport = frontend.operation_state().fixed_function().viewport(index);
            for (offset, register, expected) in [
                (8, viewport.clip_min_z(), 0),
                (12, viewport.clip_max_z(), 1.0_f32.to_bits()),
            ] {
                assert_eq!(register.raw(), Some(expected));
                assert_eq!(register.value().map(|value| value.get()), Some(expected));

                assert_eq!(
                    super::verified_raw_register_reset(nixe_gpu::GpuMethodId(
                        0x0c00 + index as u32 * 16 + offset
                    )),
                    Some(expected)
                );
            }
            assert!((0..3).all(|axis| viewport.scale(axis).value().is_none()));
        }
    }
}
