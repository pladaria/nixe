//! Maxwell image/descriptor interpretation. Native resource ownership and
//! persistent selections belong to the renderer caches.
//!
//! Maxwell GPU virtual addresses and PTE kinds are interpreted here, while
//! canonical storage and neutral resource views retain their own identities.
//! No host layout, swizzle operation, or backend object crosses this boundary.

mod attachment;
use nixe_gpu::native::{ImageInfo, ImagePacking};
mod targets;
pub use attachment::{
    AttachmentInfo, attachment_description, blit_image_infos, render_target_info, solid_image_info,
};
pub mod texture;
use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{
    AddressMode, BlockLinearLayout, ComponentSwizzle, FilterMode, ImageDescription, ImageDimension,
    ImageExtent, ImageFormat, ImageKind, ImageMemoryLayout, SampleCount, SamplerDescription,
    Swizzle,
};

use crate::{MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellMappingId};

use super::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MaxwellThreeDAttachmentReadiness,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetState, MaxwellThreeDDepthArrayControl,
    MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState,
    MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionValue, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDSampleMode, MaxwellThreeDState,
    MaxwellThreeDUnresolvedAddress,
};

// Single-sample 2CZ depth kinds preserve distinct guest depth/stencil packing.
// They require materialization, not a generic block-linear byte import.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h#L35-L100
// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.cpp#L35-L73
const MAXWELL_S8Z24_2CZ_KIND: u8 = 0x17;
const MAXWELL_Z24S8_2CZ_KIND: u8 = 0x51;
// Four-sample kinds selected by deko3d. Their bytes are never imported as
// single-sample texels, even when compression of subsequent writes is off.
// https://github.com/devkitPro/deko3d/blob/master/source/maxwell/image_formats.cpp
// https://github.com/switchbrew/libnx/blob/master/nx/include/switch/nvidia/types.h
const MAXWELL_C32_MS4_2CBR_KIND: u8 = 0xe0;

/// Frontend role of one completely resolved resource.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellThreeDResourceRole {
    SampledImage {
        texture: MaxwellThreeDTextureReference,
        dimension: MaxwellThreeDTextureDimension,
        with_sampler: bool,
    },
    Sampler(MaxwellThreeDTextureReference),
    ColorTarget(u8),
    DepthStencilTarget,
    BlitSource,
    BlitDestination,
}

/// Shader-visible dimensionality required from a Maxwell sampled-image TIC.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellThreeDTextureDimension {
    Two,
    Three,
    TwoArray,
    Cube,
    CubeArray,
    One,
    OneArray,
}

/// Draw-local location of a raw Maxwell TIC/TSC handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaxwellThreeDTextureReference {
    pub(super) group: u8,
    pub(super) constant_buffer_slot: u8,
    byte_offset: u32,
    shift_left: u8,
    pub(super) secondary: Option<(u8, u32, u8)>,
}

impl MaxwellThreeDTextureReference {
    #[must_use]
    pub const fn new(group: u8, constant_buffer_slot: u8, byte_offset: u32) -> Self {
        Self {
            group,
            constant_buffer_slot,
            byte_offset,
            shift_left: 0,
            secondary: None,
        }
    }

    pub fn expression(
        group: u8,
        primary: (u8, u32, u8),
        secondary: Option<(u8, u32, u8)>,
    ) -> Option<Self> {
        let valid = |(slot, _, shift): (u8, u32, u8)| {
            usize::from(slot) < MAXWELL_CONSTANT_BUFFER_SLOT_COUNT && shift < 32
        };
        if usize::from(group) >= MAXWELL_BIND_GROUP_COUNT
            || !valid(primary)
            || secondary.is_some_and(|part| !valid(part))
        {
            return None;
        }
        Some(Self {
            group,
            constant_buffer_slot: primary.0,
            byte_offset: primary.1,
            shift_left: primary.2,
            secondary,
        })
    }
}

/// Original Maxwell texel encoding, independent of native component semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDGuestImageFormat {
    /// Canonical texels authored by another device frontend (for example VIC).
    Canonical(ImageFormat),
    Color(MaxwellThreeDColorTargetFormat),
    DepthStencil(MaxwellThreeDDepthStencilFormat),
    Texture(u32),
}

/// Source-preserving neutral interpretation of one Maxwell TSC entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedSampler {
    role: MaxwellThreeDResourceRole,
    compare: Option<nixe_gpu::DepthCompareOperation>,
    reduction: nixe_gpu::SamplerReduction,
    min_filter: FilterMode,
    mag_filter: FilterMode,
    mip_filter: FilterMode,
    address_modes: [AddressMode; 3],
    border_color: Option<[u32; 4]>,
    lod_bias_fixed: i16,
    lod_min_fixed: u16,
    lod_max_fixed: u16,
    max_anisotropy: u8,
}

impl MaxwellThreeDResolvedSampler {
    #[must_use]
    pub const fn role(self) -> MaxwellThreeDResourceRole {
        self.role
    }

    pub fn description(self) -> Result<SamplerDescription, MaxwellThreeDResourceError> {
        SamplerDescription::new(
            self.min_filter,
            self.mag_filter,
            self.mip_filter,
            self.address_modes,
            f32::from(self.lod_min_fixed) / 256.0,
            f32::from(self.lod_max_fixed) / 256.0,
            f32::from(self.max_anisotropy),
        )
        .map(|mut description| {
            description.border_color = self.border_color;
            description.compare = self.compare;
            description.reduction = self.reduction;
            description.lod_bias = f32::from(self.lod_bias_fixed) / 256.0;
            description
        })
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role: self.role })
    }
}

struct SampledMipChain {
    layouts: Vec<(u64, ImageMemoryLayout)>,
    layer_stride: u64,
    layer_size: u64,
}

struct MaxwellImageDescriptionRequest {
    dimension: ImageDimension,
    extent: ImageExtent,
    format: ImageFormat,
    kind: ImageKind,
    layers: u16,
    role: MaxwellThreeDResourceRole,
}

// Preserve precision, numeric representation, transfer function and channel
// order for render targets and 2D resolves. The L components denote sRGB;
// alpha remains linear UNORM in those formats.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
fn color_image_format(
    format: u8,
    role: MaxwellThreeDResourceRole,
) -> Result<ImageFormat, MaxwellThreeDResourceError> {
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/surface.cpp
    match format {
        0xc1 => Ok(ImageFormat::Rgba32Sint),
        0xc2 => Ok(ImageFormat::Rgba32Uint),
        0xc4 => Ok(ImageFormat::Rgba32Sint),
        0xc5 => Ok(ImageFormat::Rgba32Uint),
        0xc8 => Ok(ImageFormat::Rgba16Sint),
        0xc9 => Ok(ImageFormat::Rgba16Uint),
        0xcc => Ok(ImageFormat::Rg32Sint),
        0xcd => Ok(ImageFormat::Rg32Uint),
        0xd8 => Ok(ImageFormat::Rgba8Sint),
        0xd9 => Ok(ImageFormat::Rgba8Uint),
        0xdc => Ok(ImageFormat::Rg16Sint),
        0xdd => Ok(ImageFormat::Rg16Uint),
        0xe3 => Ok(ImageFormat::R32Sint),
        0xe4 => Ok(ImageFormat::R32Uint),
        0xec => Ok(ImageFormat::Rg8Sint),
        0xed => Ok(ImageFormat::Rg8Uint),
        0xf0 => Ok(ImageFormat::R16Sint),
        0xf1 => Ok(ImageFormat::R16Uint),
        0xf5 => Ok(ImageFormat::R8Sint),
        0xf6 => Ok(ImageFormat::R8Uint),
        0xc6 => Ok(ImageFormat::Rgba16Unorm),
        0xc7 => Ok(ImageFormat::Rgba16Snorm),
        0xd7 => Ok(ImageFormat::Rgba8Snorm),
        0xda => Ok(ImageFormat::Rg16Unorm),
        0xdb => Ok(ImageFormat::Rg16Snorm),
        0xdf => Ok(ImageFormat::Bgr10a2Unorm),
        0xe9 | 0xf8 => Ok(ImageFormat::A1Rgb5Unorm),
        0xeb => Ok(ImageFormat::Rg8Snorm),
        0xee => Ok(ImageFormat::R16Unorm),
        0xef => Ok(ImageFormat::R16Snorm),
        0xf4 => Ok(ImageFormat::R8Snorm),
        0xc0 | 0xc3 => Ok(ImageFormat::Rgba32Float),
        0xca => Ok(ImageFormat::Rgba16Float),
        0xcb => Ok(ImageFormat::Rg32Float),
        0xce => Ok(ImageFormat::Rgba16Float),
        0xcf => Ok(ImageFormat::Bgra8Unorm),
        0xd0 => Ok(ImageFormat::Bgra8Srgb),
        0xd1 => Ok(ImageFormat::Rgb10a2Unorm),
        0xd5 => Ok(ImageFormat::Rgba8Unorm),
        0xd6 => Ok(ImageFormat::Rgba8Srgb),
        0xde => Ok(ImageFormat::Rg16Float),
        0xe0 => Ok(ImageFormat::Rg11b10Ufloat),
        0xe5 => Ok(ImageFormat::R32Float),
        0xe6 => Ok(ImageFormat::Bgra8Unorm),
        0xe7 => Ok(ImageFormat::Bgra8Srgb),
        0xe8 => Ok(ImageFormat::Rgb565Unorm),
        0xea => Ok(ImageFormat::Rg8Unorm),
        0xf2 => Ok(ImageFormat::R16Float),
        0xf3 => Ok(ImageFormat::R8Unorm),
        0xf9 => Ok(ImageFormat::Rgba8Unorm),
        0xfa => Ok(ImageFormat::Rgba8Srgb),
        _ => Err(MaxwellThreeDResourceError::UnsupportedColorFormat { role, format }),
    }
}

/// Decodes verified Maxwell TIC format/component/swizzle tuples into their
/// neutral storage and sampling semantics. Numeric component and swizzle values are
/// defined by deko3d's pinned public Maxwell tables:
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
fn decode_sampled_texture_format(
    image_format: u8,
    components: [u8; 4],
    swizzle: [u8; 4],
    srgb: bool,
    reserved: u32,
) -> Option<ImageFormat> {
    if reserved != 0 || !swizzle.iter().all(|c| matches!(c, 0 | 2..=7)) {
        return None;
    }
    if components == [2; 4] {
        let astc = match (image_format, srgb) {
            (0x40, false) => Some(ImageFormat::Astc4x4Unorm),
            (0x40, true) => Some(ImageFormat::Astc4x4Srgb),
            (0x50, false) => Some(ImageFormat::Astc5x4Unorm),
            (0x50, true) => Some(ImageFormat::Astc5x4Srgb),
            (0x41, false) => Some(ImageFormat::Astc5x5Unorm),
            (0x41, true) => Some(ImageFormat::Astc5x5Srgb),
            (0x51, false) => Some(ImageFormat::Astc6x5Unorm),
            (0x51, true) => Some(ImageFormat::Astc6x5Srgb),
            (0x42, false) => Some(ImageFormat::Astc6x6Unorm),
            (0x42, true) => Some(ImageFormat::Astc6x6Srgb),
            (0x55, false) => Some(ImageFormat::Astc8x5Unorm),
            (0x55, true) => Some(ImageFormat::Astc8x5Srgb),
            (0x52, false) => Some(ImageFormat::Astc8x6Unorm),
            (0x52, true) => Some(ImageFormat::Astc8x6Srgb),
            (0x44, false) => Some(ImageFormat::Astc8x8Unorm),
            (0x44, true) => Some(ImageFormat::Astc8x8Srgb),
            (0x56, false) => Some(ImageFormat::Astc10x5Unorm),
            (0x56, true) => Some(ImageFormat::Astc10x5Srgb),
            (0x57, false) => Some(ImageFormat::Astc10x6Unorm),
            (0x57, true) => Some(ImageFormat::Astc10x6Srgb),
            (0x53, false) => Some(ImageFormat::Astc10x8Unorm),
            (0x53, true) => Some(ImageFormat::Astc10x8Srgb),
            (0x45, false) => Some(ImageFormat::Astc10x10Unorm),
            (0x45, true) => Some(ImageFormat::Astc10x10Srgb),
            (0x54, false) => Some(ImageFormat::Astc12x10Unorm),
            (0x54, true) => Some(ImageFormat::Astc12x10Srgb),
            (0x46, false) => Some(ImageFormat::Astc12x12Unorm),
            (0x46, true) => Some(ImageFormat::Astc12x12Srgb),
            _ => None,
        };
        if astc.is_some() {
            return astc;
        }
    }
    match (image_format, components, swizzle, srgb) {
        // Component type 4 is UINT, type 3 is SINT; selectors do not
        // change the integer storage or its shader-visible component type.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/format_lookup_table.cpp
        (0x1d, [4, 4, 4, 4], _, false) => Some(ImageFormat::R8Uint),
        (0x1d, [3, 3, 3, 3], _, false) => Some(ImageFormat::R8Sint),
        (0x18, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rg8Uint),
        (0x18, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rg8Sint),
        (0x08, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rgba8Uint),
        (0x08, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rgba8Sint),
        (0x1b, [4, 4, 4, 4], _, false) => Some(ImageFormat::R16Uint),
        (0x1b, [3, 3, 3, 3], _, false) => Some(ImageFormat::R16Sint),
        (0x0c, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rg16Uint),
        (0x0c, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rg16Sint),
        (0x03, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rgba16Uint),
        (0x03, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rgba16Sint),
        (0x0f, [4, 4, 4, 4], _, false) => Some(ImageFormat::R32Uint),
        (0x0f, [3, 3, 3, 3], _, false) => Some(ImageFormat::R32Sint),
        (0x04, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rg32Uint),
        (0x04, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rg32Sint),
        (0x01, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rgba32Uint),
        (0x01, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rgba32Sint),
        (0x1d, [1, 1, 1, 1], _, false) => Some(ImageFormat::R8Snorm),
        (0x18, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rg8Snorm),
        (0x08, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rgba8Snorm),
        (0x1b, [2, 2, 2, 2], _, false) => Some(ImageFormat::R16Unorm),
        (0x1b, [1, 1, 1, 1], _, false) => Some(ImageFormat::R16Snorm),
        (0x0c, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rg16Unorm),
        (0x0c, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rg16Snorm),
        (0x03, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgba16Unorm),
        (0x03, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rgba16Snorm),
        (0x15, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgb565Unorm),
        // Z24S8 uses unsigned stencil R and normalized depth G. Selecting G
        // for RGB samples the depth aspect, with alpha forced to one.
        (0x29, [4, 2, 2, 2], [3, 3, 3, 7], false) => Some(ImageFormat::Depth24UnormStencil8Uint),
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/format_lookup_table.cpp
        (0x3a, [2, 2, 2, 2] | [2, 4, 4, 4], _, false) => Some(ImageFormat::Depth16Unorm),
        // ZF32 uses a float depth channel, replicated to RGB by its TIC mapping.
        // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/image_formats.h
        (0x2f, [7, 7, 7, 7] | [7, 4, 4, 4] | [7, 2, 2, 2], _, false) => {
            Some(ImageFormat::Depth32Float)
        }
        // BC6H floating-point component types follow Maxwell's TIC table.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/format_lookup_table.cpp#L190-L193
        (0x10, [7, 7, 7, 7], _, false) => Some(ImageFormat::Bc6hRgbSfloat),
        (0x11, [7, 7, 7, 7], _, false) => Some(ImageFormat::Bc6hRgbUfloat),
        (0x27, [2, 2, 2, 2], _, false) => Some(ImageFormat::Bc4RUnorm),
        (0x28, [2, 2, 2, 2], _, false) => Some(ImageFormat::Bc5RgUnorm),
        // Component type 1 is SNORM; signed RGTC endpoints are not UNORM.
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
        (0x27, [1, 1, 1, 1], _, false) => Some(ImageFormat::Bc4RSnorm),
        (0x28, [1, 1, 1, 1], _, false) => Some(ImageFormat::Bc5RgSnorm),
        (0x17, [2, 2, 2, 2], _, srgb) => Some(if srgb {
            ImageFormat::Bc7RgbaSrgb
        } else {
            ImageFormat::Bc7RgbaUnorm
        }),
        (0x26, [2, 2, 2, 2], _, srgb) => Some(if srgb {
            ImageFormat::Bc3RgbaSrgb
        } else {
            ImageFormat::Bc3RgbaUnorm
        }),
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_compression_s3tc.txt
        // DXT1 storage and TIC selectors are independent. The opaque RGB
        // variant is equivalent only when no output observes the stored alpha.
        (0x24, [2, 2, 2, 2], mapping, srgb) => Some(match (mapping.contains(&5), srgb) {
            (false, false) => ImageFormat::Bc1RgbUnorm,
            (false, true) => ImageFormat::Bc1RgbSrgb,
            (true, false) => ImageFormat::Bc1RgbaUnorm,
            (true, true) => ImageFormat::Bc1RgbaSrgb,
        }),
        // Single-channel masks can independently route R or constants to
        // every sampled component, including white RGB with red as alpha.
        (0x1d, [2, 2, 2, 2], _, false) => Some(ImageFormat::R8Unorm),
        // Storage and view selectors are independent; luminance-alpha uses
        // R,R,R,G without changing the two-byte RG8 storage format.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/format_lookup_table.cpp
        (0x18, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rg8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], false) => Some(ImageFormat::Bgra8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], true) => Some(ImageFormat::Bgra8Srgb),
        (0x08, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgba8Unorm),
        (0x08, [2, 2, 2, 2], _, true) => Some(ImageFormat::Rgba8Srgb),
        (0x1b, [7, 7, 7, 7], _, false) => Some(ImageFormat::R16Float),
        (0x0c, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg16Float),
        (0x03, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rgba16Float),
        (0x21, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg11b10Ufloat),
        (0x09, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgb10a2Unorm),
        (0x0f, [7, 7, 7, 7], _, false) => Some(ImageFormat::R32Float),
        (0x04, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg32Float),
        (0x01, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rgba32Float),
        _ => None,
    }
}

fn unresolved(upper: Option<&u8>, lower: Option<&u32>) -> Option<MaxwellThreeDUnresolvedAddress> {
    Some(MaxwellThreeDUnresolvedAddress::new(*upper?, *lower?))
}
fn align_up(
    value: u64,
    alignment: u64,
    role: MaxwellThreeDResourceRole,
) -> Result<u64, MaxwellThreeDResourceError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })
}

fn image_description(
    request: MaxwellImageDescriptionRequest,
) -> Result<ImageDescription, MaxwellThreeDResourceError> {
    let extent = ImageExtent::new(
        request.extent.width,
        request.extent.height,
        request.extent.depth,
    )
    .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role: request.role })?;
    ImageDescription::new(
        request.dimension,
        extent,
        request.format,
        request.kind,
        1,
        request.layers,
        SampleCount::One,
    )
    .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role: request.role })
}

fn descriptor_words(bytes: [u8; 32]) -> [u32; 8] {
    std::array::from_fn(|index| {
        u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
    })
}

pub fn decode_sampler(
    texture_reference: MaxwellThreeDTextureReference,
    descriptor: u32,
    bytes: [u8; 32],
) -> Result<MaxwellThreeDResolvedSampler, MaxwellThreeDResourceError> {
    // Maxwell TSC field placement and public enums follow deko3d's pinned
    // descriptor definition and generator:
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_sampler_control_block.h
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/tsc_generate.cpp
    let words = descriptor_words(bytes);
    let role = MaxwellThreeDResourceRole::Sampler(texture_reference);
    // Eden TSCEntry::reduction_filter, word 1 bits 10..11.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/textures/texture.h
    let reduction = match (words[1] >> 10) & 3 {
        0 => nixe_gpu::SamplerReduction::WeightedAverage,
        1 => nixe_gpu::SamplerReduction::Min,
        2 => nixe_gpu::SamplerReduction::Max,
        value => {
            return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                descriptor,
                field: "reduction filter",
                value,
            });
        }
    };
    let compare = (words[0] & (1 << 9) != 0).then(|| {
        use nixe_gpu::DepthCompareOperation as C;
        [
            C::Never,
            C::Less,
            C::Equal,
            C::LessEqual,
            C::Greater,
            C::NotEqual,
            C::GreaterEqual,
            C::Always,
        ][((words[0] >> 10) & 7) as usize]
    });
    let address_mode = |raw: u32| match raw {
        0 => Ok(AddressMode::Repeat),
        1 => Ok(AddressMode::MirroredRepeat),
        2 => Ok(AddressMode::ClampToEdge),
        3 => Ok(AddressMode::ClampToBorder),
        4 => Ok(AddressMode::Clamp),
        _ => Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "address mode",
            value: raw,
        }),
    };
    let filter = |raw: u32, field| match raw {
        1 => Ok(FilterMode::Nearest),
        2 => Ok(FilterMode::Linear),
        _ => Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field,
            value: raw,
        }),
    };
    let mip_filter = match (words[1] >> 6) & 3 {
        1 | 2 => FilterMode::Nearest,
        3 => FilterMode::Linear,
        value => {
            return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                descriptor,
                field: "mip filter",
                value,
            });
        }
    };
    let bias = ((words[1] >> 12) & 0x1fff) as i32;
    let signed_bias = (bias << 19) >> 19;
    let lod_min_fixed = (words[2] & 0xfff) as u16;
    let lod_max_fixed = ((words[2] >> 12) & 0xfff) as u16;
    if lod_min_fixed > lod_max_fixed {
        return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "LOD clamp",
            value: words[2] & 0x00ff_ffff,
        });
    }
    let max_anisotropy = [1, 2, 4, 6, 8, 10, 12, 16][((words[0] >> 20) & 7) as usize];
    let min_filter = filter((words[1] >> 4) & 3, "minification filter")?;
    let mag_filter = filter(words[1] & 3, "magnification filter")?;
    let address_modes = [
        address_mode(words[0] & 7)?,
        address_mode((words[0] >> 3) & 7)?,
        address_mode((words[0] >> 6) & 7)?,
    ];
    let samples_border = address_modes.contains(&AddressMode::ClampToBorder)
        || (mag_filter == FilterMode::Linear && address_modes.contains(&AddressMode::Clamp));
    let border_color = samples_border.then_some([words[4], words[5], words[6], words[7]]);
    if let Some(color) = border_color
        && words[0] & (1 << 13) != 0
    {
        let srgb = [
            words[2] >> 24,
            (words[3] >> 12) & 255,
            (words[3] >> 20) & 255,
        ];
        for (bits, encoded) in color[..3].iter().copied().zip(srgb) {
            if (bits == 0 && encoded != 0) || (bits == 1.0_f32.to_bits() && encoded != 255) {
                return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                    descriptor,
                    field: "distinct sRGB border color",
                    value: words[2] ^ words[3],
                });
            }
        }
    }
    Ok(MaxwellThreeDResolvedSampler {
        role,
        compare,
        reduction,
        min_filter,
        mag_filter,
        mip_filter,
        lod_bias_fixed: signed_bias as i16,
        address_modes,
        border_color,
        lod_min_fixed,
        lod_max_fixed,
        max_anisotropy,
    })
}

#[derive(Clone, Debug)]
pub enum MaxwellThreeDResourceError {
    IncompleteState {
        role: MaxwellThreeDResourceRole,
    },
    ContradictoryState {
        role: MaxwellThreeDResourceRole,
    },
    AddressOutOfRange {
        role: MaxwellThreeDResourceRole,
        address: u64,
    },
    ArithmeticOverflow {
        role: MaxwellThreeDResourceRole,
    },
    Resolution {
        role: MaxwellThreeDResourceRole,
        error: MaxwellGpuAccessError,
    },
    StaleMapping {
        mapping: MaxwellMappingId,
        generation: nixe_gpu::MappingGeneration,
    },
    UnsupportedColorFormat {
        role: MaxwellThreeDResourceRole,
        format: u8,
    },
    UnsupportedSparseDepth {
        value: u8,
    },
    UnsupportedDepthFormat {
        role: MaxwellThreeDResourceRole,
        format: u32,
    },
    UnsupportedKind {
        role: MaxwellThreeDResourceRole,
        expected: u8,
        actual: u8,
    },
    UnsupportedImageLayout {
        role: MaxwellThreeDResourceRole,
    },
    UnsupportedSampleMode {
        role: MaxwellThreeDResourceRole,
        mode: MaxwellThreeDSampleMode,
    },
    UnsupportedTextureDescriptor {
        descriptor: u32,
        field: &'static str,
        value: u32,
    },
    UnsupportedSamplerDescriptor {
        descriptor: u32,
        field: &'static str,
        value: u32,
    },
    InvalidNeutralView {
        role: MaxwellThreeDResourceRole,
    },
    Memory(Arc<nixe_memory::MemoryError>),
    ResourceExhausted,
}

impl Display for MaxwellThreeDResourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompleteState { role } => {
                write!(formatter, "incomplete Maxwell resource state: {role:?}")
            }
            Self::ContradictoryState { role } => {
                write!(formatter, "contradictory Maxwell resource state: {role:?}")
            }
            Self::AddressOutOfRange { role, address } => write!(
                formatter,
                "Maxwell resource address is outside the profile: role={role:?} address=0x{address:016x}"
            ),
            Self::ArithmeticOverflow { role } => {
                write!(formatter, "Maxwell resource range overflows: {role:?}")
            }
            Self::Resolution { role, error } => write!(
                formatter,
                "Maxwell resource resolution failed: role={role:?} error=[{error}]"
            ),
            Self::StaleMapping {
                mapping,
                generation,
            } => write!(
                formatter,
                "Maxwell resource retained a stale mapping: {mapping} {generation}"
            ),
            Self::UnsupportedColorFormat { role, format } => write!(
                formatter,
                "Maxwell color format has no neutral interpretation: role={role:?} format=0x{format:02x}"
            ),
            Self::UnsupportedSparseDepth { value } => write!(
                formatter,
                "sparse depth attachment semantics are not implemented: SET_ZT_SPARSE={value:#x}"
            ),
            Self::UnsupportedDepthFormat { role, format } => write!(
                formatter,
                "Maxwell depth format has no neutral interpretation: role={role:?} format=0x{format:02x}"
            ),
            Self::UnsupportedKind {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "Maxwell PTE kind contradicts image layout: role={role:?} expected=0x{expected:02x} actual=0x{actual:02x}"
            ),
            Self::UnsupportedImageLayout { role } => write!(
                formatter,
                "Maxwell image layout has no neutral interpretation: {role:?}"
            ),
            Self::UnsupportedSampleMode { role, mode } => write!(
                formatter,
                "Maxwell sample mode has no neutral image interpretation: role={role:?} mode={mode:?}"
            ),
            Self::UnsupportedTextureDescriptor {
                descriptor,
                field,
                value,
            } => write!(
                formatter,
                "Maxwell texture descriptor {descriptor} has unsupported {field}: value=0x{value:08x}"
            ),
            Self::UnsupportedSamplerDescriptor {
                descriptor,
                field,
                value,
            } => write!(
                formatter,
                "Maxwell sampler descriptor {descriptor} has unsupported {field}: value=0x{value:08x}"
            ),
            Self::InvalidNeutralView { role } => write!(
                formatter,
                "resolved Maxwell resource cannot form a neutral view: {role:?}"
            ),
            Self::Memory(error) => write!(formatter, "resource memory access failed: {error}"),
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell resource resolution exhausted host bookkeeping")
            }
        }
    }
}

impl std::error::Error for MaxwellThreeDResourceError {}
