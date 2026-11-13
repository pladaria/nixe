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

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{
    BlockLinearLayout, ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    ImageMemoryLayout, SampleCount, Swizzle,
};

use crate::{MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellMappingId};

use super::{
    MaxwellThreeDAttachmentReadiness, MaxwellThreeDColorTargetFormat,
    MaxwellThreeDColorTargetState, MaxwellThreeDDepthArrayControl, MaxwellThreeDDepthStencilFormat,
    MaxwellThreeDDepthStencilTargetState, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDImageKind, MaxwellThreeDImageLayout,
    MaxwellThreeDSampleMode, MaxwellThreeDState, MaxwellThreeDUnresolvedAddress,
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
    ColorTarget(u8),
    DepthStencilTarget,
    BlitSource,
    BlitDestination,
}

/// Original Maxwell texel encoding, independent of native component semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDGuestImageFormat {
    /// Canonical texels authored by another device frontend (for example VIC).
    Canonical(ImageFormat),
    Color(MaxwellThreeDColorTargetFormat),
    DepthStencil(MaxwellThreeDDepthStencilFormat),
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
