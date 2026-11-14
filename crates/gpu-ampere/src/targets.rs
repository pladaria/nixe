//! C797 render-target storage and clear selection.
//! https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc797.h
use crate::{Registers, Result};
use nixe_gpu::{
    native::{ImageInfo, ImagePacking},
    *,
};
use std::sync::Arc;

pub fn target(r: &Registers, slot: usize) -> Result<Option<ImageInfo>> {
    let count = r.get(0x121c) & 15;
    if count > 8 {
        return Err("color target count exceeds eight slots".into());
    }
    let (address, width, height, third, format, memory, pitch, dimension, kind, packing) =
        if slot < 8 {
            let base = 0x800 + slot as u32 * 64;
            let raw = r.get(base + 16);
            if slot >= count as usize || raw == 0 {
                return Ok(None);
            }
            let memory = r.get(base + 20);
            (
                r.address(base, base + 4),
                r.get(base + 8),
                r.get(base + 12),
                r.get(base + 24),
                color_format(raw)?,
                memory,
                r.get(base + 28),
                if memory & 0x10000 != 0 {
                    ImageDimension::Three
                } else {
                    ImageDimension::Two
                },
                ImageKind::Color,
                ImagePacking::Native,
            )
        } else {
            if r.get(0x1538) == 0 {
                return Ok(None);
            }
            if r.get(0x1538) != 1 {
                return Err("invalid depth-target enable".into());
            }
            let raw = r.get(0xfe8);
            let format = match raw {
                0x13 => ImageFormat::Depth16Unorm,
                0x14 | 0x16 => ImageFormat::Depth24UnormStencil8Uint,
                0xa => ImageFormat::Depth32Float,
                0x19 => ImageFormat::Depth32FloatStencil8Uint,
                _ => return Err(format!("unsupported depth-target format {raw:#x}").into()),
            };
            let third = r.get(0x1230);
            if third & !0x1ffff != 0 {
                return Err("invalid depth-target array control".into());
            }
            (
                r.address(0xfe0, 0xfe4),
                r.get(0x1228),
                r.get(0x122c),
                if third & 0x10000 != 0 {
                    1
                } else {
                    third & 0xffff
                },
                format,
                r.get(0xfec),
                r.get(0xff0),
                ImageDimension::Two,
                ImageKind::DepthStencil,
                if raw == 0x14 {
                    ImagePacking::Depth24High
                } else {
                    ImagePacking::Native
                },
            )
        };
    if address == 0 {
        return Ok(None);
    }
    let linear = memory & 0x1000 != 0;
    let bh = ((memory >> 4) & 15) as u8;
    let bd = ((memory >> 8) & 15) as u8;
    if memory & !0x11ff0 != 0 || bh > 5 || bd > 5 {
        return Err(format!("unsupported target memory layout {memory:#x}").into());
    }
    let sample_grid = match r.get(0xfb8) {
        0 => [1, 1],
        2 => [2, 2],
        value => return Err(format!("unsupported render-target sample mode {value:#x}").into()),
    };
    if linear && sample_grid != [1, 1] {
        return Err("multisampled pitch-linear target".into());
    }
    let bpp = u64::from(
        format
            .plane_bytes_per_texel(0)
            .ok_or("invalid target format")?,
    );
    let stored_width = if linear {
        (u64::from(width) / bpp) as u32
    } else {
        width
    };
    let (dimension, depth, layers) = if linear {
        (ImageDimension::Two, 1, 1)
    } else if dimension == ImageDimension::Three {
        (dimension, third, 1)
    } else {
        (
            dimension,
            1,
            u16::try_from(third).map_err(|_| "target layer count overflow")?,
        )
    };
    let extent = ImageExtent::new(
        stored_width.div_ceil(sample_grid[0]),
        height.div_ceil(sample_grid[1]),
        depth,
    )
    .map_err(|e| e.to_string().into_boxed_str())?;
    let description = ImageDescription::new(
        dimension,
        extent,
        format,
        kind,
        1,
        layers,
        if sample_grid == [1, 1] {
            SampleCount::One
        } else {
            SampleCount::Four
        },
    )
    .map_err(|e| e.to_string().into_boxed_str())?;
    let row = u64::from(stored_width) * bpp;
    let compact = row
        .checked_mul(u64::from(height))
        .and_then(|s| s.checked_mul(u64::from(depth)))
        .ok_or("target size overflow")?;
    // ARRAY_PITCH counts dwords. WIDTH counts stored samples for block-linear
    // targets and bytes for pitch-linear targets.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/texture_cache/image_info.cpp
    let array_pitch = if linear { 0 } else { u64::from(pitch) * 4 };
    let minimum = if linear {
        u64::from(width) * u64::from(height)
    } else {
        row.next_multiple_of(64)
            .checked_mul(u64::from(height).next_multiple_of(8 << bh))
            .and_then(|s| s.checked_mul(u64::from(depth).next_multiple_of(1 << bd)))
            .ok_or("target storage overflow")?
    };
    let stride = if array_pitch == 0 {
        minimum
    } else {
        array_pitch
    };
    if stride < compact || (sample_grid != [1, 1] && stride < minimum) {
        return Err("target array pitch does not cover its samples".into());
    }
    let layout = if linear {
        ImageMemoryLayout::PitchLinear {
            row_pitch: u64::from(width),
            layer_stride: stride,
        }
    } else {
        ImageMemoryLayout::BlockLinearAmpere(BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: bh,
            block_depth_log2: bd,
            layer_stride: stride,
        })
    };
    Ok(Some(ImageInfo {
        address,
        description,
        swizzle: Swizzle::IDENTITY,
        layout,
        mip_layouts: Arc::from([(0, layout)]),
        layer_stride: array_pitch,
        maybe_unaligned_layer_stride: array_pitch,
        // A single layer occupies its own tile footprint, even when ARRAY_PITCH
        // still describes the parent texture's complete mip chain.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/texture_cache/util.cpp#L583-L596
        size: if layers == 1 {
            minimum
        } else {
            stride
                .checked_mul(u64::from(layers))
                .ok_or("target size overflow")?
        },
        packing,
        sample_grid,
    }))
}

pub(crate) fn color_format(format: u32) -> Result<ImageFormat> {
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
        _ => Err(format!("unsupported color-target format {format:#x}").into()),
    }
}
