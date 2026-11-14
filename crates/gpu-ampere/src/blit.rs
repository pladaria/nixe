//! 902D pixel copies with native image storage and signed 32.32 sampling coordinates.
//! https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/twod/cl902d.h
use crate::{Registers, Result};
use nixe_gpu::{
    native::{ImageInfo, ImagePacking},
    *,
};
use std::sync::Arc;

pub struct Blit {
    pub images: [ImageInfo; 2],
    pub destination: ScissorRect,
    pub first: [i64; 2],
    pub step: [i64; 2],
    pub linear: bool,
}

impl Blit {
    /// Same-format, unit-step copies preserve texel bits, including compressed
    /// blocks carried in a 128-bit surface. Sampling them as floating-point
    /// colors can change NaNs and loses their encoded-storage identity.
    /// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/engines/sw_blitter/blitter.cpp
    pub fn copy_source_rect(&self) -> Option<ScissorRect> {
        let [src, dst] = &self.images;
        if src.description.format() != dst.description.format()
            || self.step != [1 << 32; 2]
            || self.linear && self.first.iter().any(|&v| v as u32 != 0x8000_0000)
        {
            return None;
        }
        Some(ScissorRect {
            x: (self.first[0] >> 32) as u32,
            y: (self.first[1] >> 32) as u32,
            ..self.destination
        })
    }
    pub fn byte_copy(&self) -> Result<Option<crate::dma::Copy>> {
        let Some(rect) = self.copy_source_rect() else {
            return Ok(None);
        };
        let [src, dst] = &self.images;
        Ok(Some(crate::dma::Copy {
            source: copy_surface(src, rect)?,
            destination: copy_surface(dst, self.destination)?,
        }))
    }
}

fn copy_surface(info: &ImageInfo, rect: ScissorRect) -> Result<crate::dma::Surface> {
    use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage};
    let bpp = u32::from(
        info.description
            .format()
            .plane_bytes_per_texel(0)
            .ok_or("invalid copy format")?,
    );
    let extent = info.description.extent();
    let storage = match info.layout {
        ImageMemoryLayout::PitchLinear { row_pitch, .. } => ImageStorage::Pitch { row_pitch },
        ImageMemoryLayout::BlockLinearAmpere(block) => ImageStorage::BlockLinearAmpere {
            block_height_log2: block.block_height_log2,
            block_depth_log2: block.block_depth_log2,
        },
        _ => return Err("invalid native 2D copy layout".into()),
    };
    let columns = extent.width.checked_mul(bpp).ok_or("copy width overflow")?;
    crate::dma::Surface::new(
        info.address,
        ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns,
            rows: extent.height,
            packed_bytes: 1,
            aspect_offset: 0,
            aspect_bytes: 1,
            storage,
            native_base: 0,
            native_element_bytes: 1,
            native_row_pitch: u64::from(columns),
            rect: ByteRect {
                x: rect.x.checked_mul(bpp).ok_or("copy origin overflow")?,
                y: rect.y,
                width: rect.width.checked_mul(bpp).ok_or("copy width overflow")?,
                height: rect.height,
            },
        },
    )
}

pub(crate) fn decode(r: &Registers) -> Result<Option<Blit>> {
    if r.get(0x2ac) != 3 || r.get(0x29c) != 0 || r.get(0x870) != 0 {
        return Err("2D blit requires SRCCOPY without color key or endian conversion".into());
    }
    match r.get(0x2dc) {
        2 => return Ok(None),
        1 => (),
        0 => match r.get(0x26c) {
            0 => return Ok(None),
            1 => (),
            mode => {
                return Err(
                    format!("2D conditional render mode {mode:#x} is not implemented").into(),
                );
            }
        },
        mode => return Err(format!("invalid 2D render override {mode:#x}").into()),
    }
    let mode = r.get(0x88c);
    if mode & !0x11 != 0 {
        return Err("invalid 2D sampling mode".into());
    }
    let mut destination = ScissorRect {
        x: r.get(0x8b0),
        y: r.get(0x8b4),
        width: r.get(0x8b8),
        height: r.get(0x8bc),
    };
    if destination.width == 0 || destination.height == 0 {
        return Ok(None);
    }
    let fixed = |low| r.address(low + 4, low) as i64;
    let step = [fixed(0x8c0), fixed(0x8c8)];
    let adjustment = if mode & 1 == 0 { 1_i64 << 31 } else { 0 };
    let mut first = [0; 2];
    for (axis, low) in [0x8d0, 0x8d8].into_iter().enumerate() {
        first[axis] = fixed(low)
            .checked_add(adjustment)
            .ok_or("2D sampling coordinate overflow")?;
    }
    match r.get(0x290) {
        0 => (),
        1 => {
            let old = [destination.x, destination.y];
            let extent = [destination.width, destination.height];
            let mut origins = [0; 2];
            let mut counts = [0; 2];
            for axis in 0..2 {
                let clip = r.get(0x280 + axis as u32 * 4);
                let end = u64::from(old[axis]) + u64::from(extent[axis]);
                let clip_end = u64::from(clip) + u64::from(r.get(0x288 + axis as u32 * 4));
                origins[axis] = old[axis].max(clip);
                counts[axis] = end.min(clip_end).saturating_sub(u64::from(origins[axis])) as u32;
                first[axis] = i64::try_from(
                    i128::from(first[axis])
                        + i128::from(origins[axis] - old[axis]) * i128::from(step[axis]),
                )
                .map_err(|_| "clipped 2D sampling coordinate overflow")?;
            }
            destination = ScissorRect {
                x: origins[0],
                y: origins[1],
                width: counts[0],
                height: counts[1],
            };
            if counts.contains(&0) {
                return Ok(None);
            }
        }
        _ => return Err("invalid 2D clipping enable".into()),
    }
    let images = [surface(r, 0x230)?, surface(r, 0x200)?];
    let dst = images[1].description.extent();
    if u64::from(destination.x) + u64::from(destination.width) > u64::from(dst.width)
        || u64::from(destination.y) + u64::from(destination.height) > u64::from(dst.height)
    {
        return Err("2D destination rectangle exceeds its surface".into());
    }
    let src = images[0].description.extent();
    for axis in 0..2 {
        let last = i128::from(first[axis])
            + i128::from(step[axis])
                * i128::from([destination.width, destination.height][axis] - 1);
        let limit = i128::from([src.width, src.height][axis]) << 32;
        if i128::from(first[axis]).min(last) < 0 || i128::from(first[axis]).max(last) >= limit {
            return Err("2D sampling rectangle exceeds its source surface".into());
        }
    }
    Ok(Some(Blit {
        images,
        destination,
        first,
        step,
        linear: mode & 16 != 0,
    }))
}

fn surface(r: &Registers, base: u32) -> Result<ImageInfo> {
    let format = crate::targets::color_format(r.get(base))?;
    if matches!(
        r.get(base),
        0xc1 | 0xc2
            | 0xc4
            | 0xc5
            | 0xc8
            | 0xc9
            | 0xcc
            | 0xcd
            | 0xd8
            | 0xd9
            | 0xdc
            | 0xdd
            | 0xe3
            | 0xe4
            | 0xec
            | 0xed
            | 0xf0
            | 0xf1
            | 0xf5
            | 0xf6
    ) {
        return Err(format!(
            "integer 2D blit format {:#x} is not implemented",
            r.get(base)
        )
        .into());
    }
    let width = r.get(base + 24);
    let height = r.get(base + 28);
    if r.get(base + 12) != 1 || (base == 0x200 && r.get(base + 16) != 0) {
        return Err("2D blit requires a single-layer 2D surface".into());
    }
    let bpp = u64::from(
        format
            .plane_bytes_per_texel(0)
            .ok_or("invalid 2D color format")?,
    );
    let (layout, size) = match r.get(base + 4) {
        0 => {
            let block = r.get(base + 8);
            let height_log2 = ((block >> 4) & 15) as u8;
            if block & !0x70 != 0 || height_log2 > 5 {
                return Err(format!("unsupported 2D block size {block:#x}").into());
            }
            let size = (u64::from(width) * bpp)
                .next_multiple_of(64)
                .checked_mul(u64::from(height).next_multiple_of(8 << height_log2))
                .ok_or("2D surface storage overflow")?;
            (
                ImageMemoryLayout::BlockLinearAmpere(BlockLinearLayout {
                    block_width_log2: 0,
                    block_height_log2: height_log2,
                    block_depth_log2: 0,
                    layer_stride: size,
                }),
                size,
            )
        }
        1 => {
            let pitch = u64::from(r.get(base + 20));
            if pitch < u64::from(width) * bpp {
                return Err("2D row pitch does not cover the surface width".into());
            }
            let size = pitch * u64::from(height);
            (
                ImageMemoryLayout::PitchLinear {
                    row_pitch: pitch,
                    layer_stride: size,
                },
                size,
            )
        }
        mode => return Err(format!("invalid 2D memory layout {mode:#x}").into()),
    };
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(width, height, 1).map_err(|e| e.to_string().into_boxed_str())?,
        format,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .map_err(|e| e.to_string().into_boxed_str())?;
    Ok(ImageInfo {
        address: r.address(base + 32, base + 36),
        description,
        swizzle: Swizzle::IDENTITY,
        layout,
        mip_layouts: Arc::from([(0, layout)]),
        layer_stride: size,
        maybe_unaligned_layer_stride: size,
        size,
        packing: ImagePacking::Native,
        sample_grid: [1, 1],
    })
}
