//! Native image preferences are a GSV ABI, distinct from hardware texture headers.
//! https://switchbrew.org/wiki/Switch_2%3A_GSV_services
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Debug, Default)]
pub(crate) struct Context {
    pub memories: BTreeMap<u32, Memory>,
    pub next_memory: u32,
    pub pools: BTreeMap<u32, Arc<Pool>>,
    pub window_pools: BTreeMap<u32, Arc<Pool>>,
    pub spaces: BTreeMap<u32, Arc<Mutex<super::gpu::GpuAddressSpace>>>,
    pub next_image: u32,
}

#[derive(Debug)]
pub(crate) struct Memory {
    pub size: u64,
    pub backing: Option<Arc<Backing>>,
}

#[derive(Debug)]
pub(crate) struct Backing {
    pub ranges: nixe_memory::CapturedMemory,
    pub source: Arc<nixe_runtime::memory::GpuMemoryLease>,
    pub offset: u64,
}

#[derive(Debug)]
pub(crate) struct Pool {
    pub preference: [u32; 16],
    pub count: u32,
    pub images: Vec<std::sync::OnceLock<Presentation>>,
    pub backing: Arc<Backing>,
}

/// @SPECULATIVE: the preference words and alignment policy follow the SDK
/// observations recorded in the reference; the complete ABI is not published.
/// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_image_layout.h
pub(crate) fn requirements(preference: [u32; 16], count: u32) -> Result<(u64, u64), Box<str>> {
    let p = preference;
    if count == 0
        || p[0] != 0
        || !matches!(p[1], 1 | 2)
        || p[3] != 0
        || p[7] != 1
        || p[8..].iter().any(|&v| v != 0)
        || !(3..=8).contains(&p[4])
    {
        return Err(
            format!("unsupported native image preference: count={count} words={p:08x?}").into(),
        );
    }
    let width = u64::from((p[2] & 0xffff) + 1);
    let height = u64::from((p[2] >> 16) + 1);
    let bytes_per_pixel = match p[6] {
        1 | 4 => 4,
        5 => 8,
        _ => return Err(format!("unsupported native image format {}", p[6]).into()),
    };
    let pitch = u64::from(p[5]) * 64;
    if pitch < width * bytes_per_pixel {
        return Err("native image pitch is shorter than a row".into());
    }
    let block_rows = 1_u64 << p[4];
    let size = pitch
        .checked_mul(height.div_ceil(block_rows) * block_rows)
        .and_then(|n| n.checked_mul(u64::from(count)))
        .ok_or("native image size overflow")?;
    let alignment = if p[1] == 2 && count == 1 && p[6] == 1 {
        0x1000
    } else {
        (64 * block_rows).max(0x1000)
    };
    Ok((size, alignment))
}

/// A native image keeps the original device mapping alive, so presentation can
/// select its render target directly without a second mapping or pixel copy.
#[derive(Debug)]
pub(crate) struct Presentation {
    pub image: nixe_gpu::native::NativePresentationImage,
    pub _mapping: Arc<super::gpu::PhysicalMap>,
}
impl Pool {
    pub fn presentation(
        &self,
        index: u32,
        spaces: &BTreeMap<u32, Arc<Mutex<super::gpu::GpuAddressSpace>>>,
    ) -> Result<&Presentation, Box<str>> {
        let slot = self
            .images
            .get(index as usize)
            .ok_or("native image index out of range")?;
        if let Some(image) = slot.get() {
            return Ok(image);
        }
        let p = self.preference;
        let (size, _) = requirements(p, 1)?;
        let offset = self.backing.offset + size * u64::from(index);
        let format = match p[6] {
            1 => nixe_gpu::PresentationImageFormat::Rgba8,
            other => {
                return Err(
                    format!("native presentation format {other} is not implemented").into(),
                );
            }
        };
        for space in spaces.values() {
            let space = space.lock().unwrap();
            for (&start, mapping) in &space.mappings {
                let super::gpu::Backing::Cpu(source) = &mapping.source else {
                    continue;
                };
                if !Arc::ptr_eq(source, &self.backing.source)
                    || offset < mapping.offset
                    || offset + size > mapping.offset + mapping.end - start
                {
                    continue;
                }
                let extent = [(p[2] & 0xffff) + 1, (p[2] >> 16) + 1];
                let image = nixe_gpu::native::NativePresentationImage {
                    address: mapping.physical.pin.address() + offset - mapping.offset,
                    size,
                    extent,
                    row_pitch: p[5] * 64,
                    layout: nixe_gpu::ImageMemoryLayout::BlockLinearAmpere(
                        nixe_gpu::BlockLinearLayout {
                            block_width_log2: 0,
                            block_height_log2: (p[4] - 3) as u8,
                            block_depth_log2: 0,
                            layer_stride: size,
                        },
                    ),
                    crop: nixe_gpu::FrameCrop {
                        left: 0,
                        top: 0,
                        width: extent[0],
                        height: extent[1],
                    },
                    transform: nixe_gpu::FrameTransform::default(),
                    format,
                    allow_canonical_import: false,
                };
                let _ = slot.set(Presentation {
                    image,
                    _mapping: mapping.physical.clone(),
                });
                return Ok(slot.get().unwrap());
            }
        }
        Err("native presentation image has no contiguous device mapping; fragmented native image presentation is not implemented".into())
    }
}
