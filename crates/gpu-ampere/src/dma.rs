//! C7B5 byte-copy geometry. Memory coherence and storage transforms stay shared.
//! https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc7b5.h
use crate::{Registers, Result};
use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage};

pub struct Surface {
    pub address: u64,
    pub layout: ImageFootprint,
    pub bytes: u64,
}
pub struct Copy {
    pub source: Surface,
    pub destination: Surface,
}
impl Copy {
    pub(crate) fn decode(registers: &Registers, flags: u32) -> Result<Self> {
        let width = registers.get(0x418);
        let height = if flags & 0x200 != 0 {
            registers.get(0x41c)
        } else {
            1
        };
        if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 64 * 1024 * 1024 {
            return Err("invalid DMA copy dimensions".into());
        }
        Ok(Self {
            source: Surface::decode(registers, true, flags & 0x80 != 0, width, height)?,
            destination: Surface::decode(registers, false, flags & 0x100 != 0, width, height)?,
        })
    }
}
impl Surface {
    pub(crate) fn new(address: u64, layout: ImageFootprint) -> Result<Self> {
        let bytes = layout
            .storage_bytes()
            .map_err(|e| e.to_string().into_boxed_str())?;
        if bytes > 64 * 1024 * 1024 || address.checked_add(bytes).is_none_or(|end| end > 1 << 40) {
            return Err("copy storage range exceeds native address/transfer limits".into());
        }
        Ok(Self {
            address,
            layout,
            bytes,
        })
    }
    fn decode(r: &Registers, source: bool, pitch: bool, width: u32, height: u32) -> Result<Self> {
        let address = if source {
            r.address(0x400, 0x404)
        } else {
            r.address(0x408, 0x40c)
        };
        let (columns, rows, rect, slice, storage) = if pitch {
            let row_pitch = if height == 1 {
                width
            } else {
                r.get(if source { 0x410 } else { 0x414 })
            };
            (
                width,
                height,
                ByteRect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                0,
                ImageStorage::Pitch {
                    row_pitch: u64::from(row_pitch),
                },
            )
        } else {
            let base = if source { 0x728 } else { 0x70c };
            let block = r.get(base);
            let depth = r.get(base + 12);
            let slice = r.get(base + 16);
            if block & !0x1ff0 != 0 || block >> 12 != 1 || slice >= depth {
                return Err(format!("unsupported DMA block size/layer: block={block:#x}, layer={slice}, depth={depth}").into());
            }
            let origin = if source { 0x744 } else { 0x74c };
            (
                r.get(base + 4),
                r.get(base + 8),
                ByteRect {
                    x: r.get(origin),
                    y: r.get(origin + 4),
                    width,
                    height,
                },
                slice,
                ImageStorage::BlockLinearAmpere {
                    block_height_log2: ((block >> 4) & 15) as u8,
                    block_depth_log2: ((block >> 8) & 15) as u8,
                },
            )
        };
        let layout = ImageFootprint {
            depth_slice: slice,
            base: 0,
            columns,
            rows,
            packed_bytes: 1,
            aspect_offset: 0,
            aspect_bytes: 1,
            storage,
            native_base: 0,
            native_element_bytes: 1,
            native_row_pitch: u64::from(columns),
            rect,
        };
        Self::new(address, layout)
    }
}
