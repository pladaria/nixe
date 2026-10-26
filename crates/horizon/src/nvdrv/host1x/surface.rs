//! Tegra VIC surface geometry and guest memory layouts.
use super::video::{Mappings, gob_offset, read, region, word};

pub(super) struct Surface {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) pitch: usize,
    pub(super) rows: usize,
    pub(super) kind: u32,
    pub(super) block_height: u32,
    pub(super) format: u32,
}
impl Surface {
    pub(super) fn gpu_image(
        &self,
        mappings: &Mappings,
        address: u64,
        format: nixe_gpu::ImageFormat,
    ) -> Result<nixe_gpu::ConversionImage, String> {
        use nixe_gpu::*;
        let (whole, offset) = region(mappings, address, self.length())?;
        let (_, mapping) = mappings
            .range(..=u32::try_from(address).map_err(|_| "VIC address exceeds 32 bits")?)
            .next_back()
            .ok_or("VIC mapping was lost")?;
        let allocation =
            GpuAllocationDescription::new(whole.bytes(), 1).map_err(|e| e.to_string())?;
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(self.width as u32, self.height as u32, 1)
                .map_err(|e| e.to_string())?,
            format,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .map_err(|e| e.to_string())?;
        let backing = BackingView::new(
            GpuAllocationId::new(mapping.object.id().raw()),
            allocation,
            offset,
            whole
                .slice(offset, self.length() as u64)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let layout = if self.kind == 1 {
            ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: self.block_height as u8,
                block_depth_log2: 0,
                layer_stride: self.length() as u64,
            })
        } else {
            ImageMemoryLayout::PitchLinear {
                row_pitch: self.pitch as u64,
                layer_stride: self.length() as u64,
            }
        };
        let view = ImageView::new(
            ImageId::new(0),
            description,
            Swizzle::IDENTITY,
            vec![(
                ImageSubresourceRange {
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: 1,
                    plane: 0,
                },
                layout,
                backing,
            )],
        )
        .map_err(|e| e.to_string())?;
        Ok(ConversionImage { description, view })
    }
    pub(super) fn decode(bytes: &[u8], bytes_per_pixel: usize) -> Result<Self, String> {
        let layout = word(bytes, 0);
        let size = word(bytes, 4);
        let allocation = word(bytes, 8);
        let width = (size & 0x3fff) as usize + 1;
        let height = (size >> 14 & 0x3fff) as usize + 1;
        let pitch = ((allocation & 0x3fff) as usize + 1) * bytes_per_pixel;
        let rows = (allocation >> 14 & 0x3fff) as usize + 1;
        let kind = layout >> 11 & 15;
        let block_height = layout >> 15 & 15;
        if kind > 1 || block_height > 5 || width * bytes_per_pixel > pitch || height > rows {
            return Err("invalid/unsupported VIC surface layout".into());
        }
        Ok(Self {
            width,
            height,
            pitch: if kind == 1 {
                pitch.next_multiple_of(64)
            } else {
                pitch.next_multiple_of(16)
            },
            rows,
            kind,
            block_height,
            format: layout & 127,
        })
    }
    pub(super) fn length(&self) -> usize {
        self.pitch
            * if self.kind == 1 {
                self.rows.next_multiple_of(8 << self.block_height)
            } else {
                self.rows
            }
    }
    pub(super) fn offset(&self, x: usize, y: usize) -> usize {
        if self.kind == 1 {
            gob_offset(x, y, self.pitch, self.block_height)
        } else {
            y * self.pitch + x
        }
    }
    /// Visible bytes in storage order. Full GOBs are contiguous 512-byte runs;
    /// only edge GOBs need the 16-byte lane permutation. Adjacent runs coalesce
    /// during traversal instead of sorting a per-row fragment list each frame.
    pub(super) fn written_spans(&self, bytes_per_pixel: usize) -> Vec<std::ops::Range<usize>> {
        let width = self.width * bytes_per_pixel;
        let mut spans: Vec<std::ops::Range<usize>> = Vec::new();
        let mut emit = |start: usize, bytes: usize| {
            if let Some(last) = spans.last_mut()
                && last.end == start
            {
                last.end += bytes;
            } else {
                spans.push(start..start + bytes);
            }
        };
        if self.kind == 0 {
            for y in 0..self.height {
                emit(y * self.pitch, width);
            }
        } else {
            let block_height = 1 << self.block_height;
            let block_rows = block_height * 8;
            let columns = self.pitch / 64;
            for block_y in 0..self.height.div_ceil(block_rows) {
                for gob_x in 0..width.div_ceil(64) {
                    let columns_left = (width - gob_x * 64).min(64);
                    let rows_left = (self.height - block_y * block_rows).min(block_rows);
                    for gob_y in 0..rows_left.div_ceil(8) {
                        let rows = (rows_left - gob_y * 8).min(8);
                        let base = ((block_y * columns + gob_x) * block_height + gob_y) * 512;
                        if columns_left == 64 && rows == 8 {
                            emit(base, 512);
                        } else {
                            for offset in (0..512).step_by(16) {
                                let x = (offset / 256) * 32 + (offset / 32 % 2) * 16;
                                let y = (offset / 64 % 4) * 2 + offset / 16 % 2;
                                if x < columns_left && y < rows {
                                    emit(base + offset, (columns_left - x).min(16));
                                }
                            }
                        }
                    }
                }
            }
        }
        spans
    }

    pub(super) fn read_linear(&self, mappings: &Mappings, address: u64) -> Result<Vec<u8>, String> {
        let stored = read(mappings, address, self.length())?;
        if self.kind == 0 {
            return Ok(stored);
        }
        let mut linear = vec![0; self.pitch * self.height];
        for y in 0..self.height {
            for x in (0..self.width).step_by(16) {
                let count = (self.width - x).min(16);
                let source = self.offset(x, y);
                let target = y * self.pitch + x;
                linear[target..target + count].copy_from_slice(&stored[source..source + count]);
            }
        }
        Ok(linear)
    }
}
