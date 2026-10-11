//! Canonical Tegra surface interpretations shared by NVDEC and VIC.
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
        plane: bool,
    ) -> Result<nixe_gpu::ConversionImage, String> {
        use nixe_gpu::*;
        let (whole, offset) = region(mappings, address, self.length())?;
        let (_, mapping) = mappings
            .range(..=u32::try_from(address).map_err(|_| "VIC address exceeds 32 bits")?)
            .next_back()
            .ok_or("VIC mapping was lost")?;
        let allocation =
            GpuAllocationDescription::new(whole.size(), 1).map_err(|e| e.to_string())?;
        let bytes = usize::from(
            format
                .plane_bytes_per_texel(0)
                .ok_or("invalid VIC GPU image format")?,
        );
        // NVDEC may allocate a wider GOB stride than its visible source. Plane
        // textures retain that stride; the validated source crop selects texels.
        let width = if plane && self.kind == 1 {
            self.pitch / bytes
        } else {
            self.width / if plane { bytes } else { 1 }
        };
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(width as u32, self.height as u32, 1).map_err(|e| e.to_string())?,
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
                .snapshot_subrange(offset, self.length() as u64)
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
        Ok(ConversionImage {
            description,
            allocation,
            view,
        })
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
