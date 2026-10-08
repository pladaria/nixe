//! Page demands copy only the corresponding complete image format blocks.
use super::*;
impl WgpuBackendDriver {
    pub(super) fn supports_image_page_writeback(
        &self,
        handle: BackendResourceHandle,
        binding: usize,
        _page: CanonicalPageId,
    ) -> bool {
        let Ok(record) = self.resource_record(handle) else {
            return false;
        };
        let BackendResourceCreateInfo::Image { description, .. } = &record.immutable else {
            return false;
        };
        let Some(domain) = record
            .content
            .as_ref()
            .and_then(|c| c.image_domains.get(binding))
        else {
            return false;
        };
        domain.subresources.layer_count == 1
            && description.extent().depth == 1
            && description.format() != ImageFormat::Rgb565Unorm
            && match domain.layout {
                ImageMemoryLayout::PitchLinear { .. } => true,
                ImageMemoryLayout::BlockLinear(layout) => {
                    layout.block_width_log2 == 0 && layout.block_depth_log2 == 0
                }
            }
    }
    pub(super) fn encode_image_page_writeback(
        &mut self,
        encoder: &mut CommandEncoder,
        handle: BackendResourceHandle,
        binding: usize,
        page: CanonicalPageId,
        output: &mut Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        let (texture, description) = match self.resource(handle)? {
            Resource::Image {
                texture,
                description,
                ..
            } => (texture.clone(), *description),
            _ => return Err(unsupported("image page demand names a non-image")),
        };
        let domain = self
            .resource_record(handle)?
            .content
            .as_ref()
            .unwrap()
            .image_domains[binding]
            .clone();
        let extent = description
            .mip_extent(domain.subresources.mip_level)
            .ok_or_else(|| unsupported("image page mip"))?;
        let [bw, bh] = description.format().block_extent();
        let columns = extent.width.div_ceil(bw);
        let rows = extent.height.div_ceil(bh);
        let bytes = usize::from(
            description
                .format()
                .plane_bytes_per_block(domain.subresources.plane)
                .ok_or_else(|| unsupported("image page format"))?,
        );
        let mut intervals = Vec::new();
        let mut offset = 0;
        for segment in domain.backing.segments() {
            if segment.page() == page {
                intervals.push((offset, vec![0; segment.size() as usize].into_boxed_slice()));
            }
            offset += segment.size();
        }
        let regions = dirty_image_regions(&intervals, domain.layout, columns, rows, bytes)?;
        for region in regions {
            let row_pitch = align_u32(
                region.width * bytes as u32,
                wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
            )?;
            let staging = self.take_readback_buffer(
                u64::from(row_pitch) * u64::from(region.height),
                "Nixe image page readback",
            );
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: u32::from(domain.subresources.mip_level),
                    origin: Origin3d {
                        x: region.x * bw,
                        y: region.y * bh,
                        z: u32::from(domain.subresources.base_layer),
                    },
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_pitch),
                        rows_per_image: Some(region.height),
                    },
                },
                Extent3d {
                    width: region.width * bw,
                    height: region.height * bh,
                    depth_or_array_layers: 1,
                },
            );
            let mut runs = Vec::new();
            for row in 0..region.height {
                let mut byte = 0;
                while byte < region.width as usize * bytes {
                    let x = u64::from(region.x) * bytes as u64 + byte as u64;
                    let y = region.y + row;
                    let (position, count) = match domain.layout {
                        ImageMemoryLayout::PitchLinear { row_pitch, .. } => (
                            u64::from(y) * row_pitch + x,
                            region.width as usize * bytes - byte,
                        ),
                        ImageMemoryLayout::BlockLinear(layout) => (
                            block_linear_byte_offset(
                                layout,
                                (u64::from(columns) * bytes as u64).div_ceil(64),
                                1 << layout.block_height_log2,
                                0,
                                y,
                                x,
                            )? as u64,
                            (16 - x as usize % 16).min(region.width as usize * bytes - byte),
                        ),
                    };
                    let mut base = 0;
                    for segment in domain.backing.segments() {
                        let from = position.max(base);
                        let to = (position + count as u64).min(base + segment.size());
                        if segment.page() == page && from < to {
                            runs.push(PageWriteRun {
                                page_offset: (segment.offset() + from - base) as usize,
                                staging_offset: row as usize * row_pitch as usize
                                    + byte
                                    + (from - position) as usize,
                                size: (to - from) as usize,
                            });
                        }
                        base += segment.size();
                    }
                    byte += count;
                }
            }
            nixe_trace::event("gpu.readback_bytes", 0, staging.size());
            output.push(PendingWriteback::ImagePage {
                staging,
                page,
                runs,
            });
        }
        Ok(())
    }
}
