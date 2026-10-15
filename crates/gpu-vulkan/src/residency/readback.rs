//! Bounded, versioned row windows amortize neighboring canonical CPU faults.
//! Missing windows share staging batches; only authoritative bytes are published.
use super::image::ImagePiece;
use super::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WindowKey {
    image: ImageId,
    version: u64,
    mip: u32,
    layer: u32,
    aspect: u32,
    row: u32,
}
struct WindowPlane {
    plane: ImagePlane,
    end: u64,
    row_bytes: u64,
    rows: u32,
    blocks: u32,
    window_shift: u32,
}
impl WindowPlane {
    fn new(plane: ImagePlane, limit: u64) -> Self {
        let row_bytes = plane.row_bytes();
        let rows = plane.rows();
        // Leave staging space for overlapping producers/aspects. Power-of-two
        // windows preserve locality and make row selection a shift per run.
        let capacity = (limit / 4 / row_bytes).max(1).min(u64::from(rows));
        Self {
            plane,
            end: plane.end().unwrap(),
            row_bytes,
            rows,
            blocks: plane.extent.width.div_ceil(plane.block_extent.width),
            window_shift: capacity.ilog2(),
        }
    }
}
struct Scatter {
    source: usize,
    destination: usize,
    length: usize,
}
#[derive(Default)]
pub(super) struct ImageReadbacks {
    windows: FxHashMap<WindowKey, Box<[u8]>>,
    order: VecDeque<WindowKey>,
    bytes: usize,
}
impl VulkanResidency {
    pub(super) fn cached_image_authorities(
        &mut self,
        id: ImageId,
        writes: &[Authority],
        target: &mut [u8],
    ) -> Result<bool, VulkanError> {
        let entry = self.image(id)?;
        let multisample = entry.description.samples != vk::SampleCountFlags::TYPE_1;
        let limit = self.device.submission.configuration.arena_bytes;
        // Exact regions remain necessary for partial images and rows wider than
        // staging. Prefetch never consumes undefined texels.
        if !entry.discarded.is_empty()
            || entry
                .planes
                .iter()
                .any(|p| entry.storage_plane(*p).row_bytes() > limit)
        {
            return Ok(false);
        }
        let version = entry.version;
        let planes: Vec<_> = entry
            .planes
            .iter()
            .map(|&plane| WindowPlane::new(entry.storage_plane(plane), limit))
            .collect();
        let mut needed = FxHashMap::default();
        for write in writes {
            image_chunks(
                id,
                version,
                &planes,
                write,
                |key, piece, source, destination, length| {
                    needed
                        .entry(key)
                        .or_insert_with(|| (piece, Vec::new()))
                        .1
                        .push(Scatter {
                            source: source as usize,
                            destination: destination as usize,
                            length: length as usize,
                        });
                },
            )?;
        }
        let bytes: u64 = needed.values().map(|(p, _)| p.length).sum();
        // A page whose projected native working set cannot fit the bounded
        // cache uses exact bounded regions rather than thrashing the directory.
        if bytes > limit.saturating_mul(2) || needed.len() > 32 {
            return Ok(false);
        }
        let mut missing: Vec<_> = needed
            .iter()
            .filter(|(key, _)| !self.image_readbacks.windows.contains_key(key))
            .map(|(&key, (piece, _))| (key, *piece))
            .collect();
        missing.sort_unstable_by_key(|(_, piece)| piece.logical);
        let mut next = 0;
        while next < missing.len() {
            let first = next;
            let mut bytes = 0;
            let mut alignment = 4;
            let mut offsets = Vec::new();
            while let Some((_, piece)) = missing.get(next) {
                let align = u64::from(piece.plane.bytes_per_block).max(4);
                let offset = crate::queue::align_up(bytes, align).ok_or(
                    VulkanError::InvalidResource("image readback alignment overflow"),
                )?;
                if offset + piece.length > limit {
                    break;
                }
                offsets.push(offset);
                alignment = alignment.max(align);
                bytes = offset + piece.length;
                next += 1;
            }
            let batch = &missing[first..next];
            let pieces: Vec<_> = batch.iter().map(|(_, p)| *p).collect();
            let mut converted = Vec::new();
            let ordinary = if multisample {
                for piece in &pieces {
                    converted.push(self.sample_readback_window(id, *piece)?);
                }
                None
            } else {
                Some(self.image_regions_access(id, &pieces, vk::AccessFlags2::TRANSFER_READ)?)
            };
            let (index, slice, ticket) = self
                .device
                .submission
                .reserve_readback_aligned(bytes, alignment)?;
            let command = self.device.submission.slots[index].command;
            let buffer = self.device.submission.slots[index]
                .readback_buffer(slice)
                .handle;
            if let Some((image, layout)) = ordinary {
                let regions: Vec<_> = pieces
                    .iter()
                    .zip(&offsets)
                    .map(|(piece, offset)| piece.region(slice.offset + offset))
                    .collect();
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device
                        .owner
                        .device
                        .cmd_copy_image_to_buffer(command, image, layout, buffer, &regions);
                }
            } else {
                for ((attachment, piece), offset) in
                    converted.into_iter().zip(&pieces).zip(&offsets)
                {
                    let image = self.attachment_access(
                        attachment,
                        AccessScope::new(
                            vk::PipelineStageFlags2::COPY,
                            vk::AccessFlags2::TRANSFER_READ,
                        ),
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    )?;
                    self.barriers.emit(&mut self.device);
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(slice.offset + offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: piece.blocks,
                            height: piece.rows,
                            depth: 1,
                        });
                    unsafe {
                        self.device.owner.device.cmd_copy_image_to_buffer(
                            command,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            buffer,
                            &[region],
                        );
                    }
                    self.release_attachment(attachment)?;
                }
            }
            super::recording::host_read_barrier(&self.device, command, buffer, slice.offset, bytes);
            self.flush()?;
            let data = self.device.readback(ticket)?;
            self.statistics.native_readback_bytes += bytes;
            for ((key, piece), offset) in batch.iter().zip(offsets) {
                let bytes: Box<[u8]> =
                    data[offset as usize..(offset + piece.length) as usize].into();
                let cache = &mut self.image_readbacks;
                while cache.bytes + bytes.len() > limit.saturating_mul(2) as usize
                    || cache.windows.len() >= 32
                {
                    let index = cache
                        .order
                        .iter()
                        .position(|key| !needed.contains_key(key))
                        .expect("bounded working set leaves an evictable window");
                    let old = cache.order.remove(index).unwrap();
                    cache.bytes -= cache.windows.remove(&old).unwrap().len();
                }
                cache.bytes += bytes.len();
                cache.order.push_back(*key);
                cache.windows.insert(*key, bytes);
            }
        }
        // Authority intervals are disjoint canonical destinations. Grouping
        // their copies by window resolves geometry and the cache lookup once,
        // rather than repeating both for every small tiled run during scatter.
        for (key, (_, copies)) in needed {
            let bytes = &self.image_readbacks.windows[&key];
            for copy in copies {
                target[copy.destination..copy.destination + copy.length]
                    .copy_from_slice(&bytes[copy.source..copy.source + copy.length]);
            }
        }
        Ok(true)
    }
}
fn image_chunks(
    id: ImageId,
    version: u64,
    planes: &[WindowPlane],
    write: &Authority,
    mut visit: impl FnMut(WindowKey, ImagePiece, u64, u64, u64),
) -> Result<(), VulkanError> {
    let mut copied = 0;
    while copied < write.end - write.start {
        let logical = write.offset + copied;
        let window = planes
            .iter()
            .find(|p| logical >= p.plane.offset && logical < p.end)
            .ok_or(VulkanError::InvalidResource(
                "image producer crosses a missing plane",
            ))?;
        let p = window.plane;
        let row = ((logical - p.offset) / p.row_pitch) as u32;
        let x = (logical - p.offset) % p.row_pitch;
        if x >= window.row_bytes {
            return Err(VulkanError::InvalidResource(
                "image producer crosses row padding",
            ));
        }
        let first = row >> window.window_shift << window.window_shift;
        let rows = (1u32 << window.window_shift).min(window.rows - first);
        let piece = ImagePiece {
            plane: p,
            row: first,
            rows,
            first: 0,
            blocks: window.blocks,
            skip: 0,
            logical: p.offset + u64::from(first) * p.row_pitch,
            length: window.row_bytes * u64::from(rows),
        };
        let key = WindowKey {
            image: id,
            version,
            mip: p.mip_level,
            layer: p.array_layer,
            aspect: p.aspect.as_raw(),
            row: first,
        };
        let source = u64::from(row - first) * window.row_bytes + x;
        let available = if p.row_pitch == window.row_bytes {
            piece.length - source
        } else {
            window.row_bytes - x
        };
        let length = available.min(write.end - write.start - copied);
        visit(key, piece, source, write.start + copied, length);
        copied += length;
    }
    Ok(())
}
