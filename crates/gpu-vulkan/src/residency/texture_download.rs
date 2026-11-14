//! TextureCache owns reactive whole-image reads and preemptive fence downloads.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L614-L664
use super::*;
use crate::transfer::host_read_barrier;

pub(super) struct ImageDownload {
    work: WorkTicket,
    ticket: crate::ReadbackTicket,
    planes: Vec<PlaneDownload>,
    // Keep source images alive until their copies and publication retire.
    _sources: Vec<Arc<AllocatedImage>>,
}
// Pending image reads are captured together at the fence; DMA buffers already
// own their command-time staging. One ordered prefix contains both kinds.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L865-L950
pub(super) enum PendingDownload {
    Image(ImageId),
    Buffer(buffer_download::DownloadEnd),
}
enum Publication {
    Image(std::ops::Range<usize>),
    Buffer(buffer_download::DownloadEnd),
}
pub(super) struct Download {
    images: Option<ImageDownload>,
    publications: Vec<Publication>,
}
impl Download {
    pub(super) fn work(&self) -> Option<WorkTicket> {
        self.images
            .as_ref()
            .map(|image| image.work)
            .into_iter()
            .chain(
                self.publications
                    .iter()
                    .filter_map(|publication| match publication {
                        Publication::Image(_) => None,
                        Publication::Buffer(buffer) => Some(buffer.work()),
                    }),
            )
            .max_by_key(|work| work.tick())
    }
}
struct PlaneDownload {
    source: u64,
    size: u64,
    native_start: u64,
    geometry: Arc<Footprint>,
    inputs: Vec<nixe_memory::AllocationRange>,
}
impl Residency<'_> {
    pub(crate) fn bind_render_target(&mut self, slot: usize, selected: Option<ImageViewId>) {
        if self.texture_cache.bindings.channel.render_targets.views[slot] == selected {
            return;
        }
        if let Some(view) = selected
            && self.texture_cache.views[view].downloaded
        {
            let image = self.texture_cache.views[view].image;
            self.texture_cache
                .pending_downloads
                .push(PendingDownload::Image(image));
        }
        let bank = &mut self.texture_cache.bindings.channel;
        bank.render_targets.views[slot] = selected;
        bank.render_targets_changed();
    }
}
impl TextureCache {
    pub(super) fn flush_area<const REACTIVE: bool>(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Option<nixe_memory::NativeReadArea> {
        let end = address + bytes;
        let mut area = nixe_memory::NativeReadArea {
            start: address,
            end,
            preemptive: true,
        };
        let mut found = false;
        for page in address >> 20..=(end - 1) >> 20 {
            let Some(ids) = self.image_pages.get(&page) else {
                continue;
            };
            for &id in ids {
                let entry = &mut self.images[id];
                if !entry.gpu_modified
                    || !entry
                        .maps
                        .iter()
                        .any(|&(base, size)| base < end && address < base + size)
                {
                    continue;
                }
                // Native reads cannot wait for publication while admitted.
                // Leave this image untouched so the suspended caller still
                // observes the required reactive download.
                if !REACTIVE && !entry.forced_flushed {
                    return None;
                }
                found = true;
                for &(base, size) in &entry.maps {
                    area.start = area.start.min(base);
                    area.end = area.end.max(base + size);
                }
                for &view in &entry.views {
                    self.views[view].downloaded = true;
                }
                area.preemptive &= entry.forced_flushed;
                entry.forced_flushed = true;
            }
        }
        if !found {
            area.start = address / 4096 * 4096;
            area.end = end.div_ceil(4096) * 4096;
        }
        Some(area)
    }
}
impl Residency<'_> {
    fn record_texture_downloads(
        &mut self,
        ids: &[ImageId],
    ) -> Result<Option<ImageDownload>, VulkanError> {
        if ids.is_empty() {
            return Ok(None);
        }
        self.runtime.device.realize_deferred_clear()?;
        for &id in ids {
            self.scale_image(id, false, false)?;
        }
        let sources = ids
            .iter()
            .map(|&id| Arc::clone(&self.texture_cache.get(id).native))
            .collect();
        let mut descriptions = Vec::new();
        let mut total = 0u64;
        for &id in ids {
            let entry = self.texture_cache.get(id);
            for (index, &plane) in entry.planes.iter().enumerate() {
                let plane = entry.storage_plane(plane);
                let size = plane.end().unwrap() - plane.offset;
                total = crate::queue::align_up(total, 64).ok_or(VulkanError::InvalidResource(
                    "image download offset overflows",
                ))?;
                descriptions.push((id, index, plane, total, size));
                total = total.checked_add(size).ok_or(VulkanError::InvalidResource(
                    "image download size overflows",
                ))?;
            }
        }
        if total == 0 {
            return Ok(None);
        }
        let mut converted = Vec::new();
        for &(id, _, plane, _, _) in &descriptions {
            if self.texture_cache.get(id).description.samples == vk::SampleCountFlags::TYPE_1 {
                converted.push(None);
            } else {
                converted.push(Some(self.sample_readback_plane(id, plane)?));
            }
        }
        let (slice, ticket) = self
            .runtime
            .device
            .submission
            .reserve_readback_aligned(total, 64)?;
        let buffer = self.runtime.device.submission.staging_buffer(&slice).handle;
        let mut planes = Vec::new();
        for ((id, index, plane, offset, size), converted) in descriptions.into_iter().zip(converted)
        {
            let (native, layout, aspect, layer, mip, depth) = if let Some(attachment) = converted {
                let (native, range) = self.transfer_attachment(attachment)?;
                (native, range, vk::ImageAspectFlags::COLOR, 0, 0, 0)
            } else {
                let (native, layout) = self.image_planes_transfer(id, [plane])?;
                (
                    native,
                    layout,
                    plane.aspect,
                    plane.array_layer,
                    plane.mip_level,
                    plane.depth_slice,
                )
            };
            let row_length = (plane.row_pitch / u64::from(plane.bytes_per_block)) as u32
                * plane.block_extent.width;
            let region = vk::BufferImageCopy::default()
                .buffer_offset(slice.offset + offset)
                .buffer_row_length(row_length)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect)
                        .mip_level(mip)
                        .base_array_layer(layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: 0,
                    y: 0,
                    z: depth as i32,
                })
                .image_extent(vk::Extent3D {
                    width: plane.extent.width,
                    height: plane.extent.height,
                    depth: 1,
                });
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            unsafe {
                self.runtime.device.dispatch.copy_image_to_buffer(
                    native,
                    layout,
                    buffer,
                    &[region],
                );
            }
            if let Some(attachment) = converted {
                self.release_attachment(attachment)?;
            }
            let geometry = Arc::clone(&self.texture_cache.get(id).contents.geometry[index]);
            let bounds = geometry.bounds();
            let mut inputs = Vec::new();
            for span in geometry.mapping().spans() {
                let start = span.logical.max(bounds.start);
                let end = (span.logical + span.bytes).min(bounds.end);
                if start < end {
                    geometry
                        .mapping()
                        .pin_into(start, end - start, &mut inputs)?;
                }
            }
            planes.push(PlaneDownload {
                source: offset,
                size,
                native_start: plane.offset,
                geometry,
                inputs,
            });
        }
        host_read_barrier(&mut self.runtime.device, buffer, slice.offset, total);
        Ok(Some(ImageDownload {
            work: ticket.work(),
            ticket,
            planes,
            _sources: sources,
        }))
    }
    pub(super) fn download_image_memory(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let entry = self.texture_cache.get(id);
        if !entry.gpu_modified || !entry.synchronized {
            return Ok(());
        }
        if let Some(download) = self.record_texture_downloads(&[id])? {
            self.flush()?;
            self.runtime
                .device
                .wait_for_work(download.work, std::time::Duration::MAX)?;
            self.texture_cache.publish_image_download(download)?;
            self.texture_cache.images[id].gpu_modified = false;
        }
        Ok(())
    }
    pub(crate) fn commit_texture_downloads(&mut self) -> Result<(), VulkanError> {
        let mut pending = std::mem::take(&mut self.texture_cache.pending_downloads);
        let ids: Vec<_> = pending
            .iter()
            .filter_map(|download| match download {
                PendingDownload::Image(id) => Some(*id),
                PendingDownload::Buffer(_) => None,
            })
            .collect();
        // All non-DMA image planes share one 64-byte-aligned reservation.
        let images = self.record_texture_downloads(&ids)?;
        let mut publications = Vec::with_capacity(pending.len());
        let mut first_plane = 0;
        for download in pending.drain(..) {
            publications.push(match download {
                PendingDownload::Image(id) => {
                    let end = first_plane + self.texture_cache.get(id).planes.len();
                    let range = first_plane..end;
                    first_plane = end;
                    Publication::Image(range)
                }
                PendingDownload::Buffer(buffer) => Publication::Buffer(buffer),
            });
        }
        self.texture_cache.downloads.push_back(Download {
            images,
            publications,
        });
        self.texture_cache.pending_downloads = pending;
        Ok(())
    }
    pub(crate) fn visibility_work(&self) -> Option<WorkTicket> {
        self.buffer_cache
            .visibility_work()
            .into_iter()
            .chain(self.texture_visibility_work())
            .max_by_key(|work| work.tick())
    }

    pub(crate) fn texture_visibility_work(&self) -> Option<WorkTicket> {
        self.texture_cache.downloads.back().and_then(Download::work)
    }
    pub(crate) fn download_texture_memory(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let mut ids = Vec::new();
        for page in address >> 20..=(address + bytes - 1) >> 20 {
            if let Some(entries) = self.texture_cache.image_pages.get(&page) {
                for &id in entries {
                    let entry = self.texture_cache.get(id);
                    if entry.gpu_modified
                        && entry.synchronized
                        && !ids.contains(&id)
                        && entry
                            .maps
                            .iter()
                            .any(|&(base, size)| base < address + bytes && address < base + size)
                    {
                        ids.push(id);
                    }
                }
            }
        }
        self.download_images(ids)
    }
    pub(super) fn preserve_texture_writes(&mut self, writes: &RangeSet) -> Result<(), VulkanError> {
        let mut ids = Vec::new();
        for (address, bytes) in writes.iter() {
            for page in address >> 20..=(address + bytes - 1) >> 20 {
                if let Some(entries) = self.texture_cache.image_pages.get(&page) {
                    for &id in entries {
                        let entry = self.texture_cache.get(id);
                        if entry.gpu_modified
                            && entry.synchronized
                            && !ids.contains(&id)
                            && entry.maps.iter().any(|&(a, b)| writes.intersects(a, b))
                            && !entry.maps.iter().all(|&(a, b)| writes.contains(a, b))
                        {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        self.download_images(ids)
    }
    fn download_images(&mut self, mut ids: Vec<ImageId>) -> Result<(), VulkanError> {
        ids.sort_by_key(|&id| self.texture_cache.get(id).modification_tick);
        for &id in &ids {
            self.texture_cache.images[id].gpu_modified = false;
        }
        if let Some(download) = self.record_texture_downloads(&ids)? {
            self.flush()?;
            self.runtime
                .device
                .wait_for_work(download.work, std::time::Duration::MAX)?;
            self.texture_cache.publish_image_download(download)?;
        }
        Ok(())
    }
}

impl TextureCache {
    fn publish_image_download(&mut self, download: ImageDownload) -> Result<(), VulkanError> {
        let data = download.ticket.completed()?;
        // DownloadMemory orders images by modification tick, then consumes
        // FullDownloadCopies in image order. Later aliases publish last.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L614-L639
        for plane in &download.planes {
            self.publish_plane(&data, plane)?;
        }
        Ok(())
    }
    fn publish_plane(&mut self, data: &[u8], plane: &PlaneDownload) -> Result<(), VulkanError> {
        // Publication changes streamed RAM too, including physical buffer aliases.
        self.uniform_sync
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        let source = &data[plane.source as usize..(plane.source + plane.size) as usize];
        self.device_memory
            .machine()
            .write_resources(&plane.inputs, |writer| {
                plane
                    .geometry
                    .publish_resource(writer, plane.native_start, source)
            })?;
        self.native_readback_bytes += plane.size;
        self.canonical_readback_bytes += plane.size;

        Ok(())
    }
    pub(crate) fn pop_async_flush(&mut self, buffers: &mut BufferCache) -> Result<(), VulkanError> {
        if let Some(download) = self.downloads.pop_front() {
            // Completed staging is borrowed once and retained through the whole
            // prefix. Publication neither flushes nor waits for another fence.
            let (data, planes, _sources) = if let Some(images) = download.images {
                (
                    Some(images.ticket.completed()?),
                    images.planes,
                    images._sources,
                )
            } else {
                (None, Vec::new(), Vec::new())
            };
            for publication in download.publications.into_iter().rev() {
                match publication {
                    Publication::Image(range) => {
                        // PopAsyncFlushes reverses pending downloads; each
                        // image still consumes its FullDownloadCopies in order.
                        for plane in &planes[range] {
                            self.publish_plane(
                                data.as_ref().expect("captured image staging"),
                                plane,
                            )?;
                        }
                    }
                    Publication::Buffer(buffer) => buffers.publish_dma_download(buffer)?,
                }
            }
        }
        Ok(())
    }
}
