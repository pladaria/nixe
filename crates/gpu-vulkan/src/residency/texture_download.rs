//! TextureCache owns reactive whole-image reads and preemptive fence downloads.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L614-L664
use super::*;
use crate::transfer::host_read_barrier;

const NVN_READBACK_HISTORY: usize = 16;
const NVN_READBACK_MAX_BYTES: u64 = 1 << 20;

pub(super) struct NvnReadback {
    image: ImageId,
    last_read: u64,
    repeated: bool,
    snapshot: Option<StagedNvnReadback>,
}

struct StagedNvnReadback {
    version: u64,
    mapping_generation: u64,
    download: ImageDownload,
}

impl ImageEntry {
    fn nvn_readback_size(&self) -> Option<u64> {
        let [plane] = self.planes.as_ref() else {
            return None;
        };
        if !self.reactive_download
            || !self.synchronized
            || self.rescaled
            || self.description.samples != vk::SampleCountFlags::TYPE_1
            || self.dma.pending()
            || self.dma.mixed_authority
        {
            return None;
        }
        let plane = self.storage_plane(*plane);
        let size = plane.end()? - plane.offset;
        (size <= NVN_READBACK_MAX_BYTES).then_some(size)
    }
}

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
    superseded: RangeSet,
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
    pub(super) fn forget_nvn_readback(&mut self, id: ImageId) {
        self.nvn_readbacks.retain(|read| read.image != id);
    }

    fn observe_nvn_readback(&mut self, id: ImageId) {
        let entry = self.get(id);
        if entry.nvn_readback_size().is_none() {
            return;
        }
        let version = entry.modification_tick;
        let repeated = if let Some(index) = self.nvn_readbacks.iter().position(|r| r.image == id) {
            let previous = self.nvn_readbacks.remove(index);
            previous.repeated || previous.last_read != version
        } else {
            if self.nvn_readbacks.len() == NVN_READBACK_HISTORY {
                self.nvn_readbacks.remove(0);
            }
            false
        };
        self.nvn_readbacks.push(NvnReadback {
            image: id,
            last_read: version,
            repeated,
            snapshot: None,
        });
    }

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
        let mut exact_read = false;
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
                if entry.dma.covers_read(&self.device_memory, address, bytes) {
                    exact_read = true;
                    continue;
                }
                // Native reads cannot wait for publication while admitted.
                // Leave this image untouched so the suspended caller still
                // observes the required reactive download.
                // NVN renders to an explicit framebuffer without the Maxwell
                // bind-time scheduling of learned fence downloads.
                let preemptive = entry.forced_flushed && !entry.reactive_download;
                if !REACTIVE && !preemptive {
                    return None;
                }
                found = true;
                for &(base, size) in &entry.maps {
                    area.start = area.start.min(base);
                    area.end = area.end.max(base + size);
                }
                if !entry.reactive_download {
                    for &view in &entry.views {
                        self.views[view].downloaded = true;
                    }
                    entry.forced_flushed = true;
                }
                area.preemptive &= preemptive;
            }
        }
        if !found && !exact_read {
            area.start = address / 4096 * 4096;
            area.end = end.div_ceil(4096) * 4096;
        }
        Some(area)
    }
}
impl Residency<'_> {
    pub(crate) fn stage_nvn_readbacks(&mut self) -> Result<(), VulkanError> {
        if self.texture_cache.nvn_readbacks.is_empty() {
            return Ok(());
        }
        let mapping_generation = self.runtime.device_memory.generation();
        for index in 0..self.texture_cache.nvn_readbacks.len() {
            let read = &self.texture_cache.nvn_readbacks[index];
            let id = read.image;
            let entry = self.texture_cache.get(id);
            let version = entry.modification_tick;
            // @SPECULATIVE A texture read back in successive versions is likely
            // to be needed again after the next NVN fence. Stage at most once
            // per observed read, and keep the bytes private until requested.
            if !read.repeated
                || read.snapshot.is_some()
                || !entry.gpu_modified
                || read.last_read == version
            {
                continue;
            }
            let Some(size) = entry.nvn_readback_size() else {
                continue;
            };
            if let Some(download) =
                self.record_texture_downloads(&[id], dma_upload::DmaUse::Fence)?
            {
                nixe_gpu::trace::event("nvn_readback_stage", download.work.tick(), size, 0);
                self.texture_cache.nvn_readbacks[index].snapshot = Some(StagedNvnReadback {
                    version,
                    mapping_generation,
                    download,
                });
            }
        }
        Ok(())
    }

    fn record_texture_downloads(
        &mut self,
        ids: &[ImageId],
        reason: dma_upload::DmaUse,
    ) -> Result<Option<ImageDownload>, VulkanError> {
        if ids.is_empty() {
            return Ok(None);
        }
        self.runtime.device.realize_deferred_clear()?;
        for &id in ids {
            // Readbacks consume original guest texels. Apply partial DMA writes
            // there directly, avoiding a round trip through scaled storage.
            self.scale_image(id, false, false)?;
            self.apply_dma_patches(id, reason)?;
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
                superseded: RangeSet::default(),
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
        if let Some(download) = self.record_texture_downloads(&[id], dma_upload::DmaUse::Evict)? {
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
        let images = self.record_texture_downloads(&ids, dma_upload::DmaUse::Fence)?;
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
    pub(super) fn download_texture_memory(
        &mut self,
        address: u64,
        bytes: u64,
        cpu_read: bool,
    ) -> Result<(), VulkanError> {
        let mut ids = Vec::new();
        for page in address >> 20..=(address + bytes - 1) >> 20 {
            if let Some(entries) = self.texture_cache.image_pages.get(&page) {
                for &id in entries {
                    let entry = self.texture_cache.get(id);
                    if entry.gpu_modified
                        && entry.synchronized
                        && !entry
                            .dma
                            .covers_read(&self.runtime.device_memory, address, bytes)
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
        #[cfg(feature = "performance-counters")]
        let _trace = (!ids.is_empty())
            .then(|| nixe_gpu::trace::span("read_texture", address, bytes, ids.len() as u64));
        self.download_images(ids, cpu_read)
    }
    /// An inline upload updates RAM, whereas a dirty image may hold the only
    /// copy of its other texels. Preserve those texels before the RAM store
    /// and exclude the store itself from older fence snapshots.
    pub(crate) fn prepare_inline_write(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let memory = self.runtime.device_memory.clone();
        let mut ranges = RangeSet::from_range(address, bytes);
        memory.apply_device_aliases(address, bytes, |a, b| ranges.insert(a, b));
        let physical = || -> Result<RangeSet, VulkanError> {
            Ok(memory
                .read_mapping(address, bytes)?
                .spans()
                .iter()
                .map(|span| (span.start.get(), span.bytes))
                .collect())
        };
        if self
            .texture_cache
            .downloads
            .iter()
            .any(|download| download.images.is_some())
        {
            let writes = physical()?;
            self.texture_cache.supersede_image_downloads(&writes);
            self.preserve_texture_writes(&ranges, || Ok(writes))
        } else {
            self.preserve_texture_writes(&ranges, physical)
        }
    }
    pub(super) fn preserve_texture_writes(
        &mut self,
        writes: &RangeSet,
        physical_writes: impl FnOnce() -> Result<RangeSet, VulkanError>,
    ) -> Result<(), VulkanError> {
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
        if !ids.is_empty() {
            let physical = physical_writes()?;
            let mut preserved = Vec::with_capacity(ids.len());
            for id in ids {
                for plane in &self.texture_cache.get(id).contents.geometry {
                    if !plane.covered_by_writes(&physical)? {
                        preserved.push(id);
                        break;
                    }
                }
            }
            return self.download_images(preserved, false);
        }
        Ok(())
    }
    fn download_images(
        &mut self,
        mut ids: Vec<ImageId>,
        cpu_read: bool,
    ) -> Result<(), VulkanError> {
        if ids.is_empty() {
            return Ok(());
        }
        ids.sort_by_key(|&id| self.texture_cache.get(id).modification_tick);
        // Later producers publish last. An older representation fully covered
        // by a later one would only download and copy bytes immediately replaced
        // by that publication. Compare exact physical footprints, including all
        // layers/aspects, before omitting it. Partial aliases retain their order.
        let mut selected = Vec::with_capacity(ids.len());
        for &id in ids.iter().rev() {
            let image = self.texture_cache.get(id);
            let covered = selected.iter().any(|&newer| {
                let newer = self.texture_cache.get(newer);
                image.contents.geometry.iter().all(|plane| {
                    newer
                        .contents
                        .geometry
                        .iter()
                        .any(|other| other.covers(plane))
                })
            });
            if !covered {
                selected.push(id);
            }
        }
        selected.reverse();
        let mut staged = smallvec::SmallVec::<[ImageDownload; 1]>::new();
        if !self.texture_cache.nvn_readbacks.is_empty() {
            let mapping_generation = self.runtime.device_memory.generation();
            for &id in &selected {
                let entry = self.texture_cache.get(id);
                let version = entry.modification_tick;
                if entry.nvn_readback_size().is_none() {
                    continue;
                }
                if let Some(snapshot) = self
                    .texture_cache
                    .nvn_readbacks
                    .iter_mut()
                    .find(|read| read.image == id)
                    .and_then(|read| read.snapshot.take())
                    .filter(|s| s.version == version && s.mapping_generation == mapping_generation)
                {
                    staged.push(snapshot.download);
                }
            }
        }
        for &id in &ids {
            self.texture_cache.images[id].gpu_modified = false;
        }
        if staged.len() == selected.len() {
            let work = staged
                .iter()
                .map(|d| d.work)
                .max_by_key(|w| w.tick())
                .unwrap();
            self.runtime
                .device
                .wait_for_work(work, std::time::Duration::MAX)?;
            nixe_gpu::trace::event("nvn_readback_hit", work.tick(), staged.len() as u64, 0);
            for download in staged {
                self.texture_cache.publish_image_download(download)?;
            }
        } else if let Some(download) = {
            // Preserve the reactive path's publication order when some aliases
            // have no staged version. A single reservation covers that read.
            drop(staged);
            self.record_texture_downloads(&selected, dma_upload::DmaUse::Readback)?
        } {
            self.flush()?;
            self.runtime
                .device
                .wait_for_work(download.work, std::time::Duration::MAX)?;
            self.texture_cache.publish_image_download(download)?;
        }
        if cpu_read {
            for id in selected {
                self.texture_cache.observe_nvn_readback(id);
            }
        }
        Ok(())
    }
}

impl TextureCache {
    // A fence captured before a write-through DMA can still own other bytes
    // of that same image. A subsequent CPU store supersedes those bytes too.
    pub(super) fn supersede_mixed_cpu_store(&mut self, address: u64, end: u64) {
        if self.mixed_download_planes == 0 {
            return;
        }
        let mut writes = RangeSet::default();
        let mut at = address;
        while at < end {
            let bytes = (end - at).min(nixe_memory::PAGE_SIZE - at % nixe_memory::PAGE_SIZE);
            if let Some(start) = self.device_memory.physical_address(at) {
                writes.insert(start.get(), bytes);
            }
            at += bytes;
        }
        self.mask_image_downloads(&writes, true);
    }
    pub(super) fn supersede_image_downloads(&mut self, writes: &RangeSet) {
        self.mask_image_downloads(writes, false);
    }
    fn mask_image_downloads(&mut self, writes: &RangeSet, mixed_only: bool) {
        for snapshot in self.downloads.iter_mut().filter_map(|f| f.images.as_mut()) {
            for plane in &mut snapshot.planes {
                let untouched = plane.superseded.is_empty();
                if mixed_only && untouched {
                    continue;
                }
                for extent in plane
                    .inputs
                    .iter()
                    .flat_map(nixe_memory::AllocationRange::physical_extents)
                {
                    for (at, bytes) in writes.intersections(extent.start.get(), extent.bytes) {
                        plane.superseded.insert(at, bytes);
                    }
                }
                if untouched && !plane.superseded.is_empty() {
                    self.mixed_download_planes += 1;
                }
            }
        }
    }
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
                plane.geometry.publish_resource_except(
                    writer,
                    plane.native_start,
                    source,
                    &plane.superseded,
                )
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
                if self.mixed_download_planes != 0 {
                    self.mixed_download_planes -= images
                        .planes
                        .iter()
                        .filter(|p| !p.superseded.is_empty())
                        .count();
                }
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
