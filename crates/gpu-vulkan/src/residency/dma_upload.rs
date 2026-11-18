//! Full pitch-to-tiled DMA uploads. RAM and the sampled image receive the same
//! source snapshot. Other GPU images retain their untouched texels until use.
use super::*;
use nixe_gpu::physical::WriteDomain;
use nixe_memory::{ArenaOffset, Extent};
use std::sync::Weak;

struct UploadLayout {
    plane: Arc<Footprint>,
    ranges: Arc<RangeSet>,
    pins: Vec<nixe_memory::AllocationRange>,
}

impl UploadLayout {
    fn new(plane: Arc<Footprint>) -> Result<Self, VulkanError> {
        let mut ranges = RangeSet::default();
        let mut pins = Vec::new();
        for (index, span) in plane.mapping().spans().iter().enumerate() {
            plane.visit(index, span.start, span.bytes, |run| {
                ranges.insert(run.start.get(), run.bytes);
                true
            })?;
        }
        let bounds = plane.bounds();
        plane
            .mapping()
            .pin_into(bounds.start, bounds.end - bounds.start, &mut pins)?;
        Ok(Self {
            plane,
            ranges: Arc::new(ranges),
            pins,
        })
    }
}

#[derive(Clone, Copy)]
struct CopyRun {
    source: usize,
    destination: usize,
    bytes: usize,
}

struct Patch {
    ranges: Arc<RangeSet>,
    upload: Weak<UploadLayout>,
    copies: Vec<CopyRun>,
    regions: Vec<vk::BufferImageCopy>,
    bytes: usize,
}

impl Patch {
    fn build(upload: &Arc<UploadLayout>, entry: &ImageEntry) -> Result<Option<Self>, VulkanError> {
        if entry.description.samples != vk::SampleCountFlags::TYPE_1 || entry.decoded.is_some() {
            return Ok(None);
        }
        let mut patch = Self {
            ranges: Arc::clone(&upload.ranges),
            upload: Arc::downgrade(upload),
            copies: Vec::new(),
            regions: Vec::new(),
            bytes: 0,
        };
        let WriteDomain::Image(source) = upload.plane.domain() else {
            unreachable!()
        };
        for (plane, target) in entry.planes.iter().zip(&entry.contents.geometry) {
            if plane.block_extent.width != 1 || plane.block_extent.height != 1 {
                return Ok(None);
            }
            let pixel = plane.bytes_per_block as usize;
            let mut runs = Vec::new();
            for (ti, t) in target.mapping().spans().iter().enumerate() {
                for (si, s) in upload.plane.mapping().spans().iter().enumerate() {
                    if t.allocation != s.allocation {
                        continue;
                    }
                    let start = s.start.get().max(t.start.get());
                    let stop = (s.start.get() + s.bytes).min(t.start.get() + t.bytes);
                    if start >= stop {
                        continue;
                    }
                    let mut error = None;
                    upload
                        .plane
                        .visit(si, ArenaOffset::new(start), stop - start, |src| {
                            if let Err(err) = target.visit(ti, src.start, src.bytes, |dst| {
                                runs.push(CopyRun {
                                    source: (src.native - source.native_base + dst.start.get()
                                        - src.start.get())
                                        as usize,
                                    destination: (dst.native - plane.offset) as usize,
                                    bytes: dst.bytes as usize,
                                });
                                true
                            }) {
                                error = Some(err);
                                return false;
                            }
                            true
                        })?;
                    if let Some(err) = error {
                        return Err(err.into());
                    }
                }
            }
            runs.sort_unstable_by_key(|r| r.destination);
            let mut i = 0;
            while i < runs.len() {
                let start = runs[i].destination;
                let row = start / plane.row_pitch as usize;
                let x = start % plane.row_pitch as usize;
                let offset = patch.bytes.next_multiple_of(pixel.max(4));
                let mut end = start;
                while i < runs.len()
                    && runs[i].destination == end
                    && end / plane.row_pitch as usize == row
                {
                    let mut run = runs[i];
                    end += run.bytes;
                    run.destination = offset + run.destination - start;
                    if let Some(last) = patch.copies.last_mut()
                        && last.source + last.bytes == run.source
                        && last.destination + last.bytes == run.destination
                    {
                        last.bytes += run.bytes;
                    } else {
                        patch.copies.push(run);
                    }
                    i += 1;
                }
                if !x.is_multiple_of(pixel)
                    || !(end - start).is_multiple_of(pixel)
                    || x + end - start > plane.extent.width as usize * pixel
                    || row >= plane.extent.height as usize
                    || runs.get(i).is_some_and(|r| r.destination < end)
                {
                    return Ok(None);
                }
                patch.bytes = offset + end - start;
                let width = ((end - start) / pixel) as u32;
                let layers = plane.layers();
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset as u64)
                    .image_subresource(layers)
                    .image_offset(vk::Offset3D {
                        x: (x / pixel) as i32,
                        y: row as i32,
                        z: plane.depth_slice as i32,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height: 1,
                        depth: 1,
                    });
                if let Some(last) = patch.regions.last_mut()
                    && last.image_subresource.aspect_mask == layers.aspect_mask
                    && last.image_subresource.mip_level == layers.mip_level
                    && last.image_subresource.base_array_layer == layers.base_array_layer
                    && last.image_offset.x == region.image_offset.x
                    && last.image_offset.z == region.image_offset.z
                    && last.image_offset.y + last.image_extent.height as i32 == row as i32
                    && last.image_extent.width == width
                    && last.buffer_offset
                        + u64::from(width) * u64::from(last.image_extent.height) * pixel as u64
                        == offset as u64
                {
                    last.image_extent.height += 1;
                } else {
                    patch.regions.push(region);
                }
            }
        }
        Ok(Some(patch))
    }
}

#[derive(Default)]
pub(super) struct DmaImage {
    layout: Option<Arc<UploadLayout>>,
    device_ranges: Option<(u64, Arc<RangeSet>)>,
    cached: Vec<Arc<Patch>>,
    pub(super) mixed_authority: bool,
    pending: Vec<(Arc<Patch>, Arc<Vec<u8>>)>,
}
impl DmaImage {
    pub(super) fn pending(&self) -> bool {
        !self.pending.is_empty()
    }
    pub(super) fn discard(&mut self, reason: DmaUse) {
        for (patch, _) in self.pending.drain(..) {
            reason.record(patch.bytes);
        }
    }
    pub(super) fn covers_read(
        &self,
        memory: &nixe_memory::device_memory::DeviceMemory,
        address: u64,
        bytes: u64,
    ) -> bool {
        self.pending()
            && bytes <= nixe_memory::PAGE_SIZE - (address % nixe_memory::PAGE_SIZE)
            && memory.physical_address(address).is_some_and(|start| {
                self.pending
                    .iter()
                    .any(|(p, _)| p.ranges.contains(start.get(), bytes))
            })
    }
}

#[derive(Clone, Copy)]
pub(super) enum DmaUse {
    Graphics,
    Alias,
    Readback,
    Fence,
    Evict,
    Clear,
    Superseded,
    Invalidate,
}
impl DmaUse {
    #[inline]
    fn record(self, _bytes: usize) {
        #[cfg(feature = "performance-counters")]
        nixe_gpu::trace::event("dma_alias_use", self as u64, _bytes as u64, 0);
    }
}

impl Residency<'_> {
    pub(super) fn apply_dma_patches(
        &mut self,
        id: ImageId,
        reason: DmaUse,
    ) -> Result<(), VulkanError> {
        if !self.texture_cache.get(id).dma.pending() {
            return Ok(());
        }
        let scaled = self.texture_cache.get(id).rescaled;
        if scaled {
            self.switch_image_scale(id, false, false, false)?;
        }
        let mut patches = std::mem::take(&mut self.texture_cache.images[id].dma.pending);
        for (patch, source) in patches.drain(..) {
            let slice = self.runtime.device.submission.reserve_slice(
                patch.bytes as u64,
                crate::queue::staging::Usage::Upload,
                16,
            )?;
            // SAFETY: this fresh slice remains owned until the recorded copies finish.
            unsafe {
                self.runtime
                    .device
                    .submission
                    .staging
                    .write_with(&slice, |out| {
                        for run in &patch.copies {
                            out[run.destination..run.destination + run.bytes]
                                .copy_from_slice(&source[run.source..run.source + run.bytes]);
                        }
                        Ok(())
                    })?;
            }
            let planes = self.texture_cache.get(id).planes.clone();
            let (native, layout) = self.image_planes_transfer(id, planes.iter().copied())?;
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            let regions = &mut self.runtime.dma_copy_regions;
            regions.clear();
            regions.extend(patch.regions.iter().map(|region| vk::BufferImageCopy {
                buffer_offset: slice.offset + region.buffer_offset,
                ..*region
            }));
            unsafe {
                self.runtime.device.dispatch.copy_buffer_to_image(
                    self.runtime.device.submission.staging_buffer(&slice).handle,
                    native,
                    layout,
                    regions,
                );
            }
            self.runtime.statistics.cpu_uploaded_bytes += patch.bytes as u64;
            reason.record(patch.bytes);
        }
        self.texture_cache.images[id].dma.pending = patches;
        if scaled {
            self.switch_image_scale(id, true, false, false)?;
        }
        Ok(())
    }

    pub(super) fn upload_clean_dma(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        address: u64,
        bytes: u64,
    ) -> Result<bool, VulkanError> {
        let entry = self.texture_cache.get(id);
        let [geometry] = entry.contents.geometry.as_slice() else {
            return Ok(false);
        };
        let WriteDomain::Image(domain) = geometry.domain() else {
            return Ok(false);
        };
        if entry.planes.len() != 1
            || entry.sample_grid != [1, 1]
            || domain.rect.x != 0
            || domain.rect.y != 0
            || domain.aspect_bytes != domain.packed_bytes
            || domain.aspect_bytes != domain.native_element_bytes
            || domain.rect.width != domain.columns
            || domain.rect.height != domain.rows
            || domain.native_row_pitch
                != u64::from(domain.columns) * u64::from(domain.native_element_bytes)
            || domain.native_row_pitch * u64::from(domain.rows) != bytes
            || geometry.mapping().spans().len() != 1
            || !entry
                .cached
                .as_ref()
                .is_some_and(|c| c.packing() == nixe_gpu::native::ImagePacking::Native)
        {
            return Ok(false);
        }
        let bounds = geometry.bounds();
        let span = &geometry.mapping().spans()[0];
        if span.logical > bounds.start || span.logical + span.bytes < bounds.end {
            return Ok(false);
        }
        if !self.physical_read_is_clean(address, bytes) {
            return Ok(false);
        }
        let layout = if let Some(layout) = &entry.dma.layout {
            Arc::clone(layout)
        } else {
            let layout = Arc::new(UploadLayout::new(Arc::clone(geometry))?);
            self.texture_cache.images[id].dma.layout = Some(Arc::clone(&layout));
            layout
        };
        let generation = self.runtime.device_memory.generation();
        let dma = &mut self.texture_cache.images[id].dma;
        let device_ranges = match &dma.device_ranges {
            Some((stamp, ranges)) if *stamp == generation => Arc::clone(ranges),
            _ => {
                let mut ranges = RangeSet::default();
                for (start, bytes) in layout.ranges.iter() {
                    for (address, bytes) in self.runtime.device_memory.device_ranges(Extent {
                        start: ArenaOffset::new(start),
                        bytes,
                    }) {
                        ranges.insert(address, bytes);
                    }
                }
                let ranges = Arc::new(ranges);
                dma.device_ranges = Some((generation, Arc::clone(&ranges)));
                ranges
            }
        };
        let mut overlaps = smallvec::SmallVec::<[ImageId; 8]>::new();
        for (start, bytes) in device_ranges.iter() {
            for page in start >> 20..=(start + bytes - 1) >> 20 {
                if let Some(ids) = self.texture_cache.image_pages.get(&page) {
                    for &other in ids {
                        if other != id
                            && !overlaps.contains(&other)
                            && self
                                .texture_cache
                                .get(other)
                                .maps
                                .iter()
                                .any(|&(a, b)| device_ranges.intersects(a, b))
                        {
                            overlaps.push(other);
                        }
                    }
                }
            }
        }
        let mut patches = smallvec::SmallVec::<[(ImageId, Arc<Patch>); 4]>::new();
        for &other in &overlaps {
            let entry = self.texture_cache.get(other);
            if !entry.gpu_modified || !entry.synchronized {
                continue;
            }
            let patch = if let Some(p) = entry
                .dma
                .cached
                .iter()
                .find(|p| std::ptr::eq(p.upload.as_ptr(), Arc::as_ptr(&layout)))
            {
                Arc::clone(p)
            } else {
                let Some(p) = Patch::build(&layout, entry)? else {
                    return Ok(false);
                };
                if p.bytes != 0 && !entry.dma.mixed_authority {
                    // The CPU dirty journal rounds to 64 bytes. A partial store
                    // on these pages must preserve the other native texels first.
                    for plane in &entry.contents.geometry {
                        for span in plane.mapping().spans() {
                            self.caches
                                .ordered_store_pages
                                .mark(span.start.get(), span.bytes);
                        }
                    }
                }
                let dma = &mut self.texture_cache.images[other].dma;
                dma.mixed_authority |= p.bytes != 0;
                let p = Arc::new(p);
                let cached = &mut dma.cached;
                cached.retain(|p| p.upload.strong_count() > 0);
                cached.push(Arc::clone(&p));
                p
            };
            if patch.bytes != 0 {
                patches.push((other, patch));
            }
        }
        // Capture before touching the destination, including when the two DMA operands alias.
        let mut source = vec![0; bytes as usize];
        self.runtime
            .device_reader
            .read_block_unsafe(address, &mut source)?;
        let source = Arc::new(source);
        self.texture_cache.supersede_image_downloads(&layout.ranges);
        self.runtime
            .memory
            .write_resources(&layout.pins, |writer| {
                layout
                    .plane
                    .publish_resource(writer, domain.native_base, &source)
            })?;
        for other in overlaps {
            let entry = &mut self.texture_cache.images[other];
            if !entry.gpu_modified && entry.synchronized {
                entry.synchronized = false;
                entry.untrack_cpu(&self.runtime.device_memory);
                entry.dma.discard(DmaUse::Invalidate);
            }
        }
        for (other, patch) in patches {
            let pending = &mut self.texture_cache.images[other].dma.pending;
            pending.retain(|(prior, _)| {
                let replaced = Arc::ptr_eq(&prior.ranges, &patch.ranges)
                    || prior
                        .ranges
                        .iter()
                        .all(|(a, b)| patch.ranges.contains(a, b));
                if replaced {
                    DmaUse::Superseded.record(prior.bytes);
                }
                !replaced
            });
            pending.push((patch, Arc::clone(&source)));
        }
        self.invalidate_cpu_ranges(device_ranges.iter(), RamWrite::MirroredImage)?;
        self.scale_image(id, false, true)?;
        self.prepare_image(id, false, true)?;
        self.texture_cache.advance_image_modification(id);
        let slice = self.runtime.device.submission.reserve_slice(
            bytes,
            crate::queue::staging::Usage::Upload,
            16,
        )?;
        unsafe {
            self.runtime
                .device
                .submission
                .staging
                .write_with(&slice, |out| {
                    out.copy_from_slice(&source);
                    Ok(())
                })?;
        }
        self.record_cpu_image_plane(id, plane, &slice, 0..bytes)?;
        self.runtime.statistics.dma_image_uploads += 1;
        self.runtime.statistics.gpu_copied_bytes += bytes;
        Ok(true)
    }
}
