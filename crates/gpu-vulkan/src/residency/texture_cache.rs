//! Eden's DAddr page table selects ImageInfo before building physical planes.
//! Entries and views retire with the actual native image, including sparse maps.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L1214-L1273
use super::*;
use nixe_gpu::{ImageDimension, ImageMemoryLayout};
use nixe_gpu_maxwell::engines::threed::ImageInfo;
use nixe_gpu_maxwell::{MaxwellAddressSpaceId, MaxwellGpuMemory};
const PAGE_BITS: u32 = 20;
pub struct ImageRegistration {
    pub(super) info: ImageInfo,
    pub(super) address_space: MaxwellAddressSpaceId,
    pub(super) maps: Vec<(u64, u64)>,
    pub(super) cpu_address: u64,
}
pub(super) struct CachedImage {
    info: ImageInfo,
    address_space: MaxwellAddressSpaceId,
    remapped: bool,
    picked: bool,
    cpu_address: u64,
    slices: smallvec::SmallVec<[(u64, [u32; 2]); 16]>,
}
fn failure(error: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}
fn image_type(dimension: ImageDimension) -> u8 {
    match dimension {
        ImageDimension::One => 1,
        ImageDimension::Two | ImageDimension::Cube => 2,
        ImageDimension::Three => 3,
    }
}
impl CachedImage {
    pub(super) fn can_rescale(&self, depth: bool) -> bool {
        let info = &self.info;
        matches!(info.layout, ImageMemoryLayout::BlockLinear(block) if block.block_depth_log2 == 0)
            && info.description.dimension() == ImageDimension::Two
            && info.description.mip_levels() == 1
            && (info.description.extent().height > 288 || depth)
    }
    pub(super) fn new(
        info: ImageInfo,
        address_space: MaxwellAddressSpaceId,
        cpu_address: u64,
    ) -> Self {
        let mut slices = smallvec::SmallVec::new();
        if info.description.dimension() == ImageDimension::Three {
            let [bw, bh] = info.description.format().block_extent();
            let bytes = u64::from(info.description.format().plane_bytes_per_block(0).unwrap());
            for (level, &(mip_offset, layout)) in info.mip_layouts.iter().enumerate() {
                let ImageMemoryLayout::BlockLinear(block) = layout else {
                    unreachable!("volume layout")
                };
                let size = info.description.mip_extent(level as u8).unwrap();
                let gob_shift = block.block_height_log2 + 9;
                let slice_size = ((u64::from(size.width.div_ceil(bw)) * bytes).div_ceil(64)
                    * u64::from(size.height.div_ceil(bh)).div_ceil(8 << block.block_height_log2))
                    << gob_shift;
                let z_mask = (1u64 << block.block_depth_log2) - 1;
                for slice in 0..u64::from(size.depth) {
                    let offset = mip_offset
                        + ((slice & z_mask) << gob_shift)
                        + (slice & !z_mask) * slice_size;
                    slices.push((offset, [level as u32, slice as u32]));
                }
            }
        }
        Self {
            info,
            address_space,
            remapped: false,
            picked: false,
            cpu_address,
            slices,
        }
    }
    fn dma_base(
        &self,
        address: u64,
        width: u32,
        height: u32,
        samples: [u32; 2],
    ) -> Option<[u32; 2]> {
        let info = &self.info;
        if !matches!(info.layout, ImageMemoryLayout::BlockLinear(_))
            || info.description.dimension() == ImageDimension::Three
        {
            return None;
        }
        let diff = address.checked_sub(info.address)?;
        if diff >= info.size {
            return None;
        }
        let layer = diff.checked_div(info.layer_stride).unwrap_or(0);
        let offset = diff.checked_rem(info.layer_stride).unwrap_or(diff);
        if layer >= u64::from(info.description.array_layers()) {
            return None;
        }
        let level = info
            .mip_layouts
            .iter()
            .position(|(base, _)| *base == offset)?;
        let size = info.description.mip_extent(level as u8)?;
        let bpp = u64::from(info.description.format().plane_bytes_per_block(0)?);
        // IsSubCopy's BPP-relaxed comparison uses GOB-aligned byte widths and
        // eight-row heights; DMA's image operand has one byte per element.
        if (u64::from(size.width) * u64::from(samples[0]) * bpp).div_ceil(64)
            != u64::from(width).div_ceil(64)
            || (u64::from(size.height) * u64::from(samples[1])).div_ceil(8)
                != u64::from(height).div_ceil(8)
        {
            return None;
        }
        Some([level as u32, layer as u32])
    }
    fn base(&self, other: &ImageInfo) -> Option<[u32; 2]> {
        let existing = &self.info;
        let a = existing.description;
        let b = other.description;
        if image_type(a.dimension()) != image_type(b.dimension()) || a.samples() != b.samples() {
            return None;
        }
        let diff = other.address.checked_sub(existing.address)?;
        if diff >= existing.size {
            return None;
        }
        match (existing.layout, other.layout) {
            (
                ImageMemoryLayout::PitchLinear { row_pitch: x, .. },
                ImageMemoryLayout::PitchLinear { row_pitch: y, .. },
            ) => {
                return (diff == 0 && x == y && a.extent() == b.extent()).then_some([0, 0]);
            }
            (ImageMemoryLayout::BlockLinear(_), ImageMemoryLayout::BlockLinear(_)) => (),
            _ => return None,
        }
        let (level, layer) = if a.dimension() == ImageDimension::Three {
            let &(_, [level, layer]) = self.slices.iter().find(|(offset, _)| *offset == diff)?;
            (level, u64::from(layer))
        } else {
            let layer = diff.checked_div(existing.layer_stride).unwrap_or(0);
            let offset = diff.checked_rem(existing.layer_stride).unwrap_or(diff);
            let level = existing
                .mip_layouts
                .iter()
                .position(|(offset_, _)| *offset_ == offset)? as u32;
            (level, layer)
        };
        let size = a.mip_extent(level as u8)?;
        if size.width != b.extent().width
            || size.height != b.extent().height
            || level + u32::from(b.mip_levels()) > u32::from(a.mip_levels())
            || (a.dimension() != ImageDimension::Three
                && layer + u64::from(b.array_layers()) > u64::from(a.array_layers()))
            || u64::from(size.depth)
                < u64::from(b.extent().depth)
                    + if a.dimension() == ImageDimension::Three {
                        layer
                    } else {
                        0
                    }
        {
            return None;
        }
        if existing.layer_stride != 0
            && other.layer_stride != 0
            && existing.layer_stride != other.layer_stride
            && existing.maybe_unaligned_layer_stride != other.maybe_unaligned_layer_stride
        {
            return None;
        }
        Some([level, layer as u32])
    }
}
impl TextureCache {
    pub(super) fn region_gpu_modified(&self, address: u64, bytes: u64) -> bool {
        for page in address >> PAGE_BITS..(address + bytes).div_ceil(1 << PAGE_BITS) {
            if let Some(images) = self.image_pages.get(&page) {
                for &id in images {
                    let entry = self.get(id);
                    if entry.gpu_modified
                        && entry
                            .maps
                            .iter()
                            .any(|&(a, b)| a < address + bytes && address < a + b)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
    pub(super) fn track_image_cpu(&mut self, id: ImageId) -> Result<(), VulkanError> {
        if self.get(id).cpu_tracked {
            return Ok(());
        }
        let entry = &mut self.images[id];
        for &(address, bytes) in &entry.maps {
            self.device_memory
                .update_pages_cached_count(address, bytes, 1)?;
        }
        entry.cpu_tracked = true;
        Ok(())
    }
    pub(super) fn register_image_mappings(&mut self, id: ImageId) {
        if let Some(cached) = &self.images[id].cached {
            for page in cached.info.address >> PAGE_BITS
                ..(cached.info.address + cached.info.size).div_ceil(1 << PAGE_BITS)
            {
                self.gpu_image_pages
                    .entry((cached.address_space, page))
                    .or_default()
                    .push(id);
            }
        }
        let entry = &self.images[id];
        let unbacked = entry
            .cached
            .as_ref()
            .filter(|_| entry.maps.is_empty())
            .map(|cached| (cached.cpu_address, cached.info.size));
        for (address, size) in entry.maps.iter().copied().chain(unbacked) {
            for page in address >> PAGE_BITS..=(address + size - 1) >> PAGE_BITS {
                let entries = self.image_pages.entry(page).or_default();
                if !entries.contains(&id) {
                    entries.push(id);
                }
            }
        }
    }
    pub(super) fn unregister_image_mappings(&mut self, id: ImageId) {
        let entry = &mut self.images[id];
        entry.untrack_cpu(&self.device_memory);
        if let Some(cached) = &entry.cached {
            for page in cached.info.address >> PAGE_BITS
                ..(cached.info.address + cached.info.size).div_ceil(1 << PAGE_BITS)
            {
                let key = (cached.address_space, page);
                if let Some(ids) = self.gpu_image_pages.get_mut(&key) {
                    ids.retain(|&image| image != id);
                    if ids.is_empty() {
                        self.gpu_image_pages.remove(&key);
                    }
                }
            }
        }
        let unbacked = entry
            .cached
            .as_ref()
            .filter(|_| entry.maps.is_empty())
            .map(|cached| (cached.cpu_address, cached.info.size));
        for (address, size) in entry.maps.iter().copied().chain(unbacked) {
            for page in address >> PAGE_BITS..=(address + size - 1) >> PAGE_BITS {
                if let Some(ids) = self.image_pages.get_mut(&page) {
                    ids.retain(|&image| image != id);
                    if ids.is_empty() {
                        self.image_pages.remove(&page);
                    }
                }
            }
        }
    }
}
impl TextureCache {
    pub(super) fn delete_image(&mut self, id: ImageId) {
        self.unregister_image_aliases(id);
        self.unregister_image_mappings(id);
        let view_ids = std::mem::take(&mut self.images[id].views);
        self.bindings.invalidate_image_views(&view_ids);
        self.remove_framebuffers(&view_ids);
        self.lru.unlink(&mut self.images, id);
        self.binding_revision = self
            .binding_revision
            .checked_add(1)
            .expect("native binding revision exhausted");
        // These are uncommitted preemptive image reads. Literal DMA snapshots
        // and already committed image publications retain their own storage.
        self.pending_downloads.retain(|pending| {
            !matches!(pending,
            texture_download::PendingDownload::Image(image) if *image == id)
        });
        let entry = self.images.erase(id);
        let views = view_ids
            .iter()
            .map(|&id| self.views.erase(id).native)
            .collect();
        self.retired
            .push(entry.native, views, entry.contents.last_use);
        self.retired
            .push(entry.original, Vec::new(), entry.contents.last_use);
        if let Some(scaled) = entry.scaled {
            self.retired
                .push(scaled, Vec::new(), entry.contents.last_use);
        }
    }
    /// TextureCache::UnmapMemory unregisters actual DAddr overlaps and erases
    /// their logical slots. Native images/views survive in the destruction ring.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L667-L679
    pub(super) fn unmap_memory(&mut self, address: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = address + bytes;
        let mut deleted = smallvec::SmallVec::<[ImageId; 16]>::new();
        for page in address >> PAGE_BITS..(end.div_ceil(1 << PAGE_BITS)) {
            let Some(ids) = self.image_pages.get(&page) else {
                continue;
            };
            for &id in ids {
                if !deleted.contains(&id)
                    && self.images[id]
                        .maps
                        .iter()
                        .any(|&(base, size)| base < end && address < base + size)
                {
                    deleted.push(id);
                }
            }
        }
        for id in deleted {
            self.delete_image(id);
        }
    }
    /// UnmapGPUMemory marks actual overlapping GPU images remapped and
    /// CPU-modified. Native producers and physical captures are not retired.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L681-L698
    pub(super) fn modify_gpu_memory(
        &mut self,
        address_space: MaxwellAddressSpaceId,
        address: u64,
        bytes: u64,
    ) {
        if bytes == 0 {
            return;
        }
        let mut views = smallvec::SmallVec::<[ImageViewId; 16]>::new();
        for page in address >> PAGE_BITS..(address + bytes).div_ceil(1 << PAGE_BITS) {
            let Some(ids) = self.gpu_image_pages.get(&(address_space, page)) else {
                continue;
            };
            for &id in ids {
                let entry = &mut self.images[id];
                let cached = entry.cached.as_mut().expect("registered GPU image");
                if cached.info.address < address + bytes
                    && address < cached.info.address + cached.info.size
                {
                    if !cached.remapped {
                        views.extend_from_slice(&entry.views);
                    }
                    cached.remapped = true;
                    entry.synchronized = false;
                    entry.untrack_cpu(&self.device_memory);
                }
            }
        }
        // Reselect affected slots through FindImage, which excludes remapped entries.
        // Old image/map versions remain indexed for already committed physical uses.
        self.bindings.invalidate_image_views(&views);
        self.remove_framebuffers(&views);
    }
    /// WriteMemory keeps native images/views and marks intersecting DAddr
    /// contents CPU-modified. Already-modified images need no second update.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L586-L597
    pub(super) fn write_memory(&mut self, address: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = address + bytes;
        for page in address >> PAGE_BITS..=(end - 1) >> PAGE_BITS {
            let Some(ids) = self.image_pages.get(&page) else {
                continue;
            };
            for &id in ids {
                let entry = &mut self.images[id];
                if !entry.synchronized {
                    continue;
                }
                if entry
                    .maps
                    .iter()
                    .any(|&(base, size)| base < end && address < base + size)
                {
                    entry.synchronized = false;
                    entry.untrack_cpu(&self.device_memory);
                }
            }
        }
    }
}
impl Residency<'_> {
    /// FindDMAImage selects the newest compatible existing representation;
    /// DmaImageId then applies GPU-modified and first-download decisions.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L1835-L1892
    pub(super) fn dma_image_id(
        &mut self,
        space: &MaxwellGpuMemory,
        address: u64,
        bytes: u64,
        layout: nixe_gpu_maxwell::MaxwellMemoryCopyLayout,
        upload: bool,
    ) -> Result<Option<(ImageId, [u32; 2])>, VulkanError> {
        let nixe_gpu_maxwell::MaxwellMemoryCopyLayout::BlockLinear {
            surface_width,
            surface_height,
            ..
        } = layout
        else {
            return Ok(None);
        };
        let gpu = space.address(address).map_err(failure)?;
        let Ok(cpu) = space.device_address(gpu) else {
            return Ok(None);
        };
        let mut seen = smallvec::SmallVec::<[ImageId; 16]>::new();
        let mut found = None;
        let mut tick = 0;
        let cache = &mut *self.texture_cache;
        for page in cpu >> PAGE_BITS..(cpu + bytes).div_ceil(1 << PAGE_BITS) {
            let Some(ids) = cache.image_pages.get(&page) else {
                continue;
            };
            for &id in ids {
                if seen.contains(&id) {
                    continue;
                }
                seen.push(id);
                let entry = &mut cache.images[id];
                if !entry
                    .maps
                    .iter()
                    .any(|&(a, b)| a < cpu + bytes && cpu < a + b)
                {
                    continue;
                }
                let Some(cached) = entry.cached.as_mut() else {
                    continue;
                };
                if cached.address_space != space.id() || cached.remapped {
                    continue;
                }
                let Some(base) =
                    cached.dma_base(address, surface_width, surface_height, entry.sample_grid)
                else {
                    continue;
                };
                if found.is_none() || tick <= entry.modification_tick {
                    found = Some((id, base));
                    tick = entry.modification_tick;
                }
            }
        }
        if let Some((id, _)) = found {
            let entry = &mut cache.images[id];
            if !entry.gpu_modified {
                return Ok(None);
            }
            if !upload && !entry.dma_downloaded {
                entry.dma_downloaded = true;
                return Ok(None);
            }
            self.touch(Resource::Image(id));
            self.runtime.statistics.cache_hits += 1;
        }
        Ok(found)
    }
    /// InsertImage keeps hardware shape and current mapped segments separate.
    /// Empty map views are a GPU-only image, as Eden virtual_invalid_space.
    pub(crate) fn insert_guest_image(
        &mut self,
        info: &ImageInfo,
        space: &MaxwellGpuMemory,
    ) -> Result<ImageId, VulkanError> {
        let start = space.address(info.address).map_err(failure)?;
        let mut parts = Vec::new();
        let mut maps = RangeSet::default();
        let mut first_cpu = None;
        for (gpu, bytes) in space
            .get_submapped_range(start, info.size)
            .map_err(failure)?
        {
            // Like Eden InsertImage, resolve residency independently of the
            // image format and tiling, which come from ImageInfo's registers/TIC.
            // Compression PTE kinds do not select a different image layout here.
            // https://github.com/eden-emulator/mirror/blob/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/image_info.cpp
            let cpu = space.device_address(gpu).map_err(failure)?;
            first_cpu.get_or_insert(cpu);
            let physical = self.runtime.device_memory.read_mapping(cpu, bytes)?;
            parts.push((gpu.get() - info.address, physical));
            maps.insert(cpu, bytes);
        }
        let mapping = if parts.len() == 1 && parts[0].0 == 0 && parts[0].1.bytes() == info.size {
            parts.pop().unwrap().1
        } else {
            nixe_memory::PhysicalMapping::sparse(self.runtime.memory.identity(), info.size, &parts)?
        };
        let (geometry, planes) = crate::native::maxwell::image::planes(info, mapping)?;
        let description = crate::native::images::description(info.description);
        let cpu_address = if let Some(cpu) = first_cpu {
            cpu
        } else {
            let offset = self.texture_cache.virtual_invalid_space;
            let next = offset
                .checked_add(info.size.next_multiple_of(32))
                .filter(|&n| n < (1u64 << 40))
                .ok_or(VulkanError::InvalidResource(
                    "GPU-only image namespace exhausted",
                ))?;
            self.texture_cache.virtual_invalid_space = next;
            self.runtime.statistics.gpu_only_image_allocations += 1;
            (!(1u64 << 40))
                .checked_add(offset)
                .ok_or(VulkanError::InvalidResource(
                    "GPU-only image address overflows",
                ))?
        };
        let registration = Some(ImageRegistration {
            cpu_address,
            info: info.clone(),
            address_space: space.id(),
            maps: maps.iter().collect(),
        });
        if info.description.samples() != nixe_gpu::SampleCount::One {
            self.multisample_image_for(
                geometry,
                description,
                planes,
                crate::native::maxwell::image::sample_grid(info.description.samples())
                    .map_err(failure)?,
                registration,
            )
        } else if info.description.format().block_extent() == [1, 1] {
            self.image_for(geometry, description, planes, registration)
        } else {
            self.sampled_image_for(geometry, description, planes, registration)
        }
    }
    pub(crate) fn find_image(
        &mut self,
        info: &ImageInfo,
        space: &MaxwellGpuMemory,
    ) -> Result<Option<(ImageId, [u32; 2])>, VulkanError> {
        let address = space.address(info.address).map_err(failure)?;
        let cpu = match space.device_address(address) {
            Ok(cpu) => cpu,
            Err(nixe_gpu_maxwell::MaxwellGpuAccessError::UnmappedAddress { .. }) => {
                let ranges = space
                    .get_submapped_range(address, info.size)
                    .map_err(failure)?;
                let Some(&(first, _)) = ranges.first() else {
                    return Ok(None);
                };
                space.device_address(first).map_err(failure)?
            }
            Err(error) => return Err(failure(error)),
        };
        let end = cpu
            .checked_add(info.size)
            .ok_or(VulkanError::InvalidResource("image DAddr range overflows"))?;
        let mut picked = smallvec::SmallVec::<[ImageId; 16]>::new();
        let mut found = None;
        let mut tick = 0;
        let cache = &mut *self.texture_cache;
        'pages: for page in cpu >> PAGE_BITS..=(end - 1) >> PAGE_BITS {
            let Some(ids) = cache.image_pages.get(&page) else {
                continue;
            };
            for &id in ids {
                let entry = &mut cache.images[id];
                let Some(cached) = entry.cached.as_mut() else {
                    continue;
                };
                if cached.picked
                    || !entry
                        .maps
                        .iter()
                        .any(|&(base, bytes)| base < end && cpu < base + bytes)
                {
                    continue;
                }
                cached.picked = true;
                picked.push(id);
                if cached.address_space != space.id() || cached.remapped {
                    continue;
                }
                let Some(base) = cached.base(info) else {
                    continue;
                };
                // Preserve Maxwell's two distinct packed depth encodings.
                // Their neutral Vulkan format is shared, their byte layout is not.
                if matches!(
                    cached.info.guest_format,
                    nixe_gpu_maxwell::engines::threed::MaxwellThreeDGuestImageFormat::DepthStencil(
                        _
                    )
                ) && cached.info.guest_format != info.guest_format
                {
                    continue;
                }
                let requested = super::format::view_compatible(
                    entry.description.format,
                    crate::native::images::format(info.description.format()),
                );
                let exact = cached.info.description.format() == info.description.format();
                if !exact
                    && (!requested
                        || entry.decoded.is_some()
                        || !entry
                            .description
                            .flags
                            .contains(vk::ImageCreateFlags::MUTABLE_FORMAT))
                {
                    continue;
                }
                if found.is_none() || tick <= entry.modification_tick {
                    found = Some((id, base));
                    tick = entry.modification_tick;
                }
                if exact {
                    break 'pages;
                }
            }
        }
        for id in picked {
            self.texture_cache.images[id]
                .cached
                .as_mut()
                .unwrap()
                .picked = false;
        }
        if let Some((id, _)) = found {
            self.touch(Resource::Image(id));
            self.runtime.statistics.cache_hits += 1;
        }
        Ok(found)
    }
    pub(super) fn mark_image_modified(&mut self, id: ImageId) -> Result<(), VulkanError> {
        if self.texture_cache.get(id).decoded.is_some() {
            return Err(VulkanError::InvalidResource(
                "decoded compressed textures are read-only; writes require the encoded canonical representation",
            ));
        }
        self.texture_cache.image_modification_tick = self
            .texture_cache
            .image_modification_tick
            .checked_add(1)
            .expect("image modification tick exhausted");
        self.texture_cache.images[id].modification_tick =
            self.texture_cache.image_modification_tick;
        self.texture_cache.images[id].gpu_modified = true;
        Ok(())
    }
    pub fn attachment_view(
        &mut self,
        image: ImageId,
        plane: usize,
    ) -> Result<ImageViewId, VulkanError> {
        let entry = self.texture_cache.get(image);
        let p = *entry
            .planes
            .get(plane)
            .ok_or(VulkanError::InvalidResource("attachment plane is absent"))?;
        let aspects = entry
            .planes
            .iter()
            .filter(|other| {
                other.mip_level == p.mip_level
                    && other.array_layer == p.array_layer
                    && other.depth_slice == p.depth_slice
            })
            .fold(vk::ImageAspectFlags::empty(), |a, other| a | other.aspect);
        self.image_view(
            image,
            ImageViewDescription {
                size: p.extent,
                format: entry.description.format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects,
                base_mip: p.mip_level,
                mip_count: 1,
                base_layer: p.array_layer + p.depth_slice,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
    }
    pub(crate) fn image_view_description(
        &self,
        view: ImageViewId,
    ) -> Result<ImageViewDescription, VulkanError> {
        Ok(self.texture_cache.views[view].description)
    }
    pub(crate) fn image_view_plane(
        &self,
        view: ImageViewId,
        aspect: vk::ImageAspectFlags,
    ) -> Result<usize, VulkanError> {
        let d = self.image_view_description(view)?;
        self.texture_cache
            .get(self.texture_cache.views[view].image)
            .planes
            .iter()
            .position(|p| {
                p.mip_level == d.base_mip
                    && p.array_layer + p.depth_slice == d.base_layer
                    && p.aspect == aspect
            })
            .ok_or(VulkanError::InvalidResource("image view aspect is absent"))
    }
}
