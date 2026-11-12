//! Buffer and texture caches own DAddr coherence and native resource lifetimes.
mod color_blit;
pub use color_blit::color_blit_pipeline;
mod buffer;
pub use buffer::BufferRange;
mod buffer_download;
pub(crate) mod cache_owners;
mod command_pages;
mod multi_range;
mod quad_indices;
pub(crate) mod query;
mod virtual_range_cache;
use cache_owners::{BufferCache, CacheOwners, TextureCache};
pub(crate) mod commands;
mod conversion;
pub(crate) use conversion::stage as helper_stage;
mod budget;
mod cpu_readback;
mod decoded_texture;
mod dma;
pub(crate) mod execution;
mod format;
mod image;
pub(crate) mod presentation;
mod raster_transfer;
mod recording;
mod retirement;
mod sample_readback;
mod scaling;
mod texture_cache;
mod texture_download;
mod transfers;
pub use commands::{AttachmentLoad, AttachmentStore};
pub(crate) use conversion::{Nv12Inputs, nv12_pipeline};
mod sync;
use crate::{
    AllocatedBuffer, AllocatedImage, AllocationMode, MemoryLocation, VulkanDevice, VulkanError,
    WorkTicket,
    memory::{
        AllocationBudget, AllocationCharge, NativeImageDescription, allocate_buffer_charged,
        allocate_image_charged,
    },
};
use ash::vk;
use nixe_gpu::physical::Footprint;
use nixe_memory::Machine;
use nixe_memory::PhysicalMapping;
use nixe_memory::RangeSet;
use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;
pub use sync::AccessScope;

use crate::slot_vector::{SlotId, SlotVector};
pub type BufferId = SlotId;
pub type ImageId = SlotId;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Resource {
    Buffer(BufferId),
    Image(ImageId),
}
pub type ImageViewId = SlotId;
pub(crate) struct ImageView {
    pub(crate) image: ImageId,
    pub(crate) description: ImageViewDescription,
    pub(crate) native: Arc<crate::AllocatedImageView>,
    pub(crate) supports_minmax_filter: bool,
    storage_views: [Option<ImageViewId>; 8],
    downloaded: bool,
}
mod attachment;
pub use attachment::AttachmentId;
#[derive(Clone, Copy)]
struct LruLinks {
    previous: Option<SlotId>,
    next: Option<SlotId>,
}
struct Contents {
    lru: Option<LruLinks>,
    geometry: Vec<Arc<Footprint>>,
    last_use: Option<WorkTicket>,
    touched: u64,
}
struct BufferEntry {
    invalid: RangeSet,
    growth: u32,
    sparse_compatible: bool,
    write_tick: u64,
    binding_valid: bool,
    device_address: Option<u64>,
    picked: bool,
    contents: Contents,
    native: Arc<AllocatedBuffer>,
    usage: vk::BufferUsageFlags,
    used: sync::UsageTracker,
}
pub(crate) struct ImageEntry {
    synchronized: bool,
    aliases: Vec<image::ImageAlias>,
    cpu_tracked: bool,
    maps: smallvec::SmallVec<[(u64, u64); 1]>,
    cached: Option<texture_cache::CachedImage>,
    modification_tick: u64,
    gpu_modified: bool,
    forced_flushed: bool,
    dma_downloaded: bool,
    contents: Contents,
    native: Arc<AllocatedImage>,
    original: Arc<AllocatedImage>,
    scaled: Option<Arc<AllocatedImage>>,
    pub(crate) rescaled: bool,
    original_initialized: bool,
    scaled_initialized: bool,
    can_rescale: bool,
    checking_rescale: bool,
    rescale_checked: bool,
    scale_rating: u32,
    scale_tick: u64,
    pub(crate) description: NativeImageDescription,
    pub(crate) planes: Arc<[ImagePlane]>,
    sample_grid: [u32; 2],
    views: Vec<ImageViewId>,
    initialized: bool,
    decoded: Option<decoded_texture::DecodedImage>,
}
impl ImageEntry {
    fn untrack_cpu(&mut self, device: &nixe_memory::device_memory::DeviceMemory) {
        if !self.cpu_tracked {
            return;
        }
        for &(address, bytes) in &self.maps {
            device
                .update_pages_cached_count(address, bytes, -1)
                .expect("image cached-page untracking");
        }
        self.cpu_tracked = false;
    }
    fn current(&self) -> bool {
        self.synchronized
    }
    pub(crate) fn view_format(&self, requested: vk::Format) -> vk::Format {
        if self.decoded.is_some() {
            self.description.format
        } else {
            requested
        }
    }
    fn storage_plane(&self, mut plane: ImagePlane) -> ImagePlane {
        plane.extent.width *= self.sample_grid[0];
        plane.extent.height *= self.sample_grid[1];
        plane
    }
}
/// Native copy plane in the canonical byte stream. Guest block-linear decoding
/// and other guest encodings are legalized by the selected frontend, not this cache.
/// Row padding remains canonical CPU storage; GPU image writes own texel bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImagePlane {
    pub depth_slice: u32,
    pub offset: u64,
    pub row_pitch: u64,
    pub mip_level: u32,
    pub array_layer: u32,
    pub aspect: vk::ImageAspectFlags,
    pub extent: vk::Extent3D,
    pub block_extent: vk::Extent2D,
    pub bytes_per_block: u32,
}
impl ImagePlane {
    fn row_bytes(self) -> u64 {
        u64::from(self.extent.width.div_ceil(self.block_extent.width))
            * u64::from(self.bytes_per_block)
    }
    fn rows(self) -> u32 {
        self.extent.height.div_ceil(self.block_extent.height)
    }
    fn end(self) -> Option<u64> {
        self.offset
            .checked_add(self.row_pitch.checked_mul(u64::from(self.rows() - 1))?)
            .and_then(|v| v.checked_add(self.row_bytes()))
    }
    fn layers(self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(self.aspect)
            .mip_level(self.mip_level)
            .base_array_layer(self.array_layer)
            .layer_count(1)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageViewDescription {
    pub size: vk::Extent3D,
    pub format: vk::Format,
    pub view_type: vk::ImageViewType,
    pub aspects: vk::ImageAspectFlags,
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
    pub swizzle: [vk::ComponentSwizzle; 4],
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidencyStatistics {
    pub cpu_uploaded_bytes: u64,
    pub cpu_streamed_bytes: u64,
    pub gpu_copied_bytes: u64,
    pub dma_buffer_copies: u64,
    pub dma_image_uploads: u64,
    pub dma_image_downloads: u64,
    pub dma_cpu_fallbacks: u64,
    pub alias_copy_commands: u64,
    pub alias_copy_regions: u64,
    pub decoded_upload_commands: u64,
    /// Exact bytes transferred to host memory, excluding GPU-only realignment.
    pub native_readback_bytes: u64,
    pub canonical_readback_bytes: u64,
    pub cache_hits: u64,
    pub buffer_allocations: u64,
    pub image_allocations: u64,
    pub gpu_only_image_allocations: u64,
    pub image_layout_plans: u64,
    pub image_view_allocations: u64,
    pub buffer_merges: u64,
    pub evictions: u64,
}
/// Concrete single-owner storage, selected once at startup. No generic backend
/// handle registry, virtual resource dispatch or guest-ISA branch in its hot path.
mod bindings;
pub(crate) use bindings::{StorageBuffer, StorageBufferKey};
pub(crate) mod texture_bindings;
pub struct VulkanResidency {
    pub(crate) caches: Arc<CacheOwners>,
    pub(crate) runtime: ResidencyRuntime,
}

/// Resource access scoped to one graphics-owner operation. Both actual caches
/// stay locked while preparation selects resources and records their native use.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L236-L258
pub struct Residency<'a> {
    caches: &'a Arc<CacheOwners>,
    pub(crate) shader_cache: &'a nixe_gpu_maxwell::recompiler::ShaderCache,
    pub(crate) buffer_cache: std::sync::MutexGuard<'a, BufferCache>,
    pub(crate) texture_cache: std::sync::MutexGuard<'a, TextureCache>,
    pub(crate) runtime: &'a mut ResidencyRuntime,
}
pub(crate) struct ResidencyRuntime {
    quad_indices: quad_indices::QuadIndices,
    pub(crate) buffer_bindings: bindings::BufferBindings,
    pub(crate) device: VulkanDevice,
    device_memory: nixe_memory::device_memory::DeviceMemory,
    pub(crate) device_reader: nixe_memory::device_memory::DeviceMemoryReader,
    epoch: u64,
    budget_ceiling: u64,
    collection_target: u64,
    maintenance_at: std::time::Instant,
    memory: Machine,
    budget: AllocationBudget,
    statistics: ResidencyStatistics,
    barriers: sync::Barriers,
    multi_range: multi_range::MultiRangeCache,
    /// Entries needed by a cold reconstruction cannot be evicted by its allocations.
    pub(crate) constructing: Vec<Resource>,
    descriptors: crate::cache::DescriptorCache,
    descriptor_resources: Vec<crate::cache::DescriptorResource>,
    descriptor_queue: crate::cache::UpdateDescriptorQueue,
    compute_pass_descriptor_queue: crate::cache::UpdateDescriptorQueue,
    descriptor_offsets: Vec<u32>,
    uniform_updates: Vec<Option<crate::queue::Slice>>,
    streamed_uses: Vec<usize>,
}
impl VulkanResidency {
    pub(crate) fn shader_cache(&self) -> nixe_gpu_maxwell::recompiler::ShaderCache {
        self.caches.shader_cache.clone()
    }

    pub fn access(&mut self) -> Residency<'_> {
        Residency {
            caches: &self.caches,
            shader_cache: &self.caches.shader_cache,
            buffer_cache: self.caches.buffer_cache.lock().unwrap(),
            texture_cache: self.caches.texture_cache.lock().unwrap(),
            runtime: &mut self.runtime,
        }
    }
    pub fn new(
        device_memory: nixe_memory::device_memory::DeviceMemory,
        device: VulkanDevice,
        bytes: u64,
    ) -> Result<Self, VulkanError> {
        let memory = device_memory.machine().clone();
        if bytes == 0 {
            return Err(VulkanError::InvalidResource("empty residency budget"));
        }
        let descriptors = crate::cache::DescriptorCache::new(&device);
        let multi_range = multi_range::MultiRangeCache::new(&device);
        let mut cache = Self {
            caches: Arc::new(CacheOwners::new(
                device_memory.clone(),
                Arc::clone(&device.submission.progress),
                Arc::clone(&device.submission.timeline),
                crate::cache::SamplerCache::new(&device),
            )),
            runtime: ResidencyRuntime {
                quad_indices: Default::default(),
                buffer_bindings: Default::default(),
                device,
                descriptors,
                device_reader: device_memory.reader(),
                device_memory,
                epoch: 0,
                budget_ceiling: bytes,
                collection_target: bytes / 8 * 7,
                maintenance_at: std::time::Instant::now(),
                memory,
                budget: AllocationBudget::new(bytes),
                statistics: ResidencyStatistics::default(),
                barriers: sync::Barriers::default(),
                multi_range,
                constructing: Vec::new(),
                descriptor_resources: Vec::new(),
                descriptor_queue: crate::cache::UpdateDescriptorQueue::new(
                    crate::cache::UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                ),
                compute_pass_descriptor_queue: crate::cache::UpdateDescriptorQueue::new(
                    crate::cache::UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE,
                ),
                descriptor_offsets: Vec::new(),
                uniform_updates: Vec::new(),
                streamed_uses: Vec::new(),
            },
        };
        cache.access().maintain_budget(true)?;
        let interface: Arc<dyn nixe_memory::device_memory::DeviceMemoryInterface> =
            cache.caches.clone();
        cache
            .runtime
            .device_memory
            .bind_interface(Arc::downgrade(&interface))?;
        Ok(cache)
    }
    pub fn device(&self) -> &VulkanDevice {
        &self.runtime.device
    }
    pub fn device_memory(&self) -> &nixe_memory::device_memory::DeviceMemory {
        &self.runtime.device_memory
    }
    pub(crate) fn memory(&self) -> &Machine {
        &self.runtime.memory
    }
    pub(crate) fn descriptor_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.runtime.descriptors.cache_counters()
    }
    pub fn statistics(&self) -> ResidencyStatistics {
        let buffer = self.caches.buffer_cache.lock().unwrap();
        let texture = self.caches.texture_cache.lock().unwrap();
        let mut statistics = self.runtime.statistics;
        statistics.native_readback_bytes +=
            buffer.native_readback_bytes + texture.native_readback_bytes;
        statistics.canonical_readback_bytes += buffer.canonical_readback_bytes
            + texture.canonical_readback_bytes
            + self
                .caches
                .query_cache
                .lock()
                .unwrap()
                .canonical_readback_bytes;
        statistics
    }
}

impl Residency<'_> {
    /// Apply a completed RAM store without discarding GPU-owned bytes elsewhere
    /// on its pages. Partial uploads keep those pages resident on the GPU.
    /// https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L225-L296
    pub(crate) fn write_memory(&mut self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        self.write_cpu_ranges(std::iter::once((address, bytes)))
    }
    pub(crate) fn write_cpu_ranges(
        &mut self,
        ranges: impl Iterator<Item = (u64, u64)> + Clone,
    ) -> Result<(), VulkanError> {
        let ranges = ranges.filter(|&(_, bytes)| bytes != 0);
        // Claim the entire batch before deciding which pages still need GPU
        // ownership. Several CPU stores can together replace all dirty bytes.
        // https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L2640-L2650
        for (address, bytes) in ranges.clone() {
            self.buffer_cache.claim_cpu_write(address, bytes);
        }
        for (address, bytes) in ranges {
            if !self.upload_partial_write(address, bytes)? {
                self.buffer_cache.memory_tracker.cpu_write(address, bytes);
            }
            self.texture_cache.write_memory(address, bytes);
            self.caches
                .query_cache
                .lock()
                .unwrap()
                .invalidate_region(address, bytes);
            self.shader_cache.invalidate_region(address, bytes);
        }
        Ok(())
    }
    pub(crate) fn write_regions(
        &mut self,
        regions: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), VulkanError> {
        let device = self.runtime.device_memory.clone();
        for range in regions {
            self.write_memory(range.address, range.size)?;
            // GPU virtual stores name one device mapping. Other mappings of
            // the same RAM must receive the same content invalidation.
            let mut result = Ok(());
            device.apply_device_aliases(range.address, range.size, |alias, bytes| {
                if result.is_ok() {
                    result = self.write_memory(alias, bytes);
                }
            });
            result?;
        }
        Ok(())
    }
    pub fn device_memory(&self) -> &nixe_memory::device_memory::DeviceMemory {
        &self.runtime.device_memory
    }
    pub fn device(&self) -> &VulkanDevice {
        &self.runtime.device
    }
    pub(crate) fn device_mut(&mut self) -> &mut VulkanDevice {
        &mut self.runtime.device
    }
    pub fn statistics(&self) -> ResidencyStatistics {
        let mut statistics = self.runtime.statistics;
        statistics.native_readback_bytes +=
            self.buffer_cache.native_readback_bytes + self.texture_cache.native_readback_bytes;
        statistics.canonical_readback_bytes += self.buffer_cache.canonical_readback_bytes
            + self.texture_cache.canonical_readback_bytes
            + self
                .caches
                .query_cache
                .lock()
                .unwrap()
                .canonical_readback_bytes;
        statistics
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.runtime.budget.used()
    }
    /// Cold accounting includes allocator slabs and the separately bounded
    /// staging arenas. The resource budget charges native memory requirements,
    /// not logical image texel sizes; allocator capacity includes fragmentation.
    pub fn allocator_capacity_bytes(&self) -> u64 {
        self.runtime
            .device
            .owner
            .allocator
            .lock()
            .unwrap()
            .as_ref()
            .expect("live allocator")
            .capacity()
    }
    fn contents(&self, resource: Resource) -> &Contents {
        match resource {
            Resource::Buffer(id) => &self.buffer_cache.get(id).contents,
            Resource::Image(id) => &self.texture_cache.get(id).contents,
        }
    }
    fn contents_mut(&mut self, resource: Resource) -> &mut Contents {
        match resource {
            Resource::Buffer(id) => &mut self.buffer_cache.buffers[id].contents,
            Resource::Image(id) => &mut self.texture_cache.images[id].contents,
        }
    }
    // Allocation accounting cannot run cache deletion inside a draw's resource
    // preparation. Eden collects aged cache entries from TickFrame instead.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    fn charge(&self, bytes: u64) -> Result<Option<AllocationCharge>, VulkanError> {
        self.runtime
            .budget
            .charge(bytes)
            .map(Some)
            .ok_or(VulkanError::InvalidResource(
                "native allocation accounting overflow",
            ))
    }
    fn invalidate_bindings(&mut self) {
        self.texture_cache.binding_revision = self
            .texture_cache
            .binding_revision
            .checked_add(1)
            .expect("native binding revision exhausted");
    }
    fn retire_entry(&mut self, resource: Resource) -> Result<(), VulkanError> {
        if let Resource::Image(id) = resource
            && self.texture_cache.pending_downloads.iter().any(|pending| matches!(pending, texture_download::PendingDownload::Image(source) if *source == id))
        {
            self.commit_texture_downloads()?;
        }
        match resource {
            Resource::Buffer(id) => {
                self.invalidate_bindings();
                self.unlink_lru(resource);
                self.drop_multi_range_owner(id)?;
                self.runtime.buffer_bindings.invalidate_selections(id);
                if let Some(address) = self.buffer_cache.get(id).device_address
                    && self.buffer_cache.buffer_index.at(address) == Some(id)
                {
                    let bytes = self.buffer_cache.get(id).native.size();
                    self.buffer_cache.memory_tracker.cpu_write(address, bytes);
                    self.buffer_cache.clear_download(address, bytes);
                }
                let entry = self.buffer_cache.buffers.erase(id);
                if let Some(address) = entry.device_address {
                    self.buffer_cache
                        .buffer_index
                        .remove(id, address, entry.native.size());
                }
                if let Some(last_use) = entry.contents.last_use {
                    self.runtime.device.retire_buffer(entry.native, last_use)?;
                }
            }
            Resource::Image(id) => {
                self.texture_cache.delete_image(id);
            }
        }
        Ok(())
    }
    pub(crate) fn check_machine(&self, machine: nixe_memory::MachineId) -> Result<(), VulkanError> {
        if machine != self.runtime.memory.identity() {
            return Err(VulkanError::InvalidResource(
                "memory notification belongs to another machine",
            ));
        }
        Ok(())
    }
    pub(crate) fn retire_physical_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
        extents: &[nixe_memory::Extent],
    ) -> Result<(), VulkanError> {
        self.check_machine(machine)?;
        self.realize_clears_overlapping(extents.iter().copied())?;
        // Allocation retirement is cold. Accepted downloads retain their own
        // input pins; cached entries keep weak mappings and can now be deleted.
        let overlaps = |contents: &Contents| {
            contents.geometry.iter().any(|geometry| {
                geometry.mapping().spans().iter().any(|span| {
                    span.allocation == allocation
                        && extents.iter().any(|extent| {
                            span.start.get() < extent.start.get() + extent.bytes
                                && extent.start.get() < span.start.get() + span.bytes
                        })
                })
            })
        };
        let mut retired = Vec::new();
        for (id, entry) in self.buffer_cache.buffers.iter() {
            if overlaps(&entry.contents) {
                retired.push(Resource::Buffer(id));
            }
        }
        for (id, entry) in self.texture_cache.images.iter() {
            if overlaps(&entry.contents) {
                retired.push(Resource::Image(id));
            }
        }
        for resource in retired {
            self.retire_entry(resource)?;
        }
        Ok(())
    }
    fn unpin(&mut self, resource: Resource) {
        let index = self
            .runtime
            .constructing
            .iter()
            .rposition(|item| *item == resource)
            .expect("active construction pin");
        self.runtime.constructing.remove(index);
    }
    fn temporary_buffer(
        &mut self,
        size: u64,
        references: &[Resource],
    ) -> Result<AllocatedBuffer, VulkanError> {
        self.runtime.constructing.extend_from_slice(references);
        let result = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            0,
        );
        for &resource in references.iter().rev() {
            self.unpin(resource);
        }
        result
    }
    fn allocate_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        minimum_alignment: u64,
    ) -> Result<AllocatedBuffer, VulkanError> {
        let owner = Arc::clone(&self.runtime.device.owner);
        allocate_buffer_charged(
            &owner,
            size,
            usage,
            MemoryLocation::GpuOnly,
            AllocationMode::Managed,
            "resident buffer",
            minimum_alignment,
            |bytes| self.charge(bytes),
        )
    }
    fn finish_buffer_write(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let work = self.runtime.device.current_work();
        let entry = &mut self.buffer_cache.buffers[id];

        entry.write_tick = work.tick();
        entry.invalid.subtract(offset, bytes);
        entry.contents.last_use = Some(work);
        self.buffer_cache.mark_written(id, offset, bytes)?;
        Ok(())
    }
    pub fn write_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        let size = bytes.len() as u64;
        self.check_buffer_range(id, offset, size)?;
        if self.buffer_cache.get(id).device_address.is_some() {
            self.ensure_buffer_range(id, offset, size)?;
        }
        let slice = self
            .runtime
            .device
            .submission
            .reserve_upload_aligned(bytes, 4)?;
        self.record_buffer_upload(
            id,
            slice,
            smallvec::smallvec![vk::BufferCopy::default().dst_offset(offset).size(size)],
        )?;
        self.finish_buffer_write(id, offset, size)?;
        Ok(self.runtime.device.current_work())
    }

    pub(crate) fn report_samples_passed(
        &mut self,
        target: &[nixe_memory::AllocationRange],
        short: bool,
        timestamp: u64,
    ) -> Result<(), VulkanError> {
        let mapping = PhysicalMapping::new(target)?;
        let range = self.buffer_for_deferred_import(
            mapping,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let size = if short { 4 } else { 16 };
        self.ensure_buffer_range(range.buffer, range.offset, size)?;
        let output = self.use_buffer(range.buffer, range.offset, size)?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let source = self.runtime.device.sample_report_buffer()?;
        unsafe {
            self.runtime.device.dispatch.cmd_copy_buffer(
                source.handle,
                output,
                &[vk::BufferCopy::default()
                    .dst_offset(range.offset)
                    .size(if short { 4 } else { 8 })],
                true,
            )
        };
        self.finish_buffer_write(range.buffer, range.offset, size)?;
        if !short {
            self.write_buffer(range.buffer, range.offset + 8, &timestamp.to_le_bytes())?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<WorkTicket, VulkanError> {
        self.flush_with_sync(&[], &[])
    }
    pub(crate) fn flush_with_sync(
        &mut self,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<WorkTicket, VulkanError> {
        self.runtime.device.realize_deferred_clear()?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime.device.submission.flush(
                self.runtime.device.queue,
                &mut self.runtime.device.dispatch,
                waits,
                signals,
            )
        }
    }

    pub fn read_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, VulkanError> {
        self.ensure_buffer_range(id, offset, size)?;
        self.read_buffer_native(id, offset, size)
    }
}
impl Drop for VulkanResidency {
    fn drop(&mut self) {
        // Drain accepted native uses before entries destroy their objects. The
        // queue's fatal/device-lost teardown policy remains the single authority.
        if let Err(error) = self.runtime.device.shutdown()
            && !self.runtime.device.submission.slots.is_empty()
        {
            log::error!("Cannot safely destroy Vulkan residency before queue completion: {error}");
            std::process::abort();
        }
    }
}
fn memory_error(error: nixe_memory::MemoryError) -> VulkanError {
    VulkanError::from(error)
}
