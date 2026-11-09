//! Shared buffer/texture contents used by CPU coherence and fence publication.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache_base.h
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache_base.h
use super::*;
pub(crate) struct CacheOwners {
    pub(crate) shader_cache: nixe_gpu_maxwell::recompiler::ShaderCache,
    device_memory: nixe_memory::device_memory::DeviceMemory,
    pub(crate) query_cache: std::sync::Mutex<query::QueryCache>,
    pub(crate) buffer_cache: std::sync::Mutex<BufferCache>,
    pub(crate) texture_cache: std::sync::Mutex<TextureCache>,
    gpu_invalidations: Arc<virtual_range_cache::Invalidations>,
}
impl CacheOwners {
    pub(crate) fn new(
        device: nixe_memory::device_memory::DeviceMemory,
        progress: Arc<crate::queue::progress::SubmissionProgress>,
        samplers: crate::cache::SamplerCache,
    ) -> Self {
        let buffer_cache = BufferCache::new(device.clone());
        let gpu_invalidations = Arc::clone(&buffer_cache.virtual_ranges.invalidations);
        Self {
            shader_cache: nixe_gpu_maxwell::recompiler::ShaderCache::new(device.clone()),
            device_memory: device.clone(),
            query_cache: Default::default(),
            buffer_cache: std::sync::Mutex::new(buffer_cache),
            texture_cache: std::sync::Mutex::new(TextureCache::new(device, progress, samplers)),
            gpu_invalidations,
        }
    }
    /// UnmapMemory deletes overlapping textures, whereas CPU writes only mark
    /// their contents modified. Committed downloads own their physical inputs.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
    fn unmap_memory(&self, address: u64, bytes: u64) {
        self.texture_cache
            .lock()
            .unwrap()
            .unmap_memory(address, bytes);
        self.buffer_cache
            .lock()
            .unwrap()
            .write_memory(address, bytes);
        self.query_cache
            .lock()
            .unwrap()
            .invalidate_region(address, bytes);
        self.shader_cache.on_cache_invalidation(address, bytes);
    }
}
impl nixe_gpu_maxwell::MaxwellGpuMemoryInterface for CacheOwners {
    fn unmap_memory(&self, address: u64, bytes: u64) {
        self.unmap_memory(address, bytes);
    }
    fn modify_gpu_memory(
        &self,
        address_space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
        address: u64,
        bytes: u64,
    ) {
        self.texture_cache
            .lock()
            .unwrap()
            .modify_gpu_memory(address_space, address, bytes);
        // UnmapGPUMemory only takes VirtualRangeCache's deferred mutex; the
        // caller does not lock BufferCache or mutate its GPU-owned entries.
        self.gpu_invalidations.unmap(address_space, address, bytes);
    }
}
impl std::fmt::Debug for CacheOwners {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanCacheOwners").finish_non_exhaustive()
    }
}
impl nixe_memory::device_memory::DeviceMemoryInterface for CacheOwners {
    fn unmap_memory(&self, address: u64, bytes: u64) {
        self.unmap_memory(address, bytes);
    }
    fn must_flush_region(&self, address: u64, bytes: u64) -> bool {
        if self
            .buffer_cache
            .lock()
            .unwrap()
            .memory_tracker
            .gpu_modified(address, bytes)
        {
            return true;
        }
        self.texture_cache
            .lock()
            .unwrap()
            .region_gpu_modified(address, bytes)
    }
    fn on_cpu_read(&self, address: u64, bytes: u64) -> nixe_memory::NativeReadArea {
        let area = self
            .texture_cache
            .lock()
            .unwrap()
            .flush_area(address, bytes);
        nixe_memory::NativeReadArea {
            start: area.start,
            end: area.end,
            preemptive: area.preemptive,
        }
    }

    fn on_cpu_write(&self, address: u64, bytes: u64) -> bool {
        {
            let mut cache = self.buffer_cache.lock().unwrap();
            let registered = (address >> 16..(address + bytes).div_ceil(1 << 16))
                .any(|page| cache.buffer_index.at(page << 16).is_some());
            if registered {
                if cache.memory_tracker.gpu_modified(address, bytes) {
                    return true;
                }
                cache.write_memory(address, bytes);
            }
        }
        self.texture_cache
            .lock()
            .unwrap()
            .write_memory(address, bytes);
        self.query_cache
            .lock()
            .unwrap()
            .invalidate_region(address, bytes);
        self.shader_cache.invalidate_region(address, bytes);
        false
    }

    fn invalidate_region(&self, address: u64, bytes: u64) {
        self.query_cache
            .lock()
            .unwrap()
            .invalidate_region(address, bytes);
        self.shader_cache.invalidate_region(address, bytes);
        self.texture_cache
            .lock()
            .unwrap()
            .write_memory(address, bytes);
        self.buffer_cache
            .lock()
            .unwrap()
            .write_memory(address, bytes);
    }
}
pub(crate) struct BufferCache {
    pub(super) virtual_ranges: virtual_range_cache::VirtualRangeCache,
    pub(super) lru: budget::ResourceLru,
    pub(super) device_memory: nixe_memory::device_memory::DeviceMemory,
    pub(super) native_readback_bytes: u64,
    pub(super) canonical_readback_bytes: u64,
    pub(super) buffers: SlotVector<BufferEntry>,
    pub(super) buffer_index: buffer::BufferIndex,
    pub(super) dma_buffer: Vec<u8>,
    pub(super) memory_tracker: nixe_gpu::memory_tracker::MemoryTracker,
    pub(super) gpu_modified_ranges: RangeSet,
    pub(super) device_mapping_generation: u64,
    pub(super) uncommitted_gpu_modified_ranges: RangeSet,
    pub(super) committed_gpu_modified_ranges: Vec<RangeSet>,
    pub(super) async_downloads: nixe_memory::OverlapRangeSet,
    pub(super) buffer_downloads: std::collections::VecDeque<Option<buffer_download::Download>>,
}
impl BufferCache {
    pub(super) fn write_memory(&mut self, address: u64, bytes: u64) {
        if self.gpu_modified_ranges.intersects(address, bytes) {
            self.clear_download(address, bytes);
            self.gpu_modified_ranges.subtract(address, bytes);
        }
        self.memory_tracker.cpu_write(address, bytes);
    }
    pub(super) fn new(device: nixe_memory::device_memory::DeviceMemory) -> Self {
        Self {
            virtual_ranges: Default::default(),
            lru: Default::default(),
            device_memory: device.clone(),
            native_readback_bytes: 0,
            canonical_readback_bytes: 0,
            buffers: Default::default(),
            buffer_index: buffer::BufferIndex::new(device.address_bits()),
            dma_buffer: Vec::new(),
            memory_tracker: nixe_gpu::memory_tracker::MemoryTracker::new(device),
            gpu_modified_ranges: RangeSet::default(),
            device_mapping_generation: 0,
            uncommitted_gpu_modified_ranges: RangeSet::default(),
            committed_gpu_modified_ranges: Vec::new(),
            async_downloads: Default::default(),
            buffer_downloads: Default::default(),
        }
    }
}
pub(crate) struct TextureCache {
    // Native samplers live with the texture cache, through queue shutdown,
    // including replaced TSC selections and exceptional native teardown.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache_base.h
    pub(crate) samplers: crate::cache::SamplerCache,
    pub(super) retired: retirement::TextureRetirement,
    pub(crate) bindings: texture_bindings::TextureBindings,
    pub(super) framebuffers: HashMap<texture_bindings::FramebufferKey, SlotId>,
    pub(crate) slot_framebuffers: SlotVector<Arc<execution::PreparedTargets>>,
    pub(super) binding_revision: u64,
    pub(super) lru: budget::ResourceLru,
    pub(super) native_readback_bytes: u64,
    pub(super) canonical_readback_bytes: u64,
    pub(super) device_memory: nixe_memory::device_memory::DeviceMemory,
    pub(super) images: SlotVector<ImageEntry>,
    pub(crate) views: SlotVector<ImageView>,
    pub(super) attachments: SlotVector<attachment::AttachmentEntry>,
    pub(super) image_pages: HashMap<u64, Vec<ImageId>>,
    pub(super) gpu_image_pages:
        HashMap<(nixe_gpu_maxwell::MaxwellAddressSpaceId, u64), Vec<ImageId>>,
    pub(super) image_modification_tick: u64,
    pub(super) virtual_invalid_space: u64,
    pub(super) pending_downloads: Vec<texture_download::PendingDownload>,
    pub(super) downloads: std::collections::VecDeque<texture_download::Download>,
    pub(super) decoder: nixe_gpu::texture_decode::BlockDecoder,
    pub(super) sample_conversion: Option<sample_readback::SampleConversion>,
}

impl BufferCache {
    #[inline]
    pub(super) fn get_mut(&mut self, id: BufferId) -> &mut BufferEntry {
        &mut self.buffers[id]
    }
    #[inline]
    pub(super) fn get(&self, id: BufferId) -> &BufferEntry {
        &self.buffers[id]
    }
}

impl TextureCache {
    pub(super) fn new(
        device_memory: nixe_memory::device_memory::DeviceMemory,
        progress: Arc<crate::queue::progress::SubmissionProgress>,
        samplers: crate::cache::SamplerCache,
    ) -> Self {
        Self {
            samplers,
            bindings: Default::default(),
            framebuffers: Default::default(),
            slot_framebuffers: Default::default(),
            binding_revision: 1,
            retired: retirement::TextureRetirement::new(progress),
            lru: Default::default(),
            native_readback_bytes: 0,
            canonical_readback_bytes: 0,
            device_memory,
            images: Default::default(),
            views: Default::default(),
            attachments: Default::default(),
            image_pages: Default::default(),
            gpu_image_pages: Default::default(),
            image_modification_tick: Default::default(),
            virtual_invalid_space: 0,
            pending_downloads: Default::default(),
            downloads: Default::default(),
            decoder: Default::default(),
            sample_conversion: Default::default(),
        }
    }
    #[inline]
    pub(crate) fn get(&self, id: ImageId) -> &ImageEntry {
        &self.images[id]
    }
}

impl BufferCache {
    pub(crate) fn binding_current(&self, id: BufferId) -> bool {
        self.buffers
            .get(id)
            .is_some_and(|entry| entry.binding_valid)
    }
}

impl TextureCache {
    pub(crate) fn binding_current(&self, id: ImageId) -> bool {
        self.images.contains(id)
    }
}

impl Drop for TextureCache {
    fn drop(&mut self) {
        for (_, entry) in self.images.iter_mut() {
            entry.untrack_cpu(&self.device_memory);
        }
    }
}

impl CacheOwners {
    /// InnerInvalidation visits each actual cache under its own lock, before
    /// draw preparation takes both locks.
    pub(crate) fn invalidate_regions(&self, regions: &[nixe_gpu::native::GuestBufferRange]) {
        {
            let mut textures = self.texture_cache.lock().unwrap();
            for range in regions {
                textures.write_memory(range.address, range.size);
            }
        }
        {
            let mut buffers = self.buffer_cache.lock().unwrap();
            for range in regions {
                buffers.write_memory(range.address, range.size);
            }
        }
        for range in regions {
            self.query_cache
                .lock()
                .unwrap()
                .invalidate_region(range.address, range.size);
            self.shader_cache.invalidate_device_write(
                &self.device_memory,
                range.address,
                range.size,
            );
        }
    }

    /// FenceManager::PopAsyncFlushes runs after the native fence, under the
    /// actual cache locks, and consumes exactly one committed prefix per cache.
    pub(crate) fn pop_async_flushes(&self) -> Result<(), VulkanError> {
        let mut buffers = self.buffer_cache.lock().unwrap();
        let mut textures = self.texture_cache.lock().unwrap();
        textures.pop_async_flush(&mut buffers)?;
        buffers.pop_async_flush()
    }
}
