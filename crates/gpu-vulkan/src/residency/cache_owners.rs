//! Shared buffer/texture contents used by CPU coherence and fence publication.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache_base.h
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache_base.h
use super::*;
pub(crate) struct CacheOwners {
    uniform_sync: Arc<std::sync::atomic::AtomicU64>,
    pub(super) command_pages: Arc<command_pages::CommandPages>,
    pub(crate) shader_cache: nixe_gpu_maxwell::recompiler::ShaderCache,
    device_memory: nixe_memory::device_memory::DeviceMemory,
    pub(super) progress: Arc<crate::queue::progress::SubmissionProgress>,
    pub(super) timeline: Arc<crate::queue::progress::Timeline>,
    pub(crate) query_cache: std::sync::Mutex<query::QueryCache>,
    pub(crate) buffer_cache: std::sync::Mutex<BufferCache>,
    pending_buffer_reads: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) texture_cache: std::sync::Mutex<TextureCache>,
    gpu_invalidations: Arc<virtual_range_cache::Invalidations>,
}
impl CacheOwners {
    pub(crate) fn guest_memory_sync(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(&self.uniform_sync)
    }
    /// A guest synchronization can expose CPU writes to unprotected streamed RAM.
    pub(crate) fn sync_guest_memory(&self) {
        self.uniform_sync
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
    pub(crate) fn pending_read_count(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.pending_buffer_reads)
    }
    pub(super) fn finish_buffer_reads(&self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        if self
            .pending_buffer_reads
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return Ok(());
        }
        let end = self
            .buffer_cache
            .lock()
            .unwrap()
            .pending_read(address, bytes);
        if let Some(end) = end {
            self.timeline.wait(&self.progress, end.work().tick())?;
            self.buffer_cache
                .lock()
                .unwrap()
                .publish_reads_through(end)?;
        }
        Ok(())
    }
    pub(crate) fn new(
        device: nixe_memory::device_memory::DeviceMemory,
        progress: Arc<crate::queue::progress::SubmissionProgress>,
        timeline: Arc<crate::queue::progress::Timeline>,
        samplers: crate::cache::SamplerCache,
    ) -> Self {
        let command_pages = Arc::new(command_pages::CommandPages::default());
        let uniform_sync = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let buffer_cache = BufferCache::new(
            device.clone(),
            Arc::clone(&command_pages),
            Arc::clone(&uniform_sync),
        );
        let pending_buffer_reads = Arc::clone(&buffer_cache.pending_read_count);
        let gpu_invalidations = Arc::clone(&buffer_cache.virtual_ranges.invalidations);
        Self {
            command_pages,
            progress: Arc::clone(&progress),
            timeline,
            shader_cache: nixe_gpu_maxwell::recompiler::ShaderCache::new(device.clone()),
            device_memory: device.clone(),
            query_cache: std::sync::Mutex::new(query::QueryCache::new(Arc::clone(&uniform_sync))),
            buffer_cache: std::sync::Mutex::new(buffer_cache),
            pending_buffer_reads,
            texture_cache: std::sync::Mutex::new(TextureCache::new(
                device,
                progress,
                samplers,
                Arc::clone(&uniform_sync),
            )),
            uniform_sync,
            gpu_invalidations,
        }
    }
    /// UnmapMemory deletes overlapping textures, whereas CPU writes only mark
    /// their contents modified. Committed downloads own their physical inputs.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
    fn unmap_memory(&self, address: u64, bytes: u64) {
        self.device_memory.invalidate_command_pages(address, bytes);
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
    fn finish_read(&self, address: u64, bytes: u64) -> Result<(), Arc<nixe_memory::MemoryError>> {
        self.finish_buffer_reads(address, bytes).map_err(|error| {
            Arc::new(nixe_memory::MemoryError::Execution(
                error.to_string().into(),
            ))
        })
    }
    fn command_tick_done(&self, tick: u64) -> Result<bool, Arc<nixe_memory::MemoryError>> {
        self.timeline
            .is_done(&self.progress, tick)
            .map_err(|e| Arc::new(nixe_memory::MemoryError::Execution(e.to_string().into())))
    }
    fn read_command(
        &self,
        address: u64,
        output: &mut [u8],
        past_sync: bool,
    ) -> Result<nixe_gpu_maxwell::CommandRead, Arc<nixe_memory::MemoryError>> {
        self.command_pages
            .read(&self.device_memory, address, output, past_sync)
            .map_err(|e| Arc::new(nixe_memory::MemoryError::Execution(e.to_string().into())))
    }
    fn must_flush_region(&self, address: u64, bytes: u64) -> bool {
        // Lock each owner once for the range and all its physical aliases.
        // Never-written pages were rejected by the command reader already.
        let mut aliases = smallvec::SmallVec::<[(u64, u64); 4]>::new();
        aliases.push((address, bytes));
        self.device_memory
            .apply_device_aliases(address, bytes, |at, size| aliases.push((at, size)));
        {
            let cache = self.buffer_cache.lock().unwrap();
            if aliases
                .iter()
                .any(|&(at, size)| cache.memory_tracker.gpu_modified(at, size))
            {
                return true;
            }
        }
        {
            let cache = self.texture_cache.lock().unwrap();
            if aliases
                .iter()
                .any(|&(at, size)| cache.region_gpu_modified(at, size))
            {
                return true;
            }
        }
        let cache = self.query_cache.lock().unwrap();
        aliases
            .iter()
            .any(|&(at, size)| cache.must_flush_region(at, size))
    }

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
            .modify_gpu_memory(address_space.get(), address, bytes);
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
    fn try_on_cpu_read(&self, address: u64, bytes: u64) -> Option<nixe_memory::NativeReadArea> {
        // The graphics owner can hold this mutex while waiting for native
        // quiescence. An admitted vCPU must never block on that mutex.
        let mut textures = self.texture_cache.try_lock().ok()?;
        let area = textures.flush_area::<false>(address, bytes)?;
        // Check after locking textures: the owner takes that lock while adding
        // a read. Sampling the count beforehand could miss a just-prepared copy.
        // Buffer acquisition is nonblocking even though this is the opposite
        // order from the owner; failure releases textures and suspends the read.
        if self
            .pending_buffer_reads
            .load(std::sync::atomic::Ordering::Acquire)
            != 0
        {
            let buffers = self.buffer_cache.try_lock().ok()?;
            if buffers
                .pending_read(area.start, area.end - area.start)
                .is_some()
            {
                return None;
            }
        }
        Some(area)
    }
    fn on_cpu_read(&self, address: u64, bytes: u64) -> nixe_memory::NativeReadArea {
        let mut buffers = self.buffer_cache.lock().unwrap();
        let mut area = self
            .texture_cache
            .lock()
            .unwrap()
            .flush_area::<true>(address, bytes)
            .expect("reactive reads always resolve an area");
        area.preemptive &= buffers
            .pending_read(area.start, area.end - area.start)
            .is_none();
        if !area.preemptive {
            buffers.note_read_back(address, bytes);
        }
        area
    }

    fn on_cpu_write(&self, address: u64, bytes: u64) -> bool {
        {
            let mut cache = self.buffer_cache.lock().unwrap();
            let registered = (address >> 16..(address + bytes).div_ceil(1 << 16))
                .any(|page| cache.buffer_index.at(page << 16).is_some());
            if registered {
                if cache.memory_tracker.gpu_modified(address, bytes) {
                    cache.note_uniform_write(address, bytes);
                    // Invalidate now and again when the deferred write is applied;
                    // a producer can create a new command copy in between.
                    self.device_memory.write_command_pages(address, bytes);
                    return true;
                }
                cache.write_memory(address, bytes);
            } else {
                self.device_memory.write_command_pages(address, bytes);
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
    pub(super) uniform_streams: uniform_stream::UniformStreams,
    pub(super) command_pages: Arc<command_pages::CommandPages>,
    pub(super) collect_command_copies: bool,
    pub(super) command_copy_requests: Vec<(BufferId, u64, u64)>,
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
    pub(super) pending_download_ranges: RangeSet,
    pub(super) download_serial: u64,
    pub(super) pending_read_count: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) pending_reports: Vec<Arc<query::Query>>,
    pub(super) read_back_pages: Vec<buffer_download::ReadBackPage>,
    pub(super) read_back_reads: u64,
    pub(super) read_download_ranges: RangeSet,
    pub(super) read_downloads:
        std::collections::VecDeque<(buffer_download::DownloadEnd, buffer_download::ReadDownload)>,
    // DMA staging belongs to the texture cache's publication order, including
    // interleaved images. A buffer prefix must not consume these entries.
    pub(super) dma_downloads:
        std::collections::VecDeque<(buffer_download::DownloadEnd, buffer_download::ReadDownload)>,
    pub(super) buffer_downloads: std::collections::VecDeque<Option<buffer_download::Download>>,
}
impl BufferCache {
    pub(super) fn claim_cpu_write(&mut self, address: u64, bytes: u64) {
        self.note_uniform_write(address, bytes);
        // Include ordered host writes, even if they were already notified before
        // being deferred. Versions follow physical pages, including all aliases.
        // https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L225
        self.device_memory.write_command_pages(address, bytes);
        if !self.read_downloads.is_empty()
            || !self.buffer_downloads.is_empty()
            || !self.dma_downloads.is_empty()
            || self.gpu_modified_ranges.intersects(address, bytes)
        {
            self.clear_download(address, bytes);
            self.gpu_modified_ranges.subtract(address, bytes);
        }
    }
    pub(super) fn write_memory(&mut self, address: u64, bytes: u64) {
        self.claim_cpu_write(address, bytes);
        self.memory_tracker.cpu_write(address, bytes);
    }
    pub(super) fn new(
        device: nixe_memory::device_memory::DeviceMemory,
        command_pages: Arc<command_pages::CommandPages>,
        uniform_sync: Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        Self {
            uniform_streams: uniform_stream::UniformStreams::new(uniform_sync),
            command_pages,
            collect_command_copies: false,
            command_copy_requests: Vec::new(),
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
            pending_download_ranges: RangeSet::default(),
            download_serial: 0,
            pending_read_count: Default::default(),
            pending_reports: Vec::new(),
            read_back_pages: Vec::new(),
            read_back_reads: 0,
            read_download_ranges: Default::default(),
            read_downloads: Default::default(),
            dma_downloads: Default::default(),
            buffer_downloads: Default::default(),
        }
    }
}
impl Drop for BufferCache {
    fn drop(&mut self) {
        self.pending_read_count
            .store(0, std::sync::atomic::Ordering::Release);
    }
}
pub(crate) struct TextureCache {
    pub(super) uniform_sync: Arc<std::sync::atomic::AtomicU64>,
    pub(super) frame_tick: u64,
    // Native samplers live with the texture cache, through queue shutdown,
    // including replaced TSC selections and exceptional native teardown.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache_base.h
    pub(crate) samplers: crate::cache::SamplerCache,
    pub(super) retired: retirement::TextureRetirement,
    pub(crate) bindings: texture_bindings::TextureBindings,
    pub(super) framebuffers: HashMap<texture_bindings::FramebufferKey, SlotId>,
    pub(crate) slot_framebuffers: SlotVector<Arc<execution::PreparedTargets>>,
    pub(crate) binding_revision: u64,
    pub(super) lru: budget::ResourceLru,
    pub(super) native_readback_bytes: u64,
    pub(super) canonical_readback_bytes: u64,
    pub(super) device_memory: nixe_memory::device_memory::DeviceMemory,
    pub(super) images: SlotVector<ImageEntry>,
    pub(crate) views: SlotVector<ImageView>,
    pub(super) attachments: SlotVector<attachment::AttachmentEntry>,
    pub(super) image_pages: HashMap<u64, Vec<ImageId>>,
    pub(super) gpu_image_pages: HashMap<(u64, u64), Vec<ImageId>>,
    pub(super) image_modification_tick: u64,
    pub(super) virtual_invalid_space: u64,
    pub(super) pending_downloads: Vec<texture_download::PendingDownload>,
    pub(super) downloads: std::collections::VecDeque<texture_download::Download>,
    pub(super) decoder: nixe_gpu::texture_decode::BlockDecoder,
    pub(super) sample_conversion: Option<sample_readback::SampleConversion>,
    pub(super) scaling: Option<scaling::Helpers>,
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
        uniform_sync: Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        Self {
            uniform_sync,
            samplers,
            frame_tick: 0,
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
            scaling: None,
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
    /// FenceManager::PopAsyncFlushes runs after the native fence, under the
    /// actual cache locks, and consumes exactly one committed prefix per cache.
    pub(crate) fn pop_async_flushes(&self) -> Result<(), VulkanError> {
        let mut buffers = self.buffer_cache.lock().unwrap();
        let mut textures = self.texture_cache.lock().unwrap();
        textures.pop_async_flush(&mut buffers)?;
        buffers.pop_async_flush()
    }
}
