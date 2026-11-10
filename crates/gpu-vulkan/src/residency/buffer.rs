//! Device-address buffer cache and immutable uploads.
use super::*;
const CACHE_PAGE_SIZE: u64 = 64 * 1024;
// Eden BufferCache::FindBuffer/ResolveOverlaps use one buffer slot per DAddr page.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
pub(super) struct BufferIndex {
    pages: Vec<u32>,
}
impl BufferIndex {
    pub(super) fn new(bits: u32) -> Self {
        Self {
            pages: vec![0; 1usize << (bits - 16)],
        }
    }
    pub(super) fn at(&self, address: u64) -> Option<SlotId> {
        self.pages
            .get((address / CACHE_PAGE_SIZE) as usize)
            .copied()
            .filter(|&slot| slot != 0)
            .map(|slot| SlotId::from_index(slot - 1))
    }
    fn insert(&mut self, slot: SlotId, address: u64, bytes: u64) {
        let value = slot.value() + 1;
        for page in address / CACHE_PAGE_SIZE..(address + bytes).div_ceil(CACHE_PAGE_SIZE) {
            self.pages[page as usize] = value;
        }
    }
    pub(super) fn remove(&mut self, slot: SlotId, address: u64, bytes: u64) {
        for page in address / CACHE_PAGE_SIZE..(address + bytes).div_ceil(CACHE_PAGE_SIZE) {
            let entry = &mut self.pages[page as usize];
            if *entry == slot.value() + 1 {
                *entry = 0;
            }
        }
    }
}
/// A retained native buffer and the addressed subrange within it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferRange {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
}
#[derive(Clone, Copy)]
pub(super) struct BufferStreamRequest {
    pub id: BufferId,
    pub offset: u64,
    pub size: u64,
    pub alignment: u64,
}
pub(super) type BufferStream = Option<crate::queue::Slice>;
impl Residency<'_> {
    /// Match the cache/stream preference over recent guest frames, independently
    /// of host presentation frequency. Alignment copies remain mandatory.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    pub(crate) fn tick_frame(&mut self) {
        self.texture_cache.frame_tick += 1;
        self.runtime.descriptor_queue.tick_frame();
        self.runtime.compute_pass_descriptor_queue.tick_frame();
        let completed = self.runtime.device.completed_tick();
        self.runtime.device.submission.staging.tick_frame(completed);
        for (_, entry) in self.buffer_cache.buffers.iter_mut() {
            if entry
                .contents
                .last_use
                .is_none_or(|work| work.tick() <= completed)
            {
                entry.used.reset();
            }
        }
        let hits: u64 = self
            .runtime
            .buffer_bindings
            .channel
            .uniform_cache_hits
            .iter()
            .sum();
        let shots: u64 = self
            .runtime
            .buffer_bindings
            .channel
            .uniform_cache_shots
            .iter()
            .sum();
        self.runtime.buffer_bindings.channel.uniform_stream_limit =
            if hits * 256 < shots * 251 { 4096 } else { 0 };
        self.runtime
            .buffer_bindings
            .channel
            .uniform_cache_hits
            .copy_within(..17, 1);
        self.runtime
            .buffer_bindings
            .channel
            .uniform_cache_shots
            .copy_within(..17, 1);
        self.runtime.buffer_bindings.channel.uniform_cache_hits[0] = 0;
        self.runtime.buffer_bindings.channel.uniform_cache_shots[0] = 0;
    }
}
impl Residency<'_> {
    fn ordinary_buffer_usage(usage: vk::BufferUsageFlags) -> vk::BufferUsageFlags {
        usage
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
            | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
    }
    pub fn buffer_for(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferRange, VulkanError> {
        self.create_buffer(&backing, 0..backing.bytes(), usage, true)
    }
    pub fn buffer_for_deferred_import(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferRange, VulkanError> {
        self.create_buffer(&backing, 0..backing.bytes(), usage, false)
    }
    pub(crate) fn invalidate_device_mappings(&mut self) -> Result<(), VulkanError> {
        let generation = self.runtime.device_memory.generation();
        if generation == self.buffer_cache.device_mapping_generation {
            return Ok(());
        }
        let changed = self.runtime.device_memory.take_changes();
        self.buffer_cache.device_mapping_generation = generation;
        let mut seen = smallvec::SmallVec::<[SlotId; 16]>::new();
        for (address, bytes) in changed.iter() {
            self.buffer_cache.write_memory(address, bytes);
        }
        for (address, bytes) in changed.iter() {
            for page in address / CACHE_PAGE_SIZE..(address + bytes).div_ceil(CACHE_PAGE_SIZE) {
                if let Some(slot) = self.buffer_cache.buffer_index.at(page * CACHE_PAGE_SIZE)
                    && !seen.contains(&slot)
                {
                    seen.push(slot);
                }
            }
        }
        for slot in seen {
            let entry = &self.buffer_cache.buffers[slot];
            let base = entry.device_address.unwrap();
            let size = entry.native.size();
            let current = self.runtime.device_memory.buffer_mapping(base, size)?;
            if &current == entry.contents.geometry[0].mapping() {
                let entry = &mut self.buffer_cache.buffers[slot];
                for (start, bytes) in changed.intersections(base, size) {
                    entry.invalid.insert(start - base, bytes);
                }
            } else {
                // Keep the physical producer and its pending publications alive.
                // Only the current guest binding is retired by a device remap.
                self.buffer_cache.memory_tracker.cpu_write(base, size);
                self.buffer_cache.clear_download(base, size);
                self.buffer_cache.buffer_index.remove(slot, base, size);
                let id = slot;
                self.runtime.buffer_bindings.invalidate_selections(id);
                self.buffer_cache.buffers[slot].binding_valid = false;
                self.invalidate_bindings();
            }
        }

        Ok(())
    }
    fn create_buffer(
        &mut self,
        requested: &Arc<PhysicalMapping>,
        range: std::ops::Range<u64>,
        usage: vk::BufferUsageFlags,
        initialize: bool,
    ) -> Result<BufferRange, VulkanError> {
        self.invalidate_device_mappings()?;
        if requested.machine() != self.runtime.memory.identity() {
            return Err(VulkanError::InvalidResource(
                "buffer belongs to another machine",
            ));
        }
        if let Some(address) = self
            .runtime
            .device_memory
            .find_address(requested, range.clone())?
        {
            return self.device_buffer(address, range.end - range.start, usage, initialize, false);
        }
        // Host-authored physical resources have no guest DAddr. Their storage
        // retains the ordinary page geometry, without a physical lookup cache.
        let requested_bytes = range.end - range.start;
        let normalized = if range.start == 0 && range.end == requested.bytes() {
            Arc::clone(requested)
        } else {
            requested.slice(range.start, requested_bytes)?
        };
        let mapping = normalized.expanded_buffer(CACHE_PAGE_SIZE)?;
        let offset = mapping.containing_offset(requested, range.clone()).ok_or(
            VulkanError::InvalidResource("host buffer expansion changed addressing"),
        )?;
        let bytes = mapping.bytes();
        let footprint = Footprint::new(
            mapping,
            nixe_gpu::physical::WriteDomain::Buffer { start: 0, bytes },
        )?;
        let id = self.allocate_buffer_entry(
            footprint,
            bytes,
            Self::ordinary_buffer_usage(usage),
            false,
        )?;
        if initialize {
            self.ensure_buffer_range(id, offset, requested_bytes)?;
        }
        Ok(BufferRange {
            buffer: id,
            offset,
            size: requested_bytes,
        })
    }
    pub(crate) fn device_buffer(
        &mut self,
        address: u64,
        requested_bytes: u64,
        usage: vk::BufferUsageFlags,
        initialize: bool,
        sparse_compatible: bool,
    ) -> Result<BufferRange, VulkanError> {
        self.invalidate_device_mappings()?;
        let usage = Self::ordinary_buffer_usage(usage);
        let requested_end = address
            .checked_add(requested_bytes)
            .filter(|&end| {
                requested_bytes != 0 && end <= 1 << self.runtime.device_memory.address_bits()
            })
            .ok_or(VulkanError::InvalidResource(
                "buffer exceeds device address space",
            ))?;
        if let Some(slot) = self.buffer_cache.buffer_index.at(address) {
            let buffer = &self.buffer_cache.buffers[slot];
            let entry = buffer;
            let base = entry.device_address.unwrap();
            if entry.binding_valid
                && entry.usage.contains(usage)
                && (!sparse_compatible || entry.sparse_compatible)
                && address >= base
                && requested_end <= base + entry.native.size()
            {
                let id = slot;
                let offset = address - base;
                if initialize {
                    self.ensure_buffer_range(id, offset, requested_bytes)?;
                }
                self.touch(Resource::Buffer(id));
                self.runtime.statistics.cache_hits += 1;
                return Ok(BufferRange {
                    buffer: id,
                    offset,
                    size: requested_bytes,
                });
            }
        }
        let mut begin = address / CACHE_PAGE_SIZE * CACHE_PAGE_SIZE;
        let mut end = requested_end.div_ceil(CACHE_PAGE_SIZE) * CACHE_PAGE_SIZE;
        let mut cursor = begin;
        let mut overlaps = smallvec::SmallVec::<[BufferId; 16]>::new();
        let mut score = 0u32;
        let mut stream_leap = false;
        while cursor < end {
            let Some(slot) = self.buffer_cache.buffer_index.at(cursor) else {
                cursor += CACHE_PAGE_SIZE;
                continue;
            };
            let buffer = &self.buffer_cache.buffers[slot];
            let entry = buffer;
            if entry.picked {
                cursor += CACHE_PAGE_SIZE;
                continue;
            }
            let base = entry.device_address.unwrap();
            let stop = base + entry.native.size();
            let left = base < begin;
            let right = stop > end;
            begin = begin.min(base);
            end = end.max(stop);
            score = score.saturating_add(entry.growth);
            let id = slot;
            self.buffer_cache.buffers[slot].picked = true;
            overlaps.push(id);
            if score > 16 && !stream_leap {
                stream_leap = true;
                if right {
                    begin = begin
                        .saturating_sub(CACHE_PAGE_SIZE * 128)
                        .max(CACHE_PAGE_SIZE);
                    cursor = begin - CACHE_PAGE_SIZE;
                }
                if left {
                    end = (end + CACHE_PAGE_SIZE * 128)
                        .min(1 << self.runtime.device_memory.address_bits());
                }
            }
            cursor += CACHE_PAGE_SIZE;
        }
        for id in &overlaps {
            self.buffer_cache.buffers[*id].picked = false;
        }
        let growth = if stream_leap {
            0
        } else {
            overlaps
                .iter()
                .map(|id| self.buffer_cache.buffers[*id].growth.saturating_add(1))
                .sum()
        };
        let offset = address - begin;
        let bytes = end - begin;
        let mapping = self.runtime.device_memory.buffer_mapping(begin, bytes)?;
        let footprint = Footprint::new(
            mapping,
            nixe_gpu::physical::WriteDomain::Buffer { start: 0, bytes },
        )?;
        let range = 0..requested_bytes;
        self.runtime
            .constructing
            .extend(overlaps.iter().copied().map(Resource::Buffer));
        let created = self.allocate_buffer_entry(footprint, bytes, usage, sparse_compatible);
        for &id in overlaps.iter().rev() {
            self.unpin(Resource::Buffer(id));
        }
        let id = created?;
        self.buffer_cache.buffers[id].device_address = Some(begin);
        self.buffer_cache.get_mut(id).invalid.clear();
        self.buffer_cache.buffer_index.insert(id, begin, bytes);
        let handle = self.use_buffer(id, 0, bytes)?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime
                .device
                .dispatch
                .cmd_fill_buffer(handle, 0, bytes, 0);
        }

        self.runtime.constructing.push(Resource::Buffer(id));
        let merged = (|| {
            for old in overlaps {
                if old == id {
                    continue;
                }
                let _merge = nixe_gpu::trace::span(
                    "buffer_merge",
                    u64::from(old.value()),
                    u64::from(id.value()),
                    0,
                );
                let old_entry = self.buffer_cache.get(old);
                let destination = old_entry.device_address.unwrap() - begin;
                let old_bytes = old_entry.native.size();
                // Preserve native contents without downloading to RAM. Dirty
                // CPU ranges remain dirty in the new entry until consumption.
                self.copy_buffer(old, 0, id, destination, old_bytes)?;
                self.retire_entry(Resource::Buffer(old))?;
                self.runtime.statistics.buffer_merges += 1;
            }
            self.buffer_cache.buffers[id].growth = growth;
            if initialize {
                self.ensure_buffer_range(id, offset, range.end - range.start)?;
            }
            self.touch(Resource::Buffer(id));
            Ok(BufferRange {
                buffer: id,
                offset,
                size: range.end - range.start,
            })
        })();
        self.unpin(Resource::Buffer(id));
        merged
    }
    fn allocate_buffer_entry(
        &mut self,
        footprint: Arc<Footprint>,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        sparse_compatible: bool,
    ) -> Result<BufferId, VulkanError> {
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && !self.runtime.device.profile().buffer_device_address
        {
            return Err(VulkanError::InvalidResource(
                "buffer device address not enabled",
            ));
        }
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let geometry = vec![footprint];
        let alignment = if sparse_compatible {
            self.runtime.multi_range.block_size
        } else {
            0
        };
        let native = self.allocate_buffer(bytes, usage, alignment)?;
        #[cfg(feature = "binding-probe")]
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                self.runtime.device.owner.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(native.handle),
                )
            }
        } else {
            0
        };
        let id = self.buffer_cache.buffers.insert(BufferEntry {
            growth: 0,
            sparse_compatible,
            write_tick: 0,
            binding_valid: true,
            device_address: None,
            picked: false,
            invalid: RangeSet::from_range(0, bytes),
            contents: Contents {
                lru: None,
                geometry: geometry.clone(),
                last_use: None,
                touched: 0,
            },
            native: Arc::new(native),
            usage,
            #[cfg(feature = "binding-probe")]
            address,
            used: sync::UsageTracker::new(bytes),
        });
        self.runtime.statistics.buffer_allocations += 1;
        self.touch(Resource::Buffer(id));
        Ok(id)
    }
    /// Frequently rewritten CPU uniforms receive immutable upload slices. They
    /// stay dirty and do not install page observers or quiesce running cores.
    pub(super) fn stream_buffer(
        &mut self,
        request: BufferStreamRequest,
    ) -> Result<BufferStream, VulkanError> {
        self.check_buffer_range(request.id, request.offset, request.size)?;
        if request.size > 64 * 1024 {
            return Ok(None);
        }
        let entry = self.buffer_cache.get(request.id);
        let [full] = entry.contents.geometry.as_slice() else {
            return Ok(None);
        };
        if !matches!(
            full.domain(),
            nixe_gpu::physical::WriteDomain::Buffer { .. }
        ) {
            return Ok(None);
        }
        // Stream only CPU-owned bytes, using the same DAddr dirty tracker as
        // SynchronizeBuffer. Physical producer searches are unnecessary here.
        if entry.device_address.is_some_and(|address| {
            self.buffer_cache
                .gpu_modified_ranges
                .intersects(address + request.offset, request.size)
        }) {
            return Ok(None);
        }
        let mapping = full.mapping();
        let address = entry.device_address;
        let slice = self.runtime.device.submission.reserve_slice(
            request.size,
            crate::queue::staging::Usage::Upload,
            request.alignment.max(4),
        )?;
        unsafe {
            self.runtime
                .device
                .submission
                .staging
                .write_with(&slice, |bytes| {
                    if let Some(address) = address {
                        self.runtime
                            .device_reader
                            .read_block_unsafe(address + request.offset, bytes)
                            .map_err(memory_error)
                    } else {
                        mapping
                            .capture_stream(request.offset, bytes)
                            .map_err(memory_error)
                    }
                })?;
        }
        self.touch(Resource::Buffer(request.id));
        self.runtime.statistics.cpu_streamed_bytes += request.size;
        Ok(Some(slice))
    }
    pub(super) fn aligned_uniform_copy(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<crate::queue::Slice, VulkanError> {
        self.ensure_buffer_range(id, offset, size)?;
        let alignment = self
            .runtime
            .device
            .profile()
            .limits
            .min_uniform_buffer_offset_alignment;
        let slice = self.runtime.device.submission.reserve_slice(
            size,
            crate::queue::staging::Usage::Upload,
            alignment.max(4),
        )?;
        let copy = slice.buffer;
        let source = self.use_buffer(id, offset, size)?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;

        unsafe {
            self.runtime.device.dispatch.cmd_copy_buffer(
                source,
                copy,
                &[vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(slice.offset)
                    .size(size)],
                true,
            );
            let barrier = [vk::BufferMemoryBarrier2::default()
                .buffer(copy)
                .offset(slice.offset)
                .size(size)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::UNIFORM_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
            self.runtime
                .device
                .dispatch
                .barrier(&vk::DependencyInfo::default().buffer_memory_barriers(&barrier));
        }
        self.runtime.statistics.gpu_copied_bytes += size;
        Ok(slice)
    }
    pub(super) fn ensure_buffer_range(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        self.check_buffer_range(id, offset, size)?;
        if !self.buffer_cache.binding_current(id) {
            return Err(VulkanError::InvalidResource(
                "buffer binding has retired physical coverage",
            ));
        }
        let entry = self.buffer_cache.get(id);
        let cpu_modified = entry.device_address.is_some_and(|address| {
            self.buffer_cache
                .memory_tracker
                .cpu_modified(address + offset, size)
        });
        if !cpu_modified && !entry.invalid.intersects(offset, size) {
            return Ok(());
        }
        self.runtime.constructing.push(Resource::Buffer(id));
        let result = if let Some(address) = self.buffer_cache.get(id).device_address {
            self.synchronize_device_buffer(id, address, offset, size)
        } else {
            self.upload_host_buffer(id, offset, size)
        };
        self.unpin(Resource::Buffer(id));
        result
    }
    /// SynchronizeBuffer consumes CPU page ranges, subtracting exact GPU
    /// modified bytes. Ordinary RAM uploads do not resolve physical producers.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L1740-L1772
    fn synchronize_device_buffer(
        &mut self,
        id: BufferId,
        address: u64,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        let uploads = self
            .buffer_cache
            .memory_tracker
            .take_uploads(address + offset, size);
        let mut copies = smallvec::SmallVec::<[vk::BufferCopy; 8]>::new();
        let mut packed = 0;
        let mut add = |start: u64, end: u64| {
            if start == end {
                return;
            }
            copies.push(
                vk::BufferCopy::default()
                    .src_offset(packed)
                    .dst_offset(start - address)
                    .size(end - start),
            );
            packed += end - start;
        };
        for (start, size) in uploads {
            let mut cursor = start;
            for (gpu, bytes) in self
                .buffer_cache
                .gpu_modified_ranges
                .intersections(start, size)
            {
                add(cursor, gpu);
                cursor = gpu + bytes;
            }
            add(cursor, start + size);
        }
        if packed != 0 {
            let slice = self.runtime.device.submission.reserve_slice(
                packed,
                crate::queue::staging::Usage::Upload,
                4,
            )?;
            unsafe {
                self.runtime
                    .device
                    .submission
                    .staging
                    .write_with(&slice, |output| {
                        for copy in &copies {
                            self.runtime
                                .device_reader
                                .read_block_unsafe(
                                    address + copy.dst_offset,
                                    &mut output[copy.src_offset as usize
                                        ..(copy.src_offset + copy.size) as usize],
                                )
                                .map_err(memory_error)?;
                        }
                        Ok(())
                    })?;
            }
            self.record_buffer_upload(id, slice, copies)?;
            self.runtime.statistics.cpu_uploaded_bytes += packed;
        }
        Ok(())
    }
    /// Host-authored resources have no SMMU address. Their CPU storage is
    /// uploaded directly; guest resources are synchronized through DAddr.
    fn upload_host_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        let entry = self.buffer_cache.get(id);
        let mapping = Arc::clone(entry.contents.geometry[0].mapping());
        let page = nixe_memory::PAGE_SIZE;
        let begin = offset / page * page;
        let end = ((offset + size).div_ceil(page) * page).min(entry.native.size());
        let ranges: smallvec::SmallVec<[(u64, u64); 8]> =
            entry.invalid.intersections(begin, end - begin).collect();
        for (offset, bytes) in ranges {
            self.buffer_cache
                .get_mut(id)
                .invalid
                .subtract(offset, bytes);
            let slice = self.runtime.device.submission.reserve_slice(
                bytes,
                crate::queue::staging::Usage::Upload,
                4,
            )?;
            unsafe {
                self.runtime
                    .device
                    .submission
                    .staging
                    .write_with(&slice, |output| {
                        mapping.capture_stream(offset, output).map_err(memory_error)
                    })?;
            }
            self.record_buffer_upload(
                id,
                slice,
                smallvec::smallvec![vk::BufferCopy::default().dst_offset(offset).size(bytes)],
            )?;
            self.runtime.statistics.cpu_uploaded_bytes += bytes;
        }
        Ok(())
    }
    /// AccelerateInlineToMemory preserves GPU-owned neighbours while uploading
    /// the engine's written bytes in native queue order.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L179-L199
    pub(super) fn inline_memory(&mut self, address: u64, bytes: u64) -> Result<bool, VulkanError> {
        let registered = (address >> 16..(address + bytes).div_ceil(1 << 16))
            .any(|page| self.buffer_cache.buffer_index.at(page << 16).is_some());
        if !registered {
            return Ok(false);
        }
        let page = nixe_memory::PAGE_SIZE;
        let first = address / page * page;
        let end = (address + bytes).div_ceil(page) * page;
        if !self
            .buffer_cache
            .gpu_modified_ranges
            .intersects(first, end - first)
        {
            return Ok(false);
        }
        self.buffer_cache.clear_download(address, bytes);
        self.buffer_cache
            .gpu_modified_ranges
            .subtract(address, bytes);
        let range = self.device_buffer(
            address,
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
            false,
        )?;
        let slice = self.runtime.device.submission.reserve_slice(
            bytes,
            crate::queue::staging::Usage::Upload,
            4,
        )?;
        unsafe {
            self.runtime
                .device
                .submission
                .staging
                .write_with(&slice, |output| {
                    self.runtime
                        .device_reader
                        .read_block_unsafe(address, output)
                        .map_err(memory_error)
                })?;
        }
        self.record_buffer_upload(
            range.buffer,
            slice,
            smallvec::smallvec![
                vk::BufferCopy::default()
                    .dst_offset(range.offset)
                    .size(bytes)
            ],
        )?;
        self.runtime.statistics.cpu_uploaded_bytes += bytes;
        Ok(true)
    }
    pub fn buffer_lease(&self, id: BufferId) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        Ok(Arc::clone(&self.buffer_cache.get(id).native))
    }
}
