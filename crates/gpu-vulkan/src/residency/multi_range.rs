//! Native multi-range storage buffers, with sparse aliases or readonly gathers.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_multi_range_buffer.cpp
use super::*;
use ash::vk::Handle;

const DEFAULT_BLOCK_SIZE: u64 = 64 * 1024;
const MAX_RETIRED: usize = 256;

#[derive(Clone)]
pub struct MultiRangeBinding {
    pub(crate) native: Arc<AllocatedBuffer>,
    pub(super) offset: u64,
    pub(super) size: u64,
    pub(super) sources: Arc<[BufferRange]>,
    pub(super) sparse: bool,
    owners: Arc<[Arc<AllocatedBuffer>]>,
}
struct Entry {
    binding: MultiRangeBinding,
    geometry: u64,
}
struct Retired {
    _binding: MultiRangeBinding,
    work: WorkTicket,
}
pub(super) struct MultiRangeCache {
    entries: HashMap<u64, Entry>,
    retired: Vec<Retired>,
    pub(super) block_size: u64,
    pub(super) use_sparse: bool,
    memory_types: u32,
    usage: vk::BufferUsageFlags,
}
impl MultiRangeCache {
    pub(super) fn new(device: &VulkanDevice) -> Self {
        let mut usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        if device.profile().buffer_device_address {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        let mut cache = Self {
            entries: HashMap::default(),
            retired: Vec::new(),
            block_size: DEFAULT_BLOCK_SIZE,
            use_sparse: false,
            memory_types: 0,
            usage,
        };
        if device.profile().core.sparse_binding == 0
            || device.profile().core.sparse_residency_aliased == 0
        {
            return cache;
        }
        let families = unsafe {
            device
                .owner
                .instance
                .instance
                .get_physical_device_queue_family_properties(device.owner.physical)
        };
        if !families[device.queues().graphics_compute_family as usize]
            .queue_flags
            .contains(vk::QueueFlags::SPARSE_BINDING)
        {
            return cache;
        }
        let Ok(probe) = AllocatedBuffer::sparse(&device.owner, DEFAULT_BLOCK_SIZE, usage) else {
            return cache;
        };
        let requirements = unsafe {
            device
                .owner
                .device
                .get_buffer_memory_requirements(probe.handle)
        };
        if requirements.alignment != 0 && requirements.memory_type_bits != 0 {
            cache.block_size = requirements.alignment;
            cache.memory_types = requirements.memory_type_bits;
            cache.use_sparse = true;
        }
        cache
    }
}
struct Source {
    range: BufferRange,
    native: Arc<AllocatedBuffer>,
}
fn hash(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(0xcbf29ce484222325, |hash, value| {
        (hash ^ value).wrapping_mul(0x100000001b3)
    })
}
impl Residency<'_> {
    pub(crate) fn multi_range_storage(
        &mut self,
        key: u64,
        ranges: &[nixe_gpu::native::GuestBufferRange],
        written: bool,
    ) -> Result<MultiRangeBinding, VulkanError> {
        if ranges.len() < 2 {
            return Err(VulkanError::InvalidResource(
                "multi-range storage needs at least two device ranges",
            ));
        }
        let pinned = self.runtime.constructing.len();
        let result = (|| {
            let prefer_sparse = self.runtime.multi_range.use_sparse;
            let mut sources = smallvec::SmallVec::<[Source; 8]>::new();
            let mut total = 0u64;
            // Resolve every allocation before retaining selections: a later
            // overlapping range can merge away an earlier buffer slot.
            // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/buffer_cache/multi_range_refresh.h
            for guest in ranges {
                let range = self.device_buffer(
                    guest.address,
                    guest.size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    false,
                    prefer_sparse,
                )?;
                self.runtime
                    .constructing
                    .push(Resource::Buffer(range.buffer));
            }
            for guest in ranges {
                let range = self.device_buffer(
                    guest.address,
                    guest.size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    false,
                    prefer_sparse,
                )?;
                self.runtime
                    .constructing
                    .push(Resource::Buffer(range.buffer));
                let entry = self.buffer_cache.get(range.buffer);
                sources.push(Source {
                    range,
                    native: Arc::clone(&entry.native),
                });
                total = total
                    .checked_add(range.size)
                    .ok_or(VulkanError::InvalidResource(
                        "multi-range storage size overflow",
                    ))?;
            }
            for source in &sources {
                let range = source.range;
                self.ensure_buffer_range(range.buffer, range.offset, range.size)?;
            }
            if total
                > u64::from(
                    self.runtime
                        .device
                        .profile()
                        .limits
                        .max_storage_buffer_range,
                )
            {
                return Err(VulkanError::InvalidResource(
                    "multi-range storage exceeds native storage buffer range",
                ));
            }
            self.drain_multi_range_retired();
            let geometry = hash(
                sources
                    .iter()
                    .flat_map(|s| [s.native.handle.as_raw(), s.range.offset, s.range.size]),
            );
            let reusable = self
                .runtime
                .multi_range
                .entries
                .get(&key)
                .is_some_and(|entry| {
                    entry.geometry == geometry
                        && entry.binding.size == total
                        && entry.binding.sources.len() == sources.len()
                        && entry
                            .binding
                            .sources
                            .iter()
                            .zip(entry.binding.owners.iter())
                            .zip(&sources)
                            .all(|((&range, native), source)| {
                                range == source.range && Arc::ptr_eq(native, &source.native)
                            })
                });
            if !reusable {
                if let Some(old) = self.runtime.multi_range.entries.remove(&key) {
                    self.retire_multi_range(old.binding)?;
                }
                let native = self.create_sparse_multi_range(&sources, total)?;
                let sparse = native.is_some();
                if written && !sparse {
                    return Err(VulkanError::Compilation(format!(
                        "writable discontinuous storage cannot bind sparse: key={key:#x} block={:#x} sparse_available={} total={total:#x} regions={:?}",
                        self.runtime.multi_range.block_size, self.runtime.multi_range.use_sparse,
                        sources.iter().map(|source| { let (memory, offset, memory_type) = source.native.memory_location();
                            (source.range.offset, source.range.size, source.native.size(), memory.as_raw(), offset, memory_type) }).collect::<Vec<_>>()
                    ).into()));
                }
                let (native, offset) = match native {
                    Some(buffer) => buffer,
                    None => (
                        Arc::new(self.allocate_buffer(total, self.runtime.multi_range.usage, 0)?),
                        0,
                    ),
                };
                self.runtime.multi_range.entries.insert(
                    key,
                    Entry {
                        binding: MultiRangeBinding {
                            native,
                            offset,
                            size: total,
                            sparse,
                            // Both representations depend on source geometry.
                            // Keep owners alive and invalidate their selections
                            // when a physical buffer slot is retired or merged.
                            owners: sources
                                .iter()
                                .map(|source| Arc::clone(&source.native))
                                .collect(),
                            sources: sources
                                .iter()
                                .map(|source| source.range)
                                .collect::<Vec<_>>()
                                .into(),
                        },
                        geometry,
                    },
                );
            }
            let entry = &self.runtime.multi_range.entries[&key];
            let binding = entry.binding.clone();
            if written && !binding.sparse {
                return Err(VulkanError::InvalidResource(
                    "writable discontinuous storage cannot use a readonly gathered buffer",
                ));
            }
            if !binding.sparse {
                // Submission ticks do not identify content: CPU uploads and
                // multiple GPU writes can share one tick. Refresh gathered
                // reads at consumption; sparse bindings alias current storage.
                // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/buffer_cache/multi_range_refresh.h
                self.gather_multi_range(&sources, &binding.native)?;
            }
            Ok(binding)
        })();
        self.runtime.constructing.truncate(pinned);
        result
    }
    fn create_sparse_multi_range(
        &self,
        sources: &[Source],
        total: u64,
    ) -> Result<Option<(Arc<AllocatedBuffer>, u64)>, VulkanError> {
        let cache = &self.runtime.multi_range;
        if !cache.use_sparse {
            return Ok(None);
        }
        // Sparse binds cover complete host memory blocks. The descriptor can
        // start inside the first block, while its range exposes only guest
        // bytes. Padding must stay inside the retained source allocations;
        // interior boundaries must still meet the host block alignment.
        // https://docs.vulkan.org/spec/latest/chapters/sparsemem.html#sparsememory-resource-binding
        let block = cache.block_size;
        let (_, first_base, _) = sources[0].native.memory_location();
        let padding = (first_base + sources[0].range.offset) % block;
        let descriptor_alignment = self
            .runtime
            .device
            .profile()
            .limits
            .min_storage_buffer_offset_alignment;
        if !padding.is_multiple_of(descriptor_alignment) {
            return Ok(None);
        }
        let mut at = 0u64;
        let mut binds = Vec::with_capacity(sources.len());
        for (index, source) in sources.iter().enumerate() {
            let (memory, base, memory_type) = source.native.memory_location();
            let prefix = if index == 0 { padding } else { 0 };
            let Some(offset) = source.range.offset.checked_sub(prefix) else {
                return Ok(None);
            };
            let bytes = source.range.size + prefix;
            let bytes = if index + 1 == sources.len() {
                bytes.next_multiple_of(block)
            } else {
                bytes
            };
            if memory == vk::DeviceMemory::null()
                || memory_type >= 32
                || cache.memory_types & (1 << memory_type) == 0
                || !(base + offset).is_multiple_of(block)
                || !bytes.is_multiple_of(block)
                || offset + bytes > source.native.size()
            {
                return Ok(None);
            }
            binds.push(
                vk::SparseMemoryBind::default()
                    .resource_offset(at)
                    .size(bytes)
                    .memory(memory)
                    .memory_offset(base + offset),
            );
            at += bytes;
        }
        debug_assert_eq!(at, (total + padding).next_multiple_of(block));
        let Ok(buffer) = AllocatedBuffer::sparse(&self.runtime.device.owner, at, cache.usage)
        else {
            return Ok(None);
        };
        let buffers = [vk::SparseBufferMemoryBindInfo::default()
            .buffer(buffer.handle)
            .binds(&binds)];
        let info = [vk::BindSparseInfo::default().buffer_binds(&buffers)];
        let owner = &self.runtime.device.owner;
        let fence = unsafe {
            owner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| VulkanError::vk("vkCreateFence(multi-range)", error))?;
        let bound = {
            let _submit = owner.queue_mutex.lock().unwrap();
            unsafe {
                owner
                    .device
                    .queue_bind_sparse(self.runtime.device.queue, &info, fence)
            }
        };
        let result = match bound {
            Ok(()) => unsafe { owner.device.wait_for_fences(&[fence], true, u64::MAX) }
                .map(|_| Some((Arc::new(buffer), padding)))
                .map_err(|error| VulkanError::vk("vkWaitForFences(multi-range)", error)),
            Err(vk::Result::ERROR_DEVICE_LOST) => Err(VulkanError::vk(
                "vkQueueBindSparse",
                vk::Result::ERROR_DEVICE_LOST,
            )),
            Err(_) => Ok(None),
        };
        unsafe {
            owner.device.destroy_fence(fence, None);
        }
        result
    }
    fn gather_multi_range(
        &mut self,
        sources: &[Source],
        target: &AllocatedBuffer,
    ) -> Result<(), VulkanError> {
        self.runtime.device.end_rendering()?;
        for source in sources {
            self.use_buffer(source.range.buffer, source.range.offset, source.range.size)?;
        }
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let stages = vk::PipelineStageFlags2::ALL_GRAPHICS
            | vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::ALL_TRANSFER;
        let pre = [vk::MemoryBarrier2::default()
            .src_stage_mask(stages)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE)];
        let post = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        unsafe {
            self.runtime
                .device
                .dispatch
                .barrier(&vk::DependencyInfo::default().memory_barriers(&pre));
            let mut at = 0;
            for source in sources {
                self.runtime.device.dispatch.cmd_copy_buffer(
                    source.native.handle,
                    target.handle,
                    &[vk::BufferCopy::default()
                        .src_offset(source.range.offset)
                        .dst_offset(at)
                        .size(source.range.size)],
                    false,
                );
                at += source.range.size;
            }
            self.runtime
                .device
                .dispatch
                .barrier(&vk::DependencyInfo::default().memory_barriers(&post));
        }
        self.runtime.statistics.gpu_copied_bytes += target.size();
        Ok(())
    }
    fn drain_multi_range_retired(&mut self) {
        let completed = self.runtime.device.completed_tick();
        let retired = &mut self.runtime.multi_range.retired;
        let mut index = 0;
        while index < retired.len() {
            if retired[index].work.tick() <= completed {
                retired.swap_remove(index);
            } else {
                index += 1;
            }
        }
    }
    fn retire_multi_range(&mut self, binding: MultiRangeBinding) -> Result<(), VulkanError> {
        if self.runtime.multi_range.retired.len() == MAX_RETIRED {
            self.drain_multi_range_retired();
        }
        if self.runtime.multi_range.retired.len() == MAX_RETIRED {
            let work = self
                .runtime
                .multi_range
                .retired
                .iter()
                .min_by_key(|entry| entry.work.tick())
                .unwrap()
                .work;
            self.flush()?;
            self.runtime
                .device
                .wait_for_work(work, std::time::Duration::MAX)?;
            self.drain_multi_range_retired();
        }
        let work = self.runtime.device.current_work();
        self.runtime.multi_range.retired.push(Retired {
            _binding: binding,
            work,
        });
        Ok(())
    }
    pub(super) fn drop_multi_range_owner(&mut self, id: BufferId) -> Result<(), VulkanError> {
        let keys: smallvec::SmallVec<[u64; 8]> = self
            .runtime
            .multi_range
            .entries
            .iter()
            .filter_map(|(&key, entry)| {
                entry
                    .binding
                    .sources
                    .iter()
                    .any(|source| source.buffer == id)
                    .then_some(key)
            })
            .collect();
        for key in keys {
            let entry = self.runtime.multi_range.entries.remove(&key).unwrap();
            self.retire_multi_range(entry.binding)?;
        }
        Ok(())
    }
}
