//! Current buffer selections belong to the GPU owner, independently of
//! immutable pipelines and native command captures.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache_base.h
use super::*;
use crate::residency::execution::{Draw, IndexBuffer, NativeIndexBuffer, VertexBuffer};
use nixe_gpu::native::{GraphicsDirty, GuestBufferRange};
/// GPU-owner channel storage; CPU coherence and fence publication share only
/// BufferCache contents. Channel state and engine dirty flags never leave this owner.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/control/channel_state_cache.h
#[derive(Default)]
pub(crate) struct BufferBindings {
    pub(crate) channel: Box<BufferChannel>,
    channels: HashMap<u64, Option<Box<BufferChannel>>>,
    current_channel: Option<u64>,
}
pub(crate) struct BufferChannel {
    // Buffer retirement and remapping invalidate actual current slots. Engine
    // register changes are consumed directly from the GPU-owned register bank.
    pub(crate) invalidated_buffers: u128,
    pub(crate) vertex_buffers: Vec<Option<VertexBuffer>>,
    pub(crate) enabled_vertex_buffers: u32,
    // Selection updates consume hardware dirty bits; native binds consume
    // these slot bits after resource preparation has finished.
    pub(crate) dirty_vertex_buffers: u32,
    pub(crate) index_buffer: Option<IndexBuffer>,
    pub(crate) storage_buffers: [Box<[Option<StorageBuffer>]>; 6],
    pub(crate) compute_uniform_buffers: Box<[Option<UniformBuffer>]>,
    pub(crate) has_deleted_buffers: bool,
    pub(crate) streamed_uniforms: [u32; 6],
    pub(crate) uniform_buffers: Vec<Box<[Option<UniformBuffer>]>>,
    pub(crate) uniform_bind_changes: [u32; nixe_gpu_maxwell::MAXWELL_BIND_GROUP_COUNT],
    pub(crate) address_space: Option<u64>,
    pub(crate) mapping_generation: u64,
    pub(crate) indexed: bool,
    pub(crate) uniform_cache_hits: [u64; 18],
    pub(crate) uniform_cache_shots: [u64; 18],
    pub(crate) uniform_stream_limit: u64,
}
impl Default for BufferChannel {
    fn default() -> Self {
        Self {
            invalidated_buffers: 0,
            vertex_buffers: Vec::new(),
            enabled_vertex_buffers: 0,
            dirty_vertex_buffers: 0,
            index_buffer: None,
            uniform_buffers: Vec::new(),
            uniform_bind_changes: [0; nixe_gpu_maxwell::MAXWELL_BIND_GROUP_COUNT],
            storage_buffers: std::array::from_fn(|_| {
                vec![None; nixe_shader_recompiler::artifact::MAX_SSBOS].into_boxed_slice()
            }),
            compute_uniform_buffers: vec![None; 8].into_boxed_slice(),
            has_deleted_buffers: false,
            streamed_uniforms: [0; 6],
            address_space: None,
            mapping_generation: 0,
            indexed: false,
            uniform_cache_hits: [0; 18],
            uniform_cache_shots: [0; 18],
            uniform_stream_limit: 4096,
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) struct UniformBuffer {
    pub(crate) source: GuestBufferRange,
    /// Shader coverage already resolved against the current guest mapping.
    /// A hardware bank may declare more bytes than its backed extent.
    pub(crate) resolved_size: u64,
    pub(crate) native: Option<BufferRange>,
}
/// Descriptor contents and virtual mappings identify a storage selection.
/// Physical buffer retirement invalidates its native binding independently.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageBufferKey {
    pub(crate) descriptor: [u64; 2],
    pub(crate) address_space: u64,
    pub(crate) mapping_generation: u64,
    pub(crate) written: bool,
}
#[derive(Clone)]
pub(crate) struct StorageBuffer {
    pub(crate) key: StorageBufferKey,
    pub(crate) source: GuestBufferRange,
    pub(crate) gpu_address: u64,
    pub(crate) native: Option<StorageBufferBinding>,
}
#[derive(Clone)]
pub(crate) enum StorageBufferBinding {
    Buffer(BufferRange),
    MultiRange(super::multi_range::MultiRangeBinding),
    PointerTable(Arc<super::pointer_table::PointerTableBinding>),
}
impl BufferBindings {
    pub(super) fn current_channel(&self) -> Option<u64> {
        self.current_channel
    }
    pub(crate) fn create_channel(&mut self, id: u64, vertices: usize, groups: usize, slots: usize) {
        let channel = BufferChannel {
            address_space: None,
            mapping_generation: 0,
            indexed: false,
            vertex_buffers: vec![None; vertices],
            index_buffer: None,
            uniform_buffers: (0..groups)
                .map(|_| vec![None; slots].into_boxed_slice())
                .collect(),
            ..Default::default()
        };
        assert!(
            self.channels.insert(id, Some(Box::new(channel))).is_none(),
            "buffer channel already exists"
        );
    }
    pub(crate) fn bind_channel(&mut self, id: u64) {
        if self.current_channel == Some(id) {
            return;
        }
        let next = self
            .channels
            .get_mut(&id)
            .expect("registered buffer channel")
            .take()
            .expect("inactive buffer channel");
        let old = std::mem::replace(&mut self.channel, next);
        if let Some(previous) = self.current_channel.replace(id) {
            *self.channels.get_mut(&previous).unwrap() = Some(old);
        }
    }
    pub(crate) fn close_channel(&mut self, id: u64) {
        self.channels.remove(&id);
        if self.current_channel == Some(id) {
            self.current_channel = None;
            *self.channel = Default::default();
        }
    }
    fn invalidate_buffer_channel(channel: &mut BufferChannel, id: BufferId) {
        channel.has_deleted_buffers = true;
        for storage in channel
            .storage_buffers
            .iter_mut()
            .flat_map(|s| s.iter_mut())
            .flatten()
        {
            if storage.native.as_ref().is_some_and(|native| match native {
                StorageBufferBinding::Buffer(range) => range.buffer == id,
                StorageBufferBinding::PointerTable(binding) => {
                    binding.targets.iter().any(|r| r.buffer == id)
                }
                StorageBufferBinding::MultiRange(binding) => {
                    binding.sources.iter().any(|s| s.buffer == id)
                }
            }) {
                storage.native = None;
            }
        }
        for uniform in channel.compute_uniform_buffers.iter_mut().flatten() {
            if uniform.native.is_some_and(|range| range.buffer == id) {
                uniform.native = None;
            }
        }
        let mut dirty = 0;
        for (slot, binding) in channel.vertex_buffers.iter_mut().enumerate() {
            if binding.as_ref().is_some_and(|binding| binding.buffer == id) {
                dirty |= 1_u128 << (GraphicsDirty::VERTEX_SHIFT + slot as u32);
                *binding = None;
                channel.enabled_vertex_buffers &= !(1 << slot);
            }
        }
        if channel
            .index_buffer
            .as_ref()
            .is_some_and(|binding| binding.buffer == id)
        {
            dirty |= GraphicsDirty::INDEX;
            channel.index_buffer = None;
        }
        channel.invalidated_buffers |= dirty;
        for uniform in channel
            .uniform_buffers
            .iter_mut()
            .flat_map(|group| group.iter_mut())
            .flatten()
        {
            if uniform.native.is_some_and(|range| range.buffer == id) {
                uniform.native = None;
            }
        }
    }
    pub(super) fn invalidate_selections(&mut self, id: BufferId) {
        Self::invalidate_buffer_channel(&mut self.channel, id);
        for channel in self.channels.values_mut().flatten() {
            Self::invalidate_buffer_channel(channel, id);
        }
    }
    pub(crate) fn bind_compute_uniform(&mut self, slot: usize, source: GuestBufferRange) {
        if self.channel.compute_uniform_buffers[slot]
            .is_some_and(|binding| binding.source == source)
        {
            return;
        }
        self.channel.compute_uniform_buffers[slot] = Some(UniformBuffer {
            resolved_size: source.size,
            source,
            native: None,
        });
    }
    /// A repeated binding retains its selection. Content changes are tracked
    /// by BufferCache; physical remaps and retirement invalidate native IDs.
    pub(crate) fn bind_graphics_uniform(
        &mut self,
        group: u8,
        slot: u8,
        source: Option<GuestBufferRange>,
    ) {
        if self.channel.uniform_buffers[group as usize][slot as usize].map(|binding| binding.source)
            == source
        {
            return;
        }
        self.channel.uniform_buffers[group as usize][slot as usize] =
            source.map(|source| UniformBuffer {
                resolved_size: source.size,
                source,
                native: None,
            });
        self.channel.uniform_bind_changes[group as usize] |= 1 << slot;
    }
}
impl Residency<'_> {
    /// Bind the actual buffer-cache slot. Native recording reads this bank;
    /// there is no copied geometry list owned by a prepared draw.
    pub fn bind_vertex_buffer(
        &mut self,
        slot: u32,
        vertex: Option<VertexBuffer>,
    ) -> Result<(), VulkanError> {
        if let Some(vertex) = vertex {
            self.check_buffer_range(vertex.buffer, vertex.offset, vertex.size)?;
            if vertex.binding != slot
                || !self
                    .buffer_cache
                    .get(vertex.buffer)
                    .usage
                    .contains(vk::BufferUsageFlags::VERTEX_BUFFER)
            {
                return Err(VulkanError::InvalidResource(
                    "vertex stream requires matching VERTEX_BUFFER storage",
                ));
            }
        }
        let slots = &mut self.runtime.buffer_bindings.channel.vertex_buffers;
        if slots.len() <= slot as usize {
            slots.resize(slot as usize + 1, None);
        }
        slots[slot as usize] = vertex;
        self.runtime.buffer_bindings.channel.dirty_vertex_buffers |= 1 << slot;
        if vertex.is_some() {
            self.runtime.buffer_bindings.channel.enabled_vertex_buffers |= 1 << slot;
        } else {
            self.runtime.buffer_bindings.channel.enabled_vertex_buffers &= !(1 << slot);
        }
        Ok(())
    }
    /// Index conversion finishes before any native geometry binding, including
    /// the cold LUT path that replaces storage after finishing a scheduler prefix.
    pub(crate) fn bind_host_geometry_buffers(
        &mut self,
        channel: Option<u64>,
        enabled: u32,
        draw: Draw,
        native_index: Option<&NativeIndexBuffer>,
        stride: impl Fn(u32) -> u32,
    ) -> Result<(), VulkanError> {
        if draw.is_empty() {
            return Ok(());
        }
        if matches!(
            draw,
            Draw::Indexed { .. } | Draw::Indirect { indexed: true, .. }
        ) {
            self.bind_host_index_buffer(draw, native_index)?;
        }
        self.bind_host_vertex_buffers(channel, enabled, stride)
    }
    fn bind_host_index_buffer(
        &mut self,
        draw: Draw,
        native: Option<&NativeIndexBuffer>,
    ) -> Result<(), VulkanError> {
        let (buffer, offset, size, kind) = if let Some(native) = native {
            (native.buffer, native.offset, native.size, native.kind)
        } else {
            let index = self.runtime.buffer_bindings.channel.index_buffer.ok_or(
                VulkanError::InvalidResource("indexed draw has no index stream"),
            )?;
            if index.kind == vk::IndexType::UINT8_EXT && !self.runtime.device.profile().index_uint8
            {
                return Err(VulkanError::InvalidResource(
                    "UINT8 index consumption requires indexTypeUint8 or native index conversion",
                ));
            }
            self.ensure_buffer_range(index.buffer, index.offset, index.size)?;
            let handle = self.use_buffer(index.buffer, index.offset, index.size)?;
            (handle, index.offset, index.size, index.kind)
        };
        if let Draw::Indexed {
            indices,
            first_index,
            ..
        } = draw
        {
            let element = match kind {
                vk::IndexType::UINT8_EXT => 1,
                vk::IndexType::UINT16 => 2,
                _ => 4,
            };
            if (u64::from(first_index) + u64::from(indices)) * element > size {
                return Err(VulkanError::InvalidResource(
                    "indexed draw exceeds bound index window",
                ));
            }
        }
        self.runtime.device.submission.statistics.index_binds += 1;
        unsafe {
            self.runtime
                .device
                .dispatch
                .cmd_bind_index_buffer(buffer, offset, kind);
        }
        Ok(())
    }
    /// Synchronize active slots and bind changed contiguous groups, as Eden's
    /// BindHostVertexBuffers. The command context owns only channel identity;
    /// resource selections and dirty flags remain in BufferCache.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L810-L860
    fn bind_host_vertex_buffers(
        &mut self,
        channel: Option<u64>,
        mut enabled: u32,
        stride: impl Fn(u32) -> u32,
    ) -> Result<(), VulkanError> {
        let work = self.runtime.device.current_work();
        let recording = self.runtime.device.submission.recording.unwrap();
        let slot = &mut self.runtime.device.submission.slots[recording];
        if channel.is_none() || slot.vertex_channel != channel {
            self.runtime.buffer_bindings.channel.dirty_vertex_buffers |= enabled;
        }
        slot.vertex_channel = channel;
        let mut first = 0;
        let mut buffers = smallvec::SmallVec::<[vk::Buffer; 32]>::new();
        let mut offsets = smallvec::SmallVec::<[u64; 32]>::new();
        let mut sizes = smallvec::SmallVec::<[u64; 32]>::new();
        let mut strides = smallvec::SmallVec::<[u64; 32]>::new();
        while enabled != 0 {
            let index = enabled.trailing_zeros();
            enabled &= enabled - 1;
            let vertex = self.runtime.buffer_bindings.channel.vertex_buffers[index as usize]
                .ok_or(VulkanError::InvalidResource("active vertex slot is null"))?;
            self.touch(Resource::Buffer(vertex.buffer));
            self.ensure_buffer_range(vertex.buffer, vertex.offset, vertex.size)?;
            let dirty = self.runtime.buffer_bindings.channel.dirty_vertex_buffers;
            if dirty & (1 << index) == 0 {
                self.emit_vertex_buffers(first, &buffers, &offsets, &sizes, &strides);
                buffers.clear();
                offsets.clear();
                sizes.clear();
                strides.clear();
                continue;
            }
            self.runtime.buffer_bindings.channel.dirty_vertex_buffers &= !(1 << index);
            let entry = &mut self.buffer_cache.buffers[vertex.buffer];
            entry.used.mark(vertex.offset, vertex.size);
            entry.contents.last_use = Some(work);
            let handle = entry.native.handle;
            if !buffers.is_empty() && index != first + buffers.len() as u32 {
                self.emit_vertex_buffers(first, &buffers, &offsets, &sizes, &strides);
                buffers.clear();
                offsets.clear();
                sizes.clear();
                strides.clear();
            }
            if buffers.is_empty() {
                first = index;
            }
            buffers.push(handle);
            offsets.push(vertex.offset);
            sizes.push(vertex.size);
            strides.push(u64::from(stride(index)));
        }
        self.emit_vertex_buffers(first, &buffers, &offsets, &sizes, &strides);
        Ok(())
    }
    fn emit_vertex_buffers(
        &mut self,
        first: u32,
        buffers: &[vk::Buffer],
        offsets: &[u64],
        sizes: &[u64],
        strides: &[u64],
    ) {
        let available = self
            .runtime
            .device
            .profile()
            .limits
            .max_vertex_input_bindings
            .saturating_sub(first) as usize;
        let count = buffers.len().min(available);
        if count == 0 {
            return;
        }
        let buffers = &buffers[..count];
        let offsets = &offsets[..count];
        let sizes = &sizes[..count];
        let strides = &strides[..count];
        self.runtime.device.submission.statistics.vertex_binds += 1;
        // Dispatch captures the native arguments; actual use ticks retain storage.
        unsafe {
            if self.runtime.device.profile().dynamic_state.base {
                self.runtime
                    .device
                    .dispatch
                    .cmd_bind_vertex_buffers2(first, buffers, offsets, sizes, strides);
            } else {
                self.runtime
                    .device
                    .dispatch
                    .cmd_bind_vertex_buffers(first, buffers, offsets);
            }
        }
    }
    pub fn bind_index_buffer(&mut self, index: Option<IndexBuffer>) -> Result<(), VulkanError> {
        if let Some(index) = index {
            let element = match index.kind {
                vk::IndexType::UINT16 => 2,
                vk::IndexType::UINT32 => 4,
                vk::IndexType::UINT8_EXT => 1,
                _ => {
                    return Err(VulkanError::InvalidResource(
                        "native index type requires UINT16/UINT32; frontend must widen unsupported UINT8",
                    ));
                }
            };
            self.check_buffer_range(index.buffer, index.offset, index.size)?;
            if !index.offset.is_multiple_of(element)
                || !self
                    .buffer_cache
                    .get(index.buffer)
                    .usage
                    .contains(vk::BufferUsageFlags::INDEX_BUFFER)
            {
                return Err(VulkanError::InvalidResource(
                    "index stream requires aligned INDEX_BUFFER storage",
                ));
            }
        }
        self.runtime.buffer_bindings.channel.index_buffer = index;
        Ok(())
    }
    pub(crate) fn update_storage_buffer(
        &mut self,
        stage: usize,
        slot: usize,
        space: &impl nixe_gpu::native::ImageAddressSpace,
    ) -> Result<(), VulkanError> {
        self.invalidate_device_mappings()?;
        let Some(binding) =
            self.runtime.buffer_bindings.channel.storage_buffers[stage][slot].as_ref()
        else {
            return Ok(());
        };
        if matches!(binding.native, Some(StorageBufferBinding::Buffer(_))) {
            return Ok(());
        }
        let source = binding.source;
        let gpu_address = binding.gpu_address;
        let written = binding.key.written;
        let generation = binding.key.mapping_generation;
        // Resolve scatter geometry before allocating physical storage. A
        // fragmented virtual span must not allocate an unused contiguous
        // buffer across unrelated device addresses. Copy only scatter regions
        // before buffer lookup can mutate the cache's slots.
        let segments =
            self.buffer_cache
                .virtual_ranges
                .query(space, gpu_address, source.size, generation)?;
        let multiple = (segments.len() >= 2)
            .then(|| smallvec::SmallVec::<[GuestBufferRange; 8]>::from_slice(segments));
        let native = if let Some(segments) = multiple {
            StorageBufferBinding::MultiRange(self.multi_range_storage(
                (space.image_space_id() << 48) ^ gpu_address,
                &segments,
                written,
            )?)
        } else {
            let ordinary = self.device_buffer(
                source.address,
                source.size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
                false,
            )?;
            StorageBufferBinding::Buffer(ordinary)
        };
        self.runtime.buffer_bindings.channel.storage_buffers[stage][slot]
            .as_mut()
            .unwrap()
            .native = Some(native);
        Ok(())
    }
    pub(crate) fn compute_uniform(
        &mut self,
        slot: usize,
    ) -> Result<Option<BufferRange>, VulkanError> {
        let Some(binding) = self.runtime.buffer_bindings.channel.compute_uniform_buffers[slot]
        else {
            return Ok(None);
        };
        if let Some(native) = binding.native {
            return Ok(Some(native));
        }
        let range = self.device_buffer(
            binding.source.address,
            binding.source.size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            false,
            false,
        )?;
        self.runtime.buffer_bindings.channel.compute_uniform_buffers[slot]
            .as_mut()
            .unwrap()
            .native = Some(range);
        Ok(Some(range))
    }
    pub(crate) fn graphics_uniform(
        &mut self,
        group: u8,
        slot: u8,
    ) -> Result<Option<BufferRange>, VulkanError> {
        let Some(binding) =
            self.runtime.buffer_bindings.channel.uniform_buffers[group as usize][slot as usize]
        else {
            return Ok(None);
        };
        if let Some(range) = binding.native {
            return Ok(Some(range));
        }
        let source = binding.source;
        let range = self.device_buffer(
            source.address,
            source.size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            false,
            false,
        )?;
        self.runtime.buffer_bindings.channel.uniform_buffers[group as usize][slot as usize]
            .as_mut()
            .unwrap()
            .native = Some(range);
        Ok(Some(range))
    }
}
