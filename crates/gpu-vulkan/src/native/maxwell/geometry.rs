//! BufferCache consumes hardware geometry directly into its current native slots.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L1279-L1419
use super::*;
use crate::VulkanError as V;
use crate::residency::execution::{IndexBuffer, VertexBuffer};
use crate::residency::{BufferRange, Residency};
use ash::vk;
use nixe_gpu::native::GraphicsDirty;
impl Residency<'_> {
    fn geometry_buffer(
        &mut self,
        space: &MaxwellGpuMemory,
        begin: u64,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Option<BufferRange>, V> {
        if size == 0 {
            return Ok(None);
        }
        let address = space
            .address(begin)
            .map_err(|e| V::Compilation(e.to_string().into()))?;
        let device = match space.device_address(address) {
            Ok(address) => address,
            Err(model::MaxwellGpuAccessError::UnmappedAddress { .. }) => return Ok(None),
            Err(e) => return Err(V::Compilation(e.to_string().into())),
        };
        self.device_buffer(device, size, usage, false, false)
            .map(Some)
    }
    fn update_vertex_slot(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        index: usize,
    ) -> Result<Option<VertexBuffer>, V> {
        let vertex = state.vertex_input().streams(index);
        if !vertex
            .format()
            .value()
            .is_some_and(|format| format.enabled())
        {
            return Ok(None);
        }
        let (Some(base), Some(limit)) = (vertex.address(), vertex.limit()) else {
            return Ok(None);
        };
        let end = limit.get() + 1;
        let mut size = u64::from(end.wrapping_sub(base.get()) as u32);
        if size >= 64 * 1024 * 1024 || space.address(end).is_err() {
            let address = space
                .address(base.get())
                .map_err(|e| V::Compilation(e.to_string().into()))?;
            size = space
                .max_continuous_range(address, size)
                .map_err(|e| V::Compilation(e.to_string().into()))?;
        }
        Ok(self
            .geometry_buffer(space, base.get(), size, vk::BufferUsageFlags::VERTEX_BUFFER)?
            .map(|buffer| VertexBuffer {
                binding: index as u32,
                buffer: buffer.buffer,
                offset: buffer.offset,
                size: buffer.size,
            }))
    }
    fn update_index_slot(
        &mut self,
        draw: &MaxwellThreeDDrawState,
        space: &MaxwellGpuMemory,
    ) -> Result<Option<IndexBuffer>, V> {
        if !draw.indexed || draw.inline_indices().is_some() {
            return Ok(None);
        }
        let index = draw.index;
        let (bytes, kind) = match index.format {
            0 => (1, vk::IndexType::UINT8_EXT),
            1 => (2, vk::IndexType::UINT16),
            2 => (4, vk::IndexType::UINT32),
            value => {
                return Err(registers::unsupported(
                    0x17d8,
                    value,
                    "unsupported index format",
                ));
            }
        };
        let address_size = u64::from(index.limit.wrapping_sub(index.address) as u32);
        let draw_size = u64::from(
            index
                .first
                .checked_add(index.count)
                .ok_or(V::InvalidResource("index count overflows"))?,
        ) * bytes;
        let size = address_size.min(draw_size);
        Ok(self
            .geometry_buffer(
                space,
                index.address,
                size,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?
            .map(|buffer| IndexBuffer {
                buffer: buffer.buffer,
                offset: buffer.offset,
                size: buffer.size,
                kind,
            }))
    }
    pub(in crate::native) fn update_geometry(
        &mut self,
        state: &MaxwellThreeDState,
        draw: &MaxwellThreeDDrawState,
        space: &MaxwellGpuMemory,
    ) -> Result<bool, V> {
        self.invalidate_device_mappings()?;
        let indexed = draw.indexed;
        let mut changed = self.runtime.buffer_bindings.channel.indexed != indexed;
        self.runtime.buffer_bindings.channel.indexed = indexed;
        let generation = space.mapping_generation().get();
        if self.runtime.buffer_bindings.channel.address_space != Some(space.id().get())
            || self.runtime.buffer_bindings.channel.mapping_generation != generation
        {
            self.runtime.buffer_bindings.channel.invalidated_buffers |= GraphicsDirty::BUFFERS;
            self.runtime.buffer_bindings.channel.address_space = Some(space.id().get());
            self.runtime.buffer_bindings.channel.mapping_generation = generation;
        }
        loop {
            let mask = GraphicsDirty::VERTICES | if indexed { GraphicsDirty::INDEX } else { 0 };
            let bank = &mut self.runtime.buffer_bindings.channel;
            let dirty = state.graphics_dirty().take(mask) | (bank.invalidated_buffers & mask);
            bank.invalidated_buffers &= !mask;
            if dirty == 0 {
                break;
            }
            let mut vertices = (dirty >> GraphicsDirty::VERTEX_SHIFT) as u32;
            while vertices != 0 {
                let slot = vertices.trailing_zeros() as usize;
                vertices &= vertices - 1;
                let next = self.update_vertex_slot(state, space, slot)?;
                changed |= self.runtime.buffer_bindings.channel.vertex_buffers[slot] != next;
                self.bind_vertex_buffer(slot as u32, next)?;
            }
            if dirty & GraphicsDirty::INDEX != 0 {
                let next = self.update_index_slot(draw, space)?;
                changed |= self.runtime.buffer_bindings.channel.index_buffer != next;
                self.bind_index_buffer(next)?;
            }
        }
        Ok(changed)
    }
}
