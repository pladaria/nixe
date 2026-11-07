//! Native initialization and operation ordering, independent of guest-byte provenance.
//! https://docs.vulkan.org/guide/latest/synchronization_examples.html
use super::*;

/// One bit per 64 bytes, as the native buffer cache's usage tracker.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/usage_tracker.h
pub(super) struct UsageTracker {
    pages: Vec<u64>,
}
impl UsageTracker {
    pub fn new(size: u64) -> Self {
        Self {
            pages: vec![0; size.div_ceil(4096) as usize],
        }
    }
    fn bounds(offset: u64, size: u64) -> (usize, usize, u64, u64) {
        let first = offset / 64;
        let last = (offset + size - 1) / 64;
        (
            (first / 64) as usize,
            (last / 64) as usize,
            u64::MAX << (first % 64),
            u64::MAX >> (63 - last % 64),
        )
    }
    pub fn used(&self, offset: u64, size: u64) -> bool {
        let (first, last, first_mask, last_mask) = Self::bounds(offset, size);
        if first == last {
            return self.pages[first] & first_mask & last_mask != 0;
        }
        self.pages[first] & first_mask != 0
            || self.pages[last] & last_mask != 0
            || self.pages[first + 1..last].iter().any(|&page| page != 0)
    }
    pub fn mark(&mut self, offset: u64, size: u64) {
        let (first, last, first_mask, last_mask) = Self::bounds(offset, size);
        if first == last {
            self.pages[first] |= first_mask & last_mask;
            return;
        }
        self.pages[first] |= first_mask;
        self.pages[last] |= last_mask;
        // Only the edge words need masks. Large vertex windows mark whole
        // interior pages in bulk, as Eden's UsageTracker::Track does.
        self.pages[first + 1..last].fill(u64::MAX);
    }
    pub fn reset(&mut self) {
        self.pages.fill(0);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AccessScope {
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}
impl AccessScope {
    pub const fn new(stages: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Self {
        Self { stages, access }
    }
    pub(super) fn buffer_alignment(self, limits: &vk::PhysicalDeviceLimits) -> u64 {
        let uniform = self.access.contains(vk::AccessFlags2::UNIFORM_READ);
        let storage = self.access.intersects(
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let mut alignment = 4;
        if uniform {
            alignment = alignment.max(limits.min_uniform_buffer_offset_alignment);
        }
        if storage || !uniform {
            alignment = alignment.max(limits.min_storage_buffer_offset_alignment);
        }
        alignment
    }
    pub fn writes(self) -> bool {
        self.access.intersects(
            vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::TRANSFER_WRITE
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::HOST_WRITE
                | vk::AccessFlags2::MEMORY_WRITE,
        )
    }
}
#[derive(Default)]
pub(super) struct Barriers {
    images: Vec<vk::ImageMemoryBarrier2<'static>>,
}
impl Barriers {
    pub fn queue_image(&mut self, barrier: vk::ImageMemoryBarrier2<'static>) {
        self.images.push(barrier);
    }
    pub(super) fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
    pub fn emit(&mut self, device: &mut VulkanDevice) -> Result<(), VulkanError> {
        // Transfers, compute, and barriers are recorded outside render passes.
        // Graphics alone may retain the scope when no dependency is pending.
        device.end_rendering()?;
        if self.is_empty() {
            return Ok(());
        }
        unsafe {
            device
                .dispatch
                .barrier(&vk::DependencyInfo::default().image_memory_barriers(&self.images));
        }
        self.images.clear();
        Ok(())
    }
}
impl Residency<'_> {
    /// WFI sets and waits for the rasterizer's event outside render passes.
    /// Pixel output is excluded, while fragment storage writes remain ordered.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L955-L975
    pub(crate) fn order_device(&mut self) -> Result<(), VulkanError> {
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let mut stages = vk::PipelineStageFlags::DRAW_INDIRECT
            | vk::PipelineStageFlags::VERTEX_INPUT
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER
            | vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER
            | vk::PipelineStageFlags::GEOMETRY_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COMPUTE_SHADER
            | vk::PipelineStageFlags::TRANSFER;
        if self.runtime.device.profile().transform_feedback {
            stages |= vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT;
        }
        unsafe {
            self.runtime
                .device
                .dispatch
                .wait_for_idle(self.runtime.device.submission.wait_for_idle_event, stages);
        }
        self.buffer_cache.accumulate_flushes();
        Ok(())
    }
    pub(super) fn materialize_buffer_literals(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        // Logical completion effects retain an exact literal and producer
        // ticket. First physical use materializes the literal in queue order;
        // subsequent partial writers therefore preserve the remaining bytes.
        // Clear the flag before upload_buffer re-enters use_buffer.
        let pending: Vec<_> = self.buffer_cache.buffers[id]
            .known
            .iter_mut()
            .filter_map(|known| {
                if !known.pending_upload
                    || known.offset >= offset + size
                    || known.offset + known.bytes.len() as u64 <= offset
                {
                    return None;
                }
                known.pending_upload = false;
                Some((known.offset, Arc::clone(&known.bytes)))
            })
            .collect();
        for (offset, bytes) in pending {
            self.upload_buffer(id, offset, &bytes)?;
        }
        Ok(())
    }

    pub(super) fn use_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<vk::Buffer, VulkanError> {
        self.check_buffer_range(id, offset, size)?;
        self.materialize_buffer_literals(id, offset, size)?;
        let work = self.runtime.device.current_work();
        let entry = &mut self.buffer_cache.buffers[id];
        let handle = entry.native.handle;
        entry.used.mark(offset, size);
        entry.contents.last_use = Some(work);
        self.touch(Resource::Buffer(id));
        Ok(handle)
    }

    /// Native image storage stays GENERAL between operations. Initialization
    /// covers the complete native allocation once, independently of CPU/GPU
    /// modification flags and the logical image-plane publication geometry.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L3154-L3181
    pub(super) fn use_image(&mut self, id: ImageId) -> Result<vk::Image, VulkanError> {
        let work = self.runtime.device.current_work();
        let entry = &mut self.texture_cache.images[id];
        if !entry.initialized {
            self.runtime.barriers.queue_image(initialize_image(
                entry.native.handle,
                native_image_aspects(entry.description.format),
                entry.description.mip_levels,
                entry.description.array_layers,
            ));
            entry.initialized = true;
        }
        entry.contents.last_use = Some(work);
        let handle = entry.native.handle;
        self.touch(Resource::Image(id));
        Ok(handle)
    }
    pub(super) fn shader_image(&mut self, id: ImageId) -> Result<vk::Image, VulkanError> {
        if self
            .runtime
            .device
            .submission
            .deferred_clear
            .as_ref()
            .is_some_and(|clear| clear.overlaps(&self.texture_cache.get(id).maps))
        {
            self.runtime.barriers.emit(&mut self.runtime.device)?;
        }
        self.use_image(id)
    }
    pub(super) fn transfer_image(
        &mut self,
        id: ImageId,
        mut range: vk::ImageSubresourceRange,
    ) -> Result<(vk::Image, vk::ImageSubresourceRange), VulkanError> {
        let description = self.texture_cache.get(id).description;
        if range.level_count == 0
            || range.layer_count == 0
            || range
                .base_mip_level
                .checked_add(range.level_count)
                .is_none_or(|end| end > description.mip_levels)
            || range
                .base_array_layer
                .checked_add(range.layer_count)
                .is_none_or(|end| end > description.array_layers)
        {
            return Err(VulkanError::InvalidResource(
                "image transfer exceeds subresources",
            ));
        }
        let aspects = native_image_aspects(description.format);
        if !self.runtime.device.profile().separate_depth_stencil_layouts
            && aspects.contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        {
            range.aspect_mask = aspects;
        }
        Ok((self.use_image(id)?, range))
    }
}

pub(super) fn native_image_aspects(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}
pub(super) fn initialize_image(
    image: vk::Image,
    aspects: vk::ImageAspectFlags,
    mips: u32,
    layers: u32,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_stage_mask(
            vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspects)
                .level_count(mips)
                .layer_count(layers),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn usage_covers_partial_blocks_and_page_edges_without_marking_neighbors() {
        const SIZE: u64 = 32771;
        let mut tracker = UsageTracker::new(SIZE);
        for offset in [0, 1, 63, 64, 65, 4095, 4096, 4097, SIZE - 1] {
            for size in [1, 2, 63, 64, 65, 4095, 4096, 8193] {
                let size = size.min(SIZE - offset);
                tracker.reset();
                tracker.mark(offset, size);
                let first_block = offset / 64;
                let last_block = (offset + size - 1) / 64;
                for query in 0..SIZE {
                    assert_eq!(
                        tracker.used(query, 1),
                        (first_block..=last_block).contains(&(query / 64)),
                        "offset={offset} size={size} query={query}"
                    );
                }
                assert!(tracker.used(offset, size));
                assert!(tracker.used(0, SIZE));
            }
        }
        tracker.reset();
        assert!(!tracker.used(0, SIZE));
        tracker.mark(0, 1);
        tracker.mark(SIZE - 1, 1);
        assert!(!tracker.used(64, SIZE / 64 * 64 - 64));
        assert!(tracker.used(64, SIZE - 64));
    }
}
