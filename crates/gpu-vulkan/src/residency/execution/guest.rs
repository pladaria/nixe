//! Guest Configure uses the cache's hardware banks and writes native descriptor
//! entries directly. Host utility captures are independent of this path.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
use super::graphics::ShaderImage;
use super::*;
use crate::cache::DescriptorEntry;
use crate::native::pipeline_resources::{
    PipelineResources, StageResources, StorageSlot, UniformSlot,
};
use crate::residency::bindings::StorageBufferBinding;

impl Residency<'_> {
    fn guest_uniform<const STAGE: usize>(
        &self,
        slot: &UniformSlot,
        group: Option<u8>,
    ) -> Option<(u64, BufferRange)> {
        let binding = if STAGE == 5 {
            self.runtime.buffer_bindings.channel.compute_uniform_buffers[slot.slot]
        } else {
            self.runtime.buffer_bindings.channel.uniform_buffers
                [group.expect("active uniform stage") as usize][slot.slot]
        };
        let binding = binding?;
        let mut range = binding.native?;
        range.size = range.size.min(u64::from(slot.size));
        Some((binding.source.address, range))
    }
    fn guest_storage(&self, stage: usize, slot: &StorageSlot) -> Option<StorageBufferBinding> {
        self.runtime.buffer_bindings.channel.storage_buffers[stage][slot.slot]
            .as_ref()
            .and_then(|s| s.native.clone())
    }
    pub(crate) fn begin_guest_descriptors(
        &mut self,
        pipeline: &NativePipeline,
    ) -> Result<(), VulkanError> {
        self.runtime.descriptor_queue.acquire(
            &mut self.runtime.device.dispatch,
            pipeline.layout.key.bindings.len(),
        )?;
        self.runtime.buffer_bindings.channel.streamed_uniforms = [0; 6];
        Ok(())
    }
    pub(crate) fn import_guest_stage_storage<const STAGE: usize>(
        &mut self,
        resources: &StageResources,
    ) -> Result<(), VulkanError> {
        // Finish imports before registering scopes: importing an overlapping
        // physical resource may replace a cache entry's prior reader scopes.
        for slot in &resources.storage {
            match self.guest_storage(STAGE, slot) {
                Some(StorageBufferBinding::Buffer(range)) => {
                    self.ensure_buffer_range(range.buffer, range.offset, range.size)?
                }
                Some(StorageBufferBinding::PointerTable(binding)) => {
                    for range in &binding.targets {
                        self.ensure_buffer_range(range.buffer, range.offset, range.size)?;
                    }
                }
                _ => (),
            }
        }
        Ok(())
    }
    pub(crate) fn configure_guest_stage<const STAGE: usize, const STORAGE: bool>(
        &mut self,
        pipeline: &NativePipeline,
        resources: &StageResources,
        group: Option<u8>,
        images: &[ShaderImage],
    ) -> Result<(), VulkanError> {
        for slot in &resources.uniforms {
            let Some((address, range)) = self.guest_uniform::<STAGE>(slot, group) else {
                self.buffer_cache.uniform_streams.forget::<STAGE>(slot.slot);
                if !self.runtime.device.profile().null_descriptor {
                    return Err(VulkanError::InvalidResource(
                        "disabled guest uniform requires nullDescriptor support",
                    ));
                }
                self.runtime.descriptor_queue.write(
                    slot.descriptor,
                    DescriptorEntry {
                        buffer: vk::DescriptorBufferInfo::default().range(vk::WHOLE_SIZE),
                    },
                );
                continue;
            };
            let scope = pipeline.resources[slot.descriptor];
            if range.size == 0
                || range.size
                    > u64::from(
                        self.runtime
                            .device
                            .profile()
                            .limits
                            .max_uniform_buffer_range,
                    )
            {
                return Err(VulkanError::InvalidResource(
                    "guest uniform descriptor exceeds maxUniformBufferRange",
                ));
            }
            let alignment = scope.buffer_alignment(&self.runtime.device.profile().limits);
            let unaligned = !range.offset.is_multiple_of(alignment);
            self.runtime.buffer_bindings.channel.uniform_cache_shots[0] += 1;
            // BindHostGraphicsUniformBuffer consumes the hardware bank's DAddr
            // directly. Guest uniforms never reclassify physical footprints or
            // fall through the host utility program's generic stream selector.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/buffer_cache/buffer_cache.h#L931-L980
            let native =
                if let Some(native) = self.kept_uniform::<STAGE>(slot.slot, address, range.size) {
                    self.touch(Resource::Buffer(range.buffer));
                    self.runtime.buffer_bindings.channel.streamed_uniforms[STAGE] |= 1 << slot.slot;
                    (native.buffer, native.offset)
                } else if (unaligned
                    || range.size <= self.runtime.buffer_bindings.channel.uniform_stream_limit)
                    && !self
                        .buffer_cache
                        .gpu_modified_ranges
                        .intersects(address, range.size)
                {
                    let native = self.stream_guest_uniform::<STAGE>(
                        slot.slot,
                        address,
                        range,
                        alignment.max(4),
                    )?;
                    self.touch(Resource::Buffer(range.buffer));
                    self.runtime.buffer_bindings.channel.streamed_uniforms[STAGE] |= 1 << slot.slot;
                    (native.buffer, native.offset)
                } else if unaligned {
                    self.runtime.buffer_bindings.channel.streamed_uniforms[STAGE] |= 1 << slot.slot;
                    let copy = self.aligned_uniform_copy(range.buffer, range.offset, range.size)?;
                    (copy.buffer, copy.offset)
                } else {
                    let uploaded = self.runtime.statistics.cpu_uploaded_bytes;
                    self.ensure_buffer_range(range.buffer, range.offset, range.size)?;
                    self.use_buffer(range.buffer, range.offset, range.size)?;
                    if self.runtime.statistics.cpu_uploaded_bytes == uploaded {
                        self.runtime.buffer_bindings.channel.uniform_cache_hits[0] += 1;
                    }
                    (
                        self.buffer_cache.get(range.buffer).native.handle,
                        range.offset,
                    )
                };
            self.runtime.descriptor_queue.write(
                slot.descriptor,
                DescriptorEntry {
                    buffer: vk::DescriptorBufferInfo::default()
                        .buffer(native.0)
                        .offset(native.1)
                        .range(range.size),
                },
            );
        }
        if STORAGE {
            for slot in &resources.storage {
                match self.guest_storage(STAGE, slot) {
                    Some(StorageBufferBinding::Buffer(range)) => {
                        let limits = &self.runtime.device.profile().limits;
                        if range.size == 0
                            || range.size > u64::from(limits.max_storage_buffer_range)
                            || !range
                                .offset
                                .is_multiple_of(limits.min_storage_buffer_offset_alignment.max(1))
                        {
                            return Err(VulkanError::InvalidResource(
                                "guest storage descriptor exceeds host size/alignment support",
                            ));
                        }
                        self.use_buffer(range.buffer, range.offset, range.size)?;
                        self.runtime.descriptor_queue.write(
                            slot.descriptor,
                            DescriptorEntry {
                                buffer: vk::DescriptorBufferInfo::default()
                                    .buffer(self.buffer_cache.get(range.buffer).native.handle)
                                    .offset(range.offset)
                                    .range(range.size),
                            },
                        );
                    }
                    Some(StorageBufferBinding::PointerTable(binding)) => {
                        for range in &binding.targets {
                            self.use_buffer(range.buffer, range.offset, range.size)?;
                        }
                        self.runtime.descriptor_queue.write(
                            slot.descriptor,
                            DescriptorEntry {
                                buffer: vk::DescriptorBufferInfo::default()
                                    .buffer(binding.table.handle)
                                    .offset(binding.offset)
                                    .range(binding.size),
                            },
                        );
                    }
                    Some(StorageBufferBinding::MultiRange(binding)) => {
                        if binding.sparse {
                            for &source in binding.sources.iter() {
                                self.use_buffer(source.buffer, source.offset, source.size)?;
                            }
                        }
                        self.runtime.descriptor_queue.write(
                            slot.descriptor,
                            DescriptorEntry {
                                buffer: vk::DescriptorBufferInfo::default()
                                    .buffer(binding.native.handle)
                                    .range(binding.native.size()),
                            },
                        );
                    }
                    None if self.runtime.device.profile().null_descriptor => {
                        self.runtime.descriptor_queue.write(
                            slot.descriptor,
                            DescriptorEntry {
                                buffer: vk::DescriptorBufferInfo::default().range(vk::WHOLE_SIZE),
                            },
                        );
                    }
                    None => {
                        return Err(VulkanError::InvalidResource(
                            "null guest storage requires nullDescriptor support",
                        ));
                    }
                }
            }
        }
        for image in images {
            if let Some(view) = image.view {
                let id = self.texture_cache.views[view].image;
                let scope = pipeline.resources[image.descriptor];
                if scope.writes() {
                    self.mark_image_modified(id)?;
                }
                self.shader_image(id)?;
            } else if !self.runtime.device.profile().null_descriptor {
                return Err(VulkanError::InvalidResource(
                    "null guest image requires nullDescriptor support",
                ));
            };
            self.runtime.descriptor_queue.write(
                image.descriptor,
                DescriptorEntry {
                    image: image.native,
                },
            );
        }
        Ok(())
    }
    pub(crate) fn guest_descriptor_update(&self) -> DescriptorQueue {
        DescriptorQueue {
            dynamic_offsets: None,
            record: None,
            update: Some(self.runtime.descriptor_queue.update_data()),
        }
    }
    pub(crate) fn finish_guest_writes(
        &mut self,
        pipeline: &NativePipeline,
        resources: &PipelineResources,
    ) -> Result<(), VulkanError> {
        if !pipeline.shader_writes {
            return Ok(());
        }
        for (stage, resources) in resources.stages.iter().enumerate() {
            for slot in &resources.storage {
                let scope = pipeline.resources[slot.descriptor];
                match self.guest_storage(stage, slot) {
                    Some(StorageBufferBinding::Buffer(range)) => {
                        if scope.writes() {
                            self.finish_buffer_write(range.buffer, range.offset, range.size)?;
                        }
                    }
                    Some(StorageBufferBinding::PointerTable(binding)) if scope.writes() => {
                        for range in &binding.targets {
                            self.finish_buffer_write(range.buffer, range.offset, range.size)?;
                        }
                    }
                    Some(StorageBufferBinding::MultiRange(binding)) if binding.sparse => {
                        for source in binding.sources.iter() {
                            if scope.writes() {
                                self.finish_buffer_write(
                                    source.buffer,
                                    source.offset,
                                    source.size,
                                )?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    pub(crate) fn update_compute_buffers(
        &mut self,
        resources: &StageResources,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
    ) -> Result<(), VulkanError> {
        self.invalidate_device_mappings()?;
        loop {
            self.runtime.buffer_bindings.channel.has_deleted_buffers = false;
            for uniform in &resources.uniforms {
                self.compute_uniform(uniform.slot)?;
            }
            for storage in &resources.storage {
                self.update_storage_buffer(5, storage.slot, space)?;
            }
            if !self.runtime.buffer_bindings.channel.has_deleted_buffers {
                return Ok(());
            }
        }
    }
    pub(crate) fn dispatch_guest(
        &mut self,
        pipeline: &NativePipeline,
        resources: &PipelineResources,
        images: &[ShaderImage],
        groups: [u32; 3],
        push: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        if groups
            .iter()
            .zip(
                self.runtime
                    .device
                    .profile()
                    .limits
                    .max_compute_work_group_count,
            )
            .any(|(&n, max)| n > max)
        {
            return Err(VulkanError::InvalidResource(
                "dispatch exceeds maxComputeWorkGroupCount",
            ));
        }
        if groups.contains(&0) {
            return Ok(self.runtime.device.current_work());
        }
        self.begin_guest_descriptors(pipeline)?;
        self.import_guest_stage_storage::<5>(&resources.stages[5])?;
        self.configure_guest_stage::<5, true>(pipeline, &resources.stages[5], None, images)?;
        let descriptors = self.guest_descriptor_update();
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        self.runtime
            .device
            .configure_pipeline(pipeline, push, None, &[], descriptors.update)?;
        unsafe {
            self.runtime
                .device
                .dispatch
                .cmd_dispatch(groups[0], groups[1], groups[2]);
        }
        let work = self.runtime.device.current_work();
        self.buffer_cache.collect_command_copies = true;
        let writes = self.finish_guest_writes(pipeline, resources);
        self.buffer_cache.collect_command_copies = false;
        writes?;
        self.copy_command_pages()?;
        Ok(work)
    }
}
