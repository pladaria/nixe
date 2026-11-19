//! Resolve the two counter words at command issue. Keep device-produced operands on the
//! graphics queue, with a stable result until the next predicate for that NVN queue.
use super::*;
use crate::cache::{
    DescriptorEntry, NativePipeline, PipelineCompiler, PipelineRecipe, ShaderArtifact,
};
use nixe_gpu::{
    ShaderStage,
    shader_module::{Binding, DescriptorKind, ResourceId},
};

#[derive(Default)]
pub(crate) struct RenderConditions {
    queues: HashMap<u64, QueueCondition>,
    timestamps: HashMap<u64, u64>,
    pipeline: Option<NativePipeline>,
}

#[derive(Default)]
struct QueueCondition {
    skip: bool,
    active: bool,
    buffer: Option<Arc<AllocatedBuffer>>,
}

impl RenderConditions {
    pub(crate) fn remove(&mut self, queue: u64) {
        self.queues.remove(&queue);
    }
    pub(crate) fn begin_list(&mut self) {
        self.timestamps.clear();
    }

    pub(crate) fn timestamp(
        &mut self,
        memory: &nixe_memory::device_memory::DeviceMemory,
        address: u64,
    ) {
        // A timestamp report stores a zero counter followed by the clock. Physical
        // page versions invalidate forwarding on CPU/GPU overwrites through any alias.
        if address & 7 == 0
            && memory.register_command_pages(address, 8)
            && let Some(at) = memory.physical_address(address)
        {
            self.timestamps
                .insert(at.get(), memory.command_version(at.get()));
        }
    }

    pub(crate) fn draw(&self, queue: u64) -> Option<Option<Arc<AllocatedBuffer>>> {
        match self.queues.get(&queue) {
            Some(q) if q.skip => None,
            Some(q) if q.active => Some(q.buffer.clone()),
            _ => Some(None),
        }
    }
}

enum Operand {
    Known(u64),
    DeviceBuffer,
    OtherProducer,
}
impl Residency<'_> {
    fn counter_word(
        &self,
        conditions: &RenderConditions,
        address: u64,
    ) -> Result<Operand, VulkanError> {
        let memory = &self.runtime.device_memory;
        if address & 7 == 0
            && let Some(at) = memory.physical_address(address)
            && conditions
                .timestamps
                .get(&at.get())
                .is_some_and(|&v| v == memory.command_version(at.get()))
        {
            return Ok(Operand::Known(0));
        }
        let native = |at, bytes| {
            self.buffer_cache.gpu_modified_ranges.intersects(at, bytes)
                || self.buffer_cache.pending_read(at, bytes).is_some()
        };
        let other = |at, bytes| {
            self.texture_cache.region_gpu_modified(at, bytes)
                || (self.caches.query_pages.any(at, bytes)
                    && self
                        .caches
                        .query_cache
                        .lock()
                        .unwrap()
                        .must_flush_region(at, bytes))
        };
        let mut foreign = other(address, 8);
        memory.apply_device_aliases(address, 8, |at, bytes| {
            foreign |= native(at, bytes) || other(at, bytes)
        });
        if foreign {
            return Ok(Operand::OtherProducer);
        }
        if native(address, 8) {
            return Ok(Operand::DeviceBuffer);
        }
        let mut value = [0; 8];
        self.runtime
            .device_reader
            .read_block_unsafe(address, &mut value)?;
        Ok(Operand::Known(u64::from_le_bytes(value)))
    }

    pub(crate) fn set_nvn_condition(
        &mut self,
        conditions: &mut RenderConditions,
        compiler: &PipelineCompiler,
        queue: u64,
        mode: u32,
        address: u64,
    ) -> Result<(), VulkanError> {
        let enabled = match mode {
            2 => Some(true),
            3 => Some(false),
            0 | 1 => {
                let other = address
                    .checked_add(16)
                    .filter(|&at| at <= (1u64 << self.device_memory().address_bits()) - 8)
                    .ok_or(VulkanError::InvalidResource(
                        "invalid NVN condition address",
                    ))?;
                match (
                    self.counter_word(conditions, address)?,
                    self.counter_word(conditions, other)?,
                ) {
                    (Operand::Known(a), Operand::Known(b)) => Some((a == b) == (mode == 0)),
                    (Operand::DeviceBuffer, Operand::Known(_) | Operand::DeviceBuffer)
                    | (Operand::Known(_), Operand::DeviceBuffer)
                        if self.device().profile().conditional_rendering
                            && address.is_multiple_of(4) =>
                    {
                        None
                    }
                    _ => {
                        let mut words = [0u64; 3];
                        self.flush_read_span(address, 24)?;
                        self.runtime
                            .device_reader
                            .read_block_unsafe(address, bytemuck::bytes_of_mut(&mut words))?;
                        Some((words[0] == words[2]) == (mode == 0))
                    }
                }
            }
            _ => {
                return Err(VulkanError::InvalidResource(
                    "unknown conditional render mode",
                ));
            }
        };
        let q = conditions.queues.entry(queue).or_default();
        q.skip = enabled == Some(false);
        q.active = enabled.is_none();
        if q.active {
            if conditions.pipeline.is_none() {
                let mut stage = super::conversion::stage(
                    ShaderStage::Compute,
                    include_bytes!("../../shaders/spv/nvn-predicate.comp.spv"),
                    [false, true]
                        .into_iter()
                        .enumerate()
                        .map(|(i, writable)| Binding {
                            resource: ResourceId::new(i as u32),
                            binding: i as u32,
                            kind: DescriptorKind::StorageBuffer,
                            count: 1,
                            sampler: None,
                            readable: !writable,
                            writable,
                        })
                        .collect(),
                    8,
                );
                stage.workgroup_size = Some([1, 1, 1]);
                conditions.pipeline = Some(
                    compiler
                        .request(PipelineRecipe::Compute(Arc::new(ShaderArtifact::new(
                            Box::new([stage]),
                        ))))
                        .map_err(|e| VulkanError::Compilation(e.to_string().into()))?,
                );
            }
            if q.buffer.is_none() {
                q.buffer = Some(Arc::new(self.runtime.device.allocate_buffer(
                    4,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT,
                    MemoryLocation::GpuOnly,
                    AllocationMode::Managed,
                    "NVN render predicate",
                )?));
            }
            self.resolve_predicate(
                conditions.pipeline.as_ref().unwrap(),
                q.buffer.as_ref().unwrap(),
                address,
                mode,
            )?;
        }
        Ok(())
    }

    fn resolve_predicate(
        &mut self,
        pipeline: &NativePipeline,
        result: &Arc<AllocatedBuffer>,
        address: u64,
        invert: u32,
    ) -> Result<(), VulkanError> {
        let r = self.device_buffer(
            address,
            24,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
            false,
        )?;
        let source = self.use_buffer(r.buffer, r.offset, 24)?;
        let align = self
            .device()
            .profile()
            .limits
            .min_storage_buffer_offset_alignment;
        let base = r.offset / align * align;
        let offset = r.offset - base;
        let device = &mut self.runtime.device;
        device.realize_deferred_clear()?;
        self.runtime.barriers.emit(device)?;
        device.end_rendering()?;
        let cs = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let cond = vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT;
        let read = vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT;
        let write = vk::AccessFlags2::SHADER_STORAGE_WRITE;
        let scope = |buffer, offset, size, src, access, dst, next| {
            vk::BufferMemoryBarrier2::default()
                .buffer(buffer)
                .offset(offset)
                .size(size)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_stage_mask(src)
                .src_access_mask(access)
                .dst_stage_mask(dst)
                .dst_access_mask(next)
        };
        let all = vk::PipelineStageFlags2::ALL_COMMANDS;
        let writes = vk::AccessFlags2::MEMORY_WRITE;
        let load = vk::AccessFlags2::SHADER_STORAGE_READ;
        // Snapshot both operands at this command, before a later report or draw
        // overwrites them. Reusing the result also orders earlier predicated draws.
        unsafe {
            device
                .dispatch
                .barrier(&vk::DependencyInfo::default().buffer_memory_barriers(&[
                    scope(source, r.offset, 24, all, writes, cs, load),
                    scope(result.handle, 0, 4, cond | cs, read | write, cs, write),
                ]));
        }
        let descriptors = &mut self.runtime.compute_pass_descriptor_queue;
        descriptors.acquire(&mut device.dispatch, 2)?;
        for (i, info) in [(source, base, offset + 24), (result.handle, 0, 4)]
            .into_iter()
            .enumerate()
        {
            descriptors.write(
                i,
                DescriptorEntry {
                    buffer: vk::DescriptorBufferInfo::default()
                        .buffer(info.0)
                        .offset(info.1)
                        .range(info.2),
                },
            );
        }
        let update = descriptors.update_data();
        device.configure_pipeline(
            pipeline,
            bytemuck::bytes_of(&[(offset / 4) as u32, invert]),
            None,
            &[0; 2][..pipeline.layout.dynamic_bindings.len()],
            Some(update),
        )?;
        unsafe {
            device.submission.retain_predicate(result);
            device.dispatch.cmd_dispatch(1, 1, 1);
            device
                .dispatch
                .barrier(&vk::DependencyInfo::default().buffer_memory_barriers(&[
                    scope(result.handle, 0, 4, cs, write, cond, read),
                    scope(source, r.offset, 24, cs, load, all, writes),
                ]));
        }
        Ok(())
    }
}
