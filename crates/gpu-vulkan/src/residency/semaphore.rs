//! Ordered semaphore reductions remain device writes until normal publication.
use super::*;
use crate::cache::{
    DescriptorEntry, NativePipeline, PipelineCompiler, PipelineRecipe, ShaderArtifact,
};
use nixe_gpu::{
    ShaderStage,
    shader_module::{Binding, DescriptorKind, ResourceId},
};

pub(crate) fn semaphore_pipeline(
    compiler: &PipelineCompiler,
) -> Result<NativePipeline, VulkanError> {
    let mut stage = helper_stage(
        ShaderStage::Compute,
        include_bytes!("../../shaders/spv/semaphore-reduce.comp.spv"),
        Box::new([Binding {
            resource: ResourceId::new(0),
            binding: 0,
            kind: DescriptorKind::StorageBuffer,
            count: 1,
            sampler: None,
            readable: true,
            writable: true,
        }]),
        16,
    );
    stage.workgroup_size = Some([1, 1, 1]);
    compiler
        .request(PipelineRecipe::Compute(Arc::new(ShaderArtifact::new(
            Box::new([stage]),
        ))))
        .map_err(|e| VulkanError::Compilation(e.to_string().into()))
}

impl Residency<'_> {
    pub(crate) fn reduce_semaphore(
        &mut self,
        pipeline: &NativePipeline,
        target: &[nixe_memory::AllocationRange],
        payload: u32,
        operation: u32,
        signed: bool,
    ) -> Result<(), VulkanError> {
        if operation > 7 {
            return Err(VulkanError::InvalidResource("invalid semaphore reduction"));
        }
        self.caches.sync_guest_memory();
        let mapping = PhysicalMapping::new(target)?;
        if mapping.bytes() != 4 {
            return Err(VulkanError::InvalidResource(
                "semaphore reduction requires one word",
            ));
        }
        let range =
            self.buffer_for_deferred_import(mapping, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        if range.offset & 3 != 0 {
            return Err(VulkanError::InvalidResource("unaligned physical semaphore"));
        }
        self.ensure_buffer_range(range.buffer, range.offset, 4)?;
        let buffer = self.use_buffer(range.buffer, range.offset, 4)?;
        let alignment = self
            .device()
            .profile()
            .limits
            .min_storage_buffer_offset_alignment;
        let aligned = range.offset / alignment * alignment;
        self.runtime.device.realize_deferred_clear()?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        self.runtime
            .compute_pass_descriptor_queue
            .acquire(&mut self.runtime.device.dispatch, 1)?;
        self.runtime.compute_pass_descriptor_queue.write(
            0,
            DescriptorEntry {
                buffer: vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(aligned)
                    .range(range.offset - aligned + 4),
            },
        );
        let update = self.runtime.compute_pass_descriptor_queue.update_data();
        let push = [
            ((range.offset - aligned) / 4) as u32,
            payload,
            operation,
            u32::from(signed),
        ]
        .map(u32::to_le_bytes);
        self.runtime.device.configure_pipeline(
            pipeline,
            push.as_flattened(),
            None,
            &[0; 1][..pipeline.layout.dynamic_bindings.len()],
            Some(update),
        )?;
        unsafe {
            self.runtime.device.dispatch.cmd_dispatch(1, 1, 1);
        }
        self.finish_buffer_write(range.buffer, range.offset, 4)?;
        Ok(())
    }
}
