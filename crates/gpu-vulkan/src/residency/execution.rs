//! Prepared native execution. Descriptor reflection supplies the access scopes;
//! callers cannot omit shader writes from canonical ownership or synchronization.
use super::*;
use crate::cache::{
    BindingRecord, DescriptorCache, DescriptorResource, GraphicsState, NativePipeline, Sampler,
    ShaderArtifact,
};
mod graphics;
mod state;
pub use graphics::{
    AttachmentSource, Draw, IndexBuffer, PreparedDraw, PreparedTargets, RenderAttachment,
    RenderingTargets, VertexBuffer,
};
pub use state::{GraphicsParameters, PreparedGraphics};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShaderAccess {
    pub read: vk::PipelineStageFlags2,
    pub write: vk::PipelineStageFlags2,
}
pub(crate) fn shader_accesses(
    shader: &ShaderArtifact,
    layout: &crate::cache::LayoutKey,
) -> Box<[ShaderAccess]> {
    layout
        .bindings
        .iter()
        .map(|b| {
            let mut access = ShaderAccess::default();
            for stage in &shader.stages {
                let flags = shader_stage(stage.stage);
                for binding in stage
                    .module
                    .bindings
                    .iter()
                    .filter(|r| r.binding == b.binding)
                {
                    if binding.readable {
                        access.read |= flags;
                    }
                    if binding.writable {
                        access.write |= flags;
                    }
                }
            }
            access
        })
        .collect()
}
fn shader_stage(stage: nixe_gpu::ShaderStage) -> vk::PipelineStageFlags2 {
    use nixe_gpu::ShaderStage;
    match stage {
        ShaderStage::Vertex => vk::PipelineStageFlags2::VERTEX_SHADER,
        ShaderStage::TessellationControl => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
        ShaderStage::TessellationEvaluation => {
            vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER
        }
        ShaderStage::Geometry => vk::PipelineStageFlags2::GEOMETRY_SHADER,
        ShaderStage::Fragment => vk::PipelineStageFlags2::FRAGMENT_SHADER,
        ShaderStage::Compute => vk::PipelineStageFlags2::COMPUTE_SHADER,
    }
}
#[derive(Clone)]
pub enum Binding {
    Buffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    Image(ImageViewId),
    Sampler(Arc<Sampler>),
}
#[derive(Clone, Copy)]
pub(super) enum Use {
    Buffer {
        id: BufferId,
        offset: u64,
        size: u64,
        scope: AccessScope,
    },
    Image {
        id: ImageId,
        plane: ImagePlane,
        scope: AccessScope,
    },
}
/// Build once when the pipeline or bindings change. Resources and the immutable
/// descriptor set stay leased until the prepared state and its GPU uses retire.
pub struct PreparedProgram {
    pub(super) pipeline: Arc<NativePipeline>,
    record: Option<Arc<BindingRecord>>,
    pub(super) uses: Box<[Use]>,
}
#[derive(Clone, Copy, Debug)]
pub enum Dispatch {
    Direct([u32; 3]),
    Indirect { buffer: BufferId, offset: u64 },
}
impl VulkanResidency {
    /// Host-only sampled resources are synchronized by their concrete owner;
    /// they have no invented canonical guest-memory access declarations.
    pub(crate) fn prepare_host_graphics(
        &self,
        pipeline: Arc<NativePipeline>,
        record: Arc<BindingRecord>,
        parameters: GraphicsParameters,
    ) -> Result<PreparedGraphics, VulkanError> {
        self.prepare_graphics(
            PreparedProgram {
                pipeline,
                record: Some(record),
                uses: Box::new([]),
            },
            parameters,
        )
    }
    pub(crate) fn bind_host_graphics(
        &mut self,
        graphics: &PreparedGraphics,
        push: &[u8],
    ) -> Result<(), VulkanError> {
        self.graphics_state(graphics)?;
        self.program_bind(&graphics.program, push).map(|_| ())
    }
    pub fn prepare_program(
        &self,
        pipeline: Arc<NativePipeline>,
        bindings: &[Binding],
        descriptors: &mut DescriptorCache,
    ) -> Result<PreparedProgram, VulkanError> {
        if !Arc::ptr_eq(&pipeline.owner, &self.device.owner)
            || bindings.len() != pipeline.layout.key.bindings.len()
        {
            return Err(VulkanError::InvalidResource(
                "prepared program has a foreign pipeline or incorrect binding count",
            ));
        }
        let mut resources = Vec::with_capacity(bindings.len());
        let mut uses = Vec::<Use>::new();
        for ((binding, reflected), access) in bindings
            .iter()
            .zip(&pipeline.layout.key.bindings)
            .zip(&pipeline.resources)
        {
            let stages = access.read | access.write;
            let mut flags = vk::AccessFlags2::empty();
            if !access.read.is_empty() {
                flags |= if reflected.kind == vk::DescriptorType::SAMPLED_IMAGE {
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                } else {
                    vk::AccessFlags2::SHADER_STORAGE_READ
                };
            }
            if !access.write.is_empty() {
                flags |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
            }
            let scope = AccessScope::new(stages, flags);
            match binding {
                Binding::Buffer {
                    buffer,
                    offset,
                    size,
                } => {
                    self.check_buffer_range(*buffer, *offset, *size)?;
                    resources.push(DescriptorResource::Buffer {
                        buffer: self.buffer_lease(*buffer)?,
                        offset: *offset,
                        range: *size,
                    });
                    if !stages.is_empty() {
                        uses.push(Use::Buffer {
                            id: *buffer,
                            offset: *offset,
                            size: *size,
                            scope,
                        });
                    }
                }
                Binding::Image(view) => {
                    let image = self.image(view.image)?;
                    let description = image
                        .views
                        .get(view.view)
                        .ok_or(VulkanError::InvalidResource("absent prepared image view"))?;
                    if reflected.kind == vk::DescriptorType::STORAGE_IMAGE
                        && description.swizzle != [vk::ComponentSwizzle::IDENTITY; 4]
                    {
                        return Err(VulkanError::InvalidResource(
                            "storage image descriptors require identity component mapping",
                        ));
                    }
                    if description.aspects
                        == vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                    {
                        return Err(VulkanError::InvalidResource(
                            "shader descriptors require an individual depth or stencil view aspect",
                        ));
                    }
                    for mip in description.base_mip..description.base_mip + description.mip_count {
                        for layer in
                            description.base_layer..description.base_layer + description.layer_count
                        {
                            for aspect in [
                                vk::ImageAspectFlags::COLOR,
                                vk::ImageAspectFlags::DEPTH,
                                vk::ImageAspectFlags::STENCIL,
                            ] {
                                if description.aspects.contains(aspect)
                                    && !image.planes.iter().any(|p| {
                                        p.aspect == aspect
                                            && p.mip_level == mip
                                            && p.array_layer == layer
                                    })
                                {
                                    return Err(VulkanError::InvalidResource(
                                        "sampled/storage view consumes a subresource without an initialized native plane",
                                    ));
                                }
                            }
                        }
                    }
                    let usage = if reflected.kind == vk::DescriptorType::STORAGE_IMAGE {
                        vk::ImageUsageFlags::STORAGE
                    } else {
                        vk::ImageUsageFlags::SAMPLED
                    };
                    if !image.description.usage.contains(usage) {
                        return Err(VulkanError::InvalidResource(
                            "shader image requires sampled/storage usage",
                        ));
                    }
                    // GENERAL is valid for sampled and storage descriptors. Selecting
                    // it here keeps a prepared descriptor immutable across hazards.
                    resources.push(DescriptorResource::Image {
                        view: self.image_view_lease(*view)?,
                        layout: vk::ImageLayout::GENERAL,
                    });
                    for &plane in &image.planes {
                        if description.aspects.intersects(plane.aspect)
                            && (description.base_mip..description.base_mip + description.mip_count)
                                .contains(&plane.mip_level)
                            && (description.base_layer
                                ..description.base_layer + description.layer_count)
                                .contains(&plane.array_layer)
                            && !stages.is_empty()
                        {
                            uses.push(Use::Image {
                                id: view.image,
                                plane,
                                scope,
                            });
                        }
                    }
                }
                Binding::Sampler(sampler) => {
                    resources.push(DescriptorResource::Sampler(Arc::clone(sampler)))
                }
            }
        }
        // Union simultaneous accesses, rather than generating false intra-command
        // dependencies for aliased bindings. Buffer windows are merged only when
        // they overlap; independent ranges retain independent hazards.
        let mut merged = Vec::<Use>::new();
        for mut next in uses {
            let mut i = 0;
            while i < merged.len() {
                match (next, merged[i]) {
                    (
                        Use::Buffer {
                            id,
                            offset,
                            size,
                            mut scope,
                        },
                        Use::Buffer {
                            id: old,
                            offset: start,
                            size: length,
                            scope: previous,
                        },
                    ) if id == old && offset <= start + length && start <= offset + size => {
                        scope.stages |= previous.stages;
                        scope.access |= previous.access;
                        let end = (offset + size).max(start + length);
                        next = Use::Buffer {
                            id,
                            offset: offset.min(start),
                            size: end - offset.min(start),
                            scope,
                        };
                        merged.swap_remove(i);
                        i = 0;
                    }
                    (
                        Use::Image {
                            id,
                            plane,
                            mut scope,
                        },
                        Use::Image {
                            id: old,
                            plane: p,
                            scope: previous,
                        },
                    ) if id == old && plane == p => {
                        scope.stages |= previous.stages;
                        scope.access |= previous.access;
                        next = Use::Image { id, plane, scope };
                        merged.swap_remove(i);
                        i = 0;
                    }
                    _ => i += 1,
                }
            }
            merged.push(next);
        }
        let record = if resources.is_empty() {
            None
        } else {
            Some(descriptors.get(Arc::clone(&pipeline.layout), resources.into_boxed_slice())?)
        };
        Ok(PreparedProgram {
            pipeline,
            record,
            uses: merged.into_boxed_slice(),
        })
    }
    pub(super) fn program_access(&mut self, program: &PreparedProgram) -> Result<(), VulkanError> {
        for &resource in &program.uses {
            match resource {
                Use::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } => {
                    self.ensure_buffer(id)?;
                    // Preserve bytes not actually written by a conditional shader
                    // store, even in a descriptor classified as write-only.
                    if scope.writes() {
                        self.prepare_write(Resource::Buffer(id), offset, size)?;
                    }
                    self.buffer_access_range(id, offset, size, scope)?;
                }
                Use::Image { id, plane, scope } => {
                    self.ensure_image_initialized(id, plane, transfers::full_rect(plane))?;
                    if scope.writes() {
                        self.prepare_image_rect(id, plane, transfers::full_rect(plane))?;
                    }
                    self.image_access_subresources(
                        id,
                        commands::plane_range(plane),
                        scope,
                        vk::ImageLayout::GENERAL,
                    )?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn program_written(&mut self, program: &PreparedProgram) -> Result<(), VulkanError> {
        for &resource in &program.uses {
            match resource {
                Use::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } if scope.writes() => self.record_authority(Resource::Buffer(id), offset, size)?,
                Use::Image { id, plane, scope } if scope.writes() => {
                    self.finish_image_rect(id, plane, transfers::full_rect(plane))?
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub(super) fn program_bind(
        &mut self,
        program: &PreparedProgram,
        push: &[u8],
    ) -> Result<vk::CommandBuffer, VulkanError> {
        if push.len() < program.pipeline.layout.key.push_bytes as usize
            || !push.len().is_multiple_of(4)
            || push.len() > self.device.profile().limits.max_push_constants_size as usize
        {
            return Err(VulkanError::InvalidResource(
                "push constants must cover the compiled ABI within host size/alignment limits",
            ));
        }
        self.device.bind_native_pipeline(&program.pipeline)?;
        if let Some(record) = &program.record {
            self.device
                .bind_native_descriptors(&program.pipeline, record)?;
        }
        let index = self.device.submission.begin()?;
        let command = self.device.submission.slots[index].command;
        let slot = &mut self.device.submission.slots[index];
        let layout = program.pipeline.layout.handle;
        let stages = program.pipeline.layout.key.push_stages;
        let push = &push[..program.pipeline.layout.key.push_bytes as usize];
        if !push.is_empty()
            && (slot.push_layout != layout || slot.push_stages != stages || slot.push_bytes != push)
        {
            unsafe {
                self.device
                    .owner
                    .device
                    .cmd_push_constants(command, layout, stages, 0, push);
            }
            self.device.submission.statistics.push_updates += 1;
            slot.push_layout = layout;
            slot.push_stages = stages;
            slot.push_bytes.clear();
            slot.push_bytes.extend_from_slice(push);
        }
        Ok(command)
    }
    pub fn dispatch(
        &mut self,
        program: &PreparedProgram,
        dispatch: Dispatch,
        push: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        if program.pipeline.bind_point != vk::PipelineBindPoint::COMPUTE {
            return Err(VulkanError::InvalidResource(
                "dispatch requires a compute pipeline",
            ));
        }
        if let Dispatch::Direct(groups) = dispatch {
            if groups
                .iter()
                .zip(self.device.profile().limits.max_compute_work_group_count)
                .any(|(&n, max)| n > max)
            {
                return Err(VulkanError::InvalidResource(
                    "dispatch exceeds maxComputeWorkGroupCount",
                ));
            }
            if groups.contains(&0) {
                return self.device.work_ticket();
            }
        }
        self.program_access(program)?;
        let indirect = match dispatch {
            Dispatch::Indirect { buffer, offset } => {
                Some(self.indirect_buffer(buffer, offset, 12)?)
            }
            _ => None,
        };
        self.barriers.emit(&self.device);
        let command = self.program_bind(program, push)?;
        unsafe {
            match dispatch {
                Dispatch::Direct([x, y, z]) => {
                    self.device.owner.device.cmd_dispatch(command, x, y, z)
                }
                Dispatch::Indirect { offset, .. } => self
                    .device
                    .owner
                    .device
                    .cmd_dispatch_indirect(command, indirect.unwrap(), offset),
            }
        }
        self.program_written(program)?;
        self.device.work_ticket()
    }
    pub(super) fn indirect_buffer(
        &mut self,
        buffer: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<vk::Buffer, VulkanError> {
        if !offset.is_multiple_of(4)
            || !self
                .buffer(buffer)?
                .usage
                .contains(vk::BufferUsageFlags::INDIRECT_BUFFER)
        {
            return Err(VulkanError::InvalidResource(
                "indirect execution requires aligned INDIRECT_BUFFER storage",
            ));
        }
        self.ensure_buffer(buffer)?;
        self.buffer_access_range(
            buffer,
            offset,
            size,
            AccessScope::new(
                vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
        )
    }
}
#[cfg(test)]
pub(crate) mod tests;
