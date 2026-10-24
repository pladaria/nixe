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
    NullBuffer,
    TexelBuffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
        format: vk::Format,
        element_bytes: u32,
    },
    Buffer {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
    Image(ImageViewId),
    CombinedImageSampler {
        image: ImageViewId,
        sampler: Arc<Sampler>,
    },
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
/// Prepared guest state retains resources, not native descriptor sets. Sets
/// belong to recording slots and recycle after their actual GPU use retires.
pub struct PreparedProgram {
    pub(super) pipeline: Arc<NativePipeline>,
    // Explicit host bindings are owned by the host texture cache.
    record: Option<Arc<BindingRecord>>,
    bindings: Box<[Binding]>,
    pub(super) uses: Box<[Use]>,
    // Descriptor indices grouped by their streamed canonical access window.
    uniform_bindings: Box<[usize]>,
    uniforms: Box<[(usize, buffer::BufferStreamRequest, std::ops::Range<usize>)]>,
    unaligned_uniforms: bool,
    dynamic_offsets: Box<[u32]>,
}
impl PreparedProgram {
    pub(crate) fn pipeline(&self) -> &Arc<NativePipeline> {
        &self.pipeline
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.uses.as_ref())
            + std::mem::size_of_val(self.uniform_bindings.as_ref())
            + std::mem::size_of_val(self.uniforms.as_ref())
            + std::mem::size_of_val(self.dynamic_offsets.as_ref())
            + std::mem::size_of_val(self.bindings.as_ref())
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Dispatch {
    Direct([u32; 3]),
    Indirect { buffer: BufferId, offset: u64 },
}
struct GraphicsStreams {
    dynamic_offsets: Option<Vec<u32>>,
    record: Option<Arc<BindingRecord>>,
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
        let dynamic_offsets = vec![0; pipeline.layout.dynamic_bindings.len()].into_boxed_slice();
        self.prepare_graphics(
            PreparedProgram {
                pipeline,
                dynamic_offsets,
                record: Some(record),
                bindings: Box::new([]),
                uses: Box::new([]),
                uniform_bindings: Box::new([]),
                uniforms: Box::new([]),
                unaligned_uniforms: false,
            },
            parameters,
        )
    }
    pub(crate) fn bind_host_graphics(
        &mut self,
        graphics: &PreparedGraphics,
        push: &[u8],
    ) -> Result<(), VulkanError> {
        self.device.end_rendering();
        self.graphics_state(graphics)?;
        self.program_bind(&graphics.program, push).map(|_| ())
    }
    pub(crate) fn program_current(&self, program: &PreparedProgram) -> bool {
        self.bindings_current(&program.bindings)
    }
    pub(crate) fn bindings_current(&self, bindings: &[Binding]) -> bool {
        bindings.iter().all(|binding| match binding {
            Binding::Buffer { buffer, .. } | Binding::TexelBuffer { buffer, .. } => {
                self.buffer_binding_current(*buffer)
            }
            Binding::Image(view) | Binding::CombinedImageSampler { image: view, .. } => {
                self.image_binding_current(view.image)
            }
            _ => true,
        })
    }
    pub fn prepare_program(
        &self,
        pipeline: Arc<NativePipeline>,
        bindings: &[Binding],
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
                flags |= if matches!(
                    reflected.kind,
                    vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                ) {
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                } else if matches!(
                    reflected.kind,
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                ) {
                    vk::AccessFlags2::UNIFORM_READ
                } else {
                    vk::AccessFlags2::SHADER_STORAGE_READ
                };
            }
            if !access.write.is_empty() {
                flags |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
            }
            let scope = AccessScope::new(stages, flags);
            match binding {
                Binding::NullBuffer => {
                    if !self.device.profile().null_descriptor
                        || !matches!(
                            reflected.kind,
                            vk::DescriptorType::STORAGE_BUFFER
                                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                                | vk::DescriptorType::UNIFORM_BUFFER
                                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        )
                    {
                        return Err(VulkanError::InvalidResource(
                            "null buffer descriptor is not supported for this binding",
                        ));
                    }
                    resources.push(DescriptorResource::NullBuffer);
                }
                Binding::Buffer {
                    buffer,
                    offset,
                    size,
                }
                | Binding::TexelBuffer {
                    buffer,
                    offset,
                    size,
                    ..
                } => {
                    self.check_buffer_range(*buffer, *offset, *size)?;
                    let lease = self.buffer_lease(*buffer)?;
                    resources.push(match binding {
                        Binding::TexelBuffer {
                            format,
                            element_bytes,
                            ..
                        } => DescriptorResource::TexelBuffer {
                            buffer: lease,
                            offset: *offset,
                            range: *size,
                            format: *format,
                            element_bytes: *element_bytes,
                        },
                        _ => DescriptorResource::Buffer {
                            buffer: lease,
                            offset: *offset,
                            range: *size,
                        },
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
                Binding::Image(view) | Binding::CombinedImageSampler { image: view, .. } => {
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
                    let view_lease = self.image_view_lease(*view)?;
                    resources.push(match binding {
                        Binding::CombinedImageSampler { sampler, .. } => {
                            DescriptorResource::CombinedImageSampler {
                                view: view_lease,
                                layout: vk::ImageLayout::GENERAL,
                                sampler: Arc::clone(sampler),
                            }
                        }
                        _ => DescriptorResource::Image {
                            view: view_lease,
                            layout: vk::ImageLayout::GENERAL,
                        },
                    });
                    for &plane in image.planes.iter() {
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
        let binding_uses: Box<[_]> = bindings.iter().map(|binding| match binding {
            Binding::Buffer { buffer, offset, size } | Binding::TexelBuffer { buffer, offset, size, .. } => merged.iter().position(|usage| matches!(usage,
                Use::Buffer { id, offset: start, size: length, .. } if id == buffer && *start <= *offset && *offset + *size <= *start + *length
            )),
            _ => None,
        }).collect();
        let limits = self.device.profile().limits;
        let mut buffer_alignments: Box<[_]> = merged
            .iter()
            .map(|usage| match usage {
                Use::Buffer { scope, .. } => scope.buffer_alignment(&limits),
                _ => 1,
            })
            .collect();
        for (binding, usage) in bindings.iter().zip(&binding_uses) {
            if matches!(binding, Binding::TexelBuffer { .. })
                && let Some(index) = usage
            {
                buffer_alignments[*index] =
                    buffer_alignments[*index].max(limits.min_texel_buffer_offset_alignment);
            }
        }
        // Unaligned uniforms are rebound to an aligned immutable allocation at
        // consumption, either a CPU stream or a native copy of retained data.
        let mut unaligned_uniforms = false;
        for (resource, binding) in resources.iter_mut().zip(&pipeline.layout.key.bindings) {
            if matches!(
                binding.kind,
                vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            ) && let DescriptorResource::Buffer { offset, .. } = resource
                && !offset.is_multiple_of(limits.min_uniform_buffer_offset_alignment)
            {
                *offset = 0;
                unaligned_uniforms = true;
            }
        }
        let dynamic_offsets = pipeline
            .layout
            .split_dynamic_offsets(&mut resources, &self.device.profile().limits, Vec::new())?
            .into_boxed_slice();
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        let mut uniform_bindings: Vec<_> = binding_uses
            .iter()
            .enumerate()
            .filter_map(|(binding, &usage)| {
                let index = usage?;
                matches!(merged[index], Use::Buffer { scope, size, .. }
                    if scope.access == vk::AccessFlags2::UNIFORM_READ && size <= 64 * 1024)
                .then_some((index, binding))
            })
            .collect();
        uniform_bindings.sort_unstable_by_key(|&(index, _)| index);
        let uniforms = merged
            .iter()
            .zip(&buffer_alignments)
            .enumerate()
            .filter_map(|(index, (usage, &alignment))| match *usage {
                Use::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } if scope.access == vk::AccessFlags2::UNIFORM_READ && size <= 64 * 1024 => Some((
                    index,
                    buffer::BufferStreamRequest {
                        id,
                        offset,
                        size,
                        alignment,
                    },
                    uniform_bindings.partition_point(|&(usage, _)| usage < index)
                        ..uniform_bindings.partition_point(|&(usage, _)| usage <= index),
                )),
                _ => None,
            })
            .collect();
        Ok(PreparedProgram {
            pipeline,
            record: None,
            bindings: bindings.to_vec().into_boxed_slice(),
            uses: merged.into_boxed_slice(),
            uniform_bindings: uniform_bindings
                .into_iter()
                .map(|(_, binding)| binding)
                .collect(),
            uniforms,
            unaligned_uniforms,
            dynamic_offsets,
        })
    }
    fn stream_graphics(
        &mut self,
        program: &PreparedProgram,
    ) -> Result<GraphicsStreams, VulkanError> {
        self.streamed_uses.clear();
        let mut resources = std::mem::take(&mut self.descriptor_resources);
        resources.clear();
        for (index, request, bindings) in &program.uniforms {
            self.uniform_cache_shots[0] += 1;
            let unaligned = !request.offset.is_multiple_of(request.alignment);
            if !unaligned && request.size > self.uniform_stream_limit {
                continue;
            }
            let Some((buffer, base)) = self.stream_buffer(*request)? else {
                continue;
            };
            if resources.is_empty() {
                self.resolve_bindings(program, &mut resources)?;
            }
            for &binding in &program.uniform_bindings[bindings.clone()] {
                match &mut resources[binding] {
                    DescriptorResource::Buffer {
                        buffer: lease,
                        offset,
                        ..
                    }
                    | DescriptorResource::TexelBuffer {
                        buffer: lease,
                        offset,
                        ..
                    } => {
                        *lease = Arc::clone(&buffer);
                        *offset = base + *offset - request.offset;
                    }
                    _ => (),
                }
            }
            self.streamed_uses.push(*index);
        }
        if resources.is_empty() && !program.unaligned_uniforms {
            self.descriptor_resources = resources;
            return Ok(GraphicsStreams {
                dynamic_offsets: None,
                record: None,
            });
        }
        if resources.is_empty() {
            self.resolve_bindings(program, &mut resources)?;
        }
        let limits = self.device.profile().limits;
        if program.unaligned_uniforms {
            for ((resource, binding), description) in resources
                .iter_mut()
                .zip(&program.bindings)
                .zip(&program.pipeline.layout.key.bindings)
            {
                if matches!(
                    description.kind,
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                ) && let DescriptorResource::Buffer {
                    buffer: native,
                    offset,
                    range,
                } = resource
                    && !offset.is_multiple_of(limits.min_uniform_buffer_offset_alignment)
                    && let Binding::Buffer {
                        buffer,
                        offset: source,
                        ..
                    } = binding
                {
                    *native = self.aligned_uniform_copy(*buffer, *source, *range)?;
                    *offset = 0;
                }
            }
        }
        let offsets = std::mem::take(&mut self.descriptor_offsets);
        let dynamic_offsets =
            program
                .pipeline
                .layout
                .split_dynamic_offsets(&mut resources, &limits, offsets)?;
        let result = self.slot_descriptors(&program.pipeline.layout, &resources);
        resources.clear();
        self.descriptor_resources = resources;
        Ok(GraphicsStreams {
            dynamic_offsets: Some(dynamic_offsets),
            record: Some(result?),
        })
    }
    fn program_access_except(
        &mut self,
        program: &PreparedProgram,
    ) -> Result<smallvec::SmallVec<[WriterId; 16]>, VulkanError> {
        let mut writers = smallvec::SmallVec::new();
        for (index, &resource) in program.uses.iter().enumerate() {
            if self.streamed_uses.binary_search(&index).is_ok() {
                continue;
            }
            match resource {
                Use::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } => {
                    let uploaded = self.statistics.cpu_uploaded_bytes;
                    self.ensure_buffer_range(id, offset, size)?;
                    if scope.access == vk::AccessFlags2::UNIFORM_READ
                        && size <= 64 * 1024
                        && self.statistics.cpu_uploaded_bytes == uploaded
                    {
                        self.uniform_cache_hits[0] += 1;
                    }
                    // Preserve bytes not actually written by a conditional shader
                    // store, even in a descriptor classified as write-only.
                    if scope.writes() {
                        writers.push(self.prepare_write(Resource::Buffer(id), offset, size)?);
                    }
                    self.buffer_access_range(id, offset, size, scope)?;
                }
                Use::Image { id, plane, scope } => {
                    // A sampled view can include an unwritten native border.
                    // Refresh actual producers without claiming that every texel
                    // in the view was consumed or importing undefined padding.
                    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
                    if scope.writes() {
                        self.ensure_image_initialized(id, plane, transfers::full_rect(plane))?;
                        writers.push(self.prepare_image_rect(
                            id,
                            plane,
                            transfers::full_rect(plane),
                        )?);
                    } else {
                        self.ensure_image(id)?;
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
        Ok(writers)
    }
    pub(super) fn program_written(&mut self, program: &PreparedProgram) -> Result<(), VulkanError> {
        for &resource in &program.uses {
            match resource {
                Use::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } if scope.writes() => self.finish_buffer_write(id, offset, size)?,
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
        let record = self.program_record(program)?;
        self.program_bind_record(program, push, record.as_ref(), &program.dynamic_offsets)
    }
    fn program_record(
        &mut self,
        program: &PreparedProgram,
    ) -> Result<Option<Arc<BindingRecord>>, VulkanError> {
        if let Some(record) = &program.record {
            return Ok(Some(Arc::clone(record)));
        }
        if program.bindings.is_empty() {
            return Ok(None);
        }
        let mut resources = std::mem::take(&mut self.descriptor_resources);
        resources.clear();
        self.resolve_bindings(program, &mut resources)?;
        for (&index, &dynamic) in program
            .pipeline
            .layout
            .dynamic_bindings
            .iter()
            .zip(&program.dynamic_offsets)
        {
            if let DescriptorResource::Buffer { offset, .. } = &mut resources[index] {
                *offset -= u64::from(dynamic);
            }
        }
        let result = self.slot_descriptors(&program.pipeline.layout, &resources);
        resources.clear();
        self.descriptor_resources = resources;
        result.map(Some)
    }
    fn resolve_bindings(
        &self,
        program: &PreparedProgram,
        resources: &mut Vec<DescriptorResource>,
    ) -> Result<(), VulkanError> {
        for binding in &program.bindings {
            resources.push(match binding {
                Binding::NullBuffer => DescriptorResource::NullBuffer,
                Binding::Buffer {
                    buffer,
                    offset,
                    size,
                } => DescriptorResource::Buffer {
                    buffer: self.buffer_lease(*buffer)?,
                    offset: *offset,
                    range: *size,
                },
                Binding::TexelBuffer {
                    buffer,
                    offset,
                    size,
                    format,
                    element_bytes,
                } => DescriptorResource::TexelBuffer {
                    buffer: self.buffer_lease(*buffer)?,
                    offset: *offset,
                    range: *size,
                    format: *format,
                    element_bytes: *element_bytes,
                },
                Binding::Image(image) => DescriptorResource::Image {
                    view: self.image_view_lease(*image)?,
                    layout: vk::ImageLayout::GENERAL,
                },
                Binding::CombinedImageSampler { image, sampler } => {
                    DescriptorResource::CombinedImageSampler {
                        view: self.image_view_lease(*image)?,
                        layout: vk::ImageLayout::GENERAL,
                        sampler: Arc::clone(sampler),
                    }
                }
                Binding::Sampler(sampler) => DescriptorResource::Sampler(Arc::clone(sampler)),
            });
        }
        Ok(())
    }
    pub(crate) fn retain_program_effect(&mut self, program: &PreparedProgram) {
        self.constructing
            .extend(program.bindings.iter().filter_map(|binding| match binding {
                Binding::Buffer { buffer, .. } | Binding::TexelBuffer { buffer, .. } => {
                    Some(Resource::Buffer(*buffer))
                }
                Binding::Image(view) | Binding::CombinedImageSampler { image: view, .. } => {
                    Some(Resource::Image(view.image))
                }
                _ => None,
            }));
    }
    fn slot_descriptors(
        &mut self,
        layout: &Arc<crate::cache::PipelineLayout>,
        resources: &[DescriptorResource],
    ) -> Result<Arc<BindingRecord>, VulkanError> {
        let limits = self.device.profile().limits;
        let null_descriptor = self.device.profile().null_descriptor;
        let index = self.device.begin_recording()?;
        let descriptors = self.device.submission.slots[index]
            .stream_descriptors
            .get_or_insert_with(|| {
                DescriptorCache::from_limits(
                    Arc::clone(&self.device.owner),
                    limits,
                    4096,
                    null_descriptor,
                )
            });
        let before = descriptors.statistics();
        let result = descriptors.get(layout, resources);
        let after = descriptors.statistics();
        self.device.submission.statistics.stream_descriptor_hits += after.hits - before.hits;
        self.device
            .submission
            .statistics
            .stream_descriptor_allocations += after.allocated - before.allocated;
        result
    }
    fn program_bind_record(
        &mut self,
        program: &PreparedProgram,
        push: &[u8],
        record: Option<&Arc<BindingRecord>>,
        dynamic_offsets: &[u32],
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
        if let Some(record) = record {
            self.device
                .bind_native_descriptors(&program.pipeline, record, dynamic_offsets)?;
        }
        let index = self.device.begin_recording()?;
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
                    .dispatch
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
        let retained = self.constructing.len();
        self.retain_program_effect(program);
        if let Dispatch::Indirect { buffer, .. } = dispatch {
            self.constructing.push(Resource::Buffer(buffer));
        }
        let result = (|| loop {
            let result = self.dispatch_inner(program, dispatch, push);
            self.device.submission.release_stream_uploads();
            match result {
                Err(VulkanError::DescriptorPressure) => {
                    self.flush()?;
                }
                result => return result,
            }
        })();
        self.constructing.truncate(retained);
        result
    }
    fn dispatch_inner(
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
        let streams = self.stream_graphics(program)?;
        let writers = self.program_access_except(program)?;
        let indirect = match dispatch {
            Dispatch::Indirect { buffer, offset } => {
                Some(self.indirect_buffer(buffer, offset, 12)?)
            }
            _ => None,
        };
        self.barriers.emit(&mut self.device);
        let record = match streams.record {
            Some(record) => Some(record),
            None => self.program_record(program)?,
        };
        let command = self.program_bind_record(
            program,
            push,
            record.as_ref(),
            streams
                .dynamic_offsets
                .as_deref()
                .unwrap_or(&program.dynamic_offsets),
        )?;
        if let Some(offsets) = streams.dynamic_offsets {
            self.descriptor_offsets = offsets;
        }
        self.publish_effect(&writers, |device| unsafe {
            match dispatch {
                Dispatch::Direct([x, y, z]) => device.dispatch.cmd_dispatch(command, x, y, z),
                Dispatch::Indirect { offset, .. } => {
                    device
                        .dispatch
                        .cmd_dispatch_indirect(command, indirect.unwrap(), offset)
                }
            }
        })?;
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
        self.ensure_buffer_range(buffer, offset, size)?;
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
