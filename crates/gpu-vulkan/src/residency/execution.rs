//! Prepared native execution. Descriptor reflection supplies the access scopes;
//! callers cannot omit shader writes from canonical ownership or synchronization.
use super::*;
use crate::cache::{
    BindingRecord, DescriptorResource, GraphicsState, NativePipeline, Sampler, ShaderArtifact,
};
mod graphics;
mod state;
pub use graphics::{
    AttachmentSource, Draw, IndexBuffer, PreparedDraw, PreparedTargets, RenderAttachment,
    RenderingTargets, VertexBuffer,
};
pub(crate) use state::PreparedGraphicsState;
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
        range: vk::ImageSubresourceRange,
        scope: AccessScope,
    },
}
/// Prepared guest state retains resources, not native descriptor sets. Sets
/// belong to native layout banks and recycle after their actual GPU use retires.
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
        self.set_feedback_loop(vk::ImageAspectFlags::empty());
        self.program_bind(&graphics.program, push).map(|_| ())
    }
    fn set_feedback_loop(&mut self, aspects: vk::ImageAspectFlags) {
        if !self.device.profile().attachment_feedback_loop {
            return;
        }
        let slot = &mut self.device.submission.slots[self
            .device
            .submission
            .recording
            .expect("active graphics recording")];
        if slot.feedback_aspects == Some(aspects) {
            return;
        }
        unsafe {
            self.device.dispatch.cmd_set_feedback_loop(aspects);
        }
        slot.feedback_aspects = Some(aspects);
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
                    // The chosen layout remains fixed for this image's lifetime,
                    // including when it is used as a rendering attachment.
                    let layout = shader_attachment_layout(image.description.usage);
                    let view_lease = self.image_view_lease(*view)?;
                    resources.push(match binding {
                        Binding::CombinedImageSampler { sampler, .. } => {
                            DescriptorResource::CombinedImageSampler {
                                view: view_lease,
                                layout,
                                sampler: Arc::clone(sampler),
                            }
                        }
                        _ => DescriptorResource::Image {
                            view: view_lease,
                            layout,
                        },
                    });
                    // Views are validated once when created. Keep their native
                    // ranges instead of expanding and rejoining every mip/layer.
                    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L519-L538
                    if !stages.is_empty() {
                        let volume = image.description.image_type == vk::ImageType::TYPE_3D;
                        for aspect in [
                            vk::ImageAspectFlags::COLOR,
                            vk::ImageAspectFlags::DEPTH,
                            vk::ImageAspectFlags::STENCIL,
                        ] {
                            if description.aspects.contains(aspect) {
                                uses.push(Use::Image {
                                    id: view.image,
                                    range: vk::ImageSubresourceRange::default()
                                        .aspect_mask(aspect)
                                        .base_mip_level(description.base_mip)
                                        .level_count(description.mip_count)
                                        .base_array_layer(if volume {
                                            0
                                        } else {
                                            description.base_layer
                                        })
                                        .layer_count(if volume {
                                            1
                                        } else {
                                            description.layer_count
                                        }),
                                    scope,
                                });
                            }
                        }
                    }
                }
                Binding::Sampler(sampler) => {
                    resources.push(DescriptorResource::Sampler(Arc::clone(sampler)))
                }
            }
        }
        let merged = merge_program_uses(uses);
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
        let result = self.native_descriptors(&program.pipeline.layout, &resources);
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
        let mut prepared_image = None;
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
                Use::Image { id, range, scope } => {
                    // Image uses are grouped by ID in the prepared program.
                    if prepared_image != Some(id) {
                        self.ensure_image(id)?;
                        prepared_image = Some(id);
                    }
                    if scope.writes() {
                        let planes = self.image(id)?.planes.clone();
                        for &plane in planes.iter().filter(|plane| plane_in_range(**plane, range)) {
                            self.require_image_initialized(id, plane, transfers::full_rect(plane))?;
                            writers.push(self.prepare_image_rect(
                                id,
                                plane,
                                transfers::full_rect(plane),
                            )?);
                        }
                    }
                    self.image_access_subresources(
                        id,
                        range,
                        scope,
                        shader_attachment_layout(self.image(id)?.description.usage),
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
                Use::Image { id, range, scope } if scope.writes() => {
                    let planes = self.image(id)?.planes.clone();
                    for &plane in planes.iter().filter(|plane| plane_in_range(**plane, range)) {
                        self.finish_image_rect(id, plane, transfers::full_rect(plane))?;
                    }
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
    ) -> Result<(), VulkanError> {
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
        let result = self.native_descriptors(&program.pipeline.layout, &resources);
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
                    layout: shader_attachment_layout(self.image(image.image)?.description.usage),
                },
                Binding::CombinedImageSampler { image, sampler } => {
                    DescriptorResource::CombinedImageSampler {
                        view: self.image_view_lease(*image)?,
                        layout: shader_attachment_layout(
                            self.image(image.image)?.description.usage,
                        ),
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
    fn native_descriptors(
        &mut self,
        layout: &Arc<crate::cache::PipelineLayout>,
        resources: &[DescriptorResource],
    ) -> Result<Arc<BindingRecord>, VulkanError> {
        self.device.begin_recording()?;
        let descriptors = &mut self.descriptors;
        let before = descriptors.hits();
        let result = descriptors.get(layout, resources);
        let hits = descriptors.hits() - before;
        self.device.submission.statistics.descriptor_payload_hits += hits;
        self.device
            .submission
            .statistics
            .descriptor_payload_captures += u64::from(result.is_ok() && hits == 0);
        result
    }
    fn program_bind_record(
        &mut self,
        program: &PreparedProgram,
        push: &[u8],
        record: Option<&Arc<BindingRecord>>,
        dynamic_offsets: &[u32],
    ) -> Result<(), VulkanError> {
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
                    .cmd_push_constants(layout, stages, 0, push);
            }
            self.device.submission.statistics.push_updates += 1;
            slot.push_layout = layout;
            slot.push_stages = stages;
            slot.push_bytes.clear();
            slot.push_bytes.extend_from_slice(push);
        }
        Ok(())
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
        let result = self.dispatch_inner(program, dispatch, push);
        self.device.submission.release_stream_uploads();
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
        self.program_bind_record(
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
                Dispatch::Direct([x, y, z]) => device.dispatch.cmd_dispatch(x, y, z),
                Dispatch::Indirect { offset, .. } => device
                    .dispatch
                    .cmd_dispatch_indirect(indirect.unwrap(), offset),
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

fn plane_in_range(plane: ImagePlane, range: vk::ImageSubresourceRange) -> bool {
    range.aspect_mask.intersects(plane.aspect)
        && (range.base_mip_level..range.base_mip_level + range.level_count)
            .contains(&plane.mip_level)
        && (range.base_array_layer..range.base_array_layer + range.layer_count)
            .contains(&plane.array_layer)
}

// Only descriptor boundaries partition simultaneous image accesses. Texture
// dimensions and layer counts do not determine the size of this preparation.
fn merge_program_uses(mut uses: Vec<Use>) -> Vec<Use> {
    uses.sort_unstable_by_key(|usage| match *usage {
        Use::Buffer { id, offset, .. } => [0, id.slot as u64, u64::from(id.generation), 0, offset],
        Use::Image { id, range, .. } => [
            1,
            id.slot as u64,
            u64::from(id.generation),
            u64::from(range.aspect_mask.as_raw()),
            0,
        ],
    });
    let mut merged = Vec::with_capacity(uses.len());
    let mut group = None;
    let mut group_start = 0;
    for usage in uses {
        match usage {
            Use::Buffer {
                id,
                offset,
                size,
                scope,
            } => {
                if let Some(Use::Buffer {
                    id: previous,
                    offset: start,
                    size: length,
                    scope: access,
                }) = merged.last_mut()
                    && *previous == id
                    && offset <= *start + *length
                {
                    *length = (*start + *length).max(offset + size) - *start;
                    access.stages |= scope.stages;
                    access.access |= scope.access;
                } else {
                    merged.push(usage);
                }
            }
            Use::Image { id, range, scope } => {
                let key = (id, range.aspect_mask);
                if group != Some(key) {
                    group = Some(key);
                    group_start = merged.len();
                }
                let mut pending: smallvec::SmallVec<[vk::ImageSubresourceRange; 4]> =
                    smallvec::smallvec![range];
                let mut pieces: smallvec::SmallVec<[Use; 4]> = smallvec::SmallVec::new();
                for old in &mut merged[group_start..] {
                    let Use::Image {
                        range: previous,
                        scope: access,
                        ..
                    } = *old
                    else {
                        unreachable!()
                    };
                    let Some(overlap) = image_range_overlap(previous, range) else {
                        continue;
                    };
                    let combined = AccessScope::new(
                        access.stages | scope.stages,
                        access.access | scope.access,
                    );
                    if combined != access {
                        subtract_image_range(previous, overlap, |range| {
                            pieces.push(Use::Image {
                                id,
                                range,
                                scope: access,
                            });
                        });
                        *old = Use::Image {
                            id,
                            range: overlap,
                            scope: combined,
                        };
                    }
                    let mut index = 0;
                    while index < pending.len() {
                        let piece = pending[index];
                        if let Some(overlap) = image_range_overlap(piece, previous) {
                            let _ = pending.swap_remove(index);
                            subtract_image_range(piece, overlap, |range| pending.push(range));
                        } else {
                            index += 1;
                        }
                    }
                    if pending.is_empty() {
                        break;
                    }
                }
                merged.extend(pieces);
                merged.extend(
                    pending
                        .into_iter()
                        .map(|range| Use::Image { id, range, scope }),
                );
            }
        }
    }
    coalesce_image_uses(&mut merged);
    merged
}

fn image_range_overlap(
    a: vk::ImageSubresourceRange,
    b: vk::ImageSubresourceRange,
) -> Option<vk::ImageSubresourceRange> {
    let mip = a.base_mip_level.max(b.base_mip_level);
    let mip_end = (a.base_mip_level + a.level_count).min(b.base_mip_level + b.level_count);
    let layer = a.base_array_layer.max(b.base_array_layer);
    let layer_end = (a.base_array_layer + a.layer_count).min(b.base_array_layer + b.layer_count);
    (mip < mip_end && layer < layer_end).then(|| vk::ImageSubresourceRange {
        aspect_mask: a.aspect_mask,
        base_mip_level: mip,
        level_count: mip_end - mip,
        base_array_layer: layer,
        layer_count: layer_end - layer,
    })
}

fn subtract_image_range(
    range: vk::ImageSubresourceRange,
    cut: vk::ImageSubresourceRange,
    mut emit: impl FnMut(vk::ImageSubresourceRange),
) {
    for (mip, mip_end, layer, layer_end) in [
        (
            range.base_mip_level,
            cut.base_mip_level,
            range.base_array_layer,
            range.base_array_layer + range.layer_count,
        ),
        (
            cut.base_mip_level + cut.level_count,
            range.base_mip_level + range.level_count,
            range.base_array_layer,
            range.base_array_layer + range.layer_count,
        ),
        (
            cut.base_mip_level,
            cut.base_mip_level + cut.level_count,
            range.base_array_layer,
            cut.base_array_layer,
        ),
        (
            cut.base_mip_level,
            cut.base_mip_level + cut.level_count,
            cut.base_array_layer + cut.layer_count,
            range.base_array_layer + range.layer_count,
        ),
    ] {
        if mip < mip_end && layer < layer_end {
            emit(vk::ImageSubresourceRange {
                base_mip_level: mip,
                level_count: mip_end - mip,
                base_array_layer: layer,
                layer_count: layer_end - layer,
                ..range
            });
        }
    }
}

fn coalesce_image_uses(uses: &mut Vec<Use>) {
    for layers in [true, false] {
        uses.sort_unstable_by_key(|usage| match *usage {
            Use::Buffer {
                id, offset, size, ..
            } => [
                0,
                id.slot as u64,
                u64::from(id.generation),
                0,
                0,
                0,
                0,
                0,
                offset,
                size,
            ],
            Use::Image { id, range, scope } => {
                let (base, count, next, length) = if layers {
                    (
                        range.base_mip_level,
                        range.level_count,
                        range.base_array_layer,
                        range.layer_count,
                    )
                } else {
                    (
                        range.base_array_layer,
                        range.layer_count,
                        range.base_mip_level,
                        range.level_count,
                    )
                };
                [
                    1,
                    id.slot as u64,
                    u64::from(id.generation),
                    scope.stages.as_raw(),
                    scope.access.as_raw(),
                    u64::from(range.aspect_mask.as_raw()),
                    u64::from(base),
                    u64::from(count),
                    u64::from(next),
                    u64::from(length),
                ]
            }
        });
        let mut count = 0;
        for index in 0..uses.len() {
            let next = uses[index];
            if count != 0
                && let (
                    Use::Image { id, range, scope },
                    Use::Image {
                        id: other,
                        range: b,
                        scope: previous,
                    },
                ) = (&mut uses[count - 1], next)
                && *id == other
                && *scope == previous
                && range.aspect_mask == b.aspect_mask
            {
                if layers
                    && range.base_mip_level == b.base_mip_level
                    && range.level_count == b.level_count
                    && range.base_array_layer + range.layer_count == b.base_array_layer
                {
                    range.layer_count += b.layer_count;
                    continue;
                }
                if !layers
                    && range.base_array_layer == b.base_array_layer
                    && range.layer_count == b.layer_count
                    && range.base_mip_level + range.level_count == b.base_mip_level
                {
                    range.level_count += b.level_count;
                    continue;
                }
            }
            uses[count] = next;
            count += 1;
        }
        uses.truncate(count);
    }
}
