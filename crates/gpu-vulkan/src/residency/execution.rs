//! Prepared native execution. Descriptor reflection supplies the resource roles;
//! callers cannot omit shader writes from canonical ownership or synchronization.
use super::*;
use crate::cache::{
    BindingRecord, DescriptorResource, GraphicsState, NativePipeline, Sampler, ShaderArtifact,
};
use std::sync::atomic::{AtomicU64, Ordering};
mod graphics;
mod guest;
mod host;
mod state;
pub use graphics::{
    AttachmentSource, Draw, IndexBuffer, PreparedTargets, RenderAttachment, RenderingTargets,
    VertexBuffer,
};
pub(crate) use graphics::{DeferredClear, RenderPassState, ShaderImage, attachment_read_only};
pub(crate) use graphics::{MaskedClearPipelines, NativeIndexBuffer};
pub use state::{GraphicsParameters, HostGraphics};

pub(crate) fn shader_accesses(
    shader: &ShaderArtifact,
    layout: &crate::cache::LayoutKey,
) -> Box<[AccessScope]> {
    layout
        .bindings
        .iter()
        .map(|b| {
            let mut read = vk::PipelineStageFlags2::empty();
            let mut write = vk::PipelineStageFlags2::empty();
            for stage in &shader.stages {
                let flags = shader_stage(stage.stage);
                for binding in stage
                    .module
                    .bindings
                    .iter()
                    .filter(|r| r.binding == b.binding)
                {
                    if binding.readable {
                        read |= flags;
                    }
                    if binding.writable {
                        write |= flags;
                    }
                }
            }
            let mut flags = vk::AccessFlags2::empty();
            if !read.is_empty() {
                flags |= if matches!(
                    b.kind,
                    vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                ) {
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                } else if matches!(
                    b.kind,
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                ) {
                    vk::AccessFlags2::UNIFORM_READ
                } else {
                    vk::AccessFlags2::SHADER_STORAGE_READ
                };
            }
            if !write.is_empty() {
                flags |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
            }
            AccessScope::new(read | write, flags)
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
/// Host utility inputs retain their actual native allocation. Compact cache
/// IDs select coherence metadata; native identity prevents reuse from accepting
/// an unrelated resource after explicit retirement or an overlapping merge.
#[derive(Clone)]
pub struct HostBuffer {
    pub(super) id: BufferId,
    native: Arc<AllocatedBuffer>,
}
#[derive(Clone)]
pub struct HostImage {
    pub(super) id: ImageViewId,
    pub(super) image: ImageId,
    native: Arc<crate::AllocatedImageView>,
}
#[derive(Clone)]
pub enum HostBinding {
    NullBuffer,
    TexelBuffer {
        buffer: HostBuffer,
        offset: u64,
        size: u64,
        format: vk::Format,
        element_bytes: u32,
    },
    Buffer {
        buffer: HostBuffer,
        offset: u64,
        size: u64,
    },
    Image(HostImage),
    CombinedImageSampler {
        image: HostImage,
        sampler: Sampler,
    },
    Sampler(Sampler),
}
#[derive(Clone, Copy)]
pub(super) enum HostBindingAccess {
    Buffer {
        id: BufferId,
        offset: u64,
        size: u64,
        scope: AccessScope,
    },
    Image {
        id: ImageId,
        scope: AccessScope,
    },
}
/// Host utility shader captures. Guest Configure uses cache-owned hardware
/// banks and DescriptorQueue directly; it never constructs this object.
pub struct HostProgram {
    binding_revision: AtomicU64,
    pub(super) pipeline: NativePipeline,
    // Explicit host bindings are owned by the host texture cache.
    record: std::cell::RefCell<ProgramRecord>,
    // Explicit host resources use owned captures; guest resources use residency
    // use ticks and the native execution owner's persistent sampler cache.
    bindings: Vec<HostBinding>,
    unaligned_uniforms: usize,
    dynamic_offsets: Vec<u32>,
}
#[derive(Default)]
struct ProgramRecord {
    current: Option<Arc<BindingRecord>>,
    streamed: bool,
}
#[derive(Clone, Copy, Debug)]
pub enum Dispatch {
    Direct([u32; 3]),
    Indirect { buffer: BufferId, offset: u64 },
}
pub(crate) struct DescriptorQueue {
    pub(crate) dynamic_offsets: Option<Vec<u32>>,
    pub(crate) record: Option<Arc<BindingRecord>>,
    pub(crate) update: Option<crate::cache::DescriptorUpdate>,
}
impl Residency<'_> {
    pub fn host_buffer(&self, id: BufferId) -> Result<HostBuffer, VulkanError> {
        let entry = self
            .buffer_cache
            .buffers
            .get(id)
            .filter(|entry| entry.binding_valid)
            .ok_or(VulkanError::InvalidResource("retired host buffer input"))?;
        Ok(HostBuffer {
            id,
            native: Arc::clone(&entry.native),
        })
    }
    pub fn host_image(&self, id: ImageViewId) -> Result<HostImage, VulkanError> {
        let view = self
            .texture_cache
            .views
            .get(id)
            .filter(|view| self.texture_cache.binding_current(view.image))
            .ok_or(VulkanError::InvalidResource("retired host image input"))?;
        Ok(HostImage {
            id,
            image: view.image,
            native: Arc::clone(&view.native),
        })
    }
    fn host_binding_usage(&self, program: &HostProgram, index: usize) -> Option<HostBindingAccess> {
        let scope = program.pipeline.resources[index];
        if scope.stages.is_empty() {
            return None;
        }
        match &program.bindings[index] {
            HostBinding::Buffer {
                buffer,
                offset,
                size,
            }
            | HostBinding::TexelBuffer {
                buffer,
                offset,
                size,
                ..
            } => Some(HostBindingAccess::Buffer {
                id: buffer.id,
                offset: *offset,
                size: *size,
                scope,
            }),
            HostBinding::Image(id) | HostBinding::CombinedImageSampler { image: id, .. } => {
                Some(HostBindingAccess::Image {
                    id: id.image,
                    scope,
                })
            }
            HostBinding::NullBuffer | HostBinding::Sampler(_) => None,
        }
    }
    fn canonical_resource(
        &self,
        program: &HostProgram,
        index: usize,
    ) -> Result<DescriptorResource, VulkanError> {
        let mut resource = match &program.bindings[index] {
            HostBinding::NullBuffer => DescriptorResource::NullBuffer,
            HostBinding::Buffer {
                buffer,
                offset,
                size,
            } => DescriptorResource::Buffer {
                buffer: Arc::clone(&buffer.native),
                offset: *offset,
                range: *size,
            },
            HostBinding::TexelBuffer {
                buffer,
                offset,
                size,
                format,
                element_bytes,
            } => DescriptorResource::TexelBuffer {
                buffer: Arc::clone(&buffer.native),
                offset: *offset,
                range: *size,
                format: *format,
                element_bytes: *element_bytes,
            },
            HostBinding::Image(id) | HostBinding::CombinedImageSampler { image: id, .. } => {
                let layout = vk::ImageLayout::GENERAL;
                let view = Arc::clone(&id.native);
                if let HostBinding::CombinedImageSampler { sampler, .. } = &program.bindings[index]
                {
                    DescriptorResource::CombinedImageSampler {
                        view,
                        layout,
                        sampler: sampler.clone(),
                    }
                } else {
                    DescriptorResource::Image { view, layout }
                }
            }
            HostBinding::Sampler(sampler) => DescriptorResource::Sampler(sampler.clone()),
        };
        if let Ok(slot) = program
            .pipeline
            .layout
            .dynamic_bindings
            .binary_search(&index)
            && let DescriptorResource::Buffer { offset, .. } = &mut resource
        {
            *offset -= u64::from(program.dynamic_offsets[slot]);
        }
        Ok(resource)
    }
    fn canonical_resources(
        &self,
        program: &HostProgram,
        resources: &mut Vec<DescriptorResource>,
    ) -> Result<(), VulkanError> {
        for index in 0..program.bindings.len() {
            resources.push(self.canonical_resource(program, index)?);
        }
        Ok(())
    }
    /// Host-only sampled resources are synchronized by their concrete owner;
    /// they have no invented canonical guest-memory access declarations.
    pub(crate) fn prepare_host_graphics(
        &self,
        pipeline: NativePipeline,
        record: Arc<BindingRecord>,
        parameters: GraphicsParameters,
    ) -> Result<HostGraphics, VulkanError> {
        let dynamic_offsets = vec![0; pipeline.layout.dynamic_bindings.len()];
        self.prepare_host_pipeline(
            HostProgram {
                binding_revision: self.texture_cache.binding_revision.into(),
                pipeline,
                dynamic_offsets,
                record: std::cell::RefCell::new(ProgramRecord {
                    current: Some(record),
                    streamed: false,
                }),
                bindings: Vec::new(),
                unaligned_uniforms: 0,
            },
            parameters,
        )
    }
    pub(crate) fn bind_host_graphics(
        &mut self,
        graphics: &HostGraphics,
        push: &[u8],
    ) -> Result<(), VulkanError> {
        self.runtime.device.end_rendering()?;
        self.graphics_state(&graphics.program.pipeline, &graphics.parameters)?;
        self.set_feedback_loop(vk::ImageAspectFlags::empty());
        self.bind_host_program(&graphics.program, push).map(|_| ())
    }
    fn set_feedback_loop(&mut self, aspects: vk::ImageAspectFlags) {
        if !self.runtime.device.profile().attachment_feedback_loop {
            return;
        }
        let slot = &mut self.runtime.device.submission.slots[self
            .runtime
            .device
            .submission
            .recording
            .expect("active graphics recording")];
        if slot.feedback_aspects == Some(aspects) {
            return;
        }
        unsafe {
            self.runtime.device.dispatch.cmd_set_feedback_loop(aspects);
        }
        slot.feedback_aspects = Some(aspects);
    }
    pub(crate) fn host_program_current(&self, program: &HostProgram) -> bool {
        if program.binding_revision.load(Ordering::Relaxed) == self.texture_cache.binding_revision {
            return true;
        }
        if !program
            .bindings
            .iter()
            .all(|binding| self.input_binding_current(binding))
        {
            return false;
        }
        program
            .binding_revision
            .store(self.texture_cache.binding_revision, Ordering::Relaxed);
        true
    }
    fn input_binding_current(&self, binding: &HostBinding) -> bool {
        match binding {
            HostBinding::Buffer { buffer, .. } | HostBinding::TexelBuffer { buffer, .. } => self
                .buffer_cache
                .buffers
                .get(buffer.id)
                .is_some_and(|entry| {
                    entry.binding_valid && Arc::ptr_eq(&entry.native, &buffer.native)
                }),
            HostBinding::Image(view) | HostBinding::CombinedImageSampler { image: view, .. } => {
                self.texture_cache.views.get(view.id).is_some_and(|entry| {
                    self.texture_cache.binding_current(entry.image)
                        && Arc::ptr_eq(&entry.native, &view.native)
                })
            }
            _ => true,
        }
    }
    fn stream_graphics(&mut self, program: &HostProgram) -> Result<DescriptorQueue, VulkanError> {
        self.runtime.streamed_uses.clear();
        if program.bindings.is_empty() || program.pipeline.layout.uniform_bindings.is_empty() {
            // Host descriptors and layouts without uniforms need no per-use
            // uniform selection.
            return Ok(DescriptorQueue {
                dynamic_offsets: None,
                record: None,
                update: None,
            });
        }
        let mut updates = std::mem::take(&mut self.runtime.uniform_updates);
        // The layout already assigns each uniform its descriptor slot. Keep a
        // direct indexed selection through streaming, realignment and binding,
        // as Eden's BindHostGraphicsUniformBuffer does for its binding_index.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L932
        updates.resize_with(program.pipeline.layout.uniform_bindings.len(), || None);
        debug_assert!(updates.iter().all(Option::is_none));
        // Eden selects streaming from this uniform range's GPU modifications.
        // Read-only aliases and adjacent bindings do not change its contents.
        // stream_buffer checks exact producers before capturing RAM; prior GPU
        // writes still use the native buffer or an aligned device-side copy.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L932
        for (slot, uniform) in program.pipeline.layout.uniform_bindings.iter().enumerate() {
            let Some(HostBindingAccess::Buffer {
                id,
                offset,
                size,
                scope,
            }) = self.host_binding_usage(program, uniform.resource)
            else {
                continue;
            };
            if size > 64 * 1024 {
                continue;
            }
            let request = buffer::BufferStreamRequest {
                id,
                offset,
                size,
                alignment: scope.buffer_alignment(&self.runtime.device.profile().limits),
            };
            self.runtime.buffer_bindings.channel.uniform_cache_shots[0] += 1;
            if request.offset.is_multiple_of(request.alignment)
                && request.size > self.runtime.buffer_bindings.channel.uniform_stream_limit
            {
                continue;
            }
            let Some(update) = self.stream_buffer(request)? else {
                continue;
            };
            updates[slot] = Some(update);
            self.runtime.streamed_uses.push(uniform.resource);
        }
        let uniform_alignment = self
            .runtime
            .device
            .profile()
            .limits
            .min_uniform_buffer_offset_alignment;
        if program.unaligned_uniforms != 0 {
            for (index, slot) in program.pipeline.layout.uniform_bindings.iter().enumerate() {
                let HostBinding::Buffer {
                    buffer: id,
                    offset: source,
                    size,
                } = &program.bindings[slot.resource]
                else {
                    continue;
                };
                let offset = updates[index]
                    .as_ref()
                    .map_or(*source, |update| update.offset);
                if !offset.is_multiple_of(uniform_alignment) {
                    updates[index] = Some(self.aligned_uniform_copy(id.id, *source, *size)?);
                }
            }
        }
        let mut dynamic_offsets = std::mem::take(&mut self.runtime.descriptor_offsets);
        dynamic_offsets.clear();
        dynamic_offsets.extend_from_slice(&program.dynamic_offsets);
        let mut selected = program.record.borrow_mut();
        let mut resources = std::mem::take(&mut self.runtime.descriptor_resources);
        resources.clear();
        if selected.current.is_none() {
            self.canonical_resources(program, &mut resources)?;
        }
        let limits = &self.runtime.device.profile().limits;
        // Uniform slots update offsets directly. Clone immutable images,
        // samplers and other buffers only when a native descriptor base changes.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
        for (index, slot) in program.pipeline.layout.uniform_bindings.iter().enumerate() {
            // Consume the selection now so no streamed buffer survives into a
            // later draw or a program with a different uniform layout.
            let update = updates[index].take();
            let canonical = &program.bindings[slot.resource];
            let (buffer, mut offset, range) = match canonical {
                HostBinding::Buffer {
                    buffer: id,
                    offset,
                    size,
                } => {
                    let native = &self.buffer_cache.get(id.id).native;
                    let (buffer, offset) = update.as_ref().map_or((native, *offset), |update| {
                        (
                            self.runtime.device.submission.staging.buffer(update),
                            update.offset,
                        )
                    });
                    (Some(buffer), offset, *size)
                }
                HostBinding::NullBuffer => (None, 0, 0),
                _ => {
                    return Err(VulkanError::InvalidResource(
                        "uniform descriptor requires a buffer",
                    ));
                }
            };
            if let Some(dynamic) = slot.dynamic_offset {
                let (base, value) = if let Some(buffer) = buffer {
                    program.pipeline.layout.split_buffer_offset(
                        slot.resource,
                        buffer.size(),
                        offset,
                        range,
                        limits,
                    )?
                } else {
                    (0, 0)
                };
                dynamic_offsets[dynamic] = value;
                offset = base;
            }
            let unchanged = selected.current.as_ref().is_some_and(|record| {
                match (&record.resources[slot.resource], buffer) {
                    (
                        DescriptorResource::Buffer {
                            buffer: old,
                            offset: previous,
                            range: length,
                        },
                        Some(buffer),
                    ) => Arc::ptr_eq(old, buffer) && *previous == offset && *length == range,
                    (DescriptorResource::NullBuffer, None) => true,
                    _ => false,
                }
            });
            if !unchanged && resources.is_empty() {
                // Earlier slots already match this record. Starting with its
                // selection preserves their streamed bases without another pass.
                resources.extend_from_slice(&selected.current.as_ref().unwrap().resources);
            }
            if !resources.is_empty() {
                resources[slot.resource] = if let Some(buffer) = buffer {
                    DescriptorResource::Buffer {
                        buffer: Arc::clone(buffer),
                        offset,
                        range,
                    }
                } else {
                    DescriptorResource::NullBuffer
                };
            }
        }
        let changed = !resources.is_empty();
        let result = if changed {
            self.native_descriptors(&program.pipeline.layout, &resources)
        } else {
            self.runtime
                .device
                .submission
                .statistics
                .descriptor_payload_hits += 1;
            Ok(Arc::clone(selected.current.as_ref().unwrap()))
        };
        resources.clear();
        debug_assert!(updates.iter().all(Option::is_none));
        self.runtime.descriptor_resources = resources;
        self.runtime.uniform_updates = updates;
        let record = result?;
        if changed {
            selected.current = Some(Arc::clone(&record));
        }
        selected.streamed = true;
        Ok(DescriptorQueue {
            dynamic_offsets: Some(dynamic_offsets),
            record: Some(record),
            update: None,
        })
    }
    fn host_program_access(&mut self, program: &HostProgram) -> Result<(), VulkanError> {
        // Synchronize current cache selections before recording shader access.
        // Overlapping bindings share the cache-owned modified ranges.
        let mut images: smallvec::SmallVec<[ImageId; 16]> = smallvec::SmallVec::new();
        for index in 0..program.bindings.len() {
            if self.runtime.streamed_uses.binary_search(&index).is_ok() {
                continue;
            }
            let Some(resource) = self.host_binding_usage(program, index) else {
                continue;
            };
            match resource {
                HostBindingAccess::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } => {
                    let uploaded = self.runtime.statistics.cpu_uploaded_bytes;
                    self.ensure_buffer_range(id, offset, size)?;
                    if scope.access == vk::AccessFlags2::UNIFORM_READ
                        && size <= 64 * 1024
                        && self.runtime.statistics.cpu_uploaded_bytes == uploaded
                    {
                        self.runtime.buffer_bindings.channel.uniform_cache_hits[0] += 1;
                    }
                }
                HostBindingAccess::Image { id, .. } => {
                    if !images.contains(&id) {
                        self.prepare_image(id, false, false)?;
                        images.push(id);
                    }
                }
            }
        }
        for index in 0..program.bindings.len() {
            if self.runtime.streamed_uses.binary_search(&index).is_ok() {
                continue;
            }
            let Some(resource) = self.host_binding_usage(program, index) else {
                continue;
            };
            match resource {
                HostBindingAccess::Buffer {
                    id, offset, size, ..
                } => {
                    self.use_buffer(id, offset, size)?;
                }
                HostBindingAccess::Image { id, scope, .. } => {
                    if scope.writes() {
                        self.mark_image_modified(id)?;
                    }
                    self.shader_image(id)?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn host_program_written(
        &mut self,
        program: &HostProgram,
    ) -> Result<(), VulkanError> {
        if program.bindings.is_empty() {
            return Ok(());
        }
        if !program.pipeline.shader_writes {
            return Ok(());
        }
        for index in 0..program.bindings.len() {
            if self.runtime.streamed_uses.binary_search(&index).is_ok() {
                continue;
            }
            match self.host_binding_usage(program, index) {
                Some(HostBindingAccess::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                }) if scope.writes() => self.finish_buffer_write(id, offset, size)?,
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn bind_host_program(
        &mut self,
        program: &HostProgram,
        push: &[u8],
    ) -> Result<(), VulkanError> {
        let record = self.host_descriptor_record(program)?;
        self.runtime.device.configure_pipeline(
            &program.pipeline,
            push,
            record.as_ref(),
            &program.dynamic_offsets,
            None,
        )
    }
    fn host_descriptor_record(
        &mut self,
        program: &HostProgram,
    ) -> Result<Option<Arc<BindingRecord>>, VulkanError> {
        let mut selected = program.record.borrow_mut();
        if !selected.streamed
            && let Some(record) = &selected.current
        {
            return Ok(Some(Arc::clone(record)));
        }
        if program.bindings.is_empty() {
            return Ok(None);
        }
        let mut resources = std::mem::take(&mut self.runtime.descriptor_resources);
        resources.clear();
        self.canonical_resources(program, &mut resources)?;
        let result = self.native_descriptors(&program.pipeline.layout, &resources);
        resources.clear();
        self.runtime.descriptor_resources = resources;
        let record = result?;
        selected.current = Some(Arc::clone(&record));
        selected.streamed = false;
        Ok(Some(record))
    }
    pub(crate) fn retain_host_program(&mut self, program: &HostProgram) {
        for binding in &program.bindings {
            match binding {
                HostBinding::Buffer { buffer: id, .. }
                | HostBinding::TexelBuffer { buffer: id, .. } => {
                    self.runtime.constructing.push(Resource::Buffer(id.id))
                }
                HostBinding::Image(id) | HostBinding::CombinedImageSampler { image: id, .. } => {
                    self.runtime.constructing.push(Resource::Image(id.image))
                }
                _ => {}
            }
        }
    }
    fn native_descriptors(
        &mut self,
        layout: &Arc<crate::cache::PipelineLayout>,
        resources: &[DescriptorResource],
    ) -> Result<Arc<BindingRecord>, VulkanError> {
        let descriptors = &mut self.runtime.descriptors;
        let before = descriptors.hits();
        let result = descriptors.get(layout, resources);
        let hits = descriptors.hits() - before;
        self.runtime
            .device
            .submission
            .statistics
            .descriptor_payload_hits += hits;
        self.runtime
            .device
            .submission
            .statistics
            .descriptor_payload_captures += u64::from(result.is_ok() && hits == 0);
        result
    }
    pub fn dispatch(
        &mut self,
        program: &HostProgram,
        dispatch: Dispatch,
        push: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        let retained = self.runtime.constructing.len();
        self.retain_host_program(program);
        if let Dispatch::Indirect { buffer, .. } = dispatch {
            self.runtime.constructing.push(Resource::Buffer(buffer));
        }
        let result = self.dispatch_inner(program, dispatch, push);
        self.runtime.constructing.truncate(retained);
        result
    }
    fn dispatch_inner(
        &mut self,
        program: &HostProgram,
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
        }
        let streams = self.stream_graphics(program)?;
        self.host_program_access(program)?;
        let indirect = match dispatch {
            Dispatch::Indirect { buffer, offset } => {
                Some(self.indirect_buffer(buffer, offset, 12)?)
            }
            _ => None,
        };
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let record = match streams.record {
            Some(record) => Some(record),
            None => self.host_descriptor_record(program)?,
        };
        self.runtime.device.configure_pipeline(
            &program.pipeline,
            push,
            record.as_ref(),
            streams
                .dynamic_offsets
                .as_deref()
                .unwrap_or(&program.dynamic_offsets),
            streams.update,
        )?;
        if let Some(offsets) = streams.dynamic_offsets {
            self.runtime.descriptor_offsets = offsets;
        }
        {
            let device = &mut self.runtime.device;
            unsafe {
                match dispatch {
                    Dispatch::Direct([x, y, z]) => device.dispatch.cmd_dispatch(x, y, z),
                    Dispatch::Indirect { offset, .. } => device
                        .dispatch
                        .cmd_dispatch_indirect(indirect.unwrap(), offset),
                }
            }
        };
        self.host_program_written(program)?;
        Ok(self.runtime.device.current_work())
    }
    pub(super) fn indirect_buffer(
        &mut self,
        buffer: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<vk::Buffer, VulkanError> {
        if !offset.is_multiple_of(4)
            || !self
                .buffer_cache
                .get(buffer)
                .usage
                .contains(vk::BufferUsageFlags::INDIRECT_BUFFER)
        {
            return Err(VulkanError::InvalidResource(
                "indirect execution requires aligned INDIRECT_BUFFER storage",
            ));
        }
        self.ensure_buffer_range(buffer, offset, size)?;
        self.use_buffer(buffer, offset, size)
    }
}
