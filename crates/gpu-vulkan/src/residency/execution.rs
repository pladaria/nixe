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
pub(crate) use graphics::{HostDrawState, MaskedClearPipelines, NativeIndexBuffer};
pub(crate) use state::DynamicValues;
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

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{AllocationRange, Machine, PAGE_SIZE, device_memory::DeviceMemory};

    mod support {
        use crate as nixe_gpu_vulkan;
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/device.rs"
        ));
    }

    #[test]
    fn indirect_instance_uniform_reads_each_gpu_produced_argument_version()
    -> Result<(), Box<dyn std::error::Error>> {
        let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
            return Ok(());
        };
        let validation = device.validation_monitor();
        let machine = Machine::new(16 << 20)?;
        let allocation = machine.account(4 << 20)?.allocate(2 * PAGE_SIZE)?;
        let memory = DeviceMemory::new(machine, 32);
        let address = 0x10000;
        let _mapping = memory.map(
            address,
            Arc::from([AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE)?]),
            true,
        )?;
        let mut cache =
            VulkanResidency::new(memory, device, 64 << 20, BufferDownloadPolicy::OnRead)?;
        let mut r = cache.access();
        let size = size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let args = r.device_buffer(
            address + 32,
            size,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
            false,
            false,
        )?;
        let template = [0x5a; 0x650];
        let output = r.device_buffer(
            address + PAGE_SIZE,
            2 * template.len() as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            false,
            false,
        )?;
        let draw = Draw::Indirect {
            buffer: args.buffer,
            offset: args.offset,
            count: 1,
            stride: size as u32,
            indexed: true,
        };
        r.write_buffer(
            args.buffer,
            args.offset,
            bytemuck::cast_slice(&[6u32, 1, 0, 0, 0]),
        )?;
        let first_instance = std::mem::offset_of!(vk::DrawIndexedIndirectCommand, first_instance);
        let field = 0x644;
        let values = [37u32, 4097];
        let mut joined = Vec::new();
        for (index, value) in values.into_iter().enumerate() {
            r.fill_buffer(args.buffer, args.offset + first_instance as u64, 4, value)?;
            let slice = r.stream_indirect_instance_uniform(&template, field, Some(draw))?;
            let offset = output.offset + (index * template.len()) as u64;
            let target = r.use_buffer(output.buffer, offset, slice.size)?;
            unsafe {
                r.runtime.device.dispatch.cmd_copy_buffer(
                    slice.buffer,
                    target,
                    &[vk::BufferCopy::default()
                        .src_offset(slice.offset)
                        .dst_offset(offset)
                        .size(slice.size)],
                    true,
                );
            }
            r.finish_buffer_write(output.buffer, offset, slice.size)?;
            let mut expected = template;
            expected[field..field + 4].copy_from_slice(&value.to_le_bytes());
            joined.extend_from_slice(&expected);
        }
        assert_eq!(
            r.read_buffer(output.buffer, output.offset, output.size)?,
            joined
        );
        drop(r);
        drop(cache);
        validation.check()?;
        Ok(())
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
    /// Immutable host constants streamed when the command is recorded.
    UniformBytes(Arc<[u8]>),
    /// Host constants with the base instance supplied by one indexed indirect draw.
    UniformBytesWithIndirectInstance {
        bytes: Arc<[u8]>,
        offset: usize,
    },
    NullBuffer,
    PointerTable(Arc<super::pointer_table::PointerTableBinding>),
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
enum HostBindingAccess<'a> {
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
    PointerTable(&'a [BufferRange]),
}
/// Host utility shader captures. Guest Configure uses cache-owned hardware
/// banks and DescriptorQueue directly; it never constructs this object.
pub struct HostProgram {
    binding_revision: AtomicU64,
    pub(crate) pipeline: NativePipeline,
    // Explicit host bindings are owned by the host texture cache.
    record: std::cell::RefCell<ProgramRecord>,
    // Explicit host resources use owned captures; guest resources use residency
    // use ticks and the native execution owner's persistent sampler cache.
    bindings: Vec<HostBinding>,
    direct_entries: Option<std::cell::RefCell<Vec<crate::cache::DescriptorEntry>>>,
    dynamic_offsets: Vec<u32>,
}
impl HostProgram {
    pub(crate) fn image_binding_view(&self, index: usize) -> Option<ImageViewId> {
        match self.bindings.get(index)? {
            HostBinding::Image(image) | HostBinding::CombinedImageSampler { image, .. } => {
                Some(image.id)
            }
            _ => None,
        }
    }
}
#[derive(Default)]
struct ProgramRecord {
    payload: Option<(u64, u64, crate::cache::DescriptorUpdate)>,
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
    fn host_binding_usage(program: &HostProgram, index: usize) -> Option<HostBindingAccess<'_>> {
        let scope = program.pipeline.resources[index];
        if let HostBinding::PointerTable(binding) = &program.bindings[index] {
            return Some(HostBindingAccess::PointerTable(&binding.targets));
        }
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
            HostBinding::NullBuffer
            | HostBinding::UniformBytes(_)
            | HostBinding::UniformBytesWithIndirectInstance { .. }
            | HostBinding::Sampler(_)
            | HostBinding::PointerTable(_) => None,
        }
    }
    fn canonical_resource(
        &self,
        program: &HostProgram,
        index: usize,
    ) -> Result<DescriptorResource, VulkanError> {
        let mut resource = match &program.bindings[index] {
            HostBinding::NullBuffer
            | HostBinding::UniformBytes(_)
            | HostBinding::UniformBytesWithIndirectInstance { .. } => {
                DescriptorResource::NullBuffer
            }
            HostBinding::Buffer {
                buffer,
                offset,
                size,
            } => DescriptorResource::Buffer {
                buffer: Arc::clone(&buffer.native),
                offset: *offset,
                range: *size,
            },
            HostBinding::PointerTable(binding) => DescriptorResource::Buffer {
                buffer: Arc::clone(&binding.table),
                offset: binding.offset,
                range: binding.size,
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
                    ..Default::default()
                }),
                bindings: Vec::new(),
                direct_entries: None,
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
            HostBinding::PointerTable(binding) => binding.targets.iter().all(|range| {
                self.buffer_cache
                    .buffers
                    .get(range.buffer)
                    .is_some_and(|entry| entry.binding_valid)
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
    fn nvn_payload(
        &mut self,
        entries: &[crate::cache::DescriptorEntry],
        record: &mut ProgramRecord,
    ) -> Result<crate::cache::DescriptorUpdate, VulkanError> {
        let tick = self.runtime.device.current_work().tick();
        let queue = &mut self.runtime.descriptor_queue;
        let stats = &mut self.runtime.device.submission.statistics;
        if let Some((previous, generation, data)) = record.payload
            && previous == tick
            && generation == queue.generation()
        {
            stats.descriptor_payload_hits += 1;
            return Ok(data);
        }
        queue.acquire(&mut self.runtime.device.dispatch, entries.len())?;
        queue.write_entries(0, entries);
        let data = queue.update_data();
        record.payload = Some((tick, queue.generation(), data));
        stats.descriptor_payload_captures += 1;
        Ok(data)
    }
    fn stream_graphics(
        &mut self,
        program: &HostProgram,
        draw: Option<Draw>,
    ) -> Result<DescriptorQueue, VulkanError> {
        self.runtime.streamed_uses.clear();
        if program.bindings.is_empty() || program.pipeline.layout.uniform_bindings.is_empty() {
            // Host descriptors and layouts without uniforms need no per-use
            // uniform selection.
            if let Some(entries) = &program.direct_entries
                && !entries.borrow().is_empty()
            {
                let update =
                    self.nvn_payload(&entries.borrow(), &mut program.record.borrow_mut())?;
                return Ok(DescriptorQueue {
                    dynamic_offsets: None,
                    record: None,
                    update: Some(update),
                });
            }
            return Ok(DescriptorQueue {
                dynamic_offsets: None,
                record: None,
                update: None,
            });
        }
        let nvn = program.direct_entries.is_some();
        self.prepare_host_uniforms(program.pipeline.layout.uniform_bindings.len(), nvn);
        let uniform_alignment = self
            .runtime
            .device
            .profile()
            .limits
            .min_uniform_buffer_offset_alignment;
        let mut dynamic_offsets = std::mem::take(&mut self.runtime.descriptor_offsets);
        dynamic_offsets.clear();
        dynamic_offsets.extend_from_slice(&program.dynamic_offsets);
        let mut direct = program
            .direct_entries
            .as_ref()
            .map(|entries| entries.borrow_mut());
        let mut selected = program.record.borrow_mut();
        let mut resources = std::mem::take(&mut self.runtime.descriptor_resources);
        resources.clear();
        if direct.is_none() && selected.current.is_none() {
            self.canonical_resources(program, &mut resources)?;
        }
        // Uniform slots update offsets directly. Clone immutable images,
        // samplers and other buffers only when a native descriptor base changes.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
        for (index, slot) in program.pipeline.layout.uniform_bindings.iter().enumerate() {
            let (native, source_offset, range, update) = match &program.bindings[slot.resource] {
                HostBinding::Buffer {
                    buffer,
                    offset,
                    size,
                } => {
                    let scope = program.pipeline.resources[slot.resource];
                    let mut update = None;
                    if !scope.stages.is_empty() && *size <= 64 * 1024 {
                        let alignment =
                            scope.buffer_alignment(&self.runtime.device.profile().limits);
                        self.runtime.buffer_bindings.channel.uniform_cache_shots[0] += 1;
                        if !offset.is_multiple_of(alignment)
                            || *size <= self.runtime.buffer_bindings.channel.uniform_stream_limit
                        {
                            update = self.stream_buffer(
                                index,
                                buffer::BufferStreamRequest {
                                    id: buffer.id,
                                    offset: *offset,
                                    size: *size,
                                    alignment,
                                },
                                nvn,
                            )?;
                            if update.is_some() {
                                self.runtime.streamed_uses.push(slot.resource);
                            }
                        }
                    }
                    if !update
                        .as_ref()
                        .map_or(*offset, |slice| slice.offset)
                        .is_multiple_of(uniform_alignment)
                    {
                        update = Some(self.aligned_uniform_copy(buffer.id, *offset, *size)?);
                    }
                    (Some(&buffer.native), *offset, *size, update)
                }
                HostBinding::UniformBytes(bytes) => {
                    let tick = self.runtime.device.current_work().tick();
                    let kept = &mut self.buffer_cache.uniform_streams.host[index].immutable;
                    let slice = if let Some((source, previous, slice)) = kept
                        && *previous == tick
                        && Arc::ptr_eq(source, bytes)
                    {
                        self.runtime.statistics.uniform_keeps += 1;
                        *slice
                    } else {
                        let slice = self
                            .runtime
                            .device
                            .submission
                            .reserve_upload_aligned(bytes, uniform_alignment)?;
                        *kept = Some((bytes.clone(), tick, slice));
                        slice
                    };
                    (None, 0, bytes.len() as u64, Some(slice))
                }
                HostBinding::UniformBytesWithIndirectInstance { bytes, offset } => {
                    let slice = self.stream_indirect_instance_uniform(bytes, *offset, draw)?;
                    (None, 0, bytes.len() as u64, Some(slice))
                }
                HostBinding::NullBuffer => (None, 0, 0, None),
                _ => {
                    return Err(VulkanError::InvalidResource(
                        "uniform descriptor requires a buffer",
                    ));
                }
            };
            let mut offset = update.as_ref().map_or(source_offset, |slice| slice.offset);
            if let Some(entries) = direct.as_mut()
                && slot.dynamic_offset.is_none()
            {
                let handle = update.as_ref().map_or_else(
                    || native.map_or(vk::Buffer::null(), |buffer| buffer.handle),
                    |slice| slice.buffer,
                );
                Self::select_uniform_entry(
                    &mut entries[slot.resource],
                    &mut selected,
                    handle,
                    offset,
                    range,
                );
                continue;
            }
            let buffer = update.as_ref().map_or(native, |slice| {
                Some(self.runtime.device.submission.staging.buffer(slice))
            });
            if let Some(dynamic) = slot.dynamic_offset {
                let (base, value) = if let Some(buffer) = buffer {
                    program.pipeline.layout.split_buffer_offset(
                        slot.resource,
                        buffer.size(),
                        offset,
                        range,
                        &self.runtime.device.profile().limits,
                    )?
                } else {
                    (0, 0)
                };
                dynamic_offsets[dynamic] = value;
                offset = base;
            }
            if let Some(entries) = direct.as_mut() {
                let handle = buffer.map_or(vk::Buffer::null(), |buffer| buffer.handle);
                Self::select_uniform_entry(
                    &mut entries[slot.resource],
                    &mut selected,
                    handle,
                    offset,
                    range,
                );
                continue;
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
        if let Some(entries) = direct {
            self.runtime.descriptor_resources = resources;
            let update = self.nvn_payload(&entries, &mut selected)?;
            return Ok(DescriptorQueue {
                dynamic_offsets: Some(dynamic_offsets),
                record: None,
                update: Some(update),
            });
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
        self.runtime.descriptor_resources = resources;
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
    fn select_uniform_entry(
        entry: &mut crate::cache::DescriptorEntry,
        record: &mut ProgramRecord,
        handle: vk::Buffer,
        offset: u64,
        range: u64,
    ) {
        let range = if handle == vk::Buffer::null() {
            vk::WHOLE_SIZE
        } else {
            range
        };
        // SAFETY: uniform slots are populated with buffer info during binding capture.
        let previous = unsafe { entry.buffer };
        if (previous.buffer, previous.offset, previous.range) != (handle, offset, range) {
            *entry = crate::cache::DescriptorEntry {
                buffer: vk::DescriptorBufferInfo::default()
                    .buffer(handle)
                    .offset(offset)
                    .range(range),
            };
            record.payload = None;
        }
    }
    fn host_program_access(
        &mut self,
        program: &HostProgram,
        images_prepared: bool,
    ) -> Result<(), VulkanError> {
        if program.bindings.is_empty() {
            return Ok(());
        }
        let mut streamed = std::mem::take(&mut self.runtime.streamed_uses);
        let result = self.synchronize_host_bindings(program, &streamed, images_prepared);
        streamed.clear();
        self.runtime.streamed_uses = streamed;
        result
    }
    fn synchronize_host_bindings(
        &mut self,
        program: &HostProgram,
        streamed: &[usize],
        images_prepared: bool,
    ) -> Result<(), VulkanError> {
        let accesses = || {
            let mut streamed = streamed.iter().copied().peekable();
            program
                .pipeline
                .access_bindings
                .iter()
                .filter_map(move |&index| {
                    if streamed.next_if_eq(&index).is_some() {
                        None
                    } else {
                        Self::host_binding_usage(program, index)
                    }
                })
        };
        // Finish imports before marking uses. Overlapping bindings can share
        // pending uploads, and preparing a later alias must precede shader writes.
        let mut images: smallvec::SmallVec<[ImageId; 16]> = smallvec::SmallVec::new();
        for resource in accesses() {
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
                HostBindingAccess::PointerTable(ranges) => {
                    for range in ranges {
                        self.ensure_buffer_range(range.buffer, range.offset, range.size)?;
                    }
                }
                HostBindingAccess::Image { id, .. } => {
                    if !images_prepared && !images.contains(&id) {
                        self.prepare_image(id, false, false)?;
                        images.push(id);
                    }
                }
            }
        }
        for resource in accesses() {
            match resource {
                HostBindingAccess::Buffer {
                    id, offset, size, ..
                } => {
                    self.use_buffer(id, offset, size)?;
                }
                HostBindingAccess::PointerTable(ranges) => {
                    for range in ranges {
                        self.use_buffer(range.buffer, range.offset, range.size)?;
                    }
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
        for &index in &program.pipeline.written_buffers {
            match Self::host_binding_usage(program, index) {
                Some(HostBindingAccess::Buffer {
                    id, offset, size, ..
                }) => self.finish_buffer_write(id, offset, size)?,
                Some(HostBindingAccess::PointerTable(ranges)) => {
                    for range in ranges {
                        self.finish_buffer_write(range.buffer, range.offset, range.size)?;
                    }
                }
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
                HostBinding::PointerTable(binding) => {
                    self.runtime.constructing.extend(
                        binding
                            .targets
                            .iter()
                            .map(|range| Resource::Buffer(range.buffer)),
                    );
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
        self.dispatch_bound(program, dispatch, push, false)
    }
    pub(crate) fn dispatch_bound(
        &mut self,
        program: &HostProgram,
        dispatch: Dispatch,
        push: &[u8],
        images_prepared: bool,
    ) -> Result<WorkTicket, VulkanError> {
        let retained = self.runtime.constructing.len();
        self.retain_host_program(program);
        if let Dispatch::Indirect { buffer, .. } = dispatch {
            self.runtime.constructing.push(Resource::Buffer(buffer));
        }
        let result = self.dispatch_inner(program, dispatch, push, images_prepared);
        self.runtime.constructing.truncate(retained);
        result
    }
    fn dispatch_inner(
        &mut self,
        program: &HostProgram,
        dispatch: Dispatch,
        push: &[u8],
        images_prepared: bool,
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
        let streams = self.stream_graphics(program, None)?;
        self.host_program_access(program, images_prepared)?;
        let indirect = match dispatch {
            Dispatch::Indirect { buffer, offset } => {
                Some(self.indirect_buffer(buffer, offset, 12)?)
            }
            _ => None,
        };
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let record = match streams.record {
            Some(record) => Some(record),
            None if streams.update.is_some() => None,
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

    fn stream_indirect_instance_uniform(
        &mut self,
        bytes: &[u8],
        offset: usize,
        draw: Option<Draw>,
    ) -> Result<crate::queue::Slice, VulkanError> {
        let Some(Draw::Indirect {
            buffer,
            offset: arguments,
            count: 1,
            indexed: true,
            ..
        }) = draw
        else {
            return Err(VulkanError::InvalidResource(
                "indirect instance uniform requires one indexed indirect draw",
            ));
        };
        let instance_offset = std::mem::offset_of!(vk::DrawIndexedIndirectCommand, first_instance);
        let field_size = size_of::<u32>() as u64;
        self.runtime.device.end_rendering()?;
        let source = self.indirect_buffer(
            buffer,
            arguments,
            size_of::<vk::DrawIndexedIndirectCommand>() as u64,
        )?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let device = &mut self.runtime.device;
        let alignment = device.profile().limits.min_uniform_buffer_offset_alignment;
        // Each draw gets a new slice. Its arguments may be produced on the GPU
        // or overwritten by an intervening command at the same address.
        let slice = device
            .submission
            .reserve_upload_aligned(bytes, alignment.max(4))?;
        unsafe {
            device.dispatch.cmd_copy_buffer(
                source,
                slice.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(arguments + instance_offset as u64)
                    .dst_offset(slice.offset + offset as u64)
                    .size(field_size)],
                true,
            );
        }
        self.runtime.statistics.cpu_streamed_bytes += bytes.len() as u64;
        self.runtime.statistics.gpu_copied_bytes += field_size;
        self.runtime.statistics.uniform_uploads += 1;
        Ok(slice)
    }
}
