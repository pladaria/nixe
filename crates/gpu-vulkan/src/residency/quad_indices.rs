//! Buffer-cache runtime index assembly, independent of shader compilation.
//! Array draws reuse the native LUT; indexed draws assemble in device-local
//! staging with a compute pass. No guest index bytes are downloaded to the CPU.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_buffer_cache.cpp
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_compute_pass.cpp
use super::execution::{Draw, NativeIndexBuffer};
use super::*;
use crate::cache::{
    DescriptorEntry, NativePipeline, PipelineCompiler, PipelineRecipe, ShaderArtifact,
};
use crate::queue::staging::Usage;
use nixe_gpu::{
    ShaderStage,
    shader_module::{Binding, DescriptorKind, ResourceId},
};
use std::{sync::OnceLock, time::Duration};

#[derive(Default)]
pub(super) struct QuadIndices {
    arrays: [Option<QuadLut>; 2],
    compiler: Option<Arc<PipelineCompiler>>,
    pipeline: Option<Arc<NativePipeline>>,
}
struct QuadLut {
    buffer: AllocatedBuffer,
    vertices: u32,
    quads: u32,
    kind: vk::IndexType,
    bytes: u64,
}
fn quads(vertices: u32, strip: bool) -> u32 {
    if strip {
        vertices.saturating_sub(2) / 2
    } else {
        vertices / 4
    }
}
fn stride(kind: vk::IndexType) -> u64 {
    match kind {
        vk::IndexType::UINT8_EXT => 1,
        vk::IndexType::UINT16 => 2,
        _ => 4,
    }
}
fn artifact() -> Arc<ShaderArtifact> {
    static ARTIFACT: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    Arc::clone(ARTIFACT.get_or_init(|| {
        Arc::new(ShaderArtifact::new(Box::new([super::conversion::stage(
            ShaderStage::Compute,
            include_bytes!("../../shaders/spv/quad-indexed.comp.spv"),
            [false, true]
                .into_iter()
                .enumerate()
                .map(|(index, writable)| Binding {
                    resource: ResourceId::new(index as u32),
                    binding: index as u32,
                    kind: DescriptorKind::StorageBuffer,
                    count: 1,
                    sampler: None,
                    readable: !writable,
                    writable,
                })
                .collect(),
            16,
        )])))
    }))
}
impl Residency<'_> {
    pub(crate) fn enable_quad_indices(&mut self, compiler: Arc<PipelineCompiler>) {
        self.runtime.quad_indices.compiler = Some(compiler);
    }
    pub(crate) fn assemble_quads(
        &mut self,
        topology: u32,
        draw: Draw,
        inline: Option<&[u8]>,
    ) -> Result<(Draw, Option<NativeIndexBuffer>), VulkanError> {
        if !matches!(topology, 7 | 8) {
            let index = if !draw.is_empty() {
                inline
                    .map(|bytes| self.inline_index_buffer(bytes))
                    .transpose()?
            } else {
                None
            };
            return Ok((draw, index));
        }
        let strip = topology == 8;
        match draw {
            Draw::Direct {
                vertices,
                first_vertex,
                instances,
                first_instance,
            } => {
                let indices = quads(vertices, strip)
                    .checked_mul(6)
                    .ok_or(VulkanError::InvalidResource("quad index count overflows"))?;
                let binding = if indices == 0 || instances == 0 {
                    None
                } else {
                    Some(self.quad_lut(first_vertex, vertices, strip)?)
                };
                Ok((
                    Draw::Indexed {
                        indices,
                        instances,
                        first_index: 0,
                        vertex_offset: 0,
                        first_instance,
                    },
                    binding,
                ))
            }
            Draw::Indexed {
                indices: vertices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            } => {
                let indices = quads(vertices, strip)
                    .checked_mul(6)
                    .ok_or(VulkanError::InvalidResource("quad index count overflows"))?;
                let binding = if indices == 0 || instances == 0 {
                    None
                } else {
                    Some(self.indexed_quads(
                        first_index,
                        vertices,
                        vertex_offset as u32,
                        indices,
                        strip,
                        inline,
                    )?)
                };
                Ok((
                    Draw::Indexed {
                        indices,
                        instances,
                        first_index: 0,
                        vertex_offset: 0,
                        first_instance,
                    },
                    binding,
                ))
            }
            Draw::Indirect { .. } => Err(VulkanError::InvalidResource(
                "quad indirect assembly requires the native indirect conversion pass",
            )),
        }
    }
    fn inline_index_buffer(&mut self, bytes: &[u8]) -> Result<NativeIndexBuffer, VulkanError> {
        let device = &mut self.runtime.device;
        let slice = device.submission.reserve_upload(bytes)?;
        let buffer = device.submission.staging_buffer(&slice).handle;
        self.runtime.statistics.cpu_streamed_bytes += bytes.len() as u64;
        Ok(NativeIndexBuffer {
            buffer,
            offset: slice.offset,
            size: slice.size,
            kind: vk::IndexType::UINT32,
        })
    }
    fn quad_lut(
        &mut self,
        first: u32,
        count: u32,
        strip: bool,
    ) -> Result<NativeIndexBuffer, VulkanError> {
        let vertices = first
            .checked_add(count)
            .ok_or(VulkanError::InvalidResource("quad vertex range overflows"))?;
        let index = usize::from(strip);
        if self.runtime.quad_indices.arrays[index]
            .as_ref()
            .is_none_or(|lut| vertices > lut.vertices)
        {
            // Eden QuadIndexBuffer::UpdateBuffer finishes the existing scheduler
            // prefix before replacing storage. Warm draws never enter this path.
            let prefix = self.runtime.device.current_work();
            self.flush()?;
            self.runtime.device.wait_for_work(prefix, Duration::MAX)?;
            let kind = if vertices < 256 && self.runtime.device.profile().index_uint8 {
                vk::IndexType::UINT8_EXT
            } else if vertices < 65536 {
                vk::IndexType::UINT16
            } else {
                vk::IndexType::UINT32
            };
            let quads = quads(vertices, strip);
            let bytes = u64::from(quads) * 6 * stride(kind) * 4;
            let buffer = self.runtime.device.allocate_buffer(
                bytes,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                AllocationMode::Managed,
                "quad index LUT",
            )?;
            let fill = |destination: &mut [u8]| -> Result<(), VulkanError> {
                let step = if strip { 2 } else { 4 };
                let corners = if strip {
                    [0u32, 3, 1, 0, 2, 3]
                } else {
                    [0u32, 1, 2, 0, 2, 3]
                };
                let mut offset = 0usize;
                for first in 0..4 {
                    for quad in 0..quads {
                        for corner in corners {
                            let word = first + quad * step + corner;
                            let value = word.to_le_bytes();
                            let size = stride(kind) as usize;
                            destination[offset..offset + size].copy_from_slice(&value[..size]);
                            offset += size;
                        }
                    }
                }
                Ok(())
            };
            if buffer.is_host_visible() {
                unsafe {
                    buffer.write_with(0, bytes, fill)?;
                }
            } else {
                let device = &mut self.runtime.device;
                let upload = device.submission.reserve_slice(bytes, Usage::Upload, 4)?;
                unsafe {
                    device.submission.staging.write_with(&upload, fill)?;
                }
                let source = upload.buffer;
                device.end_rendering()?;
                unsafe {
                    device.dispatch.cmd_copy_buffer(
                        source,
                        buffer.handle,
                        &[vk::BufferCopy::default()
                            .src_offset(upload.offset)
                            .size(bytes)],
                        false,
                    );
                }
                self.index_barrier(
                    buffer.handle,
                    0,
                    bytes,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                );
            }
            self.runtime.quad_indices.arrays[index] = Some(QuadLut {
                buffer,
                vertices,
                quads,
                kind,
                bytes,
            });
        }
        let lut = self.runtime.quad_indices.arrays[index].as_ref().unwrap();
        let step = if strip { 2 } else { 4 };
        // Select the first-vertex remainder table and skip preceding primitives.
        let offset = (u64::from(first % step) * u64::from(lut.quads) + u64::from(first / step))
            * 6
            * stride(lut.kind);
        Ok(NativeIndexBuffer {
            buffer: lut.buffer.handle,
            offset,
            size: lut.bytes - offset,
            kind: lut.kind,
        })
    }
    fn indexed_quads(
        &mut self,
        first: u32,
        vertices: u32,
        base: u32,
        indices: u32,
        strip: bool,
        inline: Option<&[u8]>,
    ) -> Result<NativeIndexBuffer, VulkanError> {
        let index = self
            .runtime
            .buffer_bindings
            .channel
            .index_buffer
            .filter(|_| inline.is_none());
        let (buffer, offset, kind) = match (index, inline) {
            (Some(index), None) => {
                let element = stride(index.kind);
                let offset = u64::from(first) * element;
                let size = u64::from(vertices) * element;
                if offset + size > index.size {
                    return Err(VulkanError::InvalidResource(
                        "quad source exceeds bound index window",
                    ));
                }
                // The compute shader loads packed u32 words even for byte/short
                // indices. Coherence and dependencies cover those actual loads.
                let read_begin = (index.offset + offset) & !3;
                let read_end = (index.offset + offset + size).div_ceil(4) * 4;
                self.ensure_buffer_range(index.buffer, read_begin, read_end - read_begin)?;
                self.use_buffer(index.buffer, read_begin, read_end - read_begin)?;
                (
                    Arc::clone(&self.buffer_cache.get(index.buffer).native),
                    index.offset + offset,
                    index.kind,
                )
            }
            (None, Some(bytes)) => {
                let offset = u64::from(first) * 4;
                let size = u64::from(vertices) * 4;
                if offset + size > bytes.len() as u64 {
                    return Err(VulkanError::InvalidResource(
                        "quad source exceeds inline indices",
                    ));
                }
                let device = &mut self.runtime.device;
                let input = device.submission.reserve_upload(bytes)?;
                let buffer = Arc::clone(device.submission.staging_buffer(&input));
                (buffer, input.offset + offset, vk::IndexType::UINT32)
            }
            _ => {
                return Err(VulkanError::InvalidResource(
                    "quad indexed draw has no unique index stream",
                ));
            }
        };
        let limits = self.runtime.device.profile().limits;
        if limits.max_compute_work_group_invocations < 1024
            || limits.max_compute_work_group_size[0] < 1024
        {
            return Err(VulkanError::InvalidResource(
                "quad compute assembly requires 1024-invocation workgroups",
            ));
        }
        let pipeline = match &self.runtime.quad_indices.pipeline {
            Some(p) => Arc::clone(p),
            None => {
                let compiler = self.runtime.quad_indices.compiler.as_ref().ok_or(
                    VulkanError::InvalidResource(
                        "quad conversion requires the native pipeline compiler",
                    ),
                )?;
                let pipeline = compiler
                    .request(PipelineRecipe::Compute(artifact()))
                    .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
                self.runtime.quad_indices.pipeline = Some(Arc::clone(&pipeline));
                pipeline
            }
        };
        let input_offset = offset / limits.min_storage_buffer_offset_alignment
            * limits.min_storage_buffer_offset_alignment;
        let input_size =
            (offset - input_offset + u64::from(vertices) * stride(kind)).div_ceil(4) * 4;
        let output_size = u64::from(indices) * 4;
        if input_offset + input_size > buffer.size()
            || input_size > u64::from(limits.max_storage_buffer_range)
            || output_size > u64::from(limits.max_storage_buffer_range)
            || indices.div_ceil(1024) > limits.max_compute_work_group_count[0]
        {
            return Err(VulkanError::InvalidResource(
                "quad compute windows exceed native buffer/dispatch limits",
            ));
        }
        let device = &mut self.runtime.device;
        let output = device.submission.reserve_slice(
            output_size,
            Usage::DeviceLocal,
            limits.min_storage_buffer_offset_alignment.max(4),
        )?;
        let output_buffer = device.submission.staging_buffer(&output).handle;
        self.runtime
            .compute_pass_descriptor_queue
            .acquire(&mut device.dispatch, 2)?;
        self.runtime.compute_pass_descriptor_queue.write(
            0,
            DescriptorEntry {
                buffer: vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(input_offset)
                    .range(input_size),
            },
        );
        self.runtime.compute_pass_descriptor_queue.write(
            1,
            DescriptorEntry {
                buffer: vk::DescriptorBufferInfo::default()
                    .buffer(output_buffer)
                    .offset(output.offset)
                    .range(output_size),
            },
        );
        let update = self.runtime.compute_pass_descriptor_queue.update_data();
        let push = [
            base,
            stride(kind).ilog2(),
            u32::from(strip),
            ((offset - input_offset) / stride(kind)) as u32,
        ];
        let push = push.map(u32::to_le_bytes);
        self.runtime.device.realize_deferred_clear()?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        self.runtime.device.end_rendering()?;
        self.bind_pipeline_descriptors(&pipeline, push.as_flattened(), None, &[], Some(update))?;
        let device = &mut self.runtime.device;
        unsafe {
            device.dispatch.cmd_dispatch(indices.div_ceil(1024), 1, 1);
        }
        // Physical source storage stays alive through this compute command even
        // if a subsequent cache merge replaces its logical buffer slot.
        let slot = device.submission.recording.unwrap();
        device.submission.slots[slot].retired_buffers.push(buffer);
        self.index_barrier(
            output_buffer,
            output.offset,
            output_size,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        Ok(NativeIndexBuffer {
            buffer: output_buffer,
            offset: output.offset,
            size: output_size,
            kind: vk::IndexType::UINT32,
        })
    }
    fn index_barrier(
        &mut self,
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
        source: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
        let barrier = vk::BufferMemoryBarrier2::default()
            .buffer(buffer)
            .offset(offset)
            .size(size)
            .src_stage_mask(source)
            .src_access_mask(access)
            .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
            .dst_access_mask(vk::AccessFlags2::INDEX_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        unsafe {
            self.runtime
                .device
                .dispatch
                .barrier(&vk::DependencyInfo::default().buffer_memory_barriers(&[barrier]));
        }
    }
}
