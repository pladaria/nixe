//! Fixed-point NV12 is a host utility, compiled through the ordinary pipeline
//! cache. Decoded inputs are host-only textures; only VIC output owns guest bytes.
use super::*;
use crate::{Draw, GraphicsParameters, PreparedDraw, RenderAttachment, RenderingTargets, cache::*};
use nixe_gpu::{
    Nv12ColorTransform, Nv12Planes, ShaderStage,
    shader_module::{self as spirv, ResourceId},
};
use std::sync::OnceLock;
pub(crate) fn stage(
    stage: ShaderStage,
    bytes: &[u8],
    bindings: Box<[spirv::Binding]>,
    push: u32,
) -> CompiledStage {
    CompiledStage::new(
        stage,
        spirv::Module {
            words: bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            bindings,
            push_constant_bytes: push,
        },
        None,
        0,
        None,
    )
}
pub(crate) struct Nv12Inputs {
    pub(crate) extent: [u32; 2],
    uploaded_extent: [u32; 2],
    views: [Arc<crate::AllocatedImageView>; 2],
    initialized: bool,
}
impl Nv12Inputs {
    pub(crate) fn dimensions(source: &Nv12Planes) -> Result<[u32; 2], VulkanError> {
        let width = source.widths[0];
        if width == 0
            || source.widths[1] != width.div_ceil(2)
            || source.bytes[0].is_empty()
            || !source.bytes[0].len().is_multiple_of(width as usize)
        {
            return Err(VulkanError::InvalidResource(
                "invalid owned NV12 row extents",
            ));
        }
        let height = u32::try_from(source.bytes[0].len() / width as usize)
            .map_err(|_| VulkanError::InvalidResource("NV12 height overflows"))?;
        if source.bytes[1].len() as u64
            != u64::from(source.widths[1]) * 2 * u64::from(height.div_ceil(2))
        {
            return Err(VulkanError::InvalidResource(
                "invalid owned NV12 chroma extent",
            ));
        }
        Ok([width, height])
    }
    pub(crate) fn new(cache: &mut VulkanResidency, extent: [u32; 2]) -> Result<Self, VulkanError> {
        let y = cache
            .allocate_host_image(extent, vk::Format::R8_UNORM)?
            .view_2d(vk::Format::R8_UNORM)?;
        let uv = cache
            .allocate_host_image(extent.map(|v| v.div_ceil(2)), vk::Format::R8G8_UNORM)?
            .view_2d(vk::Format::R8G8_UNORM)?;
        Ok(Self {
            extent,
            uploaded_extent: [0, 0],
            views: [y, uv],
            initialized: false,
        })
    }
    pub(crate) fn upload(
        &mut self,
        cache: &mut VulkanResidency,
        source: &Nv12Planes,
    ) -> Result<(), VulkanError> {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        cache.device.end_rendering();
        cache.device.begin_recording()?;
        for (index, view) in self.views.iter().enumerate() {
            cache.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(if self.initialized {
                        vk::PipelineStageFlags2::FRAGMENT_SHADER
                    } else {
                        vk::PipelineStageFlags2::NONE
                    })
                    .src_access_mask(if self.initialized {
                        vk::AccessFlags2::SHADER_SAMPLED_READ
                    } else {
                        vk::AccessFlags2::NONE
                    })
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(if self.initialized {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(view.image.handle)
                    .subresource_range(range),
            );
            cache.barriers.emit(&mut cache.device);
            let width = source.widths[index];
            let pitch = u64::from(width) * (index as u64 + 1);
            let bytes = &source.bytes[index];
            let slice = cache.device.submission.reserve_upload_aligned(
                &mut cache.device.dispatch,
                bytes,
                4,
            )?;
            let buffer = cache.device.submission.staging_buffer(&slice).handle;
            let work = cache.device.work_ticket()?;
            cache.device.retire_image(Arc::clone(&view.image), work)?;
            let region = vk::BufferImageCopy::default()
                .buffer_offset(slice.offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height: (bytes.len() as u64 / pitch) as u32,
                    depth: 1,
                });
            unsafe {
                cache.device.dispatch.cmd_copy_buffer_to_image(
                    buffer,
                    view.image.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
            cache.statistics.cpu_uploaded_bytes += bytes.len() as u64;
            cache.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(view.image.handle)
                    .subresource_range(range),
            );
        }
        cache.barriers.emit(&mut cache.device);
        self.uploaded_extent = [
            source.widths[0],
            (source.bytes[0].len() / source.widths[0] as usize) as u32,
        ];
        self.initialized = true;
        Ok(())
    }
}
pub(crate) struct PreparedNv12 {
    draw: PreparedDraw,
    parameters: [u8; 80],
}
/// Request ahead of consumption, exactly like a guest graphics pipeline. The
/// owner polls the returned publication; no synchronous compilation in recording.
pub(crate) fn nv12_pipeline(
    compiler: &PipelineCompiler,
    format: vk::Format,
) -> Result<Arc<NativePipeline>, CompileError> {
    if !matches!(
        format,
        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
    ) {
        return Err(CompileError::Failed(
            "NV12 output requires RGBA8/BGRA8 UNORM".into(),
        ));
    }
    static SHADER: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    let shader = SHADER.get_or_init(|| {
        Arc::new(ShaderArtifact::new(Box::new([
            stage(
                ShaderStage::Vertex,
                include_bytes!("../../shaders/spv/solid.vert.spv"),
                Box::new([]),
                0,
            ),
            stage(
                ShaderStage::Fragment,
                include_bytes!("../../shaders/spv/nv12.frag.spv"),
                (0..2)
                    .map(|binding| spirv::Binding {
                        resource: ResourceId::new(binding),
                        binding,
                        kind: spirv::DescriptorKind::SampledImage,
                        count: 1,
                        sampler: None,
                        readable: true,
                        writable: false,
                    })
                    .collect(),
                80,
            ),
        ])))
    });
    compiler.request(PipelineRecipe::Graphics {
        shader: Arc::clone(shader),
        state: GraphicsState::color(format),
    })
}
impl VulkanResidency {
    pub(crate) fn prepare_nv12(
        &mut self,
        pipeline: Arc<NativePipeline>,
        descriptors: &mut DescriptorCache,
        sources: &Nv12Inputs,
        destination: ImageId,
        plane: usize,
        transform: Nv12ColorTransform,
    ) -> Result<PreparedNv12, VulkanError> {
        let t = transform;
        if t.shift > 15
            || t.clamp[0] < 0
            || t.clamp[1] > 1023
            || t.clamp[0] > t.clamp[1]
            || t.coefficients
                .iter()
                .flatten()
                .any(|c| !(-524288..=524287).contains(c))
        {
            return Err(VulkanError::InvalidResource(
                "NV12 fixed-point coefficients, shift or clamp exceed the supported integer contract",
            ));
        }
        let extent = self
            .image(destination)?
            .planes
            .get(plane)
            .ok_or(VulkanError::InvalidResource(
                "NV12 destination plane is absent",
            ))?
            .extent;
        if t.source_origin[0]
            .checked_add(extent.width)
            .is_none_or(|v| v > sources.uploaded_extent[0])
            || t.source_origin[1]
                .checked_add(extent.height)
                .is_none_or(|v| v > sources.uploaded_extent[1])
        {
            return Err(VulkanError::InvalidResource(
                "NV12 source crop exceeds decoded frame",
            ));
        }
        let format = self.image(destination)?.description.format;
        if !matches!(
            format,
            vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
        ) {
            return Err(VulkanError::InvalidResource(
                "NV12 destination requires RGBA8/BGRA8 UNORM",
            ));
        }
        let record = descriptors.get(
            &pipeline.layout,
            &sources
                .views
                .each_ref()
                .map(|view| DescriptorResource::Image {
                    view: Arc::clone(view),
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }),
        )?;
        let graphics = self.prepare_host_graphics(
            pipeline,
            record,
            GraphicsParameters::new(
                GraphicsState::color(format),
                vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            ),
        )?;
        let targets = self.prepare_targets(RenderingTargets {
            area: transfers::full_rect(self.image(destination)?.planes[plane]),
            colors: Box::new([Some(RenderAttachment::color(
                destination,
                plane,
                AttachmentLoad::Discard,
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })?;
        let draw = self.prepare_draw(graphics, targets, Box::new([]), None)?;
        let words: Vec<_> = t
            .coefficients
            .into_iter()
            .flatten()
            .map(|v| v as u32)
            .chain([
                t.source_origin[0],
                t.source_origin[1],
                t.shift,
                u32::from(t.enabled),
            ])
            .chain([t.clamp[0] as u32, t.clamp[1] as u32, u32::from(t.alpha), 0])
            .collect();
        let parameters = std::array::from_fn(|i| words[i / 4].to_le_bytes()[i % 4]);
        Ok(PreparedNv12 { draw, parameters })
    }
    pub(crate) fn convert_nv12(
        &mut self,
        prepared: &PreparedNv12,
    ) -> Result<WorkTicket, VulkanError> {
        self.draw(
            &prepared.draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &prepared.parameters,
        )
    }
}
