//! Spatial antialiasing before crop, transform and presentation scaling.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/present/smaa.cpp
use super::{
    ScreenRenderer,
    filters::{FilterTarget, Program},
    render::{self, LayerImage},
};
use crate::{GraphicsParameters, VulkanError, cache::*, memory::AllocatedImageView};
use ash::vk;
use nixe_video::{AntiAliasing, ScalingFilter};
use std::sync::Arc;

const COLOR: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Intermediates belong to the composition slot that retires their GPU uses.
/// Only the selected algorithm has resources; pipelines remain in the native cache.
pub(super) struct Targets {
    mode: AntiAliasing,
    output: FilterTarget,
    smaa: Option<[FilterTarget; 2]>,
    decoded: Option<FilterTarget>,
}
impl Targets {
    pub fn new(
        cache: &mut crate::residency::Residency<'_>,
        extent: [u32; 2],
        mode: AntiAliasing,
        packed: bool,
    ) -> Result<Self, VulkanError> {
        Ok(Self {
            mode,
            output: FilterTarget::new(cache, extent, COLOR)?,
            smaa: if mode == AntiAliasing::Smaa {
                Some([
                    FilterTarget::new(cache, extent, vk::Format::R16G16_SFLOAT)?,
                    FilterTarget::new(cache, extent, COLOR)?,
                ])
            } else {
                None
            },
            decoded: if packed {
                Some(FilterTarget::new(cache, extent, COLOR)?)
            } else {
                None
            },
        })
    }
    pub fn matches(&self, source: &LayerImage, mode: AntiAliasing) -> bool {
        self.mode == mode
            && self.output.extent == source.extent
            && self.decoded.is_some() == (source.format == vk::Format::R16_UINT)
    }
    pub fn draw(
        &self,
        renderer: &mut ScreenRenderer,
        cache: &mut crate::residency::Residency<'_>,
        compiler: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        source: &LayerImage,
    ) -> Result<LayerImage, VulkanError> {
        let decoded;
        let input = if let Some(target) = &self.decoded {
            // Integer framebuffers must be decoded before filtering color samples.
            // Keep the full source coordinates; crop and orientation are applied later.
            let mut raw = source.clone();
            raw.crop = nixe_gpu::FrameCrop {
                left: 0,
                top: 0,
                width: source.extent[0],
                height: source.extent[1],
            };
            raw.transform = Default::default();
            let pipeline =
                renderer.screen_pipeline(compiler, COLOR, ScalingFilter::Nearest, &raw)?;
            cache.device_mut().end_rendering()?;
            prepare_target(cache, target);
            renderer.screen(
                cache,
                descriptors,
                pipeline,
                &raw,
                &target.framebuffer,
                extent(target.extent),
                COLOR,
                false,
                ScalingFilter::Nearest,
            )?;
            finish_target(cache, target)?;
            decoded = target.layer(source.encoding, Default::default());
            &decoded
        } else {
            source
        };
        if let Some([edges, weights]) = &self.smaa {
            if renderer.aa_lookup.is_none() {
                renderer.aa_lookup = Some(LookupTables::new(cache)?);
            }
            draw(
                renderer,
                cache,
                compiler,
                descriptors,
                Program::SmaaEdges,
                &[&input.view],
                edges,
            )?;
            let lookups = renderer.aa_lookup.as_ref().unwrap();
            let area = Arc::clone(&lookups.area);
            let search = Arc::clone(&lookups.search);
            draw(
                renderer,
                cache,
                compiler,
                descriptors,
                Program::SmaaWeights,
                &[&edges.view, &area, &search],
                weights,
            )?;
            draw(
                renderer,
                cache,
                compiler,
                descriptors,
                Program::SmaaBlend,
                &[&input.view, &weights.view],
                &self.output,
            )?;
        } else {
            draw(
                renderer,
                cache,
                compiler,
                descriptors,
                Program::Fxaa,
                &[&input.view],
                &self.output,
            )?;
        }
        let mut output = self.output.layer(source.encoding, source.transform);
        output.crop = source.crop;
        Ok(output)
    }
}

pub(super) struct LookupTables {
    area: Arc<AllocatedImageView>,
    search: Arc<AllocatedImageView>,
}
impl LookupTables {
    fn new(cache: &mut crate::residency::Residency<'_>) -> Result<Self, VulkanError> {
        Ok(Self {
            area: upload(
                cache,
                [160, 560],
                vk::Format::R8G8_UNORM,
                include_bytes!("../../shaders/data/smaa-area.bin"),
            )?,
            search: upload(
                cache,
                [64, 16],
                vk::Format::R8_UNORM,
                include_bytes!("../../shaders/data/smaa-search.bin"),
            )?,
        })
    }
}
fn extent(size: [u32; 2]) -> vk::Extent2D {
    vk::Extent2D {
        width: size[0],
        height: size[1],
    }
}
fn upload(
    cache: &mut crate::residency::Residency<'_>,
    size: [u32; 2],
    format: vk::Format,
    bytes: &[u8],
) -> Result<Arc<AllocatedImageView>, VulkanError> {
    let image = cache.allocate_host_image(size, format)?;
    let view = Arc::clone(&image).view_2d(format)?;
    cache.device_mut().end_rendering()?;
    let device = cache.device_mut();
    let slice = device.submission.reserve_upload(bytes)?;
    render::transition(
        device,
        image.handle,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags2::NONE,
        vk::AccessFlags2::NONE,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
    );
    let copy = vk::BufferImageCopy::default()
        .buffer_offset(slice.offset)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: size[0],
            height: size[1],
            depth: 1,
        });
    unsafe {
        device.dispatch.cmd_copy_buffer_to_image(
            device.submission.staging_buffer(&slice).handle,
            image.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
    }
    render::transition(
        device,
        image.handle,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::GENERAL,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    let work = device.current_work();
    device.retire_image_views(vec![Arc::clone(&view)], work)?;
    Ok(view)
}
fn prepare_target(cache: &mut crate::residency::Residency<'_>, target: &FilterTarget) {
    // The slot's earlier uses have completed. Edge detection can discard pixels,
    // so every pass also clears its attachment to zero before drawing.
    render::transition(
        cache.device_mut(),
        target.view.image.handle,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::PipelineStageFlags2::NONE,
        vk::AccessFlags2::NONE,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    );
}
fn finish_target(
    cache: &mut crate::residency::Residency<'_>,
    target: &FilterTarget,
) -> Result<(), VulkanError> {
    render::transition(
        cache.device_mut(),
        target.view.image.handle,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::GENERAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    let work = cache.device_mut().current_work();
    cache
        .device_mut()
        .retire_image_views(vec![Arc::clone(&target.view)], work)
}
#[allow(clippy::too_many_arguments)]
fn draw(
    renderer: &mut ScreenRenderer,
    cache: &mut crate::residency::Residency<'_>,
    compiler: &PipelineCompiler,
    descriptors: &mut DescriptorCache,
    program: Program,
    images: &[&Arc<AllocatedImageView>],
    target: &FilterTarget,
) -> Result<(), VulkanError> {
    let pipeline = renderer.pipeline(compiler, target.format, program)?;
    let resources: Vec<_> = images
        .iter()
        .map(|view| DescriptorResource::CombinedImageSampler {
            view: Arc::clone(view),
            layout: vk::ImageLayout::GENERAL,
            sampler: renderer.linear.clone(),
        })
        .collect();
    let record = descriptors.get(&pipeline.layout, &resources)?;
    let size = extent(target.extent);
    let parameters = GraphicsParameters::new(GraphicsState::color(target.format), size);
    let graphics = cache.prepare_host_graphics(pipeline, record, parameters)?;
    cache.bind_host_graphics(&graphics, &metrics(target.extent))?;
    cache.device_mut().end_rendering()?;
    prepare_target(cache, target);
    let device = cache.device_mut();
    let mut pass = target.framebuffer.key;
    pass.color_clear = 1;
    device.begin_render_pass(
        &target.framebuffer,
        pass,
        vk::Rect2D {
            offset: Default::default(),
            extent: size,
        },
        &[vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.; 4] },
        }],
    )?;
    unsafe {
        device.dispatch.cmd_draw(3, 1, 0, 0);
        device.dispatch.end_rendering();
    }
    finish_target(cache, target)
}
fn metrics(size: [u32; 2]) -> [u8; 16] {
    let [width, height] = size.map(|v| v as f32);
    let mut bytes = [0; 16];
    for (i, value) in [1. / width, 1. / height, width, height]
        .into_iter()
        .enumerate()
    {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}
