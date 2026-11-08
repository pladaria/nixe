//! Eden BlitColor: fullscreen triangle, unnormalized source coordinates and
//! native sampler filtering. Selected image views supply format/subresources.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/blit_image.cpp#L661-L744
use super::*;
use crate::{Draw, GraphicsParameters, HostBinding, cache::*};
use nixe_gpu::{
    ShaderStage,
    shader_module::{self as spirv, ResourceId},
};
use std::sync::OnceLock;
pub fn color_blit_pipeline(
    compiler: &PipelineCompiler,
    format: vk::Format,
) -> Result<NativePipeline, CompileError> {
    static SHADER: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    let shader = SHADER.get_or_init(|| {
        Arc::new(ShaderArtifact::new(Box::new([
            conversion::stage(
                ShaderStage::Vertex,
                include_bytes!("../../shaders/spv/blit.vert.spv"),
                Box::new([]),
                16,
            ),
            conversion::stage(
                ShaderStage::Fragment,
                include_bytes!("../../shaders/spv/blit.frag.spv"),
                Box::new([spirv::Binding {
                    resource: ResourceId::new(0),
                    binding: 0,
                    kind: spirv::DescriptorKind::CombinedImageSampler,
                    count: 1,
                    sampler: Some(ResourceId::new(0)),
                    readable: true,
                    writable: false,
                }]),
                0,
            ),
        ])))
    });
    compiler.request(PipelineRecipe::Graphics {
        shader: Arc::clone(shader),
        state: GraphicsState::color(format),
    })
}
impl Residency<'_> {
    #[allow(clippy::too_many_arguments)]
    pub fn blit_color(
        &mut self,
        pipeline: NativePipeline,
        source: ImageViewId,
        destination_view: ImageViewId,
        plane: usize,
        source_rect: vk::Rect2D,
        destination_rect: vk::Rect2D,
        first: [i64; 2],
        step: [i64; 2],
        sampler: crate::cache::Sampler,
    ) -> Result<WorkTicket, VulkanError> {
        let destination = self.texture_cache.views[destination_view].image;
        let source_image = self
            .texture_cache
            .get(self.texture_cache.views[source].image);
        let view = &self.texture_cache.views[source].description;
        if view.view_type != vk::ImageViewType::TYPE_2D
            || view.aspects != vk::ImageAspectFlags::COLOR
            || view.mip_count != 1
            || view.layer_count != 1
            || source_image.description.samples != vk::SampleCountFlags::TYPE_1
        {
            return Err(VulkanError::InvalidResource(
                "color blit requires a single-sample 2D color view",
            ));
        }
        let src_plane = source_image
            .planes
            .iter()
            .position(|p| {
                p.mip_level == view.base_mip
                    && p.array_layer + p.depth_slice == view.base_layer
                    && p.aspect == view.aspects
            })
            .ok_or(VulkanError::InvalidResource(
                "color blit source plane is absent",
            ))?;
        self.checked_image_rect(
            self.texture_cache.views[source].image,
            src_plane,
            source_rect,
        )?;
        if source_rect.extent.width == 0 || source_rect.extent.height == 0 {
            return Err(VulkanError::InvalidResource(
                "color blit source rectangle is empty",
            ));
        }
        let p = self.checked_image_rect(destination, plane, destination_rect)?;
        let format = self.image_view_description(destination_view)?.format;
        let program = self.prepare_host_program(
            pipeline,
            &[HostBinding::CombinedImageSampler {
                image: self.host_image(source)?,
                sampler,
            }],
        )?;
        let mut parameters = GraphicsParameters::new(
            GraphicsState::color(format),
            vk::Extent2D {
                width: p.extent.width,
                height: p.extent.height,
            },
        );
        parameters.scissors[0] = destination_rect;
        parameters.viewports[0] = vk::Viewport {
            x: destination_rect.offset.x as f32,
            y: destination_rect.offset.y as f32,
            width: destination_rect.extent.width as f32,
            height: destination_rect.extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        };
        let graphics = self.prepare_host_pipeline(program, parameters)?;
        let targets = self.framebuffer_for_view(destination_view)?;
        let unit = 4294967296.0_f64;
        let scale = [step[0] as f64 / unit, step[1] as f64 / unit];
        let push = [
            (scale[0] * f64::from(destination_rect.extent.width)) as f32,
            (scale[1] * f64::from(destination_rect.extent.height)) as f32,
            (first[0] as f64 / unit - scale[0] * 0.5) as f32,
            (first[1] as f64 / unit - scale[1] * 0.5) as f32,
        ]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
        self.draw(
            &graphics,
            &targets,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &push,
        )
    }
}
