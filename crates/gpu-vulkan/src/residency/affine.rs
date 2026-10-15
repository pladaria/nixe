//! Maxwell 2D sampling retains its signed 32.32 accumulator in the host shader.
use super::*;
use crate::{Binding, Draw, GraphicsParameters, RenderAttachment, RenderingTargets, cache::*};
use nixe_gpu::{
    ShaderResourceKind, ShaderStage,
    ssa::{ResourceId, spirv},
};
use std::sync::OnceLock;
pub fn affine_pipeline(
    compiler: &PipelineCompiler,
    format: vk::Format,
) -> Result<Pending<NativePipeline>, CompileError> {
    static SHADER: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    let shader = SHADER.get_or_init(|| {
        Arc::new(ShaderArtifact::new(Box::new([
            conversion::stage(
                ShaderStage::Vertex,
                include_bytes!("../../shaders/spv/solid.vert.spv"),
                Box::new([]),
                0,
            ),
            conversion::stage(
                ShaderStage::Fragment,
                include_bytes!("../../shaders/spv/affine.frag.spv"),
                Box::new([spirv::Binding {
                    resource: ResourceId::new(0),
                    binding: 0,
                    kind: ShaderResourceKind::SampledImage,
                    readable: true,
                    writable: false,
                }]),
                64,
            ),
        ])))
    });
    compiler.request(PipelineRecipe::Graphics {
        shader: Arc::clone(shader),
        state: GraphicsState::color(format),
    })
}
impl VulkanResidency {
    #[allow(clippy::too_many_arguments)]
    pub fn affine_blit(
        &mut self,
        pipeline: Arc<NativePipeline>,
        source: ImageViewId,
        destination: ImageId,
        plane: usize,
        source_rect: vk::Rect2D,
        destination_rect: vk::Rect2D,
        first: [i64; 2],
        step: [i64; 2],
        linear: bool,
        descriptors: &mut DescriptorCache,
    ) -> Result<WorkTicket, VulkanError> {
        let source_image = self.image(source.image)?;
        let view = source_image
            .views
            .get(source.view)
            .ok_or(VulkanError::InvalidResource("affine source view is absent"))?;
        if view.view_type != vk::ImageViewType::TYPE_2D
            || view.aspects != vk::ImageAspectFlags::COLOR
            || view.mip_count != 1
            || view.layer_count != 1
            || source_image.description.samples != vk::SampleCountFlags::TYPE_1
        {
            return Err(VulkanError::InvalidResource(
                "affine sampling requires a single-sample 2D color view",
            ));
        }
        let src_plane = source_image
            .planes
            .iter()
            .position(|p| {
                p.mip_level == view.base_mip
                    && p.array_layer == view.base_layer
                    && p.aspect == view.aspects
            })
            .ok_or(VulkanError::InvalidResource(
                "affine source plane is absent",
            ))?;
        self.checked_image_rect(source.image, src_plane, source_rect)?;
        if source_rect.extent.width == 0 || source_rect.extent.height == 0 {
            return Err(VulkanError::InvalidResource(
                "affine source rectangle is empty",
            ));
        }
        if self.device.profile().core.shader_int64 == 0 {
            return Err(VulkanError::InvalidResource(
                "exact Maxwell 32.32 affine sampling requires enabled shaderInt64",
            ));
        }
        let p = self.checked_image_rect(destination, plane, destination_rect)?;
        let format = self.image(destination)?.description.format;
        let program = self.prepare_program(pipeline, &[Binding::Image(source)], descriptors)?;
        let mut parameters = GraphicsParameters::new(
            GraphicsState::color(format),
            vk::Extent2D {
                width: p.extent.width,
                height: p.extent.height,
            },
        );
        parameters.scissors[0] = destination_rect;
        let graphics = self.prepare_graphics(program, parameters)?;
        let targets = self.prepare_targets(RenderingTargets {
            area: destination_rect,
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
        let mut push = [0_u8; 64];
        for (i, v) in [
            source_rect.offset.x,
            source_rect.offset.y,
            source_rect.offset.x + source_rect.extent.width as i32,
            source_rect.offset.y + source_rect.extent.height as i32,
            destination_rect.offset.x,
            destination_rect.offset.y,
            i32::from(linear),
            0,
        ]
        .into_iter()
        .enumerate()
        {
            push[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in first.into_iter().chain(step).enumerate() {
            push[32 + i * 8..40 + i * 8].copy_from_slice(&v.to_le_bytes());
        }
        self.draw(
            &draw,
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
