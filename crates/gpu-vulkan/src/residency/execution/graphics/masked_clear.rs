//! Stencil replacement through the stencil write mask preserves untouched bits.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/renderer_vulkan/blit_image.cpp
use super::*;
use crate::cache::{BlendState, PipelineCompiler, PipelineRecipe, StencilState};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    colors: [vk::Format; 8],
    color_count: usize,
    depth: vk::Format,
    stencil: vk::Format,
    samples: vk::SampleCountFlags,
    clear_depth: bool,
}

#[derive(Default)]
pub(crate) struct MaskedClearPipelines(HashMap<Key, HostGraphics>);

impl Residency<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn clear_masked_stencil(
        &mut self,
        cache: &mut MaskedClearPipelines,
        compiler: &PipelineCompiler,
        targets: &Arc<PreparedTargets>,
        rect: vk::ClearRect,
        mask: u32,
        value: vk::ClearDepthStencilValue,
        clear_depth: bool,
    ) -> Result<WorkTicket, VulkanError> {
        if !clear::contains_rect(targets.area, rect.rect)
            || rect.layer_count != 1
            || rect.base_array_layer >= targets.framebuffer.layers
            || targets.stencil.is_none()
            || (clear_depth && (!value.depth.is_finite() || !(0. ..=1.).contains(&value.depth)))
        {
            return Err(VulkanError::InvalidResource("invalid masked stencil clear"));
        }
        if rect.rect.extent.width == 0
            || rect.rect.extent.height == 0
            || (mask & 255 == 0 && !clear_depth)
        {
            return Ok(self.runtime.device.current_work());
        }
        // An ordinary draw addresses the first framebuffer layer. Other layers
        // use a cached single-layer view of the same allocation, without copies.
        let layer_targets;
        let targets = if rect.base_array_layer == 0 {
            targets
        } else {
            let source = targets.stencil.as_ref().unwrap().description.source;
            let AttachmentSource::Canonical { view, .. } = source else {
                return Err(VulkanError::InvalidResource(
                    "layered masked clear requires a canonical image view",
                ));
            };
            let view = &self.texture_cache.views[view];
            let image = view.image;
            let mut description = view.description;
            description.base_layer += rect.base_array_layer;
            description.layer_count = 1;
            description.view_type = vk::ImageViewType::TYPE_2D;
            let view = self.image_view(image, description)?;
            layer_targets = self.framebuffer_for_view(view)?;
            &layer_targets
        };
        let key = Key {
            colors: targets.pass.colors,
            color_count: targets.colors.len(),
            depth: targets
                .depth
                .as_ref()
                .map_or(vk::Format::UNDEFINED, |a| a.format),
            stencil: targets
                .stencil
                .as_ref()
                .map_or(vk::Format::UNDEFINED, |a| a.format),
            samples: targets.pass.samples,
            clear_depth,
        };
        let graphics = match cache.0.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let mut state = GraphicsState::color(vk::Format::UNDEFINED);
                state.colors = key.colors[..key.color_count]
                    .iter()
                    .map(|&format| {
                        (
                            format,
                            BlendState {
                                write_mask: vk::ColorComponentFlags::empty(),
                                ..BlendState::opaque()
                            },
                        )
                    })
                    .collect();
                state.depth_format = key.depth;
                state.stencil_format = key.stencil;
                state.samples = key.samples;
                state.depth_test = clear_depth;
                state.depth_write = clear_depth;
                state.stencil_test = true;
                state.front = StencilState {
                    pass: vk::StencilOp::REPLACE,
                    ..Default::default()
                };
                state.back = state.front;
                let shader = Arc::new(ShaderArtifact::new(Box::new([
                    super::super::super::helper_stage(
                        nixe_gpu::ShaderStage::Vertex,
                        include_bytes!("../../../../shaders/spv/clear-stencil.vert.spv"),
                        Box::new([]),
                        0,
                    ),
                    super::super::super::helper_stage(
                        nixe_gpu::ShaderStage::Fragment,
                        include_bytes!("../../../../shaders/spv/clear-stencil.frag.spv"),
                        Box::new([]),
                        4,
                    ),
                ])));
                let pipeline = compiler
                    .request(PipelineRecipe::Graphics {
                        shader,
                        state: state.clone(),
                    })
                    .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
                let program = self.prepare_host_program(pipeline, &[])?;
                entry.insert(self.prepare_host_pipeline(
                    program,
                    GraphicsParameters::new(state, targets.area.extent),
                )?)
            }
        };
        let parameters = &mut graphics.parameters;
        parameters.viewports[0].x = targets.area.offset.x as f32;
        parameters.viewports[0].y = targets.area.offset.y as f32;
        parameters.viewports[0].width = targets.area.extent.width as f32;
        parameters.viewports[0].height = targets.area.extent.height as f32;
        parameters.scissors[0] = rect.rect;
        parameters.stencil_write_mask = [mask & 255; 2];
        parameters.stencil_reference = [value.stencil & 255; 2];
        self.validate_dynamic_parameters(parameters)?;
        self.draw(
            graphics,
            targets,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &value.depth.to_ne_bytes(),
        )
    }
}
