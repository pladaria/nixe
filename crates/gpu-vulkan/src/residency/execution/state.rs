use super::*;
pub struct GraphicsParameters {
    pub state: GraphicsState,
    pub(crate) vertex_input: crate::native::VertexInput,
    pub(crate) blending: crate::native::Blending,
    pub viewports: Box<[vk::Viewport]>,
    pub scissors: Box<[vk::Rect2D]>,
    pub line_width: f32,
    pub depth_bias: [f32; 3], // constant, clamp, slope
    pub blend_constants: [f32; 4],
    pub depth_bounds: [f32; 2],
    pub stencil_compare_mask: [u32; 2],
    pub stencil_write_mask: [u32; 2],
    pub stencil_reference: [u32; 2],
}
impl GraphicsParameters {
    pub fn new(state: GraphicsState, extent: vk::Extent2D) -> Self {
        Self {
            vertex_input: crate::native::VertexInput::host(&state),
            blending: crate::native::Blending::host(&state),
            state,
            viewports: Box::new([vk::Viewport {
                x: 0.,
                y: 0.,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.,
                max_depth: 1.,
            }]),
            scissors: Box::new([vk::Rect2D {
                offset: Default::default(),
                extent,
            }]),
            line_width: 1.,
            depth_bias: [0.; 3],
            blend_constants: [0.; 4],
            depth_bounds: [0., 1.],
            stencil_compare_mask: [u32::MAX; 2],
            stencil_write_mask: [u32::MAX; 2],
            stencil_reference: [0; 2],
        }
    }
}
pub struct HostGraphics {
    pub(crate) program: HostProgram,
    pub(crate) parameters: GraphicsParameters,
}
impl Residency<'_> {
    pub fn prepare_host_pipeline(
        &self,
        program: HostProgram,
        parameters: GraphicsParameters,
    ) -> Result<HostGraphics, VulkanError> {
        let pipeline = &program.pipeline;
        if pipeline.bind_point != vk::PipelineBindPoint::GRAPHICS
            || pipeline.graphics.as_ref()
                != Some(
                    &parameters
                        .state
                        .clone()
                        .normalized(&pipeline.dynamic_states),
                )
        {
            return Err(VulkanError::InvalidResource(
                "graphics parameters differ from the compiled static pipeline state",
            ));
        }
        parameters
            .vertex_input
            .validate(self.runtime.device.profile())?;
        parameters
            .blending
            .validate(self.runtime.device.profile(), &parameters.state)?;
        self.validate_dynamic_parameters(&parameters)?;
        Ok(HostGraphics {
            program,
            parameters,
        })
    }
    pub(crate) fn validate_dynamic_parameters(
        &self,
        parameters: &GraphicsParameters,
    ) -> Result<(), VulkanError> {
        let profile = self.runtime.device.profile();
        let limits = &profile.limits;
        if parameters.viewports.len() != parameters.state.viewport_count as usize
            || parameters.scissors.len() != parameters.viewports.len()
        {
            return Err(VulkanError::InvalidResource(
                "viewport/scissor counts differ from the pipeline",
            ));
        }
        for v in &parameters.viewports {
            if [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]
                .iter()
                .any(|f| !f.is_finite())
                || v.width <= 0.
                || v.height == 0.
                || v.width > limits.max_viewport_dimensions[0] as f32
                || v.height.abs() > limits.max_viewport_dimensions[1] as f32
                || v.x < limits.viewport_bounds_range[0]
                || v.x + v.width > limits.viewport_bounds_range[1]
                || v.y.min(v.y + v.height) < limits.viewport_bounds_range[0]
                || v.y.max(v.y + v.height) > limits.viewport_bounds_range[1]
                || !(0. ..=1.).contains(&v.min_depth)
                || !(0. ..=1.).contains(&v.max_depth)
            {
                return Err(VulkanError::InvalidResource(
                    "viewport is non-finite or exceeds enabled host bounds",
                ));
            }
        }
        if parameters.scissors.iter().any(|r| {
            r.offset.x < 0
                || r.offset.y < 0
                || (r.offset.x as u32)
                    .checked_add(r.extent.width)
                    .is_none_or(|n| n > i32::MAX as u32)
                || (r.offset.y as u32)
                    .checked_add(r.extent.height)
                    .is_none_or(|n| n > i32::MAX as u32)
        }) {
            return Err(VulkanError::InvalidResource(
                "scissor exceeds native signed coordinate bounds",
            ));
        }
        let width = parameters.line_width;
        if !width.is_finite()
            || width <= 0.
            || width < limits.line_width_range[0]
            || width > limits.line_width_range[1]
            || (width != 1. && profile.core.wide_lines == 0)
        {
            return Err(VulkanError::InvalidResource(
                "line width exceeds enabled native wideLines/range",
            ));
        }
        if parameters.depth_bias.iter().any(|v| !v.is_finite())
            || (parameters.depth_bias[1] != 0. && profile.core.depth_bias_clamp == 0)
        {
            return Err(VulkanError::InvalidResource(
                "depth bias requires finite values and enabled depthBiasClamp",
            ));
        }
        if parameters.state.depth_bounds
            && (profile.core.depth_bounds == 0
                || parameters
                    .depth_bounds
                    .iter()
                    .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
                || parameters.depth_bounds[0] > parameters.depth_bounds[1])
        {
            return Err(VulkanError::InvalidResource(
                "depth bounds exceed enabled normalized depthBounds contract",
            ));
        }
        Ok(())
    }
    pub(crate) fn graphics_state_cached(
        &mut self,
        pipeline: &NativePipeline,
        parameters: &GraphicsParameters,
        identity: u64,
    ) -> Result<(), VulkanError> {
        let index = self.runtime.device.recording_slot()?;
        if self.runtime.device.submission.slots[index]
            .graphics_state
            .parameters_id
            == Some(identity)
        {
            return Ok(());
        }
        self.graphics_state(pipeline, parameters)?;
        self.runtime.device.submission.slots[index]
            .graphics_state
            .parameters_id = Some(identity);
        Ok(())
    }
    pub(crate) fn graphics_state(
        &mut self,
        pipeline: &NativePipeline,
        p: &GraphicsParameters,
    ) -> Result<(), VulkanError> {
        let index = self.runtime.device.recording_slot()?;
        // Explicit state emission replaces the previous native context. The
        // next guest draw consumes its complete invalidation set.
        self.runtime.device.submission.slots[index].graphics_state = Default::default();
        let raw = &mut self.runtime.device.dispatch;
        unsafe {
            if pipeline
                .dynamic_states
                .contains(&vk::DynamicState::COLOR_BLEND_ENABLE_EXT)
                && p.blending.count != 0
            {
                let count = p.blending.count;
                raw.cmd_set_color_blend_enable(
                    &p.blending.attachments.map(|b| u32::from(b.enabled))[..count],
                );
                raw.cmd_set_color_write_mask(
                    &p.blending.attachments.map(|b| b.write_mask)[..count],
                );
                raw.cmd_set_color_blend_equation(
                    &p.blending.attachments.map(|b| vk::ColorBlendEquationEXT {
                        src_color_blend_factor: b.src_color,
                        dst_color_blend_factor: b.dst_color,
                        color_blend_op: b.color_op,
                        src_alpha_blend_factor: b.src_alpha,
                        dst_alpha_blend_factor: b.dst_alpha,
                        alpha_blend_op: b.alpha_op,
                    })[..count],
                );
            }
            if pipeline
                .dynamic_states
                .contains(&vk::DynamicState::VERTEX_INPUT_EXT)
            {
                raw.cmd_set_vertex_input(&p.vertex_input.bindings, &p.vertex_input.attributes);
            }
            raw.cmd_set_viewport(0, &p.viewports);
            raw.cmd_set_scissor(0, &p.scissors);
            raw.cmd_set_line_width(p.line_width);
            raw.cmd_set_depth_bias(p.depth_bias[0], p.depth_bias[1], p.depth_bias[2]);
            raw.cmd_set_blend_constants(&p.blend_constants);
            for (index, face) in [vk::StencilFaceFlags::FRONT, vk::StencilFaceFlags::BACK]
                .into_iter()
                .enumerate()
            {
                raw.cmd_set_stencil_compare_mask(face, p.stencil_compare_mask[index]);
                raw.cmd_set_stencil_write_mask(face, p.stencil_write_mask[index]);
                raw.cmd_set_stencil_reference(face, p.stencil_reference[index]);
            }
            if pipeline
                .dynamic_states
                .contains(&vk::DynamicState::DEPTH_BOUNDS)
            {
                raw.cmd_set_depth_bounds(p.depth_bounds[0], p.depth_bounds[1]);
            }
            raw.dynamic_state(&p.state);
        }
        self.runtime.device.submission.statistics.graphics_updates += 1;
        Ok(())
    }
}
