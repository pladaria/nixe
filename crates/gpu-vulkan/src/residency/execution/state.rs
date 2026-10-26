use super::*;
/// Immutable changed-state preparation. Unchanged draws compare one Arc identity;
/// values are not hashed and setters are not repeated on the recording path.
pub struct GraphicsParameters {
    pub state: GraphicsState,
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
    fn same_state(&self, other: &Self) -> bool {
        self.state == other.state
            && self.viewports.len() == other.viewports.len()
            && self.viewports.iter().zip(&other.viewports).all(|(a, b)| {
                [a.x, a.y, a.width, a.height, a.min_depth, a.max_depth].map(f32::to_bits)
                    == [b.x, b.y, b.width, b.height, b.min_depth, b.max_depth].map(f32::to_bits)
            })
            && self.scissors == other.scissors
            && self.line_width.to_bits() == other.line_width.to_bits()
            && self.depth_bias.map(f32::to_bits) == other.depth_bias.map(f32::to_bits)
            && self.blend_constants.map(f32::to_bits) == other.blend_constants.map(f32::to_bits)
            && self.depth_bounds.map(f32::to_bits) == other.depth_bounds.map(f32::to_bits)
            && self.stencil_compare_mask == other.stencil_compare_mask
            && self.stencil_write_mask == other.stencil_write_mask
            && self.stencil_reference == other.stencil_reference
    }

    pub fn new(state: GraphicsState, extent: vk::Extent2D) -> Self {
        Self {
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
pub struct PreparedGraphics {
    pub(super) program: Arc<PreparedProgram>,
    pub(super) parameters: Arc<GraphicsParameters>,
}
pub(crate) struct PreparedGraphicsState {
    pipeline: Arc<NativePipeline>,
    parameters: Arc<GraphicsParameters>,
}
impl PreparedGraphicsState {
    pub(crate) fn bind(
        &self,
        program: Arc<PreparedProgram>,
    ) -> Result<PreparedGraphics, VulkanError> {
        if !Arc::ptr_eq(&self.pipeline, &program.pipeline) {
            return Err(VulkanError::InvalidResource(
                "graphics state belongs to another pipeline",
            ));
        }
        Ok(PreparedGraphics {
            program,
            parameters: Arc::clone(&self.parameters),
        })
    }
}
impl VulkanResidency {
    pub fn prepare_graphics(
        &self,
        program: PreparedProgram,
        parameters: GraphicsParameters,
    ) -> Result<PreparedGraphics, VulkanError> {
        self.prepare_graphics_state(Arc::clone(&program.pipeline), parameters)?
            .bind(Arc::new(program))
    }
    pub(crate) fn prepare_graphics_state(
        &self,
        pipeline: Arc<NativePipeline>,
        parameters: GraphicsParameters,
    ) -> Result<PreparedGraphicsState, VulkanError> {
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
        let profile = self.device.profile();
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
        Ok(PreparedGraphicsState {
            pipeline,
            parameters: Arc::new(parameters),
        })
    }
    pub(crate) fn graphics_state(
        &mut self,
        graphics: &PreparedGraphics,
    ) -> Result<(), VulkanError> {
        let index = self.device.begin_recording()?;
        let slot = &mut self.device.submission.slots[index];
        if slot
            .graphics_parameters
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, &graphics.parameters))
        {
            return Ok(());
        }
        if slot
            .graphics_parameters
            .as_ref()
            .is_some_and(|old| old.same_state(&graphics.parameters))
        {
            slot.graphics_parameters = Some(Arc::clone(&graphics.parameters));
            return Ok(());
        }
        self.device.submission.statistics.graphics_updates += 1;
        let p = &graphics.parameters;
        let raw = &mut self.device.dispatch;

        unsafe {
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
            if graphics
                .program
                .pipeline
                .dynamic_states
                .contains(&vk::DynamicState::DEPTH_BOUNDS)
            {
                raw.cmd_set_depth_bounds(p.depth_bounds[0], p.depth_bounds[1]);
            }
            if self.device.dispatch.state.is_some() {
                self.device.dispatch.dynamic_state(&p.state);
            }
        }
        slot.graphics_parameters = Some(Arc::clone(&graphics.parameters));
        Ok(())
    }
}
