//! Synchronous Vulkan construction on compilation workers.
use super::*;
pub(super) struct ShaderModules {
    owner: Arc<DeviceOwner>,
    handles: Vec<(ShaderStage, vk::ShaderModule)>,
}
impl Drop for ShaderModules {
    fn drop(&mut self) {
        for &(_, h) in &self.handles {
            unsafe {
                self.owner.device.destroy_shader_module(h, None);
            }
        }
    }
}
impl Compiler {
    pub(super) fn modules(&self, shader: &ShaderArtifact) -> Result<ShaderModules, CompileError> {
        let mut result = ShaderModules {
            owner: Arc::clone(&self.owner),
            handles: Vec::new(),
        };
        for s in &shader.stages {
            let h = unsafe {
                self.owner.device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&s.module.words),
                    None,
                )
            }
            .map_err(|e| VulkanError::vk("native shader module", e))?;
            result.handles.push((s.stage, h));
        }
        Ok(result)
    }
    pub(super) fn compute(
        &self,
        modules: &ShaderModules,
        layout: &PipelineLayout,
        metrics: &mut PipelineMetrics,
    ) -> Result<vk::Pipeline, CompileError> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(modules.handles[0].1)
            .name(c"main");
        let mut feedback = vk::PipelineCreationFeedback::default();
        let mut stage_feedback = [vk::PipelineCreationFeedback::default()];
        let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut feedback)
            .pipeline_stage_creation_feedbacks(&mut stage_feedback);
        let mut info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout.handle);
        if self.feedback {
            info = info.push_next(&mut feedback_info);
        }
        let handle = unsafe {
            self.owner
                .device
                .create_compute_pipelines(self.driver.handle, &[info], None)
        }
        .map_err(|(handles, error)| {
            for handle in handles {
                unsafe {
                    self.owner.device.destroy_pipeline(handle, None);
                }
            }
            CompileError::from(VulkanError::vk("native compute pipeline", error))
        })?[0];
        feedback_metrics(metrics, feedback);
        Ok(handle)
    }
    pub(super) fn graphics(
        &self,
        modules: &ShaderModules,
        layout: &PipelineLayout,
        s: &GraphicsState,
        metrics: &mut PipelineMetrics,
    ) -> Result<vk::Pipeline, CompileError> {
        let stages: Vec<_> = modules
            .handles
            .iter()
            .map(|&(stage, module)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage_flags(stage))
                    .module(module)
                    .name(c"main")
            })
            .collect();
        let bindings: Vec<_> = s
            .vertex_bindings
            .iter()
            .map(|b| vk::VertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride,
                input_rate: b.rate,
            })
            .collect();
        let attributes: Vec<_> = s
            .vertex_attributes
            .iter()
            .map(|a| vk::VertexInputAttributeDescription {
                location: a.location,
                binding: a.binding,
                format: a.format,
                offset: a.offset,
            })
            .collect();
        let divisors: Vec<_> = s
            .vertex_divisors
            .iter()
            .map(
                |&(binding, divisor)| vk::VertexInputBindingDivisorDescriptionEXT {
                    binding,
                    divisor,
                },
            )
            .collect();
        let mut divisor_info = vk::PipelineVertexInputDivisorStateCreateInfoEXT::default()
            .vertex_binding_divisors(&divisors);
        let mut vertex = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        if !divisors.is_empty() {
            vertex = vertex.push_next(&mut divisor_info);
        }
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(s.topology)
            .primitive_restart_enable(s.primitive_restart);
        // Neutral tessellation winding uses a lower-left domain. Vulkan's
        // default upper-left origin reverses generated triangles without
        // changing TessCoord; front-face and viewport state are independent.
        // https://docs.vulkan.org/spec/latest/chapters/tessellation.html#tessellation-winding
        let mut domain = vk::PipelineTessellationDomainOriginStateCreateInfo::default()
            .domain_origin(vk::TessellationDomainOrigin::LOWER_LEFT);
        let tessellation = vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(s.patch_points)
            .push_next(&mut domain);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(s.viewport_count)
            .scissor_count(s.viewport_count);
        let mut provoking = vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT::default()
            .provoking_vertex_mode(s.provoking);
        let mut lines = vk::PipelineRasterizationLineStateCreateInfoEXT::default()
            .line_rasterization_mode(s.line_mode);
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(s.polygon)
            .cull_mode(s.cull)
            .front_face(s.front_face)
            .line_width(1.)
            .depth_clamp_enable(s.depth_clamp)
            .rasterizer_discard_enable(s.raster_discard)
            .depth_bias_enable(s.depth_bias);
        // Native modes replace WebGPU index rotation and line approximations.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineRasterizationProvokingVertexStateCreateInfoEXT.html
        if s.provoking != vk::ProvokingVertexModeEXT::FIRST_VERTEX {
            raster = raster.push_next(&mut provoking);
        }
        if s.line_mode != vk::LineRasterizationModeEXT::DEFAULT {
            raster = raster.push_next(&mut lines);
        }
        let samples = [s.sample_mask as u32, (s.sample_mask >> 32) as u32];
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(s.samples)
            .sample_mask(&samples)
            .alpha_to_coverage_enable(s.alpha_to_coverage);
        let stencil = |s: StencilState| vk::StencilOpState {
            fail_op: s.fail,
            pass_op: s.pass,
            depth_fail_op: s.depth_fail,
            compare_op: s.compare,
            compare_mask: u32::MAX,
            write_mask: u32::MAX,
            reference: 0,
        };
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(s.depth_test)
            .depth_write_enable(s.depth_write)
            .depth_compare_op(s.depth_compare)
            .depth_bounds_test_enable(s.depth_bounds)
            .stencil_test_enable(s.stencil_test)
            .front(stencil(s.front))
            .back(stencil(s.back))
            .min_depth_bounds(0.)
            .max_depth_bounds(1.);
        let attachments: Vec<_> = s
            .colors
            .iter()
            .take(
                s.colors
                    .iter()
                    .rposition(|(format, _)| *format != vk::Format::UNDEFINED)
                    .map_or(0, |i| i + 1),
            )
            .map(|(_, b)| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(b.enabled)
                    .src_color_blend_factor(b.src_color)
                    .dst_color_blend_factor(b.dst_color)
                    .color_blend_op(b.color_op)
                    .src_alpha_blend_factor(b.src_alpha)
                    .dst_alpha_blend_factor(b.dst_alpha)
                    .alpha_blend_op(b.alpha_op)
                    .color_write_mask(b.write_mask)
            })
            .collect();
        let color = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&attachments)
            .logic_op_enable(s.logic_enabled)
            .logic_op(s.logic_op);
        let dynamic_states = self.dynamic_states(s);
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut pass_key = crate::render_pass::Key {
            samples: s.samples,
            depth: if s.depth_format != vk::Format::UNDEFINED {
                s.depth_format
            } else {
                s.stencil_format
            },
            ..Default::default()
        };
        for (slot, &(format, _)) in pass_key.colors.iter_mut().zip(s.colors.iter()) {
            *slot = format;
        }
        let render_pass = self.owner.render_pass(pass_key)?;
        let mut feedback = vk::PipelineCreationFeedback::default();
        let mut stage_feedback = vec![vk::PipelineCreationFeedback::default(); stages.len()];
        let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut feedback)
            .pipeline_stage_creation_feedbacks(&mut stage_feedback);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .layout(layout.handle)
            .stages(&stages)
            .render_pass(render_pass)
            .subpass(0);
        info = info
            .vertex_input_state(&vertex)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .depth_stencil_state(&depth)
            .multisample_state(&multisample)
            .color_blend_state(&color)
            .dynamic_state(&dynamic);
        if s.patch_points != 0 {
            info = info.tessellation_state(&tessellation);
        }
        if self.feedback {
            info = info.push_next(&mut feedback_info);
        }
        let handle = unsafe {
            self.owner
                .device
                .create_graphics_pipelines(self.driver.handle, &[info], None)
        }
        .map_err(|(handles, error)| {
            for handle in handles {
                unsafe {
                    self.owner.device.destroy_pipeline(handle, None);
                }
            }
            CompileError::from(VulkanError::vk("native graphics pipeline", error))
        })?[0];
        feedback_metrics(metrics, feedback);
        Ok(handle)
    }
}
fn feedback_metrics(metrics: &mut PipelineMetrics, feedback: vk::PipelineCreationFeedback) {
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreationFeedback.html
    if feedback
        .flags
        .contains(vk::PipelineCreationFeedbackFlags::VALID)
    {
        metrics.driver_ns = Some(feedback.duration);
        metrics.application_cache_hit = feedback
            .flags
            .contains(vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT);
    }
}
