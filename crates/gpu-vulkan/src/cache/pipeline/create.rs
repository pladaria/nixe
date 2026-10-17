//! Synchronous Vulkan construction on compilation workers.
use super::*;
// One creation implementation with explicit Vulkan subsets, rather than
// independent renderer paths or ambiguous part/link argument combinations.
// https://docs.vulkan.org/samples/latest/samples/extensions/graphics_pipeline_library/README.html
pub(super) enum GraphicsCompilation<'a> {
    Monolithic,
    Library(vk::GraphicsPipelineLibraryFlagsEXT),
    Link(&'a [vk::Pipeline]),
}
struct ShaderModules {
    owner: Arc<DeviceOwner>,
    handles: Vec<vk::ShaderModule>,
}
impl Drop for ShaderModules {
    fn drop(&mut self) {
        for &h in &self.handles {
            unsafe {
                self.owner.device.destroy_shader_module(h, None);
            }
        }
    }
}
impl Compiler {
    fn modules(&self, stages: &[&CompiledStage]) -> Result<ShaderModules, CompileError> {
        let mut result = ShaderModules {
            owner: Arc::clone(&self.owner),
            handles: Vec::new(),
        };
        for s in stages {
            let h = if self.inline_modules {
                vk::ShaderModule::null()
            } else {
                unsafe {
                    self.owner.device.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&s.module.words),
                        None,
                    )
                }
                .map_err(|e| VulkanError::vk("native shader module", e))?
            };
            result.handles.push(h);
        }
        Ok(result)
    }
    pub(super) fn compute(
        &self,
        worker: usize,
        shader: &ShaderArtifact,
        layout: &PipelineLayout,
        metrics: &mut PipelineMetrics,
    ) -> Result<vk::Pipeline, CompileError> {
        let stages = [shader.stages[0].as_ref()];
        let modules = self.modules(&stages)?;
        let mut module_info = vk::ShaderModuleCreateInfo::default().code(&stages[0].module.words);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(modules.handles[0])
            .name(c"main");
        if self.inline_modules {
            stage = stage.push_next(&mut module_info);
        }
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
        let handle = self.driver.with(worker, |cache| {
            let mut infos = [info];
            // A driver-cache hit avoids compilation. A miss is compiled here on
            // the worker, never dropped or moved onto the queue owner.
            if self.cache_control {
                infos[0].flags |= vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
            }
            let first = unsafe {
                self.owner
                    .device
                    .create_compute_pipelines(cache, &infos, None)
            };
            let result = if matches!(&first, Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED))) {
                destroy_partial(&self.owner, &first);
                infos[0].flags &= !vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
                unsafe {
                    self.owner
                        .device
                        .create_compute_pipelines(cache, &infos, None)
                }
            } else {
                first
            };
            let handle = result.map_err(|(handles, e)| {
                for h in handles {
                    unsafe {
                        self.owner.device.destroy_pipeline(h, None);
                    }
                }
                CompileError::from(VulkanError::vk("native compute pipeline", e))
            })?[0];
            Ok(handle)
        })?;
        feedback_metrics(metrics, feedback);
        Ok(handle)
    }
    pub(super) fn graphics(
        &self,
        worker: usize,
        shader: &ShaderArtifact,
        layout: &PipelineLayout,
        s: &GraphicsState,
        mode: GraphicsCompilation<'_>,
        metrics: &mut PipelineMetrics,
    ) -> Result<vk::Pipeline, CompileError> {
        let (part, libraries) = match mode {
            GraphicsCompilation::Monolithic => (None, &[][..]),
            GraphicsCompilation::Library(part) => (Some(part), &[][..]),
            GraphicsCompilation::Link(libraries) => (None, libraries),
        };
        let linking = !libraries.is_empty();
        let mask = if linking {
            vk::GraphicsPipelineLibraryFlagsEXT::empty()
        } else {
            part.unwrap_or(
                vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE
                    | vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS
                    | vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER
                    | vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE,
            )
        };
        let vi = mask.contains(vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE);
        let pre = mask.contains(vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS);
        let fs = mask.contains(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER);
        let out = mask.contains(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE);
        let selected: Vec<_> = shader
            .stages
            .iter()
            .map(AsRef::as_ref)
            .filter(|s| {
                if s.stage == ShaderStage::Fragment {
                    fs
                } else {
                    pre
                }
            })
            .collect();
        let modules = self.modules(&selected)?;
        let mut module_infos: Vec<_> = selected
            .iter()
            .map(|s| vk::ShaderModuleCreateInfo::default().code(&s.module.words))
            .collect();
        let stages: Vec<_> = selected
            .iter()
            .zip(&modules.handles)
            .zip(module_infos.iter_mut())
            .map(|((s, &module), info)| {
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage_flags(s.stage))
                    .module(module)
                    .name(c"main");
                if self.inline_modules {
                    stage.push_next(info)
                } else {
                    stage
                }
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
        let vertex = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
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
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic);
        let formats: Vec<_> = if out || linking {
            s.colors.iter().map(|(f, _)| *f).collect()
        } else {
            Vec::new()
        };
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(if fs || out || linking {
                s.depth_format
            } else {
                vk::Format::UNDEFINED
            })
            .stencil_attachment_format(if fs || out || linking {
                s.stencil_format
            } else {
                vk::Format::UNDEFINED
            });
        let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(mask);
        let mut linking_info = vk::PipelineLibraryCreateInfoKHR::default().libraries(libraries);
        let mut feedback = vk::PipelineCreationFeedback::default();
        let mut stage_feedback = vec![vk::PipelineCreationFeedback::default(); stages.len()];
        let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut feedback)
            .pipeline_stage_creation_feedbacks(&mut stage_feedback);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .layout(layout.handle)
            .stages(&stages)
            .push_next(&mut rendering);
        if vi {
            info = info
                .vertex_input_state(&vertex)
                .input_assembly_state(&assembly);
        }
        if pre {
            info = info.viewport_state(&viewport).rasterization_state(&raster);
            if s.patch_points != 0 {
                info = info.tessellation_state(&tessellation);
            }
        }
        if fs {
            info = info.depth_stencil_state(&depth);
        }
        if fs || out {
            info = info.multisample_state(&multisample);
        }
        if out {
            info = info.color_blend_state(&color);
        }
        if !linking {
            info = info.dynamic_state(&dynamic);
        }
        if part.is_some() {
            info = info
                .flags(vk::PipelineCreateFlags::LIBRARY_KHR)
                .push_next(&mut library);
        }
        if linking {
            info = info.push_next(&mut linking_info);
        }
        if self.feedback {
            info = info.push_next(&mut feedback_info);
        }
        let handle = self.driver.with(worker, |cache| {
            let mut infos = [info];
            if self.cache_control {
                infos[0].flags |= vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
            }
            let first = unsafe {
                self.owner
                    .device
                    .create_graphics_pipelines(cache, &infos, None)
            };
            let result = if matches!(&first, Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED))) {
                destroy_partial(&self.owner, &first);
                infos[0].flags &= !vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
                unsafe {
                    self.owner
                        .device
                        .create_graphics_pipelines(cache, &infos, None)
                }
            } else {
                first
            };
            let handle = result.map_err(|(handles, e)| {
                for h in handles {
                    unsafe {
                        self.owner.device.destroy_pipeline(h, None);
                    }
                }
                CompileError::from(VulkanError::vk("native graphics pipeline", e))
            })?[0];
            Ok(handle)
        })?;
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
fn destroy_partial(
    owner: &DeviceOwner,
    result: &Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
) {
    if let Err((handles, _)) = result {
        for &h in handles {
            unsafe {
                owner.device.destroy_pipeline(h, None);
            }
        }
    }
}
