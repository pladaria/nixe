//! Native pipeline objects, construction and RAII destruction.
use super::*;

// Same fixed-function contract as ordinary wgpu; no shader-based blend emulation.
// https://docs.vulkan.org/spec/latest/chapters/framebuffer.html#framebuffer-blending
fn color_output(state: nixe_gpu::ColorOutputState) -> vk::PipelineColorBlendAttachmentState {
    let mut result = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
        vk::ColorComponentFlags::from_raw(u32::from(state.write_mask.bits())),
    );
    if let Some(blend) = state.blend {
        result = result
            .blend_enable(true)
            .color_blend_op(blend_operation(blend.color.operation))
            .src_color_blend_factor(blend_factor(blend.color.source))
            .dst_color_blend_factor(blend_factor(blend.color.destination))
            .alpha_blend_op(blend_operation(blend.alpha.operation))
            .src_alpha_blend_factor(blend_factor(blend.alpha.source))
            .dst_alpha_blend_factor(blend_factor(blend.alpha.destination));
    }
    result
}

fn blend_operation(value: nixe_gpu::BlendOperation) -> vk::BlendOp {
    use nixe_gpu::BlendOperation as O;
    match value {
        O::Add => vk::BlendOp::ADD,
        O::Subtract => vk::BlendOp::SUBTRACT,
        O::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        O::Min => vk::BlendOp::MIN,
        O::Max => vk::BlendOp::MAX,
    }
}

fn blend_factor(value: nixe_gpu::BlendFactor) -> vk::BlendFactor {
    use nixe_gpu::BlendFactor as F;
    match value {
        F::Zero => vk::BlendFactor::ZERO,
        F::One => vk::BlendFactor::ONE,
        F::SourceColor => vk::BlendFactor::SRC_COLOR,
        F::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        F::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
        F::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        F::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
        F::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        F::DestinationColor => vk::BlendFactor::DST_COLOR,
        F::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        F::SourceAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
    }
}

pub(super) struct NativePipeline {
    _device: Device,
    pub(super) raw: ash::Device,
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) pass: vk::RenderPass,
    pub(super) shaders: NativeShaders,
    pub(super) set_layout: vk::DescriptorSetLayout,
    pub(super) descriptor_stages: vk::PipelineStageFlags,
    pub(super) descriptors: Mutex<descriptors::DescriptorArena>,
}
impl Drop for NativePipeline {
    fn drop(&mut self) {
        // Submitted users retain an Arc until completion, including cache eviction.
        unsafe {
            self.raw.destroy_pipeline(self.pipeline, None);
            self.raw.destroy_pipeline_layout(self.layout, None);
            self.raw
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.raw.destroy_render_pass(self.pass, None);
        }
    }
}

fn format(format: ImageFormat, float_depth24: bool) -> Result<vk::Format, BackendDriverError> {
    Ok(match format {
        ImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        ImageFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        ImageFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        ImageFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        ImageFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        ImageFormat::Depth16Unorm => vk::Format::D16_UNORM,
        ImageFormat::Depth24UnormStencil8Uint => {
            if float_depth24 {
                vk::Format::D32_SFLOAT_S8_UINT
            } else {
                vk::Format::D24_UNORM_S8_UINT
            }
        }
        ImageFormat::Depth32Float => vk::Format::D32_SFLOAT,
        _ => return Err(unsupported("native tessellation attachment format")),
    })
}

impl NativePipeline {
    pub(super) fn new(
        device: &Device,
        key: &PipelineKey,
        shaders: NativeShaders,
        capabilities: crate::VulkanNativeCapabilities,
        cache: &mut persistent_cache::NativePipelineCache,
    ) -> Result<Arc<Self>, BackendDriverError> {
        let (bindings, descriptor_stages) =
            descriptors::layout(shaders.bindings(), capabilities.graphics_limits)?;
        if !bindings.is_empty() && !capabilities.robust_buffer_access {
            return Err(unsupported(
                "native constant-buffer reads require enabled buffer robustness",
            ));
        }
        let float_depth24 = capabilities.depth24_stencil8_uses_float32;
        if key.draw.color_outputs[0].blend.is_some() && key.color == ImageFormat::Rgba32Float {
            return Err(unsupported(
                "native float32 attachment blending requires format capability negotiation",
            ));
        }
        let raw = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| unsupported("native tessellation needs a Vulkan device"))?
            .raw_device()
            .clone();
        let mut p = Self {
            _device: device.clone(),
            raw,
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            pass: vk::RenderPass::null(),
            shaders,
            set_layout: vk::DescriptorSetLayout::null(),
            descriptor_stages,
            descriptors: Mutex::new(descriptors::DescriptorArena::default()),
        };
        let mut descriptions = vec![
            vk::AttachmentDescription::default()
                .format(format(key.color, float_depth24)?)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];
        if let Some(depth) = key.depth {
            descriptions.push(
                vk::AttachmentDescription::default()
                    .format(format(depth, float_depth24)?)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::LOAD)
                    .stencil_store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            );
        }
        let colors = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&colors);
        if key.depth.is_some() {
            subpass = subpass.depth_stencil_attachment(&depth_ref);
        }
        let subpasses = [subpass];
        let constants = (p.shaders.push_constant_bytes() != 0).then_some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::TESSELLATION_CONTROL,
            offset: 0,
            size: p.shaders.push_constant_bytes(),
        });
        // All array storage lives through creation; RAII cleans up partial failure.
        unsafe {
            let set_layouts = if bindings.is_empty() {
                Vec::new()
            } else {
                p.set_layout = p
                    .raw
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                    .map_err(error)?;
                vec![p.set_layout]
            };
            p.pass = p
                .raw
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&descriptions)
                        .subpasses(&subpasses),
                    None,
                )
                .map_err(error)?;
            p.layout = p
                .raw
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(constants.as_slice()),
                    None,
                )
                .map_err(error)?;
        }
        struct Modules<'a>(&'a ash::Device, Vec<vk::ShaderModule>);
        impl Drop for Modules<'_> {
            fn drop(&mut self) {
                for &m in &self.1 {
                    unsafe { self.0.destroy_shader_module(m, None) };
                }
            }
        }
        let mut modules = Modules(&p.raw, Vec::new());
        for module in p.shaders.modules() {
            modules.1.push(
                unsafe {
                    p.raw.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(module.words()),
                        None,
                    )
                }
                .map_err(error)?,
            );
        }
        let stages: Vec<_> = modules
            .1
            .iter()
            .zip(p.shaders.stages())
            .map(|(&module, &stage)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .module(module)
                    .stage(stage)
                    .name(c"main")
            })
            .collect();
        let mut bindings = Vec::new();
        let mut attributes = Vec::new();
        for (index, layout) in key.draw.vertex_buffers.iter().enumerate() {
            bindings.push(vk::VertexInputBindingDescription {
                binding: index as u32,
                stride: u32::try_from(layout.array_stride).map_err(error)?,
                input_rate: match layout.step_mode {
                    VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                },
            });
            for attribute in layout.attributes.iter() {
                let f = match attribute.format {
                    VertexFormat::Float32 => vk::Format::R32_SFLOAT,
                    VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
                    VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
                    VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
                    _ => return Err(unsupported("native tessellation vertex format")),
                };
                attributes.push(vk::VertexInputAttributeDescription {
                    location: attribute.shader_location,
                    binding: index as u32,
                    format: f,
                    offset: u32::try_from(attribute.offset).map_err(error)?,
                });
            }
        }
        let vertex = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(match key.draw.topology {
                PrimitiveTopology::Patches => vk::PrimitiveTopology::PATCH_LIST,
                PrimitiveTopology::Lines => vk::PrimitiveTopology::LINE_LIST,
                PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
                _ => return Err(unsupported("native primitive topology")),
            });
        // Neutral winding follows the lower-left GLSL domain. Vulkan defaults
        // to upper-left, reversing winding even though TessCoord is unchanged.
        // Do not compensate by changing frontFace or the viewport: both also
        // govern guest state independently of the tessellator.
        // https://docs.vulkan.org/spec/latest/chapters/tessellation.html#tessellation-winding
        // Native device creation already requires Vulkan 1.1.
        let mut domain = vk::PipelineTessellationDomainOriginStateCreateInfo::default()
            .domain_origin(vk::TessellationDomainOrigin::LOWER_LEFT);
        let tess = vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(u32::from(p.shaders.input_control_points()))
            .push_next(&mut domain);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let mut line = vk::PipelineRasterizationLineStateCreateInfoKHR::default();
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match key.draw.cull_mode {
                nixe_gpu::CullMode::None => vk::CullModeFlags::NONE,
                nixe_gpu::CullMode::Front => vk::CullModeFlags::FRONT,
                nixe_gpu::CullMode::Back => vk::CullModeFlags::BACK,
                nixe_gpu::CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
            })
            .front_face(match key.draw.front_face {
                nixe_gpu::FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
                nixe_gpu::FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
            })
            .line_width(1.0);
        if let TriangleRasterization::Wireframe { smooth, .. } = key.draw.triangle_rasterization {
            // Explicit mode, never implementation-dependent DEFAULT lines.
            // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-lines-smooth
            line.line_rasterization_mode = if smooth {
                vk::LineRasterizationModeKHR::RECTANGULAR_SMOOTH
            } else {
                vk::LineRasterizationModeKHR::RECTANGULAR
            };
            raster = raster
                .polygon_mode(vk::PolygonMode::LINE)
                .push_next(&mut line);
        } else if let Some(state) = key.draw.line_rasterization {
            line.line_rasterization_mode = if state.smooth {
                vk::LineRasterizationModeKHR::RECTANGULAR_SMOOTH
            } else {
                vk::LineRasterizationModeKHR::RECTANGULAR
            };
            // Direct line primitives keep polygonMode FILL; fillModeNonSolid
            // is only needed for polygon wireframe, not for line strips.
            raster = raster.push_next(&mut line);
        }
        let samples = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let state = key.draw.depth_state;
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(state.test_enabled)
            .depth_write_enable(state.test_enabled && state.write_enabled)
            .depth_compare_op(match state.compare {
                DepthCompareOperation::Never => vk::CompareOp::NEVER,
                DepthCompareOperation::Less => vk::CompareOp::LESS,
                DepthCompareOperation::Equal => vk::CompareOp::EQUAL,
                DepthCompareOperation::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
                DepthCompareOperation::Greater => vk::CompareOp::GREATER,
                DepthCompareOperation::NotEqual => vk::CompareOp::NOT_EQUAL,
                DepthCompareOperation::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
                DepthCompareOperation::Always => vk::CompareOp::ALWAYS,
            });
        let writes = [color_output(key.draw.color_outputs[0])];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&writes);
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
        ];
        let count = if matches!(
            key.draw.triangle_rasterization,
            TriangleRasterization::Wireframe { .. }
        ) || key.draw.line_rasterization.is_some()
        {
            3
        } else {
            2
        };
        let dynamic =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states[..count]);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&samples)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(p.layout)
            .render_pass(p.pass);
        if key.draw.tessellation.is_some() {
            info = info.tessellation_state(&tess);
        }
        p.pipeline = match unsafe {
            p.raw
                .create_graphics_pipelines(cache.handle(), &[info], None)
        } {
            Ok(created) => created[0],
            Err((partial, result)) => {
                for handle in partial {
                    unsafe { p.raw.destroy_pipeline(handle, None) };
                }
                return Err(error(result));
            }
        };
        drop(modules);
        Ok(Arc::new(p))
    }
}
