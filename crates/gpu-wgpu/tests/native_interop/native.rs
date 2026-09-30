use std::{path::Path, process::Command, sync::Arc};

use ash::vk;

use super::{SIZE, device::Context};

// All native objects are test-owned. Nothing here is a guest shader workaround.
pub struct Pipeline {
    _device: wgpu::Device,
    raw: ash::Device,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pass: vk::RenderPass,
    sampler: vk::Sampler,
    modules: Vec<vk::ShaderModule>,
    patch_points: u32,
    parameter_bytes: u32,
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
}

fn compile(stage: &str, dir: &Path) -> Vec<u32> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/native_interop/shaders")
        .join(format!("patch.{stage}"));
    let output = dir.join(format!("{stage}.spv"));
    let result =
        Command::new(std::env::var_os("GLSLANG_VALIDATOR").unwrap_or("glslangValidator".into()))
            .args(["-V", "--target-env", "vulkan1.1", "-o"])
            .arg(&output)
            .arg(source)
            .output()
            .expect("glslangValidator required for host fixtures");
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let result = Command::new(std::env::var_os("SPIRV_VAL").unwrap_or("spirv-val".into()))
        .args(["--target-env", "vulkan1.1"])
        .arg(&output)
        .output()
        .expect("spirv-val required");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(output).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

impl Pipeline {
    pub fn new(ctx: &Context) -> Arc<Self> {
        let dir = tempfile::tempdir().unwrap();
        let code = ["vert", "tesc", "tese", "frag"].map(|stage| compile(stage, dir.path()));
        let bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::TESSELLATION_CONTROL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::TESSELLATION_EVALUATION),
        ];
        Self::with_patch(ctx, &code, 3, 0, bindings)
    }

    pub fn from_tessellation(
        ctx: &Context,
        shaders: &nixe_gpu::SpirvTessellationShaders,
    ) -> Arc<Self> {
        use nixe_gpu::PipelineStages as S;
        let bindings = shaders
            .bindings()
            .iter()
            .map(|binding| {
                assert_eq!(
                    binding.resource.kind(),
                    nixe_gpu::ShaderResourceKind::ConstantBuffer
                );
                let mut flags = vk::ShaderStageFlags::empty();
                for (neutral, native) in [
                    (S::VERTEX_SHADER, vk::ShaderStageFlags::VERTEX),
                    (
                        S::TESSELLATION_CONTROL_SHADER,
                        vk::ShaderStageFlags::TESSELLATION_CONTROL,
                    ),
                    (
                        S::TESSELLATION_EVALUATION_SHADER,
                        vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                    ),
                    (S::FRAGMENT_SHADER, vk::ShaderStageFlags::FRAGMENT),
                ] {
                    if binding.stages.contains(neutral) {
                        flags |= native;
                    }
                }
                vk::DescriptorSetLayoutBinding::default()
                    .binding(u32::from(binding.resource.binding()))
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(flags)
            })
            .collect();
        let code = std::array::from_fn(|i| shaders.modules()[i].words().to_vec());
        Self::with_patch(
            ctx,
            &code,
            u32::from(shaders.input_control_points()),
            shaders.push_constant_bytes(),
            bindings,
        )
    }

    fn with_patch(
        ctx: &Context,
        code: &[Vec<u32>; 4],
        patch_points: u32,
        parameter_bytes: u32,
        bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    ) -> Arc<Self> {
        let mut result = Self {
            _device: ctx.device.clone(),
            raw: ctx.raw(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pass: vk::RenderPass::null(),
            sampler: vk::Sampler::null(),
            modules: Vec::new(),
            patch_points,
            parameter_bytes,
            bindings,
        };
        // All referenced arrays live until Vulkan has consumed the create infos.
        unsafe {
            result.set_layout = result
                .raw
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&result.bindings),
                    None,
                )
                .unwrap();
            let push = (parameter_bytes != 0).then_some(vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::TESSELLATION_CONTROL,
                offset: 0,
                size: parameter_bytes,
            });
            result.layout = result
                .raw
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[result.set_layout])
                        .push_constant_ranges(push.as_slice()),
                    None,
                )
                .unwrap();
            result.sampler = result
                .raw
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .max_lod(0.0),
                    None,
                )
                .unwrap();
            let attachments = [
                vk::AttachmentDescription::default()
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                vk::AttachmentDescription::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            ];
            let colors = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors)
                .depth_stencil_attachment(&depth)];
            result.pass = result
                .raw
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses),
                    None,
                )
                .unwrap();
            for code in code {
                result.modules.push(
                    result
                        .raw
                        .create_shader_module(
                            &vk::ShaderModuleCreateInfo::default().code(code),
                            None,
                        )
                        .unwrap(),
                );
            }
            let flags = [
                vk::ShaderStageFlags::VERTEX,
                vk::ShaderStageFlags::TESSELLATION_CONTROL,
                vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                vk::ShaderStageFlags::FRAGMENT,
            ];
            let stages: Vec<_> = result
                .modules
                .iter()
                .zip(flags)
                .map(|(&module, stage)| {
                    vk::PipelineShaderStageCreateInfo::default()
                        .module(module)
                        .stage(stage)
                        .name(c"main")
                })
                .collect();
            let vertex = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::PATCH_LIST);
            let tess = vk::PipelineTessellationStateCreateInfo::default()
                .patch_control_points(patch_points);
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: SIZE as f32,
                height: SIZE as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: SIZE,
                    height: SIZE,
                },
            }];
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewports)
                .scissors(&scissors);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .line_width(1.0);
            let samples = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
            let color = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&color);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex)
                .input_assembly_state(&assembly)
                .tessellation_state(&tess)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&samples)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .layout(result.layout)
                .render_pass(result.pass);
            result.pipeline =
                match result
                    .raw
                    .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                {
                    Ok(pipelines) => pipelines[0],
                    Err((partial, error)) => {
                        for pipeline in partial {
                            result.raw.destroy_pipeline(pipeline, None);
                        }
                        panic!("native pipeline creation: {error:?}");
                    }
                };
        }
        Arc::new(result)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // Ownership is retained by completion callbacks; no GPU work references
        // these objects at destruction. wgpu still owns the logical device.
        unsafe {
            self.raw.destroy_pipeline(self.pipeline, None);
            self.raw.destroy_pipeline_layout(self.layout, None);
            self.raw
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.raw.destroy_render_pass(self.pass, None);
            self.raw.destroy_sampler(self.sampler, None);
            for module in &self.modules {
                self.raw.destroy_shader_module(*module, None);
            }
        }
    }
}

pub struct Bindings {
    pipeline: Arc<Pipeline>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    framebuffer: vk::Framebuffer,
    // Raw encoding is invisible to wgpu's lifetime tracker. These owners must
    // survive submission completion; explicitly calling destroy() is forbidden.
    _buffer: wgpu::Buffer,
    _textures: [wgpu::Texture; 3],
    _views: [wgpu::TextureView; 3],
}

impl Bindings {
    pub fn new(
        pipeline: Arc<Pipeline>,
        buffer: &wgpu::Buffer,
        textures: [&wgpu::Texture; 3],
        views: [&wgpu::TextureView; 3],
    ) -> Arc<Self> {
        let mut result = Self {
            pipeline,
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            framebuffer: vk::Framebuffer::null(),
            _buffer: buffer.clone(),
            _textures: textures.map(Clone::clone),
            _views: views.map(Clone::clone),
        };
        unsafe {
            let raw = &result.pipeline.raw;
            let sizes: Vec<_> = result
                .pipeline
                .bindings
                .iter()
                .map(|b| vk::DescriptorPoolSize {
                    ty: b.descriptor_type,
                    descriptor_count: 1,
                })
                .collect();
            result.pool = raw
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .unwrap();
            result.set = raw
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(result.pool)
                        .set_layouts(&[result.pipeline.set_layout]),
                )
                .unwrap()[0];
            let raw_views =
                views.map(|v| v.as_hal::<wgpu::hal::api::Vulkan>().unwrap().raw_handle());
            let image_info = [vk::DescriptorImageInfo {
                sampler: result.pipeline.sampler,
                image_view: raw_views[2],
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let buffer_info = [vk::DescriptorBufferInfo {
                buffer: buffer
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .unwrap()
                    .raw_handle(),
                offset: 0,
                range: 16,
            }];
            let writes: Vec<_> = result
                .pipeline
                .bindings
                .iter()
                .map(|binding| {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(result.set)
                        .dst_binding(binding.binding)
                        .descriptor_type(binding.descriptor_type);
                    match binding.descriptor_type {
                        vk::DescriptorType::STORAGE_BUFFER => write.buffer_info(&buffer_info),
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER => write.image_info(&image_info),
                        _ => panic!("unhandled fixture descriptor"),
                    }
                })
                .collect();
            raw.update_descriptor_sets(&writes, &[]);
            result.framebuffer = raw
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(result.pipeline.pass)
                        .attachments(&raw_views[..2])
                        .width(SIZE)
                        .height(SIZE)
                        .layers(1),
                    None,
                )
                .unwrap();
        }
        Arc::new(result)
    }

    pub fn encode(&self, ctx: &Context, draws: u32) -> wgpu::CommandBuffer {
        self.encode_parameters(ctx, draws, &[])
    }

    pub fn encode_parameters(
        &self,
        ctx: &Context,
        draws: u32,
        parameters: &[u8],
    ) -> wgpu::CommandBuffer {
        assert_eq!(parameters.len(), self.pipeline.parameter_bytes as usize);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("native tessellation segment"),
            });
        // No wgpu recording APIs on this encoder, including transition_resources.
        // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-core/src/command/mod.rs
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
                let cmd = hal.unwrap().raw_handle();
                let raw = &self.pipeline.raw;
                let tess_stages = vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER
                    | vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER;
                let native_shader_stages = tess_stages
                    | vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER;
                let attachment_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                let normal_stages = vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | attachment_stages;
                let writes = vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                let attachment_access = vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                // Segment-level dependency, not a per-draw ALL_COMMANDS barrier.
                // wgpu transitions establish layouts; these extend visibility to
                // TCS/TES, which HAL 30's generic shader stage masks omit.
                // https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-dependencies
                raw.cmd_pipeline_barrier(
                    cmd,
                    normal_stages,
                    native_shader_stages | attachment_stages,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(writes)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ | attachment_access)],
                    &[],
                    &[],
                );
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(self.pipeline.pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: SIZE,
                            height: SIZE,
                        },
                    });
                raw.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
                raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &[self.set],
                    &[],
                );
                for _ in 0..draws {
                    if !parameters.is_empty() {
                        raw.cmd_push_constants(
                            cmd,
                            self.pipeline.layout,
                            vk::ShaderStageFlags::TESSELLATION_CONTROL,
                            0,
                            parameters,
                        );
                    }
                    raw.cmd_draw(cmd, self.pipeline.patch_points, 1, 0, 0);
                }
                raw.cmd_end_render_pass(cmd);
                // Includes execution ordering for subsequent writes after TCS/TES
                // reads. Layouts stay exactly as declared to the wgpu tracker.
                raw.cmd_pipeline_barrier(
                    cmd,
                    native_shader_stages | attachment_stages,
                    normal_stages,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .dst_access_mask(
                            writes
                                | vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::TRANSFER_READ
                                | attachment_access,
                        )],
                    &[],
                    &[],
                );
            });
        }
        encoder.finish()
    }
}

impl Drop for Bindings {
    fn drop(&mut self) {
        unsafe {
            self.pipeline
                .raw
                .destroy_framebuffer(self.framebuffer, None);
            self.pipeline.raw.destroy_descriptor_pool(self.pool, None);
        }
    }
}
