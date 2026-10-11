//! GPU-resident color conversion and filtering. Immutable per-transform bindings
//! avoid overwriting parameters still referenced by queued command buffers.
use super::*;
use nixe_gpu::{BlitFilter, BlitOperation};

#[derive(Default)]
pub(super) struct BlitCache {
    layout: Option<wgpu::BindGroupLayout>,
    pipelines: HashMap<ImageFormat, RenderPipeline>,
    pub(super) bindings: HashMap<BindingKey, BindGroup>,
    #[cfg(not(target_os = "macos"))]
    depth_transfer_supported: Option<bool>,
}
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct BindingKey {
    pub(super) source: BackendResourceHandle,
    subresource: ImageSubresourceRange,
    parameters: [u32; 16],
}

const SHADER: &str = r#"
struct Parameters { source: vec4<u32>, destination: vec4<u32>, control: vec4<u32>, sampling: vec4<f32> }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4(positions[index], 0.0, 1.0);
}
fn load(position: vec2<i32>) -> vec4<f32> {
    let low = vec2<i32>(parameters.source.xy);
    let high = low + vec2<i32>(parameters.source.zw) - vec2(1);
    return textureLoad(image, clamp(position, low, high), 0);
}
@fragment fn fragment(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let position = parameters.sampling.xy +
        (pixel.xy - vec2<f32>(parameters.destination.xy) - vec2(0.5)) * parameters.sampling.zw;
    if parameters.control.x == 0u { return load(vec2<i32>(floor(position))); }
    let centered = position - vec2(0.5);
    let base = vec2<i32>(floor(centered));
    let fraction = fract(centered);
    return mix(mix(load(base), load(base + vec2(1,0)), fraction.x),
               mix(load(base + vec2(0,1)), load(base + vec2(1,1)), fraction.x), fraction.y);
}
"#;

impl WgpuBackendDriver {
    #[cfg(not(target_os = "macos"))]
    pub(super) fn encode_depth_stencil_blit(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources<'_>,
        blit: &BlitOperation,
    ) -> Result<(wgpu::CommandBuffer, Vec<Texture>), BackendDriverError> {
        use ash::vk;
        // Vulkan preserves both aspects of a same-format nearest blit. Linear
        // depth filtering and depth format conversion are explicitly forbidden.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBlitImage.html
        if blit.filter != BlitFilter::Point
            || blit.formats != [ImageFormat::Depth24UnormStencil8Uint; 2]
        {
            return Err(unsupported("native depth/stencil blit format or filter"));
        }
        let hal = unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| unsupported("depth/stencil blit requires Vulkan"))?;
        if self.blits.depth_transfer_supported.is_none() {
            let caps = self
                .native
                .capabilities
                .ok_or_else(|| unsupported("native Vulkan capabilities unavailable"))?;
            let format = if caps.depth24_stencil8_uses_float32 {
                vk::Format::D32_SFLOAT_S8_UINT
            } else {
                vk::Format::D24_UNORM_S8_UINT
            };
            let properties = unsafe {
                hal.shared_instance()
                    .raw_instance()
                    .get_physical_device_format_properties(hal.raw_physical_device(), format)
            };
            self.blits.depth_transfer_supported = Some(
                properties
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST),
            );
        }
        if self.blits.depth_transfer_supported != Some(true) {
            return Err(unsupported("host depth/stencil format cannot blit"));
        }
        let raw_device = hal.raw_device().clone();
        drop(hal);
        let mut textures = Vec::with_capacity(2);
        for region in [blit.source, blit.destination] {
            let handle = dependency_handle(dependencies, ResourceDependency::Image(region.image))?;
            let Resource::Image {
                texture,
                description,
                ..
            } = self.resource(handle)?
            else {
                return Err(kind_mismatch(handle));
            };
            if description.kind() != nixe_gpu::ImageKind::DepthStencil
                || description.samples() != SampleCount::One
                || description.dimension() != ImageDimension::Two
            {
                return Err(unsupported(
                    "depth/stencil blit requires single-sample 2D images",
                ));
            }
            image_region_is_full(*description, region)?;
            textures.push(texture.clone());
            let view = self.attachment_view(
                dependencies,
                nixe_gpu::RenderAttachment {
                    image: region.image,
                    subresources: region.subresources,
                    format: description.format(),
                    kind: description.kind(),
                    samples: description.samples(),
                    load: nixe_gpu::AttachmentLoad::Load,
                    store: nixe_gpu::AttachmentStore::Store,
                },
            )?;
            // Register initialization with wgpu before the raw transfer, retaining
            // prior contents outside the destination rectangle.
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Nixe native blit initialization"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                ..Default::default()
            }));
        }
        encoder.transition_resources(
            std::iter::empty(),
            textures
                .iter()
                .enumerate()
                .map(|(index, texture)| wgpu::TextureTransition {
                    texture,
                    selector: None,
                    state: if index == 0 {
                        wgpu::TextureUses::COPY_SRC
                    } else {
                        wgpu::TextureUses::COPY_DST
                    },
                }),
        );
        let mut raw = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Nixe native depth/stencil blit"),
            });
        let layers = |region: ImageRegion, aspect| {
            vk::ImageSubresourceLayers::default()
                .aspect_mask(aspect)
                .mip_level(u32::from(region.subresources.mip_level))
                .base_array_layer(u32::from(region.subresources.base_layer))
                .layer_count(1)
        };
        let offsets = |region: ImageRegion| {
            [
                vk::Offset3D {
                    x: region.origin.x as i32,
                    y: region.origin.y as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: (region.origin.x + region.extent.width) as i32,
                    y: (region.origin.y + region.extent.height) as i32,
                    z: 1,
                },
            ]
        };
        let regions = [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL].map(|aspect| {
            vk::ImageBlit::default()
                .src_subresource(layers(blit.source, aspect))
                .dst_subresource(layers(blit.destination, aspect))
                .src_offsets(offsets(blit.source))
                .dst_offsets(offsets(blit.destination))
        });
        // Keep texture owners alive until the sole wgpu queue completes this
        // submission. Raw encoders do not register resource references themselves.
        unsafe {
            let src = textures[0]
                .as_hal::<wgpu::hal::api::Vulkan>()
                .unwrap()
                .raw_handle();
            let dst = textures[1]
                .as_hal::<wgpu::hal::api::Vulkan>()
                .unwrap()
                .raw_handle();
            raw.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
                let command = hal.unwrap().raw_handle();
                raw_device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                        )],
                    &[],
                    &[],
                );
                raw_device.cmd_blit_image(
                    command,
                    src,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                    vk::Filter::NEAREST,
                );
                raw_device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                        )],
                    &[],
                    &[],
                );
            });
        }
        Ok((raw.finish(), textures))
    }

    pub(super) fn encode_blit(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dependencies: &ResolvedBackendResources<'_>,
        blit: &BlitOperation,
    ) -> Result<(), BackendDriverError> {
        let source = dependency_handle(dependencies, ResourceDependency::Image(blit.source.image))?;
        let destination = dependency_handle(
            dependencies,
            ResourceDependency::Image(blit.destination.image),
        )?;
        for (handle, region, format) in [
            (source, blit.source, blit.formats[0]),
            (destination, blit.destination, blit.formats[1]),
        ] {
            let Resource::Image { description, .. } = self.resource(handle)? else {
                return Err(kind_mismatch(handle));
            };
            if description.format() != format
                || description.kind() != nixe_gpu::ImageKind::Color
                || description.dimension() != ImageDimension::Two
                || description.samples() != SampleCount::One
            {
                return Err(unsupported(
                    "color blit requires single-sample 2D color images",
                ));
            }
            if !matches!(
                format,
                ImageFormat::R8Unorm
                    | ImageFormat::Rgba8Unorm
                    | ImageFormat::Bgra8Unorm
                    | ImageFormat::Rgba8Srgb
                    | ImageFormat::Bgra8Srgb
                    | ImageFormat::Rgba16Float
                    | ImageFormat::Rg11b10Ufloat
            ) {
                return Err(unsupported(
                    "color blit requires float-sampled color formats",
                ));
            }
            image_region_is_full(*description, region)?;
        }
        if self.blits.layout.is_none() {
            self.blits.layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Nixe color blit layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: BufferSize::new(64),
                            },
                            count: None,
                        },
                    ],
                },
            ));
        }
        let layout = self.blits.layout.as_ref().unwrap();
        if !self.blits.pipelines.contains_key(&blit.formats[1]) {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Nixe color blit shader"),
                    source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Nixe color blit pipeline layout"),
                        bind_group_layouts: &[Some(layout)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Nixe color blit pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fragment"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: texture_format(blit.formats[1]).unwrap(),
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: self.pipeline_cache.as_ref(),
                });
            self.blits.pipelines.insert(blit.formats[1], pipeline);
        }
        let parameters = [
            blit.source.origin.x,
            blit.source.origin.y,
            blit.source.extent.width,
            blit.source.extent.height,
            blit.destination.origin.x,
            blit.destination.origin.y,
            blit.destination.extent.width,
            blit.destination.extent.height,
            u32::from(blit.filter == BlitFilter::Linear),
            0,
            0,
            0,
            blit.sampling_bits()[0],
            blit.sampling_bits()[1],
            blit.sampling_bits()[2],
            blit.sampling_bits()[3],
        ];
        let key = BindingKey {
            source,
            subresource: blit.source.subresources,
            parameters,
        };
        if !self.blits.bindings.contains_key(&key) {
            let Resource::Image { texture, .. } = self.resource(source)? else {
                unreachable!()
            };
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: u32::from(key.subresource.mip_level),
                mip_level_count: Some(1),
                base_array_layer: u32::from(key.subresource.base_layer),
                array_layer_count: Some(1),
                ..Default::default()
            });
            let buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("Nixe color blit parameters"),
                size: 64,
                usage: BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            {
                let mut bytes = buffer
                    .slice(..)
                    .get_mapped_range_mut()
                    .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                let packed = parameters.map(u32::to_le_bytes);
                bytes.copy_from_slice(packed.as_flattened());
            }
            buffer.unmap();
            let binding = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Nixe color blit binding"),
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            // Bound auxiliary metadata even for applications with many boxes.
            // Encoded commands retain their own references when an entry leaves.
            if self.blits.bindings.len() >= 256 {
                self.blits.bindings.clear();
            }
            self.blits.bindings.insert(key, binding);
        }
        let target = self.attachment_view(
            dependencies,
            RenderAttachment {
                image: blit.destination.image,
                subresources: blit.destination.subresources,
                kind: nixe_gpu::ImageKind::Color,
                format: blit.formats[1],
                samples: SampleCount::One,
                load: nixe_gpu::AttachmentLoad::Load,
                store: nixe_gpu::AttachmentStore::Store,
            },
        )?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Nixe color blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.blits.pipelines[&blit.formats[1]]);
        pass.set_bind_group(0, &self.blits.bindings[&key], &[]);
        pass.set_viewport(
            blit.destination.origin.x as f32,
            blit.destination.origin.y as f32,
            blit.destination.extent.width as f32,
            blit.destination.extent.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            blit.destination.origin.x,
            blit.destination.origin.y,
            blit.destination.extent.width,
            blit.destination.extent.height,
        );
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
