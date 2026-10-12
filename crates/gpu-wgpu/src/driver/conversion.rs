//! Integer NV12 conversion into ordinary, canonically backed resident images.
use super::*;
use nixe_gpu::{ImageExtent, Nv12ColorTransform, Nv12Conversion};

#[derive(Default)]
pub(super) struct ConversionCache {
    layout: Option<wgpu::BindGroupLayout>,
    pipelines: HashMap<ImageFormat, RenderPipeline>,
    pub(super) bindings: HashMap<BindingKey, BindGroup>,
}
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct BindingKey {
    pub(super) sources: [BackendResourceHandle; 2],
    transform: Nv12ColorTransform,
}

const SHADER: &str = r#"
struct Parameters {
    red: vec4<i32>, green: vec4<i32>, blue: vec4<i32>,
    control: vec4<u32>, clamp_alpha: vec4<i32>,
}
@group(0) @binding(0) var luma: texture_2d<f32>;
@group(0) @binding(1) var chroma: texture_2d<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4(positions[index], 0.0, 1.0);
}
fn channel(value: vec3<i32>, row: vec4<i32>) -> i32 {
    return (((value.x * row.x + value.y * row.y + value.z * row.z)
        >> parameters.control.z) + row.w) >> 8u;
}
@fragment fn fragment(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let position = vec2<i32>(pixel.xy) + vec2<i32>(parameters.control.xy);
    let y = i32(round(textureLoad(luma, position, 0).r * 255.0));
    let uv = vec2<i32>(round(textureLoad(chroma, position / 2, 0).rg * 255.0));
    let value = vec3<i32>(y, uv) * 4;
    var rgb = value;
    if parameters.control.w != 0u {
        rgb = vec3<i32>(channel(value, parameters.red), channel(value, parameters.green), channel(value, parameters.blue));
    }
    rgb = clamp(clamp(rgb, vec3(parameters.clamp_alpha.x), vec3(parameters.clamp_alpha.y)), vec3(0), vec3(1023)) >> vec3(2u);
    return vec4<f32>(vec3<f32>(rgb), f32(parameters.clamp_alpha.z)) / 255.0;
}
"#;

impl WgpuBackendDriver {
    pub(super) fn encode_nv12_conversion(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources<'_>,
        conversion: &Nv12Conversion,
    ) -> Result<(), BackendDriverError> {
        let t = conversion.transform;
        // Three signed 20-bit coefficients times a 10-bit component fit i32,
        // including translation after the first shift.
        if t.shift > 15
            || t.clamp[0] < 0
            || t.clamp[1] > 1023
            || t.clamp[0] > t.clamp[1]
            || t.coefficients
                .iter()
                .flatten()
                .any(|c| !(-524288..=524287).contains(c))
        {
            return Err(unsupported("NV12 fixed-point matrix or clamp"));
        }
        let sources = [conversion.luma, conversion.chroma]
            .map(|region| dependency_handle(dependencies, ResourceDependency::Image(region.image)));
        let sources = [sources[0].clone()?, sources[1].clone()?];
        let destination = dependency_handle(
            dependencies,
            ResourceDependency::Image(conversion.destination.image),
        )?;
        let mut extents = [ImageExtent {
            width: 0,
            height: 0,
            depth: 1,
        }; 2];
        for (index, ((handle, region), format)) in sources
            .into_iter()
            .zip([conversion.luma, conversion.chroma])
            .zip([ImageFormat::R8Unorm, ImageFormat::Rg8Unorm])
            .enumerate()
        {
            let Resource::Image { description, .. } = self.resource(handle)? else {
                return Err(kind_mismatch(handle));
            };
            if description.format() != format
                || description.dimension() != ImageDimension::Two
                || description.samples() != SampleCount::One
                || description.array_layers() != 1
                || description.mip_levels() != 1
                || !image_region_is_full(*description, region)?
            {
                return Err(unsupported(
                    "NV12 source must be a complete single-sample 2D plane",
                ));
            }
            extents[index] = description.extent();
        }
        let Resource::Image { description, .. } = self.resource(destination)? else {
            return Err(kind_mismatch(destination));
        };
        if !matches!(
            conversion.destination_format,
            ImageFormat::Rgba8Unorm | ImageFormat::Bgra8Unorm
        ) || description.format() != conversion.destination_format
            || description.dimension() != ImageDimension::Two
            || description.samples() != SampleCount::One
            || description.array_layers() != 1
            || description.mip_levels() != 1
            || !image_region_is_full(*description, conversion.destination)?
        {
            return Err(unsupported(
                "NV12 destination must be a complete RGBA8/BGRA8 image",
            ));
        }
        let end = [
            t.source_origin[0].checked_add(description.extent().width),
            t.source_origin[1].checked_add(description.extent().height),
        ];
        if end[0].is_none_or(|v| v > extents[0].width || v.div_ceil(2) > extents[1].width)
            || end[1].is_none_or(|v| v > extents[0].height || v.div_ceil(2) > extents[1].height)
        {
            return Err(unsupported("NV12 source crop exceeds its planes"));
        }
        if self.conversions.layout.is_none() {
            let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            self.conversions.layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Nixe NV12 conversion layout"),
                    entries: &[
                        texture_entry(0),
                        texture_entry(1),
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: BufferSize::new(80),
                            },
                            count: None,
                        },
                    ],
                },
            ));
        }
        let layout = self.conversions.layout.as_ref().unwrap();
        if !self
            .conversions
            .pipelines
            .contains_key(&conversion.destination_format)
        {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Nixe integer NV12 conversion"),
                    source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Nixe NV12 pipeline layout"),
                        bind_group_layouts: &[Some(layout)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Nixe NV12 conversion"),
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
                            format: texture_format(conversion.destination_format).unwrap(),
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
            self.conversions
                .pipelines
                .insert(conversion.destination_format, pipeline);
        }
        let key = BindingKey {
            sources,
            transform: t,
        };
        if !self.conversions.bindings.contains_key(&key) {
            let views = sources.map(|handle| {
                let Resource::Image { texture, .. } = self.resource(handle).unwrap() else {
                    unreachable!("NV12 source was validated above")
                };
                texture.create_view(&Default::default())
            });
            let parameters = [
                t.coefficients[0],
                t.coefficients[1],
                t.coefficients[2],
                [
                    t.source_origin[0] as i32,
                    t.source_origin[1] as i32,
                    t.shift as i32,
                    i32::from(t.enabled),
                ],
                [t.clamp[0], t.clamp[1], i32::from(t.alpha), 0],
            ];
            let buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("Nixe immutable NV12 parameters"),
                size: 80,
                usage: BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            {
                let mut mapped = buffer
                    .slice(..)
                    .get_mapped_range_mut()
                    .map_err(|e| BackendDriverError::failure(e.to_string()))?;
                for (index, value) in parameters.into_iter().flatten().enumerate() {
                    mapped
                        .slice(index * 4..index * 4 + 4)
                        .copy_from_slice(&value.to_le_bytes());
                }
            }
            buffer.unmap();
            let binding = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Nixe NV12 binding"),
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&views[0]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&views[1]),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            if self.conversions.bindings.len() >= 256 {
                self.conversions.bindings.clear();
            }
            self.conversions.bindings.insert(key, binding);
        }
        let target = self.attachment_view(
            dependencies,
            RenderAttachment {
                image: conversion.destination.image,
                subresources: conversion.destination.subresources,
                kind: nixe_gpu::ImageKind::Color,
                format: conversion.destination_format,
                samples: SampleCount::One,
                load: nixe_gpu::AttachmentLoad::Discard,
                store: nixe_gpu::AttachmentStore::Store,
            },
        )?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Nixe resident NV12 conversion"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.conversions.pipelines[&conversion.destination_format]);
        pass.set_bind_group(0, &self.conversions.bindings[&key], &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl WgpuBackendDriver {
    pub(super) fn encode_device_video(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources<'_>,
        destinations: [ImageRegion; 2],
        frame: &nixe_gpu::DeviceImageFrame,
        token: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        let uuid = self
            .video_device
            .ok_or_else(|| unsupported("external device memory unavailable"))?;
        let limit = self.device.limits().max_texture_dimension_2d;
        if frame.width == 0 || frame.height == 0 || frame.width > limit || frame.height > limit {
            return Err(unsupported("device video dimensions"));
        }
        for (plane, destination) in destinations.iter().enumerate() {
            let handle =
                dependency_handle(dependencies, ResourceDependency::Image(destination.image))?;
            let Resource::Image { description, .. } = self.resource(handle)? else {
                return Err(kind_mismatch(handle));
            };
            image_region_is_full(*description, *destination)?;
            let (width, height, format) = if plane == 0 {
                (frame.width, frame.height, ImageFormat::R8Unorm)
            } else {
                (
                    frame.width.div_ceil(2),
                    frame.height.div_ceil(2),
                    ImageFormat::Rg8Unorm,
                )
            };
            if description.format() != format
                || description.samples() != SampleCount::One
                || description.dimension() != ImageDimension::Two
                || description.array_layers() != 1
                || description.mip_levels() != 1
                || destination.subresources.layer_count != 1
                || destination.subresources.plane != 0
                || destination.extent
                    != (ImageExtent {
                        width,
                        height,
                        depth: 1,
                    })
            {
                return Err(unsupported("device NV12 destination layout"));
            }
        }
        let pitch = u64::from(frame.width.div_ceil(2) * 2).next_multiple_of(256);
        let chroma_offset = pitch * u64::from(frame.height);
        let size = chroma_offset + pitch * u64::from(frame.height.div_ceil(2));
        let reusable = self.video_buffers.iter().position(|buffer| {
            buffer.size == size
                && buffer.pending.is_none_or(|token| {
                    self.submissions
                        .get(&token)
                        .is_none_or(|submission| submission.completed)
                })
        });
        let index = if let Some(index) = reusable {
            index
        } else {
            self.video_buffers
                .push(crate::native::vulkan::video::VideoBuffer::new(
                    &self.device,
                    size,
                    frame,
                    pitch,
                    chroma_offset,
                )?);
            self.video_buffers.len() - 1
        };
        let buffer = &mut self.video_buffers[index];
        if reusable.is_some() {
            frame
                .source
                .copy_nv12(nixe_gpu::ExternalImageBuffer {
                    fd: buffer.fd(),
                    semaphore_fd: buffer.semaphore_fd(),
                    allocation_size: buffer.allocation_size,
                    device_uuid: uuid,
                    row_pitch: pitch,
                    chroma_offset,
                    initialize: false,
                })
                .map_err(BackendDriverError::failure)?;
        }
        buffer.wait_cuda(&self.device, &self.queue_access)?;
        buffer.pending = Some(token);
        let source = buffer.buffer.clone();
        for (plane, destination) in destinations.into_iter().enumerate() {
            let handle =
                dependency_handle(dependencies, ResourceDependency::Image(destination.image))?;
            let Resource::Image { texture, .. } = self.resource(handle)? else {
                return Err(kind_mismatch(handle));
            };
            let (width, height) = (destination.extent.width, destination.extent.height);
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &source,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: if plane == 0 { 0 } else { chroma_offset },
                        bytes_per_row: Some(pitch as u32),
                        rows_per_image: None,
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: u32::from(destination.subresources.mip_level),
                    origin: wgpu::Origin3d {
                        x: destination.origin.x,
                        y: destination.origin.y,
                        z: u32::from(destination.subresources.base_layer),
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }
}
