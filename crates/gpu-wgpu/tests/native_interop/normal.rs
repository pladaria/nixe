use super::{SIZE, device::Context};
use wgpu::util::DeviceExt;

pub struct Resources {
    pub color: wgpu::Texture,
    pub depth: wgpu::Texture,
    pub image: wgpu::Texture,
    pub buffer: wgpu::Buffer,
    pub color_views: [wgpu::TextureView; 2],
    pub depth_views: [wgpu::TextureView; 2],
    pub image_view: wgpu::TextureView,
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    size: u32,
    layers: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

impl Resources {
    pub fn new(ctx: &Context) -> Self {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let color = texture(
            &ctx.device,
            "shared color, two independent layers",
            wgpu::TextureFormat::Rgba8Unorm,
            SIZE,
            2,
            usage,
        );
        let depth = texture(
            &ctx.device,
            "shared depth, two independent layers",
            wgpu::TextureFormat::Depth32Float,
            SIZE,
            2,
            usage,
        );
        let views = |t: &wgpu::Texture| {
            std::array::from_fn(|layer| {
                t.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
        };
        let color_views = views(&color);
        let depth_views = views(&depth);
        let image = texture(
            &ctx.device,
            "TES sampled image / compute destination",
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        let image_view = image.create_view(&Default::default());
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TCS parameters / compute destination"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            color,
            depth,
            image,
            buffer,
            color_views,
            depth_views,
            image_view,
        }
    }

    pub fn initialize(&self, encoder: &mut wgpu::CommandEncoder) {
        // A once-per-allocation, semantically permitted clear establishes wgpu's
        // initialization state. transition_resources alone DOES NOT initialize.
        // No recurring hidden clear or native-first read of undefined host memory.
        for layer in 0..2 {
            let color = if layer == 0 {
                wgpu::Color {
                    r: 1.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                }
            } else {
                wgpu::Color::BLACK
            };
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("allocation initialization"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_views[layer],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_views[layer],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if layer == 0 { 0.75 } else { 1.0 }),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
    }

    pub fn upload(&self, ctx: &Context, encoder: &mut wgpu::CommandEncoder) {
        let floats: Vec<_> = [1.0f32; 4].into_iter().flat_map(f32::to_le_bytes).collect();
        let upload = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("upload fixture"),
                contents: &floats,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        encoder.copy_buffer_to_buffer(&upload, 0, &self.buffer, 0, 16);
        let image = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image upload fixture"),
                contents: &[0, 255, 0, 255],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &image,
                layout: wgpu::TexelCopyBufferLayout::default(),
            },
            self.image.as_image_copy(),
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn handoff(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.transition_resources(
            [wgpu::BufferTransition {
                buffer: &self.buffer,
                state: wgpu::BufferUses::STORAGE_READ_ONLY,
            }]
            .into_iter(),
            [
                wgpu::TextureTransition {
                    texture: &self.color,
                    selector: Some(wgpu_types::TextureSelector {
                        mips: 0..1,
                        layers: 1..2,
                    }),
                    state: wgpu::TextureUses::COLOR_TARGET,
                },
                wgpu::TextureTransition {
                    texture: &self.depth,
                    selector: Some(wgpu_types::TextureSelector {
                        mips: 0..1,
                        layers: 1..2,
                    }),
                    state: wgpu::TextureUses::DEPTH_STENCIL_WRITE,
                },
                wgpu::TextureTransition {
                    texture: &self.image,
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                },
            ]
            .into_iter(),
        );
    }
}

pub struct NormalPipelines {
    before: wgpu::RenderPipeline,
    behind: wgpu::RenderPipeline,
    front: wgpu::RenderPipeline,
    sample: wgpu::RenderPipeline,
    compute: wgpu::ComputePipeline,
}

impl NormalPipelines {
    pub fn new(ctx: &Context) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("normal interop fixture"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normal.wgsl").into()),
            });
        let pipeline = |vertex, fragment, depth: bool| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(vertex),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vertex),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: depth.then_some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fragment),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let producer = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("compute producer"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/producer.wgsl").into()),
            });
        let compute = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute producer"),
                layout: None,
                module: &producer,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            before: pipeline("before", "red", true),
            behind: pipeline("behind", "blue", true),
            front: pipeline("front", "blue", true),
            sample: pipeline("front", "sample_result", false),
            compute,
        }
    }

    pub fn produce(&self, ctx: &Context, r: &Resources, encoder: &mut wgpu::CommandEncoder) {
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute writes consumed by TCS and TES"),
            layout: &self.compute.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&r.image_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.compute);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn draw(&self, r: &Resources, encoder: &mut wgpu::CommandEncoder, before: bool) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("normal draw shares native color and depth"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &r.color_views[1],
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &r.depth_views[1],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        if before {
            pass.set_pipeline(&self.before);
            pass.set_scissor_rect(0, 0, SIZE / 4, SIZE);
            pass.draw(0..3, 0..1);
        } else {
            // Behind native depth: must not overwrite native output.
            pass.set_pipeline(&self.behind);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.front);
            pass.set_scissor_rect(SIZE * 3 / 4, 0, SIZE / 4, SIZE);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn sample(
        &self,
        ctx: &Context,
        r: &Resources,
        encoder: &mut wgpu::CommandEncoder,
        layer: usize,
    ) -> wgpu::Buffer {
        let output = texture(
            &ctx.device,
            "sampling oracle",
            wgpu::TextureFormat::Rgba8Unorm,
            SIZE,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = output.create_view(&Default::default());
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("read shared native color and depth"),
            layout: &self.sample.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&r.color_views[layer]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&r.depth_views[layer]),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.sample);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test oracle readback"),
            size: u64::from(256 * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            output.size(),
        );
        readback
    }
}

pub fn check(ctx: &Context, buffer: wgpu::Buffer, expected: impl Fn(u32) -> [u8; 4]) {
    let bytes = read_pixels(ctx, buffer);
    for y in 0..SIZE {
        for x in 0..SIZE {
            assert_eq!(
                bytes[(y * SIZE + x) as usize],
                expected(x),
                "pixel ({x}, {y})"
            );
        }
    }
}

pub fn read_pixels(ctx: &Context, buffer: wgpu::Buffer) -> Vec<[u8; 4]> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = buffer.get_mapped_range(..).unwrap();
    let mut pixels = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let start = (y * 256 + x * 4) as usize;
            pixels.push(bytes[start..start + 4].try_into().unwrap());
        }
    }
    drop(bytes);
    buffer.unmap();
    pixels
}
