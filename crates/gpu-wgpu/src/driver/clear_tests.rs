//! Pixel-level checks of backend-only depth/stencil lowering.

use super::*;

#[test]
fn partial_depth_stencil_clears_preserve_other_aspects_and_outside_texels() {
    let backend = nixe_gpu::BackendInstanceId::new(800);
    let device_id = nixe_memory::NonCpuDeviceId::new(800);
    let Ok(initialized) = crate::initialize_backend(backend, device_id, Default::default()) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated clear test");
        return;
    };
    let context = initialized.presentation_context();
    let device = context.device();
    let mut driver = WgpuBackendDriver::new(
        backend,
        WgpuExecutionContext {
            device: device.clone(),
            queue: context.queue().clone(),
            queue_access: context.queue_access().clone(),
        },
        Arc::new(WgpuVisibilityCoordinator::new(device_id)),
        None,
        None,
        GpuCacheConfiguration::default(),
    );
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Depth/stencil clear assertion"),
        size: Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth24PlusStencil8,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let description = ImageDescription::new(
        ImageDimension::Two,
        nixe_gpu::ImageExtent::new(8, 8, 1).unwrap(),
        ImageFormat::Depth24UnormStencil8Uint,
        nixe_gpu::ImageKind::DepthStencil,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let view = texture.create_view(&Default::default());
    encoder.begin_render_pass(&RenderPassDescriptor {
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: StoreOp::Store,
            }),
            stencil_ops: Some(Operations {
                load: LoadOp::Clear(0),
                store: StoreOp::Store,
            }),
        }),
        ..Default::default()
    });
    for (start, size, value) in [
        (
            1,
            6,
            ClearValue::DepthStencil {
                depth: 0.25,
                stencil: 0x11,
            },
        ),
        (2, 4, ClearValue::Depth(0.75)),
        (3, 2, ClearValue::Stencil(0xa5)),
    ] {
        let clear = ClearOperation::image(
            ImageRegion {
                image: nixe_gpu::ImageId::new(800),
                subresources: ImageSubresourceRange {
                    plane: 0,
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: 1,
                },
                origin: ImageOrigin {
                    x: start,
                    y: start,
                    z: 0,
                },
                extent: nixe_gpu::ImageExtent::new(size, size, 1).unwrap(),
            },
            nixe_gpu::ImageKind::DepthStencil,
            description.format(),
            SampleCount::One,
            value,
        )
        .unwrap();
        driver
            .encode_partial_image_clear(&mut encoder, &texture, description, &clear)
            .unwrap();
    }

    // D24 is opaque in WebGPU. Sample its depth aspect instead of depending
    // on a vendor-specific packed representation; stencil has a byte copy.
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var source: texture_depth_2d;
            @group(0) @binding(1) var<storage, read_write> depths: array<f32>;
            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                depths[id.y * 8u + id.x] = textureLoad(source, vec2<i32>(id.xy), 0);
            }
        "#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let depths = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 256,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let depth_view = texture.create_view(&TextureViewDescriptor {
        aspect: TextureAspect::DepthOnly,
        ..Default::default()
    });
    let bindings = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&depth_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: depths.as_entire_binding(),
            },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 256 + 8 * 256,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&depths, 0, &readback, 0, 256);
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            aspect: TextureAspect::StencilOnly,
            ..texture.as_image_copy()
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 256,
                bytes_per_row: Some(256),
                rows_per_image: Some(8),
            },
        },
        Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
    );
    context.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    readback.map_async(MapMode::Read, .., move |result| {
        sender.send(result).unwrap()
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    driver.require_device().unwrap();
    let bytes = readback.get_mapped_range(..).unwrap();
    for y in 0..8 {
        for x in 0..8 {
            let inside = |start, end| (start..end).contains(&x) && (start..end).contains(&y);
            let expected_depth = if inside(2, 6) {
                0.75
            } else if inside(1, 7) {
                0.25
            } else {
                1.0
            };
            let offset = (y * 8 + x) * 4;
            let depth = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert!(
                (depth - expected_depth).abs() < 0.000001,
                "depth at ({x}, {y}): {depth}"
            );
            let stencil = if inside(3, 5) {
                0xa5
            } else if inside(1, 7) {
                0x11
            } else {
                0
            };
            assert_eq!(bytes[256 + y * 256 + x], stencil, "stencil at ({x}, {y})");
        }
    }
}
