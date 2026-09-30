//! Exact, sampler-free loads through the production neutral-to-WGSL lowering.
use super::*;
use nixe_gpu::{
    ShaderResourceAccess, ShaderResourceKind, ShaderRoundingMode, ShaderTextureSampleOutput,
};

#[test]
fn rgba16_texel_fetch_preserves_coordinates_mip_and_hdr_without_sampler() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(740),
        NonCpuDeviceId::new(740),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let presentation = initialized.presentation_context();
    let device = presentation.device();
    let queue = presentation.queue();
    let mut instructions = vec![load_input(8, 0, 0, ShaderIoLocation::Position, 2)];
    for index in 0..2 {
        instructions.push(ShaderInstruction::new(
            ShaderSourceLocation::new(16 + index * 8),
            ShaderPredicate::Always,
            ShaderOperation::ConvertFloat32ToInteger {
                destination: ShaderRegister::new(index as u16),
                source: ShaderRegister::new(index as u16),
                destination_type: ShaderScalarType::Signed32,
                destination_bits: 32,
                rounding: ShaderRoundingMode::TowardZero,
                flush_denormals_to_zero: false,
            },
        ));
    }
    instructions.extend([
        ShaderInstruction::new(
            ShaderSourceLocation::new(32),
            ShaderPredicate::Always,
            ShaderOperation::LoadTexture2D {
                outputs: (0..4)
                    .map(|component| {
                        ShaderTextureSampleOutput::new(
                            ShaderRegister::new(u16::from(component)),
                            component,
                        )
                        .unwrap()
                    })
                    .collect(),
                coordinates: [ShaderRegister::new(0), ShaderRegister::new(1)],
                image_binding: 0,
                mip_level: 1,
            },
        ),
        store_output(40, 0, ShaderIoLocation::Color(0), 4),
        exit(48),
    ]);
    let interface = |location, count| {
        (0..count)
            .map(|component| {
                ShaderInterfaceElement::new(location, component, ShaderScalarType::Float32, None)
                    .unwrap()
            })
            .collect()
    };
    let shader = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Fragment,
        interface(ShaderIoLocation::Position, 2),
        interface(ShaderIoLocation::Color(0), 4),
        vec![ShaderResourceAccess::new(0, ShaderResourceKind::SampledImage, true, false).unwrap()],
        instructions,
    ))
    .unwrap();
    let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(&shader).unwrap();
    assert!(wgsl.source().contains("textureLoad("));
    assert!(!wgsl.source().contains("var sampler_"));
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("texel fetch"),
        source: wgpu::ShaderSource::Wgsl(wgsl.source().into()),
    });
    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl("@vertex fn main(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> { let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)); return vec4(p[id], 0.0, 1.0); }".into()) });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 2,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let extent = wgpu::Extent3d {
        width: 2,
        height: 2,
        depth_or_array_layers: 1,
    };
    let halves: [u16; 16] = [
        0x4000, 0xbc00, 0x3800, 0x3c00, 0, 0x3c00, 0, 0x3800, 0x3c00, 0, 0x4000, 0, 0x3800, 0x4000,
        0xbc00, 0x3c00,
    ];
    let expected = [
        2.0_f32, -1.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 2.0, 0.0, 0.5, 2.0, -1.0, 1.0,
    ];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            mip_level: 1,
            ..texture.as_image_copy()
        },
        &halves
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: None,
        },
        extent,
    );
    let input_view = texture.create_view(&Default::default());
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&input_view),
        }],
    });
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output.create_view(&Default::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 512,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
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
        extent,
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap()
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = readback.get_mapped_range(..).unwrap();
    for (row, expected) in bytes.chunks_exact(256).zip(expected.chunks_exact(8)) {
        for (actual, expected) in row[..32].chunks_exact(4).zip(expected) {
            assert_eq!(
                u32::from_le_bytes(actual.try_into().unwrap()),
                expected.to_bits()
            );
        }
    }
}
