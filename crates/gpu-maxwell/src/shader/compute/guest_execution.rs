//! Execute the original captured kernel after production Maxwell/WGSL lowering.
//! Readback and numeric checks are test-only; the guest path needs no CPU copy.
use super::tests::sinewave_kernel;
use wgpu::util::DeviceExt;

use super::super::hardware;

#[test]
#[ignore = "requires a physical Vulkan GPU"]
fn captured_compute_kernel_writes_positions_colors_and_preserves_sentinels() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Some(adapter) = hardware::adapter(&instance, wgpu::Backends::VULKAN) else {
        return;
    };
    eprintln!("captured compute kernel adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let program = sinewave_kernel();
    let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(program.module.ir()).unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("captured Maxwell compute kernel"),
        source: wgpu::ShaderSource::Wgsl(wgsl.source().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    // Vary the launch size without recompiling and force a low-word pointer
    // carry. Storage relocation must not depend on host/guest VA coincidence.
    for (groups, pointer, phase) in [
        (8, 0x04_ffff_fff0_u64, 0.125_f32),
        (16, 0x02_0000_1000, -0.25),
    ] {
        let vertices = groups * 32;
        let size = u64::from(vertices) * 32;
        assert_eq!(
            program
                .global_buffers
                .byte_extents([groups, 1, 1], [32, 1, 1]),
            [size]
        );
        let mut driver = vec![0_u8; 0x148];
        driver[12..16].copy_from_slice(&groups.to_le_bytes());
        driver[0x140..0x148].copy_from_slice(&pointer.to_le_bytes());
        let colors = [[1.0_f32, 0.25, 0.0, 0.5], [0.0, 0.5, 1.0, 1.0]];
        let params: Vec<u8> = colors
            .into_iter()
            .flatten()
            .chain([phase, 0.375])
            .flat_map(f32::to_le_bytes)
            .collect();
        let buffer = |label, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let driver = buffer("driver constants", &driver);
        let params = buffer("user constants", &params);
        let output = buffer(
            "generated vertices and guard",
            &vec![0xa5; size as usize + 16],
        );
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: driver.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size + 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size + 16);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.get_mapped_range(..).unwrap();
        assert_eq!(&bytes[size as usize..], &[0xa5; 16]);
        for id in 0..vertices {
            let x = id as f32 / (vertices - 1) as f32;
            let mut expected = [0.0; 8];
            expected[..4].copy_from_slice(&[
                x * 2.0 - 1.0,
                0.375 * ((phase + x) * std::f32::consts::TAU).sin(),
                0.5,
                1.0,
            ]);
            for component in 0..4 {
                expected[component + 4] =
                    colors[0][component] * (1.0 - x) + colors[1][component] * x;
            }
            for (component, expected) in expected.into_iter().enumerate() {
                let offset = id as usize * 32 + component * 4;
                let actual = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                assert!(
                    (actual - expected).abs() < 0.00002,
                    "groups={groups} vertex={id} component={component}: {actual} != {expected}"
                );
            }
        }
        drop(bytes);
        readback.unmap();
    }
}
