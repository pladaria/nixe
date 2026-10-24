//! Focused pipeline construction/GPU comparison, outside production paths.
use super::*;
use std::time::Instant;
fn set_dynamic_defaults(
    device: &mut VulkanDevice,
    pipeline: &NativePipeline,
    command: vk::CommandBuffer,
) {
    unsafe {
        let raw = &mut device.dispatch;
        raw.cmd_set_viewport(
            command,
            0,
            &[vk::Viewport {
                x: 0.,
                y: 0.,
                width: 32.,
                height: 32.,
                min_depth: 0.,
                max_depth: 1.,
            }],
        );
        raw.cmd_set_scissor(
            command,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 32,
                    height: 32,
                },
            }],
        );
        raw.cmd_set_line_width(command, 1.);
        raw.cmd_set_depth_bias(command, 0., 0., 0.);
        raw.cmd_set_blend_constants(command, &[0.; 4]);
        raw.cmd_set_stencil_compare_mask(command, vk::StencilFaceFlags::FRONT_AND_BACK, u32::MAX);
        raw.cmd_set_stencil_write_mask(command, vk::StencilFaceFlags::FRONT_AND_BACK, u32::MAX);
        raw.cmd_set_stencil_reference(command, vk::StencilFaceFlags::FRONT_AND_BACK, 0);
        if pipeline
            .dynamic_states
            .contains(&vk::DynamicState::DEPTH_BOUNDS)
        {
            raw.cmd_set_depth_bounds(command, 0., 1.);
        }
        if pipeline
            .dynamic_states
            .contains(&vk::DynamicState::CULL_MODE)
        {
            raw.dynamic_state(command, &GraphicsState::color(vk::Format::R8G8B8A8_UNORM));
        }
    }
}
fn raster_sample(
    device: &mut VulkanDevice,
    pipeline: &Arc<NativePipeline>,
    draws: u32,
) -> (f64, f64, Vec<u8>) {
    let owner = Arc::clone(&device.owner);
    let image = Arc::new(
        device
            .allocate_image_2d(
                32,
                32,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                AllocationMode::Managed,
                "pipeline cache raster check",
            )
            .unwrap(),
    );
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let view = unsafe {
        owner.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image.handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(range),
            None,
        )
    }
    .unwrap();
    let view = crate::AllocatedImageView::from_handle(image, view);
    let buffer = device
        .allocate_buffer(
            32 * 32 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            AllocationMode::Managed,
            "pipeline cache raster bytes",
        )
        .unwrap();
    let query = unsafe {
        owner.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .unwrap();
    let slot = device.submission.begin(&mut device.dispatch).unwrap();
    let command = device.submission.slots[slot].command;
    unsafe {
        device.dispatch.cmd_reset_query_pool(command, query, 0, 2);
        let barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(view.image.handle)
            .subresource_range(range)];
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().image_memory_barriers(&barrier),
        );
        device.dispatch.cmd_write_timestamp(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            query,
            0,
        );
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view.handle)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 0., 1.],
                },
            })];
        device.dispatch.begin_rendering(
            command,
            &vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: Default::default(),
                    extent: vk::Extent2D {
                        width: 32,
                        height: 32,
                    },
                })
                .layer_count(1)
                .color_attachments(&attachments),
        );
    }
    device.bind_native_pipeline(pipeline).unwrap();
    set_dynamic_defaults(device, pipeline, command);
    let start = Instant::now();
    for _ in 0..draws {
        device.bind_native_pipeline(pipeline).unwrap();
        unsafe {
            device.dispatch.cmd_draw(command, 3, 1, 0, 0);
        }
    }
    let cpu = start.elapsed().as_nanos() as f64 / draws as f64;
    unsafe {
        device.dispatch.end_rendering(command);
        device.dispatch.cmd_write_timestamp(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            query,
            1,
        );
        let barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(view.image.handle)
            .subresource_range(range)];
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().image_memory_barriers(&barrier),
        );
        let copies = [vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: 32,
                height: 32,
                depth: 1,
            })];
        device.dispatch.cmd_copy_image_to_buffer(
            command,
            view.image.handle,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.handle,
            &copies,
        );
        let barrier = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.handle)
            .size(buffer.size())];
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&barrier),
        );
    }
    let work = device.flush().unwrap().unwrap();
    assert!(device.wait_for_work(work, Duration::from_secs(10)).unwrap());
    let mut ticks = [0u64; 2];
    unsafe {
        owner.device.get_query_pool_results(
            query,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .unwrap();
    let properties = unsafe {
        owner
            .instance
            .instance
            .get_physical_device_queue_family_properties(owner.physical)
    };
    let bits = properties[device.queues().graphics_compute_family as usize].timestamp_valid_bits;
    assert!(bits != 0);
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let gpu = (ticks[1].wrapping_sub(ticks[0]) & mask) as f64
        * device.profile().limits.timestamp_period as f64
        / draws as f64;
    let pixels = unsafe { buffer.read_range(0, buffer.size()) }.unwrap();
    unsafe {
        owner.device.destroy_query_pool(query, None);
    }
    (cpu, gpu, pixels)
}
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
#[test]
#[ignore = "focused real-GPU comparison of pipeline construction strategies"]
fn compare_monolithic_and_library_pipeline_creation_and_warm_execution() {
    let Some(mut device) = support::device_for_test_with_validation(
        ApiPreference::Auto,
        if std::env::var("NIXE_VULKAN_PIPELINE_PROBE_NO_VALIDATION").as_deref() == Ok("1") {
            ValidationMode::Disabled
        } else {
            ValidationMode::Enabled
        },
    ) else {
        return;
    };
    let validation = device.validation_monitor();
    assert!(
        device.profile().limits.timestamp_compute_and_graphics != 0,
        "GPU timestamps required by this focused comparison"
    );
    let config = CacheConfiguration {
        workers: 2,
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(2, 32).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let mut artifacts = Vec::new();
    let mut translation_ns = 0;
    let mut spirv_ns = 0;
    for seed in 0..8u32 {
        let shader = ready(
            shaders
                .request_guest(
                    captured_graphics(seed, &device.profile().shader_compiler),
                    Arc::new(device.profile().shader_compiler.clone()),
                    Some(Default::default()),
                )
                .unwrap(),
        );
        translation_ns += shader.timings.translation_ns;
        spirv_ns += shader.timings.spirv_ns;
        artifacts.push(shader);
    }
    eprintln!(
        "PIPELINE_PROBE gpu={:?} fast_link={} translation_total_ns={translation_ns} spirv_total_ns={spirv_ns} shader_cache={:?}",
        device.profile().name,
        device.profile().graphics_pipeline_library_fast_linking,
        shaders.usage()
    );
    let strategies = if device.profile().graphics_pipeline_library {
        vec![PipelineStrategy::Monolithic, PipelineStrategy::Libraries]
    } else {
        vec![PipelineStrategy::Monolithic]
    };
    let only = std::env::var("NIXE_VULKAN_PIPELINE_PROBE_STRATEGY").ok();
    let strategies: Vec<_> = strategies
        .into_iter()
        .filter(|s| {
            only.as_deref().is_none_or(|value| match value {
                "mono" => *s == PipelineStrategy::Monolithic,
                "gpl" => *s == PipelineStrategy::Libraries,
                _ => panic!("probe strategy must be mono or gpl"),
            })
        })
        .collect();
    let mut golden = Vec::new();
    for strategy in strategies {
        let before = rss_kib();
        let pipelines =
            PipelineCompiler::with_strategy(&device, Arc::clone(&pool), &config, strategy).unwrap();
        let mut cold = Vec::new();
        let mut feedback = Vec::new();
        let mut linking = Vec::new();
        let mut libs_wall = 0;
        let mut libs_driver = 0;
        let mut cache_hits = 0;
        let mut draws = Vec::new();
        for (index, shader) in artifacts.iter().enumerate() {
            for mask in [
                vk::ColorComponentFlags::RGBA,
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B,
            ] {
                let mut state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
                std::sync::Arc::make_mut(&mut state.colors)[0].1.write_mask = mask;
                let pipeline = pipelines
                    .request(PipelineRecipe::Graphics {
                        shader: Arc::clone(shader),
                        state,
                    })
                    .unwrap();
                cold.push(pipeline.metrics().unwrap().wall_ns as f64);
                if let Some(ns) = pipeline.metrics().unwrap().driver_ns {
                    feedback.push(ns as f64);
                }
                linking.push(pipeline.metrics().unwrap().link_ns as f64);
                libs_wall += pipeline.metrics().unwrap().library_wall_ns;
                libs_driver += pipeline.metrics().unwrap().library_driver_ns.unwrap_or(0);
                cache_hits += u32::from(pipeline.metrics().unwrap().application_cache_hit);
                if mask == vk::ColorComponentFlags::RGBA {
                    draws.push((index, pipeline));
                }
            }
        }
        let rss_delta = rss_kib().saturating_sub(before);
        let mut cpu = Vec::new();
        let mut gpu = Vec::new();
        for (index, pipeline) in draws {
            for trial in 0..7 {
                let (c, g, pixels) = raster_sample(&mut device, &pipeline, 512);
                cpu.push(c);
                gpu.push(g);
                // UNORM framebuffer quantization permits a one-unit rounding
                // difference from the CPU expression. Both strategies must
                // nevertheless produce byte-identical images on this device.
                let expected = (
                    ((index + 1) as f32 / 16. * 255.).round() as u8,
                    64u8,
                    128u8,
                    255u8,
                );
                for pixel in pixels.chunks_exact(4) {
                    assert!(pixel[0].abs_diff(expected.0) <= 1);
                    assert!(pixel[1].abs_diff(expected.1) <= 1);
                    assert!(pixel[2].abs_diff(expected.2) <= 1);
                    assert_eq!(pixel[3], expected.3);
                }
                if trial == 0 {
                    eprintln!(
                        "PIPELINE_IMAGE strategy={strategy:?} seed={index} digest={}",
                        ContentId::of(b"probe-image", &[&pixels]).hex()
                    );
                    if strategy == PipelineStrategy::Monolithic {
                        golden.push(pixels);
                    } else if !golden.is_empty() {
                        assert_eq!(pixels, golden[index]);
                    }
                }
            }
        }
        eprintln!(
            "PIPELINE_PROBE strategy={strategy:?} pipelines={} libraries={} cold_total_ns={} cold_median_ns={} feedback_valid={} feedback_median_ns={} application_hits={cache_hits} link_median_ns={} library_wall_total_ns={libs_wall} library_feedback_total_ns={libs_driver} warm_cpu_ns={} warm_gpu_ns={} rss_delta_kib={rss_delta}",
            cold.len(),
            pipelines.library_entries(),
            cold.iter().sum::<f64>(),
            median(&mut cold),
            feedback.len(),
            if feedback.is_empty() {
                0.
            } else {
                median(&mut feedback)
            },
            median(&mut linking),
            median(&mut cpu),
            median(&mut gpu)
        );
        drop(pipelines);
    }
    drop(artifacts);
    drop(shaders);
    drop(pool);
    drop(device);
    validation.check().unwrap();
}

pub(super) fn captured_graphics(
    seed: u32,
    config: &nixe_gpu::shader_compilation::Configuration,
) -> Arc<nixe_gpu::shader_compilation::ShaderInputs> {
    use nixe_gpu::shader_compilation::{ShaderInputs, StageSource};
    use nixe_shader_recompiler::{
        Compiler, Environment, Stage,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        maxwell::{decode::Opcode, header::ProgramHeader},
        types::ProgramOffset,
    };
    let word = |op: Opcode| op.encoding().value | 7 << 16;
    let mov = |register: u64, value: f32| {
        word(Opcode::MOV32I) | 15 << 12 | (value.to_bits() as u64) << 20 | register
    };
    let mut vertex = vec![
        word(Opcode::ALD) | 255 << 39 | 0x2fc << 20 | 255 << 8 | 4,
        mov(5, 3.),
        mov(6, -1.),
    ];
    for (component, index) in [(0, 1), (1, 2)] {
        vertex.push(word(Opcode::ISETP_imm) | 2 << 49 | 7 << 39 | index << 20 | 4 << 8 | 7);
        vertex.push(word(Opcode::SEL_reg) | 6 << 20 | 5 << 8 | component);
    }
    vertex.extend([
        mov(2, 0.),
        mov(3, 1.),
        word(Opcode::AST) | 3 << 47 | 255 << 39 | 0x70 << 20 | 255 << 8,
        word(Opcode::EXIT) | 15,
    ]);
    let fragment = vec![
        mov(0, (seed + 1) as f32 / 16.),
        mov(1, 0.25),
        mov(2, 0.5),
        mov(3, 1.),
        word(Opcode::EXIT) | 15,
    ];
    let mut compiler = Compiler::new().unwrap();
    let sources = [(Stage::VertexB, vertex), (Stage::Fragment, fragment)]
        .into_iter()
        .map(|(stage, code)| {
            let mut bytes = [0; 80];
            if stage == Stage::Fragment {
                bytes[72] = 15;
            }
            let mut pc = 0x50;
            let instructions = code
                .into_iter()
                .map(|word| {
                    if pc % 32 == 0 {
                        pc += 8;
                    }
                    let pair = (ProgramOffset(pc), word);
                    pc += 8;
                    pair
                })
                .collect();
            let mut environment = SnapshotEnvironment::new(
                EnvironmentMetadata {
                    stage,
                    sph: ProgramHeader::from_bytes(&bytes),
                    ..Default::default()
                },
                Observations {
                    instructions,
                    viewport_transform_state: Some(1),
                    ..Default::default()
                },
            );
            let id = compiler
                .translate(&mut environment, &config.host, &config.settings)
                .unwrap();
            assert_eq!(
                compiler.program(id).unwrap().stage,
                environment.metadata().stage
            );
            StageSource {
                environment,
                info: compiler.program(id).unwrap().info.clone(),
                binding_group: None,
            }
        })
        .collect::<Vec<_>>();
    Arc::new(ShaderInputs::new(sources.into(), config))
}

#[test]
fn captured_probe_modules_validate_without_a_gpu() {
    let config = nixe_gpu::shader_compilation::Configuration {
        profile: nixe_shader_recompiler::Profile {
            supported_spirv: 0x10500,
            unified_descriptor_binding: true,
            max_user_clip_distances: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut compiler = nixe_shader_recompiler::Compiler::new().unwrap();
    let artifacts = captured_graphics(0, &config)
        .compile(&mut compiler, &config, Some(&Default::default()))
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    for artifact in artifacts {
        let path = directory.path().join(format!("{:?}.spv", artifact.stage));
        let bytes: Vec<_> = artifact
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
        let output = std::process::Command::new(
            std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into()),
        )
        .args(["--target-env", "vulkan1.2"])
        .arg(path)
        .output()
        .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
