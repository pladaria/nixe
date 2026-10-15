use super::*;
use crate::residency::tests::{backing_allocation, cpu_read};
use nixe_gpu::*;
use std::time::Instant;
mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}
fn ready(execution: &mut NativeGpuExecution, submission: &NativeSubmission) {
    let started = Instant::now();
    while !execution
        .prepare(ExecutionBatch {
            commands: submission,
            definitions: &[],
            retirements: &[],
        })
        .unwrap()
    {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "native compilation must publish readiness"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn native_owner_renders_exact_pixels_reuses_prepared_draws_and_resets_command_state() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (pixels, backing) = backing_allocation(256, 256, 0);
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(8, 8, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let image = GuestImage {
        sample_grid: [1, 1],
        description,
        swizzle: Swizzle::IDENTITY,
        canonical_texels: true,
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            backing: backing.clone(),
            row_pitch: 32,
            bytes_per_block: 4,
        }]),
    };
    let (vertices, vertex_backing) = backing_allocation(256, 256, 0);
    let vertex_backing = vertex_backing.snapshot_subrange(0, 72).unwrap();
    let words: [f32; 18] = [
        -1., -1., 0., 0., 1., 0., 3., -1., 0., 0., 1., 0., -1., 3., 0., 0., 1., 0.,
    ];
    vertices
        .write(
            0,
            &words
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let state = Arc::new(GuestDrawState {
        program: GuestProgram {
            identity: PreparedStateId::new(1, 1),
            shader: Arc::new(GuestShaderCode {
                identity: ssa::artifact::ContentId::of(b"native-draw-test", &[]),
                functions: crate::residency::execution::tests::vertex_fragment(),
            }),
            bindings: Box::new([]),
            shared_memory_bytes: 0,
            local_memory_bytes: 0,
        },
        vertices: Box::new([GuestVertexBuffer {
            binding: 0,
            backing: vertex_backing,
            stride: 24,
            step: VertexStepMode::Vertex,
            attributes: Box::new([
                nixe_gpu::VertexAttribute {
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                    offset: 0,
                },
                nixe_gpu::VertexAttribute {
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                    offset: 12,
                },
            ]),
        }]),
        index: None,
        colors: Box::new([image]),
        depth_stencil: None,
        extent: [8, 8],
        topology: PrimitiveTopology::Triangles,
        restart: false,
        provoking_first: false,
        raster: TriangleRasterization::Fill,
        line: None,
        front_face: FrontFace::CounterClockwise,
        cull: CullMode::None,
        depth_bias: None,
        viewport: None,
        position_transform: None,
        scissor: ScissorRect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        },
        depth: DepthState::DISABLED,
        stencil: None,
        color_outputs: Box::new([ColorOutputState::REPLACE]),
        blend_constants: [0.; 4],
        alpha_test: None,
        tessellation: None,
        sample_mask: u64::MAX,
        alpha_to_coverage: false,
    });
    let draw = NativeCommand::Draw {
        state: Arc::clone(&state),
        arguments: DrawArguments::NonIndexed {
            first_vertex: 0,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        },
    };
    let submission = NativeSubmission {
        frontend: FrontendSubmissionId::new(1),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![draw; 16],
    };
    ready(&mut execution, &submission);
    assert_eq!(execution.statistics.prepared_draws, 1);
    for frontend in 1..=2 {
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(frontend),
            commands: submission.commands.clone(),
            ..submission
        };
        ready(&mut execution, &submission);
        execution
            .submit(ExecutionBatch {
                commands: &submission,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
        assert_eq!(
            execution
                .wait_for_completion_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .frontend(),
            submission.frontend
        );
        let stats = execution.recording_statistics();
        assert_eq!(stats.pipeline_binds, frontend);
        assert_eq!(stats.vertex_binds, frontend);
        assert_eq!(stats.graphics_updates, frontend);
        assert_eq!(stats.rendering_scopes, frontend);
    }
    assert_eq!(execution.statistics.prepared_draws, 1);
    assert_eq!(execution.statistics.draws, 32);
    assert_eq!(execution.statistics.warm_draws, 31);
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|p| p == [0, 255, 0, 255]));
    assert_eq!(execution.residency.statistics().cpu_uploaded_bytes, 328);
    // Acceptance of a read snapshot need not submit a command buffer. A
    // following CPU write is uploaded after that read, with a native WAR edge.
    let prefix = NativeSubmission {
        frontend: FrontendSubmissionId::new(3),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: submission.commands.clone(),
    };
    ready(&mut execution, &prefix);
    let first = execution
        .submit(ExecutionBatch {
            commands: &prefix,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert!(execution.poll_completion().unwrap().is_none());
    match backing.pages().next().unwrap().visibility_state() {
        nixe_memory::VisibilityState::GpuNewer { device, .. } => {
            assert_eq!(first.instance().get(), device.get());
        }
        state => panic!("accepted write must carry its actual native owner: {state:?}"),
    }
    let mut red = words;
    for color in red.chunks_exact_mut(6) {
        color[3] = 1.;
        color[4] = 0.;
    }
    vertices
        .write(
            0,
            &red.into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut changed = (*state).clone();
    changed.program.identity = PreparedStateId::new(2, 1);
    changed.scissor.x = 4;
    changed.scissor.width = 4;
    let final_segment = NativeSubmission {
        frontend: prefix.frontend,
        segment: prefix.segment,
        final_segment: true,
        commands: vec![NativeCommand::Draw {
            state: Arc::new(changed),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        }],
    };
    ready(&mut execution, &final_segment);
    let second = execution
        .submit(ExecutionBatch {
            commands: &final_segment,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_ne!(first, second);
    let a = execution
        .wait_for_completion_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let b = execution
        .wait_for_completion_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(a.submission(), first);
    assert_eq!(b.submission(), second);
    assert_eq!(a.visibility(), b.visibility());
    assert_eq!(execution.recording_statistics().pipeline_binds, 3);
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            }
        );
    }
    // A CPU/frontend demand after accepting an unfinished native delivery
    // must find a pending producer, force its submission and observe its bytes.
    // Returning the preceding green/red image would silently read stale data.
    let pending = NativeSubmission {
        frontend: FrontendSubmissionId::new(4),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: submission.commands.clone(),
    };
    ready(&mut execution, &pending);
    let submitted = execution.residency.device().queue_usage().submitted_tick;
    let token = execution
        .submit(ExecutionBatch {
            commands: &pending,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_eq!(
        execution.residency.device().queue_usage().submitted_tick,
        submitted,
        "acceptance alone need not submit the native command buffer"
    );
    assert!(backing.pages().all(|page| matches!(
        page.visibility_state(),
        nixe_memory::VisibilityState::GpuNewer { .. }
    )));
    vertices
        .write(
            0,
            &words
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let repeated = execution
        .submit(ExecutionBatch {
            commands: &pending,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    assert_eq!(
        execution.residency.device().queue_usage().submitted_tick,
        submitted
    );
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    assert!(execution.residency.device().queue_usage().submitted_tick > submitted);
    assert_eq!(
        execution
            .wait_for_completion_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap()
            .submission(),
        token
    );
    assert_eq!(
        execution
            .wait_for_completion_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap()
            .submission(),
        repeated
    );
    drop(pixels);
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}
