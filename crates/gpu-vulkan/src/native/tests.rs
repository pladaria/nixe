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
        identity: PreparedStateId::new(1, 1),
        program: Arc::new(GuestProgram {
            identity: PreparedStateId::new(1, 1),
            shader: Arc::new(GuestShaderCode {
                identity: ssa::artifact::ContentId::of(b"native-draw-test", &[]),
                functions: crate::residency::execution::tests::vertex_fragment(),
            }),
            bindings: Box::new([]),
            shared_memory_bytes: 0,
            local_memory_bytes: 0,
        }),
        vertices: Box::new([GuestVertexBuffer {
            binding: 0,
            backing: vertex_backing,
        }]),
        index: None,
        colors: Box::new([image]),
        depth_stencil: None,
        parameters: Arc::new(GuestDrawParameters {
            identity: PreparedStateId::new(2, 1),
            vertices: Box::new([GuestVertexLayout {
                binding: 0,
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
            color_formats: Box::new([ImageFormat::Rgba8Unorm]),
            depth_stencil_format: None,
            samples: SampleCount::One,
            extent: [8, 8],
            render_area: ScissorRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
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
        }),
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
    assert!(execution.residency.statistics().cpu_streamed_bytes > 0);
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
    changed.identity = PreparedStateId::new(2, 1);
    Arc::make_mut(&mut changed.parameters).identity = PreparedStateId::new(3, 1);
    Arc::make_mut(&mut changed.parameters).scissor.x = 4;
    Arc::make_mut(&mut changed.parameters).scissor.width = 4;
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
    // Different bindings and scissors across frontend deliveries share native
    // rendering, but preserve draw order and their own prepared semantics.
    let (other_vertices, other_backing) = backing_allocation(256, 256, 0);
    other_vertices
        .write(
            0,
            &red.into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let other_backing = other_backing.snapshot_subrange(0, 72).unwrap();
    let mut other = (*state).clone();
    other.identity = PreparedStateId::new(10, 1);
    other.vertices[0].backing = other_backing;
    let other = Arc::new(other);
    let mut clipped = (*other).clone();
    clipped.identity = PreparedStateId::new(11, 1);
    let parameters = Arc::make_mut(&mut clipped.parameters);
    parameters.identity = PreparedStateId::new(12, 1);
    parameters.scissor.width = 4;
    let clipped = Arc::new(clipped);
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut sent = Vec::new();
    for (index, current) in [&other, &state, &clipped].into_iter().enumerate() {
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(10 + index as u64),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: false,
            commands: vec![NativeCommand::Draw {
                state: Arc::clone(current),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            }],
        };
        ready(&mut execution, &submission);
        sent.push(
            execution
                .submit(ExecutionBatch {
                    commands: &submission,
                    definitions: &[],
                    retirements: &[],
                })
                .unwrap(),
        );
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    assert!(execution.statistics.reused_graphics > 0);
    assert!(execution.statistics.reused_programs > 0);
    assert!(execution.residency.statistics().cpu_stream_hits > 0);
    assert!(execution.residency.statistics().attachment_publication_hits >= 2);
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                &[255, 0, 0, 255]
            } else {
                &[0, 255, 0, 255]
            }
        );
    }
    for token in sent {
        assert_eq!(
            execution
                .wait_for_completion_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .submission(),
            token
        );
    }
    // CPU updates between accepted draws must preserve distinct read snapshots,
    // including shader descriptors, without ending rendering or submitting early.
    let (uniform, uniform_backing) = backing_allocation(256, 256, 0);
    let uniform_backing = uniform_backing.snapshot_subrange(0, 16).unwrap();
    let encode = |rgba: [f32; 4]| {
        rgba.into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut uniform_draw = (*state).clone();
    uniform_draw.identity = PreparedStateId::new(20, 1);
    uniform_draw.program = Arc::new(GuestProgram {
        identity: PreparedStateId::new(21, 1),
        shader: Arc::new(GuestShaderCode {
            identity: ssa::artifact::ContentId::of(b"streamed-uniform-test", &[]),
            functions: uniform_color_shader(),
        }),
        bindings: Box::new([(
            ssa::ResourceId::new(42),
            GuestBinding::Buffer(uniform_backing.clone()),
        )]),
        shared_memory_bytes: 0,
        local_memory_bytes: 0,
    });
    Arc::make_mut(&mut uniform_draw.parameters).identity = PreparedStateId::new(22, 1);
    let uniform_draw = Arc::new(uniform_draw);
    let mut inputs = Vec::new();
    for (index, rgba) in [[0., 1., 0., 1.], [1., 0., 0., 1.], [0., 0., 1., 1.]]
        .into_iter()
        .enumerate()
    {
        let mut draw = (*uniform_draw).clone();
        draw.identity = PreparedStateId::new(23 + index as u64, 1);
        let parameters = Arc::make_mut(&mut draw.parameters);
        parameters.identity = PreparedStateId::new(26 + index as u64, 1);
        if index > 0 {
            parameters.scissor.x = if index == 1 { 4 } else { 0 };
            parameters.scissor.width = 4;
        }
        let submission = NativeSubmission {
            frontend: FrontendSubmissionId::new(20 + index as u64),
            segment: FrontendSubmissionSegment::FIRST,
            final_segment: false,
            commands: vec![NativeCommand::Draw {
                state: Arc::new(draw),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            }],
        };
        ready(&mut execution, &submission);
        inputs.push((submission, encode(rgba)));
    }
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut sent = Vec::new();
    for (submission, bytes) in &inputs {
        uniform.write(0, bytes).unwrap();
        sent.push(
            execution
                .submit(ExecutionBatch {
                    commands: submission,
                    definitions: &[],
                    retirements: &[],
                })
                .unwrap(),
        );
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % 8 < 4 {
                &[0, 0, 255, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
    }
    for token in sent {
        assert_eq!(
            execution
                .wait_for_completion_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .submission(),
            token
        );
    }
    // Streaming has not acknowledged the persistent buffer's dirty observer.
    let native_uniform = execution
        .residency
        .buffer_for(uniform_backing, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap();
    assert_eq!(
        execution
            .residency
            .read_buffer(native_uniform, 0, 16)
            .unwrap(),
        encode([0., 0., 1., 1.])
    );
    // A following GPU writer replaces CPU authority and must bypass CPU streams.
    let upload = execution
        .residency
        .write_buffer(native_uniform, 0, &encode([1., 1., 0., 1.]))
        .unwrap();
    assert_eq!(upload.consumed, 16);
    let submission = NativeSubmission {
        frontend: FrontendSubmissionId::new(30),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: true,
        commands: vec![NativeCommand::Draw {
            state: uniform_draw,
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        }],
    };
    ready(&mut execution, &submission);
    execution
        .submit(ExecutionBatch {
            commands: &submission,
            definitions: &[],
            retirements: &[],
        })
        .unwrap();
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(
        bytes
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 255, 0, 255])
    );
    execution
        .wait_for_completion_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    render_area_preserves_undefined_padding(&mut execution, &state);
    drop(pixels);
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}

fn render_area_preserves_undefined_padding(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
) {
    let (_, backing) = backing_allocation(256, 256, 0x73);
    let mut draw = (**state).clone();
    draw.identity = PreparedStateId::new(100, 1);
    draw.colors[0].planes = Arc::from([GuestImagePlane {
        backing: backing.clone(),
        ..draw.colors[0].planes[0].clone()
    }]);
    draw.colors[0].canonical_texels = false;
    let parameters = Arc::make_mut(&mut draw.parameters);
    parameters.identity = PreparedStateId::new(101, 1);
    parameters.render_area = ScissorRect {
        x: 1,
        y: 1,
        width: 5,
        height: 4,
    };
    parameters.scissor = parameters.render_area;
    parameters.scissor.width = 2;
    let image = draw.colors[0].clone();
    let area = parameters.render_area;
    let arguments = DrawArguments::NonIndexed {
        first_vertex: 0,
        vertex_count: 3,
        first_instance: 0,
        instance_count: 1,
    };
    let first = NativeSubmission {
        frontend: FrontendSubmissionId::new(100),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: vec![
            NativeCommand::Clear {
                image: image.clone(),
                rectangle: area,
                value: ClearValue::Color([0., 0., 1., 1.]),
            },
            NativeCommand::Draw {
                state: Arc::new(draw.clone()),
                arguments,
            },
        ],
    };
    let parameters = Arc::make_mut(&mut draw.parameters);
    parameters.identity = PreparedStateId::new(102, 1);
    parameters.scissor.x = 3;
    draw.identity = PreparedStateId::new(103, 1);
    let second = NativeSubmission {
        frontend: FrontendSubmissionId::new(101),
        segment: FrontendSubmissionSegment::FIRST,
        final_segment: false,
        commands: vec![NativeCommand::Draw {
            state: Arc::new(draw),
            arguments,
        }],
    };
    let scopes = execution.recording_statistics().rendering_scopes;
    for submission in [&first, &second] {
        ready(execution, submission);
        execution
            .submit(ExecutionBatch {
                commands: submission,
                definitions: &[],
                retirements: &[],
            })
            .unwrap();
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    // Both dynamic scissors share one scope. Neither the attachment's
    // physical padding nor viewport coordinates change with the render area.
    let bytes = cpu_read(&mut execution.residency, &backing);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        let (x, y) = (index % 8, index / 8);
        if (1..6).contains(&x) && (1..5).contains(&y) {
            if x == 5 {
                assert_eq!(pixel, [0, 0, 255, 255]);
            } else {
                assert_eq!(pixel, [0, 255, 0, 255]);
            }
        } else {
            assert_eq!(pixel, [0x73; 4]);
        }
    }
}

fn uniform_color_shader() -> Box<[ssa::Function]> {
    use ssa::*;
    let vertex = crate::residency::execution::tests::vertex_fragment()[0].clone();
    let mut builder = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = builder.entry();
    let pc = ShaderSourceLocation::new(8);
    for component in 0..4 {
        let address = builder.emit(
            entry,
            pc,
            Op::Constant(Constant {
                ty: Type::U32,
                bits: u64::from(component * 4),
            }),
            vec![],
        )[0];
        let value = builder.emit(
            entry,
            pc,
            Op::Load {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::F32,
            },
            vec![address.into()],
        )[0];
        builder.emit(
            entry,
            pc,
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component,
                ty: Type::F32,
            },
            vec![value.into()],
        );
    }
    builder.terminate(entry, pc, Terminator::Return);
    let mut fragment = builder.finish().unwrap();
    fragment.outputs = (0..4)
        .map(|component| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                component,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap()
        })
        .collect();
    fragment.optimize().unwrap();
    Box::new([vertex, fragment])
}
