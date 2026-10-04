//! Exercise the production compute lowering, caches and demanded CPU coherence.
use super::*;
use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, DescriptorKind, DescriptorTableBinding,
    DescriptorTableDescription, DescriptorTableId, DispatchOperation, PipelineStages,
    ResourceAccess, ResourceUsage, ShaderBackendModule, ShaderComputeBuiltin, ShaderFloatControl,
    ShaderResourceAccess, ShaderResourceKind,
};

const RECORD_WORDS: usize = 15;
const WORDS: usize = 8 * 6 * 4 * RECORD_WORDS;

#[test]
fn compute_rejects_mismatched_stages_and_oversized_workgroups() {
    let _guard = accelerated_test_guard();
    for (stage, kind, size, expected) in [
        (
            ShaderStage::Vertex,
            PipelineKind::Compute,
            None,
            "dispatch requires a compute shader",
        ),
        (
            ShaderStage::Compute,
            PipelineKind::Graphics,
            Some([1, 1, 1]),
            "dispatch requires a compute pipeline",
        ),
        (
            ShaderStage::Compute,
            PipelineKind::Compute,
            Some([u32::MAX, 1, 1]),
            "compute workgroup size exceeds host limits",
        ),
    ] {
        let Some(initialized) = initialize_backend(
            BackendInstanceId::new(763),
            NonCpuDeviceId::new(763),
            WgpuBackendConfiguration {
                pipeline_cache_directory: None,
                ..Default::default()
            },
        ) else {
            return;
        };
        let runtime = RuntimeOwner::new(initialized.into_runtime());
        let ir = ShaderIr::new(stage, vec![], vec![], vec![], vec![exit(0)]);
        let ir = if let Some(size) = size {
            ir.with_workgroup_size(size)
        } else {
            ir
        };
        let creations = [
            BackendResourceCreateInfo::Pipeline {
                id: PipelineId::new(1),
                description: PipelineDescription { kind },
            },
            BackendResourceCreateInfo::Shader {
                id: ShaderId::new(1),
                description: ShaderDescription { stage },
                module: ShaderBackendModule::new(VerifiedShaderIr::verify(ir).unwrap()),
            },
        ];
        let submission = OperationSubmission::new(
            FrontendSubmissionId::new(1),
            vec![],
            vec![GpuOperation::new(
                GpuCommand::Dispatch(
                    DispatchOperation::new(PipelineId::new(1), ShaderId::new(1), vec![], [1, 1, 1])
                        .unwrap(),
                ),
                [],
                [],
                CapabilityRequirements::none(),
            )],
        )
        .unwrap();
        let error = runtime
            .runtime()
            .submit(&creations, &[], &submission)
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

fn kernel(increment: bool) -> ShaderBackendModule {
    let r = ShaderRegister::new;
    let mut ops = Vec::new();
    for (base, builtin, count) in [
        (0, ShaderComputeBuiltin::GlobalInvocationId, 3),
        (3, ShaderComputeBuiltin::LocalInvocationId, 3),
        (6, ShaderComputeBuiltin::WorkgroupId, 3),
        (9, ShaderComputeBuiltin::NumWorkgroups, 3),
        (12, ShaderComputeBuiltin::LocalInvocationIndex, 1),
    ] {
        for component in 0..count {
            ops.push(ShaderOperation::LoadComputeBuiltin32 {
                destination: r(base + u16::from(component)),
                builtin,
                component,
            });
        }
    }
    let immediate = |destination, bits| ShaderOperation::MoveImmediate32 {
        destination: r(destination),
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    };
    let mul = |destination, left, right| ShaderOperation::Multiply32 {
        destination: r(destination),
        left: r(left),
        right: r(right),
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    };
    let add = |destination, left, right| ShaderOperation::Add32 {
        destination: r(destination),
        left: r(left),
        right: r(right),
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    };
    // Global row-major invocation index, then builtins plus sum/carry outputs.
    ops.extend([
        immediate(13, 8),
        mul(14, 1, 13),
        add(14, 14, 0),
        immediate(13, 48),
        mul(15, 2, 13),
        add(14, 14, 15),
        immediate(13, RECORD_WORDS as u32),
        mul(14, 14, 13),
        immediate(13, 1),
        immediate(17, 7),
    ]);
    if !increment {
        ops.extend([
            immediate(18, u32::MAX),
            ShaderOperation::AddCarry32 {
                destination: r(18),
                carry_out: r(19),
                left: r(18),
                right: r(0),
                carry_in: Some(r(5)),
            },
        ]);
    }
    for value in 0..RECORD_WORDS as u16 {
        if increment {
            ops.push(ShaderOperation::LoadStorageBuffer32 {
                destination: r(16),
                binding: 0,
                word_index: r(14),
            });
            ops.push(add(16, 16, 17));
        }
        ops.push(ShaderOperation::StoreStorageBuffer32 {
            source: r(if increment {
                16
            } else if value < 13 {
                value
            } else {
                value + 5
            }),
            binding: 0,
            word_index: r(14),
        });
        ops.push(add(14, 14, 13));
    }
    ops.push(ShaderOperation::Exit);
    ShaderBackendModule::new(
        VerifiedShaderIr::verify(
            ShaderIr::new(
                ShaderStage::Compute,
                vec![],
                vec![],
                vec![
                    ShaderResourceAccess::new(
                        0,
                        ShaderResourceKind::StorageBuffer,
                        increment,
                        true,
                    )
                    .unwrap(),
                ],
                ops.into_iter()
                    .enumerate()
                    .map(|(i, op)| {
                        ShaderInstruction::new(
                            ShaderSourceLocation::new(i as u32 * 8),
                            ShaderPredicate::Always,
                            op,
                        )
                    })
                    .collect(),
            )
            .with_workgroup_size([4, 2, 2]),
        )
        .unwrap(),
    )
}

#[test]
fn compute_dispatch_preserves_xyz_builtins_storage_and_queued_write_visibility() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(761),
        NonCpuDeviceId::new(761),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            cache: nixe_gpu::GpuCacheConfiguration::new(6, 1, 1, 1, 1024).unwrap(),
            ..Default::default()
        },
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let size = (WORDS as u64 + 1) * 4;
    let canonical = CanonicalAllocation::zeroed(size as usize, 4096).unwrap();
    let allocation = GpuAllocationId::new(1);
    let allocation_description = GpuAllocationDescription::new(size, 4).unwrap();
    let backing = BackingView::new(
        allocation,
        allocation_description,
        0,
        canonical
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let buffer = BufferId::new(1);
    let description = BufferDescription::new(size).unwrap();
    let mut creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(BufferView::new(buffer, description, 0, backing.clone()).unwrap()),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(1),
            description: PipelineDescription {
                kind: PipelineKind::Compute,
            },
        },
        BackendResourceCreateInfo::DescriptorTable {
            id: DescriptorTableId::new(1),
            description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer]).unwrap(),
            bindings: vec![DescriptorTableBinding {
                binding: 0,
                resource: ResourceDependency::Buffer(buffer),
            }]
            .into_boxed_slice(),
        },
    ];
    for (id, increment) in [(1, false), (2, true)] {
        creations.push(BackendResourceCreateInfo::Shader {
            id: ShaderId::new(id),
            description: ShaderDescription {
                stage: ShaderStage::Compute,
            },
            module: kernel(increment),
        });
    }
    // Force pipeline/bind-group eviction, then reuse the resident variant.
    for (index, shader) in [1, 2, 1, 2, 2, 2].into_iter().enumerate() {
        let serial = index as u64 + 1;
        let operation = GpuOperation::new(
            GpuCommand::Dispatch(
                DispatchOperation::new(
                    PipelineId::new(1),
                    ShaderId::new(shader),
                    vec![DescriptorTableId::new(1)],
                    [2, 3, 2],
                )
                .unwrap(),
            ),
            [ResourceAccess::new(
                AccessTarget::Buffer {
                    buffer,
                    range: BufferRange::new(0, WORDS as u64 * 4).unwrap(),
                },
                AccessScope::new(
                    PipelineStages::COMPUTE_SHADER,
                    if shader == 1 {
                        AccessMode::Write
                    } else {
                        AccessMode::ReadWrite
                    },
                    ResourceUsage::StorageBuffer,
                )
                .unwrap(),
            )],
            [ResourceDependency::Buffer(buffer)],
            CapabilityRequirements::none(),
        );
        assert!(
            operation
                .dependencies()
                .contains(&ResourceDependency::Shader(ShaderId::new(shader)))
        );
        let submission = OperationSubmission::new(
            FrontendSubmissionId::new(serial),
            if serial == 1 {
                vec![]
            } else {
                vec![FrontendSubmissionId::new(serial - 1)]
            },
            vec![operation],
        )
        .unwrap();
        runtime
            .runtime()
            .submit(&creations, &[], &submission)
            .unwrap();
        creations.clear();
    }
    // The first CPU demand is after all six dispatches, not between them.
    let mut bytes = vec![0xff; size as usize];
    backing.range().read(0, &mut bytes).unwrap();
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for z in 0..4 {
        for y in 0..6 {
            for x in 0..8 {
                let first = (x + y * 8 + z * 48) as usize * RECORD_WORDS;
                let total = u64::from(u32::MAX) + u64::from(x) + u64::from(z % 2);
                let expected = [
                    x,
                    y,
                    z,
                    x % 4,
                    y % 2,
                    z % 2,
                    x / 4,
                    y / 2,
                    z / 2,
                    2,
                    3,
                    2,
                    x % 4 + (y % 2) * 4 + (z % 2) * 8,
                    total as u32,
                    (total >> 32) as u32,
                ]
                .map(|value| value.wrapping_add(21));
                assert_eq!(
                    &values[first..first + RECORD_WORDS],
                    &expected,
                    "invocation {x},{y},{z}"
                );
            }
        }
    }
    assert_eq!(
        values[WORDS], 0,
        "dispatch must not modify the trailing word"
    );
}

#[test]
fn compute_generated_vertices_are_drawn_without_cpu_materialization() {
    let _guard = accelerated_test_guard();
    compute_generated_vertices(None, false);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn compute_generated_smooth_line_strip_stays_on_gpu() {
    let _guard = accelerated_test_guard();
    let mut previous = 0;
    for width in [1.0, 4.0, 16.0] {
        let Some(coverage) = compute_generated_vertices(Some(width), false) else {
            return;
        };
        assert!(
            coverage > previous,
            "increasing line width must increase covered pixels"
        );
        previous = coverage;
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn native_line_barriers_preserve_both_draws_and_gpu_vertex_ownership() {
    let _guard = accelerated_test_guard();
    compute_generated_vertices(Some(4.0), true);
}

fn compute_generated_vertices(line_width: Option<f32>, split: bool) -> Option<u64> {
    let initialized = initialize_backend(
        BackendInstanceId::new(762),
        NonCpuDeviceId::new(762),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    )?;
    if let Some(width) = line_width {
        let Some(caps) = initialized.adapter.native_vulkan else {
            eprintln!("SKIP: physical GPU lacks native Vulkan rasterization");
            return None;
        };
        if !caps.raster.smooth_lines
            || (width != 1.0 && !caps.raster.wide_lines)
            || width > f32::from_bits(caps.raster.line_width_range_bits[1])
        {
            eprintln!("SKIP: physical GPU lacks the requested smooth line width");
            return None;
        }
    }
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let side = if line_width.is_some() { 64 } else { 4 };
    let page = initialized_page(&vec![0; side * side * 4]);
    let (mut creations, image_backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, side as u32, side as u32, &[page]);
    let vertices = if line_width.is_some() {
        [
            [-0.75_f32, -0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.75, -0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ]
    } else {
        [
            [-1.0_f32, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [3.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [-1.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ]
    };
    let bytes: Vec<_> = vertices
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut buffer_backings = Vec::new();
    for (id, data) in [(10, bytes), (11, vec![0; 96])] {
        let allocation = GpuAllocationId::new(id);
        let allocation_description = GpuAllocationDescription::new(96, 4).unwrap();
        let page = initialized_page(&data);
        let view = backing(allocation, allocation_description, &page);
        let buffer = BufferId::new(id);
        let description = BufferDescription::new(96).unwrap();
        creations.extend([
            BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            },
            BackendResourceCreateInfo::Buffer {
                id: buffer,
                description,
                view: Some(BufferView::new(buffer, description, 0, view.clone()).unwrap()),
            },
        ]);
        buffer_backings.push(view);
    }
    let compute = ShaderBackendModule::new(
        VerifiedShaderIr::verify(
            ShaderIr::new(
                ShaderStage::Compute,
                vec![],
                vec![],
                vec![
                    ShaderResourceAccess::new(0, ShaderResourceKind::StorageBuffer, false, true)
                        .unwrap(),
                    ShaderResourceAccess::new(1, ShaderResourceKind::StorageBuffer, true, false)
                        .unwrap(),
                ],
                [
                    ShaderOperation::LoadComputeBuiltin32 {
                        destination: ShaderRegister::new(0),
                        builtin: ShaderComputeBuiltin::GlobalInvocationId,
                        component: 0,
                    },
                    ShaderOperation::LoadStorageBuffer32 {
                        destination: ShaderRegister::new(1),
                        binding: 1,
                        word_index: ShaderRegister::new(0),
                    },
                    ShaderOperation::StoreStorageBuffer32 {
                        source: ShaderRegister::new(1),
                        binding: 0,
                        word_index: ShaderRegister::new(0),
                    },
                    ShaderOperation::Exit,
                ]
                .into_iter()
                .enumerate()
                .map(|(i, op)| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(i as u32 * 8),
                        ShaderPredicate::Always,
                        op,
                    )
                })
                .collect(),
            )
            .with_workgroup_size([8, 1, 1]),
        )
        .unwrap(),
    );
    for (id, module) in [
        (1, compute),
        (
            2,
            interpolated_vertex_module(ShaderInterpolation::Perspective, false),
        ),
        (
            3,
            interpolated_fragment_module(ShaderInterpolation::Perspective, false, 1.0),
        ),
    ] {
        creations.push(BackendResourceCreateInfo::Shader {
            id: ShaderId::new(id),
            description: ShaderDescription {
                stage: module.stage(),
            },
            module,
        });
    }
    for (id, kind) in [(1, PipelineKind::Compute), (2, PipelineKind::Graphics)] {
        creations.push(BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(id),
            description: PipelineDescription { kind },
        });
    }
    creations.push(BackendResourceCreateInfo::DescriptorTable {
        id: DescriptorTableId::new(1),
        description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer; 2]).unwrap(),
        bindings: vec![
            DescriptorTableBinding {
                binding: 0,
                resource: ResourceDependency::Buffer(BufferId::new(11)),
            },
            DescriptorTableBinding {
                binding: 1,
                resource: ResourceDependency::Buffer(BufferId::new(10)),
            },
        ]
        .into_boxed_slice(),
    });
    let pass = RenderPassId::new(1);
    let pass_description = RenderPassDescription::new(vec![RenderPassAttachmentDescription {
        kind: ImageKind::Color,
        format: ImageFormat::Rgba8Unorm,
        samples: SampleCount::One,
    }])
    .unwrap();
    creations.push(BackendResourceCreateInfo::RenderPass {
        id: pass,
        description: pass_description.clone(),
    });
    let mut prepared = PreparedDraw::new(
        PipelineId::new(2),
        pass,
        if line_width.is_some() {
            PrimitiveTopology::LineStrip
        } else {
            PrimitiveTopology::Triangles
        },
        vec![],
        vec![
            VertexBufferLayout::new(
                BufferRegion {
                    buffer: BufferId::new(11),
                    range: BufferRange::new(0, 96).unwrap(),
                },
                32,
                VertexStepMode::Vertex,
                vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 16,
                        shader_location: 1,
                    },
                ],
            )
            .unwrap(),
        ],
        None,
    )
    .unwrap();
    prepared.line_rasterization = line_width.map(|width| nixe_gpu::LineRasterization {
        width_bits: width.to_bits(),
        smooth: true,
    });
    let op = |command| GpuOperation::new(command, [], [], CapabilityRequirements::none());
    let mut operations = vec![
        GpuOperation::new(
            GpuCommand::Dispatch(
                DispatchOperation::new(
                    PipelineId::new(1),
                    ShaderId::new(1),
                    vec![DescriptorTableId::new(1)],
                    [3, 1, 1],
                )
                .unwrap(),
            ),
            [(10, AccessMode::Read), (11, AccessMode::Write)].map(|(id, mode)| {
                ResourceAccess::new(
                    AccessTarget::Buffer {
                        buffer: BufferId::new(id),
                        range: BufferRange::new(0, 96).unwrap(),
                    },
                    AccessScope::new(
                        PipelineStages::COMPUTE_SHADER,
                        mode,
                        ResourceUsage::StorageBuffer,
                    )
                    .unwrap(),
                )
            }),
            [
                ResourceDependency::Buffer(BufferId::new(10)),
                ResourceDependency::Buffer(BufferId::new(11)),
            ],
            CapabilityRequirements::none(),
        ),
        op(GpuCommand::RenderPass(
            RenderPassOperation::begin(
                pass,
                pass_description,
                vec![RenderAttachment {
                    image,
                    subresources,
                    kind: ImageKind::Color,
                    format: ImageFormat::Rgba8Unorm,
                    samples: SampleCount::One,
                    load: AttachmentLoad::Clear(ClearValue::Color([0.0; 4])),
                    store: AttachmentStore::Store,
                }],
            )
            .unwrap(),
        )),
        GpuOperation::new(
            GpuCommand::Draw(
                DrawOperation::new(
                    Arc::new(prepared),
                    DrawArguments::NonIndexed {
                        first_vertex: 0,
                        vertex_count: 3,
                        first_instance: 0,
                        instance_count: 1,
                    },
                )
                .unwrap(),
            ),
            [],
            [
                ResourceDependency::Shader(ShaderId::new(2)),
                ResourceDependency::Shader(ShaderId::new(3)),
            ],
            CapabilityRequirements::none(),
        ),
        op(GpuCommand::RenderPass(RenderPassOperation::End {
            render_pass: pass,
        })),
    ];
    if line_width.is_some() {
        use nixe_gpu::{BarrierOperation, ResourceTransition};
        // This is the frontend's real ordering: Begin, compute->vertex barrier,
        // Draw, End. The native entry dependency must honor it without CPU work.
        let incoming = op(GpuCommand::Barrier(
            BarrierOperation::new(vec![
                ResourceTransition::new(
                    AccessTarget::Buffer {
                        buffer: BufferId::new(11),
                        range: BufferRange::new(0, 96).unwrap(),
                    },
                    AccessScope::new(
                        PipelineStages::COMPUTE_SHADER,
                        AccessMode::Write,
                        ResourceUsage::StorageBuffer,
                    )
                    .unwrap(),
                    AccessScope::new(
                        PipelineStages::VERTEX_INPUT,
                        AccessMode::Read,
                        ResourceUsage::VertexBuffer,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ])
            .unwrap(),
        ));
        operations.insert(2, incoming);
        if split {
            let first = operations[3]
                .with_draw_arguments(DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 2,
                    first_instance: 0,
                    instance_count: 1,
                })
                .unwrap();
            let second = operations[3]
                .with_draw_arguments(DrawArguments::NonIndexed {
                    first_vertex: 1,
                    vertex_count: 2,
                    first_instance: 0,
                    instance_count: 1,
                })
                .unwrap();
            let barrier = op(GpuCommand::Barrier(
                BarrierOperation::new(vec![
                    ResourceTransition::new(
                        AccessTarget::Image {
                            image,
                            subresources,
                        },
                        AccessScope::new(
                            PipelineStages::COLOR_OUTPUT,
                            AccessMode::Write,
                            ResourceUsage::ColorAttachment,
                        )
                        .unwrap(),
                        AccessScope::new(
                            PipelineStages::COLOR_OUTPUT,
                            AccessMode::ReadWrite,
                            ResourceUsage::ColorAttachment,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ])
                .unwrap(),
            ));
            // Consecutive and trailing barriers must neither produce an empty
            // raw pass nor replay the guest clear on the second segment.
            operations.splice(
                3..4,
                [first, barrier.clone(), barrier.clone(), second, barrier],
            );
        }
    }
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &OperationSubmission::new(FrontendSubmissionId::new(1), vec![], operations).unwrap(),
        )
        .unwrap();
    let mut pixels = vec![0; side * side * 4];
    image_backing.range().read(0, &mut pixels).unwrap();
    if line_width.is_some() {
        assert!(
            pixels.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255),
            "smooth edge coverage"
        );
        assert_eq!(&pixels[..4], &[0; 4], "untouched background");
        for right in [false, true] {
            assert!(
                pixels.chunks_exact(4).enumerate().any(|(index, p)| {
                    let x = index % side;
                    p[3] > 0
                        && if right {
                            x > 2 * side / 3
                        } else {
                            x < side / 3
                        }
                }),
                "both line segments must survive: right={right}"
            );
        }
        assert!(
            pixels.chunks_exact(4).all(|p| p[1] == 0 && p[2] == 0),
            "only red vertices"
        );
    } else {
        assert!(
            pixels
                .chunks_exact(4)
                .all(|pixel| pixel == [255, 0, 0, 255])
        );
    }
    assert!(
        matches!(
            buffer_backings[1].range().segments()[0].visibility_state(),
            nixe_memory::VisibilityState::GpuNewer { .. }
        ),
        "graphics must consume GPU-generated vertices without downloading the buffer"
    );
    Some(pixels.chunks_exact(4).map(|p| u64::from(p[3])).sum())
}

#[test]
fn float32_multiply_rz_ftz_matches_exact_products_on_gpu() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(764),
        NonCpuDeviceId::new(764),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    let mut cases = Vec::new();
    for a in [
        0, 0x80000000, 1, 0x807fffff, 0x00800000, 0x3f800001, 0x3fc00000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234,
    ] {
        for b in [
            0, 0x80000000, 0x3f800001, 0x3f000000, 0xbf800001, 0x7f7fffff, 0x7f800000, 0x7fc05678,
        ] {
            cases.push((a, b));
        }
    }
    let mut random = 0x5ac32d19_u32;
    for _ in 0..8192 {
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        let a = random;
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        cases.push((a, random));
    }
    cases.resize(cases.len().div_ceil(64) * 64, (0, 0));
    let size = cases.len() * 16;
    let canonical = CanonicalAllocation::zeroed(size, 4096).unwrap();
    let mut input = Vec::with_capacity(size);
    for &(a, b) in &cases {
        for word in [a, b, 0, 0] {
            input.extend_from_slice(&word.to_le_bytes());
        }
    }
    canonical.write(0, &input).unwrap();
    let allocation = GpuAllocationId::new(1);
    let allocation_description = GpuAllocationDescription::new(size as u64, 4).unwrap();
    let backing = BackingView::new(
        allocation,
        allocation_description,
        0,
        canonical
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let buffer = BufferId::new(1);
    let description = BufferDescription::new(size as u64).unwrap();
    let r = ShaderRegister::new;
    let immediate = |destination, bits| ShaderOperation::MoveImmediate32 {
        destination: r(destination),
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    };
    let control = ShaderFloatControl::new(
        nixe_gpu::ShaderRoundingMode::TowardZero,
        nixe_gpu::ShaderNanMode::Propagate,
        true,
        true,
        false,
    );
    let ops = vec![
        ShaderOperation::LoadComputeBuiltin32 {
            destination: r(0),
            builtin: ShaderComputeBuiltin::GlobalInvocationId,
            component: 0,
        },
        immediate(1, 4),
        ShaderOperation::Multiply32 {
            destination: r(0),
            left: r(0),
            right: r(1),
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        },
        ShaderOperation::LoadStorageBuffer32 {
            destination: r(2),
            binding: 0,
            word_index: r(0),
        },
        immediate(1, 1),
        ShaderOperation::Add32 {
            destination: r(0),
            left: r(0),
            right: r(1),
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        },
        ShaderOperation::LoadStorageBuffer32 {
            destination: r(3),
            binding: 0,
            word_index: r(0),
        },
        ShaderOperation::Multiply32 {
            destination: r(4),
            left: r(2),
            right: r(3),
            scalar_type: ShaderScalarType::Float32,
            float_control: control,
        },
        ShaderOperation::FloatMultiplyZero32 {
            destination: r(2),
            left: r(2),
            right: r(3),
            float_control: control,
        },
        ShaderOperation::Add32 {
            destination: r(0),
            left: r(0),
            right: r(1),
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        },
        ShaderOperation::StoreStorageBuffer32 {
            source: r(4),
            binding: 0,
            word_index: r(0),
        },
        ShaderOperation::Add32 {
            destination: r(0),
            left: r(0),
            right: r(1),
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        },
        ShaderOperation::StoreStorageBuffer32 {
            source: r(2),
            binding: 0,
            word_index: r(0),
        },
        ShaderOperation::Exit,
    ];
    let ir = ShaderIr::new(
        ShaderStage::Compute,
        vec![],
        vec![],
        vec![ShaderResourceAccess::new(0, ShaderResourceKind::StorageBuffer, true, true).unwrap()],
        ops.into_iter()
            .enumerate()
            .map(|(i, op)| {
                ShaderInstruction::new(
                    ShaderSourceLocation::new(i as u32 * 8),
                    ShaderPredicate::Always,
                    op,
                )
            })
            .collect(),
    )
    .with_workgroup_size([64, 1, 1]);
    let creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(BufferView::new(buffer, description, 0, backing.clone()).unwrap()),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(1),
            description: PipelineDescription {
                kind: PipelineKind::Compute,
            },
        },
        BackendResourceCreateInfo::DescriptorTable {
            id: DescriptorTableId::new(1),
            description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer]).unwrap(),
            bindings: vec![DescriptorTableBinding {
                binding: 0,
                resource: ResourceDependency::Buffer(buffer),
            }]
            .into_boxed_slice(),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(1),
            description: ShaderDescription {
                stage: ShaderStage::Compute,
            },
            module: ShaderBackendModule::new(VerifiedShaderIr::verify(ir).unwrap()),
        },
    ];
    let operation = GpuOperation::new(
        GpuCommand::Dispatch(
            DispatchOperation::new(
                PipelineId::new(1),
                ShaderId::new(1),
                vec![DescriptorTableId::new(1)],
                [cases.len() as u32 / 64, 1, 1],
            )
            .unwrap(),
        ),
        [ResourceAccess::new(
            AccessTarget::Buffer {
                buffer,
                range: BufferRange::new(0, size as u64).unwrap(),
            },
            AccessScope::new(
                PipelineStages::COMPUTE_SHADER,
                AccessMode::ReadWrite,
                ResourceUsage::StorageBuffer,
            )
            .unwrap(),
        )],
        [ResourceDependency::Buffer(buffer)],
        CapabilityRequirements::none(),
    );
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &OperationSubmission::new(FrontendSubmissionId::new(1), vec![], vec![operation])
                .unwrap(),
        )
        .unwrap();
    let mut bytes = vec![0; size];
    backing.range().read(0, &mut bytes).unwrap();
    for (index, &(a, b)) in cases.iter().enumerate() {
        let flush = |bits: u32| {
            if bits & 0x7f800000 == 0 {
                bits & 0x80000000
            } else {
                bits
            }
        };
        let a = flush(a);
        let b = flush(b);
        let exact = f64::from(f32::from_bits(a)) * f64::from(f32::from_bits(b));
        let rounded = exact as f32;
        let mut expected = rounded.to_bits();
        if exact.is_finite() && f64::from(rounded).abs() > exact.abs() {
            expected -= 1;
        }
        expected = flush(expected);
        for (word, zero_absorbs) in [(2, false), (3, true)] {
            let expected = if zero_absorbs && (a & 0x7fffffff == 0 || b & 0x7fffffff == 0) {
                0
            } else {
                expected
            };
            let offset = index * 16 + word * 4;
            let actual = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            if f32::from_bits(expected).is_nan() {
                assert!(f32::from_bits(actual).is_nan());
            } else {
                assert_eq!(
                    actual, expected,
                    "case {index}: {a:08x} * {b:08x}, zero_absorbs={zero_absorbs}"
                );
            }
        }
    }
}
