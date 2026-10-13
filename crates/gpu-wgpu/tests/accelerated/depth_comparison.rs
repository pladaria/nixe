//! Comparison filtering must filter test results, rather than filtered depth.
use super::*;
use nixe_gpu::{
    AddressMode, DepthCompareOperation as Compare, DescriptorKind, DescriptorTableBinding,
    DescriptorTableDescription, DescriptorTableId, FilterMode, SamplerDescription, SamplerId,
    ShaderResourceAccess, ShaderResourceKind, ShaderTextureSampleOutput,
};

#[test]
fn depth_comparison_filters_tests_and_selects_array_layers() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(770),
        NonCpuDeviceId::new(770),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let (mut creations, output, color, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 4, 4, &[initialized_page(&[0; 64])]);
    let vertices = [
        [-1_f32, -1., 0., 1., 1., 1., 1.],
        [3., -1., 0., 1., 1., 1., 1.],
        [-1., 3., 0., 1., 1., 1., 1.],
    ]
    .into_iter()
    .flatten()
    .flat_map(f32::to_le_bytes)
    .collect::<Vec<_>>();
    let allocation = GpuAllocationId::new(770);
    let allocation_description = GpuAllocationDescription::new(vertices.len() as u64, 4).unwrap();
    let buffer = BufferId::new(770);
    let description = BufferDescription::new(vertices.len() as u64).unwrap();
    creations.extend([
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(
                BufferView::new(
                    buffer,
                    description,
                    0,
                    backing(
                        allocation,
                        allocation_description,
                        &initialized_page(&vertices),
                    ),
                )
                .unwrap(),
            ),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(770),
            description: ShaderDescription {
                stage: ShaderStage::Vertex,
            },
            module: interpolated_vertex_module(ShaderInterpolation::Perspective, false),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(770),
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        },
    ]);
    let pass = RenderPassId::new(770);
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
    let operation = |command| GpuOperation::new(command, [], [], CapabilityRequirements::none());
    let mut serial = 1;
    for (format_index, format) in [
        ImageFormat::Depth32Float,
        ImageFormat::Depth24UnormStencil8Uint,
    ]
    .into_iter()
    .enumerate()
    {
        let depth = ImageId::new(770 + format_index as u64);
        creations.push(BackendResourceCreateInfo::Image {
            id: depth,
            description: ImageDescription::new(
                ImageDimension::Two,
                ImageExtent::new(2, 2, 1).unwrap(),
                format,
                ImageKind::DepthStencil,
                1,
                2,
                SampleCount::One,
            )
            .unwrap(),
            view: None,
        });
        let clear = |layer, x, width, value| {
            operation(GpuCommand::Clear(
                ClearOperation::image(
                    ImageRegion {
                        image: depth,
                        subresources: ImageSubresourceRange {
                            base_layer: layer,
                            ..subresources
                        },
                        origin: ImageOrigin { x, y: 0, z: 0 },
                        extent: ImageExtent::new(width, 2, 1).unwrap(),
                    },
                    ImageKind::DepthStencil,
                    format,
                    SampleCount::One,
                    ClearValue::Depth(value),
                )
                .unwrap(),
            ))
        };
        let clears = vec![
            clear(0, 0, 2, 0.9),
            clear(1, 0, 2, 0.25),
            clear(1, 1, 1, 0.75),
        ];
        runtime
            .runtime()
            .submit(
                &creations,
                &[],
                &OperationSubmission::new(FrontendSubmissionId::new(serial), vec![], clears)
                    .unwrap(),
            )
            .unwrap();
        serial += 1;
        creations.clear();
        // Replace layer zero through a real depth-only pass. The .9 clear
        // would produce a different comparison result if the draw were skipped.
        let shadow_pass = RenderPassId::new(8000 + format_index as u64);
        let shadow_fragment = ShaderId::new(8000 + format_index as u64);
        let shadow_description =
            RenderPassDescription::new(vec![RenderPassAttachmentDescription {
                kind: ImageKind::DepthStencil,
                format,
                samples: SampleCount::One,
            }])
            .unwrap();
        creations.extend([
            BackendResourceCreateInfo::RenderPass {
                id: shadow_pass,
                description: shadow_description.clone(),
            },
            BackendResourceCreateInfo::Shader {
                id: shadow_fragment,
                description: ShaderDescription {
                    stage: ShaderStage::Fragment,
                },
                module: nixe_gpu::ShaderBackendModule::new(
                    VerifiedShaderIr::verify(ShaderIr::new(
                        ShaderStage::Fragment,
                        vec![],
                        std::iter::once(
                            ShaderInterfaceElement::new(
                                ShaderIoLocation::FragmentDepth,
                                0,
                                ShaderScalarType::Float32,
                                None,
                            )
                            .unwrap(),
                        )
                        .chain((0..4).map(|component| {
                            ShaderInterfaceElement::new(
                                ShaderIoLocation::Color(0),
                                component,
                                ShaderScalarType::Float32,
                                None,
                            )
                            .unwrap()
                        }))
                        .collect(),
                        vec![],
                        vec![
                            move_f32(0, 0, 0.25),
                            store_output(8, 0, ShaderIoLocation::FragmentDepth, 1),
                            ShaderInstruction::new(
                                ShaderSourceLocation::new(16),
                                ShaderPredicate::Always,
                                ShaderOperation::StoreOutput {
                                    sources: vec![ShaderRegister::new(0); 4].into(),
                                    location: ShaderIoLocation::Color(0),
                                    first_component: 0,
                                    scalar_type: ShaderScalarType::Float32,
                                },
                            ),
                            exit(24),
                        ],
                    ))
                    .unwrap(),
                ),
            },
        ]);
        let mut shadow = PreparedDraw::new(
            PipelineId::new(770),
            shadow_pass,
            PrimitiveTopology::Triangles,
            vec![],
            vec![
                VertexBufferLayout::new(
                    BufferRegion {
                        buffer,
                        range: BufferRange::new(0, vertices.len() as u64).unwrap(),
                    },
                    28,
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
        shadow.depth_state = nixe_gpu::DepthState {
            test_enabled: true,
            write_enabled: true,
            compare: Compare::Always,
        };
        let begin = RenderPassOperation::begin(
            shadow_pass,
            shadow_description,
            vec![RenderAttachment {
                image: depth,
                subresources,
                kind: ImageKind::DepthStencil,
                format,
                samples: SampleCount::One,
                load: AttachmentLoad::Load,
                store: AttachmentStore::Store,
            }],
        )
        .unwrap();
        let write = GpuOperation::new(
            GpuCommand::Draw(
                DrawOperation::new(
                    Arc::new(shadow),
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
                ResourceDependency::Buffer(buffer),
                ResourceDependency::Shader(ShaderId::new(770)),
                ResourceDependency::Shader(shadow_fragment),
            ],
            CapabilityRequirements::none(),
        );
        runtime
            .runtime()
            .submit(
                &creations,
                &[],
                &OperationSubmission::new(
                    FrontendSubmissionId::new(serial),
                    vec![],
                    vec![
                        operation(GpuCommand::RenderPass(begin)),
                        write,
                        operation(GpuCommand::RenderPass(RenderPassOperation::end(
                            shadow_pass,
                        ))),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        serial += 1;
        creations.clear();
        for (case, (layer, reference, compare, offset, expected)) in [
            (0, 0.6, Compare::LessEqual, None, 0),
            (1, 0.6, Compare::LessEqual, None, 128),
            (1, 0.2, Compare::LessEqual, None, 255),
            (1, 0.6, Compare::Greater, None, 128),
            (1, 0.6, Compare::LessEqual, Some([-1, 0]), 0),
            (1, 0.6, Compare::LessEqual, Some([1, 0]), 255),
            (1, 0.6, Compare::Greater, Some([-8, 7]), 255),
            (1, 0.6, Compare::Greater, Some([7, -8]), 0),
        ]
        .into_iter()
        .enumerate()
        {
            let id = 780 + format_index as u64 * 10 + case as u64;
            let sampler = SamplerId::new(id);
            let table = DescriptorTableId::new(id);
            let fragment = ShaderId::new(id);
            let mut description = SamplerDescription::new(
                FilterMode::Linear,
                FilterMode::Linear,
                FilterMode::Nearest,
                [AddressMode::ClampToEdge; 3],
                0.,
                0.,
                1.,
            )
            .unwrap();
            description.compare = Some(compare);
            creations.push(BackendResourceCreateInfo::Sampler {
                id: sampler,
                description,
            });
            creations.push(BackendResourceCreateInfo::Shader {
                id: fragment,
                description: ShaderDescription {
                    stage: ShaderStage::Fragment,
                },
                module: fragment_shader(layer, reference, offset),
            });
            creations.push(BackendResourceCreateInfo::DescriptorTable {
                id: table,
                description: DescriptorTableDescription::new(vec![
                    DescriptorKind::SampledImage,
                    DescriptorKind::Sampler,
                ])
                .unwrap(),
                bindings: vec![
                    DescriptorTableBinding {
                        binding: 0,
                        resource: ResourceDependency::Image(depth),
                    },
                    DescriptorTableBinding {
                        binding: 1,
                        resource: ResourceDependency::Sampler(sampler),
                    },
                ]
                .into_boxed_slice(),
            });
            let prepared = PreparedDraw::new(
                PipelineId::new(770),
                pass,
                PrimitiveTopology::Triangles,
                vec![table],
                vec![
                    VertexBufferLayout::new(
                        BufferRegion {
                            buffer,
                            range: BufferRange::new(0, vertices.len() as u64).unwrap(),
                        },
                        28,
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
            let begin = RenderPassOperation::begin(
                pass,
                pass_description.clone(),
                vec![RenderAttachment {
                    image: color,
                    subresources,
                    kind: ImageKind::Color,
                    format: ImageFormat::Rgba8Unorm,
                    samples: SampleCount::One,
                    load: AttachmentLoad::Clear(ClearValue::Color([0.; 4])),
                    store: AttachmentStore::Store,
                }],
            )
            .unwrap();
            let draw = GpuOperation::new(
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
                    ResourceDependency::Buffer(buffer),
                    ResourceDependency::Shader(ShaderId::new(770)),
                    ResourceDependency::Shader(fragment),
                    ResourceDependency::Image(depth),
                    ResourceDependency::Sampler(sampler),
                ],
                CapabilityRequirements::none(),
            );
            runtime
                .runtime()
                .submit(
                    &creations,
                    &[],
                    &OperationSubmission::new(
                        FrontendSubmissionId::new(serial),
                        vec![],
                        vec![
                            operation(GpuCommand::RenderPass(begin)),
                            draw,
                            operation(GpuCommand::RenderPass(RenderPassOperation::end(pass))),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
            serial += 1;
            creations.clear();
            let mut pixels = [0; 64];
            output.range().read(0, &mut pixels).unwrap();
            for pixel in pixels.chunks_exact(4) {
                assert!(
                    pixel[..3].iter().all(|v| v.abs_diff(expected) <= 1) && pixel[3] == 255,
                    "format={format:?} layer={layer} reference={reference} compare={compare:?} pixel={pixel:?} expected={expected}"
                );
            }
        }
    }
}

fn fragment_shader(
    layer: u16,
    reference: f32,
    offset: Option<[i32; 2]>,
) -> nixe_gpu::ShaderBackendModule {
    let r = ShaderRegister::new;
    let instructions = vec![
        move_f32(0, 0, 0.5),
        move_f32(8, 1, 0.5),
        move_f32(16, 2, reference),
        ShaderInstruction::new(
            ShaderSourceLocation::new(24),
            ShaderPredicate::Always,
            ShaderOperation::MoveImmediate32 {
                destination: r(3),
                bits: u32::from(layer),
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ),
        ShaderInstruction::new(
            ShaderSourceLocation::new(32),
            ShaderPredicate::Always,
            ShaderOperation::SampleDepthCompare2D {
                offset,
                outputs: (0..4)
                    .map(|c| ShaderTextureSampleOutput::new(r(4 + u16::from(c)), c).unwrap())
                    .collect(),
                coordinates: [r(0), r(1)],
                array_index: Some(r(3)),
                reference: r(2),
                image_binding: 0,
                sampler_binding: 1,
                level_zero: true,
            },
        ),
        store_output(40, 4, ShaderIoLocation::Color(0), 4),
        exit(48),
    ];
    nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            (0..4)
                .map(|c| {
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        c,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap()
                })
                .collect(),
            vec![
                ShaderResourceAccess::new(
                    0,
                    ShaderResourceKind::SampledDepthImage2DArray,
                    true,
                    false,
                )
                .unwrap(),
                ShaderResourceAccess::new(1, ShaderResourceKind::ComparisonSampler, true, false)
                    .unwrap(),
            ],
            instructions,
        ))
        .unwrap(),
    )
}
