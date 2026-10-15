//! Physical oracle for unequal attachment sizes and sampled read-only stencil/depth.
use super::*;
use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, DepthCompareOperation, DepthState, DescriptorKind,
    DescriptorTableBinding, DescriptorTableDescription, DescriptorTableId, PipelineStages,
    ResourceAccess, ResourceUsage, ShaderBackendModule, ShaderResourceAccess, ShaderResourceKind,
    ShaderTextureSampleOutput, StencilTestState,
};

#[test]
fn cropped_color_preserves_padding_and_readonly_stencil_shares_sampled_depth() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(752),
        NonCpuDeviceId::new(752),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let original = [16, 32, 48, 255].repeat(32);
    let (mut creations, output_backing, color, subresources) = backed_color_image(
        ImageFormat::Rgba8Unorm,
        8,
        4,
        &[initialized_page(&original)],
    );
    let depth = ImageId::new(752);
    let extent = ImageExtent {
        width: 4,
        height: 4,
        depth: 1,
    };
    creations.push(BackendResourceCreateInfo::Image {
        id: depth,
        description: ImageDescription::new(
            ImageDimension::Two,
            extent,
            ImageFormat::Depth24UnormStencil8Uint,
            ImageKind::DepthStencil,
            1,
            1,
            SampleCount::One,
        )
        .unwrap(),
        view: None,
    });
    let vertices: Vec<u8> = [
        [-1.0_f32, -1.0, 0.25, 1.0, 1.0, 1.0, 1.0],
        [3.0, -1.0, 0.25, 1.0, 1.0, 1.0, 1.0],
        [-1.0, 3.0, 0.25, 1.0, 1.0, 1.0, 1.0],
    ]
    .into_iter()
    .flatten()
    .flat_map(f32::to_le_bytes)
    .collect();
    let allocation = GpuAllocationId::new(752);
    let allocation_description = GpuAllocationDescription::new(vertices.len() as u64, 4).unwrap();
    let buffer = BufferId::new(752);
    let description = BufferDescription::new(vertices.len() as u64).unwrap();
    creations.push(BackendResourceCreateInfo::Allocation {
        id: allocation,
        description: allocation_description,
    });
    creations.push(BackendResourceCreateInfo::Buffer {
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
    });
    let vertex = ShaderId::new(752);
    let fragment = ShaderId::new(753);
    creations.push(BackendResourceCreateInfo::Shader {
        id: vertex,
        description: ShaderDescription {
            stage: ShaderStage::Vertex,
        },
        module: interpolated_vertex_module(ShaderInterpolation::Perspective, false),
    });
    let input_zero = |register| {
        ShaderInstruction::new(
            ShaderSourceLocation::new(u32::from(register) * 8),
            ShaderPredicate::Always,
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(register),
                bits: 0,
                scalar_type: ShaderScalarType::Signed32,
            },
        )
    };
    let module = ShaderBackendModule::new(
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
                ShaderResourceAccess::new(0, ShaderResourceKind::SampledDepthImage, true, false)
                    .unwrap(),
            ],
            vec![
                input_zero(0),
                input_zero(1),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::LoadTexture2D {
                        outputs: (0..4)
                            .map(|c| {
                                ShaderTextureSampleOutput::new(ShaderRegister::new(u16::from(c)), c)
                                    .unwrap()
                            })
                            .collect(),
                        coordinates: [ShaderRegister::new(0), ShaderRegister::new(1)],
                        image_binding: 0,
                        mip_level: 0,
                    },
                ),
                store_output(24, 0, ShaderIoLocation::Color(0), 4),
                exit(32),
            ],
        ))
        .unwrap(),
    );
    creations.push(BackendResourceCreateInfo::Shader {
        id: fragment,
        description: ShaderDescription {
            stage: ShaderStage::Fragment,
        },
        module,
    });
    let pipeline = PipelineId::new(752);
    creations.push(BackendResourceCreateInfo::Pipeline {
        id: pipeline,
        description: PipelineDescription {
            kind: PipelineKind::Graphics,
        },
    });
    let table = DescriptorTableId::new(752);
    creations.push(BackendResourceCreateInfo::DescriptorTable {
        id: table,
        description: DescriptorTableDescription::new(vec![DescriptorKind::SampledImage]).unwrap(),
        bindings: vec![DescriptorTableBinding {
            binding: 0,
            resource: ResourceDependency::Image(depth),
        }]
        .into_boxed_slice(),
    });
    let pass = RenderPassId::new(752);
    let pass_description = RenderPassDescription::new(vec![
        RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
        },
        RenderPassAttachmentDescription {
            kind: ImageKind::DepthStencil,
            format: ImageFormat::Depth24UnormStencil8Uint,
            samples: SampleCount::One,
        },
    ])
    .unwrap();
    creations.push(BackendResourceCreateInfo::RenderPass {
        id: pass,
        description: pass_description.clone(),
    });
    let prepare = |revision| {
        let mut prepared = PreparedDraw::new(
            nixe_gpu::PreparedStateId::new(1, revision),
            pipeline,
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
        prepared.edit_state().depth_state =
            DepthState::new(true, false, DepthCompareOperation::Always);
        prepared
    };
    let depth_transform = |scale, offset| {
        Some(nixe_gpu::FragmentDepthTransform::new(scale, offset, 0.0, 1.0).unwrap())
    };
    for (index, (reference, transform, compare, passed)) in [
        (2, None, DepthCompareOperation::Always, false),
        (0, None, DepthCompareOperation::Always, true),
        (
            0,
            depth_transform(2.0, -1.0),
            DepthCompareOperation::Less,
            true,
        ),
        (
            0,
            depth_transform(2.0, 0.0),
            DepthCompareOperation::Less,
            false,
        ),
        (
            0,
            depth_transform(2.0, -1.0),
            DepthCompareOperation::Less,
            true,
        ),
        (2, None, DepthCompareOperation::Always, false),
    ]
    .into_iter()
    .enumerate()
    {
        let mut prepared = prepare(index as u64 + 1);
        prepared.edit_state().depth_state = DepthState::new(true, false, compare);
        prepared.edit_state().viewport_transform = transform.map(|depth| {
            nixe_gpu::ViewportTransform::new([2.0, -2.0, 1.0], [2.0, 2.0, 0.0], [0.0, 1.0])
                .unwrap()
                .with_fragment_depth_transform(depth)
        });
        prepared.edit_state().stencil_state = Some(StencilTestState {
            front: nixe_gpu::StencilFaceState::read_only(
                DepthCompareOperation::NotEqual,
                2,
                reference,
            ),
            back: nixe_gpu::StencilFaceState::read_only(
                DepthCompareOperation::NotEqual,
                2,
                reference,
            ),
        });
        let begin = RenderPassOperation::begin(
            pass,
            pass_description.clone(),
            vec![
                RenderAttachment {
                    image: color,
                    subresources,
                    kind: ImageKind::Color,
                    format: ImageFormat::Rgba8Unorm,
                    samples: SampleCount::One,
                    load: AttachmentLoad::Load,
                    store: AttachmentStore::Store,
                },
                RenderAttachment {
                    image: depth,
                    subresources,
                    kind: ImageKind::DepthStencil,
                    format: ImageFormat::Depth24UnormStencil8Uint,
                    samples: SampleCount::One,
                    load: AttachmentLoad::Load,
                    store: AttachmentStore::Store,
                },
            ],
        )
        .unwrap();
        let clear = ClearOperation::image(
            nixe_gpu::ImageRegion {
                image: depth,
                subresources,
                origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
                extent,
            },
            ImageKind::DepthStencil,
            ImageFormat::Depth24UnormStencil8Uint,
            SampleCount::One,
            ClearValue::DepthStencil {
                depth: 0.25,
                stencil: 2,
            },
        )
        .unwrap();
        let reset_color = ClearOperation::image(
            nixe_gpu::ImageRegion {
                image: color,
                subresources,
                origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
                extent: ImageExtent {
                    width: 8,
                    height: 4,
                    depth: 1,
                },
            },
            ImageKind::Color,
            ImageFormat::Rgba8Unorm,
            SampleCount::One,
            ClearValue::Color([16.0 / 255.0, 32.0 / 255.0, 48.0 / 255.0, 1.0]),
        )
        .unwrap();
        let ops = vec![
            GpuOperation::new(
                GpuCommand::Clear(reset_color),
                [],
                [],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::Clear(clear),
                [],
                [],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::RenderPass(begin),
                [],
                [],
                CapabilityRequirements::none(),
            ),
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
                [ResourceAccess::new(
                    AccessTarget::Image {
                        image: depth,
                        subresources,
                    },
                    AccessScope::new(
                        PipelineStages::FRAGMENT_SHADER,
                        AccessMode::Read,
                        ResourceUsage::SampledImage,
                    )
                    .unwrap(),
                )],
                [
                    ResourceDependency::Buffer(buffer),
                    ResourceDependency::Shader(vertex),
                    ResourceDependency::Shader(fragment),
                    ResourceDependency::Image(depth),
                ],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::RenderPass(RenderPassOperation::end(pass)),
                [],
                [],
                CapabilityRequirements::none(),
            ),
        ];
        runtime
            .runtime()
            .submit(
                &creations,
                &[],
                &OperationSubmission::new(FrontendSubmissionId::new(index as u64 + 1), vec![], ops)
                    .unwrap(),
            )
            .unwrap();
        creations.clear();
        let mut pixels = [0; 128];
        output_backing.range().read(0, &mut pixels).unwrap();
        for y in 0..4 {
            for x in 0..8 {
                let actual = &pixels[(y * 8 + x) * 4..(y * 8 + x + 1) * 4];
                let expected = if x < 4 && passed {
                    [64, 64, 64, 255]
                } else {
                    [16, 32, 48, 255]
                };
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        a.abs_diff(e) <= 1,
                        "({x},{y}) ref={reference}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }
}
