//! Render to several resident targets, then fetch every result in a later pass.
use super::*;
use nixe_gpu::{
    BlendComponent, BlendFactor, BlendOperation, ColorBlendState, ColorOutputState, ColorWriteMask,
    DescriptorKind, DescriptorTableBinding, DescriptorTableDescription, DescriptorTableId,
    ShaderBackendModule, ShaderResourceAccess, ShaderResourceKind, ShaderTextureSampleOutput,
};

const SIZE: u32 = 4;
const COLORS: [[f32; 4]; 3] = [
    [0.25, 0.125, 0.75, 1.0],
    [0.75, 0.5, 0.25, 0.5],
    [0.125, 0.75, 0.5, 1.0],
];

fn interface(location: ShaderIoLocation) -> Vec<ShaderInterfaceElement> {
    (0..4)
        .map(|component| {
            ShaderInterfaceElement::new(location, component, ShaderScalarType::Float32, None)
                .unwrap()
        })
        .collect()
}

fn fragment(count: u8) -> ShaderBackendModule {
    let mut instructions = Vec::new();
    let mut outputs = Vec::new();
    for slot in 0..count {
        outputs.extend(interface(ShaderIoLocation::Color(slot)));
        for component in 0..4 {
            instructions.push(move_f32(
                instructions.len() as u32 * 8,
                component,
                COLORS[slot as usize][component as usize],
            ));
        }
        instructions.push(store_output(
            instructions.len() as u32 * 8,
            0,
            ShaderIoLocation::Color(slot),
            4,
        ));
    }
    instructions.push(exit(instructions.len() as u32 * 8));
    ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            outputs,
            vec![],
            instructions,
        ))
        .unwrap(),
    )
}

fn fetch_fragment() -> ShaderBackendModule {
    let zero = |register| {
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
    ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            interface(ShaderIoLocation::Color(0)),
            vec![
                ShaderResourceAccess::new(0, ShaderResourceKind::SampledImage, true, false)
                    .unwrap(),
            ],
            vec![
                zero(0),
                zero(1),
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
    )
}

#[test]
fn multiple_color_targets_preserve_slot_formats_blending_masks_and_sampling() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(741),
        NonCpuDeviceId::new(741),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let presentation = initialized.presentation_context();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let page = initialized_page(&vec![0; (SIZE * SIZE * 4) as usize]);
    let (mut creations, output_backing, output, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, SIZE, SIZE, &[page]);
    let formats = [
        ImageFormat::Rgba16Float,
        ImageFormat::Rgba8Unorm,
        ImageFormat::Rgba16Float,
        ImageFormat::Rgba16Float,
    ];
    for (index, format) in formats.into_iter().enumerate() {
        let id = ImageId::new(10 + index as u64);
        creations.push(BackendResourceCreateInfo::Image {
            id,
            description: ImageDescription::new(
                ImageDimension::Two,
                ImageExtent::new(SIZE, SIZE, 1).unwrap(),
                format,
                ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap(),
            view: None,
        });
        creations.push(BackendResourceCreateInfo::DescriptorTable {
            id: DescriptorTableId::new(10 + index as u64),
            description: DescriptorTableDescription::new(vec![DescriptorKind::SampledImage])
                .unwrap(),
            bindings: vec![DescriptorTableBinding {
                binding: 0,
                resource: ResourceDependency::Image(id),
            }]
            .into_boxed_slice(),
        });
    }
    let vertex_data: Vec<u8> = [
        [-1.0_f32, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ]
    .into_iter()
    .flatten()
    .flat_map(f32::to_le_bytes)
    .collect();
    let vertex_page = initialized_page(&vertex_data);
    let allocation = GpuAllocationId::new(1);
    let allocation_description = GpuAllocationDescription::new(48, 4).unwrap();
    let buffer_description = BufferDescription::new(48).unwrap();
    let buffer = BufferId::new(1);
    creations.extend([
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description: buffer_description,
            view: Some(
                BufferView::new(
                    buffer,
                    buffer_description,
                    0,
                    backing(allocation, allocation_description, &vertex_page),
                )
                .unwrap(),
            ),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(1),
            description: ShaderDescription {
                stage: ShaderStage::Vertex,
            },
            module: ShaderBackendModule::new(
                VerifiedShaderIr::verify(ShaderIr::new(
                    ShaderStage::Vertex,
                    interface(ShaderIoLocation::Generic(0)),
                    interface(ShaderIoLocation::Position),
                    vec![],
                    vec![
                        load_input(0, 0, 0, ShaderIoLocation::Generic(0), 4),
                        store_output(8, 0, ShaderIoLocation::Position, 4),
                        exit(16),
                    ],
                ))
                .unwrap(),
            ),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(5),
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: fetch_fragment(),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(1),
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        },
    ]);
    creations.push(BackendResourceCreateInfo::Image {
        id: ImageId::new(14),
        description: ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(SIZE, SIZE, 1).unwrap(),
            ImageFormat::Depth32Float,
            ImageKind::DepthStencil,
            1,
            1,
            SampleCount::One,
        )
        .unwrap(),
        view: None,
    });
    for count in 1..=3 {
        creations.push(BackendResourceCreateInfo::Shader {
            id: ShaderId::new(1 + u64::from(count)),
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: fragment(count),
        });
    }
    let layout = VertexBufferLayout::new(
        BufferRegion {
            buffer,
            range: BufferRange::new(0, 48).unwrap(),
        },
        16,
        VertexStepMode::Vertex,
        vec![VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        }],
    )
    .unwrap();
    let op = |command| GpuOperation::new(command, [], [], CapabilityRequirements::none());
    let draw_op = |prepared, shader, sampled: Option<ImageId>| {
        GpuOperation::new(
            GpuCommand::Draw(
                DrawOperation::new(
                    prepared,
                    DrawArguments::NonIndexed {
                        first_vertex: 0,
                        vertex_count: 3,
                        first_instance: 0,
                        instance_count: 1,
                    },
                )
                .unwrap(),
            ),
            sampled.map(|image| {
                nixe_gpu::ResourceAccess::new(
                    nixe_gpu::AccessTarget::Image {
                        image,
                        subresources,
                    },
                    nixe_gpu::AccessScope::new(
                        nixe_gpu::PipelineStages::FRAGMENT_SHADER,
                        nixe_gpu::AccessMode::Read,
                        nixe_gpu::ResourceUsage::SampledImage,
                    )
                    .unwrap(),
                )
            }),
            [
                ResourceDependency::Shader(ShaderId::new(1)),
                ResourceDependency::Shader(ShaderId::new(shader)),
            ]
            .into_iter()
            .chain(sampled.map(ResourceDependency::Image)),
            CapabilityRequirements::none(),
        )
    };
    let mut serial = 0;
    let mut submit = |creations: &[BackendResourceCreateInfo], operations| {
        serial += 1;
        runtime
            .runtime()
            .submit(
                creations,
                &[],
                &OperationSubmission::new(FrontendSubmissionId::new(serial), vec![], operations)
                    .unwrap(),
            )
            .unwrap();
    };
    // Reuse one neutral pipeline across target counts, slot formats and masks.
    for (variant, (indices, masked)) in [
        (vec![0, 1, 2], false),
        (vec![0, 3, 2], true),
        (vec![0, 1], true),
        (vec![0], false),
        (vec![0, 1, 2], false),
    ]
    .into_iter()
    .enumerate()
    {
        let pass = RenderPassId::new(20 + variant as u64);
        let mut attachment_descriptions: Vec<_> = indices
            .iter()
            .map(|i| RenderPassAttachmentDescription {
                kind: ImageKind::Color,
                format: formats[*i],
                samples: SampleCount::One,
            })
            .collect();
        attachment_descriptions.push(RenderPassAttachmentDescription {
            kind: ImageKind::DepthStencil,
            format: ImageFormat::Depth32Float,
            samples: SampleCount::One,
        });
        let description = RenderPassDescription::new(attachment_descriptions).unwrap();
        creations.push(BackendResourceCreateInfo::RenderPass {
            id: pass,
            description: description.clone(),
        });
        let mut attachments: Vec<_> = indices
            .iter()
            .map(|i| RenderAttachment {
                image: ImageId::new(10 + *i as u64),
                subresources,
                kind: ImageKind::Color,
                format: formats[*i],
                samples: SampleCount::One,
                load: AttachmentLoad::Clear(ClearValue::Color([0.125, 0.25, 0.375, 1.0])),
                store: AttachmentStore::Store,
            })
            .collect();
        attachments.push(RenderAttachment {
            image: ImageId::new(14),
            subresources,
            kind: ImageKind::DepthStencil,
            format: ImageFormat::Depth32Float,
            samples: SampleCount::One,
            load: AttachmentLoad::Clear(ClearValue::Depth(1.0)),
            store: AttachmentStore::Store,
        });
        let mut prepared = PreparedDraw::new(
            PipelineId::new(1),
            pass,
            PrimitiveTopology::Triangles,
            vec![],
            vec![layout.clone()],
            None,
        )
        .unwrap()
        .with_depth_state(nixe_gpu::DepthState::new(
            true,
            true,
            nixe_gpu::DepthCompareOperation::Less,
        ));
        if masked {
            let blend = BlendComponent {
                operation: BlendOperation::Add,
                source: BlendFactor::SourceAlpha,
                destination: BlendFactor::OneMinusSourceAlpha,
            };
            prepared.color_outputs[1] = ColorOutputState {
                blend: Some(ColorBlendState {
                    color: blend,
                    alpha: blend,
                }),
                write_mask: ColorWriteMask::new(false, true, true, false),
            };
        }
        let prepared = Arc::new(prepared);
        for _ in 0..2 {
            submit(
                &creations,
                vec![
                    op(GpuCommand::RenderPass(
                        RenderPassOperation::begin(pass, description.clone(), attachments.clone())
                            .unwrap(),
                    )),
                    draw_op(Arc::clone(&prepared), 1 + indices.len() as u64, None),
                    op(GpuCommand::RenderPass(RenderPassOperation::end(pass))),
                ],
            );
            creations.clear();
        }
        for (slot, index) in indices.into_iter().enumerate() {
            let pass = RenderPassId::new(100 + variant as u64 * 4 + slot as u64);
            let description = RenderPassDescription::new(vec![RenderPassAttachmentDescription {
                kind: ImageKind::Color,
                format: ImageFormat::Rgba8Unorm,
                samples: SampleCount::One,
            }])
            .unwrap();
            let prepared = Arc::new(
                PreparedDraw::new(
                    PipelineId::new(1),
                    pass,
                    PrimitiveTopology::Triangles,
                    vec![DescriptorTableId::new(10 + index as u64)],
                    vec![layout.clone()],
                    None,
                )
                .unwrap(),
            );
            submit(
                &[BackendResourceCreateInfo::RenderPass {
                    id: pass,
                    description: description.clone(),
                }],
                vec![
                    op(GpuCommand::RenderPass(
                        RenderPassOperation::begin(
                            pass,
                            description,
                            vec![RenderAttachment {
                                image: output,
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
                    draw_op(prepared, 5, Some(ImageId::new(10 + index as u64))),
                    op(GpuCommand::RenderPass(RenderPassOperation::end(pass))),
                ],
            );
            let resident = runtime
                .runtime()
                .acquire_presentable_image(PresentationImageRequest {
                    allow_canonical_import: true,
                    cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(
                        output_backing.range(),
                    )
                    .unwrap(),
                    backing: output_backing.clone(),
                    width: SIZE,
                    height: SIZE,
                    format: PresentationImageFormat::Rgba8,
                    layout: ImageMemoryLayout::PitchLinear {
                        row_pitch: u64::from(SIZE * 4),
                        layer_stride: u64::from(SIZE * SIZE * 4),
                    },
                    row_pitch: SIZE * 4,
                })
                .unwrap();
            let expected = if masked && slot == 1 {
                [0.125, 0.375, 0.3125, 1.0]
            } else {
                COLORS[slot]
            };
            for pixel in read_presented_rgba(&presentation, &resident).chunks_exact(4) {
                for (actual, expected) in pixel.iter().zip(expected) {
                    let expected = (expected * 255.0_f32).round() as u8;
                    assert!(
                        actual.abs_diff(expected) <= 2,
                        "variant={variant} slot={slot}: {pixel:?}, expected channel {expected}"
                    );
                }
            }
        }
    }
}
