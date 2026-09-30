use super::*;
use nixe_gpu::{
    AddressMode, DescriptorKind, DescriptorTableBinding, DescriptorTableDescription,
    DescriptorTableId, FilterMode, SamplerDescription, SamplerId, ShaderResourceAccess,
    ShaderResourceKind, ShaderTextureSampleOutput,
};

#[test]
fn bc1_sampling_preserves_alpha_srgb_block_layout_and_cached_pipeline_variants() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(720),
        NonCpuDeviceId::new(720),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let presentation = initialized.presentation_context();
    if !presentation
        .device()
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    {
        eprintln!("SKIP: physical GPU lacks native BC texture compression");
        return;
    }
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let (mut creations, target_backing, target, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 4, 4, &[initialized_page(&[0; 64])]);
    let vertex_buffer = BufferId::new(720);
    let vertex_allocation = GpuAllocationId::new(720);
    let vertices = [-1.0_f32, -1.0, 0.0, 3.0, -1.0, 0.0, -1.0, 3.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let vertex_description = BufferDescription::new(vertices.len() as u64).unwrap();
    let allocation_description = GpuAllocationDescription::new(vertices.len() as u64, 4).unwrap();
    creations.push(BackendResourceCreateInfo::Allocation {
        id: vertex_allocation,
        description: allocation_description,
    });
    creations.push(BackendResourceCreateInfo::Buffer {
        id: vertex_buffer,
        description: vertex_description,
        view: Some(
            BufferView::new(
                vertex_buffer,
                vertex_description,
                0,
                backing(
                    vertex_allocation,
                    allocation_description,
                    &initialized_page(&vertices),
                ),
            )
            .unwrap(),
        ),
    });
    let vertex = ShaderId::new(720);
    let vertex_module = nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            (0..3)
                .map(|component| interface(ShaderIoLocation::Generic(0), component))
                .collect(),
            (0..4)
                .map(|component| interface(ShaderIoLocation::Position, component))
                .collect(),
            vec![],
            vec![
                load_input(8, 0, 0, ShaderIoLocation::Generic(0), 3),
                move_f32(16, 3, 1.0),
                store_output(24, 0, ShaderIoLocation::Position, 4),
                exit(32),
            ],
        ))
        .unwrap(),
    );
    creations.push(BackendResourceCreateInfo::Shader {
        id: vertex,
        description: ShaderDescription {
            stage: ShaderStage::Vertex,
        },
        module: vertex_module,
    });
    let pipeline = PipelineId::new(720);
    creations.push(BackendResourceCreateInfo::Pipeline {
        id: pipeline,
        description: PipelineDescription {
            kind: PipelineKind::Graphics,
        },
    });
    let render_pass = RenderPassId::new(720);
    let pass_description = RenderPassDescription::new(vec![RenderPassAttachmentDescription {
        kind: ImageKind::Color,
        format: ImageFormat::Rgba8Unorm,
        samples: SampleCount::One,
    }])
    .unwrap();
    creations.push(BackendResourceCreateInfo::RenderPass {
        id: render_pass,
        description: pass_description.clone(),
    });
    let sampler = SamplerId::new(720);
    creations.push(BackendResourceCreateInfo::Sampler {
        id: sampler,
        description: SamplerDescription::new(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
            [AddressMode::ClampToEdge; 3],
            0.0,
            0.0,
            1.0,
        )
        .unwrap(),
    });
    let formats = [
        ImageFormat::Bc1RgbUnorm,
        ImageFormat::Bc1RgbaUnorm,
        ImageFormat::Bc1RgbSrgb,
        ImageFormat::Bc1RgbaSrgb,
    ];
    let mut serial = 720;
    for array in [false, true] {
        let fragment = ShaderId::new(if array { 722 } else { 721 });
        creations.push(BackendResourceCreateInfo::Shader {
            id: fragment,
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: sampled_fragment(array),
        });
        let layers = if array { 2 } else { 1 };
        let mut draws = Vec::new();
        for (index, format) in formats.into_iter().enumerate() {
            let id = 730 + index as u64 + if array { 4 } else { 0 };
            let allocation = GpuAllocationId::new(id);
            let image = ImageId::new(id);
            let table = DescriptorTableId::new(id);
            // Two rows of two BC1 blocks. The bottom-right block is selected.
            // In a 64-byte x 8-row GOB, block (1,1) starts at byte 24.
            let mut bytes = vec![0; usize::from(layers) * 512];
            let selected = usize::from(layers - 1) * 512 + 24;
            bytes[selected..selected + 8].copy_from_slice(&[0, 0, 255, 255, 255, 255, 255, 255]);
            let allocation_description =
                GpuAllocationDescription::new(bytes.len() as u64, 4).unwrap();
            let texture_page = initialized_page(&bytes);
            let image_backing = backing(allocation, allocation_description, &texture_page);
            let description = ImageDescription::new(
                ImageDimension::Two,
                ImageExtent::new(8, 8, 1).unwrap(),
                format,
                ImageKind::Color,
                1,
                layers,
                SampleCount::One,
            )
            .unwrap();
            creations.push(BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            });
            creations.push(BackendResourceCreateInfo::Image {
                id: image,
                description,
                view: Some(
                    ImageView::new(
                        image,
                        description,
                        Swizzle::IDENTITY,
                        vec![(
                            ImageSubresourceRange {
                                layer_count: layers,
                                ..subresources
                            },
                            ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                                block_width_log2: 0,
                                block_height_log2: 0,
                                block_depth_log2: 0,
                                layer_stride: 512,
                            }),
                            image_backing.clone(),
                        )],
                    )
                    .unwrap(),
                ),
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
                        resource: ResourceDependency::Image(image),
                    },
                    DescriptorTableBinding {
                        binding: 1,
                        resource: ResourceDependency::Sampler(sampler),
                    },
                ]
                .into_boxed_slice(),
            });
            let prepared = Arc::new(
                PreparedDraw::new(
                    pipeline,
                    render_pass,
                    PrimitiveTopology::Triangles,
                    vec![table],
                    vec![
                        VertexBufferLayout::new(
                            BufferRegion {
                                buffer: vertex_buffer,
                                range: BufferRange::new(0, vertices.len() as u64).unwrap(),
                            },
                            12,
                            VertexStepMode::Vertex,
                            vec![VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            }],
                        )
                        .unwrap(),
                    ],
                    None,
                )
                .unwrap(),
            );
            draws.push((prepared, image, texture_page, selected));
        }
        // Revisit the RGB and RGBA pipelines after their initial compilation.
        for (iteration, index) in [0, 1, 2, 3, 0, 1, 2, 3].into_iter().enumerate() {
            let (prepared, image, texture_page, selected) = &draws[index];
            let colored = iteration >= 4;
            if colored {
                // Mid-gray RGB565 endpoint, selector zero: sRGB must decode.
                texture_page.prepare_write().unwrap();
                let generation = texture_page.content_generation();
                texture_page
                    .write_preflighted(
                        *selected,
                        &[0x10, 0x84, 0, 0, 0, 0, 0, 0],
                        generation,
                        generation.next().unwrap(),
                    )
                    .unwrap();
            }
            let begin = RenderPassOperation::begin(
                render_pass,
                pass_description.clone(),
                vec![RenderAttachment {
                    image: target,
                    subresources,
                    kind: ImageKind::Color,
                    format: ImageFormat::Rgba8Unorm,
                    samples: SampleCount::One,
                    load: AttachmentLoad::Clear(ClearValue::Color([1.0, 0.0, 1.0, 1.0])),
                    store: AttachmentStore::Store,
                }],
            )
            .unwrap();
            let submission = OperationSubmission::new(
                FrontendSubmissionId::new(serial),
                vec![],
                vec![
                    GpuOperation::new(
                        GpuCommand::RenderPass(begin),
                        [],
                        [],
                        CapabilityRequirements::none(),
                    ),
                    GpuOperation::new(
                        GpuCommand::Draw(
                            DrawOperation::new(
                                prepared.clone(),
                                DrawArguments::NonIndexed {
                                    first_vertex: 0,
                                    vertex_count: 3,
                                    first_instance: 0,
                                    instance_count: 1,
                                },
                            )
                            .unwrap(),
                        ),
                        [nixe_gpu::ResourceAccess::new(
                            nixe_gpu::AccessTarget::Image {
                                image: *image,
                                subresources: ImageSubresourceRange {
                                    layer_count: layers,
                                    ..subresources
                                },
                            },
                            nixe_gpu::AccessScope::new(
                                nixe_gpu::PipelineStages::FRAGMENT_SHADER,
                                nixe_gpu::AccessMode::Read,
                                nixe_gpu::ResourceUsage::SampledImage,
                            )
                            .unwrap(),
                        )],
                        [
                            ResourceDependency::Buffer(vertex_buffer),
                            ResourceDependency::Shader(vertex),
                            ResourceDependency::Shader(fragment),
                            ResourceDependency::Image(*image),
                            ResourceDependency::Sampler(sampler),
                        ],
                        CapabilityRequirements::none(),
                    ),
                    GpuOperation::new(
                        GpuCommand::RenderPass(RenderPassOperation::end(render_pass)),
                        [],
                        [],
                        CapabilityRequirements::none(),
                    ),
                ],
            )
            .unwrap();
            runtime
                .runtime()
                .submit(&creations, &[], &submission)
                .unwrap();
            creations.clear();
            let mut pixels = [0; 64];
            target_backing.range().read(0, &mut pixels).unwrap();
            let expected = if colored {
                if index >= 2 {
                    [59_u8, 57, 59, 255]
                } else {
                    [132, 130, 132, 255]
                }
            } else {
                [
                    0,
                    0,
                    0,
                    if formats[index].has_opaque_bc1_alpha() {
                        255
                    } else {
                        0
                    },
                ]
            };
            for pixel in pixels.chunks_exact(4) {
                for (actual, expected) in pixel.iter().zip(expected) {
                    assert!(
                        actual.abs_diff(expected) <= 1,
                        "format={:?} array={array} colored={colored}: pixel={pixel:?}",
                        formats[index]
                    );
                }
            }
            serial += 1;
        }
    }
}

fn interface(location: ShaderIoLocation, component: u8) -> ShaderInterfaceElement {
    ShaderInterfaceElement::new(location, component, ShaderScalarType::Float32, None).unwrap()
}

fn sampled_fragment(array: bool) -> nixe_gpu::ShaderBackendModule {
    let outputs = (0..4)
        .map(|component| {
            ShaderTextureSampleOutput::new(ShaderRegister::new(2 + u16::from(component)), component)
                .unwrap()
        })
        .collect();
    let coordinates = [ShaderRegister::new(0), ShaderRegister::new(1)];
    let sample = if array {
        ShaderOperation::SampleTexture2DArray {
            outputs,
            coordinates,
            array_index: ShaderRegister::new(6),
            image_binding: 0,
            sampler_binding: 1,
        }
    } else {
        ShaderOperation::SampleTexture2D {
            outputs,
            coordinates,
            image_binding: 0,
            sampler_binding: 1,
        }
    };
    nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            (0..4)
                .map(|component| interface(ShaderIoLocation::Color(0), component))
                .collect(),
            vec![
                ShaderResourceAccess::new(
                    0,
                    if array {
                        ShaderResourceKind::SampledImage2DArray
                    } else {
                        ShaderResourceKind::SampledImage
                    },
                    true,
                    false,
                )
                .unwrap(),
                ShaderResourceAccess::new(1, ShaderResourceKind::Sampler, true, false).unwrap(),
            ],
            vec![
                move_f32(8, 0, 0.75),
                move_f32(16, 1, 0.75),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(6),
                        bits: 1,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Always,
                    sample,
                ),
                store_output(40, 2, ShaderIoLocation::Color(0), 4),
                exit(48),
            ],
        ))
        .unwrap(),
    )
}
