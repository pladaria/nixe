use super::*;
use nixe_gpu::{
    AddressMode, ComponentSwizzle, DescriptorKind, DescriptorTableBinding,
    DescriptorTableDescription, DescriptorTableId, FilterMode, SamplerDescription, SamplerId,
    ShaderResourceAccess, ShaderResourceKind, ShaderTextureSampleOutput,
};

#[test]
fn compressed_sampling_preserves_formats_swizzles_block_layout_and_cached_variants() {
    sampling(None, false, 0.0, None);
}

#[test]
fn clamp_to_border_sampling_preserves_transparent_black_opaque_black_and_white() {
    for border in [[0; 4], [0, 0, 0, 1.0_f32.to_bits()], [1.0_f32.to_bits(); 4]] {
        sampling(Some(border), false, 0.0, None);
    }
}

#[test]
fn compressed_mip_sampling_uploads_selected_levels_and_dirty_array_layers() {
    sampling(None, true, 0.0, None);
}

#[test]
fn sampler_lod_bias_shifts_explicit_mips_before_clamping() {
    sampling(None, true, -1.0, None);
}

#[test]
fn sampler_lod_bias_shifts_implicit_derivatives_before_clamping() {
    sampling(None, true, -2.0, None);
}

#[test]
fn sampled_texel_offsets_preserve_filtering_mips_arrays_and_border_color() {
    // The unshifted coordinate is inside; +7 texels must select border color.
    // An omitted offset fails even for a uniform compressed texture.
    for mipmapped in [false, true] {
        sampling(Some([1.0_f32.to_bits(); 4]), mipmapped, 0.0, Some([7, 0]));
    }
}

fn sampling(
    border: Option<[u32; 4]>,
    mipmapped: bool,
    lod_bias: f32,
    texel_offset: Option<[i32; 2]>,
) {
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
    let packed_supported = presentation
        .device()
        .features()
        .contains(wgpu::Features::FLOAT32_FILTERABLE);
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
    let window_vertex_buffer = BufferId::new(721);
    let window_allocation = GpuAllocationId::new(721);
    let window_vertices = [0.0_f32, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    creations.push(BackendResourceCreateInfo::Allocation {
        id: window_allocation,
        description: allocation_description,
    });
    creations.push(BackendResourceCreateInfo::Buffer {
        id: window_vertex_buffer,
        description: vertex_description,
        view: Some(
            BufferView::new(
                window_vertex_buffer,
                vertex_description,
                0,
                backing(
                    window_allocation,
                    allocation_description,
                    &initialized_page(&window_vertices),
                ),
            )
            .unwrap(),
        ),
    });
    let vertex = ShaderId::new(720);
    let vertex_module = nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            (0..2)
                .map(|component| interface(ShaderIoLocation::Generic(0), component))
                .collect(),
            (0..4)
                .map(|component| interface(ShaderIoLocation::Position, component))
                .collect(),
            vec![],
            (0..4)
                .flat_map(|component| {
                    [
                        move_f32(0, 10, if component == 3 { 1.0 } else { 0.0 }),
                        ShaderInstruction::new(
                            ShaderSourceLocation::new(0),
                            ShaderPredicate::Always,
                            ShaderOperation::StoreOutput {
                                sources: vec![ShaderRegister::new(10)].into_boxed_slice(),
                                location: ShaderIoLocation::Position,
                                first_component: component,
                                scalar_type: ShaderScalarType::Float32,
                            },
                        ),
                    ]
                })
                .chain([
                    load_input(8, 0, 0, ShaderIoLocation::Generic(0), 2),
                    store_output(24, 0, ShaderIoLocation::Position, 2),
                    exit(32),
                ])
                .collect(),
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
    let mut description = SamplerDescription::new(
        FilterMode::Linear,
        FilterMode::Linear,
        FilterMode::Nearest,
        [if border.is_some() {
            AddressMode::ClampToBorder
        } else {
            AddressMode::ClampToEdge
        }; 3],
        0.0,
        if mipmapped { 1.0 } else { 0.0 },
        1.0,
    )
    .unwrap();
    description.border_color = border;
    description.lod_bias = lod_bias;
    creations.push(BackendResourceCreateInfo::Sampler {
        id: sampler,
        description,
    });
    let mut formats = vec![
        (ImageFormat::Bc1RgbUnorm, Swizzle::IDENTITY),
        (ImageFormat::Bc1RgbaUnorm, Swizzle::IDENTITY),
        (ImageFormat::Bc1RgbSrgb, Swizzle::IDENTITY),
        (ImageFormat::Bc1RgbaSrgb, Swizzle::IDENTITY),
        (ImageFormat::Bc3RgbaUnorm, Swizzle::IDENTITY),
        (ImageFormat::Bc3RgbaSrgb, Swizzle::IDENTITY),
        (ImageFormat::Bc7RgbaUnorm, Swizzle::IDENTITY),
        (ImageFormat::Bc7RgbaSrgb, Swizzle::IDENTITY),
        (ImageFormat::Bc4RUnorm, Swizzle::IDENTITY),
        (
            ImageFormat::Bc4RUnorm,
            Swizzle([
                ComponentSwizzle::Red,
                ComponentSwizzle::Red,
                ComponentSwizzle::Red,
                ComponentSwizzle::One,
            ]),
        ),
        (
            ImageFormat::Bc4RUnorm,
            Swizzle([
                ComponentSwizzle::One,
                ComponentSwizzle::Red,
                ComponentSwizzle::Zero,
                ComponentSwizzle::Red,
            ]),
        ),
    ];
    if border.is_none() && lod_bias == 0.0 {
        formats.extend([
            (
                ImageFormat::Bc1RgbUnorm,
                Swizzle([
                    ComponentSwizzle::Red,
                    ComponentSwizzle::Green,
                    ComponentSwizzle::Blue,
                    ComponentSwizzle::Zero,
                ]),
            ),
            (
                ImageFormat::Bc1RgbaUnorm,
                Swizzle([
                    ComponentSwizzle::Alpha,
                    ComponentSwizzle::Green,
                    ComponentSwizzle::Blue,
                    ComponentSwizzle::One,
                ]),
            ),
            (
                ImageFormat::Bc1RgbSrgb,
                Swizzle([
                    ComponentSwizzle::Blue,
                    ComponentSwizzle::Green,
                    ComponentSwizzle::Red,
                    ComponentSwizzle::Zero,
                ]),
            ),
        ]);
    }
    formats.push((
        ImageFormat::Rg8Unorm,
        Swizzle([
            ComponentSwizzle::Red,
            ComponentSwizzle::Red,
            ComponentSwizzle::Red,
            ComponentSwizzle::Green,
        ]),
    ));
    formats.push((
        ImageFormat::R8Unorm,
        Swizzle([
            ComponentSwizzle::One,
            ComponentSwizzle::One,
            ComponentSwizzle::One,
            ComponentSwizzle::Red,
        ]),
    ));
    if packed_supported {
        formats.push((ImageFormat::Rgb565Unorm, Swizzle::IDENTITY));
    }
    formats.push((
        ImageFormat::Bc5RgUnorm,
        Swizzle([
            ComponentSwizzle::Red,
            ComponentSwizzle::Red,
            ComponentSwizzle::Red,
            ComponentSwizzle::Green,
        ]),
    ));
    for format in [ImageFormat::Bc4RSnorm, ImageFormat::Bc5RgSnorm] {
        formats.push((
            format,
            Swizzle([
                ComponentSwizzle::Red,
                ComponentSwizzle::Red,
                ComponentSwizzle::Red,
                if format == ImageFormat::Bc5RgSnorm {
                    ComponentSwizzle::Green
                } else {
                    ComponentSwizzle::One
                },
            ]),
        ));
    }
    formats.extend([
        (ImageFormat::Astc8x8Unorm, Swizzle::IDENTITY),
        (ImageFormat::Astc8x8Srgb, Swizzle::IDENTITY),
        (ImageFormat::Astc5x4Unorm, Swizzle::IDENTITY),
        (ImageFormat::Astc5x4Srgb, Swizzle::IDENTITY),
    ]);
    if border.is_some() {
        formats = vec![(ImageFormat::Bc1RgbaUnorm, Swizzle::IDENTITY)];
    }
    if mipmapped {
        formats = vec![
            (ImageFormat::Bc3RgbaUnorm, Swizzle::IDENTITY),
            (ImageFormat::Bc3RgbaSrgb, Swizzle::IDENTITY),
        ];
    }
    if lod_bias != 0.0 {
        formats = vec![
            (ImageFormat::Bc3RgbaUnorm, Swizzle::IDENTITY),
            (ImageFormat::Bc3RgbaSrgb, Swizzle::IDENTITY),
        ];
    }
    let mut serial = 720;
    for (variant, (array, layers)) in [(false, 1_u16), (true, 1), (true, 2)]
        .into_iter()
        .enumerate()
    {
        let fragment = ShaderId::new(721 + variant as u64);
        creations.push(BackendResourceCreateInfo::Shader {
            id: fragment,
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: sampled_fragment(
                array,
                border.is_some() && texel_offset.is_none(),
                mipmapped,
                false,
                lod_bias == -2.0,
                texel_offset,
            ),
        });
        let signed_fragment = ShaderId::new(9000 + variant as u64);
        if border.is_none() {
            creations.push(BackendResourceCreateInfo::Shader {
                id: signed_fragment,
                description: ShaderDescription {
                    stage: ShaderStage::Fragment,
                },
                module: sampled_fragment(array, false, mipmapped, true, false, texel_offset),
            });
        }
        let mut draws = Vec::new();
        for (index, (format, swizzle)) in formats.iter().copied().enumerate() {
            let id = 730 + index as u64 + variant as u64 * formats.len() as u64;
            let allocation = GpuAllocationId::new(id);
            let image = ImageId::new(id);
            let table = DescriptorTableId::new(id);
            // Two rows of two BC1 blocks. The bottom-right block is selected.
            // In a 64-byte x 8-row GOB, block (1,1) starts at byte 24.
            // Mip zero intentionally stays empty. Level one contains the selected
            // block; using a base-only upload or sampling view produces the wrong pixel.
            let layer_stride = if mipmapped { 1536 } else { 512 };
            let level_offset = if mipmapped { 1024 } else { 0 };
            let mut bytes = vec![0; usize::from(layers) * layer_stride];
            let selected = usize::from(layers - 1) * layer_stride
                + level_offset
                + if format.plane_bytes_per_block(0) == Some(16) {
                    48
                } else {
                    24
                };
            if format.astc_profile().is_some() {
                // Three blocks per row, including clipped right/bottom edges.
                // All unused blocks are green, so a GOB addressing mistake is visible.
                for y in 0..3 {
                    for x in 0..3 {
                        let offset = usize::from(layers - 1) * 512
                            + ((x / 2) * 256 + (x % 2) * 32 + (y / 2) * 64 + (y % 2) * 16);
                        bytes[offset..offset + 16]
                            .copy_from_slice(&astc_constant([0, 255, 0, 255]));
                    }
                }
                bytes[selected..selected + 16].copy_from_slice(&astc_constant([128, 64, 32, 255]));
            } else if format == ImageFormat::R8Unorm {
                for offset in [149, 150, 197, 198] {
                    bytes[usize::from(layers - 1) * 512 + offset] = 73;
                }
            } else if format == ImageFormat::Rg8Unorm {
                for offset in [154, 156, 202, 204] {
                    let offset = usize::from(layers - 1) * 512 + offset;
                    bytes[offset..offset + 2].copy_from_slice(&[73, 191]);
                }
            } else if format == ImageFormat::Rgb565Unorm {
                // UV .75 samples texels (5,5), (6,5), (5,6), (6,6).
                // Distinct endpoint channels expose RGB/BGR swaps and filtering.
                for offset in [154, 156, 202, 204] {
                    let offset = usize::from(layers - 1) * 512 + offset;
                    bytes[offset..offset + 2].copy_from_slice(&0xf800_u16.to_le_bytes());
                }
            } else if matches!(format, ImageFormat::Bc7RgbaUnorm | ImageFormat::Bc7RgbaSrgb) {
                bytes[selected..selected + 16].copy_from_slice(&bc7_constant([128, 64, 32, 254]));
            } else if matches!(format, ImageFormat::Bc3RgbaUnorm | ImageFormat::Bc3RgbaSrgb) {
                // Use an RGB endpoint: native BC interpolation precision can
                // differ from the software decoder. Alpha still interpolates.
                bytes[selected..selected + 16].copy_from_slice(&[
                    200, 60, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49, 0, 0, 255, 255, 0, 0, 0, 0,
                ]);
            } else if format == ImageFormat::Bc5RgSnorm {
                bytes[selected..selected + 16]
                    .copy_from_slice(&[128, 127, 0, 0, 0, 0, 0, 0, 127, 128, 0, 0, 0, 0, 0, 0]);
            } else if format == ImageFormat::Bc4RSnorm {
                bytes[selected..selected + 8].copy_from_slice(&[128, 127, 0, 0, 0, 0, 0, 0]);
            } else if format == ImageFormat::Bc5RgUnorm {
                bytes[selected..selected + 16]
                    .copy_from_slice(&[73, 0, 0, 0, 0, 0, 0, 0, 191, 0, 0, 0, 0, 0, 0, 0]);
            } else if format == ImageFormat::Bc4RUnorm {
                // Selector 2 in every texel; the eight-value palette yields 180.
                bytes[selected..selected + 8]
                    .copy_from_slice(&[200, 60, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49]);
            } else {
                bytes[selected..selected + 8]
                    .copy_from_slice(&[0, 0, 255, 255, 255, 255, 255, 255]);
            }
            let allocation_description =
                GpuAllocationDescription::new(bytes.len() as u64, 4).unwrap();
            let texture_page = initialized_page(&bytes);
            let image_backing = backing(allocation, allocation_description, &texture_page);
            let description = ImageDescription::new(
                ImageDimension::Two,
                ImageExtent::new(
                    if format.astc_profile().is_some()
                        || format == ImageFormat::Bc5RgUnorm
                        || (array && format == ImageFormat::Bc5RgSnorm)
                        || (array
                            && matches!(
                                format,
                                ImageFormat::Bc3RgbaUnorm
                                    | ImageFormat::Bc3RgbaSrgb
                                    | ImageFormat::Bc7RgbaUnorm
                                    | ImageFormat::Bc7RgbaSrgb
                            ))
                    {
                        (format.block_extent()[0] * 2 + 1) * if mipmapped { 2 } else { 1 }
                    } else {
                        if mipmapped { 16 } else { 8 }
                    },
                    if format.astc_profile().is_some()
                        || format == ImageFormat::Bc5RgUnorm
                        || (array && format == ImageFormat::Bc5RgSnorm)
                        || (array
                            && matches!(
                                format,
                                ImageFormat::Bc3RgbaUnorm
                                    | ImageFormat::Bc3RgbaSrgb
                                    | ImageFormat::Bc7RgbaUnorm
                                    | ImageFormat::Bc7RgbaSrgb
                            ))
                    {
                        (format.block_extent()[1] * 2 + 1) * if mipmapped { 2 } else { 1 }
                    } else {
                        if mipmapped { 16 } else { 8 }
                    },
                    1,
                )
                .unwrap(),
                format,
                ImageKind::Color,
                if mipmapped { 2 } else { 1 },
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
                        swizzle,
                        if mipmapped {
                            (0..layers)
                                .flat_map(|layer| {
                                    let image_backing = &image_backing;
                                    [(0, 0, 1024), (1, 1024, 512)].into_iter().map(
                                        move |(mip_level, offset, size)| {
                                            let offset =
                                                u64::from(layer) * layer_stride as u64 + offset;
                                            let range = image_backing
                                                .range()
                                                .snapshot_subrange(offset, size)
                                                .unwrap();
                                            (
                                                ImageSubresourceRange {
                                                    mip_level,
                                                    base_layer: layer,
                                                    layer_count: 1,
                                                    ..subresources
                                                },
                                                ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                                                    block_width_log2: 0,
                                                    block_height_log2: 0,
                                                    block_depth_log2: 0,
                                                    layer_stride: size,
                                                }),
                                                BackingView::new(
                                                    allocation,
                                                    allocation_description,
                                                    offset,
                                                    range,
                                                )
                                                .unwrap(),
                                            )
                                        },
                                    )
                                })
                                .collect()
                        } else {
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
                            )]
                        },
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
            let window_coordinates = index % 2 != 0;
            let draw_vertex_buffer = if window_coordinates {
                window_vertex_buffer
            } else {
                vertex_buffer
            };
            let mut prepared = PreparedDraw::new(
                nixe_gpu::PreparedStateId::new(1, 1),
                pipeline,
                render_pass,
                PrimitiveTopology::Triangles,
                vec![table],
                vec![
                    VertexBufferLayout::new(
                        BufferRegion {
                            buffer: draw_vertex_buffer,
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
            .unwrap();
            if window_coordinates {
                prepared.edit_state().vertex_position_transform =
                    Some(nixe_gpu::VertexPositionTransform::framebuffer(4, 4).unwrap());
            }
            let prepared = Arc::new(prepared);
            draws.push((prepared, image, texture_page, selected, draw_vertex_buffer));
        }
        // Revisit the RGB and RGBA pipelines after their initial compilation.
        for (iteration, index) in (0..formats.len())
            .cycle()
            .take(formats.len() * 3)
            .enumerate()
        {
            let (prepared, image, texture_page, selected, draw_vertex_buffer) = &draws[index];
            let fragment = if matches!(
                formats[index].0,
                ImageFormat::Bc4RSnorm | ImageFormat::Bc5RgSnorm
            ) {
                signed_fragment
            } else {
                fragment
            };
            let colored = iteration >= formats.len() * 2;
            if colored {
                // Mid-gray RGB565 endpoint, selector zero: sRGB must decode.
                texture_page.prepare_write().unwrap();
                let generation = texture_page.content_generation();
                if formats[index].0.astc_profile().is_some() {
                    texture_page
                        .write_preflighted(
                            *selected,
                            &astc_constant([0, 0, 255, 255]),
                            generation,
                            generation.next().unwrap(),
                        )
                        .unwrap();
                } else if matches!(
                    formats[index].0,
                    ImageFormat::Bc7RgbaUnorm | ImageFormat::Bc7RgbaSrgb
                ) {
                    texture_page
                        .write_preflighted(
                            *selected,
                            &bc7_constant([192, 96, 16, 128]),
                            generation,
                            generation.next().unwrap(),
                        )
                        .unwrap();
                } else if matches!(
                    formats[index].0,
                    ImageFormat::Bc3RgbaUnorm | ImageFormat::Bc3RgbaSrgb
                ) {
                    texture_page
                        .write_preflighted(
                            *selected,
                            &[90, 180, 0, 0, 0, 0, 0, 0, 0x10, 0x84, 0, 0, 0, 0, 0, 0],
                            generation,
                            generation.next().unwrap(),
                        )
                        .unwrap();
                } else if matches!(
                    formats[index].0,
                    ImageFormat::Bc4RSnorm | ImageFormat::Bc5RgSnorm
                ) {
                    let block = [192, 127, 0, 0, 0, 0, 0, 0, 127, 128, 0, 0, 0, 0, 0, 0];
                    let count = if formats[index].0 == ImageFormat::Bc5RgSnorm {
                        16
                    } else {
                        8
                    };
                    texture_page
                        .write_preflighted(
                            *selected,
                            &block[..count],
                            generation,
                            generation.next().unwrap(),
                        )
                        .unwrap();
                } else if formats[index].0 == ImageFormat::Bc5RgUnorm {
                    texture_page
                        .write_preflighted(
                            *selected,
                            &[149, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0],
                            generation,
                            generation.next().unwrap(),
                        )
                        .unwrap();
                } else if formats[index].0 == ImageFormat::R8Unorm {
                    for offset in [149, 150, 197, 198] {
                        let generation = texture_page.content_generation();
                        texture_page
                            .write_preflighted(
                                usize::from(layers - 1) * 512 + offset,
                                &[149],
                                generation,
                                generation.next().unwrap(),
                            )
                            .unwrap();
                    }
                } else if matches!(
                    formats[index].0,
                    ImageFormat::Rgb565Unorm | ImageFormat::Rg8Unorm
                ) {
                    for (offset, value) in [154, 156, 202, 204].into_iter().zip([
                        (1_u16 << 11) | (3 << 5) | 5,
                        (2 << 11) | (3 << 5) | 6,
                        (1 << 11) | (4 << 5) | 5,
                        (2 << 11) | (4 << 5) | 6,
                    ]) {
                        let generation = texture_page.content_generation();
                        texture_page
                            .write_preflighted(
                                usize::from(layers - 1) * 512 + offset,
                                &if formats[index].0 == ImageFormat::Rg8Unorm {
                                    u16::from_le_bytes([149, 37])
                                } else {
                                    value
                                }
                                .to_le_bytes(),
                                generation,
                                generation.next().unwrap(),
                            )
                            .unwrap();
                    }
                } else {
                    texture_page
                        .write_preflighted(
                            *selected,
                            if formats[index].0 == ImageFormat::Bc4RUnorm {
                                // Six-value palette, selector 6 is exactly zero.
                                &[20, 220, 0xb6, 0x6d, 0xdb, 0xb6, 0x6d, 0xdb]
                            } else {
                                &[0x10, 0x84, 0, 0, 0, 0, 0, 0]
                            },
                            generation,
                            generation.next().unwrap(),
                        )
                        .unwrap();
                }
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
                                    mip_level: if mipmapped { 1 } else { 0 },
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
                            ResourceDependency::Buffer(*draw_vertex_buffer),
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
            let expected = if lod_bias != 0.0 {
                [0; 4]
            } else if let Some(border) = border {
                border.map(|bits| (f32::from_bits(bits) * 255.0) as u8)
            } else if let Some(srgb) = formats[index].0.astc_profile() {
                if colored {
                    [0, 0, 255, 255]
                } else if srgb {
                    [55, 13, 4, 255]
                } else {
                    [128, 64, 32, 255]
                }
            } else if matches!(
                formats[index].0,
                ImageFormat::Bc7RgbaUnorm | ImageFormat::Bc7RgbaSrgb
            ) {
                match (colored, formats[index].0 == ImageFormat::Bc7RgbaSrgb) {
                    (false, false) => [128, 64, 32, 254],
                    (false, true) => [55, 13, 4, 254],
                    (true, false) => [192, 96, 16, 128],
                    (true, true) => [134, 30, 1, 128],
                }
            } else if matches!(
                formats[index].0,
                ImageFormat::Bc3RgbaUnorm | ImageFormat::Bc3RgbaSrgb
            ) {
                if colored {
                    if formats[index].0 == ImageFormat::Bc3RgbaSrgb {
                        [59, 57, 59, 90]
                    } else {
                        [132, 130, 132, 90]
                    }
                } else {
                    [0, 0, 0, 180]
                }
            } else if matches!(
                formats[index].0,
                ImageFormat::Bc4RSnorm | ImageFormat::Bc5RgSnorm
            ) {
                if colored {
                    [129, 129, 129, 255]
                } else {
                    [255; 4]
                }
            } else if formats[index].0 == ImageFormat::R8Unorm {
                [255, 255, 255, if colored { 149 } else { 73 }]
            } else if matches!(
                formats[index].0,
                ImageFormat::Rg8Unorm | ImageFormat::Bc5RgUnorm
            ) {
                if colored {
                    [149, 149, 149, 37]
                } else {
                    [73, 73, 73, 191]
                }
            } else if formats[index].0 == ImageFormat::Rgb565Unorm {
                if colored {
                    [12, 14, 45, 255]
                } else {
                    [255, 0, 0, 255]
                }
            } else if formats[index].0 == ImageFormat::Bc4RUnorm {
                let value = if colored { 0 } else { 180 };
                formats[index].1.0.map(|channel| match channel {
                    ComponentSwizzle::Red => value,
                    ComponentSwizzle::Green | ComponentSwizzle::Blue | ComponentSwizzle::Zero => 0,
                    ComponentSwizzle::Alpha | ComponentSwizzle::One => 255,
                })
            } else {
                let texel = if colored {
                    if matches!(
                        formats[index].0,
                        ImageFormat::Bc1RgbSrgb | ImageFormat::Bc1RgbaSrgb
                    ) {
                        [59, 57, 59, 255]
                    } else {
                        [132, 130, 132, 255]
                    }
                } else {
                    [
                        0,
                        0,
                        0,
                        if formats[index].0.has_opaque_bc1_alpha() {
                            255
                        } else {
                            0
                        },
                    ]
                };
                formats[index].1.0.map(|channel| match channel {
                    ComponentSwizzle::Red => texel[0],
                    ComponentSwizzle::Green => texel[1],
                    ComponentSwizzle::Blue => texel[2],
                    ComponentSwizzle::Alpha => texel[3],
                    ComponentSwizzle::Zero => 0,
                    ComponentSwizzle::One => 255,
                })
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

fn sampled_fragment(
    array: bool,
    outside: bool,
    mipmapped: bool,
    signed: bool,
    implicit_lod: bool,
    texel_offset: Option<[i32; 2]>,
) -> nixe_gpu::ShaderBackendModule {
    let outputs = (0..4)
        .map(|component| {
            ShaderTextureSampleOutput::new(ShaderRegister::new(2 + u16::from(component)), component)
                .unwrap()
        })
        .collect();
    let coordinates = [ShaderRegister::new(0), ShaderRegister::new(1)];
    let sample = if array {
        ShaderOperation::SampleTexture2DArray {
            offset: texel_offset,
            outputs,
            coordinates,
            array_index: ShaderRegister::new(6),
            image_binding: 0,
            lod: (mipmapped && !implicit_lod).then_some(ShaderRegister::new(7)),
            sampler_binding: 1,
        }
    } else {
        ShaderOperation::SampleTexture2D {
            offset: texel_offset,
            outputs,
            coordinates,
            image_binding: 0,
            lod: (mipmapped && !implicit_lod).then_some(ShaderRegister::new(7)),
            sampler_binding: 1,
        }
    };
    let source_inputs = if implicit_lod {
        vec![
            interface(ShaderIoLocation::Position, 0),
            interface(ShaderIoLocation::Position, 1),
        ]
    } else {
        vec![]
    };
    nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            source_inputs,
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
                move_f32(8, 0, if outside { 1.25 } else { 0.75 }),
                move_f32(16, 1, 0.75),
                move_f32(20, 7, 1.0),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(6),
                        bits: 1,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
            ]
            .into_iter()
            .chain(if implicit_lod {
                vec![
                    load_input(25, 0, 0, ShaderIoLocation::Position, 2),
                    move_f32(26, 8, 0.25),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(27),
                        ShaderPredicate::Always,
                        ShaderOperation::Multiply32 {
                            destination: ShaderRegister::new(0),
                            left: ShaderRegister::new(0),
                            right: ShaderRegister::new(8),
                            scalar_type: ShaderScalarType::Float32,
                            float_control: nixe_gpu::ShaderFloatControl::PRECISE,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(28),
                        ShaderPredicate::Always,
                        ShaderOperation::Multiply32 {
                            destination: ShaderRegister::new(1),
                            left: ShaderRegister::new(1),
                            right: ShaderRegister::new(8),
                            scalar_type: ShaderScalarType::Float32,
                            float_control: nixe_gpu::ShaderFloatControl::PRECISE,
                        },
                    ),
                ]
            } else {
                vec![]
            })
            .chain([ShaderInstruction::new(
                ShaderSourceLocation::new(32),
                ShaderPredicate::Always,
                sample,
            )])
            .chain((2..5).filter(|_| signed).map(|reg| {
                ShaderInstruction::new(
                    ShaderSourceLocation::new(40 + (reg - 2) * 8),
                    ShaderPredicate::Always,
                    ShaderOperation::FloatNegate32 {
                        destination: ShaderRegister::new(reg as u16),
                        source: ShaderRegister::new(reg as u16),
                    },
                )
            }))
            .chain([store_output(64, 2, ShaderIoLocation::Color(0), 4), exit(72)])
            .collect(),
        ))
        .unwrap(),
    )
}

fn astc_constant(rgba: [u8; 4]) -> [u8; 16] {
    // LDR void-extent block: constant UNORM16 RGBA, unrestricted extent.
    let mut block = [0xff; 16];
    block[0] = 0xfc;
    block[1] = 0xfd;
    for (channel, value) in rgba.into_iter().enumerate() {
        block[8 + channel * 2..10 + channel * 2]
            .copy_from_slice(&(u16::from(value) * 257).to_le_bytes());
    }
    block
}

// One-subset mode 6: seven bits per RGBA endpoint and one shared parity bit.
// https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference#mode-6
fn bc7_constant(rgba: [u8; 4]) -> [u8; 16] {
    let parity = rgba[0] & 1;
    assert!(rgba.iter().all(|v| v & 1 == parity));
    let mut bits = 1_u128 << 6;
    let mut position = 7;
    for channel in rgba {
        for _ in 0..2 {
            bits |= u128::from(channel >> 1) << position;
            position += 7;
        }
    }
    bits |= u128::from(parity) << 63;
    bits |= u128::from(parity) << 64;
    bits.to_le_bytes()
}
