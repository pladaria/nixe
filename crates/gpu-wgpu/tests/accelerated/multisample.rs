//! Pixel oracles for resident MSAA, per-sample depth, and ordered resolves.
use super::*;
use nixe_gpu::{
    DepthCompareOperation, DepthState, ImageOrigin, ImageRegion, IndexType, ResolveOperation,
};

const SIZE: u32 = 32;
const SUBRESOURCE: ImageSubresourceRange = ImageSubresourceRange {
    plane: 0,
    mip_level: 0,
    base_layer: 0,
    layer_count: 1,
};

fn operation(command: GpuCommand) -> GpuOperation {
    GpuOperation::new(command, [], [], CapabilityRequirements::none())
}

fn region(image: ImageId) -> ImageRegion {
    ImageRegion {
        image,
        subresources: SUBRESOURCE,
        origin: ImageOrigin { x: 0, y: 0, z: 0 },
        extent: ImageExtent::new(SIZE, SIZE, 1).unwrap(),
    }
}

fn pass_description(
    samples: SampleCount,
    color_format: ImageFormat,
    depth_format: ImageFormat,
) -> RenderPassDescription {
    RenderPassDescription::new(vec![
        RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: color_format,
            samples,
        },
        RenderPassAttachmentDescription {
            kind: ImageKind::DepthStencil,
            format: depth_format,
            samples,
        },
    ])
    .unwrap()
}

#[test]
fn four_sample_color_depth_resolve_and_single_sample_pipeline_reuse() {
    for depth_format in [
        ImageFormat::Depth32Float,
        ImageFormat::Depth24UnormStencil8Uint,
    ] {
        for color_format in [
            ImageFormat::Rgba8Unorm,
            ImageFormat::Bgra8Unorm,
            ImageFormat::Rgba8Srgb,
            ImageFormat::Bgra8Srgb,
        ] {
            color_depth_resolve(depth_format, color_format, None);
        }
    }
}

#[test]
fn indexed_u16_u32_msaa_fetch_preserves_first_index_and_signed_base_vertex() {
    for kind in [IndexType::Uint16, IndexType::Uint32] {
        color_depth_resolve(
            ImageFormat::Depth32Float,
            ImageFormat::Rgba8Srgb,
            Some(kind),
        );
    }
}

fn color_depth_resolve(
    depth_format: ImageFormat,
    color_format: ImageFormat,
    index_type: Option<IndexType>,
) {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x44),
        NonCpuDeviceId::new(0x44),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let presentation = initialized.presentation_context();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let page = initialized_page(&vec![0; (SIZE * SIZE * 4) as usize]);
    let (mut creations, output_backing, output, _) =
        backed_color_image(color_format, SIZE, SIZE, &[page]);
    for (index, samples) in [SampleCount::One, SampleCount::Four]
        .into_iter()
        .enumerate()
    {
        for (offset, format, kind) in [
            (0, color_format, ImageKind::Color),
            (1, depth_format, ImageKind::DepthStencil),
        ] {
            if index == 0 && offset == 0 {
                continue;
            }
            creations.push(BackendResourceCreateInfo::Image {
                id: ImageId::new(10 + index as u64 * 2 + offset),
                description: ImageDescription::new(
                    ImageDimension::Two,
                    region(output).extent,
                    format,
                    kind,
                    1,
                    1,
                    samples,
                )
                .unwrap(),
                view: None,
            });
        }
        creations.push(BackendResourceCreateInfo::RenderPass {
            id: RenderPassId::new(10 + index as u64),
            description: pass_description(samples, color_format, depth_format),
        });
    }

    // Two coincident triangles: near red then far green. The far draw must
    // fail depth independently at every covered sample, not overwrite the red.
    let mut vertices = Vec::new();
    for (z, color) in [(0.25, [0.25, 0.0, 0.0]), (0.75, [0.0, 1.0, 0.0])] {
        for [x, y] in [[-0.5_f32, -0.5], [0.5, -0.5], [0.0, 0.5]] {
            for component in [x, y, z, 1.0, color[0], color[1], color[2]] {
                vertices.extend_from_slice(&component.to_le_bytes());
            }
        }
    }
    let vertex_page = initialized_page(&vertices);
    let allocation = GpuAllocationId::new(3);
    let allocation_description = GpuAllocationDescription::new(vertices.len() as u64, 4).unwrap();
    let buffer = BufferId::new(3);
    let buffer_description = BufferDescription::new(vertices.len() as u64).unwrap();
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
            module: interpolated_vertex_module(ShaderInterpolation::Perspective, false),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(2),
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: interpolated_fragment_module(ShaderInterpolation::Perspective, false, 1.0),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(1),
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        },
    ]);
    let index_buffer = index_type.map(|kind| {
        // Skip the poison prefix, permute each triangle, and subtract one via
        // baseVertex. Reading sequential vertices or forgetting either offset
        // cannot produce the same color/depth/coverage oracle.
        let values = [65535u32, 3, 1, 2, 6, 4, 5];
        let bytes: Vec<_> = values
            .into_iter()
            .flat_map(|value| match kind {
                IndexType::Uint16 => (value as u16).to_le_bytes().to_vec(),
                IndexType::Uint32 => value.to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect();
        let index_page = initialized_page(&bytes);
        let allocation = GpuAllocationId::new(4);
        let allocation_description = GpuAllocationDescription::new(bytes.len() as u64, 4).unwrap();
        let buffer = BufferId::new(4);
        let description = BufferDescription::new(bytes.len() as u64).unwrap();
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
                    backing(allocation, allocation_description, &index_page),
                )
                .unwrap(),
            ),
        });
        (
            BufferRegion {
                buffer,
                range: BufferRange::new(0, bytes.len() as u64).unwrap(),
            },
            kind,
        )
    });
    let prepared = [0, 1].map(|index| {
        Arc::new(
            PreparedDraw::new(
                PipelineId::new(1),
                RenderPassId::new(10 + index),
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
                index_buffer,
            )
            .unwrap()
            .with_viewport_transform(
                ViewportTransform::new([16.0, -16.0, 0.5], [16.0, 16.0, 0.5], [0.0, 1.0]).unwrap(),
            )
            .with_depth_state(DepthState::new(true, true, DepthCompareOperation::Less)),
        )
    });
    let mut serial = 1;
    let mut submit = |creations: &[BackendResourceCreateInfo], operations| {
        runtime
            .runtime()
            .submit(
                creations,
                &[],
                &OperationSubmission::new(FrontendSubmissionId::new(serial), vec![], operations)
                    .unwrap(),
            )
            .unwrap();
        serial += 1;
    };
    let mut previous = [None, None];
    // Exercise shared shader/pipeline handles across 1x/4x cache entries, then
    // reuse the exact same prepared draws on subsequent frames.
    for index in [0, 1, 0, 1] {
        let samples = [SampleCount::One, SampleCount::Four][index];
        let color = if index == 0 { output } else { ImageId::new(12) };
        let depth = ImageId::new(11 + index as u64 * 2);
        let pass = RenderPassId::new(10 + index as u64);
        let mut operations = vec![];
        if samples == SampleCount::Four {
            // The guest also clears images outside draw render passes.
            for (image, kind, format, value) in [
                (
                    color,
                    ImageKind::Color,
                    color_format,
                    ClearValue::Color([0.0; 4]),
                ),
                (
                    depth,
                    ImageKind::DepthStencil,
                    depth_format,
                    ClearValue::Depth(1.0),
                ),
            ] {
                operations.push(operation(GpuCommand::Clear(
                    ClearOperation::image(region(image), kind, format, samples, value).unwrap(),
                )));
            }
        }
        let load = |value| {
            if samples == SampleCount::Four {
                AttachmentLoad::Load
            } else {
                AttachmentLoad::Clear(value)
            }
        };
        operations.push(operation(GpuCommand::RenderPass(
            RenderPassOperation::begin(
                pass,
                pass_description(samples, color_format, depth_format),
                vec![
                    RenderAttachment {
                        image: color,
                        subresources: SUBRESOURCE,
                        kind: ImageKind::Color,
                        format: color_format,
                        samples,
                        load: load(ClearValue::Color([0.0; 4])),
                        store: AttachmentStore::Store,
                    },
                    RenderAttachment {
                        image: depth,
                        subresources: SUBRESOURCE,
                        kind: ImageKind::DepthStencil,
                        format: depth_format,
                        samples,
                        load: load(ClearValue::Depth(1.0)),
                        store: AttachmentStore::Store,
                    },
                ],
            )
            .unwrap(),
        )));
        for first_vertex in [0, 3] {
            operations.push(GpuOperation::new(
                GpuCommand::Draw(
                    DrawOperation::new(
                        Arc::clone(&prepared[index]),
                        if index_type.is_some() {
                            DrawArguments::Indexed {
                                first_index: first_vertex + 1,
                                index_count: 3,
                                vertex_offset: -1,
                                first_instance: 0,
                                instance_count: 1,
                            }
                        } else {
                            DrawArguments::NonIndexed {
                                first_vertex,
                                vertex_count: 3,
                                first_instance: 0,
                                instance_count: 1,
                            }
                        },
                    )
                    .unwrap(),
                ),
                [],
                [
                    ResourceDependency::Shader(ShaderId::new(1)),
                    ResourceDependency::Shader(ShaderId::new(2)),
                ],
                CapabilityRequirements::none(),
            ));
        }
        operations.push(operation(GpuCommand::RenderPass(RenderPassOperation::end(
            pass,
        ))));
        submit(&creations, operations);
        creations.clear();
        let transfer = if samples == SampleCount::Four {
            Some(GpuCommand::Resolve(
                ResolveOperation::new(region(color), region(output), color_format, samples)
                    .unwrap(),
            ))
        } else {
            None
        };
        for repetition in 0..2 {
            // A separate submission must resolve the stored samples; a second
            // resolve must retain the same result rather than consume them.
            if let Some(transfer) = &transfer {
                submit(&[], vec![operation(transfer.clone())]);
            }
            // Inspect device-authored presentation before CPU visibility turns
            // the backing clean (a later CPU-owned import normalizes to RGBA).
            let presented = if repetition == 0 {
                let resident = runtime
                    .runtime()
                    .acquire_presentable_image(PresentationImageRequest {
                        cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(
                            output_backing.range(),
                        )
                        .unwrap(),
                        backing: output_backing.clone(),
                        width: SIZE,
                        height: SIZE,
                        format: match color_format {
                            ImageFormat::Bgra8Unorm | ImageFormat::Bgra8Srgb => {
                                PresentationImageFormat::Bgra8
                            }
                            _ => PresentationImageFormat::Rgba8,
                        },
                        layout: ImageMemoryLayout::PitchLinear {
                            row_pitch: u64::from(SIZE * 4),
                            layer_stride: u64::from(SIZE * SIZE * 4),
                        },
                        row_pitch: SIZE * 4,
                    })
                    .unwrap();
                assert_eq!(resident.description().format(), color_format);
                // Match scanout: an UNORM view exposes display-encoded bytes with
                // no implicit sRGB decode and no replacement/imported texture.
                let texture = resident_texture(&resident).unwrap();
                let _scanout = texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(texture.format().remove_srgb_suffix()),
                    ..Default::default()
                });
                Some(read_presented_rgba(&presentation, &resident))
            } else {
                None
            };
            let mut pixels = vec![0; (SIZE * SIZE * 4) as usize];
            output_backing.range().read(0, &mut pixels).unwrap();
            if let Some(presented) = presented {
                assert_eq!(
                    presented, pixels,
                    "scanout/readback must preserve stored bytes"
                );
            }
            let interior = ((16 * SIZE + 16) * 4) as usize;
            let (red, blue) = match color_format {
                ImageFormat::Bgra8Unorm | ImageFormat::Bgra8Srgb => (2, 0),
                _ => (0, 2),
            };
            let srgb = matches!(
                color_format,
                ImageFormat::Rgba8Srgb | ImageFormat::Bgra8Srgb
            );
            let encode = |linear: f64| -> u8 {
                let encoded = if !srgb {
                    linear
                } else if linear <= 0.0031308 {
                    12.92 * linear
                } else {
                    1.055 * linear.powf(1.0 / 2.4) - 0.055
                };
                (encoded * 255.0).round() as u8
            };
            assert!(pixels[interior + red].abs_diff(encode(0.25)) <= 1);
            assert_eq!(pixels[interior + 3], 255);
            assert_eq!(&pixels[..4], &[0; 4]);
            let mut partial = 0;
            for pixel in pixels.chunks_exact(4) {
                // RGB is encoded after linear-light averaging, whereas alpha
                // remains linear. Mid-tone interiors also catch double/missing
                // sRGB conversion independently of edge sample coverage.
                let coverage = (f64::from(pixel[3]) / 255.0 * 4.0).round() / 4.0;
                assert!(
                    pixel[red].abs_diff(encode(0.25 * coverage)) <= 2,
                    "format={color_format:?} samples={samples:?} pixel={pixel:?}"
                );
                assert_eq!(pixel[1], 0, "far green must fail per-sample depth");
                assert_eq!(pixel[blue], 0);
                if samples == SampleCount::One {
                    assert!(matches!(pixel[3], 0 | 255));
                } else {
                    assert!(
                        [0u8, 64, 128, 191, 255]
                            .iter()
                            .any(|alpha| alpha.abs_diff(pixel[3]) <= 1)
                    );
                    partial += usize::from(pixel[3] > 0 && pixel[3] < 255);
                }
            }
            if samples == SampleCount::Four {
                assert!(partial >= 16, "no antialiased triangle edges");
            }
            if let Some(previous) = &previous[index] {
                assert_eq!(&pixels, previous);
            }
            previous[index] = Some(pixels);
        }
        if let Some(transfer) = transfer {
            submit(
                &[],
                vec![
                    operation(GpuCommand::Clear(
                        ClearOperation::image(
                            region(color),
                            ImageKind::Color,
                            color_format,
                            samples,
                            ClearValue::Color([0.0, 0.0, 0.25, 0.5]),
                        )
                        .unwrap(),
                    )),
                    operation(transfer),
                ],
            );
            let mut pixels = vec![0; (SIZE * SIZE * 4) as usize];
            output_backing.range().read(0, &mut pixels).unwrap();
            let expected = match color_format {
                ImageFormat::Rgba8Unorm => [0, 0, 64, 128],
                ImageFormat::Bgra8Unorm => [64, 0, 0, 128],
                ImageFormat::Rgba8Srgb => [0, 0, 137, 128],
                ImageFormat::Bgra8Srgb => [137, 0, 0, 128],
                _ => unreachable!(),
            };
            for pixel in pixels.chunks_exact(4) {
                assert!(
                    pixel
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| actual.abs_diff(expected) <= 1),
                    "a later clear must encode RGB, preserve linear alpha and replace every sample: {color_format:?} {pixel:?}"
                );
            }
        }
    }
}
