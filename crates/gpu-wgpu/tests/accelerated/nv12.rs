use super::*;
use nixe_gpu::{Nv12ColorTransform, Nv12Conversion};

fn image(
    id: u64,
    format: ImageFormat,
    width: u32,
    height: u32,
    pitch: u64,
    data: &[u8],
) -> (
    Vec<BackendResourceCreateInfo>,
    CanonicalAllocation,
    nixe_gpu::ImageRegion,
) {
    let memory = CanonicalAllocation::zeroed(data.len(), 16).unwrap();
    memory.write(0, data).unwrap();
    let allocation = GpuAllocationId::new(id);
    let allocation_description = GpuAllocationDescription::new(data.len() as u64, 1).unwrap();
    let backing = BackingView::new(
        allocation,
        allocation_description,
        0,
        memory.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
    )
    .unwrap();
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(width, height, 1).unwrap(),
        format,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let subresources = ImageSubresourceRange {
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
        plane: 0,
    };
    let region = nixe_gpu::ImageRegion {
        image: ImageId::new(id),
        subresources,
        origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
        extent: description.extent(),
    };
    (
        vec![
            BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            },
            BackendResourceCreateInfo::Image {
                id: region.image,
                description,
                view: Some(
                    ImageView::new(
                        region.image,
                        description,
                        Swizzle::IDENTITY,
                        vec![(
                            subresources,
                            ImageMemoryLayout::PitchLinear {
                                row_pitch: pitch,
                                layer_stride: data.len() as u64,
                            },
                            backing,
                        )],
                    )
                    .unwrap(),
                ),
            },
        ],
        memory,
        region,
    )
}

#[test]
fn integer_nv12_conversion_is_exact_resident_and_preserves_canonical_padding() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(845),
        NonCpuDeviceId::new(845),
        Default::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let luma: Vec<u8> = (0..16 * 9).map(|i| (i * 47 + 13) as u8).collect();
    let uv: Vec<u8> = (0..16 * 5).map(|i| (i * 71 + 23) as u8).collect();
    let (mut creations, luma_memory, luma_region) =
        image(801, ImageFormat::R8Unorm, 12, 9, 16, &luma);
    let (chroma_creations, chroma_memory, chroma_region) =
        image(802, ImageFormat::Rg8Unorm, 6, 5, 16, &uv);
    creations.extend(chroma_creations);
    let mut serial = 1;
    for format in [ImageFormat::Rgba8Unorm, ImageFormat::Bgra8Unorm] {
        let (output_creations, output_memory, output_region) =
            image(810 + serial, format, 7, 5, 40, &[0xa5; 256]);
        let sampled_format = if format == ImageFormat::Rgba8Unorm {
            ImageFormat::Rgba8Srgb
        } else {
            ImageFormat::Bgra8Srgb
        };
        let (copy_creations, copy_memory, copy_region) =
            image(820 + serial, sampled_format, 7, 5, 40, &[0x7b; 256]);
        let mut all = std::mem::take(&mut creations);
        all.extend(output_creations);
        all.extend(copy_creations);
        for (iteration, (shift, enabled, clamp, origin, coefficients)) in [
            (
                9,
                true,
                [0, 1023],
                [1, 3],
                [
                    [152618, 0, 209194, -228942],
                    [152618, -51348, -106556, 139237],
                    [152618, 264403, 0, -284313],
                ],
            ),
            (
                0,
                true,
                [75, 900],
                [3, 1],
                [
                    [524287, -524288, 524287, -524288],
                    [-524288, -524288, 524287, 524287],
                    [524287, 524287, -524288, 12345],
                ],
            ),
            (
                1,
                true,
                [0, 1023],
                [0, 0],
                [
                    [-131071, 65537, 8191, 777],
                    [123, 321, -456, 1023],
                    [999, -777, 555, -333],
                ],
            ),
            (
                15,
                true,
                [1, 1022],
                [1, 1],
                [[524287, 524287, 524287, 524287]; 3],
            ),
            (9, false, [0, 1023], [3, 3], [[0; 4]; 3]),
            (9, false, [0, 1023], [3, 3], [[0; 4]; 3]),
        ]
        .into_iter()
        .enumerate()
        {
            // Reuse the same planes and destination after an actual CPU write.
            let y_data = luma
                .iter()
                .map(|v| v.wrapping_add(iteration as u8 * 37))
                .collect::<Vec<_>>();
            let uv_data = uv
                .iter()
                .map(|v| v.wrapping_sub(iteration as u8 * 41))
                .collect::<Vec<_>>();
            luma_memory.write(0, &y_data).unwrap();
            chroma_memory.write(0, &uv_data).unwrap();
            output_memory.write(0, &[0xa5; 256]).unwrap();
            let transform = Nv12ColorTransform {
                coefficients,
                shift,
                enabled,
                clamp,
                alpha: 197,
                source_origin: origin,
            };
            let operations = vec![
                GpuOperation::new(
                    GpuCommand::ConvertNv12(Nv12Conversion {
                        luma: luma_region,
                        chroma: chroma_region,
                        destination: output_region,
                        destination_format: format,
                        transform,
                    }),
                    [],
                    [],
                    CapabilityRequirements::none(),
                ),
                GpuOperation::new(
                    GpuCommand::Copy(CopyOperation::ImageToImage {
                        source: output_region,
                        destination: copy_region,
                    }),
                    [],
                    [],
                    CapabilityRequirements::none(),
                ),
            ];
            let mut uploads = Vec::new();
            if iteration < 2 {
                for (region, pitch, data) in
                    [(chroma_region, 16, &uv_data), (luma_region, 16, &y_data)]
                {
                    let row_bytes = region.extent.width as usize
                        * if region.image == luma_region.image {
                            1
                        } else {
                            2
                        };
                    let bytes: Vec<u8> = (0..region.extent.height as usize)
                        .flat_map(|row| data[row * pitch..row * pitch + row_bytes].iter().copied())
                        .collect();
                    uploads.push(GpuOperation::new(
                        GpuCommand::UploadImage {
                            destination: region,
                            bytes: bytes.into(),
                        },
                        [],
                        [],
                        CapabilityRequirements::none(),
                    ));
                }
            }
            #[cfg(feature = "performance-counters")]
            let before = nixe_gpu::metrics::snapshot()
                .into_iter()
                .find(|(name, _)| *name == "ImageReadbackBytes")
                .unwrap()
                .1;
            if !uploads.is_empty() {
                runtime
                    .runtime()
                    .submit(
                        &all,
                        &[],
                        &OperationSubmission::new(
                            FrontendSubmissionId::new(10_000 + serial),
                            vec![],
                            uploads,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                all.clear();
            }
            runtime
                .runtime()
                .submit(
                    &all,
                    &[],
                    &OperationSubmission::new(
                        FrontendSubmissionId::new(serial),
                        vec![],
                        operations,
                    )
                    .unwrap(),
                )
                .unwrap();
            all.clear();
            #[cfg(feature = "performance-counters")]
            assert_eq!(
                before,
                nixe_gpu::metrics::snapshot()
                    .into_iter()
                    .find(|(name, _)| *name == "ImageReadbackBytes")
                    .unwrap()
                    .1,
                "conversion and texture consumption must not read pixels back"
            );
            let mut expected = vec![0xa5; 256];
            for y in 0..5 {
                for x in 0..7 {
                    let sx = x + origin[0] as usize;
                    let sy = y + origin[1] as usize;
                    let components = [
                        y_data[sy * 16 + sx],
                        uv_data[(sy / 2) * 16 + (sx / 2) * 2],
                        uv_data[(sy / 2) * 16 + (sx / 2) * 2 + 1],
                    ]
                    .map(|v| i64::from(v) * 4);
                    let mut rgb = std::array::from_fn::<_, 3, _>(|row| {
                        let c = coefficients[row].map(i64::from);
                        let value = if enabled {
                            (((components[0] * c[0] + components[1] * c[1] + components[2] * c[2])
                                >> shift)
                                + c[3])
                                >> 8
                        } else {
                            components[row]
                        };
                        (value
                            .clamp(i64::from(clamp[0]), i64::from(clamp[1]))
                            .clamp(0, 1023)
                            >> 2) as u8
                    });
                    if format == ImageFormat::Bgra8Unorm {
                        rgb.swap(0, 2);
                    }
                    expected[y * 40 + x * 4..y * 40 + x * 4 + 4]
                        .copy_from_slice(&[rgb[0], rgb[1], rgb[2], 197]);
                }
            }
            let mut actual = vec![0; 256];
            output_memory.read(0, &mut actual).unwrap();
            assert_eq!(
                actual, expected,
                "format={format:?} shift={shift} crop={origin:?}"
            );
            let mut copied = vec![0; 256];
            copy_memory.read(0, &mut copied).unwrap();
            for y in 0..5 {
                assert_eq!(&copied[y * 40..y * 40 + 28], &expected[y * 40..y * 40 + 28]);
            }
            serial += 1;
        }
    }
    runtime.runtime().teardown().unwrap();
}

#[test]
fn decoded_plane_uploads_preserve_gob_padding_and_cpu_alias_writes() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(846),
        NonCpuDeviceId::new(846),
        Default::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    for (id, format, texture_width, visible_width, height, length, bpp) in [
        (901, ImageFormat::R8Unorm, 128, 67, 37, 8192, 1),
        (902, ImageFormat::Rg8Unorm, 64, 34, 19, 4096, 2),
    ] {
        let (mut creations, memory, mut region) =
            image(id, format, texture_width, height, 128, &vec![0xa5; length]);
        for creation in &mut creations {
            if let BackendResourceCreateInfo::Image {
                id,
                description,
                view: Some(view),
            } = creation
            {
                let binding = &view.bindings()[0];
                *view = ImageView::new(
                    *id,
                    *description,
                    Swizzle::IDENTITY,
                    vec![(
                        binding.subresources(),
                        ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                            block_width_log2: 0,
                            block_height_log2: 2,
                            block_depth_log2: 0,
                            layer_stride: length as u64,
                        }),
                        binding.backing().clone(),
                    )],
                )
                .unwrap();
            }
        }
        region.extent.width = visible_width;
        let offset = |x: usize, y: usize| {
            (y / 32) * 128 * 32
                + (x / 64) * 512 * 4
                + (y % 32 / 8) * 512
                + (x % 64 / 32) * 256
                + (y % 8 / 2) * 64
                + (x % 32 / 16) * 32
                + (y % 2) * 16
                + x % 16
        };
        let mut expected = vec![0xa5; length];
        for iteration in 0..2 {
            if iteration == 1 {
                let padding = offset(100, 0);
                memory.write(padding, &[0x3b]).unwrap();
                expected[padding] = 0x3b;
            }
            let bytes = (0..visible_width as usize * bpp * height as usize)
                .map(|i| (i * 7 + iteration * 41) as u8)
                .collect::<Vec<_>>();
            for y in 0..height as usize {
                for x in 0..visible_width as usize * bpp {
                    expected[offset(x, y)] = bytes[y * visible_width as usize * bpp + x];
                }
            }
            runtime
                .runtime()
                .submit(
                    &creations,
                    &[],
                    &OperationSubmission::new(
                        FrontendSubmissionId::new(id * 2 + iteration as u64),
                        vec![],
                        vec![GpuOperation::new(
                            GpuCommand::UploadImage {
                                destination: region,
                                bytes: bytes.into(),
                            },
                            [],
                            [],
                            CapabilityRequirements::none(),
                        )],
                    )
                    .unwrap(),
                )
                .unwrap();
            creations.clear();
            let mut actual = vec![0; length];
            memory.read(0, &mut actual).unwrap();
            let differences = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect::<Vec<_>>();
            assert!(
                differences.is_empty(),
                "format={format:?} iteration={iteration} mismatches={} first={:?}",
                differences.len(),
                &differences[..differences.len().min(10)]
            );
        }
    }
    runtime.runtime().teardown().unwrap();
}
