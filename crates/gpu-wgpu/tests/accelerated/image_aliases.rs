use super::*;

#[test]
fn retained_read_only_image_alias_refreshes_after_device_writes() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(840),
        NonCpuDeviceId::new(840),
        Default::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[32, 64, 96, 255].repeat(16));
    let (mut creations, shared, writer, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 4, 4, &[page]);
    let reader = ImageId::new(702);
    let destination = ImageId::new(703);
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(3, 4, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    creations.push(BackendResourceCreateInfo::Image {
        id: reader,
        description,
        view: Some(
            ImageView::new(
                reader,
                description,
                Swizzle::IDENTITY,
                vec![(
                    subresources,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: 16,
                        layer_stride: 64,
                    },
                    shared,
                )],
            )
            .unwrap(),
        ),
    });
    let allocation = GpuAllocationId::new(703);
    let allocation_description = GpuAllocationDescription::new(48, 4).unwrap();
    let pixels = CanonicalAllocation::zeroed(48, 4).unwrap();
    let backing = BackingView::new(
        allocation,
        allocation_description,
        0,
        pixels.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
    )
    .unwrap();
    creations.push(BackendResourceCreateInfo::Allocation {
        id: allocation,
        description: allocation_description,
    });
    creations.push(BackendResourceCreateInfo::Image {
        id: destination,
        description,
        view: Some(
            ImageView::new(
                destination,
                description,
                Swizzle::IDENTITY,
                vec![(
                    subresources,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: 12,
                        layer_stride: 48,
                    },
                    backing,
                )],
            )
            .unwrap(),
        ),
    });
    let region = |image| nixe_gpu::ImageRegion {
        image,
        subresources,
        origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
        extent: ImageExtent::new(3, 4, 1).unwrap(),
    };
    let copy = |serial| {
        OperationSubmission::new(
            FrontendSubmissionId::new(serial),
            vec![],
            vec![GpuOperation::new(
                GpuCommand::Copy(CopyOperation::ImageToImage {
                    source: region(reader),
                    destination: region(destination),
                }),
                [],
                [],
                CapabilityRequirements::none(),
            )],
        )
        .unwrap()
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime.runtime().submit(&creations, &[], &copy(1)).unwrap();
    let mut actual = [0; 48];
    pixels.read(0, &mut actual).unwrap();
    assert_eq!(actual.to_vec(), [32, 64, 96, 255].repeat(12));
    // The smaller read view remains allocated while the wider alias writes.
    // No CPU write or resource destruction may be needed to refresh it.
    for (serial, color, expected) in [
        (2, [1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
        (4, [0.0, 1.0, 0.0, 1.0], [0, 255, 0, 255]),
    ] {
        runtime
            .runtime()
            .submit(
                &[],
                &[],
                &color_clear_submission(
                    writer,
                    subresources,
                    ImageFormat::Rgba8Unorm,
                    4,
                    4,
                    color,
                    serial,
                ),
            )
            .unwrap();
        runtime
            .runtime()
            .submit(&[], &[], &copy(serial + 1))
            .unwrap();
        pixels.read(0, &mut actual).unwrap();
        assert_eq!(actual.to_vec(), expected.repeat(12));
    }
}
