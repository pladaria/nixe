use super::*;

#[test]
fn retired_images_preserve_contents_without_occupying_reused_logical_slots() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(740),
        NonCpuDeviceId::new(740),
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter unavailable; skipping resource lifetime acceptance test");
        return;
    };
    let presentation = initialized.presentation_context();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(4, 4, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let allocation_description = GpuAllocationDescription::new(64, 4).unwrap();
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let layout = ImageMemoryLayout::PitchLinear {
        row_pitch: 16,
        layer_stride: 64,
    };
    let image = ImageId::new(740);
    let colors = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let mut creations = Vec::new();
    let backings = (0..4)
        .map(|generation| {
            let allocation = GpuAllocationId::new(740 + generation);
            creations.push(BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            });
            backing(
                allocation,
                allocation_description,
                &initialized_page(&[0; 64]),
            )
        })
        .collect::<Vec<_>>();
    for (generation, image_backing) in backings.iter().enumerate() {
        // All allocations exist before the first image. Each recreation must
        // therefore reuse the same vacant image slot with a newer generation.
        creations.push(BackendResourceCreateInfo::Image {
            id: image,
            description,
            view: Some(
                ImageView::new(
                    image,
                    description,
                    Swizzle::IDENTITY,
                    vec![(subresources, layout, image_backing.clone())],
                )
                .unwrap(),
            ),
        });
        let invalidations = if generation < 3 {
            vec![ResourceDependency::Image(image)]
        } else {
            vec![]
        };
        runtime
            .runtime()
            .submit(
                &creations,
                &invalidations,
                &color_clear_submission(
                    image,
                    subresources,
                    ImageFormat::Rgba8Unorm,
                    4,
                    4,
                    colors[generation],
                    740 + generation as u64,
                ),
            )
            .unwrap();
        runtime.runtime().wait_for_completion().unwrap().unwrap();
        creations.clear();
        assert!(matches!(
            image_backing.range().segments()[0].visibility_state(),
            nixe_memory::VisibilityState::GpuNewer { .. }
        ));
    }
    for (generation, image_backing) in backings.iter().enumerate() {
        let expected = colors[generation]
            .map(|component| (component * 255.0) as u8)
            .repeat(16);
        // Presentation must also resolve retained generations after slot reuse.
        let resident = runtime
            .runtime()
            .acquire_presentable_image(PresentationImageRequest {
                allow_canonical_import: true,
                cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(
                    image_backing.range(),
                )
                .unwrap(),
                backing: image_backing.clone(),
                width: 4,
                height: 4,
                format: PresentationImageFormat::Rgba8,
                layout,
                row_pitch: 16,
            })
            .unwrap();
        assert_eq!(read_presented_rgba(&presentation, &resident), expected);
        let mut bytes = [0; 64];
        image_backing.range().read(0, &mut bytes).unwrap();
        assert_eq!(bytes.as_slice(), expected);
    }
    // Reclaiming the old generations must not remove the active slot's record.
    runtime
        .runtime()
        .submit(
            &[],
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba8Unorm,
                4,
                4,
                [1.0; 4],
                750,
            ),
        )
        .unwrap();
    let mut bytes = [0; 64];
    backings[3].range().read(0, &mut bytes).unwrap();
    assert_eq!(bytes, [255; 64]);
}
