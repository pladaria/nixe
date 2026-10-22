use super::execution::tests::Utilities;
use super::tests::{backing_allocation, buffer, cache, cpu_read};
use super::tests::{footprint_buffer, footprint_bytes};
use super::*;
use crate::ApiPreference;
#[cfg(feature = "binding-probe")]
use crate::BufferBinding;
fn color_image() -> (NativeImageDescription, Vec<ImagePlane>) {
    let extent = vk::Extent3D {
        width: 8,
        height: 6,
        depth: 1,
    };
    (
        NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        },
        vec![ImagePlane {
            offset: 32,
            row_pitch: 40,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        }],
    )
}
#[test]
fn cropped_native_alias_skips_buffer_reconstruction_and_preserves_undefined_padding() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_1, storage) = backing_allocation(&cache, 4096, 256, 0x73);
    let make = |cache: &mut VulkanResidency, width, height| {
        let backing = Footprint::new(
            storage.clone(),
            nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
                base: 0,
                columns: width,
                rows: height,
                packed_bytes: 4,
                aspect_offset: 0,
                aspect_bytes: 4,
                storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                    block_height_log2: 0,
                },
                native_base: 0,
                native_element_bytes: 4,
                native_row_pitch: (width as u64) * 4_u64,
                rect: nixe_gpu::physical::ByteRect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
            }),
        )
        .unwrap();
        let (mut description, mut planes) = color_image();
        description.extent.width = width;
        description.extent.height = height;
        planes[0].offset = 0;
        planes[0].row_pitch = u64::from(width) * 4;
        planes[0].extent = description.extent;
        let id = cache
            .image_for_device_write(vec![backing.clone()], description, planes)
            .unwrap();
        (id, backing)
    };
    let (source, _) = make(&mut cache, 19, 9);
    let (target, bytes) = make(&mut cache, 17, 7);
    let plane = cache.image(target).unwrap().planes[0];
    let area = transfers::full_rect(plane);
    for color in [[1., 0., 0., 1.], [0., 1., 0., 1.]] {
        cache
            .clear_image_rect(
                source,
                0,
                area,
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: color },
                },
            )
            .unwrap();
        cache.ensure_image_initialized(target, plane, area).unwrap();
        assert_eq!(cache.statistics().cpu_uploaded_bytes, 0);
        assert!(
            cache.buffers.iter().all(|slot| slot.entry.is_none()),
            "a compatible rectangle must transfer directly between native images"
        );
        assert!(cache.image(target).unwrap().discarded.is_empty());
        assert!(!cache.image(source).unwrap().discarded.is_empty());
    }
    cache.flush().unwrap();
    let readback = footprint_buffer(&mut cache, bytes.clone());
    let actual = cache
        .read_buffer(readback, 0, footprint_bytes(&bytes))
        .unwrap();
    for (index, pixel) in actual.chunks_exact(4).enumerate() {
        assert_eq!(pixel, [0, 255, 0, 255], "pixel {index}");
    }
    let (unwritten, _) = make(&mut cache, 18, 7);
    let plane = cache.image(unwritten).unwrap().planes[0];
    assert!(
        cache
            .ensure_image_initialized(unwritten, plane, transfers::full_rect(plane))
            .is_err(),
        "native alias projection must not substitute undefined source texels"
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn dense_image_buffer_alias_preserves_compact_authority_and_rejects_stale_images() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_2, physical) = backing_allocation(&cache, 16384, 4096, 0x31);
    let backing = Footprint::new(
        physical.clone(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            base: 0,
            columns: 64,
            rows: 64,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
            },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 64_u64 * 4_u64,
            rect: nixe_gpu::physical::ByteRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        }),
    )
    .unwrap();
    let reshaped = Footprint::new(
        physical.clone(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            base: 0,
            columns: 32,
            rows: 128,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
            },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 32_u64 * 4_u64,
            rect: nixe_gpu::physical::ByteRect {
                x: 0,
                y: 0,
                width: 32,
                height: 128,
            },
        }),
    )
    .unwrap();
    assert_ne!(backing, reshaped);
    let (mut description, mut planes) = color_image();
    description.extent.width = 64;
    description.extent.height = 64;
    planes[0].offset = 0;
    planes[0].row_pitch = 256;
    planes[0].extent = description.extent;
    let image = cache
        .image_for_device_write(vec![backing.clone()], description, planes)
        .unwrap();
    for color in [[1., 0., 0., 1.], [0., 0., 1., 1.]] {
        cache
            .clear_color_image(image, 0, vk::ClearColorValue { float32: color })
            .unwrap();
        let alias = footprint_buffer(&mut cache, backing.clone());
        let expected: Vec<_> = color.map(|v| (v * 255.) as u8).repeat(4096);
        assert_eq!(cache.read_buffer(alias, 0, 16384).unwrap(), expected);
        let reshape_alias = footprint_buffer(&mut cache, reshaped.clone());
        assert_eq!(
            cache.read_buffer(reshape_alias, 0, 16384).unwrap(),
            expected
        );
    }
    // A foreign buffer producer makes the image stale. Its previous native
    // pixels cannot be substituted for those new canonical bytes.
    let foreign = buffer(&mut cache, physical);
    cache.fill_buffer(foreign, 0, 4, 0x44332211).unwrap();
    let reshaped_alias = footprint_buffer(&mut cache, reshaped);
    assert_eq!(
        cache.read_buffer(reshaped_alias, 0, 4).unwrap(),
        [0x11, 0x22, 0x33, 0x44]
    );
    let alias = footprint_buffer(&mut cache, backing);
    assert_eq!(
        cache.read_buffer(alias, 0, 4).unwrap(),
        [0x11, 0x22, 0x33, 0x44]
    );
    // The foreign producer may publish its four bytes at the fence; no image
    // content should round-trip through canonical memory.
    assert!(cache.statistics().canonical_readback_bytes <= 4);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn image_alias_invalidation_preserves_unrelated_buffer_pages() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_3, backing) = backing_allocation(&cache, 3 * 4096, 4096, 0x19);
    let alias = buffer(&mut cache, backing.clone());
    let (description, planes) = color_image();
    for offset in [0, 8192] {
        let image = cache
            .image_for_device_write(
                crate::residency::tests::image_geometry(
                    backing.slice(offset, 4096).unwrap(),
                    description,
                    &planes.clone(),
                ),
                description,
                planes.clone(),
            )
            .unwrap();
        cache
            .clear_color_image(
                image,
                0,
                vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            )
            .unwrap();
    }
    assert!(
        cache
            .buffer(alias)
            .unwrap()
            .contents
            .device
            .iter()
            .any(ResourceObservation::is_invalid)
    );
    let bytes = cache.read_buffer(alias, 0, 3 * 4096).unwrap();
    assert_eq!(&bytes[4096..8192], &[0x19; 4096]);
    for offset in [0, 8192] {
        for y in 0..6 {
            for x in 0..8 {
                let at = offset + 32 + y * 40 + x * 4;
                assert_eq!(&bytes[at..at + 4], &[255, 0, 0, 255]);
            }
        }
        assert_eq!(&bytes[offset..offset + 32], &[0x19; 32]);
    }
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    drop(cache);
    validation.check().unwrap();
}
fn exercise(api: ApiPreference, recording: RecordingPolicy) {
    let Some(mut cache) = cache(api, 16 * 1024 * 1024) else {
        return;
    };
    cache.unified_layouts = cache.device.profile().unified_image_layouts
        && recording.layouts == LayoutPolicy::Automatic;
    cache.barrier_policy = recording.barriers;
    let validation = cache.device.validation_monitor();
    let (_allocation_4, bytes) = backing_allocation(&cache, 4096, 4096, 0x1b);
    let id = buffer(&mut cache, bytes.clone());
    cache.fill_buffer(id, 4, 64, 0x44332211).unwrap();
    cache.conversion_fixture(id, 4, 16).unwrap();
    let alias = buffer(&mut cache, bytes.slice(4, 64).unwrap());
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    assert_eq!(
        cache.read_buffer(alias, 0, 64).unwrap(),
        0x44112233u32.to_ne_bytes().repeat(16)
    );
    let (_allocation_5, image_bytes) = backing_allocation(&cache, 4096, 4096, 0x77);
    let (description, planes) = color_image();
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(
                image_bytes.clone(),
                description,
                &planes.clone(),
            ),
            description,
            planes.clone(),
        )
        .unwrap();
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [0.0, 0.0, 1.0, 1.0],
            },
        )
        .unwrap();
    let rect = vk::Rect2D {
        offset: vk::Offset2D { x: 2, y: 1 },
        extent: vk::Extent2D {
            width: 3,
            height: 2,
        },
    };
    cache
        .raster_fixture(
            image,
            0,
            rect,
            [1.0, 0.0, 0.0, 1.0],
            AttachmentLoad::Load,
            AttachmentStore::Store,
        )
        .unwrap();
    let image_alias = buffer(&mut cache, image_bytes.clone());
    cache.conversion_fixture(image_alias, 32 + 40, 8).unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &image_bytes);
    for y in 0..6 {
        for x in 0..8 {
            let offset = 32 + y * 40 + x * 4;
            let red = (1..3).contains(&y) && (2..5).contains(&x);
            let mut expected = if red {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            if y == 1 {
                expected.swap(0, 2);
            }
            assert_eq!(&result[offset..offset + 4], &expected);
        }
        assert_eq!(&result[64 + y * 40..72 + y * 40], &[0x77; 8]);
    }
    // RAR in the same scope has no new barrier. Moving a writer to a different
    // reader stage still requires visibility even after its first read.
    let before = cache.barrier_statistics().calls;
    for _ in 0..20 {
        cache
            .buffer_access_range(
                id,
                4,
                64,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )
            .unwrap();
        cache.barriers.emit(&mut cache.device);
    }
    assert_eq!(cache.barrier_statistics().calls, before);
    assert_eq!(cache.device.queue_usage().retained_readbacks, 0);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn ordered_compute_raster_aliases_core_unified_memory() {
    exercise(ApiPreference::Auto, RecordingPolicy::default());
}
#[test]
fn ordered_compute_raster_aliases_core_conventional_resource() {
    exercise(
        ApiPreference::Auto,
        RecordingPolicy {
            layouts: LayoutPolicy::Conventional,
            barriers: BarrierPolicy::PerResource,
        },
    );
}
#[test]
fn ordered_compute_raster_aliases_khr_conventional_memory() {
    exercise(
        ApiPreference::Compatibility12,
        RecordingPolicy {
            layouts: LayoutPolicy::Conventional,
            barriers: BarrierPolicy::ScopedMemory,
        },
    );
}
#[cfg(feature = "binding-probe")]
#[test]
fn binding_candidates_execute_identical_words_and_recycle_descriptors() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let bindings = [
        BufferBinding::DescriptorSet,
        BufferBinding::DeviceAddress,
        BufferBinding::DescriptorBuffer,
    ];
    for binding in bindings {
        if binding == BufferBinding::DeviceAddress
            && (!cache.device.profile().buffer_device_address
                || cache.device.profile().core.shader_int64 == 0)
            || binding == BufferBinding::DescriptorBuffer
                && !cache.device.profile().descriptor_buffer
        {
            continue;
        }
        let (_allocation_6, bytes) = backing_allocation(&cache, 4096, 4096, 0x01);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | if binding == BufferBinding::DescriptorSet {
                vk::BufferUsageFlags::empty()
            } else {
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            };
        let id = cache.buffer_for(bytes.clone(), usage).unwrap();
        for _ in 0..10 {
            cache.fill_buffer(id, 12, 128, 0xbbaaccdd).unwrap();
            cache.swap_rgba8_words_binding(id, 12, 32, binding).unwrap();
            let work = cache.flush().unwrap().unwrap();
            cache
                .device
                .wait_for_work(work, std::time::Duration::from_secs(5))
                .unwrap();
        }
        assert_eq!(
            cache.read_buffer(id, 12, 128).unwrap(),
            0xbbddccaau32.to_ne_bytes().repeat(32)
        );
        assert_eq!(
            cpu_read(&mut cache, &bytes)[12..140],
            0xbbddccaau32.to_ne_bytes().repeat(32)
        );
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn undefined_and_discarded_reads_require_an_actual_producer() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_7, bytes) = backing_allocation(&cache, 4096, 4096, 0x5b);
    let (description, planes) = color_image();
    let full = transfers::full_rect(planes[0]);
    let plane = planes[0];
    let image = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(bytes.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        0,
        "allocation must not fabricate initialization"
    );
    assert!(cache.require_image_initialized(image, plane, full).is_err());
    cache
        .raster_fixture(
            image,
            0,
            full,
            [1.0, 0.0, 0.0, 1.0],
            AttachmentLoad::Discard,
            AttachmentStore::Store,
        )
        .unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &bytes);
    assert_eq!(&data[32..36], &[255, 0, 0, 255]);
    assert_eq!(&data[..32], &[0x5b; 32]);
    cache
        .raster_fixture(
            image,
            0,
            full,
            [0.0; 4],
            AttachmentLoad::Load,
            AttachmentStore::Discard,
        )
        .unwrap();
    assert!(cache.require_image_initialized(image, plane, full).is_err());
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [0.0, 1.0, 0.0, 1.0],
            },
        )
        .unwrap();
    cache.flush().unwrap();
    assert_eq!(&cpu_read(&mut cache, &bytes)[32..36], &[0, 255, 0, 255]);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn native_copy_preserves_other_mips_layers_and_pitch_padding() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    cache.unified_layouts = false;
    let validation = cache.device.validation_monitor();
    let (_allocation_8, a) = backing_allocation(&cache, 4096, 4096, 0x19);
    let (_allocation_9, b) = backing_allocation(&cache, 4096, 4096, 0x73);
    let (mut description, mut planes) = color_image();
    description.mip_levels = 2;
    description.array_layers = 2;
    let mut extra = planes[0];
    extra.offset = 512;
    extra.mip_level = 1;
    extra.array_layer = 1;
    extra.extent.width /= 2;
    extra.extent.height /= 2;
    extra.row_pitch = 24;
    planes.push(extra);
    let source = cache
        .image_for(
            crate::residency::tests::image_geometry(a, description, &planes.clone()),
            description,
            planes.clone(),
        )
        .unwrap();
    let destination = cache
        .image_for(
            crate::residency::tests::image_geometry(b.clone(), description, &planes.clone()),
            description,
            planes.clone(),
        )
        .unwrap();
    cache
        .clear_color_image(
            source,
            1,
            vk::ClearColorValue {
                float32: [1.0, 0.0, 0.0, 1.0],
            },
        )
        .unwrap();
    let rect = vk::Rect2D {
        offset: vk::Offset2D { x: 1, y: 0 },
        extent: vk::Extent2D {
            width: 2,
            height: 2,
        },
    };
    cache
        .copy_image_region(source, 1, rect, destination, 1, vk::Offset2D { x: 0, y: 1 })
        .unwrap();
    cache.flush().unwrap();
    let result = cpu_read(&mut cache, &b);
    assert_eq!(&result[32..64], &[0x73; 32]);
    assert_eq!(&result[536..544], &[255, 0, 0, 255, 255, 0, 0, 255]);
    assert_eq!(&result[552..560], &[0x73; 8]);
    assert_eq!(cache.statistics().canonical_readback_bytes, 16);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn multisample_native_draw_resolves_then_publishes_exact_texels() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    cache.unified_layouts = false;
    let validation = cache.device.validation_monitor();
    let (_allocation_10, bytes) = backing_allocation(&cache, 4096, 4096, 0x35);
    let (description, planes) = color_image();
    let image = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(bytes.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    cache
        .multisample_fixture(image, 0, vk::SampleCountFlags::TYPE_4, [1.0, 0.0, 0.0, 1.0])
        .unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &bytes);
    for y in 0..6 {
        assert_eq!(&data[32 + y * 40..64 + y * 40], &[255, 0, 0, 255].repeat(8));
        assert_eq!(&data[64 + y * 40..72 + y * 40], &[0x35; 8]);
    }
    assert_eq!(cache.statistics().canonical_readback_bytes, 192);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn same_image_copy_snapshots_overlapping_source_pixels() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    cache.unified_layouts = false;
    let validation = cache.device.validation_monitor();
    let (_allocation_11, bytes) = backing_allocation(&cache, 4096, 4096, 0x79);
    let (description, planes) = color_image();
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(bytes.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [0.0, 0.0, 1.0, 1.0],
            },
        )
        .unwrap();
    let red = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: 3,
            height: 1,
        },
    };
    cache
        .raster_fixture(
            image,
            0,
            red,
            [1.0, 0.0, 0.0, 1.0],
            AttachmentLoad::Load,
            AttachmentStore::Store,
        )
        .unwrap();
    let src = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: 5,
            height: 1,
        },
    };
    cache
        .copy_image_region(image, 0, src, image, 0, vk::Offset2D { x: 2, y: 0 })
        .unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &bytes);
    for x in 0..8 {
        let expected = if x < 5 {
            [255, 0, 0, 255]
        } else {
            [0, 0, 255, 255]
        };
        assert_eq!(&data[32 + x * 4..36 + x * 4], &expected);
    }
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn independent_depth_stencil_clear_preserves_the_other_aspect() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    cache.unified_layouts = false;
    let validation = cache.device.validation_monitor();
    let (allocation, bytes) = backing_allocation(&cache, 4096, 4096, 0x31);
    crate::residency::tests::cpu_write(
        &mut cache,
        &allocation,
        0,
        &0.5f32.to_ne_bytes().repeat(16),
    )
    .unwrap();
    crate::residency::tests::cpu_write(&mut cache, &allocation, 128, &[7; 16]).unwrap();
    let extent = vk::Extent3D {
        width: 4,
        height: 4,
        depth: 1,
    };
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::D32_SFLOAT_S8_UINT,
        extent,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        usage: vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        flags: vk::ImageCreateFlags::empty(),
    };
    let depth = ImagePlane {
        offset: 0,
        row_pitch: 16,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::DEPTH,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let stencil = ImagePlane {
        offset: 128,
        row_pitch: 4,
        aspect: vk::ImageAspectFlags::STENCIL,
        bytes_per_block: 1,
        ..depth
    };
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(bytes.clone(), description, &[depth, stencil]),
            description,
            vec![depth, stencil],
        )
        .unwrap();
    cache.clear_depth_stencil(image, 1, f32::NAN, 23).unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &bytes);
    assert_eq!(&data[..64], &0.5f32.to_ne_bytes().repeat(16));
    assert_eq!(&data[128..144], &[23; 16]);
    cache.clear_depth_stencil(image, 0, 0.25, 0).unwrap();
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &bytes);
    assert_eq!(&data[..64], &0.25f32.to_ne_bytes().repeat(16));
    assert_eq!(&data[128..144], &[23; 16]);
    // A depth overwrite must not import the old depth, while the CPU's newly
    // written stencil remains dirty until a native read actually consumes it.
    crate::residency::tests::cpu_write(&mut cache, &allocation, 128, &[17; 16]).unwrap();
    let uploaded = cache.statistics().cpu_uploaded_bytes;
    cache.clear_depth_stencil(image, 0, 0.75, 0).unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, uploaded);
    cache.ensure_image(image).unwrap();
    let (_allocation_12, output) = backing_allocation(&cache, 4096, 4096, 0);
    let destination = buffer(&mut cache, output);
    cache
        .copy_source(Resource::Image(image), 128, destination, 0, 16)
        .unwrap();
    assert_eq!(cache.read_buffer(destination, 0, 16).unwrap(), vec![17; 16]);
    let data = cpu_read(&mut cache, &bytes);
    assert_eq!(&data[..64], &0.75f32.to_ne_bytes().repeat(16));
    assert_eq!(&data[128..144], &[17; 16]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn packed_depth_stencil_authority_preserves_both_aspects_and_padding() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    // One physical page exceeds the test readback arena once native D24
    // transfer padding and the separate stencil plane are included.
    let pixels = 64 * 64;
    let (_allocation, physical) = backing_allocation(&cache, pixels * 4, pixels * 4, 0x31);
    let geometry = {
        use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage, WriteDomain};
        let depth = ImageFootprint {
            base: 0,
            columns: 64,
            rows: 64,
            packed_bytes: 4,
            aspect_offset: 1,
            aspect_bytes: 3,
            storage: ImageStorage::Pitch { row_pitch: 256 },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 256,
            rect: ByteRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        };
        vec![
            Footprint::new(physical.clone(), WriteDomain::Image(depth)).unwrap(),
            Footprint::new(
                physical.clone(),
                WriteDomain::Image(ImageFootprint {
                    aspect_offset: 0,
                    aspect_bytes: 1,
                    native_base: pixels as u64 * 4,
                    native_element_bytes: 1,
                    native_row_pitch: 64,
                    ..depth
                }),
            )
            .unwrap(),
        ]
    };
    let extent = vk::Extent3D {
        width: 64,
        height: 64,
        depth: 1,
    };
    let depth = ImagePlane {
        offset: 0,
        row_pitch: 256,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::DEPTH,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let stencil = ImagePlane {
        offset: pixels as u64 * 4,
        row_pitch: 64,
        aspect: vk::ImageAspectFlags::STENCIL,
        bytes_per_block: 1,
        ..depth
    };
    let image = cache
        .image_for_device_write(
            geometry,
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::D24_UNORM_S8_UINT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![depth, stencil],
        )
        .unwrap();
    cache.clear_depth_stencil(image, 0, 1., 0).unwrap();
    cache.clear_depth_stencil(image, 1, 0., 41).unwrap();
    // Establish both aspect mappings, then advance real submission ticks
    // without rebuilding per-texel authority when those mappings stay fixed.
    cache.clear_depth_stencil(image, 0, 1., 0).unwrap();
    for _ in 0..4 {
        cache.flush().unwrap();
        cache.clear_depth_stencil(image, 1, 0., 41).unwrap();
        cache.clear_depth_stencil(image, 0, 1., 0).unwrap();
    }
    let before = cache.statistics();
    cache.ensure_image(image).unwrap();
    let after = cache.statistics();
    assert_eq!(before.gpu_copied_bytes, after.gpu_copied_bytes);
    assert_eq!(before.cpu_uploaded_bytes, after.cpu_uploaded_bytes);
    cache.clear_depth_stencil(image, 0, 1., 0).unwrap();
    cache.flush().unwrap();
    let submitted = cache.device.submission.scheduled;
    assert_eq!(
        cpu_read(&mut cache, &physical),
        [41, 255, 255, 255].repeat(pixels)
    );
    assert!(
        cache.device.submission.scheduled - submitted <= 2,
        "one page must use staging-sized batches, never a GPU wait per texel"
    );
    // A partial foreign producer invalidates the cached mapping proof. A
    // subsequent stencil write must retain that producer's new depth bytes.
    let buffer = cache
        .buffer_for(physical.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    cache.fill_buffer_range(buffer, 0, 4, 25).unwrap();
    cache.clear_depth_stencil(image, 1, 0., 41).unwrap();
    cache.flush().unwrap();
    let mut expected = [41, 255, 255, 255].repeat(pixels);
    expected[..4].copy_from_slice(&[41, 0, 0, 0]);
    assert_eq!(cpu_read(&mut cache, &physical), expected);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn neighboring_cpu_pages_share_one_versioned_native_image_download() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_13, backing) = backing_allocation(&cache, 512, 64, 0x31);
    let (mut description, mut planes) = color_image();
    description.extent.height = 16;
    planes[0].offset = 0;
    planes[0].row_pitch = 32;
    planes[0].extent.height = 16;
    let image = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    for (value, expected) in [
        ([0., 1., 0., 1.], [0, 255, 0, 255]),
        ([0., 0., 1., 1.], [0, 0, 255, 255]),
    ] {
        cache
            .clear_color_image(image, 0, vk::ClearColorValue { float32: value })
            .unwrap();
        cache.flush().unwrap();
        let before = cache.device.queue_usage().scheduled_tick;
        // Two exact operations share the same native producer download.
        let downloaded = cache.statistics.native_readback_bytes;
        let mut reads = Vec::new();
        for offset in [0, 64] {
            let (request, pending) = cache
                .memory()
                .prepare_device_access(
                    backing.pin(offset, 64).unwrap(),
                    nixe_memory::AccessOperation::Read,
                )
                .unwrap();
            cache.queue_device_access(request).unwrap();
            reads.push(pending);
        }
        cache.flush().unwrap();
        let point = cache.readback_point().unwrap();
        cache
            .device
            .wait_for_work(
                cache.device.submission.ticket(point),
                std::time::Duration::from_secs(5),
            )
            .unwrap();
        cache.progress_device_accesses().unwrap();
        for pending in &mut reads {
            let nixe_memory::AccessResult::Read(bytes) =
                pending.poll().expect("completed read").unwrap()
            else {
                panic!("expected read")
            };
            assert_eq!(&*bytes, expected.repeat(16));
        }
        assert_eq!(cache.statistics.native_readback_bytes - downloaded, 512);
        assert_eq!(cpu_read(&mut cache, &backing), expected.repeat(128));
        assert_eq!(
            cache.device.queue_usage().scheduled_tick - before,
            1,
            "neighboring page faults must share one completed native download"
        );
    }
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn interleaved_scanout_rows_batch_and_scatter_multiple_versioned_windows() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_14, physical) = backing_allocation(&cache, 16384, 4096, 0x31);
    // Every canonical page contains rows from all four native readback windows.
    let mut segments = Vec::new();
    for row in 0..128 {
        let physical_row = row % 32 * 4 + row / 32;
        segments.push(physical.slice(physical_row * 128, 128).unwrap());
    }
    let backing = Footprint::new(
        PhysicalMapping::join(&segments).unwrap(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            base: 0,
            columns: 32,
            rows: 128,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
            },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 32_u64 * 4_u64,
            rect: nixe_gpu::physical::ByteRect {
                x: 0,
                y: 0,
                width: 32,
                height: 128,
            },
        }),
    )
    .unwrap();
    let (mut description, mut planes) = color_image();
    description.extent.width = 32;
    description.extent.height = 128;
    planes[0].offset = 0;
    planes[0].row_pitch = 128;
    planes[0].extent = description.extent;
    let image = cache
        .image_for_device_write(vec![backing.clone()], description, planes.clone())
        .unwrap();
    for rotation in 0..2 {
        cache
            .clear_color_image(image, 0, vk::ClearColorValue { float32: [0.; 4] })
            .unwrap();
        let colors = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255; 4],
        ];
        let mut expected = vec![0; 16384];
        for band in 0..4 {
            let color = colors[(band + rotation) % 4];
            cache
                .clear_image_rect(
                    image,
                    0,
                    vk::Rect2D {
                        offset: vk::Offset2D {
                            x: 0,
                            y: band as i32 * 32,
                        },
                        extent: vk::Extent2D {
                            width: 32,
                            height: 32,
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: color.map(|v| f32::from(v) / 255.),
                        },
                    },
                )
                .unwrap();
            for row in 0..32 {
                let y = band * 32 + row;
                for x in 0..32 {
                    let byte = x * 4;
                    let start = (y / 8) * 1024
                        + (byte / 64) * 512
                        + (byte % 64 / 32) * 256
                        + (y % 8 / 2) * 64
                        + (byte % 32 / 16) * 32
                        + (y % 2) * 16
                        + byte % 16;
                    let logical_row = start / 128;
                    let physical_row = logical_row % 32 * 4 + logical_row / 32;
                    let at = physical_row * 128 + start % 128;
                    expected[at..at + 4].copy_from_slice(&color);
                }
            }
        }
        cache.flush().unwrap();
        let compact_bytes = backing.mapping().metadata_bytes();
        assert_eq!(
            cache.resident_frame_source(&backing, description, &planes),
            Some(image)
        );
        assert_eq!(
            backing.mapping().metadata_bytes(),
            compact_bytes,
            "scanout lookup must keep the tiled mapping compact"
        );
        let before = cache.device.queue_usage().scheduled_tick;
        assert_eq!(cpu_read(&mut cache, &physical), expected);
        assert_eq!(cache.device.queue_usage().scheduled_tick - before, 1);
    }
    drop(cache);
    validation.check().unwrap();
}
