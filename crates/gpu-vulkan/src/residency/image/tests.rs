use super::*;
use crate::ApiPreference;
use crate::residency::tests::{backing_allocation, cache, cpu_read};

#[test]
fn native_depth_reuse_does_not_expand_or_initialize_unwritten_stencil() {
    use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage, WriteDomain};
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, mapping) = backing_allocation(&cache, 4096, 4096, 0x73);
    let domain = ImageFootprint {
        base: 0,
        columns: 8,
        rows: 8,
        packed_bytes: 4,
        aspect_offset: 1,
        aspect_bytes: 3,
        storage: ImageStorage::BlockLinear {
            block_height_log2: 0,
        },
        native_base: 0,
        native_element_bytes: 4,
        native_row_pitch: 32,
        rect: ByteRect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        },
    };
    let geometry = vec![
        Footprint::new(mapping.clone(), WriteDomain::Image(domain)).unwrap(),
        Footprint::new(
            mapping.clone(),
            WriteDomain::Image(ImageFootprint {
                aspect_offset: 0,
                aspect_bytes: 1,
                native_base: 256,
                native_element_bytes: 1,
                native_row_pitch: 8,
                ..domain
            }),
        )
        .unwrap(),
    ];
    let extent = vk::Extent3D {
        width: 8,
        height: 8,
        depth: 1,
    };
    let depth = ImagePlane {
        offset: 0,
        row_pitch: 32,
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
        offset: 256,
        row_pitch: 8,
        aspect: vk::ImageAspectFlags::STENCIL,
        bytes_per_block: 1,
        ..depth
    };
    let id = cache
        .image_for_device_write(
            geometry,
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::D24_UNORM_S8_UINT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![depth, stencil],
        )
        .unwrap();
    cache.clear_depth_stencil(id, 0, 1., 0).unwrap();
    let buffers = cache.buffers.len();
    assert!(
        cache.refresh_native_alias(id).unwrap(),
        "own depth and an unproduced stencil require no byte reconstruction"
    );
    assert_eq!(cache.buffers.len(), buffers);
    let rect = transfers::full_rect(depth);
    cache.require_image_initialized(id, depth, rect).unwrap();
    assert!(cache.require_image_initialized(id, stencil, rect).is_err());
    cache.ensure_image(id).unwrap();
    assert!(cache.require_image_initialized(id, stencil, rect).is_err());
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &mapping);
    assert_eq!(&bytes[..4], &[0x73, 0xff, 0xff, 0xff]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn alternating_partial_aliases_copy_only_newest_texels_and_bound_replay_plans() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, mapping) = backing_allocation(&cache, 16384, 16384, 0x31);
    let extent = vk::Extent3D {
        width: 64,
        height: 64,
        depth: 1,
    };
    let plane = ImagePlane {
        offset: 0,
        row_pitch: 256,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        extent,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        flags: vk::ImageCreateFlags::empty(),
    };
    let geometry = crate::residency::tests::image_geometry(mapping.clone(), description, &[plane]);
    let first = cache
        .image_for(geometry.clone(), description, vec![plane])
        .unwrap();
    let second = cache
        .image_for(
            geometry,
            NativeImageDescription {
                format: vk::Format::R8G8B8A8_SRGB,
                ..description
            },
            vec![plane],
        )
        .unwrap();
    cache.ensure_image(first).unwrap();
    cache.ensure_image(second).unwrap();
    let initial = cache.statistics();
    let mut expected = vec![0x31; 16384];
    for i in 0..160 {
        let (source, target) = if i % 2 == 0 {
            (first, second)
        } else {
            (second, first)
        };
        let x = i % 64;
        let y = i / 64;
        let rect = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        };
        cache
            .clear_image_rect(
                source,
                0,
                rect,
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1., 0., 0., 1.],
                    },
                },
            )
            .unwrap();
        let before = cache.statistics().gpu_copied_bytes;
        cache.ensure_image(target).unwrap();
        cache.ensure_image(target).unwrap();
        assert_eq!(cache.statistics().gpu_copied_bytes - before, 4);
        let offset = (y * 256 + x * 4) as usize;
        expected[offset..offset + 4].copy_from_slice(&[255, 0, 0, 255]);
    }
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        initial.cpu_uploaded_bytes
    );
    assert_eq!(
        cache.statistics().native_readback_bytes,
        initial.native_readback_bytes
    );
    assert_eq!(
        cache.statistics().buffer_allocations,
        initial.buffer_allocations
    );
    assert!(cache.image(first).unwrap().write_plans.len() <= 64);
    assert!(cache.image(second).unwrap().write_plans.len() <= 64);
    assert_eq!(cpu_read(&mut cache, &mapping), expected);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn image_eviction_preserves_authority_and_recycles_generation_after_views_retire() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, mapping) = backing_allocation(&cache, 4096, 4096, 0x31);
    let extent = vk::Extent3D {
        width: 4,
        height: 4,
        depth: 1,
    };
    let plane = ImagePlane {
        offset: 0,
        row_pitch: 16,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        extent,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        usage: vk::ImageUsageFlags::SAMPLED,
        flags: vk::ImageCreateFlags::empty(),
    };
    let geometry = crate::residency::tests::image_geometry(mapping.clone(), description, &[plane]);
    let image = cache
        .image_for(geometry.clone(), description, vec![plane])
        .unwrap();
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [0., 1., 0., 1.],
            },
        )
        .unwrap();
    let view_id = cache
        .image_view(
            image,
            ImageViewDescription {
                format: description.format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: vk::ImageAspectFlags::COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
        .unwrap();
    let held = cache.image_view_lease(view_id).unwrap();
    let work = cache.flush().unwrap().unwrap();
    cache
        .device
        .wait_for_work(work, std::time::Duration::MAX)
        .unwrap();
    assert!(!cache.evict_one().unwrap());
    drop(held);
    assert!(cache.evict_one().unwrap());
    assert!(cache.image_view_lease(view_id).is_err());
    assert_eq!(
        &cpu_read(&mut cache, &mapping)[..64],
        &[0, 255, 0, 255].repeat(16)
    );
    assert_eq!(cache.authority.resource_count(), 0);
    assert_eq!(cache.authority.writer_count(), 0);
    let next = cache.image_for(geometry, description, vec![plane]).unwrap();
    assert_eq!(image.slot, next.slot);
    assert_ne!(image.generation, next.generation);
    assert!(cache.image(image).is_err());
    assert!(
        cache.images[next.slot]
            .entry
            .as_ref()
            .unwrap()
            .native_views
            .is_empty()
    );
    drop(cache);
    validation.check().unwrap();
}

fn rgba64_layout() -> (NativeImageDescription, ImagePlane) {
    let extent = vk::Extent3D {
        width: 64,
        height: 64,
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
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        },
        ImagePlane {
            offset: 0,
            row_pitch: 256,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        },
    )
}

#[test]
fn opaque_cpu_overwrite_revokes_only_replaced_texels() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(&cache, 16384, 4096, 0x31);
    let (description, plane) = rgba64_layout();
    let geometry = super::super::tests::image_geometry(mapping, description, &[plane]);
    let id = cache
        .image_for_device_write(geometry, description, vec![plane])
        .unwrap();
    cache
        .clear_color_image(
            id,
            0,
            vk::ClearColorValue {
                float32: [1., 0., 0., 1.],
            },
        )
        .unwrap();
    super::super::tests::cpu_write(&mut cache, &allocation, 0, &[0, 255, 0, 255]).unwrap();
    let result = cache.ensure_image_initialized(
        id,
        plane,
        vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        },
    );
    cache
        .ensure_image_initialized(
            id,
            plane,
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 0 },
                extent: vk::Extent2D {
                    width: 63,
                    height: 64,
                },
            },
        )
        .unwrap();
    for (offset, bytes, x, y) in [
        (257usize, 1usize, 0, 1),
        (0, 2048, 63, 7),
        (0, 16384, 63, 63),
    ] {
        cache
            .clear_color_image(
                id,
                0,
                vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            )
            .unwrap();
        super::super::tests::cpu_write(&mut cache, &allocation, offset, &vec![0; bytes]).unwrap();
        assert!(
            cache
                .ensure_image_initialized(
                    id,
                    plane,
                    vk::Rect2D {
                        offset: vk::Offset2D { x, y },
                        extent: vk::Extent2D {
                            width: 1,
                            height: 1
                        },
                    }
                )
                .is_err(),
            "partial texels and every overwritten row must lose initialization"
        );
    }
    cache.flush().unwrap();
    drop(cache);
    validation.check().unwrap();
    assert!(
        result.is_err(),
        "CPU-replaced opaque texel cannot be accepted as the old native contents"
    );
}

#[test]
fn tiled_alias_batches_a_partial_rectangle() {
    use nixe_gpu::physical::{ImageStorage, WriteDomain};
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, mapping) = backing_allocation(&cache, 16384, 4096, 0x31);
    let (description, plane) = rgba64_layout();
    let original = super::super::tests::image_geometry(mapping.clone(), description, &[plane]);
    let WriteDomain::Image(mut domain) = original[0].domain() else {
        unreachable!()
    };
    domain.storage = ImageStorage::BlockLinear {
        block_height_log2: 0,
    };
    let geometry = vec![Footprint::new(mapping, WriteDomain::Image(domain)).unwrap()];
    let a = cache
        .image_for(geometry.clone(), description, vec![plane])
        .unwrap();
    let b = cache
        .image_for(
            geometry,
            NativeImageDescription {
                format: vk::Format::R8G8B8A8_SRGB,
                ..description
            },
            vec![plane],
        )
        .unwrap();
    cache.ensure_image(a).unwrap();
    cache.ensure_image(b).unwrap();
    cache
        .clear_image_rect(
            a,
            0,
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: 64,
                    height: 32,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1., 0., 0., 1.],
                },
            },
        )
        .unwrap();
    let before = cache.statistics();
    cache.ensure_image(b).unwrap();
    assert_eq!(
        cache.statistics().alias_copy_commands - before.alias_copy_commands,
        1
    );
    assert_eq!(
        cache.statistics().alias_copy_regions - before.alias_copy_regions,
        1
    );
    let (_output_allocation, output) = backing_allocation(&cache, 16384, 4096, 0);
    let destination = cache
        .image_for_device_write(
            super::super::tests::image_geometry(output.clone(), description, &[plane]),
            description,
            vec![plane],
        )
        .unwrap();
    cache
        .copy_image_region(
            b,
            0,
            transfers::full_rect(plane),
            destination,
            0,
            Default::default(),
        )
        .unwrap();
    let pixels = cpu_read(&mut cache, &output);
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index / 64 < 32 {
                &[255, 0, 0, 255]
            } else {
                &[0x31; 4]
            }
        );
    }
    cache.flush().unwrap();
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn decoded_image_uploads_bounded_rectangles() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    if cache.device.profile().core.texture_compression_astc_ldr != 0 {
        return;
    }
    let validation = cache.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(&cache, 4096, 4096, 0);
    let mut block = [255u8; 16];
    block[..2].copy_from_slice(&[0xfc, 0xfd]);
    for (c, v) in [65535u16, 0, 0, 65535].into_iter().enumerate() {
        block[8 + c * 2..10 + c * 2].copy_from_slice(&v.to_le_bytes());
    }
    super::super::tests::cpu_write(&mut cache, &allocation, 0, &block.repeat(256)).unwrap();
    let (mut description, mut plane) = rgba64_layout();
    description.format = vk::Format::ASTC_4X4_UNORM_BLOCK;
    description.usage = vk::ImageUsageFlags::SAMPLED;
    plane.row_pitch = 256;
    plane.block_extent = vk::Extent2D {
        width: 4,
        height: 4,
    };
    plane.bytes_per_block = 16;
    let geometry = super::super::tests::image_geometry(mapping, description, &[plane]);
    let image = cache
        .sampled_image_for(geometry, description, vec![plane])
        .unwrap();
    cache.ensure_image(image).unwrap();
    let before = cache.statistics();
    block[8..10].copy_from_slice(&0u16.to_le_bytes());
    super::super::tests::cpu_write(&mut cache, &allocation, 0, &block).unwrap();
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes - before.cpu_uploaded_bytes,
        32768
    );
    assert_eq!(
        cache.statistics().decoded_upload_commands - before.decoded_upload_commands,
        2
    );
    let after = cache.statistics();
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cache.statistics().decoded_upload_commands,
        after.decoded_upload_commands
    );
    let native_description = cache.image(image).unwrap().description;
    let native_plane = cache.image(image).unwrap().planes[0];
    let (_output_allocation, output) = backing_allocation(&cache, 32768, 4096, 0);
    let destination = cache
        .image_for_device_write(
            super::super::tests::image_geometry(
                output.clone(),
                native_description,
                &[native_plane],
            ),
            native_description,
            vec![native_plane],
        )
        .unwrap();
    cache
        .copy_image_region(
            image,
            0,
            transfers::full_rect(native_plane),
            destination,
            0,
            Default::default(),
        )
        .unwrap();
    let pixels = cpu_read(&mut cache, &output);
    for (index, pixel) in pixels.chunks_exact(8).enumerate() {
        let red = if index % 64 < 4 && index / 64 < 4 {
            0u16
        } else {
            0x3c00
        };
        let expected: Vec<_> = [red, 0, 0, 0x3c00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(pixel, expected);
    }
    cache.flush().unwrap();
    drop(cache);
    validation.check().unwrap();
}

mod visibility {
    use super::*;
    use nixe_gpu::physical::{
        ByteRect, ImageFootprint, ImageStorage, PhysicalMapping, WriteDomain,
    };
    use nixe_memory::{AllocationRange, Machine, PAGE_SIZE};
    #[test]
    fn cropped_native_aliases_copy_without_a_linear_buffer_representation() {
        let Some(device) = crate::test_device::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
        let mapping = PhysicalMapping::new(&[input]).unwrap();
        let image = |width, height, base| {
            let extent = vk::Extent3D {
                width,
                height,
                depth: 1,
            };
            let geometry = Footprint::new(
                Arc::clone(&mapping),
                WriteDomain::Image(ImageFootprint {
                    base,
                    columns: width,
                    rows: height,
                    packed_bytes: 4,
                    aspect_offset: 0,
                    aspect_bytes: 4,
                    storage: ImageStorage::Pitch { row_pitch: 80 },
                    native_base: 0,
                    native_element_bytes: 4,
                    native_row_pitch: u64::from(width) * 4,
                    rect: ByteRect {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    },
                }),
            )
            .unwrap();
            let description = NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UINT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            };
            let plane = ImagePlane {
                offset: 0,
                row_pitch: u64::from(width) * 4,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::COLOR,
                extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            };
            (vec![geometry], description, vec![plane])
        };
        let mut residency =
            VulkanResidency::new(machine.clone(), device, 32 * 1024 * 1024).unwrap();
        let (geometry, description, planes) = image(17, 13, 0);
        let source = residency
            .image_for_device_write(geometry, description, planes)
            .unwrap();
        let (geometry, description, planes) = image(13, 9, 0);
        let target = residency.image_for(geometry, description, planes).unwrap();
        let allocated = residency.allocated_bytes();
        for color in [[12, 34, 56, 78], [87, 65, 43, 21]] {
            residency
                .clear_color_image(source, 0, vk::ClearColorValue { uint32: color })
                .unwrap();
            residency.ensure_image(target).unwrap();
            assert_eq!(residency.allocated_bytes(), allocated);
        }
        assert_eq!(machine.tracking_statistics().captured_bytes, 0);
        // Publish a distinct destination from the cropped native image, then read
        // its actual GPU bytes rather than its source authority's representation.
        let (geometry, description, planes) = image(13, 9, PAGE_SIZE);
        let destination = residency
            .image_for_device_write(geometry, description, planes)
            .unwrap();
        residency
            .copy_image_region(
                target,
                0,
                vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: 13,
                        height: 9,
                    },
                },
                destination,
                0,
                vk::Offset2D::default(),
            )
            .unwrap();
        let output = AllocationRange::new(allocation.pin(), PAGE_SIZE, 9 * 80).unwrap();
        let nixe_memory::AccessResult::Read(bytes) = residency
            .access_memory(vec![output], nixe_memory::AccessOperation::Read)
            .unwrap()
        else {
            panic!("physical read");
        };
        for row in bytes.chunks_exact(80) {
            for pixel in row[..13 * 4].chunks_exact(4) {
                assert_eq!(pixel, [87, 65, 43, 21]);
            }
            assert!(row[13 * 4..].iter().all(|&byte| byte == 0));
        }
        drop(residency);
        validation.check().unwrap();
    }

    #[test]
    fn separate_depth_and_stencil_writes_preserve_native_contents_without_ram_reconstruction() {
        let Some(device) = crate::test_device::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 256).unwrap();
        let mapping = PhysicalMapping::new(std::slice::from_ref(&input)).unwrap();
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::D24_UNORM_S8_UINT,
            extent: vk::Extent3D {
                width: 8,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        };
        let mut planes = Vec::new();
        let mut geometry = Vec::new();
        for (aspect, offset, bytes, native_bytes, native_base) in [
            (vk::ImageAspectFlags::DEPTH, 1, 3, 4, 0),
            (vk::ImageAspectFlags::STENCIL, 0, 1, 1, 256),
        ] {
            geometry.push(
                Footprint::new(
                    Arc::clone(&mapping),
                    WriteDomain::Image(ImageFootprint {
                        base: 0,
                        columns: 8,
                        rows: 8,
                        packed_bytes: 4,
                        aspect_offset: offset,
                        aspect_bytes: bytes,
                        storage: ImageStorage::Pitch { row_pitch: 32 },
                        native_base,
                        native_element_bytes: native_bytes,
                        native_row_pitch: 8 * u64::from(native_bytes),
                        rect: ByteRect {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                        },
                    }),
                )
                .unwrap(),
            );
            planes.push(ImagePlane {
                offset: native_base,
                row_pitch: 8 * u64::from(native_bytes),
                mip_level: 0,
                array_layer: 0,
                aspect,
                extent: description.extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: native_bytes,
            });
        }
        let mut residency =
            VulkanResidency::new(machine.clone(), device, 32 * 1024 * 1024).unwrap();
        let image = residency.image_for(geometry, description, planes).unwrap();
        let before = machine.tracking_statistics().captured_bytes;
        for stencil in [0x37, 0x5a] {
            residency.clear_depth_stencil(image, 0, 1.0, 0).unwrap();
            residency
                .clear_depth_stencil(image, 1, 0.0, stencil)
                .unwrap();
            residency.ensure_image(image).unwrap();
        }
        assert_eq!(
            machine.tracking_statistics().captured_bytes,
            before,
            "native aspect writes must not recapture their old RAM inputs"
        );
        let nixe_memory::AccessResult::Read(result) = residency
            .access_memory(vec![input], nixe_memory::AccessOperation::Read)
            .unwrap()
        else {
            panic!("physical read");
        };
        for pixel in result.chunks_exact(4) {
            assert_eq!(pixel, [0x5a, 0xff, 0xff, 0xff]);
        }
        drop(residency);
        validation.check().unwrap();
    }
}
