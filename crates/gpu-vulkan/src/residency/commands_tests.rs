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
            depth_slice: 0,
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
fn packed_rgb10a2_clear_preserves_native_bits_and_guest_row_padding() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation, storage) = backing_allocation(cache.device_memory(), 4096, 0x31);
    let (mut description, planes) = color_image();
    description.format = vk::Format::A2B10G10R10_UNORM_PACK32;
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(storage.clone(), description, &planes),
            description,
            planes,
            None,
        )
        .unwrap();
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [1.0, 0.0, 0.0, 1.0],
            },
        )
        .unwrap();
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &storage);
    assert_eq!(&bytes[..32], &[0x31; 32]);
    for row in 0..6 {
        let start = 32 + row * 40;
        assert_eq!(
            &bytes[start..start + 32],
            0xc00003ff_u32.to_le_bytes().repeat(8)
        );
        assert_eq!(&bytes[start + 32..start + 40], &[0x31; 8]);
    }
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn dense_image_buffer_alias_preserves_compact_authority_and_rejects_stale_images() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_2, physical) = backing_allocation(cache.device_memory(), 16384, 0x31);
    let backing = Footprint::new(
        physical.clone(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: 64,
            rows: 64,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
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
            depth_slice: 0,
            base: 0,
            columns: 32,
            rows: 128,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
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
        .image_for(vec![backing.clone()], description, planes, None)
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn image_alias_invalidation_preserves_unrelated_buffer_pages() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_3, backing) = backing_allocation(cache.device_memory(), 3 * 4096, 0x19);
    let alias = buffer(&mut cache, backing.clone());
    cache
        .ensure_buffer_range(alias, 0, backing.bytes())
        .unwrap();
    let (description, planes) = color_image();
    for offset in [0, 8192] {
        let image = cache
            .image_for(
                crate::residency::tests::image_geometry(
                    backing.slice(offset, 4096).unwrap(),
                    description,
                    &planes.clone(),
                ),
                description,
                planes.clone(),
                None,
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
    let copied = cache.statistics().gpu_copied_bytes;
    let uploaded = cache.statistics().cpu_uploaded_bytes;
    // The middle page consumes neither image. Their writes must stay pending
    // until their own ranges are requested. The synchronized middle page stays valid.
    assert_eq!(cache.read_buffer(alias, 4096, 32).unwrap(), [0x19; 32]);
    assert_eq!(cache.statistics().gpu_copied_bytes, copied);
    assert_eq!(cache.statistics().cpu_uploaded_bytes - uploaded, 0);
    assert_eq!(cache.read_buffer(alias, 4096, 32).unwrap(), [0x19; 32]);
    assert_eq!(cache.statistics().cpu_uploaded_bytes - uploaded, 0);
    let first = cache.read_buffer(alias, 0, 4096).unwrap();
    assert!(cache.statistics().gpu_copied_bytes > copied);
    let copied_first = cache.statistics().gpu_copied_bytes;
    assert_eq!(cache.read_buffer(alias, 0, 4096).unwrap(), first);
    assert_eq!(cache.statistics().gpu_copied_bytes, copied_first);
    let bytes = cache.read_buffer(alias, 0, 3 * 4096).unwrap();
    assert!(cache.statistics().gpu_copied_bytes > copied_first);
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[cfg(feature = "binding-probe")]
#[test]
fn neighboring_cpu_reads_share_one_full_image_download() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_13, backing) = backing_allocation(cache.device_memory(), 512, 0x31);
    let (mut description, mut planes) = color_image();
    description.extent.height = 16;
    planes[0].offset = 0;
    planes[0].row_pitch = 32;
    planes[0].extent.height = 16;
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes),
            description,
            planes,
            None,
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
        let before = cache.runtime.device.queue_usage().scheduled_tick;
        // The first safe read flushes the full image; the next consumes RAM.
        let downloaded = cache.statistics().native_readback_bytes;
        let mut reads = Vec::new();
        for offset in [0, 64] {
            let (request, pending) = cache
                .runtime
                .memory
                .prepare_device_access(
                    backing.pin(offset, 64).unwrap(),
                    nixe_memory::AccessOperation::Read(Vec::new()),
                )
                .unwrap();
            cache.queue_device_access(request).unwrap();
            reads.push(pending);
        }
        cache.flush().unwrap();
        for pending in &mut reads {
            let nixe_memory::AccessResult::Read(bytes) =
                pending.poll().expect("completed read").unwrap()
            else {
                panic!("expected read")
            };
            assert_eq!(&*bytes, expected.repeat(16));
        }
        assert_eq!(cache.statistics().native_readback_bytes - downloaded, 512);
        assert_eq!(cpu_read(&mut cache, &backing), expected.repeat(128));
        assert_eq!(
            cache.runtime.device.queue_usage().scheduled_tick - before,
            1,
            "neighboring safe reads must share one completed image download"
        );
    }
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn interleaved_scanout_rows_batch_and_scatter_multiple_versioned_windows() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_14, physical) = backing_allocation(cache.device_memory(), 16384, 0x31);
    // Every canonical page contains rows from all four native readback windows.
    let mut segments = Vec::new();
    for row in 0..128 {
        let physical_row = row % 32 * 4 + row / 32;
        segments.push(physical.slice(physical_row * 128, 128).unwrap());
    }
    let backing = Footprint::new(
        PhysicalMapping::join(&segments).unwrap(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: 32,
            rows: 128,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
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
        .image_for(vec![backing.clone()], description, planes.clone(), None)
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
            let clear_view = cache.attachment_view(image, 0).unwrap();
            cache
                .clear_view_rect(
                    clear_view,
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
        let before = cache.runtime.device.queue_usage().scheduled_tick;
        assert_eq!(cpu_read(&mut cache, &physical), expected);
        assert_eq!(
            cache.runtime.device.queue_usage().scheduled_tick - before,
            1
        );
    }
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}
