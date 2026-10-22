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
