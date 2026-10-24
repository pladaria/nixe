use super::*;
use crate::cache::*;
use crate::residency::tests::backing_allocation;
use ash::vk;
use nixe_gpu::physical::PhysicalMapping;
use nixe_gpu::{
    PresentationImageFormat as F,
    compilation::CompilePool,
    native::{GuestImageAspect, GuestImagePlane, NativePresentationImage},
};
use std::time::Duration;

#[test]
fn vsync_selects_the_requested_surface_mode_without_a_synchronized_fallback() {
    use vk::PresentModeKHR as Mode;
    let modes = [Mode::MAILBOX, Mode::FIFO, Mode::IMMEDIATE];
    assert_eq!(wsi::select_present_mode(true, &modes).unwrap(), Mode::FIFO);
    assert_eq!(
        wsi::select_present_mode(false, &modes).unwrap(),
        Mode::IMMEDIATE
    );
    assert!(wsi::select_present_mode(false, &[Mode::FIFO, Mode::MAILBOX]).is_err());
}

use crate::test_device as support;
fn pipeline(
    renderer: &mut ScreenRenderer,
    compiler: &PipelineCompiler,
    kind: u8,
) -> Arc<NativePipeline> {
    renderer
        .pipeline(compiler, vk::Format::R8G8B8A8_UNORM, kind)
        .unwrap()
}

fn read(cache: &mut VulkanResidency, view: &AllocatedImageView, extent: [u32; 2]) -> Vec<u8> {
    read_in_layout(
        cache,
        view,
        extent,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    )
}
fn read_in_layout(
    cache: &mut VulkanResidency,
    view: &AllocatedImageView,
    extent: [u32; 2],
    layout: vk::ImageLayout,
) -> Vec<u8> {
    let device = cache.device_mut();
    let (slot, slice, ticket) = device
        .submission
        .reserve_readback(
            &mut device.dispatch,
            u64::from(extent[0]) * u64::from(extent[1]) * 4,
        )
        .unwrap();
    let device = cache.device_mut();
    let command = device.submission.slots[slot].command;
    render::transition(
        device,
        command,
        view.image.handle,
        layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_READ,
    );
    let subresource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    unsafe {
        device.dispatch.cmd_copy_image_to_buffer(
            command,
            view.image.handle,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            device.submission.slots[slot].readback_buffer(slice).handle,
            &[vk::BufferImageCopy::default()
                .buffer_offset(slice.offset)
                .image_subresource(subresource)
                .image_extent(vk::Extent3D {
                    width: extent[0],
                    height: extent[1],
                    depth: 1,
                })],
        );
    }
    cache.flush().unwrap();
    cache.device_mut().readback(ticket).unwrap()
}
#[test]
fn fragmented_cpu_scanout_refreshes_dirty_pages_and_preserves_old_snapshots() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        16 * 1024 * 1024,
    )
    .unwrap();
    let mut descriptors = DescriptorCache::new(cache.device(), 64);
    let mut frames = Frames::new();
    let (allocation, physical) = backing_allocation(&cache, 8192, 4096, 7);
    // Reverse 16-byte fragments within each physical page: logical texels
    // retain a permutation even when every physical page becomes dirty.
    let mut segments = Vec::new();
    let mut offsets = Vec::new();
    for page in 0..2 {
        for fragment in (0..256).rev() {
            let offset = page * 4096 + fragment * 16;
            segments.push(physical.slice(offset, 16).unwrap());
            offsets.push(offset as usize);
        }
    }
    let backing = PhysicalMapping::join(&segments).unwrap();
    let request = NativePresentationImage {
        inputs: (backing).pin(0, (backing).bytes()).unwrap(),
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: crate::residency::tests::plane_geometry(backing, [64, 32], 256, 4),
            row_pitch: 256,
            bytes_per_block: 4,
        }]),
        extent: [64, 32],
        format: F::Rgba8,
        allow_canonical_import: true,
    };
    let original = frames
        .capture(&mut cache, request.clone(), &mut descriptors)
        .unwrap();
    let mut bytes: Vec<_> = (0..8192).map(|i| (i % 251) as u8).collect();
    crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &bytes).unwrap();
    let before = cache.statistics().cpu_uploaded_bytes;
    let changed = frames
        .capture(&mut cache, request.clone(), &mut descriptors)
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes - before, 8192);
    let expected: Vec<_> = offsets
        .iter()
        .flat_map(|&offset| bytes[offset..offset + 16].iter().copied())
        .collect();
    // A later write affects only the second physical page. Acknowledging the
    // first refresh must not hide it or change the older immutable versions.
    bytes[4096..].fill(23);
    crate::residency::tests::cpu_write(&mut cache, &allocation, 4096, &bytes[4096..]).unwrap();
    let partial = frames
        .capture(&mut cache, request, &mut descriptors)
        .unwrap();
    let partial_expected: Vec<_> = offsets
        .iter()
        .flat_map(|&offset| bytes[offset..offset + 16].iter().copied())
        .collect();
    for (snapshot, expected) in [
        (&original, vec![7; 8192]),
        (&changed, expected),
        (&partial, partial_expected),
    ] {
        let view = Arc::clone(&frames.get(snapshot).unwrap().view);
        assert_eq!(
            read_in_layout(
                &mut cache,
                &view,
                [64, 32],
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            ),
            expected
        );
    }
    drop((original, changed, partial));
    drop(frames);
    drop(cache);
    drop(descriptors);
    monitor.check().unwrap();
}

#[test]
fn native_frame_snapshots_preserve_versions_crop_formats_and_every_transform() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let pool = Arc::new(CompilePool::new(2, 32).unwrap());
    let config = CacheConfiguration {
        workers: 2,
        pending_jobs: 32,
        ..Default::default()
    };
    let compiler = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let mut descriptors = DescriptorCache::new(&device, 64);
    let mut renderer = ScreenRenderer::new(&device).unwrap();
    let p = pipeline(&mut renderer, &compiler, 0);
    let packed = pipeline(&mut renderer, &compiler, 1);
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        64 * 1024 * 1024,
    )
    .unwrap();
    let mut frames = Frames::new();
    const ORDER: [[u8; 6]; 8] = [
        [1, 2, 3, 4, 5, 6],
        [2, 1, 4, 3, 6, 5],
        [5, 6, 3, 4, 1, 2],
        [6, 5, 4, 3, 2, 1],
        [5, 3, 1, 6, 4, 2],
        [6, 4, 2, 5, 3, 1],
        [1, 3, 5, 2, 4, 6],
        [2, 4, 6, 1, 3, 5],
    ];
    for format in [F::Rgba8, F::Rgbx8, F::Bgra8, F::Rgb565, F::Rgba4444] {
        let bytes_per_pixel = if matches!(format, F::Rgb565 | F::Rgba4444) {
            2
        } else {
            4
        };
        let (allocation, backing) = backing_allocation(&cache, 4 * 5 * bytes_per_pixel, 128, 199);
        let mut bytes = vec![199; 4 * 5 * bytes_per_pixel];
        let mut expected = [[0u8; 4]; 6];
        for y in 0..3 {
            for x in 0..2 {
                let n = (y * 2 + x + 1) as u8;
                let index = ((y + 1) * 4 + x + 1) * bytes_per_pixel;
                let pixel = match format {
                    F::Rgb565 => {
                        let raw =
                            (u16::from(n) << 11) | (u16::from(n + 10) << 5) | u16::from(n + 20);
                        bytes[index..index + 2].copy_from_slice(&raw.to_le_bytes());
                        [
                            ((u32::from(n) * 255 + 15) / 31) as u8,
                            ((u32::from(n + 10) * 255 + 31) / 63) as u8,
                            ((u32::from(n + 20) * 255 + 15) / 31) as u8,
                            255,
                        ]
                    }
                    F::Rgba4444 => {
                        let raw = u16::from(n)
                            | (u16::from(n + 1) << 4)
                            | (u16::from(n + 2) << 8)
                            | (7 << 12);
                        bytes[index..index + 2].copy_from_slice(&raw.to_le_bytes());
                        [n * 17, (n + 1) * 17, (n + 2) * 17, 119]
                    }
                    _ => {
                        let rgba = [n, n + 10, n + 20, 0];
                        let raw = if format == F::Bgra8 {
                            [rgba[2], rgba[1], rgba[0], rgba[3]]
                        } else {
                            rgba
                        };
                        bytes[index..index + 4].copy_from_slice(&raw);
                        [
                            rgba[0],
                            rgba[1],
                            rgba[2],
                            if format == F::Rgbx8 { 255 } else { 0 },
                        ]
                    }
                };
                expected[y * 2 + x] = pixel;
            }
        }
        crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &bytes).unwrap();
        let request = NativePresentationImage {
            inputs: (backing.clone()).pin(0, (backing.clone()).bytes()).unwrap(),
            planes: Arc::from([GuestImagePlane {
                mip_level: 0,
                array_layer: 0,
                aspect: GuestImageAspect::Color,
                footprint: crate::residency::tests::plane_geometry(
                    backing.clone(),
                    [4, 5],
                    (4 * bytes_per_pixel) as u64,
                    bytes_per_pixel as u32,
                ),
                row_pitch: (4 * bytes_per_pixel) as u64,
                bytes_per_block: bytes_per_pixel as u32,
            }]),
            extent: [4, 5],
            format,
            allow_canonical_import: true,
        };
        let image = frames
            .capture(&mut cache, request.clone(), &mut descriptors)
            .unwrap();
        // Later CPU contents create a new version; they cannot change this lease.
        crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &vec![0; bytes.len()])
            .unwrap();
        let changed = frames
            .capture(&mut cache, request, &mut descriptors)
            .unwrap();
        assert_ne!(image.lease().image(), changed.lease().image());
        for (bits, order) in ORDER.iter().enumerate() {
            let source = Source {
                image: image.clone(),
                crop: nixe_video::FrameCrop {
                    left: 1,
                    top: 1,
                    width: 2,
                    height: 3,
                },
                transform: nixe_video::FrameTransform {
                    flip_horizontal: bits & 1 != 0,
                    flip_vertical: bits & 2 != 0,
                    rotate_90_clockwise: bits & 4 != 0,
                },
                sequence: bits as u64,
            };
            let extent = source.dimensions();
            let native = cache
                .allocate_host_image(extent, vk::Format::R8G8B8A8_UNORM)
                .unwrap();
            let target = render::view(cache.device(), native, vk::Format::R8G8B8A8_UNORM).unwrap();
            let device = cache.device_mut();
            let slot = device.submission.begin(&mut device.dispatch).unwrap();
            let command = cache.device().submission.slots[slot].command;
            render::transition(
                cache.device_mut(),
                command,
                target.image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            renderer
                .screen(
                    &mut cache,
                    &mut descriptors,
                    if matches!(format, F::Rgba4444 | F::Rgb565) {
                        Arc::clone(&packed)
                    } else {
                        Arc::clone(&p)
                    },
                    frames.get(&source.image).unwrap(),
                    &source,
                    target.handle,
                    vk::Extent2D {
                        width: extent[0],
                        height: extent[1],
                    },
                    vk::Format::R8G8B8A8_UNORM,
                    false,
                )
                .unwrap();
            let actual = read(&mut cache, &target, extent);
            let expected: Vec<_> = order
                .iter()
                .flat_map(|n| expected[usize::from(*n - 1)])
                .collect();
            assert_eq!(actual, expected, "{format:?} transform={bits}");
        }
    }
    cache.flush().unwrap();
    drop(frames);
    drop(renderer);
    drop(descriptors);
    drop(compiler);
    drop(cache);
    monitor.check().unwrap();
}

#[test]
fn native_ui_texture_updates_blending_clipping_and_free() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let pool = Arc::new(CompilePool::new(2, 32).unwrap());
    let config = CacheConfiguration {
        workers: 2,
        pending_jobs: 32,
        ..Default::default()
    };
    let compiler = PipelineCompiler::new(&device, pool, &config).unwrap();
    let mut descriptors = DescriptorCache::new(&device, 64);
    let mut samplers = SamplerCache::new(&device, 64);
    let mut renderer = ScreenRenderer::new(&device).unwrap();
    let p = pipeline(&mut renderer, &compiler, 2);
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        64 * 1024 * 1024,
    )
    .unwrap();
    let id = egui::TextureId::Managed(0);
    for partial in [false, true] {
        let color = if partial {
            egui::Color32::GREEN
        } else {
            egui::Color32::WHITE
        };
        let image = egui::ColorImage::filled(if partial { [1, 1] } else { [2, 2] }, color);
        let delta = if partial {
            egui::epaint::ImageDelta::partial([0, 0], image, egui::TextureOptions::NEAREST)
        } else {
            egui::epaint::ImageDelta::full(image, egui::TextureOptions::NEAREST)
        };
        let tint = if partial {
            egui::Color32::from_rgba_premultiplied(128, 128, 128, 128)
        } else {
            egui::Color32::WHITE
        };
        let mut mesh = egui::Mesh::with_texture(id);
        mesh.add_rect_with_uv(
            egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(8., 8.)),
            egui::Rect::from_min_max(egui::pos2(0.25, 0.25), egui::pos2(0.25, 0.25)),
            tint,
        );
        let mut ui = UiFrame {
            primitives: vec![egui::ClippedPrimitive {
                clip_rect: egui::Rect::from_min_max(egui::pos2(2., 1.), egui::pos2(6., 7.)),
                primitive: egui::epaint::Primitive::Mesh(mesh),
            }],
            textures: Default::default(),
            pixels_per_point: if partial { 2. } else { 1. },
        };
        ui.textures.set.entry(id).or_default().push(delta);
        if partial {
            ui.textures.free.insert(id);
        }
        renderer
            .update_textures(&mut cache, &mut ui, &mut descriptors, &mut samplers)
            .unwrap();
        let image = cache
            .allocate_host_image([8, 8], vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        let target = render::view(cache.device(), image, vk::Format::R8G8B8A8_UNORM).unwrap();
        let device = cache.device_mut();
        let slot = device.submission.begin(&mut device.dispatch).unwrap();
        let command = cache.device().submission.slots[slot].command;
        render::transition(
            cache.device_mut(),
            command,
            target.image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        unsafe {
            cache.device_mut().dispatch.cmd_clear_color_image(
                command,
                target.image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0., 0., 0., 1.],
                },
                &[render::color_range()],
            );
        }
        render::transition(
            cache.device_mut(),
            command,
            target.image.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        renderer
            .ui(
                &mut cache,
                &mut descriptors,
                Arc::clone(&p),
                &mut ui,
                target.handle,
                vk::Extent2D {
                    width: 8,
                    height: 8,
                },
                vk::Format::R8G8B8A8_UNORM,
            )
            .unwrap();
        let pixels = read(&mut cache, &target, [8, 8]);
        for y in 0..8 {
            for x in 0..8 {
                let inside = if partial {
                    (4..8).contains(&x) && (2..8).contains(&y)
                } else {
                    (2..6).contains(&x) && (1..7).contains(&y)
                };
                let expected = if inside {
                    if partial {
                        [0, 128, 0, 255]
                    } else {
                        [255, 255, 255, 255]
                    }
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(
                    &pixels[(y * 8 + x) * 4..(y * 8 + x + 1) * 4],
                    expected,
                    "partial={partial}, {x},{y}"
                );
            }
        }
        assert!(ui.textures.free.is_empty());
    }
    // A coalesced frame still consumes texture frees, even without a draw.
    let id = egui::TextureId::Managed(42);
    let mut skipped = UiFrame {
        primitives: vec![],
        textures: Default::default(),
        pixels_per_point: 1.,
    };
    skipped
        .textures
        .set
        .entry(id)
        .or_default()
        .push(egui::epaint::ImageDelta::full(
            egui::ColorImage::filled([1, 1], egui::Color32::WHITE),
            egui::TextureOptions::NEAREST,
        ));
    skipped.textures.free.insert(id);
    renderer
        .update_textures(&mut cache, &mut skipped, &mut descriptors, &mut samplers)
        .unwrap();
    renderer.free_textures(&mut skipped, &mut descriptors);
    assert!(skipped.textures.free.is_empty());
    skipped
        .textures
        .set
        .entry(id)
        .or_default()
        .push(egui::epaint::ImageDelta::partial(
            [0, 0],
            egui::ColorImage::filled([1, 1], egui::Color32::WHITE),
            egui::TextureOptions::NEAREST,
        ));
    assert!(
        renderer
            .update_textures(&mut cache, &mut skipped, &mut descriptors, &mut samplers)
            .is_err()
    );
    skipped.textures.clear(); // The rejected delta is deliberately canceled.
    cache.flush().unwrap();
    drop(renderer);
    drop(cache);
    drop(descriptors);
    drop(samplers);
    drop(compiler);
    monitor.check().unwrap();
}

#[test]
fn opaque_scanout_requires_a_current_resident_producer() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        16 * 1024 * 1024,
    )
    .unwrap();
    let mut descriptors = DescriptorCache::new(cache.device(), 64);
    let mut frames = Frames::new();
    let (allocation, backing) = backing_allocation(&cache, 64, 128, 199);
    let request = NativePresentationImage {
        inputs: (backing.clone()).pin(0, (backing.clone()).bytes()).unwrap(),
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: crate::residency::tests::plane_geometry(backing.clone(), [4, 4], 16, 4),
            row_pitch: 16,
            bytes_per_block: 4,
        }]),
        extent: [4, 4],
        format: F::Rgba8,
        allow_canonical_import: false,
    };
    assert!(
        frames
            .capture(&mut cache, request.clone(), &mut descriptors)
            .is_err()
    );
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 0);
    let extent = vk::Extent3D {
        width: 4,
        height: 4,
        depth: 1,
    };
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(
                backing,
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::B8G8R8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[ImagePlane {
                    offset: 0,
                    row_pitch: 16,
                    mip_level: 0,
                    array_layer: 0,
                    extent,
                    aspect: vk::ImageAspectFlags::COLOR,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::B8G8R8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![ImagePlane {
                offset: 0,
                row_pitch: 16,
                mip_level: 0,
                array_layer: 0,
                extent,
                aspect: vk::ImageAspectFlags::COLOR,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            }],
        )
        .unwrap();
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [0., 0., 1., 1.],
            },
        )
        .unwrap();
    let uploaded = cache.statistics().cpu_uploaded_bytes;
    let snapshot = frames
        .capture(&mut cache, request.clone(), &mut descriptors)
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, uploaded);
    assert_eq!(cache.statistics().native_readback_bytes, 0);
    // A CPU store outside the texels can share their tracking page. It must
    // neither invalidate exact scanout authority nor cause an opaque import.
    crate::residency::tests::cpu_write(&mut cache, &allocation, 128, &[0x5a]).unwrap();
    let unchanged = frames
        .capture(&mut cache, request.clone(), &mut descriptors)
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, uploaded);
    drop(unchanged);
    // A real CPU demand first resolves the producer through the owner-local
    // visibility path; the later write then invalidates resident scanout.
    crate::residency::tests::cpu_read(&mut cache, request.planes[0].footprint.mapping());
    crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &[0; 64]).unwrap();
    assert!(
        frames
            .capture(&mut cache, request, &mut descriptors)
            .is_err()
    );
    assert!(frames.get(&snapshot).is_ok());
    drop(snapshot);
    drop(frames);
    drop(cache);
    drop(descriptors);
    monitor.check().unwrap();
}

#[test]
fn scanout_retargets_a_new_producer_and_ignores_undefined_padding() {
    use crate::residency::execution::tests::Utilities;
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        32 * 1024 * 1024,
    )
    .unwrap();
    let mut descriptors = DescriptorCache::new(cache.device(), 64);
    let mut frames = Frames::new();
    let (_allocation, whole) = backing_allocation(&cache, 96, 128, 199);
    let prefix = whole.slice(0, 64).unwrap();
    let mut request = NativePresentationImage {
        inputs: (prefix).pin(0, (prefix).bytes()).unwrap(),
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: crate::residency::tests::plane_geometry(prefix, [4, 4], 16, 4),
            row_pitch: 16,
            bytes_per_block: 4,
        }]),
        extent: [4, 4],
        format: F::Rgba8,
        allow_canonical_import: true,
    };
    let initial = frames
        .capture(&mut cache, request.clone(), &mut descriptors)
        .unwrap();
    let extent = vk::Extent3D {
        width: 4,
        height: 6,
        depth: 1,
    };
    let parent = cache
        .image_for_device_write(
            crate::residency::tests::image_geometry(
                whole,
                NativeImageDescription {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::B8G8R8A8_UNORM,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                },
                &[ImagePlane {
                    offset: 0,
                    row_pitch: 16,
                    mip_level: 0,
                    array_layer: 0,
                    extent,
                    aspect: vk::ImageAspectFlags::COLOR,
                    block_extent: vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    bytes_per_block: 4,
                }],
            ),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::B8G8R8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![ImagePlane {
                offset: 0,
                row_pitch: 16,
                mip_level: 0,
                array_layer: 0,
                extent,
                aspect: vk::ImageAspectFlags::COLOR,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            }],
        )
        .unwrap();
    cache
        .raster_fixture(
            parent,
            0,
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: 4,
                    height: 4,
                },
            },
            [0., 0., 1., 1.],
            crate::residency::AttachmentLoad::Clear([0., 0., 1., 1.]),
            crate::residency::AttachmentStore::Store,
        )
        .unwrap();
    let before = cache.statistics();
    // Same request/planes, new authoritative producer. Its uninitialized two
    // padding rows are neither consumed nor reconstructed through a buffer.
    let canonical = frames
        .capture(&mut cache, request.clone(), &mut descriptors)
        .unwrap();
    assert_eq!(before.gpu_copied_bytes, cache.statistics().gpu_copied_bytes);
    drop(canonical);
    request.allow_canonical_import = false;
    let snapshot = frames
        .capture(&mut cache, request, &mut descriptors)
        .unwrap();
    let after = cache.statistics();
    assert_eq!(before.cpu_uploaded_bytes, after.cpu_uploaded_bytes);
    assert_eq!(before.gpu_copied_bytes, after.gpu_copied_bytes);
    assert_eq!(after.native_readback_bytes, 0);
    assert_ne!(initial.lease().producer(), snapshot.lease().producer());
    assert_eq!(frames.get(&snapshot).unwrap().extent, [4, 4]);
    // Verify the raw size-compatible BGRA -> RGBA copy: blue BGRA bits become
    // red RGBA texels, without a color conversion or CPU reconstruction.
    let pool = Arc::new(CompilePool::new(2, 32).unwrap());
    let config = CacheConfiguration {
        workers: 2,
        pending_jobs: 32,
        ..Default::default()
    };
    let compiler = PipelineCompiler::new(cache.device(), pool, &config).unwrap();
    let mut renderer = ScreenRenderer::new(cache.device()).unwrap();
    let p = pipeline(&mut renderer, &compiler, 0);
    let native = cache
        .allocate_host_image([4, 4], vk::Format::R8G8B8A8_UNORM)
        .unwrap();
    let target = render::view(cache.device(), native, vk::Format::R8G8B8A8_UNORM).unwrap();
    let device = cache.device_mut();
    let slot = device.submission.begin(&mut device.dispatch).unwrap();
    let command = cache.device().submission.slots[slot].command;
    render::transition(
        cache.device_mut(),
        command,
        target.image.handle,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::NONE,
        vk::AccessFlags2::NONE,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    );
    let source = Source {
        image: snapshot,
        crop: nixe_video::FrameCrop {
            left: 0,
            top: 0,
            width: 4,
            height: 4,
        },
        transform: Default::default(),
        sequence: 1,
    };
    renderer
        .screen(
            &mut cache,
            &mut descriptors,
            p,
            frames.get(&source.image).unwrap(),
            &source,
            target.handle,
            vk::Extent2D {
                width: 4,
                height: 4,
            },
            vk::Format::R8G8B8A8_UNORM,
            false,
        )
        .unwrap();
    assert_eq!(
        read(&mut cache, &target, [4, 4]),
        [255, 0, 0, 255].repeat(16)
    );
    drop(initial);
    drop(source);
    drop(frames);
    drop(renderer);
    drop(cache);
    drop(descriptors);
    drop(compiler);
    monitor.check().unwrap();
}

#[test]
fn dropping_a_frame_pool_retains_an_in_flight_snapshot_destination() {
    mod gate {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/gate.rs"
        ));
    }
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut gate = gate::Gate::new(&device);
    device.submission.begin(&mut device.dispatch).unwrap();
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    gate.last_use = unsafe {
        device
            .submission
            .flush(device.queue, &mut device.dispatch, &waits, &[])
            .unwrap()
            .unwrap()
            .tick()
    };
    let mut descriptors = DescriptorCache::new(&device, 64);
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        16 * 1024 * 1024,
    )
    .unwrap();
    let mut frames = Frames::new();
    let (_allocation, backing) = backing_allocation(&cache, 64, 128, 199);
    let request = NativePresentationImage {
        inputs: (backing).pin(0, (backing).bytes()).unwrap(),
        planes: Arc::from([GuestImagePlane {
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: crate::residency::tests::plane_geometry(backing, [4, 4], 16, 4),
            row_pitch: 16,
            bytes_per_block: 4,
        }]),
        extent: [4, 4],
        format: F::Rgba8,
        allow_canonical_import: true,
    };
    let snapshot = frames
        .capture(&mut cache, request, &mut descriptors)
        .unwrap();
    let work = frames.get(&snapshot).unwrap().work;
    assert!(work.tick() > cache.device().queue_usage().completed_tick);
    drop(snapshot);
    drop(frames);
    monitor.check().unwrap();
    gate.signal();
    cache
        .device_mut()
        .wait_for_work(work, Duration::from_secs(5))
        .unwrap();
    drop(gate);
    drop(descriptors);
    drop(cache);
    monitor.check().unwrap();
}

#[test]
fn canceled_or_rejected_host_packets_dispose_unapplied_ui_deltas() {
    fn packet() -> HostFrame {
        let mut textures = egui::TexturesDelta::default();
        textures
            .set
            .entry(egui::TextureId::Managed(0))
            .or_default()
            .push(egui::epaint::ImageDelta::full(
                egui::ColorImage::filled([1, 1], egui::Color32::WHITE),
                egui::TextureOptions::LINEAR,
            ));
        HostFrame {
            frame: None,
            size: [8, 8],
            ui: Some(UiFrame {
                primitives: vec![],
                textures,
                pixels_per_point: 1.,
            }),
            screenshot: None,
        }
    }
    let client = PresentationClient::default();
    client.submit(packet()).unwrap();
    assert!(!client.ready().unwrap());
    assert!(client.submit(packet()).is_err());
    client.close();
    assert!(client.stopped());
    assert!(client.submit(packet()).is_err());
    let failed = PresentationClient::default();
    failed.fail("presentation failed".into());
    assert!(failed.ready().is_err());
    assert_eq!(failed.submit(packet()).unwrap_err(), "presentation failed");
}

mod snapshot_replacement {
    use super::*;
    use crate::VulkanError;

    mod gate {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/gate.rs"
        ));
    }

    fn hold_gpu(cache: &mut VulkanResidency) -> gate::Gate {
        let mut gate = gate::Gate::new(cache.device());
        let device = cache.device_mut();
        device.submission.begin(&mut device.dispatch).unwrap();
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(gate.handle)
            .value(1)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        gate.last_use = unsafe {
            device
                .submission
                .flush(device.queue, &mut device.dispatch, &waits, &[])
                .unwrap()
                .unwrap()
                .tick()
        };
        gate
    }

    fn cache(budget: u64) -> Option<VulkanResidency> {
        Some(
            VulkanResidency::new(
                nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
                support::device_for_test(ApiPreference::Auto)?,
                budget,
            )
            .unwrap(),
        )
    }
    fn allocation_charge() -> Option<u64> {
        let mut cache = cache(16 * 1024 * 1024)?;
        let image = cache
            .allocate_host_image([64, 64], vk::Format::R8G8B8A8_UNORM)
            .unwrap();
        let charge = cache.allocated_bytes();
        drop(image);
        Some(charge)
    }
    fn request(
        mapping: &Arc<PhysicalMapping>,
        extent: [u32; 2],
        format: F,
    ) -> NativePresentationImage {
        let pitch = u64::from(extent[0]) * 4;
        NativePresentationImage {
            inputs: mapping.pin(0, mapping.bytes()).unwrap(),
            planes: Arc::from([GuestImagePlane {
                mip_level: 0,
                array_layer: 0,
                aspect: GuestImageAspect::Color,
                footprint: crate::residency::tests::plane_geometry(
                    Arc::clone(mapping),
                    extent,
                    pitch,
                    4,
                ),
                row_pitch: pitch,
                bytes_per_block: 4,
            }]),
            extent,
            format,
            allow_canonical_import: true,
        }
    }
    fn finish(cache: &mut VulkanResidency, frames: &Frames, frame: &nixe_gpu::ResidentImage) {
        let work = frames.get(frame).unwrap().work;
        assert!(
            cache
                .device_mut()
                .wait_for_work(work, Duration::from_secs(5))
                .unwrap()
        );
    }
    fn pixels(cache: &mut VulkanResidency, frames: &Frames, frame: &nixe_gpu::ResidentImage) {
        let snapshot = frames.get(frame).unwrap();
        let view = Arc::clone(&snapshot.view);
        let extent = snapshot.extent;
        let data = read_in_layout(
            cache,
            &view,
            extent,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        assert_eq!(
            data,
            vec![0x51; extent[0] as usize * extent[1] as usize * 4]
        );
    }

    #[test]
    fn completed_snapshot_replacement_fits_the_same_budget_for_formats_and_extents() {
        let Some(charge) = allocation_charge() else {
            return;
        };
        let mut cache = cache(2 * charge).unwrap();
        let validation = cache.device().validation_monitor();
        let mut descriptors = DescriptorCache::new(cache.device(), 64);
        let mut frames = Frames::new();
        let (_allocation, mapping) = backing_allocation(&cache, 16384, 4096, 0x51);
        let mut previous = None;
        for (extent, format) in [
            ([64, 64], F::Rgba8),
            ([64, 64], F::Bgra8),
            ([128, 32], F::Bgra8),
            ([64, 64], F::Rgba8),
        ] {
            let frame = frames
                .capture(
                    &mut cache,
                    request(&mapping, extent, format),
                    &mut descriptors,
                )
                .unwrap();
            let id = frame.lease().image();
            if let Some(old) = previous {
                assert_ne!(id, old, "replacement cannot reuse an old image identity");
            }
            assert_eq!(
                id.slot, 0,
                "replace the vacant slot rather than growing the pool"
            );
            previous = Some(id);
            finish(&mut cache, &frames, &frame);
            pixels(&mut cache, &frames, &frame);
            assert!(cache.allocated_bytes() <= 2 * charge);
            drop(frame);
        }
        drop(frames);
        drop(descriptors);
        drop(cache);
        validation.check().unwrap();
    }

    #[test]
    fn failed_snapshot_replacement_keeps_other_leases_valid_and_the_slot_reusable() {
        let Some(charge) = allocation_charge() else {
            return;
        };
        let mut cache = cache(3 * charge).unwrap();
        let validation = cache.device().validation_monitor();
        let mut descriptors = DescriptorCache::new(cache.device(), 64);
        let mut frames = Frames::new();
        let (_allocation, mapping) = backing_allocation(&cache, 65536, 4096, 0x51);
        let first = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Rgba8),
                &mut descriptors,
            )
            .unwrap();
        finish(&mut cache, &frames, &first);
        let second = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Rgba8),
                &mut descriptors,
            )
            .unwrap();
        finish(&mut cache, &frames, &second);
        let old_id = second.lease().image();
        let old_view = Arc::downgrade(&frames.get(&second).unwrap().view);
        drop(second);
        // Exceed the budget in actual texel bytes, independently of the host's
        // allocation alignment or small-image granularity.
        let height = (3 * charge / 256 + 1) as u32;
        let (_large_allocation, large_mapping) =
            backing_allocation(&cache, height as usize * 256, 4096, 0x51);
        let failure = frames.capture(
            &mut cache,
            request(&large_mapping, [64, height], F::Bgra8),
            &mut descriptors,
        );
        assert!(matches!(
            failure,
            Err(VulkanError::ResidencyPressure { .. })
        ));
        assert!(
            old_view.upgrade().is_none(),
            "allocation failure must not retain the disposable snapshot"
        );
        assert!(
            frames.get(&first).is_ok(),
            "an active lease keeps its stable slot"
        );
        let replacement = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Bgra8),
                &mut descriptors,
            )
            .unwrap();
        assert_eq!(replacement.lease().image().slot, old_id.slot);
        assert_ne!(replacement.lease().image().generation, old_id.generation);
        finish(&mut cache, &frames, &replacement);
        pixels(&mut cache, &frames, &first);
        pixels(&mut cache, &frames, &replacement);
        drop(first);
        drop(replacement);
        drop(frames);
        drop(descriptors);
        drop(cache);
        validation.check().unwrap();
    }

    #[test]
    fn compatible_capture_retains_an_unleased_frame_until_its_gpu_work_finishes() {
        let Some(mut cache) = cache(16 * 1024 * 1024) else {
            return;
        };
        let validation = cache.device().validation_monitor();
        let mut descriptors = DescriptorCache::new(cache.device(), 64);
        let mut frames = Frames::new();
        let (_allocation, mapping) = backing_allocation(&cache, 16384, 4096, 0x51);
        let mut gate = hold_gpu(&mut cache);
        let first = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Rgba8),
                &mut descriptors,
            )
            .unwrap();
        let old_id = first.lease().image();
        let old_view = Arc::downgrade(&frames.get(&first).unwrap().view);
        drop(first);
        let second = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Rgba8),
                &mut descriptors,
            )
            .unwrap();
        assert_ne!(second.lease().image().slot, old_id.slot);
        assert!(
            old_view.upgrade().is_some(),
            "unfinished native work protects the old snapshot"
        );
        gate.signal();
        finish(&mut cache, &frames, &second);
        pixels(&mut cache, &frames, &second);
        drop(second);
        let third = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Rgba8),
                &mut descriptors,
            )
            .unwrap();
        assert_eq!(
            third.lease().image(),
            old_id,
            "completed compatible storage can be reused"
        );
        finish(&mut cache, &frames, &third);
        pixels(&mut cache, &frames, &third);
        drop(third);
        drop(gate);
        drop(frames);
        drop(descriptors);
        drop(cache);
        validation.check().unwrap();
    }

    #[test]
    fn pending_snapshot_replacement_fits_the_same_budget_for_formats_and_extents() {
        let Some(charge) = allocation_charge() else {
            return;
        };
        for (extent, format) in [([64, 64], F::Bgra8), ([128, 32], F::Rgba8)] {
            let mut cache = cache(2 * charge).unwrap();
            let validation = cache.device().validation_monitor();
            let mut descriptors = DescriptorCache::new(cache.device(), 64);
            let mut frames = Frames::new();
            let (_allocation, mapping) = backing_allocation(&cache, 16384, 4096, 0x51);
            let mut gate = hold_gpu(&mut cache);
            let first = frames
                .capture(
                    &mut cache,
                    request(&mapping, [64, 64], F::Rgba8),
                    &mut descriptors,
                )
                .unwrap();
            let old_id = first.lease().image();
            let old_view = Arc::downgrade(&frames.get(&first).unwrap().view);
            // The most recent use can still be recording and be later than
            // the snapshot's copy. Replacement must submit and wait for it.
            let last_use = cache.device_mut().work_ticket().unwrap();
            assert!(last_use.tick() > frames.get(&first).unwrap().work.tick());
            frames.used(&first, last_use).unwrap();
            assert!(last_use.tick() > cache.device().completed_tick());
            drop(first);
            let pending_view = old_view.clone();
            let signal = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                assert!(
                    pending_view.upgrade().is_some(),
                    "unfinished use retains its view"
                );
                gate.signal();
            });
            let replacement = frames.capture(
                &mut cache,
                request(&mapping, extent, format),
                &mut descriptors,
            );
            signal.join().unwrap();
            let replacement = replacement.unwrap();
            assert!(cache.device().completed_tick() >= last_use.tick());
            assert!(old_view.upgrade().is_none());
            assert_eq!(replacement.lease().image().slot, old_id.slot);
            assert_ne!(replacement.lease().image().generation, old_id.generation);
            finish(&mut cache, &frames, &replacement);
            pixels(&mut cache, &frames, &replacement);
            assert!(cache.allocated_bytes() <= 2 * charge);
            drop(replacement);
            drop(frames);
            drop(descriptors);
            drop(cache);
            validation.check().unwrap();
        }
    }

    #[test]
    fn format_change_preserves_a_leased_pending_snapshot() {
        let Some(mut cache) = cache(16 * 1024 * 1024) else {
            return;
        };
        let validation = cache.device().validation_monitor();
        let mut descriptors = DescriptorCache::new(cache.device(), 64);
        let mut frames = Frames::new();
        let (_allocation, mapping) = backing_allocation(&cache, 16384, 4096, 0x51);
        let mut gate = hold_gpu(&mut cache);
        let first = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Rgba8),
                &mut descriptors,
            )
            .unwrap();
        let second = frames
            .capture(
                &mut cache,
                request(&mapping, [64, 64], F::Bgra8),
                &mut descriptors,
            )
            .unwrap();
        assert_ne!(first.lease().image().slot, second.lease().image().slot);
        assert!(frames.get(&first).unwrap().work.tick() > cache.device().completed_tick());
        gate.signal();
        finish(&mut cache, &frames, &second);
        pixels(&mut cache, &frames, &first);
        pixels(&mut cache, &frames, &second);
        drop(first);
        drop(second);
        drop(gate);
        drop(frames);
        drop(descriptors);
        drop(cache);
        validation.check().unwrap();
    }
}
