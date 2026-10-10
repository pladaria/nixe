use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

#[path = "../test-support/hardware.rs"]
mod hardware;

#[path = "accelerated/compressed_textures.rs"]
mod compressed_textures;

#[path = "accelerated/resource_lifetime.rs"]
mod resource_lifetime;

#[path = "accelerated/cpu_overlap.rs"]
mod cpu_overlap;

#[path = "accelerated/segmented_uploads.rs"]
mod segmented_uploads;

#[path = "accelerated/multisample.rs"]
mod multisample;

#[path = "accelerated/texel_fetch.rs"]
mod texel_fetch;

#[path = "accelerated/multiple_targets.rs"]
mod multiple_targets;

#[path = "accelerated/compute.rs"]
mod compute;

use hardware::initialize_backend;
use nixe_gpu::{
    AttachmentLoad, AttachmentStore, BackendInstanceId, BackendResourceCreateInfo,
    BackendVisibilityRequester, BackingView, BlockLinearLayout, BufferDescription, BufferId,
    BufferRange, BufferRegion, BufferView, CapabilityRequirements, ClearOperation, ClearValue,
    CopyOperation, DrawArguments, DrawOperation, FrontendSubmissionId, GpuAllocationDescription,
    GpuAllocationId, GpuCommand, GpuOperation, ImageDescription, ImageDimension, ImageExtent,
    ImageFormat, ImageId, ImageKind, ImageMemoryLayout, ImageSubresourceRange, ImageView,
    NeutralBackendRuntime, OperationSubmission, PipelineDescription, PipelineId, PipelineKind,
    PreparedDraw, PresentationImageFormat, PresentationImageRequest, PrimitiveTopology,
    RenderAttachment, RenderPassAttachmentDescription, RenderPassDescription, RenderPassId,
    RenderPassOperation, ResourceDependency, SampleCount, ShaderDescription, ShaderId,
    ShaderInstruction, ShaderInterfaceElement, ShaderInterpolation, ShaderIoLocation, ShaderIr,
    ShaderOperation, ShaderPredicate, ShaderRegister, ShaderScalarType, ShaderSourceLocation,
    ShaderStage, Swizzle, VerifiedShaderIr, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexStepMode, ViewportTransform,
};
use nixe_gpu_wgpu::{WgpuBackendConfiguration, resident_texture};
use nixe_memory::{
    CanonicalAllocation, CanonicalBackingPage, CanonicalBackingRange, CanonicalBackingSegment,
    CanonicalBackingStore, ContentGeneration, CpuVisibilityRequest, DeviceVisibilityPoint,
    GuestPhysicalPageId, MappingGeneration, MemoryPermissions, NonCpuDeviceId,
    VisibilityCoordinatorError,
};

struct RuntimeOwner {
    runtime: Mutex<Box<dyn NeutralBackendRuntime>>,
}

fn backed_color_image(
    format: ImageFormat,
    width: u32,
    height: u32,
    pages: &[CanonicalBackingPage],
) -> (
    Vec<BackendResourceCreateInfo>,
    BackingView,
    ImageId,
    ImageSubresourceRange,
) {
    let id = ImageId::new(701);
    let allocation = GpuAllocationId::new(701);
    let size = pages.iter().map(|p| p.size() as u64).sum();
    let allocation_description = GpuAllocationDescription::new(size, 4).unwrap();
    let range = CanonicalBackingRange::new(
        pages
            .iter()
            .map(|p| {
                CanonicalBackingSegment::new(
                    p.clone(),
                    0,
                    p.size() as u64,
                    MemoryPermissions::READ_WRITE,
                    MappingGeneration::INITIAL,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let backing = BackingView::new(allocation, allocation_description, 0, range).unwrap();
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
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let layout = ImageMemoryLayout::PitchLinear {
        row_pitch: u64::from(width) * u64::from(format.plane_bytes_per_texel(0).unwrap()),
        layer_stride: size,
    };
    let view = ImageView::new(
        id,
        description,
        Swizzle::IDENTITY,
        vec![(subresources, layout, backing.clone())],
    )
    .unwrap();
    (
        vec![
            BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            },
            BackendResourceCreateInfo::Image {
                id,
                description,
                view: Some(view),
            },
        ],
        backing,
        id,
        subresources,
    )
}

fn color_clear_submission(
    id: ImageId,
    subresources: ImageSubresourceRange,
    format: ImageFormat,
    width: u32,
    height: u32,
    color: [f32; 4],
    serial: u64,
) -> OperationSubmission {
    OperationSubmission::new(
        FrontendSubmissionId::new(serial),
        vec![],
        vec![GpuOperation::new(
            GpuCommand::Clear(
                ClearOperation::image(
                    nixe_gpu::ImageRegion {
                        image: id,
                        subresources,
                        origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
                        extent: ImageExtent::new(width, height, 1).unwrap(),
                    },
                    ImageKind::Color,
                    format,
                    SampleCount::One,
                    ClearValue::Color(color),
                )
                .unwrap(),
            ),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap()
}

#[test]
fn resident_image_copy_preserves_extent_and_refreshes_cpu_writes() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(701),
        NonCpuDeviceId::new(701),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0; 13 * 8 * 4]);
    let (mut creations, backing, destination, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 13, 8, &[page]);
    let source = ImageId::new(702);
    let source_pixels = CanonicalAllocation::zeroed(16 * 8 * 4, 4).unwrap();
    let source_allocation = GpuAllocationId::new(702);
    let source_allocation_description = GpuAllocationDescription::new(16 * 8 * 4, 4).unwrap();
    let source_backing = BackingView::new(
        source_allocation,
        source_allocation_description,
        0,
        source_pixels
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    creations.push(BackendResourceCreateInfo::Allocation {
        id: source_allocation,
        description: source_allocation_description,
    });
    let source_description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(16, 8, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    creations.push(BackendResourceCreateInfo::Image {
        id: source,
        description: source_description,
        view: Some(
            ImageView::new(
                source,
                source_description,
                Swizzle::IDENTITY,
                vec![(
                    subresources,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: 16 * 4,
                        layer_stride: 16 * 8 * 4,
                    },
                    source_backing,
                )],
            )
            .unwrap(),
        ),
    });
    let mut operations = color_clear_submission(
        source,
        subresources,
        ImageFormat::Rgba8Unorm,
        16,
        8,
        [1.0, 0.0, 0.0, 1.0],
        701,
    )
    .operations()
    .to_vec();
    operations.extend_from_slice(
        color_clear_submission(
            source,
            subresources,
            ImageFormat::Rgba8Unorm,
            5,
            8,
            [0.0, 0.0, 1.0, 1.0],
            701,
        )
        .operations(),
    );
    let region = |image| nixe_gpu::ImageRegion {
        image,
        subresources,
        origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
        extent: ImageExtent::new(13, 8, 1).unwrap(),
    };
    operations.push(GpuOperation::new(
        GpuCommand::Copy(CopyOperation::ImageToImage {
            source: region(source),
            destination: region(destination),
        }),
        [],
        [],
        CapabilityRequirements::none(),
    ));
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &OperationSubmission::new(FrontendSubmissionId::new(701), vec![], operations).unwrap(),
        )
        .unwrap();
    let mut pixels = [0; 13 * 8 * 4];
    backing.range().read(0, &mut pixels).unwrap();
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        let expected = if index % 13 < 5 {
            [0, 0, 255, 255]
        } else {
            [255, 0, 0, 255]
        };
        assert_eq!(pixel, expected, "pixel {index}");
    }
    // Reading the smaller copy must upload a direct producer again after CPU
    // writes, even though the destination has no CPU changes of its own.
    let mut updated = [0; 16 * 8 * 4];
    for (index, pixel) in updated.chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(if index % 16 == 0 {
            &[255, 255, 0, 255]
        } else {
            &[0, 255, 0, 255]
        });
    }
    source_pixels.write(0, &updated).unwrap();
    runtime
        .runtime()
        .submit(
            &[],
            &[],
            &OperationSubmission::new(
                FrontendSubmissionId::new(702),
                vec![FrontendSubmissionId::new(701)],
                vec![GpuOperation::new(
                    GpuCommand::Copy(CopyOperation::ImageToImage {
                        source: region(source),
                        destination: region(destination),
                    }),
                    [],
                    [],
                    CapabilityRequirements::none(),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    backing.range().read(0, &mut pixels).unwrap();
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        let expected = if index % 13 == 0 {
            [255, 255, 0, 255]
        } else {
            [0, 255, 0, 255]
        };
        assert_eq!(pixel, expected, "updated pixel {index}");
    }
}

#[test]
fn partial_float32_clear_writes_unblended_values() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(701),
        NonCpuDeviceId::new(701),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0; 8 * 8 * 16]);
    let (creations, backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba32Float, 8, 8, &[page]);
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let mut operations = color_clear_submission(
        image,
        subresources,
        ImageFormat::Rgba32Float,
        6,
        6,
        [0.25, 0.5, 0.75, 1.0],
        700,
    )
    .operations()
    .to_vec();
    operations.extend_from_slice(
        color_clear_submission(
            image,
            subresources,
            ImageFormat::Rgba32Float,
            4,
            4,
            [2.0, -1.0, 0.5, 1.0],
            701,
        )
        .operations(),
    );
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &OperationSubmission::new(FrontendSubmissionId::new(701), vec![], operations).unwrap(),
        )
        .unwrap();
    let mut pixel = [0; 16];
    backing.range().read(0, &mut pixel).unwrap();
    let values = pixel
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, [2.0, -1.0, 0.5, 1.0]);
    backing.range().read(5 * 16, &mut pixel).unwrap();
    let values = pixel
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, [0.25, 0.5, 0.75, 1.0]);
    backing.range().read(7 * 16, &mut pixel).unwrap();
    assert_eq!(pixel, [0; 16]);
}

#[test]
fn partial_float16_clear_replaces_nan() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(703),
        NonCpuDeviceId::new(703),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0x00, 0x7e].repeat(8 * 8 * 4));
    let (creations, backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba16Float, 8, 8, &[page]);
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba16Float,
                4,
                4,
                [2.0, -1.0, 0.5, 1.0],
                703,
            ),
        )
        .unwrap();
    let mut pixel = [0; 8];
    backing.range().read(0, &mut pixel).unwrap();
    assert_eq!(pixel, [0x00, 0x40, 0x00, 0xbc, 0x00, 0x38, 0x00, 0x3c]);
}

#[test]
fn partial_clear_after_cpu_write_and_full_clear() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(704),
        NonCpuDeviceId::new(704),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0, 255, 0, 255].repeat(64));
    let (creations, backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 8, 8, std::slice::from_ref(&page));
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba8Unorm,
                4,
                4,
                [1.0, 0.0, 0.0, 1.0],
                704,
            ),
        )
        .unwrap();
    page.prepare_write().unwrap();
    let generation = page.content_generation();
    page.write_preflighted(0, &[0, 0, 255, 255], generation, generation.next().unwrap())
        .unwrap();
    let generation_after_cpu_write = page.content_generation();
    runtime
        .runtime()
        .submit(
            &[],
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba8Unorm,
                8,
                8,
                [1.0, 0.0, 0.0, 1.0],
                705,
            ),
        )
        .unwrap();
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
                [1.0, 1.0, 0.0, 1.0],
                706,
            ),
        )
        .unwrap();
    let mut pixel = [0; 4];
    assert_eq!(
        page.content_generation(),
        generation_after_cpu_write,
        "full clear must supersede dirty CPU bytes without a readback"
    );
    backing.range().read(7 * 4, &mut pixel).unwrap();
    assert_eq!(pixel, [255, 0, 0, 255]);
}

#[test]
fn cpu_authored_rgb565_is_converted_by_a_reusable_gpu_import() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x15);
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x15),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let mut bytes = vec![0_u8; 512];
    for (offset, pixel) in [(0, 0xf800_u16), (2, 0x07e0), (4, 0x001f), (6, 0xffff)] {
        bytes[offset..offset + 2].copy_from_slice(&pixel.to_le_bytes());
    }
    let page = initialized_page(&bytes);
    let allocation = GpuAllocationId::new(5);
    let description = GpuAllocationDescription::new(512, 4).unwrap();
    let source = backing(allocation, description, &page);
    let presentation = initialized.presentation_context();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let request = PresentationImageRequest {
        allow_canonical_import: true,
        cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(source.range()).unwrap(),
        backing: source,
        width: 4,
        height: 2,
        format: PresentationImageFormat::Rgb565,
        layout: ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            layer_stride: 512,
        }),
        row_pitch: 64,
    };

    let first = runtime
        .runtime()
        .acquire_presentable_image(request.clone())
        .unwrap();
    let second = runtime
        .runtime()
        .acquire_presentable_image(request.clone())
        .unwrap();
    let generation = page.content_generation();
    page.prepare_write().unwrap();
    page.write_preflighted(
        0,
        &0x07e0_u16.to_le_bytes(),
        generation,
        generation.next().unwrap(),
    )
    .unwrap();
    let updated = runtime
        .runtime()
        .acquire_presentable_image(request)
        .unwrap();

    assert_eq!(first.description().format(), ImageFormat::Rgba8Unorm);
    assert_eq!(second.description(), first.description());
    assert_eq!(updated.description(), first.description());
    assert!(resident_texture(&first).is_some());
    let texture = resident_texture(&updated).unwrap();
    let readback = presentation
        .device()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nixe presentation import test readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    let mut encoder =
        presentation
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Nixe presentation import test copy"),
            });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(2),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    presentation.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    presentation
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = readback.get_mapped_range(..).unwrap();
    assert_eq!(
        &mapped[0..16],
        &[
            0, 255, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255
        ]
    );
}

#[test]
fn partial_clear_initializes_a_new_image_from_a_device_authored_alias() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(705),
        NonCpuDeviceId::new(705),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0, 255, 0, 255].repeat(64));
    let (creations, backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 8, 8, &[page]);
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba8Unorm,
                8,
                8,
                [1.0, 0.0, 0.0, 1.0],
                705,
            ),
        )
        .unwrap();
    let BackendResourceCreateInfo::Image {
        description,
        view: Some(view),
        ..
    } = &creations[1]
    else {
        unreachable!()
    };
    let alias = ImageId::new(702);
    let alias_view = ImageView::new(
        alias,
        *description,
        Swizzle::IDENTITY,
        vec![(subresources, view.bindings()[0].layout(), backing.clone())],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(
            &[BackendResourceCreateInfo::Image {
                id: alias,
                description: *description,
                view: Some(alias_view),
            }],
            &[],
            &color_clear_submission(
                alias,
                subresources,
                ImageFormat::Rgba8Unorm,
                4,
                4,
                [1.0, 1.0, 0.0, 1.0],
                706,
            ),
        )
        .unwrap();
    let mut pixels = [0; 8 * 8 * 4];
    backing.range().read(0, &mut pixels).unwrap();
    for y in 0..8 {
        for x in 0..8 {
            let offset = (y * 8 + x) * 4;
            assert_eq!(
                &pixels[offset..offset + 4],
                if x < 4 && y < 4 {
                    &[255, 255, 0, 255]
                } else {
                    &[255, 0, 0, 255]
                }
            );
        }
    }
}

struct RuntimeRequester(Weak<RuntimeOwner>);

impl BackendVisibilityRequester for RuntimeRequester {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        self.0
            .upgrade()
            .ok_or_else(|| VisibilityCoordinatorError::new("test runtime owner stopped"))?
            .runtime()
            .make_cpu_visible(request)
            .map_err(|error| VisibilityCoordinatorError::new(error.to_string()))
    }
}

impl RuntimeOwner {
    fn new(runtime: Box<dyn NeutralBackendRuntime>) -> Arc<Self> {
        let owner = Arc::new(Self {
            runtime: Mutex::new(runtime),
        });
        let requester: Arc<dyn BackendVisibilityRequester> =
            Arc::new(RuntimeRequester(Arc::downgrade(&owner)));
        owner
            .runtime
            .try_lock()
            .expect("GPU owner must resolve its own visibility inline")
            .bind_visibility_requester(requester)
            .unwrap();
        owner
    }

    fn runtime(&self) -> MutexGuard<'_, Box<dyn NeutralBackendRuntime>> {
        self.runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn initialized_page(bytes: &[u8]) -> CanonicalBackingPage {
    let store = CanonicalBackingStore::allocate().unwrap();
    CanonicalBackingPage::initialized(
        &store,
        GuestPhysicalPageId::new(1),
        bytes,
        ContentGeneration::INITIAL,
    )
    .unwrap()
}

fn read_presented_rgba(
    presentation: &nixe_gpu_wgpu::WgpuPresentationContext,
    resident: &nixe_gpu::ResidentImage,
) -> Vec<u8> {
    let texture = resident_texture(resident).unwrap();
    let extent = resident.description().extent();
    let pitch = (extent.width * 4).div_ceil(256) * 256;
    let buffer = presentation
        .device()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("Presentation assertion"),
            size: u64::from(pitch * extent.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    let mut encoder = presentation
        .device()
        .create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(extent.height),
            },
        },
        wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: 1,
        },
    );
    presentation.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap()
    });
    presentation
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = buffer.get_mapped_range(..).unwrap();
    mapped
        .chunks_exact(pitch as usize)
        .flat_map(|row| row[..extent.width as usize * 4].iter().copied())
        .collect()
}

#[test]
fn opaque_presentation_requires_a_current_device_producer() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(707),
        NonCpuDeviceId::new(707),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0; 256]);
    let (creations, backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 8, 8, std::slice::from_ref(&page));
    let presentation = initialized.presentation_context();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let request = PresentationImageRequest {
        allow_canonical_import: false,
        cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(backing.range()).unwrap(),
        backing: backing.clone(),
        width: 8,
        height: 8,
        format: PresentationImageFormat::Rgba8,
        layout: ImageMemoryLayout::PitchLinear {
            row_pitch: 32,
            layer_stride: 256,
        },
        row_pitch: 32,
    };
    assert!(
        runtime
            .runtime()
            .acquire_presentable_image(request.clone())
            .is_err()
    );
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba8Unorm,
                8,
                8,
                [1.0, 0.0, 0.0, 1.0],
                707,
            ),
        )
        .unwrap();
    let destination = nixe_gpu::ImageRegion {
        image,
        subresources,
        origin: nixe_gpu::ImageOrigin { x: 2, y: 3, z: 0 },
        extent: ImageExtent::new(3, 1, 1).unwrap(),
    };
    runtime
        .runtime()
        .submit(
            &[],
            &[],
            &OperationSubmission::new(
                FrontendSubmissionId::new(708),
                vec![],
                vec![GpuOperation::new(
                    GpuCommand::UploadImage {
                        destination,
                        bytes: [0, 0, 255, 255].repeat(3).into(),
                    },
                    [],
                    [],
                    CapabilityRequirements::none(),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let resident = runtime
        .runtime()
        .acquire_presentable_image(request.clone())
        .unwrap();
    let mut expected = [255, 0, 0, 255].repeat(64);
    expected[(3 * 8 + 2) * 4..(3 * 8 + 5) * 4].copy_from_slice(&[0, 0, 255, 255].repeat(3));
    assert_eq!(read_presented_rgba(&presentation, &resident), expected);
    page.prepare_write().unwrap();
    let generation = page.content_generation();
    page.write_preflighted(0, &[0, 255, 0, 255], generation, generation.next().unwrap())
        .unwrap();
    let request = PresentationImageRequest {
        cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(backing.range()).unwrap(),
        ..request
    };
    assert!(
        runtime
            .runtime()
            .acquire_presentable_image(request)
            .is_err()
    );
}

#[test]
fn presentation_reconciles_mixed_cpu_and_device_pages() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(706),
        NonCpuDeviceId::new(706),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let pages = [initialized_page(&[0; 4096]), initialized_page(&[0; 4096])];
    let (creations, backing, image, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 64, 32, &pages);
    let presentation = initialized.presentation_context();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                image,
                subresources,
                ImageFormat::Rgba8Unorm,
                64,
                32,
                [1.0, 0.0, 0.0, 1.0],
                706,
            ),
        )
        .unwrap();
    backing.range().read(0, &mut [0; 4]).unwrap();
    assert!(matches!(
        backing.range().segments()[0].visibility_state(),
        nixe_memory::VisibilityState::Clean
    ));
    assert!(matches!(
        backing.range().segments()[1].visibility_state(),
        nixe_memory::VisibilityState::GpuNewer { .. }
    ));
    let request = PresentationImageRequest {
        allow_canonical_import: true,
        cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(backing.range()).unwrap(),
        backing,
        width: 64,
        height: 32,
        format: PresentationImageFormat::Rgba8,
        layout: ImageMemoryLayout::PitchLinear {
            row_pitch: 256,
            layer_stride: 8192,
        },
        row_pitch: 256,
    };
    let resident = runtime
        .runtime()
        .acquire_presentable_image(request)
        .unwrap();
    assert_eq!(
        read_presented_rgba(&presentation, &resident),
        [255, 0, 0, 255].repeat(64 * 32)
    );
}

fn accelerated_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn backing(
    allocation: GpuAllocationId,
    allocation_description: GpuAllocationDescription,
    page: &CanonicalBackingPage,
) -> BackingView {
    let segment = CanonicalBackingSegment::new(
        page.clone(),
        0,
        page.size() as u64,
        MemoryPermissions::READ_WRITE,
        MappingGeneration::INITIAL,
    )
    .unwrap();
    BackingView::new(
        allocation,
        allocation_description,
        0,
        CanonicalBackingRange::new(vec![segment]).unwrap(),
    )
    .unwrap()
}

#[test]
fn accelerated_submissions_remain_in_flight_until_cpu_demand() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x11);
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x11),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let page = initialized_page(&[0x5a; 64]);
    let allocation = GpuAllocationId::new(1);
    let allocation_description = GpuAllocationDescription::new(64, 4).unwrap();
    let backing = backing(allocation, allocation_description, &page);
    let buffer = BufferId::new(1);
    let creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description: BufferDescription::new(64).unwrap(),
            view: Some(
                BufferView::new(
                    buffer,
                    BufferDescription::new(64).unwrap(),
                    0,
                    backing.clone(),
                )
                .unwrap(),
            ),
        },
    ];
    let clear = ClearOperation::buffer(
        BufferRegion {
            buffer,
            range: BufferRange::new(0, 64).unwrap(),
        },
        0x1122_3344,
    )
    .unwrap();
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(1),
        vec![],
        vec![GpuOperation::new(
            GpuCommand::Clear(clear),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(&creations, &[], &submission)
        .unwrap();
    assert_eq!(
        page.visibility_state(),
        nixe_memory::VisibilityState::GpuNewer {
            device: device_id,
            visible_at: DeviceVisibilityPoint::new(1),
        }
    );
    let second_clear = ClearOperation::buffer(
        BufferRegion {
            buffer,
            range: BufferRange::new(0, 64).unwrap(),
        },
        0xaabb_ccdd,
    )
    .unwrap();
    let second = OperationSubmission::new(
        FrontendSubmissionId::new(2),
        vec![submission.id()],
        vec![GpuOperation::new(
            GpuCommand::Clear(second_clear),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    runtime.runtime().submit(&[], &[], &second).unwrap();
    assert_eq!(
        page.visibility_state(),
        nixe_memory::VisibilityState::GpuNewer {
            device: device_id,
            visible_at: DeviceVisibilityPoint::new(2),
        }
    );
    let mut bytes = [0xff; 64];
    backing.range().read(0, &mut bytes).unwrap();
    assert!(
        bytes
            .chunks_exact(4)
            .all(|word| word == 0xaabb_ccdd_u32.to_le_bytes())
    );
}

#[test]
fn demanded_buffer_visibility_downloads_only_the_written_page_interval() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x14);
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x14),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let canonical = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
    let allocation = GpuAllocationId::new(14);
    let allocation_description = GpuAllocationDescription::new(0x2000, 4).unwrap();
    let backing = BackingView::new(
        allocation,
        allocation_description,
        0,
        canonical
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let buffer = BufferId::new(14);
    let description = BufferDescription::new(0x2000).unwrap();
    let creations = [
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(BufferView::new(buffer, description, 0, backing.clone()).unwrap()),
        },
    ];
    let clear = ClearOperation::buffer(
        BufferRegion {
            buffer,
            range: BufferRange::new(0x1100, 4).unwrap(),
        },
        0x1122_3344,
    )
    .unwrap();
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(14),
        vec![],
        vec![GpuOperation::new(
            GpuCommand::Clear(clear),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(&creations, &[], &submission)
        .unwrap();

    let mut untouched = [0xff; 4];
    backing.range().read(0x100, &mut untouched).unwrap();
    assert_eq!(untouched, [0; 4]);
    let mut written = [0; 4];
    backing.range().read(0x1100, &mut written).unwrap();
    assert_eq!(written, 0x1122_3344_u32.to_le_bytes());
}

#[test]
fn neutral_runtime_reports_completion_separately_from_submission() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x12),
        NonCpuDeviceId::new(0x12),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let page = initialized_page(&[0x5a; 64]);
    let allocation = GpuAllocationId::new(12);
    let allocation_description = GpuAllocationDescription::new(64, 4).unwrap();
    let backing = backing(allocation, allocation_description, &page);
    let buffer = BufferId::new(12);
    let buffer_description = BufferDescription::new(64).unwrap();
    let creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description: buffer_description,
            view: Some(BufferView::new(buffer, buffer_description, 0, backing.clone()).unwrap()),
        },
    ];
    let clear = ClearOperation::buffer(
        BufferRegion {
            buffer,
            range: BufferRange::new(0, 64).unwrap(),
        },
        0x5566_7788,
    )
    .unwrap();
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(12),
        Vec::new(),
        vec![GpuOperation::new(
            GpuCommand::Clear(clear),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let mut owner = runtime.runtime();
    owner.submit(&creations, &[], &submission).unwrap();
    let completed = owner
        .wait_for_completion()
        .unwrap()
        .expect("submitted WGPU work has one completion");
    assert_eq!(completed.frontend(), submission.id());
    drop(owner);

    let mut bytes = [0; 64];
    backing.range().read(0, &mut bytes).unwrap();
    assert!(
        bytes
            .chunks_exact(4)
            .all(|word| word == 0x5566_7788_u32.to_le_bytes())
    );
}

#[test]
fn accelerated_copy_uploads_cpu_newer_input_before_backend_consumption() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x13);
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x13),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let source_page = initialized_page(&[0x3c; 64]);
    let destination_page = initialized_page(&[0; 64]);
    let description = GpuAllocationDescription::new(64, 4).unwrap();
    let source_allocation = GpuAllocationId::new(3);
    let destination_allocation = GpuAllocationId::new(4);
    let source_backing = backing(source_allocation, description, &source_page);
    let destination_backing = backing(destination_allocation, description, &destination_page);
    let source = BufferId::new(3);
    let destination = BufferId::new(4);
    let creations = [source_allocation, destination_allocation]
        .into_iter()
        .map(|id| BackendResourceCreateInfo::Allocation { id, description })
        .chain(
            [
                (source, source_backing.clone()),
                (destination, destination_backing.clone()),
            ]
            .into_iter()
            .map(|(id, backing)| BackendResourceCreateInfo::Buffer {
                id,
                description: BufferDescription::new(64).unwrap(),
                view: Some(
                    BufferView::new(id, BufferDescription::new(64).unwrap(), 0, backing).unwrap(),
                ),
            }),
        )
        .collect::<Vec<_>>();
    let copy = CopyOperation::buffer_to_buffer(
        BufferRegion {
            buffer: source,
            range: BufferRange::new(0, 64).unwrap(),
        },
        BufferRegion {
            buffer: destination,
            range: BufferRange::new(0, 64).unwrap(),
        },
    )
    .unwrap();
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(3),
        vec![],
        vec![GpuOperation::new(
            GpuCommand::Copy(copy),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(&creations, &[], &submission)
        .unwrap();
    let mut bytes = [0; 64];
    destination_backing.range().read(0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x3c; 64]);
}

#[test]
fn many_buffer_domains_upload_before_cache_eviction() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x14),
        NonCpuDeviceId::new(0x14),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let source_page = initialized_page(&[0x3c; 1024]);
    let destination_page = initialized_page(&[0; 1024]);
    let description = GpuAllocationDescription::new(1024, 4).unwrap();
    let source_allocation = GpuAllocationId::new(3);
    let destination_allocation = GpuAllocationId::new(4);
    let source_backing = backing(source_allocation, description, &source_page);
    let destination_backing = backing(destination_allocation, description, &destination_page);
    let source = BufferId::new(3);
    let destination = BufferId::new(4);
    let creations = [source_allocation, destination_allocation]
        .into_iter()
        .map(|id| BackendResourceCreateInfo::Allocation { id, description })
        .chain(
            [
                (source, source_backing.clone()),
                (destination, destination_backing.clone()),
            ]
            .into_iter()
            .map(|(id, backing)| BackendResourceCreateInfo::Buffer {
                id,
                description: BufferDescription::new(1024).unwrap(),
                view: Some(
                    BufferView::new(id, BufferDescription::new(1024).unwrap(), 0, backing).unwrap(),
                ),
            }),
        )
        .collect::<Vec<_>>();
    // Distinct, nonadjacent domains exceed the retained-domain cache limit.
    // Every earlier observation must survive until its host copy is encoded.
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(4),
        vec![],
        (0..65)
            .map(|index| {
                let range = BufferRange::new(index * 8, 4).unwrap();
                GpuOperation::new(
                    GpuCommand::Copy(
                        CopyOperation::buffer_to_buffer(
                            BufferRegion {
                                buffer: source,
                                range,
                            },
                            BufferRegion {
                                buffer: destination,
                                range,
                            },
                        )
                        .unwrap(),
                    ),
                    [],
                    [],
                    CapabilityRequirements::none(),
                )
            })
            .collect(),
    )
    .unwrap();
    runtime
        .runtime()
        .submit(&creations, &[], &submission)
        .unwrap();
    let mut bytes = [0; 1024];
    destination_backing.range().read(0, &mut bytes).unwrap();
    for index in 0..65 {
        assert_eq!(&bytes[index * 8..index * 8 + 4], &[0x3c; 4]);
        assert_eq!(&bytes[index * 8 + 4..index * 8 + 8], &[0; 4]);
    }
    assert_eq!(&bytes[520..], &[0; 504]);
}

#[test]
fn partial_image_clear_preserves_texels_and_reuses_device_authored_presentation() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x16);
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x16),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let presentation = initialized.presentation_context();
    let initial = [0, 255, 0, 255].repeat((WIDTH * HEIGHT) as usize);
    let page = initialized_page(&initial);
    let allocation = GpuAllocationId::new(0x16);
    let presentation_allocation = GpuAllocationId::new(0x18);
    let allocation_description =
        GpuAllocationDescription::new(u64::from(WIDTH * HEIGHT * 4), 4).unwrap();
    let image_backing = backing(allocation, allocation_description, &page);
    let presentation_backing = backing(presentation_allocation, allocation_description, &page);
    let image = ImageId::new(0x16);
    let image_description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Allocation {
            id: presentation_allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Image {
            id: image,
            description: image_description,
            view: Some(
                ImageView::new(
                    image,
                    image_description,
                    Swizzle::IDENTITY,
                    vec![(
                        subresources,
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: u64::from(WIDTH * 4),
                            layer_stride: u64::from(WIDTH * HEIGHT * 4),
                        },
                        image_backing.clone(),
                    )],
                )
                .unwrap(),
            ),
        },
    ];
    let clear = ClearOperation::image(
        nixe_gpu::ImageRegion {
            image,
            subresources,
            origin: nixe_gpu::ImageOrigin { x: 2, y: 3, z: 0 },
            extent: ImageExtent::new(3, 2, 1).unwrap(),
        },
        ImageKind::Color,
        ImageFormat::Rgba8Unorm,
        SampleCount::One,
        ClearValue::Color([1.0, 0.0, 0.0, 1.0]),
    )
    .unwrap();
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(0x16),
        vec![],
        vec![GpuOperation::new(
            GpuCommand::Clear(clear),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(&creations, &[], &submission)
        .unwrap();

    let mut pixels = vec![0; initial.len()];
    image_backing.range().read(0, &mut pixels).unwrap();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = ((y * WIDTH + x) * 4) as usize;
            let pixel = &pixels[offset..offset + 4];
            if (2..5).contains(&x) && (3..5).contains(&y) {
                assert_eq!(pixel, [255, 0, 0, 255], "pixel ({x}, {y})");
            } else {
                assert_eq!(pixel, [0, 255, 0, 255], "pixel ({x}, {y})");
            }
        }
    }

    let presentation_request = |backing: BackingView, cpu_writes| PresentationImageRequest {
        allow_canonical_import: true,
        backing,
        width: WIDTH,
        height: HEIGHT,
        format: PresentationImageFormat::Rgba8,
        layout: ImageMemoryLayout::PitchLinear {
            row_pitch: u64::from(WIDTH * 4),
            layer_stride: u64::from(WIDTH * HEIGHT * 4),
        },
        row_pitch: WIDTH * 4,
        cpu_writes,
    };
    let first_presentation_writes =
        nixe_memory::CanonicalCpuWriteDependency::capture(image_backing.range()).unwrap();
    let first_resident = runtime
        .runtime()
        .acquire_presentable_image(presentation_request(
            image_backing.clone(),
            first_presentation_writes.clone(),
        ))
        .unwrap();
    assert!(resident_texture(&first_resident).is_some());

    let generation = page.content_generation();
    page.prepare_write().unwrap();
    page.write_preflighted(0, &[0, 0, 255, 255], generation, generation.next().unwrap())
        .unwrap();
    assert!(!first_presentation_writes.remains_current());
    let second_presentation_writes =
        nixe_memory::CanonicalCpuWriteDependency::capture(presentation_backing.range()).unwrap();

    let second_clear = ClearOperation::image(
        nixe_gpu::ImageRegion {
            image,
            subresources,
            origin: nixe_gpu::ImageOrigin { x: 4, y: 4, z: 0 },
            extent: ImageExtent::new(2, 2, 1).unwrap(),
        },
        ImageKind::Color,
        ImageFormat::Rgba8Unorm,
        SampleCount::One,
        ClearValue::Color([1.0, 1.0, 0.0, 1.0]),
    )
    .unwrap();
    let second_submission = OperationSubmission::new(
        FrontendSubmissionId::new(0x18),
        vec![submission.id()],
        vec![GpuOperation::new(
            GpuCommand::Clear(second_clear),
            [],
            [],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(&[], &[], &second_submission)
        .unwrap();
    let resident = runtime
        .runtime()
        .acquire_presentable_image(presentation_request(
            presentation_backing,
            second_presentation_writes,
        ))
        .unwrap();
    let pixels = read_presented_rgba(&presentation, &resident);
    assert_eq!(&pixels[..4], &[0, 0, 255, 255]);
    assert_eq!(
        &pixels[(4 * 8 + 4) * 4..(4 * 8 + 5) * 4],
        &[255, 255, 0, 255]
    );
}

#[test]
fn partial_depth_stencil_clear_modes_are_accepted() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x17),
        NonCpuDeviceId::new(0x17),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let image = ImageId::new(0x17);
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(8, 8, 1).unwrap(),
        ImageFormat::Depth24UnormStencil8Uint,
        ImageKind::DepthStencil,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let clear = |origin, extent, value| {
        GpuOperation::new(
            GpuCommand::Clear(
                ClearOperation::image(
                    nixe_gpu::ImageRegion {
                        image,
                        subresources,
                        origin,
                        extent,
                    },
                    ImageKind::DepthStencil,
                    ImageFormat::Depth24UnormStencil8Uint,
                    SampleCount::One,
                    value,
                )
                .unwrap(),
            ),
            [],
            [],
            CapabilityRequirements::none(),
        )
    };
    let operations = vec![
        clear(
            nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
            ImageExtent::new(8, 8, 1).unwrap(),
            ClearValue::DepthStencil {
                depth: 1.0,
                stencil: 0,
            },
        ),
        clear(
            nixe_gpu::ImageOrigin { x: 1, y: 1, z: 0 },
            ImageExtent::new(6, 6, 1).unwrap(),
            ClearValue::Depth(0.25),
        ),
        clear(
            nixe_gpu::ImageOrigin { x: 2, y: 2, z: 0 },
            ImageExtent::new(4, 4, 1).unwrap(),
            ClearValue::Stencil(0x5a),
        ),
        clear(
            nixe_gpu::ImageOrigin { x: 3, y: 3, z: 0 },
            ImageExtent::new(2, 2, 1).unwrap(),
            ClearValue::DepthStencil {
                depth: 0.75,
                stencil: 0xa5,
            },
        ),
    ];
    let submission =
        OperationSubmission::new(FrontendSubmissionId::new(0x17), vec![], operations).unwrap();
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let mut owner = runtime.runtime();
    owner
        .submit(
            &[BackendResourceCreateInfo::Image {
                id: image,
                description,
                view: None,
            }],
            &[],
            &submission,
        )
        .unwrap();
    assert!(owner.wait_for_completion().unwrap().is_some());
}

#[test]
fn accelerated_triangle_draw_matches_geometry_clear_and_interpolation_contract() {
    accelerated_polygon_draw(
        PrimitiveTopology::Triangles,
        ShaderInterpolation::Perspective,
        false,
        VertexFormat::Float32x3,
    );
}

#[test]
fn accelerated_quads_cover_both_triangles_with_the_last_vertex_color() {
    accelerated_polygon_draw(
        PrimitiveTopology::Quads,
        ShaderInterpolation::Constant,
        false,
        VertexFormat::Float32x3,
    );
}

#[test]
fn accelerated_quads_interpolate_over_the_zero_two_diagonal() {
    for interpolation in [
        ShaderInterpolation::Perspective,
        ShaderInterpolation::ScreenLinear,
    ] {
        accelerated_polygon_draw(
            PrimitiveTopology::Quads,
            interpolation,
            false,
            VertexFormat::Float32x3,
        );
    }
}

#[test]
fn accelerated_quads_preserve_mixed_smooth_and_constant_attributes() {
    accelerated_polygon_draw(
        PrimitiveTopology::Quads,
        ShaderInterpolation::Perspective,
        true,
        VertexFormat::Float32x3,
    );
}

#[test]
fn accelerated_flat_quads_fetch_native_and_scaled_color_formats() {
    for format in [
        VertexFormat::Unorm8x4,
        VertexFormat::Snorm8x4,
        VertexFormat::Unorm16x4,
        VertexFormat::Snorm16x4,
        VertexFormat::Float16x4,
        VertexFormat::Unorm10_10_10_2,
        VertexFormat::Sscaled {
            width: nixe_gpu::VertexComponentWidth::Bits16,
            components: nixe_gpu::VertexComponentCount::Four,
        },
    ] {
        accelerated_polygon_draw(
            PrimitiveTopology::Quads,
            ShaderInterpolation::Perspective,
            true,
            format,
        );
    }
}

fn accelerated_polygon_draw(
    topology: PrimitiveTopology,
    interpolation: ShaderInterpolation,
    mixed: bool,
    color_format: VertexFormat,
) {
    accelerated_polygon_color_draw(
        topology,
        interpolation,
        mixed,
        color_format,
        None,
        None,
        false,
        None,
    );
}

#[test]
fn accelerated_blending_and_write_masks_preserve_destination_components() {
    use nixe_gpu::{
        BlendComponent, BlendFactor as F, BlendOperation as O, ColorBlendState, ColorOutputState,
        ColorWriteMask,
    };
    let component = |operation, source, destination| BlendComponent {
        operation,
        source,
        destination,
    };
    for (output, expected) in [
        (
            ColorOutputState {
                blend: Some(ColorBlendState {
                    color: component(O::Add, F::SourceAlpha, F::OneMinusSourceAlpha),
                    alpha: component(O::Add, F::One, F::Zero),
                }),
                write_mask: ColorWriteMask::new(true, false, true, true),
            },
            [204, 77, 19, 128],
        ),
        (
            ColorOutputState {
                blend: None,
                write_mask: ColorWriteMask::new(false, true, false, false),
            },
            [51, 255, 77, 255],
        ),
        (
            ColorOutputState {
                blend: Some(ColorBlendState {
                    color: component(O::ReverseSubtract, F::One, F::One),
                    alpha: component(O::Max, F::Zero, F::Zero),
                }),
                write_mask: ColorWriteMask::ALL,
            },
            [0, 0, 77, 255],
        ),
        (
            ColorOutputState {
                blend: Some(ColorBlendState {
                    color: component(O::Min, F::Zero, F::Zero),
                    alpha: component(O::Min, F::Zero, F::Zero),
                }),
                write_mask: ColorWriteMask::ALL,
            },
            [51, 77, 0, 128],
        ),
    ] {
        accelerated_polygon_color_draw(
            PrimitiveTopology::Quads,
            ShaderInterpolation::Constant,
            false,
            VertexFormat::Float32x3,
            Some((output, expected)),
            None,
            false,
            None,
        );
    }
}

#[test]
fn accelerated_polygon_facing_preserves_winding_for_triangles_and_quads() {
    use nixe_gpu::{CullMode as C, FrontFace as F};
    for topology in [PrimitiveTopology::Triangles, PrimitiveTopology::Quads] {
        for (front, cull, visible) in [
            (F::CounterClockwise, C::Back, true),
            (F::CounterClockwise, C::Front, false),
            (F::Clockwise, C::Back, false),
            (F::Clockwise, C::Front, true),
        ] {
            accelerated_polygon_color_draw(
                topology,
                ShaderInterpolation::Perspective,
                false,
                VertexFormat::Float32x3,
                None,
                Some((front, cull, visible)),
                false,
                None,
            );
        }
    }
}

#[test]
fn accelerated_positive_y_viewport_preserves_interpolation_and_polygon_facing() {
    for topology in [PrimitiveTopology::Triangles, PrimitiveTopology::Quads] {
        accelerated_polygon_color_draw(
            topology,
            ShaderInterpolation::Perspective,
            false,
            VertexFormat::Float32x3,
            None,
            Some((
                nixe_gpu::FrontFace::CounterClockwise,
                nixe_gpu::CullMode::Back,
                true,
            )),
            true,
            None,
        );
    }
}

#[derive(Clone, Copy)]
enum ScissorScenario {
    Partial,
    Empty,
    Reset,
}

#[test]
fn accelerated_draw_scissor_clips_fragments_and_resets_between_draws() {
    for scenario in [
        ScissorScenario::Partial,
        ScissorScenario::Empty,
        ScissorScenario::Reset,
    ] {
        accelerated_polygon_color_draw(
            PrimitiveTopology::Triangles,
            ShaderInterpolation::Perspective,
            false,
            VertexFormat::Float32x3,
            None,
            None,
            false,
            Some(scenario),
        );
    }
}

#[test]
fn accelerated_triangle_fans_preserve_interpolation_and_last_provoking_vertex() {
    for interpolation in [
        ShaderInterpolation::Constant,
        ShaderInterpolation::Perspective,
        ShaderInterpolation::ScreenLinear,
    ] {
        accelerated_polygon_color_draw(
            PrimitiveTopology::TriangleFan,
            interpolation,
            false,
            VertexFormat::Float32x3,
            None,
            None,
            false,
            None,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn accelerated_polygon_color_draw(
    topology: PrimitiveTopology,
    interpolation: ShaderInterpolation,
    mixed: bool,
    color_format: VertexFormat,
    color_output: Option<(nixe_gpu::ColorOutputState, [u8; 4])>,
    facing: Option<(nixe_gpu::FrontFace, nixe_gpu::CullMode, bool)>,
    positive_y: bool,
    scissor: Option<ScissorScenario>,
) {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x12);
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(0x12),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    let page = initialized_page(&vec![0; (WIDTH * HEIGHT * 4) as usize]);
    let allocation = GpuAllocationId::new(2);
    let allocation_description =
        GpuAllocationDescription::new(u64::from(WIDTH * HEIGHT * 4), 4).unwrap();
    let mut creations = vec![BackendResourceCreateInfo::Allocation {
        id: allocation,
        description: allocation_description,
    }];
    let image_backing = backing(allocation, allocation_description, &page);
    let image = ImageId::new(2);
    let image_description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    creations.push(BackendResourceCreateInfo::Image {
        id: image,
        description: image_description,
        view: Some(
            ImageView::new(
                image,
                image_description,
                Swizzle::IDENTITY,
                vec![(
                    subresources,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: u64::from(WIDTH * 4),
                        layer_stride: u64::from(WIDTH * HEIGHT * 4),
                    },
                    image_backing.clone(),
                )],
            )
            .unwrap(),
        ),
    });

    let square = matches!(
        topology,
        PrimitiveTopology::Quads | PrimitiveTopology::TriangleFan
    );
    let mut vertex_bytes = Vec::new();
    let triangle_vertices = [
        -0.5_f32, -0.5, 0.0, 1.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        1.0,
    ];
    let quad_vertices = [
        // An unused prefix tests a nonzero, non-quad-aligned first vertex.
        0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 1.0,
        0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, -0.5, 0.5, 0.0, 1.0, 1.0, 0.0,
        // An incomplete trailing quad must not add any coverage.
        -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ];
    // Repeat the quad at a different first-vertex alignment. Both draws share
    // one prepared pipeline; the second draw must update its immediate value.
    let repeated_quads = [
        &quad_vertices[..30],
        &quad_vertices[..6],
        &quad_vertices[6..],
    ]
    .concat();
    let vertices: &[f32] = if square {
        &repeated_quads
    } else {
        &triangle_vertices
    };
    for (index, vertex) in vertices.chunks_exact(6).enumerate() {
        // Different clip W values distinguish perspective from screen-linear
        // interpolation while keeping the projected square at pixels 8..24.
        let w = if square && index % 5 != 0 && index < 10 {
            [1.0_f32, 2.0, 4.0, 2.0][index % 5 - 1]
        } else {
            1.0
        };
        for component in [vertex[0] * w, vertex[1] * w, vertex[2] * w, w] {
            vertex_bytes.extend_from_slice(&component.to_le_bytes());
        }
        let color = [vertex[3], vertex[4], vertex[5], 1.0];
        match color_format {
            VertexFormat::Float32x3 => {
                for component in &color[..3] {
                    vertex_bytes.extend_from_slice(&component.to_le_bytes());
                }
            }
            VertexFormat::Unorm8x4 | VertexFormat::Snorm8x4 => {
                let one = if color_format == VertexFormat::Unorm8x4 {
                    255.0
                } else {
                    127.0
                };
                vertex_bytes.extend(color.map(|value| (value * one) as u8));
            }
            VertexFormat::Unorm16x4
            | VertexFormat::Snorm16x4
            | VertexFormat::Float16x4
            | VertexFormat::Sscaled { .. } => {
                let one = match color_format {
                    VertexFormat::Unorm16x4 => 65535,
                    VertexFormat::Snorm16x4 => 32767,
                    VertexFormat::Float16x4 => 0x3c00,
                    _ => 1,
                };
                for component in color {
                    vertex_bytes.extend_from_slice(&((component as u16) * one).to_le_bytes());
                }
            }
            VertexFormat::Unorm10_10_10_2 => {
                let packed = (color[0] as u32 * 1023)
                    | ((color[1] as u32 * 1023) << 10)
                    | ((color[2] as u32 * 1023) << 20)
                    | (3 << 30);
                vertex_bytes.extend_from_slice(&packed.to_le_bytes());
            }
            _ => unreachable!("test color format"),
        }
    }
    let vertex_size = vertex_bytes.len() as u64;
    let vertex_page = initialized_page(&vertex_bytes);
    let vertex_allocation = GpuAllocationId::new(3);
    let vertex_allocation_description = GpuAllocationDescription::new(vertex_size, 4).unwrap();
    creations.push(BackendResourceCreateInfo::Allocation {
        id: vertex_allocation,
        description: vertex_allocation_description,
    });
    let vertex_backing = backing(
        vertex_allocation,
        vertex_allocation_description,
        &vertex_page,
    );
    let vertex_buffer = BufferId::new(3);
    creations.push(BackendResourceCreateInfo::Buffer {
        id: vertex_buffer,
        description: BufferDescription::new(vertex_size).unwrap(),
        view: Some(
            BufferView::new(
                vertex_buffer,
                BufferDescription::new(vertex_size).unwrap(),
                0,
                vertex_backing.clone(),
            )
            .unwrap(),
        ),
    });

    let vertex = ShaderId::new(1);
    let fragment = ShaderId::new(2);
    creations.push(BackendResourceCreateInfo::Shader {
        id: vertex,
        description: ShaderDescription {
            stage: ShaderStage::Vertex,
        },
        module: interpolated_vertex_module(interpolation, mixed),
    });
    creations.push(BackendResourceCreateInfo::Shader {
        id: fragment,
        description: ShaderDescription {
            stage: ShaderStage::Fragment,
        },
        module: interpolated_fragment_module(
            interpolation,
            mixed,
            if color_output.is_some() { 0.5 } else { 1.0 },
        ),
    });
    let pipeline = PipelineId::new(1);
    creations.push(BackendResourceCreateInfo::Pipeline {
        id: pipeline,
        description: PipelineDescription {
            kind: PipelineKind::Graphics,
        },
    });
    let render_pass = RenderPassId::new(1);
    let render_pass_description =
        RenderPassDescription::new(vec![RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
        }])
        .unwrap();
    creations.push(BackendResourceCreateInfo::RenderPass {
        id: render_pass,
        description: render_pass_description.clone(),
    });
    let attachment = RenderAttachment {
        image,
        subresources,
        kind: ImageKind::Color,
        format: ImageFormat::Rgba8Unorm,
        samples: SampleCount::One,
        load: AttachmentLoad::Clear(ClearValue::Color([0.2, 0.3, 0.3, 1.0])),
        store: AttachmentStore::Store,
    };
    let begin =
        RenderPassOperation::begin(render_pass, render_pass_description, vec![attachment]).unwrap();
    let prepared = PreparedDraw::new(
        pipeline,
        render_pass,
        topology,
        vec![],
        vec![
            VertexBufferLayout::new(
                BufferRegion {
                    buffer: vertex_buffer,
                    range: BufferRange::new(0, vertex_size).unwrap(),
                },
                16 + color_format.size(),
                VertexStepMode::Vertex,
                vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: color_format,
                        offset: 16,
                        shader_location: 1,
                    },
                ],
            )
            .unwrap(),
        ],
        None,
    )
    .unwrap()
    .with_viewport_transform(
        ViewportTransform::new(
            [16.0, if positive_y { 16.0 } else { -16.0 }, 0.5],
            [16.0, 16.0, 0.5],
            [0.0, 1.0],
        )
        .unwrap(),
    );
    let mut prepared = prepared;
    prepared.scissor = match scissor {
        Some(ScissorScenario::Partial) => Some(nixe_gpu::ScissorRect {
            x: 8,
            y: 8,
            width: 16,
            height: 16,
        }),
        Some(ScissorScenario::Empty) => Some(nixe_gpu::ScissorRect {
            x: 8,
            y: 8,
            width: 0,
            height: 16,
        }),
        _ => None,
    };
    if let Some((front, cull, _)) = facing {
        prepared.front_face = if positive_y {
            match front {
                nixe_gpu::FrontFace::Clockwise => nixe_gpu::FrontFace::CounterClockwise,
                nixe_gpu::FrontFace::CounterClockwise => nixe_gpu::FrontFace::Clockwise,
            }
        } else {
            front
        };
        prepared.cull_mode = cull;
    }
    if let Some((output, _)) = color_output {
        prepared.color_outputs[0] = output;
    }
    let mut draw = DrawOperation::new(
        Arc::new(prepared),
        DrawArguments::NonIndexed {
            first_vertex: u32::from(square),
            vertex_count: if topology == PrimitiveTopology::TriangleFan {
                4
            } else if square {
                7
            } else {
                3
            },
            first_instance: 2,
            instance_count: 2,
        },
    )
    .unwrap();
    let second_draw = if square {
        DrawOperation::new(
            Arc::clone(&draw.prepared),
            DrawArguments::NonIndexed {
                first_vertex: 6,
                vertex_count: if topology == PrimitiveTopology::TriangleFan {
                    4
                } else {
                    7
                },
                first_instance: 2,
                instance_count: 2,
            },
        )
        .unwrap()
    } else {
        draw.clone()
    };
    if color_output.is_some() || facing.is_some() || scissor.is_some() {
        // Same neutral pipeline/shaders, one changed pipeline field. The first
        // draw is invisible; a cache hit must restore the second draw's state.
        let mut prepared = (*draw.prepared).clone();
        if scissor.is_some() {
            prepared.scissor = Some(nixe_gpu::ScissorRect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            });
        }
        if color_output.is_some() {
            prepared.color_outputs[0].write_mask = nixe_gpu::ColorWriteMask::NONE;
        }
        if let Some((front, _, true)) = facing {
            if front == nixe_gpu::FrontFace::CounterClockwise {
                prepared.front_face = nixe_gpu::FrontFace::Clockwise;
            } else {
                prepared.cull_mode = nixe_gpu::CullMode::Back;
            }
        }
        draw.prepared = Arc::new(prepared);
    }
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(2),
        vec![],
        vec![
            GpuOperation::new(
                GpuCommand::RenderPass(begin),
                [],
                [],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::Draw(draw),
                [],
                [
                    ResourceDependency::Buffer(vertex_buffer),
                    ResourceDependency::Shader(vertex),
                    ResourceDependency::Shader(fragment),
                ],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::Draw(second_draw),
                [],
                [
                    ResourceDependency::Buffer(vertex_buffer),
                    ResourceDependency::Shader(vertex),
                    ResourceDependency::Shader(fragment),
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
    let resident = runtime
        .runtime()
        .acquire_presentable_image(PresentationImageRequest {
            allow_canonical_import: true,
            cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(image_backing.range())
                .unwrap(),
            backing: image_backing.clone(),
            width: WIDTH,
            height: HEIGHT,
            format: PresentationImageFormat::Rgba8,
            layout: ImageMemoryLayout::PitchLinear {
                row_pitch: u64::from(WIDTH * 4),
                layer_stride: u64::from(WIDTH * HEIGHT * 4),
            },
            row_pitch: WIDTH * 4,
        })
        .unwrap();
    assert!(resident_texture(&resident).is_some());
    let mut pixels = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    image_backing.range().read(0, &mut pixels).unwrap();
    let pixel = |x: u32, y: u32| {
        let y = if positive_y { HEIGHT - 1 - y } else { y };
        let offset = ((y * WIDTH + x) * 4) as usize;
        <[u8; 4]>::try_from(&pixels[offset..offset + 4]).unwrap()
    };
    let clear = pixel(0, 0);
    assert!(clear[0].abs_diff(51) <= 1);
    assert!(clear[1].abs_diff(77) <= 1);
    assert!(clear[2].abs_diff(77) <= 1);
    assert_eq!(clear[3], 255);
    assert_eq!(pixel(WIDTH - 1, HEIGHT - 1), clear);

    match scissor {
        Some(ScissorScenario::Partial) => {
            let mut written = 0;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    if (8..24).contains(&x) && (8..24).contains(&y) {
                        written += usize::from(pixel(x, y) != clear);
                    } else {
                        assert_eq!(pixel(x, y), clear, "outside scissor ({x},{y})");
                    }
                }
            }
            assert!(written > 0);
            return;
        }
        Some(ScissorScenario::Empty) => {
            assert!(pixels.chunks_exact(4).all(|p| p == clear));
            return;
        }
        _ => {}
    }
    if let Some((_, _, false)) = facing {
        assert!(pixels.chunks_exact(4).all(|p| p == clear));
        return;
    }

    if square {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let expected = if (8..24).contains(&x) && (8..24).contains(&y) {
                    if let Some((_, expected)) = color_output {
                        expected
                    } else if interpolation == ShaderInterpolation::Constant {
                        if topology == PrimitiveTopology::TriangleFan && x + y >= 31 {
                            [0, 0, 255, 255]
                        } else {
                            [255, 255, 0, 255]
                        }
                    } else {
                        let u = (x as f32 + 0.5 - 8.0) / 16.0;
                        let v = (24.0 - y as f32 - 0.5) / 16.0;
                        let mut weights = if u >= v {
                            [1.0 - u, u - v, v, 0.0]
                        } else {
                            [1.0 - v, 0.0, u, v - u]
                        };
                        if interpolation == ShaderInterpolation::Perspective {
                            for (weight, w) in weights.iter_mut().zip([1.0, 2.0, 4.0, 2.0]) {
                                *weight /= w;
                            }
                            let sum: f32 = weights.iter().sum();
                            for weight in &mut weights {
                                *weight /= sum;
                            }
                        }
                        let rgb = [
                            weights[0] + weights[3],
                            // Second draw: last vertex 9, last instance 3.
                            if mixed {
                                (9.0 + 3.0) / 16.0
                            } else {
                                weights[1] + weights[3]
                            },
                            weights[2],
                        ];
                        [
                            (rgb[0] * 255.0).round() as u8,
                            (rgb[1] * 255.0).round() as u8,
                            (rgb[2] * 255.0).round() as u8,
                            255,
                        ]
                    }
                } else {
                    clear
                };
                let actual = pixel(x, y);
                assert!(
                    actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 1),
                    "pixel ({x}, {y}): actual={actual:?}, expected={expected:?}, interpolation={interpolation:?}, mixed={mixed}"
                );
            }
        }
        return;
    }

    let top = pixel(16, 10);
    let bottom_left = pixel(10, 22);
    let bottom_right = pixel(22, 22);
    assert!(top[2] > top[0] && top[2] > top[1], "top={top:?}");
    assert!(
        bottom_left[0] > bottom_left[1] && bottom_left[0] > bottom_left[2],
        "bottom-left={bottom_left:?}"
    );
    assert!(
        bottom_right[1] > bottom_right[0] && bottom_right[1] > bottom_right[2],
        "bottom-right={bottom_right:?}"
    );
    let drawn = pixels
        .chunks_exact(4)
        .filter(|candidate| *candidate != clear)
        .count();
    assert!((100..=160).contains(&drawn), "drawn pixels={drawn}");
}

fn interpolated_vertex_module(
    interpolation: ShaderInterpolation,
    mixed: bool,
) -> nixe_gpu::ShaderBackendModule {
    let mut inputs: Vec<_> = (0..2)
        .flat_map(|location| {
            (0..if location == 0 { 4 } else { 3 }).map(move |component| (location, component))
        })
        .map(|(location, component)| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Generic(location),
                component,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap()
        })
        .collect();
    let outputs = (0..4)
        .map(|component| (ShaderIoLocation::Position, component))
        // Match producers that export RGBA while the fragment stage consumes
        // only RGB. The unconsumed alpha has no neutral interpolation contract.
        .chain((0..4).map(|component| (ShaderIoLocation::Generic(0), component)))
        .chain(
            (0..if mixed { 3 } else { 0 })
                .map(|component| (ShaderIoLocation::Generic(1), component)),
        )
        .map(|(location, component)| {
            ShaderInterfaceElement::new(
                location,
                component,
                ShaderScalarType::Float32,
                match location {
                    ShaderIoLocation::Generic(0) if component < 3 => Some(interpolation),
                    ShaderIoLocation::Generic(1) => Some(ShaderInterpolation::Constant),
                    _ => None,
                },
            )
            .unwrap()
        })
        .collect();
    let mut instructions = vec![
        load_input(8, 0, 0, ShaderIoLocation::Generic(0), 4),
        store_output(24, 0, ShaderIoLocation::Position, 4),
        load_input(32, 4, 0, ShaderIoLocation::Generic(1), 3),
        move_f32(36, 7, 0.25),
        store_output(40, 4, ShaderIoLocation::Generic(0), 4),
    ];
    if mixed {
        // A flat output depends on computed results, not just vertex-buffer
        // bytes. Preserve both guest builtins when evaluating the last corner.
        for (index, location) in [ShaderIoLocation::VertexId, ShaderIoLocation::InstanceId]
            .into_iter()
            .enumerate()
        {
            inputs.push(
                ShaderInterfaceElement::new(location, 0, ShaderScalarType::Unsigned32, None)
                    .unwrap(),
            );
            instructions.push(ShaderInstruction::new(
                ShaderSourceLocation::new(48 + index as u32 * 8),
                ShaderPredicate::Always,
                ShaderOperation::LoadInput {
                    destinations: vec![ShaderRegister::new(8 + index as u16)].into(),
                    location,
                    first_component: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            ));
        }
        instructions.push(ShaderInstruction::new(
            ShaderSourceLocation::new(64),
            ShaderPredicate::Always,
            ShaderOperation::Add32 {
                destination: ShaderRegister::new(8),
                left: ShaderRegister::new(8),
                right: ShaderRegister::new(9),
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
        ));
        instructions.push(ShaderInstruction::new(
            ShaderSourceLocation::new(72),
            ShaderPredicate::Always,
            ShaderOperation::ConvertIntegerToFloat32 {
                destination: ShaderRegister::new(8),
                source: ShaderRegister::new(8),
                source_type: ShaderScalarType::Unsigned32,
            },
        ));
        instructions.push(move_f32(80, 9, 1.0 / 16.0));
        for (offset, right) in [(88, 9), (96, 5)] {
            instructions.push(ShaderInstruction::new(
                ShaderSourceLocation::new(offset),
                ShaderPredicate::Always,
                ShaderOperation::Multiply32 {
                    destination: ShaderRegister::new(8),
                    left: ShaderRegister::new(8),
                    right: ShaderRegister::new(right),
                    scalar_type: ShaderScalarType::Float32,
                    float_control: nixe_gpu::ShaderFloatControl::PRECISE,
                },
            ));
        }
        instructions.push(store_output(104, 4, ShaderIoLocation::Generic(1), 3));
        instructions.push(ShaderInstruction::new(
            ShaderSourceLocation::new(112),
            ShaderPredicate::Always,
            ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(8)].into(),
                location: ShaderIoLocation::Generic(1),
                first_component: 1,
                scalar_type: ShaderScalarType::Float32,
            },
        ));
    }
    instructions.push(exit(120));
    let verified = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Vertex,
        inputs,
        outputs,
        vec![],
        instructions,
    ))
    .unwrap();
    nixe_gpu::ShaderBackendModule::new(verified)
}

fn interpolated_fragment_module(
    interpolation: ShaderInterpolation,
    mixed: bool,
    alpha: f32,
) -> nixe_gpu::ShaderBackendModule {
    let inputs = (0..3)
        .map(|component| (0, component))
        .chain((0..if mixed { 3 } else { 0 }).map(|component| (1, component)))
        .map(|(location, component)| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Generic(location),
                component,
                ShaderScalarType::Float32,
                Some(if location == 0 {
                    interpolation
                } else {
                    ShaderInterpolation::Constant
                }),
            )
            .unwrap()
        })
        .collect();
    let outputs = (0..4)
        .map(|component| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                component,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap()
        })
        .collect();
    let mut instructions = vec![load_input(8, 0, 0, ShaderIoLocation::Generic(0), 3)];
    if mixed {
        instructions.push(load_input(12, 1, 1, ShaderIoLocation::Generic(1), 1));
    }
    instructions.extend([
        move_f32(16, 3, alpha),
        store_output(24, 0, ShaderIoLocation::Color(0), 4),
        exit(32),
    ]);
    let verified = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Fragment,
        inputs,
        outputs,
        vec![],
        instructions,
    ))
    .unwrap();
    nixe_gpu::ShaderBackendModule::new(verified)
}

fn load_input(
    offset: u32,
    first_register: u16,
    first_component: u8,
    location: ShaderIoLocation,
    components: u16,
) -> ShaderInstruction {
    ShaderInstruction::new(
        ShaderSourceLocation::new(offset),
        ShaderPredicate::Always,
        ShaderOperation::LoadInput {
            destinations: (first_register..first_register + components)
                .map(ShaderRegister::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            location,
            first_component,
            scalar_type: ShaderScalarType::Float32,
        },
    )
}

fn move_f32(offset: u32, destination: u16, value: f32) -> ShaderInstruction {
    ShaderInstruction::new(
        ShaderSourceLocation::new(offset),
        ShaderPredicate::Always,
        ShaderOperation::MoveImmediate32 {
            destination: ShaderRegister::new(destination),
            bits: value.to_bits(),
            scalar_type: ShaderScalarType::Float32,
        },
    )
}

fn store_output(
    offset: u32,
    first_register: u16,
    location: ShaderIoLocation,
    components: u16,
) -> ShaderInstruction {
    ShaderInstruction::new(
        ShaderSourceLocation::new(offset),
        ShaderPredicate::Always,
        ShaderOperation::StoreOutput {
            sources: (first_register..first_register + components)
                .map(ShaderRegister::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            location,
            first_component: 0,
            scalar_type: ShaderScalarType::Float32,
        },
    )
}

fn exit(offset: u32) -> ShaderInstruction {
    ShaderInstruction::new(
        ShaderSourceLocation::new(offset),
        ShaderPredicate::Always,
        ShaderOperation::Exit,
    )
}

#[test]
fn separate_array_layers_preserve_cpu_bytes_sharing_a_physical_page() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(970),
        NonCpuDeviceId::new(970),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
    allocation.write(2048, &[0x35; 2048]).unwrap();
    let allocation_id = GpuAllocationId::new(970);
    let allocation_description = GpuAllocationDescription::new(4096, 4).unwrap();
    let backing = BackingView::new(
        allocation_id,
        allocation_description,
        0,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let image = ImageId::new(970);
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(64, 8, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        2,
        SampleCount::One,
    )
    .unwrap();
    let all = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 2,
    };
    let creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation_id,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Image {
            id: image,
            description,
            view: Some(
                ImageView::new(
                    image,
                    description,
                    Swizzle::IDENTITY,
                    vec![(
                        all,
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: 256,
                            layer_stride: 2048,
                        },
                        backing.clone(),
                    )],
                )
                .unwrap(),
            ),
        },
    ];
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let first = ImageSubresourceRange {
        layer_count: 1,
        ..all
    };
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                image,
                first,
                ImageFormat::Rgba8Unorm,
                64,
                8,
                [1.0, 0.0, 0.0, 1.0],
                970,
            ),
        )
        .unwrap();
    let mut bytes = [0; 4096];
    backing.range().read(0, &mut bytes).unwrap();
    assert!(
        bytes[..2048]
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 0, 0, 255])
    );
    assert_eq!(bytes[2048..], [0x35; 2048]);
    let second = ImageSubresourceRange {
        base_layer: 1,
        ..first
    };
    runtime
        .runtime()
        .submit(
            &[],
            &[],
            &color_clear_submission(
                image,
                second,
                ImageFormat::Rgba8Unorm,
                64,
                8,
                [0.0, 0.0, 1.0, 1.0],
                971,
            ),
        )
        .unwrap();
    backing.range().read(0, &mut bytes).unwrap();
    assert!(
        bytes[..2048]
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 0, 0, 255])
    );
    assert!(
        bytes[2048..]
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 0, 255, 255])
    );
}

#[test]
fn a_single_dirty_image_page_refreshes_only_that_upload_region() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(980),
        NonCpuDeviceId::new(980),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let bytes = 1024 * 1024 * 4;
    let source_memory = CanonicalAllocation::zeroed(bytes, 4096).unwrap();
    let destination_memory = CanonicalAllocation::zeroed(bytes, 4096).unwrap();
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(1024, 1024, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let source = ImageId::new(980);
    let destination = ImageId::new(981);
    let mut creations = Vec::new();
    for (image, memory) in [(source, &source_memory), (destination, &destination_memory)] {
        let allocation = GpuAllocationId::new(image.get());
        let allocation_description = GpuAllocationDescription::new(bytes as u64, 4).unwrap();
        let backing = BackingView::new(
            allocation,
            allocation_description,
            0,
            memory.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
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
                    Swizzle::IDENTITY,
                    vec![(
                        subresources,
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: 4096,
                            layer_stride: bytes as u64,
                        },
                        backing,
                    )],
                )
                .unwrap(),
            ),
        });
    }
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    runtime
        .runtime()
        .submit(
            &creations,
            &[],
            &color_clear_submission(
                source,
                subresources,
                ImageFormat::Rgba8Unorm,
                1024,
                1024,
                [1.0, 0.0, 0.0, 1.0],
                980,
            ),
        )
        .unwrap();
    source_memory.write(4096 * 256, &[0, 255, 0, 255]).unwrap();
    #[cfg(feature = "performance-counters")]
    let before = nixe_gpu::metrics::snapshot()
        .into_iter()
        .find(|(name, _)| *name == "ImageUploadedBytes")
        .unwrap()
        .1;
    let region = |image| nixe_gpu::ImageRegion {
        image,
        subresources,
        origin: nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
        extent: ImageExtent::new(1024, 1024, 1).unwrap(),
    };
    runtime
        .runtime()
        .submit(
            &[],
            &[],
            &OperationSubmission::new(
                FrontendSubmissionId::new(981),
                vec![],
                vec![GpuOperation::new(
                    GpuCommand::Copy(CopyOperation::ImageToImage {
                        source: region(source),
                        destination: region(destination),
                    }),
                    [],
                    [],
                    CapabilityRequirements::none(),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    #[cfg(feature = "performance-counters")]
    {
        let after = nixe_gpu::metrics::snapshot()
            .into_iter()
            .find(|(name, _)| *name == "ImageUploadedBytes")
            .unwrap()
            .1;
        assert_eq!(
            after - before,
            4096,
            "one changed page must not upload the 4 MiB image"
        );
    }
    let mut pixels = vec![0; bytes];
    destination_memory.read(0, &mut pixels).unwrap();
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        let expected = if index == 1024 * 256 {
            [0, 255, 0, 255]
        } else {
            [255, 0, 0, 255]
        };
        assert_eq!(pixel, expected, "pixel {index}");
    }
}
