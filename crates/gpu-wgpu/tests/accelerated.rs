use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

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
    VertexStepMode, ViewportTransform, lower_shader_ir_to_wgsl,
};
use nixe_gpu_wgpu::{WgpuBackendConfiguration, initialize_backend, resident_texture};
use nixe_memory::{
    CanonicalAllocation, CanonicalBackingPage, CanonicalBackingRange, CanonicalBackingSegment,
    CanonicalBackingStore, ContentGeneration, CpuVisibilityRequest, DeviceVisibilityPoint,
    GuestPhysicalPageId, MappingGeneration, MemoryPermissions, NonCpuDeviceId,
    VisibilityCoordinatorError,
};

struct RuntimeOwner {
    runtime: Mutex<Box<dyn NeutralBackendRuntime>>,
}

#[test]
fn cpu_authored_rgb565_is_converted_by_a_reusable_gpu_import() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x15);
    let Ok(initialized) = initialize_backend(
        BackendInstanceId::new(0x15),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
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
            .runtime()
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
        page.content_generation(),
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
    let Ok(initialized) = initialize_backend(
        BackendInstanceId::new(0x11),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
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
    let Ok(initialized) = initialize_backend(
        BackendInstanceId::new(0x14),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
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
    let Ok(initialized) = initialize_backend(
        BackendInstanceId::new(0x12),
        NonCpuDeviceId::new(0x12),
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated runtime test");
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
    let Ok(initialized) = initialize_backend(
        BackendInstanceId::new(0x13),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
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
fn accelerated_triangle_draw_matches_geometry_clear_and_interpolation_contract() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x12);
    let Ok(initialized) = initialize_backend(
        BackendInstanceId::new(0x12),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
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

    let mut vertex_bytes = Vec::new();
    for component in [
        -0.5_f32, -0.5, 0.0, 1.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        1.0,
    ] {
        vertex_bytes.extend_from_slice(&component.to_le_bytes());
    }
    let vertex_page = initialized_page(&vertex_bytes);
    let vertex_allocation = GpuAllocationId::new(3);
    let vertex_allocation_description = GpuAllocationDescription::new(72, 4).unwrap();
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
        description: BufferDescription::new(72).unwrap(),
        view: Some(
            BufferView::new(
                vertex_buffer,
                BufferDescription::new(72).unwrap(),
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
        module: triangle_vertex_module(),
    });
    creations.push(BackendResourceCreateInfo::Shader {
        id: fragment,
        description: ShaderDescription {
            stage: ShaderStage::Fragment,
        },
        module: triangle_fragment_module(),
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
        PrimitiveTopology::Triangles,
        vec![],
        vec![
            VertexBufferLayout::new(
                BufferRegion {
                    buffer: vertex_buffer,
                    range: BufferRange::new(0, 72).unwrap(),
                },
                24,
                VertexStepMode::Vertex,
                vec![
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 12,
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
        ViewportTransform::new([16.0, -16.0, 0.5], [16.0, 16.0, 0.5], [0.0, 1.0]).unwrap(),
    );
    let draw = DrawOperation::new(
        Arc::new(prepared),
        DrawArguments::NonIndexed {
            first_vertex: 0,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        },
    )
    .unwrap();
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
        let offset = ((y * WIDTH + x) * 4) as usize;
        <[u8; 4]>::try_from(&pixels[offset..offset + 4]).unwrap()
    };
    let clear = pixel(0, 0);
    assert!(clear[0].abs_diff(51) <= 1);
    assert!(clear[1].abs_diff(77) <= 1);
    assert!(clear[2].abs_diff(77) <= 1);
    assert_eq!(clear[3], 255);
    assert_eq!(pixel(WIDTH - 1, HEIGHT - 1), clear);

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

fn triangle_vertex_module() -> nixe_gpu::ShaderBackendModule {
    let inputs = (0..2)
        .flat_map(|location| (0..3).map(move |component| (location, component)))
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
        .chain((0..3).map(|component| (ShaderIoLocation::Generic(0), component)))
        .map(|(location, component)| {
            ShaderInterfaceElement::new(location, component, ShaderScalarType::Float32, None)
                .unwrap()
        })
        .collect();
    let verified = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Vertex,
        inputs,
        outputs,
        vec![],
        vec![
            load_input(8, 0, 0, ShaderIoLocation::Generic(0), 3),
            move_f32(16, 3, 1.0),
            store_output(24, 0, ShaderIoLocation::Position, 4),
            load_input(32, 4, 0, ShaderIoLocation::Generic(1), 3),
            store_output(40, 4, ShaderIoLocation::Generic(0), 3),
            exit(48),
        ],
    ))
    .unwrap();
    lower_shader_ir_to_wgsl(&verified).unwrap()
}

fn triangle_fragment_module() -> nixe_gpu::ShaderBackendModule {
    let inputs = (0..3)
        .map(|component| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Generic(0),
                component,
                ShaderScalarType::Float32,
                Some(ShaderInterpolation::Perspective),
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
    let verified = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Fragment,
        inputs,
        outputs,
        vec![],
        vec![
            load_input(8, 0, 0, ShaderIoLocation::Generic(0), 3),
            move_f32(16, 3, 1.0),
            store_output(24, 0, ShaderIoLocation::Color(0), 4),
            exit(32),
        ],
    ))
    .unwrap();
    lower_shader_ir_to_wgsl(&verified).unwrap()
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
