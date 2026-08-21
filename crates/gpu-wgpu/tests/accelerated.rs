use std::sync::Arc;
use std::sync::{Mutex, MutexGuard, OnceLock};

use nixe_gpu::{
    AttachmentLoad, AttachmentStore, BackendInstanceId, BackendResourceCreateInfo, BackingView,
    BufferDescription, BufferId, BufferRange, BufferRegion, BufferView, CapabilityRequirements,
    ClearOperation, ClearValue, CopyOperation, DrawArguments, DrawOperation, FrontendSubmissionId,
    GpuAllocationDescription, GpuAllocationId, GpuCommand, GpuOperation, ImageDescription,
    ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind, ImageMemoryLayout,
    ImageSubresourceRange, ImageView, OperationSubmission, PipelineDescription, PipelineId,
    PipelineKind, PrimitiveTopology, RenderAttachment, RenderPassAttachmentDescription,
    RenderPassDescription, RenderPassId, RenderPassOperation, ResourceDependency, SampleCount,
    ShaderDescription, ShaderId, ShaderInstruction, ShaderInterfaceElement, ShaderInterpolation,
    ShaderIoLocation, ShaderIr, ShaderOperation, ShaderPredicate, ShaderRegister, ShaderScalarType,
    ShaderSourceLocation, ShaderStage, Swizzle, VerifiedShaderIr, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexStepMode, ViewportTransform, lower_shader_ir_to_wgsl,
};
use nixe_gpu_wgpu::{WgpuBackendConfiguration, WgpuVisibilityCoordinator, initialize_backend};
use nixe_memory::{
    CanonicalBackingPage, CanonicalBackingRange, CanonicalBackingSegment, CanonicalBackingStore,
    ContentGeneration, DeviceAccessDeclaration, DeviceVisibilityPoint, GuestPhysicalPageId,
    MappingGeneration, MemoryPermissions, NonCpuDeviceId, VisibilityCoordinator,
};

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

fn prepare_write(
    backing: &BackingView,
    visibility: &Arc<WgpuVisibilityCoordinator>,
    point: u64,
) -> DeviceAccessDeclaration {
    let declaration = DeviceAccessDeclaration::write(
        visibility.device(),
        DeviceVisibilityPoint::new(point),
        DeviceVisibilityPoint::new(point),
    )
    .unwrap();
    let coordinator: Arc<dyn VisibilityCoordinator> = visibility.clone();
    backing
        .range()
        .prepare_device_access(declaration, coordinator)
        .unwrap();
    declaration
}

fn prepare_read(backing: &BackingView, visibility: &Arc<WgpuVisibilityCoordinator>, point: u64) {
    let coordinator: Arc<dyn VisibilityCoordinator> = visibility.clone();
    backing
        .range()
        .prepare_device_access(
            DeviceAccessDeclaration::read(visibility.device(), DeviceVisibilityPoint::new(point)),
            coordinator,
        )
        .unwrap();
}

#[test]
fn accelerated_buffer_clear_completes_before_canonical_visibility_is_published() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x11);
    let Ok(mut initialized) = initialize_backend(
        BackendInstanceId::new(0x11),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
        return;
    };
    let page = initialized_page(&[0x5a; 64]);
    let allocation = GpuAllocationId::new(1);
    let allocation_description = GpuAllocationDescription::new(64, 4).unwrap();
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        })
        .unwrap();
    let backing = backing(allocation, allocation_description, &page);
    let buffer = BufferId::new(1);
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Buffer {
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
        })
        .unwrap();
    let declaration = prepare_write(&backing, &initialized.visibility, 1);
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
    let token = initialized.backend.submit(&submission).unwrap();
    assert_eq!(initialized.backend.has_completed(token), Ok(true));
    assert_ne!(
        page.visibility_state(),
        nixe_memory::VisibilityState::GpuNewer {
            device: device_id,
            visible_at: DeviceVisibilityPoint::new(1),
        }
    );
    let coordinator: Arc<dyn VisibilityCoordinator> = initialized.visibility.clone();
    backing
        .range()
        .complete_device_write(declaration, coordinator)
        .unwrap();
    let mut bytes = [0xff; 64];
    backing.range().read(0, &mut bytes).unwrap();
    assert!(
        bytes
            .chunks_exact(4)
            .all(|word| word == 0x1122_3344_u32.to_le_bytes())
    );
}

#[test]
fn neutral_runtime_executes_and_publishes_a_complete_transaction() {
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
    let mut runtime = initialized.into_runtime();
    let completed = runtime.execute(&creations, &[], &submission).unwrap();
    assert_eq!(completed.frontend(), submission.id());

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
    let Ok(mut initialized) = initialize_backend(
        BackendInstanceId::new(0x13),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
        return;
    };
    let source_page = initialized_page(&[0x3c; 64]);
    let destination_page = initialized_page(&[0; 64]);
    let description = GpuAllocationDescription::new(64, 4).unwrap();
    let source_allocation = GpuAllocationId::new(3);
    let destination_allocation = GpuAllocationId::new(4);
    for allocation in [source_allocation, destination_allocation] {
        initialized
            .backend
            .create_resource(BackendResourceCreateInfo::Allocation {
                id: allocation,
                description,
            })
            .unwrap();
    }
    let source_backing = backing(source_allocation, description, &source_page);
    let destination_backing = backing(destination_allocation, description, &destination_page);
    let source = BufferId::new(3);
    let destination = BufferId::new(4);
    for (id, backing) in [
        (source, source_backing.clone()),
        (destination, destination_backing.clone()),
    ] {
        initialized
            .backend
            .create_resource(BackendResourceCreateInfo::Buffer {
                id,
                description: BufferDescription::new(64).unwrap(),
                view: Some(
                    BufferView::new(id, BufferDescription::new(64).unwrap(), 0, backing).unwrap(),
                ),
            })
            .unwrap();
    }
    prepare_read(&source_backing, &initialized.visibility, 3);
    let destination_declaration = prepare_write(&destination_backing, &initialized.visibility, 3);
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
    initialized.backend.submit(&submission).unwrap();
    let coordinator: Arc<dyn VisibilityCoordinator> = initialized.visibility.clone();
    destination_backing
        .range()
        .complete_device_write(destination_declaration, coordinator)
        .unwrap();
    let mut bytes = [0; 64];
    destination_backing.range().read(0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x3c; 64]);
}

#[test]
fn accelerated_triangle_draw_matches_geometry_clear_and_interpolation_contract() {
    let _guard = accelerated_test_guard();
    let device_id = NonCpuDeviceId::new(0x12);
    let Ok(mut initialized) = initialize_backend(
        BackendInstanceId::new(0x12),
        device_id,
        WgpuBackendConfiguration::default(),
    ) else {
        eprintln!("Vulkan adapter is unavailable; skipping accelerated acceptance test");
        return;
    };
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    let page = initialized_page(&vec![0; (WIDTH * HEIGHT * 4) as usize]);
    let allocation = GpuAllocationId::new(2);
    let allocation_description =
        GpuAllocationDescription::new(u64::from(WIDTH * HEIGHT * 4), 4).unwrap();
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        })
        .unwrap();
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
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Image {
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
        })
        .unwrap();

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
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Allocation {
            id: vertex_allocation,
            description: vertex_allocation_description,
        })
        .unwrap();
    let vertex_backing = backing(
        vertex_allocation,
        vertex_allocation_description,
        &vertex_page,
    );
    let vertex_buffer = BufferId::new(3);
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Buffer {
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
        })
        .unwrap();

    let vertex = ShaderId::new(1);
    let fragment = ShaderId::new(2);
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Shader {
            id: vertex,
            description: ShaderDescription {
                stage: ShaderStage::Vertex,
            },
            module: triangle_vertex_module(),
        })
        .unwrap();
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Shader {
            id: fragment,
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: triangle_fragment_module(),
        })
        .unwrap();
    let pipeline = PipelineId::new(1);
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::Pipeline {
            id: pipeline,
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        })
        .unwrap();
    let render_pass = RenderPassId::new(1);
    let render_pass_description =
        RenderPassDescription::new(vec![RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
        }])
        .unwrap();
    initialized
        .backend
        .create_resource(BackendResourceCreateInfo::RenderPass {
            id: render_pass,
            description: render_pass_description.clone(),
        })
        .unwrap();
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
    let draw = DrawOperation::new(
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
        DrawArguments::NonIndexed {
            first_vertex: 0,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        },
    )
    .unwrap()
    .with_viewport_transform(
        ViewportTransform::new([16.0, -16.0, 0.5], [16.0, 16.0, 0.5], [0.0, 1.0]).unwrap(),
    );
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
    prepare_read(&vertex_backing, &initialized.visibility, 2);
    let declaration = prepare_write(&image_backing, &initialized.visibility, 2);
    initialized.backend.submit(&submission).unwrap();
    let coordinator: Arc<dyn VisibilityCoordinator> = initialized.visibility.clone();
    image_backing
        .range()
        .complete_device_write(declaration, coordinator)
        .unwrap();
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
