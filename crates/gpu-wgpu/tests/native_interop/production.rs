//! Public neutral-runtime acceptance: ordinary clear -> native patches -> ordinary
//! partial clear -> resident presentation/readback, including cached repeated draws.
use super::{SIZE, default_control, device::Context, emitted, normal};
use nixe_gpu::*;
use nixe_memory::*;
use std::sync::Arc;

fn operation(command: GpuCommand) -> GpuOperation {
    GpuOperation::new(command, [], [], CapabilityRequirements::none())
}

fn constant_buffer_read(buffer: BufferId, size: u64) -> ResourceAccess {
    ResourceAccess::new(
        BufferRegion {
            buffer,
            range: BufferRange::new(0, size).unwrap(),
        }
        .target(),
        AccessScope::new(
            PipelineStages::VERTEX_SHADER
                .union(PipelineStages::TESSELLATION_CONTROL_SHADER)
                .union(PipelineStages::TESSELLATION_EVALUATION_SHADER)
                .union(PipelineStages::FRAGMENT_SHADER),
            AccessMode::Read,
            ResourceUsage::UniformBuffer,
        )
        .unwrap(),
    )
}

pub fn check() {
    check_chain(false, Resources::None, None, None);
    check_chain(true, Resources::None, None, None);
    check_chain(true, Resources::Builtins, None, None);
    check_chain(true, Resources::ColorOutput, None, None);
    for resources in [
        Resources::Churn,
        Resources::Generations,
        Resources::Aliases,
        Resources::Residency,
        Resources::SharedVertex,
    ] {
        check_chain(true, resources, None, None);
    }
    for kind in [IndexType::Uint16, IndexType::Uint32] {
        check_chain(false, Resources::None, Some(kind), None);
        check_chain(true, Resources::None, Some(kind), None);
    }
    let directory = tempfile::tempdir().unwrap();
    // Separate devices/runtimes, same disk cache and independently checked pixels.
    for _ in 0..2 {
        check_chain(true, Resources::Aliases, None, Some(directory.path()));
        assert!(std::fs::read_dir(directory.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("native-vulkan-")
        }));
    }
}

pub fn check_wireframe() {
    for smooth in [false, true] {
        eprintln!("native wireframe raster oracle: smooth={smooth}");
        check_chain(true, Resources::Wireframe(smooth), None, None);
    }
}

pub fn check_culling() {
    for positive_y in [false, true] {
        for clockwise in [false, true] {
            check_chain(true, Resources::Culling(positive_y, clockwise), None, None);
        }
    }
}

#[cfg(not(debug_assertions))]
pub fn benchmark() {
    if std::env::var_os("NIXE_TEST_CAPTURE_DIR").is_some() {
        super::device::enable_logging();
    }
    check_chain(true, Resources::Measure, None, None);
}

#[cfg(not(debug_assertions))]
pub fn benchmark_pipeline_cache() {
    assert!(!wgpu::InstanceFlags::default().contains(wgpu::InstanceFlags::VALIDATION));
    let directory = std::path::PathBuf::from(
        std::env::var_os("NIXE_TEST_NATIVE_CACHE_DIR")
            .expect("set NIXE_TEST_NATIVE_CACHE_DIR to a dedicated existing test directory"),
    );
    assert!(directory.is_dir());
    let cache_files = std::fs::read_dir(&directory)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("native-vulkan-")
        })
        .count();
    eprintln!("CACHE_MEASURE existing_native_files={cache_files}");
    check_chain(true, Resources::MeasureCold, None, Some(&directory));
}

#[derive(Clone, Copy, PartialEq)]
enum Resources {
    None,
    Churn,
    Generations,
    Aliases,
    Residency,
    Builtins,
    ColorOutput,
    Wireframe(bool),
    Culling(bool, bool),
    SharedVertex,
    Measure,
    MeasureCold,
}

fn check_chain(
    guest_control: bool,
    scenario: Resources,
    indexed: Option<IndexType>,
    directory: Option<&std::path::Path>,
) {
    #[cfg(not(target_os = "linux"))]
    assert!(
        std::env::var_os("NIXE_TEST_CAPTURE_DIR").is_none(),
        "test capture automation currently requires Linux"
    );
    let resources = !matches!(
        scenario,
        Resources::None
            | Resources::Builtins
            | Resources::ColorOutput
            | Resources::Wireframe(_)
            | Resources::Culling(..)
    );
    let measuring_cold =
        scenario == Resources::MeasureCold && std::env::var_os("NIXE_TEST_CAPTURE_DIR").is_none();
    let binding_capacity = if matches!(
        scenario,
        Resources::Generations | Resources::Aliases | Resources::Residency
    ) {
        4
    } else {
        1
    };
    let (ctx, mut runtime) = Context::with_persistence(
        true,
        if matches!(scenario, Resources::Measure | Resources::MeasureCold) {
            GpuCacheConfiguration::default()
        } else {
            GpuCacheConfiguration::new(6, 1, 1, binding_capacity, 64 * 1024 * 1024).unwrap()
        },
        directory,
    )
    .into_runtime();
    #[cfg(target_os = "linux")]
    let cold_capture = if scenario == Resources::MeasureCold {
        super::capture::Capture::from_environment()
    } else {
        None
    };
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let image = ImageId::new(1);
    let depth = ImageId::new(2);
    let allocation = GpuAllocationId::new(1);
    let size = u64::from(SIZE * SIZE * 4);
    let allocation_description = GpuAllocationDescription::new(size, 4).unwrap();
    let store = CanonicalBackingStore::allocate().unwrap();
    let page = CanonicalBackingPage::initialized(
        &store,
        GuestPhysicalPageId::new(1),
        &vec![0; size as usize],
        ContentGeneration::INITIAL,
    )
    .unwrap();
    let range = CanonicalBackingRange::new(vec![
        CanonicalBackingSegment::new(
            page,
            0,
            size,
            MemoryPermissions::READ_WRITE,
            MappingGeneration::INITIAL,
        )
        .unwrap(),
    ])
    .unwrap();
    let backing = BackingView::new(allocation, allocation_description, 0, range).unwrap();
    let layout = ImageMemoryLayout::PitchLinear {
        row_pitch: u64::from(SIZE * 4),
        layer_stride: size,
    };
    let color_description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(SIZE, SIZE, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    // Exercises HAL's D24-vs-D32S8 format choice, not an assumed VkFormat.
    let depth_format = ImageFormat::Depth24UnormStencil8Uint;
    let depth_description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(SIZE, SIZE, 1).unwrap(),
        depth_format,
        ImageKind::DepthStencil,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let pass = RenderPassId::new(1);
    let pipeline = PipelineId::new(1);
    let pass_description = RenderPassDescription::new(vec![
        RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
        },
        RenderPassAttachmentDescription {
            kind: ImageKind::DepthStencil,
            format: depth_format,
            samples: SampleCount::One,
        },
    ])
    .unwrap();
    let mut creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Image {
            id: image,
            description: color_description,
            view: Some(
                ImageView::new(
                    image,
                    color_description,
                    Swizzle::IDENTITY,
                    vec![(subresources, layout, backing.clone())],
                )
                .unwrap(),
            ),
        },
        BackendResourceCreateInfo::Image {
            id: depth,
            description: depth_description,
            view: None,
        },
        BackendResourceCreateInfo::Pipeline {
            id: pipeline,
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        },
        BackendResourceCreateInfo::RenderPass {
            id: pass,
            description: pass_description.clone(),
        },
    ];
    let points = 4;
    let vertices = BufferId::new(3);
    let vertex_allocation = GpuAllocationId::new(3);
    let vertex_bytes: Vec<_> = [
        [0.125_f32, 0.125, 0.0, 0.0], // unused prefix before binding
        [0.125, 0.125, 0.0, 0.0],     // skipped first vertex
        [0.25, 0.25, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],    // first instance (base 2)
        [0.75, 0.5, 0.0, 0.0],   // last instance
        [0.125, 0.75, 0.0, 0.0], // last point of the first four-point patch
        [0.0; 4],
        [0.0; 4], // incomplete trailing patch
    ]
    .into_iter()
    .flatten()
    .flat_map(f32::to_le_bytes)
    .collect();
    let vertex_size = vertex_bytes.len() as u64;
    let vertex_description = GpuAllocationDescription::new(vertex_size, 4).unwrap();
    let vertex_page = CanonicalBackingPage::initialized(
        &store,
        GuestPhysicalPageId::new(3),
        &vertex_bytes,
        ContentGeneration::INITIAL,
    )
    .unwrap();
    let vertex_backing = BackingView::new(
        vertex_allocation,
        vertex_description,
        0,
        CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                vertex_page.clone(),
                0,
                vertex_size,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();
    creations.extend([
        BackendResourceCreateInfo::Allocation {
            id: vertex_allocation,
            description: vertex_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: vertices,
            description: BufferDescription::new(vertex_size).unwrap(),
            view: Some(
                BufferView::new(
                    vertices,
                    BufferDescription::new(vertex_size).unwrap(),
                    0,
                    vertex_backing.clone(),
                )
                .unwrap(),
            ),
        },
    ]);
    let mut shaders = if resources {
        emitted::shared_buffer_shaders().to_vec()
    } else if guest_control {
        let mut shaders = emitted::shaders().to_vec();
        let tc = shaders[1].ir();
        let instructions = tc
            .instructions()
            .iter()
            .map(|i| {
                let op =
                    if let ShaderOperation::LoadConstantBufferIndexed32 { destination, .. } =
                        i.operation()
                    {
                        ShaderOperation::MoveImmediate32 {
                            destination: *destination,
                            bits: 1_f32.to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        }
                    } else {
                        i.operation().clone()
                    };
                ShaderInstruction::new(i.source(), i.predicate(), op)
            })
            .collect();
        shaders[1] = VerifiedShaderIr::verify(
            ShaderIr::new(
                tc.stage(),
                tc.inputs().to_vec(),
                tc.outputs().to_vec(),
                vec![],
                instructions,
            )
            .with_tessellation_control_points(Some(3)),
        )
        .unwrap();
        shaders
    } else {
        default_control::shaders(points).to_vec()
    };
    // Exercise the production staged vertex upload with a nonzero binding offset,
    // first vertex/instance, padded stride and both per-vertex/per-instance rates.
    let component = u8::from(!guest_control);
    let input = emitted::interface(
        ShaderIoLocation::Generic(0),
        component,
        ShaderScalarType::Float32,
    );
    if scenario == Resources::Builtins {
        shaders = super::builtins::shaders();
    } else if scenario == Resources::SharedVertex {
        // The same physical buffer feeds vertex fetch and TCS/TES/FS storage
        // reads in one native pass. Its second word is the checked green value.
        let vs = shaders[0].ir();
        let code = vs
            .instructions()
            .iter()
            .map(|i| {
                let op = if let ShaderOperation::LoadConstantBuffer32 { destination, .. } =
                    i.operation()
                {
                    emitted::load(
                        destination.index(),
                        ShaderIoLocation::Generic(0),
                        1,
                        ShaderScalarType::Float32,
                    )
                } else {
                    i.operation().clone()
                };
                ShaderInstruction::new(i.source(), i.predicate(), op)
            })
            .collect();
        shaders[0] = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            vec![emitted::interface(
                ShaderIoLocation::Generic(0),
                1,
                ShaderScalarType::Float32,
            )],
            vs.outputs().to_vec(),
            vec![],
            code,
        ))
        .unwrap();
    } else if !resources {
        shaders[0] = VerifiedShaderIr::verify(emitted::ir(
            ShaderStage::Vertex,
            vec![input],
            vec![input],
            vec![
                emitted::load(
                    0,
                    ShaderIoLocation::Generic(0),
                    component,
                    ShaderScalarType::Float32,
                ),
                emitted::store(0, ShaderIoLocation::Generic(0), component),
                ShaderOperation::Exit,
            ],
        ))
        .unwrap();
    }
    if scenario == Resources::ColorOutput {
        let fs = shaders[3].ir();
        let code = fs
            .instructions()
            .iter()
            .map(|i| {
                let op = match i.operation() {
                    ShaderOperation::MoveImmediate32 {
                        destination,
                        scalar_type,
                        ..
                    } if destination.index() == 3 => ShaderOperation::MoveImmediate32 {
                        destination: *destination,
                        bits: 0.5_f32.to_bits(),
                        scalar_type: *scalar_type,
                    },
                    op => op.clone(),
                };
                ShaderInstruction::new(i.source(), i.predicate(), op)
            })
            .collect();
        shaders[3] = VerifiedShaderIr::verify(ShaderIr::new(
            fs.stage(),
            fs.inputs().to_vec(),
            fs.outputs().to_vec(),
            vec![],
            code,
        ))
        .unwrap();
    }
    let table = DescriptorTableId::new(1);
    let mut descriptor_accesses = Vec::new();
    if resources {
        creations.push(BackendResourceCreateInfo::DescriptorTable {
            id: table,
            description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer]).unwrap(),
            bindings: vec![DescriptorTableBinding {
                binding: 0,
                resource: ResourceDependency::Buffer(vertices),
            }]
            .into_boxed_slice(),
        });
        descriptor_accesses.push(constant_buffer_read(vertices, vertex_size));
    }
    let regenerated: Vec<_> = creations
        .iter()
        .filter(|info| {
            [
                ResourceDependency::Buffer(vertices),
                ResourceDependency::DescriptorTable(table),
                ResourceDependency::Image(image),
            ]
            .contains(&info.dependency())
        })
        .cloned()
        .collect();
    let alias = BufferId::new(5);
    let alias_table = DescriptorTableId::new(2);
    if scenario == Resources::Aliases {
        creations.extend([
            BackendResourceCreateInfo::Buffer {
                id: alias,
                description: BufferDescription::new(vertex_size).unwrap(),
                view: Some(
                    BufferView::new(
                        alias,
                        BufferDescription::new(vertex_size).unwrap(),
                        0,
                        vertex_backing,
                    )
                    .unwrap(),
                ),
            },
            BackendResourceCreateInfo::DescriptorTable {
                id: alias_table,
                description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer]).unwrap(),
                bindings: vec![DescriptorTableBinding {
                    binding: 0,
                    resource: ResourceDependency::Buffer(alias),
                }]
                .into(),
            },
        ]);
    }
    let mut dependencies = vec![ResourceDependency::Buffer(vertices)];
    let indices = BufferId::new(4);
    let index_bias = |kind| match kind {
        // UINT16_MAX is an ordinary index: neutral patch lists do not restart.
        IndexType::Uint16 => 65_531,
        // Exercise fullDrawIndexUint32 beyond the minimum 24-bit index range.
        IndexType::Uint32 => 16_777_215,
        IndexType::Uint8 => unreachable!(),
    };
    // Offset eight bytes at bind time, then skip two elements at draw time.
    // Reordered point indices distinguish assembly from a nonindexed draw.
    let index_bytes = |serial: u64, kind| {
        let negative_base = serial % 2 == 1;
        let last = if negative_base { 4 } else { 2 };
        let mut bytes = vec![0; 8];
        for index in [0_u32, 0].into_iter().chain([2, 1, 3, last, 1, 2].map(|v| {
            if negative_base {
                v + index_bias(kind)
            } else {
                v - 1
            }
        })) {
            match kind {
                IndexType::Uint16 => bytes.extend_from_slice(&(index as u16).to_le_bytes()),
                IndexType::Uint32 => bytes.extend_from_slice(&index.to_le_bytes()),
                IndexType::Uint8 => unreachable!(),
            }
        }
        bytes
    };
    let index_backing = indexed.map(|kind| {
        let bytes = index_bytes(1, kind);
        let size = bytes.len() as u64;
        let allocation = GpuAllocationId::new(4);
        let description = GpuAllocationDescription::new(size, 4).unwrap();
        let page = CanonicalBackingPage::initialized(
            &store,
            GuestPhysicalPageId::new(4),
            &bytes,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let backing = BackingView::new(
            allocation,
            description,
            0,
            CanonicalBackingRange::new(vec![
                CanonicalBackingSegment::new(
                    page.clone(),
                    0,
                    size,
                    MemoryPermissions::READ_WRITE,
                    MappingGeneration::INITIAL,
                )
                .unwrap(),
            ])
            .unwrap(),
        )
        .unwrap();
        creations.extend([
            BackendResourceCreateInfo::Allocation {
                id: allocation,
                description,
            },
            BackendResourceCreateInfo::Buffer {
                id: indices,
                description: BufferDescription::new(size).unwrap(),
                view: Some(
                    BufferView::new(indices, BufferDescription::new(size).unwrap(), 0, backing)
                        .unwrap(),
                ),
            },
        ]);
        dependencies.push(ResourceDependency::Buffer(indices));
        (page, size)
    });
    for (i, shader) in shaders.into_iter().enumerate() {
        let id = ShaderId::new(i as u64 + 1);
        dependencies.push(ResourceDependency::Shader(id));
        creations.push(BackendResourceCreateInfo::Shader {
            id,
            description: ShaderDescription {
                stage: shader.ir().stage(),
            },
            module: ShaderBackendModule::new(shader),
        });
    }
    let mut oracles = Vec::new();
    let cases = [
        (1, 1_f32, 0.25_f32, 0.5_f32),
        (2, 4_f32, 0.75_f32, 0.25_f32),
        (3, 0_f32, 0.75_f32, 0.25_f32),
        (4, 2_f32, 0.25_f32, 0.5_f32),
    ];
    for index in 0..if scenario == Resources::Churn {
        40
    } else if indexed.is_some() || resources || matches!(scenario, Resources::Culling(..)) {
        8
    } else {
        4
    } {
        let (_, mut outer, inner, blue) = cases[index % 4];
        let serial = index as u64 + 1;
        // Each alias must consume both payloads: tying contents to alias parity
        // would accidentally let an upload-only-on-first-use implementation pass.
        let near = if scenario == Resources::Aliases {
            matches!(serial % 4, 0 | 1)
        } else {
            serial % 2 == 1
        };
        if scenario == Resources::Generations && serial > 1 {
            creations.extend(regenerated.iter().cloned());
        }
        if let (Some(kind), Some((page, _))) = (indexed, &index_backing) {
            page.prepare_write().unwrap();
            let generation = page.content_generation();
            page.write_preflighted(
                0,
                &index_bytes(serial, kind),
                generation,
                generation.next().unwrap(),
            )
            .unwrap();
            if serial >= 5 {
                outer = 1.0;
            }
        }
        let mut table = table;
        if resources && (serial <= 4 || scenario != Resources::Churn) {
            // Uploads must order after prior native readers and before every
            // consuming shader stage, including cached aliases of this page.
            let words = if near {
                [0.25_f32, 1.0, 1.0, 0.25]
            } else {
                [0.75, 0.25, 0.0, 0.75]
            };
            let bytes: Vec<_> = words.into_iter().flat_map(f32::to_le_bytes).collect();
            vertex_page.prepare_write().unwrap();
            let generation = vertex_page.content_generation();
            vertex_page
                .write_preflighted(0, &bytes, generation, generation.next().unwrap())
                .unwrap();
        }
        if scenario == Resources::Aliases {
            let selected = if serial.is_multiple_of(2) {
                table = alias_table;
                alias
            } else {
                vertices
            };
            descriptor_accesses = vec![constant_buffer_read(selected, vertex_size)];
            dependencies[0] = ResourceDependency::Buffer(selected);
        }
        if scenario == Resources::Churn && serial > 4 {
            // More than one descriptor-pool page, with a one-entry descriptor
            // cache and no intervening CPU waits. Every old set stays immutable
            // and alive while new buffer identities churn through that cache.
            let id = BufferId::new(100 + serial);
            table = DescriptorTableId::new(100 + serial);
            let allocation = GpuAllocationId::new(100 + serial);
            let description = GpuAllocationDescription::new(16, 4).unwrap();
            let words = if near {
                [0.25_f32, 1.0, 1.0, 0.25]
            } else {
                [0.75, 0.25, 0.0, 0.75]
            };
            let bytes: Vec<_> = words.into_iter().flat_map(f32::to_le_bytes).collect();
            let page = CanonicalBackingPage::initialized(
                &store,
                GuestPhysicalPageId::new(100 + serial),
                &bytes,
                ContentGeneration::INITIAL,
            )
            .unwrap();
            let backing = BackingView::new(
                allocation,
                description,
                0,
                CanonicalBackingRange::new(vec![
                    CanonicalBackingSegment::new(
                        page,
                        0,
                        16,
                        MemoryPermissions::READ_WRITE,
                        MappingGeneration::INITIAL,
                    )
                    .unwrap(),
                ])
                .unwrap(),
            )
            .unwrap();
            creations.extend([
                BackendResourceCreateInfo::Allocation {
                    id: allocation,
                    description,
                },
                BackendResourceCreateInfo::Buffer {
                    id,
                    description: BufferDescription::new(16).unwrap(),
                    view: Some(
                        BufferView::new(id, BufferDescription::new(16).unwrap(), 0, backing)
                            .unwrap(),
                    ),
                },
                BackendResourceCreateInfo::DescriptorTable {
                    id: table,
                    description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer])
                        .unwrap(),
                    bindings: vec![DescriptorTableBinding {
                        binding: 0,
                        resource: ResourceDependency::Buffer(id),
                    }]
                    .into_boxed_slice(),
                },
            ]);
            descriptor_accesses = vec![constant_buffer_read(id, 16)];
            dependencies[0] = ResourceDependency::Buffer(id);
        }
        let mut prepared = PreparedDraw::new(
            pipeline,
            pass,
            PrimitiveTopology::Patches,
            if resources { vec![table] } else { vec![] },
            if scenario == Resources::SharedVertex {
                vec![
                    VertexBufferLayout::new(
                        BufferRegion {
                            buffer: vertices,
                            range: BufferRange::new(0, vertex_size).unwrap(),
                        },
                        16,
                        VertexStepMode::Instance,
                        vec![VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    )
                    .unwrap(),
                ]
            } else if resources || scenario == Resources::Builtins {
                vec![]
            } else {
                vec![
                    VertexBufferLayout::new(
                        BufferRegion {
                            buffer: vertices,
                            range: BufferRange::new(16, vertex_size - 16).unwrap(),
                        },
                        16,
                        if guest_control {
                            VertexStepMode::Instance
                        } else {
                            VertexStepMode::Vertex
                        },
                        vec![VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    )
                    .unwrap(),
                ]
            },
            indexed.map(|kind| {
                (
                    BufferRegion {
                        buffer: indices,
                        range: BufferRange::new(8, index_backing.as_ref().unwrap().1 - 8).unwrap(),
                    },
                    kind,
                )
            }),
        )
        .unwrap();
        prepared.tessellation = Some(TessellationState {
            input_control_points: points,
            mode: TessellationMode {
                domain: TessellationDomain::Triangles,
                spacing: TessellationSpacing::Equal,
                output: TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
            },
            control: if guest_control {
                TessellationControl::Shader
            } else {
                TessellationControl::DefaultLevels {
                    outer: [
                        outer.to_bits(),
                        outer.to_bits(),
                        outer.to_bits(),
                        blue.to_bits(),
                    ],
                    inner: [inner.to_bits(), 0],
                    defined: 0b01_1111,
                }
            },
        });
        prepared.depth_state = DepthState {
            test_enabled: true,
            write_enabled: true,
            compare: if resources && serial >= 3 {
                DepthCompareOperation::GreaterEqual
            } else if guest_control && serial >= 3 {
                DepthCompareOperation::Always
            } else {
                DepthCompareOperation::LessEqual
            },
        };
        if scenario == Resources::ColorOutput {
            prepared.color_outputs[0] = color_output(serial);
        }
        if let Resources::Culling(positive_y, clockwise) = scenario {
            if clockwise {
                prepared.tessellation.as_mut().unwrap().mode.output =
                    TessellationOutput::Triangles(TessellationWinding::Clockwise);
            }
            prepared.front_face = if serial <= 2 {
                FrontFace::CounterClockwise
            } else {
                FrontFace::Clockwise
            };
            prepared.cull_mode = match serial {
                1 | 3 => CullMode::Back,
                2 | 4 => CullMode::Front,
                5 | 7 => CullMode::FrontAndBack,
                _ => CullMode::None,
            };
            prepared.viewport_transform = Some(
                ViewportTransform::new(
                    [16.0, if positive_y { 16.0 } else { -16.0 }, 1.0],
                    [16.0, 16.0, 0.0],
                    [0.0, 1.0],
                )
                .unwrap(),
            );
        }
        if let Resources::Wireframe(smooth) = scenario {
            // Lower-left-domain CCW triangles remain framebuffer CCW with
            // the negative-Y viewport. Exercise culling and wireframe together.
            prepared.front_face = FrontFace::CounterClockwise;
            prepared.cull_mode = CullMode::Back;
            prepared.triangle_rasterization = TriangleRasterization::Wireframe {
                width_bits: if serial % 2 == 1 { 1_f32 } else { 4_f32 }.to_bits(),
                smooth,
            };
            // Store coverage directly in alpha, without double-blending two
            // instances. It must survive the native -> ordinary handoff.
        }
        let draw = DrawOperation::new(
            Arc::new(prepared),
            if let Some(kind) = indexed {
                DrawArguments::Indexed {
                    first_index: 2,
                    index_count: match serial {
                        5..=7 => serial as u32 - 4,
                        _ => u32::from(points) + 2,
                    },
                    vertex_offset: if serial % 2 == 1 {
                        -(index_bias(kind) as i32)
                    } else {
                        1
                    },
                    first_instance: 2,
                    instance_count: 2,
                }
            } else {
                DrawArguments::NonIndexed {
                    first_vertex: 1,
                    vertex_count: if scenario == Resources::Builtins {
                        8
                    } else {
                        u32::from(points) + 2
                    },
                    first_instance: if scenario == Resources::Builtins {
                        2 + serial as u32 % 2
                    } else if scenario == Resources::SharedVertex {
                        0
                    } else {
                        2
                    },
                    instance_count: if matches!(
                        scenario,
                        Resources::Builtins | Resources::SharedVertex
                    ) {
                        1
                    } else {
                        2
                    },
                }
            },
        )
        .unwrap();
        let attachments = vec![
            RenderAttachment {
                image,
                subresources,
                kind: ImageKind::Color,
                format: ImageFormat::Rgba8Unorm,
                samples: SampleCount::One,
                load: AttachmentLoad::Load,
                store: AttachmentStore::Store,
            },
            RenderAttachment {
                image: depth,
                subresources,
                kind: ImageKind::DepthStencil,
                format: depth_format,
                samples: SampleCount::One,
                load: if serial == 1 || resources {
                    AttachmentLoad::Clear(ClearValue::DepthStencil {
                        // The TES descriptor read changes Z between .25 and .75.
                        // Compare against .5 so stale/missing TES reads affect
                        // pixels too, rather than only an unobserved depth value.
                        depth: if resources { 0.5 } else { 1.0 },
                        stencil: 0,
                    })
                } else {
                    AttachmentLoad::Load
                },
                store: AttachmentStore::Store,
            },
        ];
        let clear = |origin, extent, value| {
            operation(GpuCommand::Clear(
                ClearOperation::image(
                    ImageRegion {
                        image,
                        subresources,
                        origin,
                        extent,
                    },
                    ImageKind::Color,
                    ImageFormat::Rgba8Unorm,
                    SampleCount::One,
                    ClearValue::Color(value),
                )
                .unwrap(),
            ))
        };
        // In-pass state changes must not be lost when redundant native bindings
        // are suppressed. Specialized first draws have no visible color effect;
        // the second must restore its pipeline/viewport, mask or dynamic levels.
        let mut first_draw = draw.clone();
        if matches!(
            scenario,
            Resources::Builtins
                | Resources::SharedVertex
                | Resources::ColorOutput
                | Resources::Wireframe(_)
                | Resources::Culling(..)
        ) || !guest_control
        {
            let mut prepared = (*draw.prepared).clone();
            if matches!(scenario, Resources::Culling(..)) {
                prepared.cull_mode = CullMode::FrontAndBack;
            } else if matches!(scenario, Resources::Wireframe(_)) {
                prepared.color_outputs[0].write_mask = ColorWriteMask::NONE;
                // Two widths within one pass; neither pipeline selection nor
                // redundant-state suppression may drop the second width.
                if let TriangleRasterization::Wireframe { width_bits, .. } =
                    &mut prepared.triangle_rasterization
                {
                    *width_bits = 8_f32.to_bits();
                }
            } else if scenario == Resources::ColorOutput {
                // Same logical pipeline and shaders, distinct component mask.
                // The next draw must restore writes on a pipeline-cache hit.
                prepared.color_outputs[0].write_mask = ColorWriteMask::NONE;
            } else if scenario == Resources::SharedVertex {
                prepared.depth_state.compare = DepthCompareOperation::Never;
                prepared.vertex_buffers[0].buffer.range =
                    BufferRange::new(16, vertex_size - 16).unwrap();
            } else if scenario == Resources::Builtins {
                prepared.depth_state.compare = DepthCompareOperation::Never;
                prepared.viewport_transform = Some(
                    ViewportTransform::new([8.0, -8.0, 1.0], [8.0, 8.0, 0.0], [0.0, 1.0]).unwrap(),
                );
            } else if let TessellationControl::DefaultLevels { outer, .. } =
                &mut prepared.tessellation.as_mut().unwrap().control
            {
                outer[0] = 0;
            }
            first_draw.prepared = Arc::new(prepared);
        }
        let hidden_fill = matches!(scenario, Resources::Wireframe(_)).then(|| {
            let mut fill = first_draw.clone();
            let mut prepared = (*fill.prepared).clone();
            prepared.triangle_rasterization = TriangleRasterization::Fill;
            fill.prepared = Arc::new(prepared);
            fill
        });
        let mut commands = vec![
            clear(
                ImageOrigin { x: 0, y: 0, z: 0 },
                ImageExtent::new(SIZE, SIZE, 1).unwrap(),
                if scenario == Resources::ColorOutput {
                    [0.2, 0.4, 0.6, 0.8]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                },
            ),
            operation(GpuCommand::RenderPass(
                RenderPassOperation::begin(pass, pass_description.clone(), attachments.clone())
                    .unwrap(),
            )),
            GpuOperation::new(
                GpuCommand::Draw(first_draw),
                descriptor_accesses.clone(),
                dependencies.clone(),
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::Draw(draw),
                descriptor_accesses.clone(),
                dependencies.clone(),
                CapabilityRequirements::none(),
            ),
            operation(GpuCommand::RenderPass(RenderPassOperation::end(pass))),
            clear(
                ImageOrigin { x: 0, y: 0, z: 0 },
                ImageExtent::new(2, 2, 1).unwrap(),
                [1.0, 0.0, 1.0, 1.0],
            ),
        ];
        if let Some(fill) = hidden_fill {
            // Binding a static-width fill pipeline invalidates the previous
            // dynamic line width. Returning to wireframe must re-establish it.
            commands.insert(
                3,
                GpuOperation::new(
                    GpuCommand::Draw(fill),
                    descriptor_accesses.clone(),
                    dependencies.clone(),
                    CapabilityRequirements::none(),
                ),
            );
        }
        if matches!(scenario, Resources::Wireframe(_)) && serial.is_multiple_of(2) {
            // Exercise frontend-style leading and internal barriers with core
            // and synchronization validation. Other frames keep a single raw
            // segment, covering both cached paths with the same pixel oracle.
            commands.insert(
                2,
                operation(GpuCommand::Barrier(
                    BarrierOperation::new(vec![
                        ResourceTransition::new(
                            AccessTarget::Buffer {
                                buffer: vertices,
                                range: BufferRange::new(0, vertex_size).unwrap(),
                            },
                            AccessScope::new(
                                PipelineStages::COPY,
                                AccessMode::Write,
                                ResourceUsage::TransferDestination,
                            )
                            .unwrap(),
                            AccessScope::new(
                                PipelineStages::VERTEX_INPUT,
                                AccessMode::Read,
                                ResourceUsage::VertexBuffer,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                )),
            );
            commands.insert(
                4,
                operation(GpuCommand::Barrier(
                    BarrierOperation::new(vec![
                        ResourceTransition::new(
                            AccessTarget::Image {
                                image,
                                subresources,
                            },
                            AccessScope::new(
                                PipelineStages::COLOR_OUTPUT,
                                AccessMode::Write,
                                ResourceUsage::ColorAttachment,
                            )
                            .unwrap(),
                            AccessScope::new(
                                PipelineStages::COLOR_OUTPUT,
                                AccessMode::ReadWrite,
                                ResourceUsage::ColorAttachment,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                )),
            );
        }
        let discard = serial == 4
            && !matches!(
                scenario,
                Resources::Aliases
                    | Resources::Builtins
                    | Resources::ColorOutput
                    | Resources::Wireframe(_)
                    | Resources::Culling(..)
            );
        if discard {
            // Discard through ordinary wgpu after prior native use. The next raw
            // draw must establish initialization again or the following normal
            // partial clear/readback would silently erase its results.
            let mut discarded = attachments;
            discarded[0].store = AttachmentStore::Discard;
            commands[0] = operation(GpuCommand::RenderPass(
                RenderPassOperation::begin(pass, pass_description.clone(), discarded).unwrap(),
            ));
            commands.insert(
                1,
                operation(GpuCommand::RenderPass(RenderPassOperation::end(pass))),
            );
        }
        let measured_commands =
            (scenario == Resources::Measure && serial == 1).then(|| commands.clone());
        let submission =
            OperationSubmission::new(FrontendSubmissionId::new(serial), vec![], commands).unwrap();
        let invalidations = if scenario == Resources::Generations {
            vec![
                ResourceDependency::DescriptorTable(table),
                ResourceDependency::Buffer(vertices),
                ResourceDependency::Image(image),
            ]
        } else {
            vec![]
        };
        // Test-only cold-path measurement includes IR/SPIR-V lowering, host
        // resource creation/uploads, cache import and driver compilation. It is
        // deliberately not advertised as isolated vkCreateGraphicsPipelines time.
        let start = measuring_cold.then(std::time::Instant::now);
        #[cfg(target_os = "linux")]
        if serial == 1
            && let Some(capture) = &cold_capture
        {
            capture.start("cold");
        }
        runtime
            .submit(&creations, &invalidations, &submission)
            .unwrap();
        #[cfg(target_os = "linux")]
        if serial == 1
            && let Some(capture) = &cold_capture
        {
            capture.end();
        }
        if let Some(start) = start {
            eprintln!(
                "CACHE_MEASURE submission={serial} submit_ns={}",
                start.elapsed().as_nanos()
            );
        }
        creations.clear();
        let resident = runtime
            .acquire_presentable_image(PresentationImageRequest {
                allow_canonical_import: true,
                cpu_writes: CanonicalCpuWriteDependency::capture(backing.range()).unwrap(),
                backing: backing.clone(),
                width: SIZE,
                height: SIZE,
                format: PresentationImageFormat::Rgba8,
                layout,
                row_pitch: SIZE * 4,
            })
            .unwrap();
        let texture = nixe_gpu_wgpu::resident_texture(&resident).unwrap();
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("production patch oracle"),
            size: u64::from(256 * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        ctx.queue.submit([encoder.finish()]);
        let expected = if indexed.is_some() && (5..=7).contains(&serial) {
            [0, 0, 0, 255]
        } else if scenario == Resources::ColorOutput {
            // Two instances of [1, .75, 1, .5] over [.2, .4, .6, .8].
            match serial {
                1 => [204, 102, 230, 128],
                2 => [51, 191, 153, 204],
                3 => [0, 0, 0, 204],
                4 => [51, 102, 153, 128],
                _ => unreachable!(),
            }
        } else if resources {
            let passes_depth = if serial < 3 { near } else { !near };
            if !passes_depth {
                [0, 0, 0, 255]
            } else if near {
                [64, 255, 255, 255]
            } else {
                [191, 64, 0, 255]
            }
        } else if scenario == Resources::Builtins {
            [48, if serial % 2 == 1 { 96 } else { 64 }, 239, 255]
        } else if guest_control {
            [255, 191, 255, 255]
        } else if outer == 0.0 {
            [0, 0, 0, 255]
        } else {
            [
                (inner * 255.0).round() as u8,
                if indexed.is_some() && serial.is_multiple_of(2) {
                    128
                } else {
                    191
                },
                (blue * 255.0).round() as u8,
                255,
            ]
        };
        oracles.push((readback, expected, serial, discard));
        if scenario == Resources::Generations {
            // Logical IDs can be reused only after retirement. This wait is an
            // explicit test lifecycle step, not a native submission requirement.
            let completed = runtime.wait_for_completion().unwrap().unwrap();
            assert_eq!(completed.frontend(), FrontendSubmissionId::new(serial));
        }
        if scenario == Resources::Residency && serial == 1 {
            residency_pressure(runtime.as_mut());
        }
        if let Some(commands) = measured_commands {
            super::measure::run(runtime.as_mut(), &commands);
        }
    }
    // Defer all oracle readbacks until here. Apart from explicit generation
    // retirement, submissions are queued without host waits; exercise both
    // terminal teardown and direct runtime drop with outstanding work.
    if !guest_control || directory.is_some() {
        runtime.teardown().unwrap();
    }
    drop(runtime);
    for (readback, expected, serial, discard) in oracles {
        let pixels = normal::read_pixels(&ctx, readback);
        if let Resources::Culling(positive_y, clockwise) = scenario {
            // Neutral CCW triangles have positive UV shoelace area, independent
            // of Vulkan's default domain origin. TES writes UV to NDC XY; the
            // negative-Y viewport makes area negative (framebuffer CCW).
            // Test both domain windings and both viewport Y signs.
            // https://docs.vulkan.org/spec/latest/chapters/tessellation.html#tessellation-winding
            let visible = match serial {
                1 | 4 => !(positive_y ^ clockwise),
                2 | 3 => positive_y ^ clockwise,
                5 | 7 => false,
                _ => true,
            };
            for (x, y) in [(18, 13), (20, 11), (17, 7)] {
                let y = if positive_y { SIZE - 1 - y } else { y };
                let actual = pixels[(y * SIZE + x) as usize];
                let expected = if visible {
                    [255, 191, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert!(
                    actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 1),
                    "culling serial={serial} positive_y={positive_y}: {actual:?} != {expected:?}"
                );
            }
            assert_eq!(pixels[0], [255, 0, 255, 255]);
            continue;
        }
        if let Resources::Wireframe(smooth) = scenario {
            check_wireframe_pixels(&pixels, serial, smooth);
            continue;
        }
        for (x, y) in [(18, 13), (20, 11), (17, 7)] {
            let actual = pixels[(y * SIZE + x) as usize];
            for c in 0..4 {
                assert!(
                    actual[c].abs_diff(expected[c]) <= 1,
                    "production serial={serial} pixel=({x},{y}): {actual:?}, expected {expected:?}"
                );
            }
        }
        assert_eq!(pixels[0], [255, 0, 255, 255]);
        if scenario == Resources::Builtins {
            let actual = pixels[(13 * SIZE + 2) as usize];
            let expected = [175, expected[1], 239, 255];
            for c in 0..4 {
                assert!(
                    actual[c].abs_diff(expected[c]) <= 1,
                    "second patch: {actual:?} != {expected:?}"
                );
            }
            continue;
        }
        assert_eq!(
            pixels[(4 * SIZE + 4) as usize],
            if scenario == Resources::ColorOutput {
                [51, 102, 153, 204]
            } else {
                [0, 0, 0, if discard { 0 } else { 255 }]
            }
        );
    }
}

fn check_wireframe_pixels(pixels: &[[u8; 4]], serial: u64, smooth: bool) {
    // TES emits the unit barycentric triangle: framebuffer edges x=16,
    // y=16 and x-y=16. Samples below are away from endpoints/edge overlap.
    let pixel = |x, y| pixels[(y * SIZE + x) as usize];
    assert_eq!(pixel(0, 0), [255, 0, 255, 255]);
    assert_eq!(pixel(4, 4), [0, 0, 0, 255]);
    assert_eq!(
        pixel(20, 10),
        [0, 0, 0, 255],
        "wireframe must not fill the interior"
    );
    let wide = serial.is_multiple_of(2);
    for (x, covered) in [(15, true), (14, wide), (13, false)] {
        let actual = pixel(x, 8);
        if covered && (wide || !smooth) {
            // For a width-one non-smooth line exactly one side of x=16 is
            // selected by the top-left rule; test its count separately below.
            if wide {
                assert!(actual[0] >= 254, "width=4 pixel({x},8)={actual:?}");
            }
        } else if !covered {
            assert_eq!(actual, [0, 0, 0, 255], "outside line width: ({x},8)");
        }
    }
    let row = (12..20).map(|x| pixel(x, 8)).collect::<Vec<_>>();
    let drawn = row.iter().filter(|p| p[0] != 0).count();
    if smooth {
        assert!(
            drawn >= if wide { 4 } else { 2 },
            "smooth width coverage: {row:?}"
        );
        if !wide {
            // Coverage rules have implementation-dependent precision. Test a
            // genuine partial alpha, not a fabricated vendor-independent exact
            // byte, at a half-covered axis-aligned edge.
            assert!(
                row.iter().any(|p| p[0] > 0 && p[3] > 0 && p[3] < 255),
                "missing coverage alpha: {row:?}"
            );
        }
    } else {
        assert_eq!(
            drawn,
            if wide { 4 } else { 1 },
            "rectangular line width: {row:?}"
        );
        assert!(row.iter().all(|p| p[3] == 255));
    }
}

fn color_output(serial: u64) -> ColorOutputState {
    use BlendFactor as F;
    use BlendOperation as O;
    let component = |operation, source, destination| BlendComponent {
        operation,
        source,
        destination,
    };
    match serial {
        1 => ColorOutputState {
            blend: Some(ColorBlendState {
                color: component(O::Add, F::SourceAlpha, F::OneMinusSourceAlpha),
                alpha: component(O::Add, F::One, F::Zero),
            }),
            write_mask: ColorWriteMask::new(true, false, true, true),
        },
        2 => ColorOutputState {
            blend: None,
            write_mask: ColorWriteMask::new(false, true, false, false),
        },
        3 => ColorOutputState {
            blend: Some(ColorBlendState {
                color: component(O::ReverseSubtract, F::One, F::One),
                alpha: component(O::Max, F::One, F::One),
            }),
            write_mask: ColorWriteMask::ALL,
        },
        4 => ColorOutputState {
            blend: Some(ColorBlendState {
                color: component(O::Min, F::One, F::One),
                alpha: component(O::Min, F::One, F::One),
            }),
            write_mask: ColorWriteMask::ALL,
        },
        _ => unreachable!(),
    }
}

fn residency_pressure(runtime: &mut dyn NeutralBackendRuntime) {
    // Fill the production 4096-object budget with zero-byte logical allocations.
    // Touch each new batch so newly created cold objects cannot indefinitely
    // evict each other instead of the older native input buffer. This is real
    // public-runtime pressure, not a configurable test policy or a driver hook.
    for batch in 0..33 {
        let range = if batch == 0 {
            10_000..14_080
        } else {
            14_080 + batch..14_081 + batch
        };
        let creations: Vec<_> = range
            .map(|id| BackendResourceCreateInfo::Allocation {
                id: GpuAllocationId::new(id),
                description: GpuAllocationDescription::new(4, 4).unwrap(),
            })
            .collect();
        let submission = OperationSubmission::new(
            FrontendSubmissionId::new(10_000 + batch),
            vec![],
            vec![GpuOperation::new(
                GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateSamplerCaches),
                [],
                creations.iter().map(BackendResourceCreateInfo::dependency),
                CapabilityRequirements::none(),
            )],
        )
        .unwrap();
        runtime.submit(&creations, &[], &submission).unwrap();
    }
}
