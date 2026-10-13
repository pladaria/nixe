//! Verify all six physical cube faces, mip selection and dirty imports.
use super::*;
use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, AddressMode, DescriptorKind, DescriptorTableBinding,
    DescriptorTableDescription, DescriptorTableId, DispatchOperation, FilterMode, PipelineStages,
    ResourceAccess, ResourceUsage, SamplerDescription, SamplerId, ShaderBackendModule,
    ShaderComputeBuiltin, ShaderFloatControl, ShaderResourceAccess, ShaderResourceKind,
    ShaderTextureSampleOutput,
};

#[test]
fn cube_sampling_preserves_six_faces_mips_queries_and_dirty_uploads() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(770),
        NonCpuDeviceId::new(770),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let colors = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
        [255, 0, 255, 255],
        [0, 255, 255, 255],
    ];
    let directions = [
        [1_f32, 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ];
    let mut texels = Vec::new();
    for face in 0..6 {
        for (mip, count) in [(0, 16), (1, 4)] {
            for _ in 0..count {
                texels.extend(colors[(face + mip) % 6]);
            }
        }
    }
    let texture = CanonicalAllocation::zeroed(texels.len(), 4096).unwrap();
    texture.write(0, &texels).unwrap();
    let texture_allocation = GpuAllocationId::new(770);
    let texture_allocation_description =
        GpuAllocationDescription::new(texels.len() as u64, 4).unwrap();
    let texture_backing = BackingView::new(
        texture_allocation,
        texture_allocation_description,
        0,
        texture
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let image = ImageId::new(770);
    let image_description = ImageDescription::new(
        ImageDimension::Cube,
        ImageExtent::new(4, 4, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        2,
        6,
        SampleCount::One,
    )
    .unwrap();
    let view = ImageView::new(
        image,
        image_description,
        Swizzle::IDENTITY,
        (0..6)
            .flat_map(|face| {
                let texture_backing = &texture_backing;
                [(0, 0, 64, 16), (1, 64, 16, 8)].map(move |(mip, offset, size, pitch)| {
                    let offset = face * 80 + offset;
                    (
                        ImageSubresourceRange {
                            plane: 0,
                            mip_level: mip,
                            base_layer: face as u16,
                            layer_count: 1,
                        },
                        ImageMemoryLayout::PitchLinear {
                            row_pitch: pitch,
                            layer_stride: size,
                        },
                        BackingView::new(
                            texture_allocation,
                            texture_allocation_description,
                            offset,
                            texture_backing
                                .range()
                                .snapshot_subrange(offset, size)
                                .unwrap(),
                        )
                        .unwrap(),
                    )
                })
            })
            .collect(),
    )
    .unwrap();
    let mut inputs = Vec::new();
    for direction in directions {
        for level in [0_f32, 1.] {
            inputs.extend(direction.map(f32::to_bits));
            inputs.push(level.to_bits());
            inputs.extend([0; 5]);
        }
    }
    let bytes = inputs
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let canonical = CanonicalAllocation::zeroed(bytes.len(), 4096).unwrap();
    canonical.write(0, &bytes).unwrap();
    let allocation = GpuAllocationId::new(771);
    let allocation_description = GpuAllocationDescription::new(bytes.len() as u64, 4).unwrap();
    let backing = BackingView::new(
        allocation,
        allocation_description,
        0,
        canonical
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let buffer = BufferId::new(771);
    let description = BufferDescription::new(bytes.len() as u64).unwrap();
    let sampler = SamplerId::new(770);
    let r = ShaderRegister::new;
    let mut ops = vec![
        ShaderOperation::LoadComputeBuiltin32 {
            destination: r(0),
            builtin: ShaderComputeBuiltin::GlobalInvocationId,
            component: 0,
        },
        ShaderOperation::MoveImmediate32 {
            destination: r(1),
            bits: 9,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::Multiply32 {
            destination: r(2),
            left: r(0),
            right: r(1),
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        },
        ShaderOperation::MoveImmediate32 {
            destination: r(1),
            bits: 1,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    ];
    let increment = ShaderOperation::Add32 {
        destination: r(2),
        left: r(2),
        right: r(1),
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    };
    for destination in [r(3), r(4), r(5), r(6)] {
        ops.push(ShaderOperation::LoadStorageBuffer32 {
            destination,
            binding: 2,
            word_index: r(2),
        });
        ops.push(increment.clone());
    }
    ops.push(ShaderOperation::SampleTextureCube {
        outputs: (0..4)
            .map(|i| ShaderTextureSampleOutput::new(r(7 + i), i as u8).unwrap())
            .collect(),
        coordinates: [r(3), r(4), r(5)],
        image_binding: 0,
        lod: Some(r(6)),
        sampler_binding: 1,
    });
    for source in [r(7), r(8), r(9), r(10)] {
        ops.push(ShaderOperation::StoreStorageBuffer32 {
            source,
            binding: 2,
            word_index: r(2),
        });
        ops.push(increment.clone());
    }
    ops.push(ShaderOperation::QueryTextureLevels {
        destination: r(7),
        image_binding: 0,
    });
    ops.push(ShaderOperation::StoreStorageBuffer32 {
        source: r(7),
        binding: 2,
        word_index: r(2),
    });
    ops.push(ShaderOperation::Exit);
    let module = ShaderBackendModule::new(
        VerifiedShaderIr::verify(
            ShaderIr::new(
                ShaderStage::Compute,
                vec![],
                vec![],
                vec![
                    ShaderResourceAccess::new(0, ShaderResourceKind::SampledImageCube, true, false)
                        .unwrap(),
                    ShaderResourceAccess::new(1, ShaderResourceKind::Sampler, true, false).unwrap(),
                    ShaderResourceAccess::new(2, ShaderResourceKind::StorageBuffer, true, true)
                        .unwrap(),
                ],
                ops.into_iter()
                    .enumerate()
                    .map(|(i, op)| {
                        ShaderInstruction::new(
                            ShaderSourceLocation::new(i as u32 * 8),
                            ShaderPredicate::Always,
                            op,
                        )
                    })
                    .collect(),
            )
            .with_workgroup_size([1, 1, 1]),
        )
        .unwrap(),
    );
    let creations = [
        BackendResourceCreateInfo::Allocation {
            id: texture_allocation,
            description: texture_allocation_description,
        },
        BackendResourceCreateInfo::Image {
            id: image,
            description: image_description,
            view: Some(view),
        },
        BackendResourceCreateInfo::Sampler {
            id: sampler,
            description: SamplerDescription::new(
                FilterMode::Linear,
                FilterMode::Linear,
                FilterMode::Nearest,
                [AddressMode::ClampToEdge; 3],
                0.,
                1.,
                1.,
            )
            .unwrap(),
        },
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(BufferView::new(buffer, description, 0, backing.clone()).unwrap()),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(770),
            description: PipelineDescription {
                kind: PipelineKind::Compute,
            },
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(770),
            description: ShaderDescription {
                stage: ShaderStage::Compute,
            },
            module,
        },
        BackendResourceCreateInfo::DescriptorTable {
            id: DescriptorTableId::new(770),
            description: DescriptorTableDescription::new(vec![
                DescriptorKind::SampledImage,
                DescriptorKind::Sampler,
                DescriptorKind::Buffer,
            ])
            .unwrap(),
            bindings: vec![
                DescriptorTableBinding {
                    binding: 0,
                    resource: ResourceDependency::Image(image),
                },
                DescriptorTableBinding {
                    binding: 1,
                    resource: ResourceDependency::Sampler(sampler),
                },
                DescriptorTableBinding {
                    binding: 2,
                    resource: ResourceDependency::Buffer(buffer),
                },
            ]
            .into(),
        },
    ];
    for serial in [1, 2] {
        if serial == 2 {
            texture.write(2 * 80 + 64, &[255; 16]).unwrap();
        }
        let submission = OperationSubmission::new(
            FrontendSubmissionId::new(serial),
            vec![],
            vec![GpuOperation::new(
                GpuCommand::Dispatch(
                    DispatchOperation::new(
                        PipelineId::new(770),
                        ShaderId::new(770),
                        vec![DescriptorTableId::new(770)],
                        [12, 1, 1],
                    )
                    .unwrap(),
                ),
                (0..2)
                    .map(|mip| {
                        ResourceAccess::new(
                            AccessTarget::Image {
                                image,
                                subresources: ImageSubresourceRange {
                                    plane: 0,
                                    mip_level: mip,
                                    base_layer: 0,
                                    layer_count: 6,
                                },
                            },
                            AccessScope::new(
                                PipelineStages::COMPUTE_SHADER,
                                AccessMode::Read,
                                ResourceUsage::SampledImage,
                            )
                            .unwrap(),
                        )
                    })
                    .chain([ResourceAccess::new(
                        AccessTarget::Buffer {
                            buffer,
                            range: BufferRange::new(0, bytes.len() as u64).unwrap(),
                        },
                        AccessScope::new(
                            PipelineStages::COMPUTE_SHADER,
                            AccessMode::ReadWrite,
                            ResourceUsage::StorageBuffer,
                        )
                        .unwrap(),
                    )]),
                [
                    ResourceDependency::Image(image),
                    ResourceDependency::Buffer(buffer),
                    ResourceDependency::Sampler(sampler),
                ],
                CapabilityRequirements::none(),
            )],
        )
        .unwrap();
        runtime
            .runtime()
            .submit(if serial == 1 { &creations } else { &[] }, &[], &submission)
            .unwrap();
        let mut result = vec![0; bytes.len()];
        backing.range().read(0, &mut result).unwrap();
        for face in 0..6 {
            for mip in 0..2 {
                let expected = if serial == 2 && face == 2 && mip == 1 {
                    [255; 4]
                } else {
                    colors[(face + mip) % 6]
                };
                for (component, &channel) in expected.iter().enumerate() {
                    let offset = ((face * 2 + mip) * 9 + 4 + component) * 4;
                    let actual = f32::from_bits(u32::from_le_bytes(
                        result[offset..offset + 4].try_into().unwrap(),
                    ));
                    assert_eq!(
                        actual,
                        (f32::from(channel) / 255.),
                        "face={face} mip={mip} component={component}"
                    );
                }
                let offset = ((face * 2 + mip) * 9 + 8) * 4;
                assert_eq!(
                    u32::from_le_bytes(result[offset..offset + 4].try_into().unwrap()),
                    2
                );
            }
        }
    }
}
