//! Physical half FMA oracle uses independent F64 arithmetic and half conversion.
use super::*;
use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, DescriptorKind, DescriptorTableBinding,
    DescriptorTableDescription, DescriptorTableId, DispatchOperation, PipelineStages,
    ResourceAccess, ResourceUsage, ShaderBackendModule, ShaderComputeBuiltin, ShaderFloatControl,
    ShaderResourceAccess, ShaderResourceKind,
};

#[test]
fn half_fma_preserves_direct_rounding_overflow_subnormals_and_saturation() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(769),
        NonCpuDeviceId::new(769),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let mut cases = vec![
        [0x3e00, 0x3c01, 0x8001],
        [0x3e00, 0x3c03, 1],
        [0x4b80, 0x6c44, 0x8001],
        [0x4b80, 0x6c44, 1],
        [1, 0x3800, 0],
        [1, 0x3800, 1],
        [0x8000, 0x3c00, 0x8000],
        [0x7c00, 0, 0x3c00],
        [0x7e00, 0x3c00, 0],
        [0x7c00, 0x3c00, 0xfc00],
        [0xfc00, 0x3c00, 0],
    ];
    let mut seed = 0x82a6_f733_u32;
    for mantissa in (1..1024).step_by(2) {
        for a in [0x3e00, 0xbe00] {
            for c in [1, 0x8001] {
                cases.push([a, 0x3c00 | mantissa, c]);
            }
        }
    }
    for _ in 0..4096 {
        cases.push(std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u16
        }));
    }
    let bytes = cases
        .iter()
        .flat_map(|triple| {
            triple
                .map(|bits| half::f16::from_bits(bits).to_f32().to_bits())
                .into_iter()
                .chain([0, 0])
        })
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let canonical = CanonicalAllocation::zeroed(bytes.len(), 4096).unwrap();
    canonical.write(0, &bytes).unwrap();
    let allocation = GpuAllocationId::new(769);
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
    let buffer = BufferId::new(769);
    let description = BufferDescription::new(bytes.len() as u64).unwrap();
    let r = ShaderRegister::new;
    let mut ops = vec![
        ShaderOperation::LoadComputeBuiltin32 {
            destination: r(0),
            builtin: ShaderComputeBuiltin::GlobalInvocationId,
            component: 0,
        },
        ShaderOperation::MoveImmediate32 {
            destination: r(1),
            bits: 5,
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
    for destination in [r(3), r(4), r(5)] {
        ops.push(ShaderOperation::LoadStorageBuffer32 {
            destination,
            binding: 0,
            word_index: r(2),
        });
        ops.push(increment.clone());
    }
    for saturate in [false, true] {
        ops.push(ShaderOperation::FusedMultiplyAddHalf32 {
            destination: r(6),
            left: r(3),
            right: r(4),
            addend: r(5),
            saturate,
        });
        ops.push(ShaderOperation::StoreStorageBuffer32 {
            source: r(6),
            binding: 0,
            word_index: r(2),
        });
        ops.push(increment.clone());
    }
    ops.push(ShaderOperation::Exit);
    let module = ShaderBackendModule::new(
        VerifiedShaderIr::verify(
            ShaderIr::new(
                ShaderStage::Compute,
                vec![],
                vec![],
                vec![
                    ShaderResourceAccess::new(0, ShaderResourceKind::StorageBuffer, true, true)
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
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(BufferView::new(buffer, description, 0, backing.clone()).unwrap()),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(769),
            description: PipelineDescription {
                kind: PipelineKind::Compute,
            },
        },
        BackendResourceCreateInfo::DescriptorTable {
            id: DescriptorTableId::new(769),
            description: DescriptorTableDescription::new(vec![DescriptorKind::Buffer]).unwrap(),
            bindings: vec![DescriptorTableBinding {
                binding: 0,
                resource: ResourceDependency::Buffer(buffer),
            }]
            .into(),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(769),
            description: ShaderDescription {
                stage: ShaderStage::Compute,
            },
            module,
        },
    ];
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(1),
        vec![],
        vec![GpuOperation::new(
            GpuCommand::Dispatch(
                DispatchOperation::new(
                    PipelineId::new(769),
                    ShaderId::new(769),
                    vec![DescriptorTableId::new(769)],
                    [cases.len() as u32, 1, 1],
                )
                .unwrap(),
            ),
            [ResourceAccess::new(
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
            )],
            [ResourceDependency::Buffer(buffer)],
            CapabilityRequirements::none(),
        )],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(&creations, &[], &submission)
        .unwrap();
    let mut result = vec![0; bytes.len()];
    backing.range().read(0, &mut result).unwrap();
    for (index, triple) in cases.iter().enumerate() {
        let [a, b, c] = triple.map(|bits| half::f16::from_bits(bits).to_f64());
        let fused = a.mul_add(b, c);
        for (lane, saturate) in [(3, false), (4, true)] {
            let value = if saturate {
                if fused.is_nan() || fused.is_sign_negative() {
                    0.0
                } else {
                    fused.min(1.0)
                }
            } else {
                fused
            };
            let expected = reference_round(value);
            let offset = (index * 5 + lane) * 4;
            let actual = u32::from_le_bytes(result[offset..offset + 4].try_into().unwrap());
            if value.is_nan() {
                assert_eq!(actual & 0xffff_7c00, 0x7c00);
                assert_ne!(actual & 0x03ff, 0);
            } else {
                assert_eq!(
                    actual,
                    u32::from(expected),
                    "triple={triple:04x?}, saturation={saturate}"
                );
            }
        }
    }
}

// Search exact half values; independent of the shader's bit rounding and of
// half::from_f64, which may double-round through the host F16C instruction.
fn reference_round(value: f64) -> u16 {
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7e00;
    }
    if magnitude >= 65520.0 {
        return sign | 0x7c00;
    }
    let mut low = 0_u16;
    let mut high = 0x7bff_u16;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if half::f16::from_bits(middle).to_f64() <= magnitude {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    if low == 0x7bff {
        return sign | low;
    }
    let above = low + 1;
    let below_distance = magnitude - half::f16::from_bits(low).to_f64();
    let above_distance = half::f16::from_bits(above).to_f64() - magnitude;
    sign | if above_distance < below_distance
        || (above_distance == below_distance && above & 1 == 0)
    {
        above
    } else {
        low
    }
}
