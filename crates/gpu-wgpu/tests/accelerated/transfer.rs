//! Device transforms preserve byte order, overlap, padding and CPU visibility.
use super::*;
use nixe_gpu::{BufferTransform, TransferComponent as C, TransferLayout as L};
fn position(layout: L, x: u32, y: u32) -> usize {
    match layout {
        L::Pitch { pitch } => (y * pitch + x) as usize,
        L::BlockLinear {
            row_bytes,
            origin_x_bytes,
            origin_y,
            block_height_log2,
        } => {
            let x = x + origin_x_bytes;
            let y = y + origin_y;
            let h = 1 << block_height_log2;
            (y / (8 * h) * 512 * h * row_bytes.div_ceil(64)
                + x / 64 * 512 * h
                + y % (8 * h) / 8 * 512
                + x % 64 / 32 * 256
                + y % 8 / 2 * 64
                + x % 32 / 16 * 32
                + y % 2 * 16
                + x % 16) as usize
        }
    }
}
#[test]
fn device_transforms_cover_layouts_remaps_overlap_and_inflight_cpu_reads() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(764),
        NonCpuDeviceId::new(764),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let canonical = CanonicalAllocation::zeroed(65536, 4096).unwrap();
    let allocation = GpuAllocationId::new(1);
    let buffer = BufferId::new(1);
    let description = GpuAllocationDescription::new(65536, 4).unwrap();
    let backing = BackingView::new(
        allocation,
        description,
        0,
        canonical
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let buffer_description = BufferDescription::new(65536).unwrap();
    let creations = [
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description: buffer_description,
            view: Some(BufferView::new(buffer, buffer_description, 0, backing.clone()).unwrap()),
        },
    ];
    let mut serial = 0;
    for block_height_log2 in [0, 1, 3, 5] {
        for component_bytes in [1, 2, 4] {
            for source_tiled in [false, true] {
                for destination_tiled in [false, true] {
                    serial += 1;
                    let layout = |tiled| {
                        if tiled {
                            L::BlockLinear {
                                row_bytes: 256,
                                origin_x_bytes: 4 * u32::from(component_bytes),
                                origin_y: 3,
                                block_height_log2,
                            }
                        } else {
                            L::Pitch { pitch: 160 }
                        }
                    };
                    let copy = BufferTransform {
                        source: BufferRegion {
                            buffer,
                            range: BufferRange::new(5, 32768).unwrap(),
                        },
                        destination: BufferRegion {
                            buffer,
                            range: BufferRange::new(293, 32768).unwrap(),
                        },
                        source_layout: layout(source_tiled),
                        destination_layout: layout(destination_tiled),
                        width: 7,
                        height: 11,
                        component_bytes,
                        source_components: 4,
                        destination_components: 4,
                        components: [C::Source(2), C::ConstantA, C::ConstantB, C::Preserve],
                        constant_a: 0x12345678,
                        constant_b: 0x89abcdef,
                    };
                    let input: Arc<[u8]> = (0..65536)
                        .map(|i| (i * 7 + serial as usize) as u8)
                        .collect::<Vec<_>>()
                        .into();
                    let mut expected = input.to_vec();
                    for y in 0..copy.height {
                        for x in 0..copy.width {
                            let source = 5 + position(
                                copy.source_layout,
                                x * 4 * u32::from(component_bytes),
                                y,
                            );
                            let destination = 293
                                + position(
                                    copy.destination_layout,
                                    x * 4 * u32::from(component_bytes),
                                    y,
                                );
                            for (component, value) in copy.components.iter().enumerate() {
                                for byte in 0..usize::from(component_bytes) {
                                    let value = match value {
                                        C::Source(index) => {
                                            input[source
                                                + usize::from(*index)
                                                    * usize::from(component_bytes)
                                                + byte]
                                        }
                                        C::ConstantA => copy.constant_a.to_le_bytes()[byte],
                                        C::ConstantB => copy.constant_b.to_le_bytes()[byte],
                                        C::Preserve => continue,
                                    };
                                    expected[destination
                                        + component * usize::from(component_bytes)
                                        + byte] = value;
                                }
                            }
                        }
                    }
                    let operations = vec![
                        GpuOperation::new(
                            GpuCommand::UploadBuffer {
                                destination: BufferRegion {
                                    buffer,
                                    range: BufferRange::new(0, 65536).unwrap(),
                                },
                                bytes: input,
                            },
                            [],
                            [],
                            CapabilityRequirements::none(),
                        ),
                        GpuOperation::new(
                            GpuCommand::TransformBuffer(copy),
                            [],
                            [],
                            CapabilityRequirements::none(),
                        ),
                    ];
                    let submission = OperationSubmission::new(
                        FrontendSubmissionId::new(serial),
                        vec![],
                        operations,
                    )
                    .unwrap();
                    runtime
                        .runtime()
                        .submit(if serial == 1 { &creations } else { &[] }, &[], &submission)
                        .unwrap();
                    // Read before explicit completion; this must resolve the actual GPU write.
                    let mut actual = vec![0; 65536];
                    backing.range().read(0, &mut actual).unwrap();
                    assert_eq!(
                        actual, expected,
                        "bytes={component_bytes} source_tiled={source_tiled} destination_tiled={destination_tiled}"
                    );
                }
            }
        }
    }
}

#[test]
fn batched_upload_versions_retire_after_their_token_and_preserve_partial_cpu_writes() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(765),
        NonCpuDeviceId::new(765),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let allocation_description = GpuAllocationDescription::new(64, 4).unwrap();
    let buffer_description = BufferDescription::new(64).unwrap();
    let mut creations = Vec::new();
    let backings: Vec<_> = (1..=3)
        .map(|id| {
            let bytes = CanonicalAllocation::zeroed(64, 4096).unwrap();
            bytes.write(0, &[0xaa; 64]).unwrap();
            let backing = BackingView::new(
                GpuAllocationId::new(id),
                allocation_description,
                0,
                bytes.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
            )
            .unwrap();
            creations.push(BackendResourceCreateInfo::Allocation {
                id: GpuAllocationId::new(id),
                description: allocation_description,
            });
            creations.push(BackendResourceCreateInfo::Buffer {
                id: BufferId::new(id),
                description: buffer_description,
                view: Some(
                    BufferView::new(BufferId::new(id), buffer_description, 0, backing.clone())
                        .unwrap(),
                ),
            });
            backing
        })
        .collect();
    let region = |id, offset, size| BufferRegion {
        buffer: BufferId::new(id),
        range: BufferRange::new(offset, size).unwrap(),
    };
    let op = |command| GpuOperation::new(command, [], [], CapabilityRequirements::none());
    let upload = |offset, value| {
        op(GpuCommand::UploadBuffer {
            destination: region(1, offset, 4),
            bytes: Arc::from([value; 4]),
        })
    };
    let copy = |from, to| {
        op(GpuCommand::Copy(
            CopyOperation::buffer_to_buffer(from, to).unwrap(),
        ))
    };
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(1),
        vec![],
        vec![
            upload(0, 1),
            copy(region(1, 0, 16), region(2, 0, 16)),
            upload(0, 2),
            copy(region(1, 0, 16), region(2, 16, 16)),
            upload(4, 3),
            copy(region(1, 0, 16), region(3, 0, 16)),
            copy(region(3, 0, 16), region(2, 32, 16)),
        ],
    )
    .unwrap();
    let token = runtime
        .runtime()
        .submit(
            &creations,
            &[
                ResourceDependency::Buffer(BufferId::new(1)),
                ResourceDependency::Buffer(BufferId::new(3)),
            ],
            &submission,
        )
        .unwrap();
    let mut result = [0; 64];
    // No explicit wait: a canonical read must resolve the real in-flight token.
    backings[1].range().read(0, &mut result).unwrap();
    let mut expected = [0xaa; 64];
    expected[..4].fill(1);
    expected[16..20].fill(2);
    expected[32..36].fill(2);
    expected[36..40].fill(3);
    assert_eq!(result, expected);
    assert_eq!(
        runtime
            .runtime()
            .wait_for_completion()
            .unwrap()
            .unwrap()
            .submission(),
        token
    );
    // The retired device representation remains authoritative until demand.
    let mut source = [0; 64];
    backings[0].range().read(0, &mut source).unwrap();
    assert_eq!(&source[..8], &[2, 2, 2, 2, 3, 3, 3, 3]);
    let mut writes = nixe_memory::CanonicalWriteBatch::new();
    writes.stage(backings[0].range(), 8, &[4; 4]).unwrap();
    writes.commit().unwrap();
    let alias = BufferId::new(4);
    let creation = BackendResourceCreateInfo::Buffer {
        id: alias,
        description: buffer_description,
        view: Some(BufferView::new(alias, buffer_description, 0, backings[0].clone()).unwrap()),
    };
    let next = OperationSubmission::new(
        FrontendSubmissionId::new(2),
        vec![FrontendSubmissionId::new(1)],
        vec![copy(region(4, 0, 16), region(2, 48, 16))],
    )
    .unwrap();
    runtime.runtime().submit(&[creation], &[], &next).unwrap();
    backings[1].range().read(48, &mut result[..16]).unwrap();
    assert_eq!(
        &result[..16],
        &[2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0xaa, 0xaa, 0xaa, 0xaa]
    );
}

#[test]
fn cpu_updates_of_read_only_inputs_preserve_each_inflight_submission_snapshot() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(766),
        NonCpuDeviceId::new(766),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let source = CanonicalAllocation::zeroed(64, 4096).unwrap();
    let destination = CanonicalAllocation::zeroed(64, 4096).unwrap();
    source.write(0, &[0xaa; 64]).unwrap();
    let description = GpuAllocationDescription::new(64, 4).unwrap();
    let buffer_description = BufferDescription::new(64).unwrap();
    let backings: Vec<_> = [&source, &destination]
        .into_iter()
        .enumerate()
        .map(|(index, bytes)| {
            BackingView::new(
                GpuAllocationId::new(index as u64 + 1),
                description,
                0,
                bytes.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let creations: Vec<_> = backings
        .iter()
        .enumerate()
        .flat_map(|(index, backing)| {
            let id = index as u64 + 1;
            [
                BackendResourceCreateInfo::Allocation {
                    id: GpuAllocationId::new(id),
                    description,
                },
                BackendResourceCreateInfo::Buffer {
                    id: BufferId::new(id),
                    description: buffer_description,
                    view: Some(
                        BufferView::new(BufferId::new(id), buffer_description, 0, backing.clone())
                            .unwrap(),
                    ),
                },
            ]
        })
        .collect();
    let retired = [ResourceDependency::Buffer(BufferId::new(1))];
    for serial in 1..=3 {
        if serial > 1 {
            source
                .write((serial as usize - 1) * 4, &[serial as u8; 4])
                .unwrap();
        }
        let region = |id, offset| BufferRegion {
            buffer: BufferId::new(id),
            range: BufferRange::new(offset, 16).unwrap(),
        };
        let submission = OperationSubmission::new(
            FrontendSubmissionId::new(serial),
            if serial == 1 {
                vec![]
            } else {
                vec![FrontendSubmissionId::new(serial - 1)]
            },
            vec![GpuOperation::new(
                GpuCommand::Copy(
                    CopyOperation::buffer_to_buffer(region(1, 0), region(2, (serial - 1) * 16))
                        .unwrap(),
                ),
                [],
                [],
                CapabilityRequirements::none(),
            )],
        )
        .unwrap();
        runtime
            .runtime()
            .submit(
                if serial == 1 { &creations } else { &[] },
                if serial == 3 { &retired } else { &[] },
                &submission,
            )
            .unwrap();
        assert!(!matches!(
            source.pages()[0].visibility_state(),
            nixe_memory::VisibilityState::GpuNewer { .. }
        ));
    }
    let mut actual = [0; 64];
    backings[1].range().read(0, &mut actual).unwrap();
    let mut expected = [0; 64];
    expected[..48].fill(0xaa);
    expected[20..24].fill(2);
    expected[36..40].fill(2);
    expected[40..44].fill(3);
    assert_eq!(actual, expected);
}

#[test]
fn known_uploads_merge_with_real_gpu_writes_across_aliases_retirement_and_cpu_epochs() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(766),
        NonCpuDeviceId::new(766),
        WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let canonical = CanonicalAllocation::zeroed(64, 4096).unwrap();
    let source = CanonicalAllocation::zeroed(64, 4096).unwrap();
    let allocation_description = GpuAllocationDescription::new(64, 4).unwrap();
    let description = BufferDescription::new(64).unwrap();
    let backing = |id, canonical: &CanonicalAllocation| {
        BackingView::new(
            GpuAllocationId::new(id),
            allocation_description,
            0,
            canonical
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
        )
        .unwrap()
    };
    let shared = backing(1, &canonical);
    let other = backing(2, &source);
    let mut creations = vec![
        BackendResourceCreateInfo::Allocation {
            id: GpuAllocationId::new(1),
            description: allocation_description,
        },
        BackendResourceCreateInfo::Allocation {
            id: GpuAllocationId::new(2),
            description: allocation_description,
        },
    ];
    for (id, backing) in [(1, shared.clone()), (2, shared.clone()), (3, other.clone())] {
        creations.push(BackendResourceCreateInfo::Buffer {
            id: BufferId::new(id),
            description,
            view: Some(BufferView::new(BufferId::new(id), description, 0, backing).unwrap()),
        });
    }
    let region = |id, offset, size| BufferRegion {
        buffer: BufferId::new(id),
        range: BufferRange::new(offset, size).unwrap(),
    };
    let op = |command| GpuOperation::new(command, [], [], CapabilityRequirements::none());
    let upload = |id, offset, size, value| {
        op(GpuCommand::UploadBuffer {
            destination: region(id, offset, size),
            bytes: vec![value; size as usize].into(),
        })
    };
    let copy = |source, destination| {
        op(GpuCommand::Copy(
            CopyOperation::buffer_to_buffer(source, destination).unwrap(),
        ))
    };
    let submission = OperationSubmission::new(
        FrontendSubmissionId::new(1),
        vec![],
        vec![
            upload(1, 0, 64, 1),
            upload(2, 8, 4, 2),
            upload(3, 0, 64, 3),
            copy(region(3, 0, 16), region(1, 16, 16)),
            upload(2, 20, 4, 4),
        ],
    )
    .unwrap();
    runtime
        .runtime()
        .submit(
            &creations,
            &[ResourceDependency::Buffer(BufferId::new(2))],
            &submission,
        )
        .unwrap();
    let mut expected = [1; 64];
    expected[8..12].fill(2);
    expected[16..32].fill(3);
    expected[20..24].fill(4);
    let mut actual = [0; 64];
    // Demand before explicit completion, including a retired alias and unknown holes.
    shared.range().read(0, &mut actual).unwrap();
    assert_eq!(actual, expected);
    canonical.write(32, &[5; 4]).unwrap();
    expected[32..36].fill(5);
    expected[4..8].fill(6);
    let second = OperationSubmission::new(
        FrontendSubmissionId::new(2),
        vec![],
        vec![upload(1, 4, 4, 6), copy(region(1, 0, 64), region(3, 0, 64))],
    )
    .unwrap();
    runtime.runtime().submit(&[], &[], &second).unwrap();
    other.range().read(0, &mut actual).unwrap();
    assert_eq!(actual, expected);
}
