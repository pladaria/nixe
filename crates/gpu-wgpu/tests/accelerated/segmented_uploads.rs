use super::*;
use nixe_gpu::FrontendSubmissionSegment;

const SNAPSHOTS: u32 = 17;
const BYTES: usize = SNAPSHOTS as usize * 64;

#[test]
fn segmented_uploads_keep_each_snapshot_and_flush_on_final_wait_or_readback() {
    let _guard = accelerated_test_guard();
    for mode in 0..3 {
        let Some(initialized) = initialize_backend(
            BackendInstanceId::new(8200 + mode),
            NonCpuDeviceId::new(8200 + mode),
            Default::default(),
        ) else {
            return;
        };
        let runtime = RuntimeOwner::new(initialized.into_runtime());
        let source_page = initialized_page(&[0x3c; BYTES]);
        let destination_page = initialized_page(&[0; BYTES]);
        let description = GpuAllocationDescription::new(BYTES as u64, 4).unwrap();
        let source = BufferId::new(1);
        let destination = BufferId::new(2);
        let source_backing = backing(GpuAllocationId::new(1), description, &source_page);
        let destination_backing = backing(GpuAllocationId::new(2), description, &destination_page);
        let creations = [
            (source, source_backing.clone()),
            (destination, destination_backing.clone()),
        ]
        .into_iter()
        .flat_map(|(id, backing)| {
            [
                BackendResourceCreateInfo::Allocation {
                    id: backing.allocation(),
                    description,
                },
                BackendResourceCreateInfo::Buffer {
                    id,
                    description: BufferDescription::new(BYTES as u64).unwrap(),
                    view: Some(
                        BufferView::new(
                            id,
                            BufferDescription::new(BYTES as u64).unwrap(),
                            0,
                            backing,
                        )
                        .unwrap(),
                    ),
                },
            ]
        })
        .collect::<Vec<_>>();
        let segment = |ordinal, final_segment| {
            OperationSubmission::new_segment(
                FrontendSubmissionId::new(1),
                FrontendSubmissionSegment::new(ordinal),
                final_segment,
                vec![],
                vec![GpuOperation::new(
                    GpuCommand::Copy(
                        CopyOperation::buffer_to_buffer(
                            BufferRegion {
                                buffer: source,
                                range: BufferRange::new(0, 64).unwrap(),
                            },
                            BufferRegion {
                                buffer: destination,
                                range: BufferRange::new(u64::from(ordinal) * 64, 64).unwrap(),
                            },
                        )
                        .unwrap(),
                    ),
                    [],
                    [],
                    CapabilityRequirements::none(),
                )],
            )
            .unwrap()
        };
        let mut tokens = Vec::new();
        for ordinal in 0..SNAPSHOTS {
            if ordinal != 0 {
                // Mutating the same CPU-owned source cannot alter a snapshot
                // captured by an earlier segment, queued or already submitted.
                let mut write = nixe_memory::CanonicalWriteBatch::new();
                write
                    .stage(source_backing.range(), 0, &[ordinal as u8; 64])
                    .unwrap();
                write.commit().unwrap();
            }
            tokens.push(
                runtime
                    .runtime()
                    .submit(
                        if ordinal == 0 { &creations } else { &[] },
                        &[],
                        &segment(ordinal, mode == 0 && ordinal == SNAPSHOTS - 1),
                    )
                    .unwrap(),
            );
            if ordinal < 7 {
                // Accepted input snapshots are not completion evidence while
                // their first bounded batch is still being assembled.
                assert!(runtime.runtime().poll_completion().unwrap().is_none());
            }
        }
        let mut bytes = [0; BYTES];
        if mode == 2 {
            // Demanded readback must flush queued work before its own transfer.
            destination_backing.range().read(0, &mut bytes).unwrap();
        }
        for token in tokens {
            let completed = runtime.runtime().wait_for_completion().unwrap().unwrap();
            assert_eq!(completed.submission(), token);
        }
        destination_backing.range().read(0, &mut bytes).unwrap();
        for (ordinal, snapshot) in bytes.chunks_exact(64).enumerate() {
            assert_eq!(
                snapshot,
                &[if ordinal == 0 { 0x3c } else { ordinal as u8 }; 64]
            );
        }
        runtime.runtime().teardown().unwrap();
    }
}
