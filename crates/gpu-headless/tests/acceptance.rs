use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, BackendCapabilities, BackendError, BackendFeatures,
    BackendInstanceId, BackendLimits, BackendResourceCreateInfo, BackendState, BackingView,
    BarrierOperation, BufferDescription, BufferId, BufferRange, BufferView, CapabilityRequirements,
    ClearOperation, GpuAllocationDescription, GpuAllocationId, GpuCommand, GpuOperation,
    ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind,
    OperationSubmission, PipelineStages, RenderPassDescription, RenderPassId, RenderPassOperation,
    ResourceDependency, ResourceTransition, ResourceUsage, SampleCount,
};
use nixe_gpu_headless::backend as headless_backend;
use nixe_memory::{
    CanonicalBackingPage, CanonicalBackingRange, CanonicalBackingSegment, CanonicalBackingStore,
    ContentGeneration, GuestPhysicalPageId, MappingGeneration, MemoryPermissions,
};

fn capabilities(features: BackendFeatures, formats: &[ImageFormat]) -> BackendCapabilities {
    BackendCapabilities::new(
        features,
        formats.iter().copied(),
        [SampleCount::One],
        [],
        [],
        BackendLimits {
            max_color_attachments: 1,
            max_descriptor_bindings: 8,
            max_compute_workgroups: [64; 3],
        },
    )
}

fn page() -> CanonicalBackingPage {
    let store = CanonicalBackingStore::allocate().unwrap();
    CanonicalBackingPage::zeroed(
        &store,
        GuestPhysicalPageId::new(1),
        0x1000,
        ContentGeneration::INITIAL,
    )
    .unwrap()
}

fn backing(
    allocation: GpuAllocationId,
    description: GpuAllocationDescription,
    page: &CanonicalBackingPage,
    page_offset: u64,
    size: u64,
) -> BackingView {
    let segment = CanonicalBackingSegment::new(
        page.clone(),
        page_offset,
        size,
        MemoryPermissions::READ_WRITE,
        page.content_generation(),
        MappingGeneration::INITIAL,
    )
    .unwrap();
    BackingView::new(
        allocation,
        description,
        0,
        CanonicalBackingRange::new(vec![segment]).unwrap(),
    )
    .unwrap()
}

fn allocation_info(
    id: GpuAllocationId,
    description: GpuAllocationDescription,
) -> BackendResourceCreateInfo {
    BackendResourceCreateInfo::Allocation { id, description }
}

fn buffer_info(id: BufferId, view: Option<BufferView>) -> BackendResourceCreateInfo {
    BackendResourceCreateInfo::Buffer {
        id,
        description: BufferDescription::new(64).unwrap(),
        view,
    }
}

fn clear_operation(buffer: BufferId, offset: u64, size: u64) -> GpuOperation {
    let clear = ClearOperation::buffer(
        nixe_gpu::BufferRegion {
            buffer,
            range: BufferRange::new(offset, size).unwrap(),
        },
        0,
    )
    .unwrap();
    GpuOperation::new(
        GpuCommand::Clear(clear),
        [],
        [],
        CapabilityRequirements::none(),
    )
}

fn submission(
    id: u64,
    predecessors: Vec<u64>,
    operations: Vec<GpuOperation>,
) -> OperationSubmission {
    OperationSubmission::new(
        nixe_gpu::FrontendSubmissionId::new(id),
        predecessors
            .into_iter()
            .map(nixe_gpu::FrontendSubmissionId::new)
            .collect(),
        operations,
    )
    .unwrap()
}

fn write_scope() -> AccessScope {
    AccessScope::new(
        PipelineStages::COPY,
        AccessMode::Write,
        ResourceUsage::TransferDestination,
    )
    .unwrap()
}

fn barrier_operation(buffer: BufferId, offset: u64, size: u64) -> GpuOperation {
    let transition = ResourceTransition::new(
        AccessTarget::Buffer {
            buffer,
            range: BufferRange::new(offset, size).unwrap(),
        },
        write_scope(),
        AccessScope::new(
            PipelineStages::COPY,
            AccessMode::Read,
            ResourceUsage::TransferSource,
        )
        .unwrap(),
    )
    .unwrap();
    GpuOperation::new(
        GpuCommand::Barrier(BarrierOperation::new(vec![transition]).unwrap()),
        [],
        [],
        CapabilityRequirements::none(),
    )
}

#[test]
fn canonical_alias_writes_require_a_barrier_and_rejection_is_atomic() {
    let features = BackendFeatures::CLEAR.union(BackendFeatures::BARRIER);
    let (mut backend, _) = headless_backend(BackendInstanceId::new(1), capabilities(features, &[]));
    let page = page();
    let allocation_description = GpuAllocationDescription::new(64, 16).unwrap();
    let allocation_a = GpuAllocationId::new(1);
    let allocation_b = GpuAllocationId::new(2);
    backend
        .create_resource(allocation_info(allocation_a, allocation_description))
        .unwrap();
    backend
        .create_resource(allocation_info(allocation_b, allocation_description))
        .unwrap();
    let buffer_a = BufferId::new(1);
    let buffer_b = BufferId::new(2);
    backend
        .create_resource(buffer_info(
            buffer_a,
            Some(
                BufferView::new(
                    buffer_a,
                    BufferDescription::new(64).unwrap(),
                    0,
                    backing(allocation_a, allocation_description, &page, 0, 64),
                )
                .unwrap(),
            ),
        ))
        .unwrap();
    backend
        .create_resource(buffer_info(
            buffer_b,
            Some(
                BufferView::new(
                    buffer_b,
                    BufferDescription::new(64).unwrap(),
                    0,
                    backing(allocation_b, allocation_description, &page, 0, 64),
                )
                .unwrap(),
            ),
        ))
        .unwrap();

    let malformed = submission(
        1,
        vec![],
        vec![
            clear_operation(buffer_a, 0, 16),
            clear_operation(buffer_b, 0, 16),
        ],
    );
    let error = backend.submit(&malformed).unwrap_err();
    assert!(matches!(error, BackendError::Driver(_)));
    assert!(error.to_string().contains("MissingBarrier"));

    // The first operation of the rejected submission was not committed.
    let token = backend
        .submit(&submission(
            2,
            vec![],
            vec![clear_operation(buffer_a, 0, 16)],
        ))
        .unwrap();
    assert_eq!(token.generation(), 1);
}

#[test]
fn predecessor_and_transition_are_both_required_for_conflicting_work() {
    let features = BackendFeatures::CLEAR.union(BackendFeatures::BARRIER);
    let (mut backend, completion) =
        headless_backend(BackendInstanceId::new(2), capabilities(features, &[]));
    let buffer = BufferId::new(1);
    backend.create_resource(buffer_info(buffer, None)).unwrap();
    let first = backend
        .submit(&submission(1, vec![], vec![clear_operation(buffer, 0, 16)]))
        .unwrap();

    let unordered = backend
        .submit(&submission(
            2,
            vec![],
            vec![barrier_operation(buffer, 0, 16)],
        ))
        .unwrap_err();
    assert!(unordered.to_string().contains("UnorderedAccess"));

    let missing_barrier = backend
        .submit(&submission(
            3,
            vec![1],
            vec![clear_operation(buffer, 0, 16)],
        ))
        .unwrap_err();
    assert!(missing_barrier.to_string().contains("MissingBarrier"));

    let second = backend
        .submit(&submission(
            4,
            vec![1],
            vec![barrier_operation(buffer, 0, 16)],
        ))
        .unwrap();
    assert_eq!(backend.has_completed(first), Ok(false));
    assert_eq!(backend.has_completed(second), Ok(false));
    completion.complete(second).unwrap();
    assert_eq!(backend.has_completed(first), Ok(false));
    assert_eq!(backend.has_completed(second), Ok(true));
}

#[test]
fn invalid_views_accesses_and_unsupported_formats_are_rejected_before_acceptance() {
    let (mut backend, _) = headless_backend(
        BackendInstanceId::new(3),
        capabilities(BackendFeatures::CLEAR, &[ImageFormat::Rgba8Unorm]),
    );
    let page = page();
    let allocation = GpuAllocationId::new(1);
    let allocation_description = GpuAllocationDescription::new(64, 16).unwrap();
    backend
        .create_resource(allocation_info(allocation, allocation_description))
        .unwrap();
    let wrong_id = BufferId::new(99);
    let invalid_view = BufferView::new(
        wrong_id,
        BufferDescription::new(64).unwrap(),
        0,
        backing(allocation, allocation_description, &page, 0, 32),
    )
    .unwrap();
    let before = backend.driver().resource_count();
    assert!(matches!(
        backend.create_resource(buffer_info(BufferId::new(1), Some(invalid_view))),
        Err(BackendError::InvalidResource(_))
    ));
    assert_eq!(backend.driver().resource_count(), before);

    let partial = BufferId::new(2);
    let partial_view = BufferView::new(
        partial,
        BufferDescription::new(64).unwrap(),
        16,
        backing(allocation, allocation_description, &page, 0, 32),
    )
    .unwrap();
    backend
        .create_resource(buffer_info(partial, Some(partial_view)))
        .unwrap();
    let error = backend
        .submit(&submission(
            1,
            vec![],
            vec![clear_operation(partial, 0, 16)],
        ))
        .unwrap_err();
    assert!(matches!(error, BackendError::AccessOutsideBacking(_)));

    let unsupported = BackendResourceCreateInfo::Image {
        id: ImageId::new(1),
        description: ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(4, 4, 1).unwrap(),
            ImageFormat::Rgba16Float,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .unwrap(),
        view: None,
    };
    assert!(matches!(
        backend.create_resource(unsupported),
        Err(BackendError::Capability(_))
    ));
}

#[test]
fn stale_use_after_destroy_device_loss_and_teardown_are_terminal() {
    let (mut backend, completion) = headless_backend(
        BackendInstanceId::new(4),
        capabilities(BackendFeatures::CLEAR, &[]),
    );
    let buffer = backend
        .create_resource(buffer_info(BufferId::new(1), None))
        .unwrap();
    backend.destroy_resource(buffer).unwrap();
    assert_eq!(
        backend.destroy_resource(buffer),
        Err(BackendError::StaleResource(buffer))
    );
    assert_eq!(
        backend.submit(&submission(
            1,
            vec![],
            vec![clear_operation(BufferId::new(1), 0, 16)]
        )),
        Err(BackendError::UnknownResource(ResourceDependency::Buffer(
            BufferId::new(1)
        )))
    );

    completion.lose_device("synthetic removal").unwrap();
    assert_eq!(
        backend.create_resource(buffer_info(BufferId::new(2), None)),
        Err(BackendError::DeviceLost("synthetic removal".into()))
    );
    assert_eq!(backend.state(), BackendState::DeviceLost);
    assert_eq!(backend.driver().resource_count(), 0);

    let (mut backend, completion) = headless_backend(
        BackendInstanceId::new(5),
        capabilities(BackendFeatures::CLEAR, &[]),
    );
    backend
        .create_resource(buffer_info(BufferId::new(1), None))
        .unwrap();
    let token = backend
        .submit(&submission(
            1,
            vec![],
            vec![clear_operation(BufferId::new(1), 0, 16)],
        ))
        .unwrap();
    completion.complete(token).unwrap();
    backend.release_submission(token).unwrap();
    assert!(completion.complete(token).is_err());
    backend.teardown().unwrap();
    backend.teardown().unwrap();
    assert_eq!(backend.state(), BackendState::TornDown);
    assert!(completion.lose_device("late").is_err());
}

#[test]
fn operation_ordering_rejection_does_not_leave_partial_render_pass_state() {
    let (mut backend, _) = headless_backend(
        BackendInstanceId::new(6),
        capabilities(BackendFeatures::RENDER_PASS, &[]),
    );
    let render_pass = RenderPassId::new(1);
    let description = RenderPassDescription::new(vec![]).unwrap();
    backend
        .create_resource(BackendResourceCreateInfo::RenderPass {
            id: render_pass,
            description: description.clone(),
        })
        .unwrap();

    let operation = |command| {
        GpuOperation::new(
            GpuCommand::RenderPass(command),
            [],
            [],
            CapabilityRequirements::none(),
        )
    };
    let mismatch = backend
        .submit(&submission(
            1,
            vec![],
            vec![operation(RenderPassOperation::end(render_pass))],
        ))
        .unwrap_err();
    assert!(mismatch.to_string().contains("RenderPassEndMismatch"));

    let begin = RenderPassOperation::begin(render_pass, description, vec![]).unwrap();
    backend
        .submit(&submission(
            2,
            vec![],
            vec![
                operation(begin),
                operation(RenderPassOperation::end(render_pass)),
            ],
        ))
        .unwrap();
}
