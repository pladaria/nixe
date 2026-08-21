//! Transactional lowering boundary for one completely decoded submission.

use std::fmt::{Display, Formatter};

use nixe_gpu::{
    BackendExecutionCompletion, BackendResourceCreateInfo, BackendRuntimeError,
    CacheMaintenanceOperation, CapabilityRequirements, CommandDescriptionError,
    FrontendSubmissionId, FrontendSubmissionSegment, GpuCommand, GpuOperation, GuestTimelinePoint,
    NeutralBackendRuntime, OperationSubmission, ReservedTimelinePoint, ResourceDependency,
};
use nixe_memory::{CanonicalWriteBatch, CanonicalWriteBatchError, MemoryPermissions};

use crate::engines::{
    lower_maxwell_three_d_operation_into_cache,
    resolve_maxwell_three_d_resources_for_roles_with_staged_writes_and_cache,
};
use crate::{
    MaxwellComputeSynchronizationPlan, MaxwellDmaCopyError, MaxwellDmaCopyOperation,
    MaxwellEngineOperation, MaxwellEnginePacketDispatch, MaxwellGpuAccessError,
    MaxwellGpuAddressSpace, MaxwellHostMemoryOperation, MaxwellMethodSource, MaxwellResolvedRange,
    MaxwellShaderTranslationError, MaxwellThreeDLoweredWork, MaxwellThreeDLoweringCache,
    MaxwellThreeDLoweringError, MaxwellThreeDResourceError, MaxwellThreeDSynchronizationError,
    MaxwellThreeDSynchronizationPlan, lower_maxwell_compute_synchronization,
    lower_maxwell_three_d_synchronization,
    shader::{MaxwellStagedShaderWrite, prepare_maxwell_shader_translation_inputs},
};

/// One ordered operation whose inputs have been resolved without side effects.
pub enum MaxwellSubmissionExecutionStep {
    InlineWrite {
        source: MaxwellMethodSource,
        target: MaxwellResolvedRange,
        value: [u8; 4],
    },
    DmaCopy {
        operation: MaxwellDmaCopyOperation,
        source: MaxwellResolvedRange,
        destination: MaxwellResolvedRange,
    },
    PostCompletionWrite {
        source: MaxwellMethodSource,
        target: MaxwellResolvedRange,
        value: [u8; 4],
    },
    BackendOperation(GpuOperation),
    ThreeD(MaxwellThreeDLoweredWork),
}

/// Complete neutral plan awaiting backend negotiation, execution, and completion.
///
/// Only the guest-visible completion point is copied. The unforgeable
/// reservation remains owned by the scheduled dispatch until backend work and
/// memory visibility have completed.
pub struct MaxwellSubmissionExecutionPlan {
    frontend: FrontendSubmissionId,
    predecessors: Box<[FrontendSubmissionId]>,
    steps: Box<[MaxwellSubmissionExecutionStep]>,
    staged_writes: CanonicalWriteBatch,
    write_source: Option<CanonicalWriteSource>,
    staged_cache: MaxwellThreeDLoweringCache,
    completion: Option<GuestTimelinePoint>,
}

impl MaxwellSubmissionExecutionPlan {
    #[must_use]
    pub const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }

    #[must_use]
    pub fn predecessors(&self) -> &[FrontendSubmissionId] {
        &self.predecessors
    }

    #[must_use]
    pub fn steps(&self) -> &[MaxwellSubmissionExecutionStep] {
        &self.steps
    }

    #[must_use]
    pub const fn staged_cache(&self) -> &MaxwellThreeDLoweringCache {
        &self.staged_cache
    }

    #[must_use]
    pub const fn completion(&self) -> Option<GuestTimelinePoint> {
        self.completion
    }

    /// Returns whether this plan contains work which must cross a neutral GPU
    /// backend instead of the canonical-memory initialization executor.
    #[must_use]
    pub fn requires_backend(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, MaxwellSubmissionExecutionStep::ThreeD(_)))
    }
}

/// Successful immediate software execution of an initialization submission.
pub struct MaxwellSoftwareInitializationCompletion {
    staged_cache: MaxwellThreeDLoweringCache,
    completion: Option<GuestTimelinePoint>,
}

/// Successful serialized execution of a complete Maxwell submission.
pub struct MaxwellBackendExecutionCompletion {
    staged_cache: MaxwellThreeDLoweringCache,
    completion: Option<GuestTimelinePoint>,
    backend: BackendExecutionCompletion,
}

impl MaxwellBackendExecutionCompletion {
    #[must_use]
    pub const fn completion(&self) -> Option<GuestTimelinePoint> {
        self.completion
    }

    #[must_use]
    pub const fn backend(&self) -> BackendExecutionCompletion {
        self.backend
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        MaxwellThreeDLoweringCache,
        Option<GuestTimelinePoint>,
        BackendExecutionCompletion,
    ) {
        (self.staged_cache, self.completion, self.backend)
    }
}

/// Failure before guest completion publication at the Maxwell/backend bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellBackendExecutionError {
    Software(Box<MaxwellSoftwareInitializationError>),
    InvalidSubmission(CommandDescriptionError),
    Backend(BackendRuntimeError),
}

impl Display for MaxwellBackendExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Software(error) => Display::fmt(error, formatter),
            Self::InvalidSubmission(error) => {
                write!(formatter, "neutral Maxwell submission is invalid: {error}")
            }
            Self::Backend(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for MaxwellBackendExecutionError {}

impl MaxwellSoftwareInitializationCompletion {
    #[must_use]
    pub const fn completion(&self) -> Option<GuestTimelinePoint> {
        self.completion
    }

    #[must_use]
    pub fn into_parts(self) -> (MaxwellThreeDLoweringCache, Option<GuestTimelinePoint>) {
        (self.staged_cache, self.completion)
    }
}

/// Failure before an initialization submission publishes bytes or completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellSoftwareInitializationError {
    UnsupportedThreeDWork,
    StaleInlineTarget {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    InlineWrite {
        source: MaxwellMethodSource,
        error: CanonicalWriteBatchError,
    },
    DmaAccess {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    DmaTransform {
        source: MaxwellMethodSource,
        error: MaxwellDmaCopyError,
    },
    DmaTransaction {
        source: MaxwellMethodSource,
        error: CanonicalWriteBatchError,
    },
}

impl Display for MaxwellSoftwareInitializationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedThreeDWork => formatter.write_str(
                "submission contains 3D work which requires shader translation and a neutral backend",
            ),
            Self::StaleInlineTarget { source, error } => {
                write!(formatter, "inline upload target changed before execution: {source}: {error}")
            }
            Self::InlineWrite { source, error } => {
                write!(formatter, "inline upload could not be staged atomically: {source}: {error}")
            }
            Self::DmaAccess { source, error } => {
                write!(formatter, "DMA copy mapping changed before execution: {source}: {error}")
            }
            Self::DmaTransform { source, error } => {
                write!(formatter, "DMA copy transformation failed: {source}: {error}")
            }
            Self::DmaTransaction { source, error } => {
                write!(formatter, "DMA copy transaction failed: {source}: {error}")
            }
        }
    }
}

impl std::error::Error for MaxwellSoftwareInitializationError {}

/// Failure before any guest write, backend submission, or fence publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellSubmissionExecutionError {
    InlineAddress {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    DmaAddress {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    ThreeDResource(MaxwellThreeDResourceError),
    StagedMemory(Box<MaxwellSoftwareInitializationError>),
    ThreeDLowering(MaxwellThreeDLoweringError),
    ShaderTranslation(MaxwellShaderTranslationError),
    ThreeDSynchronization(MaxwellThreeDSynchronizationError),
    MissingCompletionSignal {
        reserved: GuestTimelinePoint,
        expected: u32,
        observed: u32,
    },
    DuplicateCompletionSignal {
        reserved: GuestTimelinePoint,
        expected: u32,
        observed: u32,
    },
}

impl Display for MaxwellSubmissionExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InlineAddress { source, error } => {
                write!(
                    formatter,
                    "inline upload target is invalid: {source}: {error}"
                )
            }
            Self::DmaAddress { source, error } => {
                write!(formatter, "DMA copy range is invalid: {source}: {error}")
            }
            Self::ThreeDResource(error) => Display::fmt(error, formatter),
            Self::StagedMemory(error) => {
                write!(
                    formatter,
                    "ordered submission memory preflight failed: {error}"
                )
            }
            Self::ThreeDLowering(error) => Display::fmt(error, formatter),
            Self::ShaderTranslation(error) => Display::fmt(error, formatter),
            Self::ThreeDSynchronization(error) => Display::fmt(error, formatter),
            Self::MissingCompletionSignal {
                reserved,
                expected,
                observed,
            } => write!(
                formatter,
                "submission emitted too few syncpoint increments for reserved completion {reserved}: expected={expected} observed={observed}"
            ),
            Self::DuplicateCompletionSignal {
                reserved,
                expected,
                observed,
            } => write!(
                formatter,
                "submission emitted too many syncpoint increments for reserved completion {reserved}: expected={expected} observed={observed}"
            ),
        }
    }
}

impl std::error::Error for MaxwellSubmissionExecutionError {}

/// Preflights a decoded submission in original packet and method-effect order.
///
/// This function clones the frontend cache and returns the candidate. It never
/// publishes an inline payload, invokes a backend, changes scheduler state,
/// or consumes/publishes `completion`. Earlier writes are retained only in an
/// atomically discardable transaction so later operations can read them.
pub fn preflight_maxwell_submission_execution(
    packets: &[MaxwellEnginePacketDispatch],
    address_space: &MaxwellGpuAddressSpace,
    frontend: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    completion: Option<&ReservedTimelinePoint>,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<MaxwellSubmissionExecutionPlan, MaxwellSubmissionExecutionError> {
    let mut staged_cache = cache.clone();
    let mut steps = Vec::new();
    let mut prior_work_pending = false;
    let mut completion_signal_count = 0_u32;
    let mut staged_shader_writes = Vec::new();
    let mut staged_memory_writes = CanonicalWriteBatch::new();
    let mut write_source = None;

    for operation in packets
        .iter()
        .flat_map(MaxwellEnginePacketDispatch::ordered_operations)
    {
        match operation {
            MaxwellEngineOperation::HostSynchronization(operation) => {
                let command = match operation.operation() {
                    MaxwellHostMemoryOperation::L2SysmemInvalidate { .. } => {
                        GpuCommand::CacheMaintenance(
                            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
                        )
                    }
                    MaxwellHostMemoryOperation::L2FlushDirty { .. } => {
                        GpuCommand::CacheMaintenance(
                            CacheMaintenanceOperation::FlushDirtyDeviceWrites,
                        )
                    }
                };
                steps.push(MaxwellSubmissionExecutionStep::BackendOperation(
                    GpuOperation::new(command, [], [], CapabilityRequirements::none()),
                ));
            }
            MaxwellEngineOperation::ComputeInlineToMemory(upload) => {
                let target = resolve_inline_target(
                    address_space,
                    upload.address().get(),
                    upload.offset(),
                    upload.source(),
                )?;
                staged_shader_writes.push(MaxwellStagedShaderWrite::new(
                    target.offset().get(),
                    upload.value(),
                ));
                stage_inline_write(
                    address_space,
                    &target,
                    upload.value().to_le_bytes(),
                    upload.source(),
                    &mut staged_memory_writes,
                )
                .map_err(|error| MaxwellSubmissionExecutionError::StagedMemory(Box::new(error)))?;
                write_source = Some(CanonicalWriteSource::Inline(upload.source()));
                steps.push(MaxwellSubmissionExecutionStep::InlineWrite {
                    source: upload.source(),
                    target,
                    value: upload.value().to_le_bytes(),
                });
                prior_work_pending = true;
            }
            MaxwellEngineOperation::InlineToMemory(upload) => {
                let target = resolve_inline_target(
                    address_space,
                    upload.address().get(),
                    upload.offset(),
                    upload.source(),
                )?;
                staged_shader_writes.push(MaxwellStagedShaderWrite::new(
                    target.offset().get(),
                    upload.value(),
                ));
                stage_inline_write(
                    address_space,
                    &target,
                    upload.value().to_le_bytes(),
                    upload.source(),
                    &mut staged_memory_writes,
                )
                .map_err(|error| MaxwellSubmissionExecutionError::StagedMemory(Box::new(error)))?;
                write_source = Some(CanonicalWriteSource::Inline(upload.source()));
                steps.push(MaxwellSubmissionExecutionStep::InlineWrite {
                    source: upload.source(),
                    target,
                    value: upload.value().to_le_bytes(),
                });
                prior_work_pending = true;
            }
            MaxwellEngineOperation::DmaCopy(operation) => {
                let source_address = address_space
                    .address(operation.source_address())
                    .map_err(MaxwellGpuAccessError::Address)
                    .map_err(|error| MaxwellSubmissionExecutionError::DmaAddress {
                        source: operation.source(),
                        error,
                    })?;
                let destination_address = address_space
                    .address(operation.destination_address())
                    .map_err(MaxwellGpuAccessError::Address)
                    .map_err(|error| MaxwellSubmissionExecutionError::DmaAddress {
                        source: operation.source(),
                        error,
                    })?;
                let source = address_space
                    .resolve_range(
                        source_address,
                        operation.source_range_size(),
                        MemoryPermissions::READ,
                    )
                    .map_err(|error| MaxwellSubmissionExecutionError::DmaAddress {
                        source: operation.source(),
                        error,
                    })?;
                let destination = address_space
                    .resolve_range(
                        destination_address,
                        operation.destination_range_size(),
                        MemoryPermissions::READ_WRITE,
                    )
                    .map_err(|error| MaxwellSubmissionExecutionError::DmaAddress {
                        source: operation.source(),
                        error,
                    })?;
                stage_dma_copy(
                    address_space,
                    *operation,
                    &source,
                    &destination,
                    &mut staged_memory_writes,
                )
                .map_err(|error| MaxwellSubmissionExecutionError::StagedMemory(Box::new(error)))?;
                write_source = Some(CanonicalWriteSource::Dma(operation.source()));
                steps.push(MaxwellSubmissionExecutionStep::DmaCopy {
                    operation: *operation,
                    source,
                    destination,
                });
                prior_work_pending = true;
            }
            MaxwellEngineOperation::ComputeSynchronization(operation) => {
                let plan = lower_maxwell_compute_synchronization(operation, prior_work_pending);
                match plan {
                    MaxwellComputeSynchronizationPlan::WaitForIdle { .. } => {
                        prior_work_pending = false;
                    }
                    MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches } => {
                        steps.push(MaxwellSubmissionExecutionStep::BackendOperation(
                            cache_maintenance_operation(
                                CacheMaintenanceOperation::InvalidateShaderCaches {
                                    instruction: caches.instruction(),
                                    global_data: caches.global_data(),
                                    constant: caches.constant(),
                                },
                            ),
                        ));
                    }
                }
            }
            MaxwellEngineOperation::ThreeDInlineConstantBuffer(upload) => {
                let target = resolve_inline_target(
                    address_space,
                    upload.address().get(),
                    upload.offset(),
                    upload.source(),
                )?;
                staged_shader_writes.push(MaxwellStagedShaderWrite::new(
                    target.offset().get(),
                    upload.value(),
                ));
                stage_inline_write(
                    address_space,
                    &target,
                    upload.value().to_le_bytes(),
                    upload.source(),
                    &mut staged_memory_writes,
                )
                .map_err(|error| MaxwellSubmissionExecutionError::StagedMemory(Box::new(error)))?;
                write_source = Some(CanonicalWriteSource::Inline(upload.source()));
                steps.push(MaxwellSubmissionExecutionStep::InlineWrite {
                    source: upload.source(),
                    target,
                    value: upload.value().to_le_bytes(),
                });
                prior_work_pending = true;
            }
            MaxwellEngineOperation::ThreeD(operation) => {
                if matches!(
                    operation.trigger(),
                    crate::MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
                ) {
                    let inputs = prepare_maxwell_shader_translation_inputs(
                        operation.state(),
                        address_space,
                        &staged_shader_writes,
                    )
                    .map_err(MaxwellSubmissionExecutionError::ShaderTranslation)?;
                    let programs = staged_cache
                        .resolve_shader_translation_inputs(inputs)
                        .map_err(MaxwellSubmissionExecutionError::ShaderTranslation)?;
                    let translated = staged_cache
                        .stage_shader_translations(&programs)
                        .map_err(MaxwellSubmissionExecutionError::ThreeDLowering)?;
                    let mut required_roles = translated
                        .resources()
                        .iter()
                        .map(|resource| resource.role())
                        .collect::<Vec<_>>();
                    operation
                        .trigger()
                        .append_resource_roles(operation.state(), &mut required_roles);
                    let resources =
                        resolve_maxwell_three_d_resources_for_roles_with_staged_writes_and_cache(
                            operation.state(),
                            address_space,
                            &required_roles,
                            Some(&staged_memory_writes),
                            false,
                            Some(staged_cache.retained_backings_mut()),
                        )
                        .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                    let work = lower_maxwell_three_d_operation_into_cache(
                        operation.state(),
                        &resources,
                        operation.trigger(),
                        Some(&translated),
                        frontend,
                        predecessors.clone(),
                        &mut staged_cache,
                    )
                    .map_err(MaxwellSubmissionExecutionError::ThreeDLowering)?;
                    steps.push(MaxwellSubmissionExecutionStep::ThreeD(work));
                    prior_work_pending = true;
                    continue;
                }
                let mut required_roles = Vec::new();
                operation
                    .trigger()
                    .append_resource_roles(operation.state(), &mut required_roles);
                let resources =
                    resolve_maxwell_three_d_resources_for_roles_with_staged_writes_and_cache(
                        operation.state(),
                        address_space,
                        &required_roles,
                        Some(&staged_memory_writes),
                        false,
                        Some(staged_cache.retained_backings_mut()),
                    )
                    .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                let work = lower_maxwell_three_d_operation_into_cache(
                    operation.state(),
                    &resources,
                    operation.trigger(),
                    None,
                    frontend,
                    predecessors.clone(),
                    &mut staged_cache,
                )
                .map_err(MaxwellSubmissionExecutionError::ThreeDLowering)?;
                steps.push(MaxwellSubmissionExecutionStep::ThreeD(work));
                prior_work_pending = true;
            }
            MaxwellEngineOperation::ThreeDSynchronization(operation) => {
                let plan = lower_maxwell_three_d_synchronization(
                    operation,
                    completion,
                    prior_work_pending,
                )
                .map_err(MaxwellSubmissionExecutionError::ThreeDSynchronization)?;
                if let MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
                    completion: reserved,
                    ..
                } = plan
                {
                    completion_signal_count = completion_signal_count.saturating_add(1);
                    let expected = completion.map_or(0, ReservedTimelinePoint::increments);
                    if completion_signal_count > expected {
                        return Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal {
                            reserved,
                            expected,
                            observed: completion_signal_count,
                        });
                    }
                }
                let drains_prior_work = matches!(
                    plan,
                    MaxwellThreeDSynchronizationPlan::DecompressUncompressedSurface { .. }
                        | MaxwellThreeDSynchronizationPlan::WaitForIdle { .. }
                        | MaxwellThreeDSynchronizationPlan::InvalidateShaderCaches { .. }
                        | MaxwellThreeDSynchronizationPlan::FlushPendingWrites { .. }
                        | MaxwellThreeDSynchronizationPlan::InvalidateTextureCache { .. }
                        | MaxwellThreeDSynchronizationPlan::ReportSemaphoreRelease(_)
                        | MaxwellThreeDSynchronizationPlan::IncrementSyncpoint { .. }
                );
                if let MaxwellThreeDSynchronizationPlan::ReportSemaphoreRelease(release) = plan {
                    let target = resolve_inline_target(
                        address_space,
                        release.address().get(),
                        0,
                        release.source(),
                    )?;
                    steps.push(MaxwellSubmissionExecutionStep::PostCompletionWrite {
                        source: release.source(),
                        target,
                        value: release.payload().to_le_bytes(),
                    });
                } else if let Some(operation) = three_d_synchronization_operation(plan) {
                    steps.push(MaxwellSubmissionExecutionStep::BackendOperation(operation));
                }
                if drains_prior_work {
                    prior_work_pending = false;
                }
            }
        }
    }

    if let Some(completion) = completion {
        let expected = completion.increments();
        if completion_signal_count < expected {
            return Err(MaxwellSubmissionExecutionError::MissingCompletionSignal {
                reserved: completion.point(),
                expected,
                observed: completion_signal_count,
            });
        }
    }

    Ok(MaxwellSubmissionExecutionPlan {
        frontend,
        predecessors: predecessors.into_boxed_slice(),
        steps: steps.into_boxed_slice(),
        staged_writes: staged_memory_writes,
        write_source,
        staged_cache,
        completion: completion.map(ReservedTimelinePoint::point),
    })
}

/// Executes a completely preflighted write-only initialization submission.
///
/// Inline payloads are staged into one canonical-memory transaction. Ordering
/// methods are validated before the transaction commits, and this function
/// neither consumes nor publishes a guest timeline reservation. Clear, draw,
/// or any other 3D backend work remains a typed later boundary.
pub fn execute_maxwell_software_initialization(
    plan: MaxwellSubmissionExecutionPlan,
    address_space: &MaxwellGpuAddressSpace,
) -> Result<MaxwellSoftwareInitializationCompletion, MaxwellSoftwareInitializationError> {
    if plan.requires_backend() {
        return Err(MaxwellSoftwareInitializationError::UnsupportedThreeDWork);
    }
    let mut writes = plan.staged_writes;
    let mut write_source = plan.write_source;
    for step in &plan.steps {
        if let MaxwellSubmissionExecutionStep::PostCompletionWrite {
            source,
            target,
            value,
        } = step
        {
            stage_inline_write(address_space, target, *value, *source, &mut writes)?;
            write_source = Some(CanonicalWriteSource::Inline(*source));
        }
    }
    commit_write_batch(writes, write_source)?;

    Ok(MaxwellSoftwareInitializationCompletion {
        staged_cache: plan.staged_cache,
        completion: plan.completion,
    })
}

/// Executes one preflighted submission through the selected neutral backend.
///
/// Command-processor writes are committed at their command-order boundaries.
/// Reusing one canonical buffer across draws therefore preserves the contents
/// visible to each draw instead of exposing only the final update. Report-
/// semaphore writes are committed only after host completion and GPU write
/// visibility. The caller still owns guest timeline publication and may
/// advance it only after this function succeeds.
pub fn execute_maxwell_backend_submission(
    plan: MaxwellSubmissionExecutionPlan,
    address_space: &MaxwellGpuAddressSpace,
    backend: &mut dyn NeutralBackendRuntime,
) -> Result<MaxwellBackendExecutionCompletion, MaxwellBackendExecutionError> {
    let completion = execute_maxwell_backend_steps(
        &plan,
        address_space,
        |creations, invalidations, submission| {
            backend
                .execute(creations, invalidations, submission)
                .map_err(MaxwellBackendExecutionError::Backend)
        },
    )?
    .ok_or(MaxwellBackendExecutionError::InvalidSubmission(
        CommandDescriptionError::EmptySubmission,
    ))?;
    Ok(MaxwellBackendExecutionCompletion {
        staged_cache: plan.staged_cache,
        completion: plan.completion,
        backend: completion,
    })
}

fn execute_maxwell_backend_steps<T>(
    plan: &MaxwellSubmissionExecutionPlan,
    address_space: &MaxwellGpuAddressSpace,
    mut execute_segment: impl FnMut(
        &[BackendResourceCreateInfo],
        &[ResourceDependency],
        &OperationSubmission,
    ) -> Result<T, MaxwellBackendExecutionError>,
) -> Result<Option<T>, MaxwellBackendExecutionError> {
    let mut pre_writes = CanonicalWriteBatch::new();
    let mut post_writes = CanonicalWriteBatch::new();
    let mut pre_write_source = None;
    let mut post_write_source = None;
    let mut creations = Vec::<BackendResourceCreateInfo>::new();
    let mut invalidations = Vec::<ResourceDependency>::new();
    let mut operations = Vec::<GpuOperation>::new();
    let mut last_completion = None;
    let mut segments = BackendSegmentCursor::for_plan(plan)?;

    for step in &plan.steps {
        match step {
            MaxwellSubmissionExecutionStep::BackendOperation(operation) => {
                commit_pending_backend_writes(&mut pre_writes, &mut pre_write_source)?;
                operations.push(operation.clone());
            }
            MaxwellSubmissionExecutionStep::InlineWrite {
                source,
                target,
                value,
            } => {
                flush_maxwell_backend_segment(
                    plan,
                    &mut creations,
                    &mut invalidations,
                    &mut operations,
                    &mut segments,
                    &mut execute_segment,
                    &mut last_completion,
                )?;
                stage_inline_write(address_space, target, *value, *source, &mut pre_writes)
                    .map_err(|error| MaxwellBackendExecutionError::Software(Box::new(error)))?;
                pre_write_source = Some(CanonicalWriteSource::Inline(*source));
            }
            MaxwellSubmissionExecutionStep::DmaCopy {
                operation,
                source,
                destination,
            } => {
                flush_maxwell_backend_segment(
                    plan,
                    &mut creations,
                    &mut invalidations,
                    &mut operations,
                    &mut segments,
                    &mut execute_segment,
                    &mut last_completion,
                )?;
                stage_dma_copy(
                    address_space,
                    *operation,
                    source,
                    destination,
                    &mut pre_writes,
                )
                .map_err(|error| MaxwellBackendExecutionError::Software(Box::new(error)))?;
                pre_write_source = Some(CanonicalWriteSource::Dma(operation.source()));
            }
            MaxwellSubmissionExecutionStep::PostCompletionWrite {
                source,
                target,
                value,
            } => {
                stage_inline_write(address_space, target, *value, *source, &mut post_writes)
                    .map_err(|error| MaxwellBackendExecutionError::Software(Box::new(error)))?;
                post_write_source = Some(*source);
            }
            MaxwellSubmissionExecutionStep::ThreeD(work) => {
                commit_pending_backend_writes(&mut pre_writes, &mut pre_write_source)?;
                creations.extend_from_slice(work.resource_creations());
                invalidations.extend_from_slice(work.resource_invalidations());
                operations.extend_from_slice(work.submission().operations());
            }
        }
    }

    commit_pending_backend_writes(&mut pre_writes, &mut pre_write_source)?;
    flush_maxwell_backend_segment(
        plan,
        &mut creations,
        &mut invalidations,
        &mut operations,
        &mut segments,
        &mut execute_segment,
        &mut last_completion,
    )?;
    commit_inline_batch(post_writes, post_write_source)
        .map_err(|error| MaxwellBackendExecutionError::Software(Box::new(error)))?;

    Ok(last_completion)
}

fn commit_pending_backend_writes(
    writes: &mut CanonicalWriteBatch,
    source: &mut Option<CanonicalWriteSource>,
) -> Result<(), MaxwellBackendExecutionError> {
    if writes.is_empty() {
        return Ok(());
    }
    commit_write_batch(std::mem::take(writes), source.take())
        .map_err(|error| MaxwellBackendExecutionError::Software(Box::new(error)))
}

struct BackendSegmentCursor {
    count: u32,
    next: FrontendSubmissionSegment,
}

impl BackendSegmentCursor {
    fn for_plan(
        plan: &MaxwellSubmissionExecutionPlan,
    ) -> Result<Self, MaxwellBackendExecutionError> {
        let mut count = 0_usize;
        let mut operations_pending = false;
        for step in &plan.steps {
            if backend_step_is_canonical_write(step) && operations_pending {
                count = count.checked_add(1).ok_or_else(too_many_backend_segments)?;
                operations_pending = false;
            }
            operations_pending |= backend_step_emits_operation(step);
        }
        if operations_pending {
            count = count.checked_add(1).ok_or_else(too_many_backend_segments)?;
        }
        let count = u32::try_from(count).map_err(|_| too_many_backend_segments())?;
        Ok(Self {
            count,
            next: FrontendSubmissionSegment::FIRST,
        })
    }

    fn take(&mut self) -> Result<(FrontendSubmissionSegment, bool), MaxwellBackendExecutionError> {
        let segment = self.next;
        let completed = segment
            .get()
            .checked_add(1)
            .ok_or_else(too_many_backend_segments)?;
        if completed > self.count {
            return Err(too_many_backend_segments());
        }
        let final_segment = completed == self.count;
        if !final_segment {
            self.next = segment
                .checked_next()
                .ok_or_else(too_many_backend_segments)?;
        }
        Ok((segment, final_segment))
    }
}

fn backend_step_is_canonical_write(step: &MaxwellSubmissionExecutionStep) -> bool {
    matches!(
        step,
        MaxwellSubmissionExecutionStep::InlineWrite { .. }
            | MaxwellSubmissionExecutionStep::DmaCopy { .. }
    )
}

fn backend_step_emits_operation(step: &MaxwellSubmissionExecutionStep) -> bool {
    matches!(
        step,
        MaxwellSubmissionExecutionStep::BackendOperation(_)
            | MaxwellSubmissionExecutionStep::ThreeD(_)
    )
}

fn too_many_backend_segments() -> MaxwellBackendExecutionError {
    MaxwellBackendExecutionError::InvalidSubmission(
        CommandDescriptionError::TooManySubmissionSegments,
    )
}

fn flush_maxwell_backend_segment<T>(
    plan: &MaxwellSubmissionExecutionPlan,
    creations: &mut Vec<BackendResourceCreateInfo>,
    invalidations: &mut Vec<ResourceDependency>,
    operations: &mut Vec<GpuOperation>,
    segments: &mut BackendSegmentCursor,
    execute_segment: &mut impl FnMut(
        &[BackendResourceCreateInfo],
        &[ResourceDependency],
        &OperationSubmission,
    ) -> Result<T, MaxwellBackendExecutionError>,
    last_completion: &mut Option<T>,
) -> Result<(), MaxwellBackendExecutionError> {
    if operations.is_empty() {
        return Ok(());
    }
    let (segment, final_segment) = segments.take()?;
    let submission = OperationSubmission::new_segment(
        plan.frontend,
        segment,
        final_segment,
        plan.predecessors.to_vec(),
        std::mem::take(operations),
    )
    .map_err(MaxwellBackendExecutionError::InvalidSubmission)?;
    *last_completion = Some(execute_segment(creations, invalidations, &submission)?);
    creations.clear();
    invalidations.clear();
    Ok(())
}

fn cache_maintenance_operation(maintenance: CacheMaintenanceOperation) -> GpuOperation {
    GpuOperation::new(
        GpuCommand::CacheMaintenance(maintenance),
        [],
        [],
        CapabilityRequirements::none(),
    )
}

fn three_d_synchronization_operation(
    plan: MaxwellThreeDSynchronizationPlan,
) -> Option<GpuOperation> {
    let maintenance = match plan {
        MaxwellThreeDSynchronizationPlan::InvalidateShaderCaches { maintenance, .. }
        | MaxwellThreeDSynchronizationPlan::InvalidateShaderCachesNoWfi { maintenance, .. }
        | MaxwellThreeDSynchronizationPlan::InvalidateTextureCacheNoWfi { maintenance, .. }
        | MaxwellThreeDSynchronizationPlan::InvalidateTextureCache { maintenance, .. }
        | MaxwellThreeDSynchronizationPlan::TiledCacheFlush { maintenance, .. } => maintenance,
        MaxwellThreeDSynchronizationPlan::FlushPendingWrites { .. } => {
            CacheMaintenanceOperation::FlushDirtyDeviceWrites
        }
        MaxwellThreeDSynchronizationPlan::DecompressUncompressedSurface { .. }
        | MaxwellThreeDSynchronizationPlan::WaitForIdle { .. }
        | MaxwellThreeDSynchronizationPlan::IncrementSyncpoint { .. }
        | MaxwellThreeDSynchronizationPlan::ReportSemaphoreRelease(_) => return None,
    };
    Some(cache_maintenance_operation(maintenance))
}

#[derive(Clone, Copy)]
enum CanonicalWriteSource {
    Inline(MaxwellMethodSource),
    Dma(MaxwellMethodSource),
}

fn commit_write_batch(
    writes: CanonicalWriteBatch,
    source: Option<CanonicalWriteSource>,
) -> Result<(), MaxwellSoftwareInitializationError> {
    writes.commit().map_err(|error| {
        match source.expect("a non-empty canonical write batch has an ordered method source") {
            CanonicalWriteSource::Inline(source) => {
                MaxwellSoftwareInitializationError::InlineWrite { source, error }
            }
            CanonicalWriteSource::Dma(source) => {
                MaxwellSoftwareInitializationError::DmaTransaction { source, error }
            }
        }
    })
}

fn stage_dma_copy(
    address_space: &MaxwellGpuAddressSpace,
    operation: MaxwellDmaCopyOperation,
    source: &MaxwellResolvedRange,
    destination: &MaxwellResolvedRange,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellSoftwareInitializationError> {
    let mut source_bytes = dma_buffer(operation.source_range_size(), operation.source())?;
    read_dma_bytes(
        address_space,
        source,
        &mut source_bytes,
        writes,
        operation.source(),
    )?;
    let mut destination_bytes = dma_buffer(operation.destination_range_size(), operation.source())?;
    read_dma_bytes(
        address_space,
        destination,
        &mut destination_bytes,
        writes,
        operation.source(),
    )?;
    operation
        .copy_bytes(&source_bytes, &mut destination_bytes)
        .map_err(|error| MaxwellSoftwareInitializationError::DmaTransform {
            source: operation.source(),
            error,
        })?;

    let mut copied = 0_usize;
    for segment in destination.segments() {
        let size = usize::try_from(segment.size()).map_err(|_| {
            MaxwellSoftwareInitializationError::DmaTransform {
                source: operation.source(),
                error: MaxwellDmaCopyError::ArithmeticOverflow,
            }
        })?;
        let end =
            copied
                .checked_add(size)
                .ok_or(MaxwellSoftwareInitializationError::DmaTransform {
                    source: operation.source(),
                    error: MaxwellDmaCopyError::ArithmeticOverflow,
                })?;
        writes
            .stage(
                segment.mapping().backing(),
                segment.backing_offset(),
                &destination_bytes[copied..end],
            )
            .map_err(|error| MaxwellSoftwareInitializationError::DmaTransaction {
                source: operation.source(),
                error,
            })?;
        copied = end;
    }
    Ok(())
}

fn read_dma_bytes(
    address_space: &MaxwellGpuAddressSpace,
    range: &MaxwellResolvedRange,
    output: &mut [u8],
    writes: &CanonicalWriteBatch,
    source: MaxwellMethodSource,
) -> Result<(), MaxwellSoftwareInitializationError> {
    address_space
        .read_resolved(range, output)
        .map_err(|error| MaxwellSoftwareInitializationError::DmaAccess { source, error })?;
    let mut copied = 0_usize;
    for segment in range.segments() {
        let size = usize::try_from(segment.size()).map_err(|_| {
            MaxwellSoftwareInitializationError::DmaTransform {
                source,
                error: MaxwellDmaCopyError::ArithmeticOverflow,
            }
        })?;
        let end =
            copied
                .checked_add(size)
                .ok_or(MaxwellSoftwareInitializationError::DmaTransform {
                    source,
                    error: MaxwellDmaCopyError::ArithmeticOverflow,
                })?;
        writes
            .read_staged(
                segment.mapping().backing(),
                segment.backing_offset(),
                &mut output[copied..end],
            )
            .map_err(|error| MaxwellSoftwareInitializationError::DmaTransaction {
                source,
                error,
            })?;
        copied = end;
    }
    Ok(())
}

fn dma_buffer(
    size: u64,
    source: MaxwellMethodSource,
) -> Result<Vec<u8>, MaxwellSoftwareInitializationError> {
    let size =
        usize::try_from(size).map_err(|_| MaxwellSoftwareInitializationError::DmaTransform {
            source,
            error: MaxwellDmaCopyError::ArithmeticOverflow,
        })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|_| {
        MaxwellSoftwareInitializationError::DmaTransform {
            source,
            error: MaxwellDmaCopyError::ResourceExhausted,
        }
    })?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn commit_inline_batch(
    writes: CanonicalWriteBatch,
    source: Option<MaxwellMethodSource>,
) -> Result<(), MaxwellSoftwareInitializationError> {
    writes
        .commit()
        .map_err(|error| MaxwellSoftwareInitializationError::InlineWrite {
            source: source.expect("a non-empty canonical write batch has an inline source"),
            error,
        })
}

fn stage_inline_write(
    address_space: &MaxwellGpuAddressSpace,
    target: &MaxwellResolvedRange,
    bytes: [u8; 4],
    source: MaxwellMethodSource,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellSoftwareInitializationError> {
    if target.address_space() != address_space.id() {
        return Err(MaxwellSoftwareInitializationError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::WrongAddressSpace {
                expected: target.address_space(),
                actual: address_space.id(),
            },
        });
    }
    if !target.permissions().contains(MemoryPermissions::WRITE) {
        return Err(MaxwellSoftwareInitializationError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::PermissionDenied {
                address: target.offset(),
                required: MemoryPermissions::WRITE,
                available: target.permissions(),
            },
        });
    }
    if target.size() != bytes.len() as u64 {
        return Err(MaxwellSoftwareInitializationError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::OutputSizeMismatch {
                expected: target.size(),
                actual: bytes.len() as u64,
            },
        });
    }
    for segment in target.segments() {
        if !address_space.retained_mapping_is_current(segment.mapping()) {
            return Err(MaxwellSoftwareInitializationError::StaleInlineTarget {
                source,
                error: MaxwellGpuAccessError::StaleMapping {
                    mapping: segment.mapping().id(),
                    generation: segment.mapping().generation(),
                },
            });
        }
    }

    let mut copied = 0_usize;
    for segment in target.segments() {
        let size = usize::try_from(segment.size()).map_err(|_| {
            MaxwellSoftwareInitializationError::StaleInlineTarget {
                source,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            }
        })?;
        let end = copied.checked_add(size).ok_or(
            MaxwellSoftwareInitializationError::StaleInlineTarget {
                source,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            },
        )?;
        writes
            .stage(
                segment.mapping().backing(),
                segment.backing_offset(),
                &bytes[copied..end],
            )
            .map_err(|error| MaxwellSoftwareInitializationError::InlineWrite { source, error })?;
        copied = end;
    }
    Ok(())
}

fn resolve_inline_target(
    address_space: &MaxwellGpuAddressSpace,
    base: u64,
    offset: u32,
    source: MaxwellMethodSource,
) -> Result<MaxwellResolvedRange, MaxwellSubmissionExecutionError> {
    let base = address_space
        .address(base)
        .map_err(MaxwellGpuAccessError::Address)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })?;
    let target = address_space
        .checked_add(base, u64::from(offset))
        .map_err(MaxwellGpuAccessError::Address)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })?;
    address_space
        .resolve_range(target, size_of::<u32>() as u64, MemoryPermissions::WRITE)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })
}

#[cfg(test)]
mod tests {
    use nixe_gpu::{
        FrontendSubmissionId, GpuVirtualAddress, GuestSyncpointId, GuestSyncpointValue,
        GuestTimeline, MappingGeneration, TimelineInstanceId, TimelineOwnerId,
    };
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellChannelId, MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellGpuChannel,
        MaxwellMapRequest, MaxwellMappingId, MaxwellPushbufferWord, SWITCH_1_GM20B_PROFILE,
        decode_maxwell_pushbuffer, dispatch_maxwell_engine_packet,
    };

    fn packet(
        subchannel: u32,
        method_dword: u32,
        arguments: &[u32],
    ) -> crate::MaxwellDecodedPushbuffer {
        let mut words = Vec::with_capacity(arguments.len() + 1);
        words.push(Ok(MaxwellPushbufferWord::new(
            (1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
            location(0),
        )));
        words.extend(arguments.iter().enumerate().map(|(index, argument)| {
            Ok(MaxwellPushbufferWord::new(
                *argument,
                location(index as u32 + 1),
            ))
        }));
        decode_maxwell_pushbuffer(words).unwrap()
    }

    fn non_incrementing_packet(
        subchannel: u32,
        method_dword: u32,
        arguments: &[u32],
    ) -> crate::MaxwellDecodedPushbuffer {
        let mut words = Vec::with_capacity(arguments.len() + 1);
        words.push(Ok(MaxwellPushbufferWord::new(
            (3 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
            location(0),
        )));
        words.extend(arguments.iter().enumerate().map(|(index, argument)| {
            Ok(MaxwellPushbufferWord::new(
                *argument,
                location(index as u32 + 1),
            ))
        }));
        decode_maxwell_pushbuffer(words).unwrap()
    }

    fn location(word_offset: u32) -> MaxwellGpfifoSourceLocation {
        MaxwellGpfifoSourceLocation {
            channel: MaxwellChannelId::new(1),
            frontend: FrontendSubmissionId::new(2),
            entry_index: 0,
            pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
            word_offset: u64::from(word_offset),
            mapping: MaxwellMappingId::new(1),
            generation: MappingGeneration::new(1),
        }
    }

    fn address_space() -> MaxwellGpuAddressSpace {
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        address_space
    }

    fn reservation() -> (GuestTimeline, ReservedTimelinePoint) {
        let owner = TimelineOwnerId::new(7);
        let mut timeline = GuestTimeline::new(
            GuestSyncpointId::new(1),
            TimelineInstanceId::new(1),
            owner,
            GuestSyncpointValue::new(0),
        );
        let reservation = timeline.reserve(owner, 1).unwrap();
        (timeline, reservation)
    }

    #[test]
    fn empty_preflight_is_neutral_without_a_completion() {
        let plan = preflight_maxwell_submission_execution(
            &[],
            &address_space(),
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert!(plan.steps().is_empty());
        assert_eq!(plan.completion(), None);
        assert_eq!(plan.staged_cache().revision(), 0);
    }

    #[test]
    fn captured_three_d_report_semaphore_release_writes_payload_after_prior_work() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mut address_space = address_space();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        let release = packet(
            0,
            0x1b00 / 4,
            &[
                (address >> 32) as u32,
                address as u32,
                0xcafe_babe,
                0x1000_f010,
            ],
        );
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &release.packets()[0],
        )
        .unwrap();
        let plan = preflight_maxwell_submission_execution(
            &[dispatch],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::PostCompletionWrite {
                source: _,
                target,
                value,
            }] if *value == 0xcafe_babe_u32.to_le_bytes()
                && target.offset().get() == address
        ));

        execute_maxwell_software_initialization(plan, &address_space).unwrap();
        let mut bytes = [0_u8; 4];
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), 0xcafe_babe);
    }

    #[test]
    fn reserved_completion_requires_an_exact_signal_without_publication() {
        let (timeline, reservation) = reservation();
        let before = timeline.current_point();
        assert!(matches!(
            preflight_maxwell_submission_execution(
                &[],
                &address_space(),
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &MaxwellThreeDLoweringCache::default(),
            ),
            Err(MaxwellSubmissionExecutionError::MissingCompletionSignal {
                reserved,
                expected: 1,
                observed: 0,
            }) if reserved == reservation.point()
        ));
        assert_eq!(timeline.current_point(), before);
    }

    #[test]
    fn matching_syncpoint_is_retained_once_and_never_published() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        let increment = packet(0, 0x02c8 / 4, &[1]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &increment.packets()[0],
        )
        .unwrap();
        let (timeline, reservation) = reservation();
        let before = timeline.current_point();
        let plan = preflight_maxwell_submission_execution(
            std::slice::from_ref(&dispatch),
            &address_space(),
            FrontendSubmissionId::new(2),
            Vec::new(),
            Some(&reservation),
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert_eq!(plan.completion(), Some(reservation.point()));
        assert!(plan.steps().is_empty());
        assert_eq!(timeline.current_point(), before);

        assert!(matches!(
            preflight_maxwell_submission_execution(
                &[dispatch.clone(), dispatch],
                &address_space(),
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &MaxwellThreeDLoweringCache::default(),
            ),
            Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal {
                reserved,
                expected: 1,
                observed: 2,
            }) if reserved == reservation.point()
        ));
    }

    #[test]
    fn multi_increment_reservation_requires_exact_count_and_allows_later_work() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();

        let increment = packet(0, 0x02c8 / 4, &[1]);
        let first_increment = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &increment.packets()[0],
        )
        .unwrap();
        let second_increment = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &increment.packets()[0],
        )
        .unwrap();
        let flush = packet(0, 0x1144 / 4, &[0]);
        let later_flush = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &flush.packets()[0],
        )
        .unwrap();

        let owner = TimelineOwnerId::new(7);
        let mut timeline = GuestTimeline::new(
            GuestSyncpointId::new(1),
            TimelineInstanceId::new(1),
            owner,
            GuestSyncpointValue::new(0),
        );
        let reservation = timeline.reserve(owner, 2).unwrap();
        let before = timeline.current_point();
        let address_space = address_space();
        let plan = preflight_maxwell_submission_execution(
            &[first_increment, second_increment, later_flush],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            Some(&reservation),
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();

        assert_eq!(reservation.increments(), 2);
        assert_eq!(plan.completion(), Some(reservation.point()));
        assert_eq!(plan.steps().len(), 1);
        let completion = execute_maxwell_software_initialization(plan, &address_space).unwrap();
        assert_eq!(completion.completion(), Some(reservation.point()));
        assert_eq!(timeline.current_point(), before);
    }

    #[test]
    fn cache_maintenance_without_wfi_preserves_pending_work_until_an_explicit_wait() {
        let mut address_space = address_space();
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();

        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(1, 0, &[SWITCH_1_GM20B_PROFILE.classes().compute().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        for setup in [
            packet(1, 0x0188 / 4, &[(address >> 32) as u32, address as u32]),
            packet(1, 0x0180 / 4, &[4, 1]),
            packet(1, 0x01b0 / 4, &[0x41]),
        ] {
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(2),
                &setup.packets()[0],
            )
            .unwrap();
        }
        let data = packet(1, 0x01b4 / 4, &[0xfeed_beef]);
        let data_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &data.packets()[0],
        )
        .unwrap();
        let flush = packet(6, 0x002c / 4, &[0x8000_0000]);
        let flush_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &flush.packets()[0],
        )
        .unwrap();
        let invalidate = packet(6, 0x002c / 4, &[0x7000_0000]);
        let invalidate_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &invalidate.packets()[0],
        )
        .unwrap();
        let bind_three_d = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind_three_d.packets()[0],
        )
        .unwrap();
        let texture_invalidate = packet(0, 0x1288 / 4, &[0]);
        let texture_invalidate_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &texture_invalidate.packets()[0],
        )
        .unwrap();
        let shader_invalidate = packet(0, 0x0da4 / 4, &[0x1011]);
        let shader_invalidate_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &shader_invalidate.packets()[0],
        )
        .unwrap();
        let wait = packet(1, 0x0110 / 4, &[0]);
        let wait_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &wait.packets()[0],
        )
        .unwrap();

        let plan = preflight_maxwell_submission_execution(
            &[
                data_dispatch,
                flush_dispatch,
                invalidate_dispatch,
                texture_invalidate_dispatch,
                shader_invalidate_dispatch,
                wait_dispatch,
            ],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [
                MaxwellSubmissionExecutionStep::InlineWrite { value, target, .. },
                MaxwellSubmissionExecutionStep::BackendOperation(flush),
                MaxwellSubmissionExecutionStep::BackendOperation(invalidate),
                MaxwellSubmissionExecutionStep::BackendOperation(texture),
                MaxwellSubmissionExecutionStep::BackendOperation(shader),
            ] if *value == 0xfeed_beef_u32.to_le_bytes()
                && target.offset().get() == address
                && matches!(
                    flush.command(),
                    GpuCommand::CacheMaintenance(
                        CacheMaintenanceOperation::FlushDirtyDeviceWrites
                    )
                )
                && matches!(
                    texture.command(),
                    GpuCommand::CacheMaintenance(
                        CacheMaintenanceOperation::InvalidateTextureReadCaches
                    )
                )
                && matches!(
                    shader.command(),
                    GpuCommand::CacheMaintenance(
                        CacheMaintenanceOperation::InvalidateShaderCaches {
                            instruction: true,
                            global_data: true,
                            constant: true,
                        }
                    )
                )
        ));

        let mut bytes = [0xff; 4];
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0; 4]);

        let completion = execute_maxwell_software_initialization(plan, &address_space).unwrap();
        assert_eq!(completion.completion(), None);
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, 0xfeed_beef_u32.to_le_bytes());
    }

    #[test]
    fn tiled_cache_flush_reaches_the_backend_as_ordered_write_cache_maintenance() {
        let frontend = FrontendSubmissionId::new(2);
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(&mut channel, frontend, &bind.packets()[0]).unwrap();
        let flush = packet(0, 0x0f80 / 4, &[0]);
        let dispatch =
            dispatch_maxwell_engine_packet(&mut channel, frontend, &flush.packets()[0]).unwrap();
        let plan = preflight_maxwell_submission_execution(
            &[dispatch],
            &address_space(),
            frontend,
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();

        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::BackendOperation(operation)]
                if matches!(operation.command(), GpuCommand::CacheMaintenance(
                    CacheMaintenanceOperation::FlushDirtyDeviceWrites
                ))
        ));

        let mut calls = 0;
        let completion = execute_maxwell_backend_steps(
            &plan,
            &address_space(),
            |creations, invalidations, submission| {
                calls += 1;
                assert!(creations.is_empty());
                assert!(invalidations.is_empty());
                assert_eq!(submission.id(), frontend);
                assert_eq!(submission.operations().len(), 1);
                assert!(matches!(
                    submission.operations()[0].command(),
                    GpuCommand::CacheMaintenance(CacheMaintenanceOperation::FlushDirtyDeviceWrites)
                ));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(completion, Some(()));
    }

    #[test]
    fn standalone_inline_to_memory_words_are_preflighted_and_committed_atomically() {
        let mut address_space = address_space();
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(
            2,
            0,
            &[SWITCH_1_GM20B_PROFILE.classes().inline_to_memory().0],
        );
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        for setup in [
            packet(2, 0x0180 / 4, &[8, 1]),
            packet(2, 0x0188 / 4, &[(address >> 32) as u32, address as u32, 8]),
            packet(2, 0x01b0 / 4, &[0x1001]),
        ] {
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(2),
                &setup.packets()[0],
            )
            .unwrap();
        }
        let data = non_incrementing_packet(2, 0x01b4 / 4, &[0x1122_3344, 0x5566_7788]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &data.packets()[0],
        )
        .unwrap();
        let bind_three_d = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind_three_d.packets()[0],
        )
        .unwrap();
        let invalidate = packet(0, 0x1330 / 4, &[0]);
        let invalidate_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &invalidate.packets()[0],
        )
        .unwrap();
        let plan = preflight_maxwell_submission_execution(
            &[dispatch, invalidate_dispatch],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [
                MaxwellSubmissionExecutionStep::InlineWrite {
                    value: first,
                    target: first_target,
                    ..
                },
                MaxwellSubmissionExecutionStep::InlineWrite {
                    value: second,
                    target: second_target,
                    ..
                },
                MaxwellSubmissionExecutionStep::BackendOperation(operation),
            ] if *first == 0x1122_3344_u32.to_le_bytes()
                && first_target.offset().get() == address
                && *second == 0x5566_7788_u32.to_le_bytes()
                && second_target.offset().get() == address + 4
                && matches!(operation.command(), GpuCommand::CacheMaintenance(
                    CacheMaintenanceOperation::InvalidateSamplerCaches
                ))
        ));

        let mut bytes = [0xff; 8];
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0; 8]);
        execute_maxwell_software_initialization(plan, &address_space).unwrap();
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);
    }

    #[test]
    fn backend_segments_preserve_reused_constant_buffer_versions() {
        let mut address_space = address_space();
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        let frontend = FrontendSubmissionId::new(2);
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(&mut channel, frontend, &bind.packets()[0]).unwrap();
        let selector = packet(0, 0x2380 / 4, &[4, (address >> 32) as u32, address as u32]);
        dispatch_maxwell_engine_packet(&mut channel, frontend, &selector.packets()[0]).unwrap();

        let mut dispatches = Vec::new();
        for value in [0xff00_0000, 0x00ff_0000, 0x0000_ff00] {
            let load = packet(0, 0x238c / 4, &[0, value]);
            dispatches.push(
                dispatch_maxwell_engine_packet(&mut channel, frontend, &load.packets()[0]).unwrap(),
            );
            let invalidate = packet(0, 0x1288 / 4, &[0]);
            dispatches.push(
                dispatch_maxwell_engine_packet(&mut channel, frontend, &invalidate.packets()[0])
                    .unwrap(),
            );
        }
        let plan = preflight_maxwell_submission_execution(
            &dispatches,
            &address_space,
            frontend,
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();

        let mut observed = Vec::new();
        let completion = execute_maxwell_backend_steps(
            &plan,
            &address_space,
            |creations, invalidations, submission| {
                assert!(creations.is_empty());
                assert!(invalidations.is_empty());
                assert_eq!(submission.id(), frontend);
                assert_eq!(submission.operations().len(), 1);
                assert_eq!(submission.segment().get(), observed.len() as u32);
                assert_eq!(submission.is_final_segment(), observed.len() == 2);
                let mut bytes = [0_u8; 4];
                allocation.read(0, &mut bytes).unwrap();
                observed.push(u32::from_le_bytes(bytes));
                Ok(observed.len())
            },
        )
        .unwrap();

        assert_eq!(observed, [0xff00_0000, 0x00ff_0000, 0x0000_ff00]);
        assert_eq!(completion, Some(3));
    }

    #[test]
    fn dma_copy_converts_captured_rgba_pitch_rows_to_block_linear_storage() {
        const TEXTURE_SIZE: usize = 256 * 256 * 4;
        let source_allocation = CanonicalAllocation::zeroed(TEXTURE_SIZE, 0x1000).unwrap();
        let destination_allocation = CanonicalAllocation::zeroed(TEXTURE_SIZE, 0x1000).unwrap();
        let mut linear = vec![0_u8; TEXTURE_SIZE];
        for y in 0..256_u32 {
            for x in 0..256_u32 {
                let offset = (y as usize * 256 + x as usize) * 4;
                linear[offset..offset + 4].copy_from_slice(&[
                    x as u8,
                    y as u8,
                    (x ^ y) as u8,
                    0xff,
                ]);
            }
        }
        source_allocation.write(0, &linear).unwrap();

        let mut address_space = address_space();
        let source_mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(10),
                backing: source_allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: TEXTURE_SIZE as u64,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let destination_mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(11),
                backing: destination_allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: TEXTURE_SIZE as u64,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let source_address = source_mapping.offset().get();
        let destination_address = destination_mapping.offset().get();

        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(4, 0, &[SWITCH_1_GM20B_PROFILE.classes().dma_copy().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        for setup in [
            packet(4, 0x0708 / 4, &[0x0330_3210]),
            packet(4, 0x070c / 4, &[0x1040, 256, 256, 1, 0, 0]),
            packet(
                4,
                0x0400 / 4,
                &[
                    (source_address >> 32) as u32,
                    source_address as u32,
                    (destination_address >> 32) as u32,
                    destination_address as u32,
                    0x400,
                    0x400,
                    256,
                    256,
                ],
            ),
        ] {
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(2),
                &setup.packets()[0],
            )
            .unwrap();
        }
        let launch = packet(4, 0x0300 / 4, &[0x686]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &launch.packets()[0],
        )
        .unwrap();
        let plan = preflight_maxwell_submission_execution(
            &[dispatch],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::DmaCopy { operation, .. }]
                if operation.source_address() == source_address
                    && operation.destination_address() == destination_address
                    && operation.source_range_size() == TEXTURE_SIZE as u64
                    && operation.destination_range_size() == TEXTURE_SIZE as u64
        ));

        execute_maxwell_software_initialization(plan, &address_space).unwrap();
        let mut tiled = vec![0_u8; TEXTURE_SIZE];
        destination_allocation.read(0, &mut tiled).unwrap();
        for (x, y, offset) in [
            (0_u32, 0_u32, 0_usize),
            (4, 0, 32),
            (0, 1, 16),
            (0, 2, 64),
            (8, 0, 256),
            (0, 8, 512),
            (16, 0, 8192),
            (0, 128, 131072),
        ] {
            assert_eq!(
                &tiled[offset..offset + 4],
                &[x as u8, y as u8, (x ^ y) as u8, 0xff]
            );
        }
    }

    #[test]
    fn dma_copy_rejects_physical_launches_without_committing_candidate_state() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(4, 0, &[SWITCH_1_GM20B_PROFILE.classes().dma_copy().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        let before = channel.dma_copy().clone();
        let launch = packet(4, 0x0300 / 4, &[0x1001]);
        let error = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &launch.packets()[0],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::MaxwellEngineDispatchError::InvalidDmaCopyMethodEncoding {
                method_name: "LAUNCH_DMA",
                ..
            }
        ));
        assert_eq!(channel.dma_copy(), &before);
    }
}
