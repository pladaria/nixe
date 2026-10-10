//! Ordered streaming lowering and canonical execution for a frontend delivery.

use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{
    BackendResourceCreateInfo, CacheMaintenanceOperation, CapabilityRequirements,
    CommandDescriptionError, FrontendSubmissionId, FrontendSubmissionSegment, GpuCommand,
    GpuOperation, GuestTimelinePoint, OperationSubmission, ReservedTimelinePoint,
    ResourceDependency,
};
use nixe_memory::{CanonicalWriteBatch, CanonicalWriteBatchError, MemoryPermissions};

use crate::engines::{
    MaxwellEngineEvent, PendingEngineOperation, lower_maxwell_three_d_operation_into_cache,
};
use crate::{
    MaxwellComputeSynchronizationPlan, MaxwellGpuAccessError, MaxwellGpuAddressSpace,
    MaxwellHostSynchronizationKind, MaxwellLoweredWork, MaxwellLoweringCache, MaxwellLoweringError,
    MaxwellMemoryCopyError, MaxwellMemoryCopyOperation, MaxwellMethodSource, MaxwellResolvedRange,
    MaxwellShaderTranslationError, MaxwellThreeDResourceError, MaxwellThreeDSynchronizationError,
    MaxwellThreeDSynchronizationPlan, lower_maxwell_compute_synchronization,
    lower_maxwell_three_d_synchronization,
};

/// One ordered operation whose inputs have been resolved without side effects.
pub enum MaxwellSubmissionExecutionStep {
    NotificationWrite {
        source: MaxwellMethodSource,
        target: MaxwellResolvedRange,
    },
    WaitForIdle,
    SemaphoreRelease {
        source: MaxwellMethodSource,
        target: MaxwellResolvedRange,
        payload: u32,
        short: bool,
    },
    InlineWrite {
        source: MaxwellMethodSource,
        target: MaxwellResolvedRange,
        value: Vec<u8>,
    },
    MemoryCopy {
        operation: MaxwellMemoryCopyOperation,
        source: MaxwellResolvedRange,
        destination: MaxwellResolvedRange,
    },
    PostCompletionWrite {
        source: MaxwellMethodSource,
        target: MaxwellResolvedRange,
        value: [u8; 4],
    },
    BackendOperation(GpuOperation),
    Gpu(MaxwellLoweredWork),
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
    completion: Option<GuestTimelinePoint>,
}

impl MaxwellSubmissionExecutionPlan {
    pub(crate) fn prepend_steps(mut self, mut prefix: Vec<MaxwellSubmissionExecutionStep>) -> Self {
        prefix.extend(self.steps);
        self.steps = prefix.into_boxed_slice();
        self
    }

    #[must_use]
    pub fn into_steps(self) -> Box<[MaxwellSubmissionExecutionStep]> {
        self.steps
    }
    #[must_use]
    pub const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }

    #[must_use]
    pub fn steps(&self) -> &[MaxwellSubmissionExecutionStep] {
        &self.steps
    }

    #[must_use]
    pub const fn completion(&self) -> Option<GuestTimelinePoint> {
        self.completion
    }

    /// Returns whether this plan contains work which must cross a neutral GPU
    /// backend instead of the canonical-memory initialization executor.
    #[must_use]
    pub fn requires_backend(&self) -> bool {
        self.steps.iter().any(|step| {
            matches!(
                step,
                MaxwellSubmissionExecutionStep::BackendOperation(_)
                    | MaxwellSubmissionExecutionStep::Gpu(_)
            )
        })
    }
}

/// Failure before guest completion publication at the Maxwell/backend bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellBackendExecutionError {
    Canonical(Box<MaxwellCanonicalExecutionError>),
    InvalidSubmission(CommandDescriptionError),
}

impl Display for MaxwellBackendExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Canonical(error) => Display::fmt(error, formatter),
            Self::InvalidSubmission(error) => {
                write!(formatter, "neutral Maxwell submission is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for MaxwellBackendExecutionError {}

/// Failure while executing ordered canonical-memory commands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellCanonicalExecutionError {
    StaleInlineTarget {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    InlineWrite {
        source: MaxwellMethodSource,
        error: CanonicalWriteBatchError,
    },
    MemoryCopyTransform {
        source: MaxwellMethodSource,
        error: MaxwellMemoryCopyError,
    },
    MemoryCopyTransaction {
        source: MaxwellMethodSource,
        error: CanonicalWriteBatchError,
    },
}

impl Display for MaxwellCanonicalExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleInlineTarget { source, error } => {
                write!(
                    formatter,
                    "inline upload target changed before execution: {source}: {error}"
                )
            }
            Self::InlineWrite { source, error } => {
                write!(
                    formatter,
                    "inline upload could not be staged atomically: {source}: {error}"
                )
            }
            Self::MemoryCopyTransform { source, error } => {
                write!(
                    formatter,
                    "memory copy transformation failed: {source}: {error}"
                )
            }
            Self::MemoryCopyTransaction { source, error } => {
                write!(
                    formatter,
                    "memory copy transaction failed: {source}: {error}"
                )
            }
        }
    }
}

impl std::error::Error for MaxwellCanonicalExecutionError {}

/// Failure before any guest write, backend submission, or fence publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellSubmissionExecutionError {
    ComputeLaunch(crate::MaxwellComputeLaunchError),
    InlineAddress {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    MemoryCopyAddress {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    ThreeDResource(MaxwellThreeDResourceError),
    StagedMemory(Box<MaxwellCanonicalExecutionError>),
    Lowering(MaxwellLoweringError),
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
            Self::ComputeLaunch(error) => Display::fmt(error, formatter),
            Self::InlineAddress { source, error } => {
                write!(
                    formatter,
                    "inline upload target is invalid: {source}: {error}"
                )
            }
            Self::MemoryCopyAddress { source, error } => {
                write!(formatter, "memory copy range is invalid: {source}: {error}")
            }
            Self::ThreeDResource(error) => Display::fmt(error, formatter),
            Self::StagedMemory(error) => {
                write!(
                    formatter,
                    "ordered submission memory preflight failed: {error}"
                )
            }
            Self::Lowering(error) => Display::fmt(error, formatter),
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

/// Incremental owner of one ordered frontend lowering pass.
///
/// Packet dispatch feeds effects into this value before applying following
/// packets. Consequently a trigger's state can be consumed before later
/// register writes and does not need submission-lifetime retention.
pub(crate) struct MaxwellSubmissionPlanner<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    frontend: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    completion: Option<&'a ReservedTimelinePoint>,
    cache: &'a mut MaxwellLoweringCache,
    steps: Vec<MaxwellSubmissionExecutionStep>,
    prior_work_pending: bool,
    completion_signal_count: u32,
    driver_completion_increments: u32,
    staged_memory_writes: CanonicalWriteBatch,
    inline_write_pending: bool,
    inline_image_hint: Option<usize>,
    inline_image_uploads: Vec<(nixe_gpu::ImageRegion, Vec<u8>)>,
}

impl<'a> MaxwellSubmissionPlanner<'a> {
    /// Seal bounded prefixes once their inline payload is complete. The
    /// canonical overlay and semantic state stay with the producer, while
    /// already lowered work can be captured by the backend concurrently.
    pub(crate) fn take_ready_steps(&mut self) -> Option<Vec<MaxwellSubmissionExecutionStep>> {
        if self.inline_write_pending {
            return None;
        }
        let mut operations = self
            .steps
            .iter()
            .enumerate()
            .filter(|(_, step)| backend_step_emits_operation(step));
        let mut prefix_end = 0;
        for _ in 0..7 {
            prefix_end = operations.next()?.0 + 1;
        }
        // Retain at least one backend operation until completion validation.
        // A trailing WFI may drain every emitted prefix: the backend still
        // needs a real final segment to close its accepted frontend timeline.
        operations.next()?;
        Some(self.steps.drain(..prefix_end).collect())
    }
    pub(crate) const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }
    pub(crate) fn new(
        address_space: &'a MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
        completion: Option<&'a ReservedTimelinePoint>,
        cache: &'a mut MaxwellLoweringCache,
    ) -> Self {
        Self {
            address_space,
            frontend,
            predecessors,
            completion,
            cache,
            steps: Vec::new(),
            prior_work_pending: false,
            completion_signal_count: 0,
            driver_completion_increments: 0,
            staged_memory_writes: CanonicalWriteBatch::new(),
            inline_write_pending: false,
            inline_image_hint: None,
            inline_image_uploads: Vec::new(),
        }
    }

    pub(crate) fn push_event(
        &mut self,
        operation: MaxwellEngineEvent<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let MaxwellEngineEvent {
            operation,
            three_d,
            compute,
        } = operation;
        if !matches!(
            operation,
            PendingEngineOperation::InlineToMemory(_)
                | PendingEngineOperation::ComputeInlineToMemory(_)
                | PendingEngineOperation::ThreeDInlineConstantBuffer(_)
        ) {
            self.flush_inline_writes()?;
        }
        if !matches!(operation, PendingEngineOperation::InlineToMemory(_)) {
            self.flush_inline_images()?;
        }
        match operation {
            PendingEngineOperation::Notification { address, source } => {
                let target = self
                    .address_space
                    .address(address)
                    .map_err(MaxwellGpuAccessError::Address)
                    .and_then(|address| {
                        self.address_space
                            .resolve_range(address, 16, MemoryPermissions::WRITE)
                    })
                    .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress {
                        source,
                        error,
                    })?;
                self.steps
                    .push(MaxwellSubmissionExecutionStep::NotificationWrite { source, target });
                self.prior_work_pending = false;
            }
            PendingEngineOperation::ComputeLaunch(launch) => {
                let resolved = crate::engines::resolve_compute_launch(
                    &launch,
                    compute.expect("compute launch has live state"),
                    self.address_space,
                    &self.staged_memory_writes,
                )
                .map_err(MaxwellSubmissionExecutionError::ComputeLaunch)?;
                let work = self
                    .cache
                    .lower_compute(
                        &resolved,
                        self.address_space,
                        &self.staged_memory_writes,
                        self.frontend,
                        self.predecessors.clone(),
                    )
                    .map_err(MaxwellSubmissionExecutionError::Lowering)?;
                self.steps.push(MaxwellSubmissionExecutionStep::Gpu(work));
                self.prior_work_pending = true;
            }
            PendingEngineOperation::TwoDSolid(request) => {
                let [x0, y0, x1, y1] = request.rectangle;
                if x0 < x1 && y0 < y1 {
                    let limit = self.cache.resource_cache_limit();
                    let resources = self
                        .cache
                        .resolved_resources_mut()
                        .resolve_solid_image(&request, self.address_space, limit)
                        .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                    let work = self
                        .cache
                        .lower_solid_rect(
                            &request,
                            &resources,
                            self.frontend,
                            self.predecessors.clone(),
                        )
                        .map_err(MaxwellSubmissionExecutionError::Lowering)?;
                    self.steps.push(MaxwellSubmissionExecutionStep::Gpu(work));
                    self.prior_work_pending = true;
                }
            }
            PendingEngineOperation::TwoDBlit(request) => {
                let limit = self.cache.resource_cache_limit();
                let resources = self
                    .cache
                    .resolved_resources_mut()
                    .resolve_color_images(&request, self.address_space, limit)
                    .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                let work = self
                    .cache
                    .lower_color_blit(
                        &request,
                        &resources,
                        self.frontend,
                        self.predecessors.clone(),
                    )
                    .map_err(MaxwellSubmissionExecutionError::Lowering)?;
                self.steps.push(MaxwellSubmissionExecutionStep::Gpu(work));
                self.prior_work_pending = true;
            }
            PendingEngineOperation::HostSynchronization(operation) => {
                let command = match operation.operation() {
                    MaxwellHostSynchronizationKind::SemaphoreRelease {
                        address,
                        payload,
                        short,
                        wait,
                        source,
                    } => {
                        let target =
                            self.address_space
                                .address(address)
                                .map_err(MaxwellGpuAccessError::Address)
                                .and_then(|address| {
                                    self.address_space.resolve_range(
                                        address,
                                        if short { 4 } else { 16 },
                                        MemoryPermissions::WRITE,
                                    )
                                })
                                .map_err(|error| {
                                    MaxwellSubmissionExecutionError::InlineAddress { source, error }
                                })?;
                        self.steps
                            .push(MaxwellSubmissionExecutionStep::SemaphoreRelease {
                                source,
                                target,
                                payload,
                                short,
                            });
                        if wait {
                            self.prior_work_pending = false;
                        }
                        return Ok(());
                    }
                    MaxwellHostSynchronizationKind::WaitForIdle { .. } => {
                        // All engine work on this channel shares the ordered
                        // backend stream. Both scopes therefore drain its prior
                        // segment; they never complete another channel's work.
                        self.steps.push(MaxwellSubmissionExecutionStep::WaitForIdle);
                        self.prior_work_pending = false;
                        return Ok(());
                    }
                    MaxwellHostSynchronizationKind::L2SysmemInvalidate { .. } => {
                        GpuCommand::CacheMaintenance(
                            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
                        )
                    }
                    MaxwellHostSynchronizationKind::L2FlushDirty { .. } => {
                        GpuCommand::CacheMaintenance(
                            CacheMaintenanceOperation::FlushDirtyDeviceWrites,
                        )
                    }
                };
                self.steps
                    .push(MaxwellSubmissionExecutionStep::BackendOperation(
                        GpuOperation::new(command, [], [], CapabilityRequirements::none()),
                    ));
            }
            PendingEngineOperation::ComputeInlineToMemory(upload) => {
                // Both FLUSH_DISABLE and FLUSH_ONLY use coherent canonical writes.
                // Backend execution finishes any preceding segment before these
                // writes and commits them before the following backend operation;
                // write-only initialization commits before returning completion.
                // This also satisfies enabled SYSMEMBAR without an extra host wait
                // or per-upload flush command. Neither mode releases a semaphore.
                // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L129-L165
                self.push_inline_write(
                    upload.address().get(),
                    upload.offset(),
                    upload.value(),
                    upload.source(),
                )?;
            }
            PendingEngineOperation::InlineToMemory(upload) => {
                let target = resolve_inline_target(
                    self.address_space,
                    upload.address().get(),
                    upload.offset(),
                    upload.source(),
                )?;
                if let Some((hint, region)) = self
                    .cache
                    .inline_image_word(&target, self.inline_image_hint)
                    .map_err(MaxwellSubmissionExecutionError::Lowering)?
                {
                    self.flush_inline_writes()?;
                    self.inline_image_hint = Some(hint);
                    if let Some((previous, bytes)) = self.inline_image_uploads.last_mut()
                        && previous.image == region.image
                        && previous.subresources == region.subresources
                        && previous.origin.y == region.origin.y
                        && previous.origin.x + previous.extent.width == region.origin.x
                    {
                        previous.extent.width += 1;
                        bytes.extend_from_slice(&upload.value().to_le_bytes());
                    } else {
                        self.inline_image_uploads
                            .push((region, upload.value().to_le_bytes().to_vec()));
                    }
                } else {
                    self.flush_inline_images()?;
                    self.push_inline_write(
                        upload.address().get(),
                        upload.offset(),
                        upload.value(),
                        upload.source(),
                    )?;
                }
            }
            PendingEngineOperation::MemoryCopy(operation) => {
                let source_address = self
                    .address_space
                    .address(operation.source_address())
                    .map_err(MaxwellGpuAccessError::Address)
                    .map_err(|error| MaxwellSubmissionExecutionError::MemoryCopyAddress {
                        source: operation.source(),
                        error,
                    })?;
                let destination_address = self
                    .address_space
                    .address(operation.destination_address())
                    .map_err(MaxwellGpuAccessError::Address)
                    .map_err(|error| MaxwellSubmissionExecutionError::MemoryCopyAddress {
                        source: operation.source(),
                        error,
                    })?;
                let source = self
                    .address_space
                    .resolve_range(
                        source_address,
                        operation.source_range_size(),
                        MemoryPermissions::READ,
                    )
                    .map_err(|error| MaxwellSubmissionExecutionError::MemoryCopyAddress {
                        source: operation.source(),
                        error,
                    })?;
                let destination = self
                    .address_space
                    .resolve_range(
                        destination_address,
                        operation.destination_range_size(),
                        MemoryPermissions::READ_WRITE,
                    )
                    .map_err(|error| MaxwellSubmissionExecutionError::MemoryCopyAddress {
                        source: operation.source(),
                        error,
                    })?;
                stage_memory_copy(
                    operation,
                    &source,
                    &destination,
                    &mut self.staged_memory_writes,
                )
                .map_err(|error| MaxwellSubmissionExecutionError::StagedMemory(Box::new(error)))?;
                self.steps.push(MaxwellSubmissionExecutionStep::MemoryCopy {
                    operation,
                    source,
                    destination,
                });
                if let Some((address, payload)) = operation.semaphore_release {
                    let target =
                        resolve_inline_target(self.address_space, address, 0, operation.source())?;
                    stage_semaphore_release(
                        &target,
                        payload,
                        true,
                        0,
                        operation.source(),
                        &mut self.staged_memory_writes,
                    )
                    .map_err(|error| {
                        MaxwellSubmissionExecutionError::StagedMemory(Box::new(error))
                    })?;
                    self.steps
                        .push(MaxwellSubmissionExecutionStep::SemaphoreRelease {
                            source: operation.source(),
                            target,
                            payload,
                            short: true,
                        });
                }
                self.prior_work_pending = true;
            }
            PendingEngineOperation::ComputeSynchronization(operation) => {
                let plan =
                    lower_maxwell_compute_synchronization(&operation, self.prior_work_pending);
                match plan {
                    MaxwellComputeSynchronizationPlan::WaitForIdle { .. } => {
                        self.prior_work_pending = false;
                    }
                    MaxwellComputeSynchronizationPlan::InvalidateShaderCaches {
                        caches, ..
                    }
                    | MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches } => {
                        if matches!(
                            plan,
                            MaxwellComputeSynchronizationPlan::InvalidateShaderCaches { .. }
                        ) {
                            self.prior_work_pending = false;
                        }
                        self.steps
                            .push(MaxwellSubmissionExecutionStep::BackendOperation(
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
            PendingEngineOperation::ThreeDInlineConstantBuffer(upload) => {
                self.push_inline_write(
                    upload.address().get(),
                    upload.offset(),
                    upload.value(),
                    upload.source(),
                )?;
            }
            PendingEngineOperation::ThreeD(trigger) => {
                if trigger.is_draw() {
                    let translated = if let Some(translated) =
                        self.cache.reuse_translated_shaders_for_state(
                            three_d,
                            &self.staged_memory_writes,
                            self.address_space,
                        ) {
                        translated
                    } else {
                        let programs = self
                            .cache
                            .resolve_shader_translation_for_state(
                                three_d,
                                &self.staged_memory_writes,
                                self.address_space,
                            )
                            .map_err(MaxwellSubmissionExecutionError::ShaderTranslation)?;
                        let translated = Arc::new(
                            self.cache
                                .stage_shader_translations(&programs)
                                .map_err(MaxwellSubmissionExecutionError::Lowering)?,
                        );
                        self.cache
                            .retain_translated_shader_state(&programs, Arc::clone(&translated));
                        translated
                    };
                    let mut required_roles = self.cache.take_resource_roles();
                    required_roles.extend(
                        translated
                            .resources()
                            .iter()
                            .map(|resource| resource.role()),
                    );
                    trigger.append_resource_roles(three_d, &mut required_roles);
                    required_roles.sort_unstable();
                    required_roles.dedup();
                    let resource_cache_limit = self.cache.resource_cache_limit();
                    let resources = self
                        .cache
                        .resolved_resources_mut()
                        .resolve(
                            three_d,
                            self.address_space,
                            &required_roles,
                            Some(&self.staged_memory_writes),
                            false,
                            resource_cache_limit,
                        )
                        .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                    self.cache.recycle_resource_roles(required_roles);
                    let work = lower_maxwell_three_d_operation_into_cache(
                        three_d,
                        resources.as_ref(),
                        trigger,
                        Some(translated.as_ref()),
                        self.frontend,
                        self.predecessors.clone(),
                        self.cache,
                    )
                    .map_err(MaxwellSubmissionExecutionError::Lowering)?;
                    self.steps.push(MaxwellSubmissionExecutionStep::Gpu(work));
                    self.prior_work_pending = true;
                    return Ok(());
                }
                let mut required_roles = self.cache.take_resource_roles();
                trigger.append_resource_roles(three_d, &mut required_roles);
                required_roles.sort_unstable();
                required_roles.dedup();
                let resource_cache_limit = self.cache.resource_cache_limit();
                let resources = self
                    .cache
                    .resolved_resources_mut()
                    .resolve(
                        three_d,
                        self.address_space,
                        &required_roles,
                        Some(&self.staged_memory_writes),
                        false,
                        resource_cache_limit,
                    )
                    .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                self.cache.recycle_resource_roles(required_roles);
                let work = lower_maxwell_three_d_operation_into_cache(
                    three_d,
                    resources.as_ref(),
                    trigger,
                    None,
                    self.frontend,
                    self.predecessors.clone(),
                    self.cache,
                )
                .map_err(MaxwellSubmissionExecutionError::Lowering)?;
                self.steps.push(MaxwellSubmissionExecutionStep::Gpu(work));
                self.prior_work_pending = true;
            }
            PendingEngineOperation::ThreeDSynchronization(trigger) => {
                let plan = lower_maxwell_three_d_synchronization(
                    trigger,
                    three_d,
                    self.completion,
                    self.prior_work_pending,
                )
                .map_err(MaxwellSubmissionExecutionError::ThreeDSynchronization)?;
                if let MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
                    completion: reserved,
                    ..
                } = plan
                {
                    self.completion_signal_count = self.completion_signal_count.saturating_add(1);
                    let expected = self.completion.map_or(0, ReservedTimelinePoint::increments);
                    if self.completion_signal_count > expected {
                        return Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal {
                            reserved,
                            expected,
                            observed: self.completion_signal_count,
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
                        self.address_space,
                        release.address().get(),
                        0,
                        release.source(),
                    )?;
                    self.steps
                        .push(MaxwellSubmissionExecutionStep::PostCompletionWrite {
                            source: release.source(),
                            target,
                            value: release.payload().to_le_bytes(),
                        });
                } else if let Some(operation) = three_d_synchronization_operation(plan) {
                    self.steps
                        .push(MaxwellSubmissionExecutionStep::BackendOperation(operation));
                }
                if drains_prior_work {
                    self.prior_work_pending = false;
                }
            }
        }
        Ok(())
    }

    fn flush_inline_images(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        self.inline_image_hint = None;
        if self.inline_image_uploads.is_empty() {
            return Ok(());
        }
        self.flush_inline_writes()?;
        let work = self
            .cache
            .lower_inline_images(
                std::mem::take(&mut self.inline_image_uploads),
                self.frontend,
                self.predecessors.clone(),
            )
            .map_err(MaxwellSubmissionExecutionError::Lowering)?;
        self.steps.push(MaxwellSubmissionExecutionStep::Gpu(work));
        self.prior_work_pending = true;
        Ok(())
    }

    fn push_inline_write(
        &mut self,
        address: u64,
        offset: u32,
        value: u32,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        if let Some(MaxwellSubmissionExecutionStep::InlineWrite {
            source: previous_source,
            target,
            value: bytes,
        }) = self.steps.last_mut()
            && address
                .checked_add(u64::from(offset))
                .and_then(|address| target.append_inline_word(address))
                .is_some()
        {
            bytes.extend_from_slice(&value.to_le_bytes());
            *previous_source = source;
            self.inline_write_pending = true;
            return Ok(());
        }
        self.flush_inline_writes()?;
        let target = resolve_inline_target(self.address_space, address, offset, source)?;
        self.steps
            .push(MaxwellSubmissionExecutionStep::InlineWrite {
                source,
                target,
                value: value.to_le_bytes().to_vec(),
            });
        self.inline_write_pending = true;
        self.prior_work_pending = true;
        Ok(())
    }

    fn flush_inline_writes(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        if !self.inline_write_pending {
            return Ok(());
        }
        let Some(MaxwellSubmissionExecutionStep::InlineWrite {
            source,
            target,
            value,
        }) = self.steps.last()
        else {
            unreachable!("pending inline payload is the last ordered step")
        };
        stage_inline_write(
            self.address_space,
            target,
            value,
            *source,
            &mut self.staged_memory_writes,
        )
        .map_err(|error| MaxwellSubmissionExecutionError::StagedMemory(Box::new(error)))?;
        self.inline_write_pending = false;
        Ok(())
    }

    pub(crate) fn take_mme_scratch(&mut self) -> (Vec<crate::MaxwellMethodDispatch>, Vec<u32>) {
        self.cache.take_mme_scratch()
    }

    pub(crate) fn set_driver_completion_increments(&mut self, increments: u32) {
        self.driver_completion_increments = increments;
    }

    pub(crate) fn recycle_mme_scratch(
        &mut self,
        methods: Vec<crate::MaxwellMethodDispatch>,
        parameters: Vec<u32>,
    ) {
        self.cache.recycle_mme_scratch(methods, parameters);
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<MaxwellSubmissionExecutionPlan, MaxwellSubmissionExecutionError> {
        self.flush_inline_writes()?;
        self.flush_inline_images()?;
        if let Some(completion) = self.completion {
            let expected = completion
                .increments()
                .checked_sub(self.driver_completion_increments)
                .expect("driver increments are included in the timeline reservation");
            if self.completion_signal_count > expected {
                return Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal {
                    reserved: completion.point(),
                    expected,
                    observed: self.completion_signal_count,
                });
            }
            if self.completion_signal_count < expected {
                return Err(MaxwellSubmissionExecutionError::MissingCompletionSignal {
                    reserved: completion.point(),
                    expected,
                    observed: self.completion_signal_count,
                });
            }
        }

        Ok(MaxwellSubmissionExecutionPlan {
            frontend: self.frontend,
            predecessors: self.predecessors.into_boxed_slice(),
            steps: self.steps.into_boxed_slice(),
            completion: self.completion.map(ReservedTimelinePoint::point),
        })
    }
}

#[cfg(test)]
fn lower_test_pushbuffers(
    channel: &mut crate::MaxwellGpuChannel,
    pushbuffers: &[crate::MaxwellDecodedPushbuffer],
    address_space: &MaxwellGpuAddressSpace,
    frontend: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    completion: Option<&ReservedTimelinePoint>,
    cache: &mut MaxwellLoweringCache,
) -> Result<MaxwellSubmissionExecutionPlan, MaxwellSubmissionExecutionError> {
    let mut planner =
        MaxwellSubmissionPlanner::new(address_space, frontend, predecessors, completion, cache);
    let (mut mme_methods, mut mme_parameters) = planner.take_mme_scratch();
    for pushbuffer in pushbuffers {
        for packet in pushbuffer.packets() {
            if let Err(error) = crate::engines::stream_maxwell_engine_packet(
                channel,
                frontend,
                packet,
                None,
                &mut mme_methods,
                &mut mme_parameters,
                &mut |event| planner.push_event(event),
            ) {
                match error {
                    crate::engines::MaxwellEngineStreamError::Dispatch(error) => panic!("{error}"),
                    crate::engines::MaxwellEngineStreamError::Consumer(error) => return Err(error),
                }
            }
        }
    }
    planner.recycle_mme_scratch(mme_methods, mme_parameters);
    planner.finish()
}

pub struct MaxwellBackendSegment {
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
}

impl MaxwellBackendSegment {
    #[must_use]
    pub fn creations(&self) -> &[BackendResourceCreateInfo] {
        &self.creations
    }

    #[must_use]
    pub fn invalidations(&self) -> &[ResourceDependency] {
        &self.invalidations
    }

    #[must_use]
    pub const fn submission(&self) -> &OperationSubmission {
        &self.submission
    }
}

/// Resumable Maxwell execution with ordered host submission boundaries.
///
/// Canonical command-processor writes remain at their exact command-order
/// boundaries. Host snapshots of read-only inputs may be updated after host
/// acceptance; writes to device-owned pages still wait. Post-completion writes
/// are committed only after the last host
/// segment, while guest timeline publication remains with the GPU owner.
pub struct MaxwellBackendExecution {
    frontend: FrontendSubmissionId,
    predecessors: Box<[FrontendSubmissionId]>,
    steps: Vec<MaxwellSubmissionExecutionStep>,
    next_step: usize,
    pre_writes: CanonicalWriteBatch,
    post_writes: CanonicalWriteBatch,
    pre_write_source: Option<CanonicalWriteSource>,
    post_write_source: Option<MaxwellMethodSource>,
    creations: Vec<BackendResourceCreateInfo>,
    invalidations: Vec<ResourceDependency>,
    operations: Vec<GpuOperation>,
    batchable_render_pass_begin: Option<usize>,
    segments: BackendSegmentCursor,
    completion: Option<GuestTimelinePoint>,
    awaiting_resume: bool,
    submitted_any: bool,
    has_backend_steps: bool,
    finished: bool,
    sealed: bool,
}

impl MaxwellBackendExecution {
    /// All command-processor mutations have been committed and accepted host
    /// snapshots no longer need the ordered frontend preparation boundary.
    /// Completion writes and any unexecuted suffix keep that boundary closed.
    #[must_use]
    pub fn can_prepare_following(&self) -> bool {
        self.sealed
            && self.awaiting_resume
            && self.next_step == self.steps.len()
            && self.post_writes.is_empty()
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    #[must_use]
    pub const fn awaiting_completion(&self) -> bool {
        self.awaiting_resume
    }

    pub fn begin(frontend: FrontendSubmissionId, completion: Option<GuestTimelinePoint>) -> Self {
        Self {
            frontend,
            predecessors: Box::default(),
            steps: Vec::new(),
            next_step: 0,
            pre_writes: CanonicalWriteBatch::new(),
            post_writes: CanonicalWriteBatch::new(),
            pre_write_source: None,
            post_write_source: None,
            creations: Vec::new(),
            invalidations: Vec::new(),
            operations: Vec::new(),
            batchable_render_pass_begin: None,
            segments: BackendSegmentCursor {
                next: FrontendSubmissionSegment::FIRST,
                remaining_operations: 0,
            },
            completion,
            awaiting_resume: false,
            submitted_any: false,
            has_backend_steps: false,
            finished: false,
            sealed: false,
        }
    }

    pub fn append_steps(&mut self, steps: Vec<MaxwellSubmissionExecutionStep>) {
        assert!(
            !self.sealed && !self.finished,
            "a completed delivery cannot append commands"
        );
        self.segments.remaining_operations += steps
            .iter()
            .filter(|step| backend_step_emits_operation(step))
            .count();
        self.has_backend_steps |= steps.iter().any(backend_step_emits_operation);
        self.steps.extend(steps);
    }

    pub fn seal(&mut self) {
        assert!(!self.sealed, "a frontend delivery is sealed once");
        self.sealed = true;
    }

    #[must_use]
    pub fn new(plan: MaxwellSubmissionExecutionPlan) -> Self {
        let mut execution = Self::begin(plan.frontend, plan.completion);
        execution.predecessors = plan.predecessors;
        execution.append_steps(plan.steps.into_vec());
        execution.seal();
        execution
    }

    #[must_use]
    pub const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }

    #[must_use]
    pub const fn completion(&self) -> Option<GuestTimelinePoint> {
        self.completion
    }

    pub fn resume_segment(&mut self) {
        assert!(self.awaiting_resume, "Maxwell segment was not pending");
        self.awaiting_resume = false;
    }

    /// After host acceptance, earlier reads already own their encoded input
    /// snapshots. Updating CPU-visible canonical bytes cannot alter those
    /// reads: later uploads execute after them on the same host queue. Device
    /// writes, explicit WFI, copies and guest completion still require waits.
    /// https://github.com/eden-emulator/mirror/blob/master/src/video_core/engines/maxwell_3d.cpp
    #[must_use]
    pub fn can_continue_after_submission(&self) -> bool {
        if !self.awaiting_resume
            || !matches!(
                self.steps.get(self.next_step),
                Some(MaxwellSubmissionExecutionStep::InlineWrite { .. })
            )
        {
            return false;
        }
        for step in &self.steps[self.next_step..] {
            match step {
                MaxwellSubmissionExecutionStep::InlineWrite { target, .. } => {
                    if !target.segments().iter().all(|segment| {
                        segment
                            .mapping()
                            .backing()
                            .subrange_is_cpu_visible(segment.backing_offset(), segment.size())
                            == Ok(true)
                    }) {
                        return false;
                    }
                }
                MaxwellSubmissionExecutionStep::Gpu(_)
                | MaxwellSubmissionExecutionStep::BackendOperation(_) => return true,
                _ => return false,
            }
        }
        false
    }

    pub fn next_segment(
        &mut self,
        gpu_timestamp: u64,
    ) -> Result<Option<MaxwellBackendSegment>, MaxwellBackendExecutionError> {
        assert!(
            !self.awaiting_resume,
            "Maxwell execution must resume its pending segment before advancing"
        );
        if self.finished {
            return Ok(None);
        }

        while self.next_step < self.steps.len() {
            if backend_step_requires_prior_completion(&self.steps[self.next_step])
                && let Some(segment) = self.take_segment()?
            {
                if matches!(
                    self.steps[self.next_step],
                    MaxwellSubmissionExecutionStep::InlineWrite { .. }
                ) {
                    nixe_gpu::metrics::record(
                        nixe_gpu::metrics::Counter::InlineWriteSegmentBreaks,
                        1,
                    );
                }
                return Ok(Some(segment));
            }
            match &self.steps[self.next_step] {
                MaxwellSubmissionExecutionStep::NotificationWrite { source, target } => {
                    stage_notification(target, gpu_timestamp, *source, &mut self.pre_writes)
                        .map_err(|error| {
                            MaxwellBackendExecutionError::Canonical(Box::new(error))
                        })?;
                    self.pre_write_source = Some(CanonicalWriteSource::Inline(*source));
                    commit_pending_backend_writes(
                        &mut self.pre_writes,
                        &mut self.pre_write_source,
                    )?;
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::SemaphoreRelease {
                    source,
                    target,
                    payload,
                    short,
                } => {
                    stage_semaphore_release(
                        target,
                        *payload,
                        *short,
                        gpu_timestamp,
                        *source,
                        &mut self.pre_writes,
                    )
                    .map_err(|error| MaxwellBackendExecutionError::Canonical(Box::new(error)))?;
                    self.pre_write_source = Some(CanonicalWriteSource::Inline(*source));
                    commit_pending_backend_writes(
                        &mut self.pre_writes,
                        &mut self.pre_write_source,
                    )?;
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::WaitForIdle => {
                    commit_pending_backend_writes(
                        &mut self.pre_writes,
                        &mut self.pre_write_source,
                    )?;
                    self.batchable_render_pass_begin = None;
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::BackendOperation(operation) => {
                    commit_pending_backend_writes(
                        &mut self.pre_writes,
                        &mut self.pre_write_source,
                    )?;
                    self.batchable_render_pass_begin = None;
                    self.operations.push(operation.clone());
                    self.segments.remaining_operations -= 1;
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::InlineWrite {
                    source,
                    target,
                    value,
                } => {
                    stage_resolved_inline_write(target, value, *source, &mut self.pre_writes)
                        .map_err(|error| {
                            MaxwellBackendExecutionError::Canonical(Box::new(error))
                        })?;
                    self.pre_write_source = Some(CanonicalWriteSource::Inline(*source));
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::MemoryCopy {
                    operation,
                    source,
                    destination,
                } => {
                    stage_memory_copy(*operation, source, destination, &mut self.pre_writes)
                        .map_err(|error| {
                            MaxwellBackendExecutionError::Canonical(Box::new(error))
                        })?;
                    self.pre_write_source =
                        Some(CanonicalWriteSource::MemoryCopy(operation.source()));
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::PostCompletionWrite {
                    source,
                    target,
                    value,
                } => {
                    stage_resolved_inline_write(target, value, *source, &mut self.post_writes)
                        .map_err(|error| {
                            MaxwellBackendExecutionError::Canonical(Box::new(error))
                        })?;
                    self.post_write_source = Some(*source);
                    self.next_step += 1;
                }
                MaxwellSubmissionExecutionStep::Gpu(work) => {
                    commit_pending_backend_writes(
                        &mut self.pre_writes,
                        &mut self.pre_write_source,
                    )?;
                    self.creations.extend_from_slice(work.resource_creations());
                    self.invalidations
                        .extend_from_slice(work.resource_invalidations());
                    append_batchable_operations(
                        &mut self.operations,
                        &mut self.batchable_render_pass_begin,
                        work.submission().operations(),
                    );
                    self.segments.remaining_operations -= 1;
                    self.next_step += 1;
                }
            }
        }

        if !self.sealed {
            return Ok(None);
        }
        commit_pending_backend_writes(&mut self.pre_writes, &mut self.pre_write_source)?;
        if let Some(segment) = self.take_segment()? {
            return Ok(Some(segment));
        }
        if self.has_backend_steps && !self.submitted_any {
            return Err(MaxwellBackendExecutionError::InvalidSubmission(
                CommandDescriptionError::EmptySubmission,
            ));
        }
        commit_inline_batch(
            std::mem::take(&mut self.post_writes),
            self.post_write_source.take(),
        )
        .map_err(|error| MaxwellBackendExecutionError::Canonical(Box::new(error)))?;
        self.finished = true;
        Ok(None)
    }

    fn take_segment(
        &mut self,
    ) -> Result<Option<MaxwellBackendSegment>, MaxwellBackendExecutionError> {
        if self.operations.is_empty() {
            return Ok(None);
        }
        let final_segment = self.sealed && self.segments.remaining_operations == 0;
        let segment = self.segments.take(final_segment)?;
        let submission = OperationSubmission::new_segment(
            self.frontend,
            segment,
            final_segment,
            self.predecessors.to_vec(),
            std::mem::take(&mut self.operations),
        )
        .map_err(MaxwellBackendExecutionError::InvalidSubmission)?;
        self.batchable_render_pass_begin = None;
        self.awaiting_resume = true;
        self.submitted_any = true;
        Ok(Some(MaxwellBackendSegment {
            creations: std::mem::take(&mut self.creations).into_boxed_slice(),
            invalidations: std::mem::take(&mut self.invalidations).into_boxed_slice(),
            submission,
        }))
    }
}

fn commit_pending_backend_writes(
    writes: &mut CanonicalWriteBatch,
    source: &mut Option<CanonicalWriteSource>,
) -> Result<(), MaxwellBackendExecutionError> {
    if writes.is_empty() {
        return Ok(());
    }
    commit_write_batch(std::mem::take(writes), source.take())
        .map_err(|error| MaxwellBackendExecutionError::Canonical(Box::new(error)))
}

struct BackendSegmentCursor {
    next: FrontendSubmissionSegment,
    remaining_operations: usize,
}

impl BackendSegmentCursor {
    fn take(
        &mut self,
        final_segment: bool,
    ) -> Result<FrontendSubmissionSegment, MaxwellBackendExecutionError> {
        let segment = self.next;
        if !final_segment {
            self.next = segment
                .checked_next()
                .ok_or_else(too_many_backend_segments)?;
        }
        Ok(segment)
    }
}

fn backend_step_requires_prior_completion(step: &MaxwellSubmissionExecutionStep) -> bool {
    matches!(
        step,
        MaxwellSubmissionExecutionStep::WaitForIdle
            | MaxwellSubmissionExecutionStep::SemaphoreRelease { .. }
            | MaxwellSubmissionExecutionStep::NotificationWrite { .. }
            | MaxwellSubmissionExecutionStep::InlineWrite { .. }
            | MaxwellSubmissionExecutionStep::MemoryCopy { .. }
    )
}

fn backend_step_emits_operation(step: &MaxwellSubmissionExecutionStep) -> bool {
    matches!(
        step,
        MaxwellSubmissionExecutionStep::BackendOperation(_)
            | MaxwellSubmissionExecutionStep::Gpu(_)
    )
}

fn append_batchable_operations(
    target: &mut Vec<GpuOperation>,
    current_begin: &mut Option<usize>,
    operations: &[GpuOperation],
) {
    let next = batchable_render_pass(operations);
    let can_merge = current_begin
        .and_then(|begin| target.get(begin))
        .zip(next)
        .is_some_and(|(left, right)| render_passes_can_merge(left, right));
    if can_merge {
        target.pop();
        target.extend_from_slice(&operations[1..]);
    } else {
        let begin = target.len();
        target.extend_from_slice(operations);
        *current_begin = next.map(|_| begin);
    }
}

fn batchable_render_pass(
    operations: &[GpuOperation],
) -> Option<(nixe_gpu::RenderPassId, &[nixe_gpu::RenderAttachment])> {
    let (
        Some(GpuCommand::RenderPass(nixe_gpu::RenderPassOperation::Begin {
            render_pass,
            attachments,
        })),
        Some(GpuCommand::RenderPass(nixe_gpu::RenderPassOperation::End { render_pass: end })),
    ) = (
        operations.first().map(GpuOperation::command),
        operations.last().map(GpuOperation::command),
    )
    else {
        return None;
    };
    (*render_pass == *end).then_some((*render_pass, attachments))
}

fn render_passes_can_merge(
    left: &GpuOperation,
    right: (nixe_gpu::RenderPassId, &[nixe_gpu::RenderAttachment]),
) -> bool {
    let GpuCommand::RenderPass(nixe_gpu::RenderPassOperation::Begin {
        render_pass: left_pass,
        attachments: left_attachments,
    }) = left.command()
    else {
        return false;
    };
    *left_pass == right.0
        && left_attachments.len() == right.1.len()
        && left_attachments
            .iter()
            .zip(right.1.iter())
            .all(|(left, right)| {
                left.image == right.image
                    && left.subresources == right.subresources
                    && left.kind == right.kind
                    && left.format == right.format
                    && left.samples == right.samples
                    && left.store == nixe_gpu::AttachmentStore::Store
                    && right.load == nixe_gpu::AttachmentLoad::Load
                    && right.store == nixe_gpu::AttachmentStore::Store
            })
}

fn too_many_backend_segments() -> MaxwellBackendExecutionError {
    MaxwellBackendExecutionError::InvalidSubmission(
        CommandDescriptionError::TooManySubmissionSegments,
    )
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
        MaxwellThreeDSynchronizationPlan::PixelShaderBarrier { .. }
        | MaxwellThreeDSynchronizationPlan::TiledCacheBarrier
        | MaxwellThreeDSynchronizationPlan::FlushPendingWrites { .. } => {
            // The neutral ordered-write boundary also covers fragment outputs.
            // Conservatively expose writes for both SYSMEMBAR modes: backends
            // own coherent device storage and materialize CPU reads on demand.
            // This is GPU ordering, not a CPU wait or a syncpoint completion;
            // in particular a fragment barrier does not drain unrelated stages.
            // Tile-local ordering is satisfied by this stronger whole-pass
            // boundary: host backends do not replay draws per Maxwell tile.
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
    MemoryCopy(MaxwellMethodSource),
}

fn commit_write_batch(
    writes: CanonicalWriteBatch,
    source: Option<CanonicalWriteSource>,
) -> Result<(), MaxwellCanonicalExecutionError> {
    writes.commit_ordered().map_err(|error| {
        match source.expect("a non-empty canonical write batch has an ordered method source") {
            CanonicalWriteSource::Inline(source) => {
                MaxwellCanonicalExecutionError::InlineWrite { source, error }
            }
            CanonicalWriteSource::MemoryCopy(source) => {
                MaxwellCanonicalExecutionError::MemoryCopyTransaction { source, error }
            }
        }
    })
}

fn stage_memory_copy(
    operation: MaxwellMemoryCopyOperation,
    source: &MaxwellResolvedRange,
    destination: &MaxwellResolvedRange,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellCanonicalExecutionError> {
    let mut source_bytes = memory_copy_buffer(operation.source_range_size(), operation.source())?;
    read_memory_copy_bytes(source, &mut source_bytes, writes, operation.source())?;
    if !operation.has_remap() {
        // Read and stage only touched pages. In a cropped tiled transfer, the
        // validated address span can include most of a surface before the first
        // copied row. Do not repeatedly copy or invalidate that untouched prefix.
        // Snapshot the source first so overlapping surfaces retain copy semantics.
        let mut pages = BTreeMap::<usize, Vec<u8>>::new();
        for region in operation.byte_regions() {
            let (mut src, mut dst, mut count) =
                region.map_err(
                    |error| MaxwellCanonicalExecutionError::MemoryCopyTransform {
                        source: operation.source(),
                        error,
                    },
                )?;
            while count != 0 {
                const PAGE_BYTES: usize = 4096;
                let base = dst & !(PAGE_BYTES - 1);
                let page = match pages.entry(base) {
                    std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        let size = (operation.destination_range_size() - base as u64)
                            .min(PAGE_BYTES as u64);
                        let mut bytes = memory_copy_buffer(size, operation.source())?;
                        read_memory_copy_subrange(
                            destination,
                            base,
                            &mut bytes,
                            writes,
                            operation.source(),
                        )?;
                        entry.insert(bytes)
                    }
                };
                let offset = dst - base;
                let size = count.min(page.len() - offset);
                page[offset..offset + size].copy_from_slice(&source_bytes[src..src + size]);
                src += size;
                dst += size;
                count -= size;
            }
        }
        for (base, bytes) in pages {
            stage_memory_copy_subrange(destination, base, &bytes, writes, operation.source())?;
        }
        return Ok(());
    }
    let mut destination_bytes =
        memory_copy_buffer(operation.destination_range_size(), operation.source())?;
    read_memory_copy_bytes(
        destination,
        &mut destination_bytes,
        writes,
        operation.source(),
    )?;
    operation
        .copy_bytes(&source_bytes, &mut destination_bytes)
        .map_err(
            |error| MaxwellCanonicalExecutionError::MemoryCopyTransform {
                source: operation.source(),
                error,
            },
        )?;

    stage_memory_copy_subrange(
        destination,
        0,
        &destination_bytes,
        writes,
        operation.source(),
    )
}

fn memory_copy_segments(
    range: &MaxwellResolvedRange,
    offset: usize,
    size: usize,
) -> impl Iterator<Item = (&crate::MaxwellResolvedMapping, u64, std::ops::Range<usize>)> {
    let begin = offset as u64;
    let end = begin + size as u64;
    range.segments().iter().filter_map(move |segment| {
        let segment_begin = segment.gpu_offset().get() - range.offset().get();
        let start = begin.max(segment_begin);
        let stop = end.min(segment_begin + segment.size());
        (start < stop).then(|| {
            (
                segment,
                segment.backing_offset() + start - segment_begin,
                (start - begin) as usize..(stop - begin) as usize,
            )
        })
    })
}

fn stage_memory_copy_subrange(
    range: &MaxwellResolvedRange,
    offset: usize,
    bytes: &[u8],
    writes: &mut CanonicalWriteBatch,
    source: MaxwellMethodSource,
) -> Result<(), MaxwellCanonicalExecutionError> {
    for (segment, backing_offset, indices) in memory_copy_segments(range, offset, bytes.len()) {
        writes
            .stage(segment.mapping().backing(), backing_offset, &bytes[indices])
            .map_err(
                |error| MaxwellCanonicalExecutionError::MemoryCopyTransaction { source, error },
            )?;
    }
    Ok(())
}

fn read_memory_copy_bytes(
    range: &MaxwellResolvedRange,
    output: &mut [u8],
    writes: &CanonicalWriteBatch,
    source: MaxwellMethodSource,
) -> Result<(), MaxwellCanonicalExecutionError> {
    if output.len() as u64 != range.size() {
        return Err(MaxwellCanonicalExecutionError::MemoryCopyTransform {
            source,
            error: MaxwellMemoryCopyError::RangeSizeMismatch,
        });
    }
    read_memory_copy_subrange(range, 0, output, writes, source)
}

fn read_memory_copy_subrange(
    range: &MaxwellResolvedRange,
    offset: usize,
    output: &mut [u8],
    writes: &CanonicalWriteBatch,
    source: MaxwellMethodSource,
) -> Result<(), MaxwellCanonicalExecutionError> {
    for (segment, backing_offset, indices) in memory_copy_segments(range, offset, output.len()) {
        // read_overlay establishes CPU visibility and overlays preceding ordered
        // writes; a separate canonical read would duplicate the same work.
        writes
            .read_overlay(
                segment.mapping().backing(),
                backing_offset,
                &mut output[indices],
            )
            .map_err(
                |error| MaxwellCanonicalExecutionError::MemoryCopyTransaction { source, error },
            )?;
    }
    Ok(())
}

fn memory_copy_buffer(
    size: u64,
    source: MaxwellMethodSource,
) -> Result<Vec<u8>, MaxwellCanonicalExecutionError> {
    let size =
        usize::try_from(size).map_err(|_| MaxwellCanonicalExecutionError::MemoryCopyTransform {
            source,
            error: MaxwellMemoryCopyError::ArithmeticOverflow,
        })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|_| {
        MaxwellCanonicalExecutionError::MemoryCopyTransform {
            source,
            error: MaxwellMemoryCopyError::ResourceExhausted,
        }
    })?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn commit_inline_batch(
    writes: CanonicalWriteBatch,
    source: Option<MaxwellMethodSource>,
) -> Result<(), MaxwellCanonicalExecutionError> {
    writes
        .commit_ordered()
        .map_err(|error| MaxwellCanonicalExecutionError::InlineWrite {
            source: source.expect("a non-empty canonical write batch has an inline source"),
            error,
        })
}

fn stage_inline_write(
    address_space: &MaxwellGpuAddressSpace,
    target: &MaxwellResolvedRange,
    bytes: &[u8],
    source: MaxwellMethodSource,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellCanonicalExecutionError> {
    if target.address_space() != address_space.id() {
        return Err(MaxwellCanonicalExecutionError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::WrongAddressSpace {
                expected: target.address_space(),
                actual: address_space.id(),
            },
        });
    }
    if !target.permissions().contains(MemoryPermissions::WRITE) {
        return Err(MaxwellCanonicalExecutionError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::PermissionDenied {
                address: target.offset(),
                required: MemoryPermissions::WRITE,
                available: target.permissions(),
            },
        });
    }
    if target.size() != bytes.len() as u64 {
        return Err(MaxwellCanonicalExecutionError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::OutputSizeMismatch {
                expected: target.size(),
                actual: bytes.len() as u64,
            },
        });
    }
    for segment in target.segments() {
        if !address_space.retained_mapping_is_current(segment.mapping()) {
            return Err(MaxwellCanonicalExecutionError::StaleInlineTarget {
                source,
                error: MaxwellGpuAccessError::StaleMapping {
                    mapping: segment.mapping().id(),
                    generation: segment.mapping().generation(),
                },
            });
        }
    }

    stage_resolved_inline_write(target, bytes, source, writes)
}

/// Stages bytes through mappings retained by an already accepted GPU plan.
///
/// Unmapping a GPU virtual range after submission must not invalidate work
/// which already retained its canonical mappings. Address-space currency is
/// therefore checked while preflighting the plan, not again on the backend
/// owner thread.
// The PGRAPH notifier is timestamp:u64, info32:u32, info16:u16, status:u16.
// Hardware completion clears both information fields and status after prior work.
// https://envytools.readthedocs.io/en/latest/hw/graph/intro.html#notifiers
// https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/nvidia/src/kernel/gpu/mem_mgr/method_notification.c
fn stage_notification(
    target: &MaxwellResolvedRange,
    gpu_timestamp: u64,
    source: MaxwellMethodSource,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellCanonicalExecutionError> {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&gpu_timestamp.to_le_bytes());
    stage_resolved_inline_write(target, &bytes, source, writes)
}

fn stage_semaphore_release(
    target: &MaxwellResolvedRange,
    payload: u32,
    short: bool,
    gpu_timestamp: u64,
    source: MaxwellMethodSource,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellCanonicalExecutionError> {
    if short {
        stage_resolved_inline_write(target, &payload.to_le_bytes(), source, writes)
    } else {
        let mut bytes = [0_u8; 16];
        bytes[..4].copy_from_slice(&payload.to_le_bytes());
        bytes[8..].copy_from_slice(&gpu_timestamp.to_le_bytes());
        stage_resolved_inline_write(target, &bytes, source, writes)
    }
}

fn stage_resolved_inline_write(
    target: &MaxwellResolvedRange,
    bytes: &[u8],
    source: MaxwellMethodSource,
    writes: &mut CanonicalWriteBatch,
) -> Result<(), MaxwellCanonicalExecutionError> {
    if !target.permissions().contains(MemoryPermissions::WRITE) {
        return Err(MaxwellCanonicalExecutionError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::PermissionDenied {
                address: target.offset(),
                required: MemoryPermissions::WRITE,
                available: target.permissions(),
            },
        });
    }
    if target.size() != bytes.len() as u64 {
        return Err(MaxwellCanonicalExecutionError::StaleInlineTarget {
            source,
            error: MaxwellGpuAccessError::OutputSizeMismatch {
                expected: target.size(),
                actual: bytes.len() as u64,
            },
        });
    }

    let mut copied = 0_usize;
    for segment in target.segments() {
        let size = usize::try_from(segment.size()).map_err(|_| {
            MaxwellCanonicalExecutionError::StaleInlineTarget {
                source,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            }
        })?;
        let end =
            copied
                .checked_add(size)
                .ok_or(MaxwellCanonicalExecutionError::StaleInlineTarget {
                    source,
                    error: MaxwellGpuAccessError::ArithmeticOverflow,
                })?;
        writes
            .stage(
                segment.mapping().backing(),
                segment.backing_offset(),
                &bytes[copied..end],
            )
            .map_err(|error| MaxwellCanonicalExecutionError::InlineWrite { source, error })?;
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
        AttachmentLoad, AttachmentStore, CapabilityRequirements, DrawArguments, DrawOperation,
        FrontendSubmissionId, GpuVirtualAddress, GuestSyncpointId, GuestSyncpointValue,
        GuestTimeline, ImageFormat, ImageId, ImageKind, ImageSubresourceRange, MappingGeneration,
        PipelineId, PreparedDraw, PrimitiveTopology, RenderAttachment,
        RenderPassAttachmentDescription, RenderPassDescription, RenderPassId, RenderPassOperation,
        SampleCount, TimelineInstanceId, TimelineOwnerId,
    };
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    use super::*;
    use crate::engines::dispatch_maxwell_engine_packet;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellChannelId, MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellGpuChannel,
        MaxwellMapRequest, MaxwellMappingId, MaxwellPushbufferWord, SWITCH_1_GM20B_PROFILE,
        decode_maxwell_pushbuffer,
    };

    fn execute_canonical_test_plan(
        plan: MaxwellSubmissionExecutionPlan,
        timestamp: u64,
    ) -> Result<Option<GuestTimelinePoint>, MaxwellBackendExecutionError> {
        assert!(!plan.requires_backend());
        let mut execution = MaxwellBackendExecution::new(plan);
        assert!(execution.next_segment(timestamp)?.is_none());
        assert!(execution.is_finished());
        Ok(execution.completion())
    }

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

    fn complete_backend_execution(
        plan: MaxwellSubmissionExecutionPlan,
    ) -> Option<GuestTimelinePoint> {
        let mut execution = MaxwellBackendExecution::new(plan);
        while execution.next_segment(0).unwrap().is_some() {
            execution.resume_segment();
        }
        execution.completion()
    }

    fn render_pass_operations(pass: RenderPassId, image: ImageId) -> Vec<GpuOperation> {
        let description = RenderPassDescription::new(vec![RenderPassAttachmentDescription {
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
        }])
        .unwrap();
        let attachment = RenderAttachment {
            image,
            subresources: ImageSubresourceRange {
                plane: 0,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
            load: AttachmentLoad::Load,
            store: AttachmentStore::Store,
        };
        vec![
            GpuOperation::new(
                GpuCommand::RenderPass(
                    RenderPassOperation::begin(pass, description, vec![attachment]).unwrap(),
                ),
                [],
                [],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::Draw(
                    DrawOperation::new(
                        Arc::new(
                            PreparedDraw::new(
                                PipelineId::new(1),
                                pass,
                                PrimitiveTopology::Triangles,
                                Vec::new(),
                                Vec::new(),
                                None,
                            )
                            .unwrap(),
                        ),
                        DrawArguments::NonIndexed {
                            first_vertex: 0,
                            vertex_count: 3,
                            first_instance: 0,
                            instance_count: 1,
                        },
                    )
                    .unwrap(),
                ),
                [],
                [],
                CapabilityRequirements::none(),
            ),
            GpuOperation::new(
                GpuCommand::RenderPass(RenderPassOperation::end(pass)),
                [],
                [],
                CapabilityRequirements::none(),
            ),
        ]
    }

    #[test]
    fn compatible_maxwell_render_passes_share_one_explicit_boundary() {
        let pass = RenderPassId::new(1);
        let image = ImageId::new(1);
        let mut operations = render_pass_operations(pass, image);
        let mut current = Some(0);

        append_batchable_operations(
            &mut operations,
            &mut current,
            &render_pass_operations(pass, image),
        );

        assert_eq!(operations.len(), 4);
        assert!(matches!(
            operations[0].command(),
            GpuCommand::RenderPass(RenderPassOperation::Begin { .. })
        ));
        assert!(matches!(
            operations[3].command(),
            GpuCommand::RenderPass(RenderPassOperation::End { .. })
        ));
        assert_eq!(
            operations
                .iter()
                .filter(|operation| matches!(operation.command(), GpuCommand::Draw(_)))
                .count(),
            2
        );

        append_batchable_operations(
            &mut operations,
            &mut current,
            &render_pass_operations(RenderPassId::new(2), image),
        );
        assert_eq!(operations.len(), 7);
    }

    #[test]
    fn render_pass_batching_preserves_attachment_load_store_boundaries() {
        let pass = RenderPassId::new(1);
        let image = ImageId::new(1);
        let mut operations = render_pass_operations(pass, image);
        let mut current = Some(0);
        let mut incompatible = render_pass_operations(pass, image);
        let GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) =
            incompatible[0].command()
        else {
            unreachable!();
        };
        let mut attachment = attachments[0];
        attachment.load = AttachmentLoad::Clear(nixe_gpu::ClearValue::Color([0.0; 4]));
        incompatible[0] = GpuOperation::new(
            GpuCommand::RenderPass(
                RenderPassOperation::begin(
                    pass,
                    RenderPassDescription::new(vec![RenderPassAttachmentDescription {
                        kind: ImageKind::Color,
                        format: ImageFormat::Rgba8Unorm,
                        samples: SampleCount::One,
                    }])
                    .unwrap(),
                    vec![attachment],
                )
                .unwrap(),
            ),
            [],
            [],
            CapabilityRequirements::none(),
        );

        append_batchable_operations(&mut operations, &mut current, &incompatible);

        assert_eq!(operations.len(), 6);
    }

    #[test]
    fn empty_preflight_is_neutral_without_a_completion() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[],
            &address_space(),
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();
        assert!(plan.steps().is_empty());
        assert_eq!(plan.completion(), None);
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
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[bind, release],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
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

        execute_canonical_test_plan(plan, 0).unwrap();
        let mut bytes = [0_u8; 4];
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), 0xcafe_babe);
    }

    #[test]
    fn reserved_completion_requires_an_exact_signal_without_publication() {
        let (timeline, reservation) = reservation();
        let before = timeline.current_point();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        assert!(matches!(
            lower_test_pushbuffers(
                &mut channel,
                &[],
                &address_space(),
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &mut MaxwellLoweringCache::default(),
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
    fn driver_fence_get_reserves_two_increments_separately_from_user_signals() {
        for guest_increments in [0, 3] {
            let owner = TimelineOwnerId::new(7);
            let mut timeline = GuestTimeline::new(
                GuestSyncpointId::new(1),
                TimelineInstanceId::new(1),
                owner,
                GuestSyncpointValue::new(0),
            );
            let reservation = timeline.reserve(owner, guest_increments + 2).unwrap();
            let space = address_space();
            let mut cache = MaxwellLoweringCache::default();
            let mut planner = MaxwellSubmissionPlanner::new(
                &space,
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &mut cache,
            );
            planner.set_driver_completion_increments(2);
            // The driver contributes its increments only when execution completes.
            // Guest syncpoint methods must still match their independent count.
            let mut channel = MaxwellGpuChannel::new(
                MaxwellChannelId::new(1),
                MaxwellChannelOwner::new(1),
                SWITCH_1_GM20B_PROFILE,
            );
            let (mut methods, mut parameters) = planner.take_mme_scratch();
            let mut packets = vec![packet(
                0,
                0,
                &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
            )];
            for _ in 0..guest_increments {
                packets.push(packet(0, 0x02c8 / 4, &[1]));
            }
            for pushbuffer in &packets {
                for packet in pushbuffer.packets() {
                    crate::engines::stream_maxwell_engine_packet(
                        &mut channel,
                        FrontendSubmissionId::new(2),
                        packet,
                        None,
                        &mut methods,
                        &mut parameters,
                        &mut |event| planner.push_event(event),
                    )
                    .unwrap_or_else(|_| panic!("verified driver fence test packet must lower"));
                }
            }
            planner.recycle_mme_scratch(methods, parameters);
            let plan = planner.finish().unwrap();
            assert_eq!(plan.completion(), Some(reservation.point()));
            assert_eq!(reservation.point().value().get(), guest_increments + 2);
            assert_eq!(timeline.current_point().value().get(), 0);
            let completed = execute_canonical_test_plan(plan, 0).unwrap();
            assert_eq!(completed, Some(reservation.point()));
            assert_eq!(timeline.current_point().value().get(), 0);
        }
    }

    #[test]
    fn matching_syncpoint_is_retained_once_and_never_published() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        let increment = packet(0, 0x02c8 / 4, &[1]);
        let (timeline, reservation) = reservation();
        let before = timeline.current_point();
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[bind, increment.clone()],
            &address_space(),
            FrontendSubmissionId::new(2),
            Vec::new(),
            Some(&reservation),
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();
        assert_eq!(plan.completion(), Some(reservation.point()));
        assert!(plan.steps().is_empty());
        assert_eq!(timeline.current_point(), before);

        assert!(matches!(
            lower_test_pushbuffers(
                &mut channel,
                &[increment.clone(), increment],
                &address_space(),
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &mut MaxwellLoweringCache::default(),
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
        let increment = packet(0, 0x02c8 / 4, &[1]);
        let flush = packet(0, 0x1144 / 4, &[0]);

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
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[bind, increment.clone(), increment, flush],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            Some(&reservation),
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();

        assert_eq!(reservation.increments(), 2);
        assert_eq!(plan.completion(), Some(reservation.point()));
        assert_eq!(plan.steps().len(), 1);
        let completion = complete_backend_execution(plan);
        assert_eq!(completion, Some(reservation.point()));
        assert_eq!(timeline.current_point(), before);
    }

    #[test]
    fn compute_launch_consumes_staged_qmd_through_aliases_and_produces_dispatch() {
        // QMD 1.7 geometry from the captured upload, with a synthetic nonzero
        // program offset to exercise base+offset rather than just the base.
        let mut qmd = [0u32; 64];
        qmd[0x18 / 4] = 0x40;
        qmd[0x20 / 4] = 0x120;
        qmd[0x2c / 4] = 0x0400_0000;
        qmd[0x30 / 4] = 8;
        qmd[0x34 / 4] = 0x0001_0001;
        qmd[0x48 / 4] = 0x0020_0017;
        qmd[0x4c / 4] = 0x0001_0001;
        qmd[0x50 / 4] = 0x6000_0005;
        qmd[0xb8 / 4] = 1 << 24;
        for flags in [2, 3] {
            let mut address_space = address_space();
            let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
            let request = MaxwellMapRequest {
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
            };
            let address = address_space.map(request.clone()).unwrap().offset().get();
            let alias = address_space.map(request).unwrap().offset().get();
            assert_ne!(address, alias);
            allocation
                .write(
                    0x120,
                    &[0u64, 0xf0c8000002170000, 0xe30000000007000f, 0]
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut channel = MaxwellGpuChannel::new(
                MaxwellChannelId::new(1),
                MaxwellChannelOwner::new(1),
                SWITCH_1_GM20B_PROFILE,
            );
            let (timeline, _reservation) = reservation();
            let before = timeline.current_point();
            let plan = lower_test_pushbuffers(
                &mut channel,
                &[
                    packet(1, 0, &[SWITCH_1_GM20B_PROFILE.classes().compute().0]),
                    packet(1, 0x1608 / 4, &[(address >> 32) as u32, address as u32]),
                    packet(
                        1,
                        0x0180 / 4,
                        &[0x100, 1, (address >> 32) as u32, address as u32],
                    ),
                    packet(1, 0x01b0 / 4, &[0x11]),
                    non_incrementing_packet(1, 0x01b4 / 4, &qmd[..19]),
                    non_incrementing_packet(1, 0x01b4 / 4, &qmd[19..]),
                    // Overwrite a field in the same submission; the launch must
                    // consume the latest canonical overlay rather than raw RAM.
                    packet(
                        1,
                        0x0180 / 4,
                        &[4, 1, (address >> 32) as u32, address as u32 + 0x30],
                    ),
                    packet(1, 0x01b0 / 4, &[0x11]),
                    packet(1, 0x01b4 / 4, &[9]),
                    packet(1, 0x02b4 / 4, &[(alias >> 8) as u32]),
                    packet(1, 0x02bc / 4, &[flags]),
                ],
                &address_space,
                FrontendSubmissionId::new(2),
                Vec::new(),
                None,
                &mut MaxwellLoweringCache::default(),
            )
            .unwrap();
            assert!(plan.steps().iter().any(|step| matches!(step, MaxwellSubmissionExecutionStep::Gpu(work)
                if work.submission().operations().iter().any(|op| matches!(op.command(), GpuCommand::Dispatch(dispatch) if dispatch.workgroups == [9, 1, 1])))));
            let mut bytes = [0xff; 0x100];
            allocation.read(0, &mut bytes).unwrap();
            assert_eq!(bytes, [0; 0x100]);
            assert_eq!(timeline.current_point(), before);
        }
    }

    #[test]
    fn compute_inline_flush_modes_commit_between_backend_segments_without_extra_work() {
        inline_flush_modes_commit_between_backend_segments(
            SWITCH_1_GM20B_PROFILE.classes().compute().0,
            &[0x01, 0x11, 0x41, 0x51],
        );
    }

    #[test]
    fn three_d_inline_flush_modes_commit_between_backend_segments_without_extra_work() {
        inline_flush_modes_commit_between_backend_segments(
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
            &[0x01, 0x11, 0x41, 0x51, 0x1001, 0x1011, 0x1041, 0x1051],
        );
    }

    fn inline_flush_modes_commit_between_backend_segments(class: u32, launches: &[u32]) {
        let subchannel = u32::from(class == SWITCH_1_GM20B_PROFILE.classes().compute().0);
        // Match the observed 0x340-byte upload, including a packet boundary in
        // the payload. Data is synthetic; command and visibility paths are real.
        let data: Vec<u32> = (0..0x340 / 4).map(|index| 0xcafe_0000 | index).collect();
        let expected: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
        for &raw in launches {
            for with_backend in [false, true] {
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
                let mut packets = vec![packet(subchannel, 0, &[class])];
                if with_backend {
                    if subchannel != 1 {
                        packets.push(packet(
                            1,
                            0,
                            &[SWITCH_1_GM20B_PROFILE.classes().compute().0],
                        ));
                    }
                    packets.push(packet(1, 0x1698 / 4, &[0x1000]));
                }
                packets.extend([
                    packet(
                        subchannel,
                        0x0180 / 4,
                        &[0x340, 1, (address >> 32) as u32, address as u32],
                    ),
                    packet(subchannel, 0x01b0 / 4, &[raw]),
                    non_incrementing_packet(subchannel, 0x01b4 / 4, &data[..17]),
                    non_incrementing_packet(subchannel, 0x01b4 / 4, &data[17..]),
                ]);
                if with_backend {
                    packets.push(packet(1, 0x1698 / 4, &[0x1011]));
                }
                let plan = lower_test_pushbuffers(
                    &mut channel,
                    &packets,
                    &address_space,
                    FrontendSubmissionId::new(2),
                    Vec::new(),
                    None,
                    &mut MaxwellLoweringCache::default(),
                )
                .unwrap();
                if class == SWITCH_1_GM20B_PROFILE.classes().compute().0 {
                    assert_eq!(channel.compute().inline_to_memory().pending(), None);
                }
                assert_eq!(plan.requires_backend(), with_backend);
                assert_eq!(plan.completion(), None);
                assert_eq!(plan.steps().len(), 1 + if with_backend { 2 } else { 0 });
                let upload = &plan.steps()[usize::from(with_backend)];
                assert!(
                    matches!(upload, MaxwellSubmissionExecutionStep::InlineWrite { value, target, .. }
                    if *value == expected && target.size() == expected.len() as u64)
                );
                let mut bytes = vec![0; expected.len()];
                allocation.read(0, &mut bytes).unwrap();
                assert!(bytes.iter().all(|byte| *byte == 0));

                if with_backend {
                    let mut execution = MaxwellBackendExecution::new(plan);
                    let prefix = execution.next_segment(0).unwrap().unwrap();
                    assert_eq!(prefix.submission().operations().len(), 1);
                    assert!(execution.can_continue_after_submission());
                    assert!(!execution.can_prepare_following());
                    // Acceptance must first capture preceding read inputs.
                    allocation.read(0, &mut bytes).unwrap();
                    assert!(bytes.iter().all(|byte| *byte == 0));
                    execution.resume_segment();
                    let suffix = execution.next_segment(0).unwrap().unwrap();
                    assert_eq!(suffix.submission().operations().len(), 1);
                    assert!(execution.can_prepare_following());
                    // All payload bytes are coherent before the next GPU segment.
                    allocation.read(0, &mut bytes).unwrap();
                    assert_eq!(bytes, expected);
                    execution.resume_segment();
                    assert!(execution.next_segment(0).unwrap().is_none());
                    assert_eq!(execution.completion(), None);
                } else {
                    // FLUSH_ONLY does not force software-only uploads onto the GPU.
                    assert_eq!(execute_canonical_test_plan(plan, 0).unwrap(), None);
                }
                allocation.read(0, &mut bytes).unwrap();
                assert_eq!(bytes, expected);
                let mut tail = [0xff; 4];
                allocation.read(0x340, &mut tail).unwrap();
                assert_eq!(tail, [0; 4]);
            }
        }
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
        let flush = packet(6, 0x002c / 4, &[0x8000_0000]);
        let invalidate = packet(6, 0x002c / 4, &[0x7000_0000]);
        let bind_three_d = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind_three_d.packets()[0],
        )
        .unwrap();
        let texture_invalidate = packet(0, 0x1288 / 4, &[0]);
        let shader_invalidate = packet(0, 0x0da4 / 4, &[0x1011]);
        let compute_invalidate = packet(1, 0x021c / 4, &[0x1000]);
        let fragment_barriers = packet(0, 0x0de0 / 4, &[0]);
        let fragment_system_barrier = packet(0, 0x0de0 / 4, &[1]);
        let tiled_barrier = packet(0, 0x0f7c / 4, &[0]);
        let wait = packet(1, 0x0110 / 4, &[0]);

        let plan = lower_test_pushbuffers(
            &mut channel,
            &[
                data,
                flush,
                invalidate,
                texture_invalidate,
                shader_invalidate,
                compute_invalidate,
                fragment_barriers,
                fragment_system_barrier,
                tiled_barrier,
                wait,
            ],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
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
                MaxwellSubmissionExecutionStep::BackendOperation(compute),
                MaxwellSubmissionExecutionStep::BackendOperation(fragment),
                MaxwellSubmissionExecutionStep::BackendOperation(fragment_system),
                MaxwellSubmissionExecutionStep::BackendOperation(tiled),
            ] if *value == 0xfeed_beef_u32.to_le_bytes()
                && matches!(compute.command(), GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateShaderCaches {
                    instruction: false, global_data: false, constant: true,
                }))
                && matches!(fragment.command(), GpuCommand::CacheMaintenance(CacheMaintenanceOperation::FlushDirtyDeviceWrites))
                && matches!(fragment_system.command(), GpuCommand::CacheMaintenance(CacheMaintenanceOperation::FlushDirtyDeviceWrites))
                && matches!(tiled.command(), GpuCommand::CacheMaintenance(CacheMaintenanceOperation::FlushDirtyDeviceWrites))
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

        let completion = complete_backend_execution(plan);
        assert_eq!(completion, None);
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, 0xfeed_beef_u32.to_le_bytes());
    }

    #[test]
    fn host_semaphore_release_writes_payload_and_completion_timestamp() {
        for short in [false, true] {
            for with_backend in [false, true] {
                let mut address_space = address_space();
                let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
                allocation.write(0, &[0xcc; 16]).unwrap();
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
                let mut packets = Vec::new();
                if with_backend {
                    packets.push(packet(
                        0,
                        0,
                        &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
                    ));
                    packets.push(packet(0, 0x0f7c / 4, &[0]));
                }
                packets.push(packet(
                    6,
                    0x10 / 4,
                    &[
                        (address >> 32) as u32,
                        address as u32,
                        0xcafe_babe,
                        2 | if short { 1 << 24 } else { 0 },
                    ],
                ));
                let plan = lower_test_pushbuffers(
                    &mut channel,
                    &packets,
                    &address_space,
                    FrontendSubmissionId::new(2),
                    Vec::new(),
                    None,
                    &mut MaxwellLoweringCache::default(),
                )
                .unwrap();
                let timestamp = 0x1234_5678_9abc_def0;
                if with_backend {
                    let mut execution = MaxwellBackendExecution::new(plan);
                    assert!(execution.next_segment(1).unwrap().is_some());
                    let mut unchanged = [0; 16];
                    allocation.read(0, &mut unchanged).unwrap();
                    assert_eq!(unchanged, [0xcc; 16]);
                    execution.resume_segment();
                    assert!(execution.next_segment(timestamp).unwrap().is_none());
                } else {
                    execute_canonical_test_plan(plan, timestamp).unwrap();
                }
                let mut actual = [0; 16];
                allocation.read(0, &mut actual).unwrap();
                assert_eq!(&actual[..4], &0xcafe_babe_u32.to_le_bytes());
                if short {
                    assert_eq!(&actual[4..], &[0xcc; 12]);
                } else {
                    assert_eq!(&actual[4..8], &[0; 4]);
                    assert_eq!(&actual[8..], &timestamp.to_le_bytes());
                }
            }
        }
    }

    #[test]
    fn notification_is_published_after_prior_backend_work_with_timer_and_completion_status() {
        for with_backend in [false, true] {
            let mut address_space = address_space();
            let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
            allocation.write(0, &[0xcc; 16]).unwrap();
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
            let mut packets = Vec::new();
            packets.push(packet(
                0,
                0,
                &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
            ));
            if with_backend {
                packets.push(packet(0, 0x0f7c / 4, &[0]));
            }
            packets.push(packet(
                0,
                0x0104 / 4,
                &[(address >> 32) as u32, address as u32, 0],
            ));
            packets.push(packet(0, 0x0100 / 4, &[0]));
            let plan = lower_test_pushbuffers(
                &mut channel,
                &packets,
                &address_space,
                FrontendSubmissionId::new(2),
                Vec::new(),
                None,
                &mut MaxwellLoweringCache::default(),
            )
            .unwrap();
            let timestamp = 0x1234_5678_9abc_def0;
            if with_backend {
                let mut execution = MaxwellBackendExecution::new(plan);
                assert!(execution.next_segment(1).unwrap().is_some());
                let mut unchanged = [0; 16];
                allocation.read(0, &mut unchanged).unwrap();
                assert_eq!(unchanged, [0xcc; 16]);
                execution.resume_segment();
                assert!(execution.next_segment(timestamp).unwrap().is_none());
            } else {
                execute_canonical_test_plan(plan, timestamp).unwrap();
            }
            let mut actual = [0; 16];
            allocation.read(0, &mut actual).unwrap();
            assert_eq!(&actual[..8], &timestamp.to_le_bytes());
            assert_eq!(&actual[8..], &[0; 8]);
        }
    }

    #[test]
    fn host_wfi_splits_work_at_a_real_backend_completion_boundary() {
        for scope in [0, 1] {
            let frontend = FrontendSubmissionId::new(2);
            let mut channel = MaxwellGpuChannel::new(
                MaxwellChannelId::new(1),
                MaxwellChannelOwner::new(1),
                SWITCH_1_GM20B_PROFILE,
            );
            let plan = lower_test_pushbuffers(
                &mut channel,
                &[
                    packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]),
                    packet(0, 0x0f7c / 4, &[0]),
                    packet(0, 0x78 / 4, &[scope]),
                    packet(0, 0x0f74 / 4, &[0]),
                ],
                &address_space(),
                frontend,
                Vec::new(),
                None,
                &mut MaxwellLoweringCache::default(),
            )
            .unwrap();
            assert!(matches!(
                plan.steps(),
                [
                    MaxwellSubmissionExecutionStep::BackendOperation(_),
                    MaxwellSubmissionExecutionStep::WaitForIdle,
                    MaxwellSubmissionExecutionStep::BackendOperation(_),
                ]
            ));
            let mut execution = MaxwellBackendExecution::new(plan);
            let before = execution.next_segment(0).unwrap().unwrap();
            assert_eq!(before.submission().operations().len(), 1);
            assert!(!before.submission().is_final_segment());
            assert!(execution.awaiting_resume);
            assert!(!execution.can_continue_after_submission());
            execution.resume_segment();
            let after = execution.next_segment(0).unwrap().unwrap();
            assert_eq!(after.submission().operations().len(), 1);
            assert!(after.submission().is_final_segment());
            execution.resume_segment();
            assert!(execution.next_segment(0).unwrap().is_none());
        }
    }

    #[test]
    fn inline_updates_resume_after_acceptance_only_without_a_device_dependency() {
        struct NoReadback;
        impl nixe_memory::VisibilityCoordinator for NoReadback {
            fn cache_cpu_page(
                &self,
                _: nixe_memory::DeviceVisibilityRequest,
                _: &[u8],
            ) -> Result<(), nixe_memory::VisibilityCoordinatorError> {
                Ok(())
            }
            fn make_cpu_visible(
                &self,
                _: nixe_memory::CpuVisibilityRequest,
            ) -> Result<Box<[u8]>, nixe_memory::VisibilityCoordinatorError> {
                panic!("the continuation check must not request readback")
            }
        }
        for (device_owned, explicit_wfi) in [(false, false), (true, false), (false, true)] {
            let mut address_space = address_space();
            let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
            let backing = allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap();
            let mapping = address_space
                .map(MaxwellMapRequest {
                    allocation: MaxwellAllocationId::new(1),
                    backing: backing.clone(),
                    backing_offset: 0,
                    size: 0x2000,
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
            let mut packets = vec![
                packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]),
                packet(0, 0x0f7c / 4, &[0]),
            ];
            if explicit_wfi {
                packets.push(packet(0, 0x78 / 4, &[0]));
            }
            packets.extend([
                packet(
                    0,
                    0x2380 / 4,
                    &[0x2000, (address >> 32) as u32, address as u32, 0x1000],
                ),
                packet(0, 0x2390 / 4, &[0x1234_5678]),
                packet(0, 0x0f74 / 4, &[0]),
            ]);
            let plan = lower_test_pushbuffers(
                &mut channel,
                &packets,
                &address_space,
                FrontendSubmissionId::new(2),
                Vec::new(),
                None,
                &mut MaxwellLoweringCache::default(),
            )
            .unwrap();
            let mut execution = MaxwellBackendExecution::new(plan);
            assert!(execution.next_segment(0).unwrap().is_some());
            if device_owned {
                let target = backing.snapshot_subrange(0x1000, 0x1000).unwrap();
                let point = nixe_memory::DeviceVisibilityPoint::new(1);
                let declaration = nixe_memory::DeviceAccessDeclaration::write(
                    nixe_memory::NonCpuDeviceId::new(1),
                    point,
                    point,
                )
                .unwrap();
                let owner: Arc<dyn nixe_memory::VisibilityCoordinator> = Arc::new(NoReadback);
                nixe_memory::CanonicalBackingRange::prepare_resident_device_accesses(
                    [(&target, declaration)],
                    Arc::clone(&owner),
                )
                .unwrap();
                nixe_memory::CanonicalBackingRange::publish_device_writes(
                    [(&target, declaration)],
                    owner,
                )
                .unwrap();
            }
            assert_eq!(
                execution.can_continue_after_submission(),
                !device_owned && !explicit_wfi
            );
            if !device_owned {
                execution.resume_segment();
                assert!(
                    execution
                        .next_segment(0)
                        .unwrap()
                        .unwrap()
                        .submission()
                        .is_final_segment()
                );
                assert!(!execution.can_continue_after_submission());
                let mut bytes = [0; 4];
                allocation.read(0x1000, &mut bytes).unwrap();
                assert_eq!(u32::from_le_bytes(bytes), 0x1234_5678);
            }
        }
    }

    #[test]
    fn tiled_barrier_and_texture_invalidation_stay_in_one_gpu_submission() {
        let frontend = FrontendSubmissionId::new(2);
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[
                packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]),
                packet(0, 0x0f7c / 4, &[0]),
                packet(0, 0x0f74 / 4, &[0]),
                packet(0, 0x0f78 / 4, &[1]),
            ],
            &address_space(),
            frontend,
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();
        let mut execution = MaxwellBackendExecution::new(plan);
        let segment = execution.next_segment(0).unwrap().unwrap();
        assert!(segment.creations().is_empty());
        assert!(segment.invalidations().is_empty());
        assert_eq!(segment.submission().id(), frontend);
        assert!(
            matches!(segment.submission().operations(), [barrier, invalidate]
            if matches!(barrier.command(), GpuCommand::CacheMaintenance(CacheMaintenanceOperation::FlushDirtyDeviceWrites))
                && matches!(invalidate.command(), GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateTextureReadCaches)))
        );
        execution.resume_segment();
        assert!(execution.next_segment(0).unwrap().is_none());
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
        let flush = packet(0, 0x0f80 / 4, &[0]);
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[bind, flush],
            &address_space(),
            frontend,
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();

        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::BackendOperation(operation)]
                if matches!(operation.command(), GpuCommand::CacheMaintenance(
                    CacheMaintenanceOperation::FlushDirtyDeviceWrites
                ))
        ));

        let mut execution = MaxwellBackendExecution::new(plan);
        let segment = execution.next_segment(0).unwrap().unwrap();
        assert!(segment.creations().is_empty());
        assert!(segment.invalidations().is_empty());
        assert_eq!(segment.submission().id(), frontend);
        assert_eq!(segment.submission().operations().len(), 1);
        assert!(matches!(
            segment.submission().operations()[0].command(),
            GpuCommand::CacheMaintenance(CacheMaintenanceOperation::FlushDirtyDeviceWrites)
        ));
        execution.resume_segment();
        assert!(execution.next_segment(0).unwrap().is_none());
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
        let bind_three_d = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind_three_d.packets()[0],
        )
        .unwrap();
        let invalidate = packet(0, 0x1330 / 4, &[0]);
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[data, invalidate],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [
                MaxwellSubmissionExecutionStep::InlineWrite {
                    value,
                    target,
                    ..
                },
                MaxwellSubmissionExecutionStep::BackendOperation(operation),
            ] if *value == [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]
                && target.offset().get() == address
                && target.size() == 8
                && matches!(operation.command(), GpuCommand::CacheMaintenance(
                    CacheMaintenanceOperation::InvalidateSamplerCaches
                ))
        ));

        let mut bytes = [0xff; 8];
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0; 8]);
        complete_backend_execution(plan);
        allocation.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);
    }

    #[test]
    fn streamed_prefix_retains_a_real_final_operation_even_before_trailing_wfi() {
        let space = address_space();
        let mut cache = MaxwellLoweringCache::default();
        let frontend = FrontendSubmissionId::new(2);
        let mut planner =
            MaxwellSubmissionPlanner::new(&space, frontend, Vec::new(), None, &mut cache);
        for _ in 0..8 {
            planner
                .steps
                .push(MaxwellSubmissionExecutionStep::BackendOperation(
                    cache_maintenance_operation(CacheMaintenanceOperation::FlushDirtyDeviceWrites),
                ));
        }
        planner
            .steps
            .push(MaxwellSubmissionExecutionStep::WaitForIdle);
        let prefix = planner.take_ready_steps().unwrap();
        assert_eq!(prefix.len(), 7);
        assert!(planner.take_ready_steps().is_none());
        let tail = planner.finish().unwrap();
        assert!(matches!(
            tail.steps(),
            [
                MaxwellSubmissionExecutionStep::BackendOperation(_),
                MaxwellSubmissionExecutionStep::WaitForIdle
            ]
        ));
        let mut execution = MaxwellBackendExecution::begin(frontend, None);
        execution.append_steps(prefix);
        assert!(execution.next_segment(0).unwrap().is_none());
        assert!(!execution.is_finished());
        execution.append_steps(tail.into_steps().into_vec());
        execution.seal();
        let final_segment = execution.next_segment(0).unwrap().unwrap();
        assert_eq!(final_segment.submission().operations().len(), 8);
        assert!(final_segment.submission().is_final_segment());
        assert!(!execution.can_continue_after_submission());
        execution.resume_segment();
        assert!(execution.next_segment(0).unwrap().is_none());
        assert!(execution.is_finished());
    }

    #[test]
    fn streamed_backend_segments_preserve_reused_constant_buffer_versions() {
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

        let mut pushbuffers = Vec::new();
        for value in [0xff00_0000, 0x00ff_0000, 0x0000_ff00] {
            let load = packet(0, 0x238c / 4, &[0, value]);
            pushbuffers.push(load);
            let invalidate = packet(0, 0x1288 / 4, &[0]);
            pushbuffers.push(invalidate);
        }
        let plan = lower_test_pushbuffers(
            &mut channel,
            &pushbuffers,
            &address_space,
            frontend,
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();

        let mut observed = Vec::new();
        let mut execution = MaxwellBackendExecution::begin(frontend, plan.completion());
        assert!(execution.next_segment(0).unwrap().is_none());
        assert!(!execution.is_finished());
        let mut steps = plan.into_steps().into_vec().into_iter();
        for _ in 0..3 {
            // Each prefix contains a write and its consuming GPU operation.
            // Keep that last operation buffered until the next prefix proves
            // that it is non-final, or completion validation seals the stream.
            execution.append_steps(steps.by_ref().take(2).collect());
            while let Some(segment) = execution.next_segment(0).unwrap() {
                assert!(!segment.submission().is_final_segment());
                assert_eq!(segment.submission().segment().get(), observed.len() as u32);
                assert!(execution.can_continue_after_submission());
                let mut bytes = [0; 4];
                allocation.read(0, &mut bytes).unwrap();
                observed.push(u32::from_le_bytes(bytes));
                execution.resume_segment();
            }
            assert!(!execution.is_finished());
            assert!(!execution.can_prepare_following());
        }
        assert_eq!(observed, [0xff00_0000, 0x00ff_0000]);
        assert!(steps.next().is_none());
        execution.seal();
        let final_segment = execution.next_segment(0).unwrap().unwrap();
        assert!(final_segment.submission().is_final_segment());
        assert_eq!(final_segment.submission().segment().get(), 2);
        let mut bytes = [0; 4];
        allocation.read(0, &mut bytes).unwrap();
        observed.push(u32::from_le_bytes(bytes));
        assert!(execution.awaiting_completion());
        assert!(execution.can_prepare_following());
        execution.resume_segment();
        assert!(execution.next_segment(0).unwrap().is_none());
        assert!(execution.is_finished());
        assert_eq!(observed, [0xff00_0000, 0x00ff_0000, 0x0000_ff00]);
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
            packet(
                4,
                0x0240 / 4,
                &[
                    (source_address >> 32) as u32,
                    source_address as u32,
                    0x1234_5678,
                ],
            ),
            packet(4, 0x0708 / 4, &[0x0330_3210]),
            // The first 2D copy uses the reset slice selector without writing SET_DST_LAYER.
            packet(4, 0x070c / 4, &[0x1040, 256, 256, 1]),
            packet(4, 0x0720 / 4, &[0]),
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
        let launch = packet(4, 0x0300 / 4, &[0x68e]);
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[launch],
            &address_space,
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::MemoryCopy { operation, .. }, MaxwellSubmissionExecutionStep::SemaphoreRelease { payload: 0x1234_5678, short: true, .. }]
                if operation.source_address() == source_address
                    && operation.destination_address() == destination_address
                    && operation.source_range_size() == TEXTURE_SIZE as u64
                    && operation.destination_range_size() == TEXTURE_SIZE as u64
        ));

        execute_canonical_test_plan(plan, 0).unwrap();
        let mut completion = [0; 4];
        source_allocation.read(0, &mut completion).unwrap();
        assert_eq!(u32::from_le_bytes(completion), 0x1234_5678);
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
    fn two_d_point_copy_preserves_pixels_outside_a_cropped_tiled_rectangle() {
        let input = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let output = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let linear: Vec<_> = (0..0x1000).map(|i| (i % 251) as u8).collect();
        input.write(0, &linear).unwrap();
        output.write(0, &[0xaa; 0x1000]).unwrap();
        let mut space = address_space();
        let mut addresses = Vec::new();
        for (index, allocation) in [&input, &output].into_iter().enumerate() {
            addresses.push(
                space
                    .map(MaxwellMapRequest {
                        allocation: MaxwellAllocationId::new(index as u64 + 20),
                        backing: allocation
                            .backing_range(MemoryPermissions::READ_WRITE)
                            .unwrap(),
                        backing_offset: 0,
                        size: 0x1000,
                        allocation_alignment: 0x1000,
                        page_size: 0,
                        kind: 0,
                        cacheable: true,
                        permissions: MemoryPermissions::READ_WRITE,
                        fixed_offset: None,
                    })
                    .unwrap()
                    .offset()
                    .get(),
            );
        }
        let [src, dst] = addresses[..] else {
            unreachable!()
        };
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let frontend = FrontendSubmissionId::new(2);
        for (method, value) in [
            (0, SWITCH_1_GM20B_PROFILE.classes().two_d().0),
            (0x0290, 0),
            (0x02ac, 3),
            // Pitch width includes the crop; unused block registers stay unset.
            (0x0230, 0xcf),
            (0x0234, 1),
            (0x0244, 64),
            (0x0248, 18),
            (0x024c, 3),
            (0x0250, (src >> 32) as u32),
            (0x0254, src as u32),
            (0x0200, 0xcf),
            (0x0204, 0),
            (0x0208, 0x10),
            (0x020c, 1),
            (0x0210, 0),
            (0x0218, 32),
            (0x021c, 16),
            (0x0220, (dst >> 32) as u32),
            (0x0224, dst as u32),
            (0x02d4, 0),
            (0x088c, 0),
            (0x08b0, 4),
            (0x08b4, 7),
            (0x08b8, 16),
            (0x08bc, 3),
            (0x08c0, 0),
            (0x08c4, 1),
            (0x08c8, 0),
            (0x08cc, 1),
            (0x08d0, 0),
            (0x08d4, 2),
            (0x08d8, 0),
        ] {
            let setup = packet(3, method / 4, &[value]);
            dispatch_maxwell_engine_packet(&mut channel, frontend, &setup.packets()[0]).unwrap();
        }
        let launch = packet(3, 0x08dc / 4, &[0]);
        let plan = lower_test_pushbuffers(
            &mut channel,
            &[launch],
            &space,
            frontend,
            Vec::new(),
            None,
            &mut MaxwellLoweringCache::default(),
        )
        .unwrap();
        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::MemoryCopy { .. }]
        ));
        execute_canonical_test_plan(plan, 0).unwrap();
        let mut actual = [0; 0x1000];
        output.read(0, &mut actual).unwrap();
        let mut expected = [0xaa; 0x1000];
        // Independent GOB reference: 128-byte rows, two GOBs high per block.
        for y in 0..3 {
            for x in 0..64 {
                let byte_x = x + 16;
                let byte_y = y + 7;
                let offset = (byte_x / 64) * 1024
                    + (byte_y / 8) * 512
                    + ((byte_x % 64) / 32) * 256
                    + ((byte_y % 8) / 2) * 64
                    + ((byte_x % 32) / 16) * 32
                    + (byte_y % 2) * 16
                    + byte_x % 16;
                expected[offset] = linear[y * 64 + x + 8];
            }
        }
        assert_eq!(actual, expected);
        // Fractional and scaled point blits must not become a byte copy.
        for (method, bad, good) in [(0x08d0, 1, 0), (0x08c4, 2, 1), (0x08b8, 17, 16)] {
            let setup = packet(3, method / 4, &[bad]);
            dispatch_maxwell_engine_packet(&mut channel, frontend, &setup.packets()[0]).unwrap();
            let launch = packet(3, 0x08dc / 4, &[0]);
            assert!(
                dispatch_maxwell_engine_packet(&mut channel, frontend, &launch.packets()[0])
                    .is_err()
            );
            let restore = packet(3, method / 4, &[good]);
            dispatch_maxwell_engine_packet(&mut channel, frontend, &restore.packets()[0]).unwrap();
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
        use crate::MaxwellDmaCopyRegisterName;
        for (offset, value, name) in [
            (
                0x240,
                0xab,
                MaxwellDmaCopyRegisterName::SemaphoreAddressUpper,
            ),
            (
                0x244,
                0x1234_0000,
                MaxwellDmaCopyRegisterName::SemaphoreAddressLower,
            ),
            (
                0x248,
                0xdead_beef,
                MaxwellDmaCopyRegisterName::SemaphorePayload,
            ),
            (0x25c, 1, MaxwellDmaCopyRegisterName::RenderEnableControl),
            (0x260, 2, MaxwellDmaCopyRegisterName::SourcePhysicalTarget),
            (
                0x264,
                1,
                MaxwellDmaCopyRegisterName::DestinationPhysicalTarget,
            ),
        ] {
            let command = packet(4, offset / 4, &[value]);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(2),
                &command.packets()[0],
            )
            .unwrap();
            assert!(dispatch.ordered_operations().is_empty());
            assert_eq!(channel.dma_copy().register(name).raw(), Some(value));
            assert!(channel.dma_copy().register(name).source().is_some());
        }
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
