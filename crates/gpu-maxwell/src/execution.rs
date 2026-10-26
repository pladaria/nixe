//! Consume engine effects in command order without an intermediate execution plan.
mod lowerer;
use crate::engines::{MaxwellEngineEvent, PendingEngineOperation};
use crate::{
    MaxwellComputeSynchronizationPlan, MaxwellGpuAccessError, MaxwellGpuMemory,
    MaxwellHostSynchronizationKind, MaxwellLoweringError, MaxwellMemoryCopyError,
    MaxwellMemoryCopyOperation, MaxwellMethodSource, MaxwellResolvedRange,
    MaxwellThreeDResourceError, MaxwellThreeDSynchronizationError,
    MaxwellThreeDSynchronizationPlan, lower_maxwell_compute_synchronization,
    lower_maxwell_three_d_synchronization,
};
pub use lowerer::MaxwellFrontend;
use nixe_gpu::native::{NativeCommand, NativeWork};
use nixe_gpu::{
    CacheMaintenanceOperation, FrontendSubmissionId, GuestTimelinePoint, ReservedTimelinePoint,
};
use nixe_memory::{AccessOperation, AccessResult, AllocationRange, MemoryError, MemoryPermissions};
use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};
/// Failure while executing ordered canonical-memory commands.
#[derive(Clone, Debug)]
pub enum MaxwellCanonicalExecutionError {
    StaleInlineTarget {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    InlineWrite {
        source: MaxwellMethodSource,
        error: Arc<MemoryError>,
    },
    MemoryCopyTransform {
        source: MaxwellMethodSource,
        error: MaxwellMemoryCopyError,
    },
    MemoryCopyTransaction {
        source: MaxwellMethodSource,
        error: Arc<MemoryError>,
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
                write!(formatter, "inline upload failed: {source}: {error}")
            }
            Self::MemoryCopyTransform { source, error } => {
                write!(
                    formatter,
                    "memory copy transformation failed: {source}: {error}"
                )
            }
            Self::MemoryCopyTransaction { source, error } => {
                write!(formatter, "memory copy access failed: {source}: {error}")
            }
        }
    }
}

impl std::error::Error for MaxwellCanonicalExecutionError {}

/// First failure while consuming an ordered submission; prior effects remain applied.
#[derive(Clone, Debug)]
pub enum MaxwellSubmissionExecutionError {
    AtMethod {
        source: MaxwellMethodSource,
        cause: Box<Self>,
    },
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
    Lowering(MaxwellLoweringError),
    Native(Box<str>),
    Canonical(Box<MaxwellCanonicalExecutionError>),
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
            Self::AtMethod { source, cause } => write!(formatter, "{source}: {cause}"),
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

            Self::Lowering(error) => Display::fmt(error, formatter),
            Self::Native(error) => Display::fmt(error, formatter),
            Self::Canonical(error) => Display::fmt(error, formatter),
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

/// The command owner consumes native effects and memory in the same ordered domain.
/// Implemented directly by the GPU owner, which also services control/wakeup work.
pub trait MaxwellCommandHost {
    fn emit(
        &mut self,
        work: NativeWork,
    ) -> impl std::future::Future<Output = Result<(), MaxwellSubmissionExecutionError>>;
    fn access(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<AccessResult, Arc<MemoryError>>;
    /// # Safety
    /// The caller holds the original mappings through this synchronous update.
    unsafe fn memory_written(
        &mut self,
        machine: nixe_memory::MachineId,
        extents: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), Arc<MemoryError>>;
    fn control(&mut self) -> Result<(), MaxwellSubmissionExecutionError>;
    fn timestamp(&self) -> u64;
}
pub(crate) struct MaxwellCommandEffects<'a, L: MaxwellFrontend, H: MaxwellCommandHost> {
    pub(crate) address_space: &'a MaxwellGpuMemory,
    pub(crate) frontend: FrontendSubmissionId,
    completion: Option<&'a ReservedTimelinePoint>,
    cache: &'a std::cell::RefCell<L>,
    pub(crate) host: &'a mut H,
    prior_work_pending: bool,
    completion_signal_count: u32,
    driver_completion_increments: u32,
    #[cfg(feature = "performance-counters")]
    pub(crate) consumed: std::time::Duration,
}
impl<'a, L: MaxwellFrontend, H: MaxwellCommandHost> MaxwellCommandEffects<'a, L, H> {
    pub(crate) fn new(
        address_space: &'a MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        completion: Option<&'a ReservedTimelinePoint>,
        cache: &'a std::cell::RefCell<L>,
        host: &'a mut H,
    ) -> Self {
        Self {
            address_space,
            frontend,
            completion,
            cache,
            host,
            prior_work_pending: false,
            completion_signal_count: 0,
            driver_completion_increments: 0,
            #[cfg(feature = "performance-counters")]
            consumed: std::time::Duration::ZERO,
        }
    }
    async fn emit(&mut self, work: NativeWork) -> Result<(), MaxwellSubmissionExecutionError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        let result = self.host.emit(work).await;
        #[cfg(feature = "performance-counters")]
        {
            self.consumed += started.elapsed();
        }
        result
    }
    pub(crate) fn control(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        let result = self.host.control();
        #[cfg(feature = "performance-counters")]
        {
            self.consumed += started.elapsed();
        }
        result
    }
    async fn command(
        &mut self,
        command: NativeCommand,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.emit(NativeWork {
            frontend: self.frontend,
            command,
        })
        .await
    }
    pub(crate) async fn push_event(
        &mut self,
        operation: MaxwellEngineEvent<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = match &operation.operation {
            PendingEngineOperation::Notification { source, .. } => *source,
            PendingEngineOperation::TwoDBlit(request) => request.source(),
            PendingEngineOperation::TwoDSolid(request) => request.source,
            PendingEngineOperation::HostSynchronization(request) => request.source(),
            PendingEngineOperation::ComputeInlineToMemory(request) => request.source(),
            PendingEngineOperation::InlineToMemory(request) => request.source(),
            PendingEngineOperation::MemoryCopy(request) => request.source(),
            PendingEngineOperation::ComputeSynchronization(request) => request.source(),
            PendingEngineOperation::ComputeLaunch(request) => request.source(),
            PendingEngineOperation::ThreeDInlineConstantBuffer(request) => request.source(),
            PendingEngineOperation::ThreeD(request) => request.source(),
            PendingEngineOperation::ThreeDRenderCondition(source) => *source,
            PendingEngineOperation::ThreeDSynchronization(request) => request.source(),
        };
        self.cache.borrow_mut().bind_channel(source.channel());
        let result = self.consume_event(operation).await;
        result.map_err(|cause| MaxwellSubmissionExecutionError::AtMethod {
            source,
            cause: Box::new(cause),
        })
    }
    async fn consume_event(
        &mut self,
        operation: MaxwellEngineEvent<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let MaxwellEngineEvent {
            operation,
            three_d,
            compute,
            inline_words,
            instances,
        } = operation;
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
                self.report(source, &target, 0, false, true, true).await?;
                self.prior_work_pending = false;
            }
            PendingEngineOperation::ComputeLaunch(launch) => {
                let resolved = crate::engines::resolve_compute_launch(
                    &launch,
                    compute.expect("compute launch has live state"),
                    self.address_space,
                    &mut |ranges, operation| self.host.access(ranges, operation),
                )
                .map_err(MaxwellSubmissionExecutionError::ComputeLaunch)?;
                let work = self.cache.borrow_mut().lower_launch(
                    &resolved,
                    self.address_space,
                    self.frontend,
                    &mut |ranges, operation| self.host.access(ranges, operation),
                )?;
                self.emit(work).await?;
                self.prior_work_pending = true;
            }
            PendingEngineOperation::TwoDSolid(request) => {
                let [x0, y0, x1, y1] = request.rectangle;
                if x0 < x1 && y0 < y1 {
                    let work = self.cache.borrow_mut().lower_solid(
                        &request,
                        self.address_space,
                        self.frontend,
                        &mut |ranges, operation| self.host.access(ranges, operation),
                    )?;
                    self.emit(work).await?;
                    self.prior_work_pending = true;
                }
            }
            PendingEngineOperation::TwoDBlit(request) => {
                let work = self.cache.borrow_mut().lower_blit(
                    &request,
                    self.address_space,
                    self.frontend,
                    &mut |ranges, operation| self.host.access(ranges, operation),
                )?;
                self.emit(work).await?;
                self.prior_work_pending = true;
            }
            PendingEngineOperation::HostSynchronization(operation) => {
                let command = match operation.operation() {
                    MaxwellHostSynchronizationKind::SemaphoreRelease {
                        address,
                        payload,
                        short,
                        wait,
                    } => {
                        let source = operation.source();
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
                        self.report(source, &target, payload, short, true, false)
                            .await?;
                        if wait {
                            self.prior_work_pending = false;
                        }
                        return Ok(());
                    }
                    MaxwellHostSynchronizationKind::WaitForIdle { .. } => {
                        // All engine work on this channel shares the ordered
                        // backend stream. Both scopes therefore drain its prior
                        // segment; they never complete another channel's work.
                        self.command(NativeCommand::WaitForIdle).await?;
                        self.prior_work_pending = false;
                        return Ok(());
                    }
                    MaxwellHostSynchronizationKind::L2SysmemInvalidate { .. } => {
                        CacheMaintenanceOperation::InvalidateDeviceReadCaches
                    }
                    MaxwellHostSynchronizationKind::L2FlushDirty { .. } => {
                        CacheMaintenanceOperation::FlushDirtyDeviceWrites
                    }
                };
                self.command(NativeCommand::Ordering(command)).await?;
            }
            PendingEngineOperation::ComputeInlineToMemory(upload) => {
                // Both FLUSH_DISABLE and FLUSH_ONLY use coherent canonical writes.
                // Backend execution captures preceding overlapping accesses before
                // these writes and commits them before the following operation;
                // write-only initialization commits before returning completion.
                // This also satisfies enabled SYSMEMBAR without an extra host wait
                // or per-upload flush command. Neither mode releases a semaphore.
                // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L129-L165
                if let Some(words) = inline_words {
                    self.push_inline_words(
                        upload.address().get(),
                        upload.offset(),
                        words,
                        words.len() * 4,
                        upload.source(),
                    )?;
                } else {
                    self.push_inline_write(
                        upload.address().get(),
                        upload.offset(),
                        upload.value(),
                        upload.source(),
                    )?;
                }
            }
            PendingEngineOperation::InlineToMemory(upload) => {
                if let Some(words) = inline_words {
                    self.push_inline_words(
                        upload.address().get(),
                        upload.offset(),
                        words,
                        (words.len() * 4).min(upload.remaining_bytes() as usize),
                        upload.source(),
                    )?;
                    return Ok(());
                }
                self.write(
                    upload.address().get(),
                    upload.offset(),
                    &upload.value().to_le_bytes()[..upload.remaining_bytes().min(4) as usize],
                    upload.source(),
                )?;
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
                self.copy(operation, &source, &destination).await?;
                if let Some((address, payload)) = operation.semaphore_release {
                    let target =
                        resolve_inline_target(self.address_space, address, 0, operation.source())?;
                    self.report(operation.source(), &target, payload, true, true, false)
                        .await?;
                }
                self.prior_work_pending = true;
            }
            PendingEngineOperation::ComputeSynchronization(operation) => {
                let plan =
                    lower_maxwell_compute_synchronization(operation, self.prior_work_pending);
                match plan {
                    MaxwellComputeSynchronizationPlan::WaitForIdle { .. } => {
                        self.command(NativeCommand::WaitForIdle).await?;
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
                        self.command(NativeCommand::Ordering(
                            CacheMaintenanceOperation::InvalidateShaderCaches {
                                instruction: caches.instruction(),
                                global_data: caches.global_data(),
                                constant: caches.constant(),
                            },
                        ))
                        .await?;
                    }
                }
            }
            PendingEngineOperation::ThreeDInlineConstantBuffer(upload) => {
                if let Some(words) = inline_words {
                    self.push_inline_words(
                        upload.address().get(),
                        upload.offset(),
                        words,
                        words.len() * 4,
                        upload.source(),
                    )?;
                } else {
                    self.push_inline_write(
                        upload.address().get(),
                        upload.offset(),
                        upload.value(),
                        upload.source(),
                    )?;
                }
            }
            PendingEngineOperation::ThreeDRenderCondition(_) => {
                self.cache.borrow_mut().update_render_condition(
                    three_d,
                    self.address_space,
                    &mut |ranges, operation| self.host.access(ranges, operation),
                )?;
            }
            PendingEngineOperation::ThreeD(trigger) => {
                if !self.cache.borrow().render_enabled() {
                    return Ok(());
                }
                let work = self.cache.borrow_mut().lower_three_d(
                    three_d,
                    trigger,
                    self.address_space,
                    self.frontend,
                    instances,
                    &mut |ranges, operation| self.host.access(ranges, operation),
                )?;
                self.emit(work).await?;
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
                        | MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(_)
                        | MaxwellThreeDSynchronizationPlan::IncrementSyncpoint { .. }
                );
                if let MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(report) = plan {
                    let short = report.structure_size()
                        == crate::MaxwellThreeDReportSemaphoreStructureSize::OneWord;
                    let target = self
                        .address_space
                        .address(report.address().get())
                        .map_err(MaxwellGpuAccessError::Address)
                        .and_then(|address| {
                            self.address_space.resolve_range(
                                address,
                                if short { 4 } else { 16 },
                                MemoryPermissions::WRITE,
                            )
                        })
                        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress {
                            source: report.source(),
                            error,
                        })?;
                    self.report(
                        report.source(),
                        &target,
                        report.payload(),
                        short,
                        report.is_fence(),
                        false,
                    )
                    .await?;
                } else if matches!(plan, MaxwellThreeDSynchronizationPlan::WaitForIdle { .. }) {
                    self.command(NativeCommand::WaitForIdle).await?;
                } else if let Some(operation) = three_d_synchronization_operation(plan) {
                    self.command(NativeCommand::Ordering(operation)).await?;
                }
                if drains_prior_work {
                    self.prior_work_pending = false;
                }
            }
        }
        Ok(())
    }

    fn push_inline_words(
        &mut self,
        base: u64,
        offset: u32,
        words: crate::MaxwellMethodSpan<'_>,
        byte_count: usize,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let address = self.write_address(base, offset, source)?;
        self.address_space
            .write_words(address, words, byte_count, &mut |extents| {
                // The upload holds its mapping guard through this callback.
                unsafe {
                    self.host
                        .memory_written(self.address_space.machine().identity(), extents)
                }
            })
            .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })
    }

    fn push_inline_write(
        &mut self,
        base: u64,
        offset: u32,
        value: u32,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.write(base, offset, &value.to_le_bytes(), source)
    }
    fn write(
        &mut self,
        base: u64,
        offset: u32,
        bytes: &[u8],
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let address = self.write_address(base, offset, source)?;
        self.address_space
            .write_from(address, bytes, &mut |extents| {
                // The upload holds its mapping guard through this callback.
                unsafe {
                    self.host
                        .memory_written(self.address_space.machine().identity(), extents)
                }
            })
            .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })
    }

    fn write_address(
        &self,
        base: u64,
        offset: u32,
        source: MaxwellMethodSource,
    ) -> Result<nixe_gpu::GpuVirtualAddress, MaxwellSubmissionExecutionError> {
        let address = base.checked_add(u64::from(offset)).ok_or(
            MaxwellSubmissionExecutionError::InlineAddress {
                source,
                error: MaxwellGpuAccessError::ArithmeticOverflow,
            },
        )?;
        let address = self.address_space.address(address).map_err(|error| {
            MaxwellSubmissionExecutionError::InlineAddress {
                source,
                error: MaxwellGpuAccessError::Address(error),
            }
        })?;
        Ok(address)
    }

    async fn report(
        &mut self,
        source: MaxwellMethodSource,
        target: &MaxwellResolvedRange,
        payload: u32,
        short: bool,
        fence: bool,
        notification: bool,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let mut bytes = [0u8; 16];
        if notification {
            bytes[..8].copy_from_slice(&self.host.timestamp().to_le_bytes());
        } else {
            bytes[..4].copy_from_slice(&payload.to_le_bytes());
            if !short {
                bytes[8..].copy_from_slice(&self.host.timestamp().to_le_bytes());
            }
        }
        let bytes = &bytes[..if short { 4 } else { 16 }];
        let target = physical_subrange(target, 0, bytes.len()).map_err(|error| {
            canonical(MaxwellCanonicalExecutionError::InlineWrite { source, error })
        })?;
        self.command(NativeCommand::ReportMemory {
            target,
            offset: 0,
            bytes: Arc::from(bytes),
            fence,
        })
        .await
    }
    async fn copy(
        &mut self,
        operation: MaxwellMemoryCopyOperation,
        source: &MaxwellResolvedRange,
        destination: &MaxwellResolvedRange,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        // A RAM-current tiled image uses the CPU swizzle. Native producers
        // retain GPU execution instead of forcing a readback for conversion.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/texture_cache.h
        let mut ram_layout = operation.has_block_layout();
        if ram_layout && !operation.has_remap() {
            for (layout, range) in [
                (operation.source_layout, source),
                (operation.destination_layout, destination),
            ] {
                if !matches!(
                    layout,
                    crate::engines::MaxwellMemoryCopyLayout::BlockLinear { .. }
                ) {
                    continue;
                }
                let ranges =
                    physical_subrange(range, 0, range.size() as usize).map_err(|error| {
                        canonical(MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                            source: operation.source(),
                            error,
                        })
                    })?;
                ram_layout = !ranges[0]
                    .allocation()
                    .machine()
                    .device_modified(&ranges)
                    .map_err(|error| {
                        canonical(MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                            source: operation.source(),
                            error: Arc::new(error),
                        })
                    })?;
                if !ram_layout {
                    break;
                }
            }
        }
        if operation.has_remap() || ram_layout {
            return stage_memory_copy(operation, source, destination, &mut |ranges, operation| {
                self.host.access(ranges, operation)
            })
            .map_err(canonical);
        }
        let physical = |range: &MaxwellResolvedRange| {
            let mappings = range
                .segments()
                .iter()
                .map(|segment| {
                    segment
                        .mapping()
                        .physical()
                        .slice(segment.backing_offset(), segment.size())
                })
                .collect::<Result<Vec<_>, _>>();
            mappings
                .and_then(|mappings| nixe_gpu::physical::PhysicalMapping::join(&mappings))
                .map_err(|error| {
                    canonical(MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                        source: operation.source(),
                        error: Arc::new(error),
                    })
                })
        };
        let mut regions: Vec<nixe_gpu::native::BufferCopyRegion> = Vec::new();
        for run in operation.byte_regions() {
            let (source, destination, bytes) = run.map_err(|error| {
                canonical(MaxwellCanonicalExecutionError::MemoryCopyTransform {
                    source: operation.source(),
                    error,
                })
            })?;
            if let Some(last) = regions.last_mut()
                && last.source + last.bytes == source as u64
                && last.destination + last.bytes == destination as u64
            {
                last.bytes += bytes as u64;
            } else {
                regions.push(nixe_gpu::native::BufferCopyRegion {
                    source: source as u64,
                    destination: destination as u64,
                    bytes: bytes as u64,
                });
            }
        }
        self.command(NativeCommand::CopyBuffer {
            source: physical(source)?,
            destination: physical(destination)?,
            regions: regions.into_boxed_slice(),
        })
        .await
    }
    pub(crate) fn take_mme_scratch(&mut self) -> Vec<u32> {
        self.cache.borrow_mut().take_mme_scratch()
    }

    pub(crate) fn set_driver_completion_increments(&mut self, increments: u32) {
        self.driver_completion_increments = increments;
    }

    pub(crate) fn recycle_mme_scratch(&mut self, parameters: Vec<u32>) {
        self.cache.borrow_mut().recycle_mme_scratch(parameters);
    }

    pub(crate) fn finish(self) -> Result<(), MaxwellSubmissionExecutionError> {
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

        Ok(())
    }
}

fn canonical(error: MaxwellCanonicalExecutionError) -> MaxwellSubmissionExecutionError {
    MaxwellSubmissionExecutionError::Canonical(Box::new(error))
}
fn three_d_synchronization_operation(
    plan: MaxwellThreeDSynchronizationPlan,
) -> Option<CacheMaintenanceOperation> {
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
        | MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(_) => return None,
    };
    Some(maintenance)
}

fn stage_memory_copy(
    operation: MaxwellMemoryCopyOperation,
    source: &MaxwellResolvedRange,
    destination: &MaxwellResolvedRange,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<(), MaxwellCanonicalExecutionError> {
    // Snapshot the complete source before any overlapping destination changes.
    let ranges = physical_subrange(source, 0, source.size() as usize).map_err(|error| {
        MaxwellCanonicalExecutionError::MemoryCopyTransaction {
            source: operation.source(),
            error,
        }
    })?;
    let AccessResult::Read(source_bytes) =
        access(ranges, AccessOperation::Read).map_err(|error| {
            MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                source: operation.source(),
                error,
            }
        })?
    else {
        unreachable!("read operation returns bytes")
    };
    if !operation.has_remap() {
        // The CPU swizzle preserves allocation padding in place, as the cached
        // read/write span used by the DMA engine does on the original path.
        // One destination access avoids sorting and publishing 16-byte pieces.
        let ranges =
            physical_subrange(destination, 0, destination.size() as usize).map_err(|error| {
                MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                    source: operation.source(),
                    error,
                }
            })?;
        let AccessResult::Read(mut bytes) =
            access(ranges.clone(), AccessOperation::Read).map_err(|error| {
                MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                    source: operation.source(),
                    error,
                }
            })?
        else {
            unreachable!("read operation returns bytes")
        };
        operation
            .copy_bytes(&source_bytes, &mut bytes)
            .map_err(
                |error| MaxwellCanonicalExecutionError::MemoryCopyTransform {
                    source: operation.source(),
                    error,
                },
            )?;
        let written = access(ranges, AccessOperation::Write(bytes)).map_err(|error| {
            MaxwellCanonicalExecutionError::MemoryCopyTransaction {
                source: operation.source(),
                error,
            }
        })?;
        assert!(matches!(written, AccessResult::Written));
        return Ok(());
    }
    let mut bytes = memory_copy_buffer(operation.destination_range_size(), operation.source())?;
    operation
        .copy_bytes(&source_bytes, &mut bytes)
        .map_err(
            |error| MaxwellCanonicalExecutionError::MemoryCopyTransform {
                source: operation.source(),
                error,
            },
        )?;
    for span in operation.written_spans().map_err(|error| {
        MaxwellCanonicalExecutionError::MemoryCopyTransform {
            source: operation.source(),
            error,
        }
    })? {
        write_memory_copy_subrange(
            destination,
            span.start,
            &bytes[span],
            operation.source(),
            access,
        )?;
    }
    Ok(())
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

fn physical_subrange(
    range: &MaxwellResolvedRange,
    offset: usize,
    size: usize,
) -> Result<Vec<AllocationRange>, Arc<MemoryError>> {
    let mut ranges = Vec::new();
    for (segment, backing_offset, indices) in memory_copy_segments(range, offset, size) {
        ranges.extend(
            segment
                .mapping()
                .physical()
                .pin(backing_offset, indices.len() as u64)
                .map_err(Arc::new)?,
        );
    }
    Ok(ranges)
}

fn write_memory_copy_subrange(
    range: &MaxwellResolvedRange,
    offset: usize,
    bytes: &[u8],
    source: MaxwellMethodSource,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<(), MaxwellCanonicalExecutionError> {
    let ranges = physical_subrange(range, offset, bytes.len())
        .map_err(|error| MaxwellCanonicalExecutionError::MemoryCopyTransaction { source, error })?;
    let result = access(ranges, AccessOperation::Write(bytes.into()))
        .map_err(|error| MaxwellCanonicalExecutionError::MemoryCopyTransaction { source, error })?;
    assert!(matches!(result, AccessResult::Written));
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

fn resolve_inline_target(
    address_space: &MaxwellGpuMemory,
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
