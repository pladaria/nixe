//! Consume engine effects in command order without an intermediate execution plan.
mod lowerer;
mod methods;
mod unavailable;
use crate::{
    MaxwellComputeSynchronizationPlan, MaxwellGpuAccessError, MaxwellGpuMemory,
    MaxwellHostSynchronizationKind, MaxwellLoweringError, MaxwellMemoryCopyError,
    MaxwellMemoryCopyOperation, MaxwellMethodSource, MaxwellResolvedRange,
    MaxwellThreeDResourceError, MaxwellThreeDSynchronizationError,
    MaxwellThreeDSynchronizationPlan, lower_maxwell_compute_synchronization,
    lower_maxwell_three_d_synchronization,
};
pub use lowerer::MaxwellFrontend;
use nixe_gpu::native::NativeNv12Conversion;
use nixe_gpu::{
    CacheMaintenanceOperation, FrontendSubmissionId, GuestTimelinePoint, ReservedTimelinePoint,
};
use nixe_memory::{AllocationRange, MemoryError};
use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};
pub use unavailable::MissingRenderer;
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
pub trait MaxwellCommandHost: nixe_gpu::OwnerMemory + Sized {
    fn signal_syncpoint(&mut self) -> Result<(), MaxwellSubmissionExecutionError>;
    fn report_memory(
        &mut self,
        frontend: FrontendSubmissionId,
        target: Vec<AllocationRange>,
        bytes: Arc<[u8]>,
        fence: bool,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn order_device(
        &mut self,
        frontend: FrontendSubmissionId,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn cache_maintenance(
        &mut self,
        frontend: FrontendSubmissionId,
        operation: CacheMaintenanceOperation,
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    /// DispatchCalls checks shutdown once per GPFIFO entry. This does not poll
    /// the owner inbox or advance CPU coherence requests.
    fn is_stopped(&self) -> bool;
    /// PrepareDraw services CPU requests after consuming the complete draw.
    /// The GPU owner also services them at its existing inbox boundary.
    fn tick_work(&mut self) -> Result<(), MaxwellSubmissionExecutionError>;
    fn timestamp(&self) -> u64;
}
/// The current channel executes directly against its graphics and memory owner.
/// Methods consume effects directly without a continuation or command conversion queue.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
pub(crate) struct ChannelExecution<'a, L: MaxwellFrontend, H: MaxwellCommandHost> {
    pub(crate) address_space: &'a std::cell::RefCell<MaxwellGpuMemory>,
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
impl<'a, L: MaxwellFrontend, H: MaxwellCommandHost> ChannelExecution<'a, L, H> {
    pub(crate) fn new(
        address_space: &'a std::cell::RefCell<MaxwellGpuMemory>,
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
    // RasterizerVulkan::PrepareDraw's TickWork boundary, after resources and
    // native recording have been consumed and their cache borrows released.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L235-L259
    pub(crate) fn tick_work(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        let result = self.host.tick_work();
        #[cfg(feature = "performance-counters")]
        {
            self.consumed += started.elapsed();
        }
        result
    }
    fn push_inline_words(
        &mut self,
        base: u64,
        offset: u32,
        words: crate::MaxwellDmaArguments<'_>,
        byte_count: usize,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let address = self.write_address(base, offset, source)?;
        self.address_space
            .borrow()
            .write_inline_words(address, words, byte_count, self.host)
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
            .borrow()
            .write_inline(address, bytes, self.host)
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
        let address = self
            .address_space
            .borrow()
            .address(address)
            .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress {
                source,
                error: MaxwellGpuAccessError::Address(error),
            })?;
        Ok(address)
    }

    fn report(
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
        self.host
            .report_memory(self.frontend, target, Arc::from(bytes), fence)
    }
    fn copy(
        &mut self,
        operation: MaxwellMemoryCopyOperation,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        if self.cache.borrow_mut().accelerate_dma(
            &operation,
            &self.address_space.borrow(),
            self.frontend,
        )? {
            return Ok(());
        }
        // MaxwellDMA falls back to SafeReadCachedWrite. The memory manager
        // owns the GPU translation and cached invalidation accumulator.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_dma.cpp
        let mut space = self.address_space.borrow_mut();
        let address_error = |error| MaxwellSubmissionExecutionError::MemoryCopyAddress {
            source: operation.source(),
            error,
        };
        let src = space
            .address(operation.source_address())
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(address_error)?;
        let dst = space
            .address(operation.destination_address())
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(address_error)?;
        let mut source = memory_copy_buffer(operation.source_range_size(), operation.source())
            .map_err(canonical)?;
        space
            .read_into(src, &mut source, self.host)
            .map_err(address_error)?;
        let mut bytes = memory_copy_buffer(operation.destination_range_size(), operation.source())
            .map_err(canonical)?;
        // Padding and NoWrite components retain their previous destination bytes.
        space
            .read_into(dst, &mut bytes, self.host)
            .map_err(address_error)?;
        operation.copy_bytes(&source, &mut bytes).map_err(|error| {
            canonical(MaxwellCanonicalExecutionError::MemoryCopyTransform {
                source: operation.source(),
                error,
            })
        })?;
        if operation.has_remap() {
            for span in operation.written_spans().map_err(|error| {
                canonical(MaxwellCanonicalExecutionError::MemoryCopyTransform {
                    source: operation.source(),
                    error,
                })
            })? {
                let at = space
                    .checked_add(dst, span.start as u64)
                    .map_err(MaxwellGpuAccessError::Address)
                    .map_err(address_error)?;
                space
                    .write_block_cached(at, &bytes[span])
                    .map_err(address_error)?;
            }
        } else {
            space
                .write_block_cached(dst, &bytes)
                .map_err(address_error)?;
        }
        Ok(())
    }
    pub(crate) fn set_driver_completion_increments(&mut self, increments: u32) {
        self.driver_completion_increments = increments;
    }

    pub(crate) fn finish(
        self,
        driver_wait_for_idle: bool,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
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
        // The driver appends WFI unless suppressed, then two distinct increments.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp
        if driver_wait_for_idle {
            self.host.order_device(self.frontend)?;
        }
        for _ in 0..self.driver_completion_increments {
            self.host.signal_syncpoint()?;
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
                .capture(backing_offset, indices.len() as u64)
                .map_err(Arc::new)?,
        );
    }
    Ok(ranges)
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
        .resolve_range(target, size_of::<u32>() as u64)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })
}
