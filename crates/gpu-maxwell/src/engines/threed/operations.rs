//! Host-independent `MAXWELL_B` ordering and completion operations.

use std::fmt::{Display, Formatter};

use nixe_gpu::{
    CacheMaintenanceOperation, GuestSyncpointId, GuestTimelinePoint, ReservedTimelinePoint,
};

use super::{MaxwellThreeDRegister, MaxwellThreeDState, MaxwellThreeDUnresolvedAddress};
use crate::{MaxwellMethodSource, MaxwellShaderCacheInvalidation};

/// Source-preserving setup for the `SET_REPORT_SEMAPHORE_A..C` registers.
///
/// Register D is the execution trigger and is deliberately not folded into
/// this passive state. Its operation, ordering, report, reduction, comparison,
/// and structure-size fields require a neutral execution operation.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3609-L3619>
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDReportSemaphoreState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    payload: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDReportSemaphoreState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }

    #[must_use]
    pub const fn payload(&self) -> &MaxwellThreeDRegister<u32> {
        &self.payload
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDReportSemaphoreStateWrite) {
        match write {
            MaxwellThreeDReportSemaphoreStateWrite::AddressUpper { value, source } => {
                self.address_upper =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDReportSemaphoreStateWrite::AddressLower { value, source } => {
                self.address_lower = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDReportSemaphoreStateWrite::Payload { value, source } => {
                self.payload = MaxwellThreeDRegister::programmed(value, value, source);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDReportSemaphoreStateWrite {
    AddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    AddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    Payload {
        value: u32,
        source: MaxwellMethodSource,
    },
}

/// Operation selected by `SET_REPORT_SEMAPHORE_D`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDReportSemaphoreOperation {
    Release = 0,
    Acquire = 1,
    ReportOnly = 2,
    Trap = 3,
}

/// Pipeline position at which a report semaphore becomes ordered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDReportSemaphorePipelineLocation {
    None = 0,
    DataAssembler = 1,
    VertexShader = 2,
    Vpc = 4,
    StreamingOutput = 5,
    GeometryShader = 6,
    ZCull = 7,
    TessellationInitShader = 8,
    TessellationShader = 9,
    PixelShader = 10,
    DepthTest = 12,
    All = 15,
}

impl MaxwellThreeDReportSemaphorePipelineLocation {
    const fn parse(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::None),
            1 => Some(Self::DataAssembler),
            2 => Some(Self::VertexShader),
            4 => Some(Self::Vpc),
            5 => Some(Self::StreamingOutput),
            6 => Some(Self::GeometryShader),
            7 => Some(Self::ZCull),
            8 => Some(Self::TessellationInitShader),
            9 => Some(Self::TessellationShader),
            10 => Some(Self::PixelShader),
            12 => Some(Self::DepthTest),
            15 => Some(Self::All),
            _ => None,
        }
    }
}

/// Payload layout written by a report semaphore.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDReportSemaphoreStructureSize {
    FourWords,
    OneWord,
}

/// Fully decoded `SET_REPORT_SEMAPHORE_D` control word.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3620-L3703>
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDReportSemaphoreControl {
    raw: u32,
    operation: MaxwellThreeDReportSemaphoreOperation,
    pipeline_location: MaxwellThreeDReportSemaphorePipelineLocation,
    structure_size: MaxwellThreeDReportSemaphoreStructureSize,
}

impl MaxwellThreeDReportSemaphoreControl {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x1fb7_ffff != 0 {
            return None;
        }
        let operation = match raw & 3 {
            0 => MaxwellThreeDReportSemaphoreOperation::Release,
            1 => MaxwellThreeDReportSemaphoreOperation::Acquire,
            2 => MaxwellThreeDReportSemaphoreOperation::ReportOnly,
            3 => MaxwellThreeDReportSemaphoreOperation::Trap,
            _ => unreachable!(),
        };
        let pipeline_location =
            match MaxwellThreeDReportSemaphorePipelineLocation::parse(((raw >> 12) & 0xf) as u8) {
                Some(value) => value,
                None => return None,
            };
        let report = ((raw >> 23) & 0x1f) as u8;
        if !matches!(
            report,
            0 | 1
                | 2
                | 3
                | 4
                | 5
                | 6
                | 7
                | 9
                | 10
                | 11
                | 12
                | 13
                | 14
                | 15
                | 16
                | 17
                | 18
                | 19
                | 21
                | 24
                | 25
                | 26
                | 27
                | 28
                | 29
                | 30
                | 31
        ) || (raw >> 17) & 3 > 1
        {
            return None;
        }
        Some(Self {
            raw,
            operation,
            pipeline_location,
            structure_size: if raw & (1 << 28) == 0 {
                MaxwellThreeDReportSemaphoreStructureSize::FourWords
            } else {
                MaxwellThreeDReportSemaphoreStructureSize::OneWord
            },
        })
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.raw
    }

    #[must_use]
    pub const fn operation(self) -> MaxwellThreeDReportSemaphoreOperation {
        self.operation
    }

    #[must_use]
    pub const fn pipeline_location(self) -> MaxwellThreeDReportSemaphorePipelineLocation {
        self.pipeline_location
    }

    #[must_use]
    pub const fn structure_size(self) -> MaxwellThreeDReportSemaphoreStructureSize {
        self.structure_size
    }

    #[must_use]
    pub const fn is_captured_release(self) -> bool {
        self.raw == 0x1000_f010
    }
}

/// One ordered one-word release to guest-visible GPU memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDReportSemaphoreRelease {
    address: MaxwellThreeDUnresolvedAddress,
    payload: u32,
    source: MaxwellMethodSource,
    prior_work_pending: bool,
}

impl MaxwellThreeDReportSemaphoreRelease {
    #[must_use]
    pub const fn address(self) -> MaxwellThreeDUnresolvedAddress {
        self.address
    }

    #[must_use]
    pub const fn payload(self) -> u32 {
        self.payload
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }

    #[must_use]
    pub const fn prior_work_pending(self) -> bool {
        self.prior_work_pending
    }
}

/// Decoded ordered `INVALIDATE_SHADER_CACHES` request.
///
/// The cache selectors and the two additional control flags are defined by
/// NVIDIA's pinned public `MAXWELL_B` class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L267-L282>
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDShaderCacheInvalidation {
    caches: MaxwellShaderCacheInvalidation,
    locks: bool,
    flush_data: bool,
}

impl MaxwellThreeDShaderCacheInvalidation {
    pub(crate) const fn new(
        caches: MaxwellShaderCacheInvalidation,
        locks: bool,
        flush_data: bool,
    ) -> Self {
        Self {
            caches,
            locks,
            flush_data,
        }
    }

    #[must_use]
    pub const fn caches(self) -> MaxwellShaderCacheInvalidation {
        self.caches
    }

    #[must_use]
    pub const fn locks(self) -> bool {
        self.locks
    }

    #[must_use]
    pub const fn flush_data(self) -> bool {
        self.flush_data
    }
}

// Method fields and enum values come from NVIDIA's generated MAXWELL_B
// header at the repository-pinned revision:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L284-L291
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1833-L1834
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2621-L2631
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2232-L2250

/// Work whose completion gates one `INCREMENT_SYNC_POINT` operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDSyncpointCondition {
    StreamOutWritesDone,
    RopWritesDone,
}

/// Decoded `FLUSH_PENDING_WRITES` request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDFlushPendingWrites {
    sm_does_global_store: bool,
}

/// Cache-line selection shared by Maxwell texture-data, texture-header, and
/// sampler invalidation methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDTextureCacheLines {
    All,
    One,
}

/// Maxwell texture-related cache selected by one invalidation method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDTextureCacheTarget {
    Data,
    Header,
    Sampler,
}

/// Decoded texture-related cache invalidation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTextureCacheInvalidation {
    target: MaxwellThreeDTextureCacheTarget,
    lines: MaxwellThreeDTextureCacheLines,
    tag: u32,
}

impl MaxwellThreeDTextureCacheInvalidation {
    pub(crate) const fn new(
        target: MaxwellThreeDTextureCacheTarget,
        lines: MaxwellThreeDTextureCacheLines,
        tag: u32,
    ) -> Self {
        Self { target, lines, tag }
    }

    #[must_use]
    pub const fn target(self) -> MaxwellThreeDTextureCacheTarget {
        self.target
    }

    #[must_use]
    pub const fn lines(self) -> MaxwellThreeDTextureCacheLines {
        self.lines
    }

    #[must_use]
    pub const fn tag(self) -> u32 {
        self.tag
    }
}

impl MaxwellThreeDFlushPendingWrites {
    pub(crate) const fn new(sm_does_global_store: bool) -> Self {
        Self {
            sm_does_global_store,
        }
    }

    #[must_use]
    pub const fn sm_does_global_store(self) -> bool {
        self.sm_does_global_store
    }
}

/// Decoded `INCREMENT_SYNC_POINT` request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDSyncpointIncrement {
    syncpoint: GuestSyncpointId,
    clean_l2: bool,
    condition: MaxwellThreeDSyncpointCondition,
}

impl MaxwellThreeDSyncpointIncrement {
    pub(crate) const fn new(
        syncpoint: GuestSyncpointId,
        clean_l2: bool,
        condition: MaxwellThreeDSyncpointCondition,
    ) -> Self {
        Self {
            syncpoint,
            clean_l2,
            condition,
        }
    }

    #[must_use]
    pub const fn syncpoint(self) -> GuestSyncpointId {
        self.syncpoint
    }

    #[must_use]
    pub const fn clean_l2(self) -> bool {
        self.clean_l2
    }

    #[must_use]
    pub const fn condition(self) -> MaxwellThreeDSyncpointCondition {
        self.condition
    }
}

/// One 3D execution-order trigger emitted by a class method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDSynchronizationTrigger {
    WaitForIdle {
        value: u32,
        source: MaxwellMethodSource,
    },
    InvalidateShaderCaches {
        request: MaxwellThreeDShaderCacheInvalidation,
        source: MaxwellMethodSource,
    },
    InvalidateShaderCachesNoWfi {
        caches: MaxwellShaderCacheInvalidation,
        source: MaxwellMethodSource,
    },
    InvalidateTextureCacheNoWfi {
        request: MaxwellThreeDTextureCacheInvalidation,
        source: MaxwellMethodSource,
    },
    InvalidateTextureCache {
        request: MaxwellThreeDTextureCacheInvalidation,
        source: MaxwellMethodSource,
    },
    FlushPendingWrites {
        request: MaxwellThreeDFlushPendingWrites,
        source: MaxwellMethodSource,
    },
    ReportSemaphore {
        control: MaxwellThreeDReportSemaphoreControl,
        source: MaxwellMethodSource,
    },
    IncrementSyncpoint {
        request: MaxwellThreeDSyncpointIncrement,
        source: MaxwellMethodSource,
    },
}

impl MaxwellThreeDSynchronizationTrigger {
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::WaitForIdle { source, .. }
            | Self::InvalidateShaderCaches { source, .. }
            | Self::InvalidateShaderCachesNoWfi { source, .. }
            | Self::InvalidateTextureCacheNoWfi { source, .. }
            | Self::InvalidateTextureCache { source, .. }
            | Self::FlushPendingWrites { source, .. }
            | Self::ReportSemaphore { source, .. }
            | Self::IncrementSyncpoint { source, .. } => source,
        }
    }
}

/// One synchronization trigger paired with the exact 3D state at that method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDSynchronizationOperation {
    trigger: MaxwellThreeDSynchronizationTrigger,
    state: MaxwellThreeDState,
}

impl MaxwellThreeDSynchronizationOperation {
    pub(crate) const fn new(
        trigger: MaxwellThreeDSynchronizationTrigger,
        state: MaxwellThreeDState,
    ) -> Self {
        Self { trigger, state }
    }

    #[must_use]
    pub const fn trigger(&self) -> MaxwellThreeDSynchronizationTrigger {
        self.trigger
    }

    #[must_use]
    pub const fn state(&self) -> &MaxwellThreeDState {
        &self.state
    }
}

/// Validated host-independent lowering of one 3D synchronization operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDSynchronizationPlan {
    /// Orders every earlier channel operation before later work.
    WaitForIdle {
        prior_work_pending: bool,
    },
    InvalidateShaderCaches {
        request: MaxwellThreeDShaderCacheInvalidation,
        maintenance: CacheMaintenanceOperation,
        prior_work_pending: bool,
    },
    InvalidateShaderCachesNoWfi {
        caches: MaxwellShaderCacheInvalidation,
        maintenance: CacheMaintenanceOperation,
    },
    InvalidateTextureCacheNoWfi {
        request: MaxwellThreeDTextureCacheInvalidation,
        maintenance: CacheMaintenanceOperation,
    },
    InvalidateTextureCache {
        request: MaxwellThreeDTextureCacheInvalidation,
        maintenance: CacheMaintenanceOperation,
        prior_work_pending: bool,
    },
    FlushPendingWrites {
        request: MaxwellThreeDFlushPendingWrites,
    },
    ReportSemaphoreRelease(MaxwellThreeDReportSemaphoreRelease),
    IncrementSyncpoint {
        request: MaxwellThreeDSyncpointIncrement,
        completion: GuestTimelinePoint,
    },
}

/// Inconsistent completion ownership at the 3D synchronization boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDSynchronizationError {
    UnsupportedShaderCacheLockInvalidation {
        source: MaxwellMethodSource,
    },
    UnsupportedShaderDataCacheFlush {
        source: MaxwellMethodSource,
    },
    IncompleteReportSemaphoreState {
        source: MaxwellMethodSource,
    },
    UnsupportedReportSemaphoreControl {
        source: MaxwellMethodSource,
        control: MaxwellThreeDReportSemaphoreControl,
    },
    MissingCompletionReservation {
        source: MaxwellMethodSource,
        requested: GuestSyncpointId,
    },
    WrongCompletionSyncpoint {
        source: MaxwellMethodSource,
        requested: GuestSyncpointId,
        reserved: GuestSyncpointId,
    },
}

impl std::error::Error for MaxwellThreeDSynchronizationError {}

impl Display for MaxwellThreeDSynchronizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedShaderCacheLockInvalidation { source } => write!(
                formatter,
                "MAXWELL_B shader-cache lock invalidation is not represented by neutral cache maintenance: source=[{source}]"
            ),
            Self::UnsupportedShaderDataCacheFlush { source } => write!(
                formatter,
                "MAXWELL_B shader-data cache flush is not represented by neutral cache maintenance: source=[{source}]"
            ),
            Self::IncompleteReportSemaphoreState { source } => write!(
                formatter,
                "MAXWELL_B report semaphore release has no complete address and payload: source=[{source}]"
            ),
            Self::UnsupportedReportSemaphoreControl { source, control } => write!(
                formatter,
                "MAXWELL_B report semaphore control is not implemented: source=[{source}] control=0x{:08x}",
                control.raw()
            ),
            Self::MissingCompletionReservation { source, requested } => write!(
                formatter,
                "MAXWELL_B syncpoint increment has no completion reservation: source=[{source}] requested={requested:?}"
            ),
            Self::WrongCompletionSyncpoint {
                source,
                requested,
                reserved,
            } => write!(
                formatter,
                "MAXWELL_B syncpoint reservation mismatch: source=[{source}] requested={requested:?} reserved={reserved:?}"
            ),
        }
    }
}

/// Produces an ordering plan without publishing or consuming the reservation.
///
/// The completion owner remains responsible for publishing the reservation
/// only after the requested condition, cache maintenance, and preceding work
/// have completed.
///
/// NVIDIA exposes `WAIT_FOR_IDLE` as a full-width channel-ordering method:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L51-L52>
pub fn lower_maxwell_three_d_synchronization(
    operation: &MaxwellThreeDSynchronizationOperation,
    completion: Option<&ReservedTimelinePoint>,
    prior_work_pending: bool,
) -> Result<MaxwellThreeDSynchronizationPlan, MaxwellThreeDSynchronizationError> {
    match operation.trigger() {
        MaxwellThreeDSynchronizationTrigger::WaitForIdle { .. } => {
            Ok(MaxwellThreeDSynchronizationPlan::WaitForIdle { prior_work_pending })
        }
        MaxwellThreeDSynchronizationTrigger::InvalidateShaderCaches { request, source } => {
            if request.locks() {
                return Err(
                    MaxwellThreeDSynchronizationError::UnsupportedShaderCacheLockInvalidation {
                        source,
                    },
                );
            }
            if request.flush_data() {
                return Err(
                    MaxwellThreeDSynchronizationError::UnsupportedShaderDataCacheFlush { source },
                );
            }
            let caches = request.caches();
            Ok(MaxwellThreeDSynchronizationPlan::InvalidateShaderCaches {
                request,
                maintenance: CacheMaintenanceOperation::InvalidateShaderCaches {
                    instruction: caches.instruction(),
                    global_data: caches.global_data(),
                    constant: caches.constant(),
                },
                prior_work_pending,
            })
        }
        MaxwellThreeDSynchronizationTrigger::InvalidateShaderCachesNoWfi { caches, .. } => Ok(
            MaxwellThreeDSynchronizationPlan::InvalidateShaderCachesNoWfi {
                caches,
                maintenance: CacheMaintenanceOperation::InvalidateShaderCaches {
                    instruction: caches.instruction(),
                    global_data: caches.global_data(),
                    constant: caches.constant(),
                },
            },
        ),
        MaxwellThreeDSynchronizationTrigger::InvalidateTextureCacheNoWfi { request, .. } => {
            let maintenance = texture_cache_maintenance(request);
            Ok(
                MaxwellThreeDSynchronizationPlan::InvalidateTextureCacheNoWfi {
                    request,
                    maintenance,
                },
            )
        }
        MaxwellThreeDSynchronizationTrigger::InvalidateTextureCache { request, .. } => {
            Ok(MaxwellThreeDSynchronizationPlan::InvalidateTextureCache {
                request,
                maintenance: texture_cache_maintenance(request),
                prior_work_pending,
            })
        }
        MaxwellThreeDSynchronizationTrigger::FlushPendingWrites { request, .. } => {
            Ok(MaxwellThreeDSynchronizationPlan::FlushPendingWrites { request })
        }
        MaxwellThreeDSynchronizationTrigger::ReportSemaphore { control, source } => {
            if !control.is_captured_release() {
                return Err(
                    MaxwellThreeDSynchronizationError::UnsupportedReportSemaphoreControl {
                        source,
                        control,
                    },
                );
            }
            let state = operation.state().report_semaphore();
            let (Some(address), Some(payload)) =
                (state.address(), state.payload().value().copied())
            else {
                return Err(
                    MaxwellThreeDSynchronizationError::IncompleteReportSemaphoreState { source },
                );
            };
            Ok(MaxwellThreeDSynchronizationPlan::ReportSemaphoreRelease(
                MaxwellThreeDReportSemaphoreRelease {
                    address,
                    payload,
                    source,
                    prior_work_pending,
                },
            ))
        }
        MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, source } => {
            let Some(completion) = completion else {
                return Err(
                    MaxwellThreeDSynchronizationError::MissingCompletionReservation {
                        source,
                        requested: request.syncpoint(),
                    },
                );
            };
            let point = completion.point();
            if point.syncpoint() != request.syncpoint() {
                return Err(
                    MaxwellThreeDSynchronizationError::WrongCompletionSyncpoint {
                        source,
                        requested: request.syncpoint(),
                        reserved: point.syncpoint(),
                    },
                );
            }
            Ok(MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
                request,
                completion: point,
            })
        }
    }
}

const fn texture_cache_maintenance(
    request: MaxwellThreeDTextureCacheInvalidation,
) -> CacheMaintenanceOperation {
    match request.target() {
        MaxwellThreeDTextureCacheTarget::Data => {
            CacheMaintenanceOperation::InvalidateTextureReadCaches
        }
        MaxwellThreeDTextureCacheTarget::Header => {
            CacheMaintenanceOperation::InvalidateTextureHeaderCaches
        }
        MaxwellThreeDTextureCacheTarget::Sampler => {
            CacheMaintenanceOperation::InvalidateSamplerCaches
        }
    }
}
