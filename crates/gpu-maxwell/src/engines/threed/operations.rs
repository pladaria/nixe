//! Host-independent `MAXWELL_B` ordering and completion operations.

use std::fmt::{Display, Formatter};

use nixe_gpu::{
    CacheMaintenanceOperation, GuestSyncpointId, GuestTimelinePoint, ReservedTimelinePoint,
};

use super::MaxwellThreeDState;
use crate::{MaxwellMethodSource, MaxwellShaderCacheInvalidation};

// Method fields and enum values come from NVIDIA's generated MAXWELL_B
// header at the repository-pinned revision:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L284-L291
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1833-L1834

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

/// Texture-cache line selection encoded by
/// `INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDTextureCacheLines {
    All,
    One,
}

/// Decoded texture-data cache invalidation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTextureDataCacheInvalidation {
    lines: MaxwellThreeDTextureCacheLines,
    tag: u32,
}

impl MaxwellThreeDTextureDataCacheInvalidation {
    pub(crate) const fn new(lines: MaxwellThreeDTextureCacheLines, tag: u32) -> Self {
        Self { lines, tag }
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
    InvalidateShaderCachesNoWfi {
        caches: MaxwellShaderCacheInvalidation,
        source: MaxwellMethodSource,
    },
    InvalidateTextureDataCacheNoWfi {
        request: MaxwellThreeDTextureDataCacheInvalidation,
        source: MaxwellMethodSource,
    },
    FlushPendingWrites {
        request: MaxwellThreeDFlushPendingWrites,
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
            Self::InvalidateShaderCachesNoWfi { source, .. }
            | Self::InvalidateTextureDataCacheNoWfi { source, .. }
            | Self::FlushPendingWrites { source, .. }
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
    InvalidateShaderCachesNoWfi {
        caches: MaxwellShaderCacheInvalidation,
        maintenance: CacheMaintenanceOperation,
    },
    InvalidateTextureDataCacheNoWfi {
        request: MaxwellThreeDTextureDataCacheInvalidation,
        maintenance: CacheMaintenanceOperation,
    },
    FlushPendingWrites {
        request: MaxwellThreeDFlushPendingWrites,
    },
    IncrementSyncpoint {
        request: MaxwellThreeDSyncpointIncrement,
        completion: GuestTimelinePoint,
    },
}

/// Inconsistent completion ownership at the 3D synchronization boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDSynchronizationError {
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

impl Display for MaxwellThreeDSynchronizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCompletionReservation { source, requested } => write!(
                formatter,
                "3D syncpoint increment has no reserved completion: requested={requested} {source}"
            ),
            Self::WrongCompletionSyncpoint {
                source,
                requested,
                reserved,
            } => write!(
                formatter,
                "3D syncpoint increment does not match the reserved completion: requested={requested} reserved={reserved} {source}"
            ),
        }
    }
}

impl std::error::Error for MaxwellThreeDSynchronizationError {}

/// Produces an ordering plan without publishing or consuming the reservation.
///
/// The completion owner remains responsible for publishing the reservation
/// only after the requested condition, cache maintenance, and preceding work
/// have completed.
pub fn lower_maxwell_three_d_synchronization(
    operation: &MaxwellThreeDSynchronizationOperation,
    completion: Option<&ReservedTimelinePoint>,
) -> Result<MaxwellThreeDSynchronizationPlan, MaxwellThreeDSynchronizationError> {
    match operation.trigger() {
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
        MaxwellThreeDSynchronizationTrigger::InvalidateTextureDataCacheNoWfi {
            request, ..
        } => Ok(
            MaxwellThreeDSynchronizationPlan::InvalidateTextureDataCacheNoWfi {
                request,
                maintenance: CacheMaintenanceOperation::InvalidateTextureReadCaches,
            },
        ),
        MaxwellThreeDSynchronizationTrigger::FlushPendingWrites { request, .. } => {
            Ok(MaxwellThreeDSynchronizationPlan::FlushPendingWrites { request })
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
