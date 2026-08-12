//! Host-independent `MAXWELL_COMPUTE_B` ordering operations.

use super::MaxwellComputeState;
use crate::MaxwellMethodSource;

/// Compute shader-cache families selected by an invalidation method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeShaderCacheInvalidation {
    instruction: bool,
    global_data: bool,
    constant: bool,
}

/// Engine-independent name for the identical Maxwell shader-cache selectors.
pub type MaxwellShaderCacheInvalidation = MaxwellComputeShaderCacheInvalidation;

impl MaxwellComputeShaderCacheInvalidation {
    pub(crate) const fn new(instruction: bool, global_data: bool, constant: bool) -> Self {
        Self {
            instruction,
            global_data,
            constant,
        }
    }

    #[must_use]
    pub const fn instruction(self) -> bool {
        self.instruction
    }

    #[must_use]
    pub const fn global_data(self) -> bool {
        self.global_data
    }

    #[must_use]
    pub const fn constant(self) -> bool {
        self.constant
    }
}

/// One compute execution-order trigger emitted by a class method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellComputeOperationTrigger {
    WaitForIdle {
        value: u32,
        source: MaxwellMethodSource,
    },
    InvalidateShaderCachesNoWfi {
        caches: MaxwellComputeShaderCacheInvalidation,
        source: MaxwellMethodSource,
    },
}

impl MaxwellComputeOperationTrigger {
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::WaitForIdle { source, .. } | Self::InvalidateShaderCachesNoWfi { source, .. } => {
                source
            }
        }
    }
}

/// One compute trigger paired with the exact candidate state at that method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellComputeTriggeredOperation {
    trigger: MaxwellComputeOperationTrigger,
    state: MaxwellComputeState,
}

impl MaxwellComputeTriggeredOperation {
    pub(crate) const fn new(
        trigger: MaxwellComputeOperationTrigger,
        state: MaxwellComputeState,
    ) -> Self {
        Self { trigger, state }
    }

    #[must_use]
    pub const fn trigger(&self) -> MaxwellComputeOperationTrigger {
        self.trigger
    }

    #[must_use]
    pub const fn state(&self) -> &MaxwellComputeState {
        &self.state
    }
}

/// Validated host-independent lowering of a compute ordering operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellComputeSynchronizationPlan {
    /// Orders every earlier operation in the channel before later work.
    ///
    /// `prior_work_pending` tells the execution layer whether it must drain an
    /// emitted prefix or whether the barrier is already observably satisfied.
    WaitForIdle { prior_work_pending: bool },
    InvalidateShaderCachesNoWfi {
        caches: MaxwellComputeShaderCacheInvalidation,
    },
}

/// Lowers a compute ordering operation without executing or completing it.
///
/// NVIDIA defines `WAIT_FOR_IDLE` as a channel-ordering method. Pending work
/// therefore makes it a real barrier rather than an unsupported state or a
/// no-op. The execution layer must drain that prefix before continuing.
///
/// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L51-L52
pub fn lower_maxwell_compute_synchronization(
    operation: &MaxwellComputeTriggeredOperation,
    prior_channel_work_pending: bool,
) -> MaxwellComputeSynchronizationPlan {
    match operation.trigger() {
        MaxwellComputeOperationTrigger::WaitForIdle { .. } => {
            MaxwellComputeSynchronizationPlan::WaitForIdle {
                prior_work_pending: prior_channel_work_pending,
            }
        }
        MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi { caches, .. } => {
            MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches }
        }
    }
}
