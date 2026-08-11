//! Host-independent `MAXWELL_COMPUTE_B` ordering operations.

use std::fmt::{Display, Formatter};

use super::MaxwellComputeState;
use crate::MaxwellMethodSource;

/// Compute shader-cache families selected by an invalidation method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellComputeShaderCacheInvalidation {
    instruction: bool,
    global_data: bool,
    constant: bool,
}

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
    Neutral,
    InvalidateShaderCachesNoWfi {
        caches: MaxwellComputeShaderCacheInvalidation,
    },
}

/// Missing execution semantics for a compute ordering operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellComputeSynchronizationError {
    PendingWork { source: MaxwellMethodSource },
}

impl Display for MaxwellComputeSynchronizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PendingWork { source } => write!(
                formatter,
                "compute WAIT_FOR_IDLE cannot be lowered neutrally while prior channel work is pending: {source}"
            ),
        }
    }
}

impl std::error::Error for MaxwellComputeSynchronizationError {}

/// Lowers a compute ordering operation only when it is observably neutral.
pub fn lower_maxwell_compute_synchronization(
    operation: &MaxwellComputeTriggeredOperation,
    prior_channel_work_pending: bool,
) -> Result<MaxwellComputeSynchronizationPlan, MaxwellComputeSynchronizationError> {
    match operation.trigger() {
        MaxwellComputeOperationTrigger::WaitForIdle { source, .. } => {
            if prior_channel_work_pending {
                return Err(MaxwellComputeSynchronizationError::PendingWork { source });
            }
            Ok(MaxwellComputeSynchronizationPlan::Neutral)
        }
        MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi { caches, .. } => {
            Ok(MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches })
        }
    }
}
