//! GM20B `MAXWELL_COMPUTE_B` engine boundary.

mod launch;
mod methods;
pub(super) use methods::data_words;
mod operations;
mod qmd;
mod state;

pub use launch::resolve_compute_launch;
pub use launch::{MaxwellComputeLaunch, MaxwellComputeLaunchError, MaxwellResolvedComputeLaunch};

pub use operations::{
    MaxwellComputeOperationTrigger, MaxwellComputeShaderCacheInvalidation,
    MaxwellComputeSynchronizationPlan, MaxwellShaderCacheInvalidation,
    lower_maxwell_compute_synchronization,
};

pub use state::{
    MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT, MaxwellComputeAddress,
    MaxwellComputeBindlessTextureConstantBufferSlot, MaxwellComputeCwdRefCounterIndex,
    MaxwellComputeCwdRefCounterState, MaxwellComputeCwdRefCounterValue,
    MaxwellComputeDescriptorPoolState, MaxwellComputeInlineToMemoryLaunch,
    MaxwellComputeInlineToMemoryLayout, MaxwellComputeInlineToMemoryPendingTransfer,
    MaxwellComputeInlineToMemoryState, MaxwellComputeInlineToMemoryUpload,
    MaxwellComputeLocalMemoryAllocation, MaxwellComputeLocalMemoryState,
    MaxwellComputeProgramState, MaxwellComputeRegister, MaxwellComputeRegisterOrigin,
    MaxwellComputeSmCount, MaxwellComputeSpaVersion, MaxwellComputeState, MaxwellComputeStateWrite,
};

use nixe_gpu::GpuClassId;

pub(super) use methods::call_method;

pub(super) const CLASS: GpuClassId = GpuClassId(0xb1c0);
