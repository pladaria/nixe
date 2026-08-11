//! GM20B `MAXWELL_COMPUTE_B` engine boundary.

mod methods;
mod operations;
mod state;

pub use operations::{
    MaxwellComputeOperationTrigger, MaxwellComputeShaderCacheInvalidation,
    MaxwellComputeSynchronizationPlan, MaxwellComputeTriggeredOperation,
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

use super::{MaxwellEngineDispatchError, MaxwellEngineMethodDispatch};
use crate::MaxwellMethodDispatch;

pub(super) const CLASS: GpuClassId = GpuClassId(0xb1c0);

pub(super) fn preflight(
    class: GpuClassId,
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellComputeState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    debug_assert_eq!(class, CLASS);
    methods::preflight(method, candidate)
}
