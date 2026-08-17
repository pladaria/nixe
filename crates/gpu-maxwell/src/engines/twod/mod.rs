//! GM20B `FERMI_TWOD_A` engine boundary.

mod beta;
mod methods;
mod notify;
mod pixels_from_memory;
mod render_enable;
mod state;

pub use beta::{
    MaxwellTwoDBeta1, MaxwellTwoDBeta4, MaxwellTwoDBetaState, MaxwellTwoDBetaStateWrite,
};
pub use notify::{
    MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX, MaxwellTwoDNotifyAddressLower,
    MaxwellTwoDNotifyAddressUpper, MaxwellTwoDNotifyState, MaxwellTwoDNotifyStateWrite,
};
pub use pixels_from_memory::{
    MAXWELL_TWO_D_CORRAL_SIZE_MAX, MaxwellTwoDPixelsFromMemoryCorralSize,
    MaxwellTwoDPixelsFromMemorySafeOverlap, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDPixelsFromMemoryStateWrite,
};
pub use render_enable::{
    MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableState, MaxwellTwoDRenderEnableStateWrite,
};
pub use state::{
    MaxwellTwoDClipEnable, MaxwellTwoDColorKeyEnable, MaxwellTwoDOperation,
    MaxwellTwoDProcessingClusters, MaxwellTwoDRegister, MaxwellTwoDRegisterOrigin,
    MaxwellTwoDState, MaxwellTwoDStateWrite,
};

use nixe_gpu::GpuClassId;

use super::{MaxwellEngineDispatchError, MaxwellEngineMethodDispatch};
use crate::MaxwellMethodDispatch;

pub(super) const CLASS: GpuClassId = GpuClassId(0x902d);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellTwoDState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    methods::preflight(method, candidate)
}
