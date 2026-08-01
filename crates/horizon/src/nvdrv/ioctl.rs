use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator};

use super::NvDrvFileDescriptor;
use super::nvhost_ctrl::PendingNvHostCtrlWait;

/// Fully decoded semantic ioctl request.
pub(crate) struct NvDrvIoctlRequest<'a> {
    pub fd: NvDrvFileDescriptor,
    pub request: u32,
    pub input: &'a [u8],
    pub process_id: u64,
    pub address_space: AddressSpaceId,
    pub translator: &'a dyn CanonicalRangeTranslator,
    pub thread_id: u64,
}

/// Semantic ioctl response before Horizon wire encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NvDrvIoctlResponse {
    pub output: Vec<u8>,
    pub driver_result: u32,
}

/// Semantic disposition of an ioctl before scheduler or wire adaptation.
#[derive(Clone, Debug)]
pub(crate) enum NvDrvIoctlOutcome {
    Complete(NvDrvIoctlResponse),
    PendingSyncpointWait(PendingNvHostCtrlWait),
}
