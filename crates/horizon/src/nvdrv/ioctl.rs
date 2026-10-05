use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator};

use super::NvDrvFileDescriptor;
use super::nvhost_ctrl::PendingNvHostCtrlWait;

/// Fully decoded semantic ioctl request.
pub(crate) struct NvDrvIoctlRequest<'a> {
    pub fd: NvDrvFileDescriptor,
    pub request: u32,
    pub input: &'a [u8],
    pub inline: NvDrvInlineBuffer<'a>,
    pub process_id: u64,
    pub address_space: AddressSpaceId,
    pub translator: &'a dyn CanonicalRangeTranslator,
    pub caller: NvDrvIoctlCaller<'a>,
}

/// Caller identity and the process clock consumed by waits and timer queries.
pub(crate) struct NvDrvIoctlCaller<'a> {
    pub thread_id: u64,
    pub clock: &'a nixe_runtime::VirtualClock,
}

/// Distinct extra buffers carried by Ioctl2 and Ioctl3.
#[derive(Clone, Copy)]
pub(crate) enum NvDrvInlineBuffer<'a> {
    None,
    Input(&'a [u8]),
    Output(usize),
}
impl<'a> NvDrvInlineBuffer<'a> {
    pub fn input(self) -> &'a [u8] {
        match self {
            Self::Input(bytes) => bytes,
            _ => &[],
        }
    }
    pub fn output_size(self) -> Option<usize> {
        match self {
            Self::Output(size) => Some(size),
            _ => None,
        }
    }
}

/// Semantic ioctl response before Horizon wire encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NvDrvIoctlResponse {
    pub output: Vec<u8>,
    pub additional_output: Vec<u8>,
    pub driver_result: u32,
}

/// Semantic disposition of an ioctl before scheduler or wire adaptation.
#[derive(Clone, Debug)]
pub(crate) enum NvDrvIoctlOutcome {
    Complete(NvDrvIoctlResponse),
    PendingSyncpointWait(PendingNvHostCtrlWait),
    PendingSubmission(super::PendingGpuSubmission),
}
