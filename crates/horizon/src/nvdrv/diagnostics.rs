use std::fmt::{Display, Formatter, Write};

use nixe_gpu::{FrontendSubmissionId, GraphicsGapKind};
use nixe_gpu_maxwell::{
    MaxwellFrontendDispatchBoundary, MaxwellGpfifoSourceError, MaxwellScheduleError,
    MaxwellUnsupportedGpfifoSubmission,
};
use nixe_memory::CanonicalRangeTranslationError;

use super::device::{NvDrvDeviceKind, NvDrvFileDescriptor};
use super::nvmap::NvMapHandle;

const MAX_DIAGNOSTIC_GUEST_BYTES: usize = 96;

/// Why an NVIDIA semantic operation could not be completed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NvDrvValidationReason {
    UnsupportedOperation,
    CanonicalBackingUnavailable,
    AddressSpaceUnavailable,
    AddressSpaceGenerationExhausted,
    AddressSpaceIdentityExhausted,
    DeviceStateUnavailable,
    TimelineIdentityExhausted,
    TimelineOrderingUnavailable,
    GpfifoMemoryResolutionFailed,
    GpfifoSchedulingUnavailable,
    MaxwellPacketSemanticsUnavailable,
    NeutralBackendExecutionFailed,
}

impl Display for NvDrvValidationReason {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::UnsupportedOperation => "unsupported-operation",
            Self::CanonicalBackingUnavailable => "canonical-backing-unavailable",
            Self::AddressSpaceUnavailable => "address-space-unavailable",
            Self::AddressSpaceGenerationExhausted => "address-space-generation-exhausted",
            Self::AddressSpaceIdentityExhausted => "address-space-identity-exhausted",
            Self::DeviceStateUnavailable => "device-state-unavailable",
            Self::TimelineIdentityExhausted => "timeline-identity-exhausted",
            Self::TimelineOrderingUnavailable => "timeline-ordering-unavailable",
            Self::GpfifoMemoryResolutionFailed => "gpfifo-memory-resolution-failed",
            Self::GpfifoSchedulingUnavailable => "gpfifo-scheduling-unavailable",
            Self::MaxwellPacketSemanticsUnavailable => "maxwell-packet-semantics-unavailable",
            Self::NeutralBackendExecutionFailed => "neutral-backend-execution-failed",
        })
    }
}

/// Pointer-free context attached to an NVIDIA semantic failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvDrvErrorContext {
    device: NvDrvDeviceKind,
    request: u32,
    fd: NvDrvFileDescriptor,
    allocation: Option<NvMapHandle>,
    reason: NvDrvValidationReason,
}

impl NvDrvErrorContext {
    pub const fn new(
        device: NvDrvDeviceKind,
        request: u32,
        fd: NvDrvFileDescriptor,
        allocation: Option<NvMapHandle>,
        reason: NvDrvValidationReason,
    ) -> Self {
        Self {
            device,
            request,
            fd,
            allocation,
            reason,
        }
    }

    pub const fn device(self) -> NvDrvDeviceKind {
        self.device
    }

    pub const fn request(self) -> u32 {
        self.request
    }

    pub const fn fd(self) -> NvDrvFileDescriptor {
        self.fd
    }

    pub const fn allocation(self) -> Option<NvMapHandle> {
        self.allocation
    }

    pub const fn reason(self) -> NvDrvValidationReason {
        self.reason
    }
}

/// An `nvdrv` operation for which Nixe cannot yet provide faithful semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnsupportedNvDrvOperation {
    OpenDevice {
        path: Box<[u8]>,
    },
    ServiceCommand {
        command_id: u32,
    },
    QueryEvent {
        device: NvDrvDeviceKind,
        fd: NvDrvFileDescriptor,
        event_id: u32,
    },
    Ioctl {
        context: NvDrvErrorContext,
    },
    GpfifoSubmission {
        context: NvDrvErrorContext,
        error: MaxwellUnsupportedGpfifoSubmission,
    },
    GpfifoMemory {
        context: NvDrvErrorContext,
        error: Box<MaxwellGpfifoSourceError>,
    },
    GpfifoScheduling {
        context: NvDrvErrorContext,
        error: Box<MaxwellScheduleError>,
    },
    ScheduledGpfifoSubmission {
        context: NvDrvErrorContext,
        boundary: Box<MaxwellFrontendDispatchBoundary>,
    },
    GpuExecution {
        context: NvDrvErrorContext,
        frontend: FrontendSubmissionId,
        detail: Box<str>,
    },
    CanonicalMemory {
        context: NvDrvErrorContext,
        fault: CanonicalRangeTranslationError,
    },
}

impl UnsupportedNvDrvOperation {
    /// Classifies the first missing graphics semantic layer.
    #[must_use]
    pub const fn gap_kind(&self) -> GraphicsGapKind {
        match self {
            Self::OpenDevice { .. } => GraphicsGapKind::DeviceOpen,
            Self::ServiceCommand { .. } | Self::QueryEvent { .. } => {
                GraphicsGapKind::ServiceCommand
            }
            Self::Ioctl { .. }
            | Self::GpfifoSubmission { .. }
            | Self::GpfifoMemory { .. }
            | Self::GpfifoScheduling { .. }
            | Self::CanonicalMemory { .. } => GraphicsGapKind::Ioctl,
            Self::ScheduledGpfifoSubmission { .. } | Self::GpuExecution { .. } => {
                GraphicsGapKind::GpuPacket
            }
        }
    }
}

impl Display for UnsupportedNvDrvOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "graphics-gap={} ", self.gap_kind())?;
        match self {
            Self::OpenDevice { path } => {
                formatter.write_str("nvdrv device open is not implemented: path=")?;
                write_bounded_guest_bytes(formatter, path)
            }
            Self::ServiceCommand { command_id } => write!(
                formatter,
                "nvdrv service command is not implemented: command={command_id}"
            ),
            Self::QueryEvent {
                device,
                fd,
                event_id,
            } => write!(
                formatter,
                "nvdrv QueryEvent is not implemented for device={} fd={} event-id={:#010x}",
                device.path(),
                fd,
                event_id
            ),
            Self::Ioctl { context } => write!(
                formatter,
                "nvdrv ioctl is not implemented: device={} request={:#010x} fd={} reason={}",
                context.device().path(),
                context.request(),
                context.fd(),
                context.reason()
            ),
            Self::GpfifoSubmission { context, error } => write!(
                formatter,
                "nvdrv GPFIFO submission mode is not implemented: device={} request={:#010x} fd={} reason={} detail=[{}]",
                context.device().path(),
                context.request(),
                context.fd(),
                context.reason(),
                error
            ),
            Self::GpfifoMemory { context, error } => write!(
                formatter,
                "nvdrv GPFIFO command memory is invalid or unavailable: device={} request={:#010x} fd={} reason={} detail=[{}]",
                context.device().path(),
                context.request(),
                context.fd(),
                context.reason(),
                error
            ),
            Self::GpfifoScheduling { context, error } => write!(
                formatter,
                "nvdrv GPFIFO scheduling failed: device={} request={:#010x} fd={} reason={} detail=[{}]",
                context.device().path(),
                context.request(),
                context.fd(),
                context.reason(),
                error
            ),
            Self::ScheduledGpfifoSubmission { context, boundary } => write!(
                formatter,
                "validated GPFIFO work reached the first unsupported Maxwell frontend boundary: device={} request={:#010x} fd={} reason={} dispatch=[{}]",
                context.device().path(),
                context.request(),
                context.fd(),
                context.reason(),
                boundary
            ),
            Self::GpuExecution {
                context,
                frontend,
                detail,
            } => write!(
                formatter,
                "GPU execution failed: device={} request={:#010x} fd={} reason={} {} detail=[{}]",
                context.device().path(),
                context.request(),
                context.fd(),
                context.reason(),
                frontend,
                detail
            ),
            Self::CanonicalMemory { context, fault } => {
                write!(
                    formatter,
                    "nvdrv ioctl cannot establish canonical backing: device={} \
                     request={:#010x} fd={}",
                    context.device().path(),
                    context.request(),
                    context.fd()
                )?;
                if let Some(allocation) = context.allocation() {
                    write!(formatter, " allocation={:#010x}", allocation.raw())?;
                }
                write!(formatter, " reason={} fault=[{fault}]", context.reason())
            }
        }
    }
}

fn write_bounded_guest_bytes(formatter: &mut Formatter<'_>, bytes: &[u8]) -> std::fmt::Result {
    formatter.write_str("\"")?;
    for byte in bytes.iter().take(MAX_DIAGNOSTIC_GUEST_BYTES) {
        for escaped in std::ascii::escape_default(*byte) {
            formatter.write_char(escaped as char)?;
        }
    }
    if bytes.len() > MAX_DIAGNOSTIC_GUEST_BYTES {
        write!(
            formatter,
            "...<{} bytes omitted>",
            bytes.len() - MAX_DIAGNOSTIC_GUEST_BYTES
        )?;
    }
    formatter.write_str("\"")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NvDrvCallError {
    GuestResult(u32),
    Unsupported(UnsupportedNvDrvOperation),
}

impl From<u32> for NvDrvCallError {
    fn from(result: u32) -> Self {
        Self::GuestResult(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_supplied_paths_are_bounded_and_escaped() {
        let mut path = vec![b'a'; MAX_DIAGNOSTIC_GUEST_BYTES + 20];
        path[0] = b'\n';
        let diagnostic = UnsupportedNvDrvOperation::OpenDevice { path: path.into() }.to_string();

        assert!(diagnostic.starts_with(
            "graphics-gap=device-open nvdrv device open is not implemented: path=\"\\n"
        ));
        assert!(diagnostic.ends_with("...<20 bytes omitted>\""));
        assert!(diagnostic.len() < 256);
    }
}
