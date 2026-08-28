//! Typed failures produced while validating or dispatching Horizon IPC.

use nixe_cpu::memory::DataAccessFault;

use super::message::MessageError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum IpcWireError {
    GuestMemory(DataAccessFault),
    Malformed(&'static str),
    Internal(&'static str),
    HostResourceExhausted(&'static str),
    ResponseCommit(DataAccessFault),
    GraphicsBackend(Box<str>),
    ErrorApplet(Box<crate::ErrorAppletDiagnostic>),
    UnsupportedService(UnsupportedServiceOperation),
    UnsupportedNvDrv(crate::nvdrv::UnsupportedNvDrvOperation),
    /// A decoded direct nvdrv wait which must suspend at the SVC boundary.
    PendingNvDrv(crate::nvdrv::PendingNvHostCtrlWait),
}

/// Fatal diagnostic retained when a checked HIPC/CMIF operation cannot finish.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HorizonIpcFault(IpcWireError);

impl HorizonIpcFault {
    #[must_use]
    pub const fn malformed(reason: &'static str) -> Self {
        Self(IpcWireError::Malformed(reason))
    }

    #[must_use]
    pub fn unsupported_service(operation: UnsupportedServiceOperation) -> Self {
        Self(IpcWireError::UnsupportedService(operation))
    }

    /// Returns the retained nvdrv diagnostic when graphics emulation stopped
    /// at an unsupported operation.
    #[must_use]
    pub const fn unsupported_nvdrv(&self) -> Option<&crate::nvdrv::UnsupportedNvDrvOperation> {
        match &self.0 {
            IpcWireError::UnsupportedNvDrv(operation) => Some(operation),
            _ => None,
        }
    }

    pub(crate) const fn from_wire(error: IpcWireError) -> Self {
        Self(error)
    }
}

impl std::fmt::Display for HorizonIpcFault {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            IpcWireError::GuestMemory(fault) => write!(formatter, "guest-memory fault: {fault:?}"),
            IpcWireError::Malformed(reason) => write!(formatter, "malformed IPC: {reason}"),
            IpcWireError::Internal(reason) => {
                write!(formatter, "invalid emulator IPC state: {reason}")
            }
            IpcWireError::HostResourceExhausted(operation) => {
                write!(formatter, "exhausted host resources while {operation}")
            }
            IpcWireError::ResponseCommit(fault) => {
                write!(
                    formatter,
                    "could not commit a prevalidated response: {fault:?}"
                )
            }
            IpcWireError::GraphicsBackend(reason) => {
                write!(formatter, "GPU presentation export failed: {reason}")
            }
            IpcWireError::ErrorApplet(diagnostic) => {
                write!(
                    formatter,
                    "launched the unimplemented Error library applet: {diagnostic}"
                )
            }
            IpcWireError::UnsupportedService(operation) => {
                write!(
                    formatter,
                    "reached unsupported emulator semantics: {operation}"
                )
            }
            IpcWireError::UnsupportedNvDrv(operation) => {
                write!(
                    formatter,
                    "reached unsupported emulator semantics: {operation}"
                )
            }
            IpcWireError::PendingNvDrv(_) => {
                formatter.write_str("pending nvdrv wait escaped the scheduler boundary")
            }
        }
    }
}

/// A Horizon service operation for which Nixe lacks faithful semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnsupportedServiceOperation {
    Connect {
        name: Box<[u8]>,
    },
    Command {
        service: &'static str,
        command_id: u32,
    },
    CommandVariant {
        service: &'static str,
        command_id: u32,
        detail: &'static str,
    },
    CommandSizeLimitExceeded {
        service: &'static str,
        command_id: u32,
        operation: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for UnsupportedServiceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect { name } => write!(
                formatter,
                "Horizon service is not implemented: name={:?}",
                String::from_utf8_lossy(name)
            ),
            Self::Command {
                service,
                command_id,
            } => write!(
                formatter,
                "Horizon service command is not implemented: service={service} command={command_id}"
            ),
            Self::CommandVariant {
                service,
                command_id,
                detail,
            } => write!(
                formatter,
                "Horizon service command variant is not implemented: service={service} command={command_id} detail={detail}"
            ),
            Self::CommandSizeLimitExceeded {
                service,
                command_id,
                operation,
                requested,
                limit,
            } => write!(
                formatter,
                "Horizon service command exceeds Nixe's implemented size bound: service={service} command={command_id} operation={operation} requested={requested} ({requested:#x}) limit={limit} ({limit:#x})"
            ),
        }
    }
}

pub(crate) fn unsupported_service_command<T>(
    service: &'static str,
    command_id: u32,
) -> Result<T, IpcWireError> {
    Err(IpcWireError::UnsupportedService(
        UnsupportedServiceOperation::Command {
            service,
            command_id,
        },
    ))
}

impl From<MessageError> for IpcWireError {
    fn from(error: MessageError) -> Self {
        Self::Internal(error.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unimplemented_service_command_is_a_typed_host_fault() {
        assert_eq!(
            unsupported_service_command::<()>("IExample", 77),
            Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::Command {
                    service: "IExample",
                    command_id: 77,
                }
            ))
        );
    }

    #[test]
    fn unimplemented_command_variants_are_typed_host_faults() {
        let operation = UnsupportedServiceOperation::CommandVariant {
            service: "IGraphicBufferProducer",
            command_id: 99,
            detail: "unsupported transaction",
        };
        assert_eq!(
            operation.to_string(),
            "Horizon service command variant is not implemented: service=IGraphicBufferProducer command=99 detail=unsupported transaction"
        );
    }

    #[test]
    fn command_size_limits_report_requested_and_supported_sizes() {
        let operation = UnsupportedServiceOperation::CommandSizeLimitExceeded {
            service: "IStorage",
            command_id: 0,
            operation: "read",
            requested: 0x20_0000,
            limit: 0x10_0000,
        };
        assert_eq!(
            operation.to_string(),
            "Horizon service command exceeds Nixe's implemented size bound: service=IStorage command=0 operation=read requested=2097152 (0x200000) limit=1048576 (0x100000)"
        );
    }
}
