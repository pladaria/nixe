//! Verified Horizon supervisor-call number registry.
//!
//! The registry follows the public Switchbrew SVC table at
//! <https://switchbrew.org/w/index.php?title=SVC&oldid=14679>, corroborated by
//! libnx's public syscall declarations at
//! <https://github.com/switchbrew/libnx/blob/master/nx/include/switch/kernel/svc.h>.
//! Only the original Nintendo Switch Horizon profile is represented here;
//! entries explicitly documented only for later hardware are not silently
//! treated as compatible.

use std::fmt::{Display, Formatter};

/// One verified Horizon SVC number and all documented meanings for that ID.
///
/// Most numbers have exactly one name. A few numbers were reassigned between
/// Horizon versions. Those descriptors retain every documented name so a
/// future version-aware dispatcher must select semantics explicitly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HorizonSvcDescriptor {
    immediate: u32,
    documented_names: &'static [&'static str],
}

impl HorizonSvcDescriptor {
    const fn new(immediate: u32, documented_names: &'static [&'static str]) -> Self {
        Self {
            immediate,
            documented_names,
        }
    }

    /// Returns the immediate encoded by the guest SVC instruction.
    #[must_use]
    pub const fn immediate(self) -> u32 {
        self.immediate
    }

    /// Returns every publicly documented meaning for this number.
    #[must_use]
    pub const fn documented_names(self) -> &'static [&'static str] {
        self.documented_names
    }

    /// Returns the unambiguous operation name, or `None` for an ID whose
    /// meaning depends on the emulated Horizon version.
    #[must_use]
    pub const fn unambiguous_name(self) -> Option<&'static str> {
        if self.documented_names.len() == 1 {
            Some(self.documented_names[0])
        } else {
            None
        }
    }
}

/// Structured result for an immediate absent from the verified registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UnsupportedHorizonSvc {
    immediate: u32,
}

impl UnsupportedHorizonSvc {
    /// Returns the original immediate without truncation or reinterpretation.
    #[must_use]
    pub const fn immediate(self) -> u32 {
        self.immediate
    }
}

impl Display for UnsupportedHorizonSvc {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unsupported Horizon SVC immediate {:#x}",
            self.immediate
        )
    }
}

impl std::error::Error for UnsupportedHorizonSvc {}

/// Decodes a guest SVC immediate using the verified, sorted registry.
///
/// Missing IDs are never inferred from neighboring entries. The returned
/// error preserves the complete architectural immediate for diagnostics.
pub fn decode_horizon_svc(
    immediate: u32,
) -> Result<&'static HorizonSvcDescriptor, UnsupportedHorizonSvc> {
    HORIZON_SVC_REGISTRY
        .binary_search_by_key(&immediate, |descriptor| descriptor.immediate)
        .map(|index| &HORIZON_SVC_REGISTRY[index])
        .map_err(|_| UnsupportedHorizonSvc { immediate })
}

macro_rules! svc {
    ($immediate:literal, $name:literal) => {
        HorizonSvcDescriptor::new($immediate, &[$name])
    };
    ($immediate:literal, $($name:literal),+ $(,)?) => {
        HorizonSvcDescriptor::new($immediate, &[$($name),+])
    };
}

/// Complete verified Switch 1 Horizon SVC registry, sorted by immediate.
pub static HORIZON_SVC_REGISTRY: &[HorizonSvcDescriptor] = &[
    svc!(0x01, "SetHeapSize"),
    svc!(0x02, "SetMemoryPermission"),
    svc!(0x03, "SetMemoryAttribute"),
    svc!(0x04, "MapMemory"),
    svc!(0x05, "UnmapMemory"),
    svc!(0x06, "QueryMemory"),
    svc!(0x07, "ExitProcess"),
    svc!(0x08, "CreateThread"),
    svc!(0x09, "StartThread"),
    svc!(0x0a, "ExitThread"),
    svc!(0x0b, "SleepThread"),
    svc!(0x0c, "GetThreadPriority"),
    svc!(0x0d, "SetThreadPriority"),
    svc!(0x0e, "GetThreadCoreMask"),
    svc!(0x0f, "SetThreadCoreMask"),
    svc!(0x10, "GetCurrentProcessorNumber"),
    svc!(0x11, "SignalEvent"),
    svc!(0x12, "ClearEvent"),
    svc!(0x13, "MapSharedMemory"),
    svc!(0x14, "UnmapSharedMemory"),
    svc!(0x15, "CreateTransferMemory"),
    svc!(0x16, "CloseHandle"),
    svc!(0x17, "ResetSignal"),
    svc!(0x18, "WaitSynchronization"),
    svc!(0x19, "CancelSynchronization"),
    svc!(0x1a, "ArbitrateLock"),
    svc!(0x1b, "ArbitrateUnlock"),
    svc!(0x1c, "WaitProcessWideKeyAtomic"),
    svc!(0x1d, "SignalProcessWideKey"),
    svc!(0x1e, "GetSystemTick"),
    svc!(0x1f, "ConnectToNamedPort"),
    svc!(0x20, "SendSyncRequestLight"),
    svc!(0x21, "SendSyncRequest"),
    svc!(0x22, "SendSyncRequestWithUserBuffer"),
    svc!(0x23, "SendAsyncRequestWithUserBuffer"),
    svc!(0x24, "GetProcessId"),
    svc!(0x25, "GetThreadId"),
    svc!(0x26, "Break"),
    svc!(0x27, "OutputDebugString"),
    svc!(0x28, "ReturnFromException"),
    svc!(0x29, "GetInfo"),
    svc!(0x2a, "FlushEntireDataCache"),
    svc!(0x2b, "FlushDataCache"),
    svc!(0x2c, "MapPhysicalMemory"),
    svc!(0x2d, "UnmapPhysicalMemory"),
    svc!(0x2e, "GetFutureThreadInfo", "GetDebugFutureThreadInfo"),
    svc!(0x2f, "GetLastThreadInfo"),
    svc!(0x30, "GetResourceLimitLimitValue"),
    svc!(0x31, "GetResourceLimitCurrentValue"),
    svc!(0x32, "SetThreadActivity"),
    svc!(0x33, "GetThreadContext3"),
    svc!(0x34, "WaitForAddress"),
    svc!(0x35, "SignalToAddress"),
    svc!(0x36, "SynchronizePreemptionState"),
    svc!(0x37, "GetResourceLimitPeakValue"),
    svc!(0x39, "CreateIoPool"),
    svc!(0x3a, "CreateIoRegion"),
    svc!(0x3c, "DumpInfo", "KernelDebug"),
    svc!(0x3d, "ChangeKernelTraceState"),
    svc!(0x40, "CreateSession"),
    svc!(0x41, "AcceptSession"),
    svc!(0x42, "ReplyAndReceiveLight"),
    svc!(0x43, "ReplyAndReceive"),
    svc!(0x44, "ReplyAndReceiveWithUserBuffer"),
    svc!(0x45, "CreateEvent"),
    svc!(0x46, "MapIoRegion"),
    svc!(0x47, "UnmapIoRegion"),
    svc!(0x48, "MapPhysicalMemoryUnsafe"),
    svc!(0x49, "UnmapPhysicalMemoryUnsafe"),
    svc!(0x4a, "SetUnsafeLimit"),
    svc!(0x4b, "CreateCodeMemory"),
    svc!(0x4c, "ControlCodeMemory"),
    svc!(0x4d, "SleepSystem"),
    svc!(0x4e, "ReadWriteRegister"),
    svc!(0x4f, "SetProcessActivity"),
    svc!(0x50, "CreateSharedMemory"),
    svc!(0x51, "MapTransferMemory"),
    svc!(0x52, "UnmapTransferMemory"),
    svc!(0x53, "CreateInterruptEvent"),
    svc!(0x54, "QueryPhysicalAddress"),
    svc!(0x55, "QueryIoMapping", "QueryMemoryMapping"),
    svc!(0x56, "CreateDeviceAddressSpace"),
    svc!(0x57, "AttachDeviceAddressSpace"),
    svc!(0x58, "DetachDeviceAddressSpace"),
    svc!(0x59, "MapDeviceAddressSpaceByForce"),
    svc!(0x5a, "MapDeviceAddressSpaceAligned"),
    svc!(0x5b, "MapDeviceAddressSpace"),
    svc!(0x5c, "UnmapDeviceAddressSpace"),
    svc!(0x5d, "InvalidateProcessDataCache"),
    svc!(0x5e, "StoreProcessDataCache"),
    svc!(0x5f, "FlushProcessDataCache"),
    svc!(0x60, "DebugActiveProcess"),
    svc!(0x61, "BreakDebugProcess"),
    svc!(0x62, "TerminateDebugProcess"),
    svc!(0x63, "GetDebugEvent"),
    svc!(0x64, "ContinueDebugEvent"),
    svc!(0x65, "GetProcessList"),
    svc!(0x66, "GetThreadList"),
    svc!(0x67, "GetDebugThreadContext"),
    svc!(0x68, "SetDebugThreadContext"),
    svc!(0x69, "QueryDebugProcessMemory"),
    svc!(0x6a, "ReadDebugProcessMemory"),
    svc!(0x6b, "WriteDebugProcessMemory"),
    svc!(0x6c, "SetHardwareBreakPoint"),
    svc!(0x6d, "GetDebugThreadParam"),
    svc!(0x6f, "GetSystemInfo"),
    svc!(0x70, "CreatePort"),
    svc!(0x71, "ManageNamedPort"),
    svc!(0x72, "ConnectToPort"),
    svc!(0x73, "SetProcessMemoryPermission"),
    svc!(0x74, "MapProcessMemory"),
    svc!(0x75, "UnmapProcessMemory"),
    svc!(0x76, "QueryProcessMemory"),
    svc!(0x77, "MapProcessCodeMemory"),
    svc!(0x78, "UnmapProcessCodeMemory"),
    svc!(0x79, "CreateProcess"),
    svc!(0x7a, "StartProcess"),
    svc!(0x7b, "TerminateProcess"),
    svc!(0x7c, "GetProcessInfo"),
    svc!(0x7d, "CreateResourceLimit"),
    svc!(0x7e, "SetResourceLimitLimitValue"),
    svc!(0x7f, "CallSecureMonitor"),
    svc!(0x90, "MapInsecurePhysicalMemory"),
    svc!(0x91, "UnmapInsecurePhysicalMemory"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_representative_documented_supervisor_calls() {
        let cases = [
            (0x01, "SetHeapSize"),
            (0x07, "ExitProcess"),
            (0x18, "WaitSynchronization"),
            (0x21, "SendSyncRequest"),
            (0x45, "CreateEvent"),
            (0x7f, "CallSecureMonitor"),
            (0x91, "UnmapInsecurePhysicalMemory"),
        ];

        for (immediate, expected_name) in cases {
            let decoded = decode_horizon_svc(immediate).unwrap();
            assert_eq!(decoded.immediate(), immediate);
            assert_eq!(decoded.unambiguous_name(), Some(expected_name));
        }
    }

    #[test]
    fn retains_version_dependent_names_without_selecting_one() {
        let cases = [
            (
                0x2e,
                &["GetFutureThreadInfo", "GetDebugFutureThreadInfo"][..],
            ),
            (0x3c, &["DumpInfo", "KernelDebug"][..]),
            (0x55, &["QueryIoMapping", "QueryMemoryMapping"][..]),
        ];

        for (immediate, expected_names) in cases {
            let decoded = decode_horizon_svc(immediate).unwrap();
            assert_eq!(decoded.documented_names(), expected_names);
            assert_eq!(decoded.unambiguous_name(), None);
        }
    }

    #[test]
    fn registry_is_strictly_sorted_and_contains_no_empty_names() {
        assert!(
            HORIZON_SVC_REGISTRY
                .windows(2)
                .all(|pair| { pair[0].immediate() < pair[1].immediate() })
        );
        assert!(HORIZON_SVC_REGISTRY.iter().all(|descriptor| {
            !descriptor.documented_names().is_empty()
                && descriptor
                    .documented_names()
                    .iter()
                    .all(|name| !name.is_empty())
        }));
    }

    #[test]
    fn unknown_immediates_return_a_structured_unsupported_result() {
        for immediate in [0, 0x38, 0x3b, 0x3e, 0x6e, 0x80, 0xff, 0x1_0000] {
            let error = decode_horizon_svc(immediate).unwrap_err();
            assert_eq!(error.immediate(), immediate);
            assert_eq!(
                error.to_string(),
                format!("unsupported Horizon SVC immediate {immediate:#x}")
            );
        }
    }
}
