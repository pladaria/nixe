//! Minimal verified Horizon SVC semantics for interpreter bring-up.
//!
//! ABI layouts and result values follow the public Switchbrew SVC revision
//! linked from [`crate::svc`]. Operations needing a scheduler or HIPC wire
//! transport remain explicit unsupported semantics rather than approximations.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use nixe_cpu::address::GuestVirtualAddress;
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::memory::{
    DataAccessFault, MemoryAccess, MemoryAccessSize, MemoryAttributes, MemoryMappingError,
    MemoryMappingErrorReason, MemoryMappingPurpose, MemoryPermissions, MemoryProtectionError,
    MemoryProtectionErrorReason, MemoryRegionKind, MemoryValue,
};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::state::a32::A32GeneralRegister;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_runtime::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatchRequest,
    ExceptionDispatcher, ExceptionResume, ExceptionTerminationReason, ExceptionTerminationScope,
    HandleTable, ReadableEventObject, SessionObject, ThreadObject, WritableEventObject,
};

use crate::ipc_wire::{IpcWireError, NamedPortResult, SyncRequestResult};
use crate::{UnsupportedHorizonSvc, decode_horizon_svc};

pub const CURRENT_THREAD_HANDLE: u32 = 0xffff_8000;
pub const CURRENT_PROCESS_HANDLE: u32 = 0xffff_8001;
pub const MAX_WAIT_HANDLES: u32 = 0x40;
const HORIZON_HEAP_ALIGNMENT: u64 = 0x20_0000;
const HORIZON_MAX_HEAP_SIZE: u64 = 0x1_0000_0000;

/// Verified guest-visible kernel results used by the implemented subset.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct HorizonKernelResult(u32);

impl HorizonKernelResult {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_IMPLEMENTED: Self = Self(0x4201);
    pub const THREAD_TERMINATING: Self = Self(0x7601);
    pub const INVALID_HANDLE: Self = Self(0xe401);
    pub const INVALID_POINTER: Self = Self(0xe601);
    pub const INVALID_ADDRESS: Self = Self::INVALID_POINTER;
    pub const INVALID_SIZE: Self = Self(0xca01);
    pub const TIMED_OUT: Self = Self(0xea01);
    pub const CANCELLED: Self = Self(0xec01);
    pub const OUT_OF_RANGE: Self = Self(0xee01);
    pub const INVALID_STATE: Self = Self(0xfa01);
    pub const RESOURCE_LIMIT: Self = Self(0x10801);
    pub const NOT_SUPPORTED: Self = Self(0xfe01);
    pub const NOT_FOUND: Self = Self(0xf201);
    pub const OUT_OF_HANDLES: Self = Self(0xd201);

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

fn set_heap_size(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let new_size = read_register(context.thread().state(), 1);
    let layout = context.process().memory_layout();
    if !new_size.is_multiple_of(HORIZON_HEAP_ALIGNMENT)
        || new_size > HORIZON_MAX_HEAP_SIZE
        || new_size > layout.heap().size()
    {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if context
        .process()
        .used_memory_size()
        .saturating_sub(context.process().heap_size())
        .saturating_add(new_size)
        > layout.memory_capacity()
    {
        result(context, HorizonKernelResult::RESOURCE_LIMIT);
        return resume();
    }
    let old_size = context.process().heap_size();
    match context.process().memory().resize_zeroed_mapping(
        context.process().cpu().address_space_id(),
        layout.heap().base(),
        old_size,
        new_size,
        MemoryPermissions::READ_WRITE,
        MemoryMappingPurpose::Heap,
    ) {
        Ok(()) => {
            context.process_mut().set_heap_size(new_size);
            result(context, HorizonKernelResult::SUCCESS);
            write_register(
                context.thread_mut().state_mut(),
                1,
                layout.heap().base().get(),
            );
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryMapping { fault }),
    }
}

/// Host-side reason an SVC could not be given faithful guest semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HorizonSvcFault {
    NotSupervisorCall,
    MissingImmediate,
    Unknown(UnsupportedHorizonSvc),
    UnsupportedSemantics {
        immediate: u32,
        documented_name: &'static str,
    },
    GuestMemory {
        immediate: u32,
        fault: DataAccessFault,
    },
    InvalidMemoryPermission {
        raw: u32,
    },
    InvalidMemoryAttribute {
        mask: u32,
        value: u32,
    },
    InvalidMemoryState {
        immediate: u32,
        address: GuestVirtualAddress,
        purpose: MemoryMappingPurpose,
    },
    MemoryProtection {
        fault: MemoryProtectionError,
    },
    MemoryMapping {
        fault: MemoryMappingError,
    },
    MalformedIpc {
        immediate: u32,
        reason: &'static str,
    },
}

impl HorizonSvcFault {
    /// Returns the stable Horizon result exposed for a recoverable runtime
    /// rejection, or `None` when the exception-routing contract itself failed.
    #[must_use]
    pub const fn guest_result(&self) -> Option<HorizonKernelResult> {
        match self {
            Self::Unknown(_) => Some(HorizonKernelResult::NOT_SUPPORTED),
            Self::UnsupportedSemantics { .. } => Some(HorizonKernelResult::NOT_IMPLEMENTED),
            Self::GuestMemory { .. } => Some(HorizonKernelResult::INVALID_POINTER),
            Self::InvalidMemoryPermission { .. }
            | Self::InvalidMemoryAttribute { .. }
            | Self::InvalidMemoryState { .. } => Some(HorizonKernelResult::INVALID_STATE),
            Self::MemoryProtection { fault } => Some(match fault.reason {
                MemoryProtectionErrorReason::InvalidRange
                | MemoryProtectionErrorReason::Unmapped => HorizonKernelResult::INVALID_ADDRESS,
                MemoryProtectionErrorReason::WritableExecutable => {
                    HorizonKernelResult::INVALID_STATE
                }
                MemoryProtectionErrorReason::PermissionLocked => HorizonKernelResult::INVALID_STATE,
            }),
            Self::MemoryMapping { fault } => Some(match fault.reason {
                MemoryMappingErrorReason::InvalidRange
                | MemoryMappingErrorReason::AlreadyMapped
                | MemoryMappingErrorReason::MappingStateMismatch => {
                    HorizonKernelResult::INVALID_ADDRESS
                }
                MemoryMappingErrorReason::WritableExecutable => HorizonKernelResult::INVALID_STATE,
                MemoryMappingErrorReason::ResourceExhausted => HorizonKernelResult::RESOURCE_LIMIT,
            }),
            Self::MalformedIpc { .. } => Some(HorizonKernelResult::INVALID_STATE),
            Self::NotSupervisorCall | Self::MissingImmediate => None,
        }
    }
}

impl Display for HorizonSvcFault {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSupervisorCall => formatter.write_str("exception is not a supervisor call"),
            Self::MissingImmediate => formatter.write_str("supervisor call has no immediate"),
            Self::Unknown(error) => error.fmt(formatter),
            Self::UnsupportedSemantics {
                immediate,
                documented_name,
            } => write!(
                formatter,
                "Horizon SVC {immediate:#x} ({documented_name}) has no runtime semantics"
            ),
            Self::GuestMemory { immediate, fault } => {
                write!(
                    formatter,
                    "Horizon SVC {immediate:#x} guest-memory fault: {fault:?}"
                )
            }
            Self::InvalidMemoryPermission { raw } => {
                write!(formatter, "invalid Horizon memory permission {raw:#x}")
            }
            Self::InvalidMemoryAttribute { mask, value } => write!(
                formatter,
                "invalid Horizon memory attribute mask={mask:#x} value={value:#x}"
            ),
            Self::InvalidMemoryState {
                immediate,
                address,
                purpose,
            } => write!(
                formatter,
                "Horizon SVC {immediate:#x} rejects mapping at {address} with purpose {purpose:?}"
            ),
            Self::MemoryProtection { fault } => {
                write!(formatter, "Horizon memory protection failed: {fault:?}")
            }
            Self::MemoryMapping { fault } => {
                write!(formatter, "Horizon memory mapping failed: {fault:?}")
            }
            Self::MalformedIpc { immediate, reason } => {
                write!(
                    formatter,
                    "Horizon SVC {immediate:#x} rejected malformed IPC: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for HorizonSvcFault {}

/// One bounded aggregate used to prioritize later SVC implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HorizonSvcCoverageEntry {
    pub immediate: u32,
    pub calls: u64,
    pub support: HorizonSvcSupport,
    pub resumed: u64,
    pub retried: u64,
    pub suspended: u64,
    pub rejected: u64,
    pub terminated: u64,
    pub faulted: u64,
}

/// Fidelity of the currently implemented semantic surface for one SVC.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HorizonSvcSupport {
    Unsupported,
    Partial,
    Complete,
}

/// Table-driven Horizon exception dispatcher for the current minimal subset.
#[derive(Debug, Default)]
pub struct HorizonSvcDispatcher {
    observed: BTreeMap<u32, HorizonSvcCoverageCounts>,
    unknown_calls: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct HorizonSvcCoverageCounts {
    calls: u64,
    resumed: u64,
    retried: u64,
    suspended: u64,
    rejected: u64,
    terminated: u64,
    faulted: u64,
}

impl HorizonSvcDispatcher {
    #[must_use]
    pub fn coverage(&self) -> Vec<HorizonSvcCoverageEntry> {
        self.observed
            .iter()
            .map(|(&immediate, &counts)| HorizonSvcCoverageEntry {
                immediate,
                calls: counts.calls,
                support: svc_support(immediate),
                resumed: counts.resumed,
                retried: counts.retried,
                suspended: counts.suspended,
                rejected: counts.rejected,
                terminated: counts.terminated,
                faulted: counts.faulted,
            })
            .collect()
    }

    #[must_use]
    pub const fn unknown_calls(&self) -> u64 {
        self.unknown_calls
    }

    fn observe(&mut self, immediate: u32, outcome: &ExceptionDispatchOutcome<HorizonSvcFault>) {
        let counts = self.observed.entry(immediate).or_default();
        counts.calls = counts.calls.saturating_add(1);
        match outcome {
            ExceptionDispatchOutcome::Resume(ExceptionResume::Retry) => {
                counts.retried = counts.retried.saturating_add(1);
            }
            ExceptionDispatchOutcome::Resume(_) => {
                counts.resumed = counts.resumed.saturating_add(1);
            }
            ExceptionDispatchOutcome::Suspend(_) => {
                counts.suspended = counts.suspended.saturating_add(1);
            }
            ExceptionDispatchOutcome::Reject { .. } => {
                counts.rejected = counts.rejected.saturating_add(1);
            }
            ExceptionDispatchOutcome::Terminate { .. } => {
                counts.terminated = counts.terminated.saturating_add(1);
            }
            ExceptionDispatchOutcome::Fault(_) => {
                counts.faulted = counts.faulted.saturating_add(1);
            }
        }
    }
}

impl ExceptionDispatcher for HorizonSvcDispatcher {
    type Fault = HorizonSvcFault;

    fn dispatch(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault> {
        if request.kind() != ExceptionKind::SupervisorCall {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::NotSupervisorCall);
        }
        let Some(immediate) = request
            .syndrome()
            .and_then(|value| u32::try_from(value).ok())
        else {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::MissingImmediate);
        };
        let descriptor = match decode_horizon_svc(immediate) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                self.unknown_calls = self.unknown_calls.saturating_add(1);
                return reject(context, HorizonSvcFault::Unknown(error));
            }
        };

        let outcome = match immediate {
            0x01 => set_heap_size(context),
            0x02 => set_memory_permission(context),
            0x03 => set_memory_attribute(context),
            0x06 => query_memory(context, immediate),
            0x07 => terminate(ExceptionTerminationScope::Process),
            0x0a => terminate(ExceptionTerminationScope::CurrentThread),
            0x10 => {
                write_register(context.thread_mut().state_mut(), 0, 0);
                resume()
            }
            0x11 => event_signal(context),
            0x12 => event_clear(context),
            0x16 => close_handle(context),
            0x17 => reset_signal(context),
            0x18 => wait_synchronization(context),
            0x1f => connect_to_named_port(context),
            0x21 => send_sync_request(context),
            0x24 => get_process_id(context),
            0x25 => get_thread_id(context),
            0x26 => break_process(context),
            0x29 => get_info(context),
            0x40 => create_session(context),
            0x45 => create_event(context),
            _ => reject(
                context,
                HorizonSvcFault::UnsupportedSemantics {
                    immediate,
                    documented_name: descriptor
                        .unambiguous_name()
                        .unwrap_or("version-dependent SVC"),
                },
            ),
        };
        self.observe(immediate, &outcome);
        outcome
    }
}

const fn svc_support(immediate: u32) -> HorizonSvcSupport {
    match immediate {
        0x07 | 0x0a | 0x10 | 0x16 | 0x25 | 0x45 => HorizonSvcSupport::Complete,
        0x01 | 0x02 | 0x03 | 0x06 | 0x11 | 0x12 | 0x17 | 0x18 | 0x24 | 0x26 | 0x29 | 0x40 => {
            HorizonSvcSupport::Partial
        }
        0x1f | 0x21 => HorizonSvcSupport::Partial,
        _ => HorizonSvcSupport::Unsupported,
    }
}

fn connect_to_named_port(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let name = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    match crate::ipc_wire::connect_to_named_port(context.process_mut(), name) {
        Ok(NamedPortResult::Connected(handle)) => {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
            resume()
        }
        Ok(NamedPortResult::NotFound) => {
            result(context, HorizonKernelResult::NOT_FOUND);
            resume()
        }
        Ok(NamedPortResult::NameOutOfRange) => {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            resume()
        }
        Ok(NamedPortResult::OutOfHandles) => {
            result(context, HorizonKernelResult::OUT_OF_HANDLES);
            resume()
        }
        Err(error) => reject_ipc(context, 0x1f, error),
    }
}

fn send_sync_request(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let tls = match context.thread().state() {
        ThreadCpuState::A64(state) => GuestVirtualAddress::new(state.tpidr_el0()),
        ThreadCpuState::A32(state) => GuestVirtualAddress::new(u64::from(state.tpidrurw())),
    };
    match crate::ipc_wire::send_sync_request(context.process_mut(), tls, handle) {
        Ok(SyncRequestResult::Success) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Ok(SyncRequestResult::InvalidHandle) => {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            resume()
        }
        Err(error) => reject_ipc(context, 0x21, error),
    }
}

fn reject_ipc(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    error: IpcWireError,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    match error {
        IpcWireError::GuestMemory(fault) => {
            reject(context, HorizonSvcFault::GuestMemory { immediate, fault })
        }
        IpcWireError::Malformed(reason) => {
            reject(context, HorizonSvcFault::MalformedIpc { immediate, reason })
        }
    }
}

fn get_info(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let info_type = read_register(context.thread().state(), 1) as u32;
    let handle = read_register(context.thread().state(), 2) as u32;
    let subtype = read_register(context.thread().state(), 3);
    if handle != CURRENT_PROCESS_HANDLE || subtype != 0 {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let layout = context.process().memory_layout();
    let value = match info_type {
        2 => layout.alias().base().get(),
        3 => layout.alias().size(),
        4 => layout.heap().base().get(),
        5 => layout.heap().size(),
        12 => layout.aslr().base().get(),
        13 => layout.aslr().size(),
        14 => layout.stack().base().get(),
        15 => layout.stack().size(),
        6 => layout.memory_capacity(),
        7 => context.process().used_memory_size(),
        28 => 0,
        _ => {
            return reject(
                context,
                HorizonSvcFault::UnsupportedSemantics {
                    immediate: 0x29,
                    documented_name: "GetInfo",
                },
            );
        }
    };
    result(context, HorizonKernelResult::SUCCESS);
    write_u64(context.thread_mut().state_mut(), 1, value);
    resume()
}

fn terminate(scope: ExceptionTerminationScope) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    ExceptionDispatchOutcome::Terminate {
        scope,
        exit_code: 0,
        reason: ExceptionTerminationReason::Requested,
    }
}

fn break_process(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let reason = read_register(context.thread().state(), 0);
    let info = read_register(context.thread().state(), 1);
    let size = read_register(context.thread().state(), 2);
    if reason & 0x8000_0000 != 0 {
        result(context, HorizonKernelResult::SUCCESS);
        return resume();
    }
    ExceptionDispatchOutcome::Terminate {
        scope: ExceptionTerminationScope::Process,
        exit_code: reason,
        reason: ExceptionTerminationReason::Break { reason, info, size },
    }
}

fn set_memory_permission(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    let raw = read_register(context.thread().state(), 2) as u32;
    let permissions = match raw {
        0 => MemoryPermissions::NONE,
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        _ => return reject(context, HorizonSvcFault::InvalidMemoryPermission { raw }),
    };
    let end = start.get().checked_add(size);
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    let valid_range = query.is_some_and(|query| {
        query.purpose.allows_reprotect()
            && query.base.get() <= start.get()
            && end.is_some_and(|end| query.base.get().saturating_add(query.size) >= end)
    });
    if !valid_range {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryState {
                immediate: 0x02,
                address: start,
                purpose: query.map_or(MemoryMappingPurpose::Normal, |query| query.purpose),
            },
        );
    }
    match context.process().memory().set_permissions(
        context.process().cpu().address_space_id(),
        start,
        size,
        permissions,
    ) {
        Ok(()) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryProtection { fault }),
    }
}

fn set_memory_attribute(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    let raw_mask = read_register(context.thread().state(), 2) as u32;
    let raw_value = read_register(context.thread().state(), 3) as u32;
    let uncached = MemoryAttributes::UNCACHED.bits();
    let permission_locked = MemoryAttributes::PERMISSION_LOCKED.bits();
    let valid_update = (raw_mask == uncached && raw_value & !uncached == 0)
        || (raw_mask == permission_locked && raw_value == permission_locked);
    if !valid_update {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryAttribute {
                mask: raw_mask,
                value: raw_value,
            },
        );
    }
    let (Some(mask), Some(value)) = (
        MemoryAttributes::from_bits(raw_mask),
        MemoryAttributes::from_bits(raw_value),
    ) else {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryAttribute {
                mask: raw_mask,
                value: raw_value,
            },
        );
    };
    let end = start.get().checked_add(size);
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    let valid_range = query.is_some_and(|query| {
        query.purpose.allows_attribute_change()
            && query.base.get() <= start.get()
            && end.is_some_and(|end| query.base.get().saturating_add(query.size) >= end)
    });
    if !valid_range {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryState {
                immediate: 0x03,
                address: start,
                purpose: query.map_or(MemoryMappingPurpose::Normal, |query| query.purpose),
            },
        );
    }
    match context.process().memory().set_attributes(
        context.process().cpu().address_space_id(),
        start,
        size,
        mask,
        value,
    ) {
        Ok(()) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryProtection { fault }),
    }
}

fn resume() -> ExceptionDispatchOutcome<HorizonSvcFault> {
    ExceptionDispatchOutcome::Resume(ExceptionResume::Next)
}

fn reject(
    context: &mut ExceptionDispatchContext<'_>,
    diagnostic: HorizonSvcFault,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let guest_result = diagnostic
        .guest_result()
        .expect("only recoverable Horizon failures may reject a guest operation");
    result(context, guest_result);
    ExceptionDispatchOutcome::Reject { diagnostic }
}

fn result(context: &mut ExceptionDispatchContext<'_>, value: HorizonKernelResult) {
    write_register(context.thread_mut().state_mut(), 0, u64::from(value.raw()));
}

fn read_register(state: &ThreadCpuState, index: u8) -> u64 {
    match state {
        ThreadCpuState::A64(state) => state.read_x(A64Register::General(
            A64GeneralRegister::new(index).expect("A64 ABI register index is valid"),
        )),
        ThreadCpuState::A32(state) => {
            u64::from(state.read_r(
                A32GeneralRegister::new(index).expect("AArch32 ABI register index is valid"),
            ))
        }
    }
}

fn write_register(state: &mut ThreadCpuState, index: u8, value: u64) {
    match state {
        ThreadCpuState::A64(state) => state.write_x(
            A64Register::General(
                A64GeneralRegister::new(index).expect("A64 ABI register index is valid"),
            ),
            value,
        ),
        ThreadCpuState::A32(state) => state.write_r(
            A32GeneralRegister::new(index).expect("AArch32 ABI register index is valid"),
            value as u32,
        ),
    }
}

fn write_u64(state: &mut ThreadCpuState, index: u8, value: u64) {
    match state {
        ThreadCpuState::A64(_) => write_register(state, index, value),
        ThreadCpuState::A32(state) => {
            // The Horizon AArch32 ABI returns 64-bit IDs in consecutive
            // low/high register pairs, for example R1:R2.
            state.write_r(
                A32GeneralRegister::new(index).expect("AArch32 ABI register index is valid"),
                value as u32,
            );
            state.write_r(
                A32GeneralRegister::new(index + 1).expect("AArch32 ABI register pair is valid"),
                (value >> 32) as u32,
            );
        }
    }
}

fn read_wait_timeout(state: &ThreadCpuState) -> i64 {
    match state {
        ThreadCpuState::A64(_) => read_register(state, 3) as i64,
        ThreadCpuState::A32(state) => {
            // WaitSynchronization is the exceptional AArch32 layout: the
            // timeout low/high words occupy R0:R3 rather than a pair.
            let low = u64::from(state.read_r(
                A32GeneralRegister::new(0).expect("AArch32 timeout low register is valid"),
            ));
            let high = u64::from(state.read_r(
                A32GeneralRegister::new(3).expect("AArch32 timeout high register is valid"),
            ));
            (low | (high << 32)) as i64
        }
    }
}

fn close_handle(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let code = if matches!(handle, CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE)
        || context.process_mut().handles_mut().close(handle).is_err()
    {
        HorizonKernelResult::INVALID_HANDLE
    } else {
        HorizonKernelResult::SUCCESS
    };
    result(context, code);
    resume()
}

fn event_signal(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let event = context
        .process()
        .handles()
        .get_as::<WritableEventObject>(handle)
        .cloned();
    let code = if let Some(event) = event {
        event.signal();
        HorizonKernelResult::SUCCESS
    } else {
        HorizonKernelResult::INVALID_HANDLE
    };
    result(context, code);
    resume()
}

fn event_clear(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let writable = context
        .process()
        .handles()
        .get_as::<WritableEventObject>(handle)
        .cloned();
    let readable = context
        .process()
        .handles()
        .get_as::<ReadableEventObject>(handle)
        .cloned();
    let code = if let Some(event) = writable {
        event.clear();
        HorizonKernelResult::SUCCESS
    } else if let Some(event) = readable {
        event.clear();
        HorizonKernelResult::SUCCESS
    } else {
        HorizonKernelResult::INVALID_HANDLE
    };
    result(context, code);
    resume()
}

fn reset_signal(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let readable = context
        .process()
        .handles()
        .get_as::<ReadableEventObject>(handle)
        .cloned();
    let code = match readable {
        Some(event) if event.is_signalled() => {
            event.clear();
            HorizonKernelResult::SUCCESS
        }
        Some(_) => HorizonKernelResult::INVALID_STATE,
        None => HorizonKernelResult::INVALID_HANDLE,
    };
    result(context, code);
    resume()
}

fn create_event(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let (writable, readable) = nixe_runtime::EventObject::create_pair();
    match insert_pair(context.process_mut().handles_mut(), writable, readable) {
        Ok((write_handle, read_handle)) => {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, u64::from(write_handle));
            write_register(context.thread_mut().state_mut(), 2, u64::from(read_handle));
        }
        Err(()) => result(context, HorizonKernelResult::RESOURCE_LIMIT),
    }
    resume()
}

fn create_session(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let is_light = read_register(context.thread().state(), 2) as u32;
    if is_light != 0 {
        return reject(
            context,
            HorizonSvcFault::UnsupportedSemantics {
                immediate: 0x40,
                documented_name: "CreateSession(light)",
            },
        );
    }
    let (server, client) = SessionObject::create_pair();
    match insert_pair(context.process_mut().handles_mut(), server, client) {
        Ok((server_handle, client_handle)) => {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(
                context.thread_mut().state_mut(),
                1,
                u64::from(server_handle),
            );
            write_register(
                context.thread_mut().state_mut(),
                2,
                u64::from(client_handle),
            );
        }
        Err(()) => result(context, HorizonKernelResult::RESOURCE_LIMIT),
    }
    resume()
}

fn insert_pair<A, B>(handles: &mut HandleTable, first: A, second: B) -> Result<(u32, u32), ()>
where
    A: nixe_runtime::HandleValue,
    B: nixe_runtime::HandleValue,
{
    let first_handle = handles.insert(first).map_err(|_| ())?;
    match handles.insert(second) {
        Ok(second_handle) => Ok((first_handle, second_handle)),
        Err(_) => {
            let _ = handles.close(first_handle);
            Err(())
        }
    }
}

fn get_process_id(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 1) as u32;
    if handle == CURRENT_PROCESS_HANDLE {
        let process_id = context.process().process_id();
        result(context, HorizonKernelResult::SUCCESS);
        write_u64(context.thread_mut().state_mut(), 1, process_id);
    } else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
    }
    resume()
}

fn get_thread_id(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 1) as u32;
    let thread_id = if handle == CURRENT_THREAD_HANDLE || handle == context.thread().handle() {
        Some(context.thread().object().thread_id())
    } else {
        context
            .process()
            .handles()
            .get_as::<ThreadObject>(handle)
            .map(|thread| thread.thread_id())
    };
    if let Some(thread_id) = thread_id {
        result(context, HorizonKernelResult::SUCCESS);
        write_u64(context.thread_mut().state_mut(), 1, thread_id);
    } else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
    }
    resume()
}

fn wait_synchronization(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let pointer = read_register(context.thread().state(), 1);
    let count = read_register(context.thread().state(), 2) as u32;
    let timeout = read_wait_timeout(context.thread().state());
    if count > MAX_WAIT_HANDLES {
        result(context, HorizonKernelResult::OUT_OF_RANGE);
        return resume();
    }
    let mut handles = Vec::with_capacity(count as usize);
    for index in 0..count {
        let Some(address) = pointer.checked_add(u64::from(index) * 4) else {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        };
        let value = match context.process().memory().read(
            context.process().cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            MemoryAccess::normal(MemoryAccessSize::Word),
        ) {
            Ok(read) => read.value,
            Err(_) => {
                result(context, HorizonKernelResult::INVALID_ADDRESS);
                return resume();
            }
        };
        let MemoryValue::U32(handle) = value else {
            unreachable!("word access returns a word value")
        };
        handles.push(handle);
    }
    for (index, handle) in handles.iter().copied().enumerate() {
        let Some(event) = context
            .process()
            .handles()
            .get_as::<ReadableEventObject>(handle)
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if event.is_signalled() {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, index as u64);
            return resume();
        }
    }
    if timeout == 0 {
        result(context, HorizonKernelResult::TIMED_OUT);
        resume()
    } else {
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
    }
}

fn query_memory(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let output = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let address = GuestVirtualAddress::new(read_register(context.thread().state(), 2));
    let limit = context.process().address_space_limit();
    let Some(query) = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        address,
        GuestVirtualAddress::new(limit),
    ) else {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    };
    let memory_type = match query.region {
        None => 0_u32,
        Some(MemoryRegionKind::Device) => 1,
        Some(MemoryRegionKind::Ram) => match query.purpose {
            MemoryMappingPurpose::Normal => 2,
            MemoryMappingPurpose::CodeStatic => 3,
            MemoryMappingPurpose::CodeMutable => 4,
            MemoryMappingPurpose::ModuleCodeStatic => 8,
            MemoryMappingPurpose::ModuleCodeMutable => 9,
            MemoryMappingPurpose::ThreadLocal => 0x0c,
            MemoryMappingPurpose::Heap => 5,
        },
    };
    let fields = [
        (0_u64, MemoryValue::U64(query.base.get())),
        (8, MemoryValue::U64(query.size)),
        (0x10, MemoryValue::U32(memory_type)),
        (0x14, MemoryValue::U32(query.attributes.bits())),
        (0x18, MemoryValue::U32(u32::from(query.permissions.bits()))),
        (0x1c, MemoryValue::U32(0)),
        (0x20, MemoryValue::U32(0)),
        (0x24, MemoryValue::U32(0)),
    ];
    for (offset, value) in fields {
        let Some(address) = output.checked_add(offset) else {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        };
        let access = MemoryAccess::normal(value.size());
        if let Err(fault) = context.process().memory().write(
            context.process().cpu().address_space_id(),
            address,
            access,
            value,
        ) {
            return reject(context, HorizonSvcFault::GuestMemory { immediate, fault });
        }
    }
    result(context, HorizonKernelResult::SUCCESS);
    write_register(context.thread_mut().state_mut(), 1, 0);
    resume()
}
