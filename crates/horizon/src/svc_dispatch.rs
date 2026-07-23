//! Minimal verified Horizon SVC semantics for interpreter bring-up.
//!
//! ABI layouts and result values follow the public Switchbrew SVC revision
//! linked from [`crate::svc`]. Operations needing a scheduler or HIPC wire
//! transport remain explicit unsupported semantics rather than approximations.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

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
    HandleObject, HandleTable, PortEndpoint, PortError, PortObject, ProcessObject,
    ReadableEventObject, SessionEndpoint, SessionError, SessionMessage, SessionObject,
    SessionRequestOwner, SessionRequestResult, SharedMemoryObject, ThreadObject,
    WritableEventObject,
};

use crate::ipc_message::HipcRequest;
use crate::ipc_wire::{IpcWireError, NamedPortResult, SyncRequestResult};
use crate::{UnsupportedHorizonSvc, decode_horizon_svc};

pub const CURRENT_THREAD_HANDLE: u32 = 0xffff_8000;
pub const CURRENT_PROCESS_HANDLE: u32 = 0xffff_8001;
pub const MAX_WAIT_HANDLES: u32 = 0x40;
const TLS_COMMAND_BUFFER_SIZE: usize = 0x100;
const USER_BUFFER_ALIGNMENT: u64 = 0x1000;
const HORIZON_HEAP_ALIGNMENT: u64 = 0x20_0000;
const HORIZON_MAX_HEAP_SIZE: u64 = 0x1_0000_0000;

/// Verified guest-visible kernel results used by the implemented subset.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct HorizonKernelResult(u32);

impl HorizonKernelResult {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_IMPLEMENTED: Self = Self(0x4201);
    pub const OUT_OF_SESSIONS: Self = Self(0x0e01);
    pub const THREAD_TERMINATING: Self = Self(0x7601);
    pub const INVALID_HANDLE: Self = Self(0xe401);
    pub const INVALID_POINTER: Self = Self(0xe601);
    pub const INVALID_ADDRESS: Self = Self(0xcc01);
    pub const INVALID_SIZE: Self = Self(0xca01);
    pub const INVALID_CURRENT_MEMORY: Self = Self(0xd401);
    pub const OUT_OF_RESOURCE: Self = Self(0xce01);
    pub const TIMED_OUT: Self = Self(0xea01);
    pub const CANCELLED: Self = Self(0xec01);
    pub const OUT_OF_RANGE: Self = Self(0xee01);
    pub const INVALID_STATE: Self = Self(0xfa01);
    pub const RESOURCE_LIMIT: Self = Self(0x10801);
    pub const NOT_SUPPORTED: Self = Self(0xfe01);
    pub const NOT_FOUND: Self = Self(0xf201);
    pub const SESSION_CLOSED: Self = Self(0xf601);
    pub const PORT_CLOSED: Self = Self(0x10601);
    pub const OUT_OF_HANDLES: Self = Self(0xd201);
    pub const INVALID_COMBINATION: Self = Self(0xe801);

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
#[derive(Debug)]
pub struct HorizonSvcDispatcher {
    observed: BTreeMap<u32, HorizonSvcCoverageCounts>,
    unknown_calls: u64,
    initial_operation_mode: crate::OperationMode,
    named_ports: BTreeMap<Vec<u8>, PortObject>,
    reply_sent: BTreeSet<u64>,
    wait_deadlines: BTreeMap<(u64, u32), Instant>,
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
    /// Creates a dispatcher whose applet service reports the selected initial mode.
    #[must_use]
    pub const fn new(initial_operation_mode: crate::OperationMode) -> Self {
        Self {
            observed: BTreeMap::new(),
            unknown_calls: 0,
            initial_operation_mode,
            named_ports: BTreeMap::new(),
            reply_sent: BTreeSet::new(),
            wait_deadlines: BTreeMap::new(),
        }
    }

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
            0x13 => map_shared_memory(context),
            0x14 => unmap_shared_memory(context),
            0x16 => close_handle(context),
            0x17 => reset_signal(context),
            0x18 => wait_synchronization(context),
            0x1f => self.connect_to_named_port(context),
            0x20 => send_sync_request_light(context),
            0x21 => send_sync_request(context, self.initial_operation_mode),
            0x22 => send_sync_request_with_user_buffer(context, self.initial_operation_mode),
            0x24 => get_process_id(context),
            0x25 => get_thread_id(context),
            0x26 => break_process(context),
            0x29 => get_info(context),
            0x40 => create_session(context),
            0x41 => accept_session(context),
            0x42 => self.reply_and_receive_light(context),
            0x43 => self.reply_and_receive(context, false),
            0x44 => self.reply_and_receive(context, true),
            0x45 => create_event(context),
            0x70 => create_port(context),
            0x71 => self.manage_named_port(context),
            0x72 => connect_to_port(context),
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

impl Default for HorizonSvcDispatcher {
    fn default() -> Self {
        Self::new(crate::OperationMode::default())
    }
}

const fn svc_support(immediate: u32) -> HorizonSvcSupport {
    match immediate {
        0x07 | 0x0a | 0x10 | 0x13 | 0x14 | 0x16 | 0x20 | 0x21 | 0x22 | 0x25 | 0x40 | 0x41
        | 0x42 | 0x43 | 0x44 | 0x45 | 0x70 | 0x71 | 0x72 => HorizonSvcSupport::Complete,
        0x01 | 0x02 | 0x03 | 0x06 | 0x11 | 0x12 | 0x17 | 0x18 | 0x24 | 0x26 | 0x29 => {
            HorizonSvcSupport::Partial
        }
        0x1f => HorizonSvcSupport::Complete,
        _ => HorizonSvcSupport::Unsupported,
    }
}

impl HorizonSvcDispatcher {
    fn connect_to_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        let name = match read_c_name(
            context,
            address,
            crate::ipc_wire::NAMED_PORT_NAME_SIZE,
            0x1f,
        ) {
            Ok(Some(name)) => name,
            Ok(None) => {
                result(context, HorizonKernelResult::OUT_OF_RANGE);
                return resume();
            }
            Err(outcome) => return outcome,
        };
        if let Some(port) = self.named_ports.get(&name).cloned() {
            let session = match port.connect() {
                Ok(session) => session,
                Err(PortError::SessionLimit) => {
                    result(context, HorizonKernelResult::OUT_OF_SESSIONS);
                    return resume();
                }
                Err(PortError::PeerClosed) => {
                    result(context, HorizonKernelResult::PORT_CLOSED);
                    return resume();
                }
                Err(PortError::WrongEndpoint | PortError::NoPendingSession) => {
                    result(context, HorizonKernelResult::INVALID_STATE);
                    return resume();
                }
            };
            match context.process_mut().handles_mut().insert(session) {
                Ok(handle) => {
                    result(context, HorizonKernelResult::SUCCESS);
                    write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
                }
                Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
            }
            return resume();
        }
        match crate::ipc_wire::connect_to_named_port(context.process_mut(), address) {
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

    fn manage_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        let max_sessions = read_register(context.thread().state(), 2) as u32 as i32;
        let name = match read_c_name(
            context,
            address,
            crate::ipc_wire::NAMED_PORT_NAME_SIZE,
            0x71,
        ) {
            Ok(Some(name)) => name,
            Ok(None) => {
                result(context, HorizonKernelResult::OUT_OF_RANGE);
                return resume();
            }
            Err(outcome) => return outcome,
        };
        if max_sessions < 0 {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        if max_sessions == 0 {
            if self
                .named_ports
                .get(&name)
                .is_some_and(PortObject::server_is_open)
            {
                result(context, HorizonKernelResult::INVALID_STATE);
            } else if self.named_ports.remove(&name).is_some() {
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, 0);
            } else {
                result(context, HorizonKernelResult::NOT_FOUND);
            }
            return resume();
        }
        if self.named_ports.contains_key(&name) {
            result(context, HorizonKernelResult::INVALID_STATE);
            return resume();
        }
        let (server, client) = PortObject::create_pair(max_sessions as usize, false);
        match context.process_mut().handles_mut().insert(server) {
            Ok(handle) => {
                self.named_ports.insert(name, client);
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
            }
            Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
        }
        resume()
    }
}

fn send_sync_request(
    context: &mut ExceptionDispatchContext<'_>,
    initial_operation_mode: crate::OperationMode,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let tls = match context.thread().state() {
        ThreadCpuState::A64(state) => GuestVirtualAddress::new(state.tpidr_el0()),
        ThreadCpuState::A32(state) => GuestVirtualAddress::new(u64::from(state.tpidrurw())),
    };
    match crate::ipc_wire::send_sync_request(
        context.process_mut(),
        tls,
        handle,
        initial_operation_mode,
    ) {
        Ok(SyncRequestResult::Success) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Ok(SyncRequestResult::InvalidHandle) => {
            generic_sync_request(context, tls, TLS_COMMAND_BUFFER_SIZE, handle, 0x21)
        }
        Err(error) => reject_ipc(context, 0x21, error),
    }
}

fn send_sync_request_with_user_buffer(
    context: &mut ExceptionDispatchContext<'_>,
    initial_operation_mode: crate::OperationMode,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let address = read_register(context.thread().state(), 0);
    let size = read_register(context.thread().state(), 1);
    let handle = read_register(context.thread().state(), 2) as u32;
    // Public ABI and validation reference:
    // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#SendSyncRequestWithUserBuffer
    if !address.is_multiple_of(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    if !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if size == 0 {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if address.checked_add(size).is_none_or(|end| address >= end) {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        return resume();
    }
    let Ok(size) = usize::try_from(size) else {
        result(context, HorizonKernelResult::OUT_OF_RESOURCE);
        return resume();
    };
    let address = GuestVirtualAddress::new(address);
    match crate::ipc_wire::send_sync_request_from_buffer(
        context.process_mut(),
        address,
        size,
        handle,
        initial_operation_mode,
    ) {
        Ok(SyncRequestResult::Success) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Ok(SyncRequestResult::InvalidHandle) => {
            generic_sync_request(context, address, size, handle, 0x22)
        }
        Err(error) => reject_ipc(context, 0x22, error),
    }
}

fn send_sync_request_light(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let Some(session) = context
        .process()
        .handles()
        .get_as::<SessionObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if session.endpoint() != SessionEndpoint::Client || !session.is_light() {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let owner = session_request_owner(context);
    let mut words = [0_u32; 7];
    for (index, word) in words.iter_mut().enumerate() {
        *word = read_register(context.thread().state(), index as u8 + 1) as u32;
    }
    match session.request(owner, SessionMessage::Light(words)) {
        Ok(SessionRequestResult::Submitted | SessionRequestResult::Waiting) => {
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
        }
        Ok(SessionRequestResult::Response(SessionMessage::Light(response))) => {
            for (index, word) in response.into_iter().enumerate() {
                write_register(
                    context.thread_mut().state_mut(),
                    index as u8 + 1,
                    u64::from(word),
                );
            }
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Ok(SessionRequestResult::Response(
            SessionMessage::Buffer(_) | SessionMessage::TransportedBuffer { .. },
        ))
        | Err(SessionError::MessageKindMismatch) => {
            result(context, HorizonKernelResult::INVALID_STATE);
            resume()
        }
        Err(error) => session_error(context, error),
    }
}

fn generic_sync_request(
    context: &mut ExceptionDispatchContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    handle: u32,
    immediate: u32,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let Some(session) = context
        .process()
        .handles()
        .get_as::<SessionObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if session.endpoint() != SessionEndpoint::Client || session.is_light() {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let owner = session_request_owner(context);
    match session.poll_request(owner) {
        Ok(Some(SessionRequestResult::Waiting | SessionRequestResult::Submitted)) => {
            return ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        }
        Ok(Some(SessionRequestResult::Response(response))) => {
            return finish_sync_response(context, address, size, immediate, response);
        }
        Ok(None) => {}
        Err(error) => return session_error(context, error),
    }
    let mut message = Vec::new();
    if message.try_reserve_exact(size).is_err() {
        result(context, HorizonKernelResult::OUT_OF_RESOURCE);
        return resume();
    }
    message.resize(size, 0);
    if let Err(error) = crate::ipc_wire::read_bytes(context.process(), address, &mut message) {
        return reject_ipc(context, immediate, error);
    }
    let message = match capture_message_handles(context, message, false) {
        Ok(message) => message,
        Err(code) => {
            result(context, code);
            return resume();
        }
    };
    match session.request(owner, message) {
        Ok(SessionRequestResult::Submitted | SessionRequestResult::Waiting) => {
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
        }
        Ok(SessionRequestResult::Response(response)) => {
            finish_sync_response(context, address, size, immediate, response)
        }
        Err(SessionError::MessageKindMismatch) => invalid_state(context),
        Err(error) => session_error(context, error),
    }
}

fn finish_sync_response(
    context: &mut ExceptionDispatchContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    immediate: u32,
    response: SessionMessage,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let response = match materialize_message_handles(context, response) {
        Ok(Some(response)) => response,
        Ok(None) => return invalid_state(context),
        Err(code) => {
            result(context, code);
            return resume();
        }
    };
    if response.len() > size {
        close_encoded_handles(context.process_mut().handles_mut(), &response);
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if let Err(error) = crate::ipc_wire::write_bytes(context.process(), address, &response) {
        close_encoded_handles(context.process_mut().handles_mut(), &response);
        return reject_ipc(context, immediate, error);
    }
    result(context, HorizonKernelResult::SUCCESS);
    resume()
}

fn invalid_state(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    result(context, HorizonKernelResult::INVALID_STATE);
    resume()
}

// Handle translation follows the public kernel implementation. Client requests
// may copy handles but may not move them; server replies may do both, and moved
// server handles are consumed even when a later handle makes the reply fail:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_server_session.cpp#L150-L233
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_server_session.cpp#L572-L578
fn capture_message_handles(
    context: &mut ExceptionDispatchContext<'_>,
    bytes: Vec<u8>,
    allow_move_handles: bool,
) -> Result<SessionMessage, HorizonKernelResult> {
    let Some(message) = decode_transport_header(&bytes)? else {
        return Ok(SessionMessage::Buffer(bytes));
    };
    if !allow_move_handles && !message.move_handles.is_empty() {
        return Err(HorizonKernelResult::INVALID_COMBINATION);
    }
    let mut transfer_error = None;
    let mut copy_handles = Vec::with_capacity(message.copy_handles.len());
    for handle in &message.copy_handles {
        if transfer_error.is_some() {
            copy_handles.push(None);
            continue;
        }
        match copy_ipc_object(context, *handle) {
            Ok(object) => copy_handles.push(object),
            Err(error) => {
                transfer_error = Some(error);
                copy_handles.push(None);
            }
        }
    }

    let mut move_handles = Vec::with_capacity(message.move_handles.len());
    for handle in &message.move_handles {
        if *handle == 0 {
            move_handles.push(None);
            continue;
        }
        if matches!(*handle, CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE) {
            transfer_error = Some(HorizonKernelResult::INVALID_HANDLE);
            move_handles.push(None);
            continue;
        }
        match context.process_mut().handles_mut().close(*handle) {
            Ok(object) if transfer_error.is_none() => move_handles.push(Some(object)),
            Ok(_) => move_handles.push(None),
            Err(_) => {
                transfer_error = Some(HorizonKernelResult::INVALID_HANDLE);
                move_handles.push(None);
            }
        }
    }
    if let Some(error) = transfer_error {
        return Err(error);
    }
    Ok(SessionMessage::TransportedBuffer {
        bytes,
        copy_handles,
        move_handles,
    })
}

fn copy_ipc_object(
    context: &ExceptionDispatchContext<'_>,
    handle: u32,
) -> Result<Option<HandleObject>, HorizonKernelResult> {
    match handle {
        0 => Ok(None),
        CURRENT_PROCESS_HANDLE => Ok(Some(HandleObject::new(ProcessObject::new(
            context.process().process_id(),
        )))),
        CURRENT_THREAD_HANDLE => Ok(Some(HandleObject::new(context.thread().object()))),
        _ => context
            .process()
            .handles()
            .get(handle)
            .cloned()
            .map(Some)
            .ok_or(HorizonKernelResult::INVALID_HANDLE),
    }
}

fn materialize_message_handles(
    context: &mut ExceptionDispatchContext<'_>,
    message: SessionMessage,
) -> Result<Option<Vec<u8>>, HorizonKernelResult> {
    materialize_message_handles_in_table(context.process_mut().handles_mut(), message)
}

fn materialize_message_handles_in_table(
    handles: &mut HandleTable,
    message: SessionMessage,
) -> Result<Option<Vec<u8>>, HorizonKernelResult> {
    let (mut bytes, copy_handles, move_handles) = match message {
        SessionMessage::Buffer(bytes) => return Ok(Some(bytes)),
        SessionMessage::Light(_) => return Ok(None),
        SessionMessage::TransportedBuffer {
            bytes,
            copy_handles,
            move_handles,
        } => (bytes, copy_handles, move_handles),
    };
    let handle_offset = {
        let Some(header) = decode_transport_header(&bytes)? else {
            return Err(HorizonKernelResult::INVALID_COMBINATION);
        };
        if header.copy_handles.len() != copy_handles.len()
            || header.move_handles.len() != move_handles.len()
        {
            return Err(HorizonKernelResult::INVALID_COMBINATION);
        }
        header.handle_offset()
    };

    let mut allocated = Vec::with_capacity(copy_handles.len() + move_handles.len());
    let mut encoded = Vec::with_capacity(copy_handles.len() + move_handles.len());
    for object in copy_handles.into_iter().chain(move_handles) {
        let handle = match object {
            Some(object) => match handles.insert_object(object) {
                Ok(handle) => {
                    allocated.push(handle);
                    handle
                }
                Err(_) => {
                    for handle in allocated {
                        let _ = handles.close(handle);
                    }
                    return Err(HorizonKernelResult::OUT_OF_HANDLES);
                }
            },
            None => 0,
        };
        encoded.push(handle);
    }
    for (index, handle) in encoded.into_iter().enumerate() {
        let offset = handle_offset + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&handle.to_le_bytes());
    }
    Ok(Some(bytes))
}

fn decode_transport_header(bytes: &[u8]) -> Result<Option<HipcRequest<'_>>, HorizonKernelResult> {
    let Some(word1) = bytes
        .get(4..8)
        .and_then(|word| <[u8; 4]>::try_from(word).ok())
        .map(u32::from_le_bytes)
    else {
        return Ok(None);
    };
    if word1 >> 31 == 0 {
        return Ok(None);
    }
    let bounded = &bytes[..bytes.len().min(TLS_COMMAND_BUFFER_SIZE)];
    HipcRequest::decode(bounded)
        .map(Some)
        .map_err(|_| HorizonKernelResult::INVALID_COMBINATION)
}

fn close_encoded_handles(handles: &mut HandleTable, bytes: &[u8]) {
    let Ok(Some(message)) = decode_transport_header(bytes) else {
        return;
    };
    for handle in message
        .copy_handles
        .iter()
        .chain(&message.move_handles)
        .copied()
    {
        if handle != 0 {
            let _ = handles.close(handle);
        }
    }
}

fn session_error(
    context: &mut ExceptionDispatchContext<'_>,
    error: SessionError,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let code = match error {
        SessionError::PeerClosed => HorizonKernelResult::SESSION_CLOSED,
        SessionError::QueueFull => HorizonKernelResult::OUT_OF_RESOURCE,
        SessionError::WrongEndpoint
        | SessionError::NoRequest
        | SessionError::ReplyPending
        | SessionError::MessageKindMismatch => HorizonKernelResult::INVALID_STATE,
    };
    result(context, code);
    resume()
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
        IpcWireError::ResourceExhausted => {
            result(context, HorizonKernelResult::OUT_OF_RESOURCE);
            resume()
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

fn map_shared_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let raw_permissions = read_register(context.thread().state(), 3) as u32;
    let permissions = match raw_permissions {
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        _ => {
            return reject(
                context,
                HorizonSvcFault::InvalidMemoryPermission {
                    raw: raw_permissions,
                },
            );
        }
    };
    let Some(shared_memory) = context
        .process()
        .handles()
        .get_as::<SharedMemoryObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    // Public ABI validation and register order:
    // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#MapSharedMemory
    if size == 0
        || !size.is_multiple_of(USER_BUFFER_ALIGNMENT)
        || usize::try_from(size).ok() != Some(shared_memory.size())
    {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !start.is_aligned_to(USER_BUFFER_ALIGNMENT)
        || start
            .get()
            .checked_add(size)
            .is_none_or(|end| end > context.process().address_space_limit())
    {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    if !shared_memory.remote_permissions().contains(permissions) {
        result(context, HorizonKernelResult::INVALID_STATE);
        return resume();
    }
    match context.process().memory().resize_zeroed_mapping(
        context.process().cpu().address_space_id(),
        start,
        0,
        size,
        permissions,
        MemoryMappingPurpose::SharedMemory,
    ) {
        Ok(()) => {
            log::debug!(
                "mapped temporary shared memory handle {handle:#x} at {start} ({size:#x} bytes)"
            );
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryMapping { fault }),
    }
}

fn unmap_shared_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let Some(shared_memory) = context
        .process()
        .handles()
        .get_as::<SharedMemoryObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if size == 0
        || !size.is_multiple_of(USER_BUFFER_ALIGNMENT)
        || usize::try_from(size).ok() != Some(shared_memory.size())
    {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !start.is_aligned_to(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    let Some(query) = query.filter(|mapping| {
        mapping.base == start
            && mapping.size == size
            && mapping.purpose == MemoryMappingPurpose::SharedMemory
    }) else {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    };
    match context.process().memory().resize_zeroed_mapping(
        context.process().cpu().address_space_id(),
        start,
        size,
        0,
        query.permissions,
        MemoryMappingPurpose::SharedMemory,
    ) {
        Ok(()) => {
            log::debug!(
                "unmapped temporary shared memory handle {handle:#x} from {start} ({size:#x} bytes)"
            );
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryMapping { fault }),
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

fn thread_tls(state: &ThreadCpuState) -> GuestVirtualAddress {
    match state {
        ThreadCpuState::A64(state) => GuestVirtualAddress::new(state.tpidr_el0()),
        ThreadCpuState::A32(state) => GuestVirtualAddress::new(u64::from(state.tpidrurw())),
    }
}

fn session_request_owner(context: &ExceptionDispatchContext<'_>) -> SessionRequestOwner {
    SessionRequestOwner {
        process_id: context.process().process_id(),
        thread_id: context.thread().object().thread_id(),
    }
}

fn read_guest_message(
    context: &mut ExceptionDispatchContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    immediate: u32,
) -> Result<Vec<u8>, ExceptionDispatchOutcome<HorizonSvcFault>> {
    let mut message = Vec::new();
    if message.try_reserve_exact(size).is_err() {
        result(context, HorizonKernelResult::OUT_OF_RESOURCE);
        return Err(resume());
    }
    message.resize(size, 0);
    if let Err(error) = crate::ipc_wire::read_bytes(context.process(), address, &mut message) {
        return Err(reject_ipc(context, immediate, error));
    }
    Ok(message)
}

fn read_handle_array(
    context: &mut ExceptionDispatchContext<'_>,
    pointer: u64,
    count: u32,
    immediate: u32,
) -> Result<Vec<u32>, ExceptionDispatchOutcome<HorizonSvcFault>> {
    let mut handles = Vec::with_capacity(count as usize);
    for index in 0..count {
        let Some(address) = pointer.checked_add(u64::from(index) * 4) else {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return Err(resume());
        };
        let read = context.process().memory().read(
            context.process().cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            MemoryAccess::normal(MemoryAccessSize::Word),
        );
        let value = match read {
            Ok(read) => read.value,
            Err(fault) => {
                return Err(reject(
                    context,
                    HorizonSvcFault::GuestMemory { immediate, fault },
                ));
            }
        };
        let MemoryValue::U32(handle) = value else {
            unreachable!("word access returns a word value")
        };
        handles.push(handle);
    }
    Ok(handles)
}

fn read_c_name(
    context: &mut ExceptionDispatchContext<'_>,
    start: GuestVirtualAddress,
    capacity: usize,
    immediate: u32,
) -> Result<Option<Vec<u8>>, ExceptionDispatchOutcome<HorizonSvcFault>> {
    let mut name = Vec::with_capacity(capacity);
    for index in 0..capacity {
        let Some(address) = start.checked_add(index as u64) else {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return Err(resume());
        };
        let read = context.process().memory().read(
            context.process().cpu().address_space_id(),
            address,
            MemoryAccess::normal(MemoryAccessSize::Byte),
        );
        let byte = match read {
            Ok(read) => match read.value {
                MemoryValue::U8(byte) => byte,
                _ => unreachable!("byte access returns a byte value"),
            },
            Err(fault) => {
                return Err(reject(
                    context,
                    HorizonSvcFault::GuestMemory { immediate, fault },
                ));
            }
        };
        if byte == 0 {
            return Ok(Some(name));
        }
        name.push(byte);
    }
    Ok(None)
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

fn read_reply_timeout(state: &ThreadCpuState, user_buffer: bool) -> i64 {
    match state {
        ThreadCpuState::A64(_) => read_register(state, if user_buffer { 6 } else { 4 }) as i64,
        ThreadCpuState::A32(a32) => {
            let (low_index, high_index) = if user_buffer { (5, 6) } else { (0, 4) };
            let low = u64::from(
                a32.read_r(
                    A32GeneralRegister::new(low_index)
                        .expect("AArch32 reply timeout low register is valid"),
                ),
            );
            let high = u64::from(
                a32.read_r(
                    A32GeneralRegister::new(high_index)
                        .expect("AArch32 reply timeout high register is valid"),
                ),
            );
            (low | (high << 32)) as i64
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
    // Creation, paired endpoint insertion, and light-session selection follow:
    // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_session.cpp
    let is_light = read_register(context.thread().state(), 2) as u32;
    let (server, client) = if is_light == 0 {
        SessionObject::create_pair()
    } else {
        SessionObject::create_light_pair()
    };
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

fn create_port(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    // Port creation/connect/accept and named-port validation follow:
    // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_port.cpp
    let max_sessions = read_register(context.thread().state(), 2) as u32 as i32;
    let is_light = read_register(context.thread().state(), 3) as u32 != 0;
    if max_sessions <= 0 {
        result(context, HorizonKernelResult::OUT_OF_RANGE);
        return resume();
    }
    let (server, client) = PortObject::create_pair(max_sessions as usize, is_light);
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
        Err(()) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
    }
    resume()
}

fn connect_to_port(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 1) as u32;
    let Some(port) = context
        .process()
        .handles()
        .get_as::<PortObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if port.endpoint() != PortEndpoint::Client {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let session = match port.connect() {
        Ok(session) => session,
        Err(PortError::SessionLimit) => {
            result(context, HorizonKernelResult::OUT_OF_SESSIONS);
            return resume();
        }
        Err(PortError::PeerClosed) => {
            result(context, HorizonKernelResult::PORT_CLOSED);
            return resume();
        }
        Err(PortError::WrongEndpoint | PortError::NoPendingSession) => {
            result(context, HorizonKernelResult::INVALID_STATE);
            return resume();
        }
    };
    match context.process_mut().handles_mut().insert(session) {
        Ok(session_handle) => {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(
                context.thread_mut().state_mut(),
                1,
                u64::from(session_handle),
            );
        }
        Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
    }
    resume()
}

fn accept_session(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 1) as u32;
    let Some(port) = context
        .process()
        .handles()
        .get_as::<PortObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if port.endpoint() != PortEndpoint::Server {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    match port.accept() {
        Ok(session) => match context.process_mut().handles_mut().insert(session) {
            Ok(session_handle) => {
                result(context, HorizonKernelResult::SUCCESS);
                write_register(
                    context.thread_mut().state_mut(),
                    1,
                    u64::from(session_handle),
                );
            }
            Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
        },
        Err(PortError::NoPendingSession) => result(context, HorizonKernelResult::NOT_FOUND),
        Err(PortError::PeerClosed) => result(context, HorizonKernelResult::PORT_CLOSED),
        Err(PortError::WrongEndpoint | PortError::SessionLimit) => {
            result(context, HorizonKernelResult::INVALID_STATE);
        }
    }
    resume()
}

#[derive(Clone, Debug)]
enum ReplyWaitTarget {
    Port(PortObject),
    Session(SessionObject),
}

impl HorizonSvcDispatcher {
    fn reply_and_receive(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        user_buffer: bool,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // ABI and ordering reference:
        // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#ReplyAndReceive
        // Kernel control-flow reference:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_ipc.cpp
        let (immediate, address, size, handles_address, count, reply_target, timeout) =
            if user_buffer {
                let address = read_register(context.thread().state(), 1);
                let size = read_register(context.thread().state(), 2);
                let timeout = read_reply_timeout(context.thread().state(), true);
                (
                    0x44,
                    GuestVirtualAddress::new(address),
                    size,
                    read_register(context.thread().state(), 3),
                    read_register(context.thread().state(), 4) as u32,
                    read_register(context.thread().state(), 5) as u32,
                    timeout,
                )
            } else {
                let timeout = read_reply_timeout(context.thread().state(), false);
                (
                    0x43,
                    thread_tls(context.thread().state()),
                    TLS_COMMAND_BUFFER_SIZE as u64,
                    read_register(context.thread().state(), 1),
                    read_register(context.thread().state(), 2) as u32,
                    read_register(context.thread().state(), 3) as u32,
                    timeout,
                )
            };
        if user_buffer && !address.get().is_multiple_of(USER_BUFFER_ALIGNMENT) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        if user_buffer && !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        if user_buffer && size == 0 {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        if user_buffer
            && address
                .get()
                .checked_add(size)
                .is_none_or(|end| address.get() >= end)
        {
            result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
            return resume();
        }
        let Ok(size) = usize::try_from(size) else {
            result(context, HorizonKernelResult::OUT_OF_RESOURCE);
            return resume();
        };
        if count > MAX_WAIT_HANDLES {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        let handles = match read_handle_array(context, handles_address, count, immediate) {
            Ok(handles) => handles,
            Err(outcome) => return outcome,
        };
        let mut targets = Vec::with_capacity(handles.len());
        for handle in handles {
            if let Some(port) = context
                .process()
                .handles()
                .get_as::<PortObject>(handle)
                .cloned()
                && port.endpoint() == PortEndpoint::Server
            {
                targets.push(ReplyWaitTarget::Port(port));
                continue;
            }
            if let Some(session) = context
                .process()
                .handles()
                .get_as::<SessionObject>(handle)
                .cloned()
                && session.endpoint() == SessionEndpoint::Server
                && !session.is_light()
            {
                targets.push(ReplyWaitTarget::Session(session));
                continue;
            }
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }

        let thread_id = context.thread().object().thread_id();
        if reply_target != 0 && !self.reply_sent.contains(&thread_id) {
            let Some(session) = context
                .process()
                .handles()
                .get_as::<SessionObject>(reply_target)
                .cloned()
            else {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            };
            if session.endpoint() != SessionEndpoint::Server || session.is_light() {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            }
            let reply = match read_guest_message(context, address, size, immediate) {
                Ok(reply) => reply,
                Err(outcome) => {
                    write_register(context.thread_mut().state_mut(), 1, u64::from(u32::MAX));
                    return outcome;
                }
            };
            let reply = match capture_message_handles(context, reply, true) {
                Ok(reply) => reply,
                Err(code) => {
                    result(context, code);
                    write_register(context.thread_mut().state_mut(), 1, u64::from(u32::MAX));
                    return resume();
                }
            };
            if let Err(error) = session.reply(reply) {
                write_register(context.thread_mut().state_mut(), 1, u64::from(u32::MAX));
                return session_error(context, error);
            }
            self.reply_sent.insert(thread_id);
        } else if reply_target == 0 {
            self.reply_sent.remove(&thread_id);
        }

        for (index, target) in targets.into_iter().enumerate() {
            match target {
                ReplyWaitTarget::Port(port) if port.is_signalled() => {
                    self.finish_reply_wait(thread_id, immediate);
                    result(context, HorizonKernelResult::SUCCESS);
                    write_register(context.thread_mut().state_mut(), 1, index as u64);
                    return resume();
                }
                ReplyWaitTarget::Session(session) if session.is_signalled() => {
                    match session.receive() {
                        Ok(
                            message @ (SessionMessage::Buffer(_)
                            | SessionMessage::TransportedBuffer { .. }),
                        ) => {
                            let request = match materialize_message_handles(context, message) {
                                Ok(Some(request)) => request,
                                Ok(None) => unreachable!("buffer message materializes as bytes"),
                                Err(code) => {
                                    self.finish_reply_wait(thread_id, immediate);
                                    result(context, code);
                                    return resume();
                                }
                            };
                            if request.len() > size {
                                close_encoded_handles(
                                    context.process_mut().handles_mut(),
                                    &request,
                                );
                                self.finish_reply_wait(thread_id, immediate);
                                result(context, HorizonKernelResult::INVALID_SIZE);
                                return resume();
                            }
                            if let Err(error) =
                                crate::ipc_wire::write_bytes(context.process(), address, &request)
                            {
                                close_encoded_handles(
                                    context.process_mut().handles_mut(),
                                    &request,
                                );
                                self.finish_reply_wait(thread_id, immediate);
                                return reject_ipc(context, immediate, error);
                            }
                            self.finish_reply_wait(thread_id, immediate);
                            result(context, HorizonKernelResult::SUCCESS);
                            write_register(context.thread_mut().state_mut(), 1, index as u64);
                            return resume();
                        }
                        Ok(SessionMessage::Light(_)) | Err(SessionError::MessageKindMismatch) => {
                            self.finish_reply_wait(thread_id, immediate);
                            result(context, HorizonKernelResult::INVALID_STATE);
                            return resume();
                        }
                        Err(SessionError::PeerClosed) => {
                            self.finish_reply_wait(thread_id, immediate);
                            result(context, HorizonKernelResult::SESSION_CLOSED);
                            write_register(context.thread_mut().state_mut(), 1, index as u64);
                            return resume();
                        }
                        Err(SessionError::NoRequest) => {}
                        Err(error) => {
                            self.finish_reply_wait(thread_id, immediate);
                            return session_error(context, error);
                        }
                    }
                }
                ReplyWaitTarget::Port(_) | ReplyWaitTarget::Session(_) => {}
            }
        }

        if self.wait_expired(thread_id, immediate, timeout) {
            self.finish_reply_wait(thread_id, immediate);
            result(context, HorizonKernelResult::TIMED_OUT);
            resume()
        } else {
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
        }
    }

    fn wait_expired(&mut self, thread_id: u64, immediate: u32, timeout: i64) -> bool {
        if timeout == 0 {
            return true;
        }
        if timeout < 0 {
            return false;
        }
        let now = Instant::now();
        let deadline = self
            .wait_deadlines
            .entry((thread_id, immediate))
            .or_insert_with(|| {
                now.checked_add(Duration::from_nanos(timeout as u64))
                    .unwrap_or(now)
            });
        now >= *deadline
    }

    fn finish_reply_wait(&mut self, thread_id: u64, immediate: u32) {
        self.reply_sent.remove(&thread_id);
        self.wait_deadlines.remove(&(thread_id, immediate));
    }
}

impl HorizonSvcDispatcher {
    fn reply_and_receive_light(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Light IPC carries seven u32 words in registers and uses bit 31 of the
        // first word as the reply flag:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_light_server_session.cpp
        let handle = read_register(context.thread().state(), 0) as u32;
        let Some(session) = context
            .process()
            .handles()
            .get_as::<SessionObject>(handle)
            .cloned()
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Server || !session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let mut words = [0_u32; 7];
        for (index, word) in words.iter_mut().enumerate() {
            *word = read_register(context.thread().state(), index as u8 + 1) as u32;
        }
        let thread_id = context.thread().object().thread_id();
        if words[0] & (1 << 31) != 0
            && !self.reply_sent.contains(&thread_id)
            && let Err(error) = session.reply(SessionMessage::Light(words))
        {
            return session_error(context, error);
        }
        if words[0] & (1 << 31) != 0 {
            self.reply_sent.insert(thread_id);
        } else {
            self.reply_sent.remove(&thread_id);
        }
        match session.receive() {
            Ok(SessionMessage::Light(request)) => {
                self.reply_sent.remove(&thread_id);
                for (index, word) in request.into_iter().enumerate() {
                    write_register(
                        context.thread_mut().state_mut(),
                        index as u8 + 1,
                        u64::from(word),
                    );
                }
                result(context, HorizonKernelResult::SUCCESS);
                resume()
            }
            Ok(SessionMessage::Buffer(_) | SessionMessage::TransportedBuffer { .. })
            | Err(SessionError::MessageKindMismatch) => {
                self.reply_sent.remove(&thread_id);
                result(context, HorizonKernelResult::INVALID_STATE);
                resume()
            }
            Err(SessionError::NoRequest) => {
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
            }
            Err(error) => {
                self.reply_sent.remove(&thread_id);
                session_error(context, error)
            }
        }
    }
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
    let process_id = if handle == CURRENT_PROCESS_HANDLE {
        Some(context.process().process_id())
    } else {
        context
            .process()
            .handles()
            .get_as::<ProcessObject>(handle)
            .map(|process| process.process_id())
    };
    if let Some(process_id) = process_id {
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
            MemoryMappingPurpose::SharedMemory => 6,
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

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_runtime::EventObject;

    #[test]
    fn failed_handle_materialization_rolls_back_partial_allocations() {
        let retained = HandleObject::new(EventObject::new());
        let mut handles = HandleTable::with_capacity_limit(2);
        let existing = handles.insert(ThreadObject::new(1)).unwrap();
        let mut bytes = vec![0_u8; 20];
        bytes[0..4].copy_from_slice(&4_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(1_u32 << 31).to_le_bytes());
        bytes[8..12].copy_from_slice(&(2_u32 << 1).to_le_bytes());

        assert_eq!(
            materialize_message_handles_in_table(
                &mut handles,
                SessionMessage::TransportedBuffer {
                    bytes,
                    copy_handles: vec![Some(retained.clone()), Some(retained)],
                    move_handles: Vec::new(),
                },
            ),
            Err(HorizonKernelResult::OUT_OF_HANDLES)
        );
        assert_eq!(handles.len(), 1);
        assert!(handles.get(existing).unwrap().is::<ThreadObject>());
    }
}
