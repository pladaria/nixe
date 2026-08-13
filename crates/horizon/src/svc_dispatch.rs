//! Minimal verified Horizon SVC semantics for interpreter bring-up.
//!
//! ABI layouts and result values follow the public Switchbrew SVC revision
//! linked from [`crate::svc`]. Operations needing a scheduler or HIPC wire
//! transport remain explicit unsupported semantics rather than approximations.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::time::Duration;

use nixe_cpu::address::GuestVirtualAddress;
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::memory::{
    DataAccessFault, DataAccessFaultReason, MemoryAccess, MemoryAccessSize, MemoryAttributes,
    MemoryMappingError, MemoryMappingErrorReason, MemoryMappingPurpose, MemoryPermissions,
    MemoryProtectionError, MemoryProtectionErrorReason, MemoryRegionKind, MemoryValue,
};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::state::a32::A32GeneralRegister;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_memory::{CanonicalRangeAccessError, CanonicalRangeTranslationError};
use nixe_runtime::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatchRequest,
    ExceptionDispatcher, ExceptionResume, ExceptionTerminationReason, ExceptionTerminationScope,
    HandleObject, HandleTable, PortEndpoint, PortError, PortObject, ProcessObject,
    ReadableEventObject, SessionEndpoint, SessionError, SessionMessage, SessionObject,
    SessionRequestOwner, SessionRequestResult, SharedMemoryObject, ThreadCreateError,
    ThreadCreateRequest, ThreadObject, TransferMemoryObject, WritableEventObject,
};
use nixe_scheduler::{GuestThreadId, ProcessId, VirtualCpuId};

use crate::ipc_message::HipcRequest;
use crate::ipc_wire::{IpcWireError, NamedPortResult, SyncRequestResult};
use crate::{UnsupportedHorizonSvc, decode_horizon_svc};

mod ipc;
mod memory;
mod scheduled;
mod synchronization;
mod thread;
use ipc::*;
use memory::*;
pub use scheduled::HorizonScheduledDispatchError;
use synchronization::{get_process_id, get_thread_id, insert_pair};

pub const CURRENT_THREAD_HANDLE: u32 = 0xffff_8000;
pub const CURRENT_PROCESS_HANDLE: u32 = 0xffff_8001;
pub const MAX_WAIT_HANDLES: u32 = 0x40;
const INVALID_HANDLE: u32 = 0;
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
    match context.process().resize_memory_mapping(
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
    CanonicalMemory {
        immediate: u32,
        fault: CanonicalRangeTranslationError,
    },
    CanonicalBacking {
        immediate: u32,
        fault: CanonicalRangeAccessError,
    },
    MalformedIpc {
        immediate: u32,
        reason: &'static str,
    },
    InternalIpc {
        immediate: u32,
        reason: &'static str,
    },
    InternalRuntime {
        operation: &'static str,
    },
    UnsupportedNvDrv {
        immediate: u32,
        operation: crate::nvdrv::UnsupportedNvDrvOperation,
    },
    UnsupportedService {
        immediate: u32,
        operation: crate::ipc_wire::UnsupportedServiceOperation,
    },
}

impl HorizonSvcFault {
    /// Returns the stable Horizon result exposed for a recoverable runtime
    /// rejection, or `None` when dispatch must stop on a host-side fault.
    #[must_use]
    pub const fn guest_result(&self) -> Option<HorizonKernelResult> {
        match self {
            Self::Unknown(_) | Self::UnsupportedSemantics { .. } => None,
            Self::GuestMemory { fault, .. } => match fault.reason {
                DataAccessFaultReason::ContentGenerationExhausted
                | DataAccessFaultReason::HostBacking(_) => None,
                _ => Some(HorizonKernelResult::INVALID_POINTER),
            },
            Self::InvalidMemoryPermission { .. }
            | Self::InvalidMemoryAttribute { .. }
            | Self::InvalidMemoryState { .. } => Some(HorizonKernelResult::INVALID_STATE),
            Self::MemoryProtection { fault } => match fault.reason {
                MemoryProtectionErrorReason::InvalidRange
                | MemoryProtectionErrorReason::Unmapped => {
                    Some(HorizonKernelResult::INVALID_ADDRESS)
                }
                MemoryProtectionErrorReason::WritableExecutable
                | MemoryProtectionErrorReason::PermissionLocked => {
                    Some(HorizonKernelResult::INVALID_STATE)
                }
                MemoryProtectionErrorReason::GenerationExhausted => None,
            },
            Self::MemoryMapping { fault } => match fault.reason {
                MemoryMappingErrorReason::InvalidRange
                | MemoryMappingErrorReason::AlreadyMapped
                | MemoryMappingErrorReason::MappingStateMismatch => {
                    Some(HorizonKernelResult::INVALID_ADDRESS)
                }
                MemoryMappingErrorReason::WritableExecutable => {
                    Some(HorizonKernelResult::INVALID_STATE)
                }
                MemoryMappingErrorReason::ResourceExhausted => {
                    Some(HorizonKernelResult::RESOURCE_LIMIT)
                }
                MemoryMappingErrorReason::GenerationExhausted => None,
            },
            Self::CanonicalMemory { .. }
            | Self::CanonicalBacking { .. }
            | Self::InternalIpc { .. }
            | Self::InternalRuntime { .. } => None,
            Self::MalformedIpc { .. } => Some(HorizonKernelResult::INVALID_STATE),
            Self::UnsupportedNvDrv { .. } | Self::UnsupportedService { .. } => None,
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
            Self::CanonicalMemory { immediate, fault } => write!(
                formatter,
                "Horizon SVC {immediate:#x} cannot retain canonical memory: {fault}"
            ),
            Self::CanonicalBacking { immediate, fault } => write!(
                formatter,
                "Horizon SVC {immediate:#x} cannot access retained canonical backing: {fault}"
            ),
            Self::MalformedIpc { immediate, reason } => {
                write!(
                    formatter,
                    "Horizon SVC {immediate:#x} rejected malformed IPC: {reason}"
                )
            }
            Self::InternalIpc { immediate, reason } => write!(
                formatter,
                "Horizon SVC {immediate:#x} reached invalid emulator IPC state: {reason}"
            ),
            Self::InternalRuntime { operation } => {
                write!(
                    formatter,
                    "Horizon runtime operation {operation} violated an invariant"
                )
            }
            Self::UnsupportedNvDrv {
                immediate,
                operation,
            } => write!(
                formatter,
                "Horizon SVC {immediate:#x} reached unsupported emulator semantics: {operation}"
            ),
            Self::UnsupportedService {
                immediate,
                operation,
            } => write!(
                formatter,
                "Horizon SVC {immediate:#x} reached unsupported emulator semantics: {operation}"
            ),
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
    time_environment: crate::TimeEnvironment,
    video_system: crate::VideoSystem,
    hid_system: crate::HidSystem,
    named_ports: BTreeMap<Vec<u8>, PortObject>,
    reply_sent: BTreeSet<u64>,
    wait_deadlines: BTreeMap<(u64, u32), u64>,
    virtual_clock: nixe_runtime::VirtualClock,
    pending_wakes: BTreeMap<u64, PendingThreadWake>,
    pending_runtime_requests: BTreeMap<GuestThreadId, PendingRuntimeRequest>,
}

#[derive(Clone, Debug)]
enum PendingRuntimeRequest {
    CreateThread {
        entry: GuestVirtualAddress,
        argument: u64,
        stack_top: GuestVirtualAddress,
        priority: i32,
        core_id: i32,
    },
    StartThread {
        object_id: u64,
    },
    GetThreadPriority {
        object_id: u64,
    },
    SetThreadPriority {
        object_id: u64,
        priority: i32,
    },
    GetThreadCoreMask {
        object_id: u64,
    },
    SetThreadCoreMask {
        object_id: u64,
        ideal_core: i32,
        affinity_mask: u64,
    },
    SleepThread {
        nanoseconds: i64,
    },
    SetThreadActivity {
        object_id: u64,
        paused: bool,
    },
    GetThreadContext {
        object_id: u64,
        address: GuestVirtualAddress,
    },
    InheritPriority {
        owner_object_id: u64,
        waiter_object_id: u64,
        donation_key: u64,
    },
    RestorePriority {
        object_id: u64,
        donation_key: u64,
    },
    ReapThread {
        object_id: u64,
    },
}

#[derive(Clone, Debug)]
struct PendingThreadWake {
    events: Vec<ReadableEventObject>,
    deadline: Option<u64>,
}

/// Read-only diagnostic view of one Horizon wait staged for coordinator registration.
#[derive(Clone, Debug)]
pub struct PendingThreadWait {
    event: ReadableEventObject,
    timeout: Option<Duration>,
}

impl PendingThreadWait {
    #[must_use]
    pub fn event(&self) -> ReadableEventObject {
        self.event.clone()
    }

    #[must_use]
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
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
    pub fn new(
        initial_operation_mode: crate::OperationMode,
        time_environment: crate::TimeEnvironment,
    ) -> Self {
        Self::new_with_video(
            initial_operation_mode,
            time_environment,
            crate::VideoSystem::default(),
        )
    }

    /// Creates a dispatcher publishing display output to the supplied video system.
    #[must_use]
    pub fn new_with_video(
        initial_operation_mode: crate::OperationMode,
        time_environment: crate::TimeEnvironment,
        video_system: crate::VideoSystem,
    ) -> Self {
        let virtual_clock = time_environment.clock();
        Self {
            observed: BTreeMap::new(),
            unknown_calls: 0,
            initial_operation_mode,
            time_environment,
            video_system,
            hid_system: crate::HidSystem::new(),
            named_ports: BTreeMap::new(),
            reply_sent: BTreeSet::new(),
            wait_deadlines: BTreeMap::new(),
            virtual_clock,
            pending_wakes: BTreeMap::new(),
            pending_runtime_requests: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn video_system(&self) -> crate::VideoSystem {
        self.video_system.clone()
    }

    /// Advances the guest display clock and signals VI VSync when due.
    pub fn advance_video(&self, elapsed: Duration) -> Result<u64, crate::FramebufferError> {
        self.video_system.advance(elapsed)
    }

    /// Publishes the latest player-one controller state to Horizon HID.
    pub fn advance_input(
        &mut self,
        process: &nixe_runtime::RunnableProcess,
        state: Option<&nixe_input::EmulatedControllerState>,
        delta: Duration,
    ) -> Result<(), nixe_runtime::HandleError> {
        self.hid_system.publish(state, delta)?;
        self.hid_system.synchronize(process.memory())
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

    fn take_thread_wait(
        &mut self,
        thread_id: u64,
    ) -> Option<(Vec<ReadableEventObject>, Option<Duration>)> {
        let wait = self.pending_wakes.remove(&thread_id)?;
        let timeout = wait
            .deadline
            .map(|deadline| Duration::from_nanos(deadline.saturating_sub(self.virtual_time_ns())));
        Some((wait.events, timeout))
    }

    /// Returns the staged wait that the coordinator will consume for a thread.
    #[must_use]
    pub fn pending_thread_wait(&self, thread_id: u64) -> Option<PendingThreadWait> {
        self.pending_wakes.get(&thread_id).and_then(|wait| {
            wait.events.first().cloned().map(|event| PendingThreadWait {
                event,
                timeout: wait.deadline.map(|deadline| {
                    Duration::from_nanos(deadline.saturating_sub(self.virtual_time_ns()))
                }),
            })
        })
    }

    pub fn synchronize_virtual_time(&self, nanoseconds: u64) {
        self.virtual_clock.advance_scheduler_to(nanoseconds);
    }

    fn virtual_time_ns(&self) -> u64 {
        self.virtual_clock.scheduler_time_ns()
    }

    /// Applies one Horizon-to-runtime scheduler operation staged during SVC
    /// dispatch. Returns `Ok(false)` when the calling thread staged none.
    pub fn apply_pending_runtime_request(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        process_id: ProcessId,
        thread_id: GuestThreadId,
    ) -> Result<bool, HorizonSvcFault> {
        let Some(request) = self.pending_runtime_requests.remove(&thread_id) else {
            return Ok(false);
        };
        let mut fully_handled = true;
        match request {
            PendingRuntimeRequest::CreateThread {
                entry,
                argument,
                stack_top,
                priority,
                core_id,
            } => {
                let default_vcpu = coordinator
                    .process(process_id)
                    .ok_or(HorizonSvcFault::InternalRuntime {
                        operation: "CreateThread",
                    })?
                    .initial_ideal_vcpu();
                let ideal = if core_id == -2 {
                    default_vcpu
                } else if let Ok(core) = u32::try_from(core_id) {
                    VirtualCpuId::new(core)
                } else {
                    self.finish_create_thread(
                        coordinator,
                        process_id,
                        thread_id,
                        Err(ThreadCreateError::InvalidVirtualCpu(VirtualCpuId::new(
                            u32::MAX,
                        ))),
                    )?;
                    return Ok(true);
                };
                let affinity = match coordinator.scheduler().profile().core_set([ideal]) {
                    Ok(affinity) => affinity,
                    Err(_) => {
                        self.finish_create_thread(
                            coordinator,
                            process_id,
                            thread_id,
                            Err(ThreadCreateError::InvalidVirtualCpu(ideal)),
                        )?;
                        return Ok(true);
                    }
                };
                let result = coordinator.create_thread(
                    process_id,
                    ThreadCreateRequest {
                        entry,
                        argument,
                        stack_top,
                        priority,
                        ideal_vcpu: Some(ideal),
                        affinity,
                    },
                );
                self.finish_create_thread(coordinator, process_id, thread_id, result)?;
            }
            PendingRuntimeRequest::StartThread { object_id } => {
                let result = coordinator.start_thread(process_id, object_id);
                let code = match result {
                    Ok(_) => HorizonKernelResult::SUCCESS,
                    Err(nixe_runtime::ThreadOperationError::InvalidHandle) => {
                        HorizonKernelResult::INVALID_HANDLE
                    }
                    Err(nixe_runtime::ThreadOperationError::InvalidState) => {
                        HorizonKernelResult::INVALID_STATE
                    }
                    Err(nixe_runtime::ThreadOperationError::Internal) => {
                        return Err(HorizonSvcFault::InternalRuntime {
                            operation: "StartThread",
                        });
                    }
                };
                let caller = coordinator
                    .process_mut(process_id)
                    .and_then(|process| process.thread_mut(thread_id))
                    .ok_or(HorizonSvcFault::InternalRuntime {
                        operation: "StartThread",
                    })?;
                write_register(caller.state_mut(), 0, u64::from(code.raw()));
                coordinator.make_thread_ready(thread_id).map_err(|_| {
                    HorizonSvcFault::InternalRuntime {
                        operation: "StartThread",
                    }
                })?;
            }
            PendingRuntimeRequest::GetThreadPriority { object_id } => {
                let info = coordinator.thread_scheduling_info(process_id, object_id);
                let (code, priority) = match info {
                    Ok(info) => (HorizonKernelResult::SUCCESS, Some(info.priority)),
                    Err(nixe_runtime::ThreadOperationError::InvalidHandle) => {
                        (HorizonKernelResult::INVALID_HANDLE, None)
                    }
                    Err(_) => return Err(runtime_fault("GetThreadPriority")),
                };
                let state =
                    pending_caller_state(coordinator, process_id, thread_id, "GetThreadPriority")?;
                write_register(state, 0, u64::from(code.raw()));
                if let Some(priority) = priority {
                    write_register(state, 1, priority as u32 as u64);
                }
                resume_pending_caller(coordinator, thread_id, "GetThreadPriority")?;
            }
            PendingRuntimeRequest::SetThreadPriority {
                object_id,
                priority,
            } => {
                let code = map_thread_operation(
                    coordinator.set_thread_priority(process_id, object_id, priority),
                    "SetThreadPriority",
                )?;
                write_register(
                    pending_caller_state(coordinator, process_id, thread_id, "SetThreadPriority")?,
                    0,
                    u64::from(code.raw()),
                );
                resume_pending_caller(coordinator, thread_id, "SetThreadPriority")?;
            }
            PendingRuntimeRequest::GetThreadCoreMask { object_id } => {
                let info = coordinator.thread_scheduling_info(process_id, object_id);
                let (code, values) = match info {
                    Ok(info) => {
                        let ideal = info.ideal_vcpu.map_or(-1, |vcpu| vcpu.get() as i32);
                        let Some(mask) = horizon_affinity_mask(&info.affinity) else {
                            return Err(runtime_fault(
                                "GetThreadCoreMask unrepresentable topology",
                            ));
                        };
                        (HorizonKernelResult::SUCCESS, Some((ideal, mask)))
                    }
                    Err(nixe_runtime::ThreadOperationError::InvalidHandle) => {
                        (HorizonKernelResult::INVALID_HANDLE, None)
                    }
                    Err(_) => return Err(runtime_fault("GetThreadCoreMask")),
                };
                let state =
                    pending_caller_state(coordinator, process_id, thread_id, "GetThreadCoreMask")?;
                write_register(state, 0, u64::from(code.raw()));
                if let Some((ideal, mask)) = values {
                    write_register(state, 1, ideal as u32 as u64);
                    write_u64(state, 2, mask);
                }
                resume_pending_caller(coordinator, thread_id, "GetThreadCoreMask")?;
            }
            PendingRuntimeRequest::SetThreadCoreMask {
                object_id,
                ideal_core,
                affinity_mask,
            } => {
                let mut cores = Vec::new();
                for descriptor in coordinator.scheduler().profile().vcpus() {
                    if 1_u64
                        .checked_shl(descriptor.id().get())
                        .is_some_and(|bit| affinity_mask & bit != 0)
                    {
                        cores.push(descriptor.id());
                    }
                }
                let represented = cores.iter().fold(0_u64, |mask, vcpu| {
                    mask | 1_u64.checked_shl(vcpu.get()).unwrap_or(0)
                });
                let code = if represented != affinity_mask || cores.is_empty() {
                    HorizonKernelResult::OUT_OF_RANGE
                } else {
                    let affinity = coordinator
                        .scheduler()
                        .profile()
                        .core_set(cores)
                        .map_err(|_| runtime_fault("SetThreadCoreMask"))?;
                    let ideal = if ideal_core == -1 {
                        None
                    } else if let Ok(core) = u32::try_from(ideal_core) {
                        Some(VirtualCpuId::new(core))
                    } else {
                        write_register(
                            pending_caller_state(
                                coordinator,
                                process_id,
                                thread_id,
                                "SetThreadCoreMask",
                            )?,
                            0,
                            u64::from(HorizonKernelResult::OUT_OF_RANGE.raw()),
                        );
                        resume_pending_caller(coordinator, thread_id, "SetThreadCoreMask")?;
                        return Ok(true);
                    };
                    map_thread_operation(
                        coordinator.set_thread_affinity(process_id, object_id, ideal, affinity),
                        "SetThreadCoreMask",
                    )?
                };
                write_register(
                    pending_caller_state(coordinator, process_id, thread_id, "SetThreadCoreMask")?,
                    0,
                    u64::from(code.raw()),
                );
                resume_pending_caller(coordinator, thread_id, "SetThreadCoreMask")?;
            }
            PendingRuntimeRequest::SleepThread { nanoseconds } => {
                coordinator
                    .sleep_thread(thread_id, nanoseconds)
                    .map_err(|_| runtime_fault("SleepThread"))?;
            }
            PendingRuntimeRequest::SetThreadActivity { object_id, paused } => {
                let info = coordinator
                    .thread_scheduling_info(process_id, object_id)
                    .map_err(|error| match error {
                        nixe_runtime::ThreadOperationError::InvalidHandle => {
                            runtime_fault("SetThreadActivity")
                        }
                        _ => runtime_fault("SetThreadActivity"),
                    })?;
                let code = if info.id == thread_id {
                    HorizonKernelResult::INVALID_STATE
                } else {
                    map_thread_operation(
                        coordinator.set_thread_activity(process_id, object_id, paused),
                        "SetThreadActivity",
                    )?
                };
                write_register(
                    pending_caller_state(coordinator, process_id, thread_id, "SetThreadActivity")?,
                    0,
                    u64::from(code.raw()),
                );
                resume_pending_caller(coordinator, thread_id, "SetThreadActivity")?;
            }
            PendingRuntimeRequest::GetThreadContext { object_id, address } => {
                let info = coordinator
                    .thread_scheduling_info(process_id, object_id)
                    .map_err(|_| runtime_fault("GetThreadContext3"))?;
                let code = if info.id == thread_id || !info.paused {
                    HorizonKernelResult::INVALID_STATE
                } else {
                    let state = coordinator
                        .thread_cpu_state(process_id, object_id)
                        .map_err(|_| runtime_fault("GetThreadContext3"))?;
                    let bytes = encode_thread_context(&state);
                    let process = coordinator
                        .process(process_id)
                        .ok_or_else(|| runtime_fault("GetThreadContext3"))?;
                    if !process.memory().write_mapped_ram(
                        process.cpu_context().address_space_id(),
                        address,
                        &bytes,
                    ) {
                        return Err(runtime_fault("GetThreadContext3 atomic context write"));
                    }
                    HorizonKernelResult::SUCCESS
                };
                write_register(
                    pending_caller_state(coordinator, process_id, thread_id, "GetThreadContext3")?,
                    0,
                    u64::from(code.raw()),
                );
                resume_pending_caller(coordinator, thread_id, "GetThreadContext3")?;
            }
            PendingRuntimeRequest::InheritPriority {
                owner_object_id,
                waiter_object_id,
                donation_key,
            } => {
                coordinator
                    .inherit_thread_priority(
                        process_id,
                        owner_object_id,
                        waiter_object_id,
                        donation_key,
                    )
                    .map_err(|_| runtime_fault("ArbitrateLock priority inheritance"))?;
                fully_handled = false;
            }
            PendingRuntimeRequest::RestorePriority {
                object_id,
                donation_key,
            } => {
                coordinator
                    .restore_thread_priority(process_id, object_id, donation_key)
                    .map_err(|_| runtime_fault("ArbitrateUnlock priority restoration"))?;
                fully_handled = false;
            }
            PendingRuntimeRequest::ReapThread { object_id } => {
                coordinator
                    .reap_thread(object_id)
                    .map_err(|_| runtime_fault("CloseHandle thread reaping"))?;
                resume_pending_caller(coordinator, thread_id, "CloseHandle thread reaping")?;
            }
        }
        Ok(fully_handled)
    }

    fn finish_create_thread(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        process_id: ProcessId,
        caller: GuestThreadId,
        creation: Result<nixe_runtime::ThreadCreation, ThreadCreateError>,
    ) -> Result<(), HorizonSvcFault> {
        let (code, handle) = match creation {
            Ok(creation) => (HorizonKernelResult::SUCCESS, Some(creation.handle)),
            Err(ThreadCreateError::InvalidEntry | ThreadCreateError::InvalidStack) => {
                (HorizonKernelResult::INVALID_ADDRESS, None)
            }
            Err(
                ThreadCreateError::InvalidPriority(_)
                | ThreadCreateError::InvalidVirtualCpu(_)
                | ThreadCreateError::PolicyDenied,
            ) => (HorizonKernelResult::OUT_OF_RANGE, None),
            Err(ThreadCreateError::ResourceLimit) => (HorizonKernelResult::RESOURCE_LIMIT, None),
            Err(ThreadCreateError::IdentityExhausted | ThreadCreateError::Internal) => {
                return Err(HorizonSvcFault::InternalRuntime {
                    operation: "CreateThread",
                });
            }
        };
        let state = coordinator
            .process_mut(process_id)
            .ok_or(HorizonSvcFault::InternalRuntime {
                operation: "CreateThread",
            })?
            .thread_mut(caller)
            .ok_or(HorizonSvcFault::InternalRuntime {
                operation: "CreateThread",
            })?
            .state_mut();
        write_register(state, 0, u64::from(code.raw()));
        if let Some(handle) = handle {
            write_register(state, 1, u64::from(handle));
        }
        coordinator
            .make_thread_ready(caller)
            .map_err(|_| HorizonSvcFault::InternalRuntime {
                operation: "CreateThread",
            })?;
        Ok(())
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
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Unknown(error));
            }
        };

        let outcome = match immediate {
            0x01 => set_heap_size(context),
            0x02 => set_memory_permission(context),
            0x03 => set_memory_attribute(context),
            0x06 => query_memory(context, immediate),
            0x07 => terminate(ExceptionTerminationScope::Process),
            0x08 => self.create_thread(context),
            0x09 => self.start_thread(context),
            0x0a => terminate(ExceptionTerminationScope::CurrentThread),
            0x0b => self.sleep_thread(context),
            0x0c => self.get_thread_priority(context),
            0x0d => self.set_thread_priority(context),
            0x0e => self.get_thread_core_mask(context),
            0x0f => self.set_thread_core_mask(context),
            0x10 => {
                let vcpu = context.thread().vcpu().get();
                write_register(context.thread_mut().state_mut(), 0, u64::from(vcpu));
                resume()
            }
            0x11 => event_signal(context),
            0x12 => event_clear(context),
            0x13 => map_shared_memory(context, &mut self.hid_system),
            0x14 => unmap_shared_memory(context, &mut self.hid_system),
            0x15 => create_transfer_memory(context),
            0x16 => close_handle(self, context),
            0x17 => reset_signal(context),
            0x18 => self.wait_synchronization(context),
            0x1a => self.arbitrate_lock(context),
            0x1b => self.arbitrate_unlock(context),
            0x1c => self.wait_process_wide_key_atomic(context),
            0x1d => self.signal_process_wide_key(context),
            0x1f => self.connect_to_named_port(context),
            0x20 => self.send_sync_request_light(context),
            0x21 => self.send_sync_request(context),
            0x22 => self.send_sync_request_with_user_buffer(context),
            0x24 => get_process_id(context),
            0x25 => get_thread_id(context),
            0x26 => break_process(context),
            0x29 => get_info(context),
            0x32 => self.set_thread_activity(context),
            0x33 => self.get_thread_context(context),
            0x40 => create_session(context),
            0x41 => accept_session(context),
            0x42 => self.reply_and_receive_light(context),
            0x43 => self.reply_and_receive(context, false),
            0x44 => self.reply_and_receive(context, true),
            0x45 => create_event(context),
            0x70 => create_port(context),
            0x71 => self.manage_named_port(context),
            0x72 => connect_to_port(context),
            _ => ExceptionDispatchOutcome::Fault(HorizonSvcFault::UnsupportedSemantics {
                immediate,
                documented_name: descriptor
                    .unambiguous_name()
                    .unwrap_or("version-dependent SVC"),
            }),
        };
        self.observe(immediate, &outcome);
        outcome
    }
}

impl Default for HorizonSvcDispatcher {
    fn default() -> Self {
        Self::new(
            crate::OperationMode::default(),
            crate::TimeEnvironment::default(),
        )
    }
}

const fn svc_support(immediate: u32) -> HorizonSvcSupport {
    match immediate {
        0x07 | 0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x0d | 0x0e | 0x0f | 0x10 | 0x13 | 0x14
        | 0x15 | 0x16 | 0x25 | 0x32 | 0x40 | 0x41 | 0x45 | 0x70 | 0x71 | 0x72 => {
            HorizonSvcSupport::Complete
        }
        0x01 | 0x02 | 0x03 | 0x06 | 0x11 | 0x12 | 0x17 | 0x18 | 0x1a | 0x1b | 0x1c | 0x1d
        | 0x20 | 0x21 | 0x22 | 0x24 | 0x26 | 0x29 | 0x33 | 0x42 | 0x43 | 0x44 => {
            HorizonSvcSupport::Partial
        }
        0x1f => HorizonSvcSupport::Complete,
        _ => HorizonSvcSupport::Unsupported,
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

const fn runtime_fault(operation: &'static str) -> HorizonSvcFault {
    HorizonSvcFault::InternalRuntime { operation }
}

fn pending_caller_state<'a>(
    coordinator: &'a mut nixe_runtime::RuntimeCoordinator,
    process_id: ProcessId,
    thread_id: GuestThreadId,
    operation: &'static str,
) -> Result<&'a mut ThreadCpuState, HorizonSvcFault> {
    coordinator
        .process_mut(process_id)
        .and_then(|process| process.thread_mut(thread_id))
        .map(nixe_runtime::GuestThread::state_mut)
        .ok_or_else(|| runtime_fault(operation))
}

fn resume_pending_caller(
    coordinator: &mut nixe_runtime::RuntimeCoordinator,
    thread_id: GuestThreadId,
    operation: &'static str,
) -> Result<(), HorizonSvcFault> {
    coordinator
        .make_thread_ready(thread_id)
        .map_err(|_| runtime_fault(operation))
}

fn map_thread_operation(
    result: Result<(), nixe_runtime::ThreadOperationError>,
    operation: &'static str,
) -> Result<HorizonKernelResult, HorizonSvcFault> {
    match result {
        Ok(()) => Ok(HorizonKernelResult::SUCCESS),
        Err(nixe_runtime::ThreadOperationError::InvalidHandle) => {
            Ok(HorizonKernelResult::INVALID_HANDLE)
        }
        Err(nixe_runtime::ThreadOperationError::InvalidState) => {
            Ok(HorizonKernelResult::OUT_OF_RANGE)
        }
        Err(nixe_runtime::ThreadOperationError::Internal) => Err(runtime_fault(operation)),
    }
}

fn horizon_affinity_mask(affinity: &nixe_scheduler::CoreSet) -> Option<u64> {
    affinity.iter().try_fold(0_u64, |mask, vcpu| {
        1_u64.checked_shl(vcpu.get()).map(|bit| mask | bit)
    })
}

fn encode_thread_context(state: &ThreadCpuState) -> Vec<u8> {
    match state {
        ThreadCpuState::A64(state) => {
            let mut bytes = vec![0_u8; 0x320];
            for index in 0..29_u8 {
                put_context_u64(
                    &mut bytes,
                    usize::from(index) * 8,
                    state.read_x(A64Register::General(
                        A64GeneralRegister::new(index).expect("context GPR index is valid"),
                    )),
                );
            }
            put_context_u64(
                &mut bytes,
                0xe8,
                state.read_x(A64Register::General(
                    A64GeneralRegister::new(29).expect("frame-pointer index is valid"),
                )),
            );
            put_context_u64(
                &mut bytes,
                0xf0,
                state.read_x(A64Register::General(
                    A64GeneralRegister::new(30).expect("link-register index is valid"),
                )),
            );
            put_context_u64(&mut bytes, 0xf8, state.read_x(A64Register::StackPointer));
            put_context_u64(&mut bytes, 0x100, state.pc());
            put_context_u32(&mut bytes, 0x108, state.nzcv().bits());
            for index in 0..32_u8 {
                let value = state.vector(index).expect("context vector index is valid");
                let offset = 0x110 + usize::from(index) * 16;
                bytes[offset..offset + 16].copy_from_slice(&value.to_le_bytes());
            }
            put_context_u32(&mut bytes, 0x310, state.fpcr());
            put_context_u32(&mut bytes, 0x314, state.fpsr());
            put_context_u64(&mut bytes, 0x318, state.tpidr_el0());
            bytes
        }
        ThreadCpuState::A32(state) => {
            let mut bytes = vec![0_u8; 0x158];
            for index in 0..13_u8 {
                put_context_u32(
                    &mut bytes,
                    usize::from(index) * 4,
                    state.read_r(
                        A32GeneralRegister::new(index).expect("context GPR index is valid"),
                    ),
                );
            }
            put_context_u32(
                &mut bytes,
                0x34,
                state.read_r(A32GeneralRegister::new(13).expect("SP index is valid")),
            );
            put_context_u32(
                &mut bytes,
                0x38,
                state.read_r(A32GeneralRegister::new(14).expect("LR index is valid")),
            );
            put_context_u32(&mut bytes, 0x3c, state.instruction_address());
            put_context_u32(&mut bytes, 0x40, state.cpsr().bits());
            for index in 0..32_u8 {
                put_context_u64(
                    &mut bytes,
                    0x48 + usize::from(index) * 8,
                    state
                        .read_d(index)
                        .expect("context D-register index is valid"),
                );
            }
            put_context_u32(&mut bytes, 0x148, state.fpscr());
            // FPEXC is privileged architectural state and is not part of the
            // user-mode CPU state. Horizon returns it cleared here.
            put_context_u32(&mut bytes, 0x14c, 0);
            put_context_u32(&mut bytes, 0x150, state.tpidrurw());
            bytes
        }
    }
}

fn put_context_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_context_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
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
    dispatcher: &mut HorizonSvcDispatcher,
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    if matches!(handle, CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE) {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let closed = context.process_mut().handles_mut().close(handle);
    let Ok(object) = closed else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    let reap = (object.strong_count() == 1)
        .then(|| object.downcast_ref::<ThreadObject>())
        .flatten()
        .filter(|thread| thread.is_signalled())
        .map(ThreadObject::thread_id);
    let caller = context.thread().id();
    let Some(object_id) = reap else {
        result(context, HorizonKernelResult::SUCCESS);
        return resume();
    };
    if dispatcher
        .pending_runtime_requests
        .insert(caller, PendingRuntimeRequest::ReapThread { object_id })
        .is_some()
    {
        return ExceptionDispatchOutcome::Fault(runtime_fault("CloseHandle thread reaping"));
    }
    result(context, HorizonKernelResult::SUCCESS);
    ExceptionDispatchOutcome::Suspend(ExceptionResume::Next)
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
    fn thread_context_encoding_uses_architecture_specific_horizon_layouts() {
        let mut a32 = nixe_cpu::state::a32::A32State::a32();
        a32.write_r(A32GeneralRegister::new(0).expect("R0 exists"), 0x1122_3344);
        a32.write_r(A32GeneralRegister::new(13).expect("SP exists"), 0x5566_7788);
        a32.set_instruction_address(0x1000).unwrap();
        a32.write_d(31, 0x0102_0304_0506_0708);
        a32.set_fpscr(0xaabb_ccdd);
        a32.set_tpidrurw(0xdead_beef);
        let encoded = encode_thread_context(&ThreadCpuState::A32(Box::new(a32)));
        assert_eq!(encoded.len(), 0x158);
        assert_eq!(&encoded[0..4], &0x1122_3344_u32.to_le_bytes());
        assert_eq!(&encoded[0x34..0x38], &0x5566_7788_u32.to_le_bytes());
        assert_eq!(&encoded[0x3c..0x40], &0x1000_u32.to_le_bytes());
        assert_eq!(
            &encoded[0x140..0x148],
            &0x0102_0304_0506_0708_u64.to_le_bytes()
        );
        assert_eq!(&encoded[0x148..0x14c], &0xaabb_ccdd_u32.to_le_bytes());
        assert_eq!(&encoded[0x14c..0x150], &[0; 4]);
        assert_eq!(&encoded[0x150..0x154], &0xdead_beef_u32.to_le_bytes());

        let encoded = encode_thread_context(&ThreadCpuState::A64(Box::default()));
        assert_eq!(encoded.len(), 0x320);
    }

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

    #[test]
    fn unsupported_horizon_semantics_have_no_guest_result() {
        let fault = HorizonSvcFault::UnsupportedSemantics {
            immediate: 0x23,
            documented_name: "WaitForAddress",
        };

        assert_eq!(fault.guest_result(), None);
    }

    #[test]
    fn invalid_emulator_ipc_state_has_no_guest_result() {
        let fault = HorizonSvcFault::InternalIpc {
            immediate: 0x21,
            reason: "synthetic internal invariant",
        };

        assert_eq!(fault.guest_result(), None);
    }

    #[test]
    fn unsupported_nvdrv_semantics_have_no_guest_result() {
        let fault = HorizonSvcFault::UnsupportedNvDrv {
            immediate: 0x21,
            operation: crate::nvdrv::UnsupportedNvDrvOperation::Ioctl {
                context: crate::nvdrv::NvDrvErrorContext::new(
                    crate::nvdrv::NvDrvDeviceKind::HostControlGpu,
                    0xc018_4706,
                    crate::nvdrv::NvDrvFileDescriptor::new(3),
                    None,
                    crate::nvdrv::NvDrvValidationReason::UnsupportedOperation,
                ),
            },
        };

        assert_eq!(fault.guest_result(), None);
        assert_eq!(
            fault.to_string(),
            "Horizon SVC 0x21 reached unsupported emulator semantics: graphics-gap=ioctl nvdrv \
             ioctl is not implemented: device=/dev/nvhost-ctrl-gpu request=0xc0184706 \
             fd=nvfd:0x00000003 reason=unsupported-operation"
        );
    }

    #[test]
    fn emulator_generation_exhaustion_is_never_fabricated_as_a_guest_result() {
        let address_space = nixe_cpu::address::AddressSpaceId::new(1);
        let address = GuestVirtualAddress::new(0x1000);
        let faults = [
            HorizonSvcFault::GuestMemory {
                immediate: 0x21,
                fault: DataAccessFault::new(
                    address_space,
                    address,
                    nixe_cpu::memory::DataAccessKind::Write,
                    DataAccessFaultReason::ContentGenerationExhausted,
                ),
            },
            HorizonSvcFault::MemoryProtection {
                fault: MemoryProtectionError {
                    address_space,
                    address,
                    reason: MemoryProtectionErrorReason::GenerationExhausted,
                },
            },
            HorizonSvcFault::MemoryMapping {
                fault: MemoryMappingError {
                    address_space,
                    address,
                    reason: MemoryMappingErrorReason::GenerationExhausted,
                },
            },
        ];

        for fault in faults {
            assert_eq!(fault.guest_result(), None);
        }
    }

    #[test]
    fn canonical_backing_failures_are_never_fabricated_as_guest_results() {
        let address_space = nixe_cpu::address::AddressSpaceId::new(1);
        let address = GuestVirtualAddress::new(0x1000);
        let faults = [
            HorizonSvcFault::GuestMemory {
                immediate: 0x21,
                fault: DataAccessFault::new(
                    address_space,
                    address,
                    nixe_cpu::memory::DataAccessKind::Write,
                    DataAccessFaultReason::HostBacking("allocation failed".into()),
                ),
            },
            HorizonSvcFault::CanonicalMemory {
                immediate: 0x15,
                fault: CanonicalRangeTranslationError {
                    address_space,
                    address,
                    reason: nixe_memory::CanonicalRangeTranslationErrorReason::ResourceExhausted,
                },
            },
        ];

        for fault in faults {
            assert_eq!(fault.guest_result(), None);
        }
    }
}
