//! Horizon SVC dispatch and owned kernel continuations.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::time::Duration;

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::memory::MemoryPermissions;
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_memory::GuestVirtualAddress;
use nixe_runtime::GuestCpuState;
use nixe_runtime::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatchRequest,
    ExceptionDispatcher, ExceptionResume, ExceptionTerminationReason, ExceptionTerminationScope,
    HandleObject, HandleTable, PortEndpoint, PortError, PortObject, ProcessObject,
    ReadableEventObject, SessionEndpoint, SessionError, SessionMessage, SessionObject,
    SessionRequestOwner, SessionRequestResult, SharedMemoryObject, ThreadCreateError,
    ThreadCreateRequest, ThreadObject, TransferMemoryObject, WritableEventObject,
};
use nixe_scheduler::{GuestThreadId, ProcessId, VirtualCpuId};

use crate::ipc_wire::message::HipcRequest;
use crate::ipc_wire::{HorizonIpcFault, IpcWireError, NamedPortResult, SyncRequestResult};
use crate::{
    HorizonSvcDescriptor, HorizonSvcReturnKind, UnsupportedHorizonSvc, decode_horizon_svc,
};

mod address_arbitration;
mod address_word;
mod debug_string;
mod ipc;
mod memory;
mod memory_wait;
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
    pub const INVALID_MEMORY_REGION: Self = Self(0xdc01);
    pub const OUT_OF_RESOURCE: Self = Self(0xce01);
    pub const TIMED_OUT: Self = Self(0xea01);
    pub const CANCELLED: Self = Self(0xec01);
    pub const OUT_OF_RANGE: Self = Self(0xee01);
    // https://switchbrew.org/wiki/Error_codes#Kernel
    pub const INVALID_ENUM_VALUE: Self = Self(0xf001);
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

    const fn name(self) -> &'static str {
        match self {
            Self::SUCCESS => "Success",
            Self::NOT_IMPLEMENTED => "NotImplemented",
            Self::OUT_OF_SESSIONS => "OutOfSessions",
            Self::THREAD_TERMINATING => "ThreadTerminating",
            Self::INVALID_HANDLE => "InvalidHandle",
            Self::INVALID_POINTER => "InvalidPointer",
            Self::INVALID_ADDRESS => "InvalidAddress",
            Self::INVALID_SIZE => "InvalidSize",
            Self::INVALID_CURRENT_MEMORY => "InvalidCurrentMemory",
            Self::INVALID_MEMORY_REGION => "InvalidMemoryRegion",
            Self::OUT_OF_RESOURCE => "OutOfResource",
            Self::TIMED_OUT => "TimedOut",
            Self::CANCELLED => "Cancelled",
            Self::OUT_OF_RANGE => "OutOfRange",
            Self::INVALID_ENUM_VALUE => "InvalidEnumValue",
            Self::INVALID_STATE => "InvalidState",
            Self::RESOURCE_LIMIT => "ResourceLimit",
            Self::NOT_SUPPORTED => "NotSupported",
            Self::NOT_FOUND => "NotFound",
            Self::SESSION_CLOSED => "SessionClosed",
            Self::PORT_CLOSED => "PortClosed",
            Self::OUT_OF_HANDLES => "OutOfHandles",
            Self::INVALID_COMBINATION => "InvalidCombination",
            _ => "UnknownResult",
        }
    }
}

/// Host-side reason an SVC could not be given faithful guest semantics.
#[derive(Clone, Debug)]
pub enum HorizonSvcFault {
    NotSupervisorCall,
    MissingImmediate,
    Unknown(UnsupportedHorizonSvc),
    UnsupportedSemantics {
        immediate: u32,
        documented_name: &'static str,
    },
    InvalidMemoryPermission {
        raw: u32,
    },
    InvalidMemoryAttribute {
        mask: u32,
        value: u32,
    },
    MemoryMapping {
        fault: std::sync::Arc<nixe_runtime::memory::MappingError>,
    },
    MemoryAccess {
        immediate: u32,
        fault: std::sync::Arc<nixe_memory::MemoryError>,
    },
    Ipc {
        immediate: u32,
        fault: Box<HorizonIpcFault>,
    },
    InternalRuntime {
        operation: &'static str,
    },
}

impl HorizonSvcFault {
    /// Returns the stable Horizon result exposed for a recoverable runtime
    /// rejection, or `None` when dispatch must stop on a host-side fault.
    #[must_use]
    pub fn guest_result(&self) -> Option<HorizonKernelResult> {
        match self {
            Self::Unknown(_) | Self::UnsupportedSemantics { .. } => None,
            Self::InvalidMemoryPermission { .. } | Self::InvalidMemoryAttribute { .. } => {
                Some(HorizonKernelResult::INVALID_STATE)
            }
            Self::MemoryMapping { fault } => match fault.as_ref() {
                nixe_runtime::memory::MappingError::InvalidRange
                | nixe_runtime::memory::MappingError::AlreadyMapped
                | nixe_runtime::memory::MappingError::Unmapped => {
                    Some(HorizonKernelResult::INVALID_ADDRESS)
                }
                nixe_runtime::memory::MappingError::StateMismatch => {
                    Some(HorizonKernelResult::INVALID_CURRENT_MEMORY)
                }
                nixe_runtime::memory::MappingError::PhysicalIdentityMismatch => {
                    Some(HorizonKernelResult::INVALID_MEMORY_REGION)
                }
                nixe_runtime::memory::MappingError::PermissionLocked
                | nixe_runtime::memory::MappingError::InvalidPermission => {
                    Some(HorizonKernelResult::INVALID_STATE)
                }
                nixe_runtime::memory::MappingError::Memory(error) => match error.as_ref() {
                    nixe_memory::MemoryError::ProcessLimit { .. }
                    | nixe_memory::MemoryError::PhysicalExhaustion { .. } => {
                        Some(HorizonKernelResult::RESOURCE_LIMIT)
                    }
                    _ => None,
                },
            },
            Self::MemoryAccess { fault, .. } => fault
                .is_access_fault()
                .then_some(HorizonKernelResult::INVALID_POINTER),
            Self::Ipc { fault, .. } => fault.guest_result(),
            Self::InternalRuntime { .. } => None,
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
            Self::InvalidMemoryPermission { raw } => {
                write!(formatter, "invalid Horizon memory permission {raw:#x}")
            }
            Self::InvalidMemoryAttribute { mask, value } => write!(
                formatter,
                "invalid Horizon memory attribute mask={mask:#x} value={value:#x}"
            ),
            Self::MemoryMapping { fault } => write!(formatter, "Horizon mapping failed: {fault}"),
            Self::MemoryAccess { immediate, fault } => write!(
                formatter,
                "Horizon SVC {immediate:#x} memory access failed: {fault}"
            ),
            Self::Ipc { immediate, fault } => {
                write!(formatter, "Horizon SVC {immediate:#x} {fault}")
            }
            Self::InternalRuntime { operation } => {
                write!(
                    formatter,
                    "Horizon runtime operation {operation} violated an invariant"
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
    time_environment: crate::TimeEnvironment,
    settings_environment: crate::SettingsEnvironment,
    application_language: Option<crate::SystemLanguage>,
    user_account_switch_locked: Option<bool>,
    save_data: Option<crate::SaveDataSystem>,
    shared_fonts: crate::SharedFontSession,
    system_archives: std::sync::Arc<crate::SystemArchives>,
    keyboard_host: Option<nixe_applets::keyboard::KeyboardHost>,
    diagnostics: crate::HorizonDiagnostics,
    video_system: crate::VideoSystem,
    hid_system: crate::HidSystem,
    audio_backend: Option<std::sync::Arc<dyn nixe_audio::AudioBackend>>,
    audio_renderers: crate::AudioRendererManagerSession,
    named_ports: BTreeMap<Vec<u8>, PortObject>,
    reply_sent: BTreeSet<u64>,
    wait_deadlines: BTreeMap<(u64, u32), u64>,
    virtual_clock: nixe_runtime::VirtualClock,
    pending_wakes: BTreeMap<u64, PendingThreadWake>,
    synchronization_waits: BTreeMap<u64, synchronization::SynchronizationWait>,
    session_requests: BTreeMap<u64, ipc::KernelSessionRequest>,
    reply_waits: BTreeMap<u64, synchronization::ReplyReceiveWait>,
    pending_runtime_requests: BTreeMap<GuestThreadId, PendingRuntimeRequest>,
    memory_waits: BTreeMap<GuestThreadId, memory_wait::PendingMemory>,
    service_scheduling: std::sync::Arc<crate::ipc_wire::operation::ServiceScheduling>,
    memory_completions: std::sync::Arc<memory_wait::MemoryCompletions>,
    memory_events: Option<nixe_runtime::ExternalEventSender>,
    next_memory_generation: u64,
    cancellable_waits: BTreeSet<u64>,
    cancelled_waits: BTreeSet<u64>,
    pending_wait_cancellations: BTreeSet<u64>,
}

#[derive(Debug)]
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
    CancelSynchronization {
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
        ranges: Vec<nixe_memory::AllocationRange>,
    },
    SynchronizeMutexWaits {
        addresses: Vec<u64>,
    },
    ReapThread {
        object_id: u64,
    },
    RegisterAddressWait {
        address: u64,
        deadline: Option<u64>,
        check: address_arbitration::AddressWaitCheck,
        source: nixe_cpu::location::LocationDescriptor,
        vcpu: VirtualCpuId,
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
        machine: nixe_memory::Machine,
        initial_operation_mode: crate::OperationMode,
        time_environment: crate::TimeEnvironment,
        hid_system: crate::HidSystem,
        shared_fonts: crate::SharedFontSession,
    ) -> Self {
        Self::new_with_video(
            initial_operation_mode,
            time_environment,
            hid_system,
            shared_fonts,
            crate::VideoSystem::new(machine),
        )
    }

    /// Creates a dispatcher publishing display output to the supplied video system.
    #[must_use]
    pub fn new_with_video(
        initial_operation_mode: crate::OperationMode,
        time_environment: crate::TimeEnvironment,
        hid_system: crate::HidSystem,
        shared_fonts: crate::SharedFontSession,
        video_system: crate::VideoSystem,
    ) -> Self {
        Self::new_with_video_and_settings(
            initial_operation_mode,
            time_environment,
            hid_system,
            shared_fonts,
            crate::SettingsEnvironment::default(),
            video_system,
        )
    }

    /// Creates a dispatcher with explicit guest locale and display systems.
    #[must_use]
    pub fn new_with_video_and_settings(
        initial_operation_mode: crate::OperationMode,
        time_environment: crate::TimeEnvironment,
        hid_system: crate::HidSystem,
        shared_fonts: crate::SharedFontSession,
        settings_environment: crate::SettingsEnvironment,
        video_system: crate::VideoSystem,
    ) -> Self {
        let virtual_clock = time_environment.clock().clone();
        Self {
            observed: BTreeMap::new(),
            unknown_calls: 0,
            initial_operation_mode,
            time_environment,
            settings_environment,
            application_language: None,
            user_account_switch_locked: None,
            save_data: None,
            shared_fonts,
            system_archives: std::sync::Arc::new(crate::SystemArchives::default()),
            keyboard_host: None,
            diagnostics: crate::HorizonDiagnostics::default(),
            video_system,
            hid_system,
            audio_backend: None,
            audio_renderers: crate::AudioRendererManagerSession::default(),
            named_ports: BTreeMap::new(),
            reply_sent: BTreeSet::new(),
            wait_deadlines: BTreeMap::new(),
            virtual_clock,
            pending_wakes: BTreeMap::new(),
            synchronization_waits: BTreeMap::new(),
            session_requests: BTreeMap::new(),
            reply_waits: BTreeMap::new(),
            pending_runtime_requests: BTreeMap::new(),
            memory_waits: BTreeMap::new(),
            service_scheduling: std::sync::Arc::default(),
            memory_completions: std::sync::Arc::new(memory_wait::MemoryCompletions::default()),
            memory_events: None,
            next_memory_generation: 0,
            cancellable_waits: BTreeSet::new(),
            cancelled_waits: BTreeSet::new(),
            pending_wait_cancellations: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn with_keyboard_host(mut self, host: nixe_applets::keyboard::KeyboardHost) -> Self {
        self.keyboard_host = Some(host);
        self
    }

    /// Sets the language selected from this application's control metadata.
    /// Without it, GetDesiredLanguage reports unsupported application metadata.
    #[must_use]
    pub fn with_application_language(mut self, language: crate::SystemLanguage) -> Self {
        self.application_language = Some(language);
        self
    }

    /// Installs the account-switch policy from the launched application's NACP.
    #[must_use]
    pub fn with_user_account_switch_lock(mut self, locked: bool) -> Self {
        self.user_account_switch_locked = Some(locked);
        self
    }

    /// Supplies host-generated resources for filesystem system-data requests.
    #[must_use]
    pub fn with_system_archives(mut self, archives: crate::SystemArchives) -> Self {
        self.system_archives = std::sync::Arc::new(archives);
        self
    }

    /// Installs the launched application's persistent save-data namespace.
    #[must_use]
    pub fn with_save_data(mut self, save_data: crate::SaveDataSystem) -> Self {
        self.save_data = Some(save_data);
        self
    }

    /// Applies guest-log and filesystem diagnostic policy to service dispatch.
    #[must_use]
    pub fn with_diagnostics(mut self, diagnostics: crate::HorizonDiagnostics) -> Self {
        self.diagnostics = diagnostics;
        self
    }

    /// Supplies a host PCM backend. Audio remains explicitly unavailable until configured.
    pub fn with_audio_backend(
        mut self,
        backend: std::sync::Arc<dyn nixe_audio::AudioBackend>,
    ) -> Self {
        self.audio_backend = Some(backend);
        self
    }

    /// Supplies real controller actuator output owned by the host input worker.
    pub fn with_vibration_output(self, output: nixe_input::VibrationOutput) -> Self {
        self.hid_system.set_vibration_output(output);
        self
    }

    #[must_use]
    pub fn video_system(&self) -> crate::VideoSystem {
        self.video_system.clone()
    }

    /// Advances the guest display clock and signals VI VSync when due.
    pub fn advance_video(&self, elapsed: Duration) -> Result<u64, crate::FramebufferError> {
        self.video_system.advance(elapsed)
    }

    /// Parks audio command execution while preserving guest session state.
    pub fn set_host_paused(&self, paused: bool) {
        self.audio_renderers.set_host_paused(paused);
    }

    pub fn next_video_deadline(&self) -> Duration {
        self.video_system.next_display_deadline()
    }

    /// Publishes the latest player-one controller state to Horizon HID.
    pub fn advance_input(
        &mut self,
        state: Option<&nixe_input::EmulatedControllerState>,
        delta: Duration,
    ) -> Result<(), crate::HidInputError> {
        self.hid_system.publish(state, delta)
    }

    /// Publishes the latest host touch-screen state to Horizon HID.
    pub fn advance_touch_screen(
        &mut self,
        state: &nixe_input::EmulatedTouchScreenState,
        delta: Duration,
    ) -> Result<(), crate::HidInputError> {
        self.hid_system.publish_touch_screen(state, delta)
    }

    pub fn poll_input_publication(&self) -> Result<(), crate::HidInputError> {
        self.hid_system.poll_publication()
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

    fn queue_runtime_request(
        &mut self,
        thread: GuestThreadId,
        request: PendingRuntimeRequest,
        operation: &'static str,
    ) -> Result<(), HorizonSvcFault> {
        match self.pending_runtime_requests.entry(thread) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(request);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(_) => Err(runtime_fault(operation)),
        }
    }

    fn suspend_for_runtime_request(
        &mut self,
        thread: GuestThreadId,
        request: PendingRuntimeRequest,
        operation: &'static str,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match self.queue_runtime_request(thread, request, operation) {
            Ok(()) => ExceptionDispatchOutcome::Suspend(ExceptionResume::Next),
            Err(fault) => ExceptionDispatchOutcome::Fault(fault),
        }
    }

    /// Applies one Horizon-to-runtime scheduler operation staged during SVC
    /// dispatch. Returns `Ok(false)` when the calling thread staged none.
    pub fn apply_pending_runtime_request(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        process_id: ProcessId,
        thread_id: GuestThreadId,
    ) -> Result<Option<nixe_runtime::ExceptionHandlingResult<HorizonSvcFault>>, HorizonSvcFault>
    {
        let Some(request) = self.pending_runtime_requests.remove(&thread_id) else {
            return Ok(None);
        };
        let mut fully_handled = true;
        match request {
            PendingRuntimeRequest::RegisterAddressWait {
                address,
                deadline,
                check,
                source,
                vcpu,
            } => {
                let priority = coordinator
                    .scheduler()
                    .thread(thread_id)
                    .ok_or_else(|| runtime_fault("WaitForAddress thread priority"))?
                    .effective_priority;
                let event = {
                    let mut state = coordinator
                        .process(process_id)
                        .ok_or_else(|| runtime_fault("WaitForAddress process"))?
                        .address_waits()
                        .lock();
                    let queue = state.priority_waits_mut();
                    let event = queue
                        .enqueue(address, thread_id, priority, deadline)
                        .ok_or_else(|| runtime_fault("WaitForAddress duplicate registration"))?;
                    queue.begin_check(thread_id);
                    event
                };
                self.pending_wakes.insert(
                    thread_id.get(),
                    PendingThreadWake {
                        events: vec![event],
                        deadline,
                    },
                );
                let token = coordinator
                    .register_wait(thread_id, nixe_scheduler::Readiness::Pending)
                    .map_err(|_| runtime_fault("address comparison registration"))?
                    .ok_or_else(|| runtime_fault("address comparison generation"))?;
                let handling = coordinator
                    .resume_supervisor_call(token, vcpu, source, |context| {
                        self.start_address_wait_check(context, address, check)
                    })
                    .map_err(|_| runtime_fault("address comparison continuation"))?
                    .ok_or_else(|| runtime_fault("address comparison lost generation"))?;
                return self
                    .finish_scheduled_call(coordinator, process_id, thread_id, handling)
                    .map(Some)
                    .map_err(|_| runtime_fault("address comparison scheduling"));
            }
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
                    return Ok(Some(nixe_runtime::ExceptionHandlingResult::Resumed));
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
                        return Ok(Some(nixe_runtime::ExceptionHandlingResult::Resumed));
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
                let result = coordinator.start_thread(object_id);
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
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "StartThread",
                    code,
                    |_| {},
                )?;
            }
            PendingRuntimeRequest::CancelSynchronization { object_id } => {
                let info = coordinator
                    .thread_scheduling_info(object_id)
                    .map_err(|_| runtime_fault("CancelSynchronization target"))?;
                let cancelled = if self.cancellable_waits.contains(&object_id) {
                    if let Some(token) = info.active_wait {
                        coordinator
                            .cancel_wait(token)
                            .map_err(|_| runtime_fault("CancelSynchronization wake"))?
                    } else {
                        false
                    }
                } else {
                    false
                };
                if cancelled {
                    self.cancellable_waits.remove(&object_id);
                    self.cancelled_waits.insert(object_id);
                } else {
                    self.pending_wait_cancellations.insert(object_id);
                }
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "CancelSynchronization",
                    HorizonKernelResult::SUCCESS,
                    |_| {},
                )?;
            }
            PendingRuntimeRequest::GetThreadPriority { object_id } => {
                let info = coordinator.thread_scheduling_info(object_id);
                let (code, priority) = match info {
                    Ok(info) => (HorizonKernelResult::SUCCESS, Some(info.base_priority)),
                    Err(nixe_runtime::ThreadOperationError::InvalidHandle) => {
                        (HorizonKernelResult::INVALID_HANDLE, None)
                    }
                    Err(_) => return Err(runtime_fault("GetThreadPriority")),
                };
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "GetThreadPriority",
                    code,
                    |state| {
                        if let Some(priority) = priority {
                            write_register(state, 1, priority as u32 as u64);
                        }
                    },
                )?;
            }
            PendingRuntimeRequest::SetThreadPriority {
                object_id,
                priority,
            } => {
                let code = map_thread_operation(
                    coordinator.set_thread_priority(object_id, priority),
                    "SetThreadPriority",
                )?;
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "SetThreadPriority",
                    code,
                    |_| {},
                )?;
            }
            PendingRuntimeRequest::GetThreadCoreMask { object_id } => {
                let info = coordinator.thread_scheduling_info(object_id);
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
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "GetThreadCoreMask",
                    code,
                    |state| {
                        if let Some((ideal, mask)) = values {
                            write_register(state, 1, ideal as u32 as u64);
                            write_u64(state, 2, mask);
                        }
                    },
                )?;
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
                        finish_pending_caller(
                            coordinator,
                            process_id,
                            thread_id,
                            "SetThreadCoreMask",
                            HorizonKernelResult::OUT_OF_RANGE,
                            |_| {},
                        )?;
                        return Ok(Some(nixe_runtime::ExceptionHandlingResult::Resumed));
                    };
                    map_thread_operation(
                        coordinator.set_thread_affinity(object_id, ideal, affinity),
                        "SetThreadCoreMask",
                    )?
                };
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "SetThreadCoreMask",
                    code,
                    |_| {},
                )?;
            }
            PendingRuntimeRequest::SleepThread { nanoseconds } => {
                coordinator
                    .sleep_thread(thread_id, nanoseconds)
                    .map_err(|_| runtime_fault("SleepThread"))?;
            }
            PendingRuntimeRequest::SetThreadActivity { object_id, paused } => {
                let info = coordinator
                    .thread_scheduling_info(object_id)
                    .map_err(|error| match error {
                        nixe_runtime::ThreadOperationError::InvalidHandle => {
                            runtime_fault("SetThreadActivity")
                        }
                        _ => runtime_fault("SetThreadActivity"),
                    })?;
                let code = if info.thread == thread_id {
                    HorizonKernelResult::INVALID_STATE
                } else {
                    map_thread_operation(
                        coordinator.set_thread_activity(object_id, paused),
                        "SetThreadActivity",
                    )?
                };
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "SetThreadActivity",
                    code,
                    |_| {},
                )?;
            }
            PendingRuntimeRequest::GetThreadContext { object_id, ranges } => {
                let info = coordinator
                    .thread_scheduling_info(object_id)
                    .map_err(|_| runtime_fault("GetThreadContext3"))?;
                if info.thread == thread_id || !info.paused {
                    finish_pending_caller(
                        coordinator,
                        process_id,
                        thread_id,
                        "GetThreadContext3",
                        HorizonKernelResult::INVALID_STATE,
                        |_| {},
                    )?;
                } else {
                    let state = coordinator
                        .thread_cpu_state(object_id)
                        .map_err(|_| runtime_fault("GetThreadContext3"))?;
                    let bytes = encode_thread_context(&state);
                    self.write_runtime_output(
                        coordinator,
                        process_id,
                        thread_id,
                        0x33,
                        ranges,
                        &bytes,
                    )?;
                }
            }
            PendingRuntimeRequest::SynchronizeMutexWaits { addresses } => {
                coordinator
                    .synchronize_mutex_waits(process_id, &addresses)
                    .map_err(|_| runtime_fault("mutex priority handoff"))?;
                fully_handled = false;
            }
            PendingRuntimeRequest::ReapThread { object_id } => {
                self.cancellable_waits.remove(&object_id);
                self.cancelled_waits.remove(&object_id);
                self.pending_wait_cancellations.remove(&object_id);
                coordinator
                    .reap_thread(object_id)
                    .map_err(|_| runtime_fault("CloseHandle thread reaping"))?;
                finish_pending_caller(
                    coordinator,
                    process_id,
                    thread_id,
                    "CloseHandle thread reaping",
                    HorizonKernelResult::SUCCESS,
                    |_| {},
                )?;
            }
        }
        Ok(fully_handled.then_some(nixe_runtime::ExceptionHandlingResult::Resumed))
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
        finish_pending_caller(
            coordinator,
            process_id,
            caller,
            "CreateThread",
            code,
            |state| {
                if let Some(handle) = handle {
                    write_register(state, 1, u64::from(handle));
                }
            },
        )
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
            0x01 => self.set_heap_size(context),
            0x02 => set_memory_permission(context),
            0x03 => set_memory_attribute(context),
            0x04 => map_memory(context),
            0x05 => unmap_memory(context),
            0x06 => self.query_memory(context, immediate),
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
            0x13 => map_shared_memory(context),
            0x14 => unmap_shared_memory(context),
            0x15 => create_transfer_memory(context),
            0x16 => close_handle(self, context),
            0x17 => reset_signal(context),
            0x18 => self.wait_synchronization(context),
            0x19 => self.cancel_synchronization(context),
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
            0x26 => self.break_process(context),
            0x27 => self.output_debug_string(context),
            0x29 => get_info(context),
            0x32 => self.set_thread_activity(context),
            0x33 => self.get_thread_context(context),
            0x34 => self.wait_for_address(context),
            0x35 => self.signal_to_address(context),
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
        trace_completed_svc_result(*descriptor, context.thread().state(), &outcome);
        outcome
    }
}

const fn svc_support(immediate: u32) -> HorizonSvcSupport {
    match immediate {
        0x04 | 0x05 | 0x07 | 0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x0d | 0x0e | 0x0f | 0x10
        | 0x13 | 0x14 | 0x15 | 0x16 | 0x19 | 0x25 | 0x27 | 0x32 | 0x34 | 0x35 | 0x40 | 0x41
        | 0x45 | 0x70 | 0x71 | 0x72 => HorizonSvcSupport::Complete,
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
    let Some(guest_result) = diagnostic.guest_result() else {
        return ExceptionDispatchOutcome::Fault(diagnostic);
    };
    result(context, guest_result);
    ExceptionDispatchOutcome::Reject { diagnostic }
}

fn result(context: &mut ExceptionDispatchContext<'_>, value: HorizonKernelResult) {
    write_register(context.thread_mut().state_mut(), 0, u64::from(value.raw()));
}

fn trace_completed_svc_result(
    descriptor: HorizonSvcDescriptor,
    state: &GuestCpuState,
    outcome: &ExceptionDispatchOutcome<HorizonSvcFault>,
) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }
    let Some(code) = completed_svc_guest_error(descriptor, state, outcome) else {
        return;
    };
    let name = descriptor
        .unambiguous_name()
        .unwrap_or("version-dependent SVC");
    trace_guest_result(Some(descriptor.immediate()), name, code);
}

fn completed_svc_guest_error(
    descriptor: HorizonSvcDescriptor,
    state: &GuestCpuState,
    outcome: &ExceptionDispatchOutcome<HorizonSvcFault>,
) -> Option<HorizonKernelResult> {
    if descriptor.return_kind() != HorizonSvcReturnKind::Result
        || matches!(
            outcome,
            ExceptionDispatchOutcome::Resume(ExceptionResume::Retry)
                | ExceptionDispatchOutcome::Suspend(_)
                | ExceptionDispatchOutcome::Terminate { .. }
                | ExceptionDispatchOutcome::Fault(_)
        )
    {
        return None;
    }
    let code = HorizonKernelResult(read_register(state, 0) as u32);
    if code == HorizonKernelResult::SUCCESS {
        return None;
    }
    Some(code)
}

fn trace_guest_result(immediate: Option<u32>, name: &str, code: HorizonKernelResult) {
    if code == HorizonKernelResult::SUCCESS || !log::log_enabled!(log::Level::Trace) {
        return;
    }
    if let Some(immediate) = immediate {
        log::trace!(
            "Horizon SVC returned guest error: immediate={immediate:#x} name={name} result={}({:#x})",
            code.name(),
            code.raw()
        );
    } else {
        log::trace!(
            "Horizon SVC completed asynchronously with guest error: name={name} result={}({:#x})",
            code.name(),
            code.raw()
        );
    }
}

fn thread_tls(state: &GuestCpuState) -> GuestVirtualAddress {
    // Horizon owns TPIDRRO_EL0 and uses it as the fixed thread-local IPC buffer base.
    // TPIDR_EL0 belongs to userspace and software is free to repurpose it (including
    // clearing it), so it cannot be used to locate the kernel-managed TLS region.
    GuestVirtualAddress::new(state.tpidrro_el0())
}

fn session_request_owner(context: &ExceptionDispatchContext<'_>) -> SessionRequestOwner {
    SessionRequestOwner {
        process_id: context.process().process_id(),
        thread_id: context.thread().object().thread_id(),
    }
}

fn read_register(state: &GuestCpuState, index: u8) -> u64 {
    state.read_x(A64Register::General(
        A64GeneralRegister::new(index).expect("A64 ABI register index is valid"),
    ))
}

const fn runtime_fault(operation: &'static str) -> HorizonSvcFault {
    HorizonSvcFault::InternalRuntime { operation }
}

fn finish_pending_caller(
    coordinator: &mut nixe_runtime::RuntimeCoordinator,
    process_id: ProcessId,
    thread_id: GuestThreadId,
    operation: &'static str,
    code: HorizonKernelResult,
    write_output: impl FnOnce(&mut GuestCpuState),
) -> Result<(), HorizonSvcFault> {
    let state = coordinator
        .process_mut(process_id)
        .and_then(|process| process.thread_mut(thread_id))
        .map(nixe_runtime::GuestThread::state_mut)
        .ok_or_else(|| runtime_fault(operation))?;
    write_register(state, 0, u64::from(code.raw()));
    write_output(state);
    coordinator
        .make_thread_ready(thread_id)
        .map_err(|_| runtime_fault(operation))?;
    trace_guest_result(None, operation, code);
    Ok(())
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

fn put_context_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_context_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_reply_timeout(state: &GuestCpuState, user_buffer: bool) -> i64 {
    read_register(state, if user_buffer { 6 } else { 4 }) as i64
}

fn write_register(state: &mut GuestCpuState, index: u8, value: u64) {
    state.write_x(
        A64Register::General(
            A64GeneralRegister::new(index).expect("A64 ABI register index is valid"),
        ),
        value,
    );
}

fn write_u64(state: &mut GuestCpuState, index: u8, value: u64) {
    write_register(state, index, value);
}

fn read_wait_timeout(state: &GuestCpuState) -> i64 {
    read_register(state, 3) as i64
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
    let closed = context.process_mut().handles().close(handle);
    let Ok(object) = closed else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    let reap = (object.strong_count() == 1)
        .then(|| object.downcast_ref::<ThreadObject>())
        .flatten()
        .filter(|thread| thread.is_signalled())
        .map(|thread| thread.thread_id());
    let caller = context.thread().id();
    let Some(object_id) = reap else {
        result(context, HorizonKernelResult::SUCCESS);
        return resume();
    };
    if let Err(fault) = dispatcher.queue_runtime_request(
        caller,
        PendingRuntimeRequest::ReapThread { object_id },
        "CloseHandle thread reaping",
    ) {
        return ExceptionDispatchOutcome::Fault(fault);
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
        .lookup_as::<WritableEventObject>(handle)
        .map(std::sync::Arc::unwrap_or_clone);
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
        .lookup_as::<WritableEventObject>(handle)
        .map(std::sync::Arc::unwrap_or_clone);
    let readable = context
        .process()
        .handles()
        .lookup_as::<ReadableEventObject>(handle)
        .map(std::sync::Arc::unwrap_or_clone);
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
        .lookup_as::<ReadableEventObject>(handle)
        .map(std::sync::Arc::unwrap_or_clone);
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
    match insert_pair(context.process_mut().handles(), writable, readable) {
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
    match insert_pair(context.process_mut().handles(), server, client) {
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
    match insert_pair(context.process_mut().handles(), server, client) {
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
        .lookup_as::<PortObject>(handle)
        .map(std::sync::Arc::unwrap_or_clone)
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
    match context.process_mut().handles().insert(session) {
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
        .lookup_as::<PortObject>(handle)
        .map(std::sync::Arc::unwrap_or_clone)
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if port.endpoint() != PortEndpoint::Server {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    match port.accept() {
        Ok(session) => match context.process_mut().handles().insert(session) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_runtime::EventObject;

    #[test]
    fn thread_context_encoding_uses_horizon_a64_layout() {
        let encoded = encode_thread_context(&ThreadCpuState::default());
        assert_eq!(encoded.len(), 0x320);
    }

    #[test]
    fn completed_result_svc_errors_are_selected_for_generic_tracing() {
        let mut state = GuestCpuState::default();
        write_register(
            &mut state,
            0,
            u64::from(HorizonKernelResult::RESOURCE_LIMIT.raw()),
        );
        let completed = ExceptionDispatchOutcome::Resume(ExceptionResume::Next);
        let retrying = ExceptionDispatchOutcome::Resume(ExceptionResume::Retry);

        assert_eq!(
            completed_svc_guest_error(*decode_horizon_svc(0x01).unwrap(), &state, &completed),
            Some(HorizonKernelResult::RESOURCE_LIMIT)
        );
        assert!(
            (completed_svc_guest_error(*decode_horizon_svc(0x10).unwrap(), &state, &completed))
                .is_none()
        );
        assert!(
            (completed_svc_guest_error(*decode_horizon_svc(0x01).unwrap(), &state, &retrying))
                .is_none()
        );

        write_register(&mut state, 0, u64::from(HorizonKernelResult::SUCCESS.raw()));
        assert!(
            (completed_svc_guest_error(*decode_horizon_svc(0x01).unwrap(), &state, &completed))
                .is_none()
        );
    }

    #[test]
    fn failed_handle_materialization_rolls_back_partial_allocations() {
        let retained = HandleObject::new(EventObject::new());
        let handles = std::sync::Arc::new(HandleTable::with_capacity_limit(2));
        let existing = handles.insert(ThreadObject::new(1)).unwrap();
        let mut bytes = vec![0_u8; 20];
        bytes[0..4].copy_from_slice(&4_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(1_u32 << 31).to_le_bytes());
        bytes[8..12].copy_from_slice(&(2_u32 << 1).to_le_bytes());

        assert!(matches!(
            materialize_message_handles_in_table(
                &handles,
                SessionMessage::TransportedBuffer {
                    bytes,
                    copy_handles: vec![Some(retained.clone()), Some(retained)],
                    move_handles: Vec::new(),
                },
            ),
            Err(HorizonKernelResult::OUT_OF_HANDLES)
        ));
        assert_eq!(handles.len(), 1);
        assert!(handles.lookup_as::<ThreadObject>(existing).is_some());
    }

    #[test]
    fn unsupported_horizon_semantics_have_no_guest_result() {
        let fault = HorizonSvcFault::UnsupportedSemantics {
            immediate: 0x23,
            documented_name: "WaitForAddress",
        };

        assert!((fault.guest_result()).is_none());
    }

    #[test]
    fn invalid_emulator_ipc_state_has_no_guest_result() {
        let fault = HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::from_wire(IpcWireError::Internal(
                "synthetic internal invariant",
            ))),
        };

        assert!((fault.guest_result()).is_none());
    }

    #[test]
    fn malformed_ipc_has_no_unverified_guest_result() {
        let fault = HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::malformed("synthetic malformed request")),
        };

        assert!((fault.guest_result()).is_none());
    }

    #[test]
    fn error_applet_content_is_reported_as_a_fatal_host_fault() {
        let diagnostic = crate::ErrorAppletDiagnostic::decode(&[]).with_launch_mode(0);
        let fault = HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::from_wire(IpcWireError::ErrorApplet(
                Box::new(diagnostic),
            ))),
        };

        assert!((fault.guest_result()).is_none());
        let rendered = fault.to_string();
        assert!(rendered.contains("unimplemented Error library applet"));
        assert!(rendered.contains("mode=AllForeground"));
        assert!(rendered.contains("storages=[]"));
        assert!(rendered.contains("header=[missing]"));
    }

    #[test]
    fn host_ipc_resource_exhaustion_has_no_guest_result() {
        let fault = HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::from_wire(
                IpcWireError::HostResourceExhausted("allocating a synthetic response"),
            )),
        };

        assert!((fault.guest_result()).is_none());
    }

    #[test]
    fn unsupported_nvdrv_semantics_have_no_guest_result() {
        let fault = HorizonSvcFault::Ipc {
            immediate: 0x21,
            fault: Box::new(HorizonIpcFault::from_wire(IpcWireError::UnsupportedNvDrv(
                crate::nvdrv::UnsupportedNvDrvOperation::Ioctl {
                    context: crate::nvdrv::NvDrvErrorContext::new(
                        crate::nvdrv::NvDrvDeviceKind::HostControlGpu,
                        0xc018_4706,
                        crate::nvdrv::NvDrvFileDescriptor::new(3),
                        None,
                        crate::nvdrv::NvDrvValidationReason::UnsupportedOperation,
                    ),
                },
            ))),
        };

        assert!((fault.guest_result()).is_none());
        assert_eq!(
            fault.to_string(),
            "Horizon SVC 0x21 reached unsupported emulator semantics: graphics-gap=ioctl nvdrv \
             ioctl is not implemented: device=/dev/nvhost-ctrl-gpu request=0xc0184706 \
             fd=nvfd:0x00000003 reason=unsupported-operation"
        );
    }

    #[test]
    fn emulator_generation_exhaustion_is_never_fabricated_as_a_guest_result() {
        let faults = [
            HorizonSvcFault::MemoryAccess {
                immediate: 0x21,
                fault: std::sync::Arc::new(nixe_memory::MemoryError::Invalid(
                    "mapping identity exhausted",
                )),
            },
            HorizonSvcFault::MemoryMapping {
                fault: std::sync::Arc::new(nixe_runtime::memory::MappingError::Memory(
                    std::sync::Arc::new(nixe_memory::MemoryError::Invalid(
                        "mapping identity exhausted",
                    )),
                )),
            },
        ];

        for fault in faults {
            assert!((fault.guest_result()).is_none());
        }
    }

    #[test]
    fn canonical_backing_failures_are_never_fabricated_as_guest_results() {
        let faults = [HorizonSvcFault::MemoryAccess {
            immediate: 0x15,
            fault: std::sync::Arc::new(nixe_memory::MemoryError::Execution(
                "device access failed".into(),
            )),
        }];

        for fault in faults {
            assert!((fault.guest_result()).is_none());
        }
    }
}
