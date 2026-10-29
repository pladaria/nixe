//! Horizon SVC dispatch and owned kernel continuations.

use std::collections::BTreeMap;
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
    HandleObject, HandleTable, PortEndpoint, PortError, PortObject, ReadableEventObject,
    SessionEndpoint, SessionError, SessionMessage, SessionObject, SessionRequestOwner,
    SharedMemoryObject, ThreadCreateError, ThreadCreateRequest, ThreadObject, TransferMemoryObject,
    WritableEventObject,
};
use nixe_scheduler::{GuestThreadId, ProcessId, VirtualCpuId};

use crate::ipc_wire::message::HipcHeader;
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
    pub const INVALID_PRIORITY: Self = Self(0xe001);
    pub const INVALID_CORE_ID: Self = Self(0xe201);
    pub const INVALID_POINTER: Self = Self(0xe601);
    pub const INVALID_ADDRESS: Self = Self(0xcc01);
    pub const INVALID_SIZE: Self = Self(0xca01);
    pub const INVALID_CURRENT_MEMORY: Self = Self(0xd401);
    pub const INVALID_MEMORY_REGION: Self = Self(0xdc01);
    pub const OUT_OF_RESOURCE: Self = Self(0xce01);
    pub const OUT_OF_MEMORY: Self = Self(0xd001);
    pub const TIMED_OUT: Self = Self(0xea01);
    pub const CANCELLED: Self = Self(0xec01);
    pub const OUT_OF_RANGE: Self = Self(0xee01);
    // https://switchbrew.org/wiki/Error_codes#Kernel
    pub const INVALID_ENUM_VALUE: Self = Self(0xf001);
    pub const BUSY: Self = Self(0xf401);
    pub const INVALID_STATE: Self = Self(0xfa01);
    pub const RESOURCE_LIMIT: Self = Self(0x10801);
    pub const NOT_SUPPORTED: Self = Self(0xfe01);
    pub const NOT_FOUND: Self = Self(0xf201);
    pub const SESSION_CLOSED: Self = Self(0xf601);
    pub const PORT_CLOSED: Self = Self(0x10601);
    pub const OUT_OF_HANDLES: Self = Self(0xd201);
    pub const INVALID_COMBINATION: Self = Self(0xe801);
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc_results.h#L43
    pub const MESSAGE_TOO_LARGE: Self = Self(0x20801);

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
            Self::INVALID_PRIORITY => "InvalidPriority",
            Self::INVALID_CORE_ID => "InvalidCoreId",
            Self::INVALID_POINTER => "InvalidPointer",
            Self::INVALID_ADDRESS => "InvalidAddress",
            Self::INVALID_SIZE => "InvalidSize",
            Self::INVALID_CURRENT_MEMORY => "InvalidCurrentMemory",
            Self::INVALID_MEMORY_REGION => "InvalidMemoryRegion",
            Self::OUT_OF_RESOURCE => "OutOfResource",
            Self::OUT_OF_MEMORY => "OutOfMemory",
            Self::TIMED_OUT => "TimedOut",
            Self::CANCELLED => "Cancelled",
            Self::OUT_OF_RANGE => "OutOfRange",
            Self::INVALID_ENUM_VALUE => "InvalidEnumValue",
            Self::BUSY => "Busy",
            Self::INVALID_STATE => "InvalidState",
            Self::RESOURCE_LIMIT => "ResourceLimit",
            Self::NOT_SUPPORTED => "NotSupported",
            Self::NOT_FOUND => "NotFound",
            Self::SESSION_CLOSED => "SessionClosed",
            Self::PORT_CLOSED => "PortClosed",
            Self::OUT_OF_HANDLES => "OutOfHandles",
            Self::INVALID_COMBINATION => "InvalidCombination",
            Self::MESSAGE_TOO_LARGE => "MessageTooLarge",
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
    Scheduling {
        operation: &'static str,
        fault: nixe_scheduler::SchedulerError,
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
            Self::InternalRuntime { .. } | Self::Scheduling { .. } => None,
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
            Self::Scheduling { operation, fault } => {
                write!(formatter, "Horizon scheduling {operation} failed: {fault}")
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

/// Table-driven Horizon exception dispatcher for the current minimal subset.
#[derive(Debug)]
pub struct HorizonSvcDispatcher {
    svc_calls: u64,
    initial_operation_mode: crate::OperationMode,
    time_environment: crate::TimeEnvironment,
    settings_environment: crate::SettingsEnvironment,
    application_language: Option<crate::SystemLanguage>,
    application_display_version: Option<[u8; 16]>,
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
    opus: crate::OpusManagerSession,
    named_ports: std::sync::Arc<std::sync::Mutex<BTreeMap<Vec<u8>, PortObject>>>,
    virtual_clock: nixe_runtime::VirtualClock,
    svc_waits: BTreeMap<GuestThreadId, std::sync::Arc<memory_wait::SvcWait>>,
    service_scheduling: std::sync::Arc<crate::ipc_wire::operation::ServiceScheduling>,
    memory_completions: std::sync::Arc<memory_wait::SvcCompletions>,
    next_memory_generation: u64,
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
        let video = crate::VideoSystem::new(machine, hid_system.kernel());
        Self::new_with_video(
            initial_operation_mode,
            time_environment,
            hid_system,
            shared_fonts,
            video,
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
        let audio_renderers = crate::AudioRendererManagerSession::new(hid_system.kernel());
        let service_scheduling = std::sync::Arc::new(
            crate::ipc_wire::operation::ServiceScheduling::new(hid_system.kernel()),
        );
        Self {
            svc_calls: 0,
            initial_operation_mode,
            time_environment,
            settings_environment,
            application_language: None,
            application_display_version: None,
            user_account_switch_locked: None,
            save_data: None,
            shared_fonts,
            system_archives: std::sync::Arc::new(crate::SystemArchives::default()),
            keyboard_host: None,
            diagnostics: crate::HorizonDiagnostics::default(),
            video_system,
            hid_system,
            audio_backend: None,
            audio_renderers,
            opus: crate::OpusManagerSession::default(),
            named_ports: std::sync::Arc::default(),
            virtual_clock,
            svc_waits: BTreeMap::new(),
            service_scheduling,
            memory_completions: std::sync::Arc::new(memory_wait::SvcCompletions::default()),
            next_memory_generation: 0,
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

    /// Installs the NACP display version, including its required terminating NUL.
    #[must_use]
    pub fn with_application_display_version(mut self, version: &str) -> Self {
        let mut display_version = [0; 16];
        let len = version.len().min(display_version.len() - 1);
        display_version[..len].copy_from_slice(&version.as_bytes()[..len]);
        self.application_display_version = Some(display_version);
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

    /// Number of decoded guest SVC invocations used by execution summaries.
    #[must_use]
    pub const fn svc_calls(&self) -> u64 {
        self.svc_calls
    }

    fn virtual_time_ns(&self) -> u64 {
        self.virtual_clock.scheduler_time_ns()
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
        let outcome = match immediate {
            0x01 => self.set_heap_size(context),
            0x02 => set_memory_permission(context),
            0x03 => set_memory_attribute(context),
            0x04 => map_memory(context),
            0x05 => unmap_memory(context),
            0x06 => self.query_memory(context, immediate),
            0x07 => terminate(ExceptionTerminationScope::Process),
            0x08 => self.create_thread(context),
            0x09 => thread::start_thread(context),
            0x0a => terminate(ExceptionTerminationScope::CurrentThread),
            0x0b => self.sleep_thread(context),
            0x0c => thread::get_thread_priority(context),
            0x0d => thread::set_thread_priority(context),
            0x0e => thread::get_thread_core_mask(context),
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
            0x16 => close_handle(context),
            0x17 => reset_signal(context),
            0x18 => self.wait_synchronization(context),
            0x19 => thread::cancel_synchronization(context),
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
            0x36 => self.synchronize_preemption_state(context),
            0x40 => create_session(context),
            0x41 => accept_session(context),
            0x42 => self.reply_and_receive_light(context),
            0x43 => self.reply_and_receive(context, false),
            0x44 => self.reply_and_receive(context, true),
            0x45 => create_event(context),
            0x70 => create_port(context),
            0x71 => self.manage_named_port(context),
            0x72 => connect_to_port(context),
            _ => {
                let descriptor = match decode_horizon_svc(immediate) {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Unknown(error));
                    }
                };
                ExceptionDispatchOutcome::Fault(HorizonSvcFault::UnsupportedSemantics {
                    immediate,
                    documented_name: descriptor
                        .unambiguous_name()
                        .unwrap_or("version-dependent SVC"),
                })
            }
        };
        self.svc_calls = self.svc_calls.saturating_add(1);
        trace_completed_svc_result(immediate, context.thread().state(), &outcome);
        outcome
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
    immediate: u32,
    state: &GuestCpuState,
    outcome: &ExceptionDispatchOutcome<HorizonSvcFault>,
) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }
    let descriptor = *decode_horizon_svc(immediate).expect("dispatched SVC has a registry entry");
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
        thread_id: context.thread().id().get(),
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

fn horizon_affinity_mask(affinity: &nixe_scheduler::CoreSet) -> Option<u64> {
    affinity.iter().try_fold(0_u64, |mask, vcpu| {
        1_u64.checked_shl(vcpu.get()).map(|bit| mask | bit)
    })
}

fn encode_thread_context(state: &ThreadCpuState) -> [u8; 0x320] {
    let mut bytes = [0_u8; 0x320];
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
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    if matches!(handle, CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE) {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let closed = context.process().handles().close(handle);
    let Ok(object) = closed else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    drop(object);
    result(context, HorizonKernelResult::SUCCESS);
    resume()
}

fn event_signal(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let event = context
        .process()
        .handles()
        .lookup_as::<WritableEventObject>(handle);
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
    let object = context.process().handles().lookup(handle);
    let code = if let Some(event) = object
        .as_ref()
        .and_then(|object| object.downcast_ref::<WritableEventObject>())
    {
        event.clear();
        HorizonKernelResult::SUCCESS
    } else if let Some(event) = object
        .as_ref()
        .and_then(|object| object.downcast_ref::<ReadableEventObject>())
    {
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
        .lookup_as::<ReadableEventObject>(handle);
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
    let (writable, readable) =
        nixe_runtime::EventObject::create_pair(context.process().scheduler());
    match insert_pair(context.process().handles(), writable, readable) {
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
        SessionObject::create_pair(context.process().scheduler())
    } else {
        SessionObject::create_light_pair(context.process().scheduler())
    };
    match insert_pair(context.process().handles(), server, client) {
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
    let (server, client) = PortObject::create_pair(
        context.process().scheduler(),
        max_sessions as usize,
        is_light,
    );
    match insert_pair(context.process().handles(), server, client) {
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
    let Some(port) = context.process().handles().lookup_as::<PortObject>(handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if port.endpoint() != PortEndpoint::Client {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    connect_port(context, &port)
}

fn connect_port(
    context: &mut ExceptionDispatchContext<'_>,
    port: &PortObject,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    // Both connection SVCs reserve the output handle before enqueueing a session.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_port.cpp
    let mut reservation = match context.process().handles().reserve(None) {
        Ok(reservation) => reservation,
        Err(_) => {
            result(context, HorizonKernelResult::OUT_OF_HANDLES);
            return resume();
        }
    };
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
    let session_handle = reservation.handle();
    reservation.set_object(HandleObject::new(session));
    match context.process().handles().publish(&mut [reservation]) {
        Ok(()) => {
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
    let Some(port) = context.process().handles().lookup_as::<PortObject>(handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if port.endpoint() != PortEndpoint::Server {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_session.cpp
    let mut reservation = match context.process().handles().reserve(None) {
        Ok(reservation) => reservation,
        Err(_) => {
            result(context, HorizonKernelResult::OUT_OF_HANDLES);
            return resume();
        }
    };
    match port.accept() {
        Ok(session) => {
            let session_handle = reservation.handle();
            reservation.set_object(HandleObject::new(session));
            match context.process().handles().publish(&mut [reservation]) {
                Ok(()) => {
                    result(context, HorizonKernelResult::SUCCESS);
                    write_register(
                        context.thread_mut().state_mut(),
                        1,
                        u64::from(session_handle),
                    );
                }
                Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
            }
        }
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
        let retained = HandleObject::new(EventObject::new(&crate::test_memory::kernel()));
        let handles = std::sync::Arc::new(HandleTable::with_capacity_limit(2));
        let existing = handles
            .insert(EventObject::new(&crate::test_memory::kernel()))
            .unwrap();
        let mut bytes = vec![0_u8; 20];
        bytes[0..4].copy_from_slice(&4_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&(1_u32 << 31).to_le_bytes());
        bytes[8..12].copy_from_slice(&(2_u32 << 1).to_le_bytes());

        assert!(matches!(
            materialize_message_handles(
                &handles,
                SessionMessage::TransportedBuffer {
                    bytes,
                    copy_handles: vec![Some(retained.clone()), Some(retained)],
                    move_handles: Vec::new(),
                },
                TLS_COMMAND_BUFFER_SIZE,
            ),
            Err(HorizonKernelResult::OUT_OF_HANDLES)
        ));
        assert_eq!(handles.len(), 1);
        assert!(handles.lookup_as::<EventObject>(existing).is_some());
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
