//! Synchronous Horizon service-call adapter built on the checked wire codec.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

mod codec;
mod semantic;
mod services;

use codec::{
    add, cmif_error, decode_service_name, encode_domain_response, encode_response,
    has_ipc_descriptors, read_byte, request_f32, request_i32, request_i64, request_u32,
    request_u64, write_descriptor_bytes, write_response,
};
pub(crate) use codec::{read_bytes, validate_writable_ram_range, write_bytes};
use semantic::{
    dispatch_plain_semantic_object, dispatch_semantic_service, one_auto_select_input,
    one_auto_select_output, one_receive_buffer, one_send_buffer, semantic_object_name,
    semantic_success,
};

use chrono::{Datelike, Offset, Timelike};
use chrono_tz::OffsetComponents;
use nixe_cpu::memory::{
    DataAccessFault, DataAccessFaultReason, DataAccessKind, MemoryAccess, MemoryAccessSize,
    MemoryPermissions, MemoryRegionKind, MemoryValue,
};
use nixe_memory::GuestVirtualAddress;
use nixe_runtime::{ExceptionProcessContext, TransferMemoryObject};

use crate::ipc_message::{
    BufferDescriptor, BufferMode, COMMAND_BUFFER_SIZE, CmifRequest, CmifResponse, DomainRequest,
    HipcRequest, MessageError, ReceiveStaticDescriptor, ReceiveStatics, SendStaticDescriptor,
};
use crate::nvdrv::NvDrvIoctlOutcome;
use crate::nvdrv::{NvDrvFileDescriptor, NvDrvService, NvDrvServiceError};
use crate::object::{
    AppletObject, AppletProxyKind, AppletStorageAccessError, CreateAppletStorageError,
    CreateLibraryAppletError, LibraryAppletId, LibraryAppletMode, OpenAppletStorageAccessorError,
    PrepareLibraryAppletLaunchError, PushLibraryAppletStorageError,
};
use crate::{
    AccountSession, AppletSession, DirectoryEntryKind, FileSystemAccessLogMode, HidAppletResource,
    HidSession, HidSystem, HorizonIpcObject, HorizonIpcResult, IpcDispatcher, IpcRequest,
    IpcResponse, IpcResultCode, IpcService, IpcSession, LogManagerSession, LoggerSession,
    MAX_IPC_LIST_ENTRIES, MAX_IPC_PATH_BYTES, MAX_IPC_READ_BYTES, NvDrvSession, OperationMode,
    ParentalControlFactorySession, ParentalControlSession, PerformanceManagerSession,
    PerformanceSession, SemanticIpcObject, ServiceManagerSession, SettingsEnvironment,
    SteadyClockSession, SystemClockKind, SystemClockSession, SystemLanguage, SystemSettingsSession,
    TimeEnvironment, TimeServiceSession, TimeZoneServiceSession, UserSettingsSession, ViObjectKind,
    ViServiceKind, ViSession, VideoSystem,
};

pub(crate) const NAMED_PORT_NAME_SIZE: usize = 12;
const CMIF_COMMAND_CLOSE: u16 = 2;
const CMIF_COMMAND_CONTROL: u16 = 5;
const CMIF_COMMAND_CONTROL_WITH_CONTEXT: u16 = 7;
const PERFORMANCE_MODE_NORMAL: u32 = 0;
const FS_MAX_PATH: usize = 0x301;
const FS_DIRECTORY_ENTRY_SIZE: usize = 0x310;
const FS_DIRECTORY_ENTRY_FILE: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CmifControlCommand {
    ConvertCurrentObjectToDomain,
    CloneCurrentObject,
    QueryPointerBufferSize,
    CloneCurrentObjectEx,
}

impl CmifControlCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::ConvertCurrentObjectToDomain),
            2 => Some(Self::CloneCurrentObject),
            3 => Some(Self::QueryPointerBufferSize),
            4 => Some(Self::CloneCurrentObjectEx),
            _ => None,
        }
    }

    const fn is_clone(self) -> bool {
        matches!(self, Self::CloneCurrentObject | Self::CloneCurrentObjectEx)
    }
}

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
        }
    }
}

fn unsupported_service_command<T>(
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

const fn semantic_service_name(service: IpcService) -> &'static str {
    match service {
        IpcService::FileSystem => "fsp-srv",
        IpcService::AddOnContent => "aoc:u",
    }
}

impl From<MessageError> for IpcWireError {
    fn from(error: MessageError) -> Self {
        Self::Internal(error.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NamedPortResult {
    Connected(u32),
    NotFound,
    NameOutOfRange,
    OutOfHandles,
}

pub(crate) fn connect_to_named_port(
    process: &mut ExceptionProcessContext<'_>,
    name_address: GuestVirtualAddress,
) -> Result<NamedPortResult, IpcWireError> {
    let mut name = [0_u8; NAMED_PORT_NAME_SIZE];
    for (index, byte) in name.iter_mut().enumerate() {
        *byte = read_byte(process, add(name_address, index)?)?;
        if *byte == 0 {
            let port_name = &name[..index];
            if port_name != b"sm:" {
                log::debug!(
                    "ConnectToNamedPort did not find named port {:?}",
                    String::from_utf8_lossy(port_name)
                );
                return Ok(NamedPortResult::NotFound);
            }
            log::debug!("ConnectToNamedPort opening a client session to sm:");
            return Ok(
                match process
                    .handles_mut()
                    .insert(HorizonIpcObject::ServiceManager(
                        ServiceManagerSession::new(),
                    )) {
                    Ok(handle) => NamedPortResult::Connected(handle),
                    Err(_) => NamedPortResult::OutOfHandles,
                },
            );
        }
    }
    Ok(NamedPortResult::NameOutOfRange)
}

#[derive(Clone, Copy)]
pub(crate) struct HostSystems<'a> {
    pub video: &'a VideoSystem,
    pub hid: &'a HidSystem,
    pub settings: &'a SettingsEnvironment,
    pub diagnostics: &'a crate::HorizonDiagnostics,
    pub caller_thread_id: u64,
}

impl HorizonIpcObject {
    fn is_domain(&self) -> bool {
        match self {
            Self::Applet(session) => session.is_domain(),
            Self::SemanticService(session) => session.is_domain(),
            Self::ParentalControl(session) => session.is_domain(),
            _ => false,
        }
    }

    fn diagnostic_name(&self, request: &CmifRequest<'_>) -> &'static str {
        let domain_object = match request.domain.as_ref() {
            Some(
                DomainRequest::SendMessage { object_id, .. } | DomainRequest::Close { object_id },
            ) => Some(*object_id),
            None => None,
        };
        match self {
            Self::ServiceManager(_) => "sm:",
            Self::SemanticService(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or_else(
                    || semantic_service_name(session.service()),
                    |object| semantic_object_name(&object),
                ),
            Self::SystemSettings(_) => "set:sys",
            Self::UserSettings(_) => "set",
            Self::PerformanceManager(_) => "apm",
            Self::Performance(_) => "IPerformanceSession",
            Self::Applet(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("appletOE", services::applet_object_name),
            Self::Account(_) => "acc:u0",
            Self::Hid(_) => "hid",
            Self::HidAppletResource(_) => "IAppletResource",
            Self::Time(_) => "time:u",
            Self::SystemClock(_) => "ISystemClock",
            Self::SteadyClock(_) => "ISteadyClock",
            Self::TimeZone(_) => "ITimeZoneService",
            Self::Vi(session) => services::vi_object_name(session.kind()),
            Self::NvDrv(_) => "nvdrv",
            Self::LogManager(_) => "lm",
            Self::Logger(_) => "ILogger",
            Self::ParentalControl(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("pctl", |_| "IParentalControlService"),
            Self::ParentalControlService(_) => "IParentalControlService",
            Self::SemanticObject(object) => semantic_object_name(object),
        }
    }
}

pub(crate) fn send_sync_request(
    process: &mut ExceptionProcessContext<'_>,
    tls: GuestVirtualAddress,
    handle: u32,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<SyncRequestResult, IpcWireError> {
    send_sync_request_from_buffer(
        process,
        tls,
        COMMAND_BUFFER_SIZE,
        handle,
        initial_operation_mode,
        time_environment,
        host_systems,
    )
}

pub(crate) fn send_sync_request_from_buffer(
    process: &mut ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    handle: u32,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<SyncRequestResult, IpcWireError> {
    let Some(target) = process
        .handles()
        .get_as::<HorizonIpcObject>(handle)
        .cloned()
    else {
        return Ok(SyncRequestResult::InvalidHandle);
    };

    if size < COMMAND_BUFFER_SIZE {
        return Err(IpcWireError::Malformed(
            "IPC message buffer is smaller than the TLS command buffer",
        ));
    }
    validate_writable_ram_range(process, address, size)?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(size)
        .map_err(|_| IpcWireError::HostResourceExhausted("allocating the IPC command buffer"))?;
    buffer.resize(size, 0);
    read_bytes(process, address, &mut buffer)?;
    let hipc = HipcRequest::decode(&buffer).map_err(|error| IpcWireError::Malformed(error.0))?;
    let request = CmifRequest::decode(&hipc, target.is_domain()).map_err(|error| {
        if error.0 == "unsupported HIPC command type for CMIF" {
            IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
                service: "CMIF transport",
                command_id: u32::from(hipc.command_type),
                detail: error.0,
            })
        } else {
            IpcWireError::Malformed(error.0)
        }
    })?;
    let trace_service =
        log::log_enabled!(log::Level::Trace).then(|| target.diagnostic_name(&request));
    if let Some(service) = trace_service {
        log::trace!(
            "SendSyncRequest service={service} handle={handle:#x} type={} command={} send_pid={} descriptors={}/{}/{}/{} handles={}/{}",
            request.command_type,
            request.command_id,
            hipc.pid.is_some(),
            hipc.send_statics.len(),
            hipc.send_buffers.len(),
            hipc.receive_buffers.len(),
            hipc.exchange_buffers.len(),
            hipc.copy_handles.len(),
            hipc.move_handles.len(),
        );
    }
    let command_type = request.command_type;
    let command_id = request.command_id;
    let trace_completion = |outcome: SyncRequestResult| {
        if let Some(service) = trace_service {
            log::trace!(
                "SendSyncRequest completed service={service} handle={handle:#x} type={command_type} command={command_id} outcome={outcome:?}"
            );
        }
        outcome
    };

    if request.command_type == CMIF_COMMAND_CLOSE {
        // CMIF closes the server-side session protocol, while the client's
        // kernel handle remains owned until the following CloseHandle. These
        // lifetimes must stay distinct: Nintendo SDK treats a failed
        // CloseHandle after this request as a fatal invariant violation.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L195-L209
        return Ok(trace_completion(SyncRequestResult::Success));
    }
    if matches!(
        request.command_type,
        CMIF_COMMAND_CONTROL | CMIF_COMMAND_CONTROL_WITH_CONTEXT
    ) {
        let Some(control_command) = CmifControlCommand::decode(request.command_id) else {
            return unsupported_service_command("CMIF control", request.command_id);
        };
        if control_command == CmifControlCommand::ConvertCurrentObjectToDomain
            && let HorizonIpcObject::Applet(applet) = &target
        {
            // libnx converts appletOE to a domain before opening the
            // application proxy. The control command and returned root object
            // ID follow its pinned CMIF implementation:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L250-L266
            let object_id = applet.convert_to_domain();
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &object_id.to_le_bytes(),
                None,
            )?;
            write_response(process, address, size, &response)?;
            log::debug!("appletOE converted to domain with root object {object_id:#x}");
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        if control_command == CmifControlCommand::ConvertCurrentObjectToDomain
            && let HorizonIpcObject::SemanticService(service) = &target
        {
            // Generic CMIF domain conversion and its root object response are
            // defined by libnx's pinned service implementation:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L250-L266
            let object_id = service.convert_to_domain();
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &object_id.to_le_bytes(),
                None,
            )?;
            write_response(process, address, size, &response)?;
            log::debug!(
                "{:?} converted to domain with root object {object_id:#x}",
                String::from_utf8_lossy(service.service().name())
            );
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        if control_command == CmifControlCommand::ConvertCurrentObjectToDomain
            && let HorizonIpcObject::ParentalControl(factory) = &target
        {
            // The public pctl client converts the factory before asking it to
            // create IParentalControlService. Keep that real domain boundary:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/pctl.c#L20-L24
            let object_id = factory.convert_to_domain();
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &object_id.to_le_bytes(),
                None,
            )?;
            write_response(process, address, size, &response)?;
            log::debug!("pctl converted to domain with root object {object_id:#x}");
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        if control_command.is_clone()
            && let HorizonIpcObject::SemanticService(service) = &target
        {
            // CloneCurrentObject (2) returns a moved session handle. The Ex
            // form (4) additionally carries a session-manager tag which the
            // public libnx ABI documents as unused by official servers:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L308-L337
            // A cloned `IpcSession` has a distinct process-handle object while
            // retaining the shared domain table of the source connection.
            let cloned_handle = process
                .handles_mut()
                .insert(HorizonIpcObject::SemanticService(service.clone()))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("cloning a CMIF session handle")
                })?;
            let response = match encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            ) {
                Ok(response) => response,
                Err(error) => {
                    let _ = process.handles_mut().close(cloned_handle);
                    return Err(error);
                }
            };
            if let Err(error) = write_response(process, address, size, &response) {
                let _ = process.handles_mut().close(cloned_handle);
                return Err(error);
            }
            log::debug!(
                "{:?} cloned session {handle:#x} as {cloned_handle:#x}",
                String::from_utf8_lossy(service.service().name())
            );
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        if control_command.is_clone()
            && let HorizonIpcObject::Vi(vi) = &target
        {
            let cloned_handle = process
                .handles_mut()
                .insert(HorizonIpcObject::Vi(vi.clone()))
                .map_err(|_| IpcWireError::HostResourceExhausted("cloning a VI session handle"))?;
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            )?;
            write_response(process, address, size, &response)?;
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        if control_command.is_clone()
            && let HorizonIpcObject::NvDrv(nvdrv) = &target
        {
            // CMIF cloning creates another service connection into the same
            // nvdrv client. The connections share initialization, descriptors,
            // allocations, and close effects, but retain distinct identities
            // for descriptor ownership.
            let cloned_session =
                nvdrv
                    .clone_connection()
                    .ok_or(IpcWireError::HostResourceExhausted(
                        "cloning an nvdrv connection",
                    ))?;
            let cloned_handle = process
                .handles_mut()
                .insert(HorizonIpcObject::NvDrv(cloned_session))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a cloned nvdrv session handle")
                })?;
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            )?;
            write_response(process, address, size, &response)?;
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        if control_command.is_clone()
            && let HorizonIpcObject::ParentalControl(factory) = &target
        {
            let cloned_handle = process
                .handles_mut()
                .insert(HorizonIpcObject::ParentalControl(factory.clone()))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("cloning a pctl session handle")
                })?;
            let response = match encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            ) {
                Ok(response) => response,
                Err(error) => {
                    let _ = process.handles_mut().close(cloned_handle);
                    return Err(error);
                }
            };
            if let Err(error) = write_response(process, address, size, &response) {
                let _ = process.handles_mut().close(cloned_handle);
                return Err(error);
            }
            return Ok(trace_completion(SyncRequestResult::Success));
        }
        let response = match control_command {
            // QueryPointerBufferSize. Zero makes libnx use map-alias buffers,
            // which the future descriptor bridge can validate explicitly.
            CmifControlCommand::QueryPointerBufferSize => encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &0_u16.to_le_bytes(),
                None,
            ),
            _ => return unsupported_service_command("CMIF control", request.command_id),
        }?;
        write_response(process, address, size, &response)?;
        return Ok(trace_completion(SyncRequestResult::Success));
    }
    let applet_exit_requested = match &target {
        HorizonIpcObject::Applet(session) => {
            services::applet_requests_self_exit(session, &request, &hipc)
        }
        _ => false,
    };
    let (response, created_handle) = match target {
        HorizonIpcObject::ServiceManager(manager) => services::dispatch_service_manager(
            process,
            &manager,
            request,
            hipc.pid.is_some(),
            initial_operation_mode,
            time_environment,
            host_systems,
        )?,
        HorizonIpcObject::SemanticService(service) => dispatch_semantic_service(
            process,
            &service,
            request,
            &hipc,
            host_systems.diagnostics.file_system_access_log_mode(),
        )?,
        HorizonIpcObject::SystemSettings(_) => {
            services::dispatch_system_settings(process, request, &hipc.receive_statics)?
        }
        HorizonIpcObject::UserSettings(settings) => {
            services::dispatch_user_settings(process, &settings, request, &hipc)?
        }
        HorizonIpcObject::PerformanceManager(manager) => {
            services::dispatch_performance_manager(process, &manager, request)?
        }
        HorizonIpcObject::Performance(session) => {
            services::dispatch_performance_session(&session, request)?
        }
        HorizonIpcObject::Applet(applet) => {
            services::dispatch_applet(process, &applet, request, &hipc, host_systems.video)?
        }
        HorizonIpcObject::Account(account) => {
            services::dispatch_account(process, &account, request, &hipc)?
        }
        HorizonIpcObject::Hid(hid) => {
            services::dispatch_hid(process, &hid, host_systems.hid, request, &hipc)?
        }
        HorizonIpcObject::HidAppletResource(resource) => {
            services::dispatch_hid_applet_resource(process, &resource, request)?
        }
        HorizonIpcObject::Time(time) => services::dispatch_time(process, &time, request)?,
        HorizonIpcObject::SystemClock(clock) => services::dispatch_system_clock(&clock, request)?,
        HorizonIpcObject::SteadyClock(clock) => services::dispatch_steady_clock(&clock, request)?,
        HorizonIpcObject::TimeZone(timezone) => services::dispatch_timezone(&timezone, request)?,
        HorizonIpcObject::Vi(vi) => services::dispatch_vi(process, &vi, request, &hipc)?,
        HorizonIpcObject::NvDrv(nvdrv) => match services::dispatch_nvdrv(
            process,
            &nvdrv,
            request,
            &hipc,
            host_systems.caller_thread_id,
        ) {
            Ok(response) => response,
            Err(IpcWireError::PendingNvDrv(wait)) => {
                return Ok(trace_completion(SyncRequestResult::PendingNvDrv(wait)));
            }
            Err(error) => return Err(error),
        },
        HorizonIpcObject::LogManager(_) => services::dispatch_log_manager(process, request, &hipc)?,
        HorizonIpcObject::Logger(logger) => services::dispatch_logger(
            process,
            &logger,
            request,
            &hipc,
            host_systems.diagnostics.guest_logs_level,
        )?,
        HorizonIpcObject::ParentalControl(factory) => {
            services::dispatch_parental_control(process, &factory, request, &hipc)?
        }
        HorizonIpcObject::ParentalControlService(service) => {
            services::dispatch_parental_control_service(None, &service, request, &hipc)?
        }
        HorizonIpcObject::SemanticObject(object) => {
            dispatch_plain_semantic_object(process, &object, request, &hipc)?
        }
    };
    if let Err(error) = write_response(process, address, size, &response) {
        if let Some(handle) = created_handle {
            let _ = process.handles_mut().close(handle);
        }
        return Err(error);
    }
    let outcome = if applet_exit_requested {
        SyncRequestResult::AppletExitRequested
    } else {
        SyncRequestResult::Success
    };
    Ok(trace_completion(outcome))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SyncRequestResult {
    Success,
    InvalidHandle,
    AppletExitRequested,
    PendingNvDrv(crate::nvdrv::PendingNvHostCtrlWait),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_diagnostics_derive_service_names_from_typed_targets() {
        let request = CmifRequest {
            command_type: 4,
            command_id: 3,
            token: 0,
            context: None,
            data: &[],
            domain: None,
        };

        assert_eq!(
            HorizonIpcObject::NvDrv(NvDrvSession::default()).diagnostic_name(&request),
            "nvdrv"
        );
        assert_eq!(
            HorizonIpcObject::SemanticService(IpcSession::new(IpcService::FileSystem))
                .diagnostic_name(&request),
            "fsp-srv"
        );
        assert_eq!(
            HorizonIpcObject::SystemSettings(SystemSettingsSession::new())
                .diagnostic_name(&request),
            "set:sys"
        );
    }

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
}
