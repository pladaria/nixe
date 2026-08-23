//! Synchronous Horizon service-call adapter built on the checked wire codec.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

use chrono::{Datelike, Offset, Timelike};
use chrono_tz::OffsetComponents;
use nixe_cpu::memory::{
    DataAccessFault, DataAccessFaultReason, DataAccessKind, MemoryAccess, MemoryAccessSize,
    MemoryPermissions, MemoryRegionKind, MemoryValue,
};
use nixe_memory::GuestVirtualAddress;
use nixe_runtime::{EventObject, ExceptionProcessContext, HandleObject, TransferMemoryObject};

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
    AccountSession, AppletSession, DirectoryEntryKind, HidAppletResource, HidSession, HidSystem,
    HorizonIpcResult, HostDirectoryFileSystem, HostFile, IpcDispatcher, IpcRequest, IpcResponse,
    IpcResultCode, IpcService, IpcSession, MAX_IPC_LIST_ENTRIES, MAX_IPC_PATH_BYTES,
    MAX_IPC_READ_BYTES, NvDrvSession, OperationMode, PerformanceManagerSession, PerformanceSession,
    ReadOnlyDirectory, ReadOnlyFile, ReadOnlyFileSystem, ServiceManagerSession,
    SettingsEnvironment, SteadyClockSession, SystemClockKind, SystemClockSession, SystemLanguage,
    SystemSettingsSession, TimeEnvironment, TimeServiceSession, TimeZoneServiceSession,
    UserSettingsSession, ViObjectKind, ViServiceKind, ViSession, VideoSystem,
};

pub(crate) const NAMED_PORT_NAME_SIZE: usize = 12;
const CMIF_COMMAND_CLOSE: u16 = 2;
const CMIF_COMMAND_CONTROL: u16 = 5;
const CMIF_COMMAND_CONTROL_WITH_CONTEXT: u16 = 7;
const FIRMWARE_VERSION_SIZE: usize = 0x100;
const PERFORMANCE_MODE_NORMAL: u32 = 0;
const FS_MAX_PATH: usize = 0x301;
const FS_DIRECTORY_ENTRY_SIZE: usize = 0x310;
const FS_DIRECTORY_ENTRY_FILE: u8 = 1;

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
        *byte = read_u8(process, add(name_address, index)?)?;
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
                match process.handles_mut().insert(ServiceManagerSession::new()) {
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
    pub caller_thread_id: u64,
}

#[derive(Clone)]
enum IpcTarget {
    ServiceManager(ServiceManagerSession),
    SemanticService(IpcSession),
    SystemSettings(SystemSettingsSession),
    UserSettings(UserSettingsSession),
    PerformanceManager(PerformanceManagerSession),
    Performance(PerformanceSession),
    Applet(AppletSession),
    Account(AccountSession),
    Hid(HidSession),
    HidAppletResource(HidAppletResource),
    Time(TimeServiceSession),
    SystemClock(SystemClockSession),
    SteadyClock(SteadyClockSession),
    TimeZone(TimeZoneServiceSession),
    Vi(ViSession),
    NvDrv(NvDrvSession),
    SemanticObject(HandleObject),
}

#[derive(Clone, Copy)]
enum ServiceKind {
    UserSettings,
    SystemSettings,
    Performance,
    Applet,
    Hid,
    Time,
    Account,
    Vi(ViServiceKind),
    NvDrv,
    Semantic(IpcService),
}

enum SemanticTarget {
    Root,
    Object(HandleObject),
}

impl ServiceKind {
    fn from_name(name: &[u8]) -> Option<Self> {
        match name {
            b"set" => Some(Self::UserSettings),
            b"set:sys" => Some(Self::SystemSettings),
            b"apm" => Some(Self::Performance),
            b"appletOE" => Some(Self::Applet),
            b"hid" => Some(Self::Hid),
            b"time:u" => Some(Self::Time),
            b"acc:u0" => Some(Self::Account),
            b"nvdrv" | b"nvdrv:a" | b"nvdrv:s" => Some(Self::NvDrv),
            _ => ViServiceKind::from_name(name)
                .map(Self::Vi)
                .or_else(|| IpcService::from_name(name).map(Self::Semantic)),
        }
    }
}

impl IpcTarget {
    fn from_object(object: &HandleObject) -> Option<Self> {
        if let Some(value) = object.downcast_ref::<ServiceManagerSession>() {
            Some(Self::ServiceManager(value.clone()))
        } else if let Some(value) = object.downcast_ref::<IpcSession>() {
            Some(Self::SemanticService(value.clone()))
        } else if let Some(value) = object.downcast_ref::<SystemSettingsSession>() {
            Some(Self::SystemSettings(*value))
        } else if let Some(value) = object.downcast_ref::<UserSettingsSession>() {
            Some(Self::UserSettings(value.clone()))
        } else if let Some(value) = object.downcast_ref::<PerformanceManagerSession>() {
            Some(Self::PerformanceManager(value.clone()))
        } else if let Some(value) = object.downcast_ref::<PerformanceSession>() {
            Some(Self::Performance(value.clone()))
        } else if let Some(value) = object.downcast_ref::<AppletSession>() {
            Some(Self::Applet(value.clone()))
        } else if let Some(value) = object.downcast_ref::<AccountSession>() {
            Some(Self::Account(value.clone()))
        } else if let Some(value) = object.downcast_ref::<HidSession>() {
            Some(Self::Hid(value.clone()))
        } else if let Some(value) = object.downcast_ref::<HidAppletResource>() {
            Some(Self::HidAppletResource(value.clone()))
        } else if let Some(value) = object.downcast_ref::<TimeServiceSession>() {
            Some(Self::Time(value.clone()))
        } else if let Some(value) = object.downcast_ref::<SystemClockSession>() {
            Some(Self::SystemClock(value.clone()))
        } else if let Some(value) = object.downcast_ref::<SteadyClockSession>() {
            Some(Self::SteadyClock(value.clone()))
        } else if let Some(value) = object.downcast_ref::<TimeZoneServiceSession>() {
            Some(Self::TimeZone(value.clone()))
        } else if let Some(value) = object.downcast_ref::<ViSession>() {
            Some(Self::Vi(value.clone()))
        } else if let Some(value) = object.downcast_ref::<NvDrvSession>() {
            Some(Self::NvDrv(value.clone()))
        } else if object.is::<ReadOnlyFileSystem>()
            || object.is::<HostDirectoryFileSystem>()
            || object.is::<ReadOnlyFile>()
            || object.is::<HostFile>()
            || object.is::<ReadOnlyDirectory>()
        {
            Some(Self::SemanticObject(object.clone()))
        } else {
            None
        }
    }

    fn is_domain(&self) -> bool {
        match self {
            Self::Applet(session) => session.is_domain(),
            Self::SemanticService(session) => session.is_domain(),
            _ => false,
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
        .get(handle)
        .and_then(IpcTarget::from_object)
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
    log::debug!(
        "SendSyncRequest handle={handle:#x} type={} command={} send_pid={} descriptors={}/{}/{}/{} handles={}/{}",
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

    if request.command_type == CMIF_COMMAND_CLOSE {
        // libnx sends a CMIF close before releasing an owned session handle.
        // The semantic endpoint must stop accepting work at this point even
        // though libnx subsequently issues CloseHandle as a best-effort local
        // cleanup:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L195-L209
        let _ = process.handles_mut().close(handle);
        return Ok(SyncRequestResult::Success);
    }
    if matches!(
        request.command_type,
        CMIF_COMMAND_CONTROL | CMIF_COMMAND_CONTROL_WITH_CONTEXT
    ) {
        if request.command_id == 0
            && let IpcTarget::Applet(applet) = &target
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
            return Ok(SyncRequestResult::Success);
        }
        if request.command_id == 0
            && let IpcTarget::SemanticService(service) = &target
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
            return Ok(SyncRequestResult::Success);
        }
        if matches!(request.command_id, 2 | 4)
            && let IpcTarget::SemanticService(service) = &target
        {
            // CloneCurrentObject (2) returns a moved session handle. The Ex
            // form (4) additionally carries a session-manager tag which the
            // public libnx ABI documents as unused by official servers:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L308-L337
            // A cloned `IpcSession` has a distinct process-handle object while
            // retaining the shared domain table of the source connection.
            let cloned_handle = process.handles_mut().insert(service.clone()).map_err(|_| {
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
            return Ok(SyncRequestResult::Success);
        }
        if matches!(request.command_id, 2 | 4)
            && let IpcTarget::Vi(vi) = &target
        {
            let cloned_handle = process
                .handles_mut()
                .insert(vi.clone())
                .map_err(|_| IpcWireError::HostResourceExhausted("cloning a VI session handle"))?;
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            )?;
            write_response(process, address, size, &response)?;
            return Ok(SyncRequestResult::Success);
        }
        if matches!(request.command_id, 2 | 4)
            && let IpcTarget::NvDrv(nvdrv) = &target
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
            let cloned_handle = process.handles_mut().insert(cloned_session).map_err(|_| {
                IpcWireError::HostResourceExhausted("installing a cloned nvdrv session handle")
            })?;
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            )?;
            write_response(process, address, size, &response)?;
            return Ok(SyncRequestResult::Success);
        }
        let response = match request.command_id {
            // QueryPointerBufferSize. Zero makes libnx use map-alias buffers,
            // which the future descriptor bridge can validate explicitly.
            3 => encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &0_u16.to_le_bytes(),
                None,
            ),
            command_id => return unsupported_service_command("CMIF control", command_id),
        }?;
        write_response(process, address, size, &response)?;
        return Ok(SyncRequestResult::Success);
    }
    let applet_exit_requested = match &target {
        IpcTarget::Applet(session) => applet_requests_self_exit(session, &request, &hipc),
        _ => false,
    };
    let (response, created_handle) = match target {
        IpcTarget::ServiceManager(manager) => dispatch_service_manager(
            process,
            &manager,
            request,
            hipc.pid.is_some(),
            initial_operation_mode,
            time_environment,
            host_systems,
        )?,
        IpcTarget::SemanticService(service) => {
            dispatch_semantic_service(process, &service, request, &hipc)?
        }
        IpcTarget::SystemSettings(_) => {
            dispatch_system_settings(process, request, &hipc.receive_statics)?
        }
        IpcTarget::UserSettings(settings) => {
            dispatch_user_settings(process, &settings, request, &hipc)?
        }
        IpcTarget::PerformanceManager(manager) => {
            dispatch_performance_manager(process, &manager, request)?
        }
        IpcTarget::Performance(session) => dispatch_performance_session(&session, request)?,
        IpcTarget::Applet(applet) => {
            dispatch_applet(process, &applet, request, &hipc, host_systems.video)?
        }
        IpcTarget::Account(account) => dispatch_account(process, &account, request, &hipc)?,
        IpcTarget::Hid(hid) => dispatch_hid(process, &hid, host_systems.hid, request, &hipc)?,
        IpcTarget::HidAppletResource(resource) => {
            dispatch_hid_applet_resource(process, &resource, request)?
        }
        IpcTarget::Time(time) => dispatch_time(process, &time, request)?,
        IpcTarget::SystemClock(clock) => dispatch_system_clock(&clock, request)?,
        IpcTarget::SteadyClock(clock) => dispatch_steady_clock(&clock, request)?,
        IpcTarget::TimeZone(timezone) => dispatch_timezone(&timezone, request)?,
        IpcTarget::Vi(vi) => dispatch_vi(process, &vi, request, &hipc)?,
        IpcTarget::NvDrv(nvdrv) => match dispatch_nvdrv(
            process,
            &nvdrv,
            request,
            &hipc,
            host_systems.caller_thread_id,
        ) {
            Ok(response) => response,
            Err(IpcWireError::PendingNvDrv(wait)) => {
                return Ok(SyncRequestResult::PendingNvDrv(wait));
            }
            Err(error) => return Err(error),
        },
        IpcTarget::SemanticObject(object) => {
            dispatch_plain_semantic_object(process, &object, request, &hipc)?
        }
    };
    if let Err(error) = write_response(process, address, size, &response) {
        if let Some(handle) = created_handle {
            let _ = process.handles_mut().close(handle);
        }
        return Err(error);
    }
    Ok(if applet_exit_requested {
        SyncRequestResult::AppletExitRequested
    } else {
        SyncRequestResult::Success
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SyncRequestResult {
    Success,
    InvalidHandle,
    AppletExitRequested,
    PendingNvDrv(crate::nvdrv::PendingNvHostCtrlWait),
}

fn applet_requests_self_exit(
    session: &AppletSession,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> bool {
    request.command_id == 0
        && request.data.is_empty()
        && !has_ipc_descriptors(hipc)
        && matches!(
            &request.domain,
            Some(DomainRequest::SendMessage { object_id, .. })
                if session.object(*object_id) == Some(AppletObject::SelfController)
        )
}

fn dispatch_service_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &ServiceManagerSession,
    request: CmifRequest<'_>,
    sent_pid: bool,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match request.command_id {
        0 => {
            if !sent_pid || request.data.len() < 8 {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_CLIENT,
                        &[],
                        None,
                    )?,
                    None,
                ));
            }
            manager.register_client();
            log::debug!(
                "sm:RegisterClient associated process {}",
                process.process_id()
            );
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        1 => {
            if !manager.is_registered() {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_CLIENT,
                        &[],
                        None,
                    )?,
                    None,
                ));
            }
            let Some(encoded_name) = request.data.get(..8) else {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_SERVICE_NAME,
                        &[],
                        None,
                    )?,
                    None,
                ));
            };
            let Some(name) = decode_service_name(encoded_name) else {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_SERVICE_NAME,
                        &[],
                        None,
                    )?,
                    None,
                ));
            };
            log::debug!(
                "sm:GetService requested {:?}",
                String::from_utf8_lossy(name)
            );
            if !process.mounts().allows_service(name) {
                return service_response(request.token, HorizonIpcResult::SM_NOT_ALLOWED, None);
            }
            let Some(service) = ServiceKind::from_name(name) else {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::Connect { name: name.into() },
                ));
            };
            connect_service(
                process,
                request.token,
                service,
                initial_operation_mode,
                time_environment,
                host_systems,
            )
        }
        command_id => unsupported_service_command("sm:", command_id),
    }
}

fn dispatch_account(
    process: &ExceptionProcessContext<'_>,
    session: &AccountSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match request.command_id {
        // InitializeApplicationInfo is command 100 before Horizon 6.0.0 and
        // command 140 afterward. libnx sends the caller PID descriptor and a
        // zero u64 placeholder:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/acc.c#L61-L67
        100 | 140 => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0) != Some(0)
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.initialize_application_info(process.process_id());
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        command_id => Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::Command {
                service: "acc:u0",
                command_id,
            },
        )),
    }
}

fn connect_service(
    process: &mut ExceptionProcessContext<'_>,
    token: u32,
    service: ServiceKind,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let host_resource_failure = matches!(
        service,
        ServiceKind::Account | ServiceKind::Vi(_) | ServiceKind::NvDrv
    );
    let handle = match service {
        ServiceKind::UserSettings => process
            .handles_mut()
            .insert(UserSettingsSession::new(host_systems.settings.clone())),
        ServiceKind::SystemSettings => process.handles_mut().insert(SystemSettingsSession::new()),
        ServiceKind::Performance => process
            .handles_mut()
            .insert(PerformanceManagerSession::new()),
        ServiceKind::Applet => process
            .handles_mut()
            .insert(AppletSession::new(initial_operation_mode)),
        ServiceKind::Hid => process
            .handles_mut()
            .insert(HidSession::new(host_systems.hid.shared_memory())),
        ServiceKind::Time => time_environment
            .create_service()
            .and_then(|session| process.handles_mut().insert(session)),
        // libnx opens acc:u0 for application account sessions. Retain the
        // real session identity while unsupported commands remain fail-fast.
        ServiceKind::Account => process.handles_mut().insert(AccountSession::new()),
        ServiceKind::Vi(kind) => process.handles_mut().insert(ViSession::new(
            ViObjectKind::Root(kind),
            host_systems.video.clone(),
        )),
        ServiceKind::NvDrv => process.handles_mut().insert(host_systems.video.nvdrv()),
        ServiceKind::Semantic(service) => process.handles_mut().insert(IpcSession::new(service)),
    };
    match handle {
        Ok(handle) => {
            log::debug!("sm:GetService returned session handle {handle:#x}");
            service_response(token, HorizonIpcResult::SUCCESS, Some(handle))
        }
        Err(_) if host_resource_failure => Err(IpcWireError::HostResourceExhausted(
            "installing a service handle",
        )),
        Err(_) => service_response(token, HorizonIpcResult::SM_OUT_OF_SESSIONS, None),
    }
}

fn service_response(
    token: u32,
    result: HorizonIpcResult,
    handle: Option<u32>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((encode_response(token, result, &[], handle)?, handle))
}

fn dispatch_semantic_service(
    process: &mut ExceptionProcessContext<'_>,
    session: &IpcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            let result = if session.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            return Ok((
                encode_domain_response(request.token, result, &[], &[], &[])?,
                None,
            ));
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return unsupported_service_command(
                    semantic_service_name(session.service()),
                    request.command_id,
                );
            }
            if *object_id == 1 {
                SemanticTarget::Root
            } else {
                let Some(object) = session.object(*object_id) else {
                    return semantic_error(
                        request.token,
                        session.service(),
                        Some(session),
                        HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                    );
                };
                SemanticTarget::Object(object)
            }
        }
        None if session.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain service request omitted its domain header",
            ));
        }
        None => SemanticTarget::Root,
    };
    dispatch_semantic_command(
        process,
        session.service(),
        Some(session),
        target,
        request,
        hipc,
    )
}

fn dispatch_plain_semantic_object(
    process: &mut ExceptionProcessContext<'_>,
    object: &HandleObject,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    dispatch_semantic_command(
        process,
        IpcService::FileSystem,
        None,
        SemanticTarget::Object(object.clone()),
        request,
        hipc,
    )
}

fn dispatch_semantic_command(
    process: &mut ExceptionProcessContext<'_>,
    service: IpcService,
    session: Option<&IpcSession>,
    target: SemanticTarget,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let (decoded, name) = match &target {
        SemanticTarget::Root => (
            decode_root_request(service, &request, hipc)?,
            semantic_service_name(service),
        ),
        SemanticTarget::Object(object) => (
            decode_object_request(process, object, &request, hipc)?,
            semantic_object_name(object),
        ),
    };
    let Some(decoded) = decoded else {
        return unsupported_service_command(name, request.command_id);
    };
    let result = {
        let (mounts, handles) = process.mounts_and_handles_mut();
        match &target {
            SemanticTarget::Root => IpcDispatcher::dispatch_session(
                mounts,
                handles,
                session.expect("a semantic root belongs to a session"),
                decoded,
            ),
            SemanticTarget::Object(object) => {
                IpcDispatcher::dispatch_object(mounts, handles, object, decoded)
            }
        }
    };
    match result {
        Ok(response) => {
            encode_semantic_response(process, service, session, request, hipc, response)
        }
        Err(IpcResultCode::INVALID_COMMAND) => {
            unsupported_service_command(name, request.command_id)
        }
        Err(IpcResultCode::INTERNAL_STATE) => Err(IpcWireError::Internal(
            "semantic IPC entered an invalid internal state",
        )),
        Err(error) => semantic_error(
            request.token,
            service,
            session,
            HorizonIpcResult::from_semantic(service, error),
        ),
    }
}

fn semantic_object_name(object: &HandleObject) -> &'static str {
    if object.is::<ReadOnlyFileSystem>() {
        "IFileSystem(read-only)"
    } else if object.is::<HostDirectoryFileSystem>() {
        "IFileSystem(sd-card)"
    } else if object.is::<ReadOnlyFile>() {
        "IFile(read-only)"
    } else if object.is::<HostFile>() {
        "IFile(sd-card)"
    } else if object.is::<ReadOnlyDirectory>() {
        "IDirectory"
    } else {
        "semantic IPC object"
    }
}

fn decode_root_request(
    service: IpcService,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    match (service, request.command_id) {
        // IFileSystemProxy::SetCurrentProcess. libnx sends the current PID and
        // a zero placeholder before opening the current program's data FS:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L75-L82
        (IpcService::FileSystem, 1) => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return Ok(None);
            }
            Ok(Some(IpcRequest::SetCurrentProcess))
        }
        // IFileSystemProxy::OpenDataFileSystemByCurrentProcess.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L123-L125
        (IpcService::FileSystem, 2) => Ok(Some(IpcRequest::OpenPrimaryFileSystem)),
        // IFileSystemProxy::OpenSdCardFileSystem returns an IFileSystem object:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L257-L259
        (IpcService::FileSystem, 18) => Ok(Some(IpcRequest::OpenSdCardFileSystem)),
        // aoc:u command IDs and version ranges:
        // https://switchbrew.org/w/index.php?title=NS_services&oldid=14328#aoc:u
        (IpcService::AddOnContent, 0) => Ok(Some(IpcRequest::GetIndexedAddOnContentCount)),
        (IpcService::AddOnContent, 2) if hipc.pid.is_some() => {
            Ok(Some(IpcRequest::GetIndexedAddOnContentCount))
        }
        (IpcService::AddOnContent, 1 | 3) => {
            if request.command_id == 3 && hipc.pid.is_none() {
                return Ok(None);
            }
            let offset = request_u32(request.data, 0)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(IpcWireError::Malformed(
                    "aoc:u list request omits its start index",
                ))?;
            let requested = request_u32(request.data, 4)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(IpcWireError::Malformed(
                    "aoc:u list request omits its entry count",
                ))?;
            let descriptor = one_receive_buffer(hipc)?;
            let capacity = usize::try_from(descriptor.size / 4)
                .map_err(|_| IpcWireError::Malformed("aoc:u output buffer is too large"))?;
            Ok(Some(IpcRequest::ListIndexedAddOnContent {
                offset,
                max_entries: requested.min(capacity).min(MAX_IPC_LIST_ENTRIES),
            }))
        }
        (IpcService::AddOnContent, 6 | 7) => {
            if request.command_id == 7 && hipc.pid.is_none() {
                return Ok(None);
            }
            let horizon_index = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                "aoc:u prepare request omits its content index",
            ))?;
            Ok(Some(IpcRequest::PrepareAddOnContent { horizon_index }))
        }
        (IpcService::AddOnContent, 8) => Ok(Some(IpcRequest::GetAddOnContentListChangedEvent)),
        _ => Ok(None),
    }
}

fn decode_object_request(
    process: &ExceptionProcessContext<'_>,
    object: &HandleObject,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    if object.is::<ReadOnlyFileSystem>() || object.is::<HostDirectoryFileSystem>() {
        let is_host = object.is::<HostDirectoryFileSystem>();
        return match request.command_id {
            // IFileSystem::CreateFile/CreateDirectory use the same bounded
            // input-pointer path as open operations:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L816-L840
            0 if is_host => {
                let option = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                    "create-file request omits its option",
                ))?;
                let size = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                    "create-file request omits its size",
                ))?;
                Ok(Some(IpcRequest::CreateFile {
                    path: read_path(process, hipc)?,
                    size,
                    option,
                }))
            }
            2 if is_host => Ok(Some(IpcRequest::CreateDirectory {
                path: read_path(process, hipc)?,
            })),
            // IFileSystem OpenFile/OpenDirectory use one input pointer path
            // and a u32 mode:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L878-L893
            8 => Ok(Some(IpcRequest::OpenFile {
                path: read_path(process, hipc)?,
                mode: request_u32(request.data, 0)
                    .ok_or(IpcWireError::Malformed("open-file request omits its mode"))?,
            })),
            9 => Ok(Some(IpcRequest::OpenDirectory {
                path: read_path(process, hipc)?,
                mode: request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                    "open-directory request omits its mode",
                ))?,
            })),
            _ => Ok(None),
        };
    }
    if object.is::<ReadOnlyFile>() || object.is::<HostFile>() {
        let is_host = object.is::<HostFile>();
        return match request.command_id {
            // IFile::Read input layout and map-alias output buffer:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L980-L994
            0 => {
                let offset = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                    "file read request omits its offset",
                ))?;
                let requested = request_u64(request.data, 16)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(IpcWireError::Malformed(
                        "file read request size is out of range",
                    ))?;
                let descriptor = one_receive_buffer(hipc)?;
                let capacity = usize::try_from(descriptor.size)
                    .map_err(|_| IpcWireError::Malformed("file output buffer is too large"))?;
                Ok(Some(IpcRequest::ReadFile {
                    offset,
                    size: requested.min(capacity).min(MAX_IPC_READ_BYTES),
                }))
            }
            // IFile::Write carries option/padding/offset/size and one
            // map-alias input buffer:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L994-L1017
            1 if is_host => {
                let option = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                    "file write request omits its option",
                ))?;
                let offset = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                    "file write request omits its offset",
                ))?;
                let requested = request_u64(request.data, 16)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(IpcWireError::Malformed(
                        "file write request size is out of range",
                    ))?;
                if requested > MAX_IPC_READ_BYTES {
                    return Err(IpcWireError::UnsupportedService(
                        UnsupportedServiceOperation::CommandVariant {
                            service: "IFile",
                            command_id: request.command_id,
                            detail: "requested write exceeds Nixe's implemented IPC bound",
                        },
                    ));
                }
                let descriptor = one_send_buffer(hipc)?;
                let capacity = usize::try_from(descriptor.size)
                    .map_err(|_| IpcWireError::Malformed("file input buffer is too large"))?;
                if requested > capacity {
                    return Err(IpcWireError::Malformed(
                        "file write size exceeds its input buffer",
                    ));
                }
                let mut data = vec![0; requested];
                read_bytes(
                    process,
                    GuestVirtualAddress::new(descriptor.address),
                    &mut data,
                )?;
                Ok(Some(IpcRequest::WriteFile {
                    offset,
                    data,
                    flush: option & 1 != 0,
                }))
            }
            2 if is_host => Ok(Some(IpcRequest::FlushFile)),
            3 if is_host => Ok(Some(IpcRequest::SetFileSize {
                size: request_u64(request.data, 0).ok_or(IpcWireError::Malformed(
                    "set-file-size request omits its size",
                ))?,
            })),
            4 => Ok(Some(IpcRequest::GetFileSize)),
            _ => Ok(None),
        };
    }
    if object.is::<ReadOnlyDirectory>() {
        return match request.command_id {
            // IDirectory::Read returns fixed 0x310-byte FsDirectoryEntry
            // records through one map-alias output buffer:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L1043-L1051
            0 => {
                let descriptor = one_receive_buffer(hipc)?;
                let capacity = usize::try_from(descriptor.size)
                    .ok()
                    .map(|size| size / FS_DIRECTORY_ENTRY_SIZE)
                    .ok_or(IpcWireError::Malformed(
                        "directory output buffer is too large",
                    ))?;
                Ok(Some(IpcRequest::ReadDirectory {
                    max_entries: capacity.min(MAX_IPC_LIST_ENTRIES),
                }))
            }
            1 => Ok(Some(IpcRequest::GetDirectoryEntryCount)),
            _ => Ok(None),
        };
    }
    Ok(None)
}

fn encode_semantic_response(
    process: &mut ExceptionProcessContext<'_>,
    service: IpcService,
    domain_session: Option<&IpcSession>,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    response: IpcResponse,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let is_domain = domain_session.is_some_and(IpcSession::is_domain);
    match response {
        IpcResponse::None => semantic_success(request.token, is_domain, &[], &[], &[], None),
        IpcResponse::Size(size) => semantic_success(
            request.token,
            is_domain,
            &size.to_le_bytes(),
            &[],
            &[],
            None,
        ),
        IpcResponse::Handle(handle) => {
            if is_domain {
                let object = process
                    .handles_mut()
                    .close(handle)
                    .map_err(|_| IpcWireError::Internal("semantic child handle disappeared"))?;
                let Some(object_id) =
                    domain_session.and_then(|session| session.insert_object(object))
                else {
                    return semantic_error(
                        request.token,
                        service,
                        domain_session,
                        HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                    );
                };
                semantic_success(request.token, true, &[], &[], &[object_id], None)
            } else {
                semantic_success(request.token, false, &[], &[], &[], Some(handle))
            }
        }
        IpcResponse::Data(data) => {
            let descriptor = one_receive_buffer(hipc)?;
            write_descriptor_bytes(process, descriptor, &data)?;
            let count = u64::try_from(data.len())
                .map_err(|_| IpcWireError::Malformed("file read count overflows"))?;
            semantic_success(
                request.token,
                is_domain,
                &count.to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        IpcResponse::DirectoryEntries(entries) => {
            let descriptor = one_receive_buffer(hipc)?;
            let mut encoded = Vec::new();
            encoded
                .try_reserve_exact(entries.len().saturating_mul(FS_DIRECTORY_ENTRY_SIZE))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("encoding filesystem directory entries")
                })?;
            encoded.resize(entries.len() * FS_DIRECTORY_ENTRY_SIZE, 0);
            for (index, entry) in entries.iter().enumerate() {
                let start = index * FS_DIRECTORY_ENTRY_SIZE;
                let name = entry.name().as_bytes();
                let copy_len = name.len().min(FS_MAX_PATH - 1);
                encoded[start..start + copy_len].copy_from_slice(&name[..copy_len]);
                encoded[start + 0x304] = match entry.kind() {
                    DirectoryEntryKind::Directory => 0,
                    DirectoryEntryKind::File => FS_DIRECTORY_ENTRY_FILE,
                };
                encoded[start + 0x308..start + 0x310].copy_from_slice(&entry.size().to_le_bytes());
            }
            write_descriptor_bytes(process, descriptor, &encoded)?;
            let count = u64::try_from(entries.len())
                .map_err(|_| IpcWireError::Malformed("directory entry count overflows"))?;
            semantic_success(
                request.token,
                is_domain,
                &count.to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        IpcResponse::AddOnContentEntries(entries) => {
            let descriptor = one_receive_buffer(hipc)?;
            let mut encoded = Vec::new();
            encoded
                .try_reserve_exact(entries.len().saturating_mul(4))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("encoding add-on-content entries")
                })?;
            for entry in entries {
                let Some(index) = entry.horizon_index else {
                    continue;
                };
                encoded.extend_from_slice(&index.to_le_bytes());
            }
            write_descriptor_bytes(process, descriptor, &encoded)?;
            let count = u32::try_from(encoded.len() / 4)
                .map_err(|_| IpcWireError::Malformed("add-on count overflows"))?;
            semantic_success(
                request.token,
                is_domain,
                &count.to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        IpcResponse::Event(handle) => {
            semantic_success(request.token, is_domain, &[], &[handle], &[], None)
        }
    }
}

fn semantic_success(
    token: u32,
    is_domain: bool,
    data: &[u8],
    copy_handles: &[u32],
    domain_objects: &[u32],
    move_handle: Option<u32>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let move_handles = move_handle.as_slice();
    Ok((
        CmifResponse {
            token,
            result: HorizonIpcResult::SUCCESS.raw(),
            data,
            pid: None,
            copy_handles,
            move_handles,
            send_statics: &[],
            is_domain,
            domain_objects,
        }
        .encode()?,
        move_handle.or_else(|| copy_handles.first().copied()),
    ))
}

fn semantic_error(
    token: u32,
    _service: IpcService,
    domain_session: Option<&IpcSession>,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if domain_session.is_some_and(IpcSession::is_domain) {
        Ok((encode_domain_response(token, result, &[], &[], &[])?, None))
    } else {
        cmif_error(token, result)
    }
}

fn read_path(
    process: &ExceptionProcessContext<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<String, IpcWireError> {
    enum InputDescriptor {
        Static(SendStaticDescriptor),
        Buffer(BufferDescriptor),
    }
    let descriptor = match (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice()) {
        ([descriptor], []) => InputDescriptor::Static(*descriptor),
        ([], [descriptor]) => InputDescriptor::Buffer(*descriptor),
        _ => {
            return Err(IpcWireError::Malformed(
                "filesystem path requires exactly one input descriptor",
            ));
        }
    };
    let (address, size) = match descriptor {
        InputDescriptor::Static(descriptor) => (descriptor.address, usize::from(descriptor.size)),
        InputDescriptor::Buffer(descriptor) => (
            {
                if descriptor.mode == BufferMode::Invalid {
                    return Err(IpcWireError::Malformed(
                        "filesystem path buffer has an invalid mapping mode",
                    ));
                }
                descriptor.address
            },
            usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("filesystem path buffer is too large"))?,
        ),
    };
    if size == 0 || size > FS_MAX_PATH || size > MAX_IPC_PATH_BYTES + 1 {
        return Err(IpcWireError::Malformed(
            "filesystem path descriptor has an invalid size",
        ));
    }
    let mut bytes = vec![0; size];
    read_bytes(process, GuestVirtualAddress::new(address), &mut bytes)?;
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(IpcWireError::Malformed(
            "filesystem path is not null terminated",
        ))?;
    String::from_utf8(bytes[..nul].to_vec())
        .map_err(|_| IpcWireError::Malformed("filesystem path is not UTF-8"))
}

fn one_receive_buffer(hipc: &HipcRequest<'_>) -> Result<BufferDescriptor, IpcWireError> {
    match hipc.receive_buffers.as_slice() {
        [descriptor] if descriptor.size > 0 && descriptor.mode != BufferMode::Invalid => {
            Ok(*descriptor)
        }
        _ => Err(IpcWireError::Malformed(
            "service command requires exactly one output buffer",
        )),
    }
}

fn one_send_buffer(hipc: &HipcRequest<'_>) -> Result<BufferDescriptor, IpcWireError> {
    match hipc.send_buffers.as_slice() {
        [descriptor] if descriptor.mode != BufferMode::Invalid => Ok(*descriptor),
        [_] => Err(IpcWireError::Malformed(
            "input buffer has an invalid mapping mode",
        )),
        _ => Err(IpcWireError::Malformed(
            "request requires exactly one input buffer",
        )),
    }
}

fn one_auto_select_input(hipc: &HipcRequest<'_>) -> Result<(u64, usize), IpcWireError> {
    let [pointer] = hipc.send_statics.as_slice() else {
        return Err(IpcWireError::Malformed(
            "auto-select input requires exactly one pointer descriptor",
        ));
    };
    let [map_alias] = hipc.send_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "auto-select input requires exactly one map-alias descriptor",
        ));
    };
    if pointer.index != 0 {
        return Err(IpcWireError::Malformed(
            "auto-select input pointer has an invalid index",
        ));
    }
    if map_alias.mode == BufferMode::Invalid {
        return Err(IpcWireError::Malformed(
            "auto-select input map-alias has an invalid mapping mode",
        ));
    }

    // HIPC auto-select reserves both descriptor slots and places the transfer
    // in exactly one of them, leaving the inactive side as a null placeholder:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L228-L247
    let pointer_present = pointer.address != 0 || pointer.size != 0;
    let map_alias_present = map_alias.address != 0 || map_alias.size != 0;
    match (pointer_present, map_alias_present) {
        (true, false) if pointer.address != 0 => Ok((pointer.address, usize::from(pointer.size))),
        (false, true) if map_alias.address != 0 => Ok((
            map_alias.address,
            usize::try_from(map_alias.size)
                .map_err(|_| IpcWireError::Malformed("input buffer is too large"))?,
        )),
        (false, false) => Ok((0, 0)),
        (true, true) => Err(IpcWireError::Malformed(
            "auto-select input has both descriptor sides active",
        )),
        _ => Err(IpcWireError::Malformed(
            "auto-select input has a null address with nonzero size",
        )),
    }
}

fn one_auto_select_output(hipc: &HipcRequest<'_>) -> Result<(u64, usize), IpcWireError> {
    let ReceiveStatics::Entries(pointers) = &hipc.receive_statics else {
        return Err(IpcWireError::Malformed(
            "auto-select output requires exactly one pointer descriptor",
        ));
    };
    let [
        ReceiveStaticDescriptor {
            address: pointer_address,
            size: pointer_size,
        },
    ] = pointers.as_slice()
    else {
        return Err(IpcWireError::Malformed(
            "auto-select output requires exactly one pointer descriptor",
        ));
    };
    let [map_alias] = hipc.receive_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "auto-select output requires exactly one map-alias descriptor",
        ));
    };
    if map_alias.mode == BufferMode::Invalid {
        return Err(IpcWireError::Malformed(
            "auto-select output map-alias has an invalid mapping mode",
        ));
    }

    let pointer_present = *pointer_address != 0 || *pointer_size != 0;
    let map_alias_present = map_alias.address != 0 || map_alias.size != 0;
    match (pointer_present, map_alias_present) {
        (true, false) if *pointer_address != 0 => {
            Ok((*pointer_address, usize::from(*pointer_size)))
        }
        (false, true) if map_alias.address != 0 => Ok((
            map_alias.address,
            usize::try_from(map_alias.size)
                .map_err(|_| IpcWireError::Malformed("output buffer is too large"))?,
        )),
        (false, false) => Ok((0, 0)),
        (true, true) => Err(IpcWireError::Malformed(
            "auto-select output has both descriptor sides active",
        )),
        _ => Err(IpcWireError::Malformed(
            "auto-select output has a null address with nonzero size",
        )),
    }
}

const NV_IOCTL_WRITE: u32 = 1 << 30;
const NV_IOCTL_READ: u32 = 1 << 31;
const NV_IOCTL_SIZE_MASK: u32 = 0x3fff;

// nvIoctl derives the two buffer presences and their common size directly
// from these Linux-style request fields before issuing CMIF command 1:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L137-L170

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvIoctlBuffers {
    input: Option<BufferDescriptor>,
    additional_input: Option<BufferDescriptor>,
    output: Option<BufferDescriptor>,
}

fn nv_ioctl_buffers(hipc: &HipcRequest<'_>, request: u32) -> Result<NvIoctlBuffers, IpcWireError> {
    nv_ioctl_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &hipc.receive_buffers,
        &hipc.receive_statics,
        request,
    )
}

fn nv_ioctl_buffer_descriptors(
    send_statics: &[SendStaticDescriptor],
    send_buffers: &[BufferDescriptor],
    receive_buffers: &[BufferDescriptor],
    receive_statics: &ReceiveStatics,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [input] = send_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one input auto-select buffer",
        ));
    };
    let [output] = receive_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one output auto-select buffer",
        ));
    };
    let [input_pointer] = send_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one input auto-select pointer",
        ));
    };
    let ReceiveStatics::Entries(output_pointers) = receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires one output auto-select pointer",
        ));
    };
    let [output_pointer] = output_pointers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one output auto-select pointer",
        ));
    };

    // Nixe reports a zero CMIF pointer-buffer size, so libnx selects the
    // map-alias side of each HipcAutoSelect pair and emits null pointer-side
    // placeholders. Keep this transport contract pinned to the implementation
    // used by the target homebrew:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L157-L177
    if input_pointer.address != 0
        || input_pointer.size != 0
        || output_pointer.address != 0
        || output_pointer.size != 0
    {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl auto-select pointer placeholders are not null",
        ));
    }

    let encoded_size = u64::from((request >> 16) & NV_IOCTL_SIZE_MASK);
    let select = |descriptor: BufferDescriptor,
                  present: bool,
                  present_error: &'static str,
                  absent_error: &'static str|
     -> Result<Option<BufferDescriptor>, IpcWireError> {
        if descriptor.mode == BufferMode::Invalid {
            return Err(IpcWireError::Malformed(present_error));
        }
        if present {
            if descriptor.size != encoded_size || (encoded_size != 0 && descriptor.address == 0) {
                return Err(IpcWireError::Malformed(present_error));
            }
            Ok(Some(descriptor))
        } else if descriptor.address == 0
            && descriptor.size == 0
            && descriptor.mode == BufferMode::Normal
        {
            Ok(None)
        } else {
            Err(IpcWireError::Malformed(absent_error))
        }
    };

    Ok(NvIoctlBuffers {
        input: select(
            *input,
            request & NV_IOCTL_WRITE != 0,
            "nvdrv ioctl input buffer does not match its encoded direction and size",
            "nvdrv ioctl without input carries a non-null input placeholder",
        )?,
        additional_input: None,
        output: select(
            *output,
            request & NV_IOCTL_READ != 0,
            "nvdrv ioctl output buffer does not match its encoded direction and size",
            "nvdrv ioctl without output carries a non-null output placeholder",
        )?,
    })
}

fn nv_ioctl2_buffers(hipc: &HipcRequest<'_>, request: u32) -> Result<NvIoctlBuffers, IpcWireError> {
    nv_ioctl2_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &hipc.receive_buffers,
        &hipc.receive_statics,
        request,
    )
}

fn nv_ioctl2_buffer_descriptors(
    send_statics: &[SendStaticDescriptor],
    send_buffers: &[BufferDescriptor],
    receive_buffers: &[BufferDescriptor],
    receive_statics: &ReceiveStatics,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [input, additional_input] = send_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly two input auto-select buffers",
        ));
    };
    let [output] = receive_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly one output auto-select buffer",
        ));
    };
    let [input_pointer, additional_input_pointer] = send_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly two input auto-select pointers",
        ));
    };
    let ReceiveStatics::Entries(output_pointers) = receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires one output auto-select pointer",
        ));
    };
    let [output_pointer] = output_pointers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly one output auto-select pointer",
        ));
    };
    if [input_pointer, additional_input_pointer]
        .into_iter()
        .any(|pointer| pointer.address != 0 || pointer.size != 0)
        || output_pointer.address != 0
        || output_pointer.size != 0
    {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 auto-select pointer placeholders are not null",
        ));
    }
    if additional_input.mode == BufferMode::Invalid
        || (additional_input.size != 0 && additional_input.address == 0)
    {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 additional input buffer is invalid",
        ));
    }

    let ordinary = nv_ioctl_buffer_descriptors(
        &[*input_pointer],
        &[*input],
        &[*output],
        &ReceiveStatics::Entries(vec![*output_pointer]),
        request,
    )?;
    Ok(NvIoctlBuffers {
        input: ordinary.input,
        additional_input: Some(*additional_input),
        output: ordinary.output,
    })
}

fn write_descriptor_bytes(
    process: &ExceptionProcessContext<'_>,
    descriptor: BufferDescriptor,
    bytes: &[u8],
) -> Result<(), IpcWireError> {
    if u64::try_from(bytes.len())
        .ok()
        .is_none_or(|size| size > descriptor.size)
    {
        return Err(IpcWireError::Malformed(
            "service response exceeds its output descriptor",
        ));
    }
    write_bytes(process, GuestVirtualAddress::new(descriptor.address), bytes)
}

fn dispatch_system_settings(
    process: &ExceptionProcessContext<'_>,
    request: CmifRequest<'_>,
    receive_statics: &ReceiveStatics,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // libnx uses commands 3 and 4 with a fixed-size 0x100-byte output
    // pointer. Keep this source reference beside the ABI implementation:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/set.c
    match request.command_id {
        3 | 4 => {
            let ReceiveStatics::Entries(descriptors) = receive_statics else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(descriptor) = descriptors.first() else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if usize::from(descriptor.size) < FIRMWARE_VERSION_SIZE {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            write_bytes(
                process,
                GuestVirtualAddress::new(descriptor.address),
                &emulated_firmware_version(),
            )?;
            log::debug!("set:sys returned emulated firmware version 1.0.0");
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        command_id => unsupported_service_command("set:sys", command_id),
    }
}

fn dispatch_user_settings(
    process: &ExceptionProcessContext<'_>,
    session: &UserSettingsSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // Command IDs and descriptor variants follow the pinned libnx `set`
    // client. Commands 1/3 are the pre-4.0 pointer-buffer forms; 5/6 are the
    // current map-alias forms:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/set.c
    match request.command_id {
        0 => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let language = session.environment().language();
            log::debug!("set returned current language {language:?}");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &language.code().to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        1 | 5 => {
            let descriptor = match request.command_id {
                1 => match &hipc.receive_statics {
                    ReceiveStatics::Entries(descriptors)
                        if descriptors.len() == 1
                            && hipc.receive_buffers.is_empty()
                            && descriptors[0].size > 0 =>
                    {
                        BufferDescriptor {
                            address: descriptors[0].address,
                            size: u64::from(descriptors[0].size),
                            mode: BufferMode::Normal,
                        }
                    }
                    _ => {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    }
                },
                5 => match hipc.receive_buffers.as_slice() {
                    [descriptor]
                        if descriptor.size > 0
                            && descriptor.mode != BufferMode::Invalid
                            && matches!(hipc.receive_statics, ReceiveStatics::None) =>
                    {
                        *descriptor
                    }
                    _ => {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    }
                },
                _ => unreachable!(),
            };
            if !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let capacity = usize::try_from(descriptor.size / 8).unwrap_or(usize::MAX);
            let count = capacity.min(SystemLanguage::AVAILABLE.len());
            let mut codes = Vec::with_capacity(count * 8);
            for language in &SystemLanguage::AVAILABLE[..count] {
                codes.extend_from_slice(&language.code().to_le_bytes());
            }
            write_descriptor_bytes(process, descriptor, &codes)?;
            let count = u32::try_from(count).expect("language table fits in a u32");
            log::debug!("set returned {count} available language codes");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &count.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        2 => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let Some(language) = request_u32(request.data, 0).and_then(SystemLanguage::from_raw)
            else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &language.code().to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        3 | 6 => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let count = u32::try_from(SystemLanguage::AVAILABLE.len())
                .expect("language table fits in a u32");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &count.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        4 => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let region = session.environment().region() as u32;
            log::debug!("set returned region {:?}", session.environment().region());
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &region.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        command_id => unsupported_service_command("set", command_id),
    }
}

fn has_ipc_descriptors(hipc: &HipcRequest<'_>) -> bool {
    hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.send_statics.is_empty()
        || !hipc.send_buffers.is_empty()
        || !hipc.receive_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
}

fn dispatch_performance_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &PerformanceManagerSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // Command IDs, payloads, and the returned child object follow libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/apm.c
    match request.command_id {
        0 => match process.handles_mut().insert(manager.open_session()) {
            Ok(handle) => {
                log::debug!("apm opened performance session handle {handle:#x}");
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ))
            }
            Err(_) => Err(IpcWireError::HostResourceExhausted(
                "installing a performance-manager child handle",
            )),
        },
        1 => {
            log::debug!("apm returned normal performance mode");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &PERFORMANCE_MODE_NORMAL.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        command_id => unsupported_service_command("apm", command_id),
    }
}

fn dispatch_performance_session(
    session: &PerformanceSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match request.command_id {
        0 => {
            let Some(mode) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(configuration) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Ok(mode) = usize::try_from(mode) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            if !session.set_configuration(mode, configuration) {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            log::debug!("apm stored configuration {configuration:#x} for mode {mode}");
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        1 => {
            let Some(mode) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let configuration = usize::try_from(mode)
                .ok()
                .and_then(|mode| session.configuration(mode));
            let Some(configuration) = configuration else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            log::debug!("apm returned configuration {configuration:#x} for mode {mode}");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &configuration.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        command_id => unsupported_service_command("IPerformanceSession", command_id),
    }
}

fn dispatch_vi(
    process: &mut ExceptionProcessContext<'_>,
    session: &ViSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let video = session.video();
    match session.kind() {
        ViObjectKind::Root(kind) => {
            if request.command_id != kind.required_root_command() {
                return unsupported_service_command(
                    vi_object_name(session.kind()),
                    request.command_id,
                );
            }
            vi_child(
                process,
                request.token,
                ViObjectKind::ApplicationDisplay,
                video,
            )
        }
        ViObjectKind::ApplicationDisplay => match request.command_id {
            100 => vi_child(process, request.token, ViObjectKind::BinderRelay, video),
            101 => vi_child(process, request.token, ViObjectKind::SystemDisplay, video),
            102 => vi_child(process, request.token, ViObjectKind::ManagerDisplay, video),
            1010 => {
                let Some(display_id) = video.open_display(request.data.get(..0x40).unwrap_or(&[]))
                else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &display_id.to_le_bytes(),
                        None,
                    )?,
                    None,
                ))
            }
            1020 => {
                let Some(display_id) = request_u64(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result = if VideoSystem::display_resolution(display_id).is_some() {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION
                };
                cmif_error(request.token, result)
            }
            1102 => {
                let Some(display_id) = request_u64(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some((width, height)) = VideoSystem::display_resolution(display_id) else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                let mut data = Vec::with_capacity(16);
                data.extend_from_slice(&i64::from(width).to_le_bytes());
                data.extend_from_slice(&i64::from(height).to_le_bytes());
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                    None,
                ))
            }
            2020 => {
                let Some(layer_id) = request_u64(request.data, 0x40) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some(layer) = video.layer(layer_id) else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                let descriptor = one_receive_buffer(hipc)?;
                let native_window = crate::graphics::encode_native_window(layer.binder_id);
                write_descriptor_bytes(process, descriptor, &native_window)?;
                Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &(native_window.len() as u64).to_le_bytes(),
                        None,
                    )?,
                    None,
                ))
            }
            2030 => {
                let Some(display_id) = request_u64(request.data, 8) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some(layer) = video.create_layer(display_id) else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                let descriptor = one_receive_buffer(hipc)?;
                let native_window = crate::graphics::encode_native_window(layer.binder_id);
                write_descriptor_bytes(process, descriptor, &native_window)?;
                let mut data = Vec::with_capacity(16);
                data.extend_from_slice(&layer.id.to_le_bytes());
                data.extend_from_slice(&(native_window.len() as u64).to_le_bytes());
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                    None,
                ))
            }
            2021 | 2031 => {
                let Some(layer_id) = request_u64(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result = if video.remove_layer(layer_id) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION
                };
                cmif_error(request.token, result)
            }
            2101 => {
                let Some(mode) = request_u32(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some(layer_id) = request_u64(request.data, 8) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result = if video.set_layer_scaling_mode(layer_id, mode) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION
                };
                cmif_error(request.token, result)
            }
            5202 => {
                let Some(display_id) = request_u64(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some(event) = video.vsync_event(display_id) else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                let handle = process.handles_mut().insert(event).map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a VI event handle")
                })?;
                Ok((
                    CmifResponse {
                        token: request.token,
                        result: HorizonIpcResult::SUCCESS.raw(),
                        data: &[],
                        pid: None,
                        copy_handles: &[handle],
                        move_handles: &[],
                        send_statics: &[],
                        is_domain: false,
                        domain_objects: &[],
                    }
                    .encode()?,
                    Some(handle),
                ))
            }
            command_id => unsupported_service_command("IApplicationDisplayService", command_id),
        },
        ViObjectKind::SystemDisplay => match request.command_id {
            1203 => {
                let Some(display_id) = request_u64(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some((width, height)) = VideoSystem::display_resolution(display_id) else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                let mut data = Vec::with_capacity(8);
                data.extend_from_slice(&width.to_le_bytes());
                data.extend_from_slice(&height.to_le_bytes());
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                    None,
                ))
            }
            // ABI layouts follow pinned libnx vi.c:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/vi.c#L411-L446
            2201 => {
                let (Some(x), Some(y), Some(layer_id)) = (
                    request_f32(request.data, 0),
                    request_f32(request.data, 4),
                    request_u64(request.data, 8),
                ) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result =
                    if x.is_finite() && y.is_finite() && video.set_layer_position(layer_id, x, y) {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                cmif_error(request.token, result)
            }
            2203 => {
                let (Some(layer_id), Some(width), Some(height)) = (
                    request_u64(request.data, 0),
                    request_i64(request.data, 8),
                    request_i64(request.data, 16),
                ) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result = match (u32::try_from(width), u32::try_from(height)) {
                    (Ok(width), Ok(height))
                        if width != 0
                            && height != 0
                            && video.set_layer_size(layer_id, width, height) =>
                    {
                        HorizonIpcResult::SUCCESS
                    }
                    _ => HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                };
                cmif_error(request.token, result)
            }
            2205 => {
                let (Some(layer_id), Some(z)) =
                    (request_u64(request.data, 0), request_i64(request.data, 8))
                else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result = if video.set_layer_z(layer_id, z) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION
                };
                cmif_error(request.token, result)
            }
            command_id => unsupported_service_command("ISystemDisplayService", command_id),
        },
        ViObjectKind::ManagerDisplay => match request.command_id {
            2010 | 2012 => {
                let Some(display_id) = request_u64(request.data, 8) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some(layer) = video.create_layer(display_id) else {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                };
                if request.command_id == 2012 {
                    let descriptor = one_receive_buffer(hipc)?;
                    let native_window = crate::graphics::encode_native_window(layer.binder_id);
                    write_descriptor_bytes(process, descriptor, &native_window)?;
                    let mut data = Vec::with_capacity(16);
                    data.extend_from_slice(&layer.id.to_le_bytes());
                    data.extend_from_slice(&(native_window.len() as u64).to_le_bytes());
                    return Ok((
                        encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                        None,
                    ));
                }
                Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &layer.id.to_le_bytes(),
                        None,
                    )?,
                    None,
                ))
            }
            2011 => {
                let Some(layer_id) = request_u64(request.data, 0) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let result = if video.remove_layer(layer_id) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION
                };
                cmif_error(request.token, result)
            }
            command_id => unsupported_service_command("IManagerDisplayService", command_id),
        },
        ViObjectKind::BinderRelay => dispatch_binder_relay(process, video, request, hipc),
    }
}

const fn vi_object_name(kind: ViObjectKind) -> &'static str {
    match kind {
        ViObjectKind::Root(ViServiceKind::Application) => "vi:u",
        ViObjectKind::Root(ViServiceKind::System) => "vi:s",
        ViObjectKind::Root(ViServiceKind::Manager) => "vi:m",
        ViObjectKind::ApplicationDisplay => "IApplicationDisplayService",
        ViObjectKind::BinderRelay => "IHOSBinderDriver",
        ViObjectKind::SystemDisplay => "ISystemDisplayService",
        ViObjectKind::ManagerDisplay => "IManagerDisplayService",
    }
}

fn dispatch_binder_relay(
    process: &mut ExceptionProcessContext<'_>,
    video: &VideoSystem,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(binder_id) = request_u32(request.data, 0).map(|value| value as i32) else {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    };
    match request.command_id {
        0 | 3 => {
            let Some(code) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let input = one_send_buffer(hipc)?;
            let output = one_receive_buffer(hipc)?;
            let input_size = usize::try_from(input.size)
                .map_err(|_| IpcWireError::Malformed("Binder input buffer is too large"))?;
            let mut encoded = vec![0_u8; input_size];
            read_bytes(
                process,
                GuestVirtualAddress::new(input.address),
                &mut encoded,
            )?;
            let transaction =
                video
                    .transact_binder(binder_id, code, &encoded)
                    .map_err(|error| match error {
                        crate::parcel::ParcelError::Malformed(reason) => {
                            IpcWireError::Malformed(reason)
                        }
                        crate::parcel::ParcelError::Unsupported(detail) => {
                            IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "IGraphicBufferProducer",
                                    command_id: code,
                                    detail,
                                },
                            )
                        }
                    })?;
            log::debug!(
                "Binder producer {binder_id} completed transaction {code:#x}{}",
                if transaction.queued.is_some() {
                    " and queued a frame"
                } else {
                    ""
                }
            );
            write_descriptor_bytes(process, output, &transaction.reply)?;
            if let Some(request) = transaction.queued {
                video
                    .queue_graphic_buffer(binder_id, request)
                    .map_err(|error| match error {
                        crate::graphics::FramebufferError::Malformed(reason) => {
                            IpcWireError::Malformed(reason)
                        }
                        crate::graphics::FramebufferError::Unsupported(detail) => {
                            IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "IGraphicBufferProducer",
                                    command_id: code,
                                    detail,
                                },
                            )
                        }
                        crate::graphics::FramebufferError::Backend(reason) => {
                            IpcWireError::GraphicsBackend(reason)
                        }
                        crate::graphics::FramebufferError::NvMap(_) => {
                            IpcWireError::Malformed("queued graphic-buffer image view is invalid")
                        }
                    })?;
            }
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        1 => {
            // IHOSBinderDriver::AdjustRefcount uses three signed 32-bit values;
            // reference type 0 is weak and 1 is strong:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/binder.c#L115-L126
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/display/binder.h#L33-L50
            let (Some(add_value), Some(reference_type)) =
                (request_i32(request.data, 4), request_i32(request.data, 8))
            else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if !video.adjust_binder_refcount(binder_id, add_value, reference_type) {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        2 => {
            let Some(event) = video.binder_event(binder_id) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            let handle = process.handles_mut().insert(event).map_err(|_| {
                IpcWireError::HostResourceExhausted("installing a Binder event handle")
            })?;
            Ok((
                CmifResponse {
                    token: request.token,
                    result: HorizonIpcResult::SUCCESS.raw(),
                    data: &[],
                    pid: None,
                    copy_handles: &[handle],
                    move_handles: &[],
                    send_statics: &[],
                    is_domain: false,
                    domain_objects: &[],
                }
                .encode()?,
                Some(handle),
            ))
        }
        command_id => unsupported_service_command("IHOSBinderDriver", command_id),
    }
}

fn dispatch_nvdrv(
    process: &mut ExceptionProcessContext<'_>,
    session: &NvDrvSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    caller_thread_id: u64,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let service = NvDrvService::new(session);
    match request.command_id {
        0 => {
            let descriptor = one_send_buffer(hipc)?;
            let size = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("nvdrv device path is too large"))?;
            if size == 0 || size > 0x100 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut path = vec![0_u8; size];
            read_bytes(
                process,
                GuestVirtualAddress::new(descriptor.address),
                &mut path,
            )?;
            let (fd, error) = match service.open(&path, process.process_id()) {
                Ok(fd) => (fd.raw(), crate::nvdrv::NV_SUCCESS),
                Err(NvDrvServiceError::DriverResult(error)) => (0, error),
                Err(NvDrvServiceError::Unsupported(operation)) => {
                    return Err(IpcWireError::UnsupportedNvDrv(operation));
                }
            };
            let mut data = Vec::with_capacity(8);
            data.extend_from_slice(&fd.to_le_bytes());
            data.extend_from_slice(&error.to_le_bytes());
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                None,
            ))
        }
        1 | 11 => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(ioctl) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let buffers = if request.command_id == 11 {
                // libnx `nvIoctl2` uses command 11 and carries the GPFIFO
                // descriptor array in a second input auto-select buffer:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L172-L208
                nv_ioctl2_buffers(hipc, ioctl)?
            } else {
                nv_ioctl_buffers(hipc, ioctl)?
            };
            let input_size = buffers
                .input
                .map(|descriptor| descriptor.size)
                .map_or(Ok(0), usize::try_from)
                .map_err(|_| IpcWireError::Malformed("nvdrv ioctl input is too large"))?;
            if input_size > 0x1000 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut input = vec![0_u8; input_size];
            if let Some(input_descriptor) = buffers.input {
                read_bytes(
                    process,
                    GuestVirtualAddress::new(input_descriptor.address),
                    &mut input,
                )?;
            }
            let additional_input_size = buffers
                .additional_input
                .map(|descriptor| descriptor.size)
                .map_or(Ok(0), usize::try_from)
                .map_err(|_| IpcWireError::Malformed("nvdrv Ioctl2 input is too large"))?;
            if additional_input_size > 0x1_0000 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut additional_input = vec![0_u8; additional_input_size];
            if let Some(input_descriptor) = buffers.additional_input {
                read_bytes(
                    process,
                    GuestVirtualAddress::new(input_descriptor.address),
                    &mut additional_input,
                )?;
            }
            let response = service
                .ioctl(crate::nvdrv::NvDrvIoctlRequest {
                    fd: NvDrvFileDescriptor::new(fd),
                    request: ioctl,
                    input: &input,
                    additional_input: &additional_input,
                    process_id: process.process_id(),
                    address_space: process.cpu().address_space_id(),
                    translator: process.canonical_memory(),
                    thread_id: caller_thread_id,
                })
                .map_err(IpcWireError::UnsupportedNvDrv)?;
            let response = match response {
                NvDrvIoctlOutcome::Complete(response) => response,
                NvDrvIoctlOutcome::PendingSyncpointWait(wait) => {
                    return Err(IpcWireError::PendingNvDrv(wait));
                }
            };
            if let Some(output_descriptor) = buffers.output {
                write_descriptor_bytes(process, output_descriptor, &response.output)?;
            }
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &response.driver_result.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        2 => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let error = service.close(NvDrvFileDescriptor::new(fd));
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &error.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        3 => {
            let Some(_transfer_size) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.copy_handles.len() != 2
                || hipc.copy_handles[0] != crate::CURRENT_PROCESS_HANDLE
                || process
                    .handles()
                    .get_as::<TransferMemoryObject>(hipc.copy_handles[1])
                    .is_none()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            service.initialize();
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        4 => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(event_id) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.pid.is_some()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (event, error) = service
                .query_event(NvDrvFileDescriptor::new(fd), event_id, process.process_id())
                .map_err(IpcWireError::UnsupportedNvDrv)?;
            let handle = if let Some(event) = event {
                Some(process.handles_mut().insert(event).map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing an nvdrv event handle")
                })?)
            } else {
                None
            };
            Ok((
                encode_nvdrv_query_event_response(request.token, error, handle)?,
                handle,
            ))
        }
        8 => {
            // INvDrvServices::SetAruid carries the caller PID descriptor and
            // an AppletResourceUserId. Associate both with the shared nvdrv
            // session so cloned service handles observe the same identity:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L104-L106
            let Some(applet_resource_user_id) = request_u64(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.pid.is_none()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            service.set_aruid(process.process_id(), applet_resource_user_id);
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        command_id => Err(IpcWireError::UnsupportedNvDrv(
            crate::nvdrv::UnsupportedNvDrvOperation::ServiceCommand { command_id },
        )),
    }
}

fn encode_nvdrv_query_event_response(
    token: u32,
    driver_result: u32,
    copy_handle: Option<u32>,
) -> Result<Vec<u8>, IpcWireError> {
    let copy_handles = copy_handle.as_slice();
    let response_data = driver_result.to_le_bytes();
    CmifResponse {
        token,
        result: HorizonIpcResult::SUCCESS.raw(),
        data: &response_data,
        pid: None,
        copy_handles,
        move_handles: &[],
        send_statics: &[],
        is_domain: false,
        domain_objects: &[],
    }
    .encode()
    .map_err(Into::into)
}

fn vi_child(
    process: &mut ExceptionProcessContext<'_>,
    token: u32,
    kind: ViObjectKind,
    video: &VideoSystem,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let handle = process
        .handles_mut()
        .insert(ViSession::new(kind, video.clone()))
        .map_err(|_| IpcWireError::HostResourceExhausted("installing a VI child handle"))?;
    Ok((
        encode_response(token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
        Some(handle),
    ))
}

fn dispatch_applet(
    process: &mut ExceptionProcessContext<'_>,
    session: &AppletSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    video_system: &VideoSystem,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // The startup order, command IDs, input PID/process handle, returned
    // objects, and scalar result layouts implemented below follow libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L112-L333
    let (object_id, input_objects) = match request.domain.as_ref() {
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => (*object_id, input_objects),
        Some(DomainRequest::Close { object_id }) => {
            let result = if session.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            log::debug!("appletOE closed domain object {object_id:#x}");
            return Ok((
                encode_domain_response(request.token, result, &[], &[], &[])?,
                None,
            ));
        }
        None => {
            return Err(IpcWireError::Malformed(
                "appletOE request was not sent through its domain",
            ));
        }
    };
    let Some(object) = session.object(object_id) else {
        return Ok((
            encode_domain_response(
                request.token,
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                &[],
                &[],
                &[],
            )?,
            None,
        ));
    };
    let accepts_input_objects =
        matches!(object, AppletObject::LibraryAppletAccessor { .. }) && request.command_id == 100;
    if !input_objects.is_empty() && !accepts_input_objects {
        return unsupported_service_command(applet_object_name(object), request.command_id);
    }

    match object {
        AppletObject::Root => {
            if hipc.pid.is_none()
                || hipc.copy_handles.as_slice() != [crate::CURRENT_PROCESS_HANDLE]
                || request_u64(request.data, 0) != Some(0)
                || request.data[8..].iter().any(|byte| *byte != 0)
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)
            {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            // libnx selects the root command from the process applet role:
            // application uses 0, while system applet uses 100. Keep the
            // resulting proxies distinct because their child command graphs
            // diverge after the shared controller interfaces.
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L155-L182
            let Some(kind) = applet_proxy_kind(request.command_id) else {
                return unsupported_service_command(applet_object_name(object), request.command_id);
            };
            applet_child(
                session,
                request.token,
                AppletObject::Proxy(kind),
                applet_proxy_name(kind),
            )
        }
        AppletObject::Proxy(kind) => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let Some(child) = applet_proxy_child(kind, request.command_id) else {
                return unsupported_service_command(applet_object_name(object), request.command_id);
            };
            applet_child(session, request.token, child, applet_object_name(child))
        }
        AppletObject::CommonStateGetter => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            match request.command_id {
                0 => {
                    let (_writable, readable) = EventObject::create_pair();
                    let handle = match process.handles_mut().insert(readable) {
                        Ok(handle) => handle,
                        Err(_) => {
                            return Err(IpcWireError::HostResourceExhausted(
                                "installing the common-state message event handle",
                            ));
                        }
                    };
                    log::debug!("appletOE returned message event handle {handle:#x}");
                    Ok((
                        encode_domain_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &[],
                            &[handle],
                            &[],
                        )?,
                        Some(handle),
                    ))
                }
                5 => applet_data(request.token, &[session.operation_mode().as_raw()]),
                6 => applet_data(request.token, &PERFORMANCE_MODE_NORMAL.to_le_bytes()),
                9 => applet_data(request.token, &[1]), // Application is in focus.
                command_id => unsupported_service_command("ICommonStateGetter", command_id),
            }
        }
        AppletObject::SelfController => match request.command_id {
            // SelfExit is a no-I/O request. Once AM accepts it, libnx sleeps
            // forever and relies on AM to terminate the process. The wire
            // layer therefore returns a typed lifecycle action after writing
            // the successful response instead of allowing that artificial
            // sleep loop to keep the emulated process alive:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L358-L405
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1094-L1099
            0 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                applet_data(request.token, &[])
            }
            // LockExit and UnlockExit mutate the application applet's shared
            // exit-deferral state and carry no input or output payload:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1094-L1099
            1 | 2 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                session.set_exit_locked(request.command_id == 1);
                applet_data(request.token, &[])
            }
            // GetLibraryAppletLaunchableEvent has no input and returns one
            // copied, manual-clear event handle (`autoclear=false`).
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c
            // https://switchbrew.org/w/index.php?title=Applet_Manager_services&oldid=14546#GetLibraryAppletLaunchableEvent
            9 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let handle = match process
                    .handles_mut()
                    .insert(session.library_applet_launchable_event())
                {
                    Ok(handle) => handle,
                    Err(_) => {
                        return Err(IpcWireError::HostResourceExhausted(
                            "installing the library-applet launchable event handle",
                        ));
                    }
                };
                log::debug!(
                    "ISelfController returned library-applet launchable event handle {handle:#x}"
                );
                Ok((
                    encode_domain_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &[],
                        &[handle],
                        &[],
                    )?,
                    Some(handle),
                ))
            }
            40 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let Some(layer) = video_system.create_layer(1) else {
                    return applet_error(
                        request.token,
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                    );
                };
                applet_data(request.token, &layer.id.to_le_bytes())
            }
            // These state-mutating command layouts follow pinned libnx:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1113-L1125
            11 => {
                let Some(enabled) = request.data.first() else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                if request.data[1..].iter().any(|byte| *byte != 0) || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                session.set_operation_mode_changed_notification(*enabled != 0);
                applet_data(request.token, &[])
            }
            12 => {
                let Some(enabled) = request.data.first() else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                if request.data[1..].iter().any(|byte| *byte != 0) || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                session.set_performance_mode_changed_notification(*enabled != 0);
                applet_data(request.token, &[])
            }
            13 => {
                let Some(mode) = request.data.get(..3) else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                if request.data[3..].iter().any(|byte| *byte != 0) || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                session.set_focus_handling_mode([mode[0] != 0, mode[1] != 0, mode[2] != 0]);
                applet_data(request.token, &[])
            }
            command_id => unsupported_service_command("ISelfController", command_id),
        },
        AppletObject::WindowController => match request.command_id {
            1 if request.data.is_empty() && !has_ipc_descriptors(hipc) => {
                applet_data(request.token, &process.process_id().to_le_bytes())
            }
            // AcquireForegroundRights has no input/output payload:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1271-L1279
            10 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                session.acquire_foreground_rights();
                applet_data(request.token, &[])
            }
            command_id => unsupported_service_command("IWindowController", command_id),
        },
        AppletObject::ApplicationFunctions => match request.command_id {
            40 if request.data.is_empty() && !has_ipc_descriptors(hipc) => {
                applet_data(request.token, &[1])
            }
            command_id => unsupported_service_command("IApplicationFunctions", command_id),
        },
        AppletObject::LibraryAppletCreator => match request.command_id {
            // CreateLibraryApplet takes one AppletId/LibAppletMode pair and
            // returns an ILibraryAppletAccessor domain object. libnx consumes
            // that accessor immediately to obtain its state-change event:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1516-L1564
            0 => {
                if request.data.len() < 8
                    || request.data[8..].iter().any(|byte| *byte != 0)
                    || has_ipc_descriptors(hipc)
                {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let Some(applet_id) =
                    request_u32(request.data, 0).and_then(LibraryAppletId::from_raw)
                else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                let Some(mode) = request_u32(request.data, 4).and_then(LibraryAppletMode::from_raw)
                else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                match session.create_library_applet(applet_id, mode) {
                    Ok(accessor_object_id) => {
                        log::debug!(
                            "ILibraryAppletCreator created {applet_id:?} in {mode:?} mode as domain object {accessor_object_id:#x}"
                        );
                        Ok((
                            encode_domain_response(
                                request.token,
                                HorizonIpcResult::SUCCESS,
                                &[],
                                &[],
                                &[accessor_object_id],
                            )?,
                            None,
                        ))
                    }
                    Err(CreateLibraryAppletError::Busy) => Err(IpcWireError::UnsupportedService(
                        UnsupportedServiceOperation::CommandVariant {
                            service: "ILibraryAppletCreator",
                            command_id: 0,
                            detail: "a prior library applet remains active",
                        },
                    )),
                    Err(CreateLibraryAppletError::DomainCapacityExhausted) => {
                        Err(IpcWireError::HostResourceExhausted(
                            "allocating a library-applet domain object",
                        ))
                    }
                    Err(CreateLibraryAppletError::NotDomain) => Err(IpcWireError::Internal(
                        "library applet creation escaped its domain session",
                    )),
                }
            }
            // CreateStorage allocates one zero-filled IStorage object. The
            // signed libnx size reaches CMIF as the same 64-bit bit pattern:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1761-L1767
            10 => {
                if request.data.len() < 8
                    || request.data[8..].iter().any(|byte| *byte != 0)
                    || has_ipc_descriptors(hipc)
                {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let size =
                    request_u64(request.data, 0).expect("validated CreateStorage payload length");
                match session.create_storage(size) {
                    Ok(storage_object_id) => Ok((
                        encode_domain_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &[],
                            &[],
                            &[storage_object_id],
                        )?,
                        None,
                    )),
                    Err(CreateAppletStorageError::DomainCapacityExhausted) => {
                        Err(IpcWireError::HostResourceExhausted(
                            "allocating an applet-storage domain object",
                        ))
                    }
                    Err(CreateAppletStorageError::SizeOutOfRange) => {
                        Err(IpcWireError::UnsupportedService(
                            UnsupportedServiceOperation::CommandVariant {
                                service: "ILibraryAppletCreator",
                                command_id: 10,
                                detail: "applet storage size cannot be represented by the host",
                            },
                        ))
                    }
                    Err(CreateAppletStorageError::AllocationFailed) => Err(
                        IpcWireError::HostResourceExhausted("allocating applet-storage backing"),
                    ),
                    Err(CreateAppletStorageError::NotDomain) => Err(IpcWireError::Internal(
                        "applet storage creation escaped its domain session",
                    )),
                }
            }
            command_id => unsupported_service_command("ILibraryAppletCreator", command_id),
        },
        AppletObject::LibraryAppletAccessor { .. } => match request.command_id {
            0 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let Some(event) = session.library_applet_state_changed_event(object_id) else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
                };
                let handle = process.handles_mut().insert(event).map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a library-applet event handle")
                })?;
                Ok((
                    encode_domain_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &[],
                        &[handle],
                        &[],
                    )?,
                    Some(handle),
                ))
            }
            // Start freezes the queued inputs into one launch request. Until
            // Nixe has a graphical system-applet host, Error applets become a
            // typed fatal diagnostic rather than hanging on their state event
            // or pretending that the dialog completed.
            10 => {
                if !request.data.is_empty()
                    || has_ipc_descriptors(hipc)
                    || !input_objects.is_empty()
                {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let launch = session
                    .prepare_library_applet_launch(object_id)
                    .map_err(|error| match error {
                        PrepareLibraryAppletLaunchError::AppletNotFound => IpcWireError::Internal(
                            "live library-applet accessor lost its launch state",
                        ),
                        PrepareLibraryAppletLaunchError::StorageBackingMissing => {
                            IpcWireError::Internal(
                                "library-applet input queue lost retained storage backing",
                            )
                        }
                        PrepareLibraryAppletLaunchError::AllocationFailed => {
                            IpcWireError::HostResourceExhausted(
                                "snapshotting library-applet launch inputs",
                            )
                        }
                    })?;
                if launch.applet_id != LibraryAppletId::Error {
                    return Err(IpcWireError::UnsupportedService(
                        UnsupportedServiceOperation::CommandVariant {
                            service: "ILibraryAppletAccessor",
                            command_id: 10,
                            detail: "library-applet execution coordinator is unavailable",
                        },
                    ));
                }
                Err(IpcWireError::ErrorApplet(Box::new(
                    crate::ErrorAppletDiagnostic::decode(&launch.input_storages)
                        .with_launch_mode(launch.mode.as_raw()),
                )))
            }
            // PushInData transfers one IStorage domain object into the
            // library applet's ordered input channel. libnx closes its local
            // storage object after dispatch, so the channel must retain the
            // backing independently until the applet consumes or releases it.
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L779-L793
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1754-L1758
            100 => {
                if !request.data.is_empty() || has_ipc_descriptors(hipc) || input_objects.len() != 1
                {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                match session.push_library_applet_input_storage(object_id, input_objects[0]) {
                    Ok(()) => applet_data(request.token, &[]),
                    Err(PushLibraryAppletStorageError::AllocationFailed) => Err(
                        IpcWireError::HostResourceExhausted("queuing library-applet input storage"),
                    ),
                    Err(PushLibraryAppletStorageError::AppletNotFound) => {
                        Err(IpcWireError::Internal(
                            "live library-applet accessor lost its active applet",
                        ))
                    }
                    Err(PushLibraryAppletStorageError::StorageNotFound) => {
                        Err(IpcWireError::UnsupportedService(
                            UnsupportedServiceOperation::CommandVariant {
                                service: "ILibraryAppletAccessor",
                                command_id: 100,
                                detail: "input domain object is not a live IStorage",
                            },
                        ))
                    }
                }
            }
            command_id => unsupported_service_command("ILibraryAppletAccessor", command_id),
        },
        AppletObject::Storage { storage_id } => match request.command_id {
            0 if request.data.is_empty() && !has_ipc_descriptors(hipc) => {
                let accessor_object_id = match session.open_storage_accessor(storage_id) {
                    Ok(object_id) => object_id,
                    Err(OpenAppletStorageAccessorError::StorageNotFound) => {
                        return Err(IpcWireError::Internal(
                            "live applet storage object lost its backing",
                        ));
                    }
                    Err(OpenAppletStorageAccessorError::DomainCapacityExhausted) => {
                        return Err(IpcWireError::HostResourceExhausted(
                            "allocating an applet-storage accessor domain object",
                        ));
                    }
                    Err(OpenAppletStorageAccessorError::ObjectIdExhausted) => {
                        return Err(IpcWireError::HostResourceExhausted(
                            "allocating an applet-storage accessor object ID",
                        ));
                    }
                };
                Ok((
                    encode_domain_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &[],
                        &[],
                        &[accessor_object_id],
                    )?,
                    None,
                ))
            }
            command_id => unsupported_service_command("IStorage", command_id),
        },
        AppletObject::StorageAccessor { storage_id } => match request.command_id {
            0 if request.data.is_empty() && !has_ipc_descriptors(hipc) => {
                let Some(size) = session.storage_size(storage_id) else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
                };
                applet_data(request.token, &size.to_le_bytes())
            }
            // libnx selects pointer or map-alias buffers according to the
            // transfer size and sends the byte offset as a signed 64-bit
            // value. Negative offsets retain their bit pattern and therefore
            // fail the checked storage range below:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1830-L1879
            10 => {
                if request.data.len() < 8
                    || request.data[8..].iter().any(|byte| *byte != 0)
                    || hipc.pid.is_some()
                    || !hipc.copy_handles.is_empty()
                    || !hipc.move_handles.is_empty()
                    || !matches!(hipc.receive_statics, ReceiveStatics::None)
                    || !hipc.receive_buffers.is_empty()
                    || !hipc.exchange_buffers.is_empty()
                {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let (address, size) = one_auto_select_input(hipc)?;
                let mut bytes = vec![0; size];
                read_bytes(process, GuestVirtualAddress::new(address), &mut bytes)?;
                let offset = request_u64(request.data, 0)
                    .expect("validated IStorageAccessor::Write payload length");
                match session.write_storage(storage_id, offset, &bytes) {
                    Ok(()) => applet_data(request.token, &[]),
                    Err(AppletStorageAccessError::NotFound) => {
                        applet_error(request.token, HorizonIpcResult::CMIF_TARGET_NOT_FOUND)
                    }
                    Err(AppletStorageAccessError::OutOfRange) => {
                        Err(IpcWireError::UnsupportedService(
                            UnsupportedServiceOperation::CommandVariant {
                                service: "IStorageAccessor",
                                command_id: 10,
                                detail: "out-of-range applet storage write",
                            },
                        ))
                    }
                }
            }
            11 => {
                if request.data.len() < 8
                    || request.data[8..].iter().any(|byte| *byte != 0)
                    || hipc.pid.is_some()
                    || !hipc.copy_handles.is_empty()
                    || !hipc.move_handles.is_empty()
                    || !hipc.send_statics.is_empty()
                    || !hipc.send_buffers.is_empty()
                    || !hipc.exchange_buffers.is_empty()
                {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                }
                let (address, size) = one_auto_select_output(hipc)?;
                let offset = request_u64(request.data, 0)
                    .expect("validated IStorageAccessor::Read payload length");
                match session.read_storage(storage_id, offset, size) {
                    Ok(bytes) => {
                        write_bytes(process, GuestVirtualAddress::new(address), &bytes)?;
                        applet_data(request.token, &[])
                    }
                    Err(AppletStorageAccessError::NotFound) => {
                        applet_error(request.token, HorizonIpcResult::CMIF_TARGET_NOT_FOUND)
                    }
                    Err(AppletStorageAccessError::OutOfRange) => {
                        Err(IpcWireError::UnsupportedService(
                            UnsupportedServiceOperation::CommandVariant {
                                service: "IStorageAccessor",
                                command_id: 11,
                                detail: "out-of-range applet storage read",
                            },
                        ))
                    }
                }
            }
            command_id => unsupported_service_command("IStorageAccessor", command_id),
        },
        AppletObject::HomeMenuFunctions
        | AppletObject::GlobalStateController
        | AppletObject::ApplicationCreator
        | AppletObject::AppletCommonFunctions
        | AppletObject::AudioController
        | AppletObject::DisplayController
        | AppletObject::DebugFunctions => {
            unsupported_service_command(applet_object_name(object), request.command_id)
        }
    }
}

fn dispatch_hid(
    process: &mut ExceptionProcessContext<'_>,
    session: &HidSession,
    hid_system: &HidSystem,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // libnx opens IAppletResource with command 0, sends the process ID, and
    // supplies the applet-resource user ID as one u64:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L800-L808
    match request.command_id {
        0 => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let handle = process
                .handles_mut()
                .insert(session.create_applet_resource())
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a HID applet-resource handle")
                })?;
            log::debug!("hid created IAppletResource handle {handle:#x}");
            semantic_success(request.token, false, &[], &[], &[], Some(handle))
        }
        // These commands configure Npad publication or start/stop its full-key
        // six-axis sensor. Player one is published as FullKey by HidSystem.
        // ABI: libnx hid.c at dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb.
        66 | 67 if hipc.pid.is_some() && request.data.len() >= 16 => {
            let handle = request_u32(request.data, 0).expect("validated HID handle payload");
            hid_system.set_six_axis_sensor_active(handle, request.command_id == 66);
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        100 if hipc.pid.is_some() && request.data.len() >= 16 => {
            let style_set = request_u32(request.data, 0).expect("validated HID style payload");
            hid_system.set_supported_npad_style_set(style_set);
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        102 if hipc.pid.is_some()
            && request.data.len() >= 8
            && matches!(
                (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice()),
                ([_], []) | ([], [_])
            ) =>
        {
            let (address, size) = match (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice())
            {
                ([descriptor], []) => (descriptor.address, usize::from(descriptor.size)),
                ([], [descriptor]) if descriptor.mode != BufferMode::Invalid => (
                    descriptor.address,
                    usize::try_from(descriptor.size)
                        .map_err(|_| IpcWireError::Malformed("HID Npad ID buffer is too large"))?,
                ),
                _ => unreachable!("validated HID Npad ID descriptor"),
            };
            if size == 0 || !size.is_multiple_of(4) || size > 10 * 4 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut encoded_ids = vec![0; size];
            read_bytes(process, GuestVirtualAddress::new(address), &mut encoded_ids)?;
            let ids = encoded_ids
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
            if !hid_system.set_supported_npad_ids(ids) {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        103 if hipc.pid.is_some() && request.data.len() >= 8 => {
            hid_system.activate_npad();
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        109 if hipc.pid.is_some() && request.data.len() >= 16 => {
            let revision = request_u32(request.data, 0).expect("validated HID revision payload");
            Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::CommandVariant {
                    service: "hid",
                    command_id: 109,
                    detail: match revision {
                        1 => "Npad shared-memory revision 1",
                        2 => "Npad shared-memory revision 2",
                        3 => "Npad shared-memory revision 3",
                        5 => "Npad shared-memory revision 5",
                        _ => "unknown Npad shared-memory revision",
                    },
                },
            ))
        }
        66 | 67 | 100 | 102 | 103 | 109 => {
            cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER)
        }
        command_id => unsupported_service_command("hid", command_id),
    }
}

fn dispatch_hid_applet_resource(
    process: &mut ExceptionProcessContext<'_>,
    resource: &HidAppletResource,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // IAppletResource command 0 returns the 0x40000-byte HID shared-memory
    // object as a copied handle; libnx maps it read-only immediately:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L47-L65
    if request.command_id != 0 {
        return unsupported_service_command("IAppletResource", request.command_id);
    }
    let handle = process
        .handles_mut()
        .insert(resource.shared_memory())
        .map_err(|_| {
            IpcWireError::HostResourceExhausted("installing a HID shared-memory handle")
        })?;
    log::debug!("hid returned shared-memory handle {handle:#x}");
    semantic_success(request.token, false, &[], &[handle], &[], None)
}

fn dispatch_time(
    process: &mut ExceptionProcessContext<'_>,
    session: &TimeServiceSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // The static-service object commands, returned child sessions, and copied
    // shared-memory handle follow the pinned libnx initialization sequence:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/time.c#L25-L80
    let child = match request.command_id {
        0 => Some(HandleObject::new(
            session.system_clock(SystemClockKind::User),
        )),
        1 => Some(HandleObject::new(
            session.system_clock(SystemClockKind::Network),
        )),
        2 => Some(HandleObject::new(session.steady_clock())),
        3 => Some(HandleObject::new(session.timezone_service())),
        4 => Some(HandleObject::new(
            session.system_clock(SystemClockKind::Local),
        )),
        20 => {
            let handle = process
                .handles_mut()
                .insert(session.shared_memory())
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a time shared-memory handle")
                })?;
            log::debug!("time:u returned shared-memory handle {handle:#x}");
            return semantic_success(request.token, false, &[], &[handle], &[], None);
        }
        command_id => return unsupported_service_command("time:u", command_id),
    };
    let handle = process
        .handles_mut()
        .insert_object(child.expect("time child command was selected"))
        .map_err(|_| {
            IpcWireError::HostResourceExhausted("installing a time service child handle")
        })?;
    log::debug!(
        "time:u command {} returned child session handle {handle:#x}",
        request.command_id
    );
    semantic_success(request.token, false, &[], &[], &[], Some(handle))
}

fn dispatch_system_clock(
    session: &SystemClockSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // ISystemClock command IDs and scalar layouts:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/time.c#L160-L185
    match request.command_id {
        0 => {
            let timestamp = session.current_time();
            semantic_success(
                request.token,
                false,
                &timestamp.to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        1 => {
            let Some(timestamp) = request.data.get(..8) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            session
                .set_current_time(i64::from_le_bytes(timestamp.try_into().unwrap()))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("updating the emulated system clock")
                })?;
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        command_id => unsupported_service_command("ISystemClock", command_id),
    }
}

fn dispatch_steady_clock(
    session: &SteadyClockSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match request.command_id {
        0 => {
            let (time_point, source_id) = session.time_point();
            let mut data = [0_u8; 0x18];
            data[..8].copy_from_slice(&time_point.to_le_bytes());
            data[8..].copy_from_slice(&source_id);
            semantic_success(request.token, false, &data, &[], &[], None)
        }
        200 => semantic_success(request.token, false, &0_i64.to_le_bytes(), &[], &[], None),
        command_id => unsupported_service_command("ISteadyClock", command_id),
    }
}

fn dispatch_timezone(
    session: &TimeZoneServiceSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // ITimeZoneService command 0 returns the fixed 0x24-byte location name:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/time.c#L187-L194
    match request.command_id {
        0 => semantic_success(
            request.token,
            false,
            &session.location_name(),
            &[],
            &[],
            None,
        ),
        2 => semantic_success(request.token, false, &1_u32.to_le_bytes(), &[], &[], None),
        101 => {
            let Some(timestamp) = request_u64(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Ok(timestamp) = i64::try_from(timestamp) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            let Some(data) = encode_calendar_time(session.timezone(), timestamp) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            semantic_success(request.token, false, &data, &[], &[], None)
        }
        command_id => unsupported_service_command("ITimeZoneService", command_id),
    }
}

fn encode_calendar_time(timezone: chrono_tz::Tz, timestamp: i64) -> Option<[u8; 0x20]> {
    let utc = chrono::DateTime::from_timestamp(timestamp, 0)?;
    let local = utc.with_timezone(&timezone);
    let year = u16::try_from(local.year()).ok()?;
    let mut data = [0_u8; 0x20];
    data[..2].copy_from_slice(&year.to_le_bytes());
    data[2] = local.month() as u8;
    data[3] = local.day() as u8;
    data[4] = local.hour() as u8;
    data[5] = local.minute() as u8;
    data[6] = local.second() as u8;
    data[8..12].copy_from_slice(&local.weekday().num_days_from_sunday().to_le_bytes());
    data[12..16].copy_from_slice(&local.ordinal0().to_le_bytes());
    let abbreviation = local.format("%Z").to_string();
    let abbreviation = abbreviation.as_bytes();
    let abbreviation_len = abbreviation.len().min(8);
    data[16..16 + abbreviation_len].copy_from_slice(&abbreviation[..abbreviation_len]);
    let dst = u32::from(local.offset().dst_offset().num_seconds() != 0);
    data[24..28].copy_from_slice(&dst.to_le_bytes());
    let offset = local.offset().fix().local_minus_utc();
    data[28..32].copy_from_slice(&offset.to_le_bytes());
    Some(data)
}

fn applet_child(
    session: &AppletSession,
    token: u32,
    object: AppletObject,
    name: &'static str,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(object_id) = session.insert_object(object) else {
        return Err(IpcWireError::HostResourceExhausted(
            "allocating an applet domain child object",
        ));
    };
    log::debug!("appletOE opened {name} as domain object {object_id:#x}");
    Ok((
        encode_domain_response(token, HorizonIpcResult::SUCCESS, &[], &[], &[object_id])?,
        None,
    ))
}

fn applet_data(token: u32, data: &[u8]) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((
        encode_domain_response(token, HorizonIpcResult::SUCCESS, data, &[], &[])?,
        None,
    ))
}

fn applet_error(
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((encode_domain_response(token, result, &[], &[], &[])?, None))
}

const fn applet_object_name(object: AppletObject) -> &'static str {
    match object {
        AppletObject::Root => "IApplicationProxyService",
        AppletObject::Proxy(kind) => applet_proxy_name(kind),
        AppletObject::ApplicationFunctions => "IApplicationFunctions",
        AppletObject::HomeMenuFunctions => "IHomeMenuFunctions",
        AppletObject::GlobalStateController => "IGlobalStateController",
        AppletObject::ApplicationCreator => "IApplicationCreator",
        AppletObject::AppletCommonFunctions => "IAppletCommonFunctions",
        AppletObject::LibraryAppletCreator => "ILibraryAppletCreator",
        AppletObject::CommonStateGetter => "ICommonStateGetter",
        AppletObject::SelfController => "ISelfController",
        AppletObject::WindowController => "IWindowController",
        AppletObject::AudioController => "IAudioController",
        AppletObject::DisplayController => "IDisplayController",
        AppletObject::DebugFunctions => "IDebugFunctions",
        AppletObject::LibraryAppletAccessor { .. } => "ILibraryAppletAccessor",
        AppletObject::Storage { .. } => "IStorage",
        AppletObject::StorageAccessor { .. } => "IStorageAccessor",
    }
}

const fn applet_proxy_name(kind: AppletProxyKind) -> &'static str {
    match kind {
        AppletProxyKind::Application => "IApplicationProxy",
        AppletProxyKind::SystemApplet => "ISystemAppletProxy",
    }
}

const fn applet_proxy_kind(command_id: u32) -> Option<AppletProxyKind> {
    match command_id {
        0 => Some(AppletProxyKind::Application),
        100 => Some(AppletProxyKind::SystemApplet),
        _ => None,
    }
}

const fn applet_proxy_child(kind: AppletProxyKind, command_id: u32) -> Option<AppletObject> {
    Some(match command_id {
        0 => AppletObject::CommonStateGetter,
        1 => AppletObject::SelfController,
        2 => AppletObject::WindowController,
        3 => AppletObject::AudioController,
        4 => AppletObject::DisplayController,
        11 => AppletObject::LibraryAppletCreator,
        1000 => AppletObject::DebugFunctions,
        20 => match kind {
            AppletProxyKind::Application => AppletObject::ApplicationFunctions,
            AppletProxyKind::SystemApplet => AppletObject::HomeMenuFunctions,
        },
        21 if matches!(kind, AppletProxyKind::SystemApplet) => AppletObject::GlobalStateController,
        22 if matches!(kind, AppletProxyKind::SystemApplet) => AppletObject::ApplicationCreator,
        23 if matches!(kind, AppletProxyKind::SystemApplet) => AppletObject::AppletCommonFunctions,
        _ => return None,
    })
}

fn cmif_error(
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((encode_response(token, result, &[], None)?, None))
}

fn request_u32(data: &[u8], offset: usize) -> Option<u32> {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
}

fn request_i32(data: &[u8], offset: usize) -> Option<i32> {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(i32::from_le_bytes)
}

fn request_u64(data: &[u8], offset: usize) -> Option<u64> {
    data.get(offset..offset + 8)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
}

fn request_i64(data: &[u8], offset: usize) -> Option<i64> {
    data.get(offset..offset + 8)
        .and_then(|bytes| bytes.try_into().ok())
        .map(i64::from_le_bytes)
}

fn request_f32(data: &[u8], offset: usize) -> Option<f32> {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(f32::from_le_bytes)
}

fn emulated_firmware_version() -> [u8; FIRMWARE_VERSION_SIZE] {
    // SetSysFirmwareVersion's verified field layout is defined by libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/set.h
    // The field values reproduce the documented retail NX 1.0.0 system
    // version title rather than identifying the emulator:
    // https://switchbrew.org/w/index.php?title=System_Version_Title&oldid=14763
    let mut version = [0; FIRMWARE_VERSION_SIZE];
    version[0] = 1;
    version[4] = 15;
    version[8..10].copy_from_slice(b"NX");
    version[0x28..0x50].copy_from_slice(b"84b8da475a02261c456e6472b403b31416480165");
    version[0x68..0x6d].copy_from_slice(b"1.0.0");
    version[0x80..0xa4].copy_from_slice(b"NintendoSDK Firmware for NX 1.0.0-15");
    version
}

fn encode_response(
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    move_handle: Option<u32>,
) -> Result<Vec<u8>, IpcWireError> {
    let move_handle_storage = move_handle.into_iter().collect::<Vec<_>>();
    CmifResponse {
        token,
        result: result.raw(),
        data,
        move_handles: &move_handle_storage,
        ..CmifResponse::default()
    }
    .encode()
    .map_err(Into::into)
}

#[cfg(test)]
mod semantic_wire_tests {
    use super::*;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn filesystem_proxy_decodes_the_sd_card_open_command() {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4);
        put_u32(&mut command, 4, 8);
        put_u32(&mut command, 16, 0x4943_4653);
        put_u32(&mut command, 24, 18);
        let hipc = HipcRequest::decode(&command).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(IpcService::FileSystem, &request, &hipc).unwrap(),
            Some(IpcRequest::OpenSdCardFileSystem)
        );
    }

    #[test]
    fn aoc_current_process_commands_require_pid_and_decode_bounds() {
        let mut count = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut count, 0, 4);
        put_u32(&mut count, 4, 10 | (1 << 31));
        put_u32(&mut count, 8, 1);
        put_u64(&mut count, 12, 7);
        put_u32(&mut count, 32, 0x4943_4653);
        put_u32(&mut count, 40, 2);
        let hipc = HipcRequest::decode(&count).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(IpcService::AddOnContent, &request, &hipc).unwrap(),
            Some(IpcRequest::GetIndexedAddOnContentCount)
        );

        let mut list = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut list, 0, 4 | (1 << 24));
        put_u32(&mut list, 4, 10 | (1 << 31));
        put_u32(&mut list, 8, 1);
        put_u64(&mut list, 12, 7);
        // One normal receive buffer at 0x1000, with room for four u32 indices.
        put_u32(&mut list, 20, 16);
        put_u32(&mut list, 24, 0x1000);
        put_u32(&mut list, 28, 0);
        put_u32(&mut list, 32, 0x4943_4653);
        put_u32(&mut list, 40, 3);
        put_u32(&mut list, 48, 2);
        put_u32(&mut list, 52, 10);
        let hipc = HipcRequest::decode(&list).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(IpcService::AddOnContent, &request, &hipc).unwrap(),
            Some(IpcRequest::ListIndexedAddOnContent {
                offset: 2,
                max_entries: 4,
            })
        );

        let mut without_pid = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut without_pid, 0, 4);
        put_u32(&mut without_pid, 4, 8);
        put_u32(&mut without_pid, 16, 0x4943_4653);
        put_u32(&mut without_pid, 24, 2);
        let hipc = HipcRequest::decode(&without_pid).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(IpcService::AddOnContent, &request, &hipc).unwrap(),
            None
        );
    }
}

fn encode_domain_response(
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    copy_handles: &[u32],
    domain_objects: &[u32],
) -> Result<Vec<u8>, IpcWireError> {
    CmifResponse {
        token,
        result: result.raw(),
        data,
        copy_handles,
        is_domain: true,
        domain_objects,
        ..CmifResponse::default()
    }
    .encode()
    .map_err(Into::into)
}

fn decode_service_name(encoded: &[u8]) -> Option<&[u8]> {
    let end = encoded
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(encoded.len());
    if end == 0 || encoded[end..].iter().any(|byte| *byte != 0) {
        None
    } else {
        Some(&encoded[..end])
    }
}

pub(crate) fn read_bytes(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    output: &mut [u8],
) -> Result<(), IpcWireError> {
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = read_u8(process, add(start, index)?)?;
    }
    Ok(())
}

pub(crate) fn write_bytes(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    bytes: &[u8],
) -> Result<(), IpcWireError> {
    for (index, byte) in bytes.iter().copied().enumerate() {
        process
            .memory()
            .write(
                process.cpu().address_space_id(),
                add(start, index)?,
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(byte),
            )
            .map_err(IpcWireError::GuestMemory)?;
    }
    Ok(())
}

pub(crate) fn validate_writable_ram_range(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    size: usize,
) -> Result<(), IpcWireError> {
    let address_space = process.cpu().address_space_id();
    let end = start.get().checked_add(size as u64).ok_or_else(|| {
        IpcWireError::GuestMemory(DataAccessFault::new(
            address_space,
            start,
            DataAccessKind::Write,
            DataAccessFaultReason::AddressOverflow,
        ))
    })?;
    let limit = GuestVirtualAddress::new(process.address_space_limit());
    let mut cursor = start;
    while cursor.get() < end {
        let Some(mapping) = process.memory().query_memory(address_space, cursor, limit) else {
            return Err(IpcWireError::GuestMemory(DataAccessFault::new(
                address_space,
                cursor,
                DataAccessKind::Write,
                DataAccessFaultReason::Unmapped,
            )));
        };
        if mapping.region != Some(MemoryRegionKind::Ram) {
            return Err(IpcWireError::GuestMemory(DataAccessFault::new(
                address_space,
                cursor,
                DataAccessKind::Write,
                DataAccessFaultReason::Device(
                    "IPC response buffer must be backed by ordinary RAM".into(),
                ),
            )));
        }
        if !mapping.permissions.contains(MemoryPermissions::WRITE) {
            return Err(IpcWireError::GuestMemory(DataAccessFault::new(
                address_space,
                cursor,
                DataAccessKind::Write,
                DataAccessFaultReason::WritePermissionDenied,
            )));
        }
        let mapping_end = mapping
            .base
            .get()
            .checked_add(mapping.size)
            .ok_or(IpcWireError::Internal("guest memory query range overflows"))?;
        if mapping_end <= cursor.get() {
            return Err(IpcWireError::Internal(
                "guest memory query did not advance while validating an IPC response",
            ));
        }
        cursor = GuestVirtualAddress::new(mapping_end.min(end));
    }
    Ok(())
}

fn write_response(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    capacity: usize,
    response: &[u8],
) -> Result<(), IpcWireError> {
    if response.len() > capacity {
        return Err(IpcWireError::Internal(
            "encoded IPC response exceeds its prevalidated command buffer",
        ));
    }
    write_bytes(process, start, response).map_err(|error| match error {
        IpcWireError::GuestMemory(fault) => IpcWireError::ResponseCommit(fault),
        error => error,
    })
}

fn read_u8(
    process: &ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
) -> Result<u8, IpcWireError> {
    let value = process
        .memory()
        .read(
            process.cpu().address_space_id(),
            address,
            MemoryAccess::normal(MemoryAccessSize::Byte),
        )
        .map_err(IpcWireError::GuestMemory)?
        .value;
    let MemoryValue::U8(value) = value else {
        unreachable!("byte access returns a byte value")
    };
    Ok(value)
}

fn add(address: GuestVirtualAddress, offset: usize) -> Result<GuestVirtualAddress, IpcWireError> {
    let offset = u64::try_from(offset)
        .map_err(|_| IpcWireError::Malformed("guest address offset overflows"))?;
    address
        .checked_add(offset)
        .ok_or(IpcWireError::Malformed("guest address overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_message::ReceiveStaticDescriptor;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_send_static(bytes: &mut [u8], offset: usize, address: u64, size: u16) {
        let first = (((address >> 36) as u32 & 0x3f) << 6)
            | (((address >> 32) as u32 & 0xf) << 12)
            | (u32::from(size) << 16);
        put_u32(bytes, offset, first);
        put_u32(bytes, offset + 4, address as u32);
    }

    fn put_buffer(bytes: &mut [u8], offset: usize, address: u64, size: u64) {
        put_u32(bytes, offset, size as u32);
        put_u32(bytes, offset + 4, address as u32);
        put_u32(
            bytes,
            offset + 8,
            ((address >> 36) as u32 & 0x3f_ffff) << 2
                | ((size >> 32) as u32 & 0xf) << 24
                | ((address >> 32) as u32 & 0xf) << 28,
        );
    }

    fn auto_select_input(
        pointer: (u64, u16),
        map_alias: (u64, u64),
    ) -> Result<(u64, usize), IpcWireError> {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4 | (1 << 16) | (1 << 20));
        put_send_static(&mut command, 8, pointer.0, pointer.1);
        put_buffer(&mut command, 16, map_alias.0, map_alias.1);
        let hipc = HipcRequest::decode(&command).unwrap();
        one_auto_select_input(&hipc)
    }

    fn auto_select_output(
        pointer: (u64, u16),
        map_alias: (u64, u64),
    ) -> Result<(u64, usize), IpcWireError> {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4 | (1 << 24));
        put_u32(&mut command, 4, 3 << 10);
        put_buffer(&mut command, 8, map_alias.0, map_alias.1);
        put_u32(&mut command, 20, pointer.0 as u32);
        put_u32(
            &mut command,
            24,
            ((pointer.0 >> 32) as u32 & 0xffff) | (u32::from(pointer.1) << 16),
        );
        let hipc = HipcRequest::decode(&command).unwrap();
        one_auto_select_output(&hipc)
    }

    fn nv_ioctl_descriptors(
        input: BufferDescriptor,
        output: BufferDescriptor,
        request: u32,
    ) -> Result<NvIoctlBuffers, IpcWireError> {
        nv_ioctl_buffer_descriptors(
            &[SendStaticDescriptor {
                address: 0,
                size: 0,
                index: 0,
            }],
            &[input],
            &[output],
            &ReceiveStatics::Entries(vec![ReceiveStaticDescriptor {
                address: 0,
                size: 0,
            }]),
            request,
        )
    }

    fn nv_ioctl2_descriptors(
        input: BufferDescriptor,
        additional_input: BufferDescriptor,
        output: BufferDescriptor,
        request: u32,
    ) -> Result<NvIoctlBuffers, IpcWireError> {
        nv_ioctl2_buffer_descriptors(
            &[
                SendStaticDescriptor {
                    address: 0,
                    size: 0,
                    index: 0,
                },
                SendStaticDescriptor {
                    address: 0,
                    size: 0,
                    index: 1,
                },
            ],
            &[input, additional_input],
            &[output],
            &ReceiveStatics::Entries(vec![ReceiveStaticDescriptor {
                address: 0,
                size: 0,
            }]),
            request,
        )
    }

    fn buffer(address: u64, size: u64) -> BufferDescriptor {
        BufferDescriptor {
            address,
            size,
            mode: BufferMode::Normal,
        }
    }

    #[test]
    fn auto_select_input_decodes_the_libnx_descriptor_pair() {
        assert_eq!(auto_select_input((0, 0), (0x2000, 8)), Ok((0x2000, 8)));
        assert_eq!(auto_select_input((0x3000, 4), (0, 0)), Ok((0x3000, 4)));
        assert_eq!(auto_select_input((0, 0), (0, 0)), Ok((0, 0)));
    }

    #[test]
    fn auto_select_input_rejects_ambiguous_or_invalid_pairs() {
        assert_eq!(
            auto_select_input((0x1000, 4), (0x2000, 4)),
            Err(IpcWireError::Malformed(
                "auto-select input has both descriptor sides active"
            ))
        );
        assert_eq!(
            auto_select_input((0, 4), (0, 0)),
            Err(IpcWireError::Malformed(
                "auto-select input has a null address with nonzero size"
            ))
        );
    }

    #[test]
    fn auto_select_output_decodes_the_libnx_descriptor_pair() {
        assert_eq!(auto_select_output((0, 0), (0x4000, 16)), Ok((0x4000, 16)));
        assert_eq!(auto_select_output((0x5000, 12), (0, 0)), Ok((0x5000, 12)));
        assert_eq!(
            auto_select_output((0x5000, 12), (0x4000, 16)),
            Err(IpcWireError::Malformed(
                "auto-select output has both descriptor sides active"
            ))
        );
    }

    #[test]
    fn complete_descriptor_check_includes_receive_statics() {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4);
        put_u32(&mut command, 4, 3 << 10);
        put_u32(&mut command, 8, 0x2000);
        put_u32(&mut command, 12, 4 << 16);
        let hipc = HipcRequest::decode(&command).unwrap();

        assert!(has_ipc_descriptors(&hipc));
    }

    #[test]
    fn write_only_nv_ioctl_accepts_libnx_null_output_placeholder() {
        assert_eq!(
            nv_ioctl_descriptors(buffer(0x1000, 40), buffer(0, 0), 0x4028_4109),
            Ok(NvIoctlBuffers {
                input: Some(buffer(0x1000, 40)),
                additional_input: None,
                output: None,
            })
        );
    }

    #[test]
    fn nv_ioctl_direction_does_not_hide_a_non_null_placeholder() {
        assert_eq!(
            nv_ioctl_descriptors(buffer(0x1000, 40), buffer(0x2000, 40), 0x4028_4109),
            Err(IpcWireError::Malformed(
                "nvdrv ioctl without output carries a non-null output placeholder"
            ))
        );
    }

    #[test]
    fn nv_ioctl2_retains_the_independently_sized_additional_input() {
        assert_eq!(
            nv_ioctl2_descriptors(
                buffer(0x1000, 24),
                buffer(0x2000, 0x40),
                buffer(0x3000, 24),
                0xc018_481b,
            ),
            Ok(NvIoctlBuffers {
                input: Some(buffer(0x1000, 24)),
                additional_input: Some(buffer(0x2000, 0x40)),
                output: Some(buffer(0x3000, 24)),
            })
        );
        assert_eq!(
            nv_ioctl2_descriptors(
                buffer(0x1000, 24),
                buffer(0, 8),
                buffer(0x3000, 24),
                0xc018_481b,
            ),
            Err(IpcWireError::Malformed(
                "nvdrv Ioctl2 additional input buffer is invalid"
            ))
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
    fn applet_proxy_children_follow_the_caller_role() {
        assert_eq!(applet_proxy_kind(100), Some(AppletProxyKind::SystemApplet));
        assert_eq!(applet_proxy_kind(200), None);
        assert_eq!(
            applet_proxy_child(AppletProxyKind::Application, 20),
            Some(AppletObject::ApplicationFunctions)
        );
        assert_eq!(
            applet_proxy_child(AppletProxyKind::SystemApplet, 20),
            Some(AppletObject::HomeMenuFunctions)
        );
        assert_eq!(
            applet_proxy_child(AppletProxyKind::SystemApplet, 23),
            Some(AppletObject::AppletCommonFunctions)
        );
        assert_eq!(applet_proxy_child(AppletProxyKind::Application, 23), None);
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
    fn wire_dispatch_has_no_generic_unsupported_guest_fallback() {
        let source = include_str!("ipc_wire.rs");
        let unknown_command = ["HorizonIpcResult::", "CMIF_UNKNOWN_COMMAND_ID"].concat();
        let not_supported = ["HorizonIpcResult::", "CMIF_NOT_SUPPORTED"].concat();

        assert!(!source.contains(&unknown_command));
        assert!(!source.contains(&not_supported));
    }

    #[test]
    fn service_names_require_canonical_zero_padding() {
        assert_eq!(decode_service_name(b"fsp-srv\0"), Some(&b"fsp-srv"[..]));
        assert_eq!(decode_service_name(b"aoc:u\0\0\0"), Some(&b"aoc:u"[..]));
        assert_eq!(decode_service_name(b"\0\0\0\0\0\0\0\0"), None);
        assert_eq!(decode_service_name(b"fs\0bad!!"), None);
    }

    #[test]
    fn response_layout_round_trips_libnx_parser_offsets() {
        let response = encode_response(
            7,
            HorizonIpcResult::SUCCESS,
            &0x100_u16.to_le_bytes(),
            Some(0x44),
        )
        .unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());
        assert_eq!(word(4) >> 31, 1);
        assert_eq!(word(8), 1 << 5);
        assert_eq!(word(12), 0x44);
        assert_eq!(word(16), 0x4f43_4653);
        assert_eq!(word(24), 0);
        assert_eq!(word(28), 7);
        assert_eq!(&response[32..34], &0x100_u16.to_le_bytes());
    }

    #[test]
    fn nvdrv_query_event_returns_the_libnx_copy_handle_shape() {
        let response =
            encode_nvdrv_query_event_response(9, crate::nvdrv::NV_SUCCESS, Some(0x55)).unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());

        assert_eq!(word(4) >> 31, 1);
        assert_eq!(word(8), 1 << 1);
        assert_eq!(word(12), 0x55);
        assert_eq!(word(16), 0x4f43_4653);
        assert_eq!(word(24), HorizonIpcResult::SUCCESS.raw());
        assert_eq!(word(28), 9);
        assert_eq!(word(32), crate::nvdrv::NV_SUCCESS);
    }

    #[test]
    fn response_encodes_the_typed_horizon_result_without_translation() {
        let response =
            encode_response(0x33, HorizonIpcResult::SM_NOT_REGISTERED, &[], None).unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());

        assert_eq!(word(16), 0x4f43_4653);
        assert_eq!(word(24), 0xe15);
        assert_eq!(word(28), 0x33);
    }

    #[test]
    fn madrid_calendar_conversion_applies_iana_daylight_saving_rules() {
        let winter = encode_calendar_time(
            "Europe/Madrid".parse().unwrap(),
            1_704_067_200, // 2024-01-01 00:00:00 UTC
        )
        .unwrap();
        assert_eq!(&winter[..7], &[0xe8, 0x07, 1, 1, 1, 0, 0]);
        assert_eq!(u32::from_le_bytes(winter[24..28].try_into().unwrap()), 0);
        assert_eq!(
            i32::from_le_bytes(winter[28..32].try_into().unwrap()),
            3_600
        );

        let summer = encode_calendar_time(
            "Europe/Madrid".parse().unwrap(),
            1_719_792_000, // 2024-07-01 00:00:00 UTC
        )
        .unwrap();
        assert_eq!(&summer[..7], &[0xe8, 0x07, 7, 1, 2, 0, 0]);
        assert_eq!(u32::from_le_bytes(summer[24..28].try_into().unwrap()), 1);
        assert_eq!(
            i32::from_le_bytes(summer[28..32].try_into().unwrap()),
            7_200
        );
    }

    #[test]
    fn emulated_firmware_uses_the_verified_setsys_layout() {
        let version = emulated_firmware_version();

        assert_eq!(&version[..3], &[1, 0, 0]);
        assert_eq!(&version[4..6], &[15, 0]);
        assert_eq!(&version[8..10], b"NX");
        assert_eq!(
            &version[0x28..0x50],
            b"84b8da475a02261c456e6472b403b31416480165"
        );
        assert_eq!(&version[0x68..0x6d], b"1.0.0");
        assert_eq!(
            &version[0x80..0xa4],
            b"NintendoSDK Firmware for NX 1.0.0-15"
        );
        assert_eq!(version.len(), 0x100);
    }
}
