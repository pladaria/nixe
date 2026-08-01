//! Synchronous Horizon service-call adapter built on the checked wire codec.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

use chrono::{Datelike, Offset, Timelike};
use chrono_tz::OffsetComponents;
use nixe_cpu::address::GuestVirtualAddress;
use nixe_cpu::memory::{DataAccessFault, MemoryAccess, MemoryAccessSize, MemoryValue};
use nixe_memory::CanonicalRangeAccessError;
use nixe_runtime::{EventObject, ExceptionProcessContext, HandleObject, TransferMemoryObject};

use crate::ipc_message::{
    BufferDescriptor, BufferMode, COMMAND_BUFFER_SIZE, CmifRequest, CmifResponse, DomainRequest,
    HipcRequest, MessageError, ReceiveStatics, SendStaticDescriptor,
};
use crate::nvdrv::NvDrvIoctlOutcome;
use crate::nvdrv::{NvDrvFileDescriptor, NvDrvService, NvDrvServiceError};
use crate::object::AppletObject;
use crate::{
    AccountSession, AppletSession, DirectoryEntryKind, HidAppletResource, HidSession, HidSystem,
    HorizonIpcResult, HostDirectoryFileSystem, HostFile, IpcDispatcher, IpcRequest, IpcResponse,
    IpcResultCode, IpcService, IpcSession, MAX_IPC_LIST_ENTRIES, MAX_IPC_PATH_BYTES,
    MAX_IPC_READ_BYTES, NvDrvSession, OperationMode, PerformanceManagerSession, PerformanceSession,
    ReadOnlyDirectory, ReadOnlyFile, ReadOnlyFileSystem, ServiceManagerSession, SteadyClockSession,
    SystemClockKind, SystemClockSession, SystemSettingsSession, TimeEnvironment,
    TimeServiceSession, TimeZoneServiceSession, ViObjectKind, ViServiceKind, ViSession,
    VideoSystem,
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
    ResourceExhausted,
    CanonicalBacking(CanonicalRangeAccessError),
    UnsupportedService(UnsupportedServiceOperation),
    UnsupportedNvDrv(crate::nvdrv::UnsupportedNvDrvOperation),
    /// A decoded direct nvdrv wait which must suspend at the SVC boundary.
    PendingNvDrv(crate::nvdrv::PendingNvHostCtrlWait),
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
    pub caller_thread_id: u64,
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
    let manager = process
        .handles()
        .get_as::<ServiceManagerSession>(handle)
        .cloned();
    let service = process.handles().get_as::<IpcSession>(handle).cloned();
    let settings = process
        .handles()
        .get_as::<SystemSettingsSession>(handle)
        .copied();
    let performance_manager = process
        .handles()
        .get_as::<PerformanceManagerSession>(handle)
        .cloned();
    let performance = process
        .handles()
        .get_as::<PerformanceSession>(handle)
        .cloned();
    let applet = process.handles().get_as::<AppletSession>(handle).cloned();
    let account = process.handles().get_as::<AccountSession>(handle).cloned();
    let hid = process.handles().get_as::<HidSession>(handle).cloned();
    let hid_applet_resource = process
        .handles()
        .get_as::<HidAppletResource>(handle)
        .cloned();
    let time = process
        .handles()
        .get_as::<TimeServiceSession>(handle)
        .cloned();
    let system_clock = process
        .handles()
        .get_as::<SystemClockSession>(handle)
        .cloned();
    let steady_clock = process
        .handles()
        .get_as::<SteadyClockSession>(handle)
        .cloned();
    let timezone = process
        .handles()
        .get_as::<TimeZoneServiceSession>(handle)
        .cloned();
    let vi = process.handles().get_as::<ViSession>(handle).cloned();
    let nvdrv = process.handles().get_as::<NvDrvSession>(handle).cloned();
    let semantic_object = process.handles().get(handle).cloned().filter(|object| {
        object.is::<ReadOnlyFileSystem>()
            || object.is::<HostDirectoryFileSystem>()
            || object.is::<ReadOnlyFile>()
            || object.is::<HostFile>()
            || object.is::<ReadOnlyDirectory>()
    });
    if manager.is_none()
        && service.is_none()
        && settings.is_none()
        && performance_manager.is_none()
        && performance.is_none()
        && applet.is_none()
        && account.is_none()
        && hid.is_none()
        && hid_applet_resource.is_none()
        && time.is_none()
        && system_clock.is_none()
        && steady_clock.is_none()
        && timezone.is_none()
        && vi.is_none()
        && nvdrv.is_none()
        && semantic_object.is_none()
    {
        return Ok(SyncRequestResult::InvalidHandle);
    }

    if size < COMMAND_BUFFER_SIZE {
        return Err(IpcWireError::Malformed(
            "IPC message buffer is smaller than the TLS command buffer",
        ));
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(size)
        .map_err(|_| IpcWireError::ResourceExhausted)?;
    buffer.resize(size, 0);
    read_bytes(process, address, &mut buffer)?;
    let hipc = HipcRequest::decode(&buffer).map_err(|error| IpcWireError::Malformed(error.0))?;
    let is_domain = applet.as_ref().is_some_and(AppletSession::is_domain)
        || service.as_ref().is_some_and(IpcSession::is_domain);
    let request = CmifRequest::decode(&hipc, is_domain).map_err(|error| {
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
            && let Some(applet) = &applet
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
            write_bytes(process, address, &response)?;
            log::debug!("appletOE converted to domain with root object {object_id:#x}");
            return Ok(SyncRequestResult::Success);
        }
        if request.command_id == 0
            && let Some(service) = &service
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
            write_bytes(process, address, &response)?;
            log::debug!(
                "{:?} converted to domain with root object {object_id:#x}",
                String::from_utf8_lossy(service.service().name())
            );
            return Ok(SyncRequestResult::Success);
        }
        if matches!(request.command_id, 2 | 4)
            && let Some(service) = &service
        {
            // CloneCurrentObject (2) returns a moved session handle. The Ex
            // form (4) additionally carries a session-manager tag which the
            // public libnx ABI documents as unused by official servers:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L308-L337
            // A cloned `IpcSession` has a distinct process-handle object while
            // retaining the shared domain table of the source connection.
            let cloned_handle = process
                .handles_mut()
                .insert(service.clone())
                .map_err(|_| IpcWireError::ResourceExhausted)?;
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
            if let Err(error) = write_bytes(process, address, &response) {
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
            && let Some(vi) = &vi
        {
            let cloned_handle = process
                .handles_mut()
                .insert(vi.clone())
                .map_err(|_| IpcWireError::ResourceExhausted)?;
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            )?;
            write_bytes(process, address, &response)?;
            return Ok(SyncRequestResult::Success);
        }
        if matches!(request.command_id, 2 | 4)
            && let Some(nvdrv) = &nvdrv
        {
            // CMIF cloning creates another service connection into the same
            // nvdrv client. The connections share initialization, descriptors,
            // allocations, and close effects, but retain distinct identities
            // for descriptor ownership.
            let cloned_session = nvdrv
                .clone_connection()
                .ok_or(IpcWireError::ResourceExhausted)?;
            let cloned_handle = process
                .handles_mut()
                .insert(cloned_session)
                .map_err(|_| IpcWireError::ResourceExhausted)?;
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                Some(cloned_handle),
            )?;
            write_bytes(process, address, &response)?;
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
        write_bytes(process, address, &response)?;
        return Ok(SyncRequestResult::Success);
    }
    let (response, created_handle) = if let Some(manager) = manager {
        dispatch_service_manager(
            process,
            &manager,
            request,
            hipc.pid.is_some(),
            initial_operation_mode,
            time_environment,
            host_systems,
        )?
    } else if settings.is_some() {
        dispatch_system_settings(process, request, &hipc.receive_statics)?
    } else if let Some(manager) = performance_manager {
        dispatch_performance_manager(process, &manager, request)?
    } else if let Some(session) = performance {
        dispatch_performance_session(&session, request)?
    } else if let Some(applet) = applet {
        dispatch_applet(process, &applet, request, &hipc, host_systems.video)?
    } else if let Some(account) = account {
        dispatch_account(process, &account, request, &hipc)?
    } else if let Some(hid) = hid {
        dispatch_hid(process, &hid, host_systems.hid, request, &hipc)?
    } else if let Some(resource) = hid_applet_resource {
        dispatch_hid_applet_resource(process, &resource, request)?
    } else if let Some(time) = time {
        dispatch_time(process, &time, request)?
    } else if let Some(clock) = system_clock {
        dispatch_system_clock(&clock, request)?
    } else if let Some(clock) = steady_clock {
        dispatch_steady_clock(&clock, request)?
    } else if let Some(timezone) = timezone {
        dispatch_timezone(&timezone, request)?
    } else if let Some(vi) = vi {
        dispatch_vi(process, &vi, request, &hipc)?
    } else if let Some(nvdrv) = nvdrv {
        match dispatch_nvdrv(
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
        }
    } else if let Some(service) = service {
        dispatch_semantic_service(process, &service, request, &hipc)?
    } else if let Some(object) = semantic_object {
        dispatch_plain_semantic_object(process, &object, request, &hipc)?
    } else {
        unreachable!("typed session kind was checked")
    };
    if let Err(error) = write_bytes(process, address, &response) {
        if let Some(handle) = created_handle {
            let _ = process.handles_mut().close(handle);
        }
        return Err(error);
    }
    Ok(SyncRequestResult::Success)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SyncRequestResult {
    Success,
    InvalidHandle,
    PendingNvDrv(crate::nvdrv::PendingNvHostCtrlWait),
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
            if matches!(name, b"set:sys" | b"apm" | b"appletOE" | b"hid" | b"time:u") {
                return connect_system_service(
                    process,
                    request.token,
                    name,
                    initial_operation_mode,
                    time_environment,
                    host_systems.hid,
                );
            }
            // libnx opens acc:u0 for application account sessions. Retain the
            // real session identity here; commands remain fail-fast until
            // their account semantics are implemented:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/acc.c#L25-L30
            if name == b"acc:u0" {
                if !process.mounts().allows_service(name) {
                    return Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SM_NOT_ALLOWED,
                            &[],
                            None,
                        )?,
                        None,
                    ));
                }
                let handle = process
                    .handles_mut()
                    .insert(AccountSession::new())
                    .map_err(|_| IpcWireError::ResourceExhausted)?;
                return Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ));
            }
            if let Some(kind) = ViServiceKind::from_name(name) {
                if !process.mounts().allows_service(name) {
                    return Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SM_NOT_ALLOWED,
                            &[],
                            None,
                        )?,
                        None,
                    ));
                }
                let handle = process
                    .handles_mut()
                    .insert(ViSession::new(
                        ViObjectKind::Root(kind),
                        host_systems.video.clone(),
                    ))
                    .map_err(|_| IpcWireError::ResourceExhausted)?;
                return Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ));
            }
            if matches!(name, b"nvdrv" | b"nvdrv:a" | b"nvdrv:s") {
                if !process.mounts().allows_service(name) {
                    return Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SM_NOT_ALLOWED,
                            &[],
                            None,
                        )?,
                        None,
                    ));
                }
                let handle = process
                    .handles_mut()
                    .insert(host_systems.video.nvdrv())
                    .map_err(|_| IpcWireError::ResourceExhausted)?;
                return Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ));
            }
            let Some(service) = IpcService::from_name(name) else {
                if !process.mounts().allows_service(name) {
                    return Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SM_NOT_ALLOWED,
                            &[],
                            None,
                        )?,
                        None,
                    ));
                }
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::Connect { name: name.into() },
                ));
            };
            let (mounts, handles) = process.mounts_and_handles_mut();
            match IpcDispatcher::connect(mounts, handles, service) {
                Ok(handle) => {
                    log::debug!("sm:GetService returned session handle {handle:#x}");
                    Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &[],
                            Some(handle),
                        )?,
                        Some(handle),
                    ))
                }
                Err(error) if error == IpcResultCode::ACCESS_DENIED => Ok((
                    encode_response(request.token, HorizonIpcResult::SM_NOT_ALLOWED, &[], None)?,
                    None,
                )),
                Err(error) if error == IpcResultCode::RESOURCE_LIMIT => Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_OUT_OF_SESSIONS,
                        &[],
                        None,
                    )?,
                    None,
                )),
                Err(_) => Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_NOT_REGISTERED,
                        &[],
                        None,
                    )?,
                    None,
                )),
            }
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

fn connect_system_service(
    process: &mut ExceptionProcessContext<'_>,
    token: u32,
    name: &[u8],
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    hid_system: &HidSystem,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if !process.mounts().allows_service(name) {
        return Ok((
            encode_response(token, HorizonIpcResult::SM_NOT_ALLOWED, &[], None)?,
            None,
        ));
    }
    let handle = match name {
        b"set:sys" => process.handles_mut().insert(SystemSettingsSession::new()),
        b"apm" => process
            .handles_mut()
            .insert(PerformanceManagerSession::new()),
        b"appletOE" => process
            .handles_mut()
            .insert(AppletSession::new(initial_operation_mode)),
        b"hid" => process
            .handles_mut()
            .insert(HidSession::new(hid_system.shared_memory())),
        b"time:u" => time_environment
            .create_service()
            .and_then(|session| process.handles_mut().insert(session)),
        _ => unreachable!("system service name was checked"),
    };
    match handle {
        Ok(handle) => {
            log::debug!(
                "sm:GetService returned {:?} session handle {handle:#x}",
                String::from_utf8_lossy(name)
            );
            Ok((
                encode_response(token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                Some(handle),
            ))
        }
        Err(_) => Ok((
            encode_response(token, HorizonIpcResult::SM_OUT_OF_SESSIONS, &[], None)?,
            None,
        )),
    }
}

fn dispatch_semantic_service(
    process: &mut ExceptionProcessContext<'_>,
    session: &IpcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    enum Target {
        Root,
        Object(HandleObject),
    }

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
                Target::Root
            } else {
                let Some(object) = session.object(*object_id) else {
                    return semantic_error(
                        request.token,
                        session.service(),
                        Some(session),
                        HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                    );
                };
                Target::Object(object)
            }
        }
        None if session.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain service request omitted its domain header",
            ));
        }
        None => Target::Root,
    };

    let semantic_request = match &target {
        Target::Root => decode_root_request(session.service(), &request, hipc)?,
        Target::Object(object) => decode_object_request(process, object, &request, hipc)?,
    };
    let Some(semantic_request) = semantic_request else {
        return unsupported_service_command(
            semantic_service_name(session.service()),
            request.command_id,
        );
    };

    let semantic_result = {
        let (mounts, handles) = process.mounts_and_handles_mut();
        match &target {
            Target::Root => {
                IpcDispatcher::dispatch_session(mounts, handles, session, semantic_request)
            }
            Target::Object(object) => {
                IpcDispatcher::dispatch_object(mounts, handles, object, semantic_request)
            }
        }
    };
    let response = match semantic_result {
        Ok(response) => response,
        Err(error) if error == IpcResultCode::INVALID_COMMAND => {
            return unsupported_service_command(
                semantic_service_name(session.service()),
                request.command_id,
            );
        }
        Err(error) if error == IpcResultCode::INTERNAL_STATE => {
            return Err(IpcWireError::Internal(
                "semantic service entered an invalid internal state",
            ));
        }
        Err(error) => {
            return semantic_error(
                request.token,
                session.service(),
                Some(session),
                HorizonIpcResult::from_semantic(session.service(), error),
            );
        }
    };
    encode_semantic_response(
        process,
        session.service(),
        Some(session),
        request,
        hipc,
        response,
    )
}

fn dispatch_plain_semantic_object(
    process: &mut ExceptionProcessContext<'_>,
    object: &HandleObject,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(semantic_request) = decode_object_request(process, object, &request, hipc)? else {
        return unsupported_service_command(semantic_object_name(object), request.command_id);
    };
    let result = {
        let (mounts, handles) = process.mounts_and_handles_mut();
        IpcDispatcher::dispatch_object(mounts, handles, object, semantic_request)
    };
    match result {
        Ok(response) => encode_semantic_response(
            process,
            IpcService::FileSystem,
            None,
            request,
            hipc,
            response,
        ),
        Err(error) if error == IpcResultCode::INVALID_COMMAND => {
            unsupported_service_command(semantic_object_name(object), request.command_id)
        }
        Err(error) if error == IpcResultCode::INTERNAL_STATE => Err(IpcWireError::Internal(
            "semantic object entered an invalid internal state",
        )),
        Err(error) => semantic_error(
            request.token,
            IpcService::FileSystem,
            None,
            HorizonIpcResult::from_semantic(IpcService::FileSystem, error),
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
                    return Err(IpcWireError::ResourceExhausted);
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
                .map_err(|_| IpcWireError::ResourceExhausted)?;
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
                .map_err(|_| IpcWireError::ResourceExhausted)?;
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

const NV_IOCTL_WRITE: u32 = 1 << 30;
const NV_IOCTL_READ: u32 = 1 << 31;
const NV_IOCTL_SIZE_MASK: u32 = 0x3fff;

// nvIoctl derives the two buffer presences and their common size directly
// from these Linux-style request fields before issuing CMIF command 1:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L137-L170

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvIoctlBuffers {
    input: Option<BufferDescriptor>,
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
        output: select(
            *output,
            request & NV_IOCTL_READ != 0,
            "nvdrv ioctl output buffer does not match its encoded direction and size",
            "nvdrv ioctl without output carries a non-null output placeholder",
        )?,
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
            Err(_) => cmif_error(request.token, HorizonIpcResult::SM_OUT_OF_SESSIONS),
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
                let handle = process
                    .handles_mut()
                    .insert(event)
                    .map_err(|_| IpcWireError::ResourceExhausted)?;
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
            if let Some((slot, buffer)) = transaction.queued {
                let object = video
                    .nvdrv()
                    .nvmap_object_by_id(crate::NvMapExportedId::new(buffer.nvmap_id))
                    .ok_or(IpcWireError::Malformed(
                        "queued graphic buffer references an unknown nvmap ID",
                    ))?;
                if buffer.plane_size == 0
                    || buffer.plane_size > u64::from(buffer.total_size)
                    || u64::from(buffer.offset)
                        .checked_add(buffer.plane_size)
                        .is_none_or(|end| end > u64::from(object.size()))
                {
                    return Err(IpcWireError::Malformed(
                        "queued graphic-buffer plane exceeds its nvmap allocation",
                    ));
                }
                let view = object
                    .image_view(buffer.nvmap_view_metadata())
                    .map_err(map_nvmap_view_error)?;
                let bytes = view.read_plane(0).map_err(map_nvmap_view_error)?;
                video
                    .queue_software_frame(binder_id, slot, &buffer, &bytes)
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
            let handle = process
                .handles_mut()
                .insert(event)
                .map_err(|_| IpcWireError::ResourceExhausted)?;
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

fn map_nvmap_view_error(error: crate::NvMapViewError) -> IpcWireError {
    match error {
        crate::NvMapViewError::ResourceExhausted => IpcWireError::ResourceExhausted,
        crate::NvMapViewError::Backing(error) => IpcWireError::CanonicalBacking(error),
        crate::NvMapViewError::UnallocatedObject => {
            IpcWireError::Malformed("queued nvmap object has no canonical backing")
        }
        crate::NvMapViewError::MissingPlanes => {
            IpcWireError::Malformed("queued graphic buffer has no planes")
        }
        crate::NvMapViewError::UnknownPlane => {
            IpcWireError::Malformed("queued graphic-buffer plane does not exist")
        }
        crate::NvMapViewError::PlaneOutsideObject => {
            IpcWireError::Malformed("queued graphic-buffer plane exceeds its nvmap object")
        }
        crate::NvMapViewError::RangeOverflow => {
            IpcWireError::Malformed("queued graphic-buffer plane range overflows")
        }
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
        1 => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(ioctl) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let buffers = nv_ioctl_buffers(hipc, ioctl)?;
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
            let response = service
                .ioctl(crate::nvdrv::NvDrvIoctlRequest {
                    fd: NvDrvFileDescriptor::new(fd),
                    request: ioctl,
                    input: &input,
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
                Some(
                    process
                        .handles_mut()
                        .insert(event)
                        .map_err(|_| IpcWireError::ResourceExhausted)?,
                )
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
        .map_err(|_| IpcWireError::ResourceExhausted)?;
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
    if !input_objects.is_empty() {
        return unsupported_service_command("appletOE", request.command_id);
    }
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

    match object {
        AppletObject::Root => {
            if request.command_id != 0 {
                return unsupported_service_command(applet_object_name(object), request.command_id);
            }
            if hipc.pid.is_none()
                || hipc.copy_handles.as_slice() != [crate::CURRENT_PROCESS_HANDLE]
                || request_u64(request.data, 0) != Some(0)
            {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            applet_child(
                session,
                request.token,
                AppletObject::ApplicationProxy,
                "IApplicationProxy",
            )
        }
        AppletObject::ApplicationProxy => {
            let child = match request.command_id {
                0 => AppletObject::CommonStateGetter,
                1 => AppletObject::SelfController,
                2 => AppletObject::WindowController,
                3 => AppletObject::AudioController,
                4 => AppletObject::DisplayController,
                11 => AppletObject::LibraryAppletCreator,
                20 => AppletObject::ApplicationFunctions,
                1000 => AppletObject::DebugFunctions,
                _ => {
                    return unsupported_service_command(
                        applet_object_name(object),
                        request.command_id,
                    );
                }
            };
            applet_child(session, request.token, child, applet_object_name(child))
        }
        AppletObject::CommonStateGetter => match request.command_id {
            0 => {
                let (_writable, readable) = EventObject::create_pair();
                let handle = match process.handles_mut().insert(readable) {
                    Ok(handle) => handle,
                    Err(_) => {
                        return applet_error(request.token, HorizonIpcResult::SM_OUT_OF_SESSIONS);
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
        },
        AppletObject::SelfController => match request.command_id {
            40 => {
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
                session.set_operation_mode_changed_notification(*enabled != 0);
                applet_data(request.token, &[])
            }
            12 => {
                let Some(enabled) = request.data.first() else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                session.set_performance_mode_changed_notification(*enabled != 0);
                applet_data(request.token, &[])
            }
            13 => {
                let Some(mode) = request.data.get(..3) else {
                    return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                session.set_focus_handling_mode([mode[0] != 0, mode[1] != 0, mode[2] != 0]);
                applet_data(request.token, &[])
            }
            command_id => unsupported_service_command("ISelfController", command_id),
        },
        AppletObject::WindowController => match request.command_id {
            1 => applet_data(request.token, &process.process_id().to_le_bytes()),
            // AcquireForegroundRights has no input/output payload:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1271-L1279
            10 => {
                session.acquire_foreground_rights();
                applet_data(request.token, &[])
            }
            command_id => unsupported_service_command("IWindowController", command_id),
        },
        AppletObject::ApplicationFunctions => match request.command_id {
            40 => applet_data(request.token, &[1]),
            command_id => unsupported_service_command("IApplicationFunctions", command_id),
        },
        AppletObject::LibraryAppletCreator
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
                .map_err(|_| IpcWireError::ResourceExhausted)?;
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
        .map_err(|_| IpcWireError::ResourceExhausted)?;
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
                .map_err(|_| IpcWireError::ResourceExhausted)?;
            log::debug!("time:u returned shared-memory handle {handle:#x}");
            return semantic_success(request.token, false, &[], &[handle], &[], None);
        }
        command_id => return unsupported_service_command("time:u", command_id),
    };
    let handle = process
        .handles_mut()
        .insert_object(child.expect("time child command was selected"))
        .map_err(|_| IpcWireError::ResourceExhausted)?;
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
                .map_err(|_| IpcWireError::ResourceExhausted)?;
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
        return applet_error(token, HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
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
        AppletObject::ApplicationProxy => "IApplicationProxy",
        AppletObject::ApplicationFunctions => "IApplicationFunctions",
        AppletObject::LibraryAppletCreator => "ILibraryAppletCreator",
        AppletObject::CommonStateGetter => "ICommonStateGetter",
        AppletObject::SelfController => "ISelfController",
        AppletObject::WindowController => "IWindowController",
        AppletObject::AudioController => "IAudioController",
        AppletObject::DisplayController => "IDisplayController",
        AppletObject::DebugFunctions => "IDebugFunctions",
    }
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

    fn buffer(address: u64, size: u64) -> BufferDescriptor {
        BufferDescriptor {
            address,
            size,
            mode: BufferMode::Normal,
        }
    }

    #[test]
    fn write_only_nv_ioctl_accepts_libnx_null_output_placeholder() {
        assert_eq!(
            nv_ioctl_descriptors(buffer(0x1000, 40), buffer(0, 0), 0x4028_4109),
            Ok(NvIoctlBuffers {
                input: Some(buffer(0x1000, 40)),
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
