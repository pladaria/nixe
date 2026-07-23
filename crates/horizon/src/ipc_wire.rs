//! Synchronous Horizon service-call adapter built on the checked wire codec.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

use nixe_cpu::address::GuestVirtualAddress;
use nixe_cpu::memory::{
    DataAccessFault, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue,
};
use nixe_runtime::{EventObject, ExceptionProcessContext, HandleObject, SharedMemoryObject};

use crate::ipc_message::{
    BufferDescriptor, BufferMode, COMMAND_BUFFER_SIZE, CmifRequest, CmifResponse, DomainRequest,
    HipcRequest, MessageError, ReceiveStatics, SendStaticDescriptor,
};
use crate::object::AppletObject;
use crate::{
    AppletSession, DirectoryEntryKind, HidAppletResource, HidSession, HorizonIpcResult,
    IpcDispatcher, IpcRequest, IpcResponse, IpcResultCode, IpcService, IpcSession,
    MAX_IPC_LIST_ENTRIES, MAX_IPC_PATH_BYTES, MAX_IPC_READ_BYTES, OperationMode,
    PerformanceManagerSession, PerformanceSession, ReadOnlyDirectory, ReadOnlyFile,
    ReadOnlyFileSystem, ServiceManagerSession, SystemSettingsSession,
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
const HID_SHARED_MEMORY_SIZE: usize = 0x40000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum IpcWireError {
    GuestMemory(DataAccessFault),
    Malformed(&'static str),
    ResourceExhausted,
}

impl From<MessageError> for IpcWireError {
    fn from(error: MessageError) -> Self {
        Self::Malformed(error.0)
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

pub(crate) fn send_sync_request(
    process: &mut ExceptionProcessContext<'_>,
    tls: GuestVirtualAddress,
    handle: u32,
    initial_operation_mode: OperationMode,
) -> Result<SyncRequestResult, IpcWireError> {
    send_sync_request_from_buffer(
        process,
        tls,
        COMMAND_BUFFER_SIZE,
        handle,
        initial_operation_mode,
    )
}

pub(crate) fn send_sync_request_from_buffer(
    process: &mut ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    handle: u32,
    initial_operation_mode: OperationMode,
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
    let hid = process.handles().get_as::<HidSession>(handle).cloned();
    let hid_applet_resource = process
        .handles()
        .get_as::<HidAppletResource>(handle)
        .cloned();
    let semantic_object = process.handles().get(handle).cloned().filter(|object| {
        object.is::<ReadOnlyFileSystem>()
            || object.is::<ReadOnlyFile>()
            || object.is::<ReadOnlyDirectory>()
    });
    if manager.is_none()
        && service.is_none()
        && settings.is_none()
        && performance_manager.is_none()
        && performance.is_none()
        && applet.is_none()
        && hid.is_none()
        && hid_applet_resource.is_none()
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
    let hipc = HipcRequest::decode(&buffer)?;
    let is_domain = applet.as_ref().is_some_and(AppletSession::is_domain)
        || service.as_ref().is_some_and(IpcSession::is_domain);
    let request = CmifRequest::decode(&hipc, is_domain)?;
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
        let response = match request.command_id {
            // QueryPointerBufferSize. Zero makes libnx use map-alias buffers,
            // which the future descriptor bridge can validate explicitly.
            3 => encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &0_u16.to_le_bytes(),
                None,
            ),
            0 | 1 | 2 | 4 => encode_response(
                request.token,
                HorizonIpcResult::CMIF_NOT_SUPPORTED,
                &[],
                None,
            ),
            _ => encode_response(
                request.token,
                HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID,
                &[],
                None,
            ),
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
        )?
    } else if settings.is_some() {
        dispatch_system_settings(process, request, &hipc.receive_statics)?
    } else if let Some(manager) = performance_manager {
        dispatch_performance_manager(process, &manager, request)?
    } else if let Some(session) = performance {
        dispatch_performance_session(&session, request)?
    } else if let Some(applet) = applet {
        dispatch_applet(process, &applet, request, &hipc)?
    } else if let Some(hid) = hid {
        dispatch_hid(process, &hid, request, &hipc)?
    } else if let Some(resource) = hid_applet_resource {
        dispatch_hid_applet_resource(process, &resource, request)?
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SyncRequestResult {
    Success,
    InvalidHandle,
}

fn dispatch_service_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &ServiceManagerSession,
    request: CmifRequest<'_>,
    sent_pid: bool,
    initial_operation_mode: OperationMode,
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
            if matches!(name, b"set:sys" | b"apm" | b"appletOE" | b"hid") {
                return connect_system_service(
                    process,
                    request.token,
                    name,
                    initial_operation_mode,
                );
            }
            let Some(service) = IpcService::from_name(name) else {
                let encoded_name: [u8; 8] = encoded_name
                    .try_into()
                    .expect("service name length was checked");
                if manager.first_unavailable_request(encoded_name) {
                    log::warn!(
                        "guest requested unavailable Horizon service via sm: {:?}",
                        String::from_utf8_lossy(name)
                    );
                }
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_NOT_REGISTERED,
                        &[],
                        None,
                    )?,
                    None,
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
        _ => Ok((
            encode_response(
                request.token,
                HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID,
                &[],
                None,
            )?,
            None,
        )),
    }
}

fn connect_system_service(
    process: &mut ExceptionProcessContext<'_>,
    token: u32,
    name: &[u8],
    initial_operation_mode: OperationMode,
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
        b"hid" => {
            let shared_memory = SharedMemoryObject::zeroed_with_remote_permissions(
                HID_SHARED_MEMORY_SIZE,
                MemoryPermissions::READ,
            )
            .map_err(|_| IpcWireError::ResourceExhausted)?;
            process.handles_mut().insert(HidSession::new(shared_memory))
        }
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
                return semantic_error(
                    request.token,
                    session.service(),
                    Some(session),
                    HorizonIpcResult::CMIF_NOT_SUPPORTED,
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
        return semantic_error(
            request.token,
            session.service(),
            Some(session),
            HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID,
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
        return semantic_error(
            request.token,
            IpcService::FileSystem,
            None,
            HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID,
        );
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
        Err(error) => semantic_error(
            request.token,
            IpcService::FileSystem,
            None,
            HorizonIpcResult::from_semantic(IpcService::FileSystem, error),
        ),
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
    if object.is::<ReadOnlyFileSystem>() {
        return match request.command_id {
            // IFileSystem OpenFile/OpenDirectory use one input pointer path
            // and a u32 mode:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L878-L893
            8 => Ok(Some(IpcRequest::OpenFile {
                path: read_path(process, hipc)?,
            })),
            9 => Ok(Some(IpcRequest::OpenDirectory {
                path: read_path(process, hipc)?,
            })),
            _ => Ok(None),
        };
    }
    if object.is::<ReadOnlyFile>() {
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
                    .map_err(|_| IpcWireError::Malformed("semantic child handle disappeared"))?;
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
        _ => cmif_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
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
        _ => cmif_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
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
        _ => cmif_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
    }
}

fn dispatch_applet(
    process: &mut ExceptionProcessContext<'_>,
    session: &AppletSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
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
        return Ok((
            encode_domain_response(
                request.token,
                HorizonIpcResult::CMIF_NOT_SUPPORTED,
                &[],
                &[],
                &[],
            )?,
            None,
        ));
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
                return applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID);
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
                    return applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID);
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
            _ => applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
        },
        AppletObject::SelfController => match request.command_id {
            11..=13 => applet_data(request.token, &[]),
            _ => applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
        },
        AppletObject::WindowController => match request.command_id {
            1 => applet_data(request.token, &process.process_id().to_le_bytes()),
            10 => applet_data(request.token, &[]),
            _ => applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
        },
        AppletObject::ApplicationFunctions => match request.command_id {
            40 => applet_data(request.token, &[1]),
            _ => applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID),
        },
        AppletObject::LibraryAppletCreator
        | AppletObject::AudioController
        | AppletObject::DisplayController
        | AppletObject::DebugFunctions => {
            applet_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID)
        }
    }
}

fn dispatch_hid(
    process: &mut ExceptionProcessContext<'_>,
    session: &HidSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // libnx opens IAppletResource with command 0, sends the process ID, and
    // supplies the applet-resource user ID as one u64:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L800-L808
    if request.command_id != 0 {
        return cmif_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID);
    }
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

fn dispatch_hid_applet_resource(
    process: &mut ExceptionProcessContext<'_>,
    resource: &HidAppletResource,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // IAppletResource command 0 returns the 0x40000-byte HID shared-memory
    // object as a copied handle; libnx maps it read-only immediately:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L47-L65
    if request.command_id != 0 {
        return cmif_error(request.token, HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID);
    }
    let handle = process
        .handles_mut()
        .insert(resource.shared_memory())
        .map_err(|_| IpcWireError::ResourceExhausted)?;
    log::debug!("hid returned shared-memory handle {handle:#x}");
    semantic_success(request.token, false, &[], &[handle], &[], None)
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

fn request_u64(data: &[u8], offset: usize) -> Option<u64> {
    data.get(offset..offset + 8)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
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
    fn response_encodes_the_typed_horizon_result_without_translation() {
        let response =
            encode_response(0x33, HorizonIpcResult::SM_NOT_REGISTERED, &[], None).unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());

        assert_eq!(word(16), 0x4f43_4653);
        assert_eq!(word(24), 0xe15);
        assert_eq!(word(28), 0x33);
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
