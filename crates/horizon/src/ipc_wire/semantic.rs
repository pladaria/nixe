//! Wire bridge for filesystem and add-on-content semantic IPC objects.

use super::*;

enum SemanticTarget {
    Root,
    Object(SemanticIpcObject),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FileSystemProxyCommand {
    SetCurrentProcess,
    OpenDataFileSystemByCurrentProcess,
    OpenSdCardFileSystem,
    OpenDataStorageByCurrentProcess,
    GetGlobalAccessLogMode,
}

impl FileSystemProxyCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            1 => Some(Self::SetCurrentProcess),
            2 => Some(Self::OpenDataFileSystemByCurrentProcess),
            18 => Some(Self::OpenSdCardFileSystem),
            200 => Some(Self::OpenDataStorageByCurrentProcess),
            1005 => Some(Self::GetGlobalAccessLogMode),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AddOnContentCommand {
    CountLegacy,
    ListLegacy,
    Count,
    List,
    PrepareLegacy,
    Prepare,
    GetListChangedEvent,
}

impl AddOnContentCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CountLegacy),
            1 => Some(Self::ListLegacy),
            2 => Some(Self::Count),
            3 => Some(Self::List),
            6 => Some(Self::PrepareLegacy),
            7 => Some(Self::Prepare),
            8 => Some(Self::GetListChangedEvent),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FileSystemCommand {
    CreateFile,
    CreateDirectory,
    OpenFile,
    OpenDirectory,
}

impl FileSystemCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CreateFile),
            2 => Some(Self::CreateDirectory),
            8 => Some(Self::OpenFile),
            9 => Some(Self::OpenDirectory),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FileCommand {
    Read,
    Write,
    Flush,
    SetSize,
    GetSize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StorageCommand {
    Read,
    GetSize,
}

impl StorageCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Read),
            4 => Some(Self::GetSize),
            _ => None,
        }
    }
}

impl FileCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Read),
            1 => Some(Self::Write),
            2 => Some(Self::Flush),
            3 => Some(Self::SetSize),
            4 => Some(Self::GetSize),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectoryCommand {
    Read,
    GetEntryCount,
}

impl DirectoryCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Read),
            1 => Some(Self::GetEntryCount),
            _ => None,
        }
    }
}

pub(super) fn dispatch_semantic_service(
    process: &mut ExceptionProcessContext<'_>,
    session: &IpcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    file_system_access_log_mode: FileSystemAccessLogMode,
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
        file_system_access_log_mode,
    )
}

pub(super) fn dispatch_plain_semantic_object(
    process: &mut ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
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
        FileSystemAccessLogMode::None,
    )
}

fn dispatch_semantic_command(
    process: &mut ExceptionProcessContext<'_>,
    service: IpcService,
    session: Option<&IpcSession>,
    target: SemanticTarget,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    file_system_access_log_mode: FileSystemAccessLogMode,
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
                file_system_access_log_mode,
            ),
            SemanticTarget::Object(object) => {
                IpcDispatcher::dispatch_semantic_object(mounts, handles, object, decoded)
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

pub(super) fn semantic_object_name(object: &SemanticIpcObject) -> &'static str {
    match object {
        SemanticIpcObject::ReadOnlyFileSystem(_) => "IFileSystem(read-only)",
        SemanticIpcObject::ReadOnlyStorage(_) => "IStorage(read-only)",
        SemanticIpcObject::HostDirectoryFileSystem(_) => "IFileSystem(sd-card)",
        SemanticIpcObject::ReadOnlyFile(_) => "IFile(read-only)",
        SemanticIpcObject::HostFile(_) => "IFile(sd-card)",
        SemanticIpcObject::ReadOnlyDirectory(_) => "IDirectory",
    }
}

fn decode_root_request(
    service: IpcService,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    match service {
        IpcService::FileSystem => {
            let Some(command) = FileSystemProxyCommand::decode(request.command_id) else {
                return Ok(None);
            };
            match command {
                // libnx sends the current PID and a zero placeholder:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L75-L82
                FileSystemProxyCommand::SetCurrentProcess => {
                    if hipc.pid.is_none() || request.data.len() < 8 {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::SetCurrentProcess))
                }
                FileSystemProxyCommand::OpenDataFileSystemByCurrentProcess => {
                    Ok(Some(IpcRequest::OpenPrimaryFileSystem))
                }
                FileSystemProxyCommand::OpenSdCardFileSystem => {
                    Ok(Some(IpcRequest::OpenSdCardFileSystem))
                }
                // IFileSystemProxy::OpenDataStorageByCurrentProcess returns
                // one IStorage child object and carries no input payload:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L470-L472
                FileSystemProxyCommand::OpenDataStorageByCurrentProcess => {
                    if has_ipc_descriptors(hipc) {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::OpenPrimaryStorage))
                }
                // ABI and scalar output type:
                // https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/include/stratosphere/fssrv/sf/fssrv_sf_i_file_system_proxy.hpp#L134-L136
                FileSystemProxyCommand::GetGlobalAccessLogMode => {
                    if has_ipc_descriptors(hipc) {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::GetGlobalAccessLogMode))
                }
            }
        }
        IpcService::AddOnContent => {
            let Some(command) = AddOnContentCommand::decode(request.command_id) else {
                return Ok(None);
            };
            // Versioned command IDs follow the documented aoc:u ABI:
            // https://switchbrew.org/w/index.php?title=NS_services&oldid=14328#aoc:u
            match command {
                AddOnContentCommand::CountLegacy => {
                    Ok(Some(IpcRequest::GetIndexedAddOnContentCount))
                }
                AddOnContentCommand::Count => {
                    if hipc.pid.is_none() {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::GetIndexedAddOnContentCount))
                }
                command @ (AddOnContentCommand::ListLegacy | AddOnContentCommand::List) => {
                    if command == AddOnContentCommand::List && hipc.pid.is_none() {
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
                command @ (AddOnContentCommand::PrepareLegacy | AddOnContentCommand::Prepare) => {
                    if command == AddOnContentCommand::Prepare && hipc.pid.is_none() {
                        return Ok(None);
                    }
                    let horizon_index = request_u32(request.data, 0).ok_or(
                        IpcWireError::Malformed("aoc:u prepare request omits its content index"),
                    )?;
                    Ok(Some(IpcRequest::PrepareAddOnContent { horizon_index }))
                }
                AddOnContentCommand::GetListChangedEvent => {
                    Ok(Some(IpcRequest::GetAddOnContentListChangedEvent))
                }
            }
        }
    }
}
fn decode_object_request(
    process: &ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    match object {
        SemanticIpcObject::ReadOnlyFileSystem(_)
        | SemanticIpcObject::HostDirectoryFileSystem(_) => {
            let Some(command) = FileSystemCommand::decode(request.command_id) else {
                return Ok(None);
            };
            match command {
                // IFileSystem::CreateFile/CreateDirectory use the same bounded
                // input-pointer path as open operations:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L816-L840
                FileSystemCommand::CreateFile => {
                    if !matches!(object, SemanticIpcObject::HostDirectoryFileSystem(_)) {
                        return Ok(None);
                    }
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
                FileSystemCommand::CreateDirectory => {
                    if !matches!(object, SemanticIpcObject::HostDirectoryFileSystem(_)) {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::CreateDirectory {
                        path: read_path(process, hipc)?,
                    }))
                }
                // IFileSystem OpenFile/OpenDirectory use one input pointer path
                // and a u32 mode:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L878-L893
                FileSystemCommand::OpenFile => Ok(Some(IpcRequest::OpenFile {
                    path: read_path(process, hipc)?,
                    mode: request_u32(request.data, 0)
                        .ok_or(IpcWireError::Malformed("open-file request omits its mode"))?,
                })),
                FileSystemCommand::OpenDirectory => Ok(Some(IpcRequest::OpenDirectory {
                    path: read_path(process, hipc)?,
                    mode: request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                        "open-directory request omits its mode",
                    ))?,
                })),
            }
        }
        SemanticIpcObject::ReadOnlyFile(_) | SemanticIpcObject::HostFile(_) => {
            let Some(command) = FileCommand::decode(request.command_id) else {
                return Ok(None);
            };
            match command {
                // IFile::Read input layout and map-alias output buffer:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L980-L994
                FileCommand::Read => {
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
                FileCommand::Write => {
                    if !matches!(object, SemanticIpcObject::HostFile(_)) {
                        return Ok(None);
                    }
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
                FileCommand::Flush => {
                    if !matches!(object, SemanticIpcObject::HostFile(_)) {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::FlushFile))
                }
                FileCommand::SetSize => {
                    if !matches!(object, SemanticIpcObject::HostFile(_)) {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::SetFileSize {
                        size: request_u64(request.data, 0).ok_or(IpcWireError::Malformed(
                            "set-file-size request omits its size",
                        ))?,
                    }))
                }
                FileCommand::GetSize => Ok(Some(IpcRequest::GetFileSize)),
            }
        }
        SemanticIpcObject::ReadOnlyStorage(_) => {
            let Some(command) = StorageCommand::decode(request.command_id) else {
                return Ok(None);
            };
            match command {
                // IStorage::Read carries offset/size and exactly one
                // map-alias output buffer. Unlike IFile::Read, its response
                // contains no byte-count scalar:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L975-L983
                StorageCommand::Read => {
                    let offset = request_u64(request.data, 0).ok_or(IpcWireError::Malformed(
                        "storage read request omits its offset",
                    ))?;
                    let requested = request_u64(request.data, 8)
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or(IpcWireError::Malformed(
                            "storage read request size is out of range",
                        ))?;
                    if requested > MAX_IPC_READ_BYTES {
                        return Err(IpcWireError::UnsupportedService(
                            UnsupportedServiceOperation::CommandVariant {
                                service: "IStorage",
                                command_id: request.command_id,
                                detail: "requested read exceeds Nixe's implemented IPC bound",
                            },
                        ));
                    }
                    let descriptor = one_receive_buffer(hipc)?;
                    let capacity = usize::try_from(descriptor.size).map_err(|_| {
                        IpcWireError::Malformed("storage output buffer is too large")
                    })?;
                    if requested > capacity {
                        return Err(IpcWireError::Malformed(
                            "storage read size exceeds its output buffer",
                        ));
                    }
                    Ok(Some(IpcRequest::ReadStorage {
                        offset,
                        size: requested,
                    }))
                }
                // IStorage::GetSize returns one signed 64-bit size scalar:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L1005-L1007
                StorageCommand::GetSize => {
                    if has_ipc_descriptors(hipc) {
                        return Ok(None);
                    }
                    Ok(Some(IpcRequest::GetStorageSize))
                }
            }
        }
        SemanticIpcObject::ReadOnlyDirectory(_) => {
            let Some(command) = DirectoryCommand::decode(request.command_id) else {
                return Ok(None);
            };
            match command {
                // IDirectory::Read returns fixed 0x310-byte FsDirectoryEntry
                // records through one map-alias output buffer:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L1043-L1051
                DirectoryCommand::Read => {
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
                DirectoryCommand::GetEntryCount => Ok(Some(IpcRequest::GetDirectoryEntryCount)),
            }
        }
    }
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
        IpcResponse::FileSystemAccessLogMode(mode) => semantic_success(
            request.token,
            is_domain,
            &mode.raw().to_le_bytes(),
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
                let Some(HorizonIpcObject::SemanticObject(object)) =
                    object.downcast_ref::<HorizonIpcObject>().cloned()
                else {
                    return Err(IpcWireError::Internal(
                        "semantic dispatch returned a non-semantic child handle",
                    ));
                };
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
        IpcResponse::StorageData(data) => {
            let descriptor = one_receive_buffer(hipc)?;
            write_descriptor_bytes(process, descriptor, &data)?;
            semantic_success(request.token, is_domain, &[], &[], &[], None)
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

pub(super) fn semantic_success(
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

pub(super) fn semantic_error(
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

pub(super) fn one_receive_buffer(hipc: &HipcRequest<'_>) -> Result<BufferDescriptor, IpcWireError> {
    match hipc.receive_buffers.as_slice() {
        [descriptor] if descriptor.size > 0 && descriptor.mode != BufferMode::Invalid => {
            Ok(*descriptor)
        }
        _ => Err(IpcWireError::Malformed(
            "service command requires exactly one output buffer",
        )),
    }
}

pub(super) fn one_send_buffer(hipc: &HipcRequest<'_>) -> Result<BufferDescriptor, IpcWireError> {
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

pub(super) fn one_auto_select_input(hipc: &HipcRequest<'_>) -> Result<(u64, usize), IpcWireError> {
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

pub(super) fn one_auto_select_output(hipc: &HipcRequest<'_>) -> Result<(u64, usize), IpcWireError> {
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

#[cfg(test)]
mod tests {
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
    fn filesystem_proxy_decodes_global_access_log_mode_as_a_scalar_query() {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4);
        put_u32(&mut command, 4, 8);
        put_u32(&mut command, 16, 0x4943_4653);
        put_u32(&mut command, 24, 1005);
        let hipc = HipcRequest::decode(&command).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(IpcService::FileSystem, &request, &hipc).unwrap(),
            Some(IpcRequest::GetGlobalAccessLogMode)
        );
    }

    #[test]
    fn filesystem_proxy_decodes_current_process_data_storage_open() {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4);
        put_u32(&mut command, 4, 8);
        put_u32(&mut command, 16, 0x4943_4653);
        put_u32(&mut command, 24, 200);
        let hipc = HipcRequest::decode(&command).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(IpcService::FileSystem, &request, &hipc).unwrap(),
            Some(IpcRequest::OpenPrimaryStorage)
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

    #[test]
    fn auto_select_input_decodes_exactly_one_active_descriptor_side() {
        assert_eq!(auto_select_input((0, 0), (0x2000, 8)), Ok((0x2000, 8)));
        assert_eq!(auto_select_input((0x3000, 4), (0, 0)), Ok((0x3000, 4)));
        assert_eq!(
            auto_select_input((0x1000, 4), (0x2000, 4)),
            Err(IpcWireError::Malformed(
                "auto-select input has both descriptor sides active"
            ))
        );
    }

    #[test]
    fn descriptor_presence_includes_receive_statics() {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4);
        put_u32(&mut command, 4, 3 << 10);
        put_u32(&mut command, 8, 0x2000);
        put_u32(&mut command, 12, 4 << 16);
        let hipc = HipcRequest::decode(&command).unwrap();

        assert!(has_ipc_descriptors(&hipc));
    }
}
