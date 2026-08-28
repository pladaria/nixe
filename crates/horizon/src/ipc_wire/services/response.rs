//! Encoding for responses produced by the typed filesystem and add-on backends.

use nixe_memory::GuestVirtualAddress;
use nixe_runtime::ExceptionProcessContext;

use super::super::IpcWireError;
use super::super::buffer::one_receive_buffer;
use super::super::io::{
    cmif_error, encode_domain_response, validate_writable_ram_range, write_bytes,
    write_descriptor_bytes,
};
use super::super::message::{CmifRequest, CmifResponse, HipcRequest};
use super::fsp::abi::{FS_DIRECTORY_ENTRY_FILE, FS_DIRECTORY_ENTRY_SIZE, FS_MAX_PATH};
use crate::{
    DirectoryEntryKind, HorizonIpcObject, HorizonIpcResult, IpcResponse, IpcResultCode, IpcService,
    IpcSession, ReadOnlyStorage, SemanticIpcObject,
};

const STORAGE_READ_BUFFER_BYTES: usize = 4 * 1024 * 1024;

pub(super) fn encode_semantic_response(
    process: &mut ExceptionProcessContext<'_>,
    domain_session: Option<&IpcSession>,
    target_object: Option<&SemanticIpcObject>,
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
        IpcResponse::StorageRead { offset, size } => {
            let descriptor = one_receive_buffer(hipc)?;
            let Some(SemanticIpcObject::ReadOnlyStorage(storage)) = target_object else {
                return Err(IpcWireError::Internal(
                    "storage read response did not originate from IStorage",
                ));
            };
            match write_storage_read(
                process,
                storage,
                descriptor.address,
                descriptor.size,
                offset,
                size,
            ) {
                Ok(()) => semantic_success(request.token, is_domain, &[], &[], &[], None),
                Err(StorageReadError::Source) => semantic_error(
                    request.token,
                    domain_session,
                    HorizonIpcResult::from_semantic(
                        IpcService::FileSystem,
                        IpcResultCode::STORAGE_FAILURE,
                    ),
                ),
                Err(StorageReadError::Wire(error)) => Err(error),
            }
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

enum StorageReadError {
    Source,
    Wire(IpcWireError),
}

impl From<IpcWireError> for StorageReadError {
    fn from(error: IpcWireError) -> Self {
        Self::Wire(error)
    }
}

fn write_storage_read(
    process: &ExceptionProcessContext<'_>,
    storage: &ReadOnlyStorage,
    guest_address: u64,
    descriptor_size: u64,
    storage_offset: u64,
    size: usize,
) -> Result<(), StorageReadError> {
    if u64::try_from(size)
        .ok()
        .is_none_or(|size| size > descriptor_size)
    {
        return Err(
            IpcWireError::Malformed("storage response exceeds its output descriptor").into(),
        );
    }

    let guest_address = GuestVirtualAddress::new(guest_address);
    validate_writable_ram_range(process, guest_address, size)?;
    if size == 0 {
        return Ok(());
    }

    // IStorage transfers can be hundreds of MiB. Keep host memory bounded and
    // copy each source chunk directly into the already validated guest range.
    let buffer_size = size.min(STORAGE_READ_BUFFER_BYTES);
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(buffer_size)
        .map_err(|_| IpcWireError::HostResourceExhausted("allocating the storage read buffer"))?;
    buffer.resize(buffer_size, 0);

    let mut transferred = 0_usize;
    while transferred < size {
        let chunk_size = (size - transferred).min(buffer.len());
        let source_offset = storage_offset
            .checked_add(
                u64::try_from(transferred)
                    .map_err(|_| IpcWireError::Internal("storage read offset overflows"))?,
            )
            .ok_or(IpcWireError::Internal("storage read offset overflows"))?;
        storage
            .storage()
            .read_at(source_offset, &mut buffer[..chunk_size])
            .map_err(|_| StorageReadError::Source)?;
        let destination = guest_address
            .checked_add(
                u64::try_from(transferred)
                    .map_err(|_| IpcWireError::Internal("storage read address overflows"))?,
            )
            .ok_or(IpcWireError::Internal("storage read address overflows"))?;
        write_bytes(process, destination, &buffer[..chunk_size])?;
        transferred += chunk_size;
    }
    Ok(())
}

pub(in crate::ipc_wire) fn semantic_success(
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
    domain_session: Option<&IpcSession>,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if domain_session.is_some_and(IpcSession::is_domain) {
        Ok((encode_domain_response(token, result, &[], &[], &[])?, None))
    } else {
        cmif_error(token, result)
    }
}
