//! Encoding for responses produced by the typed filesystem and add-on backends.

use nixe_runtime::ExceptionProcessContext;

use super::super::IpcWireError;
use super::super::buffer::one_receive_buffer;
use super::super::io::{cmif_error, encode_domain_response, write_descriptor_bytes};
use super::super::message::{CmifRequest, CmifResponse, HipcRequest};
use super::fsp::abi::{FS_DIRECTORY_ENTRY_FILE, FS_DIRECTORY_ENTRY_SIZE, FS_MAX_PATH};
use crate::{
    DirectoryEntryKind, HorizonIpcObject, HorizonIpcResult, IpcResponse, IpcResultCode, IpcService,
    IpcSession, SemanticIpcObject,
};

const STORAGE_READ_BUFFER_BYTES: usize = 4 * 1024 * 1024;

pub(super) fn encode_semantic_response(
    process: &mut ExceptionProcessContext<'_>,
    domain_session: Option<&IpcSession>,
    target_object: Option<&SemanticIpcObject>,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    response: IpcResponse,
    async_reply: Option<&crate::host_work::AsyncReply<'_>>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let is_domain = domain_session.is_some_and(IpcSession::is_domain);
    match response {
        IpcResponse::None => semantic_success(request.token, is_domain, &[], &[], &[], None),
        IpcResponse::FileSystemAttribute {
            name_length_max,
            path_length_max,
        } => {
            let data = file_system_attribute_data(name_length_max, path_length_max);
            semantic_success(request.token, is_domain, &data, &[], &[], None)
        }
        IpcResponse::EntryType(kind) => {
            // FsDirectoryEntryType: Directory=0, File=1.
            // https://github.com/switchbrew/libnx/blob/master/nx/include/switch/services/fs.h
            let value: u32 = match kind {
                DirectoryEntryKind::Directory => 0,
                DirectoryEntryKind::File => 1,
            };
            semantic_success(
                request.token,
                is_domain,
                &value.to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        IpcResponse::Size(size) => semantic_success(
            request.token,
            is_domain,
            &size.to_le_bytes(),
            &[],
            &[],
            None,
        ),
        IpcResponse::AccessLogProgramIndex {
            version,
            program_index,
        } => {
            let mut data = [0; 8];
            data[..4].copy_from_slice(&version.to_le_bytes());
            data[4..].copy_from_slice(&program_index.to_le_bytes());
            semantic_success(request.token, is_domain, &data, &[], &[], None)
        }
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
                encode_semantic_child(process, domain_session, request.token, object)
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
            let (storage, file_read) = match target_object {
                Some(SemanticIpcObject::ReadOnlyStorage(storage)) => {
                    (storage.storage().clone(), false)
                }
                Some(SemanticIpcObject::ReadOnlyFile(file)) => (file.storage().clone(), true),
                _ => {
                    return Err(IpcWireError::Internal(
                        "storage read response lacks a retained source",
                    ));
                }
            };
            let async_reply = async_reply.ok_or(IpcWireError::Internal(
                "storage read lacks asynchronous reply ownership",
            ))?;
            if size as u64 > descriptor.size {
                return Err(IpcWireError::Malformed(
                    "storage response exceeds its output descriptor",
                ));
            }
            let destination = if size == 0 {
                None
            } else {
                Some(crate::host_work::retain_output(
                    process,
                    descriptor.address,
                    size as u64,
                )?)
            };
            let count = (size as u64).to_le_bytes();
            let success = semantic_success(
                request.token,
                is_domain,
                if file_read { &count } else { &[] },
                &[],
                &[],
                None,
            )?
            .0;
            let failure = semantic_error(
                request.token,
                domain_session,
                HorizonIpcResult::from_semantic(
                    IpcService::FileSystem,
                    IpcResultCode::STORAGE_FAILURE,
                ),
            )?
            .0;
            async_reply.submit(process, move |cancelled| {
                let _trace = nixe_trace::Span::new("storage.read", offset, size as u64);
                let mut buffer = vec![0; size.min(STORAGE_READ_BUFFER_BYTES)];
                let mut transferred = 0;
                while transferred < size {
                    if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(Vec::new());
                    }
                    let count = (size - transferred).min(buffer.len());
                    if storage
                        .read_at(offset + transferred as u64, &mut buffer[..count])
                        .is_err()
                    {
                        return Ok(failure);
                    }
                    if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(Vec::new());
                    }
                    crate::host_work::write_retained(
                        destination.as_ref().unwrap(),
                        transferred as u64,
                        &buffer[..count],
                    )?;
                    transferred += count;
                }
                Ok(success)
            })?;
            unreachable!("accepted host work always suspends its caller")
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

pub(super) fn file_system_attribute_data(name_length_max: u32, path_length_max: u32) -> [u8; 0xc0] {
    // FsFileSystemAttribute: optional UTF-8 limits start at 0x28;
    // UTF-16-specific limits remain absent for this UTF-8 backend.
    // https://github.com/switchbrew/libnx/blob/master/nx/include/switch/services/fs.h#L302-L332
    let mut data = [0; 0xc0];
    data[..4].fill(1);
    for (offset, value) in [
        (0x28, name_length_max),
        (0x2c, name_length_max),
        (0x30, path_length_max),
        (0x34, path_length_max),
    ] {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    data
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

/// Publish a prepared child using the coordinator's process/domain authority.
pub(super) fn encode_semantic_child(
    process: &mut ExceptionProcessContext<'_>,
    session: Option<&IpcSession>,
    token: u32,
    object: SemanticIpcObject,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if let Some(session) = session.filter(|session| session.is_domain()) {
        let Some(id) = session.insert_object(object) else {
            return semantic_error(
                token,
                Some(session),
                HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
            );
        };
        semantic_success(token, true, &[], &[], &[id], None)
    } else {
        match process
            .handles_mut()
            .insert(HorizonIpcObject::SemanticObject(object))
        {
            Ok(handle) => semantic_success(token, false, &[], &[], &[], Some(handle)),
            Err(_) => semantic_error(
                token,
                session,
                HorizonIpcResult::from_semantic(
                    IpcService::FileSystem,
                    IpcResultCode::RESOURCE_LIMIT,
                ),
            ),
        }
    }
}
