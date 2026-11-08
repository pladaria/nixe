//! BCAT synchronization follows Eden's NullBcatBackend, without network delivery.
//! https://switchbrew.org/wiki/BCAT_services
use super::prelude::*;
use crate::bcat::{BcatObject, BcatSession, DeliveryDirectories, DeliveryProgress};
use md5::{Digest, Md5};
use nixe_loader_storage::{FileStorage, Storage};
use std::sync::Arc;
use std::sync::Mutex;

fn storage_error(operation: &str, error: impl std::fmt::Display) -> IpcWireError {
    IpcWireError::Bcat(format!("{operation}: {error}").into())
}
fn encoded_name(entry: &std::fs::DirEntry, file: bool) -> Result<[u8; 0x20], IpcWireError> {
    let entry_name = entry.file_name();
    let text = entry_name
        .to_str()
        .ok_or_else(|| storage_error("cache entry name", "invalid UTF-8"))?;
    let mut result = [0; 0x20];
    if text.len() >= result.len() {
        return Err(storage_error("cache entry name", text));
    }
    result[..text.len()].copy_from_slice(text.as_bytes());
    if name(&result, file).is_none() {
        return Err(storage_error("cache entry name", text));
    }
    Ok(result)
}
fn digest(file: &FileStorage) -> Result<[u8; 16], IpcWireError> {
    // Eden uses MD5 to identify delivery files with the same name.
    let size = file.len().map_err(|e| storage_error("file size", e))?;
    let mut hash = Md5::new();
    let mut buffer = [0; 0x10000];
    let mut offset = 0;
    while offset < size {
        let count = (size - offset).min(buffer.len() as u64) as usize;
        file.read_at(offset, &mut buffer[..count])
            .map_err(|e| storage_error("file digest", e))?;
        hash.update(&buffer[..count]);
        offset += count as u64;
    }
    Ok(hash.finalize().into())
}

fn name(bytes: &[u8], file: bool) -> Option<&str> {
    if bytes.len() != 0x20
        || bytes[0x1f] != 0
        || bytes.iter().all(|b| *b == 0)
        || bytes.iter().any(|b| {
            !b.is_ascii_alphanumeric()
                && !matches!(*b, 0 | b'_')
                && *b != if file { b'.' } else { b'-' }
        })
    {
        return None;
    }
    std::str::from_utf8(&bytes[..bytes.iter().position(|b| *b == 0)?]).ok()
}

fn data_response(
    session: &BcatSession,
    token: u32,
    data: &[u8],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    semantic_success(token, session.domain.is_domain(), data, &[], &[], None)
}

fn response(
    session: &BcatSession,
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((
        if session.domain.is_domain() {
            encode_domain_response(token, result, &[], &[], &[])?
        } else {
            encode_response(token, result, &[], None)?
        },
        None,
    ))
}
fn return_interface(
    process: &mut Request,
    session: &BcatSession,
    token: u32,
    object: BcatObject,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if session.domain.is_domain() {
        let Some(id) = session.domain.insert_object(object) else {
            return response(session, token, HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
        };
        semantic_success(token, true, &[], &[], &[id], None)
    } else {
        let handle = process
            .return_service(HorizonIpcObject::Bcat(BcatSession::from_object(object)))
            .map_err(IpcWireError::Handle)?;
        semantic_success(token, false, &[], &[], &[], Some(handle))
    }
}
pub(in crate::ipc_wire) async fn dispatch_bcat(
    process: &mut Request,
    session: &BcatSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return response(
                session,
                request.token,
                if session.domain.close_object(*object_id) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::CMIF_TARGET_NOT_FOUND
                },
            );
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            session.target(Some(*object_id))
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain BCAT request omitted its domain header",
            ));
        }
        None => session.target(None),
    };
    let Some(target) = target else {
        return response(
            session,
            request.token,
            HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
        );
    };
    match target {
        BcatObject::Creator(root) => {
            if !matches!(request.command_id, 0 | 1) {
                return unsupported_service_command("bcat:u", request.command_id);
            }
            if request_u64(request.data, 0).is_none()
                || hipc.pid.is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            if request.command_id == 1 {
                if let Err(error) = std::fs::create_dir_all(root.as_ref()) {
                    log::error!("cannot open BCAT directory {}: {error}", root.display());
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_FAILED_OPEN_ENTITY,
                    );
                }
                return return_interface(
                    process,
                    session,
                    request.token,
                    BcatObject::Storage {
                        directories: Arc::new(Mutex::new(DeliveryDirectories {
                            names: std::fs::read_dir(root.as_ref())
                                .map_err(|e| storage_error("enumerating directories", e))?
                                .filter_map(|entry| match entry {
                                    Ok(entry) if entry.path().is_dir() => {
                                        Some(encoded_name(&entry, false))
                                    }
                                    Ok(_) => None,
                                    Err(e) => Some(Err(storage_error("directory entry", e))),
                                })
                                .collect::<Result<_, _>>()?,
                            next_read_index: 0,
                        })),
                        root,
                    },
                );
            }
            let progress = std::array::from_fn(|_| DeliveryProgress::new(process.kernel()));
            return_interface(
                process,
                session,
                request.token,
                BcatObject::Service(Arc::new(progress)),
            )
        }
        BcatObject::Storage { root, directories } => {
            if !request.has_payload_size(0) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            match request.command_id {
                0 | 1 if !has_ipc_descriptors(hipc) => {
                    let object = if request.command_id == 0 {
                        BcatObject::File {
                            root,
                            current: Arc::new(Mutex::new(None)),
                        }
                    } else {
                        BcatObject::Directory {
                            root,
                            current: Arc::new(Mutex::new(None)),
                        }
                    };
                    return_interface(process, session, request.token, object)
                }
                10 => {
                    let output = one_receive_buffer(process, hipc)?;
                    let (mut entries, count) = {
                        let mut directories = directories.lock().unwrap();
                        let count = (output.size / 0x20)
                            .min((directories.names.len() - directories.next_read_index) as u64)
                            as usize;
                        let start = directories.next_read_index;
                        let bytes: Vec<u8> = directories.names[start..start + count]
                            .iter()
                            .flatten()
                            .copied()
                            .collect();
                        directories.next_read_index += count;
                        (bytes, count as u32)
                    };
                    process
                        .reserve_payload(entries.len().saturating_mul(2))
                        .await?;
                    process.write(output, 0, &mut entries).await?;
                    data_response(session, request.token, &count.to_le_bytes())
                }
                command => unsupported_service_command("IDeliveryCacheStorageService", command),
            }
        }
        BcatObject::File { root, current } => {
            if request.command_id == 0 {
                if !request.has_payload_size(0x40) || has_ipc_descriptors(hipc) {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    );
                }
                let (Some(dir), Some(file)) = (
                    name(&request.data[..0x20], false),
                    name(&request.data[0x20..0x40], true),
                ) else {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_INVALID_ARGUMENT,
                    );
                };
                let mut current = current.lock().unwrap();
                if current.is_some() {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_ENTITY_ALREADY_OPEN,
                    );
                }
                let path = root.join(dir).join(file);
                let file = if path.is_file() {
                    FileStorage::open(path).ok()
                } else {
                    None
                };
                let Some(file) = file else {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_FAILED_OPEN_ENTITY,
                    );
                };
                *current = Some(Arc::new(file));
                return response(session, request.token, HorizonIpcResult::SUCCESS);
            }
            let Some(file) = current.lock().unwrap().clone() else {
                return response(session, request.token, HorizonIpcResult::BCAT_NO_OPEN_ENTRY);
            };
            let size = file.len().map_err(|e| storage_error("file size", e))?;
            match request.command_id {
                1 => {
                    if !request.has_payload_size(8) {
                        return response(
                            session,
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let output = one_receive_buffer(process, hipc)?;
                    let offset = request_u64(request.data, 0).unwrap();
                    let count = output.size.min(size.saturating_sub(offset));
                    let count = usize::try_from(count).map_err(|_| {
                        IpcWireError::HostResourceExhausted("BCAT file buffer size")
                    })?;
                    process.reserve_payload(count.saturating_mul(2)).await?;
                    let mut data = vec![0; count];
                    if count != 0 {
                        file.read_at(offset, &mut data)
                            .map_err(|e| storage_error("file contents", e))?;
                    }
                    process.write(output, 0, &mut data).await?;
                    data_response(session, request.token, &(count as u64).to_le_bytes())
                }
                2 if request.has_payload_size(0) && !has_ipc_descriptors(hipc) => {
                    data_response(session, request.token, &size.to_le_bytes())
                }
                3 if request.has_payload_size(0) && !has_ipc_descriptors(hipc) => {
                    data_response(session, request.token, &digest(&file)?)
                }
                command => unsupported_service_command("IDeliveryCacheFileService", command),
            }
        }
        BcatObject::Directory { root, current } => {
            if request.command_id == 0 {
                if !request.has_payload_size(0x20) || has_ipc_descriptors(hipc) {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    );
                }
                let Some(dir) = name(&request.data[..0x20], false) else {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_INVALID_ARGUMENT,
                    );
                };
                let mut current = current.lock().unwrap();
                if current.is_some() {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_ENTITY_ALREADY_OPEN,
                    );
                }
                let dir = root.join(dir);
                if !dir.is_dir() {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_FAILED_OPEN_ENTITY,
                    );
                }
                *current = Some(dir);
                return response(session, request.token, HorizonIpcResult::SUCCESS);
            }
            let Some(dir) = current.lock().unwrap().clone() else {
                return response(session, request.token, HorizonIpcResult::BCAT_NO_OPEN_ENTRY);
            };
            if !request.has_payload_size(0) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            let files = std::fs::read_dir(dir)
                .map_err(|e| storage_error("directory listing", e))?
                .filter_map(|entry| match entry {
                    Ok(entry) if entry.path().is_file() => Some(Ok(entry)),
                    Ok(_) => None,
                    Err(e) => Some(Err(storage_error("file entry", e))),
                })
                .collect::<Result<Vec<_>, _>>()?;
            match request.command_id {
                1 => {
                    let output = one_receive_buffer(process, hipc)?;
                    let count = (output.size / 0x38).min(files.len() as u64) as usize;
                    process.reserve_payload(count.saturating_mul(0x70)).await?;
                    let mut bytes = Vec::with_capacity(count * 0x38);
                    for entry in &files[..count] {
                        let file = FileStorage::open(entry.path())
                            .map_err(|e| storage_error("opening directory file", e))?;
                        bytes.extend_from_slice(&encoded_name(entry, true)?);
                        bytes.extend_from_slice(
                            &file
                                .len()
                                .map_err(|e| storage_error("directory file size", e))?
                                .to_le_bytes(),
                        );
                        bytes.extend_from_slice(&digest(&file)?);
                    }
                    process.write(output, 0, &mut bytes).await?;
                    data_response(session, request.token, &(count as u32).to_le_bytes())
                }
                2 if !has_ipc_descriptors(hipc) => {
                    data_response(session, request.token, &(files.len() as u32).to_le_bytes())
                }
                command => unsupported_service_command("IDeliveryCacheDirectoryService", command),
            }
        }
        BcatObject::Service(progress) => match request.command_id {
            10100 | 10101 => {
                let expected = if request.command_id == 10100 { 0 } else { 0x20 };
                if !request.has_payload_size(expected) || has_ipc_descriptors(hipc) {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    );
                }
                let index = usize::from(request.command_id == 10101);
                progress[index].finish_offline_sync();
                return_interface(
                    process,
                    session,
                    request.token,
                    BcatObject::Progress(progress[index].clone()),
                )
            }
            30100 => {
                if !request.has_payload_size(8)
                    || hipc.pid.is_some()
                    || !hipc.copy_handles.is_empty()
                    || !hipc.move_handles.is_empty()
                    || !hipc.send_buffers.is_empty()
                    || !hipc.receive_buffers.is_empty()
                    || !hipc.exchange_buffers.is_empty()
                    || hipc.receive_statics != ReceiveStatics::None
                {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    );
                }
                let [buffer] = hipc.send_statics.as_slice() else {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    );
                };
                if request_u64(request.data, 0) == Some(0) || buffer.size > 0x40 {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::BCAT_INVALID_ARGUMENT,
                    );
                }
                let mut passphrase = vec![0; usize::from(buffer.size)];
                if !passphrase.is_empty() {
                    process
                        .read(&process.send_statics()[0], 0, &mut passphrase)
                        .await?;
                }
                // The offline backend has no network transport or passphrase consumer.
                response(session, request.token, HorizonIpcResult::SUCCESS)
            }
            30300 if request.has_payload_size(0) && !has_ipc_descriptors(hipc) => {
                response(session, request.token, HorizonIpcResult::SUCCESS)
            }
            command => unsupported_service_command("IBcatService", command),
        },
        BcatObject::Progress(progress) => {
            if !request.has_payload_size(0) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            match request.command_id {
                0 if !has_ipc_descriptors(hipc) => {
                    let handle = process
                        .return_handle(progress.event.1.clone())
                        .map_err(IpcWireError::Handle)?;
                    semantic_success(
                        request.token,
                        session.domain.is_domain(),
                        &[],
                        &[handle],
                        &[],
                        None,
                    )
                }
                1 => {
                    if hipc.pid.is_some()
                        || !hipc.copy_handles.is_empty()
                        || !hipc.move_handles.is_empty()
                        || !hipc.send_statics.is_empty()
                        || !hipc.send_buffers.is_empty()
                        || !hipc.receive_buffers.is_empty()
                        || !hipc.exchange_buffers.is_empty()
                        || process.receive_statics().len() != 1
                        || process.receive_statics()[0].size < 0x200
                    {
                        return response(
                            session,
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let mut state = progress.state.lock().unwrap().to_vec();
                    process
                        .write(&process.receive_statics()[0], 0, &mut state)
                        .await?;
                    response(session, request.token, HorizonIpcResult::SUCCESS)
                }
                command => unsupported_service_command("IDeliveryCacheProgressService", command),
            }
        }
    }
}
