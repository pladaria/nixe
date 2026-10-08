//! Shared session orchestration for content-backed Horizon services.
//!
//! `fsp-srv` and `aoc:u` expose different root interfaces, but both can
//! return the same filesystem child objects. Domain lifetime handling and
//! semantic result translation therefore live here, while each service owns
//! its wire command decoder.

use nixe_runtime::ExceptionProcessContext;

use crate::ipc_wire::io::encode_domain_response;
use crate::ipc_wire::message::{CmifRequest, DomainRequest, HipcRequest};
use crate::ipc_wire::{IpcWireError, unsupported_service_command};
use crate::{
    FileSystemAccessLogMode, HorizonIpcResult, IpcDispatcher, IpcRequest, IpcResultCode,
    IpcService, IpcSession, SemanticIpcObject,
};

use super::fsp;
use super::response::{encode_semantic_response, semantic_error};
use super::semantic_service_name;

enum Target {
    Root,
    Object(SemanticIpcObject),
}

pub(in crate::ipc_wire) fn dispatch_service(
    process: &mut ExceptionProcessContext<'_>,
    session: &IpcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    file_system_access_log_mode: FileSystemAccessLogMode,
    save_data: Option<&crate::SaveDataSystem>,
    async_reply: &crate::host_work::AsyncReply<'_>,
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
                Target::Root
            } else {
                let Some(object) = session.object(*object_id) else {
                    return semantic_error(
                        request.token,
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

    if matches!(target, Target::Root)
        && session.service() == IpcService::FileSystem
        && request.command_id == 51
    {
        return fsp::open_save_data(process, session, request, hipc, save_data, async_reply);
    }

    dispatch_command(
        process,
        session.service(),
        Some(session),
        target,
        request,
        hipc,
        file_system_access_log_mode,
        async_reply,
    )
}

pub(in crate::ipc_wire) fn dispatch_plain_object(
    process: &mut ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    async_reply: &crate::host_work::AsyncReply<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    dispatch_command(
        process,
        IpcService::FileSystem,
        None,
        Target::Object(object.clone()),
        request,
        hipc,
        FileSystemAccessLogMode::None,
        async_reply,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_command(
    process: &mut ExceptionProcessContext<'_>,
    service: IpcService,
    session: Option<&IpcSession>,
    target: Target,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    file_system_access_log_mode: FileSystemAccessLogMode,
    async_reply: &crate::host_work::AsyncReply<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let (decoded, name) = match &target {
        Target::Root => (
            decode_root_request(service, &request, hipc)?,
            semantic_service_name(service),
        ),
        Target::Object(object) => (
            fsp::decode_object_request(process, object, &request, hipc)?,
            fsp::object_name(object),
        ),
    };
    let Some(decoded) = decoded else {
        return unsupported_service_command(name, request.command_id);
    };

    if let Target::Object(object @ SemanticIpcObject::HostDirectoryFileSystem(_)) = &target
        && matches!(
            decoded,
            IpcRequest::OpenFile { .. } | IpcRequest::OpenDirectory { .. }
        )
    {
        let mounts = process.mounts().clone();
        let object = object.clone();
        let session = session.cloned();
        let token = request.token;
        async_reply.submit_prepared(
            process,
            move || {
                let _trace = nixe_trace::Span::new("storage.open", 0, 0);
                let mut handles = nixe_runtime::HandleTable::new();
                match IpcDispatcher::dispatch_semantic_object(
                    &mounts,
                    &mut handles,
                    &object,
                    decoded,
                ) {
                    Ok(crate::IpcResponse::Handle(handle)) => {
                        let object = handles.close(handle).map_err(|_| {
                            IpcWireError::Internal("prepared file handle disappeared")
                        })?;
                        match object.downcast_ref::<crate::HorizonIpcObject>() {
                            Some(crate::HorizonIpcObject::SemanticObject(object)) => {
                                Ok(Ok(object.clone()))
                            }
                            _ => Err(IpcWireError::Internal(
                                "prepared file object has an invalid type",
                            )),
                        }
                    }
                    Err(error) => Ok(Err(error)),
                    _ => Err(IpcWireError::Internal(
                        "file open produced an unexpected response",
                    )),
                }
            },
            move |process, result| match result {
                Ok(object) => {
                    super::response::encode_semantic_child(process, session.as_ref(), token, object)
                }
                Err(error) => semantic_error(
                    token,
                    session.as_ref(),
                    HorizonIpcResult::from_semantic(IpcService::FileSystem, error),
                ),
            },
        )?;
        unreachable!("accepted file open suspends its caller")
    }

    // Child file operations and non-handle-producing directory operations need
    // no process handle-table authority while host I/O is in progress.
    if let Target::Object(
        object @ (SemanticIpcObject::HostFile(_) | SemanticIpcObject::HostDirectoryFileSystem(_)),
    ) = &target
        && !matches!(
            decoded,
            IpcRequest::OpenFile { .. } | IpcRequest::OpenDirectory { .. }
        )
    {
        let output = if matches!(decoded, IpcRequest::ReadFile { .. }) {
            let descriptor = crate::ipc_wire::buffer::one_receive_buffer(hipc)?;
            if descriptor.size == 0 {
                None
            } else {
                Some(crate::host_work::retain_output(
                    process,
                    descriptor.address,
                    descriptor.size,
                )?)
            }
        } else {
            None
        };
        let mounts = process.mounts().clone();
        let object = object.clone();
        let token = request.token;
        let is_domain = session.is_some_and(IpcSession::is_domain);
        let command_id = request.command_id;
        async_reply.submit(process, move |cancelled| {
            let _trace = nixe_trace::Span::new("storage.operation", 0, 0);
            let mut handles = nixe_runtime::HandleTable::new();
            let response =
                IpcDispatcher::dispatch_semantic_object(&mounts, &mut handles, &object, decoded);
            let mut data = Vec::new();
            let result = match response {
                Ok(crate::IpcResponse::None) => HorizonIpcResult::SUCCESS,
                Ok(crate::IpcResponse::Size(size)) => {
                    data.extend_from_slice(&size.to_le_bytes());
                    HorizonIpcResult::SUCCESS
                }
                Ok(crate::IpcResponse::EntryType(kind)) => {
                    let kind: u32 = match kind {
                        crate::DirectoryEntryKind::Directory => 0,
                        crate::DirectoryEntryKind::File => 1,
                    };
                    data.extend_from_slice(&kind.to_le_bytes());
                    HorizonIpcResult::SUCCESS
                }
                Ok(crate::IpcResponse::FileSystemAttribute {
                    name_length_max,
                    path_length_max,
                }) => {
                    data.extend_from_slice(&super::response::file_system_attribute_data(
                        name_length_max,
                        path_length_max,
                    ));
                    HorizonIpcResult::SUCCESS
                }
                Ok(crate::IpcResponse::Data(bytes)) => {
                    if !cancelled.load(std::sync::atomic::Ordering::Acquire) && !bytes.is_empty() {
                        crate::host_work::write_retained(
                            output
                                .as_ref()
                                .ok_or(IpcWireError::Internal("file read lacks retained output"))?,
                            0,
                            &bytes,
                        )?;
                    }
                    data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                    HorizonIpcResult::SUCCESS
                }
                Ok(_) => {
                    return Err(IpcWireError::Internal(
                        "host file operation produced an unexpected response",
                    ));
                }
                Err(IpcResultCode::INVALID_COMMAND) => {
                    return unsupported_service_command(name, command_id);
                }
                Err(IpcResultCode::INTERNAL_STATE) => {
                    return Err(IpcWireError::Internal(
                        "content-service IPC entered an invalid internal state",
                    ));
                }
                Err(error) => HorizonIpcResult::from_semantic(IpcService::FileSystem, error),
            };
            Ok(crate::ipc_wire::message::CmifResponse {
                token,
                result: result.raw(),
                data: &data,
                is_domain,
                ..Default::default()
            }
            .encode()?)
        })?;
        unreachable!("host I/O suspends its guest caller")
    }

    let result = {
        let (mounts, handles) = process.mounts_and_handles_mut();
        match &target {
            Target::Root => IpcDispatcher::dispatch_session(
                mounts,
                handles,
                session.expect("a content-service root belongs to a session"),
                decoded,
                file_system_access_log_mode,
            ),
            Target::Object(object) => {
                IpcDispatcher::dispatch_semantic_object(mounts, handles, object, decoded)
            }
        }
    };

    match result {
        Ok(response) => encode_semantic_response(
            process,
            session,
            match &target {
                Target::Root => None,
                Target::Object(object) => Some(object),
            },
            request,
            hipc,
            response,
            Some(async_reply),
        ),
        Err(IpcResultCode::INVALID_COMMAND) => {
            unsupported_service_command(name, request.command_id)
        }
        Err(IpcResultCode::INTERNAL_STATE) => Err(IpcWireError::Internal(
            "content-service IPC entered an invalid internal state",
        )),
        Err(error) => semantic_error(
            request.token,
            session,
            HorizonIpcResult::from_semantic(service, error),
        ),
    }
}

fn decode_root_request(
    service: IpcService,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    match service {
        IpcService::FileSystem => fsp::decode_root_request(request, hipc),
        IpcService::AddOnContent => super::aoc::decode_root_request(request, hipc),
    }
}
