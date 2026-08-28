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

    dispatch_command(
        process,
        session.service(),
        Some(session),
        target,
        request,
        hipc,
        file_system_access_log_mode,
    )
}

pub(in crate::ipc_wire) fn dispatch_plain_object(
    process: &mut ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    dispatch_command(
        process,
        IpcService::FileSystem,
        None,
        Target::Object(object.clone()),
        request,
        hipc,
        FileSystemAccessLogMode::None,
    )
}

fn dispatch_command(
    process: &mut ExceptionProcessContext<'_>,
    service: IpcService,
    session: Option<&IpcSession>,
    target: Target,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    file_system_access_log_mode: FileSystemAccessLogMode,
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
