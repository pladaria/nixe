//! Shop-access factories follow NIM_ECA and IShopServiceAccessServer in Eden.
//! https://switchbrew.org/wiki/NIM_services#nim:eca
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/nim/nim.cpp

use super::prelude::*;
use crate::nim::{NimObject, NimSession};

fn response(
    session: &NimSession,
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

pub(in crate::ipc_wire) fn dispatch_nim(
    process: &mut Request,
    session: &NimSession,
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
                "domain NIM request omitted its domain header",
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
    let child = match (target, request.command_id) {
        (NimObject::Factory, 0 | 5) => NimObject::Server,
        (NimObject::Server, 0) => NimObject::Accessor,
        (NimObject::Factory, 4) => {
            if !request.has_payload_size(8) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            return semantic_success(
                request.token,
                session.domain.is_domain(),
                &[0],
                &[],
                &[],
                None,
            );
        }
        _ => return unsupported_service_command(target.name(), request.command_id),
    };
    if session.domain.is_domain() {
        let Some(id) = session.domain.insert_object(child) else {
            return response(
                session,
                request.token,
                HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
            );
        };
        semantic_success(request.token, true, &[], &[], &[id], None)
    } else {
        let handle = process
            .return_service(HorizonIpcObject::Nim(NimSession::from_object(child)))
            .map_err(IpcWireError::Handle)?;
        semantic_success(request.token, false, &[], &[], &[], Some(handle))
    }
}
