//! GameChat stays inactive: interface setup never creates a network session.
//! https://switchbrew.org/wiki/Switch_2%3A_Chat_services

use super::prelude::*;
use crate::chat::{ChatObject, ChatSession};

pub(in crate::ipc_wire) fn dispatch_chat(
    process: &mut Request,
    session: &ChatSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let error = |result| {
        Ok((
            if session.domain.is_domain() {
                encode_domain_response(request.token, result, &[], &[], &[])?
            } else {
                encode_response(request.token, result, &[], None)?
            },
            None,
        ))
    };
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return error(if session.domain.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            });
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.target(Some(*object_id))
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain GameChat request omitted its domain header",
            ));
        }
        None => session.target(None),
    };
    let Some(target) = target else {
        return error(HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
    };
    match (&target, request.command_id) {
        (ChatObject::Service, id @ (0 | 3)) => {
            let valid = if id == 3 {
                request.has_payload_size(8)
                    && hipc.pid.is_some()
                    && !has_ipc_descriptors_other_than_pid(hipc)
            } else {
                request.has_payload_size(0) && !has_ipc_descriptors(hipc)
            };
            if !valid {
                return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let child = if id == 3 {
                ChatObject::VirtualCamera
            } else {
                ChatObject::User(std::sync::Arc::new(std::array::from_fn(|_| {
                    nixe_runtime::EventObject::new(process.kernel())
                })))
            };
            if session.domain.is_domain() {
                let Some(id) = session.domain.insert_object(child) else {
                    return error(HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
                };
                semantic_success(request.token, true, &[], &[], &[id], None)
            } else {
                let handle = process
                    .return_service(HorizonIpcObject::Chat(ChatSession::new(child)))
                    .map_err(IpcWireError::Handle)?;
                semantic_success(request.token, false, &[], &[], &[], Some(handle))
            }
        }
        (ChatObject::Service, 10) => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            // IsChatting: an offline client never joins a room.
            semantic_success(
                request.token,
                session.domain.is_domain(),
                &[0],
                &[],
                &[],
                None,
            )
        }
        (ChatObject::User(events), id @ 1..=4) => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            // Offline chat never becomes active and its room/camera state
            // never changes. Each notification retains its own unsignalled
            // event; repeated acquisitions return the same event identity.
            let handle = process
                .return_handle(events[id as usize - 1].readable())
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
        _ => unsupported_service_command(target.name(), request.command_id),
    }
}
