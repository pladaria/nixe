//! Offline friends service, completion events and empty local notifications.
//! https://switchbrew.org/wiki/Friend_services
use super::prelude::*;
use crate::friends::{FriendObject, FriendSession};

fn response(
    session: &FriendSession,
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let bytes = if session.domain.is_domain() {
        encode_domain_response(token, result, &[], &[], &[])?
    } else {
        encode_response(token, result, &[], None)?
    };
    Ok((bytes, None))
}

pub(in crate::ipc_wire) fn dispatch_friends(
    process: &mut Request,
    session: &FriendSession,
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
                "domain friends request omitted its domain header",
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
        FriendObject::Creator => {
            let bytes = match request.command_id {
                0 => 0,
                1 => 16,
                command => return unsupported_service_command("friend:u", command),
            };
            if !request.has_payload_size(bytes) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            let event = nixe_runtime::EventObject::create_pair(process.kernel());
            let object = if request.command_id == 0 {
                FriendObject::Service { completion: event }
            } else {
                FriendObject::Notification {
                    user: u128::from_le_bytes(request.data[..16].try_into().unwrap()),
                    event,
                }
            };
            if session.domain.is_domain() {
                let Some(id) = session.domain.insert_object(object) else {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                    );
                };
                semantic_success(request.token, true, &[], &[], &[id], None)
            } else {
                let handle = process
                    .return_service(HorizonIpcObject::Friends(FriendSession::from_object(
                        object,
                    )))
                    .map_err(IpcWireError::Handle)?;
                semantic_success(request.token, false, &[], &[], &[], Some(handle))
            }
        }
        FriendObject::Service { completion } => {
            if request.command_id == 10101 {
                // GetFriendList: offset, UID, SizedFriendFilter, process ID,
                // and one output map-alias array of FriendImpl entries.
                // https://reswitched.github.io/SwIPC/ifaces.html#nn::friends::detail::ipc::IFriendService
                if !request.has_payload_size(0x2c)
                    || hipc.pid.is_none()
                    || !hipc.copy_handles.is_empty()
                    || !hipc.move_handles.is_empty()
                    || !hipc.send_statics.is_empty()
                    || !hipc.send_buffers.is_empty()
                    || hipc.receive_buffers.len() != 1
                    || !hipc.exchange_buffers.is_empty()
                    || !matches!(hipc.receive_statics, ReceiveStatics::None)
                {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    );
                }
                // The local offline account has no cached friends. Return the
                // actual count without touching unused output-array entries.
                // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/friend/friend.cpp#L198-L213
                return semantic_success(
                    request.token,
                    session.domain.is_domain(),
                    &0_u32.to_le_bytes(),
                    &[],
                    &[],
                    None,
                );
            }
            if !matches!(request.command_id, 0 | 1) {
                return unsupported_service_command("IFriendService", request.command_id);
            }
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            if request.command_id == 0 {
                // Eden signals the completion event before exporting its reader.
                completion.0.signal();
                let handle = process
                    .return_handle(completion.1)
                    .map_err(IpcWireError::Handle)?;
                semantic_success(
                    request.token,
                    session.domain.is_domain(),
                    &[],
                    &[handle],
                    &[],
                    None,
                )
            } else {
                // No asynchronous friend operation is pending on this interface.
                response(session, request.token, HorizonIpcResult::SUCCESS)
            }
        }
        FriendObject::Notification { user, event } => {
            if !matches!(request.command_id, 0..=2) {
                return unsupported_service_command("INotificationService", request.command_id);
            }
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            log::trace!(
                "friends notification command={} user={user:032x}",
                request.command_id
            );
            match request.command_id {
                0 => {
                    let handle = process
                        .return_handle(event.1)
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
                1 => response(session, request.token, HorizonIpcResult::SUCCESS),
                2 => response(
                    session,
                    request.token,
                    HorizonIpcResult::ACCOUNT_NO_NOTIFICATIONS,
                ),
                _ => unreachable!(),
            }
        }
    }
}
