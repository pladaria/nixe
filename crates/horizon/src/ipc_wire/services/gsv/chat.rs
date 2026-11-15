//! Local graphics-side GameChat lifecycle. No remote peers or streaming.
//! https://switchbrew.org/wiki/Switch_2%3A_GSV_services

use super::*;
use crate::gsv::{ChatContext, GsvInterface};

pub(super) fn context(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed("GameChat context has no inputs"));
    }
    let mut state = session.state.lock().unwrap();
    if !state.context_created {
        return Err(IpcWireError::Malformed("GSV context missing"));
    }
    match request.command_id {
        1201 => {
            if state.chat.is_some() {
                return Err(IpcWireError::Malformed("GameChat context already exists"));
            }
            state.chat = Some(ChatContext {
                initialized: false,
                event: nixe_runtime::EventObject::new(process.kernel()),
            });
        }
        1202 => {
            let chat = state
                .chat
                .as_mut()
                .filter(|chat| !chat.initialized)
                .ok_or(IpcWireError::Malformed("invalid GameChat initialization"))?;
            let handle = process
                .return_service(HorizonIpcObject::Gsv(
                    session.child(GsvInterface::Chat(chat.event.clone())),
                ))
                .map_err(IpcWireError::Handle)?;
            chat.initialized = true;
            return Ok(encode_tipc_response(0, &[], &[handle], &[])?);
        }
        1203 => {
            if state.chat.take().is_none() {
                return Err(IpcWireError::Malformed("GameChat context missing"));
            }
        }
        _ => unreachable!(),
    }
    Ok(encode_tipc_response(0, &[], &[], &[])?)
}

pub(super) fn dispatch(
    process: &mut Request,
    event: &nixe_runtime::EventObject,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed(
            "unexpected GameChat graphics descriptors",
        ));
    }
    // @SPECULATIVE: the reference reconstructs this interface from SDK calls:
    // command 1 returns event ID 0 and command 11 acquires that event. It stays
    // unsignalled while chat is inactive; this never completes GPU work.
    // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/nx2_gsv.h#L1777-L1817
    match (request.command_id, request.data.len()) {
        (1, 0) => Ok(encode_tipc_response(0, &0_u32.to_le_bytes(), &[], &[])?),
        (11, 4) if request_u32(request.data, 0) == Some(0) => {
            let handle = process
                .return_handle(event.readable())
                .map_err(IpcWireError::Handle)?;
            Ok(encode_tipc_response(0, &[], &[handle], &[])?)
        }
        (103, 12) if request_u32(request.data, 8).is_some_and(|id| id < 32) => {
            // @SPECULATIVE: nn::chat::gsv::OpenVirtualDisplay sends a packed
            // eight-byte name and a client notification slot, with no output.
            // The reference accepts this registration with an empty success.
            // Offline displays have no incoming frames or notifications.
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        _ => Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::Command {
                service: "GSV ChatSession",
                command_id: request.command_id,
            },
        )),
    }
}
