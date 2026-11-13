//! Syncpoint event slots are kernel events owned by the client's graphics context.
use super::*;
use crate::gsv::syncpt::Slot;
use std::sync::{Arc, Mutex};

pub(super) fn dispatch(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed("syncpoint request has descriptors"));
    }
    let mut state = session.state.lock().unwrap();
    let context = state
        .syncpt
        .as_mut()
        .filter(|context| context.initialized)
        .ok_or(IpcWireError::Malformed("syncpoint context not initialized"))?;
    // @SPECULATIVE: the slot-count query/configuration ABI and 32-slot capacity
    // follow the reference SDK observations; Switchbrew lists these commands
    // without their argument layouts.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
    match (request.command_id, request.data.len()) {
        (0, 8) => Ok(encode_tipc_response(0, request.data, &[], &[])?),
        (1, 0) => Ok(encode_tipc_response(0, &32_u32.to_le_bytes(), &[], &[])?),
        (21, 4) => {
            let count = request_u32(request.data, 0).unwrap() as usize;
            if count == 0 || count > 32 || !context.slots.is_empty() {
                return Err(IpcWireError::Malformed(
                    "invalid syncpoint slot configuration",
                ));
            }
            context.slots.resize(count, None);
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        (22, 4) => {
            let slot = request_u32(request.data, 0).unwrap() as usize;
            let event = context
                .slots
                .get_mut(slot)
                .ok_or(IpcWireError::Malformed("syncpoint slot out of range"))?
                .get_or_insert_with(|| {
                    Arc::new(Mutex::new(Slot::new(nixe_runtime::EventObject::new(
                        process.kernel(),
                    ))))
                });
            let handle = process
                .return_handle(event.lock().unwrap().event.readable())
                .map_err(IpcWireError::Handle)?;
            Ok(encode_tipc_response(0, &[], &[handle], &[])?)
        }
        // @SPECULATIVE: the reference/native SDK encodes SyncCounterId as
        // !syncpoint; mode 0 compares wrapping u32, mode 1 the extended u64.
        // Switchbrew names the service but does not describe these wire fields.
        (103, 4) => {
            let id = !request_u32(request.data, 0).unwrap();
            let counter = context
                .timelines
                .get(&id)
                .ok_or(IpcWireError::Malformed("unknown native counter"))?
                .lock()
                .unwrap();
            let mut data = counter.value.to_le_bytes().to_vec();
            data.extend_from_slice(&0_u32.to_le_bytes());
            Ok(encode_tipc_response(0, &data, &[], &[])?)
        }
        (201, 20) => {
            let index = request_u32(request.data, 0).unwrap() as usize;
            let id = !request_u32(request.data, 4).unwrap();
            let value = request_u64(request.data, 8).unwrap();
            let mode = request_u32(request.data, 16).unwrap();
            if mode > 1 {
                return Err(IpcWireError::Malformed("invalid native counter comparison"));
            }
            let slot = context
                .slots
                .get(index)
                .and_then(Option::as_ref)
                .ok_or(IpcWireError::Malformed("syncpoint slot event missing"))?;
            let mut counter = context
                .timelines
                .get(&id)
                .ok_or(IpcWireError::Malformed("unknown native counter"))?
                .lock()
                .unwrap();
            counter.arm(index, slot, value, mode == 1);
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        (202, 4) => {
            let index = request_u32(request.data, 0).unwrap() as usize;
            let slot = context
                .slots
                .get(index)
                .and_then(Option::as_ref)
                .ok_or(IpcWireError::Malformed("syncpoint slot event missing"))?;
            // @SPECULATIVE: cancellation acknowledges through the slot event;
            // the client rechecks its counter rather than treating this as completion.
            slot.lock().unwrap().cancel();
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        _ => Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::Command {
                service: "GSV SyncptSession",
                command_id: request.command_id,
            },
        )),
    }
}
