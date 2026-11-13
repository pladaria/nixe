//! Syncpoint event slots are kernel events owned by the client's graphics context.
use super::*;

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
                .get_or_insert_with(|| nixe_runtime::EventObject::new(process.kernel()));
            let handle = process
                .return_handle(event.readable())
                .map_err(IpcWireError::Handle)?;
            Ok(encode_tipc_response(0, &[], &[handle], &[])?)
        }
        _ => Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::Command {
                service: "GSV SyncptSession",
                command_id: request.command_id,
            },
        )),
    }
}
