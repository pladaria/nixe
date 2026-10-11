use super::prelude::*;
use crate::multimedia::MultimediaSession;

pub(in crate::ipc_wire) fn dispatch_multimedia(
    session: &MultimediaSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let command = request.command_id;
    if command > 7 {
        return unsupported_service_command("mm:u", command);
    }
    let size = if matches!(command, 0 | 2 | 4 | 6) {
        12
    } else {
        4
    };
    if !request.has_payload_size(size) || has_ipc_descriptors(hipc) {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let key = request_u32(request.data, 0).unwrap();
    let result = if matches!(command, 0 | 4) {
        let clear_mode = request_u32(request.data, 8).unwrap();
        if clear_mode > 1 {
            return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
        }
        // Priority is caller policy metadata. No completion event is exposed by
        // this interface; the execution/completion of engines remains separate.
        session
            .initialize(key)
            .map(|id| if command == 4 { Some(id) } else { None })
    } else {
        let bounds = matches!(command, 2 | 6).then(|| {
            (
                request_u32(request.data, 4).unwrap(),
                request_i32(request.data, 8).unwrap(),
            )
        });
        session.request(command, key, bounds)
    };
    let value = result.map_err(|detail| {
        IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
            service: "mm:u",
            command_id: command,
            detail,
        })
    })?;
    let data = value.map(u32::to_le_bytes);
    semantic_success(
        request.token,
        false,
        data.as_ref().map_or(&[], |v| v.as_slice()),
        &[],
        &[],
        None,
    )
}
