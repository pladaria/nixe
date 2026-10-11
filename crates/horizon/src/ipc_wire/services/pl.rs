use super::prelude::*;
use crate::SharedFontSession;

// https://github.com/switchbrew/libnx/blob/master/nx/source/services/pl.c
pub(in crate::ipc_wire) fn dispatch_shared_font(
    process: &mut ExceptionProcessContext<'_>,
    session: &SharedFontSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if request.command_id > 4 {
        return unsupported_service_command("pl:u", request.command_id);
    }
    let expected_size = if request.command_id == 4 { 0 } else { 4 };
    if !request.has_payload_size(expected_size) || has_ipc_descriptors(hipc) {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let resource_error = |reason: String| IpcWireError::SharedFont(reason.into_boxed_str());
    if request.command_id == 4 {
        let shared = session
            .shared_memory(process.memory())
            .map_err(resource_error)?;
        let handle = process
            .handles_mut()
            .insert(shared)
            .map_err(|_| IpcWireError::HostResourceExhausted("returning shared font memory"))?;
        return semantic_success(request.token, false, &[], &[handle], &[], None);
    }
    let font = request_u32(request.data, 0).unwrap();
    if !SharedFontSession::valid_type(font) {
        return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
    }
    let mut data = Vec::new();
    match request.command_id {
        0 => {
            session
                .request_load(process.memory(), font)
                .map_err(resource_error)?;
        }
        1 => {
            // Horizon prepares the system fonts independently of application
            // RequestLoad calls. Materialize that backing on first consumption;
            // never leave clients polling an idle host-side loader forever.
            session
                .request_load(process.memory(), font)
                .map_err(resource_error)?;
            data.extend_from_slice(&1_u32.to_le_bytes());
        }
        2 | 3 => {
            let region = session
                .request_load(process.memory(), font)
                .map_err(resource_error)?;
            let value = if request.command_id == 2 {
                region.size
            } else {
                region.offset
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        _ => unreachable!(),
    }
    semantic_success(request.token, false, &data, &[], &[], None)
}
