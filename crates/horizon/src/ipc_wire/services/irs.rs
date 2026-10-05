use super::prelude::*;
use crate::object::IrsSession;

pub(in crate::ipc_wire) fn dispatch_irs(
    process: &mut ExceptionProcessContext<'_>,
    session: &IrsSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if request.command_id == 311 {
        // This identifies a camera slot; it does not require a connected camera.
        // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/core/hle/service/hid/irs.cpp#L202-L214
        if !request.has_payload_size(4) || has_ipc_descriptors(hipc) {
            return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
        }
        let id = request_u32(request.data, 0).expect("validated Npad ID");
        let index = match id {
            0..=7 => id,
            0x20 => 8,
            0x10 => 9,
            _ => return cmif_error(request.token, HorizonIpcResult::HID_INVALID_NPAD_ID),
        };
        return semantic_success(request.token, false, &index.to_le_bytes(), &[], &[], None);
    }
    // Initialization and shared-memory handle ABI, including sent PID and ARUID:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/irs.c#L422-L438
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/irs.c#L602-L613
    let (offset, size) = match request.command_id {
        302..=304 => (0, 8),
        319 => (8, 16),
        _ => return unsupported_service_command("irs", request.command_id),
    };
    if hipc.pid.is_none()
        || !request.has_payload_size(size)
        || has_ipc_descriptors_other_than_pid(hipc)
    {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let aruid = request_u64(request.data, offset).expect("validated IRS payload");
    if aruid != process.process_id() {
        return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
    }
    if request.command_id == 319 && request.data[0] > 3 {
        return Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::CommandVariant {
                service: "irs",
                command_id: 319,
                detail: "unknown infrared sensor function level",
            },
        ));
    }
    if request.command_id == 304 {
        let handle = process
            .handles_mut()
            .insert(session.shared_memory())
            .map_err(|_| {
                IpcWireError::HostResourceExhausted("installing IRS shared-memory handle")
            })?;
        return semantic_success(request.token, false, &[], &[handle], &[], None);
    }
    let registered = session
        .set_active(aruid, request.command_id != 303)
        .map_err(|_| IpcWireError::Internal("updating IRS applet registration"))?;
    if !registered {
        return Err(IpcWireError::HostResourceExhausted(
            "registering IRS applet",
        ));
    }
    semantic_success(request.token, false, &[], &[], &[], None)
}
