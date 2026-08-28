use super::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AccountCommand {
    InitializeApplicationInfo,
}

impl AccountCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            // InitializeApplicationInfo moved from command 100 to 140 in
            // Horizon 6.0.0.
            100 | 140 => Some(Self::InitializeApplicationInfo),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_account(
    process: &ExceptionProcessContext<'_>,
    session: &AccountSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = AccountCommand::decode(request.command_id) else {
        return unsupported_service_command("acc:u0", request.command_id);
    };

    match command {
        // libnx sends the caller PID descriptor and a zero u64 placeholder:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/acc.c#L61-L67
        AccountCommand::InitializeApplicationInfo => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0) != Some(0)
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.initialize_application_info(process.process_id());
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
    }
}
