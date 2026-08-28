use super::super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PerformanceManagerCommand {
    OpenSession,
    GetPerformanceMode,
}

impl PerformanceManagerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::OpenSession),
            1 => Some(Self::GetPerformanceMode),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PerformanceSessionCommand {
    SetPerformanceConfiguration,
    GetPerformanceConfiguration,
}

impl PerformanceSessionCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::SetPerformanceConfiguration),
            1 => Some(Self::GetPerformanceConfiguration),
            _ => None,
        }
    }
}

// Command IDs, payloads, and the returned child object follow libnx:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/apm.c
pub(in crate::ipc_wire) fn dispatch_performance_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &PerformanceManagerSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = PerformanceManagerCommand::decode(request.command_id) else {
        return unsupported_service_command("apm", request.command_id);
    };
    match command {
        PerformanceManagerCommand::OpenSession => match process
            .handles_mut()
            .insert(HorizonIpcObject::Performance(manager.open_session()))
        {
            Ok(handle) => {
                log::debug!("apm opened performance session handle {handle:#x}");
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ))
            }
            Err(_) => Err(IpcWireError::HostResourceExhausted(
                "installing a performance-manager child handle",
            )),
        },
        PerformanceManagerCommand::GetPerformanceMode => {
            log::debug!("apm returned normal performance mode");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &PERFORMANCE_MODE_NORMAL.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_performance_session(
    session: &PerformanceSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = PerformanceSessionCommand::decode(request.command_id) else {
        return unsupported_service_command("IPerformanceSession", request.command_id);
    };
    match command {
        PerformanceSessionCommand::SetPerformanceConfiguration => {
            let Some(mode) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(configuration) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Ok(mode) = usize::try_from(mode) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            if !session.set_configuration(mode, configuration) {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            log::debug!("apm stored configuration {configuration:#x} for mode {mode}");
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        PerformanceSessionCommand::GetPerformanceConfiguration => {
            let Some(mode) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let configuration = usize::try_from(mode)
                .ok()
                .and_then(|mode| session.configuration(mode));
            let Some(configuration) = configuration else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            log::debug!("apm returned configuration {configuration:#x} for mode {mode}");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &configuration.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
    }
}
