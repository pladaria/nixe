use super::super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeServiceCommand {
    StandardUserSystemClock,
    StandardNetworkSystemClock,
    StandardSteadyClock,
    TimeZoneService,
    StandardLocalSystemClock,
    SharedMemoryNativeHandle,
}

impl TimeServiceCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::StandardUserSystemClock),
            1 => Some(Self::StandardNetworkSystemClock),
            2 => Some(Self::StandardSteadyClock),
            3 => Some(Self::TimeZoneService),
            4 => Some(Self::StandardLocalSystemClock),
            20 => Some(Self::SharedMemoryNativeHandle),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SystemClockCommand {
    GetCurrentTime,
    SetCurrentTime,
}

impl SystemClockCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetCurrentTime),
            1 => Some(Self::SetCurrentTime),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SteadyClockCommand {
    GetCurrentTimePoint,
    GetInternalOffset,
}

impl SteadyClockCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetCurrentTimePoint),
            200 => Some(Self::GetInternalOffset),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeZoneCommand {
    GetDeviceLocationName,
    GetTotalLocationNameCount,
    ToCalendarTimeWithMyRule,
}

impl TimeZoneCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetDeviceLocationName),
            2 => Some(Self::GetTotalLocationNameCount),
            101 => Some(Self::ToCalendarTimeWithMyRule),
            _ => None,
        }
    }
}

// Static-service commands and returned child sessions follow libnx:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/time.c#L25-L80
pub(in crate::ipc_wire) fn dispatch_time(
    process: &mut ExceptionProcessContext<'_>,
    session: &TimeServiceSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = TimeServiceCommand::decode(request.command_id) else {
        return unsupported_service_command("time:u", request.command_id);
    };
    let child = match command {
        TimeServiceCommand::StandardUserSystemClock => Some(HorizonIpcObject::SystemClock(
            session.system_clock(SystemClockKind::User),
        )),
        TimeServiceCommand::StandardNetworkSystemClock => Some(HorizonIpcObject::SystemClock(
            session.system_clock(SystemClockKind::Network),
        )),
        TimeServiceCommand::StandardSteadyClock => {
            Some(HorizonIpcObject::SteadyClock(session.steady_clock()))
        }
        TimeServiceCommand::TimeZoneService => {
            Some(HorizonIpcObject::TimeZone(session.timezone_service()))
        }
        TimeServiceCommand::StandardLocalSystemClock => Some(HorizonIpcObject::SystemClock(
            session.system_clock(SystemClockKind::Local),
        )),
        TimeServiceCommand::SharedMemoryNativeHandle => {
            let handle = process
                .handles_mut()
                .insert(session.shared_memory())
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a time shared-memory handle")
                })?;
            log::debug!("time:u returned shared-memory handle {handle:#x}");
            return semantic_success(request.token, false, &[], &[handle], &[], None);
        }
    };
    let handle = process
        .handles_mut()
        .insert(child.expect("time child command was selected"))
        .map_err(|_| {
            IpcWireError::HostResourceExhausted("installing a time service child handle")
        })?;
    log::debug!("time:u {command:?} returned child session handle {handle:#x}");
    semantic_success(request.token, false, &[], &[], &[], Some(handle))
}

pub(in crate::ipc_wire) fn dispatch_system_clock(
    session: &SystemClockSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = SystemClockCommand::decode(request.command_id) else {
        return unsupported_service_command("ISystemClock", request.command_id);
    };
    match command {
        SystemClockCommand::GetCurrentTime => {
            let timestamp = session.current_time();
            semantic_success(
                request.token,
                false,
                &timestamp.to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        SystemClockCommand::SetCurrentTime => {
            let Some(timestamp) = request.data.get(..8) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            session
                .set_current_time(i64::from_le_bytes(timestamp.try_into().unwrap()))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("updating the emulated system clock")
                })?;
            semantic_success(request.token, false, &[], &[], &[], None)
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_steady_clock(
    session: &SteadyClockSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = SteadyClockCommand::decode(request.command_id) else {
        return unsupported_service_command("ISteadyClock", request.command_id);
    };
    match command {
        SteadyClockCommand::GetCurrentTimePoint => {
            let (time_point, source_id) = session.time_point();
            let mut data = [0_u8; 0x18];
            data[..8].copy_from_slice(&time_point.to_le_bytes());
            data[8..].copy_from_slice(&source_id);
            semantic_success(request.token, false, &data, &[], &[], None)
        }
        SteadyClockCommand::GetInternalOffset => {
            semantic_success(request.token, false, &0_i64.to_le_bytes(), &[], &[], None)
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_timezone(
    session: &TimeZoneServiceSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = TimeZoneCommand::decode(request.command_id) else {
        return unsupported_service_command("ITimeZoneService", request.command_id);
    };
    match command {
        TimeZoneCommand::GetDeviceLocationName => semantic_success(
            request.token,
            false,
            &session.location_name(),
            &[],
            &[],
            None,
        ),
        TimeZoneCommand::GetTotalLocationNameCount => {
            semantic_success(request.token, false, &1_u32.to_le_bytes(), &[], &[], None)
        }
        TimeZoneCommand::ToCalendarTimeWithMyRule => {
            let Some(timestamp) = request_u64(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Ok(timestamp) = i64::try_from(timestamp) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            let Some(data) = encode_calendar_time(session.timezone(), timestamp) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            semantic_success(request.token, false, &data, &[], &[], None)
        }
    }
}

fn encode_calendar_time(timezone: chrono_tz::Tz, timestamp: i64) -> Option<[u8; 0x20]> {
    let utc = chrono::DateTime::from_timestamp(timestamp, 0)?;
    let local = utc.with_timezone(&timezone);
    let year = u16::try_from(local.year()).ok()?;
    let mut data = [0_u8; 0x20];
    data[..2].copy_from_slice(&year.to_le_bytes());
    data[2] = local.month() as u8;
    data[3] = local.day() as u8;
    data[4] = local.hour() as u8;
    data[5] = local.minute() as u8;
    data[6] = local.second() as u8;
    data[8..12].copy_from_slice(&local.weekday().num_days_from_sunday().to_le_bytes());
    data[12..16].copy_from_slice(&local.ordinal0().to_le_bytes());
    let abbreviation = local.format("%Z").to_string();
    let abbreviation = abbreviation.as_bytes();
    let abbreviation_len = abbreviation.len().min(8);
    data[16..16 + abbreviation_len].copy_from_slice(&abbreviation[..abbreviation_len]);
    let dst = u32::from(local.offset().dst_offset().num_seconds() != 0);
    data[24..28].copy_from_slice(&dst.to_le_bytes());
    let offset = local.offset().fix().local_minus_utc();
    data[28..32].copy_from_slice(&offset.to_le_bytes());
    Some(data)
}

#[cfg(test)]
mod tests {
    use super::encode_calendar_time;

    #[test]
    fn calendar_conversion_uses_the_configured_timezone() {
        let encoded = encode_calendar_time(chrono_tz::Europe::Madrid, 0).unwrap();
        assert_eq!(u16::from_le_bytes(encoded[..2].try_into().unwrap()), 1970);
        assert_eq!(&encoded[2..7], &[1, 1, 1, 0, 0]);
        assert_eq!(
            i32::from_le_bytes(encoded[28..32].try_into().unwrap()),
            3600
        );
    }
}
