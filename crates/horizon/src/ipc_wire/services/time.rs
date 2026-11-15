use super::prelude::*;
use crate::object::{MonotonicClockError, TimeObject};
use crate::timezone::{RULE_SIZE, TimeZoneRule, location_names};

enum TimeTarget {
    Root,
    Object(TimeObject),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeServiceCommand {
    StandardUserSystemClock,
    StandardNetworkSystemClock,
    StandardSteadyClock,
    TimeZoneService,
    StandardLocalSystemClock,
    SharedMemoryNativeHandle,
    CalculateMonotonicSystemClockBaseTimePoint,
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
            300 => Some(Self::CalculateMonotonicSystemClockBaseTimePoint),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SystemClockCommand {
    GetCurrentTime,
    SetCurrentTime,
    GetSystemClockContext,
}

impl SystemClockCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetCurrentTime),
            1 => Some(Self::SetCurrentTime),
            2 => Some(Self::GetSystemClockContext),
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
    LoadLocationNameList,
    LoadTimeZoneRule,
    ToCalendarTime,
    ToCalendarTimeWithMyRule,
}

impl TimeZoneCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetDeviceLocationName),
            2 => Some(Self::GetTotalLocationNameCount),
            3 => Some(Self::LoadLocationNameList),
            4 => Some(Self::LoadTimeZoneRule),
            100 => Some(Self::ToCalendarTime),
            101 => Some(Self::ToCalendarTimeWithMyRule),
            _ => None,
        }
    }
}

// Static-service commands and returned child sessions follow libnx:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/time.c#L25-L80
pub(in crate::ipc_wire) async fn dispatch_time(
    process: &mut Request,
    session: &TimeServiceSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            let result = if session.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            return time_response(session, request.token, result, &[], &[], &[]);
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return time_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            }
            if *object_id == 1 {
                TimeTarget::Root
            } else {
                let Some(object) = session.object(*object_id) else {
                    return time_response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                        &[],
                        &[],
                        &[],
                    );
                };
                TimeTarget::Object(object)
            }
        }
        None if session.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain time:u request omitted its domain header",
            ));
        }
        None => TimeTarget::Root,
    };

    match target {
        TimeTarget::Root => dispatch_time_root(process, session, request, hipc),
        TimeTarget::Object(TimeObject::SystemClock(clock)) => {
            dispatch_system_clock_target(Some(session), &clock, request, hipc).await
        }
        TimeTarget::Object(TimeObject::SteadyClock(clock)) => {
            dispatch_steady_clock_target(Some(session), &clock, request)
        }
        TimeTarget::Object(TimeObject::TimeZone(timezone)) => {
            dispatch_timezone_target(process, Some(session), &timezone, request, hipc).await
        }
    }
}

fn dispatch_time_root(
    process: &mut Request,
    session: &TimeServiceSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = TimeServiceCommand::decode(request.command_id) else {
        return unsupported_service_command("time:u", request.command_id);
    };
    let child = match command {
        TimeServiceCommand::StandardUserSystemClock => Some(TimeObject::SystemClock(
            session.system_clock(SystemClockKind::User),
        )),
        TimeServiceCommand::StandardNetworkSystemClock => Some(TimeObject::SystemClock(
            session.system_clock(SystemClockKind::Network),
        )),
        TimeServiceCommand::StandardSteadyClock => {
            Some(TimeObject::SteadyClock(session.steady_clock()))
        }
        TimeServiceCommand::TimeZoneService => {
            Some(TimeObject::TimeZone(session.timezone_service()))
        }
        TimeServiceCommand::StandardLocalSystemClock => Some(TimeObject::SystemClock(
            session.system_clock(SystemClockKind::Local),
        )),
        TimeServiceCommand::SharedMemoryNativeHandle => {
            let handle = process
                .return_handle(session.shared_memory())
                .map_err(IpcWireError::Handle)?;
            log::debug!("time:u returned shared-memory handle {handle:#x}");
            return time_response(
                session,
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                &[handle],
                &[],
            );
        }
        // Command 300 consumes one 0x20-byte SystemClockContext and returns
        // the POSIX base corresponding to Nixe's monotonic epoch:
        // https://switchbrew.org/w/index.php?title=PSC_services&oldid=14556#time:su,_time:s
        TimeServiceCommand::CalculateMonotonicSystemClockBaseTimePoint => {
            let Some(encoded_context) = request.data.get(..0x20) else {
                return time_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            };
            if !request.has_payload_size(0x20) || has_ipc_descriptors(hipc) {
                return time_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            }
            let result = session.calculate_monotonic_system_clock_base_time_point(
                encoded_context.try_into().unwrap(),
            );
            return match result {
                Ok(base_time) => time_response(
                    session,
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &base_time.to_le_bytes(),
                    &[],
                    &[],
                ),
                Err(MonotonicClockError::NotComparable) => time_response(
                    session,
                    request.token,
                    HorizonIpcResult::TIME_NOT_COMPARABLE,
                    &[],
                    &[],
                    &[],
                ),
                Err(MonotonicClockError::Overflowed) => time_response(
                    session,
                    request.token,
                    HorizonIpcResult::TIME_OVERFLOWED,
                    &[],
                    &[],
                    &[],
                ),
            };
        }
    };

    let child = child.expect("time child command was selected");
    if session.is_domain() {
        // Domain-converted services return child object IDs in the CMIF
        // domain response instead of allocating process handles:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L250-L266
        let Some(object_id) = session.insert_object(child) else {
            return time_response(
                session,
                request.token,
                HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                &[],
                &[],
                &[],
            );
        };
        log::debug!("time:u {command:?} returned domain object {object_id:#x}");
        return time_response(
            session,
            request.token,
            HorizonIpcResult::SUCCESS,
            &[],
            &[],
            &[object_id],
        );
    }

    let child = match child {
        TimeObject::SystemClock(clock) => HorizonIpcObject::SystemClock(clock),
        TimeObject::SteadyClock(clock) => HorizonIpcObject::SteadyClock(clock),
        TimeObject::TimeZone(timezone) => HorizonIpcObject::TimeZone(timezone),
    };
    let handle = process
        .return_service(child)
        .map_err(IpcWireError::Handle)?;
    log::debug!("time:u {command:?} returned child session handle {handle:#x}");
    semantic_success(request.token, false, &[], &[], &[], Some(handle))
}

pub(in crate::ipc_wire) async fn dispatch_system_clock(
    session: &SystemClockSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    dispatch_system_clock_target(None, session, request, hipc).await
}

async fn dispatch_system_clock_target(
    parent: Option<&TimeServiceSession>,
    session: &SystemClockSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = SystemClockCommand::decode(request.command_id) else {
        return unsupported_service_command("ISystemClock", request.command_id);
    };
    match command {
        SystemClockCommand::GetCurrentTime => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return time_object_response(
                    parent,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            }
            let timestamp = session.current_time();
            time_object_response(
                parent,
                request.token,
                HorizonIpcResult::SUCCESS,
                &timestamp.to_le_bytes(),
                &[],
                &[],
            )
        }
        SystemClockCommand::SetCurrentTime => {
            let Some(timestamp) = request.data.get(..8) else {
                return time_object_response(
                    parent,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            };
            if !request.has_payload_size(8) || has_ipc_descriptors(hipc) {
                return time_object_response(
                    parent,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            }
            session
                .set_current_time(i64::from_le_bytes(timestamp.try_into().unwrap()))
                .await
                .map_err(IpcWireError::SystemMemory)?;
            time_object_response(
                parent,
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
                &[],
                &[],
            )
        }
        // Command 2 returns the clock epoch followed by the steady-clock
        // point and its source ID as one 0x20-byte SystemClockContext:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/time.h
        SystemClockCommand::GetSystemClockContext => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return time_object_response(
                    parent,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                    &[],
                );
            }
            time_object_response(
                parent,
                request.token,
                HorizonIpcResult::SUCCESS,
                &session.context(),
                &[],
                &[],
            )
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_steady_clock(
    session: &SteadyClockSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    dispatch_steady_clock_target(None, session, request)
}

fn dispatch_steady_clock_target(
    parent: Option<&TimeServiceSession>,
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
            time_object_response(
                parent,
                request.token,
                HorizonIpcResult::SUCCESS,
                &data,
                &[],
                &[],
            )
        }
        SteadyClockCommand::GetInternalOffset => time_object_response(
            parent,
            request.token,
            HorizonIpcResult::SUCCESS,
            &0_i64.to_le_bytes(),
            &[],
            &[],
        ),
    }
}

pub(in crate::ipc_wire) async fn dispatch_timezone(
    process: &mut Request,
    session: &TimeZoneServiceSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    dispatch_timezone_target(process, None, session, request, hipc).await
}

async fn dispatch_timezone_target(
    process: &mut Request,
    parent: Option<&TimeServiceSession>,
    session: &TimeZoneServiceSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = TimeZoneCommand::decode(request.command_id) else {
        return unsupported_service_command("ITimeZoneService", request.command_id);
    };
    let response =
        |result, data: &[u8]| time_object_response(parent, request.token, result, data, &[], &[]);
    // Rules are fixed-size map-alias buffers (0x15/0x16), not auto-select
    // buffers (0x21/0x22). Calendar results are inline.
    // https://switchbrew.org/wiki/Glue_services#ITimeZoneService
    match command {
        TimeZoneCommand::GetDeviceLocationName => {
            response(HorizonIpcResult::SUCCESS, &session.location_name())
        }
        TimeZoneCommand::GetTotalLocationNameCount => response(
            HorizonIpcResult::SUCCESS,
            &(location_names().count() as u32).to_le_bytes(),
        ),
        TimeZoneCommand::LoadLocationNameList => {
            let Some(start) = request_i32(request.data, 0).filter(|&value| value >= 0) else {
                return response(HorizonIpcResult::TIME_ZONE_OUT_OF_RANGE, &[]);
            };
            let output = one_receive_buffer(process, hipc)?;
            let capacity = usize::try_from(output.size / 0x24).unwrap_or(usize::MAX);
            let names: Vec<_> = location_names()
                .skip(start as usize)
                .take(capacity)
                .collect();
            process.reserve_payload(names.len() * 0x24).await?;
            let mut data = vec![0; names.len() * 0x24];
            for (slot, name) in data.chunks_exact_mut(0x24).zip(&names) {
                slot[..name.len()].copy_from_slice(name.as_bytes());
            }
            process.write(output, 0, &mut data).await?;
            response(
                HorizonIpcResult::SUCCESS,
                &(names.len() as u32).to_le_bytes(),
            )
        }
        TimeZoneCommand::LoadTimeZoneRule => {
            let Some(name) = request.data.get(..0x24) else {
                return response(HorizonIpcResult::CMIF_INVALID_IN_HEADER, &[]);
            };
            let length = name
                .iter()
                .position(|&byte| byte == 0)
                .unwrap_or(name.len());
            let Ok(name) = std::str::from_utf8(&name[..length]) else {
                return response(HorizonIpcResult::TIME_ZONE_NOT_FOUND, &[]);
            };
            let output = one_receive_buffer(process, hipc)?;
            if output.size < RULE_SIZE as u64 {
                return response(HorizonIpcResult::TIME_ZONE_OUT_OF_RANGE, &[]);
            }
            process.reserve_payload(RULE_SIZE).await?;
            let rule = match TimeZoneRule::load(name) {
                Ok(rule) => rule,
                Err(error) => return response(error, &[]),
            };
            let mut bytes = rule.into_bytes();
            process.write(output, 0, &mut bytes).await?;
            response(HorizonIpcResult::SUCCESS, &[])
        }
        TimeZoneCommand::ToCalendarTime | TimeZoneCommand::ToCalendarTimeWithMyRule => {
            let Some(timestamp) = request_i64(request.data, 0) else {
                return response(HorizonIpcResult::CMIF_INVALID_IN_HEADER, &[]);
            };
            let explicit;
            let rule = if command == TimeZoneCommand::ToCalendarTime {
                let input = one_send_buffer(process, hipc)?;
                if input.size < RULE_SIZE as u64 {
                    return response(HorizonIpcResult::TIME_ZONE_OUT_OF_RANGE, &[]);
                }
                process.reserve_payload(RULE_SIZE).await?;
                let mut bytes = vec![0; RULE_SIZE];
                process.read(input, 0, &mut bytes).await?;
                explicit = TimeZoneRule::decode(bytes.into_boxed_slice().try_into().unwrap());
                explicit.as_ref().map_err(|&error| error)
            } else {
                session.timezone_rule()
            };
            match rule.and_then(|rule| rule.calendar(timestamp)) {
                Ok(data) => response(HorizonIpcResult::SUCCESS, &data),
                Err(error) => response(error, &[]),
            }
        }
    }
}

fn time_object_response(
    parent: Option<&TimeServiceSession>,
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    copy_handles: &[u32],
    domain_objects: &[u32],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if let Some(parent) = parent {
        time_response(parent, token, result, data, copy_handles, domain_objects)
    } else if result == HorizonIpcResult::SUCCESS {
        semantic_success(token, false, data, copy_handles, domain_objects, None)
    } else {
        Ok((encode_response(token, result, data, None)?, None))
    }
}

fn time_response(
    session: &TimeServiceSession,
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    copy_handles: &[u32],
    domain_objects: &[u32],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if session.is_domain() {
        let response = encode_domain_response(token, result, data, copy_handles, domain_objects)?;
        Ok((response, copy_handles.first().copied()))
    } else if result == HorizonIpcResult::SUCCESS {
        semantic_success(token, false, data, copy_handles, &[], None)
    } else {
        Ok((encode_response(token, result, data, None)?, None))
    }
}
