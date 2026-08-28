use super::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HidCommand {
    CreateAppletResource,
    StartSixAxisSensor,
    StopSixAxisSensor,
    SetSupportedNpadStyleSet,
    SetSupportedNpadIdType,
    ActivateNpad,
    SetSupportedNpadStyleSetUpdateEventHandle,
}

impl HidCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CreateAppletResource),
            66 => Some(Self::StartSixAxisSensor),
            67 => Some(Self::StopSixAxisSensor),
            100 => Some(Self::SetSupportedNpadStyleSet),
            102 => Some(Self::SetSupportedNpadIdType),
            103 => Some(Self::ActivateNpad),
            109 => Some(Self::SetSupportedNpadStyleSetUpdateEventHandle),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HidAppletResourceCommand {
    GetSharedMemoryHandle,
}

impl HidAppletResourceCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetSharedMemoryHandle),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_hid(
    process: &mut ExceptionProcessContext<'_>,
    session: &HidSession,
    hid_system: &HidSystem,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = HidCommand::decode(request.command_id) else {
        return unsupported_service_command("hid", request.command_id);
    };

    match command {
        // libnx sends the process ID and the applet-resource user ID:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L800-L808
        HidCommand::CreateAppletResource => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let handle = process
                .handles_mut()
                .insert(HorizonIpcObject::HidAppletResource(
                    session.create_applet_resource(),
                ))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a HID applet-resource handle")
                })?;
            log::debug!("hid created IAppletResource handle {handle:#x}");
            semantic_success(request.token, false, &[], &[], &[], Some(handle))
        }
        command @ (HidCommand::StartSixAxisSensor | HidCommand::StopSixAxisSensor) => {
            if hipc.pid.is_none() || request.data.len() < 16 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let handle = request_u32(request.data, 0).expect("validated HID handle payload");
            hid_system
                .set_six_axis_sensor_active(handle, command == HidCommand::StartSixAxisSensor);
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        HidCommand::SetSupportedNpadStyleSet => {
            if hipc.pid.is_none() || request.data.len() < 16 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let style_set = request_u32(request.data, 0).expect("validated HID style payload");
            hid_system.set_supported_npad_style_set(style_set);
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        HidCommand::SetSupportedNpadIdType => {
            if hipc.pid.is_none()
                || request.data.len() < 8
                || !matches!(
                    (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice()),
                    ([_], []) | ([], [_])
                )
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (address, size) = match (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice())
            {
                ([descriptor], []) => (descriptor.address, usize::from(descriptor.size)),
                ([], [descriptor]) if descriptor.mode != BufferMode::Invalid => (
                    descriptor.address,
                    usize::try_from(descriptor.size)
                        .map_err(|_| IpcWireError::Malformed("HID Npad ID buffer is too large"))?,
                ),
                _ => unreachable!("validated HID Npad ID descriptor"),
            };
            if size == 0 || !size.is_multiple_of(4) || size > 10 * 4 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut encoded_ids = vec![0; size];
            read_bytes(process, GuestVirtualAddress::new(address), &mut encoded_ids)?;
            let ids = encoded_ids
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
            if !hid_system.set_supported_npad_ids(ids) {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        HidCommand::ActivateNpad => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            hid_system.activate_npad();
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        HidCommand::SetSupportedNpadStyleSetUpdateEventHandle => {
            if hipc.pid.is_none() || request.data.len() < 16 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let revision = request_u32(request.data, 0).expect("validated HID revision payload");
            Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::CommandVariant {
                    service: "hid",
                    command_id: request.command_id,
                    detail: match revision {
                        1 => "Npad shared-memory revision 1",
                        2 => "Npad shared-memory revision 2",
                        3 => "Npad shared-memory revision 3",
                        5 => "Npad shared-memory revision 5",
                        _ => "unknown Npad shared-memory revision",
                    },
                },
            ))
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_hid_applet_resource(
    process: &mut ExceptionProcessContext<'_>,
    resource: &HidAppletResource,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(HidAppletResourceCommand::GetSharedMemoryHandle) =
        HidAppletResourceCommand::decode(request.command_id)
    else {
        return unsupported_service_command("IAppletResource", request.command_id);
    };
    // libnx maps the returned 0x40000-byte shared-memory object read-only:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L47-L65
    let handle = process
        .handles_mut()
        .insert(resource.shared_memory())
        .map_err(|_| {
            IpcWireError::HostResourceExhausted("installing a HID shared-memory handle")
        })?;
    log::debug!("hid returned shared-memory handle {handle:#x}");
    semantic_success(request.token, false, &[], &[handle], &[], None)
}
