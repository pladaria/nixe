use super::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HidCommand {
    CreateAppletResource,
    ActivateTouchScreen,
    ActivateMouse,
    ActivateKeyboard,
    StartSixAxisSensor,
    StopSixAxisSensor,
    SetSupportedNpadStyleSet,
    GetSupportedNpadStyleSet,
    SetSupportedNpadIdType,
    ActivateNpad,
    AcquireNpadStyleSetUpdateEventHandle,
    ActivateNpadWithRevision,
    SetNpadJoyHoldType,
    GetNpadJoyHoldType,
    GetVibrationDeviceInfo,
    SendVibrationValue,
    CreateActiveVibrationDeviceList,
}

impl HidCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CreateAppletResource),
            11 => Some(Self::ActivateTouchScreen),
            21 => Some(Self::ActivateMouse),
            31 => Some(Self::ActivateKeyboard),
            66 => Some(Self::StartSixAxisSensor),
            67 => Some(Self::StopSixAxisSensor),
            100 => Some(Self::SetSupportedNpadStyleSet),
            101 => Some(Self::GetSupportedNpadStyleSet),
            102 => Some(Self::SetSupportedNpadIdType),
            103 => Some(Self::ActivateNpad),
            106 => Some(Self::AcquireNpadStyleSetUpdateEventHandle),
            109 => Some(Self::ActivateNpadWithRevision),
            120 => Some(Self::SetNpadJoyHoldType),
            121 => Some(Self::GetNpadJoyHoldType),
            200 => Some(Self::GetVibrationDeviceInfo),
            201 => Some(Self::SendVibrationValue),
            203 => Some(Self::CreateActiveVibrationDeviceList),
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
        // The two amplitudes determine actuator force; with both zero the
        // operation stops vibration regardless of the carrier frequencies.
        // The configured input worker owns actuator output, including stops.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L1043-L1054
        HidCommand::SendVibrationValue => {
            if !request.has_payload_size(32)
                || request_u32(request.data, 20) != Some(0)
                || hipc.pid.is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let handle = request_u32(request.data, 0).unwrap();
            if crate::hid::vibration_device_position(handle).is_none() {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            let values = [4, 8, 12, 16].map(|offset| request_f32(request.data, offset).unwrap());
            if values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            let value = nixe_input::VibrationValue {
                low_amplitude: values[0],
                low_frequency: values[1],
                high_amplitude: values[2],
                high_frequency: values[3],
            };
            log::debug!("HID vibration request: handle={handle:#010x} value={value:?}");
            if let Some(result) = hid_system.send_vibration(handle, value) {
                result.map_err(|e| IpcWireError::InputBackend(e.to_string().into_boxed_str()))?;
            } else if !value.is_stopped() {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandVariant {
                        service: "hid",
                        command_id: 201,
                        detail: "nonzero vibration requires a host actuator backend",
                    },
                ));
            }
            semantic_success(request.token, false, &[], &[], &[], None)
        }

        // One u32 handle, no PID or descriptors. The output is two u32s:
        // actuator type (LRA = 1) and its left/right position (1/2).
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L1039-L1041
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L1326-L1330
        HidCommand::GetVibrationDeviceInfo => {
            let Some(handle) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            if !matches!(handle as u8, 3..=7) {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandVariant {
                        service: "hid",
                        command_id: 200,
                        detail: "vibration device style outside FullKey, Handheld and Joy-Con",
                    },
                ));
            }
            let Some(position) = crate::hid::vibration_device_position(handle) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            let mut info = [0; 8];
            info[..4].copy_from_slice(&1_u32.to_le_bytes());
            info[4..].copy_from_slice(&position.to_le_bytes());
            semantic_success(request.token, false, &info, &[], &[], None)
        }
        // No input or PID. The client expects a moved session handle.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L1066-L1072
        HidCommand::CreateActiveVibrationDeviceList => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let handle = process
                .handles_mut()
                .insert(HorizonIpcObject::HidActiveVibrationDeviceList(
                    HidActiveVibrationDeviceList::default(),
                ))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted(
                        "installing a HID active vibration device list",
                    )
                })?;
            semantic_success(request.token, false, &[], &[], &[], Some(handle))
        }
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
        // Activating the touch screen carries the caller PID and its applet
        // resource user ID. Host contacts are published through HID shared
        // memory only after this command succeeds.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L538-L543
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L734-L735
        // Mouse/keyboard use the same PID + ARUID activation ABI.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L734-L744
        command @ (HidCommand::ActivateTouchScreen
        | HidCommand::ActivateMouse
        | HidCommand::ActivateKeyboard) => {
            // Plain CMIF carries alignment slack after this u64. libnx writes
            // only the semantic payload, so those trailing bytes retain prior
            // TLS contents and are not command input:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L93-L146
            if hipc.pid.is_none()
                || request_u64(request.data, 0).is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            match command {
                HidCommand::ActivateTouchScreen => hid_system.activate_touch_screen(),
                HidCommand::ActivateMouse => hid_system.activate_mouse(),
                HidCommand::ActivateKeyboard => hid_system.activate_keyboard(),
                _ => unreachable!(),
            }
            semantic_success(request.token, false, &[], &[], &[], None)
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
        // Get carries the ARUID and caller PID and returns the configured
        // u32 style mask, independently of currently connected controllers.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c
        HidCommand::GetSupportedNpadStyleSet => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0).is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            semantic_success(
                request.token,
                false,
                &hid_system.supported_npad_style_set().to_le_bytes(),
                &[],
                &[],
                None,
            )
        }
        HidCommand::SetSupportedNpadIdType => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            // This InArray is pointer-only, so QueryPointerBufferSize must
            // advertise enough space before the SDK can serialize it.
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c
            let [descriptor] = hipc.send_statics.as_slice() else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if descriptor.index != 0 || !hipc.send_buffers.is_empty() {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (address, size) = (descriptor.address, usize::from(descriptor.size));
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
        // Set carries ARUID then the hold type as u64; Get carries only
        // ARUID and returns a u64. Both send PID, with no buffer descriptors.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L615-L622
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L922-L930
        command @ (HidCommand::SetNpadJoyHoldType | HidCommand::GetNpadJoyHoldType) => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0).is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            if command == HidCommand::SetNpadJoyHoldType {
                let Some(hold_type) = request_u64(request.data, 8) else {
                    return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                };
                if !hid_system.set_npad_joy_hold_type(hold_type) {
                    return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
                }
                semantic_success(request.token, false, &[], &[], &[], None)
            } else {
                semantic_success(
                    request.token,
                    false,
                    &hid_system.npad_joy_hold_type().to_le_bytes(),
                    &[],
                    &[],
                    None,
                )
            }
        }
        HidCommand::ActivateNpad => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            hid_system.activate_npad();
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        HidCommand::AcquireNpadStyleSetUpdateEventHandle => {
            // u32 Npad ID, alignment padding, u64 ARUID and an unused client
            // pointer. The returned handle is a copy of a readable event.
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c
            if hipc.pid.is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
                || request.data.len() < 24
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let id = request_u32(request.data, 0).expect("validated Npad ID payload");
            let Some(event) = hid_system.acquire_style_event(id) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            let handle = process.handles_mut().insert(event).map_err(|_| {
                IpcWireError::HostResourceExhausted("copying a HID style update event")
            })?;
            semantic_success(request.token, false, &[], &[handle], &[], None)
        }
        HidCommand::ActivateNpadWithRevision => {
            if hipc.pid.is_none()
                || has_ipc_descriptors_other_than_pid(hipc)
                || request.data.len() < 16
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let revision = request_u32(request.data, 0).expect("validated HID revision payload");
            // Revisions 0..=3 use the same FullKey LIFO representation that
            // this producer publishes. Revision changes must not invent a
            // different memory layout or enable unsupported controller styles.
            // https://github.com/switchbrew/libnx/blob/v3.0.0/nx/include/switch/services/hid.h
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c
            if revision <= 3 {
                hid_system.activate_npad();
                return semantic_success(request.token, false, &[], &[], &[], None);
            }

            Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::CommandVariant {
                    service: "hid",
                    command_id: request.command_id,
                    detail: match revision {
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

pub(in crate::ipc_wire) fn dispatch_hid_active_vibration_device_list(
    list: &HidActiveVibrationDeviceList,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if request.command_id != 0 {
        return unsupported_service_command("IActiveVibrationDeviceList", request.command_id);
    }
    // ActivateVibrationDevice sends one packed u32, without PID or buffers.
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L1070-L1072
    let Some(handle) = request_u32(request.data, 0) else {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    };
    if has_ipc_descriptors(hipc) {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    // Actuator types outside the currently modeled Npad styles remain an
    // explicit emulator gap, rather than masquerading as invalid guest input.
    if !matches!(handle as u8, 3..=7) {
        return Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::CommandVariant {
                service: "IActiveVibrationDeviceList",
                command_id: 0,
                detail: "vibration device style outside FullKey, Handheld and Joy-Con",
            },
        ));
    }
    if !list.activate(handle) {
        return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
    }
    semantic_success(request.token, false, &[], &[], &[], None)
}
