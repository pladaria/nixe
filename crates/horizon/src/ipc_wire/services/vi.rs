use super::super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ApplicationDisplayCommand {
    GetRelayService,
    GetSystemDisplayService,
    GetManagerDisplayService,
    OpenDisplay,
    CloseDisplay,
    GetDisplayResolution,
    OpenLayer,
    CloseLayer,
    CreateStrayLayer,
    DestroyStrayLayer,
    SetLayerScalingMode,
    GetDisplayVsyncEvent,
}

impl ApplicationDisplayCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            100 => Some(Self::GetRelayService),
            101 => Some(Self::GetSystemDisplayService),
            102 => Some(Self::GetManagerDisplayService),
            1010 => Some(Self::OpenDisplay),
            1020 => Some(Self::CloseDisplay),
            1102 => Some(Self::GetDisplayResolution),
            2020 => Some(Self::OpenLayer),
            2021 => Some(Self::CloseLayer),
            2030 => Some(Self::CreateStrayLayer),
            2031 => Some(Self::DestroyStrayLayer),
            2101 => Some(Self::SetLayerScalingMode),
            5202 => Some(Self::GetDisplayVsyncEvent),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SystemDisplayCommand {
    GetDisplayMode,
    SetLayerPosition,
    SetLayerSize,
    SetLayerZ,
}

impl SystemDisplayCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            1203 => Some(Self::GetDisplayMode),
            2201 => Some(Self::SetLayerPosition),
            2203 => Some(Self::SetLayerSize),
            2205 => Some(Self::SetLayerZ),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManagerDisplayCommand {
    CreateManaged,
    DestroyManaged,
    CreateStray,
}

impl ManagerDisplayCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            2010 => Some(Self::CreateManaged),
            2011 => Some(Self::DestroyManaged),
            2012 => Some(Self::CreateStray),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinderRelayCommand {
    TransactParcel,
    AdjustRefcount,
    GetNativeHandle,
    TransactParcelAuto,
}

impl BinderRelayCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::TransactParcel),
            1 => Some(Self::AdjustRefcount),
            2 => Some(Self::GetNativeHandle),
            3 => Some(Self::TransactParcelAuto),
            _ => None,
        }
    }
}

fn vi_child(
    process: &mut ExceptionProcessContext<'_>,
    token: u32,
    kind: ViObjectKind,
    video: &VideoSystem,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::Vi(ViSession::new(kind, video.clone())))
        .map_err(|_| IpcWireError::HostResourceExhausted("installing a VI child handle"))?;
    Ok((
        encode_response(token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
        Some(handle),
    ))
}

pub(in crate::ipc_wire) fn dispatch_vi(
    process: &mut ExceptionProcessContext<'_>,
    session: &ViSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let video = session.video();
    match session.kind() {
        ViObjectKind::Root(kind) => {
            if request.command_id != kind.required_root_command() {
                return unsupported_service_command(
                    vi_object_name(session.kind()),
                    request.command_id,
                );
            }
            vi_child(
                process,
                request.token,
                ViObjectKind::ApplicationDisplay,
                video,
            )
        }
        ViObjectKind::ApplicationDisplay => {
            let Some(command) = ApplicationDisplayCommand::decode(request.command_id) else {
                return unsupported_service_command(
                    "IApplicationDisplayService",
                    request.command_id,
                );
            };
            match command {
                ApplicationDisplayCommand::GetRelayService => {
                    vi_child(process, request.token, ViObjectKind::BinderRelay, video)
                }
                ApplicationDisplayCommand::GetSystemDisplayService => {
                    vi_child(process, request.token, ViObjectKind::SystemDisplay, video)
                }
                ApplicationDisplayCommand::GetManagerDisplayService => {
                    vi_child(process, request.token, ViObjectKind::ManagerDisplay, video)
                }
                ApplicationDisplayCommand::OpenDisplay => {
                    let Some(display_id) =
                        video.open_display(request.data.get(..0x40).unwrap_or(&[]))
                    else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &display_id.to_le_bytes(),
                            None,
                        )?,
                        None,
                    ))
                }
                ApplicationDisplayCommand::CloseDisplay => {
                    let Some(display_id) = request_u64(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = if VideoSystem::display_resolution(display_id).is_some() {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                    cmif_error(request.token, result)
                }
                ApplicationDisplayCommand::GetDisplayResolution => {
                    let Some(display_id) = request_u64(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some((width, height)) = VideoSystem::display_resolution(display_id) else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    let mut data = Vec::with_capacity(16);
                    data.extend_from_slice(&i64::from(width).to_le_bytes());
                    data.extend_from_slice(&i64::from(height).to_le_bytes());
                    Ok((
                        encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                        None,
                    ))
                }
                ApplicationDisplayCommand::OpenLayer => {
                    let Some(layer_id) = request_u64(request.data, 0x40) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some(layer) = video.layer(layer_id) else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    let descriptor = one_receive_buffer(hipc)?;
                    let native_window = crate::graphics::encode_native_window(layer.binder_id);
                    write_descriptor_bytes(process, descriptor, &native_window)?;
                    Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &(native_window.len() as u64).to_le_bytes(),
                            None,
                        )?,
                        None,
                    ))
                }
                ApplicationDisplayCommand::CreateStrayLayer => {
                    let Some(display_id) = request_u64(request.data, 8) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some(layer) = video.create_layer(display_id) else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    let descriptor = one_receive_buffer(hipc)?;
                    let native_window = crate::graphics::encode_native_window(layer.binder_id);
                    write_descriptor_bytes(process, descriptor, &native_window)?;
                    let mut data = Vec::with_capacity(16);
                    data.extend_from_slice(&layer.id.to_le_bytes());
                    data.extend_from_slice(&(native_window.len() as u64).to_le_bytes());
                    Ok((
                        encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                        None,
                    ))
                }
                ApplicationDisplayCommand::CloseLayer
                | ApplicationDisplayCommand::DestroyStrayLayer => {
                    let Some(layer_id) = request_u64(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = if video.remove_layer(layer_id) {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                    cmif_error(request.token, result)
                }
                ApplicationDisplayCommand::SetLayerScalingMode => {
                    let Some(mode) = request_u32(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some(layer_id) = request_u64(request.data, 8) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = if video.set_layer_scaling_mode(layer_id, mode) {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                    cmif_error(request.token, result)
                }
                ApplicationDisplayCommand::GetDisplayVsyncEvent => {
                    let Some(display_id) = request_u64(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some(event) = video.vsync_event(display_id) else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    let handle = process.handles_mut().insert(event).map_err(|_| {
                        IpcWireError::HostResourceExhausted("installing a VI event handle")
                    })?;
                    Ok((
                        CmifResponse {
                            token: request.token,
                            result: HorizonIpcResult::SUCCESS.raw(),
                            data: &[],
                            pid: None,
                            copy_handles: &[handle],
                            move_handles: &[],
                            send_statics: &[],
                            is_domain: false,
                            domain_objects: &[],
                        }
                        .encode()?,
                        Some(handle),
                    ))
                }
            }
        }
        ViObjectKind::SystemDisplay => {
            let Some(command) = SystemDisplayCommand::decode(request.command_id) else {
                return unsupported_service_command("ISystemDisplayService", request.command_id);
            };
            match command {
                SystemDisplayCommand::GetDisplayMode => {
                    let Some(display_id) = request_u64(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some((width, height)) = VideoSystem::display_resolution(display_id) else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    let mut data = Vec::with_capacity(8);
                    data.extend_from_slice(&width.to_le_bytes());
                    data.extend_from_slice(&height.to_le_bytes());
                    Ok((
                        encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                        None,
                    ))
                }
                // ABI layouts follow pinned libnx vi.c:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/vi.c#L411-L446
                SystemDisplayCommand::SetLayerPosition => {
                    let (Some(x), Some(y), Some(layer_id)) = (
                        request_f32(request.data, 0),
                        request_f32(request.data, 4),
                        request_u64(request.data, 8),
                    ) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = if x.is_finite()
                        && y.is_finite()
                        && video.set_layer_position(layer_id, x, y)
                    {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                    cmif_error(request.token, result)
                }
                SystemDisplayCommand::SetLayerSize => {
                    let (Some(layer_id), Some(width), Some(height)) = (
                        request_u64(request.data, 0),
                        request_i64(request.data, 8),
                        request_i64(request.data, 16),
                    ) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = match (u32::try_from(width), u32::try_from(height)) {
                        (Ok(width), Ok(height))
                            if width != 0
                                && height != 0
                                && video.set_layer_size(layer_id, width, height) =>
                        {
                            HorizonIpcResult::SUCCESS
                        }
                        _ => HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                    };
                    cmif_error(request.token, result)
                }
                SystemDisplayCommand::SetLayerZ => {
                    let (Some(layer_id), Some(z)) =
                        (request_u64(request.data, 0), request_i64(request.data, 8))
                    else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = if video.set_layer_z(layer_id, z) {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                    cmif_error(request.token, result)
                }
            }
        }
        ViObjectKind::ManagerDisplay => {
            let Some(command) = ManagerDisplayCommand::decode(request.command_id) else {
                return unsupported_service_command("IManagerDisplayService", request.command_id);
            };
            match command {
                command @ (ManagerDisplayCommand::CreateManaged
                | ManagerDisplayCommand::CreateStray) => {
                    let Some(display_id) = request_u64(request.data, 8) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let Some(layer) = video.create_layer(display_id) else {
                        return cmif_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    if command == ManagerDisplayCommand::CreateStray {
                        let descriptor = one_receive_buffer(hipc)?;
                        let native_window = crate::graphics::encode_native_window(layer.binder_id);
                        write_descriptor_bytes(process, descriptor, &native_window)?;
                        let mut data = Vec::with_capacity(16);
                        data.extend_from_slice(&layer.id.to_le_bytes());
                        data.extend_from_slice(&(native_window.len() as u64).to_le_bytes());
                        return Ok((
                            encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                            None,
                        ));
                    }
                    Ok((
                        encode_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &layer.id.to_le_bytes(),
                            None,
                        )?,
                        None,
                    ))
                }
                ManagerDisplayCommand::DestroyManaged => {
                    let Some(layer_id) = request_u64(request.data, 0) else {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    };
                    let result = if video.remove_layer(layer_id) {
                        HorizonIpcResult::SUCCESS
                    } else {
                        HorizonIpcResult::SF_PRECONDITION_VIOLATION
                    };
                    cmif_error(request.token, result)
                }
            }
        }
        ViObjectKind::BinderRelay => dispatch_binder_relay(process, video, request, hipc),
    }
}

pub(in crate::ipc_wire) const fn vi_object_name(kind: ViObjectKind) -> &'static str {
    match kind {
        ViObjectKind::Root(ViServiceKind::Application) => "vi:u",
        ViObjectKind::Root(ViServiceKind::System) => "vi:s",
        ViObjectKind::Root(ViServiceKind::Manager) => "vi:m",
        ViObjectKind::ApplicationDisplay => "IApplicationDisplayService",
        ViObjectKind::BinderRelay => "IHOSBinderDriver",
        ViObjectKind::SystemDisplay => "ISystemDisplayService",
        ViObjectKind::ManagerDisplay => "IManagerDisplayService",
    }
}

fn dispatch_binder_relay(
    process: &mut ExceptionProcessContext<'_>,
    video: &VideoSystem,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(binder_id) = request_u32(request.data, 0).map(|value| value as i32) else {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    };
    let Some(command) = BinderRelayCommand::decode(request.command_id) else {
        return unsupported_service_command("IHOSBinderDriver", request.command_id);
    };
    match command {
        BinderRelayCommand::TransactParcel | BinderRelayCommand::TransactParcelAuto => {
            let Some(code) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let input = one_send_buffer(hipc)?;
            let output = one_receive_buffer(hipc)?;
            let input_size = usize::try_from(input.size)
                .map_err(|_| IpcWireError::Malformed("Binder input buffer is too large"))?;
            let mut encoded = vec![0_u8; input_size];
            read_bytes(
                process,
                GuestVirtualAddress::new(input.address),
                &mut encoded,
            )?;
            let transaction =
                video
                    .transact_binder(binder_id, code, &encoded)
                    .map_err(|error| match error {
                        crate::parcel::ParcelError::Malformed(reason) => {
                            IpcWireError::Malformed(reason)
                        }
                        crate::parcel::ParcelError::Unsupported(detail) => {
                            IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "IGraphicBufferProducer",
                                    command_id: code,
                                    detail,
                                },
                            )
                        }
                    })?;
            log::trace!(
                "Binder producer {binder_id} completed transaction {code:#x}{}",
                if transaction.queued.is_some() {
                    " and queued a frame"
                } else {
                    ""
                }
            );
            write_descriptor_bytes(process, output, &transaction.reply)?;
            if let Some(request) = transaction.queued {
                video
                    .queue_graphic_buffer(binder_id, request)
                    .map_err(|error| match error {
                        crate::graphics::FramebufferError::Malformed(reason) => {
                            IpcWireError::Malformed(reason)
                        }
                        crate::graphics::FramebufferError::Unsupported(detail) => {
                            IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "IGraphicBufferProducer",
                                    command_id: code,
                                    detail,
                                },
                            )
                        }
                        crate::graphics::FramebufferError::Backend(reason) => {
                            IpcWireError::GraphicsBackend(reason)
                        }
                        crate::graphics::FramebufferError::NvMap(_) => {
                            IpcWireError::Malformed("queued graphic-buffer image view is invalid")
                        }
                    })?;
            }
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        BinderRelayCommand::AdjustRefcount => {
            // IHOSBinderDriver::AdjustRefcount uses three signed 32-bit values;
            // reference type 0 is weak and 1 is strong:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/binder.c#L115-L126
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/display/binder.h#L33-L50
            let (Some(add_value), Some(reference_type)) =
                (request_i32(request.data, 4), request_i32(request.data, 8))
            else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if !video.adjust_binder_refcount(binder_id, add_value, reference_type) {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            }
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        BinderRelayCommand::GetNativeHandle => {
            let Some(event) = video.binder_event(binder_id) else {
                return cmif_error(request.token, HorizonIpcResult::SF_PRECONDITION_VIOLATION);
            };
            let handle = process.handles_mut().insert(event).map_err(|_| {
                IpcWireError::HostResourceExhausted("installing a Binder event handle")
            })?;
            Ok((
                CmifResponse {
                    token: request.token,
                    result: HorizonIpcResult::SUCCESS.raw(),
                    data: &[],
                    pid: None,
                    copy_handles: &[handle],
                    move_handles: &[],
                    send_statics: &[],
                    is_domain: false,
                    domain_objects: &[],
                }
                .encode()?,
                Some(handle),
            ))
        }
    }
}
