use super::super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootCommand {
    OpenApplicationProxy,
    OpenSystemAppletProxy,
}

impl RootCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::OpenApplicationProxy),
            100 => Some(Self::OpenSystemAppletProxy),
            _ => None,
        }
    }

    const fn proxy_kind(self) -> AppletProxyKind {
        match self {
            Self::OpenApplicationProxy => AppletProxyKind::Application,
            Self::OpenSystemAppletProxy => AppletProxyKind::SystemApplet,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProxyCommand {
    CommonStateGetter,
    SelfController,
    WindowController,
    AudioController,
    DisplayController,
    LibraryAppletCreator,
    DebugFunctions,
    RoleFunctions,
    GlobalStateController,
    ApplicationCreator,
    AppletCommonFunctions,
}

impl ProxyCommand {
    const fn decode(kind: AppletProxyKind, command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CommonStateGetter),
            1 => Some(Self::SelfController),
            2 => Some(Self::WindowController),
            3 => Some(Self::AudioController),
            4 => Some(Self::DisplayController),
            11 => Some(Self::LibraryAppletCreator),
            20 => Some(Self::RoleFunctions),
            21 if matches!(kind, AppletProxyKind::SystemApplet) => {
                Some(Self::GlobalStateController)
            }
            22 if matches!(kind, AppletProxyKind::SystemApplet) => Some(Self::ApplicationCreator),
            23 if matches!(kind, AppletProxyKind::SystemApplet) => {
                Some(Self::AppletCommonFunctions)
            }
            1000 => Some(Self::DebugFunctions),
            _ => None,
        }
    }

    const fn child(self, kind: AppletProxyKind) -> AppletObject {
        match self {
            Self::CommonStateGetter => AppletObject::CommonStateGetter,
            Self::SelfController => AppletObject::SelfController,
            Self::WindowController => AppletObject::WindowController,
            Self::AudioController => AppletObject::AudioController,
            Self::DisplayController => AppletObject::DisplayController,
            Self::LibraryAppletCreator => AppletObject::LibraryAppletCreator,
            Self::DebugFunctions => AppletObject::DebugFunctions,
            Self::RoleFunctions => match kind {
                AppletProxyKind::Application => AppletObject::ApplicationFunctions,
                AppletProxyKind::SystemApplet => AppletObject::HomeMenuFunctions,
            },
            Self::GlobalStateController => AppletObject::GlobalStateController,
            Self::ApplicationCreator => AppletObject::ApplicationCreator,
            Self::AppletCommonFunctions => AppletObject::AppletCommonFunctions,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommonStateGetterCommand {
    GetEventHandle,
    ReceiveMessage,
    GetOperationMode,
    GetPerformanceMode,
    GetCurrentFocusState,
}

impl CommonStateGetterCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetEventHandle),
            1 => Some(Self::ReceiveMessage),
            5 => Some(Self::GetOperationMode),
            6 => Some(Self::GetPerformanceMode),
            9 => Some(Self::GetCurrentFocusState),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelfControllerCommand {
    Exit,
    LockExit,
    UnlockExit,
    GetLibraryAppletLaunchableEvent,
    SetOperationModeChangedNotification,
    SetPerformanceModeChangedNotification,
    SetFocusHandlingMode,
    SetOutOfFocusSuspendingEnabled,
    CreateManagedDisplayLayer,
}

impl SelfControllerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Exit),
            1 => Some(Self::LockExit),
            2 => Some(Self::UnlockExit),
            9 => Some(Self::GetLibraryAppletLaunchableEvent),
            11 => Some(Self::SetOperationModeChangedNotification),
            12 => Some(Self::SetPerformanceModeChangedNotification),
            13 => Some(Self::SetFocusHandlingMode),
            16 => Some(Self::SetOutOfFocusSuspendingEnabled),
            40 => Some(Self::CreateManagedDisplayLayer),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WindowControllerCommand {
    GetAppletResourceUserId,
    AcquireForegroundRights,
}

impl WindowControllerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            1 => Some(Self::GetAppletResourceUserId),
            10 => Some(Self::AcquireForegroundRights),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ApplicationFunctionsCommand {
    SetTerminateResult,
    NotifyRunning,
}

impl ApplicationFunctionsCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            22 => Some(Self::SetTerminateResult),
            40 => Some(Self::NotifyRunning),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LibraryAppletCreatorCommand {
    CreateLibraryApplet,
    CreateStorage,
}

impl LibraryAppletCreatorCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CreateLibraryApplet),
            10 => Some(Self::CreateStorage),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LibraryAppletAccessorCommand {
    GetAppletStateChangedEvent,
    Start,
    PushInData,
}

impl LibraryAppletAccessorCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetAppletStateChangedEvent),
            10 => Some(Self::Start),
            100 => Some(Self::PushInData),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StorageCommand {
    Open,
}

impl StorageCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Open),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StorageAccessorCommand {
    GetSize,
    Write,
    Read,
}

impl StorageAccessorCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetSize),
            10 => Some(Self::Write),
            11 => Some(Self::Read),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn applet_requests_self_exit(
    session: &AppletSession,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> bool {
    SelfControllerCommand::decode(request.command_id) == Some(SelfControllerCommand::Exit)
        && request.data.is_empty()
        && !has_ipc_descriptors(hipc)
        && matches!(
            &request.domain,
            Some(DomainRequest::SendMessage { object_id, .. })
                if session.object(*object_id) == Some(AppletObject::SelfController)
        )
}

pub(in crate::ipc_wire) fn dispatch_applet(
    process: &mut ExceptionProcessContext<'_>,
    session: &AppletSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    video_system: &VideoSystem,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // The startup order, command IDs, input PID/process handle, returned
    // objects, and scalar result layouts implemented below follow libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L112-L333
    let (object_id, input_objects) = match request.domain.as_ref() {
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => (*object_id, input_objects),
        Some(DomainRequest::Close { object_id }) => {
            let result = if session.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            log::debug!("appletOE closed domain object {object_id:#x}");
            return Ok((
                encode_domain_response(request.token, result, &[], &[], &[])?,
                None,
            ));
        }
        None => {
            return Err(IpcWireError::Malformed(
                "appletOE request was not sent through its domain",
            ));
        }
    };
    let Some(object) = session.object(object_id) else {
        return Ok((
            encode_domain_response(
                request.token,
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                &[],
                &[],
                &[],
            )?,
            None,
        ));
    };
    let accepts_input_objects = matches!(object, AppletObject::LibraryAppletAccessor { .. })
        && LibraryAppletAccessorCommand::decode(request.command_id)
            == Some(LibraryAppletAccessorCommand::PushInData);
    if !input_objects.is_empty() && !accepts_input_objects {
        return unsupported_service_command(applet_object_name(object), request.command_id);
    }

    match object {
        AppletObject::Root => {
            if hipc.pid.is_none()
                || hipc.copy_handles.as_slice() != [crate::CURRENT_PROCESS_HANDLE]
                || request_u64(request.data, 0) != Some(0)
                || request.data[8..].iter().any(|byte| *byte != 0)
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)
            {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            // libnx selects the root command from the process applet role:
            // application uses 0, while system applet uses 100. Keep the
            // resulting proxies distinct because their child command graphs
            // diverge after the shared controller interfaces.
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L155-L182
            let Some(command) = RootCommand::decode(request.command_id) else {
                return unsupported_service_command(applet_object_name(object), request.command_id);
            };
            let kind = command.proxy_kind();
            applet_child(
                session,
                request.token,
                AppletObject::Proxy(kind),
                applet_proxy_name(kind),
            )
        }
        AppletObject::Proxy(kind) => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let Some(command) = ProxyCommand::decode(kind, request.command_id) else {
                return unsupported_service_command(applet_object_name(object), request.command_id);
            };
            let child = command.child(kind);
            applet_child(session, request.token, child, applet_object_name(child))
        }
        AppletObject::CommonStateGetter => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let Some(command) = CommonStateGetterCommand::decode(request.command_id) else {
                return unsupported_service_command("ICommonStateGetter", request.command_id);
            };
            match command {
                CommonStateGetterCommand::GetEventHandle => {
                    let handle = match process.handles_mut().insert(session.message_event()) {
                        Ok(handle) => handle,
                        Err(_) => {
                            return Err(IpcWireError::HostResourceExhausted(
                                "installing the common-state message event handle",
                            ));
                        }
                    };
                    log::debug!("appletOE returned message event handle {handle:#x}");
                    Ok((
                        encode_domain_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &[],
                            &[handle],
                            &[],
                        )?,
                        Some(handle),
                    ))
                }
                CommonStateGetterCommand::ReceiveMessage => match session.receive_message() {
                    Some(message) => applet_data(request.token, &message.to_le_bytes()),
                    None => applet_error(request.token, HorizonIpcResult::AM_NO_MESSAGES),
                },
                CommonStateGetterCommand::GetOperationMode => {
                    applet_data(request.token, &[session.operation_mode().as_raw()])
                }
                CommonStateGetterCommand::GetPerformanceMode => {
                    applet_data(request.token, &PERFORMANCE_MODE_NORMAL.to_le_bytes())
                }
                CommonStateGetterCommand::GetCurrentFocusState => {
                    applet_data(request.token, &[session.current_focus_state()])
                }
            }
        }
        AppletObject::SelfController => {
            let Some(command) = SelfControllerCommand::decode(request.command_id) else {
                return unsupported_service_command("ISelfController", request.command_id);
            };
            match command {
                // SelfExit is a no-I/O request. Once AM accepts it, libnx sleeps
                // forever and relies on AM to terminate the process. The wire
                // layer therefore returns a typed lifecycle action after writing
                // the successful response instead of allowing that artificial
                // sleep loop to keep the emulated process alive:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L358-L405
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1094-L1099
                SelfControllerCommand::Exit => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    applet_data(request.token, &[])
                }
                // LockExit and UnlockExit mutate the application applet's shared
                // exit-deferral state and carry no input or output payload:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1094-L1099
                command @ (SelfControllerCommand::LockExit | SelfControllerCommand::UnlockExit) => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    session.set_exit_locked(command == SelfControllerCommand::LockExit);
                    applet_data(request.token, &[])
                }
                // GetLibraryAppletLaunchableEvent has no input and returns one
                // copied, manual-clear event handle (`autoclear=false`).
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c
                // https://switchbrew.org/w/index.php?title=Applet_Manager_services&oldid=14546#GetLibraryAppletLaunchableEvent
                SelfControllerCommand::GetLibraryAppletLaunchableEvent => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let handle = match process
                        .handles_mut()
                        .insert(session.library_applet_launchable_event())
                    {
                        Ok(handle) => handle,
                        Err(_) => {
                            return Err(IpcWireError::HostResourceExhausted(
                                "installing the library-applet launchable event handle",
                            ));
                        }
                    };
                    log::debug!(
                        "ISelfController returned library-applet launchable event handle {handle:#x}"
                    );
                    Ok((
                        encode_domain_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &[],
                            &[handle],
                            &[],
                        )?,
                        Some(handle),
                    ))
                }
                SelfControllerCommand::CreateManagedDisplayLayer => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let Some(layer) = video_system.create_layer(1) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                        );
                    };
                    applet_data(request.token, &layer.id.to_le_bytes())
                }
                // These state-mutating command layouts follow pinned libnx:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1113-L1125
                SelfControllerCommand::SetOperationModeChangedNotification => {
                    let Some(enabled) = applet_request_bool(request.data, hipc) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    session.set_operation_mode_changed_notification(enabled);
                    applet_data(request.token, &[])
                }
                SelfControllerCommand::SetPerformanceModeChangedNotification => {
                    let Some(enabled) = applet_request_bool(request.data, hipc) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    session.set_performance_mode_changed_notification(enabled);
                    applet_data(request.token, &[])
                }
                SelfControllerCommand::SetFocusHandlingMode => {
                    let Some(mode) = request.data.get(..3) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    if request.data[3..].iter().any(|byte| *byte != 0) || has_ipc_descriptors(hipc)
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    session.set_focus_handling_mode([mode[0] != 0, mode[1] != 0, mode[2] != 0]);
                    applet_data(request.token, &[])
                }
                // SetOutOfFocusSuspendingEnabled completes the focus policy set by
                // command 13. It has no immediate scheduling effect while the
                // emulated application remains permanently in focus.
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L518-L553
                SelfControllerCommand::SetOutOfFocusSuspendingEnabled => {
                    let Some(enabled) = applet_request_bool(request.data, hipc) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    session.set_out_of_focus_suspending_enabled(enabled);
                    applet_data(request.token, &[])
                }
            }
        }
        AppletObject::WindowController => {
            let Some(command) = WindowControllerCommand::decode(request.command_id) else {
                return unsupported_service_command("IWindowController", request.command_id);
            };
            match command {
                WindowControllerCommand::GetAppletResourceUserId => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return unsupported_service_command(
                            "IWindowController",
                            request.command_id,
                        );
                    }
                    applet_data(request.token, &process.process_id().to_le_bytes())
                }
                // AcquireForegroundRights has no input/output payload:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1271-L1279
                WindowControllerCommand::AcquireForegroundRights => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    session.acquire_foreground_rights();
                    applet_data(request.token, &[])
                }
            }
        }
        AppletObject::ApplicationFunctions => {
            let Some(command) = ApplicationFunctionsCommand::decode(request.command_id) else {
                return unsupported_service_command("IApplicationFunctions", request.command_id);
            };
            match command {
                // SetTerminateResult stores the application result in AM. It does
                // not terminate the process or replace the kernel exit code.
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L2672-L2685
                ApplicationFunctionsCommand::SetTerminateResult => {
                    let Some(result) = request_u32(request.data, 0) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    if request.data[4..].iter().any(|byte| *byte != 0) || has_ipc_descriptors(hipc)
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    session.set_termination_result(result);
                    applet_data(request.token, &[])
                }
                ApplicationFunctionsCommand::NotifyRunning => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return unsupported_service_command(
                            "IApplicationFunctions",
                            request.command_id,
                        );
                    }
                    applet_data(request.token, &[1])
                }
            }
        }
        AppletObject::LibraryAppletCreator => {
            let Some(command) = LibraryAppletCreatorCommand::decode(request.command_id) else {
                return unsupported_service_command("ILibraryAppletCreator", request.command_id);
            };
            match command {
                // CreateLibraryApplet takes one AppletId/LibAppletMode pair and
                // returns an ILibraryAppletAccessor domain object. libnx consumes
                // that accessor immediately to obtain its state-change event:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1516-L1564
                LibraryAppletCreatorCommand::CreateLibraryApplet => {
                    if request.data.len() < 8
                        || request.data[8..].iter().any(|byte| *byte != 0)
                        || has_ipc_descriptors(hipc)
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let Some(applet_id) =
                        request_u32(request.data, 0).and_then(LibraryAppletId::from_raw)
                    else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    let Some(mode) =
                        request_u32(request.data, 4).and_then(LibraryAppletMode::from_raw)
                    else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    };
                    match session.create_library_applet(applet_id, mode) {
                        Ok(accessor_object_id) => {
                            log::debug!(
                                "ILibraryAppletCreator created {applet_id:?} in {mode:?} mode as domain object {accessor_object_id:#x}"
                            );
                            Ok((
                                encode_domain_response(
                                    request.token,
                                    HorizonIpcResult::SUCCESS,
                                    &[],
                                    &[],
                                    &[accessor_object_id],
                                )?,
                                None,
                            ))
                        }
                        Err(CreateLibraryAppletError::Busy) => {
                            Err(IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "ILibraryAppletCreator",
                                    command_id: 0,
                                    detail: "a prior library applet remains active",
                                },
                            ))
                        }
                        Err(CreateLibraryAppletError::DomainCapacityExhausted) => {
                            Err(IpcWireError::HostResourceExhausted(
                                "allocating a library-applet domain object",
                            ))
                        }
                        Err(CreateLibraryAppletError::NotDomain) => Err(IpcWireError::Internal(
                            "library applet creation escaped its domain session",
                        )),
                    }
                }
                // CreateStorage allocates one zero-filled IStorage object. The
                // signed libnx size reaches CMIF as the same 64-bit bit pattern:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1761-L1767
                LibraryAppletCreatorCommand::CreateStorage => {
                    if request.data.len() < 8
                        || request.data[8..].iter().any(|byte| *byte != 0)
                        || has_ipc_descriptors(hipc)
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let size = request_u64(request.data, 0)
                        .expect("validated CreateStorage payload length");
                    match session.create_storage(size) {
                        Ok(storage_object_id) => Ok((
                            encode_domain_response(
                                request.token,
                                HorizonIpcResult::SUCCESS,
                                &[],
                                &[],
                                &[storage_object_id],
                            )?,
                            None,
                        )),
                        Err(CreateAppletStorageError::DomainCapacityExhausted) => {
                            Err(IpcWireError::HostResourceExhausted(
                                "allocating an applet-storage domain object",
                            ))
                        }
                        Err(CreateAppletStorageError::SizeOutOfRange) => {
                            Err(IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "ILibraryAppletCreator",
                                    command_id: 10,
                                    detail: "applet storage size cannot be represented by the host",
                                },
                            ))
                        }
                        Err(CreateAppletStorageError::AllocationFailed) => {
                            Err(IpcWireError::HostResourceExhausted(
                                "allocating applet-storage backing",
                            ))
                        }
                        Err(CreateAppletStorageError::NotDomain) => Err(IpcWireError::Internal(
                            "applet storage creation escaped its domain session",
                        )),
                    }
                }
            }
        }
        AppletObject::LibraryAppletAccessor { .. } => {
            let Some(command) = LibraryAppletAccessorCommand::decode(request.command_id) else {
                return unsupported_service_command("ILibraryAppletAccessor", request.command_id);
            };
            match command {
                LibraryAppletAccessorCommand::GetAppletStateChangedEvent => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let Some(event) = session.library_applet_state_changed_event(object_id) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                        );
                    };
                    let handle = process.handles_mut().insert(event).map_err(|_| {
                        IpcWireError::HostResourceExhausted(
                            "installing a library-applet event handle",
                        )
                    })?;
                    Ok((
                        encode_domain_response(
                            request.token,
                            HorizonIpcResult::SUCCESS,
                            &[],
                            &[handle],
                            &[],
                        )?,
                        Some(handle),
                    ))
                }
                // Start freezes the queued inputs into one launch request. Until
                // Nixe has a graphical system-applet host, Error applets become a
                // typed fatal diagnostic rather than hanging on their state event
                // or pretending that the dialog completed.
                LibraryAppletAccessorCommand::Start => {
                    if !request.data.is_empty()
                        || has_ipc_descriptors(hipc)
                        || !input_objects.is_empty()
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let launch =
                        session
                            .prepare_library_applet_launch(object_id)
                            .map_err(|error| match error {
                                PrepareLibraryAppletLaunchError::AppletNotFound => {
                                    IpcWireError::Internal(
                                        "live library-applet accessor lost its launch state",
                                    )
                                }
                                PrepareLibraryAppletLaunchError::StorageBackingMissing => {
                                    IpcWireError::Internal(
                                        "library-applet input queue lost retained storage backing",
                                    )
                                }
                                PrepareLibraryAppletLaunchError::AllocationFailed => {
                                    IpcWireError::HostResourceExhausted(
                                        "snapshotting library-applet launch inputs",
                                    )
                                }
                            })?;
                    if launch.applet_id != LibraryAppletId::Error {
                        return Err(IpcWireError::UnsupportedService(
                            UnsupportedServiceOperation::CommandVariant {
                                service: "ILibraryAppletAccessor",
                                command_id: 10,
                                detail: "library-applet execution coordinator is unavailable",
                            },
                        ));
                    }
                    Err(IpcWireError::ErrorApplet(Box::new(
                        crate::ErrorAppletDiagnostic::decode(&launch.input_storages)
                            .with_launch_mode(launch.mode.as_raw()),
                    )))
                }
                // PushInData transfers one IStorage domain object into the
                // library applet's ordered input channel. libnx closes its local
                // storage object after dispatch, so the channel must retain the
                // backing independently until the applet consumes or releases it.
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L779-L793
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1754-L1758
                LibraryAppletAccessorCommand::PushInData => {
                    if !request.data.is_empty()
                        || has_ipc_descriptors(hipc)
                        || input_objects.len() != 1
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    match session.push_library_applet_input_storage(object_id, input_objects[0]) {
                        Ok(()) => applet_data(request.token, &[]),
                        Err(PushLibraryAppletStorageError::AllocationFailed) => {
                            Err(IpcWireError::HostResourceExhausted(
                                "queuing library-applet input storage",
                            ))
                        }
                        Err(PushLibraryAppletStorageError::AppletNotFound) => {
                            Err(IpcWireError::Internal(
                                "live library-applet accessor lost its active applet",
                            ))
                        }
                        Err(PushLibraryAppletStorageError::StorageNotFound) => {
                            Err(IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "ILibraryAppletAccessor",
                                    command_id: 100,
                                    detail: "input domain object is not a live IStorage",
                                },
                            ))
                        }
                    }
                }
            }
        }
        AppletObject::Storage { storage_id } => {
            let Some(StorageCommand::Open) = StorageCommand::decode(request.command_id) else {
                return unsupported_service_command("IStorage", request.command_id);
            };
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return unsupported_service_command("IStorage", request.command_id);
            }
            let accessor_object_id = match session.open_storage_accessor(storage_id) {
                Ok(object_id) => object_id,
                Err(OpenAppletStorageAccessorError::StorageNotFound) => {
                    return Err(IpcWireError::Internal(
                        "live applet storage object lost its backing",
                    ));
                }
                Err(OpenAppletStorageAccessorError::DomainCapacityExhausted) => {
                    return Err(IpcWireError::HostResourceExhausted(
                        "allocating an applet-storage accessor domain object",
                    ));
                }
                Err(OpenAppletStorageAccessorError::ObjectIdExhausted) => {
                    return Err(IpcWireError::HostResourceExhausted(
                        "allocating an applet-storage accessor object ID",
                    ));
                }
            };
            Ok((
                encode_domain_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &[],
                    &[],
                    &[accessor_object_id],
                )?,
                None,
            ))
        }
        AppletObject::StorageAccessor { storage_id } => {
            let Some(command) = StorageAccessorCommand::decode(request.command_id) else {
                return unsupported_service_command("IStorageAccessor", request.command_id);
            };
            match command {
                StorageAccessorCommand::GetSize => {
                    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                        return unsupported_service_command("IStorageAccessor", request.command_id);
                    }
                    let Some(size) = session.storage_size(storage_id) else {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                        );
                    };
                    applet_data(request.token, &size.to_le_bytes())
                }
                // libnx selects pointer or map-alias buffers according to the
                // transfer size and sends the byte offset as a signed 64-bit
                // value. Negative offsets retain their bit pattern and therefore
                // fail the checked storage range below:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/applet.c#L1830-L1879
                StorageAccessorCommand::Write => {
                    if request.data.len() < 8
                        || request.data[8..].iter().any(|byte| *byte != 0)
                        || hipc.pid.is_some()
                        || !hipc.copy_handles.is_empty()
                        || !hipc.move_handles.is_empty()
                        || !matches!(hipc.receive_statics, ReceiveStatics::None)
                        || !hipc.receive_buffers.is_empty()
                        || !hipc.exchange_buffers.is_empty()
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let (address, size) = one_auto_select_input(hipc)?;
                    let mut bytes = vec![0; size];
                    read_bytes(process, GuestVirtualAddress::new(address), &mut bytes)?;
                    let offset = request_u64(request.data, 0)
                        .expect("validated IStorageAccessor::Write payload length");
                    match session.write_storage(storage_id, offset, &bytes) {
                        Ok(()) => applet_data(request.token, &[]),
                        Err(AppletStorageAccessError::NotFound) => {
                            applet_error(request.token, HorizonIpcResult::CMIF_TARGET_NOT_FOUND)
                        }
                        Err(AppletStorageAccessError::OutOfRange) => {
                            Err(IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "IStorageAccessor",
                                    command_id: 10,
                                    detail: "out-of-range applet storage write",
                                },
                            ))
                        }
                    }
                }
                StorageAccessorCommand::Read => {
                    if request.data.len() < 8
                        || request.data[8..].iter().any(|byte| *byte != 0)
                        || hipc.pid.is_some()
                        || !hipc.copy_handles.is_empty()
                        || !hipc.move_handles.is_empty()
                        || !hipc.send_statics.is_empty()
                        || !hipc.send_buffers.is_empty()
                        || !hipc.exchange_buffers.is_empty()
                    {
                        return applet_error(
                            request.token,
                            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                        );
                    }
                    let (address, size) = one_auto_select_output(hipc)?;
                    let offset = request_u64(request.data, 0)
                        .expect("validated IStorageAccessor::Read payload length");
                    match session.read_storage(storage_id, offset, size) {
                        Ok(bytes) => {
                            write_bytes(process, GuestVirtualAddress::new(address), &bytes)?;
                            applet_data(request.token, &[])
                        }
                        Err(AppletStorageAccessError::NotFound) => {
                            applet_error(request.token, HorizonIpcResult::CMIF_TARGET_NOT_FOUND)
                        }
                        Err(AppletStorageAccessError::OutOfRange) => {
                            Err(IpcWireError::UnsupportedService(
                                UnsupportedServiceOperation::CommandVariant {
                                    service: "IStorageAccessor",
                                    command_id: 11,
                                    detail: "out-of-range applet storage read",
                                },
                            ))
                        }
                    }
                }
            }
        }
        AppletObject::HomeMenuFunctions
        | AppletObject::GlobalStateController
        | AppletObject::ApplicationCreator
        | AppletObject::AppletCommonFunctions
        | AppletObject::AudioController
        | AppletObject::DisplayController
        | AppletObject::DebugFunctions => {
            unsupported_service_command(applet_object_name(object), request.command_id)
        }
    }
}

fn applet_child(
    session: &AppletSession,
    token: u32,
    object: AppletObject,
    name: &'static str,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(object_id) = session.insert_object(object) else {
        return Err(IpcWireError::HostResourceExhausted(
            "allocating an applet domain child object",
        ));
    };
    log::debug!("appletOE opened {name} as domain object {object_id:#x}");
    Ok((
        encode_domain_response(token, HorizonIpcResult::SUCCESS, &[], &[], &[object_id])?,
        None,
    ))
}

fn applet_data(token: u32, data: &[u8]) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((
        encode_domain_response(token, HorizonIpcResult::SUCCESS, data, &[], &[])?,
        None,
    ))
}

fn applet_error(
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((encode_domain_response(token, result, &[], &[], &[])?, None))
}

fn applet_request_bool(data: &[u8], hipc: &HipcRequest<'_>) -> Option<bool> {
    let (&value, padding) = data.split_first()?;
    (!has_ipc_descriptors(hipc) && padding.iter().all(|byte| *byte == 0)).then_some(value != 0)
}

pub(in crate::ipc_wire) const fn applet_object_name(object: AppletObject) -> &'static str {
    match object {
        AppletObject::Root => "IApplicationProxyService",
        AppletObject::Proxy(kind) => applet_proxy_name(kind),
        AppletObject::ApplicationFunctions => "IApplicationFunctions",
        AppletObject::HomeMenuFunctions => "IHomeMenuFunctions",
        AppletObject::GlobalStateController => "IGlobalStateController",
        AppletObject::ApplicationCreator => "IApplicationCreator",
        AppletObject::AppletCommonFunctions => "IAppletCommonFunctions",
        AppletObject::LibraryAppletCreator => "ILibraryAppletCreator",
        AppletObject::CommonStateGetter => "ICommonStateGetter",
        AppletObject::SelfController => "ISelfController",
        AppletObject::WindowController => "IWindowController",
        AppletObject::AudioController => "IAudioController",
        AppletObject::DisplayController => "IDisplayController",
        AppletObject::DebugFunctions => "IDebugFunctions",
        AppletObject::LibraryAppletAccessor { .. } => "ILibraryAppletAccessor",
        AppletObject::Storage { .. } => "IStorage",
        AppletObject::StorageAccessor { .. } => "IStorageAccessor",
    }
}

const fn applet_proxy_name(kind: AppletProxyKind) -> &'static str {
    match kind {
        AppletProxyKind::Application => "IApplicationProxy",
        AppletProxyKind::SystemApplet => "ISystemAppletProxy",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_commands_follow_the_caller_role() {
        assert_eq!(
            RootCommand::decode(100).map(RootCommand::proxy_kind),
            Some(AppletProxyKind::SystemApplet)
        );
        assert_eq!(RootCommand::decode(200), None);
        assert_eq!(
            ProxyCommand::decode(AppletProxyKind::Application, 20)
                .map(|command| command.child(AppletProxyKind::Application)),
            Some(AppletObject::ApplicationFunctions)
        );
        assert_eq!(
            ProxyCommand::decode(AppletProxyKind::SystemApplet, 20)
                .map(|command| command.child(AppletProxyKind::SystemApplet)),
            Some(AppletObject::HomeMenuFunctions)
        );
        assert_eq!(ProxyCommand::decode(AppletProxyKind::Application, 23), None);
    }

    #[test]
    fn application_function_ids_decode_to_semantic_commands() {
        assert_eq!(
            ApplicationFunctionsCommand::decode(22),
            Some(ApplicationFunctionsCommand::SetTerminateResult)
        );
        assert_eq!(
            ApplicationFunctionsCommand::decode(40),
            Some(ApplicationFunctionsCommand::NotifyRunning)
        );
    }
}
