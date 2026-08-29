use super::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServiceKind {
    UserSettings,
    SystemSettings,
    Performance,
    Applet,
    Hid,
    Time,
    Account,
    Bsd,
    Vi(ViServiceKind),
    NvDrv,
    LogManager,
    ParentalControl,
    NetworkInterface,
    Semantic(IpcService),
}

impl ServiceKind {
    fn from_name(name: &[u8]) -> Option<Self> {
        match name {
            b"set" => Some(Self::UserSettings),
            b"set:sys" => Some(Self::SystemSettings),
            b"apm" => Some(Self::Performance),
            b"appletOE" => Some(Self::Applet),
            b"hid" => Some(Self::Hid),
            b"time:u" => Some(Self::Time),
            b"acc:u0" => Some(Self::Account),
            b"bsd:u" => Some(Self::Bsd),
            b"nvdrv" | b"nvdrv:a" | b"nvdrv:s" => Some(Self::NvDrv),
            b"lm" => Some(Self::LogManager),
            b"pctl" | b"pctl:a" | b"pctl:r" | b"pctl:s" => Some(Self::ParentalControl),
            b"nifm:u" => Some(Self::NetworkInterface),
            _ => ViServiceKind::from_name(name)
                .map(Self::Vi)
                .or_else(|| IpcService::from_name(name).map(Self::Semantic)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServiceManagerCommand {
    RegisterClient,
    GetServiceHandle,
}

impl ServiceManagerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::RegisterClient),
            1 => Some(Self::GetServiceHandle),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_service_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &ServiceManagerSession,
    request: CmifRequest<'_>,
    sent_pid: bool,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = ServiceManagerCommand::decode(request.command_id) else {
        return unsupported_service_command("sm:", request.command_id);
    };
    match command {
        ServiceManagerCommand::RegisterClient => {
            if !sent_pid || request.data.len() < 8 {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_CLIENT,
                        &[],
                        None,
                    )?,
                    None,
                ));
            }
            manager.register_client();
            log::debug!(
                "sm:RegisterClient associated process {}",
                process.process_id()
            );
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        ServiceManagerCommand::GetServiceHandle => {
            if !manager.is_registered() {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_CLIENT,
                        &[],
                        None,
                    )?,
                    None,
                ));
            }
            let Some(encoded_name) = request.data.get(..8) else {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_SERVICE_NAME,
                        &[],
                        None,
                    )?,
                    None,
                ));
            };
            let Some(name) = decode_service_name(encoded_name) else {
                return Ok((
                    encode_response(
                        request.token,
                        HorizonIpcResult::SM_INVALID_SERVICE_NAME,
                        &[],
                        None,
                    )?,
                    None,
                ));
            };
            log::debug!(
                "sm:GetService requested {:?}",
                String::from_utf8_lossy(name)
            );
            if !process.mounts().allows_service(name) {
                return service_response(request.token, HorizonIpcResult::SM_NOT_ALLOWED, None);
            }
            let Some(service) = ServiceKind::from_name(name) else {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::Connect { name: name.into() },
                ));
            };
            connect_service(
                process,
                manager,
                request.token,
                service,
                initial_operation_mode,
                time_environment,
                host_systems,
            )
        }
    }
}

fn connect_service(
    process: &mut ExceptionProcessContext<'_>,
    manager: &ServiceManagerSession,
    token: u32,
    service: ServiceKind,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let host_resource_failure = matches!(
        service,
        ServiceKind::Account | ServiceKind::Vi(_) | ServiceKind::NvDrv
    );
    let handle = match service {
        ServiceKind::UserSettings => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::UserSettings(UserSettingsSession::new(
                    host_systems.settings.clone(),
                )))
        }
        ServiceKind::SystemSettings => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::SystemSettings(
                    SystemSettingsSession::new(),
                ))
        }
        ServiceKind::Performance => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::PerformanceManager(
                    PerformanceManagerSession::new(),
                ))
        }
        ServiceKind::Applet => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::Applet(AppletSession::new(
                    initial_operation_mode,
                )))
        }
        ServiceKind::Hid => process
            .handles_mut()
            .insert(HorizonIpcObject::Hid(HidSession::new(
                host_systems.hid.shared_memory(),
            ))),
        ServiceKind::Time => time_environment.create_service().and_then(|session| {
            process
                .handles_mut()
                .insert(HorizonIpcObject::Time(session))
        }),
        // libnx opens acc:u0 for application account sessions. Retain the
        // real session identity while unsupported commands remain fail-fast.
        ServiceKind::Account => process
            .handles_mut()
            .insert(HorizonIpcObject::Account(AccountSession::new())),
        ServiceKind::Bsd => process
            .handles_mut()
            .insert(HorizonIpcObject::Bsd(manager.bsd_session())),
        ServiceKind::Vi(kind) => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::Vi(ViSession::new(
                    ViObjectKind::Root(kind),
                    host_systems.video.clone(),
                )))
        }
        ServiceKind::NvDrv => process
            .handles_mut()
            .insert(HorizonIpcObject::NvDrv(host_systems.video.nvdrv())),
        ServiceKind::LogManager => process
            .handles_mut()
            .insert(HorizonIpcObject::LogManager(LogManagerSession::new())),
        ServiceKind::ParentalControl => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::ParentalControl(
                    ParentalControlFactorySession::new(),
                ))
        }
        ServiceKind::NetworkInterface => {
            process
                .handles_mut()
                .insert(HorizonIpcObject::NetworkInterface(
                    NetworkInterfaceManagerSession::new(),
                ))
        }
        ServiceKind::Semantic(service) => process
            .handles_mut()
            .insert(HorizonIpcObject::SemanticService(IpcSession::new(service))),
    };
    match handle {
        Ok(handle) => {
            log::debug!("sm:GetService returned session handle {handle:#x}");
            service_response(token, HorizonIpcResult::SUCCESS, Some(handle))
        }
        Err(_) if host_resource_failure => Err(IpcWireError::HostResourceExhausted(
            "installing a service handle",
        )),
        Err(_) => service_response(token, HorizonIpcResult::SM_OUT_OF_SESSIONS, None),
    }
}

fn service_response(
    token: u32,
    result: HorizonIpcResult,
    handle: Option<u32>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((encode_response(token, result, &[], handle)?, handle))
}
