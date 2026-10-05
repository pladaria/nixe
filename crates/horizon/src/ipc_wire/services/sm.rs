use super::prelude::*;
use crate::ipc_wire::message::{TipcRequest, encode_tipc_response};

pub(in crate::ipc_wire) enum ServiceManagerRequest<'a> {
    Cmif(CmifRequest<'a>),
    Tipc(TipcRequest<'a>),
}

impl ServiceManagerRequest<'_> {
    fn command_id(&self) -> u32 {
        match self {
            Self::Cmif(request) => request.command_id,
            Self::Tipc(request) => request.command_id,
        }
    }

    fn data(&self) -> &[u8] {
        match self {
            Self::Cmif(request) => request.data,
            Self::Tipc(request) => request.data,
        }
    }

    fn encode_response(
        &self,
        result: HorizonIpcResult,
        handle: Option<u32>,
    ) -> Result<Vec<u8>, IpcWireError> {
        match self {
            Self::Cmif(request) => encode_response(request.token, result, &[], handle),
            Self::Tipc(request) => {
                // TIPC response layout comes from the command signature even
                // on failure: GetService always reserves one moved handle.
                // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libstratosphere/include/stratosphere/tipc/impl/tipc_impl_command_serialization.hpp
                let handles = [handle.unwrap_or(0)];
                encode_tipc_response(
                    result.raw(),
                    &[],
                    if request.command_id == 1 {
                        &handles
                    } else {
                        &[]
                    },
                )
                .map_err(|error| IpcWireError::Malformed(error.0))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServiceKind {
    UserSettings,
    SystemSettings,
    Performance,
    Applet,
    Hid,
    Irs,
    Time,
    Account,
    Bsd,
    Ssl,
    AudioOut,
    Vi(ViServiceKind),
    NvDrv,
    LogManager,
    ErrorContextWriter,
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
            b"irs" => Some(Self::Irs),
            b"time:u" => Some(Self::Time),
            b"acc:u0" => Some(Self::Account),
            b"bsd:u" => Some(Self::Bsd),
            b"ssl" => Some(Self::Ssl),
            b"audout:u" => Some(Self::AudioOut),
            b"nvdrv" | b"nvdrv:a" | b"nvdrv:s" => Some(Self::NvDrv),
            b"lm" => Some(Self::LogManager),
            // Public ECTX application-writer service registration:
            // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/glue/ectx.cpp
            b"ectx:aw" => Some(Self::ErrorContextWriter),
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
    request: ServiceManagerRequest<'_>,
    sent_pid: bool,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = ServiceManagerCommand::decode(request.command_id()) else {
        return unsupported_service_command("sm:", request.command_id());
    };
    match command {
        ServiceManagerCommand::RegisterClient => {
            let valid_payload = match &request {
                ServiceManagerRequest::Cmif(request) => request.data.len() >= 8,
                ServiceManagerRequest::Tipc(request) => request.data.is_empty(),
            };
            if !sent_pid || !valid_payload {
                return Ok((
                    request.encode_response(HorizonIpcResult::SM_INVALID_CLIENT, None)?,
                    None,
                ));
            }
            manager.register_client();
            log::debug!(
                "sm:RegisterClient associated process {}",
                process.process_id()
            );
            Ok((
                request.encode_response(HorizonIpcResult::SUCCESS, None)?,
                None,
            ))
        }
        ServiceManagerCommand::GetServiceHandle => {
            if !manager.is_registered() {
                return Ok((
                    request.encode_response(HorizonIpcResult::SM_INVALID_CLIENT, None)?,
                    None,
                ));
            }
            let Some(encoded_name) = request.data().get(..8) else {
                return Ok((
                    request.encode_response(HorizonIpcResult::SM_INVALID_SERVICE_NAME, None)?,
                    None,
                ));
            };
            let Some(name) = decode_service_name(encoded_name) else {
                return Ok((
                    request.encode_response(HorizonIpcResult::SM_INVALID_SERVICE_NAME, None)?,
                    None,
                ));
            };
            log::debug!(
                "sm:GetService requested {:?}",
                String::from_utf8_lossy(name)
            );
            if !process.mounts().allows_service(name) {
                return service_response(&request, HorizonIpcResult::SM_NOT_ALLOWED, None);
            }
            let Some(service) = ServiceKind::from_name(name) else {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::Connect { name: name.into() },
                ));
            };
            connect_service(
                process,
                manager,
                &request,
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
    request: &ServiceManagerRequest<'_>,
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
        ServiceKind::Hid => host_systems
            .hid
            .shared_memory(process.memory())
            .and_then(|memory| {
                process
                    .handles_mut()
                    .insert(HorizonIpcObject::Hid(HidSession::new(memory)))
            }),
        ServiceKind::Irs => host_systems
            .hid
            .infrared_session(process.memory())
            .and_then(|session| process.handles_mut().insert(HorizonIpcObject::Irs(session))),
        ServiceKind::Time => {
            time_environment
                .create_service(process.memory())
                .and_then(|session| {
                    process
                        .handles_mut()
                        .insert(HorizonIpcObject::Time(session))
                })
        }
        // libnx opens acc:u0 for application account sessions. Retain the
        // real session identity while unsupported commands remain fail-fast.
        ServiceKind::Account => process
            .handles_mut()
            .insert(HorizonIpcObject::Account(AccountSession::new())),
        ServiceKind::Bsd => process
            .handles_mut()
            .insert(HorizonIpcObject::Bsd(manager.bsd_session())),
        ServiceKind::Ssl => process
            .handles_mut()
            .insert(HorizonIpcObject::Ssl(SslSession::new())),
        ServiceKind::AudioOut => process
            .handles_mut()
            .insert(HorizonIpcObject::AudioOutManager(
                crate::AudioOutManagerSession(host_systems.audio_backend.cloned()),
            )),
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
        ServiceKind::ErrorContextWriter => process
            .handles_mut()
            .insert(HorizonIpcObject::ErrorContextWriter(Default::default())),
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
            service_response(request, HorizonIpcResult::SUCCESS, Some(handle))
        }
        Err(_) if host_resource_failure => Err(IpcWireError::HostResourceExhausted(
            "installing a service handle",
        )),
        Err(_) => service_response(request, HorizonIpcResult::SM_OUT_OF_SESSIONS, None),
    }
}

fn service_response(
    request: &ServiceManagerRequest<'_>,
    result: HorizonIpcResult,
    handle: Option<u32>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((request.encode_response(result, handle)?, handle))
}
