//! Synchronous request orchestration after HIPC/CMIF decoding.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

use super::io::{read_bytes, validate_writable_ram_range, write_response};
use super::message::{COMMAND_BUFFER_SIZE, CmifRequest, DomainRequest, HipcRequest};
use super::services;
use super::services::content::{dispatch_plain_object, dispatch_service};
use super::services::fsp::object_name;
use super::services::semantic_service_name;
use super::{IpcWireError, UnsupportedServiceOperation};
use nixe_memory::GuestVirtualAddress;
use nixe_runtime::ExceptionProcessContext;

use crate::object::{NetworkInterfaceObject, TimeObject};
use crate::{
    HidSystem, HorizonIpcObject, OperationMode, SettingsEnvironment, TimeEnvironment, VideoSystem,
};

#[cfg(test)]
use crate::{IpcService, IpcSession, NvDrvSession, SystemSettingsSession};

const CMIF_COMMAND_CLOSE: u16 = 2;

#[derive(Clone, Copy)]
pub(crate) struct HostSystems<'a> {
    pub video: &'a VideoSystem,
    pub hid: &'a HidSystem,
    pub settings: &'a SettingsEnvironment,
    pub diagnostics: &'a crate::HorizonDiagnostics,
    pub caller_thread_id: u64,
}

impl HorizonIpcObject {
    fn is_domain(&self) -> bool {
        match self {
            Self::Applet(session) => session.is_domain(),
            Self::SemanticService(session) => session.is_domain(),
            Self::ParentalControl(session) => session.is_domain(),
            Self::Time(session) => session.is_domain(),
            Self::NetworkInterface(session) => session.is_domain(),
            Self::Account(session) => session.is_domain(),
            Self::Bsd(session) => session.is_domain(),
            _ => false,
        }
    }

    fn diagnostic_name(&self, request: &CmifRequest<'_>) -> &'static str {
        let domain_object = match request.domain.as_ref() {
            Some(
                DomainRequest::SendMessage { object_id, .. } | DomainRequest::Close { object_id },
            ) => Some(*object_id),
            None => None,
        };
        match self {
            Self::ServiceManager(_) => "sm:",
            Self::SemanticService(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or_else(
                    || semantic_service_name(session.service()),
                    |object| object_name(&object),
                ),
            Self::SystemSettings(_) => "set:sys",
            Self::UserSettings(_) => "set",
            Self::PerformanceManager(_) => "apm",
            Self::Performance(_) => "IPerformanceSession",
            Self::Applet(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("appletOE", services::applet_object_name),
            Self::Account(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("acc:u0", |object| match object {
                    crate::object::AccountObject::BaasManagerForApplication(_) => {
                        "IManagerForApplication"
                    }
                }),
            Self::AccountManagerForApplication(_) => "IManagerForApplication",
            Self::Bsd(_) => "bsd:u",
            Self::Hid(_) => "hid",
            Self::HidAppletResource(_) => "IAppletResource",
            Self::Time(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("time:u", |object| match object {
                    TimeObject::SystemClock(_) => "ISystemClock",
                    TimeObject::SteadyClock(_) => "ISteadyClock",
                    TimeObject::TimeZone(_) => "ITimeZoneService",
                }),
            Self::SystemClock(_) => "ISystemClock",
            Self::SteadyClock(_) => "ISteadyClock",
            Self::TimeZone(_) => "ITimeZoneService",
            Self::Vi(session) => services::vi_object_name(session.kind()),
            Self::NvDrv(_) => "nvdrv",
            Self::LogManager(_) => "lm",
            Self::Logger(_) => "ILogger",
            Self::ParentalControl(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("pctl", |_| "IParentalControlService"),
            Self::ParentalControlService(_) => "IParentalControlService",
            Self::NetworkInterface(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("nifm:u", |object| match object {
                    NetworkInterfaceObject::GeneralService(_) => "IGeneralService",
                }),
            Self::NetworkGeneralService(_) => "IGeneralService",
            Self::SemanticObject(object) => object_name(object),
        }
    }
}

pub(crate) fn send_sync_request(
    process: &mut ExceptionProcessContext<'_>,
    tls: GuestVirtualAddress,
    handle: u32,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<SyncRequestResult, IpcWireError> {
    send_sync_request_from_buffer(
        process,
        tls,
        COMMAND_BUFFER_SIZE,
        handle,
        initial_operation_mode,
        time_environment,
        host_systems,
    )
}

pub(crate) fn send_sync_request_from_buffer(
    process: &mut ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    handle: u32,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: HostSystems<'_>,
) -> Result<SyncRequestResult, IpcWireError> {
    let Some(target) = process
        .handles()
        .get_as::<HorizonIpcObject>(handle)
        .cloned()
    else {
        return Ok(SyncRequestResult::InvalidHandle);
    };

    if size < COMMAND_BUFFER_SIZE {
        return Err(IpcWireError::Malformed(
            "IPC message buffer is smaller than the TLS command buffer",
        ));
    }
    validate_writable_ram_range(process, address, size)?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(size)
        .map_err(|_| IpcWireError::HostResourceExhausted("allocating the IPC command buffer"))?;
    buffer.resize(size, 0);
    read_bytes(process, address, &mut buffer)?;
    let hipc = HipcRequest::decode(&buffer).map_err(|error| IpcWireError::Malformed(error.0))?;
    let request = CmifRequest::decode(&hipc, target.is_domain()).map_err(|error| {
        if error.0 == "unsupported HIPC command type for CMIF" {
            IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
                service: "CMIF transport",
                command_id: u32::from(hipc.command_type),
                detail: error.0,
            })
        } else {
            IpcWireError::Malformed(error.0)
        }
    })?;
    let trace_service =
        log::log_enabled!(log::Level::Trace).then(|| target.diagnostic_name(&request));
    if let Some(service) = trace_service {
        let data_prefix = &request.data[..request.data.len().min(32)];
        log::trace!(
            "SendSyncRequest service={service} handle={handle:#x} type={} command={} send_pid={} descriptors={}/{}/{}/{} handles={}/{} data_len={} data_prefix={data_prefix:02x?}",
            request.command_type,
            request.command_id,
            hipc.pid.is_some(),
            hipc.send_statics.len(),
            hipc.send_buffers.len(),
            hipc.receive_buffers.len(),
            hipc.exchange_buffers.len(),
            hipc.copy_handles.len(),
            hipc.move_handles.len(),
            request.data.len(),
        );
    }
    let command_type = request.command_type;
    let command_id = request.command_id;
    let trace_completion = |outcome: SyncRequestResult| {
        if let Some(service) = trace_service {
            log::trace!(
                "SendSyncRequest completed service={service} handle={handle:#x} type={command_type} command={command_id} outcome={outcome:?}"
            );
        }
        outcome
    };

    if request.command_type == CMIF_COMMAND_CLOSE {
        // CMIF closes the server-side session protocol, while the client's
        // kernel handle remains owned until the following CloseHandle. These
        // lifetimes must stay distinct: Nintendo SDK treats a failed
        // CloseHandle after this request as a fatal invariant violation.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L195-L209
        return Ok(trace_completion(SyncRequestResult::Success));
    }
    if super::control::dispatch_control(process, address, size, handle, &target, &request)? {
        return Ok(trace_completion(SyncRequestResult::Success));
    }
    let applet_exit_requested = match &target {
        HorizonIpcObject::Applet(session) => {
            services::applet_requests_self_exit(session, &request, &hipc)
        }
        _ => false,
    };
    let (response, created_handle) = match target {
        HorizonIpcObject::ServiceManager(manager) => services::dispatch_service_manager(
            process,
            &manager,
            request,
            hipc.pid.is_some(),
            initial_operation_mode,
            time_environment,
            host_systems,
        )?,
        HorizonIpcObject::SemanticService(service) => dispatch_service(
            process,
            &service,
            request,
            &hipc,
            host_systems.diagnostics.file_system_access_log_mode(),
        )?,
        HorizonIpcObject::SystemSettings(_) => {
            services::dispatch_system_settings(process, request, &hipc.receive_statics)?
        }
        HorizonIpcObject::UserSettings(settings) => {
            services::dispatch_user_settings(process, &settings, request, &hipc)?
        }
        HorizonIpcObject::PerformanceManager(manager) => {
            services::dispatch_performance_manager(process, &manager, request)?
        }
        HorizonIpcObject::Performance(session) => {
            services::dispatch_performance_session(&session, request)?
        }
        HorizonIpcObject::Applet(applet) => {
            services::dispatch_applet(process, &applet, request, &hipc, host_systems.video)?
        }
        HorizonIpcObject::Account(account) => {
            services::dispatch_account(process, &account, request, &hipc)?
        }
        HorizonIpcObject::AccountManagerForApplication(manager) => {
            services::dispatch_account_manager_for_application(&manager, request)?
        }
        HorizonIpcObject::Bsd(session) => {
            services::dispatch_bsd(process, &session, request, &hipc)?
        }
        HorizonIpcObject::Hid(hid) => {
            services::dispatch_hid(process, &hid, host_systems.hid, request, &hipc)?
        }
        HorizonIpcObject::HidAppletResource(resource) => {
            services::dispatch_hid_applet_resource(process, &resource, request)?
        }
        HorizonIpcObject::Time(time) => services::dispatch_time(process, &time, request, &hipc)?,
        HorizonIpcObject::SystemClock(clock) => {
            services::dispatch_system_clock(&clock, request, &hipc)?
        }
        HorizonIpcObject::SteadyClock(clock) => services::dispatch_steady_clock(&clock, request)?,
        HorizonIpcObject::TimeZone(timezone) => services::dispatch_timezone(&timezone, request)?,
        HorizonIpcObject::Vi(vi) => services::dispatch_vi(process, &vi, request, &hipc)?,
        HorizonIpcObject::NvDrv(nvdrv) => match services::dispatch_nvdrv(
            process,
            &nvdrv,
            request,
            &hipc,
            host_systems.caller_thread_id,
        ) {
            Ok(response) => response,
            Err(IpcWireError::PendingNvDrv(wait)) => {
                return Ok(trace_completion(SyncRequestResult::PendingNvDrv(wait)));
            }
            Err(error) => return Err(error),
        },
        HorizonIpcObject::LogManager(_) => services::dispatch_log_manager(process, request, &hipc)?,
        HorizonIpcObject::Logger(logger) => services::dispatch_logger(
            process,
            &logger,
            request,
            &hipc,
            host_systems.diagnostics.guest_logs_level,
        )?,
        HorizonIpcObject::ParentalControl(factory) => {
            services::dispatch_parental_control(process, &factory, request, &hipc)?
        }
        HorizonIpcObject::ParentalControlService(service) => {
            services::dispatch_parental_control_service(None, &service, request, &hipc)?
        }
        HorizonIpcObject::NetworkInterface(manager) => {
            services::dispatch_network_interface(process, &manager, request, &hipc)?
        }
        HorizonIpcObject::NetworkGeneralService(service) => {
            services::dispatch_network_general_service(&service, request)?
        }
        HorizonIpcObject::SemanticObject(object) => {
            dispatch_plain_object(process, &object, request, &hipc)?
        }
    };
    if let Some(service) = trace_service {
        let response_prefix = &response[..response.len().min(64)];
        log::trace!(
            "SendSyncRequest response service={service} handle={handle:#x} type={command_type} command={command_id} bytes={} created_handle={created_handle:?} prefix={response_prefix:02x?}",
            response.len(),
        );
    }
    if let Err(error) = write_response(process, address, size, &response) {
        if let Some(handle) = created_handle {
            let _ = process.handles_mut().close(handle);
        }
        return Err(error);
    }
    let outcome = if applet_exit_requested {
        SyncRequestResult::AppletExitRequested
    } else {
        SyncRequestResult::Success
    };
    Ok(trace_completion(outcome))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SyncRequestResult {
    Success,
    InvalidHandle,
    AppletExitRequested,
    PendingNvDrv(crate::nvdrv::PendingNvHostCtrlWait),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_diagnostics_derive_service_names_from_typed_targets() {
        let request = CmifRequest {
            command_type: 4,
            command_id: 3,
            token: 0,
            context: None,
            data: &[],
            domain: None,
        };

        assert_eq!(
            HorizonIpcObject::NvDrv(NvDrvSession::default()).diagnostic_name(&request),
            "nvdrv"
        );
        assert_eq!(
            HorizonIpcObject::SemanticService(IpcSession::new(IpcService::FileSystem))
                .diagnostic_name(&request),
            "fsp-srv"
        );
        assert_eq!(
            HorizonIpcObject::SystemSettings(SystemSettingsSession::new())
                .diagnostic_name(&request),
            "set:sys"
        );
    }
}
