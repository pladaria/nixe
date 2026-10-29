//! Synchronous request orchestration after HIPC/CMIF decoding.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

use super::message::{
    CmifRequest, DomainRequest, HipcRequest, TIPC_COMMAND_CLOSE, TIPC_COMMAND_REQUEST_BASE,
    TipcRequest,
};
use super::services;
use super::services::content::{dispatch_plain_object, dispatch_service};
use super::services::fsp::object_name;
use super::services::semantic_service_name;
use super::{IpcWireError, UnsupportedServiceOperation};
use crate::ipc_wire::request::Request;

use crate::object::{NetworkInterfaceObject, TimeObject};
use crate::{
    HidSystem, HorizonIpcObject, OperationMode, SettingsEnvironment, TimeEnvironment, VideoSystem,
};

#[cfg(test)]
use crate::{IpcService, IpcSession, NvDrvSession, SystemSettingsSession};

const CMIF_COMMAND_CLOSE: u16 = 2;

#[derive(Clone)]
pub(crate) struct HostSystems {
    pub video: VideoSystem,
    pub keyboard: Option<nixe_applets::keyboard::KeyboardHost>,
    pub audio_backend: Option<std::sync::Arc<dyn nixe_audio::AudioBackend>>,
    pub audio_renderers: crate::AudioRendererManagerSession,
    pub opus: crate::OpusManagerSession,
    pub hid: HidSystem,
    pub shared_fonts: crate::SharedFontSession,
    pub system_archives: std::sync::Arc<crate::SystemArchives>,
    pub settings: SettingsEnvironment,
    pub application_language: Option<crate::SystemLanguage>,
    pub application_display_version: Option<[u8; 16]>,
    pub user_account_switch_locked: Option<bool>,
    pub save_data: Option<crate::SaveDataSystem>,
    pub diagnostics: crate::HorizonDiagnostics,
    pub caller_thread_id: u64,
    pub scheduling: std::sync::Arc<super::operation::ServiceScheduling>,
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
            Self::Ssl(session) => session.is_domain(),
            Self::LogManager(session) => session.is_domain(),
            Self::ErrorContextWriter(session) => session.domain.is_domain(),
            Self::OpusManager(session) => session.domain.is_domain(),
            Self::OpusDecoder(session) => session.domain.is_domain(),
            Self::PlayReport(session) => session.domain.is_domain(),
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
                    crate::object::AccountObject::Profile(_) => "IProfile",
                    crate::object::AccountObject::BaasManagerForApplication(_) => {
                        "IManagerForApplication"
                    }
                }),
            Self::AccountProfile(_) => "IProfile",
            Self::AccountManagerForApplication(_) => "IManagerForApplication",
            Self::Bsd(_) => "bsd:u",
            Self::Ssl(_) => "ssl",
            Self::Multimedia(_) => "mm:u",
            Self::AudioOutManager(_) => "audout:u",
            Self::AudioOut(_) => "IAudioOut",
            Self::AudioRendererManager(_) => "audren:u",
            Self::AudioRenderer(_) => "IAudioRenderer",
            Self::AudioDevice(_) => "IAudioDevice",
            Self::OpusManager(session) => {
                if domain_object
                    .and_then(|id| session.domain.object(id))
                    .is_some()
                {
                    "IHardwareOpusDecoder"
                } else {
                    "hwopus"
                }
            }
            Self::OpusDecoder(_) => "IHardwareOpusDecoder",
            Self::Hid(_) => "hid",
            Self::SharedFont(_) => "pl:u",
            Self::Irs(_) => "irs",
            Self::HidAppletResource(_) => "IAppletResource",
            Self::HidActiveVibrationDeviceList(_) => "IActiveVibrationDeviceList",
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
            Self::ErrorContextWriter(_) => "ectx:aw",
            Self::PlayReport(_) => "prepo:u",
            Self::ParentalControl(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("pctl", |_| "IParentalControlService"),
            Self::ParentalControlService(_) => "IParentalControlService",
            Self::NetworkInterface(session) => domain_object
                .and_then(|object_id| session.object(object_id))
                .map_or("nifm:u", |object| match object {
                    NetworkInterfaceObject::GeneralService(_) => "IGeneralService",
                    NetworkInterfaceObject::Request(_) => "IRequest",
                }),
            Self::NetworkGeneralService(_) => "IGeneralService",
            Self::NetworkRequest(_) => "IRequest",
            Self::SemanticObject(object) => object_name(object),
        }
    }
}

pub(crate) async fn send_sync_request_from_buffer(
    process: &mut Request,
    handle: u32,
    initial_operation_mode: OperationMode,
    time_environment: &TimeEnvironment,
    host_systems: &HostSystems,
    turn: super::operation::ServiceTurn,
) -> Result<SyncRequestResult, IpcWireError> {
    let _trace = nixe_memory::trace::span(
        "ipc_dispatch",
        host_systems.caller_thread_id,
        u64::from(handle),
        process.command_size() as u64,
    );
    let Some(target) = process.target().downcast::<HorizonIpcObject>() else {
        return Ok(SyncRequestResult::InvalidHandle);
    };
    let buffer = process.read_command().await?;
    let hipc = HipcRequest::decode(&buffer).map_err(|error| IpcWireError::Malformed(error.0))?;
    process.accept_descriptors(&hipc)?;
    // Eden locks the shared service handler while handling a request. Keep
    // the same order with an owned turn, releasing the host core on waits.
    // https://git.eden-emu.dev/eden-emu/eden/src/branch/master/src/core/hle/service/service.cpp
    (&turn).await;
    process.begin_service()?;
    if hipc.command_type >= TIPC_COMMAND_CLOSE {
        let HorizonIpcObject::ServiceManager(manager) = target.as_ref() else {
            return Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::CommandVariant {
                    service: "TIPC transport",
                    command_id: u32::from(hipc.command_type),
                    detail: "TIPC is only implemented for sm:",
                },
            ));
        };
        if hipc.command_type == TIPC_COMMAND_CLOSE {
            if hipc.data_word_bytes() != 0 || super::io::has_ipc_descriptors(&hipc) {
                return Err(IpcWireError::Malformed(
                    "TIPC session close contains payload or descriptors",
                ));
            }
            return Ok(SyncRequestResult::Success);
        }
        debug_assert!(hipc.command_type >= TIPC_COMMAND_REQUEST_BASE);
        let request =
            TipcRequest::decode(&hipc).map_err(|error| IpcWireError::Malformed(error.0))?;
        let _command_trace = nixe_memory::trace::span(
            "ipc_tipc_dispatch",
            host_systems.caller_thread_id,
            u64::from(handle),
            u64::from(request.command_id),
        );
        if super::io::has_ipc_descriptors_other_than_pid(&hipc) {
            return Err(IpcWireError::Malformed(
                "sm: TIPC request contains unexpected descriptors",
            ));
        }
        log::trace!(
            "SendSyncRequest service=sm: transport=TIPC type={} command={} send_pid={} data_len={}",
            hipc.command_type,
            request.command_id,
            hipc.pid.is_some(),
            request.data.len()
        );
        let (response, _) = services::dispatch_service_manager(
            process,
            manager,
            services::ServiceManagerRequest::Tipc(request),
            hipc.pid.is_some(),
            initial_operation_mode,
            time_environment,
            host_systems,
        )?;
        process.write_response(&response).await?;
        process.publish_handles()?;
        return Ok(SyncRequestResult::Success);
    }
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
    let _command_trace = nixe_memory::trace::span(
        "ipc_cmif_dispatch",
        host_systems.caller_thread_id,
        u64::from(handle),
        u64::from(request.command_id),
    );
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
    if super::control::dispatch_control(process, handle, &target, &request).await? {
        process.publish_handles()?;
        return Ok(trace_completion(SyncRequestResult::Success));
    }
    let applet_exit_requested = match target.as_ref() {
        HorizonIpcObject::Applet(session) => {
            services::applet_requests_self_exit(session, &request, &hipc)
        }
        _ => false,
    };
    let (response, created_handle) = match target.as_ref() {
        HorizonIpcObject::ServiceManager(manager) => services::dispatch_service_manager(
            process,
            manager,
            services::ServiceManagerRequest::Cmif(request),
            hipc.pid.is_some(),
            initial_operation_mode,
            time_environment,
            host_systems,
        )?,
        HorizonIpcObject::SemanticService(service) => {
            dispatch_service(
                process,
                service,
                request,
                &hipc,
                host_systems.diagnostics.file_system_access_log_mode(),
                host_systems.save_data.as_ref(),
                &host_systems.system_archives,
            )
            .await?
        }
        HorizonIpcObject::SystemSettings(_) => {
            services::dispatch_system_settings(process, request, &hipc.receive_statics).await?
        }
        HorizonIpcObject::UserSettings(settings) => {
            services::dispatch_user_settings(process, settings, request, &hipc).await?
        }
        HorizonIpcObject::PerformanceManager(manager) => services::dispatch_performance_manager(
            process,
            manager,
            request,
            initial_operation_mode,
        )?,
        HorizonIpcObject::Performance(session) => {
            services::dispatch_performance_session(session, request)?
        }
        HorizonIpcObject::Applet(applet) => {
            services::dispatch_applet(process, applet, request, &hipc, host_systems).await?
        }
        HorizonIpcObject::Account(account) => {
            services::dispatch_account(
                process,
                account,
                request,
                &hipc,
                host_systems.user_account_switch_locked,
            )
            .await?
        }
        HorizonIpcObject::AccountProfile(profile) => {
            services::dispatch_account_profile(*profile, request, &hipc, false)?
        }
        HorizonIpcObject::AccountManagerForApplication(manager) => {
            services::dispatch_account_manager_for_application(manager, request)?
        }
        HorizonIpcObject::Bsd(session) => services::dispatch_bsd(process, session, request, &hipc)?,
        HorizonIpcObject::Ssl(session) => services::dispatch_ssl(session, request, &hipc)?,
        HorizonIpcObject::Multimedia(session) => {
            services::dispatch_multimedia(session, request, &hipc)?
        }
        HorizonIpcObject::AudioOutManager(session) => {
            services::dispatch_audio_out_manager(process, session, request, &hipc).await?
        }
        HorizonIpcObject::OpusManager(session) => {
            services::dispatch_opus_manager(process, session, request, &hipc).await?
        }
        HorizonIpcObject::PlayReport(session) => {
            services::dispatch_play_report(process, session, request, &hipc).await?
        }
        HorizonIpcObject::OpusDecoder(session) => {
            services::dispatch_opus_decoder(process, session, request, &hipc).await?
        }
        HorizonIpcObject::AudioOut(session) => {
            services::dispatch_audio_out(process, session, request, &hipc).await?
        }
        HorizonIpcObject::AudioRendererManager(session) => {
            services::dispatch_audio_renderer_manager(
                process,
                session,
                host_systems.audio_backend.as_ref(),
                request,
                &hipc,
            )
            .await?
        }
        HorizonIpcObject::AudioRenderer(session) => {
            services::dispatch_audio_renderer(
                process,
                session,
                host_systems.audio_backend.as_ref(),
                request,
                &hipc,
            )
            .await?
        }
        HorizonIpcObject::AudioDevice(session) => {
            services::dispatch_audio_device(process, session, request, &hipc).await?
        }
        HorizonIpcObject::Hid(hid) => {
            services::dispatch_hid(process, hid, &host_systems.hid, request, &hipc).await?
        }
        HorizonIpcObject::SharedFont(session) => {
            services::dispatch_shared_font(process, session, request, &hipc)?
        }
        HorizonIpcObject::Irs(session) => {
            services::dispatch_irs(process, session, request, &hipc, &host_systems.scheduling)
                .await?
        }
        HorizonIpcObject::HidAppletResource(resource) => {
            services::dispatch_hid_applet_resource(process, resource, request)?
        }
        HorizonIpcObject::HidActiveVibrationDeviceList(list) => {
            services::dispatch_hid_active_vibration_device_list(list, request, &hipc)?
        }
        HorizonIpcObject::Time(time) => {
            services::dispatch_time(process, time, request, &hipc, &host_systems.scheduling).await?
        }
        HorizonIpcObject::SystemClock(clock) => {
            services::dispatch_system_clock(clock, request, &hipc, &host_systems.scheduling).await?
        }
        HorizonIpcObject::SteadyClock(clock) => services::dispatch_steady_clock(clock, request)?,
        HorizonIpcObject::TimeZone(timezone) => services::dispatch_timezone(timezone, request)?,
        HorizonIpcObject::Vi(vi) => services::dispatch_vi(process, vi, request, &hipc).await?,
        HorizonIpcObject::NvDrv(nvdrv) => {
            services::dispatch_nvdrv(
                process,
                nvdrv,
                request,
                &hipc,
                host_systems.caller_thread_id,
                time_environment.clock(),
                &host_systems.scheduling,
            )
            .await?
        }
        HorizonIpcObject::LogManager(manager) => {
            services::dispatch_log_manager(
                process,
                manager,
                request,
                &hipc,
                host_systems.diagnostics.guest_logs_level,
            )
            .await?
        }
        HorizonIpcObject::Logger(logger) => {
            services::dispatch_logger(
                process,
                logger,
                request,
                &hipc,
                host_systems.diagnostics.guest_logs_level,
            )
            .await?
        }
        HorizonIpcObject::ParentalControl(factory) => {
            services::dispatch_parental_control(process, factory, request, &hipc)?
        }
        HorizonIpcObject::ParentalControlService(service) => {
            services::dispatch_parental_control_service(None, service, request, &hipc)?
        }
        HorizonIpcObject::NetworkInterface(manager) => {
            services::dispatch_network_interface(process, manager, request, &hipc)?
        }
        HorizonIpcObject::ErrorContextWriter(session) => {
            services::dispatch_error_context_writer(session, request, &hipc)?
        }
        HorizonIpcObject::NetworkGeneralService(service) => {
            services::dispatch_network_general_service(process, None, service, request, &hipc)?
        }
        HorizonIpcObject::NetworkRequest(session) => {
            services::dispatch_network_request(process, false, session, request, &hipc)?
        }
        HorizonIpcObject::SemanticObject(object) => {
            dispatch_plain_object(process, object, request, &hipc).await?
        }
    };
    if let Some(service) = trace_service {
        let response_prefix = &response[..response.len().min(64)];
        log::trace!(
            "SendSyncRequest response service={service} handle={handle:#x} type={command_type} command={command_id} bytes={} created_handle={created_handle:?} prefix={response_prefix:02x?}",
            response.len(),
        );
    }
    process.write_response(&response).await?;
    process.publish_handles()?;
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
            HorizonIpcObject::NvDrv(NvDrvSession::new(
                crate::test_memory::machine(),
                &crate::test_memory::kernel()
            ))
            .diagnostic_name(&request),
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
