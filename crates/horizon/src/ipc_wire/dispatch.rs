//! Wire ABI dispatch on the owning service worker. Request resources were
//! captured by acceptance; no CPU execution context is borrowed here.

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
use crate::ipc_wire::resources::ServiceResources;

use crate::HorizonIpcObject;
use crate::object::{NetworkInterfaceObject, TimeObject};

#[cfg(test)]
use crate::{IpcService, IpcSession, NvDrvSession, SystemSettingsSession};

const CMIF_COMMAND_CLOSE: u16 = 2;

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
            Self::Friends(session) => session.domain.is_domain(),
            Self::Bcat(session) => session.domain.is_domain(),
            Self::Nfp(session) => session.domain.is_domain(),
            Self::Nim(session) => session.domain.is_domain(),
            Self::Ngc(session) => session.domain.is_domain(),
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
                .map_or("appletOE", |object| services::applet_object_name(&object)),
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
            Self::AudioUser => "aud:u",
            Self::StreamPlay => "stpl:u",
            Self::StreamPlayUser(_) => "IUserShimScopedObject",
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
            Self::Ngc(_) => "ngc:u",
            Self::Nim(session) => session
                .target(domain_object)
                .map_or("nim:eca", |object| object.name()),
            Self::Nfp(session) => session
                .target(domain_object)
                .map_or("nfp:user", |object| object.name()),
            Self::Bcat(session) => session
                .target(domain_object)
                .map_or("bcat:u", |object| object.name()),
            Self::Friends(session) => session
                .target(domain_object)
                .map_or("friend:u", |object| object.name()),
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
            Self::NvDrv2(_) => "NvDrv2",
            Self::Gsv(_) => "GSV UserSession",
            Self::Csrng => "csrng",
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

pub(crate) async fn execute_request(
    process: &mut Request,
    handle: u32,
    resources: &ServiceResources,
    hipc: &HipcRequest<'_>,
    target: &HorizonIpcObject,
) -> Result<SyncRequestResult, IpcWireError> {
    let initial_operation_mode = resources.system.operation_mode;
    let time_environment = &resources.system.time;
    let _trace = nixe_memory::trace::span(
        "ipc_dispatch",
        process.thread_id(),
        u64::from(handle),
        process.command_size() as u64,
    );
    process.begin_service()?;
    if hipc.command_type >= TIPC_COMMAND_CLOSE {
        if hipc.command_type == TIPC_COMMAND_CLOSE {
            if hipc.data_word_bytes() != 0 || super::io::has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "TIPC session close contains payload or descriptors",
                ));
            }
            return Ok(SyncRequestResult::SessionClosed);
        }
        debug_assert!(hipc.command_type >= TIPC_COMMAND_REQUEST_BASE);
        let request =
            TipcRequest::decode(hipc).map_err(|error| IpcWireError::Malformed(error.0))?;
        let _command_trace = nixe_memory::trace::span(
            "ipc_tipc_dispatch",
            process.thread_id(),
            u64::from(handle),
            u64::from(request.command_id),
        );
        log::trace!(
            "TIPC request handle={handle:#x} type={} command={} send_pid={} handles={}/{} buffers={}/{}/{} data={:02x?}",
            hipc.command_type,
            request.command_id,
            hipc.pid.is_some(),
            hipc.copy_handles.len(),
            hipc.move_handles.len(),
            hipc.send_buffers.len(),
            hipc.receive_buffers.len(),
            hipc.exchange_buffers.len(),
            request.data
        );
        let response = match target {
            HorizonIpcObject::ServiceManager(manager) => {
                if super::io::has_ipc_descriptors_other_than_pid(hipc) {
                    return Err(IpcWireError::Malformed(
                        "sm: TIPC request contains unexpected descriptors",
                    ));
                }
                services::dispatch_service_manager(
                    process,
                    manager,
                    services::ServiceManagerRequest::Tipc(request),
                    hipc.pid.is_some(),
                    resources,
                )?
                .0
            }
            HorizonIpcObject::Gsv(session) => {
                services::dispatch_gsv(process, session, request, hipc).await?
            }
            _ => {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandVariant {
                        service: "TIPC transport",
                        command_id: request.command_id,
                        detail: "service does not implement TIPC",
                    },
                ));
            }
        };
        process.write_response(response).await?;
        process.publish_handles()?;
        return Ok(SyncRequestResult::Success);
    }
    let request = CmifRequest::decode(hipc, target.is_domain()).map_err(|error| {
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
        process.thread_id(),
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
        return Ok(trace_completion(SyncRequestResult::SessionClosed));
    }
    if super::control::dispatch_control(process, handle, target, &request).await? {
        process.publish_handles()?;
        return Ok(trace_completion(SyncRequestResult::Success));
    }
    let applet_exit_requested = match target {
        HorizonIpcObject::Applet(session) => {
            services::applet_requests_self_exit(session, &request, hipc)
        }
        _ => false,
    };
    let (response, created_handle) = match target {
        HorizonIpcObject::ServiceManager(manager) => services::dispatch_service_manager(
            process,
            manager,
            services::ServiceManagerRequest::Cmif(request),
            hipc.pid.is_some(),
            resources,
        )?,
        HorizonIpcObject::SemanticService(service) => {
            dispatch_service(
                process,
                service,
                request,
                hipc,
                resources.system.diagnostics.file_system_access_log_mode(),
                resources.files.save_data.as_ref(),
                &resources.files.archives,
            )
            .await?
        }
        HorizonIpcObject::Bcat(session) => {
            services::dispatch_bcat(process, session, request, hipc).await?
        }
        HorizonIpcObject::Nim(session) => services::dispatch_nim(process, session, request, hipc)?,
        HorizonIpcObject::Ngc(session) => services::dispatch_ngc(session, request, hipc)?,
        HorizonIpcObject::Nfp(session) => {
            services::dispatch_nfp(process, session, &resources.system.hid, request, hipc).await?
        }
        HorizonIpcObject::SystemSettings(_) => {
            services::dispatch_system_settings(process, request, &hipc.receive_statics).await?
        }
        HorizonIpcObject::UserSettings(settings) => {
            services::dispatch_user_settings(process, settings, request, hipc).await?
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
            services::dispatch_applet(
                process,
                applet,
                request,
                hipc,
                &resources.system,
                &resources.graphics,
                resources.files.save_data.as_ref(),
            )
            .await?
        }
        HorizonIpcObject::Account(account) => {
            services::dispatch_account(
                process,
                account,
                request,
                hipc,
                resources.system.user_account_switch_locked,
            )
            .await?
        }
        HorizonIpcObject::AccountProfile(profile) => {
            services::dispatch_account_profile(process, *profile, request, hipc, false).await?
        }
        HorizonIpcObject::AccountManagerForApplication(manager) => {
            services::dispatch_account_manager_for_application(
                process, manager, request, hipc, false,
            )
            .await?
        }
        HorizonIpcObject::Bsd(session) => services::dispatch_bsd(process, session, request, hipc)?,
        HorizonIpcObject::Ssl(session) => services::dispatch_ssl(session, request, hipc)?,
        HorizonIpcObject::Multimedia(session) => {
            services::dispatch_multimedia(session, request, hipc)?
        }
        HorizonIpcObject::AudioOutManager(session) => {
            services::dispatch_audio_out_manager(process, session, request, hipc).await?
        }
        HorizonIpcObject::AudioUser => {
            services::dispatch_audio_user(process, &resources.audio, request, hipc)?
        }
        HorizonIpcObject::StreamPlay | HorizonIpcObject::StreamPlayUser(_) => {
            services::dispatch_streamplay(process, target, request, hipc)?
        }
        HorizonIpcObject::OpusManager(session) => {
            services::dispatch_opus_manager(process, session, request, hipc).await?
        }
        HorizonIpcObject::PlayReport(session) => {
            services::dispatch_play_report(process, session, request, hipc).await?
        }
        HorizonIpcObject::OpusDecoder(session) => {
            services::dispatch_opus_decoder(process, session, request, hipc).await?
        }
        HorizonIpcObject::AudioOut(session) => {
            services::dispatch_audio_out(process, session, request, hipc).await?
        }
        HorizonIpcObject::AudioRendererManager(session) => {
            services::dispatch_audio_renderer_manager(
                process,
                session,
                resources.audio.backend.as_ref(),
                request,
                hipc,
            )
            .await?
        }
        HorizonIpcObject::AudioRenderer(session) => {
            services::dispatch_audio_renderer(
                process,
                session,
                resources.audio.backend.as_ref(),
                request,
                hipc,
            )
            .await?
        }
        HorizonIpcObject::AudioDevice(session) => {
            services::dispatch_audio_device(process, session, request, hipc).await?
        }
        HorizonIpcObject::Hid(hid) => {
            services::dispatch_hid(process, hid, &resources.system.hid, request, hipc).await?
        }
        HorizonIpcObject::Friends(session) => {
            services::dispatch_friends(process, session, request, hipc)?
        }
        HorizonIpcObject::SharedFont(session) => {
            services::dispatch_shared_font(process, session, request, hipc)?
        }
        HorizonIpcObject::Irs(session) => {
            services::dispatch_irs(process, session, request, hipc).await?
        }
        HorizonIpcObject::HidAppletResource(resource) => {
            services::dispatch_hid_applet_resource(process, resource, request)?
        }
        HorizonIpcObject::HidActiveVibrationDeviceList(list) => {
            services::dispatch_hid_active_vibration_device_list(list, request, hipc)?
        }
        HorizonIpcObject::Time(time) => {
            services::dispatch_time(process, time, request, hipc).await?
        }
        HorizonIpcObject::SystemClock(clock) => {
            services::dispatch_system_clock(clock, request, hipc).await?
        }
        HorizonIpcObject::SteadyClock(clock) => services::dispatch_steady_clock(clock, request)?,
        HorizonIpcObject::TimeZone(timezone) => services::dispatch_timezone(timezone, request)?,
        HorizonIpcObject::Vi(vi) => services::dispatch_vi(process, vi, request, hipc).await?,
        HorizonIpcObject::Csrng => services::dispatch_csrng(process, request, hipc).await?,
        HorizonIpcObject::Gsv(_) => {
            return Err(IpcWireError::Malformed("GSV requires TIPC requests"));
        }
        HorizonIpcObject::NvDrv2(session) => {
            services::dispatch_nvdrv2(process, session, request, hipc).await?
        }
        HorizonIpcObject::NvDrv(nvdrv) => {
            services::dispatch_nvdrv(
                process,
                nvdrv,
                request,
                hipc,
                process.thread_id(),
                time_environment.clock(),
            )
            .await?
        }
        HorizonIpcObject::LogManager(manager) => {
            services::dispatch_log_manager(
                process,
                manager,
                request,
                hipc,
                resources.system.diagnostics.guest_logs_level,
            )
            .await?
        }
        HorizonIpcObject::Logger(logger) => {
            services::dispatch_logger(
                process,
                logger,
                request,
                hipc,
                resources.system.diagnostics.guest_logs_level,
            )
            .await?
        }
        HorizonIpcObject::ParentalControl(factory) => {
            services::dispatch_parental_control(process, factory, request, hipc)?
        }
        HorizonIpcObject::ParentalControlService(service) => {
            services::dispatch_parental_control_service(None, service, request, hipc)?
        }
        HorizonIpcObject::NetworkInterface(manager) => {
            services::dispatch_network_interface(process, manager, request, hipc)?
        }
        HorizonIpcObject::ErrorContextWriter(session) => {
            services::dispatch_error_context_writer(session, request, hipc)?
        }
        HorizonIpcObject::NetworkGeneralService(service) => {
            services::dispatch_network_general_service(process, None, service, request, hipc)?
        }
        HorizonIpcObject::NetworkRequest(session) => {
            services::dispatch_network_request(process, false, session, request, hipc)?
        }
        HorizonIpcObject::SemanticObject(object) => {
            dispatch_plain_object(process, object, request, hipc).await?
        }
    };
    if let Some(service) = trace_service {
        let response_prefix = &response[..response.len().min(64)];
        log::trace!(
            "SendSyncRequest response service={service} handle={handle:#x} type={command_type} command={command_id} bytes={} created_handle={created_handle:?} prefix={response_prefix:02x?}",
            response.len(),
        );
    }
    process.write_response(response).await?;
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
    SessionClosed,
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
