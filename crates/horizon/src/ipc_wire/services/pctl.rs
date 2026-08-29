use super::prelude::*;

enum ParentalControlTarget {
    Factory,
    Service(ParentalControlSession),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParentalControlFactoryCommand {
    CreateService,
    CreateServiceWithoutInitialize,
}

impl ParentalControlFactoryCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CreateService),
            1 => Some(Self::CreateServiceWithoutInitialize),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParentalControlServiceCommand {
    Initialize,
    CheckFreeCommunicationPermission,
    IsRestrictionEnabled,
}

impl ParentalControlServiceCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            1 => Some(Self::Initialize),
            1001 => Some(Self::CheckFreeCommunicationPermission),
            1031 => Some(Self::IsRestrictionEnabled),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_parental_control(
    process: &mut ExceptionProcessContext<'_>,
    factory: &ParentalControlFactorySession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            let result = if factory.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            return parental_control_response(factory, request.token, result, &[], &[]);
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return parental_control_response(
                    factory,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                );
            }
            if *object_id == 1 {
                ParentalControlTarget::Factory
            } else {
                let Some(service) = factory.object(*object_id) else {
                    return parental_control_response(
                        factory,
                        request.token,
                        HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                        &[],
                        &[],
                    );
                };
                ParentalControlTarget::Service(service)
            }
        }
        None if factory.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain pctl request omitted its domain header",
            ));
        }
        None => ParentalControlTarget::Factory,
    };

    match target {
        ParentalControlTarget::Factory => {
            dispatch_parental_control_factory(process, factory, request, hipc)
        }
        ParentalControlTarget::Service(service) => {
            dispatch_parental_control_service(Some(factory), &service, request, hipc)
        }
    }
}

fn dispatch_parental_control_factory(
    process: &mut ExceptionProcessContext<'_>,
    factory: &ParentalControlFactorySession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = ParentalControlFactoryCommand::decode(request.command_id) else {
        return unsupported_service_command("pctl", request.command_id);
    };
    match command {
        // CreateService initializes its child as part of command 0 on older
        // Horizon versions. Since 4.0.0, command 1 creates an uninitialized
        // child and the client follows with IParentalControlService::Initialize.
        // The factory ABI sends one reserved u64 and a PID descriptor:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/pctl.c#L20-L24
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/pctl.c#L48-L55
        command @ (ParentalControlFactoryCommand::CreateService
        | ParentalControlFactoryCommand::CreateServiceWithoutInitialize) => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0) != Some(0)
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return parental_control_response(
                    factory,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                    &[],
                );
            }
            let service = ParentalControlSession::new(
                process.process_id(),
                command == ParentalControlFactoryCommand::CreateService,
            );
            if factory.is_domain() {
                let Some(object_id) = factory.insert_object(service) else {
                    return parental_control_response(
                        factory,
                        request.token,
                        HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                        &[],
                        &[],
                    );
                };
                log::debug!(
                    "pctl opened IParentalControlService as domain object {object_id:#x} for process {}",
                    process.process_id()
                );
                parental_control_response(
                    factory,
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &[],
                    &[object_id],
                )
            } else {
                let handle = process
                    .handles_mut()
                    .insert(HorizonIpcObject::ParentalControlService(service))
                    .map_err(|_| {
                        IpcWireError::HostResourceExhausted(
                            "installing a parental-control child handle",
                        )
                    })?;
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ))
            }
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_parental_control_service(
    factory: Option<&ParentalControlFactorySession>,
    service: &ParentalControlSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = ParentalControlServiceCommand::decode(request.command_id) else {
        return unsupported_service_command("IParentalControlService", request.command_id);
    };
    match command {
        // Initialize was split from factory command 1 in Horizon 4.0.0.
        // https://switchbrew.org/w/index.php?title=Parental_Control_services&oldid=14435
        ParentalControlServiceCommand::Initialize => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return parental_control_service_response(
                    factory,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            service.initialize();
            log::debug!(
                "pctl initialized IParentalControlService for process {}",
                service.process_id()
            );
            parental_control_service_response(
                factory,
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
            )
        }
        // Nixe currently has no configured parental-control profile. This is
        // an explicit unrestricted policy, so the real permission command
        // succeeds; it is not a blanket success fallback for unknown pctl
        // methods. A future profile implementation must evaluate title age,
        // communication, play-timer, and temporary-unlock state here.
        ParentalControlServiceCommand::CheckFreeCommunicationPermission => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return parental_control_service_response(
                    factory,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            if !service.is_initialized() {
                return parental_control_service_response(
                    factory,
                    request.token,
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                    &[],
                );
            }
            parental_control_service_response(
                factory,
                request.token,
                HorizonIpcResult::SUCCESS,
                &[],
            )
        }
        // IsRestrictionEnabled is the guest-visible policy query. Returning
        // false is the exact representation of Nixe's current unrestricted
        // profile, so age ratings never gate launch at this boundary.
        // The command returns one u8 bool with no input:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/pctl.c#L91-L93
        ParentalControlServiceCommand::IsRestrictionEnabled => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return parental_control_service_response(
                    factory,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            if !service.is_initialized() {
                return parental_control_service_response(
                    factory,
                    request.token,
                    HorizonIpcResult::SF_PRECONDITION_VIOLATION,
                    &[],
                );
            }
            parental_control_service_response(
                factory,
                request.token,
                HorizonIpcResult::SUCCESS,
                &[0],
            )
        }
    }
}

fn parental_control_service_response(
    factory: Option<&ParentalControlFactorySession>,
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if let Some(factory) = factory {
        parental_control_response(factory, token, result, data, &[])
    } else {
        Ok((encode_response(token, result, data, None)?, None))
    }
}

fn parental_control_response(
    factory: &ParentalControlFactorySession,
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    domain_objects: &[u32],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if factory.is_domain() {
        Ok((
            encode_domain_response(token, result, data, &[], domain_objects)?,
            None,
        ))
    } else {
        Ok((encode_response(token, result, data, None)?, None))
    }
}
