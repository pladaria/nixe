use super::prelude::*;

enum NetworkInterfaceTarget {
    Root,
    GeneralService(NetworkGeneralServiceSession),
    Request(NetworkRequestSession),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NetworkInterfaceManagerCommand {
    CreateGeneralService,
}

impl NetworkInterfaceManagerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            5 => Some(Self::CreateGeneralService),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_network_interface(
    process: &mut ExceptionProcessContext<'_>,
    manager: &NetworkInterfaceManagerSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            let result = if manager.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            return network_interface_response(manager, request.token, result, &[]);
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return network_interface_response(
                    manager,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            if *object_id == 1 {
                NetworkInterfaceTarget::Root
            } else {
                let Some(object) = manager.object(*object_id) else {
                    return network_interface_response(
                        manager,
                        request.token,
                        HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                        &[],
                    );
                };
                match object {
                    NetworkInterfaceObject::GeneralService(service) => {
                        NetworkInterfaceTarget::GeneralService(service)
                    }
                    NetworkInterfaceObject::Request(session) => {
                        NetworkInterfaceTarget::Request(session)
                    }
                }
            }
        }
        None if manager.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain nifm:u request omitted its domain header",
            ));
        }
        None => NetworkInterfaceTarget::Root,
    };

    match target {
        NetworkInterfaceTarget::Root => {
            dispatch_network_interface_manager(process, manager, request, hipc)
        }
        NetworkInterfaceTarget::GeneralService(service) => {
            dispatch_network_general_service(process, Some(manager), &service, request, hipc)
        }
        NetworkInterfaceTarget::Request(session) => {
            dispatch_network_request(process, true, &session, request, hipc)
        }
    }
}

fn dispatch_network_interface_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &NetworkInterfaceManagerSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = NetworkInterfaceManagerCommand::decode(request.command_id) else {
        return unsupported_service_command("nifm:u", request.command_id);
    };
    match command {
        // Horizon 3.0.0+ creates IGeneralService with command 5. The input is
        // a reserved zero u64 plus the caller PID:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nifm.c
        NetworkInterfaceManagerCommand::CreateGeneralService => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0) != Some(0)
                || !request.has_payload_size(8)
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return network_interface_response(
                    manager,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let service = NetworkGeneralServiceSession::new(process.process_id());
            if manager.is_domain() {
                let Some(object_id) =
                    manager.insert_object(NetworkInterfaceObject::GeneralService(service))
                else {
                    return network_interface_response(
                        manager,
                        request.token,
                        HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                        &[],
                    );
                };
                log::debug!(
                    "nifm:u opened IGeneralService as domain object {object_id:#x} for process {}",
                    service.process_id()
                );
                network_interface_response(
                    manager,
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &[object_id],
                )
            } else {
                let handle = process
                    .handles_mut()
                    .insert(HorizonIpcObject::NetworkGeneralService(service))
                    .map_err(|_| {
                        IpcWireError::HostResourceExhausted(
                            "installing a network general-service handle",
                        )
                    })?;
                log::debug!("nifm:u opened IGeneralService handle {handle:#x}");
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ))
            }
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_network_general_service(
    process: &mut ExceptionProcessContext<'_>,
    manager: Option<&NetworkInterfaceManagerSession>,
    _service: &NetworkGeneralServiceSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if request.command_id != 4 {
        return unsupported_service_command("IGeneralService", request.command_id);
    }
    let is_domain = manager.is_some_and(NetworkInterfaceManagerSession::is_domain);
    // CreateRequest takes the request revision, currently revision 2.
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nifm.c
    if !request.has_payload_size(4) || has_ipc_descriptors(hipc) {
        return network_request_response(
            request.token,
            is_domain,
            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
            &[],
        );
    }
    if request_u32(request.data, 0) != Some(2) {
        return Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::CommandVariant {
                service: "IGeneralService",
                command_id: 4,
                detail: "unsupported network request revision",
            },
        ));
    }
    let session = NetworkRequestSession::new();
    if is_domain {
        let Some(id) =
            manager.and_then(|m| m.insert_object(NetworkInterfaceObject::Request(session)))
        else {
            return network_request_response(
                request.token,
                true,
                HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                &[],
            );
        };
        semantic_success(request.token, true, &[], &[], &[id], None)
    } else {
        let handle = process
            .handles_mut()
            .insert(HorizonIpcObject::NetworkRequest(session))
            .map_err(|_| {
                IpcWireError::HostResourceExhausted("installing a network request handle")
            })?;
        semantic_success(request.token, false, &[], &[], &[], Some(handle))
    }
}

pub(in crate::ipc_wire) fn dispatch_network_request(
    process: &mut ExceptionProcessContext<'_>,
    is_domain: bool,
    session: &NetworkRequestSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if !matches!(request.command_id, 0..=4) {
        return unsupported_service_command("IRequest", request.command_id);
    }
    if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
        return network_request_response(
            request.token,
            is_domain,
            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
            &[],
        );
    }
    match request.command_id {
        0 => network_request_response(
            request.token,
            is_domain,
            HorizonIpcResult::SUCCESS,
            &1_u32.to_le_bytes(),
        ),
        1 => network_request_response(request.token, is_domain, session.result(), &[]),
        2 => {
            let mut handles = [0; 2];
            for (handle, event) in handles.iter_mut().zip(session.events()) {
                *handle = process.handles_mut().insert(event).map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing a network request event")
                })?;
            }
            semantic_success(request.token, is_domain, &[], &handles, &[], None)
        }
        3 => {
            // Offline requests finish synchronously; cancel has no pending work.
            network_request_response(request.token, is_domain, HorizonIpcResult::SUCCESS, &[])
        }
        4 => {
            session.submit();
            network_request_response(request.token, is_domain, HorizonIpcResult::SUCCESS, &[])
        }
        _ => unreachable!(),
    }
}

fn network_request_response(
    token: u32,
    is_domain: bool,
    result: HorizonIpcResult,
    data: &[u8],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((
        if is_domain {
            encode_domain_response(token, result, data, &[], &[])?
        } else {
            encode_response(token, result, data, None)?
        },
        None,
    ))
}

fn network_interface_response(
    manager: &NetworkInterfaceManagerSession,
    token: u32,
    result: HorizonIpcResult,
    domain_objects: &[u32],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if manager.is_domain() {
        Ok((
            encode_domain_response(token, result, &[], &[], domain_objects)?,
            None,
        ))
    } else {
        Ok((encode_response(token, result, &[], None)?, None))
    }
}
