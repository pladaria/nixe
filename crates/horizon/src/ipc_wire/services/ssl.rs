use super::prelude::*;

pub(in crate::ipc_wire) fn dispatch_ssl(
    session: &SslSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match &request.domain {
        Some(DomainRequest::Close { .. }) => {
            // No child contexts exist until CreateContext is implemented.
            return ssl_response(
                session,
                request.token,
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
            );
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if *object_id != 1 || !input_objects.is_empty() {
                let result = if *object_id == 1 {
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER
                } else {
                    HorizonIpcResult::CMIF_TARGET_NOT_FOUND
                };
                return ssl_response(session, request.token, result);
            }
        }
        None if session.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain ssl request omitted its domain header",
            ));
        }
        None => {}
    }

    // SetInterfaceVersion is the only SSL command required by service
    // initialization. Contexts, certificates and TLS I/O remain unsupported.
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/ssl.c
    // https://switchbrew.org/wiki/SSL_services#SetInterfaceVersion
    if request.command_id != 5 {
        return unsupported_service_command("ssl", request.command_id);
    }
    let Some(version) = request_u32(request.data, 0) else {
        return ssl_response(
            session,
            request.token,
            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
        );
    };
    if !request.has_payload_size(4) || has_ipc_descriptors(hipc) {
        return ssl_response(
            session,
            request.token,
            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
        );
    }
    if !matches!(version, 1..=3) {
        return Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::CommandVariant {
                service: "ssl",
                command_id: request.command_id,
                detail: "SSL interface versions outside 1..=3 are not implemented",
            },
        ));
    }
    session.set_interface_version(version);
    log::debug!("ssl selected interface version {version}");
    ssl_response(session, request.token, HorizonIpcResult::SUCCESS)
}

fn ssl_response(
    session: &SslSession,
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let response = if session.is_domain() {
        encode_domain_response(token, result, &[], &[], &[])?
    } else {
        encode_response(token, result, &[], None)?
    };
    Ok((response, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(data: &[u8]) -> CmifRequest<'_> {
        CmifRequest {
            command_type: 4,
            command_id: 5,
            token: 0x1234,
            context: None,
            data,
            domain: None,
        }
    }

    #[test]
    fn malformed_version_requests_do_not_change_negotiated_state() {
        let session = SslSession::new();
        session.set_interface_version(1);
        let buffer = [0_u8; 8];
        let hipc = HipcRequest::decode(&buffer).unwrap();
        for data in [&[][..], &[2, 0, 0][..]] {
            let (response, handle) = dispatch_ssl(&session, request(data), &hipc).unwrap();
            assert_eq!(
                request_u32(&response, 24),
                Some(HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw())
            );
            assert_eq!(request_u32(&response, 28), Some(0x1234));
            assert_eq!(handle, None);
            assert_eq!(session.interface_version(), 1);
        }
        for with_pid in [false, true] {
            let mut hipc = hipc.clone();
            if with_pid {
                hipc.pid = Some(42);
            } else {
                hipc.copy_handles.push(42);
            }
            let (response, _) =
                dispatch_ssl(&session, request(&2_u32.to_le_bytes()), &hipc).unwrap();
            assert_eq!(
                request_u32(&response, 24),
                Some(HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw())
            );
            assert_eq!(session.interface_version(), 1);
        }
    }

    #[test]
    fn plain_sessions_accept_versions_with_transport_padding() {
        let session = SslSession::new();
        let buffer = [0_u8; 8];
        let hipc = HipcRequest::decode(&buffer).unwrap();
        for version in 1_u32..=3 {
            let mut data = [0xa5_u8; 16];
            data[..4].copy_from_slice(&version.to_le_bytes());
            let (response, _) = dispatch_ssl(&session, request(&data), &hipc).unwrap();
            assert_eq!(request_u32(&response, 24), Some(0));
            assert_eq!(session.interface_version(), version);
        }
    }

    #[test]
    fn unsupported_versions_and_tls_commands_remain_fatal() {
        let session = SslSession::new();
        let buffer = [0_u8; 8];
        let hipc = HipcRequest::decode(&buffer).unwrap();
        for version in [0_u32, 4, 5, u32::MAX] {
            assert!(matches!(
                dispatch_ssl(&session, request(&version.to_le_bytes()), &hipc),
                Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandVariant {
                        service: "ssl",
                        command_id: 5,
                        ..
                    }
                ))
            ));
            assert_eq!(session.interface_version(), 0);
        }
        for command_id in [0, 1, 2, 3, 4, 6, 7, 8, 9, 999] {
            let mut request = request(&[]);
            request.command_id = command_id;
            assert!(matches!(
                dispatch_ssl(&session, request, &hipc),
                Err(IpcWireError::UnsupportedService(UnsupportedServiceOperation::Command {
                    service: "ssl", command_id: actual
                })) if actual == command_id
            ));
        }
    }
}
