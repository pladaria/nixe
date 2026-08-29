use super::prelude::*;

use crate::bsd::{BsdClientConfig, BsdMonitoringError, BsdRegistrationError, BsdSession};

const REGISTER_CLIENT_PAYLOAD_SIZE: usize = 0x30;
const START_MONITORING_PAYLOAD_SIZE: usize = 0x08;
const BSD_ERRNO_SUCCESS: u32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BsdCommand {
    RegisterClient,
    StartMonitoring,
}

impl BsdCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::RegisterClient),
            1 => Some(Self::StartMonitoring),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_bsd(
    process: &mut ExceptionProcessContext<'_>,
    session: &BsdSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match &request.domain {
        Some(DomainRequest::Close { .. }) => {
            return bsd_response(
                session,
                request.token,
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                &[],
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
                return bsd_response(session, request.token, result, &[]);
            }
        }
        None if session.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain bsd:u request omitted its domain header",
            ));
        }
        None => {}
    }

    let Some(command) = BsdCommand::decode(request.command_id) else {
        return unsupported_service_command("bsd:u", request.command_id);
    };

    match command {
        // The socket SDK opens bsd:u twice, then registers on its command-session
        // pool with the caller PID and a copied transfer-memory handle. Retaining
        // the copied object mirrors the service's ownership after the client
        // closes its original handle. RegisterClient's scalar result is a 4-byte
        // BSD status value, not a client identifier.
        // https://switchbrew.org/w/index.php?title=Sockets_services&oldid=14937#RegisterClient
        BsdCommand::RegisterClient => {
            let Some(config) = decode_client_config(request.data) else {
                return invalid_header(session, request.token);
            };
            let Some(declared_transfer_size) = request_u64(request.data, 0x28) else {
                return invalid_header(session, request.token);
            };
            if hipc.pid.is_none()
                || request_u64(request.data, 0x20) != Some(0)
                || !has_only_transport_padding(request.data, REGISTER_CLIENT_PAYLOAD_SIZE)
                || hipc.copy_handles.len() != 1
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)
            {
                return invalid_header(session, request.token);
            }
            let Some(transfer_memory) = process
                .handles()
                .get_as::<TransferMemoryObject>(hipc.copy_handles[0])
                .cloned()
            else {
                return invalid_header(session, request.token);
            };
            if declared_transfer_size != transfer_memory.size() {
                return invalid_header(session, request.token);
            }

            let process_id = process.process_id();
            session
                .system()
                .register_client(process_id, config, transfer_memory)
                .map_err(|error| registration_error(request.command_id, error))?;
            log::debug!(
                "bsd:u registered process {process_id} with transfer memory {declared_transfer_size:#x} bytes"
            );
            bsd_response(
                session,
                request.token,
                HorizonIpcResult::SUCCESS,
                &BSD_ERRNO_SUCCESS.to_le_bytes(),
            )
        }
        // StartMonitoring is issued on the separate monitor session. Horizon
        // identifies the client with the HIPC PID descriptor; the raw u64 is
        // reserved and zero. Both sessions resolve through the shared registry.
        // https://switchbrew.org/w/index.php?title=Sockets_services&oldid=14937#StartMonitoring
        BsdCommand::StartMonitoring => {
            if request_u64(request.data, 0) != Some(0)
                || hipc.pid.is_none()
                || !has_only_transport_padding(request.data, START_MONITORING_PAYLOAD_SIZE)
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return invalid_header(session, request.token);
            }
            session
                .system()
                .start_monitoring(process.process_id())
                .map_err(|error| monitoring_error(request.command_id, error))?;
            bsd_response(session, request.token, HorizonIpcResult::SUCCESS, &[])
        }
    }
}

fn decode_client_config(data: &[u8]) -> Option<BsdClientConfig> {
    Some(BsdClientConfig {
        version: request_u32(data, 0x00)?,
        tcp_tx_buffer_size: request_u32(data, 0x04)?,
        tcp_rx_buffer_size: request_u32(data, 0x08)?,
        tcp_tx_buffer_max_size: request_u32(data, 0x0c)?,
        tcp_rx_buffer_max_size: request_u32(data, 0x10)?,
        udp_tx_buffer_size: request_u32(data, 0x14)?,
        udp_rx_buffer_size: request_u32(data, 0x18)?,
        socket_buffer_efficiency: request_u32(data, 0x1c)?,
    })
}

fn invalid_header(
    session: &BsdSession,
    token: u32,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    bsd_response(
        session,
        token,
        HorizonIpcResult::CMIF_INVALID_IN_HEADER,
        &[],
    )
}

fn bsd_response(
    session: &BsdSession,
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let response = if session.is_domain() {
        encode_domain_response(token, result, data, &[], &[])?
    } else {
        encode_response(token, result, data, None)?
    };
    Ok((response, None))
}

fn registration_error(command_id: u32, error: BsdRegistrationError) -> IpcWireError {
    let detail = match error {
        BsdRegistrationError::AlreadyRegistered => {
            "registering the same BSD client more than once is not implemented"
        }
    };
    IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
        service: "bsd:u",
        command_id,
        detail,
    })
}

fn monitoring_error(command_id: u32, error: BsdMonitoringError) -> IpcWireError {
    let detail = match error {
        BsdMonitoringError::UnknownClient => {
            "monitoring was requested before the process registered with BSD"
        }
    };
    IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
        service: "bsd:u",
        command_id,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_configuration_decodes_the_libnx_wire_layout() {
        let mut data = [0_u8; REGISTER_CLIENT_PAYLOAD_SIZE];
        for (index, value) in [1_u32, 2, 3, 4, 5, 6, 7, 8].into_iter().enumerate() {
            data[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }

        assert_eq!(
            decode_client_config(&data),
            Some(BsdClientConfig {
                version: 1,
                tcp_tx_buffer_size: 2,
                tcp_rx_buffer_size: 3,
                tcp_tx_buffer_max_size: 4,
                tcp_rx_buffer_max_size: 5,
                udp_tx_buffer_size: 6,
                udp_rx_buffer_size: 7,
                socket_buffer_efficiency: 8,
            })
        );
        assert_eq!(decode_client_config(&data[..0x1f]), None);
    }
}
