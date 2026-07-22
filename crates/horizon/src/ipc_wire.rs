//! Synchronous Horizon service-call adapter built on the checked wire codec.
//!
//! The semantic service layer remains independent of guest wire layouts. This
//! module validates the command buffer in the current thread's TLS and bridges
//! decoded messages into the service manager and semantic service objects.

use nixe_cpu::address::GuestVirtualAddress;
use nixe_cpu::memory::{DataAccessFault, MemoryAccess, MemoryAccessSize, MemoryValue};
use nixe_runtime::ExceptionProcessContext;

use crate::ipc_message::{
    COMMAND_BUFFER_SIZE, CmifRequest, CmifResponse, HipcRequest, MessageError,
};
use crate::{IpcDispatcher, IpcResultCode, IpcService, IpcSession, ServiceManagerSession};

const NAMED_PORT_NAME_SIZE: usize = 12;
const CMIF_COMMAND_CLOSE: u16 = 2;
const CMIF_COMMAND_CONTROL: u16 = 5;
const CMIF_COMMAND_CONTROL_WITH_CONTEXT: u16 = 7;
const SM_MODULE: u32 = 21;
const SM_OUT_OF_SESSIONS: u32 = make_result(SM_MODULE, 3);
const SM_INVALID_CLIENT: u32 = make_result(SM_MODULE, 2);
const SM_INVALID_SERVICE_NAME: u32 = make_result(SM_MODULE, 6);
const SM_NOT_REGISTERED: u32 = make_result(SM_MODULE, 7);
const SM_NOT_ALLOWED: u32 = make_result(SM_MODULE, 8);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum IpcWireError {
    GuestMemory(DataAccessFault),
    Malformed(&'static str),
}

impl From<MessageError> for IpcWireError {
    fn from(error: MessageError) -> Self {
        Self::Malformed(error.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NamedPortResult {
    Connected(u32),
    NotFound,
    NameOutOfRange,
    OutOfHandles,
}

pub(crate) fn connect_to_named_port(
    process: &mut ExceptionProcessContext<'_>,
    name_address: GuestVirtualAddress,
) -> Result<NamedPortResult, IpcWireError> {
    let mut name = [0_u8; NAMED_PORT_NAME_SIZE];
    for (index, byte) in name.iter_mut().enumerate() {
        *byte = read_u8(process, add(name_address, index)?)?;
        if *byte == 0 {
            let port_name = &name[..index];
            if port_name != b"sm:" {
                log::debug!(
                    "ConnectToNamedPort did not find named port {:?}",
                    String::from_utf8_lossy(port_name)
                );
                return Ok(NamedPortResult::NotFound);
            }
            log::debug!("ConnectToNamedPort opening a client session to sm:");
            return Ok(
                match process.handles_mut().insert(ServiceManagerSession::new()) {
                    Ok(handle) => NamedPortResult::Connected(handle),
                    Err(_) => NamedPortResult::OutOfHandles,
                },
            );
        }
    }
    Ok(NamedPortResult::NameOutOfRange)
}

pub(crate) fn send_sync_request(
    process: &mut ExceptionProcessContext<'_>,
    tls: GuestVirtualAddress,
    handle: u32,
) -> Result<SyncRequestResult, IpcWireError> {
    let manager = process
        .handles()
        .get_as::<ServiceManagerSession>(handle)
        .cloned();
    let service = process.handles().get_as::<IpcSession>(handle).copied();
    if manager.is_none() && service.is_none() {
        return Ok(SyncRequestResult::InvalidHandle);
    }

    let mut buffer = [0_u8; COMMAND_BUFFER_SIZE];
    read_bytes(process, tls, &mut buffer)?;
    let hipc = HipcRequest::decode(&buffer)?;
    let request = CmifRequest::decode(&hipc, false)?;
    log::debug!(
        "SendSyncRequest handle={handle:#x} type={} command={} send_pid={} descriptors={}/{}/{}/{} handles={}/{}",
        request.command_type,
        request.command_id,
        hipc.pid.is_some(),
        hipc.send_statics.len(),
        hipc.send_buffers.len(),
        hipc.receive_buffers.len(),
        hipc.exchange_buffers.len(),
        hipc.copy_handles.len(),
        hipc.move_handles.len(),
    );

    if request.command_type == CMIF_COMMAND_CLOSE {
        return Ok(SyncRequestResult::Success);
    }
    if matches!(
        request.command_type,
        CMIF_COMMAND_CONTROL | CMIF_COMMAND_CONTROL_WITH_CONTEXT
    ) {
        let response = match request.command_id {
            // QueryPointerBufferSize. Zero makes libnx use map-alias buffers,
            // which the future descriptor bridge can validate explicitly.
            3 => encode_response(request.token, 0, &0_u16.to_le_bytes(), None),
            _ => encode_response(request.token, SM_NOT_REGISTERED, &[], None),
        }?;
        write_bytes(process, tls, &response)?;
        return Ok(SyncRequestResult::Success);
    }
    let (response, created_handle) = if let Some(manager) = manager {
        dispatch_service_manager(process, &manager, request, hipc.pid.is_some())?
    } else {
        let _service = service.expect("session kind was checked");
        (
            encode_response(request.token, SM_NOT_REGISTERED, &[], None)?,
            None,
        )
    };
    if let Err(error) = write_bytes(process, tls, &response) {
        if let Some(handle) = created_handle {
            let _ = process.handles_mut().close(handle);
        }
        return Err(error);
    }
    Ok(SyncRequestResult::Success)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SyncRequestResult {
    Success,
    InvalidHandle,
}

fn dispatch_service_manager(
    process: &mut ExceptionProcessContext<'_>,
    manager: &ServiceManagerSession,
    request: CmifRequest<'_>,
    sent_pid: bool,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match request.command_id {
        0 => {
            if !sent_pid || request.data.len() < 8 {
                return Ok((
                    encode_response(request.token, SM_INVALID_CLIENT, &[], None)?,
                    None,
                ));
            }
            manager.register_client();
            log::debug!(
                "sm:RegisterClient associated process {}",
                process.process_id()
            );
            Ok((encode_response(request.token, 0, &[], None)?, None))
        }
        1 => {
            if !manager.is_registered() {
                return Ok((
                    encode_response(request.token, SM_INVALID_CLIENT, &[], None)?,
                    None,
                ));
            }
            let Some(encoded_name) = request.data.get(..8) else {
                return Ok((
                    encode_response(request.token, SM_INVALID_SERVICE_NAME, &[], None)?,
                    None,
                ));
            };
            let Some(name) = decode_service_name(encoded_name) else {
                return Ok((
                    encode_response(request.token, SM_INVALID_SERVICE_NAME, &[], None)?,
                    None,
                ));
            };
            log::debug!(
                "sm:GetService requested {:?}",
                String::from_utf8_lossy(name)
            );
            let Some(service) = IpcService::from_name(name) else {
                let encoded_name: [u8; 8] = encoded_name
                    .try_into()
                    .expect("service name length was checked");
                if manager.first_unavailable_request(encoded_name) {
                    log::warn!(
                        "guest requested unavailable Horizon service via sm: {:?}",
                        String::from_utf8_lossy(name)
                    );
                }
                return Ok((
                    encode_response(request.token, SM_NOT_REGISTERED, &[], None)?,
                    None,
                ));
            };
            let (mounts, handles) = process.mounts_and_handles_mut();
            match IpcDispatcher::connect(mounts, handles, service) {
                Ok(handle) => {
                    log::debug!("sm:GetService returned session handle {handle:#x}");
                    Ok((
                        encode_response(request.token, 0, &[], Some(handle))?,
                        Some(handle),
                    ))
                }
                Err(error) if error == IpcResultCode::ACCESS_DENIED => Ok((
                    encode_response(request.token, SM_NOT_ALLOWED, &[], None)?,
                    None,
                )),
                Err(error) if error == IpcResultCode::RESOURCE_LIMIT => Ok((
                    encode_response(request.token, SM_OUT_OF_SESSIONS, &[], None)?,
                    None,
                )),
                Err(_) => Ok((
                    encode_response(request.token, SM_NOT_REGISTERED, &[], None)?,
                    None,
                )),
            }
        }
        _ => Ok((
            encode_response(request.token, SM_NOT_REGISTERED, &[], None)?,
            None,
        )),
    }
}

fn encode_response(
    token: u32,
    result: u32,
    data: &[u8],
    move_handle: Option<u32>,
) -> Result<Vec<u8>, IpcWireError> {
    let move_handle_storage = move_handle.into_iter().collect::<Vec<_>>();
    CmifResponse {
        token,
        result,
        data,
        move_handles: &move_handle_storage,
        ..CmifResponse::default()
    }
    .encode()
    .map_err(Into::into)
}

fn decode_service_name(encoded: &[u8]) -> Option<&[u8]> {
    let end = encoded
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(encoded.len());
    if end == 0 || encoded[end..].iter().any(|byte| *byte != 0) {
        None
    } else {
        Some(&encoded[..end])
    }
}

fn read_bytes(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    output: &mut [u8],
) -> Result<(), IpcWireError> {
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = read_u8(process, add(start, index)?)?;
    }
    Ok(())
}

fn write_bytes(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    bytes: &[u8],
) -> Result<(), IpcWireError> {
    for (index, byte) in bytes.iter().copied().enumerate() {
        process
            .memory()
            .write(
                process.cpu().address_space_id(),
                add(start, index)?,
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(byte),
            )
            .map_err(IpcWireError::GuestMemory)?;
    }
    Ok(())
}

fn read_u8(
    process: &ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
) -> Result<u8, IpcWireError> {
    let value = process
        .memory()
        .read(
            process.cpu().address_space_id(),
            address,
            MemoryAccess::normal(MemoryAccessSize::Byte),
        )
        .map_err(IpcWireError::GuestMemory)?
        .value;
    let MemoryValue::U8(value) = value else {
        unreachable!("byte access returns a byte value")
    };
    Ok(value)
}

fn add(address: GuestVirtualAddress, offset: usize) -> Result<GuestVirtualAddress, IpcWireError> {
    let offset = u64::try_from(offset)
        .map_err(|_| IpcWireError::Malformed("guest address offset overflows"))?;
    address
        .checked_add(offset)
        .ok_or(IpcWireError::Malformed("guest address overflows"))
}

const fn make_result(module: u32, description: u32) -> u32 {
    (module & 0x1ff) | ((description & 0x1fff) << 9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_names_require_canonical_zero_padding() {
        assert_eq!(decode_service_name(b"fsp-srv\0"), Some(&b"fsp-srv"[..]));
        assert_eq!(decode_service_name(b"aoc:u\0\0\0"), Some(&b"aoc:u"[..]));
        assert_eq!(decode_service_name(b"\0\0\0\0\0\0\0\0"), None);
        assert_eq!(decode_service_name(b"fs\0bad!!"), None);
    }

    #[test]
    fn response_layout_round_trips_libnx_parser_offsets() {
        let response = encode_response(7, 0, &0x100_u16.to_le_bytes(), Some(0x44)).unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());
        assert_eq!(word(4) >> 31, 1);
        assert_eq!(word(8), 1 << 5);
        assert_eq!(word(12), 0x44);
        assert_eq!(word(16), 0x4f43_4653);
        assert_eq!(word(24), 0);
        assert_eq!(word(28), 7);
        assert_eq!(&response[32..34], &0x100_u16.to_le_bytes());
    }
}
