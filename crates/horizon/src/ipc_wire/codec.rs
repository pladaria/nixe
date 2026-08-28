//! Checked CMIF response encoding, scalar decoding, and guest-memory access.

use super::*;

pub(super) fn write_descriptor_bytes(
    process: &ExceptionProcessContext<'_>,
    descriptor: BufferDescriptor,
    bytes: &[u8],
) -> Result<(), IpcWireError> {
    if u64::try_from(bytes.len())
        .ok()
        .is_none_or(|size| size > descriptor.size)
    {
        return Err(IpcWireError::Malformed(
            "service response exceeds its output descriptor",
        ));
    }
    write_bytes(process, GuestVirtualAddress::new(descriptor.address), bytes)
}

pub(super) fn has_ipc_descriptors(hipc: &HipcRequest<'_>) -> bool {
    hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.send_statics.is_empty()
        || !hipc.send_buffers.is_empty()
        || !hipc.receive_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
}

pub(super) fn cmif_error(
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((encode_response(token, result, &[], None)?, None))
}

pub(super) fn request_u32(data: &[u8], offset: usize) -> Option<u32> {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
}

pub(super) fn request_i32(data: &[u8], offset: usize) -> Option<i32> {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(i32::from_le_bytes)
}

pub(super) fn request_u64(data: &[u8], offset: usize) -> Option<u64> {
    data.get(offset..offset + 8)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
}

pub(super) fn request_i64(data: &[u8], offset: usize) -> Option<i64> {
    data.get(offset..offset + 8)
        .and_then(|bytes| bytes.try_into().ok())
        .map(i64::from_le_bytes)
}

pub(super) fn request_f32(data: &[u8], offset: usize) -> Option<f32> {
    data.get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(f32::from_le_bytes)
}

pub(super) fn encode_response(
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    move_handle: Option<u32>,
) -> Result<Vec<u8>, IpcWireError> {
    let move_handle_storage = move_handle.into_iter().collect::<Vec<_>>();
    CmifResponse {
        token,
        result: result.raw(),
        data,
        move_handles: &move_handle_storage,
        ..CmifResponse::default()
    }
    .encode()
    .map_err(Into::into)
}

pub(super) fn encode_domain_response(
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
    copy_handles: &[u32],
    domain_objects: &[u32],
) -> Result<Vec<u8>, IpcWireError> {
    CmifResponse {
        token,
        result: result.raw(),
        data,
        copy_handles,
        is_domain: true,
        domain_objects,
        ..CmifResponse::default()
    }
    .encode()
    .map_err(Into::into)
}

pub(super) fn decode_service_name(encoded: &[u8]) -> Option<&[u8]> {
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

pub(crate) fn read_bytes(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    output: &mut [u8],
) -> Result<(), IpcWireError> {
    process
        .memory()
        .read_bytes(process.cpu().address_space_id(), start, output)
        .map_err(IpcWireError::GuestMemory)
}

pub(crate) fn write_bytes(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    bytes: &[u8],
) -> Result<(), IpcWireError> {
    process
        .memory()
        .write_bytes(process.cpu().address_space_id(), start, bytes)
        .map_err(IpcWireError::GuestMemory)
}

pub(crate) fn validate_writable_ram_range(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    size: usize,
) -> Result<(), IpcWireError> {
    let address_space = process.cpu().address_space_id();
    let end = start.get().checked_add(size as u64).ok_or_else(|| {
        IpcWireError::GuestMemory(DataAccessFault::new(
            address_space,
            start,
            DataAccessKind::Write,
            DataAccessFaultReason::AddressOverflow,
        ))
    })?;
    let limit = GuestVirtualAddress::new(process.address_space_limit());
    let mut cursor = start;
    while cursor.get() < end {
        let Some(mapping) = process.memory().query_memory(address_space, cursor, limit) else {
            return Err(IpcWireError::GuestMemory(DataAccessFault::new(
                address_space,
                cursor,
                DataAccessKind::Write,
                DataAccessFaultReason::Unmapped,
            )));
        };
        if mapping.region != Some(MemoryRegionKind::Ram) {
            return Err(IpcWireError::GuestMemory(DataAccessFault::new(
                address_space,
                cursor,
                DataAccessKind::Write,
                DataAccessFaultReason::Device(
                    "IPC response buffer must be backed by ordinary RAM".into(),
                ),
            )));
        }
        if !mapping.permissions.contains(MemoryPermissions::WRITE) {
            return Err(IpcWireError::GuestMemory(DataAccessFault::new(
                address_space,
                cursor,
                DataAccessKind::Write,
                DataAccessFaultReason::WritePermissionDenied,
            )));
        }
        let mapping_end = mapping
            .base
            .get()
            .checked_add(mapping.size)
            .ok_or(IpcWireError::Internal("guest memory query range overflows"))?;
        if mapping_end <= cursor.get() {
            return Err(IpcWireError::Internal(
                "guest memory query did not advance while validating an IPC response",
            ));
        }
        cursor = GuestVirtualAddress::new(mapping_end.min(end));
    }
    Ok(())
}

pub(super) fn write_response(
    process: &ExceptionProcessContext<'_>,
    start: GuestVirtualAddress,
    capacity: usize,
    response: &[u8],
) -> Result<(), IpcWireError> {
    if response.len() > capacity {
        return Err(IpcWireError::Internal(
            "encoded IPC response exceeds its prevalidated command buffer",
        ));
    }
    write_bytes(process, start, response).map_err(|error| match error {
        IpcWireError::GuestMemory(fault) => IpcWireError::ResponseCommit(fault),
        error => error,
    })
}

pub(super) fn read_byte(
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

pub(super) fn add(
    address: GuestVirtualAddress,
    offset: usize,
) -> Result<GuestVirtualAddress, IpcWireError> {
    let offset = u64::try_from(offset)
        .map_err(|_| IpcWireError::Malformed("guest address offset overflows"))?;
    address
        .checked_add(offset)
        .ok_or(IpcWireError::Malformed("guest address overflows"))
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
        let response = encode_response(
            7,
            HorizonIpcResult::SUCCESS,
            &0x100_u16.to_le_bytes(),
            Some(0x44),
        )
        .unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());
        assert_eq!(word(4) >> 31, 1);
        assert_eq!(word(8), 1 << 5);
        assert_eq!(word(12), 0x44);
        assert_eq!(word(16), 0x4f43_4653);
        assert_eq!(word(24), 0);
        assert_eq!(word(28), 7);
        assert_eq!(&response[32..34], &0x100_u16.to_le_bytes());
    }

    #[test]
    fn response_preserves_the_typed_horizon_result() {
        let response =
            encode_response(0x33, HorizonIpcResult::SM_NOT_REGISTERED, &[], None).unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());

        assert_eq!(word(24), 0xe15);
        assert_eq!(word(28), 0x33);
    }
}
