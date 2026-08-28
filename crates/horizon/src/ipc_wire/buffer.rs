//! Validation for HIPC buffer descriptor shapes shared by service ABIs.

use super::IpcWireError;
use super::message::{
    BufferDescriptor, BufferMode, HipcRequest, ReceiveStaticDescriptor, ReceiveStatics,
};

pub(in crate::ipc_wire) fn one_receive_buffer(
    hipc: &HipcRequest<'_>,
) -> Result<BufferDescriptor, IpcWireError> {
    match hipc.receive_buffers.as_slice() {
        [descriptor] if descriptor.size > 0 && descriptor.mode != BufferMode::Invalid => {
            Ok(*descriptor)
        }
        _ => Err(IpcWireError::Malformed(
            "service command requires exactly one output buffer",
        )),
    }
}

pub(in crate::ipc_wire) fn one_send_buffer(
    hipc: &HipcRequest<'_>,
) -> Result<BufferDescriptor, IpcWireError> {
    match hipc.send_buffers.as_slice() {
        [descriptor] if descriptor.mode != BufferMode::Invalid => Ok(*descriptor),
        [_] => Err(IpcWireError::Malformed(
            "input buffer has an invalid mapping mode",
        )),
        _ => Err(IpcWireError::Malformed(
            "request requires exactly one input buffer",
        )),
    }
}

pub(in crate::ipc_wire) fn one_auto_select_input(
    hipc: &HipcRequest<'_>,
) -> Result<(u64, usize), IpcWireError> {
    let [pointer] = hipc.send_statics.as_slice() else {
        return Err(IpcWireError::Malformed(
            "auto-select input requires exactly one pointer descriptor",
        ));
    };
    let [map_alias] = hipc.send_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "auto-select input requires exactly one map-alias descriptor",
        ));
    };
    if pointer.index != 0 {
        return Err(IpcWireError::Malformed(
            "auto-select input pointer has an invalid index",
        ));
    }
    if map_alias.mode == BufferMode::Invalid {
        return Err(IpcWireError::Malformed(
            "auto-select input map-alias has an invalid mapping mode",
        ));
    }

    // HIPC auto-select reserves both descriptor slots and places the transfer
    // in exactly one of them, leaving the inactive side as a null placeholder:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L228-L247
    let pointer_present = pointer.address != 0 || pointer.size != 0;
    let map_alias_present = map_alias.address != 0 || map_alias.size != 0;
    match (pointer_present, map_alias_present) {
        (true, false) if pointer.address != 0 => Ok((pointer.address, usize::from(pointer.size))),
        (false, true) if map_alias.address != 0 => Ok((
            map_alias.address,
            usize::try_from(map_alias.size)
                .map_err(|_| IpcWireError::Malformed("input buffer is too large"))?,
        )),
        (false, false) => Ok((0, 0)),
        (true, true) => Err(IpcWireError::Malformed(
            "auto-select input has both descriptor sides active",
        )),
        _ => Err(IpcWireError::Malformed(
            "auto-select input has a null address with nonzero size",
        )),
    }
}

pub(in crate::ipc_wire) fn one_auto_select_output(
    hipc: &HipcRequest<'_>,
) -> Result<(u64, usize), IpcWireError> {
    let ReceiveStatics::Entries(pointers) = &hipc.receive_statics else {
        return Err(IpcWireError::Malformed(
            "auto-select output requires exactly one pointer descriptor",
        ));
    };
    let [
        ReceiveStaticDescriptor {
            address: pointer_address,
            size: pointer_size,
        },
    ] = pointers.as_slice()
    else {
        return Err(IpcWireError::Malformed(
            "auto-select output requires exactly one pointer descriptor",
        ));
    };
    let [map_alias] = hipc.receive_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "auto-select output requires exactly one map-alias descriptor",
        ));
    };
    if map_alias.mode == BufferMode::Invalid {
        return Err(IpcWireError::Malformed(
            "auto-select output map-alias has an invalid mapping mode",
        ));
    }

    let pointer_present = *pointer_address != 0 || *pointer_size != 0;
    let map_alias_present = map_alias.address != 0 || map_alias.size != 0;
    match (pointer_present, map_alias_present) {
        (true, false) if *pointer_address != 0 => {
            Ok((*pointer_address, usize::from(*pointer_size)))
        }
        (false, true) if map_alias.address != 0 => Ok((
            map_alias.address,
            usize::try_from(map_alias.size)
                .map_err(|_| IpcWireError::Malformed("output buffer is too large"))?,
        )),
        (false, false) => Ok((0, 0)),
        (true, true) => Err(IpcWireError::Malformed(
            "auto-select output has both descriptor sides active",
        )),
        _ => Err(IpcWireError::Malformed(
            "auto-select output has a null address with nonzero size",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_wire::message::COMMAND_BUFFER_SIZE;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_send_static(bytes: &mut [u8], offset: usize, address: u64, size: u16) {
        let first = (((address >> 36) as u32 & 0x3f) << 6)
            | (((address >> 32) as u32 & 0xf) << 12)
            | (u32::from(size) << 16);
        put_u32(bytes, offset, first);
        put_u32(bytes, offset + 4, address as u32);
    }

    fn put_buffer(bytes: &mut [u8], offset: usize, address: u64, size: u64) {
        put_u32(bytes, offset, size as u32);
        put_u32(bytes, offset + 4, address as u32);
        put_u32(
            bytes,
            offset + 8,
            ((address >> 36) as u32 & 0x3f_ffff) << 2
                | ((size >> 32) as u32 & 0xf) << 24
                | ((address >> 32) as u32 & 0xf) << 28,
        );
    }

    fn auto_select_input(
        pointer: (u64, u16),
        map_alias: (u64, u64),
    ) -> Result<(u64, usize), IpcWireError> {
        let mut command = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut command, 0, 4 | (1 << 16) | (1 << 20));
        put_send_static(&mut command, 8, pointer.0, pointer.1);
        put_buffer(&mut command, 16, map_alias.0, map_alias.1);
        let hipc = HipcRequest::decode(&command).unwrap();
        one_auto_select_input(&hipc)
    }

    #[test]
    fn auto_select_input_decodes_exactly_one_active_descriptor_side() {
        assert_eq!(auto_select_input((0, 0), (0x2000, 8)), Ok((0x2000, 8)));
        assert_eq!(auto_select_input((0x3000, 4), (0, 0)), Ok((0x3000, 4)));
        assert_eq!(
            auto_select_input((0x1000, 4), (0x2000, 4)),
            Err(IpcWireError::Malformed(
                "auto-select input has both descriptor sides active"
            ))
        );
    }
}
