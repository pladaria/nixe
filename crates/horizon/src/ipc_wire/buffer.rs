//! Validation for HIPC buffer descriptor shapes shared by service ABIs.

use super::IpcWireError;
use super::message::{HipcRequest, ReceiveStaticDescriptor, ReceiveStatics};
use super::request::{Buffer, Request};

pub(in crate::ipc_wire) fn one_receive_buffer<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
) -> Result<&'a Buffer, IpcWireError> {
    match hipc.receive_buffers.as_slice() {
        [descriptor] if descriptor.size > 0 => Ok(&process.receive_buffers()[0]),
        _ => Err(IpcWireError::Malformed(
            "service command requires exactly one output buffer",
        )),
    }
}

pub(in crate::ipc_wire) fn one_send_buffer<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
) -> Result<&'a Buffer, IpcWireError> {
    match hipc.send_buffers.as_slice() {
        [_] => Ok(&process.send_buffers()[0]),
        _ => Err(IpcWireError::Malformed(
            "request requires exactly one input buffer",
        )),
    }
}

pub(in crate::ipc_wire) fn one_auto_select_input<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
) -> Result<&'a Buffer, IpcWireError> {
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

    // HIPC auto-select reserves both descriptor slots and places the transfer
    // in exactly one of them, leaving the inactive side as a null placeholder:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L228-L247
    let pointer_present = pointer.address != 0 || pointer.size != 0;
    let map_alias_present = map_alias.address != 0 || map_alias.size != 0;
    match (pointer_present, map_alias_present) {
        (true, false) if pointer.address != 0 => Ok(&process.send_statics()[0]),
        (false, true) if map_alias.address != 0 => Ok(&process.send_buffers()[0]),
        (false, false) => Ok(&process.send_buffers()[0]),
        (true, true) => Err(IpcWireError::Malformed(
            "auto-select input has both descriptor sides active",
        )),
        _ => Err(IpcWireError::Malformed(
            "auto-select input has a null address with nonzero size",
        )),
    }
}

pub(in crate::ipc_wire) fn one_auto_select_output<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
) -> Result<&'a Buffer, IpcWireError> {
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

    let pointer_present = *pointer_address != 0 || *pointer_size != 0;
    let map_alias_present = map_alias.address != 0 || map_alias.size != 0;
    match (pointer_present, map_alias_present) {
        (true, false) if *pointer_address != 0 => Ok(&process.receive_statics()[0]),
        (false, true) if map_alias.address != 0 => Ok(&process.receive_buffers()[0]),
        (false, false) => Ok(&process.receive_buffers()[0]),
        (true, true) => Err(IpcWireError::Malformed(
            "auto-select output has both descriptor sides active",
        )),
        _ => Err(IpcWireError::Malformed(
            "auto-select output has a null address with nonzero size",
        )),
    }
}
