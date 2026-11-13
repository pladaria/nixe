//! Native display control ABI; presentation buffers have their own ownership.
use super::*;
use crate::gsv::display::DisplayEvent;
const DEFAULT_DISPLAY: u64 = 0x00746c7561666564;
pub(super) fn dispatch(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if has_ipc_descriptors(hipc) && request.command_id != 32 {
        return Err(IpcWireError::Malformed(
            "unexpected native display descriptors",
        ));
    }
    let mut state = session.state.lock().unwrap();
    let display = state
        .display
        .as_mut()
        .filter(|display| display.initialized)
        .ok_or(IpcWireError::Malformed("display context not initialized"))?;
    let data = request.data;
    let mut output = Vec::new();
    let mut handles = Vec::new();
    // @SPECULATIVE: native display query/event layouts are reconstructed from
    // SDK observations in the reference. The emulated output is one SDR display
    // at 60 Hz; these events never signal GPU fence completion.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
    match (request.command_id, data.len()) {
        (0, 8) => output.extend_from_slice(data),
        (1 | 3, 0) => output.extend_from_slice(&1_u32.to_le_bytes()),
        (5, 0) => output.extend_from_slice(&7_u32.to_le_bytes()),
        (51, 4) => {
            // @SPECULATIVE: this count follows query 5 and configures the
            // session's timestamp recorders. No presentation has occurred yet.
            let count = request_u32(data, 0).unwrap();
            if count > 7 || display.timestamp_recorders.is_some() {
                return Err(IpcWireError::Malformed(
                    "invalid timestamp recorder configuration",
                ));
            }
            display.timestamp_recorders = Some(count);
        }
        (11, 4) if request_u32(data, 0) == Some(1) => {
            // @SPECULATIVE: the global event requested with selector 1 before
            // enumerating displays is treated as a display-connection event.
            // The single virtual output remains connected for this session.
            let event = display
                .connection_event
                .get_or_insert_with(|| nixe_runtime::EventObject::new(process.kernel()));
            handles.push(
                process
                    .return_handle(event.readable())
                    .map_err(IpcWireError::Handle)?,
            );
        }
        (101, 12)
            if request_u64(data, 0) == Some(DEFAULT_DISPLAY) && request_u32(data, 8) == Some(1) =>
        {
            // @SPECULATIVE: 101(name, 1), issued before binding display events,
            // opens the default output; 102 is its matching close operation.
            display.opened = true;
        }
        (102, 8) if request_u64(data, 0) == Some(DEFAULT_DISPLAY) && display.opened => {
            display.opened = false;
        }
        (201, 16) if display.opened => {
            // @SPECULATIVE: the SDK registers a client-chosen layer token,
            // mode and buffer count. The token is also the first argument of
            // 212/228/229. It is an identity, not a pointer to dereference.
            let token = request_u64(data, 0).unwrap();
            let mode = request_u32(data, 8).unwrap();
            let buffer_count = request_u32(data, 12).unwrap();
            if token == 0
                || mode != 2
                || !(1..=15).contains(&buffer_count)
                || display.layers.contains_key(&token)
            {
                return Err(IpcWireError::Malformed(
                    "invalid native display layer registration",
                ));
            }
            display.layers.insert(
                token,
                crate::gsv::display::DisplayLayer {
                    buffer_count,
                    attached: false,
                },
            );
        }
        (202, 8) => {
            if display
                .layers
                .remove(&request_u64(data, 0).unwrap())
                .is_none()
            {
                return Err(IpcWireError::Malformed("unknown native display layer"));
            }
        }
        (212, 8) => {
            let layer = display
                .layers
                .get(&request_u64(data, 0).unwrap())
                .ok_or(IpcWireError::Malformed("unknown native display layer"))?;
            // No frame can enter this layer until PresentClientBuffer is
            // implemented with its fence/retention contract. Initial slots
            // are available; this query never completes a pending frame.
            output.extend_from_slice(&((1_u64 << layer.buffer_count) - 1).to_le_bytes());
        }
        (251 | 252, 16) if request_u64(data, 8) == Some(DEFAULT_DISPLAY) && display.opened => {
            // @SPECULATIVE: observed 251(layer token, display name) binds the
            // layer to the output; paired command 252 removes that association.
            let layer = display
                .layers
                .get_mut(&request_u64(data, 0).unwrap())
                .ok_or(IpcWireError::Malformed("unknown native display layer"))?;
            layer.attached = request.command_id == 251;
        }
        (111, 24)
            if request_u64(data, 0) == Some(DEFAULT_DISPLAY)
                && request_u64(data, 8) == Some(DEFAULT_DISPLAY) =>
        {
            // @SPECULATIVE: the observed request selects an attached layer for
            // the named display route; both endpoints are the default output.
            let token = request_u64(data, 16).unwrap();
            if !display
                .layers
                .get(&token)
                .is_some_and(|layer| layer.attached)
            {
                return Err(IpcWireError::Malformed(
                    "display route requires an attached layer",
                ));
            }
            display.active_layer = Some(token);
        }
        (31 | 32, 4) => {
            let id = request_u32(data, 0).unwrap() as usize;
            let mut slots = display.events.lock().unwrap();
            let entry = slots
                .get_mut(id)
                .ok_or(IpcWireError::Malformed("display event index out of range"))?;
            if request.command_id == 32
                && (hipc.copy_handles.len() != 1
                    || !hipc.move_handles.is_empty()
                    || !hipc.send_buffers.is_empty()
                    || !hipc.receive_buffers.is_empty()
                    || !hipc.exchange_buffers.is_empty()
                    || !process
                        .copy_handle::<nixe_runtime::ReadableEventObject>(0)
                        .is_some_and(|event| {
                            entry
                                .as_ref()
                                .is_some_and(|slot| event.same_identity(&slot.event.readable()))
                        }))
            {
                return Err(IpcWireError::Malformed(
                    "display event duplicate does not match its slot",
                ));
            }
            let slot = entry.get_or_insert_with(|| DisplayEvent {
                event: nixe_runtime::EventObject::new(process.kernel()),
                binding: None,
            });
            handles.push(
                process
                    .return_handle(slot.event.readable())
                    .map_err(IpcWireError::Handle)?,
            );
        }
        (command @ 261..=272, 12) => {
            if request_u64(data, 0) != Some(DEFAULT_DISPLAY) {
                return Err(IpcWireError::Malformed("unknown native display"));
            }
            let id = request_u32(data, 8).unwrap() as usize;
            {
                let mut slots = display.events.lock().unwrap();
                let slot = slots
                    .get_mut(id)
                    .and_then(Option::as_mut)
                    .ok_or(IpcWireError::Malformed("display event not acquired"))?;
                if command % 2 == 0 {
                    slot.binding = None;
                } else {
                    slot.binding = Some(command);
                    if command != 261 {
                        slot.event.signal();
                    }
                }
            }
            if command == 261 {
                display
                    .start_timer(process.kernel())
                    .map_err(|_| IpcWireError::Internal("native display timer stopped"))?;
            }
        }
        (command @ 103..=106, 8) if request_u64(data, 0) == Some(DEFAULT_DISPLAY) => {
            match command {
                103 => {
                    output.extend_from_slice(&60_u32.to_le_bytes());
                    output.extend_from_slice(&0_u32.to_le_bytes());
                }
                104 => {
                    output.extend_from_slice(&1_u64.to_le_bytes());
                    output.extend_from_slice(&0_u32.to_le_bytes());
                }
                105 => output.extend_from_slice(&80_u32.to_le_bytes()),
                106 => {
                    let mode = if session.operation_mode == crate::OperationMode::Console {
                        4_u64
                    } else {
                        2
                    };
                    output.extend_from_slice(&mode.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        _ => {
            return Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::Command {
                    service: "GSV DisplaySession",
                    command_id: request.command_id,
                },
            ));
        }
    }
    Ok(encode_tipc_response(0, &output, &handles, &[])?)
}
