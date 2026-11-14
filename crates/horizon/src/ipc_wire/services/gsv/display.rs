//! Native display control ABI; presentation buffers have their own ownership.
use super::*;
use crate::gsv::display::DisplayEvent;
use std::future::Future;
const DEFAULT_DISPLAY: u64 = 0x00746c7561666564;
fn client_slot(data: &[u8]) -> u32 {
    // @SPECULATIVE: the native SDK writes only the low byte of this field.
    // Subsequent presents leave previous IPC data (including SFCO bytes) in
    // bytes 9..12. Those bytes are padding, not part of the slot identity.
    u32::from(data[8])
}
pub(super) async fn dispatch(
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
    if request.command_id == 229 {
        return present(process, session, request.data).await;
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
                    available: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(
                        (1 << buffer_count) - 1,
                    )),
                    next_present_ns: 0,
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
            output.extend_from_slice(
                &u64::from(layer.available.load(std::sync::atomic::Ordering::Acquire))
                    .to_le_bytes(),
            );
        }
        (228, 12) => {
            // @SPECULATIVE: cancel returns a client-owned slot without submitting
            // an image. An in-flight slot cannot be cancelled by another request.
            let layer = display
                .layers
                .get(&request_u64(data, 0).unwrap())
                .ok_or(IpcWireError::Malformed("unknown native display layer"))?;
            let slot = client_slot(data);
            let available = layer.available.load(std::sync::atomic::Ordering::Acquire);
            if slot >= layer.buffer_count || available & (1 << slot) == 0 {
                return Err(IpcWireError::Malformed(
                    "cannot cancel an in-flight native display buffer",
                ));
            }
            output.extend_from_slice(&u64::from(available).to_le_bytes());
            output.resize(24, 0);
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
        (107, 8) if request_u64(data, 0) == Some(DEFAULT_DISPLAY) => {
            // @SPECULATIVE: only the command ID is documented. The observed
            // request names the default display; its output semantics are
            // unknown. Match the reference's explicit Unknown107 handler with
            // success and 32 zeroed words. Do not change display state, publish
            // buffers, or signal an event based on this provisional reply.
            // https://switchbrew.org/wiki/Switch_2%3A_GSV_services#DisplaySession
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            static NOTICE: std::sync::Once = std::sync::Once::new();
            NOTICE.call_once(|| {
                log::warn!(
                    "GSV DisplaySession command 107: speculative success with zeroed output; semantics undocumented"
                );
            });
            output.resize(32 * 4, 0);
        }
        (351, 16) => {
            // @SPECULATIVE: Switchbrew lists this command without a signature
            // or semantics. The observed SDK request contains two u64 words.
            // Match the reference's explicit Unknown351 zeroed reply; do not
            // interpret either word as a buffer, event or GPU completion.
            // https://switchbrew.org/wiki/Switch_2%3A_GSV_services#DisplaySession
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            static NOTICE: std::sync::Once = std::sync::Once::new();
            NOTICE.call_once(|| log::warn!("GSV DisplaySession command 351: speculative success with zeroed output; semantics undocumented"));
            output.resize(32 * 4, 0);
        }
        _ => {
            log::debug!(
                "unimplemented GSV DisplaySession command={} input={:02x?}",
                request.command_id,
                data
            );
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

async fn present(
    process: &mut Request,
    session: &crate::GsvSession,
    data: &[u8],
) -> Result<Vec<u8>, IpcWireError> {
    // @SPECULATIVE: the native presentation packet is reconstructed from the
    // reference's SDK observations: token/slot/image at 0/8/12, drawn size at
    // 20, interval:refresh at 40, inverted counter ID/value at 80/88. Remaining
    // metadata is undocumented. Only the observed 32-bit counter mode is accepted.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
    // NativeSyncHandle's mode is a byte, followed by three padding bytes.
    if data.len() != 96 || data[84] != 0 || request_u32(data, 92) != Some(0) {
        return Err(IpcWireError::Malformed(
            "unsupported native presentation packet/counter mode",
        ));
    }
    let token = request_u64(data, 0).unwrap();
    let slot = client_slot(data);
    let image_id = request_u32(data, 12).unwrap();
    let shown = request_u32(data, 20).unwrap();
    let rate = request_u32(data, 40).unwrap();
    let interval = rate >> 16;
    let refresh = rate & 0xffff;
    if refresh != 60 {
        return Err(IpcWireError::GraphicsService(
            format!("unsupported native presentation refresh {refresh} Hz").into(),
        ));
    }
    let counter_id = !request_u32(data, 80).unwrap();
    let threshold = request_u32(data, 88).unwrap();
    let queued = std::time::Instant::now();
    let (frame, counter, executor, deadline) = {
        let mut state = session.state.lock().unwrap();
        let gpu = state
            .gpu
            .as_ref()
            .ok_or(IpcWireError::Malformed("GPU context not initialized"))?;
        let executor = gpu
            .executor
            .clone()
            .ok_or(IpcWireError::Malformed("native GPU not running"))?;
        let counter = if counter_id == u32::MAX {
            None
        } else {
            Some(
                state
                    .syncpt
                    .as_ref()
                    .and_then(|context| context.timelines.get(&counter_id))
                    .cloned()
                    .ok_or(IpcWireError::Malformed("unknown native presentation fence"))?,
            )
        };
        let (&first, pool) = state
            .native_images
            .window_pools
            .range(..=image_id)
            .next_back()
            .filter(|(first, pool)| image_id - **first < pool.count)
            .ok_or(IpcWireError::Malformed("unknown native presentation image"))?;
        let pool = pool.clone();
        let mut image = pool
            .presentation(image_id - first, &gpu.spaces)
            .map_err(IpcWireError::GraphicsService)?
            .image
            .clone();
        if shown != 0 {
            let [width, height] = [(shown & 0xffff) + 1, (shown >> 16) + 1];
            if width > image.extent[0] || height > image.extent[1] {
                return Err(IpcWireError::Malformed(
                    "native presentation crop exceeds its image",
                ));
            }
            image.crop.width = width;
            image.crop.height = height;
        }
        let display = state
            .display
            .as_mut()
            .filter(|d| d.initialized && d.opened && d.active_layer == Some(token))
            .ok_or(IpcWireError::Malformed(
                "native presentation requires an active display layer",
            ))?;
        let layer = display
            .layers
            .get_mut(&token)
            .filter(|l| l.attached && slot < l.buffer_count)
            .ok_or(IpcWireError::Malformed(
                "invalid native presentation layer/slot",
            ))?;
        let bit = 1 << slot;
        layer
            .available
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |free| (free & bit != 0).then_some(free & !bit),
            )
            .map_err(|_| IpcWireError::Malformed("native presentation slot is still in flight"))?;
        let now = process.kernel().clock().scheduler_time_ns();
        let duration = u64::from(interval) * 1_000_000_000 / u64::from(refresh);
        layer.next_present_ns = if layer.next_present_ns == 0 {
            now + duration
        } else {
            layer.next_present_ns.saturating_add(duration).max(now)
        };
        let frame = crate::gsv::display::Frame {
            image,
            _pool: pool,
            available: layer.available.clone(),
            slot,
            interval,
            queued,
        };
        (frame, counter, executor, layer.next_present_ns)
    };
    if let Some(counter) = counter {
        let mut wait = std::pin::pin!(crate::gsv::syncpt::CounterWait::new(counter, threshold));
        std::future::poll_fn(|cx| {
            process.lifetime().check()?;
            wait.as_mut().poll(cx).map(Ok::<(), IpcWireError>)
        })
        .await?;
    }
    let available = frame.available.clone();
    executor
        .present(frame)
        .map_err(IpcWireError::GraphicsService)?
        .wait()
        .await
        // The owner reports the original device failure. A dropped reply
        // cancels this waiter, rather than creating a second failure at teardown.
        .map_err(|_| IpcWireError::Cancelled)?;
    let mut pacing = std::pin::pin!(process.kernel().deadline(deadline));
    std::future::poll_fn(|cx| {
        process.lifetime().check()?;
        pacing
            .as_mut()
            .poll(cx)
            .map(|result| result.map_err(|_| IpcWireError::Cancelled))
    })
    .await?;
    let mut output = u64::from(available.load(std::sync::atomic::Ordering::Acquire))
        .to_le_bytes()
        .to_vec();
    output.resize(24, 0);
    Ok(encode_tipc_response(0, &output, &[], &[])?)
}
