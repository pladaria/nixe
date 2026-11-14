//! Codec contexts own their creator-session lifecycle independently of GPU channels.
//! https://switchbrew.org/wiki/Switch_2%3A_GSV_services#CreateVideoDecoderContext
use super::*;
use crate::gsv::{GsvInterface, VideoContext};

pub(super) fn context(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    let encoder = request.command_id >= 1000;
    let operation = request.command_id % 100;
    let valid_input = match operation {
        1..=3 => request.data.is_empty(),
        _ => false,
    };
    if !valid_input || has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed("invalid GSV video context request"));
    }
    let mut state = session.state.lock().unwrap();
    if !state.context_created {
        return Err(IpcWireError::Malformed("GSV context has not been created"));
    }
    let slot = if encoder {
        &mut state.video_encoder
    } else {
        &mut state.video_decoder
    };
    match operation {
        1 => {
            if slot.is_some() {
                return Err(IpcWireError::Malformed("GSV video context already exists"));
            }
            *slot = Some(VideoContext::default());
        }
        3 => {
            slot.take()
                .ok_or(IpcWireError::Malformed("GSV video context does not exist"))?;
        }
        _ => {
            let context = slot
                .as_mut()
                .ok_or(IpcWireError::Malformed("GSV video context does not exist"))?;
            if context.initialized {
                return Err(IpcWireError::Malformed(
                    "GSV video context already initialized",
                ));
            }
            let interface = if encoder {
                GsvInterface::VideoEncoderCreator
            } else {
                GsvInterface::VideoDecoderCreator
            };
            let handle = process
                .return_service(HorizonIpcObject::Gsv(session.child(interface)))
                .map_err(IpcWireError::Handle)?;
            context.initialized = true;
            return Ok(encode_tipc_response(0, &[], &[handle], &[])?);
        }
    }
    Ok(encode_tipc_response(0, &[], &[], &[])?)
}

pub(super) fn creator(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if request.command_id != 1
        || !request.data.is_empty()
        || has_ipc_descriptors(hipc)
        || !matches!(session.interface, GsvInterface::VideoDecoderCreator)
    {
        return Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::Command {
                service: "GSV VideoCreatorSession",
                command_id: request.command_id,
            },
        ));
    }
    let (counter, publication, device) = {
        let mut s = session.state.lock().unwrap();
        if !s.video_decoder.as_ref().is_some_and(|c| c.initialized) {
            return Err(IpcWireError::Malformed(
                "decoder context is not initialized",
            ));
        }
        let sync = s
            .syncpt
            .as_mut()
            .filter(|s| s.initialized)
            .ok_or(IpcWireError::Malformed("video requires syncpoint context"))?;
        let counter = sync
            .allocate(session.process_id)
            .map_err(IpcWireError::GraphicsService)?;
        let publication = sync
            .publication()
            .map_err(|e| IpcWireError::SystemMemory(std::sync::Arc::new(e)))?;
        let device = s
            .gpu
            .as_ref()
            .and_then(|g| g.executor.as_ref())
            .and_then(|e| e.device_uuid());
        (counter, publication, device)
    };
    let decoder = crate::gsv::video::Session::new(
        process.lifetime().client().process(),
        counter,
        publication,
        device,
    )
    .map_err(IpcWireError::GraphicsService)?;
    let handle = process
        .return_service(HorizonIpcObject::Gsv(
            session.child(GsvInterface::VideoDecoder(std::sync::Arc::new(decoder))),
        ))
        .map_err(IpcWireError::Handle)?;
    Ok(encode_tipc_response(0, &[], &[handle], &[])?)
}

pub(super) fn images(
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    use crate::gsv::video_image::{Layout, Pool};
    if has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed(
            "decoder image request has descriptors",
        ));
    }
    let data = request.data;
    let mut s = session.state.lock().unwrap();
    let mut output = Vec::new();
    match (request.command_id, data.len()) {
        (991, 68) | (911, 72) => {
            let layout = Layout::new(std::array::from_fn(|i| request_u32(data, i * 4).unwrap()))
                .map_err(IpcWireError::GraphicsService)?;
            let count = request_u32(data, 64).unwrap();
            let bytes = layout.bytes(count).map_err(IpcWireError::GraphicsService)?;
            if request.command_id == 991 {
                output.extend_from_slice(&bytes.to_le_bytes());
                output.extend_from_slice(&8192_u64.to_le_bytes());
            } else {
                let id = request_u32(data, 68).unwrap();
                let backing = s
                    .native_images
                    .memories
                    .get(&id)
                    .and_then(|m| m.backing.clone())
                    .ok_or(IpcWireError::Malformed(
                        "decoder image pool requires bound memory",
                    ))?;
                if bytes > backing.ranges.bytes() {
                    return Err(IpcWireError::Malformed(
                        "decoder images exceed bound memory",
                    ));
                }
                let first = s
                    .native_images
                    .next_image
                    .checked_add(1)
                    .ok_or(IpcWireError::HostResourceExhausted("native image IDs"))?;
                s.native_images.next_image = s
                    .native_images
                    .next_image
                    .checked_add(count)
                    .ok_or(IpcWireError::HostResourceExhausted("native image IDs"))?;
                let pool = std::sync::Arc::new(Pool { layout, backing });
                for index in 0..count {
                    let id = first + index;
                    s.video_images.insert(
                        id,
                        std::sync::Arc::new(crate::gsv::video_image::Image {
                            id,
                            pool: pool.clone(),
                            index,
                        }),
                    );
                }
                output.extend_from_slice(&first.to_le_bytes());
            }
        }
        (912, 4) => {
            let id = request_u32(data, 0).unwrap();
            // @SPECULATIVE: destruction removes the public image handle, as
            // in the reference service. Pending decoder work owns its image
            // and captured backing until completion; host reference counts
            // are not a guest-visible prerequisite for destroying the handle.
            s.video_images
                .remove(&id)
                .ok_or(IpcWireError::Malformed("unknown decoder image"))?;
        }
        (961, 4) => {
            output.extend_from_slice(
                &image(&s, request_u32(data, 0).unwrap())?
                    .cpu_descriptor()
                    .map_err(IpcWireError::GraphicsService)?,
            );
        }
        _ => {
            return Err(IpcWireError::GraphicsService(
                format!(
                    "unsupported decoder image request: command={} data={data:02x?}",
                    request.command_id
                )
                .into(),
            ));
        }
    }
    Ok(encode_tipc_response(0, &output, &[], &[])?)
}
fn image(
    state: &crate::gsv::GsvState,
    id: u32,
) -> Result<std::sync::Arc<crate::gsv::video_image::Image>, IpcWireError> {
    state
        .video_images
        .get(&id)
        .cloned()
        .ok_or(IpcWireError::Malformed("unknown decoder image"))
}

pub(super) async fn dispatch(
    process: &mut Request,
    session: &crate::GsvSession,
    decoder: &crate::gsv::video::Session,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    use crate::gsv::video::{Config, Reset, WorkMemory};
    let data = request.data;
    let cmd = request.command_id;
    if hipc.pid.is_some()
        || !hipc.move_handles.is_empty()
        || !hipc.receive_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
    {
        return Err(IpcWireError::Malformed("invalid video decoder descriptors"));
    }
    if !matches!(cmd, 200..=202 | 210) && !hipc.copy_handles.is_empty() {
        return Err(IpcWireError::Malformed(
            "unexpected video decoder input handles",
        ));
    }
    if !matches!(cmd, 101 | 109 | 400)
        && (!hipc.send_buffers.is_empty() || !hipc.send_statics.is_empty())
    {
        return Err(IpcWireError::Malformed("unexpected video decoder buffers"));
    }
    // @SPECULATIVE: parameter packing and command variants below follow native
    // SDK traces in the reference. Unknown layouts remain explicit failures.
    let mut output = Vec::new();
    let mut handles = Vec::new();
    match (cmd, data.len()) {
        (200, 20) | (201, 24) | (202, 32) => {
            if hipc.copy_handles.len() != 1 {
                return Err(IpcWireError::Malformed(
                    "decoder initialize requires transfer memory",
                ));
            }
            let memory = process
                .copy_handle::<TransferMemoryObject>(0)
                .filter(|m| Some(m.size()) == request_u64(data, 0))
                .ok_or(IpcWireError::Malformed("invalid decoder work memory"))?;
            let mut slot = decoder.memory.lock().unwrap();
            if slot.is_some() {
                return Err(IpcWireError::Malformed("decoder already initialized"));
            }
            *slot = Some(WorkMemory::Transfer { _memory: memory });
        }
        (210, 16) => {
            if hipc.copy_handles.len() != 1 {
                return Err(IpcWireError::Malformed(
                    "decoder remote initialization requires a process",
                ));
            }
            let owner = process
                .copy_handle::<nixe_runtime::RunnableProcess>(0)
                .filter(|p| p.process_id() == session.process_id)
                .ok_or(IpcWireError::Malformed("invalid decoder remote process"))?;
            let mut slot = decoder.memory.lock().unwrap();
            if slot.is_some() {
                return Err(IpcWireError::Malformed("decoder already initialized"));
            }
            *slot = Some(WorkMemory::Remote { _process: owner });
        }
        (3, 0) => handles.push(
            process
                .return_handle(decoder.event())
                .map_err(IpcWireError::Handle)?,
        ),
        (101, 16 | 32) => {
            // @SPECULATIVE: newer clients append sixteen zero bytes to the
            // four documented configuration words. Their meaning is unknown;
            // accept the observed empty extension, not nonzero options.
            if data[16..].iter().any(|&b| b != 0) {
                return Err(IpcWireError::Malformed(
                    "unknown decoder configuration extension",
                ));
            }
            let buffer = process
                .send_buffers()
                .first()
                .or_else(|| process.send_statics().first())
                .ok_or(IpcWireError::Malformed(
                    "decoder configuration requires MIME name",
                ))?;
            if buffer.size > 256 {
                return Err(IpcWireError::Malformed("decoder MIME name too long"));
            }
            let mut mime = vec![0; buffer.size as usize];
            process.read(buffer, 0, &mut mime).await?;
            let name = mime.split(|b| *b == 0).next().unwrap_or_default();
            let codec = match name {
                b"video/vp9" => 9,
                b"video/avc" => 3,
                _ => {
                    return Err(IpcWireError::GraphicsService(
                        format!("unsupported video MIME {}", String::from_utf8_lossy(name)).into(),
                    ));
                }
            };
            decoder
                .configure(Config {
                    codec,
                    width: request_u32(data, 0).unwrap(),
                    height: request_u32(data, 4).unwrap(),
                })
                .map_err(IpcWireError::GraphicsService)?;
        }
        (6, 0) => decoder.start().map_err(IpcWireError::GraphicsService)?,
        (8 | 207 | 115, 0) => decoder
            .reset(match cmd {
                8 => Reset::Flush,
                207 => Reset::Stop,
                _ => Reset::Finalize,
            })
            .await
            .map_err(IpcWireError::GraphicsService)?,
        (109 | 400, 12) => {
            if process.send_buffers().len() != if cmd == 400 { 2 } else { 1 } {
                return Err(IpcWireError::Malformed("decoder packet descriptors"));
            }
            let buffer = &process.send_buffers()[0];
            // End-of-stream is an ABI byte; the three trailing padding bytes
            // are not part of the boolean value.
            if buffer.size > 64 * 1024 * 1024 || data[8] > 1 {
                return Err(IpcWireError::GraphicsService(
                    format!(
                        "invalid compressed video packet: bytes={} data={data:02x?}",
                        buffer.size
                    )
                    .into(),
                ));
            }
            if cmd == 400 && process.send_buffers()[1].size != 0x838 {
                return Err(IpcWireError::Malformed("decoder frame information layout"));
            }
            let mut bytes = vec![0; buffer.size as usize];
            process.read(buffer, 0, &mut bytes).await?;
            match decoder
                // @SPECULATIVE: input time uses signed microseconds, matching
                // the signed TimeSpan returned by GetOutputBuffer. An empty
                // EOS request may carry -1; preserve it without treating it as
                // a huge future timestamp or manufacturing a decoded picture.
                .submit(bytes, request_u64(data, 0).unwrap() as i64, data[8] != 0)
                .map_err(IpcWireError::GraphicsService)?
            {
                Ok(sequence) => output.extend_from_slice(&sequence.to_le_bytes()),
                Err(result) => return Ok(encode_tipc_response(result, &[], &[], &[])?),
            }
        }
        (311, 4 | 16) | (314, 24 | 32) => {
            // @SPECULATIVE: current clients extend registration with three
            // zero words; only the image ID has verified semantics.
            if cmd == 311 && data[4..].iter().any(|&b| b != 0) {
                return Err(IpcWireError::Malformed(
                    "unknown decoder registration options",
                ));
            }
            let (image, release) = {
                let state = session.state.lock().unwrap();
                let image = image(&state, request_u32(data, 0).unwrap())?;
                let release = if cmd == 314 {
                    // @SPECULATIVE: the extended release request appends two
                    // zero words. A zero NativeSyncHandle carries no dependency,
                    // consistent with its invalid complemented counter ID.
                    if data[24..].iter().any(|&b| b != 0) {
                        return Err(IpcWireError::Malformed("unknown decoder release options"));
                    }
                    let id = !request_u32(data, 8).unwrap();
                    if request_u32(data, 4) != Some(0)
                        // SyncValueMode occupies one byte inside NativeSyncHandle.
                        || data[12] != 0
                        || request_u32(data, 20) != Some(0)
                    {
                        return Err(IpcWireError::Malformed(
                            "unsupported decoder image release mode",
                        ));
                    }
                    if data[8..24].iter().all(|&b| b == 0) {
                        None
                    } else {
                        let counter = state
                            .syncpt
                            .as_ref()
                            .and_then(|s| s.timelines.get(&id))
                            .cloned()
                            .ok_or(IpcWireError::Malformed("unknown decoder release counter"))?;
                        Some((counter, request_u32(data, 16).unwrap()))
                    }
                } else {
                    None
                };
                (image, release)
            };
            output.extend_from_slice(
                &decoder
                    .bind(image, release, cmd == 311)
                    .map_err(IpcWireError::GraphicsService)?,
            );
        }
        (212, 4) => decoder
            .unbind(request_u32(data, 0).unwrap())
            .map_err(IpcWireError::GraphicsService)?,
        (213, 0) => {
            let Some(frame) = decoder.output() else {
                return Ok(encode_tipc_response(
                    crate::gsv::video::EMPTY,
                    &[],
                    &[],
                    &[],
                )?);
            };
            let pts = frame.pts * 1000;
            for word in [
                frame.image,
                frame.width,
                frame.height,
                frame.width,
                frame.height,
                0,
                pts as u32,
                (pts >> 32) as u32,
                u32::from(frame.end),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ] {
                output.extend_from_slice(&word.to_le_bytes());
            }
        }
        _ => {
            return Err(IpcWireError::GraphicsService(
                format!("unsupported GSV VideoDecoderSession command={cmd} data={data:02x?}")
                    .into(),
            ));
        }
    }
    Ok(encode_tipc_response(0, &output, &handles, &[])?)
}
