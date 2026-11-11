//! Application play-report IPC and debug logging.
//! https://switchbrew.org/wiki/BCAT_services#prepo:a,_prepo:a2,_prepo:m,_prepo:u,_prepo:s
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/prepo/prepo.cpp
use super::prelude::*;
use crate::PlayReportSession;

const MAX_REPORT_BYTES: u64 = 64 * 1024;
const MAX_REPORT_DEPTH: usize = 64;

pub(in crate::ipc_wire) async fn dispatch_play_report(
    process: &Request,
    session: &PlayReportSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let domain = request.domain.is_some();
    let reply = |result| {
        let bytes = if domain {
            encode_domain_response(request.token, result, &[], &[], &[])?
        } else {
            encode_response(request.token, result, &[], None)?
        };
        Ok((bytes, None))
    };
    match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return reply(if session.domain.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            });
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if *object_id != 1 {
                return reply(HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
            }
            if !input_objects.is_empty() {
                return reply(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain play report omitted its domain header",
            ));
        }
        None => {}
    }
    match request.command_id {
        10200 | 10300 | 10400 => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return reply(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            match request.command_id {
                // Acknowledge the transmission request for the offline service.
                10200 => return reply(HorizonIpcResult::SUCCESS),
                // Eden GetTransmissionStatus returns an s32 status of zero;
                // requesting transmission does not create a pending upload.
                10300 => {
                    return semantic_success(
                        request.token,
                        domain,
                        &0_i32.to_le_bytes(),
                        &[],
                        &[],
                        None,
                    );
                }
                10400 => {
                    let id = PlayReportSession::system_session_id()
                        .map_err(|e| IpcWireError::PlayReport(e.to_string().into()))?;
                    return semantic_success(
                        request.token,
                        domain,
                        &id.to_le_bytes(),
                        &[],
                        &[],
                        None,
                    );
                }
                _ => unreachable!(),
            }
        }
        10100..=10107 => {}
        _ => return unsupported_service_command("prepo:u", request.command_id),
    }
    let inline = if request.command_id % 2 == 1 { 24 } else { 8 };
    if !request.has_payload_size(inline)
        || hipc.pid.is_none()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.receive_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
        || hipc.send_buffers.len() != 1
        || hipc.send_statics.len() != 1
    {
        return reply(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let pointer = &hipc.send_statics[0];
    if pointer.index != 0 {
        return reply(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    if log::log_enabled!(log::Level::Debug)
        && let Err(error) = log_report(process).await
    {
        let error = crate::ipc_wire::HorizonIpcFault::from_wire(error);
        log::error!("[guest] [SaveReport] {error}");
    }
    semantic_success(request.token, domain, &[], &[], &[], None)
}

async fn log_report(process: &Request) -> Result<(), IpcWireError> {
    // The pointer carries the report name and the mapped buffer its MessagePack data.
    // https://git.axenov.dev/Museum/ryujinx/src/commit/6f60e102a29aa04fff5c6bcdec37213b2d8ac646/Ryujinx.Horizon/Prepo/Ipc/PrepoService.cs
    let name = &process.send_statics()[0];
    let payload = &process.send_buffers()[0];
    let size = name
        .size
        .checked_add(payload.size)
        .filter(|&size| size <= MAX_REPORT_BYTES)
        .ok_or_else(|| {
            IpcWireError::PlayReport(
                format!("exceeds the {MAX_REPORT_BYTES}-byte diagnostic limit").into(),
            )
        })?;
    process.reserve_payload(size as usize).await?;
    let mut buffers = [Vec::new(), Vec::new()];
    for (buffer, bytes) in [name, payload].into_iter().zip(&mut buffers) {
        let size = buffer.size as usize;
        bytes.try_reserve_exact(size).map_err(|_| {
            IpcWireError::HostResourceExhausted("allocating a play-report diagnostic buffer")
        })?;
        bytes.resize(size, 0);
        process.read(buffer, 0, bytes).await?;
    }
    let [name, payload] = &buffers;
    let name = std::str::from_utf8(name)
        .map_err(|error| IpcWireError::PlayReport(format!("invalid UTF-8 name: {error}").into()))?
        .trim_end_matches('\0');
    let mut remaining = payload.as_slice();
    let fields = rmpv::decode::read_value_ref_with_max_depth(&mut remaining, MAX_REPORT_DEPTH)
        .map_err(|error| {
            IpcWireError::PlayReport(format!("invalid MessagePack: {error}").into())
        })?;
    if !remaining.is_empty() {
        return Err(IpcWireError::PlayReport(
            "trailing data after the MessagePack value".into(),
        ));
    }
    log::debug!("[guest] [SaveReport] {name:?} {fields}");
    Ok(())
}
