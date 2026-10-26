//! Application play-report IPC and local persistence.
//! https://switchbrew.org/wiki/BCAT_services#prepo:a,_prepo:a2,_prepo:m,_prepo:u,_prepo:s
//! https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/prepo/prepo.cpp
use super::prelude::*;
use crate::PlayReportSession;

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
    if request.command_id == 10400 {
        if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
            return reply(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
        }
        let id = PlayReportSession::system_session_id()
            .map_err(|e| IpcWireError::PlayReport(e.to_string().into()))?;
        return semantic_success(request.token, domain, &id.to_le_bytes(), &[], &[], None);
    }
    if !(10100..=10107).contains(&request.command_id) {
        return unsupported_service_command("prepo:u", request.command_id);
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
    let mapped = &hipc.send_buffers[0];
    let pointer = &hipc.send_statics[0];
    if mapped.mode == BufferMode::Invalid || pointer.index != 0 {
        return reply(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let mut parts = [Vec::new(), Vec::new()];
    for (part, (address, size)) in parts.iter_mut().zip([
        (mapped.address, mapped.size),
        (pointer.address, u64::from(pointer.size)),
    ]) {
        let size = usize::try_from(size)
            .map_err(|_| IpcWireError::Malformed("play report size overflows"))?;
        part.try_reserve_exact(size)
            .map_err(|_| IpcWireError::HostResourceExhausted("reading a play report"))?;
        part.resize(size, 0);
        read_bytes(process, GuestVirtualAddress::new(address), part).await?;
    }
    let path = session
        .save(
            request.command_id,
            process.process_id(),
            &request.data[..inline],
            [&parts[0], &parts[1]],
        )
        .map_err(|e| IpcWireError::PlayReport(e.to_string().into()))?;
    log::debug!("saved play report {}", path.display());
    semantic_success(request.token, domain, &[], &[], &[], None)
}
