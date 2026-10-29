//! Hardware Opus manager/decoder CMIF ABI. The codec lives in nixe-audio.
//! https://switchbrew.org/wiki/Audio_services#hwopus
//! https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/audio/hardware_opus_decoder_manager.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/audio/hardware_opus_decoder.cpp
use super::prelude::*;
use crate::hwopus::{BAD_ARGUMENT, BUFFER_TOO_SMALL, INPUT_TOO_SMALL, Parameters};
use crate::{OpusDecoderSession, OpusManagerSession};

pub(in crate::ipc_wire) async fn dispatch_opus_manager(
    process: &mut Request,
    session: &OpusManagerSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            let result = if session.domain.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            return opus_error(&request, result);
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return opus_error(&request, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            if *object_id != 1 {
                let Some(decoder) = session.domain.object(*object_id) else {
                    return opus_error(&request, HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
                };
                return dispatch_decode(process, &decoder, request, hipc).await;
            }
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain hwopus request omitted its domain header",
            ));
        }
        None => {}
    }
    let command = request.command_id;
    if command > 9 {
        return unsupported_service_command("hwopus", command);
    }
    let multi = matches!(command, 2 | 3 | 6 | 7 | 9);
    let extended = command >= 4;
    let opening = matches!(command, 0 | 2 | 4 | 6);
    let parameter_size = match (multi, extended) {
        (false, false) => 8,
        (false, true) => 16,
        (true, false) => 0x110,
        (true, true) => 0x118,
    };
    let inline_size = if multi { 0 } else { parameter_size };
    if !request.has_payload_size(inline_size + if opening { 4 } else { 0 })
        || hipc.pid.is_some()
        || hipc.copy_handles.len() != usize::from(opening)
        || !hipc.move_handles.is_empty()
        || !hipc.send_buffers.is_empty()
        || !hipc.receive_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
        || hipc.send_statics.len() != usize::from(multi)
    {
        return opus_error(&request, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let mut pointed = Vec::new();
    let raw = if multi {
        let descriptor = hipc.send_statics[0];
        if usize::from(descriptor.size) != parameter_size {
            return opus_error(&request, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
        }
        pointed.resize(parameter_size, 0);
        process
            .read(&process.send_statics()[0], 0, &mut pointed)
            .await?;
        pointed.as_slice()
    } else {
        &request.data[..parameter_size]
    };
    let params = Parameters::decode(raw, multi, extended);
    if !opening {
        let size = match params.work_size() {
            Ok(size) => size,
            Err(result) => return opus_error(&request, result),
        };
        return semantic_success(
            request.token,
            request.domain.is_some(),
            &size.to_le_bytes(),
            &[],
            &[],
            None,
        );
    }
    let Some(work) = process.copy_handle::<TransferMemoryObject>(0) else {
        return opus_error(
            &request,
            HorizonIpcResult::from_raw(crate::HorizonKernelResult::INVALID_HANDLE.raw()),
        );
    };
    let size = request_u32(request.data, inline_size).unwrap();
    log::debug!(
        "opening Opus decoder: rate={} channels={} streams={:?} large-frames={} work-memory={size:#x}",
        params.sample_rate,
        params.channels,
        params.streams,
        params.large_frames
    );
    let decoder = match session.open(params, work, size) {
        Ok(decoder) => decoder,
        Err(result) => return opus_error(&request, result),
    };
    if request.domain.is_some() {
        let Some(id) = session.domain.insert_object(decoder) else {
            return opus_error(&request, HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
        };
        return semantic_success(request.token, true, &[], &[], &[id], None);
    }
    let handle = process
        .return_handle(HorizonIpcObject::OpusDecoder(decoder))
        .map_err(IpcWireError::Handle)?;
    semantic_success(
        request.token,
        request.domain.is_some(),
        &[],
        &[],
        &[],
        Some(handle),
    )
}

pub(in crate::ipc_wire) async fn dispatch_opus_decoder(
    process: &mut Request,
    session: &OpusDecoderSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match &request.domain {
        Some(DomainRequest::Close { .. }) => {
            return opus_error(&request, HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return opus_error(&request, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            if *object_id != 1 {
                return opus_error(&request, HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
            }
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain Opus decoder request omitted its domain header",
            ));
        }
        None => {}
    }
    dispatch_decode(process, session, request, hipc).await
}

async fn dispatch_decode(
    process: &mut Request,
    session: &OpusDecoderSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let command = request.command_id;
    if command > 9 {
        return unsupported_service_command("IHardwareOpusDecoder", command);
    }
    let multi = matches!(command, 2 | 3 | 5 | 7 | 9);
    let context = matches!(command, 1 | 3);
    let reset = command >= 6;
    if !request.has_payload_size(usize::from(reset))
        || hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.send_statics.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
        || hipc.send_buffers.len() != 1
        || hipc.receive_buffers.len() != usize::from(!context)
    {
        return opus_error(&request, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    if multi != session.state.params.streams.is_some() {
        return opus_error(&request, BAD_ARGUMENT);
    }
    let input = one_send_buffer(process, hipc)?;
    if context {
        // SetContext's input is unused by the service. The host decoder is
        // already resident; there is no ADSP mapping to restore.
        return semantic_success(request.token, request.domain.is_some(), &[], &[], &[], None);
    }
    let output = one_receive_buffer(process, hipc)?;
    if input.size <= 8 {
        return opus_error(&request, INPUT_TOO_SMALL);
    }
    let mut header = [0; 8];
    process.read(input, 0, &mut header).await?;
    let size = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
    if size > session.state.params.input_capacity() || size as u64 > input.size - 8 {
        return opus_error(&request, BUFFER_TOO_SMALL);
    }
    let mut packet = vec![0; size];
    process.read(input, 8, &mut packet).await?;
    let (pcm, samples, time) =
        match session
            .state
            .decode(&packet, output.size as usize, reset && request.data[0] != 0)
        {
            Ok(decoded) => decoded,
            Err(result) => return opus_error(&request, result),
        };
    write_descriptor_bytes(process, output, bytemuck::cast_slice(&pcm)).await?;
    let mut data = [0; 16];
    data[..4].copy_from_slice(&(size as u32 + 8).to_le_bytes());
    data[4..8].copy_from_slice(&samples.to_le_bytes());
    data[8..16].copy_from_slice(&time.to_le_bytes());
    semantic_success(
        request.token,
        request.domain.is_some(),
        &data[..if command >= 4 { 16 } else { 8 }],
        &[],
        &[],
        None,
    )
}

fn opus_error(
    request: &CmifRequest<'_>,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let response = if request.domain.is_some() {
        encode_domain_response(request.token, result, &[], &[], &[])?
    } else {
        encode_response(request.token, result, &[], None)?
    };
    Ok((response, None))
}
