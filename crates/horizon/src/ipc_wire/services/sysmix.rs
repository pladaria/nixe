//! Native mixer IPC creates owned streams; transport runs independently of IPC.
//! https://switchbrew.org/wiki/Audio_services#(S2)_audsmx:u
// @SPECULATIVE: stream command names and scalar layouts follow observed SDK
// calls and the reference until public protocol documentation is available.
// https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/audio/nx2_audio_device_manager.h
use super::prelude::*;
use crate::sysmix::{AudioMixerSession, MixerObject};
use std::sync::Arc;

fn child(
    process: &mut Request,
    session: &AudioMixerSession,
    token: u32,
    object: MixerObject,
    data: &[u8],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if session.domain.is_domain() {
        let Some(id) = session.domain.insert_object(object) else {
            return Ok((
                encode_domain_response(
                    token,
                    HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                    &[],
                    &[],
                    &[],
                )?,
                None,
            ));
        };
        semantic_success(token, true, data, &[], &[id], None)
    } else {
        let handle = process
            .return_service(HorizonIpcObject::AudioMixer(AudioMixerSession::new(object)))
            .map_err(IpcWireError::Handle)?;
        semantic_success(token, false, data, &[], &[], Some(handle))
    }
}
pub(in crate::ipc_wire) async fn dispatch_audio_mixer(
    process: &mut Request,
    session: &AudioMixerSession,
    audio: &crate::ipc_wire::resources::AudioServices,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    use HorizonIpcResult as R;
    let domain = session.domain.is_domain();
    let reply = |result, data: &[u8]| {
        Ok((
            if domain {
                encode_domain_response(request.token, result, data, &[], &[])?
            } else {
                encode_response(request.token, result, data, None)?
            },
            None,
        ))
    };
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return reply(
                if session.domain.close_object(*object_id) {
                    R::SUCCESS
                } else {
                    R::CMIF_TARGET_NOT_FOUND
                },
                &[],
            );
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return reply(R::CMIF_INVALID_IN_HEADER, &[]);
            }
            session.target(Some(*object_id))
        }
        None if domain => {
            return Err(IpcWireError::Malformed(
                "domain sysmix request omitted its domain header",
            ));
        }
        None => session.target(None),
    };
    let Some(target) = target else {
        return reply(R::CMIF_TARGET_NOT_FOUND, &[]);
    };
    let command = request.command_id;
    if matches!(target, MixerObject::Manager) {
        if command != 0 {
            return unsupported_service_command(target.name(), command);
        }
        if !request.has_payload_size(16)
            || hipc.pid.is_none()
            || hipc.copy_handles.len() != 1
            || !hipc.move_handles.is_empty()
            || !hipc.send_buffers.is_empty()
            || !hipc.send_statics.is_empty()
            || !hipc.receive_buffers.is_empty()
            || !hipc.exchange_buffers.is_empty()
            || !matches!(hipc.receive_statics, ReceiveStatics::None)
            || request_u64(request.data, 8) != Some(0)
        {
            return reply(R::CMIF_INVALID_IN_HEADER, &[]);
        }
        let Some(owner) = process.copy_handle::<nixe_runtime::RunnableProcess>(0) else {
            return reply(R::AUDIO_INVALID_HANDLE, &[]);
        };
        if !Arc::ptr_eq(owner.memory(), process.memory()) {
            return reply(R::AUDIO_INVALID_HANDLE, &[]);
        }
        if request_u32(request.data, 0) != Some(1) {
            return reply(R::AUDIO_INVALID_REVISION, &[]);
        }
        let Some(user) = audio.mixer.open(&owner) else {
            // @SPECULATIVE: duplicate sessions are documented as an error, but
            // its exact result is unknown. Use the audio generic failure.
            return reply(R::AUDIO_OPERATION_FAILED, &[]);
        };
        log::debug!(
            "opened native audio mixer user session for process {}",
            owner.process_id()
        );
        return child(
            process,
            session,
            request.token,
            MixerObject::User(user),
            &[],
        );
    }
    let memory = match &target {
        MixerObject::User(u) => &u.memory,
        MixerObject::Stream(s) => &s.stream.user.memory,
        _ => unreachable!(),
    };
    if !Arc::ptr_eq(memory, process.memory()) {
        return reply(R::AUDIO_INVALID_HANDLE, &[]);
    }
    let creates_stream = matches!(target, MixerObject::User(_)) && command == 1;
    if hipc.pid.is_some() != creates_stream
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.receive_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
    {
        return reply(R::CMIF_INVALID_IN_HEADER, &[]);
    }
    if let MixerObject::User(user) = &target {
        if !matches!(command, 1..=3) {
            return unsupported_service_command(target.name(), command);
        }
        if !request.has_payload_size(if command == 1 { 32 } else { 0 })
            || !hipc.send_statics.is_empty()
            || hipc.send_buffers.len() != 1
        {
            return reply(R::CMIF_INVALID_IN_HEADER, &[]);
        }
        let input = one_send_buffer(process, hipc)?;
        if command == 1 {
            if request_u32(request.data, 0) != Some(1) {
                return reply(R::AUDIO_NOT_SUPPORTED, &[]);
            }
            let Some(backend) = &audio.backend else {
                return reply(R::AUDIO_OPERATION_FAILED, &[]);
            };
            let stream = match audio.mixer.create(
                user.clone(),
                backend.clone(),
                audio.renderers.devices.clone(),
                process.kernel(),
                request_u64(request.data, 16).unwrap(),
                request_u64(request.data, 24).unwrap(),
                request_u32(request.data, 4).unwrap(),
            ) {
                Ok(s) => s,
                Err(e) => return reply(e, &[]),
            };
            let mut name = vec![0; input.size.min(256) as usize];
            process.read(input, 0, &mut name).await?;
            let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
            let Ok(name) = std::str::from_utf8(&name[..end]) else {
                return reply(R::AUDIO_INVALID_ARGUMENT, &[]);
            };
            stream.stream.rename(name.into());
            let id = stream.stream.id;
            return child(
                process,
                session,
                request.token,
                MixerObject::Stream(stream),
                &id.to_le_bytes(),
            );
        }
        if !input.size.is_multiple_of(4) {
            return reply(R::AUDIO_INVALID_ARGUMENT, &[]);
        }
        // Consume IDs in bounded groups, preserving the request's effect order.
        let mut at = 0;
        while at < input.size {
            let mut ids = vec![0; (input.size - at).min(256) as usize];
            process.read(input, at, &mut ids).await?;
            for id in ids.chunks_exact(4) {
                if let Err(e) = audio.mixer.engage(
                    process.memory(),
                    u32::from_le_bytes(id.try_into().unwrap()),
                    command == 2,
                ) {
                    return reply(e, &[]);
                }
            }
            at += ids.len() as u64;
        }
        return reply(R::SUCCESS, &[]);
    }
    let MixerObject::Stream(session) = &target else {
        unreachable!()
    };
    let stream = &session.stream;
    let scalar = match command {
        0 => 24,
        1..=3 | 5 | 6 | 8 | 12 => 0,
        4 => 16,
        7 | 9 => 8,
        10 | 11 => 4,
        _ => return unsupported_service_command(target.name(), command),
    };
    if !request.has_payload_size(scalar)
        || (command != 12 && (!hipc.send_buffers.is_empty() || !hipc.send_statics.is_empty()))
    {
        return reply(R::CMIF_INVALID_IN_HEADER, &[]);
    }
    let result = match command {
        0 => {
            stream
                .open(
                    request_u32(request.data, 0).unwrap(),
                    request_u32(request.data, 4).unwrap(),
                    request_u32(request.data, 8).unwrap(),
                    request_u32(request.data, 12).unwrap(),
                    request_u32(request.data, 16).unwrap(),
                    request_u32(request.data, 20).unwrap(),
                )
                .await
        }
        1 => {
            stream.close();
            Ok(())
        }
        2 => return reply(R::SUCCESS, &stream.status().to_le_bytes()),
        3 => return reply(R::SUCCESS, &stream.gain().to_le_bytes()),
        4 => stream.set_gain(
            request_f32(request.data, 0).unwrap(),
            request_i64(request.data, 8).unwrap(),
        ),
        5 => {
            let handle = process
                .return_handle(stream.state_event.readable())
                .map_err(IpcWireError::Handle)?;
            return semantic_success(request.token, domain, &[], &[handle], &[], None);
        }
        // The client releases its event copy separately through CloseHandle.
        6 => Ok(()),
        7 => match stream
            .notify(
                request_u32(request.data, 0).unwrap(),
                request_i32(request.data, 4).unwrap(),
            )
            .await
        {
            Ok(()) => {
                let handle = process
                    .return_handle(stream.data_event.readable())
                    .map_err(IpcWireError::Handle)?;
                return semantic_success(request.token, domain, &[], &[handle], &[], None);
            }
            Err(e) => Err(e),
        },
        8 => stream.notify(0, 0).await,
        // Clock/prime are not supported by this output transport; purge is an
        // input operation and must not consume output samples or fabricate it.
        9 | 10 => Err(R::AUDIO_NOT_SUPPORTED),
        11 => return reply(R::AUDIO_NOT_SUPPORTED, &0u32.to_le_bytes()),
        12 => {
            let input = one_send_buffer(process, hipc)?;
            let mut name = vec![0; input.size.min(256) as usize];
            process.read(input, 0, &mut name).await?;
            let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
            match std::str::from_utf8(&name[..end]) {
                Ok(name) => {
                    stream.rename(name.into());
                    Ok(())
                }
                Err(_) => Err(R::AUDIO_INVALID_ARGUMENT),
            }
        }
        _ => unreachable!(),
    };
    reply(result.err().unwrap_or(R::SUCCESS), &[])
}
