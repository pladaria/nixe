//! Native device enumeration has a distinct wire ABI from IAudioDevice.
//! https://switchbrew.org/wiki/Audio_services#(S2)_auddmg:u
// @SPECULATIVE: command names, format fields and the 0x228-byte property layout
// are reconstructed from the reference's SDK observations. Device channels are
// queried from our real backend, consistently with IAudioDevice.
// https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/audio/nx2_audio_device_manager.h

use super::super::prelude::*;
use crate::audio_device::native::{DeviceObject, NativeAudioDeviceSession, UserDevice};
use std::sync::Arc;

const NAME_SIZE: usize = 0x100;
const OUTPUT_NAME: &[u8] = b"AudioVirtualOutput0";

fn input<'a>(process: &'a Request, hipc: &HipcRequest<'_>) -> Result<&'a Buffer, IpcWireError> {
    match (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice()) {
        ([_], []) => Ok(&process.send_statics()[0]),
        ([], [_]) => one_send_buffer(process, hipc),
        _ => one_auto_select_input(process, hipc),
    }
}
fn output<'a>(process: &'a Request, hipc: &HipcRequest<'_>) -> Result<&'a Buffer, IpcWireError> {
    match (&hipc.receive_statics, hipc.receive_buffers.as_slice()) {
        (ReceiveStatics::Entries(pointers), []) if pointers.len() == 1 => {
            Ok(&process.receive_statics()[0])
        }
        (ReceiveStatics::None, [_]) => one_receive_buffer(process, hipc),
        _ => one_auto_select_output(process, hipc),
    }
}

pub(in crate::ipc_wire) async fn dispatch_native_audio_device(
    process: &mut Request,
    session: &NativeAudioDeviceSession,
    audio: &crate::ipc_wire::resources::AudioServices,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let domain = session.domain.is_domain();
    let error = |result| {
        if result == HorizonIpcResult::CMIF_INVALID_IN_HEADER {
            log::error!(
                "native audio device rejected command {}: data={:02x?} pid={:?} copies={} moves={} input-pointers={} input-buffers={} output-buffers={} output-pointers={:?}",
                request.command_id,
                request.data,
                hipc.pid,
                hipc.copy_handles.len(),
                hipc.move_handles.len(),
                hipc.send_statics.len(),
                hipc.send_buffers.len(),
                hipc.receive_buffers.len(),
                hipc.receive_statics,
            );
        }
        Ok((
            if domain {
                encode_domain_response(request.token, result, &[], &[], &[])?
            } else {
                encode_response(request.token, result, &[], None)?
            },
            None,
        ))
    };
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return error(if session.domain.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            });
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.target(Some(*object_id))
        }
        None if domain => {
            return Err(IpcWireError::Malformed(
                "domain native audio device request omitted its domain header",
            ));
        }
        None => session.target(None),
    };
    let Some(target) = target else {
        return error(HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
    };
    if let DeviceObject::Manager = target {
        if request.command_id != 0 {
            return unsupported_service_command(target.name(), request.command_id);
        }
        // The SDK sends version(u32), aligned AppletResourceUserId(u64),
        // then a PID-reserved u64 as well as the HIPC PID descriptor.
        if !request.has_payload_size(24)
            || hipc.pid.is_none()
            || has_ipc_descriptors_other_than_pid(hipc)
            || request_u64(request.data, 16) != Some(0)
        {
            return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
        }
        let Some(backend) = audio.backend.as_ref() else {
            return error(HorizonIpcResult::AUDIO_OPERATION_FAILED);
        };
        backend
            .output_layout()
            .map_err(|e| IpcWireError::AudioBackend(e.0.into_boxed_str()))?;
        let child = DeviceObject::User(Arc::new(UserDevice {
            backend: backend.clone(),
            state_event: nixe_runtime::EventObject::new(process.kernel()),
            _applet_resource_user_id: request_u64(request.data, 8).unwrap(),
        }));
        return if domain {
            let Some(id) = session.domain.insert_object(child) else {
                return error(HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
            };
            semantic_success(request.token, true, &[], &[], &[id], None)
        } else {
            let handle = process
                .return_service(HorizonIpcObject::NativeAudioDevice(
                    NativeAudioDeviceSession::new(child),
                ))
                .map_err(IpcWireError::Handle)?;
            semantic_success(request.token, false, &[], &[], &[], Some(handle))
        };
    }
    let DeviceObject::User(user) = target else {
        unreachable!()
    };
    let command = request.command_id;
    if !matches!(command, 0 | 100 | 300 | 401 | 500 | 700 | 800) {
        return unsupported_service_command("nn::audio::devmgr::IUserSession", command);
    }
    let scalar = match command {
        300 | 500 => 16,
        700 => 8,
        _ => 4,
    };
    let has_input = matches!(command, 300 | 401 | 500 | 700);
    let has_output = matches!(command, 0 | 100 | 401);
    if !request.has_payload_size(scalar)
        || hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || (!has_input && (!hipc.send_statics.is_empty() || !hipc.send_buffers.is_empty()))
        || (!has_output
            && (!hipc.receive_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)))
    {
        return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let functionality = match command {
        300 => request_u32(request.data, 12).unwrap(),
        700 => request_u32(request.data, 4).unwrap(),
        800 => 1,
        _ => request_u32(request.data, 0).unwrap(),
    };
    // There is a real playback route, but no connected microphone. Do not
    // advertise an input device that cannot deliver samples.
    // @SPECULATIVE: the SDK uses zero to enumerate available routes. Include
    // the output device for that selection, matching the reference; no input
    // route is advertised without a working capture backend.
    if !matches!(functionality, 0 | 1) && command != 500 {
        if command == 0 {
            return semantic_success(request.token, domain, &0u32.to_le_bytes(), &[], &[], None);
        }
        return error(HorizonIpcResult::AUDIO_NOT_FOUND);
    }
    if has_input {
        let buffer = input(process, hipc)?;
        // Device names arrive as bounded strings, not fixed-size name records.
        // The output listing uses padded records, while SDK input spans can
        // contain just the name and its terminator.
        let mut name = vec![0; buffer.size.min(NAME_SIZE as u64) as usize];
        process.read(buffer, 0, &mut name).await?;
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        if &name[..end] != OUTPUT_NAME {
            return error(HorizonIpcResult::AUDIO_NOT_FOUND);
        }
    }
    if command == 300 {
        let kind = request_u32(request.data, 0).unwrap();
        let channels = request_u32(request.data, 4).unwrap();
        let rate = request_u32(request.data, 8).unwrap();
        // This manager describes playback; the stream service owns transport.
        // These are the PCM formats supported by the shared host audio path.
        let supported = crate::sysmix::supported_format(kind, channels, rate);
        return if supported {
            semantic_success(request.token, domain, &[], &[], &[], None)
        } else {
            error(HorizonIpcResult::AUDIO_NOT_SUPPORTED)
        };
    }
    if matches!(command, 700 | 800) {
        if functionality != 1 {
            return error(HorizonIpcResult::AUDIO_NOT_SUPPORTED);
        }
        return match audio.mixer.engage(
            process.memory(),
            request_u32(request.data, 0).unwrap(),
            command == 700,
        ) {
            Ok(()) => semantic_success(request.token, domain, &[], &[], &[], None),
            Err(result) => error(result),
        };
    }
    if command == 500 {
        let handle = process
            .return_handle(user.state_event.readable())
            .map_err(IpcWireError::Handle)?;
        return semantic_success(request.token, domain, &[], &[handle], &[], None);
    }
    let buffer = output(process, hipc)?;
    let required = if command == 401 { 0x228 } else { NAME_SIZE };
    if buffer.size < required as u64 {
        return if command == 0 {
            semantic_success(request.token, domain, &0u32.to_le_bytes(), &[], &[], None)
        } else {
            error(HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER)
        };
    }
    let mut bytes = vec![0; required];
    bytes[..OUTPUT_NAME.len()].copy_from_slice(OUTPUT_NAME);
    if command == 401 {
        let channels = user
            .backend
            .output_layout()
            .map_err(|e| IpcWireError::AudioBackend(e.0.into_boxed_str()))?
            .channels() as u32;
        for (offset, value) in [
            (0x100, 3u32),
            (0x104, 2),
            (0x108, channels),
            (0x10c, 48000),
            (0x220, 1),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let physical =
            crate::audio_device::DEVICE_NAMES[crate::audio_device::ACTIVE_DEVICE].as_bytes();
        bytes[0x110..0x110 + physical.len()].copy_from_slice(physical);
        bytes[0x218..0x220].copy_from_slice(&10_000_000u64.to_le_bytes());
    }
    process.write(buffer, 0, &mut bytes).await?;
    semantic_success(
        request.token,
        domain,
        if command == 0 { &[1, 0, 0, 0] } else { &[] },
        &[],
        &[],
        None,
    )
}
