//! IAudioDevice wire ABI; output state is owned by the audio manager.
//!
//! https://switchbrew.org/wiki/Audio_services#IAudioDevice
//! https://github.com/switchbrew/libnx/blob/master/nx/source/services/auddev.c

use super::prelude::*;
use crate::AudioDeviceSession;
use crate::audio_device::{ACTIVE_DEVICE, DEVICE_NAMES, OUTPUT_NAMES};

const NAME_SIZE: usize = 0x100;

fn output<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
    auto: bool,
) -> Result<&'a Buffer, IpcWireError> {
    if auto {
        return one_auto_select_output(process, hipc);
    }
    let [_] = hipc.receive_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "audio device requires one output buffer",
        ));
    };
    Ok(&process.receive_buffers()[0])
}

pub(in crate::ipc_wire) async fn dispatch_audio_device(
    process: &mut Request,
    session: &AudioDeviceSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let command = request.command_id;
    if !matches!(command, 0..=8 | 10..=14) {
        return unsupported_service_command("IAudioDevice", command);
    }
    if command == 13 && !hipc.send_buffers.is_empty() {
        // Older firmware uses this ID for GetAudioSystemMasterVolumeSetting.
        return Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::CommandVariant {
                service: "IAudioDevice",
                command_id: command,
                detail: "legacy GetAudioSystemMasterVolumeSetting is not implemented",
            },
        ));
    }
    let input = matches!(command, 1 | 2 | 7 | 8);
    let has_output = matches!(command, 0 | 3 | 6 | 10 | 13 | 14);
    let auto = matches!(command, 6..=8 | 10);
    if hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || (!input && (!hipc.send_buffers.is_empty() || !hipc.send_statics.is_empty()))
        || (!has_output
            && (!hipc.receive_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)))
        || (!auto
            && (!hipc.send_statics.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)))
        || (matches!(command, 1 | 7) && !request.has_payload_size(4))
    {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    if input {
        let buffer = if auto {
            one_auto_select_input(process, hipc)?
        } else {
            one_send_buffer(process, hipc)?
        };
        let size = buffer.size as usize;
        if size < NAME_SIZE {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER);
        }
        let mut name = [0; NAME_SIZE];
        process.read(buffer, 0, &mut name).await?;
        let Some(end) = name.iter().position(|&byte| byte == 0) else {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_ARGUMENT);
        };
        let Some(device) = session.volume_device(&name[..end]) else {
            log::debug!(
                "audio device volume: command={command} unknown-name={:?}",
                String::from_utf8_lossy(&name[..end])
            );
            return cmif_error(request.token, HorizonIpcResult::AUDIO_NOT_FOUND);
        };
        if matches!(command, 1 | 7) {
            let volume = request_f32(request.data, 0).unwrap();
            if !volume.is_finite() {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_ARGUMENT);
            }
            session.state.set_volume(device, volume);
            log::debug!(
                "audio device output volume: device={} volume={volume}",
                DEVICE_NAMES[device]
            );
            return semantic_success(request.token, false, &[], &[], &[], None);
        }
        return semantic_success(
            request.token,
            false,
            &session.state.volume(device).to_le_bytes(),
            &[],
            &[],
            None,
        );
    }
    if has_output {
        let buffer = output(process, hipc, auto)?;
        let size = buffer.size as usize;
        let listing = matches!(command, 0 | 6 | 14);
        let names = if listing {
            if command == 14 {
                OUTPUT_NAMES.as_slice()
            } else {
                session.names()
            }
        } else {
            &DEVICE_NAMES[ACTIVE_DEVICE..=ACTIVE_DEVICE]
        };
        let count = (size / NAME_SIZE).min(names.len());
        if !listing && count == 0 {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER);
        }
        let mut data = [0; NAME_SIZE * 4];
        for (index, name) in names.iter().take(count).enumerate() {
            let start = index * NAME_SIZE;
            data[start..start + name.len()].copy_from_slice(name.as_bytes());
        }
        if count != 0 {
            process.write(buffer, 0, &data[..count * NAME_SIZE]).await?;
        }
        let count_bytes = (count as u32).to_le_bytes();
        return semantic_success(
            request.token,
            false,
            if listing { &count_bytes } else { &[] },
            &[],
            &[],
            None,
        );
    }
    if matches!(command, 4 | 11 | 12) {
        let event = if command == 11 {
            &session.input_event
        } else {
            &session.output_event
        };
        let handle = process
            .return_handle(event.clone())
            .map_err(IpcWireError::Handle)?;
        return semantic_success(request.token, false, &[], &[handle], &[], None);
    }
    let channels = session
        .channels()
        .map_err(|error| IpcWireError::AudioBackend(error.0.into_boxed_str()))?;
    semantic_success(
        request.token,
        false,
        &channels.to_le_bytes(),
        &[],
        &[],
        None,
    )
}
