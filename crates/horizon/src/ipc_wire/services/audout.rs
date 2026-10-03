//! PCM output IPC. Unsupported formats and commands remain explicit boundaries.
//!
//! Wire ABI: https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/audout.c
//! Queue/state/results: https://github.com/skyline-emu/audio-core/tree/76440e0a3554433398c0ef03233f862a6e86f5ee/out
//!
//! Plain CMIF includes TLS alignment padding and output-pointer sizes after the
//! command parameters. libnx does not clear padding; validate the required
//! parameter prefix and descriptors, not the contents of that transport tail.
//! https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L93-L146

use super::prelude::*;
use crate::audio::OUTPUT_FORMAT;
use crate::ipc_wire::io::validate_writable_ram_range;
use crate::{AudioOutManagerSession, AudioOutSession};

const NAME_SIZE: usize = 256;
// https://github.com/skyline-emu/audio-core/blob/76440e0a3554433398c0ef03233f862a6e86f5ee/common/common.h
const BUFFER_LIMIT: usize = 32;

fn audio_error(error: nixe_audio::AudioError) -> IpcWireError {
    IpcWireError::AudioBackend(error.0.into_boxed_str())
}

fn unsupported<T>(
    service: &'static str,
    command_id: u32,
    detail: &'static str,
) -> Result<T, IpcWireError> {
    Err(IpcWireError::UnsupportedService(
        UnsupportedServiceOperation::CommandVariant {
            service,
            command_id,
            detail,
        },
    ))
}

fn input(hipc: &HipcRequest<'_>, auto: bool) -> Result<(u64, usize), IpcWireError> {
    if auto {
        return one_auto_select_input(hipc);
    }
    let d = one_send_buffer(hipc)?;
    Ok((
        d.address,
        usize::try_from(d.size)
            .map_err(|_| IpcWireError::Malformed("audio input size overflows"))?,
    ))
}

fn output(hipc: &HipcRequest<'_>, auto: bool) -> Result<(u64, usize), IpcWireError> {
    if auto {
        return one_auto_select_output(hipc);
    }
    let d = one_receive_buffer(hipc)?;
    Ok((
        d.address,
        usize::try_from(d.size)
            .map_err(|_| IpcWireError::Malformed("audio output size overflows"))?,
    ))
}

pub(in crate::ipc_wire) fn dispatch_audio_out_manager(
    process: &mut ExceptionProcessContext<'_>,
    session: &AudioOutManagerSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let cmd = request.command_id;
    match cmd {
        0 | 2 => {
            if hipc.pid.is_some()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || (cmd == 0 && !matches!(hipc.receive_statics, ReceiveStatics::None))
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (address, size) = output(hipc, cmd == 2)?;
            let count = u32::from(size >= NAME_SIZE);
            if count != 0 {
                let mut name = [0; NAME_SIZE];
                name[..9].copy_from_slice(b"DeviceOut");
                write_bytes(process, GuestVirtualAddress::new(address), &name)?;
            }
            semantic_success(request.token, false, &count.to_le_bytes(), &[], &[], None)
        }
        1 | 3 => {
            if request.data.len() < 16
                || hipc.pid.is_none()
                || hipc.copy_handles.len() != 1
                || !hipc.move_handles.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || (cmd == 1
                    && (!hipc.send_statics.is_empty()
                        || !matches!(hipc.receive_statics, ReceiveStatics::None)))
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let owner = hipc.copy_handles[0];
            if owner != crate::CURRENT_PROCESS_HANDLE {
                let Some(owner) = process
                    .handles()
                    .get_as::<nixe_runtime::ProcessObject>(owner)
                else {
                    return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_HANDLE);
                };
                if owner.process_id() != process.process_id() {
                    return unsupported(
                        "audout:u",
                        cmd,
                        "audio output for another process is not implemented",
                    );
                }
            }
            let rate = request_u32(request.data, 0).unwrap();
            // OpenAudioOut packs a u16 channel count followed by a reserved u16.
            // libnx passes 0x00020000: the actual channel count is zero (default).
            let channels = request_u32(request.data, 4).unwrap() & 0xffff;
            if rate != 0 && rate != OUTPUT_FORMAT.sample_rate.get() {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_SAMPLE_RATE);
            }
            if !matches!(channels, 0 | 2 | 6) {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_CHANNEL_COUNT);
            }
            if channels == 6 {
                return unsupported("audout:u", cmd, "six-channel PCM output is not implemented");
            }
            let (name_address, name_size) = input(hipc, cmd == 3)?;
            let (out_address, out_size) = output(hipc, cmd == 3)?;
            if name_size < NAME_SIZE || out_size < NAME_SIZE {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER);
            }
            let mut name = [0; NAME_SIZE];
            read_bytes(process, GuestVirtualAddress::new(name_address), &mut name)?;
            let end = name.iter().position(|&v| v == 0).unwrap_or(NAME_SIZE);
            if end != 0 && &name[..end] != b"DeviceOut" {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_NOT_FOUND);
            }
            let Some(backend) = session.0.as_deref() else {
                return unsupported("audout:u", cmd, "no host audio backend is configured");
            };
            name.fill(0);
            name[..9].copy_from_slice(b"DeviceOut");
            write_bytes(process, GuestVirtualAddress::new(out_address), &name)?;
            let output = AudioOutSession::open(backend).map_err(audio_error)?;
            let handle = process
                .handles_mut()
                .insert(HorizonIpcObject::AudioOut(output))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("opening an audio output session")
                })?;
            let mut data = Vec::with_capacity(16);
            for value in [
                OUTPUT_FORMAT.sample_rate.get(),
                OUTPUT_FORMAT.layout.channels() as u32,
                2,
                1,
            ] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            semantic_success(request.token, false, &data, &[], &[], Some(handle))
        }
        _ => unsupported_service_command("audout:u", cmd),
    }
}

pub(in crate::ipc_wire) fn dispatch_audio_out(
    process: &mut ExceptionProcessContext<'_>,
    session: &AudioOutSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let cmd = request.command_id;
    if !matches!(cmd, 0..=10) {
        return unsupported_service_command("IAudioOut", cmd);
    }
    if hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || (!matches!(cmd, 3 | 7) && !hipc.send_buffers.is_empty())
        || (cmd != 7 && !hipc.send_statics.is_empty())
        || (!matches!(cmd, 5 | 8) && !hipc.receive_buffers.is_empty())
        || (cmd != 8 && !matches!(hipc.receive_statics, ReceiveStatics::None))
    {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let size = if matches!(cmd, 3 | 6 | 7) { 8 } else { 0 };
    if request.data.len() < size {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let audio = session
        .output
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut data = Vec::new();
    match cmd {
        0 => {
            data.extend_from_slice(&u32::from(!audio.started().map_err(audio_error)?).to_le_bytes())
        }
        1 => {
            if audio.started().map_err(audio_error)? {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_OPERATION_FAILED);
            }
            audio.start().map_err(audio_error)?;
        }
        2 => audio.stop().map_err(audio_error)?,
        3 | 7 => {
            if audio.owned_buffer_count().map_err(audio_error)? >= BUFFER_LIMIT {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_BUFFER_COUNT_REACHED);
            }
            let (address, size) = input(hipc, cmd == 7)?;
            if size < 40 {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER);
            }
            let mut descriptor = [0; 40];
            read_bytes(process, GuestVirtualAddress::new(address), &mut descriptor)?;
            let next = request_u64(&descriptor, 0).unwrap();
            let samples = request_u64(&descriptor, 8).unwrap();
            let capacity = request_u64(&descriptor, 16).unwrap();
            let length = request_u64(&descriptor, 24).unwrap();
            let offset = request_u64(&descriptor, 32).unwrap();
            if next != 0 || offset != 0 {
                return unsupported(
                    "IAudioOut",
                    cmd,
                    "linked audio buffers and nonzero buffer offsets are not implemented",
                );
            }
            if length > capacity || samples.checked_add(length).is_none() {
                return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_ADDRESS_INFO);
            }
            if !length.is_multiple_of(OUTPUT_FORMAT.frame_bytes() as u64) {
                return unsupported(
                    "IAudioOut",
                    cmd,
                    "partial stereo PCM frames are not implemented",
                );
            }
            let length = usize::try_from(length)
                .map_err(|_| IpcWireError::Malformed("PCM data size overflows"))?;
            let mut pcm = Vec::new();
            pcm.try_reserve_exact(length / 2)
                .map_err(|_| IpcWireError::HostResourceExhausted("copying submitted PCM"))?;
            // Read in bounded chunks; the host callback never touches process memory.
            // Ownership lasts until GetReleasedAudioOutBuffer returns the tag.
            let mut bytes = [0; 4096];
            for cursor in (0..length).step_by(bytes.len()) {
                let count = bytes.len().min(length - cursor);
                read_bytes(
                    process,
                    GuestVirtualAddress::new(samples + cursor as u64),
                    &mut bytes[..count],
                )?;
                pcm.extend(
                    bytes[..count]
                        .chunks_exact(2)
                        .map(|s| i16::from_le_bytes([s[0], s[1]])),
                );
            }
            audio
                .append(request_u64(request.data, 0).unwrap(), pcm)
                .map_err(audio_error)?;
        }
        4 => {
            // Copies refer to the same kernel event, including after handle closure.
            audio.started().map_err(audio_error)?;
            let handle = process
                .handles_mut()
                .insert(session.event.clone())
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("copying an audio buffer event")
                })?;
            return semantic_success(request.token, false, &[], &[handle], &[], None);
        }
        5 | 8 => {
            let (address, size) = output(hipc, cmd == 8)?;
            let capacity = (size / 8).min(BUFFER_LIMIT);
            validate_writable_ram_range(process, GuestVirtualAddress::new(address), capacity * 8)?;
            let tags = audio.released(capacity).map_err(audio_error)?;
            let bytes: Vec<u8> = tags.iter().flat_map(|t| t.to_le_bytes()).collect();
            write_bytes(process, GuestVirtualAddress::new(address), &bytes)?;
            data.extend_from_slice(&(tags.len() as u32).to_le_bytes());
        }
        6 => data.extend_from_slice(
            &u32::from(
                audio
                    .contains(request_u64(request.data, 0).unwrap())
                    .map_err(audio_error)?,
            )
            .to_le_bytes(),
        ),
        9 => data
            .extend_from_slice(&(audio.buffer_count().map_err(audio_error)? as u32).to_le_bytes()),
        10 => data.extend_from_slice(&audio.played_frames().map_err(audio_error)?.to_le_bytes()),
        _ => unreachable!(),
    }
    semantic_success(request.token, false, &data, &[], &[], None)
}
