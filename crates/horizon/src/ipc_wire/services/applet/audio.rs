//! IAudioController's object-owned expected volumes and fade parameters.
//! https://switchbrew.org/wiki/AM_services#IAudioController
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/am/service/audio_controller.cpp
use super::*;
use crate::object::AppletAudioController;
use std::sync::Mutex;

pub(super) fn dispatch(
    state: &Mutex<AppletAudioController>,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = AudioControllerCommand::decode(request.command_id) else {
        return unsupported_service_command("IAudioController", request.command_id);
    };
    let bytes = match command {
        AudioControllerCommand::GetMainAppletExpectedMasterVolume
        | AudioControllerCommand::GetLibraryAppletExpectedMasterVolume => 0,
        AudioControllerCommand::SetExpectedMasterVolume => 8,
        AudioControllerCommand::ChangeMainAppletMasterVolume => 16,
        AudioControllerCommand::SetTransparentVolumeRate => 4,
    };
    if !request.has_payload_size(bytes) || has_ipc_descriptors(hipc) {
        return applet_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let volume =
        |offset| f32::from_bits(request_u32(request.data, offset).unwrap()).clamp(0.0, 1.0);
    let mut state = state.lock().unwrap();
    match command {
        AudioControllerCommand::GetMainAppletExpectedMasterVolume => {
            applet_data(request.token, &state.main_volume.to_le_bytes())
        }
        AudioControllerCommand::GetLibraryAppletExpectedMasterVolume => {
            applet_data(request.token, &state.library_volume.to_le_bytes())
        }
        AudioControllerCommand::SetExpectedMasterVolume => {
            state.main_volume = volume(0);
            state.library_volume = volume(4);
            applet_data(request.token, &[])
        }
        AudioControllerCommand::ChangeMainAppletMasterVolume => {
            state.main_volume = volume(0);
            state.fade_time_ns = request_u64(request.data, 8).unwrap() as i64;
            applet_data(request.token, &[])
        }
        AudioControllerCommand::SetTransparentVolumeRate => {
            state.transparent_rate = volume(0);
            applet_data(request.token, &[])
        }
    }
}
