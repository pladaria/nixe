//! Switch 2's factory shares the existing audio subsystem owners.
//! https://switchbrew.org/wiki/Audio_services#(S2)_aud:u
use super::prelude::*;

pub(in crate::ipc_wire) fn dispatch_audio_user(
    process: &mut Request,
    audio: &crate::ipc_wire::resources::AudioServices,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if has_ipc_descriptors(hipc) || !request.has_payload_size(0) {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let object = match request.command_id {
        1 => HorizonIpcObject::AudioOutManager(audio.output_manager()),
        2 => HorizonIpcObject::AudioRendererManager(audio.renderers.clone()),
        _ => return unsupported_service_command("aud:u", request.command_id),
    };
    let handle = process
        .return_service(object)
        .map_err(IpcWireError::Handle)?;
    semantic_success(request.token, false, &[], &[], &[], Some(handle))
}
