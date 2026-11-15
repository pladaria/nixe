//! The sysmix factory copies a process handle and creates one user session per PID.
//! https://switchbrew.org/wiki/Audio_services#(S2)_audsmx:u
use super::prelude::*;
use crate::sysmix::{AudioMixerSession, MixerObject};
use std::sync::Arc;

pub(in crate::ipc_wire) fn dispatch_audio_mixer(
    process: &mut Request,
    session: &AudioMixerSession,
    system: &crate::sysmix::MixerSystem,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let domain = session.domain.is_domain();
    let error = |result| {
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
                "domain sysmix request omitted its domain header",
            ));
        }
        None => session.target(None),
    };
    let Some(target) = target else {
        return error(HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
    };
    if let MixerObject::User(user) = &target {
        if !Arc::ptr_eq(&user.memory, process.memory()) {
            return error(HorizonIpcResult::AUDIO_INVALID_HANDLE);
        }
        return unsupported_service_command(target.name(), request.command_id);
    }
    if request.command_id != 0 {
        return unsupported_service_command(target.name(), request.command_id);
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
    {
        return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let Some(owner) = process.copy_handle::<nixe_runtime::RunnableProcess>(0) else {
        return error(HorizonIpcResult::AUDIO_INVALID_HANDLE);
    };
    if !Arc::ptr_eq(owner.memory(), process.memory()) {
        return error(HorizonIpcResult::AUDIO_INVALID_HANDLE);
    }
    if request_u32(request.data, 0) != Some(1) {
        return error(HorizonIpcResult::AUDIO_INVALID_REVISION);
    }
    let Some(user) = system.open(&owner) else {
        // @SPECULATIVE: duplicate sessions are documented as an error, but its
        // exact result is unknown. Use the audio service's generic failure.
        return error(HorizonIpcResult::AUDIO_OPERATION_FAILED);
    };
    log::debug!(
        "opened native audio mixer user session for process {}",
        owner.process_id()
    );
    let child = MixerObject::User(user);
    if domain {
        let Some(id) = session.domain.insert_object(child) else {
            return error(HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
        };
        semantic_success(request.token, true, &[], &[], &[id], None)
    } else {
        let handle = process
            .return_service(HorizonIpcObject::AudioMixer(AudioMixerSession::new(child)))
            .map_err(IpcWireError::Handle)?;
        semantic_success(request.token, false, &[], &[], &[], Some(handle))
    }
}
