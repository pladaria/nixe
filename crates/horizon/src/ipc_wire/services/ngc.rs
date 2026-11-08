//! NGC's content version follows Eden's advertised interface revision.
//! Filtering commands remain unsupported; no unchecked match result is returned.
//! https://switchbrew.org/wiki/NGC_services
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/ngc/ngc.cpp
use super::prelude::*;
use crate::object::NgcSession;

pub(in crate::ipc_wire) fn dispatch_ngc(
    session: &NgcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let domain = session.domain.is_domain();
    let error = |result| -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
        if domain {
            Ok((
                encode_domain_response(request.token, result, &[], &[], &[])?,
                None,
            ))
        } else {
            Ok((encode_response(request.token, result, &[], None)?, None))
        }
    };
    match &request.domain {
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
            if *object_id != 1 {
                return error(HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
            }
            if !input_objects.is_empty() {
                return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
        }
        None if domain => {
            return Err(IpcWireError::Malformed(
                "domain NGC request omitted its domain header",
            ));
        }
        None => {}
    }
    if request.command_id != 0 {
        return unsupported_service_command("ngc:u", request.command_id);
    }
    if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
        return error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    // NgcServiceImpl::NgcContentVersion, not a claimed firmware dictionary.
    semantic_success(request.token, domain, &1_u32.to_le_bytes(), &[], &[], None)
}
