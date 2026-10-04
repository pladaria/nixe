//! Application error-context writer object lifecycle.
use super::prelude::*;
use crate::object::ErrorContextWriterSession;

pub(in crate::ipc_wire) fn dispatch_error_context_writer(
    session: &ErrorContextWriterSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let reply_error = |result| {
        Ok((
            encode_domain_response(request.token, result, &[], &[], &[])?,
            None,
        ))
    };
    let id = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return reply_error(if session.domain.close_object(*object_id) {
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
                return reply_error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            *object_id
        }
        None => {
            return Err(IpcWireError::Malformed(
                "error-context writer requires a domain request",
            ));
        }
    };
    if id != 1 {
        if session.domain.object(id).is_none() {
            return reply_error(HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
        }
        return unsupported_service_command("IContextRegistrar", request.command_id);
    }
    // CreateContextRegistrar has no input/output data and returns a child
    // interface. Registering/committing actual context bytes is a distinct
    // operation and remains explicitly unsupported.
    // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/glue/ectx.cpp
    if request.command_id != 0 {
        return unsupported_service_command("ectx:aw", request.command_id);
    }
    if !request.data.is_empty() || has_ipc_descriptors(hipc) {
        return reply_error(HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let id = session
        .domain
        .insert_object(())
        .ok_or(IpcWireError::HostResourceExhausted(
            "allocating an error-context registrar",
        ))?;
    semantic_success(request.token, true, &[], &[], &[id], None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: u32, close: bool) -> [u8; 0x100] {
        let mut bytes = [0; 0x100];
        bytes[..4].copy_from_slice(&4_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&12_u32.to_le_bytes());
        bytes[16] = if close { 2 } else { 1 };
        if !close {
            bytes[18..20].copy_from_slice(&16_u16.to_le_bytes());
            bytes[32..36].copy_from_slice(&0x4943_4653_u32.to_le_bytes());
        }
        bytes[20..24].copy_from_slice(&id.to_le_bytes());
        bytes
    }

    #[test]
    fn registrar_creation_and_close_share_domain_lifetime_but_not_unimplemented_commands() {
        let session = ErrorContextWriterSession::default();
        session.domain.convert();
        let cloned = session.clone();
        let create = request(1, false);
        let hipc = HipcRequest::decode(&create).unwrap();
        let cmif = CmifRequest::decode(&hipc, true).unwrap();
        let (response, handle) = dispatch_error_context_writer(&session, cmif, &hipc).unwrap();
        assert_eq!(handle, None);
        assert_eq!(u32::from_le_bytes(response[16..20].try_into().unwrap()), 1);
        let id = u32::from_le_bytes(response[48..52].try_into().unwrap());
        assert_eq!(id, 2);
        let call = request(id, false);
        let hipc = HipcRequest::decode(&call).unwrap();
        assert!(matches!(
            dispatch_error_context_writer(
                &cloned,
                CmifRequest::decode(&hipc, true).unwrap(),
                &hipc
            ),
            Err(IpcWireError::UnsupportedService(_))
        ));
        let close = request(id, true);
        let hipc = HipcRequest::decode(&close).unwrap();
        dispatch_error_context_writer(&cloned, CmifRequest::decode(&hipc, true).unwrap(), &hipc)
            .unwrap();
        let hipc = HipcRequest::decode(&call).unwrap();
        let (response, _) = dispatch_error_context_writer(
            &session,
            CmifRequest::decode(&hipc, true).unwrap(),
            &hipc,
        )
        .unwrap();
        assert_eq!(
            u32::from_le_bytes(response[40..44].try_into().unwrap()),
            HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
        );
    }
}
