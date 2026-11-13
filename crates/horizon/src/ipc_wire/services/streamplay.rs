//! GameShare guest-controller shared memory, separate from local HID.
//! https://switchbrew.org/wiki/Switch_2%3A_Chat_services#stpl:u
use super::prelude::*;

pub(in crate::ipc_wire) fn dispatch_streamplay(
    process: &mut Request,
    object: &HorizonIpcObject,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if has_ipc_descriptors(hipc) {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    match (object, request.command_id) {
        (HorizonIpcObject::StreamPlay, 0) => {
            let handle = process
                .return_service(HorizonIpcObject::StreamPlayUser(Default::default()))
                .map_err(IpcWireError::Handle)?;
            semantic_success(request.token, false, &[], &[], &[], Some(handle))
        }
        (HorizonIpcObject::StreamPlayUser(_), 1) => {
            if !request.has_payload_size(8) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            // @SPECULATIVE: only the command ID is documented. Observed requests
            // contain two u32 words, whose meaning is unknown. Match Lichii's
            // unimplemented-command reply: success and 32 zeroed output words,
            // without changing guest-controller memory or creating handles.
            // This is not evidence of the command's actual semantics; keep the
            // provisional policy local to this command and interface.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/service.cpp
            static NOTICE: std::sync::Once = std::sync::Once::new();
            NOTICE.call_once(|| {
                log::warn!(
                    "GameShare IUserShimScopedObject command 1: speculative success with zeroed output; semantics undocumented (words={:#x}, {:#x})",
                    request_u32(request.data, 0).unwrap(),
                    request_u32(request.data, 4).unwrap(),
                );
            });
            semantic_success(request.token, false, &[0; 32 * 4], &[], &[], None)
        }
        (HorizonIpcObject::StreamPlayUser(shared), 300) => {
            // @SPECULATIVE: the reference's SDK ABI returns {size, offset}
            // and a read-only copy handle. A zero-filled 0x9b000-byte area
            // describes no GameShare guests. No local pad is mirrored here.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/ngc/guest_hid.h
            const SIZE: usize = 0x9b000;
            let memory = {
                let mut shared = shared.lock().unwrap();
                if shared.is_none() {
                    *shared = Some(
                        crate::system_memory::create(process.memory().system_account(), SIZE, &[])
                            .map_err(IpcWireError::SystemMemory)?,
                    );
                }
                shared.as_ref().unwrap().clone()
            };
            let handle = process
                .return_handle(memory)
                .map_err(IpcWireError::Handle)?;
            let mut data = (SIZE as u64).to_le_bytes().to_vec();
            data.extend_from_slice(&0_u64.to_le_bytes());
            semantic_success(request.token, false, &data, &[handle], &[], None)
        }
        _ => unsupported_service_command(
            if matches!(object, HorizonIpcObject::StreamPlayUser(_)) {
                "IUserShimScopedObject"
            } else {
                "stpl:u"
            },
            request.command_id,
        ),
    }
}
