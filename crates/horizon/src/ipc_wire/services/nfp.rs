//! NFP service lifecycle, reader enumeration, detection and event ownership.
//! https://switchbrew.org/wiki/NFC_services#nfp:user
use super::prelude::*;
use crate::nfp::{NfpObject, NfpSession, NfpUser};
use std::sync::Arc;

fn response(
    session: &NfpSession,
    token: u32,
    result: HorizonIpcResult,
    data: &[u8],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    Ok((
        if session.domain.is_domain() {
            encode_domain_response(token, result, data, &[], &[])?
        } else {
            encode_response(token, result, data, None)?
        },
        None,
    ))
}
pub(in crate::ipc_wire) async fn dispatch_nfp(
    process: &mut Request,
    session: &NfpSession,
    hid: &HidSystem,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return response(
                session,
                request.token,
                if session.domain.close_object(*object_id) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::CMIF_TARGET_NOT_FOUND
                },
                &[],
            );
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            session.target(Some(*object_id))
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain NFP request omitted its domain header",
            ));
        }
        None => session.target(None),
    };
    let Some(target) = target else {
        return response(
            session,
            request.token,
            HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
            &[],
        );
    };
    let user = match target {
        NfpObject::Manager => {
            if request.command_id != 0 {
                return unsupported_service_command("nfp:user", request.command_id);
            }
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let user = NfpObject::User(Arc::new(NfpUser::default()));
            return if session.domain.is_domain() {
                let Some(id) = session.domain.insert_object(user) else {
                    return response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                        &[],
                    );
                };
                semantic_success(request.token, true, &[], &[], &[id], None)
            } else {
                let handle = process
                    .return_service(HorizonIpcObject::Nfp(NfpSession::from_object(user)))
                    .map_err(IpcWireError::Handle)?;
                semantic_success(request.token, false, &[], &[], &[], Some(handle))
            };
        }
        NfpObject::User(user) => user,
    };
    let ok = HorizonIpcResult::SUCCESS;
    match request.command_id {
        0 => {
            // PID, ARUID, reserved PID word and required MCU-version data. Like
            // Eden the virtual reader needs no firmware-version negotiation.
            if request.data.len() < 16
                || hipc.pid.is_none()
                || hipc.send_buffers.len() != 1
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || hipc.receive_statics != ReceiveStatics::None
            {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let reader = user.manager(hid);
            let mut state = reader.state.lock().unwrap();
            state.initialized = true;
            state.device_state = if state.connected { 0 } else { 5 };
            response(session, request.token, ok, &[])
        }
        1 => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let reader = user.reader.lock().unwrap().take();
            drop(reader);
            response(session, request.token, ok, &[])
        }
        2 => {
            if !request.has_payload_size(0) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let [output] = process.receive_statics() else {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            };
            if output.size < 8 {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::NFP_INVALID_ARGUMENT,
                    &[],
                );
            }
            let reader = user.manager(hid);
            let status = {
                let state = reader.state.lock().unwrap();
                if !state.initialized {
                    HorizonIpcResult::NFP_DISABLED
                } else if !state.connected {
                    HorizonIpcResult::NFP_DEVICE_NOT_FOUND
                } else {
                    ok
                }
            };
            if status != ok {
                return response(session, request.token, status, &[]);
            }
            let mut handle = 0_u64.to_le_bytes().to_vec();
            process.write(output, 0, &mut handle).await?;
            response(session, request.token, ok, &1_u32.to_le_bytes())
        }
        19 => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let initialized = user
                .reader
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|reader| reader.state.lock().unwrap().initialized);
            response(
                session,
                request.token,
                ok,
                &u32::from(initialized).to_le_bytes(),
            )
        }
        23 => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            }
            let reader = user.manager(hid);
            let handle = process
                .return_handle(reader.availability.1.clone())
                .map_err(IpcWireError::Handle)?;
            semantic_success(
                request.token,
                session.domain.is_domain(),
                &[],
                &[handle],
                &[],
                None,
            )
        }
        command @ (3..=18 | 20..=22 | 24..=25) => {
            let Some(handle) = request_u64(request.data, 0) else {
                return response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                    &[],
                );
            };
            let reader = user.manager(hid);
            if command == 22 {
                return response(session, request.token, ok, &0xd8_u32.to_le_bytes());
            }
            let status = {
                let mut state = reader.state.lock().unwrap();
                if command == 20 {
                    return response(
                        session,
                        request.token,
                        ok,
                        &if handle == 0 { state.device_state } else { 6 }.to_le_bytes(),
                    );
                }
                if !state.initialized {
                    HorizonIpcResult::NFP_DISABLED
                } else if handle != 0 || !state.connected {
                    HorizonIpcResult::NFP_DEVICE_NOT_FOUND
                } else if command == 3 || command == 25 {
                    if state.device_state != 0 {
                        HorizonIpcResult::NFP_WRONG_DEVICE_STATE
                    } else {
                        state.device_state = 1;
                        ok
                    }
                } else if command == 4 {
                    if matches!(state.device_state, 0 | 1) {
                        state.device_state = 0;
                        ok
                    } else {
                        HorizonIpcResult::NFP_WRONG_DEVICE_STATE
                    }
                } else if matches!(command, 17 | 18 | 21) {
                    ok
                } else {
                    HorizonIpcResult::NFP_WRONG_DEVICE_STATE
                }
            };
            if status != ok || matches!(command, 3 | 4 | 25) {
                return response(session, request.token, status, &[]);
            }
            if command == 21 {
                return response(session, request.token, ok, &0_u32.to_le_bytes());
            }
            let event = if command == 17 {
                reader.activate.1.clone()
            } else {
                reader.deactivate.1.clone()
            };
            let handle = process.return_handle(event).map_err(IpcWireError::Handle)?;
            semantic_success(
                request.token,
                session.domain.is_domain(),
                &[],
                &[handle],
                &[],
                None,
            )
        }
        command => unsupported_service_command("NFP:IUser", command),
    }
}
