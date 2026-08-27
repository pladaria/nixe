use super::*;

impl HorizonSvcDispatcher {
    pub(super) fn connect_to_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        let name = match read_c_name(
            context,
            address,
            crate::ipc_wire::NAMED_PORT_NAME_SIZE,
            0x1f,
        ) {
            Ok(Some(name)) => name,
            Ok(None) => {
                result(context, HorizonKernelResult::OUT_OF_RANGE);
                return resume();
            }
            Err(outcome) => return outcome,
        };
        if let Some(port) = self.named_ports.get(&name).cloned() {
            let session = match port.connect() {
                Ok(session) => session,
                Err(PortError::SessionLimit) => {
                    result(context, HorizonKernelResult::OUT_OF_SESSIONS);
                    return resume();
                }
                Err(PortError::PeerClosed) => {
                    result(context, HorizonKernelResult::PORT_CLOSED);
                    return resume();
                }
                Err(PortError::WrongEndpoint | PortError::NoPendingSession) => {
                    result(context, HorizonKernelResult::INVALID_STATE);
                    return resume();
                }
            };
            match context.process_mut().handles_mut().insert(session) {
                Ok(handle) => {
                    result(context, HorizonKernelResult::SUCCESS);
                    write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
                }
                Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
            }
            return resume();
        }
        match crate::ipc_wire::connect_to_named_port(context.process_mut(), address) {
            Ok(NamedPortResult::Connected(handle)) => {
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
                resume()
            }
            Ok(NamedPortResult::NotFound) => {
                result(context, HorizonKernelResult::NOT_FOUND);
                resume()
            }
            Ok(NamedPortResult::NameOutOfRange) => {
                result(context, HorizonKernelResult::OUT_OF_RANGE);
                resume()
            }
            Ok(NamedPortResult::OutOfHandles) => {
                result(context, HorizonKernelResult::OUT_OF_HANDLES);
                resume()
            }
            Err(error) => reject_ipc(context, 0x1f, error),
        }
    }

    pub(super) fn manage_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        let max_sessions = read_register(context.thread().state(), 2) as u32 as i32;
        let name = match read_c_name(
            context,
            address,
            crate::ipc_wire::NAMED_PORT_NAME_SIZE,
            0x71,
        ) {
            Ok(Some(name)) => name,
            Ok(None) => {
                result(context, HorizonKernelResult::OUT_OF_RANGE);
                return resume();
            }
            Err(outcome) => return outcome,
        };
        if max_sessions < 0 {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        if max_sessions == 0 {
            if self
                .named_ports
                .get(&name)
                .is_some_and(PortObject::server_is_open)
            {
                result(context, HorizonKernelResult::INVALID_STATE);
            } else if self.named_ports.remove(&name).is_some() {
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, 0);
            } else {
                result(context, HorizonKernelResult::NOT_FOUND);
            }
            return resume();
        }
        if self.named_ports.contains_key(&name) {
            result(context, HorizonKernelResult::INVALID_STATE);
            return resume();
        }
        let (server, client) = PortObject::create_pair(max_sessions as usize, false);
        match context.process_mut().handles_mut().insert(server) {
            Ok(handle) => {
                self.named_ports.insert(name, client);
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
            }
            Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
        }
        resume()
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn send_sync_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let tls = GuestVirtualAddress::new(context.thread().state().tpidr_el0());
        let caller_thread_id = context.thread().object().thread_id();
        match crate::ipc_wire::send_sync_request(
            context.process_mut(),
            tls,
            handle,
            self.initial_operation_mode,
            &self.time_environment,
            crate::ipc_wire::HostSystems {
                video: &self.video_system,
                hid: &self.hid_system,
                settings: &self.settings_environment,
                caller_thread_id,
            },
        ) {
            Ok(SyncRequestResult::Success) => {
                self.pending_wakes
                    .remove(&context.thread().object().thread_id());
                result(context, HorizonKernelResult::SUCCESS);
                resume()
            }
            Ok(SyncRequestResult::InvalidHandle) => {
                self.generic_sync_request(context, tls, TLS_COMMAND_BUFFER_SIZE, handle, 0x21)
            }
            Ok(SyncRequestResult::AppletExitRequested) => {
                self.pending_wakes
                    .remove(&context.thread().object().thread_id());
                log::debug!("application applet requested process exit");
                terminate(ExceptionTerminationScope::Process)
            }
            Ok(SyncRequestResult::PendingNvDrv(wait)) => {
                log::debug!(
                    "suspending thread {} for nvdrv {} wait request={:#010x} target={} timeout-us={} event-slot={:?}",
                    caller_thread_id,
                    wait.kind().as_str(),
                    wait.request(),
                    wait.target(),
                    wait.timeout_microseconds(),
                    wait.event_slot().map(|slot| slot.get()),
                );
                self.pending_wakes.insert(
                    context.thread().object().thread_id(),
                    PendingThreadWake {
                        events: vec![wait.wake_event()],
                        deadline: wait.remaining().map(|remaining| {
                            self.virtual_time_ns().saturating_add(
                                u64::try_from(remaining.as_nanos()).unwrap_or(u64::MAX),
                            )
                        }),
                    },
                );
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
            }
            Err(error) => reject_ipc(context, 0x21, error),
        }
    }

    pub(super) fn send_sync_request_with_user_buffer(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = read_register(context.thread().state(), 0);
        let size = read_register(context.thread().state(), 1);
        let handle = read_register(context.thread().state(), 2) as u32;
        // Public ABI and validation reference:
        // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#SendSyncRequestWithUserBuffer
        if !address.is_multiple_of(USER_BUFFER_ALIGNMENT) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        if !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        if size == 0 {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        if address.checked_add(size).is_none_or(|end| address >= end) {
            result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
            return resume();
        }
        let Ok(size) = usize::try_from(size) else {
            result(context, HorizonKernelResult::OUT_OF_RESOURCE);
            return resume();
        };
        let address = GuestVirtualAddress::new(address);
        let caller_thread_id = context.thread().object().thread_id();
        match crate::ipc_wire::send_sync_request_from_buffer(
            context.process_mut(),
            address,
            size,
            handle,
            self.initial_operation_mode,
            &self.time_environment,
            crate::ipc_wire::HostSystems {
                video: &self.video_system,
                hid: &self.hid_system,
                settings: &self.settings_environment,
                caller_thread_id,
            },
        ) {
            Ok(SyncRequestResult::Success) => {
                self.pending_wakes
                    .remove(&context.thread().object().thread_id());
                result(context, HorizonKernelResult::SUCCESS);
                resume()
            }
            Ok(SyncRequestResult::InvalidHandle) => {
                self.generic_sync_request(context, address, size, handle, 0x22)
            }
            Ok(SyncRequestResult::AppletExitRequested) => {
                self.pending_wakes
                    .remove(&context.thread().object().thread_id());
                log::debug!("application applet requested process exit");
                terminate(ExceptionTerminationScope::Process)
            }
            Ok(SyncRequestResult::PendingNvDrv(wait)) => {
                self.pending_wakes.insert(
                    context.thread().object().thread_id(),
                    PendingThreadWake {
                        events: vec![wait.wake_event()],
                        deadline: wait.remaining().map(|remaining| {
                            self.virtual_time_ns().saturating_add(
                                u64::try_from(remaining.as_nanos()).unwrap_or(u64::MAX),
                            )
                        }),
                    },
                );
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
            }
            Err(error) => reject_ipc(context, 0x22, error),
        }
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn send_sync_request_light(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let Some(session) = context
            .process()
            .handles()
            .get_as::<SessionObject>(handle)
            .cloned()
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Client || !session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let owner = session_request_owner(context);
        let mut words = [0_u32; 7];
        for (index, word) in words.iter_mut().enumerate() {
            *word = read_register(context.thread().state(), index as u8 + 1) as u32;
        }
        match session.request(owner, SessionMessage::Light(words)) {
            Ok(SessionRequestResult::Submitted | SessionRequestResult::Waiting) => {
                self.suspend_on_session(context, &session)
            }
            Ok(SessionRequestResult::Response(SessionMessage::Light(response))) => {
                for (index, word) in response.into_iter().enumerate() {
                    write_register(
                        context.thread_mut().state_mut(),
                        index as u8 + 1,
                        u64::from(word),
                    );
                }
                result(context, HorizonKernelResult::SUCCESS);
                resume()
            }
            Ok(SessionRequestResult::Response(
                SessionMessage::Buffer(_) | SessionMessage::TransportedBuffer { .. },
            ))
            | Err(SessionError::MessageKindMismatch) => {
                result(context, HorizonKernelResult::INVALID_STATE);
                resume()
            }
            Err(error) => session_error(context, error),
        }
    }

    pub(super) fn generic_sync_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        address: GuestVirtualAddress,
        size: usize,
        handle: u32,
        immediate: u32,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let Some(session) = context
            .process()
            .handles()
            .get_as::<SessionObject>(handle)
            .cloned()
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Client || session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let owner = session_request_owner(context);
        match session.poll_request(owner) {
            Ok(Some(SessionRequestResult::Waiting | SessionRequestResult::Submitted)) => {
                return self.suspend_on_session(context, &session);
            }
            Ok(Some(SessionRequestResult::Response(response))) => {
                return finish_sync_response(context, address, size, immediate, response);
            }
            Ok(None) => {}
            Err(error) => return session_error(context, error),
        }
        if let Err(error) =
            crate::ipc_wire::validate_writable_ram_range(context.process(), address, size)
        {
            return reject_ipc(context, immediate, error);
        }
        let mut message = Vec::new();
        if message.try_reserve_exact(size).is_err() {
            return ipc_fault(
                immediate,
                IpcWireError::HostResourceExhausted(
                    "allocating a generic session IPC command buffer",
                ),
            );
        }
        message.resize(size, 0);
        if let Err(error) = crate::ipc_wire::read_bytes(context.process(), address, &mut message) {
            return reject_ipc(context, immediate, error);
        }
        let message = match capture_message_handles(context, message, false) {
            Ok(message) => message,
            Err(code) => {
                result(context, code);
                return resume();
            }
        };
        match session.request(owner, message) {
            Ok(SessionRequestResult::Submitted | SessionRequestResult::Waiting) => {
                self.suspend_on_session(context, &session)
            }
            Ok(SessionRequestResult::Response(response)) => {
                finish_sync_response(context, address, size, immediate, response)
            }
            Err(SessionError::MessageKindMismatch) => invalid_state(context),
            Err(error) => session_error(context, error),
        }
    }

    pub(super) fn suspend_on_session(
        &mut self,
        context: &ExceptionDispatchContext<'_>,
        session: &SessionObject,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let event = session.readable_event();
        event.clear();
        self.pending_wakes.insert(
            context.thread().object().thread_id(),
            PendingThreadWake {
                events: vec![event],
                deadline: None,
            },
        );
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
    }
}

pub(super) fn finish_sync_response(
    context: &mut ExceptionDispatchContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    immediate: u32,
    response: SessionMessage,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let response = match materialize_message_handles(context, response) {
        Ok(Some(response)) => response,
        Ok(None) => return invalid_state(context),
        Err(code) => {
            result(context, code);
            return resume();
        }
    };
    if response.len() > size {
        close_encoded_handles(context.process_mut().handles_mut(), &response);
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if let Err(error) = crate::ipc_wire::write_bytes(context.process(), address, &response) {
        close_encoded_handles(context.process_mut().handles_mut(), &response);
        return match error {
            IpcWireError::GuestMemory(fault) => {
                ipc_fault(immediate, IpcWireError::ResponseCommit(fault))
            }
            error => reject_ipc(context, immediate, error),
        };
    }
    result(context, HorizonKernelResult::SUCCESS);
    resume()
}

pub(super) fn invalid_state(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    result(context, HorizonKernelResult::INVALID_STATE);
    resume()
}

// Handle translation follows the public kernel implementation. Client requests
// may copy handles but may not move them; server replies may do both, and moved
// server handles are consumed even when a later handle makes the reply fail:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_server_session.cpp#L150-L233
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_server_session.cpp#L572-L578
pub(super) fn capture_message_handles(
    context: &mut ExceptionDispatchContext<'_>,
    bytes: Vec<u8>,
    allow_move_handles: bool,
) -> Result<SessionMessage, HorizonKernelResult> {
    let Some(message) = decode_transport_header(&bytes)? else {
        return Ok(SessionMessage::Buffer(bytes));
    };
    if !allow_move_handles && !message.move_handles.is_empty() {
        return Err(HorizonKernelResult::INVALID_COMBINATION);
    }
    let mut transfer_error = None;
    let mut copy_handles = Vec::with_capacity(message.copy_handles.len());
    for handle in &message.copy_handles {
        if transfer_error.is_some() {
            copy_handles.push(None);
            continue;
        }
        match copy_ipc_object(context, *handle) {
            Ok(object) => copy_handles.push(object),
            Err(error) => {
                transfer_error = Some(error);
                copy_handles.push(None);
            }
        }
    }

    let mut move_handles = Vec::with_capacity(message.move_handles.len());
    for handle in &message.move_handles {
        if *handle == 0 {
            move_handles.push(None);
            continue;
        }
        if matches!(*handle, CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE) {
            transfer_error = Some(HorizonKernelResult::INVALID_HANDLE);
            move_handles.push(None);
            continue;
        }
        match context.process_mut().handles_mut().close(*handle) {
            Ok(object) if transfer_error.is_none() => move_handles.push(Some(object)),
            Ok(_) => move_handles.push(None),
            Err(_) => {
                transfer_error = Some(HorizonKernelResult::INVALID_HANDLE);
                move_handles.push(None);
            }
        }
    }
    if let Some(error) = transfer_error {
        return Err(error);
    }
    Ok(SessionMessage::TransportedBuffer {
        bytes,
        copy_handles,
        move_handles,
    })
}

pub(super) fn copy_ipc_object(
    context: &ExceptionDispatchContext<'_>,
    handle: u32,
) -> Result<Option<HandleObject>, HorizonKernelResult> {
    match handle {
        0 => Ok(None),
        CURRENT_PROCESS_HANDLE => Ok(Some(HandleObject::new(ProcessObject::new(
            context.process().process_id(),
        )))),
        CURRENT_THREAD_HANDLE => Ok(Some(HandleObject::new(context.thread().object()))),
        _ => context
            .process()
            .handles()
            .get(handle)
            .cloned()
            .map(Some)
            .ok_or(HorizonKernelResult::INVALID_HANDLE),
    }
}

pub(super) fn materialize_message_handles(
    context: &mut ExceptionDispatchContext<'_>,
    message: SessionMessage,
) -> Result<Option<Vec<u8>>, HorizonKernelResult> {
    materialize_message_handles_in_table(context.process_mut().handles_mut(), message)
}

pub(super) fn materialize_message_handles_in_table(
    handles: &mut HandleTable,
    message: SessionMessage,
) -> Result<Option<Vec<u8>>, HorizonKernelResult> {
    let (mut bytes, copy_handles, move_handles) = match message {
        SessionMessage::Buffer(bytes) => return Ok(Some(bytes)),
        SessionMessage::Light(_) => return Ok(None),
        SessionMessage::TransportedBuffer {
            bytes,
            copy_handles,
            move_handles,
        } => (bytes, copy_handles, move_handles),
    };
    let handle_offset = {
        let Some(header) = decode_transport_header(&bytes)? else {
            return Err(HorizonKernelResult::INVALID_COMBINATION);
        };
        if header.copy_handles.len() != copy_handles.len()
            || header.move_handles.len() != move_handles.len()
        {
            return Err(HorizonKernelResult::INVALID_COMBINATION);
        }
        header.handle_offset()
    };

    let mut allocated = Vec::with_capacity(copy_handles.len() + move_handles.len());
    let mut encoded = Vec::with_capacity(copy_handles.len() + move_handles.len());
    for object in copy_handles.into_iter().chain(move_handles) {
        let handle = match object {
            Some(object) => match handles.insert_object(object) {
                Ok(handle) => {
                    allocated.push(handle);
                    handle
                }
                Err(_) => {
                    for handle in allocated {
                        let _ = handles.close(handle);
                    }
                    return Err(HorizonKernelResult::OUT_OF_HANDLES);
                }
            },
            None => 0,
        };
        encoded.push(handle);
    }
    for (index, handle) in encoded.into_iter().enumerate() {
        let offset = handle_offset + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&handle.to_le_bytes());
    }
    Ok(Some(bytes))
}

pub(super) fn decode_transport_header(
    bytes: &[u8],
) -> Result<Option<HipcRequest<'_>>, HorizonKernelResult> {
    let Some(word1) = bytes
        .get(4..8)
        .and_then(|word| <[u8; 4]>::try_from(word).ok())
        .map(u32::from_le_bytes)
    else {
        return Ok(None);
    };
    if word1 >> 31 == 0 {
        return Ok(None);
    }
    let bounded = &bytes[..bytes.len().min(TLS_COMMAND_BUFFER_SIZE)];
    HipcRequest::decode(bounded)
        .map(Some)
        .map_err(|_| HorizonKernelResult::INVALID_COMBINATION)
}

pub(super) fn close_encoded_handles(handles: &mut HandleTable, bytes: &[u8]) {
    let Ok(Some(message)) = decode_transport_header(bytes) else {
        return;
    };
    for handle in message
        .copy_handles
        .iter()
        .chain(&message.move_handles)
        .copied()
    {
        if handle != 0 {
            let _ = handles.close(handle);
        }
    }
}

pub(super) fn session_error(
    context: &mut ExceptionDispatchContext<'_>,
    error: SessionError,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let code = match error {
        SessionError::PeerClosed => HorizonKernelResult::SESSION_CLOSED,
        SessionError::QueueFull => HorizonKernelResult::OUT_OF_RESOURCE,
        SessionError::WrongEndpoint
        | SessionError::NoRequest
        | SessionError::ReplyPending
        | SessionError::MessageKindMismatch => HorizonKernelResult::INVALID_STATE,
    };
    result(context, code);
    resume()
}

pub(super) fn reject_ipc(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    error: IpcWireError,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    match error {
        IpcWireError::GuestMemory(fault) => {
            reject(context, HorizonSvcFault::GuestMemory { immediate, fault })
        }
        error => ipc_fault(immediate, error),
    }
}

fn ipc_fault(immediate: u32, error: IpcWireError) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    ExceptionDispatchOutcome::Fault(HorizonSvcFault::Ipc {
        immediate,
        fault: Box::new(crate::ipc_wire::HorizonIpcFault::from_wire(error)),
    })
}
