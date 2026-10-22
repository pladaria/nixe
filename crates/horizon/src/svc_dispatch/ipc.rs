use super::*;

#[derive(Debug)]
pub(super) struct KernelSessionRequest {
    session: SessionObject,
    owner: SessionRequestOwner,
    reply: Option<Vec<nixe_memory::AllocationRange>>,
    size: usize,
    immediate: u32,
}
impl Drop for KernelSessionRequest {
    fn drop(&mut self) {
        self.session
            .cancel_request(self.owner)
            .expect("accepted request owns a client endpoint");
    }
}

#[derive(Clone, Copy)]
enum NamedPortOperation {
    Connect,
    Manage(i32),
}
impl NamedPortOperation {
    fn immediate(self) -> u32 {
        match self {
            Self::Connect => 0x1f,
            Self::Manage(_) => 0x71,
        }
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn connect_to_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        self.read_named_port_name(
            context,
            address,
            Vec::with_capacity(crate::ipc_wire::NAMED_PORT_NAME_SIZE),
            NamedPortOperation::Connect,
        )
    }
    pub(super) fn manage_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        let maximum = read_register(context.thread().state(), 2) as u32 as i32;
        self.read_named_port_name(
            context,
            address,
            Vec::with_capacity(crate::ipc_wire::NAMED_PORT_NAME_SIZE),
            NamedPortOperation::Manage(maximum),
        )
    }
    fn read_named_port_name(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        address: GuestVirtualAddress,
        mut name: Vec<u8>,
        operation: NamedPortOperation,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        self.read_svc_input(
            context,
            operation.immediate(),
            address,
            1,
            move |dispatcher, context, bytes| {
                if bytes[0] == 0 {
                    return match operation {
                        NamedPortOperation::Connect => dispatcher.connect_named_port(context, name),
                        NamedPortOperation::Manage(maximum) => {
                            dispatcher.manage_port(context, name, maximum)
                        }
                    };
                }
                name.push(bytes[0]);
                if name.len() == crate::ipc_wire::NAMED_PORT_NAME_SIZE {
                    result(context, HorizonKernelResult::OUT_OF_RANGE);
                    return resume();
                }
                let Some(address) = address.checked_add(1) else {
                    result(context, HorizonKernelResult::INVALID_ADDRESS);
                    return resume();
                };
                dispatcher.read_named_port_name(context, address, name, operation)
            },
        )
    }
    fn connect_named_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        name: Vec<u8>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
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
            match context.process_mut().handles().insert(session) {
                Ok(handle) => {
                    result(context, HorizonKernelResult::SUCCESS);
                    write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
                }
                Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
            }
            return resume();
        }
        match crate::ipc_wire::connect_to_named_port(context.process_mut(), &name) {
            NamedPortResult::Connected(handle) => {
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
                resume()
            }
            NamedPortResult::NotFound => {
                result(context, HorizonKernelResult::NOT_FOUND);
                resume()
            }
            NamedPortResult::OutOfHandles => {
                result(context, HorizonKernelResult::OUT_OF_HANDLES);
                resume()
            }
        }
    }
    fn manage_port(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        name: Vec<u8>,
        max_sessions: i32,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
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
        match context.process_mut().handles().insert(server) {
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
        let tls = thread_tls(context.thread().state());
        self.start_hle_request(context, tls, TLS_COMMAND_BUFFER_SIZE, handle, 0x21)
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
        self.start_hle_request(context, address, size, handle, 0x22)
    }

    fn start_hle_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        address: GuestVirtualAddress,
        size: usize,
        handle: u32,
        immediate: u32,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        if self
            .session_requests
            .contains_key(&context.thread().id().get())
        {
            return self.generic_sync_request(context, address, size, handle, immediate);
        }
        let Some(target) = context.process().handles().lookup(handle) else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if target.downcast_ref::<crate::HorizonIpcObject>().is_none() {
            return self.generic_sync_request(context, address, size, handle, immediate);
        }
        let key = crate::ipc_wire::operation::ServiceKey::Session(target.clone());
        let turn = match self.service_scheduling.enter(key) {
            Ok(turn) => turn,
            Err(error) => return ipc_fault(immediate, error),
        };
        let mut request = crate::ipc_wire::request::Request::new(context, target);
        if let Err(error) = request.pin_command(address, size) {
            return ipc_fault(immediate, error);
        }
        let mode = self.initial_operation_mode;
        let time = self.time_environment.clone();
        let systems = crate::ipc_wire::HostSystems {
            video: self.video_system.clone(),
            keyboard: self.keyboard_host.clone(),
            audio_backend: self.audio_backend.clone(),
            audio_renderers: self.audio_renderers.clone(),
            hid: self.hid_system.clone(),
            settings: self.settings_environment.clone(),
            application_language: self.application_language,
            user_account_switch_locked: self.user_account_switch_locked,
            save_data: self.save_data.clone(),
            shared_fonts: self.shared_fonts.clone(),
            system_archives: self.system_archives.clone(),
            diagnostics: self.diagnostics,
            caller_thread_id: context.thread().id().get(),
            scheduling: self.service_scheduling.clone(),
        };
        let source = context.source();
        let vcpu = context.thread().vcpu();
        self.start_owned_svc(context, request.lifetime(), None, async move {
            let result = crate::ipc_wire::send_sync_request_from_buffer(
                &mut request,
                address,
                size,
                handle,
                mode,
                &time,
                &systems,
                turn,
            )
            .await
            .map_err(|error| HorizonSvcFault::Ipc {
                immediate,
                fault: Box::new(HorizonIpcFault::from_wire(error)),
            })?;
            match result {
                SyncRequestResult::Success => Ok(super::memory_wait::SvcCompletion::success(None)),
                SyncRequestResult::AppletExitRequested => {
                    Ok(super::memory_wait::SvcCompletion::ExitProcess { source, vcpu })
                }
                SyncRequestResult::InvalidHandle => {
                    Err(runtime_fault("accepted HLE target changed type"))
                }
            }
        })
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
            .lookup_as::<SessionObject>(handle)
            .map(std::sync::Arc::unwrap_or_clone)
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
        if let Some(request) = self.session_requests.remove(&context.thread().id().get()) {
            return self.poll_session_request(context, request);
        }
        let Some(session) = context
            .process()
            .handles()
            .lookup_as::<SessionObject>(handle)
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Client || session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let ranges = match context.process().memory().space().capture(
            address,
            size as u64,
            nixe_memory::MemoryPermissions::READ_WRITE,
        ) {
            Ok(ranges) => ranges,
            Err(error) => {
                return reject(
                    context,
                    HorizonSvcFault::MemoryAccess {
                        immediate,
                        fault: std::sync::Arc::new(error),
                    },
                );
            }
        };
        let request = KernelSessionRequest {
            session: std::sync::Arc::unwrap_or_clone(session),
            owner: session_request_owner(context),
            reply: Some(ranges.clone()),
            size,
            immediate,
        };
        self.read_svc_capture(
            context,
            immediate,
            ranges,
            move |dispatcher, context, bytes| {
                let message = match capture_message_handles(context, bytes.into_vec(), false) {
                    Ok(message) => message,
                    Err(code) => {
                        result(context, code);
                        return resume();
                    }
                };
                match request.session.request(request.owner, message) {
                    Ok(SessionRequestResult::Submitted | SessionRequestResult::Waiting) => {
                        dispatcher.poll_session_request(context, request)
                    }
                    Ok(SessionRequestResult::Response(response)) => {
                        dispatcher.finish_session_response(context, request, response)
                    }
                    Err(error) => session_error(context, error),
                }
            },
        )
    }
    fn poll_session_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        request: KernelSessionRequest,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let event = request.session.readable_event();
        event.clear();
        match request.session.poll_request(request.owner) {
            Ok(Some(SessionRequestResult::Response(response))) => {
                self.finish_session_response(context, request, response)
            }
            Ok(Some(SessionRequestResult::Waiting | SessionRequestResult::Submitted)) => {
                let thread = context.thread().id().get();
                self.pending_wakes.insert(
                    thread,
                    PendingThreadWake {
                        events: vec![event],
                        deadline: None,
                    },
                );
                self.session_requests.insert(thread, request);
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
            }
            Ok(None) => ExceptionDispatchOutcome::Fault(runtime_fault(
                "accepted session request disappeared",
            )),
            Err(error) => session_error(context, error),
        }
    }
    fn finish_session_response(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        mut request: KernelSessionRequest,
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
        if response.bytes.len() > request.size {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        self.write_received_message(
            context,
            request.immediate,
            request.reply.take().expect("accepted reply capture"),
            response,
            None,
        )
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
        match context.process_mut().handles().close(*handle) {
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
            .lookup(handle)
            .map(Some)
            .ok_or(HorizonKernelResult::INVALID_HANDLE),
    }
}

pub(super) struct ReceivedMessage {
    pub(super) bytes: Vec<u8>,
    reservations: Vec<nixe_runtime::HandleReservation>,
}
pub(super) fn materialize_message_handles(
    context: &mut ExceptionDispatchContext<'_>,
    message: SessionMessage,
) -> Result<Option<ReceivedMessage>, HorizonKernelResult> {
    materialize_message_handles_in_table(context.process().handles(), message)
}
pub(super) fn materialize_message_handles_in_table(
    handles: &std::sync::Arc<HandleTable>,
    message: SessionMessage,
) -> Result<Option<ReceivedMessage>, HorizonKernelResult> {
    let (mut bytes, copied, moved) = match message {
        SessionMessage::Buffer(bytes) => {
            return Ok(Some(ReceivedMessage {
                bytes,
                reservations: Vec::new(),
            }));
        }
        SessionMessage::Light(_) => return Ok(None),
        SessionMessage::TransportedBuffer {
            bytes,
            copy_handles,
            move_handles,
        } => (bytes, copy_handles, move_handles),
    };
    let offset = {
        let header =
            decode_transport_header(&bytes)?.ok_or(HorizonKernelResult::INVALID_COMBINATION)?;
        if header.copy_handles.len() != copied.len() || header.move_handles.len() != moved.len() {
            return Err(HorizonKernelResult::INVALID_COMBINATION);
        }
        header.handle_offset()
    };
    let mut reservations = Vec::with_capacity(copied.len() + moved.len());
    for (index, object) in copied.into_iter().chain(moved).enumerate() {
        let handle = if let Some(object) = object {
            let reservation = handles
                .reserve_object(object)
                .map_err(|_| HorizonKernelResult::OUT_OF_HANDLES)?;
            let handle = reservation.handle();
            reservations.push(reservation);
            handle
        } else {
            0
        };
        bytes[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&handle.to_le_bytes());
    }
    Ok(Some(ReceivedMessage {
        bytes,
        reservations,
    }))
}
impl HorizonSvcDispatcher {
    pub(super) fn write_received_message(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        ranges: Vec<nixe_memory::AllocationRange>,
        mut message: ReceivedMessage,
        output1: Option<u64>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        if message.bytes.is_empty() {
            result(context, HorizonKernelResult::SUCCESS);
            if let Some(value) = output1 {
                write_register(context.thread_mut().state_mut(), 1, value);
            }
            return resume();
        }
        let ranges = match nixe_memory::CapturedMemory::new(ranges) {
            Ok(ranges) => ranges,
            Err(error) => {
                return reject(
                    context,
                    HorizonSvcFault::MemoryAccess {
                        immediate,
                        fault: std::sync::Arc::new(error),
                    },
                );
            }
        };
        let access = context
            .process()
            .memory()
            .account()
            .machine()
            .write_captured(&ranges, 0, &message.bytes);
        let handles = context.process().handles().clone();
        match access {
            Err(error) => reject(
                context,
                HorizonSvcFault::MemoryAccess {
                    immediate,
                    fault: std::sync::Arc::new(error),
                },
            ),
            Ok(nixe_memory::Access::Ready(())) => {
                if let Err(error) = handles.publish(&mut message.reservations) {
                    return ipc_fault(immediate, IpcWireError::Handle(error));
                }
                result(context, HorizonKernelResult::SUCCESS);
                if let Some(value) = output1 {
                    write_register(context.thread_mut().state_mut(), 1, value);
                }
                resume()
            }
            Ok(nixe_memory::Access::Pending(access)) => {
                self.start_owned_svc(context, std::sync::Arc::default(), None, async move {
                    match access
                        .await
                        .map_err(|fault| HorizonSvcFault::MemoryAccess { immediate, fault })?
                    {
                        nixe_memory::AccessResult::Written => {}
                        _ => return Err(runtime_fault("received IPC write completion")),
                    }
                    handles
                        .publish(&mut message.reservations)
                        .map_err(|error| HorizonSvcFault::Ipc {
                            immediate,
                            fault: Box::new(HorizonIpcFault::from_wire(IpcWireError::Handle(
                                error,
                            ))),
                        })?;
                    Ok(super::memory_wait::SvcCompletion::success(output1))
                })
            }
        }
    }
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

fn ipc_fault(immediate: u32, error: IpcWireError) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    ExceptionDispatchOutcome::Fault(HorizonSvcFault::Ipc {
        immediate,
        fault: Box::new(crate::ipc_wire::HorizonIpcFault::from_wire(error)),
    })
}
