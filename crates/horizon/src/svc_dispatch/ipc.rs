use super::*;

#[derive(Clone, Copy)]
enum MessagePart {
    Header,
    Special,
    Data,
}

// Capture the emitted message, as MessageBuffer's raw-data extent does.
// Receive lists belong to the destination and are not emitted source data.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L584-L615
struct MessageRead {
    machine: nixe_memory::Machine,
    ranges: nixe_memory::CapturedMemory,
    header: [u8; 12],
    bytes: Vec<u8>,
    part: MessagePart,
    pending: Option<nixe_memory::PendingAccess>,
    immediate: u32,
}
impl MessageRead {
    fn prefix(&self) -> usize {
        if u32::from_le_bytes(self.header[4..8].try_into().unwrap()) >> 31 != 0 {
            12
        } else {
            8
        }
    }
    fn read(&mut self) -> Result<nixe_memory::Access<()>, nixe_memory::MemoryError> {
        match self.part {
            MessagePart::Header => {
                self.machine
                    .read_captured_into(&self.ranges, 0, &mut self.header[..8])
            }
            MessagePart::Special => {
                self.machine
                    .read_captured_into(&self.ranges, 8, &mut self.header[8..])
            }
            MessagePart::Data => {
                let prefix = self.prefix();
                self.machine.read_captured_into(
                    &self.ranges,
                    prefix as u64,
                    &mut self.bytes[prefix..],
                )
            }
        }
    }
    fn accept(&mut self, bytes: &[u8]) {
        match self.part {
            MessagePart::Header => self.header[..8].copy_from_slice(bytes),
            MessagePart::Special => self.header[8..].copy_from_slice(bytes),
            MessagePart::Data => {
                let prefix = self.prefix();
                self.bytes[prefix..].copy_from_slice(bytes);
            }
        }
    }
    fn advance(&mut self) -> Result<bool, HorizonKernelResult> {
        match self.part {
            MessagePart::Header if self.prefix() == 12 => {
                self.part = MessagePart::Special;
                Ok(false)
            }
            MessagePart::Header | MessagePart::Special => {
                let prefix = self.prefix();
                let size = HipcHeader::decode(&self.header[..prefix])
                    .and_then(|header| header.message_size(self.ranges.bytes() as usize))
                    .map_err(|_| HorizonKernelResult::INVALID_COMBINATION)?;
                self.bytes = vec![0; size];
                self.bytes[..prefix].copy_from_slice(&self.header[..prefix]);
                self.part = MessagePart::Data;
                Ok(size == prefix)
            }
            MessagePart::Data => Ok(true),
        }
    }
}

impl std::future::Future for MessageRead {
    type Output = Result<Result<Vec<u8>, HorizonKernelResult>, HorizonSvcFault>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::{future::Future, pin::Pin, task::Poll};
        let message = self.get_mut();
        loop {
            if let Some(pending) = message.pending.as_mut() {
                let result = match Future::poll(Pin::new(pending), context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(result) => result,
                };
                message.pending = None;
                match result {
                    Ok(nixe_memory::AccessResult::Read(bytes)) => message.accept(&bytes),
                    Ok(_) => return Poll::Ready(Err(runtime_fault("IPC input read completion"))),
                    Err(fault) => {
                        return Poll::Ready(Err(HorizonSvcFault::MemoryAccess {
                            immediate: message.immediate,
                            fault,
                        }));
                    }
                }
            } else {
                match message.read() {
                    Ok(nixe_memory::Access::Ready(())) => {}
                    Ok(nixe_memory::Access::Pending(pending)) => {
                        message.pending = Some(pending);
                        continue;
                    }
                    Err(fault) => {
                        return Poll::Ready(Err(HorizonSvcFault::MemoryAccess {
                            immediate: message.immediate,
                            fault: std::sync::Arc::new(fault),
                        }));
                    }
                }
            }
            match message.advance() {
                Ok(true) => {
                    return Poll::Ready(Ok(Ok(std::mem::take(&mut message.bytes))));
                }
                Ok(false) => {}
                Err(code) => return Poll::Ready(Ok(Err(code))),
            }
        }
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn read_svc_message(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        ranges: Vec<nixe_memory::AllocationRange>,
        continuation: impl FnOnce(
            &mut Self,
            &mut ExceptionDispatchContext<'_>,
            Result<Vec<u8>, HorizonKernelResult>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
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
        let mut message = MessageRead {
            machine: context.process().memory().account().machine().clone(),
            ranges,
            header: [0; 12],
            bytes: Vec::new(),
            part: MessagePart::Header,
            pending: None,
            immediate,
        };
        // MessageRead is Unpin: ready RAM completes on this stack, and an
        // actual dependency moves the same owned operation into the kernel wait.
        match std::future::Future::poll(
            std::pin::Pin::new(&mut message),
            &mut std::task::Context::from_waker(std::task::Waker::noop()),
        ) {
            std::task::Poll::Ready(Ok(bytes)) => continuation(self, context, bytes),
            std::task::Poll::Ready(Err(fault)) => reject(context, fault),
            std::task::Poll::Pending => {
                let source = context.source();
                let vcpu = context.thread().vcpu();
                self.start_owned_svc(context, std::sync::Arc::default(), None, async move {
                    let bytes = message.await?;
                    Ok(memory_wait::SvcCompletion::kernel(
                        source,
                        vcpu,
                        move |dispatcher, context| continuation(dispatcher, context, bytes),
                    ))
                })
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct KernelSessionRequest {
    session: std::sync::Arc<SessionObject>,
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
        let port = self.named_ports.lock().unwrap().get(&name).cloned();
        if let Some(port) = port {
            return connect_port(context, &port);
        }
        match crate::ipc_wire::connect_to_named_port(context.process(), &name) {
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
        // Retain/create objects outside namespace and handle-table locks.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_port.cpp#L120-L175
        if max_sessions == 0 {
            let mut ports = self.named_ports.lock().unwrap();
            if ports.get(&name).is_some_and(PortObject::server_is_open) {
                drop(ports);
                result(context, HorizonKernelResult::INVALID_STATE);
            } else {
                let removed = ports.remove(&name);
                drop(ports);
                result(
                    context,
                    if removed.is_some() {
                        HorizonKernelResult::SUCCESS
                    } else {
                        HorizonKernelResult::NOT_FOUND
                    },
                );
                if removed.is_some() {
                    write_register(context.thread_mut().state_mut(), 1, 0);
                }
                drop(removed);
            }
            return resume();
        }
        let (server, client) =
            PortObject::create_pair(context.process().scheduler(), max_sessions as usize, false);
        let handle = match context.process().handles().insert(server) {
            Ok(handle) => handle,
            Err(_) => {
                result(context, HorizonKernelResult::OUT_OF_HANDLES);
                return resume();
            }
        };
        let inserted = {
            let mut ports = self.named_ports.lock().unwrap();
            if let std::collections::btree_map::Entry::Vacant(entry) = ports.entry(name) {
                entry.insert(client);
                true
            } else {
                false
            }
        };
        if inserted {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
        } else {
            let _ = context.process().handles().close(handle);
            result(context, HorizonKernelResult::INVALID_STATE);
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
        let Some(target) = context.process().handles().lookup(handle) else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if !target.is::<crate::HorizonIpcObject>() {
            let Some(session) = target.into_arc::<SessionObject>() else {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            };
            return self.start_session_request(context, address, size, session, immediate);
        }
        let key = crate::ipc_wire::operation::ServiceKey::Session(target.clone());
        let turn = match self.service_scheduling.enter(key) {
            Ok(turn) => turn,
            Err(error) => return ipc_fault(immediate, error),
        };
        let mut request =
            match crate::ipc_wire::request::Request::new(context, target, address, size) {
                Ok(request) => request,
                Err(error) => return ipc_fault(immediate, error),
            };
        let mode = self.initial_operation_mode;
        let time = self.time_environment.clone();
        let systems = crate::ipc_wire::HostSystems {
            video: self.video_system.clone(),
            keyboard: self.keyboard_host.clone(),
            audio_backend: self.audio_backend.clone(),
            audio_renderers: self.audio_renderers.clone(),
            opus: self.opus.clone(),
            hid: self.hid_system.clone(),
            settings: self.settings_environment.clone(),
            application_language: self.application_language,
            application_display_version: self.application_display_version,
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
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Client || !session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let request = KernelSessionRequest {
            session,
            owner: session_request_owner(context),
            reply: None,
            size: 0,
            immediate: 0x20,
        };
        let mut words = [0_u32; 7];
        for (index, word) in words.iter_mut().enumerate() {
            *word = read_register(context.thread().state(), index as u8 + 1) as u32;
        }
        match request
            .session
            .request(request.owner, SessionMessage::Light(words))
        {
            Ok(()) => self.poll_session_request(context, request),
            Err(error) => session_error(context, error),
        }
    }

    fn start_session_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        address: GuestVirtualAddress,
        size: usize,
        session: std::sync::Arc<SessionObject>,
        immediate: u32,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
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
            session,
            owner: session_request_owner(context),
            reply: Some(ranges.clone()),
            size,
            immediate,
        };
        self.read_svc_message(
            context,
            immediate,
            ranges,
            move |dispatcher, context, bytes| {
                let bytes = match bytes {
                    Ok(bytes) => bytes,
                    Err(code) => {
                        result(context, code);
                        return resume();
                    }
                };
                let message = match capture_message_handles(context, bytes, false) {
                    Ok(message) => message,
                    Err(code) => {
                        result(context, code);
                        return resume();
                    }
                };
                match request.session.request(request.owner, message) {
                    Ok(()) => dispatcher.poll_session_request(context, request),
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
        let source = context.source();
        let vcpu = context.thread().vcpu();
        self.start_owned_svc(context, std::sync::Arc::default(), None, async move {
            let response = request.session.response(request.owner).await;
            Ok(memory_wait::SvcCompletion::kernel(
                source,
                vcpu,
                move |dispatcher, context| match response {
                    Ok(response) => dispatcher.finish_session_response(context, request, response),
                    Err(error) => session_error(context, error),
                },
            ))
        })
    }

    fn finish_session_response(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        mut request: KernelSessionRequest,
        response: SessionMessage,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        if request.session.is_light() {
            let SessionMessage::Light(words) = response else {
                return invalid_state(context);
            };
            for (index, word) in words.into_iter().enumerate() {
                write_register(
                    context.thread_mut().state_mut(),
                    index as u8 + 1,
                    u64::from(word),
                );
            }
            result(context, HorizonKernelResult::SUCCESS);
            return resume();
        }
        let response = match materialize_message_handles(
            context.process().handles(),
            response,
            request.size,
        ) {
            Ok(response) => response,
            Err(code) => {
                result(context, code);
                return resume();
            }
        };
        self.write_received_message(
            context,
            request.immediate,
            request.reply.take().expect("accepted reply capture"),
            response,
            None,
        )
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
    mut bytes: Vec<u8>,
    allow_move_handles: bool,
) -> Result<SessionMessage, HorizonKernelResult> {
    // MessageRead already validated and captured the emitted extent. Plain
    // messages need no further header decoding before queueing or delivery.
    if bytes[7] & 0x80 == 0 {
        return Ok(SessionMessage::Buffer(bytes));
    }
    let header = HipcHeader::decode(&bytes).expect("captured IPC special header");
    let offset = header.handle_offset();
    let copy_end = offset + header.copy_handle_count() * 4;
    let move_end = copy_end + header.move_handle_count() * 4;
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().expect("one handle word"));
    // ProcessMessageSpecialData walks the handle words directly. Receive-list
    // metadata is consumed by the destination, not by the emitting transport.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L162-L235
    let handles = crate::ipc_wire::handles::capture(
        context.process(),
        context.thread().object(),
        bytes[offset..copy_end].chunks_exact(4).map(word),
        bytes[copy_end..move_end].chunks_exact(4).map(word),
        allow_move_handles,
    )?;
    // The kernel supplies the sending process identity, never the PID in the
    // source buffer. The fixed header/special header precede this eight-byte slot.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L162-L172
    if header.sends_pid() {
        bytes[12..20].copy_from_slice(&context.process().process_id().to_le_bytes());
    }
    Ok(SessionMessage::TransportedBuffer {
        bytes,
        copy_handles: handles.copied,
        move_handles: handles.moved,
    })
}

pub(super) struct ReceivedMessage {
    pub(super) bytes: Vec<u8>,
    reservations: Vec<nixe_runtime::HandleReservation>,
    pub(super) receive: Option<nixe_runtime::SessionReceive>,
}
impl ReceivedMessage {
    fn complete_receive(&mut self) {
        if let Some(receive) = self.receive.take() {
            receive.complete();
        }
    }
}
pub(super) fn materialize_message_handles(
    handles: &std::sync::Arc<HandleTable>,
    message: SessionMessage,
    capacity: usize,
) -> Result<ReceivedMessage, HorizonKernelResult> {
    let (mut bytes, copied, moved) = match message {
        SessionMessage::Buffer(bytes) => {
            if bytes.len() > capacity {
                return Err(HorizonKernelResult::MESSAGE_TOO_LARGE);
            }
            return Ok(ReceivedMessage {
                bytes,
                reservations: Vec::new(),
                receive: None,
            });
        }
        SessionMessage::Light(_) => unreachable!("normal session carries buffer messages"),
        SessionMessage::TransportedBuffer {
            bytes,
            copy_handles,
            move_handles,
        } => (bytes, copy_handles, move_handles),
    };
    // Source acceptance already decoded the descriptors and retained their
    // handles. Delivery only needs the fixed header's layout, as MessageBuffer.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L584-L615
    if bytes.len() > capacity {
        return Err(HorizonKernelResult::MESSAGE_TOO_LARGE);
    }
    let header = HipcHeader::decode(&bytes).expect("captured IPC special header");
    debug_assert!(header.has_special_header());
    debug_assert_eq!(header.copy_handle_count(), copied.len());
    debug_assert_eq!(header.move_handle_count(), moved.len());
    let offset = header.handle_offset();
    let mut reservations = Vec::with_capacity(copied.len() + moved.len());
    for (index, object) in copied.into_iter().chain(moved).enumerate() {
        let handle = if let Some(object) = object {
            let reservation = handles
                .reserve(Some(object))
                .map_err(|_| HorizonKernelResult::OUT_OF_HANDLES)?;
            let handle = reservation.handle();
            reservations.push(reservation);
            handle
        } else {
            0
        };
        bytes[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&handle.to_le_bytes());
    }
    Ok(ReceivedMessage {
        bytes,
        reservations,
        receive: None,
    })
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
                message.complete_receive();
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
                    message.complete_receive();
                    Ok(super::memory_wait::SvcCompletion::success(output1))
                })
            }
        }
    }
}

pub(super) fn session_error(
    context: &mut ExceptionDispatchContext<'_>,
    error: SessionError,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let code = match error {
        SessionError::TransferFailed(code) => HorizonKernelResult(code),
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
