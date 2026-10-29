use super::*;
use nixe_memory::{AccessOperation, AccessResult, DeviceAccessEndpoint, DeviceAccessRequest};
use std::sync::{Arc, mpsc};

pub(super) struct Device(pub(super) mpsc::Sender<DeviceAccessRequest>);
impl DeviceAccessEndpoint for Device {
    fn submit(&self, request: DeviceAccessRequest) {
        self.0.send(request).unwrap();
    }
    fn cancel(&self, _: nixe_memory::AccessRequestId) {}
    fn allocation_released(&self, _: nixe_memory::MachineId, _: nixe_memory::AllocationId) {}
    fn physical_retired(
        &self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: Box<[nixe_memory::Extent]>,
    ) {
    }
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
}

#[test]
fn a_hle_descriptor_keeps_its_own_capture_when_it_aliases_a_remapped_command_buffer() {
    remapped_command_output(false);
}

#[test]
fn a_hle_map_alias_output_keeps_its_capture_separate_from_the_command() {
    remapped_command_output(true);
}

fn remapped_command_output(map_alias: bool) {
    let (_directory, mut process) = fixture_process(&[svc(0x22)]);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let session = process
        .handles()
        .insert(if map_alias {
            HorizonIpcObject::UserSettings(crate::UserSettingsSession::new(Default::default()))
        } else {
            HorizonIpcObject::SystemSettings(crate::SystemSettingsSession::new())
        })
        .unwrap();
    let buffer = process
        .main_thread()
        .stack_bottom
        .checked_add(0x1000)
        .unwrap();
    let output = buffer.checked_add(1024).unwrap();
    let mut command = vec![0; 0x1000];
    if map_alias {
        put_u32(&mut command, 0, 4 | (1 << 24));
        put_u32(&mut command, 4, 8);
        put_receive_buffer(&mut command, 8, output.get(), 0x100);
        put_u32(&mut command, 32, 0x4943_4653);
        put_u32(&mut command, 40, 5);
    } else {
        put_u32(&mut command, 0, 4);
        put_u32(&mut command, 4, 8 | (3 << 10));
        put_u32(&mut command, 16, 0x4943_4653);
        put_u32(&mut command, 24, 3);
        put_u64(&mut command, 40, output.get() | (0x100_u64 << 48));
    }
    write_guest_bytes(&process, buffer, &command);
    for (index, value) in [(0, buffer.get()), (1, 0x1000), (2, u64::from(session))] {
        write_abi_register(&process, index, value);
    }
    let mapping = process.memory().query(buffer).unwrap().mapping.unwrap();
    let machine = process.memory().account().machine().clone();
    let original = process
        .memory()
        .space()
        .capture(buffer, 0x1000, MemoryPermissions::READ_WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&original, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    let read = receive.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(read.operation(), AccessOperation::Read));
    process.memory().unmap(buffer, 0x1000, mapping).unwrap();
    process
        .memory()
        .map_zeroed(buffer, 0x1000, mapping)
        .unwrap();
    write_guest_bytes(&process, output, &[0x5a; 0x100]);
    let value = machine
        .apply_resolved_access(read.ranges(), read.operation())
        .unwrap();
    read.finish(Ok(value));
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    while read_abi_register(&process, 0) != 0 {
        let write = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(write.operation(), AccessOperation::Write(_)));
        let value = machine
            .apply_resolved_access(write.ranges(), write.operation())
            .unwrap();
        write.finish(Ok(value));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    assert_eq!(
        read_guest_u32(&process, output),
        if map_alias {
            crate::SystemLanguage::AVAILABLE[0].code() as u32
        } else {
            1
        }
    );
    let AccessResult::Read(original) = machine
        .apply_resolved_access(&original, &AccessOperation::Read)
        .unwrap()
    else {
        panic!("expected the accepted command allocation")
    };
    assert_eq!(&original[24..28], &0_u32.to_le_bytes());
    assert!(original[1024..1280].iter().all(|&byte| byte == 0));
}

#[test]
fn native_hle_user_buffers_read_a_bounded_prefix_and_only_separate_receive_metadata() {
    let (_directory, mut process) = fixture_process(&[svc(0x22)]);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let session = process
        .handles()
        .insert(HorizonIpcObject::SystemSettings(
            crate::SystemSettingsSession::new(),
        ))
        .unwrap();
    let output = process.main_thread().stack_bottom;
    let buffer = output.checked_add(0x1000).unwrap();
    let mut command = vec![0; 0x1000];
    put_u32(&mut command, 0, 4);
    put_u32(&mut command, 4, 8 | (3 << 10) | (256 << 20));
    put_u32(&mut command, 16, 0x4943_4653);
    put_u32(&mut command, 24, 3);
    put_u64(&mut command, 1024, output.get() | (0x100_u64 << 48));
    write_guest_bytes(&process, buffer, &command);
    write_guest_bytes(&process, output, &[0x5a; 0x100]);
    for (index, value) in [(0, buffer.get()), (1, 0x1000), (2, u64::from(session))] {
        write_abi_register(&process, index, value);
    }
    let machine = process.memory().account().machine().clone();
    let ranges = process
        .memory()
        .space()
        .capture(buffer, 0x1000, MemoryPermissions::READ_WRITE)
        .unwrap();
    let output_ranges = process
        .memory()
        .space()
        .capture(output, 0x100, MemoryPermissions::READ_WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    for (offset, size) in [(0, 256), (1024, 8)] {
        let read = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(read.operation(), AccessOperation::Read));
        assert_eq!(read.ranges()[0].offset(), ranges[0].offset() + offset);
        assert_eq!(
            read.ranges().iter().map(|range| range.bytes()).sum::<u64>(),
            size
        );
        let value = machine
            .apply_resolved_access(read.ranges(), read.operation())
            .unwrap();
        read.finish(Ok(value));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    while read_abi_register(&process, 0) != 0 {
        let write = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(write.operation(), AccessOperation::Write(_)));
        let value = machine
            .apply_resolved_access(write.ranges(), write.operation())
            .unwrap();
        write.finish(Ok(value));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    assert_eq!(read_abi_register(&process, 0), 0);
    let AccessResult::Read(output) = machine
        .apply_resolved_access(&output_ranges, &AccessOperation::Read)
        .unwrap()
    else {
        panic!("expected firmware output")
    };
    assert_eq!(&output[..4], &1_u32.to_le_bytes());
    let AccessResult::Read(reply) = machine
        .apply_resolved_access(&ranges, &AccessOperation::Read)
        .unwrap()
    else {
        panic!("expected IPC response")
    };
    assert_eq!(&reply[24..28], &0_u32.to_le_bytes());
}

#[test]
fn kernel_transport_does_not_read_the_emitters_separate_receive_list() {
    capture_emitted_message(3, 128);
}

#[test]
fn kernel_transport_does_not_read_the_emitters_single_receive_descriptor() {
    capture_emitted_message(2, 0);
}

fn capture_emitted_message(mode: u32, list_offset: usize) {
    let (_directory, mut process) = fixture_process(&[svc(0x22)]);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(client).unwrap();
    let buffer = process.main_thread().stack_bottom;
    let mut bytes = vec![0x7e; 0x1000];
    put_u32(&mut bytes, 0, 4);
    put_u32(
        &mut bytes,
        4,
        (1 << 31) | 6 | (mode << 10) | ((list_offset as u32 / 4) << 20),
    );
    put_u32(&mut bytes, 8, 1 << 1);
    put_u32(&mut bytes, 12, CURRENT_THREAD_HANDLE);
    let descriptor = buffer.checked_add(0x200).unwrap().get() | (0x100_u64 << 48);
    let list_start = if list_offset == 0 { 40 } else { list_offset };
    put_u64(&mut bytes, list_start, descriptor);
    write_guest_bytes(&process, buffer, &bytes);
    for (index, value) in [(0, buffer.get()), (1, 0x1000), (2, u64::from(handle))] {
        write_abi_register(&process, index, value);
    }
    let machine = process.memory().account().machine().clone();
    let ranges = process
        .memory()
        .space()
        .capture(buffer, 0x1000, MemoryPermissions::READ_WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    for (offset, size) in [(0, 8), (8, 4), (12, 28)] {
        let read = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(read.operation(), AccessOperation::Read));
        assert_eq!(
            read.ranges().iter().map(|range| range.bytes()).sum::<u64>(),
            size
        );
        assert_eq!(read.ranges()[0].offset(), ranges[0].offset() + offset);
        let value = machine
            .apply_resolved_access(read.ranges(), read.operation())
            .unwrap();
        read.finish(Ok(value));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    let mut request = server.receive().unwrap();
    let SessionMessage::TransportedBuffer {
        bytes: captured,
        copy_handles,
        move_handles,
    } = request.take_message()
    else {
        panic!("expected source message capture")
    };
    assert_eq!(captured.len(), 40);
    assert_eq!(copy_handles.len(), 1);
    assert!(move_handles.is_empty());
    assert!(Arc::ptr_eq(
        &copy_handles[0]
            .as_ref()
            .unwrap()
            .downcast::<nixe_runtime::ThreadObject>()
            .unwrap(),
        &process.main_thread(),
    ));
    request.complete();
    let id = process.scheduler_process_id();
    process.coordinator_mut().terminate_process(id).unwrap();
}

#[test]
fn a_pending_guest_receive_keeps_its_request_until_output_publication() {
    guest_receive(ReceiveEnd::Success);
}

#[test]
fn a_device_owned_user_message_captures_only_its_header_and_declared_data() {
    let (_directory, mut process) = fixture_process(&[svc(0x22)]);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(client).unwrap();
    let buffer = process.main_thread().stack_bottom;
    let mut bytes = vec![0x7e; 0x1000];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8);
    write_guest_bytes(&process, buffer, &bytes);
    for (index, value) in [(0, buffer.get()), (1, 0x1000), (2, u64::from(handle))] {
        write_abi_register(&process, index, value);
    }
    let machine = process.memory().account().machine().clone();
    let ranges = process
        .memory()
        .space()
        .capture(buffer, 0x1000, MemoryPermissions::READ_WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    for expected in [8, 32] {
        let read = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(read.operation(), AccessOperation::Read));
        assert_eq!(
            read.ranges().iter().map(|range| range.bytes()).sum::<u64>(),
            expected
        );
        let value = machine
            .apply_resolved_access(read.ranges(), read.operation())
            .unwrap();
        read.finish(Ok(value));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    let mut message = server.receive().unwrap();
    let SessionMessage::Buffer(captured) = message.take_message() else {
        panic!("expected captured input")
    };
    assert_eq!(captured.len(), 40);
    assert_eq!(captured, bytes[..40]);
    message.complete();
    let id = process.scheduler_process_id();
    process.coordinator_mut().terminate_process(id).unwrap();
}

#[test]
fn terminating_a_pending_guest_receive_completes_the_client_and_ignores_late_output() {
    guest_receive(ReceiveEnd::HostStop);
}

#[test]
fn failed_guest_receive_output_releases_its_request_and_preserves_the_host_fault() {
    guest_receive(ReceiveEnd::DeviceFailure);
}

enum ReceiveEnd {
    Success,
    HostStop,
    DeviceFailure,
}

#[test]
fn a_pending_reply_retains_its_endpoint_and_completed_result_after_the_last_handle_closes() {
    let (_directory, mut process) = fixture_process(&[svc(0x43)]);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    client
        .request(request_owner(991), SessionMessage::Buffer(vec![0; 8]))
        .unwrap();
    server.receive().unwrap().complete();
    let handle = process.handles().insert(server).unwrap();
    let moved = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    let object = process.handles().lookup(moved).unwrap();
    let tls = process.main_thread().tls_base;
    let mut bytes = [0; 16];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 1 << 31);
    put_u32(&mut bytes, 8, 1 << 5);
    put_u32(&mut bytes, 12, moved);
    write_guest_bytes(&process, tls, &bytes);
    for (index, value) in [
        (1, process.main_thread().stack_bottom.get()),
        (2, 0),
        (3, u64::from(handle)),
        (4, 0),
    ] {
        write_abi_register(&process, index, value);
    }
    let machine = process.memory().account().machine().clone();
    let ranges = process
        .memory()
        .space()
        .capture(tls, 0x100, MemoryPermissions::READ_WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    drop(process.handles().close(handle).unwrap());
    for expected in [8, 4, 4] {
        assert!(process.handles().lookup(moved).is_some());
        assert_eq!(
            crate::integration_test_support::poll_session(&client, request_owner(991)),
            Ok(None)
        );
        let read = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(read.operation(), AccessOperation::Read));
        assert_eq!(
            read.ranges().iter().map(|range| range.bytes()).sum::<u64>(),
            expected
        );
        let result = machine
            .apply_resolved_access(read.ranges(), read.operation())
            .unwrap();
        read.finish(Ok(result));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
    }
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::TIMED_OUT.raw())
    );
    assert!(process.handles().lookup(moved).is_none());
    let Some(SessionMessage::TransportedBuffer {
        bytes,
        move_handles,
        ..
    }) = crate::integration_test_support::poll_session(&client, request_owner(991)).unwrap()
    else {
        panic!("the accepted reply must survive endpoint closure");
    };
    assert_eq!(bytes.len(), 16);
    assert_eq!(move_handles, vec![Some(object)]);
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(992)),
        Err(nixe_runtime::SessionError::NoRequest)
    );
}

fn guest_receive(ending: ReceiveEnd) {
    let (_directory, mut process) = fixture_process(&[svc(0x43)]);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(server.clone()).unwrap();
    let count = process.handles().len();
    let mut bytes = vec![0; 48];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8 | (1 << 31));
    put_u32(&mut bytes, 8, 1 << 1);
    put_u32(&mut bytes, 12, 0x8001);
    client
        .request(
            request_owner(991),
            SessionMessage::TransportedBuffer {
                bytes,
                copy_handles: vec![Some(nixe_runtime::HandleObject::new(EventObject::new(
                    process.scheduler(),
                )))],
                move_handles: vec![],
            },
        )
        .unwrap();
    let list = process.main_thread().stack_bottom;
    write_guest_bytes(&process, list, &handle.to_le_bytes());
    for (index, value) in [(1, list.get()), (2, 1), (3, 0), (4, 0)] {
        write_abi_register(&process, index, value);
    }
    let tls = process.main_thread().tls_base;
    let ranges = process
        .memory()
        .space()
        .capture(tls, 0x100, MemoryPermissions::READ_WRITE)
        .unwrap();
    let machine = process.memory().account().machine().clone();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    let write = receive.recv_timeout(Duration::from_secs(2)).unwrap();
    let AccessOperation::Write(bytes) = write.operation() else {
        panic!("expected incoming-message write")
    };
    assert_eq!(bytes.len(), 0x30);
    let returned = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    assert!(process.handles().lookup(returned).is_none());
    assert_eq!(
        server.reply().unwrap_err(),
        nixe_runtime::SessionError::ReplyPending
    );
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(991)),
        Ok(None)
    );
    match ending {
        ReceiveEnd::Success => {
            let value = machine
                .apply_resolved_access(write.ranges(), write.operation())
                .unwrap();
            write.finish(Ok(value));
            dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap();
            assert_eq!(read_abi_register(&process, 0), 0);
            assert!(process.handles().lookup(returned).is_some());
            assert_eq!(process.handles().len(), count + 1);
            server
                .reply()
                .unwrap()
                .reply(SessionMessage::Buffer(vec![0; 8]))
                .unwrap();
            assert!(matches!(
                crate::integration_test_support::poll_session(&client, request_owner(991)),
                Ok(Some(_))
            ));
        }
        ReceiveEnd::HostStop => {
            let id = process.scheduler_process_id();
            process.coordinator_mut().terminate_process(id).unwrap();
            assert_eq!(
                crate::integration_test_support::poll_session(&client, request_owner(991)),
                Err(nixe_runtime::SessionError::PeerClosed)
            );
            client
                .request(request_owner(992), SessionMessage::Buffer(vec![0; 8]))
                .unwrap();
            server.receive().unwrap().complete();
            write.finish(Ok(AccessResult::Written));
            dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap();
            assert!(process.handles().lookup(returned).is_none());
            server
                .reply()
                .unwrap()
                .reply(SessionMessage::Buffer(vec![0; 8]))
                .unwrap();
            assert!(matches!(
                crate::integration_test_support::poll_session(&client, request_owner(992)),
                Ok(Some(_))
            ));
        }
        ReceiveEnd::DeviceFailure => {
            write.finish(Err(Arc::new(nixe_memory::MemoryError::Execution(
                "incoming write failed".into(),
            ))));
            assert!(matches!(
                dispatcher.poll_memory_completions(process.coordinator_mut()),
                Err(HorizonSvcFault::MemoryAccess { immediate: 0x43, fault })
                    if matches!(fault.as_ref(), nixe_memory::MemoryError::Execution(message) if message.as_ref() == "incoming write failed")
            ));
            assert_eq!(
                crate::integration_test_support::poll_session(&client, request_owner(991)),
                Err(nixe_runtime::SessionError::PeerClosed)
            );
            assert_eq!(process.handles().len(), count);
            assert!(process.handles().lookup(returned).is_none());
            client
                .request(request_owner(992), SessionMessage::Buffer(vec![0; 8]))
                .unwrap();
            server.receive().unwrap().complete();
            server
                .reply()
                .unwrap()
                .reply(SessionMessage::Buffer(vec![0; 8]))
                .unwrap();
            assert!(matches!(
                crate::integration_test_support::poll_session(&client, request_owner(992)),
                Ok(Some(_))
            ));
        }
    }
}

#[test]
fn process_exit_cancels_a_pending_ipc_reply_without_publishing_its_handle() {
    pending_reply(Stop::Guest);
}

#[test]
fn a_pending_ipc_reply_publishes_its_handle_only_after_output_completes() {
    pending_reply(Stop::None);
}

enum Stop {
    None,
    Guest,
    Host,
    CpuFault,
}

fn pending_reply(ending: Stop) {
    let (_directory, mut process) = fixture_process(&[svc(0x21), svc(0x07)]);
    let entry = instruction_address(&process);
    let tls_page = process.main_thread().tls_base.get() / nixe_memory::PAGE_SIZE;
    // The exit caller must not consume the device-owned command-buffer page.
    let child = (0..8)
        .find_map(|_| {
            let (child, _) = super::mutex::add_thread(&mut process, entry + 4, 45);
            (process.thread(child).unwrap().tls_base.get() / nixe_memory::PAGE_SIZE != tls_page)
                .then_some(child)
        })
        .unwrap();
    if matches!(ending, Stop::CpuFault) {
        process.thread(child).unwrap().registers().set_pc(0);
    }
    process.thread(child).unwrap().start().unwrap();
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let manager = crate::ServiceManagerSession::new();
    manager.register_client();
    let session = process
        .handles()
        .insert(HorizonIpcObject::ServiceManager(manager))
        .unwrap();
    let count = process.handles().len();
    let buffer = process.main_thread().tls_base;
    let mut command = [0; 0x100];
    put_u32(&mut command, 0, 4);
    put_u32(&mut command, 4, 10);
    put_u32(&mut command, 16, 0x4943_4653);
    put_u32(&mut command, 24, 1);
    command[32..40].copy_from_slice(b"fsp-srv\0");
    write_guest_bytes(&process, buffer, &command);
    process.main_thread().registers().write_w(x(0), session);
    let ranges = process
        .memory()
        .space()
        .capture(buffer, command.len() as u64, MemoryPermissions::READ_WRITE)
        .unwrap();
    let machine = process.memory().account().machine().clone();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    let stop = process.run_slice().unwrap().stop;
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        process
            .route_supervisor_call(&stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::Suspended
    ));
    let read = receive.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(read.operation(), AccessOperation::Read));
    let result = machine
        .apply_resolved_access(read.ranges(), read.operation())
        .unwrap();
    read.finish(Ok(result));
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    let write = receive.recv_timeout(Duration::from_secs(2)).unwrap();
    let AccessOperation::Write(response) = write.operation() else {
        panic!("expected pending response write")
    };
    let returned = u32::from_le_bytes(response[12..16].try_into().unwrap());
    assert!(process.handles().lookup(returned).is_none());
    assert_eq!(process.handles().len(), count + 1);
    if !matches!(ending, Stop::None) {
        if matches!(ending, Stop::Host) {
            let id = process.scheduler_process_id();
            assert!(process.coordinator_mut().terminate_process(id).unwrap());
        } else if matches!(ending, Stop::CpuFault) {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while process.lifecycle() == nixe_runtime::ProcessLifecycle::Running {
                if let Some(execution) = process
                    .coordinator_mut()
                    .run_next(Duration::from_millis(20))
                    .unwrap()
                {
                    assert!(matches!(
                        execution.report.stop,
                        nixe_runtime::ExecutionStop::FetchFault { .. }
                            | nixe_runtime::ExecutionStop::Safepoint
                    ));
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "native fault did not complete"
                );
            }
            assert_eq!(process.lifecycle(), nixe_runtime::ProcessLifecycle::Faulted);
        } else {
            assert!(
                matches!(dispatch_scheduled_next(&mut process, &mut dispatcher), (id, ExceptionHandlingResult::Terminated { .. }) if id == child)
            );
        }
        // Cancellation must reach the device before any completion is polled.
        assert!(write.is_cancelled());
        assert!(process.handles().is_empty());
        assert!(process.handles().lookup(returned).is_none());
        let AccessResult::Read(bytes) = machine
            .apply_resolved_access(write.ranges(), &AccessOperation::Read)
            .unwrap()
        else {
            panic!("expected reply buffer read")
        };
        assert_eq!(&bytes[..], &command[..bytes.len()]);
    } else {
        assert!(!write.is_cancelled());
        let result = machine
            .apply_resolved_access(write.ranges(), write.operation())
            .unwrap();
        write.finish(Ok(result));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        assert!(
            process
                .handles()
                .lookup_as::<HorizonIpcObject>(returned)
                .is_some()
        );
        assert_eq!(process.handles().len(), count + 1);
    }
}

#[test]
fn host_stop_cancels_a_pending_ipc_reply_before_device_completion() {
    pending_reply(Stop::Host);
}

#[test]
fn terminal_cpu_failure_cancels_a_pending_reply_before_device_completion() {
    pending_reply(Stop::CpuFault);
}

#[test]
fn thread_creation_waits_for_device_owned_tls_initialization() {
    pending_thread_creation(false, false);
}

#[test]
fn process_exit_cancels_unpublished_thread_creation() {
    pending_thread_creation(true, false);
}

#[test]
fn failed_tls_initialization_reports_the_device_failure_without_publishing_a_thread() {
    pending_thread_creation(false, true);
}

fn pending_thread_creation(exit: bool, fail: bool) {
    let (_directory, mut process) = fixture_process(&[svc(0x08)]);
    let main = process.main_thread();
    let entry = instruction_address(&process);
    {
        let mut state = main.registers();
        for (index, value) in [
            (1, entry),
            (2, 0),
            (3, main.stack_top.get()),
            (4, 44),
            (5, 0),
        ] {
            state.write_x(x(index), value);
        }
    }
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let stop = process.run_slice().unwrap().stop;
    let page = GuestVirtualAddress::new(main.tls_base.get() & !(nixe_memory::PAGE_SIZE - 1));
    let ranges = process
        .memory()
        .space()
        .capture(page, nixe_memory::PAGE_SIZE, MemoryPermissions::READ_WRITE)
        .unwrap();
    let machine = process.memory().account().machine().clone();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(Device(send));
    machine.set_device_endpoint(&device);
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    let handles_before = process.handles().len();
    let threads_before = process.threads().len();
    let handling = process
        .route_supervisor_call(&stop, &mut dispatcher)
        .unwrap();
    assert!(
        matches!(handling, ExceptionHandlingResult::Suspended),
        "TLS initialization returned {handling:?} code={:#x}",
        read_abi_register(&process, 0)
    );
    assert_eq!(process.handles().len(), handles_before + 1);
    assert_eq!(process.threads().len(), threads_before);
    if exit || fail {
        let request = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        if exit {
            let id = process.scheduler_process_id();
            process.coordinator_mut().terminate_process(id).unwrap();
            assert!(request.is_cancelled());
            assert!(process.handles().is_empty());
            dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap();
        } else {
            request.finish(Err(Arc::new(nixe_memory::MemoryError::Execution(
                "TLS device initialization failed".into(),
            ))));
            let error = dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("TLS device initialization failed")
            );
            assert_eq!(process.handles().len(), handles_before);
            assert_eq!(process.threads().len(), threads_before);
        }
        return;
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while process.main_thread_lifecycle() == nixe_scheduler::ThreadLifecycle::Waiting {
        if let Ok(request) = receive.try_recv() {
            assert!(!request.is_cancelled());
            let result = machine
                .apply_resolved_access(request.ranges(), request.operation())
                .unwrap();
            request.finish(Ok(result));
        }
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        assert!(
            std::time::Instant::now() < deadline,
            "TLS initialization did not finish"
        );
    }
    assert_eq!(read_abi_register(&process, 0), 0);
    let handle = read_abi_register(&process, 1) as u32;
    let child = process
        .handles()
        .lookup_as::<nixe_runtime::ThreadObject>(handle)
        .unwrap();
    let ranges = process
        .memory()
        .space()
        .capture(
            child.tls_base.checked_add(0x110).unwrap(),
            4,
            MemoryPermissions::READ,
        )
        .unwrap();
    let AccessResult::Read(bytes) = machine
        .apply_resolved_access(&ranges, &AccessOperation::Read)
        .unwrap()
    else {
        panic!("expected TLS device bytes")
    };
    assert_eq!(u32::from_le_bytes(bytes[..].try_into().unwrap()), handle);
}
