use super::*;

#[test]
fn applet_proxy_uses_the_copied_process_object_instead_of_the_handle_encoding() {
    for variant in 0..4 {
        let (_directory, mut process) = fixture_process(&[svc(0x21); 2]);
        let mut dispatcher = support::dispatcher(&process);
        let handle = process
            .handles()
            .insert(HorizonIpcObject::Applet(crate::AppletSession::new(
                process.scheduler(),
                Default::default(),
            )))
            .unwrap();
        let owner = match variant {
            0 => CURRENT_PROCESS_HANDLE,
            1 => process
                .handles()
                .insert_object(nixe_runtime::HandleObject::from_arc(std::sync::Arc::clone(
                    &process,
                )))
                .unwrap(),
            2 => process
                .handles()
                .insert(EventObject::new(process.scheduler()))
                .unwrap(),
            _ => 0,
        };
        let tls = process.main_thread().tls_base;
        let mut bytes = [0; 256];
        put_u32(&mut bytes, 0, 5);
        put_u32(&mut bytes, 4, 8);
        put_u32(&mut bytes, 16, 0x4943_4653);
        write_guest_bytes(&process, tls, &bytes);
        write_abi_register(&process, 0, u64::from(handle));
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        let root = read_guest_u32(&process, tls.checked_add(32).unwrap());
        bytes.fill(0);
        put_u32(&mut bytes, 0, 4);
        put_u32(&mut bytes, 4, 12 | (1 << 31));
        put_u32(&mut bytes, 8, 3);
        put_u32(&mut bytes, 20, owner);
        bytes[32] = 1;
        bytes[34..36].copy_from_slice(&24_u16.to_le_bytes());
        put_u32(&mut bytes, 36, root);
        put_u32(&mut bytes, 48, 0x4943_4653);
        write_guest_bytes(&process, tls, &bytes);
        write_abi_register(&process, 0, u64::from(handle));
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(40).unwrap()),
            if variant < 2 {
                0
            } else {
                HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
            },
        );
        if variant < 2 {
            assert_ne!(read_guest_u32(&process, tls.checked_add(48).unwrap()), root);
        }
    }
}

#[test]
fn driver_initialization_uses_the_copied_process_object_instead_of_the_handle_encoding() {
    for variant in 0..4 {
        let (_directory, mut process) = fixture_process(&[svc(0x15), svc(0x21)]);
        let mut dispatcher = support::dispatcher(&process);
        let handle = process
            .handles()
            .insert(HorizonIpcObject::NvDrv(crate::NvDrvSession::new(
                process.memory().account().machine().clone(),
                process.scheduler(),
            )))
            .unwrap();
        let owner = match variant {
            0 => CURRENT_PROCESS_HANDLE,
            1 => process
                .handles()
                .insert_object(nixe_runtime::HandleObject::from_arc(std::sync::Arc::clone(
                    &process,
                )))
                .unwrap(),
            2 => process
                .handles()
                .insert(EventObject::new(process.scheduler()))
                .unwrap(),
            _ => 0,
        };
        let address = process.memory_layout().heap().base();
        process
            .memory()
            .map_zeroed(
                address,
                4096,
                nixe_runtime::memory::Mapping {
                    kind: nixe_runtime::memory::MemoryKind::Heap,
                    permissions: MemoryPermissions::READ_WRITE,
                    uncached: false,
                    locked: false,
                    permission_locked: false,
                },
            )
            .unwrap();
        write_abi_register(&process, 1, address.get());
        write_abi_register(&process, 2, 4096);
        write_abi_register(&process, 3, 0);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0);
        let transfer = read_abi_register(&process, 1) as u32;
        let mut bytes = [0; 256];
        put_u32(&mut bytes, 0, 4);
        put_u32(&mut bytes, 4, 10 | (1 << 31));
        put_u32(&mut bytes, 8, 5);
        put_u32(&mut bytes, 20, owner);
        put_u32(&mut bytes, 24, transfer);
        put_u32(&mut bytes, 32, 0x4943_4653);
        put_u32(&mut bytes, 40, 3);
        put_u32(&mut bytes, 48, 4096);
        let tls = process.main_thread().tls_base;
        write_guest_bytes(&process, tls, &bytes);
        write_abi_register(&process, 0, u64::from(handle));
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            if variant < 2 {
                0
            } else {
                HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
            },
        );
        if variant < 2 {
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(32).unwrap()),
                crate::nvdrv::NV_SUCCESS
            );
        }
    }
}

#[test]
fn a_single_receive_buffer_is_captured_and_written_by_a_native_hle_call() {
    firmware_receive_list(2, 0, false, 8);
}

#[test]
fn a_native_hle_call_uses_the_explicit_receive_list_offset() {
    firmware_receive_list(3, 0x80, false, 8);
}

#[test]
fn native_hle_user_buffers_accept_small_messages_without_a_tls_capacity_limit() {
    firmware_receive_list(3, 0x80, true, 8);
}

#[test]
fn native_hle_user_buffers_accept_inline_data_and_receive_metadata_beyond_tls() {
    firmware_receive_list(2, 0, true, 300);
}

#[test]
fn native_hle_user_buffers_accept_receive_metadata_straddling_the_command_prefix() {
    firmware_receive_list(3, 0xfc, true, 8);
}

fn firmware_receive_list(mode: u32, list_offset: usize, user_buffer: bool, data_words: u32) {
    let (_directory, mut process) = fixture_process(&[svc(if user_buffer { 0x22 } else { 0x21 })]);
    let mut dispatcher = support::dispatcher(&process);
    let session = process
        .handles()
        .insert(HorizonIpcObject::SystemSettings(
            crate::SystemSettingsSession::new(),
        ))
        .unwrap();
    let output = process.main_thread().stack_bottom;
    write_guest_bytes(&process, output, &[0x5a; 0x100]);
    let mut command = vec![0; if user_buffer { 0x1000 } else { 0x100 }];
    put_u32(&mut command, 0, 4);
    put_u32(
        &mut command,
        4,
        data_words | (mode << 10) | ((list_offset as u32 / 4) << 20),
    );
    put_u32(&mut command, 16, 0x4943_4653);
    put_u32(&mut command, 24, 3);
    let offset = if list_offset == 0 {
        8 + data_words as usize * 4
    } else {
        list_offset
    };
    put_u64(&mut command, offset, output.get() | (0x100_u64 << 48));
    let buffer = if user_buffer {
        output.checked_add(0x1000).unwrap()
    } else {
        process.main_thread().tls_base
    };
    write_guest_bytes(&process, buffer, &command);
    if user_buffer {
        for (index, value) in [
            (0, buffer.get()),
            (1, command.len() as u64),
            (2, u64::from(session)),
        ] {
            write_abi_register(&process, index, value);
        }
    } else {
        write_abi_register(&process, 0, u64::from(session));
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_guest_u32(&process, output), 1);
    assert_eq!(
        read_guest_u32(&process, output.checked_add(8).unwrap()),
        0x584e
    );
    assert_eq!(read_guest_u32(&process, buffer.checked_add(24).unwrap()), 0);
}

#[test]
fn guest_ipc_injects_the_actual_sending_process_id_in_both_directions() {
    let read_pid = |process: &RunnableProcess, tls: GuestVirtualAddress| {
        process
            .memory()
            .space()
            .read_value(tls.checked_add(12).unwrap(), 8)
            .unwrap()
            .into_ready()
            .unwrap() as u64
    };
    let (directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = support::dispatcher(&process);
    let path = directory.path().join("server.nro");
    fs::write(&path, synthetic_nro(&[svc(0x43); 2])).unwrap();
    let plan = Launcher::build(LauncherInput::new(path)).unwrap();
    let server = support::process_builder_with_kernel(process.scheduler().clone())
        .build(&plan)
        .unwrap();
    server
        .main_thread()
        .registers()
        .set_pc(server.entry_module().entry_address() + 0x80);
    let affinity = process.coordinator_mut().scheduler().profile().all_cores();
    let server_id = process
        .coordinator_mut()
        .register_process(
            server,
            nixe_runtime::ProcessRegistration {
                priority: 45,
                ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
                affinity,
            },
        )
        .unwrap();
    let server = process
        .coordinator_mut()
        .process(server_id)
        .unwrap()
        .clone();
    server
        .scheduler()
        .set_activity(server.main_thread_id(), true)
        .unwrap();
    assert_ne!(process.process_id(), server.process_id());

    let (server_endpoint, client_endpoint) = SessionObject::create_pair(process.scheduler());
    let server_handle = server.handles().insert(server_endpoint).unwrap();
    let client_handle = process.handles().insert(client_endpoint).unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, (1 << 31) | 1);
    put_u32(&mut request, 8, 1 | (2 << 1));
    put_u64(&mut request, 12, 0x1234_5678_9abc_def0);
    put_u32(&mut request, 20, CURRENT_THREAD_HANDLE);
    put_u32(&mut request, 24, CURRENT_PROCESS_HANDLE);
    put_u32(&mut request, 28, 0x5a5a_0101);
    write_guest_bytes(&process, tls, &request);
    write_abi_register(&process, 0, u64::from(client_handle));
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    process
        .scheduler()
        .set_activity(process.main_thread_id(), true)
        .unwrap();

    let list = server.main_thread().stack_bottom;
    write_guest_bytes(&server, list, &server_handle.to_le_bytes());
    for (index, value) in [(1, list.get()), (2, 1), (3, 0), (4, 0)] {
        write_abi_register(&server, index, value);
    }
    server
        .scheduler()
        .set_activity(server.main_thread_id(), false)
        .unwrap();
    let (thread, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(thread, server.main_thread_id());
    assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
    assert_eq!(read_abi_register(&server, 0), 0);
    let server_tls = server.main_thread().tls_base;
    let request_pid = read_pid(&server, server_tls);
    let copied_thread = read_guest_u32(&server, server_tls.checked_add(20).unwrap());
    let copied_process = read_guest_u32(&server, server_tls.checked_add(24).unwrap());
    assert!(std::sync::Arc::ptr_eq(
        &server
            .handles()
            .lookup_as::<nixe_runtime::ThreadObject>(copied_thread)
            .unwrap(),
        &process.main_thread()
    ));
    assert!(std::sync::Arc::ptr_eq(
        &server
            .handles()
            .lookup_as::<RunnableProcess>(copied_process)
            .unwrap(),
        &process
    ));
    assert_eq!(
        read_guest_u32(&server, server_tls.checked_add(28).unwrap()),
        0x5a5a_0101
    );
    assert_eq!(read_pid(&process, tls), 0x1234_5678_9abc_def0);

    let mut reply = [0; 0x100];
    put_u32(&mut reply, 4, (1 << 31) | 1);
    put_u32(&mut reply, 8, 1 | (1 << 5));
    put_u64(&mut reply, 12, 0xfedc_ba98_7654_3210);
    put_u32(&mut reply, 20, copied_thread);
    put_u32(&mut reply, 24, 0xa5a5_0202);
    write_guest_bytes(&server, server_tls, &reply);
    for (index, value) in [(1, 0), (2, 0), (3, u64::from(server_handle)), (4, 0)] {
        write_abi_register(&server, index, value);
    }
    let (thread, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(thread, server.main_thread_id());
    assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
    assert_eq!(
        read_abi_register(&server, 0),
        u64::from(HorizonKernelResult::TIMED_OUT.raw())
    );
    assert!(server.handles().lookup(copied_thread).is_none());
    server
        .scheduler()
        .set_activity(server.main_thread_id(), true)
        .unwrap();
    process
        .scheduler()
        .set_activity(process.main_thread_id(), false)
        .unwrap();
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert_eq!(read_abi_register(&process, 0), 0);
    let reply_pid = read_pid(&process, tls);
    let returned_thread = read_guest_u32(&process, tls.checked_add(20).unwrap());
    assert!(std::sync::Arc::ptr_eq(
        &process
            .handles()
            .lookup_as::<nixe_runtime::ThreadObject>(returned_thread)
            .unwrap(),
        &process.main_thread()
    ));
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        0xa5a5_0202
    );
    assert_eq!(read_pid(&server, server_tls), 0xfedc_ba98_7654_3210);
    assert_eq!(
        (request_pid, reply_pid),
        (process.process_id(), server.process_id())
    );
    process.coordinator_mut().remove_process(server_id).unwrap();
}

#[test]
fn failed_guest_receive_returns_its_transfer_error_to_the_native_client() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(client).unwrap();
    let mut bytes = [0; 0x100];
    put_u32(&mut bytes, 0, 4);
    write_guest_bytes(&process, process.main_thread().tls_base, &bytes);
    write_abi_register(&process, 0, u64::from(handle));
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    let receive = server.receive().unwrap();
    receive.fail(nixe_runtime::SessionError::TransferFailed(
        HorizonKernelResult::OUT_OF_HANDLES.raw(),
    ));
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::OUT_OF_HANDLES.raw())
    );
}

#[test]
fn failed_incoming_handle_transfer_completes_the_client_and_frees_the_session() {
    let (_directory, mut process) = fixture_process(&[svc(0x43); 2]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(server.clone()).unwrap();
    let mut remaining = None;
    while let Ok(handle) = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
    {
        remaining = Some(handle);
    }
    drop(process.handles().close(remaining.unwrap()).unwrap());
    let count = process.handles().len();
    let mut bytes = vec![0; 52];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8 | (1 << 31));
    put_u32(&mut bytes, 8, 2 << 1);
    put_u32(&mut bytes, 12, 0x8001);
    put_u32(&mut bytes, 16, 0x8002);
    client
        .request(
            request_owner(991),
            SessionMessage::TransportedBuffer {
                bytes,
                copy_handles: vec![
                    Some(nixe_runtime::HandleObject::new(EventObject::new(
                        process.scheduler(),
                    ))),
                    Some(nixe_runtime::HandleObject::new(EventObject::new(
                        process.scheduler(),
                    ))),
                ],
                move_handles: vec![],
            },
        )
        .unwrap();
    let address = process.main_thread().stack_bottom;
    write_guest_bytes(&process, address, &handle.to_le_bytes());
    for (index, value) in [(1, address.get()), (2, 1), (3, 0), (4, 0)] {
        write_abi_register(&process, index, value);
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::TIMED_OUT.raw())
    );
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(991)),
        Err(nixe_runtime::SessionError::TransferFailed(
            HorizonKernelResult::OUT_OF_HANDLES.raw()
        ))
    );
    assert_eq!(process.handles().len(), count);
    let mut bytes = vec![0; 40];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8);
    client
        .request(request_owner(992), SessionMessage::Buffer(bytes))
        .unwrap();
    for (index, value) in [(1, address.get()), (2, 1), (3, 0), (4, 0)] {
        write_abi_register(&process, index, value);
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_guest_u32(&process, process.main_thread().tls_base), 4);
}

#[test]
fn native_user_buffer_transport_checks_the_captured_extent_against_receiver_capacity() {
    for words in [8, 300] {
        let (directory, mut process) = fixture_process(&[svc(0x22)]);
        let mut dispatcher = support::dispatcher(&process);
        let path = directory.path().join("server.nro");
        fs::write(&path, synthetic_nro(&[svc(0x43); 2])).unwrap();
        let plan = Launcher::build(LauncherInput::new(path)).unwrap();
        let server = support::process_builder_with_kernel(process.scheduler().clone())
            .build(&plan)
            .unwrap();
        server
            .main_thread()
            .registers()
            .set_pc(server.entry_module().entry_address() + 0x80);
        let affinity = process.coordinator_mut().scheduler().profile().all_cores();
        let server_id = process
            .coordinator_mut()
            .register_process(
                server,
                nixe_runtime::ProcessRegistration {
                    priority: 45,
                    ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
                    affinity,
                },
            )
            .unwrap();
        let server = process
            .coordinator_mut()
            .process(server_id)
            .unwrap()
            .clone();
        server
            .scheduler()
            .set_activity(server.main_thread_id(), true)
            .unwrap();
        let (server_endpoint, client_endpoint) = SessionObject::create_pair(process.scheduler());
        let server_handle = server.handles().insert(server_endpoint).unwrap();
        let client_handle = process.handles().insert(client_endpoint).unwrap();
        let buffer = process.main_thread().stack_bottom;
        let mut bytes = vec![0x7e; 0x1000];
        put_u32(&mut bytes, 0, 4);
        put_u32(&mut bytes, 4, words);
        write_guest_bytes(&process, buffer, &bytes);
        for (index, value) in [
            (0, buffer.get()),
            (1, 0x1000),
            (2, u64::from(client_handle)),
        ] {
            write_abi_register(&process, index, value);
        }
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Suspended
        ));
        process
            .scheduler()
            .set_activity(process.main_thread_id(), true)
            .unwrap();
        let list = server.main_thread().stack_bottom;
        let tls = server.main_thread().tls_base;
        write_guest_bytes(&server, list, &server_handle.to_le_bytes());
        write_guest_bytes(&server, tls, &[0x55; 0x100]);
        for (index, value) in [(1, list.get()), (2, 1), (3, 0), (4, 0)] {
            write_abi_register(&server, index, value);
        }
        server
            .scheduler()
            .set_activity(server.main_thread_id(), false)
            .unwrap();
        let (thread, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert_eq!(thread, server.main_thread_id());
        assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
        if words == 8 {
            assert_eq!(read_abi_register(&server, 0), 0);
            assert_eq!(read_guest_u32(&server, tls), 4);
            assert_eq!(
                read_guest_u32(&server, tls.checked_add(8).unwrap()),
                0x7e7e_7e7e
            );
            assert_eq!(
                read_guest_u32(&server, tls.checked_add(40).unwrap()),
                0x5555_5555
            );
            let mut reply = [0; 12];
            put_u32(&mut reply, 0, 4);
            put_u32(&mut reply, 4, 1);
            put_u32(&mut reply, 8, 0xa5a5_a5a5);
            write_guest_bytes(&server, tls, &reply);
            for (index, value) in [(1, 0), (2, 0), (3, u64::from(server_handle)), (4, 0)] {
                write_abi_register(&server, index, value);
            }
            let (thread, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
            assert_eq!(thread, server.main_thread_id());
            assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
        } else {
            assert_eq!(read_guest_u32(&server, tls), 0x5555_5555);
        }
        assert_eq!(
            read_abi_register(&server, 0),
            u64::from(HorizonKernelResult::TIMED_OUT.raw())
        );
        server
            .scheduler()
            .set_activity(server.main_thread_id(), true)
            .unwrap();
        process
            .scheduler()
            .set_activity(process.main_thread_id(), false)
            .unwrap();
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        let expected = if words == 8 {
            assert_eq!(
                read_guest_u32(&process, buffer.checked_add(8).unwrap()),
                0xa5a5_a5a5
            );
            assert_eq!(
                read_guest_u32(&process, buffer.checked_add(12).unwrap()),
                0x7e7e_7e7e
            );
            HorizonKernelResult::SUCCESS
        } else {
            HorizonKernelResult::MESSAGE_TOO_LARGE
        };
        assert_eq!(read_abi_register(&process, 0), u64::from(expected.raw()));
    }
}

#[test]
fn a_light_session_reply_belongs_to_the_receiving_server_thread() {
    let (_directory, mut process) = fixture_process(&[svc(0x42); 3]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(server).unwrap();
    client
        .request(request_owner(991), SessionMessage::Light([7; 7]))
        .unwrap();
    write_abi_register(&process, 0, u64::from(handle));
    write_abi_register(&process, 1, 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let main = process.main_thread_id();
    let scheduler = process.scheduler().clone();
    scheduler.set_activity(main, true).unwrap();
    let child = support::ready(process.create_thread(nixe_runtime::ThreadCreateRequest {
        entry: GuestVirtualAddress::new(process.main_thread().registers().pc()),
        argument: u64::from(handle),
        fpcr: 0,
        fpsr: 0,
        stack_top: process.main_thread().stack_top,
        priority: 20,
        ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
        affinity: scheduler.profile().all_cores(),
    }))
    .unwrap();
    let child_thread = process.thread(child.id).unwrap();
    child_thread.registers().write_w(x(1), (1 << 31) | 8);
    child_thread.start().unwrap();
    let (caller, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, child.id);
    assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
    assert_eq!(
        child_thread.registers().read_w(x(0)),
        HorizonKernelResult::INVALID_STATE.raw()
    );
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(991)),
        Ok(None)
    );
    scheduler.set_activity(child.id, true).unwrap();
    scheduler.set_activity(main, false).unwrap();
    client
        .request(request_owner(992), SessionMessage::Light([9; 7]))
        .unwrap();
    write_abi_register(&process, 0, u64::from(handle));
    write_abi_register(&process, 1, u64::from((1u32 << 31) | 10));
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let Some(SessionMessage::Light(words)) =
        crate::integration_test_support::poll_session(&client, request_owner(991)).unwrap()
    else {
        panic!("the receiving server must still own the reply");
    };
    assert_eq!(words[0], (1 << 31) | 10);
    assert_eq!(read_abi_register(&process, 1), 9);
}

#[test]
fn a_light_session_allows_only_one_waiting_server_thread() {
    let (_directory, mut process) = fixture_process(&[svc(0x42); 2]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(server).unwrap();
    write_abi_register(&process, 0, u64::from(handle));
    write_abi_register(&process, 1, 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    let scheduler = process.scheduler().clone();
    let child = support::ready(process.create_thread(nixe_runtime::ThreadCreateRequest {
        entry: GuestVirtualAddress::new(process.main_thread().registers().pc()),
        argument: u64::from(handle),
        fpcr: 0,
        fpsr: 0,
        stack_top: process.main_thread().stack_top,
        priority: 20,
        ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
        affinity: scheduler.profile().all_cores(),
    }))
    .unwrap();
    let child_thread = process.thread(child.id).unwrap();
    child_thread.start().unwrap();
    let (caller, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, child.id);
    assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
    assert_eq!(
        child_thread.registers().read_w(x(0)),
        HorizonKernelResult::INVALID_STATE.raw()
    );
    scheduler.set_activity(child.id, true).unwrap();
    client
        .request(request_owner(991), SessionMessage::Light([11; 7]))
        .unwrap();
    let (caller, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, process.main_thread_id());
    assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
    assert_eq!(read_abi_register(&process, 1), 11);
}

#[test]
fn light_receive_consumes_cancellation_only_when_it_needs_to_wait() {
    let (_directory, mut process) = fixture_process(&[svc(0x19), svc(0x42), svc(0x42)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, _client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(server).unwrap();
    write_abi_register(&process, 0, u64::from(CURRENT_THREAD_HANDLE));
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    write_abi_register(&process, 0, u64::from(handle));
    write_abi_register(&process, 1, 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::CANCELLED.raw())
    );
    write_abi_register(&process, 0, u64::from(handle));
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
}

#[test]
fn cancelling_a_waiting_light_server_releases_the_receive_for_another_thread() {
    let (_directory, mut process) = fixture_process(&[svc(0x19), svc(0x42)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(server).unwrap();
    let scheduler = process.scheduler().clone();
    let core = nixe_scheduler::VirtualCpuId::new(0);
    let affinity = scheduler.profile().core_set([core]).unwrap();
    scheduler
        .migrate(
            process.main_thread_id(),
            nixe_scheduler::IdealCore::Core(core),
            affinity,
        )
        .unwrap();
    let child = support::ready(process.create_thread(nixe_runtime::ThreadCreateRequest {
        entry: GuestVirtualAddress::new(process.main_thread().registers().pc() + 4),
        argument: u64::from(handle),
        fpcr: 0,
        fpsr: 0,
        stack_top: process.main_thread().stack_top,
        priority: 20,
        ideal_vcpu: Some(core),
        affinity,
    }))
    .unwrap();
    let child_thread = process.thread(child.id).unwrap();
    write_abi_register(&process, 0, u64::from(child.handle));
    child_thread.start().unwrap();
    let (caller, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, child.id);
    assert!(matches!(outcome, ExceptionHandlingResult::Suspended));
    let (caller, outcome) = dispatch_scheduled_next(&mut process, &mut dispatcher);
    assert_eq!(caller, process.main_thread_id());
    assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
    client
        .request(request_owner(991), SessionMessage::Light([12; 7]))
        .unwrap();
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert_eq!(
        child_thread.registers().read_w(x(0)),
        HorizonKernelResult::CANCELLED.raw()
    );
    scheduler.set_activity(child.id, true).unwrap();
    write_abi_register(&process, 0, u64::from(handle));
    write_abi_register(&process, 1, 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_abi_register(&process, 1), 12);
}

#[test]
fn reply_and_receive_waits_for_a_busy_session_instead_of_rejecting_it() {
    let (_directory, mut process) = fixture_process(&[svc(0x43); 2]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let server_handle = process.handles().insert(server.clone()).unwrap();
    client
        .request(request_owner(991), SessionMessage::Buffer(vec![1]))
        .unwrap();
    server.receive().unwrap().complete();
    let mut bytes = vec![0; 40];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8);
    client
        .request(request_owner(992), SessionMessage::Buffer(bytes))
        .unwrap();
    let handles = process.main_thread().stack_bottom;
    write_guest_bytes(&process, handles, &server_handle.to_le_bytes());
    for (index, value) in [(1, handles.get()), (2, 1), (3, 0), (4, 0)] {
        write_abi_register(&process, index, value);
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::TIMED_OUT.raw())
    );
    server
        .reply()
        .unwrap()
        .reply(SessionMessage::Buffer(vec![2]))
        .unwrap();
    for (index, value) in [(1, handles.get()), (2, 1), (3, 0), (4, 0)] {
        write_abi_register(&process, index, value);
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_guest_u32(&process, process.main_thread().tls_base), 4);
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(992)),
        Ok(None)
    );
    server
        .reply()
        .unwrap()
        .reply(SessionMessage::Buffer(vec![3]))
        .unwrap();
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(992)),
        Ok(Some(SessionMessage::Buffer(vec![3])))
    );
}

#[test]
fn terminating_a_light_server_releases_its_pending_receive() {
    let (_directory, mut process) = fixture_process(&[svc(0x42)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, _client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(server.clone()).unwrap();
    write_abi_register(&process, 0, u64::from(handle));
    write_abi_register(&process, 1, 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    let id = process.scheduler_process_id();
    process.coordinator_mut().terminate_process(id).unwrap();
    assert_eq!(
        server.receive_light(request_owner(991)),
        Err(nixe_runtime::SessionError::NoRequest)
    );
    server.cancel_light_receive(request_owner(991));
}

#[test]
fn closing_a_delivered_ipc_handle_disconnects_its_peer_before_reply() {
    let (_directory, mut process) = fixture_process(&[svc(0x43), svc(0x16)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let server_handle = process.handles().insert(server.clone()).unwrap();
    let (transferred, peer) = SessionObject::create_pair(process.scheduler());
    let mut bytes = vec![0; 48];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8 | (1 << 31));
    put_u32(&mut bytes, 8, 1 << 1);
    put_u32(&mut bytes, 12, 0x8001);
    let owner = request_owner(991);
    client
        .request(
            owner,
            SessionMessage::TransportedBuffer {
                bytes,
                copy_handles: vec![Some(nixe_runtime::HandleObject::new(transferred))],
                move_handles: vec![],
            },
        )
        .unwrap();
    let handles = process.main_thread().stack_bottom;
    write_guest_bytes(&process, handles, &server_handle.to_le_bytes());
    for (index, value) in [(1, handles.get()), (2, 1), (3, 0), (4, u64::MAX)] {
        write_abi_register(&process, index, value);
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    let received = read_guest_u32(
        &process,
        process.main_thread().tls_base.checked_add(12).unwrap(),
    );
    assert!(
        process
            .handles()
            .lookup_as::<SessionObject>(received)
            .is_some()
    );
    write_abi_register(&process, 0, u64::from(received));
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(
        crate::integration_test_support::poll_session(&peer, request_owner(992)),
        Err(nixe_runtime::SessionError::NoRequest)
    );
    assert_eq!(
        crate::integration_test_support::poll_session(&client, owner),
        Ok(None)
    );
    server
        .reply()
        .unwrap()
        .reply(SessionMessage::Buffer(vec![9]))
        .unwrap();
    assert_eq!(
        crate::integration_test_support::poll_session(&client, owner),
        Ok(Some(SessionMessage::Buffer(vec![9])))
    );
}

#[test]
fn process_exit_closes_sessions_waiting_for_port_acceptance() {
    let (_directory, mut process) = fixture_process(&[svc(0x07)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client_port) =
        nixe_runtime::PortObject::create_pair(process.scheduler(), 1, false);
    process.handles().insert(server).unwrap();
    let client = client_port.connect().unwrap();
    let owner = request_owner(992);
    client
        .request(owner, SessionMessage::Buffer(vec![4]))
        .unwrap();
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Terminated { .. }
    ));
    assert_eq!(
        crate::integration_test_support::poll_session(&client, owner),
        Err(nixe_runtime::SessionError::PeerClosed)
    );
}

#[test]
fn connect_to_named_port_does_not_enqueue_a_session_when_the_handle_table_is_full() {
    let (_directory, mut process) = fixture_process(&[svc(0x71), svc(0x1f)]);
    let mut dispatcher = support::dispatcher(&process);
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"audit:\0");
    write_abi_register(&process, 1, name.get());
    write_abi_register(&process, 2, 1);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    let server = process
        .handles()
        .lookup_as::<nixe_runtime::PortObject>(read_abi_register(&process, 1) as u32)
        .unwrap();
    while process
        .handles()
        .insert(nixe_runtime::EventObject::new(process.scheduler()))
        .is_ok()
    {}
    write_abi_register(&process, 1, name.get());

    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::OUT_OF_HANDLES.raw())
    );
    assert!(!server.is_signalled());
    assert!(matches!(
        server.accept(),
        Err(nixe_runtime::PortError::NoPendingSession)
    ));
}

#[test]
fn connect_to_port_does_not_enqueue_a_session_when_the_handle_table_is_full() {
    for light in [false, true] {
        let (_directory, mut process) = fixture_process(&[svc(0x72)]);
        let mut dispatcher = support::dispatcher(&process);
        let (server, client) = nixe_runtime::PortObject::create_pair(process.scheduler(), 1, light);
        let handle = process.handles().insert(client).unwrap();
        while process
            .handles()
            .insert(nixe_runtime::EventObject::new(process.scheduler()))
            .is_ok()
        {}
        write_abi_register(&process, 1, u64::from(handle));

        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(HorizonKernelResult::OUT_OF_HANDLES.raw())
        );
        assert!(!server.is_signalled());
        assert!(matches!(
            server.accept(),
            Err(nixe_runtime::PortError::NoPendingSession)
        ));
    }
}

#[test]
fn accept_session_preserves_the_pending_client_when_the_handle_table_is_full() {
    for light in [false, true] {
        let (_directory, mut process) = fixture_process(&[svc(0x41); 2]);
        let mut dispatcher = support::dispatcher(&process);
        let (server, client_port) =
            nixe_runtime::PortObject::create_pair(process.scheduler(), 1, light);
        let client = client_port.connect().unwrap();
        let message = || {
            if light {
                SessionMessage::Light([3; 7])
            } else {
                SessionMessage::Buffer(vec![3])
            }
        };
        let owner = request_owner(991);
        client.request(owner, message()).unwrap();
        let handle = process.handles().insert(server).unwrap();
        let mut available = None;
        while let Ok(handle) = process
            .handles()
            .insert(nixe_runtime::EventObject::new(process.scheduler()))
        {
            available = Some(handle);
        }
        write_abi_register(&process, 1, u64::from(handle));

        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_abi_register(&process, 0),
            u64::from(HorizonKernelResult::OUT_OF_HANDLES.raw())
        );
        assert_eq!(
            crate::integration_test_support::poll_session(&client, owner),
            Ok(None)
        );

        process.handles().close(available.unwrap()).unwrap();
        write_abi_register(&process, 1, u64::from(handle));
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_abi_register(&process, 0), 0);
        let accepted = process
            .handles()
            .lookup_as::<SessionObject>(read_abi_register(&process, 1) as u32)
            .unwrap();
        let received = if light {
            accepted.receive_light(request_owner(993))
        } else {
            accepted.receive().map(|mut receive| {
                let message = receive.take_message();
                receive.complete();
                message
            })
        };
        assert_eq!(received, Ok(message()));
    }
}

#[test]
fn process_exit_closes_owned_session_endpoints() {
    let (_directory, mut process) = fixture_process(&[svc(0x07)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(server).unwrap();
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Terminated { .. }
    ));
    let request = client.request(request_owner(999), SessionMessage::Buffer(vec![1]));
    assert!(matches!(
        request,
        Err(nixe_runtime::SessionError::PeerClosed)
    ));
    assert!(process.handles().lookup(handle).is_none());
}

#[test]
fn created_thread_inherits_fpu_status_and_has_its_tls_handle() {
    let (_directory, mut process) = fixture_process(&[svc(0x08)]);
    let mut dispatcher = support::dispatcher(&process);
    let entry = instruction_address(&process);
    let stack = process.main_thread().stack_top.get();
    let main = process.main_thread();
    assert_eq!(
        read_guest_u32(&process, main.tls_base.checked_add(0x110).unwrap()),
        process.main_thread_handle()
    );

    {
        let caller = process.main_thread();
        let mut state = caller.registers();
        state.set_fpcr(0x0040_0000);
        state.set_fpsr(0x0800_0001);
        for (index, value) in [(1, entry), (2, 0), (3, stack), (4, 44), (5, 0)] {
            state.write_x(x(index), value);
        }
    }
    dispatch_next(&mut process, &mut dispatcher);
    assert_eq!(read_abi_register(&process, 0), 0);
    let handle = read_abi_register(&process, 1) as u32;
    let child = process
        .handles()
        .lookup_as::<nixe_runtime::ThreadObject>(handle)
        .unwrap();
    let control = child.registers().fpcr();
    let status = child.registers().fpsr();
    let tls_handle = process
        .memory()
        .space()
        .read_value(child.tls_base.checked_add(0x110).unwrap(), 4)
        .unwrap()
        .into_ready()
        .unwrap();
    let parent_control = process.main_thread().registers().fpcr();
    let parent_status = process.main_thread().registers().fpsr();
    assert_eq!(
        (control, status, tls_handle),
        (parent_control, parent_status, u128::from(handle))
    );
}

#[test]
fn create_thread_validates_only_the_abi_arguments() {
    let (_directory, mut process) = fixture_process(&[svc(0x08); 5]);
    let mut dispatcher = support::dispatcher(&process);
    let entry = instruction_address(&process);
    let stack = process.main_thread().stack_top.get();
    let mut mismatches = 0;
    for (_name, pc, sp, priority, core, expected) in [
        (
            "unmapped_entry",
            0,
            stack,
            44,
            0,
            HorizonKernelResult::SUCCESS,
        ),
        (
            "unmapped_stack",
            entry,
            0,
            44,
            0,
            HorizonKernelResult::SUCCESS,
        ),
        (
            "invalid_priority",
            entry,
            stack,
            64,
            0,
            HorizonKernelResult::INVALID_PRIORITY,
        ),
        (
            "invalid_core",
            entry,
            stack,
            44,
            99,
            HorizonKernelResult::INVALID_CORE_ID,
        ),
    ] {
        {
            let caller = process.main_thread();
            let mut state = caller.registers();
            for (index, value) in [(1, pc), (2, 0), (3, sp), (4, priority), (5, core)] {
                state.write_x(x(index), value);
            }
        }
        dispatch_next(&mut process, &mut dispatcher);
        let actual = read_abi_register(&process, 0);
        mismatches += usize::from(actual != u64::from(expected.raw()));
    }
    while process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .is_ok()
    {}
    {
        let caller = process.main_thread();
        let mut state = caller.registers();
        for (index, value) in [(1, entry), (2, 0), (3, stack), (4, 44), (5, 0)] {
            state.write_x(x(index), value);
        }
    }
    dispatch_next(&mut process, &mut dispatcher);
    let actual = read_abi_register(&process, 0);
    mismatches += usize::from(actual != u64::from(HorizonKernelResult::OUT_OF_HANDLES.raw()));
    assert_eq!(mismatches, 0);
}

#[test]
fn hle_handles_follow_null_and_client_move_rules() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 2]);
    let mut dispatcher = support::dispatcher(&process);
    let manager = crate::ServiceManagerSession::new();
    manager.register_client();
    let session = process
        .handles()
        .insert(HorizonIpcObject::ServiceManager(manager))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let mut null = [0; 0x100];
    put_u32(&mut null, 0, 4);
    put_u32(&mut null, 4, 10 | (1 << 31));
    put_u32(&mut null, 8, 1 << 1);
    put_u32(&mut null, 16, 0x4943_4653);
    put_u32(&mut null, 24, 1);
    null[32..40].copy_from_slice(b"fsp-srv\0");
    write_guest_bytes(&process, tls, &null);
    process.main_thread().registers().write_w(x(0), session);
    dispatch_next(&mut process, &mut dispatcher);
    let null_result = read_abi_register(&process, 0);
    let moved = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    let mut bad = [0; 0x100];
    put_u32(&mut bad, 0, 4);
    put_u32(&mut bad, 4, 10 | (1 << 31));
    put_u32(&mut bad, 8, 1 << 5);
    put_u32(&mut bad, 12, moved);
    put_u32(&mut bad, 16, 0x4943_4653);
    put_u32(&mut bad, 24, 1);
    bad[32..40].copy_from_slice(b"fsp-srv\0");
    write_guest_bytes(&process, tls, &bad);
    process.main_thread().registers().write_w(x(0), session);
    dispatch_next(&mut process, &mut dispatcher);
    let moved_live = process.handles().lookup(moved).is_some();
    let move_result = read_abi_register(&process, 0);
    assert_eq!(
        (null_result, move_result, moved_live),
        (
            0,
            u64::from(HorizonKernelResult::INVALID_COMBINATION.raw()),
            true
        )
    );
}

#[test]
fn null_copy_is_accepted_by_the_guest_session_transport() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(client).unwrap();
    let mut bytes = [0; 0x100];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8 | (1 << 31));
    put_u32(&mut bytes, 8, 1 << 1);
    write_guest_bytes(&process, process.main_thread().tls_base, &bytes);
    process.main_thread().registers().write_w(x(0), handle);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    let mut receive = server.receive().unwrap();
    let SessionMessage::TransportedBuffer { copy_handles, .. } = receive.take_message() else {
        panic!("expected owned transport message")
    };
    assert_eq!(copy_handles, vec![None]);
    receive.complete();
}

#[test]
fn guest_transport_rejects_client_move_handles() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = support::dispatcher(&process);
    let (_server, client) = SessionObject::create_pair(process.scheduler());
    let session = process.handles().insert(client).unwrap();
    let moved = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    let mut bytes = [0; 0x100];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8 | (1 << 31));
    put_u32(&mut bytes, 8, (1 << 1) | (1 << 5));
    put_u32(&mut bytes, 12, 0x3fff_ffff);
    put_u32(&mut bytes, 16, moved);
    write_guest_bytes(&process, process.main_thread().tls_base, &bytes);
    process.main_thread().registers().write_w(x(0), session);
    dispatch_next(&mut process, &mut dispatcher);
    let present = process.handles().lookup(moved).is_some();
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::INVALID_COMBINATION.raw())
    );
    assert!(present);
}

#[test]
fn thread_context_checks_handle_and_state_before_memory() {
    let (_directory, mut process) = fixture_process(&[svc(0x33); 3]);
    let mut dispatcher = support::dispatcher(&process);
    let entry = instruction_address(&process);
    let (_, handle) = super::mutex::add_thread(&mut process, entry, 45);
    let mut mismatches = 0;
    for (_name, handle, expected) in [
        ("invalid_handle", 0, HorizonKernelResult::INVALID_HANDLE),
        ("not_suspended", handle, HorizonKernelResult::INVALID_STATE),
        (
            "current_thread",
            CURRENT_THREAD_HANDLE,
            HorizonKernelResult::BUSY,
        ),
    ] {
        process.main_thread().registers().write_x(x(0), 0);
        process.main_thread().registers().write_w(x(1), handle);
        dispatch_next(&mut process, &mut dispatcher);
        let actual = read_abi_register(&process, 0);
        mismatches += usize::from(actual != u64::from(expected.raw()));
    }
    assert_eq!(mismatches, 0);
}

#[test]
fn host_stop_cancels_the_accepted_guest_session_request_without_polling() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let handle = process.handles().insert(client.clone()).unwrap();
    let mut bytes = [0; 0x100];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 8);
    write_guest_bytes(&process, process.main_thread().tls_base, &bytes);
    process.main_thread().registers().write_w(x(0), handle);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    assert!(matches!(
        crate::integration_test_support::poll_session(&client, request_owner(1)),
        Ok(None)
    ));
    let id = process.scheduler_process_id();
    assert!(process.coordinator_mut().terminate_process(id).unwrap());
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(1)),
        Err(nixe_runtime::SessionError::NoRequest)
    );
    assert!(matches!(
        server.receive().unwrap_err(),
        nixe_runtime::SessionError::NoRequest
    ));
    assert!(process.handles().is_empty());
}

#[test]
fn light_client_request_retains_its_endpoint_after_the_source_handle_closes() {
    let (_directory, mut process) = fixture_process(&[svc(0x20)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(client).unwrap();
    process.main_thread().registers().write_w(x(0), handle);
    process.main_thread().registers().write_w(x(1), 17);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    assert!(matches!(
        server.receive_light(request_owner(2)),
        Ok(SessionMessage::Light(_))
    ));
    drop(process.handles().close(handle).unwrap());
    assert_eq!(server.reply_light(request_owner(2), [42; 7]), Ok(()));
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_abi_register(&process, 1), 42);
}

#[test]
fn light_server_wait_retains_its_endpoint_after_the_source_handle_closes() {
    let (_directory, mut process) = fixture_process(&[svc(0x42)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(server).unwrap();
    process.main_thread().registers().write_w(x(0), handle);
    process.main_thread().registers().write_w(x(1), 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    drop(process.handles().close(handle).unwrap());
    assert_eq!(
        client.request(request_owner(2), SessionMessage::Light([42; 7])),
        Ok(())
    );
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert_eq!(read_abi_register(&process, 0), 0);
    assert_eq!(read_abi_register(&process, 1), 42);
}

#[test]
fn host_stop_cancels_light_requests_even_while_another_handle_retains_the_endpoint() {
    let (_directory, mut process) = fixture_process(&[svc(0x20)]);
    let mut dispatcher = support::dispatcher(&process);
    let (_server, client) = SessionObject::create_light_pair(process.scheduler());
    let handle = process.handles().insert(client.clone()).unwrap();
    process.main_thread().registers().write_w(x(0), handle);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Suspended
    ));
    let process_id = process.scheduler_process_id();
    process
        .coordinator_mut()
        .terminate_process(process_id)
        .unwrap();
    assert_eq!(
        crate::integration_test_support::poll_session(&client, request_owner(1)),
        Err(nixe_runtime::SessionError::NoRequest)
    );
}

#[test]
fn reply_and_receive_finishes_the_reply_before_consuming_wait_cancellation() {
    let (_directory, mut process) = fixture_process(&[svc(0x19), svc(0x43)]);
    let mut dispatcher = support::dispatcher(&process);
    let (server, client) = SessionObject::create_pair(process.scheduler());
    let server_handle = process.handles().insert(server.clone()).unwrap();
    let owner = request_owner(91);
    client
        .request(owner, SessionMessage::Buffer(vec![1]))
        .unwrap();
    server.receive().unwrap().complete();
    process
        .main_thread()
        .registers()
        .write_w(x(0), process.main_thread_handle());
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_abi_register(&process, 0), 0);
    let list = process.main_thread().stack_bottom;
    write_guest_bytes(&process, list, &server_handle.to_le_bytes());
    let reply = vec![0; 8];
    write_guest_bytes(&process, process.main_thread().tls_base, &reply);
    {
        let main = process.main_thread();
        let mut state = main.registers();
        for (index, value) in [
            (1, list.get()),
            (2, 1),
            (3, u64::from(server_handle)),
            (4, u64::MAX),
        ] {
            state.write_x(x(index), value);
        }
    }
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(
        read_abi_register(&process, 0),
        u64::from(HorizonKernelResult::CANCELLED.raw())
    );
    assert_eq!(
        crate::integration_test_support::poll_session(&client, owner),
        Ok(Some(SessionMessage::Buffer(reply)))
    );
}
