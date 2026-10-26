use super::*;

#[test]
fn audren_opens_updates_starts_and_stops() {
    for (revision, work_size) in [(7_u32, 0x66000), (8, 0x66000), (9, 0x67000), (15, 0x67000)] {
        exercise_renderer(revision, work_size);
    }
}

fn exercise_renderer(revision: u32, work_size: u32) {
    let revision_tag = (u32::from_le_bytes(*b"REV0") + (revision << 24)).to_le_bytes();
    let mut instructions = vec![svc(0x1f)];
    instructions.extend(std::iter::repeat_n(svc(0x21), 5));
    instructions.push(svc(0x15));
    instructions.extend(std::iter::repeat_n(svc(0x21), 24));
    let (_directory, mut process) = fixture_process(&instructions);
    #[derive(Debug)]
    struct Backend;
    struct Device;
    impl nixe_audio::AudioDevice for Device {}
    impl nixe_audio::AudioBackend for Backend {
        fn output_layout(&self) -> Result<nixe_audio::ChannelLayout, nixe_audio::AudioError> {
            Ok(nixe_audio::ChannelLayout::Stereo)
        }

        fn open(
            &self,
            _feed: nixe_audio::AudioFeed,
        ) -> Result<Box<dyn nixe_audio::AudioDevice>, nixe_audio::AudioError> {
            Ok(Box::new(Device))
        }
    }
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process)
        .with_memory_completions(events)
        .with_audio_backend(std::sync::Arc::new(Backend));
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let sm = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;
    let mut register = [0; 256];
    put_u32(&mut register, 0, 4);
    put_u32(&mut register, 4, 10 | (1 << 31));
    put_u32(&mut register, 8, 1);
    put_u32(&mut register, 32, 0x4943_4653);
    write_guest_bytes(&process, tls, &register);
    state(&mut process).write_w(x(0), sm);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let mut request = [0; 256];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 10);
    put_u32(&mut request, 16, 0x4943_4653);
    put_u32(&mut request, 24, 1);
    request[32..40].copy_from_slice(b"audren:u");
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), sm);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let manager = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .lookup_as::<HorizonIpcObject>(manager)
            .as_deref(),
        Some(HorizonIpcObject::AudioRendererManager(_))
    ));

    // Includes uncleared CMIF alignment padding, as in the actual SDK request.
    put_u32(&mut request, 4, 21);
    request[32..].fill(0xa5);
    request[32..84].fill(0);
    for (offset, value) in [
        (0, 48000_u32),
        (4, 240),
        (8, 30),
        (12, 1),
        (16, 96),
        (20, 1),
        (24, 14),
    ] {
        put_u32(&mut request, 32 + offset, value);
    }
    request[80..84].copy_from_slice(&revision_tag);
    for (words, revision, expected) in [
        (18, *b"REV7", HorizonIpcResult::CMIF_INVALID_IN_HEADER),
        (21, *b"bad!", HorizonIpcResult::AUDIO_INVALID_REVISION),
        (21, revision_tag, HorizonIpcResult::SUCCESS),
    ] {
        put_u32(&mut request, 4, words);
        request[80..84].copy_from_slice(&revision);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), manager);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            expected.raw()
        );
        if expected == HorizonIpcResult::SUCCESS {
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(32).unwrap()),
                work_size
            );
            assert_eq!(read_guest_u32(&process, tls.checked_add(36).unwrap()), 0);
        }
    }
    // Create actual kernel transfer memory and pass the owner pseudo-handle.
    let work_address = process.memory_layout().heap().base();
    process
        .memory()
        .map_zeroed(
            work_address,
            u64::from(work_size),
            nixe_runtime::memory::Mapping {
                kind: nixe_runtime::memory::MemoryKind::Heap,
                permissions: MemoryPermissions::READ_WRITE,
                uncached: false,
                locked: false,
                permission_locked: false,
            },
        )
        .unwrap();
    write_guest_bytes(&process, work_address, &vec![0xcc; work_size as usize]);
    state(&mut process).write_x(x(1), work_address.get());
    state(&mut process).write_x(x(2), u64::from(work_size));
    state(&mut process).write_w(x(3), 0);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(
        state(&mut process).read_w(x(0)),
        HorizonKernelResult::SUCCESS.raw()
    );
    let transfer = state(&mut process).read_w(x(1));
    let parameters: Vec<_> = request[32..84].to_vec();
    request.fill(0xa5);
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 25 | (1 << 31));
    put_u32(&mut request, 8, 5); // PID + two copied handles.
    put_u64(&mut request, 12, 0);
    put_u32(&mut request, 20, transfer);
    put_u32(&mut request, 24, CURRENT_PROCESS_HANDLE);
    put_u32(&mut request, 32, 0x4943_4653);
    put_u32(&mut request, 36, 0);
    put_u32(&mut request, 40, 0);
    put_u32(&mut request, 44, 0);
    request[48..100].copy_from_slice(&parameters);
    put_u64(&mut request, 112, 1); // aruid.
    for (handle, size, expected) in [
        (0xdead, work_size, HorizonIpcResult::AUDIO_INVALID_HANDLE),
        (
            transfer,
            work_size - 4096,
            HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER,
        ),
        (transfer, work_size, HorizonIpcResult::SUCCESS),
    ] {
        put_u32(&mut request, 20, handle);
        put_u64(&mut request, 104, u64::from(size));
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), manager);
        let outcome = dispatch_next(&mut process, &mut dispatcher);
        if handle == 0xdead {
            assert!(matches!(
                outcome,
                ExceptionHandlingResult::Rejected(HorizonSvcFault::Ipc { .. })
            ));
            assert_eq!(
                state(&mut process).read_w(x(0)),
                HorizonKernelResult::INVALID_HANDLE.raw()
            );
            continue;
        }
        assert!(matches!(outcome, ExceptionHandlingResult::Resumed));
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            expected.raw()
        );
    }
    let renderer = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process
            .handles()
            .lookup_as::<HorizonIpcObject>(renderer)
            .as_deref(),
        Some(HorizonIpcObject::AudioRenderer(_))
    ));
    let mut work_bytes = vec![0xff; work_size as usize];
    let retained = process
        .handles()
        .lookup_as::<nixe_runtime::TransferMemoryObject>(transfer)
        .unwrap();
    retained.ranges()[0]
        .allocation()
        .machine()
        .read_captured_into(retained.ranges(), 0, &mut work_bytes)
        .unwrap()
        .into_ready()
        .unwrap();
    drop(retained);
    assert!(work_bytes.iter().all(|&byte| byte == 0));
    // Renderer ownership outlives the client's transfer-memory handle.
    process.handles().close(transfer).unwrap();
    request.fill(0);
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 10);
    put_u32(&mut request, 16, 0x4943_4653);
    for (command, expected) in [(0, 48000), (1, 240), (2, 30), (3, 1), (9, 100)] {
        put_u32(&mut request, 24, command);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), renderer);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(32).unwrap()),
            expected
        );
    }
    for command in [8, 9] {
        put_u32(&mut request, 24, command);
        put_u32(&mut request, 32, 80);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), renderer);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        if command == 9 {
            assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 80);
        }
    }
    put_u32(&mut request, 24, 7);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), renderer);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let event_handle = read_guest_u32(&process, tls.checked_add(12).unwrap());
    let event = process
        .handles()
        .lookup_as::<ReadableEventObject>(event_handle)
        .unwrap();
    assert!(!event.is_signalled());
    // Stopped RequestUpdate has one input and two outputs, including the
    // optional performance output. No voice or effect is registered yet.
    let pool_count = 96 * 4 + 14;
    let voice_size = if revision >= 15 { 0x188 } else { 0x170 };
    let effect_status_size = if revision >= 9 { 0x90 } else { 0x10 };
    let input_size = 0x40
        + 0x10
        + pool_count * 0x20
        + 96 * 0x70
        + 96 * voice_size
        + 14 * 0xc0
        + 0x20
        + 0x140
        + 0x10;
    let output_size =
        0x40 + pool_count * 0x10 + 96 * 0x10 + 14 * effect_status_size + 0x20 + 0x10 + 0xb0 + 0x10;
    let mut update = vec![0; input_size];
    for (offset, value) in [
        (0, u32::from_le_bytes(revision_tag)),
        (4, 0x10),
        (8, pool_count as u32 * 0x20),
        (12, 96 * voice_size as u32),
        (16, 96 * 0x70),
        (20, 14 * 0xc0),
        (24, 0x20),
        (28, 0x140),
        (32, 0x10),
        (60, input_size as u32),
        (0x40, u32::from_le_bytes(revision_tag)),
    ] {
        put_u32(&mut update, offset, value);
    }
    let input_address = name.checked_add(0x60000).unwrap();
    let output_address = name.checked_add(0x80000).unwrap();
    write_guest_bytes(&process, input_address, &update);
    write_guest_bytes(&process, output_address, &vec![0xff; output_size]);
    request.fill(0);
    put_u32(&mut request, 0, 4 | (1 << 20) | (2 << 24));
    put_u32(&mut request, 4, 10);
    put_receive_buffer(&mut request, 8, input_address.get(), input_size as u64);
    put_receive_buffer(&mut request, 32, 0, 0);
    put_u32(&mut request, 48, 0x4943_4653);
    put_u32(&mut request, 56, 4);
    for (size, expected) in [
        (output_size - 1, HorizonIpcResult::AUDIO_INVALID_UPDATE_INFO),
        (output_size, HorizonIpcResult::SUCCESS),
    ] {
        put_receive_buffer(&mut request, 20, output_address.get(), size as u64);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), renderer);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            expected.raw()
        );
        if expected != HorizonIpcResult::SUCCESS {
            assert!(
                read_guest_bytes(&process, output_address, output_size)
                    .iter()
                    .all(|&byte| byte == 0xff)
            );
        }
    }
    assert_eq!(
        read_guest_u32(&process, output_address),
        u32::from_le_bytes(*b"REV?")
    );
    assert_eq!(
        read_guest_u32(&process, output_address.checked_add(60).unwrap()),
        output_size as u32
    );
    assert!(
        !process
            .handles()
            .lookup_as::<ReadableEventObject>(event_handle)
            .unwrap()
            .is_signalled()
    );

    // Auto-select reserves inactive null pointers alongside the map aliases.
    request.fill(0);
    put_u32(&mut request, 0, 4 | (1 << 16) | (1 << 20) | (2 << 24));
    put_u32(&mut request, 4, 10 | (4 << 10));
    put_receive_buffer(&mut request, 16, input_address.get(), input_size as u64);
    put_receive_buffer(&mut request, 28, output_address.get(), output_size as u64);
    put_u32(&mut request, 64, 0x4943_4653);
    put_u32(&mut request, 72, 10);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), renderer);
    assert!(matches!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    for (command, expected_state) in [(5, None), (3, Some(0)), (6, None), (3, Some(1))] {
        request.fill(0);
        put_u32(&mut request, 0, 4);
        put_u32(&mut request, 4, 10);
        put_u32(&mut request, 16, 0x4943_4653);
        put_u32(&mut request, 24, command);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), renderer);
        let (thread, handling) = dispatch_scheduled_next(&mut process, &mut dispatcher);
        assert!(matches!(
            handling,
            ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Suspended
        ));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while process.thread_lifecycle(thread) == nixe_scheduler::ThreadLifecycle::Waiting {
            dispatcher
                .poll_memory_completions(process.coordinator_mut())
                .unwrap();
            assert!(
                std::time::Instant::now() < deadline,
                "audio command did not complete"
            );
            std::thread::yield_now();
        }
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        if let Some(expected_state) = expected_state {
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(32).unwrap()),
                expected_state
            );
        }
    }
}
