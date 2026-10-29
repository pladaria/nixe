use super::*;
use nixe_audio::{AudioBackend, AudioDevice, AudioError, AudioFeed};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Backend {
    feed: Mutex<Option<AudioFeed>>,
    live: Arc<AtomicUsize>,
}
impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TestAudioBackend")
    }
}
struct Device(Arc<AtomicUsize>);
impl Drop for Device {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
impl AudioDevice for Device {}

impl AudioBackend for Backend {
    fn output_layout(&self) -> Result<nixe_audio::ChannelLayout, nixe_audio::AudioError> {
        Ok(nixe_audio::ChannelLayout::Stereo)
    }

    fn open(&self, feed: AudioFeed) -> Result<Box<dyn AudioDevice>, AudioError> {
        *self.feed.lock().unwrap() = Some(feed);
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Device(self.live.clone())))
    }
}

// Build the same legacy/map-alias and auto-select layouts as libnx. The inactive
// pointer descriptors and output-pointer size table are retained in auto-select
// requests. libnx does not clear TLS alignment padding when reusing the buffer:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L93-L146
pub(super) fn command(
    cmd: u32,
    data: &[u8],
    send: Option<(u64, u64)>,
    recv: Option<(u64, u64)>,
    auto: bool,
    owner: bool,
) -> [u8; 256] {
    let mut b = [0; 256];
    let mut word0 = 4;
    let mut cursor = 8;
    if owner {
        put_u32(&mut b, cursor, 1 | (1 << 1));
        put_u32(&mut b, cursor + 12, CURRENT_PROCESS_HANDLE);
        cursor += 16;
    }
    if auto && send.is_some() {
        word0 |= 1 << 16;
        cursor += 8;
    }
    if let Some((address, size)) = send {
        word0 |= 1 << 20;
        put_receive_buffer(&mut b, cursor, address, size);
        cursor += 12;
    }
    if let Some((address, size)) = recv {
        word0 |= 1 << 24;
        put_receive_buffer(&mut b, cursor, address, size);
        cursor += 12;
    }
    let raw = cursor;
    cursor = (cursor + 15) & !15;
    put_u32(&mut b, cursor, 0x4943_4653);
    put_u32(&mut b, cursor + 8, cmd);
    b[cursor + 16..cursor + 16 + data.len()].copy_from_slice(data);
    let pointer_table = (32 + data.len()).next_multiple_of(2);
    let table_size = if auto && recv.is_some() { 2 } else { 0 };
    let words = (pointer_table + table_size).div_ceil(4);
    b[raw..cursor].fill(0xa5);
    b[cursor + 16 + data.len()..raw + words * 4].fill(0xa5);
    // The inactive output pointer has size zero, independently of TLS padding.
    b[raw + pointer_table..raw + pointer_table + table_size].fill(0);
    assert!(raw + words * 4 + 8 <= 256);
    put_u32(&mut b, 0, word0);
    put_u32(
        &mut b,
        4,
        words as u32
            | if owner { 1 << 31 } else { 0 }
            | if auto && recv.is_some() { 3 << 10 } else { 0 },
    );
    b
}

pub(super) fn call(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
    handle: u32,
    b: &[u8; 256],
) -> u32 {
    let tls = process.main_thread().tls_base;
    write_guest_bytes(process, tls, b);
    process.main_thread().registers().write_w(x(0), handle);
    assert!(matches!(
        dispatch_next(process, dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    let special = read_guest_u32(process, tls.checked_add(4).unwrap()) & (1 << 31) != 0;
    // Single copy/move handles still leave CMIF at offset 16.
    assert!(!special || read_guest_u32(process, tls.checked_add(8).unwrap()) != 0);
    read_guest_u32(process, tls.checked_add(24).unwrap())
}
pub(super) fn data32(process: &ScheduledProcess) -> u32 {
    read_guest_u32(
        process,
        process.main_thread().tls_base.checked_add(32).unwrap(),
    )
}
pub(super) fn handle(process: &ScheduledProcess) -> u32 {
    read_guest_u32(
        process,
        process.main_thread().tls_base.checked_add(12).unwrap(),
    )
}

#[test]
fn queued_audio_open_retains_the_copied_process_after_its_client_handle_closes() {
    use nixe_memory::{AccessOperation, DeviceAccessEndpoint};
    use std::sync::mpsc;

    let (_directory, mut process) = fixture_process_with_svcs(&[0x21, 0x21]);
    let entry = instruction_address(&process);
    let (child, _) = super::mutex::add_thread(&mut process, entry + 4, 30);
    let caller = process.thread(child).unwrap();
    let backend = Arc::new(Backend::default());
    let manager = process
        .handles()
        .insert(HorizonIpcObject::AudioOutManager(
            crate::AudioOutManagerSession {
                backend: Some(backend.clone()),
                devices: Default::default(),
            },
        ))
        .unwrap();
    let owner = process
        .handles()
        .insert_object(nixe_runtime::HandleObject::from_arc(Arc::clone(&process)))
        .unwrap();
    let listing = process
        .main_thread()
        .stack_bottom
        .checked_add(0x1000)
        .unwrap();
    let name = listing.checked_add(0x1000).unwrap();
    let output = name.checked_add(256).unwrap();
    write_guest_bytes(&process, name, &[0; 256]);
    write_guest_bytes(
        &process,
        process.main_thread().tls_base,
        &command(0, &[], None, Some((listing.get(), 256)), false, false),
    );
    write_abi_register(&process, 0, u64::from(manager));
    let mut open = command(
        1,
        &[0; 16],
        Some((name.get(), 256)),
        Some((output.get(), 256)),
        false,
        true,
    );
    put_u32(&mut open, 20, owner);
    write_guest_bytes(&process, caller.tls_base, &open);
    caller.registers().write_w(x(0), manager);
    let events = process.coordinator_mut().event_sender();
    let mut dispatcher = support::dispatcher(&process).with_input_completions(events);
    let machine = process.memory().account().machine().clone();
    let ranges = process
        .memory()
        .space()
        .capture(listing, 256, MemoryPermissions::WRITE)
        .unwrap();
    let (send, receive) = mpsc::channel();
    let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(super::ipc_lifetime::Device(send));
    machine.set_device_endpoint(&device);
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    assert!(matches!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (id, ExceptionHandlingResult::Suspended) if id == process.main_thread_id()
    ));
    let write = receive.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(write.operation(), AccessOperation::Write(_)));
    caller.start().unwrap();
    assert!(matches!(
        dispatch_scheduled_next(&mut process, &mut dispatcher),
        (id, ExceptionHandlingResult::Suspended) if id == child
    ));
    drop(process.handles().close(owner).unwrap());
    let replacement = process
        .handles()
        .insert(EventObject::new(process.scheduler()))
        .unwrap();
    assert_eq!(replacement & 0x7fff, owner & 0x7fff);
    let value = machine
        .apply_resolved_access(write.ranges(), write.operation())
        .unwrap();
    write.finish(Ok(value));
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    dispatcher
        .poll_memory_completions(process.coordinator_mut())
        .unwrap();
    assert!(
        process
            .scheduler()
            .thread(child)
            .unwrap()
            .active_wait
            .is_none()
    );
    assert_eq!(caller.registers().read_w(x(0)), 0);
    assert_eq!(
        read_guest_u32(&process, caller.tls_base.checked_add(24).unwrap()),
        0
    );
    assert_eq!(backend.live.load(Ordering::SeqCst), 1);
    assert_eq!(
        read_guest_u32(&process, output),
        u32::from_le_bytes(*b"Devi")
    );
}

#[test]
fn output_volume_ipc_controls_already_queued_pcm_and_preserves_invalid_input_errors() {
    let (_directory, mut process) = fixture_process_with_svcs(&[0x21; 4]);
    let backend = Arc::new(Backend::default());
    let mut dispatcher = support::dispatcher(&process).with_audio_backend(backend.clone());
    let session = crate::AudioOutSession::open(
        process.scheduler(),
        backend.as_ref(),
        nixe_audio::PcmFormat::STEREO_48KHZ,
        Default::default(),
    )
    .unwrap();
    let feed = backend.feed.lock().unwrap().take().unwrap();
    let handle = process
        .handles()
        .insert(HorizonIpcObject::AudioOut(session.clone()))
        .unwrap();
    session
        .output
        .lock()
        .unwrap()
        .append(11, vec![1000, -1000, 1000, -1000])
        .unwrap();
    session.output.lock().unwrap().start().unwrap();
    assert_eq!(
        call(
            &mut process,
            &mut dispatcher,
            handle,
            &command(13, &[], None, None, false, false)
        ),
        0
    );
    assert_eq!(f32::from_bits(data32(&process)), 1.0);
    assert_eq!(
        call(
            &mut process,
            &mut dispatcher,
            handle,
            &command(12, &0.7_f32.to_le_bytes(), None, None, false, false)
        ),
        0
    );
    let mut pcm = [0; 2];
    feed.render(&mut pcm);
    assert_eq!(pcm, [700, -700]);
    assert_eq!(
        call(
            &mut process,
            &mut dispatcher,
            handle,
            &command(13, &[], None, None, false, false)
        ),
        0
    );
    assert_eq!(f32::from_bits(data32(&process)), 0.7);
    assert_eq!(
        call(
            &mut process,
            &mut dispatcher,
            handle,
            &command(12, &f32::NAN.to_le_bytes(), None, None, false, false)
        ),
        HorizonIpcResult::AUDIO_INVALID_ARGUMENT.raw()
    );
    feed.render(&mut pcm);
    assert_eq!(pcm, [700, -700]);
}

#[test]
fn nested_pcm_uses_its_own_capture_after_the_descriptor_mapping_changes() {
    use nixe_memory::{AccessOperation, DeviceAccessEndpoint};
    use std::sync::mpsc;

    for pointer in [false, true] {
        let (_directory, mut process) = fixture_process_with_svcs(&[0x21]);
        let events = process.coordinator_mut().event_sender();
        let backend = Arc::new(Backend::default());
        let mut dispatcher = support::dispatcher(&process)
            .with_audio_backend(backend.clone())
            .with_input_completions(events);
        let session = crate::AudioOutSession::open(
            process.scheduler(),
            backend.as_ref(),
            nixe_audio::PcmFormat::STEREO_48KHZ,
            Default::default(),
        )
        .unwrap();
        session.output.lock().unwrap().start().unwrap();
        let feed = backend.feed.lock().unwrap().take().unwrap();
        let handle = process
            .handles()
            .insert(HorizonIpcObject::AudioOut(session))
            .unwrap();
        let descriptor = process
            .main_thread()
            .stack_bottom
            .checked_add(0x1000)
            .unwrap();
        let samples = descriptor.checked_add(8).unwrap();
        let mut data = [0; 40];
        put_u64(&mut data, 8, samples.get());
        put_u64(&mut data, 16, 4);
        put_u64(&mut data, 24, 4);
        write_guest_bytes(&process, descriptor, &data);
        let mut bytes = command(
            if pointer { 7 } else { 3 },
            &123_u64.to_le_bytes(),
            Some((descriptor.get(), 40)),
            None,
            pointer,
            false,
        );
        if pointer {
            put_send_static(&mut bytes, 8, descriptor.get(), 40);
            bytes[16..28].fill(0);
        }
        write_guest_bytes(&process, process.main_thread().tls_base, &bytes);
        write_abi_register(&process, 0, u64::from(handle));
        let mapping = process.memory().query(descriptor).unwrap().mapping.unwrap();
        let machine = process.memory().account().machine().clone();
        let ranges = process
            .memory()
            .space()
            .capture(descriptor, 40, MemoryPermissions::READ)
            .unwrap();
        let (send, receive) = mpsc::channel();
        let device: Arc<dyn DeviceAccessEndpoint> = Arc::new(super::ipc_lifetime::Device(send));
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
        assert_eq!(read.ranges()[0].bytes(), 40);
        process.memory().unmap(descriptor, 0x1000, mapping).unwrap();
        process
            .memory()
            .map_zeroed(descriptor, 0x1000, mapping)
            .unwrap();
        write_guest_bytes(&process, samples, &[0xe8, 0x03, 0x18, 0xfc]);
        let value = machine
            .apply_resolved_access(read.ranges(), read.operation())
            .unwrap();
        read.finish(Ok(value));
        dispatcher
            .poll_memory_completions(process.coordinator_mut())
            .unwrap();
        assert_eq!(read_abi_register(&process, 0), 0);
        let mut pcm = [0; 2];
        feed.render(&mut pcm);
        assert_eq!(pcm, [1000, -1000]);
    }
}

#[test]
fn native_auto_select_rejects_invalid_modes_and_two_active_input_sides() {
    for invalid_mode in [true, false] {
        let (_directory, mut process) = fixture_process_with_svcs(&[0x21]);
        let backend = Backend::default();
        let session = crate::AudioOutSession::open(
            process.scheduler(),
            &backend,
            nixe_audio::PcmFormat::STEREO_48KHZ,
            Default::default(),
        )
        .unwrap();
        let handle = process
            .handles()
            .insert(HorizonIpcObject::AudioOut(session))
            .unwrap();
        let descriptor = process.main_thread().stack_bottom;
        let mut bytes = command(
            7,
            &123_u64.to_le_bytes(),
            Some((descriptor.get(), 40)),
            None,
            true,
            false,
        );
        let expected = if invalid_mode {
            // The A descriptor's mode is checked during common admission.
            bytes[24] |= 2;
            "invalid IPC buffer mode"
        } else {
            put_send_static(&mut bytes, 8, descriptor.get(), 40);
            "auto-select input has both descriptor sides active"
        };
        write_guest_bytes(&process, process.main_thread().tls_base, &bytes);
        write_abi_register(&process, 0, u64::from(handle));
        let mut dispatcher = support::dispatcher(&process);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Fault(HorizonSvcFault::Ipc { immediate: 0x21, fault })
                if fault.to_string() == HorizonIpcFault::malformed(expected).to_string()
        ));
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn audout_ipc_plays_pcm_releases_tags_and_restarts_through_both_buffer_abis() {
    'variants: for (auto, invalid_release, channel_count, device_gain) in [
        (false, false, 2, false),
        (true, false, 2, false),
        (false, true, 2, false),
        (true, true, 2, false),
        (false, false, 6, false),
        (true, false, 6, false),
        (false, false, 2, true),
        (true, false, 6, true),
    ] {
        let (_dir, mut process) =
            fixture_process_with_svcs(&[&[0x1f][..], &[0x21; 128][..]].concat());
        let backend = Arc::new(Backend::default());
        let mut dispatcher = support::dispatcher(&process).with_audio_backend(backend.clone());
        let service_name = process.main_thread().stack_bottom;
        write_guest_bytes(&process, service_name, b"sm:\0");
        process
            .main_thread()
            .registers()
            .write_x(x(1), service_name.get());
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        let sm = process.main_thread().registers().read_w(x(1));
        // RegisterClient sends a PID descriptor.
        let mut register = [0; 256];
        put_u32(&mut register, 0, 4);
        put_u32(&mut register, 4, 10 | (1 << 31));
        put_u32(&mut register, 8, 1);
        put_u32(&mut register, 16, 0);
        put_u32(&mut register, 32, 0x4943_4653);
        assert_eq!(call(&mut process, &mut dispatcher, sm, &register), 0);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                sm,
                &command(1, b"audout:u", None, None, false, false)
            ),
            0
        );
        let manager = handle(&process);
        let base = process.main_thread().stack_bottom;
        let device = if device_gain {
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    sm,
                    &command(1, b"audren:u", None, None, false, false)
                ),
                0
            );
            let renderer_manager = handle(&process);
            let mut params = [0; 16];
            params[..4].copy_from_slice(b"REV7");
            params[8..].copy_from_slice(&1_u64.to_le_bytes());
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    renderer_manager,
                    &command(4, &params, None, None, false, false)
                ),
                0
            );
            let device = handle(&process);
            write_guest_bytes(
                &process,
                base.checked_add(2048).unwrap(),
                b"AudioTvOutput\0",
            );
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    device,
                    &command(
                        if auto { 7 } else { 1 },
                        &1.85_f32.to_le_bytes(),
                        Some((base.get() + 2048, 256)),
                        None,
                        auto,
                        false
                    )
                ),
                0
            );
            Some(device)
        } else {
            None
        };
        let out_name = base.checked_add(256).unwrap();
        let descriptor = base.checked_add(512).unwrap();
        let samples = base.checked_add(4096).unwrap();
        let releases = base.checked_add(768).unwrap();
        write_guest_bytes(&process, base, &[0; 256]);
        let mut config = [0; 16];
        put_u32(
            &mut config,
            4,
            if channel_count == 2 { 0x0002_0000 } else { 6 },
        ); // libnx's reserved high half.
        let open = command(
            if auto { 3 } else { 1 },
            &config,
            Some((base.get(), 256)),
            Some((out_name.get(), 256)),
            auto,
            true,
        );
        // Ignoring transport padding must not bypass required owner metadata.
        let missing_owner = command(
            if auto { 3 } else { 1 },
            &config,
            Some((base.get(), 256)),
            Some((out_name.get(), 256)),
            auto,
            false,
        );
        assert_eq!(
            call(&mut process, &mut dispatcher, manager, &missing_owner),
            0x1a60a
        );
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
        for (rate, channels, error) in [(44_100, 2, 3), (48_000, 1, 10)] {
            let mut invalid = config;
            put_u32(&mut invalid, 0, rate);
            put_u32(&mut invalid, 4, channels);
            let request = command(
                if auto { 3 } else { 1 },
                &invalid,
                Some((base.get(), 256)),
                Some((out_name.get(), 256)),
                auto,
                true,
            );
            assert_eq!(
                call(&mut process, &mut dispatcher, manager, &request),
                153 | (error << 9)
            );
            assert_eq!(backend.live.load(Ordering::SeqCst), 0);
        }
        write_guest_bytes(&process, base, b"MissingDevice\0");
        assert_eq!(
            call(&mut process, &mut dispatcher, manager, &open),
            153 | (1 << 9)
        );
        write_guest_bytes(&process, base, &[0; 256]);
        assert_eq!(call(&mut process, &mut dispatcher, manager, &open), 0);
        let audio = handle(&process);
        assert_eq!(data32(&process), 48_000);
        assert_eq!(
            read_guest_u32(
                &process,
                process.main_thread().tls_base.checked_add(36).unwrap()
            ),
            channel_count as u32
        );
        assert_eq!(
            read_guest_u32(&process, out_name),
            u32::from_le_bytes(*b"Devi")
        );
        assert_eq!(backend.live.load(Ordering::SeqCst), 1);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                audio,
                &command(4, &[], None, None, false, false)
            ),
            0
        );
        let event_handle = handle(&process);
        let event = process
            .handles()
            .lookup_as::<ReadableEventObject>(event_handle)
            .unwrap()
            .clone();
        assert!(!event.is_signalled());
        let mut buffer = [0; 40];
        // Reserved fields must not chain buffers or shift the PCM address.
        put_u64(&mut buffer, 0, u64::MAX);
        put_u64(&mut buffer, 32, u64::MAX);
        put_u64(&mut buffer, 8, samples.get());
        put_u64(&mut buffer, 16, 4096);
        put_u64(&mut buffer, 24, channel_count as u64 * 4);
        write_guest_bytes(&process, descriptor, &buffer);
        let first: Vec<i16> =
            [10000, -10000, 20000, -20000, 30000, -30000][..channel_count].to_vec();
        let second: Vec<i16> = first.iter().map(|v| v / 2).collect();
        let bytes: Vec<u8> = first
            .iter()
            .chain(&second)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_guest_bytes(&process, samples, &bytes);
        let append = command(
            if auto { 7 } else { 3 },
            &123_u64.to_le_bytes(),
            Some((descriptor.get(), 40)),
            None,
            auto,
            false,
        );
        let release = command(
            if auto { 8 } else { 5 },
            &[],
            None,
            Some((releases.get(), 16)),
            auto,
            false,
        );
        assert_eq!(call(&mut process, &mut dispatcher, audio, &append), 0);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                audio,
                &command(0, &[], None, None, false, false)
            ),
            0
        );
        assert_eq!(data32(&process), 1); // stopped
        write_guest_bytes(&process, releases, &[0xa5; 24]);
        assert_eq!(call(&mut process, &mut dispatcher, audio, &release), 0);
        assert_eq!(data32(&process), 0);
        assert_eq!(read_guest_bytes(&process, releases, 16), [0; 16]);
        assert_eq!(
            read_guest_bytes(&process, releases.checked_add(16).unwrap(), 8),
            [0xa5; 8]
        );
        let feed = backend.feed.lock().unwrap().clone().unwrap();
        let mut pcm = vec![9; channel_count];
        feed.render(&mut pcm);
        assert_eq!(pcm, vec![0; channel_count]);
        for round in 0..2 {
            if round == 1 {
                // Existing streams must observe a later device update, including
                // when their manager was opened before the gain was configured.
                if let Some(device) = device {
                    assert_eq!(
                        call(
                            &mut process,
                            &mut dispatcher,
                            device,
                            &command(
                                if auto { 7 } else { 1 },
                                &0.5_f32.to_le_bytes(),
                                Some((base.get() + 2048, 256)),
                                None,
                                auto,
                                false
                            )
                        ),
                        0
                    );
                }
                assert_eq!(call(&mut process, &mut dispatcher, audio, &append), 0);
            }
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    audio,
                    &command(1, &[], None, None, false, false)
                ),
                0
            );
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    audio,
                    &command(1, &[], None, None, false, false)
                ),
                153 | (2 << 9)
            );
            feed.render(&mut pcm);
            let (expected_first, expected_second): (&[i16], &[i16]) = match (device_gain, round) {
                (true, 0) => (
                    &[18500, -18500, 32767, -32768, 32767, -32768],
                    &[9250, -9250, 18500, -18500, 27750, -27750],
                ),
                (true, _) => (
                    &[5000, -5000, 10000, -10000, 15000, -15000],
                    &[2500, -2500, 5000, -5000, 7500, -7500],
                ),
                (false, _) => (&first, &second),
            };
            assert_eq!(pcm, expected_first[..channel_count]);
            assert!(!event.is_signalled());
            feed.render(&mut pcm);
            assert_eq!(pcm, expected_second[..channel_count]);
            assert!(event.is_signalled());
            if invalid_release {
                // A release list starts in writable RAM but crosses into a
                // read-only page. Validate EVERY destination page before
                // draining tags or clearing the release event.
                let crossing = base.checked_add(4092).unwrap();
                process
                    .memory()
                    .space()
                    .protect(
                        base.checked_add(4096).unwrap(),
                        4096,
                        MemoryPermissions::READ,
                    )
                    .unwrap();
                let release = command(
                    if auto { 8 } else { 5 },
                    &[],
                    None,
                    Some((crossing.get(), 16)),
                    auto,
                    false,
                );
                write_guest_bytes(&process, process.main_thread().tls_base, &release);
                process.main_thread().registers().write_w(x(0), audio);
                let ExceptionHandlingResult::Rejected(fault) =
                    dispatch_next(&mut process, &mut dispatcher)
                else {
                    panic!("read-only release destination must reject the IPC request");
                };
                assert!(
                    matches!(fault, HorizonSvcFault::Ipc { immediate: 0x21, fault }
                    if fault.guest_result() == Some(HorizonKernelResult::INVALID_POINTER))
                );
                assert_eq!(
                    process.main_thread().registers().read_w(x(0)),
                    HorizonKernelResult::INVALID_POINTER.raw()
                );
                assert!(
                    event.is_signalled(),
                    "failed validation must retain released tags"
                );
                process
                    .memory()
                    .space()
                    .protect(
                        base.checked_add(4096).unwrap(),
                        4096,
                        MemoryPermissions::READ_WRITE,
                    )
                    .unwrap();
                assert_eq!(call(&mut process, &mut dispatcher, audio, &release), 0);
                assert_eq!(data32(&process), 1);
                assert_eq!(
                    read_guest_bytes(&process, crossing, 8),
                    123_u64.to_le_bytes()
                );
                assert!(!event.is_signalled());
                continue 'variants;
            }
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    audio,
                    &command(9, &[], None, None, false, false)
                ),
                0
            );
            assert_eq!(data32(&process), 0);
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    audio,
                    &command(6, &123_u64.to_le_bytes(), None, None, false, false)
                ),
                0
            );
            assert_eq!(data32(&process), 1);
            write_guest_bytes(&process, releases, &[0xa5; 24]);
            assert_eq!(call(&mut process, &mut dispatcher, audio, &release), 0);
            assert_eq!(data32(&process), 1);
            let mut expected = [0; 16];
            expected[..8].copy_from_slice(&123_u64.to_le_bytes());
            assert_eq!(read_guest_bytes(&process, releases, 16), expected);
            assert_eq!(
                read_guest_bytes(&process, releases.checked_add(16).unwrap(), 8),
                [0xa5; 8]
            );
            assert!(!event.is_signalled());
            // Polling again must not return the previous tag from reused RAM.
            assert_eq!(call(&mut process, &mut dispatcher, audio, &release), 0);
            assert_eq!(data32(&process), 0);
            assert_eq!(read_guest_bytes(&process, releases, 16), [0; 16]);
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    audio,
                    &command(2, &[], None, None, false, false)
                ),
                0
            );
        }
        // Released but uncollected buffers occupy slots; pending count is separate.
        for _ in 0..32 {
            assert_eq!(call(&mut process, &mut dispatcher, audio, &append), 0);
        }
        assert_eq!(
            call(&mut process, &mut dispatcher, audio, &append),
            153 | (8 << 9)
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                audio,
                &command(1, &[], None, None, false, false)
            ),
            0
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                audio,
                &command(2, &[], None, None, false, false)
            ),
            0
        );
        assert!(event.is_signalled());
        assert_eq!(
            call(&mut process, &mut dispatcher, audio, &append),
            153 | (8 << 9)
        );
        let drain = command(
            if auto { 8 } else { 5 },
            &[],
            None,
            Some((releases.get(), 256)),
            auto,
            false,
        );
        assert_eq!(call(&mut process, &mut dispatcher, audio, &drain), 0);
        assert_eq!(data32(&process), 32);
        assert!(!event.is_signalled());
        // A failed host stream wakes the event and preserves an actionable error.
        feed.fail(AudioError("test device disconnected".into()));
        assert!(event.is_signalled());
        let tls = process.main_thread().tls_base;
        write_guest_bytes(&process, tls, &release);
        process.main_thread().registers().write_w(x(0), audio);
        let ExceptionHandlingResult::Fault(fault) = dispatch_next(&mut process, &mut dispatcher)
        else {
            panic!("host audio failure must reject the IPC");
        };
        assert!(fault.to_string().contains("test device disconnected"));
        assert!(process.handles().lookup(audio).is_none());
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
    }
}
