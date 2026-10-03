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
fn command(
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

fn call(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
    handle: u32,
    b: &[u8; 256],
) -> u32 {
    let tls = process.main_thread().tls_base;
    write_guest_bytes(process, tls, b);
    state(process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(process, dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let special = read_guest_u32(process, tls.checked_add(4).unwrap()) & (1 << 31) != 0;
    // Single copy/move handles still leave CMIF at offset 16.
    assert!(!special || read_guest_u32(process, tls.checked_add(8).unwrap()) != 0);
    read_guest_u32(process, tls.checked_add(24).unwrap())
}
fn data32(process: &ScheduledProcess) -> u32 {
    read_guest_u32(
        process,
        process.main_thread().tls_base.checked_add(32).unwrap(),
    )
}
fn handle(process: &ScheduledProcess) -> u32 {
    read_guest_u32(
        process,
        process.main_thread().tls_base.checked_add(12).unwrap(),
    )
}

#[test]
fn audout_ipc_plays_pcm_releases_tags_and_restarts_through_both_buffer_abis() {
    for auto in [false, true] {
        let (_dir, mut process) =
            fixture_process_with_svcs(&[&[0x1f][..], &[0x21; 100][..]].concat());
        let backend = Arc::new(Backend::default());
        let mut dispatcher = HorizonSvcDispatcher::default().with_audio_backend(backend.clone());
        let service_name = process.main_thread().stack_bottom;
        write_guest_bytes(&process, service_name, b"sm:\0");
        state(&mut process).write_x(x(1), service_name.get());
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        let sm = state(&mut process).read_w(x(1));
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
        let out_name = base.checked_add(256).unwrap();
        let descriptor = base.checked_add(512).unwrap();
        let samples = base.checked_add(4096).unwrap();
        let releases = base.checked_add(768).unwrap();
        write_guest_bytes(&process, base, &[0; 256]);
        let mut config = [0; 16];
        put_u32(&mut config, 4, 0x0002_0000); // libnx's reserved high half.
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
            2
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
            .get_as::<ReadableEventObject>(event_handle)
            .unwrap()
            .clone();
        assert!(!event.is_signalled());
        let mut buffer = [0; 40];
        put_u64(&mut buffer, 8, samples.get());
        put_u64(&mut buffer, 16, 4096);
        put_u64(&mut buffer, 24, 8);
        write_guest_bytes(&process, descriptor, &buffer);
        write_guest_bytes(&process, samples, &[1, 0, 0xff, 0xff, 2, 0, 0xfe, 0xff]);
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
            Some((releases.get(), 8)),
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
        assert_eq!(call(&mut process, &mut dispatcher, audio, &release), 0);
        assert_eq!(data32(&process), 0);
        let feed = backend.feed.lock().unwrap().clone().unwrap();
        let mut pcm = [9; 2];
        feed.render(&mut pcm);
        assert_eq!(pcm, [0, 0]);
        for round in 0..2 {
            if round == 1 {
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
            assert_eq!(pcm, [1, -1]);
            assert!(!event.is_signalled());
            feed.render(&mut pcm);
            assert_eq!(pcm, [2, -2]);
            assert!(event.is_signalled());
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
            assert_eq!(call(&mut process, &mut dispatcher, audio, &release), 0);
            assert_eq!(data32(&process), 1);
            assert_eq!(read_guest_u32(&process, releases), 123);
            assert!(!event.is_signalled());
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
        state(&mut process).write_w(x(0), audio);
        let ExceptionHandlingResult::Fault(fault) = dispatch_next(&mut process, &mut dispatcher)
        else {
            panic!("host audio failure must reject the IPC");
        };
        assert!(fault.to_string().contains("test device disconnected"));
        process.handles_mut().close(audio).unwrap();
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
    }
}
