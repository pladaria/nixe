use super::audout::{call, command, data32, handle};
use super::*;
use nixe_audio::{AudioBackend, AudioError, AudioFeed, ChannelLayout};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
struct Backend(AtomicBool);
impl AudioBackend for Backend {
    fn output_layout(&self) -> Result<ChannelLayout, AudioError> {
        if self.0.load(Ordering::Relaxed) {
            Ok(ChannelLayout::Stereo)
        } else {
            Err(AudioError("test audio backend shut down".into()))
        }
    }
    fn open(&self, _: AudioFeed) -> Result<Box<dyn nixe_audio::AudioDevice>, AudioError> {
        panic!("device metadata must not create a playback stream")
    }
}

#[test]
fn modern_output_names_round_trip_volume_without_changing_the_legacy_port_list() {
    for auto in [false, true] {
        let (_directory, mut process) = fixture_process_with_svcs(&[0x21; 16]);
        let backend = Arc::new(Backend(AtomicBool::new(true)));
        let mut dispatcher = support::dispatcher(&process).with_audio_backend(backend);
        let manager = process
            .handles()
            .insert(HorizonIpcObject::AudioRendererManager(Default::default()))
            .unwrap();
        let mut params = [0; 16];
        params[..4].copy_from_slice(b"REV;"); // REV11, device API v2.
        params[8..].copy_from_slice(&1_u64.to_le_bytes());
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager,
                &command(4, &params, None, None, false, false)
            ),
            0
        );
        let first = handle(&process);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager,
                &command(4, &params, None, None, false, false)
            ),
            0
        );
        let second = handle(&process);
        let input = process.main_thread().stack_bottom;
        let output = input.checked_add(0x1000).unwrap();
        let mut name = [0; 256];
        name[..19].copy_from_slice(b"AudioExternalOutput");
        write_guest_bytes(&process, input, &name);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                first,
                &command(
                    if auto { 7 } else { 1 },
                    &2_f32.to_le_bytes(),
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                second,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(data32(&process), 2_f32.to_bits());
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                first,
                &command(
                    if auto { 6 } else { 0 },
                    &[],
                    None,
                    Some((output.get(), 1280)),
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(data32(&process), 4);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                first,
                &command(14, &[], None, Some((output.get(), 768)), false, false)
            ),
            0
        );
        assert_eq!(data32(&process), 3);
        assert_eq!(
            read_guest_bytes(&process, output.checked_add(512).unwrap(), 19),
            b"AudioExternalOutput"
        );
        name.fill(0);
        name[..13].copy_from_slice(b"AudioTvOutput");
        write_guest_bytes(&process, input, &name);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                second,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(data32(&process), 1_f32.to_bits());
    }
}

#[test]
fn audio_device_ipc_shares_real_gains_filters_revisions_and_separates_notifications() {
    for auto in [false, true] {
        let (_dir, mut process) = fixture_process_with_svcs(&[0x21; 100]);
        let backend = Arc::new(Backend(AtomicBool::new(true)));
        let mut dispatcher = support::dispatcher(&process).with_audio_backend(backend.clone());
        let manager = nixe_horizon::AudioRendererManagerSession::default();
        let manager_handle = process
            .handles()
            .insert(HorizonIpcObject::AudioRendererManager(manager.clone()))
            .unwrap();
        let another_manager = process
            .handles()
            .insert(HorizonIpcObject::AudioRendererManager(manager))
            .unwrap();
        let mut params = [0xa5; 16];
        params[..4].copy_from_slice(b"bad!");
        params[8..].copy_from_slice(&1_u64.to_le_bytes());
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager_handle,
                &command(4, &params, None, None, false, false)
            ),
            HorizonIpcResult::AUDIO_INVALID_REVISION.raw()
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager_handle,
                &command(4, &[], None, None, false, false)
            ),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
        );
        params[..4].copy_from_slice(b"REV7");
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager_handle,
                &command(4, &params, None, None, false, false)
            ),
            0
        );
        let device = handle(&process);
        assert!(matches!(
            process
                .handles()
                .lookup_as::<HorizonIpcObject>(device)
                .as_deref(),
            Some(HorizonIpcObject::AudioDevice(_))
        ));
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                another_manager,
                &command(2, &2_u64.to_le_bytes(), None, None, false, false)
            ),
            0
        );
        let legacy_device = handle(&process);
        process.handles().close(manager_handle).unwrap();
        process.handles().close(another_manager).unwrap();
        let input = process
            .main_thread()
            .stack_bottom
            .checked_add(0x1000)
            .unwrap();
        let output = input.checked_add(0x1000).unwrap();
        let names = [
            b"AudioStereoJackOutput".as_slice(),
            b"AudioBuiltInSpeakerOutput",
            b"AudioTvOutput",
            b"AudioUsbDeviceOutput",
        ];
        for (session, expected_count) in [(device, 4), (legacy_device, 3)] {
            write_guest_bytes(&process, output, &[0xcc; 1040]);
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    session,
                    &command(
                        if auto { 6 } else { 0 },
                        &[],
                        None,
                        Some((output.get(), 1040)),
                        auto,
                        false
                    )
                ),
                0
            );
            assert_eq!(data32(&process), expected_count);
            let bytes = read_guest_bytes(&process, output, 1040);
            for (index, name) in names.iter().take(expected_count as usize).enumerate() {
                let entry = &bytes[index * 256..(index + 1) * 256];
                assert_eq!(&entry[..name.len()], *name);
                assert!(entry[name.len()..].iter().all(|&byte| byte == 0));
            }
            assert!(
                bytes[expected_count as usize * 256..]
                    .iter()
                    .all(|&byte| byte == 0xcc)
            );
        }
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 6 } else { 0 },
                    &[],
                    None,
                    Some((0, 0)),
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(data32(&process), 0);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 10 } else { 3 },
                    &[],
                    None,
                    Some((output.get(), 255)),
                    auto,
                    false
                )
            ),
            HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER.raw()
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 10 } else { 3 },
                    &[],
                    None,
                    Some((output.get(), 256)),
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(&read_guest_bytes(&process, output, 14), b"AudioTvOutput\0");
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(5, &[], None, None, false, false)
            ),
            0
        );
        assert_eq!(data32(&process), 2);
        let mut name = [0; 256];
        name[..13].copy_from_slice(b"AudioTvOutput");
        write_guest_bytes(&process, input, &name);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 7 } else { 1 },
                    &0.25_f32.to_le_bytes(),
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                legacy_device,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(data32(&process), 0.25_f32.to_bits());
        // The actual SDK sets TV gain to 1.85: this API is not a normalized
        // [0, 1] slider, and reconnecting must not discard that amplification.
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 7 } else { 1 },
                    &1.85_f32.to_le_bytes(),
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                legacy_device,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            0
        );
        assert_eq!(data32(&process), 1.85_f32.to_bits());
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 7 } else { 1 },
                    &f32::NAN.to_le_bytes(),
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            HorizonIpcResult::AUDIO_INVALID_ARGUMENT.raw()
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 255)),
                    None,
                    auto,
                    false
                )
            ),
            HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER.raw()
        );
        name.fill(0);
        name[..20].copy_from_slice(b"AudioUsbDeviceOutput");
        write_guest_bytes(&process, input, &name);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                legacy_device,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            HorizonIpcResult::AUDIO_NOT_FOUND.raw()
        );
        write_guest_bytes(&process, input, &[b'x'; 256]);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(
                    if auto { 8 } else { 2 },
                    &[],
                    Some((input.get(), 256)),
                    None,
                    auto,
                    false
                )
            ),
            HorizonIpcResult::AUDIO_INVALID_ARGUMENT.raw()
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(4, &[], None, None, false, false)
            ),
            0
        );
        let system_event = handle(&process);
        let event = process
            .handles()
            .lookup_as::<ReadableEventObject>(system_event)
            .unwrap()
            .clone();
        assert!(event.is_signalled());
        event.clear();
        for command_id in [4, 12] {
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    device,
                    &command(command_id, &[], None, None, false, false)
                ),
                0
            );
            let copied = handle(&process);
            assert!(
                !process
                    .handles()
                    .lookup_as::<ReadableEventObject>(copied)
                    .unwrap()
                    .is_signalled()
            );
            process.handles().close(copied).unwrap();
        }
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(11, &[], None, None, false, false)
            ),
            0
        );
        assert!(
            !process
                .handles()
                .lookup_as::<ReadableEventObject>(handle(&process))
                .unwrap()
                .is_signalled()
        );
        // New output-list commands are separate from the legacy port names.
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(14, &[], None, Some((output.get(), 768)), false, false)
            ),
            0
        );
        assert_eq!(data32(&process), 3);
        assert_eq!(
            &read_guest_bytes(&process, output.checked_add(512).unwrap(), 20),
            b"AudioExternalOutput\0"
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                device,
                &command(13, &[], None, Some((output.get(), 256)), false, false)
            ),
            0
        );
        assert_eq!(&read_guest_bytes(&process, output, 14), b"AudioTvOutput\0");
        // CMIF clones retain device state and notification identity.
        let mut clone_request = command(2, &[], None, None, false, false);
        put_u32(&mut clone_request, 0, 5);
        assert_eq!(
            call(&mut process, &mut dispatcher, device, &clone_request),
            0
        );
        let cloned = handle(&process);
        process.handles().close(device).unwrap();
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                cloned,
                &command(5, &[], None, None, false, false)
            ),
            0
        );
        assert_eq!(data32(&process), 2);
        backend.0.store(false, Ordering::Relaxed);
        let request = command(5, &[], None, None, false, false);
        write_guest_bytes(&process, process.main_thread().tls_base, &request);
        state(&mut process).write_w(x(0), cloned);
        let failure = dispatch_next(&mut process, &mut dispatcher);
        assert!(matches!(failure, ExceptionHandlingResult::Fault(_)));
        assert!(format!("{failure:?}").contains("shut down"));
    }
}
