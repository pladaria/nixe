use super::audout::{call, command, data32, handle};
use super::*;

// A 440 Hz test tone encoded independently with libopus, 48 kHz mono, 20 ms.
const PACKET: &[u8] = &[
    0x78, 0x83, 0x3f, 0x2d, 0x56, 0x82, 0x74, 0xf4, 0x00, 0x00, 0x0a, 0x96, 0xde, 0x06, 0x82, 0x2a,
    0xbc, 0x32, 0x97, 0xeb, 0x59, 0x29, 0x12, 0xbb, 0x06, 0xfb, 0x22, 0xfc, 0xb4, 0xbf, 0xc3, 0x41,
    0x99, 0x88, 0x3a, 0x6c, 0xd2, 0x5c, 0x63, 0x35, 0xe8, 0x47, 0xba, 0x31, 0xfa, 0x2a, 0x76, 0xa1,
    0xe2, 0xaa, 0x57, 0x89, 0xea, 0xdb, 0x36, 0x74, 0x8c, 0x64, 0x01, 0xb3, 0x3f, 0x6b, 0xe0, 0x4f,
    0xcb, 0x54, 0x3e, 0x77, 0x68, 0xbd, 0x34, 0x00, 0xb3,
];

fn manager_command(
    id: u32,
    inline: &[u8],
    copy: Option<u32>,
    pointer: Option<(u64, u16)>,
) -> [u8; 256] {
    let mut bytes = [0xa5; 256];
    let mut cursor = 8;
    if let Some(handle) = copy {
        put_u32(&mut bytes, cursor, 2);
        put_u32(&mut bytes, cursor + 4, handle);
        cursor += 8;
    }
    if let Some((address, size)) = pointer {
        put_send_static(&mut bytes, cursor, address, size);
        cursor += 8;
    }
    let raw = cursor;
    cursor = (cursor + 15) & !15;
    put_u32(&mut bytes, cursor, 0x4943_4653);
    put_u32(&mut bytes, cursor + 4, 0);
    put_u32(&mut bytes, cursor + 8, id);
    put_u32(&mut bytes, cursor + 12, 0x1234);
    bytes[cursor + 16..cursor + 16 + inline.len()].copy_from_slice(inline);
    put_u32(
        &mut bytes,
        0,
        4 | if pointer.is_some() { 1 << 16 } else { 0 },
    );
    let words = (32 + inline.len()).div_ceil(4);
    assert!(raw + words * 4 <= 256);
    put_u32(
        &mut bytes,
        4,
        words as u32 | if copy.is_some() { 1 << 31 } else { 0 },
    );
    bytes
}

#[test]
fn opus_service_negotiates_opens_and_decodes_through_all_packet_abis() {
    for (query, open, channels, multi, extended, large) in [
        (1, 0, 1_u32, false, false, false),
        (5, 4, 2, false, true, false),
        (8, 4, 1, false, true, true),
        (3, 2, 2, true, false, false),
        (7, 6, 2, true, true, false),
        (9, 6, 2, true, true, true),
    ] {
        let mut svcs = vec![0x1f, 0x21, 0x21, 0x21, 0x15];
        svcs.extend_from_slice(&[0x21; 32]);
        let (_directory, mut process) = fixture_process_with_svcs(&svcs);
        let mut dispatcher = support::dispatcher(&process);
        let base = process.main_thread().stack_bottom;
        write_guest_bytes(&process, base, b"sm:\0");
        state(&mut process).write_x(x(1), base.get());
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        let sm = state(&mut process).read_w(x(1));
        let mut register = [0; 256];
        put_u32(&mut register, 0, 4);
        put_u32(&mut register, 4, 10 | (1 << 31));
        put_u32(&mut register, 8, 1);
        put_u32(&mut register, 32, 0x4943_4653);
        assert_eq!(call(&mut process, &mut dispatcher, sm, &register), 0);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                sm,
                &command(1, b"hwopus\0\0", None, None, false, false)
            ),
            0
        );
        let manager = handle(&process);
        let mut params = vec![
            0;
            match (multi, extended) {
                (false, false) => 8,
                (false, true) => 16,
                (true, false) => 0x110,
                (true, true) => 0x118,
            }
        ];
        put_u32(&mut params, 0, 48000);
        put_u32(&mut params, 4, channels);
        if multi {
            put_u32(&mut params, 8, 1);
            let mapping = if extended { 0x18 } else { 0x10 };
            params[mapping + 1] = 255; // Mute channel 1; channel 0 carries the tone.
        }
        if extended {
            params[if multi { 16 } else { 8 }] = u8::from(large);
        }
        let parameter_address = base.checked_add(0x200).unwrap();
        write_guest_bytes(&process, parameter_address, &params);
        let pointer = multi.then_some((parameter_address.get(), params.len() as u16));
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager,
                &manager_command(query, if multi { &[] } else { &params }, None, pointer)
            ),
            0
        );
        let required = data32(&process);
        let work_size = u64::from(required).next_multiple_of(4096);
        let work_address = process.memory_layout().heap().base();
        process
            .memory()
            .map_zeroed(
                work_address,
                work_size,
                nixe_runtime::memory::Mapping {
                    kind: nixe_runtime::memory::MemoryKind::Heap,
                    permissions: MemoryPermissions::READ_WRITE,
                    uncached: false,
                    locked: false,
                    permission_locked: false,
                },
            )
            .unwrap();
        state(&mut process).write_x(x(1), work_address.get());
        state(&mut process).write_x(x(2), work_size);
        state(&mut process).write_w(x(3), 0);
        assert!(matches!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        assert_eq!(state(&mut process).read_w(x(0)), 0);
        let transfer = state(&mut process).read_w(x(1));
        let mut opening = if multi { Vec::new() } else { params.clone() };
        opening.extend_from_slice(&(required - 1).to_le_bytes());
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager,
                &manager_command(open, &opening, Some(transfer), pointer)
            ),
            111 | (3 << 9)
        );
        let work_offset = opening.len() - 4;
        put_u32(&mut opening, work_offset, work_size as u32);
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                manager,
                &manager_command(open, &opening, Some(transfer), pointer)
            ),
            0
        );
        let decoder = handle(&process);
        process.handles().close(transfer).unwrap();
        let mut input = Vec::new();
        input.extend_from_slice(&(PACKET.len() as u32).to_be_bytes());
        input.extend_from_slice(&0_u32.to_be_bytes());
        input.extend_from_slice(PACKET);
        input.extend_from_slice(&[0xee; 4]); // Transport padding is not consumed.
        let packet_address = base.checked_add(0x1000).unwrap();
        let output_address = base.checked_add(0x2000).unwrap();
        write_guest_bytes(&process, packet_address, &input);
        let pcm_size = 960 * channels as usize * 2;
        let mut initial = Vec::new();
        for decode in if multi { [2, 5, 7, 9] } else { [0, 4, 6, 8] } {
            write_guest_bytes(&process, output_address, &vec![0xa5; pcm_size + 16]);
            assert_eq!(
                call(
                    &mut process,
                    &mut dispatcher,
                    decoder,
                    &command(
                        decode,
                        if decode >= 6 { &[1] } else { &[] },
                        Some((packet_address.get(), input.len() as u64)),
                        Some((output_address.get(), (pcm_size + 16) as u64)),
                        false,
                        false
                    )
                ),
                0
            );
            assert_eq!(data32(&process), PACKET.len() as u32 + 8);
            assert_eq!(
                read_guest_u32(
                    &process,
                    process.main_thread().tls_base.checked_add(36).unwrap()
                ),
                960
            );
            let pcm = read_guest_bytes(&process, output_address, pcm_size);
            assert!(
                pcm.chunks_exact(2 * channels as usize)
                    .any(|frame| i16::from_le_bytes(frame[..2].try_into().unwrap()).abs() > 4000)
            );
            if multi {
                assert!(pcm.chunks_exact(4).all(|frame| frame[2..] == [0, 0]));
            }
            if initial.is_empty() {
                initial = pcm;
            } else if decode >= 6 {
                assert_eq!(pcm, initial);
            }
            assert_eq!(
                read_guest_bytes(
                    &process,
                    output_address.checked_add(pcm_size as u64).unwrap(),
                    16
                ),
                vec![0xa5; 16]
            );
        }
        // Reject truncated packets and insufficient PCM capacity through CMIF.
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                decoder,
                &command(
                    if multi { 9 } else { 8 },
                    &[0],
                    Some((packet_address.get(), 8)),
                    Some((output_address.get(), pcm_size as u64)),
                    false,
                    false
                )
            ),
            111 | (8 << 9)
        );
        assert_eq!(
            call(
                &mut process,
                &mut dispatcher,
                decoder,
                &command(
                    if multi { 9 } else { 8 },
                    &[0],
                    Some((packet_address.get(), input.len() as u64)),
                    Some((output_address.get(), 2)),
                    false,
                    false
                )
            ),
            111 | (3 << 9)
        );
    }
}

fn in_domain(mut bytes: [u8; 256], object: u32, scalar_bytes: usize) -> [u8; 256] {
    let start = bytes.windows(4).position(|word| word == b"SFCI").unwrap();
    bytes.copy_within(start..240, start + 16);
    bytes[start..start + 16].fill(0);
    bytes[start] = 1;
    bytes[start + 2..start + 4].copy_from_slice(&((16 + scalar_bytes) as u16).to_le_bytes());
    put_u32(&mut bytes, start + 4, object);
    put_u32(&mut bytes, start + 28, 0);
    let words = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) + 4;
    put_u32(&mut bytes, 4, words);
    bytes
}

fn domain_call(
    process: &mut ScheduledProcess,
    dispatcher: &mut HorizonSvcDispatcher,
    handle: u32,
    bytes: &[u8; 256],
) -> u32 {
    let tls = process.main_thread().tls_base;
    write_guest_bytes(process, tls, bytes);
    state(process).write_w(x(0), handle);
    assert!(matches!(
        dispatch_next(process, dispatcher),
        ExceptionHandlingResult::Resumed
    ));
    read_guest_u32(process, tls.checked_add(40).unwrap())
}

#[test]
fn opus_domain_owns_decoders_across_clones_and_closes_each_child_independently() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 16]);
    let mut dispatcher = support::dispatcher(&process);
    let manager = process
        .handles()
        .insert(HorizonIpcObject::OpusManager(Default::default()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let mut control = command(0, &[], None, None, false, false);
    put_u32(&mut control, 0, 5);
    assert_eq!(call(&mut process, &mut dispatcher, manager, &control), 0);
    assert_eq!(data32(&process), 1);
    let mut params = [0; 16];
    put_u32(&mut params, 0, 48000);
    put_u32(&mut params, 4, 1);
    let query = in_domain(manager_command(5, &params, None, None), 1, 16);
    assert_eq!(
        domain_call(&mut process, &mut dispatcher, manager, &query),
        0
    );
    let work = read_guest_u32(&process, tls.checked_add(48).unwrap()).next_multiple_of(4096);
    let transfer = process
        .handles()
        .insert_object(nixe_runtime::HandleObject::from_arc(
            crate::test_memory::transfer(&vec![0; work as usize]),
        ))
        .unwrap();
    let mut opening = params.to_vec();
    opening.extend_from_slice(&work.to_le_bytes());
    assert_eq!(
        domain_call(
            &mut process,
            &mut dispatcher,
            manager,
            &in_domain(
                manager_command(4, &opening, Some(transfer), None),
                1,
                opening.len()
            )
        ),
        0
    );
    let decoder = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(decoder, 2);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(4).unwrap()) >> 31,
        0
    );
    process.handles().close(transfer).unwrap();
    put_u32(&mut control, 24, 2);
    assert_eq!(call(&mut process, &mut dispatcher, manager, &control), 0);
    let clone = handle(&process);
    assert_ne!(clone, manager);
    let input = process.main_thread().stack_bottom;
    let output = input.checked_add(0x1000).unwrap();
    let mut packet = vec![0; 8];
    packet[..4].copy_from_slice(&(PACKET.len() as u32).to_be_bytes());
    packet.extend_from_slice(PACKET);
    write_guest_bytes(&process, input, &packet);
    let decode = in_domain(
        command(
            8,
            &[1],
            Some((input.get(), packet.len() as u64)),
            Some((output.get(), 1920)),
            false,
            false,
        ),
        decoder,
        1,
    );
    assert_eq!(
        domain_call(&mut process, &mut dispatcher, clone, &decode),
        0
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(48).unwrap()),
        packet.len() as u32
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(52).unwrap()), 960);
    assert!(
        read_guest_bytes(&process, output, 1920)
            .chunks_exact(2)
            .any(|sample| i16::from_le_bytes(sample.try_into().unwrap()).abs() > 4000)
    );
    let mut close = [0; 256];
    put_u32(&mut close, 0, 4);
    put_u32(&mut close, 4, 8);
    close[16] = 2;
    put_u32(&mut close, 20, decoder);
    assert_eq!(domain_call(&mut process, &mut dispatcher, clone, &close), 0);
    assert_eq!(
        domain_call(&mut process, &mut dispatcher, manager, &decode),
        HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
    );
    assert_eq!(
        domain_call(&mut process, &mut dispatcher, manager, &query),
        0
    );
}
