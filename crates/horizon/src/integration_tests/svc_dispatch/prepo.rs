use super::*;

fn report_request(
    command: u32,
    inline: &[u8],
    mapped: u64,
    pointer: u64,
    domain: bool,
) -> [u8; 256] {
    let mut bytes = [0xa5; 256];
    put_u32(&mut bytes, 0, 6 | (1 << 16) | (1 << 20));
    let words = (32 + inline.len() + usize::from(domain) * 16) / 4;
    put_u32(&mut bytes, 4, words as u32 | (1 << 31));
    put_u32(&mut bytes, 8, 1);
    put_u64(&mut bytes, 12, 1);
    put_send_static(&mut bytes, 20, pointer, 5);
    put_receive_buffer(&mut bytes, 28, mapped, 7);
    let cmif = if domain { 64 } else { 48 };
    if domain {
        bytes[48..64].fill(0);
        bytes[48] = 1;
        bytes[50..52].copy_from_slice(&((16 + inline.len()) as u16).to_le_bytes());
        put_u32(&mut bytes, 52, 1);
        put_u32(&mut bytes, 60, 0x1234);
    }
    put_u32(&mut bytes, cmif, 0x4943_4653);
    put_u32(&mut bytes, cmif + 4, 1);
    put_u32(&mut bytes, cmif + 8, command);
    put_u32(&mut bytes, cmif + 12, if domain { 0 } else { 0x1234 });
    bytes[cmif + 16..cmif + 16 + inline.len()].copy_from_slice(inline);
    bytes
}

#[test]
fn play_report_protocols_persist_exact_payloads_for_plain_and_domain_sessions() {
    for domain in [false, true] {
        let (_directory, mut process) = fixture_process_with_svcs(&[0x21; 12]);
        let mut dispatcher = support::dispatcher(&process);
        let directory = tempfile::tempdir().unwrap();
        let session = crate::PlayReportSession::new(directory.path());
        let handle = process
            .handles()
            .insert(HorizonIpcObject::PlayReport(session))
            .unwrap();
        let tls = process.main_thread().tls_base;
        if domain {
            let mut convert = audout::command(0, &[], None, None, false, false);
            put_u32(&mut convert, 0, 5);
            assert_eq!(
                audout::call(&mut process, &mut dispatcher, handle, &convert),
                0
            );
            assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);
        }
        let mapped = process.main_thread().stack_bottom;
        let pointer = mapped.checked_add(0x100).unwrap();
        write_guest_bytes(&process, mapped, b"payload");
        write_guest_bytes(&process, pointer, b"event");
        let mut expected = Vec::new();
        for command in 10100_u32..=10107 {
            let mut inline = vec![0; if command % 2 == 1 { 24 } else { 8 }];
            if inline.len() == 24 {
                inline[..16].copy_from_slice(&0x1234_u128.to_le_bytes());
            }
            let pid = inline.len() - 8;
            inline[pid..].copy_from_slice(&1_u64.to_le_bytes());
            let request = report_request(command, &inline, mapped.get(), pointer.get(), domain);
            write_guest_bytes(&process, tls, &request);
            state(&mut process).write_w(x(0), handle);
            assert!(matches!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            ));
            assert_eq!(
                read_guest_u32(
                    &process,
                    tls.checked_add(if domain { 40 } else { 24 }).unwrap()
                ),
                0
            );
            let mut bytes = b"NXPR0001".to_vec();
            assert_eq!(
                read_guest_u32(
                    &process,
                    tls.checked_add(if domain { 44 } else { 28 }).unwrap()
                ),
                0x1234
            );
            bytes.extend(command.to_le_bytes());
            bytes.extend(1_u64.to_le_bytes());
            for size in [inline.len() as u64, 7, 5] {
                bytes.extend(size.to_le_bytes());
            }
            bytes.extend(inline);
            bytes.extend(b"payload");
            bytes.extend(b"event");
            expected.push(bytes);
        }
        let mut paths: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|p| p.unwrap().path())
            .collect();
        paths.sort();
        assert_eq!(paths.len(), 8);
        for (path, expected) in paths.iter().zip(expected) {
            assert_eq!(fs::read(path).unwrap(), expected);
        }
    }
}

#[test]
fn play_report_session_id_is_stable_and_storage_failures_do_not_fabricate_success() {
    let first = crate::PlayReportSession::system_session_id().unwrap();
    assert_ne!(first, 0);
    assert_eq!(
        first,
        crate::PlayReportSession::system_session_id().unwrap()
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("not-a-directory");
    fs::write(&path, b"keep").unwrap();
    let session = crate::PlayReportSession::new(&path);
    assert!(
        session
            .save(10104, 1, &1_u64.to_le_bytes(), [b"event", b"data"])
            .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), b"keep");
}
