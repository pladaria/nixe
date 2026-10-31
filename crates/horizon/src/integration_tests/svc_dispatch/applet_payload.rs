use super::*;

fn command(accessor: u32, write: bool, address: u64, size: u64, offset: u64) -> [u8; 256] {
    let mut bytes = [0; 256];
    if write {
        put_u32(&mut bytes, 0, 4 | (1 << 16) | (1 << 20));
        put_u32(&mut bytes, 4, 14);
        put_receive_buffer(&mut bytes, 16, address, size);
    } else {
        put_u32(&mut bytes, 0, 4 | (1 << 24));
        put_u32(&mut bytes, 4, 16 | (3 << 10));
        put_receive_buffer(&mut bytes, 8, address, size);
    }
    bytes[32] = 1;
    bytes[34..36].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut bytes, 36, accessor);
    put_u32(&mut bytes, 48, 0x4943_4653);
    put_u32(&mut bytes, 56, if write { 10 } else { 11 });
    put_u64(&mut bytes, 64, offset);
    bytes
}

#[test]
fn applet_storage_streams_larger_than_the_capture_quota_and_rejects_a_whole_invalid_range() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 3]);
    let dispatcher = support::dispatcher(&process);
    let size = 66 * 1024 * 1024;
    let buffer = support::ready(process.memory().resize_heap(size)).unwrap();
    let session = crate::AppletSession::new(process.scheduler(), OperationMode::Console);
    session.convert_to_domain();
    let object = session.create_storage(size).unwrap();
    let crate::object::AppletObject::Storage(backing) = session.object(object).unwrap() else {
        panic!("storage");
    };
    let accessor = session.open_storage_accessor(backing.clone()).unwrap();
    let retained = backing;
    let handle = process
        .handles()
        .insert(crate::ServiceSession::new(HorizonIpcObject::Applet(
            session,
        )))
        .unwrap();
    let markers = [0, 1024 * 1024 - 4, 1024 * 1024 + 4, size - 8];
    for offset in markers {
        write_guest_bytes(&process, buffer.checked_add(offset).unwrap(), b"test");
    }
    for (write, offset) in [(true, 0), (false, 0), (true, size - 4)] {
        let tls = process.main_thread().tls_base;
        write_guest_bytes(
            &process,
            tls,
            &command(accessor, write, buffer.get(), size, offset),
        );
        write_abi_register(&process, 0, u64::from(handle));
        assert!(matches!(
            dispatch_next(&mut process, &dispatcher),
            ExceptionHandlingResult::Resumed
        ));
        let result = read_guest_u32(&process, tls.checked_add(40).unwrap());
        if offset == 0 {
            assert_eq!(result, 0);
            for offset in markers {
                let mut data = vec![0; 4];
                support::ready(retained.read_into(offset, &mut data)).unwrap();
                assert_eq!(&data, b"test");
                if write {
                    write_guest_bytes(&process, buffer.checked_add(offset).unwrap(), &[0; 4]);
                } else {
                    assert_eq!(
                        read_guest_bytes(&process, buffer.checked_add(offset).unwrap(), 4),
                        b"test"
                    );
                }
            }
        } else {
            assert_eq!(result, (503 << 9) | 128);
            let mut data = vec![0; 4];
            support::ready(retained.read_into(size - 4, &mut data)).unwrap();
            assert_eq!(data, [0; 4]);
        }
    }
}
