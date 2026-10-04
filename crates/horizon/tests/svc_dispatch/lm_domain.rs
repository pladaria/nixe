use super::*;

fn domain_request(id: u32, pid: bool) -> [u8; 0x100] {
    let mut bytes = [0; 0x100];
    put_u32(&mut bytes, 0, 4);
    put_u32(&mut bytes, 4, 14 | (u32::from(pid) << 31));
    let start = if pid {
        put_u32(&mut bytes, 8, 1);
        32
    } else {
        16
    };
    bytes[start] = 1;
    bytes[start + 2..start + 4].copy_from_slice(&24_u16.to_le_bytes());
    put_u32(&mut bytes, start + 4, id);
    put_u32(&mut bytes, start + 16, 0x4943_4653);
    bytes
}

#[test]
fn logger_domain_shares_child_objects_with_clones_and_closes_them_independently() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 7]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::LogManager(Default::default()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let mut control = [0; 0x100];
    put_u32(&mut control, 0, 7);
    put_u32(&mut control, 4, 8);
    put_u32(&mut control, 16, 0x4943_4653);
    put_u32(&mut control, 20, 1);
    write_guest_bytes(&process, tls, &control);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    write_guest_bytes(&process, tls, &domain_request(1, false));
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );

    write_guest_bytes(&process, tls, &domain_request(1, true));
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(16).unwrap()), 1);
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    let logger = read_guest_u32(&process, tls.checked_add(48).unwrap());
    assert_eq!(logger, 2);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(4).unwrap()) >> 31,
        0
    );

    put_u32(&mut control, 24, 2);
    write_guest_bytes(&process, tls, &control);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let clone = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert_ne!(clone, handle);

    let address = process
        .main_thread()
        .stack_bottom
        .checked_add(0x400)
        .unwrap();
    let mut packet = [0; 24];
    packet[16] = 3;
    write_guest_bytes(&process, address, &packet);
    let mut log = [0; 0x100];
    put_u32(&mut log, 0, 4 | (1 << 16) | (1 << 20));
    put_u32(&mut log, 4, 12);
    put_send_static(&mut log, 8, address.get(), 24);
    log[32] = 1;
    log[34..36].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut log, 36, logger);
    put_u32(&mut log, 48, 0x4943_4653);
    write_guest_bytes(&process, tls, &log);
    state(&mut process).write_w(x(0), clone);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(
        read_guest_u32(&process, address),
        process.process_id() as u32
    );

    let mut close = [0; 0x100];
    put_u32(&mut close, 0, 4);
    put_u32(&mut close, 4, 8);
    close[16] = 2;
    put_u32(&mut close, 20, logger);
    write_guest_bytes(&process, tls, &close);
    state(&mut process).write_w(x(0), clone);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    write_guest_bytes(&process, tls, &log);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(40).unwrap()),
        HorizonIpcResult::CMIF_TARGET_NOT_FOUND.raw()
    );
    assert!(process.handles().get(handle).is_some());
    assert!(process.handles().get(clone).is_some());
}
