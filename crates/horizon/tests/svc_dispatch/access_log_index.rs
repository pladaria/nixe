use super::*;

#[test]
fn filesystem_access_log_returns_version_then_program_index_in_both_transports() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 3]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 8);
    put_u32(&mut request, 16, 0x4943_4653);
    put_u32(&mut request, 24, 1011);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 2);
    assert_eq!(read_guest_u32(&process, tls.checked_add(36).unwrap()), 0);

    put_u32(&mut request, 0, 5);
    put_u32(&mut request, 24, 0);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(32).unwrap()), 1);

    request.fill(0);
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 12);
    request[16] = 1;
    request[18..20].copy_from_slice(&16_u16.to_le_bytes());
    put_u32(&mut request, 20, 1);
    put_u32(&mut request, 32, 0x4943_4653);
    put_u32(&mut request, 40, 1011);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(40).unwrap()), 0);
    assert_eq!(read_guest_u32(&process, tls.checked_add(48).unwrap()), 2);
    assert_eq!(read_guest_u32(&process, tls.checked_add(52).unwrap()), 0);
}

#[test]
fn filesystem_session_advertises_space_for_two_aligned_pointer_paths() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0; 0x100];
    put_u32(&mut request, 0, 5);
    put_u32(&mut request, 4, 8);
    put_u32(&mut request, 16, 0x4943_4653);
    put_u32(&mut request, 24, 3);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(32).unwrap()),
        0x800
    );
}
