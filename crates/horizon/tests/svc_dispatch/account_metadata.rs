use super::*;

#[test]
fn user_existence_distinguishes_the_local_uid_from_absent_users() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 3]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::Account(Default::default()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    for uid in [1_u128, 2, 0] {
        let mut request = [0; 0x100];
        put_u32(&mut request, 0, 4);
        put_u32(&mut request, 4, 12);
        put_u32(&mut request, 16, 0x4943_4653);
        put_u32(&mut request, 24, 1);
        request[32..48].copy_from_slice(&uid.to_le_bytes());
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(32).unwrap()),
            u32::from(uid == 1)
        );
    }
}

#[test]
fn account_switch_lock_uses_application_policy_instead_of_user_count() {
    for locked in [false, true] {
        let (_directory, mut process) = fixture_process(&[svc(0x21); 2]);
        let mut dispatcher = HorizonSvcDispatcher::default().with_user_account_switch_lock(locked);
        let handle = process
            .handles_mut()
            .insert(HorizonIpcObject::Account(Default::default()))
            .unwrap();
        let tls = process.main_thread().tls_base;
        let mut request = [0; 0x100];
        put_u32(&mut request, 0, 4);
        put_u32(&mut request, 4, 8);
        put_u32(&mut request, 16, 0x4943_4653);
        put_u32(&mut request, 24, 150);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(32).unwrap()),
            u32::from(locked)
        );

        put_u32(&mut request, 32, 1);
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    }
}

#[test]
fn application_info_v2_requires_a_pid_and_zero_placeholder() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 2]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::Account(Default::default()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 10 | (1 << 31));
    put_u32(&mut request, 8, 1);
    put_u32(&mut request, 32, 0x4943_4653);
    put_u32(&mut request, 40, 160);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);

    put_u32(&mut request, 48, 1);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(
        read_guest_u32(&process, tls.checked_add(24).unwrap()),
        HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
    );
}
