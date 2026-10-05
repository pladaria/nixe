use super::*;

#[test]
fn user_registration_is_unavailable_and_validates_the_pid_placeholder() {
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
    put_u32(&mut request, 40, 50);
    for invalid in [false, true] {
        put_u32(&mut request, 48, u32::from(invalid));
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(24).unwrap()),
            if invalid {
                HorizonIpcResult::CMIF_INVALID_IN_HEADER.raw()
            } else {
                0
            }
        );
        if !invalid {
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(32).unwrap()) & 0xff,
                0
            );
        }
    }
}

#[test]
fn last_opened_user_returns_the_application_launch_uid() {
    let (_directory, mut process) = fixture_process(&[svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::Account(Default::default()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let mut request = [0; 0x100];
    put_u32(&mut request, 0, 4);
    put_u32(&mut request, 4, 8);
    put_u32(&mut request, 16, 0x4943_4653);
    put_u32(&mut request, 24, 4);
    write_guest_bytes(&process, tls, &request);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    for offset in [0, 4, 8, 12] {
        assert_eq!(
            read_guest_u32(&process, tls.checked_add(32 + offset).unwrap()),
            u32::from(offset == 0)
        );
    }
}

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

#[test]
fn account_pointer_budget_and_enumeration_agree_with_the_opened_profile() {
    let (_directory, mut process) = fixture_process(&[svc(0x21); 5]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::Account(Default::default()))
        .unwrap();
    let tls = process.main_thread().tls_base;
    let destination = process
        .main_thread()
        .stack_bottom
        .checked_add(0x100)
        .unwrap();
    let mut query = [0; 0x100];
    put_u32(&mut query, 0, 5);
    put_u32(&mut query, 4, 8);
    put_u32(&mut query, 16, 0x4943_4653);
    put_u32(&mut query, 24, 3);
    write_guest_bytes(&process, tls, &query);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert!(read_guest_u32(&process, tls.checked_add(32).unwrap()) >= 128);
    for command in [2, 3] {
        let mut list = [0; 0x100];
        put_u32(&mut list, 0, 4);
        put_u32(&mut list, 4, 8 | (3 << 10));
        put_u32(&mut list, 16, 0x4943_4653);
        put_u32(&mut list, 24, command);
        put_u64(&mut list, 40, destination.get() | (128_u64 << 48));
        write_guest_bytes(&process, destination, &[0xff; 128]);
        write_guest_bytes(&process, tls, &list);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
        let users = read_guest_bytes(&process, destination, 128);
        assert_eq!(&users[..16], &1_u128.to_le_bytes());
        assert_eq!(&users[16..], &[0; 112]);
    }
    let mut open = [0; 0x100];
    put_u32(&mut open, 0, 4);
    put_u32(&mut open, 4, 12);
    put_u32(&mut open, 16, 0x4943_4653);
    put_u32(&mut open, 24, 5);
    open[32..48].copy_from_slice(&1_u128.to_le_bytes());
    write_guest_bytes(&process, tls, &open);
    state(&mut process).write_w(x(0), handle);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let profile = read_guest_u32(&process, tls.checked_add(12).unwrap());
    assert!(matches!(
        process.handles().get_as::<HorizonIpcObject>(profile),
        Some(HorizonIpcObject::AccountProfile(_))
    ));
    let mut get = [0; 0x100];
    put_u32(&mut get, 0, 4);
    put_u32(&mut get, 4, 8);
    put_u32(&mut get, 16, 0x4943_4653);
    put_u32(&mut get, 24, 1);
    write_guest_bytes(&process, tls, &get);
    state(&mut process).write_w(x(0), profile);
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    assert_eq!(read_guest_u32(&process, tls.checked_add(24).unwrap()), 0);
    let base = read_guest_bytes(&process, tls.checked_add(32).unwrap(), 56);
    assert_eq!(&base[..16], &1_u128.to_le_bytes());
    assert_eq!(&base[24..29], b"Nixe\0");
}
