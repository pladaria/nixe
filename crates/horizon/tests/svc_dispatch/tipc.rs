use super::*;

fn sm_request(tipc: bool, command: u32, pid: bool, data: &[u8]) -> [u8; 0x100] {
    let mut bytes = [0; 0x100];
    put_u32(&mut bytes, 0, if tipc { command + 16 } else { 4 });
    if pid {
        put_u32(&mut bytes, 8, 1);
    }
    let data_offset = if pid { 20 } else { 8 };
    if tipc {
        put_u32(
            &mut bytes,
            4,
            (data.len().div_ceil(4) as u32) | (u32::from(pid) << 31),
        );
        bytes[data_offset..data_offset + data.len()].copy_from_slice(data);
    } else {
        let cmif_offset = (data_offset + 15) & !15;
        put_u32(
            &mut bytes,
            4,
            ((32 + data.len()).div_ceil(4) as u32) | (u32::from(pid) << 31),
        );
        put_u32(&mut bytes, cmif_offset, 0x4943_4653);
        put_u32(&mut bytes, cmif_offset + 8, command);
        put_u32(&mut bytes, cmif_offset + 12, 0x3456);
        bytes[cmif_offset + 16..cmif_offset + 16 + data.len()].copy_from_slice(data);
    }
    bytes
}

#[test]
fn sm_registration_and_service_handles_work_across_tipc_and_cmif() {
    for tipc_register in [false, true] {
        for tipc_get in [false, true] {
            let (_directory, mut process) = fixture_process(&[
                svc(0x1f),
                svc(0x21),
                svc(0x21),
                svc(0x21),
                svc(0x21),
                svc(0x16),
            ]);
            let mut dispatcher = HorizonSvcDispatcher::default();
            let name = process.main_thread().stack_bottom;
            write_guest_bytes(&process, name, b"sm:\0");
            state(&mut process).write_x(x(1), name.get());
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            let sm_handle = state(&mut process).read_w(x(1));
            let tls = process.main_thread().tls_base;
            let get = sm_request(tipc_get, 1, false, b"set:sys\0");

            write_guest_bytes(&process, tls, &get);
            state(&mut process).write_w(x(0), sm_handle);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            let result_offset = if tipc_get { 16 } else { 24 };
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(result_offset).unwrap()),
                (2 << 9) | 21
            );

            let register = sm_request(
                tipc_register,
                0,
                true,
                if tipc_register { &[] } else { &[0; 8] },
            );
            write_guest_bytes(&process, tls, &register);
            state(&mut process).write_w(x(0), sm_handle);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(state(&mut process).read_w(x(0)), 0);
            assert_eq!(
                read_guest_u32(
                    &process,
                    tls.checked_add(if tipc_register { 8 } else { 24 }).unwrap()
                ),
                0
            );

            write_guest_bytes(&process, tls, &get);
            state(&mut process).write_w(x(0), sm_handle);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(
                read_guest_u32(&process, tls.checked_add(8).unwrap()),
                1 << 5
            );
            let service = read_guest_u32(&process, tls.checked_add(12).unwrap());
            assert!(matches!(
                process.handles().get_as::<HorizonIpcObject>(service),
                Some(HorizonIpcObject::SystemSettings(_))
            ));
            assert_eq!(
                read_guest_u32(
                    &process,
                    tls.checked_add(if tipc_get { 16 } else { 24 }).unwrap()
                ),
                0
            );
            if !tipc_get {
                assert_eq!(
                    read_guest_u32(&process, tls.checked_add(16).unwrap()),
                    0x4f43_4653
                );
                assert_eq!(
                    read_guest_u32(&process, tls.checked_add(28).unwrap()),
                    0x3456
                );
            }

            let mut close = [0; 0x100];
            put_u32(&mut close, 0, 15);
            write_guest_bytes(&process, tls, &close);
            state(&mut process).write_w(x(0), sm_handle);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(state(&mut process).read_w(x(0)), 0);
            assert!(process.handles().get(sm_handle).is_some());
            state(&mut process).write_w(x(0), sm_handle);
            assert_eq!(
                dispatch_next(&mut process, &mut dispatcher),
                ExceptionHandlingResult::Resumed
            );
            assert_eq!(state(&mut process).read_w(x(0)), 0);
            assert!(process.handles().get(sm_handle).is_none());
            assert!(process.handles().get(service).is_some());
        }
    }
}

#[test]
fn tipc_register_client_requires_a_sent_pid() {
    let (_directory, mut process) = fixture_process(&[svc(0x1f), svc(0x21), svc(0x21)]);
    let mut dispatcher = HorizonSvcDispatcher::default();
    let name = process.main_thread().stack_bottom;
    write_guest_bytes(&process, name, b"sm:\0");
    state(&mut process).write_x(x(1), name.get());
    assert_eq!(
        dispatch_next(&mut process, &mut dispatcher),
        ExceptionHandlingResult::Resumed
    );
    let handle = state(&mut process).read_w(x(1));
    let tls = process.main_thread().tls_base;
    for request in [
        sm_request(true, 0, false, &[]),
        sm_request(true, 1, false, b"set:sys\0"),
    ] {
        write_guest_bytes(&process, tls, &request);
        state(&mut process).write_w(x(0), handle);
        assert_eq!(
            dispatch_next(&mut process, &mut dispatcher),
            ExceptionHandlingResult::Resumed
        );
        assert_eq!(state(&mut process).read_w(x(0)), 0);
        assert_eq!(
            read_guest_u32(
                &process,
                tls.checked_add(if request[0] == 17 { 16 } else { 8 })
                    .unwrap()
            ),
            (2 << 9) | 21
        );
    }
}
