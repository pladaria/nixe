use std::thread;

use nixe_runtime::{
    EventObject, HandleError, HandleTable, PortError, PortObject, SessionError, SessionMessage,
    SessionObject, SessionRequestOwner, SessionRequestResult, ThreadObject,
};

#[test]
fn failed_cross_process_handle_transfer_keeps_source_ownership() {
    let mut source = HandleTable::new();
    let source_handle = source.insert(EventObject::new()).unwrap();
    let mut destination = HandleTable::with_capacity_limit(1);
    destination.insert(ThreadObject::new(2)).unwrap();

    assert_eq!(
        source.transfer_to(&mut destination, source_handle),
        Err(HandleError::Exhausted)
    );
    assert!(source.get(source_handle).is_some());
    assert_eq!(source.len(), 1);
    assert_eq!(destination.len(), 1);
}

#[test]
fn concurrent_port_connections_respect_the_atomic_session_limit() {
    let (server, client) = PortObject::create_pair(4, false);
    let workers: Vec<_> = (0..16)
        .map(|_| {
            let client = client.clone();
            thread::spawn(move || client.connect())
        })
        .collect();
    let mut connected = Vec::new();
    let mut rejected = 0;
    for worker in workers {
        match worker.join().unwrap() {
            Ok(session) => connected.push(session),
            Err(PortError::SessionLimit) => rejected += 1,
            Err(error) => panic!("unexpected connection error: {error:?}"),
        }
    }
    assert_eq!(connected.len(), 4);
    assert_eq!(rejected, 12);
    for _ in 0..4 {
        server.accept().unwrap();
    }
    assert!(matches!(server.accept(), Err(PortError::NoPendingSession)));
}

#[test]
fn concurrent_session_clients_deliver_and_receive_exactly_once() {
    const CLIENTS: u64 = 16;
    let (server, client) = SessionObject::create_pair();
    let server_guard = server.clone();
    let server_worker = thread::spawn(move || {
        for _ in 0..CLIENTS {
            let message = loop {
                match server.receive() {
                    Ok(message) => break message,
                    Err(SessionError::NoRequest) => thread::yield_now(),
                    Err(error) => panic!("unexpected receive error: {error:?}"),
                }
            };
            server.reply(message).unwrap();
        }
    });

    let clients: Vec<_> = (0..CLIENTS)
        .map(|thread_id| {
            let client = client.clone();
            thread::spawn(move || {
                let owner = SessionRequestOwner {
                    process_id: 1,
                    thread_id,
                };
                let expected = vec![thread_id as u8];
                assert_eq!(
                    client
                        .request(owner, SessionMessage::Buffer(expected.clone()))
                        .unwrap(),
                    SessionRequestResult::Submitted
                );
                loop {
                    match client.poll_request(owner).unwrap() {
                        Some(SessionRequestResult::Response(SessionMessage::Buffer(bytes))) => {
                            assert_eq!(bytes, expected);
                            break;
                        }
                        Some(SessionRequestResult::Waiting) | None => thread::yield_now(),
                        other => panic!("unexpected request result: {other:?}"),
                    }
                }
            })
        })
        .collect();
    for client in clients {
        client.join().unwrap();
    }
    server_worker.join().unwrap();
    drop(server_guard);
}
