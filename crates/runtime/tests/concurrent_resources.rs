use std::thread;

use nixe_runtime::{
    EventObject, HandleTable, PortError, PortObject, SessionError, SessionMessage, SessionObject,
    SessionRequestOwner,
};

#[test]
fn concurrent_queries_cannot_resolve_a_closed_handle_to_its_replacement() {
    use std::sync::{Arc, Barrier};
    let handles = Arc::new(HandleTable::with_capacity_limit(1));
    let first = handles.insert(EventObject::new(&kernel())).unwrap();
    let retained = handles.lookup(first).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        let table = handles.clone();
        let worker_barrier = barrier.clone();
        let original = retained.clone();
        scope.spawn(move || {
            worker_barrier.wait();
            for _ in 0..4096 {
                if let Some(object) = table.lookup(first) {
                    assert!(object.same_identity(&original));
                }
            }
        });
        barrier.wait();
        drop(handles.close(first).unwrap());
        let replacement = handles.insert(EventObject::new(&kernel())).unwrap();
        assert_ne!(first, replacement);
        handles
            .lookup_as::<EventObject>(replacement)
            .unwrap()
            .signal();
        assert!(handles.lookup(first).is_none());
    });
    assert!(!retained.downcast::<EventObject>().unwrap().is_signalled());
}

#[test]
fn concurrent_port_connections_respect_the_atomic_session_limit() {
    let (server, client) = PortObject::create_pair(&kernel(), 4, false);
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
    let (server, client) = SessionObject::create_pair(&kernel());
    let server_guard = server.clone();
    let server_worker = thread::spawn(move || {
        for _ in 0..CLIENTS {
            let mut receive = loop {
                match server.receive() {
                    Ok(message) => break message,
                    Err(SessionError::NoRequest) => thread::yield_now(),
                    Err(error) => panic!("unexpected receive error: {error:?}"),
                }
            };
            let message = receive.take_message();
            receive.complete();
            server.reply().unwrap().reply(message).unwrap();
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
                client
                    .request(owner, SessionMessage::Buffer(expected.clone()))
                    .unwrap();
                loop {
                    match poll_session(&client, owner).unwrap() {
                        Some(SessionMessage::Buffer(bytes)) => {
                            assert_eq!(bytes, expected);
                            break;
                        }
                        None => thread::yield_now(),
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

#[test]
fn concurrent_session_admission_keeps_one_request_and_does_not_consume_its_reply() {
    use std::sync::Arc;

    const CLIENTS: usize = 8;
    let (server, client) = SessionObject::create_pair(&kernel());
    let owner = SessionRequestOwner {
        process_id: 1,
        thread_id: 1,
    };
    let barrier = Arc::new(std::sync::Barrier::new(CLIENTS));
    let clients: Vec<_> = (0..CLIENTS)
        .map(|index| {
            let client = client.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                (
                    index,
                    client
                        .request(owner, SessionMessage::Buffer(vec![index as u8]))
                        .map(|_| ()),
                )
            })
        })
        .collect();
    let mut winner = None;
    for client in clients {
        let (index, result) = client.join().unwrap();
        match result {
            Ok(()) => assert!(winner.replace(index).is_none(), "one accepted caller"),
            Err(error) => assert_eq!(error, SessionError::ReplyPending),
        }
    }
    let mut receive = server.receive().unwrap();
    assert_eq!(
        receive.take_message(),
        SessionMessage::Buffer(vec![winner.unwrap() as u8])
    );
    receive.complete();
    server
        .reply()
        .unwrap()
        .reply(SessionMessage::Buffer(vec![0xa5]))
        .unwrap();
    assert_eq!(
        client.request(owner, SessionMessage::Buffer(vec![0x5a])),
        Err(SessionError::ReplyPending)
    );
    assert_eq!(
        poll_session(&client, owner),
        Ok(Some(SessionMessage::Buffer(vec![0xa5])))
    );
    assert_eq!(server.receive().unwrap_err(), SessionError::NoRequest);
    client
        .request(owner, SessionMessage::Buffer(vec![0x5a]))
        .unwrap();
    let mut receive = server.receive().unwrap();
    assert_eq!(receive.take_message(), SessionMessage::Buffer(vec![0x5a]));
    receive.complete();
}

#[test]
fn batch_lookup_retains_endpoint_lifetime_and_preserves_duplicate_identity() {
    let table = HandleTable::new();
    let (server, client) = SessionObject::create_pair(&kernel());
    let handle = table.insert(server).unwrap();
    let objects = table
        .lookup_many(&[handle, handle], |object| {
            object.downcast::<SessionObject>()
        })
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&objects[0], &objects[1]));
    table.close_all();
    assert_eq!(
        client.request(
            SessionRequestOwner {
                process_id: 1,
                thread_id: 1
            },
            SessionMessage::Buffer(vec![1])
        ),
        Ok(())
    );
    let mut receive = objects[0].receive().unwrap();
    assert_eq!(receive.take_message(), SessionMessage::Buffer(vec![1]));
    receive.complete();
    drop(objects);
    assert_eq!(
        client.request(
            SessionRequestOwner {
                process_id: 1,
                thread_id: 2
            },
            SessionMessage::Buffer(vec![2])
        ),
        Err(SessionError::PeerClosed)
    );
}

#[test]
fn batch_lookup_cannot_accept_handles_that_never_coexist() {
    use std::sync::{Arc, Barrier};
    let table = Arc::new(HandleTable::new());
    let first = table.insert(EventObject::new(&kernel())).unwrap();
    let reserved = table
        .reserve(Some(nixe_runtime::HandleObject::new(EventObject::new(
            &kernel(),
        ))))
        .unwrap();
    let second = reserved.handle();
    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        let reader = table.clone();
        let start = barrier.clone();
        scope.spawn(move || {
            start.wait();
            for _ in 0..4096 {
                assert!(
                    reader
                        .lookup_many(&[first, second], |object| object.downcast::<EventObject>())
                        .is_err()
                );
            }
        });
        barrier.wait();
        drop(table.close(first).unwrap());
        table.publish(&mut [reserved]).unwrap();
    });
}

fn kernel() -> std::sync::Arc<nixe_scheduler::SchedulerState> {
    use nixe_scheduler::{
        MachineSchedulerProfile, PriorityRange, SchedulerState, VirtualClock, VirtualClockMode,
        VirtualCpuDescriptor, VirtualCpuId,
    };
    SchedulerState::new(
        MachineSchedulerProfile::new(
            vec![VirtualCpuDescriptor::new(VirtualCpuId::new(0), 0)],
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        VirtualClock::new(VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}

fn poll_session(
    session: &nixe_runtime::SessionObject,
    owner: nixe_runtime::SessionRequestOwner,
) -> Result<Option<nixe_runtime::SessionMessage>, nixe_runtime::SessionError> {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };
    match Pin::new(&mut session.response(owner)).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Pending => Ok(None),
        Poll::Ready(result) => result.map(Some),
    }
}
