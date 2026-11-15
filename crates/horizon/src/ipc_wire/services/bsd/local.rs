//! BSD eventfd operations and passive waits; no network transport.
//! https://github.com/switchbrew/libnx/blob/master/nx/source/services/bsd.c

use super::*;
use crate::bsd::event_fd::{Access, Descriptors, INVAL};
use crate::ipc_wire::operation::ServiceEvent;
use std::{
    future::{Future, poll_fn},
    pin::Pin,
    task::Poll,
};

pub(super) async fn dispatch(
    process: &Request,
    session: &BsdSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    command: BsdCommand,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let descriptors = match session.system().descriptors(process.process_id()) {
        Ok(table) => table,
        Err(error) => return reply(session, request.token, Err(error)),
    };
    let scalar_size = match command {
        BsdCommand::EventFd => 16,
        BsdCommand::Fcntl => 12,
        BsdCommand::Poll => 8,
        _ => 4,
    };
    if !request.has_payload_size(scalar_size) {
        return invalid_header(session, request.token);
    }
    let fd = request_i32(request.data, 0).unwrap();
    let result = match command {
        BsdCommand::EventFd => {
            if has_ipc_descriptors(hipc) {
                return invalid_header(session, request.token);
            }
            let initial = request_u64(request.data, 8).unwrap();
            let result = descriptors.create(process.kernel(), initial, fd as u32);
            log::debug!("bsd:u EventFd: flags={fd:#x}, initial={initial}, result={result:?}");
            result
        }
        BsdCommand::Close => {
            if has_ipc_descriptors(hipc) {
                return invalid_header(session, request.token);
            }
            descriptors.close(fd)
        }
        BsdCommand::Fcntl => {
            if has_ipc_descriptors(hipc) {
                return invalid_header(session, request.token);
            }
            descriptors.get(fd).and_then(|event| {
                event.fcntl(
                    request_i32(request.data, 4).unwrap(),
                    request_u32(request.data, 8).unwrap(),
                )
            })
        }
        BsdCommand::Read | BsdCommand::Write => {
            let write = command == BsdCommand::Write;
            let buffer = if write {
                input(process, hipc)?
            } else {
                output(process, hipc)?
            };
            if buffer.size < 8 {
                return reply(session, request.token, Err(INVAL));
            }
            let event = match descriptors.get(fd) {
                Ok(event) => event,
                Err(error) => return reply(session, request.token, Err(error)),
            };
            process.reserve_payload(8).await?;
            let mut bytes = vec![0; 8];
            let value = if write {
                process.read(buffer, 0, &mut bytes).await?;
                Some(u64::from_le_bytes(bytes[..].try_into().unwrap()))
            } else {
                None
            };
            loop {
                match event.access(value) {
                    Ok(Access::Ready(value)) => {
                        if !write {
                            bytes.copy_from_slice(&value.to_le_bytes());
                            process.write(buffer, 0, &mut bytes).await?;
                        }
                        break Ok(8);
                    }
                    Ok(Access::Wait(changed)) => {
                        ServiceEvent::new(process.kernel(), changed, None, process.lifetime())
                            .await?;
                    }
                    Err(error) => break Err(error),
                }
            }
        }
        BsdCommand::Poll => {
            poll(
                process,
                &descriptors,
                hipc,
                fd,
                request_i32(request.data, 4).unwrap(),
            )
            .await?
        }
        _ => unreachable!(),
    };
    reply(session, request.token, result)
}

fn reply(
    session: &BsdSession,
    token: u32,
    result: Result<i32, u32>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let (value, errno) = match result {
        Ok(value) => (value, 0),
        Err(error) => (-1, error),
    };
    let mut data = [0; 8];
    data[..4].copy_from_slice(&value.to_le_bytes());
    data[4..].copy_from_slice(&errno.to_le_bytes());
    bsd_response(session, token, HorizonIpcResult::SUCCESS, &data)
}

// SDK revisions expose these buffers as map-alias or auto-select buffers.
fn input<'a>(process: &'a Request, hipc: &HipcRequest<'_>) -> Result<&'a Buffer, IpcWireError> {
    if hipc.send_statics.is_empty() {
        one_send_buffer(process, hipc)
    } else {
        one_auto_select_input(process, hipc)
    }
}
fn output<'a>(process: &'a Request, hipc: &HipcRequest<'_>) -> Result<&'a Buffer, IpcWireError> {
    if matches!(hipc.receive_statics, ReceiveStatics::None) {
        one_receive_buffer(process, hipc)
    } else {
        one_auto_select_output(process, hipc)
    }
}

async fn poll(
    process: &Request,
    descriptors: &Descriptors,
    hipc: &HipcRequest<'_>,
    count: i32,
    timeout: i32,
) -> Result<Result<i32, u32>, IpcWireError> {
    if count < 0 || timeout < -1 {
        return Ok(Err(INVAL));
    }
    let input = input(process, hipc)?;
    let output = output(process, hipc)?;
    let size = count as usize * 8;
    if input.size < size as u64 || output.size < size as u64 {
        return Ok(Err(INVAL));
    }
    process.reserve_payload(size).await?;
    let mut bytes = vec![0; size];
    process.read(input, 0, &mut bytes).await?;
    // Retain descriptor identities across waits. Closing and reusing an integer
    // slot must not redirect an outstanding poll to the replacement descriptor.
    let entries: Vec<_> = bytes
        .chunks_exact(8)
        .map(|entry| {
            let fd = i32::from_le_bytes(entry[..4].try_into().unwrap());
            (fd, descriptors.get(fd).ok())
        })
        .collect();
    let expires = (timeout >= 0).then(|| {
        process
            .kernel()
            .clock()
            .scheduler_time_ns()
            .saturating_add(timeout as u64 * 1_000_000)
    });
    loop {
        let mut ready = 0;
        let mut changed = Vec::new();
        for (entry, (fd, event)) in bytes.chunks_exact_mut(8).zip(&entries) {
            let events = u16::from_le_bytes(entry[4..6].try_into().unwrap());
            let (revents, wait) = if *fd < 0 {
                (0, None)
            } else if let Some(event) = event {
                event.poll(events)
            } else {
                (0x20, None)
            };
            entry[6..8].copy_from_slice(&revents.to_le_bytes());
            ready += i32::from(revents != 0);
            if let Some(wait) = wait {
                changed.push(wait.wait_async());
            }
        }
        if ready != 0
            || expires.is_some_and(|end| process.kernel().clock().scheduler_time_ns() >= end)
        {
            process.write(output, 0, &mut bytes).await?;
            return Ok(Ok(ready));
        }
        let mut deadline = expires.map(|end| process.kernel().deadline(end));
        poll_fn(|context| {
            process.lifetime().check()?;
            for event in &mut changed {
                if Pin::new(event).poll(context).is_ready() {
                    return Poll::Ready(Ok(()));
                }
            }
            if let Some(deadline) = &mut deadline {
                match Pin::new(deadline).poll(context) {
                    Poll::Ready(Ok(())) => return Poll::Ready(Ok(())),
                    Poll::Ready(Err(_)) => return Poll::Ready(Err(IpcWireError::Cancelled)),
                    Poll::Pending => {}
                }
            }
            Poll::Pending
        })
        .await?;
    }
}
