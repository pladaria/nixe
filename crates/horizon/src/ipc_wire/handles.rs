//! Kernel IPC header acceptance shared by built-in and guest servers.
use crate::{CURRENT_PROCESS_HANDLE, CURRENT_THREAD_HANDLE, HorizonKernelResult};
use nixe_runtime::{HandleObject, RunnableProcess, ThreadObject};
use std::sync::Arc;

pub(crate) struct MessageHandles {
    pub copied: Vec<Option<HandleObject>>,
    pub moved: Vec<Option<HandleObject>>,
}

// Client requests prohibit moves before processing any handle. Server replies
// consume all non-null move descriptors, including after a transfer error.
// https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_server_session.cpp
pub(crate) fn capture(
    process: &Arc<RunnableProcess>,
    thread: &Arc<ThreadObject>,
    copies: impl ExactSizeIterator<Item = u32>,
    moves: impl ExactSizeIterator<Item = u32>,
    allow_moves: bool,
) -> Result<MessageHandles, HorizonKernelResult> {
    if !allow_moves && moves.len() != 0 {
        return Err(HorizonKernelResult::INVALID_COMBINATION);
    }
    let mut error = None;
    let mut copied = Vec::with_capacity(copies.len());
    for handle in copies {
        let object = if error.is_some() {
            None
        } else {
            match handle {
                0 => None,
                CURRENT_PROCESS_HANDLE => Some(HandleObject::from_arc(process.clone())),
                CURRENT_THREAD_HANDLE => Some(HandleObject::from_arc(thread.clone())),
                _ => match process.handles().lookup(handle) {
                    Some(object) => Some(object),
                    None => {
                        error = Some(HorizonKernelResult::INVALID_HANDLE);
                        None
                    }
                },
            }
        };
        copied.push(object);
    }
    let mut moved = Vec::with_capacity(moves.len());
    for handle in moves {
        let object = match handle {
            0 => None,
            CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE => {
                error = Some(HorizonKernelResult::INVALID_HANDLE);
                None
            }
            _ => match process.handles().close(handle) {
                Ok(object) if error.is_none() => Some(object),
                Ok(_) => None,
                Err(_) => {
                    error = Some(HorizonKernelResult::INVALID_HANDLE);
                    None
                }
            },
        };
        moved.push(object);
    }
    match error {
        Some(error) => Err(error),
        None => Ok(MessageHandles { copied, moved }),
    }
}
