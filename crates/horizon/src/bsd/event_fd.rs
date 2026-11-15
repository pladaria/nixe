//! Local BSD synchronization descriptors. These do not open host sockets.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/sockets/bsd.cpp#L213-L304

use nixe_runtime::{EventObject, ReadableEventObject};
use nixe_scheduler::SchedulerState;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

pub(crate) const BADF: u32 = 9;
pub(crate) const AGAIN: u32 = 11;
pub(crate) const INVAL: u32 = 22;
const MFILE: u32 = 24;
const MAX_COUNTER: u64 = u64::MAX - 1;
const NONBLOCK: u32 = 0x800;

#[derive(Debug, Default)]
pub(crate) struct Descriptors(Mutex<BTreeMap<i32, Arc<EventFd>>>);

impl Descriptors {
    pub fn create(
        &self,
        kernel: &Arc<SchedulerState>,
        initial: u64,
        flags: u32,
    ) -> Result<i32, u32> {
        if initial > MAX_COUNTER || flags & !5 != 0 {
            return Err(INVAL);
        }
        let mut table = self.0.lock().unwrap();
        let fd = (0..128).find(|fd| !table.contains_key(fd)).ok_or(MFILE)?;
        table.insert(
            fd,
            Arc::new(EventFd {
                kernel: kernel.clone(),
                state: Mutex::new(State {
                    counter: initial,
                    semaphore: flags & 1 != 0,
                    flags: if flags & 4 != 0 { NONBLOCK } else { 0 },
                    closed: false,
                    changed: None,
                }),
            }),
        );
        Ok(fd)
    }

    pub fn get(&self, fd: i32) -> Result<Arc<EventFd>, u32> {
        self.0.lock().unwrap().get(&fd).cloned().ok_or(BADF)
    }

    pub fn close(&self, fd: i32) -> Result<i32, u32> {
        let event = self.0.lock().unwrap().remove(&fd).ok_or(BADF)?;
        let changed = {
            let mut state = event.state.lock().unwrap();
            state.closed = true;
            state.changed.take()
        };
        if let Some(changed) = changed {
            changed.signal();
        }
        Ok(0)
    }
}

#[derive(Debug)]
pub(crate) struct EventFd {
    kernel: Arc<SchedulerState>,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    counter: u64,
    semaphore: bool,
    flags: u32,
    closed: bool,
    // A shared one-shot notification is allocated only when a call must wait.
    // Taking it under the counter lock prevents lost wakeups and lets all
    // observers recheck after a change, including a descriptor close.
    changed: Option<EventObject>,
}

pub(crate) enum Access {
    Ready(u64),
    Wait(ReadableEventObject),
}

impl EventFd {
    pub fn access(&self, write: Option<u64>) -> Result<Access, u32> {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(BADF);
        }
        if write == Some(u64::MAX) {
            return Err(INVAL);
        }
        let blocked = match write {
            Some(value) => value > MAX_COUNTER - state.counter,
            None => state.counter == 0,
        };
        if blocked {
            return if state.flags & NONBLOCK != 0 {
                Err(AGAIN)
            } else {
                Ok(Access::Wait(
                    state
                        .changed
                        .get_or_insert_with(|| EventObject::new(&self.kernel))
                        .readable(),
                ))
            };
        }
        let value = if let Some(value) = write {
            state.counter += value;
            8
        } else {
            let value = if state.semaphore { 1 } else { state.counter };
            state.counter -= value;
            value
        };
        let changed = state.changed.take();
        drop(state);
        if let Some(changed) = changed {
            changed.signal();
        }
        Ok(Access::Ready(value))
    }

    pub fn poll(&self, events: u16) -> (u16, Option<ReadableEventObject>) {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return (0x20, None);
        }
        let ready = (if state.counter != 0 { events & 0x41 } else { 0 })
            | (if state.counter < MAX_COUNTER {
                events & 4
            } else {
                0
            });
        let changed = (ready == 0).then(|| {
            state
                .changed
                .get_or_insert_with(|| EventObject::new(&self.kernel))
                .readable()
        });
        (ready, changed)
    }

    pub fn fcntl(&self, command: i32, flags: u32) -> Result<i32, u32> {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(BADF);
        }
        match command {
            3 => Ok(state.flags as i32),
            4 => {
                state.flags = flags & NONBLOCK;
                let changed = state.changed.take();
                drop(state);
                if let Some(changed) = changed {
                    changed.signal();
                }
                Ok(0)
            }
            _ => Err(INVAL),
        }
    }
}
