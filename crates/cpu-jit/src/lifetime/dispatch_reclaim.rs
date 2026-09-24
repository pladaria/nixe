//! Cold intrusive FIFO: retired dispatch owners are not reusable capacity until
//! their last unit/worker pin and reader epoch have gone. Never scan live slots.
use super::*;

#[derive(Default)]
pub(super) struct Queue {
    head: Option<Handle<DispatchSlot>>,
    tail: Option<Handle<DispatchSlot>>,
    pub(super) len: usize,
}

impl State {
    pub(super) fn retire_dispatch_slot(
        &mut self,
        handle: Handle<DispatchSlot>,
        epoch: ExecutionEpoch,
    ) {
        let slot = self.dispatch.get_mut(handle).unwrap();
        assert!(slot.retired.is_none());
        slot.retired = Some(epoch);
        self.enqueue_retired_dispatch(handle);
    }

    fn enqueue_retired_dispatch(&mut self, handle: Handle<DispatchSlot>) {
        debug_assert!(self.dispatch.get(handle).unwrap().retired_next.is_none());
        if let Some(tail) = self.retired_dispatch.tail {
            self.dispatch.get_mut(tail).unwrap().retired_next = Some(handle);
        } else {
            self.retired_dispatch.head = Some(handle);
        }
        self.retired_dispatch.tail = Some(handle);
        self.retired_dispatch.len += 1;
    }

    fn pop_retired_dispatch(&mut self) -> Option<Handle<DispatchSlot>> {
        let handle = self.retired_dispatch.head?;
        self.retired_dispatch.head = self.dispatch.get_mut(handle).unwrap().retired_next.take();
        self.retired_dispatch.len -= 1;
        if self.retired_dispatch.head.is_none() {
            self.retired_dispatch.tail = None;
        }
        Some(handle)
    }
}

impl Lifetime {
    pub(crate) fn collect_dispatch(&self) -> Result<usize, Error> {
        self.collect_retired_dispatch(usize::MAX)
    }

    pub(super) fn collect_retired_dispatch(&self, limit: usize) -> Result<usize, Error> {
        let visits = {
            let state = self.lock();
            state.healthy()?;
            state.retired_dispatch.len.min(limit)
        };
        let mut count = 0;
        for _ in 0..visits {
            let removed = {
                let mut state = self.lock();
                state.healthy()?;
                let Some(handle) = state.pop_retired_dispatch() else {
                    break;
                };
                let slot = state.dispatch.get(handle).unwrap();
                if slot.units != 0
                    || slot.optimization.pinned()
                    || slot.reshape.pinned()
                    || !state.quiescent(slot.retired.unwrap())
                {
                    state.enqueue_retired_dispatch(handle);
                    continue;
                }
                state.dispatch.remove(handle).unwrap()
            };
            // Payload/owner charges are released without holding JIT state.
            drop(removed);
            count += 1;
        }
        Ok(count)
    }
}
