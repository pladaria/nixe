//! Byte admission for the frontend/owner handoff. Control requests bypass it.

use std::sync::{Arc, Condvar, Mutex};

const MAX_QUEUED_COMMAND_BYTES: usize = 64 * 1024 * 1024;

pub(super) struct QueuedBytesBudget {
    limit: usize,
    state: Mutex<State>,
    available: Condvar,
}

#[derive(Default)]
struct State {
    bytes: usize,
    stopped: bool,
}

impl Default for QueuedBytesBudget {
    fn default() -> Self {
        Self::new(MAX_QUEUED_COMMAND_BYTES)
    }
}

impl std::fmt::Debug for QueuedBytesBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedBytesBudget")
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

impl QueuedBytesBudget {
    fn new(limit: usize) -> Self {
        assert!(limit > 0);
        Self {
            limit,
            state: Mutex::default(),
            available: Condvar::new(),
        }
    }

    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Result<QueuedBytesPermit, Box<str>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // An indivisible upload larger than the budget is admitted alone. This
        // bounds ordinary buffering without inventing a guest upload-size limit.
        while !state.stopped && state.bytes != 0 && bytes > self.limit.saturating_sub(state.bytes) {
            state = self
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        if state.stopped {
            return Err("GPU owner stopped during command-byte admission".into());
        }
        state.bytes = state
            .bytes
            .checked_add(bytes)
            .expect("bounded handoff byte count cannot overflow");
        Ok(QueuedBytesPermit {
            budget: Arc::clone(self),
            bytes,
        })
    }

    pub(super) fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped = true;
        self.available.notify_all();
    }
}

pub(super) struct QueuedBytesPermit {
    budget: Arc<QueuedBytesBudget>,
    bytes: usize,
}

impl Drop for QueuedBytesPermit {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.bytes -= self.bytes;
        self.budget.available.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread, time::Duration};

    #[test]
    fn full_byte_budget_blocks_producer_until_owner_releases_consumed_batch() {
        let budget = Arc::new(QueuedBytesBudget::new(8));
        let first = budget.reserve(8).unwrap();
        let next = Arc::clone(&budget);
        let (sent, received) = mpsc::channel();
        let worker = thread::spawn(move || sent.send(next.reserve(1)).unwrap());
        assert!(matches!(
            received.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(first);
        drop(
            received
                .recv_timeout(Duration::from_secs(3))
                .unwrap()
                .unwrap(),
        );
        worker.join().unwrap();
        assert_eq!(budget.state.lock().unwrap().bytes, 0);
    }

    #[test]
    fn shutdown_wakes_a_blocked_frontend_without_requiring_more_commands() {
        let budget = Arc::new(QueuedBytesBudget::new(8));
        let first = budget.reserve(8).unwrap();
        let next = Arc::clone(&budget);
        let worker = thread::spawn(move || next.reserve(1).is_err());
        budget.stop();
        assert!(worker.join().unwrap());
        drop(first);
        assert!(budget.reserve(1).is_err());
    }

    #[test]
    fn indivisible_oversize_payload_is_admitted_alone_and_charged_in_full() {
        let budget = Arc::new(QueuedBytesBudget::new(8));
        let payload = budget.reserve(16).unwrap();
        assert_eq!(budget.state.lock().unwrap().bytes, 16);
        let next = Arc::clone(&budget);
        let (sent, received) = mpsc::channel();
        let worker = thread::spawn(move || sent.send(next.reserve(1)).unwrap());
        assert!(matches!(
            received.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(payload);
        drop(
            received
                .recv_timeout(Duration::from_secs(3))
                .unwrap()
                .unwrap(),
        );
        worker.join().unwrap();
    }
}
