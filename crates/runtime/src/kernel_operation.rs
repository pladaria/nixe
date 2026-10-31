//! Owned kernel continuations delivered to the physical core that accepted them.
use crate::{ExceptionDispatchContext, ExceptionDispatchOutcome, ThreadObject};
use nixe_cpu::{execution::CpuControl, location::LocationDescriptor};
use nixe_scheduler::WakeToken;
use std::{
    collections::VecDeque,
    error::Error,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    task::{Poll, Wake},
};

pub type DispatchFault = Box<dyn Error + Send + Sync>;
pub enum WaitCompletion {
    Cancelled,
    Return { x0: u64, x1: Option<u64> },
    Resume,
}

/// A blocked call owns its accepted effects until completion or cancellation.
pub trait ThreadWait: Send + Sync {
    fn cancel(self: Arc<Self>);
    fn caller(&self) -> &Arc<ThreadObject>;
    fn source(&self) -> LocationDescriptor;
    fn token(&self) -> Option<WakeToken>;
    /// Report completion once. Duplicate notifications of a ready continuation
    /// stay pending until its owner resumes or cancels it.
    fn poll(self: Arc<Self>) -> Poll<Result<WaitCompletion, DispatchFault>>;
    fn resume(
        self: Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<DispatchFault>;
}

#[derive(Default)]
#[repr(align(64))]
pub(crate) struct CoreSignal {
    sequence: AtomicU64,
    completions: AtomicU64,
    state: Mutex<SignalState>,
    changed: Condvar,
}
#[derive(Default)]
struct SignalState {
    native: Option<CpuControl>,
    quiescent: u64,
    finished: bool,
    retire: VecDeque<nixe_scheduler::ProcessId>,
}
impl CoreSignal {
    // Only the owning physical core writes this counter.
    pub(crate) fn record_completion(&self, completed: u64) {
        self.completions.store(completed, Ordering::Relaxed);
    }
    pub(crate) fn completions(&self) -> u64 {
        self.completions.load(Ordering::Relaxed)
    }
    pub(crate) fn take_control(&self) -> (u64, VecDeque<nixe_scheduler::ProcessId>) {
        let mut state = self.state.lock().unwrap();
        (self.sequence(), std::mem::take(&mut state.retire))
    }
    pub(crate) fn retire_process(&self, process: nixe_scheduler::ProcessId) {
        self.state.lock().unwrap().retire.push_back(process);
        self.notify();
    }
    pub(crate) fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }
    pub(crate) fn bind_native(&self, control: Option<CpuControl>) {
        self.state.lock().unwrap().native = control;
    }
    pub(crate) fn wait(&self, observed: u64) {
        drop(
            self.changed
                .wait_while(self.state.lock().unwrap(), |_| self.sequence() == observed)
                .unwrap(),
        );
    }
    pub(crate) fn acknowledge(&self, generation: u64) {
        self.state.lock().unwrap().quiescent = generation;
        self.changed.notify_all();
    }
    pub(crate) fn finish(&self) {
        self.state.lock().unwrap().finished = true;
        self.changed.notify_all();
    }
    pub(crate) fn wait_acknowledged(&self, generation: u64) -> bool {
        let state = self
            .changed
            .wait_while(self.state.lock().unwrap(), |state| {
                state.quiescent < generation && !state.finished
            })
            .unwrap();
        state.quiescent >= generation
    }
    pub(crate) fn notify(&self) {
        let native = {
            let state = self.state.lock().unwrap();
            self.sequence.fetch_add(1, Ordering::Release);
            self.changed.notify_one();
            state.native.clone()
        };
        if let Some(native) = native {
            native.request_preemption();
        }
    }
}
impl Wake for CoreSignal {
    fn wake(self: Arc<Self>) {
        self.notify();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}

/// Actual ready operations, rather than per-execution CPU result messages.
pub struct KernelQueue {
    state: Mutex<QueueState>,
    waker: std::task::Waker,
    adopted: std::sync::OnceLock<crate::ExternalEventSender>,
}
#[derive(Default)]
struct QueueState {
    ready: VecDeque<Arc<dyn ThreadWait>>,
    notified: bool,
}
impl KernelQueue {
    pub fn new(waker: std::task::Waker) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::default(),
            waker,
            adopted: std::sync::OnceLock::new(),
        })
    }
    pub(crate) fn adopt(&self, events: crate::ExternalEventSender) {
        self.adopted
            .set(events)
            .expect("one failed-core cleanup owner");
        let ready = !self.state.lock().unwrap().ready.is_empty();
        if ready {
            self.adopted
                .get()
                .unwrap()
                .notify_host_service(crate::ExternalEventSource::Worker);
        }
    }
    pub fn enqueue(&self, operation: Arc<dyn ThreadWait>) {
        let notify = {
            let mut state = self.state.lock().unwrap();
            state.ready.push_back(operation);
            !std::mem::replace(&mut state.notified, true)
        };
        if notify {
            if let Some(events) = self.adopted.get() {
                events.notify_host_service(crate::ExternalEventSource::Worker);
            } else {
                self.waker.wake_by_ref();
            }
        }
    }
    pub fn begin_batch(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        state.notified = false;
        state.ready.len()
    }
    pub fn pop(&self) -> Option<Arc<dyn ThreadWait>> {
        self.state.lock().unwrap().ready.pop_front()
    }
}
