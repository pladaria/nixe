//! Physical-core interrupts and host idle waits.
use nixe_cpu::execution::CpuControl;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    task::Wake,
};

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
