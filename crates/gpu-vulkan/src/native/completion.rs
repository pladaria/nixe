//! One sleeping timeline observer. It owns no command/resource state and never
//! publishes guest effects. The owner rechecks the real timeline after a wakeup.
use super::*;
use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[derive(Default)]
struct State {
    targets: BTreeSet<u64>,
    completed: u64,
    stopped: bool,
    failure: Option<vk::Result>,
}
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}
pub(super) struct CompletionWakeup {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}
impl CompletionWakeup {
    pub fn new(device: &VulkanDevice, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        let shared = Arc::new(Shared::default());
        let pending = Arc::clone(&shared);
        let owner = Arc::clone(&device.owner);
        let timeline = device.submission.timeline;
        let worker = thread::Builder::new()
            .name("nixe-gpu-fence".into())
            .spawn(move || {
                loop {
                    let target = {
                        let mut state = pending.state.lock().unwrap();
                        while !state.stopped && state.targets.is_empty() {
                            state = pending.changed.wait(state).unwrap();
                        }
                        if state.stopped {
                            break;
                        }
                        *state.targets.first().expect("pending completion target")
                    };
                    let semaphores = [timeline];
                    let values = [target];
                    let info = vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values);
                    // Concurrent timeline waits do not require queue ownership.
                    // A bounded wait allows shutdown even after an external stall.
                    // https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitSemaphores.html
                    let result = unsafe { owner.device.wait_semaphores(&info, 10_000_000) };
                    let mut state = pending.state.lock().unwrap();
                    if state.stopped {
                        break;
                    }
                    match result {
                        Ok(()) => {
                            state.completed = state.completed.max(target);
                            while state
                                .targets
                                .first()
                                .is_some_and(|&pending| pending <= target)
                            {
                                state.targets.pop_first();
                            }
                        }
                        Err(vk::Result::TIMEOUT) => continue,
                        Err(error) => {
                            state.failure = Some(error);
                            state.stopped = true;
                        }
                    }
                    let failed = state.failure.is_some();
                    drop(state);
                    nixe_gpu::trace::detail_event(
                        "completion_notification",
                        target,
                        u64::from(failed),
                        0,
                    );
                    wake();
                    if failed {
                        break;
                    }
                }
            })
            .expect("failed to create the GPU timeline observer");
        Self {
            shared,
            worker: Some(worker),
        }
    }
    pub fn check(&self) -> Result<(), VulkanError> {
        if let Some(error) = self.shared.state.lock().unwrap().failure {
            return Err(VulkanError::vk("completion notification wait", error));
        }
        Ok(())
    }
    pub fn arm(&self, target: u64) -> Result<bool, VulkanError> {
        let mut state = self.shared.state.lock().unwrap();
        if let Some(error) = state.failure {
            return Err(VulkanError::vk("completion notification wait", error));
        }
        // Requests arrive from both the ordered completion list and external
        // CPU consumers. Waiting for only the greatest value can hide an earlier
        // release needed to unblock that later GPU work. On timeout the worker
        // reselects the earliest target, including requests armed during its wait.
        if target <= state.completed {
            return Ok(false);
        }
        if state.targets.insert(target) {
            self.shared.changed.notify_one();
        }
        Ok(true)
    }
}
impl Drop for CompletionWakeup {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap();
            state.stopped = true;
            self.shared.changed.notify_one();
        }
        if let Some(worker) = self.worker.take() {
            worker.join().expect("GPU timeline observer panicked");
        }
    }
}
