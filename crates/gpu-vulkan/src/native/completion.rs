//! A sleeping timeline observer; resource retirement remains with the owner.
use super::*;
use crate::queue::progress::{NativeFailure, SubmissionProgress};
use std::thread::{self, JoinHandle};

pub(super) struct CompletionWakeup {
    progress: Arc<SubmissionProgress>,
    worker: Option<JoinHandle<()>>,
}
struct ObserverScope(Arc<SubmissionProgress>);
impl Drop for ObserverScope {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.fail(self.0.completed() + 1, NativeFailure::Panicked);
        }
    }
}
impl CompletionWakeup {
    pub fn new(
        device: &VulkanDevice,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Self, VulkanError> {
        let progress = Arc::clone(&device.submission.progress);
        progress.bind_wakeup(wake);
        let pending = Arc::clone(&progress);
        let owner = Arc::clone(&device.owner);
        let timeline = device.submission.timeline;
        let worker = thread::Builder::new()
            .name("nixe-gpu-fence".into())
            .spawn(move || {
                let _scope = ObserverScope(Arc::clone(&pending));
                while let Some(target) = pending.next_target() {
                    let semaphores = [timeline];
                    let values = [target];
                    let info = vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values);
                    // Reselect the earliest demand after every bounded wait,
                    // including when an earlier demand arrives during this wait.
                    // https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitSemaphores.html
                    match unsafe { owner.device.wait_semaphores(&info, 10_000_000) } {
                        Ok(()) => {
                            nixe_gpu::trace::detail_event("completion_notification", target, 0, 0);
                            pending.observe_completed(target);
                        }
                        Err(vk::Result::TIMEOUT) => {}
                        Err(result) => pending.fail(
                            pending.completed() + 1,
                            NativeFailure::Vulkan {
                                operation: "completion notification wait",
                                result,
                            },
                        ),
                    }
                }
            })
            .map_err(|error| {
                VulkanError::Loader(format!("GPU timeline observer: {error}").into())
            })?;
        Ok(Self {
            progress,
            worker: Some(worker),
        })
    }
    pub fn arm(&self, target: u64) -> Result<(), VulkanError> {
        self.progress.arm(target)
    }
}
impl Drop for CompletionWakeup {
    fn drop(&mut self) {
        self.progress.stop_observer();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.progress
                .fail(self.progress.completed() + 1, NativeFailure::Panicked);
        }
    }
}
