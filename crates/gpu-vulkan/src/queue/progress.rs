//! Actual native submission progress and notifications, shared with its worker.
use crate::VulkanError;
use ash::vk;
use std::sync::{
    Arc, Condvar, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

/// Shared ownership keeps pending completion queries valid through queue teardown.
pub(crate) struct Timeline {
    owner: Arc<crate::device::DeviceOwner>,
    pub handle: vk::Semaphore,
}
impl Timeline {
    pub fn new(owner: Arc<crate::device::DeviceOwner>) -> Result<Self, VulkanError> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        let handle = unsafe { owner.device.create_semaphore(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateSemaphore(owner timeline)", e))?;
        Ok(Self { owner, handle })
    }

    // Refresh pending reads from native completion, independently of back-thread activity.
    // https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L2418
    pub fn is_done(&self, progress: &SubmissionProgress, tick: u64) -> Result<bool, VulkanError> {
        progress.check(tick)?;
        if progress.completed() >= tick {
            return Ok(true);
        }
        let submitted = progress.submitted();
        if submitted < tick {
            return Ok(false);
        }
        let completed = unsafe { self.owner.device.get_semaphore_counter_value(self.handle) }
            .map_err(|result| {
                progress.fail(
                    tick,
                    NativeFailure::Vulkan {
                        operation: "vkGetSemaphoreCounterValue(command read)",
                        result,
                    },
                );
                VulkanError::vk("vkGetSemaphoreCounterValue(command read)", result)
            })?;
        progress.observe_completed(completed.min(submitted));
        progress.check(tick)?;
        Ok(progress.completed() >= tick)
    }
}
impl Drop for Timeline {
    fn drop(&mut self) {
        unsafe { self.owner.device.destroy_semaphore(self.handle, None) };
    }
}

#[derive(Clone, Debug)]
pub(crate) enum NativeFailure {
    Compilation(Arc<nixe_gpu::compilation::CompileError>),
    Vulkan {
        operation: &'static str,
        result: vk::Result,
    },
    Panicked,
    Stopped,
}
impl NativeFailure {
    pub fn error(self) -> VulkanError {
        match self {
            Self::Compilation(error) => VulkanError::Compilation(error.to_string().into()),
            Self::Vulkan { operation, result } => VulkanError::vk(operation, result),
            Self::Panicked => VulkanError::InvalidResource("Vulkan scheduler panicked"),
            Self::Stopped => VulkanError::InvalidResource("Vulkan scheduler stopped"),
        }
    }
    pub fn device_lost(&self) -> bool {
        matches!(
            self,
            Self::Vulkan {
                result: vk::Result::ERROR_DEVICE_LOST,
                ..
            }
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SubmissionFailure {
    pub tick: u64,
    pub cause: NativeFailure,
}
impl SubmissionFailure {
    pub fn error(self) -> VulkanError {
        VulkanError::SubmissionFailed {
            tick: self.tick,
            cause: Box::new(self.cause.error()),
        }
    }
}

#[derive(Default)]
struct Notification {
    sequence: u64,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

#[derive(Default)]
pub(crate) struct SubmissionProgress {
    submitted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicBool,
    failure: OnceLock<SubmissionFailure>,
    notification: Mutex<Notification>,
    changed: Condvar,
}
impl SubmissionProgress {
    pub fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }
    pub fn failure(&self) -> Option<SubmissionFailure> {
        self.failure.get().cloned()
    }
    pub fn check(&self, tick: u64) -> Result<(), VulkanError> {
        if let Some(failure) = self.failure().filter(|failure| tick >= failure.tick) {
            return Err(failure.error());
        }
        if self.cancelled() && tick > self.submitted() {
            return Err(VulkanError::QueueStopped);
        }
        Ok(())
    }
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    pub fn cancel_pending(&self) {
        self.cancelled.store(true, Ordering::Release);
        let wake = {
            let mut state = self.notification.lock().unwrap();
            state.sequence = state.sequence.wrapping_add(1);
            self.changed.notify_all();
            state.wake.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
    }
    pub fn submitted_successfully(&self, tick: u64) {
        self.submitted.store(tick, Ordering::Release);
        let mut state = self.notification.lock().unwrap();
        state.sequence = state.sequence.wrapping_add(1);
        self.changed.notify_all();
    }
    pub fn fail(&self, tick: u64, cause: NativeFailure) {
        if self.failure.set(SubmissionFailure { tick, cause }).is_err() {
            return;
        }
        let wake = {
            let mut state = self.notification.lock().unwrap();
            state.sequence = state.sequence.wrapping_add(1);
            self.changed.notify_all();
            state.wake.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
    }
    pub fn bind_wakeup(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        let failed = {
            let mut state = self.notification.lock().unwrap();
            state.wake = Some(Arc::clone(&wake));
            self.failure.get().is_some()
        };
        if failed {
            wake();
        }
    }
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }
    pub fn observe_completed(&self, tick: u64) {
        self.completed.fetch_max(tick, Ordering::Release);
    }
    pub fn wait_submitted(
        &self,
        tick: u64,
        deadline: Option<Instant>,
    ) -> Result<bool, VulkanError> {
        self.check(tick)?;
        if self.submitted() >= tick {
            return Ok(true);
        }
        let mut state = self.notification.lock().unwrap();
        loop {
            // Register under the notification lock, then recheck both outcomes.
            // A publication cannot pass between this check and parking.
            let sequence = state.sequence;
            self.check(tick)?;
            if self.submitted() >= tick {
                return Ok(true);
            }
            if let Some(deadline) = deadline {
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    return Ok(false);
                };
                state = self
                    .changed
                    .wait_timeout_while(state, remaining, |s| s.sequence == sequence)
                    .unwrap()
                    .0;
            } else {
                state = self
                    .changed
                    .wait_while(state, |s| s.sequence == sequence)
                    .unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};

    #[test]
    fn failure_wakes_all_submission_waiters_without_hiding_prior_success() {
        let progress = Arc::new(SubmissionProgress::default());
        progress.submitted_successfully(1);
        let (ready, arrivals) = std::sync::mpsc::channel();
        let waiters: Vec<_> = (2..=5)
            .map(|tick| {
                let progress = Arc::clone(&progress);
                let ready = ready.clone();
                thread::spawn(move || {
                    ready.send(()).unwrap();
                    progress.wait_submitted(tick, Some(Instant::now() + Duration::from_secs(1)))
                })
            })
            .collect();
        for _ in 2..=5 {
            arrivals.recv().unwrap();
        }
        progress.fail(
            2,
            NativeFailure::Vulkan {
                operation: "vkQueueSubmit2",
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            },
        );
        progress.fail(3, NativeFailure::Panicked);
        assert!(progress.wait_submitted(1, None).unwrap());
        for waiter in waiters {
            assert!(
                matches!(waiter.join().unwrap(), Err(VulkanError::SubmissionFailed { tick: 2, cause })
                if matches!(*cause, VulkanError::Vulkan { result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY, .. }))
            );
        }
        assert_eq!(progress.submitted(), 1);
        progress.observe_completed(1);
        assert_eq!(progress.completed(), 1);
        assert!(progress.check(2).is_err());
    }

    #[test]
    fn submission_acknowledgement_waits_for_its_target_only() {
        let progress = Arc::new(SubmissionProgress::default());
        progress.submitted_successfully(7);
        assert!(progress.wait_submitted(7, Some(Instant::now())).unwrap());
        assert!(!progress.wait_submitted(8, Some(Instant::now())).unwrap());
        let pending = Arc::clone(&progress);
        let waiter = thread::spawn(move || pending.wait_submitted(8, None));
        progress.cancel_pending();
        assert!(matches!(
            waiter.join().unwrap(),
            Err(VulkanError::QueueStopped)
        ));
        assert!(progress.wait_submitted(7, None).unwrap());
    }

    #[test]
    fn registration_after_failure_still_delivers_a_notification() {
        let progress = SubmissionProgress::default();
        progress.fail(1, NativeFailure::Panicked);
        let (sent, received) = std::sync::mpsc::channel();
        progress.bind_wakeup(Arc::new(move || sent.send(()).unwrap()));
        received.recv_timeout(Duration::from_secs(1)).unwrap();
    }
}
