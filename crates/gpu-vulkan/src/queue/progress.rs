//! Actual native submission progress and notifications, shared with its worker.
use crate::VulkanError;
use ash::vk;
use std::collections::BTreeSet;
use std::sync::{
    Arc, Condvar, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

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
    targets: BTreeSet<u64>,
    stopping: bool,
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
        let wake = {
            let mut state = self.notification.lock().unwrap();
            self.completed.fetch_max(tick, Ordering::Release);
            let covered = state.targets.first().is_some_and(|&target| target <= tick);
            while state.targets.first().is_some_and(|&target| target <= tick) {
                state.targets.pop_first();
            }
            covered.then(|| state.wake.clone()).flatten()
        };
        if let Some(wake) = wake {
            wake();
        }
    }
    pub fn arm(&self, tick: u64) -> Result<(), VulkanError> {
        let mut state = self.notification.lock().unwrap();
        self.check(tick)?;
        if tick <= self.completed() {
            // Completion may precede registration after the owner's last
            // poll. Eden RegisterAction invokes an already-reached action
            // immediately; preserve that wake before the owner can sleep.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/host1x/syncpoint_manager.cpp
            let wake = state.wake.clone();
            drop(state);
            if let Some(wake) = wake {
                wake();
            }
            return Ok(());
        }
        if state.stopping {
            return Err(VulkanError::QueueStopped);
        }
        if state.targets.insert(tick) {
            self.changed.notify_all();
        }
        Ok(())
    }
    pub fn next_target(&self) -> Option<u64> {
        let mut state = self.notification.lock().unwrap();
        loop {
            if state.stopping {
                return None;
            }
            if self.cancelled() {
                state.targets.retain(|&tick| tick <= self.submitted());
            }
            if let Some(failure) = self.failure() {
                state.targets.retain(|&tick| tick < failure.tick);
            }
            if let Some(&target) = state.targets.first()
                && target <= self.submitted()
            {
                return Some(target);
            }
            state = self.changed.wait(state).unwrap();
        }
    }
    pub fn stop_observer(&self) {
        let mut state = self.notification.lock().unwrap();
        state.stopping = true;
        self.changed.notify_all();
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
    fn observer_parks_until_actual_submission_and_reselects_the_earliest_target() {
        let progress = Arc::new(SubmissionProgress::default());
        progress.arm(9).unwrap();
        let (sent, received) = std::sync::mpsc::channel();
        let pending = Arc::clone(&progress);
        let observer = thread::spawn(move || sent.send(pending.next_target()).unwrap());
        assert!(received.recv_timeout(Duration::from_millis(20)).is_err());
        progress.arm(4).unwrap();
        progress.submitted_successfully(4);
        assert_eq!(
            received.recv_timeout(Duration::from_secs(1)).unwrap(),
            Some(4)
        );
        observer.join().unwrap();
        progress.observe_completed(4);
        progress.arm(4).unwrap();
        progress.stop_observer();
        assert_eq!(progress.next_target(), None);
    }

    #[test]
    fn completed_target_notifies_when_registered_after_the_owner_poll() {
        let progress = SubmissionProgress::default();
        let (sent, received) = std::sync::mpsc::channel();
        progress.bind_wakeup(Arc::new(move || sent.send(()).unwrap()));
        // No target was armed when the GPU completed. Registration is the
        // owner's final check before sleeping for another message.
        progress.submitted_successfully(1);
        progress.observe_completed(1);
        assert!(received.try_recv().is_err());
        progress.arm(1).unwrap();
        received.recv_timeout(Duration::from_secs(1)).unwrap();
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
