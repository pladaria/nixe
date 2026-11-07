//! Eden's asynchronous FenceManager: wait, publish caches, then run CPU actions.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/fence_manager.h
use super::*;
use crate::queue::progress::{NativeFailure, SubmissionProgress};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

struct Fence {
    work: Option<WorkTicket>,
    flushes: bool,
    action: Option<nixe_gpu::FenceAction>,
}
#[derive(Default)]
struct Queue {
    pending: VecDeque<Fence>,
    active: bool,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}
#[derive(Default)]
struct Shared {
    queue: Mutex<Queue>,
    failure: OnceLock<Arc<nixe_memory::MemoryError>>,
    stopping: AtomicBool,
    changed: Condvar,
}
pub(super) struct FenceManager {
    shared: Arc<Shared>,
    progress: Arc<SubmissionProgress>,
    worker: Option<JoinHandle<()>>,
}
struct WorkerScope {
    progress: Arc<SubmissionProgress>,
    shared: Arc<Shared>,
}
impl Drop for WorkerScope {
    fn drop(&mut self) {
        if thread::panicking() {
            let (wake, cancelled) = {
                let mut state = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
                let _ = self
                    .shared
                    .failure
                    .set(failure("GPU fencing thread panicked"));
                state.active = false;
                let cancelled = std::mem::take(&mut state.pending);
                self.shared.changed.notify_all();
                (state.wake.clone(), cancelled)
            };
            self.progress
                .fail(self.progress.completed() + 1, NativeFailure::Panicked);
            for mut fence in cancelled {
                if let Some(action) = fence.action.take() {
                    let _ = action(Err(failure("GPU fencing thread panicked")));
                }
            }
            if let Some(wake) = wake {
                wake();
            }
        }
    }
}
fn failure(error: impl std::fmt::Display) -> Arc<nixe_memory::MemoryError> {
    Arc::new(nixe_memory::MemoryError::Execution(
        error.to_string().into(),
    ))
}
#[cfg(target_os = "linux")]
fn configure_fencing_thread() {
    // Common::SetCurrentThreadPriority(High), respecting RLIMIT_NICE.
    // SetCurrentThreadToPerformanceCores is Android-only in the reference.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/thread.cpp#L47-L85
    let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    let lowest = if unsafe { libc::getrlimit(libc::RLIMIT_NICE, limit.as_mut_ptr()) } == 0 {
        20 - unsafe { limit.assume_init() }.rlim_cur.min(40) as i32
    } else {
        0
    };
    let nice = (-4).max(lowest.min(0));
    if unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) } != 0 {
        log::debug!(
            "Cannot set GPU fencing thread priority: {}",
            std::io::Error::last_os_error()
        );
    }
}
impl FenceManager {
    pub fn new(
        device: &VulkanDevice,
        caches: Arc<crate::residency::cache_owners::CacheOwners>,
    ) -> Result<Self, VulkanError> {
        let shared = Arc::new(Shared::default());
        let queue = Arc::clone(&shared);
        let progress = Arc::clone(&device.submission.progress);
        let pending = Arc::clone(&progress);
        let owner = Arc::clone(&device.owner);
        let timeline = device.submission.timeline;
        let worker = thread::Builder::new()
            .name("nixe-gpu-fence".into())
            .spawn(move || {
                #[cfg(target_os = "linux")]
                configure_fencing_thread();
                let _scope = WorkerScope {
                    progress: Arc::clone(&pending),
                    shared: Arc::clone(&queue),
                };
                loop {
                    let mut fence = {
                        let mut state = queue.queue.lock().unwrap();
                        while state.pending.is_empty() && !queue.stopping.load(Ordering::Acquire) {
                            state = queue.changed.wait(state).unwrap();
                        }
                        if queue.stopping.load(Ordering::Acquire) {
                            return;
                        }
                        let fence = state.pending.pop_front().unwrap();
                        state.active = true;
                        fence
                    };
                    let result = (|| {
                        if let Some(work) = fence.work {
                            loop {
                                if queue.stopping.load(Ordering::Acquire) {
                                    return Err(failure("GPU fencing thread stopped"));
                                }
                                pending.check(work.tick()).map_err(failure)?;
                                if work.tick() <= pending.completed() {
                                    break;
                                }
                                // MasterSemaphore::Wait refreshes the physical
                                // counter before falling back to a native wait.
                                let counter =
                                    unsafe { owner.device.get_semaphore_counter_value(timeline) }
                                        .map_err(|result| {
                                        pending.fail(
                                            work.tick(),
                                            NativeFailure::Vulkan {
                                                operation: "GPU fence counter",
                                                result,
                                            },
                                        );
                                        failure(VulkanError::vk("GPU fence counter", result))
                                    })?;
                                pending.observe_completed(counter);
                                if counter >= work.tick() {
                                    break;
                                }
                                if !pending
                                    .wait_submitted(
                                        work.tick(),
                                        Some(Instant::now() + Duration::from_millis(10)),
                                    )
                                    .map_err(failure)?
                                {
                                    continue;
                                }
                                let semaphores = [timeline];
                                let values = [work.tick()];
                                let info = vk::SemaphoreWaitInfo::default()
                                    .semaphores(&semaphores)
                                    .values(&values);
                                match unsafe { owner.device.wait_semaphores(&info, 10_000_000) } {
                                    Ok(()) => {
                                        pending.observe_completed(work.tick());
                                        break;
                                    }
                                    Err(vk::Result::TIMEOUT) => {}
                                    Err(result) => {
                                        pending.fail(
                                            work.tick(),
                                            NativeFailure::Vulkan {
                                                operation: "GPU fence wait",
                                                result,
                                            },
                                        );
                                        return Err(failure(VulkanError::vk(
                                            "GPU fence wait",
                                            result,
                                        )));
                                    }
                                }
                            }
                        }
                        if fence.flushes {
                            caches.pop_async_flushes().map_err(failure)?;
                        }
                        Ok(())
                    })();
                    let result = if let Some(action) = fence.action.take() {
                        let action_result = action(result.clone());
                        result.and(action_result)
                    } else {
                        result
                    };
                    let (wake, cancelled) = {
                        let mut state = queue.queue.lock().unwrap();
                        state.active = false;
                        let cancelled = if let Err(cause) = result {
                            let _ = queue.failure.set(cause);
                            std::mem::take(&mut state.pending)
                        } else {
                            VecDeque::new()
                        };
                        let wake = state.wake.clone();
                        queue.changed.notify_all();
                        (wake, cancelled)
                    };
                    for mut fence in cancelled {
                        if let Some(action) = fence.action.take() {
                            let _ = action(Err(failure("preceding GPU fence failed")));
                        }
                    }
                    if let Some(wake) = wake {
                        wake();
                    }
                }
            })
            .map_err(|error| VulkanError::Loader(format!("GPU fencing thread: {error}").into()))?;
        Ok(Self {
            shared,
            progress,
            worker: Some(worker),
        })
    }
    pub fn bind_wakeup(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.progress.bind_wakeup(Arc::clone(&wake));
        let ready = {
            let mut state = self.shared.queue.lock().unwrap();
            state.wake = Some(Arc::clone(&wake));
            self.shared.failure.get().is_some()
        };
        if ready {
            wake();
        }
    }
    pub fn queue(
        &self,
        work: Option<WorkTicket>,
        flushes: bool,
        action: Option<nixe_gpu::FenceAction>,
    ) -> Result<(), VulkanError> {
        let mut state = self.shared.queue.lock().unwrap();
        if let Some(cause) = self.shared.failure.get() {
            return Err(VulkanError::Memory(Arc::clone(cause)));
        }
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(VulkanError::QueueStopped);
        }
        state.pending.push_back(Fence {
            work,
            flushes,
            action,
        });
        self.shared.changed.notify_one();
        Ok(())
    }
    pub fn check(&self) -> Result<(), VulkanError> {
        if let Some(cause) = self.shared.failure.get() {
            return Err(VulkanError::Memory(Arc::clone(cause)));
        }
        Ok(())
    }
    pub fn wait_pending(&self) -> Result<(), VulkanError> {
        let mut state = self.shared.queue.lock().unwrap();
        loop {
            if let Some(cause) = self.shared.failure.get() {
                return Err(VulkanError::Memory(Arc::clone(cause)));
            }
            if !state.active && state.pending.is_empty() {
                return Ok(());
            }
            state = self.shared.changed.wait(state).unwrap();
        }
    }
}
impl Drop for FenceManager {
    fn drop(&mut self) {
        let pending = {
            let mut state = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.stopping.store(true, Ordering::Release);
            self.shared.changed.notify_all();
            std::mem::take(&mut state.pending)
        };
        for mut fence in pending {
            if let Some(action) = fence.action.take() {
                let _ = action(Err(failure("GPU fencing thread stopped")));
            }
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.progress
                .fail(self.progress.completed() + 1, NativeFailure::Panicked);
        }
    }
}
