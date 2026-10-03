//! Thread-owned host input with a bounded latest-state mailbox.

use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::{
    ControllerState, GamepadProfiles, InputManager, ProfiledControllerState, sdl::SdlInputBackend,
};

const POLL_INTERVAL: Duration = Duration::from_millis(5);

#[derive(Debug)]
pub struct InputWorkerError(String);

impl fmt::Display for InputWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InputWorkerError {}

struct Shared<T> {
    pending: Mutex<Option<InputSample<T>>>,
    available: AtomicBool,
    stop: AtomicBool,
    finished: AtomicBool,
    failure: Mutex<Option<String>>,
}

#[derive(Debug)]
/// Complete state captured by one host poll, with its host monotonic time.
pub struct InputSample<T> {
    pub captured_at: Instant,
    pub state: T,
}

struct NotifyOnExit<'a, N: Fn()>(&'a N);

impl<N: Fn()> Drop for NotifyOnExit<'_, N> {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// Owns the input thread and consumes its most recent complete state.
///
/// SDL initialization and subsystem shutdown stay on the calling main thread.
/// Device I/O, profile mapping and device destruction happen on `nixe-input`.
/// A separate InputReader can be moved to the emulation thread.
/// Reading does not wait for device I/O or for the mailbox lock. Dropping the
/// worker requests shutdown and joins it, including any in-flight device I/O.
pub struct InputWorker<T> {
    shared: Arc<Shared<T>>,
    thread: Option<JoinHandle<Result<(), String>>>,
    _subsystems: Option<crate::sdl::InputSubsystems>,
}

impl InputWorker<Option<ProfiledControllerState>> {
    /// Starts profiled sampling. `notify` wakes the consumer when a previously
    /// empty mailbox receives a sample, and when the worker exits. It runs on
    /// the input thread and must return promptly without panicking.
    pub fn with_profiles(
        sdl: &sdl3::Sdl,
        profiles: GamepadProfiles,
        notify: impl Fn() + Send + 'static,
    ) -> io::Result<Self> {
        let owner = crate::sdl::InputSubsystems::new(sdl).map_err(io::Error::other)?;
        let subsystems = WorkerSubsystems(owner.clone());
        let mut worker = Self::spawn(
            move || {
                let backend = SdlInputBackend::new(subsystems.take());
                let mut input = InputManager::with_profiles(backend, profiles);
                Ok::<_, crate::sdl::SdlInputError>(move || input.read_profiled_input())
            },
            notify,
        )?;
        worker._subsystems = Some(owner);
        Ok(worker)
    }
}

impl InputWorker<Option<ControllerState>> {
    pub fn unmapped(sdl: &sdl3::Sdl) -> io::Result<Self> {
        let owner = crate::sdl::InputSubsystems::new(sdl).map_err(io::Error::other)?;
        let subsystems = WorkerSubsystems(owner.clone());
        let mut worker = Self::spawn(
            move || {
                let backend = SdlInputBackend::new(subsystems.take());
                let mut input = InputManager::new(backend);
                Ok::<_, crate::sdl::SdlInputError>(move || input.read_input())
            },
            || {},
        )?;
        worker._subsystems = Some(owner);
        Ok(worker)
    }
}

impl<T: Send + 'static> InputWorker<T> {
    // The factory crosses threads, but the poller does not need to be Send:
    // it is constructed and destroyed exclusively on the input thread.
    fn spawn<F, P, E, N>(initialize: F, notify: N) -> io::Result<Self>
    where
        F: FnOnce() -> Result<P, E> + Send + 'static,
        P: FnMut() -> Result<T, E>,
        E: fmt::Display,
        N: Fn() + Send + 'static,
    {
        let shared = Arc::new(Shared {
            pending: Mutex::new(None),
            available: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            failure: Mutex::new(None),
        });
        let producer = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("nixe-input".to_owned())
            .spawn(move || {
                let _notify_on_exit = NotifyOnExit(&notify);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut poll = initialize().map_err(|error| error.to_string())?;
                    while !producer.stop.load(Ordering::Acquire) {
                        let deadline = Instant::now() + POLL_INTERVAL;
                        let state = poll().map_err(|error| error.to_string())?;
                        // Never hold the mailbox across backend work or destruction
                        // of an overwritten state. Slow consumers skip old samples.
                        let old = {
                            let mut pending = producer.pending.lock().unwrap();
                            let old = pending.replace(InputSample {
                                captured_at: Instant::now(),
                                state,
                            });
                            producer.available.store(true, Ordering::Release);
                            old
                        };
                        if old.is_none() {
                            notify();
                        }
                        drop(old);
                        while !producer.stop.load(Ordering::Acquire) {
                            let remaining = deadline.saturating_duration_since(Instant::now());
                            if remaining.is_zero() {
                                break;
                            }
                            thread::park_timeout(remaining);
                        }
                    }
                    Ok(())
                }))
                .unwrap_or_else(|payload| Err(panic_message(payload)));
                *producer.failure.lock().unwrap() = Some(match &result {
                    Ok(()) => "input worker stopped unexpectedly".to_owned(),
                    Err(error) => error.clone(),
                });
                producer.finished.store(true, Ordering::Release);
                result
            })?;
        Ok(Self {
            shared,
            thread: Some(thread),
            _subsystems: None,
        })
    }
}

/// SDL remains owned by InputWorker; only sampled states cross to consumers.
pub struct InputReader<T> {
    shared: Arc<Shared<T>>,
}

impl<T> InputWorker<T> {
    pub fn reader(&self) -> InputReader<T> {
        InputReader {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn take_latest(&mut self) -> Result<Option<InputSample<T>>, InputWorkerError> {
        self.reader().take_latest()
    }
}

impl<T> InputReader<T> {
    /// Takes the latest unread sample without waiting for device I/O.
    pub fn take_latest(&mut self) -> Result<Option<InputSample<T>>, InputWorkerError> {
        if self.shared.finished.load(Ordering::Acquire) {
            return Err(InputWorkerError(
                self.shared
                    .failure
                    .lock()
                    .unwrap()
                    .clone()
                    .expect("finished worker publishes its failure first"),
            ));
        }
        if !self.shared.available.load(Ordering::Acquire) {
            return Ok(None);
        }
        let pending = match self.shared.pending.try_lock() {
            Ok(mut pending) => {
                let sample = pending.take();
                self.shared.available.store(false, Ordering::Release);
                sample
            }
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(_)) => {
                return Err(InputWorkerError("input state mailbox poisoned".to_owned()));
            }
        };
        Ok(pending)
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    let reason = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic payload");
    format!("input worker panicked: {reason}")
}

fn join(thread: JoinHandle<Result<(), String>>) -> Result<(), String> {
    thread
        .join()
        .unwrap_or_else(|payload| Err(panic_message(payload)))
}

// Only clones cross threads; the !Send InputWorker keeps the original SDL
// references and joins before releasing them on the main thread. These
// subsystem operations are thread-safe, unlike initialization/finalization:
// https://wiki.libsdl.org/SDL3/SDL_UpdateGamepads
// https://wiki.libsdl.org/SDL3/SDL_OpenGamepad
// https://wiki.libsdl.org/SDL3/SDL_AddEventWatch
struct WorkerSubsystems(crate::sdl::InputSubsystems);
unsafe impl Send for WorkerSubsystems {}
impl WorkerSubsystems {
    fn take(self) -> crate::sdl::InputSubsystems {
        self.0
    }
}

impl<T> Drop for InputWorker<T> {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            if let Err(error) = join(thread) {
                log::error!("input worker shutdown: {error}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::mpsc::{self, Sender};
    use std::thread::ThreadId;

    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(3);

    struct ThreadOwned {
        owner: ThreadId,
        dropped: Sender<ThreadId>,
        _not_send: Rc<()>,
    }

    impl ThreadOwned {
        fn check(&self) {
            assert_eq!(thread::current().id(), self.owner);
        }
    }

    impl Drop for ThreadOwned {
        fn drop(&mut self) {
            self.check();
            let _ = self.dropped.send(thread::current().id());
        }
    }

    fn wait_for_exit<T>(worker: &InputWorker<T>) {
        let deadline = Instant::now() + TEST_TIMEOUT;
        while !worker.thread.as_ref().unwrap().is_finished() {
            assert!(Instant::now() < deadline, "input worker did not exit");
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn take_state(worker: &mut InputWorker<Option<u32>>) -> Option<Option<u32>> {
        worker.take_latest().unwrap().map(|sample| sample.state)
    }

    #[test]
    fn slow_device_work_never_blocks_reads_and_only_latest_state_is_kept() {
        let (initialized, initialization_started) = mpsc::channel();
        let (allow_init, init_gate) = mpsc::channel();
        let (poll_started, polling) = mpsc::channel();
        let (samples, commands) = mpsc::channel::<Result<Option<u32>, String>>();
        let (dropped, destruction) = mpsc::channel();
        let (notified, notifications) = mpsc::channel();
        let mut worker = InputWorker::spawn(
            move || {
                let owner = thread::current().id();
                assert_eq!(thread::current().name(), Some("nixe-input"));
                initialized.send(owner).unwrap();
                init_gate
                    .recv_timeout(TEST_TIMEOUT)
                    .map_err(|error| error.to_string())?;
                let device = ThreadOwned {
                    owner,
                    dropped,
                    _not_send: Rc::new(()),
                };
                Ok::<_, String>(move || {
                    device.check();
                    poll_started.send(()).unwrap();
                    commands
                        .recv_timeout(TEST_TIMEOUT)
                        .map_err(|error| error.to_string())?
                })
            },
            move || {
                let _ = notified.send(());
            },
        )
        .unwrap();

        let owner = initialization_started.recv_timeout(TEST_TIMEOUT).unwrap();
        assert_ne!(owner, thread::current().id());
        // Initialization is still blocked, without pretending a device has
        // already been observed as disconnected.
        assert_eq!(take_state(&mut worker), None);
        allow_init.send(()).unwrap();
        polling.recv_timeout(TEST_TIMEOUT).unwrap();
        assert_eq!(take_state(&mut worker), None);

        samples.send(Ok(Some(1))).unwrap();
        polling.recv_timeout(TEST_TIMEOUT).unwrap();
        notifications.recv_timeout(TEST_TIMEOUT).unwrap();
        let first = worker.take_latest().unwrap().unwrap();
        assert_eq!(first.state, Some(1));
        // No duplicate publication while the next device operation is blocked.
        assert_eq!(take_state(&mut worker), None);

        // A slow consumer gets the newest sample, not a backlog.
        samples.send(Ok(Some(2))).unwrap();
        polling.recv_timeout(TEST_TIMEOUT).unwrap();
        notifications.recv_timeout(TEST_TIMEOUT).unwrap();
        samples.send(Ok(Some(3))).unwrap();
        polling.recv_timeout(TEST_TIMEOUT).unwrap();
        assert_eq!(notifications.try_recv(), Err(mpsc::TryRecvError::Empty));
        let shared = Arc::clone(&worker.shared);
        let locked = shared.pending.lock().unwrap();
        assert_eq!(take_state(&mut worker), None);
        drop(locked);
        let newest = worker.take_latest().unwrap().unwrap();
        assert_eq!(newest.state, Some(3));
        assert!(newest.captured_at > first.captured_at);
        assert_eq!(take_state(&mut worker), None);

        samples.send(Ok(None)).unwrap();
        polling.recv_timeout(TEST_TIMEOUT).unwrap();
        assert_eq!(take_state(&mut worker), Some(None));
        samples.send(Ok(Some(4))).unwrap();
        polling.recv_timeout(TEST_TIMEOUT).unwrap();
        assert_eq!(take_state(&mut worker), Some(Some(4)));

        samples
            .send(Err("device enumeration failed".to_owned()))
            .unwrap();
        wait_for_exit(&worker);
        for _ in 0..2 {
            assert_eq!(
                worker.take_latest().unwrap_err().to_string(),
                "device enumeration failed"
            );
        }
        assert_eq!(destruction.recv_timeout(TEST_TIMEOUT).unwrap(), owner);
    }

    #[test]
    fn initialization_failures_are_reported() {
        let (notified, notifications) = mpsc::channel();
        let mut worker = InputWorker::spawn(
            || Err::<fn() -> Result<(), &'static str>, _>("SDL initialization failed"),
            move || {
                let _ = notified.send(());
            },
        )
        .unwrap();
        notifications.recv_timeout(TEST_TIMEOUT).unwrap();
        wait_for_exit(&worker);
        assert_eq!(
            worker.take_latest().unwrap_err().to_string(),
            "SDL initialization failed"
        );
    }

    #[test]
    fn panics_are_reported_instead_of_leaving_stale_input() {
        let mut worker = InputWorker::spawn(
            || Ok::<_, &'static str>(|| -> Result<(), &'static str> { panic!("device panic") }),
            || {},
        )
        .unwrap();
        wait_for_exit(&worker);
        assert_eq!(
            worker.take_latest().unwrap_err().to_string(),
            "input worker panicked: device panic"
        );
    }

    #[test]
    fn shutdown_joins_and_destroys_the_device_on_its_owner_thread() {
        let (initialized, ready) = mpsc::channel();
        let (dropped, destruction) = mpsc::channel();
        let worker = InputWorker::spawn(
            move || {
                let owner = thread::current().id();
                let device = ThreadOwned {
                    owner,
                    dropped,
                    _not_send: Rc::new(()),
                };
                initialized.send(owner).unwrap();
                Ok::<_, &'static str>(move || {
                    device.check();
                    Ok(())
                })
            },
            || {},
        )
        .unwrap();
        let owner = ready.recv_timeout(TEST_TIMEOUT).unwrap();
        drop(worker);
        assert_eq!(destruction.try_recv().unwrap(), owner);
    }
    #[test]
    fn reader_crosses_threads_without_owning_sdl_or_outliving_shutdown_silently() {
        let (notified, notifications) = mpsc::channel();
        let worker = InputWorker::spawn(
            || Ok::<_, &'static str>(|| Ok(Some(7_u32))),
            move || {
                let _ = notified.send(());
            },
        )
        .unwrap();
        notifications.recv_timeout(TEST_TIMEOUT).unwrap();
        let mut reader = worker.reader();
        let mut reader = thread::spawn(move || {
            assert_eq!(reader.take_latest().unwrap().unwrap().state, Some(7));
            reader
        })
        .join()
        .unwrap();
        drop(worker);
        assert_eq!(
            reader.take_latest().unwrap_err().to_string(),
            "input worker stopped unexpectedly"
        );
    }
}
