//! CoreTiming signals VI; the display worker owns composition and VSync events.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/vi/conductor.cpp
use super::{FramebufferError, VideoState, VideoSystem};
use std::sync::{
    Arc, Condvar, Mutex, Weak,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Wake, Waker};
use std::thread::JoinHandle;
use std::time::Instant;

const FRAME_NS: u64 = 1_000_000_000 / 60;

#[derive(Debug, Default)]
struct State {
    tick: Option<Instant>,
    registered: bool,
    stopping: bool,
    failure: Option<Box<str>>,
}
#[derive(Debug, Default)]
struct Signals {
    state: Mutex<State>,
    tick: Condvar,
    registered: Condvar,
    failed: AtomicBool,
}
impl Signals {
    fn wait_tick(&self) -> Option<Instant> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !state.stopping && state.tick.is_none() {
            state = self
                .tick
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        if state.stopping {
            None
        } else {
            state.tick.take()
        }
    }
    fn wait_registration(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !state.stopping && !state.registered {
            state = self
                .registered
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.registered = false;
        !state.stopping
    }
    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopping = true;
        self.tick.notify_all();
        self.registered.notify_all();
    }
    fn is_stopping(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopping
    }
    fn fail(&self, detail: Box<str>, events: &nixe_runtime::ExternalEventSender) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .failure = Some(detail);
        self.failed.store(true, Ordering::Release);
        events.notify_host_service(nixe_runtime::ExternalEventSource::Worker);
    }
}
struct Tick(Arc<Signals>);
impl Wake for Tick {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let now = Instant::now();
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.stopping && state.tick.is_none() {
            state.tick = Some(now);
            self.0.tick.notify_one();
        }
    }
}
struct Registration(Arc<Signals>);
impl Wake for Registration {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.stopping && !state.registered {
            state.registered = true;
            self.0.registered.notify_one();
        }
    }
}

#[derive(Debug)]
pub(super) struct Conductor {
    signals: Arc<Signals>,
    timer: Arc<nixe_scheduler::LoopingEvent>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
impl Conductor {
    pub(super) fn new(
        kernel: Arc<nixe_scheduler::SchedulerState>,
        video: Weak<Mutex<VideoState>>,
        events: nixe_runtime::ExternalEventSender,
    ) -> std::io::Result<Self> {
        let signals = Arc::new(Signals::default());
        let interval = std::num::NonZeroU64::new(FRAME_NS).unwrap();
        let timer = Arc::new(
            kernel
                .schedule_looping_event(
                    kernel.clock().scheduler_time_ns().saturating_add(FRAME_NS),
                    interval,
                    Waker::from(Arc::new(Tick(Arc::clone(&signals)))),
                )
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "VI timer driver is stopped",
                    )
                })?,
        );
        let worker_signals = Arc::clone(&signals);
        let worker_timer = Arc::clone(&timer);
        let worker = std::thread::Builder::new()
            .name("nixe-vi".into())
            .spawn(move || {
                let registration = Waker::from(Arc::new(Registration(Arc::clone(&worker_signals))));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Self::run(
                        &kernel,
                        &video,
                        &worker_signals,
                        &worker_timer,
                        &registration,
                    )
                }));
                match result {
                    Ok(Ok(())) => (),
                    Ok(Err(error)) => worker_signals.fail(error.to_string().into(), &events),
                    Err(payload) => {
                        let reason = payload
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| payload.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic payload");
                        worker_signals
                            .fail(format!("VI worker panicked: {reason}").into(), &events);
                    }
                }
            })?;
        Ok(Self {
            signals,
            timer,
            worker: Mutex::new(Some(worker)),
        })
    }
    fn run(
        kernel: &nixe_scheduler::SchedulerState,
        video: &Weak<Mutex<VideoState>>,
        signals: &Signals,
        timer: &nixe_scheduler::LoopingEvent,
        registration: &Waker,
    ) -> Result<(), FramebufferError> {
        while let Some(vsync) = signals.wait_tick() {
            if timer.is_stopped() {
                if signals.is_stopping() {
                    return Ok(());
                }
                return Err(FramebufferError::Backend(
                    "VI timer driver stopped during execution".into(),
                ));
            }
            if kernel.clock().is_paused() {
                continue;
            }
            let Some(video) = video.upgrade() else {
                break;
            };
            // WaitForComposite precedes acquisition for the next composition.
            // Completion is signalled independently of the periodic tick, so
            // an unresolved registration neither spins nor occupies CoreTiming.
            loop {
                let mut state = video
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match VideoSystem::poll_composite_registration(&mut state)? {
                    true => {
                        if !kernel.clock().is_paused() {
                            VideoSystem::compose(&mut state, registration, vsync)?;
                        }
                        break;
                    }
                    false => drop(state),
                }
                if !signals.wait_registration() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
    pub(super) fn check(&self) -> Result<(), FramebufferError> {
        if self.signals.failed.load(Ordering::Acquire) {
            let state = self
                .signals
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            return Err(FramebufferError::Backend(
                state
                    .failure
                    .as_ref()
                    .expect("published VI failure")
                    .clone(),
            ));
        }
        Ok(())
    }
    pub(super) fn stop(&self) {
        self.signals.stop();
        self.timer.cancel();
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            worker.join().expect("VI worker panic handler failed");
        }
    }
}
impl Drop for Conductor {
    fn drop(&mut self) {
        self.stop();
    }
}
