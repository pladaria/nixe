//! Independent guest display deadlines; no host surface calls run on this owner.
use nixe_horizon::VideoSystem;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};
pub(super) struct DisplayClock {
    stop: mpsc::Sender<()>,
    worker: Option<JoinHandle<()>>,
    failure: Arc<Mutex<Option<String>>>,
    failed: Arc<AtomicBool>,
}
impl DisplayClock {
    pub(super) fn start(video: VideoSystem, origin: Instant) -> Result<Self, String> {
        let (stop, receiver) = mpsc::channel();
        let failure = Arc::new(Mutex::new(None));
        let worker_failure = failure.clone();
        let failed = Arc::new(AtomicBool::new(false));
        let worker_failed = failed.clone();
        let worker = thread::Builder::new()
            .name("nixe-display-clock".into())
            .spawn(move || {
                loop {
                    let timeout = video
                        .next_display_deadline()
                        .saturating_sub(origin.elapsed());
                    match receiver.recv_timeout(timeout) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    if let Err(error) = video.advance(origin.elapsed()) {
                        *worker_failure.lock().unwrap() = Some(error.to_string());
                        worker_failed.store(true, Ordering::Release);
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stop,
            worker: Some(worker),
            failure,
            failed,
        })
    }
    pub(super) fn require_healthy(&self) -> Result<(), String> {
        if !self.failed.load(Ordering::Acquire) {
            return Ok(());
        }
        match &*self.failure.lock().unwrap() {
            Some(failure) => Err(failure.clone()),
            None => Ok(()),
        }
    }
}
impl Drop for DisplayClock {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn display_deadlines_progress_without_coordinator_service() {
        let video = VideoSystem::default();
        let first = video.next_display_deadline();
        let owner = DisplayClock::start(video.clone(), Instant::now()).unwrap();
        let limit = Instant::now() + Duration::from_secs(1);
        while video.next_display_deadline() == first {
            assert!(
                Instant::now() < limit,
                "independent display owner did not advance"
            );
            thread::sleep(Duration::from_millis(1));
        }
        owner.require_healthy().unwrap();
        drop(owner);
        assert!(video.next_display_deadline() > first);
    }
}
