//! Optional bounded timeline capture; disk I/O runs outside emulation threads.
use std::{
    path::PathBuf,
    sync::mpsc,
    thread::{self, JoinHandle},
    time::Duration,
};
pub struct Timeline {
    stop: mpsc::Sender<()>,
    worker: JoinHandle<Result<(), String>>,
}
impl Timeline {
    pub fn start() -> Result<Option<Self>, String> {
        let Some(path) = std::env::var_os("NIXE_TRACE") else {
            return Ok(None);
        };
        let path = PathBuf::from(path);
        let trigger = std::env::var_os("NIXE_TRACE_TRIGGER").map(PathBuf::from);
        let (stop, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("nixe-trace-control".into())
            .spawn(move || {
                let mut capture = None;
                loop {
                    let recording = trigger.as_ref().is_none_or(|path| path.exists());
                    if recording && capture.is_none() {
                        capture = Some(nixe_trace::Capture::start().map_err(|e| e.to_string())?);
                    }
                    if receiver.recv_timeout(Duration::from_millis(20))
                        != Err(mpsc::RecvTimeoutError::Timeout)
                    {
                        break;
                    }
                    if !recording && capture.is_some() {
                        break;
                    }
                }
                if let Some(capture) = capture {
                    capture.finish(&path).map_err(|e| e.to_string())?;
                }
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        Ok(Some(Self { stop, worker }))
    }
    pub fn finish(self) -> Result<(), String> {
        let _ = self.stop.send(());
        self.worker
            .join()
            .map_err(|_| "trace writer panicked".to_owned())?
    }
}
