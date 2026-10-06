//! Optional aggregate counter capture, excluded from production builds.
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub(super) struct Capture {
    stop: mpsc::Sender<()>,
    worker: thread::JoinHandle<io::Result<()>>,
}
impl Capture {
    pub(super) fn start() -> io::Result<Option<Self>> {
        let Some(path) = std::env::var_os("NIXE_PERFORMANCE_COUNTERS") else {
            return Ok(None);
        };
        let mut output = BufWriter::new(File::create(path)?);
        let (stop, receiver) = mpsc::channel();
        let counters = || {
            let mut values = nixe_memory::metrics::snapshot();
            values.extend(nixe_gpu::metrics::snapshot());
            values
        };
        write!(output, "seconds")?;
        for (name, _) in counters() {
            write!(output, ",{name}")?;
        }
        writeln!(output)?;
        let worker = thread::spawn(move || {
            let started = Instant::now();
            loop {
                let stop = !matches!(
                    receiver.recv_timeout(Duration::from_secs(1)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );
                write!(output, "{:.6}", started.elapsed().as_secs_f64())?;
                for (_, value) in counters() {
                    write!(output, ",{value}")?;
                }
                writeln!(output)?;
                if stop {
                    return output.flush();
                }
            }
        });
        Ok(Some(Self { stop, worker }))
    }
    pub(super) fn finish(self) -> io::Result<()> {
        let _ = self.stop.send(());
        self.worker
            .join()
            .map_err(|_| io::Error::other("performance capture worker panicked"))?
    }
}
