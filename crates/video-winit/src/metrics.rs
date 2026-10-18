//! Bounded intervals between host presentations of fresh guest frames.
//! 100 us bins, last bin is overflow (>= 204.8 ms); not GPU render time.
use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::Instant,
};
const BINS: usize = 2049;
static HISTOGRAM: [AtomicU64; BINS] = [const { AtomicU64::new(0) }; BINS];
static FRAMES: AtomicU64 = AtomicU64::new(0);
thread_local! { static LAST: Cell<Option<Instant>> = const { Cell::new(None) }; }
thread_local! { static CAPTURE: RefCell<Option<FrameCapture>> = const { RefCell::new(None) }; }

struct FrameCapture {
    output: BufWriter<File>,
    started: Instant,
    error: Option<io::Error>,
}

/// Raw fresh-frame intervals for opt-in development captures. This module is
/// excluded entirely from builds without `performance-counters`.
pub fn start_frame_capture(path: &Path, started: Instant) -> io::Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    writeln!(output, "seconds,frame,interval_ns")?;
    CAPTURE.with(|capture| {
        *capture.borrow_mut() = Some(FrameCapture {
            output,
            started,
            error: None,
        });
    });
    Ok(())
}

pub fn finish_frame_capture() -> io::Result<()> {
    CAPTURE.with(|capture| {
        let Some(mut capture) = capture.borrow_mut().take() else {
            return Ok(());
        };
        if let Some(error) = capture.error {
            return Err(error);
        }
        capture.output.flush()
    })
}

pub(crate) fn presented_at(now: Instant) {
    let frame = FRAMES.fetch_add(1, Relaxed) + 1;
    let interval = LAST.with(|last| {
        let interval = last
            .replace(Some(now))
            .map(|previous| now.duration_since(previous));
        if let Some(interval) = interval {
            let bin = (interval.as_micros() / 100).min((BINS - 1) as u128) as usize;
            HISTOGRAM[bin].fetch_add(1, Relaxed);
        }
        interval
    });
    CAPTURE.with(|capture| {
        let mut capture = capture.borrow_mut();
        if let Some(capture) = capture.as_mut()
            && capture.error.is_none()
        {
            let result = match interval {
                Some(interval) => writeln!(
                    capture.output,
                    "{:.9},{frame},{}",
                    now.duration_since(capture.started).as_secs_f64(),
                    interval.as_nanos()
                ),
                None => writeln!(
                    capture.output,
                    "{:.9},{frame},",
                    now.duration_since(capture.started).as_secs_f64()
                ),
            };
            if let Err(error) = result {
                capture.error = Some(error);
            }
        }
    });
}
pub fn snapshot() -> Vec<(&'static str, u64)> {
    vec![("PresentedGuestFrames", FRAMES.load(Relaxed))]
}
pub fn histogram() -> Vec<u64> {
    HISTOGRAM.iter().map(|c| c.load(Relaxed)).collect()
}
