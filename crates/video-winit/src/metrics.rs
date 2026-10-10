//! Bounded intervals between host presentations of fresh guest frames.
//! 100 us bins, last bin is overflow (>= 204.8 ms); not GPU render time.
use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::Instant,
};
const BINS: usize = 2049;
static HISTOGRAM: [AtomicU64; BINS] = [const { AtomicU64::new(0) }; BINS];
static FRAMES: AtomicU64 = AtomicU64::new(0);
thread_local! { static LAST: Cell<Option<Instant>> = const { Cell::new(None) }; }
pub(crate) fn presented() {
    let now = Instant::now();
    FRAMES.fetch_add(1, Relaxed);
    LAST.with(|last| {
        if let Some(previous) = last.replace(Some(now)) {
            let bin =
                (now.duration_since(previous).as_micros() / 100).min((BINS - 1) as u128) as usize;
            HISTOGRAM[bin].fetch_add(1, Relaxed);
        }
    });
}
pub fn snapshot() -> Vec<(&'static str, u64)> {
    vec![("PresentedGuestFrames", FRAMES.load(Relaxed))]
}
pub fn histogram() -> Vec<u64> {
    HISTOGRAM.iter().map(|c| c.load(Relaxed)).collect()
}
