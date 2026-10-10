//! Active playback shortages, not silence inferred from PCM amplitude.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
static CALLBACKS: AtomicU64 = AtomicU64::new(0);
static SHORT_CALLBACKS: AtomicU64 = AtomicU64::new(0);
static MISSING_SAMPLES: AtomicU64 = AtomicU64::new(0);
pub(crate) fn rendered(requested: usize, written: usize) {
    CALLBACKS.fetch_add(1, Relaxed);
    if written < requested {
        SHORT_CALLBACKS.fetch_add(1, Relaxed);
        MISSING_SAMPLES.fetch_add((requested - written) as u64, Relaxed);
    }
}
pub fn snapshot() -> Vec<(&'static str, u64)> {
    vec![
        ("AudioActiveCallbacks", CALLBACKS.load(Relaxed)),
        ("AudioShortCallbacks", SHORT_CALLBACKS.load(Relaxed)),
        ("AudioMissingSamples", MISSING_SAMPLES.load(Relaxed)),
    ]
}
