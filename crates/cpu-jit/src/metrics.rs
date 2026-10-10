//! Bounded host-side measurements; guest ticks are Dynarmic budget units.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Counter {
    NativeEntries,
    GuestTicks,
    // Kept for baseline comparisons; direct lending performs no Rust copying.
    RustStateBytes,
    NativeStateBytes,
    // No Rust marshalling timer remains after T03; this category stays zero.
    RustStateNanoseconds,
    NativeStateNanoseconds,
    NativeRunNanoseconds,
    FaultDispatches,
    CodeFetches,
    MemoryCallbacks,
    InvalidationCalls,
    InvalidationBytes,
    FullInvalidations,
    InvalidationNanoseconds,
    ContextLoads,
    ContextSaves,
}
#[cfg(feature = "performance-counters")]
static COUNTERS: [std::sync::atomic::AtomicU64; 16] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 16];
#[inline]
pub fn record(counter: Counter, amount: u64) {
    #[cfg(feature = "performance-counters")]
    COUNTERS[counter as usize].fetch_add(amount, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "performance-counters"))]
    let _ = (counter, amount);
}
#[cfg(feature = "performance-counters")]
pub fn snapshot() -> Vec<(&'static str, u64)> {
    const NAMES: &[&str] = &[
        "JitNativeEntries",
        "JitGuestTicks",
        "JitRustStateBytes",
        "JitNativeStateBytes",
        "JitRustStateNanoseconds",
        "JitNativeStateNanoseconds",
        "JitNativeRunNanoseconds",
        "JitFaultDispatches",
        "JitCodeFetches",
        "JitMemoryCallbacks",
        "JitInvalidationCalls",
        "JitInvalidationBytes",
        "JitFullInvalidations",
        "JitInvalidationNanoseconds",
        "JitContextLoads",
        "JitContextSaves",
    ];
    NAMES
        .iter()
        .zip(&COUNTERS)
        .map(|(n, c)| (*n, c.load(std::sync::atomic::Ordering::Relaxed)))
        .collect()
}
// Only present in diagnostic builds.
pub struct Timer {
    #[cfg(feature = "performance-counters")]
    start: std::time::Instant,
    #[cfg(feature = "performance-counters")]
    counter: Counter,
}
impl Timer {
    #[inline]
    pub fn new(counter: Counter) -> Self {
        #[cfg(not(feature = "performance-counters"))]
        let _ = counter;
        Self {
            #[cfg(feature = "performance-counters")]
            start: std::time::Instant::now(),
            #[cfg(feature = "performance-counters")]
            counter,
        }
    }
}
impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "performance-counters")]
        record(
            self.counter,
            self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
    }
}

#[cfg(feature = "performance-counters")]
#[unsafe(no_mangle)]
extern "C" fn nixe_jit_measure_native_state(nanoseconds: u64, bytes: u64) {
    record(Counter::NativeStateNanoseconds, nanoseconds);
    // Architectural field payload, excluding C ABI padding and compiler spills.
    record(Counter::NativeStateBytes, bytes);
}
