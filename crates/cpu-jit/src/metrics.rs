//! Optional host-side native execution measurements.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Counter {
    NativeEntries,
    NativeStateBytes,
    NativeStateNanoseconds,
    NativeRunNanoseconds,
    FaultDispatches,
    CodeFetches,
    ReadCallbacks,
    WriteCallbacks,
    AtomicCallbacks,
    InvalidationCalls,
    InvalidationBytes,
    InvalidationNanoseconds,
    ContextLoads,
    ContextSaves,
    InternalHaltResumes,
    AdmissionNanoseconds,
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
        "JitNativeStateBytes",
        "JitNativeStateNanoseconds",
        "JitNativeRunNanoseconds",
        "JitFaultDispatches",
        "JitCodeFetches",
        "JitReadCallbacks",
        "JitWriteCallbacks",
        "JitAtomicCallbacks",
        "JitInvalidationCalls",
        "JitInvalidationBytes",
        "JitInvalidationNanoseconds",
        "JitContextLoads",
        "JitContextSaves",
        "JitInternalHaltResumes",
        "JitAdmissionNanoseconds",
    ];
    let values: [u64; 16] =
        std::array::from_fn(|index| COUNTERS[index].load(std::sync::atomic::Ordering::Relaxed));
    let total = values[Counter::ReadCallbacks as usize]
        + values[Counter::WriteCallbacks as usize]
        + values[Counter::AtomicCallbacks as usize];
    let mut result = Vec::with_capacity(NAMES.len() + 1);
    result.extend(NAMES.iter().copied().zip(values));
    result.push(("JitMemoryCallbacks", total));
    result
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
