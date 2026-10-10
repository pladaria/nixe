//! Opt-in aggregate counters for performance investigations.
//! Disabled builds inline recording away; no production atomics or timers.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Counter {
    RangeChecks,
    CpuDependencyChecks,
    PageVisibilityChecks,
    OwnershipPublications,
    PageOwnershipUpdates,
    DirtyGroupChecks,
    DirtyPageChecks,
    TrackingLocks,
    SnapshotRequestedBytes,
    SnapshotCopiedBytes,
    StreamingComparedBytes,
    RangeEqualityChecks,
    RangeStructuralComparisons,
    RangeEqualityInputSegments,
    GateSharedAcquisitions,
    GateExclusiveAcquisitions,
    GateSharedAdmissionNanoseconds,
    GateExclusiveAdmissionNanoseconds,
    GateSafepointNotifications,
    DirectProtectionCalls,
    DirectProtectionBytes,
    BackingAllocatedBytes,
    BackingReleasedBytes,
    BackingAllocationCount,
    BackingLiveBytes,
    BackingPeakLiveBytes,
    BackingPeakStoreOffset,
    BackingReservedBytes,
    BackingMappedBytes,
}

#[cfg(feature = "performance-counters")]
static COUNTERS: [std::sync::atomic::AtomicU64; 29] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 29];

#[inline]
pub fn record(counter: Counter, amount: u64) {
    #[cfg(feature = "performance-counters")]
    COUNTERS[counter as usize].fetch_add(amount, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "performance-counters"))]
    let _ = (counter, amount);
}

#[cfg(feature = "performance-counters")]
#[must_use]
pub fn snapshot() -> Vec<(&'static str, u64)> {
    const NAMES: &[&str] = &[
        "RangeChecks",
        "CpuDependencyChecks",
        "PageVisibilityChecks",
        "OwnershipPublications",
        "PageOwnershipUpdates",
        "DirtyGroupChecks",
        "DirtyPageChecks",
        "TrackingLocks",
        "SnapshotRequestedBytes",
        "SnapshotCopiedBytes",
        "StreamingComparedBytes",
        "RangeEqualityChecks",
        "RangeStructuralComparisons",
        "RangeEqualityInputSegments",
        "GateSharedAcquisitions",
        "GateExclusiveAcquisitions",
        "GateSharedAdmissionNanoseconds",
        "GateExclusiveAdmissionNanoseconds",
        "GateSafepointNotifications",
        "DirectProtectionCalls",
        "DirectProtectionBytes",
        "BackingAllocatedBytes",
        "BackingReleasedBytes",
        "BackingAllocationCount",
        "BackingLiveBytes",
        "BackingPeakLiveBytes",
        "BackingPeakStoreOffset",
        "BackingReservedBytes",
        "BackingMappedBytes",
    ];
    NAMES
        .iter()
        .zip(&COUNTERS)
        .map(|(name, counter)| (*name, counter.load(std::sync::atomic::Ordering::Relaxed)))
        .collect()
}

/// Time a whole boundary, including mutex admission, only in diagnostic builds.
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
pub(crate) fn allocate_backing(bytes: u64, end: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    record(Counter::BackingAllocatedBytes, bytes);
    record(Counter::BackingAllocationCount, 1);
    let live = COUNTERS[Counter::BackingLiveBytes as usize].fetch_add(bytes, Relaxed) + bytes;
    COUNTERS[Counter::BackingPeakLiveBytes as usize].fetch_max(live, Relaxed);
    COUNTERS[Counter::BackingPeakStoreOffset as usize].fetch_max(end, Relaxed);
}
#[cfg(feature = "performance-counters")]
pub(crate) fn subtract(counter: Counter, bytes: u64) {
    COUNTERS[counter as usize].fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
}
