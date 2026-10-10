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
    // Sampled at direct write-fault resolver entry; another worker may
    // already have repaired a captured native fault.
    WriteFaultCleanUnobserved,
    WriteFaultCleanObserved,
    WriteFaultCpuUnobserved,
    WriteFaultCpuObserved,
    WriteFaultDeviceOwned,
    WriteFaultInvalid,
    ObserverArms,
    // CPU-to-Clean preparation for either read-only or writable device use.
    DeviceReadPreparations,
    DeviceWritePublications,
    CpuReadbacks,
    DirectAliasRegistrations,
}

#[cfg(feature = "performance-counters")]
static COUNTERS: [std::sync::atomic::AtomicU64; 40] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 40];

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
        "WriteFaultCleanUnobserved",
        "WriteFaultCleanObserved",
        "WriteFaultCpuUnobserved",
        "WriteFaultCpuObserved",
        "WriteFaultDeviceOwned",
        "WriteFaultInvalid",
        "ObserverArms",
        "DeviceReadPreparations",
        "DeviceWritePublications",
        "CpuReadbacks",
        "DirectAliasRegistrations",
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

// Detailed page attribution is enabled only by the diagnostic CLI capture.
// It is never allocated or consulted in normal builds.
#[cfg(feature = "performance-counters")]
type PageTracking = std::collections::BTreeMap<crate::CanonicalPageId, [u64; 11]>;
#[cfg(feature = "performance-counters")]
static PAGE_TRACKING: std::sync::Mutex<Option<PageTracking>> = std::sync::Mutex::new(None);

#[cfg(feature = "performance-counters")]
pub fn start_page_tracking() {
    *PAGE_TRACKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        Some(std::collections::BTreeMap::new());
}

#[inline]
pub(crate) fn record_page(page: crate::CanonicalPageId, counter: Counter) {
    record(counter, 1);
    #[cfg(feature = "performance-counters")]
    if let Some(pages) = PAGE_TRACKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
    {
        pages.entry(page).or_insert([0; 11])
            [counter as usize - Counter::WriteFaultCleanUnobserved as usize] += 1;
    }
    #[cfg(not(feature = "performance-counters"))]
    let _ = page;
}

#[cfg(feature = "performance-counters")]
pub fn write_page_tracking(path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut output = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(output, "page")?;
    for (name, _) in snapshot()
        .into_iter()
        .skip(Counter::WriteFaultCleanUnobserved as usize)
    {
        write!(output, ",{name}")?;
    }
    writeln!(output)?;
    let mut tracking = PAGE_TRACKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(pages) = tracking.take() {
        for (page, counts) in pages {
            write!(output, "{page}")?;
            for count in counts {
                write!(output, ",{count}")?;
            }
            writeln!(output)?;
        }
    }
    output.flush()
}
