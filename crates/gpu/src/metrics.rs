//! Opt-in aggregate counters for performance investigations.
//! Disabled builds inline recording away; no production atomics or timers.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Counter {
    ResolvedDependencies,
    SubmissionOperations,
    DependencyOccurrences,
    AccessIntervals,
    AccessPreparationNanoseconds,
    BufferUploadedBytes,
    ImageLinearizedBytes,
    ImageUploadedBytes,
    ImageRequestedBytes,
    ImageSnapshottedBytes,
    AliasComparisons,
    AliasInputSpans,
    CanonicalBackingComparisons,
    BufferReadbackBytes,
    ImageReadbackBytes,
    DeviceReadbackCopies,
    BackendSubmissions,
    BackendSubmissionNanoseconds,
    BackendCompletionWaits,
    BackendCompletionWaitNanoseconds,
    FrontendSubmissions,
    FrontendLoweringNanoseconds,
    InlineWriteSegmentBreaks,
    HostQueueSubmissions,
    HostQueueGpuNanoseconds,
    HostQueueTimedSubmissions,
    HostQueueWaitNanoseconds,
    HostQueueWaits,
    HostQueueSubmitNanoseconds,
    ShaderTranslationMisses,
    ShaderTranslationNanoseconds,
    HostPipelineCompilations,
    HostPipelineCompilationNanoseconds,
    InputSnapshotNanoseconds,
    ImageLinearizationNanoseconds,
    ImageTransferStagingNanoseconds,
    NativeQueuedChunks,
    NativeChunkCommands,
    NativeChunkHandoffNanoseconds,
    NativeChunkAdmissionNanoseconds,
}

#[cfg(feature = "performance-counters")]
static COUNTERS: [std::sync::atomic::AtomicU64;
    Counter::NativeChunkAdmissionNanoseconds as usize + 1] =
    [const { std::sync::atomic::AtomicU64::new(0) };
        Counter::NativeChunkAdmissionNanoseconds as usize + 1];

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
        "ResolvedDependencies",
        "SubmissionOperations",
        "DependencyOccurrences",
        "AccessIntervals",
        "AccessPreparationNanoseconds",
        "BufferUploadedBytes",
        "ImageLinearizedBytes",
        "ImageUploadedBytes",
        "ImageRequestedBytes",
        "ImageSnapshottedBytes",
        "AliasComparisons",
        "AliasInputSpans",
        "CanonicalBackingComparisons",
        "BufferReadbackBytes",
        "ImageReadbackBytes",
        "DeviceReadbackCopies",
        "BackendSubmissions",
        "BackendSubmissionNanoseconds",
        "BackendCompletionWaits",
        "BackendCompletionWaitNanoseconds",
        "FrontendSubmissions",
        "FrontendLoweringNanoseconds",
        "InlineWriteSegmentBreaks",
        "HostQueueSubmissions",
        "HostQueueGpuNanoseconds",
        "HostQueueTimedSubmissions",
        "HostQueueWaitNanoseconds",
        "HostQueueWaits",
        "HostQueueSubmitNanoseconds",
        "ShaderTranslationMisses",
        "ShaderTranslationNanoseconds",
        "HostPipelineCompilations",
        "HostPipelineCompilationNanoseconds",
        "InputSnapshotNanoseconds",
        "ImageLinearizationNanoseconds",
        "ImageTransferStagingNanoseconds",
        "NativeQueuedChunks",
        "NativeChunkCommands",
        "NativeChunkHandoffNanoseconds",
        "NativeChunkAdmissionNanoseconds",
    ];
    NAMES
        .iter()
        .zip(&COUNTERS)
        .map(|(name, counter)| (*name, counter.load(std::sync::atomic::Ordering::Relaxed)))
        .collect()
}
