//! Checked, retained and pointer-free canonical backing ranges.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use crate::{
    AddressSpaceId, CanonicalBackingPage, CanonicalPageError, CanonicalPageId,
    DeviceAccessDeclaration, GuestVirtualAddress, MappingGeneration, MemoryPermissions,
    VisibilityCoordinator, VisibilityError, VisibilityState,
};

mod mapping;
use mapping::ByteMapping;
pub use mapping::{CanonicalByteLayout, CanonicalByteTile};
use smallvec::SmallVec;

type CanonicalPageReader<'a> =
    dyn FnMut(&CanonicalBackingPage, usize, &mut [u8]) -> Result<(), CanonicalPageError> + 'a;

/// One contiguous segment of a translated canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBackingSegment {
    backing: CanonicalBackingPage,
    offset: u64,
    size: u64,
    permissions: MemoryPermissions,
    mapping_generation: MappingGeneration,
}

impl CanonicalBackingSegment {
    pub(crate) const fn backing(&self) -> &CanonicalBackingPage {
        &self.backing
    }

    /// Creates a checked segment and retains its canonical page.
    pub fn new(
        backing: CanonicalBackingPage,
        offset: u64,
        size: u64,
        permissions: MemoryPermissions,
        mapping_generation: MappingGeneration,
    ) -> Result<Self, CanonicalRangeError> {
        Self::new_captured(backing, offset, size, permissions, mapping_generation)
    }

    fn snapshot(
        backing: CanonicalBackingPage,
        offset: u64,
        size: u64,
        permissions: MemoryPermissions,
        mapping_generation: MappingGeneration,
    ) -> Result<Self, CanonicalRangeError> {
        Self::new_captured(backing, offset, size, permissions, mapping_generation)
    }

    fn new_captured(
        backing: CanonicalBackingPage,
        offset: u64,
        size: u64,
        permissions: MemoryPermissions,
        mapping_generation: MappingGeneration,
    ) -> Result<Self, CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::SegmentOverflow)?;
        if size == 0 || end > backing.size() as u64 {
            return Err(CanonicalRangeError::InvalidSegmentBounds);
        }
        Ok(Self {
            backing,
            offset,
            size,
            permissions,
            mapping_generation,
        })
    }

    /// Returns the stable page identity, never a host pointer.
    #[must_use]
    pub fn page(&self) -> CanonicalPageId {
        self.backing.identity()
    }

    /// Returns the first byte within the canonical page.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the number of bytes in this segment.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Retains a checked subrange of this segment in constant time, preserving
    /// the captured page, permissions and mapping generation.
    pub fn snapshot_subrange(&self, offset: u64, size: u64) -> Result<Self, CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::RangeOverflow)?;
        if size == 0 || end > self.size {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        Ok(Self {
            offset: self.offset + offset,
            size,
            ..self.clone()
        })
    }

    /// Returns whether this segment reaches the end of its canonical page.
    #[must_use]
    pub fn ends_at_page_boundary(&self) -> bool {
        self.offset + self.size == self.backing.size() as u64
    }

    /// Returns the permissions of the CPU mapping used for translation.
    #[must_use]
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    /// Returns the mapping generation captured during translation.
    #[must_use]
    pub const fn mapping_generation(&self) -> MappingGeneration {
        self.mapping_generation
    }

    /// Returns the conservative visibility authority shared by all aliases.
    #[must_use]
    pub fn visibility_state(&self) -> VisibilityState {
        self.backing.visibility_state()
    }
}

/// A validated logical byte range represented by retained page segments.
#[derive(Clone, Debug)]
pub struct CanonicalBackingRange {
    layout: Arc<CanonicalRangeLayout>,
    size: u64,
}

/// Immutable physical byte mapping identity for representation lookup. Capture
/// permissions, generations and CPU/GPU content epochs are deliberately absent.
/// Different expressions of equivalent permutations may remain distinct keys.
#[derive(Clone, Debug)]
pub struct CanonicalByteMappingKey(CanonicalBackingRange);
impl PartialEq for CanonicalByteMappingKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_byte_mapping(&other.0)
    }
}
impl Eq for CanonicalByteMappingKey {}
impl Hash for CanonicalByteMappingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.byte_mapping_hash());
    }
}

type CanonicalPageSegments = BTreeMap<CanonicalPageId, Vec<(u64, u64, u64)>>;

#[derive(Debug)]
struct CanonicalRangeLayout {
    topology: std::sync::OnceLock<CanonicalSegmentTopology>,
    mapping: Option<ByteMapping>,
    page_segments: std::sync::OnceLock<CanonicalPageSegments>,
    segment_runs: std::sync::OnceLock<Box<[CanonicalSegmentRun]>>,
    byte_mapping_hash: std::sync::OnceLock<u64>,
    // Retained physical pages are independent of the logical byte permutation.
    pages: Box<[CanonicalBackingPage]>,
    // Gate identity order deduplicates stores sharing one non-reentrant gate.
    gates: Box<[crate::ExecutionGate]>,
    owner: Mutex<Option<Arc<crate::backing::RangeDeviceOwner>>>,
    visibility_epoch: std::sync::OnceLock<Arc<AtomicU64>>,
    clean_read_epoch: AtomicU64,
    cpu_summary: Mutex<std::sync::Weak<CpuWriteSummary>>,
}

#[derive(Debug)]
struct CanonicalSegmentTopology {
    segments: Box<[CanonicalBackingSegment]>,
    segment_ends: Box<[u64]>,
}

/// Lossless run encoding of the ordered segment sequence. Segments only share
/// a run when every field except the consecutive physical page number matches.
/// In particular, aliases, boundaries, permissions and mapping generations
/// remain part of equality; this is not a fingerprint of byte coverage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CanonicalSegmentRun {
    first_page: CanonicalPageId,
    last_page: CanonicalPageId,
    offset: u64,
    size: u64,
    permissions: MemoryPermissions,
    mapping_generation: MappingGeneration,
}

impl PartialEq for CanonicalBackingRange {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "performance-counters")]
        {
            crate::metrics::record(crate::metrics::Counter::RangeEqualityChecks, 1);
            if self.size == other.size && !Arc::ptr_eq(&self.layout, &other.layout) {
                crate::metrics::record(crate::metrics::Counter::RangeStructuralComparisons, 1);
                crate::metrics::record(
                    crate::metrics::Counter::RangeEqualityInputSegments,
                    (self.layout.topology.get().map_or(0, |t| t.segments.len())
                        + other.layout.topology.get().map_or(0, |t| t.segments.len()))
                        as u64,
                );
            }
        }
        self.size == other.size
            && (Arc::ptr_eq(&self.layout, &other.layout)
                || match (&self.layout.mapping, &other.layout.mapping) {
                    (Some(a), Some(b)) => {
                        a == b || self.layout.segment_runs() == other.layout.segment_runs()
                    }
                    _ => self.layout.segment_runs() == other.layout.segment_runs(),
                })
    }
}

impl Eq for CanonicalBackingRange {}

impl CanonicalRangeLayout {
    fn topology(&self) -> &CanonicalSegmentTopology {
        self.topology.get_or_init(|| {
            let mapping = self.mapping.as_ref().expect("deferred byte topology");
            let size = match mapping {
                ByteMapping::Rect { layout, .. } => {
                    u64::from(layout.columns)
                        * u64::from(layout.rows)
                        * u64::from(layout.element_bytes)
                }
                ByteMapping::Concat(ranges) => ranges.iter().map(CanonicalBackingRange::size).sum(),
                ByteMapping::Slice { size, .. } => *size,
            };
            let mut segments = Vec::new();
            mapping
                .snapshot(0, size, &mut segments)
                .expect("validated byte mapping");
            let mut end = 0;
            let segment_ends = segments
                .iter()
                .map(|s| {
                    end += s.size();
                    end
                })
                .collect();
            CanonicalSegmentTopology {
                segments: segments.into(),
                segment_ends,
            }
        })
    }

    /// Structural comparisons need a lossless encoding; shared topology uses
    /// pointer identity and never allocates duplicate fragment metadata.
    fn segment_runs(&self) -> &[CanonicalSegmentRun] {
        self.segment_runs.get_or_init(|| {
            let mut runs: Vec<CanonicalSegmentRun> = Vec::new();
            for segment in &self.topology().segments {
                let page = segment.page();
                if let Some(run) = runs.last_mut()
                    && run.last_page.store() == page.store()
                    && run.last_page.page().get().checked_add(1) == Some(page.page().get())
                    && run.offset == segment.offset
                    && run.size == segment.size
                    && run.permissions == segment.permissions
                    && run.mapping_generation == segment.mapping_generation
                {
                    run.last_page = page;
                } else {
                    runs.push(CanonicalSegmentRun {
                        first_page: page,
                        last_page: page,
                        offset: segment.offset,
                        size: segment.size,
                        permissions: segment.permissions,
                        mapping_generation: segment.mapping_generation,
                    });
                }
            }
            runs.into_boxed_slice()
        })
    }
}

struct CpuWriteDependencyPage {
    page: CanonicalBackingPage,
    observed_epoch: AtomicU64,
}

pub(crate) const CPU_WRITE_GROUP_PAGES: usize = 64;

pub(crate) struct CpuWriteSummary {
    epoch: AtomicU64,
    groups: Box<[AtomicU64]>,
}

impl CpuWriteSummary {
    pub(crate) fn publish(&self, group: usize) {
        // Saturation remains permanently dirty rather than allowing epoch
        // reuse to make an older observation current.
        let advance = |epoch: &AtomicU64| {
            let _ = epoch.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                Some(value.saturating_add(1))
            });
        };
        advance(&self.groups[group]);
        advance(&self.epoch);
    }
}

struct CanonicalCpuWriteDependencyInner {
    domains: Box<[CanonicalBackingRange]>,
    pages: Box<[CpuWriteDependencyPage]>,
    summary: Arc<CpuWriteSummary>,
    observed_summary: AtomicU64,
    observed_groups: Box<[AtomicU64]>,
    streaming_pages: AtomicU64,
    clean_since_snapshot: AtomicBool,
    adaptive: Mutex<AdaptiveCpuTracking>,
}

/// Compact physical coverage, independent of logical ordering, permissions,
/// generations and byte offsets within conservatively tracked physical pages.
#[derive(Clone, Eq, Hash, PartialEq)]
struct PageCoverage(Box<[(CanonicalPageId, CanonicalPageId)]>);
impl PageCoverage {
    fn from_sorted(pages: impl IntoIterator<Item = CanonicalPageId>) -> Self {
        let mut runs: Vec<(CanonicalPageId, CanonicalPageId)> = Vec::new();
        for page in pages {
            if let Some((_, end)) = runs.last_mut()
                && end.store() == page.store()
                && end.page().get().checked_add(1) == Some(page.page().get())
            {
                *end = page;
            } else {
                runs.push((page, page));
            }
        }
        Self(runs.into_boxed_slice())
    }
}

/// Weak interning retains no pages, allocations or independent visibility
/// authority. Every summary is derived from the existing canonical page state.
struct SummaryInterner<T> {
    entries: std::collections::HashMap<PageCoverage, std::sync::Weak<T>>,
    next_collection: usize,
}
impl<T> Default for SummaryInterner<T> {
    fn default() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            next_collection: 1024,
        }
    }
}
impl<T> SummaryInterner<T> {
    fn find(&self, coverage: &PageCoverage) -> Option<Arc<T>> {
        self.entries
            .get(coverage)
            .and_then(std::sync::Weak::upgrade)
    }
    fn insert(&mut self, coverage: PageCoverage, summary: &Arc<T>) {
        if self.entries.len() >= self.next_collection {
            self.entries.retain(|_, entry| entry.strong_count() != 0);
            self.next_collection = self.entries.len().saturating_add(1024);
        }
        self.entries.insert(coverage, Arc::downgrade(summary));
    }
}
static CPU_SUMMARIES: std::sync::LazyLock<Mutex<SummaryInterner<CpuWriteSummary>>> =
    std::sync::LazyLock::new(|| Mutex::new(SummaryInterner::default()));
static VISIBILITY_SUMMARIES: std::sync::LazyLock<Mutex<SummaryInterner<AtomicU64>>> =
    std::sync::LazyLock::new(|| Mutex::new(SummaryInterner::default()));

const STREAMING_DIRTY_THRESHOLD: u8 = 5;
const STREAMING_QUIET_THRESHOLD: u8 = 3;

struct AdaptiveCpuTracking {
    streaks: Box<[u8]>,
    streaming: BTreeMap<usize, StreamingCpuPage>,
}

struct StreamingCpuPage {
    bytes: Box<[u8]>,
    quiet: u8,
}

/// Logical offsets and copied bytes from a canonical dependency snapshot.
pub type CanonicalByteSnapshots = Vec<(u64, Box<[u8]>)>;

/// One resource's requested CPU-byte sampling boundary.
/// A batch does not share or refresh another resource's dirty observation.
pub struct CpuWriteSnapshotRequest<'a> {
    pub dependency: &'a CanonicalCpuWriteDependency,
    pub range: &'a CanonicalBackingRange,
    pub selection: CpuWriteSnapshotSelection,
    pub alignment: u64,
}

/// Cloneable page-granular observation of CPU writes.
///
/// Capturing establishes a read-only baseline through every direct alias.
/// The first later CPU write advances the physical page's dirty epoch; no
/// subsequent store in that dirty epoch performs observer publication.
#[derive(Clone)]
pub struct CanonicalCpuWriteDependency {
    inner: Arc<CanonicalCpuWriteDependencyInner>,
}

pub(crate) struct CpuWriteObservation {
    pub(crate) summary: Arc<CpuWriteSummary>,
    pub(crate) epochs: Vec<u64>,
    pub(crate) summary_epoch: u64,
    pub(crate) group_epochs: Vec<u64>,
}

impl CpuWriteObservation {
    pub(crate) fn new(summary: Arc<CpuWriteSummary>, epochs: Vec<u64>) -> Self {
        let summary_epoch = summary.epoch.load(Ordering::Acquire);
        let group_epochs = summary
            .groups
            .iter()
            .map(|epoch| epoch.load(Ordering::Acquire))
            .collect();
        Self {
            summary,
            epochs,
            summary_epoch,
            group_epochs,
        }
    }
}

impl CanonicalCpuWriteDependency {
    /// Captures and arms every distinct physical page in one range.
    /// Call without a native epoch, memory lease or cache lock.
    pub fn capture(range: &CanonicalBackingRange) -> Result<Self, CanonicalRangeAccessError> {
        Self::capture_ranges([range])
    }

    /// Captures several ranges as one page-granular dependency domain.
    /// Page locks and alias protection stabilize the physical regions under
    /// shared mapping admission. Hosts without the native write barrier drain
    /// CPU execution when establishing protection. Call without an execution
    /// lease or cache lock. Empty input is an error.
    pub fn capture_ranges<'a>(
        ranges: impl IntoIterator<Item = &'a CanonicalBackingRange>,
    ) -> Result<Self, CanonicalRangeAccessError> {
        let domains = ranges.into_iter().cloned().collect::<Box<[_]>>();
        let mut execution_gates = BTreeMap::new();
        let mut pages = BTreeMap::new();
        for range in &domains {
            for gate in range.execution_gates() {
                execution_gates
                    .entry(gate.identity())
                    .or_insert_with(|| gate.clone());
            }
            for page in range.pages() {
                pages.entry(page.identity()).or_insert_with(|| page.clone());
            }
        }
        if pages.is_empty() {
            return Err(CanonicalRangeAccessError::IncompleteRange);
        }
        let execution_gates = execution_gates.into_values().collect::<Vec<_>>();
        let group_count = pages.len().div_ceil(CPU_WRITE_GROUP_PAGES);
        // Consumers of identical retained topology share one page observer,
        // while their baseline epochs remain independent. Refreshing content
        // never makes an older opaque interpretation current again.
        let page_list = pages.into_values().collect::<Vec<_>>();
        let mut select_summary = || {
            let existing = if domains.len() == 1 {
                domains[0]
                    .layout
                    .cpu_summary
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .upgrade()
            } else {
                None
            };
            let (summary, shared) = if let Some(summary) = existing {
                (summary, true)
            } else {
                let coverage =
                    PageCoverage::from_sorted(page_list.iter().map(CanonicalBackingPage::identity));
                let mut interner = CPU_SUMMARIES
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(summary) = interner.find(&coverage) {
                    (summary, true)
                } else {
                    let summary = Arc::new(CpuWriteSummary {
                        epoch: AtomicU64::new(0),
                        groups: (0..group_count).map(|_| AtomicU64::new(0)).collect(),
                    });
                    interner.insert(coverage, &summary);
                    (summary, false)
                }
            };
            if domains.len() == 1 {
                *domains[0]
                    .layout
                    .cpu_summary
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::downgrade(&summary);
            }
            (summary, shared)
        };
        let observation = {
            let _leases: SmallVec<[_; 2]> = execution_gates
                .iter()
                .map(crate::ExecutionGate::acquire_shared)
                .collect();
            crate::backing::observe_cpu_pages(&page_list, &mut select_summary)
                .map_err(CanonicalRangeAccessError::Backing)?
        };
        let pages = page_list
            .into_iter()
            .zip(observation.epochs)
            .map(|(page, epoch)| CpuWriteDependencyPage {
                page,
                observed_epoch: AtomicU64::new(epoch),
            })
            .collect::<Box<[_]>>();
        Ok(Self {
            inner: Arc::new(CanonicalCpuWriteDependencyInner {
                domains,
                adaptive: Mutex::new(AdaptiveCpuTracking {
                    streaks: vec![0; pages.len()].into_boxed_slice(),
                    streaming: BTreeMap::new(),
                }),
                pages,
                observed_summary: AtomicU64::new(observation.summary_epoch),
                observed_groups: observation
                    .group_epochs
                    .into_iter()
                    .map(AtomicU64::new)
                    .collect(),
                summary: observation.summary,
                streaming_pages: AtomicU64::new(0),
                clean_since_snapshot: AtomicBool::new(false),
            }),
        })
    }

    /// Returns whether every captured physical page remains in its armed epoch.
    #[must_use]
    pub fn remains_current(&self) -> bool {
        crate::metrics::record(crate::metrics::Counter::CpuDependencyChecks, 1);
        if self.inner.streaming_pages.load(Ordering::Acquire) != 0 {
            return false;
        }
        let epoch = self.inner.summary.epoch.load(Ordering::Acquire);
        let current =
            epoch != u64::MAX && epoch == self.inner.observed_summary.load(Ordering::Acquire);
        if current {
            self.inner
                .clean_since_snapshot
                .store(true, Ordering::Release);
        }
        current
    }

    /// Confirms the previously consumed bytes of a single captured range after
    /// a CPU-write invalidation. Only an exact match refreshes the observation;
    /// a mismatch stays dirty, so a later check cannot validate stale content.
    /// This avoids repeated reads/protection when unrelated bytes share a page.
    pub fn confirm_unchanged(&self, expected: &[u8]) -> Result<bool, CanonicalRangeAccessError> {
        let [range] = self.inner.domains.as_ref() else {
            return Err(CanonicalRangeAccessError::DependencyMismatch);
        };
        if range.size() != expected.len() as u64 {
            return Err(CanonicalRangeAccessError::DependencyMismatch);
        }
        if self.remains_current() {
            return Ok(true);
        }
        let pages: SmallVec<[&CanonicalBackingPage; 8]> =
            self.inner.pages.iter().map(|p| &p.page).collect();
        let mut gates: SmallVec<[&crate::ExecutionGate; 2]> =
            pages.iter().map(|p| p.store().execution_gate()).collect();
        gates.sort_unstable_by_key(|g| g.identity());
        gates.dedup_by_key(|g| g.identity());
        loop {
            let checked = {
                let _shared: SmallVec<[_; 2]> = gates.iter().map(|g| g.acquire_shared()).collect();
                crate::backing::ProtectedPageSet::with(&pages, |protected| {
                    if self.remains_current() {
                        return Ok(Some(true));
                    }
                    for page in &pages {
                        if !protected
                            .cpu_visible(page)
                            .map_err(CanonicalRangeAccessError::Backing)?
                        {
                            return Ok(None);
                        }
                    }
                    let mut bytes = [0; 256];
                    for (index, expected) in expected.chunks(bytes.len()).enumerate() {
                        let output = &mut bytes[..expected.len()];
                        range.read_with((index * 256) as u64, output, &mut |p, at, out| {
                            protected.read(p, at, out)
                        })?;
                        crate::metrics::record(
                            crate::metrics::Counter::SnapshotCopiedBytes,
                            output.len() as u64,
                        );
                        if output != expected {
                            return Ok(Some(false));
                        }
                    }
                    self.rearm_stable(protected)?;
                    Ok(Some(true))
                })?
            };
            if let Some(matches) = checked {
                return Ok(matches);
            }
            // GPU demand runs without a mapping lease or any physical lock.
            for page in &pages {
                page.prepare_cpu_access()
                    .map_err(CanonicalRangeAccessError::Backing)?;
            }
        }
    }

    /// Establishes a protected baseline after the consumer incorporates or
    /// completely overwrites these bytes. Call outside native execution and
    /// without a memory lease: protection changes rendezvous with the engine.
    pub fn rearm(&self) -> Result<(), CanonicalRangeAccessError> {
        Self::snapshot_batch_with_resolver(
            &[CpuWriteSnapshotRequest {
                dependency: self,
                range: &self.inner.domains[0],
                selection: CpuWriteSnapshotSelection::Rearm,
                alignment: 1,
            }],
            &mut |_, _| unreachable!("tracking rearm never demands CPU visibility"),
        )?;
        Ok(())
    }

    fn rearm_stable(
        &self,
        protected: &mut crate::backing::ProtectedPageSet<'_>,
    ) -> Result<(), CanonicalRangeAccessError> {
        let mut adaptive = self
            .inner
            .adaptive
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.arm_pages(0..self.inner.pages.len(), protected)?;
        adaptive.streaks.fill(0);
        adaptive.streaming.clear();
        self.inner.streaming_pages.store(0, Ordering::Release);
        self.record_summary();
        Ok(())
    }

    /// Copies every dirty page intersection and establishes the next clean
    /// baseline before direct CPU execution resumes.
    /// Call outside native execution and without a memory lease or cache lock.
    ///
    /// Returned offsets are logical offsets within `range`. Ranges are
    /// expanded to `alignment` where possible so device backends can satisfy
    /// copy constraints without taking a second, racy snapshot.
    pub fn snapshot_dirty_pages(
        &self,
        range: &CanonicalBackingRange,
        alignment: u64,
    ) -> Result<CanonicalByteSnapshots, CanonicalRangeAccessError> {
        self.snapshot_bytes(range, CpuWriteSnapshotSelection::DirtyPages, alignment)
    }

    /// Copies the complete range only when at least one represented page is
    /// dirty, then establishes the next clean baseline atomically with that
    /// snapshot.
    /// Call outside native execution and without a memory lease or cache lock.
    pub fn snapshot_whole_if_dirty(
        &self,
        range: &CanonicalBackingRange,
    ) -> Result<Option<Box<[u8]>>, CanonicalRangeAccessError> {
        let mut snapshots =
            self.snapshot_bytes(range, CpuWriteSnapshotSelection::WholeIfDirty, 1)?;
        Ok(snapshots.pop().map(|(_, bytes)| bytes))
    }

    /// Copies the complete range and establishes the next clean baseline
    /// atomically with that snapshot.
    /// Call outside native execution and without a memory lease or cache lock.
    pub fn snapshot_all(
        &self,
        range: &CanonicalBackingRange,
    ) -> Result<Box<[u8]>, CanonicalRangeAccessError> {
        let mut snapshots = self.snapshot_bytes(range, CpuWriteSnapshotSelection::All, 1)?;
        snapshots
            .pop()
            .map(|(_, bytes)| bytes)
            .ok_or(CanonicalRangeAccessError::IncompleteRange)
    }

    fn snapshot_bytes(
        &self,
        range: &CanonicalBackingRange,
        selection: CpuWriteSnapshotSelection,
        alignment: u64,
    ) -> Result<CanonicalByteSnapshots, CanonicalRangeAccessError> {
        self.snapshot_with_resolver(range, selection, alignment, &mut |coordinator, request| {
            coordinator.make_cpu_visible(request)
        })
    }

    /// Takes a snapshot using an explicit visibility resolver. Device owners
    /// must resolve their own writes inline instead of waiting for a request
    /// queued to the thread which is currently taking this snapshot.
    /// The resolver runs without page locks or execution mutation guards.
    /// Call outside native execution and without a memory lease or cache lock.
    pub fn snapshot_with_resolver(
        &self,
        range: &CanonicalBackingRange,
        selection: CpuWriteSnapshotSelection,
        alignment: u64,
        resolve: &mut crate::CpuVisibilityResolver<'_>,
    ) -> Result<CanonicalByteSnapshots, CanonicalRangeAccessError> {
        self.validate_snapshot_range(range, alignment)?;
        if matches!(
            selection,
            CpuWriteSnapshotSelection::DirtyPages | CpuWriteSnapshotSelection::WholeIfDirty
        ) && self.remains_current()
        {
            return Ok(Vec::new());
        }
        Self::snapshot_batch_with_resolver(
            &[CpuWriteSnapshotRequest {
                dependency: self,
                range,
                selection,
                alignment,
            }],
            resolve,
        )
        .map(|mut snapshots| snapshots.pop().expect("one snapshot request"))
    }

    /// Samples compatible inputs under one ordered set of execution gates.
    /// Independent resource observations and logical byte layouts remain distinct.
    /// A required readback ends the current exclusion phase; already copied bytes
    /// remain retained while pending requests revalidate after the callback.
    pub fn snapshot_batch_with_resolver(
        requests: &[CpuWriteSnapshotRequest<'_>],
        resolve: &mut crate::CpuVisibilityResolver<'_>,
    ) -> Result<Vec<CanonicalByteSnapshots>, CanonicalRangeAccessError> {
        for request in requests {
            request
                .dependency
                .validate_snapshot_range(request.range, request.alignment)?;
            if request.selection != CpuWriteSnapshotSelection::Rearm {
                crate::metrics::record(
                    crate::metrics::Counter::SnapshotRequestedBytes,
                    request.range.size(),
                );
            }
        }
        let mut snapshots = vec![Vec::new(); requests.len()];
        let mut pending = (0..requests.len())
            .filter(|index| {
                let request = &requests[*index];
                matches!(
                    request.selection,
                    CpuWriteSnapshotSelection::All | CpuWriteSnapshotSelection::Rearm
                ) || !request.dependency.remains_current()
            })
            .collect::<Vec<_>>();
        while !pending.is_empty() {
            let mut pages: SmallVec<[&CanonicalBackingPage; 16]> = SmallVec::new();
            for &index in &pending {
                pages.extend(
                    requests[index]
                        .dependency
                        .inner
                        .pages
                        .iter()
                        .map(|page| &page.page),
                );
            }
            pages.sort_unstable_by_key(|page| page.identity());
            pages.dedup_by_key(|page| page.identity());
            let mut gates: SmallVec<[&crate::ExecutionGate; 2]> = pages
                .iter()
                .map(|page| page.store().execution_gate())
                .collect();
            gates.sort_unstable_by_key(|gate| gate.identity());
            gates.dedup_by_key(|gate| gate.identity());
            // Only transitions require draining native writers. A protected
            // read set excludes checked writes and write-fault repair locally;
            // no global safepoint or content copy is needed for current inputs.
            // Cover the whole dependency, including its adaptive state, so two
            // concurrent snapshots cannot invert adaptive/page lock ordering.
            let (demands, retry) = {
                let _leases: SmallVec<[_; 2]> =
                    gates.iter().map(|gate| gate.acquire_shared()).collect();
                crate::metrics::record(
                    crate::metrics::Counter::RegionSnapshotPages,
                    pages.len() as u64,
                );
                crate::backing::ProtectedPageSet::with(&pages, |protected| {
                    Self::snapshot_pending(requests, &pending, &mut snapshots, protected)
                })?
            };
            for page in demands.values() {
                page.ensure_cpu_visible_with(resolve, true)
                    .map_err(|error| {
                        CanonicalRangeAccessError::Backing(crate::CanonicalPageError::Visibility(
                            error,
                        ))
                    })?;
            }
            pending = retry;
        }
        Ok(snapshots)
    }

    fn snapshot_pending(
        requests: &[CpuWriteSnapshotRequest<'_>],
        pending: &[usize],
        snapshots: &mut [CanonicalByteSnapshots],
        protected: &mut crate::backing::ProtectedPageSet<'_>,
    ) -> Result<
        (BTreeMap<CanonicalPageId, CanonicalBackingPage>, Vec<usize>),
        CanonicalRangeAccessError,
    > {
        let mut demands = BTreeMap::new();
        let mut retry = Vec::new();
        for &index in pending {
            let request = &requests[index];
            match request.dependency.snapshot_stable(
                request.range,
                request.selection,
                request.alignment,
                protected,
            )? {
                Ok(bytes) => snapshots[index] = bytes,
                Err(pages) => {
                    for page in pages {
                        demands.entry(page.identity()).or_insert(page);
                    }
                    retry.push(index);
                }
            }
        }
        Ok((demands, retry))
    }

    fn validate_snapshot_range(
        &self,
        range: &CanonicalBackingRange,
        alignment: u64,
    ) -> Result<(), CanonicalRangeAccessError> {
        if alignment == 0 {
            return Err(CanonicalRangeAccessError::InvalidAlignment(alignment));
        }
        let known_domain = self
            .inner
            .domains
            .iter()
            .any(|domain| domain.shares_layout(range));
        if !known_domain
            && !range.pages().all(|page| {
                self.inner
                    .pages
                    .binary_search_by_key(&page.identity(), |page| page.page.identity())
                    .is_ok()
            })
        {
            return Err(CanonicalRangeAccessError::DependencyMismatch);
        }
        Ok(())
    }

    fn snapshot_stable(
        &self,
        range: &CanonicalBackingRange,
        selection: CpuWriteSnapshotSelection,
        alignment: u64,
        protected: &mut crate::backing::ProtectedPageSet<'_>,
    ) -> Result<Result<CanonicalByteSnapshots, Vec<CanonicalBackingPage>>, CanonicalRangeAccessError>
    {
        if selection == CpuWriteSnapshotSelection::Rearm {
            self.rearm_stable(protected)?;
            return Ok(Ok(Vec::new()));
        }
        if selection != CpuWriteSnapshotSelection::All && self.remains_current() {
            return Ok(Ok(Vec::new()));
        }
        let mut adaptive = self
            .inner
            .adaptive
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self
            .inner
            .clean_since_snapshot
            .swap(false, Ordering::AcqRel)
        {
            adaptive.streaks.fill(0);
        }
        if selection != CpuWriteSnapshotSelection::All {
            let device_owned = adaptive
                .streaming
                .keys()
                .copied()
                .filter(|index| {
                    let page = &self.inner.pages[*index];
                    range.contains_page(page.page.identity())
                        && page.page.cpu_dirty_epoch()
                            == page.observed_epoch.load(Ordering::Acquire)
                        && matches!(
                            protected.visibility(&page.page),
                            VisibilityState::GpuNewer { .. }
                        )
                })
                .collect::<Vec<_>>();
            self.arm_pages(device_owned.iter().copied(), protected)?;
            for index in device_owned {
                // Device ownership already revokes CPU writes. Restore a
                // protected observation without downloading unchanged GPU
                // bytes merely to compare a former CPU streaming shadow.
                adaptive.streaming.remove(&index);
                adaptive.streaks[index] = 0;
            }
        }
        let mut candidates = adaptive.streaming.keys().copied().collect::<BTreeSet<_>>();
        for (group, pages) in self.inner.pages.chunks(CPU_WRITE_GROUP_PAGES).enumerate() {
            crate::metrics::record(crate::metrics::Counter::DirtyGroupChecks, 1);
            let epoch = self.inner.summary.groups[group].load(Ordering::Acquire);
            if epoch == u64::MAX
                || epoch != self.inner.observed_groups[group].load(Ordering::Acquire)
            {
                crate::metrics::record(
                    crate::metrics::Counter::DirtyPageChecks,
                    pages.len() as u64,
                );
                for (offset, page) in pages.iter().enumerate() {
                    if page.page.cpu_dirty_epoch() != page.observed_epoch.load(Ordering::Acquire) {
                        candidates.insert(group * CPU_WRITE_GROUP_PAGES + offset);
                    }
                }
            }
        }
        candidates.retain(|index| range.contains_page(self.inner.pages[*index].page.identity()));
        let mut needed = if selection == CpuWriteSnapshotSelection::All {
            self.inner
                .pages
                .iter()
                .enumerate()
                .filter_map(|(index, page)| {
                    range.contains_page(page.page.identity()).then_some(index)
                })
                .collect::<BTreeSet<_>>()
        } else {
            candidates.clone()
        };
        // Streaming pages have no protected observation. Compare their
        // exact bytes while CPU execution is quiescent; unchanged samples
        // transfer nothing and eventually recover protected tracking.
        let visible = needed.iter().try_fold(true, |visible, index| {
            Ok::<_, CanonicalRangeAccessError>(
                visible
                    && protected
                        .cpu_visible(&self.inner.pages[*index].page)
                        .map_err(CanonicalRangeAccessError::Backing)?,
            )
        })?;
        if !visible {
            drop(adaptive);
            return Ok(Err(needed
                .iter()
                .map(|index| self.inner.pages[*index].page.clone())
                .collect()));
        }
        let mut dirty = candidates.clone();
        let streaming = adaptive
            .streaming
            .keys()
            .copied()
            .filter(|index| range.contains_page(self.inner.pages[*index].page.identity()))
            .collect::<Vec<_>>();
        for index in streaming {
            let page = &self.inner.pages[index];
            crate::metrics::record(
                crate::metrics::Counter::StreamingComparedBytes,
                page.page.size() as u64,
            );
            let mut bytes = vec![0; page.page.size()].into_boxed_slice();
            protected
                .read(&page.page, 0, &mut bytes)
                .map_err(CanonicalRangeAccessError::Backing)?;
            page.observed_epoch
                .store(page.page.cpu_dirty_epoch(), Ordering::Release);
            let streaming = adaptive
                .streaming
                .get_mut(&index)
                .expect("retained streaming page");
            if streaming.bytes == bytes {
                dirty.remove(&index);
                streaming.quiet = streaming.quiet.saturating_add(1);
                if streaming.quiet >= STREAMING_QUIET_THRESHOLD {
                    self.arm_pages([index].into_iter(), protected)?;
                    adaptive.streaming.remove(&index);
                    adaptive.streaks[index] = 0;
                }
            } else {
                streaming.bytes = bytes;
                streaming.quiet = 0;
                dirty.insert(index);
            }
        }
        let dirty_pages = dirty
            .iter()
            .map(|index| self.inner.pages[*index].page.identity())
            .collect::<BTreeSet<_>>();
        let mut intervals = Vec::new();
        if selection == CpuWriteSnapshotSelection::All
            || (selection == CpuWriteSnapshotSelection::WholeIfDirty && !dirty.is_empty())
            || (selection == CpuWriteSnapshotSelection::DirtyPages
                && range.layout.pages.len() == dirty_pages.len())
        {
            intervals.push((0, range.size()));
        } else if selection == CpuWriteSnapshotSelection::DirtyPages {
            for page in &dirty_pages {
                let mut overflow = false;
                range.visit_page_mappings(*page, &mut |offset, _, size| {
                    let start = offset / alignment * alignment;
                    if let Some(end) = (offset + size).checked_add(alignment - 1) {
                        intervals.push((start, (end / alignment * alignment).min(range.size())));
                    } else {
                        overflow = true;
                    }
                });
                if overflow {
                    return Err(CanonicalRangeAccessError::RangeOverflow);
                }
            }
            normalize_intervals(&mut intervals);
        }
        // Retain only the copied windows. A sparse update in a large tiled
        // view must not expand the entire byte topology to identify a handful
        // of aligned physical pages or copy those same windows afterwards.
        let windows = intervals
            .into_iter()
            .map(|(offset, end)| {
                range
                    .snapshot_subrange(offset, end - offset)
                    .map(|window| (offset, window))
                    .map_err(|_| CanonicalRangeAccessError::IncompleteRange)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Alignment or WholeIfDirty can include additional pages. Resolve
        // only pages actually copied, never unrelated GPU-owned bindings.
        for (_, window) in &windows {
            for page in window.pages() {
                let index = self
                    .inner
                    .pages
                    .binary_search_by_key(&page.identity(), |page| page.page.identity())
                    .expect("validated dependency page");
                needed.insert(index);
            }
        }
        let visible = needed.iter().try_fold(true, |visible, index| {
            Ok::<_, CanonicalRangeAccessError>(
                visible
                    && protected
                        .cpu_visible(&self.inner.pages[*index].page)
                        .map_err(CanonicalRangeAccessError::Backing)?,
            )
        })?;
        if !visible {
            drop(adaptive);
            return Ok(Err(needed
                .iter()
                .map(|index| self.inner.pages[*index].page.clone())
                .collect()));
        }
        let mut snapshots = Vec::new();
        for (offset, window) in windows {
            let mut bytes = vec![
                0;
                usize::try_from(window.size())
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?
            ];
            crate::metrics::record(
                crate::metrics::Counter::SnapshotCopiedBytes,
                bytes.len() as u64,
            );
            window.read_with(0, &mut bytes, &mut |page, offset, bytes| {
                protected.read(page, offset, bytes)
            })?;
            snapshots.push((offset, bytes.into_boxed_slice()));
        }
        for index in &dirty {
            // Protected host-written pages do not need the adaptive native
            // streaming policy. Keep their existing protection and baseline.
            if protected.is_observed(&self.inner.pages[*index].page) {
                continue;
            }
            if adaptive.streaming.contains_key(index) {
                continue;
            }
            let streak = adaptive.streaks[*index].saturating_add(1);
            adaptive.streaks[*index] = streak;
            let page = &self.inner.pages[*index];
            if streak >= STREAMING_DIRTY_THRESHOLD {
                let mut bytes = vec![0; page.page.size()].into_boxed_slice();
                protected
                    .read(&page.page, 0, &mut bytes)
                    .map_err(CanonicalRangeAccessError::Backing)?;
                page.observed_epoch
                    .store(page.page.cpu_dirty_epoch(), Ordering::Release);
                adaptive
                    .streaming
                    .insert(*index, StreamingCpuPage { bytes, quiet: 0 });
            }
        }
        if selection == CpuWriteSnapshotSelection::All {
            self.arm_pages(
                (0..self.inner.pages.len()).filter(|index| {
                    range.contains_page(self.inner.pages[*index].page.identity())
                        && !adaptive.streaming.contains_key(index)
                }),
                protected,
            )?;
        } else {
            self.arm_pages(
                dirty
                    .iter()
                    .copied()
                    .filter(|index| !adaptive.streaming.contains_key(index)),
                protected,
            )?;
        }
        self.inner
            .streaming_pages
            .store(adaptive.streaming.len() as u64, Ordering::Release);
        self.record_summary();
        Ok(Ok(snapshots))
    }

    fn arm_pages(
        &self,
        indices: impl Iterator<Item = usize> + Clone,
        protected: &mut crate::backing::ProtectedPageSet<'_>,
    ) -> Result<(), CanonicalRangeAccessError> {
        for index in indices {
            let page = &self.inner.pages[index];
            page.observed_epoch
                .store(protected.arm(&page.page), Ordering::Release);
        }
        Ok(())
    }

    fn record_summary(&self) {
        // A dependency may include stores outside the snapshotted range.
        // Observe the global epoch before checking groups so a concurrent
        // publication cannot be acknowledged without its changed page.
        let summary_epoch = self.inner.summary.epoch.load(Ordering::Acquire);
        let mut current = true;
        for (group, pages) in self.inner.pages.chunks(CPU_WRITE_GROUP_PAGES).enumerate() {
            let epoch = self.inner.summary.groups[group].load(Ordering::Acquire);
            let observed = &self.inner.observed_groups[group];
            if epoch != observed.load(Ordering::Acquire) {
                if pages.iter().all(|page| {
                    page.page.cpu_dirty_epoch() == page.observed_epoch.load(Ordering::Acquire)
                }) {
                    observed.store(epoch, Ordering::Release);
                } else {
                    current = false;
                }
            }
        }
        if current {
            self.inner
                .observed_summary
                .store(summary_epoch, Ordering::Release);
        }
    }

    /// Reports whether some hot pages are being compared without arming faults.
    #[must_use]
    pub fn has_streaming_pages(&self) -> bool {
        self.inner.streaming_pages.load(Ordering::Acquire) != 0
    }
}

/// Which bytes a CPU-write dependency snapshot must materialize.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuWriteSnapshotSelection {
    /// Rearm the complete dependency after overwrite, without copying/readback.
    Rearm,
    /// Only intersections with pages whose CPU-write epochs changed.
    DirtyPages,
    /// The entire range if any observed page changed.
    WholeIfDirty,
    /// The entire range, including its initial contents.
    All,
}

fn normalize_intervals(intervals: &mut Vec<(u64, u64)>) {
    intervals.sort_unstable_by_key(|&(start, _)| start);
    let mut output = 0_usize;
    for input in 0..intervals.len() {
        let current = intervals[input];
        if output != 0 {
            let previous = &mut intervals[output - 1];
            if current.0 <= previous.1 {
                previous.1 = previous.1.max(current.1);
                continue;
            }
        }
        intervals[output] = current;
        output += 1;
    }
    intervals.truncate(output);
}

impl std::fmt::Debug for CanonicalCpuWriteDependency {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let pages = self
            .inner
            .pages
            .iter()
            .map(|page| {
                (
                    page.page.identity(),
                    page.observed_epoch.load(Ordering::Acquire),
                )
            })
            .collect::<Vec<_>>();
        formatter
            .debug_struct("CanonicalCpuWriteDependency")
            .field("pages", &pages)
            .field(
                "streaming_pages",
                &self.inner.streaming_pages.load(Ordering::Acquire),
            )
            .finish()
    }
}

impl PartialEq for CanonicalCpuWriteDependency {
    fn eq(&self, other: &Self) -> bool {
        self.inner.pages.len() == other.inner.pages.len()
            && self
                .inner
                .pages
                .iter()
                .zip(&other.inner.pages)
                .all(|(left, right)| left.page.identity() == right.page.identity())
    }
}

impl Eq for CanonicalCpuWriteDependency {}

impl CanonicalBackingRange {
    fn execution_gates(&self) -> impl Iterator<Item = &crate::ExecutionGate> {
        self.layout.gates.iter()
    }

    /// Unique retained pages sorted by physical identity, independent of logical byte order.
    /// Mapped views do not expand their byte permutation for ownership tracking.
    pub fn pages(&self) -> impl Iterator<Item = &CanonicalBackingPage> {
        self.layout.pages.iter()
    }
    fn contains_page(&self, page: CanonicalPageId) -> bool {
        self.layout
            .pages
            .binary_search_by_key(&page, CanonicalBackingPage::identity)
            .is_ok()
    }

    /// Conservative physical-page overlap, without expanding byte permutations.
    /// Retained pages are sorted independently of the ordered logical segments.
    /// Disjoint allocations reject in constant time; otherwise search the larger
    /// page set using the smaller set. Slice metadata may conservatively overlap.
    pub fn may_share_pages(&self, other: &Self) -> bool {
        let a = &self.layout.pages;
        let b = &other.layout.pages;
        if a.last().unwrap().identity() < b[0].identity()
            || b.last().unwrap().identity() < a[0].identity()
        {
            return false;
        }
        let (small, large) = if a.len() < b.len() { (a, b) } else { (b, a) };
        small.iter().any(|page| {
            large
                .binary_search_by_key(&page.identity(), CanonicalBackingPage::identity)
                .is_ok()
        })
    }

    /// Creates a non-empty range and checks its total length.
    pub fn new(segments: Vec<CanonicalBackingSegment>) -> Result<Self, CanonicalRangeError> {
        if segments.is_empty() {
            return Err(CanonicalRangeError::Empty);
        }
        let mut size = 0_u64;
        let mut segment_ends = Vec::with_capacity(segments.len());
        for segment in &segments {
            size = size
                .checked_add(segment.size)
                .ok_or(CanonicalRangeError::RangeOverflow)?;
            segment_ends.push(size);
        }
        let mut pages = BTreeMap::new();
        let mut stores = BTreeMap::new();
        let mut previous_page = None;
        for segment in &segments {
            if previous_page != Some(segment.page()) {
                pages.entry(segment.page()).or_insert_with(|| {
                    stores
                        .entry(segment.backing().store().execution_gate().identity())
                        .or_insert_with(|| segment.backing().store().execution_gate().clone());
                    segment.backing().clone()
                });
            }
            previous_page = Some(segment.page());
        }
        let pages: Box<[_]> = pages.into_values().collect();
        let stores = stores.into_values().collect();
        Ok(Self {
            layout: Arc::new(CanonicalRangeLayout {
                topology: std::sync::OnceLock::from(CanonicalSegmentTopology {
                    segments: segments.into(),
                    segment_ends: segment_ends.into(),
                }),
                mapping: None,
                page_segments: std::sync::OnceLock::new(),
                segment_runs: std::sync::OnceLock::new(),
                byte_mapping_hash: std::sync::OnceLock::new(),
                pages,
                gates: stores,
                owner: Mutex::new(None),
                visibility_epoch: std::sync::OnceLock::new(),
                clean_read_epoch: AtomicU64::new(u64::MAX),
                cpu_summary: Mutex::new(std::sync::Weak::new()),
            }),
            size,
        })
    }

    /// Whether two ranges retain the same immutable ordered page topology.
    /// A false result does not imply different physical coverage.
    #[must_use]
    pub fn shares_layout(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.layout, &other.layout)
    }

    /// Returns the logical byte length.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the ordered canonical segments.
    #[must_use]
    pub fn segments(&self) -> &[CanonicalBackingSegment] {
        &self.layout.topology().segments
    }

    /// Iterates the page segments intersecting a validated logical [offset, end)
    /// window in O(log n + affected segments), without constructing a subrange.
    /// Returned segments retain their original logical starts and are not clipped.
    pub fn segments_between(
        &self,
        offset: u64,
        end: u64,
    ) -> impl Iterator<Item = (u64, &CanonicalBackingSegment)> {
        let first = self
            .layout
            .topology()
            .segment_ends
            .partition_point(|&limit| limit <= offset);
        let last = self
            .layout
            .topology()
            .segment_ends
            .partition_point(|&limit| limit < end);
        let limit = if offset < end {
            (last + 1).min(self.layout.topology().segments.len())
        } else {
            first
        };
        (first..limit).map(|index| {
            let start = if index == 0 {
                0
            } else {
                self.layout.topology().segment_ends[index - 1]
            };
            (start, &self.layout.topology().segments[index])
        })
    }

    /// Iterates the segments intersecting a checked logical subrange.
    /// The offset is the segment's start in this range, before clipping.
    pub fn subrange_segments(
        &self,
        offset: u64,
        size: u64,
    ) -> Result<impl Iterator<Item = (u64, &CanonicalBackingSegment)>, CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::RangeOverflow)?;
        if size == 0 || end > self.size {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        Ok(self.segments_between(offset, end))
    }

    /// Observes whether a retained subrange needs no device readback. This does
    /// not acquire a lease; callers still perform the normal checked access.
    pub fn subrange_is_cpu_visible(
        &self,
        offset: u64,
        size: u64,
    ) -> Result<bool, CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::RangeOverflow)?;
        if size == 0 || end > self.size {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        Ok(self.segments_between(offset, end).all(|(_, segment)| {
            matches!(
                segment.visibility_state(),
                VisibilityState::Clean | VisibilityState::CpuNewer
            )
        }))
    }

    /// Returns physical byte coverage in page/offset order, retaining holes.
    /// This is a dependency/publication domain, not the source's logical order.
    /// Adjacent/overlapping segments merge only with identical mapping facts.
    pub fn physical_union(&self) -> Result<Self, CanonicalRangeError> {
        let mut segments = self.segments().to_vec();
        segments.sort_unstable_by_key(|s| (s.page(), s.offset));
        let mut merged: Vec<CanonicalBackingSegment> = Vec::new();
        for segment in segments {
            if let Some(last) = merged.last_mut()
                && last.page() == segment.page()
                && segment.offset <= last.offset + last.size
                && last.permissions == segment.permissions
                && last.mapping_generation == segment.mapping_generation
            {
                last.size =
                    (segment.offset + segment.size).max(last.offset + last.size) - last.offset;
            } else {
                merged.push(segment);
            }
        }
        Self::new(merged)
    }

    /// Retains a checked logical subrange with the same canonical page identity.
    pub fn snapshot_subrange(&self, offset: u64, size: u64) -> Result<Self, CanonicalRangeError> {
        if offset == 0 && size == self.size {
            return Ok(self.clone());
        }
        let mut captured = Vec::new();
        self.snapshot_subrange_into(offset, size, &mut captured)?;
        Self::new(captured)
    }

    /// Appends a retained subrange directly to an existing segment builder.
    ///
    /// The output is unchanged on failure. Resource resolvers use this to
    /// assemble one canonical range across mappings without allocating and
    /// cloning an intermediate range for every mapping fragment.
    pub fn snapshot_subrange_into(
        &self,
        offset: u64,
        size: u64,
        output: &mut Vec<CanonicalBackingSegment>,
    ) -> Result<(), CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::RangeOverflow)?;
        if size == 0 || end > self.size {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        if let Some(mapping) = &self.layout.mapping {
            let original_len = output.len();
            let result = mapping.snapshot(offset, size, output);
            if result.is_err() {
                output.truncate(original_len);
            }
            return result;
        }
        let original_len = output.len();
        let result = (|| {
            for (logical_start, segment) in self.segments_between(offset, end) {
                let logical_end = logical_start
                    .checked_add(segment.size)
                    .ok_or(CanonicalRangeError::RangeOverflow)?;
                let capture_start = offset.max(logical_start);
                let capture_end = end.min(logical_end);
                if capture_start < capture_end {
                    let within_segment = capture_start - logical_start;
                    let page_offset = segment
                        .offset
                        .checked_add(within_segment)
                        .ok_or(CanonicalRangeError::SegmentOverflow)?;
                    output.push(CanonicalBackingSegment::snapshot(
                        segment.backing.clone(),
                        page_offset,
                        capture_end - capture_start,
                        segment.permissions,
                        segment.mapping_generation,
                    )?);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            output.truncate(original_len);
        }
        result
    }

    /// Copies a checked logical subrange from retained canonical storage.
    ///
    /// Reads walk canonical page segments directly. A CPU virtual address is
    /// neither required nor reconstructed, so aliases and unmapped-but-retained
    /// storage preserve the same byte identity.
    pub fn read(&self, offset: u64, output: &mut [u8]) -> Result<(), CanonicalRangeAccessError> {
        self.read_with_resolver(offset, output, &mut |coordinator, request| {
            coordinator.make_cpu_visible(request)
        })
    }

    /// Normal CPU, DSP and command word loads with hardware atomicity against native
    /// writers. Physical locks serialize checked producers and GPU writeback;
    /// this does not create a whole-range snapshot or arm native protection.
    /// Resource observations use the stable read/snapshot operations instead.
    pub fn read_word_stream(
        &self,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), CanonicalRangeAccessError> {
        let end = offset
            .checked_add(output.len() as u64)
            .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
        if end > self.size {
            return Err(CanonicalRangeAccessError::OutOfBounds {
                offset,
                size: output.len() as u64,
                range_size: self.size,
            });
        }
        if output.is_empty() {
            return Ok(());
        }
        let mut pages: SmallVec<[&CanonicalBackingPage; 8]> = self
            .segments_between(offset, end)
            .map(|(_, s)| s.backing())
            .collect();
        pages.sort_unstable_by_key(|p| p.identity());
        pages.dedup_by_key(|p| p.identity());
        loop {
            let demand = {
                let _admission: SmallVec<[_; 2]> = self
                    .execution_gates()
                    .map(crate::ExecutionGate::acquire_shared)
                    .collect();
                let mut accesses = SmallVec::<[crate::CanonicalCpuAccess<'_>; 8]>::new();
                let mut demand = None;
                for page in &pages {
                    if let Some(access) = page
                        .try_cpu_access()
                        .map_err(CanonicalRangeAccessError::Backing)?
                    {
                        accesses.push(access);
                    } else {
                        demand = Some((*page).clone());
                        break;
                    }
                }
                if demand.is_none() {
                    self.read_with(offset, output, &mut |page, at, bytes| {
                        let index = pages
                            .binary_search_by_key(&page.identity(), |p| p.identity())
                            .expect("the consumed physical page is retained");
                        accesses[index].read(at, bytes)
                    })?;
                    return Ok(());
                }
                demand
            };
            // The actual device producer may wait or call back into memory.
            // Release every page lock and mapping admission before asking it.
            demand
                .expect("an incomplete CPU read has one visibility demand")
                .prepare_cpu_access()
                .map_err(CanonicalRangeAccessError::Backing)?;
        }
    }

    /// Reads with explicit visibility routing, allowing a device owner to
    /// resolve its own writes inline. The resolver runs outside memory locks.
    pub fn read_with_resolver(
        &self,
        offset: u64,
        output: &mut [u8],
        resolve: &mut crate::CpuVisibilityResolver<'_>,
    ) -> Result<(), CanonicalRangeAccessError> {
        let output_size =
            u64::try_from(output.len()).map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
        let end = offset
            .checked_add(output_size)
            .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
        if end > self.size {
            return Err(CanonicalRangeAccessError::OutOfBounds {
                offset,
                size: output_size,
                range_size: self.size,
            });
        }
        if output.is_empty() {
            return Ok(());
        }
        // Naturally atomic command/descriptor scalars and already protected
        // input bytes need neither a native stop nor permission changes.
        if self.layout.mapping.is_none() || self.layout.topology.get().is_some() {
            let mut segments = self.segments_between(offset, end);
            if let Some((logical, segment)) = segments.next()
                && segments.next().is_none()
            {
                let page_offset = usize::try_from(segment.offset + (offset - logical))
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
                let _lease = segment.backing.store().execution_gate().acquire_shared();
                if segment
                    .backing
                    .read_immediate(page_offset, output)
                    .map_err(CanonicalRangeAccessError::Backing)?
                {
                    return Ok(());
                }
            }
        }
        // Retain only the consumed pages, in physical lock order. Most command
        // fetches fit on the stack; a large retained view must not enumerate
        // unrelated pages for a small read.
        let mut pages: SmallVec<[&CanonicalBackingPage; 8]> =
            if offset == 0 && end == self.size && self.layout.mapping.is_some() {
                self.pages().collect()
            } else {
                self.segments_between(offset, end)
                    .map(|(_, segment)| &segment.backing)
                    .collect()
            };
        pages.sort_unstable_by_key(|page| page.identity());
        pages.dedup_by_key(|page| page.identity());
        loop {
            {
                let _leases: SmallVec<[_; 2]> = self
                    .execution_gates()
                    .map(crate::ExecutionGate::acquire_shared)
                    .collect();
                crate::metrics::record(
                    crate::metrics::Counter::RegionRawReadPages,
                    pages.len() as u64,
                );
                if crate::backing::ProtectedPageSet::with(&pages, |protected| {
                    if !pages
                        .iter()
                        .try_fold(true, |visible, page| {
                            protected.cpu_visible(page).map(|v| visible && v)
                        })
                        .map_err(CanonicalRangeAccessError::Backing)?
                    {
                        return Ok(false);
                    }
                    self.read_with(offset, output, &mut |page, offset, bytes| {
                        protected.read(page, offset, bytes)
                    })?;
                    Ok(true)
                })? {
                    return Ok(());
                }
            }
            // Visibility callbacks must never retain admission or page locks.
            for page in &pages {
                page.prepare_cpu_access_with(resolve)
                    .map_err(CanonicalRangeAccessError::Backing)?;
            }
        }
    }

    fn read_with(
        &self,
        offset: u64,
        output: &mut [u8],
        read: &mut CanonicalPageReader<'_>,
    ) -> Result<(), CanonicalRangeAccessError> {
        let output_size =
            u64::try_from(output.len()).map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
        let end = offset
            .checked_add(output_size)
            .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
        if end > self.size {
            return Err(CanonicalRangeAccessError::OutOfBounds {
                offset,
                size: output_size,
                range_size: self.size,
            });
        }
        if offset == 0
            && output_size == self.size
            && let Some(mapping) = &self.layout.mapping
            && mapping.read_tiled_with(output, read)?
        {
            return Ok(());
        }
        let mut copied = 0_usize;
        for (logical_start, segment) in self.segments_between(offset, end) {
            let logical_end = logical_start
                .checked_add(segment.size)
                .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
            let read_start = offset.max(logical_start);
            let read_end = end.min(logical_end);
            if read_start < read_end {
                let within_segment = read_start - logical_start;
                let page_offset = segment
                    .offset
                    .checked_add(within_segment)
                    .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                let copy_size = usize::try_from(read_end - read_start)
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
                let page_offset = usize::try_from(page_offset)
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
                let copied_end = copied
                    .checked_add(copy_size)
                    .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                read(
                    &segment.backing,
                    page_offset,
                    &mut output[copied..copied_end],
                )
                .map_err(CanonicalRangeAccessError::Backing)?;
                copied = copied_end;
            }
        }
        if copied != output.len() {
            return Err(CanonicalRangeAccessError::IncompleteRange);
        }
        Ok(())
    }

    fn visibility_summary(&self) -> &Arc<AtomicU64> {
        self.layout.visibility_epoch.get_or_init(|| {
            let mut pages = self.pages().collect::<Vec<_>>();
            pages.sort_unstable_by_key(|page| page.identity());
            let coverage = PageCoverage::from_sorted(pages.iter().map(|page| page.identity()));
            let mut interner = VISIBILITY_SUMMARIES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(summary) = interner.find(&coverage) {
                return summary;
            }
            let summary = Arc::new(AtomicU64::new(0));
            // Serialize registration through the interner until every page is
            // subscribed. A second view cannot cache a partial observation.
            for page in pages {
                page.observe_visibility_summary(&summary);
            }
            interner.insert(coverage, &summary);
            summary
        })
    }

    fn resident_access_ready(&self, declaration: DeviceAccessDeclaration) -> bool {
        crate::metrics::record(crate::metrics::Counter::RangeChecks, 1);
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        if self
            .layout
            .owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|owner| owner.is_current_for(declaration))
        {
            return true;
        }
        let visibility_summary = (!declaration.kind().writes()).then(|| self.visibility_summary());
        let visibility_epoch =
            visibility_summary.map_or(u64::MAX, |summary| summary.load(Ordering::Acquire));
        if !declaration.kind().writes()
            && visibility_epoch != u64::MAX
            && self.layout.clean_read_epoch.load(Ordering::Acquire) == visibility_epoch
        {
            return true;
        }
        let mut all_clean = true;
        // These accesses change no visibility, tracking, permissions or bytes.
        // A later CPU store is observed by the resident resource's dirty
        // dependency; taking its upload snapshot still closes execution.
        // Only actual ownership transitions need the mutation handshake.
        if self.pages().all(|page| {
            crate::metrics::record(crate::metrics::Counter::PageVisibilityChecks, 1);
            match page.visibility_state() {
                VisibilityState::Clean => !declaration.kind().writes(),
                VisibilityState::GpuNewer { device, visible_at } => {
                    all_clean = false;
                    device == declaration.device() && visible_at <= declaration.device_visible_at()
                }
                _ => false,
            }
        }) {
            if all_clean
                && visibility_epoch != u64::MAX
                && visibility_summary
                    .is_some_and(|summary| summary.load(Ordering::Acquire) == visibility_epoch)
            {
                self.layout
                    .clean_read_epoch
                    .store(visibility_epoch, Ordering::Release);
            }
            return true;
        }
        false
    }

    /// Prepares compatible accesses once per physical page and execution gate.
    /// Resource views and completion retains stay with the original requests.
    /// CPU-page caching is bounded; host submission and CPU demand happen later.
    pub fn prepare_resident_device_accesses<'a>(
        accesses: impl IntoIterator<Item = (&'a Self, DeviceAccessDeclaration)>,
        coordinator: &dyn VisibilityCoordinator,
    ) -> Result<(), VisibilityError> {
        let mut boundary = None;
        let mut write_boundary = None;
        let mut pages = SmallVec::<[(CanonicalBackingPage, DeviceAccessDeclaration); 8]>::new();
        for (range, declaration) in accesses {
            let current = (declaration.device(), declaration.device_visible_at());
            if boundary.is_some_and(|prior| prior != current) {
                return Err(VisibilityError::IncompatibleDeclarations);
            }
            boundary = Some(current);
            if let Some(point) = declaration.cpu_visible_at() {
                if write_boundary.is_some_and(|prior| prior != point) {
                    return Err(VisibilityError::IncompatibleDeclarations);
                }
                write_boundary = Some(point);
            }
            if !range.resident_access_ready(declaration) {
                pages.extend(range.pages().map(|page| (page.clone(), declaration)));
            }
        }
        if pages.is_empty() {
            return Ok(());
        }
        pages.sort_unstable_by_key(|(page, _)| page.identity());
        let mut distinct = 0;
        for index in 0..pages.len() {
            if distinct != 0 && pages[distinct - 1].0.identity() == pages[index].0.identity() {
                let previous = pages[distinct - 1].1;
                let declaration = pages[index].1;
                let kind = match (
                    previous.kind().reads() || declaration.kind().reads(),
                    previous.kind().writes() || declaration.kind().writes(),
                ) {
                    (true, true) => crate::DeviceAccessKind::ReadWrite,
                    (false, true) => crate::DeviceAccessKind::Write,
                    _ => crate::DeviceAccessKind::Read,
                };
                pages[distinct - 1].1 = DeviceAccessDeclaration::new(
                    declaration.device(),
                    kind,
                    declaration.device_visible_at(),
                    previous.cpu_visible_at().or(declaration.cpu_visible_at()),
                )
                .map_err(|_| VisibilityError::IncompatibleDeclarations)?;
            } else {
                pages.swap(distinct, index);
                distinct += 1;
            }
        }
        pages.truncate(distinct);
        let mut gates: SmallVec<[&crate::ExecutionGate; 2]> = pages
            .iter()
            .map(|(p, _)| p.store().execution_gate())
            .collect();
        gates.sort_unstable_by_key(|g| g.identity());
        gates.dedup_by_key(|g| g.identity());
        let _shared: SmallVec<[_; 2]> = gates.iter().map(|g| g.acquire_shared()).collect();
        crate::backing::prepare_device_inputs(&pages, coordinator)
    }

    fn advance_device_owner(&self, declaration: DeviceAccessDeclaration) -> bool {
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        self.layout
            .owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|owner| owner.advance(declaration))
    }

    /// Publishes compatible writes from one accepted device producer.
    ///
    /// Physical pages, execution gates and alias protections are updated once.
    /// Resource ranges retain independent ownership records: advancing one
    /// overlapping view must never advance the completion of untouched pages.
    /// The point may be pending or in flight; CPU consumers wait through the
    /// coordinator. An owner accepting recorded work before host submission
    /// must submit that work on demand and invalidate ownership on failure.
    pub fn publish_device_writes<'a>(
        accesses: impl IntoIterator<Item = (&'a Self, DeviceAccessDeclaration)>,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        let mut accesses = accesses.into_iter();
        let Some(first) = accesses.next() else {
            return Ok(());
        };
        let second = accesses.next();
        if second.is_none() {
            if !first.1.kind().writes() || first.1.cpu_visible_at().is_none() {
                return Err(VisibilityError::DeclarationDoesNotWrite);
            }
            // The common resident single-writer case needs neither a batch
            // allocation nor a physical-page walk. This is the same authority
            // advancement used for each independent range in a larger batch.
            if first.0.advance_device_owner(first.1) {
                crate::metrics::record(crate::metrics::Counter::OwnershipPublications, 1);
                return Ok(());
            }
        }
        let accesses = std::iter::once(first)
            .chain(second)
            .chain(accesses)
            .collect::<Vec<_>>();
        let mut boundary = None;
        for (_, declaration) in &accesses {
            let point = declaration
                .cpu_visible_at()
                .filter(|_| declaration.kind().writes())
                .ok_or(VisibilityError::DeclarationDoesNotWrite)?;
            let current = (declaration.device(), declaration.device_visible_at(), point);
            if boundary.is_some_and(|previous| previous != current) {
                return Err(VisibilityError::IncompatibleDeclarations);
            }
            boundary = Some(current);
        }
        let mut layouts = BTreeSet::new();
        let mut owners = Vec::new();
        let mut pages = BTreeMap::new();
        for (range, declaration) in accesses {
            if !layouts.insert(Arc::as_ptr(&range.layout).addr()) {
                continue;
            }
            crate::metrics::record(crate::metrics::Counter::OwnershipPublications, 1);
            if range.advance_device_owner(declaration) {
                continue;
            }
            // No ownership mutex is retained across an execution handshake.
            let owner = Arc::new(crate::backing::RangeDeviceOwner::new(
                declaration.device(),
                declaration.cpu_visible_at().expect("validated write"),
                Arc::clone(&coordinator),
            ));
            for page in range.pages() {
                if let Some((_, _, displaced)) = pages.insert(
                    page.identity(),
                    (page.clone(), declaration, Arc::clone(&owner)),
                ) {
                    // Only the last range owns this physical page. A displaced
                    // range cannot later use its whole-range advancement cache.
                    displaced.detach();
                }
            }
            owners.push((range, owner));
        }
        let mut transitions_required = Vec::new();
        for (page, declaration, owner) in pages.into_values() {
            crate::metrics::record(crate::metrics::Counter::PageOwnershipUpdates, 1);
            if !page.attach_resident_device_owner(declaration, &owner)? {
                transitions_required.push((page, declaration, owner));
            }
        }
        if !transitions_required.is_empty() {
            let mut gates = BTreeMap::new();
            for (page, _, _) in &transitions_required {
                let gate = page.store().execution_gate();
                gates.entry(gate.identity()).or_insert_with(|| gate.clone());
            }
            let shared: SmallVec<[_; 2]> = gates
                .values()
                .map(crate::ExecutionGate::acquire_shared)
                .collect();
            if !crate::backing::publish_device_pages(
                &transitions_required,
                coordinator.as_ref(),
                false,
            )? {
                drop(shared);
                // Executable content invalidation must drain running JIT code.
                // Contents/ownership do not change guest mapping identity.
                let _exclusive: SmallVec<[_; 2]> = gates
                    .values()
                    .map(crate::ExecutionGate::acquire_exclusive)
                    .collect();
                crate::backing::publish_device_pages(
                    &transitions_required,
                    coordinator.as_ref(),
                    true,
                )?;
            }
        }
        for (range, owner) in owners {
            *range
                .layout
                .owner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(owner);
        }
        Ok(())
    }

    /// Invalidates retained physical pages after an unrecoverable transition.
    /// Aliases sharing pages or execution gates do not repeat the handshake.
    pub fn invalidate_visibility_ranges<'a>(
        ranges: impl IntoIterator<Item = &'a Self>,
    ) -> Result<(), VisibilityError> {
        Self::invalidate_visibility_pages(ranges.into_iter().flat_map(Self::pages))
    }

    /// Invalidates retained pages directly; no logical byte topology is needed
    /// when a device owner releases its authoritative contents during teardown.
    pub fn invalidate_visibility_pages<'a>(
        pages: impl IntoIterator<Item = &'a CanonicalBackingPage>,
    ) -> Result<(), VisibilityError> {
        let pages = pages
            .into_iter()
            .map(|page| (page.identity(), page.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut gates = BTreeMap::new();
        for page in pages.values() {
            let gate = page.store().execution_gate();
            gates.entry(gate.identity()).or_insert_with(|| gate.clone());
        }
        let _shared: SmallVec<[_; 2]> = gates
            .values()
            .map(crate::ExecutionGate::acquire_shared)
            .collect();
        crate::backing::invalidate_device_pages(&pages.into_values().collect::<Vec<_>>())
    }
}

/// Invalid construction of a canonical range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalRangeError {
    Empty,
    InvalidSubrange,
    SegmentOverflow,
    InvalidSegmentBounds,
    RangeOverflow,
}

impl Display for CanonicalRangeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "canonical backing range is empty",
            Self::InvalidSubrange => "canonical backing subrange is empty or out of bounds",
            Self::SegmentOverflow => "canonical segment end overflows",
            Self::InvalidSegmentBounds => "canonical segment is outside its retained page",
            Self::RangeOverflow => "canonical backing range length overflows",
        })
    }
}

impl std::error::Error for CanonicalRangeError {}

/// Failure while accessing a retained canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalRangeAccessError {
    RangeOverflow,
    InvalidAlignment(u64),
    DependencyMismatch,
    ResourceExhausted,
    OutOfBounds {
        offset: u64,
        size: u64,
        range_size: u64,
    },
    IncompleteRange,
    Backing(CanonicalPageError),
}

impl Display for CanonicalRangeAccessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RangeOverflow => formatter.write_str("canonical range access overflows"),
            Self::InvalidAlignment(alignment) => write!(
                formatter,
                "canonical snapshot alignment is zero: {alignment}"
            ),
            Self::DependencyMismatch => formatter
                .write_str("CPU-write dependency does not represent exactly this canonical range"),
            Self::ResourceExhausted => formatter.write_str("canonical snapshot allocation failed"),
            Self::OutOfBounds {
                offset,
                size,
                range_size,
            } => write!(
                formatter,
                "canonical range access offset={offset:#x} size={size:#x} exceeds \
                 range-size={range_size:#x}"
            ),
            Self::IncompleteRange => {
                formatter.write_str("canonical range segments do not cover the requested bytes")
            }
            Self::Backing(error) => write!(formatter, "canonical backing access failed: {error}"),
        }
    }
}

impl std::error::Error for CanonicalRangeAccessError {}

/// Why a CPU virtual range could not be translated to canonical RAM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalRangeTranslationErrorReason {
    Empty,
    AddressOverflow,
    Unmapped,
    PermissionDenied,
    DeviceMemory,
    InconsistentBacking,
    ResourceExhausted,
}

/// Pointer-free failure from a canonical range translation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalRangeTranslationError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: CanonicalRangeTranslationErrorReason,
}

impl Display for CanonicalRangeTranslationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "canonical range translation failed in {} at {}: {:?}",
            self.address_space, self.address, self.reason
        )
    }
}

impl std::error::Error for CanonicalRangeTranslationError {}

/// Device-neutral boundary for validated CPU-VA to backing translation.
pub trait CanonicalRangeTranslator {
    /// Translates the complete virtual range or returns its first failing byte.
    fn translate_canonical_range(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        size: u64,
        required_permissions: MemoryPermissions,
    ) -> Result<CanonicalBackingRange, CanonicalRangeTranslationError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_tiled_cpu_snapshot_does_not_expand_unmodified_topology() {
        let allocation = CanonicalAllocation::zeroed(512 * 512 * 4, 4096).unwrap();
        let physical = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let range = CanonicalBackingRange::mapped(
            physical,
            CanonicalByteLayout {
                base: 0,
                columns: 512,
                rows: 512,
                element_bytes: 4,
                source_element_bytes: 4,
                source_byte_offset: 0,
                source_row_pitch: 2048,
                tile: Some(CanonicalByteTile::new(4, 1, vec![2, 1, 0, 3]).unwrap()),
            },
        )
        .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let before = range.metadata_bytes();
        allocation
            .write(3 * 4096 + 100, &[0x51, 0x62, 0x73, 0x84])
            .unwrap();
        let snapshots = dependency.snapshot_dirty_pages(&range, 4).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].0, 3 * 4096);
        let mut expected = vec![0; 4096];
        expected[100..104].copy_from_slice(&[0x73, 0x62, 0x51, 0x84]);
        assert_eq!(&*snapshots[0].1, expected);
        assert!(
            range.metadata_bytes() < before + 128 * 1024,
            "one dirty page must not materialize a million texel segments"
        );
        assert!(dependency.remains_current());
    }

    #[test]
    fn segment_subranges_keep_captured_mapping_and_bounds() {
        let allocation = crate::CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let whole = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let segment = whole.segments()[0].snapshot_subrange(17, 61).unwrap();
        let nested = segment.snapshot_subrange(3, 7).unwrap();
        let reference = whole.snapshot_subrange(20, 7).unwrap();
        assert_eq!(&nested, &reference.segments()[0]);
        assert_eq!(segment.snapshot_subrange(0, 61).unwrap(), segment);
        for (offset, size) in [(0, 0), (61, 1), (60, 2), (u64::MAX, 2)] {
            assert!(segment.snapshot_subrange(offset, size).is_err());
        }
    }

    #[test]
    fn compressed_topology_equality_matches_the_full_ordered_segments() {
        let allocation = crate::CanonicalAllocation::zeroed(8 * 4096, 4096).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let original = full.segments().to_vec();
        let mut variants = vec![original.clone(), original.clone()];
        let mut reversed = original.clone();
        reversed.reverse();
        variants.push(reversed);
        for field in 0..5 {
            let mut changed = original.clone();
            match field {
                0 => changed[3].permissions = MemoryPermissions::READ,
                1 => changed[3].mapping_generation = MappingGeneration::new(7),
                2 => changed[3].offset = 1,
                3 => changed[3].size -= 1,
                4 => changed[3].backing = changed[2].backing.clone(),
                _ => unreachable!(),
            }
            if field == 2 {
                changed[3].size -= 1;
            }
            variants.push(changed);
        }
        let mut split = original.clone();
        split[0].size = 2048;
        let mut tail = split[0].clone();
        tail.offset = 2048;
        split.insert(1, tail);
        variants.push(split);
        let ranges = variants
            .iter()
            .cloned()
            .map(|segments| CanonicalBackingRange::new(segments).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranges[0].layout.segment_runs().len(), 1);
        for (left, a) in ranges.iter().enumerate() {
            for (right, b) in ranges.iter().enumerate() {
                assert_eq!(
                    a == b,
                    variants[left] == variants[right],
                    "{left} vs {right}"
                );
                assert_eq!(
                    a.same_byte_mapping(b),
                    variants[left]
                        .iter()
                        .map(|s| (s.page(), s.offset(), s.size()))
                        .eq(variants[right]
                            .iter()
                            .map(|s| (s.page(), s.offset(), s.size()))),
                    "physical identity {left} vs {right}"
                );
            }
        }
    }
    use crate::{
        CanonicalAllocation, CanonicalBackingStore, CanonicalWriteBatch, ContentGeneration,
        CpuVisibilityRequest, DeviceVisibilityPoint, DeviceVisibilityRequest, GuestPhysicalPageId,
        NonCpuDeviceId, VisibilityCoordinatorError,
    };

    struct UnexpectedCpuVisibility;

    #[test]
    fn protected_multi_page_reads_and_snapshots_keep_native_admission_open() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        allocation.write(0, &vec![0x42; 0x2000]).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observation = CanonicalCpuWriteDependency::capture(&range).unwrap();
        // Reversed and repeated aliases must not take a physical lock twice.
        let alias = CanonicalBackingRange::new(vec![
            range.segments()[1].clone(),
            range.segments()[0].clone(),
            range.segments()[1].clone(),
        ])
        .unwrap();
        let alias_observation = CanonicalCpuWriteDependency::capture(&alias).unwrap();
        let gate = range
            .pages()
            .next()
            .unwrap()
            .store()
            .execution_gate()
            .clone();
        let active = gate.acquire_shared();
        let epoch = gate.epoch();
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            for _ in 0..8 {
                let mut bytes = vec![0; 0x3000];
                alias.read(0, &mut bytes).unwrap();
                assert!(bytes.iter().all(|&b| b == 0x42));
                assert_eq!(observation.snapshot_all(&range).unwrap().len(), 0x2000);
                assert!(
                    observation
                        .snapshot_dirty_pages(&range, 4)
                        .unwrap()
                        .is_empty()
                );
                alias_observation.rearm().unwrap();
            }
            send.send(()).unwrap();
        });
        let early = receive.recv_timeout(std::time::Duration::from_secs(2));
        drop(active);
        worker.join().unwrap();
        early.expect("protected reads and snapshots must finish during native execution");
        assert_eq!(gate.epoch(), epoch);
    }

    #[test]
    fn exact_confirmation_refreshes_only_equal_bytes_and_keeps_other_observers_stale() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let range = full.snapshot_subrange(0xffc, 24).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let older = CanonicalCpuWriteDependency::capture(&range).unwrap();
        allocation.write(0x80, &[1]).unwrap();
        assert!(!dependency.remains_current());
        let gate = allocation.pages()[0].store().execution_gate();
        let epoch = gate.epoch();
        let active = gate.acquire_shared();
        assert!(dependency.confirm_unchanged(&[0; 24]).unwrap());
        assert!(dependency.remains_current());
        assert!(!older.remains_current());
        assert_eq!(gate.epoch(), epoch);
        drop(active);
        allocation.write(0x1004, &[0x41]).unwrap();
        for _ in 0..2 {
            assert!(!dependency.confirm_unchanged(&[0; 24]).unwrap());
            assert!(
                !dependency.remains_current(),
                "a mismatch cannot acknowledge new bytes"
            );
        }
        allocation.write(0x1004, &[0]).unwrap();
        assert!(dependency.confirm_unchanged(&[0; 24]).unwrap());
        assert!(dependency.remains_current());
        assert!(!older.remains_current());
        assert_eq!(
            dependency.confirm_unchanged(&[0; 23]),
            Err(CanonicalRangeAccessError::DependencyMismatch)
        );
    }

    #[test]
    fn protected_host_write_snapshot_keeps_independent_observers_and_epochs() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observation = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let older = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let mut write = CanonicalWriteBatch::new();
        write.stage(&range, 0xffe, &[1, 2, 3, 4]).unwrap();
        write.commit_ordered().unwrap();
        assert!(!older.remains_current());
        let gate = range
            .pages()
            .next()
            .unwrap()
            .store()
            .execution_gate()
            .clone();
        let epoch = gate.epoch();
        let active = gate.acquire_shared();
        let input = range.clone();
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let snapshots = observation.snapshot_dirty_pages(&input, 4).unwrap();
            assert_eq!(&snapshots[0].1[0xffe..0x1002], &[1, 2, 3, 4]);
            assert!(observation.remains_current());
            send.send(()).unwrap();
        });
        let early = receive.recv_timeout(std::time::Duration::from_secs(2));
        drop(active);
        worker.join().unwrap();
        early.expect(
            "a host write retained protection; refreshing it must not drain native execution",
        );
        assert!(!older.remains_current());
        assert_eq!(gate.epoch(), epoch);
    }

    #[test]
    fn protected_scalar_read_uses_shared_admission_without_refreshing_dirty_observations() {
        for protected in [false, true] {
            let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
            allocation.write(12, &[1, 2, 3, 4]).unwrap();
            let range = allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap();
            let observation =
                protected.then(|| CanonicalCpuWriteDependency::capture(&range).unwrap());
            let gate = range
                .pages()
                .next()
                .unwrap()
                .store()
                .execution_gate()
                .clone();
            let lease = gate.acquire_shared();
            let initial_epoch = gate.epoch();
            let (send, receive) = std::sync::mpsc::channel();
            let input = range.clone();
            let worker = std::thread::spawn(move || {
                let mut bytes = [0; 4];
                let result = input.read(12, &mut bytes);
                send.send((result, bytes)).unwrap();
            });
            let early = receive.recv_timeout(std::time::Duration::from_millis(100));
            assert!(
                early.is_ok(),
                "page locks suffice when no writable native aliases exist"
            );
            drop(lease);
            let (result, bytes) = early.unwrap_or_else(|_| {
                receive
                    .recv_timeout(std::time::Duration::from_secs(3))
                    .unwrap()
            });
            worker.join().unwrap();
            result.unwrap();
            assert_eq!(bytes, [1, 2, 3, 4]);
            assert_eq!(gate.epoch(), initial_epoch);
            if let Some(observation) = observation {
                assert!(observation.remains_current());
                allocation.write(12, &[0x42]).unwrap();
                assert!(!observation.remains_current());
                let mut bytes = [0; 4];
                range.read(12, &mut bytes).unwrap();
                assert_eq!(bytes, [0x42, 2, 3, 4]);
                assert!(!observation.remains_current());
            }
        }
    }

    #[test]
    fn protected_observer_capture_does_not_stop_readers_and_keeps_independent_baselines() {
        for protected in [false, true] {
            for keep_existing in [false, true] {
                let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
                let range = allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap();
                let existing =
                    protected.then(|| CanonicalCpuWriteDependency::capture(&range).unwrap());
                let existing = if keep_existing { existing } else { None };
                let gate = range
                    .pages()
                    .next()
                    .unwrap()
                    .store()
                    .execution_gate()
                    .clone();
                let lease = gate.acquire_shared();
                let initial_epoch = gate.epoch();
                let (send, receive) = std::sync::mpsc::channel();
                let input = range.clone();
                let worker = std::thread::spawn(move || {
                    send.send(CanonicalCpuWriteDependency::capture(&input))
                        .unwrap();
                });
                let early = receive.recv_timeout(std::time::Duration::from_millis(100));
                assert!(
                    early.is_ok(),
                    "observer arm without native aliases requires no CPU stop"
                );
                drop(lease);
                let observation = early
                    .unwrap_or_else(|_| {
                        receive
                            .recv_timeout(std::time::Duration::from_secs(3))
                            .unwrap()
                    })
                    .unwrap();
                worker.join().unwrap();
                assert_eq!(gate.epoch(), initial_epoch);
                assert!(observation.remains_current());
                allocation.write(0, &[0x42]).unwrap();
                assert!(!observation.remains_current());
                if let Some(existing) = existing {
                    assert!(!existing.remains_current());
                }
                let refreshed = CanonicalCpuWriteDependency::capture(&range).unwrap();
                assert!(refreshed.remains_current());
                assert!(!observation.remains_current());
            }
        }
    }

    #[test]
    fn device_preparation_keeps_native_execution_running_before_ownership_handoff() {
        for protected in [false, true] {
            for kind in [
                crate::DeviceAccessKind::Read,
                crate::DeviceAccessKind::Write,
            ] {
                let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
                let range = allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap();
                if protected {
                    CanonicalCpuWriteDependency::capture(&range).unwrap();
                    let mut batch = CanonicalWriteBatch::new();
                    batch.stage(&range, 0, &[0x42]).unwrap();
                    batch.commit().unwrap();
                } else {
                    allocation.write(0, &[0x42]).unwrap();
                }
                let gate = range
                    .pages()
                    .next()
                    .unwrap()
                    .store()
                    .execution_gate()
                    .clone();
                let lease = gate.acquire_shared();
                let initial_epoch = gate.epoch();
                let (send, receive) = std::sync::mpsc::channel();
                let input = range.clone();
                let worker = std::thread::spawn(move || {
                    let declaration = DeviceAccessDeclaration::new(
                        NonCpuDeviceId::new(1),
                        kind,
                        DeviceVisibilityPoint::new(1),
                        kind.writes().then_some(DeviceVisibilityPoint::new(2)),
                    )
                    .unwrap();
                    let result = CanonicalBackingRange::prepare_resident_device_accesses(
                        [(&input, declaration)],
                        &UnexpectedCpuVisibility,
                    );
                    send.send(result).unwrap();
                });
                let early = receive.recv_timeout(std::time::Duration::from_millis(200));
                assert!(early.is_ok());
                drop(lease);
                early
                    .unwrap_or_else(|_| {
                        receive
                            .recv_timeout(std::time::Duration::from_secs(3))
                            .unwrap()
                    })
                    .unwrap();
                worker.join().unwrap();
                assert_eq!(gate.epoch(), initial_epoch);
                assert_eq!(
                    range.pages().next().unwrap().visibility_state(),
                    VisibilityState::Clean
                );
                let mut bytes = [0];
                allocation.read(0, &mut bytes).unwrap();
                assert_eq!(bytes, [0x42]);
            }
        }
    }

    #[test]
    fn indexed_subranges_preserve_fragment_order_and_exact_boundaries() {
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut expected = Vec::new();
        let mut segments = Vec::new();
        for (index, offset, size, byte) in [
            (2, 17, 10, 0x11),
            (0, 300, 25, 0x22),
            (2, 40, 7, 0x33),
            (1, 5, 19, 0x44),
        ] {
            allocation
                .write(index * 0x1000 + offset, &vec![byte; size])
                .unwrap();
            segments.push(
                CanonicalBackingSegment::new(
                    full.segments()[index].backing().clone(),
                    offset as u64,
                    size as u64,
                    MemoryPermissions::READ_WRITE,
                    MappingGeneration::new(index as u64 + 1),
                )
                .unwrap(),
            );
            expected.extend(std::iter::repeat_n(byte, size));
        }
        let range = CanonicalBackingRange::new(segments).unwrap();
        for offset in 0..=expected.len() {
            for size in 0..=expected.len() - offset {
                let mut bytes = vec![0; size];
                range.read(offset as u64, &mut bytes).unwrap();
                assert_eq!(bytes, expected[offset..offset + size]);
                if size != 0 {
                    range
                        .snapshot_subrange(offset as u64, size as u64)
                        .unwrap()
                        .read(0, &mut bytes)
                        .unwrap();
                    assert_eq!(bytes, expected[offset..offset + size]);
                }
            }
        }
        let mut writes = CanonicalWriteBatch::new();
        writes.stage(&range, 8, &[0x55; 40]).unwrap();
        assert!(!writes.overlaps(&range, 0, 8).unwrap());
        assert!(writes.overlaps(&range, 8, 40).unwrap());
        assert!(!writes.overlaps(&range, 48, 13).unwrap());
        writes.read_staged(&range, 0, &mut expected).unwrap();
        assert_eq!(&expected[8..48], &[0x55; 40]);
        writes.commit().unwrap();
        let mut actual = vec![0; expected.len()];
        range.read(0, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn publication_batch_revokes_all_aliases_once_and_invalidates_executable_pages() {
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x2000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x2000).unwrap();
        let clone = a.clone();
        let log = Arc::new(crate::MemoryInvalidationLog::default());
        let arenas = [
            crate::DirectArena::new(0x5000).unwrap(),
            crate::DirectArena::new(0x5000).unwrap(),
        ];
        for (index, page) in full.pages().enumerate() {
            assert!(page.observe_executable_content(log.clone()));
            for arena in &arenas {
                let backing = page.direct_backing().unwrap();
                let address = (index as u64 + 1) * 0x1000;
                arena
                    .map_pages(&[crate::DirectMapRequest {
                        guest_address: address,
                        backing: &backing,
                        protection: crate::DirectProtection::Read,
                    }])
                    .unwrap();
                page.register_direct_alias(arena, address, crate::DirectProtection::ReadWrite)
                    .unwrap();
            }
        }
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&a, write), (&b, write)],
            coordinator.as_ref(),
        )
        .unwrap();
        let gate = full.pages().next().unwrap().store().execution_gate();
        let epoch = gate.epoch();
        let summary = full.visibility_summary().load(Ordering::Acquire);
        let cursor = log.cursor();
        CanonicalBackingRange::publish_device_writes(
            [(&a, write), (&clone, write), (&b, write)],
            coordinator,
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert!(!gate.transition_pending());
        assert_eq!(
            full.visibility_summary().load(Ordering::Acquire),
            summary + 3
        );
        let mut changes = Vec::new();
        log.read_since(cursor, &mut changes).unwrap();
        assert_eq!(changes.len(), 3);
        assert!(
            changes
                .iter()
                .all(|change| change.origin == crate::MemoryInvalidationOrigin::DeviceWrite)
        );
        for arena in &arenas {
            for address in [0x1000, 0x2000, 0x3000] {
                assert_eq!(
                    arena.protection_at(address),
                    Some(crate::DirectProtection::None)
                );
            }
        }
    }

    #[test]
    fn failed_alias_protection_preserves_error_and_terminal_invalidation_releases_execution() {
        let memory = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = memory.backing_range(MemoryPermissions::READ_WRITE).unwrap();
        let arena = crate::DirectArena::new(0x4000).unwrap();
        for (index, page) in range.pages().enumerate() {
            let backing = page.direct_backing().unwrap();
            let address = (index as u64 + 1) * 0x1000;
            arena
                .map_pages(&[crate::DirectMapRequest {
                    guest_address: address,
                    backing: &backing,
                    protection: crate::DirectProtection::Read,
                }])
                .unwrap();
            page.register_direct_alias(&arena, address, crate::DirectProtection::ReadWrite)
                .unwrap();
        }
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, write)],
            coordinator.as_ref(),
        )
        .unwrap();
        // Inject an invalid retained alias through the real registration
        // boundary. A terminal transition must preserve the host-range error
        // even when protection fails before all aliases have been revoked.
        assert!(
            memory.pages()[1]
                .register_direct_alias(&arena, 0x4000, crate::DirectProtection::ReadWrite)
                .is_err()
        );
        let error = CanonicalBackingRange::publish_device_writes([(&range, write)], coordinator)
            .unwrap_err();
        assert!(matches!(error, VisibilityError::HostMemory(_)));
        assert!(error.to_string().contains("outside the reservation"));
        let gate = range.pages().next().unwrap().store().execution_gate();
        assert!(!gate.transition_pending());
        assert!(matches!(
            CanonicalBackingRange::invalidate_visibility_ranges([&range]),
            Err(VisibilityError::HostMemory(_))
        ));
        assert!(
            range
                .pages()
                .all(|page| page.visibility_state() == VisibilityState::Invalid)
        );
        assert!(!gate.transition_pending());
    }

    #[test]
    fn overlapping_batch_owners_never_advance_untouched_pages() {
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x2000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x2000).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let device = NonCpuDeviceId::new(1);
        let write = |point| {
            DeviceAccessDeclaration::write(
                device,
                DeviceVisibilityPoint::new(point),
                DeviceVisibilityPoint::new(point),
            )
            .unwrap()
        };
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&a, write(1)), (&b, write(1))],
            coordinator.as_ref(),
        )
        .unwrap();
        CanonicalBackingRange::publish_device_writes(
            [(&a, write(1)), (&b, write(1))],
            coordinator.clone(),
        )
        .unwrap();
        let gate = full.pages().next().unwrap().store().execution_gate();
        let epoch = gate.epoch();
        let shared = gate.acquire_shared();
        CanonicalBackingRange::publish_device_writes([(&a, write(2))], coordinator.clone())
            .unwrap();
        assert_eq!(gate.epoch(), epoch);
        let points = full
            .pages()
            .map(|page| page.visibility_state())
            .collect::<Vec<_>>();
        assert_eq!(
            points,
            [2, 2, 1].map(|point| VisibilityState::GpuNewer {
                device,
                visible_at: DeviceVisibilityPoint::new(point)
            })
        );
        CanonicalBackingRange::publish_device_writes([(&b, write(3))], coordinator).unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert_eq!(
            full.pages()
                .map(|page| page.visibility_state())
                .collect::<Vec<_>>(),
            [2, 3, 3].map(|point| VisibilityState::GpuNewer {
                device,
                visible_at: DeviceVisibilityPoint::new(point)
            })
        );
        drop(shared);
    }

    #[test]
    fn published_aliases_download_current_bytes_once_per_physical_page_without_exclusion() {
        struct Readback {
            gate: crate::ExecutionGate,
            requests: Mutex<Vec<CpuVisibilityRequest>>,
        }
        impl VisibilityCoordinator for Readback {
            fn cache_cpu_page(
                &self,
                _: DeviceVisibilityRequest,
                _: &[u8],
            ) -> Result<(), VisibilityCoordinatorError> {
                Ok(())
            }
            fn make_cpu_visible(
                &self,
                request: CpuVisibilityRequest,
            ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
                let gate = self.gate.clone();
                let (sent, received) = std::sync::mpsc::channel();
                let worker = std::thread::spawn(move || {
                    let _cpu = gate.acquire_shared();
                    sent.send(()).unwrap();
                });
                received
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .expect("readback must not exclude CPU execution");
                worker.join().unwrap();
                self.requests.lock().unwrap().push(request);
                Ok(vec![request.visible_at.get() as u8; request.size].into_boxed_slice())
            }
        }
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x2000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x2000).unwrap();
        let readback = Arc::new(Readback {
            gate: full
                .pages()
                .next()
                .unwrap()
                .store()
                .execution_gate()
                .clone(),
            requests: Mutex::new(Vec::new()),
        });
        let device = NonCpuDeviceId::new(1);
        let write = |point| {
            DeviceAccessDeclaration::write(
                device,
                DeviceVisibilityPoint::new(point),
                DeviceVisibilityPoint::new(point),
            )
            .unwrap()
        };
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&a, write(1)), (&b, write(1))],
            readback.as_ref(),
        )
        .unwrap();
        CanonicalBackingRange::publish_device_writes(
            [(&a, write(1)), (&b, write(1))],
            readback.clone(),
        )
        .unwrap();
        CanonicalBackingRange::publish_device_writes([(&a, write(2))], readback.clone()).unwrap();
        let mut bytes = vec![0; 0x2000];
        b.read(0, &mut bytes).unwrap();
        assert_eq!(&bytes[..0x1000], &[2; 0x1000]);
        assert_eq!(&bytes[0x1000..], &[1; 0x1000]);
        a.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, vec![2; 0x2000]);
        let mut all = vec![0; 0x3000];
        full.read(0, &mut all).unwrap();
        assert_eq!(&all[..0x2000], &[2; 0x2000]);
        assert_eq!(&all[0x2000..], &[1; 0x1000]);
        let requests = readback.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests
                .iter()
                .map(|request| request.page)
                .collect::<BTreeSet<_>>()
                .len(),
            3
        );
    }

    #[test]
    fn publication_rejects_mixed_points_before_advancing_cached_ownership() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x1000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x1000).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let device = NonCpuDeviceId::new(1);
        let write = |point| {
            DeviceAccessDeclaration::write(
                device,
                DeviceVisibilityPoint::new(point),
                DeviceVisibilityPoint::new(point),
            )
            .unwrap()
        };
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&a, write(1)), (&b, write(1))],
            coordinator.as_ref(),
        )
        .unwrap();
        CanonicalBackingRange::publish_device_writes(
            [(&a, write(1)), (&b, write(1))],
            coordinator.clone(),
        )
        .unwrap();
        assert_eq!(
            CanonicalBackingRange::publish_device_writes(
                [(&a, write(2)), (&b, write(3))],
                coordinator
            ),
            Err(VisibilityError::IncompatibleDeclarations)
        );
        assert!(full.pages().all(|page| page.visibility_state()
            == VisibilityState::GpuNewer {
                device,
                visible_at: DeviceVisibilityPoint::new(1)
            }));
    }

    #[test]
    fn publication_and_invalidation_deduplicate_distinct_stores_sharing_one_gate() {
        let gate = crate::ExecutionGate::new();
        let pages = (0..2)
            .map(|_| {
                let store =
                    CanonicalBackingStore::allocate_with_execution_gate(gate.clone()).unwrap();
                CanonicalBackingPage::zeroed(
                    &store,
                    GuestPhysicalPageId::new(1),
                    0x1000,
                    ContentGeneration::INITIAL,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let ranges = pages
            .iter()
            .map(|page| {
                CanonicalBackingRange::new(vec![
                    CanonicalBackingSegment::new(
                        page.clone(),
                        0,
                        0x1000,
                        MemoryPermissions::READ_WRITE,
                        MappingGeneration::INITIAL,
                    )
                    .unwrap(),
                ])
                .unwrap()
            })
            .collect::<Vec<_>>();
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(1),
        )
        .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        CanonicalBackingRange::prepare_resident_device_accesses(
            ranges.iter().map(|range| (range, write)),
            coordinator.as_ref(),
        )
        .unwrap();
        let epoch = gate.epoch();
        CanonicalBackingRange::publish_device_writes(
            ranges.iter().rev().map(|range| (range, write)),
            coordinator,
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        CanonicalBackingRange::invalidate_visibility_ranges(
            ranges.iter().chain(ranges.iter().rev()),
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert!(
            pages
                .iter()
                .all(|page| page.visibility_state() == VisibilityState::Invalid)
        );
        assert!(!gate.transition_pending());
    }

    #[test]
    fn reversed_multi_gate_topology_is_safe_under_contention() {
        let stores = [
            CanonicalBackingStore::allocate().unwrap(),
            CanonicalBackingStore::allocate().unwrap(),
        ];
        let pages = stores
            .iter()
            .map(|store| {
                CanonicalBackingPage::zeroed(
                    store,
                    GuestPhysicalPageId::new(1),
                    0x1000,
                    ContentGeneration::INITIAL,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let segments = pages
            .iter()
            .map(|page| {
                CanonicalBackingSegment::new(
                    page.clone(),
                    0,
                    0x1000,
                    MemoryPermissions::READ_WRITE,
                    MappingGeneration::INITIAL,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let forward = CanonicalBackingRange::new(segments.clone()).unwrap();
        let reversed = CanonicalBackingRange::new(segments.into_iter().rev().collect()).unwrap();
        let start = Arc::new(std::sync::Barrier::new(2));
        let (done, received) = std::sync::mpsc::channel();
        let workers = [forward, reversed]
            .into_iter()
            .map(|range| {
                let start = start.clone();
                let done = done.clone();
                std::thread::spawn(move || {
                    start.wait();
                    for _ in 0..32 {
                        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
                        assert_eq!(dependency.snapshot_all(&range).unwrap().len(), 0x2000);
                    }
                    done.send(()).unwrap();
                })
            })
            .collect::<Vec<_>>();
        for _ in 0..2 {
            received
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("multi-gate operations must share one acquisition order");
        }
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn failed_cache_batch_invalidates_changed_pages_and_revokes_all_aliases() {
        struct FailedCache(std::sync::atomic::AtomicUsize);
        impl VisibilityCoordinator for FailedCache {
            fn cache_cpu_page(
                &self,
                _: DeviceVisibilityRequest,
                _: &[u8],
            ) -> Result<(), VisibilityCoordinatorError> {
                if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(())
                } else {
                    Err(VisibilityCoordinatorError::new(
                        "injected page-cache failure",
                    ))
                }
            }
            fn make_cpu_visible(
                &self,
                _: CpuVisibilityRequest,
            ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
                unreachable!()
            }
        }
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let arena = crate::DirectArena::new(0x5000).unwrap();
        for (index, page) in range.pages().enumerate() {
            let backing = page.direct_backing().unwrap();
            let address = 0x1000 + index as u64 * 0x1000;
            arena
                .map_pages(&[crate::DirectMapRequest {
                    guest_address: address,
                    backing: &backing,
                    protection: crate::DirectProtection::Read,
                }])
                .unwrap();
            page.register_direct_alias(&arena, address, crate::DirectProtection::ReadWrite)
                .unwrap();
        }
        let alias = range.snapshot_subrange(0x800, 0x1000).unwrap();
        let write = DeviceAccessDeclaration::read_write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        assert!(
            CanonicalBackingRange::prepare_resident_device_accesses(
                [(&range, write), (&alias, write)],
                &FailedCache(std::sync::atomic::AtomicUsize::new(0)),
            )
            .is_err()
        );
        assert!(
            range
                .pages()
                .all(|page| page.visibility_state() == VisibilityState::Invalid)
        );
        assert_eq!(
            arena.protection_at(0x1000),
            Some(crate::DirectProtection::None)
        );
        assert_eq!(
            arena.protection_at(0x2000),
            Some(crate::DirectProtection::None)
        );
    }

    #[test]
    fn batched_overwrite_rearms_without_downloading_device_owned_bytes() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, write)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, write)], coordinator)
            .unwrap();
        let before = range
            .pages()
            .map(CanonicalBackingPage::visibility_state)
            .collect::<Vec<_>>();
        let bytes = CanonicalCpuWriteDependency::snapshot_batch_with_resolver(
            &[CpuWriteSnapshotRequest {
                dependency: &dependency,
                range: &range,
                selection: CpuWriteSnapshotSelection::Rearm,
                alignment: 1,
            }],
            &mut |_, _| panic!("overwrite tracking must not download old bytes"),
        )
        .unwrap();
        assert!(bytes[0].is_empty());
        assert_eq!(
            range
                .pages()
                .map(CanonicalBackingPage::visibility_state)
                .collect::<Vec<_>>(),
            before
        );
    }

    #[test]
    fn resident_batch_unions_aliases_and_keeps_clean_reads_without_exclusion() {
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x2000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x2000).unwrap();
        let alias = full.snapshot_subrange(0, 0x3000).unwrap();
        let gate = full.segments()[0].backing().store().execution_gate();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let read =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(1), DeviceVisibilityPoint::new(7));
        allocation.write(0, &[1]).unwrap();
        allocation.write(0x1000, &[2]).unwrap();
        allocation.write(0x2000, &[3]).unwrap();
        let epoch = gate.epoch();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&a, read), (&b, read), (&alias, read)],
            coordinator.as_ref(),
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert!(
            full.pages()
                .all(|page| page.visibility_state() == VisibilityState::Clean)
        );
        let active = gate.acquire_shared();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(&a, read), (&b, read), (&alias, read)],
            coordinator.as_ref(),
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        drop(active);
    }

    #[test]
    fn batches_do_not_merge_incompatible_visibility_points() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x1000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x1000).unwrap();
        let gate = full.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        let device = NonCpuDeviceId::new(1);
        let first = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        let later = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(3),
        )
        .unwrap();
        assert_eq!(
            CanonicalBackingRange::prepare_resident_device_accesses(
                [(&a, first), (&b, later)],
                Arc::new(UnexpectedCpuVisibility).as_ref()
            ),
            Err(VisibilityError::IncompatibleDeclarations)
        );
        assert_eq!(gate.epoch(), epoch);
    }

    #[test]
    fn snapshot_batch_shares_exclusion_but_not_resource_observations() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let alias = full.snapshot_subrange(0x100, 0x1100).unwrap();
        let a = CanonicalCpuWriteDependency::capture(&full).unwrap();
        let b = CanonicalCpuWriteDependency::capture(&alias).unwrap();
        let old = CanonicalCpuWriteDependency::capture(&full).unwrap();
        allocation.write(0x180, &[0x42]).unwrap();
        let requests = [
            CpuWriteSnapshotRequest {
                dependency: &a,
                range: &full,
                selection: CpuWriteSnapshotSelection::DirtyPages,
                alignment: 4,
            },
            CpuWriteSnapshotRequest {
                dependency: &b,
                range: &alias,
                selection: CpuWriteSnapshotSelection::DirtyPages,
                alignment: 16,
            },
        ];
        let gate = full.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        let samples =
            CanonicalCpuWriteDependency::snapshot_batch_with_resolver(&requests, &mut |_, _| {
                panic!("CPU-only snapshot does not need readback")
            })
            .unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert_eq!(samples[0][0].1[0x180], 0x42);
        assert_eq!(samples[1][0].1[0x80], 0x42);
        assert!(a.remains_current() && b.remains_current());
        assert!(!old.remains_current());
        let active = gate.acquire_shared();
        let clean = CanonicalCpuWriteDependency::snapshot_batch_with_resolver(
            &requests,
            &mut |_, _| unreachable!(),
        )
        .unwrap();
        assert!(clean.iter().all(Vec::is_empty));
        assert_eq!(gate.epoch(), epoch);
        drop(active);
    }

    #[test]
    fn snapshot_batch_releases_exclusion_and_retains_samples_across_readback() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let a = full.snapshot_subrange(0, 0x1000).unwrap();
        let b = full.snapshot_subrange(0x1000, 0x1000).unwrap();
        let first = CanonicalCpuWriteDependency::capture(&a).unwrap();
        let second = CanonicalCpuWriteDependency::capture(&b).unwrap();
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&b, write)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&b, write)], coordinator).unwrap();
        allocation.write(0x80, &[0x11]).unwrap();
        let mut calls = 0;
        let samples = CanonicalCpuWriteDependency::snapshot_batch_with_resolver(
            &[
                CpuWriteSnapshotRequest {
                    dependency: &first,
                    range: &a,
                    selection: CpuWriteSnapshotSelection::DirtyPages,
                    alignment: 4,
                },
                CpuWriteSnapshotRequest {
                    dependency: &second,
                    range: &b,
                    selection: CpuWriteSnapshotSelection::All,
                    alignment: 1,
                },
            ],
            &mut |_, request| {
                calls += 1;
                // This write acquires the same gate. No snapshot/adaptive/page guard
                // may survive the callback, and it must not erase the earlier sample.
                allocation.write(0x80, &[0x22]).unwrap();
                Ok(vec![0x55; request.size].into_boxed_slice())
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(samples[0][0].1[0x80], 0x11);
        assert_eq!(samples[1][0].1[0], 0x55);
        assert!(!first.remains_current());
        assert_eq!(first.snapshot_dirty_pages(&a, 4).unwrap()[0].1[0x80], 0x22);
    }

    #[test]
    fn distinct_stores_sharing_one_gate_do_not_reacquire_it() {
        let (done, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let gate = crate::ExecutionGate::new();
            let stores = [
                CanonicalBackingStore::allocate_with_execution_gate(gate.clone()).unwrap(),
                CanonicalBackingStore::allocate_with_execution_gate(gate.clone()).unwrap(),
            ];
            let pages = stores
                .iter()
                .map(|store| {
                    CanonicalBackingPage::zeroed(
                        store,
                        GuestPhysicalPageId::new(1),
                        0x1000,
                        ContentGeneration::INITIAL,
                    )
                    .unwrap()
                })
                .map(|page| {
                    CanonicalBackingSegment::new(
                        page,
                        0,
                        0x1000,
                        MemoryPermissions::READ_WRITE,
                        MappingGeneration::INITIAL,
                    )
                    .unwrap()
                })
                .collect();
            let range = CanonicalBackingRange::new(pages).unwrap();
            assert_eq!(range.execution_gates().count(), 1);
            let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
            assert_eq!(dependency.snapshot_all(&range).unwrap().len(), 0x2000);
            dependency.rearm().unwrap();
            let mut writes = CanonicalWriteBatch::new();
            writes.stage(&range, 0xfff, &[0x11, 0x22]).unwrap();
            writes.commit().unwrap();
            let read = DeviceAccessDeclaration::read(
                NonCpuDeviceId::new(1),
                DeviceVisibilityPoint::new(7),
            );
            CanonicalBackingRange::prepare_resident_device_accesses(
                [(&range, read)],
                Arc::new(UnexpectedCpuVisibility).as_ref(),
            )
            .unwrap();
            done.send(()).unwrap();
        });
        received
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("shared gate must not self-deadlock");
        worker.join().unwrap();
    }

    impl VisibilityCoordinator for UnexpectedCpuVisibility {
        fn cache_cpu_page(
            &self,
            _request: DeviceVisibilityRequest,
            _canonical_bytes: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            Ok(())
        }

        fn make_cpu_visible(
            &self,
            _request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            panic!("a clean CPU-write snapshot must not request GPU materialization")
        }
    }

    #[test]
    fn segment_construction_checks_bounds() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();

        assert_eq!(
            CanonicalBackingSegment::new(
                page.clone(),
                0xfff,
                2,
                MemoryPermissions::READ,
                MappingGeneration::new(1),
            ),
            Err(CanonicalRangeError::InvalidSegmentBounds)
        );
    }

    #[test]
    fn retained_range_reads_checked_subranges_across_pages() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let first = CanonicalBackingPage::initialized(
            &store,
            GuestPhysicalPageId::new(1),
            &[0x10, 0x11, 0x12, 0x13],
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let second = CanonicalBackingPage::initialized(
            &store,
            GuestPhysicalPageId::new(2),
            &[0x20, 0x21, 0x22, 0x23],
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                first,
                1,
                3,
                MemoryPermissions::READ,
                MappingGeneration::new(1),
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second,
                0,
                3,
                MemoryPermissions::READ,
                MappingGeneration::new(2),
            )
            .unwrap(),
        ])
        .unwrap();

        let mut bytes = [0_u8; 4];
        range.read(1, &mut bytes).unwrap();
        assert_eq!(bytes, [0x12, 0x13, 0x20, 0x21]);
        assert_eq!(
            range.read(5, &mut [0_u8; 2]),
            Err(CanonicalRangeAccessError::OutOfBounds {
                offset: 5,
                size: 2,
                range_size: 6,
            })
        );
    }

    #[test]
    fn snapshot_subrange_retains_exact_bytes_and_page_identity() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let first = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let second = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(2),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                first.clone(),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::new(1),
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second.clone(),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::new(2),
            )
            .unwrap(),
        ])
        .unwrap();

        first.prepare_write().unwrap();
        first
            .write_preflighted(
                0xff0,
                &[0x5a; 0x10],
                ContentGeneration::INITIAL,
                ContentGeneration::new(1),
            )
            .unwrap();
        second.prepare_write().unwrap();
        second
            .write_preflighted(
                0,
                &[0xa5; 0x10],
                ContentGeneration::INITIAL,
                ContentGeneration::new(1),
            )
            .unwrap();
        let snapshot = range.snapshot_subrange(0xff0, 0x20).unwrap();
        let mut appended = Vec::new();
        range
            .snapshot_subrange_into(0xff0, 0x20, &mut appended)
            .unwrap();
        assert_eq!(
            CanonicalBackingRange::new(appended.clone()).unwrap(),
            snapshot
        );

        assert_eq!(snapshot.size(), 0x20);
        assert_eq!(snapshot.segments().len(), 2);
        assert_eq!(snapshot.segments()[0].offset(), 0xff0);
        assert_eq!(snapshot.segments()[0].page(), first.identity());
        assert_eq!(snapshot.segments()[1].page(), second.identity());
        let mut bytes = [0; 0x20];
        snapshot.read(0, &mut bytes).unwrap();
        assert_eq!(&bytes[..0x10], &[0x5a; 0x10]);
        assert_eq!(&bytes[0x10..], &[0xa5; 0x10]);

        assert_eq!(
            range.snapshot_subrange(0, 0),
            Err(CanonicalRangeError::InvalidSubrange)
        );
        assert_eq!(
            range.snapshot_subrange(0x1ff0, 0x20),
            Err(CanonicalRangeError::InvalidSubrange)
        );
        let original = appended.clone();
        assert_eq!(
            range.snapshot_subrange_into(0x1ff0, 0x20, &mut appended),
            Err(CanonicalRangeError::InvalidSubrange)
        );
        assert_eq!(appended, original);
    }

    #[test]
    fn cloning_a_canonical_range_shares_its_immutable_segments() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                page,
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::INITIAL,
            )
            .unwrap(),
        ])
        .unwrap();

        let cloned = range.clone();

        assert!(Arc::ptr_eq(&range.layout, &cloned.layout));
        assert_eq!(range, cloned);
    }

    #[test]
    fn range_layout_deduplicates_physical_aliases_and_orders_store_transitions() {
        let first = CanonicalBackingStore::allocate().unwrap();
        let second = CanonicalBackingStore::allocate().unwrap();
        let a = CanonicalBackingPage::zeroed(
            &first,
            GuestPhysicalPageId::new(3),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let b = CanonicalBackingPage::zeroed(
            &second,
            GuestPhysicalPageId::new(3),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let c = CanonicalBackingPage::zeroed(
            &first,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let segment = |page| {
            CanonicalBackingSegment::new(
                page,
                0,
                0x100,
                MemoryPermissions::READ_WRITE,
                MappingGeneration::INITIAL,
            )
            .unwrap()
        };
        let range = CanonicalBackingRange::new(vec![
            segment(b.clone()),
            segment(a.clone()),
            segment(b.clone()),
            segment(c.clone()),
            segment(a.clone()),
        ])
        .unwrap();
        assert_eq!(range.size(), 0x500);
        assert_eq!(
            range
                .pages()
                .map(CanonicalBackingPage::identity)
                .collect::<Vec<_>>(),
            [c.identity(), a.identity(), b.identity()]
        );
        assert_eq!(
            range
                .execution_gates()
                .map(crate::ExecutionGate::identity)
                .collect::<Vec<_>>(),
            {
                let mut ids = [
                    first.execution_gate().identity(),
                    second.execution_gate().identity(),
                ];
                ids.sort_unstable();
                ids
            }
        );
        let subrange = range.snapshot_subrange(0x100, 0x100).unwrap();
        assert_eq!(
            subrange
                .pages()
                .map(CanonicalBackingPage::identity)
                .collect::<Vec<_>>(),
            [a.identity()]
        );
        assert_eq!(subrange.execution_gates().count(), 1);
    }

    #[test]
    fn cpu_write_dependency_invalidates_at_page_granularity() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observed = mapped.snapshot_subrange(0x100, 0x100).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&observed).unwrap();

        assert!(dependency.remains_current());
        allocation.write(0x1100, &[2]).unwrap();
        assert!(dependency.remains_current());
        allocation.write(0x300, &[1]).unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn cpu_write_dependency_normalizes_repeated_page_segments() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = mapped.snapshot_subrange(0x100, 0x100).unwrap();
        let second = mapped.snapshot_subrange(0x300, 0x100).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture_ranges([&first, &second]).unwrap();

        allocation.write(0x280, &[1]).unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn cpu_write_dependency_captures_ranges_translated_at_different_epochs() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = mapped.snapshot_subrange(0, 0x1000).unwrap();
        allocation.write(0x1000, &[1]).unwrap();
        let second = mapped.snapshot_subrange(0x1000, 0x1000).unwrap();

        let dependency = CanonicalCpuWriteDependency::capture_ranges([&first, &second]).unwrap();

        assert!(dependency.remains_current());
        allocation.write(0x100, &[2]).unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn cpu_write_dependency_observes_only_committed_page_writes() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observed = mapped.snapshot_subrange(0x100, 0x100).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&observed).unwrap();

        let mut disjoint = CanonicalWriteBatch::new();
        disjoint.stage(&mapped, 0x300, &[1]).unwrap();
        assert!(dependency.remains_current());
        disjoint.commit().unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn streaming_pages_copy_only_changes_and_recover_after_quiet_samples() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        for cycle in 1..=8 {
            allocation.pages()[0].prepare_cpu_write().unwrap();
            allocation.write(0, &[cycle]).unwrap();
            let snapshots = dependency.snapshot_dirty_pages(&range, 4).unwrap();
            assert_eq!(snapshots.len(), 1);
            assert_eq!(snapshots[0].0, 0);
            assert_eq!(snapshots[0].1.len(), 0x1000);
            assert_eq!(snapshots[0].1[0], cycle);
        }
        assert!(dependency.has_streaming_pages());
        // Another page remains protected and independently dirtied.
        allocation.write(0x9000, &[0x55]).unwrap();
        let snapshots = dependency.snapshot_dirty_pages(&range, 4).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].0, 0x9000);
        for _ in 1..STREAMING_QUIET_THRESHOLD {
            assert!(
                dependency
                    .snapshot_dirty_pages(&range, 4)
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(!dependency.has_streaming_pages());
        assert!(dependency.remains_current());
        allocation.write(0, &[0x66]).unwrap();
        assert!(!dependency.remains_current());
        assert_eq!(
            dependency.snapshot_dirty_pages(&range, 4).unwrap()[0].1[0],
            0x66
        );
    }

    #[test]
    fn snapshotting_one_range_does_not_acknowledge_other_dirty_ranges() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = range.snapshot_subrange(0, 0x1000).unwrap();
        let second = range.snapshot_subrange(0x1000, 0x1000).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture_ranges([&first, &second]).unwrap();
        allocation.write(0x1000, &[0x44]).unwrap();
        assert_eq!(dependency.snapshot_all(&first).unwrap()[0], 0);
        assert!(!dependency.remains_current());
        assert_eq!(
            dependency.snapshot_dirty_pages(&second, 1).unwrap()[0].1[0],
            0x44
        );
        assert!(dependency.remains_current());
    }

    #[test]
    fn multi_page_observer_rearm_preserves_alias_permissions_and_independent_epochs() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let arena = crate::DirectArena::new(0x40000).unwrap();
        for (index, page) in range.pages().enumerate() {
            let backing = page.direct_backing().unwrap();
            for (base, maximum) in [
                (0x1000, crate::DirectProtection::ReadWrite),
                (0x20000, crate::DirectProtection::Read),
            ] {
                let address = base + index as u64 * 0x1000;
                arena
                    .map_pages(&[crate::DirectMapRequest {
                        guest_address: address,
                        backing: &backing,
                        protection: crate::DirectProtection::Read,
                    }])
                    .unwrap();
                page.register_direct_alias(&arena, address, maximum)
                    .unwrap();
            }
        }
        let first = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let second = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let mut expected = vec![0; 0x10000];
        for index in 0..16 {
            expected[index * 0x1000] = index as u8 + 1;
            allocation
                .write(index * 0x1000, &[index as u8 + 1])
                .unwrap();
        }
        assert!(!first.remains_current());
        assert!(!second.remains_current());
        assert_eq!(&*first.snapshot_all(&range).unwrap(), expected);
        assert!(first.remains_current());
        assert!(!second.remains_current());
        for index in 0..16 {
            for base in [0x1000, 0x20000] {
                assert_eq!(
                    arena.protection_at(base + index * 0x1000),
                    Some(crate::DirectProtection::Read)
                );
            }
        }
        assert_eq!(&*second.snapshot_all(&range).unwrap(), expected);
        allocation.write(0xfffc, &[0xaa; 4]).unwrap();
        first.rearm().unwrap();
        assert!(first.remains_current());
        assert!(!second.remains_current());
        assert_eq!(
            arena.protection_at(0x10000),
            Some(crate::DirectProtection::Read)
        );
        assert_eq!(
            arena.protection_at(0x2f000),
            Some(crate::DirectProtection::Read)
        );
        expected[0xfffc..].fill(0xaa);
        assert_eq!(&*second.snapshot_all(&range).unwrap(), expected);
    }

    #[test]
    fn explicit_rearm_restores_protected_tracking_after_overwrites() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();

        for cycle in 0..8 {
            allocation.write(0, &[cycle]).unwrap();
            assert!(!dependency.remains_current());
            dependency.rearm().unwrap();
            assert!(dependency.remains_current());
        }
        assert!(!dependency.has_streaming_pages());
    }

    #[test]
    fn rearm_preserves_invalid_backing_errors() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        crate::CanonicalBackingRange::invalidate_visibility_ranges([&range]).unwrap();
        assert_eq!(
            dependency.rearm(),
            Err(CanonicalRangeAccessError::Backing(
                CanonicalPageError::Visibility(VisibilityError::InvalidState),
            ))
        );
        assert!(!dependency.has_streaming_pages());
    }

    #[test]
    fn initial_tracking_capture_preserves_empty_and_invalid_backing_errors() {
        assert_eq!(
            CanonicalCpuWriteDependency::capture_ranges([]).unwrap_err(),
            CanonicalRangeAccessError::IncompleteRange
        );
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        crate::CanonicalBackingRange::invalidate_visibility_ranges([&range]).unwrap();
        assert_eq!(
            CanonicalCpuWriteDependency::capture(&range).unwrap_err(),
            CanonicalRangeAccessError::Backing(CanonicalPageError::Visibility(
                VisibilityError::InvalidState
            ))
        );
    }

    #[test]
    fn dirty_snapshot_rearms_before_a_later_write_can_resume() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();

        allocation.write(7, &[0x11]).unwrap();
        let dirty = dependency.snapshot_dirty_pages(&range, 4).unwrap();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0].0, 0);
        assert_eq!(dirty[0].1[7], 0x11);
        assert!(dependency.remains_current());

        allocation.write(7, &[0x22]).unwrap();
        assert!(!dependency.remains_current());
        let dirty = dependency.snapshot_dirty_pages(&range, 4).unwrap();
        assert_eq!(dirty[0].1[7], 0x22);
    }

    #[test]
    fn clean_snapshot_does_not_materialize_device_newer_pages() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, declaration)], coordinator)
            .unwrap();

        assert!(dependency.remains_current());
        assert_eq!(dependency.snapshot_whole_if_dirty(&range).unwrap(), None);
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::GpuNewer {
                device: NonCpuDeviceId::new(1),
                visible_at: DeviceVisibilityPoint::new(2),
            }
        );
    }

    #[test]
    fn resident_reads_keep_clean_and_device_owned_pages_without_mutation() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let device = NonCpuDeviceId::new(1);
        let write = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, write)],
            coordinator.as_ref(),
        )
        .unwrap();
        let gate = range.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((
                &range,
                DeviceAccessDeclaration::read(device, DeviceVisibilityPoint::new(2)),
            )),
            coordinator.as_ref(),
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        crate::CanonicalBackingRange::publish_device_writes(
            [(&range, write)],
            Arc::clone(&coordinator),
        )
        .unwrap();
        let epoch = gate.epoch();
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((
                &range,
                DeviceAccessDeclaration::read(device, DeviceVisibilityPoint::new(2)),
            )),
            coordinator.as_ref(),
        )
        .unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert_eq!(
            CanonicalBackingRange::prepare_resident_device_accesses(
                std::iter::once((
                    &range,
                    DeviceAccessDeclaration::read(device, DeviceVisibilityPoint::new(1))
                )),
                coordinator.as_ref()
            ),
            Err(VisibilityError::ConflictingAccess)
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Conflicting
        );
    }

    #[test]
    fn resident_device_writes_advance_points_without_closing_cpu_execution() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let device = NonCpuDeviceId::new(1);
        let first = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, first)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, first)], coordinator.clone())
            .unwrap();
        let gate = range.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        let active = gate.acquire_shared();
        let next = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(2),
            DeviceVisibilityPoint::new(4),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((&range, next)),
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, next)], coordinator).unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert!(!gate.transition_pending());
        for page in range.pages() {
            assert_eq!(page.content_generation(), ContentGeneration::INITIAL);
            assert_eq!(
                page.visibility_state(),
                VisibilityState::GpuNewer {
                    device,
                    visible_at: DeviceVisibilityPoint::new(4),
                }
            );
        }
        drop(active);
    }

    #[test]
    fn many_partial_aliases_share_physical_summaries_and_arm_once() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = CanonicalCpuWriteDependency::capture(&range).unwrap();
        assert!(
            range.layout.visibility_epoch.get().is_none(),
            "CPU captures must not register GPU visibility observers"
        );
        let gate = range.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let read =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(1), DeviceVisibilityPoint::new(1));
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((&range, read)),
            coordinator.as_ref(),
        )
        .unwrap();
        let mut aliases = Vec::new();
        for offset in 0..128 {
            let alias = range.snapshot_subrange(offset, 0x100).unwrap();
            let dependency = CanonicalCpuWriteDependency::capture(&alias).unwrap();
            assert!(Arc::ptr_eq(&first.inner.summary, &dependency.inner.summary));
            CanonicalBackingRange::prepare_resident_device_accesses(
                std::iter::once((&alias, read)),
                coordinator.as_ref(),
            )
            .unwrap();
            assert!(Arc::ptr_eq(
                range.visibility_summary(),
                alias.visibility_summary()
            ));
            aliases.push(dependency);
        }
        assert_eq!(
            gate.epoch(),
            epoch,
            "protected aliases need no mutation handshake"
        );
        allocation.write(0x800, &[0x5a]).unwrap();
        assert!(!first.remains_current());
        assert!(aliases.iter().all(|alias| !alias.remains_current()));
    }

    #[test]
    fn repeated_topology_shares_dirty_publication_without_refreshing_old_observations() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = CanonicalCpuWriteDependency::capture(&range).unwrap();
        allocation.write(0x1010, &[0x42]).unwrap();
        let alias = allocation.backing_range(MemoryPermissions::READ).unwrap();
        assert!(!range.shares_layout(&alias));
        let second = CanonicalCpuWriteDependency::capture(&alias).unwrap();
        assert!(Arc::ptr_eq(&first.inner.summary, &second.inner.summary));
        assert!(!first.remains_current());
        assert!(second.remains_current());
        allocation.write(0, &[0x24]).unwrap();
        assert!(!first.remains_current());
        assert!(!second.remains_current());
        first.snapshot_all(&range).unwrap();
        assert!(first.remains_current());
        assert!(!second.remains_current());
    }

    #[test]
    fn partial_owner_publication_does_not_advance_untouched_pages() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first_page = full.snapshot_subrange(0, 0x1000).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let device = NonCpuDeviceId::new(1);
        let first = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&full, first)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&full, first)], coordinator.clone())
            .unwrap();
        let partial = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(2),
            DeviceVisibilityPoint::new(3),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((&first_page, partial)),
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes(
            [(&first_page, partial)],
            coordinator.clone(),
        )
        .unwrap();
        assert_eq!(
            full.segments()[0].visibility_state(),
            VisibilityState::GpuNewer {
                device,
                visible_at: DeviceVisibilityPoint::new(3)
            }
        );
        assert_eq!(
            full.segments()[1].visibility_state(),
            VisibilityState::GpuNewer {
                device,
                visible_at: DeviceVisibilityPoint::new(2)
            }
        );
        let next = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(3),
            DeviceVisibilityPoint::new(4),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((&full, next)),
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&full, next)], coordinator).unwrap();
        assert!(full.pages().all(|page| page.visibility_state()
            == VisibilityState::GpuNewer {
                device,
                visible_at: DeviceVisibilityPoint::new(4)
            }));
    }

    #[test]
    fn gpu_ownership_ends_cpu_streaming_without_an_unchanged_download() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        for value in 1..=8 {
            allocation.pages()[0].prepare_cpu_write().unwrap();
            allocation.write(0, &[value]).unwrap();
            assert!(
                !dependency
                    .snapshot_dirty_pages(&range, 16)
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(dependency.has_streaming_pages());
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let write = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, write)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, write)], coordinator)
            .unwrap();
        assert!(
            dependency
                .snapshot_dirty_pages(&range, 16)
                .unwrap()
                .is_empty()
        );
        assert!(!dependency.has_streaming_pages());
        assert!(dependency.remains_current());
    }

    #[test]
    fn cpu_writeback_retries_a_concurrent_resident_device_write() {
        assert_cpu_writeback_retries_resident_device_write(4);
    }

    #[test]
    fn cpu_writeback_retries_a_resident_write_at_the_same_completion_point() {
        assert_cpu_writeback_retries_resident_device_write(2);
    }

    fn assert_cpu_writeback_retries_resident_device_write(next_point: u64) {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let device = NonCpuDeviceId::new(1);
        let first = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, first)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, first)], coordinator.clone())
            .unwrap();
        let mut requests = Vec::new();
        let snapshot = dependency
            .snapshot_with_resolver(
                &range,
                CpuWriteSnapshotSelection::All,
                1,
                &mut |_, request| {
                    requests.push(request.visible_at);
                    if requests.len() == 1 {
                        let next = DeviceAccessDeclaration::write(
                            device,
                            DeviceVisibilityPoint::new(2),
                            DeviceVisibilityPoint::new(next_point),
                        )
                        .unwrap();
                        crate::CanonicalBackingRange::publish_device_writes(
                            [(&range, next)],
                            coordinator.clone(),
                        )
                        .unwrap();
                        Ok(vec![0x11; request.size].into_boxed_slice())
                    } else {
                        Ok(vec![0x44; request.size].into_boxed_slice())
                    }
                },
            )
            .unwrap();
        assert_eq!(
            requests,
            [
                DeviceVisibilityPoint::new(2),
                DeviceVisibilityPoint::new(next_point)
            ]
        );
        assert_eq!(snapshot[0].1.as_ref(), &[0x44; 0x1000]);
        assert_eq!(
            range.pages().next().unwrap().visibility_state(),
            VisibilityState::Clean
        );
        // Materialization restores CPU authority. The next handoff revokes
        // aliases locally without changing the guest mapping epoch.
        let gate = range.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        let next = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(4),
            DeviceVisibilityPoint::new(6),
        )
        .unwrap();
        CanonicalBackingRange::prepare_resident_device_accesses(
            std::iter::once((&range, next)),
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, next)], coordinator).unwrap();
        assert_eq!(gate.epoch(), epoch);
    }

    #[test]
    fn device_owner_snapshot_resolves_visibility_inline_and_rearms() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(UnexpectedCpuVisibility);
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(1),
            DeviceVisibilityPoint::new(2),
        )
        .unwrap();
        crate::CanonicalBackingRange::prepare_resident_device_accesses(
            [(&range, declaration)],
            coordinator.as_ref(),
        )
        .unwrap();
        crate::CanonicalBackingRange::publish_device_writes([(&range, declaration)], coordinator)
            .unwrap();
        let mut requests = 0;
        let snapshots = dependency
            .snapshot_with_resolver(
                &range,
                CpuWriteSnapshotSelection::All,
                1,
                &mut |_, request| {
                    requests += 1;
                    assert_eq!(request.page, range.segments()[0].page());
                    assert_eq!(request.visible_at, DeviceVisibilityPoint::new(2));
                    Ok(vec![0x5a; request.size].into_boxed_slice())
                },
            )
            .unwrap();
        assert_eq!(requests, 1);
        assert_eq!(snapshots[0].1.as_ref(), &[0x5a; 0x1000]);
        assert!(dependency.remains_current());
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Clean
        );
        allocation.write(0, &[0x33]).unwrap();
        assert!(!dependency.remains_current());
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn physical_union_keeps_holes_and_page_index_is_independent_of_logical_order() {
        let allocation = crate::CanonicalAllocation::zeroed(8192, 4096).unwrap();
        let full = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut segments = Vec::new();
        for (offset, size) in [(4096, 4), (0x100, 8), (0x108, 8), (0x104, 8), (0x200, 8)] {
            full.snapshot_subrange_into(offset, size, &mut segments)
                .unwrap();
        }
        let logical = CanonicalBackingRange::new(segments).unwrap();
        assert_eq!(logical.size(), 36);
        assert_eq!(
            logical
                .pages()
                .map(CanonicalBackingPage::identity)
                .collect::<Vec<_>>(),
            vec![full.segments()[0].page(), full.segments()[1].page()]
        );
        assert_eq!(logical.segments()[0].page(), full.segments()[1].page());
        let physical = logical.physical_union().unwrap();
        assert_eq!(physical.size(), 28);
        assert_eq!(
            physical
                .segments()
                .iter()
                .map(|s| (s.page(), s.offset(), s.size()))
                .collect::<Vec<_>>(),
            vec![
                (full.segments()[0].page(), 0x100, 16),
                (full.segments()[0].page(), 0x200, 8),
                (full.segments()[1].page(), 0, 4)
            ]
        );
    }
}
