//! Checked, retained and pointer-free canonical backing ranges.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
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

#[derive(Debug)]
struct CanonicalRangeLayout {
    segments: Box<[CanonicalBackingSegment]>,
    // Indices retain first-occurrence order without duplicating page authority.
    pages: Box<[usize]>,
    // Stable identity order is also the lock order for multi-store transitions.
    stores: Box<[CanonicalRangeStore]>,
    owner: Mutex<Option<Arc<crate::backing::RangeDeviceOwner>>>,
    visibility_epoch: std::sync::OnceLock<Arc<AtomicU64>>,
    clean_read_epoch: AtomicU64,
    cpu_summary: Mutex<std::sync::Weak<CpuWriteSummary>>,
}

#[derive(Debug)]
struct CanonicalRangeStore {
    store: crate::CanonicalBackingStore,
    changes: Box<[crate::MemoryInvalidationKind]>,
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
                    (self.layout.segments.len() + other.layout.segments.len()) as u64,
                );
            }
        }
        self.size == other.size
            && (Arc::ptr_eq(&self.layout, &other.layout)
                || self.layout.segments == other.layout.segments)
    }
}

impl Eq for CanonicalBackingRange {}

struct CpuWriteDependencyPage {
    page: CanonicalBackingPage,
    observed_epoch: AtomicU64,
}

const CPU_WRITE_GROUP_PAGES: usize = 64;

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

/// Cloneable page-granular observation of CPU writes.
///
/// Capturing establishes a read-only baseline through every direct alias.
/// The first later CPU write advances the physical page's dirty epoch; no
/// subsequent store in that dirty epoch performs observer publication.
#[derive(Clone)]
pub struct CanonicalCpuWriteDependency {
    inner: Arc<CanonicalCpuWriteDependencyInner>,
}

impl CanonicalCpuWriteDependency {
    /// Captures and arms every distinct physical page in one range.
    /// Call without a native epoch, memory lease or cache lock.
    pub fn capture(range: &CanonicalBackingRange) -> Result<Self, CanonicalRangeAccessError> {
        Self::capture_ranges([range])
    }

    /// Captures several ranges as one page-granular dependency domain.
    /// Restrictive protections are established while every affected backing
    /// store and its bound engine are quiescent. No native epoch, memory lease
    /// or cache lock may survive into this operation. Empty input is an error.
    pub fn capture_ranges<'a>(
        ranges: impl IntoIterator<Item = &'a CanonicalBackingRange>,
    ) -> Result<Self, CanonicalRangeAccessError> {
        let domains = ranges.into_iter().cloned().collect::<Box<[_]>>();
        let mut execution_stores = BTreeMap::new();
        let mut pages = BTreeMap::new();
        for range in &domains {
            for segment in range.segments() {
                execution_stores
                    .entry(segment.backing.store().identity())
                    .or_insert_with(|| segment.backing.store().clone());
                pages
                    .entry(segment.page())
                    .or_insert_with(|| segment.backing().clone());
            }
        }
        if pages.is_empty() {
            return Err(CanonicalRangeAccessError::IncompleteRange);
        }
        let execution_stores = execution_stores.into_values().collect::<Vec<_>>();
        // An initially unarmed page needs the engine handshake before waiting
        // for execution leases. Treat this inspection only as a hint: a CPU
        // write can disarm tracking before admission closes, so check again
        // under exclusion and retry with the mutation handshake if necessary.
        let mut arm_tracking = pages.values().try_fold(false, |needed, page| {
            Ok::<_, CanonicalRangeAccessError>(
                needed
                    || page
                        .needs_cpu_dirty_tracking()
                        .map_err(CanonicalRangeAccessError::Backing)?,
            )
        })?;
        let _transitions = loop {
            let mut transitions = execution_stores
                .iter()
                .map(|store| store.execution_gate().acquire_capture(arm_tracking))
                .collect::<Result<Vec<_>, _>>()
                .map_err(CanonicalRangeAccessError::Mutation)?;
            let needs_tracking = pages.values().try_fold(false, |needed, page| {
                Ok::<_, CanonicalRangeAccessError>(
                    needed
                        || page
                            .needs_cpu_dirty_tracking()
                            .map_err(CanonicalRangeAccessError::Backing)?,
                )
            })?;
            if !arm_tracking && needs_tracking {
                drop(transitions);
                arm_tracking = true;
                continue;
            }
            if arm_tracking {
                for transition in &mut transitions {
                    transition.commit();
                }
            }
            break transitions;
        };
        let group_count = pages.len().div_ceil(CPU_WRITE_GROUP_PAGES);
        // Consumers of identical retained topology share one page observer,
        // while their baseline epochs remain independent. Refreshing content
        // never makes an older opaque interpretation current again.
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
            let coverage = PageCoverage::from_sorted(pages.keys().copied());
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
        let pages = pages
            .into_values()
            .enumerate()
            .map(|(index, page)| {
                let observed_epoch = if shared {
                    page.arm_cpu_dirty_observer_quiescent()
                } else {
                    page.observe_cpu_write_summary(&summary, index / CPU_WRITE_GROUP_PAGES)
                }
                .map_err(CanonicalRangeAccessError::Backing)?;
                Ok(CpuWriteDependencyPage {
                    page,
                    observed_epoch: AtomicU64::new(observed_epoch),
                })
            })
            .collect::<Result<Vec<_>, CanonicalRangeAccessError>>()?
            .into_boxed_slice();
        Ok(Self {
            inner: Arc::new(CanonicalCpuWriteDependencyInner {
                domains,
                adaptive: Mutex::new(AdaptiveCpuTracking {
                    streaks: vec![0; pages.len()].into_boxed_slice(),
                    streaming: BTreeMap::new(),
                }),
                pages,
                observed_summary: AtomicU64::new(summary.epoch.load(Ordering::Acquire)),
                observed_groups: summary
                    .groups
                    .iter()
                    .map(|epoch| AtomicU64::new(epoch.load(Ordering::Acquire)))
                    .collect(),
                summary,
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

    /// Establishes a protected baseline after the consumer incorporates or
    /// completely overwrites these bytes. Call outside native execution and
    /// without a memory lease: protection changes rendezvous with the engine.
    pub fn rearm(&self) -> Result<(), CanonicalRangeAccessError> {
        let mut stores = BTreeMap::new();
        for page in &self.inner.pages {
            stores
                .entry(page.page.store().identity())
                .or_insert_with(|| page.page.store().clone());
        }
        let stores = stores.into_values().collect::<Vec<_>>();
        let mut transitions = stores
            .iter()
            .map(|store| store.execution_gate().acquire_mutation(&[]))
            .collect::<Result<Vec<_>, _>>()
            .map_err(CanonicalRangeAccessError::Mutation)?;
        let mut adaptive = self
            .inner
            .adaptive
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for page in &self.inner.pages {
            self.arm_page(page)?;
        }
        adaptive.streaks.fill(0);
        adaptive.streaming.clear();
        self.inner.streaming_pages.store(0, Ordering::Release);
        self.record_summary();
        for transition in &mut transitions {
            transition.commit();
        }
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
        if alignment == 0 {
            return Err(CanonicalRangeAccessError::InvalidAlignment(alignment));
        }
        crate::metrics::record(
            crate::metrics::Counter::SnapshotRequestedBytes,
            range.size(),
        );
        let known_domain = self
            .inner
            .domains
            .iter()
            .any(|domain| domain.shares_layout(range));
        if !known_domain {
            let dependency_pages = self
                .inner
                .pages
                .iter()
                .map(|page| page.page.identity())
                .collect::<BTreeSet<_>>();
            if !range
                .pages()
                .all(|page| dependency_pages.contains(&page.identity()))
            {
                return Err(CanonicalRangeAccessError::DependencyMismatch);
            }
        }
        if selection != CpuWriteSnapshotSelection::All && self.remains_current() {
            return Ok(Vec::new());
        }
        let range_pages = range
            .pages()
            .map(CanonicalBackingPage::identity)
            .collect::<BTreeSet<_>>();
        loop {
            let mut transitions = range
                .execution_stores()
                .map(|store| store.execution_gate().acquire_mutation(&[]))
                .collect::<Result<Vec<_>, _>>()
                .map_err(CanonicalRangeAccessError::Mutation)?;
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
                        range_pages.contains(&page.page.identity())
                            && page.page.cpu_dirty_epoch()
                                == page.observed_epoch.load(Ordering::Acquire)
                            && matches!(
                                page.page.visibility_state(),
                                VisibilityState::GpuNewer { .. }
                            )
                    })
                    .collect::<Vec<_>>();
                for index in device_owned {
                    // Device ownership already revokes CPU writes. Restore a
                    // protected observation without downloading unchanged GPU
                    // bytes merely to compare a former CPU streaming shadow.
                    self.arm_page(&self.inner.pages[index])?;
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
                        if page.page.cpu_dirty_epoch()
                            != page.observed_epoch.load(Ordering::Acquire)
                        {
                            candidates.insert(group * CPU_WRITE_GROUP_PAGES + offset);
                        }
                    }
                }
            }
            candidates
                .retain(|index| range_pages.contains(&self.inner.pages[*index].page.identity()));
            let mut needed = if selection == CpuWriteSnapshotSelection::All {
                self.inner
                    .pages
                    .iter()
                    .enumerate()
                    .filter_map(|(index, page)| {
                        range_pages.contains(&page.page.identity()).then_some(index)
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
                        && self.inner.pages[*index]
                            .page
                            .cpu_visible_quiescent()
                            .map_err(CanonicalRangeAccessError::Backing)?,
                )
            })?;
            if !visible {
                drop(adaptive);
                drop(transitions);
                self.resolve_pages(&needed, resolve)?;
                continue;
            }
            let mut dirty = candidates.clone();
            let streaming = adaptive
                .streaming
                .keys()
                .copied()
                .filter(|index| range_pages.contains(&self.inner.pages[*index].page.identity()))
                .collect::<Vec<_>>();
            for index in streaming {
                let page = &self.inner.pages[index];
                crate::metrics::record(
                    crate::metrics::Counter::StreamingComparedBytes,
                    page.page.size() as u64,
                );
                let mut bytes = vec![0; page.page.size()].into_boxed_slice();
                page.page
                    .read_quiescent(0, &mut bytes)
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
                        self.arm_page(page)?;
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
            {
                intervals.push((0, range.size()));
            } else if selection == CpuWriteSnapshotSelection::DirtyPages {
                let mut offset = 0;
                for segment in range.segments() {
                    let end = offset + segment.size();
                    if dirty_pages.contains(&segment.page()) {
                        let start = offset / alignment * alignment;
                        let end = end
                            .checked_add(alignment - 1)
                            .ok_or(CanonicalRangeAccessError::RangeOverflow)?
                            / alignment
                            * alignment;
                        intervals.push((start, end.min(range.size())));
                    }
                    offset += segment.size();
                }
                normalize_intervals(&mut intervals);
            }
            // Alignment or WholeIfDirty can include additional pages. Resolve
            // only pages actually copied, never unrelated GPU-owned bindings.
            let mut offset = 0;
            for segment in range.segments() {
                let end = offset + segment.size();
                if intervals
                    .iter()
                    .any(|(start, stop)| *start < end && offset < *stop)
                {
                    let index = self
                        .inner
                        .pages
                        .binary_search_by_key(&segment.page(), |page| page.page.identity())
                        .expect("validated dependency page");
                    needed.insert(index);
                }
                offset = end;
            }
            let visible = needed.iter().try_fold(true, |visible, index| {
                Ok::<_, CanonicalRangeAccessError>(
                    visible
                        && self.inner.pages[*index]
                            .page
                            .cpu_visible_quiescent()
                            .map_err(CanonicalRangeAccessError::Backing)?,
                )
            })?;
            if !visible {
                drop(adaptive);
                drop(transitions);
                self.resolve_pages(&needed, resolve)?;
                continue;
            }
            let mut snapshots = Vec::new();
            for (offset, end) in intervals {
                let mut bytes = vec![
                    0;
                    usize::try_from(end - offset)
                        .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?
                ];
                crate::metrics::record(
                    crate::metrics::Counter::SnapshotCopiedBytes,
                    bytes.len() as u64,
                );
                range.read_quiescent(offset, &mut bytes)?;
                snapshots.push((offset, bytes.into_boxed_slice()));
            }
            for index in &dirty {
                if adaptive.streaming.contains_key(index) {
                    continue;
                }
                let streak = adaptive.streaks[*index].saturating_add(1);
                adaptive.streaks[*index] = streak;
                let page = &self.inner.pages[*index];
                if streak >= STREAMING_DIRTY_THRESHOLD {
                    let mut bytes = vec![0; page.page.size()].into_boxed_slice();
                    page.page
                        .read_quiescent(0, &mut bytes)
                        .map_err(CanonicalRangeAccessError::Backing)?;
                    page.observed_epoch
                        .store(page.page.cpu_dirty_epoch(), Ordering::Release);
                    adaptive
                        .streaming
                        .insert(*index, StreamingCpuPage { bytes, quiet: 0 });
                } else {
                    self.arm_page(page)?;
                }
            }
            if selection == CpuWriteSnapshotSelection::All {
                for (index, page) in self.inner.pages.iter().enumerate() {
                    if range_pages.contains(&page.page.identity())
                        && !adaptive.streaming.contains_key(&index)
                    {
                        self.arm_page(page)?;
                    }
                }
            }
            self.inner
                .streaming_pages
                .store(adaptive.streaming.len() as u64, Ordering::Release);
            self.record_summary();
            for transition in &mut transitions {
                transition.commit();
            }
            return Ok(snapshots);
        }
    }

    fn resolve_pages(
        &self,
        pages: &BTreeSet<usize>,
        resolve: &mut crate::CpuVisibilityResolver<'_>,
    ) -> Result<(), CanonicalRangeAccessError> {
        for index in pages {
            self.inner.pages[*index]
                .page
                .ensure_cpu_visible_with(resolve)
                .map_err(|error| {
                    CanonicalRangeAccessError::Backing(CanonicalPageError::Visibility(error))
                })?;
        }
        Ok(())
    }

    fn arm_page(&self, page: &CpuWriteDependencyPage) -> Result<(), CanonicalRangeAccessError> {
        let epoch = page
            .page
            .arm_cpu_dirty_observer_quiescent()
            .map_err(CanonicalRangeAccessError::Backing)?;
        page.observed_epoch.store(epoch, Ordering::Release);
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
    /// Retained physical identity is the authority for device ranges, even
    /// after their original virtual mapping has changed. Register all pages
    /// before closing the gate: a compiler can publish between discovery and
    /// closure, and its newly captured page must be included too. Reads also
    /// change visibility authority and can leave a page Invalid on failure;
    /// no code derived from that page may survive the transition in that case.
    fn begin_device_transition(
        &self,
    ) -> Result<Vec<crate::ExecutionMutationGuard<'_>>, VisibilityError> {
        self.layout
            .stores
            .iter()
            .map(|entry| {
                entry
                    .store
                    .execution_gate()
                    .acquire_mutation(&entry.changes)
                    .map_err(VisibilityError::ExecutionMutation)
            })
            .collect()
    }

    fn execution_stores(&self) -> impl Iterator<Item = &crate::CanonicalBackingStore> {
        self.layout.stores.iter().map(|entry| &entry.store)
    }

    fn pages(&self) -> impl Iterator<Item = &CanonicalBackingPage> {
        self.layout
            .pages
            .iter()
            .map(|index| self.layout.segments[*index].backing())
    }

    /// Creates a non-empty range and checks its total length.
    pub fn new(segments: Vec<CanonicalBackingSegment>) -> Result<Self, CanonicalRangeError> {
        if segments.is_empty() {
            return Err(CanonicalRangeError::Empty);
        }
        let mut size = 0_u64;
        for segment in &segments {
            size = size
                .checked_add(segment.size)
                .ok_or(CanonicalRangeError::RangeOverflow)?;
        }
        let mut seen_pages = BTreeSet::new();
        let mut pages = Vec::new();
        let mut stores = BTreeMap::new();
        for (index, segment) in segments.iter().enumerate() {
            if seen_pages.insert(segment.page()) {
                pages.push(index);
                let (_, changes) = stores
                    .entry(segment.page().store())
                    .or_insert_with(|| (segment.backing.store().clone(), BTreeSet::new()));
                changes.insert(segment.page().page());
            }
        }
        let stores = stores
            .into_values()
            .map(|(store, pages)| CanonicalRangeStore {
                store,
                changes: pages
                    .into_iter()
                    .map(|first| crate::MemoryInvalidationKind::ExecutableContent {
                        first,
                        second: None,
                    })
                    .collect(),
            })
            .collect();
        Ok(Self {
            layout: Arc::new(CanonicalRangeLayout {
                segments: segments.into(),
                pages: pages.into(),
                stores,
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
        &self.layout.segments
    }

    /// Retains a checked logical subrange with the same canonical page identity.
    pub fn snapshot_subrange(&self, offset: u64, size: u64) -> Result<Self, CanonicalRangeError> {
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
        let original_len = output.len();
        let mut logical_start = 0_u64;
        let result = (|| {
            for segment in self.layout.segments.iter() {
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
                logical_start = logical_end;
                if logical_start >= end {
                    break;
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
        loop {
            let mut logical_start = 0_u64;
            let mut visited = BTreeSet::new();
            for segment in self.layout.segments.iter() {
                let logical_end = logical_start
                    .checked_add(segment.size)
                    .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                if offset.max(logical_start) < end.min(logical_end)
                    && visited.insert(segment.page())
                {
                    segment
                        .backing
                        .prepare_cpu_access()
                        .map_err(CanonicalRangeAccessError::Backing)?;
                }
                logical_start = logical_end;
                if logical_start >= end {
                    break;
                }
            }

            let _transitions = self
                .execution_stores()
                .map(|store| store.execution_gate().acquire_exclusive())
                .collect::<Vec<_>>();
            let mut logical_start = 0_u64;
            let mut cpu_visible = true;
            for segment in self.layout.segments.iter() {
                let logical_end = logical_start
                    .checked_add(segment.size)
                    .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                if offset.max(logical_start) < end.min(logical_end)
                    && !segment
                        .backing
                        .cpu_visible_quiescent()
                        .map_err(CanonicalRangeAccessError::Backing)?
                {
                    cpu_visible = false;
                    break;
                }
                logical_start = logical_end;
                if logical_start >= end {
                    break;
                }
            }
            if !cpu_visible {
                continue;
            }

            self.read_quiescent(offset, output)?;
            return Ok(());
        }
    }

    fn read_quiescent(
        &self,
        offset: u64,
        output: &mut [u8],
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
        let mut logical_start = 0_u64;
        let mut copied = 0_usize;
        for segment in self.layout.segments.iter() {
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
                segment
                    .backing
                    .read_quiescent(page_offset, &mut output[copied..copied_end])
                    .map_err(CanonicalRangeAccessError::Backing)?;
                copied = copied_end;
            }
            logical_start = logical_end;
            if logical_start >= end {
                break;
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

    /// Establishes device visibility before executing a declared access.
    ///
    /// The initial implementation transitions complete canonical pages even
    /// when the logical range contains only a page fragment. This deliberately
    /// conservative granularity preserves overlapping-alias correctness.
    pub fn prepare_device_access(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        let mut transitions = self.begin_device_transition()?;
        for transition in &mut transitions {
            transition.commit();
        }
        for page in self.pages() {
            page.prepare_device_access(declaration, Arc::clone(&coordinator))?;
        }
        Ok(())
    }

    /// Establishes device visibility only where canonical CPU bytes are newer.
    ///
    /// A resident resource already owns a device representation for clean
    /// pages. Avoiding another whole-page handoff for those pages keeps the
    /// steady-state path proportional to actual CPU writes. Newly created
    /// resources remain responsible for their initial upload.
    pub fn prepare_resident_device_access(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
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
            return Ok(());
        }
        let visibility_summary = (!declaration.kind().writes()).then(|| self.visibility_summary());
        let visibility_epoch =
            visibility_summary.map_or(u64::MAX, |summary| summary.load(Ordering::Acquire));
        if !declaration.kind().writes()
            && visibility_epoch != u64::MAX
            && self.layout.clean_read_epoch.load(Ordering::Acquire) == visibility_epoch
        {
            return Ok(());
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
            return Ok(());
        }
        let mut transitions = self.begin_device_transition()?;
        for transition in &mut transitions {
            transition.commit();
        }
        for page in self.pages() {
            match page.visibility_state() {
                VisibilityState::Clean if !declaration.kind().writes() => {}
                VisibilityState::Clean => {
                    page.prepare_device_access(declaration, Arc::clone(&coordinator))?
                }
                VisibilityState::CpuNewer if !declaration.kind().writes() => {
                    page.prepare_resident_device_read(declaration)?
                }
                VisibilityState::CpuNewer => {
                    page.prepare_device_access(declaration, Arc::clone(&coordinator))?
                }
                VisibilityState::GpuNewer { device, visible_at }
                    if device == declaration.device()
                        && visible_at <= declaration.device_visible_at() => {}
                VisibilityState::GpuNewer { .. }
                | VisibilityState::Conflicting
                | VisibilityState::Invalid => {
                    page.prepare_device_access(declaration, Arc::clone(&coordinator))?
                }
            }
        }
        Ok(())
    }

    /// Publishes device ownership with its required completion point.
    ///
    /// The point may still be in flight. CPU consumers route through the
    /// visibility coordinator, which waits for that point before materializing
    /// bytes. This does not signal a guest fence or claim host completion.
    pub fn publish_device_write(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        if !declaration.kind().writes() {
            return Err(VisibilityError::DeclarationDoesNotWrite);
        }
        crate::metrics::record(crate::metrics::Counter::OwnershipPublications, 1);
        crate::metrics::record(crate::metrics::Counter::TrackingLocks, 1);
        {
            let cached = self
                .layout
                .owner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cached
                .as_ref()
                .is_some_and(|owner| owner.advance(declaration))
            {
                return Ok(());
            }
        }
        // Never retain an ownership lock across an execution handshake.
        let owner = Arc::new(crate::backing::RangeDeviceOwner::new(
            declaration.device(),
            declaration
                .cpu_visible_at()
                .ok_or(VisibilityError::DeclarationDoesNotWrite)?,
            coordinator,
        ));
        let mut pages = self.pages();
        while let Some(page) = pages.next() {
            crate::metrics::record(crate::metrics::Counter::PageOwnershipUpdates, 1);
            if page.attach_resident_device_owner(declaration, &owner)? {
                continue;
            }
            let mut transitions = self.begin_device_transition()?;
            for transition in &mut transitions {
                transition.commit();
            }
            page.publish_device_write(declaration, Arc::clone(&owner))?;
            for page in pages {
                crate::metrics::record(crate::metrics::Counter::PageOwnershipUpdates, 1);
                page.publish_device_write(declaration, Arc::clone(&owner))?;
            }
            *self
                .layout
                .owner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(owner);
            return Ok(());
        }
        *self
            .layout
            .owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(owner);
        Ok(())
    }

    /// Marks every retained page invalid after an unrecoverable residency or
    /// visibility failure.
    pub fn invalidate_visibility(&self) -> Result<(), VisibilityError> {
        let mut transitions = self.begin_device_transition()?;
        for transition in &mut transitions {
            transition.commit();
        }
        for page in self.pages() {
            page.invalidate_visibility()?;
        }
        Ok(())
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
    Mutation(crate::ExecutionMutationError),
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
            Self::Mutation(error) => error.fmt(formatter),
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
    use crate::{
        CanonicalAllocation, CanonicalBackingStore, CanonicalWriteBatch, ContentGeneration,
        CpuVisibilityRequest, DeviceVisibilityPoint, DeviceVisibilityRequest, GuestPhysicalPageId,
        NonCpuDeviceId, VisibilityCoordinatorError,
    };

    struct UnexpectedCpuVisibility;

    impl VisibilityCoordinator for UnexpectedCpuVisibility {
        fn make_device_visible(
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
            [b.identity(), a.identity(), c.identity()]
        );
        assert_eq!(
            range
                .execution_stores()
                .map(CanonicalBackingStore::identity)
                .collect::<Vec<_>>(),
            [first.identity(), second.identity()]
        );
        assert_eq!(
            &*range.layout.stores[0].changes,
            &[
                crate::MemoryInvalidationKind::ExecutableContent {
                    first: GuestPhysicalPageId::new(1),
                    second: None
                },
                crate::MemoryInvalidationKind::ExecutableContent {
                    first: GuestPhysicalPageId::new(3),
                    second: None
                },
            ]
        );
        let subrange = range.snapshot_subrange(0x100, 0x100).unwrap();
        assert_eq!(
            subrange
                .pages()
                .map(CanonicalBackingPage::identity)
                .collect::<Vec<_>>(),
            [a.identity()]
        );
        assert_eq!(subrange.execution_stores().count(), 1);
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
        range.invalidate_visibility().unwrap();
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
        range.invalidate_visibility().unwrap();
        assert_eq!(
            CanonicalCpuWriteDependency::capture(&range).unwrap_err(),
            CanonicalRangeAccessError::Backing(CanonicalPageError::Visibility(
                VisibilityError::InvalidState
            ))
        );
    }

    #[test]
    fn multi_store_capture_rejection_releases_prior_holds_before_arming_any_page() {
        use crate::{ExecutionMutation, ExecutionMutationError, ExecutionMutationObserver};
        struct Hold(Arc<AtomicBool>);
        impl ExecutionMutation for Hold {}
        impl Drop for Hold {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        struct Owner {
            active: Arc<AtomicBool>,
            reject: bool,
        }
        impl ExecutionMutationObserver for Owner {
            fn begin(
                self: Arc<Self>,
                changes: &[crate::MemoryInvalidationKind],
            ) -> Result<Box<dyn ExecutionMutation>, ExecutionMutationError> {
                assert!(changes.is_empty());
                if self.reject {
                    assert!(self.active.load(Ordering::Acquire));
                    return Err(ExecutionMutationError(
                        "second tracking owner rejected".into(),
                    ));
                }
                self.active.store(true, Ordering::Release);
                Ok(Box::new(Hold(self.active.clone())))
            }
        }
        let stores = [
            CanonicalBackingStore::allocate().unwrap(),
            CanonicalBackingStore::allocate().unwrap(),
        ];
        let active = Arc::new(AtomicBool::new(false));
        let mut pages = Vec::new();
        let mut ranges = Vec::new();
        for (index, store) in stores.iter().enumerate() {
            store
                .execution_gate()
                .set_mutation_observer(Arc::new(Owner {
                    active: active.clone(),
                    reject: index == 1,
                }))
                .unwrap();
            let page = CanonicalBackingPage::zeroed(
                store,
                GuestPhysicalPageId::new(index as u64),
                4096,
                ContentGeneration::INITIAL,
            )
            .unwrap();
            ranges.push(
                CanonicalBackingRange::new(vec![
                    CanonicalBackingSegment::new(
                        page.clone(),
                        0,
                        4096,
                        MemoryPermissions::READ_WRITE,
                        MappingGeneration::new(1),
                    )
                    .unwrap(),
                ])
                .unwrap(),
            );
            pages.push(page);
        }
        // Reverse input order; acquisition still follows stable store identity.
        assert_eq!(
            CanonicalCpuWriteDependency::capture_ranges([&ranges[1], &ranges[0]]).unwrap_err(),
            CanonicalRangeAccessError::Mutation(ExecutionMutationError(
                "second tracking owner rejected".into()
            ))
        );
        assert!(!active.load(Ordering::Acquire));
        for (store, page) in stores.iter().zip(&pages) {
            assert!(!store.execution_gate().transition_pending());
            assert_eq!(store.execution_gate().epoch(), 1);
            let epoch = page.cpu_dirty_epoch();
            page.prepare_cpu_write().unwrap();
            assert_eq!(page.cpu_dirty_epoch(), epoch); // No observer was armed.
        }
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
        range
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        range
            .publish_device_write(declaration, coordinator)
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
        range
            .prepare_device_access(write, Arc::clone(&coordinator))
            .unwrap();
        let gate = range.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        range
            .prepare_resident_device_access(
                DeviceAccessDeclaration::read(device, DeviceVisibilityPoint::new(2)),
                Arc::clone(&coordinator),
            )
            .unwrap();
        assert_eq!(gate.epoch(), epoch);
        range
            .publish_device_write(write, Arc::clone(&coordinator))
            .unwrap();
        let epoch = gate.epoch();
        range
            .prepare_resident_device_access(
                DeviceAccessDeclaration::read(device, DeviceVisibilityPoint::new(2)),
                Arc::clone(&coordinator),
            )
            .unwrap();
        assert_eq!(gate.epoch(), epoch);
        assert_eq!(
            range.prepare_resident_device_access(
                DeviceAccessDeclaration::read(device, DeviceVisibilityPoint::new(1)),
                coordinator
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
        range
            .prepare_device_access(first, coordinator.clone())
            .unwrap();
        range
            .publish_device_write(first, coordinator.clone())
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
        range
            .prepare_resident_device_access(next, coordinator.clone())
            .unwrap();
        range.publish_device_write(next, coordinator).unwrap();
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
        range
            .prepare_resident_device_access(read, coordinator.clone())
            .unwrap();
        let mut aliases = Vec::new();
        for offset in 0..128 {
            let alias = range.snapshot_subrange(offset, 0x100).unwrap();
            let dependency = CanonicalCpuWriteDependency::capture(&alias).unwrap();
            assert!(Arc::ptr_eq(&first.inner.summary, &dependency.inner.summary));
            alias
                .prepare_resident_device_access(read, coordinator.clone())
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
        full.prepare_device_access(first, coordinator.clone())
            .unwrap();
        full.publish_device_write(first, coordinator.clone())
            .unwrap();
        let partial = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(2),
            DeviceVisibilityPoint::new(3),
        )
        .unwrap();
        first_page
            .prepare_resident_device_access(partial, coordinator.clone())
            .unwrap();
        first_page
            .publish_device_write(partial, coordinator.clone())
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
        full.prepare_resident_device_access(next, coordinator.clone())
            .unwrap();
        full.publish_device_write(next, coordinator).unwrap();
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
        range
            .prepare_device_access(write, coordinator.clone())
            .unwrap();
        range.publish_device_write(write, coordinator).unwrap();
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
        range
            .prepare_device_access(first, coordinator.clone())
            .unwrap();
        range
            .publish_device_write(first, coordinator.clone())
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
                        range
                            .publish_device_write(next, coordinator.clone())
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
        // Materialization restores CPU authority. The next actual handoff
        // must close execution again, rather than using the resident update.
        let gate = range.segments()[0].backing().store().execution_gate();
        let epoch = gate.epoch();
        let next = DeviceAccessDeclaration::write(
            device,
            DeviceVisibilityPoint::new(4),
            DeviceVisibilityPoint::new(6),
        )
        .unwrap();
        range
            .prepare_resident_device_access(next, coordinator.clone())
            .unwrap();
        range.publish_device_write(next, coordinator).unwrap();
        assert!(gate.epoch() > epoch);
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
        range
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        range
            .publish_device_write(declaration, coordinator)
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
