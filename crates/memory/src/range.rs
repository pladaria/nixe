//! Checked, retained and pointer-free canonical backing ranges.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBackingRange {
    segments: Arc<[CanonicalBackingSegment]>,
    size: u64,
}

struct CpuWriteDependencyPage {
    page: CanonicalBackingPage,
    observed_epoch: AtomicU64,
}

struct CanonicalCpuWriteDependencyInner {
    pages: Box<[CpuWriteDependencyPage]>,
    consecutive_dirty: AtomicU8,
    volatile: AtomicBool,
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

impl CanonicalCpuWriteDependency {
    /// Captures and arms every distinct physical page in one range.
    #[must_use]
    pub fn capture(range: &CanonicalBackingRange) -> Option<Self> {
        Self::capture_ranges([range])
    }

    /// Captures several ranges as one page-granular dependency domain.
    /// Restrictive protections are established while every affected backing
    /// store is quiescent under its exclusive execution transition.
    #[must_use]
    pub fn capture_ranges<'a>(
        ranges: impl IntoIterator<Item = &'a CanonicalBackingRange>,
    ) -> Option<Self> {
        let mut execution_stores = BTreeMap::new();
        let mut pages = BTreeMap::new();
        for range in ranges {
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
            return None;
        }
        let execution_stores = execution_stores.into_values().collect::<Vec<_>>();
        let mut transitions = execution_stores
            .iter()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();
        for transition in &mut transitions {
            transition.commit();
        }
        let pages = pages
            .into_values()
            .map(|page| {
                let observed_epoch = page.arm_cpu_dirty_observer_quiescent().ok()?;
                Some(CpuWriteDependencyPage {
                    page,
                    observed_epoch: AtomicU64::new(observed_epoch),
                })
            })
            .collect::<Option<Vec<_>>>()?
            .into_boxed_slice();
        Some(Self {
            inner: Arc::new(CanonicalCpuWriteDependencyInner {
                pages,
                consecutive_dirty: AtomicU8::new(0),
                volatile: AtomicBool::new(false),
            }),
        })
    }

    /// Returns whether every captured physical page remains in its armed epoch.
    #[must_use]
    pub fn remains_current(&self) -> bool {
        if self.inner.volatile.load(Ordering::Acquire) {
            return false;
        }
        let current =
            self.inner.pages.iter().all(|page| {
                page.page.cpu_dirty_epoch() == page.observed_epoch.load(Ordering::Acquire)
            });
        if current {
            self.inner.consecutive_dirty.store(0, Ordering::Release);
        }
        current
    }

    /// Rearms the captured pages after their consumer has incorporated the
    /// latest bytes. Five consecutive dirty/rearm cycles make this dependency
    /// permanently conservative so frequently written pages stop faulting on
    /// its behalf.
    #[must_use]
    pub fn rearm(&self) -> bool {
        if self.inner.volatile.load(Ordering::Acquire) {
            return false;
        }
        let mut stores = BTreeMap::new();
        for page in &self.inner.pages {
            stores
                .entry(page.page.store().identity())
                .or_insert_with(|| page.page.store().clone());
        }
        let stores = stores.into_values().collect::<Vec<_>>();
        let mut transitions = stores
            .iter()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();
        for transition in &mut transitions {
            transition.commit();
        }
        let dirty =
            self.inner.pages.iter().any(|page| {
                page.page.cpu_dirty_epoch() != page.observed_epoch.load(Ordering::Acquire)
            });
        self.rearm_quiescent(dirty).unwrap_or(false)
    }

    /// Copies every dirty page intersection and establishes the next clean
    /// baseline before direct CPU execution resumes.
    ///
    /// Returned offsets are logical offsets within `range`. Ranges are
    /// expanded to `alignment` where possible so device backends can satisfy
    /// copy constraints without taking a second, racy snapshot.
    pub fn snapshot_dirty_pages(
        &self,
        range: &CanonicalBackingRange,
        alignment: u64,
    ) -> Result<Vec<(u64, Box<[u8]>)>, CanonicalRangeAccessError> {
        self.snapshot_bytes(range, CpuWriteSnapshotSelection::DirtyPages, alignment)
    }

    /// Copies the complete range only when at least one represented page is
    /// dirty, then establishes the next clean baseline atomically with that
    /// snapshot.
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
    ) -> Result<Vec<(u64, Box<[u8]>)>, CanonicalRangeAccessError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CanonicalRangeAccessError::InvalidAlignment(alignment));
        }
        let dependency_pages = self
            .inner
            .pages
            .iter()
            .map(|page| page.page.identity())
            .collect::<BTreeSet<_>>();
        let range_pages = range
            .segments()
            .iter()
            .map(CanonicalBackingSegment::page)
            .collect::<BTreeSet<_>>();
        if dependency_pages != range_pages {
            return Err(CanonicalRangeAccessError::DependencyMismatch);
        }

        loop {
            let stores = range.execution_stores();
            let transitions = stores
                .iter()
                .map(|store| store.execution_gate().acquire_exclusive())
                .collect::<Vec<_>>();
            let volatile = self.inner.volatile.load(Ordering::Acquire);
            let dirty_pages = self
                .inner
                .pages
                .iter()
                .filter_map(|page| {
                    (volatile
                        || page.page.cpu_dirty_epoch()
                            != page.observed_epoch.load(Ordering::Acquire))
                    .then_some(page.page.identity())
                })
                .collect::<BTreeSet<_>>();
            let dirty = !dirty_pages.is_empty();
            if !dirty && selection != CpuWriteSnapshotSelection::All {
                self.inner.consecutive_dirty.store(0, Ordering::Release);
                return Ok(Vec::new());
            }

            let cpu_visible = self.inner.pages.iter().try_fold(true, |visible, page| {
                Ok(visible
                    && page
                        .page
                        .cpu_visible_quiescent()
                        .map_err(CanonicalRangeAccessError::Backing)?)
            })?;
            if !cpu_visible {
                drop(transitions);
                for page in &self.inner.pages {
                    page.page
                        .prepare_cpu_access()
                        .map_err(CanonicalRangeAccessError::Backing)?;
                }
                continue;
            }

            let mut intervals = Vec::new();
            if selection != CpuWriteSnapshotSelection::DirtyPages || volatile {
                intervals.push((0, range.size()));
            } else {
                let mut logical_start = 0_u64;
                for segment in range.segments() {
                    let logical_end = logical_start
                        .checked_add(segment.size())
                        .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                    if dirty_pages.contains(&segment.page()) {
                        let aligned_start = logical_start / alignment * alignment;
                        let aligned_end = logical_end
                            .checked_add(alignment - 1)
                            .ok_or(CanonicalRangeAccessError::RangeOverflow)?
                            / alignment
                            * alignment;
                        intervals.push((aligned_start, aligned_end.min(range.size())));
                    }
                    logical_start = logical_end;
                }
                normalize_intervals(&mut intervals);
            }

            let mut snapshots = Vec::new();
            snapshots
                .try_reserve_exact(intervals.len())
                .map_err(|_| CanonicalRangeAccessError::ResourceExhausted)?;
            for (offset, end) in intervals {
                let size = usize::try_from(end - offset)
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(size)
                    .map_err(|_| CanonicalRangeAccessError::ResourceExhausted)?;
                bytes.resize(size, 0);
                range.read_quiescent(offset, &mut bytes)?;
                snapshots.push((offset, bytes.into_boxed_slice()));
            }

            if !volatile {
                self.rearm_quiescent(dirty)?;
            }
            return Ok(snapshots);
        }
    }

    fn rearm_quiescent(&self, dirty: bool) -> Result<bool, CanonicalRangeAccessError> {
        let consecutive = if dirty {
            self.inner
                .consecutive_dirty
                .load(Ordering::Acquire)
                .saturating_add(1)
        } else {
            0
        };
        self.inner
            .consecutive_dirty
            .store(consecutive, Ordering::Release);
        if consecutive >= 5 {
            self.inner.volatile.store(true, Ordering::Release);
            return Ok(false);
        }
        for page in &self.inner.pages {
            let epoch = page
                .page
                .arm_cpu_dirty_observer_quiescent()
                .map_err(CanonicalRangeAccessError::Backing)?;
            page.observed_epoch.store(epoch, Ordering::Release);
        }
        Ok(true)
    }

    /// Reports whether this dependency has stopped arming dirty faults.
    #[must_use]
    pub fn is_volatile(&self) -> bool {
        self.inner.volatile.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CpuWriteSnapshotSelection {
    DirtyPages,
    WholeIfDirty,
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
                "consecutive_dirty",
                &self.inner.consecutive_dirty.load(Ordering::Acquire),
            )
            .field("volatile", &self.inner.volatile.load(Ordering::Acquire))
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
    fn execution_stores(&self) -> Vec<crate::CanonicalBackingStore> {
        let mut stores = BTreeMap::new();
        for segment in self.segments.iter() {
            stores
                .entry(segment.backing.store().identity())
                .or_insert_with(|| segment.backing.store().clone());
        }
        stores.into_values().collect()
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
        Ok(Self {
            segments: segments.into(),
            size,
        })
    }

    /// Returns the logical byte length.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the ordered canonical segments.
    #[must_use]
    pub fn segments(&self) -> &[CanonicalBackingSegment] {
        &self.segments
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
            for segment in self.segments.iter() {
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
        let stores = self.execution_stores();
        loop {
            let mut logical_start = 0_u64;
            let mut visited = BTreeSet::new();
            for segment in self.segments.iter() {
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

            let _transitions = stores
                .iter()
                .map(|store| store.execution_gate().acquire_exclusive())
                .collect::<Vec<_>>();
            let mut logical_start = 0_u64;
            let mut cpu_visible = true;
            for segment in self.segments.iter() {
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
        for segment in self.segments.iter() {
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
        let stores = self.execution_stores();
        let mut transitions = stores
            .iter()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();
        for transition in &mut transitions {
            transition.commit();
        }
        let mut visited = BTreeSet::new();
        for segment in self.segments.iter() {
            if visited.insert(segment.page()) {
                segment
                    .backing
                    .prepare_device_access(declaration, Arc::clone(&coordinator))?;
            }
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
        let stores = self.execution_stores();
        let mut transitions = stores
            .iter()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();
        for transition in &mut transitions {
            transition.commit();
        }
        let mut visited = BTreeSet::new();
        for segment in self.segments.iter() {
            if !visited.insert(segment.page()) {
                continue;
            }
            match segment.backing.visibility_state() {
                VisibilityState::Clean if !declaration.kind().writes() => {}
                VisibilityState::Clean => segment
                    .backing
                    .prepare_device_access(declaration, Arc::clone(&coordinator))?,
                VisibilityState::CpuNewer if !declaration.kind().writes() => {
                    segment.backing.prepare_resident_device_read(declaration)?
                }
                VisibilityState::CpuNewer => segment
                    .backing
                    .prepare_device_access(declaration, Arc::clone(&coordinator))?,
                VisibilityState::GpuNewer { device, visible_at }
                    if device == declaration.device()
                        && visible_at <= declaration.device_visible_at() => {}
                VisibilityState::GpuNewer { .. }
                | VisibilityState::Conflicting
                | VisibilityState::Invalid => segment
                    .backing
                    .prepare_device_access(declaration, Arc::clone(&coordinator))?,
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
        let stores = self.execution_stores();
        let mut transitions = stores
            .iter()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();
        for transition in &mut transitions {
            transition.commit();
        }
        let mut visited = BTreeSet::new();
        for segment in self.segments.iter() {
            if visited.insert(segment.page()) {
                segment
                    .backing
                    .publish_device_write(declaration, Arc::clone(&coordinator))?;
            }
        }
        Ok(())
    }

    /// Marks every retained page invalid after an unrecoverable residency or
    /// visibility failure.
    pub fn invalidate_visibility(&self) -> Result<(), VisibilityError> {
        let stores = self.execution_stores();
        let mut transitions = stores
            .iter()
            .map(|store| store.execution_gate().acquire_exclusive())
            .collect::<Vec<_>>();
        for transition in &mut transitions {
            transition.commit();
        }
        let mut visited = BTreeSet::new();
        for segment in self.segments.iter() {
            if visited.insert(segment.page()) {
                segment.backing.invalidate_visibility()?;
            }
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
}

impl Display for CanonicalRangeAccessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RangeOverflow => formatter.write_str("canonical range access overflows"),
            Self::InvalidAlignment(alignment) => write!(
                formatter,
                "canonical snapshot alignment is not a nonzero power of two: {alignment}"
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

        assert!(Arc::ptr_eq(&range.segments, &cloned.segments));
        assert_eq!(range, cloned);
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
    fn cpu_write_dependency_becomes_volatile_after_five_dirty_rearms() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();

        for cycle in 1..=5 {
            allocation.write(0, &[cycle]).unwrap();
            assert!(!dependency.remains_current());
            assert_eq!(dependency.rearm(), cycle < 5);
        }
        assert!(dependency.is_volatile());
        assert!(!dependency.remains_current());
        assert!(!dependency.rearm());
    }

    #[test]
    fn clean_dependency_check_resets_consecutive_dirty_rearms() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();

        for cycle in 0..8 {
            allocation.write(0, &[cycle]).unwrap();
            assert!(!dependency.remains_current());
            assert!(dependency.rearm());
            assert!(dependency.remains_current());
        }
        assert!(!dependency.is_volatile());
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
}
