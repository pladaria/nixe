//! Checked, retained and pointer-free canonical backing ranges.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    AddressSpaceId, CanonicalBackingPage, CanonicalCpuWriteOverlap, CanonicalPageError,
    CanonicalPageId, ContentGeneration, CpuWriteEpoch, DeviceAccessDeclaration,
    GuestVirtualAddress, MappingGeneration, MemoryPermissions, VisibilityCoordinator,
    VisibilityError, VisibilityState,
};

/// One contiguous segment of a translated canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBackingSegment {
    backing: CanonicalBackingPage,
    offset: u64,
    size: u64,
    permissions: MemoryPermissions,
    content_generation: ContentGeneration,
    cpu_write_epoch: CpuWriteEpoch,
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
        content_generation: ContentGeneration,
        mapping_generation: MappingGeneration,
    ) -> Result<Self, CanonicalRangeError> {
        let cpu_write_epoch = backing.cpu_write_epoch();
        if content_generation != backing.content_generation() {
            return Err(CanonicalRangeError::StaleContentGeneration);
        }
        Self::new_captured(
            backing,
            offset,
            size,
            permissions,
            content_generation,
            cpu_write_epoch,
            mapping_generation,
        )
    }

    fn snapshot(
        backing: CanonicalBackingPage,
        offset: u64,
        size: u64,
        permissions: MemoryPermissions,
        mapping_generation: MappingGeneration,
    ) -> Result<Self, CanonicalRangeError> {
        // Capture the coarse epoch before the page generation. If a write
        // races this snapshot, an older epoch can only force the exact slow
        // path; it can never hide the write.
        let cpu_write_epoch = backing.cpu_write_epoch();
        let content_generation = backing.content_generation();
        Self::new_captured(
            backing,
            offset,
            size,
            permissions,
            content_generation,
            cpu_write_epoch,
            mapping_generation,
        )
    }

    fn new_captured(
        backing: CanonicalBackingPage,
        offset: u64,
        size: u64,
        permissions: MemoryPermissions,
        content_generation: ContentGeneration,
        cpu_write_epoch: CpuWriteEpoch,
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
            content_generation,
            cpu_write_epoch,
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

    /// Returns the permissions of the CPU mapping used for translation.
    #[must_use]
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    /// Returns the content generation captured during translation.
    #[must_use]
    pub const fn content_generation(&self) -> ContentGeneration {
        self.content_generation
    }

    /// Returns the store CPU-write epoch captured before generation validation.
    #[must_use]
    pub const fn captured_cpu_write_epoch(&self) -> CpuWriteEpoch {
        self.cpu_write_epoch
    }

    /// Returns the mapping generation captured during translation.
    #[must_use]
    pub const fn mapping_generation(&self) -> MappingGeneration {
        self.mapping_generation
    }

    /// Returns whether the retained bytes have changed since translation.
    #[must_use]
    pub fn content_is_current(&self) -> bool {
        self.backing.content_generation() == self.content_generation
    }

    /// Returns the conservative visibility authority shared by all aliases.
    #[must_use]
    pub fn visibility_state(&self) -> VisibilityState {
        self.backing.visibility_state()
    }

    /// Reports CPU-side writes overlapping this segment since `generation`.
    pub fn cpu_write_overlap_since(
        &self,
        generation: ContentGeneration,
    ) -> Result<CanonicalCpuWriteOverlap, CanonicalPageError> {
        self.backing
            .cpu_write_overlap_since(generation, self.offset, self.size)
    }
}

/// A validated logical byte range represented by retained page segments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBackingRange {
    segments: Arc<[CanonicalBackingSegment]>,
    size: u64,
}

struct CpuWriteDependencyStore {
    store: crate::CanonicalBackingStore,
    observed_epoch: AtomicU64,
    intervals: Box<[(CanonicalPageId, u64, u64)]>,
    pages: Box<[(CanonicalBackingPage, ContentGeneration)]>,
}

struct CanonicalCpuWriteDependencyInner {
    stores: Box<[CpuWriteDependencyStore]>,
}

/// Cloneable observation of CPU writes relevant to fixed canonical intervals.
///
/// Clones share only monotonic validation metadata. Canonical page generations
/// remain authoritative; this dependency is an exact fast rejection filter
/// backed by the store's bounded CPU-write provenance journal.
#[derive(Clone)]
pub struct CanonicalCpuWriteDependency {
    inner: Arc<CanonicalCpuWriteDependencyInner>,
}

impl CanonicalCpuWriteDependency {
    /// Captures one range if every segment from a store observed one coherent
    /// CPU-write epoch while its page generations were snapshotted.
    #[must_use]
    pub fn capture(range: &CanonicalBackingRange) -> Option<Self> {
        Self::capture_ranges([range])
    }

    /// Captures several ranges as one dependency domain.
    #[must_use]
    pub fn capture_ranges<'a>(
        ranges: impl IntoIterator<Item = &'a CanonicalBackingRange>,
    ) -> Option<Self> {
        struct StoreBuilder {
            store: crate::CanonicalBackingStore,
            observed_epoch: CpuWriteEpoch,
            intervals: Vec<(CanonicalPageId, u64, u64)>,
            pages: Vec<(CanonicalBackingPage, ContentGeneration)>,
        }

        let mut stores = BTreeMap::<crate::BackingStoreId, StoreBuilder>::new();
        for range in ranges {
            for segment in range.segments() {
                let store_id = segment.page().store();
                let observed_epoch = segment.captured_cpu_write_epoch();
                let end = segment.offset().checked_add(segment.size())?;
                if let Some(current) = stores.get_mut(&store_id) {
                    if current.observed_epoch != observed_epoch {
                        return None;
                    }
                    current
                        .intervals
                        .push((segment.page(), segment.offset(), end));
                    current
                        .pages
                        .push((segment.backing().clone(), segment.content_generation()));
                } else {
                    stores.insert(
                        store_id,
                        StoreBuilder {
                            store: segment.backing.store().clone(),
                            observed_epoch,
                            intervals: vec![(segment.page(), segment.offset(), end)],
                            pages: vec![(segment.backing().clone(), segment.content_generation())],
                        },
                    );
                }
            }
        }
        if stores.is_empty() {
            return None;
        }
        let mut dependencies = Vec::new();
        dependencies.try_reserve_exact(stores.len()).ok()?;
        for mut store in stores.into_values() {
            normalize_cpu_write_intervals(&mut store.intervals);
            store
                .pages
                .sort_unstable_by_key(|(page, _)| page.identity());
            if store
                .pages
                .windows(2)
                .any(|pair| pair[0].0.identity() == pair[1].0.identity() && pair[0].1 != pair[1].1)
            {
                return None;
            }
            store.pages.dedup_by_key(|(page, _)| page.identity());
            dependencies.push(CpuWriteDependencyStore {
                store: store.store,
                observed_epoch: AtomicU64::new(store.observed_epoch.get()),
                intervals: store.intervals.into_boxed_slice(),
                pages: store.pages.into_boxed_slice(),
            });
        }
        let stores = dependencies.into_boxed_slice();
        Some(Self {
            inner: Arc::new(CanonicalCpuWriteDependencyInner { stores }),
        })
    }

    /// Returns whether no CPU write since the last successful observation
    /// overlaps any captured canonical byte. If the bounded store journal no
    /// longer covers the observation, authoritative page provenance is checked
    /// with logarithmic page lookup. Exhausting both levels invalidates
    /// conservatively.
    #[must_use]
    pub fn remains_current(&self) -> bool {
        for dependency in &self.inner.stores {
            let observed = CpuWriteEpoch::new(dependency.observed_epoch.load(Ordering::Acquire));
            if dependency.store.cpu_write_epoch() == observed {
                continue;
            }
            let (current, overlap) = dependency
                .store
                .cpu_write_overlap_since(observed, &dependency.intervals);
            match overlap {
                CanonicalCpuWriteOverlap::Yes => return false,
                CanonicalCpuWriteOverlap::Unknown
                    if page_local_cpu_write_overlap(dependency) != CanonicalCpuWriteOverlap::No =>
                {
                    return false;
                }
                CanonicalCpuWriteOverlap::No | CanonicalCpuWriteOverlap::Unknown => {}
            }
            dependency
                .observed_epoch
                .fetch_max(current.get(), Ordering::AcqRel);
        }
        true
    }
}

fn page_local_cpu_write_overlap(dependency: &CpuWriteDependencyStore) -> CanonicalCpuWriteOverlap {
    for (page_id, start, end) in &dependency.intervals {
        let Ok(index) = dependency
            .pages
            .binary_search_by_key(page_id, |(page, _)| page.identity())
        else {
            return CanonicalCpuWriteOverlap::Unknown;
        };
        let (page, generation) = &dependency.pages[index];
        match page.cpu_write_overlap_since(*generation, *start, end - start) {
            Ok(CanonicalCpuWriteOverlap::No) => {}
            Ok(overlap) => return overlap,
            Err(_) => return CanonicalCpuWriteOverlap::Unknown,
        }
    }
    CanonicalCpuWriteOverlap::No
}

fn normalize_cpu_write_intervals(intervals: &mut Vec<(CanonicalPageId, u64, u64)>) {
    intervals.sort_unstable();
    let mut output = 0_usize;
    for input in 0..intervals.len() {
        let current = intervals[input];
        if output != 0 {
            let previous = &mut intervals[output - 1];
            if previous.0 == current.0 && current.1 <= previous.2 {
                previous.2 = previous.2.max(current.2);
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
        let stores = self
            .inner
            .stores
            .iter()
            .map(|store| {
                (
                    store.store.identity(),
                    CpuWriteEpoch::new(store.observed_epoch.load(Ordering::Acquire)),
                    store.intervals.len(),
                    store.pages.len(),
                )
            })
            .collect::<Vec<_>>();
        formatter
            .debug_struct("CanonicalCpuWriteDependency")
            .field("stores", &stores)
            .finish()
    }
}

impl PartialEq for CanonicalCpuWriteDependency {
    fn eq(&self, other: &Self) -> bool {
        self.inner.stores.len() == other.inner.stores.len()
            && self
                .inner
                .stores
                .iter()
                .zip(&other.inner.stores)
                .all(|(left, right)| {
                    left.store.identity() == right.store.identity()
                        && left.intervals == right.intervals
                })
    }
}

impl Eq for CanonicalCpuWriteDependency {}

impl CanonicalBackingRange {
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

    /// Captures a checked logical subrange with current content generations.
    ///
    /// The returned range retains the same canonical pages and permissions,
    /// but snapshots each touched page's content generation at this boundary.
    /// This is useful when a long-lived mapping is interpreted as a resource:
    /// content writes do not invalidate the mapping, while the derived view
    /// still needs an exact version key for later invalidation.
    pub fn snapshot_subrange(&self, offset: u64, size: u64) -> Result<Self, CanonicalRangeError> {
        let mut captured = Vec::new();
        self.snapshot_subrange_into(offset, size, &mut captured)?;
        Self::new(captured)
    }

    /// Appends a versioned subrange directly to an existing segment builder.
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
                    .read(page_offset, &mut output[copied..copied_end])
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

    /// Publishes newer device contents after the declaration's write point.
    ///
    /// Callers must invoke this only after the host operations which make that
    /// point true. It records logical ownership; it does not signal a guest
    /// fence or claim host queue completion.
    pub fn complete_device_write(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        if !declaration.kind().writes() {
            return Err(VisibilityError::DeclarationDoesNotWrite);
        }
        let mut visited = BTreeSet::new();
        for segment in self.segments.iter() {
            if visited.insert(segment.page()) {
                segment
                    .backing
                    .complete_device_write(declaration, Arc::clone(&coordinator))?;
            }
        }
        Ok(())
    }

    /// Marks every retained page invalid after an unrecoverable residency or
    /// visibility failure.
    pub fn invalidate_visibility(&self) -> Result<(), VisibilityError> {
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
    StaleContentGeneration,
    RangeOverflow,
}

impl Display for CanonicalRangeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "canonical backing range is empty",
            Self::InvalidSubrange => "canonical backing subrange is empty or out of bounds",
            Self::SegmentOverflow => "canonical segment end overflows",
            Self::InvalidSegmentBounds => "canonical segment is outside its retained page",
            Self::StaleContentGeneration => {
                "canonical segment generation changed during construction"
            }
            Self::RangeOverflow => "canonical backing range length overflows",
        })
    }
}

impl std::error::Error for CanonicalRangeError {}

/// Failure while accessing a retained canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalRangeAccessError {
    RangeOverflow,
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
        CanonicalAllocation, CanonicalBackingStore, CanonicalWriteBatch, GuestPhysicalPageId,
    };

    #[test]
    fn segment_construction_checks_bounds_and_generation() {
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
                ContentGeneration::INITIAL,
                MappingGeneration::new(1),
            ),
            Err(CanonicalRangeError::InvalidSegmentBounds)
        );
        assert_eq!(
            CanonicalBackingSegment::new(
                page,
                0,
                1,
                MemoryPermissions::READ,
                ContentGeneration::new(1),
                MappingGeneration::new(1),
            ),
            Err(CanonicalRangeError::StaleContentGeneration)
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
                ContentGeneration::INITIAL,
                MappingGeneration::new(1),
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second,
                0,
                3,
                MemoryPermissions::READ,
                ContentGeneration::INITIAL,
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
    fn snapshot_subrange_retains_exact_bytes_and_current_content_generations() {
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
                ContentGeneration::INITIAL,
                MappingGeneration::new(1),
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second.clone(),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                ContentGeneration::INITIAL,
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
        assert_eq!(
            snapshot.segments()[0].content_generation(),
            first.content_generation()
        );
        assert_eq!(
            snapshot.segments()[1].content_generation(),
            second.content_generation()
        );
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
                ContentGeneration::INITIAL,
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
    fn cpu_write_dependency_advances_only_past_disjoint_writes() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observed = mapped.snapshot_subrange(0x100, 0x100).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&observed).unwrap();

        assert!(dependency.remains_current());
        allocation.write(0x300, &[1]).unwrap();
        assert!(dependency.remains_current());
        allocation.write(0x1100, &[2]).unwrap();
        assert!(dependency.remains_current());
        allocation.write(0x180, &[3]).unwrap();
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
        assert!(dependency.remains_current());
        allocation.write(0x380, &[2]).unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn cpu_write_dependency_observes_only_committed_batch_ranges() {
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
        assert!(dependency.remains_current());

        let mut overlapping = CanonicalWriteBatch::new();
        overlapping.stage(&mapped, 0x180, &[2]).unwrap();
        assert!(dependency.remains_current());
        overlapping.commit().unwrap();
        assert!(!dependency.remains_current());
    }

    #[test]
    fn cpu_write_dependency_uses_page_local_provenance_after_lost_store_history() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observed = mapped.snapshot_subrange(0, 0x1000).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&observed).unwrap();

        for value in 0..=super::super::backing::STORE_CPU_WRITE_JOURNAL_CAPACITY {
            allocation.write(0x1000, &[value as u8]).unwrap();
        }

        assert!(dependency.remains_current());
    }

    #[test]
    fn cpu_write_dependency_rejects_lost_overlapping_history() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let mapped = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let observed = mapped.snapshot_subrange(0, 0x1000).unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&observed).unwrap();

        for value in 0..=super::super::backing::STORE_CPU_WRITE_JOURNAL_CAPACITY {
            allocation.write(0, &[value as u8]).unwrap();
        }

        assert!(!dependency.remains_current());
    }
}
